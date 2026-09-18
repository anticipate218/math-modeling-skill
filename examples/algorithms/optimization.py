"""线性规划、整数规划、指派与运输问题。

这个模块提供竞赛里真正会被用到的那几件事：

- ``simplex_lp``            通用线性规划（两阶段单纯形，支持等式/不等式/变量上下界）
- ``branch_and_bound_ilp``  小规模整数/0-1 规划（在 LP 松弛上分支定界）
- ``knapsack_dp``           0-1 背包的动态规划精确解
- ``assignment_hungarian``  指派问题（Kuhn-Munkres / 匈牙利算法，O(n^3)）
- ``transportation_vogel``  运输问题的 Vogel 近似法 + 位势法检验

生产环境建议直接用成熟求解器（OR-Tools 的 CP-SAT、Pyomo + HiGHS 等，见
``references/github-resources.md``）；本模块的价值在于让你看清单纯形表在做什么，
以及在论文里能写出"我们自行实现单纯形法并与求解器结果对照，误差 < 1e-9"。
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._common import as_matrix, as_vector

__all__ = [
    "simplex_lp",
    "branch_and_bound_ilp",
    "knapsack_dp",
    "assignment_hungarian",
    "transportation_vogel",
]

_EPS = 1e-9
_INF = float("inf")


# --------------------------------------------------------------------------
# 单纯形法
# --------------------------------------------------------------------------

def _pivot(tab: np.ndarray, basis: List[int], row: int, col: int, m: int) -> None:
    """对单纯形表做一次转轴（消元），把 col 列化为单位向量。"""
    piv = tab[row, col]
    if abs(piv) < 1e-12:
        raise ZeroDivisionError("转轴元素接近 0，数值上不可继续")
    tab[row, :] /= piv
    for i in range(m + 1):
        if i != row:
            factor = tab[i, col]
            if factor != 0.0:
                tab[i, :] -= factor * tab[row, :]
    basis[row] = col


def _simplex_iterate(tab: np.ndarray, basis: List[int], m: int, ncols: int,
                     forbidden: Sequence[int], max_iter: int) -> str:
    """在已消元好的表上迭代至最优。

    采用 Bland 规则（最小下标入基 + 最小基变量下标出基），保证有限步终止、不会循环。

    返回 ``"optimal"`` / ``"unbounded"`` / ``"max_iter"``。
    """
    banned = set(forbidden)
    for _ in range(max_iter):
        obj = tab[m, :ncols]
        entering = -1
        for j in range(ncols):
            if j not in banned and obj[j] < -1e-9:
                entering = j
                break
        if entering == -1:
            return "optimal"

        # 最小比值检验；比值相同时取基变量下标最小者（Bland）
        leaving = -1
        best_ratio = _INF
        best_basis = None
        for i in range(m):
            a_ij = tab[i, entering]
            if a_ij > 1e-11:
                ratio = tab[i, ncols] / a_ij
                if (ratio < best_ratio - 1e-12 or
                        (abs(ratio - best_ratio) <= 1e-12 and
                         (best_basis is None or basis[i] < best_basis))):
                    best_ratio = ratio
                    leaving = i
                    best_basis = basis[i]
        if leaving == -1:
            return "unbounded"
        _pivot(tab, basis, leaving, entering, m)
    return "max_iter"


def simplex_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
               bounds=None, maximize=False, max_iter=500) -> Dict[str, object]:
    """求解线性规划（两阶段单纯形法）。

    参数:
        c: 长度 n 的目标系数。
        A_ub: 不等式约束矩阵，``A_ub @ x <= b_ub``；None 表示无不等式约束。
        b_ub: 不等式右端项。
        A_eq: 等式约束矩阵，``A_eq @ x == b_eq``；None 表示无等式约束。
        b_eq: 等式右端项。
        bounds: 长度 n 的 ``(lo, hi)`` 列表；``lo=None`` 表示 −∞，``hi=None`` 表示 +∞。
            ``bounds=None`` 等价于所有变量 ``x >= 0``。
        maximize: True 时求最大值（内部取负）。
        max_iter: 迭代上限，防止数值病态时死循环。

    返回:
        dict，键为 ``status``（``"optimal"``/``"infeasible"``/``"unbounded"``/``"max_iter"``）、
        ``x``（最优解，np.ndarray；无解时为 None）、``fun``（最优目标值；无解时为 None）。

    算法:
        1. 变量变换：``x = shift + T @ z``，把所有变量化为 ``z >= 0``；自由变量拆成正负两部分，
           有限上界转成额外约束行。
        2. 引入松弛/剩余变量，负右端项整行取反，再为每个等式引入人工变量。
        3. **第一阶段**最小化人工变量之和；和不为 0 则原问题不可行。
        4. **第二阶段**以原目标函数继续迭代，人工变量禁止再入基。

    复杂度:
        时间最坏 O(2^n)（LP 是多项式可解的，单纯形是实用指数级但实际很快）；
        空间 O((m+1) x (n+m+人工变量))。

    陷阱:
        - **数值尺度**：系数相差 1e6 倍以上时，``1e-9`` 的判定阈值会让结果不可信。
          竞赛里遇到量纲差异大的约束，应先做无量纲化再求解。
        - **退化与循环**：理论上单纯形可能死循环，这里用 Bland 规则规避（代价是收敛变慢）。
        - 这个实现**不报告对偶变量**。需要影子价格/灵敏度报告时请用成熟求解器。
        - ``status == "unbounded"`` 往往说明你的模型漏了约束，而不是真的有无界最优解。

    参考:
        Dantzig 单纯形法；Bland (1977) 反循环规则；Chvátal《Linear Programming》。
    """
    c_vec = as_vector(c, "c")
    n = c_vec.size

    if bounds is None:
        bounds = [(0.0, None)] * n
    if len(bounds) != n:
        raise ValueError(f"bounds 长度 {len(bounds)} 与变量数 {n} 不一致")

    # --- 1) 变量变换 x = shift + T z --------------------------------
    shift = np.zeros(n)
    cols: List[np.ndarray] = []          # 每个新变量在原空间的方向向量
    # 有限上界导出的约束，先记录 (新变量下标, 右端项)，等列数确定后再拼成整行。
    # 注意：这里**不能**当场构造行向量——后续还会追加新列，短行会被 numpy 静默广播，
    # 把 "x_j <= hi" 悄悄变成 "sum of all later t <= hi"（已踩过这个坑）。
    pending_bounds: List[Tuple[int, float]] = []
    for j, (lo, hi) in enumerate(bounds):
        has_lo = lo is not None
        has_hi = hi is not None
        if not has_lo and not has_hi:
            col_p = np.zeros(n); col_p[j] = 1.0
            col_m = np.zeros(n); col_m[j] = -1.0
            cols.extend([col_p, col_m])
        elif not has_lo:                 # x <= hi，令 x = hi - t
            shift[j] = float(hi)
            col = np.zeros(n); col[j] = -1.0
            cols.append(col)
        else:                            # x = lo + t
            shift[j] = float(lo)
            col = np.zeros(n); col[j] = 1.0
            cols.append(col)
            if has_hi:
                pending_bounds.append((len(cols) - 1, float(hi) - float(lo)))

    T = np.array(cols, dtype=float).T if cols else np.zeros((n, 0))
    mz = T.shape[1]

    bound_rows: List[Tuple[np.ndarray, float]] = []
    for col_idx, rhs_val in pending_bounds:
        if rhs_val < -_EPS:
            raise ValueError("bounds 中 hi < lo，问题本身不可行")
        row = np.zeros(mz)
        row[col_idx] = 1.0
        bound_rows.append((row, rhs_val))

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    kinds: List[str] = []                # "<=" 或 "="

    if A_ub is not None and len(np.asarray(A_ub)) > 0:
        A_ub_m = as_matrix(A_ub, "A_ub")
        b_ub_v = as_vector(b_ub, "b_ub")
        if A_ub_m.shape[1] != n or A_ub_m.shape[0] != b_ub_v.size:
            raise ValueError("A_ub 形状与 b_ub / c 不匹配")
        for i in range(A_ub_m.shape[0]):
            rows.append(A_ub_m[i] @ T)
            rhs.append(b_ub_v[i] - A_ub_m[i] @ shift)
            kinds.append("<=")

    if A_eq is not None and len(np.asarray(A_eq)) > 0:
        A_eq_m = as_matrix(A_eq, "A_eq")
        b_eq_v = as_vector(b_eq, "b_eq")
        if A_eq_m.shape[1] != n or A_eq_m.shape[0] != b_eq_v.size:
            raise ValueError("A_eq 形状与 b_eq / c 不匹配")
        for i in range(A_eq_m.shape[0]):
            rows.append(A_eq_m[i] @ T)
            rhs.append(b_eq_v[i] - A_eq_m[i] @ shift)
            kinds.append("=")

    for row, r in bound_rows:
        rows.append(row)
        rhs.append(r)
        kinds.append("<=")

    # 先把所有 z 的约束写成 A z <= b 或 A z = b（无约束时也不能直接返回，要看是否有可行 z）
    n_extra = 0
    for k in kinds:
        if k == "<=":
            n_extra += 1      # 松弛
    n_art = 0
    for i, k in enumerate(kinds):
        b = rhs[i]
        if k == "=" or (k == "<=" and b < -_EPS):
            n_art += 1
    for row, r in bound_rows:
        if r < -_EPS:
            n_art += 1

    total_cols = mz + n_extra + n_art
    m_rows = len(rows)
    if m_rows == 0:
        # 无约束：直接判断有界性
        c_z = T.T @ c_vec
        if np.any(np.abs(c_z) > 1e-12):
            direction = "max" if maximize else "min"
            obj_sign = -1.0 if maximize else 1.0
            if np.any(obj_sign * c_z < -1e-12):
                return {"status": "unbounded", "x": None, "fun": None}
            del direction
        z = np.zeros(mz)
        x = shift + T @ z
        val = float(c_vec @ x)
        return {"status": "optimal", "x": x, "fun": val}

    tab = np.zeros((m_rows + 1, total_cols + 1))
    basis = [-1] * m_rows
    slack_ptr = mz
    art_ptr = mz + n_extra
    artificials: List[int] = []

    for i in range(m_rows):
        row = rows[i].copy()
        r = rhs[i]
        kind = kinds[i]
        if kind == "<=" and r >= -_EPS:
            tab[i, :mz] = row
            tab[i, slack_ptr] = 1.0
            basis[i] = slack_ptr
            slack_ptr += 1
        elif kind == "<=" and r < -_EPS:
            tab[i, :mz] = -row
            tab[i, slack_ptr] = -1.0
            slack_ptr += 1
            tab[i, art_ptr] = 1.0
            basis[i] = art_ptr
            artificials.append(art_ptr)
            art_ptr += 1
            r = -r
        else:  # "="
            tab[i, :mz] = row if r >= -_EPS else -row
            r = abs(r)
            tab[i, art_ptr] = 1.0
            basis[i] = art_ptr
            artificials.append(art_ptr)
            art_ptr += 1
        tab[i, total_cols] = r

    # --- 第一阶段 ---------------------------------------------------
    phase1_cost = np.zeros(total_cols)
    for a in artificials:
        phase1_cost[a] = 1.0
    tab[m_rows, :total_cols] = phase1_cost
    tab[m_rows, total_cols] = 0.0
    for i in range(m_rows):
        if basis[i] in artificials:
            tab[m_rows, :] -= tab[i, :]

    status = _simplex_iterate(tab, basis, m_rows, total_cols, [], max_iter)
    if status == "max_iter":
        return {"status": "max_iter", "x": None, "fun": None}
    phase1_value = -tab[m_rows, total_cols]
    if phase1_value > 1e-7:
        return {"status": "infeasible", "x": None, "fun": None}

    # 把仍在基里的人工变量赶出去（若整行在真实列上全 0，说明该约束冗余，直接置 0 行）
    for i in range(m_rows):
        if basis[i] in artificials:
            pivot_col = -1
            for j in range(mz + n_extra):
                if j not in artificials and abs(tab[i, j]) > 1e-9:
                    pivot_col = j
                    break
            if pivot_col >= 0:
                _pivot(tab, basis, i, pivot_col, m_rows)
            else:
                tab[i, :] = 0.0       # 冗余约束，保留 0=0 不影响后续

    # --- 第二阶段 ---------------------------------------------------
    obj_sign = -1.0 if maximize else 1.0
    c_z = T.T @ c_vec
    tab[m_rows, :] = 0.0
    for j in range(mz):
        tab[m_rows, j] = obj_sign * c_z[j]
    tab[m_rows, total_cols] = 0.0
    for i in range(m_rows):
        b = basis[i]
        if tab[i, b] != 0.0 and tab[m_rows, b] != 0.0:
            tab[m_rows, :] -= tab[m_rows, b] * tab[i, :]

    status = _simplex_iterate(tab, basis, m_rows, total_cols, artificials, max_iter)
    if status != "optimal":
        return {"status": status, "x": None, "fun": None}

    z = np.zeros(mz)
    for i in range(m_rows):
        if 0 <= basis[i] < mz:
            z[basis[i]] = tab[i, total_cols]
    x = shift + T @ z
    val = float(c_vec @ x)
    return {"status": "optimal", "x": x, "fun": val}


# --------------------------------------------------------------------------
# 整数规划
# --------------------------------------------------------------------------

def branch_and_bound_ilp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                         bounds=None, maximize=False, integer=None,
                         max_nodes=2000) -> Dict[str, object]:
    """小规模整数/混合整数线性规划（LP 松弛 + 分支定界）。

    参数:
        c, A_ub, b_ub, A_eq, b_eq, bounds, maximize: 同 :func:`simplex_lp`。
        integer: 长度 n 的 bool 序列，标记哪些变量必须是整数；None 表示全部整数。
        max_nodes: 搜索节点上限。

    返回:
        dict，键为 ``status``（``"optimal"``/``"infeasible"``/``"node_limit"``）、
        ``x``、``fun``、``nodes``（实际探索节点数）、``gap``（最好整数解与 LP 下界的相对间隙）。

    算法:
        对 LP 松弛求解 → 若解已全整数则更新上界 → 否则挑一个小数变量 x_j 分两支
        ``x_j <= floor`` 与 ``x_j >= ceil`` 递归；用当前最好整数解剪枝。

    复杂度:
        时间 O(2^n) 最坏；空间 O(n)（递归深度）。

    陷阱:
        - 分支定界**不是**多项式算法：变量数超过 ~50 个时请换 OR-Tools CP-SAT。
        - 目标值接近时 ``gap`` 可能因为浮点误差显示成负数，取绝对值理解即可。
        - 0-1 背包请直接用 :func:`knapsack_dp`，伪多项式 O(nC) 比分支定界快得多。

    参考:
        Land & Doig (1960) 分支定界；Dakin (1965) 整数规划分支法。
    """
    c_vec = as_vector(c, "c")
    n = c_vec.size
    if integer is None:
        integer = [True] * n
    if len(integer) != n:
        raise ValueError("integer 长度必须等于变量个数")

    if bounds is None:
        bounds = [(0.0, None)] * n
    bounds = [tuple(b) for b in bounds]

    best_x: Optional[np.ndarray] = None
    best_val = -_INF if maximize else _INF
    nodes = 0

    def lp_lower_bound(node_bounds):
        res = simplex_lp(c_vec, A_ub, b_ub, A_eq, b_eq, node_bounds,
                         maximize=maximize, max_iter=300)
        return res

    stack = [bounds]
    while stack:
        if nodes >= max_nodes:
            gap = None
            if best_x is not None:
                lb = lp_lower_bound(bounds)
                if lb["status"] == "optimal":
                    denom = max(abs(float(lb["fun"])), 1e-12)
                    gap = abs(float(lb["fun"]) - best_val) / denom
            return {"status": "node_limit", "x": best_x, "fun": best_val if best_x is not None else None,
                    "nodes": nodes, "gap": gap}
        node_bounds = stack.pop()
        nodes += 1
        res = lp_lower_bound(node_bounds)
        if res["status"] != "optimal":
            continue
        val = float(res["fun"])
        if best_x is not None:
            if maximize and val <= best_val + 1e-9:
                continue
            if not maximize and val >= best_val - 1e-9:
                continue

        x = np.asarray(res["x"], dtype=float)
        frac_idx = -1
        frac_val = 0.0
        for j in range(n):
            if integer[j]:
                d = abs(x[j] - round(x[j]))
                if d > 1e-6:
                    frac_idx = j
                    frac_val = x[j]
                    break

        if frac_idx == -1:
            if best_x is None or (maximize and val > best_val + 1e-9) or \
                    (not maximize and val < best_val - 1e-9):
                best_x = x.copy()
                best_val = val
            continue

        lo, hi = node_bounds[frac_idx]
        down = list(node_bounds)
        up = list(node_bounds)
        down[frac_idx] = (lo, math.floor(frac_val))
        up[frac_idx] = (math.ceil(frac_val), hi)

        for child in (up, down):
            clo, chi = child[frac_idx]
            if clo is not None and chi is not None and clo > chi + 1e-12:
                continue
            stack.append(child)

    if best_x is None:
        return {"status": "infeasible", "x": None, "fun": None, "nodes": nodes, "gap": None}
    return {"status": "optimal", "x": best_x, "fun": float(best_val), "nodes": nodes, "gap": 0.0}


def knapsack_dp(weights, values, capacity) -> Dict[str, object]:
    """0-1 背包问题的动态规划精确解。

    参数:
        weights: 每个物品的重量（非负整数）。
        values: 每个物品的价值。
        capacity: 背包容量（非负整数）。

    返回:
        dict，键为 ``max_value``、``chosen``（被选物品下标列表）、``total_weight``。

    算法:
        ``dp[c] = 前 i 个物品、容量 c 的最大价值``；倒序枚举容量保证每件物品只用一次，
        再回溯出选择方案。

    复杂度:
        时间 O(n * capacity) / 空间 O(capacity)。

    陷阱:
        - 这是**伪多项式**算法：容量取 1e9 时直接爆内存。容量极大但物品种类少时改用
          分支定界或对偶近似。
        - 重量必须是整数。有小数的重量要先做缩放并说明精度损失。

    参考:
        Bellman (1957) 动态规划；Dantzig 上界。
    """
    w = [int(x) for x in np.asarray(weights).ravel()]
    v = [float(x) for x in np.asarray(values).ravel()]
    cap = int(capacity)
    if len(w) != len(v):
        raise ValueError("weights 与 values 长度不一致")
    if any(x < 0 for x in w) or cap < 0:
        raise ValueError("重量与容量必须非负")
    if any(abs(float(x) - int(x)) > 1e-12 for x in np.asarray(weights).ravel()):
        raise ValueError("knapsack_dp 要求整数重量，请先缩放")

    n = len(w)
    dp = np.zeros(cap + 1)
    take = np.zeros((n, cap + 1), dtype=bool)
    for i in range(n):
        for cap_i in range(cap, w[i] - 1, -1):
            cand = dp[cap_i - w[i]] + v[i]
            if cand > dp[cap_i] + 1e-12:
                dp[cap_i] = cand
                take[i, cap_i] = True

    chosen: List[int] = []
    cap_i = cap
    for i in range(n - 1, -1, -1):
        if take[i, cap_i]:
            chosen.append(i)
            cap_i -= w[i]
    chosen.reverse()
    return {"max_value": float(dp[cap]), "chosen": chosen,
            "total_weight": int(sum(w[i] for i in chosen))}


# --------------------------------------------------------------------------
# 指派问题
# --------------------------------------------------------------------------

def assignment_hungarian(cost) -> Dict[str, object]:
    """指派问题（匈牙利 / Kuhn-Munkres 算法，O(n^3)）。

    参数:
        cost: 代价矩阵，形状 (n, m)。求"每个工人做一件事、总代价最小"的完全匹配。

    返回:
        dict，键为 ``total_cost``、``assignment``（长度 n 的列表，第 i 个工人的任务下标，
        未指派时为 -1）、``transposed``（当 n > m 时内部转了置，为 True）。

    算法:
        势函数 u/v + 增广路：对每个工人依次寻找一条使总代价下降的交替路，用 ``minv``
        维护到未访问列的最小松弛量，沿 ``way`` 回溯更新匹配。

    复杂度:
        时间 O(n^2 m)（n <= m 时即 O(n^3)）/ 空间 O(nm)。

    陷阱:
        - 经典匈牙利算法要求 **n <= m**（工人数不超过任务数）。n > m 时本实现自动转置，
          返回的 ``assignment`` 已经还原成"每个工人一个任务"的语义，但此时必然有工人闲置。
        - 负代价（收益矩阵取负）可以用，但**不能**先给矩阵加常数再求解——那会改变最优解。
          要把最大化收益转成最小化代价，请整体取负。
        - 不要求方阵；但如果你想要"每个任务也必须有人做"，那需要 n == m。

    参考:
        Kuhn (1955) 匈牙利算法；Munkres (1957) 改进；e-maxx 增广路实现范式。
    """
    a = as_matrix(cost, "cost")
    n0, m0 = a.shape
    transposed = False
    if n0 > m0:
        a = a.T
        transposed = True
    n, m = a.shape

    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)          # p[j] = 匹配到列 j 的行
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j] - 1e-15:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    row_to_col = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            row_to_col[p[j] - 1] = j - 1
    total = float(sum(a[i, row_to_col[i]] for i in range(n) if row_to_col[i] >= 0))

    if transposed:
        col_to_row = [-1] * m
        for i, j in enumerate(row_to_col):
            if j >= 0:
                col_to_row[j] = i
        return {"total_cost": total, "assignment": col_to_row, "transposed": True}
    return {"total_cost": total, "assignment": row_to_col, "transposed": False}


# --------------------------------------------------------------------------
# 运输问题
# --------------------------------------------------------------------------

def transportation_vogel(cost, supply, demand) -> Dict[str, object]:
    """运输问题：Vogel 近似法求初始可行解 + 位势法（MODI）迭代到最优。

    参数:
        cost: 单位运价矩阵，形状 (m 个产地, n 个销地)。
        supply: 各产地供应量，长度 m。
        demand: 各销地需求量，长度 n。

    返回:
        dict，键为:

        - ``total_cost`` / ``plan``：**精确最优**总运费与运输方案（由 :func:`simplex_lp` 求解等价 LP 得到）
        - ``vogel_cost`` / ``vogel_plan``：Vogel 近似法的初始方案，用于展示"启发式离最优有多远"
        - ``improved``：精确解是否严格优于 Vogel 初始解
        - ``iterations`` / ``degenerate``：Vogel 初始解里基格数量是否少于 ``m+n-1``（退化征兆）
        - ``balanced``：是否产销平衡；不平时会补一个运价为 0 的虚拟产地/销地
        - ``lp_status``：精确 LP 的求解状态

    算法:
        1. 若产销不平衡，补一个虚拟产地/销地（运价 0）使其平衡。
        2. **Vogel 近似法**：每次在行/列罚数（次小运价 − 最小运价）最大的那条线上，
           用最小运价格尽量多地运输，得到一个高质量初始可行解。
        3. **精确求解**：把运输问题写成 LP
           ``min sum c_ij x_ij, s.t. 行和 = 供应量, 列和 = 需求量, x >= 0``，
           用 :func:`simplex_lp` 求精确最优。

    复杂度:
        时间 Vogel 部分 O((m+n)^2 * (m+n))，精确 LP 部分为单纯形迭代开销；
        空间 O(mn)（LP 版本会展开成 m*n 个变量）。

    陷阱:
        - **退化**：基格少于 ``m+n-1`` 个时位势法（MODI）的解不唯一，也是手工计算最容易
          出错的地方。本实现因此**用 LP 保证最优性**，并把 Vogel 初始解单独报告出来；
          ``degenerate=True`` 只是提示"这条初始解退化，手工检验时要小心"。
        - 运价必须非负；负运价（补贴）会让"尽量多运"的直觉失效，也要求 Vogel 的罚数逻辑改写。
        - 平衡后新加的虚拟格运价为 0，直接照抄进论文会让人误以为真的免费运输，
          记得在表注里说明哪一行/哪一列是虚拟的。
        - 这里**没有**把结果强制取整：如果供应/需求是整数，最优解自然是整数（运输问题
          的约束矩阵是全幺模的），但浮点误差可能留下 1e-13 级别的残渣，本实现已做清理。

    参考:
        Vogel (1958) 近似法；Dantzig 位势法（MODI）；运输问题约束矩阵的全幺模性。
    """
    c = as_matrix(cost, "cost")
    s = np.asarray(supply, dtype=float).ravel()
    d = np.asarray(demand, dtype=float).ravel()
    m, n = c.shape
    if s.size != m or d.size != n:
        raise ValueError("supply/demand 长度与 cost 形状不匹配")
    if np.any(s < 0) or np.any(d < 0):
        raise ValueError("供应量与需求量必须非负")

    balanced = abs(s.sum() - d.sum()) < 1e-9
    c_w = c.copy()
    s_w = s.copy()
    d_w = d.copy()
    if not balanced:
        if s.sum() < d.sum():
            c_w = np.vstack([c_w, np.zeros((1, n))])
            s_w = np.append(s_w, d.sum() - s.sum())
            m += 1
        else:
            c_w = np.hstack([c_w, np.zeros((m, 1))])
            d_w = np.append(d_w, s.sum() - d.sum())
            n += 1

    plan = np.zeros((m, n))
    s_rem = s_w.copy()
    d_rem = d_w.copy()
    big = 1e12

    for _ in range(m * n + 1):
        active_rows = [i for i in range(m) if s_rem[i] > 1e-9]
        active_cols = [j for j in range(n) if d_rem[j] > 1e-9]
        if not active_rows or not active_cols:
            break
        best_pen, best_line = -1.0, None
        for i in active_rows:
            vals = sorted(c_w[i, j] for j in active_cols)
            pen = (vals[1] - vals[0]) if len(vals) > 1 else big
            if pen > best_pen:
                best_pen, best_line = pen, ("row", i)
        for j in active_cols:
            vals = sorted(c_w[i, j] for i in active_rows)
            pen = (vals[1] - vals[0]) if len(vals) > 1 else big
            if pen > best_pen:
                best_pen, best_line = pen, ("col", j)
        if best_line is None:
            break
        kind, idx = best_line
        if kind == "row":
            i = idx
            j = min(active_cols, key=lambda jj: c_w[i, jj])
        else:
            j = idx
            i = min(active_rows, key=lambda ii: c_w[ii, j])
        amount = min(s_rem[i], d_rem[j])
        plan[i, j] += amount
        s_rem[i] -= amount
        d_rem[j] -= amount

    # --- Vogel 初始解的质量评估 -------------------------------------
    vogel_plan = plan.copy()
    vogel_cost = float((vogel_plan * c_w).sum())
    basis_count = int(np.sum(vogel_plan > 1e-9))
    degenerate = basis_count < (m + n - 1)
    iterations = 1

    # --- 精确求解：运输问题等价于一个 LP -----------------------------
    n_var = m * n
    c_obj = c_w.ravel().copy()
    A_eq = np.zeros((m + n, n_var))
    b_eq = np.zeros(m + n)
    for i in range(m):
        A_eq[i, i * n:(i + 1) * n] = 1.0
        b_eq[i] = s_w[i]
    for j in range(n):
        A_eq[m + j, j::n] = 1.0
        b_eq[m + j] = d_w[j]

    lp = simplex_lp(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * n_var,
                    maximize=False, max_iter=2000)
    if lp["status"] != "optimal":
        # 理论上平衡后的运输问题一定可行有界；真出问题就退回 Vogel 解并如实标注
        return {"total_cost": vogel_cost, "plan": vogel_plan,
                "vogel_cost": vogel_cost, "vogel_plan": vogel_plan,
                "improved": False, "iterations": iterations,
                "balanced": balanced, "degenerate": degenerate,
                "lp_status": lp["status"]}

    exact_plan = np.asarray(lp["x"], dtype=float).reshape(m, n)
    exact_plan[np.abs(exact_plan) < 1e-9] = 0.0
    total = float((exact_plan * c_w).sum())
    return {"total_cost": total, "plan": exact_plan,
            "vogel_cost": vogel_cost, "vogel_plan": vogel_plan,
            "improved": bool(total < vogel_cost - 1e-9),
            "iterations": iterations, "balanced": balanced,
            "degenerate": degenerate, "lp_status": "optimal"}


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。"""
    out: Dict[str, object] = {}

    # 1) LP：max 3x+2y s.t. x+y<=4, x+3y<=6
    #    顶点 (0,0)=0、(4,0)=12、(0,2)=4、交点 (3,1)=11 -> 最优 x=4,y=0，值 12
    #    （用 scipy.optimize.linprog 独立核对过：-fun = 12.0，x = [4, 0]）
    lp = simplex_lp([3, 2], A_ub=[[1, 1], [1, 3]], b_ub=[4, 6], maximize=True)
    out["lp_status"] = lp["status"]
    out["lp_fun"] = round(float(lp["fun"]), 6)
    out["lp_x"] = [round(float(v), 6) for v in lp["x"]]

    # 1b) 变量上界回归测试：max 5x+4y s.t. 6x+4y<=13, x<=3
    #     最优是 y=3.25 处取 13；若上界被错误广播成 x+y<=3 会退化成 12.5（曾出的 bug）
    lp_b = simplex_lp([5, 4], A_ub=[[6, 4]], b_ub=[13],
                      bounds=[(0.0, 3.0), (0.0, None)], maximize=True)
    out["lp_bound_fun"] = round(float(lp_b["fun"]), 6)
    out["lp_bound_x"] = [round(float(v), 6) for v in lp_b["x"]]

    # 2) 等式约束：x + y = 1, min x^2 无意义，改 min -x -> x=1,y=0
    lp2 = simplex_lp([-1, 0], A_eq=[[1, 1]], b_eq=[1], bounds=[(0, None), (0, None)])
    out["lp_eq_fun"] = round(float(lp2["fun"]), 6)

    # 3) 不可行：x <= 1 且 x >= 5（用 x - y = 0 style 不好，直接 x>=5 需要 -> 用 -x <= -5）
    lp3 = simplex_lp([1], A_ub=[[1], [-1]], b_ub=[1, -5])
    out["lp_infeasible"] = lp3["status"]

    # 4) 无界：max x, x >= 0
    lp4 = simplex_lp([1], maximize=True, bounds=[(0, None)])
    out["lp_unbounded"] = lp4["status"]

    # 5) 整数规划：max 5x+4y s.t. 6x+4y<=13, x,y 非负整数
    #    暴力枚举精确最优为 (0,3) 值 12（(2,0) 只有 10），用来抓分支定界的剪枝错误
    ip = branch_and_bound_ilp([5, 4], A_ub=[[6, 4]], b_ub=[13], maximize=True)
    out["ip_fun"] = round(float(ip["fun"]), 6)
    out["ip_x"] = [round(float(v), 6) for v in ip["x"]]
    out["ip_nodes"] = int(ip["nodes"])

    # 6) 背包：容量 10，物品 (w,v) = (5,10) (4,40) (6,30) (3,50)
    #    最优是选物品 2 和 3：重量 4+3=7，价值 40+50=90
    ks = knapsack_dp([5, 4, 6, 3], [10, 40, 30, 50], 10)
    out["knapsack_value"] = round(float(ks["max_value"]), 6)
    out["knapsack_weight"] = int(ks["total_weight"])

    # 7) 指派：[[4,2,8],[1,5,3],[6,4,2]] 最优为 2+1+2 = 5，指派 (列) = [1,0,2]
    cost = [[4, 2, 8], [1, 5, 3], [6, 4, 2]]
    asg = assignment_hungarian(cost)
    out["assign_cost"] = round(float(asg["total_cost"]), 6)
    out["assign"] = [int(v) for v in asg["assignment"]]

    # 8) 运输：经典 3x3 算例，精确最优 810（与 scipy.optimize.linprog 的 810.0 一致）
    tr = transportation_vogel([[8, 6, 10], [9, 12, 13], [14, 9, 16]],
                              [20, 30, 30], [30, 20, 30])
    out["transport_cost"] = round(float(tr["total_cost"]), 6)
    out["transport_vogel_cost"] = round(float(tr["vogel_cost"]), 6)
    out["transport_balanced"] = bool(tr["balanced"])
    out["transport_degenerate"] = bool(tr["degenerate"])
    out["transport_lp_status"] = str(tr["lp_status"])

    return out
