"""线性规划、整数规划、指派与运输问题。

这个模块提供竞赛里真正会被用到的那几件事。**共 9 个公开函数**，按用途分成四组：

- 通用求解器：``simplex_lp``（两阶段单纯形，支持等式/不等式/变量上下界）、
  ``branch_and_bound_ilp``（小规模整数/0-1 规划，在 LP 松弛上分支定界）；
- 组合结构精确解：``knapsack_dp``（0-1 背包动态规划）、``assignment_hungarian``
  （指派问题 Kuhn-Munkres / 匈牙利算法，O(n^3)）、``transportation_vogel``
  （运输问题 Vogel 初始解 + 等价 LP 精确最优）；
- 目标规划：``goal_programming``（按优先级分层最小化偏差）；
- 不确定性与选址：``scenario_robust_lp``（情景鲁棒 LP）、``chance_constrained_lp``
  （机会约束 LP）、``facility_location``（离散选址组合枚举）。

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
    "goal_programming",
    "scenario_robust_lp",
    "chance_constrained_lp",
    "facility_location",
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
        raise ValueError(
            f"转轴元素为 0（|{piv}| < 1e-12），无法做除法：LP 在当前基下数值退化，"
            "请检查约束是否冗余/尺度是否病态后重试")
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
        - **数值退化的异常类型**：内部转轴 ``_pivot`` 遇到 ``|转轴元素| < 1e-12`` 时会抛
          ``ValueError``（提示"转轴元素为 0 / LP 数值退化"），这是本模块唯一的数值性失败路径；
          调用方只需捕获 ``ValueError``，不会漏出 ``ZeroDivisionError``。
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
        ``x``、``fun``、``nodes``（实际探索节点数）、``gap``。

        ``gap`` **只在非最优路径上才有意义**，且各路径的语义不同：

        - ``status == "optimal"``：``gap`` 被**硬编码为 ``0.0``**，并不是真实算出来的间隙
          （搜索自然结束即认为已证最优），不要把它当作"最优性证书里的间隙"来引用；
        - ``status == "node_limit"``：``gap = |lb - best_val| / max(|lb|, 1e-12)``，
          其中 ``lb`` 是**用根节点边界重解的 LP 松弛值**（见下），因此这是相对根松弛界的
          一个偏大（保守）的**启发式**相对间隙；若此时还没有可行整数解（``best_x is None``），
          ``gap`` 为 None；
        - ``status == "infeasible"``：``gap`` 为 None。

        内部算 gap 时用的那个 ``lb`` 是**根节点 LP 松弛值**（重新用最初的 ``bounds`` 调一次
        :func:`simplex_lp` 得到），不是"当前开节点里最好的界"，也不是最终下界；它只是最弱但最易得的
        一个下界，所以 ``node_limit`` 下的 gap 会偏大，只适合做粗略的"离最优还有多远"提示。

    算法:
        对 LP 松弛求解 → 若解已全整数则更新上界 → 否则挑一个小数变量 x_j 分两支
        ``x_j <= floor`` 与 ``x_j >= ceil`` 递归；用当前最好整数解剪枝。

    复杂度:
        时间 O(2^n) 最坏；空间 O(n)（递归深度）。

    陷阱:
        - 分支定界**不是**多项式算法：变量数超过 ~50 个时请换 OR-Tools CP-SAT。
        - ``gap`` 只在 ``status == "node_limit"`` 时是算出来的（且基准是根 LP 松弛值，
          gap 偏大/保守）；``status == "optimal"`` 时它恒为硬编码的 ``0.0``，不是真实的间隙。
          代码里对间隙取了绝对值，因此它**不会**出现负数。
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
    """运输问题：Vogel 近似法求初始可行解 + 等价 LP 求精确最优（**不实现 MODI 迭代**）。

    参数:
        cost: 单位运价矩阵，形状 (m 个产地, n 个销地)。
        supply: 各产地供应量，长度 m。
        demand: 各销地需求量，长度 n。

    返回:
        dict，键为:

        - ``total_cost`` / ``plan``：**精确最优**总运费与运输方案（由 :func:`simplex_lp` 求解等价 LP 得到）
        - ``vogel_cost`` / ``vogel_plan``：Vogel 近似法的初始方案，用于展示"启发式离最优有多远"
        - ``improved``：精确解是否严格优于 Vogel 初始解
        - ``iterations``：**恒为整数 ``1``，是硬编码的常量**。本实现不做 MODI（位势法）重优化迭代，
          因此它既不表示 MODI 轮数，也不表示任何真实的迭代/计数过程，只是为保持返回结构稳定而保留的
          占位键——**不要**把它当作收敛轮数或迭代次数写进论文。真实的精确最优由下面的 LP 一次性给出。
        - ``degenerate``：**独立于** ``iterations`` 的布尔标志，表示 Vogel 初始解里基格数量
          是否少于 ``m+n-1``（退化征兆）。
        - ``balanced``：是否产销平衡；不平时会补一个运价为 0 的虚拟产地/销地
        - ``lp_status``：精确 LP 的求解状态

    算法:
        1. 若产销不平衡，补一个虚拟产地/销地（运价 0）使其平衡。
        2. **Vogel 近似法**：每次在行/列罚数（次小运价 − 最小运价）最大的那条线上，
           用最小运价格尽量多地运输，得到一个高质量初始可行解。
        3. **精确求解**：把运输问题写成 LP
           ``min sum c_ij x_ij, s.t. 行和 = 供应量, 列和 = 需求量, x >= 0``，
           用 :func:`simplex_lp` 求精确最优——一次 LP 调用就得到最优解，**没有**任何
           位势法/MODI 重优化迭代（所以返回值里的 ``iterations`` 不是迭代轮数，见"返回"）。

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
# 目标规划 / 情景鲁棒 LP / 机会约束 LP / 离散选址
# --------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """标准正态分布的分位数（Acklam 有理逼近，相对误差 < 1.2e-9）。

    参数:
        p: 概率，必须严格落在 ``(0, 1)`` 内。

    返回:
        float，满足 ``P(Z <= z) = p`` 的 z。

    算法:
        Acklam (1995) 的分段有理逼近：``p < 0.02425`` 与 ``p > 1 - 0.02425`` 走尾部分支，
        其余走中心分支；这里只做一次逼近，不再做 Halley 修正——对"用
        ``b - z_alpha * sigma`` 做确定性等价"这个用途，1e-9 已经远超需要。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        - ``p`` 取 0 或 1 时精确分位数是 ±inf，本实现直接抛 ValueError 而不是返回 inf，
          否则 ``b - z*sigma`` 会变成 NaN 并静默污染整张单纯形表。
        - 这是**近似**分位数：需要 1e-15 级精度时请换更精确的算法（本仓库不引入 scipy）。
        - 逼近精度在尾部（p<0.01）最差，做 99.99% 置信度的机会约束时要留意。

    参考:
        Acklam, P. J. (1995) "An algorithm for computing the inverse normal cumulative
        distribution function"；Beasley & Springer (1977) 分位数逼近综述。
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p 必须严格落在 (0, 1) 内，得到 {p}")
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)

    low = 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    if p > 1.0 - low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -(num / den)
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num / den


def goal_programming(c, A_ub=None, b_ub=None, targets=None, weights=None,
                     priority=None) -> Dict[str, object]:
    """加权 / 分层目标规划：把"尽量达到若干目标值"写成带正负偏差变量的 LP。

    参数:
        c: 目标系数。二维 ``(n_goals, n_vars)`` 时第 k 行给出第 k 个目标函数
            ``c_k @ x``；一维长度 ``n_vars`` 时视为**单个**目标（内部补成一行）。
        A_ub: **硬约束**矩阵 ``A_ub @ x <= b_ub``（必须严格满足，与"目标"区分开）；
            None 表示无硬约束。
        b_ub: 硬约束右端项。
        targets: 长度 ``n_goals`` 的期望值（aspiration level），不能为 None。
        weights: 长度 ``n_goals`` 的非负权重（同一权重同时惩罚未达标量与超额量）；
            也可以给 ``(n_goals, 2)`` 的数组，第 k 行是 ``(w_minus, w_plus)``，
            分别惩罚"未达标量"与"超额量"（例如只在乎不超支就取 ``(0, w)``）。
            None 表示全部取 1，即最小化总绝对偏差。
        priority: 长度 ``n_goals`` 的优先级层号（整数，越小越优先）。None 表示单层加权
            求解；给了层号则做**字典序（preemptive）**求解：先最小化第一层的加权偏差，
            把该层最优值固定成新约束，再优化下一层。
            **注意：本实现不校验 priority**（不检查长度、不要求非负整数、也不校验是否取整）
            —— 层号只做 ``int()`` 截断，且代码会按目标下标 0..n_goals-1 去索引它，
            所以长度必须 >= n_goals，否则抛的是 ``IndexError`` 而不是 ``ValueError``。

    返回:
        dict，键为：
        ``x``           长度 n_vars 的最优决策向量（np.ndarray）；
        ``achieved``    长度 n_goals 的实际达成值 ``c_k @ x``（np.ndarray）；
        ``deviations``  dict：``d_minus`` / ``d_plus``（长度 n_goals 的列表）、
                        ``absolute``（``|achieved - targets|``）、``weighted_sum``；
        ``status``      :func:`simplex_lp` 的状态字符串；
        ``objective``   float，按用户权重算出的加权偏差总和（0 表示全部目标精确达成）。
        非 ``"optimal"`` 时 ``x`` / ``achieved`` / ``deviations`` / ``objective`` 均为 None。

    算法:
        1. 每个目标引一对偏差变量，写成等式
           ``c_k @ x + d_k^- - d_k^+ = targets[k]``，``d_k^-, d_k^+ >= 0``
           （二者不会同时为正，否则可以同时减去 min 而改善目标）。
        2. 目标函数 ``min sum_k (w_k^- d_k^- + w_k^+ d_k^+)``，硬约束原样保留，
           拼成一个变量数为 ``n_vars + 2*n_goals`` 的 LP 交给 :func:`simplex_lp`。
        3. ``priority`` 非空时按层号（``int()`` 截断后）升序逐层调用：第 L 层只优化本层偏差，
           求解后**无条件**把该层的加权偏差和加一条
           ``<= 最优值 + 1e-9*(1+|最优值|)`` 的约束锁住——**最后一层之后也会追加一条**，
           只不过它不会再被任何后续求解使用，因此对结果没有影响（附加约束总数 = 层数）。

    复杂度:
        时间：单层为一次单纯形迭代，分层时为 O(层数) 次调用；
        空间 O((m_hard + n_goals) x (n_vars + 3*n_goals))。

    陷阱:
        - 目标规划**必须有硬约束或变量上界**：偏差变量永远能让等式成立，没有硬约束时
          目标系数为 0 的列会无界（返回 ``status="unbounded"``），这不是数值 bug。
        - 决策变量沿用 :func:`simplex_lp` 的默认口径 ``x >= 0``；需要自由变量请先做平移。
        - 只惩罚单侧偏差时把另一侧权重设为 0 会让解"超调"到任意远处——这是目标规划最
          常见的误用；要限制超调请同时给硬约束。
        - 分层求解对容差敏感：锁层用的 ``1e-9`` 容差不能放大，否则低优先级目标会悄悄
          破坏已经达到的高优先级目标。
        - **每层求解后都会压一条锁定约束，最后一层也不例外**：代码在循环体内无条件执行
          ``extra_A.append(...)``，包括最后一层。这条多余的约束不会被再次求解，所以不改变
          任何结果，但如果你按"层数 = 附加约束数 - 1"去推理行数会算错（实际是相等）。
        - **``priority`` 完全不做校验**：层号仅经 ``int()`` 截断，因此 ``[0.9, 1.9]`` 被当成
          层 0/1，``[0.4, 0.6]`` 会**合并成同一层**（静默退化成单层加权、不再是字典序求解）；
          负层号被接受且排在最前；``priority`` 长度小于目标数时，按目标下标取层号会抛
          ``IndexError``（不是本模块惯用的 ``ValueError``）。传参前请自行保证
          "长度 = n_goals、都是非负整数"。
        - 目标数与硬约束数都算进 LP 行数，目标很多时单纯形表会明显变高。

    参考:
        Charnes & Cooper (1961) 目标规划；Ignizio (1976) 分层（preemptive）目标规划；
        Tamiz, Jones & Romero (1998) 目标规划综述。
    """
    c_goal = as_matrix(c, "c")
    n_goals, n_vars = c_goal.shape
    if targets is None:
        raise ValueError("targets 不能为 None：每个目标都需要一个期望值")
    t_vec = as_vector(targets, "targets")
    if t_vec.size != n_goals:
        raise ValueError(f"targets 长度 {t_vec.size} 与目标个数 {n_goals} 不一致")

    if weights is None:
        w_minus = np.ones(n_goals)
        w_plus = np.ones(n_goals)
    else:
        w_raw = np.asarray(weights, dtype=float)
        if w_raw.ndim == 2:
            if w_raw.shape != (n_goals, 2):
                raise ValueError(
                    f"weights 形状 {w_raw.shape} 与 (目标数, 2)=({n_goals}, 2) 不一致")
            w_minus = w_raw[:, 0].copy()
            w_plus = w_raw[:, 1].copy()
        else:
            w_vec = as_vector(weights, "weights")
            if w_vec.size != n_goals:
                raise ValueError(f"weights 长度 {w_vec.size} 与目标个数 {n_goals} 不一致")
            w_minus = w_vec.copy()
            w_plus = w_vec.copy()
    if np.any(w_minus < 0) or np.any(w_plus < 0):
        raise ValueError("weights 必须非负")

    if priority is None:
        levels = [list(range(n_goals))]
    else:
        p_raw = as_vector(priority, "priority")
        level_ids = sorted({int(v) for v in p_raw})
        levels = [[k for k in range(n_goals) if int(p_raw[k]) == lv] for lv in level_ids]

    if A_ub is not None and len(np.asarray(A_ub)) > 0:
        if b_ub is None:
            raise ValueError("给了 A_ub 就必须同时给 b_ub")
        A_ub_m = as_matrix(A_ub, "A_ub")
        b_ub_v = as_vector(b_ub, "b_ub")
        if A_ub_m.shape[1] != n_vars or A_ub_m.shape[0] != b_ub_v.size:
            raise ValueError(f"A_ub 形状 {A_ub_m.shape} 与 b_ub / c（{n_vars} 个变量）不匹配")
    else:
        A_ub_m = np.zeros((0, n_vars))
        b_ub_v = np.zeros(0)

    n_var = n_vars + 2 * n_goals
    # 硬约束只作用在决策变量上，偏差变量的系数补 0（列数必须对齐到 n_var）
    A_ub_m = np.hstack([A_ub_m, np.zeros((A_ub_m.shape[0], 2 * n_goals))])
    A_eq = np.zeros((n_goals, n_var))
    for k in range(n_goals):
        A_eq[k, :n_vars] = c_goal[k]
        A_eq[k, n_vars + k] = 1.0
        A_eq[k, n_vars + n_goals + k] = -1.0

    bounds = [(0.0, None)] * n_var
    extra_A: List[np.ndarray] = []
    extra_b: List[float] = []
    x_sol: Optional[np.ndarray] = None
    status = "infeasible"

    for level in levels:
        obj = np.zeros(n_var)
        for k in level:
            obj[n_vars + k] = w_minus[k]
            obj[n_vars + n_goals + k] = w_plus[k]
        if extra_A:
            A_all = np.vstack([A_ub_m] + extra_A)
            b_all = np.concatenate([b_ub_v, np.asarray(extra_b, dtype=float)])
        elif A_ub_m.shape[0]:
            A_all = A_ub_m
            b_all = b_ub_v
        else:
            A_all = None
            b_all = None
        res = simplex_lp(obj, A_ub=A_all, b_ub=b_all, A_eq=A_eq, b_eq=t_vec,
                         bounds=bounds)
        status = str(res["status"])
        if status != "optimal":
            return {"x": None, "achieved": None, "deviations": None,
                    "status": status, "objective": None}
        x_sol = np.asarray(res["x"], dtype=float)
        layer_value = float(obj @ x_sol)
        extra_A.append(obj.reshape(1, -1))
        extra_b.append(layer_value + 1e-9 * (1.0 + abs(layer_value)))

    x = x_sol[:n_vars]
    d_minus = x_sol[n_vars:n_vars + n_goals]
    d_plus = x_sol[n_vars + n_goals:]
    achieved = c_goal @ x
    weighted_sum = float(np.dot(w_minus, d_minus) + np.dot(w_plus, d_plus))
    deviations = {
        "d_minus": [float(v) for v in d_minus],
        "d_plus": [float(v) for v in d_plus],
        "absolute": [float(abs(v)) for v in (achieved - t_vec)],
        "weighted_sum": weighted_sum,
    }
    return {"x": x, "achieved": achieved, "deviations": deviations,
            "status": status, "objective": weighted_sum}


def scenario_robust_lp(scenarios, A_ub=None, b_ub=None, budget=0.1,
                       method="min_max_regret", bounds=None) -> Dict[str, object]:
    """离散情景下的鲁棒线性规划（最坏情形 / 最小最大后悔值）。

    参数:
        scenarios: 若干候选目标系数向量（list of 长度 n 的序列，或形状 (K, n) 的数组）。
        A_ub: 硬约束矩阵 ``A_ub @ x <= b_ub``（要求等号时写 ``-a @ x <= -b`` 两行）。
        b_ub: 硬约束右端项。
        budget: 鲁棒预算 β ∈ [0, 1]，用 Hurwicz 型凸组合在"名义均值"与"最坏情景"之间插值：
            ``min_x (1-β)·mean_s c_s@x + β·max_s c_s@x``（``"min_max"``），
            ``min_x (1-β)·mean_s r_s(x) + β·max_s r_s(x)``（``"min_max_regret"``，
            其中 ``r_s(x) = c_s@x - z_s*``，``z_s*`` 是情景 s 单独求解的最优值）。
            β=1 退化为教科书的 min-max / min-max-regret，β=0 退化为名义（平均情景）解。
        method: ``"min_max"``（最坏情形）或 ``"min_max_regret"``（最小最大后悔值）。
        bounds: 决策变量的 ``(lo, hi)`` 列表；None 表示 ``x >= 0``。

    返回:
        dict，键为：
        ``x``               长度 n 的最优决策向量（np.ndarray；无解时为 None）；
        ``objective``       float，**返回解在最坏情景下的目标值** ``max_s c_s@x``；
        ``worst_scenario``  最坏情景的下标（``"min_max"`` 取成本最大者，
                            ``"min_max_regret"`` 取后悔值最大者；并列取最小下标）。
                            鲁棒 LP 没解到最优时这里是**哨兵值 ``-1``**（见"陷阱"）；
        ``regret``          float，跨全部情景的最大后悔值 ``max_s (c_s@x - z_s*)``。

    算法:
        1. 先对每个情景单独调 :func:`simplex_lp` 求 ``z_s*``（后悔值的基准；任一情景不可行
           就无法定义后悔值，直接抛 ValueError）。
        2. 引入标量变量 t 表示最坏情形：约束 ``c_s @ x - t <= 0``（min_max）或
           ``c_s @ x - t <= z_s*``（min_max_regret），目标系数为
           ``(1-β)·mean_s c_s``（x 部分）与 ``β``（t 部分），交 :func:`simplex_lp` 求解。
        3. 用解出的 x 回代，**重新计算** ``max_s c_s@x`` 与 ``max_s r_s(x)`` 作为返回值——
           如此 ``objective`` 永远是真实的最坏情形值（β=0 时 t 是无代价变量，不能直接采信），
           从而恒有 ``objective >= max_s z_s*``。

    复杂度:
        时间 O((K+1) 次单纯形迭代) / 空间 O((m + K) x (n + 1))。

    陷阱:
        - 返回值 ``objective`` 是**最坏情景值**，而内部最小化的是混合目标；β<1 时两者不等，
          论文里报告鲁棒性时必须说明用的是哪一个（本实现用最坏情景值）。
        - 情景集合是**离散**的，没有覆盖到的情况不会被保护；``budget`` 只是凸组合权重，
          不是 Bertsimas-Sim 意义上的"不确定度预算 Γ"，用错口径会被审稿人抓。
        - ``z_s*`` 是各情景**单独**的最优值：某情景不可行、或者最优值无界时后悔值没有定义，
          本实现会抛 ValueError 而不是返回 NaN。
        - 只有 A_ub（≤ 约束）时，非负成本问题的"最坏情形最优解"常常就是 x=0；
          需要下限请自己写成 ``-a@x <= -b``。
        - **失败信号是哨兵值，而不是状态，且与 chance_constrained_lp 不对称**：鲁棒 LP 本身
          没解到最优时，本函数返回 ``{"x": None, "objective": None, "worst_scenario": -1,
          "regret": None}``——``worst_scenario = -1`` 是"失败"哨兵（而 -1 在 Python 里又恰好是
          合法下标，先判断 ``< 0`` 再索引 ``scenarios``）；返回 dict 里**根本没有 ``status`` 键**，
          调用方拿不到底层 :func:`simplex_lp` 的状态。
          对比 :func:`chance_constrained_lp`：它在返回 dict 里给出 ``status``
          （``"infeasible"``/``"unbounded"``/…），并对非法 ``alpha``/``sigma`` 直接抛 ``ValueError``。
          因此同一个"LP 没解出来"在两个函数里的表现完全不同：写统一封装时既不能假设
          "没有 status 就是成功"，也不能假设"失败一定抛异常"。
          另外，**情景 ``z_s*`` 阶段**的失败（某情景单独求解不可行/无界）本函数是抛 ``ValueError``
          的，只有最后的鲁棒 LP 失败才退化成 -1 哨兵。

    参考:
        Kouvelis & Yu (1997) 离散情景鲁棒优化；Savage (1951) 最小最大后悔准则；
        Ben-Tal, El Ghaoui & Nemirovski (2009) 鲁棒优化教材。
    """
    if method not in ("min_max", "min_max_regret"):
        raise ValueError(f"method 只支持 'min_max' / 'min_max_regret'，得到 {method!r}")
    beta = float(budget)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"budget 必须落在 [0, 1] 内，得到 {budget}")

    scen_list = [as_vector(s, f"scenarios[{i}]") for i, s in enumerate(scenarios)]
    if not scen_list:
        raise ValueError("scenarios 不能为空")
    n = scen_list[0].size
    for i, s in enumerate(scen_list):
        if s.size != n:
            raise ValueError(f"scenarios[{i}] 长度 {s.size} 与第一个情景的 {n} 不一致")
    S = np.vstack(scen_list)
    n_scen = S.shape[0]

    if A_ub is not None and len(np.asarray(A_ub)) > 0:
        if b_ub is None:
            raise ValueError("给了 A_ub 就必须同时给 b_ub")
        A_ub_m = as_matrix(A_ub, "A_ub")
        b_ub_v = as_vector(b_ub, "b_ub")
        if A_ub_m.shape[1] != n or A_ub_m.shape[0] != b_ub_v.size:
            raise ValueError(f"A_ub 形状 {A_ub_m.shape} 与 b_ub / scenarios（{n} 个变量）不匹配")
        has_rows = True
    else:
        A_ub_m = np.zeros((0, n))
        b_ub_v = np.zeros(0)
        has_rows = False

    z_star = np.zeros(n_scen)
    for k in range(n_scen):
        res_k = simplex_lp(S[k], A_ub=(A_ub_m if has_rows else None),
                           b_ub=(b_ub_v if has_rows else None), bounds=bounds)
        if res_k["status"] != "optimal":
            raise ValueError(
                f"情景 {k} 单独求解的状态为 {res_k['status']}，无法定义后悔值")
        z_star[k] = float(res_k["fun"])

    if bounds is None:
        var_bounds = [(0.0, None)] * n
    else:
        if len(bounds) != n:
            raise ValueError(f"bounds 长度 {len(bounds)} 与变量数 {n} 不一致")
        var_bounds = [tuple(b) for b in bounds]
    var_bounds = var_bounds + [(None, None)]

    n_var = n + 1
    obj = np.zeros(n_var)
    obj[:n] = (1.0 - beta) * S.mean(axis=0)
    obj[n] = beta

    A_rows = np.zeros((n_scen, n_var))
    A_rows[:, :n] = S
    A_rows[:, n] = -1.0
    b_rows = np.zeros(n_scen) if method == "min_max" else z_star.copy()

    if has_rows:
        # 硬约束只作用在 x 上，t 的系数补 0
        A_all = np.vstack([np.hstack([A_ub_m, np.zeros((A_ub_m.shape[0], 1))]), A_rows])
        b_all = np.concatenate([b_ub_v, b_rows])
    else:
        A_all = A_rows
        b_all = b_rows

    lp = simplex_lp(obj, A_ub=A_all, b_ub=b_all, bounds=var_bounds)
    if lp["status"] != "optimal":
        return {"x": None, "objective": None, "worst_scenario": -1, "regret": None}

    x = np.asarray(lp["x"], dtype=float)[:n]
    costs = S @ x
    regrets = costs - z_star
    if method == "min_max":
        worst = int(np.argmax(costs))
        objective = float(costs[worst])
        regret = float(regrets.max())
    else:
        worst = int(np.argmax(regrets))
        regret = float(regrets[worst])
        objective = float(costs[worst])
    return {"x": x, "objective": objective, "worst_scenario": worst, "regret": regret}


def chance_constrained_lp(c, A_ub=None, b_ub=None, sigma=None, alpha=0.95,
                          bounds=None, maximize=False) -> Dict[str, object]:
    """机会约束 LP 的确定性等价（约束右端项含独立正态扰动）。

    参数:
        c: 长度 n 的目标系数。
        A_ub: 不等式约束矩阵，第 i 行是 ``a_i @ x <= b_i``。
        b_ub: 名义右端项。
        sigma: 各约束右端项扰动的标准差，标量（各约束同一标准差）、长度 m 的序列，
            或 None（等价于全 0，即退化回确定性 LP）。必须非负。
        alpha: 约束成立概率，``0 < alpha < 1``；``z_alpha`` 由本模块自己的
            :func:`_norm_ppf` 给出。
        bounds / maximize: 透传给 :func:`simplex_lp`。
        （注意 ``bounds`` 对决策变量有效，不做概率化处理。）

    返回:
        dict，键为：
        ``x``           最优解（np.ndarray；无解时为 None）；
        ``objective``   float，**名义目标值** ``c @ x``（不含任何惩罚项）；
        ``z_alpha``     float，标准正态分位数 ``z_alpha``；
        ``slack``       长度 m 的**原始**约束松弛 ``b_ub - A_ub @ x``（np.ndarray）；
        ``status``      :func:`simplex_lp` 的状态字符串。

    算法:
        1. 约束 ``P(a_i @ x <= b_i + ξ_i) >= alpha``，``ξ_i ~ N(0, sigma_i^2)`` 独立。
        2. 标准化：``P(ξ_i <= b_i - a_i@x) = Φ((b_i - a_i@x)/sigma_i) >= alpha``，
           于是确定性等价为 ``a_i @ x <= b_i - z_alpha * sigma_i``（``z_alpha = Φ^{-1}(alpha)``）。
        3. 用替换后的右端项调用 :func:`simplex_lp`；``sigma = 0`` 时右端项逐位不变，
           因此结果与直接调用 :func:`simplex_lp` 完全一致（1e-9 以内，实测为 0）。

    复杂度:
        时间 O(一次单纯形迭代) / 空间 O(m x n)（与 :func:`simplex_lp` 相同）。

    陷阱:
        - 这只对**单侧**约束成立：``a@x >= b`` 形式必须写成 ``-a@x <= -b`` 再减 z*sigma，
          直接对等式约束套用会算反方向（下限应该是 ``b + z*sigma``）。
        - 假设是各右端项**独立**正态且分布已知（均值 0、标准差 sigma）；相关扰动要用
          协方差矩阵做联合正态的等效标准差，本实现不做。
        - 返回的 ``slack`` 是相对**名义**右端项的松弛，平均而言会大于 ``z_alpha*sigma``，
          但单次实现中可以更小——不要把它当成"安全余量"来报告。
        - ``alpha`` 越接近 1，``z_alpha`` 越大，可行域越小。但代码里**没有**任何形如
          "``alpha >= 0.99999`` 就不可行"的阈值：唯一的硬限制是 ``0 < alpha < 1``（严格），
          ``alpha >= 1`` 或 ``alpha <= 0`` 会被 :func:`_norm_ppf` 直接拒绝并抛
          ``ValueError``（"p 必须严格落在 (0, 1) 内"），而不是返回 ±inf 或判成不可行。
          另有一个真实的数值阈值：``_norm_ppf`` 在 ``alpha > 1 - 0.02425 = 0.97575`` 时
          切换到尾部有理逼近分支，该分支的相对误差最大，做极高置信度时精度要留有余量。
          至于"高 alpha 下模型变得不可行"，那是确定性等价把右端项压成
          ``b - z_alpha * sigma`` 之后的模型性质，取决于具体数据，与代码阈值无关。

    参考:
        Charnes & Cooper (1959) 机会约束规划；Miller & Wagner (1965) 正态扰动下的确定性
        等价；Prekopa (1995) 随机规划。
    """
    c_vec = as_vector(c, "c")
    n = c_vec.size
    z_alpha = _norm_ppf(float(alpha))

    if A_ub is not None and len(np.asarray(A_ub)) > 0:
        if b_ub is None:
            raise ValueError("给了 A_ub 就必须同时给 b_ub")
        A_ub_m = as_matrix(A_ub, "A_ub")
        b_ub_v = as_vector(b_ub, "b_ub")
        if A_ub_m.shape[1] != n or A_ub_m.shape[0] != b_ub_v.size:
            raise ValueError(f"A_ub 形状 {A_ub_m.shape} 与 b_ub / c（{n} 个变量）不匹配")
        m = A_ub_m.shape[0]
    else:
        A_ub_m = None
        b_ub_v = np.zeros(0)
        m = 0

    if sigma is None:
        sigma_v = np.zeros(m)
    else:
        sig_arr = np.asarray(sigma, dtype=float)
        if sig_arr.ndim == 0:
            sigma_v = np.full(m, float(sig_arr))
        else:
            sigma_v = as_vector(sigma, "sigma")
            if sigma_v.size != m:
                raise ValueError(f"sigma 长度 {sigma_v.size} 与不等式约束数 {m} 不一致")
    if np.any(sigma_v < 0):
        raise ValueError("sigma（标准差）必须非负")

    b_eff = b_ub_v - z_alpha * sigma_v
    res = simplex_lp(c_vec, A_ub=(A_ub_m if m else None),
                     b_ub=(b_eff if m else None), bounds=bounds, maximize=maximize)
    status = str(res["status"])
    if status != "optimal":
        return {"x": None, "objective": None, "z_alpha": z_alpha,
                "slack": None, "status": status}

    x = np.asarray(res["x"], dtype=float)
    slack = (b_ub_v - A_ub_m @ x) if m else np.zeros(0)
    return {"x": x, "objective": float(c_vec @ x), "z_alpha": z_alpha,
            "slack": slack, "status": status}


def facility_location(cost, demand, n_facilities, capacity=None,
                      max_combinations: int = 200000) -> Dict[str, object]:
    """离散选址 p-median：枚举设施组合 + 客户就近指派。

    参数:
        cost: 单位服务成本（通常取距离）矩阵，形状 ``(n_customers, n_sites)``，
            ``cost[i, j]`` 表示把客户 i 指派给候选设施 j 的成本。
        demand: 各客户的需求量（权重），长度 ``n_customers``，非负。
        n_facilities: 要建的设施数 p，``1 <= p <= n_sites``。
        capacity: 设施容量；None 表示不限容量，标量表示所有设施同一容量，长度 ``n_sites``
            的序列表示逐站容量（容量以 demand 的同一单位计）。
        max_combinations: 枚举组合数上限，防止 ``C(n_sites, p)`` 爆炸；超过即抛 ValueError。

    返回:
        dict，键为：
        ``selected``    长度 p 的设施下标列表（升序，即被选中的组合）；
        ``assignment``  长度 n_customers 的指派下标列表，元素取自 ``selected``；
        ``total_cost``  float，``sum_i demand[i] * cost[i, assignment[i]]``；
        ``n_evaluated`` 实际枚举并求解的设施组合数。

        **返回值里没有 ``status`` 键**（与本模块的 LP 求解器不同）：本函数要么正常返回，
        要么抛 ``ValueError``，因此"组合过多无法枚举"和"容量下无可行组合"这两种失败
        无法从返回值里区分（成功返回时 ``n_evaluated`` 也不告诉你是否有组合被容量淘汰）。

    算法:
        1. 用 ``itertools.combinations`` 枚举全部 ``C(n_sites, p)`` 个设施组合
           （按字典序，保证结果确定）。
        2. 无容量约束时，每个客户在所选设施中取成本最小者（并列取下标最小者）。
        3. 有容量约束时改用贪心指派：把所有 (客户, 设施) 对按成本升序排序，依次占用仍有
           剩余容量的设施；出现无法指派的客户就判定该组合不可行并跳过。
        4. 保留总成本最小的组合；成本并列时保留先枚举到的那个。

    复杂度:
        时间 O(C(n_sites, p) * n_customers * p)（有容量时多一个排序的 log 因子）；
        空间 O(n_customers * p)。

    陷阱:
        - **组合爆炸**：这是精确枚举，n_sites=30、p=5 时 C(30,5)=142506 还能勉强跑，
          n_sites=50 完全不可行；大规模请换 MILP 或贪心 + 局部搜索。
        - 有容量时"按成本升序贪心"**不保证**给出该组合下的最优指派（本质是装箱式问题），
          因此结果是"该组合在贪心指派下的成本"；容量只应作为可行性条件，不要用它比较
          不同组合的最优性。密集容量约束的场景请用运输问题式 LP。
        - ``cost`` 的行是客户、列是设施；转置后形状仍合法但语义全变，是最容易静默出错的点。
        - **同一个 ``ValueError`` 承担了两种完全不同的失败**，调用方必须靠异常消息（无法靠返回值）
          区分：

          1. "枚举量太大"：``C(n_sites, p) > max_combinations``，在**枚举开始之前**就抛
             （消息形如 ``C(n_sites, p)=... 超过枚举上限 ...，请缩小规模或改用整数规划``）；
          2. "容量不可行"：容量约束下**所有** ``C(n_sites, p)`` 个组合都指派失败，枚举跑完后才抛
             （消息形如 ``在给定容量下没有任何可行的设施组合，请放宽 capacity``）。

          返回值里**没有** ``status`` 键（LP 求解器才有），所以拿到返回值就无法知道属于哪一种；
          想提前区分请自己先比较 ``math.comb(n_sites, p)`` 与 ``max_combinations``。
          单个组合容量不可行时则被静默跳过（不计入任何"不可行组合数"），只有全军覆没才报错。

    参考:
        Hakimi (1964) p-median 模型；ReVelle & Swain (1970) 选址-分配整数规划；
        Daskin (2013) 网络与离散选址。
    """
    c_mat = as_matrix(cost, "cost")
    n_customers, n_sites = c_mat.shape
    dem = as_vector(demand, "demand")
    if dem.size != n_customers:
        raise ValueError(f"demand 长度 {dem.size} 与 cost 行数 {n_customers} 不一致")
    if np.any(dem < 0):
        raise ValueError("demand 必须非负")

    p = int(n_facilities)
    if p < 1 or p > n_sites:
        raise ValueError(f"n_facilities 必须落在 [1, {n_sites}] 内，得到 {n_facilities}")

    if capacity is None:
        cap: Optional[np.ndarray] = None
    else:
        cap_arr = np.asarray(capacity, dtype=float)
        if cap_arr.ndim == 0:
            cap = np.full(n_sites, float(cap_arr))
        else:
            cap = cap_arr.ravel().astype(float)
            if cap.size != n_sites:
                raise ValueError(f"capacity 长度 {cap.size} 与设施数 {n_sites} 不一致")
        if np.any(cap < 0):
            raise ValueError("capacity 必须非负")

    n_combos = math.comb(n_sites, p)
    if n_combos > int(max_combinations):
        raise ValueError(f"C({n_sites}, {p})={n_combos} 超过枚举上限 {max_combinations}，"
                         f"请缩小规模或改用整数规划")

    best_cost = _INF
    best_combo: Optional[Tuple[int, ...]] = None
    best_assign: Optional[List[int]] = None
    n_evaluated = 0

    for combo in itertools.combinations(range(n_sites), p):
        n_evaluated += 1
        assign: List[int] = []
        total = 0.0
        if cap is None:
            for i in range(n_customers):
                j = min(combo, key=lambda t: (c_mat[i, t], t))
                assign.append(j)
                total += dem[i] * c_mat[i, j]
        else:
            remaining = {j: cap[j] for j in combo}
            pairs = sorted(((c_mat[i, j], i, j)
                            for i in range(n_customers) for j in combo),
                           key=lambda t: (t[0], t[1], t[2]))
            assign = [-1] * n_customers
            for c_ij, i, j in pairs:
                if assign[i] == -1 and remaining[j] >= dem[i] - 1e-12:
                    assign[i] = j
                    remaining[j] -= dem[i]
                    total += dem[i] * c_ij
            if any(a < 0 for a in assign):
                continue
        if total < best_cost - 1e-12:
            best_cost = total
            best_combo = combo
            best_assign = assign

    if best_combo is None:
        raise ValueError("在给定容量下没有任何可行的设施组合，请放宽 capacity")
    return {"selected": [int(j) for j in best_combo],
            "assignment": [int(a) for a in (best_assign or [])],
            "total_cost": float(best_cost),
            "n_evaluated": n_evaluated}


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

    # 9) 目标规划：两个目标（x1+x2=6、x1+2x2=10）在 x1<=2 下**精确可达**（解唯一 x=(2,4)），
    #    因此加权偏差必须为 0——这条能抓出偏差变量的符号写反（d_minus/d_plus 互换）的错误
    gp = goal_programming([[1.0, 1.0], [1.0, 2.0]], A_ub=[[1.0, 0.0]], b_ub=[2.0],
                          targets=[6.0, 10.0], weights=[1.0, 1.0])
    gp_dev = gp["deviations"]
    gp_dev_sum = float(sum(gp_dev["absolute"]))
    if gp["status"] != "optimal" or gp_dev_sum > 1e-7 or abs(float(gp["objective"])) > 1e-7:
        raise AssertionError(
            f"goal_programming 目标可达却留下偏差：status={gp['status']}, "
            f"objective={gp['objective']}, 绝对偏差和={gp_dev_sum}")
    out["goal_programming_status"] = str(gp["status"])
    out["goal_programming_objective"] = round(float(gp["objective"]), 6)
    out["goal_programming_deviation_sum"] = round(gp_dev_sum, 6)
    out["goal_programming_x"] = [round(float(v), 6) for v in gp["x"]]
    out["goal_programming_achieved"] = [round(float(v), 6) for v in gp["achieved"]]

    # 9b) 分层目标规划：第一层 x1+x2=10 可精确达成（x1<=3 时取 x=(3,7)），
    #     第二层 x1-x2=0 只能差 4；手算偏差 = [0, 4]
    gpp = goal_programming([[1.0, 1.0], [1.0, -1.0]], A_ub=[[1.0, 0.0]], b_ub=[3.0],
                           targets=[10.0, 0.0], weights=[1.0, 1.0], priority=[1, 2])
    gpp_dev = gpp["deviations"]
    if abs(gpp_dev["absolute"][0]) > 1e-7:
        raise AssertionError(
            f"分层目标规划第一层应精确达成，实际偏差 {gpp_dev['absolute'][0]}")
    if abs(gpp_dev["absolute"][1] - 4.0) > 1e-6:
        raise AssertionError(
            f"分层目标规划第二层手算偏差为 4，实际 {gpp_dev['absolute'][1]}")
    out["goal_programming_priority_x"] = [round(float(v), 6) for v in gpp["x"]]
    out["goal_programming_priority_dev"] = [round(float(v), 6) for v in gpp_dev["absolute"]]

    # 10) 情景鲁棒 LP：x+y=4（用 -x-y<=-4 表达下限）上三个情景，
    #     手算 min-max = 6（x=y=2，两个情景同时取到），各情景单独最优值都是 4，
    #     所以 min-max 目标值 6 >= 4，且最大后悔值 = 6-4 = 2
    scen = [[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]
    A_rob = np.array([[1.0, 1.0], [-1.0, -1.0]])
    b_rob = np.array([4.0, -4.0])
    z_star = [float(simplex_lp(cs, A_ub=A_rob, b_ub=b_rob)["fun"]) for cs in scen]
    rob_mm = scenario_robust_lp(scen, A_rob, b_rob, budget=0.1, method="min_max")
    rob_mr = scenario_robust_lp(scen, A_rob, b_rob, budget=1.0, method="min_max_regret")
    if float(rob_mm["objective"]) < max(z_star) - 1e-9:
        raise AssertionError(
            f"min-max 目标值 {rob_mm['objective']} 小于任一情景的最优值 {max(z_star)}")
    if abs(float(rob_mm["objective"]) - 6.0) > 1e-6 or abs(float(rob_mm["regret"]) - 2.0) > 1e-6:
        raise AssertionError(
            f"min-max 手算应为 目标 6 / 后悔 2，实际 {rob_mm['objective']} / {rob_mm['regret']}")
    if abs(float(rob_mr["regret"]) - 2.0) > 1e-6:
        raise AssertionError(
            f"min-max-regret 手算后悔值应为 2，实际 {rob_mr['regret']}")
    out["scenario_best_single"] = round(float(max(z_star)), 6)
    out["scenario_min_max_objective"] = round(float(rob_mm["objective"]), 6)
    out["scenario_min_max_regret"] = round(float(rob_mm["regret"]), 6)
    out["scenario_min_max_x"] = [round(float(v), 6) for v in rob_mm["x"]]
    out["scenario_min_max_worst"] = int(rob_mm["worst_scenario"])
    out["scenario_regret_objective"] = round(float(rob_mr["regret"]), 6)
    out["scenario_regret_x"] = [round(float(v), 6) for v in rob_mr["x"]]

    # 11) 机会约束 LP：sigma=0 必须与直接 simplex_lp 完全一致（1e-9）；
    #     z_alpha 与标准正态分位数 1.644854 对拍（1e-6）；sigma>0 后可行域收缩、目标变差
    z095 = _norm_ppf(0.95)
    if abs(z095 - 1.644854) > 1e-6:
        raise AssertionError(f"_norm_ppf(0.95)={z095} 与 1.644854 不符")
    A_cc = np.array([[-1.0, -1.0], [1.0, -1.0]])
    b_cc = np.array([-2.0, 1.0])
    c_cc = [1.0, 2.0]
    cc0 = chance_constrained_lp(c_cc, A_cc, b_cc, sigma=0.0)
    lp_cc = simplex_lp(c_cc, A_ub=A_cc, b_ub=b_cc)
    cc_gap = max(abs(float(cc0["objective"]) - float(lp_cc["fun"])),
                 float(np.max(np.abs(np.asarray(cc0["x"]) - np.asarray(lp_cc["x"])))))
    if cc0["status"] != lp_cc["status"] or cc_gap > 1e-9:
        raise AssertionError(
            f"chance_constrained_lp(sigma=0) 与 simplex_lp 不一致，最大偏差 {cc_gap}")
    cc1 = chance_constrained_lp(c_cc, A_cc, b_cc, sigma=[0.2, 0.1], alpha=0.95)
    if abs(float(cc1["z_alpha"]) - z095) > 1e-12:
        raise AssertionError("chance_constrained_lp 返回的 z_alpha 与 _norm_ppf 不一致")
    if float(cc1["objective"]) <= float(lp_cc["fun"]) - 1e-12:
        raise AssertionError(
            f"加正态扰动后目标值反而更优（{cc1['objective']} <= {lp_cc['fun']}），"
            f"说明确定性等价的方向搞反了")
    out["chance_z_alpha"] = round(float(cc1["z_alpha"]), 6)
    out["chance_status"] = str(cc0["status"])
    out["chance_sigma0_gap"] = round(cc_gap, 12)
    out["chance_sigma0_objective"] = round(float(cc0["objective"]), 6)
    out["chance_sigma0_x"] = [round(float(v), 6) for v in cc0["x"]]
    out["chance_objective"] = round(float(cc1["objective"]), 6)
    out["chance_x"] = [round(float(v), 6) for v in cc1["x"]]
    out["chance_slack"] = [round(float(v), 6) for v in cc0["slack"]]

    # 12) p-median 选址：1 维等距（0..4 五个客户点也是候选点）、等需求，
    #     手算 p=1 最优 = 6（建在 2），p=2 最优 = 3（如 {1,3} 或 {1,4}）
    pos = np.arange(5.0)
    fcost = np.abs(pos[:, None] - pos[None, :])
    fdem = np.ones(5)
    fl2 = facility_location(fcost, fdem, 2)
    fl1 = facility_location(fcost, fdem, 1)
    if abs(float(fl2["total_cost"]) - 3.0) > 1e-9:
        raise AssertionError(f"p-median p=2 的手算最优为 3，实际 {fl2['total_cost']}")
    if abs(float(fl1["total_cost"]) - 6.0) > 1e-9:
        raise AssertionError(f"p-median p=1 的手算最优为 6，实际 {fl1['total_cost']}")
    out["facility_p2_cost"] = round(float(fl2["total_cost"]), 6)
    out["facility_p2_selected"] = [int(v) for v in fl2["selected"]]
    out["facility_p2_assignment"] = [int(v) for v in fl2["assignment"]]
    out["facility_p2_n_evaluated"] = int(fl2["n_evaluated"])
    out["facility_p1_cost"] = round(float(fl1["total_cost"]), 6)
    out["facility_p1_selected"] = [int(v) for v in fl1["selected"]]

    return out
