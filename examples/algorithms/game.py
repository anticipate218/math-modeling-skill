"""博弈论模型：零和博弈 LP、双矩阵纳什、Shapley 值、稳定匹配、演化博弈。

本模块共 8 个公开函数，按用途分成四组：
- 零和与对抗：``zero_sum_value_lp``（零和博弈的 LP 解法）/ ``minimax_alpha_beta``（极大极小搜索）；
- 非零和均衡求解：``nash_support_enumeration`` / ``nash_bargaining_solution`` / ``stackelberg_lp``；
- 合作博弈与分配：``shapley_value``；
- 匹配与演化：``gale_shapley`` / ``replicator_dynamics``。

竞赛里的典型出现场景：
- **零和/对抗**：攻防资源分配、拦截与规避、审批对抗、军事对抗推演；
- **非零和/合作**：两个主体（企业、国家、平台与商家）的收益矩阵同时给出，找纳什均衡；
- **合作博弈/分配**：给定各联盟的收益（特征函数），用 Shapley 值分成本或分收益；
- **匹配**：导师-学生、医院-住院医、志愿与岗位的双边配对（Gale-Shapley）；
- **演化博弈 / ABM**：策略在群体中的比例随时间演化（复制者动态），常与仿真题配合。

关于"解"的表述纪律：
- 纳什均衡**不唯一**时不要只报一个就完事，必须报告均衡集合（本模块的
  :func:`nash_support_enumeration` 就是干这个的）；
- 混合策略是**概率分布**，求解结果必须验证非负与和为 1，否则论文里的数字不可信；
- Shapley 值满足有效性/对称性/哑元/可加性四条公理，但计算量随人数指数增长（n! 项），
  n > 12 左右就必须改用蒙特卡洛抽样近似，本模块给出的是精确枚举版。

依赖：仅 numpy + 标准库 + ``._common`` + ``.optimization``。
"""

from __future__ import annotations

import itertools
from math import factorial
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ._common import ArrayLike, MatrixLike, as_matrix, as_vector, check_square, rng as make_rng
from .optimization import simplex_lp

__all__ = [
    "zero_sum_value_lp",
    "nash_support_enumeration",
    "shapley_value",
    "gale_shapley",
    "replicator_dynamics",
    "minimax_alpha_beta",
    "stackelberg_lp",
    "nash_bargaining_solution",
]

_TOL = 1e-7


# --------------------------------------------------------------------------- #
# 零和博弈
# --------------------------------------------------------------------------- #
def zero_sum_value_lp(payoff, maximize_row: bool = True) -> dict:
    """用线性规划求二人零和博弈的值与最优混合策略。

    参数:
        payoff: (m, n) 支付矩阵。**约定 ``payoff[i, j]`` 是行玩家 i 对列玩家 j 的收益**
            （即列玩家付给行玩家的金额，所以列玩家是"最小化者"）。
        maximize_row: True（默认）表示行玩家是最大化者、列玩家是最小化者；
            若你的矩阵记的是"列玩家的收益"，传 False 会自动转置视角。

    返回:
        ``{"value": float, "row_strategy": np.ndarray(m), "col_strategy": np.ndarray(n),
        "lp_status": str}``。
        ``value`` 是按**原始**矩阵口径的博弈值；``row_strategy`` / ``col_strategy``
        是概率分布（非负、和为 1，已归一化）。

    算法:
        设行玩家用混合策略 ``p``，其保证收益为 ``min_j (p^T A)_j``，求其最大即解
        ``max v  s.t.  sum_i p_i A[i, j] >= v``（对每个 j）。
        零和博弈的最优策略要求收益全为正（LP 标准形要求变量非负），因此先做**平移**：
        ``shift = -min(A) + 1.0``（当 ``min(A) <= 0``）或 ``shift = 0.0``（当 ``min(A) > 0``），
        ``A' = A + shift``——也就是 ``kappa`` **固定取 1.0，不按量级自适应**；解完再按
        ``value = v' - shift`` 还原（策略本身不受平移影响）。
        求解器用 :func:`optimization.simplex_lp`；变量向量是 ``[p_0..p_{m-1}, v]``，
        等式约束 ``sum p = 1``，不等式约束 ``-sum_i p_i A'[i, j] + v <= 0``。
        列玩家的策略**不是**用强对偶/单纯形表的检验数直接搬过来的：:func:`optimization.simplex_lp`
        按契约只返回原始解、不暴露对偶变量，所以本实现调用辅助函数 :func:`_zero_sum_dual_lp`
        对同一（已平移）矩阵**再显式求解一次对偶 LP**（``min u  s.t.  sum_j q_j A'[i, j] <= u,
        sum_j q_j = 1, q >= 0``），代价是多一次 LP 求解，好处是**不依赖求解器内部实现**。

    复杂度:
        时间 = 一次 LP（单纯形实际很快，最坏指数级）；本模块建表 O(mn) / 空间 O(mn)。

    陷阱:
        1. **绝不假设 value >= 0**。零和博弈的值可以是负的（例如"石头剪刀布"型对抗里
           行玩家处于劣势）。若你直接把 LP 目标当 value，平移没还原就会偏。
        2. 混合策略的**不确定支撑集**：当价值相等时，最优混合策略可能不唯一（例如
           [[1,-1],[-1,1]] 的任意 p 都是最优）。所以"用 LP 求出的策略"只是其中一个，
           论文里若声称"最优策略唯一"需要有额外论证。
        3. 平移量必须让**所有**元素为正。本实现的 ``shift`` 是固定的 ``-min(A) + 1.0``
           （``kappa`` 恒为 1.0，**不会**按矩阵量级自适应），平移后若仍有非正元素会直接抛
           ``ValueError``。因此量级悬殊的矩阵（例如含 1e9 级负数）会得到巨大的 ``shift``，
           剩余元素几乎都接近 1，``simplex_lp`` 里 1e-9 级的判定就不再可靠——**请先对支付
           矩阵做无量纲化/缩放**再传入。
        4. LP 不可行/无界的判断：零和博弈的可行域有界（策略单纯形），所以理论上总是
           可行有界；一旦求解器返回别的状态，通常说明数值尺度或平移出了问题，本实现直接抛
           ValueError，不做静默兜底。
        5. 若 ``maximize_row=False``，本函数先对矩阵**转置**再按"行玩家最大化"求解，所以返回的
           ``row_strategy`` 长度等于**你传入矩阵的列数**、其分量对应原矩阵的各**列**
           （视角换成了原来的列玩家）；``col_strategy`` 才对应原矩阵的各**行**。口径不要搞反。

    参考:
        von Neumann 1928（博弈论基本定理）；Dantzig 的 LP 等价形式；Chvátal《Linear Programming》。
    """
    A = as_matrix(payoff, "payoff")
    if not maximize_row:
        A = A.T  # 转置后统一成"行玩家最大化"
    m, n = A.shape
    if m == 0 or n == 0:
        raise ValueError("payoff 不能有空维度")

    # 平移使所有元素 > 0，满足 LP 变量非负要求
    amin = float(A.min())
    shift = -amin + 1.0 if amin <= 0 else 0.0
    Ashift = A + shift
    if Ashift.min() <= 0:
        raise ValueError("平移后仍存在非正元素，无法用标准形单纯形求解")

    # 变量：[p_0..p_{m-1}, v]，目标 maximize v
    n_var = m + 1
    c = np.zeros(n_var)
    c[-1] = 1.0

    # 不等式：-sum_i p_i A'[i, j] + v <= 0
    A_ub = np.zeros((n, n_var))
    A_ub[:, :m] = -Ashift.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)

    # 等式：sum_i p_i = 1
    A_eq = np.zeros((1, n_var))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n_var  # v 可以取负值？不需要：平移后最优 v' > 0
    res = simplex_lp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, maximize=True)
    status = str(res.get("status"))
    if status != "optimal":
        raise ValueError(
            f"零和博弈 LP 求解失败：status={status}。"
            "理论上零和博弈的可行域有界，失败通常来自数值尺度问题（请先检查支付矩阵量级）"
        )
    x = np.asarray(res["x"], dtype=float)
    row = np.clip(x[:m], 0.0, None)
    v_shifted = float(x[-1])
    s = row.sum()
    if s <= 0:
        raise ValueError("LP 返回的行策略全为 0，结果无效（数值退化）")
    row = row / s
    value = v_shifted - shift * row.sum()  # 平移还原：E[A] = E[A'] - shift

    # 列策略：对同一个（已平移）矩阵再解一次对偶 LP。
    # 说明：矩阵博弈的列策略本可以取行问题单纯形表的对偶行（检验数），但
    # optimization.simplex_lp 按契约只返回原始解、不暴露对偶变量，因此这里显式解对偶 LP，
    # 代价是多一次求解，好处是**不依赖求解器内部实现**。
    col = _zero_sum_dual_lp(Ashift, m, n)
    col = np.clip(col, 0.0, None)
    if col.sum() <= 0:
        col = np.full(n, 1.0 / n)
    col = col / col.sum()

    # 自检一：策略对纯策略的保证性（不足说明求解或还原有 bug）
    row_payoff_vs_pure_col = row @ A  # 长度 n
    col_payoff_vs_pure_row = A @ col  # 长度 m
    if row_payoff_vs_pure_col.min() < value - 1e-6:
        raise ValueError(
            f"求解结果不自洽：行策略对某个纯列策略的收益 "
            f"{row_payoff_vs_pure_col.min()} < value {value}"
        )
    if col_payoff_vs_pure_row.max() > value + 1e-6:
        raise ValueError(
            f"求解结果不自洽：列策略被某个纯行策略取得 "
            f"{col_payoff_vs_pure_row.max()} > value {value}"
        )
    return {
        "value": float(value),
        "row_strategy": row,
        "col_strategy": col,
        "lp_status": status,
    }


def _zero_sum_dual_lp(Ashift: np.ndarray, m: int, n: int) -> np.ndarray:
    """显式求解对偶 LP，得到列玩家的最优混合策略。

    参数:
        Ashift: (m, n) 已平移（元素全为正）的支付矩阵。
        m: 行数。
        n: 列数。

    返回:
        np.ndarray，长度 n 的列策略（未归一化，由调用方归一化）。求解失败时返回全 1/n。

    算法:
        列玩家最小化行玩家的最大收益：解
        ``min u  s.t.  sum_j q_j A'[i, j] <= u (∀i),  sum_j q_j = 1,  q >= 0``。
        变量向量 ``[q_0..q_{n-1}, u]``，目标最小化 ``u``。

    复杂度:
        时间 = 一次 LP / 空间 O(mn)。

    陷阱:
        这里求的是**同一个博弈**的另一个方向；不要把它和"另一个博弈"混淆。
        另外 ``u`` 必须允许取到平移后的值（全正），所以下界设 0 即可，不需要负下界。

    参考:
        矩阵博弈的 LP 对偶形式（von Neumann 定理）。
    """
    n_var = n + 1
    c = np.zeros(n_var)
    c[-1] = 1.0
    # sum_j q_j A'[i, j] - u <= 0
    A_ub = np.zeros((m, n_var))
    A_ub[:, :n] = Ashift
    A_ub[:, -1] = -1.0
    b_ub = np.zeros(m)
    A_eq = np.zeros((1, n_var))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_var
    res = simplex_lp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, maximize=False)
    if str(res.get("status")) != "optimal":
        return np.full(n, 1.0 / n)
    q = np.asarray(res["x"], dtype=float)[:n]
    return np.clip(q, 0.0, None)


# --------------------------------------------------------------------------- #
# 双矩阵纳什均衡（支撑集枚举）
# --------------------------------------------------------------------------- #
def nash_support_enumeration(A, B) -> list:
    """用支撑集枚举求双矩阵（2 人一般和）博弈的**全部**纳什均衡。

    参数:
        A: (m, n) 行玩家的收益矩阵。
        B: (m, n) 列玩家的收益矩阵（同形状）。
            零和博弈请设 ``B = -A``；对称博弈常见 ``B = A.T``（此时 A 需为方阵）。

    返回:
        list of dict，每个元素 ``{"row": np.ndarray(m), "col": np.ndarray(n)}``，
        两者都是非负、和为 1 的概率分布，并按（行支撑大小, 列支撑大小, 行策略字典序）
        稳定排序，保证同输入同输出。
        均衡集合可能为空（例如某些博弈只有混合均衡而枚举容差未命中）——但精确算法下
        有限博弈**必然至少有一个纳什均衡**（Nash 1950），返回空列表说明该问题在你的
        容差下没有数值解，应当调大容差或检查输入，而不是断言"不存在均衡"。

    算法:
        Nash 的支撑集枚举（support enumeration）：
        1. 枚举所有非空支撑对 ``(S_r, S_c)``，且**两侧支撑大小都被循环上界
           ``min(m, n)`` 卡住**：代码是 ``range(1, min(m, n) + 1)``，因此
           ``|S_r| <= min(m, n)``、``|S_c| <= min(m, n)``（**不是** ``|S_r| <= n`` /
           ``|S_c| <= m``）。矩阵非方阵时（如 m=2, n=5）两侧支撑都最多枚举到 2，
           大于 ``min(m, n)`` 的支撑对根本不会被尝试；
        2. 在"行玩家在 S_c 上无差异、列玩家在 S_r 上无差异"的线性系统里解出混合策略
           （用 numpy 最小二乘 + 秩检查，避免手写高斯消元的分支爆炸）；
        3. 回代检验全局最优性：``(A q)_i`` 在所有行上被 S_r 取到最大、``(B^T p)_j``
           在所有列上被 S_c 取到最大，且概率非负、和为 1；
        4. 归一化后按容差去重（同一均衡可能被多个支撑对生成）。

    复杂度:
        时间 O(2^m * 2^n * (m n + 线性方程组求解))——**对小规模才可行**。
        m=n=3 时约 2^6=64 个支撑对，秒级；m=n=6 时 4096 对，仍可接受；
        m=n=10 时会涨到 ~1e6 对，必须换 Lemke-Howson 或支撑集剪枝。
        空间 O(m n)。

    陷阱:
        1. **容差是这道题最大的坑**。均衡判定用的是 ``<= tol`` 的浮点比较：容差太小会漏掉
           真实的混合均衡（线性方程解出 1e-9 级别的负概率），太大又会把非均衡判成均衡。
           本实现取 ``tol=1e-7``，并且对负概率做 ``np.clip(0)`` 后依然重新验证全局最优性。
        2. **支撑集枚举的完备性有条件**：它枚举的是"所有纳什均衡"，但在**退化**博弈
           （某个纯策略的收益在多个点上并列最大）中，均衡的支撑可能是非最小的，
           此时同一个均衡会被重复生成，也可能出现"支撑对无解但均衡存在"的假象。
           去重和回代检验能兜住大部分情况，但对严重退化的问题请交叉验证。
        3. ``B`` 的口径容易搞反：``B[i, j]`` 必须是**列玩家**在组合 (i, j) 下的收益，
           不是"列玩家的损失矩阵"。搞反会得到镜像的错误均衡。
        4. 返回的是**纳什均衡集合**，不是"最优解"。多个均衡之间无法用收益比较
           （这正是博弈论区别于优化的地方），论文里要讨论均衡选择（如风险占优、帕累托占优）。
        5. 纯策略均衡也走同一条代码路径（支撑大小为 1），因此返回的仍是混合策略向量
           （退化成 one-hot），不要指望拿到"行号"。

    参考:
        Nash 1950/1951；von Stengel 2002, "Computing Equilibria for Two-Person Games"
        （支撑集枚举的标准参考）；Nisan et al.《Algorithmic Game Theory》第 3 章。
    """
    Amat = as_matrix(A, "A")
    Bmat = as_matrix(B, "B")
    if Amat.shape != Bmat.shape:
        raise ValueError(f"A 与 B 形状必须一致，得到 {Amat.shape} 与 {Bmat.shape}")
    m, n = Amat.shape
    if m == 0 or n == 0:
        raise ValueError("收益矩阵不能有空维度")
    tol = _TOL

    found: List[Dict[str, object]] = []
    rows = list(range(m))
    cols = list(range(n))

    for k_r in range(1, min(m, n) + 1):
        for k_c in range(1, min(m, n) + 1):
            for S_r in itertools.combinations(rows, k_r):
                S_r = list(S_r)
                for S_c in itertools.combinations(cols, k_c):
                    S_c = list(S_c)
                    eq = _solve_support(Amat, Bmat, S_r, S_c, tol)
                    if eq is None:
                        continue
                    p, q = eq
                    if not _is_nash(Amat, Bmat, p, q, S_r, S_c, tol):
                        continue
                    _append_unique(found, p, q, tol)

    found.sort(key=lambda d: (
        int(np.count_nonzero(np.asarray(d["row"], dtype=float) > tol)),
        int(np.count_nonzero(np.asarray(d["col"], dtype=float) > tol)),
        tuple(np.round(np.asarray(d["row"], dtype=float), 9)),
        tuple(np.round(np.asarray(d["col"], dtype=float), 9)),
    ))
    return found


def _solve_support(A: np.ndarray, B: np.ndarray, S_r: List[int], S_c: List[int], tol: float):
    """在给定支撑对上求候选均衡 (p, q)；无解返回 None。

    分三种情形：
    - ``|S_r| == |S_c|``：双方"无差异"线性系统恰好是方阵，用最小二乘解一次即可；
    - ``|S_r| > |S_c|``：行玩家无差异方程数（k_r）多于未知数（k_c+1），改为解
      "列玩家无差异 + q 非负"的**可行性 LP**；
    - ``|S_r| < |S_c|``：对称地解"行玩家无差异 + p 非负"的可行性 LP。

    参数:
        A: (m, n) 行玩家收益矩阵。
        B: (m, n) 列玩家收益矩阵。
        S_r: 行支撑（升序下标列表）。
        S_c: 列支撑（升序下标列表）。
        tol: 概率非负判定的容差。

    返回:
        ``(p, q)`` 归一化策略对，或 None。

    算法:
        方阵情形直接最小二乘；非方阵情形把"混合策略让对手在支撑上无差异"写成
        ``A[S_r][:, S_c] @ q - u * 1 = 0``、``sum(q) = 1``、``q >= 0``，
        用 :func:`optimization.simplex_lp` 找可行点（零目标），``u`` 用紧界
        ``[min(A), max(A)]`` 括住以保持问题有界。方阵情形若因退化解不出，会退回到 LP。

    复杂度:
        时间 = 一次最小二乘或一次 LP / 空间 O(|S_r| * |S_c|)。

    陷阱:
        1. **非方阵支撑对在一般（非退化）博弈里不存在均衡**，只有退化博弈才需要它；
           这一分支的用途是保证枚举的完备性，代价是要多解一次 LP，慢且数值更敏感。
        2. LP 只保证"找到一个可行点"，不保证唯一。退化博弈中同一支撑对可能对应一个
           均衡多面体（无穷多均衡），此时返回的只是其中一个代表点，返回集合并非"全部均衡"
           —— 这一点必须在论文里说清楚，标准支撑集枚举同样有这个限制。
        3. 支撑对必须按升序传入；顺序只影响内部索引映射，不影响正确性。

    参考:
        von Stengel 2002（支撑集枚举）；可行性判定用 LP 的做法见 Nisan et al.
        《Algorithmic Game Theory》第 3 章。
    """
    m, n = A.shape
    k_r, k_c = len(S_r), len(S_c)
    p = np.zeros(m)
    q = np.zeros(n)

    if k_r == k_c:
        # 未知量 = (q_{S_c}, u, p_{S_r}, w)，方程数 = k_r + k_c + 2 = 未知量数
        n_unk = k_c + 1 + k_r + 1
        mat = np.zeros((k_c + k_r + 2, n_unk))
        rhs = np.zeros(k_c + k_r + 2)
        for row_i, i in enumerate(S_r):
            for col_j, j in enumerate(S_c):
                mat[row_i, col_j] = A[i, j]
            mat[row_i, k_c] = -1.0
        for row_j, j in enumerate(S_c):
            for col_i, i in enumerate(S_r):
                mat[k_r + row_j, k_c + 1 + col_i] = B[i, j]
            mat[k_r + row_j, k_c + 1 + k_r] = -1.0
        mat[k_c + k_r, :k_c] = 1.0
        rhs[k_c + k_r] = 1.0
        mat[k_c + k_r + 1, k_c + 1:k_c + 1 + k_r] = 1.0
        rhs[k_c + k_r + 1] = 1.0

        sol = _lsq(mat, rhs)
        if sol is not None:
            qv = sol[:k_c]
            pv = sol[k_c + 1:k_c + 1 + k_r]
            if np.all(qv >= -tol) and np.all(pv >= -tol):
                q[S_c] = np.clip(qv, 0.0, None)
                p[S_r] = np.clip(pv, 0.0, None)
                if q.sum() > tol and p.sum() > tol:
                    return p / p.sum(), q / q.sum()
        # 退化：方阵奇异时退回 LP
        qq = _feasible_q(A, S_r, S_c)
        if qq is None:
            return None
        pp = _feasible_p(B, S_r, S_c)
        if pp is None:
            return None
        return pp, qq

    if k_r > k_c:
        # 未知数少于方程：行玩家的无差异条件一般无法全部满足，改求列玩家无差异的 q
        qq = _feasible_q(A, S_r, S_c)
        if qq is None:
            return None
        pp = _feasible_p(B, S_r, S_c)
        if pp is None:
            return None
        return pp, qq

    # k_r < k_c：对称处理
    pp = _feasible_p(B, S_r, S_c)
    if pp is None:
        return None
    qq = _feasible_q(A, S_r, S_c)
    if qq is None:
        return None
    return pp, qq


def _feasible_q(A: np.ndarray, S_r: List[int], S_c: List[int]) -> Optional[np.ndarray]:
    """求让行玩家在 S_r 上无差异的列混合策略 q（q 支撑于 S_c，非负、和为 1）。

    参数:
        A: 行玩家收益矩阵。
        S_r: 行支撑下标。
        S_c: 列支撑下标。

    返回:
        长度 n 的 np.ndarray（支撑外为 0），或 None（不可行）。

    算法:
        变量 ``[q_j (j in S_c), u]``，约束
        ``sum_j A[i, j] q_j - u = 0  (i in S_r)``、``sum_j q_j = 1``、``q_j >= 0``，
        ``u`` 用 ``[min(A), max(A)]`` 括住（期望收益必落在此区间，保证 LP 有界）。
        目标函数取零，只判可行性。

    复杂度:
        时间 = 一次 LP / 空间 O(|S_r| * |S_c|)。

    陷阱:
        用"零目标 + 可行性"比"解方程"稳健得多，但**单纯形求解器对退化问题可能返回
        ``"optimal"`` 却给出贴边界的解**；之后的全局最优性回代检验会兜住这种情况，
        所以调用方拿到 (p, q) 后必须再跑 :func:`_is_nash`。

    参考:
        支撑集枚举中的可行性判定（von Stengel 2002）。
    """
    k_r, k_c = len(S_r), len(S_c)
    n_var = k_c + 1
    c = np.zeros(n_var)
    A_eq = np.zeros((k_r + 1, n_var))
    b_eq = np.zeros(k_r + 1)
    for row_i, i in enumerate(S_r):
        for col_j, j in enumerate(S_c):
            A_eq[row_i, col_j] = A[i, j]
        A_eq[row_i, k_c] = -1.0
    A_eq[k_r, :k_c] = 1.0
    b_eq[k_r] = 1.0
    lo_u = float(A.min())
    hi_u = float(A.max())
    bounds = [(0.0, 1.0)] * k_c + [(lo_u, hi_u)]
    res = simplex_lp(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, maximize=False)
    if str(res.get("status")) != "optimal":
        return None
    x = np.asarray(res["x"], dtype=float)
    qv = np.clip(x[:k_c], 0.0, None)
    if qv.sum() <= 1e-9:
        return None
    qv = qv / qv.sum()
    val = A[np.ix_(S_r, S_c)] @ qv
    if float(val.max() - val.min()) > 1e-6:
        return None
    q = np.zeros(A.shape[1])
    q[S_c] = qv
    return q


def _feasible_p(B: np.ndarray, S_r: List[int], S_c: List[int]) -> Optional[np.ndarray]:
    """求让列玩家在 S_c 上无差异的行混合策略 p（p 支撑于 S_r，非负、和为 1）。

    参数:
        B: 列玩家收益矩阵。
        S_r: 行支撑下标。
        S_c: 列支撑下标。

    返回:
        长度 m 的 np.ndarray（支撑外为 0），或 None（不可行）。

    算法:
        为避免引入可能取负的无界变量 w，先用系数差分消去它：
        对 ``t = 1..k_c-1`` 要求 ``(B^T p)[S_c[0]] - (B^T p)[S_c[t]] = 0``，
        等价于 ``sum_i (B[i, S_c[0]] - B[i, S_c[t]]) p_i = 0``；
        再加上 ``sum p = 1``、``p >= 0``，用零目标 LP 判可行。

    复杂度:
        时间 = 一次 LP / 空间 O(|S_r| * |S_c|)。

    陷阱:
        差分写法要求 ``k_c >= 1``；``k_c == 1`` 时只剩归一化约束，此时任何 p 都让列玩家
        "在单点支撑上无差异"（平凡成立），真正的约束来自回代检验（p 必须使该列成为
        最优响应）。不要以为 LP 可行就等于找到了均衡。

    参考:
        同 :func:`_feasible_q`。
    """
    k_r, k_c = len(S_r), len(S_c)
    n_var = k_r
    c = np.zeros(n_var)
    n_eq = (k_c - 1) + 1
    A_eq = np.zeros((n_eq, n_var))
    b_eq = np.zeros(n_eq)
    base_j = S_c[0]
    for t in range(1, k_c):
        j = S_c[t]
        for col_i, i in enumerate(S_r):
            A_eq[t - 1, col_i] = B[i, base_j] - B[i, j]
    A_eq[n_eq - 1, :] = 1.0
    b_eq[n_eq - 1] = 1.0
    bounds = [(0.0, 1.0)] * n_var
    res = simplex_lp(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, maximize=False)
    if str(res.get("status")) != "optimal":
        return None
    x = np.asarray(res["x"], dtype=float)
    pv = np.clip(x[:k_r], 0.0, None)
    if pv.sum() <= 1e-9:
        return None
    pv = pv / pv.sum()
    val = pv @ B[np.ix_(S_r, S_c)]
    if float(val.max() - val.min()) > 1e-6:
        return None
    p = np.zeros(B.shape[0])
    p[S_r] = pv
    return p


def _lsq(mat: np.ndarray, rhs: np.ndarray) -> Optional[np.ndarray]:
    """最小二乘解线性系统；欠定或残差过大返回 None。

    参数:
        mat: 系数矩阵。
        rhs: 右端项。

    返回:
        解向量，或 None（行数少于列数、秩亏、残差超限）。

    算法:
        ``np.linalg.lstsq``（SVD 最小二乘），再检查秩与残差。

    复杂度:
        时间 O(k^3)（k 为矩阵规模）/ 空间 O(k^2)。

    陷阱:
        秩亏时直接返回 None 是**有意为之**：退化系统有无穷多解，随便取一个会让"均衡
        去重"和"全局最优性检验"都失去意义。退化情形交给 LP 分支处理。

    参考:
        Golub & Van Loan《Matrix Computations》。
    """
    if mat.shape[0] < mat.shape[1]:
        return None
    sol, residuals, rank, _ = np.linalg.lstsq(mat, rhs, rcond=None)
    if rank < mat.shape[1]:
        # 欠定：均衡可能仍存在（退化），但本方法无法唯一确定，交给 LP 分支去覆盖
        return None
    if float(np.abs(mat @ sol - rhs).max()) > 1e-7:
        return None
    return sol


def _is_nash(A, B, p, q, S_r, S_c, tol: float) -> bool:
    """回代检验全局最优性：p 的支撑必须恰好覆盖行玩家的最优纯策略，q 同理。

    参数:
        A, B: 行/列玩家收益矩阵。
        p, q: 候选混合策略。
        S_r, S_c: 候选支撑。
        tol: 判等容差。

    返回:
        bool，True 表示 ``(p, q)`` 是一个纳什均衡。

    算法:
        1. 检查概率分布合法性（和为 1、非负）；
        2. 行玩家：``A @ q`` 在 S_r 上取最小值不能低于全局最大值太多（支撑内部无差异）；
        3. 列玩家：``p @ B`` 在 S_c 上同理；
        4. **支撑外**：任何概率为正的策略都必须是（近似）最优响应，避免把"多带了次优策略"
           的伪均衡放进来。

    复杂度:
        时间 O(mn) / 空间 O(m+n)。

    陷阱:
        第 4 步容易被漏掉。只检查"支撑内部无差异"是不够的：支撑内可能整体都是**次优**响应
        （例如行玩家把概率分给两个都被严格占优的策略），此时仍满足"无差异"但不是均衡。

    参考:
        纳什均衡的定义（Nash 1950）。
    """
    if abs(float(p.sum()) - 1.0) > 1e-6 or abs(float(q.sum()) - 1.0) > 1e-6:
        return False
    if np.any(p < -tol) or np.any(q < -tol):
        return False
    row_gain = A @ q
    col_gain = p @ B
    if row_gain.max() - float(row_gain[S_r].min()) > 1e-6:
        return False
    if col_gain.max() - float(col_gain[S_c].min()) > 1e-6:
        return False
    # 支撑之内必须都是最优响应（不允许"多带"了次优的纯策略）
    for i in np.nonzero(p > tol)[0]:
        if row_gain.max() - row_gain[i] > tol:
            return False
    for j in np.nonzero(q > tol)[0]:
        if col_gain.max() - col_gain[j] > tol:
            return False
    return True


def _append_unique(found: List[Dict[str, object]], p: np.ndarray, q: np.ndarray, tol: float) -> None:
    """按容差去重后追加一个均衡。"""
    for e in found:
        if (np.abs(np.asarray(e["row"], dtype=float) - p).max() <= 1e-6
                and np.abs(np.asarray(e["col"], dtype=float) - q).max() <= 1e-6):
            return
    found.append({"row": p.copy(), "col": q.copy()})


# --------------------------------------------------------------------------- #
# Shapley 值
# --------------------------------------------------------------------------- #
def shapley_value(characteristic, n: int) -> np.ndarray:
    """精确计算 n 人合作博弈的 Shapley 值（枚举所有排列 / 所有联盟）。

    参数:
        characteristic: 特征函数 v。**本实现接受两种口径，二者等价、可混用同一种写法**：
            (a) 位掩码整数：第 i 个玩家对应第 i 位（``player i <-> (mask >> i) & 1``），
                例如 ``v(3) = v({0, 1})``；
            (b) ``frozenset``：``v(frozenset({0, 1}))``。
            可以传 dict（查表）或可调用对象（函数）。**约定 ``v(空集) = 0``**：
            若 dict 里查不到空集，按 0 处理；若函数返回非 0，本函数不报错但结果会失真，
            请自行保证 v(∅) = 0。
            玩家编号固定为 ``0..n-1``。
        n: 玩家数，必须 >= 1。

    返回:
        np.ndarray，长度 n，``result[i]`` 是玩家 i 的 Shapley 值。
        满足有效性 ``sum(result) == v(全集)``（数值误差在 1e-9 量级）。

    算法:
        ``phi_i = sum_{S ⊆ N\\{i}} |S|! (n-|S|-1)! / n! * [v(S ∪ {i}) - v(S)]``。
        实现上直接枚举 ``2^n`` 个子集，权重的阶乘比值由辅助函数 :func:`_shapley_weight`
        用 ``math.factorial`` 算出（``factorial(|S|) * factorial(n-|S|-1) / factorial(n)``，
        本模块只导入了 ``factorial``，**没有**用 ``math.comb``），避免生成 n! 个排列。

    复杂度:
        时间 O(n * 2^n)（枚举子集 O(2^n)，每个子集对每个玩家取边际贡献）/ 空间 O(2^n)（缓存）。

    陷阱:
        1. **空联盟必须为 0**。v(∅) ≠ 0 时"边际贡献"的定义被破坏，算出来的值不再满足
           有效性公理；很多公开代码在这里静默出错。本实现把空集强制视作 0 并忽略传入值。
        2. **特征函数只依赖联盟、不依赖顺序**。如果你把"先来后到"写进 v 里，
           Shapley 值就不再适用，应改用带权重的广义 Shapley 或 Shapley-Shubik 指数。
        3. 复杂度是 ``n * 2^n``：n=20 时约 2e7 次查表尚可，n=25 时 8e8 就太慢；
           n > 20 建议用蒙特卡洛抽样（随机采排列，用样本均值近似），并报告标准误。
        4. 特征函数的**量纲和量级**会直接进入结果；若 v 表示"成本节省"，结果单位也是成本，
           不要在论文里写成"收益份额"。
        5. 返回的 Shapley 值**可以为负**（某玩家是"反贡献者"），不要想当然认为是份额比例。
           尤其注意：Shapley 值可能为负而"按比例分配"永远非负，两者结论会打架。
        6. 若特征函数是超可加的，Shapley 值位于核（core）内；若只是可加性以外的任意函数，
           Shapley 值可能落在核外，此时"稳定分配"的论断不成立。

    参考:
        Shapley 1953, "A Value for n-Person Games"；Shapley & Shubik 1954（投票权力指数）；
        课本口径见 Osborne & Rubinstein《A Course in Game Theory》第 14 章。
    """
    n = int(n)
    if n < 1:
        raise ValueError("n 必须为正整数")
    full = (1 << n) - 1

    cache: Dict[int, float] = {}

    def v(mask: int) -> float:
        """统一把位掩码口径转成 float，并缓存。"""
        if mask == 0:
            return 0.0  # 强制 v(∅) = 0，见"陷阱"第 1 条
        if mask in cache:
            return cache[mask]
        key_set = frozenset(i for i in range(n) if (mask >> i) & 1)
        if callable(characteristic):
            try:
                val = float(characteristic(key_set))
            except TypeError:
                val = float(characteristic(mask))
        else:
            try:
                val = characteristic.get(key_set, characteristic.get(mask))
            except AttributeError:
                raise ValueError("characteristic 必须是 dict 或可调用对象")
            if val is None:
                # dict 缺键：明确报错，而不是静默当 0（静默当 0 会给出看似合理但错误的结果）
                raise ValueError(
                    f"特征函数缺少联盟 {sorted(key_set)}（或掩码 {mask}）的取值；"
                    "请补全所有 2^n 个联盟，或改用可调用对象"
                )
            val = float(val)
        if not np.isfinite(val):
            raise ValueError(f"特征函数在联盟 {sorted(key_set)} 上返回非有限值 {val}")
        cache[mask] = val
        return val

    # 先把全集和各子集全部算一遍，fail fast
    for mask in range(1, full + 1):
        v(mask)
    total = v(full)

    phi = np.zeros(n)
    for mask in range(full):  # mask == full 时没有"缺席玩家"，边际贡献恒为 0，可跳过
        size = bin(mask).count("1")
        weight = _shapley_weight(size, n)
        base = v(mask)
        for i in range(n):
            if (mask >> i) & 1:
                continue
            phi[i] += weight * (v(mask | (1 << i)) - base)

    sqrt_n = float(np.sqrt(n))
    # 浮点累积误差按 n 缩放的容差检查有效性；这里只在明显失真时报警
    if abs(float(phi.sum()) - total) > 1e-6 * max(1.0, abs(total)) * sqrt_n:
        raise ValueError(
            f"Shapley 值不满足有效性：sum={phi.sum()} 而 v(全集)={total}"
        )
    return phi


def _shapley_weight(size: int, n: int) -> float:
    """联盟规模为 size 时的 Shapley 权重 |S|!(n-|S|-1)!/n!。

    参数:
        size: 联盟规模 |S|，必须满足 ``0 <= size <= n - 1``。
        n: 玩家数。

    返回:
        float 权重，所有规模的权重之和为 n（对每个玩家求和为 1）。

    算法:
        直接按定义算阶乘比值。

    复杂度:
        时间 O(n)（阶乘计算）/ 空间 O(1)。

    陷阱:
        ``size == n`` 时 ``n - size - 1 == -1``，阶乘未定义。调用方必须跳过全集
        （全集中没有"缺席玩家"，边际贡献恒为 0）——这是 Shapley 公式最容易写错的下标之一。

    参考:
        Shapley 1953。
    """
    if not (0 <= size <= n - 1):
        raise ValueError(f"_shapley_weight 要求 0 <= size <= n-1，得到 size={size}, n={n}")
    return factorial(size) * factorial(n - size - 1) / factorial(n)


# --------------------------------------------------------------------------- #
# Gale-Shapley 稳定匹配
# --------------------------------------------------------------------------- #
def gale_shapley(men_prefs, women_prefs) -> dict:
    """Gale-Shapley 延迟接受算法（"男方求婚"版本），求稳定匹配。

    参数:
        men_prefs: ``{man: [w1, w2, ...]}``，每个男生的偏好降序列表（最喜欢在前）。
        women_prefs: ``{woman: [m1, m2, ...]}``，每个女生的偏好降序列表。
            两边的偏好列表必须**互相是对方的全集**（同一组男女，只是顺序不同）。若某方列表里
            出现了**不在对方字典键集合中**的人，该条目会在预处理时被**静默丢弃**（代码里按
            ``w in women_set`` / ``man in men_set`` 过滤），即被视为"不可接受"，既不会报警告，
            也不会被记入任何 ``rank`` 检查，更不会出现在返回值里（返回值**没有** ``rank`` 键）。
            偏好列表内部的重复项同样被静默去重（保留首次出现的位置）。
            **不要求**人数相等：多出来的一方按偏好列表长度自然处理。

    返回:
        ``{"matching": {...}, "matches": {...}, "n_proposals": int, "n_blocking_pairs": int}``，
        共 **4** 个键：

        - ``matching``：男 → 女 的字典，**包含所有男生**（未匹配到则为 None）；
        - ``matches``：反向字典 女 → 男，**只包含实际配对的女生**（单身女生不出现，
          且值不会是 None）——注意它不是 ``matching`` 的逐键镜像，想查"某女生是否匹配"要用
          ``woman in matches`` 而不是 ``matches[woman]``（缺键会 KeyError）；
        - ``n_proposals``：算法过程中**累计的求婚次数**（同一个男生被拒后会再次求婚，
          每次计一次），这是衡量算法代价的常用指标；
        - ``n_blocking_pairs``：返回前**实际校验**得到的阻塞对个数（正确实现应为 0）。

        （返回值里**没有** ``rank`` 键；内部的 ``rank_m`` / ``rank_w`` 只在函数内部用于比较
        和阻塞对校验，不对外暴露。）

    算法:
        男方求婚版延迟接受：
        1. 每个未匹配的男生按自己的偏好顺序依次向女生求婚；
        2. 女生若当前无配偶则暂时接受；若有配偶，则比较新来的与现任，保留更喜欢的那个，
           另一位恢复为未匹配（他继续向下一个目标求婚）；
        3. 直到所有男生都有配偶，或所有男生把所有女生都求过一遍。

    复杂度:
        时间 O(n * m)（每个男生最多求婚 m 次，每次 O(1) 比较）/
        空间 O(n + m)（偏好字典 + 匹配表）。

    陷阱:
        1. **得到的稳定匹配是谁最优的？** 男方求婚版给出**男方最优**稳定匹配
           （每个男生在所有稳定匹配中拿到自己最满意的结果），同时是**女方最差**的稳定匹配。
           论文里若说"这就是最优匹配"必须注明是从哪一方的口径；换成女方求婚版会得到另一个
           （可能不同的）稳定匹配。
        2. **稳定匹配不唯一**，但稳定匹配集合的"格结构"保证了男方最优/女方最优两个极端存在。
        3. **必须校验无阻塞对**。很多实现只跑流程不校验，结果偏好列表方向搞反（比如把
           "最喜欢在前"写成"最不喜欢在前"）时依然能输出一个匹配，只是不稳定。
           本函数在返回前做 O(n*m) 全对校验，报告 ``n_blocking_pairs``。
        4. 偏好列表里若出现未在对方字典中的人，会被当作"不在候选集"**静默丢弃**（视为不可接受），
           不报警告、不计入任何排名或校验；若两边的人名不是同一套字符串，匹配会大面积落空——
           请保证键集合一致（想自查可直接对两边的 ``keys()`` 取差集）。
           返回值里**没有** ``rank`` 键，无法从结果中看出谁被丢掉了。
        5. 偏好列表**可以是不完全列表**（有的男生只列出部分女生）：此时算法把未列出的
           视为不可接受，本实现用"不向列表外的人求婚"处理，这符合标准的
           "incomplete preferences"扩展。若你希望未列出的视为最次而非不可接受，
           请显式补全列表。
        6. 男生字典为空时返回空匹配，不报错（边界情形在竞赛里很常见）。

    参考:
        Gale & Shapley 1962, "College Admissions and the Stability of Marriage"；
        Roth & Sotomayor 1990（匹配市场设计）。
    """
    if not isinstance(men_prefs, Mapping) or not isinstance(women_prefs, Mapping):
        raise ValueError("men_prefs / women_prefs 必须是字典 {name: [偏好列表]}")

    men = list(men_prefs.keys())
    women = list(women_prefs.keys())
    men_set = set(men)
    women_set = set(women)

    # 偏好校验：去重 + 只保留对方集合内的候选（不可接受的直接剔除）
    prefs_m: Dict[Hashable, List[Hashable]] = {}
    for man in men:
        seen: List[Hashable] = []
        for w in men_prefs[man]:
            if w in women_set and w not in seen:
                seen.append(w)
        prefs_m[man] = seen
    prefs_w: Dict[Hashable, List[Hashable]] = {}
    for woman in women:
        seen_m: List[Hashable] = []
        for man in women_prefs[woman]:
            if man in men_set and man not in seen_m:
                seen_m.append(man)
        prefs_w[woman] = seen_m

    # 女方对男方的排名（越小越喜欢）；未列出的视为不可接受
    rank_w: Dict[Hashable, Dict[Hashable, int]] = {
        w: {man: i for i, man in enumerate(prefs_w[w])} for w in women
    }
    # 男方对女方的排名，用于阻塞对校验：关键是 **prefs_m 里排在前面的必须排名更小**
    rank_m: Dict[Hashable, Dict[Hashable, int]] = {
        man: {w: i for i, w in enumerate(prefs_m[man])} for man in men
    }

    next_choice = {man: 0 for man in men}  # 每个男生下一个要表白的下标
    match_w: Dict[Hashable, Hashable] = {}
    match_m: Dict[Hashable, Optional[Hashable]] = {man: None for man in men}
    n_proposals = 0

    free = [man for man in men]
    while free:
        man = free.pop(0)
        lst = prefs_m[man]
        if next_choice[man] >= len(lst):
            continue  # 表白名单用尽，保持单身
        woman = lst[next_choice[man]]
        next_choice[man] += 1
        n_proposals += 1

        current = match_w.get(woman)
        if current is None:
            match_w[woman] = man
            match_m[man] = woman
            continue
        if rank_w[woman].get(man, 10 ** 9) < rank_w[woman].get(current, 10 ** 9):
            # 女方更喜欢新来的：踢掉现任
            match_w[woman] = man
            match_m[man] = woman
            match_m[current] = None
            free.append(current)
        else:
            free.append(man)  # 被拒，继续找下一个

    # 校验无阻塞对：O(n*m)。阻塞对定义：双方都严格更喜欢对方（含单身视为"无限靠后"）。
    n_blocking = 0
    for man in men:
        cur_w = match_m[man]
        cur_rank = rank_m[man].get(cur_w, 10 ** 9) if cur_w is not None else 10 ** 9
        for woman in prefs_m[man]:
            if woman == cur_w:
                continue
            if rank_m[man][woman] >= cur_rank:
                continue  # 他并不更喜欢这个女生
            her = match_w.get(woman)
            her_rank = rank_w[woman].get(man, 10 ** 9) if her is not None else 10 ** 9
            her_cur = rank_w[woman].get(her, 10 ** 9) if her is not None else None
            if her is None or her_rank < her_cur:
                n_blocking += 1

    return {
        "matching": match_m,
        "matches": {w: mn for w, mn in match_w.items()},
        "n_proposals": int(n_proposals),
        "n_blocking_pairs": int(n_blocking),
    }


# --------------------------------------------------------------------------- #
# 复制者动态
# --------------------------------------------------------------------------- #
def replicator_dynamics(
    A, x0, T: int, dt: float = 0.01
) -> dict:
    """复制者动态（replicator dynamics）演化博弈轨迹，RK4 数值积分。

    参数:
        A: (n, n) 收益矩阵。**约定 ``A[i, j]`` 是"我采用策略 i、对手采用策略 j"时我的收益**，
           因此总体平均收益是 ``x^T A x``（对手也是同一群体，为对称博弈的"对群体平均"口径）。
        x0: 长度 n 的初始策略分布，必须非负、和 > 0（内部会归一化）。
        T: 总步数（整数）。返回轨迹长度 = T + 1（含初始点）。
        dt: 时间步长，默认 0.01。RK4 对它是四阶精度，但**dt 过大仍会震荡或越界**。

    返回:
        ``{"t": np.ndarray(T+1), "trajectory": np.ndarray(T+1, n),
        "final": np.ndarray(n), "converged": bool}``。
        轨迹每行都是概率分布（非负、和为 1）；``converged`` 表示最后两步的
        L1 变化小于 1e-8（经验判据，不等于严格证明收敛到 ESS）。

    算法:
        ``dx_i/dt = x_i * [(A x)_i - x^T A x]``。用经典四阶 Runge-Kutta（RK4）积分，
        每步之后把负的数值毛刺截断为 0 并重新归一化，保证迭代始终停留在单纯形上。

    复杂度:
        时间 O(T * n^2)（每步 4 次矩阵-向量乘）/ 空间 O(T * n)。

    陷阱:
        1. **dt 不能随手放大**。复制者动态的向量场在边界附近很陡，``dt`` 过大（例如 0.1）
           会让解冲出单纯形（出现负概率），此时"归一化"只是在掩盖积分误差，轨迹形状会失真。
           本实现默认 0.01 并显式截断负值，但论文里仍应报告 dt 并做一步 dt/2 的收敛性对照。
        2. **复制者动态的解不一定是 ESS**。它只保证"纯策略的适应度高于平均值的策略占比上升"。
           内部稳定点（如混合均衡）可能是鞍点或中心：石头剪刀布的内点均衡就是**中性稳定**
           （围绕它振荡，不发散也不收敛），此时 tracks 出来的轨迹取决于初值和 dt，
           不能说"收敛到均衡"。请务必画相图/看 ``converged``。
        3. **初始点落在边界上且该纯策略被严格占优**时，演化会停在边界（占比 0 无法回升），
           这是"不可逆"的：被淘汰的策略永远不会重新出现（除非加入变异/漂移项）。
        4. ``x^T A x`` 是**对称博弈**（同一群体）的口径。若是两个不同种群的非对称博弈，
           正确写法是双群体版本 ``dx_i/dt = x_i((A y)_i - x^T A y)``，本函数不适用。
        5. 收益矩阵整体加常数**不改变**复制者动态（因为减去平均值时被抵消），
           但乘以正数会改变时间尺度；调参时注意这一点。

    参考:
        Taylor & Jonker 1978（复制者动态的原始定义）；Hofbauer & Sigmund 1998
        《Evolutionary Games and Population Dynamics》。
    """
    Amat = as_matrix(A, "A")
    check_square(Amat, "A")
    n = Amat.shape[0]
    x = as_vector(x0, "x0")
    if x.size != n:
        raise ValueError(f"x0 长度 {x.size} 与 A 的规模 {n} 不一致")
    if np.any(x < 0):
        raise ValueError("x0 必须非负（它是策略占比）")
    if x.sum() <= 0:
        raise ValueError("x0 的和必须为正")
    x = x / x.sum()

    T = int(T)
    dt = float(dt)
    if T < 0:
        raise ValueError("T 不能为负")
    if dt <= 0:
        raise ValueError("dt 必须为正")

    def field(y: np.ndarray) -> np.ndarray:
        """复制者动态向量场。"""
        fitness = Amat @ y
        mean = float(y @ fitness)
        return y * (fitness - mean)

    traj = np.empty((T + 1, n))
    traj[0] = x
    t = np.zeros(T + 1)
    for k in range(T):
        k1 = field(x)
        k2 = field(_simplex(x + 0.5 * dt * k1))
        k3 = field(_simplex(x + 0.5 * dt * k2))
        k4 = field(_simplex(x + dt * k3))
        xn = _simplex(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        x = xn
        traj[k + 1] = x
        t[k + 1] = t[k] + dt

    converged = bool(T >= 1 and np.abs(traj[-1] - traj[-2]).sum() < 1e-8)
    return {
        "t": t,
        "trajectory": traj,
        "final": traj[-1].copy(),
        "converged": converged,
    }


def _simplex(y: np.ndarray) -> np.ndarray:
    """把向量投影回概率单纯形的一个实用近似：截负 + 归一化。"""
    z = np.clip(y, 0.0, None)
    s = z.sum()
    if s <= 0:
        return np.full(y.size, 1.0 / y.size)
    return z / s


# --------------------------------------------------------------------------- #
# α-β 剪枝极大极小搜索
# --------------------------------------------------------------------------- #
def minimax_alpha_beta(
    node: object,
    depth: int,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    maximizing: bool = True,
    evaluate_fn: Optional[Callable[[object], float]] = None,
    children_fn: Optional[Callable[[object], Sequence[object]]] = None,
) -> dict:
    """通用 α-β 剪枝极小极大搜索（alpha-beta pruning）。

    参数:
        node: 搜索起点。**本函数不解释它的内部结构**，只是把它原样交给 ``children_fn`` /
            ``evaluate_fn``；可以是棋盘、状态元组、字典键等任意对象。
        depth: 还要往下搜索的层数，必须 >= 0。``depth == 0`` 时直接调用 ``evaluate_fn``
            （不再展开子节点），所以"完全搜索"要求 depth 不小于博弈树的深度。
        alpha: 最大化者已经能保证的下界；**根节点必须传 -inf**（默认值）。
        beta: 最小化者已经能保证的上界；**根节点必须传 +inf**（默认值）。
        maximizing: ``node`` 处轮到谁走；True 表示轮到最大化者。
        evaluate_fn: ``evaluate_fn(node) -> float``，叶子（或 depth 用尽）处的静态评估值，
            **越大对最大化者越有利**。必须能对任意被展开到的节点求值。
        children_fn: ``children_fn(node) -> 子节点序列``；返回空序列表示终局。

    返回:
        dict，键为：
        ``value``   float，搜索得到的极小极大值（根节点处，maximizing 方的保证值）；
        ``best_child``   maximizing 方在 ``node`` 处的最优子节点对象，叶子处为 None
            （注意：**是子节点本身，不是下标**；若 node 处轮到最小化者，返回的是让最大化者
            收益最小的那个子节点）；
        ``n_evaluated``   int，``evaluate_fn`` 的调用次数（= 展开的叶子数）；
        ``n_pruned``   int，发生剪枝（提前 ``break``）的次数。

    算法:
        带 (alpha, beta) 窗口的递归极小极大：
        1. ``depth == 0`` 或 ``children_fn(node)`` 为空 → ``evaluate_fn(node)``，叶子计数 +1；
        2. 最大化节点：依次递归每个子节点，记录最大值与对应子节点，更新
           ``alpha = max(alpha, value)``；一旦 ``alpha >= beta`` 就停止遍历剩余子节点（β 剪枝）；
        3. 最小化节点：对称地更新 ``beta = min(beta, value)``，一旦 ``beta <= alpha`` 停止（α 剪枝）。
        剪掉的子树不可能改变根节点的取值（Knuth & Moore 1975），因此根节点在全窗口下拿到的
        值与不剪枝的完全极小极大**逐位相同**。

    复杂度:
        时间：最坏（子节点顺序最差）O(b^d)，与不剪枝的极小极大相同；子节点顺序理想时
        O(b^(d/2))（b 为平均分支因子、d 为深度）。空间 O(d)（递归栈，不含 children 列表）。

    陷阱:
        1. ``n_pruned`` 是**剪枝事件次数**，不是"省下的求值次数"。被剪掉的子树只有真的展开才知道多大，
           所以报告"剪枝节省了多少节点"时必须用 ``n_evaluated`` 与不剪枝实现对比，而不是看 ``n_pruned``。
        2. α-β 只在**叶子评估精确**时才与完全极小极大给出相同的值。若 ``depth`` 提前截断、
           ``evaluate_fn`` 给出的是启发式估值，两条路径的值就会不同——那是深度截断的误差，不是剪枝的错。
        3. 剪枝判定必须用 ``alpha >= beta``：写成严格大于只损失剪枝率（结果仍对），
           但若在**更新窗口之前**就 break，则可能剪掉真正更优的分支，得到错值。
        4. ``children_fn`` 的返回顺序决定剪枝率。同一棵树换个顺序，``value`` 不变但
           ``n_evaluated`` 可能差一个量级；为了让结果可复现，``children_fn`` 本身必须确定性
           （不要在里面用随机顺序或依赖 dict 的遍历顺序）。
        5. ``best_child`` 返回的是子节点对象的**引用**；调用方就地修改它会影响你自己的树结构。
           另外，在递归内部（非根节点）返回的 best_child 是"窗口内最优"，只有根节点全窗口调用时
           才保证是真正的最优着法。
        6. 递归深度等于 ``depth``；把它设成上百万会让 Python 直接抛 RecursionError（默认上限 1000），
           深树请自己改写成显式栈版本。

    参考:
        Knuth & Moore 1975, "An analysis of alpha-beta pruning", Artificial Intelligence 6(4)；
        Russell & Norvig《人工智能：一种现代方法》第 5 章（对抗搜索）。
    """
    if evaluate_fn is None or children_fn is None:
        raise ValueError("必须同时提供 evaluate_fn 与 children_fn")
    depth = int(depth)
    if depth < 0:
        raise ValueError(f"depth 必须 >= 0，得到 {depth}")
    a_in = float(alpha)
    b_in = float(beta)
    if not (a_in < b_in):
        raise ValueError(f"空的搜索窗口：alpha={a_in} 不小于 beta={b_in}")

    stats = {"n_evaluated": 0, "n_pruned": 0}

    def rec(cur: object, d: int, a: float, b: float, is_max: bool):
        """返回 (窗口内的最优值, 对应的子节点)；叶子返回 (评估值, None)。"""
        kids: List[object] = list(children_fn(cur)) if d > 0 else []
        if d == 0 or not kids:
            stats["n_evaluated"] += 1
            return float(evaluate_fn(cur)), None
        best_child = None
        if is_max:
            best = float("-inf")
            for kid in kids:
                v, _ = rec(kid, d - 1, a, b, False)
                if v > best:
                    best, best_child = v, kid
                if best > a:
                    a = best
                if a >= b:
                    stats["n_pruned"] += 1
                    break
        else:
            best = float("inf")
            for kid in kids:
                v, _ = rec(kid, d - 1, a, b, True)
                if v < best:
                    best, best_child = v, kid
                if best < b:
                    b = best
                if a >= b:
                    stats["n_pruned"] += 1
                    break
        return best, best_child

    value, best_child = rec(node, depth, a_in, b_in, bool(maximizing))
    return {
        "value": float(value),
        "best_child": best_child,
        "n_evaluated": int(stats["n_evaluated"]),
        "n_pruned": int(stats["n_pruned"]),
    }


# --------------------------------------------------------------------------- #
# Stackelberg（领导者-跟随者）线性博弈
# --------------------------------------------------------------------------- #
def stackelberg_lp(
    A_leader: MatrixLike,
    c_leader: ArrayLike,
    A_follower: MatrixLike,
    c_follower: ArrayLike,
    b_follower: ArrayLike,
    x_upper: Optional[ArrayLike] = None,
    n_grid: int = 21,
    leader_cost: Optional[ArrayLike] = None,
) -> dict:
    """领导者-跟随者（Stackelberg）线性博弈：跟随者解 LP，领导者在响应函数上网格搜索。

    参数:
        A_leader: 领导者工具对跟随者资源的**耦合矩阵**，形状 ``(k, m)``（k = 跟随者约束条数、
            m = 领导者决策维数）。第 i 条跟随者约束的右端项是 ``b_follower[i] - (A_leader x)[i]``：
            **取正号表示领导者的决策在消耗/限制跟随者的资源**，取负号表示在放宽它。
        c_leader: 领导者对**跟随者决策 y** 的估值向量，长度 ``n``（n = 跟随者决策维数）。
            领导者收益 = ``c_leader @ y - leader_cost @ x``。
        A_follower: 跟随者自己的约束矩阵，形状 ``(k, n)``。
        c_follower: 跟随者的目标系数，长度 ``n``（跟随者最大化 ``c_follower @ y``）。
        b_follower: 跟随者约束右端项，长度 ``k``（也是领导者工具为 0 时的资源量）。
        x_upper: 领导者每个决策变量的上界，长度 ``m`` 的非负向量；None 表示自动取
            ``max(|b_follower|)``（与资源同量级，避免网格尺度荒谬）。**上界必须有界**，
            否则领导者的搜索问题是无穷区间，网格搜索没有意义。
        n_grid: 每个领导者变量方向上的网格点数（含两端），必须 >= 2。总 LP 次数 = ``n_grid ** m``。
        leader_cost: 领导者工具的单位成本向量，长度 ``m``；None 表示工具零成本。

    返回:
        dict，键为：
        ``x_leader``   形状 (m,) 的领导者决策（网格上最优的那个点）；
        ``y_follower``   形状 (n,) 跟随者对 ``x_leader`` 的最优响应（LP 解，已截负）；
        ``leader_payoff``   float，``c_leader @ y_follower - leader_cost @ x_leader``；
        ``follower_payoff``   float，``c_follower @ y_follower``。

    算法:
        1. 跟随者问题：给定 x，解
           ``max_y c_follower @ y  s.t.  A_follower y <= b_follower - A_leader x,  y >= 0``
           （调用 :func:`optimization.simplex_lp`）；
        2. 领导者在盒式区域 ``0 <= x <= x_upper`` 上取 ``n_grid^m`` 个网格点（``itertools.product``
           的字典序，确定性），对每个点求跟随者的最优响应与领导者收益；
        3. 取领导者收益最大的网格点，严格大于才替换（并列时保留字典序最小的那个，保证可复现）；
        4. 跟随者的 LP 不可行 / 无界（status 非 ``"optimal"``）时该领导者决策被丢弃，
           若所有网格点都被丢弃则抛 ValueError。

    复杂度:
        时间 O(n_grid^m * 一次 LP)；空间 O(k n + n_grid)（不存储全部网格结果，只留当前最好）。

    陷阱:
        1. **这是网格近似，不是精确解**。领导者收益作为 x 的函数是分片线性的（一般还不凹），
           最优可能落在两个网格点之间的折点上；请把 ``n_grid`` 调大做一次收敛性对照再报数字。
        2. 领导者收益的口径必须显式选定：本实现取"领导者对跟随者行动的线性估值减去工具成本"。
           如果你的模型里领导者还从自己的决策直接获益，请把那一项并入 ``leader_cost``（取负号）。
        3. **同时决策（Nash/Cournot）基准**：把 ``x_upper`` 设为全 0 就退化成"领导者不使用承诺
           能力"的同时决策结果。网格里含 x = 0 且做的是最大化，所以 Stackelberg 领导者收益
           不会低于该基准（自测里做了这条断言）。
        4. 跟随者 LP 的**退化与多解**会让"最优响应"不唯一：此时 ``y_follower`` 只是单纯形给出的
           一个顶点，领导者收益在多个最优响应之间可能有差别（乐观/悲观口径）。本实现取乐观口径
           （领导者按跟随者会选对自己最有利的一个来评估），论文里必须写明。
        5. ``A_leader`` 的符号极易搞反：右端项是 ``b - A_leader x``，不是 ``b + A_leader x``。
        6. 网格点数随 m 指数增长：m = 3、n_grid = 21 已经是 9261 次 LP，竞赛里请控制在
           ``n_grid ** m <= 1e4`` 以内，或改用领导者收益的包络（分段线性）精确枚举。

    参考:
        von Stackelberg 1934《Marktform und Gleichgewicht》；Dempe 2002（双层规划综述）；
        线性双层规划的"领导者先动、跟随者解 LP"标准表述。
    """
    Al = as_matrix(A_leader, "A_leader")
    cl = as_vector(c_leader, "c_leader")
    Af = as_matrix(A_follower, "A_follower")
    cf = as_vector(c_follower, "c_follower")
    bf = as_vector(b_follower, "b_follower")

    k, n = Af.shape
    if bf.size != k:
        raise ValueError(f"b_follower 长度 {bf.size} 与 A_follower 的行数 {k} 不一致")
    if cf.size != n:
        raise ValueError(f"c_follower 长度 {cf.size} 与 A_follower 的列数 {n} 不一致")
    if cl.size != n:
        raise ValueError(f"c_leader 长度 {cl.size} 必须等于跟随者决策维数 n={n}")
    if Al.shape[0] != k:
        raise ValueError(f"A_leader 行数 {Al.shape[0]} 必须等于跟随者约束条数 k={k}")

    m = Al.shape[1]
    if m < 1:
        raise ValueError("A_leader 至少要有 1 列（领导者决策维数 m >= 1）")
    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError(f"n_grid 必须 >= 2，得到 {n_grid}")

    if x_upper is None:
        scale = float(np.abs(bf).max()) if bf.size else 1.0
        upper = np.full(m, scale if scale > 0 else 1.0)
    else:
        upper = as_vector(x_upper, "x_upper")
        if upper.size != m:
            raise ValueError(f"x_upper 长度 {upper.size} 与领导者决策维数 m={m} 不一致")
    if np.any(upper < 0):
        raise ValueError(f"x_upper 必须非负，得到 {upper.tolist()}")
    if not np.all(np.isfinite(upper)):
        raise ValueError("x_upper 必须全部有限（无界搜索区间上网格法没有意义）")

    if leader_cost is None:
        cost = np.zeros(m)
    else:
        cost = as_vector(leader_cost, "leader_cost")
        if cost.size != m:
            raise ValueError(f"leader_cost 长度 {cost.size} 与领导者决策维数 m={m} 不一致")

    axes = [np.linspace(0.0, float(upper[i]), n_grid) for i in range(m)]
    best: Optional[Dict[str, object]] = None
    n_failed = 0
    for combo in itertools.product(*axes):
        x = np.array(combo, dtype=float)
        rhs = bf - Al @ x
        res = simplex_lp(cf, A_ub=Af, b_ub=rhs, maximize=True)
        if str(res.get("status")) != "optimal":
            n_failed += 1
            continue
        y = np.clip(np.asarray(res["x"], dtype=float), 0.0, None)
        payoff = float(cl @ y - cost @ x)
        if best is None or payoff > float(best["leader_payoff"]):
            best = {
                "x_leader": x,
                "y_follower": y,
                "leader_payoff": payoff,
                "follower_payoff": float(cf @ y),
            }
    if best is None:
        raise ValueError(
            f"领导者的 {n_grid ** m} 个网格点上跟随者的 LP 全部非 optimal"
            "（典型原因是资源上界使得跟随者问题不可行）；请放宽 x_upper 或检查 A_leader 的符号"
        )
    return best


# --------------------------------------------------------------------------- #
# 纳什谈判解
# --------------------------------------------------------------------------- #
def nash_bargaining_solution(d, payoff_set) -> dict:
    """二人纳什谈判解：在可行集上最大化纳什积 ``(u1 - d1) * (u2 - d2)``。

    参数:
        d: 分歧点（disagreement point / 谈判破裂时的效用），长度 2。
        payoff_set: 可行集，两种口径二选一：
            (a) 离散点集：形状 ``(N, 2)`` 的数组（或 list of list），只在给定点上取最大纳什积；
            (b) 凸多边形：字典 ``{"A_ub": A, "b_ub": b}``，可行集为 ``{u : A u <= b}``，
                A 形状 ``(k, 2)``。**必须是有界多边形，且请把下界（u >= 0 之类）也显式写成 A 的行**
                （本函数不隐式假定 u >= 0，负效用坐标是允许的）。

    返回:
        dict，键为：
        ``solution``   形状 (2,) 的谈判解效用点 ``(u1*, u2*)``；
        ``utilities``   形状 (2,) 的**净收益** ``u* - d``（相对分歧点的增量，两者相乘即纳什积；
                        每个分量都大于 ``_TOL = 1e-7``，见"陷阱"第 1 条）；
        ``product``   float，纳什积 ``(u1* - d1)(u2* - d2)``（即最大化的目标值）。

    算法:
        1. 离散点集：逐点算纳什积，取最大值（严格大于才替换，并列取输入顺序在前的点，保证确定性）；
        2. 凸多边形：先求顶点（两两约束直线求交 + 可行性过滤 + 按极角排序），再在**每条边**上
           扫描：把边参数化为 ``u(t) = A + t(B - A)``，纳什积是 t 的二次函数，其内部驻点
           ``t* = -(p1 q2 + p2 q1) / (2 q1 q2)``（``p = A - d``、``q = B - A``）落在 (0, 1) 时
           作为候选点；在"所有顶点 + 所有边的内部驻点"里取纳什积最大者。
           对多边形而言这个候选集是**充分的**：log 纳什积在可行集上是凹的，最大值必落在东北边界，
           而边界由有限条线段拼成，每段上的最大值只可能在端点或该段的驻点。
        3. 个体理性按**容差**筛候选：``u - d`` 的每个分量都必须 **> ``_TOL``（``_TOL = 1e-7``）**，
           否则该候选被丢弃；只有所有候选都被丢弃时才抛 ValueError。也就是说返回解只保证
           "净收益每个分量都大于 1e-7"，而不是数学意义上的"严格大于 0"。

    复杂度:
        时间：离散 O(N)；多边形 O(k^2)（顶点枚举）+ O(k) 条边的二次求根 + 4 次辅助 LP（有界性检查）；
        空间 O(k)。

    陷阱:
        1. **纳什解要求可行集里有严格优于 d 的点**。若 d 本身就在帕累托前沿上（没有合作剩余），
           纳什积的最大值为 0，谈判解退化；本实现直接抛 ValueError，避免把 0 当作"解"报出去。
           但注意"严格优于"在代码里是**容差判定**：候选要被采纳必须满足 ``gain > _TOL``，
           而 ``_TOL = 1e-7``（硬编码的绝对容差），所以两分量只到 1e-8 量级的
           "合作剩余"会被当成没有剩余，两侧之一恰好等于 1e-7 的点也会被丢掉。效用单位很大或
           很小时这个绝对容差的实际含义会随之变化，报告结论时请说明。
        2. 离散点集口径下"解"只是给定点里纳什积最大的那个，**不是**真正的纳什谈判解；
           点足够密时才可以近似当作连续解，论文里要说明采样方式。
        3. 凸多边形口径**只支持二维**（纳什谈判解本身就是二人博弈的概念），
           三维及以上的纳什积最大化要用凸优化求解器。
        4. 对称性公理：若可行集关于 ``u1 <-> u2`` 对称且 ``d1 == d2``，解必须落在对称轴上。
           本函数不对输入做对称化，所以数值上会有 1e-15 级别的偏差——断言时请用容差，别用 ==。
        5. 多边形顶点用**两两直线求交**得到：退化多边形（三条边共点、存在冗余约束）会产生重复交点，
           本实现按容差去重；若 A 的行里出现全零行，该约束不构成直线（会被跳过），
           请在传参前把冗余约束去掉。

    参考:
        Nash 1950, "The Bargaining Problem", Econometrica 18(2)；Osborne & Rubinstein
        《A Course in Game Theory》第 15 章（公理化谈判解）。
    """
    dvec = as_vector(d, "d")
    if dvec.size != 2:
        raise ValueError(f"分歧点 d 必须是长度 2 的向量，得到长度 {dvec.size}")

    if isinstance(payoff_set, Mapping):
        if "A_ub" not in payoff_set or "b_ub" not in payoff_set:
            raise ValueError("payoff_set 用多边形口径时必须同时给出 'A_ub' 与 'b_ub' 两个键")
        A_ub = as_matrix(payoff_set["A_ub"], "A_ub")
        b_ub = as_vector(payoff_set["b_ub"], "b_ub")
        if A_ub.shape[1] != 2:
            raise ValueError(f"A_ub 必须是 (k, 2)（二维谈判问题），得到形状 {A_ub.shape}")
        if A_ub.shape[0] != b_ub.size:
            raise ValueError(f"A_ub 行数 {A_ub.shape[0]} 与 b_ub 长度 {b_ub.size} 不一致")
        verts = _polygon_vertices(A_ub, b_ub)
        candidates: List[np.ndarray] = [v.copy() for v in verts]
        for i in range(len(verts)):
            p0 = verts[i] - dvec
            q = verts[(i + 1) % len(verts)] - verts[i]
            if abs(float(q[0] * q[1])) < 1e-15:
                continue  # 该边与某条坐标轴平行：二次项为 0，极值只可能在端点
            t = -float(p0[0] * q[1] + p0[1] * q[0]) / (2.0 * float(q[0] * q[1]))
            if 0.0 < t < 1.0:
                candidates.append(verts[i] + t * q)
    else:
        pts = as_matrix(payoff_set, "payoff_set")
        if pts.shape[1] != 2:
            raise ValueError(f"payoff_set 作为离散点集必须是 (N, 2)，得到形状 {pts.shape}")
        candidates = [pts[i].copy() for i in range(pts.shape[0])]

    best_u: Optional[np.ndarray] = None
    best_prod = float("-inf")
    for u in candidates:
        gain = u - dvec
        if gain[0] <= _TOL or gain[1] <= _TOL:
            continue  # 不满足个体理性（没有严格优于分歧点）
        prod = float(gain[0] * gain[1])
        if prod > best_prod:
            best_prod, best_u = prod, u
    if best_u is None:
        raise ValueError(
            "可行集中没有严格优于分歧点 d 的点（没有合作剩余），纳什谈判解不存在；"
            f"请检查 d={dvec.tolist()} 是否已经落在可行集的帕累托前沿上"
        )
    return {
        "solution": np.asarray(best_u, dtype=float),
        "utilities": np.asarray(best_u - dvec, dtype=float),
        "product": float(best_prod),
    }


def _polygon_vertices(A_ub: np.ndarray, b_ub: np.ndarray) -> List[np.ndarray]:
    """求二维有界多边形 ``{u : A_ub u <= b_ub}`` 的顶点（按极角逆时针排序）。

    参数:
        A_ub: (k, 2) 约束矩阵。
        b_ub: 长度 k 的右端项。

    返回:
        list of np.ndarray，每个是 (2,) 顶点，按相对形心的极角升序排列（可直接当作边的顺序用）。

    算法:
        1. 先用 4 个辅助 LP（最大化/最小化 u1 与 u2）判定可行域非空且**有界**：
           若衰退锥 ``{d : A_ub d <= 0}`` 含非零方向，则某个坐标方向必无界，这 4 个 LP 会报 unbounded；
        2. 两两取约束直线求交，保留满足 ``A_ub v <= b_ub``（带容差）的交点，按容差去重；
        3. 以形心为原点按 ``arctan2`` 极角排序。

    复杂度:
        时间 O(k^2) / 空间 O(k^2)。

    陷阱:
        1. **必须显式传入下界约束**（例如 ``-u1 <= 0``）：本函数不假定 u >= 0，缺下界时第 1 步会
           直接判为无界并抛 ValueError——这是有意的，无界可行集上的纳什谈判解本身就没有定义。
        2. 求交时要用平面法向量叉积的**行列式**判平行；用斜率比较会在竖直线（a1 = 0）上除零。
        3. 容差按 ``b_ub`` 的量级缩放；若约束系数量级相差 1e6 倍以上，交点判定会不可靠，
           请先做无量纲化（与 optimization.simplex_lp 的数值要求一致）。

    参考:
        计算几何中的半平面交（Sutherland-Hodgman / 增量式 HPI）的暴力特例。
    """
    n_row = A_ub.shape[0]
    free = [(None, None), (None, None)]
    for j in range(2):
        for sign in (1.0, -1.0):
            c = np.zeros(2)
            c[j] = sign
            res = simplex_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=free, maximize=True)
            status = str(res.get("status"))
            if status == "infeasible":
                raise ValueError("payoff_set 描述的可行多边形为空（约束互相矛盾）")
            if status != "optimal":
                raise ValueError(
                    f"payoff_set 描述的多边形在 u{j + 1} 方向无界（status={status}）："
                    "纳什谈判解要求有界可行集，请补上下界约束"
                )

    tol = 1e-9 * max(1.0, float(np.abs(b_ub).max()))
    verts: List[np.ndarray] = []
    for i, j in itertools.combinations(range(n_row), 2):
        a1 = A_ub[i]
        a2 = A_ub[j]
        det = float(a1[0] * a2[1] - a1[1] * a2[0])
        if abs(det) < 1e-12:
            continue
        v = np.array([
            (b_ub[i] * a2[1] - a1[1] * b_ub[j]) / det,
            (a1[0] * b_ub[j] - b_ub[i] * a2[0]) / det,
        ])
        if not np.all(A_ub @ v <= b_ub + tol):
            continue
        if any(np.abs(v - w).max() <= 1e-9 for w in verts):
            continue
        verts.append(v)
    if len(verts) < 3:
        raise ValueError(
            f"多边形顶点数 {len(verts)} 少于 3 个：约束可能有冗余、退化，或可行域不是二维多边形"
        )
    verts_arr = np.array(verts, dtype=float)
    center = verts_arr.mean(axis=0)
    order = np.argsort(np.arctan2(verts_arr[:, 1] - center[1], verts_arr[:, 0] - center[0]))
    return [verts_arr[i] for i in order]


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    返回:
        dict，键全部为简短 ASCII，值可复现。**本模块所有函数都不含随机性**
        （复制者动态是确定性 ODE，稳定匹配是确定性算法），因此两次调用逐位相同。

    算法:
        覆盖本模块**全部 8 个**公开函数（``zero_sum_value_lp``、``nash_support_enumeration``、
        ``shapley_value``、``gale_shapley``、``replicator_dynamics``、``minimax_alpha_beta``、
        ``stackelberg_lp``、``nash_bargaining_solution``）各自的"已知答案"算例：
        - 配对硬币（matching pennies）[[1,-1],[-1,1]]：value 应为 0，双方各 0.5/0.5；
        - [[3,-1],[-2,1]]：手工可算 value = 1/7，行策略 (3/7, 4/7)，列策略 (2/7, 5/7)；
        - 囚徒困境：恰好 1 个纳什均衡，即 (D, D)；
        - 性别战：恰好 3 个纳什均衡（2 纯 + 1 混合）；
        - Shapley：3 人对称多数博弈（任意 2 人即可获胜）→ 每人 1/3；
          联合国安理会式投票博弈（5 常任 + 10 非常任，9 票且 5 常任全同意）；
        - Gale-Shapley：4 男 4 女已知偏好实例，校验阻塞对为 0；
        - 复制者动态：协调博弈下初值偏向哪个策略就收敛到哪个纯均衡；
        - α-β 剪枝：8 个数轮流取数的完全博弈树（8! = 40320 个叶子）上，与写在本函数内部的
          **不剪枝**完全极小极大对照实现逐位对拍，并与"轮流取最大"的解析值 5 对拍；
        - Stackelberg：一维领导者工具的算例，解析最优落在 x = 1 的折点上（领导者收益 3 > 同时决策的 2）；
        - 纳什谈判：线性前沿上的闭式"平分剩余"解，以及对称可行集上的对称性检验。

    复杂度:
        时间约 O(1)（最大的是 Shapley 的 n=15 → 32768 个子集）/ 空间 O(1)。

    陷阱:
        本函数的最重算例是安理会投票博弈（15 人、32768 个子集、每个子集 15 次边际贡献），
        单次约 0.5~2 秒。若 run_algorithms.py 想更快，可以只保留 3 人算例。

    参考:
        契约第 3 节"每个模块结尾提供 _self_test()"。
    """
    result: Dict[str, object] = {}

    # ---------- 配对硬币：value = 0 ----------
    mp = zero_sum_value_lp([[1.0, -1.0], [-1.0, 1.0]])
    result["mp_value"] = round(float(mp["value"]), 9)
    result["mp_row"] = [round(float(t), 9) for t in mp["row_strategy"]]
    result["mp_col"] = [round(float(t), 9) for t in mp["col_strategy"]]
    result["mp_status"] = str(mp["lp_status"])

    # ---------- [[3, -1], [-2, 1]]：value = 1/7，行 (3/7, 4/7)，列 (2/7, 5/7) ----------
    g2 = zero_sum_value_lp([[3.0, -1.0], [-2.0, 1.0]])
    result["g2_value"] = round(float(g2["value"]), 9)
    result["g2_value_expected"] = round(1.0 / 7.0, 9)
    result["g2_row"] = [round(float(t), 9) for t in g2["row_strategy"]]
    result["g2_col"] = [round(float(t), 9) for t in g2["col_strategy"]]
    result["g2_row_expected"] = [round(3.0 / 7.0, 9), round(4.0 / 7.0, 9)]
    result["g2_col_expected"] = [round(2.0 / 7.0, 9), round(5.0 / 7.0, 9)]

    # ---------- 囚徒困境：唯一均衡 (D, D) ----------
    # 行/列收益（-1 = 合作 C 的诱惑差、-5 = 互相背叛）：
    #   C,C = (-1,-1)   C,D = (-5, 0)   D,C = (0, -5)   D,D = (-3,-3)
    pd_A = np.array([[-1.0, -5.0], [0.0, -3.0]])
    pd_B = np.array([[-1.0, 0.0], [-5.0, -3.0]])
    pd_eq = nash_support_enumeration(pd_A, pd_B)
    result["pd_n_eq"] = len(pd_eq)
    result["pd_row0"] = [round(float(t), 6) for t in pd_eq[0]["row"]] if pd_eq else []
    result["pd_col0"] = [round(float(t), 6) for t in pd_eq[0]["col"]] if pd_eq else []

    # ---------- 性别战：3 个均衡 ----------
    # 收益：(O,O)=(2,1)  (O,T)=(0,0)  (T,O)=(0,0)  (T,T)=(1,2)
    bos_A = np.array([[2.0, 0.0], [0.0, 1.0]])
    bos_B = np.array([[1.0, 0.0], [0.0, 2.0]])
    bos_eq = nash_support_enumeration(bos_A, bos_B)
    result["bos_n_eq"] = len(bos_eq)
    result["bos_n_pure"] = int(sum(
        1 for e in bos_eq
        if np.count_nonzero(np.asarray(e["row"]) > _TOL) == 1
        and np.count_nonzero(np.asarray(e["col"]) > _TOL) == 1
    ))
    result["bos_mixed_row"] = [
        round(float(t), 9) for t in bos_eq[-1]["row"]
    ] if bos_eq else []
    result["bos_mixed_col"] = [
        round(float(t), 9) for t in bos_eq[-1]["col"]
    ] if bos_eq else []

    # ---------- Shapley：3 人对称多数博弈（任意 2 人获胜，总收益 1）----------
    def majority3(S: frozenset) -> float:
        return 1.0 if len(S) >= 2 else 0.0

    sh_sym = shapley_value(majority3, 3)
    result["sh_sym"] = [round(float(t), 9) for t in sh_sym]
    result["sh_sym_sum"] = round(float(sh_sym.sum()), 9)

    # ---------- Shapley：联合国安理会式投票博弈 ----------
    # 15 个成员：0..4 为常任理事国（有否决权），5..14 为非常任。
    # 决议通过条件：>= 9 票 且 5 个常任全部同意。总收益 1。
    perm = set(range(5))

    def un_sc(S: frozenset) -> float:
        if len(S) < 9:
            return 0.0
        return 1.0 if perm.issubset(S) else 0.0

    sh_un = shapley_value(un_sc, 15)
    result["sh_un_perm"] = round(float(sh_un[0]), 9)
    result["sh_un_nonperm"] = round(float(sh_un[5]), 9)
    result["sh_un_sum"] = round(float(sh_un.sum()), 9)
    result["sh_un_perm_total"] = round(float(sh_un[:5].sum()), 9)

    # ---------- 位掩码口径应与 frozenset 口径完全一致 ----------
    mask_table = {m: (1.0 if bin(m).count("1") >= 2 else 0.0) for m in range(8)}
    sh_mask = shapley_value(mask_table, 3)
    result["sh_mask_max_dev"] = round(float(np.abs(sh_mask - sh_sym).max()), 12)

    # ---------- Gale-Shapley：4x4 已知实例，阻塞对应为 0 ----------
    men_prefs = {
        "m1": ["w1", "w2", "w3", "w4"],
        "m2": ["w2", "w1", "w3", "w4"],
        "m3": ["w3", "w1", "w2", "w4"],
        "m4": ["w4", "w1", "w2", "w3"],
    }
    women_prefs = {
        "w1": ["m4", "m3", "m2", "m1"],
        "w2": ["m3", "m4", "m1", "m2"],
        "w3": ["m2", "m1", "m4", "m3"],
        "w4": ["m1", "m2", "m3", "m4"],
    }
    gs = gale_shapley(men_prefs, women_prefs)
    result["gs_matching"] = [f"{m}->{gs['matching'][m]}" for m in sorted(gs["matching"])]
    result["gs_n_proposals"] = int(gs["n_proposals"])
    result["gs_n_blocking"] = int(gs["n_blocking_pairs"])

    # 第二个实例：男生会**被拒**（不是人人都拿到第一志愿），用来真正跑通拒绝分支，
    # 并用一段与 gale_shapley 内部实现完全独立的暴力枚举代码复核稳定性。
    men_prefs2 = {
        "m1": ["w1", "w2", "w3"],
        "m2": ["w2", "w1", "w3"],
        "m3": ["w1", "w2", "w3"],
    }
    women_prefs2 = {
        "w1": ["m2", "m1", "m3"],
        "w2": ["m1", "m2", "m3"],
        "w3": ["m1", "m2", "m3"],
    }
    gs2 = gale_shapley(men_prefs2, women_prefs2)
    result["gs2_matching"] = [f"{m}->{gs2['matching'][m]}" for m in sorted(gs2["matching"])]
    result["gs2_n_proposals"] = int(gs2["n_proposals"])
    result["gs2_n_blocking"] = int(gs2["n_blocking_pairs"])

    # 独立暴力复核：对每一对 (男, 女) 直接用原始偏好列表判断是否互相更喜欢
    inv2 = {w: m for m, w in gs2["matching"].items() if w is not None}
    brute_blocking = 0
    pairs_checked = 0
    for man, lst in men_prefs2.items():
        for woman in lst:
            if gs2["matching"].get(man) == woman:
                continue
            pairs_checked += 1
            her_partner = inv2.get(woman)
            # 男方是否更喜欢她：在原始偏好列表里她的下标更小
            man_likes = lst.index(woman) < (
                lst.index(gs2["matching"][man]) if gs2["matching"][man] is not None else 10 ** 9
            )
            wlst = women_prefs2[woman]
            woman_likes = wlst.index(man) < (
                wlst.index(her_partner) if her_partner is not None else 10 ** 9
            )
            if man_likes and woman_likes:
                brute_blocking += 1
    result["gs2_pairs_checked"] = pairs_checked
    result["gs2_brute_blocking"] = brute_blocking
    # 男方最优性：男方求婚版应让每个男生拿到"所有稳定匹配中最好的"结果
    result["gs2_m1_gets_first_choice"] = gs2["matching"]["m1"] == men_prefs2["m1"][0]

    # ---------- 复制者动态：协调博弈 [[2,0],[0,1]] ----------
    coord = np.array([[2.0, 0.0], [0.0, 1.0]])
    rep_a = replicator_dynamics(coord, np.array([0.8, 0.2]), T=2000, dt=0.01)
    result["rep_final_a"] = [round(float(t), 6) for t in rep_a["final"]]
    result["rep_converged_a"] = bool(rep_a["converged"])
    result["rep_len"] = int(rep_a["trajectory"].shape[0])

    rep_b = replicator_dynamics(coord, np.array([0.2, 0.8]), T=2000, dt=0.01)
    result["rep_final_b"] = [round(float(t), 6) for t in rep_b["final"]]

    # 石头剪刀布：内点均衡是中性稳定（不发散到边界），用于说明"收敛到均衡≠稳定"
    rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    rep_rps = replicator_dynamics(rps, np.array([0.5, 0.3, 0.2]), T=500, dt=0.01)
    result["rep_rps_min"] = round(float(rep_rps["trajectory"].min()), 9)
    result["rep_rps_sum_dev"] = round(float(
        np.abs(rep_rps["trajectory"].sum(axis=1) - 1.0).max()
    ), 12)

    # ---------- α-β 剪枝：与不剪枝的完全极小极大对拍 ----------
    # 抽象完全信息博弈："两人轮流从数集里取数，最后比各自取到的数字之和"。
    # 节点 = (剩余数字元组, 最大化者得分, 最小化者得分, 轮到谁)；终局评估 = 两者分差。
    # 之所以不用井字棋：这棵树的规模（8! = 40320 个叶子）与井字棋同量级，但不需要在自测里
    # 再嵌一套棋盘胜负判定，能把"剪枝 vs 不剪枝"的对照写得更干净。
    ab_numbers = (5, 3, 8, 2, 9, 4, 1, 7)

    def ab_children(n):
        remaining, s_max, s_min, turn = n
        out = []
        for i, v in enumerate(remaining):
            rest = remaining[:i] + remaining[i + 1:]
            if turn == 0:
                out.append((rest, s_max + v, s_min, 1))
            else:
                out.append((rest, s_max, s_min + v, 0))
        return out

    def ab_evaluate(n):
        return float(n[1] - n[2])

    brute_stats = {"n_evaluated": 0}

    def brute_minimax(n, d, maximizing):
        """对照实现：完全不剪枝的极小极大（与本模块的 minimax_alpha_beta 无共享代码）。"""
        kids = ab_children(n) if d > 0 else []
        if d == 0 or not kids:
            brute_stats["n_evaluated"] += 1
            return ab_evaluate(n)
        if maximizing:
            best = -float("inf")
            for kid in kids:
                cand = brute_minimax(kid, d - 1, False)
                if cand > best:
                    best = cand
            return best
        best = float("inf")
        for kid in kids:
            cand = brute_minimax(kid, d - 1, True)
            if cand < best:
                best = cand
        return best

    ab_start = (ab_numbers, 0.0, 0.0, 0)
    ab_full = minimax_alpha_beta(ab_start, len(ab_numbers), float("-inf"), float("inf"),
                                 True, ab_evaluate, ab_children)
    brute_full = brute_minimax(ab_start, len(ab_numbers), True)
    if abs(float(ab_full["value"]) - float(brute_full)) > 1e-9:
        raise AssertionError(
            f"α-β 剪枝值 {ab_full['value']} 与不剪枝极小极大 {brute_full} 不一致"
        )
    # 独立解析结论：两人轮流取数时"每次取当前最大"是最优的，分差等于降序交错和
    # 9 - 8 + 7 - 5 + 4 - 3 + 2 - 1 = 5。
    known_value = 5.0
    if abs(float(ab_full["value"]) - known_value) > 1e-9:
        raise AssertionError(
            f"α-β 剪枝值 {ab_full['value']} 与解析值 {known_value} 不一致"
        )
    if int(ab_full["n_evaluated"]) >= int(brute_stats["n_evaluated"]):
        raise AssertionError(
            f"α-β 剪枝求值次数 {ab_full['n_evaluated']} 未严格少于不剪枝的 "
            f"{brute_stats['n_evaluated']}"
        )
    if int(ab_full["n_pruned"]) < 1:
        raise AssertionError("α-β 剪枝一次都没有发生，说明剪枝逻辑没生效")
    result["ab_value"] = round(float(ab_full["value"]), 6)
    result["ab_brute_value"] = round(float(brute_full), 6)
    result["ab_value_match"] = bool(abs(float(ab_full["value"]) - float(brute_full)) <= 1e-9)
    result["ab_known_value"] = known_value
    result["ab_n_evaluated"] = int(ab_full["n_evaluated"])
    result["ab_brute_n_evaluated"] = int(brute_stats["n_evaluated"])
    result["ab_n_pruned"] = int(ab_full["n_pruned"])
    result["ab_fewer_nodes"] = bool(
        int(ab_full["n_evaluated"]) < int(brute_stats["n_evaluated"])
    )
    result["ab_best_remaining"] = [int(v) for v in ab_full["best_child"][0]]
    result["ab_best_taken"] = int(
        sum(ab_numbers) - sum(int(v) for v in ab_full["best_child"][0])
    )

    # 截断深度下（depth = 3）同样与不剪枝对照实现一致——这条覆盖 depth == 0 的叶子分支
    brute_stats["n_evaluated"] = 0
    ab_d3 = minimax_alpha_beta(ab_start, 3, float("-inf"), float("inf"),
                               True, ab_evaluate, ab_children)
    brute_d3 = brute_minimax(ab_start, 3, True)
    if abs(float(ab_d3["value"]) - float(brute_d3)) > 1e-9:
        raise AssertionError(
            f"depth=3 时 α-β 值 {ab_d3['value']} 与不剪枝极小极大 {brute_d3} 不一致"
        )
    result["ab_depth3_value"] = round(float(ab_d3["value"]), 6)
    result["ab_depth3_brute_value"] = round(float(brute_d3), 6)
    result["ab_depth3_n_evaluated"] = int(ab_d3["n_evaluated"])
    result["ab_depth3_brute_n_evaluated"] = int(brute_stats["n_evaluated"])

    # ---------- Stackelberg：一维工具，解析最优在折点 x = 1 ----------
    # 跟随者：max 3*y1 + 4*y2  s.t.  y1 + y2 <= 10,  y1 <= 2 + 3x,  y2 <= 6 - x
    #   => x <= 1 时 y = (2 + 3x, 6 - x)；x >= 1 时 y = (4 + x, 6 - x)（容量上限接管）
    # 领导者收益 = y1 - 2x：左支 2 + x（升），右支 4 - x（降）=> 最优恰在 x = 1，收益 3。
    st_A_leader = [[0.0], [-3.0], [1.0]]
    st_c_leader = [1.0, 0.0]
    st_A_follower = [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    st_c_follower = [3.0, 4.0]
    st_b_follower = [10.0, 2.0, 6.0]
    st = stackelberg_lp(st_A_leader, st_c_leader, st_A_follower, st_c_follower,
                        st_b_follower, x_upper=[3.0], n_grid=7, leader_cost=[2.0])
    if abs(float(st["x_leader"][0]) - 1.0) > 1e-9:
        raise AssertionError(
            f"Stackelberg 领导者决策 {st['x_leader'][0]} 与解析最优 1 不符"
        )
    if abs(float(st["leader_payoff"]) - 3.0) > 1e-9:
        raise AssertionError(
            f"Stackelberg 领导者收益 {st['leader_payoff']} 与解析值 3 不符"
        )
    # 同时决策基准：领导者不使用承诺能力（x 固定为 0）时的收益
    st_nash = stackelberg_lp(st_A_leader, st_c_leader, st_A_follower, st_c_follower,
                             st_b_follower, x_upper=[0.0], n_grid=2, leader_cost=[2.0])
    if float(st["leader_payoff"]) < float(st_nash["leader_payoff"]) - 1e-9:
        raise AssertionError(
            f"Stackelberg 领导者收益 {st['leader_payoff']} 低于同时决策的 "
            f"{st_nash['leader_payoff']}"
        )
    result["st_x_leader"] = [round(float(v), 6) for v in st["x_leader"]]
    result["st_y_follower"] = [round(float(v), 6) for v in st["y_follower"]]
    result["st_leader_payoff"] = round(float(st["leader_payoff"]), 6)
    result["st_follower_payoff"] = round(float(st["follower_payoff"]), 6)
    result["st_nash_y_follower"] = [round(float(v), 6) for v in st_nash["y_follower"]]
    result["st_nash_leader_payoff"] = round(float(st_nash["leader_payoff"]), 6)
    result["st_leader_ge_nash"] = bool(
        float(st["leader_payoff"]) >= float(st_nash["leader_payoff"]) - 1e-9
    )

    # ---------- 纳什谈判解：闭式解 / 对称性 ----------
    nb_triangle = {"A_ub": [[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                   "b_ub": [2.0, 0.0, 0.0]}
    nb_d = [0.5, 0.2]
    nb_poly = nash_bargaining_solution(nb_d, nb_triangle)
    # 闭式：前沿是 u1 + u2 = 2 的直线段，纳什解把剩余 (2 - d1 - d2) 平分
    nb_surplus = 2.0 - 0.5 - 0.2
    nb_expected = np.array([0.5 + nb_surplus / 2.0, 0.2 + nb_surplus / 2.0])
    dev = float(np.abs(nb_poly["solution"] - nb_expected).max())
    if dev > 1e-9:
        raise AssertionError(
            f"纳什谈判解 {nb_poly['solution'].tolist()} 与闭式解 {nb_expected.tolist()} "
            f"相差 {dev}"
        )
    if abs(float(nb_poly["product"]) - (nb_surplus / 2.0) ** 2) > 1e-9:
        raise AssertionError(
            f"纳什积 {nb_poly['product']} 与闭式值 {(nb_surplus / 2.0) ** 2} 不符"
        )
    result["nb_poly_solution"] = [round(float(v), 9) for v in nb_poly["solution"]]
    result["nb_poly_utilities"] = [round(float(v), 9) for v in nb_poly["utilities"]]
    result["nb_poly_expected"] = [round(float(v), 9) for v in nb_expected]
    result["nb_poly_product"] = round(float(nb_poly["product"]), 9)
    result["nb_poly_dev"] = round(dev, 12)

    # 对称可行集 + 对称分歧点 => 解必须落在对称轴上（纳什对称性公理）
    nb_sym = nash_bargaining_solution([0.0, 0.0], nb_triangle)
    if abs(float(nb_sym["solution"][0]) - float(nb_sym["solution"][1])) > 1e-9:
        raise AssertionError(
            f"对称可行集上的纳什谈判解不对称：{nb_sym['solution'].tolist()}"
        )
    result["nb_sym_solution"] = [round(float(v), 9) for v in nb_sym["solution"]]
    result["nb_sym_dev"] = round(float(abs(nb_sym["solution"][0] - nb_sym["solution"][1])), 12)

    # 离散点集口径：纳什积最大者
    nb_disc = nash_bargaining_solution([0.0, 0.0],
                                       [[1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [0.0, 5.0]])
    if np.abs(nb_disc["solution"] - np.array([2.0, 2.0])).max() > 1e-12:
        raise AssertionError(
            f"离散点集上的纳什解 {nb_disc['solution'].tolist()} 应为 [2.0, 2.0]"
        )
    result["nb_disc_solution"] = [round(float(v), 9) for v in nb_disc["solution"]]
    result["nb_disc_product"] = round(float(nb_disc["product"]), 9)

    return result
