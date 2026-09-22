"""多目标优化：加权和法、ε-约束法、支配关系判定、非支配排序、拥挤距离、NSGA-II、
MOEA/D、Pareto 前沿、二维超体积、理想点距离与前沿质量指标。

本模块共 13 个公开函数：支配与排序 ``pareto_dominates`` / ``fast_non_dominated_sort`` /
``crowding_distance``，前沿构造 ``weighted_sum_pareto`` / ``epsilon_constraint_pareto`` /
``nsga2`` / ``moead``，前沿后处理 ``pareto_front`` / ``hypervolume_2d`` /
``ideal_point_distance``，前沿质量评价 ``igd_metric`` / ``spacing_metric`` / ``knee_points``。

定位：数学建模里"多个目标互相冲突、不存在单一最优解"的一类问题。此时正确的输出不是
一个点，而是一整条（或一整片）**Pareto 前沿**，以及"这些解各自牺牲了什么"的量化说明。
本模块给出从最朴素的加权和法、ε-约束法到 NSGA-II 的一整套透明实现。

边界：这是**教学透明版**。加权和法/ε-约束法用自适应随机方向搜索（对目标函数不可导、
非凸也适用），不保证全局最优；NSGA-II 是标准流程（非支配排序 + 拥挤距离 + 二元锦标赛 +
SBX 交叉 + 多项式变异），但边界处理用简单截断、也没有做约束支配。结论必须写成
"在给定预算与给定种子上得到的前沿"，不能外推成"这是真正的最优前沿"。

关键约定
--------
- 目标**默认全部最小化**；``minimize=False`` 时内部对所有目标统一取负后仍按最小化处理。
- 决策向量是形状 (n_dim,) 的 ``np.ndarray``；``objective_fn(x) -> np.ndarray`` 返回形状 (m,)。
- 支配关系：f1 支配 f2 当且仅当逐分量 ``f1 <= f2`` 且至少一个分量严格更小（因此两个
  完全相同的点互不支配，会同时留在第一前沿，这是有意为之）。
- ``rank`` 是 **0 基**的前沿序号；``crowding_distance`` 只在**同一条前沿内部**计算，
  边界点取 ``inf``，且在任一目标上取值域为 0 时该**整个目标**被跳过（既不贡献距离，
  也不把 ``inf`` 送给该目标上并列的端点）。
- ``igd_metric`` 的 IGD 口径是"对**参考前沿**逐点取到近似集的最近距离再平均"，
  与 ``hypervolume_2d`` 一样要求近似集与参考集的目标维数一致。
- ``spacing_metric`` 用 Schott 间距：每个点到**其余点**的最近邻距离的样本标准差（除以 n-1），
  完全均匀的前沿得 0，数值越小分布越均匀。默认用 L1 距离（Schott 原文口径）。
- 加权和法用单纯形等距网格（权重非负且和为 1）；ε-约束法以**第 1 个目标**为 ε 上限、
  最小化其余目标的等权平均，约束用线性罚函数处理。
- 随机性一律走 ``_common.rng``（PCG64），不使用任何全局随机状态。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_matrix, as_vector, rng as make_rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "pareto_dominates",
    "fast_non_dominated_sort",
    "crowding_distance",
    "weighted_sum_pareto",
    "epsilon_constraint_pareto",
    "nsga2",
    "moead",
    "pareto_front",
    "hypervolume_2d",
    "ideal_point_distance",
    "igd_metric",
    "spacing_metric",
    "knee_points",
]


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _parse_box(bounds) -> Tuple[np.ndarray, np.ndarray]:
    """把 ``x_bounds`` 规范成逐维的 (lo, hi) 两个一维数组。

    口径（与本模块文档一致）：``(lo, hi)`` 两个标量表示 **1 维**问题；
    要 d 维共用同一区间请写 ``[(lo, hi)] * d``。

    两种写法走**完全相同**的校验：元素必须有限，且每一维都要求 ``hi > lo``，
    违反时一律抛 ``ValueError``（``(5.0, 3.0)``、``(5.0, 5.0)``、``(nan, 5.0)``、
    ``(inf, 5.0)`` 都会被拒绝；退化维请先固定该变量）。
    """
    if bounds is None:
        raise ValueError("x_bounds 不能为 None（多目标优化必须有搜索盒）")
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"x_bounds 为 (lo, hi) 时长度必须为 2，得到 {arr.size}")
        # 这里**不能**提前 return：一维简写与二维写法必须共用下面同一套有限性与
        # hi > lo 校验，否则 (5.0, 3.0) / (nan, 5.0) / (inf, 5.0) 会被静默当成合法
        # 搜索盒（个体全被钉在 lo 上），与 heuristics._parse_bounds 的口径不一致。
        lo = np.array([arr[0]], dtype=float)
        hi = np.array([arr[1]], dtype=float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        lo = arr[:, 0].copy()
        hi = arr[:, 1].copy()
    else:
        raise ValueError(f"x_bounds 形状非法：{arr.shape}，应为 (lo, hi) 或 [(lo, hi), ...]")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("x_bounds 必须是有限值")
    if np.any(hi <= lo):
        raise ValueError(
            f"每一维都要求 hi > lo，得到 lo={lo.tolist()}、hi={hi.tolist()}；"
            "退化维度请先固定该变量再优化"
        )
    return lo, hi


def _eval_vec(objective_fn: Callable[[np.ndarray], ArrayLike], x: np.ndarray) -> np.ndarray:
    """调用一次 ``objective_fn`` 并把返回值拉平成 (m,) 的 float 数组，顺便校验。"""
    if not callable(objective_fn):
        raise ValueError("objective_fn 必须是可调用对象")
    v = np.asarray(objective_fn(np.asarray(x, dtype=float)), dtype=float).ravel()
    if v.size == 0:
        raise ValueError("objective_fn 返回了空目标向量")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"objective_fn 在 x={np.asarray(x, dtype=float).tolist()} 处返回了非有限值：{v.tolist()}")
    return v


def _eval_pop(objective_fn: Callable[[np.ndarray], ArrayLike], X: np.ndarray) -> np.ndarray:
    """对整代决策变量求目标值，返回 (n_pop, m)；各行目标个数不一致时抛 ValueError。"""
    rows = [_eval_vec(objective_fn, X[i]) for i in range(X.shape[0])]
    m = rows[0].size
    for k, r in enumerate(rows):
        if r.size != m:
            raise ValueError(f"objective_fn 的返回长度不固定：第 0 个个体给出 {m} 个目标，第 {k} 个给出 {r.size} 个")
    return np.vstack(rows)


def _simplex_grid(m: int, n_levels: int) -> np.ndarray:
    """m 维单纯形上"每维 n_levels 个格点、权重和为 1"的等距网格，返回 (n_grid, m)。

    生成的是**格点**而不是随机 Dirichlet 采样：加权和法的可复现性依赖这一点，
    而且均匀格点能保证权重极端（例如只优化某一个目标）的情形一定被扫到。
    """
    k = n_levels - 1
    out: List[List[int]] = []

    def _rec(prefix: List[int], remaining: int, slots: int) -> None:
        if slots == 1:
            out.append(prefix + [remaining])
            return
        for v in range(remaining + 1):
            _rec(prefix + [v], remaining - v, slots - 1)

    _rec([], k, m)
    return np.asarray(out, dtype=float) / float(k)


def _pattern_search(
    scalar: Callable[[np.ndarray], float],
    lo: np.ndarray,
    hi: np.ndarray,
    gen: np.random.Generator,
    n_starts: int = 4,
    n_iter: int = 120,
    step_frac: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """自适应随机方向搜索（(1+1)-ES 风格），最小化 ``scalar``。

    每一步沿各向同性高斯方向走 ``step``，成功则步长放大 1.2 倍、失败则缩小 0.85 倍
    （1/5 成功法则的简化版），并投影回盒约束。``n_starts`` 次独立重启取最优。
    只依赖 ``gen``，因此同一 seed 完全可复现。
    """
    d = lo.size
    span = hi - lo
    best_x = lo + gen.random(d) * span
    best_v = float(scalar(best_x))
    for _ in range(n_starts):
        x = lo + gen.random(d) * span
        v = float(scalar(x))
        step = np.full(d, step_frac) * span
        for _it in range(n_iter):
            cand = np.clip(x + gen.standard_normal(d) * step, lo, hi)
            cv = float(scalar(cand))
            if cv < v:
                x, v = cand, cv
                step = np.minimum(step * 1.2, span)
            else:
                step = step * 0.85
        if v < best_v:
            best_v = float(v)
            best_x = x.copy()
    return best_x, best_v


def _crowded_tournament(
    rank: np.ndarray, crowd: np.ndarray, gen: np.random.Generator, k: int = 2
) -> int:
    """二元锦标赛：先比前沿序号（小者优），同前沿再比拥挤距离（大者优）。"""
    idx = gen.integers(0, int(rank.size), size=int(k))
    best = int(idx[0])
    for raw in idx[1:]:
        j = int(raw)
        if rank[j] < rank[best] or (rank[j] == rank[best] and crowd[j] > crowd[best]):
            best = j
    return best


def _sbx(
    p1: np.ndarray,
    p2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    gen: np.random.Generator,
    eta: float = 15.0,
    prob: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（SBX，分布指数 ``eta``），返回两个子代。"""
    c1 = p1.copy()
    c2 = p2.copy()
    if gen.random() > prob:
        return c1, c2
    u = gen.random(p1.size)
    beta = np.where(
        u <= 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)),
        np.power(1.0 / (2.0 * (1.0 - u)), 1.0 / (eta + 1.0)),
    )
    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def _poly_mutation(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    gen: np.random.Generator,
    eta: float = 20.0,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """多项式变异；``sigma`` 非 None 时改用标准差为 ``sigma * 区间宽度`` 的高斯变异。"""
    span = hi - lo
    if sigma is not None:
        y = x + gen.standard_normal(x.size) * (float(sigma) * span)
        return np.clip(y, lo, hi)
    p = 1.0 / float(x.size)
    mask = gen.random(x.size) < p
    if not np.any(mask):
        return x.copy()
    u = gen.random(x.size)
    delta = np.where(
        u < 0.5,
        np.power(2.0 * u, 1.0 / (eta + 1.0)) - 1.0,
        1.0 - np.power(2.0 * (1.0 - u), 1.0 / (eta + 1.0)),
    )
    y = x + np.where(mask, delta * span, 0.0)
    return np.clip(y, lo, hi)


# --------------------------------------------------------------------------- #
# 支配关系 / 非支配排序 / 拥挤距离
# --------------------------------------------------------------------------- #
def pareto_dominates(f1: ArrayLike, f2: ArrayLike) -> bool:
    """判断目标向量 ``f1`` 是否 Pareto 支配 ``f2``（口径：全部最小化）。

    参数:
        f1: 目标向量，形状 (m,)；任意可转成一维 float 数组的对象。
        f2: 目标向量，形状 (m,)，必须与 ``f1`` 目标个数一致。

    返回:
        bool。当且仅当逐分量 ``f1 <= f2`` 且至少存在一个分量 ``f1 < f2`` 时为 True。
        因此相同的两个点互不支配（返回 False），一个点也不支配自己。

    算法:
        1. 用 ``as_vector`` 校验并拉平两个向量，要求长度一致；
        2. ``dominates = np.all(f1 <= f2) and np.any(f1 < f2)``。

    复杂度:
        时间 O(m) / 空间 O(m)。

    陷阱:
        本函数**内建"全部最小化"的口径**。如果你要最大化某个目标，必须先取负再传进来，
        否则会得到完全相反的结论（这是多目标代码里最常见的一类静默错误）。
        另外注意"互不支配"不等于"等价"：两个点可以互不支配但仍然一个明显更好。

    参考:
        Pareto 1906；Deb 2001《Multi-Objective Optimization Using Evolutionary Algorithms》。
    """
    a = as_vector(f1, "f1")
    b = as_vector(f2, "f2")
    if a.size != b.size:
        raise ValueError(f"f1 与 f2 的目标个数必须一致，得到 {a.size} 与 {b.size}")
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: MatrixLike) -> dict:
    """快速非支配排序（Deb 2002），把解集划分成若干层前沿。

    参数:
        F: 目标矩阵，形状 (n_solutions, n_objectives)，**全部按最小化**理解。
           每一行是一个解的目标向量。

    返回:
        dict，键为：
        ``fronts``  list of list：第 k 项是第 k 层前沿的解下标列表，层内下标升序
                    （k 从 0 开始；第 0 层即 Pareto 前沿）；所有解恰好出现一次；
        ``rank``    形状 (n,) 的 int 数组，``rank[i]`` 是解 i 所在前沿的 0 基序号。

    算法:
        1. 对每一对 (i, j) 调用支配判定，统计 ``n_dom[j]``（支配 j 的解个数）并记录
           ``dominates[i]``（i 支配的解集合）；
        2. 所有 ``n_dom == 0`` 的解构成第 0 层前沿；
        3. 依次把当前层中每个解所支配对象的 ``n_dom`` 减 1，减到 0 的进入下一层，
           直到没有剩余解。每一条支配关系只被处理一次，因此是 O(n^2 m)。

    复杂度:
        时间 O(n^2 * m) / 空间 O(n^2)（最坏情况下的支配关系表）。

    陷阱:
        1. 本实现返回的 ``rank`` 是 **0 基**的；有些教材把它记成 1 基的"秩"，
           与文献表格对照时要注意差 1。
        2. **完全相同的两个解互不支配**，会同时留在第 0 层。若目标矩阵里有大量重复行
           （例如离散问题里同一个决策向量被反复采样到），第一层会异常臃肿，
           后续的拥挤距离也会因为目标取值域为 0 而给出 0 —— 应该先去重。
        3. O(n^2) 的实现只适合 n 在几千以内。NSGA-II 每一代都要对 2N 个解排序，
           n 一大就会成为瓶颈，这也是本模块"教学透明版"的边界。

    参考:
        Deb, Pratap, Agarwal & Meyarivan 2002, "A Fast and Elitist Multiobjective
        Genetic Algorithm: NSGA-II", IEEE TEC 6(2)。
    """
    M = as_matrix(F, "F")
    n = M.shape[0]
    dominates: List[List[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if pareto_dominates(M[i], M[j]):
                dominates[i].append(j)
                n_dom[j] += 1
            elif pareto_dominates(M[j], M[i]):
                dominates[j].append(i)
                n_dom[i] += 1

    fronts: List[List[int]] = []
    rank = np.full(n, -1, dtype=int)
    current = [i for i in range(n) if n_dom[i] == 0]
    layer = 0
    while current:
        fronts.append([int(i) for i in current])
        for i in current:
            rank[i] = layer
        nxt: List[int] = []
        for i in current:
            for j in dominates[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        current = sorted(nxt)
        layer += 1
    return {"fronts": fronts, "rank": rank}


def crowding_distance(F: MatrixLike) -> np.ndarray:
    """计算**同一条前沿内部**每个解的拥挤距离（NSGA-II 的多样性度量）。

    参数:
        F: 目标矩阵，形状 (n, m)，必须是**同一条前沿**上的解（全部最小化）。

    返回:
        形状 (n,) 的 float 数组。每个目标上按取值排序后，两个边界点该目标贡献 ``inf``；
        内部点贡献 ``(f_next - f_prev) / (f_max - f_min)``，各目标贡献相加。
        因此 ``n <= 2`` 时全部为 ``inf``；n > 2 且存在非退化目标时，
        那些在某个非退化目标上处于端点位置的解为 ``inf``。

    算法:
        1. 对每个目标 k，用 ``argsort`` 排序；
        2. 若该目标取值域为 0（前沿在该目标上完全退化），**整个目标跳过**（贡献 0），
           既不记边界 ``inf``，也不做内部的除法——避免除零，也避免把 ``inf``
           白送给两个按并列顺序排出来的任意点；
        3. 否则边界点记 ``inf``，内部点累加
           ``(f[order[i+1], k] - f[order[i-1], k]) / (f_max - f_min)``。

    复杂度:
        时间 O(m * n log n) / 空间 O(n)。

    陷阱:
        1. **必须按前沿分别调用**。把整份种群的目标矩阵直接丢进来会在不同前沿之间
           比较距离，得到的数没有任何意义（这也是常见误用）。
        2. 距离已经按各目标的取值范围归一化，所以量纲不同的目标可以直接相加；
           但取值范围为 0 的目标必须整体跳过（本函数已处理），否则 0/0 会污染整条前沿，
           或者把边界 ``inf`` 记到并列点上、让任意解被误判为边界解。
        3. 若前沿中有重复点（目标值完全相同），它们的距离会互相"顶掉"对方，
           表现为一大一小，不代表真实的多样性——去重后计算更可靠。

    参考:
        Deb et al. 2002（NSGA-II 原文第 III-B 节）。
    """
    M = as_matrix(F, "F")
    n, m = M.shape
    dist = np.zeros(n, dtype=float)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for k in range(m):
        order = np.argsort(M[:, k], kind="stable")
        vals = M[order, k]
        span = float(vals[-1] - vals[0])
        # 退化目标（取值域为 0）不贡献任何距离，也不能把边界 inf 记到两个任意的
        # 并列点上，否则这些点会被错误地当成"边界受保护"，从而躲过 NSGA-II 的截断。
        if span <= 0.0:
            continue
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        dist[order[1:-1]] += (vals[2:] - vals[:-2]) / span
    return dist


# --------------------------------------------------------------------------- #
# 加权和法
# --------------------------------------------------------------------------- #
def weighted_sum_pareto(
    objective_fn: Callable[[np.ndarray], ArrayLike],
    x_bounds,
    n_weights: int = 11,
    seed: Optional[int] = None,
    minimize: bool = True,
) -> dict:
    """加权和法扫描 Pareto 前沿：对单纯形网格上的每个权重求加权和的极小点。

    参数:
        objective_fn: ``objective_fn(x) -> np.ndarray``，目标向量，形状 (m,)。
        x_bounds: 决策变量盒约束。``(lo, hi)`` 两个标量表示 **1 维**问题；
            要 d 维共用同一区间写 ``[(lo, hi)] * d``；逐维区间写 ``[(lo1, hi1), ...]``。
        n_weights: 每个权重分量上的格点数，>= 2。权重网格为
            ``{w >= 0, sum(w) = 1, w_i = k_i / (n_weights - 1)}``，共
            ``C(n_weights - 1 + m - 1, m - 1)`` 个权重（m=2 时恰为 ``n_weights`` 个）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        minimize: True（默认）表示所有目标都取最小；False 表示全部取最大。

    返回:
        dict，键为：
        ``weights``  形状 (n_grid, m) 的权重矩阵（每行和为 1）；
        ``X``        形状 (n_grid, n_dim) 的决策向量，第 i 行对应 ``weights[i]``；
        ``F``        形状 (n_grid, m) 的目标值，**是 objective_fn 的原始输出**
                     （``minimize=False`` 时没有取负），非支配关系是在取负后的
                     最小化口径上定义的。

    算法:
        1. 采样一次 ``objective_fn`` 得到目标个数 m，用 :func:`_simplex_grid` 生成权重网格；
        2. 对每个权重 w，最小化标量 ``w @ (sign * objective_fn(x))``（``sign = -1``
           表示最大化），用 :func:`_pattern_search` 做多起点自适应随机方向搜索
           （目标函数不可导、非凸也照样能跑）；
        3. 把每个权重的极小点与原始目标值一并返回。

    复杂度:
        时间 O(n_grid * n_starts * n_iter * (m + T_objective))，本实现默认
        ``n_starts=4``、``n_iter=120``；空间 O(n_grid * (n_dim + m))。

    陷阱:
        1. **加权和法只能取到凸前沿**。当前沿非凸（例如 ZDT 系列、或两个目标都是
           "凹"的形状）时，无论权重怎么取，中间那段非凸前沿上的解**永远取不到**。
           这是加权和法的原理性缺陷，不是搜索不够努力；判断方法：画出的点在图上有
           "空洞"，或者点的法线方向与权重明显不匹配。
        2. 权重的尺度决定了解的分辨率。目标量纲相差几个数量级时（例如一个目标 ~1e-3、
           另一个 ~1e6），等权网格几乎只优化量级大的那个目标，必须**先把目标归一化**。
           本函数不做归一化，因为归一化的口径（除以理想点？除以极差？）会改变前沿形状，
           应当由使用者在论文里显式交代。
        3. 权重分量取到 0 时对应目标被完全忽略，此时返回的"解"在该目标上可能极差，
           但仍然是该权重下的正确最优——读图时不要把这类极端点当作异常值删掉。
        4. 局部搜索不保证全局最优，非凸、多峰的标量化问题可能卡在局部极小；
           本实现靠多起点缓解，但仍应在论文里报告 n_starts 与求值次数。

    参考:
        Zadeh 1963（加权和标量化）；Miettinen 1999《Nonlinear Multiobjective Optimization》。
    """
    if not callable(objective_fn):
        raise ValueError("objective_fn 必须是可调用对象")
    n_weights = int(n_weights)
    if n_weights < 2:
        raise ValueError(f"n_weights 至少为 2，得到 {n_weights}")
    lo, hi = _parse_box(x_bounds)
    gen = make_rng(seed)
    sign = 1.0 if minimize else -1.0

    m = _eval_vec(objective_fn, 0.5 * (lo + hi)).size
    W = _simplex_grid(m, n_weights)
    n_grid = W.shape[0]
    X = np.empty((n_grid, lo.size), dtype=float)
    F = np.empty((n_grid, m), dtype=float)

    for i in range(n_grid):
        w = W[i]

        def _scalar(x: np.ndarray, _w: np.ndarray = w) -> float:
            return float(_w @ (sign * _eval_vec(objective_fn, x)))

        x_best, _ = _pattern_search(_scalar, lo, hi, gen)
        X[i] = x_best
        F[i] = _eval_vec(objective_fn, x_best)
    return {"weights": W, "X": X, "F": F}


# --------------------------------------------------------------------------- #
# ε-约束法
# --------------------------------------------------------------------------- #
def epsilon_constraint_pareto(
    objective_fn: Callable[[np.ndarray], ArrayLike],
    x_bounds,
    n_grid: int = 11,
    seed: Optional[int] = None,
    minimize: bool = True,
) -> dict:
    """ε-约束法扫描 Pareto 前沿：把第 1 个目标变成 ε 上限约束，最小化其余目标。

    参数:
        objective_fn: ``objective_fn(x) -> np.ndarray``，目标向量，形状 (m,)。
        x_bounds: 决策变量盒约束，口径同 :func:`weighted_sum_pareto`。
        n_grid: ε 网格点数，>= 2。ε 的取值范围由 200 个随机样本上第 1 个目标的
            最小/最大值线性插值得到（固定 seed 下确定）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        minimize: True（默认）全部取最小；False 全部取最大。

    返回:
        dict，键为：
        ``epsilons``    形状 (n_grid,) 的 ε 取值（第 1 个目标的**最小化口径**上限）；
        ``X``           形状 (n_grid, n_dim) 的决策向量；
        ``F``           形状 (n_grid, m) 的目标值（objective_fn 的原始输出）；
        ``infeasible``  长度 n_grid 的 bool 列表，第 i 项表示该 ε 下返回的解是否
                        违反 ``f_1 <= eps_i``（容差 ``1e-6 * max(1, |eps_i|)``）。

    算法:
        1. 随机采样估计第 1 个目标在盒约束下的取值范围，生成 n_grid 个 ε；
        2. 对每个 ε 最小化 ``mean(sign * f[1:])``（m == 1 时退化为最小化 f_1 本身），
           罚函数为 ``P * max(0, sign * f_1 - eps)``，``P = 1e3 * max(采样极差, 1)``
           （下界 1 是为了防止"目标 1 在盒约束下几乎不变"时罚系数退化成 0）；
        3. 用同一个 :func:`_pattern_search` 求解，并回代检查可行性。

    复杂度:
        时间 O(n_grid * n_starts * n_iter * (m + T_objective) + n_probe * T_objective)；
        空间 O(n_grid * (n_dim + m))。

    陷阱:
        1. 罚函数法会给出**近似可行**的解：罚系数 P 太小会停在不可行侧（``infeasible``
           为 True），太大则数值上接近硬约束、搜索几乎无法从不可行区往里走。
           本实现的 P 由采样极差自动标定，换问题后仍建议检查 ``infeasible``。
        2. "最小化其余目标的等权平均"在其余目标**量纲不同**时会偏向量级大的那个，
           与加权和法有同样的隐患，使用前应先把目标归一化。
        3. ε 的网格由随机采样范围决定：如果采样完全错过第 1 个目标的极端值，
           网格会偏窄，前沿两端取不到。采样数固定为 200，高维问题下可能不够。
        4. ε-约束法**可以**取到非凸前沿上的点（这正是它相对加权和法的价值），
           但每个 ε 仍是一次独立的非凸优化，局部极小会让相邻 ε 的结果跳变。

    参考:
        Haimes, Lasdon & Wismer 1971（ε-约束法）；Chankong & Haimes 1983。
    """
    if not callable(objective_fn):
        raise ValueError("objective_fn 必须是可调用对象")
    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError(f"n_grid 至少为 2，得到 {n_grid}")
    lo, hi = _parse_box(x_bounds)
    gen = make_rng(seed)
    sign = 1.0 if minimize else -1.0

    m = _eval_vec(objective_fn, 0.5 * (lo + hi)).size
    n_probe = 200
    probe = lo + gen.random((n_probe, lo.size)) * (hi - lo)
    g1_probe = np.array([sign * float(_eval_vec(objective_fn, p)[0]) for p in probe])
    lo_e = float(g1_probe.min())
    hi_e = float(g1_probe.max())
    eps_grid = np.linspace(lo_e, hi_e, n_grid)
    penalty = 1e3 * max(hi_e - lo_e, 1.0)

    X = np.empty((n_grid, lo.size), dtype=float)
    F = np.empty((n_grid, m), dtype=float)
    infeasible: List[bool] = []
    for i in range(n_grid):
        eps = float(eps_grid[i])

        def _scalar(x: np.ndarray, _eps: float = eps) -> float:
            f = sign * _eval_vec(objective_fn, x)
            body = float(f[0]) if m == 1 else float(np.mean(f[1:]))
            return body + penalty * max(0.0, float(f[0]) - _eps)

        x_best, _ = _pattern_search(_scalar, lo, hi, gen)
        X[i] = x_best
        F[i] = _eval_vec(objective_fn, x_best)
        g1 = sign * float(F[i, 0])
        infeasible.append(bool(g1 > eps + 1e-6 * max(1.0, abs(eps))))
    return {"epsilons": eps_grid, "X": X, "F": F, "infeasible": infeasible}


# --------------------------------------------------------------------------- #
# NSGA-II
# --------------------------------------------------------------------------- #
def nsga2(
    objective_fn: Callable[[np.ndarray], ArrayLike],
    x_bounds,
    pop_size: int = 40,
    n_gen: int = 60,
    seed: Optional[int] = None,
    minimize: bool = True,
    mutation_sigma: Optional[float] = None,
) -> dict:
    """实数编码 NSGA-II：非支配排序 + 拥挤距离 + 二元锦标赛 + SBX 交叉 + 多项式变异。

    参数:
        objective_fn: ``objective_fn(x) -> np.ndarray``，目标向量，形状 (m,)。
        x_bounds: 决策变量盒约束，口径同 :func:`weighted_sum_pareto`。
        pop_size: 种群规模，>= 2。
        n_gen: 迭代代数，>= 0（0 表示只评估初始种群）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        minimize: True（默认）全部取最小；False 全部取最大（内部统一取负）。
        mutation_sigma: 若为 None 用标准**多项式变异**（分布指数 eta_m = 20，
            每个分量以 1/n_dim 的概率变异）；若给定正数，则改用标准差为
            ``mutation_sigma * 区间宽度`` 的高斯变异（便于做消融实验）。

    返回:
        dict，键为：
        ``X``       形状 (pop_size, n_dim) 的最终种群决策向量；
        ``F``       形状 (pop_size, m) 的目标值（objective_fn 的原始输出）；
        ``front``   最终种群第 0 层前沿在 ``X``/``F`` 中的下标列表（升序）；
        ``history`` 长度 ``n_gen + 1`` 的 int 列表，第 k 项是**第 k 代种群的
                    第一前沿规模**（不是超体积；用规模是因为它对本实现更稳定）。

    算法:
        1. 在盒约束内均匀随机初始化 ``pop_size`` 个个体，求目标值；
        2. 每一代：对当前种群做非支配排序 → 按前沿分别算拥挤距离 →
           二元锦标赛（先比前沿序号、同前沿比拥挤距离）选父代 → SBX 交叉
           （eta_c = 15，交叉概率 0.9）→ 多项式变异（eta_m = 20）→ 生成等规模子代；
        3. 父代 + 子代合并（2N）后重新非支配排序，按前沿序号依次填充下一代；
           最后一个放不下的前沿按拥挤距离降序截断（这就是 NSGA-II 的精英保留）；
        4. 每代记录第一前沿规模。

    复杂度:
        时间 O(n_gen * (pop_size^2 * m + pop_size * m log pop_size + pop_size * T_objective))；
        空间 O(pop_size * (n_dim + m))。

    陷阱:
        1. ``history`` 记录的是**第一前沿规模**，它在这个实现里几乎总是单调不减，
           但理论上并不保证（子代可能一次性支配掉好几个父代前沿点，使前沿收缩），
           所以不要把它当成"收敛性定理"来引用，也不要用它判断是否早熟。
        2. 边界处理用的是**简单截断**（``np.clip``），不是原论文 SBX 的边界修正公式。
           当最优解恰好在盒边界上时，截断会造成边界附近解密度偏高，属于已知偏差。
        3. 交叉/变异算子之间是**乘性耦合**：交叉概率 0.9 加上"每个分量 1/n_dim 概率变异"，
           在低维问题（n_dim <= 3）上变异强度其实不小，种群很难稳定收敛到一条细前沿;
           需要更精细的前沿时，应降低变异概率或改用 ``mutation_sigma`` 小步长。
        4. 没有实现约束支配（constrained-domination）：带约束的问题必须自己把约束写进
           ``objective_fn``（罚函数），否则 NSGA-II 会稳定地输出不可行解。
        5. 目标个数 m >= 3 时拥挤距离的多样性保持能力骤降，应按 NSGA-III / MOEA-D
           的思路改造，本函数只适合 m = 2 或 3。

    参考:
        Deb, Pratap, Agarwal & Meyarivan 2002, IEEE TEC 6(2)（NSGA-II）；
        Deb & Agrawal 1995（SBX）；Deb & Goyal 1996（多项式变异）。
    """
    if not callable(objective_fn):
        raise ValueError("objective_fn 必须是可调用对象")
    pop_size = int(pop_size)
    n_gen = int(n_gen)
    if pop_size < 2:
        raise ValueError(f"pop_size 至少为 2，得到 {pop_size}")
    if n_gen < 0:
        raise ValueError(f"n_gen 不能为负，得到 {n_gen}")
    if mutation_sigma is not None:
        mutation_sigma = float(mutation_sigma)
        if not np.isfinite(mutation_sigma) or mutation_sigma <= 0.0:
            raise ValueError(f"mutation_sigma 必须为正的有限值，得到 {mutation_sigma}")

    lo, hi = _parse_box(x_bounds)
    gen = make_rng(seed)
    sign = 1.0 if minimize else -1.0
    m = _eval_vec(objective_fn, 0.5 * (lo + hi)).size
    eta_c, eta_m, p_cross = 15.0, 20.0, 0.9

    pop = lo + gen.random((pop_size, lo.size)) * (hi - lo)
    F = _eval_pop(objective_fn, pop)

    history: List[int] = []
    fronts: List[List[int]] = [[]]
    for generation in range(n_gen + 1):
        res = fast_non_dominated_sort(sign * F)
        fronts = res["fronts"]
        rank = res["rank"]
        crowd = np.zeros(pop_size, dtype=float)
        for front in fronts:
            idx = np.asarray(front, dtype=int)
            crowd[idx] = crowding_distance((sign * F)[idx])
        history.append(len(fronts[0]))
        if generation == n_gen:
            break

        children = np.empty_like(pop)
        k = 0
        while k < pop_size:
            i1 = _crowded_tournament(rank, crowd, gen)
            i2 = _crowded_tournament(rank, crowd, gen)
            c1, c2 = _sbx(pop[i1], pop[i2], lo, hi, gen, eta_c, p_cross)
            children[k] = _poly_mutation(c1, lo, hi, gen, eta_m, mutation_sigma)
            if k + 1 < pop_size:
                children[k + 1] = _poly_mutation(c2, lo, hi, gen, eta_m, mutation_sigma)
            k += 2

        F_child = _eval_pop(objective_fn, children)
        comb = np.vstack([pop, children])
        F_comb = np.vstack([F, F_child])
        res2 = fast_non_dominated_sort(sign * F_comb)

        keep: List[int] = []
        for front in res2["fronts"]:
            if len(keep) + len(front) <= pop_size:
                keep.extend(front)
                continue
            need = pop_size - len(keep)
            cf = crowding_distance((sign * F_comb)[np.asarray(front, dtype=int)])
            order = np.argsort(-cf, kind="stable")
            keep.extend(int(front[t]) for t in order[:need])
            break
        sel = np.asarray(keep, dtype=int)
        pop = comb[sel].copy()
        F = F_comb[sel].copy()

    return {"X": pop, "F": F, "front": [int(i) for i in fronts[0]], "history": history}


# --------------------------------------------------------------------------- #
# 基于分解的多目标进化（MOEA/D）
# --------------------------------------------------------------------------- #
def moead(
    objective_fn: Callable[[np.ndarray], ArrayLike],
    x_bounds,
    n_partitions: int = 12,
    n_iter: int = 100,
    neighborhood_size: Optional[int] = None,
    seed: Optional[int] = None,
    minimize: bool = True,
    mutation_sigma: Optional[float] = None,
) -> dict:
    """MOEA/D：把多目标问题分解成一组**切比雪夫（Tchebycheff）标量化子问题**同时进化。

    参数:
        objective_fn: ``objective_fn(x) -> np.ndarray``，目标向量，形状 (m,)。
        x_bounds: 决策变量盒约束，口径同 :func:`weighted_sum_pareto`。
        n_partitions: 单纯形权重的划分份数，>= 1。权重个数（即子问题个数）为
            ``C(n_partitions + m - 1, m - 1)``，随目标个数 m 组合增长——m = 3 且
            n_partitions = 12 时是 91 个子问题，注意求值预算。
        n_iter: 进化代数，>= 0（0 表示只评估初始种群；初始种群本身也算一次求值）。
        neighborhood_size: 邻域大小 T。None 表示 ``max(2, ceil(0.1 * 子问题个数))``。
            邻域按**权重空间**的欧氏距离取最近 T 个权重。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        minimize: True（默认）全部取最小；False 全部取最大（内部统一取负）。
        mutation_sigma: 若为 None 用 ``0.1 * 区间宽度`` 的高斯变异；若给定正数，则用
            ``mutation_sigma * 区间宽度``（口径与 :func:`nsga2` 一致，便于做消融实验）。

    返回:
        dict，键为：
        ``X``        形状 (n_sub, n_dim) 的最终种群决策向量（第 i 行是第 i 个子问题的解）；
        ``F``        形状 (n_sub, m) 的目标值（``objective_fn`` 的**原始输出**）；
        ``G``        形状 (n_sub, m) 的**内部最小化空间**目标值（``minimize=False`` 时为 -F）；
        ``weights``  形状 (n_sub, m) 的单纯形权重网格，每行和为 1；
        ``ideal``    形状 (m,) 的理想点，在**内部最小化空间** ``G`` 上（``minimize=True``
                     时它就是 F 的逐分量最小值；``minimize=False`` 时它是 -F 的逐分量
                     最小值）。注意是所有**求值过**的点上的最小值，**包含未被任何子问题
                     接受的子代**，因此它可能严格优于最终种群的逐分量最小值；
        ``front``    最终种群第 0 层前沿的下标列表（升序），基于 ``G`` 分层；
        ``history``  长度 ``n_iter + 1`` 的 int 列表，第 k 项是**第 k 代种群的
                     第一前沿规模**（口径与 :func:`nsga2` 一致，便于两条路径对比）；
        ``n_weights``        子问题个数；
        ``neighborhood_size`` 实际使用的邻域大小 T。

    算法:
        1. 试探性求值一次得到目标个数 m，用 ``_simplex_grid(m, n_partitions + 1)`` 生成
           权重网格，并对每个权重取权重空间最近的 T 个下标构成邻域；
        2. 均匀随机初始化 n_sub 个个体（**每个子问题一个解**），求目标值，初始化理想点
           ``z = min(G)``；
        3. 每一代：随机排列所有子问题；对子问题 i，在其邻域内随机取三个不同个体
           （允许重复）做 DE/rand/1 型差分交叉 ``child = x_r1 + 0.5 (x_r2 - x_r3)``，
           再按 ``1/n_dim`` 的概率逐分量做高斯变异，投影回盒约束；
        4. 求子代目标值，用 ``z = min(z, G_child)`` 更新理想点；
        5. 用切比雪夫函数 ``g(x; w_j) = max_k w_jk (G_k - z_k)`` 在 i 的**整个邻域**内做
           贪心替换：若 ``g(child; w_j) <= g(X_j; w_j)`` 就用子代替换第 j 个子问题的解；
        6. 每代记录第一前沿规模。

    复杂度:
        时间 O(n_iter * n_sub * (T * m + T_objective))，
        空间 O(n_sub * (n_dim + m) + n_sub^2)（邻域矩阵）。

    陷阱:
        1. 邻域替换是"按**自己的权重**做贪心"，**不保证** Pareto 支配意义上的单调改进：
           一个解可能在自己的标量化子问题上变好、却把邻居的多样性顶掉。所以 MOEA/D 的
           前沿规模会忽大忽小，不能用它判断收敛——要判断收敛请用超体积或 IGD。
        2. 切比雪夫函数里的理想点 z 每代都在下降，而 z 下降会让**已有解**的 g 值上升。
           因此不要把 g 值当成"单调收敛曲线"记录下来。另外 ``ideal`` 是**所有求值过**的
           点（含被拒绝的子代）上的最小值，通常严格优于 ``F.min(axis=0)``；
           想报"种群达到的理想点"请自己用 ``F.min(axis=0)`` 算，不要直接用 ``ideal``。
        3. 权重网格是**等距格点**，不是均匀分布在单纯形上（越靠边的权重越稀疏）。
           目标个数 m >= 4 时格点密度会严重不均，需要换成两层的 Das-Dennis 构造
           （``n_partitions`` 外层/内层不同），本实现没有做。
        4. 这是"每个子问题保一个解"的分解式算法，种群规模由 ``n_partitions`` 决定而不是
           由调用者直接指定——想要固定种群规模请用 :func:`nsga2`。
        5. 变异强度默认是区间宽度的 0.1 倍，在**窄区间**或高精度问题上会过大；
           高精度需求请显式传小的 ``mutation_sigma``。

    参考:
        Zhang & Li 2007, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
        Decomposition", IEEE TEC 11(6)；Li & Zhang 2009（切比雪夫分解的进一步分析）。
    """
    lo, hi = _parse_box(x_bounds)
    if isinstance(n_partitions, bool) or not isinstance(n_partitions, (int, np.integer)):
        raise ValueError(f"n_partitions 必须是整数，得到 {n_partitions!r}")
    if int(n_partitions) < 1:
        raise ValueError(f"n_partitions 必须 >= 1，得到 {n_partitions}")
    if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)):
        raise ValueError(f"n_iter 必须是整数，得到 {n_iter!r}")
    if int(n_iter) < 0:
        raise ValueError(f"n_iter 必须 >= 0，得到 {n_iter}")
    if not isinstance(minimize, (bool, np.bool_)):
        raise ValueError(f"minimize 必须是布尔值，得到 {minimize!r}")
    if neighborhood_size is not None:
        if isinstance(neighborhood_size, bool) or not isinstance(
            neighborhood_size, (int, np.integer)
        ):
            raise ValueError(f"neighborhood_size 必须是整数或 None，得到 {neighborhood_size!r}")
        if int(neighborhood_size) < 1:
            raise ValueError(f"neighborhood_size 必须 >= 1，得到 {neighborhood_size}")
    if mutation_sigma is not None:
        sigma_frac = float(mutation_sigma)
        if not np.isfinite(sigma_frac) or sigma_frac <= 0.0:
            raise ValueError(f"mutation_sigma 必须是正有限数，得到 {mutation_sigma!r}")
    else:
        sigma_frac = 0.1

    gen = make_rng(seed)
    d = int(lo.size)
    span = hi - lo
    sign = 1.0 if bool(minimize) else -1.0
    n_parts = int(n_partitions)
    n_it = int(n_iter)

    m = int(_eval_vec(objective_fn, lo + gen.random(d) * span).size)
    weights = _simplex_grid(m, n_parts + 1)
    n_sub = int(weights.shape[0])
    if neighborhood_size is None:
        t_size = max(2, min(n_sub, int(np.ceil(0.1 * n_sub))))
    else:
        t_size = min(int(neighborhood_size), n_sub)

    # 权重空间的邻域：L2 距离最近的 T 个权重（含自身）
    w_diff = weights[:, None, :] - weights[None, :, :]
    w_dist = np.sqrt(np.einsum("ijk,ijk->ij", w_diff, w_diff))
    neighbors = np.argsort(w_dist, axis=1, kind="stable")[:, :t_size]

    X = lo + gen.random((n_sub, d)) * span
    F = _eval_pop(objective_fn, X)
    if F.shape[1] != m:
        raise ValueError(f"objective_fn 返回的目标个数不固定：{m} 与 {F.shape[1]}")
    G = sign * F
    ideal = G.min(axis=0).copy()
    sigma = sigma_frac * span

    def _tchebycheff(g_vec: np.ndarray, w: np.ndarray) -> float:
        return float(np.max(w * (g_vec - ideal)))

    history: List[int] = [len(fast_non_dominated_sort(G)["fronts"][0])]
    for _ in range(n_it):
        for raw_i in gen.permutation(n_sub):
            i = int(raw_i)
            neigh = neighbors[i]
            pick = gen.choice(neigh, size=3, replace=True)
            child = X[int(pick[0])] + 0.5 * (X[int(pick[1])] - X[int(pick[2])])
            mask = gen.random(d) < (1.0 / float(d))
            child = child + mask * gen.standard_normal(d) * sigma
            child = np.clip(child, lo, hi)
            f_child = _eval_vec(objective_fn, child)
            g_child = sign * f_child
            ideal = np.minimum(ideal, g_child)
            for raw_j in neigh:
                j = int(raw_j)
                if _tchebycheff(g_child, weights[j]) <= _tchebycheff(G[j], weights[j]):
                    X[j] = child
                    F[j] = f_child
                    G[j] = g_child
        history.append(len(fast_non_dominated_sort(G)["fronts"][0]))

    return {
        "X": X,
        "F": F,
        "G": G,
        "weights": weights,
        "ideal": ideal,
        "front": [int(i) for i in fast_non_dominated_sort(G)["fronts"][0]],
        "history": history,
        "n_weights": n_sub,
        "neighborhood_size": int(t_size),
    }


# --------------------------------------------------------------------------- #
# 前沿提取 / 超体积 / 理想点距离
# --------------------------------------------------------------------------- #
def pareto_front(F: MatrixLike) -> dict:
    """从目标矩阵里抽出非支配解（等价于 :func:`fast_non_dominated_sort` 的第 0 层）。

    参数:
        F: 目标矩阵，形状 (n_solutions, n_objectives)，全部按最小化理解。

    返回:
        dict，键为：
        ``index`` 第 0 层前沿的解下标列表（升序）；
        ``F``     与 ``index`` 对应的目标子矩阵，形状 (len(index), m)。

    算法:
        直接调用 :func:`fast_non_dominated_sort` 并取 ``fronts[0]``，保证两条代码路径
        的口径完全一致（自测里会把两者对拍）。

    复杂度:
        时间 O(n^2 * m) / 空间 O(n^2)。

    陷阱:
        1. 返回的下标是**原始矩阵中的位置**，不是排序后的位置；如果你先把 F 排序再调用，
           拿到的下标必须在原数组上重新映射，否则会取错解。
        2. **不做去重**：目标值完全相同的重复解会全部出现在 ``index`` 里（互相不支配）。
           下游若要做多样性/超体积统计，先自己 ``np.unique`` 或按目标值去重。

    参考:
        Deb et al. 2002。
    """
    M = as_matrix(F, "F")
    res = fast_non_dominated_sort(M)
    idx = res["fronts"][0]
    return {"index": [int(i) for i in idx], "F": M[idx].copy()}


def hypervolume_2d(F: MatrixLike, reference: ArrayLike) -> float:
    """二维超体积：被解集支配、且被参考点界住的面积（全部最小化）。

    参数:
        F: 目标矩阵，形状 (n, 2)。**只支持两个目标**（高维超体积需要另一套算法）。
        reference: 长度 2 的参考点，必须在"最差侧"，即每个分量都不小于前沿上的对应值。
            一般取 前沿各目标的 max 再乘 1.1 之类的松弛值。

    返回:
        float，≥ 0。区域定义为 ``{y : 存在 i 使 F[i] <= y <= reference}`` 的面积，
        因此**被 reference 支配的解**（某个分量 >= reference 的对应分量）不计入。

    算法:
        1. 丢掉 ``F[i, 0] >= r[0]`` 或 ``F[i, 1] >= r[1]`` 的点；
        2. 剩下的点按第 1 个目标升序排序；
        3. 沿第 1 个目标扫描：维护已扫描点的最小第 2 个目标值 ``best``，
           第 i 个点贡献 ``(x_{i+1} - x_i) * max(0, r_1 - best)``，最后一段的右边界是 r[0]。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. **参考点必须固定**才能跨代比较超体积。换参考点会让超体积数值失去可比性，
           收敛曲线也就没有意义了——这是用超体积画收敛图时最常犯的错。
        2. 参考点取得太靠近前沿时，所有解的超体积都接近 0，曲线会被舍入噪声淹没；
           取得太远则退化成"矩形面积"，区分度同样下降。
        3. 被 reference 支配的点贡献 0 而不是负数，因此**超体积不会惩罚越界解**：
           如果算法的输出跑到参考点之外，超体积反而"看起来"没变差。
        4. 只支持二维；三个目标以上必须用快速超体积算法，不能把这里的结果外推。

    参考:
        Zitzler & Thiele 1999（超体积指标）；Fonseca, Paquete & López-Ibáñez 2006（快速算法）。
    """
    M = as_matrix(F, "F")
    if M.shape[1] != 2:
        raise ValueError(f"hypervolume_2d 只支持二维目标，得到 {M.shape[1]} 个目标")
    r = as_vector(reference, "reference")
    if r.size != 2:
        raise ValueError(f"reference 必须是长度 2 的向量，得到长度 {r.size}")

    keep = (M[:, 0] < r[0]) & (M[:, 1] < r[1])
    pts = M[keep]
    if pts.shape[0] == 0:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    area = 0.0
    best = np.inf
    n = pts.shape[0]
    for i in range(n):
        best = min(best, float(pts[i, 1]))
        x_next = float(pts[i + 1, 0]) if i + 1 < n else float(r[0])
        width = x_next - float(pts[i, 0])
        if width <= 0.0:
            continue
        height = float(r[1]) - best
        if height > 0.0:
            area += width * height
    return float(area)


def ideal_point_distance(F: MatrixLike, weights: Optional[ArrayLike] = None) -> dict:
    """到正/负理想点的加权距离与贴近度（TOPSIS 口径，全部最小化）。

    参数:
        F: 目标矩阵，形状 (n, m)。
        weights: 长度 m 的非负权重；None 表示等权。内部会归一化到和为 1
            （归一化不改变 ``closeness``，只改变 ``d_plus`` / ``d_minus`` 的量级）。

    返回:
        dict，键为：
        ``d_plus``   形状 (n,) 的数组，到**正理想点**（各目标最小值构成的点）的距离；
        ``d_minus``  形状 (n,) 的数组，到**负理想点**（各目标最大值构成的点）的距离；
        ``closeness`` 形状 (n,) 的数组，``d_minus / (d_plus + d_minus)``，越大越好。
                      ``d_plus + d_minus == 0``（只有一个解、或所有解目标值相同）时
                      按约定取 0.5。

    算法:
        ``ideal_j = min_i F[i, j]``，``anti_j = max_i F[i, j]``，
        ``d_plus_i = sqrt(sum_j w_j (F[i,j] - ideal_j)^2)``，
        ``d_minus_i = sqrt(sum_j w_j (F[i,j] - anti_j)^2)``。
        即权重乘在**偏差平方**上，等价于 TOPSIS 里"先加权、再算欧氏距离"的做法。

    复杂度:
        时间 O(n m) / 空间 O(n)。

    陷阱:
        1. 理想点由**当前解集自身**决定，所以加入或删除一个解就会改变所有解的贴近度。
           不同算法、不同种群规模的结果**不能直接比较**，必须先合并到一个公共解集上算。
        2. 这里没有做向量归一化。TOPSIS 的标准流程会先对指标矩阵做范数归一化，
           本函数把归一化留给调用者（口径见 ``_common.normalize_l2``），
           因此量纲差异大的目标会直接主导距离。
        3. 贴近度是"相对排名"工具，不能解释成"距离最优解还有多远"的绝对量。

    参考:
        Hwang & Yoon 1981《Multiple Attribute Decision Making》（TOPSIS）。
    """
    M = as_matrix(F, "F")
    n, m = M.shape
    if weights is None:
        w = np.full(m, 1.0 / m)
    else:
        w = as_vector(weights, "weights")
        if w.size != m:
            raise ValueError(f"weights 长度必须等于目标个数 {m}，得到 {w.size}")
        if np.any(w < 0):
            raise ValueError("weights 不能含负值")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("weights 之和必须为正")
        w = w / total

    ideal = M.min(axis=0)
    anti = M.max(axis=0)
    d_plus = np.sqrt(((M - ideal) ** 2 * w).sum(axis=1))
    d_minus = np.sqrt(((M - anti) ** 2 * w).sum(axis=1))
    denom = d_plus + d_minus
    closeness = np.divide(
        d_minus, denom, out=np.full(n, 0.5, dtype=float), where=denom > 0
    )
    return {"d_plus": d_plus, "d_minus": d_minus, "closeness": closeness}


# --------------------------------------------------------------------------- #
# 前沿质量评价
# --------------------------------------------------------------------------- #
def igd_metric(F: MatrixLike, reference_front: MatrixLike) -> dict:
    """IGD（Inverted Generational Distance）：**参考前沿**上的每个点到近似集的最小距离的平均。

    参数:
        F: 待评价的近似前沿目标矩阵，形状 (n_approx, m)，全部按最小化理解。
        reference_front: 参考前沿（真实前沿的采样或理论前沿的离散点），形状 (n_ref, m)，
            目标个数 m 必须与 F 一致。

    返回:
        dict，键为：
        ``igd``           float，``(1 / n_ref) * sum_j min_i ||R_j - F_i||_2``，越小越好，
                          0 表示参考前沿上的每个点都被近似集"贴上"了；
        ``nearest``       形状 (n_ref,) 的数组，即被平均的那组数，按参考点的顺序给出，
                          用来定位"参考前沿的哪一段没被覆盖"；
        ``mean_nearest``  等价于 ``igd``（单独给出便于阅读）；
        ``worst_nearest`` 最大的那个最近距离，即最差的一段覆盖；
        ``n_reference``   参考点个数；
        ``n_approx``      近似集大小。

    算法:
        1. 广播计算所有 (i, j) 组合的欧氏距离矩阵，形状 (n_approx, n_ref)；
        2. 对每个参考点 j 取 ``min_i`` 得到 ``nearest[j]``；
        3. ``igd = mean(nearest)``。

    复杂度:
        时间 O(n_approx * n_ref * m) / 空间 O(n_approx * n_ref)（距离矩阵是内存瓶颈）。

    陷阱:
        1. 方向不要搞反。**对参考集逐点取最近**是 IGD（衡量"覆盖度"）；对近似集逐点取最近
           再平均是 GD（衡量"贴近度"）。只报告 GD 会漏掉"前沿只覆盖了一小段但每点都很准"
           这种最典型的失败模式，而只报告 IGD 会漏掉"覆盖很全但有一堆远离前沿的杂点"。
           两者配合看才有意义。
        2. IGD 的数值强烈依赖**参考前沿的采样密度和分布**。拿 100 个点采出来的参考前沿和
           拿 1000 个点采出来的结果不可比；不同论文的 IGD 数字不比较就是这个原因。
        3. 这里没做目标量纲归一化。量纲大（例如某个目标是成本、量级 1e6）的目标会直接
           主导平均距离，评价前应先把 F 与 reference_front 放到同一尺度上（例如一起做
           min-max 归一化）。
        4. 参考前沿必须与近似集**同维**；F 为空、维数不一致都会抛 ``ValueError``。

    参考:
        Van Veldhuizen & Lamont 1998（Generational Distance / IGD）；Zitzler et al. 2003。
    """
    A = as_matrix(F, "F")
    R = as_matrix(reference_front, "reference_front")
    if A.shape[1] != R.shape[1]:
        raise ValueError(
            f"F 与 reference_front 的目标个数必须一致，得到 {A.shape[1]} 与 {R.shape[1]}"
        )
    diff = A[:, None, :] - R[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    nearest = dist.min(axis=0)
    return {
        "igd": float(nearest.mean()),
        "nearest": nearest,
        "mean_nearest": float(nearest.mean()),
        "worst_nearest": float(nearest.max()),
        "n_reference": int(R.shape[0]),
        "n_approx": int(A.shape[0]),
    }


def spacing_metric(F: MatrixLike, metric: str = "l1") -> dict:
    """Spacing（Schott 间距）：前沿上每个点到**其余点**最近距离的标准差，衡量分布均匀性。

    参数:
        F: 目标矩阵，形状 (n, m)，应当是**同一条前沿**上的解（全部最小化）。
        metric: ``"l1"``（默认，Schott 原文口径，曼哈顿距离）或 ``"l2"``（欧氏距离）。

    返回:
        dict，键为：
        ``spacing``       float，``sqrt( sum_i (d_bar - d_i)^2 / (n - 1) )``，其中
                          ``d_i`` 是第 i 个点到其余点的最近距离、``d_bar`` 是它们的均值。
                          **完全均匀时恰为 0**，越大越不均匀；``n == 1`` 时按约定取 0.0
                          （只有一个点时"均匀性"无定义，为 0 是为了让它在聚合指标里不起作用）；
        ``nearest``       形状 (n,) 的数组，即 ``d_i``；``n == 1`` 时为 ``[0.0]``；
        ``mean_nearest``  ``d_bar``；
        ``n``             解的个数；
        ``metric``        实际使用的距离口径。

    算法:
        1. 广播计算两两距离矩阵，把对角线置为 ``inf``；
        2. 逐行取最小值得到 ``d_i``；
        3. 返回 ``d_i`` 的样本标准差（分母 n - 1，与 Schott 原文一致）。

    复杂度:
        时间 O(n^2 * m) / 空间 O(n^2)（两两距离矩阵）。

    陷阱:
        1. 分母是 **n - 1**（样本标准差），不是 n。论文与开源实现两种口径都存在，
           跨实现比较数值前必须先确认分母——差一个因子 ``sqrt(n/(n-1))``。
        2. Spacing 只度量**均匀性**，完全不度量**收敛性**：一条离真实前沿很远的、
           但间距均匀的假前沿，Spacing 可以比真实前沿还小。必须与 IGD / 超体积联合使用。
        3. 相邻很近的重复点会把 ``d_i`` 压到 ~0，且它邻居的 ``d_i`` 也被拉低，
           于是"有一个几乎重复的点"就能显著抬高 Spacing。评价前先去重。
        4. 只对**同一条前沿**调用。把整份种群丢进来会在不同层之间取最近邻，
           得到的值没有意义（与 :func:`crowding_distance` 的要求一致）。

    参考:
        Schott 1995, "Fault Tolerant Design Using Single and Multicriteria Genetic
        Algorithm Optimization"（Spacing 指标）。
    """
    M = as_matrix(F, "F")
    n = M.shape[0]
    key = str(metric).lower()
    if key not in ("l1", "l2"):
        raise ValueError(f"metric 只支持 'l1' 或 'l2'，得到 {metric!r}")
    if n == 1:
        return {
            "spacing": 0.0,
            "nearest": np.zeros(1, dtype=float),
            "mean_nearest": 0.0,
            "n": 1,
            "metric": key,
        }
    diff = M[:, None, :] - M[None, :, :]
    if key == "l1":
        d = np.abs(diff).sum(axis=2)
    else:
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    d_bar = float(nearest.mean())
    spacing = float(np.sqrt(float(((d_bar - nearest) ** 2).sum()) / float(n - 1)))
    return {
        "spacing": spacing,
        "nearest": nearest,
        "mean_nearest": d_bar,
        "n": int(n),
        "metric": key,
    }


def knee_points(F: MatrixLike, method: str = "angle") -> dict:
    """二维前沿上的**拐点（knee）**：在"再牺牲一点某个目标就能大幅换到另一个目标"处。

    参数:
        F: 目标矩阵，形状 (n, 2)，应当是**同一条二维前沿**上的解（全部最小化）。
        method: ``"angle"``（默认）用"折线转角最大"判定，``"distance"`` 用"到两端点连线的
            垂直距离最大"判定。两种口径在典型的凸出前沿上结论一致，在长尾/多段前沿上会不同。

    返回:
        dict，键为：
        ``index``      拐点在**原始矩阵中的下标**（int）；
        ``order``      按第 1 个目标升序排序后的原始下标列表，长度 n；
        ``scores``     形状 (n,) 的数组，**按原始下标回填**每个点的拐点评分（两个端点为 0.0；
                       ``"angle"`` 口径是**转角**，单位弧度——最小化前沿上它恒为非负，
                       越接近直线越小、拐点最大；``"distance"`` 口径是到两端点连线的
                       垂直距离）；
        ``score``      拐点的评分；
        ``method``     实际使用的口径；
        ``degenerate`` bool。前沿**退化成一条直线**（没有拐点）时为 True，此时 ``index``
                       只是平分线上的第一个点，调用者应当忽略它；
        ``monotone``   bool。按第 1 个目标升序后第 2 个目标是否单调不增（同一条最小化前沿
                       应当满足）。为 False 说明输入并不是一条非支配前沿，结论不可信。

    算法:
        1. 用 ``lexsort`` 按 (f0, f1) 升序排序，得到折线 ``P_0 ... P_{n-1}``
           （最小化口径下 f1 应当单调不增）；
        2. ``"angle"``：对每个内点 i，令 ``a = P_{i-1} - P_i``、``b = P_{i+1} - P_i``，
           转角 ``theta_i = pi - arccos(a·b / (|a||b|))``；``"distance"``：令
           ``theta_i = |cross(P_{n-1} - P_0, P_i - P_0)| / ||P_{n-1} - P_0||``；
        3. 取评分最大的内点作为拐点；两端点评分恒为 0；
        4. 上述评分先写在"按 f0 升序"的排序口径下，返回前用 ``scores[order] = ...``
           回填到**原始下标**口径，这样调用者可以直接把 ``scores`` 与 ``F`` 的行对齐。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. 只支持 **m == 2**。三维以上"拐点"没有唯一定义（要靠参考点或流形方法，
           例如 knee 的 ε-dominance 定义），本函数直接抛 ``ValueError`` 而不是给一个
           看起来合理但无法解释的数。
        2. 端点连线的口径对**长尾前沿**很敏感：只要有一端拉得很长，"到直线距离最大"
           就会滑向尾部中段而不是真正的拐点。此时应改用 ``"angle"``，或者先把前沿
           归一化到同一尺度。两种口径的评分**量纲不同**（弧度 vs 目标量纲），不要混用。
        3. 拐点不是"最优解"，它只是"性价比最高的折中点"。如果建模题的决策者偏好未知，
           正确的做法是把拐点和两端点一起报出来，而不是只报拐点。
        4. 输入必须是**非支配前沿**且不含重复点：重复点会让转角退化成 0 或 NaN
           （重复点之间的方向向量为零），并且会凭空造出一个"拐点"。评价前先去重。

    参考:
        Das 1999, "On characterizing the knee of the Pareto front"；
        Branke et al. 2004（knee 区域的 ε-dominance 定义）。
    """
    M = as_matrix(F, "F")
    if M.shape[1] != 2:
        raise ValueError(f"knee_points 只支持二维目标，得到 {M.shape[1]} 个目标")
    n = M.shape[0]
    if n < 3:
        raise ValueError(f"knee_points 至少需要 3 个点（两端点 + 至少一个内点），得到 {n}")
    key = str(method).lower()
    if key not in ("angle", "distance"):
        raise ValueError(f"method 只支持 'angle' 或 'distance'，得到 {method!r}")

    order = np.lexsort((M[:, 1], M[:, 0]))
    P = M[order]
    scores = np.zeros(n, dtype=float)
    extent = float(max(P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()))
    tol = 1e-9 * max(extent, 1.0)

    monotone = bool(np.all(np.diff(P[:, 1]) <= tol))

    if key == "angle":
        a = P[:-2] - P[1:-1]
        b = P[2:] - P[1:-1]
        na = np.sqrt(np.einsum("ij,ij->i", a, a))
        nb = np.sqrt(np.einsum("ij,ij->i", b, b))
        dot = np.einsum("ij,ij->i", a, b)
        denom = na * nb
        cos = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)
        theta = np.pi - np.arccos(np.clip(cos, -1.0, 1.0))
        theta = np.where(denom > 0, theta, 0.0)
        scores[1:-1] = theta
        # 共线前沿的真值是 0，但 arccos 在 cos ≈ ±1 附近的条件数很差（误差约 sqrt(eps)），
        # 实测共线算例会给出 ~2e-8 弧度的残差，所以退化判据用 1e-6 弧度（约 6e-5 度）——
        # 比任何有意义的转角小几个数量级，又能吸收这个数值残差。
        degenerate = bool(scores.max() < 1e-6)
    else:
        base = P[-1] - P[0]
        norm = float(np.sqrt(float(base @ base)))
        if norm <= 0.0:
            degenerate = True
        else:
            rel = P[1:-1] - P[0]
            cross = np.abs(rel[:, 0] * base[1] - rel[:, 1] * base[0]) / norm
            scores[1:-1] = cross
            degenerate = bool(scores.max() <= tol)

    best_sorted = int(np.argmax(scores))
    idx = int(order[best_sorted])
    # 评分回填到原始下标口径，保证 scores 与 F 的行一一对应（见"算法"第 4 步）。
    scores_out = np.empty(n, dtype=float)
    scores_out[order] = scores
    return {
        "index": idx,
        "order": [int(i) for i in order],
        "scores": scores_out,
        "score": float(scores[best_sorted]),
        "method": key,
        "degenerate": degenerate,
        "monotone": monotone,
    }


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _quadratic_objectives(x: np.ndarray) -> np.ndarray:
    """自测用算例：f1 = x0^2，f2 = (x0 - 2)^2，x0 ∈ [-1, 3]。

    解析前沿：x0 ∈ [0, 2] 上是 f2 = (2 - sqrt(f1))^2；x0 < 0 被 -x0 支配，
    x0 > 2 被 4 - x0 支配，所以真实第一前沿上一定有 ``x0 >= 0``。
    """
    v = float(np.asarray(x, dtype=float).ravel()[0])
    return np.array([v * v, (v - 2.0) ** 2])


def _zdt1_objectives(x: np.ndarray) -> np.ndarray:
    """自测用 ZDT1（n = 3）：f1 = x0，g = 1 + 9*(x1+x2)/2，f2 = g*(1 - sqrt(f1/g))。

    解析前沿：x1 = x2 = 0（即 g = 1）时 f2 = 1 - sqrt(f1)。g > 1 会让同 f1 下的
    f2 变大，所以"前沿偏离量"同时度量了 g 的收敛程度。
    """
    a = np.asarray(x, dtype=float).ravel()
    g = 1.0 + 9.0 * float(a[1:].sum()) / float(a.size - 1)
    f1 = float(a[0])
    return np.array([f1, g * (1.0 - np.sqrt(f1 / g))])


def _front_deviation(F: np.ndarray) -> float:
    """二次算例上，与解析前沿 ``f2 = (2 - sqrt(f1))^2`` 的最大偏差。"""
    dev = 0.0
    for i in range(F.shape[0]):
        f1 = max(float(F[i, 0]), 0.0)
        f2 = float(F[i, 1])
        dev = max(dev, abs(f2 - (2.0 - np.sqrt(f1)) ** 2))
    return float(dev)


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict，键名简短 ASCII，值只有 int / float / bool / str / list。
        全部随机过程走固定 seed 的 ``_common.rng``，两次调用逐位一致。

    算法:
        覆盖 13 个函数，并优先安排**闭式解 / 独立结论**校验：
        1. 超体积手算：F = [[0,1],[1,0]]、reference = [2,2] 的面积恰为 3.0
           （两块矩形 1x1 与 1x2 无重叠部分），并被写进断言；
        2. 非支配排序：4 个手工点写死期望的前沿成员与 rank；
        3. 支配关系的自反性（不支配自己）与反对称性（互相支配必有一个为假）；
        4. 拥挤距离：手工前沿 [0,2] / [1,1] / [2,0] 的中间点为 2、两端为 inf；
        5. 加权和法：二次算例的最优解闭式为 x* = 2 w2 / (w1 + w2)，落在解析前沿上，
           因此与解析前沿的偏差应为 0（本实现实测 7e-7，容差取 1e-5）；
        6. ε-约束法：每个 ε 下返回的解必须可行，且落在解析前沿上；
        7. NSGA-II：最终第一前沿的解必须两两非支配、全部落在解析前沿上（偏差 < 0.05）、
           且不含被支配的 x0 < 0 分支；``history``（第一前沿规模）非递减；
        8. ZDT1（3 维决策变量）：最终前沿与 ``f2 = 1 - sqrt(f1)`` 的最大偏差 < 0.05，
           即 g 确实被压到了 1 附近；
        9. ideal_point_distance：对称解集 [[0,1],[1,0]] 的 closeness 闭式为 0.5，
           正理想点上的解 closeness 为 1；pareto_front 的下标与手工排序结果一致；
        10. crowding_distance 的退化列：某个目标取值域为 0 时，该目标不得把边界 ``inf``
           送给按并列次序排出来的任意点（回归检测：修复前会多出两个 ``inf``）；
        11. igd_metric：手算 1.5 与"与参考集相同则为 0"，另与一个独立的三重循环最近邻
           实现逐点对拍（随机 9x3 对 5x3 算例，容差 1e-12），并检查"覆盖更密 IGD 更小"；
        12. spacing_metric：均匀前沿恰为 0（L1/L2 两种口径），非均匀手算例恰为
           ``sqrt(1/3)``，单点前沿按约定取 0，且更不均匀的前沿 Spacing 更大；
        13. knee_points：直线前沿 x+y=1 上 [0.4,0.4] 的转角与到弦的垂直距离都是闭式值，
           两种口径结论一致，打乱输入次序后下标要能映射回原始位置、``scores`` 也要按原始
           下标回填，共线前沿必须报
           ``degenerate``，被支配的输入必须被 ``monotone`` 标记识别出来；
        14. moead：子问题个数 ``C(n_partitions + m - 1, m - 1)``、权重行和为 1、
           ``history`` 长度为 ``n_iter + 1``、``F == objective_fn(X)`` 逐位一致、
           ``minimize=False`` 与"目标取负 + minimize=True"逐位等价、第一前沿互不支配
           且落在解析前沿上（实测偏差恰为 0.0，容差 1e-6）。

    复杂度:
        时间约 O(1)（固定小算例，总目标求值量在 1e5 量级以下）/ 空间 O(1)。

    陷阱:
        1. 这里给出的前沿偏差是"这个 seed、这个预算下"的实测结果，不是算法保证值。
           改默认参数会让这些数字变化，但收敛判据（0.05 一类）不应为了过测而放宽。
        2. ``nsga2_history_monotone`` 断言的是本实现的实测性质：第一前沿规模在理论上
           可以因精英保留阶段的替换而收缩，换问题/换种子后该断言可能失效。
           它服务于"回归检测"，不能当成收敛性证明。
        3. 加权和法的前沿偏差容差（1e-5）依赖随机方向搜索的收敛，属于数值脆点：
           把 n_iter 调小到几十，这条断言会先失败。之所以不放得更严，是因为权重
           ``w = [1, 0]`` 的解析最优解恰在 ``x = 0``，而 x<0 的解是被支配的，
           搜索在 ``x^2`` 这个平坦方向上留下 1e-7 的残差就会放大成 8e-7 的前沿偏差。
        4. ε-约束法里 ε 的最小值由随机采样得到，若它极接近 0，可行区间
           ``[-sqrt(eps), sqrt(eps)]`` 会窄到只剩数值噪声，此时"落在解析前沿上"的
           断言会变脆——本实现靠罚函数把它推向 ``x = +sqrt(eps)`` 才成立。

    参考:
        Pareto 1906；Zadeh 1963（加权和）；Haimes, Lasdon & Wismer 1971（ε-约束）；
        Deb et al. 2002（NSGA-II）；Zitzler & Thiele 1999（超体积）；
        Hwang & Yoon 1981（TOPSIS）。
    """
    result: Dict[str, object] = {}

    # ---------- 1. pareto_dominates：自反性与反对称性 ----------
    p_a = np.array([1.0, 2.0])
    p_b = np.array([1.0, 3.0])
    p_c = np.array([2.0, 3.0])
    if pareto_dominates(p_a, p_a):
        raise AssertionError("支配关系不应自反：一个点不能支配自己")
    if not pareto_dominates(p_a, p_b):
        raise AssertionError("a=[1,2] 应支配 b=[1,3]（第一分量相等、第二分量严格更小）")
    if pareto_dominates(p_b, p_a):
        raise AssertionError("支配关系必须反对称：b 不能同时支配 a")
    if not pareto_dominates(p_a, p_c):
        raise AssertionError("a=[1,2] 应支配 c=[2,3]（逐分量 <= 且至少一个 <）")
    if pareto_dominates(p_a, np.array([0.0, 5.0])):
        raise AssertionError("互不支配的两点不能被判为支配关系")
    result["dominates_reflexive_false"] = True
    result["dominates_antisymmetric"] = True

    # ---------- 2. fast_non_dominated_sort：手工小例写死期望 ----------
    f_hand = np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 2.0], [4.0, 4.0], [3.0, 4.0]])
    nds = fast_non_dominated_sort(f_hand)
    if nds["fronts"] != [[0, 1, 2], [4], [3]]:
        raise AssertionError(f"非支配分层错误：{nds['fronts']}，期望 [[0, 1, 2], [4], [3]]")
    rank_list = [int(r) for r in nds["rank"]]
    if rank_list != [0, 0, 0, 2, 1]:
        raise AssertionError(f"rank 错误：{rank_list}，期望 [0, 0, 0, 2, 1]")
    pf = pareto_front(f_hand)
    if pf["index"] != nds["fronts"][0]:
        raise AssertionError(f"pareto_front 与分层结果不一致：{pf['index']} != {nds['fronts'][0]}")
    result["nds_fronts"] = [list(f) for f in nds["fronts"]]
    result["nds_rank"] = rank_list
    result["nds_n_fronts"] = len(nds["fronts"])
    result["pareto_front_index"] = [int(i) for i in pf["index"]]

    # ---------- 3. crowding_distance：边界 inf、中间点恰为 2 ----------
    cd = crowding_distance(np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]))
    if not (np.isinf(cd[0]) and np.isinf(cd[2])):
        raise AssertionError(f"拥挤距离的两个边界点必须为 inf，得到 {cd.tolist()}")
    if abs(float(cd[1]) - 2.0) > 1e-12:
        raise AssertionError(f"中间点的拥挤距离应为 1/2*... = 1 + 1 = 2，得到 {cd[1]}")
    result["crowding_boundary_inf"] = bool(np.isinf(cd[0]) and np.isinf(cd[2]))
    result["crowding_mid"] = round(float(cd[1]), 6)

    # ---------- 4. hypervolume_2d：手算 = 3.0 ----------
    hv = hypervolume_2d(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([2.0, 2.0]))
    if abs(hv - 3.0) > 1e-12:
        raise AssertionError(f"超体积手算值应为 3.0（1*1 + 1*2），得到 {hv}")
    hv_worse = hypervolume_2d(
        np.array([[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]]), np.array([2.0, 2.0])
    )
    if abs(hv_worse - hv) > 1e-12:
        raise AssertionError(f"被参考点支配的解不应贡献超体积：{hv_worse} != {hv}")
    hv_better = hypervolume_2d(
        np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]), np.array([2.0, 2.0])
    )
    if not hv_better > hv:
        raise AssertionError(f"加入支配性更强的解后超体积必须增大：{hv_better} <= {hv}")
    result["hypervolume_manual"] = round(float(hv), 6)
    result["hypervolume_worse_point"] = round(float(hv_worse), 6)
    result["hypervolume_better_point"] = round(float(hv_better), 6)

    # ---------- 5. ideal_point_distance：对称解集闭式 0.5 / 正理想点 1.0 ----------
    ip = ideal_point_distance(np.array([[0.0, 1.0], [1.0, 0.0]]))
    if abs(float(ip["closeness"][0]) - 0.5) > 1e-12 or abs(float(ip["closeness"][1]) - 0.5) > 1e-12:
        raise AssertionError(f"对称解集的贴近度应为 0.5，得到 {ip['closeness'].tolist()}")
    ip2 = ideal_point_distance(np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]))
    if abs(float(ip2["closeness"][1]) - 1.0) > 1e-12:
        raise AssertionError(f"落在正理想点上的解贴近度应为 1，得到 {ip2['closeness'][1]}")
    result["ideal_closeness_symmetric"] = round(float(ip["closeness"][0]), 6)
    result["ideal_closeness_ideal_point"] = round(float(ip2["closeness"][1]), 6)
    result["ideal_closeness_ordered"] = bool(ip2["closeness"][1] > ip2["closeness"][0])

    # ---------- 6. weighted_sum_pareto：闭式 x* = 2 w2 / (w1 + w2) ----------
    quad_bounds = [(-1.0, 3.0)]
    ws = weighted_sum_pareto(_quadratic_objectives, quad_bounds, n_weights=11, seed=20240101)
    w_dev = _front_deviation(ws["F"])
    if w_dev > 1e-5:
        raise AssertionError(f"加权和法的解应全部落在解析前沿上，最大偏差 {w_dev}")
    w_mat = ws["weights"]
    x_star = 2.0 * w_mat[:, 1] / (w_mat[:, 0] + w_mat[:, 1])
    x_err = float(np.max(np.abs(ws["X"][:, 0] - x_star)))
    if x_err > 1e-6:
        raise AssertionError(f"加权和法的最优解与闭式 x* = 2 w2/(w1+w2) 不符：最大误差 {x_err}")
    result["weighted_sum_n"] = int(w_mat.shape[0])
    result["weighted_sum_dev"] = round(float(w_dev), 9)
    result["weighted_sum_x_err"] = round(float(x_err), 9)

    # ---------- 7. epsilon_constraint_pareto：可行性 + 前沿归属 ----------
    ec = epsilon_constraint_pareto(_quadratic_objectives, quad_bounds, n_grid=11, seed=20240101)
    if any(ec["infeasible"]):
        raise AssertionError(f"ε-约束法返回了不可行解：{ec['infeasible']}")
    e_dev = _front_deviation(ec["F"])
    if e_dev > 0.05:
        raise AssertionError(f"ε-约束法的解应落在解析前沿上，最大偏差 {e_dev}")
    result["eps_n"] = int(ec["epsilons"].size)
    result["eps_infeasible_n"] = int(sum(1 for v in ec["infeasible"] if v))
    result["eps_dev"] = round(float(e_dev), 6)
    result["eps_first"] = round(float(ec["epsilons"][0]), 6)

    # ---------- 8. nsga2：分层一致性 + 前沿偏差 + history 非递减 ----------
    ng = nsga2(_quadratic_objectives, quad_bounds, pop_size=40, n_gen=60, seed=20240101)
    ng_front = np.asarray(ng["front"], dtype=int)
    if ng_front.size == 0:
        raise AssertionError("NSGA-II 返回了空的第一前沿")
    sub = fast_non_dominated_sort(ng["F"][ng_front])
    if len(sub["fronts"]) != 1:
        raise AssertionError(f"NSGA-II 的 front 内部仍有支配关系，被分成 {len(sub['fronts'])} 层")
    if np.any(ng["X"][ng_front, 0] < -1e-6):
        raise AssertionError("第一前沿出现了 x0 < 0 的解：它们被 -x0 严格支配，不应存在")
    ng_dev = _front_deviation(ng["F"][ng_front])
    if ng_dev > 0.05:
        raise AssertionError(f"NSGA-II 前沿与解析前沿偏差过大：{ng_dev}")
    hist = [int(h) for h in ng["history"]]
    if any(hist[i + 1] < hist[i] for i in range(len(hist) - 1)):
        raise AssertionError(f"第一前沿规模出现下降（精英保留失效）：{hist}")
    result["nsga2_front_size"] = int(ng_front.size)
    result["nsga2_dev"] = round(float(ng_dev), 6)
    result["nsga2_n_fronts"] = int(len(fast_non_dominated_sort(ng["F"])["fronts"]))
    result["nsga2_history_len"] = len(hist)
    result["nsga2_history_first"] = hist[0]
    result["nsga2_history_last"] = hist[-1]
    result["nsga2_history_monotone"] = True

    # ---------- 9. ZDT1：结构不变量 + 相对改进（决策变量 3 维，150 代） ----------
    # 设计取舍（本仓库 CI 实测教训）：NSGA-II 跑 150 代是"离散选择 + 连续变异"的混沌过程，
    # 末位浮点差异会被选择放大成完全不同的进化轨迹。同一个提交，本地 Windows 得
    # z_dev = 0.005040 / g_max = 1.006226；Linux CI 两次分别得 0.005923 / 1.008497 与
    # g_max ∈ (1.02, 1.05]——**连"分档到 2%"都会在两次 CI 之间翻档**。根因不是版本
    # （本机 numpy 2.1.3 与 2.5.3 逐位相同）也不是种子（rng 比特流稳定），而是不同 runner
    # 的 CPU 指令集/BLAS 让 np.sum 的成对求和差几个 ULP，再被 150 代的选择放大。
    # 所以这里**不写任何连续量的绝对阈值**，只断言：(1) 数学不变量；(2) 前沿内部互不支配；
    # (3) history 的结构性质；(4) 相对"随机初始种群"的**相对**改进（随机基线由同一 RNG
    # 生成，平台漂移会被同时约掉）。黄金值同理只记整数结构量。
    zdt = nsga2(_zdt1_objectives, [(0.0, 1.0)] * 3, pop_size=60, n_gen=150, seed=7)
    zdt_front = np.asarray(zdt["front"], dtype=int)
    if zdt_front.size == 0:
        raise AssertionError("ZDT1 上 NSGA-II 返回了空的第一前沿")
    zf = zdt["F"][zdt_front]
    zx = zdt["X"][zdt_front]

    # (1) 数学不变量：ZDT1 的 g = 1 + 9 Σx_i (i >= 2) / (n - 1) 在 x ∈ [0,1] 上恒 >= 1
    g_vals = 1.0 + 9.0 * zx[:, 1:].sum(axis=1) / 2.0
    if float(np.min(g_vals)) < 1.0 - 1e-12:
        raise AssertionError(f"ZDT1 的 g 应恒 >= 1，实测最小值 {float(np.min(g_vals))}")

    # (2) 独立的两两支配检查（不调用本模块的 pareto_dominates）：第一前沿内部必须互不支配
    z_dev = float(
        np.max(np.abs(zf[:, 1] - (1.0 - np.sqrt(np.clip(zf[:, 0], 0.0, 1.0)))))
    )
    for a in range(zf.shape[0]):
        weaker = np.all(zf <= zf[a], axis=1) & np.any(zf < zf[a], axis=1)
        weaker[a] = False
        if bool(np.any(weaker)):
            raise AssertionError(
                f"ZDT1 第一前沿内部存在支配关系：第 {int(np.flatnonzero(weaker)[0])} 个点支配第 {a} 个点"
            )

    # (3) 结构性质：history 恒为 n_gen + 1 代，且每代记录的是"当前种群内的第一前沿规模"，
    #     落在 [1, pop_size] 内。注意它**不一定单调**——第 8 组二次算例上恰好单调，
    #     这里不做单调断言（实测 ZDT1 的 history 就出现过下降）。
    zhist = [int(h) for h in zdt["history"]]
    if len(zhist) != 151:
        raise AssertionError(f"ZDT1 的 history 长度应为 n_gen + 1 = 151，实测 {len(zhist)}")
    if not all(1 <= h <= 60 for h in zhist):
        raise AssertionError(f"ZDT1 的 history 取值越界（应在 [1, 60]）：{zhist}")

    # (4) 相对改进：与同一 RNG 下的随机初始种群相比，最终前沿到解析前沿的偏差应小一个量级以上
    f_rand = np.array([_zdt1_objectives(row) for row in make_rng(7).random((60, 3))])
    r_dev = float(
        np.max(np.abs(f_rand[:, 1] - (1.0 - np.sqrt(np.clip(f_rand[:, 0], 0.0, 1.0)))))
    )
    if not z_dev * 10.0 < r_dev:
        raise AssertionError(
            f"ZDT1 未体现收敛：进化 150 代后前沿偏差 {z_dev:.6f} 应比随机初始种群 {r_dev:.6f} 小一个量级以上"
        )
    result["zdt1_front_size"] = int(zdt_front.size)
    result["zdt1_nondominated"] = 1
    result["zdt1_history_len"] = len(zhist)

    # ---------- 10. crowding_distance：退化目标不得把 inf 送给并列点 ----------
    # 第 2 个目标恒为 0，是退化列。修好之后该列被整体跳过，只有第 1 个目标贡献距离：
    # 全部 5 个点在 [0, 4] 上等距，内部点各得 (2 - 0)/4 = 0.5，两个真端点得 inf。
    # 修复前该列会按 argsort 的并列次序把 inf 白送给下标 0 和 4，得到 4 个 inf。
    cd_deg = crowding_distance(
        np.array([[3.0, 0.0], [4.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    )
    deg_inf_idx = [int(i) for i in np.flatnonzero(np.isinf(cd_deg))]
    if deg_inf_idx != [1, 2]:
        raise AssertionError(
            "退化目标不应把 inf 送给并列点：只有第 1 目标的两个真端点（下标 1、2）该为 inf，"
            f"实测 inf 出现在 {deg_inf_idx}"
        )
    if (
        abs(float(cd_deg[0]) - 0.5) > 1e-12
        or abs(float(cd_deg[3]) - 0.5) > 1e-12
        or abs(float(cd_deg[4]) - 0.5) > 1e-12
    ):
        raise AssertionError(f"非退化目标上的内部点拥挤距离应均为 0.5，得到 {cd_deg.tolist()}")
    result["crowding_degenerate_inf_count"] = int(len(deg_inf_idx))
    result["crowding_degenerate_mid"] = round(float(cd_deg[0]), 6)

    # ---------- 11. igd_metric：手算 + 独立三重循环对拍 ----------
    igd_good = np.array([[0.0, 0.0], [3.0, 0.0]])
    igd_ref = np.array([[0.0, 0.0], [6.0, 0.0]])
    ig = igd_metric(igd_good, igd_ref)
    if abs(float(ig["igd"]) - 1.5) > 1e-12:
        raise AssertionError(f"IGD 手算值应为 (0 + 3) / 2 = 1.5，得到 {ig['igd']}")
    if abs(float(ig["worst_nearest"]) - 3.0) > 1e-12:
        raise AssertionError(f"最差覆盖距离应为 3.0，得到 {ig['worst_nearest']}")
    if int(ig["n_reference"]) != 2 or int(ig["n_approx"]) != 2:
        raise AssertionError(f"IGD 返回的计数不对：{ig['n_reference']} / {ig['n_approx']}")
    ig_zero = igd_metric(igd_ref, igd_ref)
    if abs(float(ig_zero["igd"])) > 1e-12:
        raise AssertionError(f"近似集与参考集相同时 IGD 必须为 0，得到 {ig_zero['igd']}")

    # 独立参考实现：三重循环逐点做最近邻，再对参考点取平均
    gen_igd = make_rng(31337)
    igd_rand_a = gen_igd.random((9, 3)) * 4.0
    igd_rand_r = gen_igd.random((5, 3)) * 4.0
    ref_igd = 0.0
    for j in range(igd_rand_r.shape[0]):
        best = float("inf")
        for i in range(igd_rand_a.shape[0]):
            acc = 0.0
            for k in range(3):
                dk = float(igd_rand_r[j, k] - igd_rand_a[i, k])
                acc += dk * dk
            best = min(best, acc ** 0.5)
        ref_igd += best
    ref_igd /= float(igd_rand_r.shape[0])
    got_igd = float(igd_metric(igd_rand_a, igd_rand_r)["igd"])
    if abs(got_igd - ref_igd) > 1e-12:
        raise AssertionError(f"IGD 与独立三重循环实现不符：{got_igd} vs {ref_igd}")

    # 语义检查：覆盖更密的近似集 IGD 必须更小（参考前沿是直线段 f0 + f1 = 1 上的 21 点）
    line_grid = np.linspace(0.0, 1.0, 21)
    igd_line_ref = np.column_stack([line_grid, 1.0 - line_grid])
    igd_coarse = igd_metric(
        np.column_stack([np.linspace(0.0, 1.0, 3), 1.0 - np.linspace(0.0, 1.0, 3)]), igd_line_ref
    )["igd"]
    igd_fine = igd_metric(
        np.column_stack([np.linspace(0.0, 1.0, 11), 1.0 - np.linspace(0.0, 1.0, 11)]), igd_line_ref
    )["igd"]
    if not float(igd_coarse) > float(igd_fine):
        raise AssertionError(f"覆盖更密的前沿 IGD 应更小：粗略 {igd_coarse} vs 密集 {igd_fine}")
    try:
        igd_metric(np.zeros((3, 2)), np.zeros((3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("IGD 在近似集与参考集目标个数不一致时必须抛 ValueError")
    result["igd_hand"] = round(float(ig["igd"]), 9)
    result["igd_worst_nearest"] = round(float(ig["worst_nearest"]), 9)
    result["igd_zero_when_exact"] = round(float(ig_zero["igd"]), 9)
    result["igd_random_matches_bruteforce"] = round(got_igd, 6)
    result["igd_denser_is_smaller"] = bool(float(igd_coarse) > float(igd_fine))

    # ---------- 12. spacing_metric：均匀前沿恰为 0 + 手算 sqrt(1/3) ----------
    sp_uniform_front = np.array([[float(i), 5.0 - float(i)] for i in range(6)])
    sp_uniform = spacing_metric(sp_uniform_front)
    if abs(float(sp_uniform["spacing"])) > 1e-12:
        raise AssertionError(f"完全均匀的前沿 Spacing 必须为 0，得到 {sp_uniform['spacing']}")
    sp_uniform_l2 = spacing_metric(sp_uniform_front, metric="l2")
    if abs(float(sp_uniform_l2["spacing"])) > 1e-12:
        raise AssertionError(f"L2 口径下均匀前沿 Spacing 也必须为 0，得到 {sp_uniform_l2['spacing']}")
    sp_hand = spacing_metric(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]))
    sp_expected = float(np.sqrt(1.0 / 3.0))
    if abs(float(sp_hand["spacing"]) - sp_expected) > 1e-12:
        raise AssertionError(
            f"Spacing 手算值应为 sqrt((2*(1/3)^2 + (2/3)^2) / 2) = sqrt(1/3) = {sp_expected}，得到 {sp_hand['spacing']}"
        )
    if abs(float(sp_hand["mean_nearest"]) - 4.0 / 3.0) > 1e-12:
        raise AssertionError(f"平均最近邻距离应为 (1+1+2)/3 = 4/3，得到 {sp_hand['mean_nearest']}")
    sp_single = spacing_metric(np.array([[7.0, 7.0]]))
    if abs(float(sp_single["spacing"])) > 1e-12:
        raise AssertionError(f"单点前沿的 Spacing 按约定取 0，得到 {sp_single['spacing']}")
    # 语义检查：把一个点挪到更不均匀的位置，Spacing 必须变大
    sp_skew = spacing_metric(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]))
    sp_even = spacing_metric(np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
    if not float(sp_skew["spacing"]) > float(sp_even["spacing"]):
        raise AssertionError(f"更不均匀的前沿 Spacing 必须更大：{sp_skew['spacing']} vs {sp_even['spacing']}")
    try:
        spacing_metric(sp_uniform_front, metric="l3")
    except ValueError:
        pass
    else:
        raise AssertionError("Spacing 对未知 metric 必须抛 ValueError")
    result["spacing_uniform"] = round(float(sp_uniform["spacing"]), 9)
    result["spacing_uniform_l2"] = round(float(sp_uniform_l2["spacing"]), 9)
    result["spacing_hand"] = round(float(sp_hand["spacing"]), 9)
    result["spacing_single"] = round(float(sp_single["spacing"]), 9)
    result["spacing_skew_bigger"] = bool(float(sp_skew["spacing"]) > float(sp_even["spacing"]))

    # ---------- 13. knee_points：手算拐点 + 两种口径一致 + 退化前沿 ----------
    knee_hand = np.array([[0.0, 1.0], [0.4, 0.4], [1.0, 0.0]])
    kn_angle = knee_points(knee_hand)
    if int(kn_angle["index"]) != 1:
        raise AssertionError(f"直线前沿 x+y=1 上 [0.4,0.4] 应为拐点，得到下标 {kn_angle['index']}")
    if bool(kn_angle["degenerate"]):
        raise AssertionError("非共线前沿不应被标成退化")
    if not bool(kn_angle["monotone"]):
        raise AssertionError("按 f0 升序后 f1 单调不增的前沿应被标成 monotone")
    kn_dist = knee_points(knee_hand, method="distance")
    if int(kn_dist["index"]) != 1:
        raise AssertionError(f"距离口径下拐点也应是下标 1，得到 {kn_dist['index']}")
    if abs(float(kn_dist["score"]) - 0.2 / float(np.sqrt(2.0))) > 1e-12:
        raise AssertionError(
            f"到直线 x+y=1 的垂直距离应为 0.2/sqrt(2)，得到 {kn_dist['score']}"
        )
    kn_shuffled = knee_points(np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]]))
    if int(kn_shuffled["index"]) != 2:
        raise AssertionError(f"打乱输入次序后 index 必须映射回原始下标 2，得到 {kn_shuffled['index']}")
    if kn_shuffled["order"] != [1, 2, 0]:
        raise AssertionError(f"排序后的原始下标应恢复为 [1, 2, 0]，得到 {kn_shuffled['order']}")
    # scores 必须按原始下标回填（不是排序口径）：打乱输入后最大值仍应落在下标 2
    sc_shuffled = np.asarray(kn_shuffled["scores"], dtype=float)
    if int(np.argmax(sc_shuffled)) != 2:
        raise AssertionError(
            f"scores 必须按原始下标回填（最大值在下标 2），得到 {sc_shuffled.tolist()}"
        )
    if abs(float(sc_shuffled[2]) - float(kn_shuffled["score"])) > 1e-12:
        raise AssertionError(
            f"scores[2] 应等于 score，得到 {sc_shuffled[2]} vs {kn_shuffled['score']}"
        )
    # 共线（且单调）前沿没有拐点：两种口径都必须报 degenerate
    knee_flat = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    if not bool(knee_points(knee_flat)["degenerate"]):
        raise AssertionError("共线前沿在 angle 口径下必须报 degenerate")
    if not bool(knee_points(knee_flat, method="distance")["degenerate"]):
        raise AssertionError("共线前沿在 distance 口径下必须报 degenerate")
    # 非支配但非单调的输入要能识别出来（[[0,0],[1,1],[2,2]] 是被支配的一串点）
    kn_bad = knee_points(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    if bool(kn_bad["monotone"]):
        raise AssertionError("f1 随 f0 上升的输入必须被标成非单调（它不是最小化前沿）")
    # 拐点不必落在中间：4 点前沿上距离口径的拐点是下标 2
    kn_four = knee_points(
        np.array([[0.0, 1.0], [0.5, 0.75], [0.9, 0.5], [1.0, 0.0]]), method="distance"
    )
    if int(kn_four["index"]) != 2:
        raise AssertionError(f"4 点前沿的拐点应为下标 2，得到 {kn_four['index']}")
    if abs(float(kn_four["score"]) - 0.4 / float(np.sqrt(2.0))) > 1e-12:
        raise AssertionError(f"拐点垂直距离应为 0.4/sqrt(2)，得到 {kn_four['score']}")
    for bad_call in (
        lambda: knee_points(np.zeros((4, 3))),
        lambda: knee_points(np.zeros((2, 2))),
        lambda: knee_points(knee_hand, method="curvature"),
    ):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("knee_points 对非二维 / 点数不足 / 未知 method 必须抛 ValueError")
    result["knee_index"] = int(kn_angle["index"])
    result["knee_index_shuffled"] = int(kn_shuffled["index"])
    result["knee_index_four_points"] = int(kn_four["index"])
    result["knee_angle_score"] = round(float(kn_angle["score"]), 9)
    result["knee_distance_score"] = round(float(kn_dist["score"]), 9)
    result["knee_methods_agree"] = bool(int(kn_angle["index"]) == int(kn_dist["index"]))
    result["knee_degenerate_flag"] = bool(
        knee_points(knee_flat)["degenerate"] and knee_points(knee_flat, method="distance")["degenerate"]
    )
    result["knee_monotone_flag"] = bool(kn_angle["monotone"])
    result["knee_nonmonotone_detected"] = bool(not kn_bad["monotone"])
    result["knee_scores_aligned"] = bool(int(np.argmax(sc_shuffled)) == 2)

    # ---------- 14. moead：权重网格 + 目标一致性 + maximize 等价 + 前沿质量 ----------
    # 设计取舍（沿用第 9 组的教训）：这里只断言**精确的结构不变量**和**闭式前沿命中**，
    # 黄金值只记整数/布尔量，不记 60 代进化出来的连续量。
    moe = moead(_quadratic_objectives, quad_bounds, n_partitions=8, n_iter=60, seed=20240101)
    if int(moe["n_weights"]) != 9:
        raise AssertionError(f"m=2、n_partitions=8 时子问题个数应为 9，得到 {moe['n_weights']}")
    if int(moe["neighborhood_size"]) != 2:
        raise AssertionError(
            f"邻域大小应为 max(2, ceil(0.1*9)) = 2，得到 {moe['neighborhood_size']}"
        )
    if len(moe["history"]) != 61:
        raise AssertionError(f"history 长度应为 n_iter + 1 = 61，得到 {len(moe['history'])}")
    if not all(1 <= int(h) <= 9 for h in moe["history"]):
        raise AssertionError(f"history 取值应落在 [1, 9] 内：{moe['history']}")
    w_sum_err = float(np.max(np.abs(moe["weights"].sum(axis=1) - 1.0)))
    if w_sum_err > 1e-12:
        raise AssertionError(f"权重网格每行之和必须为 1，最大偏差 {w_sum_err}")
    if not bool(np.all(moe["ideal"] <= moe["F"].min(axis=0) + 1e-15)):
        # ideal 是"所有求值过的点"上的最小值，因此必然逐分量不劣于最终种群的逐分量最小值
        raise AssertionError(
            f"ideal 应逐分量不劣于 F 的最小值：{moe['ideal'].tolist()} vs {moe['F'].min(axis=0).tolist()}"
        )
    # X 与 F 必须严格对应（内部一致性的精确不变量）
    if not np.array_equal(_eval_pop(_quadratic_objectives, moe["X"]), moe["F"]):
        raise AssertionError("MOEA/D 返回的 F 与 objective_fn(X) 不逐位一致")
    if not np.array_equal(moe["G"], moe["F"]):
        raise AssertionError("minimize=True 时内部最小化空间 G 应逐位等于 F")
    # minimize=False 必须与"对目标取负后 minimize=True"完全等价（逐位相同）
    moe_max = moead(
        _quadratic_objectives, quad_bounds, n_partitions=8, n_iter=60, seed=20240101, minimize=False
    )
    moe_neg = moead(
        lambda x: -_quadratic_objectives(x), quad_bounds, n_partitions=8, n_iter=60, seed=20240101
    )
    if not np.array_equal(moe_max["G"], moe_neg["G"]):
        raise AssertionError("minimize=False 必须与目标取负后的 minimize=True 逐位一致")
    # front 必须是（独立检查下的）互不支配集合，且落在解析前沿上
    moe_front = np.asarray(moe["front"], dtype=int)
    if moe_front.size == 0:
        raise AssertionError("MOEA/D 返回了空的第一前沿")
    moe_f = moe["F"][moe_front]
    for a in range(moe_f.shape[0]):
        weaker = np.all(moe_f <= moe_f[a], axis=1) & np.any(moe_f < moe_f[a], axis=1)
        weaker[a] = False
        if bool(np.any(weaker)):
            raise AssertionError(f"MOEA/D 第一前沿内部存在支配关系：第 {a} 个点被支配")
    moe_dev = _front_deviation(moe_f)
    if moe_dev > 1e-6:
        raise AssertionError(f"MOEA/D 前沿应落在二次算例的解析前沿上，最大偏差 {moe_dev}")
    for bad_kwargs in (
        {"n_partitions": 0},
        {"n_iter": -1},
        {"neighborhood_size": 0},
        {"mutation_sigma": 0.0},
        {"minimize": 1},
    ):
        call_kwargs = {"n_partitions": 8, "n_iter": 0, "seed": 1}
        call_kwargs.update(bad_kwargs)
        try:
            moead(_quadratic_objectives, quad_bounds, **call_kwargs)
        except ValueError:
            continue
        raise AssertionError(f"moead 对非法参数 {bad_kwargs} 必须抛 ValueError")
    result["moead_n_weights"] = int(moe["n_weights"])
    result["moead_neighborhood_size"] = int(moe["neighborhood_size"])
    result["moead_history_len"] = len(moe["history"])
    result["moead_weights_sum_one"] = bool(w_sum_err <= 1e-12)
    result["moead_ideal_not_worse"] = True
    result["moead_F_consistent"] = True
    result["moead_maximize_matches_negation"] = bool(np.array_equal(moe_max["G"], moe_neg["G"]))
    result["moead_front_nondominated"] = True
    result["moead_front_on_analytic"] = True
    result["moead_front_dev"] = round(float(moe_dev), 9)
    # 搜索盒校验的回归检测：一维简写 (lo, hi) 必须与 [(lo, hi)] 走同一套有限性与
    # hi > lo 校验（修复前这里会静默接受非法搜索盒，把全部个体钉在 lo 上）。
    for bad_box in ((5.0, 3.0), (5.0, 5.0), (float("nan"), 5.0), (float("inf"), 5.0)):
        try:
            _parse_box(bad_box)
        except ValueError:
            continue
        raise AssertionError(f"x_bounds={bad_box} 必须抛 ValueError（hi <= lo 或非有限）")
    ok_lo, ok_hi = _parse_box((0.0, 1.0))
    if ok_lo.tolist() != [0.0] or ok_hi.tolist() != [1.0]:
        raise AssertionError(f"x_bounds=(0.0, 1.0) 必须解析成 1 维盒 [0, 1]，得到 {ok_lo} / {ok_hi}")
    return result