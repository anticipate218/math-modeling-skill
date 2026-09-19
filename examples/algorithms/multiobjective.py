"""多目标优化：加权和法、ε-约束法、支配关系判定、非支配排序、拥挤距离、NSGA-II、
Pareto 前沿、二维超体积与理想点距离。

本模块共 9 个公开函数：支配与排序 ``pareto_dominates`` / ``fast_non_dominated_sort`` /
``crowding_distance``，前沿构造 ``weighted_sum_pareto`` / ``epsilon_constraint_pareto`` /
``nsga2``，前沿后处理 ``pareto_front`` / ``hypervolume_2d`` / ``ideal_point_distance``。

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
  边界点取 ``inf``，且在任一目标上取值域为 0 时该目标不贡献距离。
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
    "pareto_front",
    "hypervolume_2d",
    "ideal_point_distance",
]


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _parse_box(bounds) -> Tuple[np.ndarray, np.ndarray]:
    """把 ``x_bounds`` 规范成逐维的 (lo, hi) 两个一维数组。

    口径（与本模块文档一致）：``(lo, hi)`` 两个标量表示 **1 维**问题；
    要 d 维共用同一区间请写 ``[(lo, hi)] * d``。
    """
    if bounds is None:
        raise ValueError("x_bounds 不能为 None（多目标优化必须有搜索盒）")
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"x_bounds 为 (lo, hi) 时长度必须为 2，得到 {arr.size}")
        lo = np.array([arr[0]], dtype=float)
        hi = np.array([arr[1]], dtype=float)
        return lo, hi
    if arr.ndim == 2 and arr.shape[1] == 2:
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
        因此 ``n <= 2`` 时全部为 ``inf``；n > 2 时边界点一定是 ``inf``。

    算法:
        1. 对每个目标 k，用 ``argsort`` 排序，边界点记 ``inf``；
        2. 内部点累加 ``(f[order[i+1], k] - f[order[i-1], k]) / (f_max - f_min)``；
        3. 若该目标取值域为 0（前沿在该目标上完全退化），跳过该目标（贡献 0），
           而不是除零得到 ``inf`` 或 NaN。

    复杂度:
        时间 O(m * n log n) / 空间 O(n)。

    陷阱:
        1. **必须按前沿分别调用**。把整份种群的目标矩阵直接丢进来会在不同前沿之间
           比较距离，得到的数没有任何意义（这也是常见误用）。
        2. 距离已经按各目标的取值范围归一化，所以量纲不同的目标可以直接相加；
           但取值范围为 0 的目标必须跳过，否则 0/0 会污染整条前沿。
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
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = float(vals[-1] - vals[0])
        if span <= 0.0:
            continue
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
           罚函数为 ``P * max(0, sign * f_1 - eps)``，``P = 1e3 * 目标 1 的采样极差``；
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
        返回的下标是**原始矩阵中的位置**，不是排序后的位置；如果你先把 F 排序再调用，
        拿到的下标必须在原数组上重新映射，否则会取错解。

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
        覆盖 9 个函数，并优先安排**闭式解 / 独立结论**校验：
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
           正理想点上的解 closeness 为 1；pareto_front 的下标与手工排序结果一致。

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

    # ---------- 9. ZDT1：g 收敛（决策变量 3 维） ----------
    zdt = nsga2(_zdt1_objectives, [(0.0, 1.0)] * 3, pop_size=60, n_gen=150, seed=7)
    zdt_front = np.asarray(zdt["front"], dtype=int)
    if zdt_front.size == 0:
        raise AssertionError("ZDT1 上 NSGA-II 返回了空的第一前沿")
    zf = zdt["F"][zdt_front]
    z_dev = float(np.max(np.abs(zf[:, 1] - (1.0 - np.sqrt(np.clip(zf[:, 0], 0.0, 1.0))))))
    if z_dev > 0.05:
        raise AssertionError(
            f"ZDT1 前沿与解析前沿 f2 = 1 - sqrt(f1) 偏差过大：{z_dev}，说明 g 未收敛到 1"
        )
    g_vals = 1.0 + 9.0 * zdt["X"][zdt_front, 1:].sum(axis=1) / 2.0
    if float(np.max(g_vals)) > 1.05:
        raise AssertionError(f"ZDT1 前沿解的 g 应接近 1，最大值为 {float(np.max(g_vals))}")
    result["zdt1_front_size"] = int(zdt_front.size)
    # 说明（设计取舍）：NSGA-II 跑 150 代是"离散选择 + 连续变异"的混沌过程，末位浮点
    # 差异会被放大——同一个提交在本地 Windows 得 z_dev=0.005040，在 Linux CI 得
    # 0.005923，而 front_size / history 完全一致。连续量因此**不适合当黄金值**：
    # 它会在换平台时无意义地翻红。这里只记分档后的整数/布尔指纹，收敛精度由上面
    # 两条断言把关（dev <= 0.05、g_max <= 1.05）。粗粒度指纹 + 严格断言，比一个会
    # 随平台漂移的 6 位小数可靠得多。
    result["zdt1_dev_le_2pct"] = int(1 if z_dev <= 0.02 else 0)
    result["zdt1_g_le_2pct"] = int(1 if float(np.max(g_vals)) <= 1.02 else 0)
    return result