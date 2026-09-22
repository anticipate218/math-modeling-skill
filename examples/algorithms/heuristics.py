"""启发式优化：模拟退火、遗传算法、粒子群、蚁群 TSP、差分进化、禁忌搜索、
灰狼优化、变邻域搜索，以及标准测试函数与算法对比台、人工蜂群。

本模块共 11 个公开函数，分三组：

1. 元启发式求解器（8 个）：``simulated_annealing``、``genetic_algorithm``、
   ``particle_swarm``、``ant_colony_tsp``、``differential_evolution``、
   ``tabu_search``、``grey_wolf_optimizer``、``variable_neighborhood_search``。
2. 基准与对比（2 个）：``benchmark_functions``（标准测试函数 + 已知全局最优）、
   ``benchmark_optimizers``（固定 seed 集合下统一跑多个求解器，输出
   best/mean/std/median/成功次数与排名，直接用于论文的"算法对比与鲁棒性"表）。
3. 额外的群体智能求解器（1 个）：``artificial_bee_colony``（ABC 三阶段）。

用途定位：当目标函数不可导、非凸、离散或组合爆炸（NP-hard）时，解析方法和梯度法失效，
这类"随机搜索 + 局部改进"的元启发式是竞赛里最常见的兜底手段。

**写论文时必须交代的三件事**（否则结果无法被复现，会被评委扣分）：
1. 随机种子。所有函数都接受 ``seed``，不给就用库默认 ``_common.DEFAULT_SEED``，**不碰全局随机状态**。
2. 参数与预算。``iters`` / ``generations`` / 迭代次数就是"计算预算"，报告时应给出目标函数
   的**实际求值次数**；不同预算下的解不能直接比较。
3. 最好/最差/中位数。元启发式是随机算法，**单次运行的结果不能作为结论**。
   本仓库的验证脚本对每个算法跑 ≥3 个不同种子并报告分布。

关于"最优解"的诚实表述：本模块的所有算法都**不保证全局最优**。在小算例上它们常常命中
最优，这只能说明"在这组参数与这个算例上表现良好"，不能外推成"算法更好"。

依赖：仅 numpy + 标准库 + ``._common``。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_vector, rng as make_rng

__all__ = [
    "simulated_annealing",
    "genetic_algorithm",
    "particle_swarm",
    "ant_colony_tsp",
    "differential_evolution",
    "tabu_search",
    "grey_wolf_optimizer",
    "variable_neighborhood_search",
    "benchmark_functions",
    "benchmark_optimizers",
    "artificial_bee_colony",
]


# --------------------------------------------------------------------------- #
# 模拟退火
# --------------------------------------------------------------------------- #
def simulated_annealing(
    cost: Callable[[np.ndarray], float],
    x0,
    neighbor: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    T0: float = 100.0,
    alpha: float = 0.995,
    iters: int = 5000,
    seed: Optional[int] = None,
) -> dict:
    """模拟退火（Metropolis 准则 + 几何降温），最小化 ``cost``。

    参数:
        cost: ``cost(x) -> float``，目标函数，**约定为最小化**（要最大化请自己取负）。
        x0: 初始解（任意 numpy 可转成 float 数组的对象；通常是一维向量）。
        neighbor: ``neighbor(x, rng) -> x_new``，产生邻域候选解。**必须接收 rng**，
            不要用全局 ``np.random``，否则同一 seed 无法复现。
        T0: 初始温度，必须 > 0。
        alpha: 降温系数，取值 (0, 1)，通常 0.9~0.999。越小降温越快、越早"冻结"。
        iters: 迭代（降温）步数，总求值次数 = iters + 1。
        seed: 随机种子，None 用 ``DEFAULT_SEED``。

    返回:
        ``{"best_x": np.ndarray, "best_cost": float, "history": List[float]}``。
        ``history[i]`` 是第 i 步**结束后**的当前最优代价，长度 = iters + 1，
        可用来画收敛曲线（注意它是单调不增的，画出来很"好看"但有误导性——
        论文里更该画"当前解"而不是"历史最优"）。

    算法:
        1. 令 cur = x0，best = x0，T = T0；
        2. 每步生成候选 y，按 Metropolis 准则接受：
           若 ``cost(y) < cost(cur)`` 必接受，否则以 ``exp(-(cost(y)-cost(cur))/T)`` 接受；
        3. 用 ``alpha`` 几何降温 ``T <- alpha * T``，并记录 best。

    复杂度:
        时间 O(iters * (T_cost + T_neighbor)) / 空间 O(|x| + iters)（history）。

    陷阱:
        1. **温度标定**：T0 必须与目标函数的量级匹配。目标函数取值 1e-3 量级却设 T0=100，
           前几千步几乎全盘接受，等于随机游走；反之 T0 太小则一开始就拒绝一切，
           退化成爬山。实用做法是先跑几十步采样 ``|Δcost|`` 的均值，令 T0 ≈ 若干倍该均值。
        2. ``alpha`` 与 ``iters`` 要配套：``alpha=0.995, iters=5000`` 只降 e^{-25}≈1.4e-11 倍，
           末温极低；而 ``alpha=0.9`` 时 100 步就降到 0.9^100 ≈ 2.7e-5，几乎必然早熟。
        3. 邻域尺度要随问题变化（离散问题换位/翻转，连续问题用步长）。**固定步长的
           连续邻域在高维下几乎必然拒绝所有候选**，通常需要随温度缩小步长。
        4. ``history`` 记录的是历史最优，不是当前解；用它判断"是否早熟"会看不出来。
        5. 非有限值**有**保护，但保护方式是"拒绝"而不是"报错"：初始解的目标值非有限会直接
           ``raise ValueError``；迭代中候选解的目标值非有限则被当作差解丢弃（该步照常降温、
           ``history`` 记旧值）。所以 NaN 不会污染返回值，但会被**静默忽略**——若 cost 大面积
           返回 NaN，搜索就会退化成在初始解附近空转而不报错，请自己先验证 cost 的有限性。
        6. 温度下溢：``alpha`` 小且 ``iters`` 大时 ``T`` 会下溢成精确的 ``0.0``（如
           ``alpha=0.5`` 约 1075 步之后）。本实现把 ``T == 0`` 定义为"零温极限：只接受非劣解"，
           因此不会抛 ``ZeroDivisionError``；但这之后算法已完全退化为爬山，等价于提前停止。

    参考:
        Kirkpatrick, Gelatt & Vecchi 1983；Černý 1985（Metropolis 准则源自 1953 年 Metropolis 等）。
    """
    if not callable(cost):
        raise ValueError("cost 必须是可调用对象")
    if not callable(neighbor):
        raise ValueError("neighbor 必须是可调用对象")
    T0 = float(T0)
    alpha = float(alpha)
    iters = int(iters)
    if T0 <= 0:
        raise ValueError(f"T0 必须为正，得到 {T0}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 必须在 (0, 1) 内，得到 {alpha}")
    if iters < 0:
        raise ValueError("iters 不能为负")

    gen = make_rng(seed)
    cur = np.asarray(x0, dtype=float).copy()
    if cur.size == 0:
        raise ValueError("x0 不能为空")
    cur_cost = float(cost(cur))
    if not np.isfinite(cur_cost):
        raise ValueError(f"初始解的目标值为 {cur_cost}，不是有限值")

    best = cur.copy()
    best_cost = cur_cost
    history: List[float] = [best_cost]
    T = T0
    for _ in range(iters):
        cand = np.asarray(neighbor(cur, gen), dtype=float)
        cand_cost = float(cost(cand))
        if not np.isfinite(cand_cost):
            # 与 SA 语义一致：非有限值视作"差解"，按 Metropolis 大概率拒绝；这里直接拒绝更明确
            T *= alpha
            history.append(best_cost)
            continue
        delta = cand_cost - cur_cost
        # T 在下溢后会是精确的 0.0：此时 exp(-delta / 0.0) 会抛 ZeroDivisionError，
        # 因此把 T == 0 显式解释为"零温极限：只接受非劣解"。注意这里的短路顺序保证了
        # T == 0 时**不消耗**随机数，从而不改变任何既有种子的随机序列。
        if delta <= 0.0 or (T > 0.0 and gen.random() < np.exp(-delta / T)):
            cur, cur_cost = cand, cand_cost
            if cur_cost < best_cost:
                best, best_cost = cur.copy(), cur_cost
        T *= alpha
        history.append(best_cost)
    return {"best_x": best, "best_cost": float(best_cost), "history": history}


# --------------------------------------------------------------------------- #
# 遗传算法（0/1 编码）
# --------------------------------------------------------------------------- #
def genetic_algorithm(
    fitness: Callable[[np.ndarray], float],
    n_genes: int,
    pop_size: int = 60,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.02,
    seed: Optional[int] = None,
    maximize: bool = True,
    init_density: float = 0.15,
    repair: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> dict:
    """二进制编码遗传算法（锦标赛选择 + 单点交叉 + 位翻转变异）。

    参数:
        fitness: ``fitness(genes) -> float``，``genes`` 是 0/1 的 float 数组。
            本实现保证传入的编码严格是 ``0.0``/``1.0``（交叉与变异都按位操作，不会产生非整数），
            所以你不需要在 fitness 里再做取整。
        n_genes: 基因位数（染色体长度）。
        pop_size: 种群规模，必须 >= 2（少于 2 无法配对）。
        generations: 迭代代数。
        crossover_rate: 交叉概率。
        mutation_rate: **每个基因位**的独立变异概率（不是"每条染色体"的概率）；
            所以期望变异位数 = n_genes * mutation_rate，调参时务必注意这个口径。
        seed: 随机种子。
        maximize: True 求最大值，False 求最小值（内部对适应度取负后按最大化处理）。
        init_density: 初始种群每一位取 1 的概率（默认 0.15）。**不要随手改成 0.5**：
            对 0/1 约束问题（背包等），0.5 密度下绝大多数个体不可行，而 ``-inf`` 的硬罚函数
            让所有不可行个体适应度相同，选择算子会退化成随机（见"陷阱"第 1b 条）。
        repair: 可选钩子 ``repair(genes) -> genes``，用于把不可行个体修回可行域
            （如背包的贪心去项）。必须在**进入种群时**调用并写回（本实现如此），
            否则 ``best_genes`` 与 ``best_fitness`` 会对不上号。必须返回与输入同形状的
            0/1 数组，否则抛 ``ValueError``。

    返回:
        ``{"best_genes": np.ndarray(0/1, dtype=int), "best_fitness": float, "history": list}``。
        ``history`` 长度 = generations + 1，记录每代结束时的**历史最优适应度**（按用户口径，
        即 maximize=False 时也是最小值序列，单调不增）。

    算法:
        1. 初始化：每位独立伯努利(init_density)采样；
        2. 每代：精英保留 1 个 + 二元锦标赛选择父代 → 依概率单点交叉 → 逐位变异
           → 可选 repair → 评估；
        3. 更新历史最优并记录。

    复杂度:
        时间 O(generations * pop_size * n_genes) / 空间 O(pop_size * n_genes)。

    陷阱:
        1. **不保留精英会震荡**：即使种群里出现过最优个体，下一代也可能被交叉/变异破坏。
           本实现显式保留精英，这是最简单的稳定性补丁。
        1b. **全不可行种群会让选择失效**：如果编码为 0/1 且约束用返回 ``-inf`` 的硬罚函数
           表达，那么当整个种群都不可行时所有个体适应度相同（都是 -inf），
           ``argmax`` 只能取到第 0 个，"选择"退化成"随机"。本实现因此：
           (a) 初始种群按 ``init_density=0.15`` 稀疏采样（大幅降低初始不可行率），
           (b) 提供 ``repair`` 钩子在**入种群时**把不可行个体修回可行，并写回种群——
           因此返回的 ``best_genes`` 一定就是被评估的那个个体。见下面的 ``repair`` 参数。
        2. 二进制编码对**连续变量**是有偏的：n_genes 位只能表示 2^n_genes 个格点，
           且格雷码之外的普通二进制在相邻整数间可能有多位翻转（Hamming 悬崖）。
           连续优化请优先用 :func:`particle_swarm` 或实数编码 GA。
        3. 约束必须写进 ``fitness``（罚函数或返回 -inf），否则 GA 会稳定地输出不可行解。
           但注意：返回 ``-inf`` 的个体在锦标赛里会被自动淘汰，属于一种硬约束处理。
        4. 交叉/变异率是"每位概率"还是"每染色体概率"是经典歧义；本实现取每位，已在上面写明。
        5. 早熟收敛：种群多样性丢失后，所有个体相同，交叉失效，只剩变异在局部扰动。
           建议在论文里报告"最优解首次出现的代数"和"种群平均适应度曲线"。

    参考:
        Holland 1975；Goldberg 1989《Genetic Algorithms in Search, Optimization, and Machine Learning》。
    """
    if not callable(fitness):
        raise ValueError("fitness 必须是可调用对象")
    n_genes = int(n_genes)
    pop_size = int(pop_size)
    generations = int(generations)
    if n_genes <= 0:
        raise ValueError("n_genes 必须为正")
    if pop_size < 2:
        raise ValueError("pop_size 至少为 2")
    if generations < 0:
        raise ValueError("generations 不能为负")
    if not (0.0 <= crossover_rate <= 1.0):
        raise ValueError("crossover_rate 应在 [0, 1]")
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate 应在 [0, 1]")
    if not (0.0 < init_density < 1.0):
        raise ValueError("init_density 应在 (0, 1) 内")
    if repair is not None and not callable(repair):
        raise ValueError("repair 必须是可调用对象或 None")

    gen = make_rng(seed)
    sign = 1.0 if maximize else -1.0

    def _apply_repair(arr: np.ndarray) -> np.ndarray:
        """把 repair 钩子**物化**到种群上：保证"评估的个体"就是"返回的个体"。

        陷阱：如果只在评估时临时调用 repair 而不写回，``best_genes`` 可能是未修复的
        不可行个体，而 ``best_fitness`` 却是修复后的分数——两者对不上。
        """
        if repair is None:
            return arr
        fixed = []
        for ind in arr:
            out = np.asarray(repair(ind))
            if out.shape != ind.shape:
                raise ValueError(
                    f"repair 必须返回与输入同形状的数组：输入 {ind.shape}，得到 {out.shape}"
                )
            fixed.append(out.astype(np.int8))
        return np.array(fixed, dtype=np.int8)

    def raw(ind: np.ndarray) -> float:
        """用户口径的适应度（含 maximize 符号翻转）。repair 已在入种群时物化。"""
        return sign * float(fitness(np.asarray(ind, dtype=float)))

    pop = _apply_repair((gen.random((pop_size, n_genes)) < init_density).astype(np.int8))

    scored = np.array([raw(ind) for ind in pop], dtype=float)
    best_idx = int(np.argmax(scored))
    best_genes = pop[best_idx].copy()
    best_score = float(scored[best_idx])
    history: List[float] = [float(sign * best_score)]

    for _ in range(generations):
        new_pop = np.empty_like(pop)
        new_pop[0] = best_genes  # 精英保留
        for k in range(1, pop_size, 2):
            p1 = _tournament(pop, scored, gen)
            p2 = _tournament(pop, scored, gen)
            c1, c2 = p1.copy(), p2.copy()
            if gen.random() < crossover_rate and n_genes > 1:
                point = int(gen.integers(1, n_genes))  # 单点交叉，切点在 [1, n-1]
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            for child in (c1, c2):
                mask = gen.random(n_genes) < mutation_rate
                child[mask] ^= 1
            new_pop[k] = c1
            if k + 1 < pop_size:
                new_pop[k + 1] = c2
        pop = _apply_repair(new_pop)
        scored = np.array([raw(ind) for ind in pop], dtype=float)
        idx = int(np.argmax(scored))
        if scored[idx] > best_score:
            best_score = float(scored[idx])
            best_genes = pop[idx].copy()
        history.append(float(sign * best_score))

    return {
        "best_genes": best_genes.astype(int),
        "best_fitness": float(sign * best_score),
        "history": history,
    }


def _tournament(pop: np.ndarray, scored: np.ndarray, gen: np.random.Generator, k: int = 2) -> np.ndarray:
    """二元锦标赛选择：随机抽 k 个个体，返回其中适应度最高者的副本。"""
    idx = gen.integers(0, pop.shape[0], size=k)
    winner = idx[int(np.argmax(scored[idx]))]
    return pop[winner].copy()


# --------------------------------------------------------------------------- #
# 粒子群
# --------------------------------------------------------------------------- #
def particle_swarm(
    objective: Callable[[np.ndarray], float],
    bounds,
    n_particles: int = 30,
    iters: int = 200,
    seed: Optional[int] = None,
    maximize: bool = False,
) -> dict:
    """粒子群优化（惯性权重线性递减的全局版 PSO）。

    参数:
        objective: ``objective(x) -> float``。
        bounds: 变量边界，两种写法：
            - ``(lo, hi)``：所有维度共用同一区间；
            - ``[(lo1, hi1), (lo2, hi2), ...]``：逐维区间。
            上界必须严格大于下界（否则该维退化为常数，本实现直接报错而不是静默处理）。
        n_particles: 粒子数，>= 2。
        iters: 迭代次数。
        seed: 随机种子。
        maximize: True 求最大，默认 False（最小化）。

    返回:
        ``{"best_x": np.ndarray, "best_value": float, "history": list}``。
        ``history`` 长度 = iters + 1，是按用户口径的**历史最优值**。

    算法:
        标准 PSO 速度更新：``v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)``，
        再 ``x <- clip(x + v, lo, hi)``。惯性权重 ``w`` 从 0.9 线性降到 0.4；
        ``c1 = c2 = 2.0``。初始位置在区间内均匀采样；初速度在**正负**区间宽度的 10% 内
        均匀采样，即每维 ``v_i ~ U(-0.1 * span_i, +0.1 * span_i)``（不是 "0 到 10%"）。

    复杂度:
        时间 O(iters * n_particles * (n_dim + T_objective)) / 空间 O(n_particles * n_dim)。

    陷阱:
        1. **边界处理不是"夹住"就完事**：直接把位置 clip 到边界会让粒子贴着边界"粘住"，
           实践中常同时把该维速度置零或反弹。本实现采用 clip + 速度不变（简单版），
           所以如果最优解恰好在边界上，PSO 常常收敛得慢——这是已知局限，不是 bug。
        2. 还有**初始不可行**问题：粒子初始化必须在区间内，否则第一代 gbest 可能是不可行点。
        3. ``c1/c2`` 过大（如 4.0）会导致速度爆炸、粒子全飞出可行域后被反复 clip，
           表现为"history 一条直线"。压缩因子或速度上限是常见补救。
        4. PSO 对**坐标系旋转**敏感（各维独立更新），变量之间有强耦合时收敛慢，
           可先做主成分旋转或改用 CMA-ES。
        5. 结果不可复现的常见原因是用了全局 ``np.random``；本实现用显式 Generator。

    参考:
        Kennedy & Eberhart 1995；Shi & Eberhart 1998（惯性权重线性递减）。
    """
    if not callable(objective):
        raise ValueError("objective 必须是可调用对象")
    n_particles = int(n_particles)
    iters = int(iters)
    if n_particles < 2:
        raise ValueError("n_particles 至少为 2")
    if iters < 0:
        raise ValueError("iters 不能为负")

    lo, hi = _parse_bounds(bounds)
    n_dim = lo.size
    sign = 1.0 if maximize else -1.0

    gen = make_rng(seed)
    span = hi - lo
    x = lo + gen.random((n_particles, n_dim)) * span
    v = (gen.random((n_particles, n_dim)) - 0.5) * 0.2 * span

    def score(pt: np.ndarray) -> float:
        return sign * float(objective(pt))

    fit = np.array([score(x[i]) for i in range(n_particles)])
    pbest = x.copy()
    pbest_fit = fit.copy()
    g = int(np.argmax(pbest_fit))
    gbest = pbest[g].copy()
    gbest_fit = float(pbest_fit[g])
    history: List[float] = [float(sign * gbest_fit)]

    c1 = c2 = 2.0
    for it in range(iters):
        w = 0.9 - 0.5 * (it / max(iters - 1, 1))  # 线性 0.9 -> 0.4
        r1 = gen.random((n_particles, n_dim))
        r2 = gen.random((n_particles, n_dim))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest[None, :] - x)
        x = np.clip(x + v, lo, hi)
        fit = np.array([score(x[i]) for i in range(n_particles)])
        improved = fit > pbest_fit
        pbest[improved] = x[improved]
        pbest_fit[improved] = fit[improved]
        g = int(np.argmax(pbest_fit))
        if pbest_fit[g] > gbest_fit:
            gbest_fit = float(pbest_fit[g])
            gbest = pbest[g].copy()
        history.append(float(sign * gbest_fit))

    return {"best_x": gbest, "best_value": float(sign * gbest_fit), "history": history}


def _parse_bounds(bounds) -> Tuple[np.ndarray, np.ndarray]:
    """把 bounds 规范成 (lo, hi) 两个一维数组，并统一校验有限性与 ``hi > lo``。

    陷阱:
        校验必须放在两种写法**汇合之后**。早期版本在 ``(lo, hi)`` 简写分支里直接
        ``return``，于是 ``bounds=(5, 3)``（退化盒）与 ``bounds=(nan, 5)``（非有限）
        都能绕过校验，PSO/DE/GWO 会静默返回 NaN 结果或空转，而不是按文档抛 ValueError。
    """
    if bounds is None:
        raise ValueError("bounds 不能为 None（PSO 必须有搜索盒）")
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("bounds 为 (lo, hi) 时长度必须为 2")
        lo = np.array([arr[0]], dtype=float)
        hi = np.array([arr[1]], dtype=float)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        lo = arr[:, 0].copy()
        hi = arr[:, 1].copy()
    else:
        raise ValueError(f"bounds 形状非法：{arr.shape}，应为 (lo, hi) 或 [(lo, hi), ...]")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("bounds 必须是有限值")
    if np.any(hi <= lo):
        raise ValueError("每一维都要求 hi > lo；退化维度请先固定变量再优化")
    return lo, hi


# --------------------------------------------------------------------------- #
# 蚁群 TSP
# --------------------------------------------------------------------------- #
def ant_colony_tsp(
    dist,
    n_ants: int = 20,
    iters: int = 100,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.5,
    seed: Optional[int] = None,
) -> dict:
    """蚁群算法求解对称 TSP（Ant System：概率构造 + 信息素挥发 + 优质边加强）。

    参数:
        dist: (n, n) 距离矩阵，``dist[i, j] >= 0``，须对称（非对称矩阵请改用其他模型）。
        n_ants: 每代蚂蚁数，>= 1，通常取城市数。
        iters: 迭代（代）数。
        alpha: 信息素重要程度指数（alpha=0 表示不看信息素，退化为贪心）。
        beta: 启发式（1/距离）重要程度指数。beta 太大 → 早期就贪心、易早熟。
        rho: 信息素挥发率，取值 (0, 1)；``rho=1`` 会抹掉全部历史信息。
        seed: 随机种子。

    返回:
        ``{"tour": list, "length": float, "history": list}``。``history`` 长度 = iters + 1，
        记录每代结束时的历史最优回路长度（单调不增）。

    算法:
        1. 信息素 ``tau`` 初始化为 1/n；
        2. 每只蚂蚁从随机城市出发，按 ``tau^alpha * (1/d)^beta`` 概率选择下一城（只走未访问城市）；
        3. 一代结束后全体挥发 ``tau *= (1 - rho)``，并按 ``Q / L_k`` 对每只蚂蚁走过的边加强；
        4. 记录历史最优。``Q`` 取 1.0。

    复杂度:
        时间 O(iters * n_ants * n^2) / 空间 O(n^2)。

    陷阱:
        1. **一个蚂蚁一批城市的选择概率必须重新归一化**，且已访问城市的概率置 0。若所有
           候选的概率权重都是 0（例如距离为 0 或 tau 下溢），必须显式回退成"在未访问城市中
           均匀随机"，否则 ``p.sum() == 0`` 会除零得到 NaN 并污染整个信息素矩阵。
        2. 信息素下溢：长时间挥发后 ``tau`` 可能小于浮点精度而变成 0，此时概率权重全 0，
           蚂蚁行为退化为纯随机（见上一条的回退分支）。实践中常设 ``tau_min`` 下限。
        3. 对称性假设：本实现按 ``tau[i, j] = tau[j, i]`` 更新。非对称距离矩阵下
           信息素会混淆方向，结果无效。
        4. 距离为 0 的边会使 ``1/d`` 变成 inf；本实现用 ``safe_divide`` 思路把 0 距离的
           启发式权重设为一个大数（1e12）而不是 inf，避免 NaN。
        5. ``n_ants`` 太小（如 1）时信息素更新噪声极大，等于随机重启爬山；通常取 n_ants ≈ n。

    参考:
        Dorigo, Maniezzo & Colorni 1996, "Ant System: Optimization by a Colony of Cooperating Agents"。
    """
    D = np.asarray(dist, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"dist 必须是方阵，得到形状 {D.shape}")
    n = D.shape[0]
    if n == 0:
        raise ValueError("dist 不能为空")
    if np.any(np.isnan(D)) or np.any(D < 0):
        raise ValueError("dist 不能含 NaN 或负距离")
    n_ants = int(n_ants)
    iters = int(iters)
    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    if n_ants < 1:
        raise ValueError("n_ants 至少为 1")
    if iters < 0:
        raise ValueError("iters 不能为负")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha / beta 必须非负")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho 必须在 (0, 1) 内，得到 {rho}")

    gen = make_rng(seed)
    if n == 1:
        # 单城市退化算例：没有边可走，长度恒为 0。history 仍按文档承诺返回 iters + 1 项，
        # 否则调用方按固定索引取收敛曲线尾值时会越界。
        return {"tour": [0], "length": 0.0, "history": [0.0] * (iters + 1)}

    # 启发式权重 eta[i, j] = 1 / d(i, j)，零距离用大数代替 inf
    with np.errstate(divide="ignore"):
        eta = np.where(D > 0, 1.0 / np.where(D > 0, D, 1.0), 1e12)
    np.fill_diagonal(eta, 0.0)

    tau = np.full((n, n), 1.0 / n)
    best_tour: Optional[List[int]] = None
    best_len = float("inf")
    history: List[float] = []

    all_cities = np.arange(n)
    for _ in range(iters + 1):
        tours: List[List[int]] = []
        lengths: List[float] = []
        for _ant in range(n_ants):
            start = int(gen.integers(0, n))
            tour = [start]
            visited = np.zeros(n, dtype=bool)
            visited[start] = True
            cur = start
            for _step in range(n - 1):
                weights = (tau[cur] ** alpha) * (eta[cur] ** beta)
                weights[visited] = 0.0
                total = float(weights.sum())
                if total <= 0.0 or not np.isfinite(total):
                    # 回退：在未访问城市中均匀随机（防止除零 / 信息素下溢）
                    cand = all_cities[~visited]
                    nxt = int(cand[gen.integers(0, cand.size)])
                else:
                    prob = weights / total
                    nxt = int(gen.choice(n, p=prob))
                    if visited[nxt]:  # 数值兜底，理论上不会发生
                        cand = all_cities[~visited]
                        nxt = int(cand[gen.integers(0, cand.size)])
                tour.append(nxt)
                visited[nxt] = True
                cur = nxt
            L = _cycle_length(tour, D)
            tours.append(tour)
            lengths.append(L)
            if L < best_len:
                best_len = L
                best_tour = list(tour)

        # 挥发 + 加强
        tau *= (1.0 - rho)
        for tour, L in zip(tours, lengths):
            if L <= 0:
                continue
            gain = 1.0 / L
            for k in range(n):
                a, b = tour[k], tour[(k + 1) % n]
                tau[a, b] += gain
                tau[b, a] += gain
        history.append(float(best_len))

    return {"tour": list(best_tour), "length": float(best_len), "history": history}


def _cycle_length(tour: Sequence[int], D: np.ndarray) -> float:
    """闭环回路长度（tour 不含重复起点）。"""
    n = len(tour)
    if n <= 1:
        return 0.0
    return float(sum(D[tour[k], tour[(k + 1) % n]] for k in range(n)))


# --------------------------------------------------------------------------- #
# 差分进化
# --------------------------------------------------------------------------- #
def differential_evolution(
    objective: Callable[[np.ndarray], float],
    bounds,
    pop_size: int = 40,
    n_gen: int = 100,
    F: float = 0.7,
    CR: float = 0.9,
    seed: Optional[int] = None,
) -> dict:
    """差分进化 DE/rand/1/bin（实数编码，最小化 ``objective``）。

    参数:
        objective: ``objective(x) -> float``，目标函数，**约定为最小化**（要最大化请自己取负）。
        bounds: 搜索盒，口径与 :func:`particle_swarm` 完全一致：
            ``(lo, hi)`` 表示所有维度共用区间，``[(lo1, hi1), (lo2, hi2), ...]`` 表示逐维区间；
            每维要求 ``hi > lo``。
        pop_size: 种群规模，必须 >= 4（``DE/rand/1`` 要 3 个与目标个体互不相同的供体）。
        n_gen: 迭代代数。总求值次数 **恰好** ``pop_size * (n_gen + 1)``（初始一代 + 每代每体一次）。
        F: 差分缩放因子，取值 (0, 2]，常用 0.5~0.9。越大探索越猛、越容易跳出局部最优，
            但也更容易拒绝（收敛慢）。
        CR: 交叉概率，取值 [0, 1]。注意口径是**逐分量**（binomial）而不是逐个体：
            期望有 ``CR * n_dim`` 个分量取自变异体，且强制至少一个分量取自变异体。
        seed: 随机种子，None 用 ``DEFAULT_SEED``。

    返回:
        ``{"x": np.ndarray, "fun": float, "n_eval": int, "history": List[float]}``。
        ``fun`` 是 ``x`` 处的目标值；``n_eval`` 是目标函数**实际求值次数**（写论文报预算用）；
        ``history`` 长度 = n_gen + 1，元素是每代结束时的历史最优值（单调不增）。

    算法:
        1. 初始种群在 ``[lo, hi]`` 内独立均匀采样，逐个体求值；
        2. 每代对每个个体 ``i``：随机抽取互不相同且不等于 ``i`` 的 ``a, b, c``，做
           ``v = x_a + F * (x_b - x_c)``（rand/1），随后把每个越界分量**拉回边界**
           （有界约束的标准做法；不做反射/惩罚）；
        3. binomial 交叉：``u_j = v_j`` 若 ``rand_j < CR``，否则 ``u_j = x_i,j``；
           再随机指定一个分量强制取 ``v_j``，保证试验体与父代至少差一个分量；
        4. 贪心选择：``fun(u) <= fun(x_i)`` 时用 ``u`` 替换 ``x_i``（相等也替换，
           这样种群能沿等值面漂移，是 DE 的常见实现细节）；
        5. 每代结束更新历史最优。

    复杂度:
        时间 O(n_gen * pop_size * (n_dim + T_objective)) / 空间 O(pop_size * n_dim + n_gen)。

    陷阱:
        1. **越界处理会决定成败**：本实现在变异后直接 ``clip`` 到边界。若最优解落在边界上，
           大量分量会堆积在边界平面上（"边界锁死"），表现为种群多样性骤降；
           DE 的文献里更常见的是反射（``lo + (lo - v)``）或随机重采样。这里选最简单的一种，
           并在 docstring 里写明口径。
        2. ``pop_size`` 太小（< 4）会让供体索引去重后取不满 3 个不同个体，代码直接报错而不是
           静默重复使用同一个体（重复会让差分向量恒为 0，搜索退化成随机游走）。
        3. ``CR=1`` 且 ``F`` 很小时，试验体几乎等于 ``x_a + F*(x_b-x_c)``，父代信息被完全丢弃，
           表现为收敛快但早早停滞；``CR`` 太低（如 0.1）则每代只改 1 个分量，高维下慢得离谱。
        4. ``history`` 是历史最优，单调不增，画出来很"好看"；论文里更应报告每代种群**均值**
           与标准差（多样性代理指标），否则看不出早熟。
        5. 目标函数返回 NaN 时所有比较都是 False，个体会被永久保留；本实现不做保护
           （与 :func:`simulated_annealing` 一致，调用方需保证返回有限值）。

    参考:
        Storn & Price 1997, "Differential Evolution - A Simple and Efficient Heuristic for
        Global Optimization over Continuous Spaces"； Qin, Huang & Suganthan 2009（自适应 F/CR）。
    """
    if not callable(objective):
        raise ValueError("objective 必须是可调用对象")
    pop_size = int(pop_size)
    n_gen = int(n_gen)
    F = float(F)
    CR = float(CR)
    if pop_size < 4:
        raise ValueError(f"pop_size 至少为 4（DE/rand/1 需要 3 个互不相同的供体），得到 {pop_size}")
    if n_gen < 0:
        raise ValueError(f"n_gen 不能为负，得到 {n_gen}")
    if not (0.0 < F <= 2.0):
        raise ValueError(f"F 必须在 (0, 2] 内，得到 {F}")
    if not (0.0 <= CR <= 1.0):
        raise ValueError(f"CR 必须在 [0, 1] 内，得到 {CR}")

    lo, hi = _parse_bounds(bounds)
    n_dim = lo.size
    gen = make_rng(seed)
    span = hi - lo

    pop = lo + gen.random((pop_size, n_dim)) * span
    fit = np.array([float(objective(pop[i])) for i in range(pop_size)], dtype=float)
    n_eval = pop_size

    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx].copy()
    best_fun = float(fit[best_idx])
    history: List[float] = [best_fun]

    for _ in range(n_gen):
        for i in range(pop_size):
            # 抽 3 个 [0, pop_size-2] 的下标再整体跳过 i，保证 a/b/c 互不相同且都 != i
            picks = gen.choice(pop_size - 1, size=3, replace=False)
            picks = np.where(picks >= i, picks + 1, picks)
            a, b, c = int(picks[0]), int(picks[1]), int(picks[2])
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, lo, hi)
            cross = gen.random(n_dim) < CR
            cross[int(gen.integers(0, n_dim))] = True  # 至少一个分量来自变异体
            trial = np.where(cross, mutant, pop[i])
            trial_fun = float(objective(trial))
            n_eval += 1
            if trial_fun <= fit[i]:
                pop[i] = trial
                fit[i] = trial_fun
        g = int(np.argmin(fit))
        if fit[g] < best_fun:
            best_fun = float(fit[g])
            best_x = pop[g].copy()
        history.append(best_fun)

    return {
        "x": best_x,
        "fun": float(best_fun),
        "n_eval": int(n_eval),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# 禁忌搜索
# --------------------------------------------------------------------------- #
def _solution_key(x: np.ndarray) -> tuple:
    """把解向量转成可哈希的禁忌表键（保留 12 位小数，避免浮点噪声造成"假新解"）。"""
    arr = np.asarray(x, dtype=float).ravel()
    return tuple(round(float(v), 12) for v in arr)


def tabu_search(
    init,
    neighbors_fn: Callable[[np.ndarray], List[np.ndarray]],
    objective: Callable[[np.ndarray], float],
    max_iter: int = 200,
    tabu_tenure: int = 7,
    seed: Optional[int] = None,
) -> dict:
    """禁忌搜索（best-improvement + 解禁忌表 + 藐视准则），最小化 ``objective``。

    参数:
        init: 初始解（一维向量；组合问题请自行编码成有序数组，例如 TSP 的城市访问序列）。
        neighbors_fn: ``neighbors_fn(x) -> list[候选解]``，返回当前解的**全部或部分**邻域。
            **不接收 rng**：邻域生成必须是确定性的（这是本模块对禁忌搜索的口径），
            ``seed`` 只用在一处——多个候选目标值完全相同时的随机打破平局。
        objective: ``objective(x) -> float``，约定为最小化。
        max_iter: 最大迭代步数；没有任何候选解时提前结束，故实际 ``n_iter <= max_iter``。
        tabu_tenure: 禁忌长度（代数），>= 1。在第 ``t`` 步接受解 ``s`` 后，
            ``s`` 在 ``t+1 .. t+tabu_tenure-1`` 步内被禁用。
        seed: 随机种子，None 用 ``DEFAULT_SEED``。

    返回:
        ``{"x": np.ndarray, "fun": float, "n_iter": int, "history": List[float], "n_better": int}``。
        ``x``/``fun`` 是搜索过程中找到的**历史最优**（不是最后停留的解——禁忌搜索允许走差解）；
        ``history`` 长度 = n_iter + 1，为历史最优序列（单调不增）；
        ``n_better`` 是"当前解被改得比上一步更好"的步数，可用来说明搜索有多活跃。

    算法:
        1. 令 cur = init，清空禁忌表；
        2. 每步枚举 ``neighbors_fn(cur)``，跳过非有限候选；候选若在禁忌表中且**不优于**
           历史最优则丢弃（藐视准则 aspiration：优于历史最优的禁忌解仍可取用）。
           注意这条藐视准则在**非退化输入下不可达**：写进禁忌表的那个解在写入时刻必然
           ``f >= best_fun``，而 ``best_fun`` 单调不增，所以"禁忌且未优于历史最优"恒成立。
           它只在浮点键碰撞的退化场景下才真正生效（见陷阱 4：邻域步长小于 1e-12 时两个
           不同解会共用同一个键），保留该分支是为了让那种场景下的搜索还能继续；
        3. 在剩下的候选中取目标值最小者（并列时用 ``seed`` 决定的生成器随机选一个），
           **无条件**移动过去（即使它比当前解差——这正是禁忌搜索跳出局部最优的机制）；
        4. 把新解写入禁忌表（解禁代数 = 当前步 + tabu_tenure），清理过期条目，更新历史最优。

    复杂度:
        时间 O(max_iter * (|N| * (T_neighbor + T_objective) + |禁忌表|)) / 空间 O(|禁忌表| + max_iter)。

    陷阱:
        1. **本实现禁忌的是"解"（属性禁忌）而不是"移动"**：对置换类问题，交换 (i,j) 与交换
           (j,i) 会得到同一个解，按解禁忌能顺带禁掉逆向移动；但对连续问题，只要邻域步长足够
           小就会不断产生"没被禁忌过的新点"，禁忌表几乎不起作用。要用基于移动的禁忌表，
           请把移动信息编进解向量或改用自己的实现。
        2. **必须允许走差解**。若加上"只接受改进"的条件，禁忌搜索就退化成爬山，
           禁忌表毫无意义——本实现显式不做这个限制。
        3. 禁忌长度是敏感参数：太短（如 1~2）会立刻回访刚离开的解，形成 2-循环；
           太长（远大于邻域规模）会禁掉几乎整个邻域，每步都被迫走差解。
           经验值是邻域规模的平方根量级。
        4. 浮点解的哈希：本实现把解四舍五入到 12 位小数再作键。若邻域步长小于 1e-12，
           两个"不同"的解会被判为同一个而互相禁忌。
        5. 目标函数返回 NaN 的候选会被丢弃（``np.isfinite`` 判定），而不是让整张禁忌表被污染。
        6. ``neighbors_fn`` 返回空列表时直接停止（``n_iter`` 相应变小），不会死循环。

    参考:
        Glover 1986, "Future paths for integer programming and links to artificial intelligence"；
        Glover & Laguna 1997《Tabu Search》。
    """
    if not callable(objective):
        raise ValueError("objective 必须是可调用对象")
    if not callable(neighbors_fn):
        raise ValueError("neighbors_fn 必须是可调用对象")
    max_iter = int(max_iter)
    tabu_tenure = int(tabu_tenure)
    if max_iter < 0:
        raise ValueError(f"max_iter 不能为负，得到 {max_iter}")
    if tabu_tenure < 1:
        raise ValueError(f"tabu_tenure 至少为 1，得到 {tabu_tenure}")

    gen = make_rng(seed)
    cur = as_vector(init, "init")
    cur_fun = float(objective(cur))
    if not np.isfinite(cur_fun):
        raise ValueError(f"初始解的目标值为 {cur_fun}，不是有限值")

    best_x = cur.copy()
    best_fun = cur_fun
    history: List[float] = [best_fun]
    tabu: Dict[tuple, int] = {}
    n_iter = 0
    n_better = 0

    for it in range(max_iter):
        cands = neighbors_fn(cur)
        if cands is None:
            raise ValueError("neighbors_fn 必须返回候选解序列，得到 None")
        chosen: Optional[np.ndarray] = None
        chosen_fun = float("inf")
        ties: List[np.ndarray] = []
        for cand in cands:
            arr = np.asarray(cand, dtype=float).ravel()
            if arr.size != cur.size or not np.all(np.isfinite(arr)):
                continue  # 形状不符或非有限的候选视作不可行，直接丢弃
            f = float(objective(arr))
            if not np.isfinite(f):
                continue
            if tabu.get(_solution_key(arr), -1) > it and f >= best_fun:
                continue  # 禁忌且未触发藐视准则
            if f < chosen_fun - 1e-12:
                chosen, chosen_fun, ties = arr, f, [arr]
            elif abs(f - chosen_fun) <= 1e-12:
                ties.append(arr)
        if chosen is None:
            break  # 邻域为空或全被禁忌：停止（n_iter 不增加）
        n_iter = it + 1
        if len(ties) > 1:
            chosen = ties[int(gen.integers(0, len(ties)))]
        if chosen_fun < cur_fun - 1e-12:
            n_better += 1
        cur, cur_fun = chosen, chosen_fun
        tabu[_solution_key(cur)] = it + tabu_tenure
        tabu = {k: v for k, v in tabu.items() if v > it}  # 清理过期条目
        if cur_fun < best_fun:
            best_fun = float(cur_fun)
            best_x = cur.copy()
        history.append(best_fun)

    return {
        "x": best_x,
        "fun": float(best_fun),
        "n_iter": int(n_iter),
        "history": history,
        "n_better": int(n_better),
    }


# --------------------------------------------------------------------------- #
# 灰狼优化
# --------------------------------------------------------------------------- #
def grey_wolf_optimizer(
    objective: Callable[[np.ndarray], float],
    bounds,
    n_wolves: int = 30,
    n_gen: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """灰狼优化 GWO（社会等级 alpha/beta/delta 引导包围式搜索），最小化 ``objective``。

    参数:
        objective: ``objective(x) -> float``，约定为最小化。
        bounds: 搜索盒，口径与 :func:`particle_swarm` / :func:`differential_evolution` 一致。
        n_wolves: 狼群规模，必须 >= 3（要选出 alpha/beta/delta 三匹头狼）。
        n_gen: 迭代代数；总求值次数 = n_wolves * (n_gen + 1)。
        seed: 随机种子，None 用 ``DEFAULT_SEED``。

    返回:
        ``{"x": np.ndarray, "fun": float, "history": List[float]}``。
        ``x``/``fun`` 是历史最优（即 alpha 狼曾到达的最好位置）；
        ``history`` 长度 = n_gen + 1，元素为每代结束时的 alpha 目标值（单调不增）。

    算法:
        1. 在 ``[lo, hi]`` 内均匀初始化 ``n_wolves`` 匹狼并求值，按目标值升序取
           alpha / beta / delta 三匹头狼（GWO 的"等级"结构，注意它**不是** elitism：
           头狼也会被更新）；
        2. 每代令 ``a`` 从 2 线性降到 0，对每匹狼 ``i``、每匹头狼 ``L in {alpha, beta, delta}``：
           ``A = 2*a*r1 - a``，``C = 2*r2``（``r1, r2 ~ U(0,1)^d``），
           ``D_L = |C * X_L - X_i|``，``X_L' = X_L - A * D_L``；
        3. 新位置取三匹头狼引导位置的算术平均 ``(X_alpha' + X_beta' + X_delta') / 3``，
           再 ``clip`` 回搜索盒；
        4. 全体求值后把"当前狼群 + 旧头狼"一起排序，重新选出 alpha/beta/delta
           （这一步保证历史最优不丢失），记录 alpha 值。

    复杂度:
        时间 O(n_gen * n_wolves * n_dim)（外加等量目标函数求值）/ 空间 O(n_wolves * n_dim + n_gen)。

    陷阱:
        1. **GWO 没有显式的个体记忆**（没有 pbest）：狼的位置可以变差，只有三匹头狼保留了
           历史信息。因此它比 PSO 更容易"丢失"已经找到的好解——本实现在每代重排时把旧头狼
           一起纳入排序，正是为了补这个洞（否则 ``history`` 可能不是单调不增的）。
        2. ``|A| < 1`` 时狼群向头狼收缩（开发），``|A| > 1`` 时远离头狼（探索）；``a`` 线性递减
           意味着探索窗口只有前半程。注意本实现里 ``a = 2 - 2 * it / max(n_gen, 1)``，而
           ``it`` 只取 ``0 .. n_gen - 1``，所以 ``a`` 从 2 降到 ``2 / n_gen``，**末日并不会到 0**
           （``n_gen = 1`` 时 ``a`` 恒为 2，全程都是探索）。``n_gen`` 太小会让整个搜索
           停留在探索阶段。
        3. 越界处理与 DE 一样是简单 ``clip``，最优解在边界上时狼群会贴边。
        4. 三匹头狼由**排序**选出，目标值并列时排序不稳定会让不同 numpy 版本给出不同结果；
           本实现用 ``kind="stable"`` 固定口径以保证同种子可复现。
        5. GWO 在低维、单峰问题上表现很好，对高维 Rastrigin 这类"多峰 + 变量可分离"的问题
           容易陷入局部最优；不要把它当成"一定优于 DE/PSO"的算法。

    参考:
        Mirjalili, Mirjalili & Lewis 2014, "Grey Wolf Optimizer", Advances in Engineering Software。
    """
    if not callable(objective):
        raise ValueError("objective 必须是可调用对象")
    n_wolves = int(n_wolves)
    n_gen = int(n_gen)
    if n_wolves < 3:
        raise ValueError(f"n_wolves 至少为 3（需要选出 alpha/beta/delta），得到 {n_wolves}")
    if n_gen < 0:
        raise ValueError(f"n_gen 不能为负，得到 {n_gen}")

    lo, hi = _parse_bounds(bounds)
    n_dim = lo.size
    gen = make_rng(seed)

    X = lo + gen.random((n_wolves, n_dim)) * (hi - lo)
    fit = np.array([float(objective(X[i])) for i in range(n_wolves)], dtype=float)

    order = np.argsort(fit, kind="stable")
    alpha = X[order[0]].copy()
    alpha_fit = float(fit[order[0]])
    beta = X[order[1]].copy()
    beta_fit = float(fit[order[1]])
    delta = X[order[2]].copy()
    delta_fit = float(fit[order[2]])
    history: List[float] = [alpha_fit]

    for it in range(n_gen):
        a = 2.0 - 2.0 * (it / max(n_gen, 1))  # 2 -> 0 线性递减
        for i in range(n_wolves):
            r1 = gen.random(n_dim)
            r2 = gen.random(n_dim)
            A1 = 2.0 * a * r1 - a
            C1 = 2.0 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - X[i])
            r1 = gen.random(n_dim)
            r2 = gen.random(n_dim)
            A2 = 2.0 * a * r1 - a
            C2 = 2.0 * r2
            X2 = beta - A2 * np.abs(C2 * beta - X[i])
            r1 = gen.random(n_dim)
            r2 = gen.random(n_dim)
            A3 = 2.0 * a * r1 - a
            C3 = 2.0 * r2
            X3 = delta - A3 * np.abs(C3 * delta - X[i])
            X[i] = np.clip((X1 + X2 + X3) / 3.0, lo, hi)

        fit = np.array([float(objective(X[i])) for i in range(n_wolves)], dtype=float)
        # 旧头狼一起参与排序：避免"头狼位置被改坏后历史最优丢失"
        pool_X = np.vstack([X, alpha[None, :], beta[None, :], delta[None, :]])
        pool_fit = np.concatenate([fit, np.array([alpha_fit, beta_fit, delta_fit])])
        order = np.argsort(pool_fit, kind="stable")
        alpha, alpha_fit = pool_X[order[0]].copy(), float(pool_fit[order[0]])
        beta, beta_fit = pool_X[order[1]].copy(), float(pool_fit[order[1]])
        delta, delta_fit = pool_X[order[2]].copy(), float(pool_fit[order[2]])
        history.append(alpha_fit)

    return {"x": alpha, "fun": float(alpha_fit), "history": history}


# --------------------------------------------------------------------------- #
# 变邻域搜索
# --------------------------------------------------------------------------- #
def variable_neighborhood_search(
    init,
    neighborhoods: Sequence[Callable[[np.ndarray, np.random.Generator], np.ndarray]],
    objective: Callable[[np.ndarray], float],
    max_iter: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """变邻域搜索 VNS（多尺度邻域轮换 + 只接受改进解的下降式搜索），最小化 ``objective``。

    参数:
        init: 初始解（一维向量）。
        neighborhoods: 扰动函数列表 ``[perturb(x, rng) -> 候选]``。**必须接收 rng**
            （与 :func:`simulated_annealing` 的 neighbor 同一约定），否则同一种子无法复现。
            列表中各元素的"扰动尺度"应依次变小（大尺度负责跳出局部最优，小尺度负责精修）。
        objective: ``objective(x) -> float``，约定为最小化。
        max_iter: 迭代步数；每步尝试**全部**邻域各一次。本实现**没有提前终止**（与
            :func:`tabu_search` 不同），因此返回值里的 ``n_iter`` 恒等于 ``max_iter``
            （仅当 ``max_iter = 0`` 时为 0）。
        seed: 随机种子，None 用 ``DEFAULT_SEED``。

    返回:
        ``{"x": np.ndarray, "fun": float, "n_iter": int, "history": List[float]}``。
        本实现只接受严格改进的解，故 ``x`` 既是历史最优也是最终解；
        ``history`` 长度 = n_iter + 1，为历史最优序列（单调不增）。

    算法:
        1. 令 cur = init，求值；
        2. 每步用 ``rng`` 打乱邻域顺序，对**每个**邻域各生成 1 个候选并求值；
        3. 在所有候选中取目标值最小者，若它比当前解至少好 1e-12 则接受并移动，
           否则本步不动（仅记录历史最优）；
        4. 重复至 ``max_iter`` 步。

    复杂度:
        时间 O(max_iter * |neighborhoods| * (T_perturb + T_objective)) / 空间 O(max_iter)。

    陷阱:
        1. **本实现是"下降式"VNS：只接受改进解**，因此它**不能**像禁忌搜索那样主动走差解。
           在强多峰问题上会卡在局部最优——它的作用是把"邻域结构本身"变成搜索空间的一部分
           （在大尺度邻域里跳、在小尺度邻域里精修），而不是提供逃逸机制。
        2. 邻域顺序在每步被随机打乱，但候选是**跨邻域取最优**（best improvement），
           不是"第一个改进就接受"。若改成 first improvement，收敛更快但更容易偏向大尺度邻域。
        3. 邻域尺度必须分层：全部用一个尺度等价于固定步长的随机搜索；尺度跨度过大时，
           大邻域产生的候选几乎总被拒绝，实际只有小邻域在工作（白白浪费求值预算）。
        4. 邻域函数返回的候选长度必须与当前解一致，否则抛 ``ValueError``；
           返回非有限值（NaN/inf）的候选会被跳过。**不做**边界裁剪——有界问题请让扰动函数
           自己处理边界（例如内部 ``np.clip``）。
        5. 严格不等号（好 1e-12 以上才接受）意味着在平坦区域搜索会完全停住；
           这是有意的，避免在等值面上无意义地消耗预算。

    参考:
        Mladenović & Hansen 1997, "Variable neighborhood search", Computers & Operations Research；
        Hansen, Mladenović & Moreno Pérez 2010（VNS 变体综述）。
    """
    if not callable(objective):
        raise ValueError("objective 必须是可调用对象")
    if neighborhoods is None:
        raise ValueError("neighborhoods 不能为 None")
    nbrs = list(neighborhoods)
    if not nbrs:
        raise ValueError("neighborhoods 不能为空列表")
    for k, fn in enumerate(nbrs):
        if not callable(fn):
            raise ValueError(f"neighborhoods[{k}] 必须是可调用对象（perturb(x, rng) -> 候选）")
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError(f"max_iter 不能为负，得到 {max_iter}")

    gen = make_rng(seed)
    cur = as_vector(init, "init")
    cur_fun = float(objective(cur))
    if not np.isfinite(cur_fun):
        raise ValueError(f"初始解的目标值为 {cur_fun}，不是有限值")

    best_x = cur.copy()
    best_fun = cur_fun
    history: List[float] = [best_fun]
    n_iter = 0

    for it in range(max_iter):
        n_iter = it + 1
        chosen: Optional[np.ndarray] = None
        chosen_fun = cur_fun  # 只有严格改进才会被选中
        for k in gen.permutation(len(nbrs)):
            cand = np.asarray(nbrs[int(k)](cur, gen), dtype=float).ravel()
            if cand.size != cur.size:
                raise ValueError(
                    f"邻域 {int(k)} 返回的解长度为 {cand.size}，与当前解长度 {cur.size} 不一致"
                )
            if not np.all(np.isfinite(cand)):
                continue
            f = float(objective(cand))
            if f < chosen_fun - 1e-12:
                chosen, chosen_fun = cand, f
        if chosen is not None:
            cur, cur_fun = chosen, chosen_fun
            if cur_fun < best_fun:
                best_fun = float(cur_fun)
                best_x = cur.copy()
        history.append(best_fun)

    return {"x": best_x, "fun": float(best_fun), "n_iter": int(n_iter), "history": history}


# --------------------------------------------------------------------------- #
# 标准测试函数与算法对比台
# --------------------------------------------------------------------------- #
#: ``benchmark_functions`` 支持的标准测试函数名（全部是**最小化**问题）。
_BENCHMARK_NAMES: Tuple[str, ...] = (
    "sphere",
    "rastrigin",
    "ackley",
    "rosenbrock",
    "griewank",
    "schwefel",
    "styblinski_tang",
)

#: name -> (lo, hi, 允许的最小维数, 文献已知全局最优值；None 表示按维数换算)
_BENCHMARK_META: Dict[str, Tuple[float, float, int, Optional[float]]] = {
    "sphere": (-5.12, 5.12, 1, 0.0),
    "rastrigin": (-5.12, 5.12, 1, 0.0),
    "ackley": (-32.768, 32.768, 1, 0.0),
    "rosenbrock": (-2.048, 2.048, 2, 0.0),
    "griewank": (-600.0, 600.0, 1, 0.0),
    "schwefel": (-500.0, 500.0, 1, 0.0),
    # 每维最优 -39.16616570377142（即 -2.9035340181859605 处的 0.5*(x^4-16x^2+5x)），
    # 文献里常写作四舍五入后的 -39.16599，本模块用前者以便 func(optimal_x) 与它精确闭合。
    "styblinski_tang": (-5.0, 5.0, 1, None),
}

#: name -> 全局最优解的**各分量公共取值**（所有函数的最优解都是等分量点）。
_BENCHMARK_OPTIMAL_X: Dict[str, float] = {
    "sphere": 0.0,
    "rastrigin": 0.0,
    "ackley": 0.0,
    "rosenbrock": 1.0,
    "griewank": 0.0,
    "schwefel": 420.968746,
    # g'(x) = 0.5 * (4x^3 - 32x + 5) = 0 的负根，用牛顿法解到双精度极限
    "styblinski_tang": -2.9035340181859605,
}


def benchmark_functions(name: str = "sphere", dim: int = 2) -> dict:
    """返回标准测试函数及其**独立可核对的**已知全局最优，供算法验证/对比使用。

    参数:
        name: 测试函数名，取值见 ``_BENCHMARK_NAMES``：``sphere``、``rastrigin``、
            ``ackley``、``rosenbrock``、``griewank``、``schwefel``、``styblinski_tang``。
            大小写与首尾空白不敏感。
        dim: 维数，>= 1；``rosenbrock`` 至少 2 维。

    返回:
        ``{"name": str, "dim": int, "bounds": list, "func": callable,
        "optimal_x": list, "optimal_value": float}``。
        ``bounds`` 是逐维区间 ``[(lo, hi), ...]``，**可直接喂给** :func:`particle_swarm`、
        :func:`differential_evolution`、:func:`grey_wolf_optimizer`；
        ``func(x) -> float`` 是**纯函数**（无随机、无状态，同一输入永远同一输出），
        输入一维 array_like、输出 Python float，约定为**最小化**；
        ``optimal_x`` / ``optimal_value`` 取自文献的解析最优，不是本模块算出来的。
        注意 ``func`` 是 callable，**不能直接 json 序列化**。

    算法:
        1. 名字先规范化再查表，未知名字直接报错并列出全部可用名字（不做模糊匹配，
            避免"以为在测 Rastrigin 其实测的是 Sphere"这类静默错误）；
        2. 所有函数都在**标准文献搜索盒**内定义（如 sphere 是 [-5.12, 5.12]，
            ackley 是 [-32.768, 32.768]），``bounds`` 直接返回该盒；
        3. ``func`` 逐个实现文献定义的闭式表达式，内部用 :func:`as_vector` 校验输入
            （空输入、含非有限值、非数值输入都会抛 ValueError）；
        4. 最优值：``styblinski_tang`` 是每维 ``-39.16616570377142``（精确解析值，见陷阱 1）
            故按 ``-39.16616570377142 * dim`` 换算，
            其余函数的最优点值都与维数无关。

    复杂度:
        时间 O(dim) / 空间 O(dim)（每次调用 ``func``）。

    陷阱:
        1. **本函数不会去"算"最优值**，``optimal_value`` 是文献常数。用它做"是否收敛到全局最优"
            的判据时，必须自己设容差：``schwefel`` 的最优值在双精度下只能取到 1e-13 量级，
            在 1e-15 级别上**不成立**，建议用
            ``abs(f - optimal) <= 1e-4 * max(1, abs(optimal))`` 之类的相对容差。
            （``styblinski_tang`` 的最优值本模块取的是精确值 ``-39.16616570377142 * dim``，
            文献里常见的 ``-39.16599`` 是它的四舍五入版，两者相对差约 4.4e-6。）
        2. ``rosenbrock`` 的"窄香蕉谷"使得它在 dim=2 尚可、dim>=10 时几乎所有随机算法都失败，
            不要用它单方面宣布某个算法"不行"；它衡量的是**沿谷推进**的能力。
        3. ``griewank`` / ``ackley`` 的最优盆地在标准盒内很窄（griewank 盒是 [-600, 600]，
            而有效谷宽只有个位数），固定步长的局部邻域几乎必然失败——这正是它们的考察点，
            但也很容易被误当成代码 bug。
        4. 标准盒 ``bounds`` 是文献协议的一部分，**不要为了"让算法好看"而缩小它**；
            缩小后与文献结果不可比。
        5. 返回的 ``func`` 不带任何预缩放，``schwefel`` 的量级在 1e3、``sphere`` 在 1e1，
            不同函数的目标值**不能横向比较**。

    参考:
        Jamil & Yang 2013, "A Literature Survey of Benchmark Functions For Global Optimization
        Problems"（标准盒与最优值口径）；
        Yao, Liu & Lin 1999, "Evolutionary Programming Made Faster"（sphere/rastrigin/ackley/
        griewank/schwefel/styblinski_tang 的常用参数与最优值）。
    """
    if not isinstance(name, str):
        raise ValueError(f"name 必须是字符串，得到 {type(name).__name__}")
    key = name.strip().lower()
    if key not in _BENCHMARK_META:
        raise ValueError(
            f"未知的测试函数名 {name!r}；可用：{', '.join(_BENCHMARK_NAMES)}"
        )
    try:
        dim = int(dim)
    except (TypeError, ValueError):
        raise ValueError(f"dim 必须是整数，得到 {dim!r}")
    if dim < 1:
        raise ValueError(f"dim 必须 >= 1，得到 {dim}")

    lo, hi, min_dim, table_opt = _BENCHMARK_META[key]
    if dim < min_dim:
        raise ValueError(f"{key} 至少需要 {min_dim} 维，得到 dim={dim}")

    if key == "sphere":

        def func(x) -> float:
            arr = as_vector(x, "x")
            return float(np.sum(arr ** 2))

    elif key == "rastrigin":

        def func(x) -> float:
            arr = as_vector(x, "x")
            return float(
                10.0 * arr.size + np.sum(arr ** 2 - 10.0 * np.cos(2.0 * np.pi * arr))
            )

    elif key == "ackley":

        def func(x) -> float:
            arr = as_vector(x, "x")
            n = arr.size
            return float(
                -20.0 * np.exp(-0.2 * np.sqrt(np.sum(arr ** 2) / n))
                - np.exp(np.sum(np.cos(2.0 * np.pi * arr)) / n)
                + 20.0
                + np.e
            )

    elif key == "rosenbrock":

        def func(x) -> float:
            arr = as_vector(x, "x")
            return float(
                np.sum(100.0 * (arr[1:] - arr[:-1] ** 2) ** 2 + (1.0 - arr[:-1]) ** 2)
            )

    elif key == "griewank":

        def func(x) -> float:
            arr = as_vector(x, "x")
            idx = np.sqrt(np.arange(1, arr.size + 1, dtype=float))
            return float(np.sum(arr ** 2) / 4000.0 - np.prod(np.cos(arr / idx)) + 1.0)

    elif key == "schwefel":

        def func(x) -> float:
            arr = as_vector(x, "x")
            return float(
                418.9828872724338 * arr.size
                - np.sum(arr * np.sin(np.sqrt(np.abs(arr))))
            )

    else:  # styblinski_tang

        def func(x) -> float:
            arr = as_vector(x, "x")
            return float(0.5 * np.sum(arr ** 4 - 16.0 * arr ** 2 + 5.0 * arr))

    if key == "styblinski_tang":
        # 每维最优值 -39.16616570377142（由最优分量 -2.9035340181859605 精确算出）
        optimal_value = -39.16616570377142 * dim
    else:
        optimal_value = float(table_opt)  # type: ignore[arg-type]

    return {
        "name": key,
        "dim": dim,
        "bounds": [(float(lo), float(hi))] * dim,
        "func": func,
        "optimal_x": [float(_BENCHMARK_OPTIMAL_X[key])] * dim,
        "optimal_value": float(optimal_value),
    }


#: ``benchmark_optimizers`` 支持的求解器（都是实数编码、有界、最小化的那些）。
_BENCH_ALGORITHMS: Tuple[str, ...] = ("sa", "pso", "de", "gwo")

#: ``benchmark_optimizers`` 在 seeds=None 时默认派生多少个随机种子。
_BENCH_N_SEEDS: int = 5

#: 对比台上各算法的种群/群体规模；``budget`` 按它换算代数，使各算法求值次数可比。
_BENCH_SWARM: int = 30

#: SA 的起始点用 ``make_rng(run_seed + 该偏移)`` 均匀采样，避免与 SA 自身的随机流复用；
#: 取一个素数是为了尽量避开 ``seeds`` 里可能出现的相邻整数。
_BENCH_SA_START_OFFSET: int = 7919


def benchmark_optimizers(
    name: str = "sphere",
    dim: int = 2,
    algorithms: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    budget: int = 3000,
    success_tol: Optional[float] = None,
    step_scale: float = 0.1,
    seed: Optional[int] = None,
) -> dict:
    """在**同一预算、同一随机种子集合**下横向对比多个求解器，输出可直接进论文表格的统计量。

    参数:
        name: 测试函数名，口径与 :func:`benchmark_functions` 完全一致。
        dim: 维数。
        algorithms: 要对比的算法名序列，取值 ``"sa"`` / ``"pso"`` / ``"de"`` / ``"gwo"``；
            None 表示全部四个。**不含** GA / ACO / 禁忌搜索 / VNS——它们的编码或邻域是
            0/1、TSP 回路或"由调用方提供的局部算子"，与这里的实数有界最小化协议不同，
            强行混进来会得出不可比的数字（要对比它们请各自构造等价算例）。
        seeds: 每个算法要跑的随机种子序列。给了就用它（这是复现的关键，**必须写进论文**）；
            None 则用 ``seed`` 派生 ``5`` 个种子。
        budget: 每个算法每次运行的**目标函数求值次数上限**（约等于，见下）。
            必须 >= 40，否则当前种群规模连一代都跑不满。
        success_tol: "命中全局最优"的绝对容差。None 时取 ``1e-4 * max(1, |最优值|)``。
        step_scale: 仅 SA 使用：连续邻域步长 = ``step_scale * (hi - lo)``（逐维标准差）。
        seed: 只在 ``seeds=None`` 时生效，用于派生种子集合，保证整体可复现。

    返回:
        ``{"name", "dim", "optimal_value", "success_tol", "budget", "seeds",
        "results": {算法名: {...}}, "ranking": list, "best_algorithm": str}``。
        ``results[alg]`` 含 ``best`` / ``worst`` / ``mean`` / ``std`` / ``median`` /
        ``success``（命中次数）/ ``success_rate`` / ``n_eval``（单次实际求值次数）/
        ``values``（每次运行的原始最优值，可拿去画箱线图）。``ranking`` 按 ``mean`` 升序，
        同分依次比 ``best`` 与名字，所以是**确定性**的。

    算法:
        1. 用 :func:`benchmark_functions` 取目标、标准盒与文献最优值，用 :func:`_parse_bounds`
            再做一次盒校验；
        2. 预算换算成各算法的代数，使四者单次求值次数**一致**：
           ``sa`` 用 ``iters = budget - 1``（求值 ``iters + 1 = budget`` 次）；
           ``pso`` / ``de`` / ``gwo`` 用规模 ``30``、``iters = n_gen = budget // 30 - 1``，
           求值 ``30 * (iters + 1)`` 次——budget 取 30 的整数倍时四者完全相等；
        3. SA 的温度调度由规模自适应：``T0 = mean(hi - lo)``，
           ``alpha = 1e-4 ** (1 / iters)``（即末温恰为首温的 1e-4），避免把 T0 硬编码成
           与函数量级不匹配的常数；起始点在盒内**均匀随机采样**（不是盒中心——sphere /
           rastrigin / ackley / griewank 的全局最优恰好就是盒中心，用中心起步等于白送答案），
           邻域为逐维高斯步长的截断到盒内；
        4. 对每个算法、每个种子独立运行（各算法互不共享随机数），只收集**目标值**，
           再统一算 best/mean/std/median/命中次数，最后排序。

    复杂度:
        时间 O(len(seeds) * len(algorithms) * budget) 次目标求值；
        空间 O(len(seeds) + dim)（不保存迭代历史，只存每次运行的最优值）。

    陷阱:
        1. **``mean`` 与 ``best`` 会给出不同的结论**：元启发式的 ``best`` 常被单个幸运种子主导，
            论文里报 ``best`` 必须同时给 ``mean``/``std`` 与种子；本函数正是为此把 ``values``
            全量返回。
        2. 预算一致**不等于**公平：SA 用 ``budget`` 次**单点**求值，PSO/DE/GWO 用同等次数但
            是**并行**搜索，两者的"代数"完全不同。跨族比较时请显式说明这一点。
        3. 这里所有算法都用**出厂参数**（``F=0.7``、``CR=0.9``、``c1=c2=2`` 等），
            没有做参数调优。这不是"算法本身"的排名，而是"默认参数+这个预算下"的排名。
        4. ``sa`` 的 ``neighbor`` 用固定相对步长 ``step_scale``，在高维（dim >= 30）下
            会迅速退化为爬山，表现会明显偏差；这不是 SA 的固有性质。
        5. 失败/成功判据用的是 ``success_tol`` 绝对容差，对 ``schwefel`` / ``styblinski_tang``
            这类"最优值是四舍五入常数"的函数，成功次数会偏悲观（见
            :func:`benchmark_functions` 的陷阱 1）。

    参考:
        同一组算例下报告 mean/std 与求值次数，是进化计算领域的标准评测协议；
        Derrac et al. 2011, "A practical tutorial on the use of nonparametric statistical tests as
        a methodology for comparing evolutionary and swarm intelligence algorithms"。
    """
    spec = benchmark_functions(name, dim)
    objective = spec["func"]
    optimal = float(spec["optimal_value"])
    lo, hi = _parse_bounds(spec["bounds"])
    span = hi - lo

    if algorithms is None:
        algs: List[str] = list(_BENCH_ALGORITHMS)
    else:
        if isinstance(algorithms, str):
            raise ValueError("algorithms 应为名字序列（如 ['pso', 'de']），而不是单个字符串")
        algs = [str(a).strip().lower() for a in algorithms]
        if not algs:
            raise ValueError("algorithms 不能为空序列")
        unknown = sorted({a for a in algs if a not in _BENCH_ALGORITHMS})
        if unknown:
            raise ValueError(
                f"不支持的算法 {unknown}；可用：{', '.join(_BENCH_ALGORITHMS)}"
                "（GA/ACO/禁忌/VNS 的编码不同，不在本对比台内）"
            )
        if len(set(algs)) != len(algs):
            raise ValueError(f"algorithms 里有重复项：{algs}")

    try:
        budget = int(budget)
    except (TypeError, ValueError):
        raise ValueError(f"budget 必须是整数，得到 {budget!r}")
    if budget < 40:
        raise ValueError(f"budget 必须 >= 40，得到 {budget}")

    step_scale = float(step_scale)
    if not (0.0 < step_scale < 1.0):
        raise ValueError(f"step_scale 必须在 (0, 1) 内，得到 {step_scale}")

    if success_tol is None:
        tol = 1e-4 * max(1.0, abs(optimal))
    else:
        tol = float(success_tol)
        if tol < 0.0:
            raise ValueError(f"success_tol 不能为负，得到 {tol}")

    gen = make_rng(seed)
    if seeds is None:
        seed_list = [int(v) for v in gen.integers(0, 1000000, size=_BENCH_N_SEEDS)]
    else:
        if isinstance(seeds, (int, np.integer)):
            raise ValueError("seeds 应为种子序列（如 [0, 1, 2]），而不是单个整数")
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("seeds 不能为空序列")

    results: Dict[str, dict] = {}
    for alg in algs:
        values: List[float] = []
        n_eval = 0
        for run_seed in seed_list:
            if alg == "sa":
                iters = budget - 1
                T0 = float(np.mean(span))
                alpha = float(1e-4 ** (1.0 / iters))
                x0 = lo + make_rng(run_seed + _BENCH_SA_START_OFFSET).random(lo.shape) * span
                sigma = step_scale * span

                def neighbor(x, g, _lo=lo, _hi=hi, _sigma=sigma):
                    return np.clip(x + g.normal(0.0, _sigma, size=x.shape), _lo, _hi)

                res = simulated_annealing(
                    objective, x0, neighbor, T0=T0, alpha=alpha, iters=iters, seed=run_seed
                )
                values.append(float(res["best_cost"]))
                n_eval = iters + 1
            elif alg == "pso":
                iters = max(1, budget // _BENCH_SWARM - 1)
                res = particle_swarm(
                    objective, spec["bounds"], n_particles=_BENCH_SWARM,
                    iters=iters, seed=run_seed,
                )
                values.append(float(res["best_value"]))
                n_eval = _BENCH_SWARM * (iters + 1)
            elif alg == "de":
                n_gen = max(1, budget // _BENCH_SWARM - 1)
                res = differential_evolution(
                    objective, spec["bounds"], pop_size=_BENCH_SWARM,
                    n_gen=n_gen, seed=run_seed,
                )
                values.append(float(res["fun"]))
                n_eval = int(res["n_eval"])
            else:  # gwo
                n_gen = max(1, budget // _BENCH_SWARM - 1)
                res = grey_wolf_optimizer(
                    objective, spec["bounds"], n_wolves=_BENCH_SWARM,
                    n_gen=n_gen, seed=run_seed,
                )
                values.append(float(res["fun"]))
                n_eval = _BENCH_SWARM * (n_gen + 1)

        arr = np.asarray(values, dtype=float)
        succ = int(np.sum(np.abs(arr - optimal) <= tol))
        results[alg] = {
            "best": float(np.min(arr)),
            "worst": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "success": succ,
            "success_rate": float(succ / float(arr.size)),
            "n_eval": int(n_eval),
            "values": [float(v) for v in arr],
        }

    order = sorted(algs, key=lambda a: (results[a]["mean"], results[a]["best"], a))
    return {
        "name": spec["name"],
        "dim": int(spec["dim"]),
        "optimal_value": float(optimal),
        "success_tol": float(tol),
        "budget": budget,
        "seeds": seed_list,
        "results": results,
        "ranking": order,
        "best_algorithm": order[0],
    }


# --------------------------------------------------------------------------- #
# 人工蜂群
# --------------------------------------------------------------------------- #
def artificial_bee_colony(
    objective: Callable[[np.ndarray], float],
    bounds,
    n_food: int = 20,
    limit: Optional[int] = None,
    iters: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """人工蜂群 ABC（Karaboga 2005：引领蜂 / 跟随蜂 / 侦察蜂三阶段，最小化 ``objective``）。

    参数:
        objective: ``objective(x) -> float``，目标函数，**约定为最小化**（要最大化请自己取负）。
        bounds: 搜索盒，口径与 :func:`particle_swarm` 一致：``(lo, hi)`` 或 ``[(lo_i, hi_i), ...]``，
            每维要求 ``hi > lo``（退化盒与非有限值都会抛 ValueError）。
        n_food: 蜜源（食物源）数量，必须 >= 2；引领蜂与跟随蜂各 n_food 只。
        limit: 侦察蜂阈值：某个蜜源连续 ``limit`` 轮没有被改进就被放弃、由侦察蜂随机重置。
            None 表示取 ``n_food * n_dim``（常见经验值）。
        iters: 迭代（轮）数，>= 0。
        seed: 随机种子，None 用 ``DEFAULT_SEED``。

    返回:
        ``{"best_x": np.ndarray, "best_value": float, "history": List[float],
        "n_eval": int, "n_scouts": int, "limit": int}``。
        ``history`` 长度 = ``iters + 1``，元素是每轮结束时的历史最优值（单调不增）；
        ``n_eval`` 是目标函数**实际求值次数**（写论文报预算用）；
        ``n_scouts`` 是整轮过程中侦察蜂重置蜜源的总次数（多样性代理指标）。

    算法:
        1. 初始化：在 ``[lo, hi]`` 内独立均匀采样 ``n_food`` 个蜜源并求值，
           每个蜜源的 ``trials`` 计 0；
        2. 引领蜂阶段：对每个蜜源 ``i`` 随机选一个 ``k != i``，做
           ``v = x_i + phi * (x_i - x_k)``，``phi ~ U(-1, 1)`` 逐维独立，随后把越界分量
           拉回边界；``f(v) < f(x_i)`` 时贪心替换并把 ``trials[i]`` 归零，否则 ``trials[i] += 1``；
        3. 跟随蜂阶段：按 ``p_i ∝ fit_i`` 轮盘赌选蜜源，对其做与第 2 步完全相同的邻域搜索。
            ``fit`` 用 ``1 / (1 + f)``（``f >= 0``）或 ``1 + |f|``（``f < 0``），
            **不是**目标值本身——直接用 ``f`` 做权重在最小化问题里会选到最差的蜜源；
        4. 侦察蜂阶段：任何 ``trials[i] > limit`` 的蜜源被丢弃，换成盒内均匀随机新点
           （``trials[i]`` 归零），``n_scouts`` 累加；
        5. 每轮结束更新历史最优。

    复杂度:
        时间 O(iters * n_food * (dim + T_objective)) / 空间 O(n_food * dim + iters)。

    陷阱:
        1. ``limit`` 是最关键的参数：太小（如 1~2）会让蜜源在改进一次后很快被随机重置，
            退化成随机搜索；太大（如 ``n_food * iters``）等于关掉侦察蜂，整群会一起早熟。
            本函数的默认值 ``n_food * dim`` 只是文献常用启发式，换问题必须重调。
        2. 跟随蜂用的是 ``p_i ∝ fit_i`` 的**有放回**轮盘赌，所以并不是"每个蜜源恰好被访问一次"。
            这不影响正确性，但意味着每轮的"引领蜂 + 跟随蜂"两阶段合计求值次数恰为
            ``2 * n_food``（与选择结果无关）；一旦发生侦察蜂重置，该轮会再多一次求值，
            因此总求值次数是 ``返回的 n_eval = n_food + iters * 2 * n_food + n_scouts``。
        3. 本实现每轮**恰好**放 ``n_food`` 只跟随蜂（标准 ABC 口径）。有的变体只放少量跟随蜂，
            预算会不同，跨实现比较时先对齐口径。
        4. ``objective`` 返回 NaN 时所有比较都是 False，该蜜源会被一直保留到被侦察蜂重置；
            本实现不做有限性保护（与 :func:`differential_evolution` 一致）。
        5. 与 PSO/DE 一样，ABC 也**不保证全局最优**；用 :func:`benchmark_optimizers` 时它不在
            默认名单里，需要时请自行在等预算下对比。

    参考:
        Karaboga & Basturk 2007, "A powerful and efficient algorithm for numerical function
        optimization: artificial bee colony (ABC) algorithm"；
        Karaboga & Akay 2009（``limit`` 与侦察蜂机制的参数敏感性分析）。
    """
    if not callable(objective):
        raise ValueError("objective 必须是可调用对象")
    lo, hi = _parse_bounds(bounds)
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if lo.size != hi.size or lo.size == 0:
        raise ValueError("bounds 规范后 lo / hi 长度必须一致且非空")
    n_dim = int(lo.size)

    try:
        n_food = int(n_food)
    except (TypeError, ValueError):
        raise ValueError(f"n_food 必须是整数，得到 {n_food!r}")
    if n_food < 2:
        raise ValueError(f"n_food 必须 >= 2（需要一个不同的 k），得到 {n_food}")
    try:
        iters = int(iters)
    except (TypeError, ValueError):
        raise ValueError(f"iters 必须是整数，得到 {iters!r}")
    if iters < 0:
        raise ValueError(f"iters 不能为负，得到 {iters}")
    if limit is None:
        limit = n_food * n_dim
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError(f"limit 必须是整数，得到 {limit!r}")
    if limit < 1:
        raise ValueError(f"limit 必须 >= 1，得到 {limit}")

    gen = make_rng(seed)
    foods = lo + gen.random((n_food, n_dim)) * (hi - lo)  # 广播：逐维独立均匀
    fun_vals = np.empty(n_food, dtype=float)
    for i in range(n_food):
        fun_vals[i] = float(objective(foods[i]))
    trials = np.zeros(n_food, dtype=int)
    n_eval = n_food

    best_idx = int(np.argmin(fun_vals))
    best_x = foods[best_idx].copy()
    best_value = float(fun_vals[best_idx])
    history: List[float] = [best_value]
    n_scouts = 0

    def nectar(rows: np.ndarray) -> np.ndarray:
        """把目标值（越小越好）换成蜜源"花蜜量"（越大越好），且恒为正。"""
        return np.where(rows >= 0.0, 1.0 / (1.0 + rows), 1.0 + np.abs(rows))

    for _ in range(iters):
        # --- 引领蜂 ---
        for i in range(n_food):
            k = int(gen.integers(0, n_food - 1))
            if k >= i:
                k += 1
            phi = gen.uniform(-1.0, 1.0, size=n_dim)
            cand = np.clip(foods[i] + phi * (foods[i] - foods[k]), lo, hi)
            cand_f = float(objective(cand))
            n_eval += 1
            if cand_f < fun_vals[i]:
                foods[i] = cand
                fun_vals[i] = cand_f
                trials[i] = 0
            else:
                trials[i] += 1

        # --- 跟随蜂：按花蜜量轮盘赌选蜜源 ---
        fit = nectar(fun_vals)
        total = float(np.sum(fit))
        if total > 0.0 and np.isfinite(total):
            probs = fit / total
        else:
            probs = np.full(n_food, 1.0 / n_food)
        picks = gen.choice(n_food, size=n_food, p=probs)
        for i in [int(v) for v in picks]:
            k = int(gen.integers(0, n_food - 1))
            if k >= i:
                k += 1
            phi = gen.uniform(-1.0, 1.0, size=n_dim)
            cand = np.clip(foods[i] + phi * (foods[i] - foods[k]), lo, hi)
            cand_f = float(objective(cand))
            n_eval += 1
            if cand_f < fun_vals[i]:
                foods[i] = cand
                fun_vals[i] = cand_f
                trials[i] = 0
            else:
                trials[i] += 1

        # --- 侦察蜂：放弃枯竭蜜源 ---
        for i in range(n_food):
            if trials[i] > limit:
                foods[i] = lo + gen.random(n_dim) * (hi - lo)
                fun_vals[i] = float(objective(foods[i]))
                n_eval += 1
                trials[i] = 0
                n_scouts += 1

        cur_idx = int(np.argmin(fun_vals))
        if fun_vals[cur_idx] < best_value:
            best_value = float(fun_vals[cur_idx])
            best_x = foods[cur_idx].copy()
        history.append(best_value)

    return {
        "best_x": best_x,
        "best_value": float(best_value),
        "history": history,
        "n_eval": int(n_eval),
        "n_scouts": int(n_scouts),
        "limit": int(limit),
    }


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _rastrigin(x) -> float:
    """二维 Rastrigin 函数，全局最优 0 于原点（自测内部辅助函数）。"""
    arr = as_vector(x, "x")
    return float(10.0 * arr.size + np.sum(arr ** 2 - 10.0 * np.cos(2 * np.pi * arr)))


def _ackley(x) -> float:
    """二维 Ackley 函数，全局最优 0 于原点（自测内部辅助函数）。"""
    arr = as_vector(x, "x")
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = float(np.sum(arr ** 2))
    s2 = float(np.sum(np.cos(c * arr)))
    n = arr.size
    return float(-a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.e)


def _rosenbrock(x) -> float:
    """Rosenbrock 香蕉函数，全局最优 0 于全 1 点（自测内部辅助函数）。

    峡谷形（banana valley）地形：梯度指向最优点的路径极窄，
    最速下降法会贴着谷底反复横跳，是检验"是否真的用了种群/差分信息"的标准算例。
    """
    arr = as_vector(x, "x")
    if arr.size < 2:
        raise ValueError(f"Rosenbrock 至少需要 2 维，得到 {arr.size} 维")
    return float(np.sum(100.0 * (arr[1:] - arr[:-1] ** 2) ** 2 + (1.0 - arr[:-1]) ** 2))


def _knapsack_fixture():
    """自测用 10 件 0/1 背包算例；穷举 2^10 得到唯一最优价值 94（重量恰为 60）。"""
    w = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    v = [3, 5, 8, 11, 17, 20, 26, 30, 35, 44]
    cap = 60
    return np.asarray(w, dtype=float), np.asarray(v, dtype=float), float(cap)


def _knapsack_fitness_factory(weights, values, capacity):
    """0/1 背包适应度：超重返回 -inf（硬约束），否则返回总价值。"""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    cap = float(capacity)

    def fitness(genes: np.ndarray) -> float:
        g = np.asarray(genes).astype(float)
        if float(g @ w) > cap:
            return float("-inf")
        return float(g @ v)

    return fitness


def _knapsack_repair_factory(weights, values, capacity):
    """0/1 背包贪心修复：超重时反复去掉"价值密度最低"的已选物品，直到不超重。"""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    cap = float(capacity)
    density = np.divide(v, w, out=np.zeros_like(v), where=w > 0)

    def repair(genes: np.ndarray) -> np.ndarray:
        g = np.asarray(genes).astype(np.int8).copy()
        while float(g @ w) > cap:
            chosen = np.nonzero(g)[0]
            if chosen.size == 0:
                break
            worst = chosen[int(np.argmin(density[chosen]))]
            g[worst] = 0
        return g

    return repair


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    返回:
        dict，键全部为简短 ASCII。**全部使用固定 seed 且只走 numpy Generator**，
        因此两次调用结果逐位相同（不依赖全局随机状态、不依赖运行顺序）。

    算法:
        覆盖 8 个算法各自的已知最优算例：
        - SA：二维 Rastrigin（最优 0）+ 10 件 0/1 背包（穷举最优价值 94）；
        - GA：10 件 0/1 背包（最优 94，组合数仅 1024，可与穷举对齐）；
        - PSO：二维 Rastrigin 与二维 Ackley（最优均为 0）；
        - ACO：10 城圆上均匀分布 TSP（最优 6.180339887498948）；
        - DE：10 维 Rastrigin 与 2 维 Rosenbrock，对照**同一自测里真实计算**的
          "固定种子随机采样 2000 点"最优值（断言 DE 更优，否则视为实现有 bug）；
        - GWO：同上两道题、同一对照基线；
        - 禁忌搜索：9 城圆上"打乱序"初始解的置换邻域，须不劣于初始解，
          且不超过正九边形周长 2*n*sin(pi/n)（闭式最优）；
        - VNS：一维 sum((x-3)^2) 的多尺度扰动邻域，须收敛到 x = 3（容差 1e-3）。
        另覆盖 3 个新增函数与 3 处修复的回归：
        - ``benchmark_functions``：7 个标准函数在各自声称的最优点处的取值必须与声称的
          最优值闭合（1e-9 相对容差）、标准搜索盒、名字规范化、非法入参报错；
        - ``benchmark_optimizers``：2 维 sphere、300 次求值、种子 (0,1,2) 下
          PSO/DE 的 n_eval / mean / success_rate / ranking 口径，以及"同种子可复现"；
        - ``artificial_bee_colony``：二维 Rastrigin 与二维 sphere 的收敛值、
          history 长度 = iters+1、n_eval 口径（2*n_food 每轮 + 侦察蜂）、同种子可复现；
        - 回归：SA 温度下溢到 0.0 不再抛 ZeroDivisionError、bounds 的一维简写
          ``(lo, hi)`` 同样受有限性与 ``hi > lo`` 校验、ACO 单城市算例 history 长度。

    复杂度:
        时间约 O(1)（固定小算例，总求值量在 1e5 量级）/ 空间 O(1)。

    陷阱:
        自测里给的 ``best_cost`` 是"这次固定种子跑出来的值"，**不是算法保证值**。
        如果你调了默认参数，这些数字会变——这是预期行为，但请不要为了让自测通过而
        去改断言阈值，应该先在多组种子上确认新参数确实更好。

    参考:
        契约第 3 节"每个模块结尾提供 _self_test()"。
    """
    result: Dict[str, object] = {}

    # ---------- SA：二维 Rastrigin ----------
    def sa_neighbor_rastrigin(x: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        return np.clip(x + gen.normal(0.0, 0.5, size=x.shape), -5.12, 5.12)

    sa_r = simulated_annealing(
        _rastrigin, np.array([3.7, -2.4]), sa_neighbor_rastrigin,
        T0=5.0, alpha=0.999, iters=20000, seed=7,
    )
    result["sa_rastrigin_cost"] = round(float(sa_r["best_cost"]), 8)
    result["sa_rastrigin_len"] = len(sa_r["history"])

    # ---------- SA：10 件背包（穷举最优 94）----------
    w10, v10, cap10 = _knapsack_fixture()
    knap = _knapsack_fitness_factory(w10, v10, cap10)

    def sa_neighbor_bits(x: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        y = x.copy()
        k = int(gen.integers(1, 4))  # 一次翻转 1~3 位，比单位翻转更快跳出局部最优
        idx = gen.choice(y.size, size=k, replace=False)
        y[idx] = 1.0 - y[idx]
        return y

    sa_k = simulated_annealing(
        lambda g: -knap(g), np.array([0.0] * 10), sa_neighbor_bits,
        T0=8.0, alpha=0.9995, iters=12000, seed=11,
    )
    result["sa_knapsack_value"] = round(float(-sa_k["best_cost"]), 6)
    result["sa_knapsack_bits"] = [int(b) for b in (sa_k["best_x"] > 0.5).astype(int)]

    # ---------- GA：10 件背包 ----------
    # 参数说明：pop_size=60 / generations=150 是实测在这道题上 5 个种子都稳定命中 94 的最小配置；
    # 用 pop_size=40 / generations=60 时 5 个种子都停在 93（差 1），说明预算不足会稳定次优。
    ga = genetic_algorithm(
        knap, n_genes=10, pop_size=60, generations=150,
        crossover_rate=0.9, mutation_rate=0.08, seed=2024, maximize=True,
        repair=_knapsack_repair_factory(w10, v10, cap10),
    )
    result["ga_knapsack_value"] = round(float(ga["best_fitness"]), 6)
    result["ga_knapsack_bits"] = [int(b) for b in ga["best_genes"]]
    result["ga_knapsack_len"] = len(ga["history"])

    # ---------- PSO：Rastrigin / Ackley ----------
    pso_r = particle_swarm(
        _rastrigin, [(-5.12, 5.12), (-5.12, 5.12)],
        n_particles=40, iters=300, seed=5, maximize=False,
    )
    result["pso_rastrigin_value"] = round(float(pso_r["best_value"]), 8)
    result["pso_rastrigin_x"] = [round(float(t), 6) for t in pso_r["best_x"]]

    pso_a = particle_swarm(
        _ackley, [(-5.0, 5.0), (-5.0, 5.0)],
        n_particles=40, iters=300, seed=5, maximize=False,
    )
    result["pso_ackley_value"] = round(float(pso_a["best_value"]), 8)

    # maximize=True 与边界解的对照：f(x) = -(x-3)^2，在 [0,10] 上最大值 0 于 x=3（内点）；
    # g(x) = x，在 [0,10] 上最大值 10 于 x=10（边界，专门检查 clip 处理没把解压错）。
    pso_max = particle_swarm(
        lambda x: -float((x[0] - 3.0) ** 2), [(0.0, 10.0)],
        n_particles=20, iters=100, seed=9, maximize=True,
    )
    result["pso_max_value"] = round(float(pso_max["best_value"]), 6)
    result["pso_max_x"] = round(float(pso_max["best_x"][0]), 6)

    pso_edge = particle_swarm(
        lambda x: float(x[0]), [(0.0, 10.0)],
        n_particles=20, iters=100, seed=9, maximize=True,
    )
    result["pso_edge_value"] = round(float(pso_edge["best_value"]), 6)

    # ---------- ACO：10 城 TSP（最优 6.180339887498948）----------
    n_city = 10
    ang = 2 * np.pi * np.arange(n_city) / n_city
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    aco = ant_colony_tsp(D, n_ants=10, iters=30, alpha=1.0, beta=2.0, rho=0.5, seed=3)
    result["aco_tsp_length"] = round(float(aco["length"]), 9)
    result["aco_tsp_optimal"] = round(float(2 * n_city * np.sin(np.pi / n_city)), 9)
    result["aco_tsp_len"] = len(aco["history"])

    # ---------- 对照基线：固定种子随机采样 2000 点 ----------
    def _random_sample_best(fn, box, n_samples: int, sample_seed: int) -> float:
        """在搜索盒内固定种子均匀采样 n_samples 个点，返回其中最优目标值。

        这是 DE/GWO 的**独立对照**：元启发式若连"同样预算内纯随机撒点"都赢不了，
        就说明实现有 bug（而不是"算法不够好"）。
        """
        g = make_rng(sample_seed)
        box_arr = np.asarray(box, dtype=float).reshape(-1, 2)
        lo_b, hi_b = box_arr[:, 0], box_arr[:, 1]
        pts = lo_b + g.random((n_samples, lo_b.size)) * (hi_b - lo_b)
        return min(float(fn(p)) for p in pts)

    box_r10 = [(-5.12, 5.12)] * 10
    box_ros = [(-2.0, 2.0), (-2.0, 2.0)]
    rand_r10 = _random_sample_best(_rastrigin, box_r10, 2000, 99)
    rand_ros = _random_sample_best(_rosenbrock, box_ros, 2000, 99)

    # ---------- DE：10 维 Rastrigin / 2 维 Rosenbrock ----------
    de_r = differential_evolution(
        _rastrigin, box_r10, pop_size=40, n_gen=200, F=0.7, CR=0.9, seed=13,
    )
    if de_r["n_eval"] != 40 * 201:
        raise AssertionError(f"DE 的求值次数 {de_r['n_eval']} != pop_size*(n_gen+1) = {40 * 201}")
    if not de_r["fun"] < rand_r10:
        raise AssertionError(
            f"DE 在 10 维 Rastrigin 上得到 {de_r['fun']}，未优于随机采样 2000 点的 {rand_r10}"
        )

    de_ros = differential_evolution(
        _rosenbrock, box_ros, pop_size=40, n_gen=200, F=0.7, CR=0.9, seed=13,
    )
    if not de_ros["fun"] < rand_ros:
        raise AssertionError(
            f"DE 在 Rosenbrock 上得到 {de_ros['fun']}，未优于随机采样 2000 点的 {rand_ros}"
        )

    result["de_rastrigin_fun"] = round(float(de_r["fun"]), 8)
    result["de_rastrigin_random_fun"] = round(float(rand_r10), 8)
    result["de_rosenbrock_fun"] = round(float(de_ros["fun"]), 8)
    result["de_rosenbrock_random_fun"] = round(float(rand_ros), 8)
    result["de_n_eval"] = int(de_r["n_eval"])
    result["de_len"] = len(de_r["history"])

    # ---------- GWO：同样的两道题，同样的对照 ----------
    gwo_r = grey_wolf_optimizer(_rastrigin, box_r10, n_wolves=30, n_gen=200, seed=17)
    if not gwo_r["fun"] < rand_r10:
        raise AssertionError(
            f"GWO 在 10 维 Rastrigin 上得到 {gwo_r['fun']}，未优于随机采样 2000 点的 {rand_r10}"
        )
    gwo_ros = grey_wolf_optimizer(_rosenbrock, box_ros, n_wolves=30, n_gen=200, seed=17)
    if not gwo_ros["fun"] < rand_ros:
        raise AssertionError(
            f"GWO 在 Rosenbrock 上得到 {gwo_ros['fun']}，未优于随机采样 2000 点的 {rand_ros}"
        )
    result["gwo_rastrigin_fun"] = round(float(gwo_r["fun"]), 8)
    result["gwo_rosenbrock_fun"] = round(float(gwo_ros["fun"]), 8)
    result["gwo_len"] = len(gwo_r["history"])

    # ---------- 禁忌搜索：9 城 TSP 式置换邻域（不劣于初始解）----------
    n_city9 = 9
    ang9 = 2 * np.pi * np.arange(n_city9) / n_city9
    pts9 = np.stack([np.cos(ang9), np.sin(ang9)], axis=1)
    D9 = np.sqrt(((pts9[:, None, :] - pts9[None, :, :]) ** 2).sum(-1))

    def tour_len(t) -> float:
        order = np.asarray(t, dtype=float).astype(int)
        return float(sum(D9[order[k], order[(k + 1) % n_city9]] for k in range(n_city9)))

    def swap_neighbors(t):
        base = np.asarray(t, dtype=float).copy()
        out = []
        for a in range(n_city9):
            for b in range(a + 1, n_city9):
                y = base.copy()
                y[a], y[b] = base[b], base[a]
                out.append(y)
        return out

    init_tour = np.array([0, 3, 6, 1, 4, 7, 2, 5, 8], dtype=float)
    tabu_res = tabu_search(
        init_tour, swap_neighbors, tour_len, max_iter=60, tabu_tenure=7, seed=23,
    )
    init_len = tour_len(init_tour)
    if tabu_res["fun"] > init_len + 1e-9:
        raise AssertionError(
            f"禁忌搜索得到 {tabu_res['fun']}，劣于初始解 {init_len}"
        )
    # 独立结论：9 个圆上均布城市的最优回路长度 = 2*n*sin(pi/n)（正多边形周长）
    if tabu_res["fun"] > round(float(2 * n_city9 * np.sin(np.pi / n_city9)), 6) + 1e-9:
        raise AssertionError(
            f"禁忌搜索得到 {tabu_res['fun']}，超过正九边形最优 {2 * n_city9 * np.sin(np.pi / n_city9)}"
        )
    result["tabu_tsp_length"] = round(float(tabu_res["fun"]), 6)
    result["tabu_tsp_init"] = round(float(init_len), 6)
    result["tabu_tsp_n_better"] = int(tabu_res["n_better"])
    result["tabu_tsp_n_iter"] = int(tabu_res["n_iter"])

    # ---------- VNS：一维二次函数 sum((x-3)^2) 必须收敛到 x = 3 ----------
    def quad_shift(x) -> float:
        arr = as_vector(x, "x")
        return float(np.sum((arr - 3.0) ** 2))

    def _make_perturb(scale: float):
        def perturb(x: np.ndarray, gen: np.random.Generator) -> np.ndarray:
            return np.asarray(x, dtype=float) + gen.normal(0.0, scale, size=np.asarray(x).shape)
        return perturb

    vns_res = variable_neighborhood_search(
        np.array([0.0]),
        [_make_perturb(s) for s in (1.0, 0.3, 0.1, 0.03, 0.01, 0.001)],
        quad_shift,
        max_iter=200,
        seed=31,
    )
    vns_x = float(as_vector(vns_res["x"], "x")[0])
    if abs(vns_x - 3.0) > 1e-3:
        raise AssertionError(f"VNS 在 sum((x-3)^2) 上收敛到 {vns_x}，与最优 3 的偏差超过 1e-3")
    if abs(vns_res["fun"] - quad_shift(np.array([vns_x]))) > 1e-12:
        raise AssertionError("VNS 返回的 fun 与 x 处的目标值不一致")
    result["vns_quad_x"] = round(vns_x, 6)
    result["vns_quad_fun"] = round(float(vns_res["fun"]), 10)
    result["vns_quad_n_iter"] = int(vns_res["n_iter"])

    # ---------- 新增：benchmark_functions 的"公式 ↔ 已知最优"自洽性 ----------
    # 这里用的是**独立**信息：最优分量取自解析解（牛顿法/文献），最优值取自文献常数，
    # 断言二者与实现里的闭式表达式互相闭合。任何一处抄错（比如 sphere 写成 |x|、
    # rosenbrock 的最优点写成原点）都会在这里炸出来。
    # 容差口径：绝对量级 ``1e-9 * dim``。**不要**用 ``1e-9 * max(1, |optimal_value|)``
    # ——schwefel 的最优值约 -418.98，那个写法会把它放宽到 4.2e-7，而实测偏差只有
    # 4.5e-13，等于该键几乎不设防。偏差随维数近似线性（参考最优点只给 7 位有效数字，
    # 每维贡献约 9.1e-14），所以按 dim 线性缩放。
    bm_opt_dev = 0.0
    for bm_name in _BENCHMARK_NAMES:
        bm_spec = benchmark_functions(bm_name, 3)
        bm_here = bm_spec["func"](bm_spec["optimal_x"])
        bm_dev = abs(bm_here - bm_spec["optimal_value"])
        if bm_dev > 1e-9 * 3:
            raise AssertionError(
                f"测试函数 {bm_name} 在它声称的最优点 {bm_spec['optimal_x']} 处取到 "
                f"{bm_here}，与声称的最优值 {bm_spec['optimal_value']} 不符"
            )
        bm_opt_dev = max(bm_opt_dev, bm_dev)
    result["bm_opt_dev"] = round(float(bm_opt_dev), 15)

    # 维度扫描：上面只点测了 dim=3，而偏差随维数增长，必须把各维都覆盖到
    # （实测最坏组合是 schwefel dim=5，偏差 4.547e-13；只测 dim=3 会得到
    # 2.274e-13 并让人误以为这是上界）。
    bm_dev_dims = 0.0
    bm_dev_worst_dim = 0
    bm_dim_combos = 0
    for bm_dim in range(2, 7):
        for bm_name in _BENCHMARK_NAMES:
            bm_spec = benchmark_functions(bm_name, bm_dim)
            bm_here = bm_spec["func"](bm_spec["optimal_x"])
            bm_dev = abs(bm_here - bm_spec["optimal_value"])
            if bm_dev > 1e-9 * bm_dim:
                raise AssertionError(
                    f"测试函数 {bm_name}（dim={bm_dim}）在它声称的最优点 "
                    f"{bm_spec['optimal_x']} 处取到 {bm_here}，与声称的最优值 "
                    f"{bm_spec['optimal_value']} 不符"
                )
            if bm_dev > bm_dev_dims:
                bm_dev_dims = bm_dev
                bm_dev_worst_dim = bm_dim
            bm_dim_combos += 1
    result["bm_opt_dev_dims"] = round(float(bm_dev_dims), 15)
    result["bm_opt_dev_worst_dim"] = int(bm_dev_worst_dim)
    result["bm_dim_combos"] = int(bm_dim_combos)

    # 独立闭式值：手算可核对的几组（与模块内实现无关）
    bm_sph3 = benchmark_functions("sphere", 3)
    if abs(bm_sph3["func"]([1.0, 2.0, 3.0]) - 14.0) > 1e-12:
        raise AssertionError("sphere(1,2,3) 应为 14")
    bm_ras2 = benchmark_functions("rastrigin", 2)
    if abs(bm_ras2["func"]([1.0, 1.0]) - 2.0) > 1e-12:
        raise AssertionError("rastrigin(1,1) 应为 2")
    bm_gri5 = benchmark_functions("griewank", 5)
    if abs(bm_gri5["func"](np.zeros(5))) > 1e-15:
        raise AssertionError("griewank 在原点应为 0")
    bm_ros4 = benchmark_functions("rosenbrock", 4)
    if abs(bm_ros4["func"](np.ones(4))) > 1e-30:
        raise AssertionError("rosenbrock 在全 1 点应为 0")

    # 标准文献搜索盒（协议的一部分，缩小/放大会让结果与文献不可比）
    for bm_name, bm_box in (
        ("sphere", (-5.12, 5.12)),
        ("rastrigin", (-5.12, 5.12)),
        ("ackley", (-32.768, 32.768)),
        ("rosenbrock", (-2.048, 2.048)),
        ("griewank", (-600.0, 600.0)),
        ("schwefel", (-500.0, 500.0)),
        ("styblinski_tang", (-5.0, 5.0)),
    ):
        bm_spec = benchmark_functions(bm_name, 4)
        if tuple(bm_spec["bounds"][0]) != bm_box:
            raise AssertionError(f"{bm_name} 的标准搜索盒应为 {bm_box}")
        if len(bm_spec["bounds"]) != 4:
            raise AssertionError(f"{bm_name} 应返回 4 组逐维区间")
    if benchmark_functions("rosenbrock", 4)["optimal_x"] != [1.0] * 4:
        raise AssertionError("rosenbrock 的最优点应为全 1")
    if benchmark_functions("sphere", 2)["optimal_x"] != [0.0, 0.0]:
        raise AssertionError("sphere 的最优点应为原点")
    if benchmark_functions("Sphere", 2)["name"] != "sphere":
        raise AssertionError("benchmark_functions 的名字规范化应当是大小写不敏感的")
    result["bm_dim"] = int(bm_sph3["dim"])
    result["bm_optimal_value_rosenbrock"] = float(bm_ros4["optimal_value"])

    def expect_raises(label: str, fn) -> None:
        """断言 fn() 抛 ValueError（模块内禁止用 assert 关键字，统一走这里）。"""
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"{label} 本应抛 ValueError，实际却成功返回了")

    expect_raises("benchmark_functions(name='foo')", lambda: benchmark_functions("foo"))
    expect_raises("benchmark_functions(dim=0)", lambda: benchmark_functions("sphere", 0))
    expect_raises(
        "benchmark_functions('rosenbrock', dim=1)",
        lambda: benchmark_functions("rosenbrock", 1),
    )

    # ---------- 新增：benchmark_optimizers 在多组种子上给出确定性统计 ----------
    bo = benchmark_optimizers(
        "sphere", 2, algorithms=("pso", "de"), seeds=(0, 1, 2), budget=300
    )
    if sorted(bo["ranking"]) != ["de", "pso"]:
        raise AssertionError(f"对比台的 ranking 应包含 pso 与 de，得到 {bo['ranking']}")
    if bo["best_algorithm"] != bo["ranking"][0]:
        raise AssertionError("best_algorithm 与 ranking 首项不一致")
    for bo_alg in ("pso", "de"):
        bo_r = bo["results"][bo_alg]
        if len(bo_r["values"]) != 3:
            raise AssertionError(f"{bo_alg} 应记录 3 次运行的原始值")
        if bo_r["n_eval"] != 300:
            raise AssertionError(f"{bo_alg} 单次求值次数应为 300，得到 {bo_r['n_eval']}")
        if not (bo_r["worst"] >= bo_r["median"] >= bo_r["best"]):
            raise AssertionError(f"{bo_alg} 的 worst/median/best 顺序不对")
        if abs(bo_r["mean"] - float(np.mean(bo_r["values"]))) > 1e-12:
            raise AssertionError(f"{bo_alg} 的 mean 与 values 不相符")
        if abs(bo_r["success_rate"] - bo_r["success"] / 3.0) > 1e-12:
            raise AssertionError(f"{bo_alg} 的 success_rate 与 success 不相符")
        # 300 次求值下，两个求解器在 2 维 sphere 上都该进入 5e-2 以内（实测约 3e-3 / 1e-2）
        if not (np.isfinite(bo_r["mean"]) and bo_r["mean"] < 0.05):
            raise AssertionError(f"{bo_alg} 在 2 维 sphere（300 次求值）上均值 {bo_r['mean']} 太差")
    bo2 = benchmark_optimizers(
        "sphere", 2, algorithms=("pso", "de"), seeds=(0, 1, 2), budget=300
    )
    if bo2 != bo:
        raise AssertionError("benchmark_optimizers 在相同 seeds 下结果不可复现")
    if bo["ranking"] != sorted(bo["results"], key=lambda k: bo["results"][k]["mean"]):
        raise AssertionError(f"ranking 没有按 mean 升序排列：{bo['ranking']}")
    result["bench_pso_mean"] = round(float(bo["results"]["pso"]["mean"]), 10)
    result["bench_de_mean"] = round(float(bo["results"]["de"]["mean"]), 10)
    result["bench_rank0"] = bo["ranking"][0]
    result["bench_budget"] = int(bo["budget"])
    expect_raises(
        "benchmark_optimizers(algorithms=('xyz',))",
        lambda: benchmark_optimizers("sphere", 2, algorithms=("xyz",)),
    )
    expect_raises(
        "benchmark_optimizers(algorithms=[])",
        lambda: benchmark_optimizers("sphere", 2, algorithms=[]),
    )
    expect_raises(
        "benchmark_optimizers(budget=10)",
        lambda: benchmark_optimizers("sphere", 2, budget=10),
    )
    expect_raises(
        "benchmark_optimizers(seeds=[])",
        lambda: benchmark_optimizers("sphere", 2, seeds=[]),
    )

    # ---------- 新增：人工蜂群 ABC ----------
    rast2 = benchmark_functions("rastrigin", 2)
    abc_res = artificial_bee_colony(rast2["func"], rast2["bounds"], n_food=15, iters=200, seed=5)
    if len(abc_res["history"]) != 201:
        raise AssertionError(f"ABC 的 history 长度应为 iters+1=201，得到 {len(abc_res['history'])}")
    if abs(abc_res["best_value"] - rast2["func"](abc_res["best_x"])) > 1e-12:
        raise AssertionError("ABC 返回的 best_value 与 best_x 处的目标值不一致")
    # 每轮恰好 2*n_food 次求值（引领蜂 + 跟随蜂各 n_food），外加初始化与侦察蜂重置
    abc_expect = 15 + 200 * (2 * 15) + abc_res["n_scouts"]
    if abc_res["n_eval"] != abc_expect:
        raise AssertionError(
            f"ABC 的 n_eval={abc_res['n_eval']}，按 15+200*30+{abc_res['n_scouts']} 应为 {abc_expect}"
        )
    if abc_res["limit"] != 30:
        raise AssertionError(f"ABC 的 limit 缺省应为 n_food*n_dim=30，得到 {abc_res['limit']}")
    if not (abc_res["best_value"] < 1e-8):
        raise AssertionError(f"ABC 在二维 Rastrigin 上只找到 {abc_res['best_value']}")
    abc_again = artificial_bee_colony(rast2["func"], rast2["bounds"], n_food=15, iters=200, seed=5)
    if not (
        abc_again["best_value"] == abc_res["best_value"]
        and abc_again["n_eval"] == abc_res["n_eval"]
        and abc_again["n_scouts"] == abc_res["n_scouts"]
        and abc_again["history"] == abc_res["history"]
        and bool(np.array_equal(abc_again["best_x"], abc_res["best_x"]))
    ):
        raise AssertionError("ABC 在相同种子下结果不可复现")
    result["abc_rastrigin_value"] = round(float(abc_res["best_value"]), 12)
    result["abc_rastrigin_n_eval"] = int(abc_res["n_eval"])
    result["abc_rastrigin_n_scouts"] = int(abc_res["n_scouts"])
    result["abc_history_len"] = len(abc_res["history"])

    sph2 = benchmark_functions("sphere", 2)
    abc_sph = artificial_bee_colony(sph2["func"], sph2["bounds"], n_food=20, iters=100, seed=1)
    if not (abc_sph["best_value"] < 1e-12):
        raise AssertionError(f"ABC 在二维 sphere 上只找到 {abc_sph['best_value']}")
    if abc_sph["n_eval"] != 20 + 100 * 40 + abc_sph["n_scouts"]:
        raise AssertionError("ABC 在 sphere 上的 n_eval 与预算口径不一致")
    result["abc_sphere_value"] = float(abc_sph["best_value"])

    expect_raises(
        "artificial_bee_colony(n_food=1)",
        lambda: artificial_bee_colony(sph2["func"], sph2["bounds"], n_food=1),
    )
    expect_raises(
        "artificial_bee_colony(limit=0)",
        lambda: artificial_bee_colony(sph2["func"], sph2["bounds"], limit=0),
    )
    expect_raises(
        "artificial_bee_colony(iters=-1)",
        lambda: artificial_bee_colony(sph2["func"], sph2["bounds"], iters=-1),
    )
    expect_raises(
        "artificial_bee_colony(objective=1)",
        lambda: artificial_bee_colony(1, sph2["bounds"]),
    )

    # ---------- 新增：本次修复的回归断言 ----------
    # (1) SA 在 T 下溢到 0.0 后不得抛 ZeroDivisionError，且此时应退化为"只接受非劣解"：
    #     neighbor 恒返回更差的点，所以最优值必须停在初始值 100.0，history 长度仍为 iters+1。
    def _sa_sq(x) -> float:
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    def _sa_worse(x, gen: np.random.Generator) -> np.ndarray:
        return np.asarray(x, dtype=float) + 1.0

    sa_zero = simulated_annealing(
        _sa_sq, np.array([10.0]), _sa_worse, T0=1.0, alpha=0.5, iters=1500, seed=3
    )
    if len(sa_zero["history"]) != 1501:
        raise AssertionError("SA 零温回归：history 长度应为 iters+1")
    if abs(sa_zero["best_cost"] - 100.0) > 1e-12:
        raise AssertionError(
            f"SA 零温回归：所有候选都更差，最优值应保持 100.0，得到 {sa_zero['best_cost']}"
        )
    result["sa_zeroT_cost"] = round(float(sa_zero["best_cost"]), 10)
    result["sa_zeroT_len"] = len(sa_zero["history"])

    # (2) bounds 的一维简写 (lo, hi) 必须与逐维写法受同样的校验
    expect_raises(
        "particle_swarm(bounds=(5.0, 3.0))",
        lambda: particle_swarm(_sa_sq, (5.0, 3.0), n_particles=5, iters=2),
    )
    expect_raises(
        "particle_swarm(bounds=(nan, 5.0))",
        lambda: particle_swarm(_sa_sq, (float("nan"), 5.0), n_particles=5, iters=2),
    )
    expect_raises(
        "differential_evolution(bounds=(5.0, 3.0))",
        lambda: differential_evolution(_sa_sq, (5.0, 3.0), pop_size=6, n_gen=2),
    )
    expect_raises(
        "grey_wolf_optimizer(bounds=(-inf, 5.0))",
        lambda: grey_wolf_optimizer(_sa_sq, (float("-inf"), 5.0), n_wolves=6, n_gen=2),
    )

    # (3) ACO 单城市退化算例的 history 长度也必须等于 iters + 1
    aco_one = ant_colony_tsp(np.zeros((1, 1)), iters=5, seed=0)
    if len(aco_one["history"]) != 6:
        raise AssertionError(
            f"ACO 单城市算例的 history 长度应为 6，得到 {len(aco_one['history'])}"
        )
    result["aco_single_hist_len"] = len(aco_one["history"])

    # ---------- 通用一致性：所有 history 必须是单调不增的"历史最优"序列 ----------
    for name, hist in (
        ("de", de_r["history"]),
        ("gwo", gwo_r["history"]),
        ("tabu", tabu_res["history"]),
        ("vns", vns_res["history"]),
        ("abc", abc_res["history"]),
    ):
        if any(hist[i + 1] > hist[i] + 1e-12 for i in range(len(hist) - 1)):
            raise AssertionError(f"{name} 的 history 不是单调不增的历史最优序列")
    return result
