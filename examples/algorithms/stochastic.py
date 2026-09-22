"""随机模型：蒙特卡洛估计与置信区间、M/M/1 排队论、马尔可夫链稳态与吸收、赌徒破产。

本模块共 16 个公开名称（15 个函数 + 1 个常量），15 个函数按用途分成五组，另有常量 ``Z95``：
- 蒙特卡洛：``mc_pi`` / ``mc_integrate``；
- 排队论：``mm1_metrics`` / ``mm1_simulate`` / ``mmc_metrics`` / ``mmc_simulate`` / ``mg1_metrics``；
- 马尔可夫链：``markov_steady_state`` / ``markov_absorption`` / ``gamblers_ruin``；
- 通用仿真：``discrete_event_simulation``；
- MCMC 与 Copula：``metropolis_hastings`` / ``gibbs_sampler_bivariate_normal`` / ``gaussian_copula``
  / ``t_copula``。

全部实现只依赖 numpy 与标准库，随机性统一走 ``_common.rng``（显式种子，绝不用
``np.random`` 的全局状态），因此同一份代码两次运行结果完全一致——竞赛论文里
"结果可复现"是硬要求，随机种子必须写进正文。

核心提醒：蒙特卡洛给出的永远是**估计 + 误差**，只报一个点估计是评审常见的扣分点。
本模块每个随机估计都同时返回标准误与 95% 置信区间。
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_matrix, as_vector, check_square, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

#: 标准正态 97.5% 分位数（双侧 95% 置信区间用），比粗略的 1.96 略精确。
Z95 = 1.959963984540054

__all__ = [
    "mc_pi",
    "mc_integrate",
    "mm1_metrics",
    "mm1_simulate",
    "markov_steady_state",
    "markov_absorption",
    "gamblers_ruin",
    "mmc_metrics",
    "mmck_metrics",
    "mg1_metrics",
    "discrete_event_simulation",
    "mmc_simulate",
    "metropolis_hastings",
    "gibbs_sampler_bivariate_normal",
    "gaussian_copula",
    "t_copula",
    "geometric_brownian_motion",
    "Z95",
]


def mc_pi(n: int, seed: Optional[int] = None) -> Dict[str, float]:
    """用"单位正方形内均匀撒点、统计落在四分之一圆内"的比例估计圆周率 π。

    参数:
        n: 样本点数（必须 >= 2，否则标准误无定义）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``estimate`` π 的点估计（= 4 * 命中比例）；
        ``stderr`` 估计量的标准误（= 4 * sqrt(p(1-p)/n)）；
        ``ci95_low`` / ``ci95_high`` 正态近似 95% 置信区间上下限。

    算法:
        在 [0,1)^2 上抽 n 个均匀点，令 X_i = 1{x^2+y^2 <= 1}，则 E[X]=π/4。
        π̂ = 4 * mean(X)，Var(π̂) = 16 p(1-p)/n。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. 这是**收敛很慢**的演示级估计：误差约 O(1/sqrt(n))，n=1e6 时标准误仍有
           约 0.0016，即只能保证小数点后 2~3 位。要更多位数请用级数展开或
           Gauss-Legendre 求积，别靠加大样本量硬堆。
        2. 置信区间是**正态近似**（用样本比例 p̂ 代替真 p），n 很小时覆盖率会偏低。
        3. 结果依赖种子；论文里必须同时写出 seed 与 n，否则别人无法复现同一个数。

    参考:
        蒙特卡洛方法的标准入门例子（如 Rubinstein & Kroese, "Simulation and the
        Monte Carlo Method"）。
    """
    if n < 2:
        raise ValueError(f"n 必须 >= 2，得到 {n}")
    gen = rng(seed)
    x = gen.random(n)
    y = gen.random(n)
    hits = float(np.count_nonzero(x * x + y * y <= 1.0))
    p_hat = hits / n
    estimate = 4.0 * p_hat
    stderr = 4.0 * math.sqrt(p_hat * (1.0 - p_hat) / n)
    return {
        "estimate": float(estimate),
        "stderr": float(stderr),
        "ci95_low": float(estimate - Z95 * stderr),
        "ci95_high": float(estimate + Z95 * stderr),
    }


def mc_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """用"均匀抽样求均值"估计定积分 ∫_a^b f(x) dx，并给出标准误与 95% 置信区间。

    参数:
        f: 被积函数，必须支持**数组输入**（内部是向量化调用，不是逐点循环）。
        a, b: 积分上下限，要求 b > a。
        n: 样本点数（>= 2）。
        seed: 随机种子。

    返回:
        dict，键为 ``estimate``、``stderr``、``ci95_low``、``ci95_high``。

    算法:
        U_i ~ Uniform(a, b)，I = (b-a) * mean(f(U))；
        Var(I) = (b-a)^2 * Var(f(U)) / n，用样本方差（ddof=1）估计。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. **朴素的均匀抽样在高维或尖峰函数上效率极低**：被积函数集中在很窄的区域时，
           绝大多数样本贡献接近 0，方差巨大。此时应改用重要性抽样。
        2. 标准误只衡量随机误差，不含被积函数在 a、b 处不连续/不可积带来的系统误差
           （例如端点发散的反常积分需要先做变量替换）。
        3. f 必须能接受数组；只接受标量的函数会在这里报错，请用 np.vectorize 包装
           （注意 np.vectorize 只是方便，并不加速）。

    参考:
        蒙特卡洛积分定义；重要性抽样见 Kahn & Harris 1951。
    """
    if not b > a:
        raise ValueError(f"要求 b > a，得到 a={a}, b={b}")
    if n < 2:
        raise ValueError(f"n 必须 >= 2，得到 {n}")
    gen = rng(seed)
    u = a + (b - a) * gen.random(n)
    fx = np.asarray(f(u), dtype=float)
    if fx.shape != u.shape:
        raise ValueError(f"f 必须返回与输入同形状的数组，得到 {fx.shape} vs {u.shape}")
    if not np.all(np.isfinite(fx)):
        raise ValueError("f 在样本点上返回了 NaN/inf，无法估计积分")
    estimate = float((b - a) * fx.mean())
    stderr = float((b - a) * fx.std(ddof=1) / math.sqrt(n))
    return {
        "estimate": estimate,
        "stderr": stderr,
        "ci95_low": estimate - Z95 * stderr,
        "ci95_high": estimate + Z95 * stderr,
    }


def mm1_metrics(lam: float, mu: float) -> Dict[str, float]:
    """M/M/1 排队系统（到达率 λ、服务率 μ）的稳态解析指标。

    参数:
        lam: 到达率（每单位时间到达的顾客数），必须 >= 0。
        mu: 服务率（每单位时间服务的顾客数），必须 > 0。
        要求 ``lam < mu``（否则队列无稳态，直接抛 ValueError）。

    返回:
        dict，键为：
        ``rho`` 利用率 λ/μ；
        ``P0`` 系统空闲概率 1-ρ；
        ``L`` 系统内平均顾客数 ρ/(1-ρ)；
        ``Lq`` 队列中平均等待顾客数 ρ²/(1-ρ)；
        ``W`` 平均逗留时间 1/(μ-λ)；
        ``Wq`` 平均排队等待时间 λ/(μ(μ-λ))。

    算法:
        出生-死亡过程求稳态：π_n = (1-ρ)ρ^n，再代入 Little 公式 L=λW、Lq=λWq。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. **ρ→1 时所有指标爆炸**：ρ=0.95 时 Lq=18，ρ=0.99 时 Lq=98，
           服务能力只差 4% 排队长度差 5 倍——这是排队论最重要的结论，也是论文里
           值得单独讨论的敏感性点。
        2. 这些公式只对**指数分布**的到达间隔和服务时间成立；服务时间方差更大时
           （一般分布）要用 M/G/1 的 Pollaczek-Khinchine 公式，M/M/1 会低估等待。
        3. 单位必须一致：λ 用"人/小时"时 μ 也必须用"人/小时"，混用会让 ρ 差 60 倍。

    参考:
        Erlang 1909；Kendall 1953 记号；Little 1961。
    """
    if mu <= 0:
        raise ValueError(f"mu 必须 > 0，得到 {mu}")
    if lam < 0:
        raise ValueError(f"lam 必须 >= 0，得到 {lam}")
    if lam >= mu:
        raise ValueError(f"M/M/1 要求 lam < mu（否则无稳态），得到 lam={lam}, mu={mu}")
    rho = lam / mu
    return {
        "rho": float(rho),
        "P0": float(1.0 - rho),
        "L": float(rho / (1.0 - rho)),
        "Lq": float(rho * rho / (1.0 - rho)),
        "W": float(1.0 / (mu - lam)),
        "Wq": float(lam / (mu * (mu - lam))),
    }


def mm1_simulate(lam: float, mu: float, n_customers: int, seed: Optional[int] = None) -> Dict[str, float]:
    """M/M/1 的离散事件仿真：生成 n_customers 个顾客，统计等待与队长，和解析解对照。

    参数:
        lam: 到达率（指数分布到达间隔，均值 1/λ）。
        mu: 服务率（指数分布服务时间，均值 1/μ），要求 lam < mu。
        n_customers: 仿真顾客数（>= 1）。
        seed: 随机种子。

    返回:
        dict，键为：
        ``avg_wait`` 平均**排队等待**时间（不含服务，对应解析量 Wq）；
        ``avg_queue_len`` 队列长度的**时间平均**（对应解析量 Lq）；
        ``n_customers`` 处理的顾客数（原样回传）。

    说明:
        平均逗留时间（含服务，对应解析量 W）也可用 ``avg_wait + 1/mu`` 近似核对：
        排队等待与服务时间之和就是系统内停留时间。

    算法:
        1. 到达间隔 ~ Exp(1/λ)，服务时间 ~ Exp(1/μ)，都由逆变换法抽取；
        2. 单服务台事件循环：每个顾客的开始服务时刻 = max(到达时刻, 上一顾客离开时刻)；
        3. 在相邻事件之间按"当前队长"累加时间积分，得到时间平均队长。

    复杂度:
        时间 O(n log n)（事件排序）/ 空间 O(n)。

    陷阱:
        1. 仿真从**空系统**开始，前期有一段"瞬态"（队长偏低），顾客数少时平均等待
           会被系统性低估；n 至少取几千，或丢弃前 10% 的预热样本。
        2. 单次仿真的随机误差约 O(1/sqrt(n))：λ=4、μ=5 时 n=2000 的 avg_wait
           常见与理论值差 20%~50%，**同量级即算通过**，不要期望小数点后两位吻合。
        3. 用"顾客平均队长"（每个顾客到达时看到的队长）和"时间平均队长"是两个不同的量
           （PASTA 只对泊松到达成立），本函数返回的是时间平均，与 Lq 对应。
        4. 时间平均队长按"首个到达 → 最后离开"积分；这段窗口内系统并非全程非空，
           样本量小时两端效应会带来偏差。

    参考:
        Little 1961；离散事件仿真教材（Law & Kelton, "Simulation Modeling and Analysis"）。
    """
    if mu <= 0:
        raise ValueError(f"mu 必须 > 0，得到 {mu}")
    if lam < 0:
        raise ValueError(f"lam 必须 >= 0，得到 {lam}")
    if lam >= mu:
        raise ValueError(f"M/M/1 要求 lam < mu（否则队列无稳态），得到 lam={lam}, mu={mu}")
    if n_customers < 1:
        raise ValueError(f"n_customers 必须 >= 1，得到 {n_customers}")

    gen = rng(seed)
    gaps = gen.exponential(1.0 / lam, size=n_customers)
    service = gen.exponential(1.0 / mu, size=n_customers)
    arrivals = np.cumsum(gaps)

    starts = np.empty(n_customers, dtype=float)
    departs = np.empty(n_customers, dtype=float)
    free_at = 0.0
    for i in range(n_customers):
        start = arrivals[i] if arrivals[i] > free_at else free_at
        starts[i] = start
        free_at = start + service[i]
        departs[i] = free_at

    # 事件循环：统计时间平均队长 / 系统内人数
    t_start = float(arrivals[0])
    t_end = float(departs[-1])
    queue_n = 0
    in_service = 0
    area_queue = 0.0
    t_prev = t_start
    ia = 0
    id_ = 0
    events = 2 * n_customers
    for _ in range(events):
        next_a = arrivals[ia] if ia < n_customers else math.inf
        next_d = departs[id_] if id_ < n_customers else math.inf
        if next_a == math.inf and next_d == math.inf:
            break
        if next_a <= next_d:
            t_now = float(next_a)
            area_queue += queue_n * (t_now - t_prev)
            t_prev = t_now
            ia += 1
            if in_service == 0:
                in_service = 1
            else:
                queue_n += 1
        else:
            t_now = float(next_d)
            area_queue += queue_n * (t_now - t_prev)
            t_prev = t_now
            id_ += 1
            in_service = 0
            if queue_n > 0:
                queue_n -= 1
                in_service = 1

    span = max(t_end - t_start, 1e-12)
    waits = starts - arrivals
    return {
        "avg_wait": float(waits.mean()),
        "avg_queue_len": float(area_queue / span),
        "n_customers": int(n_customers),
    }


def markov_steady_state(P: MatrixLike) -> np.ndarray:
    """求有限马尔可夫链的平稳分布 π，用线性方程组而不是幂迭代。

    参数:
        P: 行随机转移矩阵（非负、每行和 ≈ 1），形状 (n, n)。

    返回:
        np.ndarray，形状 (n,)，平稳分布（非负、和为 1），满足 πP = π。

    算法:
        线性方程组 (Pᵀ - I)π = 0 秩亏 1，用归一化条件 Σπ_i = 1 替换其中一行：
        A[:-1] = Pᵀ - I 的前 n-1 行，A[-1] = 全 1，b = [0,...,0,1]，
        再 np.linalg.solve。最后校验 max|πP - π| 并裁剪浮点小负数。

    复杂度:
        时间 O(n³)（稠密 LU）/ 空间 O(n²)。

    陷阱:
        1. **多解情形**：可约链（例如两个互不连通的状态类）的平稳分布不唯一，
           本线性系统会给出其中一个解（取决于替换行），必须结合链的常返类讨论。
        2. 周期链的平稳分布存在且唯一（若不可约），但**极限分布不存在**：
           π 是时间平均意义上的分布，不要写成"长期后处于状态 i 的概率"。
        3. 输入矩阵必须是行随机（行和为 1）；列随机矩阵要先转置，否则结果全错且不报错。
        4. 用幂迭代虽然实现简单，但收敛速度由第二大特征值决定，接近 1 时要迭代上万次，
           而且无法察觉不唯一性；线性方程组一次解出更稳。

    参考:
        有限马尔可夫链平稳分布的标准定义（如 Norris, "Markov Chains", 1997）。
    """
    p = as_matrix(P, "P")
    check_square(p, "P")
    n = p.shape[0]
    if np.any(p < -1e-12):
        raise ValueError("转移矩阵不能有负元素")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError(f"转移矩阵必须是行随机（每行和为 1），实际行和={row_sums}")

    a = p.T - np.eye(n)
    a[-1, :] = 1.0
    b = np.zeros(n, dtype=float)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:  # 理论上不该发生，防御性处理
        raise ValueError(f"平稳分布的线性方程组奇异：{exc}") from exc
    pi = np.where(np.abs(pi) < 1e-14, 0.0, pi)
    if np.any(pi < -1e-8):
        raise ValueError("求解得到的平稳分布含显著负分量，请检查转移矩阵是否合法")
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()

    residual = float(np.max(np.abs(pi @ p - pi)))
    if residual > 1e-6:
        raise ValueError(f"πP=π 校验失败，最大偏差 {residual:.3e}，矩阵可能不是随机矩阵")
    return pi


def markov_absorption(P: MatrixLike, transient_states: Sequence[int]) -> Dict[str, np.ndarray]:
    """吸收马尔可夫链：算从各瞬态出发被各吸收态吸收的概率与吸收前期望步数。

    参数:
        P: 行随机转移矩阵，形状 (n, n)。
        transient_states: 瞬态状态的下标序列；其余状态视为吸收态（要求 P_ii ≈ 1）。

    返回:
        dict，键为：
        ``absorption_probs`` 形状 (len(transient_states), n_absorbing) 的矩阵，
            第 i 行第 j 列 = 从第 i 个瞬态出发最终被第 j 个吸收态吸收的概率；
            列的顺序与"吸收态下标升序"一致（吸收态 = 全部状态去掉 transient_states）；
        ``expected_steps`` 形状 (len(transient_states),)，吸收前的期望步数。

    算法:
        把 P 分块为 [[Q, R], [0, I]]（Q 是瞬态间转移），则
        基本矩阵 N = (I - Q)^{-1}，吸收概率 B = N R，期望步数 t = N 1。

    复杂度:
        时间 O(n³)/空间 O(n²)。

    陷阱:
        1. **吸收态必须真的吸收**（对角线为 1）：如果传入的 transient_states 漏掉了
           某个带出边的状态，它会被当成吸收态，结果静默错误。本函数会校验并报错。
        2. 期望步数可能发散（瞬态子链不是"吸收必然发生"的），此时 (I-Q) 奇异，
           本函数会在 solve 阶段抛 ValueError 而不是返回 inf。
        3. 期望步数对**步数**计数，若一步代表 1 天则单位是天；换时间单位要重新标定。
        4. 吸收概率每行之和应为 1；若明显不为 1，说明有瞬态被误判，务必打印校验。

    参考:
        吸收马尔可夫链的标准结论（Kemeny & Snell, "Finite Markov Chains", 1960）。
    """
    p = as_matrix(P, "P")
    check_square(p, "P")
    n = p.shape[0]
    trans = sorted({int(s) for s in transient_states})
    if not trans:
        raise ValueError("transient_states 不能为空")
    if trans[0] < 0 or trans[-1] >= n:
        raise ValueError(f"transient_states 下标越界（合法范围 0..{n - 1}）")
    absorbing = np.array([i for i in range(n) if i not in set(trans)], dtype=int)
    if absorbing.size == 0:
        raise ValueError("必须至少有一个吸收态（不在 transient_states 中的状态）")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("转移矩阵必须是行随机（每行和为 1）")
    for i in absorbing:
        if abs(float(p[i, i]) - 1.0) > 1e-9:
            raise ValueError(f"状态 {i} 被当作吸收态，但 P[{i},{i}]={p[i, i]} != 1")

    q = p[np.ix_(trans, trans)]
    r = p[np.ix_(trans, absorbing)]
    m = np.eye(len(trans)) - q
    try:
        n_mat = np.linalg.inv(m)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"瞬态子链不满足吸收必然发生（I-Q 奇异）：{exc}") from exc

    probs = n_mat @ r
    steps = n_mat @ np.ones(len(trans), dtype=float)
    return {
        "absorption_probs": probs,
        "expected_steps": steps,
    }


def gamblers_ruin(
    p: float,
    start: int,
    target: int,
    seed: Optional[int] = None,
    n_trials: int = 20000,
) -> Dict[str, float]:
    """赌徒破产问题：从本金 start 出发、每局以概率 p 赢 1 元，求破产（到 0）概率。

    参数:
        p: 每局获胜概率，取值 (0, 1)。
        start: 初始本金（整数，0 < start < target）。
        target: 目标本金（整数），达到即停止。
        seed: 随机种子。
        n_trials: 模拟次数（>= 1）。

    返回:
        dict，键为：
        ``ruin_prob_sim`` 模拟得到的破产频率；
        ``ruin_prob_theory`` 解析破产概率；
        ``n_trials`` 模拟次数（原样回传）。
        模拟频率的标准误可按 sqrt(p̂(1-p̂)/n_trials) 自行估算（约 1/sqrt(n_trials)）。

    算法:
        解析：令 r = (1-p)/p，则获胜概率 P_win = (1-r^start)/(1-r^target)（p≠0.5），
        p=0.5 时 P_win = start/target；P_ruin = 1 - P_win。
        模拟：n_trials 条独立随机游走，用"活跃游走向量 + 掩码"批量推进，撞到 0 记破产、
        撞到 target 记成功，直到全部停止或达到步数上限。

    复杂度:
        时间 O(n_trials * 步数) / 空间 O(n_trials)。

    陷阱:
        1. **p 对破产概率极其敏感**：start=10、target=20 时，p=0.49 的破产概率约 0.599，
           p=0.51 时降到约 0.401——概率只差 0.02，结论差 20 个百分点。做投资/风险类
           题目时必须做 p 的敏感性分析。
        2. p=0.5 时公式要单独处理（r=1 会让分母为 0）；浮点下用 abs(p-0.5)<1e-12 判断。
        3. 模拟的破产频率标准误约 1/sqrt(n_trials)，n=20000 时精度约 0.003；
           解析值与模拟值差 2~3 个标准误是正常的，不要因此判定实现有错。
        4. **分母口径容易写错**：频率的分母必须是"已结束的游走"（破产的 + 到达目标的），
           若写成"破产的 + 仍在跑的"，成功离场的样本被漏掉，频率会被系统性高估
           （本模块初版就踩了这个坑：n=20000、p=0.49 时算出 1.0 而不是 0.60）。
           步数上限截断而仍未结束的游走才应剔除，本实现用 ``~alive`` 做分母。

    参考:
        经典随机游走问题（Feller, "An Introduction to Probability Theory and Its
        Applications", Vol. 1, Ch. 14）。
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p 必须在 (0, 1) 内，得到 {p}")
    if not isinstance(start, (int, np.integer)) or not isinstance(target, (int, np.integer)):
        raise ValueError("start 与 target 必须是整数")
    if start <= 0:
        raise ValueError(f"start 必须 > 0（否则一开始就破产），得到 {start}")
    if target <= start:
        raise ValueError(f"target 必须 > start，得到 start={start}, target={target}")
    if n_trials < 1:
        raise ValueError(f"n_trials 必须 >= 1，得到 {n_trials}")

    q = 1.0 - p
    if abs(p - 0.5) < 1e-12:
        p_win = start / target
    else:
        r = q / p
        p_win = (1.0 - r ** start) / (1.0 - r ** target)
    p_ruin_theory = 1.0 - p_win

    gen = rng(seed)
    fortune = np.full(n_trials, start, dtype=np.int64)
    alive = np.ones(n_trials, dtype=bool)
    ruined = np.zeros(n_trials, dtype=bool)
    max_steps = 1000 * max(target, 10)  # 破产/达标几乎总会在远小于此的步数内发生
    for _ in range(max_steps):
        if not np.any(alive):
            break
        idx = np.flatnonzero(alive)
        win = gen.random(idx.size) < p
        fortune[idx] += np.where(win, 1, -1)
        ruined[idx] |= fortune[idx] <= 0
        alive[idx] = (fortune[idx] > 0) & (fortune[idx] < target)
    # 分母必须是"已经结束的游走"（破产的 + 达到目标的），不能只数破产+在跑的，
    # 否则成功离场的那些会被漏掉，频率被系统性高估（写这段时的真实踩坑，见自测）。
    finished = ~alive
    n_valid = int(np.count_nonzero(finished))
    p_sim = float(np.count_nonzero(ruined) / max(n_valid, 1))
    return {
        "ruin_prob_sim": p_sim,
        "ruin_prob_theory": float(p_ruin_theory),
        "n_trials": int(n_trials),
    }


def mmc_metrics(lam: float, mu: float, c: int) -> Dict[str, float]:
    """M/M/c 多服务台排队系统的稳态解析指标（Erlang-C 公式）。

    参数:
        lam: 到达率（每单位时间到达的顾客数），必须 >= 0。
        mu: **单台**服务台的服务率，必须 > 0。
        c: 服务台数量，必须是 >= 1 的整数；要求 rho = lam/(c*mu) < 1。

    返回:
        dict，键为：
        ``rho`` 服务台利用率 lam/(c*mu)；
        ``p0`` 系统全空的稳态概率；
        ``p_wait`` 到达时所有服务台都忙、需要排队的概率（Erlang-C 公式）；
        ``Lq`` 队列中平均等待顾客数；``L`` 系统内平均顾客数；
        ``Wq`` 平均排队等待时间；``W`` 平均逗留时间（等待 + 服务）；
        ``erlang_c`` 与 ``p_wait`` 同值，只是给出论文里常用的名字。

    算法:
        1. 记提供负载 a = c*rho = lam/mu，稳态概率
           p0 = [ sum_{n=0}^{c-1} a^n/n! + a^c/(c!(1-rho)) ]^{-1}
           （求和项用 term_n = term_{n-1}*a/n 递推，避免显式构造阶乘）；
        2. Erlang-C：C = a^c/(c!(1-rho)) * p0 = p_wait；
        3. Lq = C*rho/(1-rho)，Wq = Lq/lam，W = Wq + 1/mu，L = Lq + a。
        c=1 时退化为 M/M/1 的 p0=1-rho、C=rho、Lq=rho^2/(1-rho)（自测里逐项核对）。

    复杂度:
        时间 O(c) / 空间 O(1)。

    陷阱:
        1. **rho 必须按 c*mu 归一**：写成 lam/mu 在 c>1 时会把利用率放大 c 倍，
           于是稍微有点负载就"报无稳态"，这是 M/M/c 最常见的实现错误。
        2. ``p_wait`` 是"需要等待的概率"，不是"被拒绝的概率"；后者属于损失制系统
           的 Erlang-B 公式，两者混用是排队论建模的经典错误。
        3. a^c/c! 在 c 大、rho 接近 1 时会溢出成 inf（a 约 700 以上），
           届时需要改到对数域计算，本实现是教学透明版，不做对数域处理。
        4. lam=0 时 Wq 按 0/0 无定义，这里显式返回 0（系统永远空闲）。

    参考:
        Erlang 1917；Kendall 1953；Gross & Harris, "Fundamentals of Queueing Theory"。
    """
    if mu <= 0:
        raise ValueError(f"mu 必须 > 0，得到 {mu}")
    if lam < 0:
        raise ValueError(f"lam 必须 >= 0，得到 {lam}")
    if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
        raise ValueError(f"c 必须是整数，得到 {c!r}")
    c = int(c)
    if c < 1:
        raise ValueError(f"c 必须 >= 1，得到 {c}")
    rho = lam / (c * mu)
    if rho >= 1.0:
        raise ValueError(
            f"M/M/c 要求 rho = lam/(c*mu) < 1（否则队列无稳态），"
            f"得到 lam={lam}, mu={mu}, c={c}, rho={rho}"
        )

    a = lam / mu
    term = 1.0          # a^n / n!，从 n=0 开始
    total = 1.0         # sum_{n=0}^{c-1} a^n/n!
    for n in range(1, c):
        term *= a / n
        total += term
    term_c = term * a / c           # a^c / c!
    geo = term_c / (1.0 - rho)      # a^c/(c!(1-rho))
    p0 = 1.0 / (total + geo)
    p_wait = geo * p0
    lq = p_wait * rho / (1.0 - rho)
    wq = lq / lam if lam > 0.0 else 0.0
    w = wq + 1.0 / mu
    return {
        "rho": float(rho),
        "p0": float(p0),
        "p_wait": float(p_wait),
        "Lq": float(lq),
        "L": float(lq + a),
        "Wq": float(wq),
        "W": float(w),
        "erlang_c": float(p_wait),
    }


def mg1_metrics(lam: float, service_mean: float, service_var: float) -> Dict[str, float]:
    """M/G/1 排队系统（泊松到达 + 一般服务时间）的 Pollaczek-Khinchine 稳态指标。

    参数:
        lam: 到达率（> 0）。
        service_mean: 服务时间均值 E[S]（> 0）。
        service_var: 服务时间方差 Var[S]（>= 0）。
        要求 rho = lam*E[S] < 1。

    返回:
        dict，键为：
        ``rho`` 利用率 lam*E[S]；
        ``Wq`` 平均排队等待时间；``W`` 平均逗留时间 Wq + E[S]；
        ``Lq`` 队列平均长度 lam*Wq；``L`` 系统内平均顾客数 lam*W。

    算法:
        Pollaczek-Khinchine 公式：Wq = lam*E[S^2] / (2(1-rho))，
        其中 E[S^2] = Var[S] + E[S]^2（用二阶矩而不是方差，是最容易写错的一步）。
        再套 Little 公式 Lq = lam*Wq、L = lam*W。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. 服务时间方差进的是**二阶矩** E[S^2]，写成 Var[S] 会把 Wq 系统性算小
           （差 lam*E[S]^2/(2(1-rho)) 这一项）。
        2. M/G/1 的 Wq 只依赖服务时间的前两阶矩，与分布形状无关——这是 P-K 公式
           的威力，也意味着不能据此推断尾概率（例如超时率）。
        3. rho→1 时同样爆炸；而 rho=0 时若 Var[S]>0，Wq 仍为 0（没有到达就没有排队）。
        4. 服务时间必须与到达过程独立；批量到达或到达与服务的相关性会破坏公式前提。

    参考:
        Pollaczek 1930；Khinchine 1932；Gross & Harris, "Fundamentals of Queueing Theory"。
    """
    if service_mean <= 0:
        raise ValueError(f"service_mean 必须 > 0，得到 {service_mean}")
    if service_var < 0:
        raise ValueError(f"service_var 必须 >= 0，得到 {service_var}")
    if lam < 0:
        raise ValueError(f"lam 必须 >= 0，得到 {lam}")
    rho = lam * service_mean
    if rho >= 1.0:
        raise ValueError(
            f"M/G/1 要求 rho = lam*E[S] < 1（否则队列无稳态），"
            f"得到 lam={lam}, E[S]={service_mean}, rho={rho}"
        )
    second_moment = service_var + service_mean * service_mean
    wq = lam * second_moment / (2.0 * (1.0 - rho))
    w = wq + service_mean
    return {
        "rho": float(rho),
        "Wq": float(wq),
        "W": float(w),
        "Lq": float(lam * wq),
        "L": float(lam * w),
    }


def _des_core(
    arrival_fn: Callable[[np.random.Generator], float],
    service_fn: Callable[[np.random.Generator], float],
    n_customers: int,
    n_servers: int,
    seed: Optional[int],
    t_max: Optional[float],
) -> Dict[str, object]:
    """离散事件引擎内核：返回公开结果之外还带 ``lq_time_avg`` / ``utilization`` / ``span``。

    参数:
        arrival_fn: 入参为 Generator、返回"下一个到达间隔"的可调用对象。
        service_fn: 入参为 Generator、返回"服务时长"的可调用对象。
        n_customers: 计划到达的顾客数。
        n_servers: 服务台数。
        seed: 随机种子。
        t_max: 仿真时间上限（含）；None 表示不限制。

    返回:
        dict：``wait`` / ``sojourn`` / ``departure`` 为长度 n_served 的数组（按到达序），
        ``server_busy`` 为长度 n_servers 的累计忙期，``n_served`` 为已服务人数，
        以及 ``lq_time_avg``（队列长度时间平均）、``utilization``、``span``。

    算法:
        1. 事件堆（heapq）按 (时刻, 序号) 排序，到达与离开都是事件；
        2. 服务台用"空闲堆"管理：到达时若空闲堆非空立即开始服务，否则进 FIFO 等待队列；
           离开时优先把等待队列队首的顾客放到刚空出的台上（FIFO、不抢占）；
        3. 队列长度只在事件时刻变化，故按"上一事件到当前事件的时长 × 当前队列长度"累加
           面积，最后除以观测窗口得到时间平均队长。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. 必须用"事件驱动"而不是先抽样全部到达时刻再排队：只有前者能在 t_max 处
           干净地截断（后者需要丢弃 t_max 之后的事件）。
        2. 等待队列必须是 FIFO：若用 LIFO（栈），等待时间的分布会被系统性高估。
        3. 时间平均队长的积分窗口取"首个到达 → 最后离开"，窗口两端系统并非全程非空，
           样本量小时会带来偏差。

    参考:
        离散事件仿真的标准"下一事件"框架（Law & Kelton）。
    """
    gen = rng(seed)
    free: List[Tuple[float, int]] = [(0.0, j) for j in range(n_servers)]
    heapq.heapify(free)
    events: List[Tuple[float, int, int, tuple]] = []
    seq = 0
    busy = np.zeros(n_servers, dtype=float)
    waits = np.full(n_customers, np.nan, dtype=float)
    sojourns = np.full(n_customers, np.nan, dtype=float)
    departs = np.full(n_customers, np.nan, dtype=float)
    waiting: "deque[Tuple[int, float]]" = deque()

    t_first = float(arrival_fn(gen))
    if not t_first >= 0.0:
        raise ValueError(f"arrival_fn 必须返回非负的到达间隔，得到 {t_first}")
    if t_max is None or t_first <= t_max:
        heapq.heappush(events, (t_first, seq, 0, (0, t_first)))
        seq += 1
    next_customer = 1

    served = 0
    qlen = 0
    area = 0.0
    t_prev: Optional[float] = None
    t_start = 0.0
    while events:
        t, _, kind, payload = heapq.heappop(events)
        if t_prev is None:
            t_prev = t
            t_start = t  # 首个到达时刻，作为观测窗口的左端
        area += qlen * (t - t_prev)
        t_prev = t

        if kind == 0:  # 到达事件
            idx, arr_t = payload
            if free:
                ft, sid = heapq.heappop(free)
                start = ft if ft > t else t
                st = float(service_fn(gen))
                if not st >= 0.0:
                    raise ValueError(f"service_fn 必须返回非负的服务时长，得到 {st}")
                heapq.heappush(events, (start + st, seq, 1, (sid, st, arr_t, start, idx)))
                seq += 1
            else:
                waiting.append((idx, arr_t))
            if next_customer < n_customers:
                gap = float(arrival_fn(gen))
                if not gap >= 0.0:
                    raise ValueError(f"arrival_fn 必须返回非负的到达间隔，得到 {gap}")
                t_next = t + gap
                if t_max is None or t_next <= t_max:
                    heapq.heappush(events, (t_next, seq, 0, (next_customer, t_next)))
                    seq += 1
            next_customer += 1
        else:  # 离开事件
            sid, st, arr_t, start, idx = payload
            busy[sid] += st
            waits[idx] = start - arr_t
            sojourns[idx] = t - arr_t
            departs[idx] = t
            served += 1
            if waiting:
                w_idx, w_arr = waiting.popleft()
                st2 = float(service_fn(gen))
                if not st2 >= 0.0:
                    raise ValueError(f"service_fn 必须返回非负的服务时长，得到 {st2}")
                heapq.heappush(events, (t + st2, seq, 1, (sid, st2, w_arr, t, w_idx)))
                seq += 1
            else:
                heapq.heappush(free, (t, sid))
        qlen = len(waiting)

    if served == 0:
        span = 0.0
        lq_avg = 0.0
        utilization = 0.0
    else:
        # 观测窗口 = 首个到达 → 最后离开；两端系统并非全程非空，样本量小时有偏差
        span = float(departs[served - 1] - t_start)
        lq_avg = area / span if span > 0 else 0.0
        utilization = float(busy.sum() / (n_servers * span)) if span > 0 else 0.0
    return {
        "wait": waits[:served],
        "sojourn": sojourns[:served],
        "departure": departs[:served],
        "server_busy": busy,
        "n_served": int(served),
        "lq_time_avg": float(lq_avg),
        "utilization": float(utilization),
        "span": float(span),
    }


def discrete_event_simulation(
    arrival_fn: Callable[[np.random.Generator], float],
    service_fn: Callable[[np.random.Generator], float],
    n_customers: int,
    n_servers: int = 1,
    seed: Optional[int] = None,
    t_max: Optional[float] = None,
) -> Dict[str, object]:
    """通用离散事件仿真（事件堆 heapq）：单队列多服务台、FIFO、不抢占。

    参数:
        arrival_fn: 可调用对象，签名 ``arrival_fn(gen) -> float``，返回**下一个到达间隔**
            （不是到达时刻）；gen 是 ``_common.rng(seed)`` 给出的 Generator，
            所以随机性完全由 seed 控制。常用写法 ``lambda g: g.exponential(1/lam)``。
        service_fn: 可调用对象，签名 ``service_fn(gen) -> float``，返回服务时长。
        n_customers: 计划到达的顾客数（>= 1）。
        n_servers: 服务台数（>= 1），各台同质、服务时间独立同分布。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        t_max: 仿真时间上限（含）；None 表示把 n_customers 全部服务完为止。
            超过 t_max 的到达不再生成，因此 ``n_served`` 可能小于 n_customers。

    返回:
        dict，键为：
        ``wait`` 形状 (n_served,) 的排队等待时间（不含服务），按到达顺序排列；
        ``sojourn`` 形状 (n_served,) 的逗留时间（等待 + 服务）；
        ``server_busy`` 形状 (n_servers,) 的**累计忙期时长**，
            利用率 = server_busy.sum() / (n_servers * 观测窗口)；
        ``n_served`` 实际完成服务的顾客数（int）；
        ``departure`` 形状 (n_served,) 的离开时刻，按到达顺序排列。

    算法:
        1. 事件堆按 (时刻, 序号) 排序，只有"到达"与"离开"两类事件；
        2. 服务台用空闲堆维护；到达时若无空台则进 FIFO 等待队列，
           离开时把等待队列队首放到刚空出的台上（不抢占）；
        3. 队列长度在事件时刻变化，按区间长度累加面积，可得到时间平均队长
           （``_des_core`` 返回，公开接口不暴露，供 ``mmc_simulate`` 使用）。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. arrival_fn 返回的是**间隔**而不是时刻，传成时刻会让所有顾客挤在同一时间到达。
        2. 仿真从"全空系统"起步：n_customers 很小（几百）时前几十个顾客几乎不等待，
           平均等待被系统性低估，必须加大样本或丢弃预热期。
        3. ``server_busy`` 是累计时长而不是比例：除以窗口长度才是利用率，
           且窗口取的是"首个到达 → 最后离开"，不是 [0, t_max]。
        4. 若服务强度 rho >= 1，队列会无限增长，仿真不会报错但结果没有意义；
           请在调用方（如 ``mmc_simulate``）自己校验参数。

    参考:
        离散事件仿真的"下一事件"框架（Law & Kelton, "Simulation Modeling and Analysis"）。
    """
    if not callable(arrival_fn):
        raise ValueError("arrival_fn 必须是可调用的（接受 Generator、返回到达间隔）")
    if not callable(service_fn):
        raise ValueError("service_fn 必须是可调用的（接受 Generator、返回服务时长）")
    if isinstance(n_customers, bool) or not isinstance(n_customers, (int, np.integer)):
        raise ValueError(f"n_customers 必须是整数，得到 {n_customers!r}")
    if isinstance(n_servers, bool) or not isinstance(n_servers, (int, np.integer)):
        raise ValueError(f"n_servers 必须是整数，得到 {n_servers!r}")
    n_customers = int(n_customers)
    n_servers = int(n_servers)
    if n_customers < 1:
        raise ValueError(f"n_customers 必须 >= 1，得到 {n_customers}")
    if n_servers < 1:
        raise ValueError(f"n_servers 必须 >= 1，得到 {n_servers}")
    if t_max is not None and not t_max > 0:
        raise ValueError(f"t_max 必须 > 0（或 None），得到 {t_max}")

    core = _des_core(arrival_fn, service_fn, n_customers, n_servers, seed, t_max)
    return {
        "wait": core["wait"],
        "sojourn": core["sojourn"],
        "server_busy": core["server_busy"],
        "n_served": core["n_served"],
        "departure": core["departure"],
    }


def mmc_simulate(
    lam: float,
    mu: float,
    c: int,
    n_customers: int = 2000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """用 ``discrete_event_simulation`` 仿真 M/M/c，得到可解析对照的排队指标。

    参数:
        lam: 到达率（> 0，指数到达间隔）。
        mu: 单台服务率（> 0，指数服务时间）。
        c: 服务台数（>= 1）；要求 lam < c*mu，否则直接抛 ValueError（不发散仿真）。
        n_customers: 仿真顾客数（>= 1）；建议 >= 20000 才能把 Wq 的相对误差压到 15% 内。
        seed: 随机种子。

    返回:
        dict，键为：
        ``Lq`` 队列长度的**时间平均**（对应解析量 Lq）；
        ``Wq`` 顾客平均排队等待时间（对应解析量 Wq）；
        ``W`` 顾客平均逗留时间（对应解析量 W）；
        ``utilization`` 服务台时间利用率 = 总忙期/(c*窗口)；
        ``n_served`` 实际完成服务的顾客数（等于 n_customers）。

    算法:
        把 Exp(1/lam) 的到达间隔与 Exp(1/mu) 的服务时长交给通用事件引擎，
        再对结果做统计；时间平均队长由引擎内部按"队列长度 × 时长"积分得到
        （PASTA 保证泊松到达下它与"顾客看到的队长"一致）。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. 单次仿真的随机误差约 O(1/sqrt(n))：c=2、lam=3、mu=2 时 n=2000 的 Wq
           相对误差常在 10%~30% 波动，必须加大样本量再下结论（自测里用 20000）。
        2. 系统从空开始，存在一段"队长偏低"的瞬态；n 越大相对偏差越小。
        3. ``Lq`` 是时间平均，与"每个顾客到达时看到的队长"在非泊松到达下并不相等。
        4. 仿真结果依赖 seed；论文中必须同时报告 seed 与样本量，否则无法复现。

    参考:
        Little 1961；Law & Kelton, "Simulation Modeling and Analysis"。
    """
    if mu <= 0:
        raise ValueError(f"mu 必须 > 0，得到 {mu}")
    if lam <= 0:
        raise ValueError(f"lam 必须 > 0（指数到达间隔要求），得到 {lam}")
    if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
        raise ValueError(f"c 必须是整数，得到 {c!r}")
    c = int(c)
    if c < 1:
        raise ValueError(f"c 必须 >= 1，得到 {c}")
    if lam >= c * mu:
        raise ValueError(
            f"M/M/c 要求 lam < c*mu（否则队列无稳态），得到 lam={lam}, c={c}, mu={mu}"
        )
    if isinstance(n_customers, bool) or not isinstance(n_customers, (int, np.integer)):
        raise ValueError(f"n_customers 必须是整数，得到 {n_customers!r}")
    n_customers = int(n_customers)
    if n_customers < 1:
        raise ValueError(f"n_customers 必须 >= 1，得到 {n_customers}")

    core = _des_core(
        lambda g: float(g.exponential(1.0 / lam)),
        lambda g: float(g.exponential(1.0 / mu)),
        n_customers,
        c,
        seed,
        None,
    )
    waits = core["wait"]
    sojourns = core["sojourn"]
    n_served = int(core["n_served"])
    return {
        "Lq": float(core["lq_time_avg"]),
        "Wq": float(np.mean(waits)) if n_served > 0 else 0.0,
        "W": float(np.mean(sojourns)) if n_served > 0 else 0.0,
        "utilization": float(core["utilization"]),
        "n_served": n_served,
    }


def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    x0: ArrayLike,
    n_samples: int = 5000,
    proposal_sd: float = 1.0,
    seed: Optional[int] = None,
    burn_in: int = 1000,
) -> Dict[str, np.ndarray]:
    """随机游走 Metropolis-Hastings 采样：只需要目标密度的**对数**（可差常数）。

    参数:
        log_target: 可调用对象，输入形状 (d,) 的 ndarray，返回对数目标密度（标量）。
        x0: 初始点，可转成一维数组（长度 d）；p 维目标就传 p 个分量。
        n_samples: 保留的样本数（>= 1），预热样本不计入。
        proposal_sd: 对称随机游走提议的步长（> 0）；多维时各维同尺度。
        seed: 随机种子。
        burn_in: 预热步数（>= 0），前 burn_in 步只更新链不记录。

    返回:
        dict，键为：
        ``samples`` 形状 (n_samples, d) 的样本矩阵；
        ``accept_rate`` 采样阶段（不含预热）的接受率；
        ``mean`` 形状 (d,) 的样本均值；``var`` 形状 (d,) 的样本方差（ddof=1）。

    算法:
        1. 提议 x' = x + sd * N(0, I)（对称提议，接受比只剩密度比）；
        2. 以概率 min(1, exp(log_target(x') - log_target(x))) 接受，
           这里用 ``log(u) < lp' - lp`` 判断，避免 exp 下溢；
        3. 前 burn_in 步不记录，之后逐步记录当前状态并统计接受率。

    复杂度:
        时间 O((n_samples + burn_in) * d) / 空间 O(n_samples * d)。

    陷阱:
        1. **步长是唯一需要调的参数**：proposal_sd 太小接受率接近 1 但链混合极慢
           （样本高度自相关，均值的标准误被低估）；太大则接受率骤降、链长期不动。
           一维标准正态的经验最优接受率约 0.44，高维约 0.234。
        2. 本函数返回的 ``var`` 是**样本方差**，不是均值的方差；要报"均值 ± 标准误"
           必须考虑自相关（用批均值法估计），直接用 s/sqrt(n) 会低估误差。
        3. 初始点若落在对数密度为 -inf 的区域，前几步只能向上爬，务必给够 burn_in。
        4. 提议是各向同性的：目标各维尺度差异大时（例如方差 1 与 100），
           必须自己做预条件（换坐标/用协方差提议），否则混合极差。

    参考:
        Metropolis et al. 1953；Hastings 1970；Robert & Casella, "Monte Carlo
        Statistical Methods"（接受率与步长的经验准则）。
    """
    x0v = as_vector(x0, "x0")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"n_samples 必须是整数，得到 {n_samples!r}")
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)):
        raise ValueError(f"burn_in 必须是整数，得到 {burn_in!r}")
    n_samples = int(n_samples)
    burn_in = int(burn_in)
    if n_samples < 1:
        raise ValueError(f"n_samples 必须 >= 1，得到 {n_samples}")
    if burn_in < 0:
        raise ValueError(f"burn_in 必须 >= 0，得到 {burn_in}")
    if not proposal_sd > 0:
        raise ValueError(f"proposal_sd 必须 > 0，得到 {proposal_sd}")

    gen = rng(seed)
    dim = x0v.size
    x = x0v.copy()
    lp = float(log_target(x))
    if math.isnan(lp):
        raise ValueError("log_target(x0) 返回了 NaN，无法开始采样")
    samples = np.empty((n_samples, dim), dtype=float)
    n_accept = 0
    for step in range(burn_in + n_samples):
        proposal = x + proposal_sd * gen.standard_normal(dim)
        lp_new = float(log_target(proposal))
        if math.isnan(lp_new):
            raise ValueError("log_target 对提议点返回了 NaN，请检查其定义域")
        accepted = math.log(gen.random()) < lp_new - lp
        if accepted:
            x = proposal
            lp = lp_new
        if step >= burn_in:
            samples[step - burn_in] = x
            if accepted:
                n_accept += 1
    return {
        "samples": samples,
        "accept_rate": float(n_accept / n_samples),
        "mean": samples.mean(axis=0),
        "var": samples.var(axis=0, ddof=1),
    }


def gibbs_sampler_bivariate_normal(
    mu: ArrayLike,
    cov: MatrixLike,
    n_samples: int = 5000,
    seed: Optional[int] = None,
    burn_in: int = 1000,
) -> Dict[str, np.ndarray]:
    """二元正态的 Gibbs 采样：用条件分布的闭式解交替更新两个分量。

    参数:
        mu: 均值向量，长度 2。
        cov: 2x2 协方差矩阵（对称正定）。
        n_samples: 保留的样本数（>= 1）。
        seed: 随机种子。
        burn_in: 预热步数（>= 0）。

    返回:
        dict，键为：
        ``samples`` 形状 (n_samples, 2) 的样本矩阵；
        ``mean`` 形状 (2,) 的样本均值；
        ``cov`` 形状 (2, 2) 的样本协方差（ddof=1）。

    算法:
        对二元正态，条件分布仍是正态：
        X1|X2=x2 ~ N(mu1 + rho*s1/s2*(x2-mu2), s1^2(1-rho^2))，
        X2|X1 对称。每次迭代先抽 X1|X2 再抽 X2|X1，预热后记录。
        （Gibbs 是接受率恒为 1 的 MH，因此不需要拒绝步骤。）

    复杂度:
        时间 O(n_samples) / 空间 O(n_samples)。

    陷阱:
        1. **必须真的交替使用最新值**：用同一轮的旧 X2 去抽 X2 会退化成独立采样，
           样本相关结构全错（这在实现里非常常见）。
        2. 相关系数接近 ±1 时 Gibbs 混合极慢（自相关约 rho^2），需要大量样本；
           此时应先做变量替换（例如对 (X1, X2-X1) 采样）再变换回去。
        3. 样本协方差是估计量：n_samples=5000 时每个元素的随机误差约 0.02~0.03，
           不要用 1e-3 级别的容差去断言。
        4. 这里只支持二元；多元要用逐分量条件分布，且需自己处理协方差求逆。

    参考:
        Geman & Geman 1984；Casella & George 1992（Gibbs 采样的入门讲解）。
    """
    m = as_vector(mu, "mu")
    s = as_matrix(cov, "cov")
    if m.size != 2:
        raise ValueError(f"mu 必须是长度 2 的向量，得到长度 {m.size}")
    if s.shape != (2, 2):
        raise ValueError(f"cov 必须是 2x2 矩阵，得到形状 {s.shape}")
    if abs(s[0, 1] - s[1, 0]) > 1e-12:
        raise ValueError(f"cov 必须对称，得到 off-diagonal {s[0, 1]} vs {s[1, 0]}")
    if not (s[0, 0] > 0 and s[1, 1] > 0):
        raise ValueError(f"cov 对角线必须为正，得到 {s[0, 0]}, {s[1, 1]}")
    sd1 = math.sqrt(float(s[0, 0]))
    sd2 = math.sqrt(float(s[1, 1]))
    rho = float(s[0, 1]) / (sd1 * sd2)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"cov 必须正定（|相关系数| < 1），得到 rho={rho}")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError(f"n_samples 必须是整数，得到 {n_samples!r}")
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)):
        raise ValueError(f"burn_in 必须是整数，得到 {burn_in!r}")
    n_samples = int(n_samples)
    burn_in = int(burn_in)
    if n_samples < 1:
        raise ValueError(f"n_samples 必须 >= 1，得到 {n_samples}")
    if burn_in < 0:
        raise ValueError(f"burn_in 必须 >= 0，得到 {burn_in}")

    gen = rng(seed)
    x1, x2 = float(m[0]), float(m[1])
    cond_sd1 = sd1 * math.sqrt(1.0 - rho * rho)
    cond_sd2 = sd2 * math.sqrt(1.0 - rho * rho)
    samples = np.empty((n_samples, 2), dtype=float)
    for step in range(burn_in + n_samples):
        mean1 = m[0] + rho * sd1 / sd2 * (x2 - m[1])
        x1 = float(mean1 + cond_sd1 * gen.standard_normal())
        mean2 = m[1] + rho * sd2 / sd1 * (x1 - m[0])
        x2 = float(mean2 + cond_sd2 * gen.standard_normal())
        if step >= burn_in:
            samples[step - burn_in, 0] = x1
            samples[step - burn_in, 1] = x2
    return {
        "samples": samples,
        "mean": samples.mean(axis=0),
        "cov": np.cov(samples, rowvar=False, ddof=1),
    }


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """标准正态分布函数 Φ(z)，用 math.erf 逐元素实现（numpy 不提供 erf）。

    参数:
        z: 任意形状的数组。

    返回:
        与输入同形状的数组，取值 (0, 1)。

    算法:
        Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))。

    复杂度:
        时间 O(size) / 空间 O(size)。

    陷阱:
        极端负值（z < -37）下 1+erf 会发生灾难性抵消，结果直接变成 0；
        教学场景够用，需要高精度尾概率时应改用 erfc。

    参考:
        标准正态分布函数的定义。
    """
    flat = np.asarray(z, dtype=float).ravel()
    out = np.empty(flat.size, dtype=float)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i, v in enumerate(flat):
        out[i] = 0.5 * (1.0 + math.erf(float(v) * inv_sqrt2))
    return out.reshape(np.shape(z))


def _betacf(a: float, b: float, x: float, max_iter: int = 300) -> float:
    """正则化不完全 Beta 函数的连分式部分（Lentz 修正的 Numerical Recipes 版本）。

    参数:
        a, b: Beta 函数参数（> 0）。
        x: 自变量，0 <= x <= 1。
        max_iter: 连分式最大迭代次数。

    返回:
        float，连分式的收敛值。

    算法:
        modified Lentz 算法（乘性修正，避免 0/0），收敛判据 |delta-1| < 3e-14。

    复杂度:
        时间 O(max_iter) / 空间 O(1)。

    陷阱:
        x 接近 1 且 a >> b 时收敛会变慢；``_betainc`` 用对称变换规避。

    参考:
        Press et al., "Numerical Recipes", 6.4 节。
    """
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。

    参数:
        a, b: Beta 函数参数（> 0）。
        x: 自变量，区间 [0, 1]。

    返回:
        float，I_x(a, b) = B(x; a, b) / B(a, b)。

    算法:
        先用 lgamma 算前置因子，再按 x 与 (a+1)/(a+b+2) 的大小选直接连分式或
        对称变换 I_x(a,b) = 1 - I_{1-x}(b,a)（哪边收敛快用哪边）。

    复杂度:
        时间 O(max_iter) / 空间 O(1)。

    陷阱:
        区间外（x<0 或 x>1）直接截断返回 0/1，不报错——调用方需自己保证 x 合法。

    参考:
        Press et al., "Numerical Recipes", 6.4 节。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(x: float, df: float) -> float:
    """Student t 分布函数 P(T <= x)，自由度 df > 0。

    参数:
        x: 分位点。
        df: 自由度（> 0，允许非整数）。

    返回:
        float，取值 (0, 1)。

    算法:
        令 z = df/(df + x^2)，则 x<0 时 P(T<=x) = 0.5*I_z(df/2, 0.5)，
        x>0 时为 1 - 0.5*I_z(df/2, 0.5)（t 分布的对称性 + 不完全 Beta 表示）。

    复杂度:
        时间 O(max_iter) / 空间 O(1)。

    陷阱:
        df 很大且 |x| 很大时 z 极其接近 1，是靠 ``_betainc`` 的对称分支保证收敛；
        直接对 z 用连分式会不收敛。

    参考:
        Abramowitz & Stegun 26.7.4；Student 1908。
    """
    if not df > 0:
        raise ValueError(f"df 必须 > 0，得到 {df}")
    if x == 0.0:
        return 0.5
    z = df / (df + x * x)
    half = 0.5 * _betainc(0.5 * df, 0.5, z)
    return half if x < 0 else 1.0 - half


def _student_t_cdf_array(x: np.ndarray, df: float) -> np.ndarray:
    """对数组逐元素调用 ``_student_t_cdf``（t 分布函数没有 numpy 向量化版本）。

    参数:
        x: 任意形状的数组。
        df: 自由度（> 0）。

    返回:
        与输入同形状的数组。

    算法:
        逐元素调用 ``_student_t_cdf``。

    复杂度:
        时间 O(size * max_iter) / 空间 O(size)。

    陷阱:
        逐元素 Python 循环，样本量很大时会成为瓶颈（n_draws=1e6 就别用了）。

    参考:
        本模块 ``_student_t_cdf``。
    """
    flat = np.asarray(x, dtype=float).ravel()
    out = np.empty(flat.size, dtype=float)
    for i, v in enumerate(flat):
        out[i] = _student_t_cdf(float(v), df)
    return out.reshape(np.shape(x))


def _rank_average(x: np.ndarray) -> np.ndarray:
    """平均秩：并列值取名次的平均值（Spearman 相关的标准处理）。

    参数:
        x: 一维数组。

    返回:
        长度相同的一维数组，秩从 1 开始。

    算法:
        先按稳定排序给出名次，再对相邻相等值整段替换为平均名次。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        不处理并列会让秩相关在离散/含大量重复的数据上被系统性高估。

    参考:
        Spearman 1904。
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_x = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return ranks


def _spearman_to_pearson(u: np.ndarray) -> np.ndarray:
    """由伪观测 U 估计 Spearman 秩相关矩阵，再换算成高斯 copula 的 Pearson ρ。

    参数:
        u: 形状 (n, d) 的伪观测，取值 [0, 1]。

    返回:
        形状 (d, d) 的对称矩阵 ρ = 2*sin(π*ρ_s/6)，对角线为 1。

    算法:
        1. 对每列取平均秩，算秩的 Pearson 相关矩阵得到 ρ_s；
        2. 用高斯 copula 的精确换算 ρ = 2*sin(π*ρ_s/6)（由 E[Φ(X)Φ(Y)] 的
           反正弦公式 ρ_s = (6/π) arcsin(ρ/2) 反解）；
        3. 特征值截断到 1e-10 保证正定（小样本 + 高维时 ρ_s 可能非正定）。

    复杂度:
        时间 O(n d log n + d^3) / 空间 O(d^2)。

    陷阱:
        1. 换算的是 Pearson ρ，不是秩相关本身；直接拿 ρ_s 当 ρ 用会让 t copula
           的尾相依系数明显偏小。
        2. 样本量小（n < 50）时 ρ_s 波动很大，换算出的 ρ 可能超出 (-1, 1) 边界或
           使矩阵非正定，必须做截断。

    参考:
        Kruskal 1958（Spearman 与 Pearson 的换算）；Nelsen, "An Introduction to
        Copulas", 2nd ed.。
    """
    n, d = u.shape
    ranks = np.empty((n, d), dtype=float)
    for j in range(d):
        ranks[:, j] = _rank_average(u[:, j])
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = centered / std
    rho_s = (z.T @ z) / n
    rho_s = np.clip(rho_s, -1.0, 1.0)
    rho = 2.0 * np.sin(math.pi * rho_s / 6.0)
    np.fill_diagonal(rho, 1.0)
    rho = (rho + rho.T) / 2.0
    eig = np.linalg.eigvalsh(rho)
    if float(eig.min()) < 1e-10:
        vals, vecs = np.linalg.eigh(rho)
        vals = np.clip(vals, 1e-10, None)
        rho = vecs @ np.diag(vals) @ vecs.T
        diag = np.sqrt(np.diag(rho))
        rho = rho / np.outer(diag, diag)
        np.fill_diagonal(rho, 1.0)
    return rho


def _average_offdiag_rho(rho: np.ndarray) -> float:
    """取相关矩阵非对角元的平均值（d=2 时就是唯一的相关系数）。

    参数:
        rho: 形状 (d, d) 的对称相关矩阵，d >= 2。

    返回:
        float，所有 i<j 元素的均值。

    算法:
        对上三角非对角元求和再除以对数 d(d-1)/2。

    复杂度:
        时间 O(d^2) / 空间 O(1)。

    陷阱:
        d>2 时"一个 ρ"只是概要统计；严格的尾相依要逐对报告，
        本模块的 tail_dependence 因此只对 d=2 有精确解释（已在 docstring 说明）。

    参考:
        无。
    """
    d = rho.shape[0]
    if d < 2:
        raise ValueError(f"相关矩阵至少需要 2 维，得到 {d}")
    idx = np.triu_indices(d, k=1)
    return float(rho[idx].mean())


def gaussian_copula(
    U: MatrixLike,
    n_draws: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """高斯 copula：由伪观测估计相关结构，再生成相依的均匀样本。

    参数:
        U: 形状 (n, d) 的伪观测矩阵，每列取值 [0, 1]（秩变换/经验分布函数得到），
            n >= 3 且 d >= 2。
        n_draws: 生成的样本数（>= 1）。
        seed: 随机种子。

    返回:
        dict，键为：
        ``rho`` 形状 (d, d) 的高斯 copula 相关矩阵
            （由 Spearman 秩相关换算：ρ = 2*sin(π*ρ_s/6)）；
        ``samples`` 形状 (n_draws, d) 的均匀样本，每列边缘为 U(0,1)；
        ``tail_dependence`` 下尾相依系数，高斯 copula 恒为 0.0（渐近独立）。

    算法:
        1. 对 U 每列取平均秩得到 Spearman 相关矩阵 ρ_s，换算成 Pearson 相关 ρ；
        2. Cholesky 分解 ρ = LLᵀ，抽 Z = L*N(0, I)，则 Z ~ N(0, ρ)；
        3. 逐元素取标准正态分布函数得到 U_i = Φ(Z_i)，每列严格 U(0,1)
           且保留 ρ 刻画的相依结构。

    复杂度:
        时间 O(n d log n + d^3 + n_draws * d^2) / 空间 O(n_draws * d)。

    陷阱:
        1. **高斯 copula 没有尾部相依**（λ_U = λ_L = 0）：它无法刻画"极端事件同时
           发生"的风险，2008 年金融危机中 CDO 定价误用高斯 copula 正是栽在这里。
           要建模尾部相依必须换成 t copula（见 ``t_copula``）。
        2. U 必须是**伪观测**（[0,1] 上的秩），不是原始数据；直接喂原始值会得到
           毫无意义的 ρ。
        3. d>2 时返回的 ``tail_dependence`` 恒为 0 是理论值，不是 d 的函数；
           逐对尾相依请自行按 ``rho[i, j]`` 单独计算。
        4. 样本量小的时候 ρ_s 可能使 ρ 非正定，本实现做特征值截断，
           代价是略微改变相关结构（截断量级 1e-10，不影响教学结论）。

    参考:
        Sklar 1959；Nelsen, "An Introduction to Copulas"；Embrechts, McNeil &
        Straumann 2002（copula 与尾部风险）。
    """
    u = as_matrix(U, "U")
    n, d = u.shape
    if n < 3:
        raise ValueError(f"U 至少需要 3 行伪观测，得到 {n}")
    if d < 2:
        raise ValueError(f"U 至少需要 2 列，得到 {d}")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("U 必须是伪观测（每列取值在 [0, 1] 内）")
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise ValueError(f"n_draws 必须是整数，得到 {n_draws!r}")
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(f"n_draws 必须 >= 1，得到 {n_draws}")

    rho = _spearman_to_pearson(u)
    gen = rng(seed)
    chol = np.linalg.cholesky(rho)
    z = gen.standard_normal((n_draws, d)) @ chol.T
    return {
        "rho": rho,
        "samples": _norm_cdf(z),
        "tail_dependence": 0.0,
    }


def t_copula(
    U: MatrixLike,
    df: int = 5,
    n_draws: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """t copula：用 t 分布替换正态，从而刻画**尾部相依**（极端事件同时发生）。

    参数:
        U: 形状 (n, d) 的伪观测矩阵（取值 [0, 1]），n >= 3 且 d >= 2。
        df: 自由度（整数 >= 1）；df 越小尾部越厚、尾相依越强，df→∞ 退化为高斯 copula。
        n_draws: 生成的样本数（>= 1）。
        seed: 随机种子。

    返回:
        dict，键为：
        ``rho`` 形状 (d, d) 的相关矩阵（同样由 ρ = 2*sin(π*ρ_s/6) 换算）；
        ``df`` 自由度（原样回传）；
        ``samples`` 形状 (n_draws, d) 的均匀样本；
        ``tail_dependence`` 下尾相依系数
            λ = 2*t_df( -sqrt((df+1)(1-ρ)/(1+ρ)) )，
            其中 ρ 取相关矩阵非对角元的平均值（d=2 时即唯一的相关系数）。

    算法:
        1. 同 ``gaussian_copula`` 估计 ρ 并做 Cholesky；
        2. 抽 Z ~ N(0, ρ) 与 W ~ χ²_df（卡方用 Gamma(df/2, 2) 抽样），
           令 T = Z / sqrt(W/df)，则 T 的每个边缘是 t_df，联合是 t copula；
        3. 逐元素取 t 分布函数（``_student_t_cdf``，用不完全 Beta 实现）得到均匀样本；
        4. 尾相依系数按闭式 λ = 2*t_df(-sqrt((df+1)(1-ρ)/(1+ρ))) 直接返回
           （正尾相同，t copula 上下尾对称）。

    复杂度:
        时间 O(n d log n + d^3 + n_draws * d * max_iter) / 空间 O(n_draws * d)。

    陷阱:
        1. λ 只在 ρ>0 时非零：ρ<=0 时 t copula 也没有下尾相依（公式会给出
           t_df(-∞) 之外的值，注意 ρ=-1 时 (1-ρ)/(1+ρ) 发散，本函数对 ρ→-1 会
           返回 0，因为 sqrt 项趋于 +inf、CDF 趋于 0）。
        2. λ 是**渐近**量：n_draws=1000 时按 5% 分位点的经验估计非常不稳定，
           不要拿经验尾频去和 λ 对拍，应该用闭式公式对拍（自测就是这么做的）。
        3. 混合抽样是"同一个 W 作用于整行向量"：若每维用独立的 W，得到的联合分布
           不是 t copula（而是一个更复杂的椭球 copula）。
        4. 边缘样本是 t 分布函数变换后的均匀值，``samples`` 的每列仍严格 U(0,1)，
           不要把 t 分位数当作最终样本。

    参考:
        Embrechts, McNeil & Straumann 2002；Demarta & McNeil 2005（t copula 的
        尾相依与自由度估计）。
    """
    u = as_matrix(U, "U")
    n, d = u.shape
    if n < 3:
        raise ValueError(f"U 至少需要 3 行伪观测，得到 {n}")
    if d < 2:
        raise ValueError(f"U 至少需要 2 列，得到 {d}")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("U 必须是伪观测（每列取值在 [0, 1] 内）")
    if isinstance(df, bool) or not isinstance(df, (int, np.integer)):
        raise ValueError(f"df 必须是整数，得到 {df!r}")
    df = int(df)
    if df < 1:
        raise ValueError(f"df 必须 >= 1，得到 {df}")
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)):
        raise ValueError(f"n_draws 必须是整数，得到 {n_draws!r}")
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(f"n_draws 必须 >= 1，得到 {n_draws}")

    rho = _spearman_to_pearson(u)
    gen = rng(seed)
    chol = np.linalg.cholesky(rho)
    z = gen.standard_normal((n_draws, d)) @ chol.T
    w = gen.chisquare(df, size=n_draws)
    t = z / np.sqrt(w[:, None] / float(df))
    rho_bar = _average_offdiag_rho(rho)
    if rho_bar > -1.0 + 1e-12:
        inner = -math.sqrt((df + 1.0) * (1.0 - rho_bar) / (1.0 + rho_bar))
        tail = 2.0 * _student_t_cdf(inner, float(df))
    else:
        tail = 0.0
    return {
        "rho": rho,
        "df": df,
        "samples": _student_t_cdf_array(t, float(df)),
        "tail_dependence": float(tail),
    }


def mmck_metrics(lam: float, mu: float, c: int, K: int) -> Dict[str, object]:
    """M/M/c/K 有限容量排队系统（c 个并联服务台、系统最多容纳 K 个顾客）的稳态解析指标。

    参数:
        lam: 到达率 λ（每单位时间到达的顾客数），必须 >= 0。λ=0 时系统恒空，全部指标为 0。
        mu: 单个服务台的服务率 μ（每单位时间服务完的顾客数），必须 > 0。
        c: 服务台数，必须 >= 1（不足 K 台时空闲台不产生服务能力）。
        K: 系统总容量（含正在被服务的 c 个），必须 >= c。K=c 时是**损失制**（无等待位置），
           新到顾客直接被拒绝；K>c 时是"最多排 K-c 个人的等待制"。
        注意：这里**不要求 λ < cμ**。有限容量的出生-死亡链状态空间只有 K+1 个状态，
        即使 ρ=λ/(cμ) >= 1 稳态仍然存在（这正是加容量与不加容量的本质区别）。

    返回:
        dict，键为：
        ``rho`` 利用率 λ/(cμ)（可以 >= 1，不会发散）；
        ``p0`` 系统全空的稳态概率；
        ``pn`` 长度 K+1 的 ndarray，p_n（n=0..K）的完整稳态分布；
        ``lambda_eff`` 有效到达率 λ(1-p_K)——被拒的顾客没有真正进入系统；
        ``blocking_probability`` 阻塞（拒绝）概率 p_K；
        ``L`` 系统内平均顾客数 Σ n·p_n；
        ``Lq`` 平均等待顾客数 Σ max(n-c,0)·p_n；
        ``W`` 平均逗留时间（含服务），W = L/λ_eff；
        ``Wq`` 平均排队等待时间，Wq = Lq/λ_eff；
        ``c`` / ``K`` 原样回显的台数与容量。

    算法:
        出生-死亡过程求稳态。令 a = λ/μ，未归一化权重
        w_n = a^n / n!（n <= c）、w_n = a^n / (c! c^(n-c))（c < n <= K），
        p_n = w_n / Σ_{j=0..K} w_j。用递推 w_n = w_{n-1}·a/n（n<=c）、w_n = w_{n-1}·a/c（n>c）
        避免阶乘溢出。再由 p_n 直接求和得 L、Lq、λ_eff = λ(1-p_K)，最后套 Little 公式 L=λ_eff·W、
        Lq=λ_eff·Wq（**注意用 λ_eff 而不是 λ**，这是有限容量最容易写错的地方）。

    复杂度:
        时间 O(K) / 空间 O(K)。K 很大时（如 K>1e5）递推仍然稳定，因为每步只乘一个常数。

    陷阱:
        1. **Little 公式必须用 λ_eff**：用 λ 会让 W、Wq 偏小约 (1-p_K) 倍。λ_eff = λ(1-p_K) 才是
           真正进入系统并被服务的顾客流强度。
        2. ρ >= 1 时**不能**套用 M/M/c 的 Erlang-C 公式（那里 p_0 的分母是无限和、会发散）；
           有限容量必须走本函数的截断和。ρ >= 1 且 K 有限时 L 会随 K 近似线性增长。
        3. λ=0、μ>0 时 λ_eff=0，W=L/λ_eff 是 0/0；本函数按极限返回 W=Wq=0，不会抛 ZeroDivisionError。
        4. p_K 是"到达即被拒"的概率。若题目考察的是"顾客愿意排队但被拒绝"，直接报 p_K 即可；
           若考虑的是**复呼**（被拒后过一会儿再来），实际通过的到达率会高于 λ(1-p_K)，
           需要另建模型，不要用本函数的结果。
        5. 状态空间只有 K+1 个，p_n 是**真分布**（Σp_n=1）；M/M/c 的 p_n 是无限和截断，
           两者只在 K→∞ 时一致。

    参考:
        Erlang 1917（损失制）；Kendall 1953 记号；Gross & Harris, "Fundamentals of Queueing Theory"，
        有限容量 M/M/c/K 一节；Little 1961。
    """
    if mu <= 0:
        raise ValueError(f"mu 必须 > 0，得到 {mu}")
    if lam < 0:
        raise ValueError(f"lam 必须 >= 0，得到 {lam}")
    if int(c) != c or int(c) < 1:
        raise ValueError(f"服务台数 c 必须是 >= 1 的整数，得到 {c}")
    if int(K) != K or int(K) < int(c):
        raise ValueError(f"容量 K 必须是 >= c(={int(c)}) 的整数，得到 {K}")
    n_servers = int(c)
    capacity = int(K)

    a = lam / mu
    weights = np.empty(capacity + 1, dtype=float)
    weights[0] = 1.0
    for n in range(1, capacity + 1):
        weights[n] = weights[n - 1] * a / n if n <= n_servers else weights[n - 1] * a / n_servers
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"稳态权重之和非法（{total}），请检查 lam/mu/c/K 的量级")
    pn = weights / total
    p0 = float(pn[0])

    idx = np.arange(capacity + 1, dtype=float)
    l_sys = float(np.sum(idx * pn))
    lq = float(np.sum(np.maximum(idx - n_servers, 0.0) * pn))
    blocking = float(pn[capacity])
    lambda_eff = float(lam * (1.0 - blocking))

    if lambda_eff > 0.0:
        w_sys = l_sys / lambda_eff
        wq = lq / lambda_eff
    else:  # λ=0（或 λ>0 但容量 0 的退化情形）按极限取 0
        w_sys = 0.0
        wq = 0.0

    return {
        "rho": float(a / n_servers),
        "p0": p0,
        "pn": pn,
        "lambda_eff": lambda_eff,
        "blocking_probability": blocking,
        "L": l_sys,
        "Lq": lq,
        "W": w_sys,
        "Wq": wq,
        "c": n_servers,
        "K": capacity,
    }


def geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """几何布朗运动（GBM）的**精确解**路径模拟：dS = μS dt + σS dW。

    参数:
        s0: 初始价格/存量，必须 > 0（GBM 永远取正值；s0=0 会被吸收在 0，不是本模型）。
        mu: 年化（或与 t 同单位）漂移率，可为负。
        sigma: 波动率，必须 >= 0；sigma=0 退化为确定性指数增长 dS=μS dt。
        t: 模拟总时长（必须 > 0）。
        n_steps: 时间步数（必须 >= 1），步长 dt = t/n_steps。
        n_paths: 路径条数（必须 >= 1）。只要统计量就取 1；要估矩/风险取 >= 1e4。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``paths`` 形状 (n_paths, n_steps+1) 的价格矩阵，第 0 列恒为 s0；
        ``times`` 长度 n_steps+1 的时间网格（0..t）；
        ``log_returns`` 形状 (n_paths, n_steps) 的**逐步对数收益** ΔlnS（= drift·dt + σ√dt·Z）；
        ``terminal_mean`` / ``terminal_std`` 终值 S_T 的样本均值与样本标准差（ddof=1）；
        ``theoretical_terminal_mean`` = s0·exp(μt)（对数正态一阶矩，**不是** s0·exp((μ-σ²/2)t)）；
        ``theoretical_terminal_var`` = s0²·e^{2μt}·(e^{σ²t}-1)；
        ``theoretical_terminal_std`` 上式开方；
        ``dt`` 步长；``drift_per_step`` = (μ-σ²/2)dt；``diffusion_per_step`` = σ√dt；
        ``n_paths`` / ``n_steps`` 原样回显。

    算法:
        对 dS=μS dt+σS dW 用 Itô 公式得 lnS 服从带漂移的布朗运动，其**精确解**
        S_{t+Δ} = S_t·exp((μ-σ²/2)Δ + σ√Δ·Z)，Z~N(0,1)。因此直接抽样对数增量再累加，
        得到的是连续过程在网格点上的**精确分布**（不是离散化近似，无步长偏差）。
        这也是它比 Euler-Maruyama 更适合 GBM 的原因：Euler 有 O(√Δt) 的弱误差，
        而本函数只有蒙特卡洛误差。

    复杂度:
        时间 O(n_paths·n_steps) / 空间 O(n_paths·(n_steps+1))。

    陷阱:
        1. **均值别写错**：E[S_T]=s0·e^{μT}，而**中位数**才是 s0·e^{(μ-σ²/2)T}。波动越大两者
           差得越远（σ=0.3、T=1 时相差约 4.6%）。论文里报"预期终值"必须报均值，
           报中位数要写明，否则会被判为把对数漂移当成了算术漂移。
        2. 方差随 σ²T 指数放大：σ=0.3、T=1 时终值标准差约为均值的 31%，用 1e4 条路径时
           均值的相对标准误 ≈ 0.31/100 = 0.3%，够用；但**分位数**（如 VaR）需要更多路径，
           且尾部收敛慢。
        3. μ、σ、t 必须同单位（都用"年"或都用"月"）。σ 按年给而 t 用月是最常见的量纲错误，
           会让 σ√t 差 √12≈3.46 倍。
        4. 结果依赖 seed；论文中同时给出 seed、n_paths、n_steps。
        5. 对数收益的样本均值是 (μ-σ²/2)dt 的无偏估计，可以据此反推 μ 做参数校验，
           但**不能**直接对价格序列做算术平均来估 μ。

    参考:
        Itô 1951；Black & Scholes 1973；Hull, "Options, Futures, and Other Derivatives"，
        几何布朗运动与蒙特卡洛定价一节。
    """
    if s0 <= 0:
        raise ValueError(f"s0 必须 > 0（GBM 取正值），得到 {s0}")
    if sigma < 0:
        raise ValueError(f"sigma 必须 >= 0，得到 {sigma}")
    if t <= 0:
        raise ValueError(f"t 必须 > 0，得到 {t}")
    if int(n_steps) != n_steps or int(n_steps) < 1:
        raise ValueError(f"n_steps 必须是 >= 1 的整数，得到 {n_steps}")
    if int(n_paths) != n_paths or int(n_paths) < 1:
        raise ValueError(f"n_paths 必须是 >= 1 的整数，得到 {n_paths}")
    steps = int(n_steps)
    paths_n = int(n_paths)

    dt = t / steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    gen = rng(seed)
    z = gen.standard_normal((paths_n, steps))
    log_inc = drift + diffusion * z
    log_paths = np.concatenate([np.zeros((paths_n, 1), dtype=float), np.cumsum(log_inc, axis=1)], axis=1)
    paths = s0 * np.exp(log_paths)
    times = np.linspace(0.0, t, steps + 1)

    terminal = paths[:, -1]
    terminal_mean = float(np.mean(terminal))
    terminal_std = float(np.std(terminal, ddof=1)) if paths_n > 1 else 0.0
    theory_mean = float(s0 * math.exp(mu * t))
    theory_var = float(s0 * s0 * math.exp(2.0 * mu * t) * (math.exp(sigma * sigma * t) - 1.0))

    return {
        "paths": paths,
        "times": times,
        "log_returns": np.diff(log_paths, axis=1) if steps > 0 else np.zeros((paths_n, 0)),
        "terminal_mean": terminal_mean,
        "terminal_std": terminal_std,
        "theoretical_terminal_mean": theory_mean,
        "theoretical_terminal_var": theory_var,
        "theoretical_terminal_std": float(math.sqrt(theory_var)),
        "dt": float(dt),
        "drift_per_step": float(drift),
        "diffusion_per_step": float(diffusion),
        "n_paths": paths_n,
        "n_steps": steps,
    }


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict（int/float），固定种子下两次调用完全一致：
        ``mc_pi_est`` / ``mc_pi_stderr`` / ``mc_pi_lo`` / ``mc_pi_hi`` π 的蒙特卡洛估计与区间；
        ``mc_int_est`` 对 ∫_0^1 x² dx = 1/3 的估计；
        ``mm1_rho`` / ``mm1_Lq`` / ``mm1_Wq`` λ=4、μ=5 的解析指标；
        ``mm1_sim_wait`` / ``mm1_sim_queue_len`` 同参数下 20000 个顾客的仿真值；
        ``markov_pi0`` 两状态链的平稳概率、``markov_residual`` 实测 max|πP-π|；
        ``absorb_prob`` 赌徒破产型吸收链的吸收概率、``absorb_steps`` 期望步数；
        ``ruin_sim`` / ``ruin_theory`` 赌徒破产的模拟值与解析值；
        ``mmck_*`` M/M/1/2 手算闭式解、K→∞ 收敛到 M/M/1 的偏差、Erlang-B 阻塞率对拍、
        Little 定律与分布归一化残差、ρ>1 时阻塞率单调性；
        ``gbm_*`` GBM 终值均值/标准差及与解析解的相对误差、对数收益矩误差、中位数误差、
        σ=0 退化值与同种子复现最大偏差。

    算法:
        固定种子 2024（π 用 n=200000）、排队仿真 20000 个顾客、破产问题 p=0.49、
        start=10、target=20、100000 次模拟。

    复杂度:
        时间 O(1e6) / 空间 O(1e5)。

    陷阱:
        自测里的随机量都用了较大的样本量，否则"估计值接近真值"这类断言的容忍度必须放得很宽；
        但排队仿真的等待时间即使 n=20000 也仍会与解析值差 10%~30%，这是仿真本身的性质。

    参考:
        本模块各函数参考文献。
    """
    pi_est = mc_pi(200000, seed=2024)
    integ = mc_integrate(lambda u: u ** 2, 0.0, 1.0, 200000, seed=2024)

    m = mm1_metrics(4.0, 5.0)
    sim = mm1_simulate(4.0, 5.0, 20000, seed=2024)

    p_two = np.array([[0.7, 0.3], [0.4, 0.6]])
    pi_vec = markov_steady_state(p_two)
    resid = float(np.max(np.abs(pi_vec @ p_two - pi_vec)))

    # 吸收链：状态 0,1,2 为瞬态，3、4 为吸收态（对应"以概率 1 被吸收"的标准算例）
    p_abs = np.array([
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.2, 0.0, 0.3, 0.0, 0.5],
        [0.0, 0.4, 0.0, 0.0, 0.6],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    abs_res = markov_absorption(p_abs, [0, 1, 2])

    ruins = gamblers_ruin(0.49, 10, 20, seed=2024, n_trials=100000)

    # ---- 以下为追加的自测：M/M/c、M/G/1、离散事件仿真、MCMC、Copula ----
    # 断言 1（最强一致性）：c=1 的 M/M/c 闭式解必须逐项等于 M/M/1 的解析解
    mmc1 = mmc_metrics(4.0, 5.0, 1)
    _pairs = (("rho", "rho"), ("p0", "P0"), ("L", "L"), ("Lq", "Lq"), ("W", "W"), ("Wq", "Wq"))
    mmc_c1_max_diff = max(abs(float(mmc1[k_new]) - float(m[k_old])) for k_new, k_old in _pairs)
    if mmc_c1_max_diff > 1e-9:
        raise AssertionError(f"M/M/c(c=1) 与 M/M/1 的最大偏差 {mmc_c1_max_diff} 超过 1e-9")

    # 断言 2：Erlang-C 手算对拍（c=2, lam=3, mu=2）。手算：p0=1/7，C=9/14≈0.642857
    mmc2 = mmc_metrics(3.0, 2.0, 2)
    erlang_c_hand = 9.0 / 14.0
    # 独立的第二条路线：Erlang-B 阻塞率递推 + C = B/(1-rho(1-B))，与上式推导完全不同
    b_loss = 1.0
    a_load = 3.0 / 2.0
    for n_load in range(1, 3):
        b_loss = a_load * b_loss / (n_load + a_load * b_loss)
    erlang_c_recursion = b_loss / (1.0 - 0.75 * (1.0 - b_loss))
    if abs(float(mmc2["p_wait"]) - erlang_c_hand) > 1e-12:
        raise AssertionError(f"M/M/2 的 p_wait={mmc2['p_wait']} 与手算值 {erlang_c_hand} 不符")
    if abs(erlang_c_recursion - erlang_c_hand) > 1e-12:
        raise AssertionError(f"Erlang-B 递推给出的 C={erlang_c_recursion} 与手算 {erlang_c_hand} 不符")
    if abs(float(mmc2["Wq"]) - erlang_c_hand) > 1e-12:
        raise AssertionError(f"M/M/2 的 Wq 应等于 C/(c*mu-lam)=C={erlang_c_hand}，实际 {mmc2['Wq']}")

    # 断言 3：指数服务时间下 M/G/1 的 P-K 公式必须退化成 M/M/1（Var=1/mu^2）
    mg1 = mg1_metrics(4.0, 0.2, 0.04)
    if abs(float(mg1["Wq"]) - float(m["Wq"])) > 1e-12:
        raise AssertionError(f"M/G/1(指数服务) Wq={mg1['Wq']} 应等于 M/M/1 的 {m['Wq']}")
    if abs(float(mg1["Lq"]) - float(m["Lq"])) > 1e-12:
        raise AssertionError(f"M/G/1(指数服务) Lq={mg1['Lq']} 应等于 M/M/1 的 {m['Lq']}")

    # 断言 4：常量到达/服务的 D/D/1 不应有任何等待，逗留时间精确等于服务时长
    dd = discrete_event_simulation(lambda g: 1.0, lambda g: 0.5, 1000, n_servers=1, seed=11)
    if int(dd["n_served"]) != 1000:
        raise AssertionError(f"D/D/1 应服务完 1000 人，实际 {dd['n_served']}")
    if float(np.max(dd["wait"])) > 1e-12:
        raise AssertionError(f"D/D/1 不应有等待，实际最大等待 {float(np.max(dd['wait']))}")
    if abs(float(np.mean(dd["sojourn"])) - 0.5) > 1e-12:
        raise AssertionError(f"D/D/1 逗留时间应为 0.5，实际 {float(np.mean(dd['sojourn']))}")

    # 断言 5：M/M/2 仿真（n=20000）的 Wq 与 Erlang-C 闭式解相对误差 < 15%
    closed_wq = float(mmc2["Wq"])
    mmc_sim = mmc_simulate(3.0, 2.0, 2, n_customers=20000, seed=7)
    if int(mmc_sim["n_served"]) != 20000:
        raise AssertionError(f"M/M/2 仿真应服务完 20000 人，实际 {mmc_sim['n_served']}")
    mmc_sim_rel_err = abs(float(mmc_sim["Wq"]) - closed_wq) / closed_wq
    if mmc_sim_rel_err > 0.15:
        raise AssertionError(
            f"M/M/2 仿真 Wq={mmc_sim['Wq']} 与闭式解 {closed_wq} 的相对误差 "
            f"{mmc_sim_rel_err:.3%} 超过 15%"
        )

    # 断言 6：Metropolis-Hastings 对标准正态目标（log_target = -x^2/2）的矩与接受率
    mh = metropolis_hastings(
        lambda x: -0.5 * float(np.sum(x * x)), [0.0],
        n_samples=5000, proposal_sd=1.0, seed=2024, burn_in=1000,
    )
    mh_mean = float(mh["mean"][0])
    mh_var = float(mh["var"][0])
    mh_accept = float(mh["accept_rate"])
    if abs(mh_mean) > 0.1:
        raise AssertionError(f"MH 对标准正态的样本均值 {mh_mean} 偏离 0 超过 0.1")
    if not 0.8 < mh_var < 1.25:
        raise AssertionError(f"MH 对标准正态的样本方差 {mh_var} 不在 (0.8, 1.25)")
    if not 0.2 < mh_accept < 0.8:
        raise AssertionError(f"MH 接受率 {mh_accept} 不在 (0.2, 0.8)")

    # 断言 7：Gibbs 采样的样本协方差与目标协方差逐项误差 < 0.1
    target_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    gibbs = gibbs_sampler_bivariate_normal([0.0, 0.0], target_cov, n_samples=5000, seed=2024, burn_in=1000)
    gibbs_max_cov_err = float(np.max(np.abs(np.asarray(gibbs["cov"]) - target_cov)))
    if gibbs_max_cov_err > 0.1:
        raise AssertionError(f"Gibbs 样本协方差与目标的最大偏差 {gibbs_max_cov_err} 超过 0.1")

    # 断言 8：copula。用已知 Pearson rho=0.5 的高斯 copula 数据反解，再对拍 t 尾相依闭式
    gen_u = rng(2024)
    chol_true = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    u_obs = _norm_cdf(gen_u.standard_normal((1500, 2)) @ chol_true.T)
    gcop = gaussian_copula(u_obs, n_draws=1000, seed=2024)
    gc_rho = float(np.asarray(gcop["rho"])[0, 1])
    if abs(gc_rho - 0.5) > 0.05:
        raise AssertionError(f"高斯 copula 反解的 rho={gc_rho} 偏离真值 0.5 超过 0.05")
    gcop_tail = float(gcop["tail_dependence"])
    if gcop_tail != 0.0:
        raise AssertionError(f"高斯 copula 的尾相依必须为 0，实际 {gcop_tail}")
    # t 分布函数本身的独立校验：df=1 是柯西（P(T<=1)=3/4），df=2 有闭式
    if abs(_student_t_cdf(1.0, 1.0) - 0.75) > 1e-12:
        raise AssertionError(f"t_cdf(1, df=1)={_student_t_cdf(1.0, 1.0)} 应为 0.75（柯西分布）")
    t_cdf_df2 = 0.5 - 1.0 / (2.0 * math.sqrt(3.0))
    if abs(_student_t_cdf(-1.0, 2.0) - t_cdf_df2) > 1e-12:
        raise AssertionError(f"t_cdf(-1, df=2)={_student_t_cdf(-1.0, 2.0)} 应为 {t_cdf_df2}")
    tcop = t_copula(u_obs, df=5, n_draws=1000, seed=2024)
    tcop_rho = float(np.asarray(tcop["rho"])[0, 1])
    tcop_tail = float(tcop["tail_dependence"])
    tcop_tail_closed = 2.0 * _student_t_cdf(
        -math.sqrt((5.0 + 1.0) * (1.0 - tcop_rho) / (1.0 + tcop_rho)), 5.0
    )
    if abs(tcop_tail - tcop_tail_closed) > 1e-12:
        raise AssertionError(f"t copula 尾相依 {tcop_tail} 与闭式 {tcop_tail_closed} 不符")
    if not tcop_tail > 0.05:
        raise AssertionError(f"df=5、rho≈0.5 的 t copula 应有正的尾相依，实际 {tcop_tail}")
    # df→∞ 时 t copula 必须退化为高斯 copula（尾相依 → 0）
    tcop_tail_large_df = 2.0 * _student_t_cdf(
        -math.sqrt((1000.0 + 1.0) * (1.0 - tcop_rho) / (1.0 + tcop_rho)), 1000.0
    )
    if not 0.0 <= tcop_tail_large_df < 0.01:
        raise AssertionError(f"df=1000 的 t copula 尾相依应趋于 0，实际 {tcop_tail_large_df}")

    # 断言 9：M/M/1/2（λ=1, μ=2, K=2）的**手算闭式解**。ρ=0.5，
    # p0=1/(1+0.5+0.25)=4/7，p1=2/7，p2=1/7；λ_eff=1·(1-p2)=6/7；
    # L=0·4/7+1·2/7+2·1/7=4/7，Lq=(2-1)·1/7=1/7，W=L/λ_eff=2/3，Wq=1/6。
    mm1k = mmck_metrics(1.0, 2.0, 1, 2)
    hand = {
        "p0": 4.0 / 7.0,
        "L": 4.0 / 7.0,
        "Lq": 1.0 / 7.0,
        "W": 2.0 / 3.0,
        "Wq": 1.0 / 6.0,
        "blocking_probability": 1.0 / 7.0,
        "lambda_eff": 6.0 / 7.0,
    }
    mmck_hand_max_diff = max(abs(float(mm1k[k]) - v) for k, v in hand.items())
    if mmck_hand_max_diff > 1e-12:
        raise AssertionError(f"M/M/1/2 与手算闭式解的最大偏差 {mmck_hand_max_diff} 超过 1e-12")

    # 断言 10：K→∞ 时 M/M/c/K（c=1）必须收敛到 M/M/1 的解析解（ρ=0.8 < 1）
    mm1k_big = mmck_metrics(4.0, 5.0, 1, 200)
    mmck_limit_max_diff = max(
        abs(float(mm1k_big[key]) - float(m[key])) for key in ("L", "Lq", "W", "Wq")
    )
    if mmck_limit_max_diff > 1e-9:
        raise AssertionError(f"K=200 的 M/M/1/K 与 M/M/1 的偏差 {mmck_limit_max_diff} 超过 1e-9")

    # 断言 11：K=c 的纯损失制必须等于 Erlang-B 递推（独立路线：B1=1.5/2.5=0.6，B2=0.9/2.9）
    erlang_b_loss = 1.0
    for n_load in range(1, 3):
        erlang_b_loss = 1.5 * erlang_b_loss / (n_load + 1.5 * erlang_b_loss)
    mmck_loss = mmck_metrics(3.0, 2.0, 2, 2)
    if abs(float(mmck_loss["blocking_probability"]) - erlang_b_loss) > 1e-12:
        raise AssertionError(
            f"M/M/2/2 阻塞率 {mmck_loss['blocking_probability']} 应等于 Erlang-B {erlang_b_loss}"
        )

    # 断言 12：Little 定律 L=λ_eff·W、Lq=λ_eff·Wq 与分布归一化，跨 4 组参数（含 ρ>1、λ=0）
    little_resid = 0.0
    pn_resid = 0.0
    for (lam_i, mu_i, c_i, k_i) in ((4.0, 5.0, 2, 6), (6.0, 2.0, 2, 8), (10.0, 3.0, 3, 3), (0.0, 2.0, 1, 4)):
        res_i = mmck_metrics(lam_i, mu_i, c_i, k_i)
        pn_resid = max(pn_resid, abs(float(np.sum(np.asarray(res_i["pn"]))) - 1.0))
        little_resid = max(little_resid, abs(float(res_i["L"]) - float(res_i["lambda_eff"]) * float(res_i["W"])))
        little_resid = max(little_resid, abs(float(res_i["Lq"]) - float(res_i["lambda_eff"]) * float(res_i["Wq"])))
    if pn_resid > 1e-12:
        raise AssertionError(f"M/M/c/K 稳态分布之和偏离 1：{pn_resid}")
    if little_resid > 1e-12:
        raise AssertionError(f"Little 定律残差 {little_resid} 超过 1e-12")
    # ρ=1.5 > 1 时容量越大阻塞率越低（单调性）
    if not float(mmck_metrics(6.0, 2.0, 2, 8)["blocking_probability"]) > float(
        mmck_metrics(6.0, 2.0, 2, 9)["blocking_probability"]
    ):
        raise AssertionError("ρ>1 时阻塞率应随容量 K 增大而下降")

    # 断言 13：GBM 的终值矩与对数收益矩对上解析解（20000 条路径 × 252 步）
    gbm = geometric_brownian_motion(100.0, 0.08, 0.2, 1.0, 252, n_paths=20000, seed=2024)
    gbm_paths = np.asarray(gbm["paths"])
    gbm_mean_rel_err = abs(float(gbm["terminal_mean"]) - float(gbm["theoretical_terminal_mean"])) / float(
        gbm["theoretical_terminal_mean"]
    )
    if float(gbm["theoretical_terminal_std"]) <= 0:
        raise AssertionError("GBM 理论标准差应 > 0")
    gbm_std_rel_err = abs(float(gbm["terminal_std"]) - float(gbm["theoretical_terminal_std"])) / float(
        gbm["theoretical_terminal_std"]
    )
    if gbm_mean_rel_err > 0.05:
        raise AssertionError(f"GBM 终值均值相对误差 {gbm_mean_rel_err:.3%} 超过 5%")
    if gbm_std_rel_err > 0.08:
        raise AssertionError(f"GBM 终值标准差相对误差 {gbm_std_rel_err:.3%} 超过 8%")
    if float(np.max(np.abs(gbm_paths[:, 0] - 100.0))) != 0.0:
        raise AssertionError("GBM 路径第 0 列必须精确等于 s0")
    log_ret = np.asarray(gbm["log_returns"])
    gbm_logret_mean_err = abs(float(np.mean(log_ret)) - float(gbm["drift_per_step"])) / abs(
        float(gbm["drift_per_step"])
    )
    gbm_logret_std_err = abs(float(np.std(log_ret)) - float(gbm["diffusion_per_step"])) / float(
        gbm["diffusion_per_step"]
    )
    if gbm_logret_mean_err > 0.10:
        raise AssertionError(f"GBM 对数收益均值偏离 (μ-σ²/2)dt 达 {gbm_logret_mean_err:.3%}")
    if gbm_logret_std_err > 0.02:
        raise AssertionError(f"GBM 对数收益标准差偏离 σ√dt 达 {gbm_logret_std_err:.3%}")
    # 终值中位数应贴近 s0·exp((μ-σ²/2)t)，与均值 s0·exp(μt) 明确区分
    gbm_median_theory = 100.0 * math.exp((0.08 - 0.5 * 0.2 * 0.2) * 1.0)
    gbm_median_rel_err = abs(float(np.median(gbm_paths[:, -1])) - gbm_median_theory) / gbm_median_theory
    if gbm_median_rel_err > 0.02:
        raise AssertionError(f"GBM 终值中位数相对误差 {gbm_median_rel_err:.3%} 超过 2%")
    # σ=0 时必须退化为确定性指数增长；同种子必须逐元素可复现
    gbm_det = geometric_brownian_motion(100.0, 0.08, 0.0, 1.0, 252, n_paths=5, seed=2024)
    gbm_det_terminal = float(np.max(np.asarray(gbm_det["paths"])[:, -1]))
    if abs(gbm_det_terminal - 100.0 * math.exp(0.08)) > 1e-9:
        raise AssertionError(f"σ=0 时终值应为 {100.0 * math.exp(0.08)}，实际 {gbm_det_terminal}")
    gbm_again = geometric_brownian_motion(100.0, 0.08, 0.2, 1.0, 252, n_paths=20000, seed=2024)
    gbm_repro_max_diff = float(np.max(np.abs(np.asarray(gbm_again["paths"]) - gbm_paths)))
    if gbm_repro_max_diff != 0.0:
        raise AssertionError(f"同种子 GBM 路径不可复现，最大偏差 {gbm_repro_max_diff}")

    return {
        "mc_pi_est": round(float(pi_est["estimate"]), 6),
        "mc_pi_stderr": round(float(pi_est["stderr"]), 6),
        "mc_pi_lo": round(float(pi_est["ci95_low"]), 6),
        "mc_pi_hi": round(float(pi_est["ci95_high"]), 6),
        "mc_int_est": round(float(integ["estimate"]), 6),
        "mm1_rho": round(float(m["rho"]), 6),
        "mm1_Lq": round(float(m["Lq"]), 6),
        "mm1_Wq": round(float(m["Wq"]), 6),
        "mm1_sim_wait": round(float(sim["avg_wait"]), 6),
        "mm1_sim_queue_len": round(float(sim["avg_queue_len"]), 6),
        "markov_pi0": round(float(pi_vec[0]), 8),
        "markov_residual": float(resid),
        "absorb_prob_0": round(float(abs_res["absorption_probs"][0, 0]), 6),
        "absorb_steps_0": round(float(abs_res["expected_steps"][0]), 6),
        "ruin_sim": round(float(ruins["ruin_prob_sim"]), 6),
        "ruin_theory": round(float(ruins["ruin_prob_theory"]), 6),
        "mmc_rho": round(float(mmc2["rho"]), 6),
        "mmc_p_wait": round(float(mmc2["p_wait"]), 6),
        "mmc_erlang_c": round(float(mmc2["erlang_c"]), 6),
        "mmc_Lq": round(float(mmc2["Lq"]), 6),
        "mmc_Wq": round(float(mmc2["Wq"]), 6),
        "mmc_W": round(float(mmc2["W"]), 6),
        "mmc_c1_max_diff": mmc_c1_max_diff,
        "mmc_erlang_c_hand": round(erlang_c_hand, 6),
        "mmc_erlang_c_recursion": round(erlang_c_recursion, 6),
        "mg1_rho": round(float(mg1["rho"]), 6),
        "mg1_Wq": round(float(mg1["Wq"]), 6),
        "mg1_Lq": round(float(mg1["Lq"]), 6),
        "des_n_served": int(dd["n_served"]),
        "des_max_wait": round(float(np.max(dd["wait"])), 12),
        "des_mean_sojourn": round(float(np.mean(dd["sojourn"])), 9),
        "des_total_busy": round(float(np.sum(dd["server_busy"])), 6),
        "mmc_sim_Wq": round(float(mmc_sim["Wq"]), 6),
        "mmc_sim_Lq": round(float(mmc_sim["Lq"]), 6),
        "mmc_sim_util": round(float(mmc_sim["utilization"]), 6),
        "mmc_sim_rel_err": round(mmc_sim_rel_err, 6),
        "mh_mean": round(mh_mean, 6),
        "mh_var": round(mh_var, 6),
        "mh_accept": round(mh_accept, 6),
        "gibbs_mean_0": round(float(gibbs["mean"][0]), 6),
        "gibbs_cov_00": round(float(gibbs["cov"][0, 0]), 6),
        "gibbs_cov_01": round(float(gibbs["cov"][0, 1]), 6),
        "gibbs_max_cov_err": round(gibbs_max_cov_err, 6),
        "gcop_rho": round(gc_rho, 6),
        "gcop_tail": round(gcop_tail, 6),
        "tcop_df": int(tcop["df"]),
        "tcop_rho": round(tcop_rho, 6),
        "tcop_tail": round(tcop_tail, 8),
        "tcop_tail_closed": round(tcop_tail_closed, 8),
        "mmck_p0_hand": round(float(mm1k["p0"]), 12),
        "mmck_L_hand": round(float(mm1k["L"]), 12),
        "mmck_Lq_hand": round(float(mm1k["Lq"]), 12),
        "mmck_W_hand": round(float(mm1k["W"]), 12),
        "mmck_Wq_hand": round(float(mm1k["Wq"]), 12),
        "mmck_blocking_hand": round(float(mm1k["blocking_probability"]), 12),
        "mmck_lambda_eff_hand": round(float(mm1k["lambda_eff"]), 12),
        "mmck_hand_max_diff": mmck_hand_max_diff,
        "mmck_limit_max_diff": mmck_limit_max_diff,
        "mmck_loss_blocking": round(float(mmck_loss["blocking_probability"]), 12),
        "mmck_erlang_b": round(float(erlang_b_loss), 12),
        "mmck_pn_resid": pn_resid,
        "mmck_little_resid": little_resid,
        "mmck_rho_gt1_L": round(float(mmck_metrics(6.0, 2.0, 2, 8)["L"]), 9),
        "mmck_rho_gt1_blocking8": round(float(mmck_metrics(6.0, 2.0, 2, 8)["blocking_probability"]), 9),
        "mmck_rho_gt1_blocking9": round(float(mmck_metrics(6.0, 2.0, 2, 9)["blocking_probability"]), 9),
        "gbm_terminal_mean": round(float(gbm["terminal_mean"]), 6),
        "gbm_terminal_std": round(float(gbm["terminal_std"]), 6),
        "gbm_theory_mean": round(float(gbm["theoretical_terminal_mean"]), 6),
        "gbm_theory_std": round(float(gbm["theoretical_terminal_std"]), 6),
        "gbm_mean_rel_err": round(gbm_mean_rel_err, 8),
        "gbm_std_rel_err": round(gbm_std_rel_err, 8),
        "gbm_logret_mean_err": round(gbm_logret_mean_err, 8),
        "gbm_logret_std_err": round(gbm_logret_std_err, 8),
        "gbm_median_rel_err": round(gbm_median_rel_err, 8),
        "gbm_det_terminal": round(gbm_det_terminal, 9),
        "gbm_repro_max_diff": gbm_repro_max_diff,
    }
