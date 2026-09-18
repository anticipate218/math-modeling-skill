"""随机模型：蒙特卡洛估计与置信区间、M/M/1 排队论、马尔可夫链稳态与吸收、赌徒破产。

全部实现只依赖 numpy 与标准库，随机性统一走 ``_common.rng``（显式种子，绝不用
``np.random`` 的全局状态），因此同一份代码两次运行结果完全一致——竞赛论文里
"结果可复现"是硬要求，随机种子必须写进正文。

核心提醒：蒙特卡洛给出的永远是**估计 + 误差**，只报一个点估计是评审常见的扣分点。
本模块每个随机估计都同时返回标准误与 95% 置信区间。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from ._common import as_matrix, check_square, rng

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
        ``ruin_sim`` / ``ruin_theory`` 赌徒破产的模拟值与解析值。

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
    }
