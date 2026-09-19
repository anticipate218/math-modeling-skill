"""常微分方程初值问题：Euler/RK4/RK45 积分、隐式 Euler、SIR/SEIR 传染病模型与基本再生数、
logistic 增长与 logistic 映射、Lotka-Volterra 捕食者-被捕食者、SIR 参数最小二乘拟合、
收敛阶估计与 Jacobian 稳定性。

本模块共 15 个公开函数，按用途分成四组：
- 通用积分器：``solve_ivp_euler`` / ``solve_ivp_rk4`` / ``solve_ivp_rk45`` / ``implicit_euler``；
- 机理右端项与离散映射：``sir_rhs`` / ``seir_rhs`` / ``lotka_volterra_rhs`` / ``logistic_growth`` / ``logistic_map``；
- 机理仿真与派生量：``simulate_sir`` / ``simulate_seir`` / ``basic_reproduction_number``；
- 参数辨识与数值诊断：``fit_sir_least_squares`` / ``estimate_convergence_order`` / ``jacobian_stability``。

这些都是**教学透明版**：定步长、显式格式、几行就能看懂的一阶/四阶方法。
刚性方程、长时间积分、事件检测（如阈值触发）请换成熟求解器（scipy.integrate），
本模块的价值在于让论文能写清楚"我们用什么格式、步长多少、误差怎么估"。

通用约定
--------
- 初值问题写作 ``dy/dt = f(t, y)``，``y`` 可以是标量也可以是一维数组。
- ``t_span`` 是 ``(t0, t1)`` 二元组，要求 ``t1 > t0``。
- 步长 ``h`` 是**建议值**：实际会取均匀步数 ``n = ceil((t1-t0)/h)``，
  真步长为 ``(t1-t0)/n``，保证最后一个点正好落在 ``t1``。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_vector, check_same_length

ArrayLike = Union[Sequence[float], np.ndarray]
RhsFn = Callable[[float, np.ndarray], ArrayLike]

__all__ = [
    "solve_ivp_euler",
    "solve_ivp_rk4",
    "sir_rhs",
    "simulate_sir",
    "logistic_growth",
    "lotka_volterra_rhs",
    "fit_sir_least_squares",
    "estimate_convergence_order",
    "seir_rhs",
    "simulate_seir",
    "basic_reproduction_number",
    "implicit_euler",
    "solve_ivp_rk45",
    "jacobian_stability",
    "logistic_map",
]


def _grid(t_span: Tuple[float, float], h: float) -> np.ndarray:
    """内部工具：按建议步长 h 生成均匀时间网格，末点严格等于 t1。"""
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError(f"t_span 必须满足 t1 > t0，得到 {t_span}")
    if h <= 0:
        raise ValueError(f"步长 h 必须 > 0，得到 {h}")
    n = int(math.ceil((t1 - t0) / h))
    return np.linspace(t0, t1, n + 1)


def solve_ivp_euler(f: RhsFn, y0: ArrayLike, t_span: Tuple[float, float], h: float) -> Tuple[np.ndarray, np.ndarray]:
    """显式（前向）Euler 法：y_{n+1} = y_n + h f(t_n, y_n)。

    参数:
        f: 右端函数 ``f(t, y) -> dy/dt``，y 是一维数组。
        y0: 初值，长度 m 的一维数组。
        t_span: ``(t0, t1)``。
        h: 建议步长（见模块说明，实际步长会被微调以正好落在 t1）。

    返回:
        ``(t, Y)``：``t`` 形状 (n+1,)，``Y`` 形状 (n+1, m)，第 i 行是 t[i] 处的解。

    算法:
        一阶显式格式，每个时间步只调用一次 f；局部截断误差 O(h^2)，整体 O(h)。

    复杂度:
        时间 O(n m)（n 为步数，m 为状态维数）/ 空间 O(n m)。

    陷阱:
        1. **只有一阶精度**：把 h 减半误差只减半；算传染病峰值时 h=0.01 才勉强够用，
           默认拿它当"够准"是常见误判。
        2. **数值不稳定**：对 y' = -100y 这类快变问题，h > 2/100 时解会振荡发散，
           这不是 bug 而是显式 Euler 的稳定域限制。
        3. 状态变量可能变成负数（如 SIR 的 S 被减成负数），本函数不做裁剪，
           需要非负约束请换格式或在模型层面处理。

    参考:
        Euler 1768；任何数值分析教材（如 Burden & Faires）的初值问题章节。
    """
    y = as_vector(y0, "y0").copy()
    t = _grid(t_span, h)
    dt = float(t[1] - t[0])
    out = np.empty((t.size, y.size), dtype=float)
    out[0] = y
    for i in range(t.size - 1):
        y = y + dt * as_vector(f(float(t[i]), y), "f 返回值")
        out[i + 1] = y
    return t, out


def solve_ivp_rk4(f: RhsFn, y0: ArrayLike, t_span: Tuple[float, float], h: float) -> Tuple[np.ndarray, np.ndarray]:
    """经典四阶 Runge-Kutta 法（RK4），每步四次右端求值。

    参数:
        f: 右端函数 ``f(t, y) -> dy/dt``。
        y0: 初值，长度 m 的一维数组。
        t_span: ``(t0, t1)``。
        h: 建议步长。

    返回:
        ``(t, Y)``：``t`` 形状 (n+1,)，``Y`` 形状 (n+1, m)。

    算法:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h k1/2)
        k3 = f(t + h/2, y + h k2/2)
        k4 = f(t + h, y + h k3)
        y_{n+1} = y + h (k1 + 2k2 + 2k3 + k4) / 6

    复杂度:
        时间 O(4 n m) / 空间 O(n m)。

    陷阱:
        1. 四次求值的代价换四阶精度：步长减半误差约降 1/16（见
           ``estimate_convergence_order`` 的实测值）。但**定步长**在解变化剧烈的区间
           （如 SIR 爆发期）仍会失准，必要时改自适应步长。
        2. 步长取得比特征时间尺度大很多时，RK4 同样会给出看起来"平滑但错误"的解，
           务必做步长收敛测试（h、h/2、h/4 结果应基本重合）。
        3. 不要拿 RK4 去解刚性方程，稳定域有限。

    参考:
        Kutta 1901；Runge 1895；Burden & Faires《Numerical Analysis》。
    """
    y = as_vector(y0, "y0").copy()
    t = _grid(t_span, h)
    dt = float(t[1] - t[0])
    out = np.empty((t.size, y.size), dtype=float)
    out[0] = y
    for i in range(t.size - 1):
        ti = float(t[i])
        k1 = as_vector(f(ti, y), "f 返回值")
        k2 = as_vector(f(ti + 0.5 * dt, y + 0.5 * dt * k1), "f 返回值")
        k3 = as_vector(f(ti + 0.5 * dt, y + 0.5 * dt * k2), "f 返回值")
        k4 = as_vector(f(ti + dt, y + dt * k3), "f 返回值")
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = y
    return t, out


def sir_rhs(beta: float, gamma: float) -> RhsFn:
    """返回 SIR 模型的右端函数，y = [S, I, R]。

    参数:
        beta: 传染率（单位时间每个感染者有效接触并传染的人数比例）。
        gamma: 恢复率，1/gamma 是平均感染期。

    返回:
        callable ``f(t, y) -> [dS/dt, dI/dt, dR/dt]``，其中
        dS/dt = -beta S I / N，dI/dt = beta S I / N - gamma I，dR/dt = gamma I，
        总人口 N = S + I + R 由状态向量自身求和得到（守恒量不显式传入）。

    算法:
        标准 SIR 常微分方程组（Kermack-McKendrick 1927），此处不做任何离散化。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. 分母用当前状态的 S+I+R，如果状态被数值误差推到负值，N 会变小甚至为 0；
           ``safe`` 保护只能挡住 0，挡不住物理上无意义的负值。
        2. ``gamma=0`` 时模型退化为纯 SI（无人恢复），感染会单调增到全人口，
           这时"峰值时间"永远落在终点附近——不是程序出错。
        3. 这里假设人口封闭、无出生死亡、无潜伏期；要建模隔离/疫苗需扩展为 SEIR
           或在 beta 上加时变项。

    参考:
        Kermack & McKendrick, "A contribution to the mathematical theory of epidemics",
        Proc. R. Soc. A, 1927。
    """
    b = float(beta)
    g = float(gamma)
    if b < 0 or g < 0:
        raise ValueError(f"beta 与 gamma 必须非负，得到 beta={beta}, gamma={gamma}")

    def f(t: float, y: ArrayLike) -> np.ndarray:
        s, i, r = as_vector(y, "y")[:3]
        n = s + i + r
        if n <= 0:
            return np.zeros(3, dtype=float)
        new_inf = b * s * i / n
        return np.array([-new_inf, new_inf - g * i, g * i], dtype=float)

    return f


def simulate_sir(
    S0: float,
    I0: float,
    R0: float,
    beta: float,
    gamma: float,
    T: float,
    h: float = 0.01,
) -> Dict[str, object]:
    """用 RK4 数值求解 SIR，并给出峰值与最终规模等常用指标。

    参数:
        S0, I0, R0: 初始易感/感染/移除人数（非负，且总数 > 0）。
        beta: 传染率。
        gamma: 恢复率。
        T: 模拟时长（从 t=0 积到 t=T）。
        h: 步长，默认 0.01（时间单位通常取"天"，所以这是 0.01 天）。

    返回:
        dict，键为：
        ``t`` 形状 (n+1,) 时间网格；
        ``S``/``I``/``R`` 各 (n+1,) 的轨迹；
        ``peak_infected`` float，感染者峰值；
        ``peak_time`` float，峰值出现时间（网格上第一个取到峰值的时刻）；
        ``final_size`` float，最终累计感染比例 R(T)/N。

    算法:
        拼装 ``sir_rhs``，用 ``solve_ivp_rk4`` 积分，然后 np.argmax 找峰值。

    复杂度:
        时间 O(T/h) / 空间 O(T/h)。

    陷阱:
        1. **峰值是网格上的峰**：h=0.01 时峰值时间有约 ±0.01 的分辨率误差，
           论文报"第 23.4 天达峰"时不要给不必要的小数位。
        2. ``final_size`` 是 **T 时刻**的 R/N，不是真正的 R(∞)/N；T 太小时会明显低估。
           若需要极限值，应解超越方程 R∞ = 1 - exp(-beta/gamma * R∞)（见模块自测）。
        3. 参数含义依赖时间单位：beta=0.3 表示"每天"还是"每 0.1 天"，会让 R0 差 10 倍，
           必须和 T、h 保持一致。

    参考:
        Kermack & McKendrick 1927；SIR 数值实验标准做法。
    """
    if S0 < 0 or I0 < 0 or R0 < 0:
        raise ValueError("S0/I0/R0 必须非负")
    n_total = float(S0 + I0 + R0)
    if n_total <= 0:
        raise ValueError("总人口 S0+I0+R0 必须 > 0")
    f = sir_rhs(beta, gamma)
    t, y = solve_ivp_rk4(f, np.array([S0, I0, R0], dtype=float), (0.0, float(T)), h)
    s_arr, i_arr, r_arr = y[:, 0], y[:, 1], y[:, 2]
    peak_idx = int(np.argmax(i_arr))
    return {
        "t": t,
        "S": s_arr,
        "I": i_arr,
        "R": r_arr,
        "peak_infected": float(i_arr[peak_idx]),
        "peak_time": float(t[peak_idx]),
        "final_size": float(r_arr[-1] / n_total),
    }


def logistic_growth(r: float, K: float, y0: float, T: float, h: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """logistic 增长模型 y' = r y (1 - y/K)，返回数值解与解析解对照所需的时间网格。

    参数:
        r: 内禀增长率。
        K: 环境容纳量（必须 > 0）。
        y0: 初值（必须 > 0）。
        T: 模拟时长。
        h: 步长。

    返回:
        ``(t, y)``：``t`` 形状 (n+1,)，``y`` 形状 (n+1,)，数值解（RK4）。

    算法:
        用 RK4 积分；解析解为 y(t) = K / (1 + (K/y0 - 1) e^{-r t})，
        可在验证时直接对比（见 ``_self_test``）。

    复杂度:
        时间 O(T/h) / 空间 O(T/h)。

    陷阱:
        1. y0 <= 0 时解析解无意义（对数发散），本函数直接抛 ValueError。
        2. y0 > K 时解单调下降趋近 K，不会出现"负增长到 0 以下"，
           但若用显式 Euler 且 h 过大，会数值过冲到负值再跳到 K——这是格式问题不是模型问题。
        3. 参数 r 的单位是 1/时间，和 T、h 的单位必须一致。

    参考:
        Verhulst 1838；Pearl & Reed 1920 对美国人口的应用。
    """
    if K <= 0:
        raise ValueError(f"K 必须 > 0，得到 {K}")
    if y0 <= 0:
        raise ValueError(f"y0 必须 > 0，得到 {y0}")

    def f(t: float, y: ArrayLike) -> np.ndarray:
        val = float(as_vector(y, "y")[0])
        return np.array([r * val * (1.0 - val / K)], dtype=float)

    t, y = solve_ivp_rk4(f, np.array([y0], dtype=float), (0.0, float(T)), h)
    return t, y[:, 0]


def lotka_volterra_rhs(alpha: float, beta: float, delta: float, gamma: float) -> RhsFn:
    """返回 Lotka-Volterra 捕食者-被捕食者模型的右端函数，y = [x, y]。

    参数:
        alpha: 被捕食者的内禀增长率（x' 中的 +alpha x）。
        beta: 捕食导致的被捕食者死亡率系数。
        delta: 捕食转化为捕食者增长的效率。
        gamma: 捕食者的自然死亡率。

    返回:
        callable ``f(t, y) -> [alpha x - beta x y, delta x y - gamma y]``。

    算法:
        经典 Lotka-Volterra 方程（Lotka 1925；Volterra 1926）。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. 该模型**结构不稳定**：闭环轨道是中性稳定的，RK4 的数值耗散会让振幅缓慢
           漂移（长期积分后轨道螺旋向内或向外），这不是 bug；长时间模拟应改用辛格式
           或监控守恒量 V = delta x - gamma ln x + beta y - alpha ln y。
        2. 数值误差可能把种群推成负值，正反馈后直接爆掉；需要时对状态做截断。
        3. 参数 alpha/gamma 与 delta/beta 的量纲不同（1/时间、1/(个体·时间)），
           直接比较大小没有意义，平衡点才是可解释量：(gamma/delta, alpha/beta)。

    参考:
        Lotka, "Elements of Physical Biology", 1925；Volterra, 1926。
    """
    a, b, d, g = float(alpha), float(beta), float(delta), float(gamma)

    def f(t: float, y: ArrayLike) -> np.ndarray:
        x, yy = as_vector(y, "y")[:2]
        return np.array([a * x - b * x * yy, d * x * yy - g * yy], dtype=float)

    return f


def _sir_trajectory(
    beta: float,
    gamma: float,
    s0: float,
    i0: float,
    r0: float,
    t_end: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """内部工具：纯标量 RK4 的 SIR 轨迹，专供参数拟合里的上万次重复调用。

    与 ``solve_ivp_rk4`` 算法完全一致（经典 RK4、均匀步长），只是避免每步构造
    numpy 数组的开销；返回 (S, I, R) 三个长度 n_steps+1 的数组。
    """
    dt = float(t_end) / n_steps
    s, i, r = float(s0), float(i0), float(r0)
    s_out = np.empty(n_steps + 1)
    i_out = np.empty(n_steps + 1)
    r_out = np.empty(n_steps + 1)
    s_out[0], i_out[0], r_out[0] = s, i, r

    def deriv(s: float, i: float, r: float) -> Tuple[float, float, float]:
        n = s + i + r
        if n <= 0.0:
            return 0.0, 0.0, 0.0
        new_inf = beta * s * i / n
        return -new_inf, new_inf - gamma * i, gamma * i

    for step in range(n_steps):
        k1 = deriv(s, i, r)
        k2 = deriv(s + 0.5 * dt * k1[0], i + 0.5 * dt * k1[1], r + 0.5 * dt * k1[2])
        k3 = deriv(s + 0.5 * dt * k2[0], i + 0.5 * dt * k2[1], r + 0.5 * dt * k2[2])
        k4 = deriv(s + dt * k3[0], i + dt * k3[1], r + dt * k3[2])
        s += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        i += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        r += (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        s_out[step + 1], i_out[step + 1], r_out[step + 1] = s, i, r
    return s_out, i_out, r_out


def fit_sir_least_squares(
    times: ArrayLike,
    observed_I: ArrayLike,
    N: float,
    beta_bounds: Tuple[float, float],
    gamma_bounds: Tuple[float, float],
    n_grid: int = 40,
) -> Dict[str, float]:
    """用"粗网格搜索 + 逐轮局部细化"拟合 SIR 的 (beta, gamma)，不使用 scipy.optimize。

    参数:
        times: 观测时刻，长度 m 的一维数组（要求非负、递增、times[0] >= 0）。
        observed_I: 对应时刻的**现存感染者数**（不是累计），长度 m。
        N: 总人口。初始条件取 I(0)=observed_I[0]、S(0)=N-I(0)、R(0)=0。
        beta_bounds: beta 的搜索区间 ``(lo, hi)``，lo < hi。
        gamma_bounds: gamma 的搜索区间 ``(lo, hi)``。
        n_grid: 每轮每个参数的网格点数（总评估次数约为 n_grid^2 * (1 + refine_rounds)）。

    返回:
        dict，键为：
        ``beta``/``gamma`` 最优参数（float）；
        ``sse`` 最优参数下的残差平方和（对 I 的原始尺度）。

    算法:
        1. 在 [lo, hi] 上取 n_grid 个等距点组成 (beta, gamma) 网格，逐个用定步长 RK4
           积分 SIR 并在观测时刻线性插值出 I_hat，累计 SSE；
        2. 以当前最优点为中心，把搜索区间收缩为原区间的 1/4（至少保留网格间距），
           重复 refine_rounds=4 轮；共 5 轮，等效分辨率约 (1/4)^4 / (n_grid-1)。

    复杂度:
        时间 O(rounds * n_grid^2 * n_steps * m) / 空间 O(n_steps)。

    陷阱:
        1. **SSE 曲面有强相关脊**：beta 与 gamma 只通过 R0=beta/gamma 和绝对水平部分
           耦合，数据只覆盖早期上升段时两参数几乎不可辨识（多个组合 SSE 接近）。
           必须报告参数的置信区间或 SSE 等高线，不能只报一个点估计。
        2. 这里只拟合 I(t)，没有拟合累计量；若数据是**累计确诊**，要先把观测换成
           I = 累计 - 累计(t-1) 或直接对累计量建残差，否则参数会被系统性低估。
        3. 初值固定为 I(0)=observed_I[0]，若第一个观测本身有报告延迟，beta 会被带偏。
        4. 网格搜索只保证找到"网格分辨率下的最优"，不做导数，不保证全局最优。

    参考:
        模型参数反演的通用做法（grid search + local refinement）；SIR 参数辨识见
        Brauer & Castillo-Chavez, "Mathematical Models in Population Biology and
        Epidemiology", 2012。
    """
    t_obs = as_vector(times, "times")
    i_obs = as_vector(observed_I, "observed_I")
    check_same_length(t_obs, i_obs)
    if np.any(t_obs < 0):
        raise ValueError("times 必须非负")
    if np.any(np.diff(t_obs) <= 0):
        raise ValueError("times 必须严格递增")
    if N <= 0:
        raise ValueError(f"N 必须 > 0，得到 {N}")
    b_lo, b_hi = float(beta_bounds[0]), float(beta_bounds[1])
    g_lo, g_hi = float(gamma_bounds[0]), float(gamma_bounds[1])
    if not b_hi > b_lo or not g_hi > g_lo:
        raise ValueError("beta_bounds / gamma_bounds 必须满足 lo < hi")
    if n_grid < 2:
        raise ValueError(f"n_grid 必须 >= 2，得到 {n_grid}")

    i0 = float(max(i_obs[0], 1e-9))
    s0 = float(N - i0)
    if s0 <= 0:
        raise ValueError("N 必须大于首个观测感染数")
    t_end = float(t_obs[-1])
    n_steps = max(50, int(math.ceil(t_end / 0.05)))  # 拟合内部步长约 0.05
    grid_t = np.linspace(0.0, t_end, n_steps + 1)

    def sse_of(beta: float, gamma: float) -> float:
        _, i_curve, _ = _sir_trajectory(beta, gamma, s0, i0, 0.0, t_end, n_steps)
        i_at_obs = np.interp(t_obs, grid_t, i_curve)
        resid = i_at_obs - i_obs
        return float(np.dot(resid, resid))

    refine_rounds = 4
    for _ in range(refine_rounds + 1):
        b_grid = np.linspace(b_lo, b_hi, n_grid)
        g_grid = np.linspace(g_lo, g_hi, n_grid)
        best = (math.inf, b_lo, g_lo)
        for b in b_grid:
            for g in g_grid:
                val = sse_of(float(b), float(g))
                if val < best[0]:
                    best = (val, float(b), float(g))
        _, b_best, g_best = best
        b_span = (b_hi - b_lo) / 4.0
        g_span = (g_hi - g_lo) / 4.0
        b_lo, b_hi = b_best - b_span / 2.0, b_best + b_span / 2.0
        g_lo, g_hi = g_best - g_span / 2.0, g_best + g_span / 2.0

    best_sse = sse_of(b_best, g_best)
    return {"beta": b_best, "gamma": g_best, "sse": best_sse}


def estimate_convergence_order(
    f: RhsFn,
    y_exact: Union[ArrayLike, Callable[[float], ArrayLike]],
    y0: ArrayLike,
    t_end: float,
    h_list: Sequence[float],
    method: str = "rk4",
) -> float:
    """用一系列步长估计数值方法的收敛阶 p：e(h) ≈ C h^p。

    参数:
        f: 右端函数。
        y_exact: 精确解，可以是 ``callable(t) -> y``，也可以是**终点处**的精确值数组。
        y0: 初值。
        t_end: 积分终点（从 t=0 开始）。
        h_list: 至少两个步长，建议按 h, h/2, h/4 递减。
        method: ``"rk4"``（默认）或 ``"euler"``。

    返回:
        float，最小二乘（对 log h 与 log e 做线性回归）得到的阶数 p。
        误差用终点处的欧氏范数 ||y_h(t_end) - y_exact(t_end)||。

    算法:
        1. 对每个 h 用指定格式积分到 t_end，算终点误差 e(h)；
        2. 令 x = ln h，y = ln e，最小二乘直线斜率即 -p，返回 p = -slope。

    复杂度:
        时间 O(sum 1/h_i) / 空间 O(max 1/h_i)。

    陷阱:
        1. h 太大时误差进入"非渐近区"，测出来的阶数会偏离理论值（RK4 甚至可能测出 3 或 5）；
           h 太小时舍入误差占主导，阶数会虚高甚至变负。请用 2~3 个数量级内的 h。
        2. 必须用**同一台机器、同一函数**比较；换用不同精确解口径（例如把插值值当精确值）
           会系统性偏移结果。
        3. 多个状态变量时用欧氏范数合成误差，量纲不同的分量会互相影响；必要时先无量纲化。

    参考:
        数值分析中"收敛阶/误差阶"的标准实验定义（如 Burden & Faires 第 5 章）。
    """
    hs = [float(h) for h in h_list]
    if len(hs) < 2:
        raise ValueError("h_list 至少需要两个步长")
    if any(h <= 0 for h in hs):
        raise ValueError("h_list 中的步长必须 > 0")
    solver = solve_ivp_rk4 if method == "rk4" else solve_ivp_euler
    if method not in ("rk4", "euler"):
        raise ValueError(f"method 只支持 rk4/euler，得到 {method!r}")

    if callable(y_exact):
        exact = as_vector(y_exact(float(t_end)), "y_exact(t_end)")
    else:
        exact = as_vector(y_exact, "y_exact")

    errs: List[float] = []
    for h in hs:
        _, y = solver(f, y0, (0.0, float(t_end)), h)
        errs.append(float(np.linalg.norm(y[-1] - exact)))
    log_h = np.log(np.array(hs, dtype=float))
    log_e = np.log(np.array(errs, dtype=float))
    # log e ≈ log C + p log h，所以回归斜率本身就是阶数 p。
    return float(np.polyfit(log_h, log_e, 1)[0])


def seir_rhs(
    t: float,
    y: ArrayLike,
    beta: float,
    sigma: float,
    gamma: float,
    mu: float = 0.0,
) -> np.ndarray:
    """SEIR 模型的右端项，y = [S, E, I, R]（含出生/死亡，mu=0 时是封闭人口）。

    参数:
        t: 当前时刻；本模型是自治系统，t 不参与计算，保留它是为了直接匹配
            ``solve_ivp_euler`` / ``solve_ivp_rk4`` 的 ``f(t, y)`` 调用口径。
        y: 状态向量 ``[S, E, I, R]``，长度 4。
        beta: 传染率（单位时间每个感染者有效接触并传染的比例）。
        sigma: 潜伏期出率，1/sigma 是平均潜伏期（E→I 的速率）。
        gamma: 恢复率，1/gamma 是平均感染期。
        mu: 自然出生/死亡率（出生项按 mu*N 补进 S，四个仓室各自按 mu 流失），默认 0。

    返回:
        np.ndarray，形状 (4,)，``[dS/dt, dE/dt, dI/dt, dR/dt]``：
        dS/dt = mu N - beta S I / N - mu S；
        dE/dt = beta S I / N - sigma E - mu E；
        dI/dt = sigma E - gamma I - mu I；
        dR/dt = gamma I - mu R。
        其中 N = S + E + I + R 由状态自身求和得到（守恒量不显式传入）。

    算法:
        标准 SEIR 常微分方程组（Hethcote 2000 综述里的出生-死亡版本），不做任何离散化。
        ``mu=0`` 时 N 严格守恒，是传染病建模最常用的封闭人口口径。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. 与 ``sir_rhs`` 的签名**不同**：``sir_rhs`` 是"先给参数、返回闭包"，
           本函数是直接的 ``f(t, y)`` 右端项（任务单指定），要把参数放在 t/y 之后。
           拼装时写 ``lambda t, y: seir_rhs(t, y, beta, sigma, gamma, mu)``，
           参数顺序写错不会报错只会给出另一组动力学。
        2. ``sigma`` 很大时方程变刚性：E 的时间尺度是 1/sigma，显式 RK4 要求
           ``sigma * dt`` 落在稳定域内（负实轴约 2.8），sigma=1e4 配 dt=0.1 会直接爆掉。
           要逼近 SIR 极限必须同时缩小 dt（本模块自测取 sigma=50、dt=1e-3）。
        3. 分母 N 用当前状态的求和；数值误差把状态推成负值后 N 会变小，这里只能挡住
           N<=0 的情况（返回全 0），挡不住物理上无意义的负值。
        4. R0 口径是 ``beta*sigma/((sigma+mu)(gamma+mu))``（见
           ``basic_reproduction_number``），sigma→∞ 时才退化为 SIR 的 ``beta/(gamma+mu)``；
           sigma 小的时候两个口径差别很大，论文里必须写清用的是哪一个。

    参考:
        Hethcote, "The mathematics of infectious diseases", SIAM Review 42(4), 2000。
    """
    b, s, g, m = float(beta), float(sigma), float(gamma), float(mu)
    if b < 0 or s < 0 or g < 0 or m < 0:
        raise ValueError(
            f"beta/sigma/gamma/mu 必须非负，得到 beta={beta}, sigma={sigma}, "
            f"gamma={gamma}, mu={mu}"
        )
    states = as_vector(y, "y")
    if states.size != 4:
        raise ValueError(f"y 必须是长度 4 的 [S, E, I, R]，得到长度 {states.size}")
    s_val, e_val, i_val, r_val = (float(v) for v in states)
    n = s_val + e_val + i_val + r_val
    if n <= 0:
        return np.zeros(4, dtype=float)
    new_inf = b * s_val * i_val / n
    return np.array(
        [
            m * n - new_inf - m * s_val,
            new_inf - s * e_val - m * e_val,
            s * e_val - g * i_val - m * i_val,
            g * i_val - m * r_val,
        ],
        dtype=float,
    )


def simulate_seir(
    y0: ArrayLike,
    beta: float,
    sigma: float,
    gamma: float,
    mu: float = 0.0,
    t_end: float = 100.0,
    dt: float = 0.1,
    method: str = "rk4",
) -> Dict[str, object]:
    """定步长求解 SEIR，并给出感染峰值与最终规模等常用指标。

    参数:
        y0: 初值 ``[S0, E0, I0, R0]``，长度 4，各分量非负且总和 > 0。
        beta: 传染率。
        sigma: 潜伏期出率。
        gamma: 恢复率。
        mu: 自然出生/死亡率，默认 0（封闭人口）。
        t_end: 模拟时长（从 t=0 积到 t=t_end）。
        dt: **建议**步长；实际步长会被微调成 ``t_end/ceil(t_end/dt)``，
            保证最后一个点正好落在 t_end（与 ``solve_ivp_euler`` 口径一致）。
        method: ``"rk4"``（默认，四阶）或 ``"euler"``（一阶，仅用于演示误差）。

    返回:
        dict，键为：
        ``t`` 形状 (n+1,) 的时间网格；
        ``S``/``E``/``I``/``R`` 各 (n+1,) 的轨迹；
        ``peak_I`` float，现存感染者峰值（数值解在网格上的最大值）；
        ``peak_time`` float，峰值出现时间（网格上**第一个**取到峰值的时刻）；
        ``final_size`` float，T 时刻的累计感染比例 R(T)/N0。

    算法:
        把 ``[S,E,I,R]`` 交给 ``seir_rhs`` 组成右端项，用 ``solve_ivp_rk4``（或
        ``solve_ivp_euler``）积分，再用 ``np.argmax`` 找 I 的峰值。

    复杂度:
        时间 O(t_end/dt) / 空间 O(t_end/dt)。

    陷阱:
        1. **峰值是网格上的峰**：dt 决定峰值时间的分辨率，dt=0.1 时不要报"第 38.55 天"。
        2. ``final_size`` 是 **t_end 时刻**的 R/N0，不是 R(∞)/N0；t_end 太小时会低估。
        3. dt 与 sigma 必须配套：显式格式要求 ``sigma*dt`` 在稳定域内，否则解会指数爆炸
           而不报任何错误（详见 ``seir_rhs`` 的陷阱 2）。
        4. 这里不做非负裁剪：Euler 大步长可能把某个仓室算成负数，后续 N 变小会让
           ``beta S I / N`` 被高估，出现"越算越大"的假象。

    参考:
        Hethcote 2000；SEIR 数值实验的标准做法。
    """
    states = as_vector(y0, "y0").copy()
    if states.size != 4:
        raise ValueError(f"y0 必须是长度 4 的 [S0, E0, I0, R0]，得到长度 {states.size}")
    if np.any(states < 0):
        raise ValueError(f"y0 各分量必须非负，得到 {states.tolist()}")
    n_total = float(states.sum())
    if n_total <= 0:
        raise ValueError("总人口 S0+E0+I0+R0 必须 > 0")
    if float(t_end) <= 0:
        raise ValueError(f"t_end 必须 > 0，得到 {t_end}")
    if float(dt) <= 0:
        raise ValueError(f"dt 必须 > 0，得到 {dt}")
    if method not in ("rk4", "euler"):
        raise ValueError(f"method 只支持 rk4/euler，得到 {method!r}")

    def f(t: float, y: ArrayLike) -> np.ndarray:
        return seir_rhs(t, y, beta, sigma, gamma, mu)

    solver = solve_ivp_rk4 if method == "rk4" else solve_ivp_euler
    t, y = solver(f, states, (0.0, float(t_end)), float(dt))
    s_arr, e_arr, i_arr, r_arr = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    peak_idx = int(np.argmax(i_arr))
    return {
        "t": t,
        "S": s_arr,
        "E": e_arr,
        "I": i_arr,
        "R": r_arr,
        "peak_I": float(i_arr[peak_idx]),
        "peak_time": float(t[peak_idx]),
        "final_size": float(r_arr[-1] / n_total),
    }


def basic_reproduction_number(
    beta: float,
    sigma: float,
    gamma: float,
    mu: float = 0.0,
) -> float:
    """SEIR 的基本再生数闭式解 R0 = beta*sigma / ((sigma+mu)(gamma+mu))。

    参数:
        beta: 传染率。
        sigma: 潜伏期出率（E→I 的速率），1/sigma 是平均潜伏期。
        gamma: 恢复率。
        mu: 自然出生/死亡率（与 ``seir_rhs`` 同口径），默认 0。

    返回:
        float，基本再生数 R0。

    算法:
        用下一代矩阵法对 ``seir_rhs`` 的感染仓室 [E, I] 求谱半径：
        K = [[beta*S/N, beta*S/N], [0, 0]] 型矩阵经 V = [[sigma+mu, 0], [-sigma, gamma+mu]]
        消去后，R0 = beta*sigma / ((sigma+mu)(gamma+mu))（在 S=N 的初始时刻取值 1 归一）。
        **口径说明**：这是"考虑潜伏期"的版本；若把潜伏期并入感染期（或令 sigma→∞），
        退化为常见教科书口径 ``beta/(gamma+mu)``。本函数实现前者，
        且恒有 ``R0_seir = (sigma/(sigma+mu)) * (beta/(gamma+mu)) <= beta/(gamma+mu)``。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. 两个口径在 sigma 很大时几乎相等，但 sigma 与 gamma 同量级时能差一倍以上；
           用哪个口径必须写进论文，否则复现者算出的"R0=3"可能是另一个模型。
        2. ``sigma=0`` 时本函数返回 0（无人从 E 转为 I，疫情传不起来），而
           ``beta/(gamma+mu)`` 口径会给出一个正数——这不是 bug，是口径差异。
        3. 这里的 R0 是**初始时刻 S=N** 的值；流行过程中有效再生数
           ``Rt = R0 * S(t)/N`` 才是控制阈值。

    参考:
        Diekmann, Heesterbeek & Metz, "On the definition and the computation of the
        basic reproduction ratio R0", J. Math. Biol. 28, 1990。
    """
    b, s, g, m = float(beta), float(sigma), float(gamma), float(mu)
    if b < 0 or s < 0 or g < 0 or m < 0:
        raise ValueError(
            f"beta/sigma/gamma/mu 必须非负，得到 beta={beta}, sigma={sigma}, "
            f"gamma={gamma}, mu={mu}"
        )
    denom_g = g + m
    if denom_g <= 0:
        raise ValueError(f"gamma+mu 必须 > 0，得到 {denom_g}")
    if s + m <= 0:
        # sigma=mu=0：没人能从 E 转为 I，疫情传不起来，R0 定义为 0（见陷阱 2）。
        return 0.0
    return b * s / ((s + m) * denom_g)


def _num_jacobian(f: RhsFn, t: float, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """内部工具：中心差分数值雅可比 J[i, j] = d f_i / d y_j（在时刻 t、状态 y 处）。

    步长按 ``eps * max(1, |y_j|)`` 缩放，避免状态分量量级差异导致的精度退化。
    """
    m = y.size
    jac = np.empty((m, m), dtype=float)
    for j in range(m):
        step = eps * max(1.0, abs(float(y[j])))
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[j] += step
        y_minus[j] -= step
        f_plus = as_vector(f(t, y_plus), "f 返回值")
        f_minus = as_vector(f(t, y_minus), "f 返回值")
        if f_plus.size != m or f_minus.size != m:
            raise ValueError(
                f"f 返回值长度 {(f_plus.size, f_minus.size)} 与状态维数 {m} 不一致"
            )
        jac[:, j] = (f_plus - f_minus) / (2.0 * step)
    return jac


def implicit_euler(
    rhs: RhsFn,
    y0: ArrayLike,
    t_end: float,
    dt: float,
    newton_tol: float = 1e-10,
    max_iter: int = 50,
) -> Dict[str, np.ndarray]:
    """隐式（后向）Euler 法：y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})，每步用牛顿迭代求解。

    参数:
        rhs: 右端函数 ``f(t, y) -> dy/dt``。
        y0: 初值，长度 m 的一维数组。
        t_end: 积分终点（从 t=0 开始）。
        dt: 建议步长（按模块约定会微调以正好落在 t_end）。
        newton_tol: 牛顿迭代的收敛容差，用残差/修正量的无穷范数判定。
        max_iter: 每个时间步的最大牛顿迭代次数。

    返回:
        dict，键为：
        ``t`` 形状 (n+1,) 时间网格；
        ``y`` 形状 (n+1, m) 的解，第 i 行对应 t[i]。

    算法:
        1. 用显式 Euler 一步 ``y_n + h f(t_n, y_n)`` 作为预测子（比直接用 y_n 收敛快）；
        2. 牛顿迭代解非线性方程 ``G(y) = y - y_n - h f(t_{n+1}, y) = 0``：
           ``J_G = I - h * J_f``（``J_f`` 用 ``_num_jacobian`` 中心差分），
           ``y <- y + solve(J_G, -G(y))``；
        3. 残差或修正量的无穷范数小于 ``newton_tol`` 即认为该步收敛，
           超过 ``max_iter`` 次未收敛则抛 ValueError（不静默返回错误结果）。

    复杂度:
        时间 O(n * max_iter * m^2)（m 次额外的右端求值 + 一个 m×m 线性方程组）/
        空间 O(n m + m^2)。

    陷阱:
        1. **无条件稳定不等于无条件准确**：y'=-20y、h=0.25 时隐式 Euler 有界且收敛，
           但单步误差仍是 O(h)，它只是"不发散"，精度提升要靠减小 h。
        2. 线性问题牛顿一步就收敛；**强非线性**问题（如指数增长项）可能不收敛或跳到
           无意义的解，此时应减小 dt 或改用阻尼牛顿。
        3. 数值雅可比的默认 eps=1e-6 是绝对量级；状态分量极小或极大时差分会被舍入
           误差污染（``_num_jacobian`` 只做了 max(1,|y_j|) 的粗缩放），必要时手写解析雅可比。
        4. 步长仍由 ``_grid`` 微调，最后一步可能比 dt 小很多，此时该步的截断误差也更小，
           不会破坏整体 O(h) 的结论。

    参考:
        Burden & Faires《Numerical Analysis》后向 Euler/隐式格式章节；
        Hairer & Wanner, "Solving Ordinary Differential Equations II"（刚性问题的隐式格式）。
    """
    y = as_vector(y0, "y0").copy()
    if float(newton_tol) <= 0:
        raise ValueError(f"newton_tol 必须 > 0，得到 {newton_tol}")
    if int(max_iter) < 1:
        raise ValueError(f"max_iter 必须 >= 1，得到 {max_iter}")
    t = _grid((0.0, float(t_end)), float(dt))
    h = float(t[1] - t[0])
    m = y.size
    eye = np.eye(m)

    def rhs_at(t_val: float, y_val: np.ndarray) -> np.ndarray:
        val = as_vector(rhs(t_val, y_val), "rhs 返回值")
        if val.size != m:
            raise ValueError(f"rhs 返回值长度 {val.size} 与状态维数 {m} 不一致")
        return val

    def jac_at(t_val: float, y_val: np.ndarray) -> np.ndarray:
        return _num_jacobian(lambda tt, yy: rhs_at(tt, yy), t_val, y_val)

    out = np.empty((t.size, m), dtype=float)
    out[0] = y
    for i in range(t.size - 1):
        t_next = float(t[i + 1])
        y_old = y
        y_new = y_old + h * rhs_at(float(t[i]), y_old)
        converged = False
        for _ in range(int(max_iter)):
            resid = y_new - y_old - h * rhs_at(t_next, y_new)
            if float(np.max(np.abs(resid))) < float(newton_tol):
                converged = True
                break
            delta = np.linalg.solve(eye - h * jac_at(t_next, y_new), -resid)
            y_new = y_new + delta
            if float(np.max(np.abs(delta))) < float(newton_tol):
                converged = True
                break
        if not converged:
            raise ValueError(
                f"隐式 Euler 的牛顿迭代在 t={t_next} 处 {max_iter} 次内未收敛"
                f"（newton_tol={newton_tol}）；请减小 dt 或放宽 newton_tol"
            )
        y = y_new
        out[i + 1] = y
    return {"t": t, "y": out}


def solve_ivp_rk45(
    rhs: RhsFn,
    y0: ArrayLike,
    t_end: float,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt_init: Optional[float] = None,
) -> Dict[str, object]:
    """自适应步长的 Dormand-Prince RK45（5(4) 嵌入对 + 误差控制）。

    参数:
        rhs: 右端函数 ``f(t, y) -> dy/dt``。
        y0: 初值，长度 m 的一维数组。
        t_end: 积分终点（从 t=0 开始）。
        rtol: 相对容差（作用在 RMS 加权误差范数上）。
        atol: 绝对容差（同上，状态接近 0 时由它接管）。
        dt_init: 初始步长；None 表示取 ``t_end/100``。

    返回:
        dict，键为：
        ``t`` 形状 (n+1,) 的**被接受**的时间点（非均匀，末点严格等于 t_end）；
        ``y`` 形状 (n+1, m) 的解；
        ``n_steps`` int，被接受的步数；
        ``n_rejected`` int，被拒绝（误差超容差后重试）的步数。

    算法:
        1. 7 级 Dormand-Prince 格式：k1..k7，五阶解 ``y5 = y + h Σ b5_i k_i``，
           嵌入四阶解 ``y4 = y + h Σ b4_i k_i``，误差估计 ``err = y5 - y4``；
        2. 加权 RMS 范数 ``||err / (atol + rtol * max(|y|, |y5|))||_2 / sqrt(m)``，
           小于等于 1 则接受该步；
        3. 步长更新 ``h <- h * min(5, max(0.2, 0.9 * errnorm^{-1/5}))``（接受时），
           拒绝时 ``h <- h * max(0.1, 0.9 * errnorm^{-1/5})``；
        4. 末步把 h 裁剪到刚好落在 t_end，避免最后一个"无穷小步"。

    复杂度:
        时间 O(7 * n_steps * m) / 空间 O(n_steps * m)。

    陷阱:
        1. **不做稠密输出**：返回的 t 是非均匀的求解器步点，画图/比较前要用
           ``np.interp`` 重采样到你要的时间网格，别假设它是等距的。
        2. 误差控制是**局部**的：rtol 限制的是单步误差，长时间积分后全局误差会累积
           （通常仍是 O(rtol) 量级，但混沌系统会指数放大）。
        3. 用 ``atol`` 兜住接近 0 的分量；若某个分量自然量级是 1e-12，
           默认 atol=1e-9 会把它当"零"而完全不管误差。
        4. 步数超过 2e6 或建议步长小于 1e-14*t_end 时直接抛 ValueError，
           而不是死循环——这通常意味着方程有奇异点或容差过严。

    参考:
        Dormand & Prince, "A family of embedded Runge-Kutta formulae",
        J. Comput. Appl. Math. 6(1), 1980；Hairer, Norsett & Wanner, 1993。
    """
    y = as_vector(y0, "y0").copy()
    t_end = float(t_end)
    if t_end <= 0:
        raise ValueError(f"t_end 必须 > 0，得到 {t_end}")
    if float(rtol) <= 0:
        raise ValueError(f"rtol 必须 > 0，得到 {rtol}")
    if float(atol) < 0:
        raise ValueError(f"atol 必须 >= 0，得到 {atol}")
    m = y.size

    # Dormand-Prince 系数：A[s] 是第 s+1 级（k_{s+1}）的系数，C[s-1] 是其时间系数。
    c_stage = (0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0)
    a_stage = (
        (),
        (0.2,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
        (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
        (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
    )
    b5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
    b4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
          -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)

    h = t_end / 100.0 if dt_init is None else float(dt_init)
    if h <= 0:
        raise ValueError(f"dt_init 必须 > 0，得到 {dt_init}")
    if not math.isfinite(h):
        raise ValueError(f"dt_init 必须是有限值，得到 {dt_init}")
    h = min(h, t_end)

    t_cur = 0.0
    ts: List[float] = [0.0]
    ys: List[np.ndarray] = [y.copy()]
    n_steps = 0
    n_rejected = 0
    max_attempts = 2000000
    min_h = 1e-14 * max(1.0, t_end)

    while t_cur < t_end:
        if n_steps + n_rejected >= max_attempts:
            raise ValueError(
                f"RK45 尝试步数超过上限 {max_attempts}（t={t_cur}），"
                f"请检查 rhs 是否含奇异点或放宽 rtol/atol"
            )
        h = min(h, t_end - t_cur)
        if h < min_h:
            raise ValueError(
                f"RK45 建议步长 {h} 小于下限 {min_h}（t={t_cur}），容差可能过严"
            )
        k = [None] * 7
        k[0] = as_vector(rhs(t_cur, y), "rhs 返回值")
        if k[0].size != m:
            raise ValueError(f"rhs 返回值长度 {k[0].size} 与状态维数 {m} 不一致")
        for s in range(1, 7):
            acc = np.zeros(m, dtype=float)
            for j in range(s):
                acc = acc + a_stage[s][j] * k[j]
            k[s] = as_vector(rhs(t_cur + c_stage[s - 1] * h, y + h * acc), "rhs 返回值")
        y5 = y.copy()
        y4 = y.copy()
        for j in range(7):
            y5 = y5 + h * b5[j] * k[j]
            y4 = y4 + h * b4[j] * k[j]
        scale = float(atol) + float(rtol) * np.maximum(np.abs(y), np.abs(y5))
        err_norm = float(np.sqrt(np.mean(((y5 - y4) / scale) ** 2)))

        if err_norm <= 1.0:
            t_cur = t_cur + h
            y = y5
            n_steps += 1
            ts.append(t_cur)
            ys.append(y.copy())
            factor = 5.0 if err_norm <= 0.0 else min(5.0, max(0.2, 0.9 * err_norm ** -0.2))
        else:
            n_rejected += 1
            factor = max(0.1, 0.9 * err_norm ** -0.2)
        h = h * factor

    return {
        "t": np.array(ts, dtype=float),
        "y": np.vstack(ys),
        "n_steps": int(n_steps),
        "n_rejected": int(n_rejected),
    }


def jacobian_stability(rhs: RhsFn, y_eq: ArrayLike, eps: float = 1e-6) -> Dict[str, object]:
    """在平衡点处数值求雅可比并判断线性稳定性（全部特征值实部 < 0 记为稳定）。

    参数:
        rhs: 右端函数 ``f(t, y) -> dy/dt``。
        y_eq: 平衡点（或任意工作点）状态向量，长度 m。
        eps: 中心差分步长（按 ``eps * max(1, |y_j|)`` 逐分量缩放）。

    返回:
        dict，键为：
        ``jacobian`` 形状 (m, m) 的数值雅可比；
        ``eigenvalues`` 长度 m 的复特征值数组（``np.linalg.eigvals`` 的顺序不保证稳定）；
        ``stable`` bool，所有特征值实部严格 < 0 时为 True。

    算法:
        1. 逐分量中心差分 ``J[:, j] = (f(y + eps e_j) - f(y - eps e_j)) / (2 eps)``；
        2. ``np.linalg.eigvals(J)`` 求特征值；
        3. 判据：``max Re(lambda) < 0``（Lyapunov 意义下的局部渐近稳定）。

    复杂度:
        时间 O(m^2)（含 2m 次右端求值）/ 空间 O(m^2)。

    陷阱:
        1. 这里固定取 ``t = 0`` 求雅可比：**只对自治系统有意义**。非自治系统请把时刻
           烘焙进闭包（``lambda t, y: f(t0, y)``），否则得到的是 t=0 处的"冻结"矩阵。
        2. 判据是**线性化**结论：实部恰好为 0（中心/临界情形）时返回 False，
           此时稳定性由非线性项决定，本函数给出的结论是"不能判定为稳定"。
        3. 特征值对 eps 敏感：eps 太大引入截断误差，太小被舍入误差淹没；
           对病态雅可比（条件数极大）做尺度缩放后再解释结果。
        4. 返回的 ``eigenvalues`` 是复数组，写进 JSON 前要自己取 ``.real/.imag``。

    参考:
        Strogatz, "Nonlinear Dynamics and Chaos"（线性稳定性分析）；
        任何数值分析教材的差分近似雅可比章节。
    """
    y = as_vector(y_eq, "y_eq").copy()
    if float(eps) <= 0:
        raise ValueError(f"eps 必须 > 0，得到 {eps}")
    jac = _num_jacobian(rhs, 0.0, y, float(eps))
    eig = np.linalg.eigvals(jac)
    stable = bool(np.all(np.real(eig) < 0.0))
    return {"jacobian": jac, "eigenvalues": eig, "stable": stable}


def logistic_map(
    r: float,
    x0: float = 0.4,
    n_steps: int = 200,
    n_transient: int = 100,
) -> Dict[str, object]:
    """离散 logistic 映射 x_{n+1} = r x_n (1 - x_n)：轨道、Lyapunov 指数与周期检测。

    参数:
        r: 增长参数（要求 >= 0；r>4 时轨道会逃出 [0,1]，本函数不拦）。
        x0: 初值，必须落在 (0, 1) 内。
        n_steps: 丢弃暂态后记录的迭代步数（>= 2）。
        n_transient: 丢弃的暂态步数（>= 0）。

    返回:
        dict，键为：
        ``x`` 形状 (n_steps,) 的轨道（暂态已丢弃，x[0] 是第 n_transient 次迭代的结果）；
        ``lyapunov`` float，Lyapunov 指数 ``mean(ln|r(1-2x_n)|)``（对记录的 n_steps 步取平均）；
        ``period`` int，检测到的周期；≤33 的周期内找不到重复则返回 0（表示"长周期/无周期"）。

    算法:
        1. 先迭代 n_transient 次丢弃暂态，再记录 n_steps 个点；
        2. Lyapunov 指数用轨道上的导数乘积的平均对数：
           ``lambda = (1/n) Σ ln|f'(x_n)| = (1/n) Σ ln|r(1-2x_n)|``；
        3. 周期检测：在最后 ``min(n_steps, 100)`` 个点组成的尾巴上，对 p=1..len/3
           检查 ``max|x_i - x_{i-p}| < 1e-6``，第一个满足的 p 即周期，都不满足返回 0。

    复杂度:
        时间 O(n_steps + n_transient) / 空间 O(n_steps)。

    陷阱:
        1. **对初值和暂态长度敏感**：周期检测用的是"尾巴上逐点重合"的强判据，
           若 n_transient 不够（轨道还没落到周期轨道上）或 r 落在周期窗口里，
           检测结果会变；报告周期时必须同时给出 n_steps/n_transient。
        2. ``period=0`` 的含义是"在 ≤33 的窗口内没检出周期"，**不等于**已证明混沌
           （可能有 64 周期），要结合 Lyapunov 指数正负一起解释。
        3. Lyapunov 指数由有限步平均得到，n_steps 太小时波动大；
           r 略大于 3 的倍周期分岔点附近，指数在 0 附近，符号不稳定。
        4. r=0 或轨道恰好命中 x=0.5（f'=0）时对数发散，本函数把导数下限钳到 1e-300，
           避免 -inf 污染结果。

    参考:
        May, "Simple mathematical models with very complicated dynamics", Nature 261, 1976；
        Strogatz《Nonlinear Dynamics and Chaos》第 10 章。
    """
    r_val = float(r)
    if not math.isfinite(r_val) or r_val < 0.0:
        raise ValueError(f"r 必须是 >= 0 的有限值，得到 {r}")
    x_val = float(x0)
    if not (0.0 < x_val < 1.0):
        raise ValueError(f"x0 必须落在 (0, 1) 内，得到 {x0}")
    n_steps_i = int(n_steps)
    n_trans_i = int(n_transient)
    if n_steps_i < 2:
        raise ValueError(f"n_steps 必须 >= 2，得到 {n_steps}")
    if n_trans_i < 0:
        raise ValueError(f"n_transient 必须 >= 0，得到 {n_transient}")

    for _ in range(n_trans_i):
        x_val = r_val * x_val * (1.0 - x_val)

    xs = np.empty(n_steps_i, dtype=float)
    lyap_sum = 0.0
    for i in range(n_steps_i):
        xs[i] = x_val
        deriv = abs(r_val * (1.0 - 2.0 * x_val))
        lyap_sum += math.log(deriv) if deriv > 1e-300 else math.log(1e-300)
        x_val = r_val * x_val * (1.0 - x_val)

    tail_len = min(n_steps_i, 100)
    tail = xs[-tail_len:]
    period = 0
    for p in range(1, tail_len // 3 + 1):
        if float(np.max(np.abs(tail[p:] - tail[:-p]))) < 1e-6:
            period = p
            break

    return {
        "x": xs,
        "lyapunov": float(lyap_sum / n_steps_i),
        "period": int(period),
    }


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict（全部为 int/float），固定种子下两次调用完全一致：
        ``euler_y1`` / ``rk4_y1`` 对 y'=-y 积分到 t=1 的终点值（精确值 e^{-1}=0.367879）、
        ``rk4_order`` / ``euler_order`` 实测收敛阶、
        ``sir_peak`` / ``sir_peak_time`` / ``sir_final_size`` SIR 基准算例指标、
        ``sir_final_size_inf`` 由超越方程解出的真实 R(∞)/N 供对照、
        ``logistic_y20`` logistic 在 t=20 的数值解、``lv_x_at_10`` Lotka-Volterra 的 x(10)、
        ``fit_beta`` / ``fit_gamma`` / ``fit_sse`` 用自造数据拟合的 SIR 参数。

        [T11 追加] ``seir_peak`` / ``seir_peak_time`` / ``seir_final_size`` /
        ``seir_sir_peak_rel`` / ``seir_r0``：SEIR 与大 sigma 极限算例的指标；
        ``seir_subcritical_r0`` / ``seir_subcritical_final_size``：R0<1 的阈值算例；
        ``implicit_y1`` / ``implicit_y1_closed`` / ``implicit_stiff_y1`` /
        ``explicit_stiff_y1``：隐式与显式 Euler 的闭式对拍与刚性对照；
        ``rk45_y1`` / ``rk45_rel_error`` / ``rk45_n_steps`` / ``rk45_n_rejected``：
        自适应步长 RK45 的结果；``jac_logistic_stable_at_K`` /
        ``jac_logistic_max_real_at_K`` / ``jac_logistic_stable_at_0``：雅可比稳定性；
        ``logistic_map_lyap_32`` / ``logistic_map_period_32`` / ``logistic_map_lyap_39`` /
        ``logistic_map_period_39``：logistic 映射的 Lyapunov 指数与周期。

    算法:
        SIR 基准：N=1000、I0=1、beta=0.3、gamma=0.1（R0=3），积分 160 天，h=0.01。
        拟合算例：用 beta=0.25、gamma=0.1 生成 45 天、每 2 天一个点的"观测"（无噪声），
        覆盖上升段与过峰后的下降段，看拟合能否回到真值附近。

    复杂度:
        时间 O(1e6) / 空间 O(1e4)（含一次网格拟合）。

    陷阱:
        拟合算例特意取**无噪声且覆盖过峰段**的数据：若只给前 15 天的上升段，
        beta 与 gamma 只在差值 beta-gamma 上可辨识，SSE 曲面会出现一条平坦脊，
        拟合值会偏离真值（本模块实测：只给上升段时 beta≈0.278、gamma≈0.127，
        虽然 beta-gamma≈0.150 与真值几乎一致，但 R0 被显著低估）。
        这说明"无噪声数据"并不保证参数唯一，论文里必须报告可辨识性分析。

    参考:
        本模块各函数参考文献。
    """
    def f_decay(t: float, y: ArrayLike) -> np.ndarray:
        return -as_vector(y, "y")

    exact_end = float(np.exp(-1.0))
    _, y_euler = solve_ivp_euler(f_decay, np.array([1.0]), (0.0, 1.0), 0.01)
    _, y_rk4 = solve_ivp_rk4(f_decay, np.array([1.0]), (0.0, 1.0), 0.01)
    order_rk4 = estimate_convergence_order(
        f_decay, np.array([exact_end]), np.array([1.0]), 1.0, [0.2, 0.1, 0.05, 0.025]
    )
    order_euler = estimate_convergence_order(
        f_decay, np.array([exact_end]), np.array([1.0]), 1.0, [0.02, 0.01, 0.005, 0.0025],
        method="euler",
    )

    sir = simulate_sir(999.0, 1.0, 0.0, 0.3, 0.1, 160.0, 0.01)
    # 真实最终规模解 R∞ = 1 - exp(-R0 * R∞)（此处 R0 = beta/gamma = 3）。
    r_inf = 0.9
    for _ in range(200):
        r_inf = 1.0 - math.exp(-3.0 * r_inf)

    _, y_log = logistic_growth(0.4, 100.0, 5.0, 20.0, 0.01)
    lv = lotka_volterra_rhs(1.0, 0.1, 0.075, 1.5)
    _, y_lv = solve_ivp_rk4(lv, np.array([10.0, 5.0]), (0.0, 10.0), 0.01)

    true_beta, true_gamma = 0.25, 0.1
    times = np.arange(0.0, 46.0, 2.0)
    _, i_true, _ = _sir_trajectory(true_beta, true_gamma, 999.0, 1.0, 0.0, 45.0, 900)
    obs = np.interp(times, np.linspace(0.0, 45.0, 901), i_true)
    fit = fit_sir_least_squares(times, obs, 1000.0, (0.05, 0.6), (0.02, 0.4), n_grid=16)

    # ---------- T11 新增算例：SEIR / 隐式格式 / 自适应步长 / 稳定性 / 混沌 ----------
    # (1) SEIR 的 R0 闭式解：mu=0 时恒等于 SIR 的 beta/gamma 口径；
    #     mu>0 时对 sigma 单调递增，且 sigma→∞ 退化为 beta/(gamma+mu)。
    r0_seir = basic_reproduction_number(0.3, 50.0, 0.1, 0.0)
    if abs(r0_seir - 0.3 / 0.1) > 1e-12:
        raise AssertionError(
            f"SEIR 的 R0={r0_seir} 在 mu=0 时应等于 SIR 的 beta/gamma={0.3 / 0.1}"
        )
    r0_mu = basic_reproduction_number(0.3, 1.0e6, 0.1, 0.05)
    if abs(r0_mu - 0.3 / (0.1 + 0.05)) > 1e-4:
        raise AssertionError(
            f"sigma→∞ 时 R0 应为 beta/(gamma+mu)={0.3 / (0.1 + 0.05)}，得到 {r0_mu}"
        )
    if not (basic_reproduction_number(0.3, 1.0, 0.1, 0.05)
            < basic_reproduction_number(0.3, 5.0, 0.1, 0.05)):
        raise AssertionError("R0 关于 sigma 应单调递增（潜伏期越短、传播越快）")
    if basic_reproduction_number(0.3, 0.0, 0.1, 0.0) != 0.0:
        raise AssertionError("sigma=0（无人转为感染）时 R0 必须为 0")

    # (2) SIR 是 SEIR 在 sigma→∞ 的极限：大 sigma 下 I 峰必须与 simulate_sir 接近（5%）。
    seir = simulate_seir([999.0, 0.0, 1.0, 0.0], 0.3, 50.0, 0.1, 0.0, 80.0, 1e-3)
    sir_ref = simulate_sir(999.0, 1.0, 0.0, 0.3, 0.1, 80.0, 5e-3)
    seir_sir_rel = abs(seir["peak_I"] - sir_ref["peak_infected"]) / sir_ref["peak_infected"]
    if seir_sir_rel > 0.05:
        raise AssertionError(
            f"sigma=50 的 SEIR 峰值 {seir['peak_I']} 与 SIR 峰值 "
            f"{sir_ref['peak_infected']} 相对差 {seir_sir_rel} 超过 5%"
        )

    # (3) R0<1 时疫情必须熄灭（final_size 接近 0），这是 R0 阈值的独立校验。
    sub = simulate_seir([999.0, 0.0, 1.0, 0.0], 0.05, 1.0, 0.1, 0.0, 60.0, 0.01)
    sub_r0 = basic_reproduction_number(0.05, 1.0, 0.1, 0.0)
    if not sub_r0 < 1.0:
        raise AssertionError(f"该算例本应 R0<1，得到 R0={sub_r0}")
    if sub["final_size"] > 0.01:
        raise AssertionError(
            f"R0<1 时 SEIR 不应爆发，得到 final_size={sub['final_size']}"
        )

    # (4) 隐式 Euler 对 y'=-y 有闭式递推 y_n = (1/(1+h))^n，直接对拍。
    imp = implicit_euler(f_decay, np.array([1.0]), 1.0, 0.01)
    imp_y = imp["y"]
    imp_closed = (1.0 / 1.01) ** 100
    if abs(float(imp_y[-1, 0]) - imp_closed) > 1e-9:
        raise AssertionError(f"隐式 Euler 终点 {imp_y[-1, 0]} 与闭式解 {imp_closed} 不符")

    # (5) 刚性算例 y'=-20y、h=0.25：隐式有界（闭式 (1/6)^4），显式 Euler 发散到 (-4)^4。
    def f_stiff(t: float, y: ArrayLike) -> np.ndarray:
        return -20.0 * as_vector(y, "y")

    imp_stiff = implicit_euler(f_stiff, np.array([1.0]), 1.0, 0.25)
    imp_stiff_y = imp_stiff["y"]
    _, exp_stiff = solve_ivp_euler(f_stiff, np.array([1.0]), (0.0, 1.0), 0.25)
    imp_stiff_closed = (1.0 / 6.0) ** 4
    if abs(float(imp_stiff_y[-1, 0]) - imp_stiff_closed) > 1e-12:
        raise AssertionError(
            f"隐式 Euler 在刚性算例上应等于闭式 {imp_stiff_closed}，"
            f"得到 {imp_stiff_y[-1, 0]}"
        )
    if abs(float(imp_stiff_y[-1, 0])) > 1.0:
        raise AssertionError(f"隐式 Euler 在刚性大步长下发散：{imp_stiff_y[-1, 0]}")
    if abs(float(exp_stiff[-1, 0])) <= 1.0:
        raise AssertionError(
            f"显式 Euler 在 h=0.25、y'=-20y 上本应发散，"
            f"实际 |y|={abs(float(exp_stiff[-1, 0]))}"
        )

    # (6) RK45 对 y'=y 的精度：终点必须逼近 e（相对误差 < 1e-5）。
    def f_exp_growth(t: float, y: ArrayLike) -> np.ndarray:
        return as_vector(y, "y")

    rk45 = solve_ivp_rk45(f_exp_growth, np.array([1.0]), 1.0, 1e-6, 1e-9)
    rk45_rel = abs(float(rk45["y"][-1, 0]) - math.e) / math.e
    if rk45_rel > 1e-5:
        raise AssertionError(f"RK45 对 exp(t) 的相对误差 {rk45_rel} 超过 1e-5")
    if abs(float(rk45["t"][-1]) - 1.0) > 1e-12:
        raise AssertionError(f"RK45 末点 {rk45['t'][-1]} 未落在 t_end=1.0")

    # (7) 雅可比稳定性：logistic 在 K 处稳定（特征值 -r）、在 0 处不稳定（特征值 +r）。
    def f_logistic(t: float, y: ArrayLike) -> np.ndarray:
        val = float(as_vector(y, "y")[0])
        return np.array([0.4 * val * (1.0 - val / 100.0)], dtype=float)

    jac_k = jacobian_stability(f_logistic, np.array([100.0]))
    jac_zero = jacobian_stability(f_logistic, np.array([0.0]))
    if not jac_k["stable"]:
        raise AssertionError(f"logistic 在 K 处应稳定，特征值 {jac_k['eigenvalues']}")
    if jac_zero["stable"]:
        raise AssertionError(f"logistic 在 0 处应不稳定，特征值 {jac_zero['eigenvalues']}")
    if abs(float(np.real(jac_k["eigenvalues"][0])) + 0.4) > 1e-6:
        raise AssertionError(
            f"logistic 在 K 处的特征值应为 -r=-0.4，得到 {jac_k['eigenvalues']}"
        )

    # (8) logistic 映射：r=3.2 是 2 周期（Lyapunov<0），r=3.9 混沌（Lyapunov>0、无短周期）。
    lm32 = logistic_map(3.2, 0.4, 200, 100)
    lm39 = logistic_map(3.9, 0.4, 200, 100)
    if not lm32["lyapunov"] < 0.0:
        raise AssertionError(f"r=3.2 应为稳定周期，得到 lyapunov={lm32['lyapunov']}")
    if lm32["period"] != 2:
        raise AssertionError(f"r=3.2 应检出周期 2，得到 {lm32['period']}")
    # r=3.2 的 2 周期是闭式的：乘子 mu = 4 + 2r - r^2 = 0.16，Lyapunov = ln|mu|/2。
    lyap32_closed = 0.5 * math.log(abs(4.0 + 2.0 * 3.2 - 3.2 ** 2))
    if abs(lm32["lyapunov"] - lyap32_closed) > 1e-9:
        raise AssertionError(
            f"r=3.2 的 Lyapunov {lm32['lyapunov']} 与 2 周期乘子闭式解 "
            f"{lyap32_closed} 不符"
        )
    if not lm39["lyapunov"] > 0.0:
        raise AssertionError(f"r=3.9 应为混沌，得到 lyapunov={lm39['lyapunov']}")
    if 0 < lm39["period"] <= 10:
        raise AssertionError(f"r=3.9 不应检出短周期，得到 {lm39['period']}")
    # 两个精确基准：r=2.5 的不动点乘子 2-r（Lyapunov=ln|2-r|）、r=4 的 Lyapunov = ln 2。
    lm25 = logistic_map(2.5, 0.4, 200, 100)
    if abs(lm25["lyapunov"] - math.log(abs(2.0 - 2.5))) > 1e-9:
        raise AssertionError(
            f"r=2.5 的 Lyapunov {lm25['lyapunov']} 应等于 ln|2-r|={math.log(0.5)}"
        )
    lm40 = logistic_map(4.0, 0.4, 200, 100)
    if abs(lm40["lyapunov"] - math.log(2.0)) > 2e-3:
        raise AssertionError(
            f"r=4 的 Lyapunov {lm40['lyapunov']} 应逼近 ln2={math.log(2.0)}（容差 2e-3）"
        )

    return {
        "euler_y1": round(float(y_euler[-1, 0]), 8),
        "rk4_y1": round(float(y_rk4[-1, 0]), 10),
        "rk4_order": round(float(order_rk4), 4),
        "euler_order": round(float(order_euler), 4),
        "sir_peak": round(float(sir["peak_infected"]), 4),
        "sir_peak_time": round(float(sir["peak_time"]), 4),
        "sir_final_size": round(float(sir["final_size"]), 6),
        "sir_final_size_inf": round(float(r_inf), 6),
        "logistic_y20": round(float(y_log[-1]), 8),
        "lv_x_at_10": round(float(y_lv[-1, 0]), 6),
        "fit_beta": round(float(fit["beta"]), 6),
        "fit_gamma": round(float(fit["gamma"]), 6),
        "fit_sse": round(float(fit["sse"]), 8),
        "seir_peak": round(float(seir["peak_I"]), 4),
        "seir_peak_time": round(float(seir["peak_time"]), 4),
        "seir_final_size": round(float(seir["final_size"]), 6),
        "seir_sir_peak_rel": round(float(seir_sir_rel), 6),
        "seir_r0": round(float(r0_seir), 6),
        "seir_subcritical_r0": round(float(sub_r0), 6),
        "seir_subcritical_final_size": round(float(sub["final_size"]), 6),
        "implicit_y1": round(float(imp_y[-1, 0]), 10),
        "implicit_y1_closed": round(float(imp_closed), 10),
        "implicit_stiff_y1": round(float(imp_stiff_y[-1, 0]), 10),
        "explicit_stiff_y1": round(float(exp_stiff[-1, 0]), 6),
        "rk45_y1": round(float(rk45["y"][-1, 0]), 8),
        "rk45_rel_error": round(float(rk45_rel), 10),
        "rk45_n_steps": int(rk45["n_steps"]),
        "rk45_n_rejected": int(rk45["n_rejected"]),
        "jac_logistic_stable_at_K": bool(jac_k["stable"]),
        "jac_logistic_max_real_at_K": round(float(np.max(np.real(jac_k["eigenvalues"]))), 6),
        "jac_logistic_stable_at_0": bool(jac_zero["stable"]),
        "logistic_map_lyap_32": round(float(lm32["lyapunov"]), 6),
        "logistic_map_period_32": int(lm32["period"]),
        "logistic_map_lyap_39": round(float(lm39["lyapunov"]), 6),
        "logistic_map_period_39": int(lm39["period"]),
    }
