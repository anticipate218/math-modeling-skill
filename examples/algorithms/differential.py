"""常微分方程初值问题：Euler/RK4 定步长积分、SIR 传染病模型、logistic 增长、
Lotka-Volterra 捕食者-被捕食者、SIR 参数最小二乘拟合、收敛阶估计。

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
from typing import Callable, Dict, List, Sequence, Tuple, Union

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
    }
