"""时间序列进阶与状态空间：GM(1,1) 灰色预测、ARIMA/SARIMA、GARCH(1,1)、卡尔曼滤波、Ljung-Box 白噪声检验。

本模块是 ``forecasting`` 的**进阶补充**，不重复其中的移动平均、指数平滑、Holt-Winters、
Yule-Walker AR、ACF/PACF、ADF 与精度指标。这里给出四类"更进一步"的方法：

1. **灰色系统**：GM(1,1) 的一次累加—紧邻均值—最小二乘—累减还原全流程，以及后验差检验；
2. **Box-Jenkins 类模型**：ARIMA(p,d,q) 与 SARIMA(p,d,q)(P,D,Q)_s，用条件最小二乘估计；
3. **条件异方差**：GARCH(1,1) 的正态拟极大似然（QML）估计与多步方差预测；
4. **状态空间**：局部水平模型与一般线性高斯模型的卡尔曼滤波，以及 Ljung-Box 残差检验。

函数清单（与 ``__all__`` 一致，共 13 个）
----------------------------------------
- ``gm11``：GM(1,1) 一次累加—紧邻均值—最小二乘—累减还原与样本外预测；
- ``gm11_posterior_check``：后验差检验（方差比 C、小误差概率 P、精度等级）；
- ``difference_series``：d 阶差分，**保持长度**（前 d 位为 ``NaN``）；
- ``arima_fit``：ARIMA(p,d,q) 的条件最小二乘拟合与 AIC/BIC；
- ``arima_forecast``：多步点预测与预测标准差（支持普通差分 ``d >= 0`` 的逐阶累加反差分，
  仅季节差分 ``D == 0`` 不支持）；
- ``arima_order_select``：``(p,d,q)`` 网格上的 AIC/BIC 选阶；
- ``sarima_fit``：SARIMA(p,d,q)(P,D,Q)_s 的扩展条件最小二乘拟合；
- ``garch11_fit``：GARCH(1,1) 的 QML 估计（多起点模式搜索）；
- ``garch11_forecast``：GARCH(1,1) 的多步条件方差预测；
- ``kalman_filter_local_level``：局部水平模型的单变量卡尔曼滤波；
- ``kalman_filter_linear``：一般线性高斯状态空间模型的卡尔曼滤波；
- ``kalman_smoother_linear``：一般线性高斯模型的卡尔曼滤波 + RTS 固定区间平滑；
- ``ljung_box``：Ljung-Box 白噪声（自相关）检验。

关键约定
--------
- 序列一律按时间先后排列（``x[0]`` 最早）；本模块**不做任何随机打乱**。
- 差分口径分两套，务必分清：``difference_series`` 返回**与输入等长**的数组、前 d 位为
  ``NaN``；模型内部（``arima_fit`` / ``sarima_fit``）的差分会**缩短序列**（同 ``np.diff``）。
- 所有带随机性的过程（这里只有 ``garch11_fit`` 的多起点）都显式接收 ``seed``，
  默认 ``DEFAULT_SEED``（见 ``_common.rng``），绝不使用 ``np.random.*`` 全局状态。
- **教学透明版的边界（诚实声明）**：``arima_fit`` / ``sarima_fit`` 用的是**条件最小二乘**
  （把滞后残差当已知的迭代线性回归），不是精确 MLE/CSS-ML；``garch11_fit`` 用的是
  **模式搜索（compass search）**，不是 BFGS/数值梯度拟牛顿。二者在样本充足时与标准软件
  结果接近，但**不保证**逐位一致。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_matrix, as_vector, check_same_length, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "gm11",
    "gm11_posterior_check",
    "difference_series",
    "arima_fit",
    "arima_forecast",
    "arima_order_select",
    "sarima_fit",
    "garch11_fit",
    "garch11_forecast",
    "kalman_filter_local_level",
    "kalman_filter_linear",
    "kalman_smoother_linear",
    "ljung_box",
]

#: ``arima_order_select`` 支持的准则名。
_ORDER_CRITERIA = ("aic", "bic")

#: 条件最小二乘的默认迭代上限与收敛容差（系数向量的最大逐分量变化）。
_CLS_MAX_ITER = 200
_CLS_TOL = 1e-10

#: GARCH 模式搜索的参数约束：``alpha + beta`` 必须不超过它（``_garch_feasible`` 用的是
#: ``<=``，即取到 ``0.999`` 本身仍算可行）。
_GARCH_PERSIST_MAX = 0.999


# --------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------

def _as_int(value, name: str, minimum: int = 0) -> int:
    """把输入校验为 >= ``minimum`` 的整数并返回；不合法抛 ``ValueError``。

    参数:
        value: 待校验对象（必须是 int 或 numpy 整数，bool 被拒绝）。
        name: 出错信息里显示的参数名。
        minimum: 允许的最小值。

    返回:
        int。

    算法:
        类型判断 + 下界检查，不做四舍五入（浮点 2.0 会被拒绝，避免"悄悄取整"）。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        ``isinstance(True, int)`` 在 Python 里为真，因此必须显式排除 bool，
        否则 ``d=True`` 会被当成 1 阶差分悄悄通过。

    参考:
        Python 数据模型对 bool 继承 int 的说明。
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} 必须是整数，得到 {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，得到 {v}")
    return v


def _as_positive(value, name: str, strict: bool = False) -> float:
    """把输入校验为有限的正数（或非负数）并返回；不合法抛 ``ValueError``。

    参数:
        value: 待校验对象。
        name: 出错信息里显示的参数名。
        strict: 为真时要求严格大于 0，否则允许 0。

    返回:
        float。

    算法:
        先 ``float()`` 归一，再检查有限性与符号。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        ``np.nan`` 与 ``np.inf`` 都能通过 ``float()``，必须在符号检查之前用
        ``np.isfinite`` 拦掉，否则 NaN 会一路传进似然函数变成 NaN 而不是报错。

    参考:
        IEEE-754 对 NaN 的比较语义（NaN 与任何数比较均为 False）。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数值，得到 {value!r}")
    if not np.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值，得到 {value!r}")
    if strict:
        if v <= 0.0:
            raise ValueError(f"{name} 必须严格为正，得到 {v}")
    elif v < 0.0:
        raise ValueError(f"{name} 必须非负，得到 {v}")
    return v


def _gammainc_upper_reg(a: float, x: float) -> float:
    """正则化上不完全 gamma 函数 ``Q(a, x) = 1 - P(a, x)``。

    参数:
        a: 形状参数，要求 > 0。
        x: 自变量，要求 >= 0。

    返回:
        float，落在 [0, 1]。

    算法:
        1. ``x < a + 1`` 时先算 lower 级数
           ``P(a,x) = x^a e^{-x} / Gamma(a) * sum_{n>=0} x^n / (a(a+1)...(a+n))``，
           再取 ``Q = 1 - P``（该区间级数收敛快）；
        2. 否则用 Lentz 修正连分式直接算 ``Q(a,x)``（Numerical Recipes 的 ``gammq`` 口径），
           前置因子 ``exp(-x + a ln x - lgamma(a))``。

    复杂度:
        时间 O(迭代次数)（最多 1000 次）/ 空间 O(1)。

    陷阱:
        - 前置因子 ``x^a e^{-x}/Gamma(a)`` 在 ``a`` 很大时会下溢到 0，此时返回值直接是 0
          （数学上确实极小），调用方不要把它当成"没算出来"。
        - 级数分支与连分式分支在 ``x ≈ a+1`` 接缝处相对误差约 1e-14，不影响检验结论。
        - ``a`` 必须来自 ``df/2``，``df`` 为 0 或负数会让 ``lgamma`` 报错，调用方需先校验。

    参考:
        Press et al., "Numerical Recipes", 3rd ed., §6.2（gser/gcf）；
        Abramowitz & Stegun (1964) §6.5。
    """
    if a <= 0.0:
        raise ValueError(f"a 必须 > 0，得到 {a}")
    if x < 0.0:
        raise ValueError(f"x 必须 >= 0，得到 {x}")
    if x == 0.0:
        return 1.0
    log_pref = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) <= abs(total) * 1e-16:
                break
        p = total * math.exp(log_pref)
        return min(1.0, max(0.0, 1.0 - p))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if abs(b) > tiny else 1.0 / tiny
    h = d
    for i in range(1, 1000):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-16:
            break
    return min(1.0, max(0.0, math.exp(log_pref) * h))


def _chi2_sf(x: float, df: int) -> float:
    """卡方分布的生存函数 ``P(X > x)``，``X ~ chi2(df)``。

    参数:
        x: 统计量，>= 0。
        df: 自由度，正整数。

    返回:
        float，落在 [0, 1]；``x <= 0`` 时返回 1.0。

    算法:
        ``P(X > x) = Q(df/2, x/2)``，``Q`` 是正则化上不完全 gamma 函数
        （本模块 ``_gammainc_upper_reg`` 自实现，不依赖 scipy）。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        ``df`` 越大越依赖连分式分支；自由度超过约 1e5 时前置因子下溢，本函数会返回 0.0
        而不是 NaN——这是"p 值极小"的正确表现，不是失败。

    参考:
        卡方分布与不完全 gamma 函数的标准关系（Abramowitz & Stegun §26.4）。
    """
    d = _as_int(df, "df", minimum=1)
    xx = float(x)
    if not np.isfinite(xx):
        return 0.0 if xx > 0 else 1.0
    if xx <= 0.0:
        return 1.0
    return float(_gammainc_upper_reg(0.5 * d, 0.5 * xx))


def _arma_loglik(sigma2: float, n_eff: int, n_params: int) -> Tuple[float, float, float]:
    """条件高斯对数似然、AIC 与 BIC。

    参数:
        sigma2: 条件残差方差（MLE 口径，分母为 ``n_eff``）。
        n_eff: 参与条件似然的观测数。
        n_params: 待估参数个数（AR + MA + sigma2）。

    返回:
        ``(loglik, aic, bic)`` 三个 float。

    算法:
        ``loglik = -0.5 * n_eff * (ln 2pi + ln sigma2 + 1)``（高斯 MLE 的对数似然在
        sigma2 取样本均方时取到该值）；``AIC = -2 loglik + 2 k``；
        ``BIC = -2 loglik + ln(n_eff) * k``。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        ``n_eff`` 随 ``(p, q)`` 变化，因此 **AIC 只在同一差分阶数下可比**；跨 ``d`` 比较
        是把"不同数据的似然"放一起比，严格说无意义（``arima_order_select`` 中同样提醒）。

    参考:
        Akaike (1974); Schwarz (1978)。
    """
    ll = -0.5 * float(n_eff) * (math.log(2.0 * math.pi) + math.log(sigma2) + 1.0)
    aic = -2.0 * ll + 2.0 * float(n_params)
    bic = -2.0 * ll + math.log(float(n_eff)) * float(n_params)
    return float(ll), float(aic), float(bic)


def _arma_cls(
    y: np.ndarray,
    ar_lags: Sequence[int],
    ma_lags: Sequence[int],
    max_iter: int = _CLS_MAX_ITER,
    tol: float = _CLS_TOL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """用**条件最小二乘**（CLS）估计（季节）ARMA 的 AR/MA 系数。

    参数:
        y: 已**中心化**的一维序列。
        ar_lags: AR 项滞后集合，例如 ``(1, 2)`` 或 ``(1, 4, 8)``（季节模型）。
        ma_lags: MA 项滞后集合，例如 ``(1,)`` 或 ``(1, 4)``。
        max_iter: 迭代上限（>= 1）。
        tol: 收敛容差：系数向量最大逐分量变化小于它即停。

    返回:
        ``(phi, theta, resid, sigma2, n_eff, n_iter)``：phi/theta 按滞后顺序排列，
        resid 长度与 ``y`` 相同（前 ``t0 = max(lags)`` 位为 0，条件似然不定义它们），
        sigma2 为条件残差方差（分母 ``n_eff``），n_iter 为实际迭代次数
        （0 表示无需迭代的纯白噪声情形）。

    算法:
        把模型写成线性回归 ``y_t = sum_i phi_i y_{t-i} + sum_j theta_j e_{t-j} + e_t``：
        1. 初始残差 ``e = 0``；
        2. 用 ``t = t0..m-1`` 的样本构造设计矩阵（AR 列用真实滞后值、MA 列用当前残差估计），
           解最小二乘得 ``(phi, theta)``；
        3. 用新系数**递归重算**残差 ``e_t = y_t - AR - MA``（t 从 t0 递增）；
        4. 系数变化小于 ``tol`` 即停，否则回到第 2 步。

    复杂度:
        时间 O(n_iter * m * (p + q)^2) / 空间 O(m * (p + q))。

    陷阱:
        - **不是精确 MLE**：它忽略前 ``t0`` 个观测的似然贡献（MA 的初始残差当作 0），
          样本很短或 MA 特征根接近单位圆时与 CSS-ML 差别明显。
        - MA 与残差互相依赖，迭代**可能不收敛**（周期振荡）；此时返回最后一次迭代结果，
          调用方应检查 ``n_iter == max_iter``。实现里不做随机重启以避免不确定性。
          若迭代中系数爆炸使残差溢出为非有限值，则立即抛 ``ValueError``（而不是把 inf/NaN
          送进 ``lstsq`` 换来一句看不懂的 LAPACK 报错）。
        - 残差方差分母是 ``n_eff = m - t0`` 而非 m：与 statsmodels 的 CSS 口径一致，
          但与"整段残差均方"不同，短序列上可差百分之几。
        - 设计矩阵可能病态（AR 与 MA 列高度相关，接近可约简的 ARMA）；用 ``lstsq``（SVD）
          而不是正规方程，正是为了避免显式求逆放大误差。

    参考:
        Box, Jenkins & Reinsel, "Time Series Analysis", 3rd ed., §7.1.2（条件平方和）；
        Hannan & Rissanen (1982)（ARMA 迭代估计原理）。
    """
    m = y.size
    p = len(ar_lags)
    q = len(ma_lags)
    t0 = 0
    if p:
        t0 = max(t0, int(max(ar_lags)))
    if q:
        t0 = max(t0, int(max(ma_lags)))
    if p == 0 and q == 0:
        resid = y.copy()
        sigma2 = float(np.dot(y, y) / m)
        if sigma2 <= 0.0:
            raise ValueError("残差方差为 0（序列为常数），条件似然无定义")
        return np.zeros(0), np.zeros(0), resid, sigma2, m, 0
    if m - t0 <= p + q:
        raise ValueError(
            f"可用于条件最小二乘的观测数 {m - t0} 不足（需大于参数个数 {p + q}）"
        )

    ar_idx = [int(v) for v in ar_lags]
    ma_idx = [int(v) for v in ma_lags]
    e = np.zeros(m, dtype=float)
    beta = np.zeros(p + q, dtype=float)
    rows = m - t0
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        X = np.empty((rows, p + q), dtype=float)
        for i, lag in enumerate(ar_idx):
            X[:, i] = y[t0 - lag: m - lag]
        for j, lag in enumerate(ma_idx):
            X[:, p + j] = e[t0 - lag: m - lag]
        new = np.linalg.lstsq(X, y[t0:], rcond=None)[0]
        e2 = np.zeros(m, dtype=float)
        # MA 递推在系数爆炸时会溢出成 inf/NaN，这里显式校验后再进入下一轮：
        # 否则非有限的设计矩阵会被送进 lstsq（LAPACK 直接报 DLASCL/DGELSD 错误），
        # 结果同样不可用，但错误信息完全指不到原因。
        with np.errstate(over="ignore", invalid="ignore"):
            for t in range(t0, m):
                acc = 0.0
                for i, lag in enumerate(ar_idx):
                    acc += new[i] * y[t - lag]
                for j, lag in enumerate(ma_idx):
                    acc += new[p + j] * e2[t - lag]
                e2[t] = y[t] - acc
        if not np.all(np.isfinite(e2[t0:])) or not np.all(np.isfinite(new)):
            raise ValueError(
                "条件最小二乘迭代发散（MA 系数爆炸导致残差非有限），该阶数组合不可用"
            )
        converged = it > 0 and float(np.max(np.abs(new - beta))) <= tol
        beta, e = new, e2
        if converged:
            break

    n_eff = m - t0
    # 残差虽然有限，但量级可能极大（MA 系数接近发散边界时），点积会先溢出成 inf。
    # 这属于"该阶数组合不可用"的正常情形，交由下面的有限性校验报错，故屏蔽溢出警告。
    with np.errstate(over="ignore", invalid="ignore"):
        sigma2 = float(np.dot(e[t0:], e[t0:]) / n_eff)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("条件最小二乘得到非正残差方差，模型不可用")
    return beta[:p], beta[p:], e, sigma2, n_eff, n_iter


def _psi_weights(ar_lags: Sequence[int], phi: np.ndarray,
                 ma_lags: Sequence[int], theta: np.ndarray, n: int) -> np.ndarray:
    """ARMA 的 ``psi`` 权重（MA(inf) 表示的系数）``psi_0..psi_{n-1}``。

    参数:
        ar_lags / phi: AR 滞后集合与对应系数。
        ma_lags / theta: MA 滞后集合与对应系数。
        n: 需要的权重个数（>= 1）。

    返回:
        长度 n 的数组，``psi[0] = 1``。

    算法:
        ``psi_k = sum_{i in ar_lags, i<=k} phi_i psi_{k-i} + (k in ma_lags ? theta_k : 0)``
        （AR 多项式与 MA 多项式之比的级数展开）。

    复杂度:
        时间 O(n * (p + q)) / 空间 O(n)。

    陷阱:
        AR 特征根接近单位圆时 ``psi_k`` 衰减极慢，长步长预测的标准差近似线性增长
        （随机游走的 psi 恒为 1），这是模型性质而非数值发散。

    参考:
        Box, Jenkins & Reinsel §5.1.1（psi 权重）；Brockwell & Davis §3.3。
    """
    psi = np.zeros(n, dtype=float)
    psi[0] = 1.0
    al = [int(v) for v in ar_lags]
    ml = [int(v) for v in ma_lags]
    for k in range(1, n):
        acc = 0.0
        for i, lag in enumerate(al):
            if lag <= k:
                acc += float(phi[i]) * psi[k - lag]
        if k in ml:
            acc += float(theta[ml.index(k)])
        psi[k] = acc
    return psi


# --------------------------------------------------------------------------
# GM(1,1) 灰色预测
# --------------------------------------------------------------------------

def gm11(x: ArrayLike, n_forecast: int = 1) -> Dict[str, object]:
    """经典 GM(1,1) 灰色预测：一次累加 → 紧邻均值生成 → 最小二乘估参 → 累减还原。

    参数:
        x: 一维原始序列（按时间先后排列，长度 >= 4，否则紧邻均值与残差都不稳）。
        n_forecast: 样本外预测步数（>= 1）。

    返回:
        dict，键为：
        ``a``              发展系数（float，通常为负表示增长）；
        ``b``              灰作用量（float）；
        ``fitted``         形状 (n,) 的拟合值，``fitted[0] = x[0]``（还原公式在 k=0 退化）；
        ``forecast``       形状 (n_forecast,) 的样本外预测，``forecast[h-1]`` 是未来第 h 期；
        ``residual``       形状 (n,) 的残差 ``x - fitted``；
        ``relative_error`` 形状 (n,) 的**百分数**相对误差（与 ``forecasting.mape`` 同口径），
                           ``x[k] == 0`` 处为 ``NaN``。

    算法:
        1. 一次累加 ``X1_k = sum_{i<=k} x_i``（k = 0..n-1，0 基下标）；
        2. 紧邻均值生成 ``z_k = 0.5 (X1_k + X1_{k-1})``（k = 1..n-1）；
        3. 最小二乘解白化方程的离散形式 ``x_k + a z_k = b``：
           ``[a, b]^T = argmin ||B [a,b]^T - Y||^2``，``B = [-z, 1]``、``Y = x_{1..n-1}``；
        4. 时间响应函数 ``X1hat_k = (x_0 - b/a) e^{-a k} + b/a``；
        5. 累减还原 ``xhat_k = X1hat_k - X1hat_{k-1}``（``xhat_0 = x_0``）；
        6. 未来第 h 期 = ``X1hat_{n-1+h} - X1hat_{n-2+h}``。

    复杂度:
        时间 O(n)（含一次 2 列最小二乘，``lstsq`` 走 SVD）/ 空间 O(n)。

    陷阱:
        - **"对纯指数序列精确"是模型形式层面的说法，不是估计层面的**：紧邻均值（梯形）
          离散化与白化方程的连续解不同源，最小二乘估计的 ``a`` 有 O(1/n) 的系统偏差，
          于是纯指数序列 ``x_k = 2 e^{0.3k}`` 的多步预测相对误差约 2%~5% 并随步长累积。
          本模块自测把该真实误差当键返回，不假装它是 1e-6。
        - 只有**由离散白化方程本身生成的数据**（``x_k`` 为比例
          ``(1 - a/2)/(1 + a/2)`` 的几何序列）残差才恒为 0，此时最小二乘能精确还原
          ``(a, b)``——自测就构造这种数据来验证参数恢复。
        - ``a -> 0`` 会让 ``b/a`` 溢出，本实现直接抛 ``ValueError``；经验上 ``|a| < 0.3``
          才适合中长期预测。
        - 紧邻均值固定用 0.5 权重；换成"背景值优化"会得到不同的 ``a``，论文必须写明口径。
        - 原始序列含 0 或负值时相对误差无意义（置 ``NaN``），且灰色模型本身要求非负。
        - n < 4 时最小二乘解方差极大，"预测"几乎等于外推噪声。

    参考:
        邓聚龙 (1982)《灰色系统理论》；刘思峰等《灰色系统理论及其应用》（第 8 版）第 5 章；
        Wang et al. (2010) 关于 GM(1,1) 背景值与参数估计偏差的讨论。
    """
    xv = as_vector(x, "x")
    h = _as_int(n_forecast, "n_forecast", minimum=1)
    n = xv.size
    if n < 4:
        raise ValueError(f"gm11 至少需要 4 个观测，得到 {n}")

    X1 = np.cumsum(xv)
    z = 0.5 * (X1[1:] + X1[:-1])
    B = np.column_stack([-z, np.ones(n - 1, dtype=float)])
    ab = np.linalg.lstsq(B, xv[1:], rcond=None)[0]
    a = float(ab[0])
    b = float(ab[1])
    if not np.isfinite(a) or abs(a) < 1e-12:
        raise ValueError(
            f"发展系数 a={a} 退化为 0，白化方程的时间响应函数不可用（序列近似线性）"
        )
    c = float(xv[0]) - b / a

    k = np.arange(n, dtype=float)
    X1h = c * np.exp(-a * k) + b / a
    fitted = np.empty(n, dtype=float)
    fitted[0] = float(xv[0])
    fitted[1:] = np.diff(X1h)

    kf = np.arange(n, n + h, dtype=float)
    X1f = c * np.exp(-a * kf) + b / a
    full = np.concatenate([X1h, X1f])
    forecast = np.diff(full)[n - 1:]

    residual = xv - fitted
    rel = np.full(n, np.nan, dtype=float)
    nz = xv != 0.0
    rel[nz] = np.abs(residual[nz] / xv[nz]) * 100.0
    return {
        "a": a,
        "b": b,
        "fitted": fitted,
        "forecast": forecast,
        "residual": residual,
        "relative_error": rel,
    }


def gm11_posterior_check(x: ArrayLike, fitted: ArrayLike) -> Dict[str, object]:
    """GM(1,1) 后验差检验：方差比 C、小误差概率 P 与精度等级。

    参数:
        x: 一维原始序列。
        fitted: 与 ``x`` **等长**的拟合序列（通常取 ``gm11(...)["fitted"]``）。

    返回:
        dict，键为：
        ``c_ratio``       float，``C = S2 / S1``（残差标准差 / 原始序列标准差）；
        ``p_small_error`` float，``P = #{|e_k - mean(e)| < 0.6745 S1} / n``；
        ``grade``         str，``"好"``/``"合格"``/``"勉强"``/``"不合格"``。

    算法:
        1. 残差 ``e = x - fitted``；
        2. ``S1 = std(x, ddof=1)``、``S2 = std(e, ddof=1)``（**样本标准差口径**）；
        3. ``C = S2 / S1``；
        4. ``P`` 用样本频率直接计数，阈值 ``0.6745 S1``（正态下约 0.75 分位对应的绝对偏差界）；
        5. 等级判定（两条件同时满足才升档）：
           ``C < 0.35 且 P > 0.95`` → "好"；``C < 0.5 且 P > 0.8`` → "合格"；
           ``C < 0.65 且 P > 0.7`` → "勉强"；否则 "不合格"。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - **口径必须交代**：``S1``/``S2`` 用 ddof=1 还是 ddof=0 会改变 ``C`` 的第四位小数，
          足以在 0.35/0.5/0.65 边界上翻转等级；本实现固定 ddof=1。
        - ``fitted`` 必须与 ``x`` 逐点对应（同一时间轴、同样的差分口径）；若 ``fitted`` 来自
          差分后序列，长度/相位错位会算出"看起来还行"的假象。
        - ``P`` 是样本频率而非渐近概率，n 小时只能取到 1/n 的倍数，等级判定本身很粗糙；
          不要据此声称"模型精度 95%"。
        - 原始序列为常数（``S1 = 0``）时 ``C`` 无定义，本实现抛 ``ValueError``。

    参考:
        邓聚龙 (1982)；刘思峰等《灰色系统理论及其应用》第 5 章（后验差检验表）。
    """
    xv = as_vector(x, "x")
    fv = as_vector(fitted, "fitted")
    check_same_length(xv, fv)
    n = xv.size
    if n < 2:
        raise ValueError(f"后验差检验至少需要 2 个观测，得到 {n}")
    resid = xv - fv
    s1 = float(np.std(xv, ddof=1))
    if s1 <= 0.0:
        raise ValueError("原始序列标准差为 0（常数序列），方差比 C 无定义")
    s2 = float(np.std(resid, ddof=1))
    c_ratio = s2 / s1
    dev = np.abs(resid - float(resid.mean()))
    p_small = float(np.mean(dev < 0.6745 * s1))
    if c_ratio < 0.35 and p_small > 0.95:
        grade = "好"
    elif c_ratio < 0.5 and p_small > 0.8:
        grade = "合格"
    elif c_ratio < 0.65 and p_small > 0.7:
        grade = "勉强"
    else:
        grade = "不合格"
    return {"c_ratio": float(c_ratio), "p_small_error": p_small, "grade": grade}


def difference_series(x: ArrayLike, d: int = 1) -> np.ndarray:
    """d 阶差分，**保持与原序列等长**（前 d 个位置填 ``NaN``）。

    参数:
        x: 一维时间序列。
        d: 差分阶数（>= 0）；``0`` 返回原序列副本。

    返回:
        长度与 ``x`` 相同的数组；``out[k] = x^(d)[k]``（有效区间 ``[d, n-1]``），
        前 ``d`` 个位置为 ``NaN``。

    算法:
        用 ``np.diff(x, n=d)`` 在**有效段**上直接算 d 阶差分（不做任何边缘填充），
        再左对齐放入长度为 n 的数组并前置 ``NaN``。

    复杂度:
        时间 O(d * n) / 空间 O(n)。

    陷阱:
        - 本函数是**对齐口径**（长度不变、前面补 NaN），与模型内部的缩短口径
          （``np.diff`` 返回变短数组）**不同**；混用会造成相位错位。
        - 先补 NaN 再逐阶差分是错的：第一阶差分产生的 NaN 会把后面所有值传染成 NaN。
          本实现刻意在有效段上一次算完 d 阶。
        - 差分会放大噪声并损失 d 个自由度；``d >= n`` 直接抛 ``ValueError``。
        - 返回值中的 ``NaN`` 会污染 ``mean``/``std``，请先用 ``np.isfinite`` 过滤。

    参考:
        Box, Jenkins & Reinsel §4.1（差分算子）；Hyndman & Athanasopoulos §9.1。
    """
    xv = as_vector(x, "x")
    dd = _as_int(d, "d", minimum=0)
    n = xv.size
    if dd >= n:
        raise ValueError(f"差分阶数 d={dd} 必须小于序列长度 {n}")
    out = np.full(n, np.nan, dtype=float)
    if dd == 0:
        out[:] = xv
        return out
    out[dd:] = np.diff(xv, n=dd)
    return out


# --------------------------------------------------------------------------
# ARIMA / SARIMA（条件最小二乘）
# --------------------------------------------------------------------------

def _differenced_tail(xv: np.ndarray, d: int) -> np.ndarray:
    """返回普通 d 阶差分各中间序列的**最后一个值**，供反差分（积分）使用。

    参数:
        xv: 一维原始序列。
        d: 普通差分阶数（>= 0）。

    返回:
        长度 d 的数组 ``tail``，``tail[j]`` 是第 j 阶差分序列的末值（``tail[0] = xv[-1]``）。

    算法:
        逐阶 ``np.diff`` 并记录每阶末值。

    复杂度:
        时间 O(d * n) / 空间 O(n)。

    陷阱:
        反差分只需要"每一阶的末值"这一个数：第 j 阶差分序列的预测由
        ``末值 + 累加下一阶的预测`` 递推得到，因此不必保存整条中间序列（但要求输入序列
        的差分口径与建模时**完全一致**，包括是否先去均值）。

    参考:
        Box, Jenkins & Reinsel §4.1（差分算子的逆）。
    """
    tail = np.empty(d, dtype=float)
    s = xv
    for j in range(d):
        tail[j] = float(s[-1])
        s = np.diff(s)
    return tail


def _build_arma_model(
    w: np.ndarray,
    ar_lags: List[int],
    ma_lags: List[int],
    d: int,
    D: int,
    period: int,
    x_tail: np.ndarray,
) -> Dict[str, object]:
    """中心化 → CLS 估参 → 组装统一的（S）ARIMA 模型字典。

    参数:
        w: 已差分（且已季节差分）的序列。
        ar_lags / ma_lags: AR / MA 的滞后集合（升序）。
        d / D: 普通 / 季节差分阶数（仅记录，用于反差分与文档）。
        period: 季节周期（无季节项时为 0）。
        x_tail: 普通差分各阶末值（见 ``_differenced_tail``）。

    返回:
        模型字典，键为 ``phi``、``theta``、``sigma2``、``loglik``、``aic``、``bic``、
        ``d``、``D``、``period``、``p``、``q``、``mean``、``residual``、``n_eff``、
        ``n_iter``、``ar_lags``、``ma_lags``、``w_tail``、``x_tail``。

    算法:
        减去样本均值后交给 ``_arma_cls`` 做条件最小二乘；参数个数取
        ``len(ar_lags) + len(ma_lags) + 1``（含 sigma2），据此算 AIC/BIC；
        并额外保存最后 ``max(ar_lags)`` 个 w 值（``w_tail``）供预测递推使用。

    复杂度:
        时间 O(n_iter * m * (p + q)^2) / 空间 O(m * (p + q))。

    陷阱:
        - 均值是在**差分后的序列**上减去的：``d > 0`` 时它代表"差分序列的均值"，
          反差分后对应原序列的**线性漂移**，不是水平。别把它当原始均值解释。
        - ``w_tail`` 必须按"最后 max(ar_lags) 个观测"截取；截短了预测递推会取到错误的历史。
        - AIC/BIC 中的 ``n_eff`` 是条件似然的有效样本数（比差分后的长度还少 t0），
          与"原始观测数"不同，写报告时要说清用的是哪一个。

    参考:
        Box, Jenkins & Reinsel §7.1（模型拟合与信息准则）。
    """
    mu = float(w.mean())
    y = w - mu
    phi, theta, resid, sigma2, n_eff, n_iter = _arma_cls(y, ar_lags, ma_lags)
    n_params = len(ar_lags) + len(ma_lags) + 1
    ll, aic, bic = _arma_loglik(sigma2, n_eff, n_params)
    max_ar = max(ar_lags) if ar_lags else 0
    w_tail = w[-max_ar:].copy() if max_ar else np.zeros(0, dtype=float)
    return {
        "phi": phi,
        "theta": theta,
        "sigma2": float(sigma2),
        "loglik": ll,
        "aic": aic,
        "bic": bic,
        "d": int(d),
        "D": int(D),
        "period": int(period),
        "p": int(len(ar_lags)),
        "q": int(len(ma_lags)),
        "mean": mu,
        "residual": resid,
        "n_eff": int(n_eff),
        "n_iter": int(n_iter),
        "ar_lags": [int(v) for v in ar_lags],
        "ma_lags": [int(v) for v in ma_lags],
        "w_tail": w_tail,
        "x_tail": x_tail.copy(),
    }


def arima_fit(x: ArrayLike, p: int = 1, d: int = 0, q: int = 1) -> Dict[str, object]:
    """ARIMA(p, d, q) 的**条件最小二乘**拟合（含均值项与信息准则）。

    参数:
        x: 一维时间序列。
        p: 非季节 AR 阶数（>= 0）。
        d: 普通差分阶数（>= 0，须小于序列长度）。
        q: 非季节 MA 阶数（>= 0）。

    返回:
        模型字典，键为：
        ``phi``/``theta``  AR/MA 系数（长度 p / q 的数组，按滞后升序）；
        ``mean``           差分后序列的均值（d>0 时对应原序列的线性漂移）；
        ``sigma2``         条件残差方差（分母为 n_eff）；
        ``loglik``/``aic``/``bic``  条件高斯对数似然与信息准则；
        ``residual``       长度 m 的条件残差（前 t0 位为 0）；
        ``n_eff``/``n_iter``  有效观测数与 CLS 迭代次数；
        ``d``/``p``/``q``/``D``/``period``  阶数信息（此处 ``D = period = 0``）；
        ``ar_lags``/``ma_lags``/``w_tail``/``x_tail``  预测所需的滞后集合与尾部历史
        （内部约定，供 ``arima_forecast`` 使用）。

    算法:
        1. 做 d 阶差分得 ``w``（长度 ``m = n - d``），并记录每阶末值 ``x_tail``；
        2. ``w`` 中心化后交给 ``_arma_cls`` 做迭代条件最小二乘；
        3. ``loglik = -0.5 n_eff (ln 2pi + ln sigma2 + 1)``，``k = p + q + 1``，
           据此算 AIC/BIC。

    复杂度:
        时间 O(n_iter * m * (p + q)^2) / 空间 O(m * (p + q))。

    陷阱:
        - **估计量是 CLS 不是 MLE**：AR(1) 情形下它等于"去掉第一个观测的 OLS"，
          与 Yule-Walker（``forecasting.ar_model``）的差别来自分母少一项 ``x_0^2``，
          量级 O(1/n) 且依赖最后一个观测的取值，因此**不能承诺任意数据上都一致到 1e-6**；
          自测给出固定算例上的实测差值，并另用独立 OLS 闭式解做精确校验。
        - 均值在差分后序列上估计，等同于假设原序列含**确定性线性漂移**（d=1 时）；
          解释常数项时注意这一点。
        - ``p = q = 0`` 时退化为"差分序列的均值 + 白噪声"，此时 n_iter = 0、phi/theta 为空数组；
          ``forecasting.ar_model`` 不支持这种情形，不要互相替代。
        - 完全线性（残差方差为 0）的数据会抛 ``ValueError``（对数似然无定义），
          ``arima_order_select`` 会捕获并跳过这类组合。
        - 阶数越高 n_eff 越小，AIC 跨不同 ``d`` 不可直接比较（见 ``_arma_loglik``）。

    参考:
        Box, Jenkins & Reinsel §7.1；statsmodels ``ARIMA``（本实现为其 CLS 近似）。
    """
    xv = as_vector(x, "x")
    pp = _as_int(p, "p", minimum=0)
    dd = _as_int(d, "d", minimum=0)
    qq = _as_int(q, "q", minimum=0)
    n = xv.size
    if dd >= n:
        raise ValueError(f"差分阶数 d={dd} 必须小于序列长度 {n}")
    x_tail = _differenced_tail(xv, dd)
    w = xv if dd == 0 else np.diff(xv, n=dd)
    if w.size < 4:
        raise ValueError(f"差分后序列长度 {w.size} 过短，无法拟合 ARIMA")
    ar_lags = list(range(1, pp + 1))
    ma_lags = list(range(1, qq + 1))
    return _build_arma_model(w, ar_lags, ma_lags, dd, 0, 0, x_tail)


def arima_forecast(model: Dict[str, object], n_ahead: int = 1) -> Dict[str, object]:
    """由 ``arima_fit`` / ``sarima_fit`` 的模型字典做多步点预测与预测标准差。

    参数:
        model: ``arima_fit``（或 ``sarima_fit``，但要求 ``D == 0``）返回的字典。
        n_ahead: 预测步数（>= 1）。

    返回:
        dict，键为：
        ``forecast`` 形状 (n_ahead,) 的点预测（已反差分回原序列量纲）；
        ``se``       形状 (n_ahead,) 的预测标准差；
        ``lower``/``upper``  点预测 ± 1.96 se 的正态近似 95% 区间。

    算法:
        1. 递推点预测：``w_hat_h = mu + sum_i phi_i (w_{m+h-i} - mu) + sum_j theta_j e_{m+h-j}``，
           未来残差取 0、历史残差不足处取 0；
        2. 由 ``psi`` 权重构造下三角矩阵 ``Psi[k,j] = psi_{k-j}``；由 d 阶累加算子
           ``C = tril(ones)`` 构造反差分算子 ``M = C^d``；
        3. 误差传播系数 ``c = M @ Psi``，故 ``se_h = sqrt(sigma2 * sum_j c_hj^2)``；
           ``d = 0`` 时退化为教科书公式 ``se_h = sqrt(sigma2 * sum_{l<h} psi_l^2)``，
           ``d = 1`` 且 ``p = q = 0``（随机游走）时退化为 ``sqrt(sigma2 * h)``；
        4. 反差分：从各阶中间序列的末值 ``x_tail[j]`` 起逐阶累加。

    复杂度:
        时间 O(n_ahead^3 + m*(p+q)) / 空间 O(n_ahead^2)。

    陷阱:
        - **只支持 ``D == 0``**：季节差分模型的反差分未实现（点预测的相位对齐需要保存每一阶
          季节差分的尾部窗口），``D > 0`` 时本函数直接抛 ``ValueError``，请自行对
          ``sarima_fit`` 的差分序列累加。
        - 置信区间是**条件正态近似**：忽略参数估计不确定性（mu、phi、theta、sigma2 都当已知）
          与分布非正态性，短序列下实际覆盖率偏低。
        - 只有当 AR 特征根在单位圆内、且 d 与数据生成过程一致时，se 才随步长发散得合理；
          用错 d 会让区间宽窄完全失真。
        - ``w_tail`` 必须来自同一模型字典（长度 >= max(ar_lags)），手工裁剪会静默取错历史。

    参考:
        Box, Jenkins & Reinsel §5.1（预测与 psi 权重）；Hyndman & Athanasopoulos §8.8。
    """
    if not isinstance(model, dict):
        raise ValueError(f"model 必须是 arima_fit 返回的字典，得到 {type(model)!r}")
    required = ("phi", "theta", "sigma2", "residual", "mean",
                "w_tail", "x_tail", "d", "ar_lags", "ma_lags")
    missing = [k for k in required if k not in model]
    if missing:
        raise ValueError(f"model 缺少必要键：{missing}")
    h_len = _as_int(n_ahead, "n_ahead", minimum=1)
    if int(model.get("D", 0)) > 0:
        raise ValueError("arima_forecast 不支持 D > 0 的季节差分模型（反差分未实现）")

    phi = np.asarray(model["phi"], dtype=float).ravel()
    theta = np.asarray(model["theta"], dtype=float).ravel()
    ar_lags = [int(v) for v in model["ar_lags"]]
    ma_lags = [int(v) for v in model["ma_lags"]]
    if phi.size != len(ar_lags) or theta.size != len(ma_lags):
        raise ValueError("model 中 phi/theta 与 ar_lags/ma_lags 长度不一致")
    sigma2 = _as_positive(model["sigma2"], "model['sigma2']", strict=True)
    mu = float(model["mean"])
    dd = _as_int(model["d"], "model['d']", minimum=0)
    resid = np.asarray(model["residual"], dtype=float).ravel()
    w_tail = np.asarray(model["w_tail"], dtype=float).ravel()
    x_tail = np.asarray(model["x_tail"], dtype=float).ravel()
    if x_tail.size != dd:
        raise ValueError(f"model['x_tail'] 长度 {x_tail.size} 与差分阶数 d={dd} 不一致")
    m = resid.size
    max_ar = max(ar_lags) if ar_lags else 0
    if max_ar and w_tail.size < max_ar:
        raise ValueError(
            f"model['w_tail'] 长度 {w_tail.size} 小于最大 AR 滞后 {max_ar}，无法递推预测"
        )

    e_ext = np.concatenate([resid, np.zeros(h_len, dtype=float)])
    hist = [float(v) for v in w_tail]
    w_hat = np.empty(h_len, dtype=float)
    for h in range(1, h_len + 1):
        ar = 0.0
        for lag, c in zip(ar_lags, phi):
            ar += float(c) * (hist[-lag] - mu)
        ma = 0.0
        for lag, c in zip(ma_lags, theta):
            idx = m + h - lag - 1
            if idx >= 0:
                ma += float(c) * e_ext[idx]
        w_hat[h - 1] = mu + ar + ma
        hist.append(w_hat[h - 1])

    psi = _psi_weights(ar_lags, phi, ma_lags, theta, h_len)
    Psi = np.zeros((h_len, h_len), dtype=float)
    for kk in range(1, h_len + 1):
        Psi[kk - 1, :kk] = psi[kk - 1::-1]
    C = np.tril(np.ones((h_len, h_len), dtype=float))
    M = np.eye(h_len, dtype=float)
    for _ in range(dd):
        M = C @ M
    coef = M @ Psi
    se = np.sqrt(sigma2 * np.sum(coef * coef, axis=1))

    cum = w_hat
    for j in range(dd - 1, -1, -1):
        cum = float(x_tail[j]) + np.cumsum(cum)
    fc = cum
    return {
        "forecast": fc,
        "se": se,
        "lower": fc - 1.96 * se,
        "upper": fc + 1.96 * se,
    }


def arima_order_select(
    x: ArrayLike,
    p_max: int = 3,
    d_max: int = 2,
    q_max: int = 3,
    criterion: str = "aic",
) -> Dict[str, object]:
    """在 ``(p, d, q)`` 网格上拟合 ARIMA，按 AIC 或 BIC 选阶。

    参数:
        x: 一维时间序列。
        p_max / d_max / q_max: 各阶数上界（均 >= 0，含上界）。
        criterion: ``"aic"`` 或 ``"bic"``（大小写不敏感）。

    返回:
        dict，键为：
        ``best``  最优组合字典 ``{"p","d","q","aic","bic"}``；
        ``table`` 所有成功拟合的组合（同结构）组成的列表，按 ``(d, p, q)`` 升序排列。

    算法:
        三重循环遍历 ``d = 0..d_max``、``p = 0..p_max``、``q = 0..q_max``，逐个调用
        ``arima_fit`` 收集残差；拟合失败（样本不足、残差方差为 0）的组合被跳过；
        在同一 ``d`` 内把各组合的条件似然**对齐到共同样本**（把起点统一定为该 ``d`` 下
        最大的 ``t0 = max(lags)``）后重算 AIC/BIC；最后按 ``criterion`` 取最小值，
        并列时按 ``(d, p, q)`` 字典序取最小者。

    复杂度:
        时间 O((p_max+1)(d_max+1)(q_max+1) * 单次拟合) / 空间 O(组合数)。

    陷阱:
        - **跨 ``d`` 比较 AIC 理论上不成立**：不同 d 下似然对应不同的数据（差分后序列的
          长度与含义都变了）。这里按惯例仍做全局最小，但更稳妥的做法是先用
          ``forecasting.adf_test`` 定 d，再在固定 d 上按 AIC 选 p、q。
        - 网格搜索**不保证找到全局最优模型**：更大的 p_max 可能选出过拟合的低 AIC 模型；
          也不检查系数显著性与残差白噪声（可配合 ``ljung_box``）。
        - 表内 ``aic``/``bic`` 是本函数**按共同样本重算**的值，与单独调用 ``arima_fit``
          得到的同名键不逐位相等（后者用的是该组合自己的 ``n_eff``）；这是刻意为之——
          各组合的 ``t0 = max(lags)`` 不同，不先对齐样本，AIC 之差里会混进"观测数不同"
          的伪项（每少一个观测约值 ``ln 2pi + ln sigma2 + 1 ~ 2.9`` 个 AIC 点），
          足以把真实阶数选错；对标 R ``arima`` / statsmodels 的"同一样本比较"惯例。
        - 搜索成本是乘积级：``p_max = q_max = 5, d_max = 2`` 要拟合 216 次，谨慎调大。

    参考:
        Hyndman & Athanasopoulos §8.7（AIC 选阶）；Box, Jenkins & Reinsel §7.1.3。
    """
    xv = as_vector(x, "x")
    pm = _as_int(p_max, "p_max", minimum=0)
    dm = _as_int(d_max, "d_max", minimum=0)
    qm = _as_int(q_max, "q_max", minimum=0)
    if not isinstance(criterion, str):
        raise ValueError(f"criterion 必须是字符串，得到 {criterion!r}")
    crit = criterion.strip().lower()
    if crit not in _ORDER_CRITERIA:
        raise ValueError(f"criterion 必须是 {_ORDER_CRITERIA} 之一，得到 {criterion!r}")

    table: List[Dict[str, object]] = []
    for d in range(dm + 1):
        rows_d: List[Dict[str, object]] = []
        for p in range(pm + 1):
            for q in range(qm + 1):
                try:
                    mdl = arima_fit(xv, p, d, q)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                resid = np.asarray(mdl["residual"], dtype=float)
                rows_d.append({
                    "p": int(p),
                    "d": int(d),
                    "q": int(q),
                    "residual": resid,
                    # 条件似然的起点：前 t0 位残差是 0（未参与条件似然），t0 = m - n_eff。
                    "t0": int(resid.size) - int(mdl["n_eff"]),
                    "n_params": int(p) + int(q) + 1,
                })
        if not rows_d:
            continue
        # 同一差分阶数内，把条件似然重新对齐到**共同样本**上再算 AIC/BIC：
        # 各 (p, q) 组合的条件似然起点 t0 = max(lags) 不同，直接用 arima_fit 各自的
        # AIC 比较等于拿"不同长度数据的似然"作比，每丢一个观测就白送约
        # (ln 2pi + ln sigma2 + 1) ≈ 2.9 个 AIC 点，会系统性地偏向/背离某个阶数。
        t0_common = max(int(r["t0"]) for r in rows_d)
        for r in rows_d:
            tail = np.asarray(r["residual"], dtype=float)[t0_common:]
            n_common = int(tail.size)
            s2 = float(np.dot(tail, tail) / n_common)
            if n_common <= 0 or not np.isfinite(s2) or s2 <= 0.0:
                continue
            _ll, aic, bic = _arma_loglik(s2, n_common, int(r["n_params"]))
            table.append({
                "p": int(r["p"]),
                "d": int(r["d"]),
                "q": int(r["q"]),
                "aic": float(aic),
                "bic": float(bic),
            })
    if not table:
        raise ValueError(
            f"在 p<={pm}, d<={dm}, q<={qm} 的网格上没有任何可拟合的 ARIMA 模型"
            "（可能是序列过短或为常数）"
        )
    best = min(table, key=lambda r: (float(r[crit]), int(r["d"]), int(r["p"]), int(r["q"])))
    return {"best": dict(best), "table": table}


def sarima_fit(
    x: ArrayLike,
    period: int,
    p: int = 1,
    d: int = 0,
    q: int = 1,
    P: int = 1,
    D: int = 1,
    Q: int = 0,
) -> Dict[str, object]:
    """SARIMA(p,d,q)(P,D,Q)_s 的**可加季节项**条件最小二乘拟合。

    参数:
        x: 一维时间序列。
        period: 季节周期 ``s``（>= 2）。
        p / d / q: 非季节 AR / 差分 / MA 阶数（>= 0）。
        P / D / Q: 季节 AR / 差分 / MA 阶数（>= 0）。

    返回:
        模型字典，键与 ``arima_fit`` **一致**，另外包含：
        ``period``、``D``、``P``、``Q``、``seasonal_phi``（长度 P）、``seasonal_theta``（长度 Q）。
        ``phi``/``theta`` 只装**非季节**系数，季节系数单独放在 ``seasonal_phi`` /
        ``seasonal_theta`` 中；``ar_lags``/``ma_lags`` 是两者合并后的完整滞后集合。

    算法:
        1. 先做 D 次季节差分 ``w = w[s:] - w[:-s]``，再做 d 次普通差分；
        2. 把（S）ARMA 展开成**对滞后项的可加线性回归**：
           AR 滞后集合 ``{1..p} ∪ {s, 2s, ..., Ps}``，MA 滞后集合 ``{1..q} ∪ {s, ..., Qs}``；
        3. 用与 ``arima_fit`` 相同的迭代 CLS（``_arma_cls``）估计全部系数。

    复杂度:
        时间 O(n_iter * m * (p+q+P+Q)^2) / 空间 O(m * (p+q+P+Q))。

    陷阱:
        - **这是简化版**：真正的 SARIMA 季节多项式与 AR/MA 多项式是**乘积**关系
          （``(1 - phi B)(1 - Phi B^s)``），本实现把它们当**可加**滞后项合并回归。
          当 ``(P, Q)`` 与 ``(p, q)`` 同时非零时两者不等价，交叉项（如滞后 ``s+1``）会被漏掉；
          自测因此只用乘积与可加一致的情形（``p = q = 0`` 或 ``P = Q = 0``）做校验。
        - 一次季节差分就损失 s 个观测，``D = 1`` 且 s 较大时可用样本骤减，
          样本不足会抛 ``ValueError``。
        - 反差分：``arima_forecast`` 不支持 ``D > 0``（见其陷阱说明）。
        - ``P = D = Q = 0`` 时本函数与 ``arima_fit(x, p, d, q)`` 走**同一条代码路径**，
          结果逐位相同（自测据此做交叉验证）。

    参考:
        Box, Jenkins & Reinsel §9.2（季节模型）；Hyndman & Athanasopoulos §8.9。
    """
    xv = as_vector(x, "x")
    s = _as_int(period, "period", minimum=2)
    pp = _as_int(p, "p", minimum=0)
    dd = _as_int(d, "d", minimum=0)
    qq = _as_int(q, "q", minimum=0)
    PP = _as_int(P, "P", minimum=0)
    DD = _as_int(D, "D", minimum=0)
    QQ = _as_int(Q, "Q", minimum=0)

    w = xv
    for _ in range(DD):
        if w.size <= s:
            raise ValueError(f"序列长度 {w.size} 不足以做季节周期 {s} 的差分")
        w = w[s:] - w[:-s]
    if dd >= w.size:
        raise ValueError(f"差分阶数 d={dd} 必须小于季节差分后的长度 {w.size}")
    x_tail = _differenced_tail(w, dd)
    w = w if dd == 0 else np.diff(w, n=dd)
    if w.size < 4:
        raise ValueError(f"差分后序列长度 {w.size} 过短，无法拟合 SARIMA")

    ar_lags = list(range(1, pp + 1)) + [s * k for k in range(1, PP + 1)]
    ma_lags = list(range(1, qq + 1)) + [s * k for k in range(1, QQ + 1)]
    model = _build_arma_model(w, ar_lags, ma_lags, dd, DD, s, x_tail)
    model["p"] = int(pp)
    model["q"] = int(qq)
    model["P"] = int(PP)
    model["Q"] = int(QQ)
    phi = np.asarray(model["phi"], dtype=float)
    theta = np.asarray(model["theta"], dtype=float)
    model["seasonal_phi"] = phi[pp:].copy()
    model["seasonal_theta"] = theta[qq:].copy()
    return model


# --------------------------------------------------------------------------
# GARCH(1,1)
# --------------------------------------------------------------------------

def _garch_eval(
    params: np.ndarray, r: np.ndarray, s2_init: float
) -> Tuple[float, np.ndarray, float]:
    """GARCH(1,1) 的正态负对数似然（给定参数）。

    参数:
        params: ``(omega, alpha, beta)``。
        r: 已去均值的收益率序列。
        s2_init: ``sigma2_0`` 的初值（取样本方差）。

    返回:
        ``(neg_loglik, sigma2_path, sigma2_next)``：sigma2_path 长度与 ``r`` 相同
        （``sigma2_path[t]`` 是 ``r_t`` 的条件方差），``sigma2_next`` 是下一次观测的
        **一步向前**条件方差。

    算法:
        ``sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1}``（``sigma2_0`` 用初值），
        逐项累加 ``-0.5 (ln 2pi + ln sigma2_t + r_t^2 / sigma2_t)``。

    复杂度:
        时间 O(T) / 空间 O(T)。

    陷阱:
        - 参数必须已满足 ``omega > 0、alpha, beta >= 0、alpha + beta < 1``；
          本函数不做校验（由调用方在候选点上过滤），传入非法参数会得到 NaN/Inf。
        - 初值 ``sigma2_0`` 的选择影响前若干项似然；T 较小时不同软件（用无条件方差还是
          用第一期的样本方差）会给出略微不同的估计。
        - 对数似然里用的是**正态**假定，而收益率常有厚尾；正态 QML 仍给出相合估计，
          但标准误需要稳健（Bollerslev-Wooldridge）修正，本模块不提供。

    参考:
        Bollerslev (1986)；Engle (1982)；Berndt et al. (1974)（数值优化口径）。
    """
    omega = float(params[0])
    alpha = float(params[1])
    beta = float(params[2])
    t_len = r.size
    path = np.empty(t_len, dtype=float)
    s2 = float(s2_init)
    nll = 0.0
    for t in range(t_len):
        path[t] = s2
        nll += -0.5 * (math.log(2.0 * math.pi) + math.log(s2) + r[t] * r[t] / s2)
        s2 = omega + alpha * r[t] * r[t] + beta * s2
    return -float(nll), path, float(s2)


def _garch_feasible(params: np.ndarray) -> bool:
    """判断 GARCH 参数是否落在可估区域内。

    参数:
        params: ``(omega, alpha, beta)``。

    返回:
        bool。

    算法:
        要求 ``omega > 0``、``alpha >= 0``、``beta >= 0`` 且
        ``alpha + beta <= _GARCH_PERSIST_MAX``（默认 0.999），即方差过程平稳且非退化。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        把上界设成 0.999 而不是 1 是为了避免 IGARCH 边界上似然无界、参数被推到边界；
        这意味着**真实持续性接近 1 的数据会被系统性低估**，报告里必须说明。

    参考:
        Bollerslev (1986)（平稳性条件 ``alpha + beta < 1``）。
    """
    omega = float(params[0])
    alpha = float(params[1])
    beta = float(params[2])
    return omega > 0.0 and alpha >= 0.0 and beta >= 0.0 and (alpha + beta) <= _GARCH_PERSIST_MAX


def garch11_fit(
    r: ArrayLike, max_iter: int = 500, seed: Optional[int] = None
) -> Dict[str, object]:
    """GARCH(1,1) 的正态拟极大似然（QML）估计，用**模式搜索**求最大值。

    参数:
        r: 一维收益率序列（本函数**内部去均值**，不需要调用方预处理）。
        max_iter: 模式搜索的最大外层迭代次数（>= 1）。
        seed: 随机多起点的种子；``None`` 表示使用 ``_common.DEFAULT_SEED``。

    返回:
        dict，键为：
        ``omega``/``alpha``/``beta``  参数估计（float）；
        ``persistence``   ``alpha + beta``（波动率持续性的常用度量）；
        ``sigma2``        形状 (T,) 的条件方差序列，``sigma2[t]`` 是 ``r_t`` 的条件方差，
                          ``sigma2[0]`` 是初值（去均值序列的样本方差）；
        ``sigma2_next``   下一期（样本外第一步）的一步向前条件方差；
        ``loglik``        最大化的正态对数似然；
        ``n_iter``        实际外层迭代次数；
        ``mean``          被减掉的样本均值。

    算法:
        1. 去均值 ``r <- r - mean(r)``，取 ``sigma2_0 = var(r, ddof=1)``；
        2. 从若干**确定性起点**（``(0.05 var, 0.05, 0.90)``、``(0.10 var, 0.10, 0.80)``）
           与若干**随机起点**（``rng(seed)`` 抽取、经可行性过滤）出发；
        3. 每个起点做**坐标模式搜索**：依次沿 omega/alpha/beta 的正负方向试探，
           omega 的步长按样本方差缩放；无改进则步长减半，直到步长 < 1e-6 或达到 max_iter；
        4. 取各起点中似然最大的参数，并返回该点的条件方差路径。

    复杂度:
        时间 O(n_starts * n_iter * T) / 空间 O(T)。

    陷阱:
        - 优化器是**模式搜索**（无导数、收敛慢），不是 BFGS；似然面在 ``alpha``/``beta``
          高度相关时它是"沿坐标轴走"，可能提前停住。多起点是为了缓解这一点，但**不保证**
          全局最优，与 R 的 ``rugarch``/``fGarch`` 数值可能有百分之几的差异。
        - 内部**去均值**：返回值 ``mean`` 是样本均值。若数据有明显漂移，等价于先减常数，
          与"含常数项的 GARCH-M/AR 均值方程"不同。
        - 约束 ``alpha + beta <= 0.999`` 会把接近 IGARCH 的真实过程压向边界内（见
          ``_garch_feasible``）；``omega`` 被约束为正，因此无法表示"零方差"退化情形。
        - ``sigma2[0]`` 是初值不是估计值，序列的前几项受初值影响；T 很小时别把
          ``sigma2`` 的头几个值直接当"波动率估计"。
        - 正态 QML 对厚尾数据给出相合但非有效的估计，标准误需稳健修正，本模块不提供。

    参考:
        Bollerslev (1986)；Engle (1982)；Shephard & Sheppard (2010)
        "Realising the future: forecasting with high-frequency-based volatility models"。
    """
    rv = as_vector(r, "r")
    it_max = _as_int(max_iter, "max_iter", minimum=1)
    if rv.size < 20:
        raise ValueError(f"garch11_fit 至少需要 20 个观测，得到 {rv.size}")
    mu = float(rv.mean())
    xc = rv - mu
    var = float(np.var(xc, ddof=1))
    if not np.isfinite(var) or var <= 0.0:
        raise ValueError("收益率序列方差为 0（常数序列），GARCH 无法估计")

    gen = rng(seed)
    starts: List[np.ndarray] = [
        np.array([max(var * 0.05, 1e-12), 0.05, 0.90], dtype=float),
        np.array([max(var * 0.10, 1e-12), 0.10, 0.80], dtype=float),
    ]
    for _ in range(2):
        for _try in range(20):
            cand = np.array([
                max(var * float(gen.uniform(0.01, 0.20)), 1e-12),
                float(gen.uniform(0.01, 0.30)),
                float(gen.uniform(0.50, 0.95)),
            ], dtype=float)
            if _garch_feasible(cand):
                starts.append(cand)
                break

    step0 = 0.2
    tol = 1e-6
    best: Optional[Tuple[float, np.ndarray, np.ndarray, float, int]] = None
    for x0 in starts:
        p_vec = x0.copy()
        nll, path, s2_next = _garch_eval(p_vec, xc, var)
        step = step0
        n_iter = 0
        while step > tol and n_iter < it_max:
            n_iter += 1
            improved = False
            for i in range(3):
                for sgn in (1.0, -1.0):
                    cand = p_vec.copy()
                    cand[i] += sgn * step * (var if i == 0 else 1.0)
                    if not _garch_feasible(cand):
                        continue
                    f_new, path_new, s2_new = _garch_eval(cand, xc, var)
                    if f_new < nll - 1e-12:
                        p_vec, nll, path, s2_next = cand, f_new, path_new, s2_new
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
        if best is None or nll < best[0]:
            best = (nll, p_vec, path, s2_next, n_iter)
    if best is None:
        raise ValueError("GARCH 模式搜索未能得到任何可行参数（序列可能退化）")
    nll, p_best, path, s2_next, n_iter = best
    omega, alpha, beta = float(p_best[0]), float(p_best[1]), float(p_best[2])
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "sigma2": path,
        "sigma2_next": float(s2_next),
        "loglik": -float(nll),
        "n_iter": int(n_iter),
        "mean": mu,
    }


def garch11_forecast(model: Dict[str, object], n_ahead: int = 1) -> Dict[str, object]:
    """GARCH(1,1) 的多步条件方差预测（方差均值回复公式）。

    参数:
        model: ``garch11_fit`` 返回的字典。
        n_ahead: 预测步数（>= 1）。

    返回:
        dict，键为：
        ``variance``         形状 (n_ahead,) 的条件方差预测，第 h 项对应未来第 h 期
                             （``variance[0] == model["sigma2_next"]``）；
        ``volatility``       其平方根（条件标准差）；
        ``long_run_variance``  长期方差 ``omega / (1 - alpha - beta)``。

    算法:
        对 GARCH(1,1) 有 ``E[sigma2_{T+h} | F_T] = LR + (alpha+beta)^{h-1} (E[sigma2_{T+1}|F_T] - LR)``
        （h >= 1），其中 ``LR = omega/(1-persistence)``，而基准项
        ``E[sigma2_{T+1}|F_T]`` 正是 ``garch11_fit`` 给出的 **下一期一步向前方差**
        ``model["sigma2_next"] = omega + alpha*eps_T^2 + beta*sigma2_T``。
        因此 ``variance[0] == model["sigma2_next"]``（h = 1 时指数为 0）。
        直接向量化计算 h = 1..n_ahead。

    复杂度:
        时间 O(n_ahead) / 空间 O(n_ahead)。

    陷阱:
        - **指数是 h-1 而不是 h**：基准必须是一步向前方差 ``sigma2_next``。若把时间 T
          **已实现**的条件方差 ``sigma2_T`` 当基准传进来（``garch11_fit`` 也返回该键），
          整条预测曲线会整体前移一期，第 1 项将被错误地当成 h = 2 的结果。
        - 该公式是**条件方差的期望**，不是"波动率的期望"：``E[sigma_{T+h}] <= sqrt(E[sigma2])``
          （Jensen 不等式），所以 ``volatility`` 是下偏的。
        - 只对线性 GARCH(1,1) 严格成立；换成 GJR/EGARCH 或多变量模型，均值回复形式不同。
        - ``persistence`` 越接近 1 回复越慢，长期方差对参数误差极敏感（分母 ``1-persistence``
          很小），因此 ``long_run_variance`` 的不确定性远大于点估计值本身。
        - 模型字典必须含 ``sigma2_next``；若手工构造而漏了它，本函数抛 ``ValueError``。

    参考:
        Bollerslev (1986) §2；Tsay, "Analysis of Financial Time Series", §3.4。
    """
    if not isinstance(model, dict):
        raise ValueError(f"model 必须是 garch11_fit 返回的字典，得到 {type(model)!r}")
    for key in ("omega", "alpha", "beta", "sigma2_next"):
        if key not in model:
            raise ValueError(f"model 缺少必要键 {key!r}")
    h_len = _as_int(n_ahead, "n_ahead", minimum=1)
    omega = _as_positive(model["omega"], "model['omega']", strict=True)
    alpha = _as_positive(model["alpha"], "model['alpha']")
    beta = _as_positive(model["beta"], "model['beta']")
    s2_next = _as_positive(model["sigma2_next"], "model['sigma2_next']", strict=True)
    persistence = alpha + beta
    if persistence >= 1.0:
        raise ValueError(f"alpha + beta = {persistence} >= 1，方差过程非平稳，无长期方差")
    long_run = omega / (1.0 - persistence)
    # 指数 h-1：variance[0] 必须等于一步向前方差 sigma2_next（E[sigma2_{T+1}|F_T]）。
    hh = np.arange(0, h_len, dtype=float)
    var_h = long_run + np.power(persistence, hh) * (s2_next - long_run)
    return {
        "variance": var_h,
        "volatility": np.sqrt(var_h),
        "long_run_variance": float(long_run),
    }


# --------------------------------------------------------------------------
# 卡尔曼滤波
# --------------------------------------------------------------------------

def kalman_filter_local_level(
    y: ArrayLike, q: float = 1.0, r: float = 1.0
) -> Dict[str, object]:
    """局部水平模型（随机游走 + 观测噪声）的标量卡尔曼滤波与似然。

    参数:
        y: 一维观测序列。
        q: 状态噪声方差（> 0），即 ``x_t = x_{t-1} + w_t``，``w_t ~ N(0, q)``。
        r: 观测噪声方差（> 0），即 ``y_t = x_t + v_t``，``v_t ~ N(0, r)``。

    返回:
        dict，键为：
        ``filtered``           形状 (n,) 的滤波均值 ``E[x_t | y_{1..t}]``；
        ``predicted``          形状 (n,) 的一步向前预测均值 ``E[x_t | y_{1..t-1}]``，
                               ``predicted[0] = y[0]``（初值）；
        ``variance``           形状 (n,) 的滤波方差；
        ``predicted_variance`` 形状 (n,) 的预测方差；
        ``gain``               形状 (n,) 的卡尔曼增益 ``K_t = P_pred / (P_pred + r)``；
        ``loglik``             预测误差分解得到的高斯对数似然。

    算法:
        初值 ``x_0 = y[0]``，状态的先验方差取 ``P_0 = 0``（即把第一个观测当成已知常数，
        实现上直接令第一步的预测方差 ``p_pred = q``，等价于 ``P_0 + q``），随后对每个 t：
        预测 ``P_pred = P_{t-1} + q``；增益 ``K = P_pred/(P_pred + r)``；
        更新 ``x_t = x_pred + K (y_t - x_pred)``、``P_t = (1 - K) P_pred``；
        似然累加 ``-0.5 (ln 2pi + ln(P_pred + r) + innovation^2/(P_pred + r))``。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - **初值是"先验"而不是"估计"**：``x_0 = y[0]`` 使第一个观测的新息恰好为 0
          （注意其方差仍是 ``P_pred + r = q + r``，不是 0），等价于把 ``y[0]`` 当成已知常数。
          这不是标准 MLE 的似然（少了一个自由参数），
          与 ``statsmodels.UnobservedComponents`` 的默认初值处理不同，对数似然**不可直接比较**。
        - ``r -> 0`` 时滤波值收敛到观测值（``K -> 1``），此时似然会发散到 +inf；
          本函数对 ``r`` 只要求 > 0，极小 ``r`` 下 ``loglik`` 数值上很大是正常的。
        - 该模型只能拟合"水平缓慢漂移"的序列；有明显趋势/季节时必须改用带斜率或季节分量的
          状态空间模型（可用 ``kalman_filter_linear`` 自行组装）。
        - **没有平滑**（``E[x_t | y_{1..n}]``）：本函数只做滤波，平滑需要后向递推，未实现。

    参考:
        Kalman (1960)；Harvey, "Forecasting, Structural Time Series Models and the
        Kalman Filter", §3.2；Durbin & Koopman, "Time Series Analysis by State Space
        Methods", §2.3。
    """
    yv = as_vector(y, "y")
    qq = _as_positive(q, "q", strict=True)
    rr = _as_positive(r, "r", strict=True)
    n = yv.size
    pred = np.empty(n, dtype=float)
    filt = np.empty(n, dtype=float)
    pv = np.empty(n, dtype=float)
    ppred = np.empty(n, dtype=float)
    gain = np.empty(n, dtype=float)
    x_pred = float(yv[0])
    p_pred = qq
    loglik = 0.0
    for t in range(n):
        pred[t] = x_pred
        ppred[t] = p_pred
        s_innov = p_pred + rr
        k_gain = p_pred / s_innov
        gain[t] = k_gain
        innov = float(yv[t]) - x_pred
        filt[t] = x_pred + k_gain * innov
        pv[t] = (1.0 - k_gain) * p_pred
        loglik += -0.5 * (math.log(2.0 * math.pi) + math.log(s_innov) + innov * innov / s_innov)
        if t + 1 < n:
            x_pred = filt[t]
            p_pred = pv[t] + qq
    return {
        "filtered": filt,
        "predicted": pred,
        "variance": pv,
        "predicted_variance": ppred,
        "gain": gain,
        "loglik": float(loglik),
    }


def kalman_filter_linear(
    y: ArrayLike,
    F: MatrixLike,
    H: MatrixLike,
    Q: MatrixLike,
    R: MatrixLike,
    x0: ArrayLike,
    P0: MatrixLike,
) -> Dict[str, object]:
    """一般线性高斯状态空间模型的卡尔曼滤波（多状态、多观测通道）。

    参数:
        y: 观测值。一维 (T,) 表示单通道；二维 (T, m) 表示 m 个通道。
        F: 状态转移矩阵，形状 (k, k)。
        H: 观测矩阵，形状 (m, k)（传一维 ``(k,)`` 会被当成 1 行）。
        Q: 状态噪声协方差，形状 (k, k)。
        R: 观测噪声协方差，形状 (m, m)。
        x0: 初始状态均值，形状 (k,)。
        P0: 初始状态协方差，形状 (k, k)。

    返回:
        dict，键为：
        ``state``       (T, k) 的滤波状态均值 ``E[alpha_t | y_{1..t}]``；
        ``cov``         (T, k, k) 的滤波协方差；
        ``predicted``   (T, k) 的一步向前预测均值；
        ``innovation``  (T, m) 的新息 ``y_t - H alpha_t^{pred}``；
        ``loglik``      预测误差分解的高斯对数似然（含 ``-0.5 m ln 2pi`` 与 ``logdet S_t``）。

    算法:
        标准 Kalman 递推（前向）：
        1. 预测 ``a = F x``、``P = F P F' + Q``；
        2. 新息 ``v = y_t - H a``、``S = H P H' + R``；
        3. 增益 ``K = P H' S^{-1}``；
        4. 更新 ``x = a + K v``、``P = P - K H P``；
        5. 似然累加 ``-0.5 (m ln 2pi + ln|S| + v' S^{-1} v)``。

    复杂度:
        时间 O(T (k^3 + m k^2 + m^3)) / 空间 O(T (k^2 + m))。

    陷阱:
        - ``S`` 用显式求逆（``np.linalg.inv``），数值上不如 Cholesky 稳定；``R`` 接近奇异或
          状态维度很高时应改用平方根滤波，本实现不做。
        - 协方差更新用的是 ``P - K H P``（而非 Joseph 形式），两者数学等价但前者在极端
          病态问题下可能失去对称正定性；``np.linalg.slogdet`` 报出的符号 ``sign <= 0``
          （即行列式为 0 或为负，包含 `logdet = -inf` 的奇异情形）时本函数抛 ``ValueError``。
        - 初值 ``x0``/``P0`` 是**先验**：本函数在 t=0 先预测再加 Q（即
          ``P_pred(0) = F P0 F' + Q``），因此它与 ``kalman_filter_local_level`` 完全一致
          当且仅当取 ``P0 = 0``（见后者的陷阱说明）。
        - 不做平滑、不做缺失值处理（``y`` 中不能有 NaN，``as_vector``/本函数都会拒绝）。

    参考:
        Kalman (1960)；Durbin & Koopman §4.2；Särkkä, "Bayesian Filtering and Smoothing",
        Ch. 4。
    """
    Fm = as_matrix(F, "F")
    k = Fm.shape[0]
    if Fm.shape[1] != k:
        raise ValueError(f"F 必须是方阵，得到形状 {Fm.shape}")
    Hm = as_matrix(H, "H")
    if Hm.shape[1] != k:
        raise ValueError(f"H 的列数必须等于状态维数 {k}，得到形状 {Hm.shape}")
    m_obs = Hm.shape[0]
    Qm = as_matrix(Q, "Q")
    if Qm.shape != (k, k):
        raise ValueError(f"Q 必须是 ({k}, {k})，得到 {Qm.shape}")
    Rm = as_matrix(R, "R")
    if Rm.shape != (m_obs, m_obs):
        raise ValueError(f"R 必须是 ({m_obs}, {m_obs})，得到 {Rm.shape}")
    x0v = as_vector(x0, "x0")
    if x0v.size != k:
        raise ValueError(f"x0 长度必须为 {k}，得到 {x0v.size}")
    P0m = as_matrix(P0, "P0")
    if P0m.shape != (k, k):
        raise ValueError(f"P0 必须是 ({k}, {k})，得到 {P0m.shape}")

    yv = np.asarray(y, dtype=float)
    if yv.ndim == 1:
        obs = yv.reshape(-1, 1)
    elif yv.ndim == 2:
        obs = yv
    else:
        raise ValueError(f"y 必须是一维或二维数组，得到 ndim={yv.ndim}")
    if obs.size == 0:
        raise ValueError("y 不能为空")
    if not np.all(np.isfinite(obs)):
        raise ValueError("y 含 NaN 或 inf（本实现不支持缺失观测）")
    t_len = obs.shape[0]
    if obs.shape[1] != m_obs:
        raise ValueError(
            f"y 的列数 {obs.shape[1]} 与 H 的行数（观测维数）{m_obs} 不一致"
        )

    state = np.empty((t_len, k), dtype=float)
    predicted = np.empty((t_len, k), dtype=float)
    cov = np.empty((t_len, k, k), dtype=float)
    innov = np.empty((t_len, m_obs), dtype=float)
    eye_k = np.eye(k)
    x_cur = x0v.copy()
    p_cur = P0m.copy()
    loglik = 0.0
    for t in range(t_len):
        a_pred = Fm @ x_cur
        p_pred = Fm @ p_cur @ Fm.T + Qm
        v = obs[t] - Hm @ a_pred
        s_mat = Hm @ p_pred @ Hm.T + Rm
        sign, logdet = np.linalg.slogdet(s_mat)
        if sign <= 0.0:
            raise ValueError(
                f"t={t} 时新息协方差 S 非正定（logdet={logdet}），检查 R/Q/H 的设定"
            )
        s_inv = np.linalg.inv(s_mat)
        k_gain = p_pred @ Hm.T @ s_inv
        x_cur = a_pred + k_gain @ v
        p_cur = p_pred - k_gain @ Hm @ p_pred
        predicted[t] = a_pred
        innov[t] = v
        state[t] = x_cur
        cov[t] = p_cur
        loglik += -0.5 * (
            m_obs * math.log(2.0 * math.pi) + logdet + float(v @ s_inv @ v)
        )
    return {
        "state": state,
        "cov": cov,
        "predicted": predicted,
        "innovation": innov,
        "loglik": float(loglik),
    }


def kalman_smoother_linear(
    y: ArrayLike,
    transition: MatrixLike,
    observation: MatrixLike,
    process_cov: MatrixLike,
    obs_cov: MatrixLike,
    initial_state: ArrayLike,
    initial_cov: MatrixLike,
) -> Dict[str, object]:
    """一般线性高斯状态空间模型的卡尔曼滤波 + RTS 固定区间平滑。

    参数:
        y: 观测值。一维 (T,) 表示单通道；二维 (T, m) 表示 m 个通道。
        transition: 状态转移矩阵 F，形状 (k, k)。
        observation: 观测矩阵 H，形状 (m, k)（传一维 ``(k,)`` 会被当成 1 行）。
        process_cov: 状态噪声协方差 Q，形状 (k, k)。
        obs_cov: 观测噪声协方差 R，形状 (m, m)。
        initial_state: 初始状态均值 x0，形状 (k,)，解释为 t=0 的**先验**。
        initial_cov: 初始状态协方差 P0，形状 (k, k)。

    返回:
        dict，键为：
        ``filtered_states``   (T, k) 滤波均值 ``E[alpha_t | y_{1..t}]``；
        ``filtered_covs``     (T, k, k) 滤波协方差 ``Var[alpha_t | y_{1..t}]``；
        ``smoothed_states``   (T, k) 平滑均值 ``E[alpha_t | y_{1..T}]``（用到全样本）；
        ``smoothed_covs``     (T, k, k) 平滑协方差 ``Var[alpha_t | y_{1..T}]``；
        ``log_likelihood``    预测误差分解的高斯对数似然（标量，与
                              ``kalman_filter_linear`` 的 ``loglik`` 完全一致）；
        ``predicted_states``  (T, k) 一步向前预测均值 ``E[alpha_t | y_{1..t-1}]``；
        ``predicted_covs``    (T, k, k) 一步向前预测协方差 ``F P_{t-1} F' + Q``
                              （``predicted_covs[0] = F P0 F' + Q``，与滤波器的先验口径一致）；
        ``innovations``       (T, m) 新息 ``y_t - H alpha_t^{pred}``；
        ``smoother_gain``     (T-1, k, k) RTS 平滑增益 ``J_t``（``T == 1`` 时为空数组）。

    算法:
        前向滤波与 ``kalman_filter_linear`` 完全相同（直接复用该函数，保证两者口径逐位一致）：
        预测 ``a_t = F x_{t-1}``、``P_t = F P_{t-1} F' + Q``，更新
        ``x_t = a_t + K_t v_t``、``P_t = P_t^{pred} - K_t H P_t^{pred}``。
        反向平滑（Rauch–Tung–Striebel, RTS）从 ``t = T-1`` 往前递推，初值
        ``x^s_{T-1} = x_{T-1}``、``P^s_{T-1} = P_{T-1}``：
        ``J_t = P_t F' (P_{t+1}^{pred})^{-1}``；
        ``x^s_t = x_t + J_t (x^s_{t+1} - a_{t+1})``；
        ``P^s_t = P_t + J_t (P^s_{t+1} - P_{t+1}^{pred}) J_t'``。
        由于前向滤波已经算过一次，``P_{t+1}^{pred}`` 在第 2 步独立重算（不重复滤波循环）。
        对 ``P^s_t`` 做强制对称化 ``(X + X')/2`` 以抑制舍入误差。

    复杂度:
        时间 O(T (k^3 + m k^2 + m^3)) / 空间 O(T (k^2 + m))。

    陷阱:
        - **初值口径**：``initial_state``/``initial_cov`` 是先验，t=0 先预测再加 Q（即
          ``P_pred(0) = F P0 F' + Q``），与 ``kalman_filter_linear`` 完全一致；要与
          ``kalman_filter_local_level`` 对齐必须取 ``P0 = 0``（见那两个函数的陷阱说明）。
        - 平滑协方差**在 Loewner（半正定）序下**才保证 ``P^s_t <= P_t``：对角元（各分量的
          后验方差）逐元素满足，非对角元（协方差）不一定逐元素变小；不要拿"元素逐个变小"
          当通用结论。
        - ``P_{t+1}^{pred}`` 需要求逆；``Q = 0`` 且 ``P0`` 退化时它可能奇异，此时本函数抛
          ``ValueError``（与滤波器的 ``S`` 非正定检查同口径），不做伪逆回退。
        - 平滑用到全样本，因此**不能用于实时在线预测**：``smoothed_states`` 在每个时刻都用
          到了该时刻之后的数据，把它当"预测值"画图会严重高估精度。
        - 不做缺失值处理（``y`` 中不能有 NaN）。``T == 1`` 时平滑结果等于滤波结果，
          ``smoother_gain`` 为空数组。

    参考:
        Rauch, Tung & Striebel (1965)；Durbin & Koopman §4.3；Särkkä, Ch. 8。
    """
    filt = kalman_filter_linear(
        y, transition, observation, process_cov, obs_cov, initial_state, initial_cov
    )
    f_state = np.asarray(filt["state"], dtype=float)
    f_cov = np.asarray(filt["cov"], dtype=float)
    pred_state = np.asarray(filt["predicted"], dtype=float)
    innov = np.asarray(filt["innovation"], dtype=float)
    t_len, k = f_state.shape

    Fm = as_matrix(transition, "transition")
    Qm = as_matrix(process_cov, "process_cov")
    P0m = as_matrix(initial_cov, "initial_cov")

    # 重算一步向前预测协方差：P_pred(0) = F P0 F' + Q，P_pred(t) = F P_f(t-1) F' + Q
    pred_cov = np.empty((t_len, k, k), dtype=float)
    p_prev = P0m
    for t in range(t_len):
        pred_cov[t] = Fm @ p_prev @ Fm.T + Qm
        p_prev = f_cov[t]

    s_state = np.empty((t_len, k), dtype=float)
    s_cov = np.empty((t_len, k, k), dtype=float)
    gain = np.empty((max(t_len - 1, 0), k, k), dtype=float)
    s_state[t_len - 1] = f_state[t_len - 1]
    s_cov[t_len - 1] = f_cov[t_len - 1]
    for t in range(t_len - 2, -1, -1):
        pp_next = pred_cov[t + 1]
        sign, logdet = np.linalg.slogdet(pp_next)
        if sign <= 0.0:
            raise ValueError(
                f"t={t + 1} 时预测协方差 P_pred 非正定（logdet={logdet}），"
                "Q=0 且 P0 退化时 RTS 平滑无定义，检查 process_cov/initial_cov 的设定"
            )
        rhs = f_cov[t] @ Fm.T
        j_mat = np.linalg.solve(pp_next.T, rhs.T).T
        s_state[t] = f_state[t] + j_mat @ (s_state[t + 1] - pred_state[t + 1])
        sc = f_cov[t] + j_mat @ (s_cov[t + 1] - pp_next) @ j_mat.T
        s_cov[t] = 0.5 * (sc + sc.T)
        gain[t] = j_mat
    return {
        "filtered_states": f_state,
        "filtered_covs": f_cov,
        "smoothed_states": s_state,
        "smoothed_covs": s_cov,
        "log_likelihood": float(filt["loglik"]),
        "predicted_states": pred_state,
        "predicted_covs": pred_cov,
        "innovations": innov,
        "smoother_gain": gain,
    }


# --------------------------------------------------------------------------
# Ljung-Box 白噪声检验
# --------------------------------------------------------------------------

def ljung_box(residual: ArrayLike, lags: int = 10) -> Dict[str, object]:
    """Ljung-Box 残差白噪声检验（自相关联合为 0 的卡方检验）。

    参数:
        residual: 一维残差序列（例如 ``arima_fit(...)["residual"]``；注意其前 t0 位是占位的 0）。
        lags: 检验使用的最大滞后阶数 L（>= 1，且必须小于序列长度）。

    返回:
        dict，键为：
        ``stat``    统计量 ``Q = n(n+2) sum_{k=1..L} rho_k^2 / (n-k)``；
        ``df``      自由度（本实现取 ``L``，**未扣除已估参数个数**）；
        ``p_value`` 卡方生存函数 ``P(chi2(L) > Q)``；
        ``acf``     长度 L 的自相关序列 ``rho_1..rho_L``（1/n 归一化）。

    算法:
        1. ``x = residual - mean(residual)``；
        2. ``rho_k = (1/n) sum_{t=k+1..n} x_t x_{t-k} / rho_0``（统一用 1/n，等价于带均值修正）；
        3. 统计量按上式加权 ``1/(n-k)``（Box-Pierce 的有限样本修正）；
        4. p 值 = ``Q(L/2, Q/2)``，由本模块自实现的上不完全 gamma 函数给出。

    复杂度:
        时间 O(L n) / 空间 O(n)。

    陷阱:
        - **自由度口径**：严谨做法是 ``df = L - (已估参数个数)``；本实现取 ``df = L``，
          因此对拟合后的残差检验偏**保守**（p 值偏大、不容易拒绝白噪声）。做严格结论时
          请自行按 ``forecasting.ar_model`` 的阶数等扣减自由度。
        - 输入若带占位的 0（ARIMA 条件残差的前 t0 位），会人为压低自相关，使检验偏向
          "不拒绝"。**正确做法**是先按 t0 截掉占位项，本函数不做这个截断（不猜测调用方意图）。
        - 序列长度为 0 方差（常数残差）时无定义，抛 ``ValueError``。
        - Ljung-Box 对**高阶滞后**或长记忆过程功效有限；``lags`` 的经验取法是
          ``min(10, n/5)`` 或 ``2 * 周期长度``，取太大会严重损失功效。

    参考:
        Ljung & Box (1978), Biometrika 65(2):297-303；Box & Pierce (1970)。
    """
    rv = as_vector(residual, "residual")
    ell = _as_int(lags, "lags", minimum=1)
    n = rv.size
    if ell >= n:
        raise ValueError(f"lags={ell} 必须小于残差长度 {n}")
    x = rv - float(rv.mean())
    r0 = float(np.dot(x, x) / n)
    if r0 <= 0.0:
        raise ValueError("残差方差为 0（常数残差），自相关无定义")
    acf = np.empty(ell, dtype=float)
    for k in range(1, ell + 1):
        acf[k - 1] = float(np.dot(x[k:], x[:n - k]) / n) / r0
    ks = np.arange(1, ell + 1, dtype=float)
    stat = float(n * (n + 2) * np.sum(acf * acf / (n - ks)))
    p_value = 1.0 if stat <= 0.0 else _chi2_sf(stat, ell)
    return {
        "stat": stat,
        "df": int(ell),
        "p_value": float(p_value),
        "acf": acf,
    }


def _expect(cond: bool, msg: str) -> None:
    """自测用的断言：条件不成立时抛 ``AssertionError``（不使用 ``assert`` 语句）。

    参数:
        cond: 期望成立的条件。
        msg: 失败时附带的中文说明。

    返回:
        None（失败则抛 ``AssertionError``）。

    算法:
        显式的 ``if not cond: raise``，因此 **加 -O 也不会被优化掉**，
        这是本仓库禁用 ``assert`` 的原因之一。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        断言里不要写"两个浮点数完全相等"，浮点比较必须给容差；本模块所有近似相等
        都用显式容差判断。

    参考:
        Python 官方文档关于 ``assert`` 与 ``-O`` 的说明。
    """
    if not cond:
        raise AssertionError("timeseries._self_test 失败：" + msg)


def _self_test() -> Dict[str, object]:
    """模块自测：返回可 JSON 序列化的键值字典（键一律以 ``ts_`` 开头）。

    参数:
        无。

    返回:
        dict：键为 ``ts_<含义>``，值为 float（已 ``round``）/ int / bool / str / list / dict。
        全部为**纯 Python 类型**，可直接 ``json.dumps``。

    算法:
        分组验证，每组都包含**闭式解或独立实现**的交叉校验：
        1. GM(1,1)：构造严格满足离散白化方程 ``x_k + a0 z_k = b0`` 的几何序列，
           闭式验证最小二乘能精确还原 ``(a0, b0)``（残差恒为 0 ⇒ LS 解的独一最小值点）；
           再用纯指数序列报告"离散—连续不匹配"造成的真实预测误差（诚实报告，不假装为 0）。
        2. 后验差检验：与独立 numpy 公式逐项对比；完美拟合必须给出 ``C = 0、P = 1、等级=好``。
        3. 差分：``k^2`` 的二阶差分闭式为常数 2。
        4. ARIMA：AR(1) 的 CLS 与**独立 OLS 闭式解**一致；与
           ``forecasting.ar_model``（Yule-Walker）的差值实测并报告；白噪声情形与
           ``-0.5 n (ln 2pi + ln sigma2 + 1)`` 的 AIC/BIC 闭式一致；ARMA(1,1) 能还原
           模拟参数；``arima_forecast`` 的 se 与三种闭式解（AR(1)、白噪声、随机游走
           ``sigma sqrt(h)``）一致，随机游走的点预测与 ``x_n + h mu`` 一致。
        5. 选阶：网格结果的 ``best`` 必须等于表内按准则的最小值，AR(1) 数据上应选到 p=1、d=0。
        6. SARIMA：``P=D=Q=0`` 时与 ``arima_fit`` **逐位相同**；季节 AR(1) 能还原系数；
           ``D>0`` 时 ``arima_forecast`` 必须抛 ``ValueError``（把限制写成被测行为）。
        7. GARCH：模拟 (0.1, 0.15, 0.8) 能还原到合理精度、持续性 < 1；对数似然必须优于
           独立计算的常数方差似然；多步方差预测与均值回复闭式一致且单调靠近长期方差。
        8. 卡尔曼：预测方差收敛到 Riccati 闭式稳态 ``u = (q + sqrt(q^2+4qr))/2``；
           ``r -> 0`` 时滤波值趋于观测值；局部水平与一般线性滤波**逐位一致**；
           一般线性滤波的对数似然与**独立构造的联合高斯似然**一致（含二维常速模型）。
        9. Ljung-Box：``_chi2_sf`` 与闭式（df=2、df=4）一致；统计量与独立公式一致；
           iid 残差不拒绝白噪声、周期残差强烈拒绝。
        10. 输入校验：对非整数的阶数、越界的 ``d``/``lags``、空序列等非法入参，
            统计实际抛出的 ``ValueError`` 条数（``ts_validation_valueerror_count``），
            确认校验没有被后续改动绕过。

    复杂度:
        时间 O(数秒)（GARCH 的模式搜索占大头，T 已压到几百）/ 空间 O(T)。

    陷阱:
        - 自测里所有随机数都来自 ``_common.rng``（默认 ``DEFAULT_SEED``），**顺序固定**；
          中间插入任何一次抽样都会改变后续所有数据，因此新增检查请**追加在末尾**。
        - 容差是"够用就好"的经验值：GARCH/ARMA 的系数还原用的是 0.05~0.2 量级的松容差
          （它们本质是统计估计，不是精确计算），闭式校验才用 1e-9 级紧容差。
        - 本自测**不覆盖**性能与数值病态场景（近单位根、超高阶、极短序列）。

    参考:
        与 ``forecasting._self_test`` 同风格；跨模块交叉验证见本文件 ``arima_fit`` 的
        "与 Yule-Walker 对比"一节。
    """
    out: Dict[str, object] = {}
    gen = rng()

    # ---------------- GM(1,1) ----------------
    a0 = -0.6
    ratio = (1.0 - a0 / 2.0) / (1.0 + a0 / 2.0)
    kk = np.arange(12, dtype=float)
    x_exact = 4.0 * ratio ** kk
    x1 = np.cumsum(x_exact)
    z_bg = 0.5 * (x1[1:] + x1[:-1])
    b_series = x_exact[1:] + a0 * z_bg
    b0 = float(b_series[0])
    out["ts_gm11_whitening_b_spread"] = round(float(np.max(b_series) - np.min(b_series)), 12)
    _expect(
        float(np.max(np.abs(b_series - b0))) < 1e-12,
        "构造数据应严格满足离散白化方程 x_k + a0 z_k = b0（残差恒为 0）",
    )
    gm = gm11(x_exact, n_forecast=3)
    out["ts_gm11_exact_a_err"] = round(abs(float(gm["a"]) - a0), 12)
    out["ts_gm11_exact_b_err"] = round(abs(float(gm["b"]) - b0), 10)
    out["ts_gm11_exact_forecast_err"] = round(
        float(np.max(np.abs(np.asarray(gm["forecast"]) - 4.0 * ratio ** np.arange(12, 15)))), 8
    )
    out["ts_gm11_exact_forecast_rel_err"] = round(
        float(np.max(
            np.abs(np.asarray(gm["forecast"]) - 4.0 * ratio ** np.arange(12, 15))
            / (4.0 * ratio ** np.arange(12, 15))
        )), 6
    )
    # 独立验证 (a,b) 确实是白化方程 LS 目标的最小值点：任何 ±1e-5 扰动都不能更优。
    fit_chk = np.asarray(gm["fitted"])
    z_chk = 0.5 * (x1[1:] + x1[:-1])
    tgt = x_exact[1:]

    def _obj(aa: float, bb: float) -> float:
        r = tgt + aa * z_chk - bb
        return float(np.dot(r, r))

    obj_best = _obj(float(gm["a"]), float(gm["b"]))
    worse = 0
    for da in (-1e-5, 1e-5):
        for db in (-1e-5, 1e-5):
            if _obj(float(gm["a"]) + da, float(gm["b"]) + db) >= obj_best:
                worse += 1
    out["ts_gm11_ls_resid_ss"] = round(obj_best, 12)
    out["ts_gm11_ls_perturbations_worse"] = int(worse)
    out["ts_gm11_fit_ratio_err"] = round(
        float(np.max(np.abs(fit_chk[2:] / fit_chk[1:-1] - math.exp(-float(gm["a"]))))), 12
    )
    _expect(float(out["ts_gm11_exact_a_err"]) < 1e-9, "GM(1,1) 应精确还原 a0（残差为 0 时 LS 唯一）")
    _expect(float(out["ts_gm11_exact_b_err"]) < 1e-9, "GM(1,1) 应精确还原 b0")
    _expect(int(worse) == 4, "还原出的 (a,b) 应是 LS 目标的严格最小值点（四向扰动都更差）")
    _expect(
        float(out["ts_gm11_fit_ratio_err"]) < 1e-9,
        "拟合值应是公比为 exp(-a) 的等比序列（代码确实实现了该公式）",
    )
    _expect(
        float(out["ts_gm11_exact_forecast_rel_err"]) < 0.30,
        "即使参数精确还原，预测仍带离散化偏差（e^{-a} 与 (1-a/2)/(1+a/2) 的差），应 <30%",
    )
    # 偏差是 O(a^3) 的：|a| 变小时应迅速消失（这是"GM(1,1) 对指数序列精确"说法的真实边界）。
    a_small = -0.1
    r_small = (1.0 - a_small / 2.0) / (1.0 + a_small / 2.0)
    x_small = 4.0 * r_small ** np.arange(12.0)
    gm_small = gm11(x_small, n_forecast=3)
    truth_small = 4.0 * r_small ** np.arange(12, 15)
    out["ts_gm11_small_a_forecast_rel_err"] = round(
        float(np.max(np.abs(np.asarray(gm_small["forecast"]) - truth_small) / truth_small)), 9
    )
    out["ts_gm11_bias_ratio_small_over_large"] = round(
        float(out["ts_gm11_small_a_forecast_rel_err"]) / float(out["ts_gm11_exact_forecast_rel_err"]), 9
    )
    _expect(
        float(out["ts_gm11_small_a_forecast_rel_err"]) < 5e-3,
        "|a| 很小时离散化偏差应降到 5e-3 以下（偏差随 a^3 收缩）",
    )

    k2 = np.arange(10, dtype=float)
    x_exp = 2.0 * np.exp(0.3 * k2)
    gm_exp = gm11(x_exp, n_forecast=3)
    fit_exp = np.asarray(gm_exp["fitted"])
    ratio_fit = fit_exp[2:] / fit_exp[1:-1]
    out["ts_gm11_exp_fit_ratio_err"] = round(
        float(np.max(np.abs(ratio_fit - math.exp(-float(gm_exp["a"]))))), 12
    )
    truth_exp = 2.0 * np.exp(0.3 * np.arange(10, 13))
    rel_exp = np.abs(np.asarray(gm_exp["forecast"]) - truth_exp) / truth_exp
    out["ts_gm11_exp_a"] = round(float(gm_exp["a"]), 6)
    out["ts_gm11_exp_forecast_rel_err"] = round(float(np.max(rel_exp)), 6)
    out["ts_gm11_exp_forecast"] = [round(float(v), 6) for v in gm_exp["forecast"]]
    _expect(float(out["ts_gm11_exp_fit_ratio_err"]) < 1e-9, "GM(1,1) 的拟合值应是等比序列 exp(-a)")
    _expect(
        float(out["ts_gm11_exp_forecast_rel_err"]) < 0.05,
        "纯指数序列的 GM(1,1) 预测相对误差应在 5% 以内（离散—连续不匹配的偏差）",
    )

    # ---------------- 后验差检验 ----------------
    post = gm11_posterior_check(x_exp, fit_exp)
    resid_exp = x_exp - fit_exp
    s1_ind = float(np.std(x_exp, ddof=1))
    s2_ind = float(np.std(resid_exp, ddof=1))
    p_ind = float(np.mean(np.abs(resid_exp - resid_exp.mean()) < 0.6745 * s1_ind))
    out["ts_gm_posterior_c_ratio"] = round(float(post["c_ratio"]), 9)
    out["ts_gm_posterior_c_diff"] = round(abs(float(post["c_ratio"]) - s2_ind / s1_ind), 12)
    out["ts_gm_posterior_p"] = round(float(post["p_small_error"]), 9)
    out["ts_gm_posterior_p_diff"] = round(abs(float(post["p_small_error"]) - p_ind), 12)
    out["ts_gm_posterior_grade"] = str(post["grade"])
    _expect(float(out["ts_gm_posterior_c_diff"]) < 1e-12, "方差比 C 应与独立公式一致")
    _expect(float(out["ts_gm_posterior_p_diff"]) < 1e-12, "小误差概率 P 应与独立计数一致")
    perfect = gm11_posterior_check(x_exp, x_exp)
    out["ts_gm_posterior_perfect_c"] = round(float(perfect["c_ratio"]), 12)
    out["ts_gm_posterior_perfect_p"] = round(float(perfect["p_small_error"]), 12)
    out["ts_gm_posterior_perfect_grade"] = str(perfect["grade"])
    _expect(
        float(perfect["c_ratio"]) == 0.0
        and float(perfect["p_small_error"]) == 1.0
        and str(perfect["grade"]) == "好",
        "完美拟合时应有 C=0、P=1、等级为“好”",
    )

    # ---------------- 差分 ----------------
    kd = np.arange(9, dtype=float)
    d2 = difference_series(kd ** 2, 2)
    out["ts_diff_len"] = int(d2.size)
    out["ts_diff_second_of_square_err"] = round(float(np.max(np.abs(d2[2:] - 2.0))), 12)
    out["ts_diff_lead_nan"] = bool(np.all(np.isnan(d2[:2])) and np.all(np.isfinite(d2[2:])))
    _expect(int(out["ts_diff_len"]) == 9, "difference_series 必须保持长度不变")
    _expect(float(out["ts_diff_second_of_square_err"]) < 1e-12, "k^2 的二阶差分闭式为 2")
    _expect(bool(out["ts_diff_lead_nan"]), "前 d 位应为 NaN 且其余为有限值")
    d0 = difference_series(kd, 0)
    out["ts_diff_order0_same"] = bool(np.allclose(d0, kd))

    # ---------------- ARIMA：估计 ----------------
    n_ar = 4000
    eps = gen.standard_normal(n_ar)
    y_ar = np.empty(n_ar, dtype=float)
    y_ar[0] = eps[0]
    for t in range(1, n_ar):
        y_ar[t] = 0.7 * y_ar[t - 1] + eps[t]
    m_ar = arima_fit(y_ar, 1, 0, 0)
    phi_hat = float(np.asarray(m_ar["phi"])[0])
    yc = y_ar - y_ar.mean()
    phi_ols = float(np.dot(yc[1:], yc[:-1]) / np.dot(yc[:-1], yc[:-1]))
    from .forecasting import ar_model as _ar_model

    phi_yw = float(np.asarray(_ar_model(y_ar, 1)["coef"])[0])
    out["ts_arima_ar1_phi"] = round(phi_hat, 9)
    out["ts_arima_ar1_phi_ols_closed_err"] = round(abs(phi_hat - phi_ols), 12)
    out["ts_arima_ar1_phi_yule_walker"] = round(phi_yw, 9)
    out["ts_arima_ar1_phi_vs_yw_diff"] = round(abs(phi_hat - phi_yw), 9)
    out["ts_arima_ar1_n_iter"] = int(m_ar["n_iter"])
    _expect(
        float(out["ts_arima_ar1_phi_ols_closed_err"]) < 1e-12,
        "AR(1) 的 CLS 必须等于去掉首点的 OLS 闭式解",
    )
    _expect(
        int(m_ar["n_iter"]) <= 2,
        "纯 AR（q=0）的设计矩阵不依赖残差，CLS 应一次最小二乘即收敛（最多再确认一轮）",
    )
    _expect(
        float(out["ts_arima_ar1_phi_vs_yw_diff"]) < 5e-3,
        "CLS 与 Yule-Walker 的差应远小于抽样误差（同阶一致，但非逐位相等）",
    )

    m_wn = arima_fit(eps, 0, 0, 0)
    sigma2_wn = float(m_wn["sigma2"])
    var_wn = float(np.var(eps))
    ll_wn = -0.5 * n_ar * (math.log(2.0 * math.pi) + math.log(var_wn) + 1.0)
    out["ts_arima_wn_sigma2_err"] = round(abs(sigma2_wn - var_wn), 12)
    out["ts_arima_wn_aic_err"] = round(abs(float(m_wn["aic"]) - (-2.0 * ll_wn + 2.0)), 10)
    out["ts_arima_wn_bic_err"] = round(
        abs(float(m_wn["bic"]) - (-2.0 * ll_wn + math.log(n_ar))), 10
    )
    out["ts_arima_wn_p"] = int(np.asarray(m_wn["phi"]).size)
    _expect(float(out["ts_arima_wn_sigma2_err"]) < 1e-12, "白噪声的 sigma2 应等于样本方差（ddof=0）")
    _expect(float(out["ts_arima_wn_aic_err"]) < 1e-9, "白噪声 AIC 应与闭式一致")
    _expect(float(out["ts_arima_wn_bic_err"]) < 1e-9, "白噪声 BIC 应与闭式一致")

    n_11 = 4000
    e11 = gen.standard_normal(n_11)
    y11 = np.empty(n_11, dtype=float)
    y11[0] = e11[0]
    for t in range(1, n_11):
        y11[t] = 0.6 * y11[t - 1] + e11[t] + 0.5 * e11[t - 1]
    m11 = arima_fit(y11, 1, 0, 1)
    phi11 = float(np.asarray(m11["phi"])[0])
    th11 = float(np.asarray(m11["theta"])[0])
    out["ts_arima_arma11_phi"] = round(phi11, 6)
    out["ts_arima_arma11_theta"] = round(th11, 6)
    out["ts_arima_arma11_n_iter"] = int(m11["n_iter"])
    out["ts_arima_arma11_sigma2"] = round(float(m11["sigma2"]), 6)
    _expect(abs(phi11 - 0.6) < 0.05, "ARMA(1,1) 的 phi 应能还原到 0.6 附近")
    _expect(abs(th11 - 0.5) < 0.06, "ARMA(1,1) 的 theta 应能还原到 0.5 附近")
    _expect(int(m11["n_iter"]) > 1, "ARMA(1,1) 的 CLS 需要迭代（MA 项依赖残差）")

    # ---------------- ARIMA：预测 ----------------
    h_fc = 5
    fc_ar = arima_forecast(m_ar, h_fc)
    psi_ar = np.array([phi_hat ** i for i in range(h_fc)])
    se_ar_closed = np.sqrt(float(m_ar["sigma2"]) * np.cumsum(psi_ar ** 2))
    out["ts_arima_fc_ar1_se_closed_err"] = round(
        float(np.max(np.abs(np.asarray(fc_ar["se"]) - se_ar_closed))), 12
    )
    out["ts_arima_fc_ar1_se"] = [round(float(v), 6) for v in fc_ar["se"]]
    _expect(
        float(out["ts_arima_fc_ar1_se_closed_err"]) < 1e-10,
        "AR(1) 的预测标准差应等于 sigma*sqrt(sum_l psi_l^2) 的闭式",
    )

    fc_wn = arima_forecast(m_wn, 4)
    # 原断言口径错误：它把 ARMA(0,0) 的白噪声当成了随机游走。白噪声的 psi 权重是
    # psi_0=1、psi_{1..h-1}=0（实测 _psi_weights 返回 [1,0,0,0]），于是
    # Var = sigma2 * Σ_l psi_l^2 = sigma2 * 1，se 恒为 sigma，与步长 h 无关；
    # sigma*sqrt(h) 是 d=1 随机游走（psi_l 恒为 1，Σ psi^2 = h）的闭式，见下面 rw 一节。
    se_wn_closed = math.sqrt(sigma2_wn) * np.ones(4, dtype=float)
    out["ts_arima_fc_wn_se_closed_err"] = round(
        float(np.max(np.abs(np.asarray(fc_wn["se"]) - se_wn_closed))), 12
    )
    _expect(
        float(out["ts_arima_fc_wn_se_closed_err"]) < 1e-10,
        "白噪声（p=q=0）的预测标准差应恒为 sigma（psi_0=1 而其余为 0，与 h 无关）",
    )

    fc_11 = arima_forecast(m11, 3)
    psi11 = np.array([1.0, phi11 + th11, (phi11 + th11) * phi11])
    se_11_closed = np.sqrt(float(m11["sigma2"]) * np.cumsum(psi11 ** 2))
    out["ts_arima_fc_arma11_se_closed_err"] = round(
        float(np.max(np.abs(np.asarray(fc_11["se"]) - se_11_closed))), 10
    )
    _expect(
        float(out["ts_arima_fc_arma11_se_closed_err"]) < 1e-9,
        "ARMA(1,1) 的 psi 权重闭式为 (phi+theta)phi^(k-1)",
    )

    rw = np.cumsum(gen.standard_normal(600)) + 5.0
    m_rw = arima_fit(rw, 0, 1, 0)
    mu_rw = float(m_rw["mean"])
    fc_rw = arima_forecast(m_rw, 4)
    fc_rw_closed = float(rw[-1]) + np.arange(1.0, 5.0) * mu_rw
    se_rw_closed = math.sqrt(float(m_rw["sigma2"])) * np.sqrt(np.arange(1.0, 5.0))
    out["ts_arima_fc_rw_point_err"] = round(
        float(np.max(np.abs(np.asarray(fc_rw["forecast"]) - fc_rw_closed))), 12
    )
    out["ts_arima_fc_rw_se_closed_err"] = round(
        float(np.max(np.abs(np.asarray(fc_rw["se"]) - se_rw_closed))), 12
    )
    out["ts_arima_fc_rw_forecast"] = [round(float(v), 6) for v in fc_rw["forecast"]]
    _expect(
        float(out["ts_arima_fc_rw_point_err"]) < 1e-10,
        "带漂移随机游走的点预测闭式为 x_n + h*mu",
    )
    _expect(
        float(out["ts_arima_fc_rw_se_closed_err"]) < 1e-10,
        "随机游走（d=1）的 h 步预测标准差应等于 sigma*sqrt(h)",
    )

    # ---------------- 选阶 ----------------
    n_sel = 600
    e_sel = gen.standard_normal(n_sel + 100)
    y_sel = np.empty(n_sel + 100, dtype=float)
    y_sel[0] = e_sel[0]
    for t in range(1, n_sel + 100):
        y_sel[t] = 0.8 * y_sel[t - 1] + e_sel[t]
    y_sel = y_sel[100:]
    sel = arima_order_select(y_sel, p_max=3, d_max=1, q_max=2, criterion="aic")
    best = sel["best"]
    table = sel["table"]
    best_min = min(table, key=lambda r: (float(r["aic"]), int(r["d"]), int(r["p"]), int(r["q"])))
    out["ts_arima_sel_n_entries"] = int(len(table))
    out["ts_arima_sel_best"] = [int(best["p"]), int(best["d"]), int(best["q"])]
    out["ts_arima_sel_best_aic"] = round(float(best["aic"]), 6)
    out["ts_arima_sel_matches_min"] = bool(
        int(best["p"]) == int(best_min["p"])
        and int(best["d"]) == int(best_min["d"])
        and int(best["q"]) == int(best_min["q"])
    )
    _expect(bool(out["ts_arima_sel_matches_min"]), "best 必须是表内按 AIC 的最小值")
    _expect(
        int(best["p"]) == 1 and int(best["d"]) == 0,
        "对 AR(1)（phi=0.8）数据应选出 p=1、d=0",
    )
    sel_bic = arima_order_select(y_sel, p_max=2, d_max=0, q_max=2, criterion="BIC")
    out["ts_arima_sel_bic_best"] = [int(sel_bic["best"]["p"]), int(sel_bic["best"]["q"])]
    _expect(
        int(sel_bic["best"]["p"]) == 1,
        "BIC 在 AR(1) 数据上也应选出 p=1",
    )

    # ---------------- SARIMA ----------------
    s_per = 12
    n_s = 500
    e_sea = gen.standard_normal(n_s + s_per)
    y_sea = np.zeros(n_s + s_per, dtype=float)
    for t in range(s_per, n_s + s_per):
        y_sea[t] = 0.7 * y_sea[t - s_per] + e_sea[t]
    y_sea = y_sea[s_per:]
    m_sea = sarima_fit(y_sea, s_per, 0, 0, 0, 1, 0, 0)
    sph = float(np.asarray(m_sea["seasonal_phi"])[0])
    out["ts_sarima_seasonal_phi"] = round(sph, 6)
    out["ts_sarima_seasonal_phi_err"] = round(abs(sph - 0.7), 6)
    out["ts_sarima_period"] = int(m_sea["period"])
    out["ts_sarima_P_D_Q"] = [int(m_sea["P"]), int(m_sea["D"]), int(m_sea["Q"])]
    _expect(abs(sph - 0.7) < 0.05, "季节 AR(1)（周期 12）的系数应能还原到 0.7 附近")

    m_a = arima_fit(y_sel, 1, 1, 1)
    m_s = sarima_fit(y_sel, s_per, 1, 1, 1, 0, 0, 0)
    same = max(
        float(np.max(np.abs(np.asarray(m_s["phi"]) - np.asarray(m_a["phi"])))),
        float(np.max(np.abs(np.asarray(m_s["theta"]) - np.asarray(m_a["theta"])))),
        abs(float(m_s["sigma2"]) - float(m_a["sigma2"])),
    )
    out["ts_sarima_no_season_matches_arima"] = round(float(same), 15)
    _expect(
        float(out["ts_sarima_no_season_matches_arima"]) == 0.0,
        "P=D=Q=0 时 SARIMA 与 ARIMA 应走同一条路径、结果逐位相同",
    )

    m_seaD = sarima_fit(y_sea, s_per, 0, 0, 0, 1, 1, 0)
    out["ts_sarima_D1_n_eff"] = int(m_seaD["n_eff"])
    rejected = False
    try:
        arima_forecast(m_seaD, 2)
    except ValueError:
        rejected = True
    out["ts_sarima_D1_forecast_rejected"] = bool(rejected)
    _expect(bool(rejected), "D>0 的模型调用 arima_forecast 必须抛 ValueError（未实现反差分）")

    # ---------------- GARCH(1,1) ----------------
    n_g = 600
    burn = 200
    w_g = gen.standard_normal(n_g + burn)
    r_g = np.empty(n_g + burn, dtype=float)
    s2_g = 0.1 / (1.0 - 0.95)
    for t in range(n_g + burn):
        s2_g = 0.1 + 0.15 * (r_g[t - 1] ** 2 if t > 0 else 0.0) + 0.8 * s2_g
        r_g[t] = math.sqrt(s2_g) * w_g[t]
    r_g = r_g[burn:]
    mg = garch11_fit(r_g, max_iter=400)
    om = float(mg["omega"])
    al = float(mg["alpha"])
    be = float(mg["beta"])
    xc_g = r_g - r_g.mean()
    var_g = float(np.var(xc_g, ddof=1))
    ll_const = -0.5 * n_g * (math.log(2.0 * math.pi) + math.log(var_g) + 1.0)
    out["ts_garch_omega"] = round(om, 6)
    out["ts_garch_alpha"] = round(al, 6)
    out["ts_garch_beta"] = round(be, 6)
    out["ts_garch_persistence"] = round(al + be, 6)
    out["ts_garch_persistence_err"] = round(abs(al + be - 0.95), 6)
    out["ts_garch_loglik"] = round(float(mg["loglik"]), 6)
    out["ts_garch_loglik_const"] = round(float(ll_const), 6)
    out["ts_garch_n_iter"] = int(mg["n_iter"])
    out["ts_garch_sigma2_len"] = int(np.asarray(mg["sigma2"]).size)
    _expect(al > 0.0 and be > 0.0 and om > 0.0, "GARCH 参数应为正")
    _expect(al + be < 1.0, "持续性必须在平稳域内（< 1）")
    _expect(abs(al + be - 0.95) < 0.15, "持续性应能还原到 0.95 附近（±0.15）")
    _expect(abs(al - 0.15) < 0.10, "alpha 应能还原到 0.15 附近（±0.10）")
    _expect(
        float(mg["loglik"]) > ll_const,
        "GARCH 的对数似然必须优于常数方差基准（否则模型没有意义）",
    )
    _expect(int(out["ts_garch_sigma2_len"]) == n_g, "条件方差序列长度应等于样本量")

    gfc = garch11_forecast(mg, 20)
    lr = om / (1.0 - al - be)
    out["ts_garch_fc_long_run"] = round(float(gfc["long_run_variance"]), 6)
    out["ts_garch_fc_long_run_err"] = round(abs(float(gfc["long_run_variance"]) - lr), 12)
    out["ts_garch_fc_first_err"] = round(abs(float(gfc["variance"][0]) - float(mg["sigma2_next"])), 12)
    out["ts_garch_fc_last"] = round(float(gfc["variance"][-1]), 6)
    out["ts_garch_fc_volatility_head"] = [round(float(v), 6) for v in gfc["volatility"][:3]]
    _expect(float(out["ts_garch_fc_long_run_err"]) < 1e-12, "长期方差闭式为 omega/(1-alpha-beta)")
    _expect(
        float(out["ts_garch_fc_first_err"]) < 1e-12,
        "variance[0] 必须等于一步向前方差 sigma2_next（回复指数是 h-1 而不是 h）",
    )
    _expect(
        abs(float(gfc["variance"][-1]) - lr) < abs(float(gfc["variance"][0]) - lr),
        "多步方差预测应单调靠近长期方差",
    )

    # ---------------- 卡尔曼滤波 ----------------
    q_k, r_k = 1.0, 1.0
    y_k = np.cumsum(gen.standard_normal(200)) + gen.standard_normal(200)
    kl = kalman_filter_local_level(y_k, q=q_k, r=r_k)
    u_ss = 0.5 * (q_k + math.sqrt(q_k * q_k + 4.0 * q_k * r_k))
    out["ts_kf_local_steady_var"] = round(float(np.asarray(kl["predicted_variance"])[-1]), 9)
    out["ts_kf_local_riccati_var"] = round(u_ss, 9)
    out["ts_kf_local_steady_err"] = round(
        abs(float(np.asarray(kl["predicted_variance"])[-1]) - u_ss), 9
    )
    out["ts_kf_local_steady_gain_err"] = round(abs(float(np.asarray(kl["gain"])[-1]) - u_ss / (u_ss + r_k)), 12)
    out["ts_kf_local_loglik"] = round(float(kl["loglik"]), 6)
    _expect(float(out["ts_kf_local_steady_err"]) < 1e-8, "预测方差应收敛到 Riccati 闭式稳态解")
    _expect(float(out["ts_kf_local_steady_gain_err"]) < 1e-9, "稳态增益应为 u/(u+r)")

    kl_tiny = kalman_filter_local_level(y_k, q=1.0, r=1e-9)
    out["ts_kf_local_r0_max_diff"] = round(
        float(np.max(np.abs(np.asarray(kl_tiny["filtered"]) - y_k))), 9
    )
    _expect(
        float(out["ts_kf_local_r0_max_diff"]) < 1e-6,
        "r -> 0 时滤波值应趋于观测值",
    )

    kl_lin = kalman_filter_linear(
        y_k, [[1.0]], [[1.0]], [[q_k]], [[r_k]], [float(y_k[0])], [[0.0]]
    )
    out["ts_kf_linear_matches_local_state"] = round(
        float(np.max(np.abs(np.asarray(kl_lin["state"])[:, 0] - np.asarray(kl["filtered"])))), 12
    )
    out["ts_kf_linear_matches_local_loglik"] = round(abs(float(kl_lin["loglik"]) - float(kl["loglik"])), 9)
    _expect(
        float(out["ts_kf_linear_matches_local_state"]) < 1e-12
        and float(out["ts_kf_linear_matches_local_loglik"]) < 1e-8,
        "一般线性滤波在 F=H=1、P0=0 时必须与局部水平滤波一致",
    )

    t_j = 12
    y_j = y_k[:t_j]
    p0_j = 2.0
    kl_j = kalman_filter_linear(
        y_j, [[1.0]], [[1.0]], [[q_k]], [[r_k]], [0.0], [[p0_j]]
    )
    sig_j = np.empty((t_j, t_j), dtype=float)
    for i in range(t_j):
        for j in range(t_j):
            sig_j[i, j] = p0_j + q_k * (min(i, j) + 1)
    sig_j += r_k * np.eye(t_j)
    sign_j, ld_j = np.linalg.slogdet(sig_j)
    ll_direct = -0.5 * (
        t_j * math.log(2.0 * math.pi) + ld_j + float(y_j @ np.linalg.solve(sig_j, y_j))
    )
    out["ts_kf_linear_loglik"] = round(float(kl_j["loglik"]), 9)
    out["ts_kf_linear_loglik_direct"] = round(float(ll_direct), 9)
    out["ts_kf_linear_loglik_direct_err"] = round(abs(float(kl_j["loglik"]) - ll_direct), 9)
    _expect(
        float(out["ts_kf_linear_loglik_direct_err"]) < 1e-8,
        "卡尔曼对数似然必须等于独立构造的联合高斯似然（局部水平）",
    )

    F2 = np.array([[1.0, 1.0], [0.0, 1.0]])
    H2 = np.array([[1.0, 0.0]])
    Q2 = np.array([[0.5, 0.0], [0.0, 0.2]])
    R2 = np.array([[0.7]])
    P02 = np.array([[1.0, 0.0], [0.0, 0.5]])
    y2 = y_k[:t_j]
    kl2 = kalman_filter_linear(y2, F2, H2, Q2, R2, np.zeros(2), P02)
    sig2 = np.empty((t_j, t_j), dtype=float)
    for i in range(t_j):
        for j in range(t_j):
            cov2 = (np.linalg.matrix_power(F2, i + 1) @ P02
                    @ np.linalg.matrix_power(F2, j + 1).T)
            for s_ in range(min(i, j) + 1):
                # 原对照把 Q 的传播写成 F^s Q (F^s)'，漏了 F 的幂次差：状态
                # alpha_i = F^{i+1} alpha_{-1} + Σ_s F^{i-s} eta_s，第 k 个共同噪声
                # 在 alpha_i / alpha_j 里的系数是 F^{i-k} 与 F^{j-k}，故协方差贡献为
                # F^{i-k} Q (F^{j-k})'（只在 F=I 时写法与原对照相同）。
                Fi = np.linalg.matrix_power(F2, i - s_)
                Fj = np.linalg.matrix_power(F2, j - s_)
                cov2 = cov2 + Fi @ Q2 @ Fj.T
            # 注意：H2 @ cov2 @ H2.T 是 (1, 1) 的二维数组，numpy >= 2.5 起
            # float() 只接受 0 维数组，直接 float(...) 会 TypeError，故取值后再转。
            sig2[i, j] = float((H2 @ cov2 @ H2.T)[0, 0])
    sig2 += float(R2[0, 0]) * np.eye(t_j)
    sign_2, ld_2 = np.linalg.slogdet(sig2)
    ll_direct2 = -0.5 * (
        t_j * math.log(2.0 * math.pi) + ld_2 + float(y2 @ np.linalg.solve(sig2, y2))
    )
    out["ts_kf_2state_loglik"] = round(float(kl2["loglik"]), 9)
    out["ts_kf_2state_loglik_direct"] = round(float(ll_direct2), 9)
    out["ts_kf_2state_loglik_err"] = round(abs(float(kl2["loglik"]) - ll_direct2), 9)
    out["ts_kf_2state_shape"] = [int(v) for v in np.asarray(kl2["state"]).shape]
    _expect(
        float(out["ts_kf_2state_loglik_err"]) < 1e-8,
        "二维常速模型的对数似然必须等于独立构造的联合高斯似然",
    )
    _expect(
        list(np.asarray(kl2["state"]).shape) == [t_j, 2]
        and list(np.asarray(kl2["cov"]).shape) == [t_j, 2, 2],
        "二维状态滤波的返回形状应为 (T, k) 与 (T, k, k)",
    )

    # ---------------- Ljung-Box ----------------
    out["ts_lb_chi2_df2"] = round(_chi2_sf(3.0, 2), 9)
    out["ts_lb_chi2_df2_closed"] = round(math.exp(-1.5), 9)
    out["ts_lb_chi2_df2_err"] = round(abs(_chi2_sf(3.0, 2) - math.exp(-1.5)), 12)
    out["ts_lb_chi2_df4_err"] = round(
        abs(_chi2_sf(3.0, 4) - math.exp(-1.5) * 2.5), 12
    )
    out["ts_lb_chi2_df10"] = round(_chi2_sf(18.307, 10), 6)
    _expect(float(out["ts_lb_chi2_df2_err"]) < 1e-10, "df=2 的生存函数闭式为 exp(-x/2)")
    _expect(float(out["ts_lb_chi2_df4_err"]) < 1e-10, "df=4 的生存函数闭式为 exp(-x/2)(1+x/2)")
    _expect(abs(float(out["ts_lb_chi2_df10"]) - 0.05) < 1e-3, "df=10 的 5% 临界值 18.307 应给出 0.05")

    ell = 10
    lb = ljung_box(eps, ell)
    x_lb = eps - eps.mean()
    r0_lb = float(np.dot(x_lb, x_lb) / n_ar)
    rho_ind = np.array([
        float(np.dot(x_lb[k:], x_lb[:n_ar - k]) / n_ar) / r0_lb for k in range(1, ell + 1)
    ])
    stat_ind = float(n_ar * (n_ar + 2.0) * np.sum(rho_ind ** 2 / (n_ar - np.arange(1.0, ell + 1.0))))
    out["ts_lb_stat"] = round(float(lb["stat"]), 6)
    out["ts_lb_stat_formula_err"] = round(abs(float(lb["stat"]) - stat_ind), 12)
    out["ts_lb_df"] = int(lb["df"])
    out["ts_lb_iid_p"] = round(float(lb["p_value"]), 6)
    out["ts_lb_acf1"] = round(float(np.asarray(lb["acf"])[0]), 6)
    _expect(float(out["ts_lb_stat_formula_err"]) < 1e-10, "Q 统计量必须与独立公式一致")
    _expect(float(lb["p_value"]) > 0.05, "iid 噪声残差不应该被拒绝为白噪声")
    per_res = np.sin(2.0 * math.pi * np.arange(200) / 12.0)
    lb_per = ljung_box(per_res, 12)
    out["ts_lb_periodic_p"] = round(float(lb_per["p_value"]), 12)
    _expect(float(lb_per["p_value"]) < 1e-6, "强周期残差必须被拒绝为白噪声")

    # ---------------- ARIMA：d >= 1 反差分的独立校验 ----------------
    # 这些检查放在所有既有 gen 抽样之后：新增抽样不会移动上面任何既有键的随机数。
    # 反差分恒等式：d=1 直接拟合与"先手动差分、再对差分序列按 d=0 拟合"必须给出同一
    # 系数，且点预测满足逐阶累加的定义式 y_hat_h = y_T + Σ_{j<=h} diff_hat_j。
    m_rw1 = arima_fit(rw, 1, 1, 0)
    m_diff1 = arima_fit(np.diff(rw), 1, 0, 0)
    fc_rw1 = arima_forecast(m_rw1, h_fc)
    fc_diff1 = arima_forecast(m_diff1, h_fc)
    back_integrated = float(rw[-1]) + np.cumsum(np.asarray(fc_diff1["forecast"]))
    out["ts_arima_fc_d1_backdiff_err"] = round(
        float(np.max(np.abs(np.asarray(fc_rw1["forecast"]) - back_integrated))), 12
    )
    out["ts_arima_fc_d1_phi_diff_err"] = round(
        abs(float(np.asarray(m_rw1["phi"])[0]) - float(np.asarray(m_diff1["phi"])[0])), 12
    )
    out["ts_arima_fc_d1_mean_diff_err"] = round(
        abs(float(m_rw1["mean"]) - float(m_diff1["mean"])), 12
    )
    out["ts_arima_fc_d1_forecast"] = [round(float(v), 6) for v in fc_rw1["forecast"]]
    _expect(
        float(out["ts_arima_fc_d1_backdiff_err"]) < 1e-10
        and float(out["ts_arima_fc_d1_phi_diff_err"]) < 1e-12
        and float(out["ts_arima_fc_d1_mean_diff_err"]) < 1e-12,
        "d=1 的反差分恒等式 y_hat_h = y_T + Σ_{j<=h} diff_hat_j 必须逐位成立",
    )

    # d 用错会系统性滞后：线性趋势 + 噪声上，留出 RMSE 必须 d=1 << d=0。
    n_tr, h_tr = 300, 40
    tr_series = 10.0 + 0.5 * np.arange(n_tr) + 0.3 * gen.standard_normal(n_tr)
    tr_fit_s = tr_series[: n_tr - h_tr]
    tr_true = tr_series[n_tr - h_tr :]
    f_d0 = np.asarray(arima_forecast(arima_fit(tr_fit_s, 1, 0, 0), h_tr)["forecast"])
    f_d1 = np.asarray(arima_forecast(arima_fit(tr_fit_s, 1, 1, 0), h_tr)["forecast"])
    rmse_d0 = float(np.sqrt(np.mean((f_d0 - tr_true) ** 2)))
    rmse_d1 = float(np.sqrt(np.mean((f_d1 - tr_true) ** 2)))
    out["ts_arima_fc_trend_rmse_d0"] = round(rmse_d0, 6)
    out["ts_arima_fc_trend_rmse_d1"] = round(rmse_d1, 6)
    out["ts_arima_fc_trend_rmse_ratio"] = round(rmse_d0 / rmse_d1, 3)
    _expect(
        rmse_d1 < 0.2 * rmse_d0,
        "线性趋势上 d=1 的留出 RMSE 必须远小于 d=0（否则反差分路径有误）",
    )

    # d=2、p=q=0 的 se 闭式：h 步误差系数为 h, h-1, ..., 1（两次累加各贡献一层前向求和），
    # 故 se_h = sigma * sqrt(h(h+1)(2h+1)/6)。
    m_d2 = arima_fit(tr_series, 0, 2, 0)
    fc_d2 = arima_forecast(m_d2, 5)
    hh_d2 = np.arange(1.0, 6.0)
    se_d2_closed = math.sqrt(float(m_d2["sigma2"])) * np.sqrt(
        hh_d2 * (hh_d2 + 1.0) * (2.0 * hh_d2 + 1.0) / 6.0
    )
    out["ts_arima_fc_d2_se_closed_err"] = round(
        float(np.max(np.abs(np.asarray(fc_d2["se"]) - se_d2_closed))), 12
    )
    out["ts_arima_fc_d2_se"] = [round(float(v), 6) for v in fc_d2["se"]]
    _expect(
        float(out["ts_arima_fc_d2_se_closed_err"]) < 1e-10,
        "d=2 的预测标准差应为 sigma*sqrt(h(h+1)(2h+1)/6)",
    )

    # ---------------- 卡尔曼 RTS 平滑 ----------------
    # (d) 手写展开的 T=3、一维 RTS 递推（不调用 kalman_smoother_linear）
    Fs = np.array([[1.0]])
    Hs = np.array([[1.0]])
    Qs = np.array([[0.5]])
    Rs = np.array([[0.7]])
    x0s = np.array([0.0])
    P0s = np.array([[1.0]])
    ys3 = np.array([1.0, 2.0, 3.0])
    a_h: List[np.ndarray] = []
    p_pred_h: List[np.ndarray] = []
    xf_h: List[np.ndarray] = []
    pf_h: List[np.ndarray] = []
    ll_h = 0.0
    a_c, p_c = x0s.copy(), P0s.copy()
    for t_h in range(3):
        a_p = Fs @ a_c
        p_p = Fs @ p_c @ Fs.T + Qs
        v_h = ys3[t_h : t_h + 1] - Hs @ a_p
        s_h = Hs @ p_p @ Hs.T + Rs
        k_h = p_p @ Hs.T @ np.linalg.inv(s_h)
        a_c = a_p + k_h @ v_h
        p_c = p_p - k_h @ Hs @ p_p
        a_h.append(a_p)
        p_pred_h.append(p_p)
        xf_h.append(a_c)
        pf_h.append(p_c)
        _sgn, _ld = np.linalg.slogdet(s_h)
        ll_h += -0.5 * (
            math.log(2.0 * math.pi) + _ld + float(v_h @ np.linalg.solve(s_h, v_h))
        )
    xs_h = [np.zeros(1), np.zeros(1), xf_h[2].copy()]
    ps_h = [np.zeros((1, 1)), np.zeros((1, 1)), pf_h[2].copy()]
    for t_h in (1, 0):
        j_h = pf_h[t_h] @ Fs.T @ np.linalg.inv(p_pred_h[t_h + 1])
        xs_h[t_h] = xf_h[t_h] + j_h @ (xs_h[t_h + 1] - a_h[t_h + 1])
        ps_h[t_h] = pf_h[t_h] + j_h @ (ps_h[t_h + 1] - p_pred_h[t_h + 1]) @ j_h.T
    rts = kalman_smoother_linear(ys3, Fs, Hs, Qs, Rs, x0s, P0s)
    out["ts_kfs_hand_state_err"] = round(
        float(np.max(np.abs(np.asarray(rts["smoothed_states"]).ravel() - np.array(xs_h).ravel()))), 12
    )
    out["ts_kfs_hand_cov_err"] = round(
        float(np.max(np.abs(np.asarray(rts["smoothed_covs"]).ravel() - np.array(ps_h).ravel()))), 12
    )
    out["ts_kfs_hand_filter_err"] = round(
        float(np.max(np.abs(np.asarray(rts["filtered_states"]).ravel() - np.array(xf_h).ravel()))), 12
    )
    out["ts_kfs_hand_loglik_err"] = round(abs(float(rts["log_likelihood"]) - ll_h), 12)
    out["ts_kfs_hand_smoothed"] = [
        round(float(v), 6) for v in np.asarray(rts["smoothed_states"]).ravel()
    ]
    out["ts_kfs_hand_smoothed_cov"] = [
        round(float(v), 6) for v in np.asarray(rts["smoothed_covs"]).ravel()
    ]
    out["ts_kfs_shapes"] = [list(np.asarray(rts[k]).shape) for k in (
        "filtered_states", "filtered_covs", "smoothed_states", "smoothed_covs", "smoother_gain"
    )]
    _expect(
        float(out["ts_kfs_hand_state_err"]) < 1e-12
        and float(out["ts_kfs_hand_cov_err"]) < 1e-12
        and float(out["ts_kfs_hand_filter_err"]) < 1e-12
        and float(out["ts_kfs_hand_loglik_err"]) < 1e-12,
        "RTS 平滑必须与手写展开的 T=3 一维递推逐位一致（含滤波段与似然）",
    )
    _expect(
        list(np.asarray(rts["smoothed_states"]).shape) == [3, 1]
        and list(np.asarray(rts["smoothed_covs"]).shape) == [3, 1, 1]
        and list(np.asarray(rts["smoother_gain"]).shape) == [2, 1, 1],
        "平滑返回形状应为 (T,k)、(T,k,k) 与 (T-1,k,k)",
    )

    # (c) 平滑后验方差不得大于滤波后验方差（对角元逐元素 + 协方差差半正定）。
    rts2 = kalman_smoother_linear(y2, F2, H2, Q2, R2, np.zeros(2), P02)
    fc_f = np.asarray(rts2["filtered_covs"], dtype=float)
    fc_s = np.asarray(rts2["smoothed_covs"], dtype=float)
    var_gap = np.array([np.diag(fc_f[t] - fc_s[t]) for t in range(fc_f.shape[0])])
    loewner = np.array([
        np.linalg.eigvalsh(0.5 * ((fc_f[t] - fc_s[t]) + (fc_f[t] - fc_s[t]).T))
        for t in range(fc_f.shape[0])
    ])
    out["ts_kfs_var_min_gap"] = round(float(var_gap.min()), 12)
    out["ts_kfs_cov_loewner_min_eig"] = round(float(loewner.min()), 12)
    out["ts_kfs_var_leq_filtered"] = bool(float(var_gap.min()) >= -1e-9)
    out["ts_kfs_last_endpoint_err"] = round(
        float(np.max(np.abs(np.asarray(rts2["smoothed_states"])[-1] - np.asarray(rts2["filtered_states"])[-1]))), 15
    )
    _expect(
        bool(out["ts_kfs_var_leq_filtered"]) and float(loewner.min()) >= -1e-9,
        "平滑后验方差必须逐元素 <= 滤波后验方差，且两者之差在半正定序下非负",
    )
    _expect(
        float(out["ts_kfs_last_endpoint_err"]) < 1e-14,
        "末时刻没有未来信息，平滑值必须等于滤波值",
    )

    # (a) 两个极限：观测噪声 -> 0 时平滑退化到观测值；过程噪声 -> 0（状态恒为常数）时
    # 各时刻的平滑均值被拉平到同一水平。
    rts_r0 = kalman_smoother_linear(y_k, [[1.0]], [[1.0]], [[1.0]], [[1e-12]], [0.0], [[1.0]])
    out["ts_kfs_no_obs_noise_max_diff"] = round(
        float(np.max(np.abs(np.asarray(rts_r0["smoothed_states"])[:, 0] - y_k))), 12
    )
    _expect(
        float(out["ts_kfs_no_obs_noise_max_diff"]) < 1e-8,
        "观测噪声 -> 0 时平滑状态应退化到观测值",
    )
    rts_q0 = kalman_smoother_linear(y_j, [[1.0]], [[1.0]], [[0.0]], [[1.0]], [0.0], [[1e6]])
    sm_q0 = np.asarray(rts_q0["smoothed_states"])[:, 0]
    out["ts_kfs_constant_state_spread"] = round(float(sm_q0.max() - sm_q0.min()), 12)
    out["ts_kfs_constant_state_mean_err"] = round(abs(float(sm_q0[0]) - float(y_j.mean())), 9)
    _expect(
        float(out["ts_kfs_constant_state_spread"]) < 1e-9
        and float(out["ts_kfs_constant_state_mean_err"]) < 1e-4,
        "过程噪声 -> 0 时平滑序列应被拉平且趋于全样本均值（扩散先验下即样本均值）",
    )

    # ---------------- 输入校验（把限制写成被测行为） ----------------
    bad = 0
    try:
        difference_series(np.arange(5.0), 5)
    except ValueError:
        bad += 1
    try:
        gm11_posterior_check(np.ones(5), np.zeros(4))
    except ValueError:
        bad += 1
    try:
        kalman_filter_linear(y_j, [[1.0, 1.0]], [[1.0, 0.0]], [[1.0]], [[1.0]], [0.0], [[1.0]])
    except ValueError:
        bad += 1
    out["ts_validation_valueerror_count"] = int(bad)
    _expect(bad == 3, "三类非法输入都应抛 ValueError（差分超阶、长度不匹配、维度不匹配）")

    out["ts_self_test_key_count"] = int(len(out) + 1)
    return out


