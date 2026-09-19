"""时间序列预测：移动平均、指数平滑、Holt、Holt-Winters、AR(Yule-Walker)、ACF/PACF、ADF、精度指标与回测切分。

本模块共 19 个公开函数，按用途分成五组：
- 平滑类：``moving_average`` / ``exponential_smoothing`` / ``holt_linear`` / ``holt_winters``
  / ``holt_winters_multiplicative``；
- 自回归与差分：``ar_model`` / ``difference``；
- 自相关与平稳性检验：``acf`` / ``pacf`` / ``adf_test`` / ``mackinnon_crit``；
- 精度指标：``mape`` / ``rmse`` / ``mae`` / ``theil_u``；
- 基线、切分与分解：``naive_forecast`` / ``train_test_split_ts`` / ``rolling_origin_cv``
  / ``seasonal_decompose``。

本模块的共同约定
----------------
- 输入 ``y`` 一律按**时间先后**排列（``y[0]`` 最早）。全模块**不做任何随机打乱**：
  ``train_test_split_ts`` 与 ``rolling_origin_cv`` 都严格按时间顺序切分。
- 平滑类函数统一返回字典：``fitted`` 是**样本内一步预测**（与 ``y`` 等长，只用 t 时刻
  之前的信息），``forecast`` 是样本外预测。初始化阶段没有可用历史的那些位置填 ``NaN``，
  而不是用未来数据回填（后者会造成"预测精度虚高"）。
- 随机过程一律显式接收 ``seed``，默认 ``DEFAULT_SEED``（见 ``_common.rng``），不使用全局随机状态。
- 本模块只依赖 ``numpy`` 与 Python 标准库；生产环境可以用 statsmodels 复核结果
  （见 ``references/github-resources.md``），但论文里的中间量建议用本模块显式打印出来。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_vector, check_same_length, rng

__all__ = [
    "moving_average",
    "exponential_smoothing",
    "holt_linear",
    "holt_winters",
    "ar_model",
    "difference",
    "acf",
    "pacf",
    "adf_test",
    "mackinnon_crit",
    "mape",
    "rmse",
    "mae",
    "theil_u",
    "train_test_split_ts",
    "rolling_origin_cv",
    "naive_forecast",
    "seasonal_decompose",
    "holt_winters_multiplicative",
]

# --------------------------------------------------------------------------
# MacKinnon 响应面系数（ADF 临界值）
# --------------------------------------------------------------------------
# 形式：crit(n) = b0 + b1/n + b2/n^2 + b3/n^3，n 为 ADF 回归的实际观测数，
# b0 即 n -> inf 的渐近临界值。
#
# 下列系数引自 MacKinnon, J.G. (2010) "Critical Values for Cointegration Tests",
# Queen's Economics Department Working Paper No. 1227, Table 1（N = 1 的情形，
# 即单变量的 ADF 检验；该表是 MacKinnon (1994, JBES 12(2):167-176) 响应面近似的更新版）。
# 数值与 statsmodels 0.14.4 ``statsmodels/tsa/adfvalues.py`` 中的 ``tau_c_2010`` /
# ``tau_ct_2010`` / ``tau_nc_2010`` 第一行逐项一致，并在本仓库中用
# ``statsmodels.tsa.adfvalues.mackinnoncrit`` 做过数值复核（见交付报告）。
_MACKINNON_2010_N1 = {
    "c": {  # 含常数项、无趋势
        0.01: (-3.43035, -6.5393, -16.786, -79.433),
        0.05: (-2.86154, -2.8903, -4.234, -40.040),
        0.10: (-2.56677, -1.5384, -2.809, 0.0),
    },
    "ct": {  # 含常数项与线性趋势
        0.01: (-3.95877, -9.0531, -28.428, -134.155),
        0.05: (-3.41049, -4.3904, -9.036, -45.374),
        0.10: (-3.12705, -2.5856, -3.925, -22.380),
    },
    "n": {  # 无常数项
        0.01: (-2.56574, -2.2358, -3.627, 0.0),
        0.05: (-1.94100, -0.2686, -3.365, 31.223),
        0.10: (-1.61682, 0.2656, -2.714, 25.364),
    },
}

#: ADF 检验中 ``regression`` 参数的合法取值及其含义。
_REGRESSION_ALIASES = {"c": "c", "n": "n", "ct": "ct", "tc": "ct"}

#: 用于近似 MacKinnon 临界值的样本量下限（更小的样本请直接查表或模拟）。
_MIN_NOBS_FOR_CRIT = 5


def _check_unit_interval(value: float, name: str, lo_open: bool = True) -> float:
    """校验平滑参数落在 (0,1)（或 [0,1]）内，返回 float；不合法抛 ValueError。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是数值，得到 {value!r}")
    if not np.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值，得到 {value!r}")
    if lo_open:
        if not (0.0 < v < 1.0):
            raise ValueError(f"{name} 必须落在 (0, 1) 内，得到 {v}")
    else:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} 必须落在 [0, 1] 内，得到 {v}")
    return v


def _validate_positive_int(value, name: str, minimum: int = 1) -> int:
    """校验是 >= minimum 的整数，返回 int；不合法抛 ValueError。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} 必须是整数，得到 {value!r}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}，得到 {v}")
    return v


def _autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """样本自协方差 r[0..max_lag]，统一用 1/n 归一化（有偏估计）。

    参数:
        x: 已中心化的一维序列。
        max_lag: 最大滞后阶数。

    返回:
        长度 max_lag+1 的数组，r[k] = (1/n) * sum_{t=k}^{n-1} x[t] * x[t-k]。

    算法:
        直接按定义求和的向量化点积，不用 FFT。

    复杂度:
        时间 O(n * max_lag) / 空间 O(max_lag)。

    陷阱:
        这里刻意用 1/n 而不是 1/(n-k)：只有 1/n 归一化能保证自协方差矩阵半正定，
        Levinson-Durbin 递推才不会在中途出现负方差；代价是 k 接近 n 时 r[k] 明显偏小。

    参考:
        Brockwell & Davis, "Time Series: Theory and Methods", 2nd ed., §7.2。
    """
    n = x.size
    r = np.empty(max_lag + 1, dtype=float)
    r[0] = float(np.dot(x, x)) / n
    for k in range(1, max_lag + 1):
        r[k] = float(np.dot(x[k:], x[: n - k])) / n
    return r


def _levinson_durbin(r: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Levinson-Durbin 递推：由自协方差序列求 AR(order) 系数、偏自相关与残差方差。

    参数:
        r: 长度 >= order+1 的自协方差序列（r[0] 为方差）。
        order: AR 阶数 p。

    返回:
        (phi, pacf, sigma2)：phi 为长度 p 的 AR 系数（phi[j-1] 对应滞后 j），
        pacf 为长度 p 的偏自相关（pacf[k-1] = 第 k 阶的最后一个系数），
        sigma2 为阶数 p 下的残差方差。

    算法:
        标准递推：kappa_k = (r_k - sum_{j<k} phi_{k-1,j} r_{k-j}) / v_{k-1}；
        phi_{k,j} = phi_{k-1,j} - kappa_k * phi_{k-1,k-j}；v_k = v_{k-1}(1 - kappa_k^2)。

    复杂度:
        时间 O(p^2) / 空间 O(p)。

    陷阱:
        递推过程中 v_k 必须保持严格为正，等价于自协方差矩阵正定。若序列不平稳、
        或 p 取得过大（接近 n），v_k 会掉到 0 附近并让系数爆炸——本实现直接抛
        ValueError，而不是返回一串看起来正常的巨大系数。

    参考:
        Levinson (1947); Durbin (1960); Brockwell & Davis §5.2。
    """
    if order < 1:
        raise ValueError(f"阶数必须 >= 1，得到 {order}")
    if r.size < order + 1:
        raise ValueError(f"自协方差长度 {r.size} 不足以估计 {order} 阶模型")
    if not np.isfinite(r[0]) or r[0] <= 0.0:
        raise ValueError("序列方差必须为正（序列可能为常数）")

    phi = np.zeros(0, dtype=float)
    pacf = np.zeros(order, dtype=float)
    v = float(r[0])
    for k in range(1, order + 1):
        if k == 1:
            num = float(r[1])
        else:
            num = float(r[k]) - float(np.dot(phi, r[k - 1:0:-1]))
        kappa = num / v
        if k > 1:
            phi = phi - kappa * phi[::-1]
        phi = np.concatenate([phi, np.array([kappa])])
        pacf[k - 1] = kappa
        v = v * (1.0 - kappa * kappa)
        if not np.isfinite(v) or v <= 1e-300:
            raise ValueError(
                "Levinson-Durbin 递推中出现非正方差：序列可能不平稳，或 AR 阶数 p 过大"
            )
    return phi, pacf, v


# --------------------------------------------------------------------------
# 平滑类
# --------------------------------------------------------------------------

def moving_average(
    y: Sequence[float], window: int, centered: bool = False
) -> Dict[str, object]:
    """简单移动平均（SMA），返回平滑序列与首个有效位置。

    参数:
        y: 一维时间序列，按时间先后排列。
        window: 窗口长度（正整数，且不超过序列长度）。
        centered: True 表示居中窗口（用于提取趋势项）；False 表示只使用过去 window
            个观测的**尾部窗口**（可用于预测，因为不引入未来信息）。

    返回:
        dict：
        - ``trend``：与 ``y`` 等长的数组，无法计算的位置为 ``NaN``；
        - ``valid_from``：第一个有效值下标。尾部窗口的有效区间是 ``[valid_from, n-1]``；
          居中窗口的有效区间是 ``[valid_from, n-window+offset]``（``offset = window // 2``），
          区间两端都会缺值——注意本函数只返回左端点，右端点请自行用 ``n - window + offset`` 算。

    算法:
        尾部窗口：``trend[t] = mean(y[t-window+1 : t+1])``，``valid_from = window - 1``。
        居中窗口：``trend[t] = mean(y[t-offset : t-offset+window])``，``offset = window // 2``，
        ``valid_from = offset``。

    复杂度:
        时间 O(n * window)（滑动点积，未做前缀和优化）/ 空间 O(n)。

    陷阱:
        - **首尾确实会缺值**：尾部窗口缺前 window-1 个点，居中窗口首尾都缺。缺值一律写
          ``NaN``，不要用 ``0`` 或边界值填充：填 0 会让后续的误差指标（如 MAPE）直接炸掉。
        - **偶数窗口的居中平均并不对称**：``window=4`` 时 ``trend[t]`` 平均的是
          ``y[t-2..t+1]``，其重心落在 ``t-0.5``。论文里画趋势线时要说明这一点，
          否则和原始序列对不齐。
        - ``centered=False`` 的窗口是**单边**的，用它做"预测"实际上是拿最近 window 期的
          均值当下一期预测，存在 window/2 期左右的滞后，转折点会被抹平。

    参考:
        移动平均的教科书定义；"居中 vs 尾部"的区别见 Hyndman & Athanasopoulos,
        "Forecasting: Principles and Practice", 3rd ed., §3.3。
    """
    yv = as_vector(y, "y")
    w = _validate_positive_int(window, "window")
    n = yv.size
    if w > n:
        raise ValueError(f"window={w} 超过序列长度 {n}")

    trend = np.full(n, np.nan, dtype=float)
    if centered:
        offset = w // 2
        for t in range(offset, n - w + offset + 1):
            trend[t] = float(yv[t - offset: t - offset + w].mean())
        valid_from = offset
    else:
        for t in range(w - 1, n):
            trend[t] = float(yv[t - w + 1: t + 1].mean())
        valid_from = w - 1
    return {"trend": trend, "valid_from": int(valid_from)}


def exponential_smoothing(
    y: Sequence[float], alpha: float, initial: Optional[float] = None
) -> Dict[str, object]:
    """一次指数平滑（SES）：对水平项做指数加权。

    参数:
        y: 一维时间序列。
        alpha: 平滑系数，必须落在 (0, 1)。越大越贴近近期观测（alpha=1 时退化为朴素预测）。
        initial: 初始水平 l_0；``None`` 表示用 ``y[0]``。

    返回:
        dict：
        - ``fitted``：长度 n 的一步预测，``fitted[0] = l_0``，``fitted[t] = l_{t-1}``（t>=1）；
        - ``level``：最终水平 ``l_{n-1}``（标量）；
        - ``forecast``：下一期预测，等于 ``level``。

    算法:
        ``l_t = alpha * y_t + (1 - alpha) * l_{t-1}``；
        一步预测即为上一期水平：``yhat_t = l_{t-1}``。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - 指数平滑**不能**用来预测有明显趋势或季节性的序列：它预测出一条水平直线，
          对趋势数据会持续低估（或高估）。有趋势请用 ``holt_linear``，有季节性请用
          ``holt_winters``。
        - ``initial`` 的选择对短序列影响很大。用 ``y[0]`` 是常见默认，但当 ``y[0]``
          恰好是异常点时，前若干期预测都会被带偏。论文中应报告 initial 的取值方式。
        - 本函数**不做** alpha 的自动优化（MLE/最小 SSE），alpha 由使用者给定，
          这样每一步都能在论文里复现。

    参考:
        Brown (1956) 一次指数平滑；Hyndman & Athanasopoulos §8.1。
    """
    yv = as_vector(y, "y")
    a = _check_unit_interval(alpha, "alpha")
    n = yv.size

    l0 = float(yv[0]) if initial is None else float(initial)
    if not np.isfinite(l0):
        raise ValueError("initial 必须是有限数值")

    fitted = np.empty(n, dtype=float)
    fitted[0] = l0
    level = l0
    for t in range(1, n):
        fitted[t] = level
        level = a * float(yv[t]) + (1.0 - a) * level
    return {"fitted": fitted, "level": float(level), "forecast": float(level)}


def holt_linear(
    y: Sequence[float],
    alpha: float,
    beta: float,
    initial_level: Optional[float] = None,
    initial_trend: Optional[float] = None,
) -> Dict[str, object]:
    """Holt 线性趋势法：对水平项与趋势项分别做指数平滑。

    参数:
        y: 一维时间序列（长度 >= 2）。
        alpha: 水平平滑系数，落在 (0, 1)。
        beta: 趋势平滑系数，落在 (0, 1)。beta 越小趋势越"迟钝"。
        initial_level: 初始水平 l_0；``None`` 表示用 ``y[0]``。
        initial_trend: 初始趋势 b_0；``None`` 表示用 ``y[1] - y[0]``。

    返回:
        dict：
        - ``fitted``：长度 n 的一步预测，``fitted[0] = l_0``，``fitted[t] = l_{t-1} + b_{t-1}``；
        - ``level`` / ``trend``：最后一期的水平与趋势（标量）；
        - ``forecast``：下一期预测 ``l_{n-1} + b_{n-1}``。

    算法:
        ``yhat_t = l_{t-1} + b_{t-1}``；
        ``l_t = alpha*y_t + (1-alpha)*(l_{t-1} + b_{t-1})``；
        ``b_t = beta*(l_t - l_{t-1}) + (1-beta)*b_{t-1}``。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - Holt 法的预测是**直线**，会无限外推趋势；对存在周期或增长饱和的序列，
          外推几步之后就会明显偏离。长期预测务必先做残差诊断。
        - ``beta`` 过大（接近 1）时，趋势项会被最后一期的噪声主导，出现"预测线剧烈摆动"，
          这在实际数据上很常见，不是 bug。
        - 初始趋势用 ``y[1] - y[0]`` 对首两个观测的噪声极其敏感；序列前几个点波动大时，
          建议显式传入 ``initial_trend``（例如用前 1/4 段的平均斜率）。

    参考:
        Holt (1957); Hyndman & Athanasopoulos §8.2。
    """
    yv = as_vector(y, "y")
    if yv.size < 2:
        raise ValueError("holt_linear 至少需要 2 个观测")
    a = _check_unit_interval(alpha, "alpha")
    b = _check_unit_interval(beta, "beta")
    n = yv.size

    lvl = float(yv[0]) if initial_level is None else float(initial_level)
    trd = float(yv[1] - yv[0]) if initial_trend is None else float(initial_trend)
    if not np.isfinite(lvl) or not np.isfinite(trd):
        raise ValueError("initial_level / initial_trend 必须是有限数值")

    fitted = np.empty(n, dtype=float)
    fitted[0] = lvl
    for t in range(1, n):
        prev_l, prev_b = lvl, trd
        fitted[t] = prev_l + prev_b
        lvl = a * float(yv[t]) + (1.0 - a) * (prev_l + prev_b)
        trd = b * (lvl - prev_l) + (1.0 - b) * prev_b
    return {
        "fitted": fitted,
        "level": float(lvl),
        "trend": float(trd),
        "forecast": float(lvl + trd),
    }


def holt_winters(
    y: Sequence[float],
    period: int,
    alpha: float,
    beta: float,
    gamma: float,
    mode: str = "additive",
) -> Dict[str, object]:
    """Holt-Winters 三参数指数平滑（水平 + 趋势 + 季节）。

    参数:
        y: 一维时间序列，长度至少为 ``2 * period``。
        period: 季节周期 m（如月度数据 m=12、季度数据 m=4）。
        alpha / beta / gamma: 水平 / 趋势 / 季节的平滑系数，均落在 (0, 1)。
        mode: ``"additive"``（加法，季节幅度不随水平变化）或 ``"multiplicative"``
            （乘法，季节幅度与水平成正比）。

    返回:
        dict：
        - ``fitted``：长度 n 的一步预测，前 ``2*period`` 个位置为 ``NaN``（初始化期无历史）；
        - ``level`` / ``trend``：末期水平与趋势（标量）；
        - ``seasonal``：末期季节因子，长度 period（下标 j 对应 ``t % period == j``）；
        - ``forecast``：未来 1..period 期的预测，长度 period 的数组。

    算法:
        用前两个完整季节做经典分解初始化季节因子（加法：减去季节均值使和为 0；
        乘法：除以季节均值使均值为 1），水平初值取第一个季节均值，趋势初值取两季均值的
        平均斜率；随后对 t >= 2m 迭代：
        加法 ``l_t = alpha*(y_t - s_{t-m}) + (1-alpha)*(l_{t-1}+b_{t-1})``，
        ``s_t = gamma*(y_t - l_t) + (1-gamma)*s_{t-m}``；
        乘法把 ``y_t - s_{t-m}`` 换成 ``y_t / s_{t-m}``，``y_t - l_t`` 换成 ``y_t / l_t``。
        趋势式两者相同。

    复杂度:
        时间 O(n) / 空间 O(n + period)。

    陷阱:
        - **乘法模式要求数据严格为正**：含 0 或负值时 ``y_t / s_{t-m}`` 无意义，本实现直接
          抛 ``ValueError``，而不是返回 NaN/inf 让结果一路污染。若数据里有 0（例如"某月销量为 0"），
          正确做法是先做加法模式，或对数据做平移/对数变换并说明。
        - 乘法模式下如果初季节因子出现非正值（数据波动极端），同样抛 ``ValueError``。
        - 前 ``2*period`` 期没有预测值（``NaN``），这是初始化代价，不是数据缺失。用整段
          ``fitted`` 直接算 MAPE 会被 NaN 污染，请先用 ``np.isfinite`` 过滤。
        - ``period`` 必须由业务周期决定（月/季/周），**不要**用 ACF 峰值随口定；
          周期取错时 gamma 会把趋势的一部分"吸"进季节因子，预测形状会系统性走偏。

    参考:
        Winters (1960); Hyndman & Athanasopoulos §8.3。
    """
    yv = as_vector(y, "y")
    m = _validate_positive_int(period, "period")
    a = _check_unit_interval(alpha, "alpha")
    b = _check_unit_interval(beta, "beta")
    g = _check_unit_interval(gamma, "gamma")
    if mode not in ("additive", "multiplicative"):
        raise ValueError("mode 只能是 'additive' 或 'multiplicative'，得到 {0!r}".format(mode))
    n = yv.size
    if n < 2 * m:
        raise ValueError(f"Holt-Winters 至少需要 2*period = {2 * m} 个观测，得到 {n}")
    if mode == "multiplicative" and np.any(yv <= 0.0):
        raise ValueError(
            "乘法 Holt-Winters 要求所有观测严格为正；数据含非正值时请改用加法模式或先做平移/对数变换"
        )

    # ---- 经典分解初始化 ----
    level0 = float(yv[:m].mean())
    level1 = float(yv[m: 2 * m].mean())
    trend0 = (level1 - level0) / m
    t_idx = np.arange(2 * m, dtype=float)
    base = level0 + trend0 * t_idx
    if mode == "additive":
        detrended = yv[: 2 * m] - base
    else:
        if np.any(base <= 0.0):
            raise ValueError("乘法模式的初始水平/趋势非正，无法做经典分解初始化")
        detrended = yv[: 2 * m] / base
    raw = detrended.reshape(2, m).mean(axis=0)
    if mode == "additive":
        seas = raw - raw.mean()
    else:
        if np.any(raw <= 0.0):
            raise ValueError(
                "乘法模式下初季节因子出现非正值（数据波动过大），请改用加法模式"
            )
        seas = raw / raw.mean()

    lvl, trd = level0, trend0
    fitted = np.full(n, np.nan, dtype=float)
    for t in range(2 * m, n):
        j = t % m
        s_prev = float(seas[j])
        if mode == "additive":
            fitted[t] = lvl + trd + s_prev
            new_l = a * (float(yv[t]) - s_prev) + (1.0 - a) * (lvl + trd)
            new_b = b * (new_l - lvl) + (1.0 - b) * trd
            new_s = g * (float(yv[t]) - new_l) + (1.0 - g) * s_prev
        else:
            if abs(s_prev) < 1e-12:
                raise ValueError("乘法模式下季节因子退化为 0，无法继续递推")
            fitted[t] = (lvl + trd) * s_prev
            new_l = a * (float(yv[t]) / s_prev) + (1.0 - a) * (lvl + trd)
            new_b = b * (new_l - lvl) + (1.0 - b) * trd
            new_s = g * (float(yv[t]) / new_l) + (1.0 - g) * s_prev
        lvl, trd, seas[j] = new_l, new_b, new_s

    fc = np.empty(m, dtype=float)
    for h in range(1, m + 1):
        s_h = float(seas[(n + h - 1) % m])
        fc[h - 1] = (lvl + trd * h) + s_h if mode == "additive" else (lvl + trd * h) * s_h
    return {
        "fitted": fitted,
        "level": float(lvl),
        "trend": float(trd),
        "seasonal": seas.astype(float),
        "forecast": fc,
    }


# --------------------------------------------------------------------------
# 自相关 / 偏自相关 / AR
# --------------------------------------------------------------------------

def acf(y: Sequence[float], nlags: int) -> np.ndarray:
    """样本自相关函数（ACF）。

    参数:
        y: 一维时间序列。
        nlags: 最大滞后阶数，要求 ``0 <= nlags <= n-1``。

    返回:
        长度 ``nlags+1`` 的数组，下标 k 为滞后 k 的自相关，``acf[0] = 1``。

    算法:
        先中心化，再算 1/n 归一化自协方差 ``r_k``，最后 ``rho_k = r_k / r_0``。

    复杂度:
        时间 O(n * nlags) / 空间 O(nlags)。

    陷阱:
        - 归一化口径必须说清楚：本实现用 **1/n**（有偏），k 较大时自相关被系统性压低；
          有些软件用 1/(n-k)（无偏），两者在 n 小、k 大时可以差 10% 以上。
        - ACF 的 ±2/sqrt(n) 参考线只是**白噪声**下的渐近近似；序列有趋势或季节时，
          ACF 衰减极慢是正常现象，不能据此直接说"存在长期记忆"。
        - 有缺失值或异常点时 ACF 会被整体拉高，先做清洗再看图。

    参考:
        Brockwell & Davis §7.2；Box, Jenkins & Reinsel, "Time Series Analysis", §3.2.3。
    """
    yv = as_vector(y, "y")
    n = yv.size
    k = _validate_positive_int(nlags, "nlags", minimum=0)
    if k > n - 1:
        raise ValueError(f"nlags={k} 必须 <= n-1={n - 1}")
    x = yv - yv.mean()
    r = _autocovariance(x, k)
    if r[0] <= 0.0:
        raise ValueError("序列方差为 0（常数序列），无法计算自相关")
    return r / r[0]


def pacf(y: Sequence[float], nlags: int) -> np.ndarray:
    """样本偏自相关函数（PACF），用 Durbin-Levinson 递推。

    参数:
        y: 一维时间序列。
        nlags: 最大滞后阶数，要求 ``0 <= nlags <= n-2``。

    返回:
        长度 ``nlags+1`` 的数组，下标 k 为滞后 k 的偏自相关，``pacf[0] = 1``。

    算法:
        先由 ACF 得到自协方差 ``r_0..r_nlags``，再跑 Levinson-Durbin 递推；
        第 k 阶递推的最后一个系数 ``kappa_k`` 就是滞后 k 的偏自相关。**不使用**逐阶
        最小二乘回归法（后者在阶数高时数值上更不稳，且与 ACF 口径可能不一致）。

    复杂度:
        时间 O(n * nlags + nlags^2) / 空间 O(nlags)。

    陷阱:
        - PACF 的截尾/拖尾判读（AR(p) 的 PACF 在 p 阶后截尾）是**渐近**结论；n 小于
          约 50 时，PACF 的抽样波动很大，滞后 1 的 |PACF| 常常就能超过 0.3，不要据此
          把阶数定得过高。
        - 与 ``acf`` 一样用 1/n 归一化；若序列不平稳，Levinson-Durbin 递推可能因方差
          非正而报错——这本身就是"需要先差分"的信号。

    参考:
        Durbin (1960); Box, Jenkins & Reinsel §3.2.4。
    """
    yv = as_vector(y, "y")
    n = yv.size
    k = _validate_positive_int(nlags, "nlags", minimum=0)
    if k > n - 2:
        raise ValueError(f"nlags={k} 必须 <= n-2={n - 2}")
    x = yv - yv.mean()
    r = _autocovariance(x, k)
    if k == 0:
        return np.array([1.0])
    _, pac, _ = _levinson_durbin(r, k)
    return np.concatenate([np.array([1.0]), pac])


def ar_model(y: Sequence[float], p: int) -> Dict[str, object]:
    """用 Yule-Walker 方程估计带常数项的 AR(p) 模型。

    参数:
        y: 一维时间序列，长度 n 必须远大于 p。
        p: 自回归阶数（正整数）。

    返回:
        dict：
        - ``coef``：长度 p 的数组，``coef[j-1]`` 是滞后 j 项的系数 phi_j；
        - ``intercept``：常数项 ``mean(y) * (1 - sum(phi))``；
        - ``sigma2``：残差方差 ``r_0 - sum_j phi_j r_j``（白噪声方差估计）;
        - ``pacf``：长度 p 的偏自相关（Levinson-Durbin 递推的中间量）。

    算法:
        1. 去均值得到 ``x = y - mean(y)``；
        2. 用 1/n 归一化算自协方差 ``r_0..r_p``；
        3. 解 Yule-Walker 方程 ``R phi = r``，其中 R 是 Toeplitz 矩阵，用
           **Levinson-Durbin 递推**求解（O(p^2)，且顺带给出 PACF 与 sigma2）；
        4. 由去均值关系还原常数项。

    复杂度:
        时间 O(n * p + p^2) / 空间 O(n + p)。

    陷阱:
        - **只对平稳序列有意义**。含趋势/单位根的序列直接做 Yule-Walker 会得到
          ``sum(phi) ≈ 1`` 的伪回归结果，t 检验全部失效。先差分或先做 ADF 检验。
        - 归一化用 1/n 还是 1/(n-k) 会改变系数估计，本实现统一 1/n（保证正定）；
          论文里必须写明口径，否则和别人复现不出同样的数。
        - 这里估计的是**自协方差法（矩估计）**，不是条件最小二乘/极大似然。n 小或序列
          接近单位根时三者可以差不少；p 的选取不要靠目测 PACF，建议配合 AIC/BIC 或
          滚动外推误差（``rolling_origin_cv``）。
        - ``p`` 取得过大（接近 n）会让 Levinson-Durbin 递推出现非正方差并抛
          ``ValueError``，这是保护而不是失败。

    参考:
        Yule (1927); Walker (1931); Levinson (1947); Brockwell & Davis §8.1。
    """
    yv = as_vector(y, "y")
    order = _validate_positive_int(p, "p")
    n = yv.size
    if order > n - 2:
        raise ValueError(f"p={order} 过大：至少需要 p+2={order + 2} 个观测，得到 {n}")
    mu = float(yv.mean())
    x = yv - mu
    r = _autocovariance(x, order)
    phi, pac, sigma2 = _levinson_durbin(r, order)
    intercept = mu * (1.0 - float(phi.sum()))
    return {
        "coef": phi,
        "intercept": float(intercept),
        "sigma2": float(sigma2),
        "pacf": pac,
    }


def difference(
    y: Sequence[float], order: int = 1, seasonal: Optional[int] = None
) -> np.ndarray:
    """差分（普通差分 + 季节差分），返回差分后的序列。

    参数:
        y: 一维时间序列。
        order: 普通差分次数（>= 0）。0 表示不做普通差分。
        seasonal: 季节周期 m（>= 2）；``None`` 表示不做季节差分。只做一次季节差分
            （``y_t - y_{t-m}``），不做高阶季节差分。

    返回:
        差分后的数组，长度 ``n - order - (seasonal or 0)``。若长度 <= 0 抛 ``ValueError``。

    算法:
        **先做季节差分，再做普通差分**（顺序会改变中间量，虽然最终阶数一致时
        滞后算子可交换，但论文里报告的中间序列不同）。

    复杂度:
        时间 O((order + 1) * n) / 空间 O(n)。

    陷阱:
        - 差分顺序和次数必须交代清楚：``d=1, D=1`` 与 ``d=1`` 完全是两个模型。
        - **过度差分**会把白噪声放大成 MA 结构，预测反而变差。判断依据是差分后序列的
          ACF 是否在滞后 1 出现大幅负值（约 -0.5 以下）——出现就说明差分多了。
        - 差分会损失信息：季节差分至少丢掉一个整周期，短序列做两次以上差分后所剩无几。
        - 本函数**不做**逆差分（还原）——需要对预测值做逆变换时请自行累加，并注意季节差分
          的还原也要先还原普通差分。

    参考:
        Box, Jenkins & Reinsel §4.1；Hyndman & Athanasopoulos §9.1。
    """
    yv = as_vector(y, "y")
    d = _validate_positive_int(order, "order", minimum=0)
    m = 0 if seasonal is None else _validate_positive_int(seasonal, "seasonal", minimum=2)
    out = yv.copy()
    if m:
        if out.size <= m:
            raise ValueError(f"季节差分需要长度 > seasonal={m}，得到 {out.size}")
        out = out[m:] - out[:-m]
    for _ in range(d):
        if out.size <= 1:
            raise ValueError("差分后序列长度不足")
        out = np.diff(out)
    if out.size < 1:
        raise ValueError("差分后序列为空，请减小 order/seasonal")
    return out


def mackinnon_crit(
    nobs: int, regression: str = "c", level: float = 0.05
) -> float:
    """MacKinnon 响应面近似给出的 ADF/单位根检验临界值。

    参数:
        nobs: ADF 回归实际使用的观测数（不是原始序列长度）。
        regression: ``"c"``（含常数项、无趋势，ADF 最常用）、``"ct"``（含常数项和线性趋势）、
            ``"n"``（无常数项）。不接受 ``"ctt"``（常数 + 线性 + 二次趋势，本模块不实现）。
        level: 显著性水平，仅支持 0.01 / 0.05 / 0.10。

    返回:
        临界值（float，负值）。检验统计量小于该值即在该水平上拒绝"存在单位根"的原假设。

    算法:
        响应面回归形式 ``crit(n) = b0 + b1/n + b2/n^2 + b3/n^3``，系数取自
        MacKinnon (2010) Table 1 的 N=1 情形（本模块文件头部常量
        ``_MACKINNON_2010_N1``，每个系数旁标注了出处）；``n -> inf`` 时 ``crit -> b0``，
        即教科书中常引用的渐近临界值（含常数项情形约为 -3.43 / -2.86 / -2.57）。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        - **临界值是查表近似**：样本量很小时（例如 nobs < 20）误差较大，而且这里只有
          N=1（单变量）的系数，**不能**用于协整检验（N>1 时需要另一张表）。
        - ``nobs`` 必须传"回归里真正用到的观测数"：差分一次、又加了 p 个滞后项之后，
          nobs 比原始序列长度小。传错会让临界值与统计量不匹配，结论可能反转。
        - 响应面给出的是**渐近分布**的分位点近似；序列有结构突变、条件异方差时，
          实际检验水平会明显偏离名义水平。正式论文建议用 statsmodels 的 ``adfuller``
          （同一张 MacKinnon 表）或自举法（bootstrap）复核。
        - 本模块不提供 MacKinnon 的 p 值近似（Thompson 型响应面），只给临界值；
          需要精确 p 值请用 statsmodels。

    参考:
        MacKinnon, J.G. (1994) "Approximate Asymptotic Distribution Functions for Unit-Root
        and Cointegration Tests", Journal of Business & Economic Statistics 12(2): 167-176；
        MacKinnon, J.G. (2010) "Critical Values for Cointegration Tests", Queen's Economics
        Department Working Paper No. 1227, Table 1。
    """
    reg = _REGRESSION_ALIASES.get(str(regression).lower())
    if reg is None:
        raise ValueError(
            "regression 只能是 'c' / 'ct' / 'n'（'ctt' 未实现），得到 {0!r}".format(regression)
        )
    lv = float(level)
    table = _MACKINNON_2010_N1[reg]
    if lv not in table:
        raise ValueError(f"level 只支持 0.01 / 0.05 / 0.10，得到 {level!r}")
    n = _validate_positive_int(nobs, "nobs")
    if n < _MIN_NOBS_FOR_CRIT:
        raise ValueError(
            f"nobs={n} 太小，响应面近似不可靠（建议至少 {_MIN_NOBS_FOR_CRIT}，且实际建模中远大于此）"
        )
    b0, b1, b2, b3 = table[lv]
    inv = 1.0 / float(n)
    return float(b0 + b1 * inv + b2 * inv ** 2 + b3 * inv ** 3)


def adf_test(
    y: Sequence[float], max_lag: Optional[int] = None, regression: str = "c"
) -> Dict[str, object]:
    """增广 Dickey-Fuller（ADF）单位根检验：自己构造回归并算 t 统计量。

    参数:
        y: 一维时间序列。
        max_lag: 滞后阶数上限。``None`` 表示用 Schwert 经验规则
            ``int(ceil(12 * (n/100)^0.25))``；给定整数则在该上限内用 AIC 选阶。
        regression: ``"c"``（含常数项）、``"ct"``（含常数项与线性趋势）、``"n"``（无常数项）。

    返回:
        dict：
        - ``stat``：gamma（滞后水平项系数）的 t 统计量；
        - ``lags``：AIC 选出的滞后阶数 p；
        - ``nobs``：回归实际使用的观测数（用于查临界值）；
        - ``crit``：``{"1%": ..., "5%": ..., "10%": ...}`` 临界值字典。

    算法:
        回归式 ``dy_t = a + gamma * y_{t-1} + sum_{i=1}^{p} d_i * dy_{t-i} + e_t``
        （``regression="ct"`` 时再加一个时间趋势项），用最小二乘闭式解
        ``beta = (X'X)^{-1} X'y`` 估计，``se(gamma) = sqrt(sigma2 * [(X'X)^{-1}]_gg)``，
        ``sigma2 = RSS / (nobs - k)``；在 p = 0..max_lag 中选 AIC 最小的阶数，
        AIC 口径为 ``nobs * ln(RSS/nobs) + 2k``。临界值来自 ``mackinnon_crit``。

    复杂度:
        时间 O(max_lag * n * max_lag^2)（每个候选阶数解一次最小二乘）/ 空间 O(n * max_lag)。

    陷阱:
        - ADF 的原假设是**存在单位根（序列非平稳）**：统计量**越小（越负）**才越倾向拒绝。
          把结论说反是本类检验最常见的低级错误。
        - 滞后阶数选法会改变统计量：本实现固定用 AIC，和 statsmodels 默认的
          ``autolag='AIC'`` 可能差 1 阶，统计量因此可能差 0.1 以上。论文里应写明
          ``max_lag`` 与选阶准则。
        - **ADF 对结构突变无能为力**：序列在样本中期发生水平跳变时，ADF 会误判为单位根
          （Perron 批评）。有突变应先做突变检验或分段处理。
        - 回归里必须包含足够的滞后项以消除残差自相关；nobs 随之减少，临界值也随 nobs 变化，
          本实现已把 ``nobs`` 一并返回供核对。
        - 只做一次 ADF 就下结论是不够的：建议同时看 ACF 衰减、PP 检验或 KPSS 检验
          （原假设相反）作为交叉证据。

    参考:
        Dickey & Fuller (1979, JASA 74:427-431); Said & Dickey (1984);
        Schwert (1989, JBES 7:147-159); MacKinnon (2010)。
    """
    yv = as_vector(y, "y")
    reg = _REGRESSION_ALIASES.get(str(regression).lower())
    if reg is None:
        raise ValueError("regression 只能是 'c' / 'ct' / 'n'，得到 {0!r}".format(regression))
    n = yv.size
    if n < 10:
        raise ValueError(f"ADF 检验需要至少 10 个观测，得到 {n}")
    if max_lag is None:
        ml = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    else:
        ml = _validate_positive_int(max_lag, "max_lag", minimum=0)
    # 保证每个候选阶数都有足够多的观测解回归
    ml = min(ml, max(0, (n - 1) // 2 - 3))
    if ml < 0:
        raise ValueError("序列过短，无法构造 ADF 回归")

    dy = np.diff(yv)
    m = dy.size
    trend = np.arange(n, dtype=float)

    def _build(p: int):
        """构造 p 阶的 ADF 回归矩阵与响应，返回 (X, yy, n_used)。"""
        idx = np.arange(p, m)  # 对应 dy[t] 的行下标
        yy = dy[idx]
        cols: List[np.ndarray] = []
        if p > 0:
            for j in range(1, p + 1):
                cols.append(dy[idx - j])
        cols.append(yv[idx])  # 滞后水平 y_{t-1}（0 基下标 t 即上一期水平）
        if reg in ("c", "ct"):
            cols.append(np.ones(idx.size, dtype=float))
        if reg == "ct":
            cols.append(trend[idx])
        X = np.column_stack(cols)
        return X, yy, idx.size

    def _fit(p: int):
        X, yy, n_used = _build(p)
        k = X.shape[1]
        if n_used <= k:
            return None
        xtx = X.T @ X
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            return None
        beta = xtx_inv @ (X.T @ yy)
        resid = yy - X @ beta
        rss = float(resid @ resid)
        if not np.isfinite(rss) or rss <= 0.0:
            return None
        sigma2 = rss / (n_used - k)
        se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
        # gamma 的位置：p 个差分滞后项之后的第一列
        gidx = p
        if se[gidx] <= 0.0:
            return None
        stat = float(beta[gidx] / se[gidx])
        aic = n_used * np.log(rss / n_used) + 2.0 * k
        return {"stat": stat, "aic": float(aic), "nobs": int(n_used), "beta": beta}

    best: Optional[Dict[str, object]] = None
    best_p = 0
    for p in range(0, ml + 1):
        fit = _fit(p)
        if fit is None:
            continue
        if best is None or float(fit["aic"]) < float(best["aic"]):
            best, best_p = fit, p
    if best is None:
        raise ValueError("ADF 回归无法估计（样本过短或数值奇异）")

    nobs = int(best["nobs"])
    crit = {
        "1%": mackinnon_crit(nobs, reg, 0.01),
        "5%": mackinnon_crit(nobs, reg, 0.05),
        "10%": mackinnon_crit(nobs, reg, 0.10),
    }
    return {"stat": float(best["stat"]), "lags": int(best_p), "nobs": nobs, "crit": crit}


# --------------------------------------------------------------------------
# 精度指标
# --------------------------------------------------------------------------

def _paired(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    """把真值与预测值转成一维等长数组。"""
    a = as_vector(y_true, "y_true")
    b = as_vector(y_pred, "y_pred")
    check_same_length(a, b)
    return a, b


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """平均绝对百分比误差 MAPE（返回百分数，如 12.3 表示 12.3%）。

    参数:
        y_true: 真实值，等长一维序列。
        y_pred: 预测值。

    返回:
        float，``100 * mean(|(y_true - y_pred) / y_true|)``。

    算法:
        逐点算绝对百分比误差再取均值。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - **真值中出现 0 时必须报错**：除以 0 会得到 inf，均值变成 inf/NaN，图表全毁。
          本实现直接抛 ``ValueError`` 而不是返回 inf。真值接近 0 时 MAPE 同样会爆炸
          （分母 0.01 会把微小绝对误差放大成 1000%），这种数据应当改用 MAE/RMSE 或
          MASE，并在论文里说明为什么不用 MAPE。
        - MAPE 对**高估与低估不对称**：它惩罚"预测值偏大"的力度小于"偏小"，
          因此最小值追踪类模型（如需求预测）用 MAPE 选参数会系统性偏向低估。
        - 真值有正有负时 MAPE 没有意义（误差百分比可能为负、均值会被抵消）。

    参考:
        Hyndman & Koehler (2006) "Another look at measures of forecast accuracy",
        International Journal of Forecasting 22(4): 679-688。
    """
    a, b = _paired(y_true, y_pred)
    if np.any(a == 0.0):
        raise ValueError("mape: y_true 含 0，百分比误差无定义（请改用 mae/rmse）")
    return float(np.mean(np.abs((a - b) / a)) * 100.0)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """均方根误差 RMSE。

    参数:
        y_true: 真实值。
        y_pred: 预测值。

    返回:
        float，``sqrt(mean((y_true - y_pred)^2))``，与数据同量纲。

    算法:
        误差平方取均值后开方（分母为 n，是总体口径而非 n-1；这是预测误差的通用定义）。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - RMSE 对**大误差特别敏感**（平方放大），少数离群点就能主导数值；比较模型时
          若一条序列含突变点，RMSE 的排序可能完全由那个点决定，建议同时报告 MAE。
        - 只报告 RMSE 而不说明数据量纲是没意义的（"RMSE=3.2" 无法判断好坏），
          论文里应给出 RMSE/均值 或用 MASE 归一化。

    参考:
        Hyndman & Koehler (2006)。
    """
    a, b = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """平均绝对误差 MAE。

    参数:
        y_true: 真实值。
        y_pred: 预测值。

    返回:
        float，``mean(|y_true - y_pred|)``。

    算法:
        绝对误差取均值。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        MAE 的最优预测是**条件中位数**而 RMSE 的最优预测是条件均值：两条指标同时报告
        才说明误差分布形状。只看 MAE 会掩盖少数大偏差（例如极端月份的严重低估）。

    参考:
        Hyndman & Koehler (2006)。
    """
    a, b = _paired(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def theil_u(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Theil U2 不等系数（无量纲，0 表示完美预测，1 表示与"全预测 0"一样差）。

    参数:
        y_true: 真实值。
        y_pred: 预测值。

    返回:
        float，``sqrt(sum((y_pred - y_true)^2)) / (sqrt(sum(y_true^2)) + sqrt(sum(y_pred^2)))``。

    算法:
        分子是预测的 RMS 误差（未除以 n），分母是真实值与预测值各自 RMS 之和；
        该口径下 ``U2`` 落在 [0, 1]（预测与真值成比例且系数为负等极端情形除外）。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - Theil 有 **U1 与 U2 两个不同定义**，数值完全不同：U1 是
          ``sqrt(mean(((y_pred-y_true)/y_true)^2))``（同样会被 0 值毁掉）。
          论文里必须写明用的是哪一个，本实现是 U2。
        - U2 的分母含预测值自身，因此**模型之间只有在同一测试集上才可比**；跨数据集
          比较 U2 大小是无意义的。
        - 真值与预测值全为 0 时分母为 0，本实现抛 ``ValueError``。

    参考:
        Theil, H. (1966) "Applied Economic Forecasting", North-Holland, Ch. 2。
    """
    a, b = _paired(y_true, y_pred)
    den = float(np.sqrt(np.sum(a ** 2)) + np.sqrt(np.sum(b ** 2)))
    if den <= 0.0:
        raise ValueError("theil_u: 真值与预测值全为 0，不等系数无定义")
    return float(np.sqrt(np.sum((b - a) ** 2)) / den)


# --------------------------------------------------------------------------
# 切分与回测
# --------------------------------------------------------------------------

def train_test_split_ts(
    y: Sequence[float], test_size: Union[int, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """时间序列的**顺序**切分：前段训练、后段测试。

    参数:
        y: 一维时间序列，按时间先后排列。
        test_size: 测试集长度。``int`` 表示条数；``float``（落在 (0,1)）表示比例，
            实际条数取 ``ceil(n * test_size)``。

    返回:
        ``(train, test)`` 两个数组，``train`` 是前缀、``test`` 是后缀，
        且 ``len(train) + len(test) == n``，``test`` 的最后一点就是序列的最后一点。

    算法:
        下标切分：``train = y[:n-k]``，``test = y[n-k:]``，**没有任何洗牌**。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - **时间序列绝对不能随机切分**。随机抽点会让"用未来的点预测过去"这种信息泄漏
          进入测试集，得到远低于真实水平的误差；一旦真的用于决策就会翻车。
          这也是本函数不提供 ``shuffle`` 参数的原因。
        - 单次切分的测试集可能恰好落在某个特殊季节/突变段上，误差不具代表性；
          正式报告建议用 ``rolling_origin_cv`` 做多折滚动外推。
        - 切分后不能跨边界做平滑/差分/标准化：差分要用 ``y[n-k-1]`` 才能得到测试集第一个
          差分值，标准化统计量只能用训练段计算，否则同样是信息泄漏。

    参考:
        Hyndman & Athanasopoulos §5.1、§5.3（"randomly sampled" 对时间序列为何不适用）。
    """
    yv = as_vector(y, "y")
    n = yv.size
    if isinstance(test_size, bool):
        raise ValueError("test_size 必须是 int 或 (0,1) 内的 float")
    if isinstance(test_size, (int, np.integer)):
        k = int(test_size)
    elif isinstance(test_size, (float, np.floating)):
        ts = float(test_size)
        if not (0.0 < ts < 1.0):
            raise ValueError(f"float 形式的 test_size 必须落在 (0,1) 内，得到 {ts}")
        k = int(np.ceil(n * ts))
    else:
        raise ValueError("test_size 必须是 int 或 (0,1) 内的 float")
    if k < 1:
        raise ValueError(f"测试集长度必须 >= 1，得到 {k}")
    if k >= n:
        raise ValueError(f"测试集长度 {k} 必须小于序列长度 {n}（训练集不能为空）")
    return yv[: n - k], yv[n - k:]


def rolling_origin_cv(
    y: Sequence[float], initial: int, horizon: int, step: int = 1
) -> List[Tuple[int, int, int]]:
    """滚动原点交叉验证（rolling-origin / walk-forward）的折索引。

    参数:
        y: 一维时间序列。
        initial: 第一折的训练窗口长度（>= 1）。
        horizon: 每折的预测步长（>= 1）。
        step: 相邻两折训练窗口前进的步数（>= 1）。

    返回:
        折列表，每折为 ``(train_end, test_start, test_end)`` 三元组（都是下标，右端**不含**）：
        - 训练集 = ``y[:train_end]``，测试集 = ``y[test_start:test_end]``；
        - 本实现固定 ``test_start == train_end``（测试窗口紧接训练窗口，中间不跳空）。

    算法:
        令 ``train_end`` 从 ``initial`` 开始、每折加 ``step``，只要
        ``train_end + horizon <= n`` 就生成一折 ``(train_end, train_end, train_end + horizon)``。

    复杂度:
        时间 O(n / step)（只生成索引，不训练模型）/ 空间 O(n / step)。

    陷阱:
        - 训练窗口是**扩张**（expanding）的：``y[:train_end]`` 随折数变长。如果想让窗口
          固定长度（滑动窗口），请自行取 ``y[train_end-window:train_end]``，本函数只给边界。
        - ``initial`` 必须留出足够长的训练段（至少要覆盖模型阶数 + 一个完整季节周期），
          否则前几折的"预测"其实是随机数，会污染汇总误差。
        - 各折误差**不能简单平均**就下结论：不同折的点位不同（有的折落在旺季），
          建议同时报告每折误差与折间波动；``horizon > 1`` 时还应按步长分别汇总。
        - 不做任何洗牌，折与折之间存在重叠，误差序列本身是自相关的，不能套用
          "独立同分布"的置信区间公式。

    参考:
        Hyndman & Athanasopoulos §5.4（tsCV / rolling forecast origin）；
        Tashman, L.J. (2000) "Out-of-sample tests of forecasting accuracy",
        International Journal of Forecasting 16(4): 437-450。
    """
    yv = as_vector(y, "y")
    n = yv.size
    init = _validate_positive_int(initial, "initial")
    hor = _validate_positive_int(horizon, "horizon")
    st = _validate_positive_int(step, "step")
    if init + hor > n:
        raise ValueError(
            f"initial + horizon = {init + hor} 超过序列长度 {n}，无法构造任何一折"
        )
    folds: List[Tuple[int, int, int]] = []
    train_end = init
    while train_end + hor <= n:
        folds.append((int(train_end), int(train_end), int(train_end + hor)))
        train_end += st
    return folds


# --------------------------------------------------------------------------
# 朴素基线 / 经典分解 / 乘法 Holt-Winters
# --------------------------------------------------------------------------

#: ``naive_forecast`` 支持的基线方法。
_NAIVE_METHODS = ("last", "mean", "drift", "seasonal")


def _centered_moving_average(x: np.ndarray, period: int) -> np.ndarray:
    """周期为 ``period`` 的居中移动平均（用于提取趋势），首尾无法计算处填 ``NaN``。

    参数:
        x: 一维序列（已过 ``as_vector`` 校验）。
        period: 季节周期 m，必须与调用方的 ``period`` 一致。

    返回:
        与 ``x`` 等长的数组。``period`` 为奇数时是等权窗口 ``x[t-half..t+half]``；
        为偶数时是 2×m 移动平均（端点权重 0.5、其余 1，再除以 m），
        有效区间都是 ``[half, n-1-half]``（``half = period // 2``）。

    算法:
        奇数 m：``trend[t] = mean(x[t-half : t+half+1])``；
        偶数 m：``trend[t] = (0.5*x[t-half] + x[t-half+1] + ... + x[t+half-1] + 0.5*x[t+half]) / m``。

    复杂度:
        时间 O(n * period) / 空间 O(period)。

    陷阱:
        偶数周期的 2×m 平均权重不对称地落在端点上（t-half 与 t+half 同相位、各占 0.5），
        这正是经典分解的通行口径；它能把**线性趋势**逐点精确复现，但对二次以上的趋势有偏。

    参考:
        Hyndman & Athanasopoulos, "Forecasting: Principles and Practice", 3rd ed., §3.4。
    """
    n = x.size
    trend = np.full(n, np.nan, dtype=float)
    half = period // 2
    if period % 2 == 1:
        for t in range(half, n - half):
            trend[t] = float(x[t - half: t + half + 1].mean())
    else:
        w = np.ones(period + 1, dtype=float)
        w[0] = 0.5
        w[-1] = 0.5
        w /= float(period)
        for t in range(half, n - half):
            trend[t] = float(np.dot(w, x[t - half: t + half + 1]))
    return trend


def _decomposition_strength(resid: np.ndarray, component: np.ndarray) -> float:
    """分解强度 ``max(0, 1 - Var(resid) / Var(component))``（只在两者都有限的位置上算）。

    参数:
        resid: 残差序列（可能含 ``NaN``）。
        component: 对照分量（趋势强度用 ``x - trend``，季节强度用 ``x - seasonal``；
            乘法模型下用对应的比值序列）。

    返回:
        float，落在 [0, 1]；``component`` 方差为 0 时返回 1.0（残差也为 0）或 0.0。

    算法:
        取 ``resid`` 与 ``component`` 都有限的位置，算两者的总体方差（分母 n）再套公式。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        只在 ``[half, n-1-half]`` 上有值，若序列长度刚够 ``2*period``，可用点数很少，
        强度估计极不稳定；这也是经典分解在短序列上不可信的原因。

    参考:
        Wang, Smith & Hyndman (2006) "Characteristic-based clustering for time series data",
        Data Mining and Knowledge Discovery 13(3): 335-364（强度指标出处）。
    """
    mask = np.isfinite(resid) & np.isfinite(component)
    if int(mask.sum()) < 2:
        return 0.0
    var_r = float(np.var(resid[mask]))
    var_c = float(np.var(component[mask]))
    if var_c <= 0.0:
        return 1.0 if var_r <= 0.0 else 0.0
    return max(0.0, 1.0 - var_r / var_c)


def naive_forecast(
    x: Sequence[float],
    n_ahead: int = 1,
    method: str = "last",
    period: Optional[int] = None,
) -> Dict[str, object]:
    """朴素预测基线：最后一期 / 历史均值 / 线性漂移 / 上一周期。

    参数:
        x: 一维时间序列，按时间先后排列（长度 >= 1；``"drift"`` 需要 >= 2）。
        n_ahead: 预测步数（>= 1）。
        method: ``"last"``（用 ``x[-1]`` 平推）、``"mean"``（用 ``mean(x)`` 平推）、
            ``"drift"``（从 ``x[0]`` 到 ``x[-1]`` 的直线外推）、
            ``"seasonal"``（用上一周期的同相位观测，需要 ``period``）。
        period: 季节周期 m，仅 ``method="seasonal"`` 时使用；``None`` 会抛 ``ValueError``。

    返回:
        dict，键为：
        ``forecast``  长度 ``n_ahead`` 的预测数组，``forecast[h-1]`` 是未来第 h 期；
        ``method``    实际使用的方法名（str）。

    算法:
        1. ``"last"``：``f_h = x_{n-1}``；
        2. ``"mean"``：``f_h = mean(x)``；
        3. ``"drift"``：``f_h = x_{n-1} + h * (x_{n-1} - x_0) / (n - 1)``；
        4. ``"seasonal"``：``f_h = x_{n - m + ((h-1) % m)}``（下标对 n 取模的同相位观测，
           例如 n=12、m=4 时 ``f_1 = x_8``）。
        四种方法都只用历史数据，不含任何待估参数。

    复杂度:
        时间 O(n + n_ahead) / 空间 O(n_ahead)。

    陷阱:
        - 朴素方法在基准对比里是**及格线而不是模型**：一个复杂模型若打不赢
          ``"last"``/``"seasonal"``，说明它的参数估计没有带来信息（Hyndman 的 MASE
          正是用季节朴素法做分母）。论文里必须报告这条基线。
        - ``"seasonal"`` 的相位对齐依赖 ``x`` 的起点：若序列被截断过（例如从某年 3 月开始），
          ``period`` 的相位就错了，误差会凭空变大。截断数据要先按相位补齐或改用
          ``seasonal_decompose`` 显式估季节因子。
        - ``"drift"`` 把首末两点的噪声当成斜率，对首末点的异常值极其敏感；
          外推步数越多，误差线性放大。
        - 高估方法：``"mean"`` 对含趋势的序列系统性滞后；不要因为它"误差看起来平滑"就选用。

    参考:
        Hyndman & Athanasopoulos §5.2（naive / seasonal naive / drift methods）；
        Hyndman & Koehler (2006)（MASE 以季节朴素法为基准）。
    """
    xv = as_vector(x, "x")
    h = _validate_positive_int(n_ahead, "n_ahead")
    if method not in _NAIVE_METHODS:
        raise ValueError(
            "method 只能是 {0} 之一，得到 {1!r}".format(", ".join(_NAIVE_METHODS), method)
        )
    n = xv.size
    if method == "last":
        fc = np.full(h, float(xv[-1]), dtype=float)
    elif method == "mean":
        fc = np.full(h, float(xv.mean()), dtype=float)
    elif method == "drift":
        if n < 2:
            raise ValueError(f"method='drift' 至少需要 2 个观测，得到 {n}")
        slope = float(xv[-1] - xv[0]) / float(n - 1)
        fc = np.array([float(xv[-1]) + slope * k for k in range(1, h + 1)], dtype=float)
    else:
        m = _validate_positive_int(period, "period")
        if n < m:
            raise ValueError(f"method='seasonal' 需要至少 period = {m} 个观测，得到 {n}")
        fc = np.array(
            [float(xv[n - m + ((k - 1) % m)]) for k in range(1, h + 1)], dtype=float
        )
    return {"forecast": fc, "method": str(method)}


def seasonal_decompose(
    x: Sequence[float],
    period: int,
    model: str = "additive",
    n_iter: int = 2,
) -> Dict[str, object]:
    """经典季节分解：居中移动平均取趋势 + 按相位平均取季节项 + 残差。

    参数:
        x: 一维时间序列，长度至少为 ``2 * period``。
        period: 季节周期 m（如季度 m=4、月度 m=12）。
        model: ``"additive"``（``x = trend + seasonal + resid``）或
            ``"multiplicative"``（``x = trend * seasonal * resid``，要求 ``x`` 严格为正）。
        n_iter: 分解迭代次数（>= 1）；第 1 轮用原始序列取趋势，之后每轮先用上一轮的
            季节项把序列去季节再重估趋势，反复冲洗掉"季节项混进趋势"的污染。

    返回:
        dict，键为：
        ``trend``     形状 (n,) 的居中移动平均，首尾 ``half = period // 2`` 个位置为 ``NaN``；
        ``seasonal``  形状 (n,) 的周期延拓季节项（``seasonal[t]`` 对应相位 ``t % period``，
                      因此 ``seasonal[:period]`` 就是一个完整周期的因子；加法模型下均值为 0，
                      乘法模型下均值为 1）；
        ``resid``     形状 (n,) 的残差，``trend`` 为 ``NaN`` 处同样为 ``NaN``；
                      （加法：``x - trend - seasonal``；乘法：``x / (trend * seasonal)``）；
        ``strength_trend``    float，``max(0, 1 - Var(resid) / Var(x - trend))``；
        ``strength_seasonal`` float，``max(0, 1 - Var(resid) / Var(x - seasonal))``；
                      两者的方差都只在 ``resid`` 与对照分量都有限的位置上计算（`_decomposition_strength`），
                      乘法模型下这两个对照分量改为比值序列 ``x / trend``、``x / seasonal``。

    算法:
        1. 令 ``seasonal`` 初值为 0（加法）或 1（乘法），重复 ``n_iter`` 轮：
           2. ``adjusted = x - seasonal``（加法）或 ``x / seasonal``（乘法）；
              ``trend = _centered_moving_average(adjusted, period)``；
           3. ``detrended = x - trend``（加法）或 ``x / trend``（乘法）；
           4. 对每个相位 ``j = 0..period-1`` 取 ``detrended`` 在 ``t % period == j`` 上的
              均值（忽略 ``NaN``）得原始季节因子；加法减去其均值、乘法除以其均值归一；
           5. 把因子按相位延拓成长度 n 的 ``seasonal``。
        6. 由最终的 ``trend`` 与 ``seasonal`` 算 ``resid``，再用 `_decomposition_strength` 算两个强度。
        与 STL 不同，这里不做局部加权回归、不迭代内循环，季节项在全样本上是**常数**。

    复杂度:
        时间 O(n_iter * n * period) / 空间 O(n)。

    陷阱:
        - **移动平均趋势本身会被季节项污染**：序列长度不是周期的整数倍、或真正的季节形态
          随时间变化（幅度增长/形状漂移）时，居中平均里残留的季节成分会被塞进趋势，
          表现为趋势线出现周期性"波纹"。这正是要多迭代 ``n_iter`` 次的原因，但迭代不能
          根治，只是减轻；形态漂移明显时应改用 STL 类方法。
        - 首尾 ``half`` 个点没有趋势值，因此**残差与强度都只在中段计算**；若序列长度刚好
          ``2*period``，可用点极少，强度会非常不稳定，不要据此下结论。
        - 乘法模型要求 ``x``、趋势与初季节因子全为正，本实现对非正值直接抛 ``ValueError``，
          而不是返回 ``NaN`` 让残差一路污染；含 0 的"某月销量为 0"应改加法模式或先平移/取对数。
        - ``period`` 取错时季节项会把趋势吸收进去、残差看起来"变小"，强度指标反而变好看——
          强度高不等于模型对，必须与 ACF 的周期证据、业务周期一起判断。
        - 本函数的分解是**确定性描述**，不是概率模型，不提供预测；要预测请用
          ``holt_winters`` / ``holt_winters_multiplicative``。

    参考:
        Hyndman & Athanasopoulos §3.4（classical decomposition）；
        Wang, Smith & Hyndman (2006)（强度指标）。
    """
    xv = as_vector(x, "x")
    m = _validate_positive_int(period, "period")
    iters = _validate_positive_int(n_iter, "n_iter")
    if model not in ("additive", "multiplicative"):
        raise ValueError(f"model 只能是 'additive' 或 'multiplicative'，得到 {model!r}")
    n = xv.size
    if n < 2 * m:
        raise ValueError(f"seasonal_decompose 至少需要 2*period = {2 * m} 个观测，得到 {n}")
    if model == "multiplicative" and np.any(xv <= 0.0):
        raise ValueError(
            "乘法分解要求所有观测严格为正；含非正值时请改用加法模式或先做平移/对数变换"
        )

    seasonal_rep = (
        np.ones(n, dtype=float) if model == "multiplicative" else np.zeros(n, dtype=float)
    )
    phase_index = np.arange(n) % m
    trend = np.full(n, np.nan, dtype=float)
    for _ in range(iters):
        adjusted = xv - seasonal_rep if model == "additive" else xv / seasonal_rep
        trend = _centered_moving_average(adjusted, m)
        finite_trend = trend[np.isfinite(trend)]
        if model == "multiplicative" and np.any(finite_trend <= 0.0):
            raise ValueError("乘法分解的趋势项出现非正值，请改用加法模式")
        detrended = xv - trend if model == "additive" else xv / trend

        raw = np.full(m, np.nan, dtype=float)
        for j in range(m):
            vals = detrended[phase_index == j]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                raise ValueError(f"季节相位 {j} 没有任何可用的去趋势观测，无法估计季节因子")
            raw[j] = float(vals.mean())
        if model == "additive":
            raw = raw - raw.mean()
        else:
            if np.any(raw <= 0.0):
                raise ValueError(
                    "乘法分解的季节因子出现非正值（数据波动过大），请改用加法模式"
                )
            raw = raw / raw.mean()
        seasonal_rep = raw[phase_index]

    if model == "additive":
        resid = xv - trend - seasonal_rep
        detrended_all = xv - trend
        deseasonalized_all = xv - seasonal_rep
    else:
        resid = xv / (trend * seasonal_rep)
        detrended_all = xv / trend
        deseasonalized_all = xv / seasonal_rep

    return {
        "trend": trend,
        "seasonal": seasonal_rep,
        "resid": resid,
        "strength_trend": float(_decomposition_strength(resid, detrended_all)),
        "strength_seasonal": float(_decomposition_strength(resid, deseasonalized_all)),
    }


def holt_winters_multiplicative(
    x: Sequence[float],
    period: int,
    alpha: float,
    beta: float,
    gamma: float,
    n_ahead: int = 1,
) -> Dict[str, object]:
    """乘法季节 Holt-Winters 三参数指数平滑（水平 × 趋势 × 季节因子）。

    参数:
        x: 一维时间序列，长度至少为 ``2 * period``，且**必须严格为正**。
        period: 季节周期 m（如季度 m=4、月度 m=12）。
        alpha / beta / gamma: 水平 / 趋势 / 季节的平滑系数，均落在 (0, 1)。
        n_ahead: 样本外预测步数（>= 1），本函数不再限制为 period。

    返回:
        dict，键为：
        ``fitted``    长度 n 的一步预测，前 ``2*period`` 个位置为 ``NaN``（初始化期无历史）；
        ``forecast``  长度 ``n_ahead`` 的多步预测，``forecast[h-1]`` 是未来第 h 期；
        ``level``     末期水平 l（标量）；
        ``trend``     末期趋势 b（标量）；
        ``seasonal``  末期季节因子，长度 period，下标 j 对应 ``t % period == j``，均值近似 1。

    算法:
        初始化与 ``holt_winters(mode="multiplicative")`` 完全一致：用前两个完整季节做经典分解
        （``base = level0 + trend0 * t``，``detrended = y / base``，两个季节按相位平均后
        除以其均值），水平初值取第一季均值、趋势初值取两季均值的平均斜率；随后对
        ``t >= 2m``（``j = t % period``）迭代
        ``yhat_t = (l_{t-1} + b_{t-1}) * s_{t-m}``；
        ``l_t = alpha*(y_t / s_{t-m}) + (1-alpha)*(l_{t-1} + b_{t-1})``；
        ``b_t = beta*(l_t - l_{t-1}) + (1-beta)*b_{t-1}``；
        ``s_t = gamma*(y_t / l_t) + (1-gamma)*s_{t-m}``；
        预测 ``f_h = (l + b*h) * s_{(n+h-1) % period}``。

    复杂度:
        时间 O(n + n_ahead) / 空间 O(n + period)。

    陷阱:
        - **乘法模式要求数据严格为正**：含 0 或负值时 ``y_t / s_{t-m}`` 无意义，本实现直接
          抛 ``ValueError``（与 ``holt_winters`` 同口径），而不是返回 ``NaN``/``inf`` 污染后续指标。
        - 季节因子若被数据波动推到 0 附近，递推里的除法会爆炸；本实现对退化的 ``s_prev`` 与
          ``l_t`` 都显式抛 ``ValueError``。乘法模型的常见兜底是先取对数再套**加法**模式。
        - 序列长度刚好 ``2*period`` 时递推循环一次都不执行，``fitted`` 全是 ``NaN``、只有
          ``level``/``trend``/``seasonal`` 和 ``forecast`` 可用；这不是 bug 而是初始化代价。
        - 乘法模型的 ``fitted`` 首 ``2*period`` 个位置是 ``NaN``，直接整段算 MAPE 会被污染，
          请先用 ``np.isfinite`` 过滤（与 ``holt_winters`` 的说明一致）。
        - 与 ``holt_winters`` 的差别只有两点：强制乘法、``forecast`` 长度由 ``n_ahead`` 决定。
          两者在同一 ``(x, period, alpha, beta, gamma)`` 下的 ``fitted``、水平和前 period 步
          预测必须逐点一致（``_self_test`` 里做了对拍）。

    参考:
        Winters (1960) "Forecasting sales by exponentially weighted moving averages",
        Management Science 6(3): 324-342；Hyndman & Athanasopoulos §8.3。
    """
    xv = as_vector(x, "x")
    m = _validate_positive_int(period, "period")
    a = _check_unit_interval(alpha, "alpha")
    b = _check_unit_interval(beta, "beta")
    g = _check_unit_interval(gamma, "gamma")
    h = _validate_positive_int(n_ahead, "n_ahead")
    n = xv.size
    if n < 2 * m:
        raise ValueError(f"乘法 Holt-Winters 至少需要 2*period = {2 * m} 个观测，得到 {n}")
    if np.any(xv <= 0.0):
        raise ValueError(
            "乘法 Holt-Winters 要求所有观测严格为正；数据含非正值时请改用 holt_winters "
            "的加法模式或先做平移/对数变换"
        )

    # ---- 经典分解初始化（与 holt_winters 的乘法分支逐字一致）----
    level0 = float(xv[:m].mean())
    level1 = float(xv[m: 2 * m].mean())
    trend0 = (level1 - level0) / m
    base = level0 + trend0 * np.arange(2 * m, dtype=float)
    if np.any(base <= 0.0):
        raise ValueError("乘法模式的初始水平/趋势非正，无法做经典分解初始化")
    raw = (xv[: 2 * m] / base).reshape(2, m).mean(axis=0)
    if np.any(raw <= 0.0):
        raise ValueError("乘法模式下初季节因子出现非正值（数据波动过大），请改用加法模式")
    seas = raw / raw.mean()

    lvl, trd = level0, trend0
    fitted = np.full(n, np.nan, dtype=float)
    for t in range(2 * m, n):
        j = t % m
        s_prev = float(seas[j])
        if abs(s_prev) < 1e-12:
            raise ValueError("乘法模式下季节因子退化为 0，无法继续递推")
        fitted[t] = (lvl + trd) * s_prev
        new_l = a * (float(xv[t]) / s_prev) + (1.0 - a) * (lvl + trd)
        if not np.isfinite(new_l) or abs(new_l) < 1e-12:
            raise ValueError("乘法模式下水平项退化为 0，无法继续递推")
        new_b = b * (new_l - lvl) + (1.0 - b) * trd
        new_s = g * (float(xv[t]) / new_l) + (1.0 - g) * s_prev
        lvl, trd, seas[j] = new_l, new_b, new_s

    fc = np.array(
        [(lvl + trd * k) * float(seas[(n + k - 1) % m]) for k in range(1, h + 1)],
        dtype=float,
    )
    return {
        "fitted": fitted,
        "forecast": fc,
        "level": float(lvl),
        "trend": float(trd),
        "seasonal": seas.astype(float),
    }


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。"""
    out: Dict[str, object] = {}

    # 1) 移动平均：尾部窗口首 window-1 个位置为 NaN；居中窗口首尾都缺
    y_ma = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    ma_tail = moving_average(y_ma, 3)
    ma_cent = moving_average(y_ma, 3, centered=True)
    out["ma_tail_valid_from"] = int(ma_tail["valid_from"])
    out["ma_tail_first"] = round(float(ma_tail["trend"][2]), 6)
    out["ma_tail_nan_lead"] = int(np.sum(~np.isfinite(ma_tail["trend"])))
    out["ma_centered_valid_from"] = int(ma_cent["valid_from"])
    out["ma_centered_nan_total"] = int(np.sum(~np.isfinite(ma_cent["trend"])))
    out["ma_centered_mid"] = round(float(ma_cent["trend"][3]), 6)

    # 2) 一次指数平滑：常数序列必须完全复现（任何 alpha 下 level 恒等于该常数）
    es = exponential_smoothing([5.0, 5.0, 5.0, 5.0], 0.3)
    out["es_level_const_series"] = round(float(es["level"]), 9)
    es2 = exponential_smoothing([1.0, 2.0, 3.0, 4.0], 0.5, initial=1.0)
    out["es_level_ramp"] = round(float(es2["level"]), 6)
    out["es_fitted_first"] = round(float(es2["fitted"][0]), 6)

    # 3) Holt 线性：对严格线性序列，末期趋势必须回到真实斜率附近
    ramp = np.arange(1.0, 21.0)
    hl = holt_linear(ramp, 0.6, 0.4)
    out["holt_trend_on_ramp"] = round(float(hl["trend"]), 6)
    out["holt_forecast_on_ramp"] = round(float(hl["forecast"]), 6)

    # 4) Holt-Winters：加法/乘法都能跑通，乘法预测必须全正
    r = rng(20240101)
    t = np.arange(48, dtype=float)
    seas_add = 3.0 * np.sin(2 * np.pi * t / 12.0)
    y_add = 20.0 + 0.5 * t + seas_add + r.normal(scale=0.3, size=48)
    hw_a = holt_winters(y_add, 12, 0.35, 0.1, 0.35, mode="additive")
    out["hw_add_forecast_head"] = [round(float(v), 6) for v in hw_a["forecast"][:3]]
    out["hw_add_seasonal_sum"] = round(float(np.sum(hw_a["seasonal"])), 6)
    out["hw_add_fitted_nan"] = int(np.sum(~np.isfinite(hw_a["fitted"])))

    y_mul = (20.0 + 0.5 * t) * (1.0 + 0.15 * np.sin(2 * np.pi * t / 12.0))
    hw_m = holt_winters(y_mul, 12, 0.35, 0.1, 0.35, mode="multiplicative")
    out["hw_mul_forecast_head"] = [round(float(v), 6) for v in hw_m["forecast"][:3]]
    out["hw_mul_all_positive"] = bool(np.all(hw_m["forecast"] > 0.0))
    out["hw_mul_seasonal_mean"] = round(float(np.mean(hw_m["seasonal"])), 6)

    # 乘法模式遇到非正值必须抛 ValueError（真实陷阱，不是 warning）
    try:
        holt_winters([1.0] * 23 + [0.0], 12, 0.3, 0.1, 0.3, mode="multiplicative")
        out["hw_mul_nonpositive_raises"] = False
    except ValueError:
        out["hw_mul_nonpositive_raises"] = True

    # 5) AR(2)：Yule-Walker 系数应接近真值 (0.5, -0.3)
    r2 = rng(7)
    n_ar = 4000
    e = r2.normal(size=n_ar)
    x = np.zeros(n_ar)
    for i in range(2, n_ar):
        x[i] = 0.5 * x[i - 1] - 0.3 * x[i - 2] + e[i]
    ar = ar_model(x, 2)
    out["ar2_coef"] = [round(float(v), 6) for v in ar["coef"]]
    out["ar2_max_abs_err"] = round(
        float(np.max(np.abs(np.asarray(ar["coef"]) - np.array([0.5, -0.3])))), 6
    )
    out["ar2_sigma2"] = round(float(ar["sigma2"]), 6)
    out["ar2_intercept"] = round(float(ar["intercept"]), 6)

    # 6) ACF / PACF：白噪声的 ACF 应落在 ±2/sqrt(n) 内；AR(1) 的 PACF 在滞后 2 之后应很小
    r3 = rng(11)
    wn = r3.normal(size=2000)
    a_wn = acf(wn, 10)
    out["acf_wn_lag1"] = round(float(a_wn[1]), 6)
    out["acf_wn_max_abs_tail"] = round(float(np.max(np.abs(a_wn[1:]))), 6)
    out["acf_wn_bound_2_over_sqrt_n"] = round(float(2.0 / np.sqrt(2000.0)), 6)
    out["acf_wn_within_bound"] = bool(np.all(np.abs(a_wn[1:]) <= 2.0 / np.sqrt(2000.0)))

    x1 = np.zeros(2000)
    e1 = r3.normal(size=2000)
    for i in range(1, 2000):
        x1[i] = 0.7 * x1[i - 1] + e1[i]
    pa = pacf(x1, 8)
    out["pacf_ar1_lag1"] = round(float(pa[1]), 6)
    out["pacf_ar1_max_abs_after_lag1"] = round(float(np.max(np.abs(pa[2:]))), 6)

    # 7) ADF：随机游走不应拒绝单位根（stat 明显大于 5% 临界值）；白噪声应拒绝
    r4 = rng(2024)
    rw = np.cumsum(r4.normal(size=400))
    adf_rw = adf_test(rw)
    out["adf_random_walk_stat"] = round(float(adf_rw["stat"]), 6)
    out["adf_random_walk_lags"] = int(adf_rw["lags"])
    out["adf_random_walk_nobs"] = int(adf_rw["nobs"])
    out["adf_random_walk_crit5"] = round(float(adf_rw["crit"]["5%"]), 6)
    out["adf_random_walk_not_reject"] = bool(float(adf_rw["stat"]) > float(adf_rw["crit"]["5%"]))
    adf_wn = adf_test(r4.normal(size=400))
    out["adf_white_noise_stat"] = round(float(adf_wn["stat"]), 6)
    out["adf_white_noise_reject"] = bool(float(adf_wn["stat"]) < float(adf_wn["crit"]["5%"]))

    # 8) MacKinnon 临界值：渐近值与教科书常引用的 -3.43 / -2.86 / -2.57 一致
    out["mackinnon_c_1pct_asym"] = round(float(mackinnon_crit(10 ** 9, "c", 0.01)), 5)
    out["mackinnon_c_5pct_asym"] = round(float(mackinnon_crit(10 ** 9, "c", 0.05)), 5)
    out["mackinnon_c_10pct_asym"] = round(float(mackinnon_crit(10 ** 9, "c", 0.10)), 5)
    out["mackinnon_c_5pct_n100"] = round(float(mackinnon_crit(100, "c", 0.05)), 6)
    out["mackinnon_ct_5pct_n100"] = round(float(mackinnon_crit(100, "ct", 0.05)), 6)
    out["mackinnon_n_5pct_n100"] = round(float(mackinnon_crit(100, "n", 0.05)), 6)

    # 9) 差分：一阶差分 + 季节差分后的长度
    series = np.arange(1.0, 25.0)
    out["diff_order1_len"] = int(difference(series, 1).size)
    out["diff_order1_first"] = round(float(difference(series, 1)[0]), 6)
    out["diff_seasonal12_then_order1_len"] = int(difference(series, 1, seasonal=12).size)
    out["diff_order0_is_identity"] = bool(np.allclose(difference(series, 0), series))

    # 10) 精度指标
    yt = np.array([100.0, 120.0, 90.0, 110.0])
    yp = np.array([105.0, 115.0, 95.0, 100.0])
    out["mape"] = round(mape(yt, yp), 6)
    out["rmse"] = round(rmse(yt, yp), 6)
    out["mae"] = round(mae(yt, yp), 6)
    out["theil_u"] = round(theil_u(yt, yp), 6)
    try:
        mape([1.0, 0.0, 3.0], [1.0, 1.0, 3.0])
        out["mape_zero_raises"] = False
    except ValueError:
        out["mape_zero_raises"] = True

    # 11) 切分与回测索引
    tr, te = train_test_split_ts(np.arange(20.0), 0.25)
    out["split_train_len"] = int(tr.size)
    out["split_test_len"] = int(te.size)
    out["split_test_head"] = round(float(te[0]), 6)
    out["split_test_tail"] = round(float(te[-1]), 6)
    folds = rolling_origin_cv(np.arange(20.0), initial=10, horizon=3, step=2)
    out["cv_n_folds"] = len(folds)
    out["cv_first_fold"] = [int(v) for v in folds[0]]
    out["cv_last_fold"] = [int(v) for v in folds[-1]]

    # 12) 朴素基线：纯周期序列上 seasonal 基线必须一步不差；drift 在线性序列上等于直线外推
    cyc = np.tile(np.array([10.0, 20.0, 30.0, 40.0]), 3)
    nf_sea = naive_forecast(cyc, n_ahead=4, method="seasonal", period=4)
    sea_err = float(np.max(np.abs(np.asarray(nf_sea["forecast"]) - np.array([10.0, 20.0, 30.0, 40.0]))))
    if sea_err > 1e-12:
        raise AssertionError(f"naive_forecast(seasonal) 在纯周期序列上的最大误差应为 0，得到 {sea_err}")
    if str(nf_sea["method"]) != "seasonal":
        raise AssertionError(f"naive_forecast 的 method 回显错误：{nf_sea['method']!r}")
    out["naive_seasonal_max_abs_err"] = round(sea_err, 12)
    out["naive_seasonal_forecast"] = [round(float(v), 6) for v in nf_sea["forecast"]]
    out["naive_seasonal_method"] = str(nf_sea["method"])

    ramp10 = np.arange(1.0, 11.0)
    nf_drift = naive_forecast(ramp10, n_ahead=3, method="drift")
    if not np.allclose(np.asarray(nf_drift["forecast"]), np.array([11.0, 12.0, 13.0]), rtol=0.0, atol=1e-12):
        raise AssertionError(f"drift 基线对线性序列应外推为 11,12,13，得到 {nf_drift['forecast']}")
    out["naive_drift_head"] = [round(float(v), 6) for v in nf_drift["forecast"]]
    out["naive_last_forecast"] = round(float(naive_forecast(ramp10, 2, "last")["forecast"][0]), 6)
    out["naive_mean_forecast"] = round(float(naive_forecast(ramp10, 1, "mean")["forecast"][0]), 6)
    try:
        naive_forecast(cyc, 1, "seasonal")
        out["naive_seasonal_needs_period"] = False
    except ValueError:
        out["naive_seasonal_needs_period"] = True

    # 13) 经典分解：x = 线性趋势 + 周期季节项 + 0 时，残差必须恒为 0、季节强度必须为 1
    t4 = np.arange(24, dtype=float)
    se4 = np.array([-3.0, 1.0, 4.0, -2.0])  # 一个周期的季节项，和恰为 0
    x4 = (5.0 + 0.5 * t4) + se4[np.arange(24) % 4]
    sd = seasonal_decompose(x4, 4, model="additive", n_iter=2)
    sd_resid = np.asarray(sd["resid"])
    sd_finite = np.isfinite(sd_resid)
    sd_max_resid = float(np.max(np.abs(sd_resid[sd_finite])))
    # 阈值：无残差合成序列的残差应到浮点噪声量级（取 1e-9，实测为 0.0）
    if sd_max_resid > 1e-9:
        raise AssertionError(f"无残差合成序列的分解残差应约为 0，得到 {sd_max_resid}")
    # 阈值：Var(resid)/Var(去季节序列) = 0，故强度应恰为 1；取 0.999 容忍浮点与窗口边界
    if float(sd["strength_seasonal"]) < 0.999:
        raise AssertionError(f"strength_seasonal 应接近 1，得到 {sd['strength_seasonal']}")
    if float(sd["strength_trend"]) < 0.999:
        raise AssertionError(f"strength_trend 应接近 1，得到 {sd['strength_trend']}")
    sd_fac = np.asarray(sd["seasonal"])[:4]
    if float(np.max(np.abs(sd_fac - se4))) > 1e-9:
        raise AssertionError(f"分解出的季节因子应等于真值 {se4}，得到 {sd_fac}")
    out["sdecomp_resid_max_abs"] = round(sd_max_resid, 12)
    out["sdecomp_strength_seasonal"] = round(float(sd["strength_seasonal"]), 6)
    out["sdecomp_strength_trend"] = round(float(sd["strength_trend"]), 6)
    out["sdecomp_trend_mid"] = round(float(sd["trend"][12]), 6)
    out["sdecomp_seasonal_factors"] = [round(float(v), 6) for v in sd_fac]
    out["sdecomp_seasonal_mean"] = round(float(sd_fac.mean()), 9)
    out["sdecomp_resid_nan_count"] = int(np.sum(~sd_finite))

    # 乘法分解：x = 线性趋势 × (1 + 0.2 sin) 时季节因子应回到真值、残差应接近 1
    xm = (5.0 + 0.5 * t4) * (1.0 + 0.2 * np.sin(2 * np.pi * t4 / 4.0))
    sdm = seasonal_decompose(xm, 4, model="multiplicative", n_iter=5)
    true_fac = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(4) / 4.0)
    true_fac = true_fac / true_fac.mean()
    mul_fac_err = float(np.max(np.abs(np.asarray(sdm["seasonal"])[:4] - true_fac)))
    sdm_resid = np.asarray(sdm["resid"])
    mul_resid_dev = float(np.max(np.abs(sdm_resid[np.isfinite(sdm_resid)] - 1.0)))
    # 阈值：迭代 5 轮后实测 2.1e-9 / 9.2e-9，取 1e-6
    if mul_fac_err > 1e-6:
        raise AssertionError(f"乘法分解的季节因子应回到真值，最大偏差 {mul_fac_err}")
    if mul_resid_dev > 1e-6:
        raise AssertionError(f"乘法分解的残差应接近 1，最大偏差 {mul_resid_dev}")
    out["sdecomp_mul_seasonal_max_err"] = round(mul_fac_err, 12)
    out["sdecomp_mul_resid_max_dev"] = round(mul_resid_dev, 12)
    try:
        seasonal_decompose(x4, 4, model="log-additive")
        out["sdecomp_bad_model_raises"] = False
    except ValueError:
        out["sdecomp_bad_model_raises"] = True

    # 14) 乘法 Holt-Winters：常数水平 × 周期季节因子上，拟合值必须一步不差地复现原序列
    xh = 50.0 * np.array([0.8, 1.1, 1.3, 0.8])[np.arange(12) % 4]
    hwm = holt_winters_multiplicative(xh, 4, 0.4, 0.2, 0.3, n_ahead=5)
    hwm_fit = np.asarray(hwm["fitted"])
    hwm_ok = np.isfinite(hwm_fit)
    hwm_rel = float(np.max(np.abs(hwm_fit[hwm_ok] - xh[hwm_ok]) / xh[hwm_ok]))
    # 阈值：常数水平下递推是恒等映射，实测相对误差 0.0；取 1e-6
    if hwm_rel > 1e-6:
        raise AssertionError(f"常数水平×季节因子序列的乘法 HW 拟合相对误差应为 0，得到 {hwm_rel}")
    if abs(float(hwm["trend"])) > 1e-9:
        raise AssertionError(f"常数水平序列的趋势项应为 0，得到 {hwm['trend']}")
    if abs(float(hwm["level"]) - 50.0) > 1e-9:
        raise AssertionError(f"常数水平序列的末期水平应为 50，得到 {hwm['level']}")
    # 独立实现互证：与已有的 holt_winters(mode="multiplicative") 逐点一致
    hw_ref = holt_winters(xh, 4, 0.4, 0.2, 0.3, mode="multiplicative")
    if not np.allclose(
        np.asarray(hwm["forecast"])[:4], np.asarray(hw_ref["forecast"]), rtol=0.0, atol=1e-9
    ):
        raise AssertionError(
            f"乘法 HW 的前 4 步预测应与 holt_winters 一致：{hwm['forecast']} vs {hw_ref['forecast']}"
        )
    if not np.allclose(
        hwm_fit[hwm_ok], np.asarray(hw_ref["fitted"])[hwm_ok], rtol=0.0, atol=1e-12
    ):
        raise AssertionError("乘法 HW 的 fitted 应与 holt_winters(mode='multiplicative') 一致")
    hwm_fc = np.asarray(hwm["forecast"])
    out["hwm_fitted_max_rel_err"] = round(hwm_rel, 12)
    out["hwm_forecast"] = [round(float(v), 6) for v in hwm_fc]
    out["hwm_forecast_periodic"] = bool(abs(float(hwm_fc[4]) - float(hwm_fc[0])) <= 1e-9)
    out["hwm_seasonal"] = [round(float(v), 6) for v in np.asarray(hwm["seasonal"])]
    out["hwm_level"] = round(float(hwm["level"]), 6)
    out["hwm_trend"] = round(float(hwm["trend"]), 9)
    out["hwm_matches_holt_winters"] = True
    try:
        holt_winters_multiplicative([1.0] * 11 + [0.0], 4, 0.3, 0.1, 0.3)
        out["hwm_nonpositive_raises"] = False
    except ValueError:
        out["hwm_nonpositive_raises"] = True

    return out
