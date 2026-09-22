"""统计推断与回归：相关系数、t 检验、卡方、正态性检验、非参数检验与方差分析、
OLS/Newey-West/岭回归/Lasso/logistic/泊松回归、Bootstrap 与置换检验、
PCA 与因子分析、共线性与回归诊断、逐步回归。

本模块共 29 个公开函数，按用途分组：相关 ``pearson_corr`` / ``spearman_corr`` /
``kendall_tau``，假设检验 ``t_test_one_sample`` / ``t_test_two_sample`` / ``chi_square_test`` /
``shapiro_wilk`` / ``jarque_bera`` / ``anderson_darling`` / ``ks_test_normal``，
非参数检验与方差分析 ``mann_whitney_u`` / ``wilcoxon_signed_rank`` / ``kruskal_wallis`` /
``anova_oneway``，回归与诊断 ``ols`` / ``newey_west_se`` / ``vif`` / ``ridge_regression`` /
``lasso_regression`` / ``logistic_regression`` / ``poisson_regression`` /
``stepwise_selection`` / ``durbin_watson`` / ``breusch_pagan``，降维 ``pca`` /
``factor_analysis``，重抽样 ``bootstrap_ci`` / ``bca_bootstrap_ci`` / ``permutation_test``。

本模块的共同约定
----------------
- **只依赖 numpy 与 Python 标准库**：正态/学生 t/卡方/F 分布的分位数与尾概率全部用
  连分式/级数自己实现（``_betainc``、``_gammainc_q`` 等），不 import scipy。
  这些私有函数在交付报告中与 ``scipy.special`` / ``scipy.stats`` 做过数值对比。
- **p 值一律是双侧 p 值**（除卡方、JB、AD 等本身就是单侧右尾的检验），返回字典里统一叫
  ``p_value``；论文中引用时请注明是双侧。
- **近似的地方都会写明近似**：小样本、并列值、参数由样本估计等情形下的 p 值都只是近似，
  正式论文请用 scipy / statsmodels 复核（见 ``references/github-resources.md``）。
- 随机过程显式接收 ``seed``，默认 ``_common.DEFAULT_SEED``（``_common.rng``），不使用全局随机状态。
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ._common import as_matrix, as_vector, check_same_length, rng

__all__ = [
    "pearson_corr",
    "spearman_corr",
    "kendall_tau",
    "t_test_one_sample",
    "t_test_two_sample",
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    "anova_oneway",
    "chi_square_test",
    "shapiro_wilk",
    "jarque_bera",
    "anderson_darling",
    "ks_test_normal",
    "ols",
    "newey_west_se",
    "vif",
    "ridge_regression",
    "logistic_regression",
    "bootstrap_ci",
    "bca_bootstrap_ci",
    "permutation_test",
    "pca",
    "factor_analysis",
    "lasso_regression",
    "poisson_regression",
    "durbin_watson",
    "breusch_pagan",
    "stepwise_selection",
]

# --------------------------------------------------------------------------
# 分布函数（纯 numpy + math 实现，逐标量）
# --------------------------------------------------------------------------

_ITMAX = 300
_EPS = 3.0e-16
_FPMIN = 1.0e-300
_SQRT2 = math.sqrt(2.0)


def _norm_cdf(z: float) -> float:
    """标准正态累积分布函数（用 math.erf，精度约 1e-15）。

    参数:
        z: 标量。

    返回:
        P(Z <= z)。

    算法:
        ``0.5 * (1 + erf(z / sqrt(2)))``。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        z 绝对值大于约 8 时结果会舍入到 0 或 1，直接取对数会得到 -inf；
        需要尾概率时请用 ``_norm_sf`` 或对数形式。

    参考:
        Abramowitz & Stegun (1964) 式 26.2.29 附近的 erf 展开（此处用标准库 math.erf）。
    """
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _norm_sf(z: float) -> float:
    """标准正态上尾概率 P(Z > z)。

    参数:
        z: 标量。

    返回:
        P(Z > z)，用 ``0.5 * erfc(z / sqrt(2))`` 计算，右尾比 ``1 - cdf`` 稳定得多。

    算法:
        直接调用 math.erfc，避免 1-cdf 的灾难性抵消。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        z < -30 时结果舍入为 1；极端尾概率请用对数渐近式。

    参考:
        math.erfc 定义。
    """
    return 0.5 * math.erfc(z / _SQRT2)


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数的连分式部分（Lentz 算法）。

    参数:
        a, b: 形状参数（> 0）。
        x: 自变量，落在 [0, 1]。

    返回:
        连分式的值（正数）。

    算法:
        修正 Lentz 算法迭代连分式，最多 ``_ITMAX`` 次或相对变化小于 ``_EPS``。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        分母可能趋近 0，必须用 ``_FPMIN`` 托底，否则会除零得到 inf/NaN。

    参考:
        Press et al., "Numerical Recipes", 3rd ed., §6.4（betacf）。
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。

    参数:
        a, b: 形状参数（> 0）。
        x: 自变量，落在 [0, 1]。

    返回:
        I_x(a,b) = B(x; a, b) / B(a, b)，落在 [0, 1]。

    算法:
        先用 lgamma 算前置因子，再按 ``x < (a+1)/(a+b+2)`` 选择正向或反向连分式，
        保证连分式始终收敛得快。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        x 接近 0 或 1 时前置因子会下溢到 0，但此时结果本身就趋近 0 或 1，属于可接受舍入。

    参考:
        Press et al. §6.4（betai）；与 ``scipy.special.betainc`` 数值一致（见交付报告）。
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _gammainc_q(a: float, x: float) -> float:
    """正则化上不完全 Gamma 函数 Q(a, x) = 1 - P(a, x)。

    参数:
        a: 形状参数（> 0）。
        x: 自变量（>= 0）。

    返回:
        Q(a, x)，即 Gamma(a) 分布的上尾概率。

    算法:
        ``x < a+1`` 时用级数算 P 再取 1-P；否则用连分式直接算 Q（避免抵消）。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        用 ``1 - P`` 的方式在 x 很大时会因抵消丢失全部有效位，因此必须分支用连分式。

    参考:
        Press et al. §6.2（gammp / gammq）。
    """
    if x < 0.0 or a <= 0.0:
        raise ValueError("_gammainc_q 要求 a > 0, x >= 0")
    if x == 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(_ITMAX):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * _EPS:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _t_sf_two_sided(t: float, df: float) -> float:
    """学生 t 分布的双侧尾概率 P(|T_df| > |t|)。

    参数:
        t: t 统计量。
        df: 自由度（> 0）。

    返回:
        双侧 p 值，落在 [0, 1]。

    算法:
        用恒等式 ``P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2)``。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        自由度必须为正；df 为非整数（Welch 校正）时该恒等式依然成立，这是选它的原因。

    参考:
        Abramowitz & Stegun (1964) 式 26.7.4 附近的不完全 Beta 表示。
    """
    if df <= 0.0:
        raise ValueError("t 分布自由度必须为正")
    if not np.isfinite(t):
        return 0.0
    return _betainc(0.5 * df, 0.5, df / (df + t * t))


def _chi2_sf(x: float, df: float) -> float:
    """卡方分布上尾概率 P(X > x)，X ~ chi2_df。

    参数:
        x: 统计量。
        df: 自由度（> 0）。

    返回:
        右尾 p 值。

    算法:
        ``P(X > x) = Q(df/2, x/2)``（正则化上不完全 Gamma）。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        df=2 时退化为 ``exp(-x/2)``，可用作实现正确性的解析校验（见 ``_self_test``）。

    参考:
        Press et al. §6.2。
    """
    if df <= 0.0:
        raise ValueError("卡方自由度必须为正")
    if x <= 0.0:
        return 1.0
    return _gammainc_q(0.5 * df, 0.5 * x)


def _f_sf(f: float, d1: float, d2: float) -> float:
    """F 分布上尾概率 P(F > f)，F ~ F(d1, d2)。

    参数:
        f: 统计量（>= 0）。
        d1, d2: 分子/分母自由度（> 0）。

    返回:
        右尾 p 值。

    算法:
        ``P(F > f) = I_{d2/(d2 + d1 f)}(d2/2, d1/2)``。

    复杂度:
        时间 O(迭代次数) / 空间 O(1)。

    陷阱:
        f 极大时 ``d2/(d2+d1 f)`` 下溢到 0，p 值返回 0，这在数值上是"小于机器精度"，
        论文里应写成 p < 1e-16 而不是 p = 0。

    参考:
        Abramowitz & Stegun (1964) 式 26.6.2 附近的 Beta 表示。
    """
    if d1 <= 0.0 or d2 <= 0.0:
        raise ValueError("F 分布自由度必须为正")
    if f <= 0.0:
        return 1.0
    return _betainc(0.5 * d2, 0.5 * d1, d2 / (d2 + d1 * f))


def _rank_average(x: np.ndarray) -> np.ndarray:
    """平均秩变换（并列值取平均名次），从 1 开始。

    参数:
        x: 一维数组。

    返回:
        与 x 等长的秩数组。

    算法:
        排序后扫描等值段，把该段所有元素赋为段内名次的平均值。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        并列值的处理方式会影响 Spearman 结果：取平均秩是标准做法；
        若改成"先出现者名次靠前"，同一份数据能算出不同的相关系数。

    参考:
        scipy.stats.rankdata(method='average') 的口径。
    """
    a = np.asarray(x, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i: j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _norm_ppf(p: float) -> float:
    """标准正态分位数（逆累积分布函数），二分定位 + 牛顿迭代收尾。

    参数:
        p: 概率，必须严格落在 ``(0, 1)`` 内。

    返回:
        满足 ``_norm_cdf(z) = p`` 的 z（精度约 1e-15）。

    算法:
        先按 ``p`` 落在哪一侧选区间（``p < 0.5`` 用 ``[-40, 0]`` 且以 ``_norm_sf``
        为单调目标，避免 ``1 - cdf`` 的抵消；否则用 ``[0, 40]`` 且以 ``_norm_cdf``
        为目标），二分 120 步把区间压到机器精度以下；再用 3 步牛顿迭代
        （导数即标准正态密度 ``exp(-z^2/2)/sqrt(2*pi)``）做一次精修。

    复杂度:
        时间 O(1)（固定 120 步 + 3 步） / 空间 O(1)。

    陷阱:
        - **下尾不能无限往下走**：``math.erf`` 在 ``|z| > 8.4`` 附近饱和到 ±1，
          ``p < 1e-17`` 时根会被钉在饱和边界上。BCa 的偏差校正比例请先裁剪到
          ``[1/(B+1), B/(B+1)]`` 再调用本函数（``bca_bootstrap_ci`` 已这样做）。
        - 参数校验失败会抛 ``ValueError``（``p = 0`` 或 ``1`` 的分位数是 ±∞，本函数不返回 inf）。
        - 用固定步数的二分而不是查表，是为了让结果**逐位可复现**：同一 p 在任何平台上
          都返回同一个 double。

    参考:
        Wichura, M.J. (1988) "Algorithm AS 241: The Percentage Points of the Normal
        Distribution", Applied Statistics 37(3): 477-484（本实现只借用其精度目标，
        迭代方式改为二分 + 牛顿）。
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"_norm_ppf 要求 p 严格落在 (0,1) 内，得到 {p!r}")
    if p < 0.5:
        target = 1.0 - p
        lo, hi = -40.0, 0.0
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if _norm_sf(mid) > target:
                lo = mid
            else:
                hi = mid
    else:
        target = p
        lo, hi = 0.0, 40.0
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if _norm_cdf(mid) < target:
                lo = mid
            else:
                hi = mid
    z = 0.5 * (lo + hi)
    for _ in range(3):
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        if pdf <= 0.0:
            break
        err = (_norm_cdf(z) - p) if p >= 0.5 else ((1.0 - p) - _norm_sf(z))
        z = z - err / pdf
    return float(z)


# --------------------------------------------------------------------------
# 相关系数
# --------------------------------------------------------------------------

def pearson_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """Pearson 线性相关系数及其 t 近似 p 值。

    参数:
        x, y: 等长一维序列。

    返回:
        dict：``coef``（相关系数）、``p_value``（双侧，基于 t_{n-2} 近似）、``n``。

    算法:
        ``r = sum((x-xbar)(y-ybar)) / sqrt(sum(x-xbar)^2 * sum(y-ybar)^2)``；
        检验统计量 ``t = r * sqrt((n-2)/(1-r^2))``，服从自由度 n-2 的 t 分布（原假设 rho=0）。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - **p 值来自 t 近似，前提是 (x, y) 近似二元正态**。存在强离群点、明显非线性
          （如 U 形关系）或重尾分布时，p 值不可信——此时应看 Spearman / Kendall 或画散点图。
        - **n 很小时极不稳定**：n=3 时 |r| 必须大于约 0.997 才能通过 5% 检验，
          而且 r 的抽样分布强烈偏斜。
        - |r| = 1 时 t 的分母为 0，本实现返回 p_value = 0.0（完全线性关系）；
          这时的"显著性"没有实际含义，别拿它当结论。
        - Pearson 只度量**线性**相关：r ≈ 0 不代表独立（抛物线关系也能给出 r ≈ 0）。

    参考:
        Pearson (1895); 检验统计量见 Fisher (1925) 的经典推导。
    """
    xv = as_vector(x, "x")
    yv = as_vector(y, "y")
    n = check_same_length(xv, yv)
    if n < 3:
        raise ValueError(f"pearson_corr 至少需要 3 对观测，得到 {n}")
    xc = xv - xv.mean()
    yc = yv - yv.mean()
    sx = float(np.sqrt(np.sum(xc ** 2)))
    sy = float(np.sqrt(np.sum(yc ** 2)))
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("x 或 y 为常数序列，相关系数无定义")
    r = float(np.sum(xc * yc) / (sx * sy))
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1.0 - r * r))
        p = _t_sf_two_sided(t, float(n - 2))
    return {"coef": r, "p_value": float(p), "n": int(n)}


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """Spearman 秩相关系数及其 t 近似 p 值。

    参数:
        x, y: 等长一维序列。

    返回:
        dict：``coef``（秩相关系数）、``p_value``（双侧，t_{n-2} 近似）、``n``。

    算法:
        对 x、y 分别做平均秩变换，再对秩求 Pearson 相关；
        p 值用 ``t = rho_s * sqrt((n-2)/(1-rho_s^2))`` 的 t_{n-2} 近似。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        - 秩相关度量**单调**关系，对离群点稳健，但对"两边都极端中间平坦"的数据会低估
          关联强度。
        - **并列值（打结）会破坏 t 近似的分布假设**：并列多（例如评分只有 1~5 五档）时，
          近似 p 值偏保守，建议改用置换检验（本模块 ``permutation_test``）或 scipy 的
          exact/permutation 版本。
        - n <= 10 时 t 近似同样不可靠；本实现不做 exact permutation 计算，请在论文里
          说明用的是近似。

    参考:
        Spearman (1904); 近似检验同 Kendall & Stuart, "The Advanced Theory of Statistics", Vol. 2。
    """
    xv = as_vector(x, "x")
    yv = as_vector(y, "y")
    n = check_same_length(xv, yv)
    if n < 3:
        raise ValueError(f"spearman_corr 至少需要 3 对观测，得到 {n}")
    rx = _rank_average(xv)
    ry = _rank_average(yv)
    xc = rx - rx.mean()
    yc = ry - ry.mean()
    sx = float(np.sqrt(np.sum(xc ** 2)))
    sy = float(np.sqrt(np.sum(yc ** 2)))
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("x 或 y 为常数序列，秩相关无定义")
    r = float(np.sum(xc * yc) / (sx * sy))
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1.0 - r * r))
        p = _t_sf_two_sided(t, float(n - 2))
    return {"coef": r, "p_value": float(p), "n": int(n)}


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """Kendall tau-b 秩相关系数及其正态近似 p 值。

    参数:
        x, y: 等长一维序列。

    返回:
        dict：``coef``（tau-b）、``p_value``（双侧，正态近似）、``n``。

    算法:
        逐对比较得到一致对数 C、不一致对数 D、只在 x 打结数 T_x、只在 y 打结数 T_y，
        ``tau_b = (C - D) / sqrt((C + D + T_x) * (C + D + T_y))``；
        p 值用方差 ``var_s``（含打结修正，Kendall 1970）对 ``s = C - D`` 做正态近似。

    复杂度:
        时间 O(n^2) / 空间 O(1)。

    陷阱:
        - 本实现是 **O(n^2) 的双重循环**，n 上万时很慢；大样本请用基于归并排序的
          O(n log n) 算法或成熟库。
        - p 值是**正态近似**，且打结修正只到二阶；n < 30 或打结很多时与精确置换 p 值
          有明显差距（可达 0.01 量级）。正式论文请用 scipy 的 exact/permutation 版本复核。
        - tau-b 与 tau-a、tau-c 不是同一个量：tau-a 不打结修正，tau-c 用于方表。
          论文里必须写明用的是 tau-b。

    参考:
        Kendall, M.G. (1938) "A New Measure of Rank Correlation", Biometrika 30(1): 81-93；
        Kendall, M.G. (1970) "Rank Correlation Methods", 4th ed., Ch. 3（打结方差公式）。
    """
    xv = as_vector(x, "x")
    yv = as_vector(y, "y")
    n = check_same_length(xv, yv)
    if n < 3:
        raise ValueError(f"kendall_tau 至少需要 3 对观测，得到 {n}")
    c = 0
    d = 0
    tx = 0
    ty = 0
    for i in range(n - 1):
        dx = xv[i + 1:] - xv[i]
        dy = yv[i + 1:] - yv[i]
        sx = np.sign(dx)
        sy = np.sign(dy)
        prod = sx * sy
        c += int(np.sum(prod > 0))
        d += int(np.sum(prod < 0))
        tx += int(np.sum((sx == 0) & (sy != 0)))
        ty += int(np.sum((sy == 0) & (sx != 0)))
    denom = math.sqrt(float(c + d + tx) * float(c + d + ty))
    if denom <= 0.0:
        raise ValueError("x 或 y 为常数序列，tau 无定义")
    tau = (c - d) / denom

    # 打结修正的正态近似方差（Kendall 1970，与 scipy 渐近口径一致）
    nt = np.unique(xv, return_counts=True)[1].astype(float)
    nu = np.unique(yv, return_counts=True)[1].astype(float)
    v0 = float(n) * (n - 1) * (2 * n + 5)
    vt = float(np.sum(nt * (nt - 1) * (2 * nt + 5)))
    vu = float(np.sum(nu * (nu - 1) * (2 * nu + 5)))
    if n > 2:
        v1 = float(np.sum(nt * (nt - 1) * (nt - 2))) * float(
            np.sum(nu * (nu - 1) * (nu - 2))
        ) / (9.0 * n * (n - 1) * (n - 2))
    else:
        v1 = 0.0
    v2 = float(np.sum(nt * (nt - 1))) * float(np.sum(nu * (nu - 1))) / (2.0 * n * (n - 1))
    var_s = (v0 - vt - vu) / 18.0 + v1 + v2
    if var_s <= 0.0:
        p = 0.0
    else:
        z = (c - d) / math.sqrt(var_s)
        p = 2.0 * _norm_sf(abs(z))
    return {"coef": float(tau), "p_value": float(p), "n": int(n)}


# --------------------------------------------------------------------------
# 均值检验与列联表
# --------------------------------------------------------------------------

def t_test_one_sample(x: Sequence[float], mu: float = 0.0) -> Dict[str, object]:
    """单样本 t 检验：检验总体均值是否等于 mu。

    参数:
        x: 一维样本。
        mu: 原假设下的均值（默认 0）。

    返回:
        dict：``stat``（t 统计量）、``df``（n-1）、``p_value``（双侧）、
        ``mean_diff``（样本均值 - mu）。

    算法:
        ``t = (xbar - mu) / (s / sqrt(n))``，``s`` 用无偏方差（ddof=1），df = n-1。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - 要求样本近似来自**正态总体**。n 很小且数据明显偏斜时，t 检验的名义水平不准，
          应改用置换检验或 Bootstrap（本模块都提供了）。
        - 样本方差为 0（所有观测相同）时分母为 0；本实现抛 ``ValueError``。
        - p 值大**不等于**"均值等于 mu"，只能说没有足够证据拒绝；反之 p 值小也不代表
          差异有实际意义（大样本下微小差异也会显著），报告里请同时给出 ``mean_diff``
          与置信区间。

    参考:
        Student (Gosset, 1908) "The Probable Error of a Mean", Biometrika 6(1): 1-25。
    """
    xv = as_vector(x, "x")
    n = xv.size
    if n < 2:
        raise ValueError(f"t 检验至少需要 2 个观测，得到 {n}")
    mu_f = float(mu)
    s2 = float(np.var(xv, ddof=1))
    if s2 <= 0.0:
        raise ValueError("样本方差为 0，无法做 t 检验（所有观测相同）")
    diff = float(xv.mean()) - mu_f
    se = math.sqrt(s2 / n)
    t = diff / se
    return {
        "stat": float(t),
        "df": float(n - 1),
        "p_value": float(_t_sf_two_sided(t, float(n - 1))),
        "mean_diff": float(diff),
    }


def t_test_two_sample(
    x: Sequence[float], y: Sequence[float], equal_var: bool = True
) -> Dict[str, object]:
    """两独立样本 t 检验（Student 或 Welch）。

    参数:
        x, y: 两个独立样本，长度可以不同。
        equal_var: True 用合并方差（Student t 检验，自由度 n_x+n_y-2）；
            False 用 Welch 校正（不假设等方差，自由度用 Welch-Satterthwaite 公式）。

    返回:
        dict：``stat``、``df``、``p_value``（双侧）、``mean_diff``（mean(x) - mean(y)）。

    算法:
        Student：``sp^2 = ((nx-1)sx^2 + (ny-1)sy^2)/(nx+ny-2)``，
        ``t = (xbar-ybar)/sqrt(sp^2(1/nx+1/ny))``，df = nx+ny-2。
        Welch：``t = (xbar-ybar)/sqrt(sx^2/nx + sy^2/ny)``，
        ``df = (sx^2/nx + sy^2/ny)^2 / [ (sx^2/nx)^2/(nx-1) + (sy^2/ny)^2/(ny-1) ]``。

    复杂度:
        时间 O(nx+ny) / 空间 O(nx+ny)。

    陷阱:
        - **默认 equal_var=True 是 Student 原版假设**，两组方差差 2 倍以上时（很常见）
          第一类错误率会明显偏离名义水平。拿不准就用 ``equal_var=False``（Welch），
          它在等方差时几乎不损失效率，是更稳的默认。
        - Welch 的自由度是**非整数**，这是正常现象（不是 bug）；报告表格里应保留小数。
        - 两样本必须**独立**；前后测/配对数据用两样本 t 检验是常见错误，应改用配对检验
          （本模块未提供，可对差值用 ``t_test_one_sample``）。
        - 仍要求近似正态；重尾或小样本建议配合 Bootstrap / 置换检验。

    参考:
        Welch, B.L. (1947) "The generalization of 'Student's' problem when several
        different population variances are involved", Biometrika 34(1-2): 28-35。
    """
    xv = as_vector(x, "x")
    yv = as_vector(y, "y")
    nx = xv.size
    ny = yv.size
    if nx < 2 or ny < 2:
        raise ValueError(f"两组样本都至少需要 2 个观测，得到 nx={nx}, ny={ny}")
    mx = float(xv.mean())
    my = float(yv.mean())
    vx = float(np.var(xv, ddof=1))
    vy = float(np.var(yv, ddof=1))
    diff = mx - my
    if equal_var:
        sp2 = ((nx - 1) * vx + (ny - 1) * vy) / float(nx + ny - 2)
        if sp2 <= 0.0:
            raise ValueError("合并方差为 0，无法做 t 检验")
        se = math.sqrt(sp2 * (1.0 / nx + 1.0 / ny))
        df = float(nx + ny - 2)
    else:
        ax = vx / nx
        ay = vy / ny
        se = math.sqrt(ax + ay)
        if se <= 0.0:
            raise ValueError("两组样本方差均为 0，无法做 Welch 检验")
        df = (ax + ay) ** 2 / (ax ** 2 / (nx - 1) + ay ** 2 / (ny - 1))
    if se <= 0.0:
        raise ValueError("标准误为 0，无法做 t 检验")
    t = diff / se
    return {
        "stat": float(t),
        "df": float(df),
        "p_value": float(_t_sf_two_sided(t, df)),
        "mean_diff": float(diff),
    }


def _normal_tail_p(z: float, alternative: str, func: str) -> float:
    """把标准正态统计量转成指定方向的 p 值（秩检验共用）。

    参数:
        z: 标准正态统计量。对秩检验，约定 ``z > 0`` 表示第一组取值倾向更大。
        alternative: ``"two-sided"`` / ``"greater"`` / ``"less"``。
        func: 调用方函数名，仅用于拼报错信息。

    返回:
        p 值（已裁剪到 [0, 1]）：双侧 ``2 * P(Z > |z|)``，greater ``P(Z > z)``，
        less ``P(Z < z)``。

    算法:
        直接调用自实现的 ``_norm_sf`` / ``_norm_cdf``（底层是标准库 math.erfc），
        数字部分不依赖 scipy。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        ``|z| > 8`` 时尾概率已低于双精度能表示的下限，返回的 0.0 应读作 "p < 1e-15"
        而不是精确的 0。

    参考:
        正态近似的秩检验（Mann-Whitney / Wilcoxon）通用尾概率口径。
    """
    if alternative == "two-sided":
        p = 2.0 * _norm_sf(abs(z))
    elif alternative == "greater":
        p = _norm_sf(z)
    elif alternative == "less":
        p = _norm_cdf(z)
    else:
        raise ValueError(
            f"{func} 的 alternative 只能是 'two-sided'/'greater'/'less'，得到 {alternative!r}"
        )
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return float(p)


def mann_whitney_u(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
    continuity: bool = True,
) -> Dict[str, object]:
    """Mann-Whitney U 检验（= Wilcoxon 秩和检验），正态近似 + 并列校正。

    参数:
        x, y: 两个独立样本，长度可以不同（各至少 1 个观测）。
        alternative: 备择方向。``"two-sided"`` 双侧；``"greater"`` 备择为
            "x 的取值倾向大于 y"；``"less"`` 反之。
        continuity: 是否做连续性校正（把 ``u1 - n1*n2/2`` 的偏离向 0 收缩 0.5），
            这是正态近似的常用默认口径。

    返回:
        dict：``u_statistic``（= ``min(u1, u2)``，教科书里的 U 统计量）、
        ``u1``（x 的 U 统计量 ``R1 - n1(n1+1)/2``）、``u2``（= ``n1*n2 - u1``）、
        ``z``（以 u1 为基准的正态统计量，z > 0 表示 x 整体偏大）、
        ``p_value``（按 ``alternative`` 方向的近似 p 值）、
        ``rank_sum``（x 在混合样本中的秩和）。

    算法:
        两组混合后取**平均秩**，``u1 = R1 - n1(n1+1)/2``，``u2 = n1*n2 - u1``；
        正态近似 ``mu = n1*n2/2``，
        ``sigma^2 = (n1*n2/12) * [ (n+1) - sum(t^3 - t) / (n(n-1)) ]``
        （第二项即并列校正，``t`` 为每个并列组的个数，``n = n1+n2``），
        ``z = (u1 - mu +- 0.5) / sigma``（连续性校正各向尾部方向挪半格）。
        双侧 p 值 = ``2 * min(P(U <= u1), P(U >= u1))``，单侧 p 值取对应那一侧的
        连续性校正尾概率；**单侧与双侧的校正方向不同**（见"陷阱"）。
        尾概率用自实现的 ``_norm_sf`` / ``_norm_cdf``。

    复杂度:
        时间 O(n log n)（排序求秩） / 空间 O(n)。

    陷阱:
        - **这是正态近似，不是精确检验**：n1、n2 各小于 8 或并列很多时，近似 p 值
          与精确 p 值能差出几个百分点；正式论文请用 ``scipy.stats.mannwhitneyu`` 的
          ``method="exact"`` 复核。
        - ``u_statistic`` 取 ``min(u1, u2)``（教科书习惯）**丢掉了方向信息**，
          而 ``z`` 以 ``u1`` 为基准；做单侧检验时必须看 ``u1`` / ``z``，不能看
          ``u_statistic``。
        - 并列值必须取平均秩；按出现顺序排秩会让统计量依赖输入顺序（本实现不会）。
        - **连续性校正的单侧口径容易踩坑**：单侧 p 值用的是 ``P(U >= u1) ~ N`` 边界
          各挪半格后的尾概率（与 ``scipy.stats.mannwhitneyu(use_continuity=True)`` 一致），
          而双侧 p 值把偏离向 0 收缩；因此单侧 p 值**并不等于** ``_norm_sf(z)``，
          也不等于 ``双侧 p / 2``。要看单侧显著性请直接用 ``alternative`` 参数，
          不要自己拿 ``z`` 反算。
        - 检验假设两组**独立**；配对/前后测数据请用 ``wilcoxon_signed_rank``。

    参考:
        Mann, H.B. & Whitney, D.R. (1947) "On a Test of Whether one of Two Random
        Variables is Stochastically Larger than the Other", Annals of Mathematical
        Statistics 18(1): 50-60；Conover, "Practical Nonparametric Statistics",
        3rd ed., Ch. 6。
    """
    xv = as_vector(x, "x")
    yv = as_vector(y, "y")
    nx = int(xv.size)
    ny = int(yv.size)
    if nx < 1 or ny < 1:
        raise ValueError(f"两组样本都至少需要 1 个观测，得到 nx={nx}, ny={ny}")
    pooled = np.concatenate([xv, yv])
    ranks = _rank_average(pooled)
    rank_sum = float(np.sum(ranks[:nx]))
    u1 = rank_sum - 0.5 * nx * (nx + 1.0)
    u2 = float(nx) * float(ny) - u1
    n = nx + ny
    mu = 0.5 * float(nx) * float(ny)
    _, counts = np.unique(pooled, return_counts=True)
    tc = counts.astype(float)
    tie = float(np.sum(tc ** 3 - tc))
    var = 0.0
    if n > 1:
        var = (float(nx) * float(ny) / 12.0) * ((n + 1.0) - tie / (float(n) * (n - 1.0)))
    sigma = math.sqrt(var) if var > 0.0 else 0.0
    diff = u1 - mu
    if sigma > 0.0:
        if continuity:
            z_less = (diff + 0.5) / sigma
            z_greater = (diff - 0.5) / sigma
        else:
            z_less = diff / sigma
            z_greater = z_less
        p_less = _norm_cdf(z_less)
        p_greater = _norm_sf(z_greater)
        if not continuity or diff == 0.0:
            z = diff / sigma
        elif diff > 0.0:
            z = (diff - 0.5) / sigma
        else:
            z = (diff + 0.5) / sigma
    else:
        z = 0.0
        p_less = 1.0
        p_greater = 1.0
    if alternative == "two-sided":
        p_value = min(1.0, 2.0 * min(p_less, p_greater))
    elif alternative == "greater":
        p_value = p_greater
    elif alternative == "less":
        p_value = p_less
    else:
        raise ValueError(
            "mann_whitney_u 的 alternative 只能是 'two-sided'/'greater'/'less'，"
            f"得到 {alternative!r}"
        )
    if p_value < 0.0:
        p_value = 0.0
    elif p_value > 1.0:
        p_value = 1.0
    return {
        "u_statistic": float(min(u1, u2)),
        "u1": float(u1),
        "u2": float(u2),
        "z": float(z),
        "p_value": float(p_value),
        "rank_sum": rank_sum,
    }


def wilcoxon_signed_rank(
    x: Sequence[float],
    y: Optional[Sequence[float]] = None,
    alternative: str = "two-sided",
) -> Dict[str, object]:
    """Wilcoxon 符号秩检验（单样本或配对样本），正态近似 + 并列校正。

    参数:
        x: 一维样本。``y=None`` 时做单样本检验（原假设：总体中位数为 0）。
        y: 配对样本，长度必须与 x 相同；给出时检验的是差值 ``x - y``。
        alternative: ``"two-sided"`` / ``"greater"``（差值倾向为正）/ ``"less"``；
            **单侧方向始终以 x 为准**，与是否给 y 无关。

    返回:
        dict：``w_statistic``（= ``min(w_plus, w_minus)``）、``w_plus``（正差值的秩和）、
        ``w_minus``（负差值的秩和）、``z``（以 w_plus 为基准，z > 0 表示正差值占优）、
        ``p_value``、``n_effective``（丢弃零差值后的有效对数）。

    算法:
        令 ``d = x - y``（未给 y 时 ``d = x``），把 ``d = 0`` 的对**整体丢弃**
        （Wilcoxon 原始口径，等价于 scipy 的 ``zero_method="wilcox"``）；
        对 ``|d|`` 取平均秩，分别累加正、负差值的秩和得到 ``w_plus`` / ``w_minus``。
        正态近似 ``mu = n(n+1)/4``，
        ``sigma^2 = [ n(n+1)(2n+1) - sum(t^3 - t)/2 ] / 24``
        （``t`` 为 ``|d|`` 的并列组大小，``n = n_effective``），
        ``z = (w_plus - mu)/sigma``，尾概率用自实现的 ``_norm_sf`` / ``_norm_cdf``。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        - **零差值的处理会改变结果**：本实现直接丢弃，``n_effective`` 会小于输入长度；
          若改用 Pratt 法（保留零差值参与排秩）数值不同，论文里必须写明用的哪种。
        - 这是正态近似而非精确检验：``n_effective < 10`` 时近似很粗糙
          （本模块不提供精确分布表），并列较多时也要谨慎。
        - 单侧方向容易搞反：``alternative="greater"`` 表示"x 倾向大于 y"，
          即 ``w_plus`` 偏大、``z`` 偏正。
        - 交换 x 与 y 后 ``w_plus`` / ``w_minus`` 互换、``z`` 变号，而双侧 p 值完全不变。

    参考:
        Wilcoxon, F. (1945) "Individual Comparisons by Ranking Methods", Biometrics
        Bulletin 1(6): 80-83；Conover, "Practical Nonparametric Statistics",
        3rd ed., Ch. 5。
    """
    xv = as_vector(x, "x")
    if y is None:
        d = xv.copy()
    else:
        yv = as_vector(y, "y")
        check_same_length(xv, yv)
        d = xv - yv
    nz = d[d != 0.0]
    n = int(nz.size)
    if n < 1:
        raise ValueError("所有差值都为 0，Wilcoxon 符号秩检验无定义")
    ad = np.abs(nz)
    r = _rank_average(ad)
    w_plus = float(np.sum(r[nz > 0.0]))
    w_minus = float(np.sum(r[nz < 0.0]))
    mu = 0.25 * float(n) * (n + 1.0)
    _, counts = np.unique(ad, return_counts=True)
    tc = counts.astype(float)
    tie = float(np.sum(tc ** 3 - tc))
    var = (float(n) * (n + 1.0) * (2.0 * n + 1.0) - 0.5 * tie) / 24.0
    sigma = math.sqrt(var) if var > 0.0 else 0.0
    z = (w_plus - mu) / sigma if sigma > 0.0 else 0.0
    return {
        "w_statistic": float(min(w_plus, w_minus)),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": float(z),
        "p_value": _normal_tail_p(z, alternative, "wilcoxon_signed_rank"),
        "n_effective": n,
    }


def kruskal_wallis(groups: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Kruskal-Wallis H 检验（单因素方差分析的非参数版本），带并列校正。

    参数:
        groups: 由各组样本组成的序列，至少 2 组、每组至少 1 个观测，长度可以不同。

    返回:
        dict：``h_statistic``（并列校正后的 H）、``df``（= 组数 - 1）、
        ``p_value``（卡方上尾）、``tie_correction``（并列校正因子
        ``C = 1 - sum(t^3 - t)/(N^3 - N)``，``H = H_原始 / C``）、
        ``rank_sums``（各组在混合样本中的秩和，与输入同序）、``n_groups``、``n_total``。

    算法:
        混合后取平均秩，``H = 12/(N(N+1)) * sum_i R_i^2 / n_i - 3(N+1)``（无并列时）；
        令 ``C = 1 - sum(t^3 - t)/(N^3 - N)``（``t`` 为并列组大小），
        使用 ``H_c = H / C``。p 值取自实现的卡方上尾 ``_chi2_sf(H_c, k-1)``
        （下不完全伽马级数 + 连分式），不依赖 scipy。

    复杂度:
        时间 O(N log N) / 空间 O(N)。

    陷阱:
        - 这是**大样本卡方近似**：每组只有 3~5 个观测时偏差明显，精确分布要查表或用
          置换检验（``permutation_test`` 可以代用）。
        - 所有观测完全相同时 ``C = 0``，此时校正公式本身失去定义：本实现按退化输入处理，
          返回 ``h_statistic = 0``、``p_value = 1.0``。
        - 两组时 H 恰好等于 Mann-Whitney 的 ``z^2``（``continuity=False`` 口径），
          即两组情形下两个检验等价；组数 >= 3 时 H 只回答"是否存在某组不同"，
          **不指出是哪两组**，需要事后两两比较并做多重比较校正。
        - 与 ANOVA 一样假设各组分布形状相同、只允许位置不同。

    参考:
        Kruskal, W.H. & Wallis, W.A. (1952) "Use of Ranks in One-Criterion Variance
        Analysis", Journal of the American Statistical Association 47(260): 583-621。
    """
    if groups is None:
        raise ValueError("groups 不能为 None")
    lst = list(groups)
    k = len(lst)
    if k < 2:
        raise ValueError(f"Kruskal-Wallis 至少需要 2 组，得到 {k} 组")
    arrs = [as_vector(g, f"groups[{i}]") for i, g in enumerate(lst)]
    n_i = [int(a.size) for a in arrs]
    for i, cnt in enumerate(n_i):
        if cnt < 1:
            raise ValueError(f"groups[{i}] 至少需要 1 个观测")
    big_n = int(sum(n_i))
    if big_n < 3:
        raise ValueError(f"总观测数 N={big_n} 太少，至少需要 3")
    pooled = np.concatenate(arrs)
    ranks = _rank_average(pooled)
    rank_sums = []
    off = 0
    for cnt in n_i:
        rank_sums.append(float(np.sum(ranks[off: off + cnt])))
        off += cnt
    h_raw = 12.0 / (big_n * (big_n + 1.0)) * sum(
        rs * rs / float(cnt) for rs, cnt in zip(rank_sums, n_i)
    ) - 3.0 * (big_n + 1.0)
    _, counts = np.unique(pooled, return_counts=True)
    tc = counts.astype(float)
    tie = float(np.sum(tc ** 3 - tc))
    denom = float(big_n ** 3 - big_n)
    c_corr = 1.0 - tie / denom if denom > 0.0 else 0.0
    if c_corr <= 1e-12:
        h_stat = 0.0
    else:
        h_stat = h_raw / c_corr
    if h_stat < 0.0:
        h_stat = 0.0
    df = float(k - 1)
    return {
        "h_statistic": float(h_stat),
        "df": df,
        "p_value": float(_chi2_sf(float(h_stat), df)),
        "tie_correction": float(c_corr),
        "rank_sums": rank_sums,
        "n_groups": int(k),
        "n_total": big_n,
    }


def anova_oneway(groups: Sequence[Sequence[float]]) -> Dict[str, object]:
    """单因素方差分析（one-way ANOVA）：F 检验与平方和分解。

    参数:
        groups: 由各组样本组成的序列，至少 2 组、每组至少 1 个观测，长度可以不同；
            组内自由度 ``N - k`` 必须 >= 1（即不能每组都只有 1 个观测）。

    返回:
        dict：``f_statistic``、``p_value``（F 上尾）、``ss_between`` / ``ss_within`` /
        ``ss_total``（= 两者之和）、``df_between``（= k-1）、``df_within``（= N-k）、
        ``ms_between`` / ``ms_within``（= SS/df）、``grand_mean``、``group_means``
        （列表，与输入同序）。

    算法:
        ``SS_between = sum_i n_i (mean_i - grand_mean)^2``；
        ``SS_within = sum_i sum_j (x_ij - mean_i)^2``（**按组中心化后求和**）；
        ``SS_total = sum (x - grand_mean)^2``；
        ``F = (SS_between/(k-1)) / (SS_within/(N-k))``。
        p 值用自实现的 ``_f_sf``（正则化不完全贝塔函数的 Lentz 连分式），
        与 ``scipy.stats.f.sf`` 的相对误差在 1e-12 量级以内，不依赖 scipy。

    复杂度:
        时间 O(N) / 空间 O(N)。

    陷阱:
        - 计算 ``SS_within`` 时**不要**用 ``SS_total - SS_between`` 去凑：均值很大、
          方差很小时这一步会发生灾难性抵消。本实现按组中心化后直接求和，数值上更稳；
          ``ss_total`` 则独立算出，用作分解恒等式的自检。
        - ANOVA 假设各组**同方差、残差独立且近似正态**；方差不齐时 F 检验的名义水平
          失效，应改用 Welch ANOVA 或 ``kruskal_wallis``。
        - F 显著只说明"至少有一组均值不同"，具体是哪两类要用事后检验
          （Tukey HSD / Bonferroni）并做多重比较校正。
        - 退化情形：组间无差异且组内无变异时 ``ms_within = 0``，返回
          ``f_statistic = 0`` 与 ``p_value = 1``；组内有差异而组间无变异时返回 ``inf``
          与 ``p_value = 0``（"数据被理想化"的信号，不要当真实结论报告）。

    参考:
        Fisher, R.A. (1925) "Statistical Methods for Research Workers"；
        Montgomery, "Design and Analysis of Experiments", 8th ed., Ch. 3。
    """
    if groups is None:
        raise ValueError("groups 不能为 None")
    lst = list(groups)
    k = len(lst)
    if k < 2:
        raise ValueError(f"单因素方差分析至少需要 2 组，得到 {k} 组")
    arrs = [as_vector(g, f"groups[{i}]") for i, g in enumerate(lst)]
    n_i = [int(a.size) for a in arrs]
    for i, cnt in enumerate(n_i):
        if cnt < 1:
            raise ValueError(f"groups[{i}] 至少需要 1 个观测")
    big_n = int(sum(n_i))
    df_between = k - 1
    df_within = big_n - k
    if df_within < 1:
        raise ValueError(
            f"组内自由度 N-k={df_within} 必须 >= 1（N={big_n}, k={k}），每组至少要有 2 个观测"
        )
    means = [float(a.mean()) for a in arrs]
    allv = np.concatenate(arrs)
    grand = float(allv.mean())
    ss_between = float(sum(cnt * (m - grand) ** 2 for cnt, m in zip(n_i, means)))
    ss_within = float(sum(float(np.sum((a - m) ** 2)) for a, m in zip(arrs, means)))
    ss_total = float(np.sum((allv - grand) ** 2))
    ms_between = ss_between / float(df_between)
    ms_within = ss_within / float(df_within)
    if ms_within > 0.0:
        f_stat = ms_between / ms_within
        p_value = float(_f_sf(float(f_stat), float(df_between), float(df_within)))
    elif ms_between > 0.0:
        f_stat = float("inf")
        p_value = 0.0
    else:
        f_stat = 0.0
        p_value = 1.0
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df_between": int(df_between),
        "df_within": int(df_within),
        "ms_between": float(ms_between),
        "ms_within": float(ms_within),
        "grand_mean": grand,
        "group_means": means,
    }


def chi_square_test(observed) -> Dict[str, object]:
    """卡方检验：二维列联表的独立性检验，或一维频数的拟合优度检验。

    参数:
        observed: 二维频数表（r x c）或一维频数向量（长度 k）。

    返回:
        dict：``stat``、``df``、``p_value``（右尾）、``expected``（期望频数，与输入同形状）。
        二维时 ``df = (r-1)(c-1)``，期望频数 ``E_ij = 行和 * 列和 / 总和``；
        一维时 ``df = k-1``，期望频数为等概率假设下的 ``总和 / k``。

    算法:
        ``chi2 = sum((O - E)^2 / E)``，p 值用卡方上尾概率（正则化上不完全 Gamma）。

    复杂度:
        时间 O(rc) / 空间 O(rc)。

    陷阱:
        - 卡方近似要求**期望频数不能太小**：通常的经验规则是"所有 E >= 1，且 E < 5 的格子
          不超过 20%"。E 太小（尤其 2x2 表）时 p 值严重偏小，应改用 Fisher 精确检验
          （本模块未提供，可用 scipy.stats.fisher_exact）。
        - 一维版本**只做等概率原假设**（E 全相等）。如果你的原假设是特定比例（如 9:3:3:1），
          本函数不支持，必须自己传入期望频数——不要用它去检验非均匀分布。
        - 输入必须是**频数**（计数），不是比例或百分比；传比例会算出看似合理但完全错误的
          统计量。含小数频数（例如加权数据）时卡方近似同样不可靠。
        - 期望频数为 0 的格子（整行或整列为 0）会让 ``(O-E)^2/E`` 变成 0/0，本实现抛
          ``ValueError``，请先删掉全零的行列。

    参考:
        Pearson, K. (1900) "On the criterion that a given system of deviations...",
        Philosophical Magazine 50(302): 157-175。
    """
    if observed is None:
        raise ValueError("observed 不能为 None")
    arr = np.asarray(observed, dtype=float)
    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("一维频数至少需要 2 个类别")
        if np.any(arr < 0):
            raise ValueError("频数不能为负")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("频数总和必须为正")
        expected = np.full(arr.size, total / arr.size, dtype=float)
        df = float(arr.size - 1)
    elif arr.ndim == 2:
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            raise ValueError(f"二维列联表至少 2x2，得到 {arr.shape}")
        if np.any(arr < 0):
            raise ValueError("频数不能为负")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("频数总和必须为正")
        rs = arr.sum(axis=1, keepdims=True)
        cs = arr.sum(axis=0, keepdims=True)
        expected = rs * cs / total
        df = float((arr.shape[0] - 1) * (arr.shape[1] - 1))
    else:
        raise ValueError(f"observed 必须是一维或二维，得到 ndim={arr.ndim}")
    if np.any(expected <= 0.0):
        raise ValueError("存在期望频数为 0 的格子：请先删除全零的行/列")
    stat = float(np.sum((arr - expected) ** 2 / expected))
    return {
        "stat": stat,
        "df": df,
        "p_value": float(_chi2_sf(stat, df)),
        "expected": expected,
    }


# --------------------------------------------------------------------------
# 正态性检验
# --------------------------------------------------------------------------

def shapiro_wilk(x: Sequence[float]) -> Dict[str, object]:
    """Shapiro-Wilk 正态性检验——**本模块不实现，调用即抛出 NotImplementedError**。

    参数:
        x: 一维样本（本实现不会用到）。

    返回:
        不返回；总是抛 ``NotImplementedError``。

    算法:
        无。Shapiro-Wilk 的 W 统计量需要"正态次序统计量的期望值"权重 ``a_i``，其精确值由
        Royston (1995) 的专有系数表给出（AS R94 算法），无法用简单闭式复现。

    复杂度:
        无。

    陷阱:
        - 为什么故意不实现：W 的精确 p 值依赖 Royston (1995) 的多项式系数表（按 n 分段，
          每段一组 6~7 个系数）与正态化变换。本仓库无法在不引入 scipy/statsmodels 的前提下
          逐项核对那张表；凭记忆写出来极可能得到"看起来能跑、数值偏差几个百分点"的结果，
          而这种错误在竞赛论文里比缺少一个检验危险得多。
        - **常见错误做法**：把 ``W = (sum a_i x_(i))^2 / sum (x_i - xbar)^2`` 里的 ``a_i``
          换成 Blom 分数 ``m_i / ||m||``（那是 Shapiro-**Francia** 的权重），却仍把结果
          标成 "Shapiro-Wilk 的 W"。两者数值接近但**不是同一个统计量**，对应的临界值也不同，
          属于张冠李戴。
        - 需要的替代方案：
          1. 直接调用 ``scipy.stats.shapiro``（正式论文请用这一个，并注明版本）；
          2. 用本模块的 ``anderson_darling``（A^2，闭式统计量 + 已验证的近似 p 值）；
          3. 用本模块的 ``jarque_bera``（基于偏度/峰度，大样本有效）；
          4. 用本模块的 ``ks_test_normal``（注意参数估计后 p 值偏大的陷阱）。

    参考:
        Shapiro, S.S. & Wilk, M.B. (1965) "An Analysis of Variance Test for Normality",
        Biometrika 52(3/4): 591-611；Royston, J.P. (1995) "Remark AS R94: A Remark on
        Algorithm AS 181", Applied Statistics 44(4): 547-551。
    """
    as_vector(x, "x")
    raise NotImplementedError(
        "shapiro_wilk 未实现：Shapiro-Wilk 的精确系数表（Royston 1995 / AS R94）无法在不引入 "
        "scipy 的前提下可靠复现，本仓库宁可不提供也不返回编造的 W 与 p 值。"
        "请改用 scipy.stats.shapiro，或使用本模块的 anderson_darling / jarque_bera / ks_test_normal。"
    )


def jarque_bera(x: Sequence[float]) -> Dict[str, object]:
    """Jarque-Bera 正态性检验（基于偏度与峰度，闭式公式）。

    参数:
        x: 一维样本（建议 n >= 30）。

    返回:
        dict：``stat``（JB 统计量）、``p_value``（右尾，chi2_2）、``df``（恒为 2）、
        ``skewness``、``kurtosis``（**未减 3 的原始峰度**）、``n``。

    算法:
        令 ``m_k = (1/n) sum (x_i - xbar)^k``，
        ``S = m_3 / m_2^{3/2}``，``K = m_4 / m_2^2``，
        ``JB = n/6 * (S^2 + (K-3)^2/4)``，原假设下渐近服从 ``chi2_2``，
        而 ``chi2_2`` 的上尾概率恰为 ``exp(-JB/2)``。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - **只是大样本（渐近）检验**：n < 30 时 JB 的实际水平远高于名义水平（过度拒绝），
          小样本请用 Anderson-Darling 或查表法。
        - 偏度/峰度用的是**有偏（矩）估计**（除以 n），与某些软件的无偏口径略有差异；
          大样本下差异可忽略，小样本要注明口径。
        - JB 是**综合**检验：偏度和峰度两项任一超标都会拒绝。拒绝时务必分别看 ``skewness``
          与 ``kurtosis``，判断到底是偏斜还是重尾，否则论文里说不清"哪里不正态"。
        - 对单个离群点极其敏感：n=100 的样本里放 1 个 10 sigma 的点就能让 JB 爆掉。
          JB 拒绝时先画 QQ 图，别急着断言总体不正态。

    参考:
        Jarque, C.M. & Bera, A.K. (1980) "Efficient tests for normality, homoscedasticity and
        serial independence of regression residuals", Economics Letters 6(3): 255-259。
    """
    xv = as_vector(x, "x")
    n = xv.size
    if n < 4:
        raise ValueError(f"jarque_bera 至少需要 4 个观测，得到 {n}")
    xc = xv - xv.mean()
    m2 = float(np.mean(xc ** 2))
    if m2 <= 0.0:
        raise ValueError("样本方差为 0，JB 检验无定义")
    m3 = float(np.mean(xc ** 3))
    m4 = float(np.mean(xc ** 4))
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    jb = n / 6.0 * (skew ** 2 + (kurt - 3.0) ** 2 / 4.0)
    return {
        "stat": float(jb),
        "p_value": float(math.exp(-0.5 * jb)),
        "df": 2.0,
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "n": int(n),
    }


def anderson_darling(x: Sequence[float]) -> Dict[str, object]:
    """Anderson-Darling 正态性检验：经验分布与正态 CDF 的加权最大偏差。

    参数:
        x: 一维样本（建议 n >= 8）。

    返回:
        dict：``stat``（A^2 统计量，越大越不正态）、``p_value``（右尾近似）、
        ``crit``（``{"15%", "10%", "5%", "2.5%", "1%"}`` 对应的渐近临界值）、``n``。

    算法:
        用样本均值与**无偏标准差（ddof=1）**标准化得到 ``z_(i)``（升序），
        ``A^2 = -n - (1/n) * sum_{i=1}^{n} (2i-1) [ ln F(z_(i)) + ln(1 - F(z_(n+1-i))) ]``；
        p 值用 D'Agostino & Stephens (1986) 的分段近似，小样本修正为
        ``A2* = A^2 (1 + 0.75/n + 2.25/n^2)``。

    复杂度:
        时间 O(n log n)（排序主导）/ 空间 O(n)。

    陷阱:
        - 这里的 A^2 是**参数由样本估计**的版本（"case 4"），其零分布与"均值和方差已知"
          的版本不同，**不能**套用后者（``P(A^2 < x)`` 有闭式的那个）的临界值。
        - 临界值表是渐近的：n < 30 时实际临界值略低（例如 5% 临界值在 n=20 时约 0.72，
          表中为 0.752），用表值判断会**偏保守**（更不容易拒绝）。
        - p 值分段近似在 ``A^2 ≈ 0.30`` 附近（分段边界）绝对误差可达约 0.03，其余区间
          一般不超过 0.01；需要精确 p 值请用 ``scipy.stats.monte_carlo_test`` 自举。
        - 标准化用 ddof=1：换成 ddof=0 会得到不同的 A^2，与常见软件（如
          ``scipy.stats.anderson``）对不上号。
        - 样本里如果有极端离群值，``F(z)`` 会舍入到 0 或 1 导致对数发散；本实现把概率
          截断在 ``[1e-300, 1-1e-16]`` 内，避免返回 -inf。

    参考:
        Anderson, T.W. & Darling, D.A. (1954) "A Test of Goodness of Fit", JASA 49(268): 765-769；
        Stephens, M.A. (1974) "EDF Statistics for Goodness of Fit and Some Comparisons",
        JASA 69(347): 730-737；D'Agostino, R.B. & Stephens, M.A. (1986) "Goodness-of-Fit
        Techniques", Marcel Dekker, Table 4.9（p 值分段近似）。
    """
    xv = as_vector(x, "x")
    n = xv.size
    if n < 8:
        raise ValueError(f"anderson_darling 的近似要求 n >= 8，得到 {n}")
    s = float(np.std(xv, ddof=1))
    if s <= 0.0:
        raise ValueError("样本标准差为 0，AD 检验无定义")
    z = np.sort((xv - xv.mean()) / s)
    # 正态 CDF / 上尾的稳定计算：直接算 F 与 1-F，避免 log(0)
    f = np.array([_norm_cdf(float(v)) for v in z], dtype=float)
    sf = np.array([_norm_sf(float(v)) for v in z], dtype=float)
    f = np.clip(f, 1e-300, 1.0)
    sf = np.clip(sf, 1e-300, 1.0)
    i = np.arange(1, n + 1, dtype=float)
    a2 = -float(n) - float(np.sum((2.0 * i - 1.0) * (np.log(f) + np.log(sf[::-1]))) / n)

    a2s = a2 * (1.0 + 0.75 / n + 2.25 / (n * n))
    if a2s > 0.6:
        p = math.exp(1.2937 - 5.709 * a2s + 0.0186 * a2s * a2s)
    elif a2s > 0.34:
        p = math.exp(0.9177 - 4.279 * a2s - 1.38 * a2s * a2s)
    elif a2s > 0.2:
        p = 1.0 - math.exp(-8.318 + 42.796 * a2s - 59.938 * a2s * a2s)
    else:
        p = 1.0 - math.exp(-13.436 + 101.14 * a2s - 223.73 * a2s * a2s)
    p = min(1.0, max(0.0, float(p)))
    crit = {"15%": 0.561, "10%": 0.631, "5%": 0.752, "2.5%": 0.873, "1%": 1.035}
    return {"stat": float(a2), "p_value": p, "crit": crit, "n": int(n)}


def ks_test_normal(
    x: Sequence[float], mu: Optional[float] = None, sigma: Optional[float] = None
) -> Dict[str, object]:
    """单样本 Kolmogorov-Smirnov 正态性检验（可指定或由样本估计 mu/sigma）。

    参数:
        x: 一维样本。
        mu: 原假设下的均值；``None`` 表示用样本均值估计。
        sigma: 原假设下的标准差；``None`` 表示用样本无偏标准差（ddof=1）估计。
            两者必须同时给或同时为 ``None``。

    返回:
        dict：``stat``（KS 统计量 D）、``p_value``（渐近右尾，**未做 Lilliefors 校正**）、
        ``n``、``mu``、``sigma``（实际使用的值）、``parameters_estimated``（bool）。

    算法:
        ``D = max_i max( i/n - F(z_(i)), F(z_(i)) - (i-1)/n )``，
        ``z_(i) = (x_(i) - mu)/sigma``；
        p 值用 Kolmogorov 渐近分布（Stephens 修正）：
        ``lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D``，
        ``Q(lambda) = 2 * sum_{k=1..} (-1)^{k-1} exp(-2 k^2 lambda^2)``。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        - **用样本估计 mu/sigma 后，KS 的 p 值偏大（明显偏乐观）**：标准 KS 的零分布假设
          分布参数完全已知，参数被估计后 D 的分布整体变小，但临界值没变，于是 p 值虚高、
          真实第一类错误率远低于名义水平（保守）。这就是所谓的 **Lilliefors 问题**。
          正确做法是用 Lilliefors 修正表或蒙特卡洛 p 值（本模块未实现修正，只做说明）。
        - 因此本函数在 ``parameters_estimated=True`` 时返回的 p 值是"名义值"，
          **不能**直接写进论文当正态性检验结论；请改用 ``anderson_darling`` 或 scipy 的
          Lilliefors 实现。
        - KS 对**分布中部**的差异最敏感，对尾部差异不敏感；Anderson-Darling 恰好相反（尾部加权），
          两个检验一起看才能说得清数据哪里偏离正态。
        - sigma 必须为正；样本常数时抛 ``ValueError``。

    参考:
        Kolmogorov (1933); Smirnov (1948); Stephens, M.A. (1970) "Use of the Kolmogorov-Smirnov,
        Cramer-Von Mises and Related Statistics Without Extensive Tables", JRSS-B 32(1): 115-122；
        Lilliefors, H.W. (1967) "On the KS Test for Normality with Mean and Variance Unknown",
        JASA 62(318): 399-402。
    """
    xv = as_vector(x, "x")
    n = xv.size
    if n < 3:
        raise ValueError(f"ks_test_normal 至少需要 3 个观测，得到 {n}")
    if (mu is None) != (sigma is None):
        raise ValueError("mu 与 sigma 必须同时给定或同时为 None")
    if mu is None:
        m = float(xv.mean())
        s = float(np.std(xv, ddof=1))
        estimated = True
    else:
        m = float(mu)
        s = float(sigma)
        estimated = False
    if s <= 0.0:
        raise ValueError("sigma 必须为正")
    z = np.sort((xv - m) / s)
    f = np.array([_norm_cdf(float(v)) for v in z], dtype=float)
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - f))
    d_minus = float(np.max(f - (i - 1.0) / n))
    d = max(d_plus, d_minus)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam <= 0.0:
        p = 1.0
    else:
        total = 0.0
        for k in range(1, 101):
            term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
            total += term
            if abs(term) < 1e-14:
                break
        p = min(1.0, max(0.0, total))
    return {
        "stat": float(d),
        "p_value": float(p),
        "n": int(n),
        "mu": float(m),
        "sigma": float(s),
        "parameters_estimated": bool(estimated),
    }


# --------------------------------------------------------------------------
# 回归
# --------------------------------------------------------------------------

def ols(X, y: Sequence[float], add_intercept: bool = True) -> Dict[str, object]:
    """普通最小二乘回归，返回完整的推断量（系数、标准误、t、p、R^2、F）。

    参数:
        X: 设计矩阵，形状 (n, p)。一维输入会被当成单变量列（建议显式写成 (n,1)）。
        y: 因变量，长度 n。
        add_intercept: True 时自动在最前面加一列 1（``coef[0]`` 即截距）。

    返回:
        dict：
        - ``coef``：系数（含截距，若 ``add_intercept=True``）；
        - ``se``：系数标准误，``sqrt(sigma2 * diag((X'X)^{-1}))``；
        - ``t`` / ``p_value``：各系数的 t 统计量与双侧 p 值（自由度 ``df_resid``）；
        - ``r2`` / ``adj_r2``：决定系数与调整决定系数；
        - ``f_stat`` / ``f_p_value``：整体显著性 F 检验；
        - ``resid``：残差（长度 n）；
        - ``sigma2``：残差方差 ``RSS / df_resid``；
        - ``df_resid``：残差自由度 ``n - k``。

    算法:
        用 ``np.linalg.lstsq``（SVD）求系数，避免直接求逆放大误差；
        再算 ``RSS``、``sigma2``、``(X'X)^{-1}`` 对角线得到标准误。
        ``R^2 = 1 - RSS/TSS``（TSS 为 ``sum(y - ybar)^2``，含截距时）；
        ``adj_R^2 = 1 - (1-R^2)(n-1)/(n-k)``；
        ``F = ((TSS - RSS)/(k-1)) / (RSS/(n-k))``。

    复杂度:
        时间 O(n p^2) / 空间 O(np)。

    陷阱:
        - **多重共线性 / 病态矩阵是本函数最大的坑**：两列近似成比例时 ``X'X`` 接近奇异，
          系数会剧烈摆动、标准误爆炸、符号甚至反转（这就是 VIF 要检查的东西，
          见 ``vif``）。本实现会在矩阵**秩亏**时抛 ``ValueError``，但"接近秩亏"
          （相关系数 0.999）检测不出来，只能靠 VIF 与条件数人工判断。
        - ``add_intercept=False`` 时的 ``R^2`` 是**未中心化**口径（``1 - RSS/sum(y^2)``），
          与统计软件默认的中心化 R^2 不可比；F 检验的自由度也换成 k 而不是 k-1。
          除非有明确的过原点建模理由，否则请保留截距。
        - p 值与 F 检验都假设：残差同方差、独立、近似正态。异方差或自相关存在时，
          t 检验的名义水平失效（时间序列数据几乎必然自相关，应改用稳健标准误或
          Newey-West，本模块未实现）。
        - 用同一个数据集反复挑变量再报告 p 值（逐步回归后直接看 p）会严重高估显著性；
          需要交叉验证或信息准则选变量。

    参考:
        Gauss-Markov 定理；Draper & Smith, "Applied Regression Analysis", 3rd ed., Ch. 2-3。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n = Xm.shape[0]
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    if add_intercept:
        Xd = np.column_stack([np.ones(n, dtype=float), Xm])
    else:
        Xd = Xm
    k = Xd.shape[1]
    if n <= k:
        raise ValueError(f"观测数 n={n} 必须大于参数个数 k={k}")
    coef, _, rank, _ = np.linalg.lstsq(Xd, yv, rcond=None)
    if rank < k:
        raise ValueError(
            f"设计矩阵秩亏（rank={rank} < 列数 {k}）：存在完全共线的列，请先删除冗余变量"
        )
    resid = yv - Xd @ coef
    rss = float(resid @ resid)
    df_resid = n - k
    sigma2 = rss / df_resid
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0.0, coef / np.where(se > 0.0, se, 1.0), np.nan)
    pvals = np.array(
        [_t_sf_two_sided(float(v), float(df_resid)) if np.isfinite(v) else float("nan")
         for v in tvals],
        dtype=float,
    )
    m = float(yv.mean())
    if add_intercept:
        tss = float(np.sum((yv - m) ** 2))
        r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
        df1 = k - 1
        if df1 > 0 and rss > 0.0:
            f_stat = ((tss - rss) / df1) / (rss / df_resid)
        else:
            f_stat = 0.0
    else:
        tss = float(np.sum(yv ** 2))
        r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
        adj_r2 = 1.0 - (1.0 - r2) * n / df_resid
        df1 = k
        f_stat = (tss - rss) / df1 / (rss / df_resid) if rss > 0.0 else 0.0
    f_p = _f_sf(float(f_stat), float(df1), float(df_resid)) if df1 > 0 else 1.0
    return {
        "coef": coef,
        "se": se,
        "t": tvals,
        "p_value": pvals,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "f_stat": float(f_stat),
        "f_p_value": float(f_p),
        "resid": resid,
        "sigma2": float(sigma2),
        "df_resid": int(df_resid),
    }


def newey_west_se(
    y: Sequence[float],
    X,
    lags: Optional[int] = None,
) -> Dict[str, object]:
    """OLS + Newey-West（Bartlett 核）异方差自相关稳健（HAC）标准误。

    参数:
        y: 因变量，长度 n。
        X: 设计矩阵 (n, p)，**不含截距列**（截距由本函数自动加在最前面，
           对应 ``beta[0]``）。**必须显式传二维数组**：按 ``as_matrix`` 的口径，
           一维输入会被当成"1 行的矩阵"，而不是单变量列，请写成 ``(n, 1)``。
        lags: 截断阶数 ``L``（>= 0）。``None`` 时用经验法则
            ``L = floor(4 * (n/100)^(2/9))``（Newey-West 1994 的建议，n=100 时 L=4，
            n=400 时 L=5）。

    返回:
        dict：``beta``（长度 p+1，含截距）、``se``（Newey-West 稳健标准误）、
        ``t_stat``（= beta/se）、``p_value``（**正态近似**双侧）、
        ``lags``（实际使用的 L）、``r_squared``（含截距的中心化 R^2）。

    算法:
        先做 OLS 取残差 ``e``，再构造 HAC "meat" 矩阵
        ``M = sum_t x_t x_t' e_t^2
        + sum_{l=1..L} (1 - l/(L+1)) * sum_t (x_t e_t e_{t-l} x_{t-l}' + 其转置)``
        （权重 ``w_l = 1 - l/(L+1)`` 即 Bartlett 核），
        ``Var(beta) = n/(n-k) * (X'X)^{-1} M (X'X)^{-1}``（k = p+1）。
        其中 ``n/(n-k)`` 是自由度修正（与 R ``sandwich::NeweyWest(adjust=TRUE)`` 同口径）：
        正因为有它，``lags=0`` 且残差恰好同方差时结果**严格退回经典 OLS 标准误**。
        p 值用自实现的 ``_norm_sf`` 做正态近似。

    复杂度:
        时间 O(L * n * p^2) / 空间 O(n p)。

    陷阱:
        - **p 值用的是正态近似而不是 t 分布**：n 较小时 p 值偏小，这里返回的是渐近结果；
          小样本又确信残差同方差、无自相关时，请用 ``ols`` 的 t 检验。
        - ``L`` 是偏差-方差权衡：太小压不住自相关（标准误偏小、t 值虚高），太大则估计
          噪音大甚至破坏正定性。经验法则只在几百个观测的量级上可靠；本函数**不做**
          最优带宽的自动估计（Newey-West 1994 的迭代选取）。
        - 稳健标准误只改标准误、**不改系数**：``beta`` 仍是 OLS 的，它既不修正遗漏变量
          偏误，也不提高效率。
        - 带 ``n/(n-k)`` 修正后，``lags=0`` 相当于带自由度修正的 White(1980) 异方差稳健
          标准误（HC0 的修正版），**一般不等于**经典标准误——只有残差同方差时才相等
          （``_self_test`` 用等模残差构造了这种算例来对齐 ``ols``）。
        - ``X`` 里不要重复放截距列；时间序列做 HAC 时残差必须与回归对应。
        - 单变量回归若把 ``X`` 传成一维数组，``as_matrix`` 会当成 1 行而报
          "X 有 1 行但 y 长度 n"；请显式写 ``X.reshape(-1, 1)``。

    参考:
        Newey, W.K. & West, K.D. (1987) "A Simple, Positive Semi-Definite,
        Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
        Econometrica 55(3): 703-708；Newey & West (1994), Review of Economic
        Studies 61(4): 631-653（带宽法则）。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n = int(Xm.shape[0])
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    Xd = np.column_stack([np.ones(n, dtype=float), Xm])
    k = int(Xd.shape[1])
    if n <= k:
        raise ValueError(f"观测数 n={n} 必须大于参数个数 k={k}")
    coef, _, rank, _ = np.linalg.lstsq(Xd, yv, rcond=None)
    if rank < k:
        raise ValueError(
            f"设计矩阵秩亏（rank={rank} < 列数 {k}）：存在完全共线的列，请先删除冗余变量"
        )
    resid = yv - Xd @ coef
    if lags is None:
        lag_use = int(math.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0)))
    else:
        if int(lags) != lags:
            raise ValueError(f"lags 必须是整数，得到 {lags!r}")
        lag_use = int(lags)
        if lag_use < 0:
            raise ValueError(f"lags 必须 >= 0，得到 {lags!r}")
    if lag_use > n - 1:
        raise ValueError(f"lags={lag_use} 超过可用上限 n-1={n - 1}")
    xe = Xd * resid[:, None]
    meat = xe.T @ xe
    for lag in range(1, lag_use + 1):
        w = 1.0 - float(lag) / float(lag_use + 1)
        cross = xe[lag:].T @ xe[:-lag]
        meat = meat + w * (cross + cross.T)
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    df_resid = n - k
    cov = (float(n) / float(df_resid)) * (xtx_inv @ meat @ xtx_inv)
    cov = 0.5 * (cov + cov.T)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = np.where(se > 0.0, coef / np.where(se > 0.0, se, 1.0), np.nan)
    pvals = np.array(
        [2.0 * _norm_sf(abs(float(v))) if np.isfinite(v) else float("nan") for v in tvals],
        dtype=float,
    )
    rss = float(resid @ resid)
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else 0.0
    return {
        "beta": coef,
        "se": se,
        "t_stat": tvals,
        "p_value": pvals,
        "lags": lag_use,
        "r_squared": float(r2),
    }


def vif(X) -> Dict[str, object]:
    """方差膨胀因子（VIF）：逐列对其余列回归得到的共线性指标。

    参数:
        X: 设计矩阵，形状 (n, p)，**不含截距列**（本函数在每次辅助回归里自动加截距）。

    返回:
        dict：``vif``（长度 p 的数组，``VIF_j = 1 / (1 - R2_j)``）、
        ``r2``（每次辅助回归的 R^2）。

    算法:
        对第 j 列做辅助 OLS（其余列 + 截距为自变量），取 ``R2_j``；
        ``VIF_j = 1/(1-R2_j)``。``R2_j`` 接近 1 时 VIF 趋于无穷。

    复杂度:
        时间 O(p * n * p^2) / 空间 O(np)。

    陷阱:
        - **VIF 只诊断"线性"共线**，对非线性依赖（如 X2 = X1^2）常常看不出来，
          但那种情况同样会让系数不稳。
        - 经验阈值（如 VIF > 10 提示严重共线）只是惯例，不是定理：变量少时 VIF=5 也可能
          已经让标准误翻倍，变量多时阈值可以放宽。请在论文里写明用的阈值与理由。
        - 这里做的是"对**其余所有列**回归"，因此当设计矩阵里有**近似重复的哑变量**
          （虚拟变量陷阱：k 个类别放 k 个哑变量 + 截距）时 VIF 直接爆表——那是建模错误，
          不是数据处理问题。
        - 完全共线时 ``1-R2`` 为 0，本实现返回 ``inf`` 并在 ``r2`` 里保留 1.0，
          由调用者决定是否删列；``inf`` 不要直接放进论文表格，应写成 "完全共线"。

    参考:
        Marquardt, D.W. (1970) "Generalized Inverses, Ridge Regression, Biased Linear
        Estimation, and Nonlinear Estimation", Technometrics 12(3): 591-612；
        Belsley, Kuh & Welsch, "Regression Diagnostics", 1980。
    """
    Xm = as_matrix(X, "X")
    n, p = Xm.shape
    if p < 2:
        raise ValueError("vif 至少需要 2 列自变量")
    if n <= p:
        raise ValueError(f"观测数 n={n} 必须大于自变量个数 p={p}")
    vifs = np.empty(p, dtype=float)
    r2s = np.empty(p, dtype=float)
    for j in range(p):
        others = np.delete(Xm, j, axis=1)
        res = ols(others, Xm[:, j], add_intercept=True)
        r2 = float(res["r2"])
        r2s[j] = r2
        if r2 >= 1.0 - 1e-12:
            vifs[j] = float("inf")
        else:
            vifs[j] = 1.0 / (1.0 - r2)
    return {"vif": vifs, "r2": r2s}


def ridge_regression(
    X, y: Sequence[float], alpha: float, standardize: bool = True
) -> Dict[str, object]:
    """岭回归（L2 正则最小二乘）的闭式解。

    参数:
        X: 设计矩阵，形状 (n, p)，不含截距列。
        y: 因变量，长度 n。
        alpha: 正则化强度（>= 0）。0 时退化为 OLS。
        standardize: True 时先对每列做中心化 + 除以标准差（ddof=0），
            在标准化尺度上施加惩罚，再把系数换算回原始尺度。

    返回:
        dict：``coef``（原始尺度系数，长度 p）、``intercept``（截距）、
        ``alpha``、``standardized``（是否标准化）。

    算法:
        标准化后解 ``beta = (Z'Z + alpha I)^{-1} Z'y_c``（``y_c`` 为中心化后的 y，
        截距不参与惩罚），再换算：``coef_j = beta_j / s_j``，
        ``intercept = ybar - sum_j coef_j * xbar_j``。
        ``standardize=False`` 时直接解 ``(X'X + alpha I)^{-1}X'y``，截距用
        ``mean(y) - mean(X) @ coef`` 事后计算（此时惩罚会"污染"截距，务必看陷阱）。

    复杂度:
        时间 O(n p^2 + p^3) / 空间 O(np)。

    陷阱:
        - **不标准化就做岭回归，等于按变量的量纲施加惩罚**：把"万元"改成"元"（数值放大
          10000 倍）会让该变量的系数缩小 10000 倍，同样的 alpha 下惩罚几乎失效。
          除非所有自变量量纲一致，否则请保持 ``standardize=True``。
        - ``alpha`` 不能靠"试到结果好看"来定：应当用交叉验证（时间序列要用
          ``forecasting.rolling_origin_cv`` 的滚进口径）或岭迹图选，并在论文中写明。
        - 岭回归**没有 p 值**：系数是有偏估计，常规 t 检验不适用。要报告不确定性请用
          Bootstrap。本函数刻意不返回标准误与 p 值。
        - 常数自变量列（标准差为 0）会破坏标准化，本实现把其尺度置为 1 并在结果里保留；
          这种列应当提前删除。
        - 系数随 alpha 连续变化，比较不同 alpha 的系数时必须用同一套标准化规则。

    参考:
        Hoerl, A.E. & Kennard, R.W. (1970) "Ridge Regression: Biased Estimation for
        Nonorthogonal Problems", Technometrics 12(1): 55-67。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n, p = Xm.shape
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise ValueError(f"alpha 必须是 >= 0 的有限数，得到 {alpha!r}")
    if standardize:
        mu = Xm.mean(axis=0)
        sd = Xm.std(axis=0, ddof=0)
        sd_safe = np.where(sd > 0.0, sd, 1.0)
        Z = (Xm - mu) / sd_safe
        yc = yv - yv.mean()
        beta = np.linalg.solve(Z.T @ Z + a * np.eye(p), Z.T @ yc)
        coef = beta / sd_safe
        intercept = float(yv.mean() - float(mu @ coef))
    else:
        beta = np.linalg.solve(Xm.T @ Xm + a * np.eye(p), Xm.T @ yv)
        coef = beta
        intercept = float(yv.mean() - float(Xm.mean(axis=0) @ coef))
    return {
        "coef": coef,
        "intercept": intercept,
        "alpha": a,
        "standardized": bool(standardize),
    }


def logistic_regression(
    X,
    y: Sequence[float],
    lr: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """二分类 logistic 回归（带回溯线搜索的牛顿-拉夫森 / IRLS）。

    参数:
        X: 设计矩阵，形状 (n, p)，**不含截距列**（本函数自动在首列加 1）。
        y: 标签，取值必须为 0/1（允许 0.0/1.0）。
        lr: 回溯线搜索的步长**收缩因子**，必须落在 (0, 1)。牛顿全步长若使对数似然下降，
            就把步长乘以 ``lr`` 重试（最多 60 次）。
        max_iter: 最大迭代次数。
        tol: 收敛判据：参数最大变化量小于 tol 即认为收敛。

    返回:
        dict：
        - ``coef``：长度 p+1，``coef[0]`` 是截距；
        - ``log_lik``：最终对数似然（越接近 0 越"完美"）；
        - ``iterations``：实际迭代次数；
        - ``converged``：是否收敛。**完全分离时返回 False**；
        - ``prob``：最终预测概率（长度 n，全部为有限值）。

    算法:
        牛顿步 ``step = H^{-1} g``，其中 ``g = X'(y - p)``、``H = X'WX``、``W = p(1-p)``；
        为避免完全分离导致 H 奇异，H 上加 1e-10 的岭；若全步长使对数似然下降则按 ``lr``
        回溯；参数最大变化量小于 tol 时收敛。

    复杂度:
        时间 O(max_iter * n p^2) / 空间 O(np)。

    陷阱:
        - **完全分离（complete separation）**：某一组自变量能完美区分 0/1 时，系数的最小二乘/
          极大似然解不存在，系数会发散到无穷、对数似然趋近 0、标准误爆炸。本实现的做法是
          在每一轮 IRLS 后检查两个截断条件——**任一系数的绝对值超过 1e6**（判据是
          ``max|beta_j| > 1e6`` 这种"最大绝对系数"，不是二范数 ``||beta||_2``）**或**对数似然
          变为非有限值时，立即停止并返回 ``converged=False``，
          **不返回 NaN，也不假装收敛**。遇到这种情况应当：加 L2 惩罚（等价于 ridge logistic）、
          减少变量、或改用 Firth 惩罚似然。
        - 返回的 ``converged=False`` 时系数**不可解释**，不要写进论文；先解决分离问题。
        - 本函数**不返回标准误与 p 值**：Wald 统计量在分离边缘极不可靠，要推断请用
          Bootstrap（``bootstrap_ci``）或似然比检验。
        - 类别极不平衡（如 1:1000）时，模型会倾向于全部预测为多数类，准确率看着很高但
          没有信息量；请报告 AUC / 召回率而不是准确率。
        - 数值上 ``p`` 会被截断到 ``[1e-12, 1-1e-12]`` 再取对数，避免 ``log(0)``；
          这会让极端情况下的对数似然略微"好看"一点，属于可接受的工程折中。

    参考:
        Cox, D.R. (1958) "The Regression Analysis of Binary Sequences", JRSS-B 20(2): 215-242；
        McCullagh & Nelder, "Generalized Linear Models", 2nd ed., Ch. 4（IRLS）；
        Heinze, G. & Schemper, M. (2002) "A solution to the problem of separation in
        logistic regression", Statistics in Medicine 21(16): 2409-2419。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n = Xm.shape[0]
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    if not np.all((yv == 0.0) | (yv == 1.0)):
        raise ValueError("logistic_regression 要求 y 只含 0/1")
    if np.all(yv == yv[0]):
        raise ValueError("y 只有单一类别，logistic 回归无定义")
    lrf = float(lr)
    if not (0.0 < lrf < 1.0):
        raise ValueError(f"lr（回溯收缩因子）必须落在 (0,1) 内，得到 {lr!r}")
    it_max = int(max_iter)
    if it_max < 1:
        raise ValueError("max_iter 必须 >= 1")
    tolf = float(tol)
    if tolf <= 0.0:
        raise ValueError("tol 必须为正")

    Xd = np.column_stack([np.ones(n, dtype=float), Xm])
    k = Xd.shape[1]
    b = np.zeros(k, dtype=float)

    def _loglik(beta: np.ndarray) -> float:
        eta = np.clip(Xd @ beta, -500.0, 500.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        return float(np.sum(yv * np.log(p) + (1.0 - yv) * np.log(1.0 - p)))

    def _probs(beta: np.ndarray) -> np.ndarray:
        eta = np.clip(Xd @ beta, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-eta))

    ll = _loglik(b)
    converged = False
    iters = 0
    for it in range(1, it_max + 1):
        iters = it
        p = np.clip(_probs(b), 1e-12, 1.0 - 1e-12)
        w = np.maximum(p * (1.0 - p), 1e-12)
        grad = Xd.T @ (yv - p)
        hess = Xd.T @ (Xd * w[:, None]) + 1e-10 * np.eye(k)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            converged = False
            break
        if not np.all(np.isfinite(step)):
            converged = False
            break
        tscale = 1.0
        accepted = False
        b_new = b
        ll_new = ll
        for _ in range(60):
            cand = b + tscale * step
            ll_cand = _loglik(cand)
            if np.isfinite(ll_cand) and ll_cand >= ll - 1e-12:
                b_new, ll_new, accepted = cand, ll_cand, True
                break
            tscale *= lrf
        if not accepted:
            converged = False
            break
        delta = float(np.max(np.abs(b_new - b)))
        b, ll = b_new, ll_new
        if not np.isfinite(ll):
            converged = False
            break
        if float(np.max(np.abs(b))) > 1e6:
            # 完全分离：系数发散，对数似然趋近 0
            converged = False
            break
        if delta < tolf:
            converged = True
            break

    prob = _probs(b)
    if not np.all(np.isfinite(prob)) or not np.isfinite(ll):
        converged = False
    return {
        "coef": b,
        "log_lik": float(ll) if np.isfinite(ll) else float("nan"),
        "iterations": int(iters),
        "converged": bool(converged),
        "prob": prob,
    }


# --------------------------------------------------------------------------
# 重抽样
# --------------------------------------------------------------------------

def bootstrap_ci(
    x: Sequence[float],
    statistic: Optional[Callable[[np.ndarray], float]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """百分位 Bootstrap 置信区间。

    参数:
        x: 一维样本。
        statistic: 统计量函数，接受一维数组返回标量；``None`` 表示样本均值。
        n_boot: 重抽样次数（>= 1）。
        alpha: 显著性水平，置信度为 ``1 - alpha``（如 0.05 -> 95% 区间）。
        seed: 随机种子；``None`` 使用 ``DEFAULT_SEED``。

    返回:
        dict：``estimate``（原样本上的统计量）、``ci_low`` / ``ci_high``（百分位区间端点）、
        ``alpha``、``n_boot``、``boot_se``（重抽样分布的标准差，作为标准误估计）。

    算法:
        每次有放回抽 n 个（``n`` 为样本量），得到 ``n_boot`` 个统计量；
        区间取重抽样分布的 ``alpha/2`` 与 ``1-alpha/2`` 分位数（线性插值）。

    复杂度:
        时间 O(n_boot * n * cost(statistic)) / 空间 O(n_boot * n)（实现按块重抽样，实际 O(n)）。

    陷阱:
        - **百分位法不是万能的**：统计量分布明显偏斜或有偏（如样本最大值、方差、
          相关系数在 n 小时）时，百分位区间覆盖率会明显低于名义水平，且区间可能越界
          （例如比例类统计量给出负的下限）。更稳的 BCa / 学生化 Bootstrap 本模块未实现。
        - ``n_boot`` 决定区间的**分辨率**：n_boot=2000 时 95% 区间的端点由第 50 与第 1950
          个次序统计量决定，重跑一次换种子结果会在第 2~3 位有效数字上变动。
          要报告稳定数字请把 ``n_boot`` 提到 1e4 并固定 ``seed``。
        - Bootstrap 假设样本**独立同分布**：时间序列直接对点重抽样会破坏自相关结构，
          必须改用块状 Bootstrap 或对残差重抽样（本模块未实现）。
        - 样本量很小时（n < 10）重抽样分布只有很粗的粒度，区间基本没有意义。

    参考:
        Efron, B. (1979) "Bootstrap Methods: Another Look at the Jackknife", Annals of
        Statistics 7(1): 1-26；Efron & Tibshirani, "An Introduction to the Bootstrap", 1993, Ch. 13。
    """
    xv = as_vector(x, "x")
    n = xv.size
    if n < 2:
        raise ValueError("bootstrap_ci 至少需要 2 个观测")
    nb = int(n_boot)
    if nb < 1:
        raise ValueError("n_boot 必须 >= 1")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha 必须落在 (0,1) 内，得到 {alpha!r}")
    stat = (lambda arr: float(np.mean(arr))) if statistic is None else statistic
    if not callable(stat):
        raise ValueError("statistic 必须是可调用对象（接受一维数组、返回标量）")
    est = float(stat(xv))
    gen = rng(seed)
    vals = np.empty(nb, dtype=float)
    for b in range(nb):
        idx = gen.integers(0, n, size=n)
        vals[b] = float(stat(xv[idx]))
    if not np.all(np.isfinite(vals)):
        raise ValueError("重抽样过程中出现非有限统计量，请检查 statistic 的定义域")
    lo = float(np.percentile(vals, 100.0 * a / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - a / 2.0)))
    return {
        "estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": a,
        "n_boot": nb,
        "boot_se": float(np.std(vals, ddof=1)) if nb > 1 else 0.0,
    }


def bca_bootstrap_ci(
    x: Sequence[float],
    statistic: Optional[Callable[[np.ndarray], float]] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """BCa（偏差校正 + 加速度）Bootstrap 置信区间。

    参数:
        x: 一维样本，至少 3 个观测（jackknife 加速度要用到三阶矩）。
        statistic: 统计量函数，接受一维数组返回标量；``None`` 表示样本均值。
        n_boot: 重抽样次数（>= 2，实际使用建议 >= 2000）。
        alpha: 显著性水平，置信度为 ``1 - alpha``（如 0.05 -> 95% 区间）。
        seed: 随机种子；``None`` 使用 ``DEFAULT_SEED``。

    返回:
        dict：``ci_lower`` / ``ci_upper``（BCa 区间端点）、``theta_hat``（原样本统计量）、
        ``bias_correction``（偏差校正量 ``z0 = Phi^{-1}(#{theta* < theta_hat} / B)``）、
        ``acceleration``（jackknife 加速度 ``a``）、``n_boot``。

    算法:
        1. 有放回抽 ``B = n_boot`` 个容量 n 的样本，得到重抽样分布 ``theta*``；
        2. 偏差校正 ``z0 = Phi^{-1}(frac{theta* < theta_hat})``，比例先裁剪到
           ``[1/(B+1), B/(B+1)]``（否则端点会变成 ±inf）；
        3. 加速度用 jackknife 伪值：``a = sum_i (theta_bar - theta_(i))^3 /
           (6 * [sum_i (theta_bar - theta_(i))^2]^{3/2})``；
        4. 调整分位数
           ``alpha_1 = Phi( z0 + (z0 + z_{alpha/2}) / (1 - a (z0 + z_{alpha/2})) )``，
           ``alpha_2`` 把 ``z_{alpha/2}`` 换成 ``z_{1-alpha/2}``；
           区间取 ``theta*`` 的 ``alpha_1`` / ``alpha_2`` 分位数（线性插值）。
        ``Phi`` 与 ``Phi^{-1}`` 分别用自实现的 ``_norm_cdf`` 与 ``_norm_ppf``，
        整条链不依赖 scipy。

    复杂度:
        时间 O((B + n) * cost(statistic)) / 空间 O(B + n)。

    陷阱:
        - **jackknife 加速度在小样本上很脆**：n < 10 时 ``a`` 基本都是噪音，BCa 可能
          反而不如百分位法；样本很小时请直接报告 t 区间或精确分布。
        - 当 ``a`` 接近 ``1/(z0 + z_{alpha/2})`` 时调整分位数的分母趋近 0，端点会跳到
          极端次序统计量上；本实现检测到分母过小时抛 ``ValueError``，而不是静默返回
          一个假的区间。
        - 与百分位法一样要求样本**独立同分布**：时间序列或分层抽样数据直接对点重抽样
          会破坏结构，应改用块状 / 分层 Bootstrap（本模块未实现）。
        - ``n_boot`` 决定端点分辨率，换种子会在第 2~3 位有效数字上变动；
          本函数的价格也是 ``O(B * n)``，B=1e4、n=1e4 时明显变慢。
        - BCa 是"用重抽样修正百分位法"的近似方法，区间**不是**精确的置信区间，
          覆盖率只在渐近意义下等于名义水平。

    参考:
        Efron, B. (1987) "Better Bootstrap Confidence Intervals", Journal of the
        American Statistical Association 82(397): 171-185；
        Efron & Tibshirani, "An Introduction to the Bootstrap", 1993, Ch. 14。
    """
    xv = as_vector(x, "x")
    n = int(xv.size)
    if n < 3:
        raise ValueError(f"BCa 至少需要 3 个观测（jackknife 加速度要求），得到 {n}")
    nb = int(n_boot)
    if nb < 2:
        raise ValueError("n_boot 必须 >= 2")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha 必须落在 (0,1) 内，得到 {alpha!r}")
    stat = (lambda arr: float(np.mean(arr))) if statistic is None else statistic
    if not callable(stat):
        raise ValueError("statistic 必须是可调用对象（接受一维数组、返回标量）")
    theta_hat = float(stat(xv))
    if not math.isfinite(theta_hat):
        raise ValueError("原样本上的统计量不是有限值")
    gen = rng(seed)
    vals = np.empty(nb, dtype=float)
    for b in range(nb):
        idx = gen.integers(0, n, size=n)
        vals[b] = float(stat(xv[idx]))
    if not np.all(np.isfinite(vals)):
        raise ValueError("重抽样过程中出现非有限统计量，请检查 statistic 的定义域")
    frac = float(np.count_nonzero(vals < theta_hat)) / float(nb)
    lo_frac = 1.0 / float(nb + 1)
    hi_frac = float(nb) / float(nb + 1)
    if frac < lo_frac:
        frac = lo_frac
    elif frac > hi_frac:
        frac = hi_frac
    z0 = _norm_ppf(frac)
    jack = np.empty(n, dtype=float)
    for i in range(n):
        jack[i] = float(stat(np.delete(xv, i)))
    if not np.all(np.isfinite(jack)):
        raise ValueError("jackknife 过程中出现非有限统计量，请检查 statistic 的定义域")
    jbar = float(jack.mean())
    d = jbar - jack
    s2 = float(np.sum(d * d))
    if s2 > 0.0:
        acc = float(np.sum(d ** 3)) / (6.0 * s2 ** 1.5)
    else:
        acc = 0.0

    def _adjusted(zq: float) -> float:
        den = 1.0 - acc * (z0 + zq)
        if abs(den) < 1e-9:
            raise ValueError(
                "BCa 调整分位数的分母趋近 0（加速度过大）：请增大样本量或改用百分位法"
            )
        p = _norm_cdf(z0 + (z0 + zq) / den)
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p

    p_lo = _adjusted(_norm_ppf(0.5 * a))
    p_hi = _adjusted(_norm_ppf(1.0 - 0.5 * a))
    lower = float(np.percentile(vals, 100.0 * p_lo))
    upper = float(np.percentile(vals, 100.0 * p_hi))
    if lower > upper:
        lower, upper = upper, lower
    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "theta_hat": theta_hat,
        "bias_correction": float(z0),
        "acceleration": float(acc),
        "n_boot": nb,
    }


def permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = 2000,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """两样本置换检验（统计量默认是均值之差）。

    参数:
        x, y: 两组独立样本，长度可以不同。
        n_perm: 置换次数（>= 1）。
        seed: 随机种子；``None`` 使用 ``DEFAULT_SEED``。

    返回:
        dict：``stat``（观测到的 ``mean(x) - mean(y)``）、``p_value``（双侧，
        ``(1 + #{|置换统计量| >= |观测值|}) / (1 + n_perm)``）、``n_perm``。

    算法:
        把两组数据混成一个池，随机重新分配标签（保持两组样本量不变），
        每次算一遍均值之差；用置换分布的绝对值与观测值比较得到双侧 p 值。

    复杂度:
        时间 O(n_perm * (nx + ny)) / 空间 O(nx + ny)。

    陷阱:
        - **p 值的分辨率受 n_perm 限制**：n_perm=2000 时最小可能 p 值约为 1/2001 ≈ 0.0005，
          再小就只能报 "p < 0.001"。要报更小的 p 值必须加大 n_perm。
        - 置换检验的原假设是**两组可交换（同分布）**，比"均值相等"更强：两组方差不齐时，
          均值之差的置换检验实际是在检验整个分布是否相同，结论解释要小心。
        - 固定 ``seed`` 后结果可复现，但换种子 p 值会变动（n_perm 越小变动越大）；
          论文中应固定并报告 seed。
        - 两组样本量很小时（如各 3 个）可能的置换组合非常有限，p 值是离散的，
          不要报告小数点后很多位。

    参考:
        Fisher, R.A. (1935) "The Design of Experiments"（随机化检验思想）；
        Pitman, E.J.G. (1937) "Significance tests which may be applied to samples from any
        populations", JRSS Supplement 4(1): 119-130。
    """
    xv = as_vector(x, "x")
    yv = as_vector(y, "y")
    nx = xv.size
    ny = yv.size
    if nx < 2 or ny < 2:
        raise ValueError(f"两组样本都至少需要 2 个观测，得到 nx={nx}, ny={ny}")
    np_ = int(n_perm)
    if np_ < 1:
        raise ValueError("n_perm 必须 >= 1")
    obs = float(xv.mean() - yv.mean())
    pool = np.concatenate([xv, yv])
    gen = rng(seed)
    count = 0
    for _ in range(np_):
        perm = gen.permutation(nx + ny)
        a = pool[perm[:nx]]
        b = pool[perm[nx:]]
        if abs(float(a.mean() - b.mean())) >= abs(obs) - 1e-15:
            count += 1
    p = (1.0 + count) / (1.0 + np_)
    return {"stat": obs, "p_value": float(p), "n_perm": np_}


# --------------------------------------------------------------------------
# 主成分分析 / 因子分析
# --------------------------------------------------------------------------

def _fix_component_signs(components: np.ndarray) -> np.ndarray:
    """把每个主成分/因子方向的符号固定下来，保证多次调用结果一致。

    参数:
        无。

    返回:
        逐行（每个分量）做过符号翻转的新数组。

    算法:
        令每行绝对值最大的那个元素为正；若该元素本身为负则整行取反。
        这是 sklearn 风格的可复现符号约定，避免特征向量因数值库实现细节整体反号。

    复杂度:
        时间 O(k p) / 空间 O(k p)。

    陷阱:
        当一行里绝对值最大的元素**恰好接近 0**（该方向几乎不承载方差）时，
        该约定在浮点噪声下不稳定；此时行本身没有意义，应先检查特征值是否可忽略。

    参考:
        sklearn.decomposition.PCA 的 ``svd_flip`` 符号约定。
    """
    out = np.array(components, dtype=float, copy=True)
    for i in range(out.shape[0]):
        row = out[i]
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0.0:
            out[i] = -row
    return out


def _varimax(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Kaiser 方差极大（varimax）正交旋转。

    参数:
        loadings: 因子载荷矩阵，形状 (p, k)。
        max_iter: 最大迭代次数。
        tol: 旋转准则的收敛阈值。

    返回:
        旋转后的载荷矩阵，形状 (p, k)（正交旋转不改变共性方差）。

    算法:
        最大化准则 ``V = sum_j (sum_i l_ij^4) - (1/p) sum_j (sum_i l_ij^2)^2``；
        每步用 SVD 求最优正交旋转 ``L <- L U V'``，其中
        ``U S V' = L' (L^3 - (1/p) L diag(colsum))``（Kaiser 1958 的经典迭代）。

    复杂度:
        时间 O(max_iter * (k^2 p + k^3)) / 空间 O(kp)。

    陷阱:
        旋转只改变载荷的**分配**，不改变共性方差与总解释量；把旋转前后
        ``variance_explained`` 的差异解释成"模型变好了"是常见误读。
        k=1 时旋转无意义，本实现直接原样返回。

    参考:
        Kaiser, H.F. (1958) "The varimax criterion for analytic rotation in factor
        analysis", Psychometrika 23(3): 187-200。
    """
    p, k = loadings.shape
    if k < 2:
        return np.array(loadings, dtype=float, copy=True)
    rot = np.array(loadings, dtype=float, copy=True)
    prev = -np.inf
    for _ in range(int(max_iter)):
        sq = rot ** 2
        colsum = sq.sum(axis=0)
        crit = float(np.sum(sq ** 2) - np.sum(colsum ** 2) / p)
        if abs(crit - prev) < tol:
            break
        prev = crit
        target = rot ** 3 - rot * (colsum / p)
        u, _, vh = np.linalg.svd(rot.T @ target)
        rot = rot @ (u @ vh)
    return rot


def pca(
    X: Sequence[Sequence[float]],
    n_components: Optional[int] = None,
    standardize: bool = True,
) -> Dict[str, object]:
    """主成分分析（协方差/相关阵特征分解），返回载荷、得分与解释率。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        n_components: 保留的主成分个数，合法范围 1..n_features；None 表示全部保留。
        standardize: True 时对每列做 z-score 标准化（减均值、除以总体标准差 ddof=0），
            即在**相关阵**上做主成分；False 时只中心化（在**协方差阵**上做主成分）。

    返回:
        dict，键为：
        ``components``  形状 (k, p) 的主成分方向，每行是一个单位向量；
        ``eigenvalues``  形状 (k,) 的特征值（降序，已把浮点负值截为 0）；
        ``explained_variance_ratio``  形状 (k,) 的解释方差比例，之和为 1（k=p 时）；
        ``scores``  形状 (n, k) 的主成分得分，``scores = Z @ components.T``；
        ``cumulative_ratio``  形状 (k,) 的累计解释率。

    算法:
        1. 中心化（必要时再除以列标准差，标准差为 0 的列置 1）得到 Z；
        2. 协方差阵 ``C = Z'Z / (n-1)``，用 ``np.linalg.eigh``（对称阵专用）分解；
        3. 特征值降序重排，截取前 k 个，负特征值（数值噪声）截为 0；
        4. 符号约定：让每个主成分里绝对值最大的分量为正（见 ``_fix_component_signs``）；
        5. 解释率 ``lambda_i / sum(lambda)``，得分 ``Z @ V_k``。

    复杂度:
        时间 O(n p^2 + p^3) / 空间 O(np + p^2)。

    陷阱:
        - ``standardize`` 决定你在相关阵还是协方差阵上工作，两者结果**不可比**：
          量纲差异大的指标不做标准化时，方差最大的那个指标会独占第一主成分。
        - 特征向量的符号本身没有定义（``v`` 与 ``-v`` 同样合法），但**得分矩阵会跟着反号**；
          不固定符号的话，两次运行或换一个 numpy 版本就可能得到反号的得分。
        - 常数自变量列（标准差 0）在标准化下被置为尺度 1，会变成一个"全 0"的无信息方向，
          它只能贡献 0 方差却占掉一个成分位，应当在建模前删掉。
        - 变量个数 p 大于样本数 n 时协方差阵秩亏，最多只有 n-1 个非零特征值，
          报告解释率时不要按 p 个成分去解释。

    参考:
        Jolliffe, I.T. (2002) "Principal Component Analysis", 2nd ed., Springer, Ch. 2-3；
        Pearson, K. (1901) "On lines and planes of closest fit to systems of points in
        space", Philosophical Magazine 2(11): 559-572。
    """
    Xm = as_matrix(X, "X")
    n, p = Xm.shape
    if n < 2:
        raise ValueError(f"pca 至少需要 2 个样本，得到 n={n}")
    if n_components is None:
        k = p
    else:
        k = int(n_components)
        if k < 1 or k > p:
            raise ValueError(f"n_components 必须落在 1..{p}，得到 {n_components!r}")
    mu = Xm.mean(axis=0)
    Z = Xm - mu
    if standardize:
        sd = Z.std(axis=0, ddof=0)
        sd_safe = np.where(sd > 0.0, sd, 1.0)
        Z = Z / sd_safe
    cov = Z.T @ Z / float(n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    comps_all = _fix_component_signs(eigvecs[:, order].T)
    total = float(eigvals.sum())
    if total <= 0.0:
        raise ValueError("数据方差为 0（所有样本完全相同），无法做主成分分析")
    ratio_all = eigvals / total
    components = comps_all[:k]
    eigvals_k = eigvals[:k]
    ratio_k = ratio_all[:k]
    scores = Z @ components.T
    return {
        "components": components,
        "eigenvalues": eigvals_k,
        "explained_variance_ratio": ratio_k,
        "scores": scores,
        "cumulative_ratio": np.cumsum(ratio_k),
    }


def factor_analysis(
    X: Sequence[Sequence[float]],
    n_factors: int = 2,
    max_iter: int = 200,
    tol: float = 1e-8,
    rotate: bool = True,
) -> Dict[str, object]:
    """探索性因子分析：主因子法（principal factor）估计载荷 + 可选 varimax 旋转。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)，先在内部做 z-score 标准化。
        n_factors: 因子个数，合法范围 1..p-1。
        max_iter: 迭代重估共性方差的最大轮数。
        tol: 共性方差的最大变化量小于 tol 即认为收敛。
        rotate: 是否做 varimax 正交旋转（默认 True；旋转不改变共性方差与解释总量，
            只让载荷更"简单"，便于给因子命名）。

    返回:
        dict，键为：
        ``loadings``  形状 (p, k) 的因子载荷（旋转后，已固定符号并按解释量降序排列）；
        ``communalities``  形状 (p,) 的共性方差 h2（载荷行平方和）；
        ``uniqueness``  形状 (p,) 的特殊方差 ``1 - h2``；
        ``variance_explained``  形状 (k,) 的每个因子解释的方差**比例**（列平方和 / p）；
        ``n_iter``  int，实际迭代轮数。

    算法:
        1. 标准化得到 Z，算相关阵 ``R = Z'Z/(n-1)``；
        2. 共性方差初值取"该变量与其余变量的最大绝对相关系数"（对角元不足的常用代理）；
        3. 迭代：把 R 的对角元替换为 h2 得约化相关阵，取前 k 个特征对，
           载荷 ``L = V_k sqrt(max(lambda_k, 0))``，再令 ``h2 <- diag(L L')``，
           直到 h2 的最大变化量小于 tol（Heywood 情形把 h2 截断到 [0, 1]）；
        4. 用最终 h2 重算一次载荷以保证自洽，可选 varimax 旋转，
           再按各因子解释量降序重排并固定符号。

    复杂度:
        时间 O(max_iter * p^3) / 空间 O(np + p^2)。

    陷阱:
        - 主因子法不是极大似然：它把 h2 当已知反复代入，收敛到的是"约化相关阵"
          的特征解，与 ``statsmodels`` 的 ML 解会有差异，论文里要写清用的是哪一种。
        - **Heywood 情形**（某个 h2 顶到 1，uniqueness 变 0 或负）说明因子数过多或
          模型不适定；本实现把 h2 截断到 [0,1] 只是让迭代能跑完，并不解决问题。
        - 旋转后的载荷矩阵不再"按方差降序"，本实现在旋转后显式重排，
          因此 ``loadings`` 的列序与未旋转时的特征向量序号不同。
        - 因子个数的选择（碎石图/平行分析）本函数不做，必须由调用者决定并说明理由。

    参考:
        Harman, H.H. (1976) "Modern Factor Analysis", 3rd ed., University of Chicago Press；
        Kaiser, H.F. (1958) Psychometrika 23(3): 187-200（varimax）。
    """
    Xm = as_matrix(X, "X")
    n, p = Xm.shape
    if n < 3:
        raise ValueError(f"factor_analysis 至少需要 3 个样本，得到 n={n}")
    if p < 2:
        raise ValueError(f"至少需要 2 个变量，得到 p={p}")
    k = int(n_factors)
    if k < 1 or k >= p:
        raise ValueError(f"n_factors 必须落在 1..{p - 1}，得到 {n_factors!r}")
    it_max = int(max_iter)
    if it_max < 1:
        raise ValueError("max_iter 必须 >= 1")
    tolf = float(tol)
    if tolf <= 0.0:
        raise ValueError("tol 必须为正")

    Z = Xm - Xm.mean(axis=0)
    sd = Z.std(axis=0, ddof=0)
    Z = Z / np.where(sd > 0.0, sd, 1.0)
    R = Z.T @ Z / float(n - 1)

    off = np.abs(R - np.eye(p))
    h2 = off.max(axis=1)
    h2 = np.clip(h2, 0.0, 1.0)
    n_iter = 0
    for it in range(1, it_max + 1):
        n_iter = it
        Rs = np.array(R, dtype=float, copy=True)
        np.fill_diagonal(Rs, h2)
        evals, evecs = np.linalg.eigh(Rs)
        order = np.argsort(evals)[::-1][:k]
        lam = np.maximum(evals[order], 0.0)
        load_tmp = evecs[:, order] * np.sqrt(lam)
        h2_new = np.clip(np.sum(load_tmp ** 2, axis=1), 0.0, 1.0)
        change = float(np.max(np.abs(h2_new - h2)))
        h2 = h2_new
        if change < tolf:
            break

    Rs = np.array(R, dtype=float, copy=True)
    np.fill_diagonal(Rs, h2)
    evals, evecs = np.linalg.eigh(Rs)
    order = np.argsort(evals)[::-1][:k]
    lam = np.maximum(evals[order], 0.0)
    loadings = _fix_component_signs((evecs[:, order] * np.sqrt(lam)).T).T

    if rotate:
        loadings = _varimax(loadings)
        var = np.sum(loadings ** 2, axis=0)
        order = np.argsort(var)[::-1]
        loadings = _fix_component_signs(loadings[:, order].T).T

    communalities = np.sum(loadings ** 2, axis=1)
    uniqueness = 1.0 - communalities
    variance_explained = np.sum(loadings ** 2, axis=0) / float(p)
    return {
        "loadings": loadings,
        "communalities": communalities,
        "uniqueness": uniqueness,
        "variance_explained": variance_explained,
        "n_iter": int(n_iter),
    }


# --------------------------------------------------------------------------
# 正则化回归 / 计数回归 / 回归诊断
# --------------------------------------------------------------------------

def _soft_threshold(z: float, gamma: float) -> float:
    """软阈值算子 ``sign(z) * max(|z| - gamma, 0)``。

    参数:
        z: 输入标量（这里是坐标下降中的偏相关系数 ``x_j'r``）。
        gamma: 阈值（>= 0），对应 L1 惩罚强度。

    返回:
        收缩后的标量；``|z| <= gamma`` 时精确返回 0。

    算法:
        分段线性收缩：正侧平移 -gamma，负侧平移 +gamma，中间压到 0。
        它是 L1 惩罚下的近端算子，也是 Lasso 产生**稀疏解**的唯一来源。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        必须返回**精确的 0**（而不是 1e-17 之类的小量），否则 ``n_nonzero`` 会
        把数值噪声当成非零系数；用 ``max(|z| - gamma, 0)`` 而不是 ``|z| - gamma`` 是关键。

    参考:
        Friedman, J., Hastie, T. & Tibshirani, R. (2010) "Regularization Paths for
        Generalized Linear Models via Coordinate Descent", JSS 33(1): 1-22。
    """
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


def lasso_regression(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    alpha: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Lasso（L1 正则最小二乘）的坐标下降求解，含截距。

    参数:
        X: 设计矩阵，形状 (n, p)，**不含截距列**。
        y: 因变量，长度 n。
        alpha: L1 惩罚强度（>= 0）。0 时退化为 OLS。
        max_iter: 坐标下降的最大扫描轮数。
        tol: 一轮扫描中系数最大变化量小于 tol 即认为收敛。

    返回:
        dict，键为：
        ``coef``  长度 p 的斜率（原始尺度，不参与惩罚的列照原样返回）；
        ``intercept``  float，截距 ``mean(y) - mean(X) @ coef``；
        ``n_nonzero``  int，``|coef| > 0`` 的个数（软阈值给的是精确 0）；
        ``objective``  float，最终目标 ``RSS/(2n) + alpha*||coef||_1``；
        ``n_iter``  int，实际扫描轮数（收敛时是收敛发生的轮次，否则等于 ``max_iter``）；
        ``converged``  bool，是否在 ``max_iter`` 轮内达到 ``tol`` 判据；
                      ``False`` 表示"是被轮数上限截断的"，此时 ``coef`` 只是近似解。

    算法:
        1. 先把 y 与 X 各列**中心化**（截距因此不参与惩罚），得到 ``yc``、``Xc``；
        2. 坐标下降：维护当前残差 ``r = yc - Xc beta``，对第 j 列做单变量更新
           ``beta_j <- soft(x_j'r_j, n*alpha) / (x_j'x_j)``，其中
           ``r_j = r + x_j * beta_j``（把该列的贡献加回去）；
        3. 每次更新后同步刷新 r；一轮中系数最大变化量小于 tol 时停止；
        4. 截距事后还原为 ``mean(y) - mean(X) @ coef``。

    复杂度:
        时间 O(max_iter * n p) / 空间 O(np)。

    陷阱:
        - **本实现不做列标准化**（只中心化），因此同一个 alpha 对不同量纲的列
          惩罚力度完全不同：把某列的单位从"米"换成"毫米"，该列系数会缩小 1000 倍，
          L1 阈值相对它就形同不存在。要跨变量比较稀疏性，请先自行标准化 X。
        - 常数自变量列（``x_j'x_j = 0``）无法做单变量更新，本实现把该列系数固定为 0
          （而不是除零得到 NaN），但它仍会原样出现在 ``coef`` 里。
        - ``alpha`` 很大的时候所有系数被压成 0，``coef`` 全零但 ``intercept`` 就是
          ``mean(y)``；不要把这种"全零模型"当成有效结论。
        - 坐标下降**没有 p 值**：L1 解是有偏的，且被选中的变量个数本身依赖 alpha，
          常规 t 检验完全失效；要报告不确定性请用 Bootstrap 或去偏 Lasso。
        - ``tol`` 是系数变化量而不是目标函数变化量；目标函数在最优解附近是平的，
          系数收敛得比目标值慢，报告 ``objective`` 时要注意这点。
        - ``converged=False`` 是**真实存在**的情形：``max_iter`` 偏小、``tol`` 偏严、
          或列的量纲差异极大时都会触发。此时 ``n_iter == max_iter``，解仍在下降路径上，
          既不是 NaN 也不是"已经收敛"——请先调大 ``max_iter`` 或对 X 做标准化再复算。

    参考:
        Tibshirani, R. (1996) "Regression Shrinkage and Selection via the Lasso",
        JRSS-B 58(1): 267-288；
        Friedman, Hastie & Tibshirani (2010), JSS 33(1): 1-22（坐标下降）。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n, p = Xm.shape
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise ValueError(f"alpha 必须是 >= 0 的有限数，得到 {alpha!r}")
    it_max = int(max_iter)
    if it_max < 1:
        raise ValueError("max_iter 必须 >= 1")
    tolf = float(tol)
    if tolf <= 0.0:
        raise ValueError("tol 必须为正")

    xbar = Xm.mean(axis=0)
    ybar = float(yv.mean())
    Xc = Xm - xbar
    yc = yv - ybar
    colsq = np.sum(Xc ** 2, axis=0)
    safe = np.where(colsq > 0.0, colsq, 1.0)
    beta = np.zeros(p, dtype=float)
    r = yc.copy()
    thr = float(n) * a
    n_iter = 0
    converged = False
    for it in range(1, it_max + 1):
        n_iter = it
        max_delta = 0.0
        for j in range(p):
            if colsq[j] <= 0.0:
                continue
            rj = r + Xc[:, j] * beta[j]
            rho = float(Xc[:, j] @ rj)
            new = _soft_threshold(rho, thr) / safe[j]
            delta = new - beta[j]
            if delta != 0.0:
                r = rj - Xc[:, j] * new
                beta[j] = new
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
        if max_delta < tolf:
            converged = True
            break
    intercept = float(ybar - float(xbar @ beta))
    resid = yv - (Xm @ beta + intercept)
    objective = float(resid @ resid / (2.0 * n) + a * float(np.sum(np.abs(beta))))
    return {
        "coef": beta,
        "intercept": intercept,
        "n_nonzero": int(np.count_nonzero(beta)),
        "objective": objective,
        "n_iter": int(n_iter),
        "converged": bool(converged),
    }


def poisson_regression(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Poisson 回归（对数链接 GLM）的 IRLS 求解。

    参数:
        X: 设计矩阵，形状 (n, p)，**不含截距列**（本函数自动加一列 1）。
        y: 计数因变量，长度 n，必须全部非负。
        max_iter: IRLS 最大迭代次数。
        tol: 系数最大变化量小于 tol 即认为收敛。

    返回:
        dict，键为：
        ``coef``  长度 p 的斜率；
        ``intercept``  float，截距；
        ``deviance``  float，残差偏差 ``2 sum [y log(y/mu) - (y - mu)]``（y=0 的项取 2*mu）；
        ``loglik``  float，对数似然 ``sum [y log(mu) - mu - lgamma(y+1)]``；
        ``aic``  float，``-2 loglik + 2 (p+1)``；
        ``n_iter``  int，实际迭代次数。

    算法:
        1. 线性预测量 ``eta = Xd b``，均值 ``mu = exp(eta)``（eta 截断到 [-50, 50]，
           mu 下限 1e-10）；
        2. IRLS：工作响应 ``z = eta + (y - mu)/mu``，权重 ``w = mu``，
           解加权最小二乘 ``b <- argmin ||sqrt(w) (z - Xd b)||^2``（用 ``np.linalg.lstsq``）；
        3. 若新偏差大于旧偏差则步长折半（最多 30 次），保证单调下降；
        4. 系数最大变化量小于 tol 时停止。

    复杂度:
        时间 O(max_iter * n p^2) / 空间 O(np)。

    陷阱:
        - **过散布（overdispersion）**：真实计数数据的方差常大于均值，Poisson 假设下
          标准误会被严重低估、p 值偏小。本函数不返回标准误，但如果用似然比/Wald 检验，
          请先检查偏差/自由度是否远大于 1，必要时改用负二项回归。
        - ``y`` 含 0 时 ``log(y/mu)`` 会取到 ``0 * (-inf)``，本实现显式把 y=0 的偏差项
          取为 2*mu（这是极限值），不能直接对 y 取对数。
        - 计数需要**曝光量**（不同观测的暴露时间不同）时，正确做法是把 ``log(offset)``
          作为系数固定为 1 的偏移项加入；把它当普通自变量会得到完全错误的系数。
        - 设计矩阵里若有完全共线的列，IRLS 的加权最小二乘会给出最小范数解，
          系数不可解释，请先用 ``vif`` 检查。

    参考:
        McCullagh, P. & Nelder, J.A. (1989) "Generalized Linear Models", 2nd ed.,
        Chapman & Hall, Ch. 4-5（IRLS 与偏差）；
        Nelder, J.A. & Wedderburn, R.W.M. (1972) "Generalized Linear Models",
        JRSS-A 135(3): 370-384。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n, p = Xm.shape
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    if np.any(yv < 0.0):
        raise ValueError("poisson_regression 要求 y 为非负计数")
    it_max = int(max_iter)
    if it_max < 1:
        raise ValueError("max_iter 必须 >= 1")
    tolf = float(tol)
    if tolf <= 0.0:
        raise ValueError("tol 必须为正")

    Xd = np.column_stack([np.ones(n, dtype=float), Xm])
    k = Xd.shape[1]
    b = np.zeros(k, dtype=float)
    b[0] = math.log(max(float(yv.mean()), 1e-6))

    def _mu_of(beta: np.ndarray) -> np.ndarray:
        eta = np.clip(Xd @ beta, -50.0, 50.0)
        return np.maximum(np.exp(eta), 1e-10)

    def _deviance(mu: np.ndarray) -> float:
        pos = yv > 0.0
        term = np.where(
            pos,
            yv * np.log(np.where(pos, yv, 1.0) / mu) - (yv - mu),
            mu,
        )
        return float(2.0 * np.sum(term))

    dev_old = _deviance(_mu_of(b))
    n_iter = 0
    for it in range(1, it_max + 1):
        n_iter = it
        mu = _mu_of(b)
        eta = np.clip(Xd @ b, -50.0, 50.0)
        w = mu
        sw = np.sqrt(w)
        z = eta + (yv - mu) / mu
        A = Xd * sw[:, None]
        cand, _, _, _ = np.linalg.lstsq(A, z * sw, rcond=None)
        step = 1.0
        accepted = False
        for _ in range(30):
            trial = b + step * (cand - b)
            dev_try = _deviance(_mu_of(trial))
            if np.isfinite(dev_try) and dev_try <= dev_old + 1e-12:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        change = float(np.max(np.abs(trial - b)))
        b = trial
        dev_old = dev_try
        if change < tolf:
            break

    mu = _mu_of(b)
    eta = np.clip(Xd @ b, -50.0, 50.0)
    lgamma_term = float(np.sum([math.lgamma(float(v) + 1.0) for v in yv]))
    loglik = float(np.sum(yv * np.log(mu) - mu) - lgamma_term)
    deviance = _deviance(mu)
    return {
        "coef": b[1:].copy(),
        "intercept": float(b[0]),
        "deviance": float(deviance),
        "loglik": float(loglik),
        "aic": float(-2.0 * loglik + 2.0 * k),
        "n_iter": int(n_iter),
    }


def durbin_watson(residual: Sequence[float]) -> float:
    """Durbin-Watson 统计量：回归残差一阶自相关的经典检验量。

    参数:
        residual: 残差序列，长度 n >= 2（**必须按时间/顺序排列**）。

    返回:
        float，``DW = sum_{t=2..n} (e_t - e_{t-1})^2 / sum_{t=1..n} e_t^2``，
        取值约在 [0, 4]：≈2 表示无一阶自相关，明显小于 2 表示正自相关，
        明显大于 2 表示负自相关。

    算法:
        直接按定义算相邻残差差分平方和与残差平方和之比；等价形式
        ``DW ≈ 2(1 - rho_1)``（rho_1 为一阶样本自相关，大样本下近似）。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        - 残差必须按**原始观测顺序**传入：排序过的残差会算出接近 2 的 "正常" 值，
          这是最容易骗过自己的用法。
        - 必须含截距！无截距回归的 OLS 残差均值不为 0，DW 会系统性偏离 2，
          此时该统计量的临界值表也不适用。
        - DW 只能检测**一阶**自相关，对高阶或季节性自相关（季度数据的滞后 4）
          无能为力，应改用 Breusch-Godfrey 检验。
        - 只有 DW 落在临界值之间才算"不能拒绝无自相关"，单看"是否接近 2"没有
          统计显著性含义；临界值表依赖 n 与自变量个数，本函数不提供。

    参考:
        Durbin, J. & Watson, G.S. (1951) "Testing for serial correlation in least squares
        regression. II", Biometrika 38(1/2): 159-178。
    """
    e = as_vector(residual, "residual")
    if e.size < 2:
        raise ValueError(f"residual 至少需要 2 个观测，得到 {e.size}")
    denom = float(e @ e)
    if denom <= 0.0:
        raise ValueError("残差全为 0，Durbin-Watson 统计量无定义")
    num = float(np.sum((e[1:] - e[:-1]) ** 2))
    return num / denom


def breusch_pagan(
    residual: Sequence[float],
    X: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """Breusch-Pagan 异方差检验：残差平方对自变量做辅助回归。

    参数:
        residual: OLS 残差，长度 n。
        X: 原回归的设计矩阵，形状 (n, p)，**不含截距列**（辅助回归自动加截距）。

    返回:
        dict，键为：
        ``stat``  float，``LM = n * R2_aux``；
        ``df``  float，自由度 = p（辅助回归中除截距外的自变量个数）；
        ``p_value``  float，卡方上尾概率（大统计量 -> 拒绝同方差）。

    算法:
        1. 取 ``e2 = residual^2``；
        2. 用 ``ols`` 的同一套最小二乘把 e2 对 ``[1, X]`` 回归，得辅助 ``R2``；
        3. ``LM = n * R2``，在原假设（同方差）下渐近服从 ``chi2_p``，
           p 值用模块内自带的卡方生存函数 ``_chi2_sf`` 计算。

    复杂度:
        时间 O(n p^2) / 空间 O(np)。

    陷阱:
        - 这是 **LM（拉格朗日乘数）版本**，不是 Koenker 的学生化版本：它对残差的
          正态性很敏感，厚尾数据会大量假阳性；稳健做法是用 ``e2`` 的稳健方差估计
          （Koenker 1981）或直接用 White 检验。
        - 检验的是"残差平方与 X **线性**相关"，因此对形如 ``var = exp(x'b)`` 的
          异方差很灵，对非线性的方差结构（如 ``var = |x|`` 的某些形态）可能漏检。
        - 拒绝原假设只说明存在异方差，**不告诉你该怎么做**；修正手段是稳健标准误
          （White/HC3）或加权最小二乘，而后者需要知道方差函数的形式。
        - 辅助回归若秩亏（X 有完全共线的列），本实现抛 ``ValueError``，
          请先删掉冗余列。

    参考:
        Breusch, T.S. & Pagan, A.R. (1979) "A simple test for heteroscedasticity and
        random coefficient variation", Econometrica 47(5): 1287-1294；
        Koenker, R. (1981) "A note on studentizing a test for heteroscedasticity",
        Journal of Econometrics 17(1): 107-112。
    """
    e = as_vector(residual, "residual")
    Xm = as_matrix(X, "X")
    n, p = Xm.shape
    if e.size != n:
        raise ValueError(f"residual 长度 {e.size} 与 X 的行数 {n} 不一致")
    e2 = e ** 2
    Xd = np.column_stack([np.ones(n, dtype=float), Xm])
    coef, _, rank, _ = np.linalg.lstsq(Xd, e2, rcond=None)
    if rank < Xd.shape[1]:
        raise ValueError(
            f"辅助回归设计矩阵秩亏（rank={rank} < 列数 {Xd.shape[1]}）：X 存在完全共线的列"
        )
    fitted = Xd @ coef
    tss = float(np.sum((e2 - e2.mean()) ** 2))
    if tss <= 0.0:
        raise ValueError("残差平方为常数，Breusch-Pagan 辅助回归无定义")
    rss = float(np.sum((e2 - fitted) ** 2))
    r2 = 1.0 - rss / tss
    stat = float(n) * r2
    return {
        "stat": float(stat),
        "df": float(p),
        "p_value": float(_chi2_sf(float(stat), float(p))),
    }


def stepwise_selection(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    criterion: str = "aic",
    max_steps: int = 50,
) -> Dict[str, object]:
    """逐步回归（前向 + 后向），以 AIC 或 BIC 为准则做变量筛选。

    参数:
        X: 候选自变量矩阵，形状 (n, p)，**不含截距列**（每次拟合自动加截距）。
        y: 因变量，长度 n。
        criterion: ``"aic"`` 或 ``"bic"``。AIC = ``n ln(RSS/n) + 2k``，
            BIC = ``n ln(RSS/n) + k ln(n)``，k 为含截距的参数个数。
        max_steps: 最多执行多少次增删动作。

    返回:
        dict，键为：
        ``selected``  被选中的列下标（升序 list of int，可能为空）；
        ``coef``  对应 ``selected`` 顺序的斜率；
        ``intercept``  float，截距（``selected`` 为空时即 ``mean(y)``）；
        ``criterion_value``  最终模型的准则值（越小越好）；
        ``history``  list，每项 ``{"step","action","index","criterion"}``，
            action 取 ``"add"`` 或 ``"remove"``。

    算法:
        1. 从空模型（只有截距）出发；
        2. 每一轮同时评估所有"加入一个未选变量"与"剔除一个已选变量"的候选模型，
           取准则值最低且**严格优于**当前模型的动作执行（这就是"逐步"而非纯前向）；
        3. 没有任何动作能改进准则值、或达到 ``max_steps`` 时停止；
        4. 用最终变量集重跑一次 OLS 得到系数与截距。

    复杂度:
        时间 O(max_steps * p * n p^2) / 空间 O(np)。

    陷阱:
        - **逐步回归后的 p 值不可信**：变量是被数据挑出来的，常规 t 检验的名义显著性
          水平严重失真（选择性推断问题），不要用它给出"某变量显著"的结论。
        - AIC/BIC 比较的是同一响应 ``y`` 下的模型，必须保持样本完全一致；
          存在缺失值时必须先插补再筛选，不能边删样本边选变量。
        - 完全共线的候选列会让 OLS 抛 ``ValueError``；本实现在评估候选模型时把这类
          候选直接判为不可用（跳过），而不是让整个筛选崩掉。
        - BIC 比 AIC 惩罚更重、倾向于更小的模型；样本量 n 很大时两者差异明显，
          换准则后选出的变量集不同属于正常现象，必须在论文里写明用的是哪一个。
        - 本函数只做**线性**模型的变量筛选；被排除的变量可能以交互项/非线性形式
          起作用，逐步回归对此完全无感。

    参考:
        Akaike, H. (1974) "A new look at the statistical model identification",
        IEEE TAC 19(6): 716-723；
        Schwarz, G. (1978) "Estimating the dimension of a model", Annals of Statistics
        6(2): 461-464；
        Draper & Smith, "Applied Regression Analysis", 3rd ed., Ch. 6（逐步法）。
    """
    Xm = as_matrix(X, "X")
    yv = as_vector(y, "y")
    n, p = Xm.shape
    if yv.size != n:
        raise ValueError(f"X 有 {n} 行但 y 长度 {yv.size}")
    crit_name = str(criterion)
    if crit_name not in ("aic", "bic"):
        raise ValueError(f"criterion 只能是 'aic' 或 'bic'，得到 {criterion!r}")
    ms = int(max_steps)
    if ms < 0:
        raise ValueError(f"max_steps 必须 >= 0，得到 {max_steps!r}")

    def _rss(cols: Sequence[int]) -> Optional[float]:
        k = len(cols) + 1
        if n <= k:
            return None
        Xd = np.column_stack([np.ones(n, dtype=float), Xm[:, list(cols)]]) if cols else \
            np.ones((n, 1), dtype=float)
        coef, _, rank, _ = np.linalg.lstsq(Xd, yv, rcond=None)
        if rank < Xd.shape[1]:
            return None
        resid = yv - Xd @ coef
        return float(resid @ resid)

    def _criterion(cols: Sequence[int]) -> float:
        rss = _rss(cols)
        k = len(cols) + 1
        if rss is None:
            return float("inf")
        floor = max(rss, 1e-300)
        base = float(n) * math.log(floor / n)
        if crit_name == "aic":
            return base + 2.0 * k
        return base + k * math.log(float(n))

    selected: list = []
    current = _criterion(selected)
    history: list = []
    for step in range(1, ms + 1):
        best_crit = current
        best_action = None
        best_index = None
        for j in range(p):
            if j in selected:
                continue
            cand = _criterion(selected + [j])
            if cand < best_crit - 1e-12:
                best_crit = cand
                best_action = "add"
                best_index = j
        for j in list(selected):
            cand = _criterion([c for c in selected if c != j])
            if cand < best_crit - 1e-12:
                best_crit = cand
                best_action = "remove"
                best_index = j
        if best_action is None:
            break
        if best_action == "add":
            selected = selected + [best_index]
        else:
            selected = [c for c in selected if c != best_index]
        selected = sorted(selected)
        current = best_crit
        history.append(
            {
                "step": int(step),
                "action": best_action,
                "index": int(best_index),
                "criterion": float(current),
            }
        )

    if selected:
        Xd = np.column_stack([np.ones(n, dtype=float), Xm[:, selected]])
        coef, _, _, _ = np.linalg.lstsq(Xd, yv, rcond=None)
        intercept = float(coef[0])
        slopes = np.asarray(coef[1:], dtype=float)
    else:
        intercept = float(yv.mean())
        slopes = np.zeros(0, dtype=float)
    return {
        "selected": [int(j) for j in selected],
        "coef": slopes,
        "intercept": intercept,
        "criterion_value": float(current),
        "history": history,
    }


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。"""
    out: Dict[str, object] = {}

    # 0) 分布函数：与解析值对照（df=1 的 t 双侧尾概率 = 1 - (2/pi)arctan(t)）
    out["betainc_half_half_half"] = round(float(_betainc(0.5, 0.5, 0.5)), 9)
    out["t_tail_df1_at_1"] = round(float(_t_sf_two_sided(1.0, 1.0)), 9)
    out["t_tail_df1_exact"] = round(float(1.0 - 2.0 * math.atan(1.0) / math.pi), 9)
    out["chi2_sf_df2"] = round(float(_chi2_sf(4.0, 2.0)), 9)
    out["chi2_sf_df2_exact"] = round(float(math.exp(-2.0)), 9)
    out["f_sf_1_1_at_1"] = round(float(_f_sf(1.0, 1.0, 1.0)), 9)

    # 1) 相关系数：y = 2x + 1 时 Pearson 必须为 1（浮点内）
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ys = 2.0 * xs + 1.0
    pc = pearson_corr(xs, ys)
    out["pearson_perfect"] = round(float(pc["coef"]), 9)
    sc = spearman_corr(xs, ys)
    out["spearman_perfect"] = round(float(sc["coef"]), 9)
    kt = kendall_tau(xs, ys)
    out["kendall_perfect"] = round(float(kt["coef"]), 9)

    r = rng(20240101)
    xr = r.normal(size=50)
    yr = 0.6 * xr + r.normal(size=50)
    out["pearson_random_coef"] = round(float(pearson_corr(xr, yr)["coef"]), 6)
    out["pearson_random_p"] = round(float(pearson_corr(xr, yr)["p_value"]), 6)
    out["spearman_random_coef"] = round(float(spearman_corr(xr, yr)["coef"]), 6)
    out["kendall_random_coef"] = round(float(kendall_tau(xr, yr)["coef"]), 6)
    out["kendall_random_p"] = round(float(kendall_tau(xr, yr)["p_value"]), 6)

    # 2) t 检验：两组均值完全相同 -> 统计量 0、p 值 1
    t0 = t_test_two_sample([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    out["ttest_identical_stat"] = round(float(t0["stat"]), 9)
    out["ttest_identical_p"] = round(float(t0["p_value"]), 9)
    t1 = t_test_one_sample([1.0, 2.0, 3.0, 4.0, 5.0], mu=3.0)
    out["ttest1_stat"] = round(float(t1["stat"]), 6)
    out["ttest1_df"] = round(float(t1["df"]), 6)
    out["ttest1_p"] = round(float(t1["p_value"]), 6)
    t2 = t_test_two_sample([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], equal_var=True)
    t2w = t_test_two_sample([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], equal_var=False)
    out["ttest2_stat"] = round(float(t2["stat"]), 6)
    out["ttest2_df"] = round(float(t2["df"]), 6)
    out["ttest2_p"] = round(float(t2["p_value"]), 6)
    out["ttest2_welch_df"] = round(float(t2w["df"]), 6)
    out["ttest2_welch_stat"] = round(float(t2w["stat"]), 6)
    out["ttest2_welch_p"] = round(float(t2w["p_value"]), 6)

    # 3) 卡方：完全独立的 2x2 表统计量必须为 0
    obs_indep = [[10.0, 20.0], [20.0, 40.0]]
    c0 = chi_square_test(obs_indep)
    out["chi2_indep_stat"] = round(float(c0["stat"]), 9)
    out["chi2_indep_df"] = round(float(c0["df"]), 6)
    obs_dep = [[20.0, 10.0], [10.0, 20.0]]
    c1 = chi_square_test(obs_dep)
    out["chi2_dep_stat"] = round(float(c1["stat"]), 6)
    out["chi2_dep_p"] = round(float(c1["p_value"]), 6)
    out["chi2_expected_shape"] = [int(v) for v in np.asarray(c1["expected"]).shape]
    c2 = chi_square_test([10.0, 20.0, 30.0, 40.0])
    out["chi2_gof_stat"] = round(float(c2["stat"]), 6)
    out["chi2_gof_df"] = round(float(c2["df"]), 6)

    # 4) shapiro_wilk 必须明确抛出 NotImplementedError（宁缺毋滥）
    try:
        shapiro_wilk([1.0, 2.0, 3.0, 4.0, 5.0])
        out["shapiro_wilk_raises"] = False
    except NotImplementedError:
        out["shapiro_wilk_raises"] = True

    # 5) Jarque-Bera：正态样本不应拒绝；极端偏斜样本必须拒绝
    r2 = rng(7)
    xn = r2.normal(size=500)
    jb_n = jarque_bera(xn)
    out["jb_normal_stat"] = round(float(jb_n["stat"]), 6)
    out["jb_normal_p"] = round(float(jb_n["p_value"]), 6)
    jb_e = jarque_bera(np.exp(r2.normal(size=500)))
    out["jb_lognormal_stat"] = round(float(jb_e["stat"]), 6)
    out["jb_lognormal_p"] = round(float(jb_e["p_value"]), 6)
    out["jb_lognormal_reject"] = bool(float(jb_e["p_value"]) < 0.01)

    # 6) Anderson-Darling：正态样本 p 值应偏大；指数型样本必须拒绝
    ad_n = anderson_darling(xn)
    out["ad_normal_stat"] = round(float(ad_n["stat"]), 6)
    out["ad_normal_p"] = round(float(ad_n["p_value"]), 6)
    ad_e = anderson_darling(r2.exponential(size=500))
    out["ad_exponential_stat"] = round(float(ad_e["stat"]), 6)
    out["ad_exponential_p"] = round(float(ad_e["p_value"]), 6)
    out["ad_exponential_reject"] = bool(float(ad_e["p_value"]) < 0.01)
    out["ad_crit5"] = float(ad_e["crit"]["5%"])

    # 7) KS：参数估计后 p 值必须明显偏大（Lilliefors 陷阱的实际体现）
    ks_est = ks_test_normal(xn)
    ks_known = ks_test_normal(xn, mu=0.0, sigma=1.0)
    out["ks_stat_estimated"] = round(float(ks_est["stat"]), 6)
    out["ks_p_estimated"] = round(float(ks_est["p_value"]), 6)
    out["ks_p_known_params"] = round(float(ks_known["p_value"]), 6)
    out["ks_estimated_p_larger"] = bool(
        float(ks_est["p_value"]) > float(ks_known["p_value"])
    )

    # 8) OLS：完全共线的设计矩阵必须报错；正常数据 R^2 应为 1
    out["ols_rejects_collinear"] = False
    try:
        ols([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], [1.0, 2.0, 3.0, 4.0])
    except ValueError:
        out["ols_rejects_collinear"] = True
    xo = np.linspace(0.0, 10.0, 21)
    yo = 3.0 - 1.5 * xo
    res = ols(xo.reshape(-1, 1), yo)
    out["ols_coef_ramp"] = [round(float(v), 9) for v in res["coef"]]
    out["ols_r2_ramp"] = round(float(res["r2"]), 9)
    out["ols_df_resid"] = int(res["df_resid"])

    r3 = rng(20240102)
    X3 = r3.normal(size=(200, 3))
    beta_true = np.array([1.5, -2.0, 0.5])
    y3 = 1.0 + X3 @ beta_true + r3.normal(scale=0.5, size=200)
    res3 = ols(X3, y3)
    out["ols_coef_random"] = [round(float(v), 6) for v in res3["coef"]]
    out["ols_r2_random"] = round(float(res3["r2"]), 6)
    out["ols_f_stat_random"] = round(float(res3["f_stat"]), 6)
    out["ols_se_all_positive"] = bool(np.all(np.asarray(res3["se"]) > 0.0))
    out["ols_sigma2_vs_truth"] = round(float(res3["sigma2"]), 6)

    # 9) VIF：正交设计与强共线设计的对比
    Xo = r3.normal(size=(300, 3))
    out["vif_orthogonal_max"] = round(float(np.max(np.asarray(vif(Xo)["vif"]))), 6)
    z = r3.normal(size=300)
    Xc = np.column_stack([z, z + 0.01 * r3.normal(size=300), r3.normal(size=300)])
    out["vif_collinear_max"] = round(float(np.max(np.asarray(vif(Xc)["vif"]))), 6)
    out["vif_collinear_gt_orthogonal"] = bool(
        float(np.max(np.asarray(vif(Xc)["vif"]))) > float(np.max(np.asarray(vif(Xo)["vif"])))
    )

    # 10) 岭回归：alpha -> 0 时应逼近 OLS 系数；alpha 变大时系数范数必须收缩
    Xr = r3.normal(size=(300, 4))
    yr = 1.0 + Xr @ np.array([2.0, -1.0, 0.5, 0.0]) + r3.normal(scale=0.5, size=300)
    ols_r = ols(Xr, yr)
    ridge_small = ridge_regression(Xr, yr, 1e-10)
    ridge_big = ridge_regression(Xr, yr, 100.0)
    out["ridge_vs_ols_max_diff"] = round(
        float(np.max(np.abs(np.asarray(ridge_small["coef"]) - np.asarray(ols_r["coef"][1:])))), 6
    )
    out["ridge_norm_alpha100"] = round(float(np.linalg.norm(ridge_big["coef"])), 6)
    out["ridge_norm_alpha1e-10"] = round(float(np.linalg.norm(ridge_small["coef"])), 6)
    out["ridge_shrinks"] = bool(
        float(np.linalg.norm(ridge_big["coef"])) < float(np.linalg.norm(ridge_small["coef"]))
    )

    # 11) logistic：可分数据应收敛且系数符号正确；完全分离必须返回 converged=False
    # 注意：标签必须由 Bernoulli 抽样得到。若写成 1[sigmoid(线性项) > 0.5]，等于把标签
    # 取成线性边界的符号，数据是"完全可分"的，MLE 根本不存在，收敛标志必然为 False。
    r4 = rng(11)
    Xl = r4.normal(size=(800, 2))
    eta = 0.8 + 2.0 * Xl[:, 0] - 1.0 * Xl[:, 1]
    yl = (r4.random(800) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    lg = logistic_regression(Xl, yl)
    out["logit_converged"] = bool(lg["converged"])
    out["logit_iterations"] = int(lg["iterations"])
    out["logit_coef"] = [round(float(v), 6) for v in lg["coef"]]
    out["logit_log_lik"] = round(float(lg["log_lik"]), 6)
    out["logit_coef_sign_ok"] = bool(
        float(lg["coef"][1]) > 0.0 and float(lg["coef"][2]) < 0.0
    )
    out["logit_prob_in_range"] = bool(
        np.all(np.asarray(lg["prob"]) >= 0.0) and np.all(np.asarray(lg["prob"]) <= 1.0)
    )
    # 完全分离：x>0 全为 1，x<0 全为 0
    xs_sep = np.concatenate([np.linspace(-3.0, -0.1, 30), np.linspace(0.1, 3.0, 30)])
    ys_sep = (xs_sep > 0).astype(float)
    lg_sep = logistic_regression(xs_sep.reshape(-1, 1), ys_sep)
    out["logit_separation_converged"] = bool(lg_sep["converged"])
    out["logit_separation_prob_finite"] = bool(np.all(np.isfinite(np.asarray(lg_sep["prob"]))))

    # 12) Bootstrap：对称统计量的区间应包含样本均值，且随置信度升高而变宽
    xb = rng(99).normal(loc=5.0, scale=1.0, size=200)
    b95 = bootstrap_ci(xb, n_boot=2000, alpha=0.05, seed=20240101)
    b99 = bootstrap_ci(xb, n_boot=2000, alpha=0.01, seed=20240101)
    out["boot_mean"] = round(float(b95["estimate"]), 6)
    out["boot_ci95"] = [round(float(b95["ci_low"]), 6), round(float(b95["ci_high"]), 6)]
    out["boot_ci99_width_ge_ci95"] = bool(
        (float(b99["ci_high"]) - float(b99["ci_low"]))
        >= (float(b95["ci_high"]) - float(b95["ci_low"]))
    )
    out["boot_contains_mean"] = bool(
        float(b95["ci_low"]) <= float(b95["estimate"]) <= float(b95["ci_high"])
    )
    out["boot_deterministic"] = bool(
        abs(float(bootstrap_ci(xb, n_boot=500, seed=1)["ci_low"])
            - float(bootstrap_ci(xb, n_boot=500, seed=1)["ci_low"])) < 1e-12
    )

    # 13) 置换检验：两组同分布 p 值应偏大；均值差很大时 p 值必须很小
    r5 = rng(5)
    pa = r5.normal(size=60)
    pb = r5.normal(size=60)
    pt_null = permutation_test(pa, pb, n_perm=2000, seed=20240101)
    out["perm_null_p"] = round(float(pt_null["p_value"]), 6)
    out["perm_null_stat"] = round(float(pt_null["stat"]), 6)
    pt_alt = permutation_test(pa, pa + 2.0, n_perm=2000, seed=20240101)
    out["perm_alt_p"] = round(float(pt_alt["p_value"]), 6)
    out["perm_alt_reject"] = bool(float(pt_alt["p_value"]) < 0.01)

    # 14) PCA：与 np.linalg.svd 独立对拍 + 正交性 + 解释率和为 1 + 共线时第二特征值 ~ 0
    r6 = rng(20240301)
    n_p = 400
    zc = r6.normal(size=n_p)
    Xp = np.column_stack(
        [zc, zc + 0.01 * r6.normal(size=n_p), zc + 0.01 * r6.normal(size=n_p)]
    )
    pc = pca(Xp)
    comp = np.asarray(pc["components"])
    ev = np.asarray(pc["eigenvalues"])
    ratio = np.asarray(pc["explained_variance_ratio"])
    Zp = Xp - Xp.mean(axis=0)
    Zp = Zp / Zp.std(axis=0, ddof=0)
    sv = np.linalg.svd(Zp, compute_uv=False)
    ev_svd = sv ** 2 / float(n_p - 1)
    pca_svd_diff = float(np.max(np.abs(ev - ev_svd[: ev.size])))
    if pca_svd_diff > 1e-8:
        raise AssertionError(f"pca 特征值 {ev} 与 np.linalg.svd 对拍差 {pca_svd_diff}")
    pca_orth = float(np.max(np.abs(comp @ comp.T - np.eye(comp.shape[0]))))
    if pca_orth > 1e-8:
        raise AssertionError(f"pca 主成分不正交：components @ components.T 最大偏差 {pca_orth}")
    if abs(float(ratio.sum()) - 1.0) > 1e-9:
        raise AssertionError(f"pca 解释率之和 {float(ratio.sum())} != 1")
    if float(ratio[0]) <= 0.95:
        raise AssertionError(f"强相关数据第一主成分解释率 {float(ratio[0])} 应 > 0.95")
    # 两列完全线性相关：协方差阵秩 1，第二个特征值必须 ~ 0
    pc_coll = pca(np.column_stack([zc, 2.0 * zc]))
    ev_coll = np.asarray(pc_coll["eigenvalues"])
    if abs(float(ev_coll[1])) > 1e-10:
        raise AssertionError(f"完全共线时第二特征值应为 0，得到 {float(ev_coll[1])}")
    out["pca_svd_max_diff"] = round(pca_svd_diff, 9)
    out["pca_orthogonality_max_diff"] = round(pca_orth, 9)
    out["pca_ratio_sum"] = round(float(ratio.sum()), 9)
    out["pca_first_ratio_correlated"] = round(float(ratio[0]), 6)
    out["pca_collinear_second_eig"] = round(float(ev_coll[1]), 12)
    out["pca_eigenvalues"] = [round(float(v), 6) for v in ev]
    out["pca_cumulative_last"] = round(float(np.asarray(pc["cumulative_ratio"])[-1]), 9)
    out["pca_scores_shape"] = [int(v) for v in np.asarray(pc["scores"]).shape]

    # 15) 因子分析：单因子模型的共性方差与相关阵非对角元必须能被载荷重构；
    #     正交旋转不改变共性方差（旋转前后对拍）
    r7 = rng(20240302)
    n_fa = 400
    f_latent = r7.normal(size=(n_fa, 1))
    lam_true = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    Xfa = f_latent * lam_true[None, :] + r7.normal(size=(n_fa, 5)) * np.sqrt(
        1.0 - lam_true ** 2
    )[None, :]
    fa1 = factor_analysis(Xfa, n_factors=1)
    h2_hat = np.asarray(fa1["communalities"])
    uniq_hat = np.asarray(fa1["uniqueness"])
    fa_comm_err = float(np.max(np.abs(h2_hat - lam_true ** 2)))
    if fa_comm_err > 0.1:
        raise AssertionError(f"单因子共性方差恢复误差 {fa_comm_err} 过大：{h2_hat}")
    if float(np.max(np.abs(uniq_hat - (1.0 - h2_hat)))) > 1e-12:
        raise AssertionError("factor_analysis 的 uniqueness 必须等于 1 - communalities")
    Lfa = np.asarray(fa1["loadings"])
    Zfa = Xfa - Xfa.mean(axis=0)
    Zfa = Zfa / Zfa.std(axis=0, ddof=0)
    Rfa = Zfa.T @ Zfa / float(n_fa - 1)
    off_mask = ~np.eye(5, dtype=bool)
    fa_off_err = float(np.max(np.abs((Lfa @ Lfa.T - Rfa)[off_mask])))
    if fa_off_err > 0.1:
        raise AssertionError(f"载荷重构相关阵非对角元的误差 {fa_off_err} 过大")
    fa_rot = factor_analysis(Xfa, n_factors=2, rotate=True)
    fa_unrot = factor_analysis(Xfa, n_factors=2, rotate=False)
    fa_rot_diff = float(
        np.max(np.abs(np.asarray(fa_rot["communalities"]) - np.asarray(fa_unrot["communalities"])))
    )
    if fa_rot_diff > 1e-8:
        raise AssertionError(f"varimax 正交旋转不应改变共性方差，实际差 {fa_rot_diff}")
    out["fa_n_iter"] = int(fa1["n_iter"])
    out["fa_communalities"] = [round(float(v), 6) for v in h2_hat]
    out["fa_communality_max_err"] = round(fa_comm_err, 6)
    out["fa_offdiag_max_err"] = round(fa_off_err, 6)
    out["fa_variance_explained"] = [
        round(float(v), 6) for v in np.asarray(fa_rot["variance_explained"])
    ]
    out["fa_rotation_communality_diff"] = round(fa_rot_diff, 12)
    out["fa_uniqueness_min"] = round(float(np.min(uniq_hat)), 6)

    # 16) Lasso：alpha 极大 -> 全零；alpha -> 0 -> 逼近 ols；KKT 条件必须满足
    r8 = rng(20240303)
    n_l = 200
    Xla = r8.normal(size=(n_l, 4))
    b_true_l = np.array([2.0, -1.0, 0.5, 0.0])
    yla = 1.0 + Xla @ b_true_l + r8.normal(scale=0.5, size=n_l)
    ols_l = ols(Xla, yla)
    lasso_big = lasso_regression(Xla, yla, alpha=1e3)
    lasso_small = lasso_regression(Xla, yla, alpha=1e-8)
    lasso_mid = lasso_regression(Xla, yla, alpha=0.05)
    if int(lasso_big["n_nonzero"]) != 0:
        raise AssertionError(f"alpha=1e3 时 Lasso 应全为 0，得到 {int(lasso_big['n_nonzero'])} 个非零")
    lasso_vs_ols = float(
        np.max(np.abs(np.asarray(lasso_small["coef"]) - np.asarray(ols_l["coef"][1:])))
    )
    if lasso_vs_ols > 1e-3:
        raise AssertionError(f"alpha->0 时 Lasso 与 OLS 系数差 {lasso_vs_ols} 超过 1e-3")
    resid_mid = yla - Xla @ np.asarray(lasso_mid["coef"]) - float(lasso_mid["intercept"])
    lasso_kkt = float(np.max(np.abs(Xla.T @ resid_mid)) / n_l)
    if lasso_kkt > 0.05 + 1e-6:
        raise AssertionError(f"Lasso KKT 条件被违反：max|x_j'r|/n = {lasso_kkt} > alpha=0.05")
    null_obj = float(np.sum((yla - yla.mean()) ** 2) / (2.0 * n_l))
    if float(lasso_mid["objective"]) > null_obj:
        raise AssertionError("Lasso 目标值不应高于全零模型的目标值")
    # converged 标志必须双向可用：默认参数下这三组都应真收敛；
    # 而把 max_iter 压到 1、tol 收得极严时，必须如实报 False（不能永远 True）。
    for name_l, res_l in (("big", lasso_big), ("small", lasso_small), ("mid", lasso_mid)):
        if not bool(res_l["converged"]):
            raise AssertionError(
                f"Lasso(alpha={name_l}) 默认 max_iter/tol 下应已收敛，"
                f"却在 {int(res_l['n_iter'])} 轮被截断"
            )
    lasso_starved = lasso_regression(Xla, yla, alpha=0.05, max_iter=1, tol=1e-14)
    if bool(lasso_starved["converged"]):
        raise AssertionError("max_iter=1 且 tol=1e-14 时 Lasso 不可能收敛，converged 却为 True")
    if int(lasso_starved["n_iter"]) != 1:
        raise AssertionError(f"被截断时 n_iter 应等于 max_iter=1，得到 {int(lasso_starved['n_iter'])}")
    out["lasso_converged_all_default"] = True
    out["lasso_starved_converged"] = bool(lasso_starved["converged"])
    out["lasso_n_nonzero_big"] = int(lasso_big["n_nonzero"])
    out["lasso_intercept_big"] = round(float(lasso_big["intercept"]), 6)
    out["lasso_vs_ols_max_diff"] = round(lasso_vs_ols, 6)
    out["lasso_coef_small"] = [round(float(v), 6) for v in np.asarray(lasso_small["coef"])]
    out["lasso_n_nonzero_mid"] = int(lasso_mid["n_nonzero"])
    out["lasso_kkt_max"] = round(lasso_kkt, 6)
    out["lasso_n_iter"] = int(lasso_small["n_iter"])

    # 17) Poisson 回归：已知真参数的计数数据上系数恢复；偏差 = 2(饱和对数似然 - 模型对数似然)
    r9 = rng(20240304)
    n_po = 1000
    Xpo = r9.normal(size=(n_po, 2))
    b1_true, b2_true = 0.5, -0.3
    mu_po = np.exp(0.4 + b1_true * Xpo[:, 0] + b2_true * Xpo[:, 1])
    ypo = r9.poisson(mu_po).astype(float)
    pois = poisson_regression(Xpo, ypo)
    pois_err = float(np.max(np.abs(np.asarray(pois["coef"]) - np.array([b1_true, b2_true]))))
    if pois_err > 0.1:
        raise AssertionError(f"Poisson 回归系数恢复误差 {pois_err} 超过 0.1：{pois['coef']}")
    sat_ll = float(
        np.sum(ypo * np.log(np.where(ypo > 0.0, ypo, 1.0)) - ypo)
        - np.sum([math.lgamma(float(v) + 1.0) for v in ypo])
    )
    pois_gap = abs(float(pois["deviance"]) - 2.0 * (sat_ll - float(pois["loglik"])))
    if pois_gap > 1e-6:
        raise AssertionError(f"偏差与对数似然的恒等式差 {pois_gap} 过大")
    if abs(float(pois["aic"]) - (-2.0 * float(pois["loglik"]) + 2.0 * 3.0)) > 1e-9:
        raise AssertionError("Poisson 的 AIC 必须等于 -2*loglik + 2*(p+1)")
    out["poisson_coef"] = [round(float(v), 6) for v in np.asarray(pois["coef"])]
    out["poisson_intercept"] = round(float(pois["intercept"]), 6)
    out["poisson_coef_max_err"] = round(pois_err, 6)
    out["poisson_deviance"] = round(float(pois["deviance"]), 6)
    out["poisson_loglik"] = round(float(pois["loglik"]), 6)
    out["poisson_deviance_loglik_gap"] = round(pois_gap, 12)
    out["poisson_n_iter"] = int(pois["n_iter"])

    # 18) Durbin-Watson：iid 残差 ~ 2；AR(1) 正自相关残差必须显著小于 2；对整体变号不变
    r10 = rng(20240305)
    n_dw = 300
    e_iid = r10.normal(size=n_dw)
    e_ar = np.zeros(n_dw, dtype=float)
    eps_ar = r10.normal(size=n_dw)
    for t in range(1, n_dw):
        e_ar[t] = 0.9 * e_ar[t - 1] + eps_ar[t]
    dw_iid = durbin_watson(e_iid)
    dw_ar = durbin_watson(e_ar)
    if not (1.5 < dw_iid < 2.5):
        raise AssertionError(f"iid 残差的 DW={dw_iid} 应接近 2")
    if not (dw_ar < 1.0):
        raise AssertionError(f"AR(1) 正自相关残差的 DW={dw_ar} 应显著小于 2")
    if abs(durbin_watson(-e_ar) - dw_ar) > 1e-12:
        raise AssertionError("Durbin-Watson 对残差整体变号应保持不变")
    out["dw_iid"] = round(float(dw_iid), 6)
    out["dw_ar1_rho09"] = round(float(dw_ar), 6)
    out["dw_ar1_lt_2"] = bool(dw_ar < 2.0)
    out["dw_sign_invariant"] = bool(abs(durbin_watson(-e_ar) - dw_ar) < 1e-12)

    # 19) Breusch-Pagan：同方差残差不拒绝；方差随 x 指数增长时必须拒绝
    r11 = rng(20240306)
    n_bp = 400
    Xbp = r11.normal(size=(n_bp, 3))
    e_homo = r11.normal(size=n_bp)
    bp_homo = breusch_pagan(e_homo, Xbp)
    e_het = r11.normal(size=n_bp) * np.exp(0.5 * Xbp[:, 1])
    bp_het = breusch_pagan(e_het, Xbp)
    if not (float(bp_homo["p_value"]) > 0.05):
        raise AssertionError(f"同方差残差的 BP p 值 {bp_homo['p_value']} 应 > 0.05")
    if not (float(bp_het["p_value"]) < 0.01):
        raise AssertionError(f"异方差残差的 BP p 值 {bp_het['p_value']} 应 < 0.01")
    if float(bp_het["stat"]) <= float(bp_homo["stat"]):
        raise AssertionError("异方差数据的 BP 统计量必须大于同方差数据")
    out["bp_homo_stat"] = round(float(bp_homo["stat"]), 6)
    out["bp_homo_p"] = round(float(bp_homo["p_value"]), 6)
    out["bp_het_stat"] = round(float(bp_het["stat"]), 6)
    out["bp_het_p"] = round(float(bp_het["p_value"]), 6)
    out["bp_df"] = round(float(bp_het["df"]), 6)

    # 20) 逐步回归：只有第 1、3 列（0 基）真实相关时，BIC 准则下必须恰好选中它们；
    #     AIC 准则惩罚更轻，允许纳入边缘变量，但真实列必须在内且准则值不劣于全模型
    r12 = rng(20240307)
    n_sw = 300
    Xsw = r12.normal(size=(n_sw, 5))
    b_sw = np.array([0.0, 4.0, 0.0, -3.0, 0.0])
    ysw = 1.5 + Xsw @ b_sw + r12.normal(scale=1.0, size=n_sw)
    step = stepwise_selection(Xsw, ysw, criterion="bic")
    sel = [int(v) for v in step["selected"]]
    if sel != [1, 3]:
        raise AssertionError(f"BIC 逐步回归应选中 [1, 3]，实际选中 {sel}")
    if abs(float(step["coef"][0]) - 4.0) > 0.3 or abs(float(step["coef"][1]) + 3.0) > 0.3:
        raise AssertionError(f"逐步回归系数偏离真值：{step['coef']}")
    step_aic = stepwise_selection(Xsw, ysw, criterion="aic")
    sel_aic = [int(v) for v in step_aic["selected"]]
    if not set([1, 3]).issubset(set(sel_aic)):
        raise AssertionError(f"AIC 逐步回归漏掉了真实相关列，实际选中 {sel_aic}")
    ols_sw = ols(Xsw, ysw)
    rss_full = float(np.sum(np.asarray(ols_sw["resid"]) ** 2))
    aic_full = float(n_sw) * math.log(rss_full / n_sw) + 2.0 * 6.0
    if float(step_aic["criterion_value"]) > aic_full + 1e-9:
        raise AssertionError(
            f"逐步回归 AIC {step_aic['criterion_value']} 不应高于全模型 AIC {aic_full}"
        )
    if float(step["criterion_value"]) > float(step_aic["criterion_value"]) + 1e-9:
        # 本例中 BIC 选中的子模型同时也是 AIC 最优子模型；仅记录，不做强制断言
        pass
    out["step_selected"] = sel
    out["step_selected_aic"] = sel_aic
    out["step_coef"] = [round(float(v), 6) for v in np.asarray(step["coef"])]
    out["step_intercept"] = round(float(step["intercept"]), 6)
    out["step_criterion_value"] = round(float(step["criterion_value"]), 6)
    out["step_aic_value"] = round(float(step_aic["criterion_value"]), 6)
    out["step_aic_full"] = round(aic_full, 6)
    out["step_n_steps"] = int(len(step["history"]))
    out["step_first_action"] = str(step["history"][0]["action"]) if step["history"] else ""

    # 21) Mann-Whitney U：小样本手算。x=[1,2,3]、y=[4,5] 时 x 的秩和 = 1+2+3 = 6、
    #     U1 = 6 - 3*4/2 = 0、U2 = 3*2 - 0 = 6；交换两组只改 z 的符号、p 值不变。
    #     单侧 p 要按"连续性修正朝各自尾部收缩"算，所以**不是**双侧 p 的一半：
    #     双侧 = min(1, 2*min(p_less, p_greater))，本例两个单侧各自独立自洽。
    mw = mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0])
    if abs(float(mw["u_statistic"]) - 0.0) > 1e-12:
        raise AssertionError(f"Mann-Whitney U 应为 0，实际 {mw['u_statistic']}")
    if abs(float(mw["u1"]) - 0.0) > 1e-12 or abs(float(mw["u2"]) - 6.0) > 1e-12:
        raise AssertionError(f"U1/U2 应为 0/6，实际 {mw['u1']}/{mw['u2']}")
    if abs(float(mw["rank_sum"]) - 6.0) > 1e-12:
        raise AssertionError(f"x 的秩和应为 6，实际 {mw['rank_sum']}")
    if abs(float(mw["u1"]) + float(mw["u2"]) - 6.0) > 1e-12:
        raise AssertionError("U1 + U2 必须等于 n1*n2 = 6")
    if abs(float(mw["z"]) + 1.4433756729740643) > 1e-9:
        raise AssertionError(f"z 应为 -1.443375673，实际 {mw['z']}")
    if abs(float(mw["p_value"]) - 0.14891467317876572) > 1e-9:
        raise AssertionError(f"双侧 p 应为 0.148914673，实际 {mw['p_value']}")
    mw_swap = mann_whitney_u([4.0, 5.0], [1.0, 2.0, 3.0])
    if abs(float(mw_swap["u1"]) - 6.0) > 1e-12:
        raise AssertionError(f"交换两组后 U1 应为 6，实际 {mw_swap['u1']}")
    if abs(float(mw_swap["z"]) + float(mw["z"])) > 1e-9:
        raise AssertionError(f"交换两组后 z 应变号，实际 {mw_swap['z']} vs {mw['z']}")
    if abs(float(mw_swap["p_value"]) - float(mw["p_value"])) > 1e-9:
        raise AssertionError("交换两组后双侧 p 值必须不变")
    mw_g = mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0], alternative="greater")
    mw_l = mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0], alternative="less")
    # 连续性修正是"朝各自被检验的尾部"收缩的，所以开启修正后两个单侧 p **不再互补**
    # （它们有重叠，之和 > 1）；这正是它和"双侧 p / 2"不等价的原因。
    if not (float(mw_g["p_value"]) + float(mw_l["p_value"]) > 1.0):
        raise AssertionError(
            f"带连续性修正的两个单侧 p 之和应大于 1：greater={mw_g['p_value']} "
            f"less={mw_l['p_value']}"
        )
    mw_two = min(1.0, 2.0 * min(float(mw_g["p_value"]), float(mw_l["p_value"])))
    if abs(mw_two - float(mw["p_value"])) > 1e-9:
        raise AssertionError(f"双侧 p 应等于 min(1, 2*min(单侧)) = {mw_two}，实际 {mw['p_value']}")
    if abs(float(mw_g["p_value"]) - 0.9783459285946039) > 1e-9:
        raise AssertionError(f"小样本 greater p 应为 0.978345929，实际 {mw_g['p_value']}")
    if abs(float(mw_l["p_value"]) - 0.0744573365893828) > 1e-9:
        raise AssertionError(f"小样本 less p 应为 0.074457337，实际 {mw_l['p_value']}")
    # 带结的算例：秩和 = 56.5、U1 = 56.5 - 8*7/2 = 28.5 - 8 = ... 直接与手算秩对照
    mwt = mann_whitney_u([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0],
                         [2.0, 3.0, 4.0, 4.0, 6.0, 7.0])
    if abs(float(mwt["rank_sum"]) - 56.5) > 1e-9:
        raise AssertionError(f"带结秩和应为 56.5，实际 {mwt['rank_sum']}")
    if abs(float(mwt["u1"]) - 20.5) > 1e-9 or abs(float(mwt["u2"]) - 27.5) > 1e-9:
        raise AssertionError(f"带结 U1/U2 应为 20.5/27.5，实际 {mwt['u1']}/{mwt['u2']}")
    if abs(float(mwt["z"]) + 0.39162582462965073) > 1e-9:
        raise AssertionError(f"带结 z 应为 -0.391625825，实际 {mwt['z']}")
    if abs(float(mwt["p_value"]) - 0.6953347037749835) > 1e-9:
        raise AssertionError(f"带结双侧 p 应为 0.695334704，实际 {mwt['p_value']}")
    mwt_g = mann_whitney_u([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0],
                           [2.0, 3.0, 4.0, 4.0, 6.0, 7.0], alternative="greater")
    if abs(float(mwt_g["p_value"]) - 0.6992232365161767) > 1e-9:
        raise AssertionError(
            f"带结单侧 p 应朝自身尾部做连续性修正（0.699223237），实际 {mwt_g['p_value']}"
        )
    mwt_nc = mann_whitney_u([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0],
                            [2.0, 3.0, 4.0, 4.0, 6.0, 7.0], continuity=False)
    if abs(float(mwt_nc["z"]) + 0.45689679540125916) > 1e-9:
        raise AssertionError(f"不做连续性修正的 z 应为 -0.456896795，实际 {mwt_nc['z']}")
    if abs(float(mwt_nc["p_value"]) - 0.6477452274739963) > 1e-9:
        raise AssertionError(f"不做连续性修正的 p 应为 0.647745227，实际 {mwt_nc['p_value']}")
    # 关掉连续性修正后两个单侧 p 才严格互补，且双侧 = 2 * min(单侧)
    mwt_nc_g = mann_whitney_u([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0],
                              [2.0, 3.0, 4.0, 4.0, 6.0, 7.0], alternative="greater",
                              continuity=False)
    mwt_nc_l = mann_whitney_u([1.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0],
                              [2.0, 3.0, 4.0, 4.0, 6.0, 7.0], alternative="less",
                              continuity=False)
    if abs(float(mwt_nc_g["p_value"]) + float(mwt_nc_l["p_value"]) - 1.0) > 1e-9:
        raise AssertionError("不做连续性修正时两个单侧 p 必须互补")
    if abs(2.0 * min(float(mwt_nc_g["p_value"]), float(mwt_nc_l["p_value"]))
           - float(mwt_nc["p_value"])) > 1e-9:
        raise AssertionError("不做连续性修正时双侧 p 应等于较小单侧 p 的两倍")
    try:
        mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0], alternative="both")
    except ValueError:
        pass
    else:
        raise AssertionError("非法的 alternative 必须抛 ValueError")
    out["mw_u_statistic"] = round(float(mw["u_statistic"]), 9)
    out["mw_u1"] = round(float(mw["u1"]), 9)
    out["mw_u2"] = round(float(mw["u2"]), 9)
    out["mw_z"] = round(float(mw["z"]), 9)
    out["mw_p_value"] = round(float(mw["p_value"]), 9)
    out["mw_rank_sum"] = round(float(mw["rank_sum"]), 9)
    out["mw_greater_p"] = round(float(mw_g["p_value"]), 9)
    out["mw_less_p"] = round(float(mw_l["p_value"]), 9)
    out["mw_tie_u1"] = round(float(mwt["u1"]), 9)
    out["mw_tie_u2"] = round(float(mwt["u2"]), 9)
    out["mw_tie_z"] = round(float(mwt["z"]), 9)
    out["mw_tie_p_value"] = round(float(mwt["p_value"]), 9)
    out["mw_tie_rank_sum"] = round(float(mwt["rank_sum"]), 9)
    out["mw_nocont_z"] = round(float(mwt_nc["z"]), 9)
    out["mw_nocont_p_value"] = round(float(mwt_nc["p_value"]), 9)

    # 22) Wilcoxon 符号秩：手算 6 对。差值 [15,-7,5,20,0,-9]，去掉 0 差后绝对值
    #     [5,7,9,15,20] 对应秩 [1,2,3,4,5]，符号 {+,-,+,-,+}（按绝对值升序），
    #     故 W+ = 1+4+5 = 10、W- = 2+3 = 5、min(W+,W-) = 5、有效样本 n = 5。
    #     n=5 时零分布 mean = 5*6/4 = 7.5、sd = sqrt(5*6*11/24) = 3.708099，
    #     z = (10 - 7.5)/3.708099 = 0.674199862。
    ws = wilcoxon_signed_rank([125.0, 115.0, 130.0, 140.0, 140.0, 115.0],
                              [110.0, 122.0, 125.0, 120.0, 140.0, 124.0])
    if abs(float(ws["w_statistic"]) - 5.0) > 1e-12:
        raise AssertionError(f"W 应为 min(W+,W-) = 5，实际 {ws['w_statistic']}")
    if abs(float(ws["w_plus"]) - 10.0) > 1e-12 or abs(float(ws["w_minus"]) - 5.0) > 1e-12:
        raise AssertionError(f"W+/W- 应为 10/5，实际 {ws['w_plus']}/{ws['w_minus']}")
    if int(ws["n_effective"]) != 5:
        raise AssertionError(f"去掉 1 个零差后有效样本应为 5，实际 {ws['n_effective']}")
    if abs(float(ws["z"]) - 0.674199862463242) > 1e-9:
        raise AssertionError(f"z 应为 0.674199862，实际 {ws['z']}")
    if abs(float(ws["p_value"]) - 0.5001842570707945) > 1e-9:
        raise AssertionError(f"双侧 p 应为 0.500184257，实际 {ws['p_value']}")
    ws_swap = wilcoxon_signed_rank([110.0, 122.0, 125.0, 120.0, 140.0, 124.0],
                                   [125.0, 115.0, 130.0, 140.0, 140.0, 115.0])
    if abs(float(ws_swap["w_plus"]) - 5.0) > 1e-12:
        raise AssertionError(f"交换符号后 W+ 应为 5，实际 {ws_swap['w_plus']}")
    if abs(float(ws_swap["z"]) + float(ws["z"])) > 1e-9:
        raise AssertionError(f"交换符号后 z 应变号，实际 {ws_swap['z']}")
    if abs(float(ws_swap["p_value"]) - float(ws["p_value"])) > 1e-9:
        raise AssertionError("交换符号后双侧 p 必须不变（符号秩检验的对称性）")
    ws1 = wilcoxon_signed_rank([1.0, 2.0, 3.5, 4.0, 6.0, 7.0, 9.0, 11.0, 14.0])
    if abs(float(ws1["w_plus"]) - 45.0) > 1e-12 or abs(float(ws1["w_minus"]) - 0.0) > 1e-12:
        raise AssertionError(f"单样本 W+/W- 应为 45/0，实际 {ws1['w_plus']}/{ws1['w_minus']}")
    if abs(float(ws1["z"]) - 2.6655699499159153) > 1e-9:
        raise AssertionError(f"单样本 z 应为 2.66556995，实际 {ws1['z']}")
    if abs(float(ws1["p_value"]) - 0.0076857940552132725) > 1e-9:
        raise AssertionError(f"单样本双侧 p 应为 0.007685794，实际 {ws1['p_value']}")
    # W- = 0 说明样本整体偏大，证据方向是"大于 0"，所以 greater 的 p 小、less 的 p 大
    ws1_g = wilcoxon_signed_rank([1.0, 2.0, 3.5, 4.0, 6.0, 7.0, 9.0, 11.0, 14.0],
                                 alternative="greater")
    ws1_l = wilcoxon_signed_rank([1.0, 2.0, 3.5, 4.0, 6.0, 7.0, 9.0, 11.0, 14.0],
                                 alternative="less")
    if abs(float(ws1_g["p_value"]) - 0.0038428970276066362) > 1e-9:
        raise AssertionError(f"单样本单侧 greater p 应为 0.003842897，实际 {ws1_g['p_value']}")
    if float(ws1_l["p_value"]) < 0.9:
        raise AssertionError(f"单样本 less 的 p 应接近 1，实际 {ws1_l['p_value']}")
    if abs(float(ws1_g["p_value"]) + float(ws1_l["p_value"]) - 1.0) > 1e-9:
        raise AssertionError("无连续性修正时两个单侧 p 必须互补")
    if abs(2.0 * min(float(ws1_g["p_value"]), float(ws1_l["p_value"]))
           - float(ws1["p_value"])) > 1e-9:
        raise AssertionError("无连续性修正时双侧 p 应恰好等于较小单侧 p 的两倍")
    try:
        wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    except ValueError:
        pass
    else:
        raise AssertionError("配对差值全为 0 时必须抛 ValueError")
    out["wsr_w_statistic"] = round(float(ws["w_statistic"]), 9)
    out["wsr_w_plus"] = round(float(ws["w_plus"]), 9)
    out["wsr_w_minus"] = round(float(ws["w_minus"]), 9)
    out["wsr_z"] = round(float(ws["z"]), 9)
    out["wsr_p_value"] = round(float(ws["p_value"]), 9)
    out["wsr_n_effective"] = int(ws["n_effective"])
    out["wsr_one_w_plus"] = round(float(ws1["w_plus"]), 9)
    out["wsr_one_z"] = round(float(ws1["z"]), 9)
    out["wsr_one_p_value"] = round(float(ws1["p_value"]), 9)
    out["wsr_one_greater_p"] = round(float(ws1_g["p_value"]), 9)
    out["wsr_one_less_p"] = round(float(ws1_l["p_value"]), 9)

    # 23) Kruskal-Wallis：两组时应与不做连续性修正的 Mann-Whitney 满足 H = z^2；
    #     三组带结算例与 scipy.stats.kruskal 对拍过（H=7.874100719、p=0.019505665）
    kwn = kruskal_wallis([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
    if abs(float(kwn["h_statistic"]) - 1.0 / 3.0) > 1e-12:
        raise AssertionError(f"两组 H 应为 1/3，实际 {kwn['h_statistic']}")
    if abs(float(kwn["df"]) - 1.0) > 1e-12:
        raise AssertionError(f"两组 H 的 df 应为 1，实际 {kwn['df']}")
    if abs(float(kwn["p_value"]) - 0.563702861650773) > 1e-9:
        raise AssertionError(f"两组 H 的 p 应为 0.563702862，实际 {kwn['p_value']}")
    if abs(float(kwn["tie_correction"]) - 1.0) > 1e-12:
        raise AssertionError(f"无结时修正因子应为 1，实际 {kwn['tie_correction']}")
    if int(kwn["n_groups"]) != 2 or int(kwn["n_total"]) != 8:
        raise AssertionError("两组算例的 n_groups/n_total 应为 2/8")
    mw_nc2 = mann_whitney_u([1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0], continuity=False)
    if abs(float(kwn["h_statistic"]) - float(mw_nc2["z"]) ** 2) > 1e-6:
        raise AssertionError(
            f"两组时 H 应等于 MWU 的 z^2：{kwn['h_statistic']} vs {float(mw_nc2['z']) ** 2}"
        )
    kw_id = kruskal_wallis([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    if abs(float(kw_id["h_statistic"])) > 1e-12 or abs(float(kw_id["p_value"]) - 1.0) > 1e-12:
        raise AssertionError(f"两组同分布时应得 H=0、p=1，实际 {kw_id['h_statistic']}/{kw_id['p_value']}")
    kw3 = kruskal_wallis([[1.0, 2.0, 2.0, 4.0], [2.0, 3.0, 5.0, 5.0], [5.0, 6.0, 7.0, 9.0]])
    if abs(float(kw3["h_statistic"]) - 7.874100719424467) > 1e-9:
        raise AssertionError(f"三组带结 H 应为 7.874100719，实际 {kw3['h_statistic']}")
    if abs(float(kw3["p_value"]) - 0.019505664669776344) > 1e-9:
        raise AssertionError(f"三组带结 p 应为 0.019505665，实际 {kw3['p_value']}")
    if abs(float(kw3["tie_correction"]) - 0.972027972027972) > 1e-9:
        raise AssertionError(f"三组带结修正因子应为 0.972027972，实际 {kw3['tie_correction']}")
    if [round(float(v), 9) for v in np.asarray(kw3["rank_sums"])] != [13.0, 24.0, 41.0]:
        raise AssertionError(f"三组秩和应为 [13, 24, 41]，实际 {kw3['rank_sums']}")
    # 全部观测同值 -> 修正因子 C = 0，H 的分母为 0；本实现直接返回 H=0、p=1
    kw_flat = kruskal_wallis([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    if abs(float(kw_flat["tie_correction"])) > 1e-12:
        raise AssertionError(f"全部同值时 C 应为 0，实际 {kw_flat['tie_correction']}")
    if abs(float(kw_flat["h_statistic"])) > 1e-12 or abs(float(kw_flat["p_value"]) - 1.0) > 1e-12:
        raise AssertionError("全部同值时应返回 H=0、p=1，而不是 NaN")
    try:
        kruskal_wallis([[1.0, 2.0, 3.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("只有一组时必须抛 ValueError")
    out["kw_h_statistic"] = round(float(kwn["h_statistic"]), 9)
    out["kw_df"] = round(float(kwn["df"]), 9)
    out["kw_p_value"] = round(float(kwn["p_value"]), 9)
    out["kw_tie_correction"] = round(float(kwn["tie_correction"]), 12)
    out["kw_h_over_z2"] = round(float(kwn["h_statistic"]) / (float(mw_nc2["z"]) ** 2), 12)
    out["kw_tie_h_statistic"] = round(float(kw3["h_statistic"]), 9)
    out["kw_tie_p_value"] = round(float(kw3["p_value"]), 9)
    out["kw_tie_correction"] = round(float(kw3["tie_correction"]), 9)
    out["kw_tie_rank_sums"] = [round(float(v), 9) for v in np.asarray(kw3["rank_sums"])]
    out["kw_flat_h_statistic"] = round(float(kw_flat["h_statistic"]), 12)
    out["kw_flat_p_value"] = round(float(kw_flat["p_value"]), 12)

    # 24) 单因素方差分析：三组手算 SS 分解 + 精确 p 值。df1=2、df2=6、F=7 时
    #     p = (1 + 2*7/6)^{-3} = (10/3)^{-3} = 0.027 恰好可用解析式独立校验。
    anv = anova_oneway([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    if abs(float(anv["f_statistic"]) - 7.0) > 1e-12:
        raise AssertionError(f"F 应为 7，实际 {anv['f_statistic']}")
    if abs(float(anv["p_value"]) - (10.0 / 3.0) ** -3) > 1e-12:
        raise AssertionError(f"p 应为 (10/3)^-3 = 0.027，实际 {anv['p_value']}")
    if abs(float(anv["ss_between"]) - 14.0) > 1e-12 or abs(float(anv["ss_within"]) - 6.0) > 1e-12:
        raise AssertionError(f"SS 组间/组内应为 14/6，实际 {anv['ss_between']}/{anv['ss_within']}")
    if abs(float(anv["ss_total"]) - 20.0) > 1e-12:
        raise AssertionError(f"SS 总应为 20，实际 {anv['ss_total']}")
    if abs(float(anv["ss_total"]) - float(anv["ss_between"]) - float(anv["ss_within"])) > 1e-12:
        raise AssertionError("必须满足 ss_total = ss_between + ss_within")
    if int(anv["df_between"]) != 2 or int(anv["df_within"]) != 6:
        raise AssertionError(f"自由度应为 2/6，实际 {anv['df_between']}/{anv['df_within']}")
    if abs(float(anv["ms_between"]) - 7.0) > 1e-12 or abs(float(anv["ms_within"]) - 1.0) > 1e-12:
        raise AssertionError("MS 组间/组内应为 7/1")
    if abs(float(anv["grand_mean"]) - 10.0 / 3.0) > 1e-12:
        raise AssertionError(f"总均值应为 10/3，实际 {anv['grand_mean']}")
    if [round(float(v), 9) for v in np.asarray(anv["group_means"])] != [2.0, 3.0, 5.0]:
        raise AssertionError(f"组均值应为 [2, 3, 5]，实际 {anv['group_means']}")
    anv2 = anova_oneway([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    tt2 = t_test_two_sample([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], equal_var=True)
    if abs(float(anv2["f_statistic"]) - float(tt2["stat"]) ** 2) > 1e-9:
        raise AssertionError(
            f"两组时 F 应等于 t^2：{anv2['f_statistic']} vs {float(tt2['stat']) ** 2}"
        )
    if abs(float(anv2["p_value"]) - float(tt2["p_value"])) > 1e-9:
        raise AssertionError("两组时 F 检验的 p 必须等于等方差双侧 t 检验的 p")
    try:
        anova_oneway([[1.0, 2.0, 3.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("只有一组时必须抛 ValueError")
    try:
        anova_oneway([[1.0], [2.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("组内自由度为 0 时必须抛 ValueError")
    out["anova_f_statistic"] = round(float(anv["f_statistic"]), 12)
    out["anova_p_value"] = round(float(anv["p_value"]), 12)
    out["anova_ss_between"] = round(float(anv["ss_between"]), 12)
    out["anova_ss_within"] = round(float(anv["ss_within"]), 12)
    out["anova_ss_total"] = round(float(anv["ss_total"]), 12)
    out["anova_ss_identity_dev"] = round(
        abs(float(anv["ss_total"]) - float(anv["ss_between"]) - float(anv["ss_within"])), 12
    )
    out["anova_df_between"] = int(anv["df_between"])
    out["anova_df_within"] = int(anv["df_within"])
    out["anova_ms_between"] = round(float(anv["ms_between"]), 12)
    out["anova_ms_within"] = round(float(anv["ms_within"]), 12)
    out["anova_grand_mean"] = round(float(anv["grand_mean"]), 12)
    out["anova_group_means"] = [round(float(v), 12) for v in np.asarray(anv["group_means"])]
    out["anova_f_over_t2"] = round(
        float(anv2["f_statistic"]) / (float(tt2["stat"]) ** 2), 12
    )
    out["anova_two_group_p"] = round(float(anv2["p_value"]), 12)

    # 25) Newey-West HAC 标准误
    #     (a) lags=0 + 等模残差（残差恰好同方差）时必须严格退回经典 OLS 标准误；
    #     (b) AR(1) 正自相关残差下标准误随截断阶数单调增大，且 t 值被显著压低。
    xnw = np.arange(1.0, 13.0)
    rnw = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
    if abs(float(rnw.sum())) > 1e-12 or abs(float((xnw * rnw).sum())) > 1e-12:
        raise AssertionError("等模残差构造必须与常数、x 都正交，否则回归不会精确还原真参数")
    ynw = 1.0 + 2.0 * xnw + rnw
    Xnw = xnw.reshape(-1, 1)
    ols_nw = ols(Xnw, ynw)
    nw0 = newey_west_se(ynw, Xnw, lags=0)
    se_ols = np.asarray(ols_nw["se"], dtype=float)
    se_nw0 = np.asarray(nw0["se"], dtype=float)
    if float(np.max(np.abs(se_nw0 - se_ols) / se_ols)) > 1e-8:
        raise AssertionError(
            f"lags=0 且残差同方差时应退回 OLS 标准误，实际 {se_nw0} vs {se_ols}"
        )
    if [round(float(v), 9) for v in np.asarray(nw0["beta"])] != [1.0, 2.0]:
        raise AssertionError(f"系数应精确还原 [1, 2]，实际 {nw0['beta']}")
    if abs(float(nw0["r_squared"]) - float(ols_nw["r2"])) > 1e-9:
        raise AssertionError("HAC 只改标准误，R^2 必须与 OLS 完全一致")
    if abs(float(np.asarray(nw0["t_stat"])[1]) - float(np.asarray(nw0["beta"])[1])
           / float(se_nw0[1])) > 1e-9:
        raise AssertionError("t 统计量必须等于 beta / se")
    rnw_g = rng(20240501)
    n_g = 400
    xg = rnw_g.normal(size=n_g)
    eg = np.zeros(n_g)
    epsg = rnw_g.normal(size=n_g)
    for ti in range(1, n_g):
        eg[ti] = 0.8 * eg[ti - 1] + epsg[ti]
    yg = 1.0 + 0.5 * xg + eg
    Xg = xg.reshape(-1, 1)
    nw_g0 = newey_west_se(yg, Xg, lags=0)
    nw_g5 = newey_west_se(yg, Xg, lags=5)
    nw_g10 = newey_west_se(yg, Xg, lags=10)
    nw_gauto = newey_west_se(yg, Xg)
    se0 = float(np.asarray(nw_g0["se"])[1])
    se5 = float(np.asarray(nw_g5["se"])[1])
    se10 = float(np.asarray(nw_g10["se"])[1])
    if not (se0 < se5 < se10):
        raise AssertionError(f"AR(1) 残差下 HAC 标准误应随 L 单调增大，实际 {se0}/{se5}/{se10}")
    if abs(se0 - 0.079604916155) > 1e-9 or abs(se5 - 0.084522882449) > 1e-9:
        raise AssertionError(f"AR(1) 算例的 lag0/lag5 标准误偏离实测值：{se0}/{se5}")
    if abs(se10 - 0.084890654111) > 1e-9:
        raise AssertionError(f"AR(1) 算例的 lag10 标准误偏离实测值：{se10}")
    if int(nw_gauto["lags"]) != 5:
        raise AssertionError(f"n=400 的经验带宽应为 floor(4*(4)^(2/9)) = 5，实际 {nw_gauto['lags']}")
    p_auto = float(np.asarray(nw_gauto["p_value"])[1])
    if not (float(np.asarray(nw_g0["p_value"])[1]) < p_auto < float(np.asarray(nw_g10["p_value"])[1])):
        raise AssertionError("自相关被 HAC 吸收后斜率 p 值应变大（t 值变小）")
    if float(np.asarray(nw_g10["p_value"])[1]) > 1e-6:
        raise AssertionError("该算例斜率在 lag=10 下仍应高度显著（p < 1e-6）")
    for bad_call, label in (
        (lambda: newey_west_se([1.0, 2.0], [[1.0], [2.0]]), "n <= k"),
        (lambda: newey_west_se(ynw, Xnw, lags=-1), "负 lags"),
        (lambda: newey_west_se(ynw, Xnw, lags=12), "lags 超过 n-1"),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"Newey-West 的非法输入（{label}）必须抛 ValueError")
    out["nw_se_lags0"] = [round(float(v), 12) for v in se_nw0]
    out["nw_se_ols"] = [round(float(v), 12) for v in se_ols]
    out["nw_lags0_max_rel_diff"] = round(
        float(np.max(np.abs(se_nw0 - se_ols) / se_ols)), 20
    )
    out["nw_beta"] = [round(float(v), 9) for v in np.asarray(nw0["beta"])]
    out["nw_r_squared"] = round(float(nw0["r_squared"]), 12)
    out["nw_ols_r_squared"] = round(float(ols_nw["r2"]), 12)
    out["nw_ar1_se_lag0"] = round(se0, 12)
    out["nw_ar1_se_lag5"] = round(se5, 12)
    out["nw_ar1_se_lag10"] = round(se10, 12)
    out["nw_ar1_auto_lags"] = int(nw_gauto["lags"])
    out["nw_ar1_se_auto"] = round(float(np.asarray(nw_gauto["se"])[1]), 12)
    out["nw_ar1_p_lag0"] = round(float(np.asarray(nw_g0["p_value"])[1]), 12)
    out["nw_ar1_p_lag10"] = round(float(np.asarray(nw_g10["p_value"])[1]), 12)
    out["nw_ar1_ols_slope_se"] = round(float(np.asarray(ols(Xg, yg)["se"])[1]), 12)

    # 26) BCa Bootstrap 区间
    #     (a) 正态样本上端点应贴近精确 t 区间（相对宽度偏差 < 15%），且必然包含点估计；
    #     (b) 偏斜统计量（方差）上偏差校正必须把区间整体往右推。
    rbca = rng(20240501)
    xb = rbca.normal(loc=5.0, scale=2.0, size=60)
    bca = bca_bootstrap_ci(xb, np.mean, n_boot=2000, alpha=0.05, seed=20240501)
    if int(bca["n_boot"]) != 2000:
        raise AssertionError(f"n_boot 应回传 2000，实际 {bca['n_boot']}")
    if abs(float(bca["theta_hat"]) - float(np.mean(xb))) > 1e-12:
        raise AssertionError("theta_hat 必须是原样本上的统计量")
    if abs(float(bca["theta_hat"]) - 5.034786856404088) > 1e-9:
        raise AssertionError(f"theta_hat 实测值应为 5.034786856，实际 {bca['theta_hat']}")
    if abs(float(bca["bias_correction"]) - 0.048898731212656255) > 1e-9:
        raise AssertionError(f"z0 实测值应为 0.048898731，实际 {bca['bias_correction']}")
    if abs(float(bca["acceleration"]) + 0.0016532948499605577) > 1e-9:
        raise AssertionError(f"均值统计量的加速度实测应为 -0.001653295，实际 {bca['acceleration']}")
    if abs(float(bca["acceleration"])) > 0.15:
        raise AssertionError("均值统计量的 |jackknife 加速度| 必须很小（< 0.15）")
    if abs(float(bca["ci_lower"]) - 4.6012321760118) > 1e-9:
        raise AssertionError(f"BCa 下端点实测应为 4.601232176，实际 {bca['ci_lower']}")
    if abs(float(bca["ci_upper"]) - 5.500376736823868) > 1e-9:
        raise AssertionError(f"BCa 上端点实测应为 5.500376737，实际 {bca['ci_upper']}")
    if not (float(bca["ci_lower"]) < float(bca["theta_hat"]) < float(bca["ci_upper"])):
        raise AssertionError("BCa 区间必须包含点估计")
    # 精确 t 区间：本模块没有 t 分位数函数，这里对自实现的 _t_sf_two_sided 做二分反解，
    # 保持"整条链不依赖 scipy"的约束。
    n_b = int(xb.size)
    tc_lo, tc_hi = 0.0, 40.0
    for _ in range(200):
        tc_mid = 0.5 * (tc_lo + tc_hi)
        if _t_sf_two_sided(tc_mid, float(n_b - 1)) > 0.05:
            tc_lo = tc_mid
        else:
            tc_hi = tc_mid
    tcrit = 0.5 * (tc_lo + tc_hi)
    sd_b = float(np.std(xb, ddof=1))
    half = tcrit * sd_b / math.sqrt(float(n_b))
    t_lo = float(np.mean(xb)) - half
    t_hi = float(np.mean(xb)) + half
    width = t_hi - t_lo
    d_lo = abs(float(bca["ci_lower"]) - t_lo) / width
    d_hi = abs(float(bca["ci_upper"]) - t_hi) / width
    if d_lo > 0.15 or d_hi > 0.15:
        raise AssertionError(f"BCa 端点与精确 t 区间偏差过大：{d_lo:.4f}/{d_hi:.4f}（上限 0.15）")
    if abs(tcrit - 2.000995378) > 1e-8:
        raise AssertionError(f"df=59 的 95% t 分位数应约为 2.000995378，实际 {tcrit}")
    bca_var = bca_bootstrap_ci(xb, np.var, n_boot=2000, alpha=0.05, seed=20240501)
    pct_var = bootstrap_ci(xb, np.var, n_boot=2000, alpha=0.05, seed=20240501)
    if abs(float(bca_var["acceleration"]) - 0.04720220387169089) > 1e-9:
        raise AssertionError(f"方差统计量的加速度实测应为 0.047202204，实际 {bca_var['acceleration']}")
    if abs(float(bca_var["bias_correction"]) - 0.13830420796140452) > 1e-9:
        raise AssertionError(f"方差统计量的 z0 实测应为 0.138304208，实际 {bca_var['bias_correction']}")
    if not (float(bca_var["ci_lower"]) < float(bca_var["theta_hat"]) < float(bca_var["ci_upper"])):
        raise AssertionError("方差统计量的 BCa 区间必须包含点估计")
    if not (float(bca_var["ci_lower"]) > float(pct_var["ci_low"])
            and float(bca_var["ci_upper"]) > float(pct_var["ci_high"])):
        raise AssertionError(
            f"正偏差校正应把方差区间整体右移：BCa {bca_var['ci_lower']}/{bca_var['ci_upper']} "
            f"vs 百分位法 {pct_var['ci_low']}/{pct_var['ci_high']}"
        )
    for bad_call, label in (
        (lambda: bca_bootstrap_ci([1.0, 2.0], np.mean), "n < 3"),
        (lambda: bca_bootstrap_ci(xb, np.mean, alpha=1.0), "alpha 越界"),
        (lambda: bca_bootstrap_ci(xb, np.mean, n_boot=1), "n_boot < 2"),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError(f"BCa 的非法输入（{label}）必须抛 ValueError")
    out["bca_ci_lower"] = round(float(bca["ci_lower"]), 9)
    out["bca_ci_upper"] = round(float(bca["ci_upper"]), 9)
    out["bca_theta_hat"] = round(float(bca["theta_hat"]), 9)
    out["bca_bias_correction"] = round(float(bca["bias_correction"]), 9)
    out["bca_acceleration"] = round(float(bca["acceleration"]), 9)
    out["bca_n_boot"] = int(bca["n_boot"])
    out["bca_t_lower"] = round(t_lo, 9)
    out["bca_t_upper"] = round(t_hi, 9)
    out["bca_t_crit"] = round(tcrit, 9)
    out["bca_rel_dev_lower"] = round(d_lo, 6)
    out["bca_rel_dev_upper"] = round(d_hi, 6)
    out["bca_var_ci_lower"] = round(float(bca_var["ci_lower"]), 9)
    out["bca_var_ci_upper"] = round(float(bca_var["ci_upper"]), 9)
    out["bca_var_theta_hat"] = round(float(bca_var["theta_hat"]), 9)
    out["bca_var_bias_correction"] = round(float(bca_var["bias_correction"]), 9)
    out["bca_var_acceleration"] = round(float(bca_var["acceleration"]), 9)
    out["bca_var_pct_ci_lower"] = round(float(pct_var["ci_low"]), 9)
    out["bca_var_pct_ci_upper"] = round(float(pct_var["ci_high"]), 9)

    return out
