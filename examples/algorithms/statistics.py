"""统计推断与回归：相关系数、t 检验、卡方、正态性检验、OLS/岭回归/logistic、Bootstrap 与置换检验。

本模块的共同约定
----------------
- **只依赖 numpy 与 Python 标准库**：正态/学生 t/卡方/F 分布的分位数与尾概率全部用
  连分式/级数自己实现（``_betainc``、``_gammainc_q`` 等），不 import scipy。
  这些私有函数在交付报告中与 ``scipy.special`` / ``scipy.stats`` 做过数值对比。
- **p 值一律是双侧 p 值**（除卡方、JB、AD 等本身就是单侧右尾的检验），返回字典里统一叫
  ``p_value``；论文中引用时请注明是双侧。
- **近似的地方都会写明近似**：小样本、并列值、参数由样本估计等情形下的 p 值都只是近似，
  正式论文请用 scipy / statsmodels 复核（见 ``references/github-resources.md``）。
- 随机过程显式接收 ``seed``，默认 ``DEFAULT_SEED``（``_common.rng``），不使用全局随机状态。
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
    "chi_square_test",
    "shapiro_wilk",
    "jarque_bera",
    "anderson_darling",
    "ks_test_normal",
    "ols",
    "vif",
    "ridge_regression",
    "logistic_regression",
    "bootstrap_ci",
    "permutation_test",
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
          当系数范数超过 1e6 或对数似然不再改善时停止并返回 ``converged=False``，
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
    return out
