"""灵敏度分析与数据清洗：OAT/弹性系数、Morris 筛选、Sobol 一阶与总效应、缺失值插补、异常检测。

本模块共 11 个公开函数，按用途分成两组：

- **建模后诊断（参数灵敏度）**：``oat_sensitivity``（单因子扰动曲线）、``elasticity``
  （点弹性系数）、``morris_screening``（基本效应筛选）、``sobol_first_order``
  （Saltelli 一阶指数 S1）与 ``sobol_total_effect``（Jansen 总效应指数 ST，均带置信半宽）；
- **建模前处理（数据清洗）**：插补 ``impute_mean`` / ``impute_knn`` / ``impute_regression``，
  异常检测 ``detect_outliers_zscore`` / ``detect_outliers_iqr`` / ``detect_outliers_mad``。

本模块的定位
------------
一半是**建模前处理**（缺失值插补、异常值检测），一半是**建模后诊断**（参数灵敏度）。
两件事在数学建模论文里都极易写成"调库一句话"，但评委恰恰要看的就是这里的口径：
插补用的是哪一列的信息、异常判据是几倍 MAD、敏感性是在哪个尺度上算的。
本模块给出可以逐行写进附录的"教学透明版"实现，正式解题时建议与 scipy / SALib 的结果对拍。

关键约定
--------
- 缺失值一律用 ``np.nan`` 表示。插补类函数允许输入含 nan（这是它们唯一放宽校验的地方），
  但仍拒绝 inf，并要求没有任何一行/任何一列是"全缺失"（否则该行/列的信息量为零）。
- ``fn`` 的统一接口是 ``fn(x: np.ndarray) -> float``，``x`` 是长度 d 的一维参数向量。
- ``bounds`` 形状为 (d, 2)，第 i 行是第 i 个参数的下界与上界，下界必须严格小于上界。
- 灵敏度全部按**物理单位**报告：Morris 的步长与 Sobol 的抽样都换算回真实参数尺度，
  因此对线性函数 f = Σ c_i x_i，基本效应与弹性系数能直接对上解析值。
- 随机性一律走 ``_common.rng``（Saltelli 抽样、Morris 轨迹），不使用全局随机状态；
  插补与异常检测本身不含随机过程。
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_vector, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "oat_sensitivity",
    "elasticity",
    "morris_screening",
    "sobol_first_order",
    "sobol_total_effect",
    "impute_mean",
    "impute_knn",
    "impute_regression",
    "detect_outliers_zscore",
    "detect_outliers_iqr",
    "detect_outliers_mad",
]

# --------------------------------------------------------------------------
# 私有辅助
# --------------------------------------------------------------------------

#: KNN 插补里避免除零的极小量。
_EPS = 1e-12


def _as_float_matrix(a: MatrixLike, name: str = "X") -> np.ndarray:
    """转成二维 float64 数组，**允许 NaN**（缺失值），但拒绝 inf。

    参数:
        a: 待转换对象。
        name: 出错信息里的名字。

    返回:
        np.ndarray，形状 (n_samples, n_features)。

    算法:
        与 ``_common.as_matrix`` 相同的形状校验，唯一区别是不把 NaN 当成非法值——
        插补类函数必须能"看见"缺失位置。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        一行或一列全为 NaN 在数学上无解；这里只检查形状，全缺失的检查交给调用方，
        因为不同函数对全缺失的处理不同（插补要报错，异常检测不关心）。

    参考:
        无。
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，得到 ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if np.isinf(arr).any():
        raise ValueError(f"{name} 含 inf；缺失值请用 np.nan 表示")
    return arr


def _check_bounds(bounds: MatrixLike) -> np.ndarray:
    """校验参数边界并转成 (d, 2) 的 float64 数组。

    参数:
        bounds: 形状 (d, 2) 的下界/上界。

    返回:
        np.ndarray，形状 (d, 2)。

    算法:
        检查二维、第 2 维长度为 2、全部有限、且每行下界 < 上界。

    复杂度:
        时间 O(d) / 空间 O(d)。

    陷阱:
        下界等于上界（参数被固定）会让 Sobol 的方差估计分母为 0、Morris 的步长为 0，
        这里直接报错而不是返回一堆 NaN。

    参考:
        Saltelli et al. (2010) 的抽样约定。
    """
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"bounds 必须是形状 (d, 2) 的数组，得到 {b.shape}")
    if b.shape[0] == 0:
        raise ValueError("bounds 至少要有 1 个参数")
    if not np.all(np.isfinite(b)):
        raise ValueError(f"bounds 含 NaN 或 inf：{b.tolist()}")
    if np.any(b[:, 1] <= b[:, 0]):
        raise ValueError(f"bounds 的下界必须严格小于上界，得到 {b.tolist()}")
    return b


def _eval_fn(fn: Callable[[np.ndarray], float], x: np.ndarray,
             name: str = "fn") -> float:
    """调用一次 ``fn`` 并校验返回值是有限标量。

    参数:
        fn: ``fn(x)->float``。
        x: 长度 d 的一维参数向量。
        name: 出错信息里的名字。

    返回:
        float。

    算法:
        直接调用后转 float，并检查 isfinite。

    复杂度:
        时间 O(1)（不含 fn 本身的代价）/ 空间 O(1)。

    陷阱:
        fn 返回 numpy 标量、列表或 nan 时，如果不显式转 float 并检查，
        后续的方差/比值会在很远的地方才炸掉，报错信息完全指不到问题参数。

    参考:
        无。
    """
    v = float(fn(x))
    if not math.isfinite(v):
        raise ValueError(f"{name} 在 {np.asarray(x).tolist()} 处返回了非有限值 {v}")
    return v


def _eval_fn_matrix(fn: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    """对样本矩阵逐行调用 ``fn``，返回长度 n 的响应向量。

    参数:
        fn: ``fn(x)->float``。
        X: 形状 (n, d) 的样本矩阵。

    返回:
        np.ndarray，形状 (n,)。

    算法:
        逐行调用（fn 的接口是单点标量函数，不做向量化假设），结果堆成数组。

    复杂度:
        时间 O(n·C_fn) / 空间 O(n)。

    陷阱:
        如果为了"快"而假设 fn 支持批量向量化，用户传进来的标量函数会静默广播出
        错误形状；这里坚持逐行调用。

    参考:
        无。
    """
    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        out[i] = _eval_fn(fn, X[i])
    return out


def _check_missing_layout(arr: np.ndarray, name: str = "X") -> np.ndarray:
    """返回缺失掩码，并拒绝"整行缺失"或"整列缺失"的输入。

    参数:
        arr: 已转好的二维数组（可含 NaN）。
        name: 出错信息里的名字。

    返回:
        bool 数组，形状同 arr，True 表示该位置缺失。

    算法:
        ``np.isnan(arr)`` 后检查每行、每列的缺失计数。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        整列缺失时列均值是 nan，插补结果会静默变成 nan 并污染后续所有计算；
        整行缺失时 KNN 找不到任何"非缺失维度"，距离全为 0 会退化成均值。
        两种情形都应当由调用方显式报错。

    参考:
        无。
    """
    mask = np.isnan(arr)
    if not mask.any():
        return mask
    all_missing_rows = np.flatnonzero(mask.all(axis=1))
    if all_missing_rows.size:
        raise ValueError(f"{name} 有整行缺失的行下标 {all_missing_rows.tolist()}，无法插补")
    all_missing_cols = np.flatnonzero(mask.all(axis=0))
    if all_missing_cols.size:
        raise ValueError(f"{name} 有整列缺失的列下标 {all_missing_cols.tolist()}，无法插补")
    return mask


def _design_matrix(X: np.ndarray) -> np.ndarray:
    """构造带截距的 OLS 设计矩阵 ``[1, X]``。

    参数:
        X: 形状 (n, p) 的自变量矩阵。

    返回:
        np.ndarray，形状 (n, p+1)。

    算法:
        在最左列拼一列 1。

    复杂度:
        时间 O(np) / 空间 O(np)。

    陷阱:
        回归插补必须含截距：数据未中心化时（例如量纲在 100 以上），去掉截距会让拟合
        被迫过原点，插补值系统性偏低，而且这种偏差在残差图里很容易被忽略。

    参考:
        无。
    """
    return np.column_stack([np.ones(X.shape[0], dtype=float), X])


# --------------------------------------------------------------------------
# 一次一因子（OAT）与弹性系数
# --------------------------------------------------------------------------


def oat_sensitivity(fn: Callable[[np.ndarray], float], x0: ArrayLike,
                    rel_step: float = 0.1) -> dict:
    """逐个参数做 ±rel_step 的相对扰动，给出 OAT 灵敏度表。

    参数:
        fn: ``fn(x)->float``，x 为长度 d 的参数向量。
        x0: 基准点，长度 d。
        rel_step: 相对扰动幅度，必须 > 0；参数为 0 时退化为绝对步长 rel_step。

    返回:
        dict，键为：
        ``base``            基准点函数值；
        ``table``           list，每项 ``{"index","low","high","delta_low","delta_high","sensitivity"}``；
        ``max_abs_change``  所有扰动中 |f(扰动点) - f(基准点)| 的最大值。

    算法:
        1. 对第 i 个参数，步长 ``step_i = rel_step * |x0_i|``；若 x0_i == 0 则取
           ``step_i = rel_step``（否则 low == high，斜率无定义）。
        2. ``low = x0_i - step_i``、``high = x0_i + step_i``，其余参数固定在基准值。
        3. ``delta_low = f(low) - base``、``delta_high = f(high) - base``。
        4. ``sensitivity`` 用中心差分斜率 ``(f(high) - f(low)) / (high - low)``，
           对线性函数即为该参数的系数。

    复杂度:
        时间 O(d · C_fn) / 空间 O(d)。

    陷阱:
        1. OAT 在**非可加**模型上会误导：f = x0·x1 时两个参数的斜率都依赖另一个参数的
           当前取值，换个基准点结论就变了，此时应当用 Morris/Sobol 而不是 OAT。
        2. 相对步长在参数穿越符号时（如 x0_i 很小）会让中心差分分辨率骤降；
           若参数可能取 0，请改用绝对步长版本或先做无量纲化。
        3. 这里不检查扰动点是否落在参数可行域内——OAT 是局部方法，可行性由调用方保证。

    参考:
        局部灵敏度分析的通用做法（一次一因子法）。
    """
    if rel_step <= 0:
        raise ValueError(f"rel_step 必须为正，得到 {rel_step}")
    vec = as_vector(x0, "x0")
    base = _eval_fn(fn, vec)
    d = vec.size

    table: List[dict] = []
    max_abs_change = 0.0
    for i in range(d):
        step = rel_step * abs(vec[i]) if vec[i] != 0.0 else rel_step
        low_x = vec.copy()
        high_x = vec.copy()
        low_x[i] = vec[i] - step
        high_x[i] = vec[i] + step
        f_low = _eval_fn(fn, low_x)
        f_high = _eval_fn(fn, high_x)
        delta_low = f_low - base
        delta_high = f_high - base
        slope = (f_high - f_low) / (high_x[i] - low_x[i])
        table.append({
            "index": int(i),
            "low": float(low_x[i]),
            "high": float(high_x[i]),
            "delta_low": float(delta_low),
            "delta_high": float(delta_high),
            "sensitivity": float(slope),
        })
        max_abs_change = max(max_abs_change, abs(delta_low), abs(delta_high))

    return {"base": float(base), "table": table, "max_abs_change": float(max_abs_change)}


def elasticity(fn: Callable[[np.ndarray], float], x0: ArrayLike, i: int,
               rel_step: float = 0.01) -> float:
    """第 i 个参数在基准点处的点弹性 ``(∂y/∂x_i)·(x_i/y)``，用中心差分估计。

    参数:
        fn: ``fn(x)->float``。
        x0: 基准点，长度 d。
        i: 参数下标，0 <= i < d。
        rel_step: 中心差分的相对步长，必须 > 0。

    返回:
        float，无量纲弹性（含义：x_i 变化 1% 时 y 变化百分之几）。

    算法:
        1. ``h = rel_step * |x0_i|``（x0_i == 0 时弹性本身就是 0，见陷阱）。
        2. 中心差分 ``g = (f(x + h e_i) - f(x - h e_i)) / (2h)``。
        3. ``E = g * x0_i / f(x0)``。

    复杂度:
        时间 O(C_fn) / 空间 O(d)。

    陷阱:
        1. ``x0_i == 0`` 时弹性恒为 0（乘了 x_i），但此时相对扰动无意义，本实现直接返回 0
           而不是抛错，因为"零弹性"在数学上是对的；若你要的是斜率，请用 ``oat_sensitivity``。
           **注意检查顺序**：本函数先求 ``y0 = f(x0)`` 并检查 ``|y0| <= 1e-12``，之后才轮到
           ``x0_i == 0``。因此当 ``f(x0) == 0`` 与 ``x0_i == 0`` 同时成立时，**"f(x0) 为 0"
           这条先赢**：抛 ValueError（``f(x0)=... 为 0，弹性无定义``），绝不会返回 0.0；
           只有 ``f(x0) != 0`` 时 ``x0_i == 0`` 才真正返回 0.0。
        2. ``f(x0) == 0`` 时弹性无定义（分母为 0），这里抛 ValueError——静默返回 inf
           会让排序类后续处理得出荒谬结论。
        3. 中心差分对二阶项精确、对三阶项有 O(h²) 误差；h 取太小会被浮点抵消淹没，
           rel_step=0.01 是常用的折中。

    参考:
        经济学弹性的标准定义。
    """
    if rel_step <= 0:
        raise ValueError(f"rel_step 必须为正，得到 {rel_step}")
    vec = as_vector(x0, "x0")
    d = vec.size
    if not isinstance(i, (int, np.integer)):
        raise ValueError(f"i 必须是整数下标，得到 {i!r}")
    if i < 0 or i >= d:
        raise ValueError(f"i={i} 超出参数下标范围 [0, {d - 1}]")

    y0 = _eval_fn(fn, vec)
    if abs(y0) <= _EPS:
        raise ValueError(f"f(x0)={y0} 为 0，弹性无定义")
    if vec[i] == 0.0:
        return 0.0

    h = rel_step * abs(vec[i])
    hi = vec.copy()
    lo = vec.copy()
    hi[i] = vec[i] + h
    lo[i] = vec[i] - h
    slope = (_eval_fn(fn, hi) - _eval_fn(fn, lo)) / (2.0 * h)
    return float(slope * vec[i] / y0)


# --------------------------------------------------------------------------
# Morris 筛选
# --------------------------------------------------------------------------


def morris_screening(fn: Callable[[np.ndarray], float], bounds: MatrixLike,
                     n_trajectories: int = 10, n_levels: int = 4,
                     seed: Optional[int] = None) -> dict:
    """Morris 基本效应筛选：用少量轨迹给出每个参数的 μ、μ*、σ 与排序。

    参数:
        fn: ``fn(x)->float``。
        bounds: 形状 (d, 2) 的参数上下界。
        n_trajectories: 轨迹条数，>= 1；条数越多 σ 越可靠。
        n_levels: 每个参数的网格水平数 p，>= 2（通常 4 或 8）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``mu``       长度 d 的基本效应均值（带符号，反映方向）；
        ``mu_star``  长度 d 的 |基本效应| 均值（Morris 的推荐排序指标）；
        ``sigma``    长度 d 的基本效应样本标准差（ddof=1，反映非线性/交互）；
        ``ranking``  按 ``mu_star`` 从大到小的参数下标 list。

    算法:
        1. 把参数归一化到 [0,1]：``u_i = (x_i - lo_i)/(hi_i - lo_i)``；网格步长
           ``step = 1/(p-1)``，Morris 步长 ``delta = p/(2(p-1))``。
        2. 每条轨迹：在合法基点集合 ``{0, step, ..., 1-delta}`` 上独立均匀取基点，
           独立打乱参数顺序，依次对每个参数沿 ±delta 方向移动一步。
        3. 第 i 个参数的**物理单位**基本效应为
           ``EE_i = [f(x + Δ_i e_i) - f(x)] / Δ_i``，其中 ``Δ_i = delta·(hi_i - lo_i)``。
        4. ``mu_i = mean_t EE_i^(t)``，``mu_star_i = mean_t |EE_i^(t)|``，
           ``sigma_i = std_t(EE_i^(t), ddof=1)``。

    复杂度:
        时间 O(n_trajectories · (d + 1) · C_fn) / 空间 O(n_trajectories · d)。

    陷阱:
        1. ``n_trajectories == 1`` 时样本标准差无定义：本实现返回 0 而不是 NaN，
           但一条轨迹的 σ **没有**任何统计意义，不要据此判断非线性。
        2. ``delta`` 与 p 绑定：p=4 时 delta=2/3，单步就跨越了三分之二的参数范围，
           此时的 "μ" 是**大范围平均斜率**而不是局部导数，与 OAT 的斜率不可直接比较。
        3. 基本效应在物理单位下计算，因此 bounds 的宽度会直接改变 μ 的量纲；
           比较不同参数的重要性时这恰恰是想要的（"参数变 1 个单位，输出变多少"）。

    参考:
        Morris (1991) "Factorial sampling plans for preliminary computational experiments"；
        Saltelli et al. (2008) Global Sensitivity Analysis: The Primer 第 3 章。
    """
    if n_trajectories < 1:
        raise ValueError(f"n_trajectories 必须 >= 1，得到 {n_trajectories}")
    if n_levels < 2:
        raise ValueError(f"n_levels 必须 >= 2，得到 {n_levels}")
    b = _check_bounds(bounds)
    d = b.shape[0]
    lo = b[:, 0]
    hi = b[:, 1]

    step = 1.0 / (n_levels - 1)
    delta = n_levels / (2.0 * (n_levels - 1))
    grid = np.arange(n_levels, dtype=float) * step
    base_candidates = grid[grid <= 1.0 - delta + 1e-12]
    if base_candidates.size == 0:
        raise ValueError(f"n_levels={n_levels} 太小，无法在 [0,1] 内构造 Morris 基点")

    gen = rng(seed)
    ee = np.zeros((d, n_trajectories), dtype=float)
    for t in range(n_trajectories):
        norm = np.array([base_candidates[gen.integers(base_candidates.size)]
                         for _ in range(d)], dtype=float)
        order = gen.permutation(d)
        cur_norm = norm.copy()
        f_cur = _eval_fn(fn, lo + cur_norm * (hi - lo))
        for i in order:
            i = int(i)
            if cur_norm[i] + delta <= 1.0 + 1e-12:
                direction = 1.0
            elif cur_norm[i] - delta >= -1e-12:
                direction = -1.0
            else:  # 理论上不可达：基点集合保证了 +delta 一定合法
                direction = 1.0
            nxt_norm = cur_norm.copy()
            nxt_norm[i] = cur_norm[i] + direction * delta
            f_next = _eval_fn(fn, lo + nxt_norm * (hi - lo))
            phys_step = direction * delta * (hi[i] - lo[i])
            ee[i, t] = (f_next - f_cur) / phys_step
            cur_norm = nxt_norm
            f_cur = f_next

    mu = ee.mean(axis=1)
    mu_star = np.abs(ee).mean(axis=1)
    sigma = ee.std(axis=1, ddof=1) if n_trajectories > 1 else np.zeros(d, dtype=float)
    ranking = [int(i) for i in np.argsort(-mu_star)]
    return {"mu": mu, "mu_star": mu_star, "sigma": sigma, "ranking": ranking}


# --------------------------------------------------------------------------
# Sobol（Saltelli 估计量）
# --------------------------------------------------------------------------


def _sobol_design(bounds: np.ndarray, n_samples: int,
                  seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """生成 Saltelli 估计所需的 A、B 两个独立样本矩阵（物理单位）。

    参数:
        bounds: 已校验的 (d, 2) 边界。
        n_samples: 每个矩阵的样本量 N。
        seed: 随机种子。

    返回:
        (A, B)，均为形状 (N, d) 的 np.ndarray。

    算法:
        一次抽取 ``2N·d`` 个 U(0,1) 随机数，前一半作 A、后一半作 B，
        再用 ``lo + u·(hi-lo)`` 线性映射到参数区间（对每个参数独立均匀）。

    复杂度:
        时间 O(Nd) / 空间 O(Nd)。

    陷阱:
        A 与 B 必须**独立**；如果图省事把 B 写成 A 的行置换，Saltelli 估计量会有偏，
        而且偏差方向恰好是"低估交互"，非常难在结果里看出来。

    参考:
        Saltelli (2002) "Making best use of model evaluations to compute sensitivity indices"。
    """
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    d = bounds.shape[0]
    u = rng(seed).random((n_samples, 2 * d))
    a = lo + u[:, :d] * (hi - lo)
    b = lo + u[:, d:] * (hi - lo)
    return a, b


def _sobol_common(fn: Callable[[np.ndarray], float], bounds: MatrixLike,
                  n_samples: int, seed: Optional[int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Sobol 两个估计量的公共骨架：抽样、求 fA/fB、逐维替换列。

    参数:
        fn: ``fn(x)->float``。
        bounds: (d, 2) 参数边界。
        n_samples: Saltelli 样本量 N，>= 2。
        seed: 随机种子。

    返回:
        ``(A, fA, fB, F_AB, var, n_eval)``：
        A 形状 (N,d)；fA、fB 形状 (N,)；F_AB 形状 (d, N) 为逐维替换后的响应；
        var 为总方差估计；n_eval 为函数求值次数 N·(d+2)。

    算法:
        1. 生成 A、B；``fA = fn(A)``、``fB = fn(B)``；
        2. 对每个 i，``AB_i`` = A 的第 i 列换成 B 的第 i 列，求 ``fAB_i``；
        3. 总方差用 A、B 合并样本的**无偏**估计 ``var(fA ∪ fB, ddof=1)``。

    复杂度:
        时间 O(N·d·C_fn) / 空间 O(Nd)。

    陷阱:
        方差必须用**输出**的样本方差，不能想当然地用参数方差；另外若 fn 是常数
        （方差为 0），S1/ST 的分母为 0，本函数直接抛 ValueError 而不是返回 inf。

    参考:
        Saltelli et al. (2010) Computer Physics Communications 181:259-270。
    """
    if n_samples < 2:
        raise ValueError(f"n_samples 必须 >= 2，得到 {n_samples}")
    b = _check_bounds(bounds)
    d = b.shape[0]
    a, bm = _sobol_design(b, n_samples, seed)
    f_a = _eval_fn_matrix(fn, a)
    f_b = _eval_fn_matrix(fn, bm)

    f_ab = np.empty((d, n_samples), dtype=float)
    for i in range(d):
        ab = a.copy()
        ab[:, i] = bm[:, i]
        f_ab[i] = _eval_fn_matrix(fn, ab)

    var = float(np.var(np.concatenate([f_a, f_b]), ddof=1))
    if var <= 0.0:
        raise ValueError(f"输出样本方差为 {var}，Sobol 指标无定义（fn 近似常数）")
    n_eval = n_samples * (d + 2)
    return a, f_a, f_b, f_ab, var, n_eval


def sobol_first_order(fn: Callable[[np.ndarray], float], bounds: MatrixLike,
                      n_samples: int = 512, seed: Optional[int] = None) -> dict:
    """Saltelli 估计量计算 Sobol 一阶指数 S1（各参数独立贡献的方差占比）。

    参数:
        fn: ``fn(x)->float``。
        bounds: (d, 2) 参数边界。
        n_samples: 基样本量 N（总求值次数为 N·(d+2)），>= 2。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``S1``       长度 d 的一阶指数；
        ``S1_conf``  长度 d 的 95% 置信半宽（估计量的正态近似）；
        ``n_eval``   函数求值总次数 = N·(d+2)。

    算法:
        1. 生成独立样本矩阵 A、B（各 N 行），``AB_i`` 为 A 的第 i 列换成 B 的第 i 列。
        2. 记 ``V = Var(y)``（用 A∪B 合并样本的无偏估计）。
        3. ``S1_i = mean(fB · (fAB_i - fA)) / V``（Saltelli 2010 式 (10)）。
        4. 置信半宽取 ``1.96 · std(fB·(fAB_i-fA), ddof=1) / sqrt(N) / V``，
           即把该均值估计量当近似正态处理。

    复杂度:
        时间 O(N·d·C_fn) / 空间 O(Nd)。

    陷阱:
        1. N 太小时 S1 可能为负（估计量有偏噪声），小负值**不代表**参数有害，
           应报告为"≈0"；本实现不截断，保留原始估计以示诚实。
        2. 对强交互模型（如 y = x0·x1），一阶指数之和远小于 1，这不是 bug，
           差额就是交互贡献——必须同时看 ``sobol_total_effect``。
        3. 置信区间是正态近似，且忽略了 A/B 共用同一批随机数带来的相关性，
           只用于量级判断，不要写进论文当严格区间。

    参考:
        Saltelli et al. (2010) 式 (10)；Sobol' (2001)。
    """
    _a, f_a, f_b, f_ab, var, n_eval = _sobol_common(fn, bounds, n_samples, seed)
    d = f_ab.shape[0]
    s1 = np.empty(d, dtype=float)
    conf = np.empty(d, dtype=float)
    for i in range(d):
        term = f_b * (f_ab[i] - f_a)
        s1[i] = term.mean() / var
        conf[i] = 1.96 * term.std(ddof=1) / math.sqrt(n_samples) / var
    return {"S1": s1, "S1_conf": conf, "n_eval": int(n_eval)}


def sobol_total_effect(fn: Callable[[np.ndarray], float], bounds: MatrixLike,
                       n_samples: int = 512, seed: Optional[int] = None) -> dict:
    """Jansen/Saltelli 估计量计算 Sobol 总效应指数 ST（含该参数的全部交互贡献）。

    参数:
        fn: ``fn(x)->float``。
        bounds: (d, 2) 参数边界。
        n_samples: 基样本量 N（总求值次数为 N·(d+2)），>= 2。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``ST``       长度 d 的总效应指数；
        ``ST_conf``  长度 d 的 95% 置信半宽。

    算法:
        1. 与 ``sobol_first_order`` 共用同一套 A/B/AB_i 设计（同一 seed 下抽样完全一致）。
        2. ``ST_i = mean((fA - fAB_i)²) / (2V)``（Jansen 1999 / Saltelli 2010 式 (12)）。
        3. 置信半宽 = ``1.96 · std((fA-fAB_i)², ddof=1)/sqrt(N) / (2V)``。

    复杂度:
        时间 O(N·d·C_fn) / 空间 O(Nd)。

    陷阱:
        1. 可加模型上应当有 ST_i ≈ S1_i；若差得远，说明 N 太小或 fn 里藏了交互。
        2. ST_i 之和 >= 1（等于 1 当且仅当模型可加），和明显大于 1 是估计噪声，
           不是"发现了新交互"。
        3. 与一阶指数一样，这里不添加任何截断，ST_i 也可能出现小的负值。

    参考:
        Jansen (1999)；Saltelli et al. (2010) 式 (12)。
    """
    _a, f_a, f_b, f_ab, var, n_eval = _sobol_common(fn, bounds, n_samples, seed)
    d = f_ab.shape[0]
    st = np.empty(d, dtype=float)
    conf = np.empty(d, dtype=float)
    for i in range(d):
        sq = (f_a - f_ab[i]) ** 2
        st[i] = sq.mean() / (2.0 * var)
        conf[i] = 1.96 * sq.std(ddof=1) / math.sqrt(n_samples) / (2.0 * var)
    return {"ST": st, "ST_conf": conf}


# --------------------------------------------------------------------------
# 缺失值插补
# --------------------------------------------------------------------------


def impute_mean(X: MatrixLike) -> dict:
    """按列均值插补缺失值（最简基线，其他插补方法都应优于它才有意义）。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)，缺失位置用 np.nan 表示。

    返回:
        dict，键为：
        ``X``           插补后的矩阵（副本，不改原输入）；
        ``n_imputed``   被插补的元素个数；
        ``method``      字符串 ``"mean"``。

    算法:
        1. 对每列用 ``nanmean``（忽略 nan）得到列均值。
        2. 用该均值填回该列所有 nan 位置。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        1. 均值插补会**压缩方差**（填进去的都是列中心），后续做回归/聚类时会让变量显得
           比实际更"整齐"，这是它作为基线而非推荐方法的根本原因。
        2. 完全随机的缺失（MCAR）下均值插补无偏；一旦缺失与取值相关（MNAR），
           均值插补会系统性偏移，此时必须用回归/多重插补并在论文里讨论。
        3. 整列全缺失时列均值是 nan，本实现直接报错（见 ``_check_missing_layout``）。

    参考:
        缺失数据处理的通用基线（Little & Rubin, 2002）。
    """
    arr = _as_float_matrix(X, "X")
    mask = _check_missing_layout(arr, "X")
    out = arr.copy()
    n_imputed = int(mask.sum())
    if n_imputed == 0:
        return {"X": out, "n_imputed": 0, "method": "mean"}
    col_mean = np.nanmean(arr, axis=0)
    idx_r, idx_c = np.nonzero(mask)
    out[idx_r, idx_c] = col_mean[idx_c]
    return {"X": out, "n_imputed": n_imputed, "method": "mean"}


def impute_knn(X: MatrixLike, k: int = 5) -> dict:
    """用"非缺失维度"上的欧氏距离做 KNN 加权插值。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)，缺失位置用 np.nan 表示。
        k: 邻居个数，>= 1。

    返回:
        dict，键为：
        ``X``           插补后的矩阵（副本）；
        ``n_imputed``   被插补的元素个数；
        ``method``      字符串 ``"knn"``。

    算法:
        对每个缺失位置 (i, j)：
        1. 候选集为第 j 列**已观测**的所有行 c（c != i）；
        2. 只在 i 与 c **同时观测**的维度上算欧氏距离
           ``d(i,c) = sqrt(Σ_{m∈共同观测维} (x_im - x_cm)²)``；
        3. 取距离最小的 k 个候选，权重 ``w = 1/(d + 1e-12)`` 做加权平均；
           若最小距离为 0（存在完全相同的邻居），只用这些零距离邻居的均值；
        4. 若不存在任何"有共同观测维度"的候选，退回该列均值。

    复杂度:
        时间 O(n_missing · n · d) / 空间 O(n · d)。

    陷阱:
        1. 只在共同观测维度上比距离是**稀疏数据的关键**：若改用全维度并把 nan 当 0，
           距离会被人为放大，邻居选择完全错乱。
        2. k 大于可用候选数时按实际候选数取（不报错），但在极稀疏数据上会退化成
           加权均值，插补质量与均值法无异，此时应改用 ``impute_regression``。
        3. 本实现逐元素处理，n 很大（>1e4）时明显偏慢；竞赛数据规模下可以接受。
        4. **整列缺失走不到循环里的兜底分支**：``_check_missing_layout`` 在循环之前就拒绝了
           "整列全为 nan"的输入（抛 ValueError），所以第 j 列 ``obs_j.size == 0`` 这个
           "退回该列均值"的分支实际上是**不可达的防御性代码**，保留只是为了代码自卫。
           想插补一个整列都缺失的特征，必须先删列或在调用前补上至少一个观测值。

    参考:
        Troyanskaya et al. (2001) "Missing value estimation methods for DNA microarrays"。
    """
    if k < 1:
        raise ValueError(f"k 必须 >= 1，得到 {k}")
    arr = _as_float_matrix(X, "X")
    mask = _check_missing_layout(arr, "X")
    out = arr.copy()
    n_imputed = int(mask.sum())
    if n_imputed == 0:
        return {"X": out, "n_imputed": 0, "method": "knn"}

    col_mean = np.nanmean(arr, axis=0)
    idx_r, idx_c = np.nonzero(mask)
    for i, j in zip(idx_r.tolist(), idx_c.tolist()):
        obs_j = np.flatnonzero(~mask[:, j])
        if obs_j.size == 0:
            out[i, j] = col_mean[j]
            continue
        shared = ~mask[i] & ~mask[obs_j]      # (n_cand, d) 共同观测维度
        diff = arr[obs_j] - arr[i]            # 用 nan 占位，下面按 shared 掩掉
        diff = np.where(shared, diff, 0.0)
        dist = np.sqrt((diff ** 2).sum(axis=1))
        valid = shared.any(axis=1)
        if not valid.any():
            out[i, j] = col_mean[j]
            continue
        cand = obs_j[valid]
        dist = dist[valid]
        order = np.argsort(dist, kind="stable")[:k]
        d_near = dist[order]
        v_near = arr[cand[order], j]
        if d_near[0] <= _EPS:
            zero = d_near <= _EPS
            out[i, j] = float(v_near[zero].mean())
        else:
            w = 1.0 / (d_near + _EPS)
            out[i, j] = float((w * v_near).sum() / w.sum())
    return {"X": out, "n_imputed": n_imputed, "method": "knn"}


def impute_regression(X: MatrixLike, max_iter: int = 10, tol: float = 1e-6) -> dict:
    """迭代回归插补：每列用其余列做含截距 OLS 预测缺失位置，循环至收敛。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)，缺失位置用 np.nan 表示；列数须 >= 2。
        max_iter: 最大迭代轮数，>= 1。
        tol: 收敛判据——本轮与上轮**所有被插补值**的最大绝对变化 <= tol 即认为收敛。

    返回:
        dict，键为：
        ``X``           插补后的矩阵（副本）；
        ``n_imputed``   被插补的元素个数；
        ``n_iter``      实际迭代轮数；
        ``converged``   bool，是否在 max_iter 内达到 tol。

    算法:
        1. 初始化：所有 nan 用所在列的观测均值填充。
        2. 每一轮：对每个含缺失的列 j，取该列已观测的行，构造设计矩阵
           ``[1, 其余列]``，用 ``np.linalg.lstsq`` 解 OLS，再对缺失行预测并回填。
           （其余列在本轮中已无 nan，因为初始化已填满。）
        3. 若被插补值的最大变化 <= tol 则提前结束。

    复杂度:
        时间 O(max_iter · m · n²) / 空间 O(mn)。

    陷阱:
        1. 这是**确定性迭代**而非多重插补：它给出单点估计，无法反映插补本身的不确定性，
           正式论文应报告敏感性（例如换 3 组随机初值或改用 MICE）。
        2. 迭代线性回归会把变量间关系"越描越真"：若两列高度线性相关，缺失会被填得
           过于完美，导致后续回归的 R² 虚高，务必在论文中说明。
        3. 未收敛（converged 为 False）时必须报告实际 n_iter，不能假装收敛。

    参考:
        Buck (1960) 的回归插补；van Buuren & Groothuis-Oudshoorn (2011) MICE。
    """
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，得到 {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol 必须为正，得到 {tol}")
    arr = _as_float_matrix(X, "X")
    mask = _check_missing_layout(arr, "X")
    n_imputed = int(mask.sum())
    if n_imputed == 0:
        return {"X": arr.copy(), "n_imputed": 0, "n_iter": 0, "converged": True}
    if arr.shape[1] < 2:
        raise ValueError(f"impute_regression 需要至少 2 列，得到 {arr.shape[1]}")

    filled = arr.copy()
    col_mean = np.nanmean(arr, axis=0)
    idx_r, idx_c = np.nonzero(mask)
    filled[idx_r, idx_c] = col_mean[idx_c]

    cols_with_missing = [int(j) for j in range(arr.shape[1]) if mask[:, j].any()]
    prev = filled[mask].copy()
    n_iter = 0
    converged = False
    for it in range(max_iter):
        n_iter = it + 1
        for j in cols_with_missing:
            obs = ~mask[:, j]
            others = [c for c in range(arr.shape[1]) if c != j]
            design = _design_matrix(filled[np.ix_(obs, others)])
            coef, _res, _rank, _sv = np.linalg.lstsq(design, filled[obs, j], rcond=None)
            miss = ~obs
            target = _design_matrix(filled[np.ix_(miss, others)])
            filled[miss, j] = target @ coef
        cur = filled[mask]
        delta = float(np.max(np.abs(cur - prev)))
        prev = cur
        if delta <= tol:
            converged = True
            break
    return {"X": filled, "n_imputed": n_imputed, "n_iter": n_iter, "converged": converged}


# --------------------------------------------------------------------------
# 异常检测
# --------------------------------------------------------------------------


def detect_outliers_zscore(x: ArrayLike, threshold: float = 3.0) -> dict:
    """Z 分数法：|(x - mean) / std| > threshold 判为异常。

    参数:
        x: 一维数据（不允许含 nan）。
        threshold: 判定阈值（总体标准差倍数），必须 > 0；常用 3。

    返回:
        dict，键为：
        ``index``     异常点下标 list（升序）；
        ``values``    对应的原始取值 list；
        ``threshold``  使用的阈值（原样回传，便于报告）。

    算法:
        标准化 ``z = (x - mean) / std``（标准差用**总体**口径 ddof=0），取 |z| > threshold。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. 均值和标准差本身被离群点污染：单个极端值会把 std 抬高，使自己的 |z| 被压缩。
           对 n 较小时，单点能取得的 |z| 上限约为 (n-1)/sqrt(n)——n=10 时只有 2.85，
           **永远达不到 3**，于是"10σ 离群点"在 z 分数法下反而漏检。样本量小时请改用
           MAD 或 IQR。
        2. 数据近似正态时才用 3 作为阈值；重尾分布（如收入、点击量）会大面积误报。
        3. 标准差为 0（常数序列）时 z 无定义，本实现抛 ValueError。

    参考:
        经典的 3σ 准则（Pukelsheim, 1994 对其理论依据的讨论）。
    """
    if threshold <= 0:
        raise ValueError(f"threshold 必须为正，得到 {threshold}")
    arr = as_vector(x, "x")
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std <= 0.0:
        raise ValueError(f"x 的标准差为 {std}，Z 分数无定义（序列为常数）")
    z = np.abs((arr - mean) / std)
    idx = np.flatnonzero(z > threshold)
    return {
        "index": [int(i) for i in idx],
        "values": [float(arr[i]) for i in idx],
        "threshold": float(threshold),
    }


def detect_outliers_iqr(x: ArrayLike, k: float = 1.5) -> dict:
    """IQR 箱线图法：落在 [Q1 - k·IQR, Q3 + k·IQR] 之外的点判为异常。

    参数:
        x: 一维数据（不允许含 nan）。
        k: 栅栏系数，必须 > 0；1.5 为"离群"、3.0 为"极端离群"。

    返回:
        dict，键为：
        ``index``  异常点下标 list（升序）；
        ``values`` 对应的原始取值 list；
        ``lower``  下栅栏 Q1 - k·IQR；
        ``upper``  上栅栏 Q3 + k·IQR。

    算法:
        分位数用 numpy 的线性插值口径（``np.percentile`` 默认），
        IQR = Q3 - Q1，栅栏外即为异常。

    复杂度:
        时间 O(n log n)（排序取分位数）/ 空间 O(n)。

    陷阱:
        1. IQR 对**偏态**分布仍会误报：右偏数据的上栅栏往往被压得过低，
           长尾的正值被成批标为异常。此时应先做变换或直接看业务含义。
        2. IQR 为 0 时（超过一半样本取同一值，例如大量 0 的稀疏数据）栅栏退化为
           [Q1, Q3]，任何轻微波动都会被判异常——这是最常见的 IQR 误用。
        3. 分位数口径（线性插值 vs 取序）会改变边界点判定，本模块统一用线性插值。

    参考:
        Tukey (1977) Exploratory Data Analysis。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正，得到 {k}")
    arr = as_vector(x, "x")
    q1 = float(np.percentile(arr, 25.0))
    q3 = float(np.percentile(arr, 75.0))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    idx = np.flatnonzero((arr < lower) | (arr > upper))
    return {
        "index": [int(i) for i in idx],
        "values": [float(arr[i]) for i in idx],
        "lower": float(lower),
        "upper": float(upper),
    }


def detect_outliers_mad(x: ArrayLike, threshold: float = 3.5) -> dict:
    """修正 Z 分数（MAD 法）：``0.6745·(x - median)/MAD > threshold`` 判为异常。

    参数:
        x: 一维数据（不允许含 nan）。
        threshold: 判定阈值，必须 > 0；Iglewicz & Hoaglin 建议 3.5。

    返回:
        dict，键为：
        ``index``   异常点下标 list（升序）；
        ``values``  对应的原始取值 list；
        ``scores``  全部样本的修正 Z 分数（与 x 等长）。

    算法:
        1. ``med = median(x)``，``MAD = median(|x - med|)``；
        2. 修正 Z 分数 ``M_i = 0.6745·(x_i - med)/MAD``（0.6745 是标准正态的 0.75 分位数，
           使 M_i 在正态数据下与标准 Z 分数同尺度）；
        3. ``|M_i| > threshold`` 即为异常。

    复杂度:
        时间 O(n log n)（**安全上界**：本实现只做常数次 ``np.median``，没有任何显式排序，
        所以这个界相当宽松）/ 空间 O(n)。
        ``np.median`` 内部走的是 introselect（基于 ``np.partition`` 的选择算法），**期望**
        O(n)，实测通常就是线性的；这里仍旧报 O(n log n) 只是为了不把复杂度押在具体实现细节上。

    陷阱:
        1. MAD 用中位数而非均值，抗污染能力强，但**对 n 很敏感**：n 小于约 10 时 MAD
           可能为 0（超过一半样本等于中位数），此时修正 Z 分数无定义，本实现抛 ValueError。
        2. 0.6745 这个常数只对正态分布成立；对其他分布，3.5 的阈值没有概率解释，
           只能当作经验规则。
        3. 当缺失/异常点占比接近 50% 时，中位数本身已被污染，MAD 完全失效。

    参考:
        Iglewicz & Hoaglin (1993) How to Detect and Handle Outliers。
    """
    if threshold <= 0:
        raise ValueError(f"threshold 必须为正，得到 {threshold}")
    arr = as_vector(x, "x")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0.0:
        raise ValueError(f"MAD={mad} 为 0，修正 Z 分数无定义（中位数附近样本过多）")
    scores = 0.6745 * (arr - med) / mad
    idx = np.flatnonzero(np.abs(scores) > threshold)
    return {
        "index": [int(i) for i in idx],
        "values": [float(arr[i]) for i in idx],
        "scores": scores,
    }


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict，键名以 ``sens_`` 开头，值均为 int/float/bool/list，固定种子下两次调用完全一致；
        键覆盖 OAT 斜率与最大变化、弹性、Morris 的 μ/μ*/σ/排序、Sobol 的一阶/总效应与
        求值次数、插补 RMSE、三种异常检测的下标集合。

    算法:
        全部算例都配上**解析闭式解**，而不是只看数字是否好看：
        1. ``y = 3x_0 + 2x_1`` 在 (1,1) 处 OAT 斜率应为 [3, 2]，
           ``max_abs_change = 3*0.1 = 0.3``（rel_step=0.1）；弹性算例用 ``y = x_0²·x_1``，
           解析弹性为 [2, 1]。
        2. ``morris_screening`` 对 ``y = 3x_0 + 2x_1``：基本效应恒等于系数，
           故 μ = μ* = [3, 2] 且 σ = 0（线性 ⇒ 无交互）。
        3. ``sobol_first_order`` 对 ``y = 2x_0 + x_1``（x~U(0,1) 独立）：
           Var = 4/12 + 1/12 = 5/12，S1 = [0.8, 0.2]（容差 0.05）。
           Saltelli 的乘积估计量方差偏大：n_samples=512 时抽样误差本身就有 0.14 量级，
           4096 时仍达 0.048（贴死容差），因此这里显式取 **n_samples=32768**，
           本 seed 下实际误差 0.012（约容差的 1/4）。交互算例样本量只需 2048。
           ``sobol_total_effect`` 在同一可加函数上应满足 ST ≈ S1（差 < 0.05），
           而在交互函数 ``y = x_0·x_1`` 上必须 ST > S1（解析值约 0.571 > 0.429）。
        4. 插补：把列间近似线性的 60×3 矩阵挖掉 10%（18 个格子），
           ``impute_regression`` 的 RMSE 必须小于 ``impute_mean`` 的 RMSE。
        5. 异常检测：31 个点的等差序列 + 一个远端离群点，三种方法必须抓到同一下标。
        Sobol 的样本量按容差要求取值（见下方常量），已在报告里说明。

    复杂度:
        时间 O(n_samples·d)（Sobol 占主要部分）/ 空间 O(n_samples·d)。

    陷阱:
        1. Sobol 断言对随机数敏感：样本量太小时 S1 的抽样误差会超过 0.05 容差，
           因此这里显式放大 n_samples 而不是放宽容差；改动 seed 后必须重跑。
        2. Morris 的 σ 在线性函数上理论为 0，浮点误差量级约 1e-15，
           断言用 1e-9 而不是严格等于 0。
        3. 插补 RMSE 的比较依赖"列间确实近似线性"这一构造；若把噪声加大到与信号同量级，
           ``impute_regression`` 就不再优于均值插补。

    参考:
        Sobol' (2001)；Saltelli et al. (2010)；Morris (1991)；
        Iglewicz & Hoaglin (1993)；Troyanskaya et al. (2001)。
    """
    # ---- 1. OAT 与弹性（闭式解对拍）----
    def lin32(x: np.ndarray) -> float:
        return 3.0 * x[0] + 2.0 * x[1]

    oat = oat_sensitivity(lin32, [1.0, 1.0], rel_step=0.1)
    slopes = [item["sensitivity"] for item in oat["table"]]
    if abs(slopes[0] - 3.0) > 1e-9 or abs(slopes[1] - 2.0) > 1e-9:
        raise AssertionError(f"oat 斜率 {slopes} 与解析系数 [3, 2] 不符")
    if abs(oat["max_abs_change"] - 0.3) > 1e-9:
        raise AssertionError(f"oat max_abs_change={oat['max_abs_change']} != 3*0.1=0.3")

    def quad(x: np.ndarray) -> float:
        return x[0] ** 2 * x[1]

    e0 = elasticity(quad, [2.0, 3.0], 0)
    e1 = elasticity(quad, [2.0, 3.0], 1)
    if abs(e0 - 2.0) > 1e-9 or abs(e1 - 1.0) > 1e-9:
        raise AssertionError(f"弹性 [{e0}, {e1}] 与解析值 [2, 1] 不符")

    # ---- 2. Morris 基本效应 = 线性系数 ----
    mor = morris_screening(lin32, [[0.0, 1.0], [0.0, 1.0]], n_trajectories=12,
                           n_levels=4, seed=5)
    if np.max(np.abs(mor["mu"] - np.array([3.0, 2.0]))) > 1e-9:
        raise AssertionError(f"morris mu={mor['mu'].tolist()} 与系数 [3, 2] 不符")
    if np.max(np.abs(mor["sigma"])) > 1e-9:
        raise AssertionError(f"morris 线性函数 sigma={mor['sigma'].tolist()} 应约为 0")

    # ---- 3. Sobol：解析 S1，样本量按容差选取 ----
    def lin21(x: np.ndarray) -> float:
        return 2.0 * x[0] + x[1]

    def inter(x: np.ndarray) -> float:
        return x[0] * x[1]

    n_sob_add = 32768     # 可加线性函数：S1 解析值 [0.8, 0.2]，本 seed 下实际误差 0.012（容差 0.05 的 1/4）
    n_sob_int = 2048      # 交互函数：只需验证 ST > S1（解析值 0.571 vs 0.429），小样本即可
    so1 = sobol_first_order(lin21, [[0.0, 1.0], [0.0, 1.0]], n_samples=n_sob_add, seed=11)
    st1 = sobol_total_effect(lin21, [[0.0, 1.0], [0.0, 1.0]], n_samples=n_sob_add, seed=11)
    s1 = np.asarray(so1["S1"], dtype=float)
    if abs(s1[0] - 0.8) > 0.05 or abs(s1[1] - 0.2) > 0.05:
        raise AssertionError(f"sobol S1={s1.tolist()} 与解析值 [0.8, 0.2] 不符（容差 0.05）")
    st_add = np.asarray(st1["ST"], dtype=float)
    if np.max(np.abs(st_add - s1)) > 0.05:
        raise AssertionError(f"可加函数 ST={st_add.tolist()} 与 S1={s1.tolist()} 差超过 0.05")

    so2 = sobol_first_order(inter, [[0.0, 1.0], [0.0, 1.0]], n_samples=n_sob_int, seed=13)
    st2 = sobol_total_effect(inter, [[0.0, 1.0], [0.0, 1.0]], n_samples=n_sob_int, seed=13)
    s1_int = np.asarray(so2["S1"], dtype=float)
    st_int = np.asarray(st2["ST"], dtype=float)
    if not np.all(st_int > s1_int):
        raise AssertionError(f"交互函数应满足 ST > S1，得到 S1={s1_int.tolist()} ST={st_int.tolist()}")

    # ---- 4. 插补：回归插补的 RMSE 必须优于均值插补 ----
    gen = rng(20240115)
    n_row = 60
    x0 = gen.uniform(0.0, 10.0, n_row)
    x1 = 2.0 * x0 + 0.10 * gen.standard_normal(n_row)
    x2 = -1.5 * x0 + 0.05 * gen.standard_normal(n_row)
    full = np.column_stack([x0, x1, x2])
    flat_idx = gen.choice(full.size, size=int(0.10 * full.size), replace=False)
    hole_r, hole_c = np.unravel_index(flat_idx, full.shape)
    damaged = full.copy()
    damaged[hole_r, hole_c] = np.nan
    truth = full[hole_r, hole_c]

    im_mean = impute_mean(damaged)
    im_knn = impute_knn(damaged, k=5)
    im_reg = impute_regression(damaged, max_iter=60, tol=1e-6)

    def _rmse(mat: np.ndarray) -> float:
        return float(np.sqrt(np.mean((mat[hole_r, hole_c] - truth) ** 2)))

    rmse_mean = _rmse(im_mean["X"])
    rmse_knn = _rmse(im_knn["X"])
    rmse_reg = _rmse(im_reg["X"])
    if im_reg["n_imputed"] != int(flat_idx.size):
        raise AssertionError(f"impute_regression n_imputed={im_reg['n_imputed']} != {flat_idx.size}")
    if not rmse_reg < rmse_mean:
        raise AssertionError(f"回归插补 RMSE {rmse_reg} 未优于均值插补 {rmse_mean}")
    if not rmse_knn < rmse_mean:
        raise AssertionError(f"KNN 插补 RMSE {rmse_knn} 未优于均值插补 {rmse_mean}")
    if not im_reg["converged"]:
        raise AssertionError(f"impute_regression 未收敛，n_iter={im_reg['n_iter']}")
    if np.isnan(im_reg["X"]).any():
        raise AssertionError("impute_regression 输出仍含 nan")

    # ---- 5. 三种异常检测必须抓到同一个远端离群点 ----
    base = np.linspace(10.0, 40.0, 30)
    outlier_val = 500.0
    series = np.concatenate([base, [outlier_val]])
    out_idx = int(series.size - 1)
    zs = detect_outliers_zscore(series, threshold=3.0)
    iq = detect_outliers_iqr(series, k=1.5)
    mad = detect_outliers_mad(series, threshold=3.5)
    if zs["index"] != [out_idx]:
        raise AssertionError(f"zscore 抓到 {zs['index']}，期望 [{out_idx}]")
    if iq["index"] != [out_idx]:
        raise AssertionError(f"iqr 抓到 {iq['index']}，期望 [{out_idx}]")
    if mad["index"] != [out_idx]:
        raise AssertionError(f"mad 抓到 {mad['index']}，期望 [{out_idx}]")

    return {
        "sens_oat_slopes": [round(float(v), 6) for v in slopes],
        "sens_oat_max_abs_change": round(float(oat["max_abs_change"]), 6),
        "sens_oat_base": round(float(oat["base"]), 6),
        "sens_elasticity": [round(float(e0), 6), round(float(e1), 6)],
        "sens_morris_mu": [round(float(v), 6) for v in mor["mu"]],
        "sens_morris_mu_star": [round(float(v), 6) for v in mor["mu_star"]],
        "sens_morris_sigma_max": round(float(np.max(mor["sigma"])), 6),
        "sens_morris_ranking": [int(i) for i in mor["ranking"]],
        "sens_sobol_s1_linear": [round(float(v), 6) for v in s1],
        "sens_sobol_s1_conf_max": round(float(np.max(so1["S1_conf"])), 6),
        "sens_sobol_s1_sum": round(float(s1.sum()), 6),
        "sens_sobol_st_linear": [round(float(v), 6) for v in st_add],
        "sens_sobol_st_minus_s1_max": round(float(np.max(np.abs(st_add - s1))), 6),
        "sens_sobol_n_eval": int(so1["n_eval"]),
        "sens_sobol_s1_interaction": [round(float(v), 6) for v in s1_int],
        "sens_sobol_st_interaction": [round(float(v), 6) for v in st_int],
        "sens_sobol_interaction_gap": round(float(np.min(st_int - s1_int)), 6),
        "sens_impute_n_missing": int(im_reg["n_imputed"]),
        "sens_impute_mean_rmse": round(rmse_mean, 6),
        "sens_impute_knn_rmse": round(rmse_knn, 6),
        "sens_impute_regression_rmse": round(rmse_reg, 6),
        "sens_impute_regression_iter": int(im_reg["n_iter"]),
        "sens_outlier_index_zscore": [int(i) for i in zs["index"]],
        "sens_outlier_index_iqr": [int(i) for i in iq["index"]],
        "sens_outlier_index_mad": [int(i) for i in mad["index"]],
        "sens_outlier_agree": bool(zs["index"] == iq["index"] == mad["index"]),
    }
