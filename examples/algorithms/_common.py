"""公共工具：输入校验、归一化、可复现随机数。

这些函数刻意做得很小，只解决"每个模块都会重复写一遍"的问题；任何有争议的建模决策
（例如正向化方式、权重归一化口径）都留在各自模块里显式写出，便于论文中交代。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from . import DEFAULT_SEED

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "as_matrix",
    "as_vector",
    "check_square",
    "check_same_length",
    "normalize_minmax",
    "normalize_l2",
    "normalize_sum",
    "safe_divide",
    "rng",
    "DEFAULT_SEED",
]


def as_matrix(a: MatrixLike, name: str = "input") -> np.ndarray:
    """转成二维 float64 数组，并顺便校验形状合法。

    参数:
        a: 任何可转成二维数值数组的对象。
        name: 出错信息里显示的名字，便于定位。

    返回:
        np.ndarray，形状 (m, n)，dtype float64。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        一维输入会被当成"1 行的矩阵"而不是报错。如果你传的是向量，请显式用 ``as_vector``，
        否则后续按列取权重时会静默拿到错误结果。

    参考:
        numpy 数组协议（numpy.asarray）。
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，得到 ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN 或 inf")
    return arr


def as_vector(a: ArrayLike, name: str = "input") -> np.ndarray:
    """转成一维 float64 数组。

    参数:
        a: 任何可转成一维数值数组的对象。
        name: 出错信息里显示的名字。

    返回:
        np.ndarray，形状 (n,)，dtype float64。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        形状 (n, 1) 的列向量会被展平为长度 n，不会报错——如果你依赖二维形状，
        请在调用处自己保留维度。

    参考:
        numpy 数组协议。
    """
    arr = np.asarray(a, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN 或 inf")
    return arr


def check_square(m: np.ndarray, name: str = "matrix") -> None:
    """校验是方阵。

    参数:
        m: 已转好的二维数组。
        name: 出错信息里显示的名字。

    返回:
        None，不合法时抛 ValueError。

    复杂度:
        时间 O(1)。

    陷阱:
        只检查形状，不检查是否对称、是否可逆；判断矩阵还需要单独做一致性检验。

    参考:
        无。
    """
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{name} 必须是方阵，得到形状 {m.shape}")


def check_same_length(*arrays: ArrayLike) -> int:
    """校验若干个一维数组长度一致，返回公共长度。

    参数:
        *arrays: 待校验的数组。

    返回:
        公共长度 n。

    复杂度:
        时间 O(k)。

    陷阱:
        不做广播——长度不一致直接报错，这是有意的，竞赛里静默广播几乎总是 bug。

    参考:
        无。
    """
    lens = [np.asarray(a).ravel().size for a in arrays]
    if len(set(lens)) != 1:
        raise ValueError(f"数组长度不一致：{lens}")
    return lens[0]


def normalize_minmax(x: ArrayLike, benefit: bool = True, axis: int = 0) -> np.ndarray:
    """极差归一化（min-max），可选正向/负向指标。

    参数:
        x: 待归一化数据，二维 (m 个方案, n 个指标) 或一维。
        benefit: True 表示越大越好（正向指标）；False 表示越小越好（成本型指标）。
        axis: 归一化方向，默认 0 表示按列（每个指标独立归一化）。

    返回:
        与输入同形状的数组，取值 [0, 1]。

    算法:
        benefit:  (x - min) / (max - min)
        非 benefit: (max - x) / (max - min)
        极差为 0 时该列全部置 0.5（表示"该指标无区分度"），并在返回值中保持有限。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        极差为 0 的列如果按 0 除处理，会得到 NaN 并一路污染到 TOPSIS 得分。
        另一种常见错误是**先归一化再正向化**，两种顺序在 min-max 下结果不同。

    参考:
        多属性决策中的数据预处理通例。
    """
    arr = np.asarray(x, dtype=float)
    lo = arr.min(axis=axis, keepdims=True)
    hi = arr.max(axis=axis, keepdims=True)
    rng_ = hi - lo
    safe = np.where(rng_ > 0, rng_, 1.0)
    out = (arr - lo) / safe if benefit else (hi - arr) / safe
    flat = rng_ <= 0
    if np.any(flat):
        out = np.where(np.broadcast_to(flat, out.shape), 0.5, out)
    return out


def normalize_l2(x: ArrayLike, axis: int = 0) -> np.ndarray:
    """向量归一化（除以 L2 范数），TOPSIS 的标准做法。

    参数:
        x: 待归一化数据。
        axis: 归一化方向，默认 0（按列）。

    返回:
        与输入同形状的数组。

    算法:
        r_ij = x_ij / sqrt(sum_i x_ij^2)

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        列全为 0 时范数为 0；这里把该列原样保留为 0（而不是 NaN），
        但列全 0 说明这个指标没有信息，应当在论文中说明并考虑剔除。

    参考:
        Hwang & Yoon (1981) TOPSIS 原始定义。
    """
    arr = np.asarray(x, dtype=float)
    norm = np.sqrt((arr ** 2).sum(axis=axis, keepdims=True))
    return np.divide(arr, norm, out=np.zeros_like(arr, dtype=float), where=norm > 0)


def normalize_sum(x: ArrayLike, axis: int = 0) -> np.ndarray:
    """和归一化（每一项除以该列总和）。

    参数:
        x: 待归一化数据（通常是非负的）。
        axis: 归一化方向，默认 0。

    返回:
        与输入同形状数组，沿 axis 求和为 1。

    算法:
        p_ij = x_ij / sum_i x_ij

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        要求非负；含负数时"比重"没有意义，熵权法会被 log 直接判 NaN。

    参考:
        信息熵权重的标准预处理。
    """
    arr = np.asarray(x, dtype=float)
    if np.any(arr < 0):
        raise ValueError("normalize_sum 要求非负输入（熵权法等比重口径）")
    total = arr.sum(axis=axis, keepdims=True)
    return np.divide(arr, total, out=np.zeros_like(arr, dtype=float), where=total > 0)


def safe_divide(a: ArrayLike, b: ArrayLike, fill: float = 0.0) -> np.ndarray:
    """逐元素相除，分母为 0 时用 fill 填充而不是产生 inf/NaN。

    参数:
        a: 分子。
        b: 分母。
        fill: 分母为 0 时的替代值。

    返回:
        与广播后形状一致的数组。

    复杂度:
        时间 O(n)。

    陷阱:
        fill 的取值会影响结果，不要随手用 0：例如"单位成本"类指标分母为 0 时填 0 会
        把不可行方案伪装成最优，填一个大数反而更保守。

    参考:
        无。
    """
    num = np.asarray(a, dtype=float)
    den = np.asarray(b, dtype=float)
    return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, float(fill)), where=den != 0)


def rng(seed: Optional[int] = None) -> np.random.Generator:
    """返回独立的随机数生成器（不使用全局状态）。

    参数:
        seed: 随机种子；None 表示使用库默认种子 DEFAULT_SEED。

    返回:
        numpy.random.Generator。

    算法:
        numpy 新版 Generator 接口（PCG64）。

    复杂度:
        时间 O(1)。

    陷阱:
        不要用 ``np.random.seed``：全局状态会让并行或多次调用的结果互相干扰，
        导致"同样的代码两次跑出不同结论"。

    参考:
        numpy.random.default_rng 文档。
    """
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)
