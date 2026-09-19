"""几何与空间分析：凸包、多边形面积、点在多边形内、球面距离、IDW 插值、普通克里金。

本模块共 13 个公开名称（12 个函数 + 1 个常量），按用途分成三组：
- 平面几何：``convex_hull`` / ``polygon_area`` / ``point_in_polygon`` / ``polygon_centroid``
  / ``segment_intersection`` / ``minimum_enclosing_circle`` / ``point_to_segment_distance``
  / ``voronoi_nearest``；
- 球面距离：``haversine`` 与常量 ``EARTH_RADIUS_KM``；
- 空间插值与地统计：``idw_interpolate`` / ``ordinary_kriging`` / ``estimate_variogram_params``。

这些都是**教学透明版**实现，只依赖 numpy 与标准库：目标是让论文能写清"数据是怎么
从若干个采样点被插值/判定出来的"，以及每种方法的隐含假设。

坐标系约定
----------
- ``convex_hull`` / ``polygon_area`` / ``point_in_polygon`` / ``idw_interpolate`` /
  ``ordinary_kriging`` 都工作在**平面直角坐标** (x, y) 上，不涉及经纬度。
- 只有 ``haversine`` 使用经纬度（度），并把地球当作半径 6371.0088 km 的球。
- 球面距离的球模型在长距离上有约 0.3% 的误差（地球是椭球）；要更高精度请用
  Vincenty 或 WGS-84 椭球公式。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_vector, check_same_length

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

#: 地球平均半径（km），IUGG 推荐的算术平均半径。
EARTH_RADIUS_KM = 6371.0088

__all__ = [
    "convex_hull",
    "polygon_area",
    "point_in_polygon",
    "haversine",
    "idw_interpolate",
    "ordinary_kriging",
    "estimate_variogram_params",
    "segment_intersection",
    "minimum_enclosing_circle",
    "polygon_centroid",
    "point_to_segment_distance",
    "voronoi_nearest",
    "EARTH_RADIUS_KM",
]


def _as_points(a: MatrixLike, name: str) -> np.ndarray:
    """内部工具：转成形状 (n, 2) 的点集并校验。"""
    pts = np.asarray(a, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} 必须是形状 (n, 2) 的点集，得到 {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} 含 NaN/inf")
    return pts


def convex_hull(points: MatrixLike) -> List[int]:
    """Andrew 单调链求二维凸包，返回凸包顶点的**原始下标**（逆时针）。

    参数:
        points: 形状 (n, 2) 的点集。

    返回:
        list[int]，凸包顶点的下标，按逆时针排列、**不含共线边上的中间点**。
        当所有点共线时返回该线段的两个端点；n=1 时返回 [0]；n=2 时返回两个下标。

    算法:
        先按 (x, y) 字典序排序，再分别扫出下链与上链：维护一个栈，若新点与栈顶两点
        构成非左转（叉积 <= 0）则弹栈。最后拼接两条链即得逆时针凸包。

    复杂度:
        时间 O(n log n)（排序主导）/ 空间 O(n)。

    陷阱:
        1. **叉积判据用 <= 还是 <**：用 ``<= 0`` 会丢掉凸包边上的共线中间点（本实现
           如此，通常正是想要的），用 ``< 0`` 会把它们保留成"伪顶点"，导致正方形边上
           多出若干点。做"凸包顶点数"这类断言时务必先确认口径。
        2. 经纬度点**不能**直接当平面坐标求凸包：高纬度处经度 1 度只有几十公里，
           凸包会明显变形。要么先投影，要么改用球面凸包。
        3. 退化输入（所有点重合、只有 1~2 个点）没有"内部"概念，本函数返回退化结果
           而不报错，调用方需要自己判断。
        4. 返回的是下标而不是坐标；若入参是 DataFrame 切片，请确认下标还原后仍对应原行。

    参考:
        Andrew, "Another efficient algorithm for convex hulls in two dimensions",
        Information Processing Letters, 1979。
    """
    pts = _as_points(points, "points")
    n = pts.shape[0]
    order = np.lexsort((pts[:, 1], pts[:, 0])).tolist()
    if n == 1:
        return [int(order[0])]
    if n == 2:
        return [int(order[0]), int(order[1])]

    def cross(o: int, a: int, b: int) -> float:
        return float(
            (pts[a, 0] - pts[o, 0]) * (pts[b, 1] - pts[o, 1])
            - (pts[a, 1] - pts[o, 1]) * (pts[b, 0] - pts[o, 0])
        )

    lower: List[int] = []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0.0:
            lower.pop()
        lower.append(i)
    upper: List[int] = []
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0.0:
            upper.pop()
        upper.append(i)
    hull = lower[:-1] + upper[:-1]
    return [int(i) for i in hull]


def polygon_area(points: MatrixLike) -> float:
    """用鞋带公式（shoelace）求简单多边形的面积，取绝对值。

    参数:
        points: 形状 (n, 2) 的多边形顶点，按顺序给出（顺/逆时针都行）。

    返回:
        float，面积（与输入坐标同单位的平方），非负。

    算法:
        A = |0.5 * Σ (x_i y_{i+1} - x_{i+1} y_i)|，下标按模 n 回绕。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. **只对简单多边形成立**：边自交（例如"8 字形"）时正负面积会互相抵消，
           结果是无意义的。遇到自交数据应先做多边形拆分或改用三角剖分。
        2. 顶点顺序决定符号（逆时针为正），本函数已取绝对值；若你需要判断方向，
           请直接算带符号面积。
        3. 经纬度直接代入会把面积算成"度²"，且随纬度严重失真；面积类计算必须
           先投影（如 UTM）或用球面多边形公式。
        4. 顶点重复（闭合点写了一头一尾两次）不影响结果，但会让 n 虚增 1。

    参考:
        鞋带公式（Gauss 1795 年左右的测量笔记），计算几何教材通例。
    """
    pts = _as_points(points, "points")
    n = pts.shape[0]
    if n < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    return float(abs(0.5 * np.sum(x * y_next - x_next * y)))


def point_in_polygon(point: ArrayLike, polygon: MatrixLike) -> bool:
    """判断点是否落在简单多边形内部（含边界），用射线法（crossing number）。

    参数:
        point: 长度 2 的坐标 ``(x, y)``。
        polygon: 形状 (n, 2) 的多边形顶点，首尾**不必**重复。

    返回:
        bool。点在多边形内部或**恰好落在边上/顶点上**时返回 True。

    算法:
        从待测点向 +x 方向作射线，统计与多边形边的交点个数：奇数在内、偶数在外。
        边界判定单独用叉积与包围盒做，命中即返回 True（避免射线刚好过顶点时的歧义）。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. **边界口径必须写清楚**：本函数把"点在边上"算作内部。而 GIS 里常把边界
           单独算一类（DE-9IM 的 boundary），论文中应显式声明，否则同样的坐标
           在不同软件里会得到不同结果。
        2. 射线法对"射线正好穿过顶点/与边重合"的情形天生有歧义，本实现用
           ``(y_i > y) != (y_j > y)`` 的半开区间写法消除大部分退化，但极端退化数据
           （大量水平边）仍可能出错。
        3. 多边形必须是**简单多边形**（边不自交）；自交时"内/外"本身无定义。
        4. 经纬度坐标在平面近似下判断"是否在行政区内"只适用于小范围，跨半球或
           高纬度区域会判错。

    参考:
        计算几何经典算法（如 O'Rourke, "Computational Geometry in C", 1998）。
    """
    p = as_vector(point, "point")
    if p.size != 2:
        raise ValueError(f"point 必须是长度 2 的坐标，得到长度 {p.size}")
    poly = _as_points(polygon, "polygon")
    n = poly.shape[0]
    if n < 3:
        raise ValueError(f"polygon 至少需要 3 个顶点，得到 {n}")
    x, y = float(p[0]), float(p[1])

    inside = False
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i + 1) % n]
        # 边界判定：点在线段包围盒内且叉积为 0
        if (min(xi, xj) - 1e-12 <= x <= max(xi, xj) + 1e-12) and (
            min(yi, yj) - 1e-12 <= y <= max(yi, yj) + 1e-12
        ):
            cross = (xj - xi) * (y - yi) - (yj - yi) * (x - xi)
            if abs(cross) <= 1e-12:
                return True
        if (yi > y) != (yj > y):
            x_cross = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x_cross > x:
                inside = not inside
    return bool(inside)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面两点的大圆距离（haversine 公式），单位公里。

    参数:
        lat1, lon1: 起点纬度、经度（**度**，北纬/东经为正）。
        lat2, lon2: 终点纬度、经度（度）。

    返回:
        float，大圆距离（km），地球半径取 6371.0088 km。

    算法:
        a = sin²(Δφ/2) + cosφ1·cosφ2·sin²(Δλ/2)，d = 2R·asin(√a)。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. **必须先把度转成弧度**：直接把度代入 sin 会得到荒谬但"看起来像数字"的结果。
        2. haversine 假设地球是正球，长距离（> 1000 km）相对 WGS-84 椭球有约 0.3%
           的误差，跨国航线、卫星轨迹类题目不能直接用它下结论。
        3. 经度差跨越 ±180°（换日线）时直接用 Δλ=lon2-lon1 会得到接近 360° 的错误值，
           必须先归一化到 (-180, 180]；本函数用 sin² 的形式天然对 Δλ+360° 不变，
           所以实际上不受影响——但换成 law of cosines 写法就会踩坑。
        4. 本函数只算"直线距离"，不含道路约束；路网距离要用最短路算法（见 graphs 模块）。

    参考:
        球面三角学的半正矢公式；半径取值见 IUGG 平均地球半径。
    """
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = phi2 - phi1
    dlam = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    a = min(1.0, max(0.0, a))  # 数值保护，避免浮点误差让 asin 定义域越界
    return float(2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a)))


def idw_interpolate(
    points: MatrixLike,
    values: ArrayLike,
    query: MatrixLike,
    power: float = 2.0,
) -> np.ndarray:
    """反距离加权插值（IDW）：离样本越近权重越大，权重 ∝ 1/d^p。

    参数:
        points: 样本点坐标，形状 (n, 2)。
        values: 样本值，长度 n。
        query: 待插值点，形状 (m, 2)。
        power: 距离幂次 p（通常 1~4，默认 2），必须 > 0。

    返回:
        np.ndarray，形状 (m,) 的插值结果。

    算法:
        w_i = 1 / d_i^p，ŷ = Σ w_i y_i / Σ w_i；当查询点与某样本点重合（d=0）时
        直接返回该样本值（避免 1/0）。

    复杂度:
        时间 O(nm) / 空间 O(nm)（距离矩阵）。

    陷阱:
        1. **IDW 不是插值器而是平滑器**：它永远不产生比样本极值更大/更小的值
           （凸性），因此会抹平极值，做"污染峰值/最高点"类预测时系统性偏低。
        2. power 越大越"贴点"（接近最近邻、产生牛眼状台阶），越小越平滑；
           论文里应做 power 的敏感性分析（例如 p=1,2,3 的交叉验证误差）。
        3. 不含地形等协变量，只依赖距离；在采样密度不均时会向密集区偏。
        4. 距离用平面欧氏距离，量纲应与坐标一致；用经纬度必须先换算成公里。

    参考:
        Shepard, "A two-dimensional interpolation function for irregularly-spaced data",
        ACM 1968。
    """
    pts = _as_points(points, "points")
    vals = as_vector(values, "values")
    check_same_length(pts[:, 0], vals)
    q = _as_points(query, "query")
    if power <= 0:
        raise ValueError(f"power 必须 > 0，得到 {power}")

    diff = q[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    out = np.empty(q.shape[0], dtype=float)
    for j in range(q.shape[0]):
        d = dist[j]
        hit = np.flatnonzero(d <= 1e-12)
        if hit.size > 0:
            out[j] = float(vals[hit[0]])  # 与样本点重合：直接取该样本值
            continue
        w = 1.0 / d ** power
        out[j] = float(np.dot(w, vals) / np.sum(w))
    return out


def _variogram_model(model: str, h: np.ndarray, nugget: float, sill: float, vrange: float) -> np.ndarray:
    """内部工具：按模型名计算半变异函数 γ(h)（h 为距离数组，单位与坐标一致）。

    sill 是总基台值（含 nugget），vrange 是变程；h=0 处恒有 γ=0（对 nugget>0 的
    理论模型而言 γ(0)=0 是"同一位置没有差异"的物理约束，nugget 只体现在 h→0+ 的跳变）。
    """
    partial = max(sill - nugget, 0.0)
    h = np.asarray(h, dtype=float)
    hr = np.divide(h, vrange, out=np.zeros_like(h), where=vrange > 0)
    if model == "spherical":
        inner = np.where(hr < 1.0, 1.5 * hr - 0.5 * hr ** 3, 1.0)
    elif model == "exponential":
        inner = 1.0 - np.exp(-3.0 * hr)
    elif model == "gaussian":
        inner = 1.0 - np.exp(-3.0 * hr ** 2)
    elif model == "linear":
        inner = np.minimum(hr, 1.0)
    else:
        raise ValueError(f"model 只支持 spherical/exponential/gaussian/linear，得到 {model!r}")
    gamma = nugget + partial * inner
    return np.where(h <= 0.0, 0.0, gamma)


def ordinary_kriging(
    points: MatrixLike,
    values: ArrayLike,
    query: MatrixLike,
    model: str = "spherical",
    nugget: float = 0.0,
    sill: Optional[float] = None,
    vrange: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """普通克里金（OK）插值：用变异函数刻画空间相关性，给出预测值与其方差。

    参数:
        points: 样本点坐标，形状 (n, 2)。
        values: 样本值，长度 n。
        query: 待预测点，形状 (m, 2)。
        model: 半变异函数模型，``"spherical"``（默认）/``"exponential"``/
            ``"gaussian"``/``"linear"``。
        nugget: 块金常数 c0（测量误差 + 微尺度变异），>= 0，默认 0。
        sill: **总基台值**（= 块金 + 偏基台）；默认取样本方差（``np.var(values)``）。
            传 None 表示自动估计。显式传入时必须满足 ``sill >= nugget``，否则抛
            ``ValueError``；注意 :func:`estimate_variogram_params` 返回的是**偏基台**
            （样本方差，不含块金），nugget > 0 时要先加回 nugget 再传进来。
        vrange: 变程（相关性的空间尺度）；默认取样本点最大两两距离的一半。

    返回:
        dict，键为：
        ``prediction`` 形状 (m,) 的克里金预测值；
        ``variance`` 形状 (m,) 的克里金方差（预测标准差的平方，已截断为 ≥ 0）。

    算法:
        以**半变异函数**为工具，对每个预测点 x0 解克里金方程组
        [Γ 1; 1ᵀ 0] [w; μ] = [γ0; 1]，其中 Γ_ij = γ(‖xi-xj‖)（对角元按理论定义取
        γ(0)=0）、γ0_i = γ(‖xi-x0‖)。
        预测值 = Σ w_i z_i，克里金方差 = Σ w_i γ0_i + μ。
        方程组用 ``np.linalg.solve`` 求解，**不依赖 scipy**。

    复杂度:
        时间 O(m n³)（每个预测点一次 n+1 阶解方程）/ 空间 O(n² + mn)。

    陷阱:
        1. **默认参数是粗糙估计**：sill 用样本方差、变程用"最大距离的一半"，这只是
           为了让你能一行调用。真正的做法是先用实验变异函数云图拟合模型参数
           （或交叉验证选参），否则预测区间会明显失真。要严格控制就显式传 sill/vrange。
        2. **块金常数的两种口径会让"是否精确过点"完全不同**：本实现按半变异函数的
           理论定义取 γ(0)=0，块金只出现在 h>0 处，因此无论 nugget 取多少，
           **预测值都精确等于样本观测值、样本点处方差为 0**（实测：四角数据
           nugget=0.3 时样本点最大偏差仍为 1.1e-16）。若采用"观测误差"口径
           （把块金放到 Γ 的对角元上），块金才会让预测变平滑、样本点处不再相等。
           论文里必须写清用的是哪一种，否则和软件结果对不上。
        3. **普通克里金假设无全局趋势**：数据有明显漂移（如自西向东升高）时应当用
           泛克里金或先去掉趋势面，否则预测会出现系统性偏差。
        4. 矩阵可能病态：样本点极其密集 + 高斯模型时 Γ 接近奇异，
           ``np.linalg.solve`` 会报 LinAlgError 或给出巨大权重；此时应拉开采样间距、
           加 nugget 或改用带惩罚的解法。
        5. 变异函数模型选错（例如数据是周期性的却用球状模型）会让预测"看起来光滑但错"。
        6. 若 ``nugget`` 超过样本方差而 ``sill`` 仍留 None，会直接抛
           ``ValueError: sill(...) 不能小于 nugget(...)``——这不是 bug：总基台值不可能
           小于块金。此时要么显式给一个更大的 ``sill``，要么承认块金估计过大。

    参考:
        Matheron, "Principles of geostatistics", Economic Geology, 1963；
        Cressie, "Statistics for Spatial Data", 1993。
    """
    pts = _as_points(points, "points")
    vals = as_vector(values, "values")
    check_same_length(pts[:, 0], vals)
    q = _as_points(query, "query")
    n = pts.shape[0]
    if n < 2:
        raise ValueError(f"克里金至少需要 2 个样本点，得到 {n}")
    if nugget < 0:
        raise ValueError(f"nugget 必须 >= 0，得到 {nugget}")

    dmat = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    max_dist = float(dmat.max())
    if max_dist <= 0.0:
        raise ValueError("所有样本点重合，无法估计变程（请去重或检查坐标）")
    if sill is None:
        sill = float(np.var(vals))
    if vrange is None:
        vrange = 0.5 * max_dist
    if sill < 0:
        raise ValueError(f"sill 必须 >= 0，得到 {sill}")
    if vrange <= 0:
        raise ValueError(f"vrange 必须 > 0，得到 {vrange}")
    if sill < nugget:
        raise ValueError(f"sill({sill}) 不能小于 nugget({nugget})")

    if sill <= 0.0:
        # 样本值全相等：半变异函数恒为 0，任何位置的预测都是该常数，方差为 0。
        const = float(vals[0])
        return {
            "prediction": np.full(q.shape[0], const),
            "variance": np.zeros(q.shape[0]),
        }

    gamma = _variogram_model(model, dmat, float(nugget), float(sill), float(vrange))
    np.fill_diagonal(gamma, 0.0)
    a_mat = np.empty((n + 1, n + 1), dtype=float)
    a_mat[:n, :n] = gamma
    a_mat[n, :n] = 1.0
    a_mat[:n, n] = 1.0
    a_mat[n, n] = 0.0

    dq = np.sqrt(np.sum((q[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    gamma_0 = _variogram_model(model, dq, float(nugget), float(sill), float(vrange))

    pred = np.empty(q.shape[0], dtype=float)
    var = np.empty(q.shape[0], dtype=float)
    for j in range(q.shape[0]):
        rhs = np.empty(n + 1, dtype=float)
        rhs[:n] = gamma_0[j]
        rhs[n] = 1.0
        try:
            sol = np.linalg.solve(a_mat, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"克里金方程组奇异（第 {j} 个预测点）：{exc}；可尝试增大 nugget 或提高 vrange"
            ) from exc
        w = sol[:n]
        mu = float(sol[n])
        pred[j] = float(np.dot(w, vals))
        var[j] = float(np.dot(w, gamma_0[j]) + mu)

    return {
        "prediction": pred,
        "variance": np.maximum(var, 0.0),
    }


def estimate_variogram_params(points: MatrixLike, values: ArrayLike, nugget: float = 0.0) -> Dict[str, float]:
    """给出 ``ordinary_kriging`` 在未显式指定参数时会用的基台值与变程（便于写进论文）。

    参数:
        points: 样本点坐标，形状 (n, 2)。
        values: 样本值，长度 n。
        nugget: 块金常数，仅用于说明 sill 与 nugget 的关系，不参与估计。

    返回:
        dict，键为：
        ``sill`` **样本方差**（``np.var(values)``，ddof=0），**不含块金**；
        ``vrange`` 变程（样本点最大两两距离的一半）；
        ``nugget`` 原样回传的块金常数。

    算法:
        粗估法：sill = var(values)，vrange = max‖xi-xj‖ / 2。这只是让一行调用能跑起来的
        默认值，不是拟合结果。

    复杂度:
        时间 O(n²) / 空间 O(n²)。

    陷阱:
        1. 这个估计**没有任何统计学最优性**：样本点分布范围越大，变程估计越大，
           预测越平滑；要得到可靠参数必须拟合实验变异函数或做交叉验证。
        2. **口径不一致，必须自己换算**：本函数返回的 ``sill`` 是样本方差（偏基台值），
           而 :func:`ordinary_kriging` 与 :func:`_variogram_model` 的 ``sill`` 参数是
           **总基台值 = 块金 + 偏基台**。因此当 ``nugget > 0`` 时，不能把本函数的结果
           直接喂回去；正确做法是传 ``sill = estimate_variogram_params(...)["sill"] + nugget``
           （否则 ``ordinary_kriging`` 内部会算成 ``sill - nugget`` 的偏基台，凭空吃掉一块
           方差，预测被系统性压平）。只有 ``nugget == 0`` 时两者才恰好相等。

    参考:
        Cressie, "Statistics for Spatial Data", 1993（实验变异函数与模型拟合）。
    """
    pts = _as_points(points, "points")
    vals = as_vector(values, "values")
    check_same_length(pts[:, 0], vals)
    if nugget < 0:
        raise ValueError(f"nugget 必须 >= 0，得到 {nugget}")
    if pts.shape[0] < 2:
        raise ValueError(f"至少需要 2 个样本点，得到 {pts.shape[0]}")
    dmat = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    max_dist = float(dmat.max())
    if max_dist <= 0.0:
        raise ValueError("所有样本点重合，无法估计变程（请去重或检查坐标）")
    return {
        "sill": float(np.var(vals)),
        "vrange": 0.5 * max_dist,
        "nugget": float(nugget),
    }


def _point_on_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """内部工具：判断点 pt 是否落在线段 ab 上（tol 为绝对距离容差）。"""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0.0:
        return float(np.hypot(pt[0] - a[0], pt[1] - a[1])) <= tol
    t = float(np.dot(pt - a, ab)) / denom
    if t < -1e-9 or t > 1.0 + 1e-9:
        return False
    t_clamped = min(1.0, max(0.0, t))
    proj = a + t_clamped * ab
    return float(np.hypot(pt[0] - proj[0], pt[1] - proj[1])) <= tol


def segment_intersection(
    p1: ArrayLike,
    p2: ArrayLike,
    p3: ArrayLike,
    p4: ArrayLike,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """判断两条平面线段 (p1,p2) 与 (p3,p4) 是否相交，并给出交点与退化标志。

    参数:
        p1, p2: 第一条线段的两个端点，各为长度 2 的坐标 ``(x, y)``。
        p3, p4: 第二条线段的两个端点，各为长度 2 的坐标 ``(x, y)``。
        eps: 相对容差，用于"是否平行"与"是否共线"的判据（默认 1e-12）。

    返回:
        dict，键为：
        ``intersect``  bool，两线段是否有公共点（含只有一个公共端点的情形）；
        ``point``  相交时的交点，形状 (2,) 的数组；不相交时为 ``None``。共线重叠时
                   返回重叠区间的**中点**（重叠退化为一点时即该点）；
        ``parallel``  bool，两线段方向平行（含共线与零长度线段）时为 True；
        ``collinear_overlap``  bool，仅当两线段共线且重叠长度**严格为正**时为 True；
                   共线但只接触一个点时返回 False（同时 ``intersect`` 为 True）。

    算法:
        1. 记 r = p2-p1，s = p4-p3，qp = p3-p1，叉积 cross(u,v) = u_x v_y - u_y v_x。
        2. 平行判据用相对形式 |cross(r, s)| <= eps·|r|·|s|（零长度线段自动落入此支）。
        3. 非平行时解参数方程 p1+t·r = p3+u·s：
           t = cross(qp, s) / cross(r, s)，u = cross(qp, r) / cross(r, s)，
           两者都落在 [0, 1]（参数容差 1e-9，即线段长度的十亿分之一）才算相交。
        4. 平行时先判共线：|cross(qp, ref)| <= eps·|qp|·|ref|（ref 取 r，r 退化时取 s）。
           共线则把四个端点投影到 |ref| 较大的那个坐标轴上，两个闭区间交非空即相交；
           区间重叠长度 > 容差记为 ``collinear_overlap``。
        5. 零长度线段（一个端点为点）退化为"点在另一线段上"的判定。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. **"相交"必须区分"一个公共点"与"一段公共线段"**：平行且共线的两线段即使
           重叠，叉积判据也永远给不出唯一交点；只写"解方程求交点"的实现在这里会除以 0
           或给出 NaN。本函数用 ``collinear_overlap`` 显式区分这两种情形。
        2. 浮点数据的共线几乎从不是精确的：eps 取太大（如 1e-6）会把近距离平行但
           不共线的线段误判为共线；取太小则真实共线数据（由计算得来的坐标）判不出来。
           eps 是**相对**容差，与坐标量纲无关，但坐标整体幅值极大/极小时仍建议复核。
        3. ``intersect`` 只回答"是否有公共点"，不回答"内部是否穿越"。T 形接触
           （一个端点落在另一线段内部）与端点重合都返回 True，需要区分时请检查交点是否
           等于某个端点。
        4. 交点由第一条线段的参数式 p1+t·r 算出，因此 t 略超出 [0,1] 时交点会落在
           线段外一点点；参数容差 1e-9 意味着"距离端点 1e-9·|r| 以内"的接触也算相交。

    参考:
        计算几何经典线段相交判据（O'Rourke, "Computational Geometry in C", 1998）。
    """
    a = as_vector(p1, "p1")
    b = as_vector(p2, "p2")
    c = as_vector(p3, "p3")
    d = as_vector(p4, "p4")
    for name, v in (("p1", a), ("p2", b), ("p3", c), ("p4", d)):
        if v.size != 2:
            raise ValueError(f"{name} 必须是长度 2 的坐标，得到长度 {v.size}")
    if eps <= 0.0:
        raise ValueError(f"eps 必须 > 0，得到 {eps}")

    coords = np.vstack([a, b, c, d])
    pt_tol = eps * max(1.0, float(np.max(np.abs(coords))))

    r = b - a
    s = d - c
    len_r = float(np.hypot(r[0], r[1]))
    len_s = float(np.hypot(s[0], s[1]))
    rxs = float(r[0] * s[1] - r[1] * s[0])
    qp = c - a
    len_qp = float(np.hypot(qp[0], qp[1]))
    qpxr = float(qp[0] * r[1] - qp[1] * r[0])
    qpxs = float(qp[0] * s[1] - qp[1] * s[0])

    if abs(rxs) <= eps * len_r * len_s:
        # 平行（含共线与零长度线段）
        if len_r > pt_tol:
            ref, origin, len_ref = r, a, len_r
        else:
            ref, origin, len_ref = s, c, len_s
        collinear = abs(float(qp[0] * ref[1] - qp[1] * ref[0])) <= eps * len_qp * len_ref
        if not collinear:
            return {
                "intersect": False,
                "point": None,
                "parallel": True,
                "collinear_overlap": False,
            }
        if len_r <= pt_tol and len_s <= pt_tol:
            same = float(np.hypot(a[0] - c[0], a[1] - c[1])) <= pt_tol
            return {
                "intersect": bool(same),
                "point": a.copy() if same else None,
                "parallel": True,
                "collinear_overlap": False,
            }
        if len_r <= pt_tol:
            on = _point_on_segment(a, c, d, pt_tol)
            return {
                "intersect": bool(on),
                "point": a.copy() if on else None,
                "parallel": True,
                "collinear_overlap": False,
            }
        if len_s <= pt_tol:
            on = _point_on_segment(c, a, b, pt_tol)
            return {
                "intersect": bool(on),
                "point": c.copy() if on else None,
                "parallel": True,
                "collinear_overlap": False,
            }
        # 共线且两段都非退化：投影到 |ref| 较大的坐标轴上比较区间
        axis = 0 if abs(ref[0]) >= abs(ref[1]) else 1
        lo1, hi1 = sorted((float(a[axis]), float(b[axis])))
        lo2, hi2 = sorted((float(c[axis]), float(d[axis])))
        lo = max(lo1, lo2)
        hi = min(hi1, hi2)
        if lo > hi + pt_tol:
            return {
                "intersect": False,
                "point": None,
                "parallel": True,
                "collinear_overlap": False,
            }
        t_mid = (0.5 * (lo + hi) - float(origin[axis])) / float(ref[axis])
        point = origin + t_mid * ref
        return {
            "intersect": True,
            "point": point,
            "parallel": True,
            "collinear_overlap": bool(hi - lo > pt_tol),
        }

    param_tol = 1e-9
    t = qpxs / rxs
    u = qpxr / rxs
    hit = (-param_tol <= t <= 1.0 + param_tol) and (-param_tol <= u <= 1.0 + param_tol)
    return {
        "intersect": bool(hit),
        "point": (a + t * r) if hit else None,
        "parallel": False,
        "collinear_overlap": False,
    }


def _circle_from_two(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """内部工具：以 ab 为直径的圆（唯一能同时过两点的最小圆）。"""
    center = 0.5 * (a + b)
    return center, float(np.hypot(a[0] - b[0], a[1] - b[1])) / 2.0


def _circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, float]:
    """内部工具：三点确定的外接圆；三点近似共线时退化为最远两点的直径圆。"""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    scale = max(
        float(np.hypot(ax - bx, ay - by)),
        float(np.hypot(ax - cx, ay - cy)),
        float(np.hypot(bx - cx, by - cy)),
    )
    if scale <= 0.0:
        return a.copy(), 0.0
    if abs(d) <= 1e-12 * scale * scale:
        pairs = ((a, b), (a, c), (b, c))
        best = max(pairs, key=lambda pr: float(np.hypot(pr[0][0] - pr[1][0], pr[0][1] - pr[1][1])))
        return _circle_from_two(best[0], best[1])
    sa = ax * ax + ay * ay
    sb = bx * bx + by * by
    sc = cx * cx + cy * cy
    ux = (sa * (by - cy) + sb * (cy - ay) + sc * (ay - by)) / d
    uy = (sa * (cx - bx) + sb * (ax - cx) + sc * (bx - ax)) / d
    center = np.array([ux, uy], dtype=float)
    return center, float(np.hypot(ux - ax, uy - ay))


def minimum_enclosing_circle(points: MatrixLike) -> Dict[str, Union[np.ndarray, float]]:
    """求点集的最小包围圆（Welzl 增量算法，**确定性固定顺序**版本）。

    参数:
        points: 形状 (n, 2) 的点集，n >= 1。

    返回:
        dict，键为：
        ``center``  形状 (2,) 的圆心坐标；
        ``radius``  float，半径（n=1 时为 0.0）。

    算法:
        1. 先取前两个点构成直径圆；随后依次加入第 i 个点（i 从 2 到 n-1）。
        2. 若第 i 个点在当前圆内则跳过；否则第 i 个点必在新圆边界上 —— 重置为
           "以第 i 个点为圆心、半径 0"的退化圆。
        3. 对 j < i：若点 j 不在圆内，则新圆必同时过 i 与 j，取两点直径圆；再对
           k < j 检查，若点 k 不在圆内，则新圆是 i、j、k 的外接圆（近似共线时退化为
           最远两点直径圆，它同样覆盖第三点）。
        4. 内点判据用 ``dist <= radius + 1e-12·max(1, radius)``；两点圆用直径、
           三点圆用外接圆闭式解，全部为确定性算术。

    复杂度:
        时间 O(n³) 最坏（本实现不做随机洗牌，因此没有"期望 O(n)"的保证）/
        空间 O(n)。

    陷阱:
        1. **不洗牌 = 放弃期望线性时间**：Welzl 算法的 O(n) 期望复杂度依赖随机化，
           这里为满足"两次调用结果必须逐位一致"而固定输入顺序，最坏 O(n³)。对竞赛
           常见的 n ≤ 数千仍然够用；若 n 很大，请自行传入已按固定种子打乱的点集
           （结果与输入顺序无关，只有耗时有关）。
        2. **最小包围圆由 2 或 3 个点唯一决定**（直径圆或外接圆），所以半径是一个
           "脆"的量：输入中一个远点就会换掉决定圆的那组点。用半径做断言时应当选
           正多边形顶点这类对称点集。
        3. 近共线的三点外接圆半径会趋于无穷，本实现用相对判据 |2·cross| <= 1e-12·L²
           （L 为三点最大边长）切到直径圆分支，否则会得到巨大的圆心与半径。
        4. **点必须互不相同**：重复点不会破坏算法（半径仍正确），但会让"内部"判定
           全部落在边界上，不影响结论。

    参考:
        Welzl, "Smallest enclosing disks (balls and ellipsoids)", LNCS 555, 1991；
        Nayuki 的确定性迭代实现说明。
    """
    pts = _as_points(points, "points")
    n = pts.shape[0]
    if n == 1:
        return {"center": pts[0].copy(), "radius": 0.0}

    center, radius = _circle_from_two(pts[0], pts[1])

    def outside(pt: np.ndarray) -> bool:
        tol = 1e-12 * max(1.0, radius)
        return float(np.hypot(pt[0] - center[0], pt[1] - center[1])) > radius + tol

    for i in range(2, n):
        if not outside(pts[i]):
            continue
        center, radius = pts[i].copy(), 0.0
        for j in range(i):
            if not outside(pts[j]):
                continue
            center, radius = _circle_from_two(pts[i], pts[j])
            for k in range(j):
                if not outside(pts[k]):
                    continue
                center, radius = _circle_from_three(pts[i], pts[j], pts[k])

    return {"center": center, "radius": float(radius)}


def polygon_centroid(polygon: MatrixLike) -> Dict[str, Union[np.ndarray, float]]:
    """求简单多边形的**面积加权**质心（不是顶点平均值），并返回带符号面积。

    参数:
        polygon: 形状 (n, 2) 的多边形顶点，按顺序给出（顺/逆时针都行），n >= 3。

    返回:
        dict，键为：
        ``centroid``  形状 (2,) 的质心坐标；
        ``signed_area``  float，带符号面积（逆时针为正、顺时针为负）。

    算法:
        记 cross_i = x_i·y_{i+1} - x_{i+1}·y_i（下标按模 n 回绕），
        A = 0.5·Σ cross_i；
        Cx = Σ (x_i + x_{i+1})·cross_i / (6A)，Cy = Σ (y_i + y_{i+1})·cross_i / (6A)。
        退化判据：设 L 为包围盒对角线长度，若 L = 0 或 |A| <= 1e-12·L² 则抛 ValueError。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. **顶点平均 ≠ 质心**：把顶点坐标直接求平均只在正多边形等特殊情形下成立，
           对一般的"细长"或顶点疏密不均的多边形会明显偏。本函数用的是面积加权公式。
        2. **自交多边形的质心无意义**：分母里的带符号面积会被正负部分抵消；本函数
           只在 |A| 退化到接近 0 时报错，面积恰好抵消为 0 的自交图形（如对称"8 字"）
           会抛异常，但面积不为 0 的自交图形不会 —— 返回值不可解释，使用前应先检查
           多边形是否简单。
        3. 首尾重复写一次闭合点不影响结果（cross 贡献为 0），但会多计一个顶点。
        4. 阈值是**相对**的（1e-12·L²）：坐标整体缩放不改变结论，但极扁多边形
           （面积远小于 L²）会被判为退化并抛错，这是有意的保护。

    参考:
        多边形质心的标准鞋带公式（计算几何教材通例）。
    """
    pts = _as_points(polygon, "polygon")
    n = pts.shape[0]
    if n < 3:
        raise ValueError(f"polygon 至少需要 3 个顶点，得到 {n}")
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * float(np.sum(cross))
    extent = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
    if extent <= 0.0 or abs(signed_area) <= 1e-12 * extent * extent:
        raise ValueError(
            f"多边形退化（带符号面积 {signed_area}，包围盒对角线 {extent}），无法定义质心"
        )
    cx = float(np.sum((x + x_next) * cross)) / (6.0 * signed_area)
    cy = float(np.sum((y + y_next) * cross)) / (6.0 * signed_area)
    return {"centroid": np.array([cx, cy], dtype=float), "signed_area": signed_area}


def point_to_segment_distance(p: ArrayLike, a: ArrayLike, b: ArrayLike) -> float:
    """点到**线段**（不是直线）的最短欧氏距离。

    参数:
        p: 待测点，长度 2 的坐标 ``(x, y)``。
        a, b: 线段的两个端点，各为长度 2 的坐标。

    返回:
        float，最短距离（非负）。

    算法:
        设 ab = b-a。若 |ab|² = 0（线段退化为点）返回 |p-a|；
        否则 t = ((p-a)·ab) / |ab|²，把 t 截断到 [0, 1] 得最近点 a + t·ab，
        返回 |p - (a + t·ab)|。投影落在线段外时 t 被截断为 0 或 1，
        距离自动等于到较近端点的距离。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        1. **不要漏掉 t 的截断**：直接算 |(p-a)×ab|/|ab| 得到的是到**直线**的距离，
           点在端点外侧时会严重偏小（例如点到其正后方 100 km 的线段，直线距离可能是
           0，而真实距离是 100 km）。这类错误在做"最近道路距离"时后果很大。
        2. 经纬度不能直接代入：Δ经度的地面长度随纬度收缩，必须先投影或用球面距离。
        3. 本函数返回的是标量而非点坐标；需要最近点本身请自己按 t 重算。
        4. 若调用方已经知道点在线段所在直线的哪一侧，用截断形式仍然正确，无需分支。

    参考:
        点到线段距离的标准投影公式（计算几何教材通例）。
    """
    pv = as_vector(p, "p")
    av = as_vector(a, "a")
    bv = as_vector(b, "b")
    for name, v in (("p", pv), ("a", av), ("b", bv)):
        if v.size != 2:
            raise ValueError(f"{name} 必须是长度 2 的坐标，得到长度 {v.size}")
    ab = bv - av
    denom = float(np.dot(ab, ab))
    if denom <= 0.0:
        return float(np.hypot(pv[0] - av[0], pv[1] - av[1]))
    t = float(np.dot(pv - av, ab)) / denom
    t = min(1.0, max(0.0, t))
    proj = av + t * ab
    return float(np.hypot(pv[0] - proj[0], pv[1] - proj[1]))


def voronoi_nearest(points: MatrixLike, queries: MatrixLike) -> Dict[str, np.ndarray]:
    """Voronoi 最近站点查询（暴力版）：对每个查询点找欧氏距离最近的站点。

    参数:
        points: 站点坐标，形状 (n, 2)，n >= 1。
        queries: 查询点，形状 (m, 2)。

    返回:
        dict，键为：
        ``index``  形状 (m,) 的 int 数组，最近站点的下标；
        ``distance``  形状 (m,) 的 float 数组，对应的最近距离。

    算法:
        1. 构造 (m, n) 的距离矩阵 D_jk = ‖q_j - p_k‖。
        2. 每行取最小值下标。并列（距离完全相同）时 ``np.argmin`` 返回**最小下标**，
           这是本实现固定的口径：查询点落在 Voronoi 边界上时归给编号最小的站点。

    复杂度:
        时间 O(mn) / 空间 O(mn)。暴力实现，n、m 都上千时请改用 KD 树或 Delaunay 对偶。

    陷阱:
        1. **Voronoi 胞元只由最近距离定义，不含任何障碍/路网约束**：把站点当设施、
           查询点当需求点算"最近设施"时，直线距离会系统性低估实际通行距离。
        2. 并列口径必须写清楚：浮点误差会让"本应并列"的点倒向任一侧，边界附近的
           归属不可依赖；需要稳定归属时应显式加权重或改用距离排序后的规则。
        3. 站点重复（重合）时只会返回其中下标最小的那个，另一个永远不被查询到。
        4. 距离是平面欧氏的；用经纬度必须先换算成公里，否则高纬度处东西向距离被高估。

    参考:
        Voronoi, "Nouvelles applications des paramètres continus...", 1908；
        最近邻查询的通例（见 computational geometry 教材）。
    """
    pts = _as_points(points, "points")
    q = _as_points(queries, "queries")
    diff = q[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    idx = np.argmin(dist, axis=1).astype(int)
    return {
        "index": idx,
        "distance": dist[np.arange(q.shape[0]), idx],
    }


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict（int/float/list，固定种子下两次调用完全一致），共 46 个键，按族分组：

        - **凸包与多边形**（7 个）：``hull_size`` 单位正方形 + 4 个共线边中点 + 2 个内部点
          的凸包顶点数（应为 4）；``hull_indices`` 凸包顶点下标（升序）；
          ``polygon_area`` 三角形 (0,0)-(4,0)-(0,3) 的面积（应为 6.0）；
          ``square_area`` 单位正方形面积（应为 1.0）；``pip_inside`` / ``pip_outside`` /
          ``pip_on_edge`` 点包含判定（1/0）。
        - **距离与插值**（3 个）：``haversine_bj_sh`` 北京→上海大圆距离（km）；
          ``idw_at_sample`` 查询点与样本重合时的 IDW 值（应精确等于样本值）；
          ``idw_center`` 四角对称采样下中心点的 IDW 值。
        - **克里金**（7 个）：``krige_max_dev_at_sample`` 样本点处预测与观测的最大偏差
          （nugget=0，应≈0）；``krige_var_at_sample`` 样本点处克里金方差的最大值（应≈0）；
          ``krige_pred_center`` / ``krige_var_center`` 中心点的预测值与方差；
          ``krige_sill`` / ``krige_vrange`` 默认参数估计出的基台值与变程；
          ``krige_keys`` 返回值字典的键名（升序）。
        - **最小包围圆**（7 个）：``mec_triangle_radius`` / ``mec_triangle_center``
          边长 1 正三角形的外接圆半径与圆心；``mec_square_radius`` /
          ``mec_square_center`` 单位正方形的最小包围圆；``mec_square_covers_all``
          是否覆盖全部顶点（1/0）；``mec_two_point_center`` / ``mec_two_point_radius``
          两点退化情形（圆心为中点）。
        - **点到线段距离**（4 个）：``pt_seg_outside_right`` / ``pt_seg_outside_left``
          投影落在线段外两侧时应等于到端点的距离；``pt_seg_perp`` 垂直投影距离；
          ``pt_seg_degenerate`` 线段退化为一点时的距离。
        - **多边形质心**（4 个）：``centroid_square`` / ``centroid_square_signed_area``
          单位正方形质心与带符号面积；``centroid_triangle`` /
          ``centroid_triangle_signed_area`` 同上，取自 (0,0)-(4,0)-(0,3)。
        - **线段相交**（12 个）：``seg_parallel_intersect`` / ``seg_parallel_flag``
          平行不共线；``seg_collinear_intersect`` / ``seg_collinear_overlap`` /
          ``seg_collinear_point`` 共线重叠的中点；``seg_touch_point_overlap`` /
          ``seg_touch_point`` 共线仅接触一点；``seg_endpoint_intersect`` /
          ``seg_endpoint_point`` 端点重合；``seg_t_junction_point`` T 形接触；
          ``seg_cross_point`` 正常交叉；``seg_disjoint_intersect`` 完全分离。
        - **Voronoi 最近站点**（2 个）：``voronoi_index`` 最近站点下标（暴力法）；
          ``voronoi_distance`` 对应距离。

    算法:
        点集与数值全部硬编码，不含随机量，因此天然可复现；克里金用默认参数
        （sill=样本方差、变程=最大距离的一半）。

    复杂度:
        时间 O(n³)（n≈9）/ 空间 O(n²)。

    陷阱:
        断言"样本点处预测=观测"在本实现的口径下（γ(0)=0）对**任意 nugget** 都成立，
        实测 nugget=0.3 时最大偏差仍为 1.1e-16；如果你改成把块金放到 Γ 对角元的
        "观测误差"口径，这条断言就会（也应该）失效，不要把它当成 bug。

    参考:
        本模块各函数参考文献。
    """
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    edges = np.array([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]])
    inner = np.array([[0.5, 0.5], [0.2, 0.8]])
    cloud = np.vstack([square, edges, inner])
    hull = convex_hull(cloud)

    tri_area = polygon_area(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]))
    sq_area = polygon_area(square)

    pip_in = point_in_polygon([0.5, 0.5], square)
    pip_out = point_in_polygon([1.5, 0.5], square)
    pip_edge = point_in_polygon([0.5, 1.0], square)

    dist_bj_sh = haversine(39.9042, 116.4074, 31.2304, 121.4737)

    sample_pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sample_vals = np.array([1.0, 2.0, 4.0, 3.0])
    idw_sample = idw_interpolate(sample_pts, sample_vals, np.array([[1.0, 1.0]]), power=2.0)
    idw_center = idw_interpolate(sample_pts, sample_vals, np.array([[0.5, 0.5]]), power=2.0)

    krig = ordinary_kriging(sample_pts, sample_vals, sample_pts, model="spherical", nugget=0.0)
    dev = float(np.max(np.abs(krig["prediction"] - sample_vals)))
    krig_center = ordinary_kriging(sample_pts, sample_vals, np.array([[0.5, 0.5]]), model="spherical")
    vparams = estimate_variogram_params(sample_pts, sample_vals)

    # ---- 新增函数的闭式解 / 退化情形断言 ----
    # 1) 最小包围圆：正三角形与正方形的外接圆有闭式解
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    mec_tri = minimum_enclosing_circle(tri)
    closed_radius_tri = 1.0 / math.sqrt(3.0)  # 边长 1 的正三角形外接圆半径
    if abs(float(mec_tri["radius"]) - closed_radius_tri) > 1e-9:
        raise AssertionError(
            f"正三角形最小包围圆半径 {mec_tri['radius']} 与外接圆闭式解 {closed_radius_tri} 不符"
        )
    closed_center_tri = np.array([0.5, math.sqrt(3.0) / 6.0])
    if float(np.max(np.abs(mec_tri["center"] - closed_center_tri))) > 1e-9:
        raise AssertionError(f"正三角形最小包围圆圆心 {mec_tri['center']} 与闭式解 {closed_center_tri} 不符")
    mec_sq = minimum_enclosing_circle(square)
    closed_radius_sq = math.sqrt(2.0) / 2.0  # 单位正方形外接圆 = 半对角线
    if abs(float(mec_sq["radius"]) - closed_radius_sq) > 1e-9:
        raise AssertionError(
            f"正方形最小包围圆半径 {mec_sq['radius']} 与外接圆闭式解 {closed_radius_sq} 不符"
        )
    if float(np.max(np.abs(mec_sq["center"] - np.array([0.5, 0.5])))) > 1e-9:
        raise AssertionError(f"正方形最小包围圆圆心 {mec_sq['center']} 应为 (0.5, 0.5)")
    if not bool(np.all(np.hypot(*(square - mec_sq["center"]).T) <= float(mec_sq["radius"]) + 1e-12)):
        raise AssertionError("正方形最小包围圆未覆盖全部顶点")
    mec_two = minimum_enclosing_circle(np.array([[0.0, 0.0], [2.0, 0.0]]))

    # 2) 点到线段距离：投影落在线段外时必须等于到端点的距离
    d_outside = point_to_segment_distance([2.0, 0.0], [0.0, 0.0], [1.0, 0.0])
    if abs(d_outside - math.hypot(2.0 - 1.0, 0.0)) > 1e-12:
        raise AssertionError(f"点在投影外侧时距离 {d_outside} 应等于到端点 b 的距离 1.0")
    d_before = point_to_segment_distance([-3.0, 4.0], [0.0, 0.0], [1.0, 0.0])
    if abs(d_before - 5.0) > 1e-12:
        raise AssertionError(f"点在投影外侧时距离 {d_before} 应等于到端点 a 的距离 5.0")

    # 3) 多边形质心：单位正方形为 (0.5, 0.5)，带符号面积 1.0
    cent_sq = polygon_centroid(square)
    if float(np.max(np.abs(cent_sq["centroid"] - np.array([0.5, 0.5])))) > 1e-12:
        raise AssertionError(f"单位正方形质心 {cent_sq['centroid']} 应为 (0.5, 0.5)")
    if abs(float(cent_sq["signed_area"]) - 1.0) > 1e-12:
        raise AssertionError(f"逆时针单位正方形带符号面积 {cent_sq['signed_area']} 应为 1.0")

    # 4) 线段相交的三种退化情形（平行 / 共线重叠 / 端点相交）各有确定结论
    seg_par = segment_intersection([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0])
    if seg_par["intersect"] or not seg_par["parallel"] or seg_par["collinear_overlap"]:
        raise AssertionError(f"平行不共线线段判定错误：{seg_par}")
    if seg_par["point"] is not None:
        raise AssertionError("不相交线段的 point 必须为 None")
    seg_col = segment_intersection([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0])
    if not (seg_col["intersect"] and seg_col["parallel"] and seg_col["collinear_overlap"]):
        raise AssertionError(f"共线重叠线段判定错误：{seg_col}")
    if float(np.max(np.abs(seg_col["point"] - np.array([1.5, 0.0])))) > 1e-12:
        raise AssertionError(f"共线重叠区间中点 {seg_col['point']} 应为 (1.5, 0.0)")
    seg_touch = segment_intersection([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0])
    if not seg_touch["intersect"] or seg_touch["collinear_overlap"]:
        raise AssertionError(f"共线仅接触一点时 collinear_overlap 必须为 False：{seg_touch}")
    seg_end = segment_intersection([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0])
    if not seg_end["intersect"] or seg_end["parallel"]:
        raise AssertionError(f"端点相交判定错误：{seg_end}")
    if float(np.max(np.abs(seg_end["point"] - np.array([1.0, 0.0])))) > 1e-12:
        raise AssertionError(f"端点相交交点 {seg_end['point']} 应为 (1.0, 0.0)")
    seg_t = segment_intersection([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 1.0])
    if not seg_t["intersect"] or float(np.max(np.abs(seg_t["point"] - np.array([1.0, 0.0])))) > 1e-12:
        raise AssertionError(f"T 形端点落在另一线段内部时判定错误：{seg_t}")
    seg_cross = segment_intersection([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0])
    if float(np.max(np.abs(seg_cross["point"] - np.array([1.0, 1.0])))) > 1e-12:
        raise AssertionError(f"交叉线段交点 {seg_cross['point']} 应为 (1.0, 1.0)")
    seg_dis = segment_intersection([0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0])

    # 5) Voronoi 最近站点查询（暴力）：方形四角站点
    vor = voronoi_nearest(square, np.array([[0.6, 0.6], [0.1, 0.1]]))
    if [int(i) for i in vor["index"]] != [2, 0]:
        raise AssertionError(f"最近站点下标 {[int(i) for i in vor['index']]} 应为 [2, 0]")
    if abs(float(vor["distance"][0]) - math.hypot(0.4, 0.4)) > 1e-12:
        raise AssertionError(f"最近距离 {vor['distance'][0]} 应为 {math.hypot(0.4, 0.4)}")

    return {
        "hull_size": len(hull),
        "hull_indices": sorted(int(i) for i in hull),
        "polygon_area": round(float(tri_area), 6),
        "square_area": round(float(sq_area), 6),
        "pip_inside": int(pip_in),
        "pip_outside": int(pip_out),
        "pip_on_edge": int(pip_edge),
        "haversine_bj_sh": round(float(dist_bj_sh), 6),
        "idw_at_sample": round(float(idw_sample[0]), 8),
        "idw_center": round(float(idw_center[0]), 8),
        "krige_max_dev_at_sample": round(dev, 10),
        "krige_var_at_sample": round(float(np.max(krig["variance"])), 10),
        "krige_pred_center": round(float(krig_center["prediction"][0]), 6),
        "krige_var_center": round(float(krig_center["variance"][0]), 6),
        "krige_sill": round(float(vparams["sill"]), 6),
        "krige_vrange": round(float(vparams["vrange"]), 6),
        "krige_keys": sorted(krig.keys()),
        "mec_triangle_radius": round(float(mec_tri["radius"]), 9),
        "mec_triangle_center": [round(float(v), 9) for v in mec_tri["center"]],
        "mec_square_radius": round(float(mec_sq["radius"]), 9),
        "mec_square_center": [round(float(v), 9) for v in mec_sq["center"]],
        "mec_square_covers_all": int(
            bool(np.all(np.hypot(*(square - mec_sq["center"]).T) <= float(mec_sq["radius"]) + 1e-12))
        ),
        "mec_two_point_center": [round(float(v), 9) for v in mec_two["center"]],
        "mec_two_point_radius": round(float(mec_two["radius"]), 9),
        "pt_seg_outside_right": round(float(d_outside), 9),
        "pt_seg_outside_left": round(float(d_before), 9),
        "pt_seg_perp": round(float(point_to_segment_distance([0.5, 3.0], [0.0, 0.0], [1.0, 0.0])), 9),
        "pt_seg_degenerate": round(
            float(point_to_segment_distance([3.0, 4.0], [1.0, 1.0], [1.0, 1.0])), 9
        ),
        "centroid_square": [round(float(v), 12) for v in cent_sq["centroid"]],
        "centroid_square_signed_area": round(float(cent_sq["signed_area"]), 12),
        "centroid_triangle": [
            round(float(v), 12)
            for v in polygon_centroid(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]))["centroid"]
        ],
        "centroid_triangle_signed_area": round(
            float(polygon_centroid(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]))["signed_area"]), 12
        ),
        "seg_parallel_intersect": int(seg_par["intersect"]),
        "seg_parallel_flag": int(seg_par["parallel"]),
        "seg_collinear_intersect": int(seg_col["intersect"]),
        "seg_collinear_overlap": int(seg_col["collinear_overlap"]),
        "seg_collinear_point": [round(float(v), 12) for v in seg_col["point"]],
        "seg_touch_point_overlap": int(seg_touch["collinear_overlap"]),
        "seg_touch_point": [round(float(v), 12) for v in seg_touch["point"]],
        "seg_endpoint_intersect": int(seg_end["intersect"]),
        "seg_endpoint_point": [round(float(v), 12) for v in seg_end["point"]],
        "seg_t_junction_point": [round(float(v), 12) for v in seg_t["point"]],
        "seg_cross_point": [round(float(v), 12) for v in seg_cross["point"]],
        "seg_disjoint_intersect": int(seg_dis["intersect"]),
        "voronoi_index": [int(i) for i in vor["index"]],
        "voronoi_distance": [round(float(v), 9) for v in vor["distance"]],
    }
