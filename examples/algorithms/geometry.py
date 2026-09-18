"""几何与空间分析：凸包、多边形面积、点在多边形内、球面距离、IDW 插值、普通克里金。

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
from typing import Dict, List, Optional, Sequence, Union

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
        sill: 总基台值；默认取样本方差（``np.var(values)``）。传 None 表示自动估计。
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
        ``sill`` 总基台值（样本方差，ddof=0）；
        ``vrange`` 变程（样本点最大两两距离的一半）；
        ``nugget`` 原样回传的块金常数。

    算法:
        粗估法：sill = var(values)，vrange = max‖xi-xj‖ / 2。这只是让一行调用能跑起来的
        默认值，不是拟合结果。

    复杂度:
        时间 O(n²) / 空间 O(n²)。

    陷阱:
        这个估计**没有任何统计学最优性**：样本点分布范围越大，变程估计越大，
        预测越平滑；要得到可靠参数必须拟合实验变异函数或做交叉验证。

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


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict（int/float/list），固定种子下两次调用完全一致：
        ``hull_size`` 单位正方形 + 2 个内部点 + 4 个共线边中点的凸包顶点数（应为 4）；
        ``hull_indices`` 凸包顶点下标（升序，便于断言）；
        ``polygon_area`` 三角形 (0,0)-(4,0)-(0,3) 的面积（应为 6.0）；
        ``square_area`` 单位正方形面积（应为 1.0）；
        ``pip_inside`` / ``pip_outside`` / ``pip_on_edge`` 点包含判定（1/0）；
        ``haversine_bj_sh`` 北京→上海的大圆距离（km）；
        ``idw_at_sample`` 查询点与样本重合时的插值值（应精确等于样本值）；
        ``idw_center`` 正方形四角对称采样时中心点的 IDW 值；
        ``krige_max_dev_at_sample`` 在样本点处预测值与观测值的最大偏差（nugget=0，应≈0）；
        ``krige_var_at_sample`` 样本点处克里金方差的最大值（应≈0）；
        ``krige_pred_center`` 中心点的克里金预测值；
        ``krige_var_center`` 中心点的克里金方差；
        ``krige_sill`` / ``krige_vrange`` 默认参数估计出的基台值与变程。

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
    }
