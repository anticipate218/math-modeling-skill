"""空间与物理场建模：一维热传导有限差分（显式 FTCS / Crank-Nicolson）、二维泊松 SOR、
森林火灾元胞自动机、Nagel-Schreckenberg 交通流、Buckingham π 量纲分析、比例缩放相似换算。

本模块共 7 个公开函数，按用途分成三组：

- 偏微分方程数值解：``heat_equation_1d_explicit``（FTCS，需 ``alpha*dt/dx**2 <= 0.5``）、
  ``heat_equation_1d_implicit``（Crank-Nicolson，无条件稳定）、``poisson_2d_sor``
  （二维泊松方程红黑 SOR 迭代）；
- 元胞自动机：``forest_fire_ca``（林火蔓延）、``traffic_ca_nagel_schreckenberg``（交通流基本图）；
- 量纲与相似：``buckingham_pi``（π 定理求无量纲组）、``scaling_similarity``（相似准则换算）。

本模块是**教学透明版**：网格小、格式简单、迭代次数显式可数，目的是让论文能写清
"我们用了什么离散格式、稳定条件是什么、误差怎么估"。真正的工程计算请换专用求解器
（有限元、多重网格、``scipy.sparse.linalg``）；本模块只依赖 numpy 与标准库。

关键约定
--------
- 热传导方程一律写作 ``u_t = alpha * u_xx``；网格**包含两个端点**，初值 ``u0`` 的长度
  就是网格点数 n。边界只支持 ``"dirichlet"``（端点值固定为 ``u0`` 的端点值）与
  ``"neumann"``（零通量，用镜像虚拟节点实现，因此离散总热量精确守恒）。
- 显式格式是 FTCS（时间一阶、空间二阶），隐式格式是 Crank-Nicolson（时间二阶），
  两者共用同一套零通量边界口径，所以同一个 ``alpha, dx, dt`` 下二者可直接对比。
- 泊松方程：单位方形域 ``-Δu = f``，零 Dirichlet 边界，n 个**内点**，步长 ``h = 1/(n+1)``；
  ``f`` 与返回值 ``u`` 都只覆盖内点（形状 (n, n)），边界恒为 0，不出现在返回值里。
- 元胞自动机与交通流都是**同步更新**（同一时刻的所有格子同时变化）；网格外一律视为
  "非燃烧" / "空"格子（非周期边界）。
- 量纲分析用**精确有理数**做行最简形求零空间，π 组按"整数、互质、首个非零指数为 +1"
  规范化，因此同一输入永远给出同一组 π（不依赖浮点 SVD 的符号/尺度）。
- 随机性一律走 ``_common.rng``，绝不使用 ``np.random.*`` 全局函数。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._common import as_matrix, as_vector, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "heat_equation_1d_explicit",
    "heat_equation_1d_implicit",
    "poisson_2d_sor",
    "forest_fire_ca",
    "traffic_ca_nagel_schreckenberg",
    "buckingham_pi",
    "scaling_similarity",
]

#: 支持的边界类型。
_BC_TYPES = ("dirichlet", "neumann")


def _check_heat_inputs(
    u0: ArrayLike,
    alpha: float,
    dx: float,
    dt: float,
    n_steps: int,
    bc: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, str]]:
    """内部工具：校验热传导格式的公共输入，返回 (初值副本, (左边界, 右边界))。

    校验收敛性之外的一切：网格点数、正系数、整数步数、边界类型字符串。
    稳定性条件（``alpha*dt/dx**2 <= 0.5``）由显式格式自己检查，隐式格式不需要。
    """
    u = as_vector(u0, "u0")
    if u.size < 3:
        raise ValueError(f"u0 至少要有 3 个网格点（含两个端点），得到 {u.size} 个")
    if alpha <= 0:
        raise ValueError(f"alpha 必须 > 0，得到 {alpha}")
    if dx <= 0:
        raise ValueError(f"dx 必须 > 0，得到 {dx}")
    if dt <= 0:
        raise ValueError(f"dt 必须 > 0，得到 {dt}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError(f"n_steps 必须是非负整数，得到 {n_steps!r}")
    if len(tuple(bc)) != 2:
        raise ValueError(f"bc 必须是长度为 2 的序列，得到 {bc!r}")
    left, right = (str(bc[0]), str(bc[1]))
    for side, name in ((left, "bc[0]"), (right, "bc[1]")):
        if side not in _BC_TYPES:
            raise ValueError(f"{name} 只支持 {_BC_TYPES}，得到 {side!r}")
    return u.copy(), (left, right)


def heat_equation_1d_explicit(
    u0: ArrayLike,
    alpha: float,
    dx: float,
    dt: float,
    n_steps: int,
    bc: Sequence[str] = ("dirichlet", "dirichlet"),
) -> dict:
    """一维热传导方程的显式 FTCS 格式（时间前向、空间中心差分）。

    参数:
        u0: 初始温度场，形状 (n,)，**含两个端点**；n >= 3。
        alpha: 热扩散系数（> 0）。
        dx: 空间步长（> 0）。
        dt: 时间步长（> 0）。
        n_steps: 迭代步数（非负整数）。
        bc: ``(左边界类型, 右边界类型)``，取值为 ``"dirichlet"``（端点值固定为 u0 的端点值）
            或 ``"neumann"``（零通量）。

    返回:
        dict，键为：
        ``u``  形状 (n,) 的终态温度场；
        ``stability_ratio``  float，r = alpha*dt/dx**2。

    算法:
        1. 网格点 i = 0..n-1 等间距，内部点用二阶中心差分：
           u_i^{k+1} = u_i^k + r (u_{i-1}^k - 2 u_i^k + u_{i+1}^k)，r = alpha*dt/dx^2。
        2. 边界：Dirichlet 直接把端点钉在 u0[0] / u0[-1]；
           零通量用镜像虚拟节点 u_{-1} = u_0、u_n = u_{n-1}，于是端点更新为
           u_0^{k+1} = u_0^k + r (u_1^k - u_0^k)（右端同理）。
        3. 稳定性：显式格式要求 r <= 0.5，否则抛 ValueError（**绝不静默发散**）。

    复杂度:
        时间 O(n_steps * n) / 空间 O(n)。

    陷阱:
        1. **r > 0.5 必然发散**，而且前几步看起来还正常，等发现时已经溢出；本函数直接报错
           而不是返回一堆 inf。
        2. 时间只有一阶精度：r 取得很小时误差 ~O(dt)+O(dx^2)。想让显式与 Crank-Nicolson
           对得上，必须把 dt 压到远小于 dx^2/alpha 的量级（见 ``_self_test``）。
        3. 零通量口径用的是"镜像虚拟节点 + 端点也参与更新"，只有这样离散总热量才严格守恒；
           若写成"端点复制邻居值"（u_0 = u_1）会引入一阶边界误差并破坏守恒。
        4. Dirichlet 边界锁的是 u0 的端点值，不是每次迭代后的端点值；中途改边界要另写接口。

    参考:
        Fourier 1822 的热传导方程；FTCS 稳定性条件见 Richtmyer & Morton,
        "Difference Methods for Initial-Value Problems", 1967。
    """
    u, (left, right) = _check_heat_inputs(u0, alpha, dx, dt, n_steps, bc)
    u_init = u.copy()
    ratio = float(alpha * dt / (dx * dx))
    if ratio > 0.5 + 1e-12:
        raise ValueError(
            f"显式 FTCS 要求稳定性比 alpha*dt/dx**2 <= 0.5，得到 {ratio}；"
            f"请减小 dt 或增大 dx，或改用 heat_equation_1d_implicit"
        )
    for _ in range(int(n_steps)):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + ratio * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        if left == "dirichlet":
            u_new[0] = u_init[0]
        else:
            u_new[0] = u[0] + ratio * (u[1] - u[0])
        if right == "dirichlet":
            u_new[-1] = u_init[-1]
        else:
            u_new[-1] = u[-1] + ratio * (u[-2] - u[-1])
        u = u_new
    return {"u": u, "stability_ratio": ratio}


def heat_equation_1d_implicit(
    u0: ArrayLike,
    alpha: float,
    dx: float,
    dt: float,
    n_steps: int,
    bc: Sequence[str] = ("dirichlet", "dirichlet"),
) -> dict:
    """一维热传导方程的 Crank-Nicolson 隐式格式，每步解一个三对角线性系统。

    参数:
        u0: 初始温度场，形状 (n,)，含两个端点；n >= 3。
        alpha: 热扩散系数（> 0）。
        dx: 空间步长（> 0）。
        dt: 时间步长（> 0）；**不受稳定性限制**，可以远大于 dx**2/(2*alpha)。
        n_steps: 迭代步数（非负整数）。
        bc: 同 ``heat_equation_1d_explicit``。

    返回:
        dict，键为：
        ``u``  形状 (n,) 的终态温度场；
        ``matrix``  形状 (n, n) 的稠密系数矩阵 M（左端矩阵，满足 M u^{k+1} = b(u^k)）。
        Dirichlet 行是单位行（端点值被钉住），零通量行的系数为 1 + r/2 与 -r/2。

    算法:
        1. Crank-Nicolson（时间二阶、空间二阶）：对内部点
           (1 + r) u_i^{k+1} - (r/2)(u_{i-1}^{k+1} + u_{i+1}^{k+1})
             = (1 - r) u_i^k + (r/2)(u_{i-1}^k + u_{i+1}^k)，r = alpha*dt/dx^2。
        2. 边界用与显式格式相同的镜像虚拟节点口径，端点的 CN 方程为
           (1 + r/2) u_0^{k+1} - (r/2) u_1^{k+1} = (1 - r/2) u_0^k + (r/2) u_1^k。
        3. 矩阵 M 与时间无关，先构造稠密矩阵，每步用 ``np.linalg.solve`` 求解。
        4. 采用**无条件稳定**的 CN 格式：任意 r > 0 都不会发散，且满足极值原理
           （解不会超过初值的最大绝对值）。

    复杂度:
        时间 O(n_steps * n^3)（每步一次稠密解方程；实际三对角可用 Thomas 算法降到 O(n)）
        / 空间 O(n^2)。

    陷阱:
        1. 无条件稳定**不等于**无条件精确：dt 很大时时间精度退化为 O(1)，解会"过度平滑"
           （把瞬态抹掉）。稳定性与精度是两件事。
        2. 每一行都必须有对角占优的符号结构；若把符号写反（+r/2 放到左端），矩阵会变成
           非对角占优，解会出现非物理振荡。
        3. 这里每步都重新解一次方程（O(n^3)），只是为了让代码短；生产代码应当先做一次
           LU 分解（或直接用 Thomas 算法）复用。
        4. 返回的 ``matrix`` **只是左端系数矩阵**，它并不携带边界数据：Dirichlet 端点值
           u0[0] / u0[-1] 是每一步单独写进右端向量 b 的（M 的 Dirichlet 行被置成单位行）。
           因此把 M 单独拿出来配一个自造的右端（例如全 0）时，首末两个元素会静默解成 0，
           而不是被钉住的温度；复用 M 必须自己把首末行右端设成 u0[0] / u0[-1]。

    参考:
        Crank & Nicolson, "A practical method for numerical evaluation of solutions of
        partial differential equations of the heat-conduction type", 1947。
    """
    u, (left, right) = _check_heat_inputs(u0, alpha, dx, dt, n_steps, bc)
    u_init = u.copy()
    r = float(alpha * dt / (dx * dx))
    n = u.size

    m = np.zeros((n, n), dtype=float)
    rhs_op = np.zeros((n, n), dtype=float)
    for i in range(1, n - 1):
        m[i, i] = 1.0 + r
        m[i, i - 1] = -0.5 * r
        m[i, i + 1] = -0.5 * r
        rhs_op[i, i] = 1.0 - r
        rhs_op[i, i - 1] = 0.5 * r
        rhs_op[i, i + 1] = 0.5 * r
    if left == "dirichlet":
        m[0, 0] = 1.0
    else:
        m[0, 0] = 1.0 + 0.5 * r
        m[0, 1] = -0.5 * r
        rhs_op[0, 0] = 1.0 - 0.5 * r
        rhs_op[0, 1] = 0.5 * r
    if right == "dirichlet":
        m[n - 1, n - 1] = 1.0
    else:
        m[n - 1, n - 1] = 1.0 + 0.5 * r
        m[n - 1, n - 2] = -0.5 * r
        rhs_op[n - 1, n - 1] = 1.0 - 0.5 * r
        rhs_op[n - 1, n - 2] = 0.5 * r

    for _ in range(int(n_steps)):
        b = rhs_op @ u
        if left == "dirichlet":
            b[0] = u_init[0]
        if right == "dirichlet":
            b[n - 1] = u_init[n - 1]
        u = np.linalg.solve(m, b)
    return {"u": u, "matrix": m}


def _poisson_residual(u: np.ndarray, h2f: np.ndarray) -> float:
    """内部工具：五点差分方程的残差无穷范数 max|4u_ij - 四邻域和 - h^2 f_ij|。

    邻域用零填充（边界 u=0），因此 u 只包含内点。
    """
    nb = np.zeros_like(u)
    nb[1:, :] += u[:-1, :]
    nb[:-1, :] += u[1:, :]
    nb[:, 1:] += u[:, :-1]
    nb[:, :-1] += u[:, 1:]
    return float(np.max(np.abs(4.0 * u - nb - h2f)))


def poisson_2d_sor(
    n: int,
    f: MatrixLike,
    omega: float = 1.5,
    tol: float = 1e-8,
    max_iter: int = 5000,
) -> dict:
    """单位方形域上 ``-Δu = f``（零 Dirichlet 边界）的红黑 SOR 迭代解。

    参数:
        n: 每个方向的**内点**个数（>= 2），步长 h = 1/(n+1)。
        f: 右端项，形状 (n, n)，取值在内点网格上。
        omega: 松弛因子，必须落在 [1, 2]；1 即 Gauss-Seidel，接近 2/(1+sin(πh)) 时最快。
        tol: 收敛判据，残差 ``max|4u_ij - 四邻域和 - h^2 f_ij| <= tol`` 时停止。
        max_iter: 最大迭代次数（>= 1）。

    返回:
        dict，键为：
        ``u``  形状 (n, n) 的内点解（边界恒为 0，不在返回值里）；
        ``n_iter``  实际迭代次数（int）；
        ``residual``  最后一次迭代后的残差（float）。

    算法:
        1. 五点差分：-(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_ij)/h^2 = f_ij，
           整理成 u_ij = (四邻域和 + h^2 f_ij)/4。
        2. 红黑（red-black）SOR：把 (i+j) 的奇偶作为两种颜色，同一颜色的节点互不相邻，
           于是可以整块更新 u_red = (1-ω)u_red + ω/4 (四邻域和 + h^2 f)，
           再更新 black。其不动点与逐点 Gauss-Seidel SOR 完全一致，但可向量化。
        3. 每步计算残差的无穷范数，达到 tol 即提前退出；若 max_iter 步后仍未达标，
           抛 ValueError（不返回一个"看着像解"的结果）。

    复杂度:
        时间 O(n_iter * n^2) / 空间 O(n^2)。

    陷阱:
        1. **ω 不是越大越快**：ω 超过最优值后收敛变慢甚至发散；最优值随网格变细趋近 2，
           实践中最优 ω ≈ 2/(1+sin(π/(n+1)))，本文默认 1.5 只是为了稳健。
        2. 残差的量纲是"h^2 倍方程残差"（没有除以 h^2），所以同样的 tol 在更细的网格上
           对应更松的真实精度；跨网格比较收敛性时要统一口径。
        3. 迭代法对高频误差收敛极快、对低频误差极慢；网格越细迭代次数 ~O(n)，
           大规模问题必须上多重网格或直接解法。
        4. 只有零 Dirichlet 边界；非零边界值要把边界项移到右端，本函数没有这个接口。
        5. 若 f 与网格不匹配（形状不是 (n, n)）会直接报错——注意传入的是**内点**采样值。

    参考:
        Young, "Iterative Solution of Large Linear Systems", 1971（SOR 理论）；
        Press et al., "Numerical Recipes"（红黑 SOR 的并行化）。
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n 必须是 >= 2 的整数，得到 {n!r}")
    n = int(n)
    fmat = as_matrix(f, "f")
    if fmat.shape != (n, n):
        raise ValueError(f"f 的形状必须与 (n, n) = ({n}, {n}) 一致，得到 {fmat.shape}")
    if not (1.0 <= float(omega) <= 2.0):
        raise ValueError(f"omega 必须落在 [1, 2]，得到 {omega}")
    if tol <= 0:
        raise ValueError(f"tol 必须 > 0，得到 {tol}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f"max_iter 必须是 >= 1 的整数，得到 {max_iter!r}")

    h = 1.0 / (n + 1)
    h2f = h * h * fmat
    u = np.zeros((n, n), dtype=float)
    idx = np.arange(n)
    parity = (idx[:, None] + idx[None, :]) % 2  # 0/1 两色
    om = float(omega)

    n_iter = 0
    residual = float("inf")
    for it in range(1, int(max_iter) + 1):
        for color in (0, 1):
            nb = np.zeros_like(u)
            nb[1:, :] += u[:-1, :]
            nb[:-1, :] += u[1:, :]
            nb[:, 1:] += u[:, :-1]
            nb[:, :-1] += u[:, 1:]
            mask = parity == color
            u[mask] = (1.0 - om) * u[mask] + om * 0.25 * (nb[mask] + h2f[mask])
        residual = _poisson_residual(u, h2f)
        n_iter = it
        if residual <= tol:
            break
    else:
        raise ValueError(
            f"SOR 在 max_iter={max_iter} 步内未收敛（n={n}, omega={omega}, "
            f"tol={tol}，最终残差 {residual}）；请增大 max_iter 或调整 omega"
        )
    return {"u": u, "n_iter": int(n_iter), "residual": residual}


def forest_fire_ca(
    n: int = 30,
    p_grow: float = 0.05,
    p_light: float = 0.3,
    n_steps: int = 50,
    seed: Optional[int] = None,
    neighborhood: str = "moore",
    initial_density: float = 0.6,
) -> dict:
    """森林火灾元胞自动机（Drossel-Schwabl 型）：空地长树、树被雷击或邻火点燃。

    参数:
        n: 方形网格边长（>= 2），共 n*n 个格子。
        p_grow: 每个空地每步长出树的概率，取值 [0, 1]。
        p_light: 每棵**未被邻火波及**的树每步被雷击点燃的概率，取值 [0, 1]。
        n_steps: 演化步数（非负整数）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        neighborhood: ``"moore"``（8 邻域，默认）或 ``"von_neumann"``（4 邻域），
            决定火势蔓延范围；网格外视为非燃烧。
        initial_density: 初始时刻每个格子是树的概率，取值 [0, 1]；初始没有燃烧的格子。

    返回:
        dict，键为：
        ``steps``  长度 n_steps 的 list[float]，第 k 项是第 k 步之后的树木占比；
        ``tree_ratio``  float，终态树木占比（= final_trees / (n*n)）；
        ``burned_total``  int，累计**点燃次数**（同一格烧完变空地后若重新长树再被点燃会再计一次）；
        ``final_trees``  int，终态的树格数。

    算法:
        状态 0 = 空地、1 = 树、2 = 燃烧。每一步**同步**执行：
        1. 所有燃烧格变为空地（燃烧只持续一步）。
        2. 树格若 8 邻域（或 4 邻域）内存在燃烧格，则被点燃；否则以 p_light 的概率被雷击点燃。
        3. 空地以 p_grow 的概率长出树。
        4. 记录树木占比；把本步新点燃的格数累加到 burned_total。

    复杂度:
        时间 O(n_steps * n^2) / 空间 O(n^2)。

    陷阱:
        1. **必须同步更新**：如果就地更新（先烧的格子立刻去点燃邻居），火势会在一"步"内
           传遍整个网格，永远看不到蔓延过程。本实现先算掩码再整体赋值。
        2. p_grow 很小而 p_light 很大时系统会落到"树刚长出来就被烧掉"的稀疏稳态，
           树占比极低；想看自组织临界性要把 p_grow/p_light 的比值调到 1e-3 量级并跑很长时间。
        3. 邻域在网格外按"非燃烧"处理（非周期），所以边界上的火势蔓延比内部慢，
           小网格下边界效应会明显压低燃烧总量。
        4. 每步固定消耗两片 n*n 的随机数（长树、雷击各一片），删除或增加一次抽样都会
           改变后续所有随机数，从而改变结果——这也是它能通过确定性自测的原因。

    参考:
        Drossel & Schwabl, "Self-organized critical forest-fire model",
        Physical Review Letters, 1992。
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n 必须是 >= 2 的整数，得到 {n!r}")
    n = int(n)
    if not (0.0 <= p_grow <= 1.0):
        raise ValueError(f"p_grow 必须落在 [0, 1]，得到 {p_grow}")
    if not (0.0 <= p_light <= 1.0):
        raise ValueError(f"p_light 必须落在 [0, 1]，得到 {p_light}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError(f"n_steps 必须是非负整数，得到 {n_steps!r}")
    if not (0.0 <= initial_density <= 1.0):
        raise ValueError(f"initial_density 必须落在 [0, 1]，得到 {initial_density}")
    if neighborhood not in ("moore", "von_neumann"):
        raise ValueError(
            f"neighborhood 只支持 'moore'/'von_neumann'，得到 {neighborhood!r}"
        )

    gen = rng(seed)
    state = np.where(gen.random((n, n)) < initial_density, 1, 0).astype(np.int64)
    burned_total = 0
    ratios: List[float] = []

    offsets = [
        (di, dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
        and not (neighborhood == "von_neumann" and di != 0 and dj != 0)
    ]

    for _ in range(int(n_steps)):
        burning = state == 2
        pad = np.zeros((n + 2, n + 2), dtype=bool)
        pad[1:-1, 1:-1] = burning
        near = np.zeros((n, n), dtype=bool)
        for di, dj in offsets:
            near |= pad[1 + di : 1 + di + n, 1 + dj : 1 + dj + n]

        draw_light = gen.random((n, n))
        draw_grow = gen.random((n, n))
        is_tree = state == 1
        ignite = is_tree & (near | (draw_light < p_light))
        grow = (state == 0) & (draw_grow < p_grow)

        new_state = state.copy()
        new_state[burning] = 0
        new_state[ignite] = 2
        new_state[grow] = 1
        state = new_state
        burned_total += int(np.count_nonzero(ignite))
        ratios.append(float(np.count_nonzero(state == 1)) / float(n * n))

    final_trees = int(np.count_nonzero(state == 1))
    return {
        "steps": ratios,
        "tree_ratio": final_trees / float(n * n),
        "burned_total": int(burned_total),
        "final_trees": final_trees,
    }


def traffic_ca_nagel_schreckenberg(
    n_cells: int = 100,
    n_cars: int = 20,
    v_max: int = 5,
    p_brake: float = 0.3,
    n_steps: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """Nagel-Schreckenberg 一维交通流元胞自动机（周期性边界）。

    参数:
        n_cells: 环形道路的格子数（>= 2）。
        n_cars: 车辆数，必须满足 1 <= n_cars < n_cells。
        v_max: 最大速度（每步最多前进的格数，>= 1 的整数）。
        p_brake: 随机慢化概率，取值 [0, 1]（0 即确定性 NaSch）。
        n_steps: 演化步数（>= 1）。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``flow``  float，最后 min(20, n_steps) 步的平均流量，口径为
            "每步移动的车辆数 / 格子总数"（车/格/步）；乘以 n_cells 就是车/步。
        ``density``  float，密度 n_cars / n_cells。
        ``mean_speed``  float，同一时间窗内的平均车速（每车每步前进的格数）。
        ``steps``  长度 n_steps 的 list[float]，每步的瞬时流量。

    算法:
        每步对全部车辆**同步**执行 NaSch 四规则（顺序不能换）：
        1. 加速：v <- min(v + 1, v_max)；
        2. 减速：v <- min(v, gap)，gap 为到前车的空格数；
        3. 随机慢化：以概率 p_brake 令 v <- max(v - 1, 0)；
        4. 前进：x <- (x + v) mod n_cells。
        初始位置用 ``rng.choice`` 不放回抽取后排序，初始速度全 0。
        由于规则 2 保证 v <= gap，车辆不会互相超越，位置序列始终有序。

    复杂度:
        时间 O(n_steps * n_cars * log n_cars)：每步除 O(n_cars) 的四条规则更新外，还要一次
        ``np.argsort`` 把位置重新排回有序（否则 ``np.roll`` 的"前车"就指错了），
        这一项在 n_cars 大时是主导；空间 O(n_cars)。

    陷阱:
        1. **规则顺序不能变**：先随机慢化再加速会得到完全不同的基本图（"慢化"变成了
           "加速前的抖动"），流量-密度曲线会整体偏移。
        2. 平均车速有一个硬上界 ``(n_cells - n_cars) / n_cars``（因为 v_i <= gap_i 且
           全部 gap 之和恰好是空格总数）。所以当密度 ρ > 1/(v_max+1)（例如 v_max=5 时
           ρ > 1/6）时，**平均车速不可能接近 v_max**，这是基本图的自由流分支上界，
           不是实现 bug。想验证"自由流接近 v_max"必须用 ρ <= 1/(v_max+1)。
        3. p_brake > 0 时会出现"幽灵堵车"（自发拥堵），流量-密度曲线在中等密度处下降；
           单次仿真的瞬时流量波动很大，必须用时间窗平均（本函数用最后 20 步）。
        4. 初始速度全 0 会让前若干步处于加速瞬态；比较不同参数时必须用相同的步数与
           相同的时间窗，否则会把瞬态差异当成参数效应。

    参考:
        Nagel & Schreckenberg, "A cellular automaton model for freeway traffic",
        Journal de Physique I, 1992。
    """
    if isinstance(n_cells, bool) or not isinstance(n_cells, (int, np.integer)) or n_cells < 2:
        raise ValueError(f"n_cells 必须是 >= 2 的整数，得到 {n_cells!r}")
    n_cells = int(n_cells)
    if isinstance(n_cars, bool) or not isinstance(n_cars, (int, np.integer)):
        raise ValueError(f"n_cars 必须是整数，得到 {n_cars!r}")
    n_cars = int(n_cars)
    if not (1 <= n_cars < n_cells):
        raise ValueError(f"n_cars 必须满足 1 <= n_cars < n_cells={n_cells}，得到 {n_cars}")
    if isinstance(v_max, bool) or not isinstance(v_max, (int, np.integer)) or v_max < 1:
        raise ValueError(f"v_max 必须是 >= 1 的整数，得到 {v_max!r}")
    v_max = int(v_max)
    if not (0.0 <= p_brake <= 1.0):
        raise ValueError(f"p_brake 必须落在 [0, 1]，得到 {p_brake}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps 必须是 >= 1 的整数，得到 {n_steps!r}")
    n_steps = int(n_steps)

    gen = rng(seed)
    pos = np.sort(gen.choice(n_cells, size=n_cars, replace=False)).astype(np.int64)
    vel = np.zeros(n_cars, dtype=np.int64)
    flows: List[float] = []
    speeds: List[float] = []

    for _ in range(n_steps):
        gap = (np.roll(pos, -1) - pos - 1) % n_cells
        vel = np.minimum(vel + 1, v_max)
        vel = np.minimum(vel, gap)
        slow = gen.random(n_cars) < p_brake
        vel = np.where(slow, np.maximum(vel - 1, 0), vel)
        pos = (pos + vel) % n_cells
        order = np.argsort(pos, kind="stable")
        pos = pos[order]
        vel = vel[order]
        flows.append(float(np.sum(vel)) / float(n_cells))
        speeds.append(float(np.mean(vel)))

    window = min(20, n_steps)
    flow = float(np.mean(flows[-window:]))
    mean_speed = float(np.mean(speeds[-window:]))
    return {
        "flow": flow,
        "density": n_cars / float(n_cells),
        "mean_speed": mean_speed,
        "steps": flows,
    }


def _canonical_int_vector(vec: Sequence[Fraction]) -> List[int]:
    """内部工具：把有理数向量化成"整数、互质、首个非零分量为 +1"的规范形式。

    π 组的指数向量只有整体比例有意义（π^c 与 π 等价），必须固定一个规范代表，
    否则同一物理问题会因为求解器不同而给出符号/倍数不同的 π 组。
    """
    den = 1
    for x in vec:
        den = den * x.denominator // math.gcd(den, x.denominator)
    ints = [int(x * den) for x in vec]
    g = 0
    for v in ints:
        g = math.gcd(g, abs(v))
    if g > 1:
        ints = [v // g for v in ints]
    for v in ints:
        if v != 0:
            if v < 0:
                ints = [-x for x in ints]
            break
    return ints


def buckingham_pi(dims: Dict[str, Sequence[float]]) -> dict:
    """Buckingham π 定理：由量纲指数矩阵求全部独立无量纲组。

    参数:
        dims: ``{变量名: [各基本量纲指数]}``，例如阻力问题
            ``{"F": [1, 1, -2], "v": [0, 1, -1], "rho": [1, -3, 0],
            "mu": [1, -1, -1], "L": [0, 1, 0]}``（基本量纲顺序取 M, L, T）。
            所有变量的指数向量长度必须一致，至少 2 个变量。

    返回:
        dict，键为：
        ``rank``  量纲矩阵的秩（int）；
        ``n_pi``  独立无量纲组个数 = 变量数 - rank（int）；
        ``pi_groups``  list，每项是长度为"变量数"的整数指数向量（变量顺序 = dims 的键顺序）；
        ``pi_names``  对应的可读字符串，例如 ``"pi_2 = F^1 * rho^-1 * v^-2 * L^-2"``。

    算法:
        1. 把 dims 组装成量纲矩阵 D（行 = 基本量纲，列 = 物理变量）。
        2. 用 ``fractions.Fraction`` 对 D 做**精确有理数**行最简形（RREF），主元列数为 rank，
           自由列数为 n_pi = 变量数 - rank。
        3. 每个自由列 c 对应一个零空间基向量：x_c = 1，主元列 p_i 上取 x_{p_i} = -RREF[i, c]，
           其余为 0；对基向量按"约去分母 → 除以最大公约数 → 首个非零分量取正"规范化成整数。
        4. 这等价于用 SVD / Gram 矩阵求零空间，但精确算术保证指数是整数且结果可复现。

    复杂度:
        时间 O(k * n^2)（k 为基本量纲数，n 为变量数，精确有理数运算）
        / 空间 O(k * n)。

    陷阱:
        1. **基本量纲必须线性无关且完整**：若把 M 和"力"同时当基本量纲（力本身 = MLT^-2），
           量纲矩阵会降秩，n_pi 会偏大，给出的 π 组没有物理意义。
        2. π 组不唯一：任何 π 组的乘积/幂仍是 π 组。本函数固定了规范化代表（整数、互质、
           首个非零指数为 +1），但换一个变量顺序或换主元列就会得到另一组同样正确的基。
           论文里报 π 组时必须同时报变量顺序与这批规范化约定。
        3. 若 rank = 0（所有变量都是无量纲的），没有任何 π 组可求，直接报 ValueError。
        4. 输入指数必须是**相对同一组基本量纲**的；混用（例如有人写 CGS 有人写 SI）不会报错，
           但结果无意义。
        5. 指数向量长度不一致会直接报错——这类错误在建模里通常意味着"漏写了一个量纲"。

    参考:
        Buckingham, "On physically similar systems; illustrations of the use of
        dimensional equations", Physical Review, 1914。
    """
    if not isinstance(dims, dict):
        raise ValueError(f"dims 必须是 dict，得到 {type(dims).__name__}")
    names = list(dims.keys())
    if len(names) < 2:
        raise ValueError(f"dims 至少要有 2 个物理变量，得到 {len(names)} 个")
    cols: List[np.ndarray] = []
    k = None
    for name in names:
        vec = as_vector(dims[name], f"dims[{name!r}]")
        if k is None:
            k = int(vec.size)
        elif int(vec.size) != k:
            raise ValueError(
                f"dims[{name!r}] 的指数向量长度 {vec.size} 与前面的 {k} 不一致"
                f"（基本量纲个数必须统一）"
            )
        cols.append(vec)

    d_mat = np.column_stack(cols)  # (k 个基本量纲, n 个变量)
    frac = [
        [Fraction(float(x)).limit_denominator(10 ** 6) for x in row]
        for row in d_mat
    ]
    n_vars = len(names)

    pivots: List[int] = []
    row = 0
    for col in range(n_vars):
        pivot = None
        for i in range(row, k):
            if frac[i][col] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        frac[row], frac[pivot] = frac[pivot], frac[row]
        pv = frac[row][col]
        frac[row] = [x / pv for x in frac[row]]
        for i in range(k):
            if i != row and frac[i][col] != 0:
                factor = frac[i][col]
                frac[i] = [a - factor * b for a, b in zip(frac[i], frac[row])]
        pivots.append(col)
        row += 1
        if row == k:
            break

    rank = len(pivots)
    if rank == 0:
        raise ValueError("量纲矩阵的秩为 0（所有变量都是无量纲的），不存在可求的 π 组")

    free = [c for c in range(n_vars) if c not in pivots]
    groups: List[List[int]] = []
    pi_names: List[str] = []
    for idx, col in enumerate(free, start=1):
        vec = [Fraction(0)] * n_vars
        vec[col] = Fraction(1)
        for i, pc in enumerate(pivots):
            vec[pc] = -frac[i][col]
        ints = _canonical_int_vector(vec)
        groups.append(ints)
        terms = [
            f"{names[j]}^{ints[j]}" for j in range(n_vars) if ints[j] != 0
        ]
        pi_names.append(f"pi_{idx} = " + (" * ".join(terms) if terms else "1"))

    return {
        "rank": int(rank),
        "n_pi": int(len(free)),
        "pi_groups": groups,
        "pi_names": pi_names,
    }


def scaling_similarity(
    model_ratio: float,
    measurements: Dict[str, float],
    target: Dict[str, float],
    exponents: Optional[Dict[str, float]] = None,
) -> dict:
    """按几何相似比与量纲幂次把缩尺模型测量值换算到原型尺度，并与原型参考值比对。

    参数:
        model_ratio: 几何相似比 λ = L_原型 / L_模型，必须 > 0。
        measurements: ``{物理量名: 缩尺模型上的实测值}``。
        target: ``{物理量名: 原型上的参考值}``；键必须与 measurements 完全一致，
            用来计算 relative_error（换算正确时应当接近 0）。
        exponents: ``{物理量名: 相对长度的量纲幂次 e}``；None 表示全部取 1（纯几何缩放）。
            例如几何相似下面积 e = 2、体积 e = 3；Froude 相似下速度 e = 0.5、力 e = 3。

    返回:
        dict，键为：
        ``factors``  {物理量名: λ^e} 的换算因子（dict[str, float]）；
        ``scaled``  {物理量名: measurements * factor} 的原型尺度换算值；
        ``relative_error``  {物理量名: |scaled - target| / |target|}，target = 0 时
            退化为绝对误差 |scaled - target|（避免 0 除）。

    算法:
        1. factor_v = λ^{e_v}（量纲幂次换算：长度量纲按 λ、面积按 λ^2、速度按 λ^{1/2} …）。
        2. scaled_v = measurements_v * factor_v。
        3. relative_error_v = |scaled_v - target_v| / |target_v|。

    复杂度:
        时间 O(k) / 空间 O(k)（k 为物理量个数）。

    陷阱:
        1. **相似比的方向**：λ 定义为"原型 / 模型"。若误用"模型 / 原型"，所有因子会整体取
           倒数，得到的"原型"值会小得离谱；论文里必须写清 λ 的定义方向。
        2. 比例缩放只在**几何相似 + 相关无量纲数相等**时成立；Froude 相似与 Reynolds 相似
           一般不能同时满足（除非用变尺度模型），所以"用同一个 λ 换算出所有量"是有前提的。
        3. 无量纲量（e = 0）的因子恒为 1，不随尺度变化；若算出它的换算值变了，说明
           exponents 传错了。
        4. target 是**独立观测**的原型参考值，不是换算结果；如果把 scaled 自己传进 target，
           relative_error 会恒等于 0，这个自测就什么也没验证。
        5. 外推范围：λ 很大时缩尺模型的粘性/表面张力等效应会被放大，纯幂次换算不再成立。

    参考:
        相似三定理与量纲分析（Buckingham 1914；Bridgman, "Dimensional Analysis", 1922）。
    """
    lam = float(model_ratio)
    if not (lam > 0.0):
        raise ValueError(f"model_ratio 必须 > 0，得到 {model_ratio}")
    if not isinstance(measurements, dict) or not isinstance(target, dict):
        raise ValueError("measurements 与 target 都必须是 dict")
    if not measurements:
        raise ValueError("measurements 不能为空")
    if set(measurements.keys()) != set(target.keys()):
        raise ValueError(
            f"measurements 与 target 的键必须一致，得到 "
            f"{sorted(measurements.keys())} vs {sorted(target.keys())}"
        )
    if exponents is None:
        exps = {name: 1.0 for name in measurements}
    else:
        if not isinstance(exponents, dict) or set(exponents.keys()) != set(measurements.keys()):
            raise ValueError(
                f"exponents 的键必须与 measurements 一致，得到 "
                f"{sorted(exponents.keys()) if isinstance(exponents, dict) else exponents}"
            )
        exps = {name: float(exponents[name]) for name in measurements}

    factors: Dict[str, float] = {}
    scaled: Dict[str, float] = {}
    rel_err: Dict[str, float] = {}
    for name in measurements:
        m_val = float(np.asarray(measurements[name], dtype=float))
        t_val = float(np.asarray(target[name], dtype=float))
        if not (math.isfinite(m_val) and math.isfinite(t_val)):
            raise ValueError(f"{name} 的测量值/参考值必须是有限数，得到 {m_val} / {t_val}")
        factor = float(lam ** exps[name])
        value = m_val * factor
        factors[name] = factor
        scaled[name] = value
        rel_err[name] = abs(value - t_val) / abs(t_val) if t_val != 0.0 else abs(value - t_val)
    return {"factors": factors, "scaled": scaled, "relative_error": rel_err}


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict（int/float/bool/list），固定种子下两次调用完全一致，键名以 ``spatial_`` 前缀区分：
        ``spatial_heat_explicit_err`` / ``spatial_heat_implicit_err`` 两种格式与解析解
        sin(πx)e^{-alpha π² t} 的最大偏差；``spatial_heat_explicit_vs_implicit`` 二者的最大差；
        ``spatial_heat_explicit_ratio`` / ``spatial_heat_cn_big_dt_ratio`` 稳定性比（后者是
        CN 在大 dt 算例里的 r=5）；
        ``spatial_heat_implicit_bounded`` 大 dt（r=5）时隐式格式是否有界；
        ``spatial_heat_conservation_error`` 零通量边界下离散总热量的守恒误差；
        ``spatial_heat_mode_shape_error`` 长时间演化后解与正弦基模的比例误差；
        ``spatial_heat_rejects_unstable`` 显式格式对 r>0.5 是否抛 ValueError；
        ``spatial_poisson_max_err`` / ``spatial_poisson_vs_direct`` / ``spatial_poisson_residual``
        / ``spatial_poisson_n_iter`` SOR 与解析解 sin(πx)sin(πy) 的偏差、与直接法解的偏差、
        残差与迭代次数；
        ``spatial_poisson_asymmetry`` 解与其转置的最大差（对称性检验）；
        ``spatial_poisson_rejects_omega`` ω 越界是否抛 ValueError；
        ``spatial_fire_no_light_burned`` / ``spatial_fire_no_grow_trees`` /
        ``spatial_fire_burned_total`` / ``spatial_fire_tree_ratio`` / ``spatial_fire_steps``
        森林火灾（无雷击时零燃烧、无生长时终态零树、燃烧总数、终态树占比、步数）；
        ``spatial_traffic_free_speed_ratio`` / ``spatial_traffic_flow`` / ``spatial_traffic_density``
        / ``spatial_traffic_speed_brake0`` / ``spatial_traffic_speed_brake03``
        / ``spatial_traffic_speed_decreases`` / ``spatial_traffic_dense_speed``
        NaSch 的自由流速度比、流量、密度、两种刹车概率下的平均速度、刹车减速单调性、高密度速度；
        ``spatial_pi_rank`` / ``spatial_pi_n_pi`` / ``spatial_pi_drag_group`` / ``spatial_pi_groups``
        / ``spatial_pi_names`` / ``spatial_pi_rank_matches`` Buckingham π 的 rank、π 个数、
        阻力无量纲组、全部规范 π 组、π 组名字，以及与 numpy 秩的交叉印证；
        ``spatial_scale_length_factor`` / ``spatial_scale_velocity_factor``
        / ``spatial_scale_max_rel_error`` 比例缩放的因子与最大相对误差。

    算法:
        1. 泊松：n=16、f = 2π²sin(πx)sin(πy)（此时解析解 u = sin(πx)sin(πy)），
           ω=1.5、tol=1e-8，与解析解对拍（误差 < 5e-3：h=1/17 时五点差分的截断误差
           ~π²h²/12 ≈ 2.8e-3 是主导项，数值解与网格无关地贴着截断误差走）；
           再用 ``np.linalg.solve`` 直接解同一套离散方程做一次**与网格无关**的独立对拍
           （差 < 1e-6），这条才真正检验"解对了方程"。
        2. 热传导：解析解 u(x,t) = sin(πx)e^{-alpha π² t}。取 dx=0.05、dt=1e-6（r=4e-4，
           远在稳定域内）、200 步，显式与 CN 都逼近解析解（误差受空间项主导，< 5e-3），
           而二者的差应当 < 1e-6（时间阶数差异 ~π⁴ dt t /2 ≈ 1e-8）。
        3. 零通量：初值取方波，跑 50 步，比较首末步的离散总和（应当精确守恒到 1e-12）。
        4. 基模形状：11 个格点（dx=0.1）、r=0.4、200 步的对称方波初值；dx 越粗相邻模态
           衰减率之比越小，200 步足以滤掉高阶模态，解应当与 sin(πx) 成比例（误差 < 1e-6）。
        5. 隐式大 dt：r = 5（显式早已发散）时 CN 解仍满足极值原理（max|u| 不增）。
        6. Buckingham：经典阻力问题（F 依赖 v, ρ, μ, L）必须 rank=3、n_pi=2，
           且其中一个规范 π 组恰为 F/(ρ v² L²)（指数向量 [1, -2, -1, 0, -2]）；
           另外用 ``np.linalg.matrix_rank`` 独立算一遍秩互相印证。
        7. 森林火灾：p_light=0 时 burned_total 必须为 0；p_grow=0 且 p_light=1 时终态树数为 0。
        8. 交通流：ρ=0.1（< 1/(v_max+1) = 1/6）时平均速度接近 v_max（> 0.9 v_max）；
           同一时间窗内 flow 必须等于 density * mean_speed；p_brake 从 0 增到 0.3 后速度下降。
        9. 比例缩放：λ=100 时长度因子 100、面积因子 1e4（纯几何）、Froude 相似下速度因子 10；
           无量纲量因子恒为 1；用解析构造的 target 反算 relative_error 应当为 0。

    复杂度:
        时间最重的是泊松（n=16 时约 155 次 SOR 迭代，O(n_iter·n²) ≈ 4e4 次格点更新），
        其余都是 ≤200 步的一维/元胞算例；整组自测在普通笔记本上 < 50 ms / 空间 O(n²)。

    陷阱:
        1. 泊松的解析解判据对网格很敏感：n=20 时截断误差 ~1.9e-3，n=16 时 ~2.8e-3，判据定在
           5e-3。这条断言失败**通常不是 bug 而是网格太粗**，真正锐利的判据是直接法对拍那条。
        2. 基模形状那条**不能**用 r = 0.5：稳定边界上 FTCS 的奇格点与偶格点完全解耦，
           等间距方波初值恰在解耦子空间里，数值解会永远停在平顶伪稳态上（实测偏差 1.2%），
           看起来像"实现错了"，其实是格式在边界上的固有陷阱。取 r = 0.4 即可避开。
        3. "显式与隐式一致到 1e-6"只在 dt 远小于 dx²/alpha 时成立；用 r=0.5 的工程步长
           两者会差百分之几（时间阶数不同），不要用这个例子去否定实现。
        4. 交通流的自由流断言必须用 ρ <= 1/(v_max+1)；ρ=0.2、v_max=5 时平均速度的硬上界
           是 (1-ρ)/ρ = 4 < 0.9 v_max，物理上不可能达到 v_max。
        5. 森林火灾的断言依赖固定种子下的具体初始布局：换种子后"无生长时终态零树"仍成立，
           但终态树占比之类的数值会变。

    参考:
        本模块各函数参考文献。
    """
    # ---------- 0. 泊松 SOR：解析解 + 直接法两重独立对拍 ----------
    # 解析解 sin(πx)sin(πy) 只能逼近到 O(h²)（n=16 时截断误差量级 π²h²/12 ≈ 2.8e-3，实测最大值 2.83e-3），
    # 所以再加一个与网格无关的独立参照：用 np.linalg.solve 直接解同一套离散方程。
    n_poisson = 16
    h = 1.0 / (n_poisson + 1)
    grid = (np.arange(1, n_poisson + 1) * h)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    u_exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    f_poisson = 2.0 * np.pi ** 2 * u_exact
    sor = poisson_2d_sor(n_poisson, f_poisson, omega=1.5, tol=1e-8, max_iter=5000)
    poisson_err = float(np.max(np.abs(sor["u"] - u_exact)))
    if poisson_err > 5e-3:
        raise AssertionError(
            f"泊松 SOR 与解析解 sin(πx)sin(πy) 的最大偏差 {poisson_err} 超过 5e-3"
        )
    lap = np.zeros((n_poisson ** 2, n_poisson ** 2))
    for i_lap in range(n_poisson):
        for j_lap in range(n_poisson):
            k_lap = i_lap * n_poisson + j_lap
            lap[k_lap, k_lap] = 4.0
            if i_lap > 0:
                lap[k_lap, k_lap - n_poisson] = -1.0
            if i_lap < n_poisson - 1:
                lap[k_lap, k_lap + n_poisson] = -1.0
            if j_lap > 0:
                lap[k_lap, k_lap - 1] = -1.0
            if j_lap < n_poisson - 1:
                lap[k_lap, k_lap + 1] = -1.0
    u_direct = np.linalg.solve(lap, f_poisson.ravel() * h * h).reshape(n_poisson, n_poisson)
    poisson_gap = float(np.max(np.abs(sor["u"] - u_direct)))
    if poisson_gap > 1e-6:
        raise AssertionError(
            f"SOR 解与直接法解相差 {poisson_gap}（>1e-6），说明迭代没收敛到真解"
        )
    if sor["residual"] > 1e-8:
        raise AssertionError(f"泊松 SOR 报收敛但残差 {sor['residual']} > tol=1e-8")
    poisson_asym = float(np.max(np.abs(sor["u"] - sor["u"].T)))
    if poisson_asym > 1e-6:
        raise AssertionError(f"对称右端项的泊松解应当对称，实测不对称度 {poisson_asym}")
    try:
        poisson_2d_sor(4, np.zeros((4, 4)), omega=2.5, tol=1e-8, max_iter=10)
        raise AssertionError("omega=2.5 本应抛 ValueError")
    except ValueError:
        poisson_omega_reject = True

    # ---------- 1. 热传导：显式 / Crank-Nicolson 与解析解对拍 ----------
    alpha, dx, dt, n_steps = 1.0, 0.05, 1e-6, 200
    x_heat = np.linspace(0.0, 1.0, 21)
    u0_sine = np.sin(np.pi * x_heat)
    u_analytic = u0_sine * math.exp(-alpha * np.pi ** 2 * dt * n_steps)
    exp_res = heat_equation_1d_explicit(u0_sine, alpha, dx, dt, n_steps)
    imp_res = heat_equation_1d_implicit(u0_sine, alpha, dx, dt, n_steps)
    heat_exp_err = float(np.max(np.abs(exp_res["u"] - u_analytic)))
    heat_imp_err = float(np.max(np.abs(imp_res["u"] - u_analytic)))
    heat_gap = float(np.max(np.abs(exp_res["u"] - imp_res["u"])))
    if heat_exp_err > 5e-3:
        raise AssertionError(
            f"显式 FTCS 与解析解 e^(-π²t) 的偏差 {heat_exp_err} 超过 5e-3"
        )
    if heat_imp_err > 5e-3:
        raise AssertionError(
            f"Crank-Nicolson 与解析解 e^(-π²t) 的偏差 {heat_imp_err} 超过 5e-3"
        )
    if heat_gap > 1e-6:
        raise AssertionError(
            f"小步长下显式与 CN 应当一致到 1e-6，实测最大差 {heat_gap}"
        )
    if abs(exp_res["stability_ratio"] - alpha * dt / dx ** 2) > 1e-15:
        raise AssertionError("显式格式返回的稳定性比与 alpha*dt/dx**2 不符")
    try:
        heat_equation_1d_explicit(u0_sine, alpha, dx, 0.5 * dx ** 2 / alpha * 1.2, 1)
        raise AssertionError("r>0.5 的显式格式本应抛 ValueError")
    except ValueError:
        heat_reject_unstable = True

    # ---------- 2. 零通量边界：离散总热量守恒 ----------
    u0_square = np.where((x_heat > 0.3) & (x_heat < 0.7), 1.0, 0.0)
    cons = heat_equation_1d_explicit(u0_square, alpha, 0.1, 0.005, 50, bc=("neumann", "neumann"))
    cons_err = abs(float(np.sum(cons["u"])) - float(np.sum(u0_square)))
    if cons_err > 1e-12:
        raise AssertionError(f"零通量边界下总热量应守恒，实测误差 {cons_err}")

    # ---------- 3. 长时间演化后应当只剩正弦基模 ----------
    # 刻意取 r = 0.4 < 0.5 而不是稳定边界 0.5：r 恰为 0.5 时 FTCS 的奇数格点与偶数格点
    # 完全解耦，若初值恰好落进解耦子空间（等间距的方波正是如此），数值解会停在一个平顶的
    # 伪稳态上永不收敛到真正的基模 sin(πx)，这是该格式在稳定边界上的一个真实陷阱。
    # 只取 11 个格点 + 200 步：dx=0.1 时相邻模态衰减率之比 0.88，200 步已足以滤掉高阶模态。
    x_mode = np.linspace(0.0, 1.0, 11)
    u0_bump = np.where(np.abs(x_mode - 0.5) < 0.25, 1.0, 0.0)
    mode = heat_equation_1d_explicit(u0_bump, alpha, 0.1, 0.4 * 0.1 ** 2 / alpha, 200)
    um = mode["u"]
    mid = float(um[um.size // 2])
    if abs(mid) < 1e-15:
        raise AssertionError("长时间演化后基模振幅为 0，说明迭代没生效")
    shape_err = float(np.max(np.abs(um / mid - np.sin(np.pi * x_mode))))
    if shape_err > 1e-6:
        raise AssertionError(f"长时间演化后解应正比于 sin(πx)，实测偏差 {shape_err}")
    if float(np.max(np.abs(um))) >= 1.0:
        raise AssertionError("零 Dirichlet 边界的解应当单调衰减，振幅未减小")

    # ---------- 4. Crank-Nicolson 在大 dt 下仍然有界（无条件稳定 + 极值原理） ----------
    big = heat_equation_1d_implicit(u0_square, alpha, 0.1, 0.05, 10)  # r = 5
    if float(np.max(np.abs(big["u"]))) > float(np.max(np.abs(u0_square))) + 1e-12:
        raise AssertionError(
            f"CN 在 r=5 时应满足极值原理，实测 max|u|={float(np.max(np.abs(big['u'])))}"
        )
    if abs(float(big["matrix"][5, 5]) - (1.0 + 5.0)) > 1e-12:
        raise AssertionError("CN 矩阵对角元应为 1+r")

    # ---------- 5. 森林火灾：两条极限情形的确定结论 ----------
    fire_dark = forest_fire_ca(20, p_grow=0.05, p_light=0.0, n_steps=30, seed=7)
    if fire_dark["burned_total"] != 0:
        raise AssertionError(
            f"p_light=0 时不应有任何燃烧，实测 burned_total={fire_dark['burned_total']}"
        )
    fire_nogrow = forest_fire_ca(20, p_grow=0.0, p_light=1.0, n_steps=30, seed=7)
    if fire_nogrow["final_trees"] != 0:
        raise AssertionError(
            f"p_grow=0 且每步雷击时终态不应有树，实测 {fire_nogrow['final_trees']}"
        )
    fire_std = forest_fire_ca(30, p_grow=0.05, p_light=0.3, n_steps=50, seed=None)
    if len(fire_std["steps"]) != 50:
        raise AssertionError("森林火灾的 steps 长度应等于 n_steps")
    if fire_std["burned_total"] <= 0:
        raise AssertionError("默认参数下应当发生过燃烧")

    # ---------- 6. NaSch 交通流：自由流速度上界、流量口径、刹车单调性 ----------
    free = traffic_ca_nagel_schreckenberg(n_cells=100, n_cars=10, v_max=5, p_brake=0.0,
                                          n_steps=100, seed=None)
    if "steps" not in free or len(free["steps"]) != 100:
        raise AssertionError("NaSch 的 steps 长度应等于 n_steps")
    if free["mean_speed"] < 0.9 * 5:
        raise AssertionError(
            f"rho=0.1 < 1/(v_max+1) 时应接近自由流，实测 mean_speed={free['mean_speed']}"
        )
    if abs(free["flow"] - free["density"] * free["mean_speed"]) > 1e-12:
        raise AssertionError("流量必须等于 密度 x 平均速度（同一时间窗）")
    braked = traffic_ca_nagel_schreckenberg(n_cells=100, n_cars=10, v_max=5, p_brake=0.3,
                                            n_steps=100, seed=None)
    if not braked["mean_speed"] < free["mean_speed"]:
        raise AssertionError(
            f"随机慢化应当降低平均速度：p_brake=0 时 {free['mean_speed']}，"
            f"p_brake=0.3 时 {braked['mean_speed']}"
        )
    dense = traffic_ca_nagel_schreckenberg(n_cells=100, n_cars=20, v_max=5, p_brake=0.0,
                                           n_steps=100, seed=None)
    if not dense["mean_speed"] <= (100 - 20) / 20.0 + 1e-9:
        raise AssertionError(
            f"rho=0.2 时平均速度不应超过 (1-rho)/rho=4，实测 {dense['mean_speed']}"
        )

    # ---------- 7. Buckingham π：rank / n_pi 与解析 π 组 ----------
    dims = {
        "F": [1.0, 1.0, -2.0],
        "v": [0.0, 1.0, -1.0],
        "rho": [1.0, -3.0, 0.0],
        "mu": [1.0, -1.0, -1.0],
        "L": [0.0, 1.0, 0.0],
    }
    pi = buckingham_pi(dims)
    if pi["rank"] != 3:
        raise AssertionError(f"阻力问题量纲矩阵的秩应为 3，得到 {pi['rank']}")
    if pi["n_pi"] != 5 - 3:
        raise AssertionError(f"阻力问题应有 5-3=2 个 π 组，得到 {pi['n_pi']}")
    drag = [1, -2, -1, 0, -2]  # F^1 rho^-1 v^-2 L^-2 = F/(rho v^2 L^2)
    if drag not in pi["pi_groups"]:
        raise AssertionError(
            f"π 组里应含 F/(rho v^2 L^2)（指数 {drag}），得到 {pi['pi_groups']}"
        )
    d_mat = np.column_stack([np.asarray(dims[k], dtype=float) for k in dims])
    if np.linalg.matrix_rank(d_mat) != pi["rank"]:
        raise AssertionError("精确有理数秩与 np.linalg.matrix_rank 不一致")

    # ---------- 8. 比例缩放：解析因子 ----------
    scale = scaling_similarity(
        100.0,
        {"length": 0.3, "area": 0.09, "velocity": 2.0, "force": 12.0, "Re": 1.0e6},
        {"length": 30.0, "area": 900.0, "velocity": 20.0, "force": 1.2e7, "Re": 1.0e6},
        exponents={"length": 1.0, "area": 2.0, "velocity": 0.5, "force": 3.0, "Re": 0.0},
    )
    if abs(scale["factors"]["length"] - 100.0) > 1e-12:
        raise AssertionError("λ=100 时长度因子应为 100")
    if abs(scale["factors"]["area"] - 1.0e4) > 1e-9:
        raise AssertionError("λ=100 时面积因子应为 λ²=1e4")
    if abs(scale["factors"]["velocity"] - 10.0) > 1e-12:
        raise AssertionError("Froude 相似下速度因子应为 sqrt(λ)=10")
    if abs(scale["factors"]["Re"] - 1.0) > 1e-12:
        raise AssertionError("无量纲量的换算因子必须恒为 1")
    max_rel = max(scale["relative_error"].values())

    return {
        "spatial_heat_explicit_err": round(heat_exp_err, 8),
        "spatial_heat_implicit_err": round(heat_imp_err, 8),
        "spatial_heat_explicit_vs_implicit": round(heat_gap, 10),
        "spatial_heat_explicit_ratio": round(float(exp_res["stability_ratio"]), 12),
        "spatial_heat_cn_big_dt_ratio": round(float(alpha * 0.05 / 0.1 ** 2), 6),
        "spatial_heat_implicit_bounded": bool(
            float(np.max(np.abs(big["u"]))) <= float(np.max(np.abs(u0_square))) + 1e-12
        ),
        "spatial_heat_conservation_error": round(cons_err, 14),
        "spatial_heat_mode_shape_error": round(shape_err, 10),
        "spatial_heat_rejects_unstable": bool(heat_reject_unstable),
        "spatial_poisson_max_err": round(poisson_err, 8),
        "spatial_poisson_vs_direct": round(poisson_gap, 10),
        "spatial_poisson_residual": round(float(sor["residual"]), 12),
        "spatial_poisson_n_iter": int(sor["n_iter"]),
        "spatial_poisson_asymmetry": round(poisson_asym, 10),
        "spatial_poisson_rejects_omega": bool(poisson_omega_reject),
        "spatial_fire_no_light_burned": int(fire_dark["burned_total"]),
        "spatial_fire_no_grow_trees": int(fire_nogrow["final_trees"]),
        "spatial_fire_burned_total": int(fire_std["burned_total"]),
        "spatial_fire_tree_ratio": round(float(fire_std["tree_ratio"]), 6),
        "spatial_fire_steps": len(fire_std["steps"]),
        "spatial_traffic_free_speed_ratio": round(float(free["mean_speed"]) / 5.0, 6),
        "spatial_traffic_flow": round(float(free["flow"]), 6),
        "spatial_traffic_density": round(float(free["density"]), 6),
        "spatial_traffic_speed_brake0": round(float(free["mean_speed"]), 6),
        "spatial_traffic_speed_brake03": round(float(braked["mean_speed"]), 6),
        "spatial_traffic_speed_decreases": bool(braked["mean_speed"] < free["mean_speed"]),
        "spatial_traffic_dense_speed": round(float(dense["mean_speed"]), 6),
        "spatial_pi_rank": int(pi["rank"]),
        "spatial_pi_n_pi": int(pi["n_pi"]),
        "spatial_pi_drag_group": [int(v) for v in drag],
        "spatial_pi_groups": [[int(v) for v in g] for g in pi["pi_groups"]],
        "spatial_pi_names": list(pi["pi_names"]),
        "spatial_pi_rank_matches": bool(np.linalg.matrix_rank(d_mat) == pi["rank"]),
        "spatial_scale_length_factor": round(float(scale["factors"]["length"]), 6),
        "spatial_scale_velocity_factor": round(float(scale["factors"]["velocity"]), 6),
        "spatial_scale_max_rel_error": round(float(max_rel), 12),
    }
