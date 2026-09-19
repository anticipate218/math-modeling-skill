"""聚类算法：K-means++、K-means、轮廓系数、肘部曲线、层次聚类、DBSCAN、
高斯混合 EM、谱聚类、模糊 C 均值、DB/CH 有效性指标与 gap statistic。

本模块共 12 个公开函数，按用途分成三组：聚类核心 ``kmeans_plusplus_init`` / ``kmeans`` /
``agglomerative`` / ``dbscan`` / ``gmm_em`` / ``spectral_clustering`` / ``fuzzy_cmeans``，
有效性评估 ``silhouette_score`` / ``davies_bouldin_score`` / ``calinski_harabasz_score`` /
``gap_statistic``，以及选 k 用的 ``elbow_curve``。

这些实现是**教学透明版**：目标是把论文里要交代的每一步都摊开写清楚，
而不是追求工业级速度。数据量超过几千条时请换成熟实现（如 sklearn），
本模块适合用来对照结果、做消融实验和解释算法内部机制。

关键约定
--------
- 所有距离都是欧氏距离，特征未做标准化时"距离"会被量纲大的特征支配，
  调用方必须自己先决定是否标准化（见 ``_common.normalize_minmax`` / ``_common.normalize_l2``）。
- 随机性只出现在 K-means++ 初始化、GMM/谱聚类/模糊 C 均值的初始化与 gap 参考集，
  统一走 ``_common.rng``，不接受全局随机状态。
- DBSCAN 的噪声点标签为 ``-1``；簇编号从 0 开始。
- 有效性指标方向固定：``silhouette_score`` / ``calinski_harabasz_score`` / ``gap_statistic``
  **越大越好**，``davies_bouldin_score`` **越小越好**；三个指标都只是相对比较口径，
  不要跨数据集比绝对值。
- ``gmm_em`` 的每个协方差矩阵都加 ``reg_covar * I`` 保正定，对数行列式用 ``np.linalg.slogdet``；
  ``spectral_clustering`` 的相似度用高斯核，``sigma`` 缺省取成对距离的中位数。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ._common import DEFAULT_SEED, as_matrix, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "kmeans_plusplus_init",
    "kmeans",
    "silhouette_score",
    "elbow_curve",
    "agglomerative",
    "dbscan",
    "gmm_em",
    "spectral_clustering",
    "fuzzy_cmeans",
    "davies_bouldin_score",
    "calinski_harabasz_score",
    "gap_statistic",
]


def _pairwise_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """内部工具：样本到中心的欧氏距离矩阵，形状 (n_samples, n_centers)。

    用展开式 ||a-b||^2 = ||a||^2 - 2a·b + ||b||^2 一次算完，避免显式三重循环；
    负数由浮点误差引起时用 clip 归零，再开方。
    """
    a2 = np.sum(x ** 2, axis=1, keepdims=True)
    b2 = np.sum(centers ** 2, axis=1, keepdims=True).T
    sq = a2 - 2.0 * (x @ centers.T) + b2
    return np.sqrt(np.clip(sq, 0.0, None))


def _check_k(n_samples: int, k: int) -> None:
    """内部工具：校验簇数合法，非法时抛 ValueError（不用 assert）。"""
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k 必须是整数")
    if k < 1:
        raise ValueError(f"k 必须 >= 1，得到 {k}")
    if k > n_samples:
        raise ValueError(f"k={k} 不能大于样本数 {n_samples}")


def kmeans_plusplus_init(X: MatrixLike, k: int, seed: Optional[int] = None) -> np.ndarray:
    """K-means++ 初始化：按 D^2 采样挑出 k 个彼此远离的初始中心。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k: 簇数。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        np.ndarray，形状 (k, n_features)，初始中心（是真实样本点的副本）。

    算法:
        1. 均匀随机取第 1 个中心；
        2. 计算每个样本到"已选中心集合"的最近距离平方 D(x)^2；
        3. 以正比于 D(x)^2 的概率抽下一个中心，重复直到选满 k 个。
        （Arthur & Vassilvitskii 2007；平滑项 eps 用于所有 D^2 同时为 0 的退化情形。）

    复杂度:
        时间 O(n k d) / 空间 O(n k)。

    陷阱:
        重复点很多时 D^2 会同时为 0，此时退化为均匀随机——所以求解时要检测重复中心
        并重采样。另外 k 个中心可能包含重复样本，这会让某个簇在第一次分配后为空，
        必须在 ``kmeans`` 里做空簇修复。

    参考:
        Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding", SODA 2007。
    """
    x = as_matrix(X, "X")
    n_samples = x.shape[0]
    _check_k(n_samples, k)
    gen = rng(seed)

    centers = np.empty((k, x.shape[1]), dtype=float)
    first = int(gen.integers(n_samples))
    centers[0] = x[first]
    closest = np.sum((x - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # 所有剩余点到已选中心的距离都是 0（数据高度重复），退化为均匀随机。
            probs = np.full(n_samples, 1.0 / n_samples)
        else:
            probs = closest / total
        idx = int(gen.choice(n_samples, p=probs))
        centers[i] = x[idx]
        closest = np.minimum(closest, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def kmeans(
    X: MatrixLike,
    k: int,
    seed: Optional[int] = None,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> dict:
    """Lloyd 迭代的 K-means：分配 → 更新中心，直到中心移动小于 tol 或达到迭代上限。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k: 簇数。
        seed: 传给 K-means++ 的随机种子。
        max_iter: 最大迭代轮数。
        tol: 收敛阈值，判据是"中心矩阵的最大位移 <= tol"。

    返回:
        dict，键为：
        ``labels``  形状 (n_samples,) 的 int 数组，取值 0..k-1；
        ``centers`` 形状 (k, n_features) 的簇中心；
        ``inertia`` float，簇内平方和 SSE（目标函数值）；
        ``n_iter``  int，实际执行的迭代轮数。

    算法:
        1. K-means++ 初始化中心；
        2. 把每个点分到最近中心；
        3. 每簇取均值作为新中心（空簇则把离自己中心最远的点拉出来当新中心）；
        4. 中心位移 <= tol 时停止。

    复杂度:
        时间 O(n k d * n_iter) / 空间 O(n k)。

    陷阱:
        1. **局部最优**：K-means 只保证收敛到局部极小，换个种子 inertia 可能差 20% 以上，
           论文中应固定种子并报告 inertia，或用多次重启取最优。
        2. **空簇**：初始化若出现重复中心，某簇可能一个点都没有，直接取均值会得到 NaN。
           这里显式做"抢走最远点"的修复（把当前分配到离自己中心最远的那个样本改判给空簇，
           该点直接当空簇中心）。
        3. 上面的空簇修复**只是启发式**，不做回滚也不重算：它改的是"本轮新中心"和临时的
           ``labels``，而下一轮会用重算的 ``labels = argmin(dist)`` 覆盖。所以修复后的
           ``centers[j] = x[far]`` 有可能在最后一轮被 argmin 重新分配掉，返回的 ``centers``
           与 ``labels`` 因此**不保证**是"每簇均值的精确自洽解"。要严格自洽请自己再用返回的
           ``labels`` 算一次均值。
        4. **量纲**：特征未标准化时 SSE 几乎没有解释力，先标准化再聚类。
        5. 收敛判据用的是中心位移而不是 inertia 变化（后者量级依赖数据尺度，阈值不好定）。

    参考:
        Lloyd, "Least squares quantization in PCM", 1982；MacQueen 1967 命名 K-means。
    """
    x = as_matrix(X, "X")
    n_samples, n_features = x.shape
    _check_k(n_samples, k)
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，得到 {max_iter}")
    if tol < 0:
        raise ValueError(f"tol 必须 >= 0，得到 {tol}")

    centers = kmeans_plusplus_init(x, k, seed=seed)
    labels = np.zeros(n_samples, dtype=int)
    n_iter = 0

    for it in range(1, max_iter + 1):
        dist = _pairwise_distances(x, centers)
        labels = np.argmin(dist, axis=1).astype(int)
        new_centers = centers.copy()
        for j in range(k):
            members = x[labels == j]
            if members.size == 0:
                # 空簇修复：把离自己所属中心最远的样本改判为该簇的单点簇。
                own = dist[np.arange(n_samples), labels]
                far = int(np.argmax(own))
                new_centers[j] = x[far]
                labels[far] = j
            else:
                new_centers[j] = members.mean(axis=0)
        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        n_iter = it
        if shift <= tol:
            break

    dist = _pairwise_distances(x, centers)
    labels = np.argmin(dist, axis=1).astype(int)
    inertia = float(np.sum(dist[np.arange(n_samples), labels] ** 2))
    return {"labels": labels, "centers": centers, "inertia": inertia, "n_iter": n_iter}


def silhouette_score(X: MatrixLike, labels: ArrayLike) -> float:
    """轮廓系数：逐点 (b - a) / max(a, b) 的均值，衡量簇内紧致与簇间分离。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        labels: 每个样本的簇标签，长度 n_samples。

    返回:
        float，全部样本轮廓系数的均值，取值 [-1, 1]；越大越好。

    算法:
        a(i) = 到同簇其他点的平均距离；b(i) = 到最近其他簇的平均距离；
        s(i) = (b - a) / max(a, b)。单点簇记为 s = 0。

    复杂度:
        时间 O(n^2 d)（需要完整距离矩阵）/ 空间 O(n^2)。

    陷阱:
        1. **k=1 或所有标签相同**时 b 不存在，公式无定义：本函数返回 0.0，
           调用方收到 0.0 必须理解为"不可用"而不是"聚类质量等于随机"。
        2. 单点簇的 a=0，s 被约定为 0，这会拉低均值；簇很多时轮廓系数天然偏低。
        3. O(n^2) 内存：n=20000 时距离矩阵约 3.2 GB，必须改用抽样版。

    参考:
        Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation
        of cluster analysis", 1987。
    """
    x = as_matrix(X, "X")
    lab = np.asarray(labels).ravel().astype(int)
    if lab.size != x.shape[0]:
        raise ValueError(f"labels 长度 {lab.size} 与样本数 {x.shape[0]} 不一致")
    unique = np.unique(lab)
    if unique.size < 2:
        # k=1 或全同标签：b(i) 无定义，按约定返回 0.0。
        return 0.0

    n = x.shape[0]
    dist = _pairwise_distances(x, x)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        same = lab == lab[i]
        same[i] = False
        n_same = int(np.count_nonzero(same))
        if n_same == 0:
            scores[i] = 0.0
            continue
        a = float(dist[i, same].mean())
        b = np.inf
        for c in unique:
            if c == lab[i]:
                continue
            other = lab == c
            b = min(b, float(dist[i, other].mean()))
        denom = max(a, b)
        scores[i] = 0.0 if denom == 0.0 else (b - a) / denom
    return float(scores.mean())


def elbow_curve(X: MatrixLike, k_range: Iterable[int], seed: Optional[int] = None) -> Dict[int, float]:
    """肘部曲线：对一串 k 各跑一次 K-means，返回 SSE 随 k 的变化。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k_range: 待尝试的 k 序列，例如 ``range(1, 11)``。
        seed: 随机种子，所有 k 共用同一个种子以保证可比。

    返回:
        dict，``{k: inertia}``，键是 int，值是 float。

    算法:
        逐个 k 调用 ``kmeans``，记录 inertia；拐点（下降速率突缓处）即经验最优 k。

    复杂度:
        时间 O(sum_k n k d * iters) / 空间 O(n^2) 以下（不含轮廓系数）。

    陷阱:
        1. 每个 k 用**不同**种子会让曲线上下抖动，拐点判断失效——这里强制同种子。
        2. inertia 随 k 单调不增，k=n 时恒为 0，所以"肘部"必须人工看或配合业务解释，
           不要写成自动最优。
        3. 如果数据本身没有簇结构，曲线是平滑的，此时不存在肘部。

    参考:
        聚类数选择的经典启发式（elbow method）；更严格的做法见 gap statistic（Tibshirani 2001）。
    """
    x = as_matrix(X, "X")
    ks: List[int] = [int(k) for k in k_range]
    if not ks:
        raise ValueError("k_range 不能为空")
    out: Dict[int, float] = {}
    for k in ks:
        out[k] = float(kmeans(x, k, seed=seed)["inertia"])
    return out


def agglomerative(X: MatrixLike, k: int, linkage: str = "average") -> np.ndarray:
    """自底向上的凝聚层次聚类，用 Lance-Williams 公式合并最近的簇。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k: 目标簇数。
        linkage: 簇间距离口径，``"single"``（最近点）、``"complete"``（最远点）或
            ``"average"``（平均点距，UPGMA）。

    返回:
        np.ndarray，形状 (n_samples,)，标签 0..k-1 的 int 数组。

    算法:
        1. 把每个样本当作一个簇，算出 n×n 距离矩阵；
        2. 找距离最小的一对簇合并，按 Lance-Williams 更新簇间距离：
           single 取 min，complete 取 max，average 取按簇大小加权的平均；
        3. 重复到只剩 k 个簇，再按簇顺序重新编号。

    复杂度:
        时间 O(n^3)（朴素实现，每轮扫描 O(n^2)）/ 空间 O(n^2)。

    陷阱:
        1. **single linkage 会"链式"串联**：两个真正分开的簇可能通过一两个中间点连成一片，
           在含噪数据上尤其明显。complete/average 更常用。
        2. 决策是**贪心且不可撤销**的，早期合并错了无法回退。
        3. 合并距离必须用 Lance-Williams 更新后的簇间距离，直接对成员点重算虽然等价但更慢；
           注意 average 口径要按簇大小加权，写成等权平均是常见错误。
        4. 本实现不支持缺失值，含 NaN/inf 会在 ``as_matrix`` 处直接报错。

    参考:
        Lance & Williams, "A general theory of classificatory sorting strategies", 1967；
        Kaufman & Rousseeuw, "Finding Groups in Data", 1990。
    """
    x = as_matrix(X, "X")
    n = x.shape[0]
    _check_k(n, k)
    if linkage not in ("single", "complete", "average"):
        raise ValueError(f"linkage 只支持 single/complete/average，得到 {linkage!r}")

    dist = _pairwise_distances(x, x)
    np.fill_diagonal(dist, np.inf)  # 自己和自己不参与合并
    sizes = np.ones(n, dtype=float)
    members: List[List[int]] = [[i] for i in range(n)]

    for _ in range(n - k):
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        if i > j:
            i, j = j, i
        # 合并 j 进 i
        if linkage == "single":
            merged = np.minimum(dist[i], dist[j])
        elif linkage == "complete":
            merged = np.maximum(dist[i], dist[j])
        else:
            merged = (sizes[i] * dist[i] + sizes[j] * dist[j]) / (sizes[i] + sizes[j])
        dist[i] = merged
        dist[:, i] = merged
        dist[i, i] = np.inf
        # 冻结 j
        dist[j, :] = np.inf
        dist[:, j] = np.inf
        sizes[i] += sizes[j]
        members[i] = members[i] + members[j]
        members[j] = []

    labels = np.empty(n, dtype=int)
    next_label = 0
    for group in members:
        if group:
            labels[group] = next_label
            next_label += 1
    return labels


def dbscan(X: MatrixLike, eps: float, min_pts: int) -> np.ndarray:
    """DBSCAN：基于密度的聚类，能自动决定簇数并把低密度点标为噪声。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        eps: 邻域半径（用同一把尺子，所以必须先标准化）。
        min_pts: 成为核心点所需的邻域内最少点数（**含自己**）。

    返回:
        np.ndarray，形状 (n_samples,)，int 标签；噪声点为 -1，簇编号 0,1,2,...

    算法:
        1. 若某点 eps 邻域内点数 >= min_pts，它是核心点；
        2. 从每个未访问的核心点出发做**深度优先**扩展（内部用列表当栈），把密度可达的点
           归入同一簇。用 DFS 还是 BFS 只影响边界点的归属顺序，核心点的可达闭包是一样的。
           （核心点的邻居即使不是核心点也属于该簇，即"边界点"）；
        3. 不属于任何簇的点标记为 -1。

    复杂度:
        时间 O(n^2)（本实现先算满距离矩阵；用 KD 树可降到 O(n log n)）/ 空间 O(n^2)。

    陷阱:
        1. **eps 极难调**：量纲不统一时同一个 eps 在不同特征上含义完全不同，
           务必先标准化；经验做法是看 k-距离曲线的拐点。
        2. min_pts 是否包含自己会差 1：本实现**包含自己**，若与 sklearn 对比请注意
           其默认为 2*维度 且同样含自己。
        3. 边界点可能同时挨着两个簇，归属取决于遍历顺序——同一份数据换个点的顺序
           结果可能微变，论文里应说明这一点。
        4. eps 太大时所有点连成一簇，太小时全是噪声(-1)，务必打印簇数与噪声比例。

    参考:
        Ester, Kriegel, Sander & Xu, "A Density-Based Algorithm for Discovering Clusters
        in Large Spatial Databases with Noise", KDD 1996。
    """
    x = as_matrix(X, "X")
    n = x.shape[0]
    if eps <= 0:
        raise ValueError(f"eps 必须 > 0，得到 {eps}")
    if min_pts < 1:
        raise ValueError(f"min_pts 必须 >= 1，得到 {min_pts}")

    dist = _pairwise_distances(x, x)
    neighbors = [np.flatnonzero(dist[i] <= eps) for i in range(n)]
    core = np.array([nb.size >= min_pts for nb in neighbors], dtype=bool)

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        # 用列表当栈做**深度优先**扩展（末尾 pop）。核心点的可达闭包与 BFS 相同，
        # 只有边界点的归属顺序可能不同——见下面的陷阱 3。
        labels[i] = cluster_id
        stack = [i]
        while stack:
            cur = stack.pop()
            for nb in neighbors[cur]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if core[nb]:
                        stack.append(int(nb))
        cluster_id += 1
    return labels


def _log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """内部工具：多元高斯对数密度，形状 (n_samples,)。

    用 Cholesky 分解（``cho_solve``）而不是 ``np.linalg.inv``：既更快又避免
    "先求逆再相乘"带来的数值残差；对数行列式用 ``slogdet`` 的符号位判断，
    符号不为正说明协方差非正定，这里直接抛 ValueError（调用方必须补 reg_covar）。
    """
    n_features = x.shape[1]
    cov = 0.5 * (cov + cov.T)  # 强制对称，消掉浮点不对称
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("协方差矩阵非正定，请提高 reg_covar")
    diff = x - mean
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - 由 slogdet 提前拦截
        raise ValueError("协方差矩阵 Cholesky 分解失败，请提高 reg_covar") from exc
    sol = np.linalg.solve(chol, np.asfortranarray(diff.T))
    maha = np.sum(sol ** 2, axis=0)
    return -0.5 * (n_features * np.log(2.0 * np.pi) + logdet + maha)


def gmm_em(
    X: MatrixLike,
    k: int,
    seed: Optional[int] = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    reg_covar: float = 1e-6,
) -> dict:
    """多元高斯混合模型的 EM 估计：E 步算后验责任，M 步加权更新权重/均值/协方差。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k: 混合分量个数，1 <= k <= n_samples。
        seed: 传给 K-means++ 的随机种子；None 表示使用 ``DEFAULT_SEED``。
        max_iter: 最大 EM 迭代轮数。
        tol: 收敛阈值，判据是"平均对数似然的增量 <= tol"。
        reg_covar: 加到每个协方差对角上的正数，保证可逆（必须 >= 0）。

    返回:
        dict，键为：
        ``labels``      形状 (n_samples,) 的 int 数组，后验责任最大的分量；
        ``means``       形状 (k, n_features) 的分量均值；
        ``covariances`` 形状 (k, n_features, n_features) 的分量协方差（**含** reg_covar）；
        ``weights``     形状 (k,) 的混合权重，和为 1；
        ``loglik``      float，收敛时的平均对数似然（每个样本）；
        ``n_iter``      int，实际迭代轮数；
        ``bic``         float，k*log(n) - 2*总对数似然（越小越好）。

    算法:
        1. 用 ``kmeans`` 的中心作为初始均值，簇内样本比例作为初始权重，
           簇内协方差（也加 reg_covar）作为初始协方差——比随机初始化稳定得多；
        2. E 步：责任 r_ij ∝ w_j * N(x_i | mu_j, Sigma_j)，按对数密度计算后再
           减去每行的最大值做 softmax（避免 exp 下溢）；
        3. M 步：N_j = sum_i r_ij，w_j = N_j/n，mu_j = sum_i r_ij x_i / N_j，
           Sigma_j = sum_i r_ij (x_i-mu_j)(x_i-mu_j)' / N_j + reg_covar*I；
        4. 平均对数似然增量 <= tol 或达到 max_iter 时停止。**空分量**（``N_j <= 1e-10``）不做
           M 步更新，而是把权重压到 1e-12、**均值重置为 ``x[0]``**、协方差重置为
           ``全局协方差 + reg_covar*I``；否则除以 ``N_j`` 会直接产生 NaN。
        5. 收敛后再用最终参数重算一次责任与对数似然，保证返回的 ``labels``/``loglik`` 自洽。

        BIC 用标准形式 ``n_params * log(n) - 2 * loglik_total``，其中
        ``n_params = (k-1) + k*d + k*d*(d+1)/2``（权重 k-1 个自由度 + 各分量均值 k*d 个 +
        各分量对称协方差 k*d*(d+1)/2 个），``loglik_total = loglik * n``（本函数返回的是
        **平均**对数似然，所以乘回 n）。注意这里**没有**直接套
        ``k*(d + d(d+1)/2)*log(n)`` —— 那种写法漏掉了权重的 k-1 个参数。
        （Dempster, Laird & Rubin 1977；BIC 见 Schwarz 1978。）

    复杂度:
        时间 O(n k d^3 * n_iter)（d^3 来自 Cholesky）/ 空间 O(n k + k d^2)。

    陷阱:
        1. **必须加 reg_covar**：某一分量只剩一个点（或所有点共线）时经验协方差奇异，
           ``slogdet`` 返回 0 甚至负号，对数似然直接变成 -inf。
        2. 权重加权后的协方差要除以 N_j = sum_i r_ij，**不是**簇内样本数——用硬标签
           计数会把责任很小的点也等权算进去。
        3. EM 只保证收敛到局部极大，且与初始化有关；labels 取的是后验 argmax，
           与 K-means 的硬分配在边界点上会不同。
        4. loglik 返回的是**平均**对数似然；BIC 里必须换回总和再乘 -2，混用会差 n 倍。

    参考:
        Dempster, Laird & Rubin, "Maximum Likelihood from Incomplete Data via the EM
        Algorithm", JRSS-B 1977；Schwarz, "Estimating the Dimension of a Model", 1978。
    """
    x = as_matrix(X, "X")
    n_samples, n_features = x.shape
    _check_k(n_samples, k)
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，得到 {max_iter}")
    if tol < 0:
        raise ValueError(f"tol 必须 >= 0，得到 {tol}")
    if reg_covar < 0:
        raise ValueError(f"reg_covar 必须 >= 0，得到 {reg_covar}")

    eye = np.eye(n_features) * reg_covar
    init = kmeans(x, k, seed=seed)
    means = np.array(init["centers"], dtype=float)
    labels = np.asarray(init["labels"], dtype=int)
    weights = np.empty(k, dtype=float)
    global_cov = np.cov(x.T, bias=True).reshape(n_features, n_features)
    covariances = np.empty((k, n_features, n_features), dtype=float)
    for j in range(k):
        members = x[labels == j]
        weights[j] = members.shape[0] / n_samples
        if members.shape[0] > n_features:
            covariances[j] = np.cov(members.T, bias=True).reshape(n_features, n_features) + eye
        else:
            covariances[j] = global_cov + eye
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()

    prev_loglik = -np.inf
    loglik = -np.inf
    n_iter = 0
    for it in range(1, max_iter + 1):
        # --- E 步 ---
        log_prob = np.empty((n_samples, k), dtype=float)
        for j in range(k):
            log_prob[:, j] = _log_gaussian_pdf(x, means[j], covariances[j])
        weighted = np.log(weights)[None, :] + log_prob
        row_max = weighted.max(axis=1, keepdims=True)
        resp = np.exp(weighted - row_max)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.where(resp_sum > 0.0, resp_sum, 1.0)
        resp = resp / resp_sum
        loglik = float(np.mean(row_max[:, 0] + np.log(resp_sum[:, 0])))

        # --- M 步 ---
        nk = resp.sum(axis=0)
        weights = nk / n_samples
        for j in range(k):
            if nk[j] <= 1e-10:
                weights[j] = 1e-12
                means[j] = x[0]
                covariances[j] = global_cov + eye
                continue
            mu = (resp[:, j][:, None] * x).sum(axis=0) / nk[j]
            diff = x - mu
            cov = (resp[:, j][:, None] * diff).T @ diff / nk[j]
            means[j] = mu
            covariances[j] = cov + eye
        weights = weights / weights.sum()

        n_iter = it
        if it > 1 and loglik - prev_loglik <= tol:
            break
        prev_loglik = loglik

    # 收敛后用最终参数重算一次责任与对数似然，保证返回的 labels/loglik 自洽。
    log_prob = np.empty((n_samples, k), dtype=float)
    for j in range(k):
        log_prob[:, j] = _log_gaussian_pdf(x, means[j], covariances[j])
    weighted = np.log(weights)[None, :] + log_prob
    row_max = weighted.max(axis=1, keepdims=True)
    resp = np.exp(weighted - row_max)
    resp_sum = resp.sum(axis=1, keepdims=True)
    resp_sum = np.where(resp_sum > 0.0, resp_sum, 1.0)
    resp = resp / resp_sum
    loglik = float(np.mean(row_max[:, 0] + np.log(resp_sum[:, 0])))
    labels = np.argmax(resp, axis=1).astype(int)

    n_params = (k - 1) + k * n_features + k * n_features * (n_features + 1) // 2
    bic = float(n_params * np.log(n_samples) - 2.0 * loglik * n_samples)
    return {
        "labels": labels,
        "means": means,
        "covariances": covariances,
        "weights": weights,
        "loglik": loglik,
        "n_iter": n_iter,
        "bic": bic,
    }


def spectral_clustering(
    X: MatrixLike,
    k: int,
    sigma: Optional[float] = None,
    seed: Optional[int] = None,
    n_neighbors: int = 3,
) -> dict:
    """谱聚类：高斯相似度 → 归一化拉普拉斯 → 前 k 个特征向量 → K-means。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k: 簇数。
        sigma: 高斯核带宽。给定时用**全局带宽**（经典 Ng-Jordan-Weiss）；
            None（默认）改用**局部尺度**：第 i 个点用自己的第 ``n_neighbors``
            近邻距离 sigma_i 做带宽（Zelnik-Manor & Perona 的 self-tuning，这里
            按无向图口径对称化，见"算法"第 1 步）。
        seed: 传给最后 K-means 的随机种子；None 表示使用 ``DEFAULT_SEED``。
        n_neighbors: 局部尺度用的近邻阶数，仅当 ``sigma is None`` 时生效，必须 >= 1。
            取值不能太大：n_neighbors 越大 sigma_i 越接近"全局中位距离"，
            密度差异大的数据会重新退化到全局带宽的效果。

    返回:
        dict，键为：
        ``labels``      形状 (n_samples,) 的 int 数组；
        ``eigenvalues`` 形状 (k,) 的归一化拉普拉斯前 k 个最小特征值（升序）；
        ``affinity``    形状 (n_samples, n_samples) 的高斯相似度矩阵（已对称化，未做行归一化）；
        ``sigma``       float，实际使用的带宽：给定 sigma 时原样返回，否则返回
                        各点局部尺度 sigma_i 的中位数（仅作报告用，不代表核是全局的）。

    算法:
        1. 相似度：
           - ``sigma`` 给定时 W_ij = exp(-||x_i-x_j||^2 / (2 sigma^2))；
           - 否则先算 D_ij 并取其第 ``n_neighbors`` 小的值作为 sigma_i，
             W_ij = exp(-||x_i-x_j||^2 / (sigma_i * sigma_j))，再取
             (W + W^T)/2 强制对称（原论文用 sigma_i^2 的算术平均，等价）。
           两种口径都把对角线置 0；
        2. 度矩阵 D = diag(row sum)，归一化拉普拉斯 ``L_sym = I - D^-1/2 W D^-1/2``；
        3. 取 ``L_sym`` 最小的 k 个特征值对应的特征向量 U（形状 n×k），
           按行归一化到单位长度（Ng-Jordan-Weiss 的做法，也叫"谱嵌入"）；
        4. 对 U 的行跑 ``kmeans``，得到的硬标签就是聚类结果。
        （Shi & Malik 2000 的归一化割；Ng, Jordan & Weiss 2002 的算法；
        Zelnik-Manor & Perona 2004 的局部尺度。）

    复杂度:
        时间 O(n^3)（稠密特征分解，与 k 无关）/ 空间 O(n^2)。

    陷阱:
        1. **带宽是真正的超参**：太小则相似度矩阵退化成近邻图的单位阵（图不连通，
           特征向量只在簇内非零，K-means 会把每簇再切开）；太大则所有点彼此相似，
           谱结构消失。全局"距离中位数"在**簇密度差异大**时必然失败：中位数由
           稠密簇的点对决定，对稀疏簇来说远大于其内部尺度，于是稀疏簇各点与其他
           所有点都"很远"，图退化成孤立点集。这正是本实现默认用局部尺度的原因。
        2. 必须做**行归一化**再用 K-means：不做的话特征向量的模长会随簇大小变化，
           K-means 会按模长而不是方向切分，小簇被吞掉。
        3. 特征值重根时（例如两个完全对称的簇）特征向量在该子空间内任意旋转，
           不同 LAPACK 版本给出的基不同——标签可能整体等价但数值不一致。
        4. 本实现是稠密的 O(n^3)，n 上千就非常慢；生产环境要用稀疏图 + Lanczos。
        5. 局部尺度核是**数据依赖**的，所以 ``affinity`` 的绝对值不能跨数据集比较；
           不同实现（sklearn 的 ``spectral_clustering`` 用最近邻图）给出的 affinity
           也完全不同，只应比对 labels。

    参考:
        Shi & Malik, "Normalized Cuts and Image Segmentation", PAMI 2000；
        Ng, Jordan & Weiss, "On Spectral Clustering: Analysis and an Algorithm", NIPS 2002；
        Zelnik-Manor & Perona, "Self-Tuning Spectral Clustering", NIPS 2004。
    """
    x = as_matrix(X, "X")
    n_samples = x.shape[0]
    _check_k(n_samples, k)
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors 必须 >= 1，得到 {n_neighbors}")

    dist = _pairwise_distances(x, x)
    if sigma is not None:
        if sigma <= 0:
            raise ValueError(f"sigma 必须 > 0，得到 {sigma}")
        affinity = np.exp(-(dist ** 2) / (2.0 * sigma * sigma))
        sigma_report = float(sigma)
    else:
        nn = min(n_neighbors, max(n_samples - 1, 1))
        work = dist.copy()
        np.fill_diagonal(work, np.inf)
        local = np.sort(work, axis=1)[:, nn - 1].copy()
        # 完全重合的点（局部尺度为 0）会给出 0/0，兜底为 1.0。
        local = np.where(local > 1e-12, local, 1.0)
        affinity = np.exp(-(dist ** 2) / (local[:, None] * local[None, :]))
        affinity = 0.5 * (affinity + affinity.T)
        sigma_report = float(np.median(local))
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    degree = np.where(degree > 0.0, degree, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    laplacian = np.eye(n_samples) - (d_inv_sqrt[:, None] * affinity) * d_inv_sqrt[None, :]
    laplacian = 0.5 * (laplacian + laplacian.T)

    values, vectors = np.linalg.eigh(laplacian)
    embedding = vectors[:, :k]
    norms = np.sqrt(np.sum(embedding ** 2, axis=1))
    norms = np.where(norms > 1e-12, norms, 1.0)
    embedding = embedding / norms[:, None]

    labels = np.asarray(kmeans(embedding, k, seed=seed)["labels"], dtype=int)
    return {
        "labels": labels,
        "eigenvalues": values[:k],
        "affinity": affinity,
        "sigma": sigma_report,
    }


def fuzzy_cmeans(
    X: MatrixLike,
    k: int,
    m: float = 2.0,
    seed: Optional[int] = None,
    max_iter: int = 150,
    tol: float = 1e-8,
) -> dict:
    """模糊 C 均值（FCM）：隶属度 u_ij ∈ [0,1] 且每行和为 1 的软划分。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k: 簇数。
        m: 模糊指数，必须 > 1；m=2 是常用默认（m→1 退化为硬 K-means）。
        seed: 随机种子，只用于"用 K-means 初始化中心"这一步。
        max_iter: 最大迭代轮数。
        tol: 收敛阈值，判据是"隶属度矩阵的最大变化量 <= tol"。

    返回:
        dict，键为：
        ``labels``     形状 (n_samples,) 的 int 数组，取隶属度最大的簇；
        ``membership`` 形状 (n_samples, k) 的隶属度矩阵，每行和为 1；
        ``centers``    形状 (k, n_features) 的模糊中心；
        ``objective``  float，目标函数 J = sum_i sum_j u_ij^m ||x_i - c_j||^2；
        ``n_iter``     int，实际迭代轮数。

    算法:
        1. 初始化：跑一次 ``kmeans``，用它的中心作为初始 c_j（比随机中心稳）；
        2. 更新隶属度：u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1))，d_ij = ||x_i - c_j||；
           某点与某中心重合（d_ij = 0）时把该点隶属度置成该簇的 one-hot；
        3. 更新中心：c_j = sum_i u_ij^m x_i / sum_i u_ij^m；
        4. 隶属度最大变化 <= tol 或达到 max_iter 时停止。
        （Bezdek 1981；Dunn 1973。）

    复杂度:
        时间 O(n k d * n_iter) / 空间 O(n k)。

    陷阱:
        1. **d_ij = 0 会让 u_ij 出现 0/0**：必须显式处理重合点，否则整行变 NaN 并
           在下一步污染所有中心。
        2. m 越接近 1，隶属度越接近 0/1，迭代越容易在两个划分之间震荡而不收敛；
           m 太大则所有 u_ij → 1/k，软划分失去分辨力。m ∈ [1.5, 2.5] 比较常用。
        3. labels 取 argmax 是**事后硬化**，扁平隶属度（多点 u≈1/k）的点归属对
           初始化和迭代轮数都敏感，报告结果时应同时给出最大隶属度的分布。
        4. 这里的收敛判据是隶属度变化而不是目标函数变化：J 在 m 较大时几乎不变，
           用它做判据会提前停止。

    参考:
        Dunn, "A Fuzzy Relative of the ISODATA Process", 1973；
        Bezdek, "Pattern Recognition with Fuzzy Objective Function Algorithms", 1981。
    """
    x = as_matrix(X, "X")
    n_samples, n_features = x.shape
    _check_k(n_samples, k)
    if m <= 1.0:
        raise ValueError(f"模糊指数 m 必须 > 1，得到 {m}")
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，得到 {max_iter}")
    if tol < 0:
        raise ValueError(f"tol 必须 >= 0，得到 {tol}")

    centers = np.array(kmeans(x, k, seed=seed)["centers"], dtype=float)
    exponent = 2.0 / (m - 1.0)
    membership = np.full((n_samples, k), 1.0 / k, dtype=float)
    n_iter = 0
    for it in range(1, max_iter + 1):
        dist = _pairwise_distances(x, centers)
        zero_rows = np.any(dist <= 1e-12, axis=1)
        safe = np.where(dist <= 1e-12, 1.0, dist)
        ratio = safe[:, :, None] / safe[:, None, :]
        new_membership = 1.0 / np.sum(ratio ** exponent, axis=2)
        if np.any(zero_rows):
            # 点与某个中心重合：把该点完全判给第一个重合的中心（硬 one-hot）。
            for i in np.flatnonzero(zero_rows):
                new_membership[i, :] = 0.0
                new_membership[i, int(np.argmin(dist[i]))] = 1.0
        change = float(np.max(np.abs(new_membership - membership)))
        membership = new_membership

        powered = membership ** m
        denom = powered.sum(axis=0)
        denom = np.where(denom > 1e-12, denom, 1e-12)
        centers = (powered.T @ x) / denom[:, None]

        n_iter = it
        if change <= tol:
            break

    dist = _pairwise_distances(x, centers)
    objective = float(np.sum((membership ** m) * (dist ** 2)))
    labels = np.argmax(membership, axis=1).astype(int)
    return {
        "labels": labels,
        "membership": membership,
        "centers": centers,
        "objective": objective,
        "n_iter": n_iter,
    }


def davies_bouldin_score(X: MatrixLike, labels: ArrayLike) -> float:
    """Davies-Bouldin 指数：簇内散度与簇心距离之比的最大值再对簇平均，越小越好。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        labels: 每个样本的簇标签，长度 n_samples。

    返回:
        float，DB 指数，取值 >= 0；完全重合的簇会让它趋于 +inf。仅 1 个簇时返回 0.0
        （"无簇间比较"，与 ``silhouette_score`` 的 0.0 约定一致，不是"最好"）。

    算法:
        S_j = 簇 j 内样本到簇心 c_j 的**平均**欧氏距离（散度）；
        M_jl = ||c_j - c_l||（簇心距离）；
        R_j = max_{l != j} (S_j + S_l) / M_jl；DB = (1/k) sum_j R_j。
        （Davies & Bouldin 1979；与 sklearn 的 ``davies_bouldin_score`` 同口径。）

    复杂度:
        时间 O(n d + k^2 d) / 空间 O(n d)。

    陷阱:
        1. 两个簇心完全重合（M_jl = 0）时比值发散：这里按约定返回 ``inf``，
           调用方应把 inf 当成"这个划分不可用"而不是"很大"。
        2. S_j 用**平均**距离而不是距离之和：用和会让指标随簇大小单调变化，
           大簇被冤枉，选出来的 k 系统性偏大。
        3. 指标依赖尺度，必须先标准化特征；它也不适用于任意形状的簇
           （DB 假设簇是球形的，DBSCAN 找出的环形簇会被判很差）。

    参考:
        Davies & Bouldin, "A Cluster Separation Measure", PAMI 1979。
    """
    x = as_matrix(X, "X")
    lab = np.asarray(labels).ravel().astype(int)
    if lab.size != x.shape[0]:
        raise ValueError(f"labels 长度 {lab.size} 与样本数 {x.shape[0]} 不一致")
    unique = np.unique(lab)
    k = unique.size
    if k < 2:
        return 0.0

    centers = np.empty((k, x.shape[1]), dtype=float)
    scatter = np.empty(k, dtype=float)
    for idx, c in enumerate(unique):
        members = x[lab == c]
        center = members.mean(axis=0)
        centers[idx] = center
        scatter[idx] = float(np.mean(np.sqrt(np.sum((members - center) ** 2, axis=1))))

    center_dist = _pairwise_distances(centers, centers)
    ratios = np.zeros(k, dtype=float)
    for j in range(k):
        best = 0.0
        for l in range(k):
            if l == j:
                continue
            denom = center_dist[j, l]
            if denom <= 1e-12:
                return float("inf")
            best = max(best, (scatter[j] + scatter[l]) / denom)
        ratios[j] = best
    return float(np.mean(ratios))


def calinski_harabasz_score(X: MatrixLike, labels: ArrayLike) -> float:
    """Calinski-Harabasz（方差比）指数：簇间离散度 / 簇内离散度，越大越好。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        labels: 每个样本的簇标签，长度 n_samples。

    返回:
        float，CH 指数，取值 >= 0；无定义时（只有 1 个簇，或簇内离散度为 0）
        返回 0.0，调用方必须理解成"不可用"而不是"最差"。

    算法:
        BGSS = sum_j n_j ||c_j - c_bar||^2（簇间平方和，自由度 k-1）；
        WGSS = sum_j sum_{i in C_j} ||x_i - c_j||^2（簇内平方和，自由度 n-k）；
        CH = [BGSS / (k-1)] / [WGSS / (n-k)]。
        （Caliński & Harabasz 1974；与 sklearn 的 ``calinski_harabasz_score`` 同口径。
        注意此处**没有**再乘 ``(n-k)/(k-1)`` —— 那是另一个被误传的"等价写法"。）

    复杂度:
        时间 O(n d) / 空间 O(n d)。

    陷阱:
        1. k = n 时 WGSS = 0、n-k = 0，公式 0/0：这里返回 0.0，不要当成最优。
        2. CH 随 k 单调倾向（分子自由度惩罚不够强），在无簇结构的数据上也会
           偏好较大的 k，所以它只能用来**比较同一份数据**的候选划分。
        3. 同样是球形簇口径，对环形/月牙形数据没有意义。

    参考:
        Caliński & Harabasz, "A Dendrite Method for Cluster Analysis", 1974。
    """
    x = as_matrix(X, "X")
    lab = np.asarray(labels).ravel().astype(int)
    n_samples, n_features = x.shape
    if lab.size != n_samples:
        raise ValueError(f"labels 长度 {lab.size} 与样本数 {n_samples} 不一致")
    unique = np.unique(lab)
    k = unique.size
    if k < 2 or k >= n_samples:
        return 0.0

    grand_mean = x.mean(axis=0)
    bgss = 0.0
    wgss = 0.0
    for c in unique:
        members = x[lab == c]
        center = members.mean(axis=0)
        bgss += members.shape[0] * float(np.sum((center - grand_mean) ** 2))
        wgss += float(np.sum((members - center) ** 2))
    if wgss <= 1e-300:
        return 0.0
    return float((bgss / (k - 1)) / (wgss / (n_samples - k)))


def gap_statistic(
    X: MatrixLike,
    k_max: int = 6,
    n_refs: int = 10,
    seed: Optional[int] = None,
) -> dict:
    """Gap 统计量：把 log(簇内离散度) 与均匀参考分布对比，挑第一个"不再显著下降"的 k。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        k_max: 尝试的最大簇数，2 <= k_max <= n_samples。
        n_refs: 每个 k 的参考数据集个数（蒙特卡洛次数）。
        seed: 随机种子；参考集、以及每个 k 的 K-means 初始化都由它派生。
            ``None`` 表示回落到 ``DEFAULT_SEED``（**不是** 0）。

    返回:
        dict，键为：
        ``gap``    形状 (k_max,) 的 float 数组，下标 k-1 对应簇数 k 的 gap 值；
        ``sk``     形状 (k_max,) 的 float 数组，gap 的标准误
                   ``sk[k] = sd_ref * sqrt(1 + 1/n_refs)``（下标同上）；
        ``k_hat``  int，第一个满足 ``gap[k] >= gap[k+1] - sk[k+1]`` 的 k（1-based）；
        ``log_w`` 形状 (k_max,) 的 float 数组，观测数据的 log(W_k)。

    算法:
        1. 对 k = 1..k_max 在**原数据**上跑 ``kmeans``，W_k = sum_j sum_{i in C_j}||x_i-c_j||^2
           （用簇内距离**和**，与原论文一致，而不是方差或均值）；
        2. 生成 n_refs 个参考集：每一维在观测数据的 [min, max] 上独立均匀采样，
           形状与原数据相同（"无簇结构"的零假设）；
        3. 对每个参考集重复第 1 步得到 W_kb；Gap(k) = (1/B) sum_b log(W_kb) - log(W_k)；
           sk(k) = sd_b(log W_kb) * sqrt(1 + 1/B)；
        4. k_hat = 第一个满足上述判据的 k；若直到 k-1 = k_max-1 都不满足，**直接返回
           ``k_max``**（即"用满你给的上限"），并不是返回"gap 最大的那个 k"。这两种退回口径
           会给出不同的答案，论文里必须写清你用的是哪一种。
        （Tibshirani, Walther & Hastie 2001。）

    复杂度:
        时间 O(n_refs * k_max * n k d * iters) / 空间 O(n d + n k_max)。

    陷阱:
        1. **参考分布必须与原数据同尺度**：对每一维用观测的 [min, max] 做均匀采样，
           而不是标准正态；否则 gap 的绝对值没有意义（原论文也强调这一点）。
        2. W_k 用距离之和，k=n 时恒为 0、log W → -inf，所以 k_max 必须显著小于 n。
        3. gap 曲线常常很平（真实 k 与 k+1 的 gap 差在 sk 之内），k_hat 对 n_refs
           和随机种子都敏感；论文里应报告 gap/±sk 曲线而不是只报一个 k_hat。
        4. 本实现为"同一个 k 用同一个种子、同一个 k 复用同一批参考集"的确定性口径，
           因此两次调用结果完全一致；换成真正独立的参考集会得到不同的 k_hat。
        5. 判据只在 k = 1..k_max-1 上扫描（要比较相邻两项），所以实际可能返回的 k_hat 是
           ``k_max`` 当且仅当一次都没触发；不要在论文里把 k_hat = k_max 解释成"最优 k 就是
           上限"，它更可能是"数据里没有清晰簇结构"或"上限给太小"。

    参考:
        Tibshirani, Walther & Hastie, "Estimating the number of clusters in a data set
        via the gap statistic", JRSS-B 2001。
    """
    x = as_matrix(X, "X")
    n_samples, n_features = x.shape
    if k_max < 2:
        raise ValueError(f"k_max 必须 >= 2，得到 {k_max}")
    _check_k(n_samples, k_max)
    if n_refs < 1:
        raise ValueError(f"n_refs 必须 >= 1，得到 {n_refs}")

    def dispersion(points: np.ndarray, k: int, s: Optional[int]) -> float:
        res = kmeans(points, k, seed=s)
        centers = res["centers"]
        lab = np.asarray(res["labels"], dtype=int)
        d = _pairwise_distances(points, centers)
        return float(np.sum(d[np.arange(points.shape[0]), lab]))

    lo = x.min(axis=0)
    hi = x.max(axis=0)
    # seed=None 时显式回落到本仓库的统一默认种子，而不是 0：否则"不传 seed"与
    # "显式传 seed=0"会得到同一组参考集，且与其它模块的 None 口径不一致。
    base_seed = int(seed) if seed is not None else int(DEFAULT_SEED)
    log_w = np.empty(k_max, dtype=float)
    log_w_ref = np.empty((n_refs, k_max), dtype=float)
    for k in range(1, k_max + 1):
        log_w[k - 1] = np.log(max(dispersion(x, k, seed), 1e-300))
        for b in range(n_refs):
            # 每个 (k, b) 用稳定的派生种子：改动 k_max 或 n_refs 都不会串味。
            ref_seed = base_seed * 1000003 + k * 1009 + b
            ref_rng = rng(ref_seed)
            ref = lo + (hi - lo) * ref_rng.random((n_samples, n_features))
            log_w_ref[b, k - 1] = np.log(max(dispersion(ref, k, ref_seed + 7919), 1e-300))

    mean_ref = log_w_ref.mean(axis=0)
    gap = mean_ref - log_w
    if n_refs > 1:
        sd_ref = log_w_ref.std(axis=0, ddof=1)
    else:
        sd_ref = np.zeros(k_max, dtype=float)
    sk = sd_ref * np.sqrt(1.0 + 1.0 / n_refs)

    k_hat = k_max
    for k in range(1, k_max):
        if gap[k - 1] >= gap[k] - sk[k]:
            k_hat = k
            break
    return {
        "gap": gap,
        "sk": sk,
        "k_hat": int(k_hat),
        "log_w": log_w,
    }


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict，键名简短 ASCII，值均为 int/float/list，固定种子下两次调用完全一致：
        ``kmeans_inertia`` 三簇数据的 SSE、``kmeans_silhouette`` 轮廓系数、
        ``kmeans_labels`` 各簇样本数、``elbow_k3`` 肘部曲线在 k=3 的值、
        ``agg_labels`` 层次聚类各簇样本数、``dbscan_labels`` DBSCAN 标签序列、
        ``dbscan_n_noise`` 噪声点数、``silhouette_single`` 单簇时的轮廓系数、
        ``gmm_recovered`` GMM 是否逐一还原三簇、``gmm_loglik`` 平均对数似然、
        ``gmm_n_iter`` EM 迭代轮数、``gmm_bic`` BIC、
        ``spectral_ring_exact`` 同心圆上谱聚类是否与真实标签完全一致、
        ``kmeans_ring_exact`` 同一数据上 K-means 是否失败（应为 False）、
        ``fuzzy_row_sums_ok`` 隶属度每行和是否为 1、``fuzzy_max_dev`` 行和最大偏差、
        ``fuzzy_n_iter`` FCM 迭代轮数、
        ``db_perfect`` / ``db_random`` 完美分组与随机标签的 DB 值、
        ``ch_perfect`` / ``ch_random`` 对应的 CH 值、``gap_k_hat`` gap 选出的 k。

    算法:
        构造三个中心分别在 (-5,-5)、(0,6)、(6,-4)、标准差 0.5 的二维高斯簇，
        每簇 30 个点，固定种子生成；k=3 时应当无歧义地恢复分组。
        另构造两个同心圆（半径 1.5 / 3.5，每环 40 点，固定角度抖动）：欧氏距离下
        K-means 必然把两个环各切一半，而谱聚类用相似度图的连通结构能完整分开——
        这是谱聚类的"存在理由"算例。

    复杂度:
        时间 O(n^2) / 空间 O(n^2)。

    陷阱:
        自测数据的簇间距（约 8~11）远大于簇内标准差 0.5，所以任何合理实现都应得到
        相同分组；如果改小间距，K-means 的随机性就会让断言变得脆弱，不要那样改。

    参考:
        本模块各函数的参考文献。
    """
    gen = rng(12345)
    centers = np.array([[-5.0, -5.0], [0.0, 6.0], [6.0, -4.0]])
    parts = [c + 0.5 * gen.standard_normal((30, 2)) for c in centers]
    x = np.vstack(parts)

    km = kmeans(x, 3, seed=7)
    labels = km["labels"]
    counts = [int(np.count_nonzero(labels == j)) for j in range(3)]
    sil = silhouette_score(x, labels)
    elbow = elbow_curve(x, [1, 2, 3, 4], seed=7)

    agg = agglomerative(x, 3, linkage="average")
    agg_counts = [int(np.count_nonzero(agg == j)) for j in range(3)]

    ds = np.vstack([
        np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.2], [-0.1, 0.15]]),
        np.array([[5.0, 5.0], [5.2, 5.1], [5.1, 4.8], [4.85, 5.05]]),
        np.array([[10.0, 0.0]]),  # 孤立点 → 噪声
    ])
    db_labels = dbscan(ds, eps=0.6, min_pts=3)

    # ---- 追加部分：GMM / 谱聚类 / FCM / DB / CH / gap ----
    # GMM 在三簇上必须逐一还原分组（这里用真值下标直接比对，是独立结论校验）。
    gm = gmm_em(x, 3, seed=11)
    gmm_labels = np.asarray(gm["labels"], dtype=int)
    gmm_recovered = True
    for j in range(3):
        seg = gmm_labels[30 * j:30 * (j + 1)]
        hit = int(np.bincount(seg, minlength=3).argmax())
        if int(np.count_nonzero(seg == hit)) != 30:
            gmm_recovered = False
    if not gmm_recovered:
        raise AssertionError(f"gmm_em 未还原三簇分组：{gmm_labels}")
    # 权重之和必须为 1（独立于 EM 的归一化自洽性检查）。
    if abs(float(np.sum(np.asarray(gm["weights"]))) - 1.0) > 1e-9:
        raise AssertionError(f"gmm 权重和 != 1：{np.sum(gm['weights'])}")

    # 同心圆算例：K-means 必败，谱聚类必须全对。
    ang = gen.random(40) * 2.0 * np.pi
    ang2 = gen.random(40) * 2.0 * np.pi
    r1 = 1.5 + 0.06 * gen.standard_normal(40)
    r2 = 3.5 + 0.06 * gen.standard_normal(40)
    ring = np.vstack([
        np.column_stack([r1 * np.cos(ang), r1 * np.sin(ang)]),
        np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)]),
    ])
    ring_truth = np.array([0] * 40 + [1] * 40, dtype=int)
    spec = spectral_clustering(ring, 2, seed=3)
    spec_labels = np.asarray(spec["labels"], dtype=int)
    spec_exact = bool(np.array_equal(spec_labels, ring_truth)) or bool(
        np.array_equal(1 - spec_labels, ring_truth)
    )
    km_ring = np.asarray(kmeans(ring, 2, seed=3)["labels"], dtype=int)
    km_ring_exact = bool(np.array_equal(km_ring, ring_truth)) or bool(
        np.array_equal(1 - km_ring, ring_truth)
    )
    if not spec_exact:
        raise AssertionError(f"谱聚类在同心圆上未完全正确：{spec_labels}")
    if km_ring_exact:
        raise AssertionError("K-means 竟然分对了同心圆，算例失去意义，请加强环形数据")

    fm = fuzzy_cmeans(x, 3, m=2.0, seed=5)
    memb = np.asarray(fm["membership"], dtype=float)
    row_sums = memb.sum(axis=1)
    fuzzy_max_dev = float(np.max(np.abs(row_sums - 1.0)))
    fuzzy_row_sums_ok = bool(fuzzy_max_dev <= 1e-9)
    if not fuzzy_row_sums_ok:
        raise AssertionError(f"fuzzy_cmeans 隶属度行和偏差 {fuzzy_max_dev} 过大")
    if float(np.min(memb)) < 0.0 or float(np.max(memb)) > 1.0 + 1e-12:
        raise AssertionError("fuzzy_cmeans 隶属度越界")

    # DB 越小越好、CH 越大越好：在完美分组 vs 固定随机标签上的方向性必须成立。
    true_labels = np.repeat([0, 1, 2], 30)
    rand_labels = np.asarray(gen.integers(0, 3, size=x.shape[0]), dtype=int)
    db_perfect = davies_bouldin_score(x, true_labels)
    db_random = davies_bouldin_score(x, rand_labels)
    ch_perfect = calinski_harabasz_score(x, true_labels)
    ch_random = calinski_harabasz_score(x, rand_labels)
    if not db_perfect * 5.0 < db_random:
        raise AssertionError(f"DB 方向性错误：perfect={db_perfect}, random={db_random}")
    if not ch_perfect > 3.0 * ch_random:
        raise AssertionError(f"CH 方向性错误：perfect={ch_perfect}, random={ch_random}")

    gs = gap_statistic(x, k_max=6, n_refs=10, seed=17)
    if int(gs["k_hat"]) != 3:
        raise AssertionError(f"gap_statistic 的 k_hat={gs['k_hat']}，期望 3")

    return {
        "kmeans_inertia": round(km["inertia"], 6),
        "kmeans_n_iter": int(km["n_iter"]),
        "kmeans_silhouette": round(sil, 6),
        "kmeans_labels": sorted(counts),
        "elbow_k3": round(elbow[3], 6),
        "agg_labels": sorted(agg_counts),
        "dbscan_labels": [int(v) for v in db_labels],
        "dbscan_n_noise": int(np.count_nonzero(db_labels == -1)),
        "silhouette_single": silhouette_score(x, np.zeros(x.shape[0], dtype=int)),
        "gmm_recovered": bool(gmm_recovered),
        "gmm_loglik": round(float(gm["loglik"]), 6),
        "gmm_n_iter": int(gm["n_iter"]),
        "gmm_bic": round(float(gm["bic"]), 6),
        "spectral_ring_exact": bool(spec_exact),
        "kmeans_ring_exact": bool(km_ring_exact),
        "spectral_eigenvalues": [round(float(v), 6) for v in np.asarray(spec["eigenvalues"]).ravel()],
        "fuzzy_row_sums_ok": bool(fuzzy_row_sums_ok),
        "fuzzy_max_dev": round(fuzzy_max_dev, 9),
        "fuzzy_n_iter": int(fm["n_iter"]),
        "db_perfect": round(float(db_perfect), 6),
        "db_random": round(float(db_random), 6),
        "ch_perfect": round(float(ch_perfect), 6),
        "ch_random": round(float(ch_random), 6),
        "gap_k_hat": int(gs["k_hat"]),
        "gap_values": [round(float(v), 6) for v in np.asarray(gs["gap"]).ravel()],
    }
