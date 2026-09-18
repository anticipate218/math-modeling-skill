"""聚类算法：K-means++、K-means、轮廓系数、肘部曲线、层次聚类、DBSCAN。

这些实现是**教学透明版**：目标是把论文里要交代的每一步都摊开写清楚，
而不是追求工业级速度。数据量超过几千条时请换成熟实现（如 sklearn），
本模块适合用来对照结果、做消融实验和解释算法内部机制。

关键约定
--------
- 所有距离都是欧氏距离，特征未做标准化时"距离"会被量纲大的特征支配，
  调用方必须自己先决定是否标准化（见 ``_common.normalize_minmax`` / ``_common.normalize_l2``）。
- 随机性只出现在 K-means++ 初始化，统一走 ``_common.rng``，不接受全局随机状态。
- DBSCAN 的噪声点标签为 ``-1``；簇编号从 0 开始。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ._common import as_matrix, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "kmeans_plusplus_init",
    "kmeans",
    "silhouette_score",
    "elbow_curve",
    "agglomerative",
    "dbscan",
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
           这里显式做"抢走最远点"的修复。
        3. **量纲**：特征未标准化时 SSE 几乎没有解释力，先标准化再聚类。
        4. 收敛判据用的是中心位移而不是 inertia 变化（后者量级依赖数据尺度，阈值不好定）。

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
        2. 从每个未访问的核心点出发 BFS，把密度可达的点归入同一簇
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
        # BFS 扩展当前簇
        labels[i] = cluster_id
        queue = [i]
        while queue:
            cur = queue.pop()
            for nb in neighbors[cur]:
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    if core[nb]:
                        queue.append(int(nb))
        cluster_id += 1
    return labels


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict，键名简短 ASCII，值均为 int/float/list，固定种子下两次调用完全一致：
        ``kmeans_inertia`` 三簇数据的 SSE、``kmeans_silhouette`` 轮廓系数、
        ``kmeans_labels`` 各簇样本数、``elbow_k3`` 肘部曲线在 k=3 的值、
        ``agg_labels`` 层次聚类各簇样本数、``dbscan_labels`` DBSCAN 标签序列、
        ``dbscan_n_noise`` 噪声点数、``silhouette_single`` 单簇时的轮廓系数。

    算法:
        构造三个中心分别在 (-5,-5)、(0,6)、(6,-4)、标准差 0.5 的二维高斯簇，
        每簇 30 个点，固定种子生成；k=3 时应当无歧义地恢复分组。

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
    }
