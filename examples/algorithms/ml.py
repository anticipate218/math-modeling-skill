"""机器学习：数据划分、标准化、混淆矩阵、分类指标、ROC-AUC 与 PR 曲线/AP、kNN、
CART 决策树（Gini/熵）、随机森林、梯度提升、高斯朴素贝叶斯、LDA、PCA、置换重要性、
SMOTE、类别权重平衡。

本模块共 27 个公开函数，按流程分组：数据准备 ``train_test_split`` / ``standardize_fit`` /
``standardize_apply`` / ``class_weight_balanced``，交叉验证划分 ``kfold_indices`` /
``stratified_kfold_indices``，评估 ``confusion_matrix`` / ``classification_metrics`` /
``roc_auc`` / ``pr_curve`` / ``average_precision_score``，模型与解释 ``knn_predict`` /
``decision_tree_fit`` / ``decision_tree_predict`` /
``random_forest_fit`` / ``random_forest_predict`` / ``random_forest_feature_importance`` /
``gradient_boosting_fit`` / ``gradient_boosting_predict`` / ``gaussian_nb_fit`` /
``gaussian_nb_predict`` / ``lda_fit`` / ``lda_transform`` /
``pca_fit`` / ``pca_transform`` / ``permutation_importance`` / ``smote``。

这些实现是**教学透明版**：目标是把论文里需要交代的每一步摊开写清楚（口径、判据、
公式），而不是追求工业级速度或数值鲁棒性。样本量上千以后请换成熟实现
（如 scikit-learn）做交叉验证，用本模块对照结果、解释内部机制。

关键约定
--------
- 特征矩阵一律 ``(n_samples, n_features)``；分类标签用整数编码，回归目标用 float。
- 距离一律欧氏距离；未标准化时距离会被量纲大的特征支配，请先 ``standardize_fit``。
- 类别顺序统一取 ``np.unique`` 的升序；投票**平票时取最小标签/最小类别下标**。
- 决策树是 CART：分类默认用 Gini 不纯度（``decision_tree_fit(criterion="entropy")``
  可换成熵 ``-Σ p log p``），回归用方差（MSE）；候选切分点是排序后相邻
  不同取值的中点，判据为 ``x[:, f] <= threshold`` 走左子树。
- 随机性一律走 ``_common.rng``，绝不使用 ``np.random.*`` 的全局函数。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ._common import as_matrix, as_vector, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "train_test_split",
    "standardize_fit",
    "standardize_apply",
    "confusion_matrix",
    "classification_metrics",
    "roc_auc",
    "pr_curve",
    "average_precision_score",
    "kfold_indices",
    "stratified_kfold_indices",
    "knn_predict",
    "decision_tree_fit",
    "decision_tree_predict",
    "random_forest_fit",
    "random_forest_predict",
    "random_forest_feature_importance",
    "gradient_boosting_fit",
    "gradient_boosting_predict",
    "gaussian_nb_fit",
    "gaussian_nb_predict",
    "lda_fit",
    "lda_transform",
    "pca_fit",
    "pca_transform",
    "permutation_importance",
    "smote",
    "class_weight_balanced",
]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _int_labels(y: ArrayLike, name: str = "y") -> np.ndarray:
    """内部工具：把标签转成一维 int 数组，并校验非空、有限。

    分类标签在返回值里统一转成 int；若传入字符串标签会直接报错，避免静默产生
    无意义的整数编码。
    """
    arr = np.asarray(y)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} 必须是数值标签数组，得到 dtype={arr.dtype}")
    arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr.astype(float))):
        raise ValueError(f"{name} 含 NaN 或 inf")
    return arr.astype(int)


def _check_len(a: np.ndarray, n: int, name: str) -> None:
    """内部工具：校验一维数组长度等于 n，不等时抛 ValueError。"""
    if a.size != n:
        raise ValueError(f"{name} 长度 {a.size} 与样本数 {n} 不一致")


def _check_task(task: str) -> None:
    """内部工具：校验 task 取值合法。"""
    if task not in ("classification", "regression"):
        raise ValueError(f"task 只支持 classification/regression，得到 {task!r}")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """内部工具：平均秩（并列取平均，1-based），用于 AUC 的 Mann-Whitney 公式。

    并列处理必须取平均秩：直接把并列值按出现顺序编号会让 AUC 依赖输入的排列顺序，
    在"大量打平"的评分（例如只有 0/1 两类评分）上会显著偏乐观。
    """
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """内部工具：样本间欧氏距离矩阵，形状 (len(x), len(y))。

    用展开式 ||a-b||^2 = ||a||^2 - 2a·b + ||b||^2 一次算完。``clip`` 只能救回**负**的
    舍入残差：两点重合或极近时相减消去后残差也可能为**正**（量级 ``eps·||a||^2``，
    坐标 ~3 时开方后约 1e-8），直接开方就会得到"自己到自己有 1e-8 距离"的假值，
    污染 kNN 的邻居排序与 SMOTE 的插值方向。因此先把低于容差的元素改用**直接差分**
    ``np.einsum("ij,ij->i", diff, diff)`` 重算：既消掉假距离（同一行/列的对角线精确为
    0），又保证其余分离良好的元素与原来逐位相同。
    """
    a2 = np.sum(x ** 2, axis=1)[:, None]
    b2 = np.sum(y ** 2, axis=1)[None, :]
    sq = a2 - 2.0 * (x @ y.T) + b2
    np.clip(sq, 0.0, None, out=sq)
    scale = max(1.0, float(a2.max()), float(b2.max()))
    tol = 1e-9 * scale
    rows, cols = np.nonzero(sq < tol)
    if rows.size:
        diff = x[rows] - y[cols]
        sq[rows, cols] = np.einsum("ij,ij->i", diff, diff)
    return np.sqrt(sq)


def _impurity(y: np.ndarray, task: str, criterion: str = "gini") -> float:
    """内部工具：节点不纯度——分类按 ``criterion`` 取 Gini 或熵，回归为方差（总体口径 MSE）。

    分类口径：``gini = 1 - Σ p_k²``、``entropy = -Σ p_k log p_k``（自然对数）。
    熵里对计数为 0 的类别取 0·log0 = 0（把 ``log`` 的参数置 1 来消除 NaN/警告）。
    ``criterion`` 只对分类有意义，回归一律用方差。
    """
    if y.size == 0:
        return 0.0
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        p = counts.astype(float) / float(y.size)
        if criterion == "entropy":
            safe = np.where(p > 0.0, p, 1.0)
            return float(-np.sum(p * np.log(safe)))
        return float(1.0 - np.sum(p ** 2))
    return float(np.mean((y - y.mean()) ** 2))


def _node_value(y: np.ndarray, task: str):
    """内部工具：节点预测值——分类为多数类（平票取最小标签），回归为均值。"""
    if task == "classification":
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[int(np.argmax(counts))])
    return float(y.mean())


def _feature_subset(n_features: int, max_features, gen: np.random.Generator) -> np.ndarray:
    """内部工具：随机森林的每次分裂候选特征子集，返回**升序**下标数组。

    升序是为了让"不纯度并列时取第一个特征"这一平局规则可复现。
    """
    if max_features is None:
        return np.arange(n_features)
    if isinstance(max_features, str):
        key = max_features.lower()
        if key == "sqrt":
            m = int(np.sqrt(n_features))
        elif key == "log2":
            m = int(np.log2(n_features))
        else:
            raise ValueError(f"max_features 只支持 'sqrt'/'log2'/int/float，得到 {max_features!r}")
    elif isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError(f"max_features 为浮点数时必须在 (0, 1]，得到 {max_features}")
        m = int(np.ceil(max_features * n_features))
    else:
        m = int(max_features)
    m = max(1, min(m, n_features))
    return np.sort(gen.choice(n_features, size=m, replace=False))


def _best_split(
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    classes: np.ndarray,
    feature_idx: np.ndarray,
    min_samples_leaf: int,
    criterion: str = "gini",
):
    """内部工具：在给定特征子集上找加权不纯度最小的切分，返回 (特征, 阈值, 左样本数)。

    对每个特征先排序，再用累积和一次算出所有候选切分点的左右不纯度，
    避免对每个候选点重扫一遍数据。分类的节点不纯度按 ``criterion`` 取 Gini 或熵
    （见 ``_impurity``），回归一律用方差；``criterion`` 有默认值，老的调用点不必改。
    """
    n = x.shape[0]
    best = None
    best_score = np.inf
    if task == "classification":
        k = classes.size
        codes = np.searchsorted(classes, y)
        onehot = np.zeros((n, k), dtype=float)
        onehot[np.arange(n), codes] = 1.0
    else:
        onehot = None

    for f in feature_idx:
        col = x[:, f]
        order = np.argsort(col, kind="mergesort")
        vals = col[order]
        distinct = np.flatnonzero(np.diff(vals) > 0)
        if distinct.size == 0:
            continue
        pos = (distinct + 1).astype(int)
        pos = pos[(pos >= min_samples_leaf) & ((n - pos) >= min_samples_leaf)]
        if pos.size == 0:
            continue
        nl = pos.astype(float)
        nr = n - nl
        if task == "classification":
            cum = np.cumsum(onehot[order], axis=0)
            left_c = cum[pos - 1]
            right_c = cum[-1] - left_c
            pl = left_c / nl[:, None]
            pr = right_c / nr[:, None]
            if criterion == "entropy":
                safe_l = np.where(pl > 0.0, pl, 1.0)
                safe_r = np.where(pr > 0.0, pr, 1.0)
                gl = -np.sum(pl * np.log(safe_l), axis=1)
                gr = -np.sum(pr * np.log(safe_r), axis=1)
            else:
                gl = 1.0 - np.sum(pl ** 2, axis=1)
                gr = 1.0 - np.sum(pr ** 2, axis=1)
            scores = (nl * gl + nr * gr) / float(n)
        else:
            ys = y[order]
            cy = np.cumsum(ys)
            cy2 = np.cumsum(ys ** 2)
            sl, sl2 = cy[pos - 1], cy2[pos - 1]
            sr, sr2 = cy[-1] - sl, cy2[-1] - sl2
            vl = np.clip(sl2 / nl - (sl / nl) ** 2, 0.0, None)
            vr = np.clip(sr2 / nr - (sr / nr) ** 2, 0.0, None)
            scores = (nl * vl + nr * vr) / float(n)
        j = int(np.argmin(scores))
        if float(scores[j]) < best_score - 1e-15:
            best_score = float(scores[j])
            left_n = int(pos[j])
            thr = float(0.5 * (vals[left_n - 1] + vals[left_n]))
            best = (int(f), thr, left_n)
    return best


def _build_tree(
    x: np.ndarray,
    y: np.ndarray,
    depth: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    task: str,
    classes: np.ndarray,
    gen: Optional[np.random.Generator],
    max_features,
    criterion: str = "gini",
) -> dict:
    """内部工具：CART 递归建树，返回嵌套 dict。

    内部节点键为 ``feature/threshold/left/right/value/n_samples/impurity``；
    叶子只含 ``value/n_samples/impurity``（不含 feature/threshold）。
    分类的不纯度按 ``criterion``（``"gini"`` 或 ``"entropy"``）计算并向上传播；
    ``criterion`` 有默认值，随机森林/梯度提升等既有调用点不必改。
    """
    n = x.shape[0]
    node = {
        "value": _node_value(y, task),
        "n_samples": int(n),
        "impurity": _impurity(y, task, criterion),
    }
    if depth >= max_depth or n < min_samples_split or node["impurity"] <= 0.0:
        return node

    if gen is None:
        feature_idx = np.arange(x.shape[1])
    else:
        feature_idx = _feature_subset(x.shape[1], max_features, gen)

    split = _best_split(x, y, task, classes, feature_idx, min_samples_leaf, criterion)
    if split is None:
        return node
    f, thr, _left_n = split
    left_mask = x[:, f] <= thr
    if not np.any(left_mask) or np.all(left_mask):
        return node
    # 切分必须真正降低不纯度，否则停在这里（避免生成无意义的空转节点）。
    left_y = y[left_mask]
    right_y = y[~left_mask]
    weighted = (
        left_y.size * _impurity(left_y, task, criterion)
        + right_y.size * _impurity(right_y, task, criterion)
    ) / float(n)
    if weighted >= node["impurity"] - 1e-12:
        return node

    node["feature"] = int(f)
    node["threshold"] = float(thr)
    node["left"] = _build_tree(
        x[left_mask], left_y, depth + 1, max_depth, min_samples_split,
        min_samples_leaf, task, classes, gen, max_features, criterion,
    )
    node["right"] = _build_tree(
        x[~left_mask], right_y, depth + 1, max_depth, min_samples_split,
        min_samples_leaf, task, classes, gen, max_features, criterion,
    )
    return node


def _tree_predict_one(tree: dict, row: np.ndarray):
    """内部工具：单样本沿树下行到叶子，返回叶子预测值。"""
    node = tree
    while "feature" in node:
        node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
    return node["value"]


def _tree_raw_importance(trees: List[dict], n_features: int) -> np.ndarray:
    """内部工具：把森林里每棵树的加权不纯度下降按特征累加（未归一化）。"""
    imp = np.zeros(n_features, dtype=float)
    for tree in trees:
        stack = [tree]
        while stack:
            node = stack.pop()
            if "feature" not in node:
                continue
            left, right = node["left"], node["right"]
            decrease = (
                node["n_samples"] * node["impurity"]
                - left["n_samples"] * left["impurity"]
                - right["n_samples"] * right["impurity"]
            )
            imp[node["feature"]] += max(float(decrease), 0.0)
            stack.append(left)
            stack.append(right)
    return imp


def _normalize_importance(imp: np.ndarray) -> np.ndarray:
    """内部工具：把重要性向量归一化到和为 1；全 0 时退化为均匀分布。"""
    total = float(np.sum(imp))
    if total <= 0.0:
        n = imp.size
        return np.full(n, 1.0 / n) if n else imp
    return imp / total


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """内部工具：数值稳定的 logistic 函数。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# 数据划分与预处理
# ---------------------------------------------------------------------------


def train_test_split(
    X: MatrixLike,
    y: ArrayLike,
    test_size: float = 0.2,
    seed: Optional[int] = None,
    stratify: Optional[ArrayLike] = None,
) -> dict:
    """把样本随机切成训练集与测试集，可选按标签分层。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 目标数组，长度 n_samples；分类标签或回归目标都可以。
        test_size: 测试集比例或条数。float 必须在 (0, 1) 内（按 ``ceil(test_size*n)``
            取条数）；int 则直接当作测试集条数，必须在 (0, n) 内。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        stratify: 若给出，则是长度 n_samples 的标签数组，按类分层抽样，
            使测试集中各类比例接近总体比例。

    返回:
        dict，键为：
        ``X_train`` 形状 (n_train, n_features)；
        ``y_train`` 形状 (n_train,)；
        ``X_test``  形状 (n_test, n_features)；
        ``y_test``  形状 (n_test,)。

    算法:
        1. 校验后把样本总数记作 n，算出测试集条数 n_test；
        2. 不分层时对 ``0..n-1`` 整体做一次 ``permutation``，前 n_test 个作测试集；
        3. 分层时对每个类别单独做 permutation，取 ``round(frac*m_c)`` 个作该类的测试样本
           （m_c >= 2 时强制至少 1 个、且至少留 1 个给训练集；m_c == 1 时全部留训练集），
           再把各类的测试下标合并后整体打乱，避免输出按类别排好序；
        4. 按下标切片返回。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. 分层抽样下各类的测试条数分别取整，**总数可能与不分层的 n_test 差 ±K**，
           论文里报"测试集占比"时应以实际返回的条数为准。
        2. 某个类别只有 1 个样本时本实现把它留在训练集；若强行分到测试集，
           训练集将完全没有这个类，后续 ``confusion_matrix`` 会缺标签。
        3. 返回的是**数组副本**（花式索引），改 ``X_train`` 不会影响原 ``X``；
           但若 X 本身是 ndarray，``as_matrix`` 不复制，所以别依赖"原数据不被改"。

    参考:
        无（通用做法；分层抽样见 Kohavi et al., "A study of cross-validation and
        bootstrap for accuracy estimation", IJCAI 1995）。
    """
    x = as_matrix(X, "X")
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"样本数必须 >= 2，得到 {n}")

    y_arr = np.asarray(y)
    if y_arr.dtype.kind not in "biuf":
        raise ValueError(f"y 必须是数值数组，得到 dtype={y_arr.dtype}")
    y_arr = y_arr.ravel()
    _check_len(y_arr, n, "y")
    if not np.all(np.isfinite(y_arr.astype(float))):
        raise ValueError("y 含 NaN 或 inf")

    if isinstance(test_size, bool) or not isinstance(test_size, (int, float, np.integer, np.floating)):
        raise ValueError(f"test_size 必须是 float 或 int，得到 {test_size!r}")
    if isinstance(test_size, (int, np.integer)):
        n_test = int(test_size)
        if not (0 < n_test < n):
            raise ValueError(f"test_size 为 int 时必须在 (0, {n}) 内，得到 {n_test}")
    else:
        if not (0.0 < float(test_size) < 1.0):
            raise ValueError(f"test_size 为 float 时必须在 (0, 1) 内，得到 {test_size}")
        n_test = int(np.ceil(float(test_size) * n))
    n_test = max(1, min(n_test, n - 1))
    frac = n_test / float(n)

    gen = rng(seed)
    if stratify is None:
        perm = gen.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
    else:
        strat = _int_labels(stratify, "stratify")
        _check_len(strat, n, "stratify")
        classes, counts = np.unique(strat, return_counts=True)
        if classes.size > n // 2:
            raise ValueError(
                f"stratify 的类别数 {classes.size} 过多（样本数 {n}），无法保证每类都进训练集"
            )
        test_parts: List[np.ndarray] = []
        train_parts: List[np.ndarray] = []
        for c, m in zip(classes, counts):
            members = np.flatnonzero(strat == c)
            perm_c = members[gen.permutation(int(m))]
            if m >= 2:
                n_c_test = int(np.floor(frac * int(m) + 0.5))
                n_c_test = min(max(n_c_test, 1), int(m) - 1)
            else:
                n_c_test = 0
            test_parts.append(perm_c[:n_c_test])
            train_parts.append(perm_c[n_c_test:])
        test_idx = np.concatenate(test_parts) if test_parts else np.zeros(0, dtype=int)
        train_idx = np.concatenate(train_parts) if train_parts else np.zeros(0, dtype=int)
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError("分层后训练集或测试集为空，请减小 test_size 或检查标签分布")
        test_idx = test_idx[gen.permutation(test_idx.size)]
        train_idx = train_idx[gen.permutation(train_idx.size)]

    return {
        "X_train": x[train_idx],
        "y_train": y_arr[train_idx],
        "X_test": x[test_idx],
        "y_test": y_arr[test_idx],
    }


def standardize_fit(X: MatrixLike) -> dict:
    """计算 Z-score 标准化的列均值与列标准差（总体口径，ddof=0）。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。

    返回:
        dict，键为：
        ``mean`` 形状 (n_features,) 的列均值；
        ``std``  形状 (n_features,) 的列标准差，**标准差为 0 的列置 1**。

    算法:
        1. ``mean = X.mean(axis=0)``；
        2. ``std = X.std(axis=0)``（ddof=0，即除以 n 而不是 n-1）；
        3. 把 ``std == 0`` 的位置替换为 1，使常数列标准化后变成全 0 而不是 NaN。

    复杂度:
        时间 O(n d) / 空间 O(d)。

    陷阱:
        1. 用 ddof=1 的样本标准差会让"标准化后整列标准差恰好为 1"不成立
           （差一个 sqrt(n/(n-1)) 的因子），跟 numpy 默认口径对不上。
        2. 常数列被置 1 后该列标准化结果是**全 0**：这列没有信息，聚类时它不贡献距离，
           但树模型仍然可以拿它做分裂（永远没有增益）。
        3. 本函数不看训练集之外的任何数据，所以"先对全体数据 fit 再划分训练/测试"
           就是典型的数据泄漏，必须先划分再 fit。

    参考:
        标准 Z-score 归一化；泄漏问题见 Kaufman et al., "Leakage in Data Mining", 2012。
    """
    x = as_matrix(X, "X")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return {"mean": mean, "std": std}


def standardize_apply(X: MatrixLike, mean: ArrayLike, std: ArrayLike) -> np.ndarray:
    """用给定的均值/标准差对数据做 Z-score 标准化。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        mean: 形状 (n_features,) 的列均值（通常来自 ``standardize_fit``）。
        std: 形状 (n_features,) 的列标准差；不允许含 0。

    返回:
        np.ndarray，形状 (n_samples, n_features)，``(X - mean) / std``。

    算法:
        逐列做 ``(x - mean) / std``，形状靠 numpy 广播对齐。

    复杂度:
        时间 O(n d) / 空间 O(n d)。

    陷阱:
        1. std 里出现 0 时本函数**直接抛 ValueError**而不是悄悄返回 inf/NaN：
           ``standardize_fit`` 的输出永远不含 0，出现 0 说明调用方自己构造了 std。
        2. 若 mean/std 来自与 X 不同的数据（例如用了全体数据的统计量），
           测试集的"标准化"就带入了训练集看不到的信息，评估会偏乐观。

    参考:
        无。
    """
    x = as_matrix(X, "X")
    mu = as_vector(mean, "mean")
    sd = as_vector(std, "std")
    if mu.size != x.shape[1]:
        raise ValueError(f"mean 长度 {mu.size} 与特征数 {x.shape[1]} 不一致")
    if sd.size != x.shape[1]:
        raise ValueError(f"std 长度 {sd.size} 与特征数 {x.shape[1]} 不一致")
    zero = np.flatnonzero(sd == 0.0)
    if zero.size:
        raise ValueError(f"std 在第 {zero[0]} 列为 0，无法标准化")
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# 分类评估
# ---------------------------------------------------------------------------


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: Optional[ArrayLike] = None
) -> dict:
    """混淆矩阵：``matrix[i, j]`` 是真实标签为 labels[i]、预测为 labels[j] 的样本数。

    参数:
        y_true: 真实标签，长度 n_samples。
        y_pred: 预测标签，长度 n_samples。
        labels: 类别顺序；None 表示取真实与预测标签并集的升序。

    返回:
        dict，键为：
        ``matrix`` K×K 的**纯 Python int 列表的列表**（可直接 json 序列化）；
        ``labels`` 长度 K 的 int 列表，行列的含义都由它定义。

    算法:
        1. 确定类别列表 labels（升序去重）；
        2. 把标签映射成 0..K-1 的下标；
        3. 用 ``np.bincount`` 在 ``true*K + pred`` 上一次性计数，再 reshape 成 K×K。

    复杂度:
        时间 O(n + K^2) / 空间 O(K^2)。

    陷阱:
        1. 当 ``labels`` 显式给出时，出现在数据里但不在 labels 中的样本会被**静默丢弃**，
           这在多折交叉验证汇总时会让总数对不上，务必检查 ``matrix.sum() == n``。
        2. 返回的是 Python list 而不是 numpy 数组，方便写进 JSON 报告；
           需要矩阵运算时请自己 ``np.asarray``。
        3. 行是真实、列是预测——写反了 accuracy 不变但 precision/recall 会互换。

    参考:
        混淆矩阵的标准定义（见任何机器学习教材，如 Bishop, PRML, 2006, §1.5）。
    """
    yt = _int_labels(y_true, "y_true")
    yp = _int_labels(y_pred, "y_pred")
    # 真正的空输入校验：原先的 ``_check_len(yt, yt.size, "y_true")`` 是自比自的空操作
    # （比较 a.size != n，而 n 就是 a.size），永远通过，等于没有校验。
    if yt.size == 0:
        raise ValueError("y_true 不能为空")
    if yp.size != yt.size:
        raise ValueError(f"y_pred 长度 {yp.size} 与 y_true 长度 {yt.size} 不一致")

    if labels is None:
        lab = np.unique(np.concatenate([yt, yp]))
    else:
        lab = _int_labels(labels, "labels")
    if lab.size == 0:
        raise ValueError("labels 不能为空")
    k = lab.size
    if np.unique(lab).size != k:
        raise ValueError("labels 中存在重复类别")

    ti = np.searchsorted(lab, yt)
    pi = np.searchsorted(lab, yp)
    valid = (ti < k) & (pi < k) & (lab[np.clip(ti, 0, k - 1)] == yt) & (lab[np.clip(pi, 0, k - 1)] == yp)
    counts = np.bincount(ti[valid] * k + pi[valid], minlength=k * k)
    matrix = counts.reshape(k, k)
    return {"matrix": [[int(v) for v in row] for row in matrix], "labels": [int(v) for v in lab]}


def classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict:
    """分类指标：准确率与宏平均的精确率/召回率/F1。

    参数:
        y_true: 真实标签，长度 n_samples。
        y_pred: 预测标签，长度 n_samples。

    返回:
        dict，键为：
        ``accuracy``       float，``(y_true == y_pred).mean()``；
        ``precision_macro`` float，各类精确率的算术平均（宏平均，类不加权）；
        ``recall_macro``   float，各类召回率的算术平均；
        ``f1_macro``       float，各类 F1 的算术平均；
        ``n_classes``      int，参与平均的类别数（真实与预测标签的并集）。

    算法:
        1. 以真实与预测标签的并集为类别集合；
        2. 对每个类 c：TP = 预测 c 且真实 c，FP = 预测 c，FN = 真实 c，
           ``precision = TP / FP``、``recall = TP / FN``、``f1 = 2PR/(P+R)``，
           分母为 0 时该项记 0；
        3. 对类别取算术平均得到宏平均指标。

    复杂度:
        时间 O(n K) / 空间 O(K)。

    陷阱:
        1. 宏平均**不给类别加权**：在 1:100 的不平衡数据上，少数类的坏表现会被放大，
           数值通常明显低于 accuracy；论文里必须写清用的是宏平均还是微平均。
        2. 只在预测里出现、真实里没有的类别（FP 类）也会被计入平均，这会拉低宏平均，
           与"只按真实类别平均"的另一种口径不同。
        3. 多分类下 F1 也可以按"先累加 TP/FP/FN 再算"的微平均得到，本函数**不做**微平均。

    参考:
        Sokolova & Lapalme, "A systematic analysis of performance measures for
        classification tasks", Information Processing & Management, 2009。
    """
    yt = _int_labels(y_true, "y_true")
    yp = _int_labels(y_pred, "y_pred")
    if yp.size != yt.size:
        raise ValueError(f"y_pred 长度 {yp.size} 与 y_true 长度 {yt.size} 不一致")
    if yt.size == 0:
        raise ValueError("y_true 不能为空")

    classes = np.unique(np.concatenate([yt, yp]))
    precision = np.zeros(classes.size, dtype=float)
    recall = np.zeros(classes.size, dtype=float)
    f1 = np.zeros(classes.size, dtype=float)
    for i, c in enumerate(classes):
        pred_c = yp == c
        true_c = yt == c
        tp = float(np.count_nonzero(pred_c & true_c))
        fp = float(np.count_nonzero(pred_c))
        fn = float(np.count_nonzero(true_c))
        p = tp / fp if fp > 0 else 0.0
        r = tp / fn if fn > 0 else 0.0
        precision[i] = p
        recall[i] = r
        f1[i] = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "accuracy": float(np.mean(yt == yp)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "n_classes": int(classes.size),
    }


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """二分类 AUC，用 Mann-Whitney U（秩和）公式计算，并列分数取平均秩。

    参数:
        y_true: 二分类真实标签，长度 n_samples，必须恰好含 2 个不同取值。
        scores: 连续打分（越大越倾向正类），长度 n_samples。

    返回:
        float，AUC = P(随机正样本得分 > 随机负样本得分) + 0.5·P(相等)。

    算法:
        1. 类别升序排列后，**较大标签记为正类**（这是个需要写进论文的口径）；
        2. 对 scores 求平均秩 rank_i（并列取平均）；
        3. ``AUC = (sum_{正类} rank_i - n_pos(n_pos+1)/2) / (n_pos * n_neg)``。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. 正类是"较大的那个标签"：若标签是 {1, 2}，2 是正类；若标签是 {-1, 1}，
           1 是正类。传反了得到的是 1 - AUC。
        2. 并列秩必须取平均，否则"全部同分"会得到依赖输入顺序的荒谬值；
           正确结果是全并列时 AUC = 0.5。
        3. 只有一个类别的输入由上面的二分类检查（``classes.size != 2``）直接拒绝：
           该检查已保证正、负类各至少 1 个样本，所以这里**不存在**"分母为 0"的分支。
           也不要把它改成返回 0.5，0.5 会被误读成"模型无区分度"。
        4. 对多分类问题本函数不适用（不做 one-vs-rest 展开）。

    参考:
        Mann & Whitney, "On a test of whether one of two random variables is
        stochastically larger than the other", Annals of Mathematical Statistics, 1947。
    """
    yt = _int_labels(y_true, "y_true")
    sc = as_vector(scores, "scores")
    if sc.size != yt.size:
        raise ValueError(f"scores 长度 {sc.size} 与 y_true 长度 {yt.size} 不一致")
    classes = np.unique(yt)
    if classes.size != 2:
        raise ValueError(f"roc_auc 只支持二分类，y_true 含 {classes.size} 个类别：{classes.tolist()}")

    pos = yt == classes[1]
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(yt.size - n_pos)
    ranks = _rankdata(sc)
    auc = (float(np.sum(ranks[pos])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def pr_curve(y_true: ArrayLike, scores: ArrayLike) -> dict:
    """二分类的精确率-召回率曲线（PR 曲线），阈值取分数的所有**不同取值**。

    参数:
        y_true: 二分类真实标签，长度 n_samples，必须恰好含 2 个不同取值。
            与 ``roc_auc`` 一致：类别升序排列后**较大的标签记为正类**。
        scores: 连续打分（越大越倾向正类），长度 n_samples。

    返回:
        dict，键为：
        ``precision`` 长度 T+1 的 float 数组，``precision[i]`` 是"打分 >= thresholds[i]"
        时的精确率 ``TP/(TP+FP)``（分母为 0 记 0）；最后一维恒为 1.0，它**没有对应阈值**；
        ``recall`` 长度 T+1 的 float 数组，``recall[i] = TP/TP_total``；最后一维恒为 0.0；
        ``thresholds`` 长度 T 的 float 数组，**升序**，恰好是 scores 的全部不同取值
        （``T = len(np.unique(scores))``）。
        于是 ``len(precision) == len(recall) == len(thresholds) + 1``。

    算法:
        1. 按分数从大到小做稳定排序；
        2. 只在"分数发生变化"的地方切阈值组——**并列分数被视为同一切点**，
           同一分数不会被拆成多个中间点；
        3. 用累积和求每个阈值处的 TP/FP：``precision = TP/(TP+FP)``、``recall = TP/TP_total``；
        4. 把数组翻转成"阈值升序、recall 递减"，再在末尾补上无阈值的约定点
           ``(precision, recall) = (1.0, 0.0)``。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. 三个数组长度不同：precision/recall 比 thresholds **长一个**，那个
           precision=1、recall=0 的点没有阈值。画图或做数值积分时必须把它和前面的点
           分开处理，否则会把阈值数组和指标数组错位对齐。
        2. 并列分数必须合并成同一切点：若按"逐个样本"推进，2 个并列的正样本会在
           recall 不变的两个点上给出**人为的** precision 差异，整条阶梯因此改变。
           本实现保证同一分数只产生一个点，这也正是与 sklearn 对拍时最容易踩的差异。
        3. ``recall`` 是**递减**的：第 0 个点对应最低阈值（把所有样本都判为正，
           故 recall=1、precision=类别比例）。这与 ROC 曲线常用的"阈值从高到低"
           叙述方向相反，论文里要写清横纵轴方向。
        4. 正类口径与 ``roc_auc`` 一致（较大的标签）；把标签反号会得到另一条曲线。
        5. 只有一个类别、或两数组长度不一致时抛 ValueError；本函数不支持多分类
           （不做 one-vs-rest 展开）。

    参考:
        sklearn.metrics.precision_recall_curve 的口径；
        Davis & Goadrich, "The relationship between Precision-Recall and ROC curves",
        ICML 2006。
    """
    yt = _int_labels(y_true, "y_true")
    sc = as_vector(scores, "scores")
    if sc.size != yt.size:
        raise ValueError(f"scores 长度 {sc.size} 与 y_true 长度 {yt.size} 不一致")
    classes = np.unique(yt)
    if classes.size != 2:
        raise ValueError(
            f"pr_curve 只支持二分类，y_true 含 {classes.size} 个类别：{classes.tolist()}"
        )

    pos = (yt == classes[1]).astype(float)
    # 稳定降序，与 sklearn 的 argsort(..., kind="mergesort")[::-1] 同序。
    order = np.argsort(sc, kind="mergesort")[::-1]
    y_sorted = pos[order]
    s_sorted = sc[order]
    # 只在分数变化处切阈值组：并列分数被合并为同一切点。
    change = np.flatnonzero(np.diff(s_sorted) != 0.0)
    idx = np.concatenate([change, np.array([s_sorted.size - 1], dtype=int)])
    tps = np.cumsum(y_sorted)[idx]
    fps = np.cumsum(1.0 - y_sorted)[idx]
    thresholds = s_sorted[idx]
    total = tps + fps
    precision = np.divide(tps, total, out=np.zeros_like(tps), where=total > 0.0)
    recall = tps / tps[-1]
    return {
        "precision": np.concatenate([precision[::-1], np.array([1.0])]),
        "recall": np.concatenate([recall[::-1], np.array([0.0])]),
        "thresholds": thresholds[::-1].copy(),
    }


def average_precision_score(y_true: ArrayLike, scores: ArrayLike) -> float:
    """平均精确率 AP：PR 曲线的**阶梯式**积分 ``Σ_n (R_n - R_{n-1}) · P_n``（不插值）。

    参数:
        y_true: 二分类真实标签，长度 n_samples，必须恰好含 2 个不同取值
            （类别升序后较大的标签记为正类，与 ``pr_curve``/``roc_auc`` 一致）。
        scores: 连续打分（越大越倾向正类），长度 n_samples。

    返回:
        float，取值 [0, 1] 的平均精确率；数值上与
        ``sklearn.metrics.average_precision_score``（同样正类口径、不加权）一致到 1e-12。

    算法:
        1. 调 ``pr_curve`` 得到阈值升序的 precision/recall（末尾多一个无阈值点）；
        2. AP 即该阶梯函数的积分：``AP = -Σ_n Δrecall_n · precision_n``，
           其中 ``Δrecall_n = recall[n+1] - recall[n] <= 0``，等价于
           ``Σ_n (R_n - R_{n-1}) · P_n``（recall 递减，所以符号取反）。
        3. 结尾截断到 ``max(0.0, ·)`` 以消除浮点造成的 -0.0。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. **不做线性插值**：这里用的是阶梯积分（与 sklearn 的
           ``average_precision_score`` 一致），不是 PR 曲线下的梯形面积；
           插值口径会系统性偏高，两者不可混用。
        2. 必须复用同一个并列合并口径：如果 PR 曲线的并列分数被拆成多个点，
           会让部分点上的 ``Δrecall = 0``，虽然对 AP 的贡献为 0，但一旦实现里
           把并列点的 precision 算错，积分结果就会偏移。
        3. 完全随机打分下 AP 约等于正类比例（不是 0.5，也不是 1.0），
           在不平衡数据上这个基线非常低，看起来"很差"其实是随机的正常水平。
        4. 输入非法（只有一个类别、长度不一致、非数值标签）时由 ``pr_curve``
           抛 ValueError，本函数不额外吞掉异常。

    参考:
        sklearn.metrics.average_precision_score；
        Wikipedia, "Evaluation measures (information retrieval) — Average precision"。
    """
    curve = pr_curve(y_true, scores)
    precision = np.asarray(curve["precision"], dtype=float)
    recall = np.asarray(curve["recall"], dtype=float)
    # precision 的最后一维恒为 1，所以直接对"前 len-1 个点"做阶梯积分即可。
    ap = -float(np.sum(np.diff(recall) * precision[:-1]))
    return float(max(0.0, ap))


# ---------------------------------------------------------------------------
# 交叉验证划分
# ---------------------------------------------------------------------------


def kfold_indices(n_samples: int, k: int = 5, seed: Optional[int] = None) -> List[List[int]]:
    """K 折交叉验证的测试集下标（随机打乱后等分）。

    参数:
        n_samples: 样本总数，必须 >= 2。
        k: 折数，1 <= k <= n_samples。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        list，长度 k；第 i 项是第 i 折**测试集**的样本下标（纯 Python int 列表）。
        训练集下标即其余各折的并集。

    算法:
        1. 对 ``0..n-1`` 做一次 ``permutation``；
        2. ``base = n // k``、``rem = n % k``：前 rem 折每折 base+1 个样本，其余每折 base 个；
        3. 逐折切出连续片段作为该折测试集。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        1. n 不能被 k 整除时各折大小相差 1，**不能假设所有折等大**；
           最后一折可能只有 `base` 个（当 rem > 0 时前 rem 折更大），写死索引会错位。
        2. 折内样本在时间序列数据上是"未来预测过去"，会造成信息泄漏；
           时间序列必须用前向滚动划分而不是本函数。
        3. 返回的是 list of list（不是 numpy 数组），方便直接 json 打印。

    参考:
        交叉验证的通用做法，见 Stone, "Cross-validatory choice and assessment of
        statistical predictions", JRSS-B, 1974。
    """
    n = int(n_samples)
    if n < 2:
        raise ValueError(f"n_samples 必须 >= 2，得到 {n}")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"k 必须是整数，得到 {k!r}")
    k = int(k)
    if not (1 <= k <= n):
        raise ValueError(f"k 必须在 [1, {n}] 内，得到 {k}")

    gen = rng(seed)
    perm = gen.permutation(n)
    base = n // k
    rem = n % k
    folds: List[List[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        folds.append([int(v) for v in perm[start:start + size]])
        start += size
    return folds


def stratified_kfold_indices(y: ArrayLike, k: int = 5, seed: Optional[int] = None) -> List[List[int]]:
    """分层 K 折交叉验证的测试集下标：每折的类别比例近似等于总体比例。

    参数:
        y: 类别标签，长度 n_samples。
        k: 折数，1 <= k <= n_samples。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        list，长度 k；第 i 项是第 i 折测试集的样本下标（升序的纯 Python int 列表）。

    算法:
        1. 对每个类别，把该类的样本下标独立打乱；
        2. 按类内顺序把该类样本轮流分给 k 折（第 j 个样本分到第 j % k 折）；
        3. 合并各类在同一折里的下标并升序排序。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        1. 类内样本数 m_c 不能被 k 整除时，**不同折拿到该类样本数会差 1**，
           比例只能"近似"一致；自测里必须用容差而不是精确相等。
        2. 当某个类的样本数少于 k 时，必然有折在该类上为空——此时分层退化，
           论文里应说明或改用 ``k <= min_c n_c``。
        3. 类别极不平衡时（例如 1:1000），分层会让每折都含少数类，这是好事，
           但每折的少数类样本可能只有 1 个，指标方差仍然很大。

    参考:
        分层抽样与交叉验证的结合，见 Kohavi, "A study of cross-validation and
        bootstrap for accuracy estimation", IJCAI 1995。
    """
    ylab = _int_labels(y, "y")
    n = ylab.size
    if n < 2:
        raise ValueError(f"样本数必须 >= 2，得到 {n}")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"k 必须是整数，得到 {k!r}")
    k = int(k)
    if not (1 <= k <= n):
        raise ValueError(f"k 必须在 [1, {n}] 内，得到 {k}")

    gen = rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    for c in np.unique(ylab):
        members = np.flatnonzero(ylab == c)
        shuffled = members[gen.permutation(members.size)]
        for pos, idx in enumerate(shuffled):
            folds[pos % k].append(int(idx))
    return [sorted(f) for f in folds]


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------


def knn_predict(
    X_train: MatrixLike,
    y_train: ArrayLike,
    X_test: MatrixLike,
    k: int = 3,
    task: str = "classification",
) -> dict:
    """k 近邻预测：分类用多数投票（平票取最小标签），回归用邻居均值。

    参数:
        X_train: 训练样本矩阵，形状 (n_train, n_features)。
        y_train: 训练标签/目标，长度 n_train。
        X_test: 待预测样本矩阵，形状 (n_test, n_features)。
        k: 邻居个数，1 <= k <= n_train。
        task: ``"classification"`` 或 ``"regression"``。

    返回:
        dict，键为：
        ``pred``     形状 (n_test,)，分类为 int 标签、回归为 float；
        ``distance`` 形状 (n_test,)，**被选中的 k 个邻居到该点的平均欧氏距离**
                     （不是最近邻距离，也不是距离倒数权重）。

    算法:
        1. 用展开式一次算出 (n_test, n_train) 欧氏距离矩阵；
        2. 每个测试点取距离最小的 k 个训练样本（``argsort(kind="mergesort")``
           保证并列时取下标较小的，结果可复现）；
        3. 分类：对邻居标签做多数投票，票数并列时取最小标签；
           回归：取邻居目标的算术平均。

    复杂度:
        时间 O(n_test · n_train · d + n_test · n_train log n_train) / 空间 O(n_test · n_train)。

    陷阱:
        1. **未标准化时 kNN 基本失效**：量纲大的特征会独占距离，先 ``standardize_fit``。
        2. 平票规则必须显式固定（这里取最小标签），否则二分类偶数 k 时结果依赖排序。
        3. ``distance`` 是邻居的**平均**距离，常被误当成"预测置信度"：
           它大只说明测试点离训练数据远，不说明预测一定错。
        4. k 取 n_train 时全部样本参与，分类退化为多数类、回归退化为全局均值。

    参考:
        Cover & Hart, "Nearest neighbor pattern classification", IEEE Trans. IT, 1967。
    """
    xtr = as_matrix(X_train, "X_train")
    xte = as_matrix(X_test, "X_test")
    _check_task(task)
    n_train = xtr.shape[0]
    if xtr.shape[1] != xte.shape[1]:
        raise ValueError(f"训练/测试特征数不一致：{xtr.shape[1]} vs {xte.shape[1]}")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"k 必须是整数，得到 {k!r}")
    k = int(k)
    if not (1 <= k <= n_train):
        raise ValueError(f"k 必须在 [1, {n_train}] 内，得到 {k}")

    if task == "classification":
        ytr = _int_labels(y_train, "y_train")
        _check_len(ytr, n_train, "y_train")
    else:
        ytr = as_vector(y_train, "y_train")
        _check_len(ytr, n_train, "y_train")

    dist = _pairwise_sq_dist(xte, xtr)
    n_test = dist.shape[0]
    mean_dist = np.empty(n_test, dtype=float)
    if task == "classification":
        pred = np.empty(n_test, dtype=int)
    else:
        pred = np.empty(n_test, dtype=float)

    for i in range(n_test):
        idx = np.argsort(dist[i], kind="mergesort")[:k]
        mean_dist[i] = float(np.mean(dist[i, idx]))
        if task == "classification":
            vals, counts = np.unique(ytr[idx], return_counts=True)
            pred[i] = int(vals[int(np.argmax(counts))])
        else:
            pred[i] = float(np.mean(ytr[idx]))
    return {"pred": pred, "distance": mean_dist}


# ---------------------------------------------------------------------------
# CART 决策树
# ---------------------------------------------------------------------------


def decision_tree_fit(
    X: MatrixLike,
    y: ArrayLike,
    max_depth: int = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    task: str = "classification",
    criterion: str = "gini",
) -> dict:
    """CART 决策树：自顶向下贪心选择最小化加权不纯度的二分切分。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 标签/目标，长度 n_samples。
        max_depth: 最大深度，>= 0（0 表示只有根节点，即常数预测）。
        min_samples_split: 节点样本数小于它就不再分裂，>= 2。
        min_samples_leaf: 分裂后每侧至少保留的样本数，>= 1。
        task: ``"classification"``（按 ``criterion`` 算分类不纯度）或 ``"regression"``（方差）。
        criterion: 分类不纯度口径，``"gini"``（默认，``1 - Σ p²``）或 ``"entropy"``
            （``-Σ p log p``，自然对数，0·log0 = 0）。**只对分类有意义**：
            ``task="regression"`` 时传 ``"entropy"`` 直接抛 ValueError；
            取其它字符串同样抛 ValueError。默认 ``"gini"`` 与历史行为逐位一致。

    返回:
        dict，嵌套结构；内部节点含
        ``feature``（切分特征下标）、``threshold``（切分阈值）、``left``/``right``（子节点）、
        ``value``（该节点预测值）、``n_samples``、``impurity``；
        **叶子不含 feature/threshold**，只有 ``value``/``n_samples``/``impurity``。
        判据为 ``x[:, feature] <= threshold`` 走 ``left``。

    算法:
        1. 对当前节点先算出预测值（分类取多数类，平票取最小标签；回归取均值）和不纯度；
        2. 若深度到顶、样本数不足、节点已纯（不纯度 0）则成为叶子；
        3. 否则对每个特征按取值排序，只在**相邻不同取值的中点**尝试切分，
           用累积和一次算出所有候选点的加权不纯度；
        4. 取加权不纯度最小的 (特征, 阈值) 递归建子树；若最优切分不能严格降低不纯度
           或会让某一侧为空，则该节点停在原地成为叶子。
        分类的左右不纯度按 ``criterion`` 算：Gini 为 ``1 - Σ (n_k/n)²``，熵为
        ``-Σ (n_k/n) log(n_k/n)``，再按两侧样本数加权求和后除以节点样本数。

    复杂度:
        时间 O(depth · d · n log n)（每个节点对每个特征排序一次）/ 空间 O(n · depth)。

    陷阱:
        1. **贪心且无回看**：XOR 这类"单个特征上不纯度完全不下降"的关系，树会直接停在
           根节点（本实现要求严格下降），需要先做特征交叉或加深/换模型。
        2. 阈值只取相邻取值中点，因此对连续特征做的是阶梯逼近；深度不够时线性关系
           只能被拟合成阶梯，残差不是随机误差而是系统性的。
        3. 分类树叶子返回的是**多数类标签**，不是概率；要概率需要自己在叶子里存类别分布。
        4. ``max_depth=0`` 合法但只返回常数预测，别把它当成"树没建成功"。
        5. ``criterion="entropy"`` 只对分类有意义：在回归目标上算熵没有意义，本函数
           直接抛 ValueError；换 criterion 会改变选出的切分（熵对不纯节点更敏感，
           对均衡的多分类尤其明显），因此它改变的是树的形状与预测，不只是换个名字。
        6. ``criterion`` 是完全字符串匹配：``"Entropy"``、``"GINI"`` 都会报错，
           不做大小写或别名归一。

    参考:
        Breiman, Friedman, Olshen & Stone, "Classification and Regression Trees", 1984。
        Quinlan, "Induction of Decision Trees", Machine Learning, 1986（熵/信息增益口径）。
    """
    x = as_matrix(X, "X")
    _check_task(task)
    if not isinstance(criterion, str) or criterion not in ("gini", "entropy"):
        raise ValueError(f'criterion 只支持 "gini"/"entropy"，得到 {criterion!r}')
    if task == "regression" and criterion == "entropy":
        raise ValueError(
            'criterion="entropy" 只对分类有意义，回归（task="regression"）请用 "gini"'
        )
    n = x.shape[0]
    if not isinstance(max_depth, (int, np.integer)) or isinstance(max_depth, bool):
        raise ValueError(f"max_depth 必须是整数，得到 {max_depth!r}")
    max_depth = int(max_depth)
    if max_depth < 0:
        raise ValueError(f"max_depth 必须 >= 0，得到 {max_depth}")
    if min_samples_split < 2:
        raise ValueError(f"min_samples_split 必须 >= 2，得到 {min_samples_split}")
    if min_samples_leaf < 1:
        raise ValueError(f"min_samples_leaf 必须 >= 1，得到 {min_samples_leaf}")

    if task == "classification":
        yv = _int_labels(y, "y")
        _check_len(yv, n, "y")
        classes = np.unique(yv)
        if classes.size < 1:
            raise ValueError("y 不能为空")
    else:
        yv = as_vector(y, "y")
        _check_len(yv, n, "y")
        classes = np.zeros(0, dtype=int)

    return _build_tree(
        x, yv, 0, max_depth, int(min_samples_split), int(min_samples_leaf),
        task, classes, None, None, criterion,
    )


def decision_tree_predict(tree: dict, X: MatrixLike) -> np.ndarray:
    """用已训练好的 CART 树预测：每个样本沿 ``x[:, f] <= threshold`` 下行到叶子。

    参数:
        tree: ``decision_tree_fit`` 返回的嵌套 dict。
        X: 样本矩阵，形状 (n_samples, n_features)。

    返回:
        np.ndarray，形状 (n_samples,)；分类树返回 int 标签，回归树返回 float。

    算法:
        对每个样本从根节点开始，按节点自带的 feature/threshold 二分下行，
        直到遇到不含 ``feature`` 键的叶子，取叶子 ``value``。

    复杂度:
        时间 O(n · depth) / 空间 O(n)。

    陷阱:
        1. 这是**逐样本 Python 循环**，样本量大时很慢；生产环境请把树展开成数组运算。
        2. 叶子判定依据是"没有 feature 键"：若调用方手工改过树结构（比如给叶子补了
           feature），预测会走进 ``KeyError`` 或死循环。
        3. 特征数少于树里记录的下标时不会报错，numpy 会抛 IndexError，
           信息不如显式校验清楚。

    参考:
        同 ``decision_tree_fit``。
    """
    x = as_matrix(X, "X")
    if not isinstance(tree, dict):
        raise ValueError(f"tree 必须是 dict，得到 {type(tree).__name__}")
    if "value" not in tree:
        raise ValueError("tree 缺少 'value' 键，不是 decision_tree_fit 的输出")
    values = [_tree_predict_one(tree, x[i]) for i in range(x.shape[0])]
    if all(isinstance(v, (int, np.integer)) for v in values):
        return np.asarray(values, dtype=int)
    return np.asarray(values, dtype=float)


# ---------------------------------------------------------------------------
# 随机森林
# ---------------------------------------------------------------------------


def random_forest_fit(
    X: MatrixLike,
    y: ArrayLike,
    n_trees: int = 25,
    max_depth: int = 6,
    max_features="sqrt",
    min_samples_leaf: int = 1,
    seed: Optional[int] = None,
    task: str = "classification",
) -> dict:
    """随机森林：自助采样建多棵 CART，每次分裂再随机抽一部分特征。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 标签/目标，长度 n_samples。
        n_trees: 树的数量，>= 1。
        max_depth: 每棵树的最大深度，>= 0。
        max_features: 每次分裂的候选特征数：``"sqrt"``（默认，取 ceil 前向下取整的
            ``int(sqrt(d))``）、``"log2"``、int 绝对值，或 (0,1] 的比例。
        min_samples_leaf: 叶子最少样本数，>= 1。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。
        task: ``"classification"`` 或 ``"regression"``。

    返回:
        dict，键为：
        ``trees``            长度 n_trees 的 list，每项是一棵 CART 树；
        ``feature_importance`` 形状 (n_features,)，按不纯度下降算并归一化到和为 1；
        ``oob_score``        float，袋外得分：分类是袋外**准确率**，
                             回归是袋外 **R²**（1 - SS_res/SS_tot）；
        ``n_trees``          int，实际训练的树数；
        ``classes``          分类时为形状 (K,) 的升序类别数组，回归时为**长度 0 的空数组**
                             （占位，便于 :func:`random_forest_predict` 统一处理）。

    算法:
        1. 每棵树用 ``gen.integers(0, n, n)`` 做一次有放回抽样（自助样本），
           未被抽中的样本就是该树的袋外（OOB）样本；
        2. 建树时每个节点从随机特征子集里挑最优切分（Bagging + 随机子空间）；
        3. OOB 得分：对每个样本，只用"它属于袋外"的那些树投票/取均值
           （分类多数投票、回归平均），再与真值比较；没有任何袋外预测的样本不计入。

    复杂度:
        时间 O(n_trees · depth · d · n log n) / 空间 O(n_trees · n)。

    陷阱:
        1. 自助抽样下**每棵树大约只用 63.2% 的样本**，其余是袋外；OOB 得分因此天然
           比训练集得分可信，但树很少（如 n_trees < 10）时它的方差很大。
        2. 袋外样本数为 0 的样本（理论概率 (1-1/n)^n·n 很小，但 n 小时真实存在）
           会被排除在 OOB 之外；若全部样本都没有袋外预测，``oob_score`` 返回 0.0，
           这不能解读为"模型完全不能用"。
        3. 特征重要性按**训练集**的不纯度下降累计，对高基数/连续特征有偏好
           （可能把噪声特征排到前面）；要更公平请用 ``permutation_importance``。
        4. 回归的 OOB R² 可以为负，说明还不如直接预测均值。

    参考:
        Breiman, "Random Forests", Machine Learning, 2001。
    """
    x = as_matrix(X, "X")
    _check_task(task)
    n, d = x.shape
    if not isinstance(n_trees, (int, np.integer)) or isinstance(n_trees, bool):
        raise ValueError(f"n_trees 必须是整数，得到 {n_trees!r}")
    n_trees = int(n_trees)
    if n_trees < 1:
        raise ValueError(f"n_trees 必须 >= 1，得到 {n_trees}")
    if not isinstance(max_depth, (int, np.integer)) or isinstance(max_depth, bool):
        raise ValueError(f"max_depth 必须是整数，得到 {max_depth!r}")
    max_depth = int(max_depth)
    if max_depth < 0:
        raise ValueError(f"max_depth 必须 >= 0，得到 {max_depth}")
    if min_samples_leaf < 1:
        raise ValueError(f"min_samples_leaf 必须 >= 1，得到 {min_samples_leaf}")

    if task == "classification":
        yv = _int_labels(y, "y")
        _check_len(yv, n, "y")
        classes = np.unique(yv)
    else:
        yv = as_vector(y, "y")
        _check_len(yv, n, "y")
        classes = np.zeros(0, dtype=int)

    gen = rng(seed)
    trees: List[dict] = []
    oob_indices: List[np.ndarray] = []
    for _ in range(n_trees):
        boot = gen.integers(0, n, size=n)
        in_bag = np.zeros(n, dtype=bool)
        in_bag[boot] = True
        oob = np.flatnonzero(~in_bag)
        oob_indices.append(oob)
        trees.append(
            _build_tree(
                x[boot], yv[boot], 0, max_depth, 2, int(min_samples_leaf),
                task, classes, gen, max_features,
            )
        )

    # ---- 袋外得分 ----
    oob_score = 0.0
    if task == "classification":
        counts = np.zeros((n, classes.size), dtype=float)
        for tree, oob in zip(trees, oob_indices):
            if oob.size == 0:
                continue
            p = decision_tree_predict(tree, x[oob])
            counts[oob, np.searchsorted(classes, p)] += 1.0
        has = counts.sum(axis=1) > 0
        if np.any(has):
            pred = classes[np.argmax(counts[has], axis=1)]
            oob_score = float(np.mean(pred == yv[has]))
    else:
        total = np.zeros(n, dtype=float)
        cnt = np.zeros(n, dtype=float)
        for tree, oob in zip(trees, oob_indices):
            if oob.size == 0:
                continue
            total[oob] += decision_tree_predict(tree, x[oob])
            cnt[oob] += 1.0
        has = cnt > 0
        if np.any(has):
            pred = total[has] / cnt[has]
            resid = float(np.sum((yv[has] - pred) ** 2))
            total_ss = float(np.sum((yv[has] - yv[has].mean()) ** 2))
            oob_score = 0.0 if total_ss <= 0.0 else 1.0 - resid / total_ss

    imp = _normalize_importance(_tree_raw_importance(trees, d))
    return {
        "trees": trees,
        "feature_importance": imp,
        "oob_score": float(oob_score),
        "n_trees": int(n_trees),
        # 分类时给出训练集里出现过的全部类别：投票必须覆盖所有类别，
        # 若只从各树预测结果推类别，会漏掉"所有树都没预测到"的少数类。
        # 回归时给空数组占位。
        "classes": np.asarray(classes).ravel(),
    }


def random_forest_predict(forest: dict, X: MatrixLike) -> np.ndarray:
    """随机森林预测：分类对每棵树的投票做多数表决，回归取各树均值。

    参数:
        forest: ``random_forest_fit`` 返回的 dict。
        X: 样本矩阵，形状 (n_samples, n_features)。

    返回:
        np.ndarray，形状 (n_samples,)；分类为 int 标签，回归为 float。

    算法:
        逐棵树调用 ``decision_tree_predict``，把结果堆成 (n_trees, n_samples)；
        分类时类别集合优先取 ``forest["classes"]``（fit 时记录的全部类别），
        只有该键缺失时才退回"各树预测结果的并集"；随后按各类别的票数取最多者
        （平票取最小类别）。回归取各树预测值的列均值。

    复杂度:
        时间 O(n_trees · n · depth) / 空间 O(n_trees · n)。

    陷阱:
        1. 分类投票是**硬投票**（每棵树一票），没有用叶子概率做软投票；
           树数少且叶子样本少时，硬投票的方差比软投票大。
        2. 森林里的特征顺序完全固定，预测时 ``X`` 的列必须与训练时同序；
           本函数不做任何特征名对齐。
        3. 平票取最小类别：二分类且树数为偶数时平票很常见，这不是随机事件，
           换个种子可能整片样本的预测都翻转。
        4. 类别集合来自 ``forest["classes"]``（fit 时的**训练集**类别），因此预测集里
           出现训练集没有的新类别时，该样本的投票会全部落到已知类别上——它不会报错，
           而是被强行归到某个已有类别，评估前请先确认预测集与训练集类别集合一致。
        5. 分类还是回归**不看** ``task`` 字段，而是看 ``trees[0]`` 输出的 dtype 是否为整数；
           正常 fit 出来的森林没问题，但手工拼装的 forest（或把分类树/回归树混在一起）
           会按第一棵树的口径静默走错分支，且不会报错。

    参考:
        同 ``random_forest_fit``。
    """
    if not isinstance(forest, dict) or "trees" not in forest:
        raise ValueError("forest 必须是 random_forest_fit 的返回 dict（含 'trees' 键）")
    trees = forest["trees"]
    if not isinstance(trees, (list, tuple)) or len(trees) == 0:
        raise ValueError("forest['trees'] 必须是非空 list")
    x = as_matrix(X, "X")
    stacked = np.vstack([decision_tree_predict(t, x).astype(float) for t in trees])
    first = decision_tree_predict(trees[0], x)
    if np.issubdtype(first.dtype, np.integer):
        # 类别集合优先取 fit 时记录的 classes（投票必须覆盖全部类别，
        # 只从预测结果推类别会漏掉"某些树始终没预测到"的少数类）。
        if "classes" in forest:
            classes = np.asarray(forest["classes"]).ravel().astype(int)
        else:
            classes = np.unique(stacked.astype(int))
        counts = np.zeros((x.shape[0], classes.size), dtype=float)
        for row in stacked:
            counts[np.arange(x.shape[0]), np.searchsorted(classes, row.astype(int))] += 1.0
        return classes[np.argmax(counts, axis=1)]
    return stacked.mean(axis=0)


def random_forest_feature_importance(forest: dict) -> np.ndarray:
    """随机森林的特征重要性，按不纯度下降累计并归一化到和为 1。

    参数:
        forest: ``random_forest_fit`` 返回的 dict。

    返回:
        np.ndarray，形状 (n_features,)，非负且 ``sum == 1``；若所有树都不分裂
        （例如 max_depth=0），返回均匀分布 1/d 而不是全 0。

    算法:
        遍历每棵树的所有内部节点，累加
        ``n_node·impurity - n_left·impurity_left - n_right·impurity_right``
        到该节点的切分特征（负贡献截断为 0），最后除以总和。

    复杂度:
        时间 O(n_trees · 节点数) / 空间 O(d)。

    陷阱:
        1. 这是**训练集上的不纯度重要性**，对连续/高基数特征有系统性偏好，
           不能直接当因果重要性用；需要更公平的口径请用 ``permutation_importance``。
        2. 相关性强的特征之间会"分走"重要性，即使两个特征都很重要，各自也可能
           只拿到一半；不要据此排除特征。
        3. 全部为 0 时返回均匀分布是个**约定**（不是真的等权重要），
           调用方应结合 ``max_depth`` 判断这是不是退化情形。

    参考:
        同 ``random_forest_fit``。
    """
    if not isinstance(forest, dict) or "trees" not in forest:
        raise ValueError("forest 必须是 random_forest_fit 的返回 dict（含 'trees' 键）")
    trees = forest["trees"]
    if not isinstance(trees, (list, tuple)) or len(trees) == 0:
        raise ValueError("forest['trees'] 必须是非空 list")
    n_features = int(len(forest.get("feature_importance", np.zeros(0))))
    if n_features == 0:
        # 退化：从第一棵树递归数出特征数不可靠，这里要求调用方提供 fit 的输出。
        raise ValueError("forest 里缺少 'feature_importance'，无法确定特征数")
    return _normalize_importance(_tree_raw_importance(list(trees), n_features))


# ---------------------------------------------------------------------------
# 梯度提升
# ---------------------------------------------------------------------------


def gradient_boosting_fit(
    X: MatrixLike,
    y: ArrayLike,
    n_estimators: int = 30,
    learning_rate: float = 0.1,
    max_depth: int = 2,
    task: str = "regression",
) -> dict:
    """梯度提升：每轮用一棵浅回归树拟合当前模型的负梯度（残差）。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 目标，长度 n_samples；``task="classification"`` 时必须是 0/1（或两类标签）。
        n_estimators: 提升轮数（树数），>= 1。
        learning_rate: 学习率（收缩系数），必须 > 0 且 <= 1。
        max_depth: 每棵回归树的深度，>= 0。
        task: ``"regression"``（平方损失）或 ``"classification"``（对数损失 logit）。

    返回:
        dict，键为：
        ``init``          float，初始常数预测（回归为 y 均值，分类为对数几率）；
        ``trees``         长度 n_estimators 的 list，每项是一棵拟合残差的回归树；
        ``learning_rate``  float；
        ``task``          str；
        ``train_loss``    长度 n_estimators 的 list，每轮迭代结束后的训练损失
                          （回归为 MSE，分类为二元交叉熵）。

    算法:
        1. 初始化 ``F_0``：回归取 ``mean(y)``；分类取 ``log(p/(1-p))``，p 为 y 的均值
           （截断到 [1e-6, 1-1e-6] 以避免 ±inf）；
        2. 第 t 轮：算当前预测的概率/值，取负梯度
           （回归残差 ``y - F``；分类残差 ``y - sigmoid(F)``）；
        3. 用 CART 回归树（方差不纯度）拟合该负梯度；
        4. ``F ← F + learning_rate · tree.predict(X)``，记录该轮的 train_loss。

    复杂度:
        时间 O(n_estimators · depth · d · n log n) / 空间 O(n_estimators · n)。

    陷阱:
        1. 这是**函数空间上的最速下降的近似**：每棵树叶子的取值用的是回归树的均值，
           没有做 Friedman 的"单步 Newton 修正"（叶子值线性搜索），
           因此同样轮数下收敛比标准 GBM 慢一些，但方向正确。
        2. 分类只支持二分类，且内部把类别升序排列后把**较大标签当正类**；
           多分类需要 softmax 版本，本实现不做。
        3. learning_rate 太大（如 1.0）会震荡，太小则需要很多棵树；
           本实现没有早停，训练损失单调下降不代表验证集也在下降。
        4. ``train_loss`` 是**训练集**损失，不能用来判断过拟合。

    参考:
        Friedman, "Greedy function approximation: a gradient boosting machine",
        Annals of Statistics, 2001。
    """
    x = as_matrix(X, "X")
    _check_task(task)
    n = x.shape[0]
    if not isinstance(n_estimators, (int, np.integer)) or isinstance(n_estimators, bool):
        raise ValueError(f"n_estimators 必须是整数，得到 {n_estimators!r}")
    n_estimators = int(n_estimators)
    if n_estimators < 1:
        raise ValueError(f"n_estimators 必须 >= 1，得到 {n_estimators}")
    if not (0.0 < float(learning_rate) <= 1.0):
        raise ValueError(f"learning_rate 必须在 (0, 1] 内，得到 {learning_rate}")
    if not isinstance(max_depth, (int, np.integer)) or isinstance(max_depth, bool):
        raise ValueError(f"max_depth 必须是整数，得到 {max_depth!r}")
    max_depth = int(max_depth)
    if max_depth < 0:
        raise ValueError(f"max_depth 必须 >= 0，得到 {max_depth}")

    if task == "regression":
        yv = as_vector(y, "y")
        _check_len(yv, n, "y")
        init = float(np.mean(yv))
        f_val = np.full(n, init, dtype=float)
        trees: List[dict] = []
        losses: List[float] = []
        for _ in range(n_estimators):
            residual = yv - f_val
            tree = _build_tree(
                x, residual, 0, max_depth, 2, 1, "regression",
                np.zeros(0, dtype=int), None, None,
            )
            trees.append(tree)
            f_val = f_val + float(learning_rate) * decision_tree_predict(tree, x)
            losses.append(float(np.mean((yv - f_val) ** 2)))
        return {
            "init": init,
            "trees": trees,
            "learning_rate": float(learning_rate),
            "task": task,
            "train_loss": losses,
        }

    yl = _int_labels(y, "y")
    _check_len(yl, n, "y")
    classes = np.unique(yl)
    if classes.size != 2:
        raise ValueError(f"梯度提升分类只支持二分类，y 含 {classes.size} 个类别：{classes.tolist()}")
    yb = (yl == classes[1]).astype(float)
    p0 = float(np.clip(np.mean(yb), 1e-6, 1.0 - 1e-6))
    init = float(np.log(p0 / (1.0 - p0)))
    f_val = np.full(n, init, dtype=float)
    trees = []
    losses = []
    for _ in range(n_estimators):
        p = _sigmoid(f_val)
        residual = yb - p
        tree = _build_tree(
            x, residual, 0, max_depth, 2, 1, "regression",
            np.zeros(0, dtype=int), None, None,
        )
        trees.append(tree)
        f_val = f_val + float(learning_rate) * decision_tree_predict(tree, x)
        p = np.clip(_sigmoid(f_val), 1e-12, 1.0 - 1e-12)
        losses.append(float(-np.mean(yb * np.log(p) + (1.0 - yb) * np.log(1.0 - p))))
    return {
        "init": init,
        "trees": trees,
        "learning_rate": float(learning_rate),
        "task": task,
        "train_loss": losses,
    }


def gradient_boosting_predict(model: dict, X: MatrixLike) -> np.ndarray:
    """梯度提升预测：累加各棵树的输出（乘学习率）后按 task 变换。

    参数:
        model: ``gradient_boosting_fit`` 返回的 dict。
        X: 样本矩阵，形状 (n_samples, n_features)。

    返回:
        np.ndarray，形状 (n_samples,)；回归为 float 预测值，
        分类为 int 标签 0/1（``sigmoid(F) > 0.5`` 即 F > 0）。

    算法:
        ``F = init + learning_rate · Σ_t tree_t(X)``；分类再取 ``1[F > 0]``。

    复杂度:
        时间 O(n_estimators · n · depth) / 空间 O(n)。

    陷阱:
        1. 分类返回的是**硬标签**，阈值固定 0.5；需要概率请自己调 ``_sigmoid``
           或按业务代价调整阈值，不要以为 0.5 一定最优。
        2. 累加顺序会影响浮点末位，同一模型两次调用结果**逐位一致**（同样的顺序），
           但与其他实现比对时不要用 1e-15 级别的容差。
        3. 训练时用了几棵树，预测就必须用完整的 ``model["trees"]``；
           想早停必须自己在 fit 阶段截断树列表。

    参考:
        同 ``gradient_boosting_fit``。
    """
    if not isinstance(model, dict) or "trees" not in model:
        raise ValueError("model 必须是 gradient_boosting_fit 的返回 dict（含 'trees' 键）")
    x = as_matrix(X, "X")
    lr = float(model["learning_rate"])
    f_val = np.full(x.shape[0], float(model["init"]), dtype=float)
    for tree in model["trees"]:
        f_val = f_val + lr * decision_tree_predict(tree, x)
    if model.get("task") == "classification":
        return (f_val > 0.0).astype(int)
    return f_val


# ---------------------------------------------------------------------------
# 高斯朴素贝叶斯
# ---------------------------------------------------------------------------


def gaussian_nb_fit(X: MatrixLike, y: ArrayLike) -> dict:
    """高斯朴素贝叶斯训练：按类估计先验与各特征的高斯均值/方差。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 类别标签，长度 n_samples。

    返回:
        dict，键为：
        ``classes`` 形状 (K,) 的升序类别；
        ``prior``   形状 (K,)，类先验 ``n_c / n``；
        ``mean``    形状 (K, n_features)，各类各特征的均值；
        ``var``     形状 (K, n_features)，各类各特征的方差（ddof=0），
                    已加 ``1e-9`` 平滑以避开 0 方差。

    算法:
        1. 类别取 ``np.unique(y)`` 的升序，先验用频率估计；
        2. 对每个类取子集，按列算均值与总体方差；
        3. 方差统一加 1e-9，避免某特征在某类上所有取值相同（方差 0）时
           对数似然出现除零。

    复杂度:
        时间 O(n d) / 空间 O(K d)。

    陷阱:
        1. **1e-9 是绝对量级平滑，不是相对量级**：若某特征量纲极小（如 1e-6），
           1e-9 几乎无作用；若量纲极大（如 1e6），1e-9 也几乎无作用——它只是
           防止除零，不是真正的正则化。要稳健请先标准化。
        2. 方差用 ddof=0（总体），与 sklearn 的 ``GaussianNB``（ddof=0）一致，
           但与"样本方差"教材口径差一个 n/(n-1)。
        3. 特征必须独立才能叫"朴素"；强相关特征会让似然被重复计入，
           后验过度自信。本实现不做任何相关校正。

    参考:
        Duda, Hart & Stork, "Pattern Classification", 2nd ed., 2001, §3.5。
    """
    x = as_matrix(X, "X")
    yv = _int_labels(y, "y")
    n, d = x.shape
    _check_len(yv, n, "y")

    classes = np.unique(yv)
    k = classes.size
    prior = np.zeros(k, dtype=float)
    mean = np.zeros((k, d), dtype=float)
    var = np.zeros((k, d), dtype=float)
    for i, c in enumerate(classes):
        sub = x[yv == c]
        if sub.shape[0] == 0:
            raise ValueError(f"类别 {int(c)} 没有样本")
        prior[i] = sub.shape[0] / float(n)
        mean[i] = sub.mean(axis=0)
        var[i] = sub.var(axis=0) + 1e-9
    return {"classes": classes, "prior": prior, "mean": mean, "var": var}


def gaussian_nb_predict(model: dict, X: MatrixLike) -> dict:
    """高斯朴素贝叶斯预测：用对数似然计算各类后验并取最大者。

    参数:
        model: ``gaussian_nb_fit`` 返回的 dict。
        X: 样本矩阵，形状 (n_samples, n_features)。

    返回:
        dict，键为：
        ``pred``     形状 (n_samples,) 的 int 预测标签；
        ``log_prob`` 形状 (n_samples, K) 的**归一化对数后验**
                     （每行 logsumexp = 0），等价于对数似然比，
                     用"减最大值"技巧避免下溢。

    算法:
        1. 逐类算对数似然
           ``log p(x|c) = -0.5·Σ[log(2π·var) + (x-mean)^2/var]``；
        2. 加上 ``log(prior)`` 得未归一化对数后验；
        3. 每行减去该行最大值后取 exp 归一化，再取 log 得到归一化对数后验
           （直接 exp 未归一化的值会在特征维度大时整体下溢为 0）。

    复杂度:
        时间 O(n K d) / 空间 O(n K)。

    陷阱:
        1. ``log_prob`` 是**归一化后**的对数后验，不是 sklearn 的
           ``predict_log_proba`` 之外的其他东西；它每行和为 1（指数域），
           可以放心比较两类的大小。
        2. 先验为 0 的类不会出现（``np.unique`` 只给出出现过的类），
           所以不需要 ``log(0)`` 保护；但这也意味着模型**永远不会预测没见过的类**。
        3. 特征维度很大时，即使做了归一化，所有类的后验也可能接近 0/1 的两极，
           表现为"过度自信"，这是朴素贝叶斯独立性假设的固有后果。

    参考:
        同 ``gaussian_nb_fit``。
    """
    if not isinstance(model, dict) or "classes" not in model:
        raise ValueError("model 必须是 gaussian_nb_fit 的返回 dict（含 'classes' 键）")
    x = as_matrix(X, "X")
    classes = np.asarray(model["classes"]).ravel()
    prior = as_vector(model["prior"], "prior")
    mean = as_matrix(model["mean"], "mean")
    var = as_matrix(model["var"], "var")
    k = classes.size
    if prior.size != k or mean.shape[0] != k or var.shape[0] != k:
        raise ValueError("model 内部维度不一致：prior/mean/var 的行数与 classes 不符")
    if mean.shape[1] != x.shape[1] or var.shape[1] != x.shape[1]:
        raise ValueError(f"model 的特征数 {mean.shape[1]} 与 X 的 {x.shape[1]} 不一致")
    if np.any(var <= 0):
        raise ValueError("model['var'] 必须全为正（gaussian_nb_fit 已加 1e-9 平滑）")

    n = x.shape[0]
    log_post = np.empty((n, k), dtype=float)
    for i in range(k):
        diff = x - mean[i]
        ll = -0.5 * np.sum(np.log(2.0 * np.pi * var[i]) + diff ** 2 / var[i], axis=1)
        log_post[:, i] = ll + float(np.log(prior[i]))
    shift = log_post.max(axis=1, keepdims=True)
    log_post = log_post - shift
    log_post = log_post - np.log(np.sum(np.exp(log_post), axis=1, keepdims=True))
    pred = classes[np.argmax(log_post, axis=1)]
    return {"pred": pred.astype(int), "log_prob": log_post}


# ---------------------------------------------------------------------------
# 线性判别分析
# ---------------------------------------------------------------------------


def lda_fit(X: MatrixLike, y: ArrayLike, n_components: Optional[int] = None) -> dict:
    """线性判别分析：解 ``Sw^{-1} Sb`` 的广义特征问题，得到判别方向。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 类别标签，长度 n_samples，至少 2 类。
        n_components: 保留的判别方向数；None 表示 ``min(K-1, n_features)``。
            必须满足 ``1 <= n_components <= n_features``。

    返回:
        dict，键为：
        ``scalings``       形状 (n_features, n_components)，每一列是一个判别方向；
        ``eigenvalues``    形状 (n_components,)，``Sw^{-1} Sb`` 的特征值（降序）；
        ``explained_ratio`` 形状 (n_components,)，特征值占比；
        ``classes``        形状 (K,) 的升序类别；
        ``mean``           形状 (n_features,)，总体均值；
        ``priors``         形状 (K,)，类先验 n_c/n。

    算法:
        1. 类内散度 ``Sw = Σ_c Σ_{i∈c} (x_i-μ_c)(x_i-μ_c)'``，
           类间散度 ``Sb = Σ_c n_c (μ_c-μ)(μ_c-μ)'``；
        2. 白化：``Sw = V diag(λ) V'``，取 ``W = V diag(λ^{-1/2})``（λ 截断到 1e-12）；
        3. 对 ``M = W' Sb W`` 用 ``np.linalg.eigh``（对称矩阵专用）求特征分解，
           再按特征值降序排列；
        4. ``scalings = W @ U``，特征值再做 ``clip(0, None)`` 后归一化为 explained_ratio。

    复杂度:
        时间 O(n d^2 + d^3) / 空间 O(d^2)。

    陷阱:
        1. ``Sw`` 奇异（样本数少于特征数，或某特征在类内完全不变）时白化会放大噪声方向：
           本实现把 λ 截断到 1e-12，因此**不会报错但可能给出巨大且不稳定的方向**，
           实践中应先做 PCA 降维或标准化。
        2. ``explained_ratio`` 是按**全部非负特征值**归一化的；当 ``n_components``
           小于 ``K-1`` 时，返回的若干个 ratio 之和会小于 1，这是正常的，
           不要把它当成"丢了信息"的 bug。
        3. 判别方向不唯一（同一特征值可对应任意旋转后的方向），不同实现的符号/基可能不同；
           比较时请比投影后的判别能力，而不是直接比 ``scalings`` 的数值。
        4. 只用了线性判别，没有做贝叶斯决策规则（本模块的 ``gaussian_nb`` 才是概率模型）。

    参考:
        Fisher, "The use of multiple measurements in taxonomic problems", 1936；
        Rao, "The utilization of multiple measurements in problems of biological
        classification", 1948。
    """
    x = as_matrix(X, "X")
    yv = _int_labels(y, "y")
    n, d = x.shape
    _check_len(yv, n, "y")
    classes = np.unique(yv)
    k = classes.size
    if k < 2:
        raise ValueError(f"LDA 至少需要 2 个类别，得到 {k}")
    if n_components is None:
        n_comp = min(k - 1, d)
    else:
        if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
            raise ValueError(f"n_components 必须是整数，得到 {n_components!r}")
        n_comp = int(n_components)
    if not (1 <= n_comp <= d):
        raise ValueError(f"n_components 必须在 [1, {d}] 内，得到 {n_comp}")

    overall = x.mean(axis=0)
    sw = np.zeros((d, d), dtype=float)
    sb = np.zeros((d, d), dtype=float)
    priors = np.zeros(k, dtype=float)
    for i, c in enumerate(classes):
        sub = x[yv == c]
        if sub.shape[0] == 0:
            raise ValueError(f"类别 {int(c)} 没有样本")
        mu_c = sub.mean(axis=0)
        diff = sub - mu_c
        sw += diff.T @ diff
        delta = (mu_c - overall).reshape(d, 1)
        sb += sub.shape[0] * (delta @ delta.T)
        priors[i] = sub.shape[0] / float(n)

    sw = 0.5 * (sw + sw.T)
    sb = 0.5 * (sb + sb.T)
    eig_sw, vec_sw = np.linalg.eigh(sw)
    eig_sw = np.clip(eig_sw, 1e-12, None)
    white = vec_sw / np.sqrt(eig_sw)[None, :]
    m = white.T @ sb @ white
    m = 0.5 * (m + m.T)
    eig_m, vec_m = np.linalg.eigh(m)
    order = np.argsort(-eig_m, kind="mergesort")
    eig_m = np.clip(eig_m[order], 0.0, None)

    total = float(np.sum(eig_m))
    ratio = eig_m / total if total > 0.0 else np.zeros_like(eig_m)
    return {
        "scalings": white @ vec_m[:, order][:, :n_comp],
        "eigenvalues": eig_m[:n_comp],
        "explained_ratio": ratio[:n_comp],
        "classes": classes,
        "mean": overall,
        "priors": priors,
    }


def lda_transform(model: dict, X: MatrixLike) -> np.ndarray:
    """把样本投影到 LDA 的判别方向上。

    参数:
        model: ``lda_fit`` 返回的 dict。
        X: 样本矩阵，形状 (n_samples, n_features)。

    返回:
        np.ndarray，形状 (n_samples, n_components)，``(X - mean) @ scalings``。

    算法:
        先减去训练时的总体均值，再右乘判别方向矩阵（线性投影）。

    复杂度:
        时间 O(n d c) / 空间 O(n c)。

    陷阱:
        1. **必须平移**：直接 ``X @ scalings`` 会保留全局均值，投影后的类质心位置
           虽然仍然可分，但数值与原实现不一致；判别方向本身不含截距。
        2. 投影后的各维是同一组方向的坐标，尺度没有归一化，特征值大的方向数值也大；
           画图前通常需要自己缩放。
        3. 投影维度 <= K-1，所以 K 类数据最多只能降到 K-1 维，别期望它能任意降维。

    参考:
        同 ``lda_fit``。
    """
    if not isinstance(model, dict) or "scalings" not in model:
        raise ValueError("model 必须是 lda_fit 的返回 dict（含 'scalings' 键）")
    x = as_matrix(X, "X")
    mean = as_vector(model["mean"], "mean")
    scal = as_matrix(model["scalings"], "scalings")
    if mean.size != x.shape[1]:
        raise ValueError(f"model 的 mean 长度 {mean.size} 与 X 的特征数 {x.shape[1]} 不一致")
    if scal.shape[0] != x.shape[1]:
        raise ValueError(f"model 的 scalings 行数 {scal.shape[0]} 与 X 的特征数 {x.shape[1]} 不一致")
    return (x - mean) @ scal


# ---------------------------------------------------------------------------
# 主成分分析（PCA）
# ---------------------------------------------------------------------------


def pca_fit(X: MatrixLike, n_components: Optional[int] = None) -> dict:
    """主成分分析（PCA）：中心化后做 SVD，取前 k 个奇异向量作为主成分方向。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)，至少 2 个样本
            （``explained_variance`` 用 ``n-1`` 做分母，n=1 时无定义）。
        n_components: 保留的主成分个数 k，需满足 1 <= k <= min(n_samples, n_features)；
            None 表示取满 ``min(n_samples, n_features)``。

    返回:
        dict，键为：
        ``mean`` 形状 (d,) 的列均值；
        ``components`` 形状 (k, d)，第 i 行是第 i 个主成分的**载荷向量**（单位向量）；
        ``singular_values`` 形状 (k,)，``X - mean`` 的奇异值 s_i（降序）；
        ``explained_variance`` 形状 (k,)，口径为 ``s_i² / (n_samples - 1)``，
        即样本协方差矩阵（ddof=1）的特征值；
        ``explained_variance_ratio`` 形状 (k,)，为
        ``explained_variance[i] / (Σ_{j<min(n,d)} s_j² / (n-1))``——分母取**全部**
        min(n,d) 个奇异值，因此 k 取满时各分量之和恰为 1，截断后只反映保留部分；
        ``n_components`` / ``n_features`` / ``n_samples`` 三个 int。

    算法:
        1. ``mean = X.mean(axis=0)``、``Xc = X - mean``；
        2. 对 ``Xc`` 做经济型 SVD ``Xc = U S V^T``（``np.linalg.svd``，full_matrices=False）；
        3. 取前 k 行 ``V^T`` 为 ``components``、前 k 个奇异值为 ``singular_values``；
        4. **确定性符号约定**：令每个主成分中绝对值最大的载荷为正
           （``components[i, argmax|components[i]|] > 0``，绝对值并列时取最小下标）；
        5. ``explained_variance = s²/(n-1)``，比例按全部奇异值平方和归一。

    复杂度:
        时间 O(n d min(n,d)) / 空间 O(n d + k d)。

    陷阱:
        1. 主成分的**符号只是约定**：``-v`` 与 ``v`` 表示同一个方向。这里统一取
           "绝对值最大的载荷为正"，所以同一份数据总能得到逐位相同的输出；但别的实现
           （例如 sklearn 默认的 u-based 翻转）可能整体反号。对拍时应统一符号或只比
           绝对值，不要因为符号不同就判定算错。
        2. ``explained_variance`` 的分母是 ``n-1``（样本协方差口径，与 sklearn 的
           ``PCA.explained_variance_`` 一致）；用 ``n`` 会让数值系统性偏小。
        3. 比例的归一化分母包含被截断掉的分量：k 小于 min(n,d) 时比例之和 < 1，
           这只表示"保留了多少方差"，不要当成算法有误。
        4. PCA 对**量纲**敏感：某列方差比其它列大几个数量级时会独占第一主成分，
           建库前请先 ``standardize_fit`` 再 PCA。
        5. 本函数不做任何缺失值填补，含 NaN/inf 时由 ``as_matrix`` 抛 ValueError。

    参考:
        Jolliffe, "Principal Component Analysis", 2nd ed., Springer, 2002。
    """
    x = as_matrix(X, "X")
    n, d = x.shape
    if n < 2:
        raise ValueError(f"PCA 至少需要 2 个样本（方差以 n-1 为分母），得到 {n}")
    if n_components is None:
        k = min(n, d)
    else:
        if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
            raise ValueError(f"n_components 必须是整数或 None，得到 {n_components!r}")
        k = int(n_components)
        if k < 1:
            raise ValueError(f"n_components 必须 >= 1，得到 {k}")
        if k > min(n, d):
            raise ValueError(
                f"n_components={k} 不能大于 min(n_samples, n_features)={min(n, d)}"
            )

    mean = x.mean(axis=0)
    xc = x - mean
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:k].copy()
    singular_values = s[:k].copy()
    # 确定性符号约定：让每个主成分里绝对值最大的载荷为正（并列时取最小下标）。
    for i in range(k):
        row = components[i]
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0.0:
            components[i] = -row
    full_var = (s ** 2) / float(n - 1)
    explained_variance = full_var[:k].copy()
    total = float(np.sum(full_var))
    ratio = explained_variance / total if total > 0.0 else np.zeros(k, dtype=float)
    return {
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": ratio,
        "n_components": int(k),
        "n_features": int(d),
        "n_samples": int(n),
    }


def pca_transform(model: dict, X: MatrixLike) -> np.ndarray:
    """把样本投影到 PCA 主成分空间，返回得分矩阵 ``(X - mean) @ components.T``。

    参数:
        model: ``pca_fit`` 返回的 dict（至少要含 ``mean`` 与 ``components``）。
        X: 样本矩阵，形状 (n_samples, n_features)，特征数必须与训练时一致。

    返回:
        np.ndarray，形状 (n_samples, n_components)；第 i 列是样本在第 i 个主成分上的
        得分（投影坐标，已中心化，因此各列均值为 0）。

    算法:
        先减训练集列均值，再右乘载荷矩阵的转置（纯线性投影，不含任何缩放）。

    复杂度:
        时间 O(n d k) / 空间 O(n k)。

    陷阱:
        1. **必须平移**：直接 ``X @ components.T`` 会保留训练均值，得分矩阵各列不再
           零均值，与 ``pca_fit`` 的协方差口径不一致；重建误差也会因此算错。
        2. 得分列的尺度是原始数据的量纲（奇异值大小），没有归一化；画图前通常要自己
           缩放，不要以为各列同尺度。
        3. 不做主成分个数对齐：``model["components"]`` 里有几行就投影出几列，
           想截断请自己在 ``pca_fit`` 时用 ``n_components``。
        4. 特征数与训练时不一致时抛 ValueError；本函数不接受 ``n_components`` 参数，
           传入会被忽略（Python 直接报 TypeError）。

    参考:
        同 ``pca_fit``。
    """
    if not isinstance(model, dict) or "components" not in model or "mean" not in model:
        raise ValueError("model 必须是 pca_fit 的返回 dict（含 'components' 与 'mean' 键）")
    x = as_matrix(X, "X")
    components = as_matrix(model["components"], "components")
    mean = as_vector(model["mean"], "mean")
    if mean.size != x.shape[1]:
        raise ValueError(f"model 的 mean 长度 {mean.size} 与 X 的特征数 {x.shape[1]} 不一致")
    if components.shape[1] != x.shape[1]:
        raise ValueError(
            f"model 的 components 列数 {components.shape[1]} 与 X 的特征数 {x.shape[1]} 不一致"
        )
    return (x - mean) @ components.T


# ---------------------------------------------------------------------------
# 置换重要性
# ---------------------------------------------------------------------------


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: MatrixLike,
    y: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 5,
    seed: Optional[int] = None,
) -> dict:
    """置换重要性：打乱某一列后模型得分的平均下降量。

    参数:
        predict_fn: 可调用对象，``predict_fn(X) -> 预测数组``，输入是 (n, d) 数组。
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 真值，长度 n_samples。
        metric_fn: 可调用对象，``metric_fn(y_true, y_pred) -> float``，**越大越好**
            （例如准确率、负 MSE）。
        n_repeats: 每个特征重复打乱的次数，>= 1。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``importances_mean`` 形状 (n_features,)，``baseline - 打乱后得分`` 的均值；
        ``importances_std``  形状 (n_features,)，各次重复得分的标准差（ddof=0）。

    算法:
        1. 算基准得分 ``base = metric_fn(y, predict_fn(X))``；
        2. 对每个特征 f，重复 n_repeats 次：复制 X，把第 f 列整体按随机置换重排
           （列内的值不变，只切断它与 y 的对应关系），重新预测并算得分；
        3. 重要性 = base - 打乱后得分；对重复取均值与标准差。

    复杂度:
        时间 O(n_features · n_repeats · (预测开销)) / 空间 O(n d)。

    陷阱:
        1. 只打乱**一列**时，与该列强相关的其他列仍在，模型仍能部分恢复信息，
           因此相关特征组会**分摊**重要性，单个特征的重要性看起来偏小甚至为负。
        2. 重要性可以是负数（打乱后反而变好），这是噪声或数据泄漏的信号，
           不要强行截断为 0 后再比较。
        3. 每次重复的置换不同，所以结果带随机性；必须固定 seed 才能复现。
        4. ``metric_fn`` 的方向必须是"越大越好"；传 MSE 进来会得到反号的结论。

    参考:
        Breiman, "Random Forests", 2001（OOB 置换重要性的雏形）；
        Fisher, Rudin & Dominici, "Model Class Reliance", JMLR 2019（相关特征问题）。
    """
    x = as_matrix(X, "X")
    yv = np.asarray(y)
    yv = yv.ravel()
    n, d = x.shape
    if yv.size != n:
        raise ValueError(f"y 长度 {yv.size} 与样本数 {n} 不一致")
    if not callable(predict_fn):
        raise ValueError("predict_fn 必须可调用")
    if not callable(metric_fn):
        raise ValueError("metric_fn 必须可调用")
    if not isinstance(n_repeats, (int, np.integer)) or isinstance(n_repeats, bool):
        raise ValueError(f"n_repeats 必须是整数，得到 {n_repeats!r}")
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise ValueError(f"n_repeats 必须 >= 1，得到 {n_repeats}")

    gen = rng(seed)
    base = float(metric_fn(yv, np.asarray(predict_fn(x))))
    imp = np.zeros((d, n_repeats), dtype=float)
    for f in range(d):
        for r in range(n_repeats):
            xp = x.copy()
            xp[:, f] = xp[gen.permutation(n), f]
            imp[f, r] = base - float(metric_fn(yv, np.asarray(predict_fn(xp))))
    return {
        "importances_mean": imp.mean(axis=1),
        "importances_std": imp.std(axis=1),
    }


# ---------------------------------------------------------------------------
# SMOTE 与类权重
# ---------------------------------------------------------------------------


def smote(X: MatrixLike, y: ArrayLike, k: int = 5, seed: Optional[int] = None) -> dict:
    """SMOTE 过采样：只在少数类样本与其同类近邻的连线上插值生成新样本。

    参数:
        X: 样本矩阵，形状 (n_samples, n_features)。
        y: 类别标签，长度 n_samples，至少 2 类。
        k: 用于插值的近邻个数；实际取 ``min(k, n_c - 1)``，要求 >= 1。
        seed: 随机种子；None 表示使用 ``DEFAULT_SEED``。

    返回:
        dict，键为：
        ``X``            形状 (n_total, n_features)，原样本在前、合成样本在后；
        ``y``            形状 (n_total,) 的 int 标签；
        ``n_synthetic``  int，合成样本总数。

    算法:
        1. 统计各类样本数 ``n_c``，多数类为 ``n_max``；
        2. 对每个 ``n_c < n_max`` 的类，在其自身样本间算欧氏距离，
           取 ``k' = min(k, n_c-1)`` 个最近邻；
        3. 重复 ``n_max - n_c`` 次：随机挑一个种子样本 i、随机挑一个近邻 j，
           生成 ``x_new = x_i + u·(x_j - x_i)``，``u ~ U(0,1)``；
        4. 把合成样本接到原数据后面（顺序固定，可复现）。

    复杂度:
        时间 O(Σ_c n_c^2 d + 合成数·d) / 空间 O(Σ_c n_c^2 + n_total d)。

    陷阱:
        1. **必须在划分训练/测试之后做**：先 SMOTE 再划分会让同一少数类样本的近邻
           同时出现在训练和测试集，测试指标严重偏乐观。
        2. 插值只在少数类内部进行，因此它是**过采样而非生成模型**：合成点必然落在
           原始少数类样本的凸包附近，不会创造新的分布形态，对高维稀疏数据效果很差。
        3. 少数类样本数 < 2 时无法插值，本函数抛 ValueError（调用方应改用复制或权重）。
        4. ``k`` 被静默限制为 ``n_c - 1``：少数类很少时 k 实际会更小，这一信息只能从
           返回值推不出来，论文里要写清实际用的 k。
        5. 未标准化时插值方向被量纲大的特征支配，先标准化再 SMOTE。

    参考:
        Chawla, Bowyer, Hall & Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling
        Technique", JAIR 2002。
    """
    x = as_matrix(X, "X")
    yv = _int_labels(y, "y")
    n, _d = x.shape
    _check_len(yv, n, "y")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError(f"k 必须是整数，得到 {k!r}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k 必须 >= 1，得到 {k}")

    classes, counts = np.unique(yv, return_counts=True)
    if classes.size < 2:
        raise ValueError(f"SMOTE 至少需要 2 个类别，得到 {classes.size}")
    n_max = int(counts.max())
    gen = rng(seed)

    parts_x: List[np.ndarray] = [x]
    parts_y: List[np.ndarray] = [yv]
    n_syn = 0
    for c, cnt in zip(classes, counts):
        need = n_max - int(cnt)
        if need <= 0:
            continue
        members = x[yv == c]
        if members.shape[0] < 2:
            raise ValueError(
                f"类别 {int(c)} 只有 {members.shape[0]} 个样本，无法做 SMOTE 插值"
            )
        kk = min(k, members.shape[0] - 1)
        dist = _pairwise_sq_dist(members, members)
        np.fill_diagonal(dist, np.inf)
        neighbour = np.argsort(dist, axis=1, kind="mergesort")[:, :kk]
        syn = np.empty((need, x.shape[1]), dtype=float)
        for t in range(need):
            i = int(gen.integers(members.shape[0]))
            j = int(neighbour[i, int(gen.integers(kk))])
            u = float(gen.random())
            syn[t] = members[i] + u * (members[j] - members[i])
        parts_x.append(syn)
        parts_y.append(np.full(need, int(c), dtype=int))
        n_syn += need

    return {
        "X": np.vstack(parts_x),
        "y": np.concatenate(parts_y).astype(int),
        "n_synthetic": int(n_syn),
    }


def class_weight_balanced(y: ArrayLike) -> dict:
    """按类别频率的倒数计算平衡权重 ``n / (K · n_c)``。

    参数:
        y: 类别标签，长度 n_samples。

    返回:
        dict，键为：
        ``classes`` 形状 (K,) 的升序类别；
        ``weights`` 形状 (K,)，第 i 个类的权重 ``n / (K · n_c)``。

    算法:
        1. 统计每类样本数 n_c 与类别数 K；
        2. ``w_c = n / (K · n_c)``。

    复杂度:
        时间 O(n) / 空间 O(K)。

    陷阱:
        1. 这个口径下权重之**和不为 1**（等于 ``(1/K)Σ n/n_c``），
           它是"平均权重为 1"的标度：各类加权后的总权重相同。
           若框架要求权重和为 1，需要再自己做归一化。
        2. 完全平衡的数据上每个类权重恰好为 1；越不平衡，少数类权重越大，
           等价于把少数类的损失放大——这会牺牲多数类精度换取少数类召回。
        3. 标签集合由 ``np.unique`` 决定，**没有出现的类别不会获得权重**；
           如果训练集某一折恰好缺了某个类，权重向量长度会变，下游按类别对齐时务必小心。

    参考:
        King & Zeng, "Logistic Regression in Rare Events Data", Political Analysis, 2001
        （不平衡数据加权的最小二乘/似然口径）。
    """
    yv = _int_labels(y, "y")
    classes, counts = np.unique(yv, return_counts=True)
    n = yv.size
    k = classes.size
    weights = n / (k * counts.astype(float))
    return {"classes": classes, "weights": weights}


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict，键名简短 ASCII，值为 int/float/bool/list，固定种子下两次调用完全一致：
        ``knn_accuracy``/``knn_mean_distance``、``auc_separable``/``auc_reversed``/
        ``auc_tied``、``std_col_means``/``std_col_stds``、``cm_matrix``/``cm_accuracy``/
        ``cm_f1_macro``、``kfold_sizes``、``skfold_pos_ratio``、``tree_clf_accuracy``/
        ``tree_reg_rmse``、``rf_clf_accuracy``/``rf_oob_score``/``rf_importance``/
        ``rf_importance_sum``、``gb_reg_loss_first``/``gb_reg_loss_last``/``gb_reg_mse``/
        ``gb_clf_accuracy``、``gnb_accuracy``/``gnb_logprob_max_dev``、
        ``lda_ratio_sum``/``lda_proj_accuracy``、``perm_importance_mean``、
        ``smote_n_synthetic``/``smote_class_counts``、``class_weight_balanced``、
        ``split_sizes``/``split_stratified_ratio``、``dist_self_max``/
        ``dist_direct_max_dev``/``dist_big_self_max``、``tree_criterion_root_gini``/
        ``tree_criterion_root_entropy``/``tree_criterion_pred_diff``/
        ``tree_criterion_gini_loss``/``tree_criterion_entropy_loss``、
        ``pca_explained_variance_ratio``/``pca_transform_row0``/``pca_ev_dev``/
        ``pca_recon_max_dev``/``pca_ratio_sum_k1``、``pr_curve_precision``/
        ``pr_curve_thresholds``/``ap_ties``/``ap_all_tied``、``err_paths_checked``。

    算法:
        1. 用固定种子造两簇完全可分数据（簇心相距约 11.3、簇内 sd=0.35），
           它同时充当 kNN/树/森林/朴素贝叶斯的"必须 100% 正确"算例；
        2. AUC 用完全可分（1.0）、完全反向（0.0）、全并列（0.5）三个闭式结论对拍；
        3. 混淆矩阵/分类指标用 2×2 手算例 ``y=[0,0,1,1], ŷ=[0,1,1,0]``
           （全指标 = 0.5）；标准化按"均值 0、标准差 1"独立验证；
        4. 分层 K 折用 90:30 的不平衡标签，断言每折少数类比例与 0.25 相差 < 0.05；
        5. LDA 用三类高斯数据，断言 explained_ratio 之和为 1 且投影空间里
           "最近质心分类" 100% 正确（与 LDA 自身的目标函数独立）；
        6. 置换重要性用"预测只依赖第 0 列"的模型，断言第 1 列重要性恒为 0；
        7. SMOTE 断言合成数 = 多数类 − 少数类、过采样后两类数量相等；
        8. 回归树断言"加深必然降低训练 RMSE"（单调性），梯度提升断言首尾损失下降；
        9. 距离矩阵断言 ``(x, x)`` 的自距离**严格为 0**、与"逐元素平方差"口径偏差
           < 1e-12，并额外用一个坐标 ~1e11 的正抵消算例（旧实现自距离 ~2896.31）；
        10. 决策树判据用一个 Gini 与熵**切在不同位置**的手算算例，断言两者预测确实
            不同，且各自根切分等于暴力枚举该判据下的最优切分（互为反向交叉检查）；
        11. PCA 与 ``ddof=1`` 协方差矩阵的特征分解（独立口径）对拍，并检查载荷正交、
            得分零均值、全成分可重建、k 截断后比例之和 < 1；
        12. PR 曲线断言首点 recall=1、末点 precision=1/recall=0、阈值严格升序、
            recall 单调不增、全并列分数只有 1 个切点；AP 与手算阶梯积分对拍；
        13. 最后逐个触发新增函数的非法输入，断言全部抛 ValueError。

    复杂度:
        时间 O(数万次标量运算) / 空间 O(n^2) 以下。

    陷阱:
        1. 两簇算例的间距必须远大于簇内标准差，否则 kNN/森林的 100% 断言会随机失败；
           调小间距会让这个自测变脆，不要那样改。
        2. 梯度提升的损失阈值是按固定种子实测后留了余量写的；换种子或换数据
           生成顺序都可能需要重新确认阈值。
        3. 所有断言都返回 round 后的值参与 json 比对，所以任何未固定的随机性都会
           在双跑比对里暴露。

    参考:
        本模块各函数的参考文献。
    """
    gen = rng(20240501)

    # ---- 两簇完全可分数据：kNN/树/森林/高斯NB 的"必须 100% 正确"算例 ----
    c0 = np.array([-4.0, -4.0]) + 0.35 * gen.standard_normal((40, 2))
    c1 = np.array([4.0, 4.0]) + 0.35 * gen.standard_normal((40, 2))
    x_all = np.vstack([c0, c1])
    y_all = np.array([0] * 40 + [1] * 40, dtype=int)
    x_tr = np.vstack([c0[:25], c1[:25]])
    y_tr = np.array([0] * 25 + [1] * 25, dtype=int)
    x_te = np.vstack([c0[25:], c1[25:]])
    y_te = np.array([0] * 15 + [1] * 15, dtype=int)

    knn = knn_predict(x_tr, y_tr, x_te, k=3)
    knn_acc = float(np.mean(knn["pred"] == y_te))
    if knn_acc != 1.0:
        raise AssertionError(f"kNN 在完全可分两簇上准确率应为 1.0，得到 {knn_acc}")

    # ---- AUC 的秩和闭式结论：完全可分 1.0 / 完全反向 0.0 / 全并列 0.5 ----
    auc_pos = roc_auc(y_te, x_te[:, 0])
    auc_neg = roc_auc(y_te, -x_te[:, 0])
    auc_tie = roc_auc(y_te, np.zeros(y_te.size))
    if abs(auc_pos - 1.0) > 1e-12 or abs(auc_neg) > 1e-12:
        raise AssertionError(f"AUC 应分别为 1.0/0.0，得到 {auc_pos}/{auc_neg}")
    if abs(auc_tie - 0.5) > 1e-12:
        raise AssertionError(f"全并列评分时 AUC 应为 0.5（平均秩的直接推论），得到 {auc_tie}")

    # ---- 标准化：均值 ~0、标准差 ~1；常数列 std 置 1 ----
    st = standardize_fit(x_all)
    zs = standardize_apply(x_all, st["mean"], st["std"])
    std_means = np.abs(zs.mean(axis=0))
    std_stds = np.abs(zs.std(axis=0) - 1.0)
    if std_means.max() > 1e-9 or std_stds.max() > 1e-9:
        raise AssertionError(f"标准化后应均值 0/标准差 1，得到 {std_means}/{std_stds}")
    const = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    if float(standardize_fit(const)["std"][0]) != 1.0:
        raise AssertionError("常数列的标准差应被置 1，否则标准化会除零")

    # ---- 混淆矩阵/分类指标：手算 2×2 例，全部指标 = 0.5 ----
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 0])
    if cm["matrix"] != [[1, 1], [1, 1]] or cm["labels"] != [0, 1]:
        raise AssertionError(f"手算混淆矩阵应为 [[1,1],[1,1]]，得到 {cm}")
    met = classification_metrics([0, 0, 1, 1], [0, 1, 1, 0])
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        if abs(met[key] - 0.5) > 1e-12:
            raise AssertionError(f"手算例的 {key} 应为 0.5，得到 {met[key]}")
    if met["n_classes"] != 2:
        raise AssertionError(f"手算例类别数应为 2，得到 {met['n_classes']}")

    # ---- K 折：并集必须恰好覆盖所有下标且无交集 ----
    folds = kfold_indices(20, k=4, seed=3)
    kfold_sizes = [len(f) for f in folds]
    merged = sorted(sum(folds, []))
    if merged != list(range(20)):
        raise AssertionError(f"K 折测试集下标应恰好覆盖 0..19，得到 {merged}")

    # ---- 分层 K 折：90:30 数据、k=3，每折少数类比例应精确为 0.25 ----
    y_imb = np.array([0] * 90 + [1] * 30, dtype=int)
    sfolds = stratified_kfold_indices(y_imb, k=3, seed=5)
    ratios = [float(np.count_nonzero(y_imb[f] == 1)) / len(f) for f in sfolds]
    if max(abs(r - 0.25) for r in ratios) > 0.05:
        raise AssertionError(f"分层 K 折每折少数类比例应接近 0.25，得到 {ratios}")

    # ---- 分类树：完全可分数据上训练/测试都 100% ----
    tree = decision_tree_fit(x_tr, y_tr, max_depth=4)
    tree_acc = float(np.mean(decision_tree_predict(tree, x_te) == y_te))
    if tree_acc != 1.0:
        raise AssertionError(f"决策树在完全可分两簇上准确率应为 1.0，得到 {tree_acc}")

    # ---- 回归树：加深必然降低训练 RMSE（单调性断言）----
    x_reg = np.linspace(-2.0, 2.0, 60).reshape(-1, 1)
    y_reg = 3.0 * x_reg[:, 0] + 1.0
    pred_deep = decision_tree_predict(
        decision_tree_fit(x_reg, y_reg, max_depth=5, task="regression"), x_reg
    )
    pred_shallow = decision_tree_predict(
        decision_tree_fit(x_reg, y_reg, max_depth=1, task="regression"), x_reg
    )
    rmse_deep = float(np.sqrt(np.mean((pred_deep - y_reg) ** 2)))
    rmse_shallow = float(np.sqrt(np.mean((pred_shallow - y_reg) ** 2)))
    if not (rmse_deep < rmse_shallow):
        raise AssertionError(f"加深应降低训练 RMSE：deep={rmse_deep} shallow={rmse_shallow}")
    if rmse_deep > 0.5:
        raise AssertionError(f"深度 5 的回归树拟合直线 RMSE 应 < 0.5，得到 {rmse_deep}")

    # ---- 随机森林：测试准确率 100%，袋外准确率 >= 0.9，重要性归一到 1 ----
    forest = random_forest_fit(x_tr, y_tr, n_trees=25, max_depth=6, seed=11)
    rf_acc = float(np.mean(random_forest_predict(forest, x_te) == y_te))
    rf_oob = float(forest["oob_score"])
    if rf_acc != 1.0:
        raise AssertionError(f"随机森林在完全可分两簇上准确率应为 1.0，得到 {rf_acc}")
    if rf_oob < 0.9:
        raise AssertionError(f"随机森林袋外准确率应 >= 0.9，得到 {rf_oob}")
    rf_imp = random_forest_feature_importance(forest)
    if abs(float(rf_imp.sum()) - 1.0) > 1e-12:
        raise AssertionError(f"特征重要性应归一化到和为 1，得到 {rf_imp.sum()}")

    # ---- 高斯朴素贝叶斯：完全可分数据 100%；归一化对数后验每行 logsumexp = 0 ----
    gnb = gaussian_nb_fit(x_tr, y_tr)
    gnb_out = gaussian_nb_predict(gnb, x_te)
    gnb_acc = float(np.mean(gnb_out["pred"] == y_te))
    if gnb_acc != 1.0:
        raise AssertionError(f"高斯朴素贝叶斯在完全可分两簇上准确率应为 1.0，得到 {gnb_acc}")
    lp = gnb_out["log_prob"]
    gnb_dev = float(np.max(np.abs(np.log(np.sum(np.exp(lp), axis=1)))))
    if gnb_dev > 1e-9:
        raise AssertionError(f"归一化对数后验每行的 logsumexp 应为 0，最大偏差 {gnb_dev}")

    # ---- 梯度提升回归：损失必须下降；分类：可分数据 100% ----
    x_gb = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    y_gb = 2.0 * x_gb[:, 0] + 1.0
    gbm = gradient_boosting_fit(x_gb, y_gb, n_estimators=30, learning_rate=0.2, max_depth=2)
    gb_pred = gradient_boosting_predict(gbm, x_gb)
    gb_losses = [float(v) for v in gbm["train_loss"]]
    gb_mse = float(np.mean((gb_pred - y_gb) ** 2))
    if not (gb_losses[-1] < gb_losses[0]):
        raise AssertionError(f"梯度提升训练损失应下降：{gb_losses[0]} -> {gb_losses[-1]}")
    if gb_mse > 0.05 * float(np.var(y_gb)):
        raise AssertionError(f"梯度提升拟合直线的 MSE 应远小于 y 的方差，得到 {gb_mse}")
    y_clf = (x_gb[:, 0] > 0.0).astype(int)
    gbc = gradient_boosting_fit(
        x_gb, y_clf, n_estimators=30, learning_rate=0.3, max_depth=2, task="classification"
    )
    gb_clf_acc = float(np.mean(gradient_boosting_predict(gbc, x_gb) == y_clf))
    if gb_clf_acc != 1.0:
        raise AssertionError(f"梯度提升分类在可分数据上准确率应为 1.0，得到 {gb_clf_acc}")

    # ---- LDA：explained_ratio 之和为 1；投影空间最近质心分类 100% ----
    centers3 = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
    x3 = np.vstack([c + 0.4 * gen.standard_normal((30, 2)) for c in centers3])
    y3 = np.repeat([0, 1, 2], 30).astype(int)
    lda = lda_fit(x3, y3)
    lda_ratio_sum = float(np.sum(lda["explained_ratio"]))
    if abs(lda_ratio_sum - 1.0) > 1e-9:
        raise AssertionError(f"LDA explained_ratio 之和应为 1，得到 {lda_ratio_sum}")
    proj = lda_transform(lda, x3)
    centroids = np.vstack([proj[y3 == c].mean(axis=0) for c in range(3)])
    d2 = np.sum((proj[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    lda_acc = float(np.mean(np.argmin(d2, axis=1) == y3))
    if lda_acc != 1.0:
        raise AssertionError(f"LDA 投影后最近质心分类应 100% 正确，得到 {lda_acc}")

    # ---- 置换重要性：模型只看第 0 列，第 1 列重要性必须恒为 0 ----
    x_perm = np.column_stack([gen.random(60), gen.random(60)])
    y_perm = 3.0 * x_perm[:, 0]
    perm = permutation_importance(
        lambda Z: 3.0 * Z[:, 0],
        x_perm,
        y_perm,
        lambda t, p: -float(np.mean((t - p) ** 2)),
        n_repeats=5,
        seed=13,
    )
    perm_mean = [float(v) for v in perm["importances_mean"]]
    if not (perm_mean[0] > 0.0):
        raise AssertionError(f"有效特征的置换重要性应为正，得到 {perm_mean}")
    if abs(perm_mean[1]) > 1e-9:
        raise AssertionError(f"被模型忽略的特征置换重要性应为 0，得到 {perm_mean[1]}")

    # ---- SMOTE：合成数 = 多数类 − 少数类，过采样后两类数量相等 ----
    x_sm = np.vstack([
        0.5 * gen.standard_normal((30, 2)) + 3.0,
        0.5 * gen.standard_normal((10, 2)) - 3.0,
    ])
    y_sm = np.array([0] * 30 + [1] * 10, dtype=int)
    sm = smote(x_sm, y_sm, k=3, seed=17)
    sm_counts = [int(np.count_nonzero(sm["y"] == c)) for c in (0, 1)]
    if sm["n_synthetic"] != 20 or sm_counts[0] != sm_counts[1]:
        raise AssertionError(f"SMOTE 应生成 20 个样本使两类相等，得到 {sm['n_synthetic']}/{sm_counts}")
    if sm["X"].shape != (60, 2):
        raise AssertionError(f"SMOTE 后样本数应为 60，得到 {sm['X'].shape}")

    # ---- 类权重闭式：40/(2*30) = 0.666667, 40/(2*10) = 2.0 ----
    cw = class_weight_balanced(y_sm)
    cw_expected = [40.0 / (2 * 30), 40.0 / (2 * 10)]
    if max(abs(float(a) - b) for a, b in zip(cw["weights"], cw_expected)) > 1e-12:
        raise AssertionError(f"平衡类权重应为 {cw_expected}，得到 {cw['weights'].tolist()}")

    # ---- 数据划分：条数守恒 + 分层后测试集正类比例仍为 0.5 ----
    split = train_test_split(x_all, y_all, test_size=0.25, seed=19)
    split_sizes = [int(split["X_train"].shape[0]), int(split["X_test"].shape[0])]
    if split_sizes != [60, 20]:
        raise AssertionError(f"80 个样本按 0.25 划分应为 60/20，得到 {split_sizes}")
    split_s = train_test_split(x_all, y_all, test_size=0.25, seed=19, stratify=y_all)
    strat_ratio = float(np.mean(split_s["y_test"] == 1))
    if abs(strat_ratio - 0.5) > 1e-12:
        raise AssertionError(f"分层划分后测试集正类比例应为 0.5，得到 {strat_ratio}")

    # ---- 距离矩阵：自距离必须严格为 0，且与"逐元素平方差"口径逐点一致 ----
    x_dist = np.array([[1.0, 2.0], [4.0, 6.0], [7.0, 1.0]])
    d_self = _pairwise_sq_dist(x_dist, x_dist)
    dist_self_max = float(np.max(np.abs(np.diag(d_self))))
    if dist_self_max != 0.0:
        raise AssertionError(f"(x, x) 的自距离应严格为 0，得到对角线 {np.diag(d_self)}")
    diff_d = x_dist[:, None, :] - x_dist[None, :, :]
    dist_direct = np.sqrt(np.einsum("ijk,ijk->ij", diff_d, diff_d))
    dist_dev = float(np.max(np.abs(d_self - dist_direct)))
    if dist_dev > 1e-12:
        raise AssertionError(f"距离矩阵与逐元素平方差口径的最大偏差应 < 1e-12，得到 {dist_dev}")

    # 正抵消算例：坐标 ~1e11 时 a²+b²-2ab 里两个 ~4e22 的项相减，有效位只剩 O(1)，
    # 旧实现 sqrt(clip(a²+b²-2ab)) 在这里得到约 2896.31 的**正自距离**（真值 0）。
    x_big = np.array([[100000000000.0, 100000000002.0], [100000000001.0, 100000000003.0]])
    d_big = _pairwise_sq_dist(x_big, x_big)
    dist_big_self_max = float(np.max(np.abs(np.diag(d_big))))
    if dist_big_self_max != 0.0:
        raise AssertionError(f"大坐标算例的自距离也必须严格为 0，得到 {dist_big_self_max}")
    if abs(float(d_big[0, 1]) - float(np.sqrt(2.0))) > 1e-12:
        raise AssertionError(f"大坐标算例两点距离应为 sqrt(2)，得到 {d_big[0, 1]}")

    # ---- CART 判据：同一算例上 Gini 与熵切在不同位置、预测确实不同，且都等于暴力枚举最优 ----
    x_cri = np.array([[1.0, 1.0], [3.0, 0.0], [2.0, 2.0], [0.0, 3.0], [3.0, 1.0],
                      [3.0, 2.0], [3.0, 3.0], [2.0, 1.0], [0.0, 2.0]])
    y_cri = np.array([0, 2, 2, 0, 1, 0, 0, 1, 2], dtype=int)
    crit_tree_gini = decision_tree_fit(x_cri, y_cri, max_depth=2, criterion="gini")
    crit_tree_ent = decision_tree_fit(x_cri, y_cri, max_depth=2, criterion="entropy")
    crit_root_gini = (int(crit_tree_gini["feature"]), float(crit_tree_gini["threshold"]))
    crit_root_ent = (int(crit_tree_ent["feature"]), float(crit_tree_ent["threshold"]))
    if crit_root_gini == crit_root_ent:
        raise AssertionError(f"Gini 与熵在该算例上应切在不同位置，得到同一根切分 {crit_root_gini}")
    crit_pred_gini = decision_tree_predict(crit_tree_gini, x_cri)
    crit_pred_ent = decision_tree_predict(crit_tree_ent, x_cri)
    crit_pred_diff = int(np.count_nonzero(crit_pred_gini != crit_pred_ent))
    if crit_pred_diff == 0:
        raise AssertionError("Gini 树与熵树在该算例上的预测应确实不同，实际完全一致")

    def _crit_impurity(f, thr, criterion):
        # 独立实现：给定切分的加权不纯度，不调用本模块的树代码。
        left = y_cri[x_cri[:, f] <= thr]
        right = y_cri[x_cri[:, f] > thr]
        total = 0.0
        for part in (left, right):
            cnt = np.array([np.count_nonzero(part == c) for c in (0, 1, 2)], dtype=float)
            pr = cnt / cnt.sum()
            if criterion == "gini":
                imp = 1.0 - float(np.sum(pr ** 2))
            else:
                safe = np.where(pr > 0.0, pr, 1.0)  # 0·log0 约定为 0
                imp = float(-np.sum(pr * np.log(safe)))
            total += float(cnt.sum()) * imp
        return total / float(y_cri.size)

    def _crit_brute_root(criterion):
        # 暴力枚举所有 (特征, 相邻不同取值中点)，取加权不纯度最小者。
        best = None
        best_score = np.inf
        for f in range(x_cri.shape[1]):
            vals = np.unique(x_cri[:, f])
            for a, b in zip(vals[:-1], vals[1:]):
                thr = 0.5 * (float(a) + float(b))
                score = _crit_impurity(f, thr, criterion)
                if score < best_score - 1e-15:
                    best_score = score
                    best = (int(f), thr)
        return best, float(best_score)

    crit_brute_gini, crit_score_gini = _crit_brute_root("gini")
    crit_brute_ent, crit_score_ent = _crit_brute_root("entropy")
    if crit_brute_gini != crit_root_gini:
        raise AssertionError(f"Gini 根切分应等于暴力枚举最优 {crit_brute_gini}，得到 {crit_root_gini}")
    if crit_brute_ent != crit_root_ent:
        raise AssertionError(f"熵根切分应等于暴力枚举最优 {crit_brute_ent}，得到 {crit_root_ent}")
    # 交叉检查：两个切分在**对方的**判据下都不是最优，说明差异来自判据本身而非搜索实现。
    crit_ent_of_gini = _crit_impurity(crit_root_gini[0], crit_root_gini[1], "entropy")
    crit_gini_of_ent = _crit_impurity(crit_root_ent[0], crit_root_ent[1], "gini")
    if not (crit_score_ent < crit_ent_of_gini - 1e-15):
        raise AssertionError(f"熵判据下熵树根切分应更优：{crit_score_ent} vs {crit_ent_of_gini}")
    if not (crit_score_gini < crit_gini_of_ent - 1e-15):
        raise AssertionError(f"Gini 判据下 Gini 树根切分应更优：{crit_score_gini} vs {crit_gini_of_ent}")
    if abs(crit_score_ent - crit_score_gini) < 1e-9:
        raise AssertionError("两个判据的最优加权不纯度不应数值相同，该算例无法体现判据差异")
    # criterion 也必须作用到节点不纯度本身：根节点的 impurity 应等于该判据下根分布的不纯度。
    crit_root_pr = np.array(
        [np.count_nonzero(y_cri == c) for c in (0, 1, 2)], dtype=float
    )
    crit_root_pr = crit_root_pr / crit_root_pr.sum()
    crit_root_gini_imp = 1.0 - float(np.sum(crit_root_pr ** 2))
    crit_root_ent_imp = float(-np.sum(crit_root_pr * np.log(crit_root_pr)))
    if abs(float(crit_tree_gini["impurity"]) - crit_root_gini_imp) > 1e-12:
        raise AssertionError(
            f"Gini 树根节点 impurity 应为 {crit_root_gini_imp}，"
            f"得到 {crit_tree_gini['impurity']}"
        )
    if abs(float(crit_tree_ent["impurity"]) - crit_root_ent_imp) > 1e-12:
        raise AssertionError(
            f"熵树根节点 impurity 应为 {crit_root_ent_imp}，得到 {crit_tree_ent['impurity']}"
        )

    # ---- PCA：与协方差矩阵特征分解（独立口径）对拍，并检查正交/零均值/重建 ----
    x_pca = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0], [5.0, 6.0], [6.0, 4.0]])
    pca = pca_fit(x_pca)
    if (pca["n_components"], pca["n_features"], pca["n_samples"]) != (2, 2, 6):
        raise AssertionError(
            f"PCA 形状元组应为 (2, 2, 6)，得到 "
            f"{(pca['n_components'], pca['n_features'], pca['n_samples'])}"
        )
    pca_cov = np.cov(x_pca, rowvar=False, ddof=1)
    pca_eig = np.sort(np.linalg.eigvalsh(pca_cov))[::-1]
    pca_ev_dev = float(np.max(np.abs(pca_eig - pca["explained_variance"])))
    if pca_ev_dev > 1e-9:
        raise AssertionError(
            f"explained_variance 应等于 ddof=1 协方差矩阵的特征值，最大偏差 {pca_ev_dev}"
        )
    pca_ratio_sum = float(np.sum(pca["explained_variance_ratio"]))
    if abs(pca_ratio_sum - 1.0) > 1e-12:
        raise AssertionError(f"k 取满时 explained_variance_ratio 之和应为 1，得到 {pca_ratio_sum}")
    pca_orth_dev = float(np.max(np.abs(pca["components"] @ pca["components"].T - np.eye(2))))
    if pca_orth_dev > 1e-12:
        raise AssertionError(f"主成分应两两正交且为单位向量，最大偏差 {pca_orth_dev}")
    pca_signed = pca["components"][np.arange(2), np.argmax(np.abs(pca["components"]), axis=1)]
    if np.any(pca_signed < 0.0):
        raise AssertionError("符号约定要求每个主成分里绝对值最大的载荷为正")
    pca_scores = pca_transform(pca, x_pca)
    pca_colmean_dev = float(np.max(np.abs(pca_scores.mean(axis=0))))
    if pca_colmean_dev > 1e-12:
        raise AssertionError(f"PCA 得分各列均值应为 0，最大偏差 {pca_colmean_dev}")
    pca_recon_dev = float(np.max(np.abs(pca_scores @ pca["components"] + pca["mean"] - x_pca)))
    if pca_recon_dev > 1e-9:
        raise AssertionError(f"用全部主成分应能重建中心化数据，最大偏差 {pca_recon_dev}")
    pca1 = pca_fit(x_pca, n_components=1)
    pca1_ratio = float(np.sum(pca1["explained_variance_ratio"]))
    if not (0.0 < pca1_ratio < 1.0):
        raise AssertionError(f"k < min(n,d) 时比例之和应严格小于 1，得到 {pca1_ratio}")

    # ---- PR 曲线 / AP：手算阶梯积分；全并列分数必须只产生一个切点 ----
    y_pr = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    s_pr = np.array([0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.4, 0.3])
    pr = pr_curve(y_pr, s_pr)
    if not (len(pr["precision"]) == len(pr["recall"]) == len(pr["thresholds"]) + 1):
        raise AssertionError("precision/recall 应比 thresholds 长一个（末尾那个点没有阈值）")
    if float(pr["recall"][0]) != 1.0 or float(pr["precision"][-1]) != 1.0:
        raise AssertionError("约定：首点 recall=1，末点 precision=1")
    if float(pr["recall"][-1]) != 0.0:
        raise AssertionError("约定：末点 recall=0")
    if not np.all(np.diff(pr["thresholds"]) > 0.0):
        raise AssertionError("thresholds 必须严格升序且只含不同取值")
    if not np.all(np.diff(pr["recall"]) <= 0.0):
        raise AssertionError("recall 必须随阈值升高而单调不增")
    pr_tie = pr_curve([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    if pr_tie["thresholds"].size != 1 or float(pr_tie["thresholds"][0]) != 0.5:
        raise AssertionError(f"全并列分数应只产生一个切点 0.5，得到 {pr_tie['thresholds']}")
    # 手算：升序 recall = 0,0.2,0.4,0.6,0.8,0.8,1,1 配 precision = 1,1,2/3,3/5,4/7,1/2,5/9,1/2。
    ap_hand = 0.2 * 1.0 + 0.2 * (2.0 / 3.0) + 0.2 * 0.6 + 0.2 * (4.0 / 7.0) + 0.2 * (5.0 / 9.0)
    ap_ties = average_precision_score(y_pr, s_pr)
    if abs(ap_ties - ap_hand) > 1e-12:
        raise AssertionError(f"AP 手算值应为 {ap_hand}，得到 {ap_ties}")
    if not (0.0 <= ap_ties <= 1.0):
        raise AssertionError(f"AP 应落在 [0, 1] 内，得到 {ap_ties}")
    ap_all_tied = average_precision_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    if abs(ap_all_tied - 0.5) > 1e-12:
        raise AssertionError(f"全并列时 AP 应等于正类比例 0.5，得到 {ap_all_tied}")
    if average_precision_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) != 1.0:
        raise AssertionError("完全可分时 AP 应为 1.0")

    # ---- 新增函数的错误路径：非法输入必须抛 ValueError（不许静默返回） ----
    err_cases = [
        ("pca_fit n_components=0", lambda: pca_fit(x_pca, n_components=0)),
        ("pca_fit n_components=3 超过 min(n,d)", lambda: pca_fit(x_pca, n_components=3)),
        ("pca_fit 只有 1 个样本", lambda: pca_fit(x_pca[:1])),
        ("pca_fit n_components=1.5 非整数", lambda: pca_fit(x_pca, n_components=1.5)),
        ("pca_transform 特征数不符", lambda: pca_transform(pca, np.zeros((2, 3)))),
        ("pca_transform model 缺 components", lambda: pca_transform({"mean": [0.0, 0.0]}, x_pca)),
        ("pr_curve 只有一个类别", lambda: pr_curve([1, 1, 1], [0.1, 0.2, 0.3])),
        ("pr_curve 长度不一致", lambda: pr_curve([0, 1], [0.1, 0.2, 0.3])),
        ("average_precision_score 只有一个类别",
         lambda: average_precision_score([1, 1], [0.1, 0.2])),
        ("decision_tree_fit criterion 非法",
         lambda: decision_tree_fit(x_cri, y_cri, criterion="foo")),
        ("decision_tree_fit 回归 + 熵",
         lambda: decision_tree_fit(x_cri, y_cri, task="regression", criterion="entropy")),
    ]
    err_paths_checked = 0
    for label, fn in err_cases:
        try:
            fn()
        except ValueError:
            err_paths_checked += 1
            continue
        raise AssertionError(f"{label} 应抛 ValueError，实际没有抛")

    return {
        "knn_accuracy": float(knn_acc),
        "knn_mean_distance": round(float(np.mean(knn["distance"])), 6),
        "auc_separable": round(float(auc_pos), 6),
        "auc_reversed": round(float(auc_neg), 6),
        "auc_tied": round(float(auc_tie), 6),
        "std_col_means": [round(float(v), 6) for v in std_means],
        "std_col_stds": [round(float(v), 6) for v in std_stds],
        "cm_matrix": cm["matrix"],
        "cm_accuracy": round(float(met["accuracy"]), 6),
        "cm_f1_macro": round(float(met["f1_macro"]), 6),
        "kfold_sizes": [int(v) for v in kfold_sizes],
        "skfold_pos_ratio": [round(float(v), 6) for v in ratios],
        "tree_clf_accuracy": float(tree_acc),
        "tree_reg_rmse": round(float(rmse_deep), 6),
        "rf_clf_accuracy": float(rf_acc),
        "rf_oob_score": round(float(rf_oob), 6),
        "rf_importance": [round(float(v), 6) for v in rf_imp],
        "rf_importance_sum": round(float(rf_imp.sum()), 6),
        "gb_reg_loss_first": round(float(gb_losses[0]), 6),
        "gb_reg_loss_last": round(float(gb_losses[-1]), 6),
        "gb_reg_mse": round(float(gb_mse), 6),
        "gb_clf_accuracy": float(gb_clf_acc),
        "gnb_accuracy": float(gnb_acc),
        "gnb_logprob_max_dev": float(gnb_dev),
        "lda_ratio_sum": round(float(lda_ratio_sum), 6),
        "lda_proj_accuracy": float(lda_acc),
        "perm_importance_mean": [round(float(v), 6) for v in perm_mean],
        "smote_n_synthetic": int(sm["n_synthetic"]),
        "smote_class_counts": [int(v) for v in sm_counts],
        "class_weight_balanced": [round(float(v), 6) for v in cw["weights"]],
        "split_sizes": [int(v) for v in split_sizes],
        "split_stratified_ratio": round(float(strat_ratio), 6),
        "dist_self_max": round(float(dist_self_max), 12),
        "dist_direct_max_dev": round(float(dist_dev), 12),
        "dist_big_self_max": round(float(dist_big_self_max), 12),
        "tree_criterion_root_gini": [int(crit_root_gini[0]), round(float(crit_root_gini[1]), 6)],
        "tree_criterion_root_entropy": [int(crit_root_ent[0]), round(float(crit_root_ent[1]), 6)],
        "tree_criterion_pred_diff": int(crit_pred_diff),
        "tree_criterion_gini_loss": round(float(crit_score_gini), 6),
        "tree_criterion_entropy_loss": round(float(crit_score_ent), 6),
        "pca_explained_variance_ratio": [
            round(float(v), 6) for v in pca["explained_variance_ratio"]
        ],
        "pca_transform_row0": [round(float(v), 6) for v in pca_scores[0]],
        "pca_ev_dev": round(float(pca_ev_dev), 12),
        "pca_recon_max_dev": round(float(pca_recon_dev), 12),
        "pca_ratio_sum_k1": round(float(pca1_ratio), 6),
        "pr_curve_precision": [round(float(v), 6) for v in pr["precision"]],
        "pr_curve_thresholds": [round(float(v), 6) for v in pr["thresholds"]],
        "ap_ties": round(float(ap_ties), 6),
        "ap_all_tied": round(float(ap_all_tied), 6),
        "err_paths_checked": int(err_paths_checked),
    }
