"""多准则决策扩展：PROMETHEE II、ELECTRE I/III、秩和比 RSR、Borda/Copeland 共识排序。

``evaluation.py`` 已经覆盖了 AHP/熵权/CRITIC/TOPSIS/VIKOR/灰关联/DEA/模糊综合，
本模块只补它没有的两族方法。**共 9 个公开函数**：

- **级别高于关系（outranking）**：``promethee_ii``（偏好函数 + 正负净流）、``electre_i``
  （一致性/不一致性矩阵 + 内核）、``electre_ii``（双阈值强/弱级别高于关系 + 升降蒸馏排序）、
  ``electre_iii``（带无差异/偏好/否决阈值的可信度 + 升降蒸馏）。
- **秩方法与共识**：``rank_sum_ratio`` 与 ``rsr_distribution``（加权秩和比 RSR 及其概率单位
  probit 分档）、``borda_count``、``copeland_score``，以及 ``rank_consensus``
  （多份排序之间的 Spearman 秩相关一致性诊断）。

这些实现是**教学透明版**：所有中间量（偏好度矩阵、一致性矩阵、可信度矩阵、蒸馏分组）
都显式返回，便于在论文里画出完整计算过程。生产环境可以用 ``pymcdm`` 等成熟库与这里的
结果互相印证——两边不一致通常说明阈值或指标方向的理解有偏差。

关键约定
--------
- **指标方向**：``benefit`` 是长度 n 的 bool 序列，``True`` 表示"越大越好"（正向指标），
  ``False`` 表示"越小越好"（成本型指标）；``None`` 表示全部正向。成本型指标在内部按
  ``Z = -X`` 做符号翻转，**不修改**调用方传入的矩阵。
- **权重**：长度 n、非负（负数会被拒绝）。内部把权重**正比例**缩放到和为 1，这一步只是把
  中间量（PROMETHEE 的净流、ELECTRE 的一致性、WRSR）摆到统一标度上便于读数，**不改变
  排名**：同一权重向量的任意正倍数给出完全相同的名次。但权重本身当然决定排名——
  换一组权重一般就换一个排名，"归一化不改排名"不等于"权重不影响排名"。
- **排名**：``rank`` 一律"1 为最好"，并列给相同名次（标准竞赛排名法 1,1,3 口径）。
- **标准化**：ELECTRE I/III 用向量归一化 ``r_ij = x_ij / sqrt(sum_i x_ij^2)``；
  PROMETHEE 直接用原值，因为偏好阈值 p/q 本身就带量纲，标准化会破坏它的物理含义。
- **随机性**：本模块的公开函数全部是确定性的；``_self_test`` 里的随机算例一律走
  ``_common.rng``（独立 Generator），不使用任何 ``np.random`` 全局状态。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import math

import numpy as np

from ._common import as_matrix, as_vector, rng

ArrayLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]

__all__ = [
    "promethee_ii",
    "electre_i",
    "electre_ii",
    "electre_iii",
    "rank_sum_ratio",
    "rsr_distribution",
    "borda_count",
    "copeland_score",
    "rank_consensus",
]

#: 判定用的统一容差：只用于"相等/支配"这类结构性比较，不用于数值收敛。
_EPS = 1e-12


# --------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------

def _resolve_benefit(benefit, n: int) -> np.ndarray:
    """把 benefit 统一成长度 n 的 bool 数组（None 表示全部正向指标）。

    参数:
        benefit: None / bool 序列 / 0-1 序列。
        n: 指标个数。

    返回:
        形状 (n,) 的 bool 数组。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        整数 0/1 序列会被当成 bool 处理；如果传的是"权重"之类的浮点序列，这里会报错而不是
        静默按非零当 True——静默解释是这类接口最常见的坑。
    """
    if benefit is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(benefit)
    if arr.ndim != 1:
        raise ValueError(f"benefit 必须是一维序列，得到 ndim={arr.ndim}")
    if arr.size != n:
        raise ValueError(f"benefit 长度 {arr.size} 与指标数 {n} 不一致")
    if arr.dtype == bool:
        return arr.astype(bool)
    if np.issubdtype(arr.dtype, np.integer) and set(np.unique(arr)).issubset({0, 1}):
        return arr.astype(bool)
    raise ValueError(f"benefit 必须是 bool（或 0/1）序列，得到 dtype={arr.dtype}")


def _normalize_weights(weights, n: int) -> np.ndarray:
    """把权重压成长度 n、非负、和为 1 的数组。

    参数:
        weights: 长度 n 的权重。
        n: 指标个数。

    返回:
        形状 (n,) 的 float 数组，和为 1。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        权重和为 0 或含负值时直接抛 ValueError：在 outranking 方法里负权重会让一致性指数
        失去"比例"含义，得到的结果无法解释。
    """
    w = as_vector(weights, "weights")
    if w.size != n:
        raise ValueError(f"权重长度 {w.size} 与指标数 {n} 不一致")
    if np.any(w < -_EPS):
        raise ValueError(f"权重不能为负，最小值为 {float(w.min())}")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("权重之和必须为正")
    return w / total


def _oriented(X: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """把成本型指标取负，得到"越大越好"口径的矩阵 Z。

    参数:
        X: 形状 (m, n) 的原始决策矩阵。
        benefit: 形状 (n,) 的 bool 方向标记。

    返回:
        形状 (m, n) 的 Z，成本型列已取负。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        取负只是把"越小越好"翻成"越大越好"，**没有**改变量纲；所以两种方向混用时，
        阈值 p/q/v 也必须按翻转后的量纲给（本模块的默认阈值就是从 Z 的极差推出的）。
    """
    Z = np.array(X, dtype=float, copy=True)
    cost = ~benefit
    if np.any(cost):
        Z[:, cost] = -Z[:, cost]
    return Z


def _ranks_from_scores(scores: ArrayLike) -> np.ndarray:
    """按"1 为最好"给名次，并列同名次（竞赛排名法 1,1,3）。

    参数:
        scores: 一维得分，越大越好。

    返回:
        形状 (n,) 的 int 名次数组。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        用 ``mergesort`` 稳定排序 + 1e-12 的并列容差；若得分是"越小越好"的量，
        调用方必须先取负，否则整张排序表会反。
    """
    s = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(s.size, dtype=int)
    cur = 1
    for pos, idx in enumerate(order):
        if pos > 0 and abs(s[order[pos]] - s[order[pos - 1]]) > _EPS:
            cur = pos + 1
        ranks[idx] = cur
    return ranks


def _average_ranks(scores: ArrayLike) -> np.ndarray:
    """按"1 为最好"给并列取**平均秩**（1.5, 1.5, 3 口径）。

    参数:
        scores: 一维得分，越大越好。

    返回:
        形状 (n,) 的 float 名次数组，恰好是 1..n 的一个排列的平均秩版本，
        因此恒有 ``mean(返回值) == (n + 1) / 2``。

    算法:
        稳定降序排序后，把每一段容差内相等的元素整体赋成该段位次的算术平均
        ``(first + last) / 2``（first、last 都是 1 起的竞赛位次）。

    复杂度:
        时间 O(n log n) / 空间 O(n)。

    陷阱:
        与 ``_ranks_from_scores`` 的**竞赛排名法**（1,1,3）不同：竞赛法让并列组的所有成员
        都拿到该组最好的名次，会系统性抬高并列组的秩；秩和比 RSR 的教材口径是平均秩，
        所以 ``rank_sum_ratio`` 用本函数而不是 ``_ranks_from_scores``。
    """
    s = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        k = i
        while k + 1 < s.size and abs(s[order[k + 1]] - s[order[i]]) <= _EPS:
            k += 1
        avg = 0.5 * ((i + 1) + (k + 1))
        for pos in range(i, k + 1):
            ranks[order[pos]] = avg
        i = k + 1
    return ranks


def _as_per_criterion(value, n: int, default: ArrayLike) -> np.ndarray:
    """把标量或序列形式的阈值参数扩成长度 n 的数组。

    参数:
        value: None（用 default）/ 标量 / 长度 n 的序列。
        n: 指标个数。
        default: 缺省值（长度 n 的数组）。

    返回:
        形状 (n,) 的 float 数组。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        长度既不是 1 也不是 n 的序列会被拒绝；长度 1 的序列按"所有指标同阈值"广播，
        这是刻意的，因为 ELECTRE 里"所有指标阈值相同"是常见简化。
    """
    if value is None:
        return np.asarray(default, dtype=float).reshape(-1).copy()
    arr = as_vector(value, "阈值参数")
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"阈值参数长度 {arr.size} 与指标数 {n} 不一致")
    return arr.copy()


def _level_thresholds(value, base: float, delta: float, name: str,
                      weaker_is_larger: bool) -> Tuple[float, float]:
    """把 ELECTRE II 的"强/弱两级阈值"解析成 ``(strong, weak)`` 二元组。

    参数:
        value: None（用 base/delta 推默认值）/ 长度 1 或 2 的序列；长度 2 时按
            ``(强, 弱)`` 顺序给出。
        base: 默认值的基准（这里取矩阵非对角均值）。
        delta: 默认值的级差（强阈值与弱阈值之差，恒取正数）。
        name: 出错信息里显示的名字。
        weaker_is_larger: True 表示"弱阈值 > 强阈值"（不一致性矩阵的方向）；
            False 表示"弱阈值 < 强阈值"（一致性矩阵的方向）。

    返回:
        ``(strong, weak)`` 两个 float，均落在 [0, 1]。

    算法:
        用户给定序列时直接用（越界即拒绝）；否则强阈值取基准值 base，弱阈值在 base 上按方向
        偏移 delta，再**裁到** [0, 1] 内——因为 d 的均值加 0.1 可能超过 1，裁剪是给缺省值
        兜底，不是给用户输入兜底。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        默认的 ±delta（0.1）是**约定**而不是定理：均值附近挪 0.1 只是让两级关系在默认参数下
        真的不同，从而能看出强弱区分；论文里必须显式写出所用阈值并做敏感性分析。
        另外强阈值一定不弱于弱阈值：一致性方向要求 ``strong >= weak``，不一致性方向要求
        ``strong <= weak``，违反顺序的输入会被拒绝，而不是被静默交换。
        用户显式传入的阈值**不会**被裁剪：``1.5`` 这种越界值直接报 ValueError，否则"我明明
        设了 1.5 的阈值"和"结果等价于 1.0"会被静默混淆。
    """
    if value is None:
        strong = min(max(float(base), 0.0), 1.0)
        weak = float(base) + (float(delta) if weaker_is_larger else -float(delta))
        weak = min(max(weak, 0.0), 1.0)
    else:
        arr = as_vector(value, name)
        if arr.size not in (1, 2):
            raise ValueError(f"{name} 必须是长度 1 或 2 的序列（强/弱两级阈值），"
                             f"得到长度 {arr.size}")
        strong = float(arr[0])
        weak = float(arr[-1])
        if not (0.0 <= strong <= 1.0) or not (0.0 <= weak <= 1.0):
            raise ValueError(f"{name} 的两个阈值都必须落在 [0, 1]，得到 {strong} 与 {weak}")
    if not weaker_is_larger and strong < weak - _EPS:
        raise ValueError(f"{name} 的强阈值必须不小于弱阈值，得到 strong={strong} weak={weak}")
    if weaker_is_larger and strong > weak + _EPS:
        raise ValueError(f"{name} 的强阈值必须不大于弱阈值，得到 strong={strong} weak={weak}")
    return strong, weak


def _preference_degree(kind: str, d: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """PROMETHEE 偏好函数 H(d)，d 为"方案 a 相对 b 的优势"（>0 表示 a 更好）。

    参数:
        kind: ``"usual"`` / ``"linear"`` / ``"level"`` / ``"gaussian"``。
        d: 形状 (n,) 的逐指标优势。
        p: 形状 (n,) 的偏好阈值。
        q: 形状 (n,) 的无差异阈值。

    返回:
        形状 (n,) 的偏好度，取值 [0, 1]。

    复杂度:
        时间 O(n) / 空间 O(n)。

    陷阱:
        ``linear`` 用 ``clip((d-q)/(p-q), 0, 1)`` 一次性处理三段，但前提是 ``p > q``；
        ``gaussian`` 的 ``p`` 是标准差而不是阈值，所以它**不**使用 ``q``。
    """
    d = np.asarray(d, dtype=float)
    if kind == "usual":
        return np.where(d > _EPS, 1.0, 0.0)
    if kind == "linear":
        return np.clip((d - q) / (p - q), 0.0, 1.0)
    if kind == "level":
        return np.where(d <= q, 0.0, np.where(d <= p, 0.5, 1.0))
    if kind == "gaussian":
        return np.where(d > _EPS, 1.0 - np.exp(-(d ** 2) / (2.0 * p * p)), 0.0)
    raise ValueError(f"不支持的偏好函数 preference={kind!r}，"
                     f"可选 'usual'/'linear'/'level'/'gaussian'")


def _safe_norm(x: np.ndarray) -> np.ndarray:
    """L2 归一化，范数为 0 的列原样保留为 0（不产生 NaN）。

    参数:
        x: 形状 (m, n) 的矩阵。

    返回:
        同形状矩阵。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        某列全部为 0（该指标对所有方案取值相同）时，归一化后该列仍是 0。注意它并非"被忽略"：
        下游的一致性判断用 ``V[a, j] >= V[b, j]``，``0 >= 0`` 成立，所以这个无信息指标对**每一
        对方案都被算作一致**，等于给所有方案对一致性地"平白加分"。要真正剔除它，应先按权重
        或方差把该列删掉，或把权重设为 0（本模块的 ELECTRE 都按此口径计算）。
    """
    norm = np.sqrt((x ** 2).sum(axis=0, keepdims=True))
    return np.divide(x, norm, out=np.zeros_like(x, dtype=float), where=norm > _EPS)


def _kernel(relation: np.ndarray) -> List[int]:
    """用贪心构造 ELECTRE I 的内核 K。

    参数:
        relation: 形状 (m, m) 的 0/1 支配矩阵，``relation[a, b] == 1`` 表示 a 级别高于 b。

    返回:
        升序排列的内核下标列表。

    复杂度:
        时间 O(m^2) / 空间 O(m)。

    陷阱:
        内核在一般情形下存在且唯一，但数值边界（一致性刚好等于阈值）会破坏存在性；这里的
        贪心构造在"非严格内核"时仍返回一个确定的集合，不要把它当成严格意义上的内核。
    """
    m = relation.shape[0]
    kernel: List[int] = []
    for a in range(m):
        if any(relation[k, a] == 1 for k in kernel):
            continue
        kernel = [k for k in kernel if relation[a, k] != 1]
        kernel.append(a)
    return sorted(kernel)


def _distill(credibility: np.ndarray, lam: float, severity: float,
             descending: bool) -> List[List[int]]:
    """ELECTRE III 的蒸馏（定性分档）。

    参数:
        credibility: 形状 (m, m) 的可信度矩阵 S。
        lam: 级别高于关系的 λ 切分阈值。
        severity: 显著性阈值 s(λ)，只有 ``S(a,b) > S(b,a) + s(λ)`` 才算真正级别高于。
        descending: True 做降序蒸馏（从最好开始），False 做升序蒸馏（从最差开始）。

    返回:
        list of list，每个子列表是一个并列组；降序时第一个组最好，升序时第一个组最差。

    复杂度:
        时间 O(m^3)（最坏情况每轮只挑出一个方案）/ 空间 O(m)。

    陷阱:
        若某一轮"极大集"为空（可信度矩阵过于扁平、所有方案互相以超过 s(λ) 的优势压制），
        这里把剩余方案整体作为一组，避免死循环——出现这种情况说明 λ 取得过大或
        p/q/v 阈值本身自相矛盾，应当在论文里说明，而不是假装蒸馏成功。
    """
    m = credibility.shape[0]
    remaining = list(range(m))
    groups: List[List[int]] = []
    while remaining:
        chosen: List[int] = []
        for a in remaining:
            dominated = False
            for b in remaining:
                if b == a:
                    continue
                if descending:
                    strong = (credibility[b, a] >= lam - _EPS
                              and credibility[b, a] > credibility[a, b] + severity)
                else:
                    strong = (credibility[a, b] >= lam - _EPS
                              and credibility[a, b] > credibility[b, a] + severity)
                if strong:
                    dominated = True
                    break
            if not dominated:
                chosen.append(a)
        if not chosen:
            chosen = list(remaining)
        groups.append(sorted(chosen))
        chosen_set = set(chosen)
        remaining = [x for x in remaining if x not in chosen_set]
    return groups


def _distill_relation(strong: np.ndarray, weak: np.ndarray,
                      descending: bool) -> List[List[int]]:
    """ELECTRE II 的蒸馏：只在两张 **0/1 关系矩阵**上挑极大/极小集。

    参数:
        strong: 形状 (m, m) 的强级别高于关系，``strong[a, b] = 1`` 表示 a 强级别高于 b。
        weak: 形状 (m, m) 的弱级别高于关系（**含**强关系），用于打破强关系里的并列。
        descending: True 做降序蒸馏（从最好开始），False 做升序蒸馏（从最差开始）。

    返回:
        list of list，每个子列表是一个并列组；降序时第一个组最好，升序时第一个组最差。

    算法:
        1. 取剩余方案里强关系下的极大集（降序，没有任何方案强级别高于它）或极小集（升序）。
        2. 若极大集不唯一，就在**极大集内部**反复用弱关系压缩：只保留"没有被同层其他方案
           弱级别高于"的那些方案，直到只剩一个或不再缩小为止（Roy 的 A1 ⊃ A2 ⊃ ... 序列）。
        3. 把压缩结果作为一组取出，从剩余集中删掉，回到第 1 步。

    复杂度:
        时间 O(m³)（最坏情况每轮只挑出一个方案，每轮 O(m²)）/ 空间 O(m)。

    陷阱:
        - 强关系是**不传递**的：极大集变了以后弱关系要在新集合上重算，不能沿用全局弱关系
          的极大集。
        - 若强关系下极大集为空（关系成环），这里把剩余方案整体作为一组，避免死循环；出现这
          种情况说明阈值设定让关系高度循环，应在论文里说明，而不是假装蒸馏成功。
        - 弱关系只用来打破并列，**不改变分组顺序**：分组完全由强关系决定。
    """
    m = strong.shape[0]
    remaining = list(range(m))
    groups: List[List[int]] = []
    while remaining:
        cand: List[int] = []
        for a in remaining:
            dominated = False
            for b in remaining:
                if b == a:
                    continue
                if (strong[b, a] if descending else strong[a, b]):
                    dominated = True
                    break
            if not dominated:
                cand.append(a)
        if not cand:
            cand = list(remaining)

        layer = cand
        while len(layer) > 1:
            nxt: List[int] = []
            for a in layer:
                beaten = False
                for b in layer:
                    if b == a:
                        continue
                    if (weak[b, a] if descending else weak[a, b]):
                        beaten = True
                        break
                if not beaten:
                    nxt.append(a)
            if not nxt or len(nxt) >= len(layer):
                break
            layer = nxt

        groups.append(sorted(layer))
        chosen = set(layer)
        remaining = [x for x in remaining if x not in chosen]
    return groups


def _norm_ppf(p: float) -> float:
    """标准正态分布的分位数函数（probit），Acklam 有理逼近 + 一步 Halley 修正。

    参数:
        p: 概率，必须满足 ``0 < p < 1``。

    返回:
        float，满足 ``P(Z <= x) = p``。精度**分段的**：``p ∈ [1e-4, 1 - 1e-4]`` 时与精确值
        相差约 1e-15（机器精度量级）；越靠近尾部 Halley 一步修正越吃紧，``p = 1e-8`` 时约
        4e-10、``p = 1e-12`` 时约 7e-9，但仍稳定优于 1e-8。

    算法:
        1. 分三段使用 Acklam (2003) 的有理逼近：``p < 0.02425`` 用左尾渐近式，
           ``p > 0.97575`` 用右尾（对称），中间段用一个 5/5 有理式。
        2. 用 ``e = 0.5 * erfc(-x / sqrt(2)) - p`` 做一步 Halley 迭代
           ``x <- x - u / (1 + x u / 2)``：中间段压到 ~1e-15，极端尾部压到 1e-9 以内。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        只支持开区间 ``(0, 1)``：``p = 0`` 或 ``1`` 对应 ±inf，RSR 分档时如果某组的累计
        频率恰好为 0 或 1，必须先做中位秩修正 ``(位次 - 0.5) / n`` 再用，否则会拿到 inf
        并污染整条回归线。
        精度在极端尾部会退化（见"返回"）：需要 ``p < 1e-10`` 量级的分位数时，应改用
        ``scipy.special.ndtri`` 之类的库函数，而不是指望这里的逼近。
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"_norm_ppf 要求 0 < p < 1，得到 p={p!r}")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 0.97575

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # 一步 Halley 修正（Acklam 建议的收尾）：中间段到 ~1e-15，极端尾部到 1e-9 以内
    e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return float(x - u / (1.0 + x * u / 2.0))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman 秩相关（用秩向量的 Pearson 相关实现，天然支持并列）。

    参数:
        x: 第一份"秩向量"（已编好秩，越小越好或越大越好都可以，只要两份口径一致；
            内部会再取一次平均秩，所以带并列、甚至直接传原始得分都可以）。
        y: 第二份秩向量，长度与 x 相同。

    返回:
        float，取值 [-1, 1]；任一份秩向量为常数时返回 0.0。

    算法:
        1. 先把两份输入各自换成**平均秩**（``_average_ranks``，并列组取组内平均位次）；
        2. rho = cov(rx, ry) / (sd(rx) sd(ry))，即平均秩向量上的 Pearson 相关。在无并列的
           全序情形下等价于闭式 ``1 - 6 Σd^2 / (m (m^2 - 1))``（自测里对拍验证）。

    复杂度:
        时间 O(m log m) / 空间 O(m)。

    陷阱:
        不要用闭式 ``1 - 6Σd²/(m(m²-1))``：它假定**没有并列**，一旦出现并列名次就会高估
        相关性（分母没有做并列修正）。这里用"平均秩 + Pearson"版本：输入里带不带并列都
        给出与 ``scipy.stats.spearmanr`` 一致的结果（自测对拍）。
        注意平均秩的方向约定（越大越好）在这里无关紧要：两份向量做同一种单调变换，
        Pearson 相关系数不变。
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"两份秩向量长度不一致：{a.size} vs {b.size}")
    if a.size < 2:
        raise ValueError("计算 Spearman 至少需要 2 个共同方案")
    a = _average_ranks(a)
    b = _average_ranks(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    if denom <= _EPS:
        return 0.0
    return float((a * b).sum() / denom)


def _kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall 秩相关 tau-b（带并列修正），只用于自测里比较两组名次的一致性。

    参数:
        x: 第一份名次或得分向量。
        y: 第二份名次或得分向量，长度与 x 相同。

    返回:
        float，取值 [-1, 1]；分母为 0（某一维几乎全部并列）时返回 0.0。

    算法:
        逐对枚举 (i, j)，把"同序/逆序/只在 x 上并列/只在 y 上并列"四类计数分别记为
        C、D、T、U，则 ``tau_b = (C - D) / sqrt((C + D + T)(C + D + U))``。

    复杂度:
        时间 O(m²) / 空间 O(1)。

    陷阱:
        分母用的是 tau-b 的并列修正项；若直接写 ``tau = (C - D) / (C + D)``，遇到并列名次
        （竞赛排名法必然产生并列）会高估一致性。另外这里的并列判定带 1e-12 容差，与模块内
        其它"并列"口径一致。
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"两份向量长度不一致：{a.size} vs {b.size}")
    if a.size < 2:
        raise ValueError("计算 Kendall tau 至少需要 2 个共同方案")
    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0
    for i in range(a.size):
        for j in range(i + 1, a.size):
            dx = a[i] - a[j]
            dy = b[i] - b[j]
            x_tied = abs(dx) <= _EPS
            y_tied = abs(dy) <= _EPS
            if x_tied and y_tied:
                continue
            if x_tied:
                tie_x += 1
            elif y_tied:
                tie_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(float(concordant + discordant + tie_x)
                      * float(concordant + discordant + tie_y))
    if denom <= _EPS:
        return 0.0
    return float((concordant - discordant) / denom)


def _parse_rankings(rankings) -> Tuple[List[List[int]], int]:
    """把 rankings 统一成"方案下标列表的列表"，并返回方案总数 m。

    参数:
        rankings: list of list，每项是一份从最优到最差的方案下标排序。

    返回:
        (parsed, m)：parsed 为整数下标列表的列表；``m = max(下标) + 1``。

    复杂度:
        时间 O(K·L) / 空间 O(K·L)。

    陷阱:
        只校验"同一份排名内下标不重复"，**不要求**覆盖全部方案：缺项在 Borda 里按截断选票
        记 0 分、在 Spearman 里被剔除。重复下标会被拒绝，因为"同一方案排两次"没有定义。
    """
    if not isinstance(rankings, (list, tuple)) or len(rankings) == 0:
        raise ValueError("rankings 必须是非空的『排名列表』（list of list）")
    parsed: List[List[int]] = []
    m = 0
    for k, raw in enumerate(rankings):
        arr = as_vector(raw, f"rankings[{k}]")
        if np.any(np.abs(arr - np.round(arr)) > _EPS):
            raise ValueError(f"rankings[{k}] 含非整数方案下标：{arr.tolist()}")
        idx = [int(v) for v in arr]
        if min(idx) < 0:
            raise ValueError(f"rankings[{k}] 含负下标：{idx}")
        if len(set(idx)) != len(idx):
            raise ValueError(f"rankings[{k}] 含重复方案下标：{idx}")
        parsed.append(idx)
        m = max(m, max(idx) + 1)
    return parsed, m


# --------------------------------------------------------------------------
# PROMETHEE II
# --------------------------------------------------------------------------

def promethee_ii(X: MatrixLike, weights: ArrayLike, benefit: Optional[Sequence[bool]] = None,
                 preference: str = "linear", p: Optional[ArrayLike] = None,
                 q: Optional[ArrayLike] = None) -> Dict[str, object]:
    """PROMETHEE II：用偏好函数构造逐对偏好度，再按净流排序。

    参数:
        X: 决策矩阵，形状 (m, n)，m 个方案、n 个指标。
        weights: 长度 n 的指标权重（非负，内部归一化到和为 1）。
        benefit: 长度 n 的 bool 方向序列，True 越大越好；None 表示全部正向。
        preference: 偏好函数类型，``"usual"``（0/1 型）、``"linear"``（线性型）、
            ``"level"``（阶梯型）、``"gaussian"``（高斯型，p 为标准差）。
        p: 偏好阈值（长度 n 或标量）；None 时取该指标正向化后的极差（极差为 0 时取 1.0）。
        q: 无差异阈值（长度 n 或标量）；None 时取 0。``"usual"`` 下 p、q 被忽略。

    返回:
        dict，键为：
        ``phi_plus``   形状 (m,) 的正流，越大越好；
        ``phi_minus``  形状 (m,) 的负流，越小越好；
        ``phi_net``    形状 (m,) 的净流 = phi_plus - phi_minus，恒有 sum(phi_net) == 0；
        ``rank``       形状 (m,) 的 int 名次（1 为最好，按 phi_net 降序）；
        ``pairwise``   形状 (m, m) 的偏好度矩阵 pi，``pi[a, b] = Σ_j w_j H_j(Z_a - Z_b)``。

    算法:
        1. 正向化 ``Z = X``（成本型列取负），使所有指标都变成"越大越好"。
        2. 对每个有序对 (a, b) 与指标 j，算优势 ``d = Z[a, j] - Z[b, j]``，套用偏好函数
           H_j(d) 得偏好度（usual: 1{d>0}；linear: clip((d-q)/(p-q), 0, 1)；
           level: 0 / 0.5 / 1 三段；gaussian: 1 - exp(-d²/(2p²))）。
        3. 加权汇总成 ``pi[a, b] = Σ_j w_j H_j``，再算
           ``phi_plus[a] = Σ_{b≠a} pi[a,b] / (m-1)``、
           ``phi_minus[a] = Σ_{b≠a} pi[b,a] / (m-1)``，按 phi_net 降序给名次。

    复杂度:
        时间 O(m² n) / 空间 O(m² + mn)。

    陷阱:
        - ``p`` 的默认值是该指标的极差，**带量纲**：若直接把不同量纲的指标混在一起且不显式
          给 p，等价于给每个指标一个"相对极差"阈值，这与先归一化再给固定 p 的结果不同。
        - ``"usual"`` 偏好函数对任何微小差异都给满偏好，噪声大的指标会主导排序；此时应改用
          ``"linear"`` 或 ``"gaussian"``。
        - phi_net 之和恒为 0 是结构性质（是自测里的对拍点），但它**不代表**名次对称。

    参考:
        Brans & Vincke (1985) PROMETHEE；Brans, Vincke & Mareschal (1986) 六种偏好函数。
    """
    Xm = as_matrix(X, "X")
    m, n = Xm.shape
    if m < 2:
        raise ValueError(f"PROMETHEE 至少需要 2 个方案，得到 m={m}")
    if preference not in ("usual", "linear", "level", "gaussian"):
        raise ValueError(f"不支持的偏好函数 preference={preference!r}，"
                         f"可选 'usual'/'linear'/'level'/'gaussian'")

    w = _normalize_weights(weights, n)
    ben = _resolve_benefit(benefit, n)
    Z = _oriented(Xm, ben)

    span = Z.max(axis=0) - Z.min(axis=0)
    default_p = np.where(span > _EPS, span, 1.0)
    if preference == "usual":
        p_crit = np.ones(n)
        q_crit = np.zeros(n)
    else:
        p_crit = _as_per_criterion(p, n, default_p)
        q_crit = _as_per_criterion(q, n, np.zeros(n))
        if np.any(p_crit <= 0.0):
            raise ValueError(f"偏好阈值 p 必须为正，得到 {p_crit.tolist()}")
        if np.any(q_crit < 0.0):
            raise ValueError(f"无差异阈值 q 不能为负，得到 {q_crit.tolist()}")
        if preference in ("linear", "level") and np.any(q_crit >= p_crit):
            raise ValueError(f"需要 0 <= q < p，得到 q={q_crit.tolist()} p={p_crit.tolist()}")

    pairwise = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            h = _preference_degree(preference, Z[a] - Z[b], p_crit, q_crit)
            pairwise[a, b] = float(np.dot(w, h))

    denom = float(m - 1)
    phi_plus = pairwise.sum(axis=1) / denom
    phi_minus = pairwise.sum(axis=0) / denom
    phi_net = phi_plus - phi_minus
    rank = _ranks_from_scores(phi_net)

    return {"phi_plus": phi_plus, "phi_minus": phi_minus, "phi_net": phi_net,
            "rank": rank, "pairwise": pairwise}


# --------------------------------------------------------------------------
# ELECTRE I
# --------------------------------------------------------------------------

def electre_i(X: MatrixLike, weights: ArrayLike, benefit: Optional[Sequence[bool]] = None,
              c_threshold: Optional[float] = None,
              d_threshold: Optional[float] = None) -> Dict[str, object]:
    """ELECTRE I：一致性/不一致性矩阵 + 强支配关系 + 内核。

    参数:
        X: 决策矩阵，形状 (m, n)。
        weights: 长度 n 的指标权重。
        benefit: 长度 n 的方向序列；None 表示全部正向。
        c_threshold: 一致性阈值（要求 C(a,b) >= c_threshold）；None 取非对角一致性均值。
        d_threshold: 不一致性阈值（要求 D(a,b) <= d_threshold）；None 取非对角不一致性均值。

    返回:
        dict，键为：
        ``concordance``  形状 (m, m) 的一致性矩阵，对角为 1；
        ``discordance``  形状 (m, m) 的不一致性矩阵，对角为 0；
        ``kernel``       ELECTRE I 内核（list of int，贪心构造）；
        ``relation``     形状 (m, m) 的 0/1 支配矩阵，``relation[a, b] = 1`` 表示 a 级别高于 b。

    算法:
        1. 正向化 ``Z``（成本型取负），再做向量归一化 ``r_ij = z_ij / sqrt(Σ_i z_ij²)``。
        2. 一致性：``C(a,b) = Σ_{j: r_aj >= r_bj} w_j``（权重已归一化，故 C ∈ [0,1]）。
        3. 不一致性：``D(a,b) = max_{j: r_aj < r_bj} (r_bj - r_aj) / (max_i r_ij - min_i r_ij)``；
           若 a 在所有指标上都不劣于 b，则 D(a,b) = 0。
        4. ``a S b`` 当且仅当 ``C(a,b) >= c_threshold`` 且 ``D(a,b) <= d_threshold``。
        5. 内核：按方案下标顺序贪心——若 a 已被内核中某个方案级别高于则跳过，否则把 a 加入
           内核并删掉内核中被 a 级别高于的方案。

    复杂度:
        时间 O(m² n) / 空间 O(m² + mn)。

    陷阱:
        - 不一致性用**逐指标极差**做分母：某个指标若对所有方案几乎相同（极差≈0），它一旦
          进入不一致性集合就会被放大成接近 1 的值（这里把极差 0 的列按分母 1.0 处理，等于
          忽略该列）。这就是 ELECTRE 对"无区分度指标"非常敏感的原因。
        - 阈值缺省取矩阵均值，会让结论与方案集规模强相关；正式分析里应给出阈值敏感性区间。
        - 一致性用 ``>=``（并列算"不劣"），所以完全相同方案之间的 C 为 1，会互相"级别高于"。

    参考:
        Roy (1968) ELECTRE I；Figueira, Greco & Ehrgott (2005) Multiple Criteria Decision Analysis。
    """
    Xm = as_matrix(X, "X")
    m, n = Xm.shape
    if m < 2:
        raise ValueError(f"ELECTRE I 至少需要 2 个方案，得到 m={m}")

    w = _normalize_weights(weights, n)
    ben = _resolve_benefit(benefit, n)
    Z = _oriented(Xm, ben)
    R = _safe_norm(Z)
    span_r = R.max(axis=0) - R.min(axis=0)
    denom = np.where(span_r > _EPS, span_r, 1.0)

    concordance = np.ones((m, m))
    discordance = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            not_worse = R[a] >= R[b]
            concordance[a, b] = float(w[not_worse].sum())
            worse = ~not_worse
            if np.any(worse):
                discordance[a, b] = float(((R[b] - R[a])[worse] / denom[worse]).max())

    off = ~np.eye(m, dtype=bool)
    c_thr = float(concordance[off].mean()) if c_threshold is None else float(c_threshold)
    d_thr = float(discordance[off].mean()) if d_threshold is None else float(d_threshold)
    if not (0.0 <= c_thr <= 1.0):
        raise ValueError(f"一致性阈值 c_threshold 应在 [0, 1]，得到 {c_thr}")
    if d_thr < 0.0:
        raise ValueError(f"不一致性阈值 d_threshold 不能为负，得到 {d_thr}")

    relation = ((concordance >= c_thr - _EPS) & (discordance <= d_thr + _EPS))
    np.fill_diagonal(relation, False)
    relation_int = relation.astype(int)

    return {"concordance": concordance, "discordance": discordance,
            "kernel": _kernel(relation_int), "relation": relation_int}


# --------------------------------------------------------------------------
# ELECTRE III
# --------------------------------------------------------------------------

def electre_iii(X: MatrixLike, weights: ArrayLike, benefit: Optional[Sequence[bool]] = None,
                p: Optional[ArrayLike] = None, q: Optional[ArrayLike] = None,
                v: Optional[ArrayLike] = None) -> Dict[str, object]:
    """ELECTRE III：带无差异/偏好/否决阈值的可信度矩阵 + 升/降蒸馏排序。

    参数:
        X: 决策矩阵，形状 (m, n)。
        weights: 长度 n 的指标权重。
        benefit: 长度 n 的方向序列；None 表示全部正向。
        p: 偏好阈值（长度 n 或标量）；None 取该指标正向化后的极差。
        q: 无差异阈值；None 取 0。
        v: 否决阈值；None 取极差的 2 倍。要求 ``0 <= q < p < v``。

    返回:
        dict，键为：
        ``rank``        形状 (m,) 的 int 最终名次（1 最好，为升降两种蒸馏名次的平均）；
        ``credibility`` 形状 (m, m) 的可信度矩阵 S，对角为 1；
        ``ascending``   形状 (m,) 的升序蒸馏名次（从最差开始蒸馏，1 最好）；
        ``descending``  形状 (m,) 的降序蒸馏名次（从最好开始蒸馏，1 最好）。

    算法:
        1. 正向化 Z（成本型取负），对每对 (a, b) 算逐指标劣势 ``d_j = Z[b,j] - Z[a,j]``。
        2. 局部一致性：``c_j = 1``（d <= q）、``(p - d)/(p - q)``（q < d < p）、``0``（d >= p）。
        3. 全局一致性 ``C(a,b) = Σ w_j c_j``；局部不一致性 ``d_j = 0``（d <= p 或 c_j = 1）、
           ``(d - p)/(v - p)``（p < d < v）、``1``（d >= v）。
        4. 可信度 ``S(a,b) = C(a,b) · Π_j (1 - d_j)/(1 - C(a,b))``，只对满足
           ``d_j > C(a,b)`` 的指标连乘（否决效应）。
        5. λ 取非对角可信度均值，显著性 ``s(λ) = 0.3 - 0.15λ``；降序蒸馏反复挑"没有被其余
           方案以超过 s(λ) 的优势压制"的方案构成并列组，升序蒸馏对称地从最差往上挑。
        6. 最终名次 = 升序名次与降序名次的平均，再按"越小越好"编秩（Roy 的中位数/均值折中）。

    复杂度:
        时间 O(m³ + m² n) / 空间 O(m² + mn)。

    陷阱:
        - **λ 的取法没有唯一标准**：Roy 原文里 λ 由决策者给定，这里为了接口自洽取非对角
          可信度均值并在返回值里可通过 credibility 自行复算；换 λ 会改变分档，结论必须做敏感性分析。
        - 否决阈值 v 一旦被越过，可信度会被单个指标直接压到接近 0，排序可能被一个指标"一票否决"；
          这正是 ELECTRE III 的设计意图，但在指标有量纲差异时非常容易误伤。
        - 蒸馏在可信度矩阵过于扁平时会退化成"所有方案一组并列"（``_distill`` 里显式兜底），
          出现这种情况时 rank 全为 1，不要误读成"所有方案等价"。

    参考:
        Roy (1978) ELECTRE III；Figueira, Greco & Ehrgott (2005) 第 5 章。
    """
    Xm = as_matrix(X, "X")
    m, n = Xm.shape
    if m < 2:
        raise ValueError(f"ELECTRE III 至少需要 2 个方案，得到 m={m}")

    w = _normalize_weights(weights, n)
    ben = _resolve_benefit(benefit, n)
    Z = _oriented(Xm, ben)

    span = Z.max(axis=0) - Z.min(axis=0)
    default_span = np.where(span > _EPS, span, 1.0)
    q_crit = _as_per_criterion(q, n, np.zeros(n))
    p_crit = _as_per_criterion(p, n, default_span)
    v_crit = _as_per_criterion(v, n, 2.0 * default_span)
    if np.any(q_crit < 0.0):
        raise ValueError(f"无差异阈值 q 不能为负，得到 {q_crit.tolist()}")
    if np.any(p_crit <= q_crit):
        raise ValueError(f"需要 p > q，得到 p={p_crit.tolist()} q={q_crit.tolist()}")
    if np.any(v_crit <= p_crit):
        raise ValueError(f"需要 v > p，得到 v={v_crit.tolist()} p={p_crit.tolist()}")

    concordance = np.ones((m, m))
    credibility = np.ones((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            d = Z[b] - Z[a]
            c_j = np.where(d <= q_crit, 1.0,
                           np.where(d >= p_crit, 0.0, (p_crit - d) / (p_crit - q_crit)))
            c_ab = float(np.dot(w, c_j))
            concordance[a, b] = c_ab

            d_j = np.where(c_j >= 1.0 - _EPS, 0.0,
                           np.where(d <= p_crit, 0.0,
                                    np.where(d >= v_crit, 1.0,
                                             (d - p_crit) / (v_crit - p_crit))))
            s_ab = c_ab
            for j in range(n):
                if d_j[j] > c_ab + _EPS:
                    s_ab *= (1.0 - d_j[j]) / (1.0 - c_ab)
            credibility[a, b] = float(min(max(s_ab, 0.0), 1.0))

    off = ~np.eye(m, dtype=bool)
    lam = float(credibility[off].mean())
    severity = max(0.0, 0.3 - 0.15 * lam)

    desc_groups = _distill(credibility, lam, severity, descending=True)
    asc_groups = _distill(credibility, lam, severity, descending=False)

    descending = np.empty(m, dtype=int)
    for gi, group in enumerate(desc_groups):
        for a in group:
            descending[a] = gi + 1
    n_groups = len(asc_groups)
    ascending = np.empty(m, dtype=int)
    for gi, group in enumerate(asc_groups):
        for a in group:
            ascending[a] = n_groups - gi

    avg = (descending.astype(float) + ascending.astype(float)) / 2.0
    rank = _ranks_from_scores(-avg)

    return {"rank": rank, "credibility": credibility,
            "ascending": ascending, "descending": descending}


# --------------------------------------------------------------------------
# ELECTRE II
# --------------------------------------------------------------------------

def electre_ii(X: MatrixLike, weights: ArrayLike, benefit: Optional[Sequence[bool]] = None,
               c_thresholds: Optional[ArrayLike] = None,
               d_thresholds: Optional[ArrayLike] = None,
               directions: Optional[Sequence[bool]] = None) -> Dict[str, object]:
    """ELECTRE II：双阈值（强/弱两级）级别高于关系 + 升降蒸馏排序。

    参数:
        X: 决策矩阵，形状 (m, n)。
        weights: 长度 n 的指标权重（非负，内部归一化到和为 1）。
        benefit: 长度 n 的方向序列；None 表示全部正向。
        c_thresholds: 一致性阈值，长度 1（强=弱，退化成单阈值）或长度 2（按 ``[强, 弱]``
            顺序）；None 取非对角一致性均值作强阈值、再减 0.1 作弱阈值。要求
            ``0 <= 弱 <= 强 <= 1``。
        d_thresholds: 不一致性阈值，长度 1 或 2（按 ``[强, 弱]`` 顺序）；None 取非对角不一致性
            均值作强阈值、再加 0.1 作弱阈值。要求 ``0 <= 强 <= 弱 <= 1``。
        directions: ``benefit`` 的别名（其它 MCDM 库里常叫 directions）；两个都传会被拒绝。

    返回:
        dict，键为：
        ``concordance``       形状 (m, m) 的一致性矩阵 C，对角为 1；
        ``discordance``       形状 (m, m) 的不一致性矩阵 D，对角为 0；
        ``normalized``        形状 (m, n) 的向量归一化矩阵 R；
        ``weighted``          形状 (m, n) 的加权矩阵 ``V = R * w``（C、D 就是在 V 上算的）；
        ``thresholds``        dict，四个阈值 ``c_strong`` / ``c_weak`` / ``d_strong`` / ``d_weak``；
        ``strong_matrix``     形状 (m, m) 的 0/1 强级别高于矩阵，对角为 0；
        ``weak_matrix``       形状 (m, m) 的 0/1 **弱或强**级别高于矩阵，对角为 0；
        ``strong_relations``  list，强关系有序对 ``[a, b]``（a 强级别高于 b），按字典序；
        ``weak_relations``    list，**仅弱**（非强）关系的有序对 ``[a, b]``，按字典序；
        ``incomparable``      list，无序对 ``[a, b]``（a < b），两个方向都够不上弱关系；
        ``descending``        形状 (m,) 的降序蒸馏名次（1 最好，并列同名次）；
        ``ascending``         形状 (m,) 的升序蒸馏名次（1 最好，并列同名次）；
        ``rank``              形状 (m,) 的最终名次 = 升降两侧蒸馏名次的平均后再编秩（1 最好）；
        ``ranking``           list of int，最终排序（从最好到最差；名次并列时按下标升序打破平局）；
        ``distillation_levels`` dict，``{"descending": 分组, "ascending": 分组}``，
            每个分组是 list of list（降序时第一个组最好，升序时第一个组最差）。

    算法:
        1. 正向化 ``Z``（成本型取负），向量归一化 ``R``，加权 ``V = R * w``。
        2. 一致性：``C(a,b) = Σ_{j: V[a,j] >= V[b,j]} w_j``；不一致性：
           ``D(a,b) = max_{j: V[a,j] < V[b,j]} (V[b,j] - V[a,j]) / (max_i V[i,j] - min_i V[i,j])``，
           若 a 在所有指标上都不劣于 b 则 ``D(a,b) = 0``。
           （V 的逐列极差 = w_j × R 的逐列极差，故权重只在 C 里以加权和的形式起作用，
           在 D 里自动约掉；两种写法在全正权重下完全等价。）
        3. 强关系 ``a S b ⟺ C(a,b) >= c_strong 且 D(a,b) <= d_strong``；
           弱关系 ``a W b ⟺ C(a,b) >= c_weak 且 D(a,b) <= d_weak``（弱关系**包含**强关系，
           因为 c_weak <= c_strong、d_weak >= d_strong）。
        4. 降序蒸馏：反复取剩余集合中强关系下的极大集（没有任何方案强级别高于它）；若极大集
           不唯一，则在该集合内用**弱关系**继续压缩（只保留没有被同层其他方案弱级别高于的
           方案），直到唯一或不再缩小，作为一组取出。升序蒸馏对称地取极小集。
        5. 最终名次取两侧蒸馏名次的平均后按"越小越好"编秩；``ranking`` 再给出一个明确的
           全序（并列按方案下标打破）。

    复杂度:
        时间 O(m³ + m² n) / 空间 O(m² + mn)。

    陷阱:
        - **两套阈值不是同一把尺子**：c 越大越严格、d 越小越严格，所以强关系对应
          ``c_strong >= c_weak`` 与 ``d_strong <= d_weak``；把顺序写反不会报错但会得到
          "弱关系比强关系更严格"的荒谬结果（本实现直接拒绝这类输入）。
        - 默认阈值取非对角均值 ±0.1 只是**约定**，目的是让两级关系在默认参数下真的不同；
          结论对阈值非常敏感，论文里必须给出阈值敏感性分析，不能只报一组数。
        - 强关系**不传递**：蒸馏必须逐轮重算极大集，不能用全局弱关系的极大集替代。
        - 不一致性用逐指标极差做分母：某指标若对所有方案几乎相同（极差≈0），一旦进入不一致性
          集合就会被放大成接近 1 的值（极差 0 的列按分母 1.0 处理，等于只贡献 0）。
        - 一致性用 ``>=``（并列算"不劣"），且向量归一化后某列若全为 0（无区分度指标），
          ``0 >= 0`` 使它对每一对方案都算一致，等于给所有方案对平白加分。
        - 关系可能成环，此时极大集为空；``_distill_relation`` 把剩余方案整体作为一组兜底，
          避免死循环——出现这种情况要在论文里说明，而不是假装蒸馏成功。

    参考:
        Roy & Bertier (1971) ELECTRE II；Roy (1978)；Figueira, Greco & Ehrgott (2005)
        Multiple Criteria Decision Analysis 第 5 章。
    """
    if benefit is not None and directions is not None:
        raise ValueError("benefit 与 directions 是同一个参数（指标方向），只能给一个")

    Xm = as_matrix(X, "X")
    m, n = Xm.shape
    if m < 2:
        raise ValueError(f"ELECTRE II 至少需要 2 个方案，得到 m={m}")

    w = _normalize_weights(weights, n)
    ben = _resolve_benefit(benefit if benefit is not None else directions, n)
    Z = _oriented(Xm, ben)
    R = _safe_norm(Z)
    V = R * w

    span_v = V.max(axis=0) - V.min(axis=0)
    denom = np.where(span_v > _EPS, span_v, 1.0)

    concordance = np.ones((m, m))
    discordance = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            not_worse = V[a] >= V[b]
            concordance[a, b] = float(w[not_worse].sum())
            worse = ~not_worse
            if np.any(worse):
                discordance[a, b] = float(((V[b] - V[a])[worse] / denom[worse]).max())

    off = ~np.eye(m, dtype=bool)
    c_bar = float(concordance[off].mean())
    d_bar = float(discordance[off].mean())
    c_strong, c_weak = _level_thresholds(c_thresholds, c_bar, 0.1, "c_thresholds",
                                         weaker_is_larger=False)
    d_strong, d_weak = _level_thresholds(d_thresholds, d_bar, 0.1, "d_thresholds",
                                         weaker_is_larger=True)

    strong = (concordance >= c_strong - _EPS) & (discordance <= d_strong + _EPS)
    weak_rel = (concordance >= c_weak - _EPS) & (discordance <= d_weak + _EPS)
    np.fill_diagonal(strong, False)
    np.fill_diagonal(weak_rel, False)
    strong_only = weak_rel & ~strong

    strong_matrix = strong.astype(int)
    weak_matrix = weak_rel.astype(int)
    strong_relations = [[a, b] for a in range(m) for b in range(m) if strong[a, b]]
    weak_relations = [[a, b] for a in range(m) for b in range(m) if strong_only[a, b]]
    incomparable = [[a, b] for a in range(m) for b in range(a + 1, m)
                    if not weak_rel[a, b] and not weak_rel[b, a]]

    desc_groups = _distill_relation(strong, weak_rel, descending=True)
    asc_groups = _distill_relation(strong, weak_rel, descending=False)

    descending = np.empty(m, dtype=int)
    for gi, group in enumerate(desc_groups):
        for a in group:
            descending[a] = gi + 1
    n_asc = len(asc_groups)
    ascending = np.empty(m, dtype=int)
    for gi, group in enumerate(asc_groups):
        for a in group:
            ascending[a] = n_asc - gi

    avg = (descending.astype(float) + ascending.astype(float)) / 2.0
    rank = _ranks_from_scores(-avg)
    ranking = sorted(range(m), key=lambda a: (int(rank[a]), a))

    return {"concordance": concordance, "discordance": discordance,
            "normalized": R, "weighted": V,
            "thresholds": {"c_strong": round(c_strong, 9), "c_weak": round(c_weak, 9),
                           "d_strong": round(d_strong, 9), "d_weak": round(d_weak, 9)},
            "strong_matrix": strong_matrix, "weak_matrix": weak_matrix,
            "strong_relations": strong_relations, "weak_relations": weak_relations,
            "incomparable": incomparable,
            "descending": descending, "ascending": ascending,
            "rank": rank, "ranking": ranking,
            "distillation_levels": {"descending": desc_groups, "ascending": asc_groups}}


# --------------------------------------------------------------------------
# 秩和比 RSR
# --------------------------------------------------------------------------

def rank_sum_ratio(X: MatrixLike, weights: ArrayLike,
                   benefit: Optional[Sequence[bool]] = None) -> Dict[str, object]:
    """秩和比（RSR）：按指标编秩 → 加权秩和 → 归一化，越大越优。

    参数:
        X: 决策矩阵，形状 (m, n)。
        weights: 长度 n 的指标权重（非负，内部归一化到和为 1）。
        benefit: 长度 n 的方向序列；None 表示全部正向。

    返回:
        dict，键为：
        ``rsr``   形状 (m,) 的秩和比，取值 [1/m, 1]，**越大越优**；
        ``rank``  形状 (m,) 的 int 名次（1 为最好，按 rsr 降序，并列同名次）。

    算法:
        1. 逐指标编秩：正向指标中最大值得秩 1、成本型指标中最小值得秩 1（并列取平均秩）。
           设原始秩阵为 R（1 为最优），令经典秩 ``R' = m + 1 - R``（此时 R' 是"高优指标
           秩次大"的教材口径）。
        2. 加权秩和 ``RSR_raw = Σ_j w_j R'_ij``，归一化 ``rsr = RSR_raw / m ∈ [1/m, 1]``。
           两种写法完全等价：``rsr = (m + 1 - Σ_j w_j R_ij) / m``，本实现按后者计算。
        3. ``rank`` 按 rsr 降序编秩（1 为最好）。

    复杂度:
        时间 O(mn log m) / 空间 O(mn)。

    陷阱:
        - 秩和比只用到**序信息**，指标的量级差异被完全丢弃：两个方案在某指标上差 0.001 与
          差 1000 得到同样的秩差。需要保留量级信息时应改用 TOPSIS/VIKOR。
        - 并列取平均秩会让 rsr 出现非整分数的等间隔栅格；m 很小时（例如 m <= 4）秩和比的
          分辨率极低，分档几乎没有意义。
        - 后续概率单位分档用的是"升序位次"的中位秩修正 ``(位次 - 0.5) / m``，见
          ``rsr_distribution``。

    参考:
        田凤调 (1988) 秩和比法；《中国卫生统计》秩和比法专论。
    """
    Xm = as_matrix(X, "X")
    m, n = Xm.shape
    if m < 2:
        raise ValueError(f"秩和比至少需要 2 个方案，得到 m={m}")

    w = _normalize_weights(weights, n)
    ben = _resolve_benefit(benefit, n)

    ranks = np.zeros((m, n))
    for j in range(n):
        col = Xm[:, j]
        # 教材口径的 RSR 用**平均秩**处理并列（1.5, 1.5, 3），不是竞赛排名法（1, 1, 3）
        ranks[:, j] = _average_ranks(col) if ben[j] else _average_ranks(-col)

    raw = ranks @ w                       # 加权秩和，秩 1 为最优，取值 [1, m]
    rsr = (m + 1.0 - raw) / float(m)      # 翻正：越大越优，取值 [1/m, 1]
    rank = _ranks_from_scores(rsr)
    return {"rsr": rsr, "rank": rank}


def rsr_distribution(rsr: ArrayLike, n_levels: int = 3) -> Dict[str, object]:
    """RSR 的概率单位（probit）分档：中位秩累计频率 → 正态分位数 → 线性回归定档。

    参数:
        rsr: 长度 m 的秩和比（``rank_sum_ratio`` 的输出，越大越优）。
        n_levels: 分档数，默认 3（要求 ``2 <= n_levels <= m``）。

    返回:
        dict，键为：
        ``probit``      形状 (m,) 的概率单位 ``probit_i = Φ^{-1}((位次_i - 0.5) / m)``；
        ``regression``  dict，``{"slope","intercept","r2","n"}``，
                        ``probit = intercept + slope * rsr`` 的最小二乘拟合；
        ``grades``      形状 (m,) 的 int 档位，取值 1..n_levels，**编号越大越优**。

    算法:
        1. 把 rsr 升序排序，第 pos 位（0 起）的中位秩累计频率取 ``p = (pos + 1 - 0.5)/m``，
           概率单位 ``probit = _norm_ppf(p)``（Acklam 有理逼近 + Halley 修正，精度 ~1e-9）。
        2. 用最小二乘拟合 ``probit = a + b·rsr``（``np.linalg.lstsq``），
           ``r2 = 1 - SS_res / SS_tot``；``probit`` 无变异（``SS_tot = 0``，m >= 3 时不会发生）
           时 r2 记为 1.0 以避免 0/0。rsr 全相同时回归斜率会退化为 0（设计矩阵秩亏），
           此时分档会全部落在同一档，这是正确行为而不是 bug。
        3. 分档：用标准正态分位数 ``Φ^{-1}(j / n_levels)``（j = 1..n_levels-1）作切点，
           按**拟合值** ``intercept + slope·rsr`` 落点给档位，档位编号越大越优。

    复杂度:
        时间 O(m log m) / 空间 O(m)。

    陷阱:
        - 分档切点用的是**回归拟合值**而不是原始 probit：这样即使某个方案的 probit 落在极端
          尾部，档位也由它在回归线上的位置决定，不会因为单点抖动而跳档。
        - ``m`` 很小时中位秩修正 ``(位次 - 0.5)/m`` 仍然无法覆盖 ``(0, 1)`` 的两端，但本实现
          不会取到 p = 0 或 1（p ∈ [0.5/m, 1 - 0.5/m]），所以不会出现 ±inf。
        - 回归的 ``r2`` 接近 1 只说明 probit 与 rsr 线性关系强，**不代表**分档本身有统计显著性。

    参考:
        田凤调 (1988) 秩和比法中的概率单位分档；Acklam (2003) 正态分位数逼近。
    """
    r = as_vector(rsr, "rsr")
    m = r.size
    if m < 3:
        raise ValueError(f"概率单位分档至少需要 3 个方案，得到 m={m}")
    if not (2 <= n_levels <= m):
        raise ValueError(f"n_levels 应在 [2, {m}] 内，得到 n_levels={n_levels}")

    order = np.argsort(r, kind="mergesort")
    probit = np.zeros(m)
    for pos, idx in enumerate(order):
        p = (pos + 1 - 0.5) / float(m)      # 中位秩修正：(位次 - 0.5) / m
        probit[idx] = _norm_ppf(p)

    design = np.column_stack([np.ones(m), r])
    coef, _res, _rank, _sv = np.linalg.lstsq(design, probit, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    fitted = intercept + slope * r
    ss_tot = float(((probit - probit.mean()) ** 2).sum())
    ss_res = float(((probit - fitted) ** 2).sum())
    r2 = 1.0 if ss_tot <= _EPS else 1.0 - ss_res / ss_tot

    cuts = [_norm_ppf(j / float(n_levels)) for j in range(1, n_levels)]
    grades = np.ones(m, dtype=int)
    for cut in cuts:
        grades += (fitted > cut).astype(int)

    return {"probit": probit,
            "regression": {"slope": slope, "intercept": intercept, "r2": r2, "n": int(m)},
            "grades": grades}


# --------------------------------------------------------------------------
# Borda / Copeland / 共识
# --------------------------------------------------------------------------

def borda_count(rankings) -> Dict[str, object]:
    """Borda 计分：每份排名的第 p 位（0 起）给 ``len(ranking) - 1 - p`` 分。

    参数:
        rankings: list of list，每项是一份从最优到最差的方案下标排序；
            允许截断（只排前几名），也允许不同排名覆盖不同方案子集。

    返回:
        dict，键为：
        ``scores``  形状 (m,) 的 Borda 总分（m = 出现过的最大下标 + 1），越大越好；
        ``rank``    形状 (m,) 的 int 名次（1 为最好，按总分降序，并列同名次）。

    算法:
        1. 校验每份排名是互不重复的非负整数下标（重复下标无定义，直接报错）。
        2. 对每份排名，第 p 位方案加 ``L - 1 - p`` 分（L 为该排名的长度）；未出现的方案得 0 分。
        3. 累加总分，按降序编秩。

    复杂度:
        时间 O(Σ L) / 空间 O(m + Σ L)。

    陷阱:
        - 截断选票的口径不唯一：这里按"该排名自身的长度"给分（第 1 名得 L-1 分），
          另一种常见口径是"所有方案数 m-1 分"并给缺项记平均分。两种口径在截断深度不同时
          会给出不同赢家，论文里必须写清楚用的是哪一种。
        - Borda 会被"推举无关方案"操纵（加入一个永远垫底的新方案会改变相对分差），
          这是 Borda 的著名缺陷，不是实现 bug。

    参考:
        Borda (1781)；Black (1958) The Theory of Committees and Elections。
    """
    parsed, m = _parse_rankings(rankings)
    scores = np.zeros(m)
    for idx in parsed:
        length = len(idx)
        for pos, a in enumerate(idx):
            scores[a] += float(length - 1 - pos)
    rank = _ranks_from_scores(scores)
    return {"scores": scores, "rank": rank}


def copeland_score(pairwise_wins) -> Dict[str, object]:
    """Copeland 计分：逐对比较的净胜场数。

    参数:
        pairwise_wins: 形状 (m, m) 的逐对结果矩阵，``1`` 表示 i 胜 j、``-1`` 表示 i 负 j、
            ``0`` 表示平局。要求对角线为 0 且**反对称** ``W[i, j] = -W[j, i]``。

    返回:
        dict，键为：
        ``score``  形状 (m,) 的 Copeland 分 = 胜场数 - 负场数，越大越好；
        ``rank``   形状 (m,) 的 int 名次（1 为最好，按得分降序，并列同名次）。

    算法:
        1. 校验形状方阵、取值只含 {-1, 0, 1}、对角线为 0、反对称。
        2. ``score_i = Σ_j W[i, j]``（因为平局贡献 0，求和恰好等于胜场数减负场数）。
        3. 按得分降序编秩。

    复杂度:
        时间 O(m²) / 空间 O(m²)。

    陷阱:
        - 反对称是硬要求：很多"胜场矩阵"只有 0/1（不分平局与负），传进来会被拒绝——那种矩阵
          应该先转成 -1/0/1 或改用 ``borda_count``。
        - Copeland 满足 Condorcet 准则（存在 Condorcet 赢家时它一定排第一），但会出现大面积
          并列（例如三方案循环对决时三人都是 0 分），这时名次全为 1，需要靠次级准则打破平局。

    参考:
        Copeland (1951)；Saari & Merlin (1996) The Copeland method。
    """
    W = as_matrix(pairwise_wins, "pairwise_wins")
    m = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"pairwise_wins 必须是方阵，得到形状 {W.shape}")
    if np.any(np.abs(W - np.round(W)) > _EPS):
        raise ValueError("pairwise_wins 只能取 -1 / 0 / 1")
    if np.any(np.abs(W) > 1.0 + _EPS):
        raise ValueError("pairwise_wins 只能取 -1 / 0 / 1")
    if np.any(np.abs(np.diag(W)) > _EPS):
        raise ValueError("pairwise_wins 的对角线必须为 0（方案不能与自己比较）")
    if np.any(np.abs(W + W.T) > _EPS):
        raise ValueError("pairwise_wins 必须反对称：W[i, j] 必须等于 -W[j, i]")

    scores = W.sum(axis=1)
    rank = _ranks_from_scores(scores)
    return {"score": scores, "rank": rank}


def rank_consensus(rankings) -> Dict[str, object]:
    """多份排序的一致性诊断：两两 Spearman 秩相关及其均值。

    参数:
        rankings: list of list，每项是一份从最优到最差的方案下标排序（允许截断）。

    返回:
        dict，键为：
        ``mean_spearman``  float，所有两两 Spearman 相关的均值（只有一份排名时记为 1.0）；
        ``pairwise``       形状 (K, K) 的相关矩阵（K = 排名份数），对角为 1.0、对称；
        ``is_consistent``  bool，``mean_spearman >= 0.8`` 记为一致。

    算法:
        1. 解析每份排名为下标列表（同一份排名内不允许重复下标），同时建好
           ``方案下标 -> 位次(1 为最好)`` 的字典。
        2. 对每一对排名，取**共同出现**的方案，用上一步的字典 O(1) 取出各自位次组成秩向量，
           再用 ``_spearman`` 算秩相关（其内部对并列取平均秩）。
        3. 均值 >= 0.8 判定为一致（0.8 是竞赛论文里常用的经验阈值，不是统计检验）。

    复杂度:
        时间 O(K·m + K²·m) / 空间 O(K² + K·m)。
        （位次用字典查，避免每个共同方案都做一次 ``list.index`` 的 O(m) 扫描，否则是
        O(K²·m²)。）

    陷阱:
        - 共同方案少于 2 个时直接抛 ValueError，而不是返回 0：两段几乎不相交的排名之间
          根本不存在可比较的秩相关。
        - 0.8 是**经验阈值**而非显著性检验：K 小时即使 rho=0.8 也可能不显著，反之 K 大时
          rho=0.75 也可能高度显著。需要正式结论时应补充 Kendall's W 或置换检验。
        - 该诊断只回答"排序是否相似"，不回答"哪个排序对"。

    参考:
        Spearman (1904)；Kendall & Smith (1939) 秩相关；Saaty (2008) 决策中一致性度量的讨论。
    """
    parsed, m = _parse_rankings(rankings)
    k = len(parsed)
    pos = [{a: p + 1 for p, a in enumerate(idx)} for idx in parsed]
    pairwise = np.ones((k, k))
    values: List[float] = []
    for i in range(k):
        pi = pos[i]
        for j in range(i + 1, k):
            pj = pos[j]
            common = [x for x in parsed[i] if x in pj]
            if len(common) < 2:
                raise ValueError(f"rankings[{i}] 与 rankings[{j}] 的共同方案只有 "
                                 f"{len(common)} 个，无法计算 Spearman")
            ri = [pi[x] for x in common]
            rj = [pj[x] for x in common]
            rho = _spearman(ri, rj)
            pairwise[i, j] = rho
            pairwise[j, i] = rho
            values.append(rho)

    mean_rho = float(np.mean(values)) if values else 1.0
    return {"mean_spearman": mean_rho, "pairwise": pairwise,
            "is_consistent": bool(mean_rho >= 0.8)}


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    参数:
        无。

    返回:
        dict，键名简短 ASCII，值是 int / float / bool / str / list / dict；固定种子下
        两次调用必须完全一致。

    算法:
        每个算法至少配一条"独立结论"校验，避免自测只是回显实现自己的输出：
        1. ``_norm_ppf``：对拍闭式值 Φ^{-1}(0.975) = 1.959963984540054、Φ^{-1}(0.5) = 0、
           反对称性 Φ^{-1}(p) = -Φ^{-1}(1-p)，以及 ``math.erf`` 的独立逆变换校验。
        2. PROMETHEE II：三方案单指标 usual 偏好下 phi_net 手算为 [-1, 0, 1]；
           净流之和恒为 0；成本型指标"取负 + benefit=True"与"不取负 + benefit=False"必须同解。
        3. ELECTRE I：三方案完全支配算例（0 支配 1、2；2 支配 1）下内核必须恰为 {0}，
           一致性/不一致性矩阵的值用支配结构手算对拍。
        4. ELECTRE III：完全支配算例下 S(0,1)=1、S(1,0)=0，可信度对角为 1 且落在 [0,1]，
           升降蒸馏名次都是 1..m 的合法分组编号。
        5. ELECTRE II：三方案、指标列互为排列的手算算例下 C/D 逐元素对拍（C 由权重求和、
           D 由极差归一后的最大劣势给出）；显式强/弱阈值下的强关系、仅弱关系、不可比对与
           升降蒸馏分组全部手算核对；缺省阈值等于非对角均值 ∓0.1；阈值取极端值（c=0/d=1
           与 c=1/d=0）时关系退化为"全强"与"全不可比"；并在 6×4 固定随机算例上与 ELECTRE I、
           PROMETHEE II 的名次做 Kendall tau-b 交叉验证，同时校验三张关系矩阵的结构不变式。
        6. 秩和比：单指标时 rsr 排序必须与原始指标排序一致；rsr 落在 [1/m, 1] 且
           "全优方案"的 rsr 恰为 1（闭式）；带并列的算例用**平均秩**手算核对
           （竞赛排名法会给出不同名次），并验证平均秩法下 rsr 均值恒为 (m+1)/(2m)。
        7. RSR 分档：把 probit 值本身当作 rsr 输入时，回归必须给出 slope=1、intercept=0、
           r2=1（闭式自洽对拍），从而验证中位秩修正与回归口径。
        8. Borda：三方案循环对决的三份排名旋转对称，总分必须完全并列；
           多数决算例的赢家必须与直观一致（手算）。
        9. Copeland：传递性算例得分手算为 [2, 0, -2]；循环对决下三人同分；
           并与 Borda 在同一个多数决算例上交叉验证赢家一致（两种独立实现互证）。
        10. 一致性诊断：完全相同排名 rho=1、完全反向 rho=-1（闭式），
            并用无并列的全序算例对拍闭式 ``1 - 6Σd²/(m(m²-1))``，再用带并列的算例对拍
            Spearman（=1/3）与 Kendall tau-b（=0.2）的并列修正口径。

    复杂度:
        时间 O(1)（全部是 m <= 6 的毫秒级小算例）/ 空间 O(1)。

    陷阱:
        这里的期望值都是手算/闭式推出的常数；一旦改动偏好函数口径、阈值默认值或编秩规则，
        这些常数会立刻失配——这正是它作为回归护栏的价值。反过来，**它不检验** λ、c/d 阈值
        这类"约定性参数"的合理性，换约定时自测需要同步更新。
    """
    out: Dict[str, object] = {}

    # ---- 1) 正态分位数 _norm_ppf 的闭式对拍 -----------------------------
    z975 = _norm_ppf(0.975)
    out["probit_975"] = round(z975, 9)
    out["probit_500"] = round(_norm_ppf(0.5), 9)
    out["probit_symmetry"] = round(abs(_norm_ppf(0.025) + z975), 9)
    # 独立校验：Φ(z) 用 math.erf 直接算，必须回到 0.975
    out["probit_erf_residual"] = round(abs(0.5 * math.erfc(-z975 / math.sqrt(2.0)) - 0.975), 12)
    if abs(z975 - 1.959963984540054) > 1e-9:
        raise AssertionError(f"_norm_ppf(0.975)={z975} 与闭式值 1.959963984540054 不符")
    if _norm_ppf(0.5) != 0.0:
        raise AssertionError(f"_norm_ppf(0.5)={_norm_ppf(0.5)} 应为精确 0")
    if abs(_norm_ppf(0.025) + z975) > 1e-12:
        raise AssertionError("_norm_ppf 的反对称性 Φ^{-1}(p) = -Φ^{-1}(1-p) 不成立")

    # ---- 2) PROMETHEE II：三方案单指标 usual 偏好的手算解 ----------------
    # 值 3 > 2 > 1：pi 为 0/1 支配矩阵，phi_net 手算 = [-1, 0, 1]，名次 [3, 2, 1]
    ph = promethee_ii([[1.0], [2.0], [3.0]], [1.0], preference="usual")
    out["promethee_usual_phi_net"] = [round(float(v), 6) for v in ph["phi_net"]]
    out["promethee_usual_rank"] = [int(v) for v in ph["rank"]]
    out["promethee_usual_phi_plus"] = [round(float(v), 6) for v in ph["phi_plus"]]
    if [round(float(v), 9) for v in ph["phi_net"]] != [-1.0, 0.0, 1.0]:
        raise AssertionError(f"promethee usual phi_net={ph['phi_net'].tolist()} "
                             f"与手算 [-1, 0, 1] 不符")

    # 净流之和恒为 0（结构性质，与偏好函数/权重无关）
    ph2 = promethee_ii([[5.0, 2.0], [3.0, 9.0], [7.0, 4.0], [1.0, 6.0]],
                       [0.6, 0.4], benefit=[True, False], preference="linear",
                       p=[2.0, 3.0], q=[0.5, 0.5])
    out["promethee_net_sum"] = round(float(ph2["phi_net"].sum()), 9)
    out["promethee_rank"] = [int(v) for v in ph2["rank"]]
    out["promethee_phi_net"] = [round(float(v), 6) for v in ph2["phi_net"]]
    out["promethee_phi_plus"] = [round(float(v), 6) for v in ph2["phi_plus"]]
    out["promethee_pairwise_row0"] = [round(float(v), 6) for v in ph2["pairwise"][0]]
    if abs(float(ph2["phi_net"].sum())) > 1e-12:
        raise AssertionError(f"PROMETHEE 净流之和 {float(ph2['phi_net'].sum())} 应为 0")

    # 方向不变性：成本型列取负 + benefit=True 与不取负 + benefit=False 必须同解
    gen = rng(20240101)  # 显式写死种子（= 当前 DEFAULT_SEED 的取值），使黄金值与库默认种子解耦
    Xr = np.round(gen.normal(size=(6, 4)) * 3.0, 3)
    wr = [0.3, 0.4, 0.2, 0.1]
    ben_mix = [True, False, True, True]
    inv_a = promethee_ii(Xr, wr, benefit=ben_mix, preference="linear")
    Xr_flip = np.array(Xr, dtype=float)
    Xr_flip[:, 1] = -Xr_flip[:, 1]
    inv_b = promethee_ii(Xr_flip, wr, benefit=[True, True, True, True], preference="linear")
    out["promethee_orientation_gap"] = round(
        float(np.max(np.abs(inv_a["phi_net"] - inv_b["phi_net"]))), 12)
    if float(np.max(np.abs(inv_a["phi_net"] - inv_b["phi_net"]))) > 1e-9:
        raise AssertionError("成本型指标取负与 benefit=False 两种口径给出的净流不一致")

    # ---- 3) ELECTRE I：完全支配算例，内核必须恰为 {0} -------------------
    Xe = [[5.0, 5.0], [3.0, 3.0], [4.0, 4.0]]
    e1 = electre_i(Xe, [0.5, 0.5], benefit=[True, True])
    out["electre1_kernel"] = [int(v) for v in e1["kernel"]]
    out["electre1_relation"] = [[int(v) for v in row] for row in e1["relation"]]
    out["electre1_concordance"] = [[round(float(v), 6) for v in row]
                                   for row in e1["concordance"]]
    out["electre1_discordance"] = [[round(float(v), 6) for v in row]
                                   for row in e1["discordance"]]
    out["electre1_c_threshold"] = round(
        float(e1["concordance"][~np.eye(3, dtype=bool)].mean()), 6)
    # 手算：方案 0 在两个指标上都不劣于 1，故 C(0,1)=1、D(0,1)=0；反向 C(1,0)=0
    if abs(float(e1["concordance"][0, 1]) - 1.0) > 1e-12:
        raise AssertionError(f"ELECTRE I 的 C(0,1)={e1['concordance'][0, 1]} 应为 1")
    if abs(float(e1["discordance"][0, 1])) > 1e-12:
        raise AssertionError(f"ELECTRE I 的 D(0,1)={e1['discordance'][0, 1]} 应为 0")
    if [int(v) for v in e1["kernel"]] != [0]:
        raise AssertionError(f"ELECTRE I 内核={e1['kernel']} 应为 [0]（0 支配 1、2）")

    # ---- 4) ELECTRE III：完全支配算例的可信度与蒸馏 ---------------------
    e3 = electre_iii(Xe, [0.5, 0.5], benefit=[True, True])
    out["electre3_rank"] = [int(v) for v in e3["rank"]]
    out["electre3_ascending"] = [int(v) for v in e3["ascending"]]
    out["electre3_descending"] = [int(v) for v in e3["descending"]]
    out["electre3_credibility"] = [[round(float(v), 6) for v in row]
                                   for row in e3["credibility"]]
    out["electre3_diag_min"] = round(float(np.diag(e3["credibility"]).min()), 9)
    out["electre3_in_range"] = bool(np.all(e3["credibility"] >= -1e-12)
                                    and np.all(e3["credibility"] <= 1.0 + 1e-12))
    if abs(float(e3["credibility"][0, 1]) - 1.0) > 1e-12:
        raise AssertionError(f"ELECTRE III 的 S(0,1)={e3['credibility'][0, 1]} 应为 1")
    if abs(float(e3["credibility"][1, 0])) > 1e-12:
        raise AssertionError(f"ELECTRE III 的 S(1,0)={e3['credibility'][1, 0]} 应为 0")
    if abs(float(np.diag(e3["credibility"]).min()) - 1.0) > 1e-12:
        raise AssertionError("ELECTRE III 可信度矩阵对角线应全为 1")
    if int(e3["rank"][0]) != 1 or int(e3["rank"][1]) != 3:
        raise AssertionError(f"ELECTRE III 名次 {e3['rank'].tolist()} 应为 [1, 3, 2]（0 最优）")

    # ---- 5) ELECTRE II：三方案手算 C/D + 阈值语义 + 升降蒸馏 -------------
    Xtwo = [[3.0, 2.0, 1.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0]]
    wtwo = [0.5, 0.3, 0.2]
    e2_hand = electre_ii(Xtwo, wtwo, benefit=[True, True, True],
                         c_thresholds=[0.5, 0.4], d_thresholds=[0.5, 0.6])
    out["electre2_concordance"] = [[round(float(v), 6) for v in row]
                                   for row in e2_hand["concordance"]]
    out["electre2_discordance"] = [[round(float(v), 6) for v in row]
                                   for row in e2_hand["discordance"]]
    out["electre2_thresholds"] = dict(e2_hand["thresholds"])
    out["electre2_strong_relations"] = [[int(v) for v in p]
                                        for p in e2_hand["strong_relations"]]
    out["electre2_weak_relations"] = [[int(v) for v in p]
                                      for p in e2_hand["weak_relations"]]
    out["electre2_incomparable"] = [[int(v) for v in p] for p in e2_hand["incomparable"]]
    out["electre2_rank"] = [int(v) for v in e2_hand["rank"]]
    out["electre2_ranking"] = [int(v) for v in e2_hand["ranking"]]
    out["electre2_descending"] = [int(v) for v in e2_hand["descending"]]
    out["electre2_ascending"] = [int(v) for v in e2_hand["ascending"]]
    out["electre2_levels"] = {"descending": [[int(v) for v in g] for g in
                                             e2_hand["distillation_levels"]["descending"]],
                              "ascending": [[int(v) for v in g] for g in
                                            e2_hand["distillation_levels"]["ascending"]]}
    # 手算：三列都是 {1,2,3} 的排列 ⇒ 每列 L2 范数都是 sqrt(14)、极差都是 2/sqrt(14)，
    # 于是 D(a,b) = max|X[b,j]-X[a,j]| / 2（权重在 D 里被约掉），C 见下方期望矩阵。
    c_exp = np.array([[1.0, 0.8, 0.5], [0.2, 1.0, 0.7], [0.5, 0.3, 1.0]])
    d_exp = np.array([[0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 0.5, 0.0]])
    if float(np.max(np.abs(e2_hand["concordance"] - c_exp))) > 1e-12:
        raise AssertionError(f"ELECTRE II 的 C 应为 {c_exp.tolist()}，"
                             f"实际 {e2_hand['concordance'].tolist()}")
    if float(np.max(np.abs(e2_hand["discordance"] - d_exp))) > 1e-12:
        raise AssertionError(f"ELECTRE II 的 D 应为 {d_exp.tolist()}，"
                             f"实际 {e2_hand['discordance'].tolist()}")
    if out["electre2_strong_relations"] != [[0, 2]]:
        raise AssertionError(f"手算阈值下强关系应为 [[0, 2]]，"
                             f"实际 {out['electre2_strong_relations']}")
    if out["electre2_weak_relations"] != []:
        raise AssertionError(f"手算阈值下不存在『仅弱』关系，实际 {out['electre2_weak_relations']}")
    if out["electre2_incomparable"] != [[0, 1], [1, 2]]:
        raise AssertionError(f"手算阈值下不可比对应为 [[0, 1], [1, 2]]，"
                             f"实际 {out['electre2_incomparable']}")
    if out["electre2_ranking"] != [0, 1, 2] or out["electre2_rank"] != [1, 2, 3]:
        raise AssertionError(f"手算阈值下 ranking 应为 [0, 1, 2]，"
                             f"实际 {out['electre2_ranking']} / rank {out['electre2_rank']}")

    # 缺省阈值：强 = 非对角均值，弱 = 强 ∓ 0.1（C 减、D 加），且弱关系必须包含强关系
    e2_def = electre_ii(Xtwo, wtwo, benefit=[True, True, True])
    out["electre2_default_thresholds"] = dict(e2_def["thresholds"])
    out["electre2_default_ranking"] = [int(v) for v in e2_def["ranking"]]
    out["electre2_default_strong"] = [[int(v) for v in p] for p in e2_def["strong_relations"]]
    if abs(e2_def["thresholds"]["c_strong"] - 0.5) > 1e-9:
        raise AssertionError(f"缺省 c_strong 应等于非对角一致性均值 0.5，"
                             f"实际 {e2_def['thresholds']['c_strong']}")
    if abs(e2_def["thresholds"]["c_weak"] - 0.4) > 1e-9:
        raise AssertionError(f"缺省 c_weak 应为 0.4，实际 {e2_def['thresholds']['c_weak']}")
    if abs(e2_def["thresholds"]["d_strong"] - 0.75) > 1e-9:
        raise AssertionError(f"缺省 d_strong 应等于非对角不一致性均值 0.75，"
                             f"实际 {e2_def['thresholds']['d_strong']}")
    if abs(e2_def["thresholds"]["d_weak"] - 0.85) > 1e-9:
        raise AssertionError(f"缺省 d_weak 应为 0.85，实际 {e2_def['thresholds']['d_weak']}")

    # 阈值极端值：全松（c=0, d=1）⇒ 任意有序对都强级别高于；全紧（c=1, d=0）⇒ 全不可比
    e2_loose = electre_ii(Xtwo, wtwo, benefit=[True, True, True],
                          c_thresholds=[0.0, 0.0], d_thresholds=[1.0, 1.0])
    out["electre2_loose_strong"] = len(e2_loose["strong_relations"])
    out["electre2_loose_incomparable"] = len(e2_loose["incomparable"])
    if len(e2_loose["strong_relations"]) != 6 or len(e2_loose["incomparable"]) != 0:
        raise AssertionError("阈值放宽到 c=0、d=1 时，6 个有序对应全部构成强关系且无不可比对")
    e2_tight = electre_ii(Xtwo, wtwo, benefit=[True, True, True],
                          c_thresholds=[1.0, 1.0], d_thresholds=[0.0, 0.0])
    out["electre2_tight_strong"] = len(e2_tight["strong_relations"])
    out["electre2_tight_incomparable"] = [[int(v) for v in p]
                                          for p in e2_tight["incomparable"]]
    if len(e2_tight["strong_relations"]) != 0 or len(e2_tight["incomparable"]) != 3:
        raise AssertionError("阈值收紧到 c=1、d=0 时，不应有任何关系，三对方案全部不可比")

    # 与模块内 ELECTRE I / PROMETHEE II 交叉验证名次（同为 6×4 固定随机算例）
    e2_rand = electre_ii(Xr, wr, benefit=ben_mix)
    e2_baseline = electre_i(Xr, wr, benefit=ben_mix)
    e2_prom = promethee_ii(Xr, wr, benefit=ben_mix)
    out["electre2_rand_ranking"] = [int(v) for v in e2_rand["ranking"]]
    out["electre2_rand_rank"] = [int(v) for v in e2_rand["rank"]]
    out["electre2_rand_strong"] = len(e2_rand["strong_relations"])
    out["electre2_rand_weak_only"] = len(e2_rand["weak_relations"])
    out["electre2_rand_incomparable"] = len(e2_rand["incomparable"])
    tau_e1 = _kendall_tau(-e2_rand["rank"].astype(float), e2_baseline["relation"].sum(axis=1))
    tau_p2 = _kendall_tau(-e2_rand["rank"].astype(float), e2_prom["phi_net"])
    out["electre2_rand_tau_electre1"] = round(float(tau_e1), 6)
    out["electre2_rand_tau_promethee2"] = round(float(tau_p2), 6)
    # 结构一致性：强关系 ⊆ 弱关系、对角线为 0、三个互斥列表恰好铺满所有有序/无序对
    strong_m = e2_rand["strong_matrix"]
    weak_m = e2_rand["weak_matrix"]
    if np.any(np.diag(strong_m) != 0) or np.any(np.diag(weak_m) != 0):
        raise AssertionError("ELECTRE II 的强/弱关系矩阵对角线必须为 0")
    if np.any(strong_m > weak_m):
        raise AssertionError("强级别高于关系必须包含在弱级别高于关系里")
    if len(e2_rand["strong_relations"]) + len(e2_rand["weak_relations"]) != int(weak_m.sum()):
        raise AssertionError("strong_relations + weak_relations 必须等于弱关系矩阵的非零元个数")
    # 逐无序对分类：双向弱 / 单向弱 / 双向都够不上（不可比），三类必须恰好铺满 C(6,2)=15
    mutual = sum(1 for a in range(6) for b in range(a + 1, 6)
                 if weak_m[a, b] and weak_m[b, a])
    one_way = sum(1 for a in range(6) for b in range(a + 1, 6)
                  if bool(weak_m[a, b]) != bool(weak_m[b, a]))
    none_way = sum(1 for a in range(6) for b in range(a + 1, 6)
                   if not weak_m[a, b] and not weak_m[b, a])
    out["electre2_rand_pair_classes"] = [int(mutual), int(one_way), int(none_way)]
    if mutual + one_way + none_way != 15:
        raise AssertionError("弱关系的无序对分类必须恰好铺满 C(6,2)=15 对")
    if int(weak_m.sum()) != 2 * mutual + one_way:
        raise AssertionError("弱关系矩阵非零元个数必须等于 2×双向弱 + 单向弱")
    if none_way != len(e2_rand["incomparable"]):
        raise AssertionError("incomparable 必须恰好是『双向都够不上弱关系』的无序对")
    if np.any(np.isnan(e2_rand["weighted"])) or np.any(np.isinf(e2_rand["weighted"])):
        raise AssertionError("ELECTRE II 的加权矩阵出现 NaN/inf")
    if sorted(out["electre2_rand_ranking"]) != list(range(6)):
        raise AssertionError(f"ranking 必须是 0..5 的排列，实际 {out['electre2_rand_ranking']}")
    # 三族方法在同一算例上的名次应当正相关（阈值类方法会分档，不做逐位相等的要求）
    if tau_e1 < 0.6 or tau_p2 < 0.6:
        raise AssertionError(f"ELECTRE II 与 ELECTRE I / PROMETHEE II 的名次一致性过低："
                             f"tau(e1)={tau_e1}, tau(p2)={tau_p2}")


    # ---- 6) 秩和比：单指标时排序与原始排序一致 + rsr 闭式上界 ----------
    rsr1 = rank_sum_ratio([[3.0], [1.0], [2.0]], [1.0], benefit=[True])
    out["rsr_single"] = [round(float(v), 6) for v in rsr1["rsr"]]
    out["rsr_single_rank"] = [int(v) for v in rsr1["rank"]]
    # 手算：秩 [1, 3, 2]，rsr = (3+1-R)/3 -> [1.0, 1/3, 2/3]，排序与 3>2>1 一致
    if [int(v) for v in rsr1["rank"]] != [1, 3, 2]:
        raise AssertionError(f"单指标 rsr 名次 {rsr1['rank'].tolist()} 应为 [1, 3, 2]")
    if abs(float(rsr1["rsr"][0]) - 1.0) > 1e-12:
        raise AssertionError("全优方案的 rsr 闭式值应为 1.0")

    Xrsr = [[8.0, 7.0, 3.0], [6.0, 9.0, 5.0], [9.0, 5.0, 2.0],
            [5.0, 8.0, 4.0], [7.0, 6.0, 6.0]]
    rsr = rank_sum_ratio(Xrsr, [0.4, 0.35, 0.25], benefit=[True, True, False])
    out["rsr_values"] = [round(float(v), 6) for v in rsr["rsr"]]
    out["rsr_rank"] = [int(v) for v in rsr["rank"]]
    out["rsr_min"] = round(float(rsr["rsr"].min()), 6)
    out["rsr_max"] = round(float(rsr["rsr"].max()), 6)
    if float(rsr["rsr"].min()) < 1.0 / 5.0 - 1e-12 or float(rsr["rsr"].max()) > 1.0 + 1e-12:
        raise AssertionError(f"rsr 越界：{[float(v) for v in rsr['rsr']]}")

    # 并列回归：教材口径必须用**平均秩**（1.5, 1.5, 3），用竞赛排名法（1, 1, 3）会算错
    Xtie = [[1.0, 2.0], [1.0, 3.0], [3.0, 2.0], [3.0, 5.0], [2.0, 2.0]]
    rsr_t = rank_sum_ratio(Xtie, [0.5, 0.5], benefit=[True, True])
    out["rsr_tie_values"] = [round(float(v), 6) for v in rsr_t["rsr"]]
    out["rsr_tie_rank"] = [int(v) for v in rsr_t["rank"]]
    out["rsr_tie_mean"] = round(float(rsr_t["rsr"].mean()), 9)
    # 手算平均秩：列 0 = [4.5, 4.5, 1.5, 1.5, 3.0]，列 1 = [4, 2, 4, 1, 4]，
    # raw = (R0 + R1)/2 = [4.25, 3.25, 2.75, 1.25, 3.5]，rsr = (6 - raw)/5
    if [round(float(v), 6) for v in rsr_t["rsr"]] != [0.35, 0.55, 0.65, 0.95, 0.5]:
        raise AssertionError(f"并列算例的平均秩 RSR 应为 [0.35, 0.55, 0.65, 0.95, 0.5]，"
                             f"实际 {[round(float(v), 6) for v in rsr_t['rsr']]}")
    if [int(v) for v in rsr_t["rank"]] != [5, 3, 2, 1, 4]:
        raise AssertionError(f"并列算例的 rsr 名次应为 [5, 3, 2, 1, 4]，"
                             f"实际 {rsr_t['rank'].tolist()}")
    # 不变式：平均秩法下每列秩的均值恒为 (m+1)/2，故 rsr 的均值恒为 (m+1)/(2m)
    if abs(float(rsr_t["rsr"].mean()) - 3.0 / 5.0) > 1e-12:
        raise AssertionError(f"平均秩法下 rsr 均值应为 (m+1)/(2m)=0.6，"
                             f"实际 {float(rsr_t['rsr'].mean())}")

    # ---- 7) RSR 概率单位分档：把 probit 当 rsr 输入应回归出 y = x --------
    toy = [_norm_ppf((i + 0.5) / 9.0) for i in range(9)]
    dist = rsr_distribution(toy, n_levels=3)
    reg = dist["regression"]
    out["rsr_probit_slope"] = round(float(reg["slope"]), 9)
    out["rsr_probit_intercept"] = round(float(reg["intercept"]), 9)
    out["rsr_probit_r2"] = round(float(reg["r2"]), 9)
    out["rsr_probit_n"] = int(reg["n"])
    if abs(float(reg["slope"]) - 1.0) > 1e-8 or abs(float(reg["intercept"])) > 1e-8:
        raise AssertionError(f"probit 自洽回归应得 slope=1, intercept=0，"
                            f"实际 {reg['slope']}, {reg['intercept']}")
    if float(reg["r2"]) < 1.0 - 1e-8:
        raise AssertionError(f"probit 自洽回归的 r2={reg['r2']} 应接近 1")

    dist2 = rsr_distribution(rsr["rsr"], n_levels=3)
    out["rsr_dist_probit"] = [round(float(v), 6) for v in dist2["probit"]]
    out["rsr_dist_grades"] = [int(v) for v in dist2["grades"]]
    out["rsr_dist_slope"] = round(float(dist2["regression"]["slope"]), 6)
    out["rsr_dist_r2"] = round(float(dist2["regression"]["r2"]), 6)
    if not np.all(np.asarray(dist2["grades"]) >= 1) or \
            not np.all(np.asarray(dist2["grades"]) <= 3):
        raise AssertionError(f"档位越界：{dist2['grades'].tolist()}")
    # rsr 越大越优 ⇒ 回归斜率必须为正（probit 随 rsr 单调增）
    if float(dist2["regression"]["slope"]) <= 0:
        raise AssertionError("rsr 越大越优，probit 对 rsr 的回归斜率应为正")

    # ---- 8) Borda：旋转对称的 Condorcet 循环必须完全并列 ----------------
    cyc = borda_count([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    out["borda_cycle_scores"] = [round(float(v), 6) for v in cyc["scores"]]
    out["borda_cycle_rank"] = [int(v) for v in cyc["rank"]]
    if [float(v) for v in cyc["scores"]] != [3.0, 3.0, 3.0]:
        raise AssertionError(f"旋转对称算例的 Borda 总分应全为 3，实际 {cyc['scores'].tolist()}")
    maj = borda_count([[0, 1, 2], [0, 1, 2], [0, 2, 1]])
    out["borda_scores"] = [round(float(v), 6) for v in maj["scores"]]
    out["borda_rank"] = [int(v) for v in maj["rank"]]
    if [int(v) for v in maj["rank"]] != [1, 2, 3]:
        raise AssertionError(f"多数决算例的 Borda 名次 {maj['rank'].tolist()} 应为 [1, 2, 3]")
    if abs(float(maj["scores"][0]) - 6.0) > 1e-12:
        raise AssertionError(f"方案 0 每份排名都是第 1，总分应为 6，实际 {maj['scores'][0]}")

    # ---- 9) Copeland：传递性算例手算 + 与 Borda 交叉验证赢家 -------------
    trans = copeland_score([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    out["copeland_score"] = [round(float(v), 6) for v in trans["score"]]
    out["copeland_rank"] = [int(v) for v in trans["rank"]]
    if [float(v) for v in trans["score"]] != [2.0, 0.0, -2.0]:
        raise AssertionError(f"传递性算例 Copeland 分应为 [2, 0, -2]，实际 {trans['score'].tolist()}")

    cyc_w = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=float)
    cyc_cop = copeland_score(cyc_w)
    out["copeland_cycle_score"] = [round(float(v), 6) for v in cyc_cop["score"]]
    out["copeland_cycle_rank"] = [int(v) for v in cyc_cop["rank"]]

    # 由三份排名构造多数决矩阵（独立于 Borda 的计分口径），赢家应当一致
    votes = [[0, 1, 2], [0, 1, 2], [1, 2, 0]]
    Wmaj = np.zeros((3, 3))
    for i in range(3):
        for j in range(i + 1, 3):
            prefer_i = sum(1 for v in votes if v.index(i) < v.index(j))
            prefer_j = len(votes) - prefer_i
            Wmaj[i, j] = float(np.sign(prefer_i - prefer_j))
            Wmaj[j, i] = -Wmaj[i, j]
    cop_maj = copeland_score(Wmaj)
    borda_maj = borda_count(votes)
    out["copeland_maj_score"] = [round(float(v), 6) for v in cop_maj["score"]]
    out["consensus_crosscheck"] = bool(int(np.argmax(cop_maj["score"]))
                                       == int(np.argmin(borda_maj["rank"])))
    if not out["consensus_crosscheck"]:
        raise AssertionError("Borda 与 Copeland 在同一多数决算例上给出的赢家不一致")

    # ---- 10) 排序一致性：闭式 Spearman 对拍 + 并列修正 -------------------
    same = rank_consensus([[0, 1, 2, 3], [0, 1, 2, 3]])
    out["consensus_same_mean"] = round(float(same["mean_spearman"]), 9)
    out["consensus_same_flag"] = bool(same["is_consistent"])
    rev = rank_consensus([[0, 1, 2, 3], [3, 2, 1, 0]])
    out["consensus_reversed_mean"] = round(float(rev["mean_spearman"]), 9)
    mixed = rank_consensus([[0, 1, 2, 3], [1, 0, 3, 2]])
    out["consensus_mixed_mean"] = round(float(mixed["mean_spearman"]), 9)
    out["consensus_pairwise"] = [[round(float(v), 6) for v in row]
                                 for row in mixed["pairwise"]]
    # 闭式：无并列时 rho = 1 - 6 Σd² / (m(m²-1))；下面这组 d = [1,-1,1,-1]，Σd² = 4
    closed = 1.0 - 6.0 * 4.0 / (4.0 * (16.0 - 1.0))
    if abs(float(mixed["mean_spearman"]) - closed) > 1e-12:
        raise AssertionError(f"Spearman={float(mixed['mean_spearman'])} 与闭式 "
                             f"{closed} 不符")
    if abs(float(rev["mean_spearman"]) + 1.0) > 1e-12:
        raise AssertionError(f"完全反序的 Spearman 应为 -1，实际 {rev['mean_spearman']}")
    if abs(float(same["mean_spearman"]) - 1.0) > 1e-12:
        raise AssertionError("完全相同的排名 Spearman 应为 1")

    # 并列修正：带并列的秩向量必须走平均秩，与 scipy.stats.spearmanr 口径一致
    rho_tie = _spearman([1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 1.0, 3.0])
    out["spearman_tie"] = round(float(rho_tie), 9)
    tau_tie = _kendall_tau([1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 1.0, 3.0])
    out["kendall_tie"] = round(float(tau_tie), 9)
    # 手算：平均秩后 Pearson 相关 = 1/3；Kendall tau-b = (3 - 2) / sqrt(5·5) = 0.2
    if abs(rho_tie - 1.0 / 3.0) > 1e-12:
        raise AssertionError(f"带并列的 Spearman 应为 1/3，实际 {rho_tie}")
    if abs(tau_tie - 0.2) > 1e-12:
        raise AssertionError(f"带并列的 Kendall tau-b 应为 0.2，实际 {tau_tie}")

    return out
