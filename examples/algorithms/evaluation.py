"""评价与决策模型：AHP、熵权、CRITIC、TOPSIS、VIKOR、灰关联、DEA、模糊综合。

评价类题目是国赛/研赛出现频率最高的一类，也是"看起来简单、写起来容易失分"的一类。
这个模块的共同约定：

- **指标方向**：``benefit`` 参数是一个长度 n 的 bool 序列，``True`` 表示"越大越好"（正向指标），
  ``False`` 表示"越小越好"（成本型）。传 ``None`` 表示全部按正向处理。
- **权重**：一律返回**和为 1** 的数组，方便直接放进论文表格。若权重和不为 1，函数会先归一化并
  在返回的 ``note`` 里说明——静默归一化会让你在答辩时说不清权重到底是多少。
- **排名**：``rank`` 采用"1 为最好"的口径，并列时给相同名次（标准竞赛排名法）。

生产环境可用的库见 ``references/github-resources.md``；本模块的价值在于每个中间量
（归一化矩阵、正负理想解、距离、检验数）都显式返回，便于论文里画出完整的计算过程。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._common import as_matrix, as_vector, normalize_l2, rng, safe_divide
from .optimization import simplex_lp

__all__ = [
    "ahp_weights",
    "entropy_weights",
    "critic_weights",
    "combine_weights",
    "topsis",
    "vikor",
    "grey_relational_grade",
    "dea_ccr",
    "dea_bcc",
    "fuzzy_comprehensive_eval",
    "topsis_rank_sensitivity",
]

#: Saaty 随机一致性指标 RI（n = 1..15），用于判断矩阵一致性检验。
SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
            8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56,
            14: 1.57, 15: 1.59}


def _resolve_benefit(benefit, n: int) -> np.ndarray:
    """把 benefit 参数统一成长度 n 的 bool 数组。"""
    if benefit is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(benefit)
    if arr.dtype == bool:
        if arr.size != n:
            raise ValueError(f"benefit 长度 {arr.size} 与指标数 {n} 不一致")
        return arr
    if np.issubdtype(arr.dtype, np.integer) and set(np.unique(arr)).issubset({0, 1}):
        if arr.size != n:
            raise ValueError(f"benefit 长度 {arr.size} 与指标数 {n} 不一致")
        return arr.astype(bool)
    raise ValueError("benefit 必须是 bool 序列（或 0/1 序列），或者 None（全部按正向指标处理）")


def _normalize_weights(weights, n: int) -> Tuple[np.ndarray, Optional[str]]:
    """把权重压成非负、和为 1 的数组；返回 (权重, 提示信息或 None)。"""
    w = as_vector(weights, "weights")
    if w.size != n:
        raise ValueError(f"权重长度 {w.size} 与指标数 {n} 不一致")
    if np.any(w < -1e-12):
        raise ValueError("权重不能为负")
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        raise ValueError("权重之和必须为正")
    if abs(total - 1.0) > 1e-9:
        return w / total, f"输入权重之和为 {total:.6f}，已归一化到 1"
    return w, None


def _ranks(scores: np.ndarray) -> np.ndarray:
    """按"1 为最好"给出名次，并列同名次（竞赛排名法）。"""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty(len(scores), dtype=int)
    sorted_scores = np.asarray(scores, dtype=float)[order]
    cur_rank = 1
    for pos, idx in enumerate(order):
        if pos > 0 and abs(sorted_scores[pos] - sorted_scores[pos - 1]) > 1e-12:
            cur_rank = pos + 1
        ranks[idx] = cur_rank
    return ranks


# --------------------------------------------------------------------------
# 权重确定
# --------------------------------------------------------------------------

def ahp_weights(pairwise, max_iter: int = 1000, tol: float = 1e-12) -> Dict[str, object]:
    """层次分析法（AHP）判断矩阵求权重 + 一致性检验。

    参数:
        pairwise: n x n 正互反判断矩阵（``a_ij = 1 / a_ji``，对角元为 1）。
        max_iter: 幂法迭代上限。
        tol: 幂法收敛阈值。

    返回:
        dict，键为 ``weights``（特征向量归一化后的权重）、``lambda_max``（最大特征值）、
        ``CI``（一致性指标）、``CR``（一致性比例）、``consistent``（CR < 0.1）、
        ``RI``、``n``。

    算法:
        幂法求主特征向量：``w <- A w / ||A w||`` 直到收敛；``lambda_max`` 用
        ``mean((A w)_i / w_i)`` 估计。``CI = (lambda_max - n) / (n - 1)``，
        ``CR = CI / RI``。

    复杂度:
        时间 O(max_iter * n^2) / 空间 O(n^2)。

    陷阱:
        - **n <= 2 时 CR 恒为 0**（RI=0），这不是"判断矩阵很完美"，而是无法检验，
          论文里要说明。
        - 判断矩阵必须**正互反**。常见错误是填了 ``a_ij = 3`` 却忘记 ``a_ji = 1/3``，
          本实现会直接报错而不是悄悄算出一个错权重。
        - 幂法求的是主特征向量，方向可能整体为负；本实现取绝对值后归一化。
        - CR 达标 ≠ 权重合理：CR 只说明你的判断前后不矛盾。专家打分本身偏了就救不回来。

    参考:
        Saaty (1980)《The Analytic Hierarchy Process》；Saaty 随机一致性指标 RI 表。
    """
    A = as_matrix(pairwise, "pairwise")
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("判断矩阵必须是方阵")
    if np.any(A <= 0):
        raise ValueError("判断矩阵元素必须为正")
    if np.any(np.abs(np.diag(A) - 1.0) > 1e-9):
        raise ValueError("判断矩阵对角线必须为 1")
    if np.any(np.abs(A * A.T - 1.0) > 1e-6):
        raise ValueError("判断矩阵必须正互反（a_ij * a_ji = 1）")

    w = np.ones(n) / n
    for _ in range(max_iter):
        w_new = A @ w
        norm = np.linalg.norm(w_new)
        if norm <= 0:
            raise ValueError("判断矩阵退化，幂法无法收敛")
        w_new = w_new / norm
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    w = np.abs(w)
    w = w / w.sum()
    Aw = A @ w
    lambda_max = float(np.mean(Aw / w))
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = SAATY_RI.get(n)
    if RI is None:
        return {"weights": w, "lambda_max": lambda_max, "CI": CI, "CR": None,
                "consistent": None, "RI": None, "n": n,
                "note": f"n={n} 超出内置 RI 表（1..15），请自行查表判断一致性"}
    CR = 0.0 if RI == 0 else CI / RI
    return {"weights": w, "lambda_max": lambda_max, "CI": CI, "CR": CR,
            "consistent": bool(CR < 0.1), "RI": RI, "n": n}


def entropy_weights(X, benefit=None) -> Dict[str, object]:
    """熵权法：由数据本身的离散程度确定客观权重。

    参数:
        X: 决策矩阵，形状 (m 个方案, n 个指标)。
        benefit: 指标方向；成本型指标会先做正向化（取极差反转）。

    返回:
        dict，键为 ``weights``、``entropy``（各指标信息熵 e_j）、``divergence``（1 - e_j）、
        ``p``（比重矩阵）、``normalized``（正向化并归一化后的矩阵）。

    算法:
        1. 成本型指标正向化：``x' = max - x``。
        2. 计算比重 ``p_ij = x'_ij / sum_i x'_ij``。
        3. 信息熵 ``e_j = -(1/ln m) * sum_i p_ij ln p_ij``（``p=0`` 的项按 0 处理）。
        4. 差异系数 ``d_j = 1 - e_j``，权重 ``w_j = d_j / sum d_j``。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        - **要求非负**。含负数的指标（例如"利润变化率"）必须先做平移，否则比重无意义；
          本实现会直接报错。
        - 熵权对**量纲和离散度极度敏感**：把所有指标先 min-max 归一化与直接算熵权，
          结果完全不同。论文里必须写清楚你用的是哪一种口径，否则无法复现。
        - 某指标所有方案取值相同时 ``d_j = 0``，该指标权重为 0。这通常说明指标选得不好，
          要在论文里讨论而不是假装没看见。
        - 熵权是"谁差异大谁重要"，**不等于"谁业务上重要"**，不要用它替代专家判断。

    参考:
        Shannon (1948) 信息熵；多属性决策中的熵权法（客观赋权通例）。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    ben = _resolve_benefit(benefit, n)
    if np.any(M < -1e-12):
        raise ValueError("熵权法要求非负数据；含负数时请先平移（并说明平移量）")

    pos = M.copy()
    for j in range(n):
        if not ben[j]:
            pos[:, j] = M[:, j].max() - M[:, j]

    col_sum = pos.sum(axis=0, keepdims=True)
    if np.any(col_sum <= 0):
        zero_cols = [j for j in range(n) if col_sum[0, j] <= 0]
        raise ValueError(f"以下指标列和为 0，无法计算比重：{zero_cols}")
    P = pos / col_sum

    k = 1.0 / np.log(m) if m > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        plnp = np.where(P > 0, P * np.log(P), 0.0)
    e = -k * plnp.sum(axis=0)
    e = np.clip(e, 0.0, 1.0)
    d = 1.0 - e
    if d.sum() <= 0:
        raise ValueError("所有指标的信息熵都为 1（无区分度），无法赋权")
    w = d / d.sum()
    return {"weights": w, "entropy": e, "divergence": d, "p": P, "normalized": pos}


def critic_weights(X, benefit=None) -> Dict[str, object]:
    """CRITIC 法：同时考虑指标内的对比强度与指标间的冲突性。

    参数:
        X: 决策矩阵 (m, n)。
        benefit: 指标方向（成本型先极差反转）。

    返回:
        dict，键为 ``weights``、``contrast``（标准差 σ_j）、``conflict``（sum_i (1 - r_ij)）、
        ``information``（C_j = σ_j * conflict_j）、``correlation``（相关系数矩阵）。

    算法:
        ``C_j = σ_j * sum_i (1 - r_ij)``，``w_j = C_j / sum C_j``。

    复杂度:
        时间 O(mn + n^2 m) / 空间 O(mn + n^2)。

    陷阱:
        - 标准差要用**样本标准差**（ddof=1）还是总体标准差（ddof=0），不同教材不一致。
          本实现用 ``ddof=1``；换口径权重会变，论文里要写明。
        - 与其他指标高度相关的指标会被判为"冲突小、信息量低"。如果两个指标本质重复，
          这是合理惩罚；但如果它们恰好同等重要，权重会被不合理压低，需要人为修正。
        - 和熵权一样受量纲影响，先归一化再算。

    参考:
        Diakoulaki, Mavrotas & Papayannakis (1995) CRITIC 法。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    ben = _resolve_benefit(benefit, n)
    pos = M.copy()
    for j in range(n):
        if not ben[j]:
            pos[:, j] = M[:, j].max() - M[:, j]

    if m < 2:
        raise ValueError("CRITIC 需要至少 2 个方案才能计算标准差与相关系数")
    sigma = pos.std(axis=0, ddof=1)
    R = np.corrcoef(pos, rowvar=False)
    R = np.atleast_2d(R)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    conflict = (1.0 - R).sum(axis=0)
    info = sigma * conflict
    if info.sum() <= 0:
        raise ValueError("CRITIC 信息量为 0，指标无区分度")
    w = info / info.sum()
    return {"weights": w, "contrast": sigma, "conflict": conflict,
            "information": info, "correlation": R}


def combine_weights(weight_sets, method: str = "multiplicative",
                    alphas=None) -> Dict[str, object]:
    """主客观权重组合（博弈论组合赋权 / 乘法合成 / 线性加权）。

    参数:
        weight_sets: 若干组权重，形如 ``[[w1...], [w1...]]``，每组长度相同。
        method: ``"multiplicative"``（乘法合成，默认）、``"linear"``（线性加权）、
            ``"geometric"``（几何平均）。
        alphas: 仅 ``"linear"`` 使用；各权重组的系数，None 表示等权。

    返回:
        dict，键为 ``weights``（组合权重）、``method``、``note``。

    算法:
        - multiplicative: ``w_j ∝ prod_k w_kj``
        - geometric:      ``w_j ∝ (prod_k w_kj)^(1/K)``
        - linear:         ``w_j = sum_k alpha_k w_kj``

    复杂度:
        时间 O(Kn) / 空间 O(Kn)。

    陷阱:
        - **乘法合成会把"小权重"惩罚得非常狠**：两组权重都接近 0 的指标会被彻底压没。
          如果你的指标体系里有"小而重要"的指标，乘法合成会毁掉它。
        - 组合权重没有唯一的"正确"方法。论文里必须写清楚组合方式和理由，
          并且**做一次灵敏度分析**证明排序不因组合方式而翻盘。

    参考:
        主客观组合赋权的常见做法（乘法合成、线性加权、博弈论组合赋权）。
    """
    sets = [as_vector(w, "weights") for w in weight_sets]
    if not sets:
        raise ValueError("weight_sets 不能为空")
    n = sets[0].size
    for w in sets:
        if w.size != n:
            raise ValueError("各组权重长度必须一致")
        if np.any(w < 0):
            raise ValueError("权重不能为负")
    W = np.vstack(sets)

    if method == "multiplicative":
        w = np.prod(W, axis=0)
    elif method == "geometric":
        w = np.exp(np.mean(np.log(np.clip(W, 1e-300, None)), axis=0))
    elif method == "linear":
        K = W.shape[0]
        a = np.ones(K) / K if alphas is None else as_vector(alphas, "alphas")
        if a.size != K:
            raise ValueError("alphas 长度必须等于权重组数")
        if abs(a.sum() - 1.0) > 1e-9:
            a = a / a.sum()
        w = a @ W
    else:
        raise ValueError(f"未知的组合方法：{method}")

    if w.sum() <= 0:
        raise ValueError("组合后权重全为 0")
    w = w / w.sum()
    return {"weights": w, "method": method, "note": None}


# --------------------------------------------------------------------------
# 综合评价
# --------------------------------------------------------------------------

def topsis(X, weights, benefit=None) -> Dict[str, object]:
    """TOPSIS 逼近理想解排序法。

    参数:
        X: 决策矩阵 (m 个方案, n 个指标)。
        weights: 指标权重（长度 n，会自动归一化）。
        benefit: 指标方向；``True`` 为正向指标。

    返回:
        dict，键为 ``closeness``（相对贴近度 C_i，越大越好）、``rank``（1 为最好）、
        ``d_plus``/``d_minus``（到正/负理想解的欧氏距离）、``ideal_best``/``ideal_worst``、
        ``normalized``（向量归一化矩阵 R）、``weighted``（加权规范矩阵 V）、``note``。

    算法:
        1. 向量归一化 ``r_ij = x_ij / sqrt(sum_i x_ij^2)``。
        2. 加权 ``v_ij = w_j r_ij``。
        3. 正理想解取正向指标的最大值、成本型指标的最小值（负理想解反之）。
        4. ``C_i = d_i^- / (d_i^+ + d_i^-)``。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        - **向量归一化必须做**。直接用原始数据算距离，量纲大的指标会独占权重。
          顺带一提：TOPSIS 的"正向化"要在归一化**之前**做（对成本型指标取倒数或极差反转），
          顺序颠倒会得到完全不同的排序。
        - 正负理想解是**从你给的方案集里选出来的**，不是绝对标准。加入一个很差的新方案，
          原有方案的贴近度会集体上升——所以不能跨数据集比较 C_i。
        - ``C_i`` 接近时排序不稳，务必配合 :func:`topsis_rank_sensitivity` 做扰动分析。
        - 距离用欧氏范数隐含"各指标可替代"的假设；指标间高度相关时考虑马氏距离。

    参考:
        Hwang & Yoon (1981) TOPSIS；Chen & Hwang (1992) 模糊 TOPSIS 扩展。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    ben = _resolve_benefit(benefit, n)
    w, note = _normalize_weights(weights, n)

    # 第一步：正向化（成本型指标做极差反转），必须在归一化之前
    pos = M.copy()
    for j in range(n):
        if not ben[j]:
            pos[:, j] = M[:, j].max() - M[:, j]

    # 第二步：向量归一化；第三步：加权
    R = normalize_l2(pos, axis=0)
    V = R * w[None, :]

    # 第四步：正向化之后所有指标都是"越大越好"，理想解直接取列最大/最小
    best = V.max(axis=0)
    worst = V.min(axis=0)
    d_plus = np.sqrt(((V - best[None, :]) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - worst[None, :]) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    C = np.divide(d_minus, denom, out=np.full(m, 0.5), where=denom > 0)
    return {"closeness": C, "rank": _ranks(C), "d_plus": d_plus, "d_minus": d_minus,
            "ideal_best": best, "ideal_worst": worst, "normalized": R,
            "weighted": V, "positivized": pos, "note": note}


def vikor(X, weights, benefit=None, v: float = 0.5) -> Dict[str, object]:
    """VIKOR 折衷排序法（同时考虑群体效用与个体遗憾）。

    参数:
        X: 决策矩阵 (m, n)。
        weights: 指标权重。
        benefit: 指标方向。
        v: 决策机制系数，``v > 0.5`` 偏群体效用（多数票），``v < 0.5`` 偏个体遗憾（否决权）。

    返回:
        dict，键为 ``Q``（折衷值，越小越好）、``S``（群体效用）、``R``（个体遗憾）、
        ``rank``、``conditions``（是否同时满足可接受优势与可接受决策可靠性）。

    算法:
        ``S_i = sum_j w_j (f*_j - f_ij) / (f*_j - f^-_j)``；
        ``R_i = max_j [同上]``；
        ``Q_i = v (S_i - S*) / (S^- - S*) + (1 - v)(R_i - R*) / (R^- - R*)``。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        - VIKOR 的结论**必须做两项检验**：``Q(A(1)) - Q(A(2)) >= 1/(m-1)``（可接受优势），
          且 A(1) 在 S 或 R 中也排第一（可接受决策可靠性）。只报 Q 值排序是不完整的，
          评委很可能会追问。本实现把结论放在 ``conditions`` 里。
        - 某个指标在所有方案上取值相同时分母为 0，需要特殊处理（本实现跳过该指标并记入 note）。
        - ``v`` 的取值会改变排序，论文里要写明取值并做敏感性分析。

    参考:
        Opricovic & Tzeng (2004) VIKOR 折衷排序法。
    """
    M = as_matrix(X, "M")
    m, n = M.shape
    ben = _resolve_benefit(benefit, n)
    w, note = _normalize_weights(weights, n)
    if not 0.0 <= v <= 1.0:
        raise ValueError("v 必须在 [0, 1] 内")
    if m < 2:
        raise ValueError("VIKOR 至少需要 2 个方案")

    f_best = np.where(ben, M.max(axis=0), M.min(axis=0))
    f_worst = np.where(ben, M.min(axis=0), M.max(axis=0))
    span = f_best - f_worst
    skipped = [j for j in range(n) if abs(span[j]) < 1e-12]

    ratios = np.zeros((m, n))
    for j in range(n):
        if abs(span[j]) < 1e-12:
            continue
        ratios[:, j] = (f_best[j] - M[:, j]) / span[j]

    S = (ratios * w[None, :]).sum(axis=1)
    R = (ratios * w[None, :]).max(axis=1)

    if skipped and note is None:
        note = f"以下指标在所有方案上取值相同，已跳过：{skipped}"
    elif skipped:
        note = note + f"；另有指标取值相同已跳过：{skipped}"

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    q1 = safe_divide(v * (S - S_star), S_minus - S_star, fill=0.0)
    q2 = safe_divide((1 - v) * (R - R_star), R_minus - R_star, fill=0.0)
    Q = q1 + q2

    rank = _ranks(Q)
    order = np.argsort(Q, kind="mergesort")
    first, second = order[0], order[1]
    accept_advantage = bool(Q[second] - Q[first] >= 1.0 / (m - 1) - 1e-12)
    accept_reliability = bool(int(np.argmin(S)) == first or int(np.argmin(R)) == first)
    return {"Q": Q, "S": S, "R": R, "rank": rank,
            "conditions": {"acceptable_advantage": accept_advantage,
                           "acceptable_reliability": accept_reliability,
                           "compromise_reached": bool(accept_advantage and accept_reliability)},
            "note": note}


def grey_relational_grade(X, reference=None, weights=None, benefit=None,
                          normalize: str = "minmax",
                          rho: float = 0.5) -> Dict[str, object]:
    """灰色关联分析：算各方案与参考序列的关联度。

    参数:
        X: 决策矩阵 (m, n)，或"s 个评价对象 x n 个指标"。
        reference: 参考序列（长度 n，**原始量纲**，本函数会对它施加与 X 相同的正向化与
            无量纲化处理）。None 表示用各列最优值（正向取最大、负向取最小）作参考。
        weights: 各指标权重，None 表示等权。
        benefit: 长度 n 的布尔序列，True 表示正向（效益型）指标。None 表示全部按正向处理。
            **成本型指标必须在这里标出来**，否则"越小越好"的指标会被当成"越大越好"。
        normalize: 无量纲化方式，``"minmax"``（极差化，映射到 [0, 1]）、
            ``"mean"``（均值化，除以列均值）或 ``"none"``（不做，量纲风险自负）。
        rho: 分辨系数，取值 (0, 1]，通常 0.5。

    返回:
        dict，键为 ``grade``（关联度，越大越接近参考序列）、``rank``、
        ``xi``（关联系数矩阵）、``reference``（处理后的参考序列）、
        ``normalized``（处理后的决策矩阵）。

    算法:
        1. 正向化：成本型指标 ``x' = max(x) - x``。
        2. 无量纲化：按 ``normalize`` 处理，得到 Z。
        3. 参考序列 ``z0`` = Z 的各列最优值（或由 ``reference`` 变换而来）。
        4. ``xi_ij = (d_min + rho * d_max) / (|z0_j - z_ij| + rho * d_max)``，其中
           ``d_min``、``d_max`` 是全局最小/最大绝对差；``grade_i = sum_j w_j xi_ij``。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        - 灰色关联**必须先无量纲化**，否则量纲大的指标会主导差值。本实现默认极差化。
        - **成本型指标必须先正向化**。如果忘了，一个"越小越好"的指标会把最差的方案推成
          第一名——这类错误在论文里非常常见且很难自查。
        - ``rho`` 越小，关联系数之间的差异被放得越大。很多论文直接用 0.5 却不说明，
          严格来说应当做 rho 的敏感性分析。
        - 参考序列如果是理论最优（而非样本内最优），必须明确写出它的来源，
          否则读者无法判断你的"最优"是否可达。
        - 关联度只表示"接近参考序列的程度"，**不是"距离"也不是"概率"**，
          不要跨模型直接比较数值大小。

    参考:
        邓聚龙（1985）灰色系统理论；灰色关联度分析。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    if rho <= 0 or rho > 1:
        raise ValueError("分辨系数 rho 必须在 (0, 1] 内")
    ben = _resolve_benefit(benefit, n)

    # 1) 正向化
    pos = M.copy()
    for j in range(n):
        if not ben[j]:
            pos[:, j] = M[:, j].max() - M[:, j]

    # 2) 无量纲化
    if normalize == "minmax":
        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        Z = safe_divide(pos - lo[None, :], hi - lo, fill=0.0)
        # 列恒定时 safe_divide 返回 0，表示该指标不提供区分信息
    elif normalize == "mean":
        Z = safe_divide(pos, np.abs(pos).mean(axis=0), fill=0.0)
    elif normalize == "none":
        Z = pos
    else:
        raise ValueError(f"未知的无量纲化方式：{normalize}")

    if reference is None:
        ref = Z.max(axis=0)
    else:
        rv = as_vector(reference, "reference")
        if rv.size != n:
            raise ValueError("reference 长度与指标数不一致")
        r = rv.astype(float).copy()
        for j in range(n):
            if not ben[j]:
                r[j] = M[:, j].max() - r[j]
        if normalize == "minmax":
            r = safe_divide(r - lo, hi - lo, fill=0.0)
        elif normalize == "mean":
            r = safe_divide(r, np.abs(pos).mean(axis=0), fill=0.0)
        ref = r

    diff = np.abs(Z - ref[None, :])
    d_min, d_max = float(diff.min()), float(diff.max())
    if d_max <= 1e-12:
        xi = np.ones_like(diff)
    else:
        xi = (d_min + rho * d_max) / (diff + rho * d_max)

    if weights is None:
        w = np.ones(n) / n
    else:
        w, _ = _normalize_weights(weights, n)
    grade = (xi * w[None, :]).sum(axis=1)
    return {"grade": grade, "rank": _ranks(grade), "xi": xi, "reference": ref,
            "normalized": Z}


# --------------------------------------------------------------------------
# DEA
# --------------------------------------------------------------------------

def _dea_envelopment(inputs, outputs, vrs: bool) -> Dict[str, object]:
    """DEA 包络模型（CCR: vrs=False，BCC: vrs=True）的输入导向实现。"""
    X = as_matrix(inputs, "inputs")     # (n_dmu, n_input)
    Y = as_matrix(outputs, "outputs")   # (n_dmu, n_output)
    n_dmu = X.shape[0]
    if Y.shape[0] != n_dmu:
        raise ValueError("inputs 与 outputs 的 DMU 数量不一致")
    ni, no = X.shape[1], Y.shape[1]

    efficiencies = np.zeros(n_dmu)
    lambdas = np.zeros((n_dmu, n_dmu))
    statuses: List[str] = []

    for k in range(n_dmu):
        # 变量: [theta, lambda_0..lambda_{n-1}]
        nv = 1 + n_dmu
        c = np.zeros(nv)
        c[0] = 1.0
        A_ub = []
        b_ub = []
        for i in range(ni):
            row = np.zeros(nv)
            row[0] = -X[k, i]
            row[1:] = X[:, i]
            A_ub.append(row)
            b_ub.append(0.0)
        for r in range(no):
            row = np.zeros(nv)
            row[1:] = -Y[:, r]
            A_ub.append(row)
            b_ub.append(-Y[k, r])
        A_eq = None
        b_eq = None
        if vrs:
            A_eq = np.zeros((1, nv))
            A_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])

        res = simplex_lp(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                         A_eq=A_eq, b_eq=b_eq,
                         bounds=[(0.0, None)] * nv, maximize=False, max_iter=1000)
        statuses.append(str(res["status"]))
        if res["status"] == "optimal":
            efficiencies[k] = float(res["x"][0])
            lambdas[k] = np.asarray(res["x"][1:], dtype=float)

    return {"efficiency": efficiencies, "lambda": lambdas, "statuses": statuses,
            "n_inputs": ni, "n_outputs": no, "n_dmu": n_dmu, "vrs": vrs}


def dea_ccr(inputs, outputs) -> Dict[str, object]:
    """DEA-CCR 模型（规模报酬不变）的效率评价。

    参数:
        inputs: 投入矩阵，形状 (n 个决策单元 DMU, n 项投入)。
        outputs: 产出矩阵，形状 (n 个 DMU, n 项产出)。

    返回:
        dict，键为 ``efficiency``（各 DMU 效率，取值 (0, 1]，1 表示 DEA 有效）、
        ``rank``、``lambda``（各 DMU 的参考权重，非零项即该 DMU 的"标杆"）、
        ``peers``（每个 DMU 的参考 DMU 下标）、``statuses``。

    算法:
        对每个 DMU 解输入导向包络 LP：``min theta``
        ``s.t. sum_j lambda_j x_ij <= theta x_ik``，``sum_j lambda_j y_rj >= y_rk``，
        ``lambda >= 0``。用 :func:`simplex_lp` 求解。

    复杂度:
        时间 O(n_dmu * LP(n_dmu)) / 空间 O(n_dmu^2)。

    陷阱:
        - **DEA 效率 1 的 DMU 可能有很多个**，CCR 无法再区分它们。需要区分时请用超效率
          DEA（Anderson-Peterson）或在论文里明确指出"多个 DMU 同为 DEA 有效"。
        - **投入产出指标数之和不应超过 DMU 数量的约 1/3**（经验法则）。指标太多时几乎所有
          DMU 都会变成有效，结论没有信息量。
        - 投入/产出的**方向不能搞反**：把"成本"放进 outputs 会得到完全错误的结论。
        - 负值不允许（DEA 要求非负）。含负值要先做平移，并在论文里说明。
        - 本实现是**输入导向**（在产出不变的前提下最小化投入）。输出导向会得到不同数值，
          注意与文献口径一致。

    参考:
        Charnes, Cooper & Rhodes (1978) CCR 模型；Banker, Charnes & Cooper (1984) BCC 模型。
    """
    if np.any(np.asarray(inputs, dtype=float) < -1e-12) or \
            np.any(np.asarray(outputs, dtype=float) < -1e-12):
        raise ValueError("DEA 要求投入与产出非负；含负值时请先平移并说明")
    res = _dea_envelopment(inputs, outputs, vrs=False)
    eff = res["efficiency"]
    peers = []
    for k in range(res["n_dmu"]):
        idx = [j for j in range(res["n_dmu"]) if res["lambda"][k, j] > 1e-7]
        peers.append(idx)
    return {"efficiency": eff, "rank": _ranks(eff), "lambda": res["lambda"],
            "peers": peers, "statuses": res["statuses"], "efficient_count":
            int(np.sum(eff >= 1 - 1e-6))}


def dea_bcc(inputs, outputs) -> Dict[str, object]:
    """DEA-BCC 模型（规模报酬可变）的**技术效率**。

    参数:
        inputs: 投入矩阵 (n_dmu, n_inputs)。
        outputs: 产出矩阵 (n_dmu, n_outputs)。

    返回:
        dict，键为 ``efficiency``（技术效率 TE）、``rank``、``lambda``、``peers``、
        ``statuses``、``efficient_count``。

    算法:
        在 CCR 包络模型上增加凸性约束 ``sum_j lambda_j = 1``，把"规模有效"从技术效率中剥离。

    复杂度:
        同 :func:`dea_ccr`。

    陷阱:
        - BCC 的技术效率 **一定 >= 对应 CCR 的效率**（约束更松）。如果你的结果反了，
          说明代码或数据有问题。
        - 有了 BCC 与 CCR 之后，**规模效率 SE = TE_CCR / TE_BCC**。请一并报告，
          否则"效率低"到底是因为技术差还是规模不经济说不清。
        - 与 CCR 一样，负值不允许、指标不宜过多。

    参考:
        Banker, Charnes & Cooper (1984) BCC 模型；规模效率分解 TE = PTE x SE。
    """
    if np.any(np.asarray(inputs, dtype=float) < -1e-12) or \
            np.any(np.asarray(outputs, dtype=float) < -1e-12):
        raise ValueError("DEA 要求投入与产出非负；含负值时请先平移并说明")
    res = _dea_envelopment(inputs, outputs, vrs=True)
    eff = res["efficiency"]
    peers = []
    for k in range(res["n_dmu"]):
        idx = [j for j in range(res["n_dmu"]) if res["lambda"][k, j] > 1e-7]
        peers.append(idx)
    return {"efficiency": eff, "rank": _ranks(eff), "lambda": res["lambda"],
            "peers": peers, "statuses": res["statuses"], "efficient_count":
            int(np.sum(eff >= 1 - 1e-6))}


# --------------------------------------------------------------------------
# 模糊综合评判
# --------------------------------------------------------------------------

def fuzzy_comprehensive_eval(weights, membership, operator: str = "weighted",
                             level_scores=None) -> Dict[str, object]:
    """模糊综合评判。

    参数:
        weights: 因素权重，长度 n。
        membership: 隶属度矩阵 R，形状 (n 个因素, k 个评语等级)。
        operator: ``"weighted"``（加权平均型 ``b_j = sum_i w_i r_ij``）或
            ``"max_min"``（主因素决定型 ``b_j = max_i min(w_i, r_ij)``）。
        level_scores: 可选，各评语等级的量化分值（长度 k）。给出后 ``level_value``
            为 ``sum_j Bn_j * level_scores_j``，可用于把等级换算成连续得分。

    返回:
        dict，键为 ``score``（综合隶属度向量）、``level``（按最大隶属度原则给出的等级下标）、
        ``level_value``（若提供 ``level_scores`` 则给出加权得分，否则为 ``None``）、
        ``normalized``（归一化后的综合隶属度）。

    算法:
        - weighted: 矩阵乘法 ``B = W R``，再归一化。
        - max_min:  ``B_j = max_i min(w_i, r_ij)``。

    复杂度:
        时间 O(nk) / 空间 O(nk)。

    陷阱:
        - **最大隶属度原则会丢信息**：如果 B = (0.45, 0.44, 0.11)，说"属于第一级"很勉强。
          本实现同时给出归一化向量，论文里应当把完整向量列出来，不要只报等级。
        - ``max_min`` 算子只让最大的那个因素说话，**权重小的因素完全不起作用**，
          容易得出极端结论；多数场景应当用 ``weighted``。
        - 权重与隶属度都必须落在 [0, 1]；隶属度每行理论上应和为 1（本实现会校验并提示）。

    参考:
        汪培庄（1983）模糊综合评判；Zadeh (1965) 模糊集合。
    """
    R = as_matrix(membership, "membership")
    w, note = _normalize_weights(weights, R.shape[0])
    if R.shape[0] != w.size:
        raise ValueError(f"隶属度矩阵行数 {R.shape[0]} 与权重长度 {w.size} 不一致")
    if np.any(R < -1e-12) or np.any(R > 1 + 1e-12):
        raise ValueError("隶属度必须在 [0, 1] 内")

    row_sums = R.sum(axis=1)
    hint = None
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        hint = "部分因素的各等级隶属度之和不为 1，请确认是否遗漏等级（本实现不做强制归一化）"

    if operator == "weighted":
        B = w @ R
    elif operator == "max_min":
        B = np.minimum(w[:, None], R).max(axis=0)
    else:
        raise ValueError(f"未知算子：{operator}")

    if B.sum() > 0:
        Bn = B / B.sum()
    else:
        Bn = B

    level_value = None
    if level_scores is not None:
        ls = as_vector(level_scores, "level_scores")
        if ls.size != Bn.size:
            raise ValueError(f"level_scores 长度 {ls.size} 与评语等级数 {Bn.size} 不一致")
        level_value = float(Bn @ ls)

    return {"score": B, "normalized": Bn, "level": int(np.argmax(Bn)),
            "level_value": level_value, "note": hint if hint else note}


# --------------------------------------------------------------------------
# 灵敏度
# --------------------------------------------------------------------------

def topsis_rank_sensitivity(X, weights, benefit=None, delta: float = 0.2,
                            n_samples: int = 2000, seed: Optional[int] = None) -> Dict[str, object]:
    """权重扰动下的 TOPSIS 排序稳定性分析（论文里"灵敏度分析"那一节直接用）。

    参数:
        X, weights, benefit: 同 :func:`topsis`。
        delta: 权重扰动的相对幅度，例如 0.2 表示每个权重在 ±20% 内随机变动。
        n_samples: 扰动抽样次数。
        seed: 随机种子。

    返回:
        dict，键为 ``base_rank``（原始排序）、``rank_flip_prob``（各方案名次发生变化的频率）、
        ``best_keep_prob``（原第一名保持第一的频率）、``rank_matrix``（每次抽样的名次矩阵）。

    算法:
        每次抽样生成 ``w_j * (1 + U(-delta, delta))``，非负截断后归一化，重算 TOPSIS 排序，
        统计名次变化频率。

    复杂度:
        时间 O(n_samples * mn) / 空间 O(n_samples * m)。

    陷阱:
        - 这是**局部**灵敏度分析：只在原权重附近扰动。若某个权重实际上可能取到 0 或翻倍，
          必须扩大 ``delta`` 或改为全局扫描（网格/拉丁超立方）。
        - ``rank_flip_prob`` 高不代表模型差，只代表方案之间本来就很接近；
          论文里应当据此讨论"需要更精确的数据"而不是硬挑一个第一名。
        - 随机抽样结果依赖种子。本实现默认固定种子，请在论文里报告你用的抽样次数。

    参考:
        TOPSIS 权重灵敏度分析通例；多属性决策的稳健性检验。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    ben = _resolve_benefit(benefit, n)
    w, _ = _normalize_weights(weights, n)
    if delta < 0:
        raise ValueError("delta 必须非负")

    base = topsis(M, w, ben)
    base_rank = np.asarray(base["rank"])
    generator = rng(seed)

    rank_matrix = np.zeros((n_samples, m), dtype=int)
    for s in range(n_samples):
        perturbed = w * (1.0 + generator.uniform(-delta, delta, size=n))
        perturbed = np.clip(perturbed, 0.0, None)
        if perturbed.sum() <= 0:
            perturbed = w.copy()
        rank_matrix[s] = np.asarray(topsis(M, perturbed, ben)["rank"])

    flip_prob = (rank_matrix != base_rank[None, :]).mean(axis=0)
    best_keep = float((rank_matrix[:, int(np.argmin(base_rank))] == 1).mean())
    return {"base_rank": base_rank, "rank_flip_prob": flip_prob,
            "best_keep_prob": best_keep, "rank_matrix": rank_matrix,
            "base_closeness": base["closeness"]}


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。"""
    out: Dict[str, object] = {}

    # 1) AHP：Saaty 经典 3x3 判断矩阵
    A = [[1, 1 / 2, 4], [2, 1, 7], [1 / 4, 1 / 7, 1]]
    ahp = ahp_weights(A)
    out["ahp_weights"] = [round(float(v), 6) for v in ahp["weights"]]
    out["ahp_lambda_max"] = round(float(ahp["lambda_max"]), 6)
    out["ahp_CI"] = round(float(ahp["CI"]), 6)
    out["ahp_CR"] = round(float(ahp["CR"]), 6)
    out["ahp_consistent"] = bool(ahp["consistent"])

    # 2) 熵权：三个方案四个指标，权重和为 1
    X = [[5, 3, 8, 2], [7, 4, 6, 5], [6, 5, 9, 3], [8, 2, 7, 4]]
    ben = [True, True, False, True]
    ew = entropy_weights(X, benefit=ben)
    out["entropy_weights"] = [round(float(v), 6) for v in ew["weights"]]
    out["entropy_sum"] = round(float(ew["weights"].sum()), 9)

    # 3) CRITIC
    cw = critic_weights(X, benefit=ben)
    out["critic_weights"] = [round(float(v), 6) for v in cw["weights"]]
    out["critic_sum"] = round(float(cw["weights"].sum()), 9)

    # 4) TOPSIS：贴近度在 [0,1]，且第一名的贴近度最大
    t = topsis(X, ew["weights"], benefit=ben)
    out["topsis_closeness"] = [round(float(v), 6) for v in t["closeness"]]
    out["topsis_rank"] = [int(v) for v in t["rank"]]
    out["topsis_best_index"] = int(np.argmin(t["rank"]))

    # 5) VIKOR：Q 值最小的方案名次为 1
    vk = vikor(X, ew["weights"], benefit=ben, v=0.5)
    out["vikor_Q"] = [round(float(v), 6) for v in vk["Q"]]
    out["vikor_best_index"] = int(np.argmin(vk["Q"]))
    out["vikor_compromise"] = bool(vk["conditions"]["compromise_reached"])

    # 6) 灰关联：关联度在 (0,1]；成本型指标必须正向化，否则结论会反
    gr = grey_relational_grade(X, weights=ew["weights"], benefit=ben, rho=0.5)
    out["grey_grade"] = [round(float(v), 6) for v in gr["grade"]]
    out["grey_best_index"] = int(np.argmax(gr["grade"]))
    out["grey_grade_range_ok"] = bool(np.all(gr["grade"] > 0) and np.all(gr["grade"] <= 1 + 1e-9))
    out["grey_reference"] = [round(float(v), 6) for v in gr["reference"]]
    # 同一份数据不做正向化时，成本型指标会把最差方案推上来 —— 与正向化后的结论不同
    gr_raw = grey_relational_grade(X, weights=ew["weights"], rho=0.5)
    out["grey_best_index_no_benefit"] = int(np.argmax(gr_raw["grade"]))
    out["grey_benefit_changes_result"] = bool(out["grey_best_index"] != out["grey_best_index_no_benefit"])

    # 7) DEA：构造一个明显有效的 DMU 和一个明显无效的 DMU
    inputs = [[2, 3], [3, 4], [6, 9], [1, 1]]
    outputs = [[5, 6], [6, 7], [7, 8], [1, 1]]
    ccr = dea_ccr(inputs, outputs)
    bcc = dea_bcc(inputs, outputs)
    out["dea_ccr_efficiency"] = [round(float(v), 6) for v in ccr["efficiency"]]
    out["dea_bcc_efficiency"] = [round(float(v), 6) for v in bcc["efficiency"]]
    out["dea_efficient_count"] = int(ccr["efficient_count"])
    # BCC 效率必须 >= CCR 效率（约束更松）
    out["dea_bcc_ge_ccr"] = bool(np.all(bcc["efficiency"] >= ccr["efficiency"] - 1e-7))

    # 8) 模糊综合评判
    fz = fuzzy_comprehensive_eval([0.4, 0.3, 0.3],
                                  [[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3]])
    out["fuzzy_score"] = [round(float(v), 6) for v in fz["score"]]
    out["fuzzy_level"] = int(fz["level"])
    fz_scored = fuzzy_comprehensive_eval([0.4, 0.3, 0.3],
                                         [[0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3]],
                                         level_scores=[95, 80, 60])
    out["fuzzy_level_value"] = round(float(fz_scored["level_value"]), 6)

    # 9) TOPSIS 权重灵敏度
    sens = topsis_rank_sensitivity(X, ew["weights"], benefit=ben,
                                   delta=0.2, n_samples=200, seed=42)
    out["sens_best_keep_prob"] = round(float(sens["best_keep_prob"]), 6)
    out["sens_max_flip_prob"] = round(float(np.max(sens["rank_flip_prob"])), 6)

    return out
