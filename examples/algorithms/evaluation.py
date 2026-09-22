"""评价与决策模型：AHP、熵权、CRITIC、TOPSIS、VIKOR、灰关联、DEA、模糊综合。

本模块共 14 个公开函数，按用途分成五组：
- 定权：``ahp_weights``（层次分析法）/ ``entropy_weights``（熵权）/ ``critic_weights``（CRITIC）
  / ``combine_weights``（主客观组合赋权）；
- 综合评价与排序：``topsis`` / ``vikor`` / ``grey_relational_grade`` / ``fuzzy_comprehensive_eval``
  / ``rsr_evaluation``（秩和比）/ ``promethee_ii_ranking``（PROMETHEE II 净流排序）；
- 效率评价：``dea_ccr`` / ``dea_bcc``；
- 组合评价一致性：``kendall_w_concordance``（肯德尔和谐系数 W）；
- 稳健性诊断：``topsis_rank_sensitivity``（权重扰动下的排名稳定性）。

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

from statistics import NormalDist
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
    "rsr_evaluation",
    "promethee_ii_ranking",
    "dea_ccr",
    "dea_bcc",
    "fuzzy_comprehensive_eval",
    "kendall_w_concordance",
    "topsis_rank_sensitivity",
]

#: Saaty 随机一致性指标 RI（n = 1..15），用于判断矩阵一致性检验。
SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
            8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56,
            14: 1.57, 15: 1.59}

#: 卡方分布上侧 0.05 分位数（自由度 1..30），用于肯德尔和谐系数的显著性判断。
#: 与 RI 表一样是**内置查表值**：只覆盖 df <= 30，超出范围时返回 None 而不猜。
CHI2_005 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592,
            7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307, 11: 19.675, 12: 21.026,
            13: 22.362, 14: 23.685, 15: 24.996, 16: 26.296, 17: 27.587,
            18: 28.869, 19: 30.144, 20: 31.410, 21: 32.671, 22: 33.924,
            23: 35.172, 24: 36.415, 25: 37.652, 26: 38.885, 27: 40.113,
            28: 41.337, 29: 42.557, 30: 43.773}


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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """返回 1..n 的**平均名次**（并列取平均秩），1 对应最小值。

    与 :func:`_ranks` 的区别：``_ranks`` 是"竞赛排名法"（并列同名次，会跳号），
    这里用的是编秩常用的"平均秩"（并列占用的名次取平均，不跳号）。秩和比法与
    肯德尔和谐系数的教科书公式都建立在平均秩上，两者不能混用。
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    out = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(sorted_v[j + 1] - sorted_v[i]) <= 1e-12:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


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
        ``RI``、``n``、``converged``（幂法是否在 ``max_iter`` 步内收敛）、
        ``iterations``（实际迭代步数）。``n`` 超出内置 RI 表时额外返回 ``note`` 且
        ``CR``/``consistent``/``RI`` 为 ``None``。

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
        - **收敛性必须自己看 ``converged``**。迭代次数不够（或特征值靠得很近）时，
          ``weights`` 是"第 max_iter 步的中间结果"，仍然是一组和为 1 的正数，
          不会报错。例如 4 阶矩阵取 ``max_iter=1`` 时 ``CR`` 从 0.0359 变成 0.0431，
          权重也明显偏离：审计实测 ``[0.4461, 0.2900, 0.0967, 0.1673]`` vs
          收敛值 ``[0.4717, 0.2741, 0.1045, 0.1498]``。

    参考:
        Saaty (1980)《The Analytic Hierarchy Process》；Saaty 随机一致性指标 RI 表。
    """
    A = as_matrix(pairwise, "pairwise")
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("判断矩阵必须是方阵")
    if int(max_iter) < 1:
        raise ValueError("max_iter 必须 >= 1；迭代次数为 0 时幂法不会更新初始权重")
    if np.any(A <= 0):
        raise ValueError("判断矩阵元素必须为正")
    if np.any(np.abs(np.diag(A) - 1.0) > 1e-9):
        raise ValueError("判断矩阵对角线必须为 1")
    if np.any(np.abs(A * A.T - 1.0) > 1e-6):
        raise ValueError("判断矩阵必须正互反（a_ij * a_ji = 1）")

    w = np.ones(n) / n
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        w_new = A @ w
        norm = np.linalg.norm(w_new)
        if norm <= 0:
            raise ValueError("判断矩阵退化，幂法无法收敛")
        w_new = w_new / norm
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            converged = True
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
                "converged": bool(converged), "iterations": int(iterations),
                "note": f"n={n} 超出内置 RI 表（1..15），请自行查表判断一致性"}
    CR = 0.0 if RI == 0 else CI / RI
    return {"weights": w, "lambda_max": lambda_max, "CI": CI, "CR": CR,
            "consistent": bool(CR < 0.1), "RI": RI, "n": n,
            "converged": bool(converged), "iterations": int(iterations)}


def entropy_weights(X, benefit=None) -> Dict[str, object]:
    """熵权法：由数据本身的离散程度确定客观权重。

    参数:
        X: 决策矩阵，形状 (m 个方案, n 个指标)。
        benefit: 指标方向；成本型指标会先做正向化（取极差反转）。

    返回:
        dict，键为 ``weights``、``entropy``（各指标信息熵 e_j）、``divergence``（1 - e_j）、
        ``p``（比重矩阵，``pos / pos.sum(axis=0)``）、``normalized``（**只做了正向化**的矩阵，
        成本型列被 ``max - x`` 反转，没有再做任何归一化）、``note``（无区分度指标被置 0
        权重时的提示，否则为 ``None``）。

    算法:
        1. 成本型指标正向化：``x' = max - x``。
        2. 计算比重 ``p_ij = x'_ij / sum_i x'_ij``；某列正向化后全为 0（列和为 0）时
           该列 ``p`` 记 0、``e_j = 1``、``d_j = 0``。
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
          要在论文里讨论而不是假装没看见。注意**成本型常量列**正向化后整列变成 0
          （列和为 0），只有把它按"无区分度 → 权重 0"处理才与效益型常量列口径一致；
          旧版本对这种列会直接抛 ``ValueError``，审计后已改为与效益型常量列同样的处理，
          并在 ``note`` 里列出来。
        - 熵权是"谁差异大谁重要"，**不等于"谁业务上重要"**，不要用它替代专家判断。
        - 返回键 ``normalized`` 名字有历史遗留问题：它装的是**正向化后**的矩阵，
          **不是归一化矩阵**。真正归一化过的只有比重矩阵 ``p``。要"归一化后的决策矩阵"
          请自己按列做 min-max 或向量归一化。

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
    usable = col_sum[0] > 0
    if not np.any(usable):
        raise ValueError("所有指标列的和都为 0，无法计算比重")
    zero_cols = [j for j in range(n) if not usable[j]]
    P = np.zeros_like(pos)
    P[:, usable] = pos[:, usable] / col_sum[0, usable]

    k = 1.0 / np.log(m) if m > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        plnp = np.where(P > 0, P * np.log(P), 0.0)
    e = -k * plnp.sum(axis=0)
    # 列和为 0 的指标没有任何分布信息：按"熵最大、差异系数为 0"处理（权重 0）
    e = np.where(usable, np.clip(e, 0.0, 1.0), 1.0)
    d = 1.0 - e
    if d.sum() <= 0:
        raise ValueError("所有指标的信息熵都为 1（无区分度），无法赋权")
    w = d / d.sum()
    note = None
    if zero_cols:
        note = (f"以下指标正向化后列和为 0（无区分度），熵按 1 处理、权重置 0："
                f"{zero_cols}")
    return {"weights": w, "entropy": e, "divergence": d, "p": P, "normalized": pos,
            "note": note}


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
        - **常量列**（标准差 0）与其他列的 Pearson 相关系数在数学上无定义。本实现把
          ``NaN`` 一律替换为 0（"无线性关联"）再算冲突性；由于该列 ``sigma = 0``，
          它自身权重必为 0，但**与它配对的其他列**的冲突性会被这个人为约定影响。
          本次审计对含常量列的数据实测 ``corr`` 矩阵（0 行/列）即来自该约定——
          这是**保留行为**，论文里若出现常量列应当直接剔除该指标，而不是依赖这个默认值。

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
    """主客观权重组合（乘法合成 / 几何平均 / 线性加权）。

    参数:
        weight_sets: 若干组权重，形如 ``[[w1...], [w1...]]``，每组长度相同。
        method: ``"multiplicative"``（乘法合成，默认）、``"linear"``（线性加权）、
            ``"geometric"``（几何平均）。
        alphas: 仅 ``"linear"`` 使用；各权重组的系数，None 表示等权。系数必须非负且之和
            为正，否则抛 ``ValueError``（负系数会算出负权重）；之和不为 1 时会被就地归一化。
            注意：``method`` 不是 ``"linear"`` 时**传入 ``alphas`` 会直接报错**，
            而不是被静默忽略（静默忽略会让调用方以为自己调过参，实际上权重完全没变）。

    返回:
        dict，键为 ``weights``（组合权重，已归一化到和为 1）、``method``、
        ``note``（当前实现**恒为 None**，是给调用方预留的备注位，不要依赖它携带信息）。

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
        - ``linear`` 的系数 ``alphas`` 必须非负且之和为正：负系数会算出负权重，和恰为 0
          （例如 ``[0, 0]`` 或 ``[1, -1]``）会让权重变成 ``NaN`` 或无意义的巨大值。
          这两种输入现在都会抛 ``ValueError``；旧版本会静默返回这些结果
          （审计实测：权重组为 ``[[0.6, 0.4], [0.1, 0.9]]`` 时，旧实现给
          ``alphas=[2, -1] -> [1.1, -0.1]``（负权重）、``alphas=[0, 0] -> [nan, nan]``）。
        - **``alphas`` 传给了非 ``linear`` 的 ``method`` 也会报错**。旧版本会静默忽略它——
          调用方以为调了参，实际权重完全没变，这是很难自查的一类错误。
        - **博弈论组合赋权没有实现**。真正的博弈论组合赋权要解一个以"组合权重与各单一
          权重的偏差最小化"为目标的小型 LP/QP；本函数只提供 multiplicative / geometric /
          linear 三种**纯代数**合成。论文里如果写"采用博弈论组合赋权"，必须自己补上那一步，
          不能引用这个函数。

    参考:
        主客观组合赋权的常见做法（乘法合成、几何平均、线性加权）。博弈论组合赋权的目标
        函数与求解见相关多属性决策教材（本函数未实现）。
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

    if alphas is not None and method != "linear":
        raise ValueError(f"alphas 只对 method='linear' 有效，当前 method='{method}'；"
                         "请勿传入 alphas，或改用 'linear'")

    if method == "multiplicative":
        w = np.prod(W, axis=0)
    elif method == "geometric":
        w = np.exp(np.mean(np.log(np.clip(W, 1e-300, None)), axis=0))
    elif method == "linear":
        K = W.shape[0]
        a = np.ones(K) / K if alphas is None else as_vector(alphas, "alphas")
        if a.size != K:
            raise ValueError("alphas 长度必须等于权重组数")
        if np.any(a < -1e-12):
            raise ValueError("alphas（线性加权系数）不能为负：负系数会算出负权重")
        a = np.clip(a, 0.0, None)
        if a.sum() <= 0:
            raise ValueError("alphas 之和必须为正；全 0 或正负相消会让组合权重变成 NaN")
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
        ``normalized``（向量归一化矩阵 R）、``weighted``（加权规范矩阵 V）、
        ``positivized``（**只做了正向化**的矩阵，成本型列被 ``max - x`` 反转；
        向量归一化的是它而不是原始 ``X``）、``note``。

    算法:
        1. 正向化：成本型指标 ``x' = max - x``（先正向化，后归一化）。
        2. 向量归一化 ``r_ij = x'_ij / sqrt(sum_i x'_ij^2)``。
        3. 加权 ``v_ij = w_j r_ij``。
        4. 因为第 1 步之后**所有指标都是"越大越好"**，正理想解直接取 ``V`` 的列最大、
           负理想解取列最小（不需要再按指标方向分情况）。
        5. ``C_i = d_i^- / (d_i^+ + d_i^-)``。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        - **向量归一化必须做**。直接用原始数据算距离，量纲大的指标会独占权重。
          顺带一提：TOPSIS 的"正向化"要在归一化**之前**做（对成本型指标取倒数或极差反转），
          顺序颠倒会得到完全不同的排序。本实现用的是"先正向化再向量归一化"的口径，
          另一种常见口径是"直接对原始矩阵向量归一化，理想解按指标方向分别取 min/max"；
          两者结果不同（审计实测同一数据的贴近度前者为
          ``[0.3320, 0.9766, 0.0700, 0.6617]``、后者为
          ``[0.3154, 0.9156, 0.2306, 0.6062]``，本例排序恰好一致，但不能指望总是如此），
          论文里必须写明用的是哪一种。
        - 正负理想解是**从你给的方案集里选出来的**，不是绝对标准。加入一个很差的新方案，
          原有方案的贴近度会集体上升——所以不能跨数据集比较 C_i。
        - ``C_i`` 接近时排序不稳，务必配合 :func:`topsis_rank_sensitivity` 做扰动分析。
        - 距离用欧氏范数隐含"各指标可替代"的假设；指标间高度相关时考虑马氏距离。
        - 只有一个方案（m=1）时正负理想解都是它自己，``d+ = d- = 0``，
          本实现按 ``C = 0.5`` 返回（约定值，不是真实贴近度）。

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
        ``rank``（**1 为最好**，即 ``Q`` 最小者排第 1）、``conditions``（是否同时满足
        可接受优势与可接受决策可靠性）、``note``。

    算法:
        ``S_i = sum_j w_j (f*_j - f_ij) / (f*_j - f^-_j)``；
        ``R_i = max_j [同上]``；
        ``Q_i = v (S_i - S*) / (S^- - S*) + (1 - v)(R_i - R*) / (R^- - R*)``。
        排名用 ``_ranks(-Q)``：``Q`` 越小越好，所以要先把 ``Q`` 取负再送进"1 为最好"的
        排名函数。

    复杂度:
        时间 O(mn) / 空间 O(mn)。

    陷阱:
        - VIKOR 的结论**必须做两项检验**：``Q(A(1)) - Q(A(2)) >= 1/(m-1)``（可接受优势），
          且 A(1) 在 S 或 R 中也排第一（可接受决策可靠性）。只报 Q 值排序是不完整的，
          评委很可能会追问。本实现把结论放在 ``conditions`` 里。
        - **``Q`` 越小越好，``rank`` 越大越差**。这两个方向相反是 VIKOR 最容易写错的地方：
          旧版本直接对 ``Q`` 调用"1 为最好"的排名函数，``rank`` 完全反了（审计实测：
          ``Q = [0.7403, 0.0000, 1.0000, 0.3674]`` 时返回 ``rank = [2, 4, 1, 3]``，
          把 ``Q`` 最大的第 3 个方案排成了第 1 名；正确结果是 ``[3, 1, 4, 2]``）。
          论文里凡是要给出名次，请以 ``Q`` 或 ``rank`` 中的**一个**为准并说明口径。
        - 某个指标在所有方案上取值相同时分母为 0，需要特殊处理（本实现跳过该指标并记入 note）。
        - ``v`` 的取值会改变排序，论文里要写明取值并做敏感性分析。

    参考:
        Opricovic & Tzeng (2004) VIKOR 折衷排序法。
    """
    M = as_matrix(X, "X")
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

    rank = _ranks(-Q)
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
        dict，键为 ``efficiency``（各 DMU 效率，取值 [0, 1]，1 表示 DEA 有效）、
        ``rank``、``lambda``（各 DMU 的参考权重，非零项即该 DMU 的"标杆"）、
        ``peers``（每个 DMU 的参考 DMU 下标）、``statuses``、``efficient_count``
        （效率为 1 的 DMU 个数）。

    算法:
        对每个 DMU 解输入导向包络 LP：``min theta``
        ``s.t. sum_j lambda_j x_ij <= theta x_ik``，``sum_j lambda_j y_rj >= y_rk``，
        ``lambda >= 0``。用 :func:`simplex_lp` 求解。

    复杂度:
        时间 O(n_dmu * LP(n_dmu)) / 空间 O(n_dmu^2)。

    陷阱:
        - **效率区间是 [0, 1] 而不是 (0, 1]**：若某个 DMU 的所有产出都是 0，则
          ``sum_j lambda_j y_rj >= 0`` 对任意 lambda 都成立，最优解为 ``theta = 0``，
          该 DMU 的效率就是 0.0（审计实测：投入 ``[[1, 2]]``、产出 ``[[0], [1]]`` 时
          ``efficiency = [0.0, 1.0]``，两个 LP 的 status 都是 "optimal"）。这不是
          "无效"，而是"零产出"这种退化输入本身没有效率可言，应在数据阶段排除。
        - **DEA 效率 1 的 DMU 可能有很多个**，CCR 无法再区分它们。需要区分时请用超效率
          DEA（Anderson-Peterson）或在论文里明确指出"多个 DMU 同为 DEA 有效"。
        - **投入产出指标数之和不应超过 DMU 数量的约 1/3**（经验法则）。指标太多时几乎所有
          DMU 都会变成有效，结论没有信息量。
        - 投入/产出的**方向不能搞反**：把"成本"放进 outputs 会得到完全错误的结论。
        - 负值不允许（DEA 要求非负）。含负值要先做平移，并在论文里说明。
        - 本实现是**输入导向**（在产出不变的前提下最小化投入）。输出导向会得到不同数值，
          注意与文献口径一致。
        - ``statuses`` 里出现非 "optimal" 的项时，该 DMU 的效率被**静默记为 0**。
          正式报告前请检查 ``statuses``：它非 "optimal" 通常意味着模型退化或迭代上限，
          此时的 0 不代表真实的效率水平。

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
        dict，键为 ``score``（综合隶属度向量 B）、``level``（按最大隶属度原则给出的等级下标）、
        ``level_value``（若提供 ``level_scores`` 则给出加权得分，否则为 ``None``）、
        ``normalized``（``B / B.sum()``；``B`` 全为 0 时原样返回）、``note``（提示字符串或
        ``None``：权重被自动归一化、或 ``R`` 行和偏离 1 时会给出说明，**两类提示同时成立时
        用 "；" 连接**，不会互相覆盖）。

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
        - 权重与隶属度的校验口径**不一样**：``weights`` 只校验**非负**，不要求 ≤ 1，和不为 1 时
          自动归一化并把提示写进 ``note``；``membership`` 才强制落在 ``[0, 1]``（越界即
          ``raise ValueError``）。
        - ``membership`` 每行的和理论上应为 1，但本实现**不做强制归一化**，只在行和偏离 1 超过
          1e-6 时把提示写进 ``note``。也就是说，漏写一个评语等级不会被拦下，只会得到一句提示。
        - 上面两类提示是**并列**的。旧版本用 ``hint if hint else note`` 返回，当权重需要归一化
          **且**行和也偏离 1 时，权重归一化的那条提示会被静默吞掉（审计实测：权重
          ``[2, 2, 2]`` + 行和不为 1 的 R 只返回行和提示）。现在两类提示会同时出现在 ``note`` 里。

    参考:
        汪培庄（1983）模糊综合评判；Zadeh (1965) 模糊集合。
    """
    R = as_matrix(membership, "membership")
    w, note = _normalize_weights(weights, R.shape[0])
    if np.any(R < -1e-12) or np.any(R > 1 + 1e-12):
        raise ValueError("隶属度必须在 [0, 1] 内")

    row_sums = R.sum(axis=1)
    hints = []
    if note:
        hints.append(note)
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        hints.append("部分因素的各等级隶属度之和不为 1，请确认是否遗漏等级（本实现不做强制归一化）")

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
            "level_value": level_value, "note": "；".join(hints) if hints else None}


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
        ``best_keep_prob``（原第一名保持第一的频率）、``rank_matrix``（每次抽样的名次矩阵，
        形状 ``(n_samples, m)``）、``base_closeness``（未扰动时的 TOPSIS 贴近度，方便对照
        "名次没变"时贴近度实际变动的幅度）。

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
        - 随机抽样结果依赖种子。本实现默认固定种子（``seed=None`` 时用
          :data:`_common.DEFAULT_SEED`，因此默认结果是**可复现**的），请在论文里报告你用的
          抽样次数。
        - ``rank_flip_prob`` 只统计"名次号是否变了"，**并列名次口径变化不算翻转**。
          贴近度可能明显变化而名次完全不变，这正是 ``base_closeness`` 要与
          ``rank_matrix`` 一起看的原因。

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
# 秩和比（RSR）综合评价
# --------------------------------------------------------------------------

def rsr_evaluation(X, benefit=None, weights=None,
                   n_levels: int = 3) -> Dict[str, object]:
    """秩和比（RSR）综合评价：只按名次合成的综合评价与分档。

    参数:
        X: 决策矩阵，形状 (m 个评价对象, n 个指标)。
        benefit: 长度 n 的 bool 序列，``True`` 表示正向（越大越好）；``None`` 表示全部正向。
        weights: 可选的指标权重，长度 n。``None`` 表示等权——此时
            ``RSR_i = sum_j R_ij / (m * n)``，就是教科书上的原始公式。
        n_levels: 分档档数，默认 3（例如"优 / 中 / 差"）。只有 ``m >= 4`` 且各对象的
            RSR 不全相同时才会做概率单位分档，否则 ``probit`` / ``probit_fit`` /
            ``distribution`` 一律返回 ``None``。

    返回:
        dict，键为 ``rank_matrix``（编秩矩阵，**秩越大越好**：正向指标取平均秩，成本型
        指标取 ``m + 1 - 平均秩``）、``rsr``（秩和比，越大越好）、``rank``（**1 为最好**）、
        ``probit``（概率单位，条件不满足时为 ``None``）、``probit_fit``（概率单位对 RSR 的
        线性回归 ``{"slope", "intercept", "r2"}``）、``distribution``（分档结果，
        **1 表示最好的一档**）、``note``（口径说明）。

    算法:
        1. 编秩：第 j 列取**平均秩**（并列取平均、不跳号，1 表示该列最差）；成本型指标翻转为
           ``m + 1 - r``，于是所有列都统一成"秩大者优"。注意 :func:`_ranks` 是竞赛排名法
           （并列会跳号），**不能**用于 RSR 的秩和公式，两者混用会算出错的 RSR。
        2. 合成：``RSR_i = (sum_j w_j R_ij) / (m * sum_j w_j)``，取值 (0, 1]；等权时退化为
           ``sum_j R_ij / (m n)``。
        3. 分档（仅当 m >= 4 且 RSR 不全相同）：把 RSR 升序排列，第 k 个（k 从 1 起）的
           累计频率取 ``p_k = (k - 0.5) / m``，概率单位 ``z_k = Phi^{-1}(p_k)``；对 ``z``
           关于 ``RSR`` 做最小二乘直线回归得 ``z_fit``，再与分档界
           ``Phi^{-1}(k / n_levels)``（k = 1..n_levels-1）比较得档位。``z`` 越大越好，
           故回归值越大档位编号越小；返回前已翻转为"1 为最好"。

    复杂度:
        时间 O(m log m + m n) / 空间 O(m n)。

    陷阱:
        - **RSR 只用名次，丢掉了量纲信息**：原始数据里"巨大领先"和"微弱领先"在 RSR 中完全
          相同，指标间差距悬殊时 RSR 会把差距抹平。这时候要么改用 TOPSIS/PROMETHEE，
          要么把原始数据一并列进论文。
        - **并列名次会削弱分辨力**。大量并列时多个对象的秩和相同，RSR 相同，只能并列同名次，
          排序失去意义；请检查 ``rank_matrix`` 里的并列情况。
        - **成本型指标的秩必须翻转**（本实现已经翻转）。如果排序整体反向，先检查 ``benefit``
          是否传对——这与灰关联"不正向化结论就反"是同一类错误。
        - **概率单位分档依赖正态假设，且自由度很低**。``m < 4`` 时回归几乎没有意义，本实现
          直接不做分档并写进 ``note``，绝不硬凑档位。``m`` 略大于 4 时也请以 ``probit_fit``
          里的 ``r2`` 为准：``r2`` 很低说明回归线不显著，此时档位结论不可信。
        - **档数 ``n_levels`` 是主观选择**，同一份数据换档数档位编号就变，必须在论文里写明
          你用的是几档以及为什么。
        - 分档边界附近的样本很脆弱：``z_fit`` 恰好落在分档界上时，数据的微小改动就会改档。

    参考:
        田凤调（1988）秩和比法及其在医学统计中的应用；RSR 法的原始文献。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    if m < 1:
        raise ValueError("X 至少要有一个评价对象")
    n_levels = int(n_levels)
    if n_levels < 2:
        raise ValueError("n_levels 至少为 2")
    ben = _resolve_benefit(benefit, n)
    if weights is None:
        w = np.full(n, 1.0 / n)
        note = "未提供权重，按等权处理（此时 RSR 等价于 sum_j R_ij / (m n)）"
    else:
        w, wn = _normalize_weights(weights, n)
        note = wn if wn else "权重已按和为 1 归一化"

    # 编秩：正向指标取平均秩，成本型翻转为 m + 1 - r，使各列都是"秩大者优"
    rank_matrix = np.zeros((m, n), dtype=float)
    for j in range(n):
        r = _average_ranks(M[:, j])
        if not bool(ben[j]):
            r = m + 1.0 - r
        rank_matrix[:, j] = r

    rsr = (rank_matrix @ w) / (m * float(w.sum()))

    probit = None
    probit_fit = None
    distribution = None
    if m < 4:
        note += "；m < 4，概率单位回归自由度不足，不做分档"
    elif float(np.ptp(rsr)) <= 1e-12:
        note += "；各对象的 RSR 完全相同，不做分档"
    else:
        order = np.argsort(rsr, kind="mergesort")
        pos = np.empty(m, dtype=float)
        pos[order] = np.arange(m, dtype=float)
        # p_k = (k - 0.5) / m，k 从 1 起；秩越小（越差）累计频率越低
        p = (pos + 0.5) / float(m)
        dist = NormalDist()
        probit = np.array([dist.inv_cdf(float(v)) for v in p], dtype=float)
        slope, intercept = np.polyfit(rsr, probit, 1)
        z_fit = slope * rsr + intercept
        ss_tot = float(np.sum((probit - probit.mean()) ** 2))
        ss_res = float(np.sum((probit - z_fit) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
        bounds = np.array([dist.inv_cdf(k / float(n_levels))
                           for k in range(1, n_levels)], dtype=float)
        level = np.searchsorted(bounds, z_fit) + 1  # 1 = 最差档
        distribution = (n_levels + 1 - level).astype(int)  # 翻转为 1 = 最好档
        probit_fit = {"slope": float(slope), "intercept": float(intercept),
                      "r2": float(r2)}
        note += (f"；已按 {n_levels} 档做概率单位分档，distribution 中 1 表示最好的一档"
                 f"（r2 = {r2:.4f}）")

    return {"rank_matrix": rank_matrix, "rsr": rsr, "rank": _ranks(rsr),
            "probit": probit, "probit_fit": probit_fit,
            "distribution": distribution, "note": note}


# --------------------------------------------------------------------------
# PROMETHEE II 净流排序
# --------------------------------------------------------------------------

def promethee_ii_ranking(X, weights, benefit=None, preference: str = "linear",
                         q: float = 0.0, p=None) -> Dict[str, object]:
    """PROMETHEE II：用两两比较的净流给出完全排序。

    参数:
        X: 决策矩阵，形状 (m 个方案, n 个指标)，要求 ``m >= 2``。
        weights: 指标权重，长度 n。
        benefit: 指标方向，``True`` 表示越大越好。
        preference: 偏好函数类型，``"usual"``（``P(d) = 1`` 当 ``d > 0``，否则 0）或
            ``"linear"``（``P(d) = clip((d - q) / (p - q), 0, 1)``）。
        q: 无差异阈值（``d <= q`` 视为无差别）。仅 ``linear`` 使用。
        p: 严格偏好阈值（``d >= p`` 时偏好度为 1）。``None`` 表示按各指标**正向化后数据的
            极差**自动取值。仅 ``linear`` 使用。

    返回:
        dict，键为 ``phi_plus``（正流，越大越好）、``phi_minus``（负流，越小越好）、
        ``phi_net``（净流，**越大越好**，全部净流之和为 0）、``rank``（**1 为最好**）、
        ``pi``（加权后的总体偏好矩阵，形状 (m, m)，``pi[a, b]`` 表示 a 优于 b 的程度）、
        ``pairwise_preference``（未加权的逐指标偏好度，形状 (n, m, m)）、
        ``positivized``（正向化后的矩阵）、``note``。

    算法:
        1. 正向化：成本型指标取 ``max - x``，使所有指标统一成越大越好。
        2. 逐指标两两比较 ``d = x_aj - x_bj``，按偏好函数得 ``P_j(a, b) ∈ [0, 1]``。
        3. 加权合成 ``pi(a, b) = sum_j w_j P_j(a, b)``，对角线置 0。
        4. 流：``phi_plus_a = sum_b pi(a, b) / (m - 1)``，
           ``phi_minus_a = sum_b pi(b, a) / (m - 1)``，``phi_net = phi_plus - phi_minus``。
           PROMETHEE II 按 ``phi_net`` 排序，所以排名用 ``_ranks(phi_net)``（净流越大越好）。

    复杂度:
        时间 O(n m^2) / 空间 O(n m^2)（``pairwise_preference`` 是三维数组，m 大时注意内存）。

    陷阱:
        - **PROMETHEE II 会给出"完全排序"，但它掩盖了不可比性**。PROMETHEE I 允许两个方案
          互不支配（部分序），II 强行用净流做差，把"不可比"和"相等"混为一谈。结论里出现
          净流非常接近的两个方案时，请不要断言谁更好。
        - **净流之和恒为 0**，所以它是相对量：不能跨数据集比较 ``phi_net``，也不能说
          "某方案得分为 0.44"就代表它绝对好。
        - **阈值 ``q`` / ``p`` 是主观的，且必须与偏好函数匹配**。``usual`` 型偏好函数不含
          阈值，传 ``q != 0`` 或 ``p`` 会被本实现直接报错，而不是静默忽略。
        - ``p <= q`` 时线性偏好函数的分母非正，公式失效。**显式传入**这样的 ``p`` 会报错；
          而 ``p=None`` 时，取值恒定的指标极差为 0（``p = q = 0``），本实现把这类指标的
          偏好度恒置 0 并写进 ``note``，可视为该指标不参与区分。
        - 本实现只提供 ``usual`` 与 ``linear`` 两种偏好函数。文献里还有 U 型、V 型、
          高斯型等；如果你的题目要求其中某种，请在此实现基础上扩展，不要假装它是 linear。
        - ``pi`` 单向性假设：``pi(a, b)`` 与 ``pi(b, a)`` 独立计算，两者之和不必为 1。

    参考:
        Brans & Vincke (1985) PROMETHEE；Brans, Vincke & Mareschal (1986) 排序方法比较。
    """
    M = as_matrix(X, "X")
    m, n = M.shape
    if m < 2:
        raise ValueError("PROMETHEE 至少需要 2 个方案")
    if preference not in ("usual", "linear"):
        raise ValueError(f"未知偏好函数：{preference}（仅支持 'usual' / 'linear'）")
    ben = _resolve_benefit(benefit, n)
    w, note = _normalize_weights(weights, n)

    if preference == "usual" and (float(q) != 0.0 or p is not None):
        raise ValueError("usual 型偏好函数没有无差异/严格偏好阈值，请勿传 q 或 p")

    qq = np.array([q] * n, dtype=float) if np.isscalar(q) else np.asarray(q, dtype=float).ravel()
    if qq.size != n:
        raise ValueError(f"q 的长度 {qq.size} 与指标数 {n} 不一致")
    if np.any(qq < 0):
        raise ValueError("无差异阈值 q 不能为负")

    # 正向化：成本型取 max - x，之后所有指标都是越大越好
    pos = M.copy()
    for j in range(n):
        if not ben[j]:
            pos[:, j] = M[:, j].max() - M[:, j]

    auto_p = p is None
    if auto_p:
        pp = pos.max(axis=0) - pos.min(axis=0)
        degenerate = pp <= qq + 1e-15
    else:
        pp = np.array([p] * n, dtype=float) if np.isscalar(p) else np.asarray(p, dtype=float).ravel()
        if pp.size != n:
            raise ValueError(f"p 的长度 {pp.size} 与指标数 {n} 不一致")
        if np.any(pp <= qq):
            raise ValueError("严格偏好阈值 p 必须大于无差异阈值 q")
        degenerate = np.zeros(n, dtype=bool)

    pairwise = np.zeros((n, m, m), dtype=float)
    for j in range(n):
        d = pos[:, j][:, None] - pos[:, j][None, :]
        if preference == "usual":
            pairwise[j] = (d > 0).astype(float)
        elif degenerate[j]:
            pairwise[j] = 0.0
        else:
            pairwise[j] = np.clip((d - qq[j]) / (pp[j] - qq[j]), 0.0, 1.0)

    pi = np.tensordot(w, pairwise, axes=(0, 0))
    np.fill_diagonal(pi, 0.0)
    phi_plus = pi.sum(axis=1) / (m - 1)
    phi_minus = pi.sum(axis=0) / (m - 1)
    phi_net = phi_plus - phi_minus

    if auto_p and bool(np.any(degenerate)):
        idx = [int(j) for j in range(n) if degenerate[j]]
        note = (note + "；" if note else "") + \
            f"指标 {idx}（0 起编号）正向化后极差不超过 q，偏好度恒为 0，不参与区分"

    return {"phi_plus": phi_plus, "phi_minus": phi_minus, "phi_net": phi_net,
            "rank": _ranks(phi_net), "pi": pi, "pairwise_preference": pairwise,
            "positivized": pos, "note": note}


# --------------------------------------------------------------------------
# 组合评价一致性：肯德尔和谐系数
# --------------------------------------------------------------------------

def kendall_w_concordance(rankings, as_scores: bool = False,
                          higher_is_better: bool = True) -> Dict[str, object]:
    """肯德尔和谐系数 W：判断多个评价者/多种方法的排序是否一致。

    参数:
        rankings: 形状 (k 个评价者/评价方法, m 个评价对象) 的矩阵。
        as_scores: ``False``（默认）表示 ``rankings`` 里的数**已经是名次**，1 表示最好；
            ``True`` 表示它是**得分/原始值**，函数会先转成平均秩。
        higher_is_better: 仅当 ``as_scores=True`` 时有效。``True`` 表示得分越大名次越好。

    返回:
        dict，键为 ``W``（和谐系数，取值 [0, 1]，1 表示完全一致）、``chi2``（卡方统计量
        ``k (m - 1) W``）、``df``（自由度 ``m - 1``）、``S``（各对象秩和对其均值的离差平方和）、
        ``rank_sums``（各对象的秩和）、``rank``（**1 为最好**的**综合排序**：秩和最小者第 1）、
        ``tie_correction``（并列修正量 ``sum_k sum_t (t^3 - t)``）、``critical_value``
        （df <= 30 时给出 0.05 显著性水平下的卡方临界值，否则 ``None``）、``significant``
        （``chi2 > critical_value``；临界值缺失时为 ``None``）、``note``。

    算法:
        ``S = sum_i (R_i - R_bar)^2``，其中 ``R_i`` 是第 i 个对象的秩和；
        含并列时用修正公式
        ``W = 12 S / (k^2 (m^3 - m) - k * sum_k sum_t (t^3 - t))``，
        无并列时退化为 ``W = 12 S / (k^2 (m^3 - m))``。
        显著性检验用 ``chi2 = k (m - 1) W``（df = m - 1），与 :data:`CHI2_005` 查表值比较。
        名次必须用**平均秩**：并列时如果给"竞赛排名"（会跳号），``S`` 会被系统性高估。

    复杂度:
        时间 O(k m log m) / 空间 O(k m)。

    陷阱:
        - **W 高不代表评价准确，只代表评价者/方法之间一致**。k 个方法全都用了错的权重口径，
          它们照样可以高度一致——W 检验的是"同向性"，不是"正确性"。论文里应当把
          "一致性检验"和"方法本身合理性"分开论述。
        - **``chi2 = k (m - 1) W`` 是大样本近似**。``m`` 很小时（经验上 m < 7）应当改用
          精确分布表或 Friedman 检验的精确 p 值，卡方近似会偏乐观。本实现返回临界值但
          不做 p 值近似，就是为了不给你一个假的精确度。
        - ``critical_value`` **只覆盖 df <= 30**（``CHI2_005`` 的查表范围）。超出范围时本实现
          返回 ``None`` 并把 ``significant`` 也置为 ``None``，而不是用某个公式外推——
          ``significant is None`` 时请自行查表，不要当成"不显著"。
        - **输入到底是"名次"还是"得分"必须说清楚**。默认 ``as_scores=False``，即按名次解释
          1 为最好。把得分当名次传进来（或反过来）会得到一个看似正常、实则毫无意义的 W。
        - **并列必须修正**。本实现自动计算 ``tie_correction``；如果某项为 0 而数据里确实有
          并列，说明并列没有被识别（检查你的名次是怎么排出来的）。
        - **``as_scores=False`` 时每一行的名次必须是"平均秩"**。``W`` 的数学上界是 1，如果你
          把并列写成竞赛排名（``[1, 1, 3, 4]`` 而不是 ``[1.5, 1.5, 3, 4]``），
          ``tie_correction`` 会算错、``S`` 会被虚高，``W`` 甚至能超过 1（审计实测：
          ``[[1, 1, 3, 4], [2, 1, 4, 3]]`` 会得到 ``W = 1.0921``，明显不可能）。
          本实现遇到 ``W > 1`` 会直接 ``raise ValueError``，而不是返回一个看似正常的数。
          要省事就传 ``as_scores=True``，让函数自己用 :func:`_average_ranks` 编秩。
        - ``rank`` 是**综合排序**，与 W 是两件事：W 回答"是否一致"，``rank`` 回答"综合起来谁
          最好"。W 不显著时 ``rank`` 依然可以算出来，但它没有统计意义上的支撑。

    参考:
        Kendall & Babington Smith (1939) 和谐系数；Friedman (1937) 秩和检验。
    """
    R = as_matrix(rankings, "rankings")
    k, m = R.shape
    if k < 2:
        raise ValueError("至少需要 2 个评价者/评价方法才能谈一致性")
    if m < 2:
        raise ValueError("至少需要 2 个评价对象")
    if as_scores:
        rows = []
        for i in range(k):
            rows.append(_average_ranks(-R[i] if higher_is_better else R[i]))
        ranks = np.vstack(rows)
    else:
        ranks = R.astype(float).copy()

    rank_sums = ranks.sum(axis=0)
    r_bar = float(rank_sums.mean())
    S = float(np.sum((rank_sums - r_bar) ** 2))

    # 并列修正：每行内同值成组，T = sum (t^3 - t)
    tie_total = 0.0
    for i in range(k):
        _, counts = np.unique(ranks[i], return_counts=True)
        tie_total += float(np.sum(counts ** 3 - counts))

    denom = k * k * (m ** 3 - m) - k * tie_total
    if denom <= 1e-12:
        raise ValueError("和谐系数分母非正（名次全部并列？），无法计算 W")
    W = 12.0 * S / denom
    if W > 1.0 + 1e-9:
        raise ValueError(
            f"和谐系数 W = {W:.6f} 超过 1，输入不合法：最常见的原因是某行的并列名次没有取"
            "平均秩（并列第一应写成 [1.5, 1.5, 3, ...] 而不是 [1, 1, 3, ...]）；"
            "也可能 as_scores 传反了")
    chi2 = k * (m - 1) * W
    df = m - 1
    critical = CHI2_005.get(df)
    significant = bool(chi2 > critical) if critical is not None else None

    if critical is None:
        note = f"df = {df} 超出查表范围（CHI2_005 只到 30），请自行查表判断显著性"
    else:
        note = (f"chi2 = {chi2:.6f} 与 0.05 临界值 {critical} 比较："
                + ("达到显著" if significant else "未达到显著")
                + "（卡方近似，m 较小时偏乐观）")

    return {"W": float(W), "chi2": float(chi2), "df": int(df), "S": float(S),
            "rank_sums": rank_sums, "rank": _ranks(-rank_sums),
            "tie_correction": float(tie_total), "critical_value": critical,
            "significant": significant, "note": note}


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

    # 10) 秩和比（RSR）：等权合成，m = 4 触发概率单位分档
    rsr = rsr_evaluation(X, benefit=ben)
    out["rsr_rank_matrix"] = [[round(float(v), 6) for v in row] for row in rsr["rank_matrix"]]
    out["rsr_values"] = [round(float(v), 6) for v in rsr["rsr"]]
    out["rsr_rank"] = [int(v) for v in rsr["rank"]]
    out["rsr_best_index"] = int(np.argmin(rsr["rank"]))
    out["rsr_probit"] = [round(float(v), 6) for v in rsr["probit"]]
    out["rsr_probit_fit_r2"] = round(float(rsr["probit_fit"]["r2"]), 6)
    out["rsr_probit_slope_positive"] = bool(rsr["probit_fit"]["slope"] > 0)
    out["rsr_distribution"] = [int(v) for v in rsr["distribution"]]
    # m = 3 时自由度不足，必须明确不做分档（返回 None）而不是硬凑档位
    rsr_small = rsr_evaluation([[5, 3], [7, 4], [6, 5]])
    out["rsr_small_no_distribution"] = bool(rsr_small["distribution"] is None)

    # 11) PROMETHEE II：等权、线性偏好函数（p 默认取各指标极差）
    pr = promethee_ii_ranking(X, [1, 1, 1, 1], benefit=ben)
    out["promethee_phi_plus"] = [round(float(v), 6) for v in pr["phi_plus"]]
    out["promethee_phi_minus"] = [round(float(v), 6) for v in pr["phi_minus"]]
    out["promethee_phi_net"] = [round(float(v), 6) for v in pr["phi_net"]]
    out["promethee_rank"] = [int(v) for v in pr["rank"]]
    out["promethee_best_index"] = int(np.argmin(pr["rank"]))
    # 净流之和恒为 0（相对量，不能跨数据集比较）
    out["promethee_net_sum_zero"] = bool(abs(float(pr["phi_net"].sum())) < 1e-12)
    pr_usual = promethee_ii_ranking(X, [1, 1, 1, 1], benefit=ben, preference="usual")
    out["promethee_usual_rank"] = [int(v) for v in pr_usual["rank"]]

    # 12) 肯德尔和谐系数：三个评价者给出的名次（1 为最好）
    kw = kendall_w_concordance([[1, 2, 3, 4], [2, 1, 4, 3], [1, 3, 2, 4]])
    out["kendall_W"] = round(float(kw["W"]), 6)
    out["kendall_chi2"] = round(float(kw["chi2"]), 6)
    out["kendall_df"] = int(kw["df"])
    out["kendall_S"] = round(float(kw["S"]), 6)
    out["kendall_rank_sums"] = [round(float(v), 6) for v in kw["rank_sums"]]
    out["kendall_rank"] = [int(v) for v in kw["rank"]]
    out["kendall_critical_value"] = float(kw["critical_value"])
    out["kendall_significant"] = bool(kw["significant"])
    out["kendall_tie_correction"] = round(float(kw["tie_correction"]), 6)
    # 完全一致时 W 必为 1
    kw_perfect = kendall_w_concordance([[1, 2, 3, 4], [1, 2, 3, 4]])
    out["kendall_W_perfect"] = round(float(kw_perfect["W"]), 6)
    # 含并列时必须走修正公式：平均秩 [1.5, 1.5, 3, 4] 的并列修正量为 6
    kw_tie = kendall_w_concordance([[1.5, 1.5, 3, 4], [2, 1, 4, 3]])
    out["kendall_W_with_tie"] = round(float(kw_tie["W"]), 6)
    out["kendall_tie_correction_value"] = round(float(kw_tie["tie_correction"]), 6)

    return out
