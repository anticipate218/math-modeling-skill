#!/usr/bin/env python3
"""数学建模竞赛论文配图库生成脚本（全部为原创合成数据，不含任何第三方素材）。

一键生成按题型组织的 16 张论文级配图，输出到仓库 `assets/gallery/`。
每一张图的数据都由本脚本用**固定随机种子**现场合成，不读取任何外部文件、
不下载任何图片、不复制任何他人论文中的图表。

用法:
    python scripts/make_figures.py                    # 输出到 assets/gallery/
    python scripts/make_figures.py --out out/figures  # 指定输出目录
    python scripts/make_figures.py --dpi 300          # 指定分辨率（投稿常用 300）
    python scripts/make_figures.py --only pareto      # 只重生成文件名含该子串的图
    python scripts/make_figures.py --self-test        # 只跑数值自检，不出图

依赖:
    仅 numpy + matplotlib。CI 只跑 `--self-test`（纯数值内核，只需 numpy），
    **不出图**——渲染依赖 matplotlib 与中文字体，放 CI 里既不稳也不必要，
    因此出图放在本机执行，并在这里对缺库做显式检测、给出可照抄的安装命令，
    而不是抛一条裸 ImportError。配套的 `scripts/check_palette.py` 会直接读本文件
    的设计令牌做配色体检，所以即使不出图，配色也能在 CI 里被验证。

退出码: 0 成功；1 有图生成失败或疑似空白；2 缺少依赖。
"""

from __future__ import annotations

import argparse
import heapq
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- 全局常量

SEED = 20240517            # 全脚本唯一随机种子来源，保证图可复现
FIG_DPI_DEFAULT = 160      # 落在任务要求的 150-170 区间内
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "assets" / "gallery"

#: 中文字体候选；顺序即优先级，DejaVu Sans 仅作最后兜底（不含汉字，会出方框）
CJK_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

# ------------------------------------------------------------------ 设计令牌
#: Okabe-Ito 色盲友好配色的 7 个彩色（**已剔除黑色**）。
#: 为什么保留 Okabe-Ito 而不是换成"更好看"的 seaborn/ColorBrewer：
#: 用 Machado(2009) 严重度 1.0 矩阵在**线性 sRGB** 下模拟三类色盲，再算 CIELAB ΔE*ab
#: 的两两最小距离，本组是候选里最稳的（最差 16.1）。常见"论文风"调色板的实测值：
#: seaborn deep 2.7、ColorBrewer Set2 2.5、tab10 4.6 —— 在红色盲下几乎并成一块，
#: 属于可及性倒退。ΔE≥10 才算"能分辨"，≥20 才算"轻松分辨"。
#: 黑色另作参考线/文字用（见 INK），不参与数据着色，否则会和坐标轴混淆。
#: `#F0E442`（黄）与白色的对比度仅 1.32，细线在白底上几乎看不见，故只留作最后兜底。
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442"]

#: 默认颜色循环：前 6 个彩色。相邻序号色相差异最大，便于"多条曲线按顺序取色"时区分。
CYCLE = OKABE_ITO[:6]

#: 语义化颜色角色。学术配图的关键不是"颜色多好看"，而是**非数据元素要退到背景里**：
#: 坐标轴/文字用近黑而不是纯黑（纯黑在小字号下显得生硬），网格用极浅灰，
#: 参考线用中性灰，高亮只留给"要读者看的那一个东西"。
INK = "#262626"          # 轴线、刻度、正文文字
INK_SOFT = "#5C5C5C"     # 次级标注、注释箭头
INK_MUTED = "#9A9A9A"    # 参考线、被支配解等"背景数据"
GRID = "#DCDCDC"         # 网格线
PANEL_BG = "#F4F4F5"     # 外推期/置信带等的浅底
EDGE = "#FFFFFF"         # 标记描边：白描边让重叠点仍能分辨（比黑描边更现代）
CMAP_SEQ = "viridis"     # 顺序型：感知均匀 + 色盲安全（替换掉非均匀的 YlGnBu）
CMAP_DIV = "RdBu_r"      # 发散型：以 0 为中心的相关系数

#: 颜色之外的第二、第三编码通道（WCAG 2.1 1.4.1：颜色不得是唯一区分手段）。
#: 灰度打印会抹掉颜色差异，但线型和标记点形状会保留下来。
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "D", "v", "P"]

# 评价类题目共用的 8 项指标（示例指标，实际论文里必须写清来源依据）
INDICATORS = ["经济性", "安全性", "可靠性", "环保性", "运行效率", "可扩展性", "维护成本", "用户满意度"]

#: 8 阶 AHP 判断矩阵（Saaty 1-9 标度，互反）。本脚本自检给出 λmax 与 CR，
#: 该矩阵 CR≈0.011 < 0.1，通过一致性检验；数值由脚本外的一次固定种子搜索得到并写死。
AHP_MATRIX = [
    [1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0],
    [2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    [0.5, 1.0 / 3.0, 1.0, 0.5, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5],
    [1.0, 0.5, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5],
    [1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    [2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    [2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 2.0],
    [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0],
]

#: Saaty 随机一致性指标 RI（n=1..10），用于 CR = CI / RI
SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ------------------------------------------------- matplotlib / 中文字体

try:  # pragma: no cover - 取决于运行环境
    import matplotlib

    matplotlib.use("Agg", force=True)   # 非交互后端：必须在 import pyplot 之前
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _MPL_ERROR = None
except Exception as _exc:               # noqa: BLE001 - 需要把任何缺库原因都转成可读提示
    plt = None                          # type: ignore[assignment]
    font_manager = None                 # type: ignore[assignment]
    _MPL_ERROR = _exc


def resolve_cjk_font() -> str:
    """在本机已安装字体中解析出第一个可用的中文字体名。

    参数:
        无。

    返回:
        str: 实际命中的字体名；全部候选都缺失时返回兜底的 `"DejaVu Sans"`。

    算法:
        读取 `matplotlib.font_manager` 的字体表，按 `CJK_FONT_CANDIDATES` 顺序取首个命中项。

    复杂度:
        时间 O(F)，F 为本机注册字体数（一次构建字体表，随后为集合查找）；空间 O(F)。

    陷阱:
        - 字体表在首次访问时扫描系统字体目录，冷启动可能耗时 1-3 秒，故只在 main 里调一次。
        - 返回 `"DejaVu Sans"` 说明本机没有中文字体，图里的汉字会变成方框；
          此时必须提示用户而不是静默出图。
        - 不能用 `findfont` 的 `fallback_to_default=False` 之外的花招：旧版本行为不一致。

    参考:
        Matplotlib 官方文档 "Text rendering With LaTeX / Fonts"；
        中文字体候选取自国赛论文常见配置（Windows 下 SimHei / Microsoft YaHei）。
    """
    if font_manager is None:            # pragma: no cover - 仅在缺 matplotlib 时触发
        return CJK_FONT_CANDIDATES[-1]
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:                   # pragma: no cover - 字体表构建失败的兜底
        return CJK_FONT_CANDIDATES[-1]
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            return name
    return CJK_FONT_CANDIDATES[-1]


def configure_style() -> str:
    """统一设置绘图风格与中文字体，返回实际解析到的字体名。

    参数:
        无。

    返回:
        str: 实际生效的中文字体名。

    算法:
        一次性写入整套 rcParams：字体与字号 → 坐标系外框 → 刻度 → 图例 → 线条与标记
        → 网格 → 输出。原则是"**数据是唯一的深色，其余全部退到背景**"：
        去掉上/右边框、刻度朝外、图例去边框、网格改浅灰细实线、条形图去黑描边。

    复杂度:
        时间 O(1)（不含字体表构建）；空间 O(1)。

    陷阱:
        - `axes.unicode_minus` 不设成 False，负号会渲染成"豆腐块"（方框）。
        - 字号设得比正文还大是很常见的新手错误，会让图显得"头重脚轻"。
        - 不要在这里 print：模块导入期不允许有输出。
        - `axes.spines.top/right=False` 只对**之后新建**的坐标轴生效；若某图需要完整
          边框（如热力图），要在该图内部显式打开，不能指望这里的默认值。
        - `patch.linewidth=0` 会让条形图/直方图默认无描边；个别图若显式传了
          `edgecolor`/`linewidth`，将覆盖这里的默认值——这正是"逐图微调"的入口。
        - 这里**不设** `axes.grid=True`：网格是否出现由每张图自己决定（`_grid`），
          否则热力图、雷达图、网络图会被网格线污染。

    参考:
        Matplotlib 官方 `rcParams` 说明；`assets/gallery/README.md` §5 通用绘图规范；
        Machado, Oliveira & Fernandes (2009) *A physiologically-based model for
        simulation of color vision deficiency*；Okabe & Ito (2008) 色盲友好配色。
    """
    font = resolve_cjk_font()
    plt.rcParams.update({
        # ---- 字体与字号：正文 10.5pt 时图内 8-10pt，绝不大于正文 ----
        "font.sans-serif": CJK_FONT_CANDIDATES,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        # ---- 坐标系外框：只留左+下，颜色用近黑而非纯黑 ----
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        # 子图标题一律**左对齐**顶在各自坐标轴上方：这是期刊/CVPR 的分图约定，
        # 居中式标题会被误读成"整张图的总标题"。`fig.suptitle` 仍居中，两者分工明确。
        "axes.titlelocation": "left",
        "axes.titlepad": 7.0,
        "axes.labelpad": 4.0,
        "text.color": INK,
        # ---- 刻度：朝外、短、细 ----
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        # ---- 图例：无边框、紧凑 ----
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.7,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.6,
        # ---- 线条与标记 ----
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "patch.force_edgecolor": False,
        "errorbar.capsize": 2.5,
        # ---- 网格：浅灰细实线，置于数据之下（是否显示由各图决定）----
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "grid.alpha": 0.9,
        # ---- 输出 ----
        "image.cmap": CMAP_SEQ,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.dpi": 110,
    })
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(CYCLE))
    return font


# ---------------------------------------------------------------- 数值内核
# 下面这些函数只依赖 numpy，既被绘图函数调用，也被 _self_test() 断言，
# 因此"图里的结果"和"自检里的结果"用的是同一份实现。


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """熵权法计算指标权重（数据驱动赋权）。

    参数:
        matrix: 形状 (m, n) 的决策矩阵，**必须已完成正向化与归一化**，且各列非负。

    返回:
        np.ndarray: 长度 n 的权重向量，元素非负且和为 1。

    算法:
        p_ij = x_ij / Σ_i x_ij → 信息熵 e_j = -Σ p ln p / ln m → 差异系数 d_j = 1 - e_j
        → 归一化 w_j = d_j / Σ d_j。

    复杂度:
        时间 O(m·n)；空间 O(m·n)。

    陷阱:
        - 矩阵里有 0 时 `p ln p` 取 0（0·ln0 = 0），必须显式 `np.where`，否则得到 NaN。
        - 若某列所有样本相同，e_j = 1、d_j = 0，该列权重自动为 0——这通常说明指标无区分度，
          应在论文里说明而不是硬留一个 0 权重。
        - 熵权依赖样本集：换一批评价对象，权重就会变，必须连同样本范围一起报告。

    参考:
        信息熵赋权法（Shannon 1948 信息熵；国内教材通称"熵权法"）。
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("matrix 必须是二维 (m, n) 数组")
    if x.size == 0:
        raise ValueError("matrix 不能为空")
    if np.any(x < 0):
        raise ValueError("熵权法要求归一化后的矩阵非负，请先做极差标准化")
    m = x.shape[0]
    if m < 2:
        raise ValueError("样本数 m 必须 >= 2，否则 ln(m) 为 0")
    col_sum = x.sum(axis=0, keepdims=True)
    col_sum = np.where(col_sum == 0, 1.0, col_sum)
    p = x / col_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    e = -plogp.sum(axis=0) / np.log(m)
    d = 1.0 - e
    total = d.sum()
    if total <= 0:
        return np.full(x.shape[1], 1.0 / x.shape[1])
    return d / total


def ahp_weights(judgement: Sequence[Sequence[float]]) -> Tuple[np.ndarray, float, float, float]:
    """AHP 特征向量法求权重，并顺带算出 λmax、CI 与 CR。

    参数:
        judgement: n×n 正互反判断矩阵（Saaty 1-9 标度），主对角线必须为 1。

    返回:
        (w, lmax, ci, cr)：权重向量、最大特征值、一致性指标 CI、一致性比率 CR。

    算法:
        求判断矩阵的主特征向量（实部最大特征值对应向量）并归一化；
        CI = (λmax - n) / (n - 1)；CR = CI / RI(n)。

    复杂度:
        时间 O(n³)（一般特征值分解）；空间 O(n²)。

    陷阱:
        - `CR < 0.1` 只是一致性门槛，不代表判断矩阵"客观正确"；来源仍要写清。
        - n <= 2 时 RI = 0，CR 无定义，本函数直接返回 0.0，论文里不要写"CR=0 表示完全一致"。
        - 用 `np.linalg.eig` 得到的是复数数组，取主特征值必须比较**实部**。

    参考:
        Saaty 1980 特征向量法；RI 取值见 Saaty 随机一致性指标表。
    """
    a = np.asarray(judgement, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("judgement 必须是方阵")
    n = a.shape[0]
    if n < 1:
        raise ValueError("judgement 不能为空")
    vals, vecs = np.linalg.eig(a)
    k = int(np.argmax(vals.real))
    lmax = float(vals.real[k])
    w = np.abs(vecs[:, k].real)
    s = w.sum()
    w = w / s if s > 0 else np.full(n, 1.0 / n)
    if n <= 2:
        return w, lmax, 0.0, 0.0
    ci = (lmax - n) / (n - 1)
    cr = ci / SAATY_RI.get(n, 1.49)
    return w, lmax, ci, cr


def topsis_closeness(matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """TOPSIS 相对贴近度 C_i（越大越好），全程假定指标已正向化。

    参数:
        matrix: 形状 (m, n) 的决策矩阵（原始量纲即可，函数内部做向量归一化）。
        weights: 长度 n 的权重向量。

    返回:
        np.ndarray: 长度 m 的贴近度 C_i ∈ [0, 1]。

    算法:
        向量归一化 → 加权 → 取正/负理想解 → 欧氏距离 d+、d- → C_i = d- / (d+ + d-)。

    复杂度:
        时间 O(m·n)；空间 O(m·n)。

    陷阱:
        - 只对正向指标有效；成本型指标必须先正向化（`1 - 极差标准化` 或取倒数）。
        - d+ + d- = 0 时（所有方案完全相同）返回 0，会被误读成"最差"，要在论文里说明。
        - 贴近度是**相对**排序指标，不能解释为"得分 0.87 表示完成度 87%"。

    参考:
        Hwang & Yoon 1981 TOPSIS；国内教材通称"优劣解距离法"。
    """
    x = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    if x.shape[1] != w.size:
        raise ValueError("matrix 列数必须等于 weights 长度")
    denom = np.sqrt((x ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    v = (x / denom) * w[None, :]
    ideal = v.max(axis=0)
    nadir = v.min(axis=0)
    dp = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    dm = np.sqrt(((v - nadir) ** 2).sum(axis=1))
    tot = dp + dm
    return np.where(tot > 0, dm / np.where(tot > 0, tot, 1.0), 0.0)


def ranks_desc(scores: Sequence[float]) -> np.ndarray:
    """按分数从高到低给出名次（1 = 最好，并列按原顺序打破）。

    参数:
        scores: 分数序列，越大越好。

    返回:
        np.ndarray: 与输入等长的整数名次数组。

    算法:
        两次 argsort：先按降序排序，再把序号写回原位。

    复杂度:
        时间 O(m log m)；空间 O(m)。

    陷阱:
        并列分数会得到不同名次（本实现不做平均秩），论文里若存在并列必须说明打破规则。

    参考:
        常用竞赛名次约定。
    """
    s = np.asarray(scores, dtype=float)
    return (-s).argsort().argsort() + 1


def normal_quantile(p: Sequence[float]) -> np.ndarray:
    """标准正态分布分位数（Acklam 有理逼近，无需 scipy）。

    参数:
        p: 概率序列，取值须落在 (0, 1) 开区间内。

    返回:
        np.ndarray: 与 p 同形状的标准正态分位数 z。

    算法:
        分三段（左尾 / 中部 / 右尾）使用不同的有理多项式逼近。

    复杂度:
        时间 O(len(p))；空间 O(len(p))。

    陷阱:
        - p 取 0 或 1 会得到 ±inf/NaN；画 Q-Q 图时要用 (i-0.5)/n 而不是 i/n。
        - 这是逼近式，绝对误差约 1e-9，做论文级 Q-Q 图足够，但不能用于严格统计检验。

    参考:
        Acklam, P. J. "An algorithm for computing the inverse normal cumulative
        distribution function"（公开算法说明）。
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("p 必须严格落在 (0, 1) 内")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    q = np.zeros_like(p)
    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)
    if np.any(lo):
        ql = np.sqrt(-2.0 * np.log(p[lo]))
        q[lo] = (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
                ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1.0)
    if np.any(hi):
        qh = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        q[hi] = -(((((c[0] * qh + c[1]) * qh + c[2]) * qh + c[3]) * qh + c[4]) * qh + c[5]) / \
                 ((((d[0] * qh + d[1]) * qh + d[2]) * qh + d[3]) * qh + 1.0)
    if np.any(mid):
        pm = p[mid] - 0.5
        r = pm * pm
        q[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * pm / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return q


def gm11_forecast(series: Sequence[float], steps: int) -> np.ndarray:
    """GM(1,1) 灰色预测：一次累加 → 最小二乘辨识 → 还原，返回拟合+外推序列。

    参数:
        series: 一维正序列（长度 >= 4），数量级近似指数增长时效果最好。
        steps: 向后外推的步数。

    返回:
        np.ndarray: 长度 len(series) + steps 的序列，前 len(series) 项为拟合值。

    算法:
        累加生成 x1 → 紧邻均值 z1 → 最小二乘解 (a, b) → x1_hat(k) = (x0 - b/a)e^{-ak} + b/a
        → 一次累减还原。

    复杂度:
        时间 O(n + steps)；空间 O(n + steps)。

    陷阱:
        - 序列含非正数会直接崩（对数/指数还原需要正值）；
        - a 接近 0 时 b/a 会数值爆炸，本函数对 |a| < 1e-12 做了线性退化处理；
        - 真实竞赛里必须补**后验差比检验**（C 值与小误差概率 P），只看拟合曲线不算检验；
        - 不要用 GM(1,1) 外推有季节性的序列——它只能刻画单调指数趋势。

    参考:
        邓聚龙 灰色系统理论 GM(1,1) 模型（国内教材通用写法）。
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("series 必须是一维且长度 >= 4")
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    if np.any(x <= 0):
        raise ValueError("GM(1,1) 要求序列全为正数")
    n = x.size
    x1 = np.cumsum(x)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    b_mat = np.column_stack([-z1, np.ones(n - 1)])
    coef, *_ = np.linalg.lstsq(b_mat, x[1:], rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])
    k = np.arange(0, n + steps, dtype=float)
    if abs(a_hat) < 1e-12:
        x1_hat = x[0] + b_hat * k
    else:
        x1_hat = (x[0] - b_hat / a_hat) * np.exp(-a_hat * k) + b_hat / a_hat
    x_hat = np.empty_like(x1_hat)
    x_hat[0] = x1_hat[0]
    x_hat[1:] = np.diff(x1_hat)
    return x_hat


def holt_linear(series: Sequence[float], steps: int,
                alpha: float = 0.6, beta: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Holt 双参数线性指数平滑：返回一步拟合序列与向后 steps 步预测。

    参数:
        series: 一维序列（长度 >= 3）。
        steps: 外推步数。
        alpha: 水平平滑系数，越大越贴近近期观测。
        beta: 趋势平滑系数，越大越贴近近期趋势。

    返回:
        (fitted, forecast)：长度 n 的拟合值、长度 steps 的预测值。

    算法:
        level_t = α·x_t + (1-α)(level_{t-1} + trend_{t-1})；
        trend_t = β(level_t - level_{t-1}) + (1-β)trend_{t-1}；预测 x_hat_{t+h} = level_t + h·trend_t。

    复杂度:
        时间 O(n + steps)；空间 O(n + steps)。

    陷阱:
        - 拟合值用的是**上一期**水平，属于一步预测，因此残差有 1 期滞后相关性，
          做残差白噪声检验时要注意；
        - α、β 靠目测调出来会被质疑，论文里应给出网格搜索 + 滚动回测 MAE 的选参过程；
        - 长程外推会变成直线，超出数据支持范围就不要再往外画。

    参考:
        Holt 1957 双参数指数平滑；Hyndman & Athanasopoulos《Forecasting: Principles and Practice》。
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("series 必须是一维且长度 >= 3")
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
        raise ValueError("alpha 与 beta 必须落在 (0, 1) 内")
    level = float(x[0])
    trend = float(x[1] - x[0])
    fitted = np.empty(x.size)
    fitted[0] = level
    for t in range(1, x.size):
        prev_level = level
        level = alpha * x[t] + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        fitted[t] = level + trend
    h = np.arange(1, steps + 1, dtype=float)
    return fitted, level + h * trend


def quadratic_seasonal_forecast(series: Sequence[float], steps: int,
                                period: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """二次趋势最小二乘 + 季节因子修正的预测（作为第三条对照曲线）。

    参数:
        series: 一维序列。
        steps: 外推步数。
        period: 季节周期长度。

    返回:
        (fitted, forecast)：长度 n 的拟合值、长度 steps 的预测值。

    算法:
        先对 1..n 做二次多项式最小二乘；对残差按相位取均值得到季节因子；
        预测 = 二次趋势外推 + 对应相位的季节因子。

    复杂度:
        时间 O(n + steps)；空间 O(n + steps)。

    陷阱:
        - 二次趋势会在长程外推中迅速发散，外推步数不应超过一个季节周期太多；
        - 只有 2-3 个完整周期时季节因子极不稳定，必须做滚动回测而不是只看拟合。

    参考:
        经典时间序列分解（趋势 + 季节 + 随机）的教科书做法。
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or x.size < 2 * period:
        raise ValueError("series 长度至少为 2 个完整周期")
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    n = x.size
    t = np.arange(1, n + 1, dtype=float)
    coef = np.polyfit(t, x, 2)
    trend = np.polyval(coef, t)
    resid = x - trend
    season = np.array([resid[np.arange(n) % period == r].mean() for r in range(period)])
    season = season - season.mean()
    fitted = trend + season[np.arange(n) % period]
    t_fc = np.arange(n + 1, n + steps + 1, dtype=float)
    forecast = np.polyval(coef, t_fc) + season[(np.arange(n, n + steps)) % period]
    return fitted, forecast


def sir_rk4(beta: float, gamma: float, s0: float, i0: float,
            days: int, dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用经典四阶 Runge-Kutta 求解 SIR 传染病模型。

    参数:
        beta: 感染率（每人每天有效接触数 × 传染概率），单位 1/天。
        gamma: 恢复率，等于平均病程的倒数，单位 1/天。
        s0: 初始易感者比例（无量纲，0-1）。
        i0: 初始感染者比例（无量纲，0-1）。
        days: 模拟总天数。
        dt: 步长（天）。

    返回:
        (t, s, i, r)：时间网格与三条状态曲线（均为人口比例，无量纲）。

    算法:
        dS/dt = -βSI，dI/dt = βSI - γI，dR/dt = γI；RK4 定步长积分。

    复杂度:
        时间 O(days/dt)；空间 O(days/dt)。

    陷阱:
        - 必须满足 R0 = β/γ > 1 才会出现疫情高峰，否则 I(t) 单调衰减、图上"看不到峰"；
        - 步长过大（dt > 0.1 天）会明显改变峰值，论文里要做步长减半的收敛性检验；
        - 参数带单位（1/天）而不是"无量纲系数"，写错单位是最常见扣分点；
        - SIR 假设均匀混合与终身免疫，用在有潜伏期/再感染的场景要换成 SEIR/SIRS。

    参考:
        Kermack & McKendrick 1927 传染病动力学仓室模型；RK4 为通用数值方法。
    """
    if days < 1:
        raise ValueError("days 必须 >= 1")
    if dt <= 0 or dt > 0.5:
        raise ValueError("dt 建议取 (0, 0.5] 天，否则精度不足")
    steps = int(round(days / dt))
    t = np.linspace(0.0, days, steps + 1)
    s = np.empty(steps + 1)
    i = np.empty(steps + 1)
    r = np.empty(steps + 1)
    s[0], i[0], r[0] = float(s0), float(i0), 1.0 - float(s0) - float(i0)

    def deriv(sv: float, iv: float, rv: float) -> Tuple[float, float, float]:
        return -beta * sv * iv, beta * sv * iv - gamma * iv, gamma * iv

    for k in range(steps):
        sk, ik, rk = s[k], i[k], r[k]
        k1 = deriv(sk, ik, rk)
        k2 = deriv(sk + 0.5 * dt * k1[0], ik + 0.5 * dt * k1[1], rk + 0.5 * dt * k1[2])
        k3 = deriv(sk + 0.5 * dt * k2[0], ik + 0.5 * dt * k2[1], rk + 0.5 * dt * k2[2])
        k4 = deriv(sk + dt * k3[0], ik + dt * k3[1], rk + dt * k3[2])
        s[k + 1] = sk + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        i[k + 1] = ik + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        r[k + 1] = rk + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return t, s, i, r


def pareto_front(points: np.ndarray) -> np.ndarray:
    """返回双目标（均为最小化）意义下的非支配点布尔掩码。

    参数:
        points: 形状 (m, 2) 的目标值矩阵，两列都是"越小越好"。

    返回:
        np.ndarray: 长度 m 的布尔数组，True 表示该点位于帕累托前沿。

    算法:
        对每个点检查是否存在另一点在两目标上都不劣且至少一个严格更优。

    复杂度:
        时间 O(m²)（教学实现；m 大时应先按第一目标排序再做一次线性扫描）；空间 O(m)。

    陷阱:
        - 支配方向搞反会得到"右上角前沿"这种荒谬结果，先统一各目标的最大化/最小化方向；
        - 目标含噪声时"全部点都是前沿"很常见，必须报告解的密度而不是只画一条线。

    参考:
        Pareto 最优 / 多目标优化支配关系定义（经典定义，无单一出处）。
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points 必须是 (m, 2) 数组")
    m = p.shape[0]
    mask = np.ones(m, dtype=bool)
    for a in range(m):
        if not mask[a]:
            continue
        dominated = (p[:, 0] <= p[a, 0]) & (p[:, 1] <= p[a, 1]) & \
                    ((p[:, 0] < p[a, 0]) | (p[:, 1] < p[a, 1]))
        if np.any(dominated):
            mask[a] = False
    return mask


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator,
           n_init: int = 10, max_iter: int = 300) -> Tuple[np.ndarray, np.ndarray, float]:
    """K-means 聚类（Lloyd 迭代 + k-means++ 初始化）。

    参数:
        x: 形状 (n, d) 的样本矩阵。
        k: 簇数。
        rng: `np.random.default_rng` 生成的随机数发生器（显式传入，不用全局状态）。
        n_init: 重复初始化次数，取惯性最小的一次。
        max_iter: 单次运行的最大迭代轮数。

    返回:
        (labels, centroids, inertia)：样本簇标签、k×d 质心、簇内平方和。

    算法:
        k-means++ 选初始质心 → 分配样本到最近质心 → 更新质心 → 直到标签不再变化。

    复杂度:
        时间 O(n_init · max_iter · n · k · d)；空间 O(n·d + k·d)。

    陷阱:
        - 结果依赖初始化与随机种子，论文必须固定种子并报告 inertia 与多次运行的稳定性；
        - 不做标准化时量纲大的特征会独占距离度量，聚类前必须标准化；
        - k 是超参数，必须用轮廓系数/肘部法等给出选取依据，不能"看着像 4 类就是 4 类"。

    参考:
        Lloyd 1982 迭代算法；Arthur & Vassilvitskii 2007 k-means++ 初始化。
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x 必须是二维数组")
    n = x.shape[0]
    if k < 1 or k > n:
        raise ValueError("k 必须落在 [1, n] 内")
    best_labels = None
    best_centroids = None
    best_inertia = np.inf
    for _ in range(n_init):
        idx = [int(rng.integers(n))]
        for _j in range(1, k):
            d2 = np.min(((x[:, None, :] - x[idx][None, :, :]) ** 2).sum(axis=2), axis=1)
            total = d2.sum()
            prob = np.full(n, 1.0 / n) if total <= 0 else d2 / total
            idx.append(int(rng.choice(n, p=prob)))
        cent = x[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            dist = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels
            for c in range(k):
                member = x[labels == c]
                if member.size:
                    cent[c] = member.mean(axis=0)
        inertia = float(((x - cent[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels, best_centroids = inertia, labels.copy(), cent.copy()
    return best_labels, best_centroids, best_inertia  # type: ignore[return-value]


def silhouette_scores(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """计算每个样本的轮廓系数 s_i（无需 sklearn）。

    参数:
        x: 形状 (n, d) 的样本矩阵。
        labels: 长度 n 的簇标签。

    返回:
        np.ndarray: 长度 n 的轮廓系数；单元素簇记为 0。

    算法:
        a_i = 同簇平均距离，b_i = 最近异簇平均距离，s_i = (b_i - a_i) / max(a_i, b_i)。

    复杂度:
        时间 O(n²·d)（用距离矩阵）；空间 O(n²)。

    陷阱:
        - O(n²) 空间，n 上万时必须改用抽样估计；
        - 单样本簇的轮廓系数按定义为 0，会让"平均轮廓系数"被人为拉低，必须在论文里说明；
        - 轮廓系数只能比较同一数据、同一距离度量下的不同 k，跨数据集不可比。

    参考:
        Rousseeuw 1987 silhouette 定义。
    """
    x = np.asarray(x, dtype=float)
    lab = np.asarray(labels, dtype=int)
    n = x.shape[0]
    if lab.size != n:
        raise ValueError("labels 长度必须等于样本数")
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    out = np.zeros(n)
    for i in range(n):
        same = (lab == lab[i])
        same[i] = False
        a_i = dist[i, same].mean() if np.any(same) else 0.0
        best_b = np.inf
        for c in np.unique(lab):
            if c == lab[i]:
                continue
            other = (lab == c)
            if np.any(other):
                best_b = min(best_b, float(dist[i, other].mean()))
        if not np.isfinite(best_b):
            out[i] = 0.0
        else:
            denom = max(a_i, best_b)
            out[i] = 0.0 if denom <= 0 else (best_b - a_i) / denom
    return out


def dijkstra(graph: Dict[str, List[Tuple[str, float]]], source: str) -> Dict[str, float]:
    """Dijkstra 单源最短路（边权非负，使用 heapq 的 O(m log n) 实现）。

    参数:
        graph: 邻接表 {节点: [(邻居, 边权), ...]}。
        source: 源点。

    返回:
        Dict[str, float]: 源点到各点的最短距离；不可达为 `inf`。

    算法:
        优先队列 + 惰性删除：每次弹出当前最小 tentative 距离并松弛其出边。

    复杂度:
        时间 O(m log n)；空间 O(n + m)。

    陷阱:
        - 只适用于**非负**边权；含负权要用 Bellman-Ford；
        - 记录路径时需要单独的 predecessor 表，只返回距离无法重建路径；
        - 有时间窗/容量约束的配送问题不能直接把逐段最短路拼成全局最优。

    参考:
        Dijkstra 1959 最短路算法；heapq 为 Python 标准库优先队列。
    """
    dist: Dict[str, float] = {node: float("inf") for node in graph}
    dist[source] = 0.0
    pq: List[Tuple[float, str]] = [(0.0, source)]
    seen = set()
    while pq:
        d, node = heapq.heappop(pq)
        if node in seen:
            continue
        seen.add(node)
        for nxt, w in graph.get(node, []):
            if w < 0:
                raise ValueError("Dijkstra 要求边权非负")
            nd = d + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))
    return dist


def idw_interpolate(sample_xy: np.ndarray, sample_z: np.ndarray,
                    grid_x: np.ndarray, grid_y: np.ndarray,
                    power: float = 2.0) -> np.ndarray:
    """反距离加权（IDW）插值，把散点观测插到规则网格上。

    参数:
        sample_xy: 形状 (m, 2) 的采样点坐标。
        sample_z: 长度 m 的观测值。
        grid_x, grid_y: 规则网格的横、纵坐标（由 meshgrid 生成，形状相同）。
        power: 距离幂次，常用 2（越大越"局部"）。

    返回:
        np.ndarray: 与 grid_x 同形状的插值结果。

    算法:
        z_hat(x) = Σ w_i z_i / Σ w_i，其中 w_i = 1 / (d_i^p + ε)。

    复杂度:
        时间 O(m · N)（N 为网格点数）；空间 O(N)。

    陷阱:
        - IDW 是**精确**插值：网格点落在采样点上会返回观测值，因此"插值误差为 0"是假象；
        - 它会平滑掉极值（估计的峰比真实峰低），不产生新的极值；
        - 幂次与搜索半径是人为选择，必须给交叉验证 RMSE 作为依据；
        - 不能编造"克里金插值"的名字——IDW 没有变差函数模型。

    参考:
        Shepard 1968 反距离加权插值（教科书通用方法）。
    """
    xy = np.asarray(sample_xy, dtype=float)
    z = np.asarray(sample_z, dtype=float)
    if xy.shape[0] != z.size:
        raise ValueError("采样点数与观测值数不一致")
    if power <= 0:
        raise ValueError("power 必须为正")
    gx = np.asarray(grid_x, dtype=float)
    out = np.empty(gx.shape, dtype=float)
    flat = gx.ravel()
    flat_y = np.asarray(grid_y, dtype=float).ravel()
    eps = 1e-12
    for i in range(flat.size):
        d = np.sqrt((xy[:, 0] - flat[i]) ** 2 + (xy[:, 1] - flat_y[i]) ** 2)
        if np.any(d < 1e-9):                     # 网格点正好落在采样点上
            out.ravel()[i] = z[int(np.argmin(d))]
            continue
        w = 1.0 / (d ** power + eps)
        out.ravel()[i] = float((w * z).sum() / w.sum())
    return out


def mm1_simulation(lam: float, mu: float, n_customers: int,
                   rng: np.random.Generator) -> Tuple[float, float, float]:
    """M/M/1 排队系统的离散事件仿真（Lindley 递推 + 时间加权队长积分）。

    参数:
        lam: 到达率 λ，单位 人/h。
        mu: 服务率 μ，单位 人/h，必须 μ > λ。
        n_customers: 仿真的顾客数。
        rng: 显式随机数发生器。

    返回:
        (L, Lq, Wq)：平均系统队长（人）、平均排队队长（人）、平均排队等待时间（h）。

    算法:
        指数分布抽样生成到达间隔与服务时间 → Lindley 递推求每位顾客的等待时间
        → 由 Σ(离开-到达)/观察窗得时间加权系统队长，由 Σ 等待时间/观察窗得排队队长。

    复杂度:
        时间 O(n)；空间 O(n)。

    陷阱:
        - 单次运行的置信区间很宽，必须多次重复取均值并给误差棒；
        - 利用率 ρ→1 时需要极长的预热期（warm-up），有限顾客数会系统性低估队长；
        - 稳态公式只在 ρ<1 且系统达到稳态后成立，初始空系统有瞬态偏差。

    参考:
        Kendall 记号 M/M/1；Little 1961 排队论守恒律（L = λW）。
    """
    if mu <= lam:
        raise ValueError("M/M/1 要求服务率 mu > 到达率 lam（利用率 rho < 1）")
    if n_customers < 10:
        raise ValueError("n_customers 太小，仿真无意义")
    inter = rng.exponential(1.0 / lam, n_customers)
    serv = rng.exponential(1.0 / mu, n_customers)
    arrivals = np.cumsum(inter)
    start = np.empty(n_customers)
    depart = np.empty(n_customers)
    for i in range(n_customers):
        start[i] = arrivals[i] if i == 0 else max(arrivals[i], depart[i - 1])
        depart[i] = start[i] + serv[i]
    wq = start - arrivals                       # 排队等待时间
    window = depart[-1] - arrivals[0]
    if window <= 0:
        raise ValueError("观察窗长度为 0，参数异常")
    l_sys = float((depart - arrivals).sum() / window)
    l_q = float(wq.sum() / window)
    w_q = float(wq.mean())
    return l_sys, l_q, w_q


# ---------------------------------------------------------------- 绘图工具


# 分图号前缀，如 "(a) 残差时序图" —— 用来把标签排成粗体并左对齐。
_PANEL_RE = re.compile(r"^\((?P<tag>[a-z])\)\s*(?P<rest>.*)$")


def _finish(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """给单个坐标轴统一加标题与带单位的轴标签（内部小工具）。

    参数:
        ax: matplotlib 坐标轴对象。
        title: 图内标题；投稿时应移入图注。
        xlabel / ylabel: **必须带物理量与单位**的轴标签。

    返回:
        None（原地修改 ax）。

    算法:
        标题若以 `(a)`/`(b)` 这类分图号开头，则交给 `_panel` 排成"粗体分图号 + 左对齐标题"；
        否则直接左对齐放置。随后依次调用 set_xlabel / set_ylabel。

    复杂度:
        时间 O(1)；空间 O(1)。

    陷阱:
        - 轴标签只写符号不写单位（如只写 "t"）是评审最常见的扣分点之一；
          单位用正体、量名用斜体（本脚本用 `$...$` 排量名，单位放在 `$...$` 外）。
        - 期刊排版里子图标题一律**左对齐**顶在各自子图上方，居中式标题会被误读成整图标题。

    参考:
        GB/T 3102 量与单位的一般原则；`references/paper-structure.md` 四、写作技术规范。
    """
    if title:
        m = _PANEL_RE.match(title)
        if m:
            _panel(ax, m.group("tag"), m.group("rest"))
        else:
            ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _grid(ax, axis: str = "both", which: str = "major") -> None:
    """按统一规范打开浅灰细实线网格（只应在"网格有助于读数"的图上调用）。

    参数:
        ax: matplotlib 坐标轴对象。
        axis: `"both"` / `"x"` / `"y"`，只对某一方向加网格。
        which: `"major"` / `"minor"` / `"both"`。

    返回:
        None（原地修改 ax）。

    算法:
        调用 `ax.grid(True, ...)`，颜色/线宽/线型全部取自设计令牌 `GRID`。

    复杂度:
        时间 O(1)；空间 O(1)。

    陷阱:
        - 旧写法把网格设成 `linestyle=":"` + `alpha=0.45`，远看像"马赛克"，
          黑白打印后会变成一层脏灰；改为浅灰**细实线**更接近期刊排版。
        - 热力图、雷达图、网络图不要加网格：它们有自己的单元格/轴网，会互相打架。
        - 网格必须在数据之下，这一点由 `axes.axisbelow=True` 统一保证，别在这里改。

    参考:
        `assets/gallery/README.md` §5.7 风格统一。
    """
    ax.grid(True, axis=axis, which=which, color=GRID, linewidth=0.7,
            linestyle="-", alpha=0.9)


def _series(i: int, **over) -> dict:
    """第 `i` 条数据序列的"三重编码"绘图参数：颜色 + 线型 + 标记点。

    参数:
        i: 序列序号（从 0 开始）。
        **over: 需要覆盖的额外 `plot` 关键字（如 `linewidth` / `markersize`）。

    返回:
        dict：可直接 `**` 展开进 `ax.plot(...)` 的关键字。

    算法:
        颜色取 `CYCLE[i % 6]`，标记点取 `_MARKERS[i % 6]`；线型按"轮次"取，
        第一轮全部实线、第二轮虚线、第三轮点划线、第四轮点线。于是同一轮内
        颜色和标记点两两不同，跨轮之间线型不同——任意两条曲线至少在三项里
        有两项不同。

    复杂度:
        时间 O(1)；空间 O(1)。

    陷阱:
        - **颜色不能是唯一的区分通道。** 本仓库色板的灰度间隔只有约 1.1/100，
          二色觉模拟下最差 ΔE 有 16.1——也就是说：色盲读者能靠颜色分开，
          但黑白打印出来的稿子不行。评审老师经常打印看稿，所以每条曲线
          必须同时有线型和标记点差异（灰度打印时这两项还在）。
        - 线型的可见差异比标记点更可靠：`-` / `--` / `-.` / `:` 四档在缩小
          到单栏宽度（约 8 cm）后仍能分辨，标记点则会糊成小点。因此曲线超过
          6 条时应当考虑分面（subplot）而不是继续加色。
        - 标记点密集时用 `markevery` 抽稀，否则 200 个点会连成一条粗带。

    参考:
        Okabe & Ito (2008) "Color Universal Design"；
        WCAG 2.1 1.4.1 Use of Color（禁止把颜色作为唯一区分手段）。
    """
    turn, idx = divmod(i, len(CYCLE))
    kw = {
        "color": CYCLE[idx],
        "linestyle": _LINESTYLES[turn % len(_LINESTYLES)],
        "marker": _MARKERS[idx],
    }
    kw.update(over)
    return kw


def _legend(ax, **kwargs):
    """统一风格的去边框图例。

    参数:
        ax: matplotlib 坐标轴对象。
        **kwargs: 透传给 `ax.legend`；`frameon` 默认 `False`，可显式覆盖。

    返回:
        matplotlib 的 Legend 对象。

    算法:
        设置 `frameon=False` 后调用 `ax.legend`。

    复杂度:
        时间 O(1)；空间 O(1)。

    陷阱:
        - 图例带白底边框会遮住数据点，`framealpha=0.95` 只是把遮住变得"看得见"而已；
          期刊排版普遍用无边框图例，靠摆放位置避让数据。
        - 图例遮挡数据是审稿意见里的高频问题：优先 `loc` 选空白角，其次才考虑加边框。

    参考:
        `assets/gallery/README.md` §5.7 风格统一。
    """
    kwargs.setdefault("frameon", False)
    return ax.legend(**kwargs)


def _panel(ax, tag: str, title: str = "") -> None:
    """给多子图打左上角粗体分图号（CVPR/期刊常见的 (a)(b)(c) 约定）。

    参数:
        ax: matplotlib 坐标轴对象。
        tag: 分图号，如 `"(a)"`。
        title: 该子图的标题文字；为空则只显示分图号。

    返回:
        None（原地修改 ax）。

    算法:
        用 mathtext 把分图号排成粗体，与后面的标题文字拼接后左对齐放置。

    复杂度:
        时间 O(1)；空间 O(1)。

    陷阱:
        - 分图号必须**左对齐**且位于子图上方；居中会让人误以为是整图的标题。
        - 分图号不要写进图注之外的地方重复编号；正文引用时统一用"(a)"这种形式。

    参考:
        `assets/gallery/README.md` §5.6 图注要自解释。
    """
    text = r"$\mathbf{(%s)}$" % tag.strip("()")
    if title:
        text += "  " + title
    ax.set_title(text, loc="left", fontsize=9.0, color=INK, pad=6.0)


def _note(ax, x: float, y: float, text: str, **kwargs) -> None:
    """在图内放一个"说明性文本框"，统一样式（半透明白底 + 浅灰细边）。

    参数:
        ax: matplotlib 坐标轴对象。
        x, y: 轴分数坐标（`transform=ax.transAxes`）。
        text: 说明文字，可用 `\\n` 换行。
        **kwargs: 透传给 `ax.text`（如 `ha` / `va` / `fontsize`）。

    返回:
        None。

    算法:
        以白色半透明圆角框为底，避免文字压住数据后读不清。

    复杂度:
        时间 O(1)；空间 O(1)。

    陷阱:
        - 用 `ax.transAxes`（轴分数）而不是数据坐标，图缩放时位置才稳定。
        - 底色透明度别低于 0.8，否则下面的网格会透上来干扰阅读。

    参考:
        `assets/gallery/README.md` §5.6 图注要自解释。
    """
    kwargs.setdefault("ha", "right")
    kwargs.setdefault("va", "bottom")
    kwargs.setdefault("fontsize", 7.2)
    kwargs.setdefault("color", INK_SOFT)
    kwargs.setdefault("transform", ax.transAxes)
    kwargs.setdefault("zorder", 6)
    ax.text(x, y, text,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=GRID, linewidth=0.7, alpha=0.92),
            **kwargs)


def _save(fig, out_dir: Path, name: str) -> None:
    """统一出口：按当前 dpi 保存并关闭图形。

    参数:
        fig: matplotlib Figure 对象。
        out_dir: 输出目录。
        name: 文件名（含 `.png`）。

    返回:
        None。

    算法:
        `fig.savefig(...)` 后立即 `plt.close(fig)`。

    复杂度:
        时间 O(像素数)；空间 O(像素数)。

    陷阱:
        - 不 `close` 会持续累积图形对象，跑完 16 张后内存和字体缓存都会膨胀。
        - `bbox_inches="tight"` 会裁掉多余白边，**最终像素尺寸以回读结果为准**，
          不要用 `figsize × dpi` 去推算尺寸。
        - 白色背景必须显式写：某些环境默认透明，插进 Word/LaTeX 会变成黑底。

    参考:
        Matplotlib `Figure.savefig` 文档。
    """
    fig.savefig(out_dir / name, dpi=_DPI, bbox_inches="tight",
                pad_inches=0.06, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 各题型配图


def evaluation_weights(out_dir: Path) -> None:
    """[评价类] 熵权法（数据驱动）与 AHP（专家判断）权重对比条形图。"""
    rng = np.random.default_rng(_SEED + 1)
    n_obj, n_ind = 12, len(INDICATORS)
    # 各列用不同集中的 Beta 分布，制造"区分度不同 → 熵权不同"的效果
    conc = np.array([2.0, 8.0, 3.0, 5.0, 1.5, 6.0, 4.0, 10.0])
    norm = np.column_stack([rng.beta(a, a, n_obj) for a in conc])
    norm = (norm - norm.min(axis=0)) / (norm.max(axis=0) - norm.min(axis=0))
    norm = 0.02 + 0.98 * norm
    w_ent = entropy_weights(norm)
    w_ahp, lmax, ci, cr = ahp_weights(AHP_MATRIX)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(n_ind)
    width = 0.38
    b1 = ax.bar(x - width / 2, w_ent * 100, width, label="熵权法（数据驱动）",
                color=OKABE_ITO[0], edgecolor=EDGE, linewidth=0.8)
    b2 = ax.bar(x + width / 2, w_ahp * 100, width, label="AHP（专家判断，CR=%.3f）" % cr,
                color=OKABE_ITO[1], edgecolor=EDGE, linewidth=0.8)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.25, "%.1f" % h,
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(INDICATORS, rotation=18, ha="right")
    ax.set_ylim(0, max(w_ent.max(), w_ahp.max()) * 100 * 1.22)
    _legend(ax, loc="upper right")
    _grid(ax, axis="y")
    _finish(ax, "8 项指标的两套赋权结果对比",
            "评价指标（无量纲）", "指标权重 $w_j$ / %")
    _note(ax, 0.01, 0.965, "$\\lambda_{max}$=%.3f，CI=%.4f，CR=%.3f < 0.1" % (lmax, ci, cr),
          ha="left", va="top", fontsize=7.5)
    fig.tight_layout()
    _save(fig, out_dir, "evaluation_weights.png")


def evaluation_topsis_rank(out_dir: Path) -> None:
    """[评价类] 熵权-TOPSIS 相对贴近度与最终排序。"""
    rng = np.random.default_rng(_SEED + 2)
    n_obj, n_ind = 10, len(INDICATORS)
    norm = rng.uniform(0.15, 1.0, (n_obj, n_ind))
    w_ent = entropy_weights(norm)
    c = topsis_closeness(norm, w_ent)
    order = np.argsort(-c)
    labels = ["方案 %s" % chr(ord("A") + i) for i in range(n_obj)]
    sorted_c = c[order]
    sorted_labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    colors = [OKABE_ITO[1] if r == 0 else OKABE_ITO[0] for r in range(n_obj)]
    y = np.arange(n_obj)
    ax.barh(y, sorted_c, color=colors, edgecolor=EDGE, linewidth=0.8, height=0.68)
    for i, v in enumerate(sorted_c):
        ax.text(v + 0.006, i, "%.4f（第 %d 名）" % (v, i + 1), va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(sorted_c) * 1.32)
    _grid(ax, axis="x")
    _finish(ax, "熵权-TOPSIS 综合评价结果（按贴近度降序）",
            "相对贴近度 $C_i$ / 无量纲（越大越优）", "评价对象")
    ax.text(0.985, 0.06, "橙色 = 最优方案", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=OKABE_ITO[1])
    fig.tight_layout()
    _save(fig, out_dir, "evaluation_topsis_rank.png")


def evaluation_weights_sensitivity(out_dir: Path) -> None:
    """[评价类] 权重 ±20% 扰动下的名次变化热力图（排序稳健性检验）。"""
    rng = np.random.default_rng(_SEED + 3)
    n_obj, n_ind = 10, len(INDICATORS)
    norm = rng.uniform(0.15, 1.0, (n_obj, n_ind))
    w0 = entropy_weights(norm)
    base_rank = ranks_desc(topsis_closeness(norm, w0))

    scenarios: List[Tuple[str, np.ndarray]] = [("基准", w0.copy())]
    for j in range(n_ind):
        for sign, tag in ((1.20, "+20%"), (0.80, "-20%")):
            w = w0.copy()
            w[j] = w[j] * sign
            w = w / w.sum()
            scenarios.append(("%s\n%s" % (INDICATORS[j], tag), w))

    rank_mat = np.column_stack([ranks_desc(topsis_closeness(norm, w)) for _, w in scenarios])
    labels = ["方案 %s" % chr(ord("A") + i) for i in range(n_obj)]

    fig, ax = plt.subplots(figsize=(11.2, 4.6))
    im = ax.imshow(rank_mat, cmap=CMAP_SEQ, aspect="auto", vmin=1, vmax=n_obj)
    ax.set_xticks(np.arange(rank_mat.shape[1]))
    ax.set_xticklabels([name for name, _ in scenarios], fontsize=6.2)
    ax.set_yticks(np.arange(n_obj))
    ax.set_yticklabels(labels)
    for i in range(n_obj):
        for j in range(rank_mat.shape[1]):
            ax.text(j, i, "%d" % rank_mat[i, j], ha="center", va="center", fontsize=6.0,
                    color="white" if rank_mat[i, j] > n_obj * 0.62 else "black")
    ax.axvline(0.5, color=OKABE_ITO[1], linewidth=1.3, linestyle="--")
    ax.set_xticks(np.arange(-0.5, rank_mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_obj, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, ticks=range(1, n_obj + 1), pad=0.012)
    cbar.set_label("综合排序名次 / 位（1 = 最优）")
    _finish(ax, "单指标权重 ±20% 扰动下的方案排序变化（红虚线左侧为基准权重）",
            "扰动情景（指标 + 扰动幅度） / 无量纲", "评价对象")
    fig.tight_layout()
    _save(fig, out_dir, "evaluation_weights_sensitivity.png")


def forecast_models_compare(out_dir: Path) -> None:
    """[预测类] 三种模型预测对比 + 最优模型的 95% 预测区间带。"""
    rng = np.random.default_rng(_SEED + 4)
    n_hist, horizon = 20, 8
    t = np.arange(1, n_hist + 1, dtype=float)
    y = 42.0 + 1.1 * t + 6.0 * np.sin(2 * np.pi * (t - 1) / 4.0) + rng.normal(0, 1.4, n_hist)

    gm_fit = gm11_forecast(y, horizon)
    holt_fit, holt_fc = holt_linear(y, horizon, alpha=0.55, beta=0.25)
    quad_fit, quad_fc = quadratic_seasonal_forecast(y, horizon, period=4)

    t_all = np.arange(1, n_hist + horizon + 1, dtype=float)
    fit_tail = slice(n_hist - horizon, n_hist)
    mae_gm = float(np.mean(np.abs(gm_fit[fit_tail] - y[fit_tail])))
    mae_holt = float(np.mean(np.abs(holt_fit[fit_tail] - y[fit_tail])))
    mae_quad = float(np.mean(np.abs(quad_fit[fit_tail] - y[fit_tail])))

    # 预测区间：以 Holt 的样本内残差标准差为基准，按 sqrt(h) 展宽（近似做法，需在论文中说明）
    sigma = float(np.std(y - holt_fit, ddof=1))
    h = np.arange(1, horizon + 1, dtype=float)
    half = 1.96 * sigma * np.sqrt(h)
    t_fc = t_all[n_hist:]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axvspan(n_hist + 0.5, n_hist + horizon + 0.5, color=PANEL_BG, zorder=0)
    ax.plot(t, y, "o-", color=INK, markersize=3.4, linewidth=1.3, label="历史观测值")
    ax.plot(t_fc, gm_fit[n_hist:], "s--", color=OKABE_ITO[2], markersize=3.4,
            linewidth=1.2, label="GM(1,1) 灰色预测（MAE=%.2f）" % mae_gm)
    ax.plot(t_fc, holt_fc, "^--", color=OKABE_ITO[1], markersize=3.6,
            linewidth=1.4, label="Holt 线性指数平滑（MAE=%.2f）" % mae_holt)
    ax.plot(t_fc, quad_fc, "d--", color=OKABE_ITO[3], markersize=3.4,
            linewidth=1.2, label="二次趋势+季节因子（MAE=%.2f）" % mae_quad)
    ax.fill_between(t_fc, holt_fc - half, holt_fc + half, color=OKABE_ITO[1],
                    alpha=0.18, label="Holt 模型 95% 预测区间")
    ax.plot(t, holt_fit, color=OKABE_ITO[1], linewidth=0.9, alpha=0.75)
    ax.axvline(n_hist + 0.5, color=INK_MUTED, linestyle=":", linewidth=1.2)
    ax.text(n_hist + 0.7, ax.get_ylim()[1] * 0.985, "← 拟合期 | 外推期 →",
            ha="left", va="top", fontsize=7.5, color=INK_SOFT)
    _legend(ax, loc="upper left", ncol=2)
    _grid(ax)
    _finish(ax, "三种预测模型的外推结果与 95% 预测区间",
            "期数 $t$ / 季度", "需求量 $y$ / 千件")
    fig.tight_layout()
    _save(fig, out_dir, "forecast_models_compare.png")


def forecast_residual_diagnostics(out_dir: Path) -> None:
    """[预测类] 残差诊断四联图：残差时序、正态 Q-Q、直方图、残差 vs 拟合值。"""
    rng = np.random.default_rng(_SEED + 5)
    n = 60
    t = np.arange(1, n + 1, dtype=float)
    y = 30.0 + 0.85 * t + 5.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 1.8, n)
    _, fc = holt_linear(y, 1, alpha=0.35, beta=0.12)
    trend = np.polyval(np.polyfit(t, y, 2), t)
    fitted = 0.5 * trend + 0.5 * np.roll(y, 1)
    fitted[0] = y[0]
    resid = y - fitted
    std_resid = (resid - resid.mean()) / resid.std(ddof=1)

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))

    ax = axes[0, 0]
    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.plot(t, resid, "-", color=OKABE_ITO[0], linewidth=1.1)
    ax.plot(t, resid, "o", color=OKABE_ITO[0], markersize=2.6)
    _grid(ax)
    _finish(ax, "(a) 残差时序图", "期数 $t$ / 月", "残差 $e_t$ / 千件")

    ax = axes[0, 1]
    p = (np.arange(1, n + 1) - 0.5) / n
    theo = normal_quantile(p)
    samp = np.sort(std_resid)
    ax.plot(theo, samp, "o", color=OKABE_ITO[2], markersize=3.2, label="样本分位数")
    lim = [min(theo.min(), samp.min()) - 0.4, max(theo.max(), samp.max()) + 0.4]
    ax.plot(lim, lim, "--", color=OKABE_ITO[1], linewidth=1.2, label="正态参考线 $y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _legend(ax, loc="upper left")
    _grid(ax)
    _finish(ax, "(b) 标准化残差正态 Q-Q 图",
            "理论分位数 $z_p$ / 无量纲", "样本分位数 / 无量纲")

    ax = axes[1, 0]
    ax.hist(resid, bins=12, color=OKABE_ITO[5], edgecolor=EDGE, linewidth=0.8,
            density=True, label="残差直方图")
    grid = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 200)
    pdf = np.exp(-0.5 * ((grid - resid.mean()) / resid.std(ddof=1)) ** 2) / \
        (resid.std(ddof=1) * np.sqrt(2 * np.pi))
    ax.plot(grid, pdf, "-", color=OKABE_ITO[1], linewidth=1.6, label="拟合正态密度")
    _legend(ax, loc="upper right")
    _grid(ax)
    _finish(ax, "(c) 残差分布直方图", "残差 $e_t$ / 千件", "概率密度 / 千件$^{-1}$")

    ax = axes[1, 1]
    ax.axhline(0.0, color=INK, linewidth=0.9)
    s = resid.std(ddof=1)
    ax.axhline(2 * s, color=OKABE_ITO[1], linestyle="--", linewidth=1.0, label="$\\pm 2\\sigma$ 界限")
    ax.axhline(-2 * s, color=OKABE_ITO[1], linestyle="--", linewidth=1.0)
    ax.plot(fitted, resid, "o", color=OKABE_ITO[3], markersize=3.2)
    _legend(ax, loc="upper right")
    _grid(ax)
    _finish(ax, "(d) 残差 vs 拟合值", "拟合值 $\\hat{y}_t$ / 千件", "残差 $e_t$ / 千件")

    fig.suptitle("预测模型残差诊断（理想情形：无趋势、近似正态、方差齐性、无自相关）")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, out_dir, "forecast_residual_diagnostics.png")


def optimization_pareto(out_dir: Path) -> None:
    """[优化类] 双目标帕累托前沿与支配关系示意。"""
    rng = np.random.default_rng(_SEED + 6)
    n = 90
    f1 = rng.uniform(40.0, 130.0, n)                       # 总成本 / 万元，越小越好
    f2 = 2600.0 / (f1 - 22.0) + rng.normal(0, 5.5, n)      # 碳排放 / 吨，越小越好
    pts = np.column_stack([f1, np.maximum(f2, 20.0)])
    mask = pareto_front(pts)
    front = pts[mask][np.argsort(pts[mask][:, 0])]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(pts[~mask, 0], pts[~mask, 1], s=22, c=INK_MUTED,
               edgecolors=EDGE, linewidths=0.5, label="被支配解（%d 个）" % int((~mask).sum()))
    ax.plot(front[:, 0], front[:, 1], "-", color=OKABE_ITO[1], linewidth=1.4, zorder=3)
    ax.scatter(front[:, 0], front[:, 1], s=46, marker="D", c=OKABE_ITO[1],
               edgecolors=EDGE, linewidths=0.7, zorder=4,
               label="帕累托前沿非支配解（%d 个）" % int(mask.sum()))

    # 标注一条具体的支配关系链，让"支配"这件事在图上可见
    dom_i = int(np.argmin(pts[~mask, 0] + pts[~mask, 1]))
    dom_point = pts[~mask][dom_i]
    better = pts[mask & (pts[:, 0] <= dom_point[0]) & (pts[:, 1] <= dom_point[1])]
    if better.size:
        ref = better[np.argmin(better[:, 0] + better[:, 1])]
        ax.annotate("", xy=(ref[0], ref[1]), xytext=(dom_point[0], dom_point[1]),
                    arrowprops=dict(arrowstyle="->", color=OKABE_ITO[2], lw=1.5))
        ax.scatter([dom_point[0]], [dom_point[1]], s=70, facecolors="none",
                   edgecolors=OKABE_ITO[2], linewidths=1.6, zorder=5)
        ax.text(dom_point[0] + 1.0, dom_point[1] + 6.0, "被支配解 A", fontsize=7.5,
                color=OKABE_ITO[2])
        ax.text(dom_point[0] - 26.0, dom_point[1] - 20.0,
                "解 B 在两个目标上都不劣于 A，\n且至少一个严格更优，故 B 支配 A",
                fontsize=7.5, color=OKABE_ITO[2],
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=OKABE_ITO[2], lw=0.7))

    # 用淡色标出"理想方向"
    ax.annotate("目标改进方向", xy=(48, 30), xytext=(88, 95),
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"),
                fontsize=8, color=INK_SOFT)
    _legend(ax, loc="upper right")
    _grid(ax)
    _finish(ax, "双目标优化的帕累托前沿与支配关系（两目标均为最小化）",
            "目标 1：总成本 $f_1$ / 万元", "目标 2：碳排放量 $f_2$ / 吨")
    fig.tight_layout()
    _save(fig, out_dir, "optimization_pareto.png")


def optimization_convergence(out_dir: Path) -> None:
    """[优化类] 启发式算法多次独立运行的收敛曲线（均值 ± 最差/最好范围）。"""
    rng = np.random.default_rng(_SEED + 7)
    n_runs, n_gen = 30, 200
    gen = np.arange(1, n_gen + 1, dtype=float)
    best_possible = 1840.0
    curves = np.empty((n_runs, n_gen))
    for r in range(n_runs):
        start = rng.uniform(3200.0, 3900.0)
        rate = rng.uniform(0.030, 0.048)
        floor = best_possible + rng.uniform(0.0, 55.0)
        curve = floor + (start - floor) * np.exp(-rate * gen)
        curve = curve + rng.normal(0, 8.0, n_gen) * np.exp(-gen / n_gen)
        curve = np.minimum.accumulate(curve)               # 记录"历史最优"，单调不增
        curves[r] = curve
    mean = curves.mean(axis=0)
    lo, hi = curves.min(axis=0), curves.max(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(gen, lo, hi, color=OKABE_ITO[0], alpha=0.18,
                    label="30 次独立运行的范围（最好—最差）")
    ax.plot(gen, mean, "-", color=OKABE_ITO[0], linewidth=1.8, label="30 次运行均值")
    ax.axhline(best_possible, color=OKABE_ITO[1], linestyle="--", linewidth=1.2,
               label="参考下界（松弛问题最优值 %.0f 万元）" % best_possible)
    # 收敛判据必须由数据算出，不能写成想当然的常数（曾误写为"波动 < 0.5%"）
    tail = mean[n_gen // 2:]
    rel_total = (mean[-1] - mean[n_gen // 2 - 1]) / mean[-1] * 100.0
    rel_step = float(np.max(np.abs(np.diff(tail)))) / mean[-1] * 100.0
    ax.annotate("第 %d→%d 代均值累计变化 %.1f%%，\n末 100 代单代最大变化 %.2f%%，判定收敛"
                % (n_gen // 2, n_gen, rel_total, rel_step),
                xy=(120, mean[119]), xytext=(72, mean[119] + 330),
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.0),
                fontsize=8, color=INK_SOFT,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.6))
    _note(ax, 0.985, 0.93,
          "末代：均值 %.1f，标准差 %.1f 万元" % (mean[-1], curves[:, -1].std(ddof=1)),
          ha="right", fontsize=7.8)
    ax.set_xlim(1, n_gen)
    _legend(ax, loc="upper right")
    _grid(ax)
    _finish(ax, "启发式算法收敛性：30 次独立运行的均值与范围",
            "迭代代数 $g$ / 代", "历史最优目标值 / 万元")
    fig.tight_layout()
    _save(fig, out_dir, "optimization_convergence.png")


def mechanism_sir(out_dir: Path) -> None:
    """[机理类] SIR 模型时间演化曲线 + S-I 相图。"""
    beta, gamma, s0, i0 = 0.42, 0.14, 0.99, 0.01
    days = 120
    t, s, i, r = sir_rk4(beta, gamma, s0, i0, days, dt=0.02)
    r0 = beta / gamma
    peak_idx = int(np.argmax(i))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    ax = axes[0]
    ax.plot(t, s, "-", color=OKABE_ITO[0], linewidth=1.7, label="易感者 $S(t)$")
    ax.plot(t, i, "-", color=OKABE_ITO[1], linewidth=1.7, label="感染者 $I(t)$")
    ax.plot(t, r, "-", color=OKABE_ITO[2], linewidth=1.7, label="移除者 $R(t)$")
    ax.axvline(t[peak_idx], color=INK_MUTED, linestyle=":", linewidth=1.1)
    ax.plot([t[peak_idx]], [i[peak_idx]], "o", color=OKABE_ITO[1], markersize=5)
    ax.annotate("峰值 $I_{max}$=%.3f\n出现于第 %.0f 天" % (i[peak_idx], t[peak_idx]),
                xy=(t[peak_idx], i[peak_idx]), xytext=(t[peak_idx] + 8, i[peak_idx] + 0.12),
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.0), fontsize=8)
    _legend(ax, loc="center right")
    _grid(ax)
    _finish(ax, "(a) SIR 模型状态变量时间演化",
            "时间 $t$ / 天", "人口比例 / 无量纲")

    ax = axes[1]
    ax.plot(s, i, "-", color=OKABE_ITO[3], linewidth=1.8)
    ax.scatter([s[0]], [i[0]], s=40, color=INK, zorder=4)
    ax.text(s[0] - 0.02, i[0] + 0.012, "起点 $(S_0, I_0)$", fontsize=8, ha="right")
    ax.scatter([s[peak_idx]], [i[peak_idx]], s=45, color=OKABE_ITO[1], zorder=4)
    ax.text(s[peak_idx] + 0.02, i[peak_idx] + 0.01, "峰值点", fontsize=8)
    mid = int(0.45 * peak_idx) if peak_idx > 4 else 1
    ax.annotate("", xy=(s[mid + 1], i[mid + 1]), xytext=(s[mid], i[mid]),
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1.4))
    ax.axvline(1.0 / r0, color=OKABE_ITO[2], linestyle="--", linewidth=1.1,
               label="阈值 $S=1/R_0$")
    _legend(ax, loc="upper right")
    _grid(ax)
    _finish(ax, "(b) $S$-$I$ 相图（箭头为时间推进方向）",
            "易感者比例 $S$ / 无量纲", "感染者比例 $I$ / 无量纲")

    fig.suptitle("SIR 传染病模型：$\\beta$=%.2f 天$^{-1}$，$\\gamma$=%.2f 天$^{-1}$，"
                 "$R_0=\\beta/\\gamma$=%.2f" % (beta, gamma, r0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "mechanism_sir.png")


def mechanism_param_sensitivity(out_dir: Path) -> None:
    """[机理类] 参数敏感性：感染者曲线族 + 归一化蜘蛛图。"""
    beta0, gamma0, s00, i00, days = 0.42, 0.14, 0.99, 0.01, 120
    t, s, i, r = sir_rk4(beta0, gamma0, s00, i00, days, dt=0.02)
    base_peak = float(i.max())
    base_cum = float(r[-1])
    base_peak_t = float(t[int(np.argmax(i))])
    base_dur = float(t[np.argmax(r > 0.99 * r[-1])]) if r[-1] > 0 else float(days)

    fig = plt.figure(figsize=(10.0, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    betas = [0.30, 0.36, 0.42, 0.48, 0.54]
    for k, b in enumerate(betas):
        tk, _, ik, _ = sir_rk4(b, gamma0, s00, i00, days, dt=0.05)
        ax1.plot(tk, ik, markersize=4.2, markevery=40, linewidth=1.6,
                 label="$\\beta$=%.2f 天$^{-1}$（$R_0$=%.2f）" % (b, b / gamma0),
                 **_series(k))
    _legend(ax1, loc="upper right")
    _grid(ax1)
    _finish(ax1, "(a) 感染率 $\\beta$ 取值对 $I(t)$ 的影响",
            "时间 $t$ / 天", "感染者比例 $I(t)$ / 无量纲")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    categories = ["峰值感染比例", "累计感染比例", "峰值出现时间", "疫情持续时间"]
    base_vals = [base_peak, base_cum, base_peak_t, base_dur]
    perturbs = [("$\\beta$ +20%", 1.2, 1.0, 1.0), ("$\\gamma$ +20%", 1.0, 1.2, 1.0),
                ("$S_0$ −20%", 1.0, 1.0, 0.8)]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax2.plot(angles, [1.0] * (len(categories) + 1), "-", color=INK, linewidth=1.4,
             label="基准情形（=1.0）")
    for k, (name, fb, fg, fs) in enumerate(perturbs):
        tk, _, ik, rk = sir_rk4(beta0 * fb, gamma0 * fg, s00 * fs, i00, days, dt=0.05)
        vals = [float(ik.max()) / base_peak, float(rk[-1]) / base_cum,
                float(tk[int(np.argmax(ik))]) / base_peak_t,
                float(tk[np.argmax(rk > 0.99 * rk[-1])]) / base_dur]
        vals += vals[:1]
        ax2.plot(angles, vals, linewidth=1.5, markersize=4.2, label=name, **_series(k))
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylim(0.0, 2.0)
    ax2.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax2.set_yticklabels(["0.5", "1.0", "1.5", "2.0"], fontsize=6.5)
    _legend(ax2, loc="upper right", bbox_to_anchor=(1.28, 1.10))
    _panel(ax2, "b", "参数扰动的归一化蜘蛛图（基准 = 1.0）")

    fig.tight_layout()
    _save(fig, out_dir, "mechanism_param_sensitivity.png")


def simulation_mc_convergence(out_dir: Path) -> None:
    """[仿真类] 蒙特卡洛估计值 ± 95% 置信区间随样本量的收敛过程。"""
    rng = np.random.default_rng(_SEED + 8)
    p_true = 0.0375
    n_max = 400000
    draws = rng.random(n_max) < p_true
    cum = np.cumsum(draws)
    ns = np.unique(np.round(np.logspace(2, np.log10(n_max), 120)).astype(int))
    ns = ns[(ns >= 100) & (ns <= n_max)]
    p_hat = cum[ns - 1] / ns
    half = 1.96 * np.sqrt(p_hat * (1.0 - p_hat) / ns)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    ax.axhline(p_true, color=OKABE_ITO[1], linestyle="--", linewidth=1.3,
               label="真值 $p$=%.4f" % p_true)
    ax.fill_between(ns, p_hat - half, p_hat + half, color=OKABE_ITO[0], alpha=0.22,
                    label="95% 置信区间")
    ax.plot(ns, p_hat, "-", color=OKABE_ITO[0], linewidth=1.5, label="估计值 $\\hat{p}_N$")
    ax.set_xscale("log")
    _legend(ax, loc="upper right")
    _grid(ax)
    _finish(ax, "(a) 估计值与 95% 置信区间随样本量收敛",
            "样本量 $N$ / 次（对数坐标）", "事件概率估计 $\\hat{p}_N$ / 无量纲")

    ax = axes[1]
    ax.loglog(ns, half, "-", color=OKABE_ITO[3], linewidth=1.6, label="置信区间半宽")
    ref = half[0] * np.sqrt(ns[0] / ns)
    ax.loglog(ns, ref, "--", color=INK, linewidth=1.1, label="理论斜率 $O(N^{-1/2})$")
    _legend(ax, loc="lower left")
    _grid(ax, which="both")
    _finish(ax, "(b) 置信区间半宽随样本量的衰减",
            "样本量 $N$ / 次（对数坐标）", "95% 区间半宽 / 无量纲")

    fig.suptitle("蒙特卡洛仿真收敛性（固定随机种子，逐样本累积估计）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "simulation_mc_convergence.png")


def simulation_queueing(out_dir: Path) -> None:
    """[仿真类] M/M/1 排队指标随利用率变化：理论曲线 + 离散事件仿真校验点。"""
    mu = 6.0                                   # 服务率 / (辆/h)
    rhos = np.linspace(0.10, 0.90, 17)
    lam = rhos * mu
    l_theory = rhos / (1.0 - rhos)
    lq_theory = rhos ** 2 / (1.0 - rhos)
    wq_theory_h = rhos / (mu - lam)            # 单位 h

    rng = np.random.default_rng(_SEED + 9)
    n_rep = 12
    sim_l = np.empty(rhos.size)
    sim_l_err = np.empty(rhos.size)
    sim_wq = np.empty(rhos.size)
    sim_wq_err = np.empty(rhos.size)
    for j, rho in enumerate(rhos):
        vals_l, vals_w = [], []
        for _ in range(n_rep):
            l_sys, _lq, w_q = mm1_simulation(float(rho * mu), mu, 500, rng)
            vals_l.append(l_sys)
            vals_w.append(w_q * 60.0)          # h → min
        sim_l[j] = float(np.mean(vals_l))
        sim_l_err[j] = float(np.std(vals_l, ddof=1) / np.sqrt(n_rep))
        sim_wq[j] = float(np.mean(vals_w))
        sim_wq_err[j] = float(np.std(vals_w, ddof=1) / np.sqrt(n_rep))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    ax.plot(rhos, l_theory, "-", color=OKABE_ITO[0], linewidth=1.7, label="理论 $L=\\rho/(1-\\rho)$")
    ax.errorbar(rhos, sim_l, yerr=1.96 * sim_l_err, fmt="o", color=OKABE_ITO[1],
                markersize=4, capsize=2.5, linewidth=1.1, elinewidth=0.9,
                label="仿真均值 ± 95% 置信区间")
    _legend(ax, loc="upper left")
    _grid(ax)
    _finish(ax, "(a) 平均系统队长随利用率变化",
            "利用率 $\\rho=\\lambda/\\mu$ / 无量纲", "平均队长 $L$ / 辆")

    ax = axes[1]
    ax.plot(rhos, wq_theory_h * 60.0, "-", color=OKABE_ITO[2], linewidth=1.7,
            label="理论 $W_q=\\rho/(\\mu-\\lambda)$")
    ax.errorbar(rhos, sim_wq, yerr=1.96 * sim_wq_err, fmt="s", color=OKABE_ITO[3],
                markersize=4, capsize=2.5, linewidth=1.1, elinewidth=0.9,
                label="仿真均值 ± 95% 置信区间")
    _legend(ax, loc="upper left")
    _grid(ax)
    _finish(ax, "(b) 平均排队等待时间随利用率变化",
            "利用率 $\\rho=\\lambda/\\mu$ / 无量纲", "平均等待时间 $W_q$ / min")

    fig.suptitle("M/M/1 排队系统：服务率 $\\mu$=%.1f 辆/h，每个利用率点 12 次独立仿真×500 位顾客"
                 % mu)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, "simulation_queueing.png")


def statistics_correlation(out_dir: Path) -> None:
    """[统计类] 相关系数热力图（含数值标注，仅显示下三角）。"""
    rng = np.random.default_rng(_SEED + 10)
    names = ["气温\n/℃", "湿度\n/%", "风速\n/(m·s$^{-1}$)", "辐照度\n/(W·m$^{-2}$)",
             "用电量\n/(kW·h)", "负荷率\n/%", "电价\n/(元·kW$^{-1}$h$^{-1}$)", "故障次数\n/次"]
    n_var, n_obs = len(names), 200

    # 用一个可复现的因子结构造出"有强相关也有弱相关"的数据
    latent = rng.normal(size=(n_obs, 3))
    loadings = np.array([
        [-0.75, 0.30, 0.10], [0.62, -0.25, 0.15], [-0.30, 0.55, -0.20], [0.88, -0.10, 0.05],
        [0.80, 0.25, -0.10], [-0.45, -0.35, 0.20], [0.35, 0.40, 0.15], [0.10, -0.20, 0.85],
    ])
    data = latent @ loadings.T + rng.normal(0, 0.45, (n_obs, n_var))
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    shown = np.ma.array(corr, mask=mask)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    im = ax.imshow(shown, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    for a in range(n_var):
        for b in range(n_var):
            if b <= a:
                ax.text(b, a, "%.2f" % corr[a, b], ha="center", va="center", fontsize=6.8,
                        color="white" if abs(corr[a, b]) > 0.62 else "black")
    ax.set_xticks(np.arange(n_var))
    ax.set_xticklabels(names, fontsize=6.8)
    ax.set_yticks(np.arange(n_var))
    ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xticks(np.arange(-0.5, n_var, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_var, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Pearson 相关系数 $r$ / 无量纲")
    _finish(ax, "8 项指标的相关系数矩阵（仅显示下三角，n=%d）" % n_obs,
            "指标（含单位）", "指标（含单位）")
    fig.tight_layout()
    _save(fig, out_dir, "statistics_correlation.png")


def clustering_result(out_dir: Path) -> None:
    """[分类聚类类] K-means 聚类散点图 + 轮廓系数子图。"""
    rng = np.random.default_rng(_SEED + 11)
    centers = np.array([[-3.2, 2.6], [0.4, 3.4], [2.9, -0.6], [-1.4, -3.0]])
    sizes = [110, 90, 70, 55]
    xs, ys = [], []
    for c, s in zip(centers, sizes):
        pts = rng.normal(c, [0.75, 0.85], size=(s, 2))
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
    x = np.column_stack([np.concatenate(xs), np.concatenate(ys)])

    labels, cent, inertia = kmeans(x, 4, rng, n_init=10, max_iter=300)
    sil = silhouette_scores(x, labels)
    order = np.argsort(-sil)
    sil_sorted = sil[order]
    lab_sorted = labels[order]
    avg_sil = float(sil.mean())
    cluster_sil = [float(sil[labels == c].mean()) for c in range(4)]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    for c in range(4):
        m = labels == c
        ax.scatter(x[m, 0], x[m, 1], s=18, alpha=0.8,
                   linewidths=0.4, label="簇 %d（n=%d）" % (c + 1, int(m.sum())),
                   **_series(c, edgecolors=EDGE))
    ax.scatter(cent[:, 0], cent[:, 1], marker="*", s=210, color=INK, zorder=5,
               edgecolors="white", linewidths=0.8, label="簇质心")
    _legend(ax, loc="upper left", fontsize=7.2, ncol=1)
    _grid(ax)
    _finish(ax, "(a) K-means 聚类结果（k=4，SSE=%.1f）" % inertia,
            "标准化特征 1 $z_1$ / 无量纲", "标准化特征 2 $z_2$ / 无量纲")

    ax = axes[1]
    y_pos = 0
    ticks, tick_labels = [], []
    for c in range(4):
        vals = sil_sorted[lab_sorted == c]
        ax.barh(np.arange(y_pos, y_pos + vals.size), vals, height=1.0,
                color=CYCLE[c], edgecolor=EDGE, linewidth=0.5)
        ax.text(-0.055, y_pos + vals.size / 2.0, "簇 %d" % (c + 1), fontsize=7,
                va="center", ha="right", color=CYCLE[c])
        ticks.append(y_pos + vals.size / 2.0)
        tick_labels.append("$\\bar{s}$=%.3f" % cluster_sil[c])
        y_pos += vals.size + 6
    ax.axvline(avg_sil, color=OKABE_ITO[1], linestyle="--", linewidth=1.3,
               label="平均轮廓系数 = %.3f" % avg_sil)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=6.8)
    ax.set_xlim(-0.7, 1.0)
    ax.invert_yaxis()
    _legend(ax, loc="lower left")
    _grid(ax, axis="x")
    _finish(ax, "(b) 各簇轮廓系数（按簇分组、簇内降序）",
            "轮廓系数 $s_i$ / 无量纲", "样本序号 / 个")

    fig.suptitle("K-means 聚类结果与轮廓系数诊断（k=4 为示例取值，实际须给出选 k 依据）")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, "clustering_result.png")


def graph_shortest_path(out_dir: Path) -> None:
    """[图论类] 配送网络拓扑与 Dijkstra 最短路径高亮。"""
    names = ["A 主库", "B 中转", "C 中转", "D 网点", "E 网点", "F 网点",
             "G 网点", "H 网点", "I 网点", "J 网点", "K 网点", "L 网点"]
    keys = [n.split()[0] for n in names]
    pos = {
        "A": (2.0, 9.0), "B": (7.5, 10.5), "C": (9.5, 6.5), "D": (5.0, 6.0),
        "E": (11.0, 2.5), "F": (7.0, 2.0), "G": (3.0, 3.5), "H": (14.0, 8.5),
        "I": (16.5, 5.0), "J": (13.0, 1.5), "K": (18.5, 8.0), "L": (20.0, 3.0),
    }
    edges = [
        ("A", "B", 6.2), ("A", "D", 4.8), ("A", "G", 5.5), ("B", "C", 5.1),
        ("B", "H", 7.4), ("C", "D", 6.0), ("C", "H", 4.6), ("D", "F", 4.4),
        ("D", "G", 3.2), ("E", "F", 5.0), ("E", "I", 5.3), ("F", "G", 4.1),
        ("F", "J", 6.8), ("H", "I", 4.0), ("H", "K", 5.2), ("I", "J", 5.9),
        ("I", "K", 3.4), ("J", "L", 7.1), ("K", "L", 5.6), ("G", "F", 4.1),
    ]
    seen = set()
    undirected: List[Tuple[str, str, float]] = []
    for a, b, w in edges:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        undirected.append((a, b, w))

    graph: Dict[str, List[Tuple[str, float]]] = {k: [] for k in keys}
    for a, b, w in undirected:
        graph[a].append((b, w))
        graph[b].append((a, w))

    source, target = "A", "L"
    dist = dijkstra(graph, source)
    # 回填最短路径（Dijkstra 只给距离，路径需按松弛条件反推）
    path = [target]
    cur = target
    guard = 0
    while cur != source and guard < len(keys):
        guard += 1
        for nb, w in graph[cur]:
            if abs(dist[nb] + w - dist[cur]) < 1e-9:
                cur = nb
                path.append(cur)
                break
        else:
            break
    path = path[::-1]
    path_edges = {tuple(sorted((path[k], path[k + 1]))) for k in range(len(path) - 1)}

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for a, b, w in undirected:
        key = tuple(sorted((a, b)))
        if key in path_edges:
            continue
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "-",
                color=GRID, linewidth=1.0, zorder=1)
        ax.text((pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2, "%.1f" % w,
                fontsize=6.0, color=INK_MUTED, ha="center", va="center", zorder=2)
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        w = dist[b] - dist[a]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], "-",
                color=OKABE_ITO[1], linewidth=2.6, zorder=3, alpha=0.9)
        ax.text((pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2 + 0.32,
                "%.1f" % w, fontsize=6.6, color=OKABE_ITO[1], ha="center", zorder=4)
    for k in keys:
        color = OKABE_ITO[1] if k in path else OKABE_ITO[0]
        ax.scatter([pos[k][0]], [pos[k][1]], s=170, color=color, zorder=5,
                   edgecolors=EDGE, linewidths=0.8)
        ax.text(pos[k][0], pos[k][1], k, fontsize=7.2, color="white", ha="center",
                va="center", zorder=6)
        ax.text(pos[k][0] + 0.45, pos[k][1] - 0.75, names[keys.index(k)], fontsize=6.4,
                color=INK_SOFT, zorder=6)

    ax.set_xlim(0, 23)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    _grid(ax)
    _finish(ax, "配送网络拓扑与 $A\\to L$ 最短路径（橙色，总里程 %.1f km）" % dist[target],
            "节点平面坐标 $x$ / km（示意图，非真实地理坐标）",
            "节点平面坐标 $y$ / km（示意图，非真实地理坐标）")
    _note(ax, 0.99, 0.02, "路径：%s\n边权为里程 / km；灰色为未选中的可选边" % " → ".join(path),
          ha="right", va="bottom", fontsize=7.0)
    fig.tight_layout()
    _save(fig, out_dir, "graph_shortest_path.png")


def spatial_interpolation(out_dir: Path) -> None:
    """[空间类] 散点采样与 IDW 插值等值线图。"""
    rng = np.random.default_rng(_SEED + 12)
    n_sample = 45
    sx = rng.uniform(0.0, 100.0, n_sample)
    sy = rng.uniform(0.0, 80.0, n_sample)
    # 合成一个已知的平滑"降水场"：西南高、东北低，带一个局部高值中心
    field_true = (620.0 + 2.4 * (100.0 - sx) + 1.6 * (80.0 - sy)
                  + 180.0 * np.exp(-(((sx - 62.0) ** 2 + (sy - 58.0) ** 2) / (2 * 18.0 ** 2)))
                  - 150.0 * np.exp(-(((sx - 25.0) ** 2 + (sy - 20.0) ** 2) / (2 * 15.0 ** 2))))
    sz = field_true + rng.normal(0, 18.0, n_sample)

    gx = np.linspace(0.0, 100.0, 140)
    gy = np.linspace(0.0, 80.0, 112)
    GX, GY = np.meshgrid(gx, gy)
    GZ = idw_interpolate(np.column_stack([sx, sy]), sz, GX, GY, power=2.0)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    levels = np.linspace(np.floor(GZ.min() / 25) * 25, np.ceil(GZ.max() / 25) * 25, 13)
    cf = ax.contourf(GX, GY, GZ, levels=levels, cmap=CMAP_SEQ, alpha=0.92)
    cs = ax.contour(GX, GY, GZ, levels=levels[1:-1:2], colors="black", linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=6.4, fmt="%.0f")
    sc = ax.scatter(sx, sy, c=sz, cmap=CMAP_SEQ, s=26, edgecolors=EDGE,
                    linewidths=0.6, zorder=5, vmin=levels[0], vmax=levels[-1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("年降水量 / mm")
    cb2 = fig.colorbar(sc, ax=ax, pad=0.09, fraction=0.032)
    cb2.set_label("实测点降水 / mm", fontsize=8)
    _finish(ax, "IDW（p=2）插值降水量等值线图（黑点为 %d 个虚拟雨量站）" % n_sample,
            "东向坐标 $x$ / km", "北向坐标 $y$ / km")
    _note(ax, 0.985, 0.03, "等值线为插值结果；插值不产生新的极值，\n峰值会被系统性平滑",
          ha="right", va="bottom", fontsize=7.0)
    fig.tight_layout()
    _save(fig, out_dir, "spatial_interpolation.png")


def decision_radar(out_dir: Path) -> None:
    """[决策类] 多方案多指标雷达图。"""
    indicators = ["经济性", "技术可行性", "实施周期", "风险可控性", "环境友好度", "可推广性"]
    schemes = ["方案 A（现状延续）", "方案 B（集中式）", "方案 C（分布式）", "方案 D（混合式）"]
    # 各指标已按"越大越好"正向化并归一到 0-1（无量纲）
    values = np.array([
        [0.35, 0.55, 0.80, 0.45, 0.40, 0.50],
        [0.85, 0.72, 0.40, 0.60, 0.35, 0.55],
        [0.55, 0.68, 0.85, 0.78, 0.82, 0.70],
        [0.72, 0.88, 0.62, 0.85, 0.66, 0.88],
    ])
    weights = np.array([0.24, 0.18, 0.14, 0.16, 0.13, 0.15])
    scores = values @ weights
    best = int(np.argmax(scores))

    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.6, 5.6), subplot_kw=dict(projection="polar"))
    for k in range(values.shape[0]):
        vals = values[k].tolist() + [values[k][0]]
        lw = 2.2 if k == best else 1.3
        alpha = 0.16 if k == best else 0.05
        ax.plot(angles, vals, linewidth=lw, markersize=4.2, alpha=0.95,
                label="%s（加权得分 %.3f）" % (schemes[k], scores[k]),
                **_series(k))
        ax.fill(angles, vals, color=OKABE_ITO[k], alpha=alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6.5)
    ax.set_rlabel_position(96)
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.set_title("四种方案在 6 项指标上的雷达图（指标已正向化并归一化）",
                 loc="left", pad=16)
    ax.text(0.5, -0.20, "评分为相对值，仅用于同一指标体系内的方案间比较",
            transform=ax.transAxes, ha="center", fontsize=7.2, color=INK_SOFT)
    fig.tight_layout()
    _save(fig, out_dir, "decision_radar.png")


# ---------------------------------------------------------------- 调度入口

#: (文件名, 生成函数, 一句话说明)；顺序即 README 中的推荐顺序
FIGURES: List[Tuple[str, object, str]] = [
    ("evaluation_weights.png", evaluation_weights, "评价类：熵权 vs AHP 权重对比条形图"),
    ("evaluation_topsis_rank.png", evaluation_topsis_rank, "评价类：TOPSIS 贴近度与排序"),
    ("evaluation_weights_sensitivity.png", evaluation_weights_sensitivity,
     "评价类：权重 ±20% 扰动的排序变化热力图"),
    ("forecast_models_compare.png", forecast_models_compare, "预测类：三模型预测对比 + 预测区间"),
    ("forecast_residual_diagnostics.png", forecast_residual_diagnostics,
     "预测类：残差诊断四联图"),
    ("optimization_pareto.png", optimization_pareto, "优化类：双目标帕累托前沿与支配关系"),
    ("optimization_convergence.png", optimization_convergence,
     "优化类：启发式算法多次运行收敛曲线"),
    ("mechanism_sir.png", mechanism_sir, "机理类：SIR 时间演化 + 相图"),
    ("mechanism_param_sensitivity.png", mechanism_param_sensitivity,
     "机理类：参数敏感性曲线族 + 蜘蛛图"),
    ("simulation_mc_convergence.png", simulation_mc_convergence,
     "仿真类：蒙特卡洛估计收敛与置信区间"),
    ("simulation_queueing.png", simulation_queueing, "仿真类：M/M/1 排队指标随利用率变化"),
    ("statistics_correlation.png", statistics_correlation, "统计类：相关系数热力图"),
    ("clustering_result.png", clustering_result, "分类聚类：K-means 结果 + 轮廓系数"),
    ("graph_shortest_path.png", graph_shortest_path, "图论类：网络拓扑与最短路径高亮"),
    ("spatial_interpolation.png", spatial_interpolation, "空间类：散点采样 + IDW 插值等值线"),
    ("decision_radar.png", decision_radar, "决策类：多方案多指标雷达图"),
]

#: 当前分辨率，由 main() 根据 --dpi 设定，绘图函数通过它保存
_DPI = FIG_DPI_DEFAULT

#: 当前随机种子，由 main() 根据 --seed 设定；默认即 SEED，故默认输出与历史版本逐字节一致。
#: 注意：`_self_test()` 刻意只读常量 SEED（不读本变量），以便 --seed 不影响自检期望值。
_SEED = SEED


def check_not_blank(path: Path) -> Tuple[float, Tuple[int, int]]:
    """读回已保存的 PNG，确认它不是一张纯色空白图。

    参数:
        path: PNG 文件路径。

    返回:
        (像素标准差, (高, 宽))：标准差接近 0 说明图像是空白。

    算法:
        用 `matplotlib.pyplot.imread` 把 PNG 解码成数组，对全部通道求总体标准差。

    复杂度:
        时间 O(H·W·C)；空间 O(H·W·C)。

    陷阱:
        - 全白图的标准差恰为 0，但"几乎全白"（例如只剩一行标题）标准差也很小，
          因此阈值取 0.005 用于拦住"坐标轴没画出来/数据全空"这类事故；
        - `imread` 返回的是浮点 0-1 数组，别拿它去比 255 的整数阈值。

    参考:
        Matplotlib `image.imread` 文档。
    """
    arr = plt.imread(str(path))
    a = np.asarray(arr, dtype=float)
    return float(a.std()), (int(a.shape[0]), int(a.shape[1]))


def run_all(out_dir: Path, dpi: int, only: str = "") -> Tuple[List[Tuple[str, int, float, Tuple[int, int]]], int]:
    """生成全部（或按子串筛选的）配图并做非空白校验。

    参数:
        out_dir: 输出目录（不存在则创建）。
        dpi: 保存分辨率。
        only: 只生成文件名含该子串的图；空串表示全部。

    返回:
        (records, failures)：每张图为 (文件名, 字节数, 像素标准差, (高, 宽))；failures 为失败张数。

    算法:
        逐个调用生成函数 → savefig → close → imread 回读校验。

    复杂度:
        时间 O(张数 × 单张绘制成本)；空间 O(单张图像)。

    陷阱:
        - 必须 `plt.close(fig)`，否则同时打开十几张图会让字体缓存/内存持续增长；
        - `bbox_inches="tight"` 会改变最终像素尺寸，报告尺寸时应以回读结果为准。

    参考:
        无（工程实现）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Tuple[str, int, float, Tuple[int, int]]] = []
    failures = 0
    for name, func, desc in FIGURES:
        if only and only not in name:
            continue
        t0 = time.perf_counter()
        func(out_dir)  # type: ignore[operator]
        path = out_dir / name
        if not path.is_file():
            print("  [失败] %s 未生成（%s）" % (name, desc))
            failures += 1
            continue
        std, shape = check_not_blank(path)
        size = path.stat().st_size
        flag = ""
        if std <= 0.005:
            flag = "  <-- 疑似空白图！"
            failures += 1
        elif size > 250 * 1024:
            flag = "  <-- 超过 250KB，建议降低 dpi"
        records.append((name, size, std, shape))
        print("  [%5.2fs] %-38s %7.1f KB  %5dx%-5d  std=%.4f%s"
              % (time.perf_counter() - t0, name, size / 1024.0, shape[1], shape[0], std, flag))
    return records, failures


def _self_test() -> Dict[str, float]:
    """脚本内置数值自检：返回确定性结果，供人工或下游脚本断言。

    参数:
        无。

    返回:
        Dict[str, float]: 各项小规模数值结果（全部由固定种子产生，可复现）。

    算法:
        依次调用权重、TOPSIS、预测、SIR、聚类、图论、插值、排队、仿真等内核函数的最小用例。

    复杂度:
        时间 O(10³)；空间 O(10³)。

    陷阱:
        - 自检只覆盖数值内核，不覆盖 matplotlib 渲染；渲染要靠 `run_all` 的非空白校验；
        - 修改任何内核函数都会改变这里的期望值，必须同步更新。

    参考:
        无（本仓库自检约定：固定种子 + 确定性小规模结果）。
    """
    rng = np.random.default_rng(SEED)
    norm = rng.uniform(0.05, 1.0, (8, 4))
    w_ent = entropy_weights(norm)
    w_ahp, lmax, ci, cr = ahp_weights(AHP_MATRIX)
    c = topsis_closeness(norm, w_ent)
    series = np.array([41.0, 45.0, 52.0, 58.0, 63.0, 71.0, 78.0, 86.0])
    gm = gm11_forecast(series, 3)
    _hf, hfc = holt_linear(series, 3, alpha=0.6, beta=0.3)
    _qf, qfc = quadratic_seasonal_forecast(
        np.array([10.0, 14.0, 12.0, 11.0, 13.0, 18.0, 15.0, 14.0, 16.0, 22.0, 19.0, 17.0]), 2, 4)
    t, _s, i, r = sir_rk4(0.42, 0.14, 0.99, 0.01, 120, dt=0.05)
    x = np.vstack([rng.normal([-3, 2], 0.6, (40, 2)),
                   rng.normal([3, -2], 0.6, (40, 2)),
                   rng.normal([0, 4], 0.6, (40, 2))])
    labels, _cent, inertia = kmeans(x, 3, np.random.default_rng(_SEED + 99), n_init=5)
    sil = silhouette_scores(x, labels)
    graph = {"A": [("B", 2.0), ("C", 5.0)], "B": [("C", 1.0)], "C": []}
    d = dijkstra(graph, "A")
    gx, gy = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    gz = idw_interpolate(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                         np.array([10.0, 20.0, 30.0, 40.0]), gx, gy, power=2.0)
    rng_mc = np.random.default_rng(_SEED + 8)
    p_hat = float((rng_mc.random(20000) < 0.0375).mean())
    l_sys, _lq, w_q = mm1_simulation(4.5, 6.0, 500, np.random.default_rng(_SEED + 9))
    front = pareto_front(np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 4.0], [2.5, 2.0]]))
    return {
        "entropy_weight_sum": float(w_ent.sum()),
        "entropy_weight_max": float(w_ent.max()),
        "ahp_cr": round(cr, 6),
        "ahp_lambda_max": round(lmax, 6),
        "ahp_weight_sum": float(w_ahp.sum()),
        "topsis_best_index": int(np.argmax(c)),
        "topsis_best_value": round(float(c.max()), 6),
        "gm11_next": round(float(gm[-1]), 6),
        "holt_next": round(float(hfc[-1]), 6),
        "quad_seasonal_next": round(float(qfc[-1]), 6),
        "sir_peak_infected": round(float(i.max()), 6),
        "sir_final_removed": round(float(r[-1]), 6),
        "sir_peak_day": round(float(t[int(np.argmax(i))]), 6),
        "kmeans_inertia": round(inertia, 6),
        "kmeans_cluster_sizes_max": float(np.bincount(labels).max()),
        "silhouette_mean": round(float(sil.mean()), 6),
        "dijkstra_dist_to_C": float(d["C"]),
        "idw_center": round(float(gz.mean()), 6),
        "mc_p_hat_20000": round(p_hat, 6),
        "mm1_L_sim_rho075": round(l_sys, 6),
        "mm1_Wq_sim_rho075_min": round(w_q * 60.0, 6),
        "pareto_front_count": float(front.sum()),
        "normal_quantile_0975": round(float(normal_quantile([0.975])[0]), 6),
        "ranks_desc_first": float(ranks_desc([0.3, 0.9, 0.5])[1]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：解析参数、配置字体、逐张出图并打印报告。

    参数:
        argv: 参数列表；None 表示取 `sys.argv[1:]`。

    返回:
        int: 退出码。0 成功；1 有图失败或疑似空白；2 缺少依赖。

    算法:
        argparse 解析 → 缺失 matplotlib 则打印安装指引并返回 2 → 配置风格 →
        逐张生成 + 校验 → 汇总打印。

    复杂度:
        时间 O(张数)；空间 O(1)（每张图用完即释放）。

    陷阱:
        - 默认输出目录按**脚本自身位置**推导（`<repo>/assets/gallery`），
          否则从任意 cwd 调用都会把图丢得到处都是；
        - 缺库时必须给出可照抄的 pip 命令，而不是让用户对着 ImportError 猜。

    参考:
        `references/model-library.md` §11 绘图硬要求。
    """
    global _DPI, _SEED
    parser = argparse.ArgumentParser(
        prog="make_figures.py",
        description="生成数学建模竞赛论文配图库（原创合成数据，Agg 非交互后端）。")
    parser.add_argument("--out", default=None,
                        help="输出目录，默认 <仓库根>/assets/gallery")
    parser.add_argument("--dpi", type=int, default=FIG_DPI_DEFAULT,
                        help="输出分辨率，默认 %d（投稿建议 300）" % FIG_DPI_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="随机种子（改变合成数据，不改变图形结构），默认 %d" % SEED)
    parser.add_argument("--only", default="", help="只生成文件名含该子串的图")
    parser.add_argument("--self-test", action="store_true", help="只跑数值自检，不出图")
    args = parser.parse_args(argv)

    if args.self_test:
        print("=" * 68)
        print("make_figures.py 数值自检（固定种子 %d，确定性结果）" % SEED)
        print("=" * 68)
        results = _self_test()
        for key in sorted(results):
            print("  %-28s %s" % (key, results[key]))
        return 0

    if plt is None:
        print("[错误] 未检测到 matplotlib，无法生成配图。", file=sys.stderr)
        print("       本脚本出图依赖 numpy + matplotlib（`--self-test` 只需 numpy）；", file=sys.stderr)
        print("       请在本机执行：", file=sys.stderr)
        print("           python -m pip install matplotlib numpy", file=sys.stderr)
        print("       当前解释器：%s" % sys.executable, file=sys.stderr)
        print("       原始错误：%r" % (_MPL_ERROR,), file=sys.stderr)
        return 2

    if args.seed != SEED:
        print("[提示] --seed 只重抽合成数据，图形结构不变；请据此重新核对图注中的数值。")

    _DPI = max(72, min(600, int(args.dpi)))
    _SEED = int(args.seed)
    out_dir = Path(args.out).resolve() if args.out else DEFAULT_OUT

    t_start = time.perf_counter()
    print("=" * 68)
    print("数学建模竞赛配图库生成（全部为脚本自造的合成数据，无任何第三方素材）")
    print("=" * 68)
    font = configure_style()
    if font == "DejaVu Sans":
        print("[警告] 未找到中文字体（Microsoft YaHei / SimHei），汉字可能显示为方框。")
        print("       Windows 请安装'微软雅黑'或'黑体'，Linux 可安装 fonts-noto-cjk。")
    else:
        print("[字体] 中文字体解析结果：%s（axes.unicode_minus=False 已生效）" % font)
    print("[输出] %s" % out_dir)
    print("[参数] dpi=%d，随机种子=%d\n" % (_DPI, args.seed))

    records, failures = run_all(out_dir, _DPI, args.only)
    elapsed = time.perf_counter() - t_start

    print("\n" + "-" * 68)
    if records:
        total_bytes = sum(r[1] for r in records)
        mins = min(r[2] for r in records)
        print("共生成 %d 张图，合计 %.1f KB，最小像素标准差 %.4f（> 0.005 视为非空白）"
              % (len(records), total_bytes / 1024.0, mins))
        biggest = max(records, key=lambda r: r[1])
        print("最大文件：%s（%.1f KB）" % (biggest[0], biggest[1] / 1024.0))
    print("总耗时：%.2f 秒" % elapsed)
    print("失败：%d 张" % failures)
    print("-" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())