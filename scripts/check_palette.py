#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配色的可访问性体检：把 `make_figures.py` 里的设计令牌抽出来量一遍。

为什么需要这个脚本:
    "换一套更好看的配色"是配图评审里最容易踩的坑。很多在正常色觉下看着很
    "高级"的配色（seaborn deep、ColorBrewer Set2、matplotlib tab10）在红绿色
    盲模拟下会直接塌掉：相邻两条曲线变成同一个颜色。本脚本用 Machado 等
    (2009) 的 severity-1.0 色觉缺陷矩阵（在**线性 sRGB** 空间施加，不是直接
    乘 sRGB 数值）模拟三类二色觉，再用 CIELAB 的 ΔE*ab 量出"最接近的两个颜色
    有多接近"。同时按 WCAG 2.1 相对亮度算每个颜色对白底的对比度——白底上的
    细线如果对比度不够，打印出来就是一片浅灰。

    结论会被写进 `assets/gallery/README.md` §5.8，所以这里的数字必须可复算。

用法:
    python scripts/check_palette.py            # 打印表格并断言
    python scripts/check_palette.py --quiet    # 只断言，不打印表格
    python scripts/check_palette.py --self-test  # 与 --quiet 同义（CI 里习惯这么写）

退出码:
    0 = 全部断言通过；1 = 有配色回归。

只依赖标准库 + numpy（不导入 matplotlib，CI 里无需装绘图库）。
"""
from __future__ import annotations

import argparse
import ast
import itertools
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES_PY = REPO_ROOT / "scripts" / "make_figures.py"

# 需要从 make_figures.py 里读出来的设计令牌
TOKEN_NAMES = ("OKABE_ITO", "CYCLE", "INK", "INK_SOFT", "INK_MUTED",
               "GRID", "PANEL_BG", "EDGE", "_LINESTYLES", "_MARKERS")

# Machado, Oliveira & Fernandes (2009) severity=1.0 的二色觉模拟矩阵
MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}

# 断言阈值：略低于当前实测值，这样它是"回归警报"而不是"复述现状"。
MIN_WORST_CVD_DE = 12.0     # CYCLE 在三类二色觉下的最小 ΔE 下限
MIN_NORMAL_DE = 18.0        # 正常色觉下的最小 ΔE 下限
# 灰度间隔（WCAG 相对亮度 ×100）的下限。**为什么定得这么低**：6 个颜色既要
# 在白底上够深（保证对比度），又要在二色觉下两两分开（二色觉的分离度主要靠
# 明度），可供分配的明度区间本来就很窄。实测：本仓库 1.1，Okabe-Ito 原始 1.1，
# Tol muted 2.5，Tol bright 0.9，tab10 0.9，seaborn deep 0.5，ColorBrewer Set2 0.4，
# seaborn muted 0.2。
# 也就是说 1.1 已经是这个色度约束下的正常水平，靠调色板本身解决不了灰度打印，
# 必须由 `make_figures.py` 的线型/标记点冗余通道来兜底（见下面的静态断言）。
MIN_GREY_GAP = 1.0
MIN_WCAG_CONTRAST = 2.0     # 任一数据色对白底的最低对比度

# 对照用的常见配色（用于在报告里说明"为什么不换"）
REFERENCE = {
    "Okabe-Ito 原始 8 色": ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                            "#E69F00", "#56B4E9", "#F0E442", "#000000"],
    "seaborn deep": ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                     "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"],
    "ColorBrewer Set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
                         "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "matplotlib tab10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    "Tol muted": ["#CC6677", "#332288", "#DDCC77", "#117733",
                  "#88CCEE", "#882255", "#44AA99", "#999933"],
    "Tol bright": ["#4477AA", "#EE6677", "#228833", "#CCBB44",
                   "#66CCEE", "#AA3377", "#BBBBBB"],
    "seaborn muted": ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F",
                      "#956CB4", "#8C613C", "#DC7EC0", "#797979"],
}


# ------------------------------------------------------------------ 令牌抽取
def _static_value(node, env: dict):
    """对模块级右值做**极小范围**的静态求值。

    支持：字面量、同文件里已解析出的令牌名、以及它们的切片（如 `OKABE_ITO[:6]`）。
    其他任何表达式一律报错退出——宁可检查脚本吵，也不要它悄悄漏检。
    """
    if isinstance(node, ast.Subscript):
        base = _static_value(node.value, env)
        if not isinstance(node.slice, ast.Slice) or node.slice.step is not None:
            raise ValueError("只支持 [a:b] 形式的切片")
        lo = _static_value(node.slice.lower, env) if node.slice.lower else None
        hi = _static_value(node.slice.upper, env) if node.slice.upper else None
        return base[lo:hi]
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError("引用了尚未解析的令牌 %s" % node.id)
        return env[node.id]
    return ast.literal_eval(node)


def read_tokens(path: pathlib.Path = FIGURES_PY) -> dict:
    """从 `make_figures.py` 的模块级赋值里抽出设计令牌。

    参数:
        path: `make_figures.py` 的路径。

    返回:
        dict：令牌名 → 值（字符串或字符串列表）。

    算法:
        用 `ast.parse` 解析源码，按**出现顺序**遍历模块级的 `Assign` 节点，
        只挑目标名在 `TOKEN_NAMES` 里的，右值交给 `_static_value` 求值
        （因此后定义的令牌可以引用先定义的，如 `CYCLE = OKABE_ITO[:6]`）。

    复杂度:
        时间 O(源文件大小)；空间 O(令牌数)。

    陷阱:
        - **不要 `import make_figures`**：那会强制依赖 matplotlib，CI 的
          配色检查步骤就不必装绘图库了；顺带也避开了 import 副作用。
        - 只认字面量和切片。令牌一旦被改成更复杂的表达式，这里会直接报错
          而不是悄悄漏检——这正是想要的行为。

    参考:
        Python `ast` / `ast.literal_eval` 文档。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in TOKEN_NAMES:
                try:
                    found[target.id] = _static_value(node.value, found)
                except (ValueError, SyntaxError) as exc:
                    raise SystemExit(
                        "令牌 %s 无法静态求值，check_palette 读不了：%s"
                        % (target.id, exc))
    missing = [n for n in TOKEN_NAMES if n not in found]
    if missing:
        raise SystemExit("make_figures.py 里缺少设计令牌：%s" % ", ".join(missing))
    return found


# ------------------------------------------------------------------ 色彩数学
def hex2rgb(h: str) -> np.ndarray:
    """`"#RRGGBB"` → 0..1 的 sRGB 三元组。"""
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb2lin(c: np.ndarray) -> np.ndarray:
    """sRGB 传输函数求逆 → 线性光。"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def simulate(rgb: np.ndarray, kind: str) -> np.ndarray:
    """在线性光空间施加二色觉矩阵，再回到 sRGB。"""
    lin = MACHADO[kind] @ srgb2lin(rgb)
    lin = np.clip(lin, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB → CIELAB（D65 白点）。"""
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ srgb2lin(rgb) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz),
                 xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def min_delta_e(colors, kind: str | None = None) -> float:
    """两两 CIELAB ΔE*ab 的最小值；`kind` 不为 None 时先做二色觉模拟。"""
    pts = []
    for h in colors:
        rgb = hex2rgb(h)
        pts.append(lab(simulate(rgb, kind) if kind else rgb))
    return float(min(np.linalg.norm(a - b) for a, b in itertools.combinations(pts, 2)))


def min_grey_gap(colors) -> float:
    """灰度打印时相邻亮度（WCAG 相对亮度）的最小间隔，×100 便于读。"""
    lum = sorted(0.2126 * srgb2lin(hex2rgb(h))[0]
                 + 0.7152 * srgb2lin(hex2rgb(h))[1]
                 + 0.0722 * srgb2lin(hex2rgb(h))[2] for h in colors)
    return float(min(b - a for a, b in zip(lum, lum[1:])) * 100)


def wcag_contrast(fg: str, bg: str = "#FFFFFF") -> float:
    """WCAG 2.1 对比度（1.0 ~ 21.0）；正文要求 ≥ 4.5，粗大图形 ≥ 3.0。"""
    def rel(h):
        lin = srgb2lin(hex2rgb(h))
        return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
    a, b = rel(fg), rel(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# ------------------------------------------------------------------ 主流程
def audit(palette, label: str) -> dict:
    """算出一套配色的全部指标。

    参数:
        palette: 颜色列表（`"#RRGGBB"`）。
        label: 打印用的名字。

    返回:
        dict：`n / normal / protan / deutan / tritan / worst / grey / min_contrast`。

    复杂度:
        时间 O(n²)；空间 O(n)。
    """
    p = min_delta_e(palette, "protan")
    d = min_delta_e(palette, "deutan")
    t = min_delta_e(palette, "tritan")
    return {
        "label": label,
        "n": len(palette),
        "normal": min_delta_e(palette),
        "protan": p, "deutan": d, "tritan": t,
        "worst": min(p, d, t),
        "grey": min_grey_gap(palette),
        "min_contrast": min(wcag_contrast(c) for c in palette),
        "killer": min(palette, key=wcag_contrast),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检查配图配色的可访问性")
    ap.add_argument("--quiet", action="store_true", help="只断言，不打印表格")
    ap.add_argument("--self-test", dest="quiet", action="store_true",
                    help="`--quiet` 的别名，与仓库其它校验脚本的调用习惯保持一致")
    args = ap.parse_args(argv)

    tok = read_tokens()
    cycle = list(tok["CYCLE"])
    full = list(tok["OKABE_ITO"])

    rows = [audit(cycle, "本仓库 CYCLE（数据色循环）"),
            audit(full, "本仓库 OKABE_ITO（全量色板）")]
    for name, cols in REFERENCE.items():
        rows.append(audit(cols, name))
    #: 按标签取行，避免以后往 REFERENCE 里插一行就把下面的下标全错位。
    by_label = {r["label"]: r for r in rows}

    if not args.quiet:
        print("=" * 78)
        print("配图配色可访问性体检（Machado 2009 二色觉模拟 + CIELAB ΔE*ab）")
        print("=" * 78)
        print("%-28s %3s %8s %8s %8s %8s %8s" %
              ("palette", "n", "normal", "protan", "deutan", "tritan", "worst"))
        print("-" * 78)
        for r in rows:
            print("%-28s %3d %8.1f %8.1f %8.1f %8.1f %8.1f" %
                  (r["label"], r["n"], r["normal"], r["protan"],
                   r["deutan"], r["tritan"], r["worst"]))
        print("-" * 78)
        print("判读标准：ΔE ≥ 10 可区分，≥ 20 舒适可区分，< 5 有风险（二色觉下几乎同色）。")
        print()
        print("%-28s %8s %10s %14s" % ("palette", "grey-gap", "min-contrast", "最低对比度色"))
        for r in rows:
            print("%-28s %8.1f %10.2f %14s" %
                  (r["label"], r["grey"], r["min_contrast"], r["killer"]))
        print()
        print("灰度间隔的判读：多人色板（≥6 色）普遍在 0.4~2.5 之间，**没有一个能"
              "达到 5**——因为二色觉的分离度主要靠明度，而白底又限制了可用明度区间。")
        print("所以灰度打印场景必须由线型 + 标记点兜底，不能指望配色自己解决。")
        print("对白底对比度：正文文本 ≥ 4.5，粗线/大色块 ≥ 3.0，细线建议 ≥ 3.0。")
        print()

    facts = dict(
        cycle_worst=rows[0]["worst"], cycle_normal=rows[0]["normal"],
        cycle_grey=rows[0]["grey"], cycle_min_contrast=rows[0]["min_contrast"],
        cycle_killer=rows[0]["killer"], cycle_n=rows[0]["n"],
        full_worst=rows[1]["worst"], deep_worst=by_label["seaborn deep"]["worst"],
        set2_worst=by_label["ColorBrewer Set2"]["worst"],
        tab10_worst=by_label["matplotlib tab10"]["worst"],
    )

    failures = []
    if facts["cycle_worst"] < MIN_WORST_CVD_DE:
        failures.append("CYCLE 二色觉最差 ΔE = %.1f < %.1f"
                        % (facts["cycle_worst"], MIN_WORST_CVD_DE))
    if facts["cycle_normal"] < MIN_NORMAL_DE:
        failures.append("CYCLE 正常色觉 ΔE = %.1f < %.1f"
                        % (facts["cycle_normal"], MIN_NORMAL_DE))
    if facts["cycle_grey"] < MIN_GREY_GAP:
        failures.append("CYCLE 灰度间隔 = %.1f < %.1f"
                        % (facts["cycle_grey"], MIN_GREY_GAP))
    if facts["cycle_min_contrast"] < MIN_WCAG_CONTRAST:
        failures.append("CYCLE 最低对比度 = %.2f（%s）< %.1f"
                        % (facts["cycle_min_contrast"], facts["cycle_killer"],
                           MIN_WCAG_CONTRAST))
    if len(set(cycle)) != len(cycle):
        failures.append("CYCLE 里有重复颜色")
    if not set(cycle).issubset(set(full)):
        failures.append("CYCLE 里有不属于 OKABE_ITO 的颜色")
    if "#000000" in full:
        failures.append("OKABE_ITO 里不允许有纯黑（会和坐标轴/文字抢视觉层级）")
    if "#F0E442" in cycle:
        failures.append("黄色 #F0E442 对白底对比度仅约 1.32，不能进数据色循环")

    # 颜色之外的冗余编码通道：灰度打印时唯一还能救场的两项。
    lines, marks = tok["_LINESTYLES"], tok["_MARKERS"]
    src = FIGURES_PY.read_text(encoding="utf-8")
    if "def _series(" not in src:
        failures.append("make_figures.py 缺少 _series()：线型/标记点冗余通道没接线"
                        "（WCAG 2.1 1.4.1：颜色不得是唯一区分手段）")
    if len(set(lines)) < 3:
        failures.append("线型档位少于 3，缩小到单栏宽度后区分不出来")
    if len(set(marks)) < len(cycle):
        failures.append("标记点形状少于 %d 种，颜色循环里会出现两项都相同的序列"
                        % len(cycle))

    print("[汇总] CYCLE = %d 色；正常色觉最小 ΔE = %.1f；"
          "二色觉最差 ΔE = %.1f；灰度间隔 = %.1f；最低对白底对比度 = %.2f（%s）"
          % (facts["cycle_n"], facts["cycle_normal"], facts["cycle_worst"],
             facts["cycle_grey"], facts["cycle_min_contrast"], facts["cycle_killer"]))
    print("[冗余] 灰度间隔只有 %.1f，黑白打印无法只靠颜色区分；"
          "因此每条曲线同时带线型（%d 档）和标记点（%d 种）。"
          % (facts["cycle_grey"], len(set(lines)), len(set(marks))))
    print("[对照] 二色觉最差 ΔE：seaborn deep %.1f、ColorBrewer Set2 %.1f、"
          "matplotlib tab10 %.1f —— 都低于本仓库 CYCLE 的 %.1f。"
          % (facts["deep_worst"], facts["set2_worst"], facts["tab10_worst"],
             facts["cycle_worst"]))

    if failures:
        print("\n[失败] 配色回归：")
        for f in failures:
            print("  - " + f)
        return 1
    print("\n[通过] 配色可访问性断言全部满足。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
