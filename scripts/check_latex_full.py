#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`assets/latex/full/` 下四套「完整文档类」模板的「真编译」体检。

为什么要有这个脚本（和 `check_latex.py` 的分工）:
    `assets/latex/{cumcm,yjs,mcm}/main.tex` 是**自带的自写精简模板**，只有一
    个 `main.tex` + `refs.bib`，`check_latex.py` 负责它们。

    `assets/latex/full/{hwcup2026,gmcm,cumcm,mcm}/` 是**完整文档类版本**：带
    `.cls`、`.sty`、`figures/`，`gmcm` 那份还随包带了 5 个中文字体 `.ttf`。
    这些模板是直接拿去投稿用的，能不能编过只有真编一遍才知道；而它们和精简版
    有两处本质差别，精简版的体检脚本覆盖不到：

      1. **不是两个文件**。必须整目录拷贝（`.cls`/`.sty`/`figures/`/字体都在
         旁边），少拷一个 `figures/f1.png` 就报 `File not found`。
      2. **字体要跨平台**。`hwcup2026.cls` 探测系统里的 SimSun / SimHei，
         研赛的 `gmcmthesis.cls` 用随包的 `SimSun.ttf` 等字体（字形和 Word
         一致），国赛的 `cumcmthesis.cls` 会调 Windows 自带的 Times New
         Roman / Arial。这些在 Overleaf / Linux 上都没有，必须能回落到别处。
         所以脚本对这几套模板额外跑一遍**「模拟没有 Windows 字体的机器」**：
         删掉随包 `.ttf`、把 ctex 的字体集钉成 `fandol`、把字体探测的名字换成
         一定不存在的名字，逼程序走回落分支。两条路径都编得过，模板才算真的
         可移植。

    注意回落目标不一样，CI 要装的系统字体也不一样：

      * `hwcup2026` 回落到 **Noto Serif / Sans CJK SC** 与 **Liberation
        Serif**，由 Ubuntu 的 `fonts-noto-cjk` + `fonts-liberation` 提供；
      * `gmcm` / `cumcm` 回落到 **TeX Gyre**（Termes / Heros / Cursor）与
        `fandol`，随 TeX Live 分发，不需要额外系统字体。

    脚本只往系统临时目录写文件，**不碰仓库里的任何模板**，也不在仓库里留下
    `.aux` / `.log` / `.pdf` 之类的编译垃圾。

关于「先删掉预编译好的 PDF」:
    每套模板目录里都放了一份作者预编译的样张（`preview.pdf` /
    `MathModel.pdf` / `example.pdf` / `mcmthesis-demo.pdf`）。它们**不能**
    拷进临时目录，否则编译失败时旧 PDF 还在，`pdf.exists()` 照样为真，体检就
    被骗过去了。只删与主文件同名的那一个（如 `main.tex` → `main.pdf`），
    `figures/` 里的图片 PDF 照常保留；`preview.pdf` 与主文件不同名，不会冒充
    编译产物。

用法:
    python scripts/check_latex_full.py                  # 缺 TeX 时跳过（退出码 0）
    python scripts/check_latex_full.py --require        # 缺 TeX 时视为失败（CI 用这个）
    python scripts/check_latex_full.py --only gmcm      # 只测一套（可重复）
    python scripts/check_latex_full.py --no-simulate    # 不跑「模拟无 Windows 字体」那一遍
    python scripts/check_latex_full.py --keep           # 保留临时目录，便于翻 .log
    python scripts/check_latex_full.py --tex-dir DIR    # 把 DIR 加到 PATH 最前面找引擎
    python scripts/check_latex_full.py --self-test      # 只测改写逻辑，不需要装 TeX

退出码:
    0 = 全部（模板 × 情形）编译通过或按约定跳过；1 = 有失败。

只依赖标准库；引擎由 PATH 上的 TeX 发行版提供。
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 日志解析规则、引擎查找、命令执行都只维护一份，直接复用 check_latex.py——
# 两个脚本对「什么算编译失败」的口径必须一致，否则会出现「精简版判失败、
# 完整版判通过」这种自相矛盾。
from check_latex import (  # noqa: E402
    MIN_PDF_BYTES,
    _run,
    _short,
    analyse_log,
    find_engines,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LATEX_FULL_DIR = REPO_ROOT / "assets" / "latex" / "full"

# ------------------------------------------------------------------ 模板元数据
# bundled_fonts : 随包携带的中文字体，删掉它们要能自动回落
# simulate_linux: 要不要再跑一遍「模拟无 Windows 字体的机器」
# order         : 可选的合规性顺序核对（AI 声明 vs 参考文献），查 .tex 源码
TEMPLATES = (
    {
        "name": "hwcup2026",
        "title": "2026 华为杯（第二十三届中国研究生数学建模竞赛）严格格式版",
        "engine": "xelatex",
        "entry": "main.tex",
        "min_pages": 3,              # 样张 = 封面 + 摘要 + 短正文
        # 这份模板不随包带字体：封面与摘要抬头是官方附件3的位图，
        # 正文字体靠探测系统里的 SimSun / SimHei。
        "bundled_fonts": (),
        "simulate_linux": True,      # 无 SimSun/SimHei 时回落到 Noto CJK + Liberation Serif
        # 2026 华为杯《论文格式规范》没要求 AI 使用声明，故不核对顺序
        "order": None,
    },
    {
        "name": "gmcm",
        "title": "中国研究生数学建模竞赛（华为杯 / 研赛）",
        "engine": "xelatex",
        "entry": "MathModel.tex",
        "min_pages": 6,
        "bundled_fonts": ("SimSun.ttf", "SimHei.ttf", "KaiTi.ttf", "LiSu.ttf", "STXinwei.ttf"),
        "simulate_linux": True,
        # 研赛这份是文档类作者的排版样张，不含赛事要求的 AI 声明，故不核对顺序
        "order": None,
    },
    {
        "name": "cumcm",
        "title": "全国大学生数学建模竞赛（国赛 / CUMCM）",
        "engine": "xelatex",
        "entry": "example.tex",
        "min_pages": 8,
        "bundled_fonts": (),
        "simulate_linux": True,      # cumcmthesis.cls 无条件调 Times New Roman / Arial
        "order": {
            "label": "AI 工具使用声明",
            "ai": (r"\\CumcmAINotUsed\b", r"\\CumcmAIUsed\{"),
            "bib": (r"\\begin\{thebibliography\}",),
            "ai_before_bib": True,   # 国赛《AI 工具使用规定》：声明必须在参考文献之前
        },
    },
    {
        "name": "mcm",
        "title": "美国大学生数学建模竞赛（美赛 MCM / ICM）",
        "engine": "pdflatex",        # mcmthesis 默认 CTeX=false，纯 article，pdfLaTeX 即可
        "entry": "mcmthesis-demo.tex",
        "min_pages": 8,
        "bundled_fonts": (),
        "simulate_linux": False,     # 不调任何 Windows 字体，也没有中文正文
        "order": {
            "label": "Report on Use of AI",
            "ai": (r"\\AImatter\b",),
            "bib": (r"\\begin\{thebibliography\}",),
            "ai_before_bib": False,  # COMAP：AI 报告排在 25 页正文之后，不计入页数
        },
    },
)

#: 只用来判断「有没有 Windows 字体」的字体名。模拟无 Windows 字体的机器时，
#: 把这些**探测用的名字**换成一个一定不存在的名字，逼程序走回落分支。
#:
#: 前三个是 `\IfFontExistsTF{Times New Roman}` 这种西文探测；后两个是
#: `hwcup2026.cls` 里 `\IfFontExistsTF{SimSun}` 的中文探测。带上花括号做匹配，
#: 所以 `gmcmthesis.cls` 的 `\IfFontExistsTF{SimSun.ttf}`（带扩展名，走的是
#: 「随包字体在不在」另一条逻辑）不会被误伤。
WINDOWS_FONT_PROBES = ("Times New Roman", "Courier New", "Arial", "SimSun", "SimHei")
BLIND_FONT_NAME = "NoSuchWindowsFontZZZ"

#: 模拟时把 ctex 的自动字体集钉成 fandol（随 TeX Live / MiKTeX 分发，
#: 任何平台都有），等价于 Overleaf / Linux 上 ctex 自己会做的选择。
SIMULATED_FONTSET = "fandol"

#: 匹配 `\LoadClass[...]{ctexart}` / `\RequirePackage[...]{ctex}` 两种入口
_RE_CTEX_LOAD = re.compile(
    r"\\(?P<cmd>LoadClass|RequirePackage)(?:\[(?P<opts>[^\]]*)\])?\{(?P<pkg>ctexart|ctex)\}")


# ------------------------------------------------------------ 临时副本的改写
def force_fontset_text(text: str, fontset: str = SIMULATED_FONTSET) -> tuple[str, int]:
    """把 ctex / ctexart 的字体集选项钉成 `fontset=<fontset>`，返回 (新文本, 次数)。

    已有 `fontset=xxx` 就替换值，没有就补上这个选项。独立成纯函数是为了能在
    `--self-test` 里直接测，不需要装 TeX。
    """
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        count += 1
        opts = match.group("opts") or ""
        if "fontset=" in opts:
            opts = re.sub(r"fontset=[^,\]]*", "fontset=" + fontset, opts)
        else:
            opts = (opts + "," if opts else "") + "fontset=" + fontset
        return "\\%s[%s]{%s}" % (match.group("cmd"), opts, match.group("pkg"))

    return _RE_CTEX_LOAD.sub(repl, text), count


def blind_windows_fonts_text(text: str) -> tuple[str, int]:
    """把 `\\IfFontExistsTF{<Windows 字体>}` 的探测名换成不存在的名字。

    只改探测参数、不改真分支的动作：这样在装了 Windows 字体的机器上也会走
    回落分支，等价于「在一台没有 Times New Roman 的 Linux 上编译」。
    """
    count = 0
    for name in WINDOWS_FONT_PROBES:
        needle = "\\IfFontExistsTF{%s}" % name
        hits = text.count(needle)
        if hits:
            count += hits
            text = text.replace(needle, "\\IfFontExistsTF{%s}" % BLIND_FONT_NAME)
    return text, count


def simulate_no_windows_fonts(work: pathlib.Path, bundled_fonts: tuple[str, ...]) -> list[str]:
    """就地把临时副本改造成「一台没有 Windows 字体的机器」，返回说明列表。"""
    notes: list[str] = []
    removed = [f for f in bundled_fonts if (work / f).exists()]
    for fname in removed:
        (work / fname).unlink()
    if removed:
        notes.append("临时副本已删除随包字体 %d 个（%s）"
                     % (len(removed), "、".join(removed)))
    else:
        notes.append("这套模板没有随包字体，无需删除")

    for src in sorted(work.glob("*.cls")) + sorted(work.glob("*.sty")):
        text = src.read_text(encoding="utf-8")
        patched, n_fontset = force_fontset_text(text)
        patched, n_blind = blind_windows_fonts_text(patched)
        if patched != text:
            src.write_text(patched, encoding="utf-8")
        if n_fontset or n_blind:
            notes.append("临时副本改写 %s：字体集钉成 %s（%d 处）、屏蔽 Windows 字体探测（%d 处）"
                         % (src.name, SIMULATED_FONTSET, n_fontset, n_blind))
    return notes


# ------------------------------------------------------------------ 顺序核对
def check_order(tex_text: str, order: dict) -> tuple[bool, str]:
    """核对「AI 声明」与「参考文献」在 .tex 源码里的先后顺序。

    位置取法两边不一样，这不是随意为之：

    * **AI 声明取第一处**。它只该出现一次（`\\CumcmAINotUsed` / `\\CumcmAIUsed`
      / `\\AImatter`），第一处就是真正生效的那处。
    * **参考文献取最后一处**。`example.tex` 里在讲「怎么写参考文献」的那节中，
      把 `\\begin{thebibliography}` 当成代码样例贴了一遍（第 202 行），而真正
      排版用的那份在第 534 行。取第一处会把这个样例当成正文的参考文献，于是
      把「AI 声明在参考文献之前」错判成不合规。
    """
    def positions(patterns) -> list[int]:
        # 元数据里写成 `(r"...")` 会退化成字符串，`for p in ...` 就变成逐字符遍历，
        # 于是 re 收到单个反斜杠并抛 "bad escape (end of pattern)"。这里统一收口，
        # 字符串一律当作「只有一个模式」处理，不给它退化成字符序列的机会。
        if isinstance(patterns, str):
            patterns = (patterns,)
        return [m.start() for p in patterns for m in re.finditer(p, tex_text)]

    bib_hits = positions(order["bib"])
    ai_hits = positions(order["ai"])
    label = order["label"]

    if not bib_hits:
        return False, "源码里找不到参考文献环境"
    if not ai_hits:
        return False, "源码里找不到 %s 相关命令" % label

    bib_at = max(bib_hits)
    ai_at = min(ai_hits)

    note = ""
    if len(bib_hits) > 1:
        note = "（正文含 %d 处，按最后一处算）" % len(bib_hits)

    if order["ai_before_bib"]:
        if ai_at < bib_at:
            return True, "%s 在参考文献之前 ✓%s" % (label, note)
        return False, "%s 在参考文献之后 ✗（规定要求在其之前）%s" % (label, note)
    if bib_at < ai_at:
        return True, "%s 在参考文献之后 ✓%s" % (label, note)
    return False, "%s 在参考文献之前 ✗（COMAP 要求在其之后）%s" % (label, note)


# -------------------------------------------------------------------- 真编译
def prepare(work: pathlib.Path, tpl: dict) -> None:
    """把整套模板拷进临时目录，并删掉作者预编译的样张 PDF。"""
    src = LATEX_FULL_DIR / tpl["name"]
    shutil.copytree(src, work, dirs_exist_ok=True)
    stale = work / (tpl["entry"][:-4] + ".pdf")
    if stale.exists():
        stale.unlink()


def run_case(tpl: dict, work: pathlib.Path, timeout: int,
             simulate: bool) -> dict:
    """编译一种情形并体检，返回结果字典。"""
    prepare(work, tpl)
    notes = simulate_no_windows_fonts(work, tpl["bundled_fonts"]) if simulate else []
    if not simulate:
        notes.append("按模板原样编译（用随包字体 / 本机已装的 Windows 字体）")

    tex_text = (work / tpl["entry"]).read_text(encoding="utf-8")
    order_ok, order_msg = (True, "这套模板不核对 AI 声明顺序")
    if tpl["order"]:
        order_ok, order_msg = check_order(tex_text, tpl["order"])

    engine = tpl["engine"]
    # 三套模板都用行内的 thebibliography 环境，不需要跑 bibtex，引擎跑满 3 遍即可
    sequence = [[engine, "-interaction=nonstopmode", tpl["entry"]]] * 3

    run_errors: list[str] = []
    for cmd in sequence:
        try:
            proc = _run(cmd, work, timeout)
        except Exception as exc:                    # noqa: BLE001 —— 超时/启动失败都算问题
            run_errors.append("%s 无法完成：%s" % (cmd[0], exc))
            break
        if proc.returncode != 0:
            run_errors.append("%s 退出码 %d" % (cmd[0], proc.returncode))

    log_path = work / (tpl["entry"][:-4] + ".log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    problems, stats = analyse_log(log_text)
    if not log_text:
        problems.append("没有生成 %s" % log_path.name)
    problems.extend(run_errors)

    pdf = work / (tpl["entry"][:-4] + ".pdf")
    if not pdf.exists():
        problems.append("没有生成 %s" % pdf.name)
    else:
        stats["pdf_bytes"] = pdf.stat().st_size
        if stats["pdf_bytes"] < MIN_PDF_BYTES:
            problems.append("%s 只有 %d 字节，不像是正常排版结果"
                            % (pdf.name, stats["pdf_bytes"]))

    if stats["pages"] is not None and stats["pages"] < tpl["min_pages"]:
        problems.append("只有 %d 页，低于下限 %d 页——模板可能被改空了"
                        % (stats["pages"], tpl["min_pages"]))

    if not order_ok:
        problems.append(order_msg)

    return {
        "name": tpl["name"],
        "case": "模拟无 Windows 字体" if simulate else "原样",
        "engine": engine,
        "notes": notes,
        "order_msg": order_msg,
        "stats": stats,
        "problems": problems,
        "work": work,
        "ok": not problems,
    }


# ------------------------------------------------------------------ 固件测试
def self_test() -> int:
    """不装 TeX 也能跑的固件测试：两处文本改写 + 顺序核对 + 日志口径。"""
    checks: list[tuple[str, bool]] = []

    def check(label: str, cond: bool) -> None:
        checks.append((label, bool(cond)))

    # --- 字体集钉死
    got, n = force_fontset_text(r"\LoadClass[a4paper,cs4size]{ctexart}")
    check("无 fontset 时补上选项",
          n == 1 and got == r"\LoadClass[a4paper,cs4size,fontset=fandol]{ctexart}")

    got, n = force_fontset_text(r"\LoadClass[a4paper,fontset=windows]{ctexart}")
    check("已有 fontset 时替换值",
          n == 1 and got == r"\LoadClass[a4paper,fontset=fandol]{ctexart}")

    got, n = force_fontset_text(r"\RequirePackage{ctex}")
    check("\\RequirePackage{ctex} 无选项也能补",
          n == 1 and got == r"\RequirePackage[fontset=fandol]{ctex}")

    got, n = force_fontset_text(r"\RequirePackage[UTF8,fontset=adobe]{ctex}")
    check("多选项里只动 fontset",
          n == 1 and got == r"\RequirePackage[UTF8,fontset=fandol]{ctex}")

    got, n = force_fontset_text(r"\RequirePackage{graphicx}\n\usepackage{amsmath}")
    check("不认识 ctex 时不动任何东西", n == 0 and got == r"\RequirePackage{graphicx}\n\usepackage{amsmath}")

    # --- 屏蔽 Windows 字体探测
    got, n = blind_windows_fonts_text(
        r"\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}"
        r"{\setmainfont{TeX Gyre Termes}}")
    check("探测名被换掉、动作保持不变",
          n == 1 and "IfFontExistsTF{%s}" % BLIND_FONT_NAME in got
          and r"{\setmainfont{Times New Roman}}" in got
          and r"{\setmainfont{TeX Gyre Termes}}" in got)

    _, n = blind_windows_fonts_text(r"\setmainfont{Times New Roman}")
    check("裸的 \\setmainfont 不算探测，不被改", n == 0)

    # --- 顺序核对
    cumcm_order = {
        "label": "AI 工具使用声明",
        "ai": (r"\\CumcmAINotUsed\b", r"\\CumcmAIUsed\{"),
        "bib": (r"\\begin\{thebibliography\}",),
        "ai_before_bib": True,
    }
    ok, _ = check_order("\\CumcmAINotUsed\n\\begin{thebibliography}{9}\n", cumcm_order)
    check("国赛顺序：AI 在参考文献前 → 通过", ok)
    ok, _ = check_order("\\begin{thebibliography}{9}\n\\CumcmAINotUsed\n", cumcm_order)
    check("国赛顺序：AI 在参考文献后 → 失败", not ok)
    ok, _ = check_order("\\begin{thebibliography}{9}\n\\CumcmAIUsed{润色}\n", cumcm_order)
    check("国赛顺序：\\CumcmAIUsed{...} 同样被识别且判失败", not ok)

    mcm_order = {
        "label": "Report on Use of AI",
        "ai": (r"\\AImatter\b",),
        "bib": (r"\\begin\{thebibliography\}",),
        "ai_before_bib": False,
    }
    ok, _ = check_order("\\begin{thebibliography}{99}\n\\AImatter\n", mcm_order)
    check("美赛顺序：AI 在参考文献后 → 通过", ok)
    ok, _ = check_order("\\AImatter\n\\begin{thebibliography}{99}\n", mcm_order)
    check("美赛顺序：AI 在参考文献前 → 失败", not ok)
    ok, msg = check_order("\\AImatter\n", mcm_order)
    check("找不到参考文献时报错", (not ok) and "找不到" in msg)
    ok, msg = check_order("\\begin{thebibliography}{99}\n", mcm_order)
    check("找不到 AI 命令时报错", (not ok) and "找不到" in msg)

    # 回归：example.tex 在讲解「怎么写参考文献」时把 thebibliography 当样例贴了一遍，
    # 真正排版用的那份在后面。所在位置必须按「最后一处」算，否则会错判成不合规。
    demo_then_real = (
        "\\section{怎么写参考文献}\n"
        "\\begin{lstlisting}\n"
        "\\begin{thebibliography}{9}\n"          # 第 202 行那种代码样例
        "\\end{lstlisting}\n"
        "\\section{参考文献与引用}\n"
        "\\CumcmAINotUsed\n"                    # 第 493 行：真正的 AI 声明
        "\\begin{thebibliography}{9}\n")        # 第 534 行：真正的参考文献
    ok, msg = check_order(demo_then_real, cumcm_order)
    check("正文含样例 thebibliography 时，按最后一处判 → 通过", ok)
    check("多处参考文献时给出提示", "最后一处" in msg)

    # 反过来：AI 声明真的排在最后那份参考文献之后，还得判失败
    real_bib_last = ("\\begin{thebibliography}{9}\n"
                     "\\end{thebibliography}\n"
                     "\\section*{AI工具使用详情}\n"
                     "\\CumcmAINotUsed\n")
    ok, _ = check_order(real_bib_last, cumcm_order)
    check("AI 声明排在真正的参考文献之后 → 仍然判失败", not ok)

    # 回归：元数据里漏了逗号的 `(r"...")` 是字符串不是元组，绝不能退化成逐字符遍历
    scalar_order = dict(mcm_order)
    scalar_order["bib"] = r"\\begin\{thebibliography\}"
    ok, _ = check_order("\\begin{thebibliography}{99}\n\\AImatter\n", scalar_order)
    check("bib 误写成字符串（漏逗号）时不崩且仍判通过", ok)
    ok, _ = check_order("\\AImatter\n\\begin{thebibliography}{99}\n", scalar_order)
    check("bib 误写成字符串（漏逗号）时仍能判失败", not ok)

    # 元数据本身也要挡一道：每个模式字段必须都是元组，不能再犯同一个错
    check("所有模板的 order 模式字段都是元组而非裸字符串",
          all(isinstance(t["order"][k], tuple)
              for t in TEMPLATES if t["order"] for k in ("ai", "bib")))

    # --- 和 check_latex.py 共用同一套日志口径，确认复用没跑偏
    problems, stats = analyse_log("Output written on example.pdf (12 pages, 452166 bytes).\n")
    check("复用 analyse_log：页数/字节数解析一致",
          problems == [] and stats["pages"] == 12 and stats["pdf_bytes"] == 452166)
    problems, _ = analyse_log('! Package fontspec Error: The font "Arial" cannot be found.\n')
    check("复用 analyse_log：缺字体仍判硬失败", any("字体装不上" in p for p in problems))

    # --- 模板表自检
    names = [t["name"] for t in TEMPLATES]
    check("模板名无重复", len(names) == len(set(names)))
    check("模板表覆盖 full/ 下的全部模板目录",
          sorted(names) == sorted(p.name for p in LATEX_FULL_DIR.iterdir() if p.is_dir()))
    check("每个模板的 entry 都真实存在",
          all((LATEX_FULL_DIR / t["name"] / t["entry"]).exists() for t in TEMPLATES))
    check("每个模板的随包字体都真实存在",
          all((LATEX_FULL_DIR / t["name"] / f).exists()
              for t in TEMPLATES for f in t["bundled_fonts"]))
    check("hwcup2026 / cumcm / gmcm 需要模拟无 Windows 字体，mcm 不需要",
          [t["simulate_linux"] for t in TEMPLATES] == [True, True, True, False])
    check("模拟时能屏蔽 SimSun / SimHei 的中文探测",
          blind_windows_fonts_text("\\IfFontExistsTF{SimSun}")[1] == 1
          and blind_windows_fonts_text("\\IfFontExistsTF{SimHei}")[1] == 1)
    check("屏蔽探测名不会误伤 `\\IfFontExistsTF{SimSun.ttf}`（随包字体探测）",
          blind_windows_fonts_text("\\IfFontExistsTF{SimSun.ttf}")[1] == 0)

    failed = [label for label, good in checks if not good]
    for label, good in checks:
        print("  %s %s" % ("PASS" if good else "FAIL", label))
    print()
    print("self-test: %d/%d 通过" % (len(checks) - len(failed), len(checks)))
    return 1 if failed else 0


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="编译 assets/latex/full/ 下的完整文档类模板并体检（不改动仓库文件）")
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="只测指定模板，可重复（hwcup2026 / gmcm / cumcm / mcm）")
    parser.add_argument("--require", action="store_true",
                        help="找不到编译引擎时判为失败（CI 用）；默认是跳过")
    parser.add_argument("--no-simulate", action="store_true",
                        help="不跑「模拟无 Windows 字体」那一遍，只按原样编译")
    parser.add_argument("--keep", action="store_true",
                        help="保留临时编译目录，便于翻 .log / .pdf")
    parser.add_argument("--tex-dir", metavar="DIR",
                        help="把该目录加到 PATH 最前面（例如本机 MiKTeX 的 bin 目录）")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单条命令的超时秒数（默认 300）")
    parser.add_argument("--self-test", action="store_true",
                        help="只跑改写/解析逻辑的固件测试，不需要装 TeX")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not LATEX_FULL_DIR.is_dir():
        print("找不到模板目录：%s" % LATEX_FULL_DIR)
        return 1

    selected = list(TEMPLATES)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {t["name"] for t in TEMPLATES}
        if unknown:
            print("未知模板：%s（可选 %s）"
                  % (", ".join(sorted(unknown)), " / ".join(t["name"] for t in TEMPLATES)))
            return 1
        selected = [t for t in selected if t["name"] in wanted]

    engines = find_engines(args.tex_dir)
    needed = sorted({t["engine"] for t in selected})
    missing = [e for e in needed if not engines.get(e)]

    if missing:
        print("找不到编译引擎：%s" % ", ".join(missing))
        print("本机没装 TeX 发行版（或没把它加进 PATH）。")
        print("  · 想现在就体检：装 TeX Live / MiKTeX，或用 --tex-dir 指向 bin 目录；")
        print("  · 只想跑逻辑自测：python scripts/check_latex_full.py --self-test")
        return 1 if args.require else 0

    work_root = pathlib.Path(tempfile.mkdtemp(prefix="mmlatexfull_"))
    print("临时编译目录：%s" % work_root)
    print("引擎：%s" % "，".join("%s=%s" % (e, engines[e]) for e in needed))
    print()

    results = []
    try:
        for tpl in selected:
            cases = [False]
            if tpl["simulate_linux"] and not args.no_simulate:
                cases.append(True)
            for simulate in cases:
                tag = "模拟无 Windows 字体" if simulate else "原样"
                print("=" * 70)
                print("编译 %s（%s，%s）…" % (tpl["name"], tpl["engine"], tag))
                work = work_root / ("%s-%s" % (tpl["name"], "sim" if simulate else "raw"))
                result = run_case(tpl, work, args.timeout, simulate)
                results.append(result)
                stats = result["stats"]
                print("  页数=%s  PDF字节=%s  Overfull=%d  Underfull=%d"
                      % (stats["pages"], stats["pdf_bytes"], stats["overfull"], stats["underfull"]))
                for note in result["notes"]:
                    print("  · %s" % note)
                print("  · %s" % result["order_msg"])
                for problem in result["problems"]:
                    print("  ✗ %s" % problem)
                if result["ok"]:
                    print("  ✓ 通过")
    finally:
        if args.keep:
            print()
            print("已保留临时目录：%s" % work_root)
        else:
            shutil.rmtree(work_root, ignore_errors=True)

    print()
    print("=" * 70)
    bad = [r for r in results if not r["ok"]]
    for r in results:
        print("%-6s %-9s %-22s %s"
              % (r["name"], r["engine"], r["case"],
                 "OK" if r["ok"] else "FAIL：%s" % _short(r["problems"][0], 70)))
    print()
    print("完整模板：%d/%d 通过" % (len(results) - len(bad), len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
