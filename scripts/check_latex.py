#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三个 LaTeX 论文模板的「真编译」体检：把模板拷进临时目录跑完整编译链。

为什么需要这个脚本:
    `assets/latex/{cumcm,yjs,mcm}/main.tex` 是直接发给学生用的模板，模板文件头
    写明了编译顺序（xelatex → bibtex → xelatex → xelatex）。但它们的正确性
    ——宏包是否齐全、`\\cite` 能否解析、`\\ref` 是否收敛、中文字体是否装得上
    ——只有**真正编译一遍**才能确认。CI 若不跑这一步，模板可以一直悄悄地坏：
    仓库全绿，而学生拿到手第一遍就报 `File \\`xxx.sty' not found`。

    脚本把每个模板的 `main.tex` + `refs.bib` 两个文件复制到系统临时目录编译，
    因此**不会往仓库里留下 .aux/.log/.pdf 之类的编译垃圾**，也不改动仓库里的
    任何模板文件。

关于中文字体（这是唯一一处「编译的不是原文件」）:
    国赛/研赛模板用 `fontset=windows`，直接调用 Windows 自带的宋体/黑体，
    在 Windows 上开箱即用；但 Windows 字体在 Linux 上并不存在。所以脚本在
    **临时副本**里把 `fontset=windows` 换成随 TeX 发行版自带、Windows 上同样
    可用的 `fandol`，再编译。仓库里的模板一个字都不动。

    这个替换是「尽力而为」的：如果模板里已经没有 `fontset=windows`（例如改成
    了 `auto`），脚本打印一行说明后按原样编译。真正的判据始终是编译结果——
    字体装不上时 fontspec 会报 `The font "..." cannot be found`，那是硬失败。

除了「能不能编译」，脚本还核对两条**合规性**顺序（查 .tex 源码，不查 PDF）:
    国赛/研赛 —— 「AI 工具使用声明」必须排在参考文献之前；
    MCM/ICM  —— 「Report on Use of AI」排在参考文献之后（即在 25 页正文之外）。
    查源码而不是查 PDF：这先后关系完全由 .tex 里两条命令的先后决定，查源码
    更直接，也省掉一个抽取 PDF 文本的外部依赖。

用法:
    python scripts/check_latex.py                 # 缺 TeX 时跳过（退出码 0）
    python scripts/check_latex.py --require       # 缺 TeX 时视为失败（CI 用这个）
    python scripts/check_latex.py --only cumcm    # 只测一个模板（可重复）
    python scripts/check_latex.py --keep          # 保留临时目录，便于翻 .log
    python scripts/check_latex.py --keep-fontset  # 不替换字体集，编译仓库原件
    python scripts/check_latex.py --tex-dir DIR   # 把 DIR 加到 PATH 最前面找引擎
    python scripts/check_latex.py --self-test     # 只测解析逻辑，不需要装 TeX

退出码:
    0 = 全部模板编译通过（或按约定跳过）；1 = 有模板编译失败 / 产出异常。

只依赖标准库；引擎（xelatex / pdflatex / bibtex）由 PATH 上的 TeX 发行版提供。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LATEX_DIR = REPO_ROOT / "assets" / "latex"

# 每个模板：用什么引擎、要不要 bibtex、AI 声明节标题、AI 声明该在参考文献前还是后。
# 这些值直接对应各模板文件头写明的编译顺序与官方格式要求。
TEMPLATES = (
    {
        "name": "cumcm",
        "engine": "xelatex",
        "needs_bibtex": True,
        "ai_heading": "AI 工具使用声明",
        "ai_before_bib": True,      # 国赛《AI 工具使用规定》：声明必须在参考文献之前
        "min_pages": 4,
    },
    {
        "name": "yjs",
        "engine": "xelatex",
        "needs_bibtex": True,
        "ai_heading": "AI 工具使用声明",
        "ai_before_bib": True,      # 研赛沿用同一位置要求
        "min_pages": 4,
    },
    {
        "name": "mcm",
        "engine": "pdflatex",
        "needs_bibtex": True,
        "ai_heading": "Report on Use of AI",
        "ai_before_bib": False,     # COMAP：AI 报告排在 25 页正文之后，不计入页数
        "min_pages": 4,
    },
)

# 仓库里的模板用 fontset=windows；Linux 上没有这些字体，临时副本里换成 fandol。
FONTSET_FROM = "fontset=windows"
FONTSET_TO = "fontset=fandol"

MIN_PDF_BYTES = 5000        # 小于这个体积基本可以断定不是一份正常排版的论文

# ---------------------------------------------------------------- 日志解析规则
# 硬错误：LaTeX 的报错行一律以 '!' 开头（缺宏包、字体装不上、未定义命令都在这）
_RE_HARD_ERROR = re.compile(r"^!", re.M)
# 未解析引用。新旧 LaTeX 的引号风格不同（`x' 与 'x'），所以不锚定引号字符。
_RE_UNDEF_CITE = re.compile(r"Citation[^\n]{0,300}?undefined")
_RE_UNDEF_REF = re.compile(r"Reference[^\n]{0,300}?undefined")
_RE_UNDEF_ANY = re.compile(r"There were undefined references")
_RE_MISSING_FONT = re.compile(r"font\s+\"[^\"]+\"\s+cannot be found", re.I)
_RE_EMERGENCY = re.compile(r"Emergency stop|Fatal error occurred", re.I)
# 跑完 3 遍还有这条警告，说明交叉引用/页码没收敛
_RE_RERUN = re.compile(r"Rerun to get (?:cross-references|outline) right", re.I)
# 注意字节数是可选的：TeX Live 写 "main.pdf (9 pages, 341464 bytes)."，
# 而 MiKTeX 只写 "main.pdf (9 pages)."——不能把字节数写成必填，否则在 MiKTeX
# 上会把明明编译成功的模板判成"没产出 PDF"。页数取自日志，字节数一律以磁盘
# 上真实的文件大小为准（见 run_template）。
_RE_OUTPUT = re.compile(r"Output written on (\S+?\.pdf) \((\d+) pages?(?:, (\d+) bytes)?\)")
_RE_OVERFULL = re.compile(r"Overfull \\hbox")
_RE_UNDERFULL = re.compile(r"Underfull \\hbox")
# .blg（bibtex 的日志）里真正算失败的只有「打不开 .bib / .bst」
_RE_BLG_FATAL = re.compile(r"^I couldn't open (?:database|style) file", re.M)


def _short(text: str, limit: int = 130) -> str:
    """把日志行压成一行短摘要，方便塞进汇总表。"""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _split_comment(line: str) -> tuple[str, str]:
    """把一行 LaTeX 拆成 (代码部分, 注释部分)；`\\%` 不算注释起点。"""
    for i, ch in enumerate(line):
        if ch == "%" and (i == 0 or line[i - 1] != "\\"):
            return line[:i], line[i:]
    return line, ""


def _strip_comments(text: str) -> str:
    """去掉所有注释，只留真正会被 TeX 执行的代码。

    **为什么必须做这一步**：模板文件头的说明文字里就写着
    `\\bibliography{refs}` 和 `fontset=windows`，按整篇文本搜索会把注释里那句
    当成真的命令，于是"AI 声明在参考文献之前"会被判成不合规（注释里的
    `\\bibliography` 出现在文件第 13 行，而 AI 声明在第 429 行）。
    """
    return "\n".join(_split_comment(line)[0] for line in text.splitlines())


def override_fontset_text(text: str) -> tuple[str, int]:
    """把**代码里**的 `fontset=windows` 换成 `fontset=fandol`，返回 (新文本, 次数)。

    只动代码、不动注释：模板文件头的说明文字里也有 `fontset=windows`，改了会
    把"请把 fontset=windows 换成 fontset=fandol"这句说明改成同义反复，还会让
    计数虚高。独立成纯函数是为了能在 `--self-test` 里测，不需要装 TeX。
    """
    lines = text.split("\n")
    total = 0
    for i, line in enumerate(lines):
        code, comment = _split_comment(line)
        hits = code.count(FONTSET_FROM)
        if hits:
            total += hits
            lines[i] = code.replace(FONTSET_FROM, FONTSET_TO) + comment
    return "\n".join(lines), total


def _first_index(text: str, *patterns: str) -> int | None:
    """返回任一模式在文本中最靠前的位置；都没有则 None。"""
    hits = [m.start() for p in patterns for m in re.finditer(p, text)]
    return min(hits) if hits else None


def check_source_order(tex_text: str, ai_heading: str, ai_before_bib: bool) -> tuple[bool, str]:
    """核对「AI 声明」与「参考文献」在 .tex 源码里的先后顺序。

    只看去掉注释后的代码（见 `_strip_comments`）。返回 (是否合规, 说明)。
    """
    code = _strip_comments(tex_text)
    bib_at = _first_index(code, r"\\bibliography\{", r"\\begin\{thebibliography\}")
    ai_at = _first_index(code, r"\\section\*?\{" + re.escape(ai_heading) + r"\}")

    if bib_at is None:
        return False, r"源码里找不到 \bibliography{...} 或 thebibliography 环境"
    if ai_at is None:
        return False, "源码里找不到 \\section*{%s}" % ai_heading

    if ai_before_bib:
        if ai_at < bib_at:
            return True, "AI 声明在参考文献之前 ✓"
        return False, "AI 声明在参考文献之后 ✗（规定要求在其之前）"
    if bib_at < ai_at:
        return True, "AI 声明在参考文献之后 ✓"
    return False, "AI 声明在参考文献之前 ✗（COMAP 要求在其之后）"


def analyse_log(text: str) -> tuple[list[str], dict]:
    """解析 main.log，返回 (问题列表, 统计字典)。"""
    stats = {
        "hard_errors": [ln.strip() for ln in text.splitlines() if ln.startswith("!")],
        "undef_cites": _RE_UNDEF_CITE.findall(text),
        "undef_refs": _RE_UNDEF_REF.findall(text),
        "missing_fonts": _RE_MISSING_FONT.findall(text),
        "overfull": len(_RE_OVERFULL.findall(text)),
        "underfull": len(_RE_UNDERFULL.findall(text)),
        "pages": None,
        "pdf_bytes": None,
    }
    hit = _RE_OUTPUT.search(text)
    if hit:
        stats["pages"] = int(hit.group(2))
        if hit.group(3) is not None:
            stats["pdf_bytes"] = int(hit.group(3))

    problems: list[str] = []
    if stats["hard_errors"]:
        problems.append("硬错误 %d 条（首条：%s）"
                        % (len(stats["hard_errors"]), _short(stats["hard_errors"][0])))
    if stats["undef_cites"]:
        problems.append("未解析的 \\cite 共 %d 处（首处：%s）"
                        % (len(stats["undef_cites"]), _short(stats["undef_cites"][0])))
    if stats["undef_refs"]:
        problems.append("未解析的 \\ref 共 %d 处（首处：%s）"
                        % (len(stats["undef_refs"]), _short(stats["undef_refs"][0])))
    elif _RE_UNDEF_ANY.search(text):
        problems.append("日志声称存在未解析引用（There were undefined references）")
    if stats["missing_fonts"]:
        problems.append("字体装不上：%s" % _short(stats["missing_fonts"][0]))
    if _RE_EMERGENCY.search(text):
        problems.append("出现 Emergency stop / Fatal error")
    if _RE_RERUN.search(text):
        problems.append("交叉引用未收敛：跑完 3 遍日志里仍有 Rerun to get cross-references right")
    if stats["pages"] is None:
        problems.append("日志里没有 “Output written on main.pdf”——大概率没编译出 PDF")
    return problems, stats


def find_engines(tex_dir: str | None = None) -> dict[str, str | None]:
    """在 PATH（以及可选的 tex_dir）上找编译引擎。"""
    if tex_dir:
        os.environ["PATH"] = str(tex_dir) + os.pathsep + os.environ.get("PATH", "")
    return {name: shutil.which(name) for name in ("xelatex", "pdflatex", "bibtex")}


def _run(cmd: list[str], cwd: pathlib.Path, timeout: int) -> subprocess.CompletedProcess:
    """跑一条命令，吞掉输出（判定完全靠 .log，不靠 stdout）。"""
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def run_template(tpl: dict, work_root: pathlib.Path, timeout: int,
                 keep_fontset: bool = False) -> dict:
    """编译一个模板并体检，返回结果字典。"""
    name = tpl["name"]
    src = LATEX_DIR / name
    work = work_root / name
    work.mkdir(parents=True, exist_ok=True)
    for fname in ("main.tex", "refs.bib"):
        shutil.copy(src / fname, work / fname)

    tex_path = work / "main.tex"
    tex_text = tex_path.read_text(encoding="utf-8")

    notes: list[str] = []
    if keep_fontset:
        notes.append("按 --keep-fontset 保留 %s，编译的就是仓库里那份原件" % FONTSET_FROM)
    else:
        patched, replaced = override_fontset_text(tex_text)
        if replaced:
            tex_path.write_text(patched, encoding="utf-8")
            notes.append("临时副本已把 %s 换成 %s（%d 处）"
                         % (FONTSET_FROM, FONTSET_TO, replaced))
        else:
            notes.append("模板里没有 %s，按原样编译" % FONTSET_FROM)

    order_ok, order_msg = check_source_order(tex_text, tpl["ai_heading"], tpl["ai_before_bib"])

    engine = tpl["engine"]
    sequence = [[engine, "-interaction=nonstopmode", "main.tex"]]
    if tpl["needs_bibtex"]:
        sequence.append(["bibtex", "main"])
    sequence += [[engine, "-interaction=nonstopmode", "main.tex"],
                 [engine, "-interaction=nonstopmode", "main.tex"]]

    run_errors: list[str] = []
    for cmd in sequence:
        try:
            proc = _run(cmd, work, timeout)
        except subprocess.TimeoutExpired:
            run_errors.append("%s 超过 %ds 未结束" % (cmd[0], timeout))
            break
        except OSError as exc:
            run_errors.append("%s 无法启动：%s" % (cmd[0], exc))
            break
        if proc.returncode != 0 and cmd[0] != engine:
            # bibtex 的非零退出码通常是「没有 \citation 命令」这类噪音，
            # 真正的判据在 .blg 与最终的 PDF，所以这里只记录，不直接判失败。
            run_errors.append("%s 退出码 %d" % (cmd[0], proc.returncode))

    log_path = work / "main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    problems, stats = analyse_log(log_text)

    if not log_text:
        problems.append("没有生成 main.log")

    for err in run_errors:
        problems.append(err)

    # bibtex 真的跑出结果了吗
    bbl = work / "main.bbl"
    if tpl["needs_bibtex"] and re.search(r"\\cite[a-z]*\{", tex_text):
        if not bbl.exists() or bbl.stat().st_size == 0:
            problems.append("有 \\cite 但没有生成 main.bbl——bibtex 没成功")
        blg = work / "main.blg"
        if blg.exists():
            blg_text = blg.read_text(encoding="utf-8", errors="replace")
            fatal = _RE_BLG_FATAL.search(blg_text)
            if fatal:
                problems.append("bibtex 报告：%s" % _short(fatal.group(0)))

    pdf = work / "main.pdf"
    if not pdf.exists():
        problems.append("没有生成 main.pdf")
    else:
        # 体积以磁盘上的真实文件为准：MiKTeX 的日志里根本没有字节数
        stats["pdf_bytes"] = pdf.stat().st_size
        if stats["pdf_bytes"] < MIN_PDF_BYTES:
            problems.append("main.pdf 只有 %d 字节，不像是正常排版结果" % stats["pdf_bytes"])

    if stats["pages"] is not None and stats["pages"] < tpl["min_pages"]:
        problems.append("只有 %d 页，低于下限 %d 页——模板可能被改空了"
                        % (stats["pages"], tpl["min_pages"]))

    if not order_ok:
        problems.append(order_msg)

    return {
        "name": name,
        "engine": engine,
        "notes": notes,
        "order_msg": order_msg,
        "order_ok": order_ok,
        "stats": stats,
        "problems": problems,
        "work": work,
        "ok": not problems,
    }


def self_test() -> int:
    """不装 TeX 也能跑的固件测试：日志解析 + 字体替换 + 顺序核对。"""
    checks: list[tuple[str, bool]] = []

    def check(label: str, cond: bool) -> None:
        checks.append((label, bool(cond)))

    clean = ("This is XeTeX\n"
             "Overfull \\hbox (2.0pt too wide) in paragraph at lines 10--11\n"
             "Output written on main.pdf (9 pages, 121403 bytes).\n")
    problems, stats = analyse_log(clean)
    check("干净日志：无问题", problems == [])
    check("干净日志：页数 9", stats["pages"] == 9)
    check("干净日志：字节数 121403", stats["pdf_bytes"] == 121403)
    check("干净日志：Overfull 只计数不判失败", stats["overfull"] == 1 and problems == [])

    # MiKTeX 的日志不写字节数，不能因此把编译成功判成失败
    miktex = "Output written on main.pdf (9 pages).\n"
    problems, stats = analyse_log(miktex)
    check("MiKTeX 风格日志：页数识别为 9", stats["pages"] == 9)
    check("MiKTeX 风格日志：字节数为 None 且不算问题",
          stats["pdf_bytes"] is None and problems == [])

    bad = ("! LaTeX Error: File `siunitx.sty' not found.\n"
           "Output written on main.pdf (1 page, 900 bytes).\n")
    problems, _ = analyse_log(bad)
    check("缺宏包被识别", any("硬错误" in p for p in problems))

    font = ('! Package fontspec Error: The font "SimSun" cannot be found.\n'
            "Output written on main.pdf (3 pages, 50000 bytes).\n")
    problems, stats = analyse_log(font)
    check("缺字体被识别", any("字体装不上" in p for p in problems))
    check("缺字体同时算硬错误", stats["hard_errors"])

    cite = ("LaTeX Warning: Citation `knuth1984' on page 1 undefined on input line 42.\n"
            "Output written on main.pdf (5 pages, 60000 bytes).\n")
    problems, stats = analyse_log(cite)
    check("未解析 \\cite 被识别", any("未解析" in p and "cite" in p for p in problems))
    check("未解析 \\cite 只算 1 处", len(stats["undef_cites"]) == 1)

    # 新版 LaTeX 用的是成对单引号，不能把引号风格写死进正则
    cite_new = ("LaTeX Warning: Citation 'knuth1984' on page 1 undefined on input line 42.\n"
                "Output written on main.pdf (5 pages, 60000 bytes).\n")
    problems, stats = analyse_log(cite_new)
    check("未解析 \\cite（新引号风格）被识别", len(stats["undef_cites"]) == 1)

    ref = ("LaTeX Warning: Reference `fig:one' on page 2 undefined on input line 88.\n"
           "Output written on main.pdf (5 pages, 60000 bytes).\n")
    problems, stats = analyse_log(ref)
    check("未解析 \\ref 被识别", any("未解析" in p and "ref" in p for p in problems))
    check("未解析 \\ref 只算 1 处", len(stats["undef_refs"]) == 1)

    rerun = ("LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.\n"
             "Output written on main.pdf (5 pages, 60000 bytes).\n")
    problems, _ = analyse_log(rerun)
    check("交叉引用未收敛被识别", any("未收敛" in p for p in problems))

    nopdf = "! Emergency stop.\n"
    problems, stats = analyse_log(nopdf)
    check("没产出 PDF 被识别", stats["pages"] is None and len(problems) >= 2)

    patched, n = override_fontset_text(r"\documentclass[12pt,a4paper,fontset=windows]{ctexart}")
    check("字体替换生效", n == 1 and "fontset=fandol" in patched and FONTSET_FROM not in patched)
    _, n0 = override_fontset_text(r"\documentclass[12pt,a4paper]{ctexart}")
    check("没有 fontset=windows 时替换次数为 0", n0 == 0)

    # 模板文件头的注释里也写着 fontset=windows，那是说明文字，不该被替换
    commented = ("%  中文字体：ctexart + fontset=windows 直接调用宋体\n"
                 "\\documentclass[12pt,a4paper,fontset=windows]{ctexart}\n")
    patched, n = override_fontset_text(commented)
    check("只替换代码、不碰注释里的 fontset=windows", n == 1)
    check("注释里的字眼原样保留",
          "%  中文字体：ctexart + fontset=windows 直接调用宋体" in patched)

    # 注释里出现 \bibliography 不能算数（模板第 13 行就是这么写的）
    comment_bib = ("%  若不想用 .bib，可把 \\bibliography{refs} 换成手写列表\n"
                   "\\section*{AI 工具使用声明}\n"
                   "正文\n"
                   "\\bibliography{refs}\n")
    ok, _ = check_source_order(comment_bib, "AI 工具使用声明", True)
    check("顺序核对忽略注释里的 \\bibliography", ok)

    ai_first = ("\\section*{AI 工具使用声明}\n本参赛队未使用任何AI工具。\n"
                "\\bibliography{refs}\n")
    ok, _ = check_source_order(ai_first, "AI 工具使用声明", True)
    check("国赛顺序：AI 在参考文献前 → 通过", ok)
    ok, _ = check_source_order(ai_first, "AI 工具使用声明", False)
    check("国赛顺序放到 MCM 规则下 → 失败", not ok)

    ai_last = "\\bibliography{refs}\n\\section*{Report on Use of AI}\n"
    ok, _ = check_source_order(ai_last, "Report on Use of AI", False)
    check("MCM 顺序：AI 在参考文献后 → 通过", ok)
    ok, _ = check_source_order(ai_last, "Report on Use of AI", True)
    check("MCM 顺序放到国赛规则下 → 失败", not ok)

    ok, msg = check_source_order("\\section{引言}\n", "AI 工具使用声明", True)
    check("找不到节标题时报错", (not ok) and "找不到" in msg)

    failed = [label for label, good in checks if not good]
    for label, good in checks:
        print("  %s %s" % ("PASS" if good else "FAIL", label))
    print()
    print("self-test: %d/%d 通过" % (len(checks) - len(failed), len(checks)))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="编译 assets/latex/ 下的三个论文模板并体检（不改动仓库文件）")
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="只测指定模板，可重复（cumcm / yjs / mcm）")
    parser.add_argument("--require", action="store_true",
                        help="找不到编译引擎时判为失败（CI 用）；默认是跳过")
    parser.add_argument("--keep", action="store_true",
                        help="保留临时编译目录，便于翻 .log / .pdf")
    parser.add_argument("--keep-fontset", action="store_true",
                        help="不做字体集替换，编译仓库里的原件（只有装 Windows 字体时才可能成功）")
    parser.add_argument("--tex-dir", metavar="DIR",
                        help="把该目录加到 PATH 最前面（例如本机 MiKTeX 的 bin 目录）")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单条命令的超时秒数（默认 300）")
    parser.add_argument("--self-test", action="store_true",
                        help="只跑解析逻辑的固件测试，不需要装 TeX")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    selected = list(TEMPLATES)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {t["name"] for t in TEMPLATES}
        if unknown:
            print("未知模板：%s（可选 %s）"
                  % (", ".join(sorted(unknown)), " / ".join(t["name"] for t in TEMPLATES)))
            return 1
        selected = [t for t in TEMPLATES if t["name"] in wanted]

    engines = find_engines(args.tex_dir)
    needed = sorted({t["engine"] for t in selected} | ({"bibtex"} if any(
        t["needs_bibtex"] for t in selected) else set()))
    missing = [e for e in needed if not engines.get(e)]

    if missing:
        print("找不到编译引擎：%s" % ", ".join(missing))
        print("本机没装 TeX 发行版（或没把它加进 PATH）。")
        print("  · 想现在就体检：装 TeX Live / MiKTeX，或用 --tex-dir 指向 bin 目录；")
        print("  · 只想跑逻辑自测：python scripts/check_latex.py --self-test")
        return 1 if args.require else 0

    work_root = pathlib.Path(tempfile.mkdtemp(prefix="mmlatex_"))
    print("临时编译目录：%s" % work_root)
    print("引擎：%s" % "，".join("%s=%s" % (e, engines[e]) for e in needed))
    print()

    results = []
    try:
        for tpl in selected:
            print("=" * 70)
            print("编译 %s（%s）…" % (tpl["name"], tpl["engine"]))
            result = run_template(tpl, work_root, args.timeout, args.keep_fontset)
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
        print("%-6s %-9s %s" % (r["name"], r["engine"],
                                "OK" if r["ok"] else "FAIL：%s" % _short(r["problems"][0], 80)))
    print()
    print("LaTeX 模板：%d/%d 通过" % (len(results) - len(bad), len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
