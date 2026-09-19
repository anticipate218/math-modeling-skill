#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 LaTeX 论文模板「一键拷出来」——从技能包里取出对应竞赛的模板目录。

为什么需要这个脚本:
    `assets/latex/{cumcm,yjs,mcm}/` 里各有一套开箱即用的论文模板，但用户拿到
    技能包以后往往不知道该拷哪两个文件、拷到哪、以及拷完怎么编译。手工步骤是:
    「进 assets/latex → 找到对应赛事目录 → 复制 main.tex 与 refs.bib → 粘贴到
    工作目录 → 回忆 xelatex 还是 pdflatex → 想起中文模板要换字体」。任何一步
    出错都会让人以为模板是坏的。

    本脚本把这一串操作压成一条命令，并且**顺手把容易忘的那几件事说清楚**:
    用哪个引擎、要不要 bibtex、中文模板在非 Windows 上要不要换 `fontset`。

它不联网:
    模板就在技能包里，脚本只是复制 + 说明。名字里的 "download" 指的是
    「从技能包里下载到你的工作目录」，不是从网上下载。

用法:
    python scripts/download_templates.py --list                     # 先看有哪些模板
    python scripts/download_templates.py --contest cumcm            # 拷国赛模板到 ./math-modeling-paper
    python scripts/download_templates.py --contest all --out wj     # 三套都拷，各占一个子目录
    python scripts/download_templates.py --contest yjs --fontset fandol   # 顺手把字体换成 fandol
    python scripts/download_templates.py --contest all --zip templates.zip  # 额外打一个离线包
    python scripts/download_templates.py --self-test                # 不装 TeX 也能跑的固件测试

退出码:
    0 成功；1 出错（模板缺失、目标已存在且未加 --force、zip 写不进去等）。

安全约定（重要）:
    **默认不覆盖已存在的文件**。`main.tex` 一旦被你改过就是你的论文，脚本宁可
    报错退出也不静默覆盖；确实要覆盖请显式加 `--force`。
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 模板元数据（引擎 / 是否需要 bibtex）与字体替换规则都只维护一份，直接复用
# check_latex.py 里的定义——那个脚本负责「真编译」，两边的口径必须一致，
# 否则会出现「下载脚本说用 xelatex、体检脚本说用 pdflatex」这种自相矛盾。
from check_latex import (  # noqa: E402
    FONTSET_FROM,
    FONTSET_TO,
    LATEX_DIR,
    TEMPLATES,
    override_fontset_text,
)

TEMPLATE_FILES = ("main.tex", "refs.bib")

#: 每个模板的人类可读说明（赛事 / 语言 / 一句话）。
TEMPLATE_INFO = {
    "cumcm": ("全国大学生数学建模竞赛（国赛）", "中文", "摘要页起排，正文 ≤30 页，AI 声明排在参考文献之前"),
    "yjs": ("中国研究生数学建模竞赛（华为杯·研赛）", "中文", "摘要页即第 1 页，无承诺书/编号页，禁止页眉"),
    "mcm": ("美赛 MCM / ICM（COMAP）", "英文", "Summary Sheet 独占第 1 页，整份提交 ≤25 页（含附录与代码）"),
}

#: 每个模板的编译命令序列（引擎 → bibtex main → 引擎 ×2；中文模板必须 XeLaTeX）。
def compile_commands(name: str) -> list[str]:
    """返回某个模板的编译命令序列。

    参数:
        name: 模板名，必须是 ``cumcm`` / ``yjs`` / ``mcm`` 之一。

    返回:
        长度 4 的字符串列表：引擎 → ``bibtex main`` → 引擎 → 引擎。
        两遍收尾的引擎调用是为了让 ``\\cite``/``\\ref``/``lastpage`` 收敛
        （美赛模板页眉的 ``Page X of Y`` 至少要两遍才正确）。

    算法:
        从 ``check_latex.TEMPLATES`` 里取该模板的 ``engine`` 字段，按固定序列展开。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        中文模板用 pdfLaTeX 编会直接报错（``ctex`` 需要 XeLaTeX 调系统字体），
        所以这里不允许自定义引擎——要换引擎请改 ``check_latex.TEMPLATES``，
        让「下载」与「体检」两边同时改变。

    参考:
        ``assets/latex/README.md`` 第一节。
    """
    engine = template_meta(name)["engine"]
    return [
        f"{engine} -interaction=nonstopmode main.tex",
        "bibtex main",
        f"{engine} -interaction=nonstopmode main.tex",
        f"{engine} -interaction=nonstopmode main.tex",
    ]


def template_meta(name: str) -> dict:
    """按名字取出 ``check_latex.TEMPLATES`` 里的那条元数据。

    参数:
        name: 模板名。

    返回:
        dict（含 ``engine`` / ``needs_bibtex`` 等键）。

    复杂度:
        时间 O(len(TEMPLATES)) / 空间 O(1)。

    陷阱:
        模板名拼错时抛 ``KeyError`` 会很难看，这里统一转成带可选值的 ``ValueError``。

    参考:
        ``scripts/check_latex.py``。
    """
    for tpl in TEMPLATES:
        if tpl["name"] == name:
            return tpl
    known = ", ".join(t["name"] for t in TEMPLATES)
    raise ValueError(f"未知模板 {name!r}，可选：{known}")


def template_dir(name: str) -> pathlib.Path:
    """返回模板在仓库里的目录，并校验 ``main.tex`` 与 ``refs.bib`` 都在。

    参数:
        name: 模板名。

    返回:
        ``pathlib.Path``，指向 ``assets/latex/<name>``。

    复杂度:
        时间 O(1)（外加两次 ``exists``）。

    陷阱:
        技能包被裁剪过（例如只拷了 SKILL.md 与 references/）时这里会失败——
        报错信息必须说清"缺哪个文件"，而不是让后面的复制操作抛一个
        ``FileNotFoundError``。

    参考:
        无。
    """
    d = LATEX_DIR / name
    missing = [f for f in TEMPLATE_FILES if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"模板目录 {d} 里缺少 {', '.join(missing)}；"
            f"请确认拿到的是完整技能包（assets/latex/ 必须存在）"
        )
    return d


def pick_fontset(requested: str) -> str:
    """决定要不要把中文模板的 ``fontset=windows`` 换成 ``fandol``。

    参数:
        requested: ``"auto"`` / ``"keep"`` / ``"fandol"``。
            auto  —— Windows 上保留 ``fontset=windows``（开箱即用），
                     其他平台（Linux/macOS/Overleaf）自动换成 ``fandol``；
            keep  —— 一律保留原样；
            fandol—— 一律换成 ``fandol``。

    返回:
        ``"keep"`` 或 ``"fandol"``（``auto`` 会被解析成其中一个）。

    算法:
        看 ``sys.platform``：``win32`` 视为有 Windows 字体，其余视为没有。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        "非 Windows 就一定没有 Windows 字体"并不严格成立（有人装了字体或
        用 WSL 挂载了 Windows 字体）。所以这只是**默认值**：换错了编译会报
        ``The font "SimSun" cannot be found``，那时加 ``--fontset keep`` 重来即可。
        真正的判据永远是编译结果，不是这个猜测。

    参考:
        ``assets/latex/README.md`` 第五节「常见编译错误与排查」。
    """
    if requested not in ("auto", "keep", "fandol"):
        raise ValueError(f"--fontset 只能是 auto/keep/fandol，得到 {requested!r}")
    if requested != "auto":
        return requested
    return "keep" if sys.platform.startswith("win") else "fandol"


def copy_template(name: str, dest_dir: pathlib.Path, force: bool = False,
                  fontset: str = "keep", dry_run: bool = False) -> dict:
    """把一个模板复制到目标目录（默认拒绝覆盖已有文件）。

    参数:
        name: 模板名。
        dest_dir: 目标目录；不存在时会创建。
        force: 为 True 时覆盖已存在的文件。
        fontset: ``"keep"`` 或 ``"fandol"``；前者逐字节复制，后者只在
            ``main.tex`` 的**代码行**上替换 ``fontset=windows``。
        dry_run: 为 True 时只计算要做什么，不落盘。

    返回:
        dict：
        ``files``    list，每项 ``{"name","from","to","bytes","fontset_replaced"}``；
        ``skipped``  list，已存在且未覆盖的文件名；
        ``dest``     目标目录字符串。

    算法:
        1. 校验模板目录完整（``template_dir``）；
        2. 逐个文件：已存在且非 force → 记进 ``skipped``，不写；
        3. ``main.tex`` 在 ``fontset="fandol"`` 时先做代码行替换再写盘，
           ``refs.bib`` 原样复制。

    复杂度:
        时间 O(文件总字节数) / 空间 O(最大文件字节数)。

    陷阱:
        ① **不能默认覆盖**：用户改过的 ``main.tex`` 就是他的论文，静默覆盖等于
           毁掉他的工作；所以"已存在"是报错而不是提示。
        ② 字体替换只改代码行：模板文件头的注释里写着
           "中文字体：ctexart + fontset=windows …"，改了会把说明文字改成同义反复
           （这条规则与 ``check_latex.py`` 共用同一个函数，避免两边理解不一致）。

    参考:
        ``check_latex.override_fontset_text``。
    """
    src_dir = template_dir(name)
    records: list[dict] = []
    skipped: list[str] = []
    for fname in TEMPLATE_FILES:
        src = src_dir / fname
        dst = dest_dir / fname
        if dst.exists() and not force:
            skipped.append(fname)
            continue
        replaced = 0
        if fname == "main.tex" and fontset == "fandol":
            text = src.read_text(encoding="utf-8")
            text, replaced = override_fontset_text(text)
            payload: bytes | None = text.encode("utf-8")
        else:
            payload = None
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if payload is None:
                shutil.copyfile(src, dst)
            else:
                dst.write_bytes(payload)
        records.append({
            "name": fname,
            "from": str(src),
            "to": str(dst),
            "bytes": len(payload) if payload is not None else src.stat().st_size,
            "fontset_replaced": replaced,
        })
    return {"files": records, "skipped": skipped, "dest": str(dest_dir)}


def build_zip(names: list[str], zip_path: pathlib.Path, fontset: str = "keep") -> dict:
    """把选中的模板打成 zip（离线包 / release 资源）。

    参数:
        names: 模板名列表。
        zip_path: 输出 zip 的路径。
        fontset: 同 ``copy_template``，作用于包内的 ``main.tex``。

    返回:
        dict：``{"path","bytes","entries"}``，``entries`` 是包内路径列表。

    算法:
        包内结构固定为 ``<模板名>/main.tex`` 与 ``<模板名>/refs.bib``。
        每个条目用**固定时间戳**（1980-01-01）写入，因此同一份源码反复打包
        得到**字节完全相同**的 zip——这样 release 资源的哈希才可复现，
        自测也才能断言"两次打包一致"。

    复杂度:
        时间 O(总字节数) / 空间 O(总字节数)。

    陷阱:
        ``ZipFile.write`` 会把文件的 mtime 写进条目，导致每次打包字节都不同；
        想可复现就必须走 ``writestr`` + 自建 ``ZipInfo``。

    参考:
        Python 标准库 ``zipfile`` 文档。
    """
    entries: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            src_dir = template_dir(name)
            for fname in TEMPLATE_FILES:
                data = (src_dir / fname).read_bytes()
                if fname == "main.tex" and fontset == "fandol":
                    text, _ = override_fontset_text(data.decode("utf-8"))
                    data = text.encode("utf-8")
                info = zipfile.ZipInfo(f"{name}/{fname}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, data)
                entries.append(info.filename)
    return {"path": str(zip_path), "bytes": zip_path.stat().st_size, "entries": entries}


def print_list() -> None:
    """打印模板清单（赛事 / 语言 / 引擎 / 文件与体积 / 编译命令）。

    参数:
        无。

    返回:
        None（只往标准输出写）。

    复杂度:
        时间 O(模板数) / 空间 O(1)。

    陷阱:
        体积要**实测**（``stat().st_size``），不要写死在文档里——模板一改，
        写死的数字就开始撒谎。

    参考:
        无。
    """
    print("可用 LaTeX 论文模板：")
    print()
    for tpl in TEMPLATES:
        name = tpl["name"]
        contest, lang, note = TEMPLATE_INFO[name]
        d = LATEX_DIR / name
        sizes = []
        for fname in TEMPLATE_FILES:
            p = d / fname
            sizes.append(f"{fname} {p.stat().st_size:,} B" if p.is_file() else f"{fname} 缺失")
        print(f"  {name:<6} {contest}")
        print(f"         {lang} · {note}")
        print(f"         引擎 {tpl['engine']} · {'需要' if tpl['needs_bibtex'] else '不需要'} bibtex · "
              + " / ".join(sizes))
        print(f"         编译：{compile_commands(name)[0]}  →  bibtex main  →  引擎 ×2")
        print()


def next_steps(names: list[str], dest: pathlib.Path, fontset: str) -> None:
    """打印复制完成后的「下一步怎么做」。

    参数:
        names: 本次复制了哪些模板。
        dest: 目标目录。
        fontset: 实际采用的字体策略。

    返回:
        None。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        用户最容易漏掉的两件事必须显式说出来：**中文模板要跑 bibtex**、
        **非 Windows 要换 fontset**（否则第一遍编译就红，然后以为模板坏了）。

    参考:
        ``assets/latex/README.md``。
    """
    print("下一步：")
    print()
    for name in names:
        sub = dest if len(names) == 1 else dest / name
        tpl = template_meta(name)
        print(f"  [{name}]  cd {sub}")
        print(f"          " + "\n          ".join(compile_commands(name)))
        if name != "mcm" and fontset == "keep":
            print(f"          注意：模板用的是 {FONTSET_FROM}（调用 Windows 宋体/黑体）。")
            print(f"          非 Windows / Overleaf 上请改成 {FONTSET_TO}，或重跑本脚本并加 --fontset fandol。")
        if name != "mcm" and fontset == "fandol":
            print(f"          已把 {FONTSET_FROM} 换成 {FONTSET_TO}（随 TeX 发行版自带字体，Windows 上也能用）。")
        if tpl["needs_bibtex"]:
            print("          正文里必须有 \\cite{...}，否则 bibtex 会报 I found no \\citation commands。")
        print(f"          模板说明：assets/latex/README.md（写完后用 python scripts/check_latex.py 复验能否编过）")
        print()


def self_test() -> int:
    """不需要 TeX 的固件测试：模板清单、字体替换、复制语义、zip 可复现性。

    参数:
        无。

    返回:
        int，0 表示全部通过，1 表示有断言失败。

    算法:
        逐条断言并把结果累加，最后打印 ``通过 n/m``。覆盖：
        ① 模板清单是 cumcm/yjs/mcm 三个且文件齐全；
        ② ``pick_fontset`` 的三种取值与平台默认；
        ③ 字体替换只动代码行、次数符合事实（中文模板各 1 处，美赛 0 处）；
        ④ 复制到临时目录后字节与源一致、目标目录外没有多余文件；
        ⑤ 已存在且未加 force 时必须跳过（不覆盖用户的论文）；
        ⑥ 同一个模板打两次 zip 必须字节完全相同，且解压回来内容正确。

    复杂度:
        时间 O(模板总字节数) / 空间 O(1)（都在系统临时目录里）。

    陷阱:
        测试必须写在**系统临时目录**里，不能往仓库里落文件；否则 CI 会把
        ``.aux/.pdf`` 之类的垃圾当成源码的一部分。

    参考:
        ``scripts/check_latex.py --self-test``。
    """
    total = 0
    failed = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal total, failed
        total += 1
        if cond:
            print(f"  [ok]   {label}")
        else:
            failed += 1
            print(f"  [FAIL] {label}" + (f"  —— {detail}" if detail else ""))

    names = [t["name"] for t in TEMPLATES]
    print("download_templates.py 固件测试")
    check("模板清单为 cumcm/yjs/mcm", names == ["cumcm", "yjs", "mcm"], str(names))
    for name in names:
        d = LATEX_DIR / name
        check(f"{name} 模板目录存在且含 {TEMPLATE_FILES[0]}+{TEMPLATE_FILES[1]}",
              all((d / f).is_file() for f in TEMPLATE_FILES), str(d))

    check("--fontset 取值为 auto/keep/fandol",
          pick_fontset("keep") == "keep" and pick_fontset("fandol") == "fandol")
    check("auto 在 Windows 上保留 fontset=windows",
          pick_fontset("auto") == ("keep" if sys.platform.startswith("win") else "fandol"),
          sys.platform)
    try:
        pick_fontset("bogus")
        check("非法 --fontset 必须报错", False)
    except ValueError:
        check("非法 --fontset 必须报错", True)

    # 字体替换：只动代码行，注释里的说明文字保持不变
    cumcm_src = (LATEX_DIR / "cumcm" / "main.tex").read_text(encoding="utf-8")
    patched, n_cumcm = override_fontset_text(cumcm_src)
    check("cumcm 代码行里恰好 1 处 %s" % FONTSET_FROM, n_cumcm == 1, f"实际 {n_cumcm}")
    check("替换后出现 %s" % FONTSET_TO, FONTSET_TO in patched)
    doc_line = [ln for ln in patched.split("\n") if "documentclass" in ln]
    check("documentclass 那一行已换成 %s" % FONTSET_TO,
          len(doc_line) == 1 and FONTSET_TO in doc_line[0] and FONTSET_FROM not in doc_line[0],
          str(doc_line))
    # 文件头注释里也有 fontset=windows 的说明文字（模板里共 3 处，1 处在代码上），
    # 替换必须放过注释里的那 2 处，否则"请把 A 换成 B"这句说明会变成同义反复。
    check("注释里的 %s 说明文字原样保留（只剩 2 处）" % FONTSET_FROM,
          patched.count(FONTSET_FROM) == 2, f"实际还剩 {patched.count(FONTSET_FROM)} 处")
    mcm_src = (LATEX_DIR / "mcm" / "main.tex").read_text(encoding="utf-8")
    _, n_mcm = override_fontset_text(mcm_src)
    check("mcm 不含 fontset=windows，替换次数为 0", n_mcm == 0, f"实际 {n_mcm}")

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        dest = tmp / "paper"
        res = copy_template("cumcm", dest, fontset="keep")
        check("复制了 2 个文件", len(res["files"]) == 2 and not res["skipped"], str(res["skipped"]))
        for rec in res["files"]:
            src_bytes = pathlib.Path(rec["from"]).read_bytes()
            check(f"{rec['name']} 逐字节一致", pathlib.Path(rec["to"]).read_bytes() == src_bytes)
            check(f"{rec['name']} 返回值里的字节数与磁盘一致",
                  rec["bytes"] == len(src_bytes))

        # 再拷一次：必须跳过而不是覆盖
        res2 = copy_template("cumcm", dest)
        check("已存在且未 --force 时跳过",
              res2["skipped"] == list(TEMPLATE_FILES) and not res2["files"], str(res2))
        before = (dest / "main.tex").read_bytes()
        copy_template("cumcm", dest, force=True, fontset="fandol")
        after = (dest / "main.tex").read_text(encoding="utf-8")
        after_doc = [ln for ln in after.split("\n") if "documentclass" in ln]
        check("--force 才会覆盖，且改写字体",
              before != after.encode("utf-8") and len(after_doc) == 1
              and FONTSET_TO in after_doc[0] and FONTSET_FROM not in after_doc[0],
              str(after_doc))

        # 目标目录之外不能有文件落盘
        stray = [p.name for p in tmp.iterdir() if p.name not in ("paper",)]
        check("没有往目标目录外写文件", not stray, str(stray))

        z1 = tmp / "a.zip"
        z2 = tmp / "b.zip"
        r1 = build_zip(names, z1)
        r2 = build_zip(names, z2)
        check("zip 条目为 3 套 × 2 文件", len(r1["entries"]) == 6, str(r1["entries"]))
        check("两次打包字节完全相同（可复现）", z1.read_bytes() == z2.read_bytes())
        with zipfile.ZipFile(z1) as zf:
            check("zip 内 main.tex 与源文件一致",
                  zf.read("cumcm/main.tex") == (LATEX_DIR / "cumcm" / "main.tex").read_bytes())
            check("zip 条目结构为 <模板>/<文件>",
                  sorted(zf.namelist()) == sorted(f"{n}/{f}" for n in names for f in TEMPLATE_FILES))

        # 未知模板名要给可读错误，而不是 KeyError / 一堆栈
        try:
            template_meta("nonexistent")
            check("未知模板名必须报 ValueError", False)
        except ValueError:
            check("未知模板名必须报 ValueError", True)
        try:
            template_dir("cumcm2")
            check("不存在的模板目录必须报 FileNotFoundError", False)
        except FileNotFoundError:
            check("不存在的模板目录必须报 FileNotFoundError", True)

    print()
    print(f"通过 {total - failed}/{total}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析参数 → 列出/复制/打包，并打印下一步。

    参数:
        argv: 参数列表；None 表示取 ``sys.argv[1:]``。

    返回:
        int，进程退出码（0 成功 / 1 失败）。

    复杂度:
        时间 O(模板总字节数) / 空间 O(1)。

    陷阱:
        单个竞赛时文件直接落进 ``--out``，多个竞赛时各自落进子目录——
        这两种行为不同，必须在输出里说清楚，否则用户会去错目录找 main.tex。

    参考:
        ``assets/latex/README.md``。
    """
    parser = argparse.ArgumentParser(
        description="从技能包里取出 LaTeX 论文模板（国赛/研赛/美赛），并给出编译命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="例：python scripts/download_templates.py --contest cumcm --out my-paper",
    )
    parser.add_argument("--contest", action="append", default=None,
                        help="cumcm / yjs / mcm / all，可重复；默认 all")
    parser.add_argument("--out", default="math-modeling-paper",
                        help="输出目录，默认 ./math-modeling-paper")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已存在的 main.tex / refs.bib（默认拒绝覆盖）")
    parser.add_argument("--fontset", choices=("auto", "keep", "fandol"), default="auto",
                        help="auto=Windows 保留 fontset=windows、其他平台换 fandol（默认）")
    parser.add_argument("--zip", default=None, metavar="PATH", help="额外打一个离线 zip 包")
    parser.add_argument("--list", action="store_true", help="只列出模板与编译命令")
    parser.add_argument("--self-test", action="store_true", help="跑固件测试（不需要装 TeX）")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    try:
        available = [t["name"] for t in TEMPLATES]
    except Exception as exc:  # pragma: no cover - 只会在包被裁剪时触发
        print(f"错误：读不到模板清单（{exc}）", file=sys.stderr)
        return 1

    if args.list:
        print_list()
        return 0

    picked = args.contest or ["all"]
    if "all" in picked:
        names = list(available)
    else:
        names = []
        for c in picked:
            if c not in available:
                print(f"错误：未知竞赛 {c!r}，可选：{', '.join(available + ['all'])}", file=sys.stderr)
                return 1
            if c not in names:
                names.append(c)

    fontset = pick_fontset(args.fontset)
    dest_root = pathlib.Path(args.out)
    ok = True
    for name in names:
        dest = dest_root if len(names) == 1 else dest_root / name
        try:
            res = copy_template(name, dest, force=args.force, fontset=fontset)
        except (FileNotFoundError, ValueError) as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1
        if res["skipped"]:
            ok = False
            print(f"已存在，未覆盖：{', '.join(str(pathlib.Path(res['dest']) / f) for f in res['skipped'])}")
            print("  → 想覆盖请加 --force；想保留自己改过的论文请换个 --out 目录。", file=sys.stderr)
            continue
        print(f"已复制 {name} → {res['dest']}")
        for rec in res["files"]:
            extra = f"（替换 fontset ×{rec['fontset_replaced']}）" if rec["fontset_replaced"] else ""
            print(f"  {rec['name']:<9} {rec['bytes']:>7,} B{extra}")

    if args.zip:
        try:
            zres = build_zip(names, pathlib.Path(args.zip), fontset=fontset)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"错误：打包失败（{exc}）", file=sys.stderr)
            return 1
        print(f"已打包 {len(zres['entries'])} 个文件 → {zres['path']}（{zres['bytes']:,} B）")

    if ok:
        print()
        next_steps(names, dest_root, fontset)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
