#!/usr/bin/env python3
"""技能结构校验：检查 SKILL.md frontmatter 是否合规、篇幅是否超限、文件引用是否存在。

仅用 Python 标准库，非交互。可用于本地自检与 CI。

用法：
    python scripts/validate_skill.py            # 校验本技能目录
    python scripts/validate_skill.py <技能目录>
    python scripts/validate_skill.py --strict   # 警告也视为失败（CI 用）

退出码：0 通过；1 存在错误（或用 --strict 时存在警告）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Agent Skills 开放标准的可移植字段；出现其它字段在跨工具分发时会硬报错
ALLOWED_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
MAX_NAME = 64
MAX_DESC = 1024
MAX_COMPAT = 500
MAX_BODY_LINES = 500

# SKILL.md 中引用的相对路径（references/xxx.md、scripts/xxx.py、examples/xxx.py 等）
PATH_REF_RE = re.compile(
    r"(?:references|scripts|assets|evals|examples)/[\w./-]+\.(?:md|py|json|ya?ml|tex|bib)")

# 需要检查"是否已在 SKILL.md 中被引用"的目录（渐进式披露：这些目录里的顶层文件都应有索引）
INDEXED_DIRS = ("references", "scripts", "assets", "evals", "examples")

# 反斜杠路径检测：命中即说明作者把 Windows 路径写进了文档，跨平台会失效
BACKSLASH_RE = re.compile(r"(?:references|scripts|assets|evals|examples)\\[\w.\\-]+")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(prog="validate_skill.py",
                                     description="校验 Agent Skill 的 frontmatter、篇幅与文件引用。")
    parser.add_argument("skill_dir", nargs="?", default=".", help="技能目录（含 SKILL.md），默认当前目录")
    parser.add_argument("--strict", action="store_true",
                        help="把警告也当作失败（CI 用；用于强制索引完整性）")
    args = parser.parse_args(argv)

    root = Path(args.skill_dir).resolve()
    skill = root / "SKILL.md"
    errors: list[str] = []
    warnings: list[str] = []

    if not skill.is_file():
        print(f"错误：找不到 {skill}", file=sys.stderr)
        return 1

    text = skill.read_text(encoding="utf-8")
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    if not m:
        print("错误：SKILL.md 缺少 YAML frontmatter（必须以 --- 开头并闭合）", file=sys.stderr)
        return 1
    fm = m.group(1)
    body = text[m.end():]

    # 1) 字段白名单
    keys = [line.split(":", 1)[0].strip() for line in fm.splitlines()
            if re.match(r"^[A-Za-z][A-Za-z0-9_-]*:", line)]
    illegal = sorted(set(keys) - ALLOWED_FIELDS)
    if illegal:
        errors.append(f"出现可移植规范之外的字段（跨工具分发会硬报错）：{illegal}")
    for required in ("name", "description"):
        if required not in keys:
            errors.append(f"缺少必填字段：{required}")

    # 2) name 规则：与目录同名、小写字母数字与连字符、无首尾/连续连字符、≤64
    name_m = re.search(r"^name:\s*(\S+)\s*$", fm, re.M)
    if name_m:
        name = name_m.group(1)
        if not NAME_RE.fullmatch(name):
            errors.append(f"name 不合规（仅小写字母/数字/连字符，且不得首尾或连续连字符）：{name}")
        if len(name) > MAX_NAME:
            errors.append(f"name 超过 {MAX_NAME} 字符：{len(name)}")
        if name != root.name:
            errors.append(f"name（{name}）必须与技能目录名（{root.name}）一致")

    # 3) description 长度
    desc_m = re.search(r"description:\s*(?:>-?|\|)?\s*\n?([\s\S]*?)\n(?:[a-z-]+):", fm, re.M)
    if desc_m:
        desc = re.sub(r"\s+", " ", desc_m.group(1)).strip()
        if not desc:
            errors.append("description 为空")
        if len(desc) > MAX_DESC:
            errors.append(f"description 超过 {MAX_DESC} 字符：{len(desc)}")
        elif len(desc) > 200:
            warnings.append(f"description 有 {len(desc)} 字符；规范上限 1024，但 claude.ai 上传路径"
                            "曾出现 200 字符口径，压到 200 以内更保险")

    # 4) compatibility 长度
    compat_m = re.search(r"^compatibility:\s*(.+)$", fm, re.M)
    if compat_m and len(compat_m.group(1)) > MAX_COMPAT:
        errors.append(f"compatibility 超过 {MAX_COMPAT} 字符：{len(compat_m.group(1))}")

    # 5) 正文篇幅（官方建议 <500 行）
    body_lines = len(body.splitlines())
    if body_lines > MAX_BODY_LINES:
        warnings.append(f"SKILL.md 正文 {body_lines} 行，超过官方建议的 {MAX_BODY_LINES} 行；"
                        "考虑把细节移到 references/")

    # 6) 反模式：技能内的 Windows 风格路径（文档一律用正斜杠，否则跨平台失效）
    docs = [skill] + sorted(root.glob("references/*.md")) + sorted(root.glob("assets/**/*.md")) \
        + sorted(root.glob("examples/*.md")) + sorted(root.glob("evals/*.md"))
    for path in docs:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if BACKSLASH_RE.search(line):
                errors.append(f"{path.relative_to(root)}:{i} 出现反斜杠路径（应统一正斜杠）：{line.strip()}")

    # 7) 文件引用是否真实存在
    referenced = set(PATH_REF_RE.findall(text))
    for rel in sorted(referenced):
        if not (root / rel).exists():
            errors.append(f"SKILL.md 引用了不存在的文件：{rel}")
    if referenced:
        print(f"检查了 {len(referenced)} 个文件引用")

    # 8) 目录清单提示（未在 SKILL.md 中出现的顶层文件）
    for sub in INDEXED_DIRS:
        d = root / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.name not in text:
                    warnings.append(f"{sub}/{f.name} 未在 SKILL.md 中被引用（可选：加入索引）")

    print(f"\n技能：{root.name}")
    print(f"字段：{keys}")
    print(f"正文行数：{body_lines}")
    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    failed = len(errors) > 0 or (args.strict and len(warnings) > 0)
    if failed and not errors:
        print("[ERROR] --strict 已开启：上面的警告按错误处理")
    print(f"\n结果：{len(errors)} 个错误，{len(warnings)} 个警告"
          + ("（--strict）" if args.strict else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
