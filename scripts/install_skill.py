#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 math-modeling-skill 装进宿主的技能目录——一条命令，默认不联网。

为什么需要这个脚本:
    "装技能"看着简单，实际有三个坑：

    1. **技能目录每家不一样**。DSH 认 `~/.dsh/skills`，跨宿主的 Agent Skills 约定
       是 `~/.agents/skills`，Claude Code 是 `~/.claude/skills`，而 DSH 还有项目级
       `<项目>/.dsh/skills`（优先级高于用户级）。更要命的是**安装后的目录名必须
       和 SKILL.md 里的 `name` 完全一致**，改名不会报错，只会静默不生效。
    2. **`git clone` 会带进 `.git/`**。几 MB 的历史躺在技能目录里，部分宿主扫描
       技能时看到 `.git` 还会困惑。
    3. **用户可能已经装过旧版本**。直接覆盖会抹掉人家改过的东西。

    本脚本一次处理掉这三件事：按内置的目标表定位技能根目录、按白名单复制
    （丢掉 `.git`/`__pycache__`/虚拟环境）、**默认拒绝覆盖**（要覆盖必须显式
    `--force`，而且只有当那个目录确实是本技能时才肯删）。

给 AI 助手用（这个脚本主要为"让 agent 直接跑"而设计）:
    不确定本宿主的技能根在哪时，**先跑 `--list-targets`**，它会打印每个候选根
    的绝对路径、是否已存在、以及那里是不是已经装了本技能：

        python scripts/install_skill.py --list-targets
        python scripts/install_skill.py --target auto

    `--target auto` 按"项目级优先于用户级、DSH 优先于通用约定"的顺序挑第一个
    **已存在**的技能根；一个都不存在时落到 `agents-user`（`~/.agents/skills`，
    跨宿主通用约定）。

用法:
    python scripts/install_skill.py --list-targets                  # 先看有哪些可装位置
    python scripts/install_skill.py --target auto                   # 装到自动挑出的位置
    python scripts/install_skill.py --target agents-user            # 装到 ~/.agents/skills
    python scripts/install_skill.py --target dsh-project --dry-run  # 只看要做什么，不动磁盘
    python scripts/install_skill.py --into D:/my/skills/math-modeling-skill   # 精确指定目标目录
    python scripts/install_skill.py --source D:/path/to/checkout    # 从本地另一个副本装
    python scripts/install_skill.py --from-zip math-modeling-skill-vX.Y.Z.zip  # 从 Release ZIP 装（换成实际版本号）
    python scripts/install_skill.py --download                      # 拉最新 Release 的 ZIP 再装（断线自动重试）
    python scripts/install_skill.py --self-test                     # 不联网、不动真实技能目录的固件测试

退出码:
    0 成功（含 `--dry-run` 与目标已是最新且未加 `--force` 时）；1 出错（找不到
    SKILL.md、目标已存在且未加 `--force`、目标已存在但不是本技能因而拒绝删除、
    安装后结构校验不通过等）。

安全约定（重要）:
    **默认不覆盖已存在的技能目录。** `--force` 也只在目标目录里确实有 `SKILL.md`
    且其 `name` 就是 `math-modeling-skill` 时才允许删除——这道闸门是为了防
    `--into`/`--target custom:` 手滑指到家目录，把别人的东西当旧版本删掉。
"""

from __future__ import annotations

import argparse
import fnmatch
import http.client
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

#: 技能名。**必须**与 SKILL.md 的 `name` 字段、以及安装后的目录名三者一致，
#: 任何一处不一致宿主都会找不到技能。
SKILL_NAME = "math-modeling-skill"

#: 上游仓库；`--download` 用它找最新 Release 的 ZIP（公开仓库，不需要 token）。
REPO = "anticipate218/math-modeling-skill"
RELEASE_API = "https://api.github.com/repos/" + REPO + "/releases/latest"

#: 联网重试次数（含首次）。GitHub 在部分家宽/代理/校园网下会偶发 TLS 中断
#: （`SSL: UNEXPECTED_EOF_WHILE_READING`）或连接重置，**这不是用户的操作错误**，
#: 让整条命令直接失败会逼用户手工重跑，体验很差。
RETRY_ATTEMPTS = 4
#: 退避基数（秒）：第 n 次重试前等 `RETRY_BASE_DELAY * 2**(n-1)`，即 1.5s / 3s / 6s。
RETRY_BASE_DELAY = 1.5

#: 复制时丢弃的目录：`.git` 是 clone 残留，其余是缓存/虚拟环境，与技能无关。
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".dsh-tmp", ".venv", "venv", ".idea", ".vscode",
})
IGNORE_GLOBS = ("*.pyc", "*.pyo", "*.egg-info")

#: 已知技能根目录表：`名字 -> (作用域, 相对路径, 说明)`。
#: 作用域 `project` = 相对"项目根"（最近的有 `.git` 的祖先目录，没有则当前目录）；
#: `home` = 相对用户主目录；`dsh` = 相对 DSH 主目录（`$DSH_HOME`，默认 `~/.dsh`）。
#:
#: 前四项目及优先级来自 DSH 的技能根表（`@deepseek-ai/dsh-skill-filesystem`）：
#: 项目级 rank 100/200，用户级 rank 400/500，数字小的先命中。
#: `claude-*` 两项来自 Claude Code 自己文档里的位置。**其余宿主一律不写死**——
#: 猜错的代价是把技能装到一个永远不会被扫描的目录，而且还"装成功了"，
#: 不如让 agent 去读宿主自己的文档、再用 `--into` 指定。
TARGETS = {
    "dsh-project": ("project", ".dsh/skills", "DSH 项目级技能目录（本项目内优先级最高）"),
    "agents-project": ("project", ".agents/skills", "Agent Skills 项目级技能目录（跨宿主通用约定）"),
    "dsh-user": ("dsh", "skills", "DSH 用户级技能目录（$DSH_HOME/skills，默认 ~/.dsh/skills）"),
    "agents-user": ("home", ".agents/skills", "Agent Skills 用户级技能目录（~/.agents/skills，跨宿主通用约定）"),
    "claude-project": ("project", ".claude/skills", "Claude Code 项目级技能目录"),
    "claude-user": ("home", ".claude/skills", "Claude Code 用户级技能目录"),
}

#: `--target auto` 的挑选顺序：项目级优先于用户级，DSH 优先于通用约定。
AUTO_ORDER = ("dsh-project", "agents-project", "dsh-user", "agents-user")

#: 一个都不存在时 `auto` 落到哪里。
AUTO_FALLBACK = "agents-user"

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)
_FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$")
_VERSION_RE = re.compile(r"^\s+version:\s*(.+?)\s*$", re.M)


class InstallError(Exception):
    """可预期的安装失败（打印成人话后以退出码 1 结束，不抛栈）。"""


# --------------------------------------------------------------------------
# 路径解析
# --------------------------------------------------------------------------

def find_project_root(start: pathlib.Path) -> pathlib.Path:
    """向上找最近的、含 `.git` 的目录，作为"项目根"。

    参数:
        start: 起点目录（通常是当前工作目录）。

    返回:
        含 `.git` 的最近祖先目录；一路到盘根都没找到时返回 `start` 自身
        （此时项目级技能根退化成"就在当前目录下建"）。

    算法:
        从 `start` 逐级向上，遇到 `.git`（文件或目录都算，worktree 里 `.git` 是文件）即返回。

    复杂度:
        时间 O(路径深度) / 空间 O(1)。

    陷阱:
        不要用"当前目录"当项目根——在 `src/` 子目录里跑脚本就会把技能装到
        `src/.dsh/skills`，宿主扫不到。
    """
    cur = pathlib.Path(start).resolve()
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return pathlib.Path(start).resolve()
        cur = cur.parent


def dsh_home() -> pathlib.Path:
    """返回 DSH 主目录：`$DSH_HOME`，未设置时 `~/.dsh`。"""
    env = os.environ.get("DSH_HOME")
    if env and env.strip():
        return pathlib.Path(env.strip()).expanduser()
    return pathlib.Path.home() / ".dsh"


def skills_root(name: str, project_root: pathlib.Path, home: pathlib.Path,
                dsh: pathlib.Path) -> pathlib.Path:
    """把目标名解析成"技能根目录"（技能目录的父目录）。

    参数:
        name: `TARGETS` 里的键，例如 ``dsh-user``。
        project_root: `find_project_root` 的结果。
        home: 用户主目录。
        dsh: DSH 主目录。

    返回:
        技能根目录的绝对路径（**不含** `math-modeling-skill` 这一层）。

    算法:
        按 `TARGETS` 里的作用域前缀把相对路径拼到对应基目录上。

    复杂度:
        时间 O(1) / 空间 O(1)。

    陷阱:
        返回的是"根"不是"技能目录"。技能目录 = 本函数结果 / `SKILL_NAME`。
    """
    if name not in TARGETS:
        raise InstallError(f"未知目标 {name!r}；可用目标见 --list-targets")
    scope, rel, _ = TARGETS[name]
    base = {"project": project_root, "home": home, "dsh": dsh}[scope]
    return pathlib.Path(base) / rel


def resolve_auto(project_root: pathlib.Path, home: pathlib.Path,
                 dsh: pathlib.Path) -> str:
    """按 `AUTO_ORDER` 挑第一个**已存在**的技能根；都不存在时返回 `AUTO_FALLBACK`。

    参数:
        project_root: 项目根。
        home: 用户主目录。
        dsh: DSH 主目录。

    返回:
        目标名（`TARGETS` 的键）。

    算法:
        依次探测 `AUTO_ORDER` 中每个目标的技能根目录是否存在，返回首个命中的；
        全不命中则返回 `AUTO_FALLBACK`。

    复杂度:
        时间 O(目标数) / 空间 O(1)。

    陷阱:
        "存在"指技能根目录本身已存在，不是指父目录。`~/.agents` 存在但
        `~/.agents/skills` 不存在时不算命中——那说明这个宿主还没用过这类技能。
    """
    for name in AUTO_ORDER:
        if skills_root(name, project_root, home, dsh).is_dir():
            return name
    return AUTO_FALLBACK


# --------------------------------------------------------------------------
# 技能包识别与小工具
# --------------------------------------------------------------------------

def read_frontmatter(skill_md: pathlib.Path) -> dict:
    """读 SKILL.md 的 YAML frontmatter，返回顶层字段 + 嵌套的 `version`。

    参数:
        skill_md: SKILL.md 路径。

    返回:
        字典。顶层标量字段原样入表（`description: >-` 这种只取首行），
        另外把 `metadata:` 里缩进的 `version` 以键 ``version`` 合并进来。

    算法:
        正则切出 `---` 之间的块，逐行匹配顶层 `键: 值`；再用一个缩进正则单独
        捞 `version`（它在本仓库里位于 `metadata:` 块内）。

    复杂度:
        时间 O(行数) / 空间 O(文件大小)。

    陷阱:
        这里**只做识别，不做规范校验**。字段白名单、篇幅、引用完整性由
        `scripts/validate_skill.py --strict` 负责，两件事不要混在一起。
        读进来先剥掉 UTF-8 BOM：Windows 上的编辑器（记事本、部分 IDE）会给
        文件加 BOM，带着 BOM 时 `---` 不在开头，frontmatter 会整块读不出来，
        于是 `is_our_skill` 误判为"不是本技能"，`--force` 更新就会莫名被拒。
    """
    text = skill_md.read_text(encoding="utf-8").lstrip("\ufeff")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    block = m.group(1)
    data: dict = {}
    for line in block.splitlines():
        mm = _FIELD_RE.match(line)
        if mm:
            data[mm.group(1)] = mm.group(2).strip().strip('"').strip("'")
    vm = _VERSION_RE.search(block)
    if vm:
        data["version"] = vm.group(1).strip().strip('"').strip("'")
    return data


def is_our_skill(directory: pathlib.Path) -> bool:
    """判断某个目录是不是本技能的一个安装（`SKILL.md` 里 `name` 匹配）。

    参数:
        directory: 待检查的目录。

    返回:
        目录里有 SKILL.md、能读出 frontmatter、且 `name` 等于 `SKILL_NAME` 时为 True。

    算法:
        文件存在性 + `read_frontmatter` 比对。

    复杂度:
        时间 O(SKILL.md 大小) / 空间 同。

    陷阱:
        这是 `--force` 删除前的**唯一放行条件**。不要放宽成"目录里有个 SKILL.md
        就算"——别人的技能也叫 SKILL.md。
    """
    skill_md = directory / "SKILL.md"
    if not skill_md.is_file():
        return False
    try:
        return read_frontmatter(skill_md).get("name") == SKILL_NAME
    except (OSError, UnicodeDecodeError):
        return False


def _ignore(_directory: str, names: list) -> list:
    """`shutil.copytree` 的 ignore 回调：丢掉版本库、缓存与虚拟环境。"""
    return [n for n in names
            if n in IGNORE_DIRS or any(fnmatch.fnmatch(n, g) for g in IGNORE_GLOBS)]


def copy_skill(src: pathlib.Path, dst: pathlib.Path, force: bool, dry_run: bool) -> str:
    """把技能从 `src` 复制到 `dst`，并按安全约定决定是否覆盖。

    参数:
        src: 源目录（必须含 SKILL.md）。
        dst: 目标目录（安装后的技能目录本身）。
        force: 目标已存在时是否允许覆盖。
        dry_run: 为 True 时只判断可行性并返回描述，不碰磁盘。

    返回:
        人类可读的动作描述（``安装`` / ``覆盖`` / ``跳过``）。

    算法:
        目标不存在 → 直接复制；目标存在且不是本技能 → 一律报错；目标存在且是本
        技能 → 未加 `--force` 时报错、加了则先删后复制。

    复杂度:
        时间 O(技能包文件数 × 文件大小) / 空间 同（复制一份）。

    陷阱:
        `shutil.copytree` 默认拒绝写进已存在的目录，所以覆盖必须显式 `rmtree`。
        删除前务必先过 `is_our_skill`，这是防手滑的那道闸门。
    """
    if dst.exists():
        if not is_our_skill(dst):
            raise InstallError(
                f"目标已存在且不是本技能，拒绝删除：{dst}\n"
                f"（该目录里没有 name: {SKILL_NAME} 的 SKILL.md；请换个位置或手工处理）")
        if not force:
            raise InstallError(
                f"目标已存在：{dst}\n"
                f"这是本技能的旧安装。确认要覆盖请加 --force；只想看看会做什么请加 --dry-run。")
        action = "覆盖"
    else:
        action = "安装"

    if dry_run:
        return action

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore, symlinks=False)
    return action


def resolve_into(raw: str) -> tuple:
    """把 `--into` 的取值规整成"技能目录本身"，并给出备注。

    参数:
        raw: 用户在 `--into` 后面给的原样字符串（可含 `~`）。

    返回:
        `(技能目录, 备注)`；备注为空串表示原样使用，否则是给人看的一句说明。

    算法:
        目录名已经等于 `SKILL_NAME` → 原样使用。否则看该路径下有没有 `SKILL.md`：
        没有（路径不存在，或那儿只是技能根）→ 视为技能根，追加一层 `SKILL_NAME`；
        有且 `name` 也是本技能 → 就是本技能的安装目录，原样使用；有但不是本技能
        → 报错，绝不把本技能塞进别人的技能目录。

    复杂度:
        时间 O(1)（最多读一个 SKILL.md）/ 空间 同。

    陷阱:
        技能标准要求"目录名 == frontmatter 的 name"。用户很自然会把**技能根**
        （例如 `~/.agents/skills`）喂给 `--into`；若照抄不误，技能文件会被平铺进
        技能根，宿主扫不到、还会污染目录。这里自动补一层，并把真实目标打印出来。
    """
    dst = pathlib.Path(raw).expanduser()
    try:
        dst = dst.resolve()
    except OSError:  # pragma: no cover - 盘符不存在等极端情况，退回未规整路径
        pass
    if dst.name == SKILL_NAME:
        return dst, ""
    skill_md = dst / "SKILL.md"
    if skill_md.is_file():
        try:
            other = read_frontmatter(skill_md).get("name")
        except (OSError, UnicodeDecodeError):
            other = None
        if other == SKILL_NAME:
            return dst, ""
        shown = other if other else "读不出（frontmatter 里没有 name）"
        raise InstallError(
            f"--into 指的 {dst} 已经是一个别的技能（SKILL.md 里 name: {shown}）。\n"
            f"不要把本技能塞进别人的技能目录；请改成 {dst / SKILL_NAME} 或换一个位置。")
    nested = dst / SKILL_NAME
    return nested, f"--into 给的是技能根，按标准补上技能目录名：{nested}"


# --------------------------------------------------------------------------
# 来源解析（本地目录 / ZIP / 联网下载）
# --------------------------------------------------------------------------

def extract_zip(zip_path: pathlib.Path, workdir: pathlib.Path) -> pathlib.Path:
    """把技能包 ZIP 解到临时目录，返回含 SKILL.md 的那一层。

    参数:
        zip_path: ZIP 路径（Release 资产或 `download_templates.py --zip` 产出的包都行）。
        workdir: 空的工作目录。

    返回:
        解出来的、直接含 `SKILL.md` 的目录。

    算法:
        在 namelist 里找路径最浅的 `*/SKILL.md`（或裸 `SKILL.md`），把它的目录前缀
        当作打包前缀，逐条解压并剥掉前缀；`..` 段落一律跳过（zip slip 防护）。

    复杂度:
        时间 O(压缩包大小) / 空间 同（解压一份）。

    陷阱:
        Release ZIP 顶层带 `math-modeling-skill/` 前缀，而手工 `zip -r` 的包可能
        没有前缀——两种都要能装，所以前缀是**算出来的**，不是写死的。
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n.replace("\\", "/") for n in zf.namelist()]
        candidates = [n for n in names if n == "SKILL.md" or n.endswith("/SKILL.md")]
        if not candidates:
            raise InstallError(f"{zip_path} 里找不到 SKILL.md，这不像是技能包")
        top = min(candidates, key=lambda n: n.count("/"))
        prefix = top[: -len("SKILL.md")]
        out = workdir / "extracted"
        out.mkdir(parents=True, exist_ok=True)
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            rel = name[len(prefix):]
            if not rel or ".." in pathlib.PurePosixPath(rel).parts:
                continue
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as fh_src, open(dest, "wb") as fh_dst:
                shutil.copyfileobj(fh_src, fh_dst)
        return out


def _retry_network(what: str, action, *, attempts: int = RETRY_ATTEMPTS,
                   sleeper=None, reporter=None):
    """按指数退避重试一个**联网动作**；全部失败时抛 `InstallError`。

    参数:
        what: 出错信息里的人话主语，例如 `"查询最新 Release"`、`"下载 xxx.zip"`。
        action: 无参可调用对象，每次调用都完整重做一遍联网动作并返回结果。
            **整个动作**（连接 + 读响应体 + 写文件）都在重试范围内，这样
            "连上了但传到一半断掉"也能重来，而不是留下半个文件。
        attempts: 总尝试次数（含首次），必须 >= 1。
        sleeper: 睡眠函数，默认 `time.sleep`；固件测试注入假实现以免真的等待。
        reporter: 提示输出函数，默认 `print`；每次重试前说明原因。

    返回:
        `action()` 的返回值（首次成功那次）。

    算法:
        循环 `attempts` 次：第 i 次（i>0）先按 `RETRY_BASE_DELAY * 2**(i-1)` 退避，
        再执行 `action()`；捕获连接层异常（`URLError` / `OSError` /
        `http.client.HTTPException`）继续下一次。全部失败则抛 `InstallError`，
        错误信息里给两条**可执行**的出路（直接重跑 / 手工下 ZIP + `--from-zip`）。

    复杂度:
        时间 O(attempts × 单次网络往返)，失败时额外等待 O(2**attempts) 秒 / 空间 O(1)。

    陷阱:
        只重试**偶发的连接层**失败，不做内容判断：HTTP 404 也会被重试，靠次数
        有限兜底。`attempts < 1` 会静默什么都不做，所以显式报 `ValueError`。
        `action` 必须自身幂等（例如写文件用 `"wb"` 覆盖），否则重试会叠加副作用。
    """
    if attempts < 1:
        raise ValueError("attempts 必须 >= 1")
    sleeper = sleeper or time.sleep
    reporter = reporter or print
    last: object = None
    for i in range(attempts):
        if i:
            delay = RETRY_BASE_DELAY * (2 ** (i - 1))
            reporter(f"  …{what}第 {i + 1}/{attempts} 次尝试（{delay:g}s 后重试，上次失败：{last}）")
            sleeper(delay)
        try:
            return action()
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            last = exc
    raise InstallError(
        f"{what}连续 {attempts} 次都没成功，最后一次的报错是：{last}\n"
        "  · 这一步只是网络问题，**直接重跑一次同样的命令**通常就好了；\n"
        f"  · 网络一直不稳就打开 https://github.com/{REPO}/releases ，"
        "手工下载 math-modeling-skill-vX.Y.Z.zip，再用 --from-zip 指过去。"
    )


def pick_release_asset(assets: list) -> dict:
    """从 Release 资产列表里挑出**技能包**那一个。

    参数:
        assets: `releases/latest` 接口返回的 `assets` 列表（每项是含 `name` 与
            `browser_download_url` 的字典）。

    返回:
        选中的资产字典（一定 `.zip` 且文件名以 `SKILL_NAME` 开头）。

    算法:
        先过滤出 `.zip`；再取文件名以 `math-modeling-skill` 开头的第一个。

    复杂度:
        时间 O(n)（n = 资产个数） / 空间 O(1)。

    陷阱:
        v1.9.0 起同一个 Release 里还挂着 `cumcm-template.zip` / `gmcm-template.zip` /
        `mcm-template.zip` 三个 LaTeX 模板包。早期实现是"取第一个 `.zip`"，而
        GitHub 接口**不承诺资产顺序**——模板包一旦排在前面，`--download` 就会把
        一个 LaTeX 工程当技能装下去，报错还很难懂（缺 `SKILL.md`）。所以必须按
        文件名认包，并且**宁可报错也不猜**：一个都不匹配时明确告诉用户现有资产
        叫什么，而不是随便挑一个。
    """
    zips = [a for a in assets if str(a.get("name", "")).endswith(".zip")]
    if not zips:
        raise InstallError("最新 Release 里没有 .zip 资产；请到仓库 Releases 页面手工下载后用 --from-zip")
    for asset in zips:
        if str(asset.get("name", "")).startswith(SKILL_NAME):
            return asset
    names = "、".join(str(a.get("name", "?")) for a in zips)
    raise InstallError(
        f"最新 Release 里没有以 {SKILL_NAME} 开头的 ZIP 资产（现有：{names}）。"
        f"请到 https://github.com/{REPO}/releases 手工下载技能包后用 --from-zip 指定"
    )


def download_latest_zip(workdir: pathlib.Path) -> pathlib.Path:
    """从 GitHub 拉最新 Release 里的技能包 ZIP 到临时目录。

    参数:
        workdir: 临时目录。

    返回:
        下载好的 ZIP 路径。

    算法:
        查 `releases/latest` 接口，用 `pick_release_asset` 挑出技能包，按
        `browser_download_url` 下载。**两步都带指数退避重试**（见 `_retry_network`）。

    复杂度:
        时间 O(包大小) 受网速限制 / 空间 同。

    陷阱:
        这是本脚本**唯一联网**的动作，且未加 `--download` 时完全不会走这里。
        公开仓库无需 token；但无 token 时 GitHub 接口有每小时 60 次的限额，
        连续被限流就改用 `--from-zip` 下载好的包。
        GitHub 偶发 TLS 中断（`UNEXPECTED_EOF_WHILE_READING`）在实测中很常见，
        所以**读响应体、写盘**也在重试范围内：中断后重来一遍，而不是留个半截
        ZIP 让后面的解包报"不是合法 ZIP"。
        同页还挂着三个 LaTeX 模板包，**只能按文件名认技能包**（见 `pick_release_asset`）。
    """
    import json  # 只在联网分支里用，保持顶层依赖最小

    def fetch_meta():
        req = urllib.request.Request(RELEASE_API, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "math-modeling-skill-installer",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        meta = _retry_network("查询最新 Release", fetch_meta)
    except InstallError:
        raise
    except (OSError, ValueError) as exc:  # 响应不是合法 JSON 等
        raise InstallError(f"查询最新 Release 失败（{exc}）；可以改用 --from-zip 指定本地 ZIP") from exc

    asset = pick_release_asset(list(meta.get("assets", [])))
    target = workdir / str(asset["name"])

    def fetch_zip():
        req = urllib.request.Request(str(asset["browser_download_url"]),
                                     headers={"User-Agent": "math-modeling-skill-installer"})
        # "wb" 覆盖写：重试时不会把两次的部分内容拼在一起。
        with urllib.request.urlopen(req, timeout=120) as resp, open(target, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return target.stat().st_size

    try:
        size = _retry_network(f"下载 {asset['name']}", fetch_zip)
    except InstallError:
        raise
    except (OSError, ValueError) as exc:
        raise InstallError(f"下载 {asset['name']} 失败：{exc}") from exc
    if not size:
        raise InstallError(f"下载 {asset['name']} 拿到的是 0 字节；请重试，或改用 --from-zip")
    print(f"已下载 {asset['name']}（{size} 字节）")
    return target


def prepare_source(args: argparse.Namespace, workdir: pathlib.Path) -> pathlib.Path:
    """按命令行参数确定"从哪儿装"，返回含 SKILL.md 的源目录。

    参数:
        args: 解析后的命令行参数（看 `source` / `from_zip` / `download`）。
        workdir: 临时目录（解压/下载用）。

    返回:
        源技能目录的绝对路径。

    算法:
        `--source` → `--from-zip` → `--download` → 默认（本脚本所在仓库）依次判定，
        优先级与写出顺序一致。

    复杂度:
        时间 O(1)（不含下载/解压开销） / 空间 O(1)。

    陷阱:
        默认值取 `脚本目录/..` 而不是当前目录——用户大概率是 `cd` 到仓库里跑的，
        但当前目录也可能是别处。默认路径失效时必须**给出怎么修的提示**，
        而不是甩一句"找不到 SKILL.md"。
    """
    if args.source:
        src = pathlib.Path(args.source).expanduser().resolve()
    elif args.from_zip:
        src = extract_zip(pathlib.Path(args.from_zip).expanduser().resolve(), workdir)
    elif args.download:
        src = extract_zip(download_latest_zip(workdir), workdir)
    else:
        src = pathlib.Path(__file__).resolve().parent.parent

    if not (src / "SKILL.md").is_file():
        raise InstallError(
            f"源目录里没有 SKILL.md：{src}\n"
            f"请在技能仓库根目录运行，或用 --source <仓库目录> / --from-zip <发布包.zip> 指定来源。")
    return src


# --------------------------------------------------------------------------
# 安装后校验
# --------------------------------------------------------------------------

def validate_installed(dst: pathlib.Path) -> tuple:
    """用安装副本自带的 `validate_skill.py --strict` 校验结构。

    参数:
        dst: 安装后的技能目录。

    返回:
        `(状态, 详情)`：状态取 ``"ok"``/``"fail"``/``"skip"``；详情是给人看的一句话。

    算法:
        找到 `<dst>/scripts/validate_skill.py`，用 `sys.executable` 以 `--strict` 跑一遍。

    复杂度:
        时间 O(文件数) / 空间 O(输出)。

    陷阱:
        必须用 `sys.executable` 而不是裸 `python`：Windows 上 `python` 可能是
        另一个解释器或干脆不存在。校验脚本缺失时返回 `skip` 而不是失败——
        从精简过的包里装出来时它可能真的不在。
    """
    validator = dst / "scripts" / "validate_skill.py"
    if not validator.is_file():
        return "skip", "包内没有 scripts/validate_skill.py，跳过结构校验"
    proc = subprocess.run([sys.executable, str(validator), str(dst), "--strict"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=str(dst))
    if proc.returncode == 0:
        return "ok", "validate_skill.py --strict 通过"
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    return "fail", "；".join(tail[-6:]) if tail else "validate_skill.py --strict 未通过"


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------

def cmd_list_targets(project_root: pathlib.Path, home: pathlib.Path,
                     dsh: pathlib.Path) -> int:
    """打印技能根目录表，并标注每个位置是否已存在、是否已装本技能。"""
    print(f"项目根：{project_root}")
    print(f"用户主目录：{home}")
    print(f"DSH 主目录：{dsh}")
    print()
    header = f"{'目标名':<16}{'已存在':<8}{'已装本技能':<12}路径"
    print(header)
    print("-" * len(header))
    for name in list(TARGETS) + ["auto"]:
        if name == "auto":
            continue
        root = skills_root(name, project_root, home, dsh)
        dst = root / SKILL_NAME
        installed = "-"
        if is_our_skill(dst):
            version = read_frontmatter(dst / "SKILL.md").get("version", "?")
            installed = f"v{version}"
        print(f"{name:<16}{'是' if root.is_dir() else '否':<8}{installed:<12}{dst}")
    picked = resolve_auto(project_root, home, dsh)
    print()
    print(f"--target auto 当前会选：{picked}")
    print(f"  理由：{'、'.join(AUTO_ORDER)} 里第一个已存在的技能根；都不存在则用 {AUTO_FALLBACK}")
    print()
    print("以上是已知的通用位置。若你的宿主不在这张表里，请读宿主自己的文档确认技能根，")
    print("然后用：--into <技能根>            （会自动补一层 math-modeling-skill）")
    print("或：    --into <技能根>/math-modeling-skill")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    """按参数执行安装，打印每一步结果。"""
    project_root = find_project_root(pathlib.Path.cwd())
    home = pathlib.Path.home()
    dsh = dsh_home()

    if args.into:
        dst, into_note = resolve_into(args.into)
        target_label = "into"
    else:
        into_note = ""
        name = args.target
        if name == "auto":
            name = resolve_auto(project_root, home, dsh)
        if name.startswith("custom:"):
            root = pathlib.Path(name[len("custom:"):]).expanduser().resolve()
            if not str(root):
                raise InstallError("custom: 后面要跟技能根目录，例如 --target custom:~/.agents/skills")
            target_label = name
        elif name in TARGETS:
            root = skills_root(name, project_root, home, dsh)
            target_label = name
        else:
            raise InstallError(f"未知目标 {name!r}；可用目标见 --list-targets")
        dst = root / SKILL_NAME

    with tempfile.TemporaryDirectory(prefix="mms-install-") as tmp:
        src = prepare_source(args, pathlib.Path(tmp))
        fm = read_frontmatter(src / "SKILL.md")
        src_name = fm.get("name")
        if src_name != SKILL_NAME:
            raise InstallError(
                f"源包的 SKILL.md 里 name 是 {src_name!r}，期望 {SKILL_NAME!r}；"
                f"装下去宿主按 name 找技能，会找不到。")
        version = fm.get("version", "?")

        print(f"来源：{src}（v{version}）")
        print(f"目标：{dst}   [{target_label}]")
        if into_note:
            print(f"      注意：{into_note}")
        if args.dry_run:
            action = copy_skill(src, dst, args.force, dry_run=True)
            print(f"[dry-run] 会执行：{action}；未改动磁盘。")
            return 0

        action = copy_skill(src, dst, args.force, dry_run=False)
    print(f"{action}完成。")

    status, detail = validate_installed(dst)
    print(f"校验：{detail}")
    if status == "fail":
        raise InstallError(
            f"安装后校验未通过，技能可能不完整：{dst}\n"
            f"（最常见的原因是技能目录名不是 {SKILL_NAME}——宿主按 name 找技能，改名会静默失效）")

    print()
    print("下一步：")
    print(f"  1. 确认技能已被宿主识别——目录名必须保持 {SKILL_NAME}，改名会静默失效。")
    print("  2. 让助手读一下这个技能的 SKILL.md（或直接说一句建模相关的需求看它会不会用）。")
    print(f"  3. 要更新：拿到新版包后重跑本脚本并加 --force（会先校验再覆盖自己的旧安装）。")
    print(f"  4. 要卸载：直接删掉 {dst} 即可，技能不进注册表也不留后台进程。")
    return 0


# --------------------------------------------------------------------------
# 固件测试
# --------------------------------------------------------------------------

def _make_fake_skill(root: pathlib.Path, name: str = SKILL_NAME) -> pathlib.Path:
    """造一个最小的假技能包，用于固件测试（含应被忽略的 `.git`/`__pycache__`）。"""
    src = root / "src"
    (src / "references").mkdir(parents=True)
    (src / "examples").mkdir(parents=True)
    (src / ".git").mkdir(parents=True)
    (src / "__pycache__").mkdir(parents=True)
    (src / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: 固件测试用的假技能。\n"
        "license: MIT\n"
        "metadata:\n"
        "  version: \"9.9.9\"\n"
        "---\n\n# 假技能\n",
        encoding="utf-8")
    (src / "references" / "a.md").write_text("# a\n", encoding="utf-8")
    (src / "examples" / "b.py").write_text("print('b')\n", encoding="utf-8")
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (src / "__pycache__" / "b.cpython-311.pyc").write_text("junk", encoding="utf-8")
    return src


def self_test() -> int:
    """不联网、不碰真实技能目录的固件测试；返回 0 全过 / 1 有失败。"""
    results: list = []

    def check(label: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            results.append((label, False, str(exc)))
        except Exception as exc:  # noqa: BLE001 - 固件测试就是要抓住任何异常
            results.append((label, False, f"{type(exc).__name__}: {exc}"))
        else:
            results.append((label, True, ""))

    with tempfile.TemporaryDirectory(prefix="mms-selftest-") as tmp:
        tmpdir = pathlib.Path(tmp)

        def t_copy_and_ignore() -> None:
            src = _make_fake_skill(tmpdir / "t1")
            dst = tmpdir / "t1" / "out" / SKILL_NAME
            copy_skill(src, dst, force=False, dry_run=False)
            assert (dst / "SKILL.md").is_file(), "SKILL.md 没被复制"
            assert (dst / "references" / "a.md").is_file(), "references/ 没被复制"
            assert not (dst / ".git").exists(), ".git 应被忽略"
            assert not (dst / "__pycache__").exists(), "__pycache__ 应被忽略"

        def t_refuse_then_force() -> None:
            src = _make_fake_skill(tmpdir / "t2")
            dst = tmpdir / "t2" / "out" / SKILL_NAME
            copy_skill(src, dst, force=False, dry_run=False)
            marker = dst / "references" / "user-edit.md"
            marker.write_text("我改过的东西\n", encoding="utf-8")
            try:
                copy_skill(src, dst, force=False, dry_run=False)
            except InstallError:
                pass
            else:
                raise AssertionError("目标已存在时未加 --force 应当报错")
            assert marker.is_file(), "--force 之外不应改动已存在的安装"
            copy_skill(src, dst, force=True, dry_run=False)
            assert not marker.exists(), "--force 覆盖后应只剩源包内容"

        def t_refuse_foreign_dir() -> None:
            src = _make_fake_skill(tmpdir / "t3")
            dst = tmpdir / "t3" / "out" / "someone-elses-skill"
            dst.mkdir(parents=True)
            (dst / "SKILL.md").write_text("---\nname: other-skill\n---\n", encoding="utf-8")
            try:
                copy_skill(src, dst, force=True, dry_run=True)
            except InstallError:
                pass
            else:
                raise AssertionError("不是本技能的目录即使 --force 也必须拒绝")
            assert (dst / "SKILL.md").is_file(), "拒绝时不得动人家的目录"

        def t_dry_run_touches_nothing() -> None:
            src = _make_fake_skill(tmpdir / "t4")
            dst = tmpdir / "t4" / "out" / SKILL_NAME
            assert copy_skill(src, dst, force=False, dry_run=True) == "安装"
            assert not dst.exists(), "--dry-run 不应创建目录"

        def t_reject_wrong_name() -> None:
            src = _make_fake_skill(tmpdir / "t5", name="not-the-right-name")
            assert read_frontmatter(src / "SKILL.md")["name"] == "not-the-right-name"
            assert not is_our_skill(src), "name 不匹配时 is_our_skill 必须为假"

        def t_frontmatter_version() -> None:
            src = _make_fake_skill(tmpdir / "t6")
            fm = read_frontmatter(src / "SKILL.md")
            assert fm.get("name") == SKILL_NAME, f"name 读错：{fm.get('name')!r}"
            assert fm.get("version") == "9.9.9", f"嵌套 version 没读到：{fm.get('version')!r}"

        def t_zip_roundtrip() -> None:
            src = _make_fake_skill(tmpdir / "t7")
            zip_path = tmpdir / "t7" / "pkg.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for p in sorted(src.rglob("*")):
                    if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
                        zf.write(p, str(pathlib.Path(SKILL_NAME) / p.relative_to(src)))
            out = tmpdir / "t7" / "unzip"
            out.mkdir()
            got = extract_zip(zip_path, out)
            assert (got / "SKILL.md").is_file(), "带前缀的 ZIP 没解出 SKILL.md"
            assert (got / "references" / "a.md").is_file(), "带前缀的 ZIP 没解出子目录"

        def t_zip_no_prefix() -> None:
            src = _make_fake_skill(tmpdir / "t8")
            zip_path = tmpdir / "t8" / "flat.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(src / "SKILL.md", "SKILL.md")
                zf.write(src / "references" / "a.md", "references/a.md")
            out = tmpdir / "t8" / "unzip"
            out.mkdir()
            got = extract_zip(zip_path, out)
            assert (got / "SKILL.md").is_file(), "无前缀的 ZIP 没解出 SKILL.md"

        def t_zip_rejects_non_skill() -> None:
            zip_path = tmpdir / "t9.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("random/notes.txt", "hello")
            out = tmpdir / "t9"
            out.mkdir()
            try:
                extract_zip(zip_path, out)
            except InstallError:
                pass
            else:
                raise AssertionError("不含 SKILL.md 的 ZIP 应当被拒绝")

        def t_auto_prefers_existing() -> None:
            fake_home = tmpdir / "t10" / "home"
            fake_proj = tmpdir / "t10" / "proj"
            fake_dsh = tmpdir / "t10" / "dsh"
            for d in (fake_home, fake_proj, fake_dsh):
                d.mkdir(parents=True)
            assert resolve_auto(fake_proj, fake_home, fake_dsh) == AUTO_FALLBACK, \
                "一个根都不存在时应落到 fallback"
            (fake_dsh / "skills").mkdir(parents=True)
            assert resolve_auto(fake_proj, fake_home, fake_dsh) == "dsh-user", "用户级 DSH 应先于通用约定"
            (fake_proj / ".agents" / "skills").mkdir(parents=True)
            assert resolve_auto(fake_proj, fake_home, fake_dsh) == "agents-project", "项目级应先于用户级"
            (fake_proj / ".dsh" / "skills").mkdir(parents=True)
            assert resolve_auto(fake_proj, fake_home, fake_dsh) == "dsh-project", "项目级 DSH 优先级最高"

        def t_target_paths() -> None:
            home = pathlib.Path("/home/u")
            proj = pathlib.Path("/work/repo")
            dsh = pathlib.Path("/home/u/.dsh")
            assert skills_root("agents-user", proj, home, dsh) == home / ".agents/skills"
            assert skills_root("dsh-user", proj, home, dsh) == dsh / "skills"
            assert skills_root("dsh-project", proj, home, dsh) == proj / ".dsh/skills"
            assert skills_root("claude-user", proj, home, dsh) == home / ".claude/skills"

        def t_project_root_walk() -> None:
            repo = tmpdir / "t11" / "repo"
            deep = repo / "a" / "b"
            deep.mkdir(parents=True)
            assert find_project_root(deep) == deep.resolve(), "没有 .git 时应返回起点自身"
            (repo / ".git").mkdir()
            assert find_project_root(deep) == repo.resolve(), "应向上找到含 .git 的目录"

        def t_into_nests_skills_root() -> None:
            root = tmpdir / "t12" / "skills"
            root.mkdir(parents=True)
            dst, note = resolve_into(str(root))
            assert dst == (root / SKILL_NAME).resolve(), f"--into 没补技能目录名：{dst}"
            assert note, "补了一层目录却没给出备注"

        def t_into_keeps_explicit_dir() -> None:
            exact = tmpdir / "t13" / "skills" / SKILL_NAME
            dst, note = resolve_into(str(exact))
            assert dst == exact.resolve(), f"技能目录名已正确却被改动：{dst}"
            assert note == "", "原样使用时不应有备注"

        def t_into_nests_empty_root_only() -> None:
            # 技能根里已经躺着别人的技能时，仍然把本技能装成它的兄弟目录。
            root = tmpdir / "t14" / "skills"
            other = root / "someone-elses-skill"
            other.mkdir(parents=True)
            (other / "SKILL.md").write_text("---\nname: other-skill\n---\n", encoding="utf-8")
            dst, _ = resolve_into(str(root))
            assert dst == (root / SKILL_NAME).resolve(), f"技能根解析错：{dst}"

        def t_into_rejects_foreign_skill_dir() -> None:
            foreign = tmpdir / "t15" / "someone-elses-skill"
            foreign.mkdir(parents=True)
            (foreign / "SKILL.md").write_text("---\nname: other-skill\n---\n", encoding="utf-8")
            try:
                resolve_into(str(foreign))
            except InstallError:
                pass
            else:
                raise AssertionError("不该把本技能塞进别人的技能目录")

        def t_frontmatter_bom() -> None:
            src = _make_fake_skill(tmpdir / "t16")
            p = src / "SKILL.md"
            p.write_text("\ufeff" + p.read_text(encoding="utf-8"), encoding="utf-8")
            assert read_frontmatter(p).get("name") == SKILL_NAME, "带 BOM 的 SKILL.md 读不出 name"
            assert is_our_skill(src), "带 BOM 时 is_our_skill 必须仍为真"

        def t_retry_succeeds_after_transient_failure() -> None:
            calls: list = []
            slept: list = []

            def flaky():
                calls.append(1)
                if len(calls) < 3:
                    raise urllib.error.URLError("SSL: UNEXPECTED_EOF_WHILE_READING")
                return "ok"

            got = _retry_network("测试动作", flaky, attempts=4,
                                 sleeper=slept.append, reporter=lambda *_: None)
            assert got == "ok", f"重试后应返回成功结果，实际 {got!r}"
            assert len(calls) == 3, f"应当在第 3 次尝试成功，实际调用 {len(calls)} 次"
            assert slept == [RETRY_BASE_DELAY, RETRY_BASE_DELAY * 2], f"退避时长不对：{slept}"

        def t_retry_gives_up_with_actionable_error() -> None:
            calls: list = []

            def always_broken():
                calls.append(1)
                raise urllib.error.URLError("连接被重置")

            try:
                _retry_network("测试动作", always_broken, attempts=3,
                               sleeper=lambda _s: None, reporter=lambda *_: None)
            except InstallError as exc:
                msg = str(exc)
                assert "3 次" in msg, f"错误信息没说清重试了几次：{msg}"
                assert "--from-zip" in msg, f"错误信息没给出兜底出路：{msg}"
            else:
                raise AssertionError("一直失败时必须抛 InstallError")
            assert len(calls) == 3, f"应当尝试 3 次，实际 {len(calls)} 次"

        def t_retry_attempts_must_be_positive() -> None:
            # attempts < 1 会让循环一次都不跑、静默返回 None，必须显式拦下。
            try:
                _retry_network("测试动作", lambda: "never", attempts=0,
                               sleeper=lambda _s: None, reporter=lambda *_: None)
            except ValueError:
                pass
            else:
                raise AssertionError("attempts=0 应当报 ValueError")

        def t_asset_picker_ignores_templates() -> None:
            # 模板包排在前面时，绝不能把 LaTeX 工程当技能包下回来。
            assets = [
                {"name": "gmcm-template.zip", "browser_download_url": "u1"},
                {"name": "cumcm-template.zip", "browser_download_url": "u2"},
                {"name": "mcm-template.zip", "browser_download_url": "u3"},
                {"name": f"{SKILL_NAME}-v9.9.9.zip", "browser_download_url": "u4"},
            ]
            assert pick_release_asset(assets)["name"] == f"{SKILL_NAME}-v9.9.9.zip", \
                "必须按文件名认技能包，而不是取第一个 .zip"

        def t_asset_picker_no_version_suffix() -> None:
            assets = [{"name": "notes.txt", "browser_download_url": "u0"},
                      {"name": f"{SKILL_NAME}.zip", "browser_download_url": "u1"}]
            assert pick_release_asset(assets)["name"] == f"{SKILL_NAME}.zip", \
                "不带版本号的技能包名也应被认出来"

        def t_asset_picker_refuses_to_guess() -> None:
            # 只有模板包时宁可报错，也不要随便挑一个装下去。
            assets = [{"name": "gmcm-template.zip", "browser_download_url": "u1"}]
            try:
                pick_release_asset(assets)
            except InstallError as exc:
                assert "gmcm-template.zip" in str(exc), f"报错应列出实际资产名：{exc}"
                assert "--from-zip" in str(exc), f"报错应给出兜底出路：{exc}"
            else:
                raise AssertionError("没有技能包时必须报错，而不是挑一个别的包")

        def t_asset_picker_rejects_empty() -> None:
            try:
                pick_release_asset([])
            except InstallError:
                pass
            else:
                raise AssertionError("没有 .zip 资产时应当报错")

        def t_list_targets_runs() -> None:
            rc = cmd_list_targets(pathlib.Path.cwd(), pathlib.Path.home(), dsh_home())
            assert rc == 0, f"cmd_list_targets 返回 {rc}"

        for label, fn in [
            ("复制时忽略 .git / __pycache__", t_copy_and_ignore),
            ("已存在时先拒绝、--force 才覆盖", t_refuse_then_force),
            ("不是本技能的目录一律不删", t_refuse_foreign_dir),
            ("--dry-run 不写盘", t_dry_run_touches_nothing),
            ("name 不匹配可被识别", t_reject_wrong_name),
            ("frontmatter 嵌套 version 可读", t_frontmatter_version),
            ("带前缀 ZIP 解包", t_zip_roundtrip),
            ("无前缀 ZIP 解包", t_zip_no_prefix),
            ("非技能 ZIP 被拒绝", t_zip_rejects_non_skill),
            ("auto 挑选顺序", t_auto_prefers_existing),
            ("目标路径展开", t_target_paths),
            ("项目根向上查找", t_project_root_walk),
            ("--into 给技能根时自动补一层", t_into_nests_skills_root),
            ("--into 给技能目录时原样使用", t_into_keeps_explicit_dir),
            ("技能根里有别人的技能也能装", t_into_nests_empty_root_only),
            ("--into 指向别人的技能目录时拒绝", t_into_rejects_foreign_skill_dir),
            ("带 UTF-8 BOM 的 SKILL.md 仍可识别", t_frontmatter_bom),
            ("联网动作会重试并最终成功", t_retry_succeeds_after_transient_failure),
            ("联网一直失败时报可执行的错", t_retry_gives_up_with_actionable_error),
            ("重试次数必须为正", t_retry_attempts_must_be_positive),
            ("Release 资产：模板包在前也认得技能包", t_asset_picker_ignores_templates),
            ("Release 资产：无版本号后缀的技能包也认", t_asset_picker_no_version_suffix),
            ("Release 资产：只有模板包时拒绝乱猜", t_asset_picker_refuses_to_guess),
            ("Release 资产：没有 .zip 时报错", t_asset_picker_rejects_empty),
            ("--list-targets 可运行", t_list_targets_runs),
        ]:
            check(label, fn)

    passed = sum(1 for _, ok, _ in results if ok)
    for label, ok, detail in results:
        if ok:
            print(f"PASS  {label}")
        else:
            print(f"FAIL  {label}  —— {detail}")
    print(f"\n固件测试：{passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="install_skill.py",
        description="把 math-modeling-skill 装进宿主的技能目录（默认不覆盖、默认不联网）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/install_skill.py --list-targets\n"
               "  python scripts/install_skill.py --target auto\n"
               "  python scripts/install_skill.py --into ~/.agents/skills\n"
               "  python scripts/install_skill.py --into ~/.agents/skills/math-modeling-skill\n")
    parser.add_argument("--list-targets", action="store_true",
                        help="列出已知技能根目录、是否已存在、那里装的是哪个版本，然后退出")
    parser.add_argument("--target", default="auto",
                        help="安装目标：auto（默认）或 TARGETS 里的键，"
                             "或 custom:<技能根目录>；用 --list-targets 查看可选值")
    parser.add_argument("--into", default="",
                        help="直接指定安装位置（最高优先级）。给技能根会自动补一层 "
                             "math-modeling-skill；给技能目录本身则原样使用")
    parser.add_argument("--source", default="",
                        help="本地技能仓库目录（默认：本脚本所在仓库的根目录）")
    parser.add_argument("--from-zip", default="",
                        help="从本地 ZIP 安装（Release 资产 / download_templates.py --zip 的产物）")
    parser.add_argument("--download", action="store_true",
                        help="从 GitHub 最新 Release 下载 ZIP 再安装（本脚本唯一联网的动作）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印会做什么，不写磁盘")
    parser.add_argument("--force", action="store_true",
                        help="目标已存在时覆盖（仅当目标确实是本技能时才允许删除）")
    parser.add_argument("--self-test", action="store_true",
                        help="跑固件测试：不联网、不碰真实技能目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口；返回进程退出码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    args = build_parser().parse_args(argv)

    if args.self_test:
        return self_test()

    project_root = find_project_root(pathlib.Path.cwd())
    home = pathlib.Path.home()
    dsh = dsh_home()

    if args.list_targets:
        return cmd_list_targets(project_root, home, dsh)

    try:
        return cmd_install(args)
    except InstallError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
