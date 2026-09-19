# 安装 math-modeling-skill（给 AI 助手看的说明）

> **你是一个 AI 编程助手。** 用户把这份文档的链接给了你，意思是：**请你替他把
> `math-modeling-skill` 这个技能装到他当前使用的宿主里**，装完再告诉他怎么用。
> 请按下面的步骤执行，不要只把本文档复述给用户。

技能仓库：<https://github.com/anticipate218/math-modeling-skill>
技能名（必须保持一致）：`math-modeling-skill`

---

## 你要达成的目标

把整个技能包（一个目录，含 `SKILL.md`、`references/`、`assets/`、`scripts/`、`examples/`、`evals/`）
放到**你的宿主会扫描的技能根目录**下，且**目录名正好是 `math-modeling-skill`**。

做完之后，用户下次提问"帮我看看这道建模题""写个摘要""做个灵敏度分析"时，
你的技能目录里就能看到它。

---

## 第 1 步：拿到技能包

按你手上有什么工具，选**一条**路走。优先级从上到下。

### 1A. 有 shell、有网（最常见）

```bash
git clone --depth 1 https://github.com/anticipate218/math-modeling-skill.git
```

不想用 git 就用 ZIP：

```bash
curl -L -o mms.zip https://github.com/anticipate218/math-modeling-skill/archive/refs/heads/main.zip
```

（Windows PowerShell 用 `Invoke-WebRequest -Uri <url> -OutFile mms.zip`；
`curl` 在 Windows 上是 `curl.exe`。）

### 1B. 用户已经下载了 Release ZIP

让他把 ZIP 路径给你，或者自己找 `~/Downloads` 下的 `math-modeling-skill-v*.zip`。
Release 资产地址：<https://github.com/anticipate218/math-modeling-skill/releases/latest>

### 1C. 你只能读网页、不能落盘

那就**别硬装**。把下面这段原样告诉用户，让他自己下载解压：

> 请打开 <https://github.com/anticipate218/math-modeling-skill/releases/latest>，
> 下载最新的 `math-modeling-skill-v*.zip`，解压后把里面的 `math-modeling-skill`
> 整个文件夹放到你的技能目录（见第 2 步）。解压出来的文件夹名不要改。

---

## 第 2 步：确定你宿主的技能根目录

**这一步最关键，也最容易错。** 每个宿主的技能目录都不一样。按顺序试：

### 2A. 让安装脚本自己探测（推荐，前提是能跑 Python 3.9+）

技能包里自带了一个只依赖标准库的安装器，它内置了已知的技能根目录表：

```bash
cd math-modeling-skill
python scripts/install_skill.py --list-targets
```

它会打印每个候选根目录的**绝对路径**、**是否已经存在**、以及**那里是否已经装过本技能**。
看完再决定装哪，然后：

```bash
python scripts/install_skill.py --target auto        # 自动挑第一个已存在的技能根
python scripts/install_skill.py --target agents-user # 或显式指定
```

`--target auto` 的顺序是：项目级优先于用户级，DSH 优先于通用约定；
一个候选都不存在时落到 `~/.agents/skills`（跨宿主通用的 Agent Skills 约定）。

### 2B. 自己找（脚本没覆盖你的宿主时）

**不要猜路径。** 用下面任一办法确认真实位置，然后走 2C：

1. 查你自己的配置：技能/插件目录通常写在 `settings.json`、`config.toml`、
   `.mcp.json`、`AGENTS.md` 之类的文件里。
2. 看你宿主已有的技能都放在哪：找找磁盘上有没有别的 `*/skills/*/SKILL.md`，
   它们所在的 `skills` 目录就是技能根。
3. 直接问用户："你之前装的技能放在哪个目录？"

已知且可核实的几个位置（**仅这几个，其余不要照抄**）：

| 宿主 / 约定 | 技能目录 |
|---|---|
| DSH 项目级 | `<项目根>/.dsh/skills/math-modeling-skill`（本项目内优先级最高） |
| DSH 用户级 | `~/.dsh/skills/math-modeling-skill`（`$DSH_HOME` 设置时以它为准） |
| Agent Skills 项目级（跨宿主通用约定） | `<项目根>/.agents/skills/math-modeling-skill` |
| Agent Skills 用户级（跨宿主通用约定） | `~/.agents/skills/math-modeling-skill` |
| Claude Code 项目级 | `<项目根>/.claude/skills/math-modeling-skill` |
| Claude Code 用户级 | `~/.claude/skills/math-modeling-skill` |

`<项目根>` = 从当前目录向上找到的最近一个含 `.git` 的目录；找不到就用当前目录。

### 2C. 用 `--into` 精确指定

不管你的宿主是什么，只要确定了技能根，就可以直接指定安装后的目录：

```bash
python scripts/install_skill.py --into "<技能根>/math-modeling-skill"
```

---

## 第 3 步：安装

```bash
# 在技能仓库根目录执行；默认不覆盖、不联网
python scripts/install_skill.py --target auto

# 从 Release ZIP 装（用户给的就是 zip 时）
python scripts/install_skill.py --from-zip /path/to/math-modeling-skill-v1.8.0.zip --target auto

# 想拉最新 Release 的 ZIP 再装（唯一联网的动作）
python scripts/install_skill.py --download --target auto

# 先看看会做什么，不动磁盘
python scripts/install_skill.py --target auto --dry-run
```

**没有 Python 时手工复制**（把源目录改成你 clone/解压出来的位置）：

```bash
mkdir -p ~/.agents/skills
cp -R ./math-modeling-skill ~/.agents/skills/math-modeling-skill
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME\.agents\skills" | Out-Null
Copy-Item -Recurse ".\math-modeling-skill" "$HOME\.agents\skills\math-modeling-skill"
```

手工复制时**记得把 `.git/` 删掉**——那是版本库，装进技能目录纯属累赘：

```bash
rm -rf ~/.agents/skills/math-modeling-skill/.git
```

---

## 第 4 步：校验

装完必须确认三件事，别跳过：

```bash
# 1) 结构合规（frontmatter / 篇幅 / 文件引用是否都存在）
python "<技能根>/math-modeling-skill/scripts/validate_skill.py" \
       "<技能根>/math-modeling-skill" --strict

# 2) 目录名与 SKILL.md 的 name 一致
#    SKILL.md 第 2 行应当是 name: math-modeling-skill
head -n 12 "<技能根>/math-modeling-skill/SKILL.md"

# 3) 关键内容都在
ls "<技能根>/math-modeling-skill"
#    期望看到：SKILL.md  references  assets  scripts  examples  evals  README.md
```

第 1 条期望输出 `结果：0 个错误，0 个警告（--strict）`。
如果 `scripts/validate_skill.py` 不存在，说明包不完整，重新下载。

---

## 第 5 步：告诉用户

汇报时包含这几项，**不要只说"装好了"**：

1. 装到了哪个绝对路径；
2. 装的是哪个版本（`SKILL.md` 的 `metadata.version`）；
3. 结构校验的结果；
4. 怎么开始用——例如让用户直接说需求（"帮我分析这道 2023 国赛 A 题"），
   或说"用 math-modeling-skill 帮我写摘要"；
5. 提醒：**目录名不要改**，改了宿主按 `name` 找不到。

---

## 不要做

- **不要改目录名。** 安装后的目录必须是 `math-modeling-skill`，和 `SKILL.md` 里的
  `name` 字段逐字符一致。改名不会报错，只会静默失效。
- **不要覆盖已有的技能目录。** 目标位置已经有东西时先看清楚是不是本技能的旧版本；
  是旧版本再用 `--force`，不是就换个位置。安装脚本默认拒绝覆盖，别绕开它。
- **不要把技能装进 `references/`、`assets/`、`scripts/` 这些子目录里。** 技能根目录
  是它们的上一级；装错层级宿主扫不到。
- **不要为了"验证装好了"去跑 `examples/run_algorithms.py` 或下载 LaTeX 模板。**
  那是用户做建模时才需要的东西，装技能不需要，白等几分钟还可能因为没装 numpy 报错。
- **不要编造你的宿主的技能目录路径。** 不确定就问用户，或走 2B 去找。
- **不要 `git clone` 到技能目录里。** 会连 `.git/` 一起留下。

---

## 常见问题

### 装完了，但助手好像没在用这个技能

1. 先确认目录名没被改（`SKILL.md` 的 `name` 与目录名一致）。
2. 确认装的位置真的是**宿主扫描的那个**根目录——回到第 2 步，别猜。
3. DSH 会监视技能根目录，**新增/改名/删除技能下一个技能目录快照就会生效，不需要重启**。
   其它宿主是否要重启，以它自己的文档为准；不确定就让用户重启一次试试。
4. 让用户显式点名："用 math-modeling-skill 帮我……"，看技能是否被读到。

### `--target auto` 选的位置不对

用 `--list-targets` 看完整列表，然后 `--target <名字>` 或 `--into <目录>` 显式指定。
`auto` 只是个方便，不是权威判断。

### Windows 上装到哪了

用户级默认在 `C:\Users\<你>\.agents\skills\math-modeling-skill`。
脚本接受的路径分隔符两种都行，但**文档里一律写正斜杠**。

### 覆盖时报"目标已存在且不是本技能，拒绝删除"

这是故意的安全闸门：目标目录里没有 `name: math-modeling-skill` 的 `SKILL.md`，
脚本不肯删它。换个位置装，或让用户自己确认后手工处理。

### 想装到多个宿主

技能是纯文件，装几份都行，各宿主互不干扰：

```bash
python scripts/install_skill.py --target dsh-user
python scripts/install_skill.py --target claude-user
```

---

## 更新

```bash
cd math-modeling-skill
git pull
python scripts/install_skill.py --target auto --force
```

或者下载新版 Release ZIP 后：

```bash
python scripts/install_skill.py --from-zip /path/to/math-modeling-skill-v1.9.0.zip --target auto --force
```

`--force` 只会覆盖**本技能的旧安装**（目标目录里必须有 `name: math-modeling-skill`
的 `SKILL.md`），别的目录它一律不动。

---

## 卸载

技能就是一堆文件，没有后台进程、不写注册表、不改宿主配置：

```bash
rm -rf "<技能根>/math-modeling-skill"
```

Windows PowerShell：

```powershell
Remove-Item -Recurse -Force "<技能根>\math-modeling-skill"
```

---

## 拿到帮助

- 安装/使用文档：<https://github.com/anticipate218/math-modeling-skill#readme>
- 报问题：<https://github.com/anticipate218/math-modeling-skill/issues>
- 技能支持哪些竞赛、能做什么：读安装后目录里的 `SKILL.md` 与 `references/contests.md`
