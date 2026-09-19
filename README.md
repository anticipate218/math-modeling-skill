# math-modeling-skill

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-spec%20compliant-blue.svg)](https://agentskills.io/specification)
[![CI](https://github.com/anticipate218/math-modeling-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/anticipate218/math-modeling-skill/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](scripts/check_paper.py)

一个给 AI 编码/科研助手用的 **数学建模竞赛技能**（Agent Skill）：把一道赛题变成一篇**评委愿意给高分的论文**。

覆盖三大赛事：**全国大学生数学建模竞赛（国赛 CUMCM）**、**中国研究生数学建模竞赛（华为杯研赛）**、**美国大学生数学建模竞赛（MCM/ICM 美赛）**。

> 它不是一个"自动写论文"的工具，而是一位**懂规则、懂评阅、懂建模流程的教练**：帮你审题选题、选对模型、把检验做扎实、把论文写到规范里，并在提交前逐项卡住那些"会直接出局"的红线。

**English**: An agent skill for mathematical modeling competitions (China Undergraduate/Graduate MCM and COMAP MCM/ICM), providing workflow coaching from problem analysis to submission compliance, grounded in official rules and award-winning paper practices.

---

## 目录

- [快速开始](#快速开始)
- [为什么需要它](#为什么需要它)
- [下载与安装](#下载与安装)
- [它会做什么](#它会做什么)
- [算法与代码](#算法与代码)
- [怎么做出创新点](#怎么做出创新点)
- [质量保障](#质量保障)
- [仓库结构](#仓库结构)
- [AI 使用合规](#一个重要提醒ai-使用的合规)
- [资料可信度约定](#资料可信度约定)
- [与同类项目的关系](#与同类项目的关系)
- [免责声明](#免责声明)
- [License](#license)

---

## 快速开始

**第 0 步：装上它**（初次使用，详见[下载与安装](#下载与安装)）

最省事的方式是**把这句话发给你正在用的 AI 助手，让它自己装**：

> 请阅读并按 <https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/INSTALL.md> 的说明，把 `math-modeling-skill` 这个技能安装到我当前使用的助手环境里；装完告诉我装到了哪个路径、是哪个版本、以及怎么开始用。

想自己动手的话，一行命令也行：

```bash
git clone https://github.com/anticipate218/math-modeling-skill.git ~/.agents/skills/math-modeling-skill
```

或者用技能包自带的安装器（会自动挑位置、丢掉 `.git/`、装完自校验）：

```bash
python scripts/install_skill.py --list-targets   # 先看装哪儿合适
python scripts/install_skill.py --target auto
```

**第 1 步：生成论文骨架（选择你的竞赛）**

```bash
python scripts/check_paper.py --init --contest cumcm -o my_paper.md
```

**第 2 步：拷出对应竞赛的 LaTeX 模板**（想用 LaTeX 排版的话）

```bash
python scripts/download_templates.py --contest cumcm --out my_paper
```

**第 3 步：填写【占位符】提示的内容**

**第 4 步：边写边自检**

```bash
python scripts/check_paper.py my_paper.md --contest cumcm
```

**示例输出**（FAIL=0 即可提交）：
```
论文自检报告 — my_paper.md（cumcm）
结果：FAIL 0 项，WARN 2 项

[INFO] structure: 必备章节齐全（10 项）
[INFO] abstract-ingredients: 摘要要素齐备
[INFO] anonymity: 未检出疑似身份信息
[WARN] references: 参考文献仅 3 条，偏少
[INFO] ai-disclosure: AI 工具使用声明存在（声明为：未使用）
[INFO] validation: 存在结果验证/误差分析
```

> 以上脚本都**不需要安装任何依赖**（纯 Python 标准库），也不需要联网。唯一的外部依赖是可选的：想真的把 LaTeX 模板编译成 PDF，才需要本机有 TeX 发行版。

---

## 为什么需要它

数学建模竞赛的失分，绝大多数不是因为"模型不够高级"，而是因为四类可避免的问题：

1. **没检验**——只给结果，没有误差、对比基线、灵敏度/稳健性分析。评委无从判断对错。
2. **摘要失败**——摘要里没有具体数值结果。而评委往往先读摘要再决定是否细看。
3. **格式红线**——匿名信息、页数超限、附录缺可运行程序、AI 使用声明缺失。这些可能**直接取消评奖资格**。
4. **模型堆砌**——把 AHP+熵权+TOPSIS+神经网络全塞进去，却不解释为什么要用、各解决哪一问。

本技能把这些"规则知识"和"评阅视角"固化成可执行的流程与清单，让 AI 助手在每一步都按竞赛标准来要求你。

---

## 下载与安装

技能遵循 [Agent Skills 开放标准](https://agentskills.io/specification)：一个目录 + 一个 `SKILL.md`（含 YAML frontmatter）。目录名必须与 frontmatter 里的 `name` 一致（本仓库已满足）。

### 1. 最省事：一句话让你的 AI 助手自己装

**不用记任何命令，也不用打开浏览器。** 把下面这句话（连同链接一起）发给你正在用的 AI 助手，它就会自己下载、找到技能目录、装好、再向你汇报：

> 请阅读并按 <https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/INSTALL.md> 的说明，把 `math-modeling-skill` 这个技能安装到我当前使用的助手环境里；装完告诉我装到了哪个路径、是哪个版本、以及怎么开始用。

English:

> Read and follow <https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/INSTALL.md> to install the `math-modeling-skill` agent skill into the environment I'm using. When done, tell me the install path, the version, and how to start using it.

[`INSTALL.md`](INSTALL.md) 是**专门写给 AI 助手看的**：里面写了怎么拿到技能包、怎么确定技能根目录、怎么校验，还有一张「不要做」的清单（不要改目录名、不要覆盖别人的技能、不要装错层级、不要为了验证去跑算法或下载模板）。这份文档对不熟悉本仓库的助手也是自解释的。

**如果你的助手抓不了网页链接**，让它从 Release 包装：

> 请到 <https://github.com/anticipate218/math-modeling-skill/releases/latest> 下载最新的 `math-modeling-skill-v*.zip`，解压后把里面的 `math-modeling-skill` 整个文件夹放到你（助手自己）的技能目录里——**目录名不要改**——然后告诉我放在哪了、怎么开始用。

### 2. 装到哪里（各宿主的技能目录）

| 宿主 / 约定 | 技能目录（安装后应形如 `<技能根>/math-modeling-skill/SKILL.md`） | 说明 |
|---|---|---|
| **DSH**（项目级） | `<项目根>/.dsh/skills/math-modeling-skill` | DSH 技能根表里优先级最高；只对当前项目生效 |
| **DSH**（用户级） | `~/.dsh/skills/math-modeling-skill` | 设置过 `$DSH_HOME` 时以它为准（`$DSH_HOME/skills`） |
| **Agent Skills 通用约定**（项目级） | `<项目根>/.agents/skills/math-modeling-skill` | 跨宿主通用的开放标准位置 |
| **Agent Skills 通用约定**（用户级） | `~/.agents/skills/math-modeling-skill` | 跨宿主通用的开放标准位置 |
| **Claude Code**（项目级） | `<项目根>/.claude/skills/math-modeling-skill` | |
| **Claude Code**（用户级） | `~/.claude/skills/math-modeling-skill` | |

`<项目根>` = 从当前目录向上找到的最近一个含 `.git` 的目录；找不到就用当前目录。

> **上表只列了能核实的路径。** 其它宿主（Codex、Cursor、Gemini CLI、OpenCode……）的技能目录各不相同，本仓库**故意不写死猜测值**——猜错的代价是"装成功了但永远不被扫描"，比装不上更难查。请让助手去读它自己的文档，或问它"你之前装的技能放在哪个目录"，然后用 `--into` 指定。

**不知道自己宿主的技能根在哪？** 先让助手跑一次探测（技能包自带，只用标准库）：

```bash
python scripts/install_skill.py --list-targets
```

它会打印每个候选根目录的**绝对路径**、**是否已存在**、以及**那里已经装的是哪个版本**。

### 3. 用自带安装器装（推荐）

技能包里带了一个只用标准库的安装器：它自己挑技能根、复制时丢掉 `.git/` 与各种缓存、并在覆盖前做安全校验。

```bash
python scripts/install_skill.py --list-targets            # 先看有哪些位置、哪个已存在、已装的是哪个版本
python scripts/install_skill.py --target auto             # 装到自动挑出的位置
python scripts/install_skill.py --target auto --dry-run   # 只看会做什么，不动磁盘
python scripts/install_skill.py --target dsh-user         # 显式指定（--target agents-user / claude-user / ...）
python scripts/install_skill.py --into "~/.agents/skills/math-modeling-skill"   # 精确指定目标目录本身
python scripts/install_skill.py --from-zip math-modeling-skill-v1.8.0.zip       # 从发布包装
python scripts/install_skill.py --download                # 拉最新 Release 的 ZIP 再装（唯一联网的动作）
python scripts/install_skill.py --self-test               # 固件测试：不联网、不碰真实技能目录
```

`--target auto` 的挑选顺序是 **项目级 DSH → 项目级 Agent Skills → 用户级 DSH → 用户级 Agent Skills**，取第一个**已存在**的技能根；一个都不存在时落到 `~/.agents/skills`（跨宿主通用约定）。

三条安全约定，值得知道：

- **默认拒绝覆盖**已存在的技能目录（想覆盖得显式 `--force`）；
- `--force` **只肯删"确实是本技能"的目录**——目标里必须有 `name: math-modeling-skill` 的 `SKILL.md`，这道闸门是防 `--into` 手滑指到家目录的；
- 装完自动用包内的 `scripts/validate_skill.py --strict` 校验一遍，不通过就报错退出。

### 4. 手工安装（`git clone` / Release ZIP）

**`git clone`（推荐，方便日后 `git pull` 拿规则更新）**

```bash
# Agent Skills 通用约定（用户级，跨宿主）
git clone https://github.com/anticipate218/math-modeling-skill.git ~/.agents/skills/math-modeling-skill
```

```powershell
# DSH 用户级（所有项目可用）
git clone https://github.com/anticipate218/math-modeling-skill.git "$env:USERPROFILE/.dsh/skills/math-modeling-skill"

# DSH 项目级（只对当前仓库生效）
git clone https://github.com/anticipate218/math-modeling-skill.git .dsh/skills/math-modeling-skill
```

```bash
# Claude Code 用户级
git clone https://github.com/anticipate218/math-modeling-skill.git ~/.claude/skills/math-modeling-skill
```

clone 进技能目录后**记得把 `.git/` 删掉**（几 MB 历史，技能本身用不到）：

```bash
rm -rf ~/.agents/skills/math-modeling-skill/.git
```

**下载 Release ZIP**（机器上没装 git，或要给队友打包）

到 [Releases](https://github.com/anticipate218/math-modeling-skill/releases) 下载最新一版的 `math-modeling-skill-vX.Y.Z.zip`——内含完整技能包 + 三套 LaTeX 模板，**解压出来的顶层目录就叫 `math-modeling-skill/`**，整个目录丢进技能根目录即可，不用改名。

命令行直接拿（不用打开浏览器）：

```powershell
# PowerShell / Windows
$rel = Invoke-RestMethod https://api.github.com/repos/anticipate218/math-modeling-skill/releases/latest
$zip = ($rel.assets | Where-Object name -like '*.zip' | Select-Object -First 1).browser_download_url
Invoke-WebRequest $zip -OutFile math-modeling-skill.zip
Expand-Archive math-modeling-skill.zip -DestinationPath "$env:USERPROFILE/.agents/skills"   # 解压后即 .../skills/math-modeling-skill/
```

```bash
# bash / macOS / Linux
curl -L -o mms.zip "$(curl -s https://api.github.com/repos/anticipate218/math-modeling-skill/releases/latest \
  | grep -o 'https://[^"]*\.zip' | head -1)"
unzip -q mms.zip -d ~/.agents/skills/          # 解压后即 ~/.agents/skills/math-modeling-skill/
```

（`unzip` 换成 `python -m zipfile -e mms.zip ~/.agents/skills/` 也可以，不依赖 unzip 命令。）

**GitHub 网页下载**（只想看几个文件）：仓库页 `Code → Download ZIP`，解压后**必须把目录重命名为 `math-modeling-skill`**——目录名与 `SKILL.md` 里的 `name` 不一致时宿主会静默忽略它。

**纯手工复制**：把整个目录复制到技能根下，确保路径形如 `<技能根目录>/math-modeling-skill/SKILL.md`（`SKILL.md` 就在这一层，不要再套一层）。

### 5. 依赖

| 用途 | 需要什么 |
|---|---|
| **文档、自检脚本**（`scripts/`、`references/`、`assets/`） | 只有 **Python 3.9+**，纯标准库 |
| **运行自带算法**（`examples/algorithms/`） | 额外需要 **numpy**（`pip install numpy`），**不需要** scipy / sklearn / pandas / statsmodels |
| **真编译 LaTeX 模板**（可选） | 本机装有 TeX 发行版（MiKTeX 或 TeX Live），并有 `xelatex`（中文模板）、`pdflatex`（美赛模板）与 `bibtex` |

除安装器的 `--download` 这一个开关外，所有脚本都不需要联网，也没有任何交互式提示——都能在 CI 里非交互运行。

### 6. 下载 LaTeX 论文模板

仓库自带**三套可直接编译的自包含模板**（每套只有 `main.tex` + `refs.bib` 两个文件，不依赖任何私有宏包）：

| 竞赛 | 模板 | 引擎 | 特点 |
|---|---|---|---|
| 国赛 CUMCM | `assets/latex/cumcm/` | `xelatex` | 中文；摘要页起排；AI 工具使用声明排在**参考文献之前** |
| 研赛（华为杯） | `assets/latex/yjs/` | `xelatex` | 中文；摘要页即第 1 页；无承诺书/编号页；禁止页眉 |
| 美赛 MCM/ICM | `assets/latex/mcm/` | `pdflatex` | 英文；Summary Sheet 独占第 1 页；含 `Report on Use of AI` |

**用脚本一键拷出（推荐）**——它会连编译命令和注意事项一起打印给你：

```bash
# 拷出国赛模板到 my_paper/ 目录
python scripts/download_templates.py --contest cumcm --out my_paper

# 三套一起拷出，并额外打一个 zip 方便分享
python scripts/download_templates.py --contest all --out papers --zip papers.zip

# 只看清单，不拷文件
python scripts/download_templates.py --list
```

`--contest` 可重复使用（`cumcm` / `yjs` / `mcm` / `all`）。目标文件已存在时脚本会**拒绝覆盖**，确认要覆盖再加 `--force`。

**字体的坑，脚本会自动处理**：中文模板默认写的是 `fontset=windows`（调用 Windows 的宋体/黑体，Windows 上开箱即用）。在 Linux / macOS / Overleaf 上这项会因缺字体而失败，所以脚本默认 `--fontset auto`——**在非 Windows 平台自动改写成 `fontset=fandol`**（fandol 字体随 TeX Live 分发）。想手动控制：

```bash
python scripts/download_templates.py --contest cumcm --out my_paper --fontset fandol   # 强制 Fandol（Overleaf 推荐）
python scripts/download_templates.py --contest cumcm --out my_paper --fontset keep     # 一个字都不改
```

**不想用脚本，直接从网页拿单个文件**（右键另存为即可）：

- 国赛：`https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/assets/latex/cumcm/main.tex`（参考文献库 `refs.bib` 把文件名换掉即可）
- 研赛：`.../main/assets/latex/yjs/main.tex`
- 美赛：`.../main/assets/latex/mcm/main.tex`

**拷出来之后怎么编译**（4 遍，顺序不能省，否则交叉引用和参考文献会显示成 `??`）：

```bash
xelatex -interaction=nonstopmode main.tex   # 美赛模板换成 pdflatex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

**排版细节、宏包依赖与常见报错**见 `assets/latex/README.md`；模板选择与答疑见 `references/templates.md`。写完之后可以跑一次编译体检：

```bash
python scripts/check_latex.py --require              # 需要本机有 TeX
python scripts/check_latex.py --self-test            # 不需要 TeX，只验证检查逻辑本身
```

### 7. 装完先验证一下（30 秒）

```bash
cd math-modeling-skill
python scripts/validate_skill.py . --strict    # 期望：0 个错误，0 个警告
python scripts/check_paper.py --self-test      # 期望：全部 PASS
python scripts/download_templates.py --list    # 期望：列出三套模板
python scripts/install_skill.py --self-test    # 期望：13/13 通过
```

`validate_skill.py` 报错通常意味着**目录名被改过**（必须叫 `math-modeling-skill`）或者文件没下全（Release ZIP 比单下几个文件可靠）。

### 8. 怎么更新、怎么卸载

```bash
# 更新：拿到新版（git pull 或换一个新 ZIP）后重装，加 --force 覆盖自己的旧版本
git pull
python scripts/install_skill.py --target auto --force

# 卸载：技能就是一堆文件，没有后台进程、不写注册表、不改宿主配置
rm -rf ~/.agents/skills/math-modeling-skill
```

`--force` 只覆盖**本技能的旧安装**，别的目录一律不动。

### 9. 装完没生效？按这个顺序查

1. **目录名**是否正好是 `math-modeling-skill`——改了名不会报错，只会静默失效。
2. **层级**是否是 `<技能根>/math-modeling-skill/SKILL.md`（`SKILL.md` 必须在技能目录**顶层**，不要再套一层）。
3. **位置**是否真的是宿主扫描的那个根目录——回到[第 2 节](#2-装到哪里各宿主的技能目录)对一下，或让助手读它自己的文档确认（别猜）。
4. **是否要重启**：DSH 会持续监视技能根目录，**新增/改名/删除技能在下一个技能目录快照就会生效，不需要重启**；其它宿主以它自己的文档为准，不确定就重启一次试试。
5. **是否被别的技能抢了触发**：把话说得更明确一点——开头加一句「用 math-modeling-skill 来做…」。

确认技能有没有被读到，最直接的办法是让助手**列出当前可用的技能**，或直接说一句建模需求看它会不会用。

---

## 它会做什么

### 怎么"启动"它

技能不需要安装器、不需要常驻进程，也**不需要你记住任何命令**：宿主启动时扫描技能根目录，读到 `SKILL.md` 的 frontmatter 后，按其中 `description` 写明的场景（中英文触发词都覆盖了）自动决定要不要加载。所以最省事的用法就是**把任务用中文说清楚**。

- **确认装上了**：让助手"列出当前可用的技能"，应该能看到 `math-modeling-skill`；或者直接在技能目录里跑 `python scripts/validate_skill.py . --strict`。
- **想强制指定**：开头加一句"用 math-modeling-skill 来做…"，避免与别的技能抢触发。
- **触发不灵时按顺序查四件事**：① 目录名是否正好叫 `math-modeling-skill`（必须等于 frontmatter 的 `name`）；② 路径是否形如 `<技能根目录>/math-modeling-skill/SKILL.md`（`SKILL.md` 必须在技能目录顶层，不能多套一层）；③ 装的位置是否真是宿主扫描的那个根目录（DSH 会持续监视技能根，**不需要重启**；其它宿主以各自文档为准）；④ 是否被别的技能同场景抢占——此时用上面的强制指定方式。完整排查步骤见[装完没生效？按这个顺序查](#9-装完没生效按这个顺序查)。

安装后，直接对助手说这些话即可触发：

| 你说 | 它会做 |
|---|---|
| "帮我看看这道题该怎么建模" | 拆题 → 判断问题类型 → 从增强模型库给出基线、改进方案与验证设计 |
| "这个题有没有 GitHub 代码可以参考" | 按模型类别检索已核验资源，说明 README 明确内容、许可证、版本和适用边界；不直接复制结论 |
| "帮我比较 ARIMA、XGBoost 和 LSTM" | 先给朴素/统计基线，再按样本量、可解释性和滚动回测比较升级模型 |
| "国赛论文该怎么写/帮我搭个框架" | 给章节骨架、篇幅配比、真实标题样例 |
| "帮我写摘要" | 用摘要模板要求每一问都有"方法 + 数值结果 + 检验结论" |
| "这个模型够不够" | 按官方四维标准（假设合理性/建模创造性/结果正确性/表述清晰度）逐项挑问题 |
| "提交前帮我检查一遍" | 跑自检脚本 + 逐项过对应竞赛的合规清单（含 AI 声明、匿名、页数、查重提示） |
| "美赛和国赛有什么区别" | 给三大赛事的规则/格式/评审差异对照 |
| "给我一套能直接编译的 LaTeX 论文模板" | 指向对应的自包含模板，给编译序列；提醒页数/memo 等可变项以当年官方文件为准 |
| "要 AHP/熵权/TOPSIS/灰色关联的实现" | 给自带的可运行实现 + 自检与黄金值回归，点明正向化、一致性检验等真实陷阱 |
| "帮我找几张优秀论文的图当素材" | 说明不能再分发他人图表，改用自带原创图库当画图范本，并给官方可引用来源索引 |

**增强模型库**：
- [`references/model-implementations.md`](references/model-implementations.md)：按题目类别归类模型族，逐项说明基线、改进阶梯、适用前提与验证要求。
- [`references/github-resources.md`](references/github-resources.md)：从 GitHub 一手 README/仓库页核验的 OR-Tools、Pyomo、sktime、Darts、StatsForecast、statsmodels、scikit-learn、XGBoost、SciML、FiPy、FEniCS、NetworkX、Shapely、Mesa、Nashpy 等资源。
- [`references/algorithm-implementations.md`](references/algorithm-implementations.md)：模型 → 算法 → 复杂度 → 本仓库实现 → 外部库 → 陷阱的对照索引。
- [`examples/modeling_patterns.py`](examples/modeling_patterns.py)：带详细注释的 TOPSIS、滚动均值/MAE、Dijkstra、蒙特卡洛基线；只使用示例数据，不冒充完整解题器。

**可直接用的成品件**：
- [`assets/latex/`](assets/latex/)：国赛 / 研赛 / 美赛三套**单一自包含**的 LaTeX 模板（不 `\input` 外部文件、不依赖外部图片，图表用 TikZ/pgfplots 内联），`xelatex → bibtex → xelatex ×2` 即可编译；已内置各赛事硬规则（摘要页、页码、AI 声明位置、附录源程序）。三套模板**每次 CI 都会被真正编译一遍**（见下方「质量保障」），不是"文档里写着能编"。
- [`examples/algorithms/`](examples/algorithms/)：17 个算法模块、222 个公开函数（优化/图论/启发式/预测/时间序列/统计/评价/多准则/聚类/机器学习/微分方程/随机仿真/几何/空间与物理场/博弈/多目标/灵敏度），**仅依赖 numpy 与标准库**；每个模块都带 `_self_test()`，再用 [`examples/run_algorithms.py`](examples/run_algorithms.py) 跑黄金值回归与确定性复跑。逐函数的数学形式、步骤、参数表与陷阱见 [`references/algorithm-details.md`](references/algorithm-details.md)。
- [`assets/gallery/`](assets/gallery/)：16 张**原创**论文配图（评价权重与敏感性、TOPSIS 排序、预测对比与残差诊断、SIR 机理与参数敏感性、Pareto 前沿、蒙特卡洛收敛、排队仿真、最短路、空间插值、相关矩阵等），由 [`scripts/make_figures.py`](scripts/make_figures.py) 固定种子生成——**逐字节可复现**，可当画图与图注范本。配色与排版经 [`scripts/check_palette.py`](scripts/check_palette.py) 做**可访问性体检**：Okabe-Ito 色相在该项体检里二色觉最差 ΔE 达 16.1，而常见的"论文风"配色 seaborn deep / ColorBrewer Set2 / tab10 分别只有 2.7 / 2.5 / 4.6；另有线型与标记点两条冗余通道兜住黑白打印，见 [`assets/gallery/README.md`](assets/gallery/README.md) §5.7–5.8。

**为什么仓库里没有历年优秀论文的原图**：论文插图版权归作者/出版方，即使标注出处，未经许可把它们下载进仓库再分发通常也不构成合规使用，还会带来学术诚信风险。因此本仓库改为提供「原创可复现图库 + 官方与作者授权来源的链接索引」，见 [`references/paper-examples.md`](references/paper-examples.md)。

**GitHub 资源使用原则**：固定 commit/release，逐项检查仓库/代码/数据许可证，记录访问日期和运行环境；只借实现，不借论文结论，不报告未经核验的 stars 或性能排名。

**自检脚本**（无需安装依赖，Python 3.9+ 标准库即可）：

```bash
python scripts/check_paper.py paper.md --contest cumcm   # 国赛
python scripts/check_paper.py paper.md --contest yjs     # 研赛
python scripts/check_paper.py paper.md --contest mcm     # 美赛
python scripts/check_paper.py paper.md --contest cumcm --json   # 机器可读输出（接入 CI）
python scripts/check_paper.py --self-test                # 验证脚本自身可用
```

它检查：必备章节、摘要要素（问题/方法/结果/关键词）、**匿名合规**（国赛/研赛）、图表编号连续性与**是否在正文被引用**、正文引用标注、**参考文献数量与著录完整性**、附录程序声明、**AI 工具使用声明**及其位置、结果验证痕迹、**单位混用**、篇幅提示。输出 `FAIL/WARN/INFO` 分级，有 FAIL 时退出码为 1，可直接接入 CI。

> 脚本只做**可机械校验**的结构与合规检查；模型合理性、创新性与结果正确性必须人工复核。

---

## 算法与代码

技能不只是文档——`examples/algorithms/` 里有一整套**可以直接跑、可以贴进论文附录**的算法实现：

- **依赖边界**：只用 `numpy` + Python 标准库。不含 scipy / sklearn / statsmodels / pandas / cvxpy / pulp / torch。CI 用 AST 静态扫描强制这条规则——所以你拿到的代码在任何只装了 numpy 的机器（包括评测机房）上都能跑。
- **确定性**：所有随机算法统一走 `_common.rng(seed)`，默认种子 `20240101`，**同一份代码两次运行结果逐位一致**。论文里报的数字经得起复现。
- **每个模块自带 `_self_test()`**，内含闭式解或独立实现的交叉验证（例如 PCA 与 `np.linalg.svd` 对拍、乘法 Holt-Winters 与已有实现逐点对拍、离线黄金值与闭式解对拍），并全部由 `examples/run_algorithms.py` 与黄金值基线比对，含确定性复跑。

```bash
python examples/run_algorithms.py                 # 全量自检 + 黄金值回归
python examples/run_algorithms.py --list          # 看有哪些模块
python examples/run_algorithms.py --module graphs --verbose   # 只跑一个模块
```

覆盖的算法族（按题目类型）：

| 题目族 | 模块 | 你会拿到什么 |
|---|---|---|
| 优化与规划 | `optimization.py` | 单纯形、分支定界整数规划、背包 DP、匈牙利指派、Vogel 运输 |
| 图论与网络 | `graphs.py` | 最短路、最小生成树、最大流/最小割、TSP、PageRank、连通性 |
| 元启发式 | `heuristics.py` | 模拟退火、遗传算法、粒子群、蚁群 |
| 时间序列与预测 | `forecasting.py`、`timeseries.py` | 平滑与 Holt-Winters、AR、平稳性检验、滚动回测、季节分解、进阶模型 |
| 统计与回归 | `statistics.py` | 相关/检验/正态性、OLS 与诊断、岭回归/Lasso、逻辑与泊松回归、PCA/因子分析、Bootstrap 与置换检验 |
| 评价与决策 | `evaluation.py`、`multicriteria.py` | AHP/熵权/CRITIC/组合赋权、TOPSIS、VIKOR、灰色关联、DEA、排序稳健性 |
| 机器学习 | `ml.py` | KNN、决策树、随机森林、梯度提升、朴素贝叶斯、LDA、交叉验证、不平衡处理 |
| 聚类 | `clustering.py` | K-means++、轮廓系数、层次聚类、DBSCAN |
| 微分方程与机理 | `differential.py` | Euler/RK4、SIR 仿真与拟合、Logistic、Lotka-Volterra、收敛阶 |
| 随机与仿真 | `stochastic.py` | 蒙特卡洛、排队论 M/M/1 与仿真、马尔可夫稳态/吸收 |
| 几何与空间 | `geometry.py`、`spatial.py` | 凸包/点在多边形内/Haversine/IDW/克里金/泰森多边形；一维热传导、二维 Poisson 松弛、林火与交通流元胞自动机、量纲分析 |
| 多目标与灵敏度 | `multiobjective.py`、`sensitivity.py` | Pareto 前沿与支配关系、单/多参数灵敏度与稳健性度量 |
| 博弈与分配 | `game.py` | 零和博弈值、Nash 枚举、Shapley 值、稳定匹配、演化动力学 |

**逐算法的细节（数学形式 → 步骤 → 复杂度 → 参数表 → 陷阱 → 怎么检验）见 `references/algorithm-details.md`；"该用哪个族、什么时候换成熟库"见 `references/algorithm-implementations.md`；"这题该用哪类模型"见 `references/model-library.md`。**

> **诚实提醒**：这些实现是**教学透明版**，目的是"让论文能交代清楚每一步"，不是工业级数值库。规模上到几千个决策单元/样本时请换成熟库，并用本仓库实现做原理说明、用成熟库做交叉验证。`references/algorithm-implementations.md` 第 4 节给了替换建议。

---

## 怎么做出创新点

"创新"是国赛/研赛/美赛评奖词里都出现的字眼，也是最容易被写成自嗨的部分。本技能把这件事拆成可执行的步骤，全在 `references/innovation-playbook.md`：

- **创新的五个层级**（数据与假设层 → 模型结构层 → 求解算法层 → 理论层），以及"这道题该往哪层使力"的判断依据。
- **参数创新总表**：按 **17 个算法族**逐条列出**哪些参数能动**、动了属于"调参"还是"结构化改造"，例如——
  - 传播率 β 从常数改成 **β(t)**（干预强度随时间衰减）＝结构化创新；
  - 目标函数权重改几个数 ＝ 只是调参，得配权重-解轨迹才算数；
  - 时间序列的平滑系数手调 ＝ 调参；改成**逐折滚动交叉验证选参** ＝ 消除数据泄漏，算改进。
- **四步法**：机制假设 → 可辨识化（多起点/剖面似然/断点检验）→ **消融实验**（逐个关掉你加的机制项）→ 结论边界。
- **12 条伪创新反面模式**：换库不换模型、重命名式创新、无量纲叠加、只报最好的一次运行、用未来信息、虚假对比、相关当因果……
- **定稿自查清单**：十项打勾，确保每个创新点都能填满"参数/机制 → 指标变化 → 机制解释 → 失效条件"这句话。

配套的验证工具就在 `examples/algorithms/` 里：消融和灵敏度可以调 `sensitivity.py`，多目标权衡用 `multiobjective.py`，参数辨识的残差诊断用 `statistics.py`，启发式解质量用固定种子的多次运行分布。

---

## 质量保障

这个仓库不只有文档，还带多层可自动运行的检查（CI 每次提交都会跑）：

| 层 | 命令 | 检查什么 |
|---|---|---|
| 结构 | `python scripts/validate_skill.py . --strict` | frontmatter 字段白名单、`name` 与目录一致、description/compatibility 长度、正文行数、**文件引用是否存在**、未索引文件、Windows 反斜杠路径 |
| 自检工具 | `python scripts/check_paper.py --self-test` | 用「好稿/坏稿」固件验证检查逻辑本身没坏 |
| 安装器 | `python scripts/install_skill.py --self-test` | 13 项固件测试：复制时确实丢掉 `.git`/`__pycache__`、已存在时先拒绝再 `--force`、**非本技能的目录一律不删**、`--dry-run` 不写盘、带/不带顶层前缀的 ZIP 都能解、`auto` 挑选顺序、项目根向上查找——全程不联网、不碰真实技能目录 |
| 算法回归 | `python examples/run_algorithms.py` | 自带的 **17 个算法模块、875 个断言键**逐个跑数值自检，并与 `algorithms_golden.json` 黄金值比对（含确定性复跑：每个模块跑两遍，结果必须逐位一致） |
| 配图配色 | `python scripts/check_palette.py --quiet` | 直接读 `make_figures.py` 里的设计令牌，算二色觉仿真 CIELAB ΔE、灰度间隔与 WCAG 对比度；并断言「颜色之外还有线型/标记」这条冗余编码确实存在 |
| 模板真编译 | `python scripts/check_latex.py --require`（CI）／`--self-test`（本机无需 TeX） | 在临时目录里真的编译 `assets/latex/` 三套模板（引擎 → bibtex → 引擎 ×2），核对硬错误、未解析引用、缺字体、页数下限，以及 **AI 声明与参考文献的先后顺序** |
| 依赖边界 | 见 `.github/workflows/ci.yml` | AST 扫描 `examples/algorithms/*.py`，禁止引入 scipy/sklearn/pandas 等重型依赖 |
| 触发评测 | 见 `evals/` | 22 条查询（11 正例 + 11 个 near-miss 负例）测 description 触发率；14 条行为用例含**反幻觉断言**（如"不得编造官方评分权重""不得编造 star 数与性能基准""不得再分发他人论文图表"） |
| 持续集成 | `.github/workflows/ci.yml` | 上述全部 + JSON 合法性 + 路径风格 |

其中 `validate_skill.py` 对所有 Agent Skill 作者都有用：它专门拦"跨工具分发时会硬报错"的 frontmatter 问题（比如多写了非标准字段）。

---

## 仓库结构

```
math-modeling-skill/
├── SKILL.md                      # 主控：全流程 7 步 + 官方评奖四维 + 红线 + Gotchas
├── references/                   # 按需加载的详细资料（渐进式披露第二/三层）
│   ├── contests.md               # 三大竞赛规则对照（格式硬规则、AI 规定、纪律、提交物、官方链接）
│   ├── model-library.md          # 模型方法库：赛题信号 → 方法 → 工具 → 陷阱 + 历年赛题反查
│   ├── paper-structure.md        # 论文结构与写作规范（中/英文两套骨架 + 篇幅配比 + 获奖论文章节实证）
│   ├── scoring-rubric.md         # 评阅标准与失分点（官方原文 + 非官方经验明确标注）
│   ├── checklists.md             # 提交前自检清单（通用 / AI 合规 / 三赛事各自专用）
│   ├── github-resources.md       # 已核验 GitHub 模型库与许可证/边界说明
│   ├── model-implementations.md  # 按题目类别的基线、改进阶梯和验证协议
│   ├── algorithm-implementations.md # 模型→算法→复杂度→本仓库实现→外部库→陷阱 对照索引
│   ├── algorithm-details.md      # 逐算法详解：数学形式 → 步骤 → 复杂度 → 参数表 → 陷阱 → 怎么检验
│   ├── innovation-playbook.md    # 怎么做创新：五个层级 / 参数创新总表 / 四步法 / 伪创新反面模式
│   ├── paper-examples.md         # 优秀论文与官方来源索引（只给链接，不再分发他人图表）
│   └── templates.md              # LaTeX 模板选择、编译与排版答疑
├── scripts/
│   ├── check_paper.py            # 论文自检工具（纯标准库，非交互，支持 --json/--self-test/--init）
│   ├── validate_skill.py         # 技能结构校验（frontmatter / 篇幅 / 文件引用，支持 --strict）
│   ├── check_palette.py          # 配图配色可访问性体检（二色觉 ΔE / 灰度间隔 / 对比度）
│   ├── check_latex.py            # 真编译三套 LaTeX 模板并体检（引用/字体/页数/AI 声明顺序）
│   ├── download_templates.py     # 按竞赛一键导出 LaTeX 模板目录（可打包成 zip，支持 --self-test）
│   ├── install_skill.py          # 把技能装进宿主技能目录（自动定位 / 默认不覆盖 / 装完自校验）
│   └── make_figures.py           # 生成 assets/gallery/ 原创论文配图（固定种子，可完整复现）
├── assets/
│   ├── abstract-template.md      # 摘要模板：中文（国赛/研赛）+ 英文 Summary Sheet（美赛）
│   ├── cheatsheet.md             # 一页纸红线清单（可打印，提交前 30 分钟核对）
│   ├── paper-outline.md          # 可填空论文骨架（三赛事，含占位符提示）
│   ├── latex/                    # 三套可直接编译的自包含 LaTeX 论文模板
│   │   ├── cumcm/main.tex        # 国赛（中文，ctexart + bibtex）
│   │   ├── yjs/main.tex          # 研赛（中文）
│   │   └── mcm/main.tex          # 美赛（英文 Summary Sheet + Report on Use of AI）
│   └── gallery/                  # 16 张原创配图 + 画法、配色与图注说明
├── examples/
│   ├── modeling_patterns.py      # 带注释的透明基线示例
│   ├── algorithms/               # 17 个算法模块、222 个公开函数（仅依赖 numpy + 标准库）
│   ├── run_algorithms.py         # 算法自检 + 黄金值回归（确定性复跑）
│   └── algorithms_golden.json    # 黄金值基线（CI 比对用）
├── evals/                        # 触发评测与行为用例（含 near-miss 负例）
├── .github/
│   ├── workflows/ci.yml          # 持续集成
│   └── ISSUE_TEMPLATE/           # Issue 模板（规则更新 / Bug 报告）
├── CONTRIBUTING.md               # 贡献指南（规则追踪项目的特殊要求）
├── INSTALL.md                    # 给 AI 助手看的安装说明（「一句话安装」链接的目标文档）
├── CITATION.cff                  # 引用元数据
├── README.md
├── CHANGELOG.md
└── LICENSE
```

设计上遵循 Agent Skills 的**渐进式披露**原则：`SKILL.md` 只放"每次都要用到"的核心流程（保持精简），详细资料放进 `references/` 由助手按需读取，避免一次性占满上下文。

---

## 一个重要提醒：AI 使用的合规

三大赛事都已就 AI 工具出台规定，且**违规可能直接取消评奖资格**。本技能会主动提醒并检查：

- **国赛**（《人工智能工具使用规定（2026 年试行）》，2026-09-01 起）：须在**参考文献之前**设置「AI工具使用声明」——**未使用 AI 也要声明**；使用 AI 的须在支撑材料附 `AI工具使用详情.pdf`。隐瞒或虚假声明 → 取消评奖资格。
- **研赛**（《人工智能工具及输出使用规定（2025）》）：程序与数据分析处须注释说明 AI 参与；**无推导过程、无引用标识、无法确认来源的 AI 生成模型与公式一律不予认可**。
- **美赛**（COMAP AI 政策）：须披露所用工具并在参考文献列出，另附不计页数的 `Report on Use of AI`。

**请把本技能用于备赛训练、赛后复盘、论文写作与提交前自检。** 竞赛进行中，各赛事均严禁与队外任何人讨论赛题、也禁止在公开平台发布赛题相关内容——不要把赛题贴进对话求解答。

---

## 资料可信度约定

本仓库对每条规则的来源做了**显式标注**，你不必猜哪句是官方的：

| 标注 | 含义 |
|---|---|
| `[官方]` | 来自官网/官方文件的原文或直接转述 |
| `[半官方]` | 来自官方平台转载或官方人员表述 |
| `[社区]` | 业内共识或经验值，**无官方出处** |

同时明确记录了几处**官方口径冲突与未公开项**，例如：

- **官方从未公布评分权重表**。唯一官方评奖标准是国赛《章程》第二条的四维定性描述。网上流传的"模型 30%＋求解 25%…"没有官方依据。
- 每题的**「评阅要点」是赛前发给评阅组的内部文件，不对外公布**；公开的题目级评阅思路见《数学建模及其应用》期刊的「评阅综述／问题解析」。
- 国赛全国规范**不规定**字体字号（属赛区要求），研赛规范则**规定了**具体字号——两者不可混用。
- "A/B/C 为本科组、D/E 为高职高专组"按惯例成立，但官网规范中无此明文。

细节与出处见 `references/contests.md` 与 `references/scoring-rubric.md`。

---

## 与同类项目的关系

写作本技能时参考并致谢以下公开项目与资料（各自版权归原作者）：

- [latexstudio/CUMCMThesis](https://github.com/latexstudio/CUMCMThesis) —— 国赛 LaTeX 论文模板（已适配 2026 年格式，含 AI 声明书）。
- [redefine0130/GMCMthesis-LaTeX-Template](https://github.com/redefine0130/GMCMthesis-LaTeX-Template) —— 研赛 LaTeX 模板。
- [latexstudio-org/mcmthesis](https://github.com/latexstudio-org/mcmthesis) —— 美赛 LaTeX 模板（CTAN 事实标准）。
- [handsomeZR-netizen/mathmodel-skill](https://github.com/handsomeZR-netizen/mathmodel-skill)、[LiXiang106991/MathModelAgent](https://github.com/LiXiang106991/MathModelAgent) —— 面向数学建模的 AI 技能/Agent 实践，本项目的参考资料之一（未复制其内容）。
- [agentskills.io](https://agentskills.io/specification) —— Agent Skills 开放标准（格式规范与写作方法论的依据）。
- 各竞赛官方文件：全国大学生数学建模竞赛官网、中国研究生创新实践系列大赛平台、COMAP 官方竞赛规则。

语料统计参考：[yuanchen-home/cumcm-step-review](https://github.com/yuanchen-home/cumcm-step-review) 的 64 篇国赛获奖论文结构分析。

本仓库内容为独立撰写。若你认为某处引用不当，欢迎提 issue。

---

## 免责声明

本项目**不隶属于**任何竞赛组委会，规则整理仅供备赛参考。**竞赛规则逐年修订，参赛前请务必以当年官方文件为准**（官方链接见 `references/contests.md`）。因使用本技能产生的任何竞赛后果，作者不承担责任。

---

## License

[MIT](LICENSE) © 2026 anticipate218
