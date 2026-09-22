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
- [提示词模板](#提示词模板)
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
python scripts/install_skill.py --into "~/.agents/skills"                        # 给技能根：自动补一层 math-modeling-skill
python scripts/install_skill.py --into "~/.agents/skills/math-modeling-skill"    # 精确指定目标目录本身（给技能根也一样对）
python scripts/install_skill.py --from-zip math-modeling-skill-vX.Y.Z.zip     # 从发布包装（X.Y.Z 换成 Release 页上的版本号）
python scripts/install_skill.py --download                # 拉最新 Release 的 ZIP 再装（唯一联网的动作）
python scripts/install_skill.py --self-test               # 固件测试：不联网、不碰真实技能目录
```

`--target auto` 的挑选顺序是 **项目级 DSH → 项目级 Agent Skills → 用户级 DSH → 用户级 Agent Skills**，取第一个**已存在**的技能根；一个都不存在时落到 `~/.agents/skills`（跨宿主通用约定）。

三条安全约定，值得知道：

- **默认拒绝覆盖**已存在的技能目录（想覆盖得显式 `--force`）；
- `--force` **只肯删"确实是本技能"的目录**——目标里必须有 `name: math-modeling-skill` 的 `SKILL.md`，这道闸门是防 `--into` 手滑指到家目录的；
- 装完自动用包内的 `scripts/validate_skill.py --strict` 校验一遍，不通过就报错退出。

`--into` 有个贴心之处：你给**技能根**（例如 `~/.agents/skills`）它会自动补一层 `math-modeling-skill`，你给**技能目录本身**它就原样使用——两种写法都对，实际装到哪儿会在动手前打印出来。反过来，如果那个目录里躺着**别人的技能**（`SKILL.md` 的 `name` 不是本技能），它会拒绝，绝不把文件塞进别人的技能目录。

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

到 [Releases](https://github.com/anticipate218/math-modeling-skill/releases) 下载最新一版的 `math-modeling-skill-vX.Y.Z.zip`——内含完整技能包 + 全部 LaTeX 模板（轻量三套 + 完整文档类四套），**解压出来的顶层目录就叫 `math-modeling-skill/`**，整个目录丢进技能根目录即可，不用改名。

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

仓库自带**两套并行的 LaTeX 模板层**，按用途二选一：

| 你想要 | 用哪套 | 在哪 | 形态 |
|---|---|---|---|
| **正式参赛提交**（要评委熟悉的官方版式，含封面/承诺书/编号页/摘要页等硬性版式） | **完整文档类版**（推荐） | `assets/latex/full/` | 官方 `.cls` + 完整正文骨架 + 预编译样例 PDF |
| 想自己掌控排版、或用 AI 生成的论文 Markdown 快速成稿 | 轻量自包含版 | `assets/latex/{cumcm,yjs,mcm}/` | 单个 `main.tex` + `refs.bib`，不依赖私有宏包 |

完整文档类版的取舍、改了什么、字体怎么处理，见 [`assets/latex/full/README.md`](assets/latex/full/README.md)；两套模板的完整对照见 [`assets/latex/README.md`](assets/latex/README.md)。

#### 6.1 完整文档类版（`assets/latex/full/`，推荐提交用）

四套模板的**完整可编译工程**，目录内已含所有图片与预编译样例 PDF：

| 竞赛 | 目录 | 文档类 | 引擎 | 预编译样例 |
|---|---|---|---|---|
| 研赛·华为杯 **2026 严格格式版** | [`assets/latex/full/hwcup2026/`](assets/latex/full/hwcup2026/) | `hwcup2026.cls`（本仓库自制，逐条复刻官方附件3） | `xelatex` ×3 | `preview.pdf`（3 页） |
| 国赛 CUMCM | [`assets/latex/full/cumcm/`](assets/latex/full/cumcm/) | `cumcmthesis.cls` v2.9 | `xelatex` ×3 | `example.pdf`（12 页） |
| 研赛（华为杯）通用版 | [`assets/latex/full/gmcm/`](assets/latex/full/gmcm/) | `gmcmthesis.cls` v2.2 | `xelatex` ×3 | `MathModel.pdf`（8 页） |
| 美赛 MCM/ICM | [`assets/latex/full/mcm/`](assets/latex/full/mcm/) | `mcmthesis.cls` v6.3.3 | `pdflatex` ×3 | `mcmthesis-demo.pdf`（11 页） |

> **华为杯两套并存**：投 **2026 年**研赛用 `full/hwcup2026/`——封面、4 个 Logo、
> 摘要页固定标签直接取自**官方附件3 渲染图**，版面尺寸与字号逐条对齐 2026-09-16
> 官方《论文格式规范》。`full/gmcm/` 是社区沿用的通用排版版（章节/图表/算法环境
> 更全），保留下来供参考写法。取舍详见 [`assets/latex/full/hwcup2026/README.md`](assets/latex/full/hwcup2026/README.md)。

**下载（三选一）**：

```bash
# ① 从 Release 直接下整套（含全部字体与图片，最省事）
#    https://github.com/anticipate218/math-modeling-skill/releases/latest
#    资产：hwcup2026-template.zip / gmcm-template.zip / cumcm-template.zip / mcm-template.zip

# ② 已经在技能目录里：直接拷出来
cp -r assets/latex/full/hwcup2026 my_paper

# ③ 只想下单个文件：GitHub 网页进目录逐个另存，或走 raw
#    https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/assets/latex/full/hwcup2026/main.tex
```

Release 资产的直达链接（用 `latest`，永远指向最新一版）：

- 2026 研赛·华为杯严格格式版：<https://github.com/anticipate218/math-modeling-skill/releases/latest/download/hwcup2026-template.zip>
- 研赛（华为杯）通用版：<https://github.com/anticipate218/math-modeling-skill/releases/latest/download/gmcm-template.zip>
- 国赛完整模板：<https://github.com/anticipate218/math-modeling-skill/releases/latest/download/cumcm-template.zip>
- 美赛完整模板：<https://github.com/anticipate218/math-modeling-skill/releases/latest/download/mcm-template.zip>

整套技能包（文件名带版本号，所以用安装器自动认版本最省事）：
`python scripts/install_skill.py --download`。

**编译**（四套都不需要 `bibtex`——参考文献是内联 `thebibliography`，跑三遍引擎即可）：

```bash
cd hwcup2026
xelatex -interaction=nonstopmode main.tex     # 2026 研赛；国赛换成 example.tex
xelatex -interaction=nonstopmode main.tex     # 研赛通用版换成 MathModel.tex
xelatex -interaction=nonstopmode main.tex     # 美赛换 pdflatex + mcmthesis-demo.tex
```

**字体**：`gmcm/` 默认调用 Windows 的宋体/黑体/楷体/隶书，仓库里**已附带这 5 个 `.ttf`**（约 44 MB，见下方许可提示），Windows 上开箱即用；**非 Windows 平台（含 Overleaf）会自动回落到 TeX Gyre + 系统可用字体**，不需要改任何配置——这条回落路径由 `scripts/check_latex_full.py` 在 CI 里真实编译验证。**这些 `.ttf` 必须留在模板目录根部，不要挪进 `fonts/` 子目录**（文档类按裸文件名引用它们，挪走会静默丢字）。

`hwcup2026/` **不带任何字体**：有 SimSun/SimHei/Times New Roman 就用（**2026 规范点名的就是宋体/黑体**），没有则回落 **Noto Serif/Sans CJK SC + Liberation Serif**，在 Overleaf/Linux 上照样编得过。但**最终提交版建议在 Windows 上编**，以拿到与官方 Word 附件3 一致的宋体/黑体字形。Debian/Ubuntu 上跑通回落分支需装 `fonts-noto-cjk` 与 `fonts-liberation`（CI 已点名）。

> **字体许可提示**：这 5 个 `.ttf` 是 Windows/中易（SinoType）的商业字体，**不属于本仓库的 MIT 授权范围**，随模板附上仅为保证与 Word 版式字形一致。若你不便使用，直接删掉它们即可——上面的回落路径会接管。详见 [`assets/latex/full/THIRD-PARTY.md`](assets/latex/full/THIRD-PARTY.md)。

#### 6.2 轻量自包含版（`assets/latex/`）

每套只有 `main.tex` + `refs.bib` 两个文件，不依赖任何私有宏包：

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

python scripts/check_latex_full.py --require         # 完整文档类版：真编译四套（含"本机无 Windows 字体"的回落路径）
python scripts/check_latex_full.py --self-test       # 不需要 TeX，只验证检查逻辑本身
```

### 7. 装完先验证一下（30 秒）

```bash
cd math-modeling-skill
python scripts/validate_skill.py . --strict    # 期望：0 个错误，0 个警告
python scripts/check_paper.py --self-test      # 期望：全部 PASS
python scripts/download_templates.py --list    # 期望：列出三套模板
python scripts/check_latex_full.py --self-test # 期望：26/26 通过（不需要装 TeX）
python scripts/install_skill.py --self-test    # 期望：全部通过
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
| "给我一套能直接编译的 LaTeX 论文模板" | 先问是"提交用"还是"自控排版"：默认给**完整文档类版**（官方 `.cls` + 完整骨架 + 预编译样例），或给轻量自包含版；附编译序列；提醒页数/memo 等可变项以当年官方文件为准 |
| "要 AHP/熵权/TOPSIS/灰色关联的实现" | 给自带的可运行实现 + 自检与黄金值回归，点明正向化、一致性检验等真实陷阱 |
| "帮我找几张优秀论文的图当素材" | 说明不能再分发他人图表，改用自带原创图库当画图范本，并给官方可引用来源索引 |

**增强模型库**：
- [`references/model-implementations.md`](references/model-implementations.md)：按题目类别归类模型族，逐项说明基线、改进阶梯、适用前提与验证要求。
- [`references/github-resources.md`](references/github-resources.md)：从 GitHub 一手 README/仓库页核验的 OR-Tools、Pyomo、sktime、Darts、StatsForecast、statsmodels、scikit-learn、XGBoost、SciML、FiPy、FEniCS、NetworkX、Shapely、Mesa、Nashpy 等资源。
- [`references/algorithm-implementations.md`](references/algorithm-implementations.md)：模型 → 算法 → 复杂度 → 本仓库实现 → 外部库 → 陷阱的对照索引。
- [`examples/modeling_patterns.py`](examples/modeling_patterns.py)：带详细注释的 TOPSIS、滚动均值/MAE、Dijkstra、蒙特卡洛基线；只使用示例数据，不冒充完整解题器。

**可直接用的成品件**：
- [`assets/latex/`](assets/latex/)：**两套并行的三赛事 LaTeX 模板**。① [`full/`](assets/latex/full/)＝**完整文档类版（推荐参赛提交用）**，基于各赛事官方 `.cls`，内含完整正文骨架、图片与预编译样例 PDF，`xelatex`/`pdflatex` 连跑三遍即可；研赛模板默认调用 Windows 字体，仓库已附带 `.ttf`，缺字体时**自动回落到 TeX Gyre + fandol**。② [`cumcm/`](assets/latex/cumcm/)、[`yjs/`](assets/latex/yjs/)、[`mcm/`](assets/latex/mcm/)＝**轻量自包含版**，单个 `main.tex`（不 `\input` 外部文件、图表用 TikZ/pgfplots 内联）+ `refs.bib`，走 `xelatex → bibtex → xelatex ×2`。两套都已内置各赛事硬规则（摘要页、页码、AI 声明位置、附录源程序），且**每次 CI 都会被真正编译一遍**（见下方「质量保障」），不是"文档里写着能编"。
- [`examples/algorithms/`](examples/algorithms/)：17 个算法模块、260 个公开函数（优化/图论/启发式/预测/时间序列/统计/评价/多准则/聚类/机器学习/微分方程/随机仿真/几何/空间与物理场/博弈/多目标/灵敏度），**仅依赖 numpy 与标准库**；每个模块都带 `_self_test()`，再用 [`examples/run_algorithms.py`](examples/run_algorithms.py) 跑黄金值回归与确定性复跑。逐函数的数学形式、步骤、参数表与陷阱见 [`references/algorithm-details.md`](references/algorithm-details.md)。
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

## 提示词模板

> **这一节是拿来就抄的。** 技能不用背命令、不用记参数——把下面任意一条模板原样发给你的 AI 助手，再把 `【】` 里的内容换成你自己的题目信息就行。
>
> 每条模板背后都对应技能里**真实存在的一条流程**（审题选题 → 选型 → 求解 → 检验 → 写作 → 提交自检），它们不是"咒语"：你多写两句、少写两句都不会让它跑偏；反过来，如果你只说"帮我做这道题"，拿到的多半是一份什么都沾一点、什么都不落地的泛泛之谈。
>
> 想强制触发本技能（避免被别的技能抢走），在任意模板最前面加一句：**「用 math-modeling-skill 来做：……」**

### 一、好提示词的四件套

| 要素 | 缺了会怎样 | 怎么写 |
|---|---|---|
| **① 赛事与约束** | 它按"通用论文"写，页数、匿名、AI 声明全不对 | 先报赛事（国赛 CUMCM / 研赛华为杯 / 美赛 MCM-ICM）与交付物 |
| **② 输入与数据** | 它会凭空编数据、编结论 | 贴题干原文；数据说清"有/无、多大、缺不缺、在哪个附件" |
| **③ 硬约束** | 你会得到一份"每个模型都试了一点"的论文 | 写清禁止项：不许编数据、不许三个方案都说好、必须给可复现代码 |
| **④ 交付物与验收** | 你不知道它算不算做完 | 说清要表格/代码/论文段落，以及"怎么算合格" |

同一件事的两种问法：

| ❌ 太糊 | ✅ 可用 |
|---|---|
| "这道题怎么做？" | "先别给模型：拆出这道题的问题类型、数学内核、数据要求，以及最常见的三个坑。" |
| "帮我建个模型" | "给一个朴素基线 + 两级改进，每级写清前提、代价，以及怎么验证它确实更好。" |
| "结果对不对？" | "给我能证伪的检验：灵敏度、稳健性、与独立实现的交叉验证；并指出哪一步最可能错。" |
| "帮我写论文" | "按评分四维逐条挑我稿子的毛病，给出修改后的段落，不用客气。" |
| "这段代码怎么这么慢？" | "指出复杂度瓶颈在哪、什么规模下会撑不住、什么时候该换成成熟库。" |

### 二、按阶段挑模板

| 阶段 | 模板 | 对应技能里的东西 |
|---|---|---|
| 0 · 开局 | [T0 开局设定](#t0-开局设定) | [`references/contests.md`](references/contests.md)、[`references/scoring-rubric.md`](references/scoring-rubric.md) |
| 1 · 审题 | [T1 审题选题](#t1-审题选题) | [`references/model-library.md`](references/model-library.md)、[`references/checklists.md`](references/checklists.md) |
| 2 · 选型 | [T2 模型选型](#t2-模型选型) | [`references/model-implementations.md`](references/model-implementations.md)、[`references/github-resources.md`](references/github-resources.md) |
| 3 · 求解 | [T3 算法实现](#t3-算法实现) | [`examples/algorithms/`](examples/algorithms/)、[`references/algorithm-details.md`](references/algorithm-details.md) |
| 4 · 检验 | [T4 检验与灵敏度](#t4-检验与灵敏度) | [`references/algorithm-implementations.md`](references/algorithm-implementations.md)、`examples/algorithms/sensitivity.py` |
| 5 · 写作 | [T5 论文写作](#t5-论文写作) | [`references/paper-structure.md`](references/paper-structure.md)、[`references/templates.md`](references/templates.md) |
| 6 · 提交 | [T6 提交自检](#t6-提交自检) | [`scripts/check_paper.py`](scripts/check_paper.py)、[`references/checklists.md`](references/checklists.md) |
| 7 · 专项 | [T7 专项任务](#t7-专项任务) | [`assets/latex/`](assets/latex/)、[`assets/gallery/`](assets/gallery/)、[`references/paper-examples.md`](references/paper-examples.md) |
| 8 · 全程 | [T8 全流程托管](#t8-全流程托管) | 全流程编排 + 逐步验收 |

---

### T0 开局设定

> **什么时候用**：比赛刚开始、还没贴题的时候。**目的**：把赛事、时限、角色一次性交代清楚，后面每轮都不用重复。

```text
接下来用 math-modeling-skill 带我打【国赛 CUMCM / 研赛华为杯 / 美赛 MCM-ICM】。
先记住这次比赛的约束，之后每轮回答都按它来：

- 赛程：【开始时间】—【截止时间】，我能投入约【72】小时；
- 队伍情况：我负责【建模与写作】，队友负责【编程】，我们的编程水平是【会 Python / 熟练】；
- 交付物：【中文论文 PDF（含摘要页与附录源程序）/ 英文论文 + memo】；
- 硬规则：匿名、页数上限、AI 使用声明等，一律按【当年官方文件】执行；
  你拿不准的地方直接说"不确定"，并给出官方文件链接，不要凭记忆编。

现在先做三件事，不要解题：
1. 用一张表列出这项赛事与另外两项赛事的评阅差异，标出我最容易踩的 3 条；
2. 给一份从现在到交稿的时间分配建议，必须给检验、交叉验证和排版留出时间；
3. 之后每轮回答末尾加一行「下一步该做什么」。

做完等我贴题。
```

**为什么这样写**：① 赛事差异直接影响页数、匿名、摘要页写法，早说省事；② 明确"不确定就说不确定"，能挡住它编造当年规则；③ 把"下一步"变成固定输出，你就不会推着走。

---

### T1 审题选题

> **什么时候用**：拿到 A/B/C 三道题，要在几道题之间做决定。**目的**：先拆题，不急着上模型。

```text
用 math-modeling-skill。题干和数据在下面（或见附件）：

【粘贴 A/B/C 三题的题干与附件说明】

请按顺序给我结论，不要跳过任何一步：

1. 每道题的**问题类型**（优化 / 预测 / 评价 / 机理 / 统计推断 / 仿真 / 多目标 / 图论…），
   以及它真正在考的数学内核是什么；
2. 每道题的**数据情况**：数据量级、质量、缺什么、要不要自己找数据（要就说清去哪找、怎么引用）；
3. 每道题的**能力匹配度**：我们只会基础 Python、没有专业背景，哪道题对我们更友好；
4. 每道题最容易踩的**坑**：理解歧义、指标陷阱、数据缺失、结论无法检验；
5. 最后给**选题排序**，并说明理由——不要三道题都说"可以选"。

如果某道题我理解错了，先纠正我，再往下讲。
```

**为什么这样写**：① 强制"先拆题后建模"，避免一上来堆模型；② 要求排序而非平铺，逼它表态；③ 让它主动纠正题干误读，这是最省时间的一条。

---

### T2 模型选型

> **什么时候用**：题拆清了，要定基线和改进路线。**目的**：拿到一条"朴素 → 改进 → 进阶"的阶梯，而不是一堆并列的模型名。

```text
用 math-modeling-skill。题目是【一句话概括】，我们已确定做第【2】问，输入数据情况是【…】。

请给我一条模型路线，而不是模型清单：

1. **朴素基线**：最保守、最容易实现、能被评委一眼看懂的那个模型（哪怕很土）；
2. **改进方案 1**：解决基线的哪个具体缺陷？前提假设是什么？代码量大概多少？
3. **改进方案 2**（可选进阶）：只在方案 1 明显不够时启用，说明触发条件；
4. 每个方案的**失效场景**：什么情况下它会得出错误结论；
5. **验证设计**：每个方案怎么证明它比上一层更好（指标、对比实验、独立实现对拍）；
6. **取舍建议**：按我们的时间与编程水平，你推荐停在哪一层，为什么。

要求：给出每个模型在论文里该写成什么形式（目标函数/约束/公式），并标注它属于
references/model-library.md 里的哪一类。不要推荐我们实现不了的模型。
```

**为什么这样写**：① "阶梯"比"清单"更容易落到论文的结构里；② 一旦模型要"失效场景"和"验证设计"，就没法糊；③ 点名让它对齐技能里的模型库，选型有据可依。

---

### T3 算法实现

> **什么时候用**：要把模型落成能跑、能贴进论文附录的代码。**目的**：拿到可复现、可交代、可交叉验证的实现。

```text
用 math-modeling-skill 帮我实现【算法/模型名】。

约束：
- 只允许用 numpy + Python 标准库，不要 scipy / sklearn / pandas / cvxpy 等；
  （如果确实必须用，先说清为什么、以及在评测机房没有该库时怎么退化成纯 numpy 版本）
- 随机部分必须接受显式 seed，同一份代码两次运行结果必须逐位一致；
- 代码要能直接跑，含一个 self-test，用闭式解、小规模暴力枚举或独立实现做交叉验证。

先做一件事：检查 examples/algorithms/ 里是否已经有现成实现可以直接用，
有就复用并告诉我模块名与函数名，不要重复造。

交付内容：
1. 代码（带必要注释：变量含义、边界条件、异常输入处理）；
2. 复杂度（时间/空间）以及本实现能撑到多大规模，多大就该换成熟库；
3. 这份实现的**陷阱清单**：数值稳定性、退化输入、边界条件、常见误用；
4. self-test 的实际运行输出（贴原始结果，不要只写"通过"）；
5. 论文里该怎么描述这个算法的步骤（可直接引用的段落）。
```

**为什么这样写**：① 依赖边界与 seed 是评测机房复现的前提；② 强制交叉验证，"跑通了"不等于"算对了"；③ 最后一条直接产出能进论文的算法描述，省二次加工。

---

### T4 检验与灵敏度

> **什么时候用**：结果算出来了，但你还不知道它站不站得住。**目的**：把结论变成经得起追问的结论。

```text
用 math-modeling-skill 检验我的结果。以下是我们的模型与结果：

【模型简介 + 关键参数 + 结果表格/图】

请做一份"能被评委挑刺"的检验，逐项给结论：

1. **参数灵敏度**：单因素扰动（关键参数 ±10%/±20%）与全局灵敏度（Morris 筛选或 Sobol 指数），
   指出哪几个参数是主导因素；
2. **稳健性**：换种子、换子样本、换边界条件后结论是否翻转；翻转了就直说，不要美化；
3. **误差与不确定性**：误差来源分类（模型误差 / 数据误差 / 数值误差），
   能给出置信区间或误差棒的给出，并说明假设；
4. **交叉验证**：与至少一种独立方法/独立实现对比，给出差异量级并解释差异来源；
5. **最脆弱的一步**：如果你只能让我复查一处，是哪里？为什么？
6. 最后给一段可以直接放进论文"模型的检验"章节的文字。

不要只报"检验通过"：每个结论都必须带数值和单位，没有数值的结论请标为"未验证"。
```

**为什么这样写**：① 技能里 `sensitivity.py` / `statistics.py` 提供了真实可用的检验，提示词把它们的用途说清就能直接调用；② "翻转了就直说"是诚实性的护栏；③ 要求产出论文段落，检验工作不会白做。

---

### T5 论文写作

> **什么时候用**：要写摘要、搭章节、整理图表公式。**目的**：写出评委愿意给高分的结构与表述。

```text
用 math-modeling-skill 帮我写【国赛 CUMCM / 研赛 / 美赛】论文的【摘要 / 第 3 章 / 全部章节骨架】。

素材：【模型、关键算例、结果数值、图表清单】。

要求：
1. 摘要按"每一问 = 方法 + 数值结果 + 检验结论"组织，逐问对应，
   最终结果要具体到数字，不要写"取得了较好效果"；
2. 章节骨架给出每节标题、篇幅配比（页或字数）和这一段必须回答的问题；
3. 公式要有符号表，符号在首次出现处定义，全文符号一致；
4. 图表必须有编号、标题、单位，并确保正文里有引用（"如图 3 所示"），
   缺失的引用直接告诉我缺哪张；
5. 参考文献只允许引用你能给出可核验出处（DOI / 官方页面 / 可访问链接）的条目，
   不确定的一律不要写，不要生成任何编造的文献；
6. 语言风格：结论先行、少形容词、每个论断后面跟证据。

不要替我编结果：我给你的数字之外，任何具体数值都用【待补】标出。
```

**为什么这样写**：① 摘要"方法 + 数值 + 检验"是评分表上的硬指标；② 明确禁止编文献，这是学术诚信红线（技能里另备了可引用来源索引）；③ 要求标【待补】，你就不会被"看起来很完整"的假稿骗过去。

---

### T6 提交自检

> **什么时候用**：交稿前。**目的**：把"会直接出局"的红线逐项卡住。

```text
用 math-modeling-skill 做提交前自检，我要交【国赛 / 研赛 / 美赛】。

请按这个顺序来：
1. 先跑机器能查的：
   python scripts/check_paper.py 【paper.md】 --contest 【cumcm/yjs/mcm】
   把 FAIL / WARN / INFO 逐条解释成人话，并告诉我每一处该怎么改；
2. 再逐条过红线清单（按当年官方规则，逐条给"符合 / 不符合 / 需要我确认"）：
   摘要页与页码、匿名合规（不得出现学校、姓名、队号等身份信息）、页数上限、
   必备章节、图表编号连续且被正文引用、参考文献著录完整、附录源程序、
   AI 工具使用声明的**内容与位置**、结果验证痕迹、单位是否混用；
3. 最后给"最可能让我出局的三件事"，按严重程度排序。

发现问题就直接指出并给出改法，不要只说"建议进一步检查"。
如果某条规则你不确定是当年最新要求，明确说"这条需要你对着官方文件确认"。
```

**为什么这样写**：① 脚本先跑、人工规则后过，顺序对了效率最高；② 匿名与 AI 声明是每年真实出局点；③ 要求它承认不确定项，比给你一份"万事大吉"有用得多。

---

### T7 专项任务

> **什么时候用**：需要一个具体成品件的时候。四条按需取用。

**① 要一套能直接编译的 LaTeX 模板**

```text
用 math-modeling-skill 给我【国赛 / 研赛华为杯 2026 / 研赛通用 GMCM / 美赛】的 LaTeX 模板。
先告诉我两套版本的区别（完整文档类版 vs 轻量自包含版），再按我的场景推荐一套：
- 我要正式提交、机器上装好了 TeX：给完整文档类版（assets/latex/full/…），
  并给出从 Release 资产 ZIP 下载或从技能目录拷出的命令；
- 我只是想自己排版、字体环境不确定：给轻量自包含版（assets/latex/{cumcm,yjs,mcm}），
  用 python scripts/download_templates.py --contest 【…】 --out my-paper 取出来。

最后给出完整的编译序列（xelatex/bibtex 各跑几遍）、缺字体时的回落方案、
以及"哪些是可变量、必须以当年官方文件为准"的提醒。
```

**② 要给论文配图**

```text
用 math-modeling-skill 帮我画【图名】。要求：只用 matplotlib，固定随机种子，可逐字节复现；
配色走 Okabe-Ito 这类色盲友好方案，并额外用线型/标记点做冗余通道，保证黑白打印也能区分；
字号、线宽、刻度方向按论文印刷尺寸设定；输出 PDF + PNG 两个版本。
先告诉我 assets/gallery/ 里有没有同类型的现成范本可以参照，再动手写脚本。
不要让我使用他人论文里的图：版权上不能这么干。
```

**③ 要参考文献与来源**

```text
用 math-modeling-skill 帮我找【主题】的可引用来源。
只给能核验的：官方文档、正式论文（带 DOI）、稳定维护的仓库（带许可证与版本/commit）；
对每条来源说明"它支持我论文里的哪一句结论"以及"适用边界"。
不允许给出无法访问或无法核验的条目；找不到就明确说找不到。
```

**④ 赛后复盘**

```text
用 math-modeling-skill 帮我复盘这次比赛（题干、我们的模型、论文、评委可能的质疑都在下面）。
请指出：① 我们选题判断错在哪；② 建模环节最薄弱的一步；③ 检验缺失造成的风险；
④ 论文表达上被扣分最可能的三个地方；⑤ 下次比赛前该优先补的三项能力。
每条都要引用我们的具体做法作为证据，不要给通用建议。
```

---

### T8 全流程托管

> **什么时候用**：想让助手按流程一步步带你走完，你只做决策。**目的**：把"技能的工作流"变成一次可验收的协作。

```text
用 math-modeling-skill，按下面的流程带我走完这道题。题目与数据：

【题干 + 数据说明】

流程（每个阶段结束后停下来等我确认，再进入下一阶段）：
1. 审题：问题类型、数学内核、数据要求、坑 → 给选题/建模方向建议；
2. 选型：朴素基线 → 改进 1 → 改进 2，含前提、失效场景与验证设计；
3. 求解：可复现代码（numpy + 标准库、固定种子）+ 复杂度 + 陷阱清单 + self-test 原始输出；
4. 检验：灵敏度、稳健性、误差分析、独立实现交叉验证，明确指出最脆弱的一步；
5. 写作：章节骨架、摘要、公式与符号表、图表清单；
6. 提交：跑 python scripts/check_paper.py 【paper.md】 --contest 【…】 并逐条过红线。

全程遵守三条纪律：
- 不许编数据、编文献、编结论；缺什么就说缺什么；
- 每个结论都要有数值或出处；没有的标为"未验证"；
- 每阶段给我一个"结论 + 证据 + 下一步"的小结，我确认后才继续。
```

**为什么这样写**：① 分阶段停下确认，比"一次生成一整篇"质量高得多；② 三条纪律把"它编东西"的风险压到最低；③ 验收点明确，你自己也能照着检查。

---

### 三、微调与常见失效

提示词跑偏时，多数情况是少了一句话。对照下表加一句即可：

| 症状 | 加这句 |
|---|---|
| 只有定性描述，没有数字 | "每个结论后面必须跟具体数值和单位，没有数值的标为未验证。" |
| 它开始编参考文献 | "只允许引用你能给出可核验出处的来源，找不到就说找不到。" |
| 它跳过检验直接下结论 | "没有经过灵敏度/稳健性检验的结论，请单独列一节并标注未验证。" |
| 它同时推荐了五个模型 | "只保留一个推荐方案，其余降级为备选，并说明触发切换的条件。" |
| 输出太长、抓不住重点 | "先用不超过 10 行的表格给结论，再展开细节。" |
| 它反复换模型、越改越飘 | "先固定基线，除非基线被证明失效，否则不要在基线上加复杂度。" |
| 它假装跑过代码 | "贴出实际的运行输出（原始文本），不要复述、不要写'运行成功'。" |
| 它写的内容和你的赛事规则不符 | 把当年官方 PDF 贴给它，并说"以这份文件为准，逐条对照。" |

**两条使用建议**：

- **把模板存下来**：把 T0–T8 存成你们队的 `prompts/` 目录（例如 `prompts/t1-审题.md`），赛前填好【】里的固定信息，比赛时直接复制——限时比赛里，省下的每次打字都是分数。
- **模板要跟着赛事改**：规则类内容（页数、匿名、AI 声明、摘要页格式）**以当年官方文件为准**，模板里已经把这类条款标成"需要确认"，请务必真的去确认。

> 这些模板不会替你比赛：它们的作用是把"该想清楚的事"按顺序摆到你面前，让你把时间花在判断和写作上，而不是花在跟助手来回纠正上。

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
| 优化与规划 | `optimization.py` | 单纯形（含灵敏度分析）、内点法、分支定界整数规划、背包 DP、匈牙利指派、Vogel 运输、目标规划、情景鲁棒与机会约束 LP |
| 图论与网络 | `graphs.py` | 最短路（Dijkstra/Bellman-Ford/Floyd/A\*）、最小生成树、最大流/最小割、最小费用流、TSP、PageRank、连通性、社区发现、拓扑排序与关键路径、VRP |
| 元启发式 | `heuristics.py` | 模拟退火、遗传算法、粒子群、蚁群、差分进化、禁忌搜索、灰狼、变邻域搜索、人工蜂群、标准基准函数库 |
| 时间序列与预测 | `forecasting.py`、`timeseries.py` | 平滑与 Holt-Winters、AR、平稳性检验（含趋势/漂移型）、滚动回测、季节分解、GM(1,1)、ARIMA/SARIMA、GARCH、卡尔曼滤波与平滑 |
| 统计与回归 | `statistics.py` | 相关/检验/正态性、非参数检验（Mann-Whitney/Wilcoxon/Kruskal-Wallis/ANOVA）、OLS 与诊断、HAC(Newey-West) 标准误、岭回归/Lasso、逻辑与泊松回归、PCA/因子分析、Bootstrap（含 BCa）与置换检验 |
| 评价与决策 | `evaluation.py`、`multicriteria.py` | AHP/熵权/CRITIC/组合赋权、TOPSIS、VIKOR、灰色关联、DEA（CCR/BCC）、模糊综合评价、RSR、PROMETHEE II、ELECTRE I/II/III、Borda/Copeland、Kendall 协调系数、排序稳健性 |
| 机器学习 | `ml.py` | KNN、决策树（Gini/熵）、随机森林、梯度提升、朴素贝叶斯、LDA、PCA、交叉验证、不平衡处理、PR 曲线与平均精度 |
| 聚类 | `clustering.py` | K-means++、轮廓系数、层次聚类、DBSCAN、GMM、谱聚类、模糊 C 均值、K-medoids、Gap 统计量 |
| 微分方程与机理 | `differential.py` | Euler/RK4/RK45、隐式 Euler、SIR/SEIR 仿真与拟合、Logistic、Lotka-Volterra、收敛阶、雅可比稳定性、Euler-Maruyama（SDE） |
| 随机与仿真 | `stochastic.py` | 蒙特卡洛、排队论 M/M/1、M/M/c、M/M/c/K、M/G/1 与离散事件仿真、马尔可夫稳态/吸收、MCMC（MH/Gibbs）、Copula、几何布朗运动 |
| 几何与空间 | `geometry.py`、`spatial.py` | 凸包/点在多边形内/Haversine/IDW/克里金/泰森多边形/多边形裁剪；一维热传导、二维 Poisson 松弛、林火与交通流元胞自动机、量纲分析、空间自相关 Moran's I |
| 多目标与灵敏度 | `multiobjective.py`、`sensitivity.py` | Pareto 前沿与支配关系、NSGA-II、MOEA/D、超体积/IGD/Spacing/拐点；OAT/弹性/Morris/Sobol（含二阶）、插补与异常检测 |
| 博弈与分配 | `game.py` | 零和博弈值、Nash 枚举、Shapley 值、稳定匹配、演化动力学、迭代剔除、ESS（演化稳定策略）、相关均衡 LP |

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
| 安装器 | `python scripts/install_skill.py --self-test` | 25 项固件测试：复制时确实丢掉 `.git`/`__pycache__`、已存在时先拒绝再 `--force`、**非本技能的目录一律不删**、`--dry-run` 不写盘、带/不带顶层前缀的 ZIP 都能解、`auto` 挑选顺序、项目根向上查找、**`--into` 给技能根会自动补一层 / 给技能目录则原样使用 / 指向别人的技能目录时拒绝**、**带 UTF-8 BOM 的 `SKILL.md` 仍可识别**、**联网动作会退避重试且失败时给出可执行的出路**、**Release 资产里混着 LaTeX 模板包时仍只认技能包**——全程不联网、不碰真实技能目录 |
| 算法回归 | `python examples/run_algorithms.py` | 自带的 **17 个算法模块、1361 个断言键**逐个跑数值自检，并与 `algorithms_golden.json` 黄金值比对（含确定性复跑：每个模块跑两遍，结果必须逐位一致） |
| 配图配色 | `python scripts/check_palette.py --quiet` | 直接读 `make_figures.py` 里的设计令牌，算二色觉仿真 CIELAB ΔE、灰度间隔与 WCAG 对比度；并断言「颜色之外还有线型/标记」这条冗余编码确实存在 |
| 模板真编译 | `python scripts/check_latex.py --require`（CI）／`--self-test`（本机无需 TeX） | 在临时目录里真的编译 `assets/latex/` 三套轻量模板（引擎 → bibtex → 引擎 ×2），核对硬错误、未解析引用、缺字体、页数下限，以及 **AI 声明与参考文献的先后顺序** |
| 完整模板真编译 | `python scripts/check_latex_full.py --require`（CI）／`--self-test`（本机无需 TeX） | 真的编译 `assets/latex/full/` 四套**完整文档类**模板（各 3 遍，共 7 跑），核对硬错误、缺字、未解析引用、页数下限，并**删掉随附的 `.ttf` + 强制 `fontset=fandol` + 屏蔽 Windows 字体探测后再编一遍**，验证"没有 Windows 字体也能编过、页数不变"这条回落路径 |
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
│   ├── check_latex.py            # 真编译 assets/latex/ 三套轻量模板并体检（引用/字体/页数/AI 声明顺序）
│   ├── check_latex_full.py       # 真编译 assets/latex/full/ 四套完整文档类模板（含无 Windows 字体的回落路径）
│   ├── download_templates.py     # 按竞赛一键导出 LaTeX 模板目录（可打包成 zip，支持 --self-test）
│   ├── install_skill.py          # 把技能装进宿主技能目录（自动定位 / 默认不覆盖 / 装完自校验）
│   └── make_figures.py           # 生成 assets/gallery/ 原创论文配图（固定种子，可完整复现）
├── assets/
│   ├── abstract-template.md      # 摘要模板：中文（国赛/研赛）+ 英文 Summary Sheet（美赛）
│   ├── cheatsheet.md             # 一页纸红线清单（可打印，提交前 30 分钟核对）
│   ├── paper-outline.md          # 可填空论文骨架（三赛事，含占位符提示）
│   ├── latex/                    # 两套并行的 LaTeX 模板层
│   │   ├── README.md             # 两套模板的取舍与对照（先读这个）
│   │   ├── cumcm/main.tex        # 【轻量版】国赛（中文，ctexart + bibtex）
│   │   ├── yjs/main.tex          # 【轻量版】研赛（中文）
│   │   ├── mcm/main.tex          # 【轻量版】美赛（英文 Summary Sheet + Report on Use of AI）
│   │   └── full/                 # 【完整文档类版，推荐提交用】官方 .cls + 完整骨架 + 预编译样例 PDF + 来源与许可说明
│   │       ├── README.md         # 用法、与原版的差异、字体说明、验证记录
│   │       ├── THIRD-PARTY.md    # 四套模板的来源/commit/许可，以及随附 .ttf 与官方渲染图的许可提示
│   │       ├── hwcup2026/        # hwcup2026.cls（自制，严格复刻 2026 官方附件3；main.tex，3 页样例）
│   │       ├── cumcm/            # cumcmthesis.cls v2.9（example.tex，12 页样例）
│   │       ├── gmcm/             # gmcmthesis.cls v2.2 + 5 个 .ttf（MathModel.tex，8 页样例）
│   │       └── mcm/              # mcmthesis.cls v6.3.3（mcmthesis-demo.tex，11 页样例）
│   └── gallery/                  # 16 张原创配图 + 画法、配色与图注说明
├── examples/
│   ├── modeling_patterns.py      # 带注释的透明基线示例
│   ├── algorithms/               # 17 个算法模块、260 个公开函数（仅依赖 numpy + 标准库）
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

写作本技能时参考并致谢以下公开项目与资料（各自版权归原作者）。**完整文档类模板（`assets/latex/full/`）的上游来源、固定 commit 与许可条款见 [`assets/latex/full/THIRD-PARTY.md`](assets/latex/full/THIRD-PARTY.md)**：

- [latexstudio/CUMCMThesis](https://github.com/latexstudio/CUMCMThesis) —— [`full/cumcm/`](assets/latex/full/cumcm/) 的上游（`cumcmthesis.cls` v2.9，commit `38d1f21`，2026-08-26，已适配 2026 年格式，含 AI 声明书）。上游**未附 LICENSE**，也**未上 CTAN**；本仓库按原样收录并标注来源。
- **2026 研赛·华为杯严格格式版 [`full/hwcup2026/`](assets/latex/full/hwcup2026/) 由本仓库作者自制**（`hwcup2026.cls`，无上游）。它依据 2026-09-16 官方《论文格式规范》与官方**附件3 Word 模板**逐条复刻版面尺寸与字号，封面、4 个 Logo 与摘要页固定标签直接取自**官方附件3 的渲染结果**（`assets/*.png`），因此与 Word 版逐像素一致，也不受华文新魏/隶书缺失影响。⚠️ 这几张渲染图的**版权属竞赛组委会**，不在本仓库 MIT 授权范围内。
- **研赛（华为杯）通用版模板 [`full/gmcm/`](assets/latex/full/gmcm/) 由本仓库作者自制整理**（文档类 `gmcmthesis.cls` v2.2，谱系可追溯到公开的 `springli07/GMCM_LaTeX_overleaf` 与 [`zhanwen/MathModel`](https://github.com/zhanwen/MathModel)，两者均未附 LICENSE）。作者在此基础上补了字体回落分支与说明文档。**两套华为杯并存**：投 2026 年研赛用 `hwcup2026/`，`gmcm/` 保留供参考写法。
- [latexstudio-org/mcmthesis](https://github.com/latexstudio-org/mcmthesis) —— [`full/mcm/`](assets/latex/full/mcm/) 的上游（`mcmthesis.cls` v6.3.3，commit `8ac05e2`，**LPPL 1.3c 或更高**；CTAN 事实标准），同时保留了 `mcmthesis.dtx` / `mcmthesis.ins` 以满足再分发条款。文件内容与上游一致，未作修改。
- [handsomeZR-netizen/mathmodel-skill](https://github.com/handsomeZR-netizen/mathmodel-skill)、[LiXiang106991/MathModelAgent](https://github.com/LiXiang106991/MathModelAgent) —— 面向数学建模的 AI 技能/Agent 实践，本项目的参考资料之一（未复制其内容）。
- [agentskills.io](https://agentskills.io/specification) —— Agent Skills 开放标准（格式规范与写作方法论的依据）。
- 各竞赛官方文件：全国大学生数学建模竞赛官网、中国研究生创新实践系列大赛平台、COMAP 官方竞赛规则。

> **关于模板授权**：本仓库只**标注来源**，不对上游授权状态作法律判断（其中仅 `mcmthesis` 有明确的开源许可，另外两套上游未附 LICENSE）。若你在正式场合再分发这些模板，请自行确认上游条款；随 `full/gmcm/` 附带的 5 个 `.ttf` 是商业字体，**不在本仓库 MIT 授权范围内**，删除即可（回落路径会自动接管）；`full/hwcup2026/` 里 5 张官方附件3 渲染图的版权属竞赛组委会，同样**不在 MIT 覆盖范围内**。

语料统计参考：[yuanchen-home/cumcm-step-review](https://github.com/yuanchen-home/cumcm-step-review) 的 64 篇国赛获奖论文结构分析。

本仓库内容为独立撰写。若你认为某处引用不当，欢迎提 issue。

---

## 免责声明

本项目**不隶属于**任何竞赛组委会，规则整理仅供备赛参考。**竞赛规则逐年修订，参赛前请务必以当年官方文件为准**（官方链接见 `references/contests.md`）。因使用本技能产生的任何竞赛后果，作者不承担责任。

---

## License

[MIT](LICENSE) © 2026 anticipate218
