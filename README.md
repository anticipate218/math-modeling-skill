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

## 目录结构

```
math-modeling-skill/
├── SKILL.md                      # 主控：全流程 7 步 + 官方评奖四维 + 红线 + Gotchas
├── INSTALL.md                    # 给 AI 助手看的安装说明（「一句话安装」链接的目标文档）
├── references/                   # 按需加载的详细资料（渐进式披露第二/三层）
│   ├── contests.md               # 三大竞赛规则对照（格式硬规则、AI 规定、官方链接）
│   ├── model-library.md          # 模型方法库：赛题信号 → 方法 → 工具 → 陷阱
│   ├── model-implementations.md  # 按题目类别的基线、改进阶梯和验证协议
│   ├── algorithm-implementations.md # 模型→算法→复杂度→本仓库实现→外部库→陷阱 对照索引
│   ├── algorithm-details.md      # 逐算法详解：数学形式 → 步骤 → 复杂度 → 参数表 → 陷阱 → 怎么检验
│   ├── paper-structure.md        # 论文结构与写作规范（中/英文两套骨架 + 篇幅配比）
│   ├── scoring-rubric.md         # 评阅标准与失分点（官方原文 + 非官方经验明确标注）
│   ├── checklists.md             # 提交前自检清单（通用 / AI 合规 / 三赛事各自专用）
│   ├── innovation-playbook.md    # 怎么做创新：五个层级 / 参数创新总表 / 四步法 / 伪创新反面模式
│   ├── prompt-templates.md       # 可直接抄的提示词模板（T0–T8，从开局到提交）
│   ├── github-resources.md       # 已核验 GitHub 模型库与许可证/边界说明
│   ├── paper-examples.md         # 优秀论文与官方来源索引（只给链接，不再分发他人图表）
│   └── templates.md              # LaTeX 模板选择、编译与排版答疑
├── scripts/                      # check_paper / install_skill / validate_skill / download_templates 等
├── assets/
│   ├── latex/                    # 两套并行的 LaTeX 模板（轻量自包含版 + full/ 完整文档类版）
│   ├── gallery/                  # 16 张原创论文配图（固定种子，可逐字节复现）
│   ├── abstract-template.md      # 摘要模板（中文 + 英文 Summary Sheet）
│   ├── cheatsheet.md             # 一页纸红线清单（可打印）
│   └── paper-outline.md          # 可填空论文骨架
├── examples/
│   ├── algorithms/               # 17 个算法模块、260 个公开函数（仅依赖 numpy + 标准库）
│   ├── run_algorithms.py         # 算法自检 + 黄金值回归
│   ├── algorithms_golden.json    # 黄金值基线（CI 比对用）
│   └── modeling_patterns.py      # 带注释的透明基线示例
├── evals/                        # 触发评测与行为用例（含 near-miss 负例）
├── CHANGELOG.md / CITATION.cff / CONTRIBUTING.md / LICENSE
└── SKILL.md                      # 主控入口（宿主扫描它来自动发现技能）
```

设计上遵循 Agent Skills 的**渐进式披露**原则：`SKILL.md` 只放"每次都要用到"的核心流程，详细资料放进 `references/` 由助手按需读取。

---

## 安装

### 1. 最省事：一句话让你的 AI 助手自己装

**不用记任何命令，也不用打开浏览器。** 把下面这句话发给你正在用的 AI 助手，它就会自己下载、装好、再向你汇报：

> 请阅读并按 <https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/INSTALL.md> 的说明，把 `math-modeling-skill` 这个技能安装到我当前使用的助手环境里；装完告诉我装到了哪个路径、是哪个版本、以及怎么开始用。

English:

> Read and follow <https://raw.githubusercontent.com/anticipate218/math-modeling-skill/main/INSTALL.md> to install the `math-modeling-skill` agent skill into the environment I'm using. When done, tell me the install path, the version, and how to start using it.

[`INSTALL.md`](INSTALL.md) 是**专门写给 AI 助手看的**安装说明；如果你的助手抓不了网页链接，让它去 [Releases](https://github.com/anticipate218/math-modeling-skill/releases/latest) 下载最新的 `math-modeling-skill-v*.zip` 解压安装即可。

### 2. 装到哪里（各宿主的技能目录）

安装后应形如 `<技能根>/math-modeling-skill/SKILL.md`（`SKILL.md` 在技能目录顶层，**目录名不要改**，改了宿主会静默忽略）：

| 宿主 / 约定 | 技能目录 |
|---|---|
| **DSH**（项目级 / 用户级） | `<项目根>/.dsh/skills/math-modeling-skill` ／ `~/.dsh/skills/math-modeling-skill` |
| **Agent Skills 通用约定**（项目级 / 用户级） | `<项目根>/.agents/skills/math-modeling-skill` ／ `~/.agents/skills/math-modeling-skill` |
| **Claude Code**（项目级 / 用户级） | `<项目根>/.claude/skills/math-modeling-skill` ／ `~/.claude/skills/math-modeling-skill` |

> 其它宿主（Codex、Cursor、Gemini CLI……）的技能目录各不相同，本仓库**故意不写死猜测值**——请让助手读它自己的文档，或问它"你之前装的技能放在哪个目录"，然后用 `--into` 指定。

### 3. 安装器（推荐）与手工安装

```bash
python scripts/install_skill.py --list-targets   # 先看有哪些位置、哪个已存在、已装的是哪个版本
python scripts/install_skill.py --target auto    # 装到自动挑出的位置（默认拒绝覆盖，覆盖要 --force）
python scripts/install_skill.py --into "~/.agents/skills"                   # 给技能根：自动补一层技能目录
python scripts/install_skill.py --from-zip math-modeling-skill-vX.Y.Z.zip  # 从 Release 包装
python scripts/install_skill.py --download      # 拉最新 Release 的 ZIP 再装（唯一联网的动作）
python scripts/install_skill.py --self-test     # 固件测试：不联网、不碰真实技能目录
```

装完它会自动用 `scripts/validate_skill.py --strict` 自校验。手工方式等价：

```bash
# git clone（推荐，方便日后 git pull 拿更新；clone 后删掉 .git/ 即可）
git clone https://github.com/anticipate218/math-modeling-skill.git ~/.agents/skills/math-modeling-skill
```

或到 [Releases](https://github.com/anticipate218/math-modeling-skill/releases) 下载 `math-modeling-skill-vX.Y.Z.zip`——**解压出来的顶层目录就叫 `math-modeling-skill/`**，整个目录丢进技能根目录即可。若用 GitHub 网页的 `Code → Download ZIP`，解压后必须把目录重命名为 `math-modeling-skill`（要与 `SKILL.md` 里的 `name` 一致）。

### 4. 依赖

| 用途 | 需要什么 |
|---|---|
| 文档、自检脚本（`scripts/`、`references/`、`assets/`） | 只有 **Python 3.9+**，纯标准库，不联网 |
| 运行自带算法（`examples/algorithms/`） | 额外需要 **numpy**；**不需要** scipy / sklearn / pandas / statsmodels |
| 真编译 LaTeX 模板（可选） | 本机有 TeX 发行版（MiKTeX 或 TeX Live） |

### 5. 装完 30 秒验证 / 更新 / 卸载

```bash
cd math-modeling-skill
python scripts/validate_skill.py . --strict    # 期望：0 个错误，0 个警告

# 更新：git pull 后重装覆盖旧版本；卸载：删目录即可（无后台进程、不改宿主配置）
git pull && python scripts/install_skill.py --target auto --force
rm -rf ~/.agents/skills/math-modeling-skill
```

> 装完没生效？按顺序查四件事：① 目录名是否正好叫 `math-modeling-skill`；② 路径是否形如 `<技能根>/math-modeling-skill/SKILL.md`（不要再套一层）；③ 位置是否真是宿主扫描的那个根（DSH 会持续监视技能根，**不需要重启**）；④ 是否被别的技能抢了触发——开头加一句「用 math-modeling-skill 来做…」。

---

## 下载 LaTeX 论文模板

仓库自带**两套并行的模板层**，按用途二选一：

| 你想要 | 用哪套 | 在哪 |
|---|---|---|
| **正式参赛提交**（评委熟悉的官方版式，含封面/承诺书/编号页/摘要页） | **完整文档类版（推荐）** | [`assets/latex/full/`](assets/latex/full/) |
| 想自己掌控排版、或用 AI 生成的论文 Markdown 快速成稿 | 轻量自包含版 | [`assets/latex/`](assets/latex/)（`cumcm`/`yjs`/`mcm`） |

### 完整文档类版（推荐提交用）

| 竞赛 | 目录 | 文档类 | 引擎 | Release 直达下载 |
|---|---|---|---|---|
| 研赛·华为杯 **2026 严格格式版** | [`assets/latex/full/hwcup2026/`](assets/latex/full/hwcup2026/) | `hwcup2026.cls`（本仓库自制，逐条复刻官方附件3） | `xelatex` ×3 | [hwcup2026-template.zip](https://github.com/anticipate218/math-modeling-skill/releases/latest/download/hwcup2026-template.zip) |
| 国赛 CUMCM | [`assets/latex/full/cumcm/`](assets/latex/full/cumcm/) | `cumcmthesis.cls` v2.9 | `xelatex` ×3 | [cumcm-template.zip](https://github.com/anticipate218/math-modeling-skill/releases/latest/download/cumcm-template.zip) |
| 研赛（华为杯）通用版 | [`assets/latex/full/gmcm/`](assets/latex/full/gmcm/) | `gmcmthesis.cls` v2.2 | `xelatex` ×3 | [gmcm-template.zip](https://github.com/anticipate218/math-modeling-skill/releases/latest/download/gmcm-template.zip) |
| 美赛 MCM/ICM | [`assets/latex/full/mcm/`](assets/latex/full/mcm/) | `mcmthesis.cls` v6.3.3 | `pdflatex` ×3 | [mcm-template.zip](https://github.com/anticipate218/math-modeling-skill/releases/latest/download/mcm-template.zip) |

> **华为杯两套并存**：投 **2026 年**研赛用 `full/hwcup2026/`（版面与字号逐条对齐 2026-09-16 官方《论文格式规范》）；`full/gmcm/` 是社区沿用的通用排版版，保留供参考写法。整套技能包（含全部模板）也可一次拿下：`python scripts/install_skill.py --download`，或到 [Releases](https://github.com/anticipate218/math-modeling-skill/releases/latest) 下 `math-modeling-skill-vX.Y.Z.zip`。

**编译**（四套都不需要 `bibtex`——参考文献是内联 `thebibliography`，跑三遍引擎即可）：

```bash
cd hwcup2026
xelatex -interaction=nonstopmode main.tex     # 2026 研赛；国赛换成 example.tex
xelatex -interaction=nonstopmode main.tex     # 研赛通用版换成 MathModel.tex
xelatex -interaction=nonstopmode main.tex     # 美赛换 pdflatex + mcmthesis-demo.tex
```

**字体**：`gmcm/` 需要的 5 个 Windows 字体 `.ttf` 已随仓库附带（**不在 MIT 授权范围内**，不便使用删掉即可）；非 Windows 平台（含 Overleaf）会自动回落到 TeX Gyre + 系统字体。`hwcup2026/` 不带字体：有宋体/黑体就用，没有则回落 Noto CJK + Liberation，照样编得过，但**最终提交版建议在 Windows 上编**。来源、commit 与许可条款见 [`assets/latex/full/THIRD-PARTY.md`](assets/latex/full/THIRD-PARTY.md)，字体与回落细节见 [`assets/latex/full/README.md`](assets/latex/full/README.md)。

### 轻量自包含版

每套只有 `main.tex` + `refs.bib`，不依赖私有宏包（国赛/研赛 `xelatex`，美赛 `pdflatex`）。用脚本一键拷出（自动处理非 Windows 平台的字体回落）：

```bash
python scripts/download_templates.py --contest cumcm --out my_paper   # 国赛（yjs/mcm/all 同理）
python scripts/download_templates.py --contest all --out papers --zip papers.zip
```

编译走 `xelatex → bibtex → xelatex ×2`（共 4 遍，顺序不能省）。排版细节与常见报错见 [`assets/latex/README.md`](assets/latex/README.md)，模板选择与答疑见 [`references/templates.md`](references/templates.md)。

---

## 怎么用

### 1. 用中文把任务说清楚，技能会自动触发

技能不需要常驻进程：宿主扫描技能根目录、读到 `SKILL.md` 的 frontmatter 后按场景自动加载。安装后直接对助手说这些话即可：

| 你说 | 它会做 |
|---|---|
| "帮我看看这道题该怎么建模" | 拆题 → 判断问题类型 → 给出基线、改进方案与验证设计 |
| "帮我比较 ARIMA、XGBoost 和 LSTM" | 先给朴素/统计基线，再按样本量、可解释性和滚动回测比较升级模型 |
| "国赛论文该怎么写/帮我搭个框架" | 给章节骨架、篇幅配比、真实标题样例 |
| "帮我写摘要" | 按模板要求每一问都有"方法 + 数值结果 + 检验结论" |
| "这个模型够不够" | 按官方四维标准逐项挑问题 |
| "提交前帮我检查一遍" | 跑自检脚本 + 逐项过合规清单（含 AI 声明、匿名、页数） |
| "给我一套能直接编译的 LaTeX 论文模板" | 按场景给完整文档类版或轻量版，附编译序列 |
| "要 AHP/熵权/TOPSIS 的实现" | 给自带的可运行实现 + 自检与黄金值回归，点明真实陷阱 |

### 2. 四步跑通一道题

```bash
# ① 生成论文骨架（选择你的竞赛）
python scripts/check_paper.py --init --contest cumcm -o my_paper.md
# ② 拷出 LaTeX 模板（想用 LaTeX 排版的话）
python scripts/download_templates.py --contest cumcm --out my_paper
# ③ 填写【占位符】提示的内容
# ④ 边写边自检（FAIL=0 即可提交）
python scripts/check_paper.py my_paper.md --contest cumcm
```

自检输出 `FAIL/WARN/INFO` 分级，覆盖必备章节、摘要要素、匿名合规、图表编号与引用、参考文献、**AI 工具使用声明**、结果验证痕迹等——**三大赛事对 AI 使用都有规定，虚假声明可能直接取消评奖资格**，提交前务必过一遍。模型合理性、创新性与结果正确性仍需人工复核。

### 3. 提示词模板（拿来就抄）

把 [`references/prompt-templates.md`](references/prompt-templates.md) 里的任意一条模板原样发给助手，再把 `【】` 换成你的题目信息即可。T0–T8 覆盖全流程：开局设定 → 审题选题 → 模型选型 → 算法实现 → 检验与灵敏度 → 论文写作 → 提交自检 → 专项任务 → 全流程托管，每条都附"为什么这样写"。想强制触发本技能，在模板最前面加一句：**「用 math-modeling-skill 来做：……」**

### 4. 自带算法库

`examples/algorithms/` 提供 **17 个算法模块、260 个公开函数**（优化/图论/元启发式/预测/统计/评价/多准则/聚类/机器学习/微分方程/随机仿真/几何/空间/博弈/多目标/灵敏度/时间序列），**仅依赖 numpy 与标准库**，所有随机算法固定种子、两次运行结果逐位一致：

```bash
python examples/run_algorithms.py                 # 全量自检 + 黄金值回归
python examples/run_algorithms.py --module graphs --verbose   # 只跑一个模块
```

逐算法的数学形式、步骤、参数表与陷阱见 [`references/algorithm-details.md`](references/algorithm-details.md)；"该用哪个族、什么时候换成熟库"见 [`references/algorithm-implementations.md`](references/algorithm-implementations.md)。

### 5. 更多文档

| 想了解 | 去哪 |
|---|---|
| 装到哪个目录、怎么给助手下指令 | [`INSTALL.md`](INSTALL.md) |
| 三大竞赛规则差异与官方链接 | [`references/contests.md`](references/contests.md) |
| 评阅标准、失分点与权重口径 | [`references/scoring-rubric.md`](references/scoring-rubric.md) |
| 提交前红线清单（含 AI 合规） | [`references/checklists.md`](references/checklists.md)、[`assets/cheatsheet.md`](assets/cheatsheet.md) |
| 模型选型与改进阶梯 | [`references/model-library.md`](references/model-library.md)、[`references/model-implementations.md`](references/model-implementations.md) |
| 论文结构、摘要写法 | [`references/paper-structure.md`](references/paper-structure.md)、[`assets/abstract-template.md`](assets/abstract-template.md) |
| 怎么做出创新点 | [`references/innovation-playbook.md`](references/innovation-playbook.md) |
| 质量保障与 CI 检查项 | [`.github/workflows/ci.yml`](.github/workflows/ci.yml)、[`CHANGELOG.md`](CHANGELOG.md) |
| 模板上游来源与许可 | [`assets/latex/full/THIRD-PARTY.md`](assets/latex/full/THIRD-PARTY.md) |

---

## 许可

[MIT](LICENSE) © 2026 anticipate218。本项目不隶属于任何竞赛组委会，规则整理仅供备赛参考，**参赛前请以当年官方文件为准**。随 `full/gmcm/` 附带的商业字体 `.ttf` 与 `full/hwcup2026/` 里的官方渲染图**不在 MIT 覆盖范围内**，详见 [`assets/latex/full/THIRD-PARTY.md`](assets/latex/full/THIRD-PARTY.md)。
