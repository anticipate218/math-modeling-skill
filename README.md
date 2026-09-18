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
- [安装](#安装)
- [它会做什么](#它会做什么)
- [质量保障](#质量保障)
- [仓库结构](#仓库结构)
- [AI 使用合规](#一个重要提醒ai-使用的合规)
- [资料可信度约定](#资料可信度约定)
- [与同类项目的关系](#与同类项目的关系)
- [免责声明](#免责声明)
- [License](#license)

---

## 快速开始

```bash
# 1. 生成论文骨架（选择你的竞赛）
python scripts/check_paper.py --init --contest cumcm -o my_paper.md

# 2. 填写【占位符】提示的内容

# 3. 边写边自检
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

---

## 为什么需要它

数学建模竞赛的失分，绝大多数不是因为"模型不够高级"，而是因为四类可避免的问题：

1. **没检验**——只给结果，没有误差、对比基线、灵敏度/稳健性分析。评委无从判断对错。
2. **摘要失败**——摘要里没有具体数值结果。而评委往往先读摘要再决定是否细看。
3. **格式红线**——匿名信息、页数超限、附录缺可运行程序、AI 使用声明缺失。这些可能**直接取消评奖资格**。
4. **模型堆砌**——把 AHP+熵权+TOPSIS+神经网络全塞进去，却不解释为什么要用、各解决哪一问。

本技能把这些"规则知识"和"评阅视角"固化成可执行的流程与清单，让 AI 助手在每一步都按竞赛标准来要求你。

---

## 安装

技能遵循 [Agent Skills 开放标准](https://agentskills.io/specification)：一个目录 + 一个 `SKILL.md`（含 YAML frontmatter）。目录名必须与 frontmatter 里的 `name` 一致（本仓库已满足）。

**通用安装（把仓库克隆为技能目录）**

```bash
# 通用 Agent Skills 目录（跨工具）
git clone https://github.com/anticipate218/math-modeling-skill.git ~/.agents/skills/math-modeling-skill
```

**DeepSeek Harness**

```powershell
# 用户级技能目录
git clone https://github.com/anticipate218/math-modeling-skill.git "$env:USERPROFILE/.dsh/skills/math-modeling-skill"

# 或项目级（对某个仓库生效）
git clone https://github.com/anticipate218/math-modeling-skill.git .dsh/skills/math-modeling-skill
```

**Claude Code**

```bash
git clone https://github.com/anticipate218/math-modeling-skill.git ~/.claude/skills/math-modeling-skill
```

**手动安装**：把整个仓库目录复制到你的技能根目录下，确保路径形如 `<技能根目录>/math-modeling-skill/SKILL.md`。

---

## 它会做什么

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

**增强模型库**：
- [`references/model-implementations.md`](references/model-implementations.md)：按题目类别归类模型族，逐项说明基线、改进阶梯、适用前提与验证要求。
- [`references/github-resources.md`](references/github-resources.md)：从 GitHub 一手 README/仓库页核验的 OR-Tools、Pyomo、sktime、Darts、StatsForecast、statsmodels、scikit-learn、XGBoost、SciML、FiPy、FEniCS、NetworkX、Shapely、Mesa、Nashpy 等资源。
- [`examples/modeling_patterns.py`](examples/modeling_patterns.py)：带详细注释的 TOPSIS、滚动均值/MAE、Dijkstra、蒙特卡洛基线；只使用示例数据，不冒充完整解题器。

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

## 质量保障

这个仓库不只有文档，还带四层可自动运行的检查（CI 每次提交都会跑）：

| 层 | 命令 | 检查什么 |
|---|---|---|
| 结构 | `python scripts/validate_skill.py .` | frontmatter 字段白名单、`name` 与目录一致、description/compatibility 长度、正文行数、**文件引用是否存在**、Windows 反斜杠路径 |
| 自检工具 | `python scripts/check_paper.py --self-test` | 用「好稿/坏稿」固件验证检查逻辑本身没坏 |
| 触发评测 | 见 `evals/` | 22 条查询（11 正例 + 11 个 near-miss 负例）测 description 触发率；10 条行为用例含**反幻觉断言**（如"不得编造官方评分权重""不得编造 star 数与性能基准"） |
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
│   └── model-implementations.md  # 按题目类别的基线、改进阶梯和验证协议
├── scripts/
│   ├── check_paper.py            # 论文自检工具（纯标准库，非交互，支持 --json/--self-test/--init）
│   └── validate_skill.py         # 技能结构校验（frontmatter / 篇幅 / 文件引用）
├── assets/
│   ├── abstract-template.md      # 摘要模板：中文（国赛/研赛）+ 英文 Summary Sheet（美赛）
│   ├── cheatsheet.md             # 一页纸红线清单（可打印，提交前 30 分钟核对）
│   └── paper-outline.md          # 可填空论文骨架（三赛事，含占位符提示）
├── examples/
│   └── modeling_patterns.py       # 带注释的透明基线示例
├── evals/                        # 触发评测与行为用例（含 near-miss 负例）
├── .github/
│   ├── workflows/ci.yml          # 持续集成
│   └── ISSUE_TEMPLATE/           # Issue 模板（规则更新 / Bug 报告）
├── CONTRIBUTING.md               # 贡献指南（规则追踪项目的特殊要求）
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
