# Changelog

本文件记录本项目的所有重要变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 格式。

## [1.2.0] - 2026-09-11

### 新增
- **`scripts/check_paper.py --init`**：生成对应竞赛的论文骨架（Markdown，自带所有必备章节与占位符提示）
- **`assets/cheatsheet.md`**：一页纸红线清单（可打印，提交前 30 分钟核对；覆盖三赛事通用红线 + 专项硬规则）
- **`assets/paper-outline.md`**：可填空论文骨架（国赛/研赛/美赛三套，含占位符与"此处必须出现数值"提示）
- **`CITATION.cff`**：引用元数据（GitHub 显示"Cite this repository"）
- **`CONTRIBUTING.md`**：贡献指南（规则追踪项目的特殊要求：每条规定必须标注来源，官方未公布的不编造）
- **`.github/ISSUE_TEMPLATE/`**：Issue 模板（规则更新 / Bug 报告 / config.yml）

### 改进
- **`scripts/check_paper.py`**：
  - 新增 **单位混用检测**（小时 vs 分钟、万元 vs 元、千米 vs 米）
  - 新增 **图表未引用警告**（图 1 只出现一次 → WARN "可能未在正文引用"）
  - 新增 **参考文献格式粗检**（条目数、年份缺失、GB/T 7714 风格）
  - **修正美赛摘要要素检查**：改用英文模式（problem/goal、method/model、results/conclusions），不再要求 keywords（MCM 官方未要求），不再误判英文 Summary 缺要素
  - **修正摘要区块识别正则**：要求"摘要/Summary"必须是行首标题，避免被文件头注释中的"摘要页"字样抢先匹配
  - 骨架自检：生成的模板本身不应出现任何 FAIL（用户一开局就有 FAIL=0 的起点）
- **`README.md`**：新增英文摘要、目录、快速开始示例输出、致谢项目补充

### 设计依据
- **骨架生成 `--init`**：形成"生成 → 填写 → 自检"闭环，用户从 FAIL=0 起步而非从空白页起步
- **一页纸清单**：参考飞行检查单（checklist）理念，打印后贴在电脑旁，提交前 30 分钟逐项勾选
- **开源项目完整性**：CITATION.cff + CONTRIBUTING.md + Issue 模板 → 看起来像维护项目而非一次性上传

---

## [1.1.0] - 2026-09-11

### 新增

- **references/templates.md**：模板与工具链——三大赛事 LaTeX 模板选型（含"社区模板封面字段与匿名要求冲突"的警告）、XeLaTeX / pdfLaTeX 编译差异、图表与参考文献排版规范、matplotlib 中文字体配置、结果落盘模板（从根上避免"论文数字与代码不符"）。
- **scripts/validate_skill.py**：技能结构校验器——frontmatter 字段白名单（拦截"跨工具分发会硬报错"的非标准字段）、`name` 与目录名一致性、description / compatibility 长度、正文 500 行建议值、**文件引用存在性**、Windows 反斜杠路径。对任何 Agent Skill 作者都通用。
- **evals/**：触发与行为评测。`trigger-queries.json` 20 条查询（10 正例 + 10 个 near-miss 负例）、`evals.json` 6 条行为用例（含**反幻觉断言**，如"不得编造官方评分权重"）、`README.md` 说明触发率测量方法（每条重复 3 次、阈值 0.5、train/validation 划分以防过拟合）。
- **.github/workflows/ci.yml**：持续集成——结构校验、脚本自测、JSON 合法性、路径风格检查。

### 改进

- **check_paper.py 新增三项检查**：单位混用（时间 / 金额 / 长度 / 质量，中英文单位均识别）、图表编号是否在正文被引用（编号只出现一次即提示）、参考文献数量与著录完整性（是否含出版年份）。
- **修正检查逻辑的误报**：灵敏度分析与模型评价不再作为必备章节（实证依据：2021–2025 年 64 篇国赛获奖论文中，二者独立成章的比例仅约 19% 与 28%，多数并入"建模与求解"各问之后），改为内容级 WARN；仅美赛保持 FAIL（COMAP 官方点名要求 sensitivity 与 strengths/weaknesses）。避免把结构规范的获奖论文误判为不合格。
- **自检固件增至三份**（好稿 / 国赛坏稿 / 美赛坏稿），新增覆盖美赛专属的 FAIL 路径。
- **README**：新增徽章、质量保障四层检查表，更新仓库结构与脚本用法说明。

### 设计依据

- 触发评测方法依据 Agent Skills 官方 description 优化指南：造约 20 条查询统计触发率，**near-miss 负例**最有价值，并以 train/validation 划分防止把 description 过拟合到评测集。
- 章节必备性的判定改为以**获奖论文语料实证**为准，而非凭印象规定"必须有检验章节"。

## [1.0.0] - 2026-09-11

### 新增

- **SKILL.md**：数学建模竞赛全流程主控（7 步：确认前提 → 审题选题 → 方法选型 → 假设与符号 → 建模求解 → 检验 → 写作与自检），含三条铁律、官方评奖四维对齐表、Gotchas 与反面模式。
- **references/contests.md**：三大竞赛官方规则对照表，含 2026 年国赛 AI 规定、研赛格式要求、美赛 25 页限制与 Summary Sheet 单页 12pt。
- **references/model-library.md**：模型方法库（11 大族 + 2020–2025 历年赛题信号反查表）。
- **references/paper-structure.md**：论文结构骨架（国赛/研赛 11 节 + 美赛 9 节）与真实章节标题样例。
- **references/scoring-rubric.md**：评阅标准与失分点（官方四维标准 + 社区失分点清单）。
- **references/checklists.md**：提交前检查清单（国赛/研赛/美赛分节点、含 AI 合规专项）。
- **assets/abstract-template.md**：摘要模板（中文 + MCM 英文，含四要素写法 + failure-mode 表）。
- **scripts/check_paper.py**：论文自检脚本（结构完整性、摘要要素、匿名合规、图表编号、AI 声明、附录程序、参考文献、长度估算）。
- **README.md** / **LICENSE** / **.gitignore** / **.gitattributes**。

### 设计依据

- **规则追踪为本**：每条规定标注 `[官方]/[半官方]/[社区]` 三级可信度，附原文链接与页码。
- **诚实优先于"有用"**：未找到官方说明的项目（如评分权重表）明确标注"未找到"而非编造。
- **三赛事平等对待**：国赛/研赛/美赛各有独立章节与检查分支，不偏向任一竞赛。
- **AI 合规从严**：2026 年起国赛/研赛/美赛均有 AI 使用规定，检查脚本与清单全覆盖。
