## [1.6.0] - 2026-09-18

本版补上一个**一直在漏的覆盖缺口**：`assets/latex/` 下的三套论文模板此前**从未在 CI 里被编译过**——CI 只校验仓库结构、算法模块和配图，不碰 LaTeX。也就是说模板可以一直悄悄地坏下去（宏包改名、`\cite` 打错、字体装不上），仓库照样全绿，而学生拿到手第一遍编译就报错。本版把"这三个模板真的能编译"变成 CI 上的硬门禁。

### 关键结论（先说结果）

- **中文字体是唯一不能照搬的一环，已经写进脚本说明**：国赛/研赛模板用 `fontset=windows`（调用 Windows 自带的宋体/黑体，学生开箱即用），但 Windows 字体在 Linux 上并不存在。CI 因此在**临时副本**里把它换成随发行版自带的 `fandol` 再编译；**仓库里的模板一个字都不改**。想验证仓库原件本身，在有 Windows 字体的机器上跑 `--keep-fontset` 即可——两种配置本机都实测通过。
- **判断成败不能依赖日志里的字节数**：TeX Live 写 `Output written on main.pdf (9 pages, 341464 bytes).`，而 MiKTeX 只写 `Output written on main.pdf (9 pages).`。把字节数当必填，会在 MiKTeX 上把明明编译成功的模板判成"没产出 PDF"。页数取自日志，体积一律以磁盘上真实文件为准。

### 新增

- **`scripts/check_latex.py`：三个模板的真编译体检脚本（只依赖标准库）**
  - 把每个模板的 `main.tex` + `refs.bib` 复制到系统临时目录，按模板文件头写明的顺序编译（`xelatex`/`pdflatex` → `bibtex` → 再两遍），**不在仓库里留下任何 .aux/.log/.pdf**。
  - 解析 `main.log` / `main.blg` 判定：硬错误（`!` 开头）、未解析的 `\cite` 与 `\ref`、字体缺失（`The font "..." cannot be found`）、`Emergency stop`、交叉引用未收敛（`Rerun to get cross-references right`）、是否真的产出 PDF、页数是否低于下限。
  - **顺带核对两条合规顺序**（查 .tex 源码）：国赛/研赛「AI 工具使用声明」必须在参考文献**之前**，美赛「Report on Use of AI」必须在参考文献**之后**（即 25 页正文之外）。
  - `--self-test` 用合成日志跑 **26 项**固件测试，**不需要装 TeX**；`--require` 让"找不到引擎"判为失败而不是跳过（CI 用）；另有 `--only` / `--keep` / `--keep-fontset` / `--tex-dir` 便于本地排查。
- **`.github/workflows/ci.yml` 新增 `latex` 作业**：装 TeX Live 后跑 `check_latex.py --require`。该作业**先跑不依赖 TeX 的 `--self-test`**，这样一旦 CI 红了能立刻分清是"脚本逻辑坏"还是"发行版缺宏包"。

### 实现说明（两个真踩到的坑，已修并写进自测）

- **顺序核对必须剥掉注释**：最初按整篇文本搜索 `\bibliography{`，结果命中了模板文件头第 13 行那句说明文字 `% 若你暂时不想用 .bib，可把 \bibliography{refs} 换成手写 thebibliography`，于是"AI 声明在参考文献之前"被误判成不合规（注释在第 13 行，AI 声明在第 429 行）。现在先去掉注释再匹配，并加了一条对应的自测。
- **字体集替换只能动代码行**：`fontset=windows` 在每个中文模板里出现 3 次，其中 2 次在说明文字里。整篇替换会把"Linux/macOS 请把 `fontset=windows` 换成 `fontset=fandol`"改成同义反复，替换计数也虚高成 3。现在按行拆出注释、只替换代码部分。
- **`lmodern` 不在任何 `texlive-*` 包里（第一次跑 CI 就是这么红的）**：新作业首次运行结果是 **2/3 通过**——国赛/研赛在 Linux 上用 `fandol` 编译通过（页数与 Windows 上完全一致），美赛模板却以 `! LaTeX Error: File `lmodern.sty' not found.` 直接中止。`lmodern.sty` 由 Debian/Ubuntu 的**顶层包 `lmodern`** 提供，装再多 `texlive-*` 也不会有。已在安装列表里补上 `lmodern`，并加了一行 `kpsewhich lmodern.sty` 做前置断言。这正是这个作业存在的意义：宏包清单的窟窿，只有在真编译时才会暴露。

### 变更

- `SKILL.md` 版本升至 `1.6.0`；`compatibility` 补上 `check_latex.py` 的依赖（本机需有 TeX 发行版提供 `xelatex`/`pdflatex`/`bibtex`）；参考文件索引新增一行。
- `assets/latex/README.md` §6「验证记录」改为**由 `check_latex.py` 一条命令复现**，并把实测数据按 `fontset=windows` 与 `fontset=fandol` 两种配置分开列出（此前只记了前者，且没有说明用的是哪个字体集）。
- `README.md`「质量保障」表与仓库结构树补 `scripts/check_latex.py`。
- `references/templates.md` 补"模板改完后怎么验"。
- `CITATION.cff` 同步版本与日期。

### 关键验证记录

| 项目 | 方式 | 结果 |
|---|---|---|
| 模板编译（CI 等价配置，临时副本用 fandol） | `python scripts/check_latex.py` | **3/3 通过**：cumcm 9 页 / 341,465 B；yjs 8 页 / 366,745 B；mcm 8 页 / 284,317 B。硬错误 0，未解析 `\cite`/`\ref` 各 0 |
| 模板编译（仓库原件，`fontset=windows`） | `python scripts/check_latex.py --keep-fontset` | **3/3 通过**：cumcm 9 页 / 202,842 B；yjs 8 页 / 212,439 B；mcm 8 页 / 284,317 B |
| 合规顺序 | 同一脚本的源码检查 | 国赛/研赛 AI 声明在参考文献之前 ✓；美赛在其之后 ✓ |
| 解析逻辑固件测试 | `python scripts/check_latex.py --self-test` | **26/26 通过**（无需装 TeX） |
| 仓库未被污染 | 运行前后 `git status --porcelain` | 无输出（编译只发生在系统临时目录） |
| CI 首次运行（Ubuntu + TeX Live） | GitHub Actions run `35358308247` 的 `latex` 作业 | **2/3**：国赛 9 页、研赛 8 页均通过（页数与 Windows 一致），美赛因缺 `lmodern.sty` 失败——**脚本准确报出了缺哪个包**，据此补齐安装列表 |
| CI 修复后（Ubuntu + TeX Live） | GitHub Actions run `35358882584`：`check` 与 `latex` 两个作业 | **全绿**；`latex` 作业 **3/3 通过**：国赛 9 页 / 341 040 B、研赛 8 页 / 366 285 B、美赛 8 页 / 268 965 B |

## [1.5.0] - 2026-09-18

本版**只动配图的画法与配色，不动数据**。目标是把"看起来像论文插图"这件事从审美口号变成可测的工程约束：先量化各候选配色的二色觉可区分度，再决定色板，最后把"学术感"拆成一组设计令牌写进代码。

### 关键结论（先说结果）

- 用 Machado et al. (2009) 严重度 1.0 的二色觉矩阵在**线性 sRGB** 下仿真，再用 CIELAB ΔE*ab 取最小两两距离对比候选色板，**Okabe-Ito 本来就是可访问性最好的选择**：本仓库 `CYCLE` 二色觉最差 ΔE = **16.1**，而常见的"好看配色"分别是 Tol muted **15.6**、Tol bright **13.1**、seaborn muted **11.8**、matplotlib tab10 **4.6**、seaborn deep **2.7**、ColorBrewer Set2 **2.5**。换成 CVPR 常见的 seaborn/Set2 风格会是一次**可访问性倒退**。
- 因此本版**保留 Okabe-Ito 色相**，收益全部来自"怎么用颜色"，而不是"换哪套颜色"。

### 新增

- **`scripts/check_palette.py`：配图配色可访问性体检脚本（可复现、可进 CI）**
  - 用 `ast.parse` **直接读 `make_figures.py` 里的设计令牌**，不导入 matplotlib、不依赖 `assets/gallery/*.png`，因此可在无绘图依赖的环境里跑。
  - 计算四个指标：正常色觉/三种二色觉下的最小 CIELAB ΔE、灰度（WCAG 相对亮度 ×100）最小间隔、对白底 WCAG 对比度、重复色检查。
  - 同时**静态断言冗余编码存在**：`make_figures.py` 里有 `_series(`、`_LINESTYLES` 至少 3 档、`_MARKERS` 数量不少于数据色数——防止以后有人只改颜色不补线型。
  - 阈值：二色觉最差 ΔE ≥ 12、正常色觉 ≥ 18、灰度间隔 ≥ 1.0、最低对比度 ≥ 2.0。支持 `--quiet` 与 `--self-test`。
- **`assets/gallery/README.md` 新增 §5.7 风格统一（设计令牌表）与 §5.8 配色与可访问性**：把每个令牌的取值与用途列表化，并公开上面这组对照数据、阈值来源与"为什么 ≥6 色色板的灰度间隔不可能达到 5"。

### 变更

- **`scripts/make_figures.py` 重写绘图样式层**：
  - 新增设计令牌块（`CYCLE` / `INK` / `INK_SOFT` / `INK_MUTED` / `GRID` / `PANEL_BG` / `EDGE` / `CMAP_SEQ` / `CMAP_DIV`），**全文件不再有散落的十六进制字面量**，改样式只需改这一处。
  - `configure_style()` 统一 rcParams：去顶右脊线、刻度朝外、图例无边框、浅实线网格、近黑墨色（`#262626` 而非纯黑）、连续量一律用感知均匀的 `viridis`（发散量用 `RdBu_r`）。
  - 新增五个绘图助手 `_grid` / `_series` / `_legend` / `_panel` / `_note`，并**把 16 处 `fig.savefig` 全部收敛到唯一的 `_save`**（统一 `dpi`、`bbox_inches="tight"`、白底）。
  - `_series(i)` 提供**颜色 + 线型 + 标记点**三重编码：`turn, idx = divmod(i, len(CYCLE))` 决定线型档位与颜色，标记点按颜色索引绑定。
  - 标题默认左对齐；形如 `(a) xxx` 的标题走 `_panel`，渲染为加粗的 `$\mathbf{(a)}$` 面板标记。
- **配色取舍**：从数据色循环中移除纯黑与黄色 `#F0E442`（对白底对比度仅 **1.32**，细线不可用）。`OKABE_ITO` 保留 7 色，`CYCLE = OKABE_ITO[:6]`。移除这两色**不改变**二色觉最差 ΔE（仍为 16.1）。
- **`assets/gallery/README.md` §7 表格按真实 PNG 重新生成**（尺寸/字节/体积逐张核实，合计 2,051,061 字节 = 2003.0 KB）；§5.1 补充字号与左对齐规则；§5.2 补充 300 dpi 下的预估体积；§8 补充体检脚本用法与改动后的标准流程。
- `README.md` 「质量保障」表新增"配图配色"一行；仓库结构树补 `scripts/check_palette.py`；`assets/gallery/` 说明改为"16 张原创配图 + 画法、配色与图注说明"。
- `.github/workflows/ci.yml` 新增两步：`scripts/check_palette.py --quiet` 与 `scripts/make_figures.py --self-test`。两者都**只需 numpy、不出图**，所以 CI 仍然不装 matplotlib；相应地 `make_figures.py` 里"本脚本刻意不接入 CI"的说明也改为"CI 只跑数值自检、出图在本机"。
- `SKILL.md` 版本升至 `1.5.0`，脚本清单补 `check_palette.py`；`CITATION.cff` 同步版本与日期。

### 设计原则（本版新增）

- **可访问性优先于风格模仿**："CVPR 那种好看"在工程上应拆成可执行条目（去脊线、刻度朝外、无框图例、浅网格、近黑墨色、感知均匀色图、强调色克制），**而不是照搬它的默认调色板**——后者的二色觉可区分度实测差一个数量级。
- **颜色之外必须有冗余编码**：实测 ≥6 色色板的灰度最小间隔普遍只有 0.4~2.5，**没有任何一套能达到 5**，因此黑白打印/灰度阅读场景必须由线型与标记点兜底（对应 WCAG 2.1 1.4.1「不能只靠颜色传达信息」）。
- **审美主张要能被脚本反驳**：配色选择写成 `check_palette.py` 的断言后，任何人（包括后续的自己）都可以用一条命令检验，而不是靠"我觉得更好看"。

### 关键验证记录

| 项目 | 方式 | 结果 |
|---|---|---|
| 配图配色体检 | `python scripts/check_palette.py --quiet` | 全部断言通过；二色觉最差 ΔE = **16.1**，正常色觉 26.4，灰度间隔 1.1，最低对比度 2.25 |
| 对照色板 | 同一脚本内 `REFERENCE` 对照表 | seaborn deep 2.7 / ColorBrewer Set2 2.5 / tab10 4.6 / Tol muted 15.6 / Tol bright 13.1 / seaborn muted 11.8 |
| 图库可复现性 | 重新生成后逐文件比对 | **16/16 逐字节一致**（字节数合计 2,051,061，最大 205.7 KB） |
| 图内数值未变 | `python scripts/make_figures.py --self-test` | **24/24** 数值键与改动前一致 |
| 样式未误伤标题 | 改 `axes.titlelocation` 前后比对 PNG | 逐字节一致（所有标题早已显式指定 `loc`，该改动只是把默认值改对） |
| 字面量收敛 | 统计 `make_figures.py` 中的十六进制颜色字面量 | 仅剩令牌块内的取值，无散落实例 |

## [1.4.0] - 2026-09-18

本版把技能从"文档 + 检查工具"扩展为**文档 + 可直接用的成品件**：能编译的论文模板、能运行的算法实现、能照抄的配图范本，并补齐对应的索引、评测与 CI 校验。

### 新增

- **`assets/latex/`：三套可直接编译的自包含 LaTeX 论文模板**
  - `cumcm/main.tex`（国赛，中文）、`yjs/main.tex`（研赛，中文）、`mcm/main.tex`（美赛，英文）。
  - 每套都是**单一自包含 `.tex` + `refs.bib`**：不 `\input` 外部文件、不依赖外部图片，图表用 TikZ/pgfplots/booktabs/listings 内联绘制——因此不会因为缺文件而编译失败。
  - 已内置各赛事硬规则：摘要页位置、页码、**AI 工具使用声明排在参考文献之前**（国赛/研赛）、附录源程序、美赛 `Report on Use of AI` 位于参考文献之后且不计页数。
  - `assets/latex/README.md` 给出完整编译序列（`xelatex → bibtex → xelatex ×2`）与每一步的作用。
- **`references/algorithm-implementations.md`：模型 → 算法 → 复杂度 → 本仓库实现 → 外部库 → 陷阱 对照索引**，并集中记录跨模块通用陷阱与"什么时候该换成熟库"。
- **`examples/algorithms/`：11 个可直接运行的算法模块**（`optimization` / `graphs` / `heuristics` / `forecasting` / `statistics` / `evaluation` / `clustering` / `differential` / `stochastic` / `geometry` / `game`）。
  - **仅依赖 numpy 与标准库**，Python 3.9+，非交互；CI 用 AST 静态扫描强制这条依赖边界。
  - 每个模块提供 `_self_test() -> dict`，随机算法一律走显式种子（不使用 `np.random` 全局状态），因此**结果可复现**。
  - `examples/run_algorithms.py`：递归类型感知比对 `examples/algorithms_golden.json`（`rtol=atol=1e-9`），并做确定性复跑；支持 `--module/--rtol/--atol/--list/--update-golden`。
- **`assets/gallery/`：16 张原创论文配图 + 逐图说明**，由 `scripts/make_figures.py` 固定种子生成（Agg 非交互后端），可**逐字节复现**。覆盖评价权重与敏感性、TOPSIS 排序、预测对比与残差诊断、SIR 机理与参数敏感性、Pareto 前沿、收敛性、排队仿真、最短路、空间插值、相关矩阵等。`assets/gallery/README.md` 另含绘图规范（字号、dpi、坐标轴单位、误差棒、图注自解释）与"禁止的画法"。
- **`references/paper-examples.md`：优秀论文与官方来源索引**（只给链接与查阅方式，不在仓库中转载他人图表），含三大赛事官方入口、CUMCM 官方 AI 规定、COMAP 授权与材料页，以及"看什么 / 自己画什么 / 现成实现"的逐题型对照表。
- **`evals/evals.json` 新增 e11–e14**：分别覆盖 LaTeX 模板交付、算法实现的可运行性与陷阱、**拒绝再分发他人论文图表**（合规边界）、美赛六题分类与模板差异。持续沿用"新能力 ⇒ 补用例"的规则。

### 变更

- **`scripts/validate_skill.py`**：新增 `--strict`（警告按错误处理）；索引目录扩展到 `references/ scripts/ assets/ evals/ examples/`，并扫描 `SKILL.md` + 各目录下的 `*.md`；新增 Windows 反斜杠路径检测；对未在索引中出现的顶层文件给出警告。
- **`.github/workflows/ci.yml`**：改用 `validate_skill.py . --strict`；新增对 `examples/algorithms/*.py` 的 AST 禁用依赖扫描；新增 `python examples/run_algorithms.py` 算法回归；路径风格检查扩展到 `examples/`。
- **`.gitignore`**：新增 LaTeX 构建产物（`*.aux` `*.bbl` `*.blg` `*.fls` `*.fdb_latexmk` `*.synctex.gz` `*.toc` `*.out` `*.xdv` `*.run.xml` 等），避免把编译中间件提交进仓库。
- `SKILL.md` 版本升至 `1.4.0`，参考文件索引补齐至全部新增文件。
- `README.md` 新增"可直接用的成品件"一节与仓库结构更新；`CITATION.cff` 同步版本与日期。

### 设计原则（本版新增）

- **不再分发第三方论文图表**：论文插图版权归作者/出版方，即使标注出处，未经许可下载进仓库再分发通常也不构成合规使用，并带来学术诚信风险。因此改为提供「原创可复现图库 + 官方/授权来源链接索引」。
- **算法实现是"教学透明版"而非工业库**：目的让论文能交代清每一步（松弛变量、检验数、Ljung-Box 之外的 ADF 响应面来源等），并在文档中明确规模上限与"何时该换成熟库"。
- **数值主张必须可复核**：所有关键实现都与独立参照（成熟库、解析解、穷举最优解）对照后才写入文档，并在 `references/algorithm-implementations.md` §6 留下验证记录。

### 关键验证记录

| 项目 | 方式 | 结果 |
|---|---|---|
| LaTeX 模板 | 依次执行 `xelatex → bibtex → xelatex ×2` | 国赛 9 页 / 研赛 8 页 / 美赛 8 页；**0 硬错误、0 未定义引用、0 overfull hbox** |
| AI 声明位置 | `pdftotext -enc UTF-8` 核对文本偏移 | 国赛/研赛「AI 工具使用声明」均**早于**「参考文献」；美赛 0 个中文字符，`References` 早于 `Report on Use of AI` |
| LP 正确性 | 与 `scipy.optimize.linprog(method="highs")` 随机对照 | 138 个随机 LP，**0 处不一致** |
| DEA 正确性 | 与 `linprog` 对照（`Σλ=1` 作等式） | 随机算例最大绝对偏差 **≈ 6.4e-13** |
| ADF 临界值 | 与 `statsmodels` 的 MacKinnon (2010) 响应面对照 | 63 组组合最大绝对偏差 **8.9e-16**，0 处不一致 |
| 收敛阶 | 步长序列估计 | RK4 **≈ 4.0693**（理论 4）、Euler **≈ 1.0035**（理论 1） |
| 配图库可复现性 | 重新生成后逐文件 SHA-256 比对 | **16/16 完全一致**，0 处不匹配 |
| 全部算法模块 | `python examples/run_algorithms.py` | 11 个模块全部 PASS 并命中黄金值 |

> 说明："所有的模型"按**主流竞赛模型族**作务实覆盖（11 个模块 + 索引），不是字面意义上穷尽一切模型；文档中已如实标明边界。



## [1.3.1] - 2026-09-11

### 新增

- **`evals/evals.json` 新增 e7–e10 行为用例**，为 1.3.0 引入的模型库与资源索引补齐评测覆盖：
  - `e7` 模型族归类与"基线→分级改进→逐级验证"路径，明确要求给出最小可解基线与失效条件。
  - `e8` GitHub 资源检索的引用纪律：记录许可证、版本/提交与访问日期，**不得编造 star 数、维护度或性能基准**。
  - `e9` 诚实拒绝：不推荐未经核验/已失效的专用库，不宣称"比文献更好"，改为可检验的对照设计。
  - `e10` 自带示例的可运行性：轻量依赖、预期输出形态、实际陷阱与"基线非成品"的说明。
- **`evals/trigger-queries.json` 新增 t21/t22**：一条资源检索型正例（GitHub 选库+引用规范），一条 near-miss 负例（通用图结构库推荐），用于检验 description 是否覆盖新能力而不误触发。

### 变更

- `SKILL.md` frontmatter `metadata.version` 由 `1.0.0` 对齐到实际发布线 `1.3.1`。

## [1.3.0] - 2026-09-11

### 新增

- **`references/model-implementations.md`**：按题目类别组织的增强模型库，覆盖优化/运筹、路径/调度、预测/统计、评价/风险、ODE/PDE、几何/物理、网络、博弈、图像和多问综合题；每类提供透明基线、改进阶梯、适用前提、常见错误和验证协议。
- **`references/github-resources.md`**：基于 GitHub 一手仓库页/README 核验的资源索引，覆盖 OR-Tools、Pyomo、sktime、Darts、StatsForecast、statsmodels、scikit-learn、XGBoost、SciML、FiPy、FEniCS、SimPy、NetworkX、PySAL、Shapely、Mesa、Nashpy 等；记录 README 明确能力、语言、许可证和边界。
- **`examples/modeling_patterns.py`**：带详细注释的透明基线示例，包括 TOPSIS、滚动均值/MAE、Dijkstra 和蒙特卡洛概率估计；不绑定具体题目数据，不冒充完整解题器。

### 设计原则

- **先基线再升级**：复杂模型必须通过基线、消融、敏感性或留出验证证明增益，不能以模型名称代替证据。
- **资源可追溯**：建议记录仓库、具体路径、许可证、访问日期、tag/release 或 commit SHA；不报告未经核验的 stars、维护度或性能排名。
- **许可分层**：仓库、示例代码和数据集可能有不同许可证；GPL/AGPL 代码不能未经评估直接并入本仓库的 MIT 发行物。
- **诚实排除**：已确认返回 404 的 DEApy 链接不列入资源索引；DEA 可用成熟优化器自行实现 CCR/BCC 线性规划并说明来源。

### 已核验资源

- [google/or-tools](https://github.com/google/or-tools)：CP-SAT、线性规划、MIP、TSP/VRP、流和指派。
- [Pyomo/pyomo](https://github.com/Pyomo/pyomo)：LP/QP/NLP/MILP/MIQP/MINLP 等代数建模。
- [sktime/sktime](https://github.com/sktime/sktime)、[unit8co/darts](https://github.com/unit8co/darts)、[Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)：时间序列预测与回测生态。
- [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels)、[scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)、[dmlc/xgboost](https://github.com/dmlc/xgboost)：统计推断、机器学习和表格数据基线。
- [SciML/DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)、[usnistgov/FiPy](https://github.com/usnistgov/fipy)、[FEniCS/dolfinx](https://github.com/FEniCS/dolfinx)：ODE/PDE/有限元与有限体积。
- [networkx/networkx](https://github.com/networkx/networkx)、[PySAL/pysal](https://github.com/pysal/pysal)、[Toblerity/Shapely](https://github.com/shapely/shapely)：图、空间统计和几何。
- [mesa/mesa](https://github.com/mesa/mesa)、[drvinceknight/Nashpy](https://github.com/drvinceknight/Nashpy)：ABM 与双人矩阵博弈。

---

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
