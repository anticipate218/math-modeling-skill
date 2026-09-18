# LaTeX 论文模板（国赛 CUMCM / 研赛华为杯 / 美赛 MCM-ICM）

三套**开箱即用、已在 Windows + MiKTeX 上实际编译通过**的论文模板：

| 目录 | 适用赛事 | 语言 | 编译器 |
|---|---|---|---|
| `cumcm/` | 全国大学生数学建模竞赛 | 中文 | **XeLaTeX** |
| `yjs/` | 中国研究生数学建模竞赛（华为杯） | 中文 | **XeLaTeX** |
| `mcm/` | 美赛 MCM / ICM（COMAP） | 英文 | **pdfLaTeX** |

每个目录**只有两个文件**：`main.tex` + `refs.bib`。没有共享的 preamble、不需要 `\input` 多文件结构——把这两个文件拷进你的工作目录就能编译。

> **规则来源与边界**：模板里的格式要求来自本仓库 `references/contests.md` 记录的官方文件，并在 `.tex` 注释里逐年标注。竞赛规则逐年修订，**提交前必须重新核对当年官方通知**。本模板并非官方模板，排版细节（页边距、字号）在官方未规定的部分做了常规选择，也请以本赛区/本校要求为准。

---

## 一、编译命令

三套模板的编译流程一样（先引擎跑一遍，再 BibTeX，再引擎跑两遍解析交叉引用与页数）：

```bash
# 中文模板（在 cumcm/ 或 yjs/ 目录下执行）
xelatex -interaction=nonstopmode main.tex
bibtex  main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex

# 美赛模板（在 mcm/ 目录下执行）
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

一行版（自动决定跑几遍）：

```bash
latexmk -xelatex main.tex     # cumcm / yjs
latexmk -pdf     main.tex     # mcm
```

> **`latexmk` 需要 Perl**。本仓库作者在 **Windows + MiKTeX** 上实测：MiKTeX 自带的 `latexmk.exe` 是一个 Lua/Perl 包装脚本，若系统没有 `perl`，会报
> `MiKTeX could not find the script engine 'perl' which is required to execute 'latexmk'`（在管理员权限下还可能先报 `security risk: running with elevated privileges`）。
> 这不是模板问题。解决办法：装 [Strawberry Perl](https://strawberryperl.com/) 后重试，或**直接用上面的三行 `xelatex`/`pdflatex` 序列**——那条路径已在 Windows + MiKTeX 上实际编译通过（国赛 9 页 / 研赛 8 页 / 美赛 8 页，0 硬错误、0 未定义引用）。
> 在 Linux/macOS 与 Overleaf 上 `latexmk` 通常开箱可用。

要点：

- **中文模板必须用 XeLaTeX**，不能用 pdfLaTeX。`ctexart` + `fontset=windows` 依赖 XeLaTeX 调用系统字体；用 pdfLaTeX 编会直接报错。
- **美赛模板用 pdfLaTeX**。它**没有加载任何中文宏包**（提交必须全英文），也不要在里面加 `ctex`/`xeCJK`。
- `bibtex` 只在参考文献变动后需要跑；正文里**必须有 `\cite{...}`**，否则 BibTeX 会报 `I found no \citation commands` 且参考文献列表是空的。
- 美赛模板页眉的 “Page X of Y” 用 `lastpage` 宏包，**必须至少编译两遍**才会正确（中文模板只有页脚页码，不受影响）。
- 只跑一遍快速预览（不中断、日志写进 `main.log`）：

```bash
xelatex -interaction=batchmode main.tex     # 不中断、日志写进 main.log
```

### 在 Overleaf 上编译

1. 新建项目，上传 `main.tex` 与 `refs.bib`（同一层目录）。
2. `Menu → Compiler` 选 **XeLaTeX**（美赛模板选 pdfLaTeX）。
3. **中文模板必须改字体**：把 `\documentclass[12pt,a4paper,fontset=windows]{ctexart}` 里的
   `fontset=windows` 改成 `fontset=fandol`（Overleaf/Linux 没有 Windows 自带字体）。
4. 参考文献要手动跑一次 BibTeX，或把 `Menu → Recompile` 旁的下拉设为 “Recompile from scratch”。

---

## 二、依赖宏包

全部是 TeX Live / MiKTeX **完整版的标准宏包**，无需额外下载资源文件（图、表、代码清单全部在 `.tex` 内部生成，模板不依赖任何外部图片文件）。

| 用途 | 宏包 | 用在哪 |
|---|---|---|
| 中文排版 | `ctex`（`ctexart`） | cumcm、yjs |
| 版面 | `geometry`；行距与首行缩进 `setspace`、`indentfirst`（仅中文） | 全部 / 中文 |
| 数学 | `amsmath`、`amssymb`、`bm` | 全部 |
| 三线表 | `booktabs`、`multirow`、`array`、`tabularx` | 全部 |
| 图与流程 | `graphicx`、`float`、`caption`、`tikz`、`pgfplots`；`subcaption`（仅中文） | 全部 / 中文 |
| 伪代码 | `algorithm` + `algpseudocode`（algorithmicx） | 全部 |
| 代码高亮 | `listings` + `xcolor` | 全部 |
| 页眉页脚 | `fancyhdr`（+ `lastpage` 供美赛用） | 全部 |
| 参考文献 | `natbib` + BibTeX 样式 `unsrt` / `unsrtnat` | 全部 |
| 超链接 | `hyperref` | cumcm、yjs |
| 英文字体 | `fontenc[T1]`、`lmodern` | mcm |

若你的发行版是精简安装、缺宏包，MiKTeX 首次编译会提示联网自动安装；也可手动装：

```bash
mpm --install=ctex,geometry,setspace,indentfirst,amsmath,amssymb,booktabs,multirow,array,tabularx,graphicx,float,caption,subcaption,tikz,pgfplots,algorithmicx,listings,xcolor,fancyhdr,lastpage,natbib,hyperref,lmodern
```

> 已避免使用任何需要额外字体的宏包：中文模板只用系统自带的宋体/黑体/楷体/仿宋。

---

## 三、三套模板的差异（模板层面）

| 维度 | cumcm（国赛） | yjs（研赛） | mcm（美赛） |
|---|---|---|---|
| 文档类 | `ctexart`，12pt，A4 | `ctexart`，12pt，A4 | `article`，12pt，letter |
| 第 1 页 | 摘要页（电子版不含承诺书、编号页） | 摘要页（研赛没有承诺书/编号页） | Summary Sheet（单独一页） |
| 摘要篇幅 | 摘要页含标题与关键词 ≤1 页 | 一般 ≤2 页 | Summary Sheet 独占第 1 页，篇幅随题面，通常控制在一页内 |
| 正文页数 | ≤30 页、不要目录 | 国家级规范未设上限（省级可能更严） | **整份提交 ≤25 页（含目录、参考文献、附录与代码）** |
| 页码 | 页脚中部，从摘要页连续编号 | 页脚中部，从摘要页连续编号 | **每页页眉**：`Team # 控制号, Page X of Y` |
| 页眉 | 无（本模板未设页眉） | **禁止任何页眉** | 必须有控制号 + 页码 |
| 匿名 | 全文不得出现姓名/学校/赛区 | 不得有身份标志 | 只允许控制号，不得出现姓名/导师/学校 |
| 字体规定 | 全国规范**不统一规定**字体字号（赛区可另定） | **明确规定**：题目三号黑体、一级标题四号黑体居中、其他小四宋体、单倍行距 | 只规定字号 ≥12 pt |
| 参考文献 | 科技论文规范，正文用 `[n]` | 官方给定格式，**书籍须给页码** | 不限定格式，但网上资料与 AI 工具都要列 |
| AI 声明 | **必须排在参考文献之前**（2026 试行） | 规定的是**多处标注**（结果处、程序前），未规定统一声明的位置 | 正文之后附 **Report on Use of AI**，**不计入 25 页** |
| 代码/附录 | 附录须含支撑材料文件列表 + 全部完整可运行源程序 | 题目要求时须上传源程序备查；引用程序须注明来源 | 附录与代码**计入 25 页** |

三套模板都已在需要处把这些差异写成 `.tex` 注释，并标注依据年份。

---

## 四、各模板里你必须改的地方

### cumcm / yjs（中文）

1. **题目**：`\begin{center}{\zihao{3}\heiti 【论文题目…】}` 换成你的题目。
2. **摘要**：占位文字全部替换；**摘要必须有量化结果**（误差、最优值、提升比例、灵敏度结论），不能只写“建立了合理的模型”。
3. **所有 `【…】` 占位符**：一个都不要留，也不要写“略”。
4. **图/表里的占位数据**：`图 1` 的收敛曲线与 `表 1` 的数值只是排版演示，**必须换成你程序输出的真实数值**（论文数字与代码不符是红线）。
5. **参考文献**：`refs.bib` 里的条目是通用经典文献示例，**全部换成你自己读过的文献**；中文模板还给了手写 `thebibliography` 的写法注释（研赛要求书籍给出页码）。
6. **AI 工具使用声明**：未使用 / 已使用**二选一**，把另一种情况的整段删掉或注释掉。
7. **附录代码**：示例程序换成你的完整程序；推荐改用
   `\lstinputlisting[language=Python]{code/q1_solve.py}` 直接引入真实源文件，这样论文与代码不可能不一致。

### mcm（美赛）

1. **`\newcommand{\teamcontrol}{XXXXXXX}`** 换成你的控制号（页眉会自动更新到每一页）。
2. **Summary Sheet**：占位文字全部替换；**最后写、反复改**——评委很看重摘要，弱摘要基本定档。
3. 目录：模板里 `\tableofcontents` 是**注释掉的**；要目录就取消注释，但**目录页计入 25 页**。
4. **25 页上限包含附录与代码**，正文写完要留出附录的页数；`Report on Use of AI` 放在 25 页之后、**不计入**。
5. **Memo / Letter** 是**逐题面要求**：先读题面，要求写才写（模板末尾已留占位章节，不需要就整节删掉）。
6. `refs.bib` 里的 `webexample`、`aitool` 两条是**带占位符的示例**，用到才填，且必须填真实网址与访问日期。

---

## 五、常见编译错误与排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `! Font \... not loadable` / `Font ... not found` | 在没有该字体的系统上用了 `fontset=windows`（或反之） | 改成 `fontset=fandol`（Overleaf/Linux/macOS），Windows 用 `windows` |
| `! Package ctex Error: ...` 或中文不显示 | 用 pdfLaTeX 编了中文模板 | 换成 XeLaTeX（`latexmk -xelatex`） |
| `Unicode character ... not set up for use with LaTeX` | 在 pdfLaTeX 文档里放了中文正文 | 中文用 XeLaTeX；美赛模板保持全英文 |
| 正文引用显示 `[?]`、参考文献为空 | 没跑 `bibtex`，或正文里没有 `\cite{}` | 依次跑 引擎 → `bibtex main` → 引擎 ×2；正文补 `\cite` |
| BibTeX 报 `I found no \citation commands` | 同上（正文没有任何引用） | 补 `\cite{...}` 后再跑 |
| 交叉引用显示 `??`、页码“共 ? 页” | 编译遍数不够 | 再编译一到两遍 |
| `Package fancyhdr Warning: \headheight is too small` | 页眉文字比默认高度高 | 美赛模板已用 `geometry` 的 `headheight=15pt` 解决；自己改页眉时同样处理 |
| `Overfull \hbox ... too wide` | 某行太宽（多为长英文词、长公式、过宽的表格） | 不影响编译；用 `\sloppy`、`tabularx`、缩小字号或换行处理 |
| `Package pgfplots: compat` 相关提示 | `compat` 版本设置 | 模板已写 `\pgfplotsset{compat=1.18}`，可保留 |
| MiKTeX 首次编译卡住/提示装包 | 宏包按需下载 | 允许联网自动安装，或用 `mpm --install=<包名>` |
| Linux 报 `! LaTeX Error: File 'lmodern.sty' not found.` | 精简安装的 TeX Live 只装了 `texlive-*`，而 `lmodern.sty` 由发行版的**独立包**提供（本仓库 CI 第一次跑就是这么红的） | Debian/Ubuntu：`sudo apt install lmodern`；Fedora：`sudo dnf install texlive-lm`；或删掉美赛模板里的 `\usepackage{lmodern}` 改用默认 Computer Modern（不推荐，T1 编码下字形会变差） |
| 编译输出里出现 `security risk: running with elevated privileges` | MiKTeX 检测到以管理员权限运行（**本仓库验证时也会出现**） | 属提示性警告，不影响结果；本仓库三次编译的 exit code 均为 0 |
| 编译后目录里一堆 `.aux/.log/.out/.bbl/.pdf` | LaTeX 中间文件 | 正常；只提交最终的 `main.pdf`（美赛文件名必须是 `控制号.pdf`） |

**页数超标怎么办（美赛 25 页最容易踩）**

- 附录与代码计入 25 页 → 完整程序建议放支撑材料，附录里只留关键片段与复现说明。
- Summary Sheet 单独一页，不要溢出到第 2 页。
- 图建议导出为 PDF 矢量图（matplotlib `savefig("fig.pdf")`），比 PNG 更清晰也更省空间。

---

## 六、验证记录（本仓库的实际编译结果）

**你不用信这段文字——自己跑一条命令就能复现**：

```bash
# 仓库根目录下执行；本机需有 xelatex / pdflatex / bibtex
python scripts/check_latex.py                 # 自动把 fontset=windows 换成 fandol（CI 同款路径）
python scripts/check_latex.py --keep-fontset  # 不改字体，逐字验证仓库里这一份（需要 Windows 字体）
python scripts/check_latex.py --only cumcm    # 只编一个模板
python scripts/check_latex.py --self-test     # 不需要装 TeX，只测日志解析/字体替换/顺序核对
```

`check_latex.py` 会在系统临时目录里**另建一份副本**编译（**不在仓库内编译**，所以仓库里永远不出现 `.aux/.log/.pdf`），然后逐项核对：硬错误（`^!`）、未解析的 `\cite`/`\ref`、缺字体的 `Font "…" cannot be found`、BibTeX 致命错误、页数下限，以及**AI 声明与参考文献的先后顺序**（直接在 `.tex` 源码上核对，注释会被剥掉，避免把注释里提到的 `\bibliography` 误当成正文顺序）。

### 6.1 两种 `fontset` 下都编得过

页数/体积是**实际测得**的（Windows + MiKTeX 25.12，XeTeX / pdfTeX）：

| 模板 | 引擎 | `fontset=windows`（`--keep-fontset`） | `fontset=fandol`（默认 / CI） |
|---|---|---|---|
| cumcm | `xelatex` | **9 页**，A4（595×842 pt），202 842 B | **9 页**，341 465 B |
| yjs | `xelatex` | **8 页**，A4，212 439 B | **8 页**，366 745 B |
| mcm | `pdflatex` | **8 页**，letter（612×792 pt），284 317 B | 同左（美赛模板不含中文，与字体设置无关） |

两种配置都是 **3/3 通过**：0 硬错误、0 未定义引用、编译产物均超过页数下限。两列的页数一致说明换字体只改字形与嵌入体积，不改分页。CI（Ubuntu + TeX Live）上跑出来的页数同样是国赛 9 页、研赛 8 页、美赛 8 页，只是 PDF 体积随字体嵌入略有差异（国赛 ≈341 040 B、研赛 ≈366 285 B、美赛 ≈268 965 B）——**同一份源码在不同发行版上分页一致**，这才是"模板能跨机器编译"的真正含义。（PDF 体积每次编译会有几十字节的抖动，来自时间戳与 PDF ID；页数则完全稳定。）

> **`fandol` 替换是尽力而为，不是保证**：脚本只改**临时副本**里的那一行 `fontset=windows`（且只改代码部分、不动注释），仓库里的 `.tex` 一个字节都没动。真正的判据是**编译成功 + 日志里没有 `Font "…" cannot be found`**——所以在没有 `fandol` 的机器上这一步会失败，失败信息会直接打出缺哪个字体。

编译命令序列（三套一致，脚本内部就是这一串）：

```
引擎 → bibtex main → 引擎 → 引擎          # 4 遍，解析引用与交叉引用
```

`.github/workflows/ci.yml` 的 `latex` job 装 TeX Live 后跑 `python scripts/check_latex.py --require`：**`--require` 表示"没装 TeX 就报错退出"，绝不允许因为环境缺引擎就静默跳过**。这个 job 的意义在于：模板以前从来没在 CI 里被真正编译过，文档说"能编译"而模板悄悄烂掉，CI 仍然是绿的。

内容抽查（用 `pdftotext` / `pdffonts` / `pdfinfo` 核对，非人工目测截图）：

- 中文模板渲染顺序与规范一致：摘要页 → 问题重述 → 问题分析 → 模型假设 → 符号说明 → 模型的建立与求解 → 结果分析与检验 → 模型的评价与推广 → **AI 工具使用声明** → 参考文献 → 附录（AI 声明确实排在参考文献之前）。
- 中文模板 PDF 内嵌 SimSun / SimHei / KaiTi / FangSong，正文无缺字。
- 美赛模板每一页页眉都出现 `Team # XXXXXXX, Page X of 8`，第 1 页只有 Summary Sheet。

未解决的警告（均不影响输出）：

- cumcm / yjs：`LaTeX Warning: You have requested release '2026/06/01' of LaTeX, but only release '2025-11-01' is available.`（每遍 ×5）——来自 `ctexart.cls` 请求比本机更新的 LaTeX 内核，属 MiKTeX 版本提示，模板侧无法消除，不影响排版。
- cumcm / yjs 各 1 处 `Underfull \hbox`（附录尾部段落），纯排版提示。
- mcm：日志中**无任何警告**，无 undefined reference、无 `Missing character`。

---

## 七、相关文档

- 三赛事规则对照（含官方链接、AI 政策、页数与匿名要求）：`references/contests.md`
- 章节骨架与每章“必须出现什么”：`references/paper-structure.md`
- Markdown 版论文骨架（可与 LaTeX 模板对照填写）：`assets/paper-outline.md`
- 摘要模板与检查清单：`assets/abstract-template.md`
- 提交前自检：`references/checklists.md`、`scripts/check_paper.py`
