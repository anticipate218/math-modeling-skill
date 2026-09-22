# 完整文档类模板（华为杯 2026 / 国赛 / 华为杯·研赛 / 美赛）

这里是四套**直接拿去投稿的完整 LaTeX 模板**，每套都是一个**独立可编译的目录**
（文档类 `.cls`、补充宏包 `.sty`、`figures/` 插图、示例主文件都在里面）。

| 目录 | 赛事 | 入口主文件 | 文档类 | 引擎 | 示例成品 |
|---|---|---|---|---|---|
| `hwcup2026/` | 2026 华为杯（第二十三届中国研究生数学建模竞赛）**严格格式版** | `main.tex` | `hwcup2026.cls`（本仓库自制） | **XeLaTeX** | 3 页 |
| `gmcm/` | 中国研究生数学建模竞赛（**华为杯** / 研赛） | `MathModel.tex` | `gmcmthesis.cls` v2.2 | **XeLaTeX** | 8 页 |
| `cumcm/` | 全国大学生数学建模竞赛（**国赛**） | `example.tex` | `cumcmthesis.cls` v2.9（2026/08/26） | **XeLaTeX** | 12 页 |
| `mcm/` | 美赛 MCM / ICM（COMAP） | `mcmthesis-demo.tex` | `mcmthesis.cls` v6.3.3（2024/01/22） | **pdfLaTeX** | 11 页 |

> **`hwcup2026/` 和 `gmcm/` 都是华为杯，选哪个？**
> 投 **2026 年** 研赛用 **`hwcup2026/`**——它逐条复刻了 2026-09-16 官方《论文格式规范》
> 与附件3 Word 模板（页边距、字号、编号、封面都对齐），封面直接取官方附件3 渲染图。
> `gmcm/` 是社区沿用的 `gmcmthesis.cls` 通用排版版，**两套并存、互不替代**；
> `gmcm/` 保留下来是因为它的章节/图表/算法环境自定义更全，适合参考写法。
> 详细对照见 [`hwcup2026/README.md`](hwcup2026/README.md)。

> **和上一层的 `assets/latex/{cumcm,yjs,mcm}/main.tex` 什么关系？**
> 上一层的三份是**本仓库自写的精简模板**：一个 `main.tex` + 一个 `refs.bib`，
> 没有文档类、没有 `.cls` 依赖，拿来当"章节骨架 + 合规检查"的样板最省事，
> 也可以直接喂给 `scripts/check_paper.py` 做提交前自检。
> 这一层的四份是**完整的文档类版本**，排版与官方样张一致（封面/页眉/
> 字号/编号页都由 `.cls` 管），**正式投稿建议用这一层**。
> 两层都保留：精简版负责"结构与合规"，完整版负责"成品排版"。

---

## 一、怎么用

### 方式 A：从 Release 下载打包好的 ZIP（最省事）

到本仓库的 [Releases](https://github.com/anticipate218/math-modeling-skill/releases) 页下载对应赛事的压缩包，解压后直接编译：

| 压缩包 | 解压出的目录 | 大小 |
|---|---|---|
| `hwcup2026-template.zip` | `hwcup2026/` | ≈ 1.2 MB（封面底图占大头） |
| `gmcm-template.zip` | `gmcm/` | ≈ 24 MB（含 5 个中文字体） |
| `cumcm-template.zip` | `cumcm/` | ≈ 0.9 MB |
| `mcm-template.zip` | `mcm/` | ≈ 0.7 MB |

> ⚠️ 这些 ZIP **不是**技能安装包。技能安装包（名字以 `math-modeling-skill` 开头）
> 请按 [`INSTALL.md`](../../../INSTALL.md) 里的方式装，不要混用。

解压后**进入该目录**再编译（`figures/`、`assets/` 与字体都是相对主文件查找的）：

```bash
cd hwcup2026
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

```bash
cd gmcm
xelatex -interaction=nonstopmode MathModel.tex
xelatex -interaction=nonstopmode MathModel.tex
xelatex -interaction=nonstopmode MathModel.tex
```

```bash
cd cumcm
xelatex -interaction=nonstopmode example.tex
xelatex -interaction=nonstopmode example.tex
xelatex -interaction=nonstopmode example.tex
```

```bash
cd mcm
pdflatex -interaction=nonstopmode mcmthesis-demo.tex
pdflatex -interaction=nonstopmode mcmthesis-demo.tex
pdflatex -interaction=nonstopmode mcmthesis-demo.tex
```

**为什么正好三遍？** 四套模板的参考文献都写在正文里的 `thebibliography` 环境里
（行内条目），**不需要跑 `bibtex`**；但交叉引用和页码（美赛页眉的 `Page X of Y`）
要跑满三遍才收敛。想省事也可以用 `latexmk -xelatex main.tex`（需要 Perl）。

### 方式 B：直接把目录拷进你的工作目录

每套目录都是自包含的。**整目录**拷贝即可，注意三点：

1. **不要只拷主文件**。`cumcmthesis.cls` / `gmcmthesis.cls` / `mcmthesis.cls` /
   `hwcup2026.cls` 必须在主文件同一层；`cumcm2026.sty` 也是。
2. **研赛 `gmcm/` 的 5 个 `.ttf` 必须留在主文件同一层**（不要挪进 `fonts/` 之类
   子目录），原因见下面第 3 节。
3. **`hwcup2026/` 的 `assets/` 子目录要一起拷**（封面底图与 4 个固定标签图），
   缺了封面页就出不来。

### 方式 C：在 Overleaf 上编译

1. 新建项目 → 上传**整个目录**（Overleaf 支持拖拽文件夹，或用 “New Project →
   Upload Project” 上传 ZIP）。
2. `Menu → Compiler`：华为杯两台模板与国赛选 **XeLaTeX**，美赛选 **pdfLaTeX**。
3. `gmcm/`、`cumcm/` 可以**原样编译**——`.cls` 已经做了字体回落，缺 Windows 字体时
   自动改用 TeX 发行版自带的字体（见第 3 节）。
4. `hwcup2026/` 在 Overleaf 上**能编过**，但会走 Noto CJK 回落（Overleaf 的
   `fonts-noto-cjk` 一般已装），字形与 Windows 的宋体/黑体略有差别。**最终提交版
   仍建议在 Windows 上编**，理由见第 3 节。

### 需要哪些 TeX 组件

用到的宏包都是 TeX Live 的标准件，**完整安装**（TeX Live full / MiKTeX / Overleaf）
一律自带，不用管。容易踩的只有一种情况：**在 Debian / Ubuntu 上手装 `texlive-*`**，
因为有几个文件不在 `texlive-latex-base|recommended|extra` 里，得单独点名——
本仓库 CI 就把这几个"漏了就红"的包钉死在安装步骤里了：

| 缺的文件 | Ubuntu 包名 | 谁在用 |
|---|---|---|
| `lmodern.sty` | `lmodern`（Debian 顶层包，不在任何 `texlive-*` 里） | 美赛 |
| `ulem.sty` | `texlive-plain-generic`（TDS 路径是 `tex/generic/`，所以 `latex-*` 包里没有） | 研赛、国赛（队员名单下划线） |
| `berasans.sty` | `texlive-fonts-extra` | 美赛（`\RequirePackage[scaled]{berasans}`） |
| `texgyretermes-regular.otf` 等 | `fonts-texgyre`（**不是** `texlive-` 前缀） | 研赛、国赛的西文回落字体 |
| `ts1-qtmr.tfm` 等 TeX Gyre 度量 | `tex-gyre`（**和上面是两个互不相干的包**） | 美赛（`newtxtext` 的 TS1 映射） |
| `Noto Serif/Sans CJK SC` | `fonts-noto-cjk`（**不是** `texlive-` 前缀） | **华为杯 2026** 的中文回落字体 |
| `Liberation Serif` | `fonts-liberation` | **华为杯 2026** 的西文回落字体 |

外加 `texlive-lang-chinese`（提供 `ctexart.cls` 与 fandol 字体）、`texlive-xetex`、
`texlive-pictures`、`texlive-science`、`texlive-bibtex-extra`。

---

## 二、本仓库对上游做了什么改动

`cumcm` / `gmcm` / `mcm` 三套**逐字节保留了上游的排版逻辑**，只改了字体加载，
目的是让它们在**没有 Windows 字体的机器（Overleaf / Linux / macOS）上也能编过**。
改动一共两处，都写在 `.cls` 的注释里：

### 1. `gmcm/gmcmthesis.cls` —— 修一个必然触发的字体 bug

上游有一段隶书字体的条件判断：

```latex
\ifx\lishu\undefined
 \setCJKfamilyfont{zhli}{LiSu.ttf}
 \newcommand*{\lishu}{\CJKfamily{zhli}}
\else
\fi
```

`ctex` 宏包**早就定义过 `\lishu`**，所以这个条件**永远不成立**，随包的 `LiSu.ttf`
从未被注册；编译到摘要标题那一行必然报错退出：

```
! Package fontspec Error: The font "LiSu" cannot be found
```

已改成「先 `\providecommand*` 兜底，再用 `\renewcommand*` 绑定到随包字体」，
并把同样的处理套用到 `\xinwei`。

### 2. 两处西文字体改成「有就用、没有就回落」

上游这两处是**无条件**的，在 Overleaf 上直接编译失败
（`! Package fontspec Error: The font "Times New Roman" cannot be found`）：

| 文件 | 上游写法 | 现在 |
|---|---|---|
| `gmcm/gmcmthesis.cls` | `\setmainfont{Times New Roman}` 等三行 | `\IfFontExistsTF{...}` + 回落 `TeX Gyre Termes / Heros / Cursor` |
| `cumcm/cumcmthesis.cls` | `\setmainfont{Times New Roman}`、`\setsansfont{Arial}` | `\IfFontExistsTF{...}` + 回落 `TeX Gyre Termes / Heros` |

`TeX Gyre` 系列是 Times / Arial / Courier 的**度量兼容克隆**（同样是 URW 字体的
开源版本），随 TeX Live 与 MiKTeX 一起分发，任何平台都有。**换字体只改字形与
PDF 嵌入体积，不改分页**——两套模板在两种字体下页数完全一致（见第 5 节实测表）。

**除这两处外没有任何改动。** `example.tex`、`mcmthesis-demo.tex`、`mcmthesis.dtx`、
`cumcm2026.sty`、`figures/*` 均与上游一致。

### 3. `hwcup2026/` 不是"改上游"，是自制的严格复刻

`hwcup2026.cls` 是**本仓库作者新写的**文档类（`\LoadClass{ctexart}` 为基础），
不是从任何上游 `.cls` 改来的，所以上面那两条改动都不适用。它的字体策略是
**双回落**：

| 目标字体 | 有就用 | 没有就回落 |
|---|---|---|
| 中文 | `SimSun` / `SimHei`（Windows 宋体 / 黑体） | `Noto Serif CJK SC` / `Noto Sans CJK SC` |
| 西文 | `Times New Roman` | `Liberation Serif` |

---

## 三、字体：随包携带的 5 个 `.ttf`（只属于 `gmcm/`）

`gmcm/` 目录带着 5 个 Windows 中文字体，总共约 44 MB：

| 文件 | 用途 | 大小 |
|---|---|---|
| `SimSun.ttf` | 宋体 —— CJK 主字体 | 10.0 MB |
| `SimHei.ttf` | 黑体 | 9.6 MB |
| `KaiTi.ttf` | 楷体 —— 斜体 | 11.2 MB |
| `LiSu.ttf` | 隶书 —— 封面「摘 要」 | 8.8 MB |
| `STXinwei.ttf` | 华文新魏 —— `\xinwei` 命令 | 4.1 MB |

**为什么要随包带？** 为了让字形和 Word 里看到的**完全一致**（这也是研赛官方样张
的观感）。这些字体装在系统里也能用，但很多机器上并没有全部 5 个。

**可以删掉吗？可以。** 删掉（或只删一部分）之后 `gmcmthesis.cls` 会用
`\IfFontExistsTF` 检测到文件不在，自动回落到 `ctex` 的字体集——Windows 上用系统
自带字体，Linux / Overleaf 上用随 TeX 发行版分发的自由字体 **fandol**。
**照样编得过，只是字形会变**（不再和 Word 一致）。

> ⚠️ **不要把这 5 个文件挪进子目录。** 文档类是按**裸文件名**引用它们的
> （`\setCJKmainfont{SimSun.ttf}`），只会去主文件所在目录找。挪进 `fonts/`
> 之后 `fontspec` **只警告不报错**：
> `The font "SimSun" cannot be found`，然后整篇中文被悄悄排成西文字体，
> **一个字都印不出来**（实测正文丢 1503 个字形），而编译照样退出码 0、PDF 照样
> 8 页——非常隐蔽。所以：**留在原地，或者直接删掉。**

### `hwcup2026/` 为什么不带字体？

它**一个 `.ttf` 都不带**，靠 `\IfFontExistsTF` 现场探测：

- Windows 上有 SimSun / SimHei / Times New Roman → 用它，**这就是 2026 官方
  《论文格式规范》点名的宋体/黑体**，也是官方 Word 附件3 用的字体。
- 没有 → 回落到 Noto Serif/Sans CJK SC + Liberation Serif，**照样编得过**。

封面页、4 个 Logo、摘要页顶部的赛事标题，以及「题目 / 摘要 / 关键词」这些**固定
标签**，全都直接取自官方附件3 的渲染图（`assets/*.png`），所以**不受**
STXinwei / LiSu 这类字体缺失的影响——那几个字不是排出来的，是贴上去的。

> **最终提交版请在 Windows 上编。** 回落分支只是为了让你在 Overleaf / Linux 上
> 也能先看到版式；Noto CJK 的笔画粗细与宋体不完全一致，严格对齐官方样张还是
> 用 SimSun / SimHei。

### 授权提示

`gmcm/` 里那 5 个 `.ttf` 是 Windows 系统自带的**商用**中文字体（版权归中易 ZhongYi、
华文 SinoType 等），**不是自由字体**。随包携带只为让你零配置编译、字形与 Word 一致。
如需再分发或商用，请自行确认授权；或者删掉它们改用 `fandol` 自由字体集。

`hwcup2026/` **不含任何字体文件**，不存在这个问题。

---

## 四、目录里都是什么

<details>
<summary><b>hwcup2026/</b>（2026 华为杯严格格式版，9 个文件）</summary>

| 文件 | 说明 |
|---|---|
| `main.tex` | **主文件（入口）**，论文写在这里 |
| `hwcup2026.cls` | 文档类，严格复刻 2026 官方附件3 的版面尺寸与字号 |
| `preview.pdf` | 作者预编译的 3 页样张 |
| `assets/official-cover.png` | 官方附件3 第 1 页渲染图（封面底图） |
| `assets/abstract-header.png` | 摘要页顶部赛事标题渲染图 |
| `assets/label-title.png` / `label-abstract.png` / `label-keywords.png` | 三个固定标签渲染图 |
| `README.md` | 该模板的独立说明（含 2026 官方硬性要求逐条清单） |

</details>

<details>
<summary><b>gmcm/</b>（研赛·华为杯，12 个文件）</summary>

| 文件 | 说明 |
|---|---|
| `MathModel.tex` | **主文件（入口）**，论文写在这里 |
| `gmcmthesis.cls` | 文档类，定义封面、页眉页脚、字号 |
| `MathModel.pdf` | 作者预编译的样张（可直接看排版效果） |
| `SimSun.ttf` / `SimHei.ttf` / `KaiTi.ttf` / `LiSu.ttf` / `STXinwei.ttf` | 随包中文字体 |
| `test.jpg` | 示例插图 |
| `figures/title2025.pdf` | 封面标题图 |
| `figures/logo2025.png` | 封面 logo |
| `README.md` | 该模板的独立说明 |

</details>

<details>
<summary><b>cumcm/</b>（国赛，11 个文件）</summary>

| 文件 | 说明 |
|---|---|
| `example.tex` | **主文件（入口）** |
| `cumcmthesis.cls` | 文档类 v2.9，承诺书 / 编号页 / 封面 / 摘要页 / 正文版式 |
| `cumcm2026.sty` | 2026 年格式的补充宏包（**必须**与主文件在一起） |
| `example.pdf` | 作者预编译的样张 |
| `figures/*` | 示例插图（6 个） |
| `README.md` | 该模板的独立说明 |

</details>

<details>
<summary><b>mcm/</b>（美赛，15 个文件）</summary>

| 文件 | 说明 |
|---|---|
| `mcmthesis-demo.tex` | **主文件（入口 / 示例）** |
| `mcmthesis.cls` | 文档类 v6.3.3 |
| `mcmthesis.dtx` | 文档类**源码**（docstrip literate source） |
| `mcmthesis.ins` | docstrip 驱动，用来从 `.dtx` 重新生成 `.cls` / demo |
| `mcmthesis.pdf` | 文档类使用手册（含全部选项） |
| `mcmthesis-demo.pdf` | 作者预编译的样张 |
| `LICENSE-mcmthesis` | LPPL 1.3c or later |
| `code/*` | 示例代码清单（`.m` / `.cpp`） |
| `figures/*` | 示例插图（5 个） |
| `README.md` | 该模板的独立说明 |

</details>

---

## 五、验证记录（本仓库的实际编译结果）

**自己跑一条命令就能复现**（本机需有 `xelatex` / `pdflatex`）：

```bash
# 仓库根目录下执行
python scripts/check_latex_full.py                    # 编全部：模板 ×（原样 + 无 Windows 字体）
python scripts/check_latex_full.py --only hwcup2026   # 只编 2026 华为杯
python scripts/check_latex_full.py --only gmcm        # 只编研赛
python scripts/check_latex_full.py --no-simulate      # 只按原样编，不跑回落路径
python scripts/check_latex_full.py --keep             # 保留临时目录，方便翻 .log / .pdf
python scripts/check_latex_full.py --self-test        # 不需要装 TeX，只测改写逻辑
```

脚本把**整个模板目录**拷进系统临时目录再编译（**不在仓库内编译**，仓库里永远
不出现 `.aux/.log/.pdf`），并**先删掉预编译的样张 PDF**——否则编译失败时旧 PDF
还在，体检就被骗过去了。然后逐项核对硬错误（`^!`）、未解析引用、
`Font "…" cannot be found`、页数下限，以及 cumcm/mcm 的「AI 声明 vs 参考文献」
合规顺序。

### 5.1 两条字体路径都编得过

实测环境：**Windows + MiKTeX 25.12**（XeTeX / pdfTeX），引擎各跑 3 遍。

| 模板 | 引擎 | 原样（系统 Windows 字体） | 模拟无 Windows 字体 |
|---|---|---|---|
| `hwcup2026` | `xelatex` | **3 页**，662 371 B | **3 页**，652 715 B（Noto CJK + Liberation） |
| `gmcm` | `xelatex` | **8 页**，395 954 B | **8 页**，391 121 B（fandol + TeX Gyre） |
| `cumcm` | `xelatex` | **12 页**，452 166 B | **12 页**，538 968 B（fandol + TeX Gyre） |
| `mcm` | `pdflatex` | **11 页**，279 394 B | 不适用（不含中文，不调 Windows 字体） |

**7/7 通过**（`hwcup2026` 与 `gmcm` 各两遍、`cumcm` 两遍、`mcm` 一遍）：每次都是
退出码 0、0 条硬错误、0 个缺字形、0 处未解析引用。同一模板两列**页数完全一致**——
换字体只改字形与嵌入体积，不改分页。

> 「模拟无 Windows 字体」不是靠"在这台 Windows 机器上碰运气"：脚本会把临时副本
> 里的 ctex 字体集**钉成 `fandol`**、把西文字体探测的名字**换成一定不存在的名字**
> （`\IfFontExistsTF{Times New Roman}` → `\IfFontExistsTF{NoSuchWindowsFontZZZ}`，
> 中文探测 `SimSun` / `SimHei` 同样处理），并把随包 `.ttf` 删掉——所以在任何平台上
> 这一遍走的都是真实的回落分支。
>
> 两个佐证：`cumcm` 两列体积差了 86 802 B，正是 TeX Gyre 替掉 Times New Roman /
> Arial 之后嵌入字体变大的量；`hwcup2026` 两列只差 9 656 B，因为 Noto CJK 与
> SimSun / SimHei 的字形集相近。

### 5.2 编译产物与随仓库提交的样张一致

| 模板 | 本次编译 | 仓库里的 `*.pdf` | |
|---|---|---|---|
| `gmcm` | 395 954 B | `MathModel.pdf` 395 954 B | ✅ 逐字节相同 |
| `cumcm` | 452 166 B | `example.pdf` 452 166 B | ⚠️ 差 1 字节 |
| `mcm` | 279 394 B | `mcmthesis-demo.pdf` 279 394 B | ✅ 逐字节相同 |

说明仓库里那两份样张就是这些源码在当前工具链下真实编出来的，不是别处拷来的。
`hwcup2026/preview.pdf` 用的是作者自己那台机器（引擎版本与页数一致，都是 3 页），
本仓库没有宣称它与这里的重编译结果逐字节相同。

`cumcm` 这 1 字节必须说清楚，**不能宣称逐字节相同**：差异出现在文件末尾——前
444 862 字节完全一致（占全文件 98.4%），不同的只有最后一个 Flate 压缩对象流
（`/Length 3866` vs `3867`，装的是 XMP / Info 元数据，也就是 PDF 文档 ID 与生成
时间戳）。**版面内容没有任何差别**：都是 12 页、同样的字体嵌入、同样的交叉引用。
`gmcm` / `mcm` 两份则是真正的逐字节相同。

### 5.3 CI

`.github/workflows/ci.yml` 的 `latex` 任务在装好 TeX Live 后跑
`python scripts/check_latex_full.py --require`（`--require` = 没装 TeX 就报错退出，
不允许静默跳过）。Ubuntu 上没有 Windows 字体，所以那一遍**天然就是回落路径**；
为了让 `hwcup2026` 的回落分支也有字体可用，安装步骤里点名了 `fonts-noto-cjk`
与 `fonts-liberation`，并用 `fc-list` 在安装阶段就把这两个家族钉住（缺了立刻失败，
不必等编译日志）。

### 5.4 已知的无害警告

这些警告**不影响输出**，上游就有，本仓库没有"顺手修"（改了就和上游不一致了）：

| 模板 | 警告 | 说明 |
|---|---|---|
| `hwcup2026` | `LaTeX Warning: You have requested release '2026/06/01' of LaTeX`（每遍 ×4） | `ctexart` 请求比本机更新的 LaTeX 内核，属 MiKTeX 版本提示。**无 Overfull / Underfull** |
| `gmcm` | `(\end occurred when \iftrue on line 16 was incomplete)` | `gmcmthesis.cls` 里 `\newif\if@gmcm@preface` 之后没有配对的 `\fi`，`\iftrue` 一直悬着。TeX 只在文件尾报告一次，不影响排版 |
| `gmcm` | `! ClassError{mcmthesis}` 里写错类名 | 该文件是从美赛模板改来的，报错信息里的类名忘了改。只有你用错引擎时才会看到这句话 |
| `gmcm` | 1 处 `Overfull \hbox`、1 处 `Underfull \hbox` | 纯排版提示 |
| `cumcm` | 1 处 `Overfull \hbox`；标签重复定义 | `example.tex` 里教学用的代码样例重复贴了 `\label` / `\bibitem`，产生 `multiply-defined labels` 警告 |
| `mcm` | 1 处 `Underfull \hbox`；`\AIcite` 相关 | 日志中无 undefined reference、无 `Missing character` |

---

## 六、本源与授权

**逐套的溯源信息、上游提交号、许可证状态见 [`THIRD-PARTY.md`](THIRD-PARTY.md)**，
下面是速查：

| 模板 | 上游 | 许可证 |
|---|---|---|
| `hwcup2026` | **本仓库作者自制**（依据 2026-09-16 官方《论文格式规范》与附件3 Word 模板逐条复刻；封面/标签取自官方附件3 渲染图） | 自制部分按本仓库许可；**官方附件3 渲染图版权属竞赛组委会** ⚠️ |
| `gmcm` | **本仓库作者自制**（参考谱系：`springli07/GMCM_LaTeX_overleaf` → `zhanwen/MathModel`） | 上游两处均**未附 LICENSE**；本仓库按「注明来源、原样收录」处理 |
| `cumcm` | [`latexstudio/CUMCMThesis`](https://github.com/latexstudio/CUMCMThesis) @ `38d1f21` | 上游**无 LICENSE**、未收录 CTAN ⚠️ |
| `mcm` | [`latexstudio-org/mcmthesis`](https://github.com/latexstudio-org/mcmthesis) @ `8ac05e2` | **LPPL 1.3c or later** ✅（已随包保留 `mcmthesis.dtx`） |

**美赛模板的 LPPL 两点要求**（用它定制时请注意）：

1. **生成文件的分发条件**：LPPL 要求 `mcmthesis.cls` 的分发以「原始源文件属于
   同一分发」为前提——所以这里同时保留了 `mcmthesis.dtx`。单独把 `.cls` 发给
   别人时，请连 `.dtx` 一起给。
2. **修改必须改名**：LPPL 规定修改后的文件必须用不同文件名。要定制
   `mcmthesis.cls`，请另存为 `mythesis.cls` 之类再改。

---

## 七、相关文档

- 精简版模板（自写、含合规检查）：[`assets/latex/README.md`](../README.md)
- 2026 华为杯严格格式版逐条要求：[`hwcup2026/README.md`](hwcup2026/README.md)
- 三赛事规则对照（官方链接、AI 政策、页数与匿名要求）：`references/contests.md`
- 提交前自检：`references/checklists.md`、`scripts/check_paper.py`
- 完整模板的真编译体检：`scripts/check_latex_full.py`
