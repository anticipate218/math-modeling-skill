# 华为杯 · 中国研究生数学建模竞赛 LaTeX 模板

> 竞赛：中国研究生数学建模竞赛（研赛，俗称「华为杯」）
> 文档类：`gmcmthesis.cls` v2.2
> 引擎：**XeLaTeX**（必须）
> 来源：本仓库作者自制（参考谱系与授权状态见 [`../THIRD-PARTY.md`](../THIRD-PARTY.md)）

## 快速开始

```bash
xelatex -interaction=nonstopmode MathModel.tex
xelatex -interaction=nonstopmode MathModel.tex
xelatex -interaction=nonstopmode MathModel.tex
```

编译产物为 `MathModel.pdf`（示例 8 页）。封面/报名信息在 `MathModel.tex` 开头填写。

跑三遍是为了让交叉引用与页码收敛（参考文献写在正文的 `thebibliography` 环境里，
**不需要 `bibtex`**）。也可以用 `latexmk -xelatex MathModel.tex`（需要 Perl）。

Overleaf：把本目录**整体**上传，Menu → Compiler 选 **XeLaTeX**。可以原样编译，
缺 Windows 字体时 `.cls` 会自动回落（见下）。

## 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `MathModel.tex` | **主文件（入口）**，在这里写论文 |
| `gmcmthesis.cls` | 文档类，定义封面、页眉页脚、字号等 |
| `MathModel.pdf` | 编译好的示例，可直接看排版效果 |
| `SimSun.ttf` | 宋体 —— 正文 CJK 主字体 |
| `SimHei.ttf` | 黑体 —— 加粗用 |
| `KaiTi.ttf` | 楷体 —— 斜体用 |
| `LiSu.ttf` | 隶书 —— 封面「摘要」标题用 |
| `STXinwei.ttf` | 华文新魏 —— `\xinwei` 命令用 |
| `test.jpg` | 示例插图 |
| `figures/title2025.pdf` | 封面标题图 |
| `figures/logo2025.png` | 封面 logo |

## ⚠️ 必须用 XeLaTeX

`gmcmthesis.cls` 第 34 行有 `\RequireXeTeX`，用 `pdflatex` 会直接报错退出。

## 字体：可以带，也可以删

**五个 `.ttf` 随包携带**，为的是字形和 Word 里看到的完全一致（也是研赛官方样张的
观感）。它们**不是必需的**——`gmcmthesis.cls` 用 `\IfFontExistsTF` 逐个探测，
文件不在就回落到 `ctex` 的字体集（Windows → 系统自带字体；Linux / Overleaf →
随 TeX 发行版分发的自由字体 `fandol`）。**删掉照样编得过，只是字形会变。**

### ⚠️ 不要把它们挪进子目录

文档类是按**裸文件名**引用字体的（`\setCJKmainfont[...]{SimSun.ttf}`），只会去
主文件所在目录查找。一旦把字体挪进子目录（比如 `fonts/`），`fontspec`
**只会警告不报错**：

```
The font "SimSun" cannot be found ...
```

然后整篇文档的中文会被悄悄排成西文字体，一个字都印不出来（实测正文会丢失
**1503 个字形**，而编译仍然「成功」退出，PDF 也有 8 页——非常隐蔽）。
所以：**要么留在主文件同一层，要么直接删掉。**

### 授权提示

这五个 `.ttf` 是 Windows 系统自带的**商用**中文字体（版权归中易 ZhongYi /
华文 SinoType 等），**不是自由字体**。随包携带只为你零配置编译方便。
如需再分发或商用，请自行确认授权，或删除它们并改用 `ctex` 的
`fontset=fandol` 自由字体集。

## 本仓库对上游的修正

`gmcmthesis.cls` 中原本有一段隶书字体的条件判断：

```latex
\ifx\lishu\undefined
 \setCJKfamilyfont{zhli}{LiSu.ttf}
 \newcommand*{\lishu}{\CJKfamily{zhli}}
\else
\fi
```

因为 `ctex` 早就定义过 `\lishu`，条件永远不成立，随包的 `LiSu.ttf` 从未被注册，
编译到第 367 行的摘要标题时必然报错：

```
! Package fontspec Error: The font "LiSu" cannot be found
```

已改为：

```latex
\providecommand*{\lishu}{}
\IfFontExistsTF{LiSu.ttf}{\renewcommand*{\lishu}{\CJKfamily{zhli}}}{}
```

`\xinwei` 同样处理。

英文字体也由无条件的三行改成了可回落：

```latex
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
\IfFontExistsTF{Courier New}{\setmonofont{Courier New}}{\setmonofont{TeX Gyre Cursor}}
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{\setsansfont{TeX Gyre Heros}}
```

`TeX Gyre` 系列是 Times / Arial / Courier 的度量兼容克隆，随 TeX Live / MiKTeX
分发，任何平台都有；**换字体只改字形，不改分页**（两种字体下都是 8 页）。

修正后连编 3 遍全部退出码 0、0 条硬错误、0 个丢失字形（带随包字体 395 954 B，
走 `fandol` + TeX Gyre 回落路径 391 120 B）。完整的溯源与授权信息见
[`../THIRD-PARTY.md`](../THIRD-PARTY.md)。

## 已知的无害警告

- `(\end occurred when \iftrue on line 16 was incomplete)` —— `\newif\if@gmcm@preface`
  之后没有配对的 `\fi`，`\iftrue` 一直悬着。TeX 只在文件尾报告一次，不影响排版。
- 报错信息里 `! ClassError{mcmthesis}` 写错了类名（该文件从美赛模板改来，漏改了）。
  只有你用错引擎时才会看到这句话。
- 1 处 `Overfull \hbox`、1 处 `Underfull \hbox`。
