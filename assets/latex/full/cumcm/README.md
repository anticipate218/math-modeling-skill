# 国赛 · 全国大学生数学建模竞赛 LaTeX 模板

> 竞赛：全国大学生数学建模竞赛（CUMCM，俗称「国赛」）
> 文档类：`cumcmthesis.cls` v2.9（2026/08/26）
> 引擎：**XeLaTeX**

## 快速开始

```bash
xelatex -interaction=nonstopmode example.tex
xelatex -interaction=nonstopmode example.tex
xelatex -interaction=nonstopmode example.tex
```

编译产物为 `example.pdf`（示例 12 页）。

跑三遍是为了让交叉引用与页码收敛（参考文献写在正文的 `thebibliography` 环境里，
**不需要 `bibtex`**）。也可以用 `latexmk -xelatex example.tex`（需要 Perl）。

Overleaf：把本目录**整体**上传，Menu → Compiler 选 **XeLaTeX**。可以原样编译，
缺 Windows 字体时 `.cls` 会自动回落（见下）。

## 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `example.tex` | **主文件（入口）**，在这里写论文 |
| `cumcmthesis.cls` | 文档类 v2.9，定义承诺书、编号页、封面、摘要页与正文版式 |
| `cumcm2026.sty` | 2026 年格式的补充宏包（**必须**和主文件在一起） |
| `example.pdf` | 编译好的示例 |
| `figures/*` | 示例插图 |

## 本仓库对上游的改动（只有字体加载一处）

`cumcmthesis.cls` 里原来是无条件的：

```latex
\setmainfont{Times New Roman}
%\setmonofont{Courier New}
\setsansfont{Arial}
```

在没有任何这些字体的机器（Overleaf / Linux / macOS）上会直接
`! Package fontspec Error: The font "Times New Roman" cannot be found`。
现改为存在性判断 + 回落：

```latex
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
%\setmonofont{Courier New}
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{\setsansfont{TeX Gyre Heros}}
```

`TeX Gyre Termes / Heros` 是 Times / Arial 的**度量兼容克隆**，随 TeX Live / MiKTeX
分发，任何平台都有；**换字体只改字形与 PDF 嵌入体积，不改分页**——两种字体下
都是 12 页（有 Windows 字体 452 165 B，回落 TeX Gyre 后 538 970 B）。
`\setmonofont` 上游本来就是注释掉的，保持原样未动。

`example.tex`、`cumcm2026.sty`、`figures/*` 与上游**逐字节一致**。

## 关于 2026 年的格式要求

`example.tex` 第 4–6 行：

```latex
%\documentclass{cumcmthesis}

\documentclass[withoutpreface,bwprint]{cumcmthesis} %去掉封面与编号页，电子版提交的时候使用。
```

- `withoutpreface` —— 不生成封面和编号页（**电子版提交用这个**，2026 年上传要求）
- `bwprint` —— 黑白打印模式（用 `colorprint` 则保留彩色）
- 想要带封面和编号页的完整版，就改成 `\documentclass{cumcmthesis}`

参考：<https://cumcm.cnki.net/cumcm/studentHome/noticeDetail?id=c0c5220f-0226-4cd8-bce1-e1f81e1e23f6>

## 溯源与授权

来源：[latexstudio/CUMCMThesis](https://github.com/latexstudio/CUMCMThesis)
提交 `38d1f216bec3c9ffffb7dd09bf6b6c54f486b130`。

⚠️ 上游仓库**没有 LICENSE 文件**，`.cls` 内也没有授权声明，该宏包**未收录于 CTAN**。
本仓库按「注明出处、原样收录」处理，**不做任何再许可**。
`cumcmthesis.cls` / `cumcm2026.sty` / `example.tex` 与上游逐字节一致
（唯一改动是上面那一处西文字体回落）。

如需广泛分发，建议先向上游请求补充许可证。详见 [`../THIRD-PARTY.md`](../THIRD-PARTY.md)。
