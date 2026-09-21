# 美赛 · MCM/ICM LaTeX 模板

> 竞赛：美国大学生数学建模竞赛（MCM/ICM，COMAP，俗称「美赛」）
> 文档类：`mcmthesis.cls` v6.3.3（2024/01/22）
> 引擎：**pdfLaTeX**（XeLaTeX 亦可）
> 授权：**LPPL 1.3c or later** ✅

## 快速开始

```bash
pdflatex -interaction=nonstopmode mcmthesis-demo.tex
pdflatex -interaction=nonstopmode mcmthesis-demo.tex
pdflatex -interaction=nonstopmode mcmthesis-demo.tex
```

编译产物为 `mcmthesis-demo.pdf`（示例 11 页）。

跑三遍是为了让交叉引用、`\AIcite` 标签与页眉的 `Page X of Y` 收敛
（参考文献写在正文的 `thebibliography` 环境里，**不需要 `bibtex`**）。
也可以用 `latexmk -pdf mcmthesis-demo.tex`（需要 Perl）。

Overleaf：把本目录**整体**上传，Compiler 选 **pdfLaTeX**。

> 本模板默认 `CTeX=false`，加载的是普通 `article` 类、**不含任何中文宏包**——
> 提交必须全英文，也不要在里面加 `ctex` / `xeCJK`。改 `\documentclass[CTeX=true]{mcmthesis}`
> 才会启用中文支持（需要 XeLaTeX）。

## 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `mcmthesis-demo.tex` | **主文件（入口 / 示例）**，照它写你的论文 |
| `mcmthesis.dtx` | 文档类**源码**（docstrip 的 literate source） |
| `mcmthesis.cls` | 文档类，由 `mcmthesis.dtx` 用 docstrip 生成 |
| `mcmthesis-demo.pdf` | 编译好的示例 |
| `mcmthesis.pdf` | 文档类使用手册（含全部选项说明） |
| `LICENSE-mcmthesis` | LPPL 1.3c or later |
| `code/mcmthesis-sudoku.cpp` | 示例代码清单 |
| `code/mcmthesis-matlab1.m` | 示例代码清单 |
| `figures/*` | 示例插图 |

## 重新生成文档类

`mcmthesis.cls` 和 `mcmthesis-demo.tex` 都是生成文件。改了 `mcmthesis.dtx`
之后需要重新生成——用一个 docstrip 驱动即可：

```latex
% mcmthesis.ins
\input docstrip.tex
\keepsilent
\askforoverwritefalse
\generate{%
  \file{mcmthesis.cls}{\from{mcmthesis.dtx}{class}}%
  \file{mcmthesis-demo.tex}{\from{mcmthesis.dtx}{demo}}%
}
\endbatchfile
```

```bash
pdflatex mcmthesis.ins
```

也可以直接 `xelatex mcmthesis.dtx`（会顺带排版出使用手册），但速度慢很多。

## LPPL 注意事项

本模板采用 **LaTeX Project Public License 1.3c or later**，使用时请注意两点：

1. **生成文件的分发条件**：LPPL 要求 `mcmthesis.cls` 的分发必须以
   「原始源文件属于同一分发」为前提。本仓库因此同时保留了 `mcmthesis.dtx`。
   如果你单独把 `mcmthesis.cls` 发给别人，请连 `.dtx` 一起给。
2. **修改必须改名**：LPPL 规定修改后的文件必须使用不同的文件名。
   如果你要定制 `mcmthesis.cls`，请另存为 `mythesis.cls` 之类再改，
   不要沿用 `mcmthesis.cls` 这个名字。

版权：Copyright © 2010–2015 Zhaoli Wang；2014–2019 Liam Huang；2019–2024 latexstudio。

## 溯源

来源：[latexstudio-org/mcmthesis](https://github.com/latexstudio-org/mcmthesis)
提交 `8ac05e2c3a9ef5880a15e3a3a18762a546c10b69`，
对应 CTAN 发行版 **6.3.3（2024-01-22）**。本目录下的全部文件与上游**逐字节一致，
本仓库未做任何改动**（它是三套里唯一有明确自由许可证的）。

详见 [`../THIRD-PARTY.md`](../THIRD-PARTY.md)。
