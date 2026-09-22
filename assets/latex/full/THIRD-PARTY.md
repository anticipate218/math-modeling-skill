# 第三方模板溯源与授权

`assets/latex/full/` 下四套完整 LaTeX 模板的来源、上游提交号与授权状态。
**本仓库对这些模板不做任何再许可**；许可证以各上游为准，下面逐条写明。

---

## 1. `hwcup2026/` —— 2026 华为杯严格格式版

| 项目 | 内容 |
|---|---|
| 性质 | **本仓库作者自制**（2026 年新写，不是从任何上游 `.cls` 改来的） |
| 文档类 | `hwcup2026.cls`（本仓库原创） |
| 依据 | 2026-09-16 官方《论文格式规范》 + 官方**附件3 Word 模板** |
| 收录文件 | `hwcup2026.cls`、`main.tex`、`preview.pdf`、`assets/*.png`（5 张） |
| 上游许可证 | 无上游——自制部分按本仓库 [`LICENSE`](../../../LICENSE)（MIT）覆盖 |
| ⚠️ 例外 | `assets/official-cover.png`、`assets/abstract-header.png`、`assets/label-*.png` 是**官方附件3 的渲染结果**，版权属竞赛组委会，**不在 MIT 覆盖范围内** |

### 它是什么

2026 年华为杯官方要求「必须按附件3 模板编写」，而组委会**没有发布官方 LaTeX 模板**。
这套 `.cls` 逐条复刻了附件3 的版面尺寸与字号：

1. A4 纵向；页边距按附件3 Word 内部设置：上 30.02 mm、下 18.49 mm、左 22.51 mm、右 22.47 mm。
2. 首页保留官方附件3 封皮及 4 个 Logo，不替换。
3. 摘要页起用阿拉伯数字从 1 连续编号，页码居中于页脚。
4. 不设页眉。
5. 论文题目三号黑体；一级标题四号黑体居中；其余汉字小四宋体；单倍行距。
6. 第二页起不得出现学校、队员姓名、队伍编号等身份信息。
7. 参考文献按正文引用次序列出，正文以 `[1][3]` 编号引用；书籍引用需给出页码。

### 为什么封面是图片而不是 LaTeX 重排的

官方要求「按附件3 模板编写」。封面、4 个 Logo、摘要页顶部的赛事标题，以及
「题目 / 摘要 / 关键词」这些固定标签，全部**直接取自官方附件3 的渲染结果**。
这样做有两个好处：

- 排版与 Word 版**逐像素一致**，不存在 LaTeX 与 Word 在字距、位置上的差异；
- 固定文字不受「华文新魏 / 隶书」等字体缺失的影响（这几个字不是排出来的，是贴上去的）。

### 字体策略（双回落，不随包字体）

| 目标 | 有就用 | 没有就回落 |
|---|---|---|
| 中文 | `SimSun` / `SimHei`（Windows 宋体 / 黑体，**2026 规范点名的字体**） | `Noto Serif CJK SC` / `Noto Sans CJK SC` |
| 西文 | `Times New Roman` | `Liberation Serif` |

**本模板不带任何 `.ttf`**，所以不存在第 2 节那种「商用字体随包分发」的问题。
回落分支只为让 Overleaf / Linux 上也能先看到版式；**最终提交版建议在 Windows 上编**，
以拿到与官方 Word 附件3 一致的宋体 / 黑体字形。

Debian / Ubuntu 上要跑通回落分支，需装 `fonts-noto-cjk` 与 `fonts-liberation`
（本仓库 CI 已点名并加 `fc-list` 断言）。

---

## 2. `gmcm/` —— 华为杯（中国研究生数学建模竞赛）

| 项目 | 内容 |
|---|---|
| 性质 | **本仓库作者自制**（自己整理、调整、补注释的版本） |
| 文档类 | `gmcmthesis.cls` v2.2 |
| 参考谱系 | `springli07/GMCM_LaTeX_overleaf` → `zhanwen/MathModel`（`gmcmthesis.cls` 这一支的常见流传路径） |
| 上游许可证 | **两处上游仓库均未附 LICENSE 文件**，`.cls` 内也没有授权声明 |
| 本仓库的处理 | 按「注明来源、原样收录」处理，**不做再许可** |

> **和 `hwcup2026/` 的关系**：两套都是华为杯，**并存、互不替代**。投 2026 年
> 研赛用 `hwcup2026/`（严格对齐官方附件3）；`gmcm/` 是社区沿用的通用排版版，
> 章节 / 图表 / 算法环境自定义更全，保留下来供参考写法。

### 本仓库对它的实质改动

1. **修掉一段必然触发的字体 bug**。上游的隶书判断

   ```latex
   \ifx\lishu\undefined
    \setCJKfamilyfont{zhli}{LiSu.ttf}
    \newcommand*{\lishu}{\CJKfamily{zhli}}
   \else
   \fi
   ```

   因为 `ctex` 宏包已经定义过 `\lishu`，条件**永远不成立**，随包的 `LiSu.ttf`
   从未被注册，编译到摘要标题必报
   `! Package fontspec Error: The font "LiSu" cannot be found`。
   现改为 `\providecommand*{\lishu}{}` 兜底 + `\renewcommand*` 绑定；`\xinwei` 同样处理。

2. **西文字体改成可回落**。原来无条件 `\setmainfont{Times New Roman}` 等三行，
   在没有这些字体的机器（Overleaf / Linux / macOS）上直接编译失败。
   现改为 `\IfFontExistsTF`，缺失时回落 `TeX Gyre Termes / Heros / Cursor`
   （Times / Arial / Courier 的度量兼容克隆，随 TeX Live / MiKTeX 分发）。

3. **5 个中文字体改为可缺省**。`SimSun.ttf` 等改为「文件在就用、不在就回落
   `ctex` 自动字体集」，删掉字体也能编过。

除以上三处字体相关改动外，`MathModel.tex`、`figures/*` 与上游一致。

### 随包携带的 5 个 `.ttf`（约 44 MB）

`SimSun.ttf`、`SimHei.ttf`、`KaiTi.ttf`、`LiSu.ttf`、`STXinwei.ttf` 是 **Windows
系统自带的商用中文字体**，版权归中易（ZhongYi）、华文（SinoType）等所有，
**不是自由字体、不能随本仓库的 MIT 许可证一起再分发**。

本仓库随包携带它们，只是为了让你零配置编译、且字形与 Word 完全一致。
**如需再分发或商用，请自行确认授权**；或者直接删除这 5 个文件——文档类会自动
回落到 `ctex` 的字体集（Windows → 系统自带，Linux / Overleaf → 自由字体 `fandol`），
照样编得过，只是字形会变。

---

## 3. `cumcm/` —— 国赛（全国大学生数学建模竞赛）

| 项目 | 内容 |
|---|---|
| 来源 | [`latexstudio/CUMCMThesis`](https://github.com/latexstudio/CUMCMThesis) |
| 上游提交 | `38d1f216bec3c9ffffb7dd09bf6b6c54f486b130`（2026-08-26，文档类 v2.9） |
| 收录文件 | `cumcmthesis.cls`、`cumcm2026.sty`、`example.tex`、`figures/*` |
| 许可证 | ⚠️ **上游没有 LICENSE 文件**，`.cls` / `.sty` 内也没有授权声明 |
| CTAN | **未被 CTAN 收录**（因此没有可引用的标准许可证文本） |
| 本仓库的处理 | 按「注明出处、原样收录」处理，**不做任何再许可** |

### 本仓库对它的实质改动

只有一处，且只动字体加载：`cumcmthesis.cls` 里原来是

```latex
\setmainfont{Times New Roman}
%\setmonofont{Courier New}
\setsansfont{Arial}
```

在没有任何这些字体的机器上会直接
`! Package fontspec Error: The font "Times New Roman" cannot be found`。
现改为 `\IfFontExistsTF`，缺失时回落 `TeX Gyre Termes / Heros`。
（`\setmonofont` 上游本来就是注释掉的，保持原样未动。）

`example.tex`、`cumcm2026.sty`、`figures/*` 与上游**逐字节一致**。

### 分发建议

上游既无许可证、也未上 CTAN，属于「作者未明确授权」的状态。本仓库只做**收录与
出处标注**。如果你打算在自己的项目里广泛分发这套模板，建议先向
[`latexstudio/CUMCMThesis`](https://github.com/latexstudio/CUMCMThesis) 提 issue
请上游补充许可证。

---

## 4. `mcm/` —— 美赛（MCM / ICM）

| 项目 | 内容 |
|---|---|
| 来源 | [`latexstudio-org/mcmthesis`](https://github.com/latexstudio-org/mcmthesis) |
| 上游提交 | `8ac05e2c3a9ef5880a15e3a3a18762a546c10b69`（2024-01-25，文档类 v6.3.3，CTAN 发行版 2024-01-22） |
| 收录文件 | `mcmthesis.cls`、`mcmthesis.dtx`、`mcmthesis.ins`、`mcmthesis-demo.tex`、`mcmthesis.pdf`、`mcmthesis-demo.pdf`、`code/*`、`figures/*` |
| 许可证 | ✅ **LaTeX Project Public License (LPPL) 1.3c or later**，状态 `maintained` |
| 版权 | Copyright © 2010–2015 Zhaoli Wang；2014–2019 Liam Huang；2019–2024 latexstudio |
| 本仓库的改动 | **无**。`mcmthesis.cls` 等全部文件与上游逐字节一致 |

### 使用与再分发时的两点 LPPL 要求

1. **生成文件的分发条件**：LPPL 要求 `mcmthesis.cls` 的分发以「原始源文件属于
   同一分发」为前提。本仓库因此**同时保留了 `mcmthesis.dtx`**（以及生成驱动
   `mcmthesis.ins`）。如果你单独把 `mcmthesis.cls` 发给别人，**请连 `.dtx` 一起给**。
2. **修改必须改名**：LPPL 规定修改后的文件必须使用不同的文件名。要定制
   `mcmthesis.cls`，请另存为 `mythesis.cls` 之类再改，不要沿用
   `mcmthesis.cls` 这个名字。

许可证全文：<https://www.latex-project.org/lppl.txt>

---

## 5. 本仓库自身的许可证范围

本仓库根目录的 [`LICENSE`](../../../LICENSE)（MIT）**只覆盖本仓库作者的原创内容**
（技能正文、脚本、自写的精简模板、`hwcup2026.cls` 等）。

**`assets/latex/full/` 下的第三方模板不在 MIT 的覆盖范围内**，各自的授权状态
以上面四节为准：

- `hwcup2026/` → `.cls` 与 `main.tex` 为作者自制（MIT）；**但 5 张官方附件3 渲染图
  的版权属竞赛组委会，不可再分发** ⚠️；
- `gmcm/` → 作者自制，但上游两处均未附许可证；随包 `.ttf` 为商用字体，**不可再分发**；
- `cumcm/` → 上游无许可证、未上 CTAN，本仓库仅作出处标注地收录；
- `mcm/` → **LPPL 1.3c or later**（唯一有明确自由许可证的一套）。

如果你要基于这些模板做商业分发，请先解决上表中标注 ⚠️ 的授权问题。
