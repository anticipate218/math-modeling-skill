# 第三方模板溯源与授权

`assets/latex/full/` 下三套完整 LaTeX 模板的来源、上游提交号与授权状态。
**本仓库对这些模板不做任何再许可**；许可证以各上游为准，下面逐条写明。

---

## 1. `gmcm/` —— 华为杯（中国研究生数学建模竞赛）

| 项目 | 内容 |
|---|---|
| 性质 | **本仓库作者自制**（自己整理、调整、补注释的版本） |
| 文档类 | `gmcmthesis.cls` v2.2 |
| 参考谱系 | `springli07/GMCM_LaTeX_overleaf` → `zhanwen/MathModel`（`gmcmthesis.cls` 这一支的常见流传路径） |
| 上游许可证 | **两处上游仓库均未附 LICENSE 文件**，`.cls` 内也没有授权声明 |
| 本仓库的处理 | 按「注明来源、原样收录」处理，**不做再许可** |

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

## 2. `cumcm/` —— 国赛（全国大学生数学建模竞赛）

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

## 3. `mcm/` —— 美赛（MCM / ICM）

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

## 4. 本仓库自身的许可证范围

本仓库根目录的 [`LICENSE`](../../../LICENSE)（MIT）**只覆盖本仓库作者的原创内容**
（技能正文、脚本、自写的精简模板等）。

**`assets/latex/full/` 下的第三方模板不在 MIT 的覆盖范围内**，各自的授权状态
以上面三节为准：

- `gmcm/` → 作者自制，但上游两处均未附许可证；随包 `.ttf` 为商用字体，**不可再分发**；
- `cumcm/` → 上游无许可证、未上 CTAN，本仓库仅作出处标注地收录；
- `mcm/` → **LPPL 1.3c or later**（唯一有明确自由许可证的一套）。

如果你要基于这些模板做商业分发，请先解决上表中标注 ⚠️ 的两套的授权问题。
