# 2026 “华为杯”第二十三届中国研究生数学建模竞赛 LaTeX 严格复刻版

> 这不是组委会发布的“官方 LaTeX 模板”。2026 官方要求使用附件3 Word 模板；本项目依据 2026-09-16 官方《论文格式规范》以及附件3 Word 模板逐项复刻。

## 编译

必须使用 **XeLaTeX**：

```bash
latexmk -xelatex main.tex
```

为了严格对应“宋体/黑体”，最终提交版建议在 **Windows + TeX Live/MiKTeX** 环境编译，使 XeLaTeX 能调用系统中的 **SimSun（宋体）** 和 **SimHei（黑体）**。

- 若系统存在 SimSun/SimHei，本模板自动使用它们。
- 若不存在，会退化到 Noto CJK，仅用于预览；这种情况不要视为最终严格版。
- 固定封面、4 个 Logo、摘要页顶部赛事标题以及“题目/摘要/关键词”固定标签均直接取自官方附件3的渲染结果，因此不会受“华文新魏/隶书”字体缺失影响。

## 已落实的 2026 官方硬性要求

1. A4 纵向；页面边距按附件3 Word 内部设置复刻：上约 30.02 mm、下约 18.49 mm、左约 22.51 mm、右约 22.47 mm。
2. 首页保留官方附件3封皮及 4 个 Logo，不替换。
3. 摘要页开始用阿拉伯数字从 1 连续编号，页码居页脚中部。
4. 不设置页眉。
5. 论文题目：三号黑体。
6. 一级标题：四号黑体、居中。
7. 其余汉字：小四号宋体。
8. 行距：单倍行距。
9. 摘要无需英文，一般不超过两页。
10. 第二页起不得出现学校、队员姓名、队伍编号等身份信息。
11. 参考文献按正文引用次序列出，正文以 [1][3] 等编号引用；书籍引用需指出页码。

## 只需要修改

`main.tex` 文件顶部：

```tex
\school{学校名称}
\teamno{队伍编号}
\memberone{队员一}
\membertwo{队员二}
\memberthree{队员三}
\papertitle{论文题目}
\keywords{关键词1；关键词2；关键词3}
```

然后替换摘要和正文即可。

## 重要说明

官方 2026 开赛公告要求“必须按附件3模板进行编写”。因此，本项目把附件3第一页直接作为封面底图，而不是重新绘制 Logo 或封面固定文字；这是为了最大限度避免 LaTeX 与 Word 在固定版式上的字体、字距和位置差异。

## 在本仓库里的位置

这套模板由**本仓库作者自制**，2026 年新版，逐条复刻官方附件3 Word 模板。

它和 `full/` 下另外的华为杯模板 [`../gmcm/`](../gmcm/) **并存、互不替代**：

| | `hwcup2026/`（本目录） | `../gmcm/` |
|---|---|---|
| 定位 | 2026 官方格式的**严格复刻**（封面直接用附件3 渲染图） | 社区沿用的 `gmcmthesis.cls` 通用排版版 |
| 入口 | `main.tex` | `MathModel.tex` |
| 随包字体 | **不带**（自身探测 SimSun/SimHei，缺则回落 Noto CJK） | 带 5 个 Windows 中文字体（约 44 MB） |
| 缺少 Windows 字体时 | 回落 **Noto Serif/Sans CJK SC + Liberation Serif** | 回落 **fandol + TeX Gyre** |
| 摘要页版本样式 | 严格 2026（固定标签取自附件3 渲染图） | 逐年沿用的一般样式 |

**要投 2026 年华为杯，用本目录这一套。** 老版的 `../gmcm/` 保留下来是因为它的
`gmcmthesis.cls` 更通用（章节、图表、算法环境的自定义更全），适合拿来参考写法。

在 Linux / Overleaf 上想看到接近 Windows 的字形，需要装
**Noto Serif CJK SC** 与 **Noto Sans CJK SC**（Debian/Ubuntu 包名 `fonts-noto-cjk`）
以及 **Liberation Serif**（`fonts-liberation`），否则 `fontspec` 会直接报
`The font "Noto Serif CJK SC" cannot be found` 中止编译。本仓库 CI 的 `latex`
任务就是靠这两个包把「无 Windows 字体」这一条回落路径跑通的。

从 Release 下载的 `hwcup2026-template.zip` 解压后即是本目录（4 个源文件 + 5 张官方
附件3 渲染图），**进入目录再编译**：

```bash
cd hwcup2026
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

**三遍即可，不需要 `bibtex`**——参考文献写在正文里的 `thebibliography` 环境里。
（`preview.pdf` 是作者预编译的 3 页样张，可以直接看版式效果。）
