# 模板与工具链

论文写作与编译阶段的可操作清单。**规则类要求**（页数、匿名、附录）见 `contests.md`，本文件只讲"用什么、怎么跑通"。

---

## 一、论文模板选型

**本仓库自带两套模板，按用途二选一**（对照见 `assets/latex/README.md`，取舍理由见 `assets/latex/full/README.md`）：

| 用途 | 位置 | 形态 |
|---|---|---|
| **2026 研赛·华为杯正式提交**（推荐） | `assets/latex/full/hwcup2026/` | 作者自制的 `hwcup2026.cls`，逐条复刻 2026 官方《论文格式规范》与附件3 Word 模板；封面直接用官方附件3 渲染图；`xelatex ×3`（参考文献内联，不需要 bibtex） |
| **正式参赛提交**（推荐） | `assets/latex/full/{cumcm,gmcm,mcm}/` | 完整文档类 + 完整正文骨架 + 图片 + 预编译样例 PDF；国赛/研赛 `xelatex ×3`，美赛 `pdflatex ×3`（参考文献内联，不需要 bibtex） |
| 自控排版 / 由 Markdown 快速成稿 | `assets/latex/{cumcm,yjs,mcm}/` | 单个 `main.tex` + `refs.bib`，不依赖私有宏包；`xelatex → bibtex → xelatex ×2` |

> **华为杯有两套**：投 **2026 年** 用 `full/hwcup2026/`（严格格式版，封面即附件3）；
> `full/gmcm/` 是社区沿用的 `gmcmthesis.cls` 通用排版版，**两套并存**，后者章节/
> 图表/算法环境更全，适合参考写法。

上游模板谱系（各自版权归原作者，详见 `assets/latex/full/THIRD-PARTY.md`）：

| 竞赛 | 推荐模板 | 说明 |
|---|---|---|
| 国赛 CUMCM | **CUMCMThesis**（社区维护，已适配 2026 格式） | LaTeX；2026 版已加入 AI 使用声明书结构；上游未附 LICENSE |
| 研赛 华为杯（2026） | **`hwcup2026.cls`**（本仓库作者自制） | 严格复刻 2026 官方附件3；封面/固定标签取自官方渲染图；不含随包字体 |
| 研赛 华为杯（通用） | **GMCMthesis** | 摘要页即第 1 页；本仓库这一份是作者在公开谱系上自制整理的 |
| 美赛 MCM/ICM | **mcmthesis**（CTAN，LPPL 1.3c+ 许可） | 事实标准；COMAP 官方另提供 Summary Sheet 的 Word/LaTeX 模板 |

- 链接见 `contests.md` 的"官方链接"表末尾。
- **官方模板优先**：美赛官方提供 Summary Sheet 模板，研赛竞赛系统内提供论文模板附件——能用官方就用官方。
- ⚠️ **社区模板自带封面字段（校名/姓名/队号）与匿名要求冲突**：国赛/研赛必须删掉这些字段，否则可能直接违规。用模板后**第一件事就是检查封面**。

---

## 二、编译（最容易卡住的一步）

```bash
# 中文模板（国赛/研赛，基于 ctex）：必须用 XeLaTeX
latexmk -xelatex main.tex

# 英文模板（美赛）
latexmk -pdf main.tex
```

- **中文模板用 pdfLaTeX 会在字体上直接失败**——这不是模板问题，是引擎问题。
- **不要手写字体名**（Windows/macOS/Linux 字体名不同）；交给 ctex 的 `fontset` 机制自动选择，跨机器协作才不炸。
- 图放 `figures/`，一律**相对路径 + 正斜杠**引用（`figures/flow.pdf`），不要出现 `figures\flow.pdf`。
- 编译产物不要进版本库：`.aux .log .out .toc .synctex.gz .bbl .blg`。
- **改完模板后先自检再提交**（仓库自带，纯标准库；`--require` 表示"没有 TeX 就报错"而不是静默跳过）：

```bash
python scripts/check_latex.py --self-test       # 不装 TeX 也能跑：测日志解析/字体替换/顺序核对
python scripts/check_latex.py --require         # 真编 assets/latex/ 三套轻量模板，核对引用/字体/页数/AI 声明顺序
python scripts/check_latex.py --keep-fontset    # 逐字验证仓库里这一份（本机有 Windows 字体时）

python scripts/check_latex_full.py --self-test        # 不装 TeX：测字体回落改写/顺序核对/固件自洽（29 项）
python scripts/check_latex_full.py --require          # 真编 assets/latex/full/ 四套完整文档类模板（各 3 遍，共 7 跑）
python scripts/check_latex_full.py --only hwcup2026   # 只编 2026 华为杯严格格式版
python scripts/check_latex_full.py --only gmcm        # 只编一套，改单个模板时用
```

  它会在**系统临时目录的副本**里编译，仓库内不留任何产物；`fontset=windows` 只在临时副本里被换成 `fandol`（CI 在 Linux 上跑），日志里出现 `Font "…" cannot be found` 即判失败。四套模板的这一检查已接入 CI。

  `check_latex_full.py` 额外做一件 `check_latex.py` 做不到的事：把随附的中文字体 `.ttf` **删掉**、并强制 `fontset=fandol`，再编一遍——验证**"换一台没有 Windows 字体的机器（含 Overleaf）也能编过、页数不变"**。改动文档类里的字体设置后务必跑它。

  ⚠️ **两套回落目标不同**：`cumcm` / `gmcm` 回落 **fandol + TeX Gyre**；`hwcup2026`
  回落 **Noto Serif/Sans CJK SC + Liberation Serif**（Debian/Ubuntu 上是
  `fonts-noto-cjk` + `fonts-liberation`，本仓库 CI 已点名并有 `fc-list` 断言）。

---

## 三、参考文献

| 竞赛 | 风格 | 特别注意 |
|---|---|---|
| 国赛 | 按科技论文规范（GB/T 7714 风格） | 正文引用处须标注 `[n]` |
| 研赛 | 官方给定三种写法（书籍/期刊/网上资源） | **引书必须标页码**；**引用程序须注明来源** |
| 美赛 | inline citation + References/Bibliography | 计入 25 页 |

- 研赛官方格式：
  - 书籍：`[编号] 作者，书名，出版地：出版社，起止页码，出版年。`
  - 期刊：`[编号] 作者，论文名，杂志名，卷期号：起止页码，出版年。`
  - 网上资源：`[编号] 作者，资源标题，网址，访问时间（年月日）。`
- LaTeX 可用 `gbt7714` 宏包 + bibtex/biber 自动排版；Word 用户请手动统一标点与顺序。

---

## 四、图表规范（评审对"呈现"的打分就在这里）

- **图题在图下、表题在表上**，居中；表格用**三线表**。
- 坐标轴必须写清**物理量与单位**；图例清晰；关键结论标注在图上。
- **分辨率 ≥300 dpi**；矢量图优先（PDF/EPS），截图不要直接贴。
- 有效数字位数全文一致，且与数据精度匹配。
- **matplotlib 中文字体要显式设置**，否则全是方框：

```python
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]   # Windows；macOS 用 "Heiti TC"
matplotlib.rcParams["axes.unicode_minus"] = False     # 负号正常显示
```

- 每个图/表都要在正文**被引用并解释**（不能只放图不说话；`scripts/check_paper.py` 会提示只出现一次的编号）。

---

## 五、版本管理与协作

- 用 git 管理**源文件**（`.tex`/`.bib`/`.py`），不提交编译产物与个人论文 PDF。
- 多人协作按章节拆分：`main.tex` 用 `\input{sections/model.tex}` 组织，减少冲突。
- 结果数字**不要手抄**：求解脚本直接输出 CSV/JSON，论文里的表格由脚本生成，避免"论文数字与代码不符"（国赛这条可能导致取消资格）。
- 定稿后导出 PDF 前，先跑：

```bash
python scripts/check_paper.py paper.md --contest cumcm
```

---

## 六、结果落盘模板（建议每个求解脚本都这么做）

```python
import json, random, numpy as np

random.seed(42); np.random.seed(42)          # 固定随机种子，保证可复现
result = {"q1": {"objective": 12345.6, "feasible": True}, "meta": {"solver": "scipy 1.13"}}
with open("results/q1.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))  # 数据走 stdout，日志走 stderr
```

论文中的每个数值都应能在 `results/` 里找到出处——这样附录代码与论文数字不可能对不上。
