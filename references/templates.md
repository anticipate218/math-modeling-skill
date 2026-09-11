# 模板与工具链

论文写作与编译阶段的可操作清单。**规则类要求**（页数、匿名、附录）见 `contests.md`，本文件只讲"用什么、怎么跑通"。

---

## 一、论文模板选型

| 竞赛 | 推荐模板 | 说明 |
|---|---|---|
| 国赛 CUMCM | **CUMCMThesis**（社区维护，已适配 2026 格式） | LaTeX；2026 版已加入 AI 使用声明书结构 |
| 研赛 华为杯 | **GMCMthesis**（社区） | 摘要页即第 1 页 |
| 美赛 MCM/ICM | **mcmthesis**（CTAN，LPPL 许可） | 事实标准；COMAP 官方另提供 Summary Sheet 的 Word/LaTeX 模板 |

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
