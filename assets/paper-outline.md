# 论文骨架模板（可填空版）

三套独立的骨架：国赛/研赛用中文、美赛用英文。建议用 `scripts/check_paper.py --init` 直接生成 Markdown 文件。

---

## 一、国赛 CUMCM / 研赛 华为杯骨架（中文）

```markdown
# 【一句话标题，含方法或结论】

## 摘要

针对问题一，【说明要解决什么问题】，本文建立【方法/模型名称】，
求得【具体数值结果 + 单位】，相对误差约【x%】。

针对问题二，【……】。针对问题三，【……】。

本文的创新点在于【一句话】；灵敏度分析表明在【扰动范围】内【结论】。

关键词：【关键词1】；【关键词2】；【关键词3】；【关键词4】

---

## 一、问题重述

用自己的语言复述背景、已知条件与待求目标。**不要直接复制题面**。

## 二、问题分析

### 2.1 问题一的分析
【核心矛盾、关键变量、可行方法对比（为什么选 A 而不选 B）】

### 2.2 问题二的分析
### 2.3 问题三的分析

## 三、模型假设

- **假设 1**：【内容】。理由：【依据】。对模型的影响：【影响】。
- **假设 2**：……
（建议 3–6 条，每条都要有"理由 + 对模型的影响"）

## 四、符号说明

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| $x_i$ | 【……】 | 【……】 |

## 五、模型的建立与求解

### 5.1 问题一：【模型名称】

**（1）为什么选这个模型**
【动机 + 与替代方案的对比，如"最短路模型 vs 整数规划模型"】

**（2）模型建立**
【数学公式、目标函数、约束条件】

**（3）求解算法**
【算法流程、参数设置、收敛性分析】

**（4）结果**
求解结果见图 1 与表 1。……

### 5.2 问题二：【模型名称】
### 5.3 问题三：【模型名称】

## 六、结果分析与检验

【误差分析 / 与朴素基线的对比 / 灵敏度分析 / 稳健性讨论】

- **图 1**：【图题，含单位与刻度说明】
- **表 1**：【表题，含单位】

对比结果表明【结论】；图 1 显示【趋势】，表 1 给出具体数值。

## 七、模型的评价与改进

**优点**：【……】　**缺点**：【……】　**改进方向**：【……】

## AI 工具使用声明

本参赛队在竞赛过程中未使用任何AI工具。

> 若使用了 AI：本参赛队在竞赛过程中使用了AI工具，主要用于【用途】，
> 详细使用情况见支撑材料《AI工具使用详情.pdf》。

## 参考文献

[1] 【作者】. 【题名】. 【出处】, 【年份】, 【卷(期)】: 【页码】.
[2] ……
[3] ……

## 附录

### 支撑材料文件列表

- `code/q1.py`：问题一求解程序
- `data/raw.csv`：原始数据

（若确实没有支撑材料，写"本论文没有支撑材料"）

### 程序代码

\```python
# 在此粘贴与论文结果一致的完整、可直接运行的源程序
# 结果数据应输出到 JSON/CSV 文件，避免手抄数字
\```
```

**国赛专用提醒**：
- 电子版第 1 页必须是摘要页（不含承诺书与编号专用页）
- 正文 + 附录 ≤30 页
- 页眉格式：`承诺书编号（队号后 4 位）` | `第 X 页 共 Y 页`

**研赛专用提醒**：
- 摘要页即第 1 页（无承诺书与编号专用页）
- 摘要 ≤2 页（国家级规范）；部分省级更严（如湖南 ≤1 页）
- 全文不得有页眉（只能有页码，且不含队号）

---

## 二、美赛 MCM/ICM 骨架（英文）

```markdown
# Summary

【队号与页码在页眉：Team # 0000000, Page 1 of 25】

We address the problem of 【what you are asked to do】. We develop a 【model/method】
based on 【approach】, and we find 【key result with numbers】, with an error of 【x%】.
We conclude that 【decision-relevant conclusion】.

**Key points for Summary Sheet**:
- Must be **one page, ≥12pt**, at the very front of your paper
- 250–400 words recommended; **write it last**, iterate multiple times
- COMAP FAQ: "Judges are unlikely to read beyond a poorly constructed summary."

---

## Table of Contents (Optional, but counts toward 25 pages)

1. Introduction ................................... 2
2. Assumptions ................................... 3
...

---

## Introduction

Restate the problem in your own words — **do not copy the problem statement**.
Explain the context, what is given, and what is asked. See [1].

---

## Assumptions and Justifications

- **Assumption 1**: 【content】  
  **Justification**: 【reason】  
  **Impact on the model**: 【effect】

- **Assumption 2**: ……

（3–6 assumptions; each must have justification and impact）

---

## Notations

| Symbol | Meaning | Unit |
| --- | --- | --- |
| $x_i$ | 【…】 | 【…】 |

---

## Model Development

### Subsection 1: 【Model Name】

**（1）Why this model**
【Motivation, comparison with alternatives】

**（2）Mathematical formulation**
【Objective function, constraints, equations】

**（3）Solution algorithm**
【Algorithm steps, parameter tuning, convergence analysis】

**（4）Results**
See Figure 1 and Table 1. ……

### Subsection 2: ……

---

## Results

【Present figures and tables; **reference each in the text** with explanation】

- **Figure 1**: 【Caption below the figure, with units and scale】
- **Table 1**: 【Caption above the table, with units】

The results show 【conclusion】; Figure 1 demonstrates 【trend】, and Table 1 lists the exact values.

---

## Sensitivity Analysis

【Error analysis, conditioning, and sensitivity of results to your own assumptions】

Figure 1 shows the sensitivity curve; when we perturb 【parameter】 by ±10%, the
【outcome】 changes by 【x%】, indicating 【stable / sensitive】 behavior.

**Required by COMAP**: This section must explicitly test your assumptions.

---

## Strengths and Weaknesses

**Strengths**:
- 【advantage 1】
- 【advantage 2】

**Weaknesses**:
- 【limitation 1】
- 【limitation 2】

**Required by COMAP**: Must explicitly list both.

---

## Conclusions

【Explicit answers to every question asked, with numbers and units】

For question 1, we recommend 【decision】 because 【reason with numerical support】.
For question 2, ……

---

## References

[1] 【Author】. 【Title】. 【Journal/Book】, 【Year】, 【Volume(Issue)】: 【Pages】.
[2] ……
[3] ……

（References and citations count toward the 25-page limit）

---

## Appendix

### A. Detailed Derivations
【Long proofs, supplementary tables, etc.】

### B. Complete Code

\```python
# Complete, runnable code that reproduces every number in the report
# Output results to JSON/CSV files to avoid manual transcription errors
\```

（Appendix counts toward the 25-page limit）

---

## Report on Use of AI

We did not use AI tools.

**If AI was used**:
- Tool: 【name and version, e.g., ChatGPT 4.0】
- Purpose: 【e.g., debugging code, literature search】
- Verification: 【how you validated the output】
- Add inline citations in the main text where AI output was used.

**This section goes AFTER the 25-page solution and does NOT count toward the limit.**

---

## Memo / Letter to 【Recipient】 (if required by the problem)

【Check the problem statement; if a memo/letter is required, it is usually 1–2 pages】

**Format**:
- To: 【…】
- From: Team # 0000000
- Date: 【…】
- Re: 【subject】

【Body: executive summary of your recommendations, written for a non-technical audience】
```

**美赛专用提醒**：
- 全篇 ≤25 页（含摘要、目录、参考文献、附录、代码）
- 每页页眉必须含：`Team # 控制号, Page X of Y`
- Memo/Letter 是**逐题面要求**（近年多数题目有；选题后第一时间确认并预留 1–2 页）
- Report on Use of AI 不计入 25 页

---

## 三、使用建议

1. **用脚本自动生成**：
   ```bash
   python scripts/check_paper.py --init --contest cumcm -o my_paper.md
   python scripts/check_paper.py --init --contest mcm -o my_paper.md
   ```

2. **填写时一边写一边跑自检**：
   ```bash
   python scripts/check_paper.py my_paper.md --contest cumcm
   ```

3. **结果数字不要手抄**：求解脚本直接输出 JSON/CSV，论文里的表格由代码生成，
   避免"论文数字与附录代码不符"（这是学术不端红线，可能取消资格）。

4. **摘要最后写**：全文定稿后，用摘要"倒逼"自己提炼核心结论。
   参考 `assets/abstract-template.md` 的四要素结构。

5. **每个占位符【……】都要填**：不要留空，也不要写"略"。
   - 假设必须有"理由 + 影响"
   - 方法必须有"为什么选它 + 与替代方案对比"
   - 结果必须有"数值 + 单位 + 检验"
