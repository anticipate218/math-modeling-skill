# 摘要模板（国赛/研赛 中文版 + 美赛 Summary Sheet 英文版）

摘要决定论文的第一印象：评委通常先读摘要，再决定是否细看正文。
**评判标准：不读正文的人，也能从摘要知道"问题是什么、你用了什么方法、得到什么结果、结果有多可信"。**

---

## 一、国赛 / 研赛 摘要模板（单页，约 600–1000 字）

```text
摘  要

针对问题一，本文建立了 <模型类型> 模型。首先，对 <数据/条件> 进行 <预处理/分析>，
得到 <关键量>；其次，以 <目标函数> 为目标、以 <约束条件> 为约束，构建 <模型名>；
然后采用 <求解算法/工具> 求解，得到 <具体数值结果>。最后通过 <检验方式> 检验，
<误差/偏差> 为 <数值>，表明模型 <结论>。

针对问题二，在问题一的基础上，本文引入 <新增因素/机制>，建立了 <模型名>。
通过 <方法> 求解得到 <结果>，并与 <基线/其他方法> 对比，<指标> 提升了 <百分比>。
进一步地，对 <关键参数> 进行灵敏度分析，结果表明 <结论：哪些参数敏感/不敏感>，
当 <参数> 在 <范围> 内变动时，<目标量> 的波动不超过 <数值>，说明模型具有较好的稳健性。

针对问题三，本文建立了 <模型名>，设计了 <策略/方案>。求解结果表明 <核心结论>，
<给出可执行建议>。

本文的创新点在于：<1–3 条，必须是真创新，如新增约束、改进算法、组合建模、数据融合>。

关键词：<模型名>；<方法>；<关键词3>；<关键词4>
```

### 硬性要求（对照检查）
- [ ] 每一问都出现"针对问题X"，问号与正文一致，不遗漏任何一问
- [ ] 每一问都有：方法 + 具体结果（有数字）+ 检验
- [ ] 出现具体数值（误差、提升比例、最优值），不能只有"效果良好"
- [ ] 有灵敏度/稳健性的一句结论
- [ ] 有关键词 3–5 个
- [ ] 不超过一页（含标题与关键词）
- [ ] 没有出现学校、姓名、赛区、队号等身份信息
- [ ] 没有出现"我们组""老师"等口语表达；不用第一人称"我"

### 常见摘要失败模式
| 失败写法 | 问题 | 改法 |
|---|---|---|
| "本文对问题进行了深入研究，建立了合理的模型" | 全是空话，无信息 | 写明模型名、方法、结果数值 |
| 只写方法不写结果 | 评委无法判断做对没做对 | 每问补具体数值结论 |
| 大段复述题目 | 浪费篇幅 | 问题重述压缩到一两句 |
| 出现"由于时间关系""未能完成" | 直接暴露缺陷 | 不要在摘要里自我否定 |
| 摘要与正文结论不一致 | 严重扣分 | 定稿时逐句核对 |

---

## 二、MCM/ICM Summary Sheet 模板（英文，约 250–400 词）

```text
# Summary

<1 句背景/目标: We address the problem of ... requiring ...>

For <Task 1>, we develop a <model type> model. <State the key assumption(s) and the
governing equations or objective function in one compact sentence.> Using <method/tool>,
we find that <quantitative result>. <Validation sentence: This result is consistent with /
differs from ... by ...>

For <Task 2>, we extend the model by introducing <new mechanism/variable>. <Method>.
The results show <quantitative finding>, a <X%> change relative to <baseline>.

We perform a sensitivity analysis on <parameters>. The model is <robust/sensitive>:
varying <parameter> by +/-20% changes <output> by less than <Y%>. <Implication.>

Finally, we recommend <concrete recommendation>. Our main strengths are <...>; the main
limitations are <...>.

Keywords: <model>; <method>; <application>
```

### 硬性要求
- [ ] 第一段就点明**结论导向**：评委读前三句就应知道你的方案与结果
- [ ] 每一问都有量化结果，避免纯定性描述
- [ ] 明确写出假设与模型形式（可含一两行公式）
- [ ] 有灵敏度/稳健性结论
- [ ] 有对 strengths/limitations 的诚实说明
- [ ] 语言为英文且无语法硬伤（摘要的语言质量直接影响评级）
- [ ] 全文不使用第一人称单数 "I"（用 "we" 或被动语态）
- [ ] Summary 单独成页，位于全文最前面

---

## 三、写作顺序建议（提高效率）

1. **先写摘要骨架**（在各问建模思路确定后立刻写，用占位符标出待填数值）
2. 正文完成、数值全部固定后，**回填摘要中的数字**
3. 定稿前**逐句核对摘要与正文的一致性**（结论、数字、问号顺序）
4. 用 `scripts/check_paper.py` 做机械检查，再人工读一遍摘要（限时 1 分钟，看能否读懂）
