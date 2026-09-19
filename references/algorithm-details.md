# 算法实现详解（逐算法）

> 这份文件回答的是**最细一层**的问题：**"这个算法到底怎么算的、有哪些参数、复杂度多少、哪里会算错、我该怎么验证它没算错。"**
>
> 分工：
> - `references/model-library.md` —— 这题该用哪类模型；
> - `references/model-implementations.md` —— 怎么从基线升级、验证协议怎么设计；
> - `references/algorithm-implementations.md` —— 一眼速查：哪个族用哪个函数、什么时候换成熟库；
> - **`references/algorithm-details.md`（本文件）** —— 逐算法的数学形式、步骤、参数表、陷阱与检验方法；
> - `references/innovation-playbook.md` —— 哪些参数动了算创新、怎么证明有效。

---

## 0. 怎么用这份文件

1. **先定位族**：按题目类型在下面的目录里找族（优化 / 图论 / 预测 / 统计 …）。
2. **再挑函数**：每个族一张表，列出该族所有函数；表中「数学形式」一栏足以让你判断它是不是你要的东西。
3. **复制代码**：所有实现在 `examples/algorithms/<模块>.py`，只依赖 `numpy` + 标准库，可直接贴进论文附录。
4. **照「怎么检验」跑一遍**：这一栏给的是**独立于本实现**的验证手段（闭式解、对拍、极限行为、留一验证）。论文里的"模型检验"章节可以直接照抄思路。

> **不要一次读完**。这份文件很长，按需读族即可。

---

## 1. 六个字段的含义

每个算法条目统一给六项：

| 字段 | 含义 | 论文里怎么用 |
|---|---|---|
| **数学形式** | 目标函数/递推式/估计量的式子 | 写进"模型建立"，比自然语言描述更不容易被挑错 |
| **步骤** | 可执行的计算流程（编号） | 写进"模型求解"，说明你的算法确实是自己实现的 |
| **复杂度** | 时间/空间量级（n 为样本数、p 为变量数、m 为周期） | 写进"模型评价"，说明规模上限；也是换成熟库的判断依据 |
| **参数** | 参数名、含义、取值建议、调参的后果 | 写进"参数设置"，并作为灵敏度分析的对象 |
| **陷阱** | 该算法最容易算错/最容易误用的地方 | 主动写进"模型假设与局限"，是加分项 |
| **怎么检验** | 独立于本实现的验证方法 | 写进"模型检验"章节 |

---

## 2. 通用纪律（所有族共享）

| 约定 | 具体内容 |
|---|---|
| **依赖边界** | 只允许 `numpy` + Python 标准库。CI 用 AST 静态扫描禁止 scipy / sklearn / pandas / statsmodels / cvxpy / pulp / torch / tensorflow / sympy / matplotlib。 |
| **随机性** | 一律 `from ._common import rng`，用 `rng(seed)` 取 Generator；**不使用 `np.random` 全局状态**。默认种子 `DEFAULT_SEED = 20240101`。 |
| **返回形态** | 统一返回 `dict`（键为 `weights` / `x` / `status` / `forecast` 这类具名结果），方便直接打印成论文表格。 |
| **报错** | 不写 `assert`（`python -O` 会把它删掉），参数非法一律 `raise ValueError`。 |
| **自检** | 每个模块末尾有 `_self_test() -> dict`，内含**闭式解或独立实现的交叉验证**；`examples/run_algorithms.py` 跑两遍比对确定性，并与 `examples/algorithms_golden.json` 的黄金值逐键比对。 |
| **复杂度口径** | 表中的 n/p/m 指：n = 样本数或节点数，p = 变量/指标/决策单元数，m = 季节周期或重复次数。均为朴素实现的量级，不含常数优化。 |

跑一次全量回归：

```bash
python examples/run_algorithms.py                 # 全部模块 + 黄金值
python examples/run_algorithms.py --module graphs --verbose
python examples/run_algorithms.py --list
```

---

## 3. 逐族详解

> 每族一节。条目顺序与该模块 `__all__` 的顺序一致。

### 3.1 优化与运筹 —— `examples/algorithms/optimization.py`

**这族解决什么问题**：把线性规划、整数规划、0-1 背包、指派、运输、目标规划、情景鲁棒 LP、机会约束 LP 和离散选址这九件事做成只依赖 numpy 与标准库的透明实现。数模题里凡是"在若干约束下让某个指标最大/最小"的模型，最后都会落到这几个成型结构上；它们替代的是"直接调求解器黑盒、论文里写不出算法在做什么"的做法，让你能在正文里写出"我们自行实现单纯形法并与求解器结果对照，误差 < 1e-9"（模块 docstring 原话）。生产规模的问题 docstring 也明确建议换 OR-Tools 的 CP-SAT、Pyomo + HiGHS（见 `references/github-resources.md`）。

**共同约定**：
- **不引入 scipy，全部走 `numpy` + 标准库 + `._common`**：本模块只 `from ._common import as_matrix, as_vector`，另用 `numpy`、`math`、`itertools`、`typing`；唯一写死的容差常量是模块级 `_EPS = 1e-9` 与 `_INF = float("inf")`，各函数内部另有自己的阈值（下面逐条写明）。模块 docstring 里与 `scipy.optimize.linprog` 的数值对照只记录在 `_self_test()` 的注释中，代码本身不 import scipy。
- **完全没有随机性**：九个公开函数**都不接收 `seed`**，因此也不涉及 `_common.rng` 与 `DEFAULT_SEED = 20240101` 的回落；全部是确定性算法，同一输入必然同一输出（`itertools.combinations` 按字典序、并列取最小下标等规则都是为可复现服务的）。
- **返回形态统一为 dict，但键名逐函数不同**：没有 `OptimizeResult` 那样的统一结构。最基础的状态键是 `status`，取值为 `"optimal"` / `"infeasible"` / `"unbounded"` / `"max_iter"`（`branch_and_bound_ilp` 另有 `"node_limit"`）；非最优时通常 `x` / `fun` 等为 `None` 而不是抛异常。个别函数返回裸 ndarray 字段（如 `chance_constrained_lp` 的 `slack`），没有 `success`、没有 `message`、没有迭代日志。
- **异常口径**：形状/长度不匹配、长度约束不满足、权重或标准差为负、`bounds` 长度与变量数不符等**输入契约错误一律抛 `ValueError`**（消息里带实际长度与期望长度）；`budget`、`method`、`alpha` 这类取值域错误也抛 `ValueError`。相反，**模型层面**的不可行/无界/超节点数不抛异常，而是通过 `status` 返回。唯一的例外是 `facility_location`：所有组合都不可行时抛 `ValueError`（它的返回值里没有 `status` 字段）。
- **刻意不实现的方法**：仅支持约束右端项的**独立**正态扰动，不做相关（协方差）扰动的联合正态等价；不报告对偶变量/影子价格/灵敏度分析；`budget` 不是 Bertsimas-Sim 的"不确定度预算 Γ"；不做多目标 Pareto 前沿；不做大 M 法/对偶单纯形/内点法；`knapsack_dp` 只支持整数重量，浮点重量要自己先缩放。

#### `simplex_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, maximize=False, max_iter=500)`

- **数学形式**：min cᵀx（`maximize=True` 时内部取负求 max cᵀx），约束 A_ub·x ≤ b_ub、A_eq·x = b_eq、lo ≤ x ≤ hi（lo 为 None 表示 −∞，hi 为 None 表示 +∞；`bounds=None` 等价于全部 x ≥ 0）
- **步骤**：① 变量变换 x = shift + T·z，把一切变量化为 z ≥ 0：lo 与 hi 都缺的变量拆成正负两列，只有 hi 的令 x = hi − t，有 lo 的令 x = lo + t 并为有限 hi 记一条附加行（源码注释特意强调这行**不能**当场构造，否则短行会被 numpy 静默广播，把 x_j ≤ hi 悄悄变成"后续所有 t 之和 ≤ hi"）；② 引入松弛/剩余变量，右端项为负的"≤"行整行取反，再为每个等式与取反后的行引入人工变量，拼出单纯形表；③ **第一阶段**最小化人工变量之和，`phase1_value > 1e-7` 即判 `infeasible`，随后把仍留在基里的人工变量赶出去（整行在真实列上全 0 则判为冗余约束，该行清零，保留 0=0）；④ **第二阶段**用原目标继续迭代，人工变量禁止再入基；每一步入基/出基都走 Bland 规则（最小下标入基 + 相等比值时取基变量下标最小者出基），保证有限步终止；⑤ 从基变量还原 z，再算 x = shift + T·z 与 fun = cᵀx
- **复杂度**：时间最坏 O(2ⁿ)（LP 本身多项式可解，单纯形是实用指数级但实际很快）；空间 O((m+1) × (n+m+人工变量))
- **参数**：`c` 为长度 n 的目标系数（经 `as_vector` 校验）；`A_ub`/`A_eq` 为形状匹配的二维矩阵（经 `as_matrix` 校验），**可以给 None 或缺省表示没有该类约束**；`bounds` 是长度 n 的 `(lo, hi)` 列表，长度不符抛 `ValueError`，`hi < lo`（小于 −1e-9）抛 `ValueError("bounds 中 hi < lo，问题本身不可行")`；`maximize` 默认 False，为 True 时内部把目标取负、返回的 `fun` 仍是**原方向**的目标值；`max_iter` 默认 500，迭代打满返回 `status="max_iter"` 而不是抛异常。返回键为 `status` / `x`（np.ndarray 或 None）/ `fun`（float 或 None）。本函数**没有** `method`、`integrality`、`dual`、`tol`、`callback` 参数，也没有等式形式的 `A_lb`/`A_ub` 分开口径；要整数解用 `branch_and_bound_ilp`，要多目标用 `goal_programming`，要右端项概率化用 `chance_constrained_lp`，要情景最坏情形用 `scenario_robust_lp`。
- **陷阱**：**数值尺度**——系数相差 1e6 倍以上时 1e-9 的判定阈值会让结果不可信，量纲差异大的约束应先无量纲化；**退化与循环**——理论上单纯形可能死循环，这里用 Bland 规则规避，代价是收敛变慢；这个实现**不报告对偶变量**，需要影子价格/灵敏度报告必须换成熟求解器；`status == "unbounded"` 往往说明模型漏了约束，而不是真的存在无界最优解；此外索引入基的阈值是 −1e-9、比值比较容差是 1e-12，转轴元素绝对值小于 1e-12 时 `_pivot` 抛 `ZeroDivisionError`（不是 ValueError），这类病态输入下异常类型会和别处不一致
- **怎么检验**：`_self_test()` 已给四组可手算的算例——max 3x+2y s.t. x+y≤4, x+3y≤6 的最优是 x=(4,0)、fun=12（注释写明用 `scipy.optimize.linprog` 独立核对过）；max 5x+4y s.t. 6x+4y≤13, x≤3 的最优 fun=13（若上界被错误广播成 x+y≤3 会退化成 12.5，这正是曾出的 bug，可用来回归）；x+y=1 的等式算例 fun=−1；x≤1 且 x≥5 返回 `infeasible`，max x 且 x≥0 返回 `unbounded`。独立验证就把每个算例的顶点代入目标函数穷举取最优，再与返回的 `x`/`fun` 对拍；更大规模直接与 `scipy.optimize.linprog`（或 OR-Tools）对拍目标值和最优解

#### `branch_and_bound_ilp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, maximize=False, integer=None, max_nodes=2000)`

- **数学形式**：在 `simplex_lp` 的同一个 LP 可行域上再加整数要求——min cᵀx s.t. A_ub·x ≤ b_ub、A_eq·x = b_eq、lo ≤ x ≤ hi、x_j ∈ ℤ (j ∈ J)，J 由 `integer` 指定；解法是 LP 松弛 + 分支定界
- **步骤**：① 把 `bounds` 拷贝成节点边界，用显式栈做深度优先搜索（每轮 `stack.pop()`）；② 对节点解 LP 松弛（转调 `simplex_lp`，`max_iter=300`），松弛不可行就剪掉该节点；③ 若已有最好整数解且该节点的松弛界不优于它（容差 1e-9），剪枝；④ 在 `integer` 标记的变量里找第一个小数偏差 > 1e-6 的变量 x_j，按 `x_j ≤ floor(x̄_j)` 与 `x_j ≥ ceil(x̄_j)` 分两支压栈（上支先压，因此下支先被弹出探索；上下界交叉（lo > hi + 1e-12）的分支直接丢弃）；⑤ 找不到小数变量即得到一个整数可行解，按 1e-9 容差更新最好解；⑥ 节点数达到 `max_nodes` 时返回 `node_limit`，此时若已有最好解，会用根节点的 LP 最优值算一个相对间隙
- **复杂度**：时间 O(2ⁿ) 最坏；空间 O(n)（递归深度；实现上换成显式栈，量级不变）
- **参数**：`c` 与 `A_ub`/`b_ub`/`A_eq`/`b_eq`/`bounds`/`maximize` 的语义**与 `simplex_lp` 完全相同**（`bounds=None` 同样表示 x ≥ 0）；`integer` 是长度 n 的布尔序列，标记哪些变量必须取整，长度不符抛 `ValueError("integer 长度必须等于变量个数")`，`None` 表示**全部变量**都是整数（想要混合整数规划就显式给 `integer`）；`max_nodes` 默认 2000。返回键为 `status` / `x` / `fun` / `nodes`（实际探索节点数）/ `gap`（最好整数解与 LP 下界的相对间隙）。它**没有** `time_limit`、`mip_gap`、`cuts`、`presolve` 之类的参数，也没有热启动接口；真要大 MIP 请用 OR-Tools CP-SAT，0-1 背包请直接用 `knapsack_dp`。
- **陷阱**：**分支定界不是多项式算法**，docstring 明说变量数超过 ~50 个就该换 CP-SAT；`gap` 在目标值接近时可能因为浮点误差显示成负数，取绝对值理解即可，而且 `optimal` 返回的 `gap` 是硬编码的 `0.0`，不是真实算出来的界；`node_limit` 分支里 `gap` 的基准是**根节点**那一次 `lp_lower_bound(bounds)` 的 `fun`（变量名叫 `lb`，实际就是根松弛值），按 docstring 的措辞"LP 下界"读容易和"当前最好节点界"混淆；剪枝与更新都用 1e-9 容差、判定整数性用 1e-6 容差，两个阈值不是同一个数，目标值极其接近的题目上结果对容差敏感
- **怎么检验**：`_self_test()` 给了 max 5x+4y s.t. 6x+4y≤13、x,y 非负整数的算例，并在注释里写明**暴力枚举**精确最优为 (0,3)、值 12（(2,0) 只有 10），专门用来抓剪枝错误。独立验证就对小规模问题枚举全部整数点取最优，与 `fun`/`x` 对拍；`nodes` 因分支顺序与容差写死而可复现，但它不是正确性指标，别把节点数当断言对象

#### `knapsack_dp(weights, values, capacity)`

- **数学形式**：max Σᵢ vᵢ·xᵢ s.t. Σᵢ wᵢ·xᵢ ≤ C、xᵢ ∈ {0,1}；DP 递推 dp_c^(i) = max(dp_c^(i−1), dp_{c−wᵢ}^(i−1) + vᵢ)
- **步骤**：① 归一化输入：重量逐个 `int()` 取整、价值转 float、容量 `int()` 取整；② 校验长度一致、重量与容量非负、且 `abs(重量 − int(重量)) > 1e-12` 即拒绝；③ 用一维长度 C+1 的 `dp` 数组 + 一张 `take[i, c]` 布尔表，外层遍历物品、内层**倒序**枚举容量（从 C 递减到 wᵢ），保证每件物品最多用一次；④ 从 i = n−1 反向回溯 `take` 表得到 `chosen`，再反转为升序
- **复杂度**：时间 O(n × capacity) / 空间 O(capacity)（DP 数组）+ O(n × capacity)（`take` 表）
- **参数**：`weights` 与 `values` 为等长序列，长度不符抛 `ValueError("weights 与 values 长度不一致")`；重量或容量出现负数抛 `ValueError("重量与容量必须非负")`；重量含小数抛 `ValueError("knapsack_dp 要求整数重量，请先缩放")`。`capacity` 通过 `int()` 截断成整数，**负数才报错**，浮点容量会被静默截断。返回键为 `max_value`（float）/ `chosen`（升序下标列表）/ `total_weight`（int）。它**没有** `maximize` 开关（天生求最大价值）、没有逐物品价值/重量比、没有分组背包或多重背包口径、没有回溯表开关；要"至少装满"或每组选一个的变体得自己改模型。容量很大而物品种类少时改 `branch_and_bound_ilp`。
- **陷阱**：这是**伪多项式**算法，docstring 明说容量取 1e9 时直接爆内存（`dp` 与 `take` 都按 capacity + 1 分配）；重量必须是整数，有小数的重量要先缩放并说明精度损失；把非整数重量浮点转 `int()` 会**静默截断**，因此校验用的是原始数值而不是截断后的值——反过来，像 3.9999999999999 这种本意是 4 的重量会被判非法；`values` 不做非负校验，存在负价值时 DP 的最优会倾向于不选它，语义正常但不要指望得到"必须选满"的解
- **怎么检验**：`_self_test()` 给了容量 10、物品 (w,v) = (5,10)、(4,40)、(6,30)、(3,50) 的算例，最优是选物品 2 和 3，重量 4+3=7、价值 40+50=90，可核对 `max_value` 与 `total_weight`。独立验证就枚举 2ⁿ 个 0-1 组合（n ≤ 20 时很快）对拍 `max_value`，并检查 `chosen` 的重量和等于 `total_weight`、价值等于 `max_value`；再断言重量和不超过容量

#### `assignment_hungarian(cost)`

- **数学形式**：min Σᵢ c_{i,σ(i)}，σ 是把每个工人映到互不相同的任务的匹配（列视角即 min Σⱼ c_{p(j),j}）；算法走 Kuhn-Munkres 的对偶——势函数 u、v 与互补松弛 c_{i,j} − u_i − v_j ≥ 0，沿增广路增广
- **步骤**：① 读入代价矩阵；若行数 n₀ > 列数 m₀ 先转置并记 `transposed=True`，转为 n ≤ m；② 对每个行 i = 1..n 依次找增广路：把 `p[0] = i` 当作虚拟起点，用 `minv[j]` 维护未访问列到当前交替树的最小松弛 c_{i0,j} − u_{i0} − v_j；③ 每轮取最小 `minv` 得到新列 j1，对所有已访问列更新势（u[p[j]] += δ、v[j] −= δ），未访问列 `minv[j] −= δ`；④ 遇到未匹配列（p[j0] == 0）就结束本行，沿 `way` 反向回溯改写匹配 p；⑤ 全部行处理完后从 p 还原每行的任务下标、求和；⑥ 若曾转置，把行→列的指派取反映回列→行（此时语义是"每个任务一个工人"）
- **复杂度**：时间 O(n²m)（n ≤ m 时即 O(n³)）/ 空间 O(nm)
- **参数**：`cost` 是形状 (n, m) 的代价矩阵，经 `as_matrix` 校验（一维或非数值会报错）；**不要求方阵**，也**没有** `maximize`、`forbidden`（禁止某些指派）、`capacity`、`n_jobs` 之类的参数。返回键为 `total_cost`（float，按下标求和得出）/ `assignment`（长度等于"工人数"的下标列表，未指派为 −1）/ `transposed`（n > m 时为 True）。要"每个任务也必须有人做"必须 n == m；要按容量分配请用 `transportation_vogel` 或 `facility_location`。
- **陷阱**：经典匈牙利算法要求 **n ≤ m**，n > m 时本实现自动转置，返回的 `assignment` 已还原成"每个工人一个任务"的语义，但此时必然有工人闲置（值为 −1）；`total_cost` 是转置后矩阵上的求和，转置前后数值相等但行的语义变了；负代价（收益矩阵取负）可以用，但**不能**先给矩阵加常数再求解——那会改变最优解，要把最大化收益转成最小化代价就整体取负；有工人闲置时 n < m 的方阵假设不成立，别把 `assignment` 长度误当成列数；`INF = float("inf")` 参与 `minv` 与势更新，如果某行到所有列的松弛都保持 inf（现实中不会出现，除非传入 inf 元素），`delta` 保持 inf 会让 `u`/`v` 变成 NaN 并静默污染结果
- **怎么检验**：`_self_test()` 给了 cost = [[4,2,8],[1,5,3],[6,4,2]]，注释写明最优为 2+1+2 = 5、指派（列下标）= [1,0,2]，可直接断言 `total_cost` 与 `assignment`。独立验证就对 n ≤ 8 的小矩阵用 `itertools.permutations` 穷举完全匹配取最小值对拍；再专门造一个 n < m 的矩形算例，检查 `transposed=False` 且只有 n 个列被占用；把矩阵转置后再跑一遍，断言 `transposed=True` 且两边 `total_cost` 在浮点末位上一致

#### `transportation_vogel(cost, supply, demand)`

- **数学形式**：min Σᵢ Σⱼ cᵢⱼ·xᵢⱼ s.t. Σⱼ xᵢⱼ = 供应量 sᵢ、Σᵢ xᵢⱼ = 需求量 dⱼ、xᵢⱼ ≥ 0（产销平衡时 Σsᵢ = Σdⱼ）；Vogel 近似法的行/列罚数定义为该线上"次小运价 − 最小运价"，每次在罚数最大的线上用最小运价格尽量多运
- **步骤**：① 用 `abs(s.sum() − d.sum()) < 1e-9` 判平衡，不平就补一个运价全 0 的虚拟产地（供不应求时）或虚拟销地（供过于求时）；② **Vogel 近似**：在仍有余量的行/列里算罚数（该线只剩一个可用格时罚数记为 `big = 1e12`），选罚数最大的线，在该线上取最小运价格，运输量取 `min(剩余供应, 剩余需求)`，循环最多 m×n + 1 轮；③ 把 Vogel 方案记录下来，算 `vogel_cost`，数出基格个数（`plan > 1e-9` 的格数），少于 m+n−1 就标 `degenerate=True`；④ **精确求解**：把运输问题铺成 m·n 个变量的 LP（前 m 行是行和等式、后 n 行是列和等式，系数矩阵用切片 `A_eq[i, i*n:(i+1)*n] = 1.0` 与 `A_eq[m+j, j::n] = 1.0` 拼出），所有变量 bounds 取 (0, None)，转调 `simplex_lp`（`max_iter=2000`）；⑤ 把返回的 x 重塑成 (m, n)，把绝对值小于 1e-9 的浮点残渣清零（源码注释称之为"已做清理"），重算 `total_cost`，并用 `total < vogel_cost − 1e-9` 判 `improved`
- **复杂度**：时间 Vogel 部分 O((m+n)²·(m+n))，精确 LP 部分为单纯形迭代开销；空间 O(mn)（LP 版本会展开成 m·n 个变量）
- **参数**：`cost` 是形状 (m 产地, n 销地) 的运价矩阵（经 `as_matrix`）；`supply` 长度 m、`demand` 长度 n（都按 float 读入），长度不匹配抛 `ValueError("supply/demand 长度与 cost 形状不匹配")`，出现负数抛 `ValueError("供应量与需求量必须非负")`。返回键为 `total_cost` / `plan`（精确最优）、`vogel_cost` / `vogel_plan`（Vogel 初始解）、`improved`、`iterations`、`balanced`、`degenerate`、`lp_status`。它**没有** `method` 开关（想只看启发式解就拿 `vogel_plan`）、没有产地上限/路径禁止、没有整数求积开关；密集容量约束的选址-分配请用 `facility_location`。
- **陷阱**：**退化**——基格少于 m+n−1 个时位势法（MODI）的解不唯一，也是手工计算最容易出错的地方；本实现因此用 LP 保证最优性，并把 Vogel 初始解单独报告出来，`degenerate=True` 只是提示"这条初始解退化，手工检验时要小心"；**运价必须非负**，负运价（补贴）会让"尽量多运"的直觉失效，也要求 Vogel 的罚数逻辑改写（代码里没有非负校验，负运价不会报错但启发式质量无保证）；平衡后新加的虚拟格运价为 0，直接照抄进论文会让人误以为真的免费运输，表注里要说明哪一行/哪一列是虚拟的；这里不把结果强制取整——供需为整数时最优解自然是整数（约束矩阵全幺模），浮点误差可能留下 1e-13 级别的残渣，本实现已清理；`iterations` 恒为 1，它**不是** Vogel 的迭代次数，也不是 MODI 的迭代次数（源码里初始化为 1 后再未修改），论文里不要拿它当"迭代轮数"报告；精确解走 `simplex_lp`，规模大时可能返回非 `optimal`，此时会**退回 Vogel 解**并把该状态写进 `lp_status`（返回结构的键仍然齐全，容易误读成 `total_cost` 就是最优）
- **怎么检验**：`_self_test()` 给了经典 3×3 算例 cost = [[8,6,10],[9,12,13],[14,9,16]]、supply = [20,30,30]、demand = [30,20,30]，注释写明精确最优为 **810**，并与 `scipy.optimize.linprog` 的 810.0 一致，同时可以核对 `balanced`、`degenerate`、`lp_status`。独立验证就与 `scipy.optimize.linprog`（或直接调本模块 `simplex_lp` 手工铺 LP）对拍 `total_cost`；再检查 `plan` 的行和等于供应量、列和等于需求量（浮点残差在 1e-9 量级）、且 `vogel_cost ≥ total_cost`；造一个产销不平衡的算例，断言 `balanced=False` 且返回矩阵比输入多一行/一列

#### `goal_programming(c, A_ub=None, b_ub=None, targets=None, weights=None, priority=None)`

- **数学形式**：对第 k 个目标 min Σₖ (w_k⁻·d_k⁻ + w_k⁺·d_k⁺) s.t. c_k·x + d_k⁻ − d_k⁺ = targets_k、A_ub·x ≤ b_ub、x ≥ 0、d_k⁻, d_k⁺ ≥ 0；分层时按层号 L 升序逐层求解，第 L 层额外受"已完成层的加权偏差和 ≤ 该层最优值 + 1e-9·(1 + |最优值|)"的锁层约束
- **步骤**：① 把 `c` 规整成 (n_goals, n_vars)，一维输入自动补成单目标；② 校验 `targets` 非 None 且长度等于目标数，`weights` 支持一维（同一权重同时罚 d⁻ 与 d⁺）或 (n_goals, 2)（分别给 w⁻、w⁺），形状/长度不符抛 `ValueError`，出现负权重抛 `ValueError("weights 必须非负")`；③ 把 `priority` 按层号去重升序，得到每层包含的目标下标（`None` 表示全部目标在一层里加权求解）；④ 组装变量数为 n_vars + 2·n_goals 的 LP：硬约束列补 0，每个目标一行等式 `c_k·x + d_k⁻ − d_k⁺ = targets_k`；⑤ 逐层调用 `simplex_lp`，每层把该层目标系数作为新的一行不等式压进下一层的硬约束，右端项取 `layer_value + 1e-9·(1 + |layer_value|)`；某层非 `optimal` 就整体返回该状态并把 `x`/`achieved`/`deviations`/`objective` 全部置 None；⑥ 最后一层解出来后还原 x、d⁻、d⁺，算 `achieved = c·x`、`absolute = |achieved − targets|` 与加权偏差和 `weighted_sum`
- **复杂度**：时间——单层为一次单纯形迭代，分层时为 O(层数) 次调用；空间 O((m_hard + n_goals) × (n_vars + 3·n_goals))
- **参数**：`c` 是 (n_goals, n_vars) 或长度 n_vars 的目标系数；`targets` **不能为 None**（抛 `ValueError("targets 不能为 None：每个目标都需要一个期望值")`），长度不符也抛 `ValueError`；`A_ub`/`b_ub` 是**硬约束**（必须严格满足，与"目标"区分开），给了 `A_ub` 却没给 `b_ub` 抛 `ValueError`，形状不符抛 `ValueError`；`priority` 是长度 n_goals 的整数层号，越小越优先。返回键为 `x` / `achieved`（np.ndarray）/ `deviations`（dict：`d_minus`、`d_plus`、`absolute`、`weighted_sum`）/ `status` / `objective`（float，按用户权重算的加权偏差总和，0 表示全部目标精确达成）。它**没有** `maximize` 开关、没有 `A_eq`、没有决策变量的自由变量口径、没有"只罚单侧的优先级"参数；`bounds` 也不可传，决策变量一律 x ≥ 0。
- **陷阱**：目标规划**必须有硬约束或变量上界**——偏差变量永远能让等式成立，没有硬约束时目标系数为 0 的列会无界（返回 `status="unbounded"`），这不是数值 bug；决策变量沿用 `simplex_lp` 的默认口径 x ≥ 0，需要自由变量要先做平移；**只惩罚单侧偏差时把另一侧权重设为 0 会让解"超调"到任意远处**（目标规划最常见的误用），要限制超调请同时给硬约束；分层求解对容差敏感，锁层用的 1e-9 不能放大，否则低优先级目标会悄悄破坏已经达到的高优先级目标；目标数与硬约束数都算进 LP 行数，目标很多时单纯形表明显变高；`weights` 给 (n_goals, 2) 时**不校验**第二维之外的语义，且 `priority` 若给出负数或非整数会被 `int()` 截断后按层号处理，没有额外的合法性检查
- **怎么检验**：`_self_test()` 有两组可手算的断言——（a）`goal_programming([[1,1],[1,2]], A_ub=[[1,0]], b_ub=[2], targets=[6,10])`，两个目标在 x₁ ≤ 2 下**精确可达**（唯一解 x=(2,4)），因此绝对偏差和必须 ≤ 1e-7、`objective` 绝对值 ≤ 1e-7，这条专门抓偏差变量符号写反（d_minus/d_plus 互换）的错误；（b）分层算例 `[[1,1],[1,−1]]`、A_ub=[[1,0]]、b_ub=[3]、targets=[10,0]、priority=[1,2]，第一层精确达成（偏差 ≤ 1e-7），第二层手算偏差恰为 4.0（容差 1e-6）。独立验证就用这两个手算结论，再自造"目标可达/不可达"两类算例，检查 `achieved = c·x`、`d_k⁺ − d_k⁻ = achieved_k − targets_k` 的符号关系，以及 `deviations["weighted_sum"] == objective`

#### `scenario_robust_lp(scenarios, A_ub=None, b_ub=None, budget=0.1, method="min_max_regret", bounds=None)`

- **数学形式**：给定 K 个情景系数 c_s，min_x (1−β)·mean_s c_sᵀx + β·max_s c_sᵀx（`"min_max"`）；`"min_max_regret"` 则把 max 换成 max_s r_s(x)，其中后悔值 r_s(x) = c_sᵀx − z_s*、z_s* 是情景 s 单独求解的最优值。内部用标量 t 表示最坏情形：约束 c_sᵀx − t ≤ 0（min_max）或 ≤ z_s*（min_max_regret），目标系数 x 部分为 (1−β)·mean_s c_s、t 部分为 β。β=1 退化为教科书的 min-max / min-max-regret，β=0 退化为名义（平均情景）解
- **步骤**：① 校验 `method` 只取 `"min_max"` / `"min_max_regret"`，`budget` 必须落在 [0, 1]；② 逐个情景调 `simplex_lp` 求 z_s*，任一情景非 `optimal` 直接抛 `ValueError`（"无法定义后悔值"）；③ 把决策变量 bounds 复制一份，再给 t 追加一条 `(None, None)` 的自由界；④ 组装 K 行约束 `A_rows[:, :n] = S`、`A_rows[:, n] = −1.0`，右端项在 min_max 下全 0、在 min_max_regret 下取 z_star；硬约束的 t 列补 0 后纵向拼接；⑤ 调 `simplex_lp` 求解；⑥ 用解出的 x **回代重算** `costs = S @ x` 与 `regrets = costs − z_star`，min_max 取 `argmax(costs)`、min_max_regret 取 `argmax(regrets)` 作为 `worst_scenario`（并列取最小下标），并据此返回 `objective` 与 `regret`
- **复杂度**：时间 O((K+1) 次单纯形迭代) / 空间 O((m + K) × (n + 1))
- **参数**：`scenarios` 是 K 个长度 n 的系数向量（list 或 (K, n) 数组），为空抛 `ValueError("scenarios 不能为空")`，某个情景长度与第一个不一致抛 `ValueError`；`A_ub`/`b_ub` 是硬约束（要求等号要自己写 `−a@x ≤ −b` 两行），给了 `A_ub` 却没给 `b_ub` 抛 `ValueError`；`budget` 是 [0, 1] 内的凸组合权重 β，越界抛 `ValueError`；`method` 只支持那两个字符串；`bounds` 长度必须等于 n，否则抛 `ValueError`。返回键为 `x` / `objective`（**最坏情景下的目标值** max_s c_sᵀx，无解时为 None）/ `worst_scenario`（下标，无解时为 −1）/ `regret`（跨全部情景的最大后悔值）。它**没有** `maximize` 开关（一律求最小）、没有 `Gamma` 不确定度预算、没有连续不确定集、没有多个约束同时扰动、没有 CVaR 口径；要概率化约束请用 `chance_constrained_lp`。
- **陷阱**：返回值 `objective` 是**最坏情景值**，而内部最小化的是混合目标，β < 1 时两者不等，论文里报告鲁棒性必须说明用的是哪一个（本实现用最坏情景值）；情景集合是**离散**的，没覆盖到的情况不会被保护，`budget` 只是凸组合权重、**不是** Bertsimas-Sim 意义上的"不确定度预算 Γ"，用错口径会被审稿人抓；z_s* 是各情景**单独**的最优值，某情景不可行或最优值无界时后悔值没有定义，本实现抛 `ValueError` 而不是返回 NaN；只有 A_ub（≤ 约束）时，非负成本问题的"最坏情形最优解"常常就是 x=0，需要下限请自己写成 `−a@x ≤ −b`；β=0 时 t 在目标里系数为 0、又是自由变量，一旦没有硬约束 t 就可以飘到 −∞，LP 返回 `unbounded`，此时函数返回 `x=None` 而 `worst_scenario = −1`——这是**正常返回值不是异常**，而同样的"没有有效约束"在 `chance_constrained_lp` 里却抛 `ValueError`，两个函数的无约束口径并不一致
- **怎么检验**：`_self_test()` 的手算算例是 x+y=4（用 A_ub=[[1,1],[−1,−1]]、b_ub=[4,−4] 表达）、情景 [[1,2],[2,1],[1,1]]：注释写明各情景单独最优值都是 4、min-max 目标值为 6（x=y=2 时两个情景同时取到）、最大后悔值恰为 2；它同时断言 `objective ≥ max_s z_s*`、min-max 的 `objective` 与 `regret` 分别为 6 与 2、min-max-regret（β=1）的 `regret` 为 2。独立验证就自己写 `z_star = [simplex_lp(cs, ...)["fun"] for cs in scen]` 对拍（断言里就是这么算的），再用解出的 x 回代验证 `objective == max_s c_sᵀx` 且 `regret == max_s (c_sᵀx − z_s*)`；沿着 β 从 0 到 1 扫描，检查 `objective` 单调不降

#### `chance_constrained_lp(c, A_ub=None, b_ub=None, sigma=None, alpha=0.95, bounds=None, maximize=False)`

- **数学形式**：min cᵀx s.t. P(aᵢᵀx ≤ bᵢ + ξᵢ) ≥ α，ξᵢ ~ N(0, σᵢ²) 且相互独立；标准化后 P(ξᵢ ≤ bᵢ − aᵢᵀx) = Φ((bᵢ − aᵢᵀx)/σᵢ) ≥ α，得确定性等价 aᵢᵀx ≤ bᵢ − z_α·σᵢ，其中 z_α = Φ⁻¹(α)
- **步骤**：① 先算 z_α = `_norm_ppf(alpha)`（α 不严格落在 (0, 1) 内直接抛 `ValueError`）；② 校验 A_ub 与 b_ub、c 的形状匹配，取出不等式约束条数 m；③ 处理 `sigma`：None 等价于长度 m 的全 0 向量，标量广播到 m 条约束，序列则校验长度等于 m，出现负值抛 `ValueError("sigma（标准差）必须非负")`；④ 把右端项替换成 b_eff = b_ub − z_α·σ（逐元素），转调 `simplex_lp`；⑤ 非 `optimal` 时返回 `x=None`/`objective=None`/`slack=None` 并附带状态；否则还原 x，并用**名义**右端项算 slack = b_ub − A_ub·x
- **复杂度**：时间 O(一次单纯形迭代) / 空间 O(m × n)（与 `simplex_lp` 相同）
- **参数**：`c` 长度 n，`A_ub` 形状 (m, n)，不匹配抛 `ValueError`；`b_ub` 名义右端项，给了 `A_ub` 却没给 `b_ub` 抛 `ValueError`；`sigma` 可为标量、长度 m 的序列或 None（等价于全 0，即退化回确定性 LP），长度不符或为负都抛 `ValueError`；`alpha` 默认 0.95，是约束成立概率而不是显著性水平，必须严格落在 (0, 1) 内；`bounds` / `maximize` 原样透传给 `simplex_lp`。返回键为 `x` / `objective`（**名义目标值** cᵀx，不含任何惩罚项）/ `z_alpha` / `slack`（长度 m 的**原始**约束松弛 b_ub − A_ub·x）/ `status`。它**没有** `A_eq`、没有每行独立的 `alpha`、没有相关系数/协方差矩阵接口、没有多阶段或多期补偿、没有 `method` 开关；相关扰动要自己先算出等效标准差，等式约束要自己拆成两条不等式。
- **陷阱**：确定性等价只对**单侧**约束成立——`a@x ≥ b` 形式必须写成 `−a@x ≤ −b` 再减 z·σ，直接对等式约束套用会算反方向（下限应该是 b + z·σ）；假设是各右端项**独立**正态且分布已知（均值 0、标准差 sigma），相关扰动要用协方差矩阵做联合正态的等效标准差，本实现不做；返回的 `slack` 是相对**名义**右端项的松弛，平均而言会大于 z_α·σ，但单次实现中可以更小，**不要**把它当成"安全余量"来报告；`alpha` 越接近 1、z_α 越大、可行域越小，α ≥ 0.99999 时很多模型会直接变成不可行，而这通常是模型问题而不是算法问题；`_norm_ppf` 是 Acklam 有理逼近（相对误差 < 1.2e-9，只做一次逼近、不做 Halley 修正），尾部（p < 0.01）精度最差，NaN/∞ 分位数会被拒绝而不是返回 inf——否则 b − z·σ 会变成 NaN 并静默污染整张单纯形表
- **怎么检验**：`_self_test()` 给了三组断言——（a）`_norm_ppf(0.95)` 与 1.644854 对拍（容差 1e-6），`chance_constrained_lp` 返回的 `z_alpha` 与 `_norm_ppf` 一致（容差 1e-12）；（b）**sigma = 0 必须与直接调 `simplex_lp` 完全一致**：算例 A_cc = [[−1,−1],[1,−1]]、b_cc = [−2,1]、c_cc = [1,2]，断言状态相同且目标值与解的逐元素最大偏差 ≤ 1e-9（注释写明实测为 0）；（c）sigma = [0.2, 0.1]、alpha = 0.95 时目标值必须严格变差（`objective > lp_cc["fun"] − 1e-12`），专门抓"确定性等价方向搞反"。独立验证就自己用 `math.erf` 的逆（或 `scipy.stats.norm.ppf` 对拍 `_norm_ppf`），再把 b_eff 手算出来直接调 `simplex_lp` 对拍 `x`/`objective`；扰动单调性上，固定其它参数把 sigma 或 alpha 调大，断言可行域收缩、目标值不降

#### `facility_location(cost, demand, n_facilities, capacity=None, max_combinations: int = 200000)`

- **数学形式**：p-median——min Σᵢ demandᵢ · cost_{i, σ(i)} s.t. σ(i) ∈ S、|S| = p（S 是选中的设施集合），无容量时 σ(i) = argmin_{j∈S} cost_{i,j}；有容量时附加 Σ_{i: σ(i)=j} demandᵢ ≤ capacity_j
- **步骤**：① 校验 `demand` 长度等于 `cost` 行数、非负，`p` 落在 [1, n_sites]；② 处理 `capacity`：None 为不限容量，标量广播到所有设施，序列则校验长度等于设施数、非负；③ 用 `math.comb` 算 C(n_sites, p)，超过 `max_combinations` 直接抛 `ValueError`（消息里带实际组合数与上限）；④ 用 `itertools.combinations` 按字典序枚举全部组合；⑤ 无容量时每个客户在组合内取 `min(combo, key=lambda t: (cost[i, t], t))`（成本并列取下标最小者），累加 `demand[i] * cost[i, j]`；⑥ 有容量时改用贪心指派——把全部 (客户, 设施) 对按 `(成本, 客户, 设施)` 升序排序，依次占用仍有剩余容量的设施（判剩余量用 1e-12 容差），出现无法指派的客户就把该组合标为不可行并跳过；⑦ 保留总成本最小的组合（并列时保留先枚举到的那个，比较容差 1e-12）；⑧ 全部组合都不可行才抛 `ValueError("在给定容量下没有任何可行的设施组合，请放宽 capacity")`
- **复杂度**：时间 O(C(n_sites, p) × n_customers × p)（有容量时多一个排序的 log 因子）；空间 O(n_customers × p)
- **参数**：`cost` 是 (n_customers, n_sites) 的单位服务成本矩阵（通常取距离），经 `as_matrix` 校验；`demand` 长度等于行数、非负，不符抛 `ValueError`；`n_facilities` 是设施数 p，`p < 1` 或 `p > n_sites` 抛 `ValueError`；`capacity` 可为 None、标量或长度 n_sites 的序列，长度不符或为负抛 `ValueError`；`max_combinations` 默认 200000，超限抛 `ValueError`。返回键为 `selected`（长度 p 的升序设施下标）/ `assignment`（长度 n_customers 的指派下标，元素取自 `selected`）/ `total_cost` / `n_evaluated`。它**没有** `status` 字段、没有 `maximize` 开关、没有 p 自由（不自动选 p）、没有多目标或最小化最大距离的口径、没有局部搜索；大规模的 p-median 请换 MILP 或贪心 + 局部搜索。
- **陷阱**：**组合爆炸**——这是精确枚举，docstring 明说 n_sites=30、p=5 时 C(30,5)=142506 还能勉强跑，n_sites=50 完全不可行；有容量时"按成本升序贪心"**不保证**给出该组合下的最优指派（本质是装箱式问题），因此结果只是"该组合在贪心指派下的成本"，容量只应作为可行性条件、**不要**用它比较不同组合的最优性，密集容量约束的场景请用运输问题式 LP；`cost` 的行是客户、列是设施，**转置后形状仍合法但语义全变**，是最容易静默出错的点；容量不可行时该组合被直接跳过，所有组合都不可行才抛 `ValueError`（返回值里没有单独的 `status` 字段）；无容量路径的 `total_cost` 用 Python 循环逐个累加，结果可与 `demand @ cost[:, assignment]` 对拍以排除累加顺序误差
- **怎么检验**：`_self_test()` 给了一维等距算例——客户点与候选点都是 `np.arange(5.0)`、等需求、`fcost = |pos[:,None] − pos[None,:]|`，注释写明手算 p=1 最优 = 6（建在 2）、p=2 最优 = 3（如 {1,3} 或 {1,4}），断言容差 1e-9，并核对 `selected` / `assignment` / `n_evaluated`。独立验证就自己用 `itertools.combinations` + 逐客户取 min 重写一遍对拍（对小规模是秒级），断言 `n_evaluated == math.comb(n_sites, p)`、`assignment` 的每个元素都落在 `selected` 里、`total_cost == Σ demandᵢ·cost[i, assignmentᵢ]`。有容量的情形要单独验：先造一个容量宽松到不产生约束的算例，断言它与无容量调用的结果逐位一致；再故意收紧容量，断言要么总成本不降、要么抛 `ValueError`；由于贪心指派本身不是该组合下的最优，容量算例只能与"小规模暴力枚举全部指派"对拍（那才是独立实现），不要拿贪心结果自身当基准


### 3.2 图论与网络 —— `examples/algorithms/graphs.py`

本族口径（写论文时要交代）：`adj` 统一是**出邻接字典** `{u: {v: w}}`；无穷远统一用 `float("inf")`，不用 -1 或 1e18；PageRank 返回概率分布（和为 1），阻尼默认 0.85。依赖仅 numpy + 标准库 + `._common`。

#### `dijkstra`
- **数学形式**：单源最短路。给定有向图 $G=(V,E)$、非负边权 $w(u,v)\ge 0$ 与源点 $s$，求 $d(v)=\min_{P:s\to v}\sum_{(u,x)\in P} w(u,x)$，不可达记 $d(v)=+\infty$；同时给出最短路径树的前驱 $\pi(v)$。
- **步骤**：收集节点全集（键 + 所有出现过的邻居，含无出边的节点）→ 统一校验每条边权重为有限且非负（负权直接抛 `ValueError`，零权允许）→ 初始化 `dist=inf`、`prev=None`、`dist[src]=0.0` → 二叉堆放 `(0.0, 0, src)`，自增序号打破节点不可比较的平局 → 循环弹堆：已定型（`done`）或过期记录（`d > dist[u]`）则跳过，否则定型并对其出边松弛，成功松弛则更新 `dist`/`prev` 并压入新记录（惰性删除）。
- **复杂度**：时间 O((V + E) log V)（惰性删除使堆操作常数略大）/ 空间 O(V + E)。
- **参数**：`adj`（出邻接字典 `{u: {v: w}}`，`w` 是 u→v 边权且必须 >= 0；两个端点会自动纳入节点全集，即使某节点没有出边，必填）；`src`（源点，必填；若不在 `adj` 中仍会被当作孤立源点处理）。返回字典键：`"dist"`（`{node: float}`，源点到 node 的最短距离，不可达为 `float("inf")`）、`"prev"`（`{node: Optional[node]}`，最短路上的前驱；源点与不可达点均为 `None`）。
- **陷阱**：① 负权会静默给出错误答案，所以本实现显式抛 `ValueError`，有负权请改 Bellman-Ford / Johnson；零权允许。② 返回的 `prev` 只能还原**一棵**最短路径树，存在多条等长最短路时拿到哪一条取决于堆的弹出顺序，不要声称"the"唯一路径。③ 键是**节点对象本身**而非下标；节点若是 numpy 整数请先转 `int`，否则 `0` 与 `np.int64(0)` 混用会查不到键。
- **怎么检验**：① 与 Floyd-Warshall 交叉验证：同一张图上"以每个点为源的 Dijkstra"与 `floyd_warshall` 的全点对距离取最大绝对偏差，`_self_test()` 记录为 `floyd_vs_dijkstra_max_dev`（若一边有限一边无穷则记为 `inf`，可直接断言该偏差为 0）。② 用自测的 5 节点带权有向图手算：A→B=1、B→C=2、C→D=1、D→E=3，故 `dij_a_e` 应为 1+2+1+3=7；`reconstruct_path(dj["prev"], "A", "E")` 应给出 ["A","B","C","D","E"]，而反向 `reconstruct_path(dj["prev"], "E", "A")` 因不可达应返回 `[]`（自测键 `dij_unreach`）。③ 边界：`src` 不在 `adj` 中不报错；给一条负权边应立即抛 `ValueError`；不可达点的 `dist` 为 `inf` 且 `prev` 为 `None`。

#### `floyd_warshall`
- **数学形式**：全源最短路动态规划。$D^{(0)}_{ij}=W_{ij}$，$D^{(k)}_{ij}=\min\!\big(D^{(k-1)}_{ij},\; D^{(k-1)}_{ik}+D^{(k-1)}_{kj}\big)$，$k=1..n$，收敛于所有点对最短路；同时维护"下一跳矩阵"用于路径还原。
- **步骤**：`np.asarray(W, float)` → 校验二维、方阵、非空、无 NaN、无 `-inf` → `dist = W.copy()`（不改调用方数组）→ `next_node` 全填 -1，对所有有限元 `(i,j)` 置 `next_node[i,j]=j`（直接可达时下一跳就是终点），对角线强制 -1 → 对每个中间点 k 做向量化更新 `through = dist[:,k][:,None] + dist[k,:][None,:]`，只在 `through < dist` 处更新 `dist`，并把这些位置的下一跳改成 `next_node[i,k]`（即 i 走向 k 的第一步），再填对角线为 -1 → 返回前检查对角线，若 `dist[i,i] < 0` 抛 `ValueError`（负环）。
- **复杂度**：时间 O(n^3) / 空间 O(n^2)。
- **参数**：`W`（(n,n) 距离矩阵，`W[i,j]` 为 i→j 的直接距离，无边用 `np.inf`；对角线应为 0，若给出非 0 对角元本实现按给定值处理、不做覆盖，必填）。返回字典键：`"dist"`（`np.ndarray(n,n)` 最短路距离）、`"next_node"`（`np.ndarray(n,n)` int，i→j 最短路上从 i 出发的**下一跳**；不可达或 `i==j` 时为 -1，可直接喂给 `reconstruct_path`）。
- **陷阱**：① 负环会让 `dist[i,i] < 0`，此时最短路无下界；本实现返回前检查对角线并抛 `ValueError`，而不是返回看似正常的矩阵。② 用 0 表示"无边"是错的：0 权边与无穷远必须区分，填矩阵时无边请填 `np.inf`。③ 浮点加法下 `inf` 会传播，`inf + (-inf)` 是 NaN，所以不允许出现 `-inf`。④ 内部对输入数组做 `copy()`，调用方数组不受影响（文档措辞"原地修改传入的副本"易被误读为改动了调用方数组）。
- **怎么检验**：① 交叉验证：与逐源 `dijkstra` 比对全点对距离，自测取最大绝对偏差 `floyd_vs_dijkstra_max_dev`，应为 0。② 手算实例：自测的 A..E 图（`names=["A","B","C","D","E"]`）上 `dist[0,4]`（A→E）应为 1+2+1+3=7，且 `reconstruct_path(fw["next_node"], 0, 4)` 应还原出下标序列 [0,1,2,3,4]。③ 边界：含 NaN、含 `-inf`、非方阵、空矩阵都应抛 `ValueError`；构造一个负环（如 0→1 权 -1、1→0 权 -1）应抛"检测到负环"。

#### `reconstruct_path`
- **数学形式**：从最短路前驱/下一跳数据结构还原一条 $s\to t$ 的弧序列。dict 口径沿 $\pi$ 反向回溯 $t\to s$ 后反转；矩阵口径按 `next_node[cur, dst]` 从 $s$ 正向推进。
- **步骤**：`src == dst` 直接返回 `[src]` → 若 `prev_or_next` 是 `np.ndarray`：校验 `src`/`dst` 为整数下标且不越界，从 `cur=src` 反复取 `step = next_node[cur, target]`，`step < 0` 返回 `[]`（不可达），`step == target` 返回路径，最多走 `n+1` 步，超限抛"含环" → 否则按 dict 口径：校验 `src` 在 `prev` 中，从 `dst` 不断取前驱直到 `src` 再反转，前驱为 `None` 时返回 `[]`，超过 `len(prev)+1` 步抛"含环"。
- **复杂度**：时间 O(V) / 空间 O(V)。
- **参数**：`prev_or_next`（二选一，必填：dict 即 Dijkstra 的 `prev`，`node -> 前驱`，从 dst 反向回溯；`np.ndarray` 即 Floyd 的 `next_node`，(n,n) 下一跳矩阵，从 src 正向推进）；`src`（起点，下标或节点对象）；`dst`（终点，下标或节点对象）。返回：`list`，形如 `[src, ..., dst]`；不可达返回 `[]`；`src == dst` 返回 `[src]`。（注：本函数返回裸 list，不是字典。）
- **陷阱**：① 两种口径**不能混用**：把 `next_node` 当 `prev` 传会得到反向乱序结果，本函数靠类型判断区分、不会报错，所以更要小心。② Floyd 的 `next_node` 口径要求 `src`/`dst` 是 0..n-1 的整数下标，非整数直接抛 `ValueError`；字符串节点请自己维护 `name -> index` 映射或改用 dict 口径。③ 不可达返回空列表而不抛异常，调用处必须显式判 `len(path) == 0`，否则下游把空路径当合法解会静默出错。
- **怎么检验**：① 回代验证：`p = reconstruct_path(prev, s, t)` 后逐边累加 `adj[p[k]][p[k+1]]`，应等于 `dijkstra(...)["dist"][t]`。② 自测已给出两个可断言实例：`dij_path_a_e` = reconstruct_path(dj["prev"], "A", "E") 为 ["A","B","C","D","E"]；`floyd_path_0_4` = reconstruct_path(fw["next_node"], 0, 4) 为 [0,1,2,3,4]。③ 停机性：不可达返回 `[]`，`src == dst` 返回 `[src]`；人为构造自指的 `next_node` 应抛"含环"而不是死循环。

#### `kruskal_mst`
- **数学形式**：最小生成树（森林）。无向图 $G=(V,E)$ 上求 $T\subseteq E$ 使 $\sum_{e\in T} w_e$ 最小且 $T$ 无环；等价于拟阵（图拟阵）上的贪心。权重可为负（负权时 MST 仍良定义）。
- **步骤**：校验 `n_nodes` 为正整数 → 规范化每条边为 `(min(u,v), max(u,v), w)`（保证结果可复现），校验端点落在 `0..n-1`、权重有限，**自环 `u==v` 直接跳过** → 按 `(w, u, v)` 升序排序 → 并查集（路径压缩 + 按秩合并）逐边扫描：两端点已在同一分量则丢弃，否则选入并合并 → 选满 `n_nodes-1` 条立即停止。
- **复杂度**：时间 O(E log E)（排序主导，并查集近似 O(E α(V))) / 空间 O(V + E)。
- **参数**：`n_nodes`（节点数，节点编号 `0..n_nodes-1`，必须正整数，必填）；`edges`（边列表 `[(u, v, w), ...]`，无向、权重须有限（可为负），必填）。返回字典键：`"total_weight"`（float，选中边权重之和）、`"edges"`（`[(u, v, w), ...]`，按权重升序、`u < v` 归一化）。
- **陷阱**：① 自环 `(u, u, w)` 必须跳过——即使不跳过，并查集也会判"已在同一分量"而静默丢弃，行为没错但语义含糊，本实现显式忽略更清楚。② 多重边（同 u,v 不同 w）允许，算法自动只取较小的那条。③ 不连通图**不报错**，只返回最小生成森林，`edges` 长度 < `n_nodes-1`；很多建模代码忘了这一步，直接拿 `total_weight` 当"连通成本"会得到偏小的错误结论；请用 `len(edges) == n_nodes - 1` 判断连通。④ 权重为负时 MST 仍良定义（会优先选负权边），不要误以为权重必须为正。
- **怎么检验**：① 交叉验证：与 `prim_mst` 在同一张图（同模块自测的 7 节点教材图）上比较总权重，自测要求 `abs(prim - kruskal) <= 1e-9`，并另外断言边数为 `V-1 = 6`。② 手算实例：自测图边集 (A,B,7)(A,D,5)(B,C,8)(B,D,9)(B,E,7)(C,E,5)(D,E,15)(D,F,6)(E,F,8)(E,G,9)(F,G,11)，最小生成树权重为教材值 **39**（自测键 `mst_weight`，逐边求和亦应为 39）；`mst_edge0` 是排序后的首条边，按 `(w,u,v)` 应为 `[0, 3, 5]`。③ 不连通性：删掉一条桥上的边后 `len(edges) < n-1` 且不抛异常；`total_weight` 等于逐边重算之和。

#### `max_flow_edmonds_karp`
- **数学形式**：最大流。给定有向容量 $c(u,v)\ge 0$、源 $s$、汇 $t$，求流量 $f$ 使 $0\le f(u,v)\le c(u,v)$、满足流量守恒（除 $s,t$ 外净流出为 0）且 $|f|=\sum_v f(s,v)$ 最大；由最大流最小割定理 $|f^\*|=\min_{S\ni s,\,t\notin S}\sum_{u\in S,v\notin S} c(u,v)$。
- **步骤**：`source == sink` 直接抛 `ValueError` → 校验容量字典（键与邻居都纳入节点全集、容量有限且非负）→ 映射为下标，填容量矩阵时**平行边累加** → `res = cap.copy()` 作为残量矩阵 → 循环：BFS 在残量网络（`res[u] > 0`）上找源到汇的**边数最少**增广路并记录前驱 → 沿前驱回推瓶颈 `min res[u,v]` → 正向减、**反向加** `res[v,u] += bottleneck` → 累加 `max_flow` → 无增广路时停止 → 净流量 `f = cap - res`，只输出 `f > 0` 的边。
- **复杂度**：时间 O(V E^2)（BFS 增广次数 O(VE)，每次 O(E)）/ 空间 O(V + E)。
- **参数**：`capacity`（有向容量字典 `{u: {v: cap}}`，只写 u→v 即单向边；要建无向边必须**两个方向都写同样容量**，不是自动的；必填）；`source`（源点，必填）；`sink`（汇点，必须与 source 不同，必填）。返回字典键：`"max_flow"`（float）、`"flow"`（`{u: {v: 流量}}`，只包含残量网络中净流量为正的边；每条边满足 `0 <= flow[u][v] <= capacity[u][v]`）。
- **陷阱**：① **反向边必须有**：残量网络里 `res[v][u] += pushed` 是正确性关键，只写正向边会得到偏小的流量却不报错。② 容量可以为 0（表示不允许），但**不能为负**；负容量会让残量网络出现可无限增广的假象。③ 平行边要自己合并成一条（字典天然合并，此处实现是显式累加），否则后写的会覆盖先写的。④ 浮点容量下"无增广路"用严格 `> 0` 判断，若容量是 1e-16 级的小量，数值噪声可能被当成可增广；竞赛数据请先做单位统一与量级缩放。⑤ `source == sink` 无意义，这里直接抛 `ValueError`（有些实现会返回 inf）。
- **怎么检验**：① 自测算例（CLRS 经典例）：`s` 及各中间点、汇点的容量如图，最大流应为 **23**（自测键 `max_flow`）；同时与 `min_cut_edges` 的割容量对照（自测键 `min_cut_cap` / `min_cut_n`），定理要求两者相等。② 守恒性检查：对 `flow` 的每条边统计每个节点的净流出，除源汇外应全为 0；源点净流出应等于 `max_flow`。③ 上界检查：容量全部置 1 的单位容量图上 `max_flow` 不应超过源点出边容量之和，也不应超过汇点入边容量之和。④ 边界：`source == sink`、source/sink 不在容量字典中、含负容量，都应抛 `ValueError`。

#### `min_cut_edges`
- **数学形式**：最大流最小割定理的构造侧。先求最大流得到残量网络，取源侧可达集 $S=\{v: \text{残量网络中 } s\to v \text{ 有路}\}$，输出所有 $u\in S, v\notin S$ 的**原始有向边**；其容量之和恰等于最大流。
- **步骤**：`source == sink` 抛 `ValueError` → 校验容量字典与 source/sink 归属 → 填容量矩阵（平行边累加）→ 调用 `max_flow_edmonds_karp` 拿最大流 → 用返回的 `flow` 重建残量矩阵 `res = cap - f`（正向减、反向加）→ 从 source 用显式栈 DFS 在残量正边（代码阈值 `res > 1e-12`）上求源侧可达集 → 收集所有"u 在 S、v 不在 S 且原始容量 > 0"的边 `(u, v, cap)` → 按 `(str(u), str(v))` 排序返回。
- **复杂度**：时间 O(V E^2)（含最大流）/ 空间 O(V + E)。
- **参数**：`capacity`（同 `max_flow_edmonds_karp` 的 `{u: {v: cap}}`，必填）；`source`（源点，必填）；`sink`（汇点，必填）。返回：`list`，每项 `(u, v, cap)`（原始有向边的容量）；source 与 sink 之间无路时返回 `[]`。（注：本函数返回裸 list，不是字典。）
- **陷阱**：① 返回的是**有向边** (S → V\\S)，双向图里只会列出从 S 出去的那个方向，这正是定理要的割，不要"补全"成两条。② 最小割**不唯一**：这里给出的是"离源点最近"的那个最小割（源侧集合最小），论文里断言割边集合唯一会被质疑。③ 割容量必须用**原始容量**求和，不要用残量求和（用残量会得到 0）。④ 原图有平行边时容量已合并，返回的 cap 是合并后的值，和输入逐条对不上。⑤ 可达性判定代码里用的是 `res > 1e-12` 的数值阈值（`max_flow_edmonds_karp` 内部用严格 `> 0`），小容量算例上两者的口径差异需要注意。
- **怎么检验**：① 定理验证：`sum(c for _, _, c in cut)` 应等于 `max_flow_edmonds_karp(...)["max_flow"]`，自测在 CLRS 例上把两者分别记为 `min_cut_cap` 与 `max_flow`（该例最大流为 23）。② 饱和性验证：对返回的每条割边，其流量应等于容量（可用 `flow[u][v] == cap` 检查）。③ 去源侧验证：把 S 内的节点标记出来后，逐条检查不存在"容量 > 0 且从 S 指向 V\\S"的边被漏掉。④ 边界：source 与 sink 之间无路时返回 `[]`；`source == sink` 抛 `ValueError`。

#### `tsp_nearest_neighbor`
- **数学形式**：旅行商问题（TSP）的最近邻贪心构造。求城市集合上的哈密顿回路 $\pi$ 使 $\sum_k d(\pi_k,\pi_{k+1})$（含回到起点）最小；本函数只给出一个可行解（贪心上界），不保证最优。
- **步骤**：校验距离矩阵（方阵、非负、无 NaN）→ 校验 `start` 在 `0..n-1` → `unvisited` 集合去掉 `start`，`cur = start` → 循环：每次在 `unvisited` 中选 `(D[cur, c], c)` 最小的城市（按下标 tie-break，确定性）加入 `tour` 并前移 → 全部访问后按**闭环**累加长度 `sum_k D[tour[k], tour[(k+1) mod n]]`。
- **复杂度**：时间 O(n^2) / 空间 O(n)。
- **参数**：`dist`（(n,n) 距离矩阵，`dist[i,j]` 为 i→j 距离；非对称矩阵也支持，必填）；`start`（起始城市下标，默认 `0`）。返回字典键：`"tour"`（`[i0, i1, ..., i_{n-1}]`，**不含重复的起点**，闭环由首尾隐含）、`"length"`（float，按闭环计算的总长度）。
- **陷阱**：① 最近邻可能给出比最优解差 20%~25% 的回路（n 大时），必须再接 2-opt 或 Or-opt 改进，不要直接把它的长度当"TSP 最优值"。② **起点不同结果不同**，竞赛中应报告"对每个起点各跑一次取最好"的结论，而不是随手取 `start=0`。③ 非对称矩阵（如单向道路）下闭合回路方向有意义，本函数沿行进方向累加，不做对称化处理。④ `n == 0` 抛 `ValueError`；`n == 1` 返回 `[0]`、长度 0。
- **怎么检验**：① 闭式解算例（自测）：10 个城市均匀分布在单位圆上，周长最优值为 $2n\sin(\pi/n)$，`_self_test()` 记为 `tsp_optimal_len`（$n=10$ 即 $20\sin(\pi/10)\approx 6.1803$）；该算例上最近邻恰好命中，故 `tsp_nn_len` 应等于闭式值。② 反例算例（自测）：8 个给定坐标点，穷举最优值 `tsp8_brute_opt_len = 26.484006136241458`，最近邻的 `tsp8_nn_len` 应**严格大于**它——用它来证明"最近邻次优"。③ 结构检查：`tour` 应是 `0..n-1` 的排列、长度等于 n、不含重复起点；`length` 应等于按 `tour` 闭环重算的距离和。④ 多起点：对每个 `start` 各跑一次取最小长度，应不劣于任何单起点结果。

#### `tsp_two_opt`
- **数学形式**：2-opt 局部搜索。对回路中 $0\le i<j<n$，用边 $(i,j)$ 与 $(i{+}1,j{+}1)$ 替换 $(i,i{+}1)$ 与 $(j,j{+}1)$（即反转 $tour[i{+}1..j]$），只接受长度严格下降的交换，迭代至无改进。
- **步骤**：校验距离矩阵 → `n <= 3` 直接原样返回（`improved=False`）→ 校验 `len(tour) == n` → 取当前回路与长度 → 最多 `max_pass` 轮：对全部 `0 <= i < j < n`（跳过 `j - i == 1` 的相邻交换）构造候选 `best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]`，**重算整条回路长度**，若 `cand_len < best_len - 1e-12` 则接受并继续扫描，一轮内无任何改进则提前退出。
- **复杂度**：时间 O(n^2) 每次扫描，最多 `max_pass` 轮（实践上常几轮就停）/ 空间 O(n)。
- **参数**：`tour`（初始回路，不含重复起点，如 `tsp_nearest_neighbor` 的输出，必填）；`dist`（(n,n) 距离矩阵，规模须与 tour 一致，必填）；`max_pass`（最大扫描轮数上限，每轮尝试所有 (i,j) 对，默认 `100`）。返回字典键：`"tour"`（list，改进后的回路）、`"length"`（float）、`"improved"`（bool，相比输入回路长度**严格变小**为 True，相等为 False）。
- **陷阱**：① 2-opt 是**局部最优**，`improved=False` 不代表已最优，城市多了必须配合多起点重启。② 增量式赚量算法（只算 delta）在**非对称**矩阵上不成立（反转一段会改变两端方向）；本实现直接重算整条回路长度，因此对非对称矩阵也正确但更慢。③ 传入的 `tour` 若含重复点或缺点，本函数不校验，会静默给出无意义结果；请先用 `tsp_nearest_neighbor` 或自己保证它是 `0..n-1` 的一个排列。④ 浮点比较用严格小于（阈值 `1e-12`），等长回路的交换不会发生，因此相同输入必得相同输出。
- **怎么检验**：① 单调性：返回的 `length` 必须 `<=` 输入回路的闭环长度；`improved` 为 True 时严格更小（文档承诺"严格变小"，可用闭式重算复核）。② 幂等性：把输出 `tour` 再喂回去，`improved` 应为 False、`length` 应不变（局部最优的机器精度内表现）。③ 闭式/穷举对照：自测的 10 城圆算例上 `tsp_2opt_len` 应等于闭式最优 `tsp_optimal_len = 2n sin(pi/n)`；但 8 城反例上 `tsp8_2opt_len` 仍**严格大于**穷举最优 26.484006136241458（自测键 `tsp8_2opt_improved` 记录它确实产生了改进）——这两例合起来说明"2-opt 改进过 ≠ 最优"。④ 非对称性：用非对称矩阵跑一遍，验证长度按行进方向重算而不是用对称化后的矩阵。

#### `pagerank`
- **数学形式**：有向图上的重要性向量（马尔可夫链稳态）。按行归一化得转移矩阵 $M$（$M_{ij}=w_{ij}/\sum_j w_{ij}$），幂迭代 $r \leftarrow d\,M^{\top} r + d\,\frac{\text{悬挂点质量}}{n} + \frac{1-d}{n}$，收敛到 $\sum_i r_i = 1$ 的稳态分布；悬挂点（无出边的点）的概率质量被均匀重分配。
- **步骤**：校验 `adj` 为字典、`damping ∈ [0,1)`、`tol > 0` → 用 `_ordered_nodes` 得到确定性的节点顺序（优先 `sorted`，不可比较时退化为扫描顺序）→ 空图返回 `np.zeros(0)` → 按行累加权重并归一化（边权必须非负有限，否则抛 `ValueError`）→ `dangling = out_w <= 0` → `r` 初始化为 $1/n$ → 迭代：`dangling_mass = r[dangling].sum()`，`new_r = d*(M.T @ r) + d*dangling_mass/n + (1-d)/n`，若 L1 变化 `< tol` 则接受并跳出 → 结尾再除以总和，抹掉浮点累积误差保证严格和为 1。
- **复杂度**：时间 O(max_iter * E) / 空间 O(V + E)。
- **参数**：`adj`（出邻接字典 `{u: {v: w}}`，`w` 是**转移强度**，非负、不必归一化，本函数按行归一化，用 1.0 表示普通无权有向边；必填）；`damping`（阻尼系数 d，默认 `0.85`）；`tol`（L1 收敛阈值，默认 `1e-10`）；`max_iter`（最大迭代次数，默认 `200`）。返回：`np.ndarray`，形状 (n,)，**下标对齐 `_ordered_nodes` 的顺序**（优先 `sorted(adj 中所有节点)`，节点不可比较时退化为扫描顺序），向量非负且严格和为 1。（注：本函数返回裸数组，不是字典。）
- **陷阱**：① **节点顺序**：`r[i]` 对应 `sorted(adj)` 的第 i 个节点（不可比较时为扫描顺序）；只依赖"和为 1"最安全，要和节点名对应就必须自己按同样顺序取节点；混用 int/str 键时排序失败并静默退回扫描顺序。② 悬挂点不处理会漏概率质量，迭代结果和不为 1（最常见的 bug）。③ 阻尼系数过小（如 0.1）时结果接近均匀分布，看起来"没区分度"；论文里要报告 d。④ 达到 `max_iter` 仍未收敛**不会报错**，只返回当前向量；建议同时打印残差。⑤ 每条边权重必须非负，负权重会让幂迭代失去概率意义。
- **怎么检验**：① 不变量：`sum(pagerank(...))` 应为 1（自测键 `pr_sum`）。② 对称闭式解：3 节点环 `a→b→c→a` 的稳态必为均匀分布，自测断言 `abs(pr - 1/3).max()` 记为 `pr_cycle_max_dev`，应接近 0（自测用 `damping=0.85, tol=1e-12, max_iter=500`）。③ 悬挂点算例：`{"a": {"b":1,"c":1}, "b": {"c":1}, "c": {}}` 上自测记录 `pr_dangling_sum`（应仍为 1）与 `pr_b_gt_a`（b 的分数应大于 a）。④ 节点对齐：用 `dict(zip(sorted(adj), pr))` 建立"名字 → 分数"映射再做业务判断，不要直接按整数下标解释。

#### `connected_components`
- **数学形式**：无向连通分量划分。把有向邻接字典**无向化**（忽略方向与权重）后求等价类，得到标签映射 $\ell: V \to \{0,1,\dots,k-1\}$。
- **步骤**：校验 `adj` 为字典 → `_ordered_nodes` 定序 → 构造无向邻接表（`undirected[u].append(v)` 且 `undirected[v].append(u)`）→ 标签数组初始 -1，按 `nodes` 顺序用**显式栈** DFS（非递归，避免深度问题）标记整块，每块 `comp += 1`。
- **复杂度**：时间 O(V + E) / 空间 O(V + E)。
- **参数**：`adj`（邻接字典 `{u: {v: w}}`，方向被**忽略**当作无向图、边权也忽略，必填）。返回：`list of int`，长度 = 节点数，`labels[i]` 是第 i 个节点（按 `_ordered_nodes` 顺序，即优先 `sorted(adj 中所有节点)`）的分量编号；编号从 0 开始、按分量中最早出现的节点排序。（注：本函数返回裸 list，不是字典。）
- **陷阱**：① 参数名虽是 `adj`，这里按**无向**处理；"强连通分量"是另一个概念（Tarjan / Kosaraju），不要混用结论。② 只出现在别人邻接表里的节点也会被分配标签，孤立点不会消失。③ 返回的是标签数组而不是"分量列表"，因为下游多用标签查表；需要分组时用 `collections.defaultdict(list)` 自己聚合。④ **下标顺序**：`labels[i]` 对应 `sorted(adj 中所有节点)` 的第 i 个（不可比较时退回扫描顺序），用 `dict(zip(sorted(adj), labels))` 才是可靠的"名字 → 分量"映射；直接把 `labels[i]` 当"第 i 号节点"用，在键不是 0..n-1 时会静默错位。
- **怎么检验**：① 自测算例：`{"0":{"1":1},"1":{"2":1},"2":{},"3":{"4":1},"4":{},"5":{}}`，节点排序后为 `["0".."5"]`，应得到 `cc_labels = [0,0,0,1,1,2]`、`cc_count = 3`（自测键）。② 结构不变量：`max(labels)+1` 等于分量个数；同一边两端的标签必须相同；标签取值恰为 `0..k-1` 连续无空缺。③ 方向不变性：把所有边反向，`labels` 应完全不变。④ 连通性应用：用 `labels[u] == labels[v]` 判断两点是否连通，与独立实现的 BFS 可达性结论逐对核对。

#### `prim_mst`
- **数学形式**：最小生成树（邻接矩阵 / Prim 版本）。从任一顶点出发，反复取"已入树集合 $S$ 与未入树集合 $V\setminus S$ 之间权重最小的横切边"并入新顶点，得到 $\sum_{e\in T} w_e$ 最小的生成树。
- **步骤**：用 `_weight_matrix` 把输入统一成"已对称化、对角线置 0、无边为 inf"的权重矩阵 W 与节点顺序 → 空矩阵返回空结果 → `in_tree[0] = True`，把下标 0 的所有有限邻边压入堆 → 循环：弹出最小 `(w, u, v)`，`v` 已在树中则跳过，否则入树、记录边 `[u, v, w]`、累加总权重，并把 `v` 的所有未入树有限邻边压堆 → 循环结束若 `len(edges) != n-1` 抛 `ValueError`（图不连通）。
- **复杂度**：时间 O(E log E)（E 为有效边数，等价 O(E log V)）/ 空间 O(V + E)。
- **参数**：`adj`（`n x n` 邻接矩阵，`inf` 表示无边、对角线视为 0；也接受 `{u: {v: w}}` 邻接字典，此时按无向处理并映射为下标（见 `_weight_matrix`）；必填）。返回字典键：`"edges"`（list，每项 `[u, v, w]`，端点是 `_weight_matrix` 给出的**整数下标**）、`"total_weight"`（float，总权重）。
- **陷阱**：① 与 `kruskal_mst` 的输入格式不同（这里是矩阵），但同一张图上两者总权重必须一致，本模块自测做了这项交叉验证。② 图不连通时 `raise ValueError`，不会返回"最小生成森林"（这与 `kruskal_mst` 的行为不同）。③ 返回的端点是下标而非原节点名；矩阵输入时下标即原下标，字典输入时须用同一顺序解读。④ 负权边不会被拒绝（Prim 对负权仍正确，只是"最小"的含义要自己确认）。⑤ 文档说字典输入"按键顺序映射为下标"，实际 `_weight_matrix` 走 `_ordered_nodes`（优先 `sorted`），键顺序不可比较时才退化为扫描顺序。
- **怎么检验**：① 交叉验证：与 `kruskal_mst` 在同一 7 节点教材图上比较总权重，自测断言偏差 `<= 1e-9`（键 `graphs_prim_vs_kruskal_dev`），并断言 `graphs_prim_n_edges == 6 = V-1`。② 数值实例：自测断言逐边求和 `sum(e[2] for e in prim["edges"])` 等于 39.0（不复用 `total_weight`，独立重算）。③ 结构不变量：边数 = V-1、无环、端点下标落在 `0..n-1`；`total_weight` 等于逐边求和。④ 不连通性：矩阵里挖掉一条割边后应抛 `ValueError` 而不是返回森林。

#### `degree_centrality`
- **数学形式**：度中心性（按**无向**图）。无权口径 $C_D(v)=\dfrac{|\{u: (v,u)\in E\}|}{n-1}$；带权口径 $C_D^w(v)=\dfrac{\sum_u w(v,u)}{\max_u \sum_x w(u,x)}$（按**最大强度**归一化，不是 $(n-1)\max w$）。
- **步骤**：`_weight_matrix` 统一成对称权重矩阵与节点顺序 → 空图返回空结果 → 把 inf 当 0（`finite`）→ `weighted=True` 时取每行权重和、除以最大行和（最大行和为 0 时全取 0）；`weighted=False` 时按"非零邻居计数 / (n-1)"（`n <= 1` 时全取 0）→ 组装 `{节点: 中心性}` 并按 `(-中心性, 原下标)` **稳定**降序排 `ranking`。
- **复杂度**：时间 O(V^2)（矩阵运算）/ 空间 O(V^2)（含输入的矩阵副本）。
- **参数**：`adj`（`{u: {v: w}}` 邻接字典或 `n x n` 邻接矩阵，`inf` 表示无边；必填）；`weighted`（`False` 用"不同邻居个数 / (n-1)"，`True` 用"相邻权重之和 / 最大权重和"，默认 `True`）。返回字典键：`"centrality"`（`{节点: float}`，取值 `[0,1]`）、`"ranking"`（list，节点按中心性降序，同分按 `_weight_matrix` 的节点顺序稳定排列）。
- **陷阱**：① 方向被忽略（有向图的出/入度中心性请自己按行/列统计）。② `weighted=True` 归一化用的是**最大强度**，不是 `(n-1)*max_w`；因此中心点在星形图上恰好为 1，而"绝对强度"没有单位意义。③ 权重为负时可能让强度为 0 甚至为负，此时分母取最大值仍可运行，但解释失效。④ 返回字典的键是原节点，若节点不可 JSON 序列化（如 tuple）不能直接写进黄金值。
- **怎么检验**：① 星形图闭式解（自测，4 节点星形、边权 2/4/6）：无权口径中心点应为 1、叶子应为 $1/(n-1)=1/3$；带权口径中心点（强度 12）应为 1、节点 3（强度 6）应为 $6/12=0.5$；`ranking[0]` 必须是中心点。（自测对这四件事都做了断言。）② 结构不变量：`centrality` 的值域在 `[0,1]`；带权口径下最大强度节点取值为 1；`ranking` 是中心性的非增序列。③ 无权/带权口径交叉：单位权矩阵上两种口径的排序应一致。

#### `closeness_centrality`
- **数学形式**：紧密中心性 $C_C(v)=\dfrac{\text{从 } v \text{ 可达的节点数}}{\sum_{u \text{ 可达}} d(v,u)}$（不取倒数，量纲为 1/距离）；不可达对**既不计入分子也不计入分母**，无可达点时取 0。
- **步骤**：校验 `adj` 为字典 → `_ordered_nodes` 定序 → 空图返回空结果 → 对每个节点 `s` 跑一次 `dijkstra(adj, s)`，跳过 `t == s`，对有限距离的 `t` 统计 `reach += 1`、`total += d` → `cent[s] = reach / total`（`total <= 0` 时取 0）→ 按 `(-中心性, `_ordered_nodes` 顺序)` 稳定降序排 `ranking`。
- **复杂度**：时间 O(V * (E log V)) / 空间 O(V + E)。
- **参数**：`adj`（`{u: {v: w}}` 邻接字典，边权必须非负（直接交给 `dijkstra`）；必填）。返回字典键：`"centrality"`（`{节点: float}`）、`"ranking"`（list，按中心性降序、同分按 `_ordered_nodes` 顺序稳定排列）。
- **陷阱**：① **不可达对既不计入分子也不计入分母**（不是"距离记 0"，也不是"记无穷大"）；因此孤立点中心性为 0，而"只能到达很少但很近的点"的节点中心性可能偏高——这是本实现的约定，与"只在最大连通分量上计算"的教科书写法不同。② 有向图会得到非对称结果：谁都能到的节点中心性高，别当成无向图的结论。③ 中心性的量纲是 1/距离，不要跨算例比较绝对值。
- **怎么检验**：① 路径图手算（自测 P4，节点 "0"-"3"，单位权）：中间点 "1" 到其余三点距离 1、1、2，故 $3/(1+1+2)=0.75$（自测键 `graphs_closeness_p4_mid`）；端点 "0" 的距离为 1、2、3，故 $3/(1+2+3)=0.5$（键 `graphs_closeness_p4_end`）；并断言中间点 > 端点。② 与全源最短路交叉验证：用 `dijkstra` 的 `dist` 独立重算 `reach/total`，逐节点比对。③ 边界：孤立点（有键但无出边、也无入边）中心性应为 0；两节点互连时每个点应为 $1/1=1$。

#### `betweenness_centrality`
- **数学形式**：介数中心性（Brandes，按**无权**最短路计数、无向口径）
  $C_B(v)=\dfrac{1}{C(n-1,2)}\sum_{s\ne v\ne t}\dfrac{\sigma_{st}(v)}{\sigma_{st}}$，
  其中 $\sigma_{st}$ 是最短路条数、$\sigma_{st}(v)$ 是经过 $v$ 的最短路条数；无向图每对点被数两次故先除以 2，再用 $C(n-1,2)=(n-1)(n-2)/2$ 归一化。
- **步骤**：校验 `adj` 为字典 → `_ordered_nodes` 定序 → 构建**无向无权**邻接表（忽略边权与方向，自环跳过，重复边用 `(min,max)` 去重）→ 对每个源点 `s`：BFS 记录 `depth`、最短路条数 `sigma`、前驱表 `pred` 与出栈序 `stack` → 逆序弹栈累加依赖量 `delta[v] += (sigma[v]/sigma[w]) * (1 + delta[w])`，`w != s` 时把 `delta[w]` 累加进 `bc[w]` → 全部源点跑完 `bc /= 2`；`n > 2` 时再除以 $(n-1)(n-2)/2$，否则置全 0 → 组装字典与稳定降序 `ranking`。
- **复杂度**：时间 O(V * E) / 空间 O(V + E)。
- **参数**：`adj`（`{u: {v: w}}` 邻接字典，边权与方向**都被忽略**，自环忽略、重复边去重；必填）。返回字典键：`"centrality"`（`{节点: float}`，已用 `C(n-1,2)` 归一化，取值 `[0,1]`）、`"ranking"`（list，按介数降序、同分按 `_ordered_nodes` 顺序稳定排列）。
- **陷阱**：① 只按**无权**（每条边算 1 跳）计最短路；带权图的介数（用边权求最短路）需要另写。② 归一化分母是 $C(n-1,2)=(n-1)(n-2)/2$，`n < 3` 时全部取 0（不报错）。③ "只有中间点非零"是常见误记：路径图 P5 上两端点为 0，但次中间点也非零（各 0.5）。④ 有向图必须另版实现（无向口径会把反向路径也算进依赖量）。
- **怎么检验**：① 路径图手算（自测 P5，节点 "0"-"4"）：中间点 "2" 的介数应为 $(2\times 2)/C(4,2)=2/3$（自测键 `graphs_btw_p5_mid`，因为经过它的最短路是 0-4 与 4-0 两条、总对数 $C(4,2)=6$），两端点应为 0（键 `graphs_btw_p5_ends_zero`），且中间点应是唯一的介数最大点。② 结构不变量：取值在 `[0,1]`；`ranking` 非增；所有点介数之和应等于"所有无序点对的最短路条数贡献"归一化后的总和（可用穷举枚举全部最短路条数独立重算小图）。③ 对称性：路径图/环图上介数应关于图的自同构对称；`n < 3` 时全为 0 且不报错。④ 交叉验证：用带权 `dijkstra` 计数最短路条数重算无权图的介数（单位权时应一致）。

#### `louvain_communities`
- **数学形式**：社区发现的模块度优化（**单层** Louvain）。模块度 $Q=\sum_C\big[\frac{l_C}{m}-\gamma\big(\frac{d_C}{2m}\big)^2\big]$，其中 $l_C$ 是社区内边权和、$d_C$ 是社区内节点度之和、$m$ 是总边权、$\gamma$ 是分辨率参数；节点从社区 C 移到邻居社区 D 的增益为 $\frac{k_{i\to D}-k_{i\to C}}{m}-\gamma\,\frac{k_i\,(d_D-d_C')}{2m^2}$。
- **步骤**：校验 `resolution > 0`、`max_iter >= 1` → `_weight_matrix` 统一成对称权重矩阵与节点顺序 → 空图返回空结果 → 负权/零权截断为 0、对角线清零，计算度向量 `deg`、`m2 = sum(W) = 2m`、`m`；`m <= 0` 时每个节点自成一社区 → 初始标签 `labels = arange(n)`，`tot` 为各社区总度 → 最多 `max_iter` 轮：用 `_common.rng(seed)` 做节点随机排列，对每个节点试算移到各邻居社区的增益，取增益最大且为正（大于 `1e-12`）的移动，更新标签与 `tot` → 一轮无移动则停 → 把社区编号按首次出现顺序**紧凑重编号** → 按闭式公式独立重算模块度（不累加增量）→ 返回标签、模块度、社区数。
- **复杂度**：时间 O(max_iter * E) / 空间 O(V + E)。
- **参数**：`adj`（`{u: {v: w}}` 邻接字典或 `n x n` 邻接矩阵，`inf` 表示无边；边权即无向权重，**负权按 0 处理**、自环丢弃；必填）；`resolution`（分辨率参数 $\gamma>0$，越大社区越多，默认 `1.0`）；`seed`（随机种子，`None` 时用 `_common.rng(None)` 的库默认种子，默认 `None`）；`max_iter`（局部移动的最大轮数，至少 1，默认 `20`）。返回字典键：`"labels"`（list of int，长度 = 节点数，按 `_weight_matrix` 的节点顺序排列，社区编号从 0 开始按首次出现紧凑编号）、`"modularity"`（float，最终划分在给定分辨率下的模块度）、`"n_communities"`（int）。
- **陷阱**：① 只有**一层**局部移动，没有把社区收缩成超点再迭代，大图上质量低于完整 Louvain（这也是自测只保证"两个团 + 桥"这类小算例分对的原因）。② 结果依赖节点遍历顺序：换 `seed` 可能得到不同划分（模块度也可能不同），自测只断言同 seed 可复现。③ 单层贪心**不保证全局最优**，模块度可能停在局部极大（例如先吞下桥端点）。④ 无向、无权重的自环与负权不做支持：自环丢弃、负权截断为 0。
- **怎么检验**：① 自测算例（两个三角形 + 一条桥 `(2,3)`）：`n_communities` 应为 2（键 `graphs_louvain_n_communities`）；`modularity` 应大于 0.3（键 `graphs_louvain_modularity`）；同一 `seed=0` 两次调用 `labels` 必须完全一致（键 `graphs_louvain_same_seed_equal`）；团内同社区（`lab[0]==lab[1]==lab[2]`、`lab[3]==lab[4]==lab[5]`）且两团不同社区。② 不变量：`labels` 长度 = 节点数、取值恰为 `0..n_communities-1` 连续；用返回的 `labels` 独立按闭式公式重算模块度，应与 `"modularity"` 一致。③ 单调性检查：分辨率 `resolution` 增大时社区数不应减少（趋势性检查，非严格定理）。④ 退化：空图/无边图不报错（`m <= 0` 时每个节点一个社区，模块度 0）。

#### `min_cost_flow`
- **数学形式**：最小费用流（连续最短路 SSP）。在残量网络上反复求"单位费用最小"的增广路，求满足流量守恒、边流量 $0\le f(u,v)\le c(u,v)$ 且从 $s$ 送达 $t$ 的总流量恰为 `demand` 的流中，总费用 $\sum_{u,v} f(u,v)\,a(u,v)$ 最小者。
- **步骤**：`as_matrix` 读入并校验 `cost`/`capacity`（方阵、形状一致、容量非负、费用全有限）→ 校验 `source`/`sink` 下标不越界且不同、`demand` 为正有限数 → `flow` 矩阵清零 → 循环直到 `total_flow >= demand - 1e-12`：在残量网络（`K[u,v] - flow[u,v] > 1e-12`）上用 Bellman-Ford 松弛最多 n-1 轮求源到汇最短费用路 → 不可达则抛 `ValueError`（容量不足）→ 回推瓶颈 `min(剩余需求, 路上最小残量)`，瓶颈 `<= 1e-12` 时抛 `ValueError` → 沿路 `flow[u,v] += add`、`flow[v,u] -= add`（用"流量矩阵可为负"表示反向流）、累加费用 → 输出 `f > 1e-12` 的原始方向边。
- **复杂度**：时间 O(A * V * E)（A 为增广次数，每次 Bellman-Ford 为 O(VE)）/ 空间 O(V^2)。
- **参数**：`cost`（`n x n` 费用矩阵，`cost[u][v]` 是 u→v 的单位费用，必须全部有限（无边请用容量 0 表示）；必填）；`capacity`（`n x n` 容量矩阵，`capacity[u][v]` 是 u→v 的容量上限，不能为负；必填）；`source`（源点下标，必填）；`sink`（汇点下标，必须与 source 不同，必填）；`demand`（需要从 source 送到 sink 的总流量，正数，必填）。返回字典键：`"flow"`（float，实际送达流量，一定等于 `demand` 否则报错）、`"cost"`（float，总费用）、`"edge_flow"`（list，每项 `[u, v, f]`，只列出 `f > 0` 的**原始方向**边，按 `(u, v)` 升序）。
- **陷阱**：① 容量不足时 `raise ValueError`，**不会**静默返回"能满足多少算多少"的部分流；只想要最大流请用 `max_flow_edmonds_karp`。② 依赖 `capacity - flow` 表示残量，因此 `u->v` 与 `v->u` 不能同时有独立容量，否则两条管道会被混成同一条（请把双向边拆成虚拟节点）。③ 原图存在负费用环时 SSP 不保证最优（Bellman-Ford 只跑 V-1 轮，也不报错）。④ 费用与容量都按 float 处理，大整数需求会有浮点误差；比较时请留容差。
- **怎么检验**：① 手算实例（自测）：费用矩阵 `[[0,1,5],[0,0,1],[0,0,0]]`、容量矩阵 `[[0,4,4],[0,0,4],[0,0,0]]`、`source=0, sink=2, demand=4`，两条路 `0→2` 费用 5、`0→1→2` 费用 1+1=2，最优走后者，总费用应为 $4\times 2=8$、送达流量应为 4（自测断言键 `graphs_mcf_flow` / `graphs_mcf_cost`）。② 守恒性（自测）：由 `edge_flow` 独立统计每个节点的净流出，中间节点应为 0、源点净流出应等于 4（键 `graphs_mcf_conservation`）。③ 与最大流对照：令 `demand` 极大时应在容量不足处抛 `ValueError`，此时用 `max_flow_edmonds_karp` 求出的最大流即为可送达的上界。④ 下界检查：`cost` 不应小于"把 demand 单位流量全部沿最便宜路径运送"的乐观下界。

#### `bipartite_matching`
- **数学形式**：二分图最大权匹配（指派问题）。给定 $r\times c$ 收益矩阵 $C$，求匹配 $M\subseteq\{1..r\}\times\{1..c\}$（每个左部/右部点至多配一次）最大化 $\sum_{(i,j)\in M} C_{ij}$；取**小边侧**的完备匹配，即 $|M|=\min(r,c)$。等价于对 $-C$ 求最小费用完备匹配。
- **步骤**：`as_matrix` 读入并校验（二维、非空、全部有限）→ 空矩阵直接返回空结果 → 若 `rows <= cols` 对 `-M` 调用 `_hungarian_min` 得到每行的列指派；否则对 `-M.T` 调用后把下标换回（行↔列）→ 把匹配对按 `(i, j)` 升序排序 → 用**原始** `M` 重算收益和。
- **复杂度**：时间 O(min(r,c)^2 * max(r,c)) / 空间 O(r * c)。
- **参数**：`cost`（`rows x cols` 权重矩阵，`cost[i][j]` 是左部 i 与右部 j 匹配的收益，可正可负但必须全部有限；必填）。返回字典键：`"matching"`（list，每项 `[i, j]`，按 i 升序）、`"total_cost"`（float，匹配对的收益之和）、`"size"`（int，匹配边数 `= min(rows, cols)`）。
- **陷阱**：① 返回的是**小边侧的完备匹配**（`size = min(rows, cols)`），零收益甚至负收益的边也会被选上；若只想保留正收益边，请自己按 `total_cost` 过滤。② 权重必须有限；"禁止匹配"不能写 `inf`，请用足够小的负数（如 `-1e9`）。③ 收益矩阵被当作**权重**而不是费用；若你的矩阵是成本，请传 `-cost` 并读 `-total_cost`。④ 不支持一对多/多对一（那不是匹配问题，要建流网络）。
- **怎么检验**：① 自测算例（3x3 全 1 矩阵）：`size` 应为 3、`total_cost` 应为 3（键 `graphs_bip_unit_size` / `graphs_bip_unit_cost`）。② 贪婪陷阱算例（自测 2x2 `[[10,9],[9,1]]`）：最优是 $(0,1)+(1,0)=9+9=18$，而"先取最大元素"的贪心只能得 $10+1=11$，自测断言必须等于 18（键 `graphs_bip_2x2_cost`）。③ 穷举对照（自测 3x3 矩阵 `[[10,2,8],[9,7,5],[6,4,3]]`）：结果应等于 `itertools.permutations` 穷举的最大值（键 `graphs_bip_brute_equal`）——这是最可靠的检验法，$n\le 8$ 时可直接穷举。④ 不变量：`size == min(rows, cols)`；`matching` 中左部下标互不相同、右部下标也互不相同；`total_cost` 等于按原矩阵逐对重算之和。⑤ 转置对称性：`bipartite_matching(M.T)` 的最优值应与 `bipartite_matching(M)` 相同。

#### `a_star`
- **数学形式**：A* 最短路。以 $f(v)=g(v)+h(v)$ 为优先序（$g$ 为起点到 $v$ 的已走代价、$h$ 为到终点的估计代价）扩展节点；当 $h$ 可采纳（$0\le h(v)\le$ 真实剩余代价）时返回的代价等于 Dijkstra 最短路，且扩展节点数通常更少。
- **步骤**：校验 `adj` 为字典，`start`/`goal` 必须在图中（否则抛 `ValueError`）→ `heuristic` 必须是字典或可调用对象 → 包一层 `h(v)`：字典口径缺省取 `0.0`，取值必须非负有限，并对所有节点预先求值一遍 → 初始化 `g[start]=0.0`、`prev[start]=None`，堆放 `(h(start), 递增计数, start)`，`closed` 集合与 `expanded` 计数 → 循环弹堆：已闭合则跳过，否则闭合、`expanded += 1`，若 `u == goal` 立即 break；对每条出边校验非负（负权抛 `ValueError`），`ng = g[u] + w`，若 `ng < g.get(v, inf) - 1e-15` 则更新 `g`/`prev` 并压入 `(ng + h(v), 计数, v)` → 结束后：`goal` 不在 `g` 中返回空路径与 `inf`，`goal == start` 返回 `[start]` 与代价 0，否则用 `reconstruct_path(prev, start, goal)` 还原路径。
- **复杂度**：时间 O(E log V)（最坏退化为 Dijkstra）/ 空间 O(V + E)。
- **参数**：`adj`（`{u: {v: w}}` 邻接字典，边权必须非负；必填）；`start`（起点，必须在图中；必填）；`goal`（终点，必须在图中；必填）；`heuristic`（`{节点: 估计值}` 字典或 `f(节点) -> float` 可调用对象，缺省节点按 0 处理即退化为 Dijkstra，必填）。返回字典键：`"path"`（list，从 start 到 goal 的节点序列，不可达为 `[]`）、`"cost"`（float，路径总代价，不可达为 `inf`）、`"expanded"`（int，从优先队列中真正弹出的节点个数，衡量搜索规模）。
- **陷阱**：① 启发式必须**可采纳**（$h\le$ 真实剩余代价）；不可采纳时返回的 cost 可能大于真实最短路，且不会报错；本实现只检查"非负且有限"，查不出可采纳性。② 用"出堆即闭合"的写法，配合**不一致**（inconsistent）启发式时可能失去最优性；要保证最优请用一致启发式，或允许重新打开已闭合节点。③ 只支持非负边权（负权请用 Bellman-Ford）。④ `heuristic` 用可调用对象时会对每个入堆邻居求值，代价高请先做成字典。⑤ 文档未强调：`start` 或 `goal` 不在图中会直接抛 `ValueError`（不会返回空路径）。
- **怎么检验**：① 与 Dijkstra 交叉验证（自测）：5x5 栅格挖掉中心格，用可采纳的曼哈顿启发式时 `abs(ast["cost"] - dijkstra["dist"][goal])` 记为 `graphs_astar_cost_dev`，应接近 0（自测断言 `<= 1e-9`）；`h=0` 时同样应等于 Dijkstra（键 `graphs_astar_zero_h_dev`）。② 规模断言（自测）：栅格 (0,0)→(4,4) 最短路为 **8** 跳（自测断言 `abs(ref - 8.0) <= 1e-9`，且 `len(path) - 1 == 8`）；`expanded` 不可能超过节点总数（自测断言 `<= 25`），且曼哈顿启发式不应比 `h=0` 扩展更多节点。③ 结构检查：`path[0] == start`、`path[-1] == goal`、逐边累加等于 `cost`；`goal == start` 时返回 `[start]`、代价 0；不可达时 `path=[]`、`cost=inf`。④ 可采纳性自查：对每个节点验证 $h(v)\le$ 用 Dijkstra 算出的真实剩余代价。

#### `vrp_clarke_wright`
- **数学形式**：带容量约束的车辆路径问题（CVRP，送货型）的 Clarke-Wright 节约算法。节约值 $s(i,j)=d(\text{depot},i)+d(\text{depot},j)-d(i,j)$，按 $s$ 降序把两条路线的**端点**客户 $i,j$ 合并，约束是合并后载重 $\le$ `capacity`；目标是最小化所有路线总里程（启发式，只保证可行）。
- **步骤**：`as_matrix` 读入距离矩阵并校验方阵 → 校验 `depot` 下标、`demand` 长度等于 n、`capacity` 为正有限数、距离全部有限 → 校验每个客户需求非负且不超过 `capacity`（超出直接抛 `ValueError`）→ 每个客户初始化独立路线 `[depot, i, depot]` 与载重 → 枚举全部客户对算节约值，存为 `(-s, i, j)` 并排序（确定性 tie-break）→ 按序尝试合并：同路线跳过、载重和超容量跳过、`i`/`j` 不在各自路线端点跳过；合并时按"让 i 落在前段末端、j 落在后段首端"反转，路线接成 `[depot] + a_seq + b_seq + [depot]`，更新载重与客户归属，删除旧路线 → 路线按首个客户下标升序输出，逐边累加总里程。
- **复杂度**：时间 O(n^2 log n)（节约值排序主导）/ 空间 O(n^2)。
- **参数**：`distance`（`n x n` 距离矩阵，**假设对称**，对角线视为 0；必填）；`demand`（长度 n 的需求序列，`demand[depot]` 被忽略；必填）；`capacity`（单车容量上限，正数；必填）；`depot`（车场下标，默认 `0`）。返回字典键：`"routes"`（list，每条路线是 `[depot, 客户..., depot]`，元素为下标，路线之间按首个客户下标升序排列）、`"total_distance"`（float，所有路线长度之和）、`"n_routes"`（int，路线条数）。
- **陷阱**：① 只做"端点合并"，不做 2-opt / Or-opt 改进，因此结果一般不是最优解（Clarke-Wright 本身是启发式，只保证可行）。② 距离矩阵假定对称；非对称矩阵上路径长度计算会静默偏小（回程按正向元素取）。③ 单个客户需求超过 `capacity` 时直接 `raise ValueError`，不返回不可行路线。④ 节约值相同的合并顺序由 `(-s, i, j)` 决定，是有意为之的确定性 tie-break；不同实现（不同 tie-break）会给出不同但同样可行的解。
- **怎么检验**：① 手算实例（自测）：一维坐标 `[0, -1, -2, 1, 2]`（depot=0）、每个客户需求 1、`capacity=2`，最优为 2 条路线各 `1+1+2=4`，合计 **8**（自测断言 `total_distance == 8`、`n_routes == 2`）。② 可行性检查（自测）：每条路线首尾都是 depot；逐路线载重之和不超过 `capacity`；每个客户恰好被服务一次（`sorted(seen_customers) == [1,2,3,4]`）；`total_distance` 等于按路线逐边重算之和。③ 与独立下界比较：总里程不应小于"所有客户需求之和 / 容量向上取整"所隐含的车辆数下界对应的最小可能里程，可作为健全性上界检查。④ 边界：某客户需求 > `capacity` 应抛 `ValueError`；`demand` 长度与矩阵阶数不符、`depot` 越界也应抛 `ValueError`。

#### `network_robustness`
- **数学形式**：确定性蓄意攻击下的网络鲁棒性。每轮移除剩余图中**度最大**的节点（同分取节点顺序靠前者），移除 `n_remove` 个后统计剩余子图的**最大连通分量规模**与**全局效率** $E=\dfrac{1}{m(m-1)}\sum_{i\ne j}\dfrac{1}{d(i,j)}$（$d$ 为**跳数**，不可达对贡献 0，$m$ 为剩余节点数）。
- **步骤**：`_weight_matrix` 统一成对称权重矩阵与节点顺序（无向化）→ 校验 `n_remove >= 0` → 空图返回空结果 → 邻接布尔矩阵 `adjm = isfinite(W) & (W != 0)`，`act` 标记存活 → 循环：在存活节点上按非零邻接计数求度（已删节点置 -1 避免被选），取 `argmax` 标记删除并记录，直到删够 `n_remove` 个或只剩 1 个节点 → 对剩余节点逐个 BFS 求跳数，取最大分量规模，并按有序对累加 $1/d$ → `efficiency = reach_sum / (m(m-1))`（`m <= 1` 时取 0）。
- **复杂度**：时间 O(n_remove * V^2 + V * (V + E)) / 空间 O(V + E)。
- **参数**：`adj`（`{u: {v: w}}` 邻接字典或 `n x n` 邻接矩阵，`inf` 表示无边；按**无向无权**图处理（只关心"有没有边"）；必填）；`n_remove`（要移除的节点个数，非负，最多移除到只剩 1 个节点，默认 `3`）。返回字典键：`"largest_component"`（int，移除结束后剩余图的最大连通分量规模，单个孤立点算 1，没有剩余节点时为 0）、`"efficiency"`（float，剩余图的全局效率，公式如上）、`"removed"`（list，按移除顺序排列的节点）。
- **陷阱**：① 移除顺序是**确定性贪心**（度最大 + 顺序 tie-break），不是随机攻击；随机故障鲁棒性请自行打乱顺序重复实验。② 效率用的是**跳数**而非边权，带权图上的结论可能完全不同。③ 返回的是"移除完毕之后"的指标，不含每步演化曲线；需要曲线请循环调用本函数（每次 `n_remove=1`）。④ `n_remove` 过大时只会移除到剩 1 个节点，`removed` 长度可能小于 `n_remove`。
- **怎么检验**：① 完全图闭式解（自测 K5，`n_remove=1`）：移除后剩 K4，`largest_component` 应为 4、`efficiency` 应为 1（自测断言 `abs(efficiency - 1.0) <= 1e-12`），且完全图上 tie-break 最靠前的节点被删，`removed == [0]`。② 星形图（自测，中心 0 连 1..4，`n_remove=1`）：度最大点是中心，故 `removed == [0]`、移除后只剩孤立点，`largest_component` 应为 1、`efficiency` 应为 0。③ 交叉验证：把剩余图交给 `connected_components` / `dijkstra`，独立核对最大连通分量规模与跳数距离，再按公式重算 `efficiency`。④ 边界：`n_remove = 0` 时 `removed` 为空、指标等于原图；`n_remove` 大于 n-1 时 `removed` 长度应为 `n-1`（只剩 1 个节点）；`n_remove < 0` 抛 `ValueError`。


### 3.3 组合优化与元启发式 —— `examples/algorithms/heuristics.py`

这一族解决的是**目标函数不可导、非凸、离散或组合爆炸（NP-hard）**的问题：解析法与梯度法失效时，
用"随机搜索 + 局部改进"在有限计算预算内拿到一个可用的可行解。它不保证全局最优，论文里必须把它
写成"在给定参数与预算下的最好解"，而不是"最优解"。

共同约定（写论文时逐条交代，否则结果不可复现）：

- **随机性**：所有函数接受 `seed`，不给就用 `DEFAULT_SEED`，内部只用 `numpy.random.Generator`，
  不碰全局随机状态；同一个种子两次调用逐位相同。
- **最小化口径**：除 `genetic_algorithm`（`maximize` 开关）与 `particle_swarm`（`maximize` 开关）外，
  全部约定最小化，最大化问题自己取负。
- **预算**：`iters` / `generations` / `n_gen` 就是计算预算，报告时必须同时给出**目标函数实际求值次数**
  （`differential_evolution` 直接返回 `n_eval`），不同预算下的解不可直接比较。
- **随机算法的结论**：单次运行不能作为结论，至少跑 3~10 个种子并报最好/最差/中位数。

---

#### `simulated_annealing(cost, x0, neighbor, T0=100.0, alpha=0.995, iters=5000, seed=None)`

- **数学形式**：在当前解 x 的邻域 N(x) 采样 x′，记 ΔE = cost(x′) − cost(x)；ΔE ≤ 0 必接受，否则以 min{1, exp(−ΔE/T)} 接受（Metropolis 准则）；几何降温 T_{k+1} = α·T_k。
- **步骤**：① 令 cur = best = x0，T = T0，求值；② 每步由 `neighbor(cur, rng)` 生成候选 y；③ 非有限候选直接拒绝，否则按 Metropolis 判据接受/拒绝；④ T ← α·T；⑤ 记录历史最优，循环 `iters` 次后返回。
- **复杂度**：时间 O(iters·(T_cost + T_neighbor))；空间 O(|x| + iters)——`history` 长度固定为 iters+1。
- **参数**：`T0=100.0`（初始温度，必须与目标函数量级匹配：目标值在 1e-3 量级而 T0=100 会让前几千步近似随机游走，T0 太小则退化成爬山；实用标定法是先采样几十步 |ΔE| 的均值，取 T0 ≈ 若干倍该均值）；`alpha=0.995`（降温系数，取值 (0,1)，越接近 1 降温越慢、搜索越久，0.9 时约百步就冻住）；`iters=5000`（降温步数，与 α 共同决定末端温度 α^iters）；`seed`（默认 `DEFAULT_SEED`，换种子得到的解会不同）。
- **陷阱**：**约定最小化**，最大化请取负；`T0`/`alpha`/`iters` 与邻域尺度必须写进论文；`history` 存的是历史最优（单调不增），用它看不出早熟，论文里更该画"当前解"曲线；连续问题用固定步长邻域在高维下几乎全被拒绝，步长需随温度收缩；代价函数返回 NaN/±inf 的候选会被本实现显式拒绝（不会静默卡死），但初始解非有限值会直接 `ValueError`。
- **怎么检验**：① 在已知全局最优的算例（二维 Rastrigin，最优 0 于原点；十城圆上均匀分布 TSP，最优回路长度有闭式 2n·sin(π/n)）上跑，断言 `best_cost` 与已知最优的 gap 落在你声明的容差内；② 换 10 个 `seed` 重复，报告最优值分布而不是单点值；③ 把 `iters` 加倍、`alpha` 调到 0.999 重跑，看解是否继续改善（判断是否已收敛），若毫无变化说明预算早已过剩或邻域失效；④ 做极限行为检验：令 `alpha→0` 时算法应退化为"只接受改进"的爬山，其解不应优于正常 SA。

---

#### `genetic_algorithm(fitness, n_genes, pop_size=60, generations=200, crossover_rate=0.8, mutation_rate=0.02, seed=None, maximize=True, init_density=0.15, repair=None)`

- **数学形式**：0/1 编码的种群进化：锦标赛选择 S(pop) → 单点交叉（概率 p_c）→ 逐位翻转变异（每位概率 p_m）→ 评估；选择概率由适应度排序隐含给出，历史最优用精英保留强制单调不增。
- **步骤**：① 按伯努利(`init_density`)逐位采样初始种群，可选 `repair` 写回；② 每代保留 1 个精英，其余个体经二元锦标赛选父代 → 依 `crossover_rate` 做单点交叉 → 逐位变异 → 再次 `repair` → 求值；③ 更新历史最优并记录，共 `generations` 代。
- **复杂度**：时间 O(generations·pop_size·n_genes + 适应度求值)，求值次数约 (generations+1)·pop_size；空间 O(pop_size·n_genes + generations)。
- **参数**：`pop_size=60`（必须 ≥ 2，太小则选择压力不足、早熟）；`generations=200`（进化代数，与 `pop_size` 共同构成预算）；`crossover_rate=0.8`（每对父代发生交叉的概率，取 0 等于纯变异搜索）；`mutation_rate=0.02`（**每个基因位**的独立变异概率，口径是"每位"而非"每染色体"，期望变异位数 = n_genes·mutation_rate；调大增加多样性但破坏已找到的模式）；`init_density=0.15`（初值取 1 的概率，**不要随手改 0.5**：0/1 约束问题在 0.5 密度下绝大多数个体不可行）；`maximize=True`（True 求最大，False 求最小，内部对适应度取负后统一按最大化处理）；`repair`（可选钩子，把不可行个体修回可行域，必须返回同形状 0/1 数组）；`seed`（默认 `DEFAULT_SEED`）。
- **陷阱**：不保留精英会震荡，本实现保留 1 个精英；若约束用返回 `-inf` 的硬罚函数表达，**整个种群都不可行时所有适应度相同**，`argmax` 恒定取第 0 个，选择算子退化成随机——这正是默认 `init_density=0.15` 且推荐用 `repair` 的原因；`repair` 必须在**入种群时**物化写回，否则 `best_genes` 与 `best_fitness` 会对不上号（本实现已在 `_apply_repair` 里保证一致）；二进制编码对连续变量有偏（n_genes 位只能表示 2^n_genes 个格点，且相邻整数间可能多位翻转），连续问题优先用 PSO/DE；早熟时种群趋同、交叉失效，只剩变异扰动，建议报告"最优解首次出现的代数"与种群平均适应度曲线。
- **怎么检验**：① 小组合问题上与**穷举**对拍：本模块自测用 10 件 0/1 背包算例（重量 2,3,5,7,11,13,17,19,23,29；价值 3,5,8,11,17,20,26,30,35,44；容量 60），2^10 = 1024 种组合可全部枚举，唯一最优价值为 94，GA 的结果不得超过 94 且应当命中 94；② **预算对照**：同一算例降低预算（自测里记录过 pop_size=40/generations=60 的配置在 5 个种子上都停在 93，而 pop_size=60/generations=150 在同样 5 个种子上稳定命中 94），这说明"差 1"来自预算而非算法失效——论文里报参数时必须带上预算；③ 单点检验：构造适应度只与某一位相同的"中性"问题，检查解是否收敛到该位；④ 若给出 `repair`，断言返回的 `best_genes` 一定可行（用独立的可行性检查函数，而不是算法自己的适应度）。

---

#### `particle_swarm(objective, bounds, n_particles=30, iters=200, seed=None, maximize=False)`

- **数学形式**：v ← w·v + c₁·r₁⊙(pbest − x) + c₂·r₂⊙(gbest − x)，x ← clip(x + v, lo, hi)；r₁, r₂ ~ U(0,1)^d 逐维独立；惯性权重 w 由 0.9 线性降到 0.4，c₁ = c₂ = 2.0。
- **步骤**：① 在 `bounds` 内均匀初始化位置，速度取区间跨度的 ±10%（即 U(−0.1·span, +0.1·span)）；② 求值得到 pbest 与 gbest；③ 每代按上式更新速度与位置并 clip 回搜索盒；④ 逐粒子更新 pbest（按用户口径取更优），再更新 gbest；⑤ 记录历史最优，共 `iters` 代。
- **复杂度**：时间 O(iters·n_particles·(d + T_objective))；空间 O(n_particles·d + iters)。
- **参数**：`n_particles=30`（≥ 2，粒子越多每代覆盖越广但求值次数线性增长，常见取 20~50）；`iters=200`（迭代代数，预算 = iters·n_particles 次求值）；`maximize=False`（默认最小化，True 时内部整体取负）；`bounds`（搜索盒，可写成 `(lo, hi)` 共用区间或 `[(lo₁,hi₁), ...]` 逐维区间，每维必须 `hi > lo`，退化维度直接报错）；`seed`（默认 `DEFAULT_SEED`）。
- **陷阱**：`c₁/c₂` 与 `w` 在代码里是**硬编码**的，不能通过参数调——要做惯性权重创新必须改源码并在论文里写明；边界处理是 `clip` + 速度不变，最优解落在边界上时粒子会"贴边"收敛慢，这是已知局限；初始化必须在盒内，否则第一代 gbest 可能不可行；各维独立更新意味着算法对**坐标系旋转敏感**，变量强耦合时收敛慢（可先做主成分旋转或改 CMA-ES）；速度爆炸（r 极端或 c 过大）会表现为 `history` 一条直线。
- **怎么检验**：① 二维 Rastrigin 与二维 Ackley 的全局最优都是 0，可断言 `best_value` 与 0 的差；② 构造**闭式解**问题：f(x) = −(x−3)² 在 [0,10] 上最大值 0 于内点 x = 3、g(x) = x 在 [0,10] 上最大值 10 于边界 x = 10，前者检查 `maximize=True` 的符号处理、后者专门检查 `clip` 没把解压错（自测里就是这么用的）；③ 与纯随机采样对照：同预算内均匀撒点取最优，PSO 的解应严格更优，否则是实现有 bug 而不是"算法不灵"；④ 单峰二次函数上做极限检验：`iters` 足够大时最优解应逼近解析极小点。

---

#### `ant_colony_tsp(dist, n_ants=20, iters=100, alpha=1.0, beta=2.0, rho=0.5, seed=None)`

- **数学形式**：转移概率 p(i→j) ∝ τ_ij^α · η_ij^β（η = 1/d），只对未访问城市归一化；一代结束后挥发 τ ← (1−ρ)·τ，再按 Q/L_k 对每只蚂蚁走过的边双向加强（对称 TSP 口径 τ_ij = τ_ji）。
- **步骤**：① τ 初始化为 1/n，η 取 1/d（零距离边用大数 1e12 代替 inf，对角置 0）；② 每只蚂蚁随机起点，逐步按概率选下一城；③ 概率权重和为 0 或非有限时回退为"未访问城市中均匀随机"；④ 一代结束做挥发 + 按 1/L 加强；⑤ 记录历史最优回路，共执行 `iters+1` 代。
- **复杂度**：时间 O((iters+1)·n_ants·n²)；空间 O(n²)。
- **参数**：`n_ants=20`（每代蚂蚁数，≥ 1，通常取城市数 n，太小则信息素更新噪声大、退化成随机重启爬山）；`iters=100`（代数，实际跑 iters+1 代，`history` 长度 = iters+1）；`alpha=1.0`（信息素重要度，0 表示完全不用信息素、退化为贪心构造）；`beta=2.0`（启发式 1/d 的重要度，太大则早期就贪心、易早熟）；`rho=0.5`（挥发率，取值 (0,1)，越大遗忘越快，接近 1 等于抹掉全部历史信息）；`seed`（默认 `DEFAULT_SEED`，蚂蚁起点与轮盘赌都依赖它）。
- **陷阱**：**必须重新归一化**每个蚂蚁的候选概率并把已访问城市置 0，且权重全 0（信息素下溢或零距离边）时要显式回退到均匀随机，否则除零产生 NaN 并污染整张信息素矩阵（本实现已兜底，但换实现时要自己检查）；长期挥发会让 τ 下溢到 0，实践常设 τ_min 下限（本实现没有设）；**只对对称距离矩阵有效**，非对称矩阵下信息素会混淆方向、结果无效；距离为 0 的边若直接取 1/0 会得到 inf（本实现用 1e12 代替）；`Q` 固定为 1.0，不能调。
- **怎么检验**：① 圆上均匀分布的城市，最优回路长度有**闭式解** 2n·sin(π/n)（正多边形周长）——本模块自测用 10 城圆（最优 6.180339887498948，代码里同时返回 `aco_tsp_optimal` 作对照），可断言 `length ≥ 闭式最优` 且 gap 在容差内；② 退化算例：所有城市重合（距离矩阵全 0）时不应出现 NaN，应返回有限长度；n = 1 时返回 `{"tour": [0], "length": 0.0}`；③ 断言返回的 `tour` 是 0..n−1 的一个排列（每条边都能在 `dist` 中查到，长度与 `length` 自洽）；④ 与最近邻贪心构造的对拍：在同一算例上 ACO 的有效解不应劣于（或至少应接近）已知贪心解，劣太多说明参数或实现有问题。

---

#### `differential_evolution(objective, bounds, pop_size=40, n_gen=100, F=0.7, CR=0.9, seed=None)`

- **数学形式**：DE/rand/1/bin——变异 v = x_a + F·(x_b − x_c)（a,b,c 互不相同且都 ≠ i），逐分量二项交叉 u_j = v_j（若 r_j < CR）否则 u_j = x_{i,j}，并强制至少一个分量取 v；贪心选择：fun(u) ≤ fun(x_i) 时替换。
- **步骤**：① 在 `[lo, hi]` 内独立均匀采样初始种群并逐个求值；② 每代对每个个体抽 3 个互异供体 → 变异 → `clip` 回盒 → 二项交叉（至少一位来自变异体）→ 求值 → 贪心替换（相等也替换，允许沿等值面漂移）；③ 每代结束更新历史最优。
- **复杂度**：时间 O(n_gen·pop_size·(d + T_objective))；空间 O(pop_size·d + n_gen)。求值次数**恰好** pop_size·(n_gen+1)，代码返回 `n_eval` 供论文报预算。
- **参数**：`pop_size=40`（必须 ≥ 4，`DE/rand/1` 需要 3 个互不相同且 ≠ i 的供体，取小了直接报错而不是静默重复个体）；`n_gen=100`（代数，预算 = pop_size·(n_gen+1) 次求值）；`F=0.7`（差分缩放因子，取值 (0,2]，常用 0.5~0.9：越大探索越猛、越容易跳出局部最优但也越容易被拒绝）；`CR=0.9`（交叉概率，取值 [0,1]，口径是**逐分量**：期望 CR·d 个分量取自变异体；CR 太低每代只改一个分量，高维下慢得离谱；CR=1 且 F 很小时父代信息被完全丢弃）；`seed`（默认 `DEFAULT_SEED`）；`bounds`（口径与 PSO 一致，每维 `hi > lo`）。
- **陷阱**：越界处理是简单 `clip`，最优解在边界上时会出现"边界锁死"（大量分量堆在边界平面、种群多样性骤降），文献里更常用反射或随机重采样；`pop_size < 4`、`F ∉ (0,2]`、`CR ∉ [0,1]` 都直接 `ValueError`；目标函数返回 NaN 时所有比较为 False，个体被永久保留——**调用方必须保证返回有限值**；`history` 是历史最优、单调不增，画出来很好看但看不出早熟，论文里更应报告每代种群均值与标准差。
- **怎么检验**：① **求值次数自洽**：断言 `n_eval == pop_size·(n_gen+1)`（本模块自测就检查这一条）；② 与**随机基线**对拍：同一搜索盒内用固定种子均匀采样 2000 个点取最优，DE 在 10 维 Rastrigin 与 2 维 Rosenbrock 上都应严格优于该基线——自测把这条写成了断言，跑不过即视为实现有 bug；③ 与成熟库对拍（如 scipy 的 `differential_evolution`）比较同一算例的最优值量级与收敛代数；④ 极限检验：把 `n_gen` 设为 0 时应返回初始种群的最优个体，`history` 长度为 1。

---

#### `tabu_search(init, neighbors_fn, objective, max_iter=200, tabu_tenure=7, seed=None)`

- **数学形式**：min f(x) s.t. x 在禁忌表外（除非满足藐视准则 f(x) < f*）：每步取 argmin 于 N(cur) \ Tabu ∪ Aspiration，并**无条件**移动；禁忌表以解为键、以"解禁代数 = 当前步 + tabu_tenure"为值。
- **步骤**：① cur = init，清空禁忌表；② 枚举 `neighbors_fn(cur)`，丢弃形状不符或非有限的候选；③ 禁忌且不优于历史最优的候选丢弃（优于历史最优的禁忌解因藐视准则仍可用）；④ 在剩余候选中取目标值最小者，并列时由 `seed` 决定的生成器随机打破；⑤ 无条件下移，写入禁忌表、清理过期条目、更新历史最优；⑥ 邻域为空或全被禁忌时提前结束。
- **复杂度**：时间 O(max_iter·(|N|·(T_neighbor + T_objective) + |T|))；空间 O(|T| + max_iter)。返回的 `n_iter ≤ max_iter`（提前停止时会变小）。
- **参数**：`max_iter=200`（最大步数）；`tabu_tenure=7`（禁忌长度，≥ 1：第 t 步接受的解在 t+1..t+tabu_tenure−1 步内被禁用；太短会立刻回访刚离开的解形成 2-循环，太长会禁掉几乎整个邻域、被迫每步走差解，经验值是邻域规模的平方根量级）；`neighbors_fn`（邻域函数，**不接收 rng**，邻域生成必须确定性，否则同一 seed 无法复现）；`init`（初始解，非有限目标值直接报错）；`seed`（默认 `DEFAULT_SEED`，只用于并列候选中随机打破平局）。
- **陷阱**：本实现禁忌的是**解**而不是**移动**——对置换类问题这能顺带禁掉逆向移动，但对连续问题，只要步长足够小就会不断产生"没被禁忌过的新点"，禁忌表几乎失效；**必须允许走差解**，若改成只接受改进就退化成爬山；解键把浮点四舍五入到 12 位小数，步长小于 1e-12 时两个"不同"的解会被判为同一个而互相禁忌；返回的 `x`/`fun` 是**历史最优**而不是最后停留的解（禁忌搜索允许走差解，这一点最容易误读）；`neighbors_fn` 返回 `None` 会报错，返回空列表则正常停止。
- **怎么检验**：① 用**闭式最优**的置换算例：n 个城市均匀分布在圆上时最优回路长度 = 2n·sin(π/n)，自测用 9 城圆（正九边形周长）作上界，断言 `fun` 不超过该上界且不劣于初始解；② 退化检验：`neighbors_fn` 恒返回空列表时 `n_iter` 应为 0、返回初始解而不是死循环；③ 与爬山法对拍：同一邻域、同一初值下禁忌搜索应不劣于"只接受改进"的爬山（否则说明禁忌机制或藐视准则写反了）；④ 把 `tabu_tenure` 从 1 扫到 |N| 的平方根量级，报告最优值曲线，验证"太短/太长都变差"的预期形状。

---

#### `grey_wolf_optimizer(objective, bounds, n_wolves=30, n_gen=100, seed=None)`

- **数学形式**：A = 2a·r₁ − a，C = 2r₂（r₁, r₂ ~ U(0,1)^d）；D_L = |C·X_L − X_i|，X_L′ = X_L − A·D_L（L ∈ {α, β, δ}）；新位置 X_i′ = (X_α′ + X_β′ + X_δ′)/3，再 clip 回盒；a 由 2 线性递减到 0。
- **步骤**：① 盒内均匀初始化 `n_wolves` 匹狼并求值，按目标值升序（`kind="stable"`）取 α/β/δ；② 每代按上式更新每匹狼的位置；③ 全体求值后把"当前狼群 + 旧三匹头狼"一起排序重新选头狼（保证历史最优不丢失）；④ 记录 α 的目标值，共 `n_gen` 代。
- **复杂度**：时间 O(n_gen·n_wolves·d) 外加等量目标函数求值（总求值 n_wolves·(n_gen+1)）；空间 O(n_wolves·d + n_gen)。
- **参数**：`n_wolves=30`（狼群规模，必须 ≥ 3，否则选不出 α/β/δ，代码直接报错）；`n_gen=100`（代数，预算 = n_wolves·(n_gen+1) 次求值；太小会让整个搜索停留在探索阶段）；`bounds`（搜索盒，口径同 PSO/DE）；`seed`（默认 `DEFAULT_SEED`）。注意 GWO 的 a、系数 2、三狼平均都在代码里硬编码，没有可调参数——要做"改进 GWO"（非线性收敛因子、权重平均等）必须改源码。
- **陷阱**：**GWO 没有个体记忆**（无 pbest），狼的位置可以变差，只有三匹头狼保留历史信息，因此比 PSO 更容易丢失已找到的好解（本实现用"旧头狼参与排序"补这个洞，否则 `history` 可能不是单调不增）；a 从 2 线性递减，但循环末值是 2/n_gen 而非严格 0（docstring 写"降到 0"是近似说法）；|A| < 1 收缩开发、|A| > 1 远离探索，探索窗口只有前半程；越界处理是简单 `clip`，最优解在边界上时会贴边；头狼并列时排序稳定性影响结果，本实现用 `kind="stable"` 固定口径；低维单峰表现好，高维 Rastrigin 这类多峰可分问题上容易陷入局部最优，不要当成"一定优于 DE/PSO"。
- **怎么检验**：① 与**随机基线**对拍（自测口径）：同一搜索盒固定种子均匀采样 2000 点，GWO 在 10 维 Rastrigin 与 2 维 Rosenbrock 上都必须严格更优；② 低维**闭式解**算例（2 维 Rosenbrock 最优 0 于全 1 点、Rastrigin 最优 0 于原点）上断言最终 gap；③ 单调性自检：`history` 必须是单调不增序列，若出现回升说明头狼选择或历史最优维护写错了；④ 多峰与单峰各选一个算例做对照，验证"低维单峰好、高维多峰差"的预期行为，避免把算例难度当成算法优劣。

---

#### `variable_neighborhood_search(init, neighborhoods, objective, max_iter=100, seed=None)`

- **数学形式**：多邻域下降：每步在当前解 x 上用每个邻域 N_k 各扰动一次得到 y_k，取 y = argmin_k f(y_k)，若 f(y) < f(x) − 1e-12 则 x ← y，否则不动（best improvement、严格下降）。
- **步骤**：① cur = init 并求值；② 每步用 `rng.permutation` 打乱邻域顺序，对**每个**邻域各生成 1 个候选；③ 跳过非有限候选，长度不一致直接 `ValueError`；④ 在所有候选中取最小者，严格改进才接受；⑤ 记录历史最优，共 `max_iter` 步（本实现无提前停止，`n_iter` 恒等于 `max_iter`）。
- **复杂度**：时间 O(max_iter·|neighborhoods|·(T_perturb + T_objective))；空间 O(max_iter)（history）+ O(|x|)。
- **参数**：`max_iter=100`（步数，每步消耗 |neighborhoods| 次求值，预算 = max_iter·|neighborhoods|）；`neighborhoods`（扰动函数列表 `[perturb(x, rng) -> 候选]`，**必须接收 rng**；列表中各元素的扰动尺度应依次**变小**——大尺度负责跳出局部最优、小尺度负责精修，全用同一尺度就等价于固定步长随机搜索）；`init`（初始解，非有限目标值报错）；`seed`（默认 `DEFAULT_SEED`，控制邻域顺序与扰动的随机性）。
- **陷阱**：本实现是**下降式** VNS，只接受改进解，**不能**像禁忌搜索那样主动走差解，强多峰问题上会卡在局部最优——它的价值在于把"邻域结构"纳入搜索空间，而不是提供逃逸机制；候选是**跨邻域取最优**（best improvement）而不是"第一个改进就接受"；尺度分层不当（跨度过大）会让大邻域候选几乎全被拒绝，白白浪费预算；**不做边界裁剪**，有界问题必须让扰动函数自己 `np.clip`；严格不等号（好 1e-12 以上才接受）意味着平坦区域会完全停住，这是有意设计。
- **怎么检验**：① 一维二次函数 sum((x−3)²) 的解析最优是 x = 3，配多尺度高斯扰动（尺度 1.0…0.001）时断言收敛到 |x−3| ≤ 1e-3，并断言返回的 `fun` 与在 `x` 处重算的目标值一致（自测采用这一口径）；② 与"单一邻域重复调用"对拍：多尺度 VNS 在同预算下不应劣于只用最小尺度邻域的版本；③ 用单调性检查确认 `history` 单调不增；④ 故意构造一个全平的目标函数，验证算法在 1e-12 阈值下会停住而不是无意义空转。

---


### 3.4 预测与时间序列 —— `examples/algorithms/forecasting.py`

这一族解决"**给定一条按时间先后排列的一维序列，怎么做点预测、怎么刻画不确定性、怎么判断预测好不好、怎么把残差诊断清楚**"的问题。它给出的是一条从**朴素基线**（最后一期平推 / 季节平推 / 漂移外推）→ **指数平滑族**（一次、Holt、Holt-Winters）→ **Yule-Walker AR 与自相关工具** → **单位根检验** → **精度指标与时间序列专用的切分/回测**的完整阶梯，同时把"误差指标怎么写、切分怎么做才不泄漏未来信息"这些最容易在论文里被挑错的环节固定下来。

| 共同约定 | 具体内容 |
|---|---|
| **输入方向** | `y[0]` 最早、`y[-1]` 最新；全模块**不做任何随机打乱**，切分与回测一律按时间顺序（这也是切分函数没有 `shuffle` 参数的原因）。 |
| **返回形态** | 平滑类返回 `fitted`（**样本内一步预测**，与 `y` 等长，只用到 t 时刻之前的信息）与 `forecast`（样本外预测）；初始化阶段没有历史的那些位置填 `NaN` 而**不用未来数据回填**。 |
| **依赖边界** | 只用 `numpy` + 标准库；`statsmodels` 只用于**复核**，论文里的中间量建议用本模块显式打印。 |

#### `moving_average(y, window, centered=False)`

- **数学形式**：尾部窗口 `trend[t] = (1/w)·Σ_{i=0..w-1} y[t-i]`，有效区间 `[w-1, n-1]`；居中窗口 `trend[t] = (1/w)·Σ_{i=-offset..w-1-offset} y[t+i]`，`offset = ⌊w/2⌋`。窗口内**等权**，几何衰减加权留给 `exponential_smoothing`。
- **步骤**：① 校验 `window` 为正整数且不超过 `n`；② `centered=False` 时从 `t = w-1` 起滑动平均；③ `centered=True` 时从 `t = offset` 起居中平均；④ 无法计算的位置写 `NaN`，返回首个有效下标 `valid_from`。
- **复杂度**：时间 O(n·w)（滑动点积，未做前缀和/FFT 优化），空间 O(n)；`w` 很大时本实现会明显变慢，可自行改前缀和到 O(n)。
- **参数**：`window`（必填，无默认）——窗口越长曲线越平滑、转折点滞后越重（约 w/2 期）；`centered`（`False`）——置 `True` 用于**提取趋势**（引入未来信息，不能用于预测），保持 `False` 是单边窗口、可当预测器用但有相位滞后。
- **陷阱**：首尾必然缺值，缺值一律写 `NaN`，**不要填 0 或边界值**——填 0 会让后续 `mape` 直接炸掉；偶数窗口的居中平均并不对称（`w=4` 时重心落在 `t−0.5`），画趋势线与原序列对不齐；`centered=True` 用了未来观测，拿它做"预测"属于信息泄漏；单边窗口做预测会把转折点抹平。
- **怎么检验**：对严格线性序列 `y_t = a + b·t`，尾部窗口的 `trend[t]` 应恰好等于 `y_t − b(w−1)/2`（闭式），奇数 `w` 的居中窗口应恰好等于 `y_t`（闭式，线性趋势被逐点精确复现）；常数序列任意 `w` 下 `trend ≡ 常数`；有效区间之外必须全是 `NaN`；与 pandas `Series.rolling(w).mean()` 或 `scipy.ndimage.uniform_filter1d` 在有效区间上逐点对拍。

#### `exponential_smoothing(y, alpha, initial=None)`

- **数学形式**：`l_t = α·y_t + (1−α)·l_{t−1}`，一步预测 `ŷ_t = l_{t−1}`；展开即 `l_t = Σ_{j≥0} α(1−α)^j y_{t−j}`，权重按几何衰减。
- **步骤**：① 取初值 `l_0 = y[0]`（或显式 `initial`）；② 从 `t=1` 起递推更新水平；③ `fitted[0] = l_0`、`fitted[t] = l_{t−1}`；④ 下一期预测 `forecast = l_{n−1}`。
- **复杂度**：时间 O(n)，空间 O(n)（额外空间 O(1)）。
- **参数**：`alpha`（必填，落在 (0,1)）——越大越贴近近期观测、响应快但更抖，`α→1` 时退化为"最后一期平推"；越小越平滑但滞后越重；`initial`（`None`）——`None` 表示用 `y[0]`，短序列上初值影响前若干期，`y[0]` 恰是异常点时会带偏前几期预测，论文必须报告 `initial` 的取法。
- **陷阱**：只能刻画**水平项**，对有趋势/季节的序列会预测出一条水平直线、系统性低估（或高估），有趋势请用 `holt_linear`、有季节请用 `holt_winters`；本函数**不做** α 的自动优化（不跑 MLE/最小 SSE），α 必须由使用者给定，这样每一步才可复现；`α→0` 时 `forecast` 几乎恒等于初始水平。
- **怎么检验**：常数序列上 `level` 必须恒等于该常数（任意 α，闭式）；`α=1` 时 `forecast` 应与 `naive_forecast(method="last")` 逐点相同；把递推结果与几何级数闭式 `αΣ_j (1−α)^j y_{t−j}` 手算对比；与 statsmodels `SimpleExpSmoothing(...).fit(smoothing_level=α, optimized=False, initial_level=l0)` 对拍 `fitted`。

#### `holt_linear(y, alpha, beta, initial_level=None, initial_trend=None)`

- **数学形式**：`ŷ_t = l_{t−1} + b_{t−1}`；`l_t = α·y_t + (1−α)(l_{t−1} + b_{t−1})`；`b_t = β(l_t − l_{t−1}) + (1−β)b_{t−1}`；h 步预测 `l + b·h`。
- **步骤**：① 初始化 `l_0 = y[0]`、`b_0 = y[1] − y[0]`；② 对 `t ≥ 1` 递推水平与趋势；③ `fitted[t] = l_{t−1} + b_{t−1}`；④ 下一期预测 `l + b`。
- **复杂度**：时间 O(n)，空间 O(n)。
- **参数**：`alpha`（必填，∈(0,1)）水平平滑系数；`beta`（必填，∈(0,1)）趋势平滑系数——越小趋势越"迟钝"、越平稳，越接近 1 越被最后一期的噪声主导、预测线剧烈摆动（这在真实数据上很常见，不是 bug）；`initial_level`（`None`→`y[0]`）；`initial_trend`（`None`→`y[1] − y[0]`，对前两点的噪声极敏感，序列开头波动大时建议显式传入前 1/4 段的平均斜率）。
- **陷阱**：预测是**直线**、会无限外推趋势，遇到增长饱和或周期性数据，外推几步后就会明显偏离，长期预测前务必做残差诊断；α、β 都不做自动优化；`beta` 取大导致的剧烈摆动是算法性质而非实现错误；两个初值都会影响前若干期，报告里不能省略。
- **怎么检验**：对严格线性序列，末期 `trend` 应收敛到真实斜率（取 `alpha=beta=1` 时一步即到位的退化行为可直接核对），且 `forecast` 应等于最后一点加上该斜率；`alpha→0, beta→0` 时水平与趋势应保持为初值（极限行为）；与 statsmodels `Holt(...).fit(smoothing_level=α, smoothing_trend=β, initial_level=…, initial_trend=…, optimized=False)` 逐点对拍 `fitted`；对一步预测残差做 `ljung_box`，强拒绝说明还有可利用的自相关。

#### `holt_winters(y, period, alpha, beta, gamma, mode="additive")`

- **数学形式**：加法 `ŷ_t = l_{t−1} + b_{t−1} + s_{t−m}`，乘法 `ŷ_t = (l_{t−1} + b_{t−1})·s_{t−m}`；加法 `l_t = α(y_t − s_{t−m}) + (1−α)(l_{t−1}+b_{t−1})`、`s_t = γ(y_t − l_t) + (1−γ)s_{t−m}`，乘法把两个减法换成除法；趋势式两者相同 `b_t = β(l_t − l_{t−1}) + (1−β)b_{t−1}`；h 步预测 `(l + b·h) ⊕ s_{(n+h−1) mod m}`。
- **步骤**：① 用前两个完整季节做**经典分解初始化**（加法使季节因子和为 0、乘法使其均值为 1），水平初值取第一季均值、趋势初值取两季均值的平均斜率；② 对 `t ≥ 2m`（`j = t mod m`）递推水平/趋势/该相位的季节因子；③ 前 `2m` 个 `fitted` 位置为 `NaN`；④ 输出未来 1..m 期的预测。
- **复杂度**：时间 O(n)，空间 O(n + m)。
- **参数**：`period`（必填，m）——**必须由业务周期决定**（月度 m=12、季度 m=4），取错时 γ 会把趋势的一部分吸进季节因子、预测形状系统性走偏；`alpha`/`beta`/`gamma`（必填，均 ∈(0,1)）——γ 越大季节因子更新越快、越容易吸收噪声，γ 越小季节形状越固定；`mode`（`"additive"`）——乘法适合"季节幅度随水平成正比放大"的数据，但要求序列严格为正。
- **陷阱**：**乘法模式要求数据严格为正**，含 0 或负值时 `y_t / s_{t−m}` 无意义，本实现直接抛 `ValueError`（不是返回 `NaN`/`inf` 一路污染），正确做法是改用加法模式或先平移/取对数并说明；乘法模式初季节因子出现非正值也抛错；前 `2*period` 期 `fitted` 是 `NaN`，**整段直接算 MAPE 会被污染**，必须先用 `np.isfinite` 过滤；季节因子在全样本上是常数形态，形态随时间漂移时应改用 STL 类方法；`period` 不要用 ACF 峰值随口定。
- **怎么检验**：在"常数水平 × 纯周期季节因子"的合成序列上，加法模式的 `fitted` 必须一步不差地复现原序列、`trend ≈ 0`、季节因子和恰为 0；纯周期序列上 `(l + b·h) ⊕ s` 的多步预测应与上一周期同相位观测相等（构造已知答案）；与 statsmodels `ExponentialSmoothing(..., seasonal='add'/'mul', initialization_method='known').fit(...)` 对拍；用 `rolling_origin_cv` 做样本外滚动回测，RMSE 必须打赢 `naive_forecast(method="seasonal")` 才有存在价值。

#### `ar_model(y, p)`

- **数学形式**：去均值序列满足 `x_t = Σ_{j=1..p} φ_j x_{t−j} + ε_t`；**Yule-Walker 方程** `Rφ = r`（`R` 为 Toeplitz 自协方差矩阵）用 Levinson-Durbin 递推求解，`σ² = r_0 − Σ_j φ_j r_j`；常数项 `μ(1 − Σ_j φ_j)`。
- **步骤**：① 去均值 `x = y − mean(y)`；② 用 **1/n 归一化**算自协方差 `r_0..r_p`；③ Levinson-Durbin 递推解 Yule-Walker，同时得到 φ、PACF、σ²；④ 由去均值关系还原常数项。
- **复杂度**：时间 O(n·p + p²)，空间 O(n + p)。
- **参数**：`p`（必填，正整数）——阶数越大越能刻画复杂自相关，但参数方差与过拟合风险上升、且 `p` 接近 `n` 时递推会因方差非正抛 `ValueError`（这是保护而非失败）；建议用 AIC/BIC 或 `rolling_origin_cv` 的样本外误差定阶，**不要目测 PACF 定阶**；`p` 需满足 `p ≤ n−2`。
- **陷阱**：**只对平稳序列有意义**——含趋势或单位根的序列直接做 Yule-Walker 会得到 `Σφ ≈ 1` 的伪回归结果、t 检验全部失效，应先差分或先做 ADF 检验；归一化用 1/n（保证自协方差矩阵半正定）还是 1/(n−k)（无偏）会改变系数估计，论文必须写明口径；这是**矩估计（自协方差法）**，不是条件最小二乘/极大似然，小样本或接近单位根时三者可以差不少。
- **怎么检验**：模拟已知 AR(2)（如 φ=(0.5, −0.3)）的长序列，估计应收敛到真值；AR(1) 有闭式 `φ = ρ_1`、`σ² = γ_0(1 − ρ_1²)`，可直接对拍；`pacf` 的截尾位置应等于真实阶数；与 statsmodels `yule_walker` / `AutoReg(..., method='yw')` 对拍系数；用残差 `e_t = x_t − Σφ_j x_{t−j}` 做 `ljung_box`，不拒绝才说明阶数够。

#### `difference(y, order=1, seasonal=None)`

- **数学形式**：普通差分算子 `(1−B)^d`、季节差分算子 `(1−B^m)`；本实现**先做季节差分、再做普通差分**，返回长度 `n − order − (seasonal or 0)` 的数组。
- **步骤**：① 校验 `order ≥ 0`、`seasonal`（若给）`≥ 2`；② 若给 `seasonal`，先算 `out[m:] − out[:-m]`；③ 重复 `order` 次 `np.diff`；④ 任何一步长度不足即抛 `ValueError`。
- **复杂度**：时间 O((order+1)·n)，空间 O(n)。
- **参数**：`order`（`1`）——普通差分次数；过大引入**过度差分**（白噪声被放大成 MA 结构、预测反而变差），判断信号是差分后 ACF 在滞后 1 出现约 −0.5 以下的负值；`seasonal`（`None`）——季节周期 m，只做**一次**季节差分（不做高阶），至少丢掉一个整周期。
- **陷阱**：**差分后序列变短，与 `rolling_origin_cv`（或 `train_test_split_ts`）联用时必须注意对齐**：正确的做法是先在原序列上切分，或在生成折索引时把损失的前 `order + (seasonal or 0)` 个点显式补回/偏移，否则测试段与差分序列相位错开，误差会凭空变大或变小；差分顺序与次数必须交代清楚（`d=1, D=1` 与 `d=1` 完全是两个模型）；本函数**不做逆差分**，还原时要先还原普通差分再还原季节差分；短序列做两次以上差分后所剩无几。
- **怎么检验**：k 次多项式序列的 k 阶差分应为常数（如 `k²` 的二阶差分恒为 2，闭式）；随机游走差分后应近似白噪声（`ljung_box` 不拒绝）；`order=0` 必须返回原序列副本；与 pandas `Series.diff(m)` / `np.diff` 手工左对齐的结果对拍长度与数值；把差分序列的 ACF 与"过度差分"的理论形态（滞后 1 接近 −0.5）对照。

#### `acf(y, nlags)`

- **数学形式**：`ρ_k = r_k / r_0`，其中 `r_k = (1/n)·Σ_{t=k}^{n−1} (y_t − ȳ)(y_{t−k} − ȳ)`；`ρ_0 = 1`。
- **步骤**：① 中心化；② 按定义做向量化点积得到 `r_0..r_k`；③ 除以 `r_0`（方差为 0 抛 `ValueError`）。
- **复杂度**：时间 O(n·nlags)，空间 O(nlags)。
- **参数**：`nlags`（必填，需 `≤ n−1`）——取得太大（接近 n）时每个滞后的有效样本急剧减少、估计噪声极大，经验取 `min(10, n/5)` 或 `2×周期长度` 比较稳。
- **陷阱**：归一化口径必须写明——本实现用 **1/n（有偏）**，`k` 大时自相关被系统性压低，某些软件用 1/(n−k)（无偏），n 小 k 大时两者可以差 10% 以上；`±2/√n` 参考线只是**白噪声**下的渐近近似，序列有趋势或季节时 ACF 衰减极慢是正常现象，不能据此断言"存在长期记忆"；缺失值与异常点会把 ACF 整体拉高，先清洗再看图。
- **怎么检验**：白噪声的 ACF 约有 95% 落在 `±2/√n` 内（可构造 iid 正态序列断言）；MA(q) 的 ACF 应在滞后 q 之后截尾；AR(1)（φ=0.7）的 `ρ_k` 闭式为 `φ^k`，可逐点对拍；与 statsmodels `acf(..., adjusted=False)`（同一 1/n 口径）对拍，并同时给出 `adjusted=True` 的差异以说明口径敏感性。

#### `pacf(y, nlags)`

- **数学形式**：滞后 k 的偏自相关即 AR(k) 的最后一个系数 `φ_{kk}`，由 Durbin-Levinson 递推给出：`κ_k = (r_k − Σ_{j<k} φ_{k−1,j} r_{k−j}) / v_{k−1}`，`φ_{k,j} = φ_{k−1,j} − κ_k φ_{k−1,k−j}`，`v_k = v_{k−1}(1 − κ_k²)`。
- **步骤**：① 中心化并算 `r_0..r_k`；② 跑 Levinson-Durbin 递推；③ 各阶 `κ_k` 组成 `pacf[1..k]`，`pacf[0] = 1`。
- **复杂度**：时间 O(n·nlags + nlags²)，空间 O(nlags)。
- **参数**：`nlags`（必填，需 `≤ n−2`）——比 `acf` 更受限；取得过大时递推中途可能出现非正方差并抛错，**这本身就是"序列需要先差分"的信号**，不要靠调小 nlags 绕过。
- **陷阱**：PACF 的截尾/拖尾判读（AR(p) 的 PACF 在 p 阶后截尾）是**渐近**结论；`n` 小于约 50 时抽样波动很大，滞后 1 的 |PACF| 常常就能超过 0.3，据此把阶数定高是典型错误；与 `acf` 同为 1/n 口径，混用不同口径会让 AR 系数的估计与别人对不上；不平稳序列会让递推报"非正方差"错误。
- **怎么检验**：AR(p) 数据的 PACF 在滞后 > p 应落在 `±2/√n` 内（可模拟 AR(2) 断言）；AR(1) 的 PACF 在滞后 1 之后应接近 0；与 statsmodels `pacf(..., method='ld')` 对拍；再用**逐阶 OLS 回归**独立算"最后一个系数"，两种独立算法应给出同一数值。

#### `adf_test(y, max_lag=None, regression="c")`

- **数学形式**：`Δy_t = a + γ·y_{t−1} + Σ_{i=1..p} δ_i Δy_{t−i} + e_t`（`regression="ct"` 再加时间趋势项），原假设 `γ = 0`（存在单位根）；统计量为 `t = γ̂ / se(γ̂)`，`se` 由 `σ² = RSS/(n_used − k)` 与 `(X'X)^{-1}` 得出；滞后阶数在 `p = 0..max_lag` 中按 `AIC = n_used·ln(RSS/n_used) + 2k` 选；临界值来自 `mackinnon_crit`。
- **步骤**：① 确定滞后上限（`None` 走 Schwert 经验规则 `ceil(12(n/100)^{1/4})`）；② 对每个候选 `p` 构造回归矩阵、用最小二乘闭式解估计，按 AIC 取最优 `p`；③ 用选中阶数的 t 统计量与 MacKinnon 临界值比较；④ 一并返回 `nobs` 供查表核对。
- **复杂度**：时间 O(max_lag · n · max_lag²)（每个候选阶数解一次最小二乘），空间 O(n · max_lag)。
- **参数**：`max_lag`（`None`）——`None` 用 Schwert 规则，给定整数则在该上限内用 AIC 选阶（便于复现）；滞后太少残差自相关没清干净、统计量有偏，太多则损失样本；`regression`（`"c"`）——`"c"` 含常数项、`"ct"` 含常数项与线性趋势、`"n"` 无常数项，选错会让统计量与临界值的口径不匹配、结论可能反转。论文里必须写明 `max_lag` 与选阶准则。
- **陷阱**：ADF 的原假设是**存在单位根（序列非平稳）**，统计量**越小（越负）**才越倾向拒绝，把结论说反是本类检验最常见的低级错误；**临界值来自 `mackinnon_crit` 的响应面查表近似，是渐近分布的分位点近似，且本模块不提供 MacKinnon 的 p 值近似——论文里不要报一个精确到小数点后四位的 p 值，只报"在 5%（或 1%/10%）水平上不能拒绝单位根"这类结论**；ADF **对结构突变无能为力**，序列中期水平跳变会被误判为单位根（Perron 批评），有突变应先做突变检验或分段处理；回归里必须含足够滞后项以消除残差自相关，`nobs` 随之减少；只做一次 ADF 就下结论不够，应配合 ACF 衰减、PP 检验或 KPSS（原假设相反）作为交叉证据。
- **怎么检验**：纯随机游走上不应在 5% 水平拒绝单位根，iid 白噪声上应拒绝（构造已知答案的数据）；`mackinnon_crit` 在 `nobs → ∞` 时应回到教科书渐近临界值（含常数项情形 1%/5%/10% 约为 −3.43 / −2.86 / −2.57）；与 statsmodels `adfuller(regression=…, autolag='AIC')` 对拍统计量（选阶可能差 1 阶，统计量因此可能差 0.1 以上，必须说明口径）；对同一序列做"整段 vs 分段"的 ADF，用结论差异演示结构突变的影响。

#### `mackinnon_crit(nobs, regression="c", level=0.05)`

- **数学形式**：响应面近似 `crit(n) = b0 + b1/n + b2/n² + b3/n³`，系数取自 MacKinnon (2010) Table 1 的 N = 1（单变量 ADF）行；`n → ∞` 时 `crit → b0`，即教科书中常引用的渐近临界值。
- **步骤**：① 规范化并校验 `regression`、`level`；② 查表取出四个系数；③ 代入 `1/n` 的三次多项式返回。
- **复杂度**：时间 O(1)，空间 O(1)。
- **参数**：`nobs`（必填）——必须是 **ADF 回归实际使用的观测数**（差分一次、再加 p 个滞后项之后它比原始序列长度小），传错会让临界值与统计量不匹配、结论可能反转；`regression`（`"c"`）——只接受 `"c"` / `"ct"` / `"n"`，不接受 `"ctt"`（常数 + 线性 + 二次趋势未实现）；`level`（`0.05`）——只支持 0.01 / 0.05 / 0.10。
- **陷阱**：这只是**查表近似**，`nobs` 很小时（例如 < 20）误差较大，本实现的下限是 5，但实际建模中 `nobs` 应远大于此；这里只有 **N = 1（单变量）** 的系数，**不能用于协整检验**（N > 1 需要另一张表）；响应面给出的是渐近分布分位点，序列有结构突变或条件异方差时实际检验水平会明显偏离名义水平；需要精确 p 值请用 statsmodels，本模块不提供 Thompson 型 p 值响应面。
- **怎么检验**：取极大 `nobs` 时输出应等于教科书渐近临界值（`"c"` 情形约 −3.43 / −2.86 / −2.57）；与 statsmodels `statsmodels.tsa.adfvalues.mackinnoncrit(N=1, regression=…, nobs=n)` 逐点对拍；固定 `nobs` 时临界值应随 `level` 从 1% 到 10% **单调变大**（越宽松越容易拒绝）；把 `nobs` 从 20 扫到 10⁴，检查临界值单调趋近 `b0`。

#### `mape(y_true, y_pred)`

- **数学形式**：`MAPE = (100/n)·Σ_t |(y_t − ŷ_t)/y_t|`，返回百分数（`12.3` 表示 12.3%）。
- **步骤**：① 把真值与预测值转成等长一维数组；② 若真值含 0 直接抛 `ValueError`；③ 逐点算绝对百分比误差、取均值、乘 100。
- **复杂度**：时间 O(n)，空间 O(n)。
- **参数**：只有 `y_true`/`y_pred` 两个数据参数，没有可调超参——它是被报告的评价量，而不是被调的对象；做模型选择时它充当目标函数，这个角色的敏感性分析应放在"改用哪个指标"上（见"陷阱"）。
- **陷阱**：真值含 0 时本实现**直接抛错**（除以 0 会得到 `inf`，均值变成 `inf`/`NaN`，图表全毁）；**真值接近 0 时 MAPE 同样会爆炸**（分母 0.01 会把微小绝对误差放大成 1000%），这种数据应当**改报 RMSE 或 MAE（或 MASE）**，并在论文里说明为什么不用 MAPE；MAPE 对高估与低估**不对称**（惩罚"预测偏大"的力度小于"偏小"），用 MAPE 选参会系统性偏向低估；真值有正有负时百分比无意义。
- **怎么检验**：`y_true == y_pred` 时必须为 0；把预测统一放大 10% 时（真值远离 0）MAPE 应接近 10，可手算核对；与 sklearn `mean_absolute_percentage_error` 或手写公式对拍；构造含 0 的真值断言抛 `ValueError`（把限制写成被测行为）；对同一预测同时算 MAPE/RMSE/MAE，比较三者给出的模型排序是否一致。

#### `rmse(y_true, y_pred)`

- **数学形式**：`RMSE = sqrt((1/n)·Σ_t (y_t − ŷ_t)²)`，分母取 n（总体口径，这是预测误差的通用定义），与数据同量纲。
- **步骤**：残差平方取均值后开方。
- **复杂度**：时间 O(n)，空间 O(n)。
- **参数**：无可调超参（它是最小二乘意义下的自然目标）。
- **陷阱**：RMSE 对**大误差特别敏感**（平方放大），少数离群点就能主导数值；比较模型时若测试段含突变点，RMSE 的排序可能完全由那一个点决定，建议同时报告 MAE；只报 RMSE 而不说明数据量纲没有意义（"RMSE = 3.2"无法判断好坏），应给出 `RMSE/均值` 或改用 MASE 归一化。
- **怎么检验**：`y_true == y_pred` 时为 0；常数偏移 c 时闭式为 `|c|`；必须恒有 `MAE ≤ RMSE ≤ √n·MAE`（可用随机数据断言，也是三条指标口径一致的交叉验证）；与 `np.sqrt(np.mean((a-b)**2))` 及 sklearn `mean_squared_error(..., squared=False)` 对拍。

#### `mae(y_true, y_pred)`

- **数学形式**：`MAE = (1/n)·Σ_t |y_t − ŷ_t|`。
- **步骤**：绝对误差取均值。
- **复杂度**：时间 O(n)，空间 O(n)。
- **参数**：无可调超参。
- **陷阱**：MAE 的最优预测是**条件中位数**、RMSE 的最优预测是**条件均值**，两条指标同时报告才说明误差分布的形状；只看 MAE 会掩盖少数大偏差（例如极端月份被严重低估）；MAE 与 RMSE 的差距本身是"误差是否厚尾"的信号。
- **怎么检验**：常数偏移 c 时闭式为 `|c|`；`y_true == y_pred` 时为 0；`MAE ≤ RMSE` 必须成立；与 sklearn `mean_absolute_error` 对拍；对含离群点的合成数据，验证 MAE 排序不受该点主导而 RMSE 会（说明两者互补）。

#### `theil_u(y_true, y_pred)`

- **数学形式**：本实现是 **Theil U2**：`U = ‖ŷ − y‖₂ / (‖y‖₂ + ‖ŷ‖₂)`，其中 `‖·‖₂` 是未除以 n 的欧氏范数；`U` 落在 [0, 1]，0 表示完美预测，`U = 1` 对应"预测与真值成比例但符号相反"这类极端情形（**包含"全预测 0"**）。
- **步骤**：① 校验等长；② 分母 = 真值范数 + 预测值范数，若 ≤ 0 抛 `ValueError`；③ 分子 = 误差范数。
- **复杂度**：时间 O(n)，空间 O(n)。
- **参数**：无可调超参。
- **陷阱**：Theil 有 **U1 与 U2 两个不同定义**，数值完全不同（U1 是 `sqrt(mean(((ŷ−y)/y)²))`，同样会被 0 值毁掉），论文必须写明用的是哪一个，本实现是 U2；**"U < 1 才优于朴素预测"这条判据属于"以朴素预测误差为分母"的那个口径（相对 Theil U）**，而本实现的 U2 分母含预测值自身，`U = 1` 只对应"全预测 0"这种平凡预测，**不能直接读出是否打赢了随机游走/季节朴素基线**——要回答"有没有优于朴素预测"，请另算 `RMSE(模型)/RMSE(朴素基线)`，它小于 1 才说明模型带来了信息；U2 的分母含预测值，因此只有**同一测试集**上的 U2 才可比，跨数据集比较大小无意义。
- **怎么检验**：`ŷ == y` 时闭式为 0；`ŷ ≡ 0` 时闭式为 1（可直接断言）；取 `ŷ = −c·y`（c > 0）时由三角不等式取等，U 应为 1；与手写公式对拍；把模型与 `naive_forecast` 的**同测试集**误差比值（RMSE 之比或 MASE）一并报告，作为"是否优于朴素"的真正判据。

#### `train_test_split_ts(y, test_size)`

- **数学形式**：纯下标切分 `train = y[:n−k]`、`test = y[n−k:]`，其中 `test_size` 为 `int` 时 `k = test_size`，为 `(0,1)` 内 `float` 时 `k = ceil(n · test_size)`。
- **步骤**：① 解析 `test_size`（拒绝 `bool`；`int` 取条数；`float` 取比例并向上取整）；② 校验 `1 ≤ k < n`（训练集不能为空）；③ 返回前缀与后缀。
- **复杂度**：时间 O(n)，空间 O(n)。
- **参数**：`test_size`（必填）——`int` 表示条数、`(0,1)` 内 `float` 表示比例（向上取整）；调大测试集会让训练段变短、误差更悲观但评估更稳，调小则单次评估噪声大；本函数**刻意不提供 `shuffle` 参数**。
- **陷阱**：**时间序列绝对不能随机切分**——随机抽点会把"用未来的点预测过去"的信息泄漏带进测试集，得到远低于真实水平的误差，一旦用于决策就会翻车；单次切分的测试段可能恰好落在某个特殊季节或突变段上，误差不具代表性，正式报告建议改用 `rolling_origin_cv` 做多折滚动外推；切分后**不能跨边界做平滑/差分/标准化**——差分要用 `y[n−k−1]` 才能得到测试段第一个差分值，标准化统计量只能用训练段计算，否则同样是信息泄漏。
- **怎么检验**：断言 `len(train) + len(test) == n`、`train[-1]` 紧邻 `test[0]`、`test[-1] == y[-1]`、两段无交集；对 20 点序列用比例 0.25 应得 15/5；与 sklearn `TimeSeriesSplit` 的语义对照（后者给的是滚动折而非单次切分，可用来验证"顺序切分"这一原则）；把随机切分的误差与顺序切分的误差对比，用数值演示信息泄漏有多大。

#### `rolling_origin_cv(y, initial, horizon, step=1)`

- **数学形式**：第 j 折为三元组 `(train_end, test_start, test_end) = (initial+(j−1)·step, 同值, 同值+horizon)`，只要 `train_end + horizon ≤ n`；训练集 `y[:train_end]`、测试集 `y[train_end:train_end+horizon]`（**扩张窗口**，测试窗口紧接训练窗口、中间不跳空）。
- **步骤**：① 校验 `initial`、`horizon`、`step` 均为正整数；② `initial + horizon > n` 时抛 `ValueError`；③ 循环生成折索引列表。
- **复杂度**：时间 O(n/step)、空间 O(n/step)（只生成索引、不训练模型，真正的成本在调用方每折的拟合上）。
- **参数**：`initial`（必填）——第一折训练窗长，**至少要覆盖模型阶数加一个完整季节周期**，否则前几折的"预测"其实等于外推噪声，会污染汇总误差；`horizon`（必填）——每折预测步长，`>1` 时应按步长分别汇总误差（第 1 步与第 h 步的难度不同）；`step`（`1`）——窗口前进步数，越大折数越少、折间重叠越少，越小则折间重叠越多、误差序列自相关越强。
- **陷阱**：训练窗口是**扩张式**的（`y[:train_end]` 随折数变长），要固定长度滑动窗口必须自行取 `y[train_end−w:train_end]`，本函数只给边界；各折误差**不能简单平均**就下结论（不同折落在不同季节/不同波动区间），应同时报告每折误差与折间波动；折与折之间存在重叠，误差序列本身自相关，**不能套用 i.i.d. 的置信区间公式**；`initial` 太小的前几折误差没有意义。
- **怎么检验**：断言每折 `train_end == test_start`、`test_end − test_start == horizon`，且 `test_end ≤ n`；断言折数等于 `floor((n − initial)/step) + 1`（在 `initial + horizon ≤ n` 时）；断言所有测试区间互不重叠且按时间递增；在固定折索引上把 `naive_forecast` 与复杂模型的样本外 RMSE 直接比较，检查模型的增益是否在多数折上都成立（而不是被某一折拉动）。

#### `naive_forecast(x, n_ahead=1, method="last", period=None)`

- **数学形式**：`"last"`：`f_h = x_{n−1}`；`"mean"`：`f_h = mean(x)`；`"drift"`：`f_h = x_{n−1} + h·(x_{n−1} − x_0)/(n−1)`；`"seasonal"`：`f_h = x_{n−m+((h−1) mod m)}`。四种方法都只用历史数据、**不含任何待估参数**。
- **步骤**：① 校验 `method` 与 `n_ahead`；② 按方法算出长度 `n_ahead` 的预测数组；③ 回显实际使用的 `method`。
- **复杂度**：时间 O(n + n_ahead)，空间 O(n_ahead)。
- **参数**：`n_ahead`（`1`）——预测步数；`method`（`"last"`）——`"last"` 适合随机游走型序列、`"mean"` 适合平稳无趋势序列、`"drift"` 适合线性趋势、`"seasonal"` 适合强周期（必须给 `period`）；`period`（`None`）——季节周期 m，仅 `"seasonal"` 使用，`None` 时抛 `ValueError`。
- **陷阱**：朴素方法在基准对比里是**及格线而不是模型**——复杂模型若打不赢 `"last"`/`"seasonal"`，说明它的参数估计没有带来信息（Hyndman 的 MASE 正是用季节朴素法做分母），论文里必须报告这条基线；`"seasonal"` 的**相位对齐依赖 `x` 的起点**，序列被截断过（例如从某年 3 月开始）时 `period` 的相位就错了、误差凭空变大，应先按相位补齐或改用 `seasonal_decompose` 显式估季节因子；`"drift"` 把首末两点的噪声当成斜率，对首末异常值极其敏感、误差随外推步数线性放大；`"mean"` 对含趋势的序列系统性滞后，不要因为它"误差曲线平滑"就选用。
- **怎么检验**：纯周期序列上 `"seasonal"` 的多步预测应与上一周期同相位观测一步不差（构造已知答案）；严格线性序列（如 1..10）上 `"drift"` 应精确外推为 11, 12, 13；`"last"` 应与 `x[-1]` 逐点相等、`"mean"` 应与 `mean(x)` 相等；与 statsmodels `NaiveForecaster(strategy='last'/'mean'/'drift'/'seasonal', sp=m)` 对拍；把它作为分母构造相对误差（MASE 思路）来评价其他模型。

#### `seasonal_decompose(x, period, model="additive", n_iter=2)`

- **数学形式**：加法 `x = T + S + R`、乘法 `x = T · S · R`；`T` 由居中移动平均给出（奇数 m 等权窗口，偶数 m 为端点权重 0.5 的 2×m 平均，能逐点精确复现线性趋势）；`S` 由去趋势序列按相位平均后归一（加法减均值使其和为 0，乘法除均值使其均值为 1）；`R` 由 `x ⊖ T ⊖ S` 得到；强度指标 `max(0, 1 − Var(R)/Var(对照分量))`。
- **步骤**：① `S` 初值取 0（加法）或 1（乘法）；② 重复 `n_iter` 轮：去季节 → 居中移动平均估 `T` → 去趋势 → 按相位平均得 `S` 并归一 → 按相位延拓成长度 n；③ 由最终的 `T`、`S` 算 `R` 与两个强度指标。
- **复杂度**：时间 O(n_iter · n · m)，空间 O(n)。
- **参数**：`period`（必填，m）——季节周期，取错时季节项会把趋势吸收进去、残差看起来"变小"、强度指标反而变好看（**强度高不等于模型对**），必须与 ACF 的周期证据和业务周期共同判断；`model`（`"additive"`）——乘法适合季节幅度随水平成正比的数据，但要求序列严格为正；`n_iter`（`2`）——每轮先用上一轮的季节项去季节再重估趋势，用来冲洗"季节混进趋势"的污染，迭代只能减轻不能根治形态漂移（形态漂移明显应改用 STL）。
- **陷阱**：移动平均趋势**本身会被季节项污染**——序列长度不是周期的整数倍或真实季节形态随时间变化时，居中平均里残留的季节成分会被塞进趋势，表现为趋势线出现周期性"波纹"；首尾各 `half = ⌊m/2⌋` 个点没有趋势值，因此残差与强度**只在中段计算**，序列长度刚好 `2*period` 时可用的点极少、强度极不稳定，不要据此下结论；乘法模式要求 `x`、趋势与初季节因子全为正，本实现直接抛 `ValueError` 而不是返回 `NaN`；本函数是**确定性描述、不是概率模型、不提供预测**，要预测请用 `holt_winters` / `holt_winters_multiplicative`。
- **怎么检验**：构造 `x = 线性趋势 + 已知周期季节项 + 0 残差`，分解出的季节因子应回到真值、残差应到浮点噪声量级（1e-9 以下）、两个强度应接近 1（这些是闭式可断言的结果）；构造乘法版本 `x = 线性趋势 × (1 + a·sin)`，季节因子应回到归一后的真值、残差应接近 1；与 statsmodels `seasonal_decompose(model='additive'/'multiplicative', two_sided=True)` 对拍趋势与季节项（须说明迭代次数与首尾处理的差异）；把趋势项与线性回归斜率/`difference` 结果对照；对残差做 `ljung_box`，强拒绝说明周期没提干净。

#### `holt_winters_multiplicative(x, period, alpha, beta, gamma, n_ahead=1)`

- **数学形式**：与 `holt_winters(mode="multiplicative")` 逐字一致的递推：`ŷ_t = (l_{t−1} + b_{t−1})·s_{t−m}`；`l_t = α(y_t / s_{t−m}) + (1−α)(l_{t−1} + b_{t−1})`；`b_t = β(l_t − l_{t−1}) + (1−β)b_{t−1}`；`s_t = γ(y_t / l_t) + (1−γ)s_{t−m}`；多步预测 `f_h = (l + b·h)·s_{(n+h−1) mod m}`。
- **步骤**：① 用前两个完整季节做经典分解初始化（`base = l_0 + b_0·t`，去趋势序列按相位平均后除以其均值）；② 对 `t ≥ 2m`（`j = t mod m`）递推三个状态；③ 前 `2m` 个 `fitted` 为 `NaN`；④ 按 `n_ahead` 输出任意步长的预测。
- **复杂度**：时间 O(n + n_ahead)，空间 O(n + m)。
- **参数**：`x` 必须严格为正（乘法模型的前提）；`period`/`alpha`/`beta`/`gamma` 的含义与默认要求同 `holt_winters`（三个平滑系数无默认值、必须显式给出，落在 (0,1)）；`n_ahead`（`1`）——不再限制为 `period`，可一次给出任意步长的预测，但步数越大越受趋势外推误差影响。
- **陷阱**：强制乘法，**数据必须严格为正**，含 0 或负值抛 `ValueError`（与 `holt_winters` 同口径），常见兜底是**先取对数再套加法模式**；季节因子被数据波动推到 0 附近、或水平项退化时递推会爆炸，本实现对两者都显式抛 `ValueError`；序列长度刚好 `2*period` 时递推循环一次都不执行，`fitted` 全是 `NaN`、只有 `level`/`trend`/`seasonal`/`forecast` 可用，这是初始化代价不是 bug；前 `2*period` 个 `fitted` 是 `NaN`，整段算 MAPE 会被污染，必须先用 `np.isfinite` 过滤；预测无限复用末期季节因子，形态漂移时多步预测会走偏。
- **怎么检验**：在"常数水平 × 纯周期季节因子"序列上，`fitted` 必须一步不差地复现原序列、`trend ≈ 0`、`level ≈ 常数`，且 `n_ahead > m


### 3.5 统计推断与回归 —— `examples/algorithms/statistics.py`

**这族解决什么问题**：把相关系数、均值/列联表/正态性检验、OLS 与正则回归、广义线性回归、重抽样推断和降维（PCA/因子分析）做成一批只依赖 numpy 与标准库的透明实现，每个函数直接吐出能打印成论文表格的字典。

**共同约定**：
- **分布函数全部自己实现**：正态用 `math.erf`/`erfc`，t / 卡方 / F / 不完全 Beta 的尾概率走 `_betainc`、`_gammainc_q` 的连分式与级数，不 import scipy（模块 docstring 记录这些私有函数在交付报告中与 `scipy.special`/`scipy.stats` 做过数值对比；代码本身不含该对照，CI 用 AST 静态禁止 scipy）。
- **p 值口径**：除卡方、Jarque-Bera、Anderson-Darling、Breusch-Pagan 本身就取右尾外，其余检验返回的都是**双侧** p 值，键名统一叫 `p_value`。
- **随机性**：`bootstrap_ci` / `permutation_test` 显式接收 `seed`，`None` 时回落到 `_common.rng` 的 `DEFAULT_SEED`，不使用 `np.random` 全局状态。返回形态不完全统一：绝大多数返回 dict，`durbin_watson` 返回裸 float（不是笔误）。

#### `pearson_corr(x, y)`

- **数学形式**：r = Σ(xᵢ−x̄)(yᵢ−ȳ) / √(Σ(xᵢ−x̄)² · Σ(yᵢ−ȳ)²)；检验统计量 t = r·√((n−2)/(1−r²))，在 H₀: ρ=0 下服从 t(n−2)
- **步骤**：① 校验 x/y 等长且 n ≥ 3；② 去均值得离差；③ 算 r 并截断到 [−1, 1]；④ 由 t 用恒等式 P(|T|>t) = I_{df/(df+t²)}(df/2, 1/2) 取双侧 p；⑤ 返回 `coef`/`p_value`/`n`
- **复杂度**：O(n) 时间（两次向量化求和），O(n) 空间（存离差）
- **参数**：`x`/`y`（等长序列）——长度不等、n < 3、任一列为常数都抛 `ValueError`；本函数**没有 alpha 参数**，它给的是 p 值而不是置信区间，要区间请用 `bootstrap_ci`
- **陷阱**：只度量**线性**关系（抛物线也能给出 r ≈ 0）；p 值来自 t 近似，前提是近似二元正态，强离群点/重尾/非线性时不可信；|r| = 1 时 t 分母为 0，本实现直接返回 `p_value = 0.0`，这个"显著性"没有实际含义；n 很小时 r 的抽样分布强烈偏斜
- **怎么检验**：与 `scipy.stats.pearsonr` 对拍 `coef` 与 `p_value`（应当只差浮点末位）；构造完全线性的 y = 2x+1 断言 `coef == 1`（`_self_test` 用这组数据，是可独立重跑的闭式结论）；插入一个远离直线的点，断言 |r| 明显下降；用 `permutation_test` 或自行打乱 y 得到置换 p 值，看与 t 近似的 p 是否同一量级

#### `spearman_corr(x, y)`

- **数学形式**：ρ_s = Pearson(rank(x), rank(y))，秩为平均秩（并列取名次均值）；p 值用 t = ρ_s·√((n−2)/(1−ρ_s²)) ~ t(n−2) 近似
- **步骤**：① 对 x、y 分别做 `_rank_average`（排序后扫描等值段取段内平均名次）；② 对两列秩求 Pearson 相关；③ 由 t 近似取双侧 p；④ 返回 `coef`/`p_value`/`n`
- **复杂度**：O(n log n) 时间（排序主导），O(n) 空间
- **参数**：`x`/`y`（等长序列）——长度不等、n < 3、常数序列抛 `ValueError`；并列值处理固定为平均秩，无开关可调；同样**无 alpha 参数**
- **陷阱**：度量的是**单调**关系，对离群点稳健，但对"两端极端、中间平坦"的数据会低估关联强度；**并列多时 t 近似的分布假设被破坏**（如 1~5 分的评分数据），近似 p 偏保守，应改用置换检验；n ≤ 10 时近似不可靠，本实现不做 exact permutation，论文里要写明用的是近似
- **怎么检验**：与 `scipy.stats.spearmanr` 对拍 `coef`（并列数据尤其要试，因为差异几乎全来自秩的口径）；构造严格单调（如 y = x³）数据断言 ρ_s = 1；把 y 的一段打乱后断言 ρ_s 下降但比 Pearson 下降得少（稳健性的定性验证）；小样本上自算全部 n! 置换的精确 p 与近似 p 对比

#### `kendall_tau(x, y)`

- **数学形式**：τ_b = (C − D) / √((C + D + T_x)(C + D + T_y))，C/D 为一致/不一致对数，T_x、T_y 为只在 x / 只在 y 上打结的对数；s = C − D，z = s/√var_s 做双侧正态近似，var_s = (v₀ − v_t − v_u)/18 + v₁ + v₂（Kendall 1970 打结修正）
- **步骤**：① 双重循环逐对比较符号乘积，累计 C、D、T_x、T_y；② 算 τ_b 并检查分母 > 0；③ 由 `np.unique` 的计数算 v₀/v_t/v_u/v₁/v₂；④ z = s/√var_s，p = 2·`_norm_sf`(|z|)；⑤ 返回 `coef`/`p_value`/`n`
- **复杂度**：O(n²) 时间（朴素双重循环，且每一轮都用 numpy 向量化内层），O(1) 额外空间
- **参数**：`x`/`y`（等长序列）——n < 3 或常数序列抛 `ValueError`；无其它可调参数，τ 口径固定为 **tau-b**（不是 tau-a/tau-c）
- **陷阱**：n 上万时很慢，大样本要用基于归并排序的 O(n log n) 实现或成熟库；p 值是**正态近似**且打结修正只到二阶，n < 30 或打结多时与精确置换 p 可有 0.01 量级差距；论文必须注明用的是 tau-b
- **怎么检验**：与 `scipy.stats.kendalltau` 对拍——**无并列与带并列两种数据都要试**（代码注释写明打结方差与 scipy 渐近口径一致，但仓库未给出逐位断言，请自己复核）；构造完全同序 / 完全反序数据断言 τ = 1 / −1，n = 3 这种小样本可直接枚举 3! = 6 种置换手算 p

#### `t_test_one_sample(x, mu=0.0)`

- **数学形式**：t = (x̄ − μ) / (s/√n)，s² 为无偏方差（ddof = 1），df = n − 1
- **步骤**：① 校验 n ≥ 2；② 算 ddof=1 的样本方差，为 0 则报错；③ 算均值差与标准误；④ 双侧 p 由 t(n−1) 尾概率给出；⑤ 返回 `stat`/`df`/`p_value`/`mean_diff`
- **复杂度**：O(n) 时间，O(n) 空间
- **参数**：`x`（一维样本，n < 2 抛 `ValueError`，样本方差为 0 抛 `ValueError`）；`mu`（默认 0.0）——原假设均值，改它只平移统计量并改变 `mean_diff` 的符号与大小，不影响 p 的计算方式
- **陷阱**：要求样本近似来自正态总体，小样本 + 明显偏斜时名义水平不准；p 值大**不等于**"均值等于 μ"，只是没有足够证据拒绝；大样本下微小差异也会显著，报告里要同时给 `mean_diff`（必要时配 Bootstrap 区间）
- **怎么检验**：与 `scipy.stats.ttest_1samp` 对拍 `statistic` 与 `pvalue`；手算小样本（如 [1,2,3,4,5] 对 μ=3）验证 t = 0、p = 1；构造已知均值与标准差的样本，检查 `mean_diff` 等于手算差值；对严重偏斜的小样本用 Bootstrap 分布代替 t 分布，比较两者 p 值的差异方向

#### `t_test_two_sample(x, y, equal_var=True)`

- **数学形式**：Student：sp² = ((n_x−1)s_x² + (n_y−1)s_y²)/(n_x+n_y−2)，t = (x̄−ȳ)/√(sp²(1/n_x+1/n_y))，df = n_x+n_y−2；Welch：t = (x̄−ȳ)/√(s_x²/n_x + s_y²/n_y)，df = (a_x+a_y)² / (a_x²/(n_x−1) + a_y²/(n_y−1))，a = s²/n
- **步骤**：① 校验两组各 ≥ 2 个观测；② 算两组均值与 ddof=1 方差；③ 按 `equal_var` 走合并方差或 Welch 分支算标准误与 df；④ 由 t 分布的双侧尾概率得 p；⑤ 返回 `stat`/`df`/`p_value`/`mean_diff`
- **复杂度**：O(n_x + n_y) 时间，O(n_x + n_y) 空间
- **参数**：`x`/`y`（长度可不同的两组独立样本；常量组导致合并方差为 0 时抛 `ValueError`）；`equal_var`（默认 **True = Student**）——改成 `False` 即 Welch 校正，两组方差差 2 倍以上时第一类错误率会明显偏离名义水平，拿不准就该设 False（等方差时几乎不损失效率）
- **陷阱**：默认的 Student 版本是最容易踩的坑；Welch 的 df 是**非整数**（正常现象，表格里应保留小数）；两组必须**独立**，前后测/配对数据用本函数是常见错误（应对差值用 `t_test_one_sample`）；仍要求近似正态
- **怎么检验**：与 `scipy.stats.ttest_ind(equal_var=True/False)` 分别对拍 `statistic` 与 `pvalue`（Welch 的 `df` 也要对）；构造两组完全相同的样本断言 `stat == 0`、`p == 1`；用等方差的两组数据检查 Student 与 Welch 结论一致，再用方差悬殊的两组数据观察两者分道扬镳；对同一数据跑 `permutation_test` 做非参数对照

#### `chi_square_test(observed)`

- **数学形式**：χ² = Σ (O_ij − E_ij)²/E_ij；二维表 E_ij = 行和·列和/总和、df = (r−1)(c−1)；一维频数在等概率原假设下 E = 总和/k、df = k−1；p 由卡方上尾概率 Q(df/2, x/2) 给出
- **步骤**：① 判断输入是 1 维还是 2 维并校验；② 按形状算 E 与 df；③ 检查 E 无 0 格；④ 累加 χ²；⑤ 返回 `stat`/`df`/`p_value`/`expected`
- **复杂度**：O(rc) 时间，O(rc) 空间
- **参数**：`observed`（2×2 以上的频数表或长度 ≥ 2 的频数向量）——不是频数（比例/百分比）、含负值、期望频数为 0 都抛 `ValueError`；**一维版本只支持等概率原假设**，自定义比例必须自己构造期望频数，本函数没有相应参数
- **陷阱**：卡方近似要求期望频数不能太小（经验规则：E ≥ 1 且 E < 5 的格子不超过 20%），2×2 小格应改 Fisher 精确检验（本模块未提供）；一维版本检验非均匀分布是明确的误用；含小数频数（加权数据）时近似不可靠；整行或整列为 0 会得到 0/0
- **怎么检验**：与 `scipy.stats.chi2_contingency`（二维）/ `scipy.stats.chisquare`（一维）对拍 `statistic`、`dof`、`pvalue` 与期望频数矩阵；构造完全独立表（如 [[10,20],[20,40]]）断言 χ² = 0；构造已知依赖的表手算 χ²；用 df = 2 时右尾恰为 exp(−x/2) 的解析结论核对 p 值

#### `shapiro_wilk(x)`

- **数学形式**：无——**本函数不实现任何统计量，调用即抛 `NotImplementedError`**
- **步骤**：① 对 x 做一次 `as_vector` 形状校验；② 无条件抛 `NotImplementedError`，异常信息里给出替代方案；没有返回路径
- **复杂度**：无（O(1) 校验后立刻抛出）
- **参数**：`x`（一维样本）——**参数在计算上完全不被使用**，只是保留签名以便调用方迁移；函数没有 `alpha`、没有开关可以打开实现
- **陷阱**：这是**有意的缺失**：W 统计量的权重 aᵢ 依赖 Royston (1995) / AS R94 的分段系数表，本仓库无法在不引入 scipy 的前提下逐项核对该表，宁可不提供也不返回编造的 W 与 p。把 Blom 分数 mᵢ/‖m‖ 当作 aᵢ 算出来的是 Shapiro-**Francia** 的统计量，却标成 "Shapiro-Wilk"，属于张冠李戴
- **怎么检验**：断言调用必然抛 `NotImplementedError`（`_self_test` 就这样做）；需要正态性结论时改走 `scipy.stats.shapiro`、`anderson_darling`、`jarque_bera` 或 `ks_test_normal`，并交叉比较这几个检验在同一份数据上的结论是否自洽

#### `jarque_bera(x)`

- **数学形式**：m_k = (1/n)Σ(xᵢ−x̄)^k，S = m₃/m₂^{3/2}，K = m₄/m₂²，JB = (n/6)(S² + (K−3)²/4) 渐近服从 χ²₂，其右尾恰为 exp(−JB/2)
- **步骤**：① 校验 n ≥ 4；② 去均值算 m₂、m₃、m₄，m₂ = 0 则报错；③ 算偏度与（未减 3 的）峰度；④ 组出 JB，p = exp(−JB/2)；⑤ 返回 `stat`/`p_value`/`df`/`skewness`/`kurtosis`/`n`
- **复杂度**：O(n) 时间，O(n) 空间
- **参数**：`x`（一维样本，n < 4 抛 `ValueError`，常数序列抛 `ValueError`）；无其它参数——**偏度/峰度固定用有偏（矩）估计**（除以 n），没有切换到无偏口径的开关
- **陷阱**：只是**大样本渐近**检验，n < 30 时过度拒绝、实际水平远高于名义水平；矩估计口径与部分软件不同，小样本要注明；JB 是偏度 + 峰度的**综合**检验，拒绝时必须分别看 `skewness` 与 `kurtosis` 才知道"哪里不正态"；对单个离群点极度敏感（n = 100 里放一个 10σ 点就能让它爆掉）
- **怎么检验**：与 `scipy.stats.jarque_bera` 对拍（注意 scipy 的峰度按超额口径但统计量公式相同）；用 `chi2_sf(JB, 2)` 与 `exp(−JB/2)` 互相校验（同一份实现的解析恒等式）；造标准正态大样本断言不拒绝、造对数正态/指数样本断言拒绝；手算小样本的 m₂/m₃/m₄ 复现 `stat`

#### `anderson_darling(x)`

- **数学形式**：z₍ᵢ₎ = (x₍ᵢ₎ − x̄)/s（s 用 ddof = 1），A² = −n − (1/n)·Σᵢ (2i−1)[ln F(z₍ᵢ₎) + ln(1 − F(z₍ₙ₊₁₋ᵢ₎))]；小样本修正 A²* = A²(1 + 0.75/n + 2.25/n²)；p 值用 D'Agostino & Stephens 的四段指数近似
- **步骤**：① 校验 n ≥ 8；② 用样本均值与 ddof=1 标准差标准化并升序排序；③ 逐点算 F 与上尾 1−F，分别截断到 [1e-300, 1]；④ 按加权公式累加 A²；⑤ 做小样本修正后按分段公式取 p；⑥ 返回 `stat`/`p_value`/`crit`/`n`
- **复杂度**：O(n log n) 时间（排序主导，另有逐标量 `_norm_cdf`/`_norm_sf` 的 Python 循环），O(n) 空间
- **参数**：`x`（一维样本，n < 8 抛 `ValueError`，标准差为 0 抛 `ValueError`）；无其它参数；返回的 `crit` 是固定的 {"15%": 0.561, "10%": 0.631, "5%": 0.752, "2.5%": 0.873, "1%": 1.035} 渐近临界值字典
- **陷阱**：本实现是**参数由样本估计**的 "case 4" 版本，不能套用"均值方差已知"版本（有闭式 P(A² < x)）的临界值；`crit` 是渐近的，n < 30 时实际临界值略低（docstring 记录：5% 临界值在 n = 20 时约 0.72，表中为 0.752），用表值判断会偏保守；p 值分段近似在 A² ≈ 0.30 的分段边界处绝对误差可达约 0.03，其余区间一般不超过 0.01；标准化必须 ddof = 1，否则与 `scipy.stats.anderson` 对不上号
- **怎么检验**：与 `scipy.stats.anderson(x, dist='norm')` 对拍 `statistic`（scipy 不返回 p，故 p 只能另找途径）；用标准正态大样本断言 `p_value` 明显偏大、用指数样本断言拒绝；把分段公式在 A² 的四个区间各取一点，用 `scipy.stats.monte_carlo_test` 自举得到的 p 做交叉核对；断言标准化口径换成 ddof = 0 时 `stat` 会改变，以确认口径确实生效

#### `ks_test_normal(x, mu=None, sigma=None)`

- **数学形式**：D = max_i max(i/n − F(z₍ᵢ₎), F(z₍ᵢ₎) − (i−1)/n)，z₍ᵢ₎ = (x₍ᵢ₎ − μ)/σ；p 用 Stephens 修正的 Kolmogorov 分布：λ = (√n + 0.12 + 0.11/√n)·D，Q(λ) = 2Σ_{k≥1}(−1)^{k−1}exp(−2k²λ²)
- **步骤**：① 校验 n ≥ 3，并检查 mu/sigma 是否同时给或同时为 None；② 缺失参数时用样本均值与 ddof=1 标准差估计，并置 `parameters_estimated=True`；③ 排序后算 F，取 D⁺、D⁻ 的最大值；④ 级数求和到项绝对值 < 1e-14（上限 100 项）得 p；⑤ 返回 `stat`/`p_value`/`n`/`mu`/`sigma`/`parameters_estimated`
- **复杂度**：O(n log n) 时间（排序 + 逐标量正态 CDF 循环），O(n) 空间
- **参数**：`x`（一维样本，n < 3 抛 `ValueError`，sigma ≤ 0 抛 `ValueError`）；`mu`（默认 None = 用样本均值估计）与 `sigma`（默认 None = 用 ddof=1 样本标准差估计）——**必须同时给或同时为 None**，只给一个抛 `ValueError`；给定它们就变成参数完全已知的版本
- **陷阱**：**用样本估计 μ/σ 后 p 值偏大（Lilliefors 问题）**——标准 KS 的零分布假设参数已知，估计参数后 D 的分布整体变小而临界值没变，于是 p 虚高、真实第一类错误率远低于名义水平；`parameters_estimated=True` 时返回的 p 只能当"名义值"，不能写进论文当结论；KS 对分布**中部**敏感、对尾部不敏感（与 AD 恰好互补）；sigma 为 0 时报错
- **怎么检验**：参数完全已知时与 `scipy.stats.kstest(x, 'norm', args=(mu, sigma))` 对拍；参数估计时与 `statsmodels.stats.diagnostic.lilliefors`（或 scipy 的 Lilliefors 实现）对拍，观察本函数的 p 明显偏大——`_self_test` 就断言了 `ks_p_estimated > ks_p_known_params` 这个不等式；用蒙特卡洛自举估计真实 p 值作第三方口径；级数部分可用首项 2exp(−2λ²) 与完整和在 λ 较大时比较

#### `ols(X, y, add_intercept=True)`

- **数学形式**：β̂ = argmin‖y − Xβ‖²（用 SVD 最小二乘解）；σ̂² = RSS/(n−k)，se(β̂ⱼ) = √(σ̂²·[(X'X)⁻¹]_jj)，tⱼ = β̂ⱼ/seⱼ ~ t(n−k)；R² = 1 − RSS/TSS，adj R² = 1 − (1−R²)(n−1)/(n−k)，F = ((TSS−RSS)/(k−1))/(RSS/(n−k))
- **步骤**：① 组装设计矩阵（可选在最前面插入 1 列）；② 校验 n > k，用 `np.linalg.lstsq` 求系数并检查秩不亏；③ 算残差、RSS、σ̂² 与 `(X'X)⁻¹` 的对角线得标准误；④ 逐系数算 t 与双侧 p；⑤ 按是否含截距算中心化或未中心化的 R²、adj R² 与 F 检验；⑥ 返回 `coef`/`se`/`t`/`p_value`/`r2`/`adj_r2`/`f_stat`/`f_p_value`/`resid`/`sigma2`/`df_resid`
- **复杂度**：O(n p²) 时间（SVD 求系数 + 一次显式求逆），O(np) 空间
- **参数**：`X`（(n, p) 设计矩阵，一维输入会被当成单列）、`y`（长度 n，不一致抛 `ValueError`）；`add_intercept`（默认 True）——设 False 时截距不再自动加入，R² 变成**未中心化**口径（1 − RSS/Σy²）、F 检验自由度从 k−1 变成 k，与统计软件默认结果不可比
- **陷阱**：多重共线性/病态矩阵是最大的坑——秩亏时本实现抛 `ValueError`，但"接近秩亏"（相关系数 0.999）检测不出来，只能靠 `vif` 与条件数人工判断；p 值与 F 都假设残差同方差、独立、近似正态，时间序列几乎必然自相关（应改稳健标准误或 Newey-West，本模块未实现）；用同一数据反复挑变量再报 p 值会严重高估显著性
- **怎么检验**：与 `statsmodels.api.OLS`（或 `numpy.linalg.lstsq` 手算 + `scipy.stats.t`）对拍 `coef`、`se`、`t`、`p_value`、`r2`、`adj_r2`、`f_stat`；构造完全线性数据断言 R² = 1（`_self_test` 用 y = 3 − 1.5x）；构造已知 β 的模拟数据检查系数恢复与 `sigma2` 接近真实噪声方差；用 QR/正规方程两种独立解法交叉验证系数；对无截距模型单独核对未中心化 R² 公式

#### `vif(X)`

- **数学形式**：对第 j 列做辅助 OLS（其余各列 + 截距），得 R_j²，VIF_j = 1/(1 − R_j²)
- **步骤**：① 校验 p ≥ 2 且 n > p；② 对每一列删掉自身，调用 `ols` 回归该列；③ 取辅助 R²，代入 1/(1−R²)；④ R² ≥ 1 − 1e-12 时置 `inf`；⑤ 返回 `vif` 与 `r2` 两个数组
- **复杂度**：O(p · n p²) 时间（p 次辅助 OLS），O(np) 空间
- **参数**：`X`（(n, p) 设计矩阵，**不含截距列**——辅助回归内部自动加截距；p < 2 或 n ≤ p 抛 `ValueError`）；无其它可调参数，就是 p 次全量回归，没有逐步/阈值开关
- **陷阱**：只诊断**线性**共线，对 X₂ = X₁² 这类非线性依赖常常看不出来，但同样会让系数不稳；经验阈值（VIF > 10）只是惯例不是定理，变量少时 VIF = 5 也可能让标准误翻倍，论文里要写明阈值与理由；设计矩阵里有 k 个类别放 k 个哑变量 + 截距时 VIF 直接爆表，那是建模错误；完全共线返回 `inf`，不要直接抄进表格（应写"完全共线"）
- **怎么检验**：正交设计的 VIF 应全部接近 1；手工造 X₂ = X₁ + ε 并让 ε → 0，断言 VIF 单调增大——`_self_test` 就断言了共线设计的最大 VIF 大于正交设计；用 `1/(1−R²)` 与 `np.linalg.inv(X'X)` 对角线的解析关系独立复算；与 `statsmodels.stats.outliers_influence.variance_inflation_factor` 对拍

#### `ridge_regression(X, y, alpha, standardize=True)`

- **数学形式**：标准化路径下 β = (Z'Z + αI)⁻¹Z'y_c（y_c 已中心化，截距不参与惩罚），coef_j = β_j/s_j，intercept = ȳ − Σ coef_j·x̄_j；`standardize=False` 时直接解 (X'X + αI)⁻¹X'y
- **步骤**：① 校验行列匹配与 alpha ≥ 0 且有限；② 标准化时算列均值与 ddof=0 标准差（0 标准差置 1），并中心化 y；③ 解 (Z'Z + αI)β = Z'y_c 得系数；④ 按 s_j 换算回原始尺度并事后算截距；⑤ 返回 `coef`/`intercept`/`alpha`/`standardized`
- **复杂度**：O(n p² + p³) 时间，O(np) 空间
- **参数**：`X`/`y`（X **不含截距列**，长度不一致抛 `ValueError`）；**`alpha`（无默认值，必须显式给出，≥ 0 且有限）**——α = 0 退化为 OLS，α 增大系数范数单调收缩；`standardize`（默认 True）——设 False 就变成按变量**量纲**施加惩罚，把"万元"改成"元"会让同样 α 下惩罚几乎失效
- **陷阱**：不标准化做岭回归等于按量纲惩罚，除量纲一致外都应保持 True；**α 不能"试到结果好看"**，要用交叉验证或岭迹图选并写进论文；**岭回归没有 p 值**（有偏估计，常规 t 检验不适用），本函数刻意不返回标准误与 p；常数自变量列会破坏标准化（被置尺度 1 并保留）；比较不同 α 的系数必须是同一套标准化规则
- **怎么检验**：**α → 0 时系数必须收敛到 `ols` 的斜率**——`_self_test` 断言 α = 1e-10 与 OLS 的差小于 1e-6；断言 α = 100 时系数的 2-范数小于 α = 1e-10 时的范数；用岭迹（一串 α 下的系数路径）检查单调收缩；与 `sklearn.linear_model.Ridge(alpha, fit_intercept=True)` 在标准化后的数据上对拍；用 `np.linalg.solve(Z'Z + αI, Z'y)` 的正规方程与 SVD/增广矩阵两种独立解法互检

#### `logistic_regression(X, y, lr=0.1, max_iter=1000, tol=1e-8)`

- **数学形式**：η = Xdβ，p = 1/(1 + e^{−η})，ℓ(β) = Σ[yᵢln pᵢ + (1−yᵢ)ln(1−pᵢ)]；牛顿步 step = H⁻¹g，梯度 g = Xd'(y − p)，Hessian H = Xd'WXd，W = diag(p(1−p))
- **步骤**：① 校验 y 只含 0/1 且非单一类别，参数范围检查；② 首列加 1，β 从全 0 起步；③ 每轮算 p、W、g、H（H 加 1e-10 岭），解牛顿步；④ 若全步长使对数似然下降，就按 `lr` 收缩重试（最多 60 次），否则退出；⑤ 系数最大变化 < `tol` 判收敛，系数绝对值超 1e6 或步长非有限则判完全分离并返回 `converged=False`；⑥ 返回 `coef`（含首位截距）/`log_lik`/`iterations`/`converged`/`prob`
- **复杂度**：O(max_iter · n p²) 时间，O(np) 空间
- **参数**：`X`（(n, p)，**不含截距列**，函数自动加 1）、`y`（必须 0/1，否则抛 `ValueError`）；**`lr`（默认 0.1）不是学习率而是回溯线搜索的步长收缩因子**，必须落在 (0, 1)，改小 = 每次收缩更狠、迭代更多，改大 = 更接近纯牛顿步、可能震荡；`max_iter`（默认 1000）——上限，撞到上限时 `converged=False`；`tol`（默认 1e-8）——系数最大变化量的收敛阈值，调松会提前停
- **陷阱**：完全分离时 MLE 不存在，系数发散、对数似然趋近 0，本实现在系数绝对值超 1e6（docstring 写的是"系数范数超过 1e6"，代码实际判的是最大绝对系数）或对数似然不再改善时停下并返回 `converged=False`，**不返回 NaN 也不假装收敛**；`converged=False` 的系数不可解释，别写进论文（应加 L2 惩罚、减变量或改 Firth 惩罚似然）；本函数**不返回标准误与 p 值**；类别极不平衡时准确率虚高，应报 AUC/召回；p 被截断到 [1e-12, 1−1e-12] 再取对数，极端情况的对数似然会略微"好看"
- **怎么检验**：与 `statsmodels.Logit`（或 `sklearn.linear_model.LogisticRegression(penalty=None)`）对拍系数与对数似然；造**由 Bernoulli 抽样生成**的可分不平衡数据断言系数符号正确、`converged=True`（`_self_test` 特意注明：若把标签写成 1[sigmoid(线性项) > 0.5]，数据是线性可分的，MLE 不存在，`converged` 必然是 False）；造完全分离数据断言 `converged=False` 且 `prob` 全为有限值；用有限差分检查首轮梯度 ≈ Xd'(y − p₀)；`log_lik` 与手算 Σ[y ln p + (1−y)ln(1−p)] 对照

#### `bootstrap_ci(x, statistic=None, n_boot=2000, alpha=0.05, seed=None)`

- **数学形式**：从样本中有放回抽 n 个得 θ̂*_b（b = 1..n_boot），区间取重抽样分布的 [α/2, 1−α/2] 分位数（线性插值）：CI = [Q_{α/2}(θ̂*), Q_{1−α/2}(θ̂*)]，`boot_se` 为 θ̂* 的标准差（ddof = 1）
- **步骤**：① 校验 n ≥ 2、n_boot ≥ 1、α ∈ (0,1)、statistic 可调用；② 在原样本上算 `estimate`；③ 用 `rng(seed)` 循环 n_boot 次有放回抽 n 个下标、算统计量；④ 出现非有限统计量则报错；⑤ 取分位数与标准差返回 `estimate`/`ci_low`/`ci_high`/`alpha`/`n_boot`/`boot_se`
- **复杂度**：O(n_boot · n · cost(statistic)) 时间，O(n) 额外空间（实现是逐次重抽样，没有一次性构造 (n_boot, n) 矩阵）
- **参数**：`x`（一维样本）；`statistic`（默认 None = 样本均值）——传自定义可调用对象时**必须返回有限标量**，否则报错；`n_boot`（默认 2000）——决定区间**分辨率**，2000 时 95% 区间两端由第 50/1950 个次序统计量决定，换种子结果在第 2~3 位有效数字上变动，要报稳定数字就提到 1e4；`alpha`（默认 0.05 = 95% 置信度）——**调小则区间变宽**；`seed`（默认 None → `DEFAULT_SEED`）——固定后才可复现
- **陷阱**：百分位法不万能——统计量分布偏斜或有偏（样本最大值、方差、小样本相关系数）时覆盖率明显低于名义水平，且区间可能越界（比例类统计量给负下限），更稳的 BCa/学生化 Bootstrap 本模块未实现；Bootstrap 假设**独立同分布**，时间序列直接对点重抽样会破坏自相关结构（应用块状 Bootstrap 或对残差重抽样）；n < 10 时重抽样粒度太粗，区间基本没意义
- **怎么检验**：用对称分布（如正态样本均值）断言区间包含 `estimate`，且 α = 0.01 的区间宽度 ≥ α = 0.05 的宽度（`_self_test` 断言的就是这条）；换不同 `seed` 重跑，检查端点波动幅度符合 1/√n_boot 的量级；用已知分布的解析置信区间（如正态均值的 t 区间）比较覆盖率——对均值这类近似对称统计量两者应接近；写独立的"逐次重抽样"循环复现同样的分位数（同 seed 下应逐位一致）

#### `permutation_test(x, y, n_perm=2000, seed=None)`

- **数学形式**：观测统计量 s₀ = x̄ − ȳ；把两组合并后随机重分配标签（保持 n_x、n_y 不变）得 s*_b；双侧 p = (1 + #{|s*_b| ≥ |s₀|}) / (1 + n_perm)
- **步骤**：① 校验两组各 ≥ 2 个观测、n_perm ≥ 1；② 算观测均值差；③ 合并成池，循环 n_perm 次做 `gen.permutation` 切分并算均值差；④ 用绝对值比较累计计数；⑤ 返回 `stat`/`p_value`/`n_perm`
- **复杂度**：O(n_perm · (n_x+n_y)) 时间，O(n_x+n_y) 空间
- **参数**：`x`/`y`（长度可不同的两组独立样本）；`n_perm`（默认 2000）——**直接决定 p 值分辨率**，2000 时最小可能 p ≈ 1/2001，再小只能报 "p < 0.001"，要更小必须加大；`seed`（默认 None → `DEFAULT_SEED`）——固定后结果可复现，换种子 p 会变动（n_perm 越小变动越大），论文应报告 seed。统计量固定为均值差，**没有换成中位数/方差等其它统计量的参数**
- **陷阱**：原假设是**两组可交换（同分布）**，比"均值相等"更强——两组方差不齐时，均值差的置换检验实际在检验整个分布是否相同，解释要小心；样本量很小时可能的置换组合有限、p 值是离散的，不要报很多位小数；不能用于配对/分组设计
- **怎么检验**：**小样本直接枚举全部 C(n_x+n_y, n_x) 个分组**，得到精确 p 与本实现近似 p 对比；两组同分布时 p 应偏大（`_self_test` 用了两组独立正态样本），其中一组整体平移 2 后 p 必须很小；与 `scipy.stats.permutation_test`（或 `mannwhitneyu` 在近似正态下的结论）对拍；换 seed 多次重跑，检查 p 的均值接近枚举值

#### `pca(X, n_components=None, standardize=True)`

- **数学形式**：中心化（可选再除以 ddof = 0 列标准差）得 Z，协方差阵 C = Z'Z/(n−1)，特征分解 C = VΛV'；components 取 Λ 降序前 k 个特征向量，eigenvalues = λ，解释率 = λᵢ/Σλ，scores = ZV_k'
- **步骤**：① 校验 n ≥ 2 与 1 ≤ k ≤ p；② 中心化（必要时标准化，0 标准差列置 1）；③ 用 `np.linalg.eigh` 分解对称协方差阵；④ 特征值降序、负值截 0；⑤ 用 `_fix_component_signs` 固定符号（每行绝对值最大的分量为正）；⑥ 返回 `components`/`eigenvalues`/`explained_variance_ratio`/`scores`/`cumulative_ratio`
- **复杂度**：O(n p² + p³) 时间，O(np + p²) 空间
- **参数**：`X`（(n_samples, n_features)）；`n_components`（默认 None = 全部 p 个）——**只在相关阵/协方差阵上工作**，取值必须落在 1..p，越界抛 `ValueError`；`standardize`（默认 True）——True 在**相关阵**上做主成分（量纲可比），False 只中心化、在**协方差阵**上做，两种结果不可比（不做标准化时方差最大的指标会独占第一主成分）
- **陷阱**：特征向量的符号本身无定义（v 与 −v 同样合法），但**得分矩阵会跟着反号**，不固定符号则换 numpy 版本就可能得到反号结果；常数列在标准化下变成全 0 的无信息方向，白占一个成分位，建模前应删；p > n 时协方差阵秩亏，最多 n−1 个非零特征值，报告解释率时不要按 p 个成分解释；`explained_variance_ratio` 只在 k = p 时和为 1
- **怎么检验**：**与 `np.linalg.svd(Z)` 对拍**——奇异值平方除以 (n−1) 就是特征值（`_self_test` 把这条做成断言，容差 1e-8）；断言 `components @ components.T ≈ I`（正交性）与解释率之和为 1；构造两列完全线性相关的数据，断言第二个特征值 ≈ 0；与 `sklearn.decomposition.PCA`（注意 sklearn 的 `svd_flip` 符号约定与本实现相同）对拍特征值与解释率；强相关数据的第一主成分解释率应显著偏高

#### `factor_analysis(X, n_factors=2, max_iter=200, tol=1e-8, rotate=True)`

- **数学形式**：标准化得相关阵 R；迭代：把 R 的对角元换成共性方差 h²（约化相关阵 R_s），取前 k 个特征对，载荷 L = V_k√(max(λ_k,0))，更新 h² ← diag(LL')，直到 max|Δh²| < tol；旋转后 variance_explained_j = Σᵢ l_ij²/p
- **步骤**：① 校验 n ≥ 3、p ≥ 2、1 ≤ k ≤ p−1 与参数范围；② 标准化并算相关阵 R；③ h² 初值取该变量与其余变量绝对相关系数的行最大值（截断到 [0,1]）；④ 反复替换对角元求特征对并更新 h²（Heywood 情形截断到 [0,1]）；⑤ 用最终 h² 重算载荷、可选 `_varimax` 旋转、按列平方和降序重排并固定符号；⑥ 返回 `loadings`/`communalities`/`uniqueness`/`variance_explained`/`n_iter`
- **复杂度**：O(max_iter · p³) 时间（每轮一次对称特征分解；p 大时是瓶颈），O(np + p²) 空间
- **参数**：`X`（内部自动做 z-score 标准化）；`n_factors`（默认 2）——必须落在 1..p−1，**影响均衡性**：因子数过多会出现 Heywood 情形；`max_iter`（默认 200）——迭代重估共性方差的最大轮数；`tol`（默认 1e-8）——h² 最大变化量的收敛阈值；`rotate`（默认 True）——是否做 varimax 正交旋转，旋转**不改变共性方差与解释总量**，只让载荷更简单
- **陷阱**：主因子法**不是极大似然**，收敛到的是约化相关阵的特征解，与 statsmodels 的 ML 解会有差异，论文要写清用的是哪一种；Heywood 情形（h² 顶到 1）说明因子数过多或模型不适定，截断只是让迭代跑完，不解决问题；旋转后本实现显式重排，载荷列序与未旋转时的特征向量序号不同；**因子个数的选择本函数不做**（碎石图/平行分析要自己做并说明理由）
- **怎么检验**：造**已知单因子模型**的数据（X = f·λ' + 噪声），断言估计的共性方差 h² ≈ λ²、载荷重构相关阵的非对角元接近真实值——`_self_test` 用的就是这套；断言 `uniqueness == 1 − communalities`；比较 `rotate=True` 与 `False` 的 `communalities` 应完全相同（正交旋转的不变量）；`_varimax` 的准则是旋转前后 `ΣΣl⁴ − (1/p)Σ(Σl²)²` 单调不减；与 `statsmodels.multivariate.factor.Factor` 对拍载荷（注明估计方法差异）

#### `lasso_regression(X, y, alpha=0.1, max_iter=1000, tol=1e-8)`

- **数学形式**：β̂ = argmin (1/(2n))‖y_c − X_cβ‖² + α‖β‖₁（截距不参与惩罚）；坐标下降单变量更新 β_j ← soft(x_j'r_j, nα)/(x_j'x_j)，soft(z,γ) = sign(z)max(|z|−γ, 0)
- **步骤**：① 校验行列匹配、α ≥ 0 有限、max_iter/tol 合法；② 中心化 X 与 y，算各列平方和；③ 坐标下降：维护残差 r，对每列把自身贡献加回得 r_j，算 ρ = x_j'r_j，用软阈值更新 β_j 并同步刷新 r；④ 一轮内最大系数变化 < tol 时停；⑤ 事后还原截距 mean(y) − mean(X)β；⑥ 返回 `coef`/`intercept`/`n_nonzero`/`objective`/`n_iter`
- **复杂度**：O(max_iter · n p) 时间，O(np) 空间
- **参数**：`X`（(n, p)，**不含截距列**）、`y`（长度 n）；`alpha`（默认 0.1）——L1 惩罚强度，**α = 0 退化为 OLS，α 增大非零系数个数单调减少**，α 极大时全部压成 0（此时 `coef` 全零、`intercept` 就是 mean(y)）；`max_iter`（默认 1000）——扫描轮数上限；`tol`（默认 1e-8）——**系数变化量**（不是目标函数变化量）的收敛阈值；本实现**不做列标准化**，没有对应开关
- **陷阱**：只中心化不标准化，同一个 α 对不同量纲的列惩罚力度完全不同（单位从"米"换成"毫米"，系数缩小 1000 倍，L1 阈值形同不存在），要跨变量比较稀疏性必须自己先标准化 X；常数列（x_j'x_j = 0）系数被固定为 0（不是除零得 NaN）但仍出现在 `coef` 里；**坐标下降没有 p 值**（L1 解有偏且被选变量数依赖 α）；`tol` 判的是系数而非目标，目标在最优解附近很平、系数收敛更慢
- **怎么检验**：**KKT 条件** max_j|x_j'resid|/n ≤ α——`_self_test` 就断言这条（含 1e-6 容差），它独立于坐标下降本身；α → 0 时系数应逼近 `ols` 的斜率（`_self_test` 容差 1e-3）；α = 1e3 时断言 `n_nonzero == 0`；断言 `objective` 不高于全零模型的 (1/(2n))Σ(y−ȳ)²；画 α 从大到小的正则化路径，检查非零系数个数单调不减；与 `sklearn.linear_model.Lasso(alpha, fit_intercept=True)` 在标准化数据上对拍系数

#### `poisson_regression(X, y, max_iter=200, tol=1e-8)`

- **数学形式**：η = Xdβ（Xd 首列为 1），μ = exp(η)；IRLS 的工作响应 z = η + (y−μ)/μ、权重 w = μ，解 min‖√w(z − Xdβ)‖²；偏差 D = 2Σ[y ln(y/μ) − (y−μ)]（y = 0 的项取 2μ），loglik = Σ[y ln μ − μ − ln Γ(y+1)]，AIC = −2loglik + 2(p+1)
- **步骤**：① 校验 y 非负、行列匹配、参数合法；② 首列加 1，β₀ = ln(max(ȳ, 1e-6)) 起步；③ 每轮算 μ、η、w、z，用 `np.linalg.lstsq` 解加权最小二乘得候选 β；④ 若新偏差大于旧偏差则步长折半（最多 30 次），保证偏差单调下降；⑤ 系数最大变化 < tol 时停；⑥ 返回 `coef`（斜率，**不含截距**）/`intercept`/`deviance`/`loglik`/`aic`/`n_iter`
- **复杂度**：O(max_iter · n p²) 时间，O(np) 空间
- **参数**：`X`（(n, p)，**不含截距列**，函数自动加 1）、`y`（非负计数，含负值抛 `ValueError`）；`max_iter`（默认 200）——IRLS 轮数上限；`tol`（默认 1e-8）——系数最大变化量的收敛阈值；η 内部截断到 [−50, 50]、μ 下限 1e-10 都是**硬编码的数值保护**，没有暴露成参数
- **陷阱**：**过散布**——真实计数数据方差常大于均值，Poisson 假设下标准误被严重低估、p 值偏小；本函数不返回标准误，但要用似然比/Wald 检验前应先看偏差/自由度是否远大于 1，必要时换负二项回归；y 含 0 时偏差项显式取 2μ（极限值），不能直接对 y 取对数；计数需要**曝光量**时应把 log(offset) 作为系数固定为 1 的偏移项，当普通自变量会得到完全错误的系数；完全共线列会让 IRLS 给出最小范数解，系数不可解释
- **怎么检验**：**偏差与对数似然的恒等式** deviance = 2(饱和对数似然 − 模型对数似然)——`_self_test` 用这条做断言（容差 1e-6），它完全独立于 IRLS 迭代；断言 AIC = −2loglik + 2(p+1)；在已知真参数生成的 Poisson 计数数据上检查系数恢复（`_self_test` 断言误差 < 0.1）；与 `statsmodels.GLM(family=Poisson())` 对拍系数、偏差、对数似然；用有限差分检查得分方程 Xd'(y − μ) ≈ 0 在解处成立

#### `durbin_watson(residual)`

- **数学形式**：DW = Σ_{t=2..n}(e_t − e_{t−1})² / Σ_{t=1..n}e_t²，等价地大样本下 DW ≈ 2(1 − ρ̂₁)
- **步骤**：① 校验 n ≥ 2 且残差平方和 > 0（全 0 时抛 `ValueError`）；② 算相邻差分平方和；③ 返回两者之比（**裸 float，不是 dict**）
- **复杂度**：O(n) 时间，O(n) 空间
- **参数**：`residual`（**必须按时间/原始顺序排列**的残差序列，n < 2 或全 0 抛 `ValueError`）；**没有任何可调参数**——不提供临界值表、不返回 p 值、也不能指定滞后阶数
- **陷阱**：残差排序后传入会算出接近 2 的"正常"值，这是最容易骗过自己的用法；必须含截距（无截距回归的残差均值不为 0，DW 会系统性偏离 2，临界值表也不适用）；只检测**一阶**自相关，对高阶/季节性（如季度数据的滞后 4）无能为力，应改 Breusch-Godfrey；"是否接近 2"本身没有统计显著性含义，只有落在临界值之间才能说不能拒绝；临界值表依赖 n 与自变量个数，本函数不提供
- **怎么检验**：用 iid 残差断言 DW 落在 2 附近（`_self_test` 断言 1.5 < DW < 2.5），用 ρ = 0.9 的 AR(1) 残差断言 DW < 1；断言残差整体变号不改变 DW（符号不变性）；用定义式手算一个小样本（n = 4）逐项核对；与 `statsmodels.stats.stattools.durbin_watson` 对拍

#### `breusch_pagan(residual, X)`

- **数学形式**：e² 对 [1, X] 做辅助回归得 R²_aux，LM = n·R²_aux，在 H₀（同方差）下渐近服从 χ²_p，p 用卡方上尾概率
- **步骤**：① 校验残差长度与 X 行数一致；② 取 e²；③ 组装 [1, X] 用 `np.linalg.lstsq` 求辅助回归，秩亏则抛 `ValueError`；④ 用辅助 TSS/RSS 算 R²（TSS 为 0 时报错）；⑤ LM = n·R²，p = `_chi2_sf`(LM, p)；⑥ 返回 `stat`/`df`/`p_value`
- **复杂度**：O(n p²) 时间，O(np) 空间
- **参数**：`residual`（OLS 残差，长度 n）、`X`（原回归设计矩阵 (n, p)，**不含截距列**——辅助回归自动加 1；长度不一致抛 `ValueError`）；**无其它可调参数**，df 固定等于 X 的列数 p
- **陷阱**：这是 **LM（拉格朗日乘数）版本，不是 Koenker 的学生化版本**，对残差正态性敏感、厚尾数据大量假阳性（稳健做法是 Koenker 1981 的学生化估计或 White 检验）；检验的是"残差平方与 X **线性**相关"，对 var = exp(x'b) 很灵、对某些非线性方差结构会漏检；拒绝只说明存在异方差，**不告诉你怎么办**（修正手段是稳健标准误 White/HC3 或加权最小二乘）；辅助回归秩亏时抛 `ValueError`，要先删冗余列
- **怎么检验**：同方差残差 p 值应 > 0.05，方差按 exp(0.5·x) 增长的残差 p 必须 < 0.01 且统计量更大——`_self_test` 断言的就是这两条；与 `statsmodels.stats.diagnostic.het_breuschpagan`（注意它默认给的是 Koenker 学生化版，统计量与 LM 版不同，要比的是 `lm` 与 `lm_pvalue` 的量级与结论）对拍；手写"e² 对 [1,X] 做 OLS 取 R² 再乘 n"的小脚本独立复现 `stat`

#### `stepwise_selection(X, y, criterion='aic', max_steps=50)`

- **数学形式**：以 AIC = n·ln(RSS/n) + 2k 或 BIC = n·ln(RSS/n) + k·ln(n)（k 含截距 = 选中变量数 + 1）为准则，每轮在"加入一个未选变量"与"剔除一个已选变量"的全部候选中取准则值最低、且**严格优于**当前模型者执行
- **步骤**：① 校验行列匹配与 criterion ∈ {aic, bic}、max_steps ≥ 0；② 从空模型（只有截距）出发；③ 内部 `_rss` 对每个候选变量集做 OLS（秩亏或 n ≤ k 时视为不可用、准则记 +inf）；④ 每轮同时评估全部增/删候选，取最优动作；⑤ 无动作可改进或达到 max_steps 时停；⑥ 用最终变量集重跑 OLS 得系数，返回 `selected`/`coef`/`intercept`/`criterion_value`/`history`
- **复杂度**：O(max_steps · p · n p²) 时间（每步评估约 2p 个候选，每个候选一次 OLS），O(np) 空间
- **参数**：`X`（(n, p) 候选自变量，**不含截距列**）、`y`（长度 n）；`criterion`（默认 `"aic"`）——取值只能是 `"aic"`/`"bic"`，**BIC 惩罚更重、倾向更小的模型**，n 很大时两者选出的变量集明显不同，论文必须写明用哪一个；`max_steps`（默认 50，必须 ≥ 0）——最多执行多少次增删动作，调小会得到未收敛的次优模型
- **陷阱**：**逐步回归后的 p 值不可信**——变量是被数据挑出来的，常规 t 检验的名义显著性严重失真（选择性推断问题），不能用它下"某变量显著"的结论；AIC/BIC 只比较同一响应 y、**样本必须完全一致**的模型，有缺失值要先插补；完全共线的候选列被直接判为不可用（跳过）而不会让筛选崩掉；本函数只做**线性**模型的变量筛选，被排除的变量可能以交互项/非线性形式起作用
- **怎么检验**：造**已知稀疏真值**的数据（如只有第 2、4 列真实系数非零），断言 BIC 恰好选中这两列、系数接近真值——`_self_test` 就断言 `selected == [1, 3]`（0 基）且系数误差 < 0.3；断言 AIC 结果的准则值不高于全模型 AIC；把全部子集穷举（p 小的时候）得到的全局最优 AIC/BIC 与逐步结果对比，检查是否被局部最优卡住（逐步法本来就可能卡住，这正是要报告的局限）；断言 `history` 每步的准则值严格下降

---


### 3.6 评价与决策 —— `examples/algorithms/evaluation.py`

这一族解决**多指标综合评价与排序**：把 m 个方案 × n 个指标（含效益型与成本型）的决策矩阵，
先统一方向、再无量纲化、再赋权，最后给出综合得分与名次；也可以做效率评价（DEA）与等级评价
（模糊综合评判）。

共同约定（顺序错了结论会翻盘）：

- **先正向化再无量纲化**：成本型指标（越小越好）先做极差反转 x′ = max(x) − x，再归一化。
  本模块所有函数的 `benefit` 参数都是长度 n 的 bool 序列（True 为效益型，None 表示全部按正向处理）。
- **权重一律返回和为 1 的数组**；输入权重和不为 1 时会先归一化并在 `note` 里说明（不静默处理）。
- **排名口径**：`rank` 用"1 为最好"，并列给同名次（标准竞赛排名法）。

---

#### `ahp_weights(pairwise, max_iter=1000, tol=1e-12)`

- **数学形式**：正互反判断矩阵 A（a_ij = 1/a_ji、对角为 1）的主特征向量即权重：A w = λ_max w；CI = (λ_max − n)/(n − 1)，CR = CI/RI。
- **步骤**：① 校验方阵、元素为正、对角为 1、正互反；② 幂法迭代 w ← A w/‖A w‖ 直到最大分量变化 < `tol` 或达 `max_iter`；③ 取绝对值后归一化；④ λ_max 用 mean((A w)_i / w_i) 估计；⑤ 查内置 Saaty RI 表（n = 1..15）得 CR 与 `consistent`（CR < 0.1）。
- **复杂度**：时间 O(max_iter·n²)；空间 O(n²)。
- **参数**：`max_iter=1000`（幂法迭代上限，正常判断矩阵几十步内收敛，触顶说明矩阵病态或退化）；`tol=1e-12`（收敛阈值，放宽会更快但 λ_max 估计更粗）；`pairwise`（唯一的数据输入，n 行 n 列）。n > 15 时 RI 表外，返回 `CR=None` 并给 `note`，需自行查表。
- **陷阱**：**n ≤ 2 时 CR 恒为 0**（RI = 0），这不是"判断矩阵完美"而是无法检验，论文里要说明；判断矩阵必须正互反，填了 a_ij = 3 却忘写 a_ji = 1/3 会直接报错（不会被悄悄算错）；幂法取的是主特征向量、方向可能整体为负，本实现取绝对值后归一化；**CR 达标 ≠ 权重合理**，CR 只说明判断前后不矛盾，专家打分本身有偏就救不回来；幂法对 λ_max 的估计用算术均值，比 `numpy.linalg.eig` 的直接读特征值略粗（但省掉了复数与排序的麻烦）。
- **怎么检验**：① **闭式解**：完全一致的判断矩阵（a_ij = w_i/w_j 精确成立）的 CR 必须是 0、权重必须精确等于 w；② 用 Saaty 经典 3×3 矩阵 A = [[1,1/2,4],[2,1,7],[1/4,1/7,1]] 与教材给出的权重/λ_max/CR 对拍（本模块自测就用这个矩阵，可作为独立查表对照）；③ 与 `numpy.linalg.eig` 求主导特征向量独立对拍（两者应一致到 1e-10 量级）；④ 扰动检验：把矩阵轻微改动后 CR 应随之上升，验证一致性判据的方向性。

---

#### `entropy_weights(X, benefit=None)`

- **数学形式**：p_ij = x′_ij / Σ_i x′_ij，e_j = −(1/ln m)·Σ_i p_ij ln p_ij，差异系数 d_j = 1 − e_j，w_j = d_j / Σ_j d_j。
- **步骤**：① 校验数据非负（含负数直接报错）；② 成本型指标做极差反转 x′ = max − x；③ 按列求比重矩阵 P（列和为 0 直接报错）；④ 按 p ln p（p = 0 记 0）求信息熵并 clip 到 [0,1]；⑤ 由 d = 1 − e 归一化得权重。
- **复杂度**：时间 O(mn)；空间 O(mn)。
- **参数**：`benefit=None`（长度 n 的 bool 序列或 0/1 序列，None 表示全部按正向处理；标错成本型指标会把方向搞反）；`X`（决策矩阵，**必须非负**）。本函数除 `benefit` 外无可调参数——熵权完全由数据决定。
- **陷阱**：**要求非负**，含负数的指标（如利润变化率）必须先平移并在论文里说明平移量；熵权对**量纲与离散度极度敏感**，"先 min-max 再算熵权"与"直接算熵权"结果完全不同，必须写清口径；某指标所有方案取值相同时 d_j = 0、权重为 0，这通常说明指标选得不好，要在论文里讨论；熵权是"谁差异大谁重要"，**不等于业务上重要**，不能替代专家判断；`normalized` 返回的是**正向化后**的矩阵（未归一化），比重矩阵在 `p` 里。
- **怎么检验**：① 构造**已知答案**的数据：只有一列在变化、其余列全为常数时，该列权重必须为 1、其余为 0，且常数指标的熵必须恰为 1；② 所有方案在某列上取相同值 → 该列权重 0；③ 与公式的**逐步骤手算**对拍（m = 4 时 1/ln 4 可手算验证 e_j）；④ 单调性检验：把某一列的离散程度（如极差）放大，该列权重应当**不减**；⑤ 变换检验：对全部数据乘以同一个正数与平移同一个正数（保持非负）时，熵权应保持不变或按你声明的口径变化，用来暴露量纲敏感性。

---

#### `critic_weights(X, benefit=None)`

- **数学形式**：C_j = σ_j · Σ_i (1 − r_ij)，w_j = C_j / Σ_j C_j，其中 σ_j 为（正向化后）第 j 列的**样本**标准差、r_ij 为指标 i 与 j 的皮尔逊相关系数。
- **步骤**：① 成本型指标极差反转；② 计算 ddof = 1 的列标准差（对比强度）；③ 计算相关系数矩阵（NaN 置 0、对角置 1）；④ 冲突性 = Σ_i (1 − r_ij)，信息量 = σ_j × 冲突性；⑤ 归一化得权重。
- **复杂度**：时间 O(mn + n²m)；空间 O(mn + n²)。
- **参数**：`benefit=None`（指标方向）；`X`（决策矩阵，**至少 2 个方案**，否则报错）。标准差口径在代码里固定为 `ddof=1`，不可调。
- **陷阱**：`ddof=1` 与 `ddof=0`（总体标准差）在不同教材里都有，换口径权重会变，**论文必须写明**；与其他指标高度相关的指标会被判为"冲突小、信息量低"，若两个指标本质重复这是合理惩罚，若它们恰好同等重要则权重会被不合理压低、需人为修正；与熵权一样受量纲影响，先归一化再算（本函数本身**不做**归一化，只做正向化）；某列恒定 → σ = 0 → 权重 0；恒定的列做相关会得到 NaN，本实现把 NaN 置 0 后仍可能让该列冲突性变大，需人工检查。
- **怎么检验**：① 构造两列**完全相关**的数据（第二列 = 第一列的线性变换），两列的相关系数为 1，冲突性应等于各自与其他列的冲突之和、权重显著低于同标准差但独立的列；② 构造两列**完全独立**的数据，验证权重排序由标准差主导（σ 大的列权重更大）；③ 与手算对拍：2×2 的微型矩阵（2 个指标、n 个方案）可完全手算 σ 与 r；④ 与他人实现/Tabular 教材的算例对拍（注意先统一 ddof 与是否标准化）。

---

#### `combine_weights(weight_sets, method="multiplicative", alphas=None)`

- **数学形式**：multiplicative：w_j ∝ Π_k w_kj；geometric：w_j ∝ (Π_k w_kj)^(1/K)；linear：w_j = Σ_k α_k w_kj，再统一归一化。
- **步骤**：① 把每组权重转成一维非负向量并校验长度一致；② 按 `method` 合成；③ 和为 1 归一化（全 0 直接报错）；④ 返回 `weights`/`method`/`note`。
- **复杂度**：时间 O(Kn)；空间 O(Kn)。
- **参数**：`method="multiplicative"`（合成方式，`"multiplicative"` 乘法合成最狠、`"geometric"` 几何平均最温和、`"linear"` 线性加权可控性最强）；`alphas=None`（仅 `"linear"` 使用，各权重组的系数，None 表示等权；长度必须等于权重组数，和不为 1 会被归一化）；`weight_sets`（至少一组权重，各组长度必须一致且非负）。
- **陷阱**：**乘法合成会把小权重惩罚得非常狠**——两组权重都接近 0 的指标会被彻底压没，如果指标体系里有"小而重要"的指标，乘法合成会毁掉它（实现里对数用 `clip(W, 1e-300, None)` 兜底，避免 log 0）；组合权重**没有唯一的正确方法**，论文里必须写清组合方式与理由，并做一次灵敏度分析证明排序不因组合方式而翻盘；三种方法都不做"权重合理性"检查，输入本身错了它照算不误；docstring 标题提到的"博弈论组合赋权"**在代码里没有实现**，只是线性加权的别名式说法，别在论文里声称用了它。
- **怎么检验**：① **极限/恒等检验**：只有一组权重时三种方法都必须原样返回该组权重（归一化后）；② 用 `alphas` 把某一组权重系数设为 1、其余为 0，`linear` 必须精确复现那一组；③ 手算对拍：两组二维权重 (0.8, 0.2) 与 (0.6, 0.4) 的乘法合成结果可手算（0.48, 0.08 → 归一化 0.857, 0.143）；④ 灵敏度检验：换用三种 method 重算下游 TOPSIS/VIKOR 排序，比较名次翻转比例，作为论文的稳健性证据。

---

#### `topsis(X, weights, benefit=None)`

- **数学形式**：向量归一化 r_ij = x_ij / √(Σ_i x_ij²)（x 已正向化）→ 加权 v_ij = w_j r_ij → 正负理想解 v⁺ = max_i v_ij、v⁻ = min_i v_ij → d_i^± 为欧氏距离 → C_i = d_i⁻/(d_i⁺ + d_i⁻)。
- **步骤**：① 成本型指标极差反转（**必须在归一化之前**）；② 按列 L2 归一化；③ 乘权重得加权规范矩阵；④ 取列最大/最小作为正负理想解；⑤ 算两个欧氏距离与贴近度；⑥ 按贴近度降序给名次。
- **复杂度**：时间 O(mn)；空间 O(mn)。
- **参数**：`weights`（长度 n，非负，自动归一化到和为 1，输入和不为 1 会在 `note` 中说明）；`benefit=None`（指标方向）；`X`（决策矩阵）。函数本身无算法超参，调参的对象是权重与正向化口径。
- **陷阱**：**向量归一化必须做**，直接用原始数据算距离会让量纲大的指标独占权重，且**正向化必须在归一化前**，顺序颠倒排序会整体改变；正负理想解是**从你给的方案集里选出来的**，不是绝对标准——加入一个很差的新方案会让原有方案的贴近度集体上升，所以 C_i **不能跨数据集比较**；C_i 接近时排序不稳，务必配合 `topsis_rank_sensitivity` 做扰动分析（自测数据上就用它检查第一名保持率）；距离用欧氏范数隐含"各指标可替代"的假设，指标高度相关时应考虑马氏距离；某列全为 0 时 L2 范数为 0，该列会被保留为全 0（不产生 NaN），但应视为无效指标。
- **怎么检验**：① **已知答案构造**：只有一个指标时，TOPSIS 的名次必须与该指标排序完全一致（正负理想解只由该列决定，C_i 单调于原值）；② 量纲不变性：把所有指标乘以不同正数后再跑，结论（排序）应当**不变**（这正是先归一化的目的），若变了说明归一化或正向化顺序写错；③ 与成熟库对拍（sklearn 无 TOPSIS，可用 `pymcdm`/教科书算例），注意统一正向化与归一化口径；④ 极端情形：所有方案在所有指标上完全相同 → 所有 d⁺ = d⁻ = 0，本实现返回 C_i = 0.5，应断言该退化行为而不是出现 NaN；⑤ 一致性：C_i ∈ [0,1] 且 d_i⁺ + d_i⁻ > 0。

---

#### `vikor(X, weights, benefit=None, v=0.5)`

- **数学形式**：S_i = Σ_j w_j (f*_j − f_ij)/(f*_j − f⁻_j)（群体效用），R_i = max_j 同一表达式（个体遗憾），Q_i = v(S_i − S*)/(S⁻ − S*) + (1 − v)(R_i − R*)/(R⁻ − R*)。
- **步骤**：① 成本型指标由 `benefit` 决定 f*（正向取列最大、成本取列最小）；② 跳过所有方案取值相同的指标（跨度 < 1e-12）并记入 `note`；③ 算归一化比值、加权求和得 S、逐行取最大得 R；④ 用 min-max 归一化组合成 Q；⑤ 做两项检验：Q(A⁽¹⁾) − Q(A⁽²⁾) ≥ 1/(m−1)（可接受优势）且第一名在 S 或 R 中也排第一（可接受决策可靠性），结果放进 `conditions`。
- **复杂度**：时间 O(mn)；空间 O(mn)。
- **参数**：`v=0.5`（决策机制系数，取值 [0,1]：> 0.5 偏群体效用、< 0.5 偏个体遗憾；**它会改变排序，必须写明取值并做敏感性分析**）；`weights`（自动归一化，`note` 会说明）；`benefit=None`；`X`（至少 2 个方案，否则报错）。S/R/Q 的归一化方式在代码里固定，不可调。
- **陷阱**：只报 Q 值排序是**不完整**的，论文必须报告两项检验（本实现放在 `conditions` 里，但**不会**据此自动修正名次，需要你人工判断是否同时满足）；某个指标在所有方案上取值相同时分母为 0，本实现跳过该指标并记入 `note`（**注意：跳过等价于把该指标的权重丢掉但权重并未重新归一化**，会让 S、R 的整体尺度变小，多指标恒定时影响不可忽略）；`note` 会把"权重已归一化"和"已跳过指标"两条信息合并；若 S⁻ = S*（所有 S 相同）或 R⁻ = R*，本实现用 `safe_divide(fill=0.0)` 把该项置 0；返回的 `rank` 与 `Q` 都是"越小越好"，与 `topsis` 的"越大越好"方向相反，混用会得出相反结论。
- **怎么检验**：① **与 TOPSIS 对拍方向性**：在方案差异明显、权重合理的数据上，两者第一名通常一致；不一致时应当能从 S/R 的差异解释，而不是随机；② 检查 `conditions` 的两条判据是否按定义计算：`Q[second] − Q[first] >= 1/(m−1)`（自测里 m = 4 时门槛为 1/3，可直接手算验证）；③ v 的敏感性：v 取 0、0.5、1 三档重跑，报告名次翻转情况——v = 0 时 Q 完全由 R 决定、v = 1 时完全由 S 决定，可用于检查实现是否把 S/R 权重写反；④ 极端数据：所有方案在所有指标上相同 → 全部指标被跳过 → S = R = 0、Q = 0，应返回有限值而不是 NaN。

---

#### `grey_relational_grade(X, reference=None, weights=None, benefit=None, normalize="minmax", rho=0.5)`

- **数学形式**：ξ_ij = (d_min + ρ·d_max)/(|z0_j − z_ij| + ρ·d_max)，其中 d_min、d_max 为全部 |z0_j − z_ij| 的最小/最大值；关联度 grade_i = Σ_j w_j ξ_ij。
- **步骤**：① 成本型指标极差反转；② 按 `normalize` 无量纲化（minmax 映射到 [0,1]、mean 除以列均值绝对值、none 不做）；③ 参考序列 z0 取各列最优值（或把用户给的**原始量纲** `reference` 施加同样变换）；④ 算绝对差矩阵与 d_min/d_max；⑤ 按公式得关联系数矩阵并加权求和。
- **复杂度**：时间 O(mn)；空间 O(mn)。
- **参数**：`normalize="minmax"`（无量纲化口径，三选一；`"none"` 量纲风险自负，量纲大的指标会主导差值）；`rho=0.5`（分辨系数，取值 (0,1]：越小则关联系数间差异放得越大，很多论文直接用 0.5 却不说明，严格来说应做 ρ 的敏感性分析）；`weights=None`（None 表示等权 1/n）；`reference=None`（None 用样本内各列最优；给了外部理论最优时必须写清来源，否则读者无法判断该"最优"是否可达）；`benefit=None`（**成本型指标必须标出来**）。
- **陷阱**：**必须先无量纲化**，否则量纲大的指标主导差值；**成本型指标必须先正向化**——忘了标 `benefit` 会把最差的方案推成第一名（本模块自测专门构造了这份"不做正向化时结论会变"的数据，可作反例）；d_min 只在样本行与参考序列之间取，若把参考序列自己也放进差矩阵则 d_min 恒为 0，两种口径不同、论文要写明；某列完全恒定时 `safe_divide` 返回 0（该指标不提供区分信息），不会产生 NaN；关联度只表示"接近参考序列的程度"，**不是距离也不是概率**，不要跨模型比较数值大小；`mean` 口径下除以 `|pos|` 的列均值，含 0 值较多的列会被放大，需注意。
- **怎么检验**：① **闭式/极端检验**：若某方案在所有指标上都恰好等于参考序列，其关联度必须是 1（每项 |差值| = 0，ξ = (d_min + ρd_max)/(ρd_max) = 1 当且仅当 d_min = 0；否则应等于 (d_min + ρd_max)/(ρd_max)，这一点要按你的口径解释）；② 构造单调数据验证方向性：让某方案在每个指标上都一致优于另一方案，其关联度必须更大；③ **正反例对照**：同一份数据分别带 `benefit` 与不带 `benefit` 跑，成本型指标存在时两种结论应当不同——自测就是用这一对结果作为"正向化不可省"的证据；④ ρ 的敏感性：ρ 从 0.1 扫到 1，关联度差异应随 ρ 减小而放大，名次可能翻转，报告翻转比例。

---

#### `dea_ccr(inputs, outputs)`

- **数学形式**：对每个 DMU k 解输入导向包络 LP：min θ s.t. Σ_j λ_j x_ij ≤ θ x_ik（∀ 投入 i），Σ_j λ_j y_rj ≥ y_rk（∀ 产出 r），λ_j ≥ 0。
- **步骤**：① 校验投入/产出非负、DMU 数一致；② 对每个 DMU 构造 (ni + no) 个不等式约束的 LP（共 1 + n_dmu 个变量：θ 与 λ）；③ 调用 `simplex_lp` 求解，记录 θ、λ 与求解状态；④ 取 λ > 1e-7 的 DMU 作为该单元的对标（peers）；⑤ 按 θ 降序给名次并统计"DEA 有效"（θ ≥ 1 − 1e-6）的个数。
- **复杂度**：时间 O(n_dmu·LP(n_dmu))——每个 DMU 解一个含 O(n_dmu) 变量的 LP；空间 O(n_dmu²)（λ 矩阵）。
- **参数**：`inputs`/`outputs`（形如 (n_dmu, n_input) 与 (n_dmu, n_output) 的非负矩阵；**投入产出方向不能搞反**，把成本放进 outputs 会得到完全错误的结论）。本函数没有算法超参，`peers` 的阈值 1e-7 与"有效"阈值 1e-6 写死在代码里。
- **陷阱**：**效率为 1 的 DMU 可能有很多个**，CCR 无法再区分它们（需要区分时用超效率 DEA，或在论文里明确指出"多个 DMU 同为 DEA 有效"）；**投入产出指标数之和不应超过 DMU 数量的约 1/3**（经验法则），指标太多时几乎所有 DMU 都会变成有效、结论没有信息量；负值不允许，含负值要先平移并说明；本实现是**输入导向**（产出不变、最小化投入），输出导向会得到不同数值；LP 求解失败（status 非 optimal）时该 DMU 的效率保持为 0 且 λ 全 0，**代码不报错**——必须检查 `statuses`，否则会把"求解失败"误读成"效率极低"。
- **怎么检验**：① 构造**已知答案**的 DMU：一个在所有投入上都最小、在所有产出上都最大的"绝对标杆"，其效率必须恰为 1；一个投入大而产出小的单元效率必须显著小于 1；② **BCC ≥ CCR**：同一份数据跑 `dea_bcc`，技术效率必须 ≥ CCR 效率（BCC 约束更松）——自测把这条写成断言，反了就是代码有问题；③ 规模效率分解 SE = TE_CCR / TE_BCC ∈ (0,1]，可作独立核算；④ 尺度检验：把某 DMU 的投入与产出同比例放大（规模报酬不变下）效率应当不变；⑤ 与成熟实现（如 `pyDEA`/`pulp` 直接建模）在同一算例上对拍 θ 值；⑥ 断言 λ 的非零项对应的 peers 在原始数据上确实"支配"该 DMU。

---

#### `dea_bcc(inputs, outputs)`

- **数学形式**：与 CCR 相同的包络 LP，额外加凸性约束 Σ_j λ_j = 1，得到**技术效率** TE（把规模有效从技术效率中剥离）。
- **步骤**：① 非负性与形状校验；② 每个 DMU 解一个带等式约束 Σλ = 1 的输入导向 LP；③ 记录 θ、λ、status；④ 提取 peers、名次与有效个数。
- **复杂度**：同 `dea_ccr`：时间 O(n_dmu·LP(n_dmu))；空间 O(n_dmu²)。
- **参数**：`inputs`/`outputs`（同 `dea_ccr`，非负、方向不能反）。无可调超参。
- **陷阱**：**BCC 的技术效率一定 ≥ 对应 CCR 的效率**（约束更松），结果反了说明代码或数据有问题；只有 BCC 与 CCR 都算出来才能得到规模效率 SE = TE_CCR/TE_BCC，不报 SE 就说不清"效率低"是技术差还是规模不经济；与 CCR 一样不允许负值、指标不宜过多；同样存在"多个 DMU 同为有效"与"LP 求解失败被静默记为效率 0"的问题（务必看 `statuses`）。
- **怎么检验**：① 断言 `np.all(TE_BCC >= TE_CCR − 1e-7)`（自测里的独立结论就是这条）；② 规模报酬不变的数据：把某 DMU 的投入产出同比例缩放后 TE 应不变、CCR 也不变；③ 构造只有一个投入一个产出的算例，此时 CCR 效率 = min_i(y_i/x_i)/(y_k/x_k) 有**闭式解**，可直接对拍；④ 规模效率 SE 应在 (0,1] 内，且 SE < 1 的 DMU 在 CCR 与 BCC 下的名次差应能由规模解释。

---

#### `fuzzy_comprehensive_eval(weights, membership, operator="weighted", level_scores=None)`

- **数学形式**：加权平均型 B_j = Σ_i w_i r_ij（即 B = W R）；主因素决定型 B_j = max_i min(w_i, r_ij)；综合隶属度归一化 Bn = B/ΣB；若给等级分值则 level_value = Σ_j Bn_j·level_scores_j。
- **步骤**：① 校验隶属度矩阵在 [0,1] 内、权重非负并归一化；② 检查每行隶属度和是否为 1（不为 1 时写进 `note`，**不做强制归一化**）；③ 按 `operator` 合成；④ 和为正值时归一化；⑤ 取 argmax 得等级下标，可选地按分值折算连续得分。
- **复杂度**：时间 O(nk)（n = 因素数、k = 评语等级数，注意这里 n 不是样本数）；空间 O(nk)。
- **参数**：`operator="weighted"`（`"weighted"` 加权平均型，多数场景应当用；`"max_min"` 主因素决定型，只让最大的那个因素说话、权重小的因素完全不起作用，容易得出极端结论）；`level_scores=None`（各评语等级的量化分值，长度 k；给了才返回 `level_value`，否则为 None）；`weights`（长度必须等于隶属度矩阵行数，自动归一化到和为 1）。
- **陷阱**：**最大隶属度原则会丢信息**：B = (0.45, 0.44, 0.11) 说"属于第一级"很勉强，论文里应当把完整向量列出来而不是只报等级；权重与隶属度都必须落在 [0,1]——隶属度越界会报错，但**权重 > 1 不会被拒绝**，只会被静默归一化（`note` 里会说明），这与 docstring 里"权重必须落在 [0,1]"的说法不一致，使用时要注意；返回的 `level` 是 **0 基下标**，写论文时要把它换算成"第几级"；`note` 优先显示"行和不为 1"的提示，此时权重归一化的提示会被覆盖。
- **怎么检验**：① **手算对拍**：W = [0.4, 0.3, 0.3]，R = [[0.6,0.3,0.1],[0.4,0.4,0.2],[0.2,0.5,0.3]] 时 B = [0.42, 0.39, 0.19]（加权平均型），可完全手算验证（自测就用这组数）；② 单因素退化：只有一个因素（n = 1）时 B 必须等于该行隶属度、等级为最大隶属度所在列；③ `level_scores` 的线性性：等级分值整体加常数 c，`level_value` 必须恰好平移 c；④ 与 max-min 合成对照：用同一 R 检查 max_min 的输出被"大因素"主导（每列 B_j = max_i min(w_i, r_ij)），验证算子实现无误；⑤ 权重敏感性：把最不重要因素的权重趋近 0，综合向量应连续退化而不是跳变。

---

#### `topsis_rank_sensitivity(X, weights, benefit=None, delta=0.2, n_samples=2000, seed=None)`

- **数学形式**：对每个权重加乘性扰动 w_j′ = w_j·(1 + U(−δ, δ))，非负截断并归一化后重算 TOPSIS 排序；统计每个方案名次发生变化的频率（翻转概率）与"原第一名保持第一"的频率。
- **步骤**：① 用原始权重跑一次 TOPSIS 得 `base_rank`；② 抽样 `n_samples` 次生成扰动权重，逐次重算名次，堆成 `rank_matrix`；③ `rank_flip_prob` = 各方案名次不等于基准的频率；④ `best_keep_prob` = 基准第一名的方案在抽样中仍排第一的频率。
- **复杂度**：时间 O(n_samples·mn)；空间 O(n_samples·m)（名次矩阵）。
- **参数**：`delta=0.2`（权重相对扰动幅度，0.2 表示每个权重在 ±20% 内变动；这是**局部**灵敏度——若某个权重实际可能取到 0 或翻倍，必须扩大 δ 或改做全局扫描/网格/拉丁超立方）；`n_samples=2000`（抽样次数，越大频率估计越稳，论文里要报告这个次数）；`seed=None`（默认 `DEFAULT_SEED`，随机抽样结果依赖它，报告时一并给出）；`weights`/`benefit`/`X` 同 `topsis`。
- **陷阱**：**局部**灵敏度分析，仅在原权重附近扰动，结论不能外推到任意权重变化；`rank_flip_prob` 高**不代表模型差**，只代表方案之间本来就很接近，论文里应当据此讨论"需要更精确的数据"而不是硬挑一个第一名；扰动后权重全为 0 的退化抽样会回退到原权重（发生概率极低但代码有兜底）；名次并列的口径由 `_ranks` 决定（并列同名次），翻转概率会因此被低估。
- **怎么检验**：① 极限检验：`delta=0` 时所有抽样名次必须等于 `base_rank`、`rank_flip_prob` 全 0、`best_keep_prob` = 1；② **闭式检验**：构造一个"某方案在所有指标上都严格优于其他方案"的数据，则任意扰动下它都应保持第一，`best_keep_prob` 必须 = 1；③ 再构造一组完全对称/难分的数据，`best_keep_prob` 应显著小于 1，验证指标确实能捕捉到不稳定性；④ 收敛性：把 `n_samples` 从 200 提到 5000，两次的频率估计应趋于稳定（说明蒙特卡洛误差已小），这比"换个种子看数值一致"更严格。

---


### 3.7 聚类与分类 —— `examples/algorithms/clustering.py`

这一族解决**无监督分组与簇数选择**：把 n 个样本按欧氏距离分成 k 组（硬划分或软隶属度），
并用有效性指标（轮廓系数、DB、CH、gap）判断划分好坏。

共同约定（聚类题最常失分的地方）：

- **先标准化**：所有距离都是欧氏距离，特征未标准化时"距离"会被量纲大的特征支配；
  本模块**不做**自动标准化，是否标准化必须由你决定并在论文里写明（可用 `_common` 的
  `normalize_minmax` / `normalize_l2`）。
- **有效性指标方向固定**：`silhouette_score`/`calinski_harabasz_score`/`gap_statistic` **越大越好**，
  `davies_bouldin_score` **越小越好**；都只是相对比较口径，**不要跨数据集比绝对值**。
- **随机性**只出现在 K-means++ 初始化、GMM/谱聚类/模糊 C 均值的初始化与 gap 参考集，
  统一走 `_common.rng`，`seed=None` 即 `DEFAULT_SEED`。
- **标签口径**：DBSCAN 的噪声点是 `-1`，簇编号从 0 开始（其余函数的标签也是 0 基，且编号本身无意义，
  只有"同/不同"有意义）。

---

#### `kmeans_plusplus_init(X, k, seed=None)`

- **数学形式**：D(x)² 为样本 x 到已选中心集合的最近距离平方；以概率 D(x)²/Σ_x′ D(x′)² 抽取下一个中心（Arthur–Vassilvitskii 的 D² 采样）。
- **步骤**：① 均匀随机取第 1 个中心（它是真实样本的副本）；② 维护每个样本到已选中心的最近距离平方；③ 按 D² 概率抽下一个中心；④ 重复直到选满 k 个。
- **复杂度**：时间 O(nkd)；空间 O(n + kd)（每步的距离向量 + 中心矩阵）。**注**：docstring 写的空间 O(nk) 偏保守，实际只需 O(n) 的距离缓存。
- **参数**：`k`（簇数，必须是整数、1 ≤ k ≤ n，非法直接 `ValueError`）；`seed=None`（默认 `DEFAULT_SEED`，换种子得到不同初始中心）；`X`（样本矩阵）。
- **陷阱**：**数据高度重复时 D² 会同时为 0**，此时按均匀随机抽点（本实现如此），k 个中心可能出现重复样本，进而在 K-means 第一次分配后产生空簇——所以 `kmeans` 里必须有空簇修复；D² 采样是**有随机性**的，同一个 k 不同种子会给出不同初始中心，进而不同局部最优；本函数只负责挑初始中心，不做任何迭代优化。
- **怎么检验**：① **确定性检验**：同一 `seed` 两次调用必须逐位相同（用 `np.array_equal`）；② 性质检验：返回的 k 个中心都必须是 `X` 中真实出现的行（D² 采样只从样本中抽取，不是样本均值）；③ 距离分离性：在人工构造的 k 个远离的高斯团上，固定种子跑多次应有相当比例的次数"每个团恰好抽到一个中心"，明显低于 k 个团各一个时说明采样概率实现有误；④ 与"纯均匀随机初始化"对拍：在同样的团状数据上，D² 初始化的首次 `inertia` 分布应整体更优（这正是 k-means++ 的存在理由）。

---

#### `kmeans(X, k, seed=None, max_iter=300, tol=1e-8)`

- **数学形式**：min Σ_{j=1..k} Σ_{i∈C_j} ‖x_i − c_j‖²（SSE/inertia），交替执行硬分配 c(i) = argmin_j ‖x_i − c_j‖ 与中心更新 c_j = mean{x_i : c(i)=j}。
- **步骤**：① 用 `kmeans_plusplus_init` 取初始中心；② 分配：每点归到最近中心；③ 更新：每簇取均值，**空簇则把离自己中心最远的点改判为该簇的单点簇**；④ 中心矩阵最大位移 ≤ `tol` 时停止，否则重复到 `max_iter`；⑤ 用最终中心重算一次标签并给出 `inertia`。
- **复杂度**：时间 O(nkd·iter)；空间 O(nk)（每轮的 (n, k) 距离矩阵）+ O(nd)。
- **参数**：`k`（1 ≤ k ≤ n）；`max_iter=300`（迭代上限，太小会在未收敛时截断、`n_iter` 触顶）；`tol=1e-8`（收敛阈值，判据是**中心位移**而不是 inertia 变化——后者量级依赖数据尺度，阈值不好定；tol 放宽会更快但结果更粗）；`seed=None`（只传给 K-means++，控制初始中心）。
- **陷阱**：**只保证收敛到局部极小**，换种子 `inertia` 可能明显变差（docstring 记录过"可能差 20% 以上"的量级），论文中应固定种子并报告 `inertia`，或用多次重启取最优；**空簇**：初始化出现重复中心时某簇可能一个点都没有，直接取均值会得到 NaN，本实现用"抢最远点"修复——但该修复在改判后**不会回滚已处理簇的中心**（循环里按 j 顺序计算，被抢走的点可能已计入先前簇的均值），且多个空簇可能抢到同一个点，属于已知的近似修复；**量纲**：特征未标准化时 SSE 几乎没有解释力；`inertia` 随 k 单调不增，不可用于直接选 k。
- **怎么检验**：① **构造已知答案的数据**：本模块自测构造三个中心 (−5,−5)、(0,6)、(6,−4)、标准差 0.5 的二维高斯簇，各 30 点且簇间距远大于簇内散布，任何合理实现都应无歧义还原分组（断言各簇样本数）；② **极限行为**：k = 1 时 `inertia` 必须等于全体样本的总平方和（到全局均值的距离平方和），可闭式核对；k = n 时 `inertia` 必须为 0、`labels` 是 0..n−1 的一个排列；③ 与成熟库对拍（sklearn `KMeans(n_init=1, init=<同一初始中心>)`）比较 `inertia`，注意固定初始化才能逐位对比；④ 单调性/一致性：`inertia` 必须等于用返回的 `labels` 与 `centers` 重算的 SSE（自洽性检查）；⑤ **对抗性算例**：同心圆数据上 K-means 必须失败（自测把它写成断言：K-means 分对同心圆反而说明算例失去意义），这是检验球形假设的最直接方式。

---

#### `silhouette_score(X, labels)`

- **数学形式**：a(i) = 样本 i 到**同簇**其他点的平均距离，b(i) = 到**最近其他簇**的平均距离，s(i) = (b − a)/max(a, b)，返回全部样本 s(i) 的均值。
- **步骤**：① 校验标签长度；② 若标签只有 1 类，直接返回 0.0；③ 计算完整 n×n 距离矩阵；④ 逐点取同簇均值 a 与最小异簇均值 b；⑤ 单点簇（同簇无他人）记 s = 0；⑥ 取均值。
- **复杂度**：时间 O(n²d)；空间 O(n²)（完整距离矩阵；n = 20000 时约 3.2 GB，必须改用抽样版）。
- **参数**：`X`（样本矩阵）；`labels`（长度必须等于样本数，取值任意整数，只有"相同/不同"有意义——注意 DBSCAN 的 `-1` 噪声标签会**被当成一个正常的簇**参与计算，通常应当先剔除噪声点再算轮廓系数）。函数无可调超参。
- **陷阱**：**k = 1 或所有标签相同**时 b 不存在，本函数返回 0.0，调用方必须理解成"指标不可用"而不是"聚类质量等于随机"；单点簇的 s 被约定为 0，会拉低均值，**簇很多时轮廓系数天然偏低**，不同 k 之间比较要谨慎；O(n²) 内存与时间，大样本必须抽样；轮廓系数隐含凸/球形簇假设，对 DBSCAN 找出的环形簇会给出很低的分数，这**不代表**划分错误。
- **怎么检验**：① 与成熟库对拍（sklearn `silhouette_score`）在同一份数据、同一标签上必须数值一致（注意它用 `metric='euclidean'`，与本节口径相同）；② **已知答案**：两个相距很远、内部很紧的簇，轮廓系数应接近 1；完全随机标签下应接近 0 或为负；把两组数据的间距拉大，该值必须单调不减（自测里的三簇数据就是"完美划分"的正例）；③ 恒等检验：标签整体重命名（如 0/1 → 5/7）后结果必须完全不变；④ 单点簇与 k = 1 的退化输入必须返回有限值（0.0）而不是 NaN。

---

#### `elbow_curve(X, k_range, seed=None)`

- **数学形式**：对一串 k 分别求 SSE(k)（K-means 的 inertia），曲线 SSE(k) 随 k 单调不增，其"下降速率突缓处"即经验最优 k。
- **步骤**：① 把 `k_range` 转成 int 列表（空列表报错）；② 对每个 k 调用一次 `kmeans`（**共用同一 seed**）；③ 返回 `{k: inertia}` 字典。
- **复杂度**：时间 O(Σ_k n·k·d·iter_k)；空间 O(nk)（每次调用内部的距离矩阵，不含轮廓系数）。
- **参数**：`k_range`（待尝试的 k 序列，例如 `range(1, 11)`；必须非空）；`seed=None`（所有 k 共用，保证曲线可比——**换不同种子会让曲线上下抖动、拐点判断失效**）；`X`（样本矩阵）。函数本身不做拐点检测，需要人工看或另配判据。
- **陷阱**：每个 k 用不同种子会让曲线抖动；`inertia` 随 k 单调不增，k = n 时恒为 0，所以"肘部"必须人工看或配合业务解释，**不要写成自动最优**；如果数据本身没有簇结构，曲线是平滑的、不存在肘部；`kmeans` 的局部最优也会让曲线非严格单调（同一数据不同 k 的初始化不同）——理论上 SSE 单调不增只在"嵌套最优解"下成立，实践中可能看到轻微回升。
- **怎么检验**：① **闭式检验**：k = 1 时 `inertia` 必须等于总平方和（到全局均值的距离平方和），k = n 时必须为 0——这两端可完全手算；② 构造三个分离良好的团，曲线应在 k = 3 附近出现明显拐点，且 SSE(3) 应接近"簇内真实方差 × n"，与手算值对拍；③ 与 gap statistic 交叉验证：同一数据上 `gap_statistic` 的 `k_hat` 应与肘部目测的 k 一致（自测数据上 gap 选出的 k 就是 3）；④ 断言 `dict` 的键与 `k_range` 完全一致、值都是有限非负数。

---

#### `agglomerative(X, k, linkage="average")`

- **数学形式**：自底向上合并：初始每个样本一簇，每步合并簇间距离最小的一对，簇间距离按 Lance-Williams 递推更新——single：d(i∪j, m) = min(d_im, d_jm)；complete：max；average（UPGMA）：按簇大小加权 (n_i·d_im + n_j·d_jm)/(n_i + n_j)。
- **步骤**：① 计算 n×n 距离矩阵并把对角置 inf；② 循环 n − k 次：取全局最小距离的一对 (i, j)（按扁平化 argmin），把 j 并入 i，用 Lance-Williams 更新 i 与其他簇的距离，再把 j 行/列置 inf 冻结；③ 合并结束后按簇顺序把标签重新编号为 0..k−1。
- **复杂度**：时间 O(n³)（朴素实现，每轮扫描 O(n²)、共 n 轮）；空间 O(n²)。
- **参数**：`k`（目标簇数，1 ≤ k ≤ n）；`linkage="average"`（簇间距离口径：`"single"` 最近点、`"complete"` 最远点、`"average"` 平均点距；single 会"链式"串联、对噪声敏感，complete/average 更常用）。
- **陷阱**：**决策贪心且不可撤销**，早期合并错了无法回退；`single` 链路效应会让两个真正分开的簇通过一两个中间点连成一片；`average` 口径要**按簇大小加权**，写成等权平均是常见错误（本实现用 sizes 加权）；必须用 Lance-Williams 更新后的簇间距离而不是对成员点重算（后者等价但更慢）；不支持缺失值，含 NaN/inf 会在 `as_matrix` 处直接报错；当簇间距离出现并列时，`argmin` 取扁平化后的第一个索引，结果依赖样本顺序（换行序可能得到不同的等价划分）。
- **怎么检验**：① **已知答案**：三个远离的高斯团上，三种 linkage 都应还原真值分组（可断言各簇样本数）；② 与成熟实现（scipy `linkage` + `fcluster`）对拍：同一数据、同一 linkage 与目标簇数下划分应一致（注意 scipy 的 `average` 就是 UPGMA、`single`/`complete` 口径相同，可直接比对簇成员集合）；③ **链式反例**：构造"两团 + 一两个中间点"的数据，验证 single 会把它们连成一片而 complete/average 不会——这是选择 linkage 时最该看的实验；④ 退化检验：k = n 时每个样本自成一簇（标签是 0..n−1 的排列），k = 1 时全为同一标签；⑤ `average` 加权口径：构造两个大小悬殊的簇，检查合并距离等于按大小加权的平均而不是算术平均。

---

#### `dbscan(X, eps, min_pts)`

- **数学形式**：核心点定义：|N_eps(x)| ≥ min_pts（N 含 x 自身）；密度可达的传递闭包构成簇，不属于任何簇的点标为噪声 −1。
- **步骤**：① 先算满距离矩阵（O(n²)）；② 对每个点求 eps 邻域并判定是否核心点；③ 从每个未访问的核心点出发用栈扩展，把密度可达的点归入同一簇（边界点即使不是核心点也入簇但不再扩展）；④ 剩下的点标 −1。
- **复杂度**：时间 O(n²)（本实现先算满距离矩阵；用 KD 树可降到 O(n log n)）；空间 O(n²)。
- **参数**：`eps`（邻域半径，必须 > 0；**用同一把尺子，所以必须先标准化**，量纲不统一时同一个 eps 在不同特征上含义完全不同；经验做法是看 k-距离曲线的拐点）；`min_pts`（核心点的邻域最少点数，**含自己**，必须 ≥ 1；sklearn 默认 2·维度且同样含自己，对比时注意口径）；`X`（样本矩阵）。
- **陷阱**：**eps 极难调**：太大所有点连成一簇，太小全是噪声（−1），务必在论文里打印簇数与噪声比例；`min_pts` 是否包含自己会差 1，跨实现对比时要统一；边界点可能同时挨着两个簇，**归属取决于遍历顺序**（本实现用栈 `pop()` 扩展，实际是深度优先，注释里写的"BFS"只是习惯说法），同一份数据换个点的顺序结果可能微变；本实现预先算满距离矩阵，大 n 会爆内存；不做任何自动参数选择。
- **怎么检验**：① **构造已知答案的数据**：自测用两个 4 点小团（相距约 7）加一个孤立点（(10,0)），`eps=0.6, min_pts=3` 时应得到两簇、孤立点标 −1——可闭式核对噪声点数与标签序列；② 极限行为：`eps` 极小时所有点都是噪声、`eps` 大于数据直径时所有点一簇，两种极限都必须出现（用来确认邻域与扩展逻辑方向没写反）；③ 与 sklearn `DBSCAN(eps, min_pts)` 对拍：在无边界点歧义的数据上标签应完全一致（有歧义时只有边界点可能不同，且差异应当仅出现在"到两个核心点距离都 ≤ eps"的点上）；④ 平移/旋转不变性：数据整体平移、旋转或整体缩放并同步缩放 `eps` 后，聚类结果（相对结构）应保持不变。

---

#### `gmm_em(X, k, seed=None, max_iter=200, tol=1e-8, reg_covar=1e-6)`

- **数学形式**：混合模型 p(x) = Σ_j w_j N(x | μ_j, Σ_j)；E 步责任 r_ij ∝ w_j N(x_i | μ_j, Σ_j)（按对数密度 + 行最大值 softmax 计算）；M 步 N_j = Σ_i r_ij，w_j = N_j/n，μ_j = Σ_i r_ij x_i/N_j，Σ_j = Σ_i r_ij (x_i−μ_j)(x_i−μ_j)ᵀ/N_j + reg_covar·I；对数似然单调不减直至增量 ≤ tol。
- **步骤**：① 用 `kmeans` 的中心作初始均值、簇内比例作初始权重、簇内协方差（加 reg_covar·I；样本数不超特征数时退化为全局面协方差）作初始协方差；② E 步算对数密度 → 行最大值 softmax → 责任；③ M 步加权更新三组参数，空分量把权重压到 1e-12、均值重置为 `x[0]`、协方差重置为全局协方差 + reg_covar·I；④ 平均对数似然增量 ≤ tol 或达 `max_iter` 停止；⑤ 用最终参数重算一次责任与似然，保证 labels/loglik 自洽；⑥ 输出 BIC。
- **复杂度**：时间 O(n·k·d³·iter)（d³ 来自每分量每轮的 Cholesky）；空间 O(nk + kd²)。
- **参数**：`k`（分量个数，1 ≤ k ≤ n）；`max_iter=200`（EM 迭代上限）；`tol=1e-8`（收敛阈值，判据是**平均**对数似然增量，注意与 `loglik` 的"平均"口径一致）；`reg_covar=1e-6`（加到每个协方差对角上的正数，≥ 0；**必须大于 0**，否则单点/共线分量会让协方差奇异、`slogdet` 符号非正并直接报错）；`seed=None`（只用于初始 K-means，控制初始均值）。
- **陷阱**：**必须加 reg_covar**，否则对数似然会变成 −inf 或抛 "协方差矩阵非正定"；加权协方差的分母是 N_j = Σ_i r_ij 而**不是**簇内硬标签计数，用错会把责任很小的点也等权算进去；EM 只保证收敛到**局部极大**且依赖初始化，`labels` 取后验 argmax，与 K-means 的硬分配在边界点上会不同；`loglik` 返回的是**平均**对数似然，`bic` 里必须换回总和再乘 −2（实现里乘了 n_samples），混用会差 n 倍；BIC 的参数个数取 (k−1) + k·d + k·d(d+1)/2（含权重自由度），docstring 里另一处写成 k·(d + d(d+1)/2) 与之不一致，**以代码为准**；协方差矩阵可能不是正定（数据共线或 k 过大）时直接 `ValueError` 而不是静默给 NaN。
- **怎么检验**：① **已知答案**：自测构造的三簇高斯数据上，GMM 必须**逐一还原**真值分组（按每 30 个点的真值下标比对），这是最强的独立检验；② 与成熟库对拍（sklearn `GaussianMixture(covariance_type='full')`）比较收敛后的平均对数似然与均值/协方差——注意初始化和 tol 口径不同，`loglik` 应在同一量级，`labels` 在不含边界点的数据上应一致；③ 自洽性：用返回的 `means/covariances/weights` 独立重算一次后验与对数似然，必须与返回的 `loglik` 一致（到 1e-10），并断言 `weights.sum() == 1`；④ EM 单调性：在 `max_iter` 足够大、不触发空分量处理的正常数据上，逐轮平均对数似然必须单调不减，出现下降说明 E/M 步公式或 reg_covar 处理有误；⑤ BIC 口径：参数个数与 `k*log(n) − 2L_total` 的手算值核对。

---

#### `spectral_clustering(X, k, sigma=None, seed=None, n_neighbors=3)`

- **数学形式**：相似度 W_ij = exp(−‖x_i−x_j‖²/(2σ²))（全局带宽）或 exp(−‖x_i−x_j‖²/(σ_iσ_j))（局部尺度，取 (W+Wᵀ)/2 对称化），对角置 0；度矩阵 D = diag(row sum)；归一化拉普拉斯 L_sym = I − D^(−1/2) W D^(−1/2)；取 L_sym 最小的 k 个特征向量 U 并按行归一化到单位长度；对 U 的行跑 K-means。
- **步骤**：① 算距离矩阵；② 由 `sigma` 决定全局或局部带宽构造亲和矩阵并对角置 0；③ 算度、构造对称化的 L_sym；④ `np.linalg.eigh` 取前 k 个最小特征值对应的特征向量；⑤ 行归一化；⑥ K-means 得到硬标签。
- **复杂度**：时间 O(n³)（稠密特征分解，与 k 无关）；空间 O(n²)。
- **参数**：`k`（簇数，1 ≤ k ≤ n）；`sigma=None`（高斯核带宽：None 时用**局部尺度**——第 i 个点取自己的第 `n_neighbors` 近邻距离 σ_i；给了正数则用全局带宽，经典 Ng-Jordan-Weiss 口径）；`n_neighbors=3`（局部尺度的近邻阶数，必须 ≥ 1，仅当 `sigma is None` 生效 **且取值不能太大**：越大 σ_i 越接近全局中位距离，密度差异大的数据会退化回全局带宽的效果）；`seed=None`（只传给最后的 K-means）。
- **陷阱**：**带宽是真正的超参**：太小则相似度退化成近邻图的近单位阵、图不连通，K-means 会把每簇再切开；太大则所有点彼此相似、谱结构消失；全局"距离中位数"在**簇密度差异大**时必然失败（中位数由稠密簇的点对决定，对稀疏簇来说远大于其内部尺度），这正是本实现默认用局部尺度的原因；**必须做行归一化**再用 K-means，否则特征向量的模长随簇大小变化、K-means 会按模长而不是方向切分；特征值重根时（两个完全对称的簇）特征向量在子空间内任意旋转，不同 LAPACK 版本给出的基不同，标签可能整体等价但数值不一致；稠密 O(n³)，n 上千就很慢；`affinity` 是数据依赖的，**不能跨数据集比较**，与 sklearn（用最近邻图）的 affinity 完全不同，只能比对 labels。
- **怎么检验**：① **存在理由算例**：两个同心圆（半径 1.5/3.5）上 K-means 必然把每个环各切一半，而谱聚类用相似度图的连通结构应完全分开——自测直接断言"谱聚类完全正确"且"K-means 完全错误"，两条一起才有说服力；② 与成熟库对拍：在分离良好的数据上比较 `labels` 的划分（允许整体重编号），注意 sklearn 默认用最近邻图，需把 `affinity='rbf'` 与 `gamma` 调成与你的 σ 对应才能比较亲和矩阵；③ 谱性质检验：L_sym 的最小特征值应为 0（连通图）/接近 0（近似不连通块），且当图为 c 个不相连的块时前 c 个特征值都为 0——自测数据上可检查 `eigenvalues` 的形状趋势；④ 带宽敏感性：把 `sigma` 从很小扫到很大，观察"碎成 n 簇 → 正确 k 簇 → 全部并成一簇"的相变，验证实现没有把 W 的尺度写错。

---

#### `fuzzy_cmeans(X, k, m=2.0, seed=None, max_iter=150, tol=1e-8)`

- **数学形式**：min J = Σ_i Σ_j u_ij^m ‖x_i − c_j‖²，约束 Σ_j u_ij = 1；交替更新 u_ij = 1/Σ_l (d_ij/d_il)^(2/(m−1))，c_j = Σ_i u_ij^m x_i / Σ_i u_ij^m。
- **步骤**：① 用一次 `kmeans` 的中心作初始 c_j；② 求 d_ij 与比值张量，按公式更新隶属度（某点与某中心重合 d = 0 时把该点对该簇置 one-hot）；③ 用 u^m 加权更新中心；④ 隶属度最大变化 ≤ tol 或达 `max_iter` 停止；⑤ 取 argmax 得硬标签并计算目标值 J。
- **复杂度**：时间 O(nkd·iter)；空间 O(nk)。
- **参数**：`k`（1 ≤ k ≤ n）；`m=2.0`（模糊指数，必须 > 1：m → 1 退化为硬 K-means 且容易震荡不收敛，m 太大则所有 u_ij → 1/k、软划分失去分辨力，常用 m ∈ [1.5, 2.5]）；`max_iter=150`（迭代上限）；`tol=1e-8`（收敛阈值，判据是**隶属度**最大变化而不是目标函数变化——J 在 m 较大时几乎不变，用它做判据会提前停止）；`seed=None`（只用于初始 K-means）。
- **陷阱**：`d_ij = 0` 会让 u_ij 出现 0/0，**必须显式处理重合点**，否则整行变 NaN 并在下一步污染所有中心（本实现把该点判给最近的簇并置 one-hot，同时用 `safe` 矩阵兜住除法）；`labels` 取 argmax 是**事后硬化**，扁平隶属度（多点 u ≈ 1/k）的归属对初始化和迭代轮数都敏感，报告时应同时给出最大隶属度的分布；隶属度矩阵每行和必须为 1（可断言，自测就用这条）；收敛判据不是目标函数，因此单看 `objective` 可能看不出迭代是否真的收敛。
- **怎么检验**：① **行和与取值范围**：断言行和偏差 ≤ 1e-9 且 u ∈ [0,1]（自测口径）；② 极限行为：m → 1⁺ 时隶属度应趋近 0/1（与 K-means 的硬划分趋同），m → ∞ 时所有 u_ij → 1/k——用 m = 1.01 与 m = 50 各跑一次即可验证实现的方向性；③ **构造已知答案**：三个远离的高斯团上 argmax 标签必须与真值分组一致，且各点最大隶属度应接近 1；④ 单调性：在无重合点、m 正常的数据上，目标函数 J 应逐轮不增（标准的 FCM 下降性质），出现明显上升说明更新公式或指数写错；⑤ 与成熟实现（如 `skfuzzy.cluster.cmeans`）对拍目标值与中心（注意初始化口径不同，可比目标值量级与最终中心位置）。

---

#### `davies_bouldin_score(X, labels)`

- **数学形式**：S_j = 簇 j 内样本到簇心 c_j 的**平均**欧氏距离（散度），M_jl = ‖c_j − c_l‖，R_j = max_{l≠j} (S_j + S_l)/M_jl，DB = (1/k)·Σ_j R_j。
- **步骤**：① 校验标签长度；② 只有 1 个簇时返回 0.0；③ 逐簇求均值中心与平均散布 S_j；④ 算簇心两两距离；⑤ 逐簇取比值最大者；⑥ 取平均。任意一对簇心重合（距离 ≤ 1e-12）时直接返回 `inf`。
- **复杂度**：时间 O(nd + k²d)；空间 O(nd)。
- **参数**：`X`（样本矩阵）；`labels`（长度必须等于样本数）。函数无可调超参，**口径固定为平均距离、欧氏距离**。
- **陷阱**：两个簇心完全重合时比值发散，本实现返回 `inf`，调用方应把 inf 当成"这个划分不可用"而不是"很大"；S_j 用**平均**距离而不是距离之和（用和会让指标随簇大小单调变化、大簇被冤枉，选出的 k 系统性偏大——本实现与 sklearn 同口径）；指标依赖尺度，**必须先标准化**；只适用于凸/球形簇（DBSCAN 找出的环形簇会被判很差）；只有 1 个簇时返回 0.0 是"无簇间比较"的约定，**不是"最好"**。
- **怎么检验**：① **方向性检验**：完美分组（真值标签）的 DB 值必须远小于随机标签的 DB 值——自测断言 `db_perfect * 5 < db_random`，方向反了就是公式写错；② 与 sklearn `davies_bouldin_score` 对拍，同一数据同一标签应数值一致（口径相同）；③ 手算对拍：一维、两个簇各两个点的微型算例可以完全手算 S_j、M_jl 与 DB；④ 尺度敏感性：把特征整体乘以 10，DB 值应保持不变（比值是无量纲的）——若变了说明某处多乘/少乘了尺度；⑤ 退化输入：所有点同标签 → 0.0；两个簇心重合 → `inf`。

---

#### `calinski_harabasz_score(X, labels)`

- **数学形式**：BGSS = Σ_j n_j‖c_j − c̄‖²（簇间平方和，自由度 k−1），WGSS = Σ_j Σ_{i∈C_j}‖x_i − c_j‖²（簇内平方和，自由度 n−k），CH = [BGSS/(k−1)]/[WGSS/(n−k)]。
- **步骤**：① 校验标签长度；② 若 k < 2 或 k ≥ n，返回 0.0；③ 求全局均值与各簇中心；④ 累加 BGSS 与 WGSS；⑤ WGSS ≤ 1e-300 时返回 0.0，否则按公式返回。
- **复杂度**：时间 O(nd)；空间 O(nd)。**这是本模块中最省的一个有效性指标**（不需要距离矩阵）。
- **参数**：`X`（样本矩阵）；`labels`（长度必须等于样本数）。无可调超参。口号注意：docstring 的"算法"一节写了一个带额外因子 (n−k)/(k−1) 的"等价形式"，那是错的——**代码用的是标准定义** `[BGSS/(k−1)]/[WGSS/(n−k)]`，与 sklearn 一致。
- **陷阱**：k = n 时 WGSS = 0 且 n − k = 0，公式 0/0，本实现返回 0.0，**不要当成最优**；k = 1 同样返回 0.0（"不可用"而不是"最差"）；CH 对 k 有单调倾向（分子自由度惩罚不够强），在无簇结构的数据上也会偏好较大的 k，所以**只能用来比较同一份数据的候选划分**；同样是球形簇口径，对环形/月牙形数据没有意义；`labels` 里若含 DBSCAN 的 −1，噪声点会被当成一个正常簇参与计算。
- **怎么检验**：① **方向性检验**：完美分组的 CH 必须远大于随机标签——自测断言 `ch_perfect > 3 * ch_random`；② 与 sklearn `calinski_harabasz_score` 对拍（口径相同，应数值一致）；③ **手算对拍**：一维两个簇的微型算例可手算 BGSS/WGSS 与自由度；④ **等价性检验**：CH 与 K-means 的目标函数存在恒等关系（BGSS + WGSS = 总平方和 TSS，为常数），因此对同一数据同一 k，CH 的大小关系与 `inertia` (=WGSS) 的**反向**关系一致——可用它交叉验证两个函数没有各算错一路；⑤ 退化输入：k = 1、k = n、所有点重合时都应返回有限值。

---

#### `gap_statistic(X, k_max=6, n_refs=10, seed=None)`

- **数学形式**：Gap(k) = (1/B)Σ_b log W_kb − log W_k，其中 W_k 为观测数据在 k 簇下的簇内距离**之和**；s_k = sd_b(log W_kb)·√(1 + 1/B)；判据 k̂ = 第一个满足 Gap(k) ≥ Gap(k+1) − s_{k+1} 的 k。
- **步骤**：① 对 k = 1..k_max 在观测数据上跑 `kmeans` 得 log W_k；② 生成 `n_refs` 个参考集：每维在观测数据的 [min, max] 上独立均匀采样、形状相同；③ 对每个参考集重复第 1 步得 log W_kb，按公式算 Gap 与 s_k；④ 按判据取第一个满足的 k；⑤ 直到 k_max 都不满足时**返回 k_max**（不是"gap 最大的 k"——docstring 这一句与代码不符，以代码为准）。
- **复杂度**：时间 O(n_refs·k_max·n·k·d·iter)（最外层是 k_max × n_refs 次 K-means）；空间 O(nd + n·k_max)。
- **参数**：`k_max=6`（尝试的最大簇数，必须 ≥ 2 且 ≤ n，**必须显著小于 n**：W_k 用距离之和，k = n 时恒为 0、log W → −inf）；`n_refs=10`（每个 k 的参考数据集个数，越大 sk 越稳、k̂ 越可复现，但代价是线性增长的 K-means 次数）；`seed=None`（参考集与每个 k 的 K-means 初始化都由它派生；**注意当 seed=None 时参考集种子由 0 派生而不是 DEFAULT_SEED**，与模块"不给就用 DEFAULT_SEED"的约定不一致，虽然结果仍是确定性的）。
- **陷阱**：**参考分布必须与原数据同尺度**（本实现按每维观测的 [min, max] 均匀采样，与原论文一致；用标准正态会让 gap 的绝对值失去意义）；gap 曲线常常很平，k̂ 对 `n_refs` 和随机种子都敏感，论文里应报告 gap ± sk 曲线而不是只报一个 k̂；本实现是"同一个 k 复用同一批参考集、种子由 (seed, k, b) 稳定派生"的确定性口径，两次调用结果完全一致，但换成真正独立的参考集会得到不同的 k̂；参考集是逐维独立均匀采样，**忽略了指标间的相关性**，相关性强时 gap 会偏乐观。
- **怎么检验**：① **已知答案**：自测在三簇高斯数据上要求 `k_hat == 3`，可用它验证实现；② 与肘部曲线/轮廓系数交叉验证：同一数据上三种判据给出的 k 应大体一致，不一致时应在论文里讨论（这本身就是有价值的结论）；③ 与成熟库对拍（如 R 的 `cluster::clusGap` 或 `gap_statistic` 包的 Python 移植），注意参考集生成方式与 kmeans 初始化必须统一才能比较；④ 稳定性检验：把 `n_refs` 从 5 提到 20 并换 5 个种子重复，报告 k̂ 的分布——k̂ 在多数种子上不稳定说明数据本身没有清晰簇结构，这比硬报一个 k 更诚实；⑤ 极限检验：把参考集换成观测数据本身时 Gap 应恒为 0（可临时构造脚本验证公式实现无误）。


### 3.8 微分方程与动力系统 —— `examples/algorithms/differential.py`

**这族解决什么问题**：把常微分方程初值问题（Euler / RK4 / RK45 / 隐式 Euler）、传染病仓室模型（SIR / SEIR）、种群增长与捕食者-被捕食者、基本再生数、参数最小二乘拟合、收敛阶估计和离散动力系统（logistic 映射）做成一批定步长、显式格式、几行就能看懂的教学透明版实现，只依赖 numpy + 标准库 + `._common`。它的定位不是替代 `scipy.integrate`：刚性方程、长时间积分、事件检测（阈值触发）都该换成熟求解器；本模块的价值在于让论文能写清楚"我们用什么格式、步长多少、误差怎么估"。适合建模赛题里的疫情传播预测、种群演化、参数反演与混沌演示。

**共同约定**：
- **时间网格口径**：所有含 `t_span` / `T` + `h` 的函数都把 `h` 当**建议步长**。内部 `_grid` 取均匀步数 `n = ceil((t1−t0)/h)`，再 `np.linspace(t0, t1, n+1)`，因此真步长 Δt = (t1−t0)/n ≤ h 且**最后一个点严格落在 t1**。`t_span` 必须满足 t1 > t0、h > 0，否则 `_grid` 抛 `ValueError`。
- **rhs 签名约定**：右端函数统一是 `f(t, y) -> dy/dt`，`t` 是 float、`y` 是一维 float64 数组，返回值同样被 `as_vector` 归一化。`y` 既可以是标量也可以是一维数组（模块 docstring 明说）；`sir_rhs` / `lotka_volterra_rhs` / `seir_rhs` 都是**工厂函数**：先固定参数，再返回这样一个 `f`，所以它们自己**不做任何离散化**。
- **返回形态**：积分器返回 `(t, Y)` 元组，`t` 形状 (n+1,)、`Y` 形状 (n+1, m)，第 i 行是 t[i] 处的解；`logistic_growth` 特殊，返回 `(t, y)` 且 `y` 形状 (n+1,)（单变量已降维）。模型级封装（`simulate_sir` / `simulate_seir` / `fit_sir_least_squares` / `basic_reproduction_number` / `jacobian_stability` / `logistic_map`）返回 dict，键名用 snake_case 且尽量直白可打印成表格（`peak_infected`、`peak_time`、`final_size`、`R0` 等）。
- **单文件自足**：全模块不 import scipy（CI 用 AST 静态禁止），不依赖 pandas/matplotlib；只有 `_common.as_vector`、`_common.check_same_length`（以及随机部分用 `_common.rng`）这几个外部符号，其余算法全部手写。
- **随机性与 seed 口径**：凡是带随机成分的拟合（`fit_sir_least_squares` 的多起点）显式接收 `seed`，`None` 时回落到 `_common.rng`，即 `DEFAULT_SEED = 20240101`（`examples/algorithms/__init__.py` 第 62 行）；一律走 `np.random.default_rng`（PCG64）实例，**不使用 `np.random` 全局状态**，所以两次调用同 seed 结果逐位一致。
- **步长与稳定性约定**：定步长显式格式的稳定性由格式自身的稳定域决定，模块不替用户检查：显式 Euler 对 y' = −λy 要求 h < 2/λ，RK4 稳定域更大但同样有限；所有积分器都不做非负裁剪，SIR 的 S 可以被减成负数，需要非负约束得换格式或在模型层面处理。

#### `solve_ivp_euler(f, y0, t_span, h)`

- **数学形式**：y_{n+1} = y_n + Δt · f(tₙ, yₙ)，一阶显式（前向）Euler 格式；局部截断误差 O(Δt²)，整体误差 O(Δt)
- **步骤**：① `as_vector(y0, "y0").copy()` 得到状态 y，并 `_grid(t_span, h)` 生成时间网格 t（末点严格等于 t1）；② `dt = t[1] − t[0]` 取真步长；③ 预分配 `out` 形状 (n+1, m)，`out[0] = y`；④ 循环 i = 0…n−1：`y = y + dt * as_vector(f(t[i], y), "f 返回值")`，写回 `out[i+1]`；⑤ 返回 `(t, out)`
- **复杂度**：时间 O(n·m)（n 为步数，m 为状态维数，每步只调用一次 f）／空间 O(n·m)
- **参数**：`f` —— `f(t, y) -> dy/dt`，y 是一维数组；`y0` —— 长度 m 的一维初值；`t_span` —— `(t0, t1)` 二元组，要求 t1 > t0；`h` —— 建议步长，必须 > 0。非法输入：t1 ≤ t0 或 h ≤ 0 时由 `_grid` 抛 `ValueError`；`y0` 为空、含 NaN/inf 时 `as_vector` 抛 `ValueError`；f 的返回值形状与 y 不一致时按广播/报错的 numpy 语义走。本函数**没有** `rtol`/`atol`（自适应）参数、没有事件检测、没有稠密输出、也不返回误差估计——要自适应步长用 `solve_ivp_rk45`，要非负/稳定性改善用 `implicit_euler`。
- **陷阱**：① **只有一阶精度**，h 减半误差只减半；docstring 明说算传染病峰值时 h = 0.01 才勉强够用，默认拿它当"够准"是常见误判。② **数值不稳定**：对 y' = −100y 这类快变问题，h > 2/100 时解会振荡发散，这是显式 Euler 的稳定域限制不是 bug。③ **状态变量可能变成负数**（如 SIR 的 S 被减成负数），本函数不做裁剪。④ `h` 只是建议值，真步长 ≤ h，所以"我设了 h 但报出的 dt 不一样"是设计行为不是错误。
- **怎么检验**：① 对 y' = −y、y(0) = 1 与闭式解 e^{−t} 对比，误差应随 Δt 线性下降；② 与 `solve_ivp_rk4` 在同一网格上对比，Euler 误差应明显大一个量级；③ 用 `estimate_convergence_order` 量实测阶数，应接近 1；④ 取 h 与 h/2 看误差比 ≈ 2；⑤ h > 2/λ 时观察振荡发散，确认稳定域结论。

#### `solve_ivp_rk4(f, y0, t_span, h)`

- **数学形式**：k₁ = f(t, y)，k₂ = f(t + Δt/2, y + Δt·k₁/2)，k₃ = f(t + Δt/2, y + Δt·k₂/2)，k₄ = f(t + Δt, y + Δt·k₃)，y_{n+1} = y + Δt(k₁ + 2k₂ + 2k₃ + k₄)/6；四阶精度，整体误差 O(Δt⁴)
- **步骤**：① `as_vector(y0, "y0").copy()` 与 `_grid(t_span, h)` 同上；② 取 dt = t[1] − t[0]；③ 预分配 out 并写入初值；④ 每个时间步依次求 k₁…k₄（**四次**右端求值，注意 k₂/k₃ 用半步时刻、k₄ 用整步）再组合；⑤ 返回 `(t, out)`
- **复杂度**：时间 O(4·n·m)（每步四次右端求值）／空间 O(n·m)
- **参数**：`f` / `y0` / `t_span` / `h` 语义与校验完全同 `solve_ivp_euler`；`h` 是建议步长，实际 dt ≤ h。本函数**没有**自适应步长、误差控制、事件检测、非负约束，也**不接受**参数化 rhs——要传参数请用 `sir_rhs` / `seir_rhs` / `lotka_volterra_rhs` 这类工厂先造出 `f`，或在闭包里捕获参数。要自适应请用 `solve_ivp_rk45`。
- **陷阱**：① 四次求值的代价换四阶精度，步长减半误差约降 1/16（docstring 指向 `estimate_convergence_order` 的实测值），但**定步长**在解变化剧烈的区间（如 SIR 爆发期）仍会失准，必要时改自适应步长。② 步长比特征时间尺度大很多时，RK4 同样给出看起来"平滑但错误"的解，务必做步长收敛测试（h、h/2、h/4 结果应基本重合）。③ 不要拿 RK4 解刚性方程，稳定域有限。④ 峰值/拐点位置被网格分辨率限制，别报不必要的小数位。
- **怎么检验**：① logistic 方程对照闭式解 y(t) = K / (1 + (K/y0 − 1)e^{−rt})（`_self_test` 就是这么做的），误差应远小于 Euler；② `estimate_convergence_order` 量实测阶数应接近 4；③ 取 h、h/2、h/4 三次积分并比较解的最大差，做步长收敛测试；④ 对可用 `solve_ivp_rk45` 的同一问题对比，两者在 h 足够小时应基本重合。

#### `sir_rhs(beta, gamma)`

- **数学形式**：状态 y = [S, I, R]；dS/dt = −βSI/N，dI/dt = βSI/N − γI，dR/dt = γI，其中 N = S + I + R 由状态向量自身求和得到（守恒量不显式传入）。Kermack–McKendrick（1927）标准 SIR 方程组
- **步骤**：① `b = float(beta)`、`g = float(gamma)`，若任一 < 0 抛 `ValueError`；② 定义内层 `f(t, y)`：取 `s, i, r = as_vector(y, "y")[:3]`，算 `n = s + i + r`；③ 若 n ≤ 0 直接返回 `np.zeros(3)`（safe 保护）；④ 否则算 `new_inf = b*s*i/n`，返回 `[−new_inf, new_inf − g*i, g*i]`；⑤ 返回这个 callable
- **复杂度**：时间 O(1)（每次求值常数代价）／空间 O(1)
- **参数**：`beta` —— 传染率（单位时间每个感染者有效接触并传染的人数比例）；`gamma` —— 恢复率，1/γ 是平均感染期。两者都必须非负，否则抛 `ValueError`（消息形如 `beta 与 gamma 必须非负，得到 beta=..., gamma=...`）。本函数**没有** N 参数（N 从状态求和）、**没有**出生/死亡/潜伏期项、**没有**隔离或疫苗项、也**没有**时变 beta；要潜伏期请用 `seir_rhs`，要时变/干预只能在 beta 上加时变项或自行写新的 rhs。
- **陷阱**：① 分母用当前状态的 S+I+R，如果状态被数值误差推到负值，N 会变小甚至为 0；`safe` 保护（n ≤ 0 返回零向量）只能挡住 0，**挡不住物理上无意义的负值**。② γ = 0 时模型退化为纯 SI（无人恢复），感染单调增到全人口，这时"峰值时间"永远落在终点附近——不是程序出错。③ 假设人口封闭、无出生死亡、无潜伏期；建模隔离/疫苗需扩展为 SEIR 或在 β 上加时变项。④ 只取 `y[:3]`，传长于 3 的状态向量时多余分量被静默忽略。
- **怎么检验**：① 断言 dS/dt + dI/dt + dR/dt = 0（对任意状态逐点求和应为 0，浮点意义下约 1e-16），这是守恒量的直接验证；② 用 `solve_ivp_rk4` 配上它积分后，检查 S + I + R 沿轨迹恒定（`_self_test` 的守恒量断言）；③ 与 `simulate_sir` 内部使用的同一个 rhs 对比，确认峰值/终局规模一致；④ 极端检查：n = 0 时返回零向量不抛异常，负 β/负 γ 抛 `ValueError`。

#### `simulate_sir(S0, I0, R0, beta, gamma, T, h=0.01)`

- **数学形式**：同 `sir_rhs` 的三式 SIR 方程组，初值 (S0, I0, R0)，在 [0, T] 上以 RK4 数值积分；输出峰值与最终规模：peak_infected = maxₙ I(tₙ)，peak_time = 首个取到该峰值的 t，final_size = R(T)/N，N = S0 + I0 + R0
- **步骤**：① 校验 S0/I0/R0 非负（否则抛 `ValueError`），算 n_total = S0 + I0 + R0 并校验 > 0；② `f = sir_rhs(beta, gamma)` 造右端；③ `solve_ivp_rk4(f, np.array([S0, I0, R0]), (0.0, T), h)` 积分；④ 拆出 S/I/R 三条轨迹，`peak_idx = int(np.argmax(i_arr))`；⑤ 返回 dict：`t`、`S`、`I`、`R`、`peak_infected`、`peak_time`、`final_size`
- **复杂度**：时间 O(T/h)（体现为 O(4·n) 次 rhs 调用）／空间 O(T/h)
- **参数**：`S0, I0, R0` —— 初始易感/感染/移除人数，非负且总数 > 0；`beta` —— 传染率；`gamma` —— 恢复率；`T` —— 模拟时长（从 t = 0 积到 t = T）；`h` —— 步长，**默认 0.01**（时间单位通常是"天"，所以这是 0.01 天）。本函数**没有** seed（确定性，无随机成分）、没有 `rtol`/`atol`、没有干预时点参数、也不返回 R0——要基本再生数请用 `basic_reproduction_number`，要参数拟合请用 `fit_sir_least_squares`，要潜伏期请用 `simulate_seir`。
- **陷阱**：① **峰值是网格上的峰**：h = 0.01 时峰值时间有约 ±0.01 的分辨率误差，论文报"第 23.4 天达峰"时不要给不必要的小数位。② `final_size` 是 **T 时刻**的 R/N，不是真正的 R(∞)/N；T 太小时会明显低估，若需要极限值应解超越方程 R∞ = 1 − exp(−(β/γ)·R∞)（模块自测里就这么做）。③ 参数含义依赖时间单位：β = 0.3 是"每天"还是"每 0.1 天"会让 R₀ 差 10 倍，必须和 T、h 保持一致。④ 峰值为 0 时 `np.argmax` 落在下标 0，`peak_time` 会是 0（单调下降的疫情，属预期行为）。
- **怎么检验**：① 守恒量：`sir_rhs` 的三式相加恒为 0，所以任何积分器下 S + I + R 都应沿轨迹恒定——这是可以自己独立断言的第一性检查（注意 `_self_test` 本身**没有**写这条断言，需自行补验）；② 闭式最终规模：自测用 200 次不动点迭代解超越方程 R∞ = 1 − exp(−3·R∞)（对应 β/γ = 3）得到 `sir_final_size_inf`，与 T = 160 天积分的 `final_size` 对比（`examples/algorithms_golden.json` 里冻结的基准值为 `sir_final_size` = 0.940516、`sir_final_size_inf` = 0.94048，两者相差约 4e-5，可直接复核）；③ 基准算例：`simulate_sir(999, 1, 0, 0.3, 0.1, 160, 0.01)` 的 `sir_peak` = 300.796、`sir_peak_time` = 38.36（golden 文件冻结值，自测返回时四舍五入到 4 位）；④ 与 `scipy.integrate.solve_ivp`（如 RK45）对拍 S/I/R 轨迹；⑤ 步长收敛：h、h/2、h/4 三组的 `peak_infected` 与 `peak_time` 应基本重合；⑥ 边界：I0 = 0 时 I 恒为 0、S0+I0+R0 = 0 时抛 `ValueError`、任一初值为负抛 `ValueError`。

#### `logistic_growth(r, K, y0, T, h=0.01)`

- **数学形式**：y' = r·y·(1 − y/K)，闭式解 y(t) = K / (1 + (K/y0 − 1)e^{−rt})；数值解用 RK4 积分，返回 `(t, y)`
- **步骤**：① 校验 K > 0（否则抛 `ValueError`）、y0 > 0（否则抛 `ValueError`）；② 定义内层 `f(t, y)`：取单分量 `val = float(as_vector(y, "y")[0])`，返回 `[r·val·(1 − val/K)]`；③ 用 `solve_ivp_rk4(f, np.array([y0]), (0.0, T), h)` 积分；④ 返回 `(t, y[:, 0])`——**单变量已降维成一维数组**
- **复杂度**：时间 O(T/h)（每步 4 次单变量 rhs 求值）／空间 O(T/h)
- **参数**：`r` —— 内禀增长率，单位 1/时间；`K` —— 环境容纳量，必须 > 0；`y0` —— 初值，必须 > 0；`T` —— 模拟时长；`h` —— 步长，**默认 0.01**。本函数**没有** seed、没有自适应步长、没有输出误差估计，也**不接受**自定义 rhs——要别的模型请直接用 `solve_ivp_rk4`；要拟合 r/K 请自己外接最小二乘。注意它返回的是元组 `(t, y)` 而不是 `simulate_sir` 那样的 dict。
- **陷阱**：① y0 ≤ 0 时解析解无意义（对数发散），本函数直接抛 `ValueError`，因此**不能**用它验证 y0 = 0 的平凡解。② y0 > K 时解单调下降趋近 K，不会"负增长到 0 以下"；但若用显式 Euler（本函数不用，但你自己拿 `solve_ivp_euler` 复现时）且 h 过大，会数值过冲到负值再跳回 K——这是格式问题不是模型问题。③ 参数 r 的单位是 1/时间，和 T、h 的单位必须一致；r 与 K 的符号/量级错位会让曲线形状完全变样但程序不报错。④ 返回值只含数值解，不含解析解数组，做对照要自己按闭式公式算。
- **怎么检验**：① `_self_test` 就是拿 RK4 数值解与闭式解 y(t) = K/(1 + (K/y0 − 1)e^{−rt}) 对比，断言两者接近，可独立重跑；自测算例为 `logistic_growth(0.4, 100.0, 5.0, 20.0, 0.01)`，返回键 `logistic_y20` 就是 t = 20 处的数值解，`examples/algorithms_golden.json` 冻结值为 **99.3666578**；② 与 `solve_ivp_rk45` 对拍同一问题的轨迹；③ 单调性与极限检验：y0 < K 时 y 单调递增且 lim y(t) = K（取很大的 T 看末端值是否 ≈ K）；④ r < 0 时解应衰减到 0 附近；⑤ 步长收敛：h、h/2 的解之差应远小于 Euler 同网格的差。

#### `lotka_volterra_rhs(alpha, beta, delta, gamma)`

- **数学形式**：状态 y = [x, y]（猎物 x、捕食者 y）；ẋ = αx − βxy，ẏ = δxy − γy。守恒量（首次积分）V = δx − γ ln x + βy − α ln y 沿真解恒定；非平凡平衡点为 (γ/δ, α/β)
- **步骤**：① 四个参数统一 `float` 化：`a, b, d, g = float(alpha), float(beta), float(delta), float(gamma)`；② 定义内层 `f(t, y)`：`x, yy = as_vector(y, "y")[:2]`，返回 `[a·x − b·x·yy, d·x·yy − g·yy]`；③ 返回这个 callable
- **复杂度**：时间 O(1)（每次求值常数代价）／空间 O(1)
- **参数**：`alpha` —— 被捕食者的内禀增长率（x' 中的 +αx）；`beta` —— 捕食导致的被捕食者死亡率系数；`delta` —— 捕食转化为捕食者增长的效率；`gamma` —— 捕食者的自然死亡率。**本实现不做任何校验**（负参数不会被拒绝，会静默改变动力学），也没有 N 上限、没有环境容纳量（不是 logistic 版 LV）、没有 Allee 效应、没有时变参数。需要带容纳量的模型要自己写 rhs。
- **陷阱**：① 该模型**结构不稳定**：闭环轨道只是中性稳定的，RK4 的数值耗散会让振幅缓慢漂移（长期积分后轨道螺旋向内或向外），这不是 bug；长时间模拟应改用辛格式或监控守恒量 V = δx − γ ln x + βy − α ln y。② 数值误差可能把种群推成负值，正反馈后直接爆掉；需要时对状态做截断（本模块所有积分器都不裁剪）。③ 参数 α/γ 与 δ/β 的量纲不同（1/时间 vs 1/(个体·时间)），直接比较大小没有意义，可解释量是平衡点 (γ/δ, α/β)。④ 只取 `y[:2]`，多余状态分量被静默忽略。⑤ **四个参数完全不校验**：传负值不会抛异常（与 `sir_rhs` / `seir_rhs` / `basic_reproduction_number` 都不同），得到的是另一组动力学，论文里参数符号写错不会有任何提示。
- **怎么检验**：① 守恒量检验：用 `solve_ivp_rk4` 以小 h 积分，计算 V(t) 序列，其相对漂移应很小（不是零——这正是"结构不稳定 + 数值耗散"的直接度量）；② 平衡点检验：以 (x, y) = (γ/δ, α/β) 为初值，rhs 应返回约零向量，且积分后状态几乎不动；③ 基准算例：`_self_test` 取 `lotka_volterra_rhs(1.0, 0.1, 0.075, 1.5)`、初值 [10.0, 5.0]、积分到 t = 10（h = 0.01），返回键 `lv_x_at_10`，`examples/algorithms_golden.json` 冻结值为 **8.00658**；④ 与小 h 下的 `solve_ivp_rk45` 对拍轨道；⑤ 用 `jacobian_stability` 在平衡点 (γ/δ, α/β) = (20.0, 10.0) 算 Jacobian，特征值应为纯虚数 ±i√(αγ) = ±i√1.5，`stable` 返回 False（中性稳定）。

#### `fit_sir_least_squares(times, observed_I, N, beta_bounds, gamma_bounds, n_grid=40)`

- **数学形式**：minimize_{β,γ} SSE(β,γ) = Σₖ (Î(tₖ) − I(tₖ))²，其中 Î 是用定步长 RK4 积出的 SIR 现存感染者曲线在观测时刻的线性插值；初值固定为 I(0) = observed_I[0]、S(0) = N − I(0)、R(0) = 0
- **步骤**：① 归一化并校验：`times`/`observed_I` 等长，times 非负且严格递增，N > 0，两个 bounds 满足 lo < hi，n_grid ≥ 2，否则抛 `ValueError`；② 定初值 i0 = max(observed_I[0], 1e-9)、s0 = N − i0（s0 ≤ 0 抛 `ValueError`）；③ 内层网格：`n_steps = max(50, ceil(t_end/0.05))`（拟合内部步长约 0.05），`grid_t = linspace(0, t_end, n_steps+1)`；④ 每轮在当前区间上取 n_grid×n_grid 个 (β, γ) 点，逐个用 `_sir_trajectory` 积分、`np.interp` 取观测时刻的 Î、累计 SSE，记最优点；⑤ 以最优点为中心把区间收缩为原区间的 1/4（b_span = (b_hi−b_lo)/4，区间取 best ± span/2），重复 `refine_rounds = 4` 轮，**共 5 轮**；⑥ 在最优点上重算 SSE，返回 `{"beta": ..., "gamma": ..., "sse": ...}`
- **复杂度**：时间 O(rounds · n_grid² · n_steps · m)（默认约 5 · 1600 次积分）／空间 O(n_steps)
- **参数**：`times` —— 观测时刻一维数组，要求非负、严格递增；`observed_I` —— 对应时刻的**现存感染者数**（不是累计），长度 m；`N` —— 总人口；`beta_bounds`/`gamma_bounds` —— `(lo, hi)` 搜索区间，必须 lo < hi；`n_grid` —— 每轮每参数网格点数，**默认 40**，必须 ≥ 2。本函数**没有** `seed`（完全确定性，`_sir_trajectory` 是纯标量 RK4，不走 `_common.rng`）、**没有**权重参数（等权最小二乘）、**没有**置信区间/协方差输出、也**不拟合** S/R 或累计量——要区间得自己重抽样（bootstrap）或画 SSE 等高线；要别的模型参数拟合请自行照此结构改写。
- **陷阱**：① **SSE 曲面有强相关脊**：β 与 γ 只通过 R₀ = β/γ 和绝对水平部分耦合，数据只覆盖早期上升段时两参数几乎不可辨识（多个组合 SSE 接近）；必须报告参数的置信区间或 SSE 等高线，不能只报一个点估计。② 只拟合 I(t)，**不拟合累计量**；若数据是累计确诊，要先把观测换成 I = 累计 − 累计(t−1) 或直接对累计量建残差，否则参数会被系统性低估。③ 初值固定为 I(0) = observed_I[0]，若第一个观测本身有报告延迟，β 会被带偏。④ 网格搜索只保证找到"网格分辨率下的最优"，不做导数，**不保证全局最优**。⑤ 内部步长固定约 0.05 且拟合区间固定为 [0, times[-1]]，观测间隔远小于 0.05 时插值误差会主导 SSE。⑥ bounds 只校验 lo < hi，**不检查真值是否在区间内**；真值落在边界外时最优点会被压在网格边界上，返回的是一个"贴边"的解而不会报任何警告。
- **怎么检验**：① `_self_test` 的合成数据自证：真参数 (β*, γ*) = (0.25, 0.1)，用 `_sir_trajectory` 生成 0~45 天、每 2 天一点的**无噪声**观测（N = 1000），在 bounds (0.05, 0.6) / (0.02, 0.4)、`n_grid=16` 下拟合；`examples/algorithms_golden.json` 冻结的结果是 `fit_beta` = 0.25002、`fit_gamma` = 0.100008、`fit_sse` = 0.00737216——真值在 bounds 内、网格够密时应回到真值附近；② 用更密的 `n_grid` 或更多 refine 轮跑一遍，比较 `sse` 是否明显下降——相同说明已达网格分辨率极限；③ 对同一数据画 SSE(β, γ) 等高线，确认"脊"的存在（这是对不可辨识性的直接可视化）；④ 与 `scipy.optimize.least_squares`（外部对照，仓库代码不 import scipy）比较 `sse` 与参数，量级应一致；⑤ **可辨识性的定量演示**（自测陷阱段给出的实测结论）：把观测截断成只有前 15 天上升段，拟合会漂到 β ≈ 0.278、γ ≈ 0.127，此时 β − γ ≈ 0.150 与真值几乎一致但 R₀ = β/γ 被显著低估——这组数字来自模块 docstring 的实测记录，可在同一算例上复现；⑥ 边界输入：times 含重值/递减、bounds 反序、n_grid = 1 都应抛 `ValueError`。

#### `estimate_convergence_order(f, y_exact, y0, t_end, h_list, method="rk4")`

- **数学形式**：误差模型 e(h) ≈ C·h^p；取对数 ln e = ln C + p·ln h，对 (ln hᵢ, ln eᵢ) 做最小二乘直线拟合，**回归斜率本身就是阶数 p**（源码注释原文：`log e ≈ log C + p log h，所以回归斜率本身就是阶数 p`）。误差用终点欧氏范数 e(h) = ‖y_h(t_end) − y_exact(t_end)‖₂
- **步骤**：① 把 h_list 转 float 并校验：至少两个步长、全部 > 0，否则抛 `ValueError`；② `solver = solve_ivp_rk4 if method == "rk4" else solve_ivp_euler`，并校验 method ∈ {"rk4", "euler"}（注意校验在用 `if/else` 选 solver **之后**，见"陷阱"）；③ 解析 y_exact：可调用则取 `as_vector(y_exact(t_end))`，否则 `as_vector(y_exact)` 当终点精确值；④ 对每个 h 用所选格式从 (0, t_end) 积分，算终点误差 errs[i]；⑤ 返回 `np.polyfit(log_h, log_e, 1)[0]`，即一次多项式拟合的**斜率**
- **复杂度**：时间 O(Σ 1/hᵢ)（每次积分按最细网格计）／空间 O(max 1/hᵢ)
- **参数**：`f` —— 右端函数；`y_exact` —— 精确解，**两种口径**：`callable(t) -> y`（只会在 t_end 处被调用一次）或**终点处**的精确值数组；`y0` —— 初值；`t_end` —— 积分终点（**总是从 t = 0 开始**，无法改起点）；`h_list` —— 至少两个步长，建议按 h, h/2, h/4 递减；`method` —— `"rk4"`（默认）或 `"euler"`。本函数**没有** `seed`、没有 `rtol`/`atol`、不返回每次的误差明细（只返回一个 float）、也不支持自定义格式——要测别的格式得自己复制这段回归逻辑。返回**裸 float**，不是 dict。
- **陷阱**：① h 太大时误差进入"非渐近区"，测出来的阶数会偏离理论值（RK4 甚至可能测出 3 或 5）；h 太小时舍入误差占主导，阶数会虚高甚至变负。请用 2~3 个数量级内的 h。② 必须用**同一台机器、同一函数**比较；换用不同精确解口径（例如把插值值当精确值）会系统性偏移结果。③ 多个状态变量时用欧氏范数合成误差，**量纲不同的分量会互相影响**，必要时先无量纲化。④ 误差为 0 时 `np.log(0)` 会给 −inf 并让 polyfit 返回 NaN/警告——精确解与数值解在机器精度内重合时就会出现。⑤ 非法 `method` 传入时，`solver` 先被赋成 `solve_ivp_euler` 再抛 `ValueError`——最终仍会报错，但错误顺序与"先校验后分派"的直觉不同。⑥ **docstring 的公式与返回值符号不一致**：docstring 的「算法」段写的是"斜率即 −p，返回 p = −slope"，而代码注释与实现都是**直接返回斜率**（`np.polyfit(log_h, log_e, 1)[0]`）；对 e ≈ Ch^p 而言正确的是后者的正号，实测值 4.0693 / 1.0035 也印证了这一点——引用文档里的公式时不要照抄那个负号。
- **怎么检验**：① `_self_test` 用 y' = −y、y(0) = 1、终点 t = 1 做标准算例，精确值 e⁻¹：`method="rk4"` 传 h_list = [0.2, 0.1, 0.05, 0.025]，`method="euler"` 传 [0.02, 0.01, 0.005, 0.0025]，返回键 `rk4_order` / `euler_order`；`examples/algorithms_golden.json` 冻结的实测值为 **4.0693** / **1.0035**，这两个数可以直接用来判断自己的实现有没有退化；② 用 `logistic_growth` 的闭式解做被积模型，重跑同样实验；③ 把 h_list 换成超出渐近区的粗步长（如 1.0、0.5）看阶数偏移，验证"非渐近区"警告；④ 只传一个步长或传负步长应抛 `ValueError`；⑤ 与手工计算比较：对两个步长直接算 p = log(e(h₁)/e(h₂))/log(h₁/h₂)，应与回归结果接近；⑥ 同一算例把 `y_exact` 从"终点值数组"换成"callable(t)"，结果应完全一致（自测用的是数组口径）。

#### `seir_rhs(t, y, beta, sigma, gamma, mu=0.0)`

- **数学形式**：状态 y = [S, E, I, R]；N = S + E + I + R；dS/dt = μN − βSI/N − μS，dE/dt = βSI/N − σE − μE，dI/dt = σE − γI − μI，dR/dt = γI − μR。Hethcote（2000）综述里的出生-死亡版本，μ = 0 时 N 严格守恒
- **步骤**：① 四个参数 float 化，任一 < 0 抛 `ValueError`；② `as_vector(y, "y")` 并强校验长度必须为 4，否则抛 `ValueError`；③ 展开 s/e/i/r，算 N = S+E+I+R；④ N ≤ 0 时返回 `np.zeros(4)`；⑤ 算 `new_inf = βSI/N`，返回四元组 `[μN − new_inf − μS, new_inf − σE − μE, σE − γI − μI, γI − μR]`
- **复杂度**：时间 O(1)／空间 O(1)
- **参数**：`t` —— 当前时刻，本模型自治、**t 不参与计算**，保留它是为了直接匹配 `f(t, y)` 调用口径；`y` —— `[S, E, I, R]`，长度必须恰为 4；`beta` —— 传染率；`sigma` —— 潜伏期出率，1/σ 是平均潜伏期（E→I）；`gamma` —— 恢复率，1/γ 是平均感染期；`mu` —— 自然出生/死亡率，默认 **0.0**（封闭人口）。本函数**没有** N 参数（从状态求和）、**没有**时变参数、**没有**年龄结构；要基本再生数请用 `basic_reproduction_number`，要一键积分请用 `simulate_seir`。
- **陷阱**：① 与 `sir_rhs` 的签名**不同**：`sir_rhs` 是"先给参数、返回闭包"，本函数是直接的 `f(t, y)` 右端项（参数放在 t/y 之后）。拼装时写 `lambda t, y: seir_rhs(t, y, beta, sigma, gamma, mu)`；**参数顺序写错不会报错，只会给出另一组动力学**。② σ 很大时方程变刚性：E 的时间尺度是 1/σ，显式 RK4 要求 σ·Δt 落在稳定域内（负实轴约 2.8），σ = 1e4 配 Δt = 0.1 会直接爆掉；要逼近 SIR 极限必须同时缩小 Δt（模块自测取 σ = 50、Δt = 1e-3）。③ 分母 N 用当前状态求和，数值误差把状态推成负值后 N 会变小，这里只能挡住 N ≤ 0（返回全 0），挡不住物理上无意义的负值。④ R₀ 口径是 βσ/((σ+μ)(γ+μ))，σ → ∞ 时才退化为 SIR 的 β/(γ+μ)；σ 小时两个口径差别很大，论文必须写清用的是哪一个。⑤ 长度不为 4 直接抛错，比 `sir_rhs` 的"静默取前 3 个"严格。
- **怎么检验**：① 守恒检验：μ = 0 时四式相加为 0（逐点求和 ≈ 0），因此 N 沿轨迹恒定；自测就用 `basic_reproduction_number(0.3, 50.0, 0.1, 0.0)` 断言「μ = 0 时 SEIR 的 R₀ 恒等于 SIR 的 β/γ」，容差 1e-12；② μ > 0 时检查总量行为符合 dN/dt = μN − μN = 0 的构造（总量仍守恒，只是仓室间有补充与流失）；③ 极限检验：把 σ 取得很大（自测用 σ = 50）并同步缩小 Δt（1e-3），SEIR 的 I 轨迹应与同 β、γ 的 SIR 的 I 轨迹接近；④ 与 `basic_reproduction_number` 联测：σ = μ = 0 时 R₀ 应为 0；⑤ 边界：y 长度为 3 或 5、任一参数为负都应抛 `ValueError`。

#### `simulate_seir(y0, beta, sigma, gamma, mu=0.0, t_end=100.0, dt=0.1, method="rk4")`

- **数学形式**：同 `seir_rhs` 的四式方程组，初值 y0 = [S0, E0, I0, R0]，在 [0, t_end] 上以定步长 RK4（或 Euler）积分；输出 peak_I = maxₙ I(tₙ)、peak_time = 首个取到峰值的 t、final_size = R(t_end)/N0，N0 = S0+E0+I0+R0
- **步骤**：① `as_vector(y0).copy()` 并强校验长度 4、各分量非负、总和 > 0，任一不满足抛 `ValueError`；② 校验 t_end > 0、dt > 0、method ∈ {"rk4", "euler"}，否则抛 `ValueError`；③ 定义 `f(t, y) = seir_rhs(t, y, beta, sigma, gamma, mu)`（此处就是把参数顺序**固定正确**的地方）；④ `solver(f, states, (0.0, t_end), dt)` 积分；⑤ 拆四条轨迹，`np.argmax(i_arr)` 定位峰值；⑥ 返回 dict：`t`、`S`、`E`、`I`、`R`、`peak_I`、`peak_time`、`final_size`
- **复杂度**：时间 O(t_end/dt)（体现为 4 倍 rhs 调用）／空间 O(t_end/dt)
- **参数**：`y0` —— `[S0, E0, I0, R0]` 长度 4，非负且总和 > 0；`beta`/`sigma`/`gamma` —— 同 `seir_rhs`；`mu` —— 自然出生/死亡率，默认 **0.0**；`t_end` —— 模拟时长，默认 **100.0**；`dt` —— **建议**步长，默认 **0.1**，实际步长会被微调成 t_end/ceil(t_end/dt) 以保证末点正好落在 t_end（与 `solve_ivp_euler` 口径一致）；`method` —— `"rk4"`（默认，四阶）或 `"euler"`（一阶，仅用于演示误差）。本函数**没有** seed（确定性）、没有自适应步长、不做非负裁剪、不返回 S/E/I/R 之外的派生量——要 R₀ 用 `basic_reproduction_number`，要更细的峰值精度就自己减小 dt 或直接调 `solve_ivp_rk45`。
- **陷阱**：① **峰值是网格上的峰**：dt 决定峰值时间的分辨率，dt = 0.1 时不要报"第 38.55 天"。② `final_size` 是 **t_end 时刻**的 R/N0，不是 R(∞)/N0；t_end 太小时会低估。③ dt 与 σ 必须配套：显式格式要求 σ·dt 在稳定域内，否则解会指数爆炸而**不报任何错误**（详见 `seir_rhs` 陷阱 2）。④ **不做非负裁剪**：Euler 大步长可能把某个仓室算成负数，后续 N 变小会让 βSI/N 被高估，出现"越算越大"的假象。⑤ 返回键名与 `simulate_sir` 不完全一致：这里峰值叫 `peak_I`，而 SIR 那边叫 `peak_infected`，跨模型写表格时别混用。
- **怎么检验**：① μ = 0 时断言 S + E + I + R 沿轨迹恒定（守恒量）；② 与 σ → ∞ 的 SIR 极限对拍：自测取 `simulate_seir([999, 0, 1, 0], 0.3, 50.0, 0.1, 0.0, 80.0, 1e-3)` 与 `simulate_sir(999, 1, 0, 0.3, 0.1, 80.0, 5e-3)` 比较，`peak_I` 与 `peak_infected` 的相对差必须 < 5%，否则抛 `AssertionError`（模块自测用的正是 σ = 50、dt = 1e-3 这组）；③ 步长收敛：dt、dt/2、dt/4 三组的 `peak_I` 与 `peak_time` 应基本重合；④ `method="euler"` 与 `"rk4"` 在细网格上应收敛到同一曲线，粗网格上 Euler 明显偏；⑤ 与 `scipy.integrate.solve_ivp`（外部对照）对拍轨迹；⑥ 边界：y0 长度为 3、含负分量、总和为 0 都抛 `ValueError`。

#### `basic_reproduction_number(beta, sigma, gamma, mu=0.0)`

- **数学形式**：R₀ = β·σ / ((σ+μ)(γ+μ))。由下一代矩阵法对 `seir_rhs` 的感染仓室 [E, I] 求谱半径：K = [[βS/N, βS/N], [0, 0]]，V = [[σ+μ, 0], [−σ, γ+μ]]，在 S = N 的初始时刻归一。恒有 R₀_seir = (σ/(σ+μ))·(β/(γ+μ)) ≤ β/(γ+μ)
- **步骤**：① 四参数 float 化，任一 < 0 抛 `ValueError`；② 算 `denom_g = γ + μ`，若 ≤ 0 抛 `ValueError`；③ 若 σ + μ ≤ 0（即 σ = μ = 0）**直接返回 0.0**——没人能从 E 转为 I，疫情传不起来；④ 否则返回 `β·σ / ((σ+μ)·denom_g)`
- **复杂度**：时间 O(1)／空间 O(1)
- **参数**：`beta` —— 传染率；`sigma` —— 潜伏期出率（E→I 的速率），1/σ 是平均潜伏期；`gamma` —— 恢复率；`mu` —— 自然出生/死亡率（与 `seir_rhs` 同口径），默认 **0.0**。四个参数都要求非负；γ + μ ≤ 0 时抛 `ValueError`。本函数**没有** N 参数（口径在 S = N 处取值，与人口规模无关）、**没有** S(t) 参数（因此算不了有效再生数 Rₜ）、也不是 SIR 口径——要 SIR 的 R₀ = β/γ 请自己算或把 σ 取得很大。
- **陷阱**：① 两个口径在 σ 很大时几乎相等，但 σ 与 γ 同量级时能差一倍以上；用哪个口径必须写进论文，否则复现者算出的"R₀ = 3"可能是另一个模型。② σ = 0 时本函数返回 **0**（无人从 E 转为 I，疫情传不起来），而 β/(γ+μ) 口径会给出一个正数——这不是 bug，是口径差异。③ 这里的 R₀ 是**初始时刻 S = N** 的值；流行过程中的有效再生数 Rₜ = R₀·S(t)/N 才是控制阈值（本函数不算 Rₜ）。④ 返回**裸 float**，不是 dict；别按 `basic_reproduction_number(...)["R0"]` 去取。
- **怎么检验**：① 解析式直算对照：手算 βσ/((σ+μ)(γ+μ)) 与返回值逐位比较；② 与 `seir_rhs` 的下一代矩阵（NGM）联测：无病平衡点取 (S, E, I, R) = (N, 0, 0, 0)，感染仓室 [E, I] 的 K = [[βS/N, βS/N], [0, 0]]、V = [[σ+μ, 0], [−σ, γ+μ]]，谱半径 ρ(KV⁻¹) 应等于本函数的返回值；③ 阈值行为检验：R₀ < 1 时 `simulate_seir` 的 I 单调衰减到 0，R₀ > 1 时 I 先升后降；`_self_test` 用 `simulate_seir([999, 0, 1, 0], 0.05, 1.0, 0.1, 0.0, 60.0, 0.01)` 断言其 `final_size ≤ 0.01`（这组参数算出的 R₀ = 0.05/0.1 = 0.5 < 1，自测先用 `basic_reproduction_number` 断言 `sub_r0 < 1.0`）；④ 极限检验：`basic_reproduction_number(0.3, 1e6, 0.1, 0.05)` 应逼近 β/(γ+μ) = 0.3/0.15（自测容差 **1e-4**）；自测还断言 R₀ 关于 σ **单调递增**（σ = 1 的值 < σ = 5 的值），以及 μ = 0 时 `basic_reproduction_number(0.3, 50.0, 0.1, 0.0)` **恒等于** β/γ = 3.0（容差 **1e-12**）；⑤ 边界：σ = μ = 0 返回 0.0、γ = μ = 0 抛 `ValueError`、任一参数为负抛 `ValueError`。

#### `implicit_euler(rhs, y0, t_end, dt, newton_tol=1e-10, max_iter=50)`

- **数学形式**：后向（隐式）Euler：y_{n+1} = yₙ + h·f(t_{n+1}, y_{n+1})。每步解非线性方程 G(y) = y − yₙ − h·f(t_{n+1}, y) = 0，牛顿迭代用 J_G = I − h·J_f（J_f 由中心差分数值求得），更新 y ← y + solve(J_G, −G(y))
- **步骤**：① `as_vector(y0).copy()`；② 校验 newton_tol > 0、max_iter ≥ 1，否则抛 `ValueError`；③ `t = _grid((0, t_end), dt)` 取网格，真步长 h = t[1] − t[0]；④ 每步先用**显式 Euler 预测子** `y_new = y_old + h·rhs(t[i], y_old)` 加速收敛；⑤ 牛顿循环至多 max_iter 次：算残差 `resid = y_new − y_old − h·rhs(t_next, y_new)`，若 `max|resid| < newton_tol` 判定收敛并跳出，否则解线性系统得修正量 delta、更新 y_new，若 `max|delta| < newton_tol` 也判定收敛；⑥ 若 max_iter 次内未收敛，**抛 `ValueError`**（消息给出 t、max_iter、newton_tol，并提示减小 dt 或放宽容差），不静默返回错误结果；⑦ 收敛则写回网格并继续；⑧ 返回 `{"t": t, "y": out}`
- **复杂度**：时间 O(n · max_iter · m²)（每步 m 次额外右端求值构造数值 Jacobian + 一个 m×m 线性方程组）／空间 O(n·m + m²)
- **参数**：`rhs` —— `f(t, y) -> dy/dt`（注意形参名叫 `rhs` 而不是 `f`）；`y0` —— 长度 m 的一维初值；`t_end` —— 积分终点（**从 t = 0 开始**）；`dt` —— 建议步长，按模块约定由 `_grid` 微调以正好落在 t_end；`newton_tol` —— 牛顿收敛容差，默认 **1e-10**，用**残差或修正量的无穷范数**判定，必须 > 0；`max_iter` —— 每步最大牛顿迭代次数，默认 **50**，必须 ≥ 1。本函数**没有** `t_span` 二元组（只给终点）、没有自适应步长、没有解析 Jacobian 入口（想看数值 Jacobian 请用 `jacobian_stability`）、也不返回每步的牛顿迭代次数。要显式格式请用 `solve_ivp_euler` / `solve_ivp_rk4` / `solve_ivp_rk45`。
- **陷阱**：① **无条件稳定不等于无条件准确**：y' = −20y、h = 0.25 时隐式 Euler 有界且收敛，但单步误差仍是 O(h)，它只是"不发散"，精度提升要靠减小 h。② 线性问题牛顿一步就收敛；**强非线性**问题（如指数增长项）可能不收敛或跳到无意义的解，此时应减小 dt 或改用阻尼牛顿。③ 数值雅可比的默认 eps = 1e-6 是绝对量级；状态分量极小或极大时差分会被舍入误差污染（`_num_jacobian` 只做了 max(1, |y_j|) 的粗缩放），必要时手写解析雅可比。④ 步长仍由 `_grid` 微调，最后一步可能比 dt 小很多，此时该步截断误差也更小，不会破坏整体 O(h) 的结论。⑤ rhs 返回值长度与状态维数不一致时抛 `ValueError`（在 `rhs_at` 里查）。
- **怎么检验**：① 对 y' = −y 有**闭式递推** yₙ = (1/(1+h))ⁿ：h = 0.01、100 步的闭式终点是 (1/1.01)¹⁰⁰，`_self_test` 断言 `implicit_euler` 的终点与之相差 < **1e-9**（返回键 `implicit_y1` / `implicit_y1_closed`）；② **刚性对照**：y' = −20y、h = 0.25 时闭式终点是 (1/6)⁴，自测断言隐式解与之相差 < **1e-12** 且 |y| ≤ 1（有界），同时断言**显式** Euler 在同一算例上 |y| > 1（发散，返回键 `explicit_stiff_y1`）——这一对断言把"无条件稳定 vs 有限稳定域"钉死；③ 与 `solve_ivp_euler` 在 h → 0 时对比，两者应同阶收敛到同一解（阶数 ≈ 1）；④ 线性 ODE 上检查牛顿 1 步内收敛（可以把 max_iter 设成 1 看是否仍成功）；⑤ 构造牛顿不收敛的算例（大 dt + 强非线性），确认抛 `ValueError` 而不是返回垃圾。

#### `solve_ivp_rk45(rhs, y0, t_end, rtol=1e-6, atol=1e-9, dt_init=None)`

- **数学形式**：自适应步长 Dormand–Prince RK45（5(4) 嵌入对）：7 级 k₁…k₇，五阶解 y5 = y + h Σ b5ᵢkᵢ，嵌入四阶解 y4 = y + h Σ b4ᵢkᵢ，误差估计 err = y5 − y4。加权 RMS 范数 ‖err / (atol + rtol·max(|y|, |y5|))‖₂/√m ≤ 1 则接受该步
- **步骤**：① `as_vector(y0).copy()`，校验 t_end > 0、rtol > 0、atol ≥ 0，否则抛 `ValueError`；② 取 Dormand–Prince 的 c/a/b5/b4 系数表；③ 初值步长 `h = t_end/100`（`dt_init` 给定时用它），校验 h > 0 且有限（否则抛 `ValueError`），并 `h = min(h, t_end)`；④ 主循环 while t_cur < t_end：先把 h 裁剪到不超过剩余区间；⑤ 求 k₁（校验返回长度 = m），再用 a_stage 组合出 k₂…k₇（**共 7 次**右端求值）；⑥ 算 y5、y4 与 err_norm；⑦ err_norm ≤ 1 则接受：推进 t、n_steps += 1、记录点，步长因子 `min(5, max(0.2, 0.9·err_norm^{−1/5}))`（err_norm ≤ 0 时直接取 5.0）；否则 n_rejected += 1，因子 `max(0.1, 0.9·err_norm^{−1/5})` 重试；⑧ 两道护栏：`n_steps + n_rejected ≥ 2 000 000` 或建议步长 `h < 1e-14·max(1, t_end)` 时抛 `ValueError`（提示奇异点或容差过严），避免死循环；⑨ 返回 dict：`t`（**被接受的**时间点，非均匀，末点严格等于 t_end）、`y`、`n_steps`、`n_rejected`
- **复杂度**：时间 O(7 · n_steps · m)／空间 O(n_steps · m)
- **参数**：`rhs` —— `f(t, y) -> dy/dt`；`y0` —— 长度 m 的初值；`t_end` —— 积分终点（**从 t = 0 开始**，必须 > 0）；`rtol` —— 相对容差，默认 **1e-6**，作用在 RMS 加权误差范数上，必须 > 0；`atol` —— 绝对容差，默认 **1e-9**，状态接近 0 时由它接管，必须 ≥ 0；`dt_init` —— 初始步长，默认 **None** 表示取 `t_end/100`，必须 > 0 且有限。本函数**没有** `t_eval` 之类的稠密输出参数、没有事件检测、没有 max_step 参数、也不返回建议的下一步步长——要固定网格输出请自己 `np.interp` 重采样。
- **陷阱**：① **不做稠密输出**：返回的 t 是非均匀的求解器步点，画图/比较前要用 `np.interp` 重采样到你要的时间网格，**别假设它是等距的**。② 误差控制是**局部**的：rtol 限制单步误差，长时间积分后全局误差会累积（通常仍是 O(rtol) 量级，但混沌系统会指数放大）。③ 用 atol 兜住接近 0 的分量；若某个分量自然量级是 1e-12，默认 atol = 1e-9 会把它当"零"而完全不管误差。④ 步数超过 2e6 或建议步长小于 1e-14·t_end 时直接抛 `ValueError` 而不是死循环，这通常意味着方程有奇异点或容差过严。⑤ `dt_init` 会比 `t_end` 大时被静默裁剪为 t_end；⑥ 返回键名是 `n_steps`/`n_rejected`（int），不是数组。
- **怎么检验**：① 与闭式解对拍：`_self_test` 取 y' = y（即 exp(t)）、`solve_ivp_rk45(f, [1.0], 1.0, 1e-6, 1e-9)`，断言终点与 e 的**相对误差 < 1e-5**（返回键 `rk45_rel_error`），并断言末点与 t_end = 1.0 的差 < **1e-12**；② 收紧容差：rtol 从 1e-6 收到 1e-10，`n_steps` 应明显增大而终点误差明显变小（误差控制生效的直接证据）；③ 检查返回的 `t` **不是**等距的（`np.diff(t)` 的标准差 > 0），末点严格等于 t_end；④ 与 `solve_ivp_rk4` 在小 h 下对拍轨迹（先 interp 到同一网格）；⑤ 触发护栏：给一个含奇异点的 rhs（如 y' = 1/(t−1)）或极端 rtol，确认抛 `ValueError` 而不是挂住；⑥ 用 SEIR 的刚性问题（σ 很大）验证它比显式定步长更省步数，并观察 `n_rejected` 是否随之增大。

#### `jacobian_stability(rhs, y_eq, eps=1e-6)`

- **数学形式**：在平衡点 y_eq 处求 Jacobian J[i,j] = ∂fᵢ/∂y_j（数值中心差分），对 J 求特征值 λ₁…λ_m；线性化意义下的稳定判据是 max Re(λᵢ) < 0（Lyapunov 局部渐近稳定）
- **步骤**：① `as_vector(y_eq, "y_eq").copy()`；② 校验 eps > 0，否则抛 `ValueError`；③ 用 `_num_jacobian(rhs, 0.0, y, eps)` 在 **t = 0** 处逐分量中心差分（步长按 `eps·max(1, |y_j|)` 缩放）——每列两次右端求值；④ `np.linalg.eigvals(jac)` 求复特征值；⑤ `stable = bool(np.all(np.real(eig) < 0.0))`；⑥ 返回 `{"jacobian": jac, "eigenvalues": eig, "stable": stable}`
- **复杂度**：时间 O(m²)（含 2m 次右端求值）／空间 O(m²)
- **参数**：`rhs` —— `f(t, y) -> dy/dt`；`y_eq` —— 平衡点（或任意工作点）状态向量，长度 m；`eps` —— 中心差分步长，默认 **1e-6**，必须 > 0。本函数**没有**时刻参数（固定取 t = 0）、没有解析 Jacobian 入口、没有条件数输出、也不区分中心/鞍点/焦点——它只回答 `stable` 这个布尔问题；要更细的稳定性分类请自己看返回的 `eigenvalues` 实部/虚部。
- **陷阱**：① 这里固定取 **t = 0** 求 Jacobian：**只对自治系统有意义**。非自治系统请把时刻烘焙进闭包（`lambda t, y: f(t0, y)`），否则得到的是 t = 0 处的"冻结"矩阵。② 判据是**线性化**结论：实部恰好为 0（中心/临界情形）时返回 False，此时稳定性由非线性项决定，本函数给出的结论是"**不能判定为稳定**"（例如 `lotka_volterra_rhs` 在平衡点处特征值为纯虚数，`stable` 必为 False，而轨道其实是中性稳定的）。③ 特征值对 eps 敏感：eps 太大引入截断误差，太小被舍入误差淹没；对病态雅可比（条件数极大）做尺度缩放后再解释结果。④ 返回的 `eigenvalues` 是**复数组**，写进 JSON 前要自己取 `.real/.imag`。⑤ rhs 返回值长度与状态维数不一致时 `_num_jacobian` 抛 `ValueError`。
- **怎么检验**：① `_self_test` 用 logistic 的 f(t, y) = 0.4y(1 − y/100) 做基准：在 **K = 100** 处断言 `stable == True` 且最大特征值实部 = −r = **−0.4**（容差 **1e-6**，返回键 `jac_logistic_max_real_at_K`），在 **0** 处断言 `stable == False`——这是"稳定/不稳定"两侧的定向验证，可独立重跑；② 解析对照：线性系统 ẏ = Ay（rhs 写成 `lambda t, y: A @ y`）的 Jacobian 应逐元素等于 A（eps 合理时约到 1e-6 量级），`np.linalg.eigvals(A)` 与返回的 `eigenvalues` 应一致（顺序不保证）；③ 临界情形：对 `lotka_volterra_rhs` 的平衡点 (γ/δ, α/β) 检查特征值实部 ≈ 0（`stable` 为 False，复现"中心"的结论）；④ eps 敏感性：eps 取 1e-3、1e-6、1e-9 三档，看 `stable` 的结论是否翻转（翻转意味着该平衡点接近临界或雅可比病态）。

#### `logistic_map(r, x0=0.4, n_steps=200, n_transient=100)`

- **数学形式**：离散 logistic 映射 x_{n+1} = r·xₙ(1 − xₙ)；Lyapunov 指数 λ = (1/n)Σ ln|f'(xₙ)| = (1/n)Σ ln|r(1 − 2xₙ)|；周期 p 定义为尾巴上满足 max|xᵢ − x_{i−p}| < 1e-6 的最小 p
- **步骤**：① 校验 r 有限且 ≥ 0、x0 ∈ (0, 1)、n_steps ≥ 2、n_transient ≥ 0，任一不满足抛 `ValueError`；② 先迭代 `n_transient` 次丢弃暂态；③ 记录 `n_steps` 个点：每步存 xs[i]，累加 `ln|r(1−2x)|`（导数下限钳到 **1e-300** 避免 log(0) = −inf），再迭代一次；④ `lyapunov = lyap_sum / n_steps`；⑤ 取尾巴 `tail = xs[-min(n_steps, 100):]`，对 p = 1…len(tail)//3 检查 `max|tail[p:] − tail[:-p]| < 1e-6`，第一个满足的 p 即 `period`，都不满足为 0；⑥ 返回 `{"x": xs, "lyapunov": ..., "period": ...}`
- **复杂度**：时间 O(n_steps + n_transient)／空间 O(n_steps)
- **参数**：`r` —— 增长参数，要求 ≥ 0 的**有限值**（r > 4 时轨道会逃出 [0, 1]，**本函数不拦**）；`x0` —— 初值，**默认 0.4**，必须严格落在 (0, 1) 内（端点也不行）；`n_steps` —— 丢弃暂态后记录的迭代步数，**默认 200**，必须 ≥ 2；`n_transient` —— 丢弃的暂态步数，**默认 100**，必须 ≥ 0。本函数**没有** seed（确定性，不走 `_common.rng`）、没有周期检测的容差参数（硬编码，且周期搜索上界是 len(tail)//3 ≤ 33）、没有分岔图/多初值批量入口——要画分岔图请自己循环调用。
- **陷阱**：① **对初值和暂态长度敏感**：周期检测用的是"尾巴上逐点重合"的强判据，若 n_transient 不够（轨道还没落到周期轨道上）或 r 落在周期窗口里，检测结果会变；报告周期时必须同时给出 n_steps/n_transient。② `period = 0` 的含义是"在 ≤ 33 的窗口内没检出周期"，**不等于**已证明混沌（可能有 64 周期），要结合 Lyapunov 指数正负一起解释。③ Lyapunov 指数由**有限步平均**得到，n_steps 太小时波动大；r 略大于 3 的倍周期分岔点附近，指数在 0 附近、符号不稳定。④ r = 0 或轨道恰好命中 x = 0.5（f' = 0）时对数发散，本函数把导数下限钳到 1e-300，避免 −inf 污染结果（代价是该步贡献一个很大的负数）。⑤ r > 4 时轨道逃出 [0, 1] 甚至发散到 −inf，函数不拦，Lyapunov 与周期都会失去意义。
- **怎么检验**：`_self_test` 用 `logistic_map(r, 0.4, 200, 100)` 四组精确基准，全部可独立重跑：① **r = 3.2**：断言 `lyapunov < 0` 且 `period == 2`，并与 2 周期乘子的**闭式**解对拍——乘子 μ = 4 + 2r − r² = 0.16，Lyapunov = ln|μ|/2，容差 **1e-9**；② **r = 2.5**：不动点乘子为 2 − r，断言 `lyapunov == ln|2 − r| = ln 0.5`，容差 **1e-9**；③ **r = 3.9**：断言 `lyapunov > 0`（混沌）且**不**检出 0 < period ≤ 10 的短周期；④ **r = 4**：Lyapunov 应逼近 ln 2，容差 **2e-3**（有限步平均的固有偏差）。另外可做定性验证：r = 3.5 应检出周期 4；边界检查 x0 = 0 或 1、n_steps = 1、n_transient = −1、r = −1 都应抛 `ValueError`；报告周期时务必同时给出 n_steps / n_transient。


### 3.9 随机模型与仿真 —— `examples/algorithms/stochastic.py`

这一族解决"过程本身带随机性"的建模问题：蒙特卡洛估计、排队论解析式与仿真、马尔可夫链稳态/吸收、赌徒破产、MCMC 与 copula。核心纪律是——**蒙特卡洛给出的永远是"估计 + 误差"，只报一个点估计是评审常见的扣分点**；模块级 docstring 明确承诺"每个随机估计都同时返回标准误与 95% 置信区间"，但这一承诺在 `gamblers_ruin` / `mm1_simulate` / `mmc_simulate` 上并未兑现（见各条目的陷阱栏）。

- 随机性统一走 `_common.rng(seed)`（显式种子，绝不用 `np.random` 全局状态），`seed=None` 表示 `DEFAULT_SEED = 20240101`；论文中"结果可复现"是硬要求，seed 与样本量必须写进正文。
- 置信区间统一用模块常量 `Z95 = 1.959963984540054`（比粗略的 1.96 略精确）；排队论三个解析函数（`mm1_metrics` / `mmc_metrics` / `mg1_metrics`）都有 ρ < 1 的硬前提，违反时直接 `raise ValueError` 而不是返回发散值。

#### `mc_pi(n, seed=None)`

- **数学形式**：`(x_i, y_i) ~ U[0,1)²`，`X_i = 1{x_i² + y_i² ≤ 1}`，`π̂ = 4·mean(X)`；`SE(π̂) = 4√(p̂(1−p̂)/n)`；`CI95 = π̂ ± Z95·SE`。
- **步骤**：① 校验 `n >= 2`（否则标准误无定义）；② `rng(seed)` 抽 2n 个均匀数；③ 统计命中数得 `p̂ = hits/n`；④ 返回 `estimate / stderr / ci95_low / ci95_high`。
- **复杂度**：时间 O(n) / 空间 O(n)（全向量化，一次抽 2n 个均匀数）。
- **参数**：`n` 无默认值（≥2），调大让标准误按 `1/√n` 缩小——但 `n = 1e6` 时 SE 仍有约 0.0016，即只能保证小数点后 2~3 位，靠加大样本量硬堆位数是不划算的；`seed` 默认 None（→ `DEFAULT_SEED`），换种子结果就变，论文里必须同时写出 seed 与 n。
- **陷阱**：收敛极慢是本质（误差 O(1/√n)），要更多位数请改用级数展开或 Gauss-Legendre 求积；置信区间是**正态近似**（用样本比例 `p̂` 代替真 p），n 很小时覆盖率偏低；命中判定用的是闭边界（`≤ 1`），对连续分布无影响但口径要交代清楚。
- **怎么检验**：**覆盖率检验**——固定 n、换 1000 个种子重复，统计真值 π 落入 `CI95` 的比例应接近 95%（在二项波动范围内），这是最独立的检验；`stderr` 的标度律——n 翻 4 倍断言 SE 大致减半；与 `4·mean` 之外的独立路线对拍（例如用 `np.sum` 的解析级数 `π = 4·Σ(−1)^k/(2k+1)` 取足够多项）；同一 seed 两次调用断言逐位一致（可复现性）。

#### `mc_integrate(f, a, b, n, seed=None)`

- **数学形式**：`U_i ~ U(a, b)`，`Î = (b−a)·mean(f(U))`；`SE = (b−a)·s_f/√n`，`s_f` 为 `f(U)` 的样本标准差（`ddof=1`）。
- **步骤**：① 校验 `b > a`、`n >= 2`；② 抽 n 个均匀点并**向量化**调用 f；③ 校验返回形状与输入一致、且全为有限值；④ 返回 `estimate / stderr / ci95_low / ci95_high`。
- **复杂度**：时间 O(n)（若 f 内部是逐元素 Python 循环则为 O(n·T_f)）/ 空间 O(n)。
- **参数**：`n` 无默认值（≥2），是最主要的精度旋钮（SE ∝ 1/√n）；`a`、`b` 无默认值，必须 `b > a`；`seed` 默认 None（→ `DEFAULT_SEED`）；没有自适应/重要性抽样的开关——要降方差只能自己换变量、拆分区间或改写 f。
- **陷阱**：朴素的均匀抽样在高维或尖峰被积函数上效率极低（绝大多数样本贡献接近 0、方差巨大），此时应改重要性抽样；标准误只衡量随机误差，不含端点不连续/发散等**系统误差**（端点发散的反常积分要先做变量替换）；`f` 必须支持数组输入并返回同形状数组，只接受标量的函数要用 `np.vectorize` 包装（它只是方便，并不加速）。
- **怎么检验**：闭式解对拍 `∫₀¹x²dx = 1/3`、`∫₀¹eˣdx = e−1`、`∫₀^π sin x dx = 2`；断言真值落入 CI 内（大 n 下可用更严的绝对容差）；配对比较——对 `f` 与 `g` 用**同一个 seed**（同一批 U）算 `∫(f+g)` 与分别求和，应完全一致（方差抵消）；标度律 n 翻 4 倍断言 SE 减半；覆盖率检验同 `mc_pi`。

#### `mm1_metrics(lam, mu)`

- **数学形式**：M/M/1 稳态：`ρ = λ/μ < 1`，`π_n = (1−ρ)ρⁿ`；`P0 = 1−ρ`，`L = ρ/(1−ρ)`，`Lq = ρ²/(1−ρ)`，`W = 1/(μ−λ)`，`Wq = λ/(μ(μ−λ))`；Little 公式 `L = λW`、`Lq = λWq`。
- **步骤**：① 校验 `mu > 0`、`lam >= 0`、`lam < mu`（否则直接 `ValueError`，不发散计算）；② `ρ = λ/μ`；③ 依次算 `P0 / L / Lq / W / Wq` 返回 dict（键为 `rho / P0 / L / Lq / W / Wq`）。
- **复杂度**：时间 O(1) / 空间 O(1)（纯闭式，无迭代）。
- **参数**：`lam`、`mu` 都无默认值，**单位必须一致**（λ 用"人/小时"则 μ 也必须是"人/小时"，混用会让 ρ 差 60 倍）；结果只通过 `ρ = λ/μ` 依赖两个参数，所以灵敏度分析可以直接扫 ρ = 0.5 ~ 0.99 而不必分别扫 λ、μ；`λ ≥ μ` 是硬性拒绝（无稳态）。
- **陷阱**：适用前提是**泊松到达 + 指数服务时间 + 单服务台 + FIFO + 无限容量**；服务时间方差比指数更大时 M/M/1 会**低估**等待，应改用 M/G/1 的 P-K 公式；ρ→1 时所有指标爆炸（ρ=0.95 时 Lq=18，ρ=0.99 时 Lq=98——服务能力只差 4% 排队长度差 5 倍），这是排队论最重要的结论、也是论文里值得单独讨论的敏感性点；本函数返回键 `P0` 是大写，而 `mmc_metrics` 返回的是小写 `p0`，跨函数拼表时要注意键名不一致。
- **怎么检验**：与 `mm1_simulate` 同参数对拍（λ=4、μ=5、n=20000 时**同量级即算通过**，不要期望小数点后两位吻合）；Little 公式自洽性——用返回的 `L` 与 `W` 断言 `L ≈ λW`、`Lq ≈ λWq`（这是恒等式检验，不依赖实现）；**退化检验**——断言 `mmc_metrics(λ, μ, 1)` 的 `rho/P0/L/Lq/W/Wq` 与 `mm1_metrics(λ, μ)` 逐项相等（自测容差 1e-9）；`mg1_metrics` 在指数服务（`Var[S] = 1/μ²`）下必须退化为 M/M/1；极限 `ρ→0` 时断言 `L → 0`、`Lq → 0`、`W → 1/μ`。

#### `mm1_simulate(lam, mu, n_customers, seed=None)`

- **数学形式**：到达间隔 `A_i ~ Exp(1/λ)`、服务时间 `S_i ~ Exp(1/μ)`；开始服务时刻 `s_i = max(a_i, d_{i−1})`、离开时刻 `d_i = s_i + S_i`；输出 `avg_wait = mean(s_i − a_i)` 与时间平均队长 `avg_queue_len = (1/T)∫ q(t)dt`，窗口 `T = d_last − a_first`。
- **步骤**：① 校验 μ>0、λ≥0、λ<μ、`n_customers >= 1`；② 逆变换抽 n 个间隔与服务时间，累加得到达时刻；③ 单服务台事件循环（无事件堆，直接按到达序推进）；④ 在相邻事件之间按"当前等待队长"累加时间积分，除以窗口得时间平均队长。
- **复杂度**：时间 O(n) / 空间 O(n)，n = n_customers（两次抽样 O(n) 加一遍顺序循环；实现里没有排序或堆，docstring 里写的 O(n log n)「事件排序」偏大）。
- **参数**：`n_customers` 无默认值（≥1），调大让随机误差按 1/√n 缩小，**至少要几千**，否则瞬态偏差占主导；`seed` 默认 None（→ `DEFAULT_SEED`）；`lam`、`mu` 无默认值，`λ < μ` 是硬性前提。
- **陷阱**：仿真从**空系统**开始，前期有一段"队长偏低"的瞬态，顾客数少时平均等待会被系统性低估（n 至少取几千，或丢弃前 10% 预热）；单次仿真的随机误差约 O(1/√n)——λ=4、μ=5、n=2000 时 `avg_wait` 常见与理论值差 20%~50%，**同量级即算通过**；返回的是**时间平均队长**而不是"顾客到达时看到的队长"（PASTA 只在泊松到达下把两者联系起来）；时间平均按"首个到达 → 最后离开"积分，样本量小时两端效应带来偏差；**代码不一致**——参数校验允许 `lam = 0`（只拒绝 `lam < 0`），但随后 `gen.exponential(1.0/lam)` 会抛 `ZeroDivisionError`，`lam = 0` 时请勿调用。
- **怎么检验**：与 `mm1_metrics` 解析值对拍（用 n ≈ 1e5，断言相对误差在 20% 以内；λ=4、μ=5 时自测断言 `avg_wait` 与 `Wq`、`avg_queue_len` 与 `Lq` 同量级）；Little 检验 `avg_queue_len ≈ λ·avg_wait`（本实现的 `avg_wait` 是 Wq 口径）；自洽性 `W ≈ avg_wait + 1/μ`；用常量到达/服务（D/D/1）走 `discrete_event_simulation` 断言零等待，作为排队骨架是否正确的独立验证；多 seed 重复给出均值的经验分布与区间。

#### `markov_steady_state(P)`

- **数学形式**：求满足 `πP = π`、`Σπ_i = 1`、`π ≥ 0` 的平稳分布；等价于解秩亏 1 的线性方程组 `(Pᵀ − I)π = 0`，用归一化条件替换其中一行使之可解。
- **步骤**：① `as_matrix` + 方阵校验，拒绝负元素、拒绝行和偏离 1（`atol=1e-9`）；② 构造 `A = Pᵀ − I`，令 `A[-1, :] = 1`、`b = [0,…,0,1]`；③ `np.linalg.solve`；④ 裁剪 1e-14 级浮点小负数、归一化，并校验 `max|πP − π| <= 1e-6`，超限抛 `ValueError` 而不是返回近似解。
- **复杂度**：时间 O(n³)（稠密 LU 分解）/ 空间 O(n²)，n 为状态数。
- **参数**：只有输入 `P`（形状 `(n, n)` 的行随机矩阵：非负、每行和 ≈ 1，容差 1e-9）；**没有可调参数**；列随机矩阵必须**先转置**再传入，否则结果全错且不报错；残差阈值 1e-6 是硬编码的校验而非可调精度。
- **陷阱**：**多解情形**——可约链（例如两个互不连通的状态类）平稳分布不唯一，本线性系统只给出其中一个解（取决于被替换的行），必须结合链的常返类讨论；周期链的平稳分布存在且唯一（不可约时），但**极限分布不存在**，π 是时间平均意义上的分布，不要写成"长期后处于状态 i 的概率"；用幂迭代虽然实现简单，但收敛速度由第二大特征值决定，接近 1 时要迭代上万次且察觉不到不唯一性，本实现改用线性方程组；返回值是裸 `np.ndarray` 而不是 dict，与项目"统一返回 dict"的通用约定不一致。
- **怎么检验**：两状态链有闭式 `π = (b/(a+b), a/(a+b))`（`P = [[1−a, a], [b, 1−b]]`），与实现逐项对拍；不变量——断残差 `max|πP − π|` 与 `|Σπ − 1|` 为机器精度；极限行为——把某个状态设成吸收态（该行单位向量），断言 π 集中在该状态上；与幂迭代对拍（取足够大的 k，`Pᵏ` 的每一行都趋近 π，用可算的小例）；对可约链断言"解满足 πP=π"但不声称唯一。

#### `markov_absorption(P, transient_states)`

- **数学形式**：把 P 分块为 `[[Q, R], [0, I]]`（Q 为瞬态间转移），基本矩阵 `N = (I − Q)⁻¹`；吸收概率 `B = N R`，吸收前期望步数 `t = N·1`。
- **步骤**：① 校验方阵与行随机（每行和 ≈ 1）；② 对 `transient_states` 排序去重，其余状态视为吸收态并逐个校验 `P_ii ≈ 1`（否则报错）；③ 取 `Q = P[trans, trans]`、`R = P[trans, absorbing]`，求 `(I − Q)⁻¹`；④ 返回 `B = N R`（列按吸收态下标升序）与 `t = N·1`。
- **复杂度**：时间 O(n³)（矩阵求逆）/ 空间 O(n²)。
- **参数**：`transient_states` 无默认值，必须非空且下标合法；吸收态是它的**补集**，返回矩阵的列顺序按吸收态下标升序（这一点必须对齐你的标签表）；没有可调数值参数；期望步数对"步数"计数，若一步代表 1 天则单位是天，换时间单位要重新标定。
- **陷阱**：**吸收态必须真的吸收**（对角线为 1）——如果 `transient_states` 漏掉某个带出边的状态，它会被当成吸收态并在结果里静默错误（本实现会校验并报错，是少数不静默的地方）；期望步数可能发散（瞬态子链不满足"吸收必然发生"），此时 `I − Q` 奇异，本实现抛 `ValueError` 而不是返回 `inf`；吸收概率每行之和应为 1，明显不为 1 说明有瞬态被误判，务必打印校验。
- **怎么检验**：赌徒破产链（0 与 K 为吸收态）的吸收概率应等于闭式 `(1 − r^start)/(1 − r^K)`（`r = (1−p)/p`，p=0.5 时 `start/K`），期望步数在 p=0.5 时等于 `start·(K − start)`；**不变量**——吸收概率每行之和 = 1（独立于实现的结构性结论）；与 `gamblers_ruin` 的模拟频率对拍（两条完全独立的代码路径：一条线性代数、一条蒙特卡洛）；手工分块用 2×2 的 `I − Q` 解析求逆对拍小例。

#### `gamblers_ruin(p, start, target, seed=None, n_trials=20000)`

- **数学形式**：`r = (1−p)/p`；胜出（到达 `target`）概率 `P_win = (1 − r^start)/(1 − r^target)`（`p ≠ 0.5`），`p = 0.5` 时 `P_win = start/target`；`P_ruin = 1 − P_win`；模拟频率 `p̂ = 破产条数 / 已结束条数`。
- **步骤**：① 校验 `0 < p < 1`、`start`/`target` 为整数且 `0 < start < target`、`n_trials >= 1`；② 按闭式算 `P_win`、`P_ruin`（`|p − 0.5| < 1e-12` 时走单独分支）；③ 批量随机游走（活跃掩码 + 向量化推进，撞 0 记破产、撞 `target` 记成功）；④ 用 `~alive`（已结束的游走）作分母算频率并返回三个键。
- **复杂度**：时间 O(n_trials × 步数)，步数上限写死为 `1000·max(target, 10)`；空间 O(n_trials)。
- **参数**：`n_trials` 默认 20000（≥1），调大让频率的标准误按 `1/√n_trials` 下降（20000 时约 0.003），是最主要的精度旋钮；`seed` 默认 None（→ `DEFAULT_SEED`）；`p ∈ (0,1)`、`start`、`target` 都无默认值，`p` 是最该做敏感性分析的参数。
- **陷阱**：`p` 对破产概率极其敏感——`start=10, target=20` 时 p=0.49 破产概率约 0.599、p=0.51 降到约 0.401，概率只差 0.02 结论差 20 个百分点，做投资/风险类题目必须做 p 的敏感性分析；`p = 0.5` 必须单独分支（`r = 1` 会让分母为 0）；**分母口径极易写错**——频率的分母必须是"已结束的游走"（破产的 + 到达目标的），写成"破产的 + 仍在跑的"会漏掉成功离场的样本、把频率系统性高估（本模块初版就踩过：n=20000、p=0.49 时算出 1.0 而不是 0.60），实现用 `~alive` 且步数上限截断后未结束的游走被剔除；**返回里没有 `stderr` / `ci95` 键**，与模块级"每个随机估计都返回标准误与 95% 置信区间"的承诺不符，必须自己按 `√(p̂(1−p̂)/n_valid)` 补算并在论文里报告。
- **怎么检验**：与解析式 `ruin_prob_theory` 对拍，判据用 `|sim − theory| < 3·√(p̂(1−p̂)/n)` 而不是固定阈值（解析值与模拟值差 2~3 个标准误是正常的）；`p = 0.5` 时闭式 `start/target`；与 `markov_absorption` 在同一条赌徒破产链上对拍（线性代数 vs 蒙特卡洛，两条独立路径）；覆盖率检验——换多个 seed 重复，统计解析值落入经验 95% 区间的比例应接近 95%；换元关系检查（如 p↔1−p 下破产概率与从另一端出发的胜出概率互换）。

#### `mmc_metrics(lam, mu, c)`

- **数学形式**：`a = λ/μ`（提供负载）、`ρ = a/c < 1`；`p0 = [Σ_{n=0}^{c−1} aⁿ/n! + a^c/(c!(1−ρ))]⁻¹`；Erlang-C `C = p_wait = a^c/(c!(1−ρ))·p0`；`Lq = C·ρ/(1−ρ)`，`Wq = Lq/λ`，`W = Wq + 1/μ`，`L = Lq + a`。
- **步骤**：① 校验 `mu > 0`、`lam >= 0`、`c` 为 ≥1 的整数、`ρ = λ/(cμ) < 1`；② 用 `term_n = term_{n−1}·a/n` 递推 `Σ_{n<c} aⁿ/n!` 与 `a^c/c!`（不显式构造阶乘）；③ 得 `p0`、`p_wait`、`Lq`、`Wq`、`W`、`L`；④ 返回含 `erlang_c`（与 `p_wait` 同值）。
- **复杂度**：时间 O(c)（只递推 c 项）/ 空间 O(1)。
- **参数**：`c` 无默认值（整数 ≥1），是灵敏度分析的核心变量——调大同 λ、μ 下 ρ 变小、等待骤降（这正是"多加一个服务台值不值"的量化依据）；`lam`、`mu` 无默认值，`mu` 是**单台**服务台的服务率（误填成系统总服务率会把 ρ 放大 c 倍）；`λ = 0` 时 `Wq` 按 0/0 无定义，实现显式返回 0（系统永远空闲）。
- **陷阱**：适用前提是泊松到达 + 指数服务 + c 台**同质**并行 + FIFO + 无限等待空间；**ρ 必须按 `c·μ` 归一**（写成 `λ/μ` 是最常见的实现错误，会让稍有负载就误报"无稳态"）；`p_wait` 是"需要等待的概率"不是"被拒绝的概率"（后者属于损失制系统的 Erlang-B，两者混用是排队论建模的经典错误）；**`a^c/c!` 在 a 约为 700 以上时会溢出成 `inf`，本实现不做对数域处理**（教学透明版，需要在很大 c 或很重负载下工作时必须自己改成对数域）；ρ→1 时 Lq 同样爆炸。
- **怎么检验**：**退化检验**——c=1 时必须逐项等于 M/M/1（自测断言最大偏差 < 1e-9）；手算对拍——c=2、λ=3、μ=2 时 `p0 = 1/7`、`C = 9/14 ≈ 0.642857`；**第二条独立路线**——用 Erlang-B 递推 `B = aB/(n + aB)` 再按 `C = B/(1 − ρ(1−B))` 求 C，两条推导完全不同却给出同一个数（自测两者都等于 9/14）；与 `mmc_simulate` 对拍（n=2e4 时 Wq 相对误差 < 15%）；不变量 `L = Lq + a`、`W = Wq + 1/μ`、`Lq = λWq`。

#### `mg1_metrics(lam, service_mean, service_var)`

- **数学形式**：`ρ = λE[S] < 1`；Pollaczek-Khinchine 公式 `Wq = λE[S²]/(2(1−ρ))`，其中 `E[S²] = Var[S] + E[S]²`；再套 Little：`W = Wq + E[S]`、`Lq = λWq`、`L = λW`。
- **步骤**：① 校验 `service_mean > 0`、`service_var >= 0`、`lam >= 0`、`ρ = λE[S] < 1`；② 由方差与均值平方算二阶矩 `E[S²]`；③ 代入 P-K 公式得 `Wq`；④ 由 Little 公式得 `W / Lq / L` 并返回。
- **复杂度**：时间 O(1) / 空间 O(1)。
- **参数**：`service_mean`、`service_var` 无默认值（`Var[S] = 0` 即确定性服务，此时 `Wq = ρE[S]/(2(1−ρ))`，恰好是 M/M/1 的一半）；`lam` 无默认值；三者单位必须统一到同一时间单位；`ρ = λE[S] < 1` 是硬性前提（违反直接 `ValueError`）。
- **陷阱**：适用前提是泊松到达 + 服务时间独立同分布且与到达过程独立 + 单服务台 + FIFO + 无限容量（批量到达或到达与服务的相关性会破坏公式前提）；方差必须以**二阶矩**形式进入，误把 `Var[S]` 当 `E[S²]` 会把 `Wq` 系统性算小（差 `λE[S]²/(2(1−ρ))` 这一项）；P-K 公式的威力与局限同源——`Wq` 只依赖服务时间的前两阶矩，与分布形状无关，因此**不能据此推断尾概率**（如超时率）；ρ→1 时同样爆炸；ρ=0 时若 `Var[S] > 0`，`Wq` 仍为 0（没有到达就没有排队）。
- **怎么检验**：退化检验——指数服务（`Var[S] = 1/μ²`）时必须逐项等于 `mm1_metrics`（自测断言 1e-12）；确定性服务（`Var[S] = 0`）时与 M/D/1 闭式 `Wq = ρE[S]/(2(1−ρ))` 对拍；用两点分布/重尾分布的服务时间跑 `discrete_event_simulation` 与 P-K 结果对拍（分布形状任意都应吻合，这是公式本身的强结论）；不变量 `Lq = λWq`、`L = λW`、`W = Wq + E[S]`。

#### `discrete_event_simulation(arrival_fn, service_fn, n_customers, n_servers=1, seed=None, t_max=None)`

- **数学形式**：事件集 {到达, 离开}；到达时刻由**累加到达间隔**给出，开始服务 `s = max(到达时刻, 服务台空闲时刻)`、离开 `= s + 服务时长`；队长只在事件时刻变化，时间平均队长 `= (1/T)∫ q(t)dt`，T 取"首个到达 → 最后离开"。
- **步骤**：① 校验两个回调可调用、`n_customers`/`n_servers` 为 ≥1 的整数、`t_max` 为 None 或 > 0；② 用 heapq 维护事件堆，按 `(时刻, 序号)` 排序（到达与离开都是事件）；③ 到达时若空闲堆非空立即开始服务，否则进 FIFO 等待队列；离开时把等待队首放到刚空出的台上（FIFO、不抢占）；④ 事件间按"队长 × 时长"累加面积；⑤ 返回 `wait / sojourn / server_busy / n_served / departure`。
- **复杂度**：时间 O(n log n)（n = n_customers，事件堆操作；超过 `t_max` 的事件不再生成）/ 空间 O(n)。
- **参数**：`arrival_fn` 的口径是 `arrival_fn(gen) -> float`——接受一个 numpy `Generator`、返回**下一个到达间隔**（不是到达时刻），常用写法 `lambda g: g.exponential(1/lam)`；`service_fn` 同口径、返回服务时长；`n_servers` 默认 1（≥1，各台同质、服务时间独立同分布）；`t_max` 默认 None（把 n_customers 全部服务完为止），给定后超过 t_max 的到达不再生成、`n_served` 可能小于 `n_customers`（t_max 本身是**含**的）；`seed` 默认 None（→ `DEFAULT_SEED`），随机性完全由它控制。
- **陷阱**：`arrival_fn` 返回的是**间隔而不是时刻**，传成时刻会让所有顾客挤在同一时间到达；仿真从"全空系统"起步，`n_customers` 只有几百时前几十个顾客几乎不等待、平均等待被系统性低估，必须加大样本或丢弃预热期；`server_busy` 是**累计时长而不是比例**，利用率要自己除以窗口长度，且窗口取的是"首个到达 → 最后离开"而不是 `[0, t_max]`；**本函数不校验 ρ < 1**——ρ ≥ 1 时队列持续增长、仿真不报错但结果没有意义，请在调用方（如 `mmc_simulate`）自己校验；公开返回值不含时间平均队长与利用率（内部 `_des_core` 有 `lq_time_avg`/`utilization`，只供 `mmc_simulate` 使用）。
- **怎么检验**：**D/D/1 极限**——常量到达间隔 1.0、常量服务时长 0.5 时必须零等待（`max(wait) <= 1e-12`）且平均逗留时间恰为 0.5（自测，两条断言都是解析结论）；守恒/自洽——常量服务时长下断言 `sojourn = wait + 服务时长` 逐元素成立，离开时刻序列非减、`n_served <= n_customers`；与 M/M/c 解析式对拍（经 `mmc_simulate`）；利用率交叉核对——`server_busy.sum()/(n_servers·span)` 应与 `λ·E[S]/c` 给出的 ρ 大致相符；把 `t_max` 设得很小，断言只服务了窗口内到达的顾客且 `n_served < n_customers`（截断逻辑的定向检验）。

#### `mmc_simulate(lam, mu, c, n_customers=2000, seed=None)`

- **数学形式**：把 `Exp(1/λ)` 到达间隔与 `Exp(1/μ)` 服务时长交给通用事件引擎，输出时间平均 `Lq`、顾客平均 `Wq` 与 `W`、服务台利用率 `= 总忙期/(c·窗口)`。
- **步骤**：① 校验 `mu > 0`、`lam > 0`、`c` 为 ≥1 的整数、`λ < cμ`（否则直接抛错，不发散仿真）、`n_customers >= 1`；② 用两个 `lambda g: g.exponential(...)` 回调调用 `_des_core`；③ 对 `wait`/`sojourn` 取均值，连同引擎的 `lq_time_avg` 与 `utilization` 一起返回。
- **复杂度**：时间 O(n_customers · log n_customers) / 空间 O(n_customers)。
- **参数**：`n_customers` 默认 2000（≥1），docstring 明确建议 **≥ 20000** 才能把 `Wq` 的相对误差压到 15% 以内——它是"精度旋钮"而不是模型参数，灵敏度分析里应与 seed 一起报告；`seed` 默认 None（→ `DEFAULT_SEED`）；`c` 无默认值，`λ < cμ` 是硬性前提。
- **陷阱**：单次仿真的随机误差约 O(1/√n)——c=2、λ=3、μ=2 时 n=2000 的 `Wq` 相对误差常在 10%~30% 之间波动，必须加大样本再下结论（自测用 20000）；系统从空开始存在"队长偏低"的瞬态，n 越大相对偏差越小；`Lq` 是**时间平均**，非泊松到达下与"每个顾客到达时看到的队长"并不相等（此处泊松到达使 PASTA 成立，才能在解析式之间对齐）；结果依赖 seed，论文必须同时报告 seed 与样本量；**返回里没有标准误/置信区间**，与模块级承诺不符，要自己用多 seed 独立重复给出区间。
- **怎么检验**：与 `mmc_metrics` 的 Erlang-C 闭式解对拍（自测用 n=20000 断言 `Wq` 相对误差 < 15%、`n_served` 恰为 20000）；c=1 时与 `mm1_metrics`/`mm1_simulate` 交叉对拍；不变量 `W ≈ Wq + 1/μ`、`Lq ≈ λWq`（Little）、`utilization ≈ ρ = λ/(cμ)`；多 seed 重复给出 `Wq` 的均值与区间，断言闭式值落在区间内（覆盖率检验）。

#### `metropolis_hastings(log_target, x0, n_samples=5000, proposal_sd=1.0, seed=None, burn_in=1000)`

- **数学形式**：对称随机游走提议 `x' = x + sd·N(0, I)`，接受概率 `α = min(1, exp(log_target(x') − log_target(x)))`（对称提议使接受比只剩密度比），链的平稳分布 `∝ exp(log_target)`。
- **步骤**：① 校验 `x0` 可转一维、`n_samples >= 1`、`burn_in >= 0`、`proposal_sd > 0`；② 每步抽提议并用 `log(u) < lp' − lp` 判断接受（避免 `exp` 下溢）；③ 前 `burn_in` 步只更新链、不记录；④ 记录 `n_samples` 个状态并统计**采样阶段**的接受率，返回样本矩阵、接受率、样本均值与样本方差（`ddof=1`）。
- **复杂度**：时间 O((n_samples + burn_in)·d)（每步一次 `log_target` 调用）/ 空间 O(n_samples·d)，d 为维数。
- **参数**：`proposal_sd` 默认 1.0（>0），是**唯一真正需要调的参数**——太小则接受率接近 1 但链混合极慢（样本高度自相关，均值的标准误被低估），太大则接受率骤降、链长期不动；一维标准正态的经验最优接受率约 0.44、高维约 0.234，可用它反推步长；`burn_in` 默认 1000（≥0），初始点落在低密度区时要给够；`n_samples` 默认 5000（≥1）；`seed` 默认 None（→ `DEFAULT_SEED`）；`x0` 无默认值。
- **陷阱**：返回的 `var` 是**样本方差**而不是均值的方差——要报"均值 ± 标准误"必须考虑自相关（用批均值法估计），直接用 `s/√n` 会低估误差；提议是**各向同性**的，目标各维尺度差异大时（例如方差 1 与 100）必须自己做预条件（换坐标或用协方差提议），否则混合极差；`log_target` 返回 `NaN` 会直接抛 `ValueError`（`±inf` 允许，用于表示定义域外零密度）；没有收敛诊断（Gelman-Rubin、有效样本量），只能靠接受率与多链重复自行判断。
- **怎么检验**：对标准正态目标（`log_target = −x²/2`）断言样本均值 ≈ 0（|均值| < 0.1）、样本方差 ∈ (0.8, 1.25)、接受率 ∈ (0.2, 0.8)（自测）；对一维指数或双峰目标用已知矩/已知分位数对拍；与目标 CDF 做 KS 检验（需要样本近似独立，可先抽稀）；步长扫描实验——断言接受率随 `proposal_sd` 单调下降，且批均值法给出的均值标准误在中间某个步长附近最小（自相关诊断）；换 seed 重复并用批均值置信区间检验目标矩的覆盖率。

#### `gibbs_sampler_bivariate_normal(mu, cov, n_samples=5000, seed=None, burn_in=1000)`

- **数学形式**：对二元正态，条件分布仍为正态：`X1|X2=x2 ~ N(μ1 + ρσ1/σ2·(x2−μ2), σ1²(1−ρ²))`，`X2|X1` 对称；每次迭代先用最新 `x2` 抽 `x1`、再用**刚更新的** `x1` 抽 `x2`，平稳分布为 `N(μ, Σ)`。
- **步骤**：① 校验 `mu` 长度为 2、`cov` 为 2×2 且对称、对角线为正、`|ρ| < 1`（即正定）；② 算两个条件标准差 `σ_i√(1−ρ²)`；③ 交替抽两分量（Gibbs 是接受率恒为 1 的 MH，无拒绝步骤）；④ 预热后记录，返回样本矩阵、样本均值与样本协方差（`ddof=1`）。
- **复杂度**：时间 O(n_samples + burn_in)（每步常数次运算）/ 空间 O(n_samples)。
- **参数**：`n_samples` 默认 5000（≥1）；`burn_in` 默认 1000（≥0）；`seed` 默认 None（→ `DEFAULT_SEED`）；`mu`、`cov` 无默认值。**没有步长参数**——Gibbs 的接受率恒为 1，能调的只有样本量与预热长度。
- **陷阱**：**必须真的交替使用最新值**——用同一轮的旧 `X2` 去抽 `X2` 会退化成独立采样，样本相关结构全错（实现里非常常见）；相关系数接近 ±1 时 Gibbs 混合极慢（自相关约 `ρ²`），需要大量样本，此时应先做变量替换（例如对 `(X1, X2 − X1)` 采样）再变换回去；样本协方差是估计量，`n_samples = 5000` 时每个元素的随机误差约 0.02~0.03，**不要用 1e-3 级别的容差去断言**；这里只支持二元，多元要逐分量条件分布并自己处理协方差求逆。
- **怎么检验**：与目标 `Σ` 逐项对拍（自测容差 0.1）；`ρ = 0` 时断言样本协方差非对角元 ≈ 0，且与独立采样结果不可区分；边缘矩闭式（均值 `μ`、方差 `σ_i²`、相关系数 `ρ` 与样本相关对拍）；与 `metropolis_hastings` 在**同一个**二元正态目标上对拍（两条独立链应给出同分布，用矩与分位数比较）；`burn_in = 0` 且初始点取真均值时断言均值无偏（瞬态检验）；多 seed 估计样本协方差本身的 MC 标准误，判据用 2~3 个标准误而不是固定容差。

#### `gaussian_copula(U, n_draws=1000, seed=None)`

- **数学形式**：Sklar 定理 `F(x₁,…,x_d) = C(F₁(x₁),…,F_d(x_d))`；高斯 copula `C(u) = Φ_d(Φ⁻¹(u₁),…,Φ⁻¹(u_d); ρ)`；相关性由伪观测的 Spearman 秩相关经 `ρ = 2·sin(π·ρ_s/6)` 换算；采样为 `Z = L·N(0,I)`（`ρ = LLᵀ`）后逐元素取 `U = Φ(Z)`。
- **步骤**：① 校验 `U` 为 `(n >= 3, d >= 2)` 且取值在 `[0,1]` 的伪观测；② 每列取平均秩得 Spearman 矩阵，换算成 Pearson `ρ`，并对特征值做 1e-10 截断保证正定；③ Cholesky 分解 `ρ = LLᵀ`；④ 抽 `Z ~ N(0, ρ)` 并逐元素取标准正态分布函数得均匀样本；⑤ 返回 `rho / samples / tail_dependence`。
- **复杂度**：时间 O(n·d·log n + d³ + n_draws·d²) / 空间 O(n_draws·d)，n 为伪观测行数。
- **参数**：`n_draws` 默认 1000（≥1），调大只降低生成样本的蒙特卡洛噪声、**不改变** `rho`（ρ 完全由 U 决定）；`seed` 默认 None（→ `DEFAULT_SEED`）；`U` 无默认值，必须是**伪观测**（[0,1] 上的秩或经验分布值），不是原始数据。
- **陷阱**：**高斯 copula 没有尾部相依**（`λ_U = λ_L = 0`），无法刻画"极端事件同时发生"的风险（2008 年金融危机中 CDO 定价误用高斯 copula 正栽在这里）；返回的 `tail_dependence` 是**理论值 0.0 而不是经验尾频**，不能拿它当数据诊断量，d>2 时它与 d 无关、逐对尾相依要自己按 `rho[i, j]` 单独算（高斯下仍然都是 0）；直接喂原始数据（而非伪观测）会得到毫无意义的 ρ；样本量小（n < 50）时 `ρ_s` 可能非正定，实现做特征值截断（量级 1e-10，会轻微改变相关结构）；`_norm_cdf` 用 `1 + erf` 实现，`z ≲ −37` 时灾难性抵消使结果恰好为 0，下游若再取 `Φ⁻¹` 会得到 `−inf`，极端分位要小心。
- **怎么检验**：**用已知 ρ 的高斯 copula 数据反解**——先生成真 `ρ = 0.5` 的 1500 行伪观测，再断言解出的 `rho[0,1]` 与真值差 < 0.05（自测）；边缘一致性——断言 `samples` 每列经验分布接近 `U(0,1)`（KS 检验或分位数对拍）；秩相关回归——`spearman(samples)` 应满足 `ρ_s = (6/π)·arcsin(ρ/2)`（可用成熟库独立算 Spearman 对拍）；极限——`ρ = I` 时各列独立（样本 `ρ_s ≈ 0`）；把 `n_draws` 加大，经验 copula 在网格上应收敛到理论值 `Φ_ρ(Φ⁻¹u, Φ⁻¹v)`。

#### `t_copula(U, df=5, n_draws=1000, seed=None)`

- **数学形式**：t copula `C(u) = t_{ν,ρ}(t_ν⁻¹(u₁),…,t_ν⁻¹(u_d))`；采样 `Z ~ N(0, ρ)`、`W ~ χ²_ν` 独立，`T = Z/√(W/ν)`（**整行共用一个 W**），`U = t_ν(T)`；下尾相依系数 `λ = 2·t_ν(−√((ν+1)(1−ρ)/(1+ρ)))`（正尾相同，t copula 上下尾对称）。
- **步骤**：① 校验 `U` 为伪观测、`df` 为 ≥1 的整数、`n_draws >= 1`；② 同 `gaussian_copula` 路线估计 ρ 并 Cholesky；③ 抽 `Z ~ N(0,ρ)` 与 `W ~ χ²_ν`（`Generator.chisquare`），构造 `T`；④ 逐元素调用 t 分布函数（不完全 Beta 实现）得均匀样本；⑤ 按相关矩阵**非对角元平均**取 ρ̄ 后按闭式返回尾相依系数。
- **复杂度**：时间 O(n·d·log n + d³ + n_draws·d·max_iter)（t 分布函数逐元素调用、内部是不完全 Beta 连分式）/ 空间 O(n_draws·d)。
- **参数**：`df` 默认 5（整数 ≥1），这是 t copula 唯一的"尾部强度"旋钮——**调小尾部更厚、尾相依更强**，`df → ∞` 退化为高斯 copula（尾相依 → 0），做尾部风险敏感性分析就扫它；`n_draws` 默认 1000（≥1，只影响样本量、不影响 λ）；`seed` 默认 None（→ `DEFAULT_SEED`）；`U` 同高斯 copula，必须是伪观测。
- **陷阱**：λ 只在 `ρ > 0` 时非零，`ρ <= 0` 时 t copula 也没有下尾相依；`ρ → −1` 时 `(1−ρ)/(1+ρ)` 发散、`sqrt` 项趋于 `+inf`、CDF 趋于 0，本实现对该情形返回 0（有 `1e-12` 的护栏）；λ 是**渐近量**，`n_draws = 1000` 时按 5% 分位点的经验尾频极不稳定——**不要拿经验尾频去和 λ 对拍**，应当用闭式公式对拍（自测就是这么做的）；**多维（d > 2）时按相关矩阵非对角元平均给出一个标量 ρ̄**，它只是概要统计，严格的尾相依要逐对报告，只有 d=2 时这个标量才有精确解释；混合抽样必须是"同一个 W 作用于整行向量"，若每维用独立 W，得到的联合分布不是 t copula（而是另一种椭球 copula）；`samples` 是 t 分布函数变换后的均匀值，不要把 t 分位数当作最终样本。
- **怎么检验**：尾相依闭式对拍——用返回的 `rho` 重算 `2·t_df(−√((df+1)(1−ρ)/(1+ρ)))` 断言与 `tail_dependence` 一致到 1e-12（自测）；t 分布函数的独立校验——`df = 1` 是柯西，断言 `P(T ≤ 1) = 3/4`，`df = 2` 断言 `P(T ≤ −1) = 0.5 − 1/(2√3)`（两条都是闭式，与实现无关）；极限——`df = 1000` 时断言 λ < 0.01（退化为高斯 copula），`ρ → 1⁻` 时断言 λ → 1；与 `gaussian_copula` 用同一份 U 应得到**完全相同**的 `rho` 矩阵（同一条换算代码路径，可断言逐元素相等）；样本秩相关同样应满足 `ρ_s = (6/π)·arcsin(ρ/2)`（用成熟库的 Spearman 对拍）；`df` 扫描——断言 λ 随 df 增大单调下降。

#### `Z95`

- **数学形式**：`Z95 = Φ⁻¹(0.975) = 1.959963984540054`，即标准正态双侧 95% 置信区间的分位数（比粗略的 1.96 略精确）。
- **步骤**：① 模块导入时作为常量求值（不参与任何计算）；② `mc_pi` 与 `mc_integrate` 用它把标准误换成区间上下限 `estimate ± Z95·stderr`；③ 其他返回区间的函数也应引用它而不要复制字面量。
- **复杂度**：O(1)（常量，无计算、无存储）。
- **参数**：**没有参数、不可调**；它是**常量而不是函数**（列在 `__all__` 里，可以 `from ...stochastic import Z95` 直接取）；若要改置信水平，应改成对应的分位数（90% 用 1.6449、99% 用 2.5758），而不是在调用处硬编码。
- **陷阱**：`Z95` 只对**正态近似**成立——小样本或重尾估计量（例如 `p̂` 接近 0 或 1 的比例、排队仿真的等待时间）用它算出的区间覆盖率会偏低，此时应改用 t 分位数或 bootstrap 区间；不要把 `Z95` 与 t 分布分位数混用（小样本的均值区间应用 `t_{n−1,0.975}`）；它是普通浮点常量，别在别处又抄一份字面量，否则改置信水平时会漏改。
- **怎么检验**：断言 `|Φ(Z95) − 0.975| < 1e-12`，其中 Φ 用 `math.erfc` 独立实现（与模块里的 `_norm_cdf` 走不同的数值路线）；与统计表或成熟库（`scipy.stats.norm.ppf(0.975)`）对拍；覆盖率检验——对已知真值的正态估计量重复大量实验，置信区间覆盖真值的比例应约 95%（这把常量与区间构造方式一起验了）。


### 3.10 几何与坐标 —— `examples/algorithms/geometry.py`

这一族是"平面几何量 + 空间插值"的透明实现，只依赖 numpy 与标准库，目标是让论文能写清"某个几何量/插值值是怎么从原始坐标算出来的"，以及每种口径的隐含假设。使用前必须记住模块级坐标系约定：

- `convex_hull` / `polygon_area` / `point_in_polygon` / `idw_interpolate` / `ordinary_kriging` 全部工作在**平面直角坐标** (x, y) 上，不涉及经纬度；只有 `haversine` 吃经纬度（**度**）。
- 球面距离把地球当作半径 **6371.0088 km**（IUGG 算术平均半径，模块常量 `EARTH_RADIUS_KM`）的正球，长距离（> 1000 km）相对 WGS-84 椭球有约 **0.3%** 误差。
- 所有点集入参一律经内部 `_as_points` 校验：形状必须是 (n, 2)（一维输入会被 reshape 成 (1, 2)）、非空、全有限，否则抛 `ValueError`。

#### `convex_hull(points)`

- **数学形式**：Andrew 单调链。记 `cross(o, a, b) = (a_x − o_x)(b_y − o_y) − (a_y − o_y)(b_x − o_x)`；把点按 (x, y) 字典序排序后分别扫出下链与上链，栈内若"新点与栈顶两点构成非左转"（`cross <= 0`）就弹栈；下链与上链首尾相接即为**逆时针**凸包。
- **步骤**：① `_as_points(points, "points")` 校验并转成 (n, 2) 数组；② `np.lexsort((pts[:,1], pts[:,0]))` 得到按 (x, y) 字典序的下标序列；③ n == 1 直接返回 `[0]`，n == 2 返回两个下标；④ 正序扫描构造 `lower`、逆序扫描构造 `upper`（两个栈各自"非左转即弹栈"）；⑤ 返回 `lower[:-1] + upper[:-1]` 并 `int()` 成 Python 整数。
- **复杂度**：时间 O(n log n)（排序主导，两次扫描各 O(n)）/ 空间 O(n)（下标序列 + 两个栈）。
- **参数**：只有 `points`（形状 (n, 2) 的点集，无默认值；必须非空、全有限，否则 `ValueError`）。没有可调开关——"共线边上的中间点保留还是丢弃"在实现里固定为**丢弃**。返回 `list[int]`：凸包顶点的**原始下标**（不是坐标），逆时针排列、不含共线边中间点；所有点共线时返回该线段的两个端点，n == 1 时返回 `[0]`，n == 2 时返回两个下标。
- **陷阱**：① 叉积判据用 `<= 0` 还是 `< 0` 直接决定"凸包顶点数"：本实现用 `<= 0`（丢掉共线中间点，通常正是想要的口径），换成 `< 0` 会让正方形边上多出若干"伪顶点"，做顶点数断言前务必确认口径；② **经纬度点不能直接当平面坐标求凸包**——高纬度处经度 1 度只有几十公里，凸包会明显变形，要么先投影要么改球面凸包；③ 退化输入（所有点重合、只有 1~2 个点）没有"内部"概念，本函数返回退化结果而不报错，需要调用方自己判断；④ 返回的是下标而不是坐标，入参若是 DataFrame 切片，请确认下标还原后仍对应原行。
- **怎么检验**：自测点集是"单位正方形 4 顶点 + 4 个边中点 + 2 个内部点"共 10 点，断言 `hull_size` 应为 **4**（共线边中点被丢弃）；由点集构造顺序（正方形在前）可手推出这 4 个下标就是 `[0, 1, 2, 3]`，`hull_indices` 已排序可逐位断言。再加三条交叉检验：手算三点 (0,0)/(1,0)/(0,1) 应返回 3 个顶点；用 `polygon_area` 对按 hull 下标取出的坐标算面积，应等于手算的凸包面积；用 `point_in_polygon` 断言全部输入点（含内部点与边中点）都判为内/边界。最后把输入顺序整体打乱，断言返回的**顶点集合**不变（下标会变，几何结论不变）。

#### `polygon_area(points)`

- **数学形式**：鞋带公式（shoelace）`A = |0.5 · Σ (x_i·y_{i+1} − x_{i+1}·y_i)|`，下标按模 n 回绕，结果取绝对值。
- **步骤**：① `_as_points` 校验；② n < 3 直接返回 `0.0`；③ 用 `np.roll(x, -1)` / `np.roll(y, -1)` 造出"下一个顶点"；④ 求和后取绝对值除以 2 返回 float。
- **复杂度**：时间 O(n)（全向量化，无 Python 循环）/ 空间 O(n)。
- **参数**：只有 `points`（形状 (n, 2) 的多边形顶点，按顺序给出，顺/逆时针都行，无默认值）。没有可调参数；n < 3 不报错而是返回 `0.0`。返回 float，面积与输入坐标同单位的平方，**非负**。
- **陷阱**：① **只对简单多边形成立**：边自交（"8 字形"）时正负面积互相抵消，结果无意义，遇到自交数据应先拆分多边形或改三角剖分；② 顶点顺序决定符号（逆时针为正），本函数已取绝对值，要判断方向请自己算带符号面积（`polygon_centroid` 会顺带返回）；③ 经纬度直接代入会把面积算成"度²"且随纬度严重失真，面积类计算必须先投影（如 UTM）或用球面多边形公式；④ 顶点重复（闭合点一头一尾写了两次）不影响结果（该点 cross 贡献为 0），但会让 n 虚增 1。
- **怎么检验**：自测给了两个手算实例——三角形 (0,0)/(4,0)/(0,3) 的 `polygon_area` 应为 **6.0**（底 4 高 3），单位正方形应为 **1.0**，两者都用 `round(..., 6)` 返回可逐位断言。补充检验：把顶点顺序反转，断言面积不变（绝对值口径）；对 n = 2 与 n = 1 断言返回 `0.0` 而不抛错；把多边形整体做平移（面积不变）与放大 k 倍（面积应变 k² 倍）的仿射检验；用三角剖分（任取一点连成 n−2 个三角形，三角形面积和）独立复算同一多边形做对拍。

#### `point_in_polygon(point, polygon)`

- **数学形式**：射线法（crossing number）。从待测点向 **+x** 方向作射线，与多边形各边的交点个数为**奇数则在内**、偶数则在外；边界另用"包围盒 + 叉积为零"的判据单独命中，命中即返回 True。
- **步骤**：① `as_vector(point, "point")` 校验长度为 2；② `_as_points(polygon, "polygon")` 校验，顶点数 < 3 抛 `ValueError`；③ 遍历每条边 (i, i+1 mod n)：先做包围盒判据，点在边的包围盒内且叉积为 0（即落在该边所在直线上）就返回 True；④ 再用半开区间写法 `(y_i > y) != (y_j > y)` 判是否跨越射线所在水平线，跨越则算出交点的 x 坐标，若 `x_cross > x` 就把 `inside` 取反；⑤ 返回 `bool(inside)`。
- **复杂度**：时间 O(n)（Python 层逐边循环）/ 空间 O(n)（多边形副本；边界判据只用常数空间）。
- **参数**：`point`（长度 2 的 `(x, y)`，无默认值，长度不是 2 即 `ValueError`）；`polygon`（形状 (n, 2) 的顶点，**首尾不必重复**，n >= 3，否则 `ValueError`）。没有可调参数（边界与叉积的容差写死在实现里，docstring 未给出具体数值）。返回 bool：点在内部或**恰好落在边/顶点上**时为 True。
- **陷阱**：① **边界口径必须在论文里写清楚**：本函数把"点在边上"算作内部，而 GIS 常把边界单列一类（DE-9IM 的 boundary），同样坐标在不同软件里会得到不同结果；② 射线法对"射线正好穿过顶点/与边重合"天生有歧义，本实现用 `(y_i > y) != (y_j > y)` 的半开区间写法消除大部分退化，但极端退化数据（大量水平边）仍可能出错；③ 多边形必须是**简单多边形**（边不自交），自交时"内/外"本身无定义；④ 经纬度坐标在平面近似下判断"是否在行政区内"只适用于小范围，跨半球或高纬度区域会判错。
- **怎么检验**：自测用单位正方形断言三个口径：`pip_inside` = (0.5, 0.5) 在内、`pip_outside` = (1.5, 0.5) 在外、`pip_on_edge` = (0.5, 1.0) 落在上边**算内部**（这三个值都以 1/0 形式返回，可直接断言）。补充检验：对正方形取四个顶点与四条边中点逐个断言为 True（顶点/边命中口径）；构造"点在正上方但 x 在射线上"的退化例子（如 (0.5, 2.0)、y 与两个顶点相等）验证不误判；把多边形整体平移后断言同一相对位置的点结论不变；与凸包的射线求交数实现（独立数一遍交点个数）对拍。

#### `haversine(lat1, lon1, lat2, lon2)`

- **数学形式**：haversine 大圆距离 `a = sin²(Δφ/2) + cosφ1·cosφ2·sin²(Δλ/2)`，`d = 2R·asin(√a)`，其中 φ 为弧度纬度、Δλ 为经度差（弧度），半径 `R = EARTH_RADIUS_KM = 6371.0088` km。
- **步骤**：① 四个入参全部 `float()` 并 `math.radians` 转弧度；② 算 `dphi = φ2 − φ1` 与 `dlam = radians(lon2 − lon1)`；③ 按上式算 `a`；④ 把 `a` 夹到 [0, 1] 做数值保护（避免浮点误差让 `asin` 定义域越界）；⑤ 返回 `2R·asin(√a)`。
- **复杂度**：时间 O(1) / 空间 O(1)。
- **参数**：`lat1, lon1`（起点纬度、经度，**度**，北纬/东经为正）；`lat2, lon2`（终点纬度、经度，度）。四个参数都无默认值、不校验范围（纬度传成弧度或用度的数值直接代入都不会报错，只会给出荒谬值）。返回 float 单值（不是字典），单位 km；半径口径由模块常量 `EARTH_RADIUS_KM` 提供，**不可通过参数改**。
- **陷阱**：① **必须先把度转成弧度**：直接把度代入 `sin` 会得到"看起来像数字"的荒谬结果且不报错；② haversine 假设地球是正球，长距离（> 1000 km）相对 WGS-84 椭球约 **0.3%** 误差，跨国航线/卫星轨迹类题目不能直接下结论（要更高精度用 Vincenty 或 WGS-84）；③ 经度差跨越 ±180°（换日线）时 `Δλ = lon2 − lon1` 会接近 360°，但本函数用 `sin²` 形式，天然对 `Δλ + 360°` 不变，所以实际不受影响——换成 law of cosines 写法才会踩这个坑；④ 只算"直线距离"，不含道路约束，路网距离要用最短路算法（见 graphs 模块）。
- **怎么检验**：自测算的是北京 (39.9042, 116.4074) → 上海 (31.2304, 121.4737)，键名 `haversine_bj_sh`（`round(..., 6)` 返回）；判据用对称性/零距离/解析特例更稳：断言 `haversine(a, b) == haversine(b, a)`、同点距离恰为 0、赤道上经度差 90° 的两点距离应为 `πR/2`（手算闭式解）、沿同一经线纬度差 90° 的两点距离同为 `πR/2`；再取两点与 WGS-84 椭球库（geopy / pyproj）对拍，断言相对差在 0.3% 量级内（长距离时才接近这个量级，短距离应远小于它）。

#### `idw_interpolate(points, values, query, power=2.0)`

- **数学形式**：反距离加权 `w_i = 1 / d_i^p`，`ŷ(x) = Σ w_i y_i / Σ w_i`，`d_i = ‖x − x_i‖`（平面欧氏距离）；当查询点与某样本点重合（d = 0）时直接返回该样本值，避免 1/0。
- **步骤**：① `_as_points` 校验 `points`、`as_vector` 校验 `values`、`check_same_length` 校验样本点与值的长度一致；② `_as_points` 校验 `query`；③ `power <= 0` 抛 `ValueError`；④ 用广播一次算出 (m, n) 距离矩阵；⑤ 逐查询点：若存在 `d <= 1e-12` 的样本点（即重合），直接取该样本值；否则算权重并做加权平均；⑥ 返回形状 (m,) 的 float 数组。
- **复杂度**：时间 O(nm)（距离矩阵 + 逐点加权）/ 空间 O(nm)（距离矩阵）。
- **参数**：`points`（样本点，形状 (n, 2)）；`values`（样本值，长度 n，必须与 points 行数一致）；`query`（待插值点，形状 (m, 2)，可以只有一行）；`power`（距离幂次 p，默认 **2.0**，必须 > 0，docstring 说明通常取 1~4）。返回 `np.ndarray` 形状 (m,)，无字典键。
- **陷阱**：① **IDW 是平滑器而不是插值器**：它永远不产生比样本极值更大/更小的值（凸性），会抹平极值，"污染峰值/最高点"类预测系统性偏低；② `power` 越大越"贴点"（趋近最近邻、出现牛眼状台阶），越小越平滑，论文里应做 p 的敏感性分析（例如 p = 1, 2, 3 的交叉验证误差）；③ 不含地形等协变量，只依赖距离，采样密度不均时会向密集区偏；④ 距离用平面欧氏距离，量纲必须与坐标一致，用经纬度必须先换算成公里；⑤ 重合判据是固定的小距离容差，同一位置有两个及以上样本点时只取**第一个**命中的那个值。
- **怎么检验**：自测用单位正方形四角、样本值 `[1, 2, 4, 3]`、`power = 2.0`：`idw_at_sample`（查询点取 (1, 1) 与样本重合）应**精确等于**样本值，`idw_center`（查询 (0.5, 0.5)）由四角对称权重的闭式解给出——中心到四角等距，故预测值应恰为 `(1+2+4+3)/4 = 2.5`（手算实例，可实现后逐位核对）。补充检验：断言输出恒落在 `[min(values), max(values)]` 内（凸性）；把最近样本的值改大，断言查询点预测值单调不减（幂次插值的单调性）；`power → ∞` 时结果应收敛到最近邻值（用大 p 近似验证）；`power` 取 0 或负数断言抛 `ValueError`。

#### `ordinary_kriging(points, values, query, model="spherical", nugget=0.0, sill=None, vrange=None)`

- **数学形式**：以半变异函数 γ(h) 为工具，对每个预测点 x0 解克里金方程组 `[Γ 1; 1ᵀ 0]·[w; μ] = [γ0; 1]`，其中 `Γ_ij = γ(‖x_i − x_j‖)`（对角元按理论定义取 γ(0) = 0）、`γ0_i = γ(‖x_i − x0‖)`；预测值 `= Σ w_i z_i`，克里金方差 `= Σ w_i γ0_i + μ`（截断为 ≥ 0）。四种模型（h 为距离，`hr = h/vrange`，`partial = max(sill − nugget, 0)`，γ(0) 恒为 0）：**spherical** `γ = nugget + partial·(1.5hr − 0.5hr³)`（hr < 1，否则 1）；**exponential** `γ = nugget + partial·(1 − e^{−3hr})`；**gaussian** `γ = nugget + partial·(1 − e^{−3hr²})`；**linear** `γ = nugget + partial·min(hr, 1)`。
- **步骤**：① 校验样本点/值/查询点（`_as_points` + `as_vector` + `check_same_length`），样本数 < 2、`nugget < 0` 抛 `ValueError`；② 算样本间距离矩阵，`max_dist <= 0`（全重合）抛 `ValueError`；③ `sill is None` 时取 `np.var(values)`，`vrange is None` 时取 `0.5 · max_dist`，并检查 `sill >= 0`、`vrange > 0`、`sill >= nugget`；④ 若 `sill <= 0`（样本值全相等）直接返回"处处等于该常数、方差全 0"；⑤ 组装 (n+1, n+1) 的 Γ 增广矩阵（前 n 行/列是 Γ，最后一行/一列是 1，右下角 0）；⑥ 逐预测点构造右端 `[γ0; 1]`，用 `np.linalg.solve` 解出 `[w; μ]`，奇异时把 `LinAlgError` 包成 `ValueError` 抛出；⑦ 返回 `prediction` 与 `np.maximum(var, 0)`。
- **复杂度**：时间 O(m n³)（每个预测点一次 n+1 阶稠密解方程，Γ 只组装一次）/ 空间 O(n² + mn)。
- **参数**：`points`（样本点，(n, 2)，n ≥ 2）；`values`（样本值，长度 n）；`query`（待预测点，(m, 2)）；`model`（半变异函数模型，默认 `"spherical"`，仅支持 `spherical` / `exponential` / `gaussian` / `linear`，其余抛 `ValueError`）；`nugget`（块金常数 c0，默认 **0.0**，必须 ≥ 0）；`sill`（**总**基台值，含 nugget；默认 None → `np.var(values)`，ddof = 0）；`vrange`（变程，相关性的空间尺度；默认 None → 样本点最大两两距离的一半）。返回 dict，键为 `prediction`（形状 (m,) 的预测值）与 `variance`（形状 (m,) 的克里金方差，已截断为 ≥ 0）。
- **陷阱**：① **默认参数只是粗糙估计**：sill 用样本方差、变程用"最大距离的一半"只为让一行调用能跑起来，真正做法是先拟合实验变异函数或交叉验证选参，要严格控制就显式传 `sill`/`vrange`；② **块金常数的口径决定"是否精确过样本点"**：本实现按半变异函数理论定义取 γ(0) = 0，块金只出现在 h > 0 处，因此**无论 nugget 取多少，预测值都精确等于样本观测值、样本点处方差为 0**（自测实测：四角数据 nugget = 0.3 时样本点最大偏差仍为 **1.1e-16**）；若采用"观测误差"口径（把块金放到 Γ 对角元上），块金才会让预测变平滑、样本点处不再相等，两种口径必须在论文里写明；③ **普通克里金假设无全局趋势**，数据有明显漂移（如自西向东升高）应改用泛克里金或先去趋势面；④ 矩阵可能病态：样本点极密 + 高斯模型时 Γ 接近奇异，`np.linalg.solve` 会报错或给出巨大权重，应拉开采样间距、加 nugget 或改用带惩罚的解法；⑤ 只用 `model` 一个字符串选模型，模型选错（如周期性数据用球状模型）会"看起来光滑但错"；⑥ `sill` 被当成**含 nugget 的总基台**，`sill < nugget` 直接报错——用默认 `sill=None` 且 `nugget > np.var(values)` 时也会报这个错。
- **怎么检验**：自测口径（四角样本、`model="spherical"`、`nugget=0.0`）：`krige_max_dev_at_sample`（在样本点上预测）应 ≈ 0、`krige_var_at_sample` 也应 ≈ 0，两者都以 `round(..., 10)` 返回可断言；`krige_keys` 断言返回键恰为 `["prediction", "variance"]` 排序后的两个键；`krige_pred_center` / `krige_var_center` 是中心点的预测与方差，`krige_sill` / `krige_vrange` 印证默认参数估计。补充检验：**精确插值性**（对任意 nugget，样本点处预测 = 观测、方差 = 0，这正是区分两种块金口径的判别性实验）；**权重归一性**（解出的 w 之和应为 1，即无偏约束）；用 `model="spherical"` 与 `h → 0+` 处 γ 的极限值核对块金跳变；把 `vrange` 调大断言预测趋向更平滑（方差下降）；样本值全相同时断言返回常数与零方差。

#### `estimate_variogram_params(points, values, nugget=0.0)`

- **数学形式**：粗估法 `sill = var(values)`（ddof = 0）、`vrange = max‖x_i − x_j‖ / 2`，`nugget` 原样回传；这三者正好就是 `ordinary_kriging` 在未显式给参时用的默认值。
- **步骤**：① `_as_points` / `as_vector` / `check_same_length` 校验；② `nugget < 0` 抛 `ValueError`；③ 样本数 < 2 抛 `ValueError`；④ 算样本间距离矩阵，`max_dist <= 0`（全重合）抛 `ValueError`；⑤ 返回 `{"sill": np.var(vals), "vrange": 0.5 · max_dist, "nugget": float(nugget)}`。
- **复杂度**：时间 O(n²)（完整距离矩阵）/ 空间 O(n²)。
- **参数**：`points`（样本点，(n, 2)，n ≥ 2，全重合会报错）；`values`（样本值，长度 n，须与 points 行数一致）；`nugget`（块金常数，默认 **0.0**，必须 ≥ 0，**不参与估计**、只原样回传，docstring 说它"仅用于说明 sill 与 nugget 的关系"）。返回 dict，键为 `sill`（总基台值 = 样本方差，ddof = 0）、`vrange`（变程 = 最大两两距离的一半）、`nugget`（原样回传）。
- **陷阱**：① 这个估计**没有任何统计学最优性**：样本点分布范围越大变程估计越大、预测越平滑，要可靠参数必须拟合实验变异函数或做交叉验证；② 它是 O(n²) 的稠密距离矩阵，n 大时本身就是瓶颈，且只为了取一个最大值；③ 返回的 `sill` 是样本方差、`nugget` 是独立回传的常数，而 `ordinary_kriging` / `_variogram_model` 的口径是"sill 为**含 nugget 的总基台**"，因此当 `nugget > 0` 时把这里返回的 `(sill, nugget)` 直接喂给克里金，实际参与计算的 partial 基台会被削成 `max(sill − nugget, 0)`，不再是样本方差；④ 样本值全相等时 `sill` 为 0，克里金会走"返回常数、方差为 0"的特殊分支。
- **怎么检验**：自测用四角样本（样本值 `[1, 2, 4, 3]`）返回 `krige_sill` / `krige_vrange`（各 `round(..., 6)`）；手算核对：该样本方差 `var([1,2,4,3]) = ((1−2.5)² + (2−2.5)² + (4−2.5)² + (3−2.5)²)/4 = (2.25 + 0.25 + 2.25 + 0.25)/4 = 1.25`，而单位正方形四角的最大两两距离是面对角线 `√2`，故 `vrange` 应为 `√2/2 ≈ 0.707107`——这两条都可先手算再逐位断言。补充检验：断言这两个值与 `ordinary_kriging(..., sill=None, vrange=None)` 内部实际用的参数一致（用一个"显式传参 = 用估计值"的等效性实验验证）；`nugget` 回传值必须与入参完全相等；全重合点集、单点集、`nugget < 0` 三种输入断言抛 `ValueError`。

#### `segment_intersection(p1, p2, p3, p4, eps=1e-12)`

- **数学形式**：记 `r = p2 − p1`、`s = p4 − p3`、`qp = p3 − p1`，叉积 `cross(u, v) = u_x·v_y − u_y·v_x`。**平行判据**（相对形式）`|cross(r, s)| <= eps·|r|·|s|`（零长度线段自动落入此支）；非平行时解参数方程 `p1 + t·r = p3 + u·s`，得 `t = cross(qp, s)/cross(r, s)`、`u = cross(qp, r)/cross(r, s)`，两者都在 [0, 1] 内才算相交；平行时先判共线 `|cross(qp, ref)| <= eps·|qp|·|ref|`（`ref` 取 r，r 退化时取 s），共线则把四个端点投影到 `|ref|` 较大的坐标轴上比较两个闭区间。
- **步骤**：① `as_vector` 逐个转四个端点并校验长度均为 2；② `eps <= 0` 抛 `ValueError`；③ 由四端点的最大绝对坐标算出绝对长度容差，并预先算 `r`、`s`、`qp`、三个叉积与长度；④ 走平行分支：不共线 → `intersect=False`；两个零长度线段 → 比较两点是否重合；一个零长度 → 退化为"点是否在另一线段上"；都非退化 → 投影到主坐标轴比较区间，交非空即相交，重叠长度大于容差则 `collinear_overlap=True`，交点为**重叠区间的中点**；⑤ 走非平行分支：用固定参数容差 `param_tol = 1e-9` 判 `t, u ∈ [−tol, 1 + tol]`，命中则交点为 `p1 + t·r`。
- **复杂度**：时间 O(1) / 空间 O(1)。
- **参数**：`p1, p2`（第一条线段的两个端点，各为长度 2 的 `(x, y)`，无默认值，长度不为 2 抛 `ValueError`）；`p3, p4`（第二条线段的两端点，同上）；`eps`（**相对**容差，默认 **1e-12**，用于"是否平行"与"是否共线"的判据，必须 > 0）。返回 dict，键为 `intersect`（bool，两线段是否有公共点，含只有一个公共端点的情形）、`point`（相交时的交点，形状 (2,)；**不相交时为 `None`**；共线重叠时返回重叠区间的中点，重叠退化为一点时即该点）、`parallel`（bool，方向平行，含共线与零长度线段）、`collinear_overlap`（bool，仅当共线且重叠长度**严格为正**时为 True；共线但只接触一点时为 False）。
- **陷阱**：① **"相交"必须区分"一个公共点"与"一段公共线段"**：平行且共线的两线段即使重叠，叉积判据也永远给不出唯一交点，只写"解方程求交点"的实现会除零或给 NaN，本函数用 `collinear_overlap` 显式区分；② 浮点数据的共线几乎从不是精确的：`eps` 取太大（如 1e-6）会把近距离平行但不共线的线段误判为共线，取太小则真实共线数据（由计算得来的坐标）判不出来；`eps` 是相对容差、与坐标量纲无关，但坐标整体幅值极大/极小时仍建议复核；③ `intersect` 只回答"是否有公共点"，不回答"内部是否穿越"——T 形接触（端点落在另一线段内部）与端点重合同样返回 True，要区分请检查交点是否等于某个端点；④ 交点由第一条线段的参数式 `p1 + t·r` 算出，`t` 略超出 [0, 1] 时交点会落在第一条线段外一点点；参数容差 **1e-9**（线段长度的十亿分之一）意味着"距端点 1e-9·|r| 以内"的接触也算相交，**而这个容差是写死的常量、不随 `eps` 改变**——想通过调 `eps` 放松接触判定是无效的。
- **怎么检验**：自测覆盖了五种退化情形，全部有确定结论可直接断言（交点按 `round(..., 12)` 返回）：平行不共线 (0,0)-(1,0) 与 (0,1)-(1,1) → `intersect=False`、`parallel=True`、`collinear_overlap=False`、`point is None`；共线重叠 (0,0)-(2,0) 与 (1,0)-(3,0) → 三个标志都为 True 且交点应为 **(1.5, 0.0)**（重叠区间的中点，手算）；共线仅接触一点 (0,0)-(1,0) 与 (1,0)-(2,0) → `intersect=True` 但 `collinear_overlap=False`；端点相交 (0,0)-(1,0) 与 (1,0)-(1,1) → 交点 **(1.0, 0.0)** 且 `parallel=False`；T 形 (0,0)-(2,0) 与 (1,0)-(1,1) → 交点 **(1.0, 0.0)**；正常交叉 (0,0)-(2,2) 与 (0,2)-(2,0) → 交点 **(1.0, 1.0)**。补充检验：断言**对称性** `intersect`、`parallel`、`collinear_overlap` 在交换两条线段后不变（交点可能不同，因为交点取自第一条线段的参数式）；把同一组线段整体放大/平移后断言四个标志不变（尺度不变性）；零长度线段（p1 == p2）退化为"点在线上"，与 `point_to_segment_distance(p, ...) == 0` 的结论交叉对拍。

#### `minimum_enclosing_circle(points)`

- **数学形式**：最小包围圆的确定性（不洗牌）Welzl 增量法。圆由 **2 个点**（以两点连线为直径）或 **3 个点**（外接圆）唯一决定：两点圆圆心 `(a+b)/2`、半径 `|a−b|/2`；三点圆用外接圆闭式解（分母 `d = 2·(a_x(b_y − c_y) + b_x(c_y − a_y) + c_x(a_y − b_y))`，即两倍有向面积），三点近似共线时退化为"最远两点"的直径圆。
- **步骤**：① `_as_points` 校验；② n == 1 直接返回"圆心 = 该点、半径 0.0"；③ 取前两点构成直径圆作为初始圆；④ 依次加入第 i 个点（i 从 2 到 n−1）：在圆内则跳过，否则第 i 点必在新圆边界上 → 重置为"以 i 为圆心、半径 0"的退化圆；⑤ 对 j < i 若点 j 不在圆内，则新圆必同时过 i 与 j → 取两点直径圆；再对 k < j 若点 k 不在圆内，则新圆是 i、j、k 的外接圆；⑥ 内点判据用 `dist <= radius + 1e-12·max(1, radius)`，近共线判据用 `|d| <= 1e-12·L²`（L 为三点最大边长）。
- **复杂度**：时间 O(n³) 最坏（**本实现不做随机洗牌，因此没有"期望 O(n)"的保证**）/ 空间 O(n)。
- **参数**：只有 `points`（形状 (n, 2) 的点集，n ≥ 1，无默认值）。没有可调参数，容差全部写死。返回 dict，键为 `center`（形状 (2,) 的圆心坐标）与 `radius`（float，n == 1 时为 0.0）。
- **陷阱**：① **不洗牌 = 放弃期望线性时间**：Welzl 的 O(n) 期望复杂度依赖随机化，这里为满足"两次调用逐位一致"而固定输入顺序，最坏 O(n³)；n ≤ 数千仍够用，n 很大时可自行传入已按固定种子打乱的点集（结果与输入顺序无关，只有耗时有关）；② **最小包围圆只由 2 或 3 个点唯一决定**，半径是"脆"的量——输入中一个远点就会换掉决定圆的那组点，用它做断言应选正多边形顶点这类对称点集；③ 近共线的三点外接圆半径会趋于无穷，本实现用相对判据切到直径圆分支，否则会得到巨大的圆心与半径；④ 点必须互不相同：重复点不破坏算法（半径仍正确），但会让"内部"判定全部落在边界上；⑤ 返回的 `center` 是引用自入参副本的数组，比较时必须用数值比较而非 `is`。
- **怎么检验**：自测给了两条**闭式解**，都能手算（容差 1e-9）：边长 1 的正三角形 (0,0)/(1,0)/(0.5, √3/2)，最小包围圆半径应为 `1/√3 ≈ 0.577350`、圆心应为 `(0.5, √3/6 ≈ 0.288675)`（外接圆）；单位正方形的半径应为半对角线 `√2/2 ≈ 0.707107`、圆心为 `(0.5, 0.5)`，并且四个顶点到圆心的距离都不超过半径（覆盖性断言）；两点 (0,0)/(2,0) 的圆心应为 `(1.0, 0.0)`、半径为 1.0。补充检验：**覆盖性**（对随机点集断言所有点满足 `dist <= radius`，用返回的 `mec_square_covers_all` 式判据）；**最小性**（把半径缩小一点点，断言至少有一个点跑到圆外）；**顺序不变性**（打乱输入点顺序两次，断言圆心与半径逐位相同）；n == 1 与 n == 2 的退化返回值逐位断言。

#### `polygon_centroid(polygon)`

- **数学形式**：面积加权质心（鞋带公式的质心版）。记 `cross_i = x_i·y_{i+1} − x_{i+1}·y_i`（下标模 n 回绕）、`A = 0.5·Σ cross_i`，则 `Cx = Σ (x_i + x_{i+1})·cross_i / (6A)`、`Cy = Σ (y_i + y_{i+1})·cross_i / (6A)`。退化判据：设 L 为包围盒对角线长度，`L = 0` 或 `|A| <= 1e-12·L²` 时抛 `ValueError`。
- **步骤**：① `_as_points` 校验；② n < 3 抛 `ValueError`；③ `np.roll` 造出下一个顶点，算 `cross`、`signed_area = 0.5·Σ cross`；④ 由 x、y 的极差算包围盒对角线 `extent`，退化即抛 `ValueError`；⑤ 按上式算 `cx`、`cy`；⑥ 返回 `{"centroid": np.array([cx, cy]), "signed_area": signed_area}`。
- **复杂度**：时间 O(n) / 空间 O(n)。
- **参数**：只有 `polygon`（形状 (n, 2) 的顶点，按顺序给出，顺/逆时针都行，n ≥ 3，否则 `ValueError`）。没有可调参数（退化阈值写死为相对判据，docstring 给出 `1e-12·L²`）。返回 dict，键为 `centroid`（形状 (2,) 的质心坐标）与 `signed_area`（float，**带符号**面积，逆时针为正、顺时针为负）。
- **陷阱**：① **顶点平均 ≠ 质心**：把顶点坐标直接求平均只在正多边形等特殊情形下成立，对一般"细长"或顶点疏密不均的多边形会明显偏；② **自交多边形的质心无意义**：分母里的带符号面积会被正负部分抵消，本函数只在 `|A|` 退化到接近 0 时报错——面积恰好抵消为 0 的自交图形（对称"8 字"）会抛异常，但面积不为 0 的自交图形不会，返回值不可解释，使用前应先检查多边形是否简单；③ 首尾重复写一次闭合点不影响结果（cross 贡献为 0），但会多计一个顶点；④ 阈值是**相对**的（`1e-12·L²`）：坐标整体缩放不改变结论，但极扁多边形（面积远小于 L²）会被判为退化并抛错，这是有意保护；⑤ 返回的 `signed_area` 与 `polygon_area`（取绝对值）在顺时针输入下**符号相反**，交叉比较时不要直接用等号。
- **怎么检验**：自测三条（容差 1e-12）：单位正方形（逆时针）的质心应为 **(0.5, 0.5)**、`signed_area` 应为 **1.0**；三角形 (0,0)/(4,0)/(0,3) 的质心按"三角形质心 = 三顶点平均"手算应为 **(4/3, 1)**、带符号面积应为 **6.0**（`centroid_triangle` / `centroid_triangle_signed_area` 都以 `round(..., 12)` 返回）。补充检验：把顶点顺序反转，断言质心不变而 `signed_area` 变号（这也是符号口径的判别性实验）；纯平移多边形质心应同步平移；对多边形做正仿射（绕原点缩放 k 倍）后质心应缩放 k 倍；与"按质心定义做数值积分（高密度栅格平均）"的近似结果对拍；`|A|` 退化与 n < 3 两种输入断言抛 `ValueError`。

#### `point_to_segment_distance(p, a, b)`

- **数学形式**：点到**线段**（不是直线）的最短欧氏距离。设 `ab = b − a`：若 `|ab|² = 0`（线段退化为点）返回 `|p − a|`；否则 `t = ((p − a)·ab)/|ab|²`，把 t **截断到 [0, 1]** 得最近点 `a + t·ab`，返回 `|p − (a + t·ab)|`。
- **步骤**：① `as_vector` 转三个坐标并逐个校验长度为 2（否则 `ValueError`）；② 算 `ab` 与其模长平方；③ `denom <= 0` 直接返回 `hypot(p − a)`；④ 否则算 t、`min(1, max(0, t))` 截断、算出投影点；⑤ 返回投影点到 p 的距离。
- **复杂度**：时间 O(1) / 空间 O(1)。
- **参数**：`p`（待测点，长度 2 的 `(x, y)`）；`a, b`（线段的两个端点，各为长度 2 的坐标）；三者都无默认值，长度不为 2 即 `ValueError`。返回 float 标量（非字典、非数组），最短距离，非负；**不返回最近点坐标**。
- **陷阱**：① **不要漏掉 t 的截断**：直接算 `|(p−a)×ab|/|ab|` 得到的是到**直线**的距离，点在端点外侧时会严重偏小（docstring 举的反例：点到其正后方 100 km 的线段，直线距离可能是 0，真实距离是 100 km），做"最近道路距离"时后果很大；② 经纬度不能直接代入：Δ经度的地面长度随纬度收缩，必须先投影或用球面距离；③ 返回标量而非点坐标，需要最近点本身要自己按 t 重算；④ 若调用方已知道点在直线哪一侧，用截断形式仍然正确、无需分支。
- **怎么检验**：自测三条（容差 1e-12）：(2, 0) 到 (0,0)-(1,0)，投影落在线段**外**，距离应等于到端点 b 的距离 **1.0**；(−3, 4) 到 (0,0)-(1,0)，距离应等于到端点 a 的距离 **5.0**（3-4-5 直角三角形）；(0.5, 3.0) 到 (0,0)-(1,0) 的垂直距离手算应为 **3.0**；退化线段 (1,1)-(1,1) 时 (3,4) 到它的距离手算应为 **√13 ≈ 3.605551**（`pt_seg_*` 各键都以 `round(..., 9)` 返回）。补充检验：**截断性判别实验**——固定线段 (0,0)-(1,0)，把 p 的 x 从 −5 扫到 6，断言距离先等于到 a 的距离、在线段正上方段等于垂直距离、之后等于到 b 的距离（三段折线，断点在 x = 0 与 x = 1）；对同一个点断言"到线段的距离 ≥ 到端点距离的较小者"永真；把 p、a、b 整体平移/旋转后距离不变（刚体不变性）；与 `segment_intersection` 的零长度退化分支对拍（点在线上时距离为 0）。

#### `voronoi_nearest(points, queries)`

- **数学形式**：暴力 Voronoi 最近站点查询。构造 (m, n) 距离矩阵 `D_jk = ‖q_j − p_k‖`，对每行取最小值下标：`index_j = argmin_k D_jk`、`distance_j = min_k D_jk`。
- **步骤**：① `_as_points` 校验站点集与查询点集（查询点可以只有一行）；② 用广播一次算出全部查询点到全部站点的距离；③ `np.argmin(dist, axis=1)` 取最近站点下标并转成 int；④ 用花式索引取出对应距离；⑤ 返回 `{"index": ..., "distance": ...}`。
- **复杂度**：时间 O(mn) / 空间 O(mn)（完整距离矩阵）。暴力实现，n、m 都上千时请改用 KD 树或 Delaunay 对偶。
- **参数**：`points`（站点坐标，形状 (n, 2)，n ≥ 1）；`queries`（查询点，形状 (m, 2)）；两者都无默认值，`_as_points` 会拒绝空集与 NaN/inf。没有可调参数，并列口径（距离完全相同时归给**编号最小的站点**）由 `np.argmin` 的语义固定，不可配置。返回 dict，键为 `index`（形状 (m,) 的 int 数组，最近站点下标）与 `distance`（形状 (m,) 的 float 数组，对应最近距离）。
- **陷阱**：① **Voronoi 胞元只由最近距离定义，不含任何障碍/路网约束**：把站点当设施、查询点当需求点算"最近设施"时，直线距离会系统性低估实际通行距离；② 并列口径必须写清楚：浮点误差会让"本应并列"的点倒向任一侧，边界附近的归属不可依赖，需要稳定归属时应显式加权重或改用距离排序后的规则；③ 站点重复（重合）时只会返回其中下标最小的那个，另一个永远不被查询到；④ 距离是平面欧氏的，用经纬度必须先换算成公里，否则高纬度处东西向距离被高估；⑤ 只返回最近的一个站点，要 k 近邻需自己排序或改用 KD 树接口。
- **怎么检验**：自测站点取单位正方形四角 `[[0,0],[1,0],[1,1],[0,1]]`、查询点 `[[0.6,0.6],[0.1,0.1]]`：`voronoi_index` 应为 **[2, 0]**（(0.6,0.6) 离角点 (1,1) 最近、(0.1,0.1) 离 (0,0) 最近），`voronoi_distance[0]` 应为 `hypot(0.4, 0.4) = 0.4√2 ≈ 0.565685`（手算，`round(..., 9)` 返回）。补充检验：**暴力对拍**（自己写双重循环按行取最小下标，逐元素比较，独立于 `argmin` 的并列语义）；**Voronoi 性质**——对每个查询点断言"它到被选中站点的距离 ≤ 它到任意其他站点的距离"；**胞元划分完备性**——在站点凸包内撒一密网格，断言每个格点恰好被分配给一个站点且分配结果的边界只由站点位置决定；把站点整体平移后断言 `index` 不变（形状保持）；构造严格对称的查询点（如两站点中点）断言并列口径确实返回最小下标。


### 3.11 博弈论 —— `examples/algorithms/game.py`

#### `zero_sum_value_lp`
- **数学形式**：二人零和博弈 `A ∈ R^{m×n}`，约定 `A[i, j]` 是行玩家 i 对列玩家 j 的收益（列玩家最小化）。行玩家用混合策略 `p`，其保证收益为 `min_j (p^T A)_j`，于是求 `max v  s.t.  sum_i p_i A[i, j] >= v (∀j),  sum_i p_i = 1, p >= 0`。LP 标准形要求变量非负，故先平移 `A' = A + shift`（`shift` 取 `-min(A) + 1.0`，若 `min(A) > 0` 则 shift = 0），解完按 `E[A] = E[A'] - shift` 还原博弈值；策略本身不受平移影响。列玩家策略由**强对偶**给出：对同一已平移矩阵显式再解一次对偶 LP（不依赖求解器暴露对偶变量）。
- **步骤**：① `as_matrix` 校验支付矩阵；若 `maximize_row=False` 则先转置，统一成"行玩家最大化"口径，空维度直接 `ValueError`。② 计算 `shift`，得到 `Ashift`；若平移后仍有非正元素则抛 `ValueError`。③ 变量向量 `[p_0..p_{m-1}, v]`，目标 `maximize v`（`c[-1] = 1`），不等式约束 `-sum_i p_i A'[i, j] + v <= 0`，等式约束 `sum_i p_i = 1`，边界 `[(0, None)] * n_var`。④ 调 `simplex_lp(..., maximize=True)`；`status != "optimal"` 直接抛 `ValueError`（不做静默兜底）。⑤ 取 `row = clip(x[:m], 0, None)` 并归一化（和为 0 则抛错），`value = v_shifted - shift * row.sum()`。⑥ 调 `_zero_sum_dual_lp` 解对偶得列策略，`clip` 后归一化；若列策略和 <= 0 则退化为均匀分布 `1/n`。⑦ 自检：要求 `row @ A` 的最小值 >= `value - 1e-6`，且 `A @ col` 的最大值 <= `value + 1e-6`，否则抛 `ValueError`。⑧ 返回字典。
- **复杂度**：时间 = 一次 LP（单纯形实际很快，最坏指数级），另加一次对偶 LP；建表 `O(mn)`，空间 `O(mn)`（`A_ub` 为 `n × (m+1)`）。
- **参数**：`payoff`（`(m, n)` 支付矩阵 / ArrayLike，无默认值，必填；约定 `payoff[i, j]` 为行玩家 i 对列玩家 j 的收益）；`maximize_row: bool = True`（True 表示行玩家最大化、列玩家最小化；若矩阵记的是"列玩家的收益"传 False 会自动转置视角，且返回的 `row_strategy` 仍是**传入矩阵的行**对应的策略）。返回字典键：`"value"`（float，按**原始**矩阵口径的博弈值）、`"row_strategy"`（`np.ndarray(m)`，已 clip 非负并归一化的概率分布）、`"col_strategy"`（`np.ndarray(n)`，同上）、`"lp_status"`（str，求解器状态字符串，正常为 `"optimal"`）。
- **陷阱**：① **绝不假设 value >= 0**，零和博弈的值可以为负；直接把 LP 目标当 value、忘了还原平移就会偏。② 最优混合策略可能**不唯一**（价值相等时，例如 `[[1,-1],[-1,1]]` 任意 p 都最优），LP 给的只是其中一个，论文里声称"唯一"需额外论证。③ 平移量必须让**所有**元素为正，矩阵含大负数时 `shift` 要取 `-min(A) + kappa`，不要硬编码 1.0（否则得到负元素，LP 报错或给错解）。④ 零和博弈可行域有界，理论上总可行有界；一旦求解器返回非 optimal 状态，通常说明数值尺度或平移有问题，本实现直接抛 `ValueError`。⑤ `maximize_row=False` 时函数内部转置求解，返回的 `row_strategy` 仍是传入矩阵的行，口径不要搞反。⑥ 列策略是**另解一次对偶 LP** 得到的，不是从单纯形表取对偶行——代价是多一次求解，好处是不依赖求解器内部实现。
- **怎么检验**：`_self_test()` 给了两组解析解算例。① 配对硬币 `[[1.0, -1.0], [-1.0, 1.0]]`：`value` 应为 **0**，行/列策略均 **(0.5, 0.5)**（黄金键 `mp_value` / `mp_row` / `mp_col`，均 `round(..., 9)`），且 `mp_status` 应为 `"optimal"`。② `[[3.0, -1.0], [-2.0, 1.0]]`：手工可算 `value = 1/7`、行策略 `(3/7, 4/7)`、列策略 `(2/7, 5/7)`（黄金键 `g2_value` / `g2_row` / `g2_col`，对照键 `g2_value_expected` / `g2_row_expected` / `g2_col_expected`，均 `round(..., 9)`）。③ 通用自洽检验：算 `row @ A` 的最小值应 >= value、`A @ col` 的最大值应 <= value（函数内部就用 `1e-6` 容差做了这一步，不满足会抛异常）。④ 再验概率合法性：`row >= 0`、`col >= 0`、`row.sum() == 1`、`col.sum() == 1`。

#### `nash_support_enumeration`
- **数学形式**：双矩阵（2 人一般和）博弈 `(A, B)`，`A, B ∈ R^{m×n}`，`A[i, j]` 是行玩家收益、`B[i, j]` 是列玩家收益。纳什均衡 `(p, q)`（`p ∈ Δ^m`、`q ∈ Δ^n` 为概率分布）满足：`(A q)_i` 在 `S_r = supp(p)` 上取到全局最大（行玩家不在支撑上的纯策略都不更优），`(B^T p)_j` 在 `S_c = supp(q)` 上取到全局最大。求的是**全部**纳什均衡集合。
- **步骤**：① `as_matrix` 读入 `A`、`B`，形状必须一致，否则 `ValueError`；空维度也抛错；`tol = _TOL`。② 枚举所有非空支撑对 `(S_r, S_c)`，`|S_r|`、`|S_c|` 各自从 1 到 `min(m, n)`（否则线性方程组未知数多于方程，退化处理）。③ 对每个支撑对调 `_solve_support`：在"行玩家在 `S_c` 上无差异、列玩家在 `S_r` 上无差异"的线性系统里用 numpy 最小二乘 + 秩检查求解候选 `(p, q)`，无解返回 `None` 跳过。④ 调 `_is_nash` 回代检验全局最优性（对非支撑纯策略做 `<= tol` 的浮点比较），并对负概率 `np.clip(0)` 后重新验证。⑤ `_append_unique` 按容差去重（同一均衡可能由多个支撑对生成）。⑥ 按（行支撑大小, 列支撑大小, 行策略 `round(..., 9)` 字典序, 列策略 `round(..., 9)` 字典序）稳定排序后返回。
- **复杂度**：时间 `O(2^m * 2^n * (m n + 线性方程组求解))`，**只对小规模可行**：`m=n=3` 约 `2^6 = 64` 个支撑对、秒级；`m=n=6` 为 4096 对、仍可接受；`m=n=10` 会涨到约 `1e6` 对，必须换 Lemke-Howson 或支撑集剪枝。空间 `O(m n)`。
- **参数**：`A`（`(m, n)` 行玩家收益矩阵，必填，无默认值）；`B`（`(m, n)` 列玩家收益矩阵，同形状，必填；零和博弈应设 `B = -A`，对称博弈常见 `B = A.T` 且 A 需为方阵）。内部参数：`tol = _TOL`（模块级常量 `_TOL = 1e-7`）。返回：`list of dict`，每个元素含键 `"row"`（`np.ndarray(m)`，非负、和为 1 的概率分布）与 `"col"`（`np.ndarray(n)`，同上）；列表可能为空（精确算法下有限博弈必有均衡，空列表只说明该容差下没找到数值解，应调大容差或检查输入，而不是断言"均衡不存在"）。
- **陷阱**：① **容差是最大的坑**：均衡判定用 `<= tol` 浮点比较，太小会漏掉真实混合均衡（方程解出 `1e-9` 级负概率），太大又会把非均衡判成均衡；本实现取 `tol=1e-7` 并对负概率 clip 后重新验证全局最优性。② **完备性有条件**：退化博弈（某纯策略收益在多个点并列最大）中均衡支撑可能非最小，同一均衡会被重复生成，也可能出现"支撑对无解但均衡存在"的假象；去重与回代能兜住大部分，严重退化问题请交叉验证。③ **`B` 的口径容易搞反**：`B[i, j]` 必须是列玩家在组合 `(i, j)` 下的收益，不是"列玩家的损失矩阵"，搞反会得到镜像的错误均衡。④ 返回的是**均衡集合**不是"最优解"，多均衡间无法用收益比较，论文里要讨论均衡选择（风险占优、帕累托占优）。⑤ 纯策略均衡走同一条路径（支撑大小为 1），返回的仍是混合策略向量（退化成 one-hot），不要指望拿到"行号"。
- **怎么检验**：`_self_test()` 给了两个已知答案算例。① **囚徒困境**：`pd_A = [[-1, -5], [0, -3]]`、`pd_B = [[-1, 0], [-5, -3]]`（即 C,C = (-1,-1)、C,D = (-5,0)、D,C = (0,-5)、D,D = (-3,-3)），黄金键 `pd_n_eq` 应为 **1**（恰好 1 个纳什均衡，即 (D, D)），`pd_row0` / `pd_col0` 为 6 位小数舍入的均衡策略，应为 one-hot。② **性别战**：`bos_A = [[2, 0], [0, 1]]`、`bos_B = [[1, 0], [0, 2]]`（(O,O) = (2,1)、(O,T) = (0,0)、(T,O) = (0,0)、(T,T) = (1,2)），黄金键 `bos_n_eq` 应为 **3**（2 纯 + 1 混合），`bos_n_pure` 应为 **2**（用 `np.count_nonzero(...) > _TOL` 数的单点支撑），`bos_mixed_row` / `bos_mixed_col` 为排序后最后一个（混合）均衡的 9 位小数策略。③ 通用检验：对每个返回的 `(row, col)` 验 `row >= 0`、`col >= 0`、两者和为 1，并手工回代 `(A @ col)` 的最大值是否被 `supp(row)` 取到、`(B.T @ row)` 的最大值是否被 `supp(col)` 取到。④ 排序稳定性：同一输入重复调用应逐位相同。

#### `shapley_value`
- **数学形式**：n 人合作博弈的特征函数 `v: 2^N → R`，Shapley 值 `phi_i = sum_{S ⊆ N\{i}} |S|! (n-|S|-1)! / n! * [v(S ∪ {i}) - v(S)]`（即玩家 i 对所有联盟的加权边际贡献）。权重 `|S|!(n-|S|-1)!/n!` 由 `_shapley_weight(size, n)` 给出。满足有效性 `sum(phi) == v(全集)`。
- **步骤**：① `n = int(n)`，`n < 1` 抛 `ValueError`；`full = (1 << n) - 1`。② 内部函数 `v(mask)` 统一两种口径并缓存：`mask == 0` **强制返回 0.0**（忽略传入值）；否则把掩码转成 `frozenset`，若 `characteristic` 可调用则先试 `characteristic(key_set)`，`TypeError` 时退化为 `characteristic(mask)`；若是 dict 则先查 `key_set` 再查 `mask`，**查不到直接抛 `ValueError`**（不静默当 0），非 dict/不可调用也抛 `ValueError`；返回非有限值同样抛 `ValueError`。③ 先对 `mask` 从 1 到 `full` 全部求值一遍（fail fast），取 `total = v(full)`。④ `phi = zeros(n)`，遍历 `mask` 从 0 到 `full - 1`（跳过 `mask == full`，此时没有"缺席玩家"，边际贡献恒为 0），`size = bin(mask).count("1")`、`weight = _shapley_weight(size, n)`，对每个不在 `mask` 中的 i 累加 `weight * (v(mask | (1 << i)) - v(mask))`。⑤ 有效性自检：若 `abs(phi.sum() - total) > 1e-6 * max(1.0, abs(total)) * sqrt(n)` 抛 `ValueError`。⑥ 返回 `phi`。
- **复杂度**：时间 `O(n * 2^n)`（枚举 `2^n` 个子集，每个子集对每个玩家取边际贡献；`_shapley_weight` 每次 `O(n)` 阶乘计算，文档口径为 `O(n)`/`O(1)`）；空间 `O(2^n)`（缓存）。`n = 20` 约 `2e7` 次查表尚可，`n = 25` 时 `8e8` 太慢。
- **参数**：`characteristic`（特征函数，dict 或可调用对象，必填无默认值；支持两种等价口径且可混用——(a) **位掩码整数**，玩家 i 对应第 i 位 `player i <-> (mask >> i) & 1`，例如 `v(3) = v({0, 1})`；(b) **`frozenset`**，如 `v(frozenset({0, 1}))`；约定 `v(空集) = 0`，dict 查不到空集按 0 处理，函数返回非 0 不报错但结果会失真）；`n: int`（玩家数，必须 >= 1，玩家编号固定为 `0..n-1`，必填无默认值）。返回：`np.ndarray`，长度 n，`result[i]` 是玩家 i 的 Shapley 值，满足有效性 `sum(result) == v(全集)`（数值误差在 `1e-9` 量级）。
- **陷阱**：① **空联盟必须为 0**：`v(∅) ≠ 0` 会破坏"边际贡献"定义，结果不再满足有效性公理，本实现强制视空集为 0 并忽略传入值。② **特征函数只依赖联盟、不依赖顺序**：把"先来后到"写进 `v` 后 Shapley 值不再适用，应改用带权重的广义 Shapley 或 Shapley-Shubik 指数。③ 复杂度 `n * 2^n`：`n > 20` 建议蒙特卡洛抽样（随机采排列、样本均值近似）并报告标准误。④ 特征函数的**量纲和量级**直接进入结果，`v` 表示"成本节省"时结果单位也是成本，不要写成"收益份额"。⑤ 返回的 Shapley 值**可以为负**（"反贡献者"），不等于份额比例；Shapley 值可能为负而"按比例分配"永远非负，两者结论会打架。⑥ 超可加时 Shapley 值位于核（core）内，任意函数时可能落在核外，"稳定分配"的论断不成立。⑦ `_shapley_weight` 要求 `0 <= size <= n-1`，`size == n` 时 `n - size - 1 == -1` 阶乘未定义，调用方必须跳过全集（Shapley 公式最易写错的下标之一）。
- **怎么检验**：`_self_test()` 给了三个算例（黄金键均 `round(..., 9)`）。① **3 人对称多数博弈**：`v(S) = 1.0 if len(S) >= 2 else 0.0`，调用 `shapley_value(majority3, 3)`，`_self_test` 文档明确"任意 2 人即可获胜 → 每人 1/3"，故 `sh_sym` 应为 `[1/3, 1/3, 1/3]`、`sh_sym_sum` 应为 `1.0`（= `v(全集)`）。② **联合国安理会式投票博弈**：15 个成员（`0..4` 常任有否决权，`5..14` 非常任），通过条件为"`>= 9` 票且 5 个常任全部同意"，总收益 1；调用 `shapley_value(un_sc, 15)`，黄金键 `sh_un_perm`（`sh_un[0]`）、`sh_un_nonperm`（`sh_un[5]`）、`sh_un_sum`（应等于 1）、`sh_un_perm_total`（`sh_un[:5]` 之和，即 5 个常任的权力总量）。③ **位掩码口径应与 frozenset 口径完全一致**：`mask_table = {m: (1.0 if bin(m).count("1") >= 2 else 0.0) for m in range(8)}`，`shapley_value(mask_table, 3)` 的结果应与 ① 的 frozenset 口径**逐位相同**，黄金键 `sh_mask_max_dev` 是两者逐元素最大绝对偏差（`round(..., 12)`，应约为 0）。④ 通用检验：`phi.sum()` 应等于 `v(全集)`（函数内部就用 `1e-6 * max(1.0, |total|) * sqrt(n)` 的容差做了这一步，超限抛异常）；对称玩家应得到相等的值；哑元（对任何联盟边际贡献恒为 0）应得到 0。

#### `gale_shapley`
- **数学形式**：双边一对一稳定匹配。给定男方偏好降序列表与女方偏好降序列表，"阻塞对" `(m, w)` 定义为双方都严格更喜欢对方（单身视为"无限靠后"，被理解为对现状的最差评价）。稳定匹配 = 不含任何阻塞对的匹配。算法是 **Gale-Shapley 延迟接受（男方求婚版）**：每个未匹配男生按自己偏好顺序向女生求婚；女生当前无配偶则暂时接受，否则比较新来者与现任、保留更喜欢的那个，另一人恢复未匹配继续向下一个目标求婚；直到所有男生都有配偶或所有男生把候选都求过一遍。
- **步骤**：① 校验两侧必须是 `Mapping`（dict），否则 `ValueError`。② 取出 `men` / `women` 键集合。③ **偏好净化**：每人的列表做去重，且只保留出现在对方键集合中的候选（不可接受的直接剔除），得到 `prefs_m` / `prefs_w`。④ 建排名表 `rank_w`（`{woman: {man: index}}`，越小越喜欢）、`rank_m`（`{man: {woman: index}}`，用于阻塞对校验；关键是 `prefs_m` 里排在前面的必须 rank 更小）。⑤ `next_choice` 记录每个男生下一个要表白的下标，`match_w` / `match_m` / `free` 维护匹配状态与未匹配队列，`n_proposals` 计数。⑥ 主循环：从 `free` 队首取男生，若表白名单用尽则保持单身跳过；否则向 `lst[next_choice]` 求婚并 `n_proposals += 1`；女生空则接受；否则用 `rank_w[woman].get(man, 10**9) < rank_w[woman].get(current, 10**9)` 判定是否换人（换人时给被踢者 `match_m[...] = None` 并压回 `free`），不换则把求婚者压回 `free` 被拒继续。⑦ 返回前做 **O(n*m) 全对阻塞对校验**，统计 `n_blocking`（正确实现应为 0；单身者当前排名用 `10**9`）。⑧ 返回字典。
- **复杂度**：时间 `O(n * m)`（每个男生最多求婚 m 次，每次 O(1) 比较；额外的阻塞对校验也是 O(n*m)）；空间 `O(n + m)`（偏好字典 + 匹配表）。
- **参数**：`men_prefs`（`{man: [w1, w2, ...]}`，每个男生的偏好降序列表，最喜欢在前，必须是 dict/Mapping，必填无默认值）；`women_prefs`（`{woman: [m1, m2, ...]}`，每个女生的偏好降序列表，必须是 dict/Mapping，必填无默认值）。两边偏好列表必须**互相是对方的全集**（同一组男女、只是顺序不同）；**不要求**人数相等，多出来的一方按偏好列表长度自然处理；偏好列表**可以是不完全列表**（未列出者视为不可接受，不向其求婚）。返回字典键：`"matching"`（`{man: woman 或 None}`，**包含所有男生**，未匹配为 `None`）、`"matches"`（反向字典 `{woman: man}`，只含已匹配上的女生，便于查询）、`"n_proposals"`（int，算法过程中**累计的求婚次数**，同一个男生被拒后再次求婚每次计一次，是衡量算法代价的常用指标）、`"n_blocking_pairs"`（int，返回前**实际校验**得到的阻塞对个数，正确实现应为 0）。
- **陷阱**：① **得到的稳定匹配是谁最优的？** 男方求婚版给出**男方最优**稳定匹配（每个男生在所有稳定匹配中拿到自己最满意的结果），同时是**女方最差**的稳定匹配；论文里说"这就是最优匹配"必须注明是哪一方的口径，换成女方求婚版会得到另一个（可能不同的）稳定匹配。② **稳定匹配不唯一**，但稳定匹配集合的"格结构"保证了男方最优/女方最优两个极端存在。③ **必须校验无阻塞对**：很多实现只跑流程不校验，偏好列表方向搞反（把"最喜欢在前"写成"最不喜欢在前"）时依然能输出一个匹配，只是不稳定——本函数在返回前做全对校验并报告 `n_blocking_pairs`。④ 偏好列表里出现未在对方字典中的人会被当作"不在候选集"忽略；两边人名不是同一套字符串时匹配会大面积落空，请保证键集合一致。⑤ 不完全列表按标准的"incomplete preferences"扩展处理（未列出 = 不可接受）；若希望未列出的视为最次而非不可接受，请显式补全列表。⑥ 男生字典为空时返回空匹配，不报错（边界情形在竞赛里常见）。
- **怎么检验**：`_self_test()` 给了两个实例 + 一段独立暴力复核。① **4 男 4 女已知实例**（`men_prefs` = m1:[w1,w2,w3,w4]、m2:[w2,w1,w3,w4]、m3:[w3,w1,w2,w4]、m4:[w4,w1,w2,w3]；`women_prefs` = w1:[m4,m3,m2,m1]、w2:[m3,m4,m1,m2]、w3:[m2,m1,m4,m3]、w4:[m1,m2,m3,m4]），`_self_test` 文档明确"校验阻塞对为 0"，黄金键 `gs_matching`（`["m->w", ...]` 按男名排序的字符串列表）、`gs_n_proposals`、`gs_n_blocking`（应为 **0**）。② **第二个 3×3 实例**（m1:[w1,w2,w3]、m2:[w2,w1,w3]、m3:[w1,w2,w3]；w1:[m2,m1,m3]、w2:[m1,m2,m3]、w3:[m1,m2,m3]），设计意图是"男生会**被拒**"、真正跑通拒绝分支，黄金键 `gs2_matching` / `gs2_n_proposals` / `gs2_n_blocking`（应为 0）。③ **独立暴力复核**：`_self_test` 用一段与 `gale_shapley` 内部实现**完全无共享代码**的枚举代码，对每一对 `(男, 女)` 直接用原始偏好列表判断是否互相更喜欢，黄金键 `gs2_pairs_checked`（检查的对数）与 `gs2_brute_blocking`（应为 0）。④ **男方最优性**：黄金键 `gs2_m1_gets_first_choice` 应为 `True`，即 `gs2["matching"]["m1"] == men_prefs2["m1"][0]`。⑤ 通用检验：枚举所有非配对 `(男, 女)`，若男方更喜欢她（下标更小）且女方也更喜欢他（下标更小、或她单身）则计一个阻塞对，数量必须为 0；再验 `matching` 覆盖全部男生、`matches` 是 `matching` 的一致反向字典（无重婚）。

#### `replicator_dynamics`
- **数学形式**：演化博弈的复制者动态（replicator dynamics）ODE：`dx_i/dt = x_i * [(A x)_i - x^T A x]`，其中 `A[i, j]` 是"我采用策略 i、对手采用策略 j"时我的收益，`x^T A x` 是总体平均收益（对称博弈的"对群体平均"口径）。用经典四阶 Runge-Kutta（RK4）积分。
- **步骤**：① `as_matrix` 读入 `A` 并 `check_square`（须为 `(n, n)` 方阵）。② `as_vector` 读入 `x0`，长度必须等于 `n`（否则 `ValueError`）；要求非负（否则 `ValueError`）、和为正（否则 `ValueError`），然后**内部归一化** `x = x / x.sum()`。③ `T = int(T)`、`dt = float(dt)`；`T < 0` 抛 `ValueError`，`dt <= 0` 抛 `ValueError`。④ 定义向量场 `field(y)`：`fitness = A @ y`，`mean = y @ fitness`，返回 `y * (fitness - mean)`。⑤ 主循环 T 步：依次求 `k1 = field(x)`、`k2 = field(_simplex(x + 0.5*dt*k1))`、`k3 = field(_simplex(x + 0.5*dt*k2))`、`k4 = field(_simplex(x + dt*k3))`，再取 `xn = _simplex(x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4))`；`_simplex` 即"截负 + 归一化"（若截负后和为 0 则返回均匀分布 `1/n`），保证迭代始终停留在单纯形上。⑥ 记录轨迹与时间点 `t[k+1] = t[k] + dt`。⑦ `converged = bool(T >= 1 and |traj[-1] - traj[-2]|_1 < 1e-8)`。⑧ 返回字典。
- **复杂度**：时间 `O(T * n^2)`（每步 4 次矩阵-向量乘）；空间 `O(T * n)`。
- **参数**：`A`（`(n, n)` 收益矩阵，必填无默认值；`A[i, j]` = 我采用 i、对手采用 j 时我的收益）；`x0`（长度 n 的初始策略分布，必填无默认值；必须非负、和 > 0，内部会归一化）；`T: int`（总步数，整数，**必填无默认值**；返回轨迹长度 = `T + 1`，含初始点）；`dt: float = 0.01`（时间步长，默认 0.01；RK4 对它四阶精度，但 dt 过大仍会震荡或越界）。返回字典键：`"t"`（`np.ndarray(T+1)`，时间点，`t[0]=0` 且等距 `dt`）、`"trajectory"`（`np.ndarray(T+1, n)`，每行都是概率分布，非负、和为 1，含初始点）、`"final"`（`np.ndarray(n)`，轨迹最后一行 `traj[-1].copy()`）、`"converged"`（bool，最后两步的 L1 变化 < `1e-8`，是**经验判据**，不等于严格证明收敛到 ESS）。
- **陷阱**：① **dt 不能随手放大**：向量场在边界附近很陡，`dt` 过大（例如 0.1）会让解冲出单纯形（出现负概率），此时"归一化"只是在掩盖积分误差、轨迹形状会失真；本实现默认 0.01 并显式截断负值，但论文里仍应报告 dt 并做一步 `dt/2` 的收敛性对照。② **复制者动态的解不一定是 ESS**：它只保证"纯策略适应度高于平均值的策略占比上升"；内部稳定点（如混合均衡）可能是鞍点或中心——石头剪刀布的内点均衡就是**中性稳定**（围绕它振荡，不发散也不收敛），轨迹取决于初值和 dt，不能说"收敛到均衡"，务必画相图/看 `converged`。③ **初始点落在边界上且该纯策略被严格占优**时演化会停在边界（占比 0 无法回升），被淘汰策略永远不会重新出现（除非加入变异/漂移项）。④ `x^T A x` 是**对称博弈**（同一群体）的口径；两个不同种群的非对称博弈正确写法是双群体版本 `dx_i/dt = x_i((A y)_i - x^T A y)`，本函数不适用。⑤ 收益矩阵整体加常数**不改变**复制者动态（减平均值时被抵消），但乘以正数会改变时间尺度。
- **怎么检验**：`_self_test()` 给了三个确定性算例（本模块无随机性，两次调用逐位相同）。① **协调博弈** `coord = [[2, 0], [0, 1]]`（黄金键 `rep_final_a` / `rep_final_b` 为 `round(final, 6)`，`rep_converged_a` 为 `bool(converged)`，`rep_len = trajectory.shape[0]`）：从 `np.array([0.8, 0.2])` 出发 `T=2000, dt=0.01` 应收敛到策略 0 的纯均衡，从 `np.array([0.2, 0.8])` 出发应收敛到策略 1 的纯均衡——即"初值偏向哪个策略就收敛到哪个纯均衡"（`_self_test` 文档原话）。② **石头剪刀布** `rps = [[0,-1,1],[1,0,-1],[-1,1,0]]`，初值 `np.array([0.5, 0.3, 0.2])`、`T=500, dt=0.01`：用于说明"收敛到均衡 ≠ 稳定"，黄金键 `rep_rps_min`（`round(trajectory.min(), 9)`，因为内部截负，理论上应 >= 0）与 `rep_rps_sum_dev`（`round(|trajectory.sum(axis=1) - 1|.max(), 12)`，应为约 0，验证每行都是概率分布）。③ 通用检验：检查 `trajectory.shape == (T+1, n)`、`t.shape == (T+1,)`、`t[-1] == T*dt`、每行非负且和为 1、`np.allclose(final, trajectory[-1])`；再做 **dt 减半对照**（同 `T*dt` 总时长下 `dt` 与 `dt/2` 的轨迹应接近），以及用已知闭式解/单调场景核对收敛方向。

#### `minimax_alpha_beta`
- **数学形式**：带 `(alpha, beta)` 窗口的递归极小极大（对抗搜索）：最大化节点取子节点值的 `max` 并更新 `alpha = max(alpha, value)`；最小化节点取 `min` 并更新 `beta = min(beta, value)`；一旦 `alpha >= beta` 立即停止遍历剩余子节点（剪枝）。剪掉的子树不可能改变根节点的取值（Knuth & Moore 1975），因此**根节点在全窗口下拿到的值与不剪枝的完全极小极大逐位相同**。
- **步骤**：① 校验 `evaluate_fn`、`children_fn` 必须同时提供，否则 `ValueError`。② `depth = int(depth)`，`depth < 0` 抛 `ValueError`；`a_in = float(alpha)`、`b_in = float(beta)`，若 `not (a_in < b_in)` 抛 `ValueError`（空搜索窗口）。③ 初始化统计 `stats = {"n_evaluated": 0, "n_pruned": 0}`。④ 内部递归 `rec(cur, d, a, b, is_max)`：仅当 `d > 0` 时才调 `children_fn(cur)` 展开子节点；`d == 0` 或子节点为空则 `n_evaluated += 1` 并返回 `(evaluate_fn(cur), None)`；最大化节点遍历子节点保留最大值与对应子节点（用 `v > best` 判优），更新 `a`，`a >= b` 时 `n_pruned += 1` 并 `break`；最小化节点对称处理（`v < best`、更新 `b`）。⑤ 从根节点 `rec(node, depth, a_in, b_in, bool(maximizing))` 得 `(value, best_child)`。⑥ 返回字典。
- **复杂度**：时间最坏（子节点顺序最差）`O(b^d)`，与不剪枝相同；子节点顺序理想时 `O(b^(d/2))`（b 为平均分支因子、d 为深度）。空间 `O(d)`（递归栈，不含 children 列表）。
- **参数**：`node`（搜索起点，必填无默认值；**函数不解释其内部结构**，原样交给 `children_fn` / `evaluate_fn`，可以是棋盘、状态元组、字典键等任意对象）；`depth: int`（还要往下搜索的层数，必填无默认值，必须 >= 0；`depth == 0` 时直接调 `evaluate_fn` 不再展开，"完全搜索"要求 depth 不小于博弈树深度）；`alpha: float = float("-inf")`（最大化者已能保证的下界，**根节点必须传 -inf**，即用默认值）；`beta: float = float("inf")`（最小化者已能保证的上界，**根节点必须传 +inf**，即用默认值）；`maximizing: bool = True`（`node` 处轮到谁走，True 表示轮到最大化者）；`evaluate_fn: Optional[Callable[[object], float]] = None`（叶子或 depth 用尽处的静态评估值，**越大对最大化者越有利**，必须能对任意被展开到的节点求值，必填）；`children_fn: Optional[Callable[[object], Sequence[object]]] = None`（`children_fn(node) -> 子节点序列`，返回空序列表示终局，必填）。返回字典键：`"value"`（float，根节点处 maximizing 方的保证值）、`"best_child"`（maximizing 方在 `node` 处的最优子节点**对象本身、不是下标**；叶子处为 `None`；若 node 处轮到最小化者，返回的是让最大化者收益最小的那个子节点）、`"n_evaluated"`（int，`evaluate_fn` 的调用次数 = 展开的叶子数）、`"n_pruned"`（int，发生剪枝（提前 `break`）的次数）。
- **陷阱**：① `n_pruned` 是**剪枝事件次数**，不是"省下的求值次数"；被剪子树只有真的展开才知道多大，报告"剪枝节省了多少节点"必须用 `n_evaluated` 与不剪枝实现对比，而不是看 `n_pruned`。② α-β 只在**叶子评估精确**时才与完全极小极大给出相同的值；若 `depth` 提前截断、`evaluate_fn` 是启发式估值，两条路径的值就会不同——那是深度截断的误差，不是剪枝的错。③ 剪枝判定必须用 `alpha >= beta`：写成严格大于只损失剪枝率（结果仍对），但若在**更新窗口之前**就 break，则可能剪掉真正更优的分支、得到错值。④ `children_fn` 的返回顺序决定剪枝率：同一棵树换个顺序 `value` 不变但 `n_evaluated` 可能差一个量级；为保证可复现，`children_fn` 本身必须确定性（不要用随机顺序或依赖 dict 遍历顺序）。⑤ `best_child` 返回的是子节点对象的**引用**，调用方就地修改会影响自己的树结构；且递归内部（非根节点）返回的是"窗口内最优"，只有根节点全窗口调用才保证是真正的最优着法。⑥ 递归深度等于 `depth`，设成上百万会让 Python 抛 `RecursionError`（默认上限 1000），深树请改写成显式栈版本。
- **怎么检验**：`_self_test()` 的算例：抽象完全信息博弈"两人轮流从数集取数、最后比各自取到的数字之和"，节点 = `(剩余数字元组, 最大化者得分, 最小化者得分, 轮到谁)`，`children_fn` 枚举剩余数字，`evaluate_fn(n) = n[1] - n[2]`（终局为分差）；`ab_numbers = (5, 3, 8, 2, 9, 4, 1, 7)`，起点 `ab_start = (ab_numbers, 0.0, 0.0, 0)`，满深度 `len(ab_numbers) = 8` 时共 `8! = 40320` 个叶子。① **与不剪枝对拍**：`_self_test` 内部写了一段与 `minimax_alpha_beta` **无共享代码**的 `brute_minimax`（完全不剪枝），要求 `|ab_full["value"] - brute_full| <= 1e-9`，黄金键 `ab_value` / `ab_brute_value`（`round(..., 6)`）、`ab_value_match`（应为 `True`）。② **与解析值对拍**：`known_value = 5.0`，依据是"两人轮流取数时每次取当前最大是最优的，分差等于降序交错和 `9 - 8 + 7 - 5 + 4 - 3 + 2 - 1 = 5`"，要求偏差 `<= 1e-9`，黄金键 `ab_known_value`（= 5.0）。③ **剪枝确实生效**：要求 `ab_full["n_evaluated"] < brute_stats["n_evaluated"]`（黄金键 `ab_n_evaluated` / `ab_brute_n_evaluated` / `ab_fewer_nodes` 应为 `True`）且 `ab_full["n_pruned"] >= 1`（黄金键 `ab_n_pruned`）。④ **着法一致性**：黄金键 `ab_best_remaining`（`best_child[0]` 转 int 列表）与 `ab_best_taken`（`sum(ab_numbers) - sum(best_child[0])`，即先手取到的数之和）。⑤ **截断深度分支**：`depth = 3` 时同样与不剪枝对照一致（覆盖 `depth == 0` 的叶子分支），黄金键 `ab_depth3_value` / `ab_depth3_brute_value` / `ab_depth3_n_evaluated` / `ab_depth3_brute_n_evaluated`。⑥ 通用检验：任何树都用"同 `children_fn` 的朴素极小极大"对拍 `value`（精确评估下应逐位相同）；自建一个手工可算的小树（例如深度 2、分支 2）核对 `value`、`best_child` 与 `n_evaluated`；再用同一棵树打乱 `children_fn` 顺序，验证 `value` 不变而 `n_evaluated` 变化。

#### `stackelberg_lp`
- **数学形式**：领导者-跟随者（Stackelberg）线性博弈（线性双层规划）。给定领导者决策 `x ∈ R^m`（`0 <= x <= x_upper`），跟随者解参数化 LP：`max_y c_follower @ y  s.t.  A_follower y <= b_follower - A_leader x,  y >= 0`；领导者收益为 `c_leader @ y - leader_cost @ x`。返回的是盒式区域上按网格枚举该收益得到的**近似**最优解。
- **步骤**：① `as_matrix` / `as_vector` 读入 5 个必填输入。② 一致性校验：`b_follower` 长度须等于 `A_follower` 行数 `k`；`c_follower` 长度须等于 `A_follower` 列数 `n`；`c_leader` 长度须等于 `n`；`A_leader` 行数须等于 `k`；`A_leader` 至少 1 列（`m >= 1`）；`n_grid = int(n_grid)` 且须 `>= 2`（以上任一不满足抛 `ValueError`）。③ `x_upper` 为 None 时自动取 `scale = max(|b_follower|)`（若 `bf` 为空或 `scale <= 0` 则用 1.0）填充 m 维；否则 `as_vector` 并校长度 `== m`、非负、全部有限（无界区间上网格法无意义），违者 `ValueError`。④ `leader_cost` 为 None 时取 `np.zeros(m)`，否则校长度 `== m`。⑤ 每维构造 `axes[i] = np.linspace(0.0, upper[i], n_grid)`，用 `itertools.product(*axes)` 按字典序遍历共 `n_grid ** m` 个网格点。⑥ 对每个点 `x`：`rhs = bf - Al @ x`，调 `simplex_lp(cf, A_ub=Af, b_ub=rhs, maximize=True)`；`status != "optimal"` 则 `n_failed += 1` 丢弃该点；否则 `y = clip(res["x"], 0, None)`，`payoff = cl @ y - cost @ x`，仅当 `payoff > 当前最好` 才替换（**并列时保留字典序最小的点**，保证可复现）。⑦ 若所有网格点都被丢弃（`best is None`）抛 `ValueError`。⑧ 返回 `best` 字典。
- **复杂度**：时间 `O(n_grid^m * 一次 LP)`；空间 `O(k n + n_grid)`（不存储全部网格结果，只留当前最好）。
- **参数**：`A_leader`（领导者工具对跟随者资源的**耦合矩阵**，形状 `(k, m)`，k = 跟随者约束条数、m = 领导者决策维数，必填无默认值；第 i 条跟随者约束右端项为 `b_follower[i] - (A_leader x)[i]`，**取正号表示领导者的决策在消耗/限制跟随者资源**，负号表示放宽）；`c_leader`（领导者对**跟随者决策 y** 的估值向量，长度 `n`，必填无默认值）；`A_follower`（跟随者自己的约束矩阵，形状 `(k, n)`，必填无默认值）；`c_follower`（跟随者目标系数，长度 `n`，跟随者最大化 `c_follower @ y`，必填无默认值）；`b_follower`（跟随者约束右端项，长度 `k`，也是领导者工具为 0 时的资源量，必填无默认值）；`x_upper: Optional[ArrayLike] = None`（领导者每个决策变量的上界，长度 `m` 的非负向量；None 表示自动取 `max(|b_follower|)`，与资源同量级、避免网格尺度荒谬；**上界必须有界**）；`n_grid: int = 21`（每个领导者变量方向上的网格点数，含两端，必须 >= 2；总 LP 次数 = `n_grid ** m`）；`leader_cost: Optional[ArrayLike] = None`（领导者工具的单位成本向量，长度 `m`；None 表示工具零成本）。返回字典键：`"x_leader"`（形状 `(m,)` 的领导者决策，网格上最优的那个点）、`"y_follower"`（形状 `(n,)` 跟随者对 `x_leader` 的最优响应，LP 解，已截负）、`"leader_payoff"`（float，`c_leader @ y_follower - leader_cost @ x_leader`）、`"follower_payoff"`（float，`c_follower @ y_follower`）。
- **陷阱**：① **这是网格近似，不是精确解**：领导者收益作为 x 的函数是分片线性的（一般还不凹），最优可能落在两个网格点之间的折点上；请把 `n_grid` 调大做一次收敛性对照再报数字。② 领导者收益口径必须显式选定：本实现取"领导者对跟随者行动的线性估值减去工具成本"；若模型中领导者还从自己的决策直接获益，请把那一项并入 `leader_cost`（取负号）。③ **同时决策（Nash/Cournot）基准**：把 `x_upper` 设为全 0 就退化成"领导者不使用承诺能力"的同时决策结果；网格里含 `x = 0` 且做的是最大化，所以 Stackelberg 领导者收益不会低于该基准（自测里做了这条断言）。④ 跟随者 LP 的**退化与多解**会让"最优响应"不唯一：`y_follower` 只是单纯形给出的一个顶点，领导者收益在多个最优响应之间可能有差别（乐观/悲观口径），本实现取**乐观口径**（按跟随者会选对领导者最有利的一个来评估），论文里必须写明。⑤ `A_leader` 的符号极易搞反：右端项是 `b - A_leader x`，不是 `b + A_leader x`。⑥ 网格点数随 m 指数增长：`m = 3, n_grid = 21` 已经是 9261 次 LP，竞赛里请控制在 `n_grid ** m <= 1e4` 以内，或改用领导者收益的包络（分段线性）精确枚举。
- **怎么检验**：`_self_test()` 的一维算例有完整解析推导。设定：`st_A_leader = [[0.0], [-3.0], [1.0]]`、`st_c_leader = [1.0, 0.0]`、`st_A_follower = [[1,1],[1,0],[0,1]]`、`st_c_follower = [3.0, 4.0]`、`st_b_follower = [10.0, 2.0, 6.0]`，调用参数 `x_upper=[3.0], n_grid=7, leader_cost=[2.0]`。解析：跟随者 `max 3*y1 + 4*y2  s.t.  y1 + y2 <= 10, y1 <= 2 + 3x, y2 <= 6 - x`，故 `x <= 1` 时 `y = (2 + 3x, 6 - x)`、`x >= 1` 时 `y = (4 + x, 6 - x)`（容量上限接管）；领导者收益 `= y1 - 2x`，左支 `2 + x`（升）、右支 `4 - x`（降），**最优恰在折点 x = 1、收益 3**。断言：`|st["x_leader"][0] - 1.0| <= 1e-9`、`|st["leader_payoff"] - 3.0| <= 1e-9`。黄金键：`st_x_leader`（`round(..., 6)`）、`st_y_follower`、`st_leader_payoff`、`st_follower_payoff`。**同时决策基准**：同参数但 `x_upper=[0.0], n_grid=2` 再解一次得 `st_nash`，要求 `st["leader_payoff"] >= st_nash["leader_payoff"] - 1e-9`（黄金键 `st_nash_y_follower` / `st_nash_leader_payoff` / `st_leader_ge_nash` 应为 `True`；`_self_test` 文档写明"领导者收益 3 > 同时决策的 2"，可作为基准的解析参照）。通用检验：① 取一维情形手算跟随者响应函数的分片线性表达式，要求网格最优点落在折点或随 `n_grid` 增大收敛到折点；② 与"`x_upper` 全 0"的 Nash 基准比较，`leader_payoff` 不应更低；③ 增大 `n_grid`（如 7 → 21 → 41）看收益单调不降并趋于稳定；④ 校验返回的 `y_follower` 满足 `A_follower y <= b_follower - A_leader x`（在 LP 容差内）且 `y >= 0`，`leader_payoff == c_leader @ y_follower - leader_cost @ x_leader`、`follower_payoff == c_follower @ y_follower`。

#### `nash_bargaining_solution`
- **数学形式**：二人（二维）纳什谈判解，在可行集 `U` 上最大化**纳什积** `(u1 - d1)(u2 - d2)`，其中 `d` 是分歧点（谈判破裂时的效用）。多边形口径下可行集为 `U = {u : A_ub u <= b_ub}`（`A_ub` 形状 `(k, 2)`）；因为 `log` 纳什积在可行集上是凹的，最大值必落在东北边界，而边界由有限条线段拼成，每条线段上最大值只可能在端点或该段的内部驻点，故"顶点 + 各边驻点"构成**充分**候选集。边 `u(t) = A + t(B - A)` 上纳什积是 t 的二次函数，内部驻点 `t* = -(p1 q2 + p2 q1) / (2 q1 q2)`（`p = A - d`、`q = B - A`）。
- **步骤**：① `as_vector` 读入 `d`，长度必须为 2（否则 `ValueError`）。② 若 `payoff_set` 是 `Mapping`（多边形口径）：必须同时含 `"A_ub"` 与 `"b_ub"` 两键（否则 `ValueError`）；`A_ub` 列数必须为 2、行数必须等于 `b_ub` 长度（否则 `ValueError`）；调 `_polygon_vertices(A_ub, b_ub)` 求按极角排序的顶点（该辅助函数先用 4 次 LP 最大化/最小化 `u1`、`u2` 判定可行域非空且**有界**，无界或不可行直接抛 `ValueError`；再用两两约束直线求交 `det = a1[0]*a2[1] - a1[1]*a2[0]`、`|det| < 1e-12` 视为平行跳过、带容差 `tol = 1e-9 * max(1.0, max|b_ub|)` 做可行性过滤与交点去重、顶点少于 3 个抛 `ValueError`，最后按形心极角 `arctan2` 升序排序）。候选点 = 所有顶点副本；再对每条边算 `p0 = verts[i] - dvec`、`q = verts[(i+1) % len(verts)] - verts[i]`，若 `|q[0]*q[1]| < 1e-15`（边与某坐标轴平行、二次项为 0）则跳过，否则算 `t`，仅当 `0 < t < 1` 才把 `verts[i] + t*q` 加入候选。③ 否则按离散点集口径：`as_matrix` 读入且列数必须为 2（否则 `ValueError`），候选点就是每一行。④ 遍历候选点：`gain = u - dvec`，若 `gain[0] <= _TOL or gain[1] <= _TOL` 跳过（不满足个体理性，`_TOL = 1e-7`）；算 `prod = gain[0] * gain[1]`，仅当 `prod > best_prod` 才替换（**并列取输入顺序在前的点**，保证确定性）。⑤ 若没有任何候选通过（`best_u is None`）抛 `ValueError`（可行集中没有严格优于 d 的点、无合作剩余）。⑥ 返回字典。
- **复杂度**：时间——离散口径 `O(N)`；多边形口径 `O(k^2)`（顶点枚举）+ `O(k)` 条边的二次求根 + 4 次辅助 LP（有界性检查）；空间 `O(k)`（`_polygon_vertices` 内部为 `O(k^2)`）。
- **参数**：`d`（分歧点，长度 2 的向量，必填无默认值）；`payoff_set`（可行集，必填无默认值，两种口径二选一——(a) **离散点集**：形状 `(N, 2)` 的数组或 list of list，只在给定点上取最大纳什积；(b) **凸多边形**：字典 `{"A_ub": A, "b_ub": b}`，可行集为 `{u : A u <= b}`，`A` 形状 `(k, 2)`，**必须是有界多边形，且请把下界（`u >= 0` 之类）也显式写成 A 的行**——本函数不隐式假定 `u >= 0`，负效用坐标是允许的）。返回字典键：`"solution"`（形状 `(2,)` 的谈判解效用点 `(u1*, u2*)`）、`"utilities"`（形状 `(2,)` 的**净收益** `u* - d`，相对分歧点的增量，两者相乘即纳什积）、`"product"`（float，纳什积 `(u1* - d1)(u2* - d2)`，即被最大化的目标值）。
- **陷阱**：① **纳什解要求可行集里有严格优于 d 的点**：若 d 本身就在帕累托前沿上（没有合作剩余），纳什积最大值为 0、谈判解退化，本实现直接抛 `ValueError`，避免把 0 当"解"报出去。② 离散点集口径下"解"只是给定点里纳什积最大的那个，**不是**真正的纳什谈判解；点足够密时才可近似当作连续解，论文里要说明采样方式。③ 凸多边形口径**只支持二维**（纳什谈判解本身就是二人博弈概念），三维及以上要用凸优化求解器。④ **对称性公理**：可行集关于 `u1 <-> u2` 对称且 `d1 == d2` 时解必须落在对称轴上；本函数不对输入做对称化，数值上有 `1e-15` 级偏差，断言请用容差、别用 `==`。⑤ 多边形顶点用**两两直线求交**得到：退化多边形（三边共点、存在冗余约束）会产生重复交点，本实现按容差去重；若 `A` 的行里出现全零行，该约束不构成直线（会被跳过），请在传参前去掉冗余约束。⑥ 下界约束必须显式给（如 `-u1 <= 0`），`_polygon_vertices` 不假定 `u >= 0`，缺下界会直接判为无界并抛 `ValueError`（这是有意的：无界可行集上纳什谈判解没有定义）。⑦ 求交判平行要用**行列式**，用斜率比较会在竖直线（`a1 = 0`）上除零；容差按 `b_ub` 量级缩放，约束系数量级相差 `1e6` 倍以上时交点判定不可靠，请先无量纲化。
- **怎么检验**：`_self_test()` 给了三条闭式/公理检验。① **线性前沿上的"平分剩余"闭式解**：`nb_triangle = {"A_ub": [[1,1], [-1,0], [0,-1]], "b_ub": [2.0, 0.0, 0.0]}`、`nb_d = [0.5, 0.2]`；前沿是 `u1 + u2 = 2` 的直线段，纳什解把剩余 `nb_surplus = 2.0 - 0.5 - 0.2 = 1.3` 平分，故闭式解 `nb_expected = [0.5 + 1.3/2, 0.2 + 1.3/2] = [1.15, 0.85]`；断言 `|solution - nb_expected|.max() <= 1e-9`，且 `|product - (nb_surplus/2)^2| <= 1e-9`（即 `0.65^2 = 0.4225`）。黄金键 `nb_poly_solution` / `nb_poly_utilities` / `nb_poly_expected` / `nb_poly_product`（均 `round(..., 9)`）与 `nb_poly_dev`（`round(dev, 12)`，应约为 0）。② **对称性公理**：同一三角形可行集 + 对称分歧点 `[0.0, 0.0]`，断言 `|solution[0] - solution[1]| <= 1e-9`；黄金键 `nb_sym_solution`（`round(..., 9)`，应为对称点）与 `nb_sym_dev`（`round(..., 12)`，应约为 0）。③ **离散点集口径**：`payoff_set = [[1,3], [3,1], [2,2], [0,5]]`、`d = [0.0, 0.0]`，各点纳什积为 3、3、4、0，最大值在 `(2,2)`，断言 `|solution - [2.0, 2.0]|.max() <= 1e-12`；黄金键 `nb_disc_solution` / `nb_disc_product`（均 `round(..., 9)`，product 应为 4）。④ 通用检验：把解回代进 `A_ub u <= b_ub`（含容差）确认可行；确认 `utilities = solution - d` 且 `product == utilities[0] * utilities[1]`；用连续口径的算例与手算闭式解对拍；若 `d` 落在帕累托前沿上，确认函数抛 `ValueError` 而不是返回 0。


### 3.12 时间序列 —— `examples/algorithms/timeseries.py`

**这族解决什么问题**：这是 `forecasting` 的**进阶补充**（模块 docstring 原话），不重复移动平均、指数平滑、Holt-Winters、Yule-Walker AR、ACF/PACF、ADF 与精度指标，只补四类"更进一步"的方法：灰色系统 GM(1,1)（小样本、少数据、`n >= 4` 就能给预测，数学建模赛题里"数据只有几期"的标准兜底）、Box-Jenkins 类的 ARIMA/SARIMA（条件最小二乘估计 + AIC/BIC 选阶 + 多步预测区间）、条件异方差 GARCH(1,1)（给波动率建模与风险区间，均值模型不够用时接着做）、状态空间卡尔曼滤波（局部水平模型与一般线性高斯模型，用于含噪观测的在线状态估计与超参似然）。另外自带 Ljung-Box 残差白噪声检验，用来在论文里对上面任何一个模型做"残差是否还有信息"的收尾论证。全部只依赖 numpy + 标准库 + `._common`，不 import scipy / statsmodels。

**共同约定**：
- **序列输入格式**：一切序列输入都经 `_common.as_vector` 归一成一维 `np.ndarray`，时间先后排列（`x[0]` 最早），**本模块不做任何随机打乱**。矩阵输入（`kalman_filter_linear`）走 `as_matrix`，多通道观测按行/列约定见该函数。
- **返回 dict 的键命名习惯**：模型拟合类返回一个大 dict（`phi`/`theta`/`sigma2`/`loglik`/`aic`/`bic`/`residual`/`n_eff`/`n_iter` 加阶数信息 `d`/`D`/`period`/`p`/`q`），预测类返回 `forecast`/`se`/`lower`/`upper`，检验类返回检验统计量与 p 值，灰色模型返回 `a`/`b`/`fitted`/`forecast`/`residual`/`relative_error`。**例外**：`difference_series` 返回裸 `np.ndarray`（不是 dict）。
- **差分口径分两套，务必分清**：`difference_series` 是**对齐口径**（返回与输入等长、前 d 位 `NaN`）；而 `arima_fit`/`sarima_fit` 内部差分会**缩短序列**（同 `np.diff`）。混用会造成相位错位。
- **seed 口径**：本模块只有 `garch11_fit` 带随机性（多起点模式搜索），它显式接收 `seed`，`None` 时回落到 `_common.rng` 的 `DEFAULT_SEED = 20240101`，绝不使用 `np.random.*` 全局状态。
- **刻意不实现的方法（诚实声明）**：`arima_fit`/`sarima_fit` 是**条件最小二乘**（把滞后残差当已知的迭代线性回归），不是精确 MLE/CSS-ML；`garch11_fit` 是**模式搜索（compass search）**，不是 BFGS/数值梯度拟牛顿，也不做 t 分布 QML；`arima_forecast` **不支持 `D > 0`**（季节差分反差分未实现）。二者（三者）在样本充足时与标准软件结果接近，但**不保证**逐位一致，论文里必须写明口径。
- **数值容差**：条件最小二乘的迭代上限 `_CLS_MAX_ITER = 200`、收敛容差 `_CLS_TOL = 1e-10`（系数向量最大逐分量变化）；GARCH 参数约束 `alpha + beta < _GARCH_PERSIST_MAX = 0.999`。

#### `gm11(x, n_forecast=1)`

- **数学形式**：一次累加 X1_k = Σ_{i≤k} x_i；紧邻均值 z_k = 0.5(X1_k + X1_{k−1})；白化方程离散形式 x_k + a·z_k = b，即 [a, b]ᵀ = argmin ‖B[a,b]ᵀ − Y‖²，B = [−z, 1]，Y = x_{1..n−1}；时间响应函数 X̂1_k = (x_0 − b/a)·e^{−ak} + b/a；累减还原 x̂_k = X̂1_k − X̂1_{k−1}，x̂_0 = x_0；未来第 h 期 = X̂1_{n−1+h} − X̂1_{n−2+h}
- **步骤**：① 校验 `x` 为一维向量、`n_forecast` 为 ≥1 的整数，且 n ≥ 4 否则抛 `ValueError`；② `np.cumsum` 得 X1，取梯形权重 0.5 得紧邻均值 z；③ 组装 B = [−z, 1] 用 `np.linalg.lstsq`（SVD）解出 a、b；④ 若 a 非有限或 |a| < 1e-12 抛 `ValueError`；⑤ 令 c = x_0 − b/a，按响应函数算 X̂1（k = 0..n−1），`fitted[0] = x[0]`、`fitted[1:] = np.diff(X1h)`；⑥ 在 k = n..n+h−1 上外推响应函数，拼成 `full` 后 `np.diff(full)[n-1:]` 取未来 h 期；⑦ 残差 e = x − fitted，相对误差 |e/x|×100（`x[k] == 0` 处置 `NaN`）；⑧ 返回 `a`/`b`/`fitted`/`forecast`/`residual`/`relative_error`
- **复杂度**：时间 O(n)（含一次 2 列最小二乘，`lstsq` 走 SVD）/ 空间 O(n)
- **参数**：`x` 一维原始序列（按时间先后，**长度 ≥ 4**，否则紧邻均值与残差都不稳，抛 `ValueError`）；`n_forecast` 样本外预测步数，必须是真整数（`bool` 被 `_as_int` 拒绝，浮点 `2.0` 也被拒绝、不做四舍五入）且 ≥ 1。本函数**没有 backgound-value / 背景值优化参数**（紧邻均值权重固定 0.5）、**没有 `alpha`/`seed`/`d` 参数**；要后验差检验请用 `gm11_posterior_check`，要 ARIMA 类模型请用 `arima_fit`
- **陷阱**：**"对纯指数序列精确"只是模型形式层面的说法，不是估计层面的**——紧邻均值（梯形）离散化与白化方程的连续解不同源，最小二乘估的 a 有 O(1/n) 系统偏差，纯指数序列 `x_k = 2e^{0.3k}` 的多步预测相对误差约 2%~5% 且随步长累积（自测把该真实误差当键返回，不假装是 1e-6）；只有**由离散白化方程本身生成的数据**（x_k 为比例 (1−a/2)/(1+a/2) 的几何序列）残差才恒为 0，此时最小二乘能精确还原 (a, b)，自测就构造这种数据验证参数恢复；a → 0 会让 b/a 溢出，本实现直接抛 `ValueError`，经验上 |a| < 0.3 才适合中长期预测；原始序列含 0 或负值时相对误差无意义（置 `NaN`），且灰色模型本身要求非负；n < 4 时最小二乘解方差极大，"预测"几乎等于外推噪声
- **怎么检验**：① 构造由离散白化方程生成的几何序列，断言 `a`/`b` 恢复到真值（`_self_test` 走这条路，是可独立重跑的闭式结论）；② 用纯指数序列 `x_k = 2e^{0.3k}` 独立重算时间响应函数，核对预测的相对误差量级落在 docstring 声明的 2%~5% 而不是接近 0；③ 极限行为：n_forecast 增大时预测应单调沿响应函数走，`fitted[0]` 必须恒等于 `x[0]`；④ 独立实现：用 `np.linalg.solve` 解正规方程 BᵀB[a,b]ᵀ = BᵀY，与 `lstsq` 结果对拍；⑤ 非法输入：`n < 4`、`n_forecast=0`、`d=True` 风格的非整数都应抛 `ValueError`

#### `gm11_posterior_check(x, fitted)`

- **数学形式**：残差 e = x − fitted；S1 = std(x, ddof=1)、S2 = std(e, ddof=1)；方差比 C = S2 / S1；小误差概率 P = #{|e_k − ē| < 0.6745·S1} / n（阈值 0.6745·S1 对应正态下约 0.75 分位的绝对偏差界）
- **步骤**：① `as_vector` 归一 `x`、`fitted` 并用 `check_same_length` 校验等长；② 校验 n ≥ 2 否则抛 `ValueError`；③ 算残差与 S1（ddof=1），若 S1 ≤ 0（常数序列）抛 `ValueError`；④ 算 S2（ddof=1）与 C = S2/S1；⑤ 以样本频率计数 P（严格小于 0.6745·S1）；⑥ 两条件同时满足才升档：C < 0.35 且 P > 0.95 → "好"；C < 0.5 且 P > 0.8 → "合格"；C < 0.65 且 P > 0.7 → "勉强"；否则 "不合格"；⑦ 返回 `c_ratio`/`p_small_error`/`grade`
- **复杂度**：时间 O(n) / 空间 O(n)
- **参数**：`x` 一维原始序列；`fitted` 与 `x` **等长**的拟合序列（通常取 `gm11(...)["fitted"]`）——长度不等抛 `ValueError`，n < 2 抛 `ValueError`，常数序列抛 `ValueError`。本函数**没有 `ddof` 开关**（固定 ddof=1）、**没有自定义等级阈值参数**（0.35/0.5/0.65 与 0.95/0.8/0.7 写死）、**没有 `alpha` 参数**（P 是样本频率不是渐近概率）；要算预测请用 `gm11`，要残差白噪声检验请用 `ljung_box`
- **陷阱**：**口径必须交代**——S1/S2 用 ddof=1 还是 ddof=0 会改变 C 的第四位小数，足以在 0.35/0.5/0.65 边界上翻转等级，本实现固定 ddof=1；`fitted` 必须与 `x` 逐点对应（同一时间轴、同样的差分口径），若 `fitted` 来自差分后序列，长度/相位错位会算出"看起来还行"的假象；P 是样本频率而非渐近概率，n 小时只能取到 1/n 的倍数，等级判定本身很粗糙，不要据此声称"模型精度 95%"；原始序列为常数（S1 = 0）时 C 无定义，本实现抛 `ValueError`
- **怎么检验**：① 用 `fitted = x` 构造完美拟合，断言 `c_ratio` 接近 0（严格为 0？注意 S2 = 0 合法、S1 > 0，C = 0 落入 "好" 档需同时满足 P > 0.95——此时 dev 全为 0，故 P = 1）与 `grade == "好"`；② 独立实现：自己用 `statistics`/numpy 手算 ddof=1 的两个标准差与计数式 P，与返回的 `c_ratio`/`p_small_error` 逐位比对；③ 边界翻转测试：把 C 调到 0.35/0.5/0.65 附近，检查等级随 P 同时变化（验证"两条件同时满足才升档"）；④ 非法输入：长度不等、n = 1、常数 `x` 都应抛 `ValueError`；⑤ `_self_test()` 中有对应断言（见该函数返回的 `ts_` 前缀键）

#### `difference_series(x, d=1)`

- **数学形式**：Δ^d x_k = Σ_{j=0}^{d} (−1)^j C(d, j)·x_{k−j}；输出与输入等长，`out[k] = x^(d)[k]` 在有效区间 [d, n−1]，前 d 位为 `NaN`
- **步骤**：① `as_vector` 归一 `x`，`_as_int` 校验 d ≥ 0（bool 与浮点被拒）；② 若 d ≥ n 抛 `ValueError`；③ 建长度 n 的全 `NaN` 数组；④ d == 0 时直接拷回原序列副本；⑤ 否则在**有效段**上一次性调用 `np.diff(xv, n=d)`，左对齐写入 `out[d:]`；⑥ 返回该数组
- **复杂度**：时间 O(d·n) / 空间 O(n)
- **参数**：`x` 一维时间序列；`d` 差分阶数，真整数且 ≥ 0，d = 0 返回原序列副本；d ≥ n 抛 `ValueError`。本函数**没有 `period` 参数**（不做季节差分，季节差分请自行用周期跨度重采样或用 `sarima_fit` 内部口径）、**没有填充方式开关**（固定前置 `NaN`，不做边缘填充/回填）
- **陷阱**：本函数是**对齐口径**（长度不变、前面补 `NaN`），与模型内部的**缩短口径**（`np.diff` 返回变短数组，`arima_fit`/`sarima_fit` 用这套）不同，混用会造成相位错位；**先补 NaN 再逐阶差分是错的**——第一阶差分产生的 `NaN` 会把后面所有值传染成 `NaN`，本实现刻意在有效段上一次算完 d 阶；差分会放大噪声并损失 d 个自由度；返回值中的 `NaN` 会污染 `mean`/`std`，请先用 `np.isfinite` 过滤
- **怎么检验**：① 对 d = 1 与 `np.diff(x)` 对拍（断言 `out[1:]` 逐位相等、`out[0]` 是 `NaN`）；② 对二次序列 `x_k = k²` 断言二阶差分恒为常数 2（闭式结论，且能验证"一次算完 d 阶"没有 NaN 传染）；③ 对 d = 0 断言返回的是副本（改动返回值不改变原输入）；④ 独立实现：用 `out[k] = Σ_j (−1)^j C(d,j) x_{k−j}` 显式卷积核对；⑤ 非法输入：`d = n`、`d = True`、`d = 1.0` 都应抛 `ValueError`

#### `arima_fit(x, p=1, d=0, q=1)`

- **数学形式**：差分后 w_t = Δ^d x_t；中心化 y_t = w_t − μ；条件最小二乘拟合 ARMA(p, q)：y_t = Σ_{i=1}^{p} φ_i y_{t−i} + ε_t + Σ_{j=1}^{q} θ_j ε_{t−j}；`loglik = −0.5·n_eff·(ln 2π + ln σ² + 1)`，k = p + q + 1（含 σ²），AIC = −2·loglik + 2k、BIC = −2·loglik + k·ln(n_eff)
- **步骤**：① `as_vector` 归一 `x`，`_as_int` 校验 p、d、q ≥ 0；② 若 d ≥ n 抛 `ValueError`；③ `_differenced_tail` 记录普通差分各阶中间序列的**末值** `x_tail`（供反差分）；④ w = x（d = 0）或 `np.diff(x, n=d)`，若 w.size < 4 抛 `ValueError`；⑤ 组装滞后集合 `ar_lags = 1..p`、`ma_lags = 1..q`，交给 `_build_arma_model`；⑥ `_build_arma_model` 减样本均值 μ，调 `_arma_cls` 迭代条件最小二乘（上限 200 次、容差 1e-10）得 φ、θ、条件残差、σ²（分母 n_eff）、迭代次数；⑦ 由 σ² 与 n_eff、参数个数算 loglik/AIC/BIC；⑧ 截取最后 max(ar_lags) 个 w 值存为 `w_tail`；⑨ 返回含 `phi`/`theta`/`sigma2`/`loglik`/`aic`/`bic`/`d`/`D`/`period`/`p`/`q`/`mean`/`residual`/`n_eff`/`n_iter`/`ar_lags`/`ma_lags`/`w_tail`/`x_tail` 的字典
- **复杂度**：时间 O(n_iter · m · (p + q)²) / 空间 O(m · (p + q))（m = n − d）
- **参数**：`x` 一维序列；`p` 非季节 AR 阶数（≥ 0，默认 1）；`d` 普通差分阶数（≥ 0，须 < n，默认 0）；`q` 非季节 MA 阶数（≥ 0，默认 1）。三者都必须是真整数（bool/浮点被 `_as_int` 拒绝）。本函数**没有 `P`/`D`/`Q`/`period` 季节参数**（要季节项用 `sarima_fit`）、**没有 `method` 开关**（估计量固定为 CLS，不是 MLE/CSS-ML）、**没有 `trend` 参数**（均值只在差分后序列上估计）、**没有 `seed`**（CLS 是确定性的）
- **陷阱**：**估计量是 CLS 不是 MLE**——AR(1) 情形下它等于"去掉第一个观测的 OLS"，与 Yule-Walker（`forecasting.ar_model`）的差别来自分母少一项 x_0²，量级 O(1/n) 且依赖最后一个观测的取值，因此**不能承诺任意数据上都一致到 1e-6**；均值在差分后序列上估计，等同于假设原序列含确定性**线性漂移**（d = 1 时），解释常数项时注意这一点；`p = q = 0` 时退化为"差分序列的均值 + 白噪声"，此时 n_iter = 0、phi/theta 为空数组，`forecasting.ar_model` 不支持这种情形，不要互相替代；完全线性（残差方差为 0）的数据会抛 `ValueError`（对数似然无定义），`arima_order_select` 会捕获并跳过这类组合；阶数越高 n_eff 越小，AIC 跨不同 `d` 不可直接比较
- **怎么检验**：① 自测用固定算例给出与 Yule-Walker 的**实测差值**（不是承诺 1e-6）；② **独立闭式解**：AR(1)（p=1, q=0）条件下最小二乘等价于去掉第一个观测的 OLS，用 `np.polyfit`/正规方程手算 φ̂ 对拍；③ p = q = 0 断言 `n_iter == 0`、`phi.size == 0`、`theta.size == 0`；④ 用模拟 AR(1) 大样本检查 φ̂ → 真值（一致性方向性验证）；⑤ 完全线性数据（如 x = k）应抛 `ValueError`；⑥ 与 `statsmodels.tsa.arima.model.ARIMA` 的 CSS 结果对拍，差异应随 n 增大而缩小但不保证逐位一致；⑦ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `arima_forecast(model, n_ahead=1)`

- **数学形式**：递推点预测 ŵ_h = μ + Σ_i φ_i(w_{m+h−i} − μ) + Σ_j θ_j e_{m+h−j}（未来残差取 0、历史残差不足处取 0）；ψ 权重由 MA(∞) 表示给出（ψ_0 = 1，ψ_k = Σ_{AR 滞后 ≤ k} φ·ψ_{k−lag} + θ_k）；下三角矩阵 Ψ[k,j] = ψ_{k−j}，d 阶累加算子 C = tril(ones)，反差分算子 M = C^d；误差传播系数 c = M·Ψ，se_h = √(σ²·Σ_j c_{hj}²)；区间为点预测 ± 1.96·se
- **步骤**：① 校验 `model` 是 dict 且含 `phi`/`theta`/`sigma2`/`residual`/`mean`/`w_tail`/`x_tail`/`d`/`ar_lags`/`ma_lags` 十个键，缺失抛 `ValueError`；② `_as_int` 校验 `n_ahead` ≥ 1；③ 若 `model["D"] > 0` 抛 `ValueError`；④ 校验 φ/θ 与滞后集合长度一致、σ² 严格为正、`x_tail` 长度等于 d、`w_tail` 长度 ≥ max(ar_lags)（不足抛 `ValueError`）；⑤ 把残差右侧补 h 个 0 得 `e_ext`，用 `w_tail` 作历史，逐 h 递推点预测并把预测值追加进历史；⑥ `_psi_weights` 算 ψ，组装 Ψ 与 M = C^d，得 `se`；⑦ 从 d 阶中间序列末值 `x_tail[j]` 起逐阶 `cumsum` 累加反差分回原量纲；⑧ 返回 `forecast`/`se`/`lower`/`upper`
- **复杂度**：时间 O(n_ahead³ + m(p+q)) / 空间 O(n_ahead²)
- **参数**：`model` 必须是 `arima_fit`（或 `sarima_fit` 且 `D == 0`）返回的字典——不是 dict 抛 `ValueError`，缺键抛 `ValueError`，键间不自洽（φ/θ 与滞后集合长度不符、`x_tail` 长度与 d 不符、`w_tail` 太短）也抛 `ValueError`；`n_ahead` 预测步数，真整数且 ≥ 1（默认 1）。本函数**没有 `D > 0` 的支持**（季节差分反差分未实现，直接抛 `ValueError`）、**没有 `level`/`alpha` 参数**（区间固定 95%、用 1.96 常数写死）、**没有 `seed`**（纯确定性递推）。要季节差分模型的多步预测请自行对 `sarima_fit` 的差分序列累加
- **陷阱**：**只支持 `D == 0`**——季节差分模型的反差分未实现（点预测的相位对齐需要保存每一阶季节差分的尾部窗口），`D > 0` 时本函数直接抛 `ValueError`，请自行对 `sarima_fit` 的差分序列累加；置信区间是**条件正态近似**——忽略参数估计不确定性（μ、φ、θ、σ² 都当已知）与分布非正态性，短序列下实际覆盖率偏低；只有当 AR 特征根在单位圆内、且 d 与数据生成过程一致时，se 才随步长发散得合理，用错 d 会让区间宽窄完全失真；`w_tail` 必须来自同一模型字典（长度 ≥ max(ar_lags)），手工裁剪会静默取错历史
- **怎么检验**：① **闭式退化**：`d = 0` 时断言 se_h = √(σ²·Σ_{l<h} ψ_l²)，`d = 1` 且 p = q = 0（随机游走）时断言 se_h = √(σ²·h)——这两条是教科书闭式解，可直接独立重算；② 用 `arima_fit(x, p=0, d=0, q=0)` 的模型做预测，应恒等于 `mean`（白噪声的期望）且 se 恒为 √σ²；③ 与 statsmodels `get_forecast` 的点预测对拍（同一 CLS 参数下应接近）；④ 人工篡改 `model["D"] = 1` 断言抛 `ValueError`；⑤ 从模型字典里删一个键断言抛 `ValueError`；⑥ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `arima_order_select(x, p_max=3, d_max=2, q_max=3, criterion="aic")`

- **数学形式**：在网格 p ∈ [0, p_max]、d ∈ [0, d_max]、q ∈ [0, q_max] 上最小化准则 AIC = −2·loglik + 2k 或 BIC = −2·loglik + k·ln(n_eff)，k = p + q + 1；关键在于**同一 d 内先对齐共同样本**：取该 d 下最大的 t0 = max(lags)，用尾段残差重算 s² = Σe²/n_common，再算 AIC/BIC
- **步骤**：① `as_vector` 归一 `x`，`_as_int` 校验三个上界 ≥ 0；② 校验 `criterion` 是字符串，`strip().lower()` 后必须落在 `_ORDER_CRITERIA = ("aic", "bic")` 否则抛 `ValueError`；③ 三重循环 d → p → q，逐个调 `arima_fit`，**捕获 `ValueError` 与 `np.linalg.LinAlgError` 静默跳过**失败组合（样本不足、残差方差为 0）；④ 每个成功组合记录 `t0 = resid.size − n_eff`（前 t0 位残差为 0、未参与条件似然）与 `n_params = p + q + 1`；⑤ 对每个 d，取 `t0_common = max(t0)`，把各组合残差截到 `[t0_common:]` 重算共同样本的 s² 与 AIC/BIC（n_common ≤ 0 或 s² 非正/非有限则丢弃）；⑥ 若 `table` 为空抛 `ValueError`；⑦ `min` 取准则最小者，并列时按 `(d, p, q)` 字典序最小；⑧ 返回 `best`（`{"p","d","q","aic","bic"}`）与 `table`（同结构列表，按 `(d, p, q)` 升序）
- **复杂度**：时间 O((p_max+1)(d_max+1)(q_max+1) × 单次拟合) / 空间 O(组合数)
- **参数**：`x` 一维序列；`p_max`/`d_max`/`q_max` 各阶数上界（均 ≥ 0，**含上界**，默认 3/2/3），真整数否则 `ValueError`；`criterion` 取 `"aic"` 或 `"bic"`，**大小写不敏感**（会 strip + lower），非字符串或非法名抛 `ValueError`。本函数**没有 `P_max`/`D_max`/`Q_max` 季节选阶参数**（季节模型要自己用 `sarima_fit` 手填网格）、**没有 `seed`**、**没有系数显著性/残差白噪声的自动筛选**（可配合 `ljung_box`）
- **陷阱**：**跨 d 比较 AIC 理论上不成立**——不同 d 下似然对应不同的数据（差分后序列的长度与含义都变了），这里按惯例仍做全局最小，但更稳妥的做法是先用 `forecasting.adf_test` 定 d，再在固定 d 上按 AIC 选 p、q；网格搜索**不保证找到全局最优模型**，更大的 p_max 可能选出过拟合的低 AIC 模型，也不检查系数显著性与残差白噪声；表内 `aic`/`bic` 是本函数**按共同样本重算**的值，与单独调用 `arima_fit` 得到的同名键**不逐位相等**（后者用该组合自己的 n_eff）——这是刻意为之，各组合 t0 = max(lags) 不同，不先对齐样本，AIC 之差里会混进"观测数不同"的伪项（每少一个观测约值 ln 2π + ln σ² + 1 ≈ 2.9 个 AIC 点），足以把真实阶数选错，本实现对标 R `arima` / statsmodels 的"同一样本比较"惯例；搜索成本是乘积级——`p_max = q_max = 5, d_max = 2` 要拟合 216 次，谨慎调大
- **怎么检验**：① 用已知阶数的模拟序列（如 AR(1)）跑选阶，断言 `best["p"]` 落在合理范围；② **交叉核对对齐效应**：手工复算某组合的 `t0`，自己按共同样本截断算 s² 与 AIC，断言与 `table` 里的值一致，而与该组合单独 `arima_fit` 的 `aic` 不相等；③ 退化用例：`forecasting.ar_model` 风格的可拟合短序列上 `p_max=q_max=0`，断言 `best` 唯一且 `table` 长度为 d_max+1；④ 非法输入：常数/超短序列抛 `ValueError`，`criterion="AICC"` 抛 `ValueError`；⑤ 断言 `table` 确实按 `(d, p, q)` 升序；⑥ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `sarima_fit(x, period, p=1, d=0, q=1, P=1, D=1, Q=0)`

- **数学形式**：先 D 次季节差分 w ← w[s:] − w[:−s]，再 d 次普通差分；把（S）ARMA 展开成**对滞后项的可加线性回归**：AR 滞后集合 {1..p} ∪ {s, 2s, …, Ps}，MA 滞后集合 {1..q} ∪ {s, 2s, …, Qs}；系数用与 `arima_fit` 相同的迭代条件最小二乘估计；`loglik = −0.5·n_eff·(ln 2π + ln σ² + 1)`
- **步骤**：① `as_vector` 归一 `x`，`_as_int` 校验 `period` ≥ 2 与六个阶数 ≥ 0；② 循环 DD 次做季节差分，每次若 `w.size <= s` 抛 `ValueError`；③ 若 d ≥ 季节差分后的长度抛 `ValueError`；④ `_differenced_tail` 记录普通差分各阶末值，再做 d 次普通差分；⑤ 差分后长度 < 4 抛 `ValueError`；⑥ 合并滞后集合 `ar_lags = 1..p + s·(1..P)`、`ma_lags = 1..q + s·(1..Q)`；⑦ 调 `_build_arma_model` 组装模型字典（键与 `arima_fit` 一致，`D` 与 `period` 如实记录）；⑧ 把 `p`/`q` 覆盖为非季节阶数、补 `P`/`Q`，按 `pp`/`qq` 切片出 `seasonal_phi`/`seasonal_theta`（`phi`/`theta` 里只保留非季节系数）
- **复杂度**：时间 O(n_iter · m · (p+q+P+Q)²) / 空间 O(m · (p+q+P+Q))
- **参数**：`x` 一维序列；`period` 季节周期 s，真整数且 ≥ 2（**默认值都没有，必须显式给**）；`p`/`d`/`q` 非季节 AR/差分/MA 阶数（≥ 0，默认 1/0/1）；`P`/`D`/`Q` 季节 AR/差分/MA 阶数（≥ 0，默认 1/1/0）。本函数**没有 `method` 开关**（固定 CLS）、**没有真正的乘积季节多项式**（见陷阱）、**没有 `seed`**
- **陷阱**：**这是简化版**——真正的 SARIMA 季节多项式与 AR/MA 多项式是**乘积**关系（(1 − φB)(1 − ΦB^s)），本实现把它们当**可加**滞后项合并回归，当 (P, Q) 与 (p, q) 同时非零时两者不等价，交叉项（如滞后 s+1）会被漏掉，自测因此只用乘积与可加一致的情形（p = q = 0 或 P = Q = 0）做校验；一次季节差分就损失 s 个观测，D = 1 且 s 较大时可用样本骤减，样本不足会抛 `ValueError`；反差分上 `arima_forecast` 不支持 D > 0；`P = D = Q = 0` 时本函数与 `arima_fit(x, p, d, q)` 走**同一条代码路径**，结果逐位相同（自测据此做交叉验证）
- **怎么检验**：① **交叉验证**：断言 `P = D = Q = 0` 时与 `arima_fit(x, p, d, q)` 返回的 `phi`/`theta`/`sigma2`/`aic`/`bic` **逐位相同**；② 用周期为 s 的确定性季节序列（如 sin 周期信号）检查季节系数被正确吸收；③ 独立实现：把可加滞后回归写成设计矩阵，用 `np.linalg.lstsq` 直接解一次（不迭代）与 CLS 结果比较（迭代收敛时应接近）；④ 断言 `phi.size == p`、`theta.size == q`、`seasonal_phi.size == P`、`seasonal_theta.size == Q`，且 `ar_lags`/`ma_lags` 长度等于合并后的总数；⑥ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `garch11_fit(r, max_iter=500, seed=None)`

- **数学形式**：σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}，ε_t = r_t − μ（μ 为样本均值）；正态负对数似然 −ℓ = 0.5·Σ_t [ln 2π + ln σ²_t + ε²_t/σ²_t]，最大化 ℓ 即最小化 −ℓ；持续性 persistence = α + β；可行域 α + β < 0.999、ω > 0、α, β ≥ 0
- **步骤**：① `as_vector` 归一 `r`，`_as_int` 校验 `max_iter` ≥ 1，若 T < 20 抛 `ValueError`；② 内部去均值 r ← r − mean(r)，取 σ²_0 = var(r, ddof=1)，方差为 0 抛 `ValueError`；③ `rng(seed)` 造起点：两个**确定性起点** `(0.05·var, 0.05, 0.90)`、`(0.10·var, 0.10, 0.80)`，再加 2 个**随机起点**（`gen.uniform` 抽 ω∈[0.01, 0.20]·var、α∈[0.01, 0.30]、β∈[0.50, 0.95]，每个最多试 20 次直到通过 `_garch_feasible` 过滤）；④ 每个起点做**坐标模式搜索**：步长初值 `step0 = 0.2`，依次沿 ω/α/β 的正负方向试探（ω 的步长按样本方差缩放），若目标下降超过 `1e-12` 就接受并 break；一轮三次坐标都无改进则 `step *= 0.5`，直到 `step < tol = 1e-6` 或达到 `max_iter`；⑤ 取所有起点中负对数似然最小者；若一个可行参数都没拿到抛 `ValueError`；⑥ 返回 `omega`/`alpha`/`beta`/`persistence`/`sigma2`（条件方差路径，`sigma2[0]` 是初值）/`sigma2_next`（样本外第一步）/`loglik`（取负）/`n_iter`/`mean`
- **复杂度**：时间 O(n_starts · n_iter · T) / 空间 O(T)
- **参数**：`r` 一维收益率序列——**本函数内部去均值，不需要调用方预处理**，T < 20 抛 `ValueError`，常数序列抛 `ValueError`；`max_iter` 模式搜索最大外层迭代次数（≥ 1，真整数，默认 500）；`seed` 随机多起点的种子，`None` 表示使用 `_common.DEFAULT_SEED = 20240101`（**可复现**）。本函数**没有 `dist` 参数**（固定正态 QML，不做 t 分布/偏斜 t）、**没有 `mean_model` 参数**（均值只减常数，不是 GARCH-M/AR 均值方程）、**没有 `p`/`q` 参数**（固定 (1,1)）、**没有标准误**（不返回参数协方差，稳健标准误本模块不提供）
- **陷阱**：优化器是**模式搜索**（无导数、收敛慢），不是 BFGS；似然面在 α/β 高度相关时它是"沿坐标轴走"，可能提前停住，多起点是为了缓解这一点，但**不保证**全局最优，与 R 的 `rugarch`/`fGarch` 数值可能有**百分之几**的差异；内部**去均值**——返回的 `mean` 是样本均值，若数据有明显漂移，等价于先减常数，与"含常数项的 GARCH-M/AR 均值方程"不同；约束 α + β ≤ 0.999 会把接近 IGARCH 的真实过程压向边界内（见 `_garch_feasible`），ω 被约束为正，因此无法表示"零方差"退化情形；`sigma2[0]` 是**初值不是估计值**，序列的前几项受初值影响，T 很小时别把 `sigma2` 的头几个值直接当"波动率估计"；正态 QML 对厚尾数据给出相合但非有效的估计，标准误需稳健修正，本模块不提供
- **怎么检验**：① **可复现性**：同一 `seed` 跑两次断言 `omega`/`alpha`/`beta`/`loglik` **逐位相同**，不同 seed 的结果应在同一量级（验证多起点有效但没有全局保证）；② **极限行为**：把 ω 固定为真值、构造退化序列检查 ω 被推向正下界而不会为负；③ 独立实现：自己写 GARCH(1,1) 的 σ² 递推与负对数似然，在 `garch11_fit` 返回的参数点上重算 `loglik`，断言与返回的 `loglik` 一致（**注意 `sigma2[0]` 是初值这一约定**）；④ 断言 `persistence == alpha + beta`；⑤ 用模拟 GARCH 数据检查 α + β 的估计方向合理；⑥ 非法输入：T = 19、常数序列、`max_iter=0` 抛 `ValueError`；⑦ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `garch11_forecast(model, n_ahead=1)`

- **数学形式**：E[σ²_{T+h} | F_T] = LR + (α+β)^h·(σ²_T − LR)，其中 LR = ω/(1 − α − β) 是长期方差，σ²_T 取模型给出的下一期一步向前方差 `model["sigma2_next"]`；`volatility` = √variance
- **步骤**：① 校验 `model` 是 dict 且含 `omega`/`alpha`/`beta`/`sigma2_next` 四个键，缺任一抛 `ValueError`；② `_as_int` 校验 `n_ahead` ≥ 1；③ 用 `_as_positive` 校验 ω 严格为正、α/β 非负、`sigma2_next` 严格为正（NaN/inf 被拦掉）；④ 算 persistence = α + β，若 ≥ 1 抛 `ValueError`；⑤ LR = ω/(1 − persistence)；⑥ 对 h = 1..n_ahead 向量化算 var_h = LR + persistence^h·(σ²_T − LR)；⑦ 返回 `variance`/`volatility`/`long_run_variance`
- **复杂度**：时间 O(n_ahead) / 空间 O(n_ahead)
- **参数**：`model` 必须是 `garch11_fit` 返回的字典——非 dict 抛 `ValueError`，缺键抛 `ValueError`，`omega`/`sigma2_next` 非正或 α + β ≥ 1 也抛 `ValueError`；`n_ahead` 预测步数，真整数 ≥ 1（默认 1）。本函数**没有 `level`/`alpha` 参数**（只返回方差与标准差，不返回分位数区间）、**没有 `seed`**、**没有多变量/GJR/EGARCH 支持**；要均值预测请另建均值模型
- **陷阱**：该公式是**条件方差的期望，不是"波动率的期望"**——E[σ_{T+h}] ≤ √(E[σ²])（Jensen 不等式），所以 `volatility` 是**下偏**的；只对线性 GARCH(1,1) 严格成立，换成 GJR/EGARCH 或多变量模型，均值回复形式不同；`persistence` 越接近 1 回复越慢，长期方差对参数误差极敏感（分母 1 − persistence 很小），因此 `long_run_variance` 的不确定性远大于点估计值本身；模型字典必须含 `sigma2_next`，若手工构造而漏了它，本函数抛 `ValueError`
- **怎么检验**：① **闭式退化**：h 增大时 `variance` 应收敛到 `long_run_variance`（断言 |var_h − LR| 单调下降、persistence < 1 时按几何速率衰减）；② persistence 很小的极端参数下断言 var_h 迅速贴到 LR；③ 独立实现：手工用 LR + p^h(σ²_T − LR) 重算每个 h，逐位对拍；④ 断言 `volatility` 与 `np.sqrt(variance)` 逐位相等（明确它是同一量的开方，不是独立估计）；⑤ 非法输入：篡改 `model["alpha"] + model["beta"] ≥ 1` 断言抛 `ValueError`，删 `sigma2_next` 抛 `ValueError`；⑥ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `kalman_filter_local_level(y, q=1.0, r=1.0)`

- **数学形式**：状态方程 x_t = x_{t−1} + w_t，w_t ~ N(0, q)；观测方程 y_t = x_t + v_t，v_t ~ N(0, r)。递推：P_pred = P_{t−1} + q；K = P_pred/(P_pred + r)；x_t = x_pred + K(y_t − x_pred)；P_t = (1 − K)·P_pred；似然累加 −0.5[ln 2π + ln(P_pred + r) + innovation²/(P_pred + r)]
- **步骤**：① `as_vector` 归一 `y`，用 `_as_positive(..., strict=True)` 校验 q > 0、r > 0；② 初值 `x_pred = y[0]`、`p_pred = q`（代码如此写，等价于"x_0 = y[0]、P_0 = 0 后先加一次 q"）；③ 逐 t：记录 `predicted[t] = x_pred`、`predicted_variance[t] = P_pred`；算 S = P_pred + r、K = P_pred/S；新息 innov = y_t − x_pred；`filtered[t] = x_pred + K·innov`、`variance[t] = (1 − K)·P_pred`；似然累加 −0.5(ln 2π + ln S + innov²/S)；④ 若非最后一步，把 x_pred/p_pred 推进为更新后的值再加 q；⑤ 返回 `filtered`/`predicted`/`variance`/`predicted_variance`/`gain`/`loglik`
- **复杂度**：时间 O(n) / 空间 O(n)
- **参数**：`y` 一维观测序列（时间先后）；`q` 状态噪声方差，**必须严格 > 0**（0 或负、NaN、inf 都抛 `ValueError`，`_as_positive` 用 `np.isfinite` 拦 NaN）；`r` 观测噪声方差，同样严格 > 0（默认 q = r = 1.0）。本函数**没有 `x0`/`P0` 参数**（初值写死为 y[0] 与 q，要自定义初值请用 `kalman_filter_linear`）、**没有平滑参数**（只做滤波，E[x_t|y_{1..n}] 未实现）、**没有缺失值处理**（`y` 不能含 NaN）、**没有参数估计**（q/r 必须给定，本函数不返回 MLE）
- **陷阱**：**初值是"先验"而不是"估计"**——x_0 = y[0]、P_0 = 0 会让第一个观测的新息恰好为 0，等价于把 y[0] 当成已知常数，这不是标准 MLE 的似然（少了一个自由参数），与 `statsmodels.UnobservedComponents` 的默认初值处理不同，**对数似然不可直接比较**；`r → 0` 时滤波值收敛到观测值（K → 1），此时似然会发散到 +inf，本函数对 r 只要求 > 0，极小 r 下 `loglik` 数值上很大是正常的；该模型只能拟合"水平缓慢漂移"的序列，有明显趋势/季节时必须改用带斜率或季节分量的状态空间模型（可用 `kalman_filter_linear` 自行组装）；**没有平滑**（E[x_t | y_{1..n}]），平滑需要后向递推，未实现
- **怎么检验**：① **等价性**：用 `kalman_filter_linear` 配 F = [[1]]、H = [[1]]、Q = [[q]]、R = [[r]]、x0 = [y[0]]、P0 = [[0]]，断言两函数的 `filtered`/`predicted`/`variance`/`loglik` 逐位一致（这是最有力的独立验证）；② **极限行为**：r → 0（极小值）断言 `gain` 全部接近 1、`filtered ≈ y`；q → 0 断言滤波退化为对 y[0] 的常数估计、`gain` 递减；③ 断言 `filtered[0] == y[0]`、`predicted[0] == y[0]`（第一个新息为 0 的直接后果）；④ 独立实现：手写标量递推对拍；⑤ 非法输入：`q = 0`、`r = -1`、`r = np.nan` 抛 `ValueError`；⑥ `_self_test()` 中有对应断言（`ts_` 前缀键）

#### `kalman_filter_linear(y, F, H, Q, R, x0, P0)`

- **数学形式**：状态空间 α_t = F·α_{t−1} + η_t（η ~ N(0, Q)）、y_t = H·α_t + ε_t（ε ~ N(0, R)）；递推：a = F·x、P = F·P·Fᵀ + Q；新息 v = y_t − H·a、S = H·P·Hᵀ + R；增益 K = P·Hᵀ·S⁻¹；更新 x = a + K·v、P = P − K·H·P；似然累加 −0.5[m·ln 2π + ln|S| + vᵀ·S⁻¹·v]
- **步骤**：① 用 `as_matrix`/`as_vector` 校验 F 为方阵 (k, k)、H 的列数 = k（一维 `(k,)` 会被当成 1 行）、Q 为 (k, k)、R 为 (m, m)、x0 长度 = k、P0 为 (k, k)，任一形状不符抛 `ValueError`；② `y` 允许一维 (T,)（单通道，reshape 成 (T, 1)）或二维 (T, m)，ndim > 2 抛 `ValueError`，空数组抛 `ValueError`，含 NaN/inf 抛 `ValueError`（**不支持缺失观测**），列数与 H 的行数不符抛 `ValueError`；③ 初始化 `x_cur = x0`、`p_cur = P0`，预分配 `state`/`predicted`/`cov`/`innovation`；④ 逐 t 做标准前向递推（**t = 0 也先预测再加 Q**，即 P_pred(0) = F·P0·Fᵀ + Q）；⑤ 用 `np.linalg.slogdet` 算 ln|S|，若 sign ≤ 0 抛 `ValueError`（提示检查 R/Q/H）；⑥ 用 `np.linalg.inv(S)` 显式求逆；⑦ 累加似然并写回各数组；⑧ 返回 `state`/`cov`/`predicted`/`innovation`/`loglik`
- **复杂度**：时间 O(T(k³ + m·k² + m³)) / 空间 O(T(k² + m))
- **参数**：`y` 观测值，一维 (T,) 单通道或二维 (T, m) 多通道，**不能含 NaN/inf**；`F` (k, k) 状态转移矩阵（必须方阵，否则 `ValueError`）；`H` (m, k) 观测矩阵（**一维 (k,) 会被当成 1 行**）；`Q` (k, k) 状态噪声协方差；`R` (m, m) 观测噪声协方差；`x0` (k,) 初始状态均值；`P0` (k, k) 初始状态协方差。全部为位置参数、**无默认值**，形状不符一律 `ValueError`。本函数**没有平滑参数**（只做前向滤波）、**没有缺失值/NaN 处理**、**没有平方根滤波或 Joseph 形式开关**、**没有参数估计**（F/H/Q/R 必须给定）
- **陷阱**：`S` 用显式求逆（`np.linalg.inv`），数值上不如 Cholesky 稳定，`R` 接近奇异或状态维度很高时应改用平方根滤波，本实现不做；协方差更新用 `P − K·H·P`（而非 Joseph 形式），两者数学等价但前者在极端病态问题下可能失去对称正定性，`logdet` 为负时本函数抛 `ValueError`；初值 `x0`/`P0` 是**先验**——本函数在 t = 0 先预测再加 Q（即 P_pred(0) = F·P0·Fᵀ + Q），因此它与 `kalman_filter_local_level` 完全一致**当且仅当**取 P0 = 0（见后者的陷阱说明）；不做平滑、不做缺失值处理（`y` 中不能有 NaN，`as_vector`/本函数都会拒绝）
- **怎么检验**：① **收敛到特例**：F = [[1]]、H = [[1]]、Q = [[q]]、R = [[r]]、x0 = [y[0]]、P0 = [[0]] 时与 `kalman_filter_local_level(y, q, r)` 的 `filtered`/`variance`/`loglik` **逐位相同**（`_self_test` 走的就是这条交叉验证）；② **纯先验极限**：R 很大时 `gain` → 0，`state` 应保持 x0 沿 F 演化，可与闭式 F^t·x0 对拍；③ 一维 `H`（形状 (k,)）与等价二维 `H.reshape(1, -1)` 断言结果一致（验证"一维当成 1 行"的约定）；④ 独立实现：对 k = 1、T 小的情况手写矩阵递推逐位对拍；⑤ 非法输入：F 非方阵、H 列数不符、Q/R/P0 形状不符、`y` 含 NaN、`y` 为空、`y` 的 ndim = 3，都应抛 `ValueError`；⑥ 构造 R = 0 使 S 奇异，断言抛 `ValueError`（logdet 检查）

#### `ljung_box(residual, lags=10)`

- **数学形式**：Q = n(n+2)·Σ_{k=1..L} ρ_k²/(n−k)，其中 ρ_k = [(1/n)Σ_{t=k+1..n} x_t·x_{t−k}] / ρ_0、x = residual − mean(residual)、ρ_0 = (1/n)Σ x_t²；在 H₀（前 L 阶自相关全为 0）下 Q ~ χ²(L)；p 值 = P(χ²(L) > Q) = Q(L/2, Q/2)（上不完全 gamma 的规范化形式，本模块自实现）
- **步骤**：① `as_vector` 归一 `residual`，`_as_int` 校验 `lags` ≥ 1；② 若 L ≥ n 抛 `ValueError`；③ 中心化 x = residual − mean(residual)；④ ρ_0 = xᵀx/n，若 ≤ 0（常数残差）抛 `ValueError`；⑤ 对 k = 1..L 用 `np.dot(x[k:], x[:n-k])/n/ρ_0` 算 ρ_k（**统一用 1/n 归一化**，不是 1/(n−k)）；⑥ 按 1/(n−k) 加权算 Q；⑦ 若 Q ≤ 0 直接取 `p_value = 1.0`，否则调 `_chi2_sf(Q, L)`；⑧ 返回 `stat`/`df`（= L）/`p_value`/`acf`（长度 L 的 ρ_1..ρ_L）
- **复杂度**：时间 O(L·n) / 空间 O(n)
- **参数**：`residual` 一维残差序列（例如 `arima_fit(...)["residual"]`，**注意其前 t0 位是占位的 0**）——L ≥ n 抛 `ValueError`，常数残差抛 `ValueError`；`lags` 最大滞后阶数 L，真整数且 ≥ 1（默认 10），**必须小于序列长度**。本函数**没有 `df_adjust`/`n_params` 参数**（自由度口径写死为 L，不扣已估参数个数，要严谨请自行扣减）、**没有 `fitdf` 参数**（statsmodels 风格的参数名在本仓库不存在）、**没有自动截断占位 0 的开关**、**没有 `box_pierce` 口径开关**（统计量固定为带 1/(n−k) 修正的 Ljung-Box 形式）
- **陷阱**：**自由度口径**——严谨做法是 df = L − (已估参数个数)，本实现取 df = L，因此对拟合后的残差检验偏**保守**（p 值偏大、不容易拒绝白噪声），做严格结论时请自行按 `forecasting.ar_model` 的阶数等扣减自由度；输入若带占位的 0（ARIMA 条件残差的前 t0 位），会人为压低自相关，使检验偏向"不拒绝"，**正确做法**是先按 t0 截掉占位项，本函数不做这个截断（不猜测调用方意图）；序列长度为 0 方差（常数残差）时无定义，抛 `ValueError`；Ljung-Box 对**高阶滞后**或长记忆过程功效有限，`lags` 的经验取法是 `min(10, n/5)` 或 `2 × 周期长度`，取太大会严重损失功效
- **怎么检验**：① **分布函数闭式**：断言 `_chi2_sf(3.0, 2) == exp(−1.5)`、`_chi2_sf(3.0, 4) == exp(−1.5)·2.5`，以及 df = 10 的 5% 临界值 18.307 给出 ≈ 0.05（`_self_test` 用的是 1e-10 与 1e-3 容差）；② **统计量独立实现**：用手写循环重算 ρ_k 与 Q，断言与 `stat` 逐位一致（`_self_test` 的容差是 1e-10）；③ **极限行为**：iid 正态残差上断言 `p_value > 0.05`（不拒绝白噪声），强周期残差 `np.sin(2π·k/12)` 上断言 `p_value < 1e-6`（必须拒绝）；④ 断言 `df == lags`、`acf.size == lags`；⑤ 与 `statsmodels.stats.diagnostic.acorr_ljungbox` 对拍——**注意自由度差异**（statsmodels 默认不扣参数，与本实现同口径时 `stat` 应一致）；⑥ 非法输入：`lags >= n`、常数残差抛 `ValueError`；⑦ 验证占位 0 的影响：对同一残差分别做"含前 t0 位 0"与"截掉后"的检验，断言前者 p 值更大（定性验证该陷阱）

---

**口径提示（本族跨函数）**：`arima_order_select` 的 `table` 内 `aic`/`bic` 与单独 `arima_fit` 的同名键**不逐位相等**，这是刻意的共同样本对齐；`ljung_box` 的 df 不扣参数、`gm11` 的紧邻均值权重固定 0.5、`garch11_fit` 的 α + β 上界 0.999——这三处都必须在论文的"方法/参数设定"一节写明。


### 3.13 机器学习 —— `examples/algorithms/ml.py`

**这族解决什么问题**：把建模流程里必须交代清楚的每一步——数据划分、标准化、评估指标、kNN/CART/随机森林/梯度提升/高斯朴素贝叶斯/LDA、置换重要性、SMOTE——摊开写成可读的教学透明版实现，用来对照成熟库并解释内部机制，而不是追求工业级速度。

**共同约定**：
- **形状口径**：特征矩阵一律 `(n_samples, n_features)`；分类标签用整数编码（字符串标签直接抛 `ValueError`），回归目标用 float；类别顺序统一取 `np.unique` 的升序，**投票平票时取最小标签**。
- **决策树是 CART**：分类用 Gini 不纯度、回归用方差（总体口径 MSE）；候选切分点只取排序后相邻不同取值的中点，判据恒为 `x[:, f] <= threshold` 走左子树。
- **随机性与返回形态**：所有随机过程走 `_common.rng(seed)`（`None` → `DEFAULT_SEED`），不用 `np.random.*` 全局函数；`standardize_apply`、各 `*_predict` 返回 ndarray，其余返回 dict。样本量上千以后应换 scikit-learn 做交叉验证，用本模块对照结果。

#### `train_test_split(X, y, test_size=0.2, seed=None, stratify=None)`

- **数学形式**：无解析式，是**随机划分算子**：不分层时对 0..n−1 做一次均匀随机置换，前 n_test 个为测试集；分层时对每个类别 c 独立置换，取 round(frac·m_c) 个作该类测试样本
- **步骤**：① 校验样本数 ≥ 2、y 长度与有限性；② test_size 为 float 时 n_test = ceil(test_size·n)，为 int 时直接取该值，再夹到 [1, n−1]；③ 不分层：一次 permutation 切片；④ 分层：每类独立置换，m_c ≥ 2 时该类测试数夹到 [1, m_c−1]，m_c = 1 时全部留训练集，合并后整体再打乱；⑤ 返回 `X_train`/`y_train`/`X_test`/`y_test`
- **复杂度**：O(n) 时间，O(n) 空间（返回的是花式索引产生的副本）
- **参数**：`X`/`y`（长度不一致抛 `ValueError`）；`test_size`（默认 0.2）——float 必须落在 (0,1)、int 必须落在 (0, n)，是 bool 也抛 `ValueError`（防止 `True` 被当成 1）；`seed`（默认 None → `DEFAULT_SEED`）；`stratify`（默认 None = 不分层）——给出标签数组即按类分层，类别数超过 n//2 时抛 `ValueError`
- **陷阱**：分层时各类测试条数分别取整，**总数可能与不分层的 n_test 差 ±K**，报"测试集占比"要以实际返回条数为准；某类只有 1 个样本时被留在训练集（否则训练集将完全没有这个类）；返回的是数组副本，但 `as_matrix` 对 ndarray 不复制，所以别依赖"原数据不被改"
- **怎么检验**：80 个样本按 0.25 划分断言得到 60/20——`_self_test` 就这么断言；断言训练集与测试集下标**无交集、并集恰为全体**（用固定 seed 逐个核对）；分层划分后测试集的正类比例应等于总体比例（`_self_test` 断言 0.5）；与 `sklearn.model_selection.train_test_split` 在同一 seed 下比较**划分大小与分层比例**（随机数流不同，下标不会一致，别比下标）

#### `standardize_fit(X)`

- **数学形式**：mean_j = (1/n)Σᵢx_ij，std_j = √((1/n)Σᵢ(x_ij − mean_j)²)（**ddof = 0**）；std_j = 0 的位置置 1；变换为 (x − mean)/std
- **步骤**：① 转成 (n, d) 矩阵；② 按列求均值；③ 按列求总体标准差；④ 把 0 标准差替换为 1；⑤ 返回 `mean`/`std` 两个长度 d 的数组
- **复杂度**：O(n d) 时间，O(d) 空间（不复制数据）
- **参数**：`X`（(n_samples, n_features)）；**没有可调参数**——标准化口径固定为 Z-score + ddof = 0，要改口径必须自己写
- **陷阱**：用 ddof = 1 会让"标准化后整列标准差恰好为 1"不成立（差 √(n/(n−1)) 的因子），与 numpy 默认口径对不上；常数列被置 1 后该列标准化结果是**全 0**，聚类时它不贡献距离，树模型仍可拿它分裂但永远没有增益；**本函数只看传入的数据，所以"先对全体数据 fit 再划分训练/测试"就是数据泄漏**，必须先划分再 fit
- **怎么检验**：把 `standardize_fit` 的输出喂给 `standardize_apply`，断言每一列均值 ≈ 0、标准差 ≈ 1——`_self_test` 断言两者最大偏差 < 1e-9；断言常数列的 `std` 被置为 1；与 `sklearn.preprocessing.StandardScaler`（默认 ddof = 0）的 `mean_`/`scale_` 对拍；手工用 `X.mean(0)`/`X.std(0)` 复现（这是同一口径的独立写法，可用 numpy 的 `np.average` 与 `np.sqrt(np.average((X-m)**2, axis=0))` 交叉验证）

#### `standardize_apply(X, mean, std)`

- **数学形式**：Z_ij = (X_ij − mean_j) / std_j
- **步骤**：① 转矩阵与两个向量；② 分别校验长度等于特征数（不等抛 `ValueError`）；③ 检查 std 中是否有 0（有则抛 `ValueError`）；④ 按广播返回 (X − mean)/std
- **复杂度**：O(n d) 时间，O(n d) 空间（返回新数组）
- **参数**：`X`（(n_samples, n_features)）；`mean`/`std`（长度必须等于特征数，通常来自 `standardize_fit`）——`std` 含 0 时**直接抛 `ValueError`** 而不是悄悄返回 inf/NaN
- **陷阱**：如果 mean/std 来自与 X 不同的数据（例如用了全体数据的统计量），测试集的"标准化"就带入了训练集看不到的信息，评估会偏乐观（这是最典型的泄漏形式）；本函数不校验 X 与 fit 时是否同源，也没有形状以外的一致性检查
- **怎么检验**：断言"fit 后 apply，列均值 0、标准差 1"（与 `standardize_fit` 一节同）；手工 `(X - m)/s` 与返回值逐元素比较；断言 `std=0` 时抛 `ValueError`、长度不符时抛 `ValueError`；反变换 `Z*std + mean` 应还原 X（到浮点精度）

#### `confusion_matrix(y_true, y_pred, labels=None)`

- **数学形式**：M[i, j] = #{k : y_true_k = labels[i] 且 y_pred_k = labels[j]}
- **步骤**：① 把两个标签数组转成一维 int；② 校验长度一致；③ labels 为 None 时取真实与预测标签并集的升序，否则用给定 labels（重复类别抛 `ValueError`）；④ 用 `np.searchsorted` 映射下标，并用相等性掩码剔除不在 labels 中的样本；⑤ 在 `true*K + pred` 上 `np.bincount` 后 reshape 成 K×K；⑥ 返回 `matrix`（**纯 Python int 的列表的列表**）与 `labels`
- **复杂度**：O(n + K²) 时间，O(K²) 空间
- **参数**：`y_true`/`y_pred`（长度不一致抛 `ValueError`）；`labels`（默认 None = 自动取升序并集）——显式给出时会**静默丢弃**出现在数据里但不在 labels 中的样本
- **陷阱**：显式 `labels` 造成静默丢弃会让多折交叉验证汇总时总数对不上，务必检查 `sum(matrix) == n`；返回 Python list 是为方便 json 序列化，需要矩阵运算要自己 `np.asarray`；**行是真实、列是预测**，写反了 accuracy 不变但 precision/recall 会互换
- **怎么检验**：手算 2×2 例 `y=[0,0,1,1]`、`ŷ=[0,1,1,0]` 断言矩阵为 [[1,1],[1,1]]——`_self_test` 就这么断言；断言 `np.asarray(matrix).sum() == len(y_true)`（在没传 labels 时）；与 `sklearn.metrics.confusion_matrix` 对拍（注意 sklearn 的默认标签顺序也是升序）；显式传一个缺类的 labels，验证被丢弃的样本确实不计入

#### `classification_metrics(y_true, y_pred)`

- **数学形式**：accuracy = (1/n)Σ[y_true = y_pred]；对每个类 c：P_c = TP_c/N_pred_c，R_c = TP_c/N_true_c，F1_c = 2P_cR_c/(P_c+R_c)（分母为 0 记 0）；返回三个指标的**宏平均**（各类算术平均，不加权）
- **步骤**：① 转标签、校验长度一致且非空；② 取真实与预测标签的并集为类别集合；③ 逐类算 TP/FP/FN 与 P/R/F1（分母为 0 的项记 0）；④ 对类别取算术平均；⑤ 返回 `accuracy`/`precision_macro`/`recall_macro`/`f1_macro`/`n_classes`
- **复杂度**：O(n K) 时间（每类做两次布尔比较），O(K) 空间
- **参数**：`y_true`/`y_pred`（等长；不一致或空抛 `ValueError`）；**没有 `average` 参数**——固定宏平均，也没有 `labels`、`zero_division` 之类的开关
- **陷阱**：宏平均**不给类别加权**，1:100 不平衡数据上少数类的坏表现被放大，数值通常明显低于 accuracy，论文必须写清宏/微平均；只在预测里出现、真实里没有的类别（FP 类）也计入平均，与"只按真实类别平均"的另一种口径不同；多分类 F1 也可以先累加 TP/FP/FN 再算（微平均），**本函数不做微平均**
- **怎么检验**：手算例 `y=[0,0,1,1]`、`ŷ=[0,1,1,0]` 的四个指标都应为 0.5——`_self_test` 逐项断言；与 `sklearn.metrics.accuracy_score`/`precision_score(average='macro')`/`recall_score`/`f1_score` 对拍（注意 sklearn 的 zero_division 默认值会影响退化情形）；构造完美预测断言全部为 1；构造三分类数据核对宏平均等于各类指标的算术均值

#### `roc_auc(y_true, scores)`

- **数学形式**：AUC = P(正类得分 > 负类得分) + 0.5·P(相等) = (Σ_{i∈正} rank_i − n₊(n₊+1)/2) / (n₊·n₋)，秩为平均秩（Mann-Whitney U 公式）
- **步骤**：① 转标签并校验 scores 长度一致；② 取类别升序，要求恰好 2 类，**较大标签为正类**；③ 算正负样本数，任一为 0 则抛 `ValueError`；④ 对 scores 求平均秩；⑤ 用秩和公式算 AUC 并返回裸 float
- **复杂度**：O(n log n) 时间（排序求秩），O(n) 空间
- **参数**：`y_true`（必须恰好 2 个不同取值，否则抛 `ValueError`）；`scores`（连续打分，**越大越倾向正类**）——没有阈值参数、不做多分类 one-vs-rest 展开，也没有 `average`/`multi_class` 之类的开关
- **陷阱**：**正类是"较大的那个标签"**——标签 {1,2} 时 2 是正类，{-1,1} 时 1 是正类，传反了得到 1 − AUC；并列秩必须取平均，否则"全部同分"会得到依赖输入顺序的荒谬值（正确结果是 0.5）；只有一类标签时抛 `ValueError` 而不是返回 0.5（0.5 会被误读成"无区分度"）；多分类不适用
- **怎么检验**：完全可分数据断言 AUC = 1.0、分数取反断言 0.0、**全部同分断言 0.5**（平均秩的直接推论）——`_self_test` 三条都断言；与 `sklearn.metrics.roc_auc_score` 对拍（并列数据尤其要试）；用 `scipy.stats.mannwhitneyu` 的 U 统计量按公式换算成 AUC 做第三方校验；直接枚举所有正负对、按定义统计"正 > 负"与"正 = 负"的比例，与秩和公式结果对照

#### `kfold_indices(n_samples, k=5, seed=None)`

- **数学形式**：对 0..n−1 随机置换后，按 base = n//k、rem = n%k 切成 k 段：前 rem 段长 base+1，其余长 base
- **步骤**：① 校验 n ≥ 2、k 为整数且 1 ≤ k ≤ n（bool 抛 `ValueError`）；② 用 `rng(seed)` 做一次 permutation；③ 逐折切连续片段；④ 返回 k 个纯 Python int 列表（每项是该折**测试集**下标）
- **复杂度**：O(n) 时间，O(n) 空间
- **参数**：`n_samples`（≥ 2，否则抛 `ValueError`）；`k`（默认 5）——折数，必须在 [1, n]，k = n 即留一法；`seed`（默认 None → `DEFAULT_SEED`）
- **陷阱**：n 不能被 k 整除时各折大小相差 1，**不能假设所有折等大**（前 rem 折更大），写死索引会错位；折内样本在时间序列上是"未来预测过去"，会造成信息泄漏，时间序列必须用前向滚动划分；返回 list of list 而非 numpy 数组
- **怎么检验**：断言 k 折测试集下标的并集恰好覆盖 0..n−1 且无重复——`_self_test` 对 n = 20、k = 4 就这么断言；断言各折大小只能取 {base, base+1}；与 `sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=...)` 比较**折的大小分布与覆盖性**（随机流不同，下标不会一致）

#### `stratified_kfold_indices(y, k=5, seed=None)`

- **数学形式**：对每个类别 c 的样本下标独立置换后，按类内位置轮流分给 k 折（第 j 个样本进第 j mod k 折）；第 f 折中类别 c 的样本数 = ceil 或 floor((m_c − f)/k)
- **步骤**：① 转标签、校验 n ≥ 2 与 k 范围；② 对每个类取成员下标并独立置换；③ 按 `pos % k` 把下标塞进对应折；④ 每折下标升序排序后返回 k 个列表
- **复杂度**：O(n log n) 时间（每折最后排序），O(n) 空间
- **参数**：`y`（类别标签）；`k`（默认 5）——必须在 [1, n]；`seed`（默认 None → `DEFAULT_SEED`）；**没有 `shuffle` 开关**（一定打乱），也没有按连续标签分箱的选项
- **陷阱**：m_c 不能被 k 整除时**不同折拿到该类样本数会差 1**，比例只能"近似"一致，自测必须用容差而非精确相等；某类样本数少于 k 时必然有折在该类上为空，分层退化（应保证 k ≤ min_c m_c）；极不平衡时每折少数类可能只有 1 个样本，指标方差仍然很大
- **怎么检验**：90:30 不平衡标签、k = 3 时断言每折少数类比例与 0.25 相差 < 0.05——`_self_test` 就这么断言；断言 k 折并集覆盖全部下标且无重复；与 `sklearn.model_selection.StratifiedKFold` 比较每折的类别计数（随机流不同，比的是分布不是下标）；构造某类样本数 < k 的极端数据，验证"有些折该类为空"确实发生，从而支撑文档里的告警

#### `knn_predict(X_train, y_train, X_test, k=3, task='classification')`

- **数学形式**：d(a,b) = ‖a − b‖₂（用展开式 ‖a‖² − 2a·b + ‖b‖² 一次算出）；分类取 k 个最近邻标签的多数投票（平票取最小标签），回归取 k 个邻居目标的算术平均
- **步骤**：① 校验特征数一致、k 为整数且 1 ≤ k ≤ n_train、task 合法；② 分类时标签转 int，回归时转 float；③ 用展开式算 (n_test, n_train) 距离矩阵（浮点负数 clip 后开方）；④ 每个测试点用 `argsort(kind="mergesort")` 取前 k 个最近邻；⑤ 分类投票 / 回归平均，同时记录选中邻居的**平均距离**；⑥ 返回 `pred` 与 `distance`
- **复杂度**：O(n_test·n_train·d + n_test·n_train·log n_train) 时间（矩阵乘法 + 逐行排序），O(n_test·n_train) 空间（存整张距离矩阵）
- **参数**：`X_train`/`X_test`（特征数不一致抛 `ValueError`）；`y_train`；`k`（默认 3）——必须落在 [1, n_train]，**k = n_train 时分类退化为多数类、回归退化为全局均值**；`task`（默认 `"classification"`，另一取值 `"regression"`，其它值抛 `ValueError`）；**没有距离权重参数**（不做距离倒数加权），距离口径固定欧氏
- **陷阱**：**未标准化时 kNN 基本失效**——量纲大的特征独占距离，必须先 `standardize_fit`；平票规则必须显式固定（这里取最小标签），否则二分类偶数 k 时结果依赖排序；`distance` 是邻居的**平均**距离，常被误当成"预测置信度"，它大只说明测试点离训练数据远、不说明预测一定错
- **怎么检验**：造两簇完全可分数据（簇心相距约 11.3、簇内 sd = 0.35）断言准确率恰好 1.0——`_self_test` 就这么断言；构造"最近的 k 个邻居标签已知"的小数据手算投票结果；k = 1 时预测必须等于最近邻标签；k = n_train 时分类结果必须是全局多数类；用暴力双重循环（不做展开式）独立复算距离矩阵并比较；与 `sklearn.neighbors.KNeighborsClassifier` 对拍预测

#### `decision_tree_fit(X, y, max_depth=5, min_samples_split=2, min_samples_leaf=1, task='classification')`

- **数学形式**：分类不纯度 Gini(t) = 1 − Σ_c p_c²，回归不纯度 = (1/n_t)Σ(y − ȳ_t)²；切分判据为最小化加权不纯度 (n_L·I_L + n_R·I_R)/n_t，候选阈值只取排序后相邻不同取值的中点
- **步骤**：① 校验 task、超参数范围（max_depth ≥ 0、min_samples_split ≥ 2、min_samples_leaf ≥ 1）；② 分类时标签转 int 并取升序类别集合；③ 递归建树：先算节点预测值（分类多数类/回归均值）与不纯度；④ 深度到顶、样本数 < min_samples_split 或不纯度为 0 则成为叶子；⑤ 对每个特征排序并用累积和一次算出所有合法候选切分的加权不纯度，取最优；⑥ 若最优切分不能**严格**降低不纯度或会让某侧为空则停在原地成叶，否则递归两侧
- **复杂度**：O(depth · d · n log n) 时间（每个节点对每个特征排序一次），O(n · depth) 空间（递归时切片复制）
- **参数**：`X`/`y`（长度不一致抛 `ValueError`）；`max_depth`（默认 5，≥ 0，**0 表示只有根节点即常数预测**，合法不是失败）；`min_samples_split`（默认 2，≥ 2）——调大即提前停止生长、更强正则；`min_samples_leaf`（默认 1，≥ 1）——每侧最少样本数，调大同样抑制过拟合；`task`（默认 `"classification"`，另一取值 `"regression"`）
- **陷阱**：**贪心且无回看**——XOR 这类单特征上不纯度完全不下降的关系，树会直接停在根节点（本实现要求严格下降），需要先做特征交叉或加深/换模型；阈值只取相邻取值中点，对连续特征做的是**阶梯逼近**，深度不够时线性关系的残差是系统性的而非随机误差；分类树叶子返回**多数类标签不是概率**（要概率得自己存类别分布）
- **怎么检验**：完全可分两簇数据上训练/测试准确率都断言 1.0——`_self_test` 就这么断言；回归任务上断言"**加深必然降低训练 RMSE**"（单调性，`_self_test` 断言 max_depth=5 的 RMSE < max_depth=1 的），这条是独立于实现的通用性质；用单特征一维数据手算最优切分点与阈值（应等于相邻取值中点）并核对 `tree["threshold"]`；与 `sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=...)` 对拍训练准确率（阈值取值规则不同，树结构不会完全一致）

#### `decision_tree_predict(tree, X)`

- **数学形式**：对每个样本从根开始，若 `x[feature] <= threshold` 走 left，否则走 right，直到叶子并返回叶子的 `value`
- **步骤**：① 校验 tree 是 dict 且含 `value` 键（否则抛 `ValueError`）；② 按行循环下行到叶子；③ 若所有返回值都是整数则转 int 数组，否则转 float 数组；④ 返回 ndarray
- **复杂度**：O(n · depth) 时间（逐样本 Python 循环），O(n) 空间
- **参数**：`tree`（必须是 `decision_tree_fit` 的返回 dict，缺失 `value` 键抛 `ValueError`；本函数不校验树是否被手工改过）；`X`（特征数少于树中记录的下标时不报错，numpy 会抛 `IndexError`）
- **陷阱**：**逐样本 Python 循环**，样本量大时很慢，生产环境应把树展开成数组运算；叶子判定依据是"没有 feature 键"，若调用方手工给叶子补了 `feature`，预测会走进 `KeyError` 或死循环；特征数不足的报错信息不如显式校验清楚
- **怎么检验**：断言 `predict(fit(X, y), X) == y`（在完全可分且深度足够的训练集上应完美拟合）；与手写的递归/迭代遍历器对拍每个样本的叶子值；对 `max_depth=0` 的树断言预测恒为常数（分类为多数类、回归为均值）；构造三维数据手工走一遍路径核对阈值判据是 `<=` 走左

#### `random_forest_fit(X, y, n_trees=25, max_depth=6, max_features='sqrt', min_samples_leaf=1, seed=None, task='classification')`

- **数学形式**：对每棵树做一次自助采样（有放回抽 n 个），在随机特征子集 `_feature_subset` 上按 CART 判据建树；袋外得分用"只在该样本属于袋外的那几棵树上"投票（分类）或取均值（回归）后与真值比较；特征重要性 = 各节点加权不纯度下降按特征累加后归一化到和为 1
- **步骤**：① 校验 task、n_trees ≥ 1、max_depth ≥ 0、min_samples_leaf ≥ 1；② 每棵树做 `gen.integers(0, n, n)` 自助采样，记录未被抽中的袋外下标；③ 用随机特征子集建树（min_samples_split 固定传 2）；④ 汇总袋外预测：分类按类别计数取最多者算准确率，回归取均值算 R² = 1 − SS_res/SS_tot；⑤ 累加各节点不纯度下降并按特征归一化；⑥ 返回 `trees`/`feature_importance`/`oob_score`/`n_trees`
- **复杂度**：O(n_trees · depth · d · n log n) 时间，O(n_trees · n) 空间（存所有树与自助样本切片）
- **参数**：`n_trees`（默认 25，≥ 1）——树越多袋外得分方差越小；`max_depth`（默认 6，≥ 0）；`max_features`（默认 `'sqrt'` = `int(sqrt(d))`）——还支持 `'log2'`、int、以及 (0,1] 的浮点比例（按 `ceil(frac·d)` 取），调小 = 更强的去相关、单树更弱；`min_samples_leaf`（默认 1，≥ 1）；`seed`（默认 None → `DEFAULT_SEED`）；`task`（默认 `"classification"`）
- **陷阱**：自助抽样下**每棵树大约只用 63.2% 的样本**，其余是袋外，所以 OOB 天然比训练得分可信，但树少（< 10）时方差很大；袋外样本数为 0 的样本会被排除在 OOB 之外，若全部样本都没有袋外预测则 `oob_score` 返回 0.0（不能解读为"模型完全不能用"）；特征重要性按**训练集**不纯度下降累计，对高基数/连续特征有偏好，要更公平请用 `permutation_importance`；回归的 OOB R² 可以为负
- **怎么检验**：完全可分两簇上断言测试准确率 1.0、袋外准确率 ≥ 0.9、重要性之和为 1——`_self_test` 三条都断言；用**同一 seed 重跑两次**断言结果逐位一致（确定性与随机流独立性）；把 n_trees 从 10 增到 100，检查 OOB 得分的波动变小（这是袋外估计的基本性质）；与 `sklearn.ensemble.RandomForestClassifier(oob_score=True)` 对拍 OOB 量级；**置换验证**：打乱 y 后 OOB 准确率应跌到多数类基线附近

#### `random_forest_predict(forest, X)`

- **数学形式**：分类：对每棵树取硬预测标签，按类别计数取最多者（**平票取最小类别**）；回归：各树预测的算术平均
- **步骤**：① 校验 forest 是含非空 `trees` 的 dict（否则抛 `ValueError`）；② 逐棵树调用 `decision_tree_predict` 并堆成 (n_trees, n) 矩阵；③ 用第一棵树的输出 dtype 判定分类/回归；④ 分类时取类别集合（优先 `forest["classes"]`，否则从预测结果推 `np.unique`）后计数投票，回归时取列均值；⑤ 返回 ndarray
- **复杂度**：O(n_trees · n · depth) 时间，O(n_trees · n) 空间
- **参数**：`forest`（必须是 `random_forest_fit` 的返回 dict，缺 `trees` 键抛 `ValueError`）；`X`（列必须与训练时同序，本函数**不做任何特征名对齐**）；**没有概率输出参数**——分类是硬标签
- **陷阱**：分类投票是**硬投票**（每棵树一票），没有用叶子概率做软投票，树数少且叶子样本少时方差比软投票大；平票取最小类别，二分类且树数为偶数时平票很常见，换个种子可能整片样本的预测都翻转；代码注释说类别集合"优先取 fit 时记录的 classes"，但 `random_forest_fit` 的返回 dict **并没有 `classes` 键**，所以这条分支目前实际不可达（见文末不一致报告）
- **怎么检验**：断言随机森林预测与"逐棵树预测后手动投票"的结果一致（把 `decision_tree_predict` 的堆叠结果按平票规则手工重算）；断言树数 = 1 时森林预测与那棵单树预测完全相同；分类输出必须始终落在训练时的类别集合内；与 `sklearn.ensemble.RandomForestClassifier.predict` 对拍（树结构与随机流不同，只比准确率量级）

#### `random_forest_feature_importance(forest)`

- **数学形式**：imp_f = Σ_{节点 t: feature(t)=f} max(n_t·I_t − n_L·I_L − n_R·I_R, 0)，再除以 Σ_f imp_f 归一化
- **步骤**：① 校验 forest 是含非空 `trees` 的 dict；② 从 `forest["feature_importance"]` 的长度推断特征数 d（缺失则抛 `ValueError`）；③ 用显式栈遍历每棵树的全部内部节点，累加负贡献截断为 0 的不纯度下降；④ 归一化（全 0 时返回均匀分布 1/d）；⑤ 返回长度 d 的 ndarray
- **复杂度**：O(n_trees · 节点数) 时间，O(d) 空间
- **参数**：`forest`（必须是 `random_forest_fit` 的返回 dict）；**没有可调参数**（没有归一化开关，也不支持按 OOB 置换口径计算——那要用 `permutation_importance`）
- **陷阱**：这是**训练集上的不纯度重要性**，对连续/高基数特征有系统性偏好，不能当因果重要性用；相关性强的特征之间会"分走"重要性，即使两个特征都重要，各自也可能只拿到一半，不要据此排除特征；全部为 0 时返回均匀分布是个**约定**（不是真的等权重要），调用方应结合 `max_depth` 判断是否退化
- **怎么检验**：断言返回值的和为 1（`_self_test` 就这么断言）且全部非负；`max_depth=0` 时断言返回均匀分布 1/d；造"只有第 j 列与 y 有关"的数据，断言该列重要性明显最高（定性）；与同一森林跑 `permutation_importance` 比较排名（两种口径不必一致，但强信号特征应都在前面）；把某列整体替换成噪声后重训，断言该列重要性下降

#### `gradient_boosting_fit(X, y, n_estimators=30, learning_rate=0.1, max_depth=2, task='regression')`

- **数学形式**：F₀ = mean(y)（回归）或 log(p₀/(1−p₀))（分类，p₀ 为 y 均值截断到 [1e-6, 1−1e-6]）；第 t 轮用 CART 回归树 h_t 拟合负梯度（回归残差 y − F；分类残差 y − sigmoid(F)），再令 F ← F + ν·h_t(X)；训练损失记录为 MSE（回归）或二元交叉熵（分类）
- **步骤**：① 校验 task、n_estimators ≥ 1、0 < learning_rate ≤ 1、max_depth ≥ 0；② 回归：初始化 F₀ = mean(y)，逐轮拟合残差、按学习率累加、记录 MSE；③ 分类：要求恰好 2 类，把**较大标签当正类**转成 0/1，初始化对数几率，逐轮拟合 y − sigmoid(F)、累加、记录交叉熵；④ 返回 `init`/`trees`/`learning_rate`/`task`/`train_loss`
- **复杂度**：O(n_estimators · depth · d · n log n) 时间，O(n_estimators · n) 空间
- **参数**：`n_estimators`（默认 30，≥ 1）——提升轮数/树数，调大会持续降低训练损失（本实现**没有早停**）；`learning_rate`（默认 0.1，必须落在 (0, 1]）——收缩系数，**太大会震荡、太小需要很多棵树**；`max_depth`（默认 2，≥ 0）——每棵回归树的深度，基学习器通常要浅；`task`（默认 `"regression"`，另一取值 `"classification"`，分类只支持二分类，多分类抛 `ValueError`）
- **陷阱**：这是**函数空间最速下降的近似**——叶子取值直接用回归树的均值，没做 Friedman 的"单步 Newton 修正"（叶子值线性搜索），同样轮数下收敛比标准 GBM 慢但方向正确；分类只支持二分类且把**较大标签当正类**；`train_loss` 是**训练集**损失，单调下降不代表验证集在下降，不能用来判断过拟合；`learning_rate = 1.0` 会震荡
- **怎么检验**：拟合 y = 2x + 1 时断言训练损失首尾下降、最终 MSE 远小于 y 的方差（`_self_test` 断言 MSE < 0.05·var(y)）；断言 `train_loss` 序列**单调不增**（这是提升法每一步都在拟合负梯度的直接推论，本实现用回归树拟合残差应满足）；断言 `len(model["trees"]) == n_estimators` 且 `len(train_loss) == n_estimators`；用 n_estimators=1 的模型手工核对 F = init + ν·tree(X)；分类时断言预测标签与 `sigmoid(F) > 0.5` 一致；与 `sklearn.ensemble.GradientBoostingRegressor(loss='squared_error')` 比较损失下降的**量级**（叶子取值策略不同，不要求数值一致）

#### `gradient_boosting_predict(model, X)`

- **数学形式**：回归：ŷ = init + ν·Σ_t h_t(X)；分类：ŷ = 1[init + ν·Σ_t h_t(X) > 0]（即 sigmoid > 0.5 的硬标签）
- **步骤**：① 校验 model 是含 `trees` 的 dict（否则抛 `ValueError`）；② 用 `init` 填充 F；③ 按训练顺序逐棵累加 ν·tree(X)；④ 按 `model["task"]` 返回浮点预测或 0/1 硬标签
- **复杂度**：O(n_estimators · n · depth) 时间，O(n) 空间
- **参数**：`model`（必须是 `gradient_boosting_fit` 的返回 dict）；`X`（列需与训练同序）；**没有阈值参数**——分类阈值**硬编码为 0.5**（F > 0），要按业务代价调阈值必须自己取 F 后重算
- **陷阱**：分类返回的是**硬标签**，阈值固定 0.5，不要以为它一定最优；累加顺序影响浮点末位，同一模型两次调用逐位一致，但与其他实现比对时不要用 1e-15 级容差；训练用了几棵树预测就必须用完整的 `model["trees"]`，想早停要在 fit 阶段截断树列表
- **怎么检验**：断言用该模型预测训练集时，回归的残差与 `train_loss[-1]` 满足 MSE 恒等式；断言 `n_estimators=1` 时预测等于 `init + ν·tree(X)`（可用 `decision_tree_predict` 手工重算）；断言分类输出只含 0/1 且与手工取 sigmoid 后比 0.5 的结果一致；断言重复调用同一模型结果**逐位相同**

#### `gaussian_nb_fit(X, y)`

- **数学形式**：先验 P(c) = n_c/n；每类每特征 μ_cj = mean_{i∈c}x_ij，σ²_cj = (1/n_c)Σ_{i∈c}(x_ij − μ_cj)²（**ddof = 0**）再加 1e-9 平滑
- **步骤**：① 转标签、校验长度；② 类别取升序，逐类取子集；③ 空类抛 `ValueError`；④ 算先验、逐类逐列均值与总体方差；⑤ 方差统一加 1e-9；⑥ 返回 `classes`/`prior`/`mean`/`var`
- **复杂度**：O(n d) 时间，O(K d) 空间
- **参数**：`X`/`y`（长度不一致抛 `ValueError`）；**没有可调参数**——平滑量 1e-9 是硬编码的，方差口径固定 ddof = 0，没有 `var_smoothing` 之类的开关
- **陷阱**：**1e-9 是绝对量级平滑，不是相对量级**——特征量纲极小（1e-6）或极大（1e6）时它都几乎不起作用，它只防除零、不是真正的正则化，要稳健请先标准化；ddof = 0 与 sklearn 的 `GaussianNB` 一致，但与"样本方差"教材口径差 n/(n−1)；特征必须独立才叫"朴素"，强相关特征会让似然被重复计入、后验过度自信，本实现不做相关校正
- **怎么检验**：用两簇完全可分数据断言预测准确率 1.0——`_self_test` 就这么断言；**手工按类算均值与方差**并与返回的 `mean`/`var` 逐元素比较（`var` 应恰好差 1e-9）；断言 `prior` 之和为 1；与 `sklearn.naive_bayes.GaussianNB`（`var_smoothing` 置 0 或极小以对齐平滑口径）对拍预测标签；构造方差为 0 的特征列，验证预测仍返回有限值（平滑生效）

#### `gaussian_nb_predict(model, X)`

- **数学形式**：log p(x|c) = −0.5·Σ_j[ln(2π σ²_cj) + (x_j − μ_cj)²/σ²_cj]；未归一化对数后验 = log p(x|c) + ln P(c)；再做"减最大值 + logsumexp"归一化，使每行指数域和为 1
- **步骤**：① 校验 model 含 `classes` 键、prior/mean/var 行数一致、特征数一致、var 全为正（否则抛 `ValueError`）；② 逐类算对数似然并加 `log(prior)`；③ 每行减最大值避免下溢；④ 减去 logsumexp 得归一化对数后验；⑤ 取 argmax 映射回原始标签；⑥ 返回 `pred` 与 `log_prob`
- **复杂度**：O(n K d) 时间，O(n K) 空间
- **参数**：`model`（必须是 `gaussian_nb_fit` 的返回 dict，`var` 含非正值抛 `ValueError`）；`X`（特征数与 model 不一致抛 `ValueError`）；**没有可调参数**（无温度、无阈值）
- **陷阱**：`log_prob` 是**归一化后**的对数后验，每行指数域和为 1，可以放心比较两类大小（直接比较未归一化的值在特征维度大时会整体下溢为 0——这正是要减最大值的原因）；先验为 0 的类不会出现（`np.unique` 只给出出现过的类），所以模型**永远不会预测没见过的类**；特征维度很大时所有类的后验可能接近 0/1 两极，表现为"过度自信"，这是独立性假设的固有后果
- **怎么检验**：断言每行 logsumexp ≈ 0——`_self_test` 断言最大偏差 < 1e-9；断言 `pred` 等于 `argmax(log_prob)` 映射回的标签；断言 `exp(log_prob)` 每行和为 1；用两类等方差等先验的一维数据手算决策边界（应在两均值中点）并与预测的翻转位置对照；与 `sklearn.naive_bayes.GaussianNB.predict_log_proba` 对拍（口径对齐后应一致）

#### `lda_fit(X, y, n_components=None)`

- **数学形式**：类内散度 Sw = Σ_c Σ_{i∈c}(x_i − μ_c)(x_i − μ_c)'，类间散度 Sb = Σ_c n_c(μ_c − μ)(μ_c − μ)'；白化 Sw = VΛV' → W = VΛ^{−1/2}（λ 截断到 1e-12），对 M = W'SbW 做对称特征分解，scalings = WU，特征值降序后 clip(0, ∞) 并归一化为 explained_ratio
- **步骤**：① 校验至少 2 类、n_components 为整数且在 [1, d]；② 算总体均值、逐类的 Sw 与 Sb、类先验；③ 对称化 Sw 与 Sb；④ 对 Sw 做 `eigh` 并把特征值截断到 ≥ 1e-12 后构造白化矩阵；⑤ 对 M = W'SbW 做 `eigh`，按特征值降序（`mergesort` 保证可复现）截断负值、归一化；⑥ 返回 `scalings`/`eigenvalues`/`explained_ratio`/`classes`/`mean`/`priors`
- **复杂度**：O(n d² + d³) 时间（散度矩阵累加 + 两次对称特征分解），O(d²) 空间
- **参数**：`X`/`y`（少于 2 类抛 `ValueError`，某类无样本抛 `ValueError`）；`n_components`（默认 None = `min(K−1, d)`）——显式给出时必须落在 [1, d]，**注意它允许超过 K−1**（此时会返回数值上接近 0 的多余方向），而解释率之和在截取前是对全部 d 个非负特征值归一化的
- **陷阱**：Sw 奇异（样本数少于特征数，或某特征在类内完全不变）时白化会放大噪声方向，本实现把 λ 截断到 1e-12，因此**不报错但可能给出巨大且不稳定的方向**，实践中应先 PCA 降维或标准化；`explained_ratio` 按全部非负特征值归一化，`n_components < K−1` 时返回的若干 ratio 之和会小于 1，这是正常的；判别方向不唯一（同一特征值可对应任意旋转后的方向），比较时请比投影后的判别能力而不是直接比 `scalings` 的数值；只做线性判别，没有贝叶斯决策规则（概率模型见 `gaussian_nb_*`）
- **怎么检验**：三类高斯数据上断言 `explained_ratio` 之和为 1，且在投影空间里**最近质心分类 100% 正确**——`_self_test` 两条都断言，后者与 LDA 自身的目标函数独立；断言返回的 `eigenvalues` 降序且非负；手工构造两类等协方差数据，核对第一判别方向与 `Sw^{−1}(μ₁ − μ₂)` 平行（只比方向不比长度）；与 `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` 对拍特征值比例与投影后的分类准确率

#### `lda_transform(model, X)`

- **数学形式**：T = (X − μ) · scalings，其中 μ 是训练时的总体均值
- **步骤**：① 校验 model 含 `scalings` 键（否则抛 `ValueError`）；② 校验 mean 长度与 scalings 行数都等于 X 的特征数（否则抛 `ValueError`）；③ 返回 `(x - mean) @ scal`
- **复杂度**：O(n d c) 时间，O(n c) 空间
- **参数**：`model`（必须是 `lda_fit` 的返回 dict）；`X`（特征数不一致抛 `ValueError`）；**无可调参数**
- **陷阱**：**必须平移**——直接 `X @ scalings` 会保留全局均值，投影后的类质心虽仍可分，但数值与预期不一致（判别方向本身不含截距）；投影后的各维尺度没有归一化，特征值大的方向数值也大，画图前通常要自己缩放；投影维度 ≤ K−1，别期望它任意降维
- **怎么检验**：断言 `lda_transform(model, X)` 的类质心等于 `(各类均值 − 总体均值) @ scalings`（手算，独立于投影函数本身）；断言对训练数据投影后再做最近质心分类与 `lda_fit` 的目标一致（`_self_test` 断言 100% 正确）；断言 `lda_transform(model, model_mean_row)` 的结果接近 0（平移生效的直接体现）；与 `sklearn` 的 `transform` 对拍（方向符号可能相反，比较投影后的类间可分性）

#### `permutation_importance(predict_fn, X, y, metric_fn, n_repeats=5, seed=None)`

- **数学形式**：imp_f = base − (1/n_repeats)Σ_r score(X^{(f,π_r)}, y)，其中 base = score(X, y)，X^{(f,π_r)} 是只把第 f 列按置换 π_r 重排后的矩阵；`importances_std` 为各次重复得分的标准差（ddof = 0）
- **步骤**：① 校验 y 长度、predict_fn/metric_fn 可调用、n_repeats 为整数 ≥ 1；② 算基准得分；③ 对每个特征重复 n_repeats 次：复制 X、把第 f 列整体按 `gen.permutation` 重排、重新预测并算得分；④ 重要性 = base − 打乱后得分；⑤ 返回 `importances_mean`/`importances_std`
- **复杂度**：O(n_features · n_repeats · 预测开销) 时间，O(n d) 空间（每次都复制一份 X）
- **参数**：`predict_fn`（`predict_fn(X) -> 预测数组`）；`metric_fn`（`metric_fn(y_true, y_pred) -> float`，**必须"越大越好"**，例如准确率或负 MSE）；`n_repeats`（默认 5，≥ 1）——重复次数，**调大降低重要性估计的随机波动**（`importances_std` 正是这个波动的度量）；`seed`（默认 None → `DEFAULT_SEED`）
- **陷阱**：只打乱**一列**时，与该列强相关的其他列仍在，模型仍能部分恢复信息，因此相关特征组会**分摊**重要性，单个特征的重要性看起来偏小甚至为负；重要性可以是负数（打乱后反而变好），这是噪声或数据泄漏的信号，不要强行截断为 0 后再比较；每次重复的置换不同，结果带随机性，必须固定 seed 才能复现；`metric_fn` 方向必须是"越大越好"，传 MSE 进来会得到反号的结论
- **怎么检验**：造一个"预测只依赖第 0 列"的模型（如 `lambda Z: 3*Z[:,0]`），断言第 1 列的置换重要性**恒为 0**、第 0 列为正——`_self_test` 就这么断言，这条是纯逻辑推论、与实现无关；固定 seed 重跑两次断言结果完全相同；把 n_repeats 从 5 提到 50，检查 `importances_std` 明显下降；与 `sklearn.inspection.permutation_importance` 对拍（比较 `importances_mean` 的量级与排序，随机置换不要求逐位一致）

#### `smote(X, y, k=5, seed=None)`

- **数学形式**：对每个少数类 c（n_c < n_max），取 `k' = min(k, n_c−1)` 个同类最近邻；重复 n_max − n_c 次：随机取种子样本 i 与近邻 j，生成 x_new = x_i + u·(x_j − x_i)，u ~ U(0,1)
- **步骤**：① 校验标签、k 为整数 ≥ 1；② 统计各类样本数，取多数类 n_max（少于 2 类抛 `ValueError`）；③ 对每个不足的类算同类欧氏距离矩阵、对角线置 inf、取前 k' 个邻居；④ 循环 need = n_max − n_c 次按公式插值（少数类样本数 < 2 时抛 `ValueError`）；⑤ 原样本在前、合成样本在后拼接；⑥ 返回 `X`/`y`/`n_synthetic`
- **复杂度**：O(Σ_c n_c² d) 时间（每个少数类的完整同类距离矩阵），O(Σ_c n_c² + n_total·d) 空间
- **参数**：`X`/`y`；`k`（默认 5，≥ 1）——用于插值的近邻数，**实际取 `min(k, n_c − 1)`，这一缩减不会体现在返回值里**，论文里要写清实际用的 k；`seed`（默认 None → `DEFAULT_SEED`）；**没有 `sampling_strategy` 参数**——固定把所有类补到多数类数量
- **陷阱**：**必须在划分训练/测试之后做**，先 SMOTE 再划分会让同一少数类样本的近邻同时出现在训练和测试集，测试指标严重偏乐观；插值只在少数类内部进行，是**过采样而非生成模型**，合成点必然落在原始少数类样本的凸包附近，对高维稀疏数据效果很差；少数类样本数 < 2 时无法插值（抛 `ValueError`，应改用复制或权重）；未标准化时插值方向被量纲大的特征支配
- **怎么检验**：30:10 的两类数据、k = 3 时断言合成数 = 20、过采样后两类数量相等、总样本数 60——`_self_test` 三条都断言；断言每个合成点都落在其"母样本"与某个同类近邻的连线上（可用凸组合系数反解 u ∈ [0,1]）；断言所有合成点仍属于少数类的定义域（如同一侧的簇）；断言多数类样本**完全没有被改动**（返回的 `y` 前 n 个应等于输入 `y`）；与 `imblearn.over_sampling.SMOTE` 对拍生成后的类别计数（随机流不同，别比数值）

#### `class_weight_balanced(y)`

- **数学形式**：w_c = n / (K · n_c)，n 为总样本数、K 为类别数、n_c 为第 c 类样本数
- **步骤**：① 转标签；② 用 `np.unique(..., return_counts=True)` 取类别与计数；③ 按公式算权重；④ 返回 `classes` 与 `weights`
- **复杂度**：O(n) 时间，O(K) 空间
- **参数**：`y`（类别标签，空数组抛 `ValueError`，非数值 dtype 抛 `ValueError`）；**没有可调参数**——没有 `class_weight` 字典、没有幂次调节，公式固定为频率倒数乘 K 归一
- **陷阱**：这个口径下权重之**和不为 1**（等于 (1/K)Σ n/n_c），它是"平均权重为 1"的标度（各类加权后的总权重相同），若框架要求权重和为 1 需自己再归一化；完全平衡的数据上每类权重恰好为 1，越不平衡少数类权重越大，等价于把少数类的损失放大，会牺牲多数类精度换取少数类召回；标签集合由 `np.unique` 决定，**没有出现的类别不会获得权重**，训练集某折缺类时权重向量长度会变，下游按类别对齐务必小心
- **怎么检验**：30:10 的两类数据断言权重为 [40/(2·30), 40/(2·10)] = [2/3, 2]——`_self_test` 就这么断言（闭式值）；断言完全平衡数据上所有权重恰为 1；断言 `Σ_c w_c·n_c = n`（加权后各类总权重相等，这是公式的直接推论）；断言权重与 `K·n_c` 成反比（用两个不同不平衡度的数据集验证比例关系）；与 `sklearn.utils.class_weight.compute_class_weight('balanced', classes, y)` 对拍（sklearn 的口径正是 n/(K·n_c)，应逐位一致）


### 3.14 多准则决策 —— `examples/algorithms/multicriteria.py`

**这族解决什么问题**：`evaluation.py` 已经覆盖了 AHP/熵权/CRITIC/TOPSIS/VIKOR/灰关联/DEA/模糊综合，本模块只补它没有的两族方法。第一族是**级别高于关系（outranking）**：PROMETHEE II 用偏好函数算正负净流，ELECTRE I 用一致性/不一致性矩阵加内核筛方案，ELECTRE III 再加无差异/偏好/否决阈值与升降蒸馏；第二族是**秩方法与共识**：加权秩和比 RSR 及其概率单位分档、Borda 计分、Copeland 计分，以及多份排序之间的 Spearman 秩相关一致性诊断。典型场景是"指标权重和阈值本身就是主观的、效用函数写不出来、专家只肯给排序"的选型/评价题；本模块的中间量（偏好度矩阵、一致性矩阵、可信度矩阵、蒸馏分组）全部显式返回，便于在论文里画出完整计算过程，生产环境可与 `pymcdm` 等成熟库互证。

**共同约定**：
- **指标方向**：`benefit` 是长度 n 的 bool 序列，`True` 表示越大越好（正向指标），`False` 表示越小越好（成本型）；`None` 表示全部正向。成本型指标在内部按 `Z = -X` 做符号翻转，**不修改**调用方传入的矩阵。
- **权重**：长度 n、非负，内部归一化到和为 1（归一化这一步不改变排序，只影响中间量的可读性）。
- **名次**：`rank` 一律"1 为最好"，并列给相同名次（标准竞赛排名法 1,1,3 口径），由 `_ranks_from_scores` 统一生成，返回 int 数组。
- **标准化**：ELECTRE I/III 先做向量归一化 `r_ij = x_ij / √(Σ_i x_ij²)`；PROMETHEE 直接用原值，因为偏好阈值 p/q 本身就带量纲，标准化会破坏它的物理含义。
- **容差与随机性**：结构性比较（相等/支配/对角）统一用模块级 `_EPS = 1e-12`；公开函数全部确定性，只有 `_self_test` 里的随机算例走 `_common.rng()`（独立 Generator），不使用任何 `np.random` 全局状态。
- **刻意不实现的**：没有 ELECTRE II、没有 ELECTRE III 的 λ 参数（λ 内部取非对角可信度均值）、没有把 ELECTRE 的结果直接编成名次（只给内核与支配矩阵）；TOPSIS/VIKOR/AHP/熵权等请回 `evaluation.py`。

#### `promethee_ii(X, weights, benefit=None, preference="linear", p=None, q=None)`

- **数学形式**：Z 为正向化后的矩阵（成本型列取负）；对每个有序对 (a, b) 与指标 j，优势 d = Z[a,j] − Z[b,j]，偏好函数 H_j(d) 四选一——usual 型 1{d > 0}、linear 型 clip((d − q)/(p − q), 0, 1)、level 型 0 / 0.5 / 1 三段、gaussian 型 1 − exp(−d²/(2p²))；偏好度 π(a,b) = Σ_j w_j H_j(Z_a − Z_b)，φ⁺(a) = Σ_{b≠a} π(a,b)/(m − 1)，φ⁻(a) = Σ_{b≠a} π(b,a)/(m − 1)，φ_net = φ⁺ − φ⁻，名次按 φ_net 降序。恒有 Σ_a φ_net(a) = 0
- **步骤**：① `as_matrix` 转 X，m < 2 抛 `ValueError`；② 校验 `preference` ∈ {usual, linear, level, gaussian}；③ 归一化权重、解析 `benefit`，得 Z；④ 逐指标算正向化极差 span，默认 p = span（span ≈ 0 时取 1.0）、默认 q = 0，`usual` 下 p、q 被忽略（内部取 p = 1、q = 0）；⑤ 校验 p > 0、q ≥ 0，linear/level 还要求 0 ≤ q < p；⑥ 逐有序对算 π(a,b) = Σ_j w_j H_j；⑦ 除以 m − 1 得 φ⁺、φ⁻、φ_net，按 φ_net 降序编秩；⑧ 返回 5 个键
- **复杂度**：O(m² n) 时间（双重循环 × 逐指标向量运算）/ O(m² + mn) 空间（π 矩阵 + 决策矩阵）
- **参数**：`X`（(m, n) 决策矩阵）；`weights`（长度 n、非负）；`benefit`（长度 n 的 bool 或 0/1 序列，默认 `None` = 全正向）；`preference` 默认 `"linear"`；`p` 偏好阈值，标量或长度 n，默认该指标正向化后的极差；`q` 无差异阈值，默认 0；标量/序列都接受。非法输入一律 `ValueError`：m < 2、`preference` 非法、p ≤ 0、q < 0、linear/level 下 q ≥ p。本函数**没有**否决阈值 `v`、没有蒸馏、没有 λ，要一票否决与分档请用 `electre_iii`；也没有输出置信区间或统计检验
- **陷阱**：① `p` 的默认值是该指标的极差、**带量纲**：若把不同量纲的指标混在一起又不显式给 p，等价于给每个指标一个"相对极差"阈值，这与先归一化再给固定 p 的结果不同；② `"usual"` 偏好函数对任何微小差异都给满偏好，噪声大的指标会主导排序，此时应改用 `"linear"` 或 `"gaussian"`；③ Σ φ_net = 0 是结构性质（自测的对拍点），但它**不代表**名次对称——φ_net 的绝对值大小与并列结构仍要单独看
- **怎么检验**：`_self_test` 的三条断言可直接重跑——(a) `[[1.0], [2.0], [3.0]]` 配 `[1.0]`、`preference="usual"` 时 φ_net 必须为 [−1, 0, 1]（误差 < 1e-9，手算：值 3 > 2 > 1 给出 0/1 支配矩阵）；(b) 四方案两指标例 `[[5,2],[3,9],[7,4],[1,6]]`、`weights=[0.6,0.4]`、`benefit=[True,False]`、`preference="linear"`、`p=[2,3]`、`q=[0.5,0.5]` 时 |Σ φ_net| < 1e-12；(c) 方向不变性——把成本型列取负并令 `benefit=True`，与不取负并令 `benefit=False`，两条路径的 φ_net 最大绝对差 < 1e-9（自测用 `rng()` 默认种子抽的 6×4 随机矩阵）。另外可独立手算 π 矩阵（3~4 个方案时逐格算），或验证单调性：把某个方案在所有指标上抬高，它的 φ_net 不应下降
- **补充检验（名次稳健性）**：把 `preference` 从 `"usual"` 换成 `"linear"`/`"gaussian"` 重跑，看 `rank` 是否改变——改变剧烈说明结论主要由偏好函数口径而非数据决定，论文里必须报告

#### `electre_i(X, weights, benefit=None, c_threshold=None, d_threshold=None)`

- **数学形式**：先向量归一化 r_ij = z_ij / √(Σ_i z_ij²)；一致性 C(a,b) = Σ_{j: r_aj ≥ r_bj} w_j（权重已归一化，故 C ∈ [0,1]）；不一致性 D(a,b) = max_{j: r_aj < r_bj} (r_bj − r_aj)/(max_i r_ij − min_i r_ij)，若 a 在所有指标上都不劣于 b 则 D(a,b) = 0；级别高于关系 a S b ⇔ C(a,b) ≥ c 且 D(a,b) ≤ d
- **步骤**：① `as_matrix` 转 X，m < 2 抛 `ValueError`；② 归一化权重、解析 `benefit`、得 Z；③ 向量归一化得 R，逐指标算归一化后的极差，极差 ≤ `_EPS` 的列按分母 1.0 处理；④ 逐对算 C（对角为 1）与 D（对角为 0）；⑤ 阈值缺省取非对角均值，校验 c ∈ [0,1]、d ≥ 0；⑥ `relation = (C ≥ c − _EPS) & (D ≤ d + _EPS)`，对角强制 False，存成 int；⑦ 内核按方案下标顺序贪心构造——若 a 已被内核中某方案级别高于则跳过，否则把 a 加入并删掉内核中被 a 级别高于的方案；⑧ 返回 `concordance`/`discordance`/`kernel`/`relation`
- **复杂度**：O(m² n) 时间 / O(m² + mn) 空间
- **参数**：`c_threshold` 默认非对角一致性均值，越界（不在 [0,1]）抛 `ValueError`；`d_threshold` 默认非对角不一致性均值，为负抛 `ValueError`；`X`/`weights`/`benefit` 同上一族口径。本函数**没有** p/q/v 三类阈值（那是 `electre_iii`）、**没有** λ、**不返回名次**——返回的是内核（可能含并列多个方案）与 0/1 支配矩阵，要单一排序请用 `promethee_ii`/`electre_iii`
- **陷阱**：① 不一致性用**逐指标极差**做分母：某个指标若对所有方案几乎相同（极差 ≈ 0），它一旦进入不一致性集合就会被放大成接近 1 的值（这里把极差 0 的列按分母 1.0 处理，等于忽略该列），这就是 ELECTRE 对"无区分度指标"非常敏感的原因；② 阈值缺省取矩阵均值，会让结论与方案集规模强相关，加一个无关方案就可能翻转关系，正式分析里应给出阈值敏感性区间；③ 一致性用 `>=`（并列算"不劣"），所以完全相同方案之间的 C 为 1，会互相"级别高于"
- **怎么检验**：`_self_test` 用完全支配算例 `Xe = [[5.0,5.0],[3.0,3.0],[4.0,4.0]]`、`weights=[0.5,0.5]`、`benefit=[True,True]`：C(0,1) 必须为 1（< 1e-12）、D(0,1) 必须为 0（< 1e-12）、内核必须恰为 `[0]`（0 支配 1 和 2）。另外可独立手算：3 个方案 2 个指标时 C/D 的每个格都能笔算；用返回的 `relation` 检查内核的自洽性（贪心算法保证最终内核内部两两互不"级别高于"）；把 `c_threshold` 调到 1.0、`d_threshold` 调到 0 观察 `relation` 与 `kernel` 的收缩
- **补充检验（阈值敏感性）**：在 c ∈ (0,1)、d ∈ (0,1) 的网格上重跑，记录内核尺寸与成员变化，作为论文里的敏感性表——这也是 ELECTRE I 唯一诚实的用法

#### `electre_iii(X, weights, benefit=None, p=None, q=None, v=None)`

- **数学形式**：对每对 (a,b) 定义逐指标劣势 d_j = Z[b,j] − Z[a,j]；局部一致性 c_j = 1 (d ≤ q)、(p − d)/(p − q) (q < d < p)、0 (d ≥ p)，全局一致性 C(a,b) = Σ_j w_j c_j；局部不一致性 d_j = 0 (d ≤ p 或 c_j = 1)、(d − p)/(v − p) (p < d < v)、1 (d ≥ v)；可信度 S(a,b) = C(a,b) · Π_{j: d_j > C(a,b)} (1 − d_j)/(1 − C(a,b))，截断到 [0,1]；λ 取非对角可信度均值，显著性 s(λ) = max(0, 0.3 − 0.15λ)；最终名次 = 升序名次与降序名次的平均后再按"越小越好"编秩（Roy 的中位数/均值折中）
- **步骤**：① `as_matrix` 转 X，m < 2 抛 `ValueError`；② 权重/`benefit`/Z 同上；③ 默认 q = 0、p = 该指标正向化后的极差（极差 ≈ 0 取 1.0）、v = 极差的 2 倍，校验 q ≥ 0、p > q、v > p；④ 逐对算 c_j、C、d_j、S（对角为 1）；⑤ λ = 非对角可信度均值，s(λ) = max(0, 0.3 − 0.15λ)；⑥ 用 `_distill` 分别做降序（从最好开始）与升序（从最差开始）蒸馏，得到两组分组；⑦ 分组编号映射回名次（降序正向编号、升序反向编号）；⑧ avg = (declining + ascending)/2，`rank = _ranks_from_scores(−avg)`；⑨ 返回 `rank`/`credibility`/`ascending`/`descending`
- **复杂度**：O(m³ + m² n) 时间（蒸馏的反复筛选是三次方项）/ O(m² + mn) 空间
- **参数**：`p` 默认正向化后极差、`q` 默认 0、`v` 默认极差的 2 倍，三者都可给标量或长度 n 的序列；要求 `0 <= q < p < v`，否则 `ValueError`（q < 0、p ≤ q、v ≤ p 各有独立报错）。**没有** c/d 阈值参数，**没有** λ 参数——λ 在内部取非对角可信度均值，要换 λ 只能拿返回的 `credibility` 自己重算蒸馏；也**没有**返回"每个方案的否决来源指标"
- **陷阱**：① **λ 的取法没有唯一标准**：Roy 原文里 λ 由决策者给定，这里为了接口自洽取非对角可信度均值并可通过 `credibility` 自行复算；换 λ 会改变分档，结论必须做敏感性分析；② 否决阈值 v 一旦被越过，可信度会被单个指标直接压到接近 0，排序可能被一个指标"一票否决"——这正是 ELECTRE III 的设计意图，但在指标有量纲差异时非常容易误伤；③ 蒸馏在可信度矩阵过于扁平时会退化成"所有方案一组并列"（`_distill` 里显式兜底），出现这种情况时 `rank` 全为 1，不要误读成"所有方案等价"
- **怎么检验**：`_self_test` 用同一个完全支配算例 `Xe`：(a) S(0,1) 必须为 1、S(1,0) 必须为 0（均 < 1e-12）；(b) 对角线必须全为 1；(c) 可信度整体落在 [−1e-12, 1 + 1e-12]；(d) `rank[0] == 1` 且 `rank[1] == 3`（0 最优），升降蒸馏名次都是 1..m 范围内的合法分组编号。另外可：拿返回的 `credibility` 自己复算 λ 与 s(λ) 并重建分组，检查与 `ascending`/`descending` 对齐；把 v 调到刚大于 p 的极小值，观察 S 塌陷（否决效应）；手工把某个方案在两指标上设成极端差，验证"一票否决"
- **补充检验（λ 敏感性）**：在 λ ∈ [0.5, 0.9] 上重跑蒸馏（用自己的实现或改 λ 后重算分组），看 `rank` 的分档是否稳定；不稳定则论文里不能只给一组名次

#### `rank_sum_ratio(X, weights, benefit=None)`

- **数学形式**：逐指标编秩 R（正向指标中最大值得秩 1，成本型指标中最小值得秩 1，并列取平均秩）；经典秩 R′ = m + 1 − R（"高优指标秩次大"的教材口径）；加权秩和 RSR_raw = Σ_j w_j R′_ij；归一化 rsr = RSR_raw / m = (m + 1 − Σ_j w_j R_ij)/m ∈ [1/m, 1]，**越大越优**（两种写法完全等价，实现按后者计算）
- **步骤**：① `as_matrix` 转 X，m < 2 抛 `ValueError`；② 归一化权重、解析 `benefit`；③ 逐列编秩——正向列直接 `_ranks_from_scores(col)`，成本型列取负后再编，得 (m, n) 秩阵；④ raw = ranks @ w；⑤ rsr = (m + 1.0 − raw)/m；⑥ `rank` = 按 rsr 降序编秩（并列同名次）；⑦ 返回 `rsr`/`rank`
- **复杂度**：O(mn log m) 时间（每列一次排序主导）/ O(mn) 空间
- **参数**：只有 `X`、`weights`、`benefit` 三个参数，**没有**任何阈值参数。m < 2 抛 `ValueError`。要概率单位分档请把返回的 `rsr` 交给 `rsr_distribution`；要保留指标量级信息请改用 TOPSIS/VIKOR（在 `evaluation.py`）；本函数也**没有**"是否对成本型取负"的开关，方向全靠 `benefit`
- **陷阱**：① 秩和比只用到**序信息**，指标的量级差异被完全丢弃：两个方案在某指标上差 0.001 与差 1000 得到同样的秩差，需要保留量级信息时不能用它；② 并列取平均秩会让 rsr 出现非整分数的等间隔栅格；m 很小时（例如 m ≤ 4）秩和比的分辨率极低，分档几乎没有意义；③ 后续概率单位分档用的是"升序位次"的中位秩修正 (位次 − 0.5)/m，口径见 `rsr_distribution`，不要与这里的编秩口径混着写进论文
- **怎么检验**：`_self_test` 两条可直接重跑——(a) 单指标 `[[3.0],[1.0],[2.0]]`、`weights=[1.0]`、`benefit=[True]`：名次必须恰为 [1, 3, 2]（精确整数列表断言），且"全优方案"的 rsr 恰为 1.0（< 1e-12）——手算秩为 [1,3,2]、rsr = (3+1−R)/3 给出 [1.0, 1/3, 2/3]；(b) 5×3 例 `[[8,7,3],[6,9,5],[9,5,2],[5,8,4],[7,6,6]]`、`weights=[0.4,0.35,0.25]`、`benefit=[True,True,False]`：rsr 必须落在 [1/5, 1] 内（容差 1e-12）。另外可：单指标全正向时逐点验证 rsr = (m + 1 − 秩)/m；对全并列数据断言所有 rsr 相同；把某个方案在所有指标上抬高，断言其 rsr 不下降
- **补充检验（与 TOPSIS 对拍）**：同一份数据分别跑 `rank_sum_ratio` 与 `evaluation.py` 的 TOPSIS，若名次差异很大，正是"丢量级信息"的后果——这可以作为论文里选择方法的理由

#### `rsr_distribution(rsr, n_levels=3)`

- **数学形式**：把 rsr 升序排序，第 pos 位（0 起）的中位秩累计频率 p = (pos + 1 − 0.5)/m，概率单位 probit = Φ⁻¹(p)（Acklam 有理逼近 + Halley 修正）；用最小二乘拟合 probit = intercept + slope·rsr，r2 = 1 − SS_res/SS_tot；分档切点取标准正态分位数 Φ⁻¹(j/n_levels)（j = 1..n_levels − 1），按**拟合值** intercept + slope·rsr 落点给档位，档位编号越大越优
- **步骤**：① `as_vector` 转 rsr，m < 3 抛 `ValueError`；② 校验 2 ≤ `n_levels` ≤ m，否则 `ValueError`；③ 用 `mergesort` 升序，逐位算 p 与 `_norm_ppf(p)` 填回原下标；④ 构造设计矩阵 [1, rsr]，`np.linalg.lstsq` 解出 slope/intercept，算 fitted；⑤ SS_tot ≤ `_EPS` 时 r2 记 1.0（避免 0/0），否则 1 − SS_res/SS_tot；⑥ 算 n_levels − 1 个切点；⑦ `grades` 从 1 起，对每个切点累加 (fitted > cut)；⑧ 返回 `probit`/`regression`/`grades`，其中 `regression` 的键固定为 `slope`/`intercept`/`r2`/`n`
- **复杂度**：O(m log m) 时间（排序主导）/ O(m) 空间
- **参数**：`rsr` 一维、长度 m（约定为 `rank_sum_ratio` 的输出，越大越优）；`n_levels` 默认 3，要求 2 ≤ n_levels ≤ m。m < 3 或 n_levels 越界抛 `ValueError`。**没有**"自定义切点"、**没有**"返回各档样本量"、**没有**显著性检验参数——r2 只是拟合优度，不是分档的检验统计量
- **陷阱**：① 分档切点用的是**回归拟合值**而不是原始 probit：这样即使某个方案的 probit 落在极端尾部，档位也由它在回归线上的位置决定，不会因为单点抖动而跳档；② m 很小时中位秩修正 (位次 − 0.5)/m 仍然无法覆盖 (0, 1) 的两端，但本实现不会取到 p = 0 或 1（p ∈ [0.5/m, 1 − 0.5/m]），所以不会出现 ±inf；③ 回归的 r2 接近 1 只说明 probit 与 rsr 线性关系强，**不代表**分档本身有统计显著性；④ 当 rsr 全相同时回归斜率会退化为 0（设计矩阵秩亏），此时分档会全部落在同一档，这是正确行为而不是 bug
- **怎么检验**：`_self_test` 有一个漂亮的闭式自洽对拍：把 probit 值本身当作 rsr 输入（构造 `toy = [Φ⁻¹((i + 0.5)/9), i = 0..8]`，正好等于 m = 9 时的中位秩位置），则回归必须给出 slope = 1、intercept = 0（容差 1e-8）且 r2 ≥ 1 − 1e-8——这同时验证了中位秩修正公式与回归口径。另有：对真实 rsr 的输入断言档位全部落在 [1, 3]，且回归斜率 > 0（rsr 越大越优 ⇒ probit 对 rsr 单调增）。`_norm_ppf` 自身还有独立对拍：Φ⁻¹(0.975) 与闭式值 1.959963984540054 的差 < 1e-9、Φ⁻¹(0.5) 精确为 0、反对称性 |Φ⁻¹(0.025) + Φ⁻¹(0.975)| < 1e-12，并用 `math.erfc` 独立复算 0.5·erfc(−z/√2) 与 0.975 的残差（自测把它 round 到 12 位记录，未做断言）
- **补充检验（分档单调性）**：rsr 升序排序后 `grades` 必须非降；若出现降序，说明拟合斜率为负，应直接查 `regression["slope"]` 的符号

#### `borda_count(rankings)`

- **数学形式**：设某份排名的长度为 L，第 p 位（0 起）的方案得 L − 1 − p 分，未出现的方案得 0 分；总分 scores[a] = Σ_票 (L_v − 1 − pos_v(a))，m = 出现过的最大下标 + 1；名次按总分降序
- **步骤**：① `_parse_rankings` 解析每份排名——必须是互不重复的非负整数下标（非整数、负数、重复下标三种情况各有独立 `ValueError`），同时求 m = 最大下标 + 1；② 逐票逐位累加 L − 1 − p；③ 按总分降序编秩（并列同名次）；④ 返回 `scores`/`rank`
- **复杂度**：O(Σ L) 时间（Σ L 为所有选票长度之和）/ O(m + Σ L) 空间
- **参数**：`rankings` 是 list of list，每项从最优到最差；**允许截断**（只排前几名），也**允许**不同排名覆盖不同方案子集。**没有**权重参数、**没有**"缺项记平均分"的开关、**没有** tie-break 规则。要按"所有方案数 m − 1 分"给被截断票补平均分的口径，得自己先预处理选票；输入是 0/1 逐对矩阵时请改用 `copeland_score`
- **陷阱**：① 截断选票的口径不唯一：这里按"该排名自身的长度"给分（第 1 名得 L − 1 分），另一种常见口径是"所有方案数 m − 1 分"并给缺项记平均分；两种口径在截断深度不同时会给出不同赢家，论文里必须写清楚用的是哪一种；② Borda 会被"推举无关方案"操纵（加入一个永远垫底的新方案会改变相对分差），这是 Borda 的著名缺陷，不是实现 bug；③ `rankings` 里出现重复下标会直接报错而不是按"第一次出现"处理
- **怎么检验**：`_self_test` 三条可直接重跑——(a) 旋转对称的 Condorcet 循环 `[[0,1,2],[1,2,0],[2,0,1]]` 总分必须精确等于 [3.0, 3.0, 3.0]（精确相等断言，手算：每份票分值 2/1/0）；(b) 多数决算例 `[[0,1,2],[0,1,2],[0,2,1]]` 名次必须恰为 [1, 2, 3]，方案 0 的总分必须为 6.0（< 1e-12，每份票它都是第 1 名）；(c) 与 `copeland_score` 交叉验证——自测用 votes = `[[0,1,2],[0,1,2],[1,2,0]]` 构造多数决矩阵后断言 `argmax(copeland 分) == argmin(borda 名次)`，两种独立实现互证。另外可：小样本（3 份票、3 个方案）手算全部分值
- **补充检验（截断口径对照）**：把同一份带截断的选票分别按"自身长度"和"补平均分"两种口径算一遍，比较赢家——这正是陷阱 ① 的可复现实验

#### `copeland_score(pairwise_wins)`

- **数学形式**：W 是形状 (m, m) 的逐对结果矩阵，W[i, j] = 1 表示 i 胜 j、−1 表示 i 负 j、0 表示平局；要求对角线为 0 且反对称 W[i, j] = −W[j, i]；Copeland 分 score_i = Σ_j W[i, j] = 胜场数 − 负场数
- **步骤**：① `as_matrix` 转 W；② 五道校验逐条执行——方阵、取值只含 {−1, 0, 1}（整数性 + 幅值两条）、对角线为 0、反对称，任一违反抛 `ValueError`；③ scores = W.sum(axis=1)（平局贡献 0，求和恰好等于胜场减负场）；④ 按得分降序编秩（并列同名次）；⑤ 返回 `score`/`rank`
- **复杂度**：O(m²) 时间 / O(m²) 空间
- **参数**：只有 `pairwise_wins` 一个参数，**没有**权重参数——权重必须在构造 W 之前就融进胜负判断；**没有** tie-break 参数（并列时名次全为同一值）。只有 0/1（不分平局与负）的"胜场矩阵"会被拒绝，应先转成 −1/0/1 或改用 `borda_count`
- **陷阱**：① 反对称是硬要求，很多从数据里直接数出来的 0/1 矩阵会在这里报错，报错信息明确写了"必须反对称"，不要误以为是自己数据不合法；② Copeland 满足 Condorcet 准则（存在 Condorcet 赢家时它一定排第一），但会出现大面积并列（例如三方案循环对决时三人都是 0 分），这时名次全为 1，需要靠次级准则打破平局；③ 对角线非 0 会被拒绝，因为"方案与自己比较"没有定义
- **怎么检验**：`_self_test` 两条——(a) 传递性算例 `[[0,1,1],[−1,0,1],[−1,−1,0]]` 得分必须精确等于 [2.0, 0.0, −2.0]（精确断言，手算：0 胜 1、2，1 胜 2，2 全负）；(b) 循环对决 `[[0,1,−1],[−1,0,1],[1,−1,0]]` 三人同分（自测记录了 `copeland_cycle_score`，**没有**断言，但闭式显然为全 0）。另外可：由任意一组选票独立构造 W（逐对多数决）再与 `borda_count` 比赢家——自测就是这么做的交叉验证；构造一个 Condorcet 赢家并断言它排第一
- **补充检验（反对称性自检）**：调用前用 `np.allclose(W, -W.T)` 与 `np.all(np.diag(W) == 0)` 自查，能把报错提前到数据准备阶段

#### `rank_consensus(rankings)`

- **数学形式**：对每一对排名 (i, j)，取**共同出现**的方案，用各自在原文里的位次（1 为最好）作秩向量，rho = Pearson(秩_i, 秩_j) 即 Spearman 秩相关；`mean_spearman` = 所有两两 rho 的均值；无并列时闭式 rho = 1 − 6Σd²/(m(m² − 1))；`is_consistent` = (mean_spearman ≥ 0.8)
- **步骤**：① `_parse_rankings` 解析（非整数/负数/重复下标抛 `ValueError`）；② K = 排名份数，`pairwise` 初始化为全 1 的 (K, K) 矩阵；③ 对每对 i < j，按 rankings[i] 的顺序取与 rankings[j] 的交集，共同方案 < 2 个抛 `ValueError` 而不是返回 0；④ 用 `list.index(x) + 1` 取两侧位次向量；⑤ `_spearman` 算 rho，填入对称的两个位置并收集；⑥ `mean_spearman` = 均值（只有一份排名时 `values` 为空，记为 1.0）；⑦ 返回 `mean_spearman`/`pairwise`/`is_consistent`
- **复杂度**：O(K² m) 时间（每对排名一次交集 + 一次秩相关）/ O(K² + Km) 空间
- **参数**：只有 `rankings`（允许截断，同一份内不允许重复下标）。**没有**权重参数、**没有**阈值参数——0.8 硬编码在 `is_consistent` 里，要别的阈值请自己比较 `mean_spearman`；**没有**显著性检验、**没有** Kendall's W 或置换 p 值，要正式结论需自行补充
- **陷阱**：① 共同方案少于 2 个时直接抛 `ValueError`，而不是返回 0：两段几乎不相交的排名之间根本不存在可比较的秩相关；② 0.8 是**经验阈值**而非显著性检验：K 小时即使 rho = 0.8 也可能不显著，反之 K 大时 rho = 0.75 也可能高度显著；③ 该诊断只回答"排序是否相似"，不回答"哪个排序对"；④ 并列名次不影响正确性（实现直接用原文位次作秩，位次本身无并列），但截断使不同对的共同方案集合不同，两两 rho 之间并不可直接横向比较
- **怎么检验**：`_self_test` 三条闭式对拍——(a) 完全相同的排名 `[[0,1,2,3],[0,1,2,3]]` 的 `mean_spearman` 必须为 1（< 1e-12）；(b) 完全反序 `[[0,1,2,3],[3,2,1,0]]` 必须为 −1（< 1e-12）；(c) 混合算例 `[[0,1,2,3],[1,0,3,2]]` 必须等于闭式 1 − 6·4/(4·(16 − 1))（< 1e-12，手算位次差 d = [1, −1, 1, −1]、Σd² = 4）。另外可：只传一份排名，断言 `mean_spearman == 1.0`、`pairwise` 为 1×1 且对角为 1.0；把两份排名的共同方案缩到 1 个，断言抛 `ValueError`；用平均秩口径独立实现 Spearman 与 `pairwise` 逐格对拍
- **补充检验（与置换检验对照）**：对小样本（m = 4、K = 3）自己枚举 m! 种置换，得到 rho 的精确分布，判断 0.8 在这个规模下是否有意义——这是陷阱 ② 的定量复现


### 3.15 多目标优化 —— `examples/algorithms/multiobjective.py`

这一族解决的是"多个目标互相冲突、不存在单一最优解"的问题：正确的输出不是一个点，而是一整条（或一整片）Pareto 前沿，以及"这些解各自牺牲了什么"的量化说明。本模块给出从最朴素的加权和法、ε-约束法到 NSGA-II 的透明实现，均已报告为**教学透明版**：结论只能写成"在给定预算与给定种子上得到的前沿"，不能外推成"这是真正的最优前沿"。

- 目标默认**全部最小化**：支配关系、非支配排序、拥挤距离、超体积、TOPSIS 口径一律按最小化理解；`minimize` 参数只出现在 `weighted_sum_pareto` / `epsilon_constraint_pareto` / `nsga2` 三个函数上，且实现方式是"内部对全部目标取负后仍按最小化处理"。
- 决策向量是形状 `(n_dim,)` 的 `np.ndarray`，`objective_fn(x) -> np.ndarray` 必须返回形状 `(m,)`；`x_bounds` 的口径是 `(lo, hi)` 两个标量 = **1 维**问题，d 维共用同一区间要写 `[(lo, hi)] * d`。
- 随机性一律走 `_common.rng(seed)`（PCG64），不使用任何全局随机状态；`seed=None` 表示 `DEFAULT_SEED = 20240101`，同一 seed 两次调用逐位一致。

#### `pareto_dominates(f1, f2)`

- **数学形式**：`f1 ≺ f2` ⟺ `(∀k: f1_k ≤ f2_k) ∧ (∃k: f1_k < f2_k)`（全部最小化口径）。
- **步骤**：① 用 `as_vector` 校验并拉平两个向量，要求长度一致，否则 `ValueError`；② 返回 `np.all(f1 <= f2) and np.any(f1 < f2)` 的布尔值。
- **复杂度**：时间 O(m) / 空间 O(m)，m 为目标个数（只有一次逐分量比较）。
- **参数**：无任何默认值或可调开关；两个入参 `f1`、`f2` 形状必须同为 `(m,)`。**口径不可配置**——要最大化某个目标，必须由调用方先取负再传进来，函数本身不给 `maximize` 开关。
- **陷阱**：内建"全部最小化"口径，最大化目标忘记取负会得到完全相反的结论，而且不报错（多目标代码里最常见的一类静默错误）；"互不支配"不等于"等价"，两个点可以互不支配而其中一个明显更好；完全相同的两个点互不支配（返回 False），一个点也不支配自己。
- **怎么检验**：断言自反性（`pareto_dominates(a, a)` 为 False）、反对称性（`f1≺f2` 则 `f2⊀f1`）、传递性；用 `[1,2] / [1,3] / [2,3] / [0,5]` 这类手算点逐条断言"第一分量相等、第二分量严格更小也算支配"与"互不支配"两种边界；再把整条前沿两两判一遍，断言第一前沿内部不存在任何支配对（这是独立于排序实现的交叉校验）。

#### `fast_non_dominated_sort(F)`

- **数学形式**：把解集划分为 `F_0 = {i : ∄ j, F_j ≺ F_i}`、`F_{k+1} = {i : F_i ∉ ∪_{l≤k} F_l 且支配 i 的解全部落在 ∪_{l≤k} F_l}`，返回 `fronts`（每层解的下标列表）与 0 基 `rank`。
- **步骤**：① 对所有 i<j 调用支配判定，累计 `n_dom[j]`（支配 j 的解个数）与 `dominates[i]`（i 支配的解集合）；② 所有 `n_dom == 0` 的解构成功第 0 层；③ 逐层把当前层每个解所支配对象的 `n_dom` 减 1，减到 0 的进入下一层，直到没有剩余解（每条支配关系只处理一次）。
- **复杂度**：时间 O(n²·m) / 空间 O(n²)，n 为解个数、m 为目标个数（最坏情况下的支配关系表）；n 上千就会成为瓶颈。
- **参数**：只有输入 `F`（形状 `(n_solutions, n_objectives)` 的目标矩阵，全部按最小化理解）；**没有可调参数**，规模上限完全由 O(n²) 决定。NSGA-II 每一代都要对 2N 个解排序，所以这里的 n 实际是 2×种群规模。
- **陷阱**：`rank` 是 **0 基**的，有的教材记成 1 基的"秩"，对照文献表格时差 1；**完全相同的两个解互不支配**，会同时留在第 0 层，若目标矩阵里有大量重复行（离散问题反复采样到同一决策向量），第一层会异常臃肿且后续拥挤距离退化为 0，应当先去重；O(n²) 实现只适合 n 在几千以内，再大要换成熟库。
- **怎么检验**：手工小例写死期望（`[[1,3],[3,1],[2,2],[4,4],[3,4]]` 应分层为 `[[0,1,2],[4],[3]]`、`rank = [0,0,0,2,1]`）；写一个暴力对拍实现（对每个解直接数出支配它的解个数，`n_dom==0` 即第一层，剔除后重算第二层，如此循环）比对两层以上；断言所有解恰好出现一次、每层内部两两非支配、相邻层之间至少存在一条支配关系。

#### `crowding_distance(F)`

- **数学形式**：同一条前沿内，`dist_i = Σ_k (f^k_{i+1} − f^k_{i−1}) / (f^k_max − f^k_min)`（第 k 个目标上按取值排序后的相邻解之差，用极差归一化）；每个目标的两个边界点贡献 `+inf`，取值域为 0 的目标跳过、贡献 0。
- **步骤**：① 对每个目标 `argsort(kind="stable")`，两端记 `inf`；② 内部点累加 `(vals[2:] - vals[:-2]) / span`；③ 若某目标的 `span <= 0` 只跳过该目标（边界点仍为 `inf`，不产生 0/0）；④ 返回形状 `(n,)` 的 float 数组。
- **复杂度**：时间 O(m·n log n) / 空间 O(n)，n 为该前沿上的解个数（排序是主要成本）。
- **参数**：只有输入 `F`（形状 `(n, m)`，**必须是同一条前沿上的解**，全部最小化）；没有可调参数；`n <= 2` 时全部为 `inf`，`n > 2` 时边界点一定为 `inf`。
- **陷阱**：**必须按前沿分别调用**——把整份种群的目标矩阵直接丢进来会在不同前沿之间比较距离，得到的数没有任何意义（常见误用）；取值域为 0 的目标必须跳过，否则 0/0 会污染整条前沿；前沿中的重复点会互相"顶掉"对方（表现为一大一小），不代表真实多样性，去重后计算更可靠；边界点的 `inf` 在锦标赛里等价于"必被优先选中"，这是 NSGA-II 有意为之。
- **怎么检验**：手工前沿 `[[0,2],[1,1],[2,0]]` 的中间点应恰为 2（每个目标各贡献 1）、两端为 `inf`；构造某目标完全退化的前沿（所有点该目标取值相同），断言输出全为有限值且无 NaN；对同一条前沿做正仿射变换（`a*f + b`，`a > 0`）后断言距离逐元素不变（归一化口径的必然结果）；`n = 1` 与 `n = 2` 时断言全为 `inf`。

#### `weighted_sum_pareto(objective_fn, x_bounds, n_weights=11, seed=None, minimize=True)`

- **数学形式**：对单纯形等距网格上的每个权重 `w`（`w ≥ 0`、`Σ_k w_k = 1`）求 `min_x Σ_k w_k (sign·f_k(x))`，`sign = +1`（minimize=True）或 `−1`（minimize=False）；网格为 `{w_i = k_i/(n_weights−1)}`，共 `C(n_weights−1+m−1, m−1)` 个权重。
- **步骤**：① 在盒中心求一次 `objective_fn` 得到目标个数 m，用 `_simplex_grid` 生成权重网格；② 对每个 `w` 用 `_pattern_search`（多起点自适应随机方向搜索，1/5 成功法则简化版）极小化标量；③ 回代保存 `X` 与 `objective_fn` 的**原始输出** `F`（`minimize=False` 时不取负）。
- **复杂度**：时间 O(n_grid · n_starts · n_iter · (m + T_objective))，本实现默认 `n_starts=4`、`n_iter=120`，每次迭代一次目标求值；空间 O(n_grid · (n_dim + m))。n_grid 对 m=2 恰为 `n_weights`，m=3 时 11 个格点给出 66 个权重，m=4 给出 286 个。
- **参数**：`n_weights` 默认 11、必须 ≥2——它是每个权重分量上的格点数，调大让前沿更密但代价按组合数增长（m≥3 时尤其陡），调小会让前沿点变稀甚至整段漏掉；`seed` 默认 None（→ `DEFAULT_SEED = 20240101`），换种子只改变搜索轨迹、不改变权重网格；`minimize` 默认 True，False 表示全部取最大（内部统一取负）；`x_bounds` 无默认值，`(lo, hi)` 两个标量 = 1 维，要 d 维共用同一区间写 `[(lo, hi)] * d`。
- **陷阱**：加权和法**只能取到凸前沿**——前沿非凸时无论权重怎么取，中间那段非凸前沿上的解永远取不到，这是原理性缺陷而不是搜索不够努力（诊断方法：点在图上有"空洞"、或点的法线方向与权重明显不匹配）；目标量纲相差几个数量级时等权网格几乎只优化量级大的那个目标，**必须先把目标归一化**，本函数不做归一化（归一化口径会改变前沿形状，应由使用者在论文里显式交代）；权重分量为 0 的解在某目标上可以极差，但它仍是该权重下的正确最优，读图时不要当异常值删掉；底层是**无导数多起点随机方向搜索，不保证全局最优**，非凸多峰标量化问题会卡局部极小，论文里必须报告 `n_starts` 与总求值次数。另需注意 `_pattern_search` 实际使用 `n_starts + 1` 个起点（先随机取一个基准点再重启 `n_starts` 次），与内部命名"n_starts 次重启"略有出入。
- **怎么检验**：二次算例 `f1 = x²`、`f2 = (x−2)²` 有闭式最优 `x* = 2w₂/(w₁+w₂)` 与解析前沿 `f2 = (2 − √f1)²`，逐权重断言 `|X − x*| < 1e-6` 且前沿偏差 `< 1e-5`；用成熟库（如 scipy 的 SLSQP 或 pymoo）对**单个权重**的子问题独立求解对拍；断言权重矩阵每行和恰为 1 且非负、点数等于组合数公式；极限行为 `n_weights = 2` 时只应返回两个极端权重的解（分别只优化一个目标）。

#### `epsilon_constraint_pareto(objective_fn, x_bounds, n_grid=11, seed=None, minimize=True)`

- **数学形式**：对每个 ε 求 `min_x mean_{k≥2}(sign·f_k(x))` s.t. `sign·f_1(x) ≤ ε`；约束用线性罚处理，标量为 `body(x) + P·max(0, sign·f_1(x) − ε)`，罚系数 `P = 1e3 · max(采样极差, 1.0)`。
- **步骤**：① 在盒内随机采样 200 个点，估计第 1 个目标（最小化口径 `sign·f_1`）的最小/最大值，`np.linspace` 出 `n_grid` 个 ε；② 对每个 ε 用同一个 `_pattern_search` 极小化罚函数（m == 1 时 `body` 退化为 `f_1` 本身）；③ 回代检查可行性，写入 `infeasible`（容差 `1e-6 · max(1, |ε|)`）；④ 返回 `epsilons / X / F / infeasible`。
- **复杂度**：时间 O(n_probe · T_objective + n_grid · n_starts · n_iter · (m + T_objective))，`n_probe = 200` 写死在函数里；空间 O(n_grid · (n_dim + m))。
- **参数**：`n_grid` 默认 11、必须 ≥2——ε 网格点数，调大给更细的前沿但每个点都要一次**独立**的非凸优化；`seed` 默认 None（→ `DEFAULT_SEED`），它同时决定 200 个探测样本与搜索轨迹，换种子会改变 ε 网格本身；`minimize` 默认 True，False 时 ε 是"第 1 个目标最大化口径"的上限；罚系数 `P` **不可调**，自动取 `1e3 · max(采样极差, 1.0)`；`x_bounds` 口径同 `weighted_sum_pareto`。
- **陷阱**：罚函数法只给**近似可行**解——`P` 太小会停在不可行侧（看 `infeasible`），太大则数值上接近硬约束、搜索几乎无法从不可行区往里走；本实现的 `P` 由采样极差自动标定，若第 1 个目标量级很大（如 ~1e6）或带很大的常数偏移，这个标定会失效，换问题后务必检查 `infeasible`；ε 网格来自 200 个随机样本，采样完全错过极值时网格偏窄、前沿两端取不到（高维问题下 200 个样本更不够）；"最小化其余目标的等权平均"在其余目标量纲不同时仍偏向量级大的目标，必须先归一化；ε-约束法**可以**取到非凸前沿上的点（这正是它相对加权和法的价值），但每个 ε 是一次独立非凸优化，局部极小会让相邻 ε 的结果跳变。
- **怎么检验**：二次算例上断言 `infeasible` 全为 False 且每个解都落在解析前沿 `f2 = (2 − √f1)²` 上（自测容差 0.05）；构造解析可解的小例（单变量、线性目标、盒约束 + ε 约束），最优点必在约束边界（如 `x = +√ε` 或 `−√ε` 两支），与闭式解逐 ε 对拍；与 `weighted_sum_pareto` 在同一问题上比较——凸算例两者的点集都应落在解析前沿上（可互相覆盖），非凸算例中 ε-约束法应能取到加权和法取不到的中段点（这就是两方法的判别性实验）。

#### `nsga2(objective_fn, x_bounds, pop_size=40, n_gen=60, seed=None, minimize=True, mutation_sigma=None)`

- **数学形式**：每代在父+子共 2N 个解上做非支配排序与拥挤距离，按 `(rank 升序, crowding 降序)` 选取 N 个（精英保留）；繁殖为 SBX 交叉（分布指数 `η_c = 15`、概率 `p_c = 0.9`）+ 多项式变异（`η_m = 20`，每个分量以 `1/n_dim` 的概率变异）。
- **步骤**：① 盒约束内均匀随机初始化 `pop_size` 个个体并求目标值；② 每代：非支配排序 → 按前沿分别算拥挤距离 → 二元锦标赛（先比 `rank` 小、同前沿比拥挤距离大）选父 → SBX 交叉 → 多项式变异 → 等规模子代；③ 父代+子代合并（2N）后重新非支配排序，按前沿序号依次填充下一代，最后一个放不下的前沿按拥挤距离降序截断；④ 记录当代第一前沿规模。
- **复杂度**：时间 O(n_gen · (pop_size²·m + pop_size·m log pop_size + pop_size·T_objective))，空间 O(pop_size · (n_dim + m))；瓶颈是每代对 2N 个解的 O((2N)²m) 排序。
- **参数**：`pop_size` 默认 40（≥2），调大让前沿更密，但每代成本按平方涨，也是"给定预算"里最该做灵敏度分析的量；`n_gen` 默认 60（≥0，0 表示只评估初始种群、结果就是初始种群的第 0 层）；`seed` 默认 None（→ `DEFAULT_SEED`），结论必须连同 seed 一起报告；`minimize` 默认 True，False 时全部目标取最大（内部统一取负）；`mutation_sigma` 默认 None 表示走标准多项式变异，给定正数则改用标准差为 `mutation_sigma × 区间宽度` 的高斯变异——这是做消融实验的开关，取值小前沿更细但收敛慢、取值大探索强但前沿糊；`η_c = 15`、`η_m = 20`、`p_c = 0.9` 写死在函数体内，**不可调**。
- **陷阱**：`history` 记录的是**第一前沿规模而不是超体积**，它在本实现里实测几乎总是单调不减，但理论上并不保证（子代可能一次支配掉好几个父代前沿点使前沿收缩），所以只能当回归检查，不能当收敛性定理引用，也不能用它判断早熟；SBX 与多项式变异的边界用 **`np.clip` 简单截断**，不是原论文的边界修正公式，最优解恰在盒边界时边界附近解密度偏高（已知偏差）；**没有实现约束支配**（constrained-domination），带约束的问题必须自己把约束写进 `objective_fn`（罚函数），否则会稳定地输出不可行解；交叉/变异算子在低维（`n_dim <= 3`）上存在乘性耦合，变异强度其实不小，种群很难稳定收敛到一条细前沿；拥挤距离在 m ≥ 3 时多样性保持能力骤降，本函数只适合 m = 2 或 3（更多目标应按 NSGA-III / MOEA-D 改造）。
- **怎么检验**：与解析前沿对拍——二次算例断言最终第一前沿与 `f2 = (2 − √f1)²` 的最大偏差 < 0.05 且不含被支配的 `x0 < 0` 分支；ZDT1（3 维决策变量）断言与 `f2 = 1 − √f1` 偏差 < 0.05、`g = 1 + 9(x₁+x₂)/2` 被压到 1 附近；取返回的 `front` 下标，用**独立于本模块**的手写两两比较断言前沿内部无支配对；断言 `history` 长度恒为 `n_gen + 1`；极限行为 `n_gen = 0` 时结果必须等于"只对初始种群排序后的第 0 层"；换 5~10 个 seed 重复，报告前沿偏差与超体积的**分布**（均值+区间）而不是单次值，并用同一个固定参考点算超体积做跨代/跨参数比较。**注意长程迭代的混沌性**：150 代之后，"末位浮点差异 → 改变一次选择 → 换掉整条轨迹"会被放大到可见量级（实测同一个提交，Windows/numpy 2.1 得 ZDT1 偏差 0.005040，Linux/numpy 2.5 得 0.005923，而 `front` 规模与 `history` 逐代一致）。所以**别把长程连续指标当跨机器可比量**，本仓库的黄金值对它只记 `front` 规模与分档布尔量。

#### `pareto_front(F)`

- **数学形式**：`{i : ∄ j, F_j ≺ F_i}`，即第一前沿的原始下标集合及其目标子矩阵。
- **步骤**：① 调用 `fast_non_dominated_sort`（同一代码路径，保证口径一致）；② 取 `fronts[0]`，返回 `index`（原始下标、升序）与 `F[index]`。
- **复杂度**：时间 O(n²·m) / 空间 O(n²)（与底层排序完全相同，本函数不引入额外量级）。
- **参数**：只有输入 `F`（形状 `(n_solutions, n_objectives)`，全部最小化）；无可调参数。返回的 `index` 是**原始矩阵中的位置**，不是排序后的位置——若你先对 F 排序再调用，下标必须在原数组上重新映射，否则会取错解。
- **陷阱**：它是 `fast_non_dominated_sort` 第 0 层的薄封装，不是独立实现，不要把它当成"两种方法互相验证"；重复行会全部进入第一前沿（与排序函数同一行为）。
- **怎么检验**：断言 `pareto_front(F)["index"] == fast_non_dominated_sort(F)["fronts"][0]`（口径一致性）；对返回的子矩阵再跑一次 `fast_non_dominated_sort`，断言层数恰为 1（前沿的幂等性，独立于原实现的结论）；构造性检验——加入一个被支配点断言 `index` 不变，加入一个支配性更强的点断言它出现在 `index` 中。

#### `hypervolume_2d(F, reference)`

- **数学形式**：`HV = Λ(∪_{f∈F, f<reference} [f₁, ref₁] × [f₂, ref₂])`，即被解集支配、且被参考点界定的区域面积（Λ 为二维 Lebesgue 测度，全部最小化）。
- **步骤**：① 剔除 `f₁ ≥ ref₁` 或 `f₂ ≥ ref₂` 的点；② 用 `lexsort` 按 f₁ 升序（f₂ 破平）排序；③ 从左到右单遍扫描，维护已扫描点的最小 f₂ 值 `best`，第 i 段贡献 `(x_{i+1} − x_i) · max(0, ref₂ − best)`，最后一段右边界取 `ref₁`；④ 累加返回标量（空集返回 0.0）。
- **复杂度**：时间 O(n log n)（排序主导）/ 空间 O(n)（`lexsort` 的索引数组），n 为解个数。注意空间不是 O(1)。
- **参数**：`reference` 无默认值、长度必须为 2，且必须**严格劣于**前沿上每个点（每个分量都不小于前沿对应值，实践上取前沿各目标 max 再乘 1.1 之类的松弛）；取小了会漏算面积、取大了会让不同算法的比较失去区分度，而且**跨代/跨算法比较必须固定同一个 reference**，否则数值失去可比性、收敛曲线没有意义；`F` 必须是 `(n, 2)`。
- **陷阱**：只支持 m = 2，m ≥ 3 直接抛 `ValueError`（高维超体积需要另一套快速算法，不能把二维结果外推）；被 `reference` 支配的点贡献 0 而不是负数，所以 HV **不惩罚越界解**——算法输出跑到参考点之外时 HV 反而"看起来"没变差；参考点太靠近前沿会让所有解的超体积接近 0 而被舍入噪声淹没；重复点靠 `width <= 0` 跳过（不影响正确性）。
- **怎么检验**：手算三点前沿——`F = [[0,1],[1,0]]`、`reference = [2,2]` 的面积恰为 3.0（1×1 与 1×2 两块无重叠矩形）；加入被支配点 `[3,3]` 断言 HV 不变；加入更优的点 `[0,0]` 断言 HV 严格变大；与朴素实现（沿 f₁ 做细网格数值积分被支配区域的高度）对拍；单调性不变量——若解集 A 的每一点都被 B 中某点支配，则 `HV(A) ≤ HV(B)`；极限——把所有解平移到各自目标的最小值处，HV 应趋于 `(ref₁ − min f₁)(ref₂ − min f₂)`。

#### `ideal_point_distance(F, weights=None)`

- **数学形式**：`ideal_j = min_i F_ij`、`anti_j = max_i F_ij`；`d⁺_i = √(Σ_j w_j (F_ij − ideal_j)²)`、`d⁻_i = √(Σ_j w_j (F_ij − anti_j)²)`；`closeness_i = d⁻_i / (d⁺_i + d⁻_i)`（TOPSIS 口径，越大越好）。
- **步骤**：① `weights=None` 时取等权 `1/m`，否则校验长度、非负、和为正并归一化到和为 1；② 逐列取 `min`/`max` 得正/负理想点；③ 算两个加权欧氏距离与贴近度；④ `d⁺ + d⁻ == 0`（只有一个解或所有解目标值相同）时贴近度按约定取 0.5。
- **复杂度**：时间 O(n·m) / 空间 O(n)。
- **参数**：`weights` 默认 None（等权 `1/m`），长度必须等于目标个数 m、非负、和为正，内部会归一化——归一化不改变 `closeness`（两个距离同乘 `√c`），只改变 `d_plus`/`d_minus` 的量级；调权重就是改变"哪些目标更重要"的排序，权重集中在某个目标上时排序几乎只由该目标决定。
- **陷阱**：理想点由**当前解集自身**决定，加入或删除一个解就会改变所有解的贴近度，不同算法、不同种群规模的结果**不能直接比较**，必须先合并到公共解集上算同一个理想点；这里不做向量归一化（标准 TOPSIS 会先对指标矩阵做范数归一化，口径见 `_common.normalize_l2`），量纲差异大的目标会直接主导距离；`closeness` 是"相对排名"工具，不能解释成"离最优解还有多远"的绝对量。
- **怎么检验**：对称解集 `[[0,1],[1,0]]` 的贴近度闭式为 0.5，落在正理想点上的解贴近度为 1（自测）；单解或全同解时断言恰为 0.5；手算 2×2 例子的 `d_plus`/`d_minus` 对拍；变换不变量——先做范数归一化再算，断言排序与手算的一致（TOPSIS 的标准性质）。


### 3.16 灵敏度与缺失数据 —— `examples/algorithms/sensitivity.py`

**这族解决什么问题**：一半是**建模前处理**（缺失值插补、异常值检测），一半是**建模后诊断**（参数灵敏度）。两件事在数学建模论文里都极易写成"调库一句话"，但评委恰恰要看这里的口径：插补用的是哪一列的信息、异常判据是几倍 MAD、敏感性是在哪个尺度上算的。本模块给出可以逐行写进附录的"教学透明版"实现：OAT/弹性给局部斜率、Morris 给小样本筛选指标 μ/μ*/σ、Sobol 给方差分解的一阶与总效应指数、三种异常检测给可复现的下标集合；正式解题时建议与 scipy / SALib 对拍。它替代的是"直接调 sklearn 的 SimpleImputer/IsolationForest 然后不写口径"的写法。

**共同约定**：
- **缺失值**：一律用 `np.nan` 表示。插补类函数允许输入含 nan（这是它们唯一放宽校验的地方），但仍拒绝 inf；且 `_check_missing_layout` 拒绝任何**整行缺失**或**整列缺失**的输入（那种情形信息量为零，直接 `ValueError`）。
- **`fn` 接口**：统一为 `fn(x: np.ndarray) -> float`，`x` 是长度 d 的一维参数向量；`_eval_fn` 逐点调用并检查返回值有限（非有限值抛 `ValueError`），`_eval_fn_matrix` 逐行调用，**不假设** fn 支持向量化批量输入。
- **`bounds`**：形状 (d, 2)，第 i 行是第 i 个参数的下界与上界；必须二维、第 2 维长度为 2、全部有限、d ≥ 1，且下界**严格小于**上界，任一违反抛 `ValueError`（下界等于上界会让 Sobol 方差分母为 0、Morris 步长为 0）。
- **尺度**：灵敏度全部按**物理单位**报告——Morris 的步长与 Sobol 的抽样都换算回真实参数尺度，因此对线性函数 f = Σ c_i x_i，基本效应与弹性系数能直接对上解析值。
- **随机性**：Sobol 的 Saltelli 抽样与 Morris 轨迹一律走 `_common.rng(seed)`（独立 Generator），`seed=None` 回落到库默认种子 `DEFAULT_SEED`，不使用全局随机状态；插补与异常检测本身不含随机过程。
- **返回形态**：除 `elasticity` 返回裸 float 外其余全部返回 dict；三种异常检测统一给 `index`（升序 int 下标 list）与 `values`。刻意**不实现**多重插补/MICE、时间序列插补、二阶 Sobol 指数与参数分组，只依赖 numpy + 标准库 + `._common`。

#### `oat_sensitivity(fn, x0, rel_step=0.1)`

- **数学形式**：第 i 个参数的步长 step_i = rel_step·|x0_i|（x0_i = 0 时退化为绝对步长 step_i = rel_step）；low = x0_i − step_i、high = x0_i + step_i，其余参数固定在基准值；delta_low = f(low) − f(x0)、delta_high = f(high) − f(x0)；sensitivity = (f(high) − f(low))/(high − low)，即中心差分斜率（对线性函数恰为该参数的系数）
- **步骤**：① 校验 rel_step > 0，否则 `ValueError`；② `as_vector` 转 x0，求 base = f(x0)；③ 逐参数 i：按上式定 step，复制基准向量并把第 i 个分量分别改成 low/high，各求一次函数值；④ 算 delta_low、delta_high 与中心差分斜率；⑤ 用 max(|delta_low|, |delta_high|) 更新 `max_abs_change`；⑥ 返回 `base`/`table`/`max_abs_change`，`table` 每项的键固定为 `index`/`low`/`high`/`delta_low`/`delta_high`/`sensitivity`
- **复杂度**：O(d · C_fn) 时间（C_fn 为一次 fn 求值的代价；共 2d + 1 次求值）/ O(d) 空间
- **参数**：`fn`、`x0`（长度 d 的一维基准点）、`rel_step` 默认 0.1 且必须 > 0。参数为 0 时自动退化成绝对步长。**没有**"绝对步长"开关、**没有**可行域裁剪、**没有**阈值筛选（要按重要性排序请用返回的 `sensitivity` 自己排，或改用 `morris_screening`）；要看无量纲的比例弹性请用 `elasticity`
- **陷阱**：① OAT 在**非可加**模型上会误导：f = x0·x1 时两个参数的斜率都依赖另一个参数的当前取值，换个基准点结论就变了，此时应当用 Morris/Sobol 而不是 OAT；② 相对步长在参数穿越符号时（如 x0_i 很小）会让中心差分分辨率骤降，若参数可能取 0，请改用绝对步长版本或先做无量纲化；③ 这里不检查扰动点是否落在参数可行域内——OAT 是局部方法，可行性由调用方保证；④ 返回的是**带量纲的斜率**，不同参数之间不能直接比大小
- **怎么检验**：`_self_test` 两条断言可直接重跑——对 y = 3x₀ + 2x₁、基准点 (1.0, 1.0)、rel_step = 0.1：(a) `table` 里两个 `sensitivity` 与解析系数 [3, 2] 的差都必须 < 1e-9；(b) `max_abs_change` 必须等于 3·0.1 = 0.3（< 1e-9）。另外可：对 y = x₀² 手算中心差分 = 2x₀ + O(h²)，把 rel_step 减半看斜率向解析导数收敛；对双线性函数 f = x₀·x₁ 在两个不同基准点上各跑一次，观察斜率随基点的漂移——这正是陷阱 ① 的定量复现
- **补充检验（步长收敛）**：固定基准点、把 rel_step 从 0.1 依次降到 1e-6，绘制斜率误差随步长的曲线（先降后被浮点误差抬起），据此在论文里交代为什么选某个步长

#### `elasticity(fn, x0, i, rel_step=0.01)`

- **数学形式**：第 i 个参数的点弹性 E = (∂y/∂x_i)·(x_i/y)，用中心差分估计：h = rel_step·|x0_i|，g = (f(x + h·e_i) − f(x − h·e_i))/(2h)，E = g·x0_i/f(x0)；含义是 x_i 变化 1% 时 y 变化百分之几（无量纲）
- **步骤**：① 校验 rel_step > 0；② `as_vector` 转 x0，d = 长度；③ 校验 i 是整数（`int` 或 `np.integer`）且 0 ≤ i < d，否则 `ValueError`；④ y0 = f(x0)，若 |y0| ≤ `_EPS`（1e-12）抛 `ValueError`；⑤ 若 x0_i == 0 直接返回 0.0；⑥ 令 h = rel_step·|x0_i|，构造 hi、lo 两个扰动点，算中心差分斜率；⑦ 返回 float(slope·x0_i/y0)
- **复杂度**：O(C_fn) 时间（含基准点共 3 次求值）/ O(d) 空间
- **参数**：`i` 是参数下标，必须是整数且落在 [0, d − 1]；`rel_step` 默认 **0.01**（注意与 `oat_sensitivity` 的 0.1 不同）。返回值是**裸 float 而不是 dict**（本模块唯一的例外）。**没有**批量版本（要整张表就循环调 d 次或用 `oat_sensitivity`），也**没有** f(x0) = 0 的兜底
- **陷阱**：① x0_i == 0 时弹性恒为 0（因为乘了 x_i），但此时相对扰动无意义；本实现直接返回 0 而不是抛错，因为"零弹性"在数学上是对的——若你要的是斜率，请用 `oat_sensitivity`；② f(x0) == 0 时弹性无定义（分母为 0），这里抛 `ValueError`——静默返回 inf 会让排序类后续处理得出荒谬结论；③ 中心差分对二阶项精确、对三阶项有 O(h²) 误差；h 取太小会被浮点抵消淹没，rel_step = 0.01 是常用的折中
- **怎么检验**：`_self_test` 对 y = x₀²·x₁ 在 (2.0, 3.0) 处断言解析弹性为 [2, 1]（i = 0 与 i = 1 各算一次，误差 < 1e-9，手算：∂y/∂x₀ = 2x₀x₁、∂y/∂x₁ = x₀²，乘 x/y 后分别为 2 和 1）。另外可：Cobb-Douglas 形式 f = x₀^α·x₁^β 的弹性恒等于指数 α、β，换几组基准点都应得到同样的值（弹性的定义性质）；x0_i = 0 时断言返回恰为 0.0；令 f(x0) = 0 断言抛 `ValueError`；把 rel_step 从 0.01 调到 1e-4 观察数值稳定性
- **补充检验（与 OAT 对照）**：对同一函数同时算 `elasticity` 与 `oat_sensitivity` 的斜率，用 E ≈ slope·x0_i/f(x0) 手工互相换算，两处不一致说明步长或基准点被搞错了

#### `morris_screening(fn, bounds, n_trajectories=10, n_levels=4, seed=None)`

- **数学形式**：参数归一化 u_i = (x_i − lo_i)/(hi_i − lo_i)；网格步长 step = 1/(p − 1)，Morris 步长 delta = p/(2(p − 1))；第 i 个参数的**物理单位**基本效应 EE_i = [f(x + Δ_i e_i) − f(x)]/Δ_i，其中 Δ_i = delta·(hi_i − lo_i)；μ_i = mean_t EE_i^(t)（带符号、反映方向），μ*_i = mean_t |EE_i^(t)|（推荐排序指标），σ_i = std_t(EE_i^(t), ddof=1)（反映非线性/交互）
- **步骤**：① 校验 n_trajectories ≥ 1、n_levels ≥ 2，`_check_bounds` 校验边界；② 算 step、delta 与网格 grid = {0, step, ..., 1}，合法基点集合取 grid ≤ 1 − delta（集合为空抛 `ValueError`）；③ `gen = rng(seed)`；④ 每条轨迹：在合法基点集合上对 d 个参数**独立均匀**取归一化基点，再独立打乱参数顺序；⑤ 依次对每个参数沿 +delta（若超出 1 则改 −delta）移动一步，用物理步长 Δ_i 算 EE；⑥ 沿轨迹轴求 μ 与 μ*；σ 用 ddof = 1（`n_trajectories == 1` 时返回全 0）；⑦ `ranking` = `argsort(−mu_star)`；⑧ 返回 `mu`/`mu_star`/`sigma`/`ranking`
- **复杂度**：O(n_trajectories · (d + 1) · C_fn) 时间（每条轨迹 d + 1 次求值）/ O(n_trajectories · d) 空间
- **参数**：`n_trajectories` 默认 10（≥ 1，条数越多 σ 越可靠）；`n_levels` 默认 4（≥ 2，通常 4 或 8）；`seed` 默认 `None` → `DEFAULT_SEED`。返回的 μ/μ*/σ 是形状 (d,) 的 numpy 数组，`ranking` 是下标 list。**没有**置信区间、**没有**"参数分组（groups）"、**没有**非均匀分布支持（网格是均匀的）、**没有** p 的倍率之外的自由度；要做方差分解请用 `sobol_first_order`/`sobol_total_effect`
- **陷阱**：① `n_trajectories == 1` 时样本标准差无定义：本实现返回 0 而不是 NaN，但一条轨迹的 σ **没有**任何统计意义，不要据此判断非线性；② `delta` 与 p 绑定：p = 4 时 delta = 2/3，单步就跨越了三分之二的参数范围，此时的 μ 是**大范围平均斜率**而不是局部导数，与 OAT 的斜率不可直接比较；③ 基本效应在物理单位下计算，因此 `bounds` 的宽度会直接改变 μ 的量纲；比较不同参数的重要性时这恰恰是想要的（"参数变 1 个单位，输出变多少"）；④ μ 的符号信息会被 μ* 丢掉，判断方向必须两个一起看
- **怎么检验**：`_self_test` 对 y = 3x₀ + 2x₁、`bounds = [[0,1],[0,1]]`、`n_trajectories = 12`、`n_levels = 4`、`seed = 5` 断言：(a) μ 与 [3, 2] 的差 < 1e-9（线性函数的基本效应恒等于系数）；(b) σ 的最大绝对值 < 1e-9（线性 ⇒ 无交互，理论为 0，浮点误差量级约 1e-15，所以容差用 1e-9 而不是严格为 0）。另外可：对 y = x₀·x₁ 看 σ 是否明显大于 0（交互的标志）；固定 seed 断言两次调用返回完全相同的 μ/μ*/σ/ranking；把 n_trajectories 从 10 加到 100 看 μ* 的排序是否稳定；换 seed 重跑看排序是否稳定
- **补充检验（与解析值对拍）**：对可乘函数 f = x₀·x₁ 在 [0,1]² 上，基本效应随轨迹位置变化，μ 与 μ* 的解析值可直接积分算出来，与模拟值对拍可验证轨迹构造的正确性

#### `sobol_first_order(fn, bounds, n_samples=512, seed=None)`

- **数学形式**：A、B 是两组各 N 行的独立样本，AB_i 表示把 A 的第 i 列换成 B 的第 i 列；记 V = Var(y) 用 A ∪ B 合并样本的**无偏**估计（ddof = 1）；S1_i = mean(fB·(fAB_i − fA))/V（Saltelli 2010 式 (10)），S1_conf_i = 1.96·std(fB·(fAB_i − fA), ddof = 1)/√N/V
- **步骤**：① 走 `_sobol_common`：校验 n_samples ≥ 2、`_check_bounds` 校验边界；② `_sobol_design` 一次抽 `2N·d` 个 U(0,1) 随机数，前一半作 A、后一半作 B，再用 lo + u·(hi − lo) 映射到物理参数区间；③ 逐行求 fA = fn(A)、fB = fn(B)；④ 对每个 i 复制 A 并把第 i 列换成 B 的对应列，求 fAB_i；⑤ 合并样本算 V，若 V ≤ 0 抛 `ValueError`（fn 近似常数，S1 分母为 0）；⑥ 逐维算 term = fB·(fAB_i − fA)，S1 = mean(term)/V、S1_conf = 1.96·std(term, ddof = 1)/√N/V；⑦ 返回 `S1`/`S1_conf`/`n_eval`，`n_eval = N·(d + 2)`
- **复杂度**：O(N · d · C_fn) 时间（共 N·(d + 2) 次函数求值）/ O(Nd) 空间
- **参数**：`n_samples` 默认 512，是**基样本量 N**（不是总求值次数），必须 ≥ 2；`seed` 默认 `None` → `DEFAULT_SEED`。**没有**分布参数（各参数独立均匀，分布由 `bounds` 表达）、**没有**二阶指数、**没有**参数分组。想要总效应用 `sobol_total_effect`——同一 seed 下两者共用同一套 A/B/AB_i 设计，抽样完全一致，这是刻意的
- **陷阱**：① N 太小时 S1 可能为负（估计量有偏噪声），小负值**不代表**参数有害，应报告为"≈0"；本实现不截断，保留原始估计以示诚实；② 对强交互模型（如 y = x₀·x₁），一阶指数之和远小于 1，这不是 bug，差额就是交互贡献——必须同时看 `sobol_total_effect`；③ 置信区间是正态近似，且忽略了 A/B 共用同一批随机数带来的相关性，只用于量级判断，不要写进论文当严格区间；④ A 与 B 必须独立，若图省事把 B 写成 A 的行置换，估计量会有偏且偏差方向恰好是"低估交互"，非常难在结果里看出来
- **怎么检验**：`_self_test` 用 y = 2x₀ + x₁、x ~ U(0,1) 独立，解析方差 Var = 4/12 + 1/12 = 5/12、S1 解析值为 [0.8, 0.2]；取 `n_samples = 32768`、`seed = 11`，断言两个 S1 与解析值的差都 ≤ 0.05。自测注释里记录了为什么取这么大的 N：Saltelli 的乘积估计量方差偏大，n_samples = 512 时抽样误差本身就有 0.14 量级，4096 时仍达 0.048（贴死容差），32768 下本 seed 实际误差 0.012（约容差的 1/4）。交互例 y = x₀·x₁ 取 `n_samples = 2048`、`seed = 13`，断言 ST > S1（自测注释给的解析值约 0.571 vs 0.429）。另外可：把 n_samples 翻倍看 S1 是否向解析值收敛；换几个 seed 重跑，看波动是否落在 S1_conf 的量级内；用 `n_eval` 核实求值次数恰为 N·(d + 2)
- **补充检验（独立实现对拍）**：用同一 seed 自己生成 A/B（`rng(seed).random((N, 2d))` 的同一口径），手工实现 Saltelli 式 (10)，应当与 `S1` 逐位一致——这能验证"抽样顺序"没有被搞错

#### `sobol_total_effect(fn, bounds, n_samples=512, seed=None)`

- **数学形式**：ST_i = mean((fA − fAB_i)²)/(2V)（Jansen 1999 / Saltelli 2010 式 (12)），ST_conf_i = 1.96·std((fA − fAB_i)², ddof = 1)/√N/(2V)；ST_i 含该参数的全部交互贡献
- **步骤**：① 走 `_sobol_common`（与 `sobol_first_order` 完全相同的抽样、求值与方差口径，同 seed 下抽样一致）；② 逐维算 sq = (fA − fAB_i)²；③ ST = mean(sq)/(2V)，conf = 1.96·std(sq, ddof = 1)/√N/(2V)；④ 返回 `ST`/`ST_conf`
- **复杂度**：O(N · d · C_fn) 时间 / O(Nd) 空间
- **参数**：与 `sobol_first_order` 完全相同（`n_samples` 默认 512、`seed` 默认 `None` → `DEFAULT_SEED`、`bounds` 形状 (d, 2)）。**返回键只有 `ST` 与 `ST_conf`，没有 `n_eval`**——需要求值次数请调 `sobol_first_order` 或按 N·(d + 2) 自己算；也**没有**截断开关（ST 可能为小负值）
- **陷阱**：① 可加模型上应当有 ST_i ≈ S1_i；若差得远，说明 N 太小或 fn 里藏了交互；② Σ ST_i ≥ 1（等于 1 当且仅当模型可加），和明显大于 1 是估计噪声，不是"发现了新交互"；③ 与一阶指数一样，这里不添加任何截断，ST_i 也可能出现小的负值；④ 两个函数各自独立抽样：若两次调用传了不同的 seed，"S1 与 ST 的差"里就混进了抽样误差
- **怎么检验**：`_self_test` 在同一可加函数 y = 2x₀ + x₁ 上取 `n_samples = 32768`、`seed = 11`，断言 max|ST − S1| ≤ 0.05；在交互函数 y = x₀·x₁ 上取 `n_samples = 2048`、`seed = 13`，断言两个参数都满足 ST > S1（解析值约 0.571 > 0.429）。另外可：用同一 seed 分别调一阶与总效应，确认两者抽样一致（同 seed 下 A/B 完全相同），此时 ST 与 S1 的差只来自估计量本身；对纯可加函数检查 Σ ST 是否接近 1；对纯交互函数检查 Σ S1 是否明显小于 1 而 Σ ST 接近 1

#### `impute_mean(X)`

- **数学形式**：对每一列取观测均值 x̄_j = mean{x_ij : x_ij 非缺失}，把该列所有缺失位置填成 x̄_j；`n_imputed` 为被填元素的总数
- **步骤**：① `_as_float_matrix` 转成 float64 二维数组（允许 nan、拒绝 inf）；② `_check_missing_layout` 返回缺失掩码并拒绝整行/整列全缺失；③ 若缺失数为 0，直接返回输入副本、`n_imputed = 0`、`method = "mean"`；④ 否则 `np.nanmean(arr, axis=0)` 求列均值，按 `np.nonzero(mask)` 的下标回填；⑤ 返回 `X`/`n_imputed`/`method`
- **复杂度**：O(mn) 时间 / O(mn) 空间（返回的是副本，**不修改**原输入）
- **参数**：只有 `X` 一个参数。**没有** k、**没有** max_iter/tol、**没有**"按行/按列"开关（固定按列）、**没有**插补不确定性输出。返回键固定为 `X`/`n_imputed`/`method`，`method` 恒为 `"mean"`。要更强的插补请用 `impute_knn` 或 `impute_regression`；含 nan 是允许的，含 inf 抛 `ValueError`，整行/整列缺失抛 `ValueError`
- **陷阱**：① 均值插补会**压缩方差**（填进去的都是列中心），后续做回归/聚类时会让变量显得比实际更"整齐"，这是它作为基线而非推荐方法的根本原因；② 完全随机的缺失（MCAR）下均值插补无偏；一旦缺失与取值相关（MNAR），均值插补会系统性偏移，此时必须用回归/多重插补并在论文里讨论；③ 整列全缺失时列均值是 nan，本实现直接报错（见 `_check_missing_layout`），不会静默写出 nan 污染后续计算
- **怎么检验**：`_self_test` 在 60×3 的列间近似线性矩阵（`rng(20240115)` 生成，x₁ = 2x₀ + 0.10·N(0,1)、x₂ = −1.5x₀ + 0.05·N(0,1)）上随机挖掉 10% 的格子（180 个元素中 18 个），记录三种插补方法的 RMSE，并断言 `impute_regression` 与 `impute_knn`（k = 5）的 RMSE 都**严格小于**均值插补——也就是说均值插补在这里扮演的是"及格线"。另外可：传无缺失的矩阵，断言输出与输入逐元素相等且 `n_imputed == 0`；构造单列单缺失的小例手算列均值对拍；构造一列全 nan 断言抛 `ValueError`；检查原输入矩阵在调用后未被改动
- **补充检验（方差压缩）**：对同一列比较插补前后的样本方差，插补后方差必然不增——这是陷阱 ① 的直接量化

#### `impute_knn(X, k=5)`

- **数学形式**：对每个缺失位置 (i, j)：候选集为第 j 列**已观测**的所有行 c（c ≠ i）；只在 i 与 c **同时观测**的维度上算欧氏距离 d(i, c) = √(Σ_{m ∈ 共同观测维} (x_im − x_cm)²)；取距离最小的 k 个候选，权重 w = 1/(d + 1e-12) 做加权平均；若最小距离 ≤ 1e-12（存在完全相同的邻居），只用这些零距离邻居的均值；若不存在任何"有共同观测维度"的候选，退回该列均值
- **步骤**：① 校验 k ≥ 1，否则 `ValueError`；② `_as_float_matrix` + `_check_missing_layout`；③ 无缺失直接返回副本、`n_imputed = 0`、`method = "knn"`；④ 求列均值备用；⑤ 用 `np.nonzero(mask)` 遍历每个缺失位置：取该列已观测行、构造 `shared` 共同观测掩码、把非共同维度的差置 0 后算欧氏距离；⑥ 过滤出"至少有一个共同观测维度"的候选，`np.argsort(dist, kind="stable")` 取前 k；⑦ 若最近距离 ≤ `_EPS` 走零距离分支（取这些邻居的均值），否则走 1/(d + ε) 加权平均；⑧ 返回 `X`/`n_imputed`/`method`（`"knn"`）
- **复杂度**：O(n_missing · n · d) 时间（逐元素处理，n_missing 为缺失格子数）/ O(n · d) 空间
- **参数**：`k` 默认 5、必须 ≥ 1；k 大于可用候选数时**按实际候选数取，不报错**。**没有** distance 参数（固定欧氏）、**没有** 权重开关（固定 1/(d + ε)）、**没有**自动标准化、**没有**并行/近似最近邻。若各列量纲差异大，应在调用前自己标准化，否则距离被大量纲列主导；极稀疏数据上应改用 `impute_regression`
- **陷阱**：① 只在共同观测维度上比距离是**稀疏数据的关键**：若改用全维度并把 nan 当 0，距离会被人为放大，邻居选择完全错乱，而结果看起来仍然"像那么回事"；② k 大于可用候选数时按实际候选数取（不报错），但在极稀疏数据上会退化成加权均值，插补质量与均值法无异，此时应改用 `impute_regression`；③ 本实现逐元素处理，n 很大（> 1e4）时明显偏慢；竞赛数据规模下可以接受；④ 零距离分支用的是"完全相同的邻居"本身，不是均值，两者混用时序上要看清（实现里是分支互斥的）
- **怎么检验**：`_self_test` 在与 `impute_mean` 相同的 60×3、10% 缺失矩阵上取 k = 5，断言 RMSE 严格小于均值插补的 RMSE。另外可：(a) 构造两行完全相同、其中一列缺失的小例，验证走零距离分支返回相同行的观测值；(b) 构造只有一个候选且与目标没有共同观测维度的小例，验证退回列均值；(c) 把 k 调到大于候选数量，确认不报错且结果等于按全部候选用 1/(d + ε) 加权；(d) 手工对 3~4 行的小例逐个缺失位置算距离与权重，逐格对拍
- **补充检验（稀疏性诊断）**：统计每个缺失位置实际用到的候选数与共同观测维度数，若大量位置退化成列均值，说明 k 的设定或数据稀疏度不适合 KNN

#### `impute_regression(X, max_iter=10, tol=1e-6)`

- **数学形式**：初始化把所有 nan 填成所在列的观测均值；每一轮对每个含缺失的列 j，取该列**已观测**的行构造带截距的设计矩阵 [1, 其余列]，用 `np.linalg.lstsq` 解 OLS，再对缺失行预测并回填；收敛判据是"本轮与上轮**所有被插补值**的最大绝对变化 ≤ tol"
- **步骤**：① 校验 max_iter ≥ 1、tol > 0，否则 `ValueError`；② `_as_float_matrix` + `_check_missing_layout`；③ 若无缺失，直接返回副本 + `n_imputed = 0` + `n_iter = 0` + `converged = True`（注意这条路径在列数检查**之前**）；④ 若有缺失且列数 < 2，抛 `ValueError`；⑤ 列均值初始化，记录 `prev = filled[mask]`；⑥ 列出所有含缺失的列；⑦ 每轮对每个这样的列解 OLS 并回填其缺失行，然后比较 `cur = filled[mask]` 与 `prev` 的最大绝对差，≤ tol 则置 `converged = True` 并 break；⑧ 返回 `X`/`n_imputed`/`n_iter`/`converged`
- **复杂度**：O(max_iter · m · n²) 时间（每轮每列一次最小二乘）/ O(mn) 空间
- **参数**：`max_iter` 默认 10（≥ 1）；`tol` 默认 1e-6（> 0），判据作用在**被插补值的变化**上而不是回归系数上；`X` 在有缺失时必须至少 2 列（列数为 1 且确实有缺失时抛 `ValueError`）。**没有**随机初值、**没有**多重插补的 m 参数、**没有**"每列用哪些预测变量"的开关（固定用其余全部列）、也**不返回**每轮的收敛轨迹
- **陷阱**：① 这是**确定性迭代**而非多重插补：它给出单点估计，无法反映插补本身的不确定性，正式论文应报告敏感性（例如换 3 组随机初值或改用 MICE）；② 迭代线性回归会把变量间关系"越描越真"：若两列高度线性相关，缺失会被填得过于完美，导致后续回归的 R² 虚高，务必在论文中说明；③ 未收敛（`converged` 为 `False`）时必须报告实际 `n_iter`，不能假装收敛；④ 初始化用均值、且迭代中"其余列已无缺失"，所以结果依赖列的处理顺序与被填值的当前状态——换一个列顺序理论上可能给出略微不同的不动点
- **怎么检验**：`_self_test` 在同一 60×3 矩阵上取 `max_iter = 60`、`tol = 1e-6`，断言四条：RMSE 严格小于均值插补；`n_imputed` 恰等于挖掉的格子数（`flat_idx.size`，即 18）；`converged` 为 `True`；输出矩阵不含 nan。另外可：(a) 构造两列完全线性（无噪声）的小例，验证插补值精确等于真值；(b) 把 `max_iter` 设为 1，观察 `converged` 为 `False` 且 `n_iter == 1`；(c) 同一输入两次调用断言逐位一致（确定性）；(d) 与"用已知真值做单次 OLS 预测"的独立实现对拍
- **补充检验（收敛性）**：把 `max_iter` 依次设为 1, 2, 5, 10, 60，记录 `n_iter` 与 RMSE 随轮数的变化，确认确实收敛而不是在容差边缘抖动

#### `detect_outliers_zscore(x, threshold=3.0)`

- **数学形式**：z = (x − mean)/std，标准差用**总体**口径（ddof = 0）；|z| > threshold 判为异常。理论上单个点在 n 个样本里能取得的 |z| 上限约为 (n − 1)/√n
- **步骤**：① 校验 threshold > 0，否则 `ValueError`；② `as_vector` 转 x（不允许含 nan）；③ 算 mean 与 std(ddof = 0)，std ≤ 0（常数序列）抛 `ValueError`；④ 算 z = |(x − mean)/std|；⑤ `np.flatnonzero(z > threshold)` 取下标（天然升序）；⑥ 返回 `index`/`values`/`threshold`（阈值原样回传，便于报告）
- **复杂度**：O(n) 时间（两次向量化归约）/ O(n) 空间
- **参数**：`threshold` 默认 3.0、必须 > 0，含义是**总体标准差**的倍数。**没有** ddof 开关（固定总体口径，这也是与"样本标准差"写法容易差一个系数的地方）、**没有**回传全部 z 值的键（要全量分数请看 `detect_outliers_mad` 的 `scores` 或自己算）、**没有**稳健版本（那正是 `detect_outliers_mad` 的用途）
- **陷阱**：① 均值和标准差本身被离群点污染：单个极端值会把 std 抬高，使自己的 |z| 被压缩；对 n 较小时，单点能取得的 |z| 上限约为 (n − 1)/√n——n = 10 时只有 2.85，**永远达不到 3**，于是"10σ 离群点"在 z 分数法下反而漏检，样本量小时请改用 MAD 或 IQR；② 数据近似正态时才用 3 作为阈值；重尾分布（如收入、点击量）会大面积误报；③ 标准差为 0（常数序列）时 z 无定义，本实现抛 `ValueError`；④ 判定用严格大于 `>`，恰好等于阈值不算异常
- **怎么检验**：`_self_test` 用 31 个点——`np.linspace(10.0, 40.0, 30)` 再接一个 500.0 的远端离群点，`threshold = 3.0`，断言 `index` 恰为 `[30]`（最后一个下标），并与 IQR（k = 1.5）、MAD（threshold = 3.5）两种方法的结果一致（自测另有 `sens_outlier_agree` 记录三者是否相同）。另外可：(a) 手算小例的 mean/std 与边界 |z| 值，验证 `>` 的严格性；(b) 复现陷阱 ①——构造 n = 10 的序列并把单点推到极远，观察 |z| 仍小于 3；(c) 常数序列断言抛 `ValueError`；(d) threshold ≤ 0 断言抛 `ValueError`
- **补充检验（口径核对）**：用 `x.std()`（ddof = 0）与 `x.std(ddof=1)` 各算一遍 z，看判定结果差多少——论文里必须写明用的是哪一种

#### `detect_outliers_iqr(x, k=1.5)`

- **数学形式**：Q1、Q3 用 numpy 的线性插值口径（`np.percentile` 默认）；IQR = Q3 − Q1；下栅栏 lower = Q1 − k·IQR、上栅栏 upper = Q3 + k·IQR；x < lower 或 x > upper 判为异常
- **步骤**：① 校验 k > 0，否则 `ValueError`；② `as_vector` 转 x（不允许含 nan）；③ `np.percentile(arr, 25.0)` 与 `(arr, 75.0)` 求 Q1、Q3；④ 算 IQR 与两个栅栏；⑤ `np.flatnonzero((arr < lower) | (arr > upper))` 取下标（升序）；⑥ 返回 `index`/`values`/`lower`/`upper`
- **复杂度**：O(n log n) 时间（排序取分位数）/ O(n) 空间
- **参数**：`k` 默认 1.5、必须 > 0；`k = 1.5` 对应 Tukey 的"离群"、`k = 3.0` 对应"极端离群"。**没有**分位数口径开关（固定线性插值）、**没有** return_scores、**没有** IQR 为 0 时的兜底（实现照常算栅栏，退化为 [Q1, Q3]）、**没有**偏态修正
- **陷阱**：① IQR 对**偏态**分布仍会误报：右偏数据的上栅栏往往被压得过低，长尾的正值被成批标为异常，此时应先做变换或直接看业务含义；② IQR 为 0 时（超过一半样本取同一值，例如大量 0 的稀疏数据）栅栏退化为 [Q1, Q3]，任何轻微波动都会被判异常——这是最常见的 IQR 误用；③ 分位数口径（线性插值 vs 取序）会改变边界点判定，本模块统一用线性插值，与别的软件（如某些默认取序的实现）对不上时先查这一条；④ 判定用严格不等号，恰好落在栅栏上的点**不算**异常
- **怎么检验**：`_self_test` 用与 zscore 相同的 31 点序列、`k = 1.5`，断言 `index` 恰为 `[30]`。另外可：(a) 小样本（如 8 个点）直接手算 Q1/Q3 的线性插值值与两个栅栏，逐项对拍 `lower`/`upper`；(b) 移动某个点使其恰好等于栅栏，验证严格不等号；(c) 对 `[0,0,0,...,1]` 这类 IQR = 0 的数据观察误报（陷阱 ② 的复现）；(d) 与 `numpy.percentile` 的插值口径逐值核对，确认没有换成其它分位数定义
- **补充检验（k 的敏感性）**：把 k 从 1.5 调到 3.0，记录异常点集合的收缩情况，作为论文里阈值选择的依据

#### `detect_outliers_mad(x, threshold=3.5)`

- **数学形式**：med = median(x)，MAD = median(|x − med|)；修正 Z 分数 M_i = 0.6745·(x_i − med)/MAD（0.6745 是标准正态的 0.75 分位数，使 M_i 在正态数据下与标准 Z 分数同尺度）；|M_i| > threshold 判为异常
- **步骤**：① 校验 threshold > 0，否则 `ValueError`；② `as_vector` 转 x（不允许含 nan）；③ 算 med 与 mad = median(|x − med|)，mad ≤ 0 抛 `ValueError`（中位数附近样本过多）；④ 算 scores = 0.6745·(arr − med)/mad；⑤ `np.flatnonzero(np.abs(scores) > threshold)` 取下标（升序）；⑥ 返回 `index`/`values`/`scores`（`scores` 与 x 等长，是三种异常检测里唯一回传全量分数的）
- **复杂度**：O(n log n) 时间（两次中位数选择，numpy 内部为 O(n) 选择算法，量级上按排序报告）/ O(n) 空间
- **参数**：`threshold` 默认 3.5（Iglewicz & Hoaglin 的建议值）、必须 > 0。**没有** scale 常数参数（0.6745 硬编码，换分布就要自己改口径）、**没有**栅栏系数的 k、**没有**"用 Qn/Sn 等更稳健尺度"的开关；也不允许含 nan
- **陷阱**：① MAD 用中位数而非均值，抗污染能力强，但**对 n 很敏感**：n 小于约 10 时 MAD 可能为 0（超过一半样本等于中位数），此时修正 Z 分数无定义，本实现抛 `ValueError`；② 0.6745 这个常数只对正态分布成立；对其他分布，3.5 的阈值没有概率解释，只能当作经验规则；③ 当缺失/异常点占比接近 50% 时，中位数本身已被污染，MAD 完全失效；④ 判定同样用严格大于，边界点不算异常
- **怎么检验**：`_self_test` 用与 zscore/IQR 相同的 31 点序列、`threshold = 3.5`，断言 `index` 恰为 `[30]`。另外可：(a) 直接对返回的 `scores` 做断言——中位数附近样本的分数应接近 0，远端点的 |M| 应远大于 3.5；(b) 构造 MAD = 0 的数据（例如 `[1,1,1,1,1,10]`，中位数为 1 且超过一半样本等于它），断言抛 `ValueError`；(c) 与 0.6745·(x − med)/MAD 的手算值逐点对拍；(d) 在同一个含离群点的样本上分别跑三种方法，比较各自抓到的下标集合——本自测的构造下三者一致
- **补充检验（抗污染对比）**：把最大的离群点取得更极端，看 z 分数法的结果如何变化、MAD 的结果是否稳定（理论上 `scores` 会随极值增大而增大但判定不变），这就是"稳健"二字的可复现实验


### 3.17 空间分析与选址 —— `examples/algorithms/spatial.py`

注意口径：本模块的 `spatial.py` 在源码 docstring 里自述为"**空间与物理场建模**"（一维热传导、二维泊松、森林火灾元胞自动机、NaSch 交通流、Buckingham π 量纲分析、比例缩放换算），**不含选址/设施布局模型**；涉及"最近设施、Voronoi 归属"的那类空间分析在 `examples/algorithms/geometry.py`（见 3.15 的 `voronoi_nearest`、`idw_interpolate`、`ordinary_kriging`）。以下条目按本模块的实际内容撰写，章标题沿用分块大纲给定的名称，合并进 `references/algorithm-details.md` 前应改名为"空间与物理场建模"。模块级约定：

- 热传导一律写作 `u_t = alpha·u_xx`；网格**包含两个端点**，初值 `u0` 的长度就是网格点数 n；边界只支持 `"dirichlet"`（端点值固定为 `u0` 的端点值）与 `"neumann"`（零通量，用镜像虚拟节点实现，因此离散总热量精确守恒）。
- 显式格式是 FTCS（时间一阶、空间二阶），隐式格式是 Crank-Nicolson（时间二阶），两者共用同一套零通量边界口径，同一组 `alpha, dx, dt` 下可直接对比。
- 泊松方程为单位方形域 `-Δu = f` + 零 Dirichlet 边界，n 个**内点**、步长 `h = 1/(n+1)`，`f` 与返回的 `u` 都只覆盖内点（形状 (n, n)），边界恒为 0、不出现在返回值里。
- 元胞自动机与交通流都是**同步更新**，网格外一律视为"非燃烧" / 周期边界（交通流）。
- 随机性一律走 `_common.rng(seed)`，不使用 `np.random.*` 全局函数；`seed=None` 表示 `DEFAULT_SEED`。

#### `heat_equation_1d_explicit(u0, alpha, dx, dt, n_steps, bc=("dirichlet", "dirichlet"))`

- **数学形式**：FTCS 显式格式（时间前向、空间中心差分）。内部点 `u_i^{k+1} = u_i^k + r·(u_{i−1}^k − 2u_i^k + u_{i+1}^k)`，`r = alpha·dt/dx²`；Dirichlet 边界把端点直接钉在 `u0[0]` / `u0[-1]`；零通量边界用镜像虚拟节点 `u_{−1} = u_0`、`u_n = u_{n−1}`，于是端点更新为 `u_0^{k+1} = u_0^k + r·(u_1^k − u_0^k)`（右端同理）。
- **步骤**：① 公共校验 `_check_heat_inputs`：`u0` 长度 ≥ 3、`alpha > 0`、`dx > 0`、`dt > 0`、`n_steps` 为非负整数（`bool` 被拒）、`bc` 长度为 2 且两项都属于 `("dirichlet", "neumann")`；② 复制一份 `u_init` 供 Dirichlet 钉边界；③ 算 `ratio = alpha·dt/dx²`，**超过 0.5 直接抛 `ValueError`**（绝不静默发散）；④ 每步先复制当前场、整块更新内部点，再按左右边界类型分别更新两端；⑤ 返回 `{"u": 终态场, "stability_ratio": ratio}`。
- **复杂度**：时间 O(n_steps · n)（每步两次向量化运算）/ 空间 O(n)。
- **参数**：`u0`（初始温度场，形状 (n,)，**含两个端点**，n ≥ 3）；`alpha`（热扩散系数，> 0）；`dx`（空间步长，> 0）；`dt`（时间步长，> 0）；`n_steps`（迭代步数，非负整数）；`bc`（`(左边界类型, 右边界类型)`，默认 **`("dirichlet", "dirichlet")`**，取值 `"dirichlet"` 或 `"neumann"`）。返回键：`u`（形状 (n,) 的终态温度场）、`stability_ratio`（float，`r = alpha·dt/dx²`）。
- **陷阱**：① `r > 0.5` 必然发散且**前几步看起来还正常**，等发现时已经溢出，本函数直接报错而不返回一堆 inf（判据带一个 docstring 未写明的极小正向松弛，故 r 恰在 0.5 附近仍可能放行）；② 时间只有一阶精度，误差 ~O(dt) + O(dx²)——想让显式与 Crank-Nicolson 对得上，必须把 dt 压到远小于 `dx²/alpha` 的量级；③ 零通量口径必须是"镜像虚拟节点 + 端点也参与更新"，若写成"端点复制邻居值"（`u_0 = u_1`）会引入一阶边界误差并**破坏守恒**；④ Dirichlet 锁的是 `u0` 的端点值（`u_init`），不是每次迭代后的端点值，中途改边界要另写接口；⑤ **r 恰为稳定边界 0.5 时奇数格点与偶数格点完全解耦**，等间距方波初值恰落在解耦子空间里，数值解会永远停在平顶伪稳态（自测实测偏差 1.2%），看似"实现错了"其实是格式固有陷阱；⑥ 返回的是终态，不保存中间场，要动画需自己包一层循环。
- **怎么检验**：自测口径——解析解 `u(x,t) = sin(πx)·e^{−alpha·π²·t}`，取 `alpha = 1.0`、`dx = 0.05`、`dt = 1e-6`（**r = 4e-4**，远在稳定域内）、21 个格点、200 步（t = 2e-4）：`spatial_heat_explicit_err`（与解析解的最大偏差）应 < **5e-3**（误差受空间项主导），`spatial_heat_explicit_vs_implicit`（与 CN 的最大差）应 < **1e-6**，`spatial_heat_explicit_ratio` 与 `alpha·dt/dx²` 的差应 < **1e-15**，`spatial_heat_rejects_unstable` 断言 `r > 0.5`（取 `dt = 1.2·0.5·dx²/alpha`）时确实抛 `ValueError`。零通量守恒：方波初值 `x ∈ (0.3, 0.7)` 取 1、其余取 0，`alpha = 1.0`、`dx = 0.1`、`dt = 0.005`、50 步，首末步离散总和之差应 ≤ **1e-12**（`spatial_heat_conservation_error`）。基模形状：11 格点（`dx = 0.1`）、`r = 0.4`、200 步的对称方波初值，`u / u_mid` 与 `sin(πx)` 的偏差应 < **1e-6** 且 `max|u| < 1`（零 Dirichlet 边界必须单调衰减）；**注意这条不能用 r = 0.5**。补充检验：手算单模态衰减因子 `e^{−alpha·π²·dt}` 逐模态核对；dt 减半误差约减半（时间一阶）而 dx 减半误差约降 4 倍（空间二阶）的收敛阶实验；换 `bc` 为 `("neumann", "neumann")` 后断言总热量守恒而对 Dirichlet 断言总热量必然衰减。

#### `heat_equation_1d_implicit(u0, alpha, dx, dt, n_steps, bc=("dirichlet", "dirichlet"))`

- **数学形式**：Crank-Nicolson（时间二阶、空间二阶）。内部点 `(1 + r)·u_i^{k+1} − (r/2)·(u_{i−1}^{k+1} + u_{i+1}^{k+1}) = (1 − r)·u_i^k + (r/2)·(u_{i−1}^k + u_{i+1}^k)`，`r = alpha·dt/dx²`；零通量端点用同一套镜像虚拟节点口径：`(1 + r/2)·u_0^{k+1} − (r/2)·u_1^{k+1} = (1 − r/2)·u_0^k + (r/2)·u_1^k`（右端同理，系数为 `1 + r/2` 与 `−r/2`）；Dirichlet 行是单位行，右端被强制写成 `u0` 的端点值。
- **步骤**：① 公共校验同 `heat_equation_1d_explicit`（**不检查稳定性条件**，CN 不需要）；② 算 `r` 与 `n`；③ 组装时间无关的稠密左端矩阵 `M` 与右端算子 `rhs_op`：内部行对角 `1 + r`、两侧 `−r/2`（右端算子为 `1 − r` 与 `+r/2`），Dirichlet 行为单位行，零通量行为 `1 + r/2` 与 `−r/2`；④ 每步 `b = rhs_op @ u`，再把 Dirichlet 行的 `b` 覆盖为 `u_init` 的端点值，`u = np.linalg.solve(M, b)`；⑤ 返回 `{"u": u, "matrix": M}`。
- **复杂度**：时间 O(n_steps · n³)（每步一次稠密解方程；实际三对角可用 Thomas 算法降到 O(n)）/ 空间 O(n²)（稠密矩阵）。
- **参数**：`u0`、`alpha`、`dx`、`n_steps`、`bc` 同显式格式；`dt`（时间步长，> 0）**不受稳定性限制**，可以远大于 `dx²/(2·alpha)`。返回键：`u`（形状 (n,) 的终态温度场）、`matrix`（形状 (n, n) 的稠密系数矩阵 `M`，满足 `M·u^{k+1} = b(u^k)`；Dirichlet 行是单位行，零通量行系数为 `1 + r/2` 与 `−r/2`）。
- **陷阱**：① **无条件稳定不等于无条件精确**：dt 很大时时间精度退化为 O(1)，解会"过度平滑"（把瞬态抹掉），稳定性与精度是两件事；② 每一行都必须有对角占优的符号结构，把 `+r/2` 放到左端会让矩阵失去对角占优、解出现非物理振荡；③ 每步都重新解一次方程（O(n³)）只是为了让代码短，生产代码应当先做一次 LU 分解（或直接用 Thomas 算法）复用；④ Dirichlet 行的右端被强制覆盖为常数 `u0[0]` / `u0[-1]`（docstring 陷阱段的措辞是"M 的对应用户传入的右端项会被覆盖"，实际被覆盖的是右端向量 `b`，而且本接口并没有"用户传入的右端项"这个参数，措辞与代码不符）；⑤ 返回值里的 `matrix` 是**稠密** n×n 矩阵，n 上万时内存 O(n²) 会先爆掉，不要拿它做大规模问题。
- **怎么检验**：自测口径与显式格式完全对齐（`alpha = 1.0`、`dx = 0.05`、`dt = 1e-6`、200 步）：`spatial_heat_implicit_err`（与解析解 `sin(πx)e^{−alpha·π²·t}` 的最大偏差）应 < **5e-3**，与显式格式的最大差应 < **1e-6**（此时两者时间阶数差 ~π⁴·dt·t/2 ≈ 1e-8 可忽略）。大 dt 的极值原理：方波初值、`dx = 0.1`、`dt = 0.05`（**r = 5**，显式早已发散）、10 步，`max|u|` 不得超过初值的 `max|u|`（容差 1e-12，`spatial_heat_implicit_bounded`），且断言 `matrix[5, 5]` 等于 `1 + r`（自测用 |差| ≤ 1e-12 校验对角元结构，`spatial_heat_cn_big_dt_ratio` 记录该算例的 r = 5）。补充检验：手算单模态放大因子 `g = (1 − 2r·sin²(kπh/2)) / (1 + 2r·sin²(kπh/2))`，由 `|g| < 1` 对任意 `r > 0` 恒成立即得无条件稳定与极值原理，可逐模态核对；用 `np.linalg.solve(M, b)` 与手写的 Thomas 三对角求解对拍（同一批 `b`）；断言 `M` 的 Dirichlet 行恰为单位行、零通量行系数为 `1 + r/2` 与 `−r/2`；同一算例下把 dt 缩小 10 倍，断言 CN 与显式格式的差按 O(dt) 缩小。

#### `poisson_2d_sor(n, f, omega=1.5, tol=1e-8, max_iter=5000)`

- **数学形式**：单位方形域上的 `-Δu = f`（零 Dirichlet 边界），n 个**内点**、步长 `h = 1/(n+1)`。五点差分 `-(u_{i−1,j} + u_{i+1,j} + u_{i,j−1} + u_{i,j+1} − 4u_ij)/h² = f_ij`，整理成 `u_ij = (四邻域和 + h²·f_ij)/4`；红黑 SOR 把 `(i+j)` 的奇偶当作两色（同色节点互不相邻），整块更新 `u_red = (1−ω)·u_red + (ω/4)·(四邻域和 + h²·f)`，再更新 black，其不动点与逐点 Gauss-Seidel SOR 完全一致但可向量化。
- **步骤**：① 校验 `n` 为 ≥ 2 的整数、`f` 形状必须恰为 `(n, n)`、`1.0 <= omega <= 2.0`、`tol > 0`、`max_iter` 为 ≥ 1 的整数（`bool` 被拒）；② `h = 1/(n+1)`、`h2f = h²·f`、`u` 全零初始化，用 `(i+j)%2` 预计算红黑掩码；③ 每次迭代对 color 0、1 各算一遍零填充的四邻域和并整块更新该色格点；④ 每步用 `max|4u − 四邻域和 − h²·f|` 的无穷范数作残差，`<= tol` 即提前退出；⑤ `max_iter` 步后仍未达标则抛 `ValueError`（不返回一个"看着像解"的结果）；⑥ 返回 `{"u": u, "n_iter": 迭代次数, "residual": 残差}`。
- **复杂度**：时间 O(n_iter · n²)（迭代次数受 `tol`、`omega`、`n` 共同控制）/ 空间 O(n²)。
- **参数**：`n`（每个方向的**内点**个数，≥ 2，无默认值，步长 `h = 1/(n+1)`）；`f`（右端项，形状 `(n, n)`，取值在**内点**网格上，形状不匹配直接报错）；`omega`（松弛因子，默认 **1.5**，必须落在 [1, 2]；1 即 Gauss-Seidel，接近 `2/(1+sin(πh))` 时最快）；`tol`（收敛判据，默认 **1e-8**，残差 `max|4u_ij − 四邻域和 − h²·f_ij| <= tol` 时停止，必须 > 0）；`max_iter`（最大迭代次数，默认 **5000**，≥ 1 的整数）。返回键：`u`（形状 (n, n) 的**内点**解，边界恒为 0、不在返回值里）、`n_iter`（int，实际迭代次数）、`residual`（float，最后一次迭代后的残差）。
- **陷阱**：① **ω 不是越大越快**：超过最优值后收敛变慢甚至发散，最优值随网格变细趋近 2，实践中最优 `ω ≈ 2/(1+sin(π/(n+1)))`，本模块默认 1.5 只是为了稳健；② 残差的量纲是"h² 倍方程残差"（**没有除以 h²**），同样的 `tol` 在更细的网格上对应更松的真实精度，跨网格比较收敛性时必须统一口径；③ 迭代法对高频误差收敛极快、对低频误差极慢，网格越细迭代次数 ~O(n)，大规模问题必须上多重网格或直接解法；④ 只有**零** Dirichlet 边界，非零边界值要把边界项移到右端，本函数没有这个接口；⑤ `f` 必须是**内点**采样值，形状不是 `(n, n)` 会直接报错，别把含边界点的 (n+2, n+2) 数组传进来；⑥ 不收敛时抛异常而不是返回近似解，因此调用方拿到的 `u` 一定满足 `residual <= tol`。
- **怎么检验**：自测用制造解 `u = sin(πx)·sin(πy)`、`f = 2π²·sin(πx)·sin(πy)`、`n = 16`、`ω = 1.5`、`tol = 1e-8`、`max_iter = 5000`（约 **155** 次迭代）：`spatial_poisson_max_err`（与解析解的最大偏差）应 < **5e-3**——注意这条判据对网格很敏感，h = 1/17 时五点差分的截断误差 ~π²h²/12 ≈ **2.8e-3**（docstring 陷阱段同一量在别处写作 2.9e-3）才是主导项，失败通常**不是 bug 而是网格太粗**；真正锐利的判据是 `spatial_poisson_vs_direct`——用 `np.linalg.solve` 直接解同一套离散方程（装配 4/±1 的拉普拉斯矩阵、右端 `h²·f`）做**与网格无关**的独立对拍，差应 < **1e-6**；`spatial_poisson_residual` 必须 ≤ tol = **1e-8**（若报收敛却超 tol 即为 bug）；`spatial_poisson_asymmetry`：对称右端项的解必须满足 `u = uᵀ`，不对称度应 < **1e-6**；`spatial_poisson_rejects_omega` 断言 `ω = 2.5` 抛 `ValueError`。补充检验：**制造解（MMS）**——取任意光滑 `u`，用离散拉普拉斯装配 `f`，断言数值解随 h 减半以 h² 阶收敛到 `u`；把 `omega` 从 1 递增到 `2/(1+sin(π/(n+1)))` 附近，断言 `n_iter` 先降后升（最优 ω 的存在性）；断言返回值 `u` 的边界隐含为 0（把它嵌进 (n+2, n+2) 数组后残差仍满足五点格式）。

#### `forest_fire_ca(n=30, p_grow=0.05, p_light=0.3, n_steps=50, seed=None, neighborhood="moore", initial_density=0.6)`

- **数学形式**：Drossel-Schwabl 型森林火灾元胞自动机。状态 `0 = 空地`、`1 = 树`、`2 = 燃烧`；每一步**同步**执行四件事：所有燃烧格变空地（燃烧只持续一步）→ 树格若邻域内存在燃烧格则被点燃、否则以 `p_light` 概率被雷击点燃 → 空地以 `p_grow` 概率长出树 → 记录树木占比。
- **步骤**：① 校验 `n` 为 ≥ 2 的整数、`p_grow`/`p_light`/`initial_density` 都落在 [0, 1]、`n_steps` 为非负整数（`bool` 被拒）、`neighborhood ∈ {"moore", "von_neumann"}`；② `gen = rng(seed)`，用一片 n×n 均匀随机数与 `initial_density` 比较生成初始树场（**初始没有燃烧格**）；③ 按 `neighborhood` 生成偏移表（moore 为 8 个方向，von_neumann 去掉 4 个对角方向）；④ 每步：把燃烧格零填充进 (n+2, n+2) 的 `pad`（网格外视为非燃烧），按偏移表叠加出"邻火"掩码；抽两片 n×n 随机数 `draw_light`、`draw_grow`；`ignite = 树 & (邻火 | draw_light < p_light)`、`grow = 空地 & (draw_grow < p_grow)`；⑤ 复制旧状态后整体赋值（燃烧→0、点燃→2、长树→1），这一步保证同步性；⑥ 累加本步点燃格数、记录树木占比；⑦ 返回 `{"steps", "tree_ratio", "burned_total", "final_trees"}`。
- **复杂度**：时间 O(n_steps · n²) / 空间 O(n²)。
- **参数**：`n`（方形网格边长，默认 **30**，≥ 2，共 n·n 个格子）；`p_grow`（每个空地每步长出树的概率，默认 **0.05**，取值 [0, 1]）；`p_light`（每棵**未被邻火波及**的树每步被雷击点燃的概率，默认 **0.3**，取值 [0, 1]）；`n_steps`（演化步数，默认 **50**，非负整数）；`seed`（随机种子，默认 None → `DEFAULT_SEED`）；`neighborhood`（默认 **`"moore"`**（8 邻域），或 `"von_neumann"`（4 邻域），决定火势蔓延范围，网格外视为非燃烧）；`initial_density`（初始时刻每个格子是树的概率，默认 **0.6**，取值 [0, 1]）。返回键：`steps`（长度 n_steps 的 `list[float]`，第 k 项是第 k 步**之后**的树木占比）、`tree_ratio`（float，终态树木占比 = `final_trees / (n·n)`）、`burned_total`（int，累计**点燃次数**）、`final_trees`（int，终态树格数）。
- **陷阱**：① **必须同步更新**：如果就地更新（先烧的格子立刻去点燃邻居），火势会在一"步"内传遍整个网格、永远看不到蔓延过程，本实现先算掩码再整体赋值；② `p_grow` 很小而 `p_light` 很大时系统落到"树刚长出来就被烧掉"的稀疏稳态、树占比极低，想看自组织临界性要把 `p_grow/p_light` 的比值调到 **1e-3** 量级并跑很长时间；③ 邻域在网格外按"非燃烧"处理（**非周期**），边界上的火势蔓延比内部慢，小网格下边界效应会明显压低燃烧总量；④ 每步固定消耗两片 n×n 的随机数（长树、雷击各一片），删除或增加一次抽样都会改变后续**所有**随机数从而改变结果——这也是它能通过确定性自测的原因，改代码时不要调整抽样顺序；⑤ `burned_total` 数的是"点燃次数"而不是"被烧掉的格数"：同一格烧完变空地后若重新长树再被点燃会再计一次；⑥ 燃烧态只存在一步，因此任何时刻都不会看到"持续燃烧"的格子。
- **怎么检验**：自测给了两条**极限情形的确定结论**：`p_light = 0`（`n = 20`、`p_grow = 0.05`、`n_steps = 30`、`seed = 7`）时 `burned_total` 必须恰为 **0**（`spatial_fire_no_light_burned`，无雷击则无火源）；`p_grow = 0` 且 `p_light = 1`（同参数）时 `final_trees` 必须恰为 **0**（`spatial_fire_no_grow_trees`，不长树且每步雷击）。默认参数（`n = 30`、`p_grow = 0.05`、`p_light = 0.3`、`n_steps = 50`、`seed = None`）断言 `len(steps) == 50`、`burned_total > 0`，并返回 `spatial_fire_tree_ratio`。补充检验：断言 `steps` 每项都落在 [0, 1] 且终态 `tree_ratio` 恰等于 `final_trees/(n·n)`；**同步性检验**——断言单步新增点燃格数不超过"上一步燃烧格数 × 邻域大小"（就地更新会立刻违反这条）；同一 `seed` 两次调用逐位一致（确定性）；把 `neighborhood` 换成 `"von_neumann"` 后断言"单步蔓延距离不超过 1 格"（切比雪夫距离）而 moore 允许对角蔓延；只做 `p_light = 0` 与 `p_light > 0` 的极限对照（分别对应"零燃烧"与"有燃烧"），**不要**断言中等 `p_light` 之间 `burned_total` 单调——火越猛燃料消耗越快，总点燃次数完全可以非单调。

#### `traffic_ca_nagel_schreckenberg(n_cells=100, n_cars=20, v_max=5, p_brake=0.3, n_steps=100, seed=None)`

- **数学形式**：Nagel-Schreckenberg 一维交通流元胞自动机（**周期性边界**）。每步对全部车辆**同步**执行 NaSch 四规则、顺序不能换：① 加速 `v ← min(v + 1, v_max)`；② 减速 `v ← min(v, gap)`，`gap` 为到前车的空格数；③ 随机慢化：以概率 `p_brake` 令 `v ← max(v − 1, 0)`；④ 前进 `x ← (x + v) mod n_cells`。规则 ② 保证 `v <= gap`，车辆不会互相超越，位置序列始终有序。
- **步骤**：① 校验 `n_cells` 为 ≥ 2 的整数、`n_cars` 为整数且满足 `1 <= n_cars < n_cells`、`v_max` 为 ≥ 1 的整数、`p_brake ∈ [0, 1]`、`n_steps` 为 ≥ 1 的整数（`bool` 一律被拒）；② `gen = rng(seed)`，位置用 `gen.choice(n_cells, size=n_cars, replace=False)` 不放回抽取后排序，**初始速度全 0**；③ 每步：`gap = (roll(pos, -1) − pos − 1) mod n_cells`，依次执行加速、减速、随机慢化（一片 `n_cars` 的随机数）、前进，再按新位置 `argsort(kind="stable")` 重排 `pos`/`vel` 以维持前车关系；④ 记录当步瞬时流量 `Σv / n_cells` 与平均车速；⑤ 取最后 `min(20, n_steps)` 步求均值后返回 `{"flow", "density", "mean_speed", "steps"}`。
- **复杂度**：docstring 写时间 O(n_steps · n_cars) / 空间 O(n_cars)，但每步有一次 `argsort`，实际时间应记作 **O(n_steps · n_cars·log n_cars)**（这条与 docstring 的复杂度声明略有出入）。
- **参数**：`n_cells`（环形道路格子数，默认 **100**，≥ 2）；`n_cars`（车辆数，默认 **20**，必须 `1 <= n_cars < n_cells`）；`v_max`（最大速度，即每步最多前进的格数，默认 **5**，≥ 1 的整数）；`p_brake`（随机慢化概率，默认 **0.3**，取值 [0, 1]；取 0 即确定性 NaSch）；`n_steps`（演化步数，默认 **100**，≥ 1）；`seed`（随机种子，默认 None → `DEFAULT_SEED`）。返回键：`flow`（float，最后 `min(20, n_steps)` 步的平均流量，口径是"每步移动的车辆数 / 格子总数"，单位车/格/步，乘以 `n_cells` 就是车/步）、`density`（float，密度 `n_cars / n_cells`）、`mean_speed`（float，同一时间窗内的平均车速，即每车每步前进的格数）、`steps`（长度 n_steps 的 `list[float]`，每步的瞬时流量）。返回值**不含**逐车位置与速度。
- **陷阱**：① **规则顺序不能变**：先随机慢化再加速会得到完全不同的基本图（"慢化"变成了"加速前的抖动"），流量-密度曲线会整体偏移；② 平均车速有一个硬上界 `(n_cells − n_cars)/n_cars`（因为 `v_i <= gap_i` 且全部 `gap` 之和恰好是空格总数），所以当密度 `ρ > 1/(v_max + 1)`（`v_max = 5` 时即 `ρ > 1/6`）时**平均车速不可能接近 v_max**，这是基本图自由流分支的上界、不是实现 bug，想验证"自由流接近 v_max"必须用 `ρ <= 1/(v_max + 1)`；③ `p_brake > 0` 时会出现"幽灵堵车"，流量-密度曲线在中等密度处下降，而单次仿真的瞬时流量波动很大，必须用时间窗平均（本函数用最后 20 步，`n_steps < 20` 时窗口退化为全部步数）；④ 初始速度全 0 会让前若干步处于加速瞬态，比较不同参数时必须用相同步数与相同时间窗，否则会把瞬态差异当成参数效应；⑤ 默认参数 `n_cars = 20 / n_cells = 100` 即 `ρ = 0.2 > 1/6`，此时平均速度受 `(1 − ρ)/ρ` 上界约束，不要拿"默认参数下速度太慢"当 bug；⑥ `steps` 记的是瞬时流量，直接对它求均值**不等于** `flow`（除非 `n_steps <= 20`）。
- **怎么检验**：自测四条：`ρ = 0.1`（`n_cells = 100`、`n_cars = 10`、`v_max = 5`、`p_brake = 0`、`n_steps = 100`，满足 `ρ < 1/(v_max+1) = 1/6`）时 `mean_speed` 必须 **> 0.9 × v_max = 4.5**（`spatial_traffic_free_speed_ratio` 是 `mean_speed / 5`）；同一时间窗内 `flow` 必须**逐位等于** `density × mean_speed`（差 ≤ 1e-12，`spatial_traffic_flow` / `spatial_traffic_density`）——这是流量口径最锐利的一致性检验；`p_brake` 从 0 增到 **0.3** 后 `mean_speed` 必须下降（`spatial_traffic_speed_decreases`）；`ρ = 0.2`（`n_cars = 20`）时 `mean_speed` 不得超过 `(100 − 20)/20 = 4`（加 1e-9 容差，`spatial_traffic_dense_speed`）；另外断言 `len(steps) == n_steps`。补充检验：**手算确定性特例**——`p_brake = 0`、`n_cars = 1` 时单辆车每步加速到 `v_max` 后匀速，窗口平均车速应恰为 `v_max`（无干扰闭式解），这也是"规则 ① 加速到上限"的判别性实验；`ρ → 0` 时应有 `flow ≈ ρ · v_max`（自由流基本图）；把 `p_brake` 设为 0 后连续两次调用断言结果**逐位相同**（确定性 NaSch）；同一 `seed` 两次调用逐位一致。

#### `buckingham_pi(dims)`

- **数学形式**：Buckingham π 定理。把 `dims` 组装成量纲矩阵 `D`（**行 = 基本量纲，列 = 物理变量**），则独立无量纲组个数 `n_pi = 变量数 − rank(D)`。对 `D` 做精确有理数行最简形（RREF），每个**自由列** c 对应一个零空间基向量：`x_c = 1`、主元列 `p_i` 上取 `x_{p_i} = −RREF[i, c]`，其余为 0；再把基向量规范化为"整数、互质、首个非零指数为 +1"，于是 `π = Π_k 变量_k^{x_k}`。
- **步骤**：① 校验 `dims` 是 dict、至少 2 个物理变量；用 `as_vector` 逐个校验指数向量，并强制所有向量长度一致（不一致直接报错）；② 用 `np.column_stack` 组装 (k 个基本量纲, n 个变量) 的量纲矩阵；③ 每个元素经 `Fraction(float(x)).limit_denominator(10**6)` 转精确有理数；④ 逐列选主元、行交换、归一化、消元，得到 RREF 与主元列集合 → `rank`；⑤ `rank == 0` 抛 `ValueError`；⑥ 对每个自由列构造零空间基向量并用 `_canonical_int_vector` 规范化（约去分母 → 除以最大公约数 → 首个非零分量取正）；⑦ 生成可读字符串 `"pi_{idx} = name^e * name^e ..."`（该项全零时写作 `"1"`）；⑧ 返回 `{"rank", "n_pi", "pi_groups", "pi_names"}`。
- **复杂度**：时间 O(k · n²)（k 为基本量纲数、n 为变量数，精确有理数运算比浮点慢但结果可复现）/ 空间 O(k · n)。
- **参数**：只有 `dims`（`{变量名: [各基本量纲指数]}`，无默认值）。docstring 给的阻力问题例子是 `{"F": [1, 1, -2], "v": [0, 1, -1], "rho": [1, -3, 0], "mu": [1, -1, -1], "L": [0, 1, 0]}`，基本量纲顺序取 `M, L, T`；至少 2 个变量，所有指数向量长度必须一致，否则抛 `ValueError`。没有其他可调参数（π 组的规范化约定写死在实现里，不可配置）。返回键：`rank`（int，量纲矩阵的秩）、`n_pi`（int，独立无量纲组个数 = 变量数 − rank）、`pi_groups`（list，每项是长度为"变量数"的整数指数向量，**变量顺序 = `dims` 的键顺序**）、`pi_names`（list[str]，对应的可读字符串，例如 `"pi_2 = F^1 * rho^-1 * v^-2 * L^-2"`）。
- **陷阱**：① **基本量纲必须线性无关且完整**：若把 M 和"力"同时当基本量纲（力本身 = M·L·T⁻²），量纲矩阵会降秩、`n_pi` 偏大，给出的 π 组没有物理意义；② **π 组不唯一**：任何 π 组的乘积/幂仍是 π 组，本函数固定了规范化代表，但换一个变量顺序或换主元列就会得到另一组同样正确的基，论文里报 π 组时必须同时报变量顺序与这批规范化约定；③ `rank = 0`（所有变量都是无量纲的）时没有任何 π 组可求，直接报 `ValueError`；④ 输入指数必须是**相对同一组基本量纲**的，混用（有人写 CGS 有人写 SI）不会报错但结果无意义；⑤ 指数向量长度不一致会直接报错——这类错误在建模里通常意味着"漏写了一个量纲"；⑥ 指数经 `limit_denominator` 有理化，非整数指数（如 1/3 次方）能被表达，但带浮点噪声的指数可能被吸收成奇怪的有理数，同时 `pi_groups` 返回的是**整数**向量，非整数指数只有在整体缩放后才能取整时才会保持原样。
- **怎么检验**：自测用经典阻力问题（`F, v, rho, mu, L`，5 个变量）：`spatial_pi_rank` 必须为 **3**、`spatial_pi_n_pi` 必须为 **5 − 3 = 2**；`pi_groups` 里必须含指数向量 **[1, −2, −1, 0, −2]**（即 `F/(ρ v² L²)`，`spatial_pi_drag_group`）；再用 `np.linalg.matrix_rank` 独立算一遍量纲矩阵的秩互相印证（`spatial_pi_rank_matches`）。补充检验：**定义级检验**——断言对每个 π 组都有 `D · π = 0`（指数向量代回量纲矩阵必须为零向量），这条不依赖任何实现细节；断言每组指数互质（gcd = 1）且首个非零指数为 +1（规范化约定的判别性实验）；断言 `n_pi == n_vars − rank` 恒成立；对简单算例手算核对（例如 M/L/T 下取 `v, L, g` 三变量应得到 1 个 π 组，其指数向量对应 `v²/(g·L)`）；把 `dims` 的键顺序换一遍，断言新 π 组仍满足 `D·π = 0` 且个数不变（只差一个基变换）。

#### `scaling_similarity(model_ratio, measurements, target, exponents=None)`

- **数学形式**：按几何相似比与量纲幂次做缩尺换算。`factor_v = λ^{e_v}`（λ = 原型/模型）、`scaled_v = measurements_v · factor_v`、`relative_error_v = |scaled_v − target_v| / |target_v|`（`target_v = 0` 时退化为绝对误差，避免 0 除）。典型幂次：长度 e = 1、面积 e = 2、体积 e = 3；Froude 相似下速度 e = 0.5、力 e = 3。
- **步骤**：① `lam = float(model_ratio)`，`lam > 0` 否则抛 `ValueError`；② 校验 `measurements` 与 `target` 都是非空 dict，且键集合**完全一致**（不一致报错并打印两边的键）；③ `exponents` 为 None 时全部取 1.0（纯几何缩放），否则必须是与 `measurements` 键一致的非空 dict，逐项 `float()`；④ 逐物理量把测量值与参考值转 float 并检查**有限性**（非有限即报错）；⑤ 算 `factor = λ^e`、`value = m·factor`、相对误差（`target` 为 0 时用绝对误差）；⑥ 返回 `{"factors", "scaled", "relative_error"}`。
- **复杂度**：时间 O(k) / 空间 O(k)（k 为物理量个数）。
- **参数**：`model_ratio`（几何相似比 `λ = L_原型 / L_模型`，必须 > 0，无默认值）；`measurements`（`{物理量名: 缩尺模型上的实测值}`，不能为空）；`target`（`{物理量名: 原型上的参考值}`，键必须与 `measurements` 完全一致，用来算 `relative_error`）；`exponents`（`{物理量名: 相对长度的量纲幂次 e}`，默认 **None** 表示全部取 1.0）。返回键：`factors`（`{物理量名: λ^e}`）、`scaled`（`{物理量名: measurements × factor}`）、`relative_error`（`{物理量名: |scaled − target| / |target|}`，`target = 0` 时退化为绝对误差 `|scaled − target|`）。
- **陷阱**：① **相似比的方向**：λ 定义为"原型 / 模型"，若误用"模型 / 原型"，所有因子会整体取倒数、换算出的"原型"值小得离谱，论文里必须写清 λ 的定义方向；② 比例缩放只在**几何相似 + 相关无量纲数相等**时成立，Froude 相似与 Reynolds 相似一般不能同时满足（除非用变尺度模型），所以"用同一个 λ 换算出所有量"是有前提的；③ 无量纲量（e = 0）的因子恒为 1、不随尺度变化，若算出它的换算值变了说明 `exponents` 传错了；④ `target` 必须是**独立观测**的原型参考值，如果把 `scaled` 自己传进 `target`，`relative_error` 会恒等于 0，这个自测就什么也没验证；⑤ 外推范围：λ 很大时缩尺模型的粘性/表面张力等效应会被放大，纯幂次换算不再成立；⑥ 键集合不一致、空 dict、非有限值都会报错，但"物理量名写对、量纲幂次写错"不会报错，只会静默给出错误的换算值。
- **怎么检验**：自测取 `model_ratio = 100.0`、`exponents = {"length": 1.0, "area": 2.0, "velocity": 0.5, "force": 3.0, "Re": 0.0}`：`spatial_scale_length_factor` 应等于 **100**（λ）、面积因子应等于 **λ² = 1e4**（容差 1e-9）、`spatial_scale_velocity_factor` 应等于 **sqrt(λ) = 10**（Froude 相似）、无量纲量 `Re` 的因子必须恒为 **1**（容差 1e-12）。手算核对 `scaled`（`measurements = {length: 0.3, area: 0.09, velocity: 2.0, force: 12.0, Re: 1e6}`）：`0.3 × 100 = 30`、`0.09 × 1e4 = 900`、`2.0 × 10 = 20`、`12 × λ³ = 1.2e7`、`1e6 × 1 = 1e6`——与自测构造的 `target` 逐一对应，故 `spatial_scale_max_rel_error`（各量相对误差的最大值）应为 **0**。补充检验：断言 `λ = 1` 时所有因子恒为 1；断言 e = 0 的物理量映射是恒等映射；单独构造一组带已知相对误差的 `target`，断言 `relative_error` 与手算的 `|scaled − target|/|target|` 一致；`target` 某项取 0 时断言返回的是绝对误差而不是 `inf`/`nan`；把 `scaled` 自己当 `target` 传回去，断言 `relative_error` 全 0（用来提醒自己这条检验是恒真的、不能当验证用）。

---

## 4. 验证记录

本文件中的"复杂度 / 参数 / 陷阱"三栏与 `examples/algorithms/` 的实际代码**逐条对应**，并通过以下机制防止文档漂移：

- `examples/run_algorithms.py` 在 CI 中运行全部模块并与黄金值比对——代码改坏了会被拦下；
- 每个模块的 `_self_test()` 内含闭式解/独立实现对拍，不是回显自己的输出；
- 文档中出现的函数名可在 `examples/algorithms/` 中直接检索到；若你发现文档与代码不符，请提 issue（见 `CONTRIBUTING.md`）。
