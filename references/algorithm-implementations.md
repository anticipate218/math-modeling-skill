# 模型 → 算法实现索引

> 这份索引回答一个问题：**"我要的模型，用哪个算法实现、复杂度多少、代码在哪、什么时候该换成熟库、哪里最容易算错。"**
>
> 配套关系：`references/model-library.md` 答"这题该用哪类模型"；`references/model-implementations.md` 答"怎么从基线升级"；**本文件答"具体到算法与代码"**；`references/github-resources.md` 答"外部库怎么选、怎么引用"。

---

## 1. 使用前提：这份实现的能力边界

本仓库 `examples/algorithms/` 下的实现是**教学透明版**，定位是"让论文能交代清楚每一步"，不是工业级数值库。请先接受以下约定，再决定要不要用：

| 约定 | 具体含义 |
|---|---|
| **依赖边界** | 只允许 `numpy` + Python 标准库。**不含** scipy / sklearn / statsmodels / pandas / cvxpy / pulp / torch 等。CI 用 AST 静态扫描强制这条规则。 |
| **Python 版本** | 3.9+，非交互，无网络访问。 |
| **返回形态** | 统一返回 `dict`（键为 `weights` / `x` / `status` 这类具名结果），便于打印进论文表格。 |
| **随机性纪律** | 一律走 `_common.rng(seed)`，**绝不使用 `np.random` 的全局状态**；默认种子 `DEFAULT_SEED = 20240101`。同一份代码两次运行结果完全一致。 |
| **自检** | 每个模块都有 `_self_test()`，返回 `dict`，内含该模块关键数值。`examples/run_algorithms.py` 会跑所有模块并与 `algorithms_golden.json` 比对。 |
| **规模上限** | 见各族"复杂度"列。超过几千个决策单元/样本时请换成熟库——本实现会慢，且部分算法是朴素 O(n²)/O(n³) 写法。 |

**为什么不用 scipy 反而写一遍？** 竞赛论文的附录代码是要被评委看的。调用 `scipy.optimize.linprog` 一行解决，但说不清算法；自己写的两阶段单纯形能让你在论文里写清"松弛变量、检验数、迭代终止条件"。**两者不冲突**：用本仓库实现讲原理，用成熟库做交叉验证（见第 6 节）。

---

## 2. 一眼速查表

| 题目族 | 模块 | 主要函数 | 典型赛题信号 |
|---|---|---|---|
| 优化与调度 | `optimization.py` | `simplex_lp` `branch_and_bound_ilp` `knapsack_dp` `assignment_hungarian` `transportation_vogel` | "如何分配/排班/选址使成本最小" |
| 路径与图网络 | `graphs.py` | `dijkstra` `floyd_warshall` `kruskal_mst` `max_flow_edmonds_karp` `tsp_nearest_neighbor` `tsp_two_opt` `pagerank` `connected_components` | "最优路径/管网铺设/最大运力/关键节点" |
| 组合优化与元启发式 | `heuristics.py` | `simulated_annealing` `genetic_algorithm` `particle_swarm` `ant_colony_tsp` | "NP-hard、目标不可导、规模大到精确解解不动" |
| 预测与时间序列 | `forecasting.py` | `moving_average` `exponential_smoothing` `holt_linear` `holt_winters` `ar_model` `adf_test` `acf` `pacf` `rolling_origin_cv` | "预测未来若干期的销量/需求/流量" |
| 统计推断与回归 | `statistics.py` | `pearson_corr` `spearman_corr` `kendall_tau` `t_test_one_sample` `t_test_two_sample` `chi_square_test` `jarque_bera` `anderson_darling` `ks_test_normal` `ols` `vif` `ridge_regression` `logistic_regression` `bootstrap_ci` `permutation_test` | "哪些因素显著、相关性强不强、分布是否正态"（`shapiro_wilk` 刻意不实现，见 §3.5） |
| 评价与决策 | `evaluation.py` | `ahp_weights` `entropy_weights` `critic_weights` `topsis` `vikor` `grey_relational_grade` `dea_ccr` `dea_bcc` `fuzzy_comprehensive_eval` `topsis_rank_sensitivity` | "给若干方案排序/打分/评效率" |
| 聚类与分类 | `clustering.py` | `kmeans` `kmeans_plusplus_init` `silhouette_score` `elbow_curve` `agglomerative` `dbscan` | "把样本分成几类、客户分群" |
| 微分方程与机理 | `differential.py` | `solve_ivp_euler` `solve_ivp_rk4` `simulate_sir` `logistic_growth` `lotka_volterra_rhs` `fit_sir_least_squares` `estimate_convergence_order` | "传染病/种群/物理过程随时间演化" |
| 随机模型与仿真 | `stochastic.py` | `mc_pi` `mc_integrate` `mm1_metrics` `mm1_simulate` `markov_steady_state` `markov_absorption` `gamblers_ruin` | "排队/可靠性/马尔可夫状态转移/概率估计" |
| 几何与空间 | `geometry.py` | `convex_hull` `polygon_area` `point_in_polygon` `haversine` `idw_interpolate` `ordinary_kriging` | "选址覆盖、区域面积、经纬度距离、空间插值" |
| 博弈与网络 | `game.py` | `zero_sum_value_lp` `nash_support_enumeration` `shapley_value` `gale_shapley` `replicator_dynamics` | "对抗/合作、收益分配、稳定匹配、策略演化" |

---

## 3. 逐族详解

复杂度一列是**该算法的教科书复杂度**（部分引自模块 docstring），不是本机实测基准。本实现多为朴素写法，未做索引/前缀和/稀疏化优化，常数因子通常比成熟库大。

### 3.1 优化与调度 —— `examples/algorithms/optimization.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `simplex_lp` | 两阶段单纯形，支持 `A_ub`/`A_eq`/变量上下界 | 最坏 O(2ⁿ)，实际很快 | ① 最坏指数级但**实用极快**，别在论文里写"复杂度指数所以不能用"；② `bounds` 的上下界行必须**等所有列确定后**再构造，否则 numpy 会静默广播出长度错误的约束行（本实现早期真实踩过）；③ 报 `status` 一定要检查，不可行/无界时 `x` 无意义 |
| `branch_and_bound_ilp` | LP 松弛 + 分支定界 | O(2ⁿ) 最坏 | 只适合**小规模**（几十个整数变量）；节点数上限 `max_nodes` 决定了它会给次优解，论文必须报告 gap |
| `knapsack_dp` | 0-1 背包动态规划 | O(n·capacity) | 容量必须**整数**；容量很大时伪多项式会爆内存 |
| `assignment_hungarian` | 匈牙利 / Kuhn-Munkres | O(n²m) → O(n³) | 要求完全匹配；非方阵需补虚拟行列，补的代价要说明 |
| `transportation_vogel` | Vogel 近似初始解 + 位势法（MODI） | Vogel O((m+n)³) + 单纯形 | 供需不平衡时须先补虚拟产地/销地；退化基可行解要处理 0 分配 |

**外部库**：`google/or-tools`（CP-SAT、LP、MIP、TSP/VRP）、`Pyomo/pyomo`（代数建模，可接多种求解器）。正规论文若要报"全局最优"，请用求解器并写明 gap。

### 3.2 路径与图网络 —— `examples/algorithms/graphs.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `dijkstra` | 二叉堆 + 惰性删除 | O((V+E) log V) | **边权必须非负**；有负权要用 Bellman-Ford |
| `floyd_warshall` | 动态规划全源最短路 | O(n³) | 无边用 `np.inf`；对角元按给定值处理不作强制清零 |
| `reconstruct_path` | 由 `prev`（Dijkstra）或 `next_node`（Floyd）还原路径 | O(V) | 两种口径不同，传错会得到反向或错误路径 |
| `kruskal_mst` | 并查集 + 按权排序 | O(E log E) | 只对**连通图**给生成树；不连通时会得到森林，必须检出来 |
| `max_flow_edmonds_karp` | BFS 增广（Ford-Fulkerson） | O(V·E²) | 无向边要**两个方向都写**容量；容量字典漏写方向会静默少算流 |
| `min_cut_edges` | 残量网络可达性构造最小割 | O(V·E²) | 给的是**边集**不是容量和；应与最大流值相等，这是自检点 |
| `tsp_nearest_neighbor` | 贪心构造 | O(n²) | **不保证最优**，典型比最优差 20–25%；只能当 2-opt 的起点 |
| `tsp_two_opt` | 2-opt 局部搜索 | O(n²)/轮 | 仍是局部最优；论文必须报"改进幅度"而非宣称最优 |
| `pagerank` | 幂迭代 + 悬挂点处理 | O(iter·E) | 权重是**转移强度**不必预先归一化；不处理悬挂点会让概率漏掉 |
| `connected_components` | 无向化 + 标签传播 | O(V+E) | 方向被忽略，有向图的可达性不能用它 |

**外部库**：`networkx/networkx`（图算法齐全，适合交叉验证）。

### 3.3 组合优化与元启发式 —— `examples/algorithms/heuristics.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `simulated_annealing` | Metropolis + 几何降温 | O(iters × (T_cost + T_neighbor)) | **约定最小化**，最大化请取负；`T0`/`alpha`/`iters` 三个参数必须写进论文，否则结果不可复现 |
| `genetic_algorithm` | 二进制编码 + 锦标赛 + 单点交叉 + 位翻转 | O(gens × pop × n_genes) | 必须报种群规模、代数、交叉/变异率、选择方式；不报等于没做 |
| `particle_swarm` | 惯性权重线性递减 PSO | O(iters × pop × (dim + T_obj)) | 必须报 `w` 递减区间、`c1`/`c2`、粒子数；边界处理方式会影响结果 |
| `ant_colony_tsp` | Ant System：概率构造 + 信息素挥发 + 加强 | O(iters × n_ants × n²) | 只对**对称** TSP；`alpha`/`beta`/`rho` 与信息素初值都要报 |

**写论文的硬要求**：元启发式的结果**不是"答案"而是"一次搜索的结果"**。必须交代：(1) 随机种子与重复次数；(2) 参数表；(3) 至少 10–30 次独立重复的最优值分布（最好给箱线图）；(4) 与小规模精确解对照的 gap。只跑一次就给最优解，是评委最容易识破的失分点。

**外部库**：`google/or-tools`（CP-SAT 对组合问题往往比手写元启发式更稳）。

### 3.4 预测与时间序列 —— `examples/algorithms/forecasting.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `moving_average` | 滑动平均 | O(n·window) | 窗口越大越滞后；**滞后性必须在图上体现**，否则会被问"为什么预测总是慢一拍" |
| `exponential_smoothing` | 指数平滑（SES） | O(n) | `alpha` 要标定而非拍脑袋；无趋势无季节，别拿它预测上升序列 |
| `holt_linear` | Holt 线性趋势 | O(n) | 趋势会**线性外推失控**，长跨度预测要设上限 |
| `holt_winters` | Holt-Winters 三次平滑 | O(n + period) | `mode="additive"/"multiplicative"` 选择要看季节幅度是否随水平变化；乘法型要求数据全正 |
| `acf` / `pacf` | 自/偏自相关 | O(n·nlags) / O(n·nlags + nlags²) | ±2/√n 参考线只是**白噪声下的渐近近似**；序列有趋势或季节时该线不可用 |
| `ar_model` | AR(p) 最小二乘 | O(n·p + p²) | 阶数选择要报（AIC/BIC 或 PACF 截尾），不能直接给个 p |
| `difference` | 差分（含季节差分） | O((order+1)·n) | 差分次数决定后续要不要还原（积分）；论文必须说明还原步骤 |
| `mackinnon_crit` | MacKinnon 响应面近似 | O(1) | **是查表近似不是精确分布**；`nobs` 必须传**回归实际用的观测数**（扣除滞后与差分后），传原始长度会让结论反转；nobs < 20 时近似不可靠 |
| `adf_test` | ADF 单位根检验（AIC 选滞后阶） | O(max_lag² · n) | "不拒绝单位根"≠"证明有单位根"（检验功效低）；结构突变会让 ADF 误判 |
| `mape` / `rmse` / `mae` | 误差指标 | O(n) | **MAPE 在真值接近 0 时爆炸**，这类序列改用 MAE/RMSE |
| `theil_u` | Theil's U | O(n) | U<1 才优于朴素预测；U≈1 说明你的模型没打败朴素基线 |
| `train_test_split_ts` | 时序切分 | O(n) | **绝不能随机打乱**，时序必须按时间前后切 |
| `rolling_origin_cv` | 滚动原点交叉验证 | O(n/step) | 只生成索引不训练模型；这是时序模型唯一诚实的评估方式，别用 k-fold |

**外部库**：`Nixtla/statsforecast`（统计预测基线最快）、`sktime/sktime`（统一接口 + 回测）、`unit8co/darts`（含深度学习）。

### 3.5 统计推断与回归 —— `examples/algorithms/statistics.py`

模块自带 `_betainc` / `_gammainc_q` / `_norm_cdf` 等分布函数，因此**不依赖 scipy**；这些私有函数已与 `scipy.special` 对照（最大绝对误差 ≤ 4e-13）。

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `pearson_corr` | Pearson 积矩相关 + t 检验 p 值 | O(n) | 只测**线性**关系；对离群值极敏感，先画散点图再报 r |
| `spearman_corr` | 秩相关（平均秩处理并列） | O(n log n) | 单调非线性关系下比 Pearson 合适，但**不能**解释成"线性强度" |
| `kendall_tau` | Kendall τ 双重循环 + 正态近似 p | O(n²) | n 上万会很慢；p 值是**渐近口径**，n<30 或并列多时与精确置换 p 有差距 |
| `t_test_one_sample` / `t_test_two_sample` | t 检验（`equal_var=True` 合并方差 / `False` Welch） | O(n) | 默认该用 **Welch**；方差齐性要先检验或直接放弃合并方差假设 |
| `chi_square_test` | 拟合优度 / 独立性卡方（自动判 df） | O(rc) | **期望频数 < 5 的格子超过 20% 时近似失效**，应合并类别或用精确检验 |
| `shapiro_wilk` | **不实现，直接抛 `NotImplementedError`** | 无 | 见下方说明。需要 S-W 请用 `scipy.stats.shapiro` |
| `jarque_bera` | 偏度/峰度联合检验 | O(n) | 大样本下**过度敏感**：n 上千时微小偏离也会"显著" |
| `anderson_darling` | A² 统计量 + D'Agostino-Stephens 分段 p 值 | O(n log n) | 尾部加权，与 KS 互补；**临界值表与 scipy 不同**，见下方说明 |
| `ks_test_normal` | KS 统计量 + Kolmogorov 渐近分布（Stephens 修正） | O(n log n) | **参数由样本估计后 p 值虚高**（Lilliefors 问题）；本函数不做修正，只返回 `parameters_estimated` 标志 |
| `ols` | 正规方程最小二乘 + 完整推断量 | O(n p²) | 共线设计矩阵直接抛 `ValueError`；**别用正规方程解病态问题**（应换 QR/SVD） |
| `vif` | 逐列对其余列 + 截距回归取 R² | O(p·n·p²) | **只在"线性"共线时有效**（X₂=X₁² 看不出来）；VIF>10 只是惯例不是定理 |
| `ridge_regression` | 标准化后的闭式解 | O(n p² + p³) | **不标准化等于按量纲施加惩罚**（"元"改"万元"惩罚就失效）；**刻意不返回 p 值**（有偏估计下 t 检验不适用） |
| `logistic_regression` | IRLS / Newton-Raphson | O(max_iter·n p²) | **完全分离时 MLE 不存在**，必须看 `converged`；不看收敛标志就报系数是常见错误 |
| `bootstrap_ci` | 百分位法 Bootstrap | O(n_boot·n) | 只做百分位法，**未做 BCa/学生化**，偏斜统计量覆盖率会低于名义水平；时间序列直接对点重抽样会破坏自相关 |
| `permutation_test` | 置换检验（两组均值差） | O(n_perm·(nx+ny)) | 不依赖分布假设，但**只能检验"同分布"而非单纯均值**；置换次数少时 p 值分辨率差 |

**为什么 `shapiro_wilk` 是抛异常而不是实现**：Shapiro-Wilk 的 W 需要正态次序统计量期望的权重表，精确 p 值依赖 Royston (1995) / AS R94 按 n 分段的多项式系数。在只允许 numpy + 标准库的前提下无法逐项核对那张表，**凭记忆写出来会得到"能跑但偏差几个百分点"的结果**——宁可少一个函数也不返回编造的 W 与 p 值。模块内还点명了一个常见误用：把 Blom 分数 `m_i/‖m‖` 当作 `a_i`，算出来的是 Shapiro-**Francia** 统计量却被标成 "Shapiro-Wilk"。替代路径：`scipy.stats.shapiro`（推荐），或本模块的 `anderson_darling` / `jarque_bera` / `ks_test_normal`。

**为什么 `anderson_darling` 的临界值和 scipy 对不上**：本模块用 `{15%: 0.561, 10%: 0.631, 5%: 0.752, 2.5%: 0.873, 1%: 1.035}`；`scipy.stats.anderson(x, dist="norm")` 返回 `[0.571, 0.651, 0.781, 0.911, 1.083]`。**A² 统计量本身两边逐位相同**（n=20/50/137/500 最大差 5.7e-14），差异只在临界值表。用 H₀ 下（均值方差均由样本估计）的蒙特卡洛直接数分位数可以判定：n=500、40 万次重复下 A² 的 95% 分位是 **0.7545**，与模块表的 0.752 吻合，而 scipy 表的 0.781 偏高约 4%（偏保守）。独立 10 万次 MC 复现同样结论（0.7526），且本模块 p 值分段公式与该表自洽：代入 0.752 得 p=0.050、0.631 得 0.097、1.035 得 0.010，实测各名义水平下的经验第一类错误率与名义值相差 ≤0.001。**结论：模块表更接近真实零分布，scipy 表更保守；两者都不算"错"，但论文里必须写清用的是哪一张**——直接把自己手写的结果和 scipy 的输出并排放进同一张表会被答辩问住。

**n < 30 时注意**：真实临界值比表值更低（MC 显示 n=20 的 5% 分位约 0.7205），用表值判断偏保守（更不容易拒绝）；p 值分段近似在 A²≈0.30 的段边界处绝对误差可达 0.03，其余区间 ≤0.01。

**外部库**：`statsmodels/statsmodels`（推断与诊断最权威）、`scikit-learn/scikit-learn`（预测导向的回归/降维）、`dmlc/xgboost`（表格数据强基线）。

### 3.6 评价与决策 —— `examples/algorithms/evaluation.py`

**共同约定**：`benefit` 是一个长度 n 的 bool 序列，`True` 表示正向（越大越好）指标，`False` 表示成本型。**所有函数都会先做正向化再无量纲化**——这是评价类题目最常被扣分的地方。

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `ahp_weights` | 幂法求主特征向量 + 一致性检验 | O(iter·n²) | 必须报 **λmax 与 CR**，CR < 0.1 才可用；判断矩阵要说明来源（专家打分？问卷？），不能凭空给 |
| `entropy_weights` | 熵权法（客观赋权） | O(mn) | 需先无量纲化；`p·ln p` 在 p=0 处要按 0 处理；数据全同的指标熵最大→权重趋 0 |
| `critic_weights` | CRITIC（对比强度 × 冲突性） | O(mn + n²m) | 同时用标准差与相关系数，所以**必须先正向化 + 无量纲化**，否则量纲会污染标准差 |
| `combine_weights` | 博弈论组合赋权 / 乘法合成 / 线性加权 | O(Kn) | 组合权重不是"更客观"；论文要说明为什么组合、组合后是否仍满足一致性 |
| `topsis` | 逼近理想解排序 | O(mn) | **顺序不能错**：先正向化 → 再向量归一化 → 再加权。成本型指标若在归一化之后才取负，排序会错（本实现早期真实踩过）；理想解就是加权矩阵的列最大/最小，不需额外归一化 |
| `vikor` | 折衷排序（群体效用 + 个体遗憾） | O(mn) | 要报 **v**（决策机制系数）与"可接受优势"两个条件检验；只给 Q 排序不检验是不完整的 VIKOR |
| `grey_relational_grade` | 灰色关联分析 | O(mn) | **两个真实坑**：不做无量纲化 → 大数量级指标独裁；不做正向化 → **最差方案得分最高**（本实现早期真实踩过）。`reference` 传**原始量纲**，函数会施加与 X 相同的处理 |
| `dea_ccr` | CCR（规模报酬不变） | O(n_dmu · LP) | 每个 DMU 解一次 LP；**排序要按效率值降序**（早期版本对效率取负再排名，导致最差 DMU 排第一）；投入产出必须都是"越大越好/越小越好"方向明确 |
| `dea_bcc` | BCC（规模报酬可变，技术效率） | 同 CCR | 比 CCR 多一个 Σλ=1 约束——**要作为等式而不是不等式**加入，否则会得到错误的效率值（本实现早期真实踩过，与 `linprog` 对照发现最大偏差 0.807）；CCR/BCC 之比可得规模效率 |
| `fuzzy_comprehensive_eval` | 模糊综合评判 | O(nk) | 权重维数要从 **R 的行数**推出而不是从权重自身推（否则归一化自洽、掩盖错误）；算子 `weighted`（加权平均）vs `max_min`（主因素决定）会给出不同结论，必须说明选了哪个 |
| `topsis_rank_sensitivity` | 权重扰动下的排序稳定性 | O(n_samples·mn) | 论文"灵敏度分析"一节直接用；要报**排序翻转概率**而不是只说"基本稳定" |

**关于 DEA 库**：专门的 DEA Python 库（如 `janditzen/DEApy`）曾核验为 404 不可用，因此本仓库不列。请用成熟优化器（OR-Tools / Pyomo / 本模块的单纯形）自行实现 CCR/BCC，并在论文中说明这是标准模型的标准形式。

**外部库**：`pyMCDM/pymcdm`（多准则决策）、`scikit-fuzzy/scikit-fuzzy`（模糊集）。

### 3.7 聚类与分类 —— `examples/algorithms/clustering.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `kmeans_plusplus_init` | D² 采样初始化 | O(nkd) | 初始化方式必须说明；随机初始化结果不可复现 |
| `kmeans` | Lloyd 迭代 | O(nkd·iter) | **必须先标准化**，否则量纲大的特征独裁；k 要由肘部/轮廓系数定，不能预设 |
| `silhouette_score` | 轮廓系数 | O(n²d) | 需完整距离矩阵，n 大时很慢；单簇或全噪声点的轮廓系数无定义 |
| `elbow_curve` | 肘部曲线（SSE vs k） | O(Σₖ nkd·iter) | 肘部**主观**，要配合轮廓系数一起报 |
| `agglomerative` | 凝聚层次聚类（Lance-Williams） | O(n³) 朴素 | `linkage` 口径（single/complete/average）会显著改变结果，必须报；single 易产生链状簇 |
| `dbscan` | 密度聚类 | O(n²)（无空间索引） | `eps` 用同一把尺子 → **必须先标准化**；`eps`/`min_pts` 要报；结果含噪声标签 -1，统计簇数时别把它算进去 |

**外部库**：`scikit-learn/scikit-learn`（有 KD 树加速与全套评估指标）。

### 3.8 微分方程与机理 —— `examples/algorithms/differential.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `solve_ivp_euler` | 显式（前向）Euler | O(n·m) | **一阶精度**；自检实测收敛阶 ≈ 1.0035。步长稍大就发散，别用于刚性方程 |
| `solve_ivp_rk4` | 经典四阶 Runge-Kutta | O(4nm) | **四阶精度**；自检实测收敛阶 ≈ 4.0693。定步长，长时间积分会累积误差；刚性方程仍不稳 |
| `sir_rhs` / `simulate_sir` | SIR 传染病模型 | O(T/h) | 基本再生数 R₀ = β/γ 要报；**R₀>1 才爆发**；模型假设"均匀混合 + 永久免疫"，论文必须写明 |
| `logistic_growth` | logistic 增长 | O(T/h) | 同时给数值解与解析解对照，这是证明数值方法正确的便宜手段 |
| `lotka_volterra_rhs` | 捕食者-被捕食者 | O(T/h) | 守恒量守恒性是好的自检点；数值格式会引入人工阻尼 |
| `fit_sir_least_squares` | 粗网格 + 逐轮局部细化 | O(rounds·n_grid²·n_steps·m) | **不使用 scipy.optimize**；只是局部最优，必须报搜索范围与细化轮数；参数可辨识性（β/γ 相关性）要讨论 |
| `estimate_convergence_order` | 步长序列估收敛阶 | O(Σ 1/hᵢ) | 这是论文里"数值方法可靠性"的标准证据；阶数要接近理论值 |

**外部库**：`SciML/DifferentialEquations.jl`（Julia，含自适应步长与刚性求解器）、`usnistgov/FiPy`（有限体积）、`FEniCS/dolfinx`（有限元，PDE）。

### 3.9 随机模型与仿真 —— `examples/algorithms/stochastic.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `mc_pi` | 蒙特卡洛估计 π | O(n) | 报告**标准误与 95% 置信区间**，不能只给一个数；自检 CI95 ≈ [3.1398, 3.1463] 含 π |
| `mc_integrate` | 均匀抽样估计定积分 | O(n) | 被积函数必须支持**数组输入**（向量化），逐点循环会很慢；要报标准误 |
| `mm1_metrics` | M/M/1 稳态解析式 | O(1) | 要求 **λ < μ**，ρ = λ/μ < 1 否则队列无稳态；ρ→1 时等待时间发散 |
| `mm1_simulate` | 离散事件仿真 | O(n log n) | 仿真值与解析值对照是标准验证（自检 Wq 解析 0.8000 vs 仿真 0.7695）；要报顾客数与预热期 |
| `markov_steady_state` | 解线性方程组求 π | O(n³) | 用**线性方程组而非幂迭代**，精度到机器精度（自检 max‖πP−π‖ ≈ 5.6e-17）；P 必须行随机 |
| `markov_absorption` | 吸收概率与期望吸收步数 | O(n³) | 瞬态/吸收态划分要说明；要求吸收态 P_ii ≈ 1 |
| `gamblers_ruin` | 赌徒破产蒙特卡洛 | O(n_trials × 步数) | 与解析解对照（自检 0.59835 vs 理论 0.59870）；报重复次数与标准误 |

**仿真类题目的硬要求**：**不要把仿真当证明**。必须给样本量、标准误/置信区间、收敛性诊断（如 running mean 图），以及与解析解或已知特例的对照。只跑一次就下结论等于没做。

**外部库**：`simpx/simpy`（离散事件仿真）、`openmc-dev/openmc`（蒙特卡洛粒子输运）、`mesa/mesa`（ABM）。

### 3.10 几何与空间 —— `examples/algorithms/geometry.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `convex_hull` | Andrew 单调链 | O(n log n) | 返回**原始下标**（逆时针，不含共线中间点），不是坐标本身；全共线时返回两端点 |
| `polygon_area` | 鞋带公式 | O(n) | 取绝对值，与顶点绕向无关；**自交多边形结果无意义** |
| `point_in_polygon` | 射线法（crossing number） | O(n) | 含边界；首尾不必重复；顶点/水平边是经典退化陷阱 |
| `haversine` | 大圆距离 | O(1) | 单位是**公里**，地球半径取 6371 km，论文必须写明取值；小距离下与平面近似差异小但别混用 |
| `idw_interpolate` | 反距离加权 | O(nm) | `power` 要报（常用 p=2）；样本点处会奇异（实现内处理了）；IDW **不外推趋势**，只做平滑 |
| `ordinary_kriging` | 普通克里金 | O(m·n³) | 每个预测点解一次 (n+1) 阶方程，**n 不能大**；样本点处预测值精确回原值、方差为 0（自检偏差 ~1.1e-16）；要报半变异函数模型与参数 |
| `estimate_variogram_params` | 估计基台值与变程 | O(n²) | 未显式指定参数时的默认值，把它写进论文以便复现 |

**外部库**：`pysal/pysal`（空间统计与空间权重）、`Toblerity/Shapely`（几何运算与谓词）。

### 3.11 博弈与网络 —— `examples/algorithms/game.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `zero_sum_value_lp` | 零和博弈 LP | 一次 LP（建表 O(mn)） | **约定 `payoff[i,j]` 是行玩家 i 对列玩家 j 的收益**，列玩家是最小化者；转置或取负会得到相反结论 |
| `nash_support_enumeration` | 支撑集枚举求全部纳什均衡 | O(2ᵐ·2ⁿ·(mn + 解方程)) | **只对小规模可行**；零和请设 `B = -A`；能求出**全部**均衡是它的价值，但也意味着枚举爆炸 |
| `shapley_value` | Shapley 值 | O(n·2ⁿ) | n=20 约 2e7 次、n=25 约 8e8 就太慢；特征函数接受位掩码或联盟迭代两种口径，可混用；自检 Shapley 值求和为 1 |
| `gale_shapley` | 延迟接受稳定匹配 | O(n·m) | 结果**对求婚方最优**；换边求婚会得到不同的稳定匹配（稳定匹配一般不唯一） |
| `replicator_dynamics` | 复制者动态（RK4） | O(T·n²) | **约定 `A[i,j]` 是"我选 i、对手选 j"时我的收益**；演化稳定策略（ESS）与纳什均衡不是一回事，要区分 |

**外部库**：`drvinceknight/Nashpy`（双人矩阵博弈纳什均衡）、`mesa/mesa`（ABM 演化）。

---

## 4. 什么时候该换成熟库

本仓库实现适合：**讲清原理、教学透明、受限环境（不能装包）、小规模数据**。
出现下列任一情况，请换成熟库并把库的版本与许可证写进参考文献：

| 信号 | 该换什么 |
|---|---|
| 决策变量 > 几百个，或需要全局最优 + gap | `google/or-tools`、`Pyomo/pyomo` |
| 样本量 > 几千，需要速度 | `scikit-learn`（聚类/回归/降维） |
| 需要严谨的统计推断与诊断（p 值、稳健标准误、诊断图） | `statsmodels` |
| 需要大数据时序回测与多模型竞赛 | `sktime`、`darts`、`statsforecast` |
| 刚性 ODE、自适应步长、事件检测 | `SciML/DifferentialEquations.jl` |
| 二维以上 PDE / 有限元 / 有限体积 | `FEniCS/dolfinx`、`FiPy` |
| 复杂排队网络、大规模离散事件仿真 | `SimPy` |
| 大图（百万边） | `networkx`（或更专用的图库） |
| 空间统计、空间自相关、空间权重矩阵 | `PySAL` |
| 多准则决策方法全家桶 | `pymcdm` |

**用外部库的纪律**（与 `references/github-resources.md` 一致）：
1. 记录仓库、具体路径、**许可证**、访问日期、tag/release 或 commit SHA；
2. 仓库代码许可 ≠ 数据许可 ≠ 文档许可，分别核对；
3. GPL/AGPL 代码**不能**未经评估并入本仓库的 MIT 发行物；
4. 只借实现，**不借论文结论**；不报告未经核验的 star 数、维护活跃度或性能排名。

---

## 5. 跨模块通用陷阱（最常被扣分的）

1. **量纲未统一**：评价、聚类、插值、赋权几乎都要先正向化 + 无量纲化。不统一量纲，结果基本必错。
2. **随机种子不固定**：元启发式、K-means、蒙特卡洛。不固定 → 不可复现 → 结论不受信任。
3. **只跑一次**：随机算法的单次结果不是结论。要报重复次数与最优值分布。
4. **不报参数**：SA 的 T0/alpha/iters、GA 的种群/代数/交叉率、PSO 的 w/c1/c2、K-means 的 k、DBSCAN 的 eps/min_pts、ARIMA 的阶数。**没报参数的结果等于不可复现的结果。**
5. **把仿真当证明**：蒙特卡洛/离散事件仿真必须给样本量、标准误、收敛性诊断、与解析解对照。
6. **不做灵敏度分析**：评价类题目（权重扰动）和机理类题目（参数扰动）几乎必问。`topsis_rank_sensitivity` 就是为这一节准备的。
7. **用错检验的方向**："不拒绝原假设" ≠ "证明原假设成立"。ADF 尤其容易被写成"证明了序列有单位根"。
8. **不考虑适用范围**：所有模型都有前提（M/M/1 要 ρ<1、Dijkstra 要非负权、Kriging 要空间平稳、DEA 对异常值极敏感）。论文必须有"模型适用范围与失效条件"一段。
9. **拿外部库当"标准答案"却不先对齐约定**：同一算法在不同库里的默认约定不同——`vif` 是否含截距、`ridge` 是否标准化、KS 是否做 Stephens 修正、AD 用哪张临界值表、ADF 的 `autolag` 语义、`pacf` 是否带偏差校正。**先对齐约定再比对数值**，否则会把自己正确的实现"改错"去迎合外部库。本模块在 §3.5 与 §6 里逐条记录了这类差异及判定依据（凡有分歧一律用**独立蒙特卡洛或解析值**做第三方裁判，而不是默认某个库正确）。

---

## 6. 验证记录

本索引与代码的可信度来自**可复核的验证**，而不是自我声明。重现方式见各节命令。

| 验证内容 | 方式 | 结果 |
|---|---|---|
| 全部算法模块的数值自检 | `python examples/run_algorithms.py` | 各模块 `_self_test()` 逐键通过（键数随模块增长） |
| 与黄金值回归比对 | 同上，比对 `examples/algorithms_golden.json` | 递归类型感知比对，`rtol=atol=1e-9` |
| 确定性复跑 | 同上，每模块连跑两次 | 两次结果一致（随机算法走显式种子） |
| 依赖边界 | CI 中 AST 扫描 `examples/algorithms/*.py` | 禁止 scipy/sklearn/pandas/statsmodels/torch/tensorflow/cvxpy/pulp/sympy/matplotlib |
| LP 正确性 | 与 `scipy.optimize.linprog(method="highs")` 随机对照 | 138 个随机 LP，0 处不一致 |
| DEA 正确性 | 与 `linprog` 对照（Σλ=1 必须作等式） | 随机算例最大绝对偏差 ≈ 6.4e-13 |
| 收敛阶 | `estimate_convergence_order` | RK4 ≈ 4.0693（理论 4）、Euler ≈ 1.0035（理论 1） |
| 排队论 | 解析 vs 离散事件仿真 | M/M/1 的 Wq：解析 0.8000 vs 仿真 0.7695（同一量级，差异来自有限样本） |
| 蒙特卡洛 | 置信区间覆盖率 | `mc_pi` 的 95% CI ≈ [3.1398, 3.1463]，包含 π |
| 马尔可夫 | 平稳性残差 | max‖πP − π‖ ≈ 5.6e-17（机器精度） |
| 克里金 | 样本点插值一致性 | 样本处预测偏差 ≈ 1.1e-16，方差为 0 |
| 最大流/最小割 | 两者相等 | 均为 23.0（max-flow min-cut 定理） |
| 最短路 | Dijkstra vs Floyd-Warshall | 最大偏差 0.0 |
| 博弈 | 均衡与分配性质 | 三维性别战 3 个纳什均衡；Shapley 值求和 = 1 |
| ADF 临界值 | 与 `statsmodels.tsa.adfvalues.mackinnoncrit`（MacKinnon 2010 响应面）对照 | 63 组（7 个样本量 × 3 种回归形式 × 3 个显著水平）最大绝对偏差 **8.9e-16**，0 处不一致 |
| ADF 行为 | 随机游走 vs 白噪声 | 随机游走不拒绝单位根、白噪声拒绝（方向正确） |
| 球面距离 | 已知城市对 | 北京→上海 ≈ 1067.31 km |
| 分布函数 | 与 `scipy.special` 对照 | `betainc` 3.9e-16、`gammainc_q` 4.4e-16、`chi2_sf` 3.2e-15、`f_sf` 3.3e-16、`t_sf` 3.8e-13（最大绝对误差） |
| 相关系数 | 与 `scipy.stats` 对照 | `kendall_tau` 无并列与带并列（n=200/25）**逐位相同**；`pearson`/`spearman` 一致 |
| t 检验 / 卡方 | 与 `scipy.stats` 对照 | Welch `df=92.71635504803814` 完全一致；卡方（独立性 + 拟合优度）完全一致 |
| OLS / VIF / logistic | 与 `statsmodels` 对照 | `ols` 系数 1.3e-15、`se` 2.8e-17、`r2` 0.0；`vif` 三个值**逐位相同**；`logistic` 系数 2.2e-16、loglik 差 0.0 |
| Bootstrap / 置换 | 与 `scipy.stats.bootstrap` 同种子对照 | 百分位区间**完全相同**；置换 p 一致到 ~5e-5（2 万次） |
| **AD 临界值** | **H₀ 蒙特卡洛直接数分位数**（均值方差均由样本估计） | n=500、40 万次：A² 的 95% 分位 **0.7545**，与模块表 0.752 吻合；scipy 表 0.781 偏高约 4%（见 §3.5）。A² 统计量本身与 scipy **逐位相同** |
| **AD p 值标定** | 经验第一类错误率 vs 名义水平 | 名义 1%/2.5%/5%/10%/15% 下经验率 0.0098/0.0244/0.0493/0.0994/0.1509，**绝对误差 ≤ 0.001** |
| **KS p 值** | 模块 Stephens 修正 vs scipy 纯渐近 | 同一 D=0.088131259：模块 0.000776280、scipy 0.000790495。差异来自**有意施加的 Stephens 修正**，非错误 |

> **诚实声明**：以上验证证明的是"实现与教科书公式/成熟库一致"，**不证明**这些模型适用于你的具体赛题。模型选择、假设合理性与结论正确性仍需你自己判断并做灵敏度分析。
