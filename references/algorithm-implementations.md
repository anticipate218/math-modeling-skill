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
| 时间序列建模 | `timeseries.py` | `gm11` `arima_fit` `arima_forecast` `arima_order_select` `sarima_fit` `garch11_fit` `kalman_filter_local_level` `ljung_box` | "小样本趋势外推（灰色）、非平稳序列定阶、波动率聚集、状态空间滤波" |
| 机器学习 | `ml.py` | `train_test_split` `confusion_matrix` `classification_metrics` `roc_auc` `kfold_indices` `knn_predict` `decision_tree_fit` `random_forest_fit` `gradient_boosting_fit` `smote` | "要把分类/回归做成可解释模型、样本不平衡、需要交叉验证与特征重要性" |
| 多准则决策 | `multicriteria.py` | `promethee_ii` `electre_i` `electre_iii` `rank_sum_ratio` `rsr_distribution` `borda_count` `copeland_score` `rank_consensus` | "多方法排序结果打架，要出一份稳健名次表" |
| 多目标优化 | `multiobjective.py` | `pareto_dominates` `fast_non_dominated_sort` `crowding_distance` `nsga2` `pareto_front` `hypervolume_2d` | "两个以上目标互相冲突，要交一组 Pareto 方案而不是单一最优" |
| 灵敏度与数据清洗 | `sensitivity.py` | `oat_sensitivity` `elasticity` `morris_screening` `sobol_first_order` `sobol_total_effect` `impute_knn` `detect_outliers_iqr` | "哪些参数最关键、结论稳不稳、数据有缺失或异常值" |
| 空间与物理场 | `spatial.py` | `heat_equation_1d_explicit` `heat_equation_1d_implicit` `poisson_2d_sor` `forest_fire_ca` `traffic_ca_nagel_schreckenberg` `buckingham_pi` `scaling_similarity` | "温度/电势场随空间演化、元胞自动机（林火/交通流）、量纲分析与相似缩放" |

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
| `goal_programming` | 每个目标引一对偏差变量，写成等式 `c_k @ x + d_k^- - d_k^+ = targets[k]`，`d_k^-, d_k^+ >= 0` （二者不会同时为正，否则可以同时减去 min 而改善目标）。 | 时间：单层为一次单纯形迭代，分层时为 O(层数) 次调用… | 目标规划**必须有硬约束或变量上界**：偏差变量永远能让等式成立，没有硬约束时 |
| `scenario_robust_lp` | 先对每个情景单独调 `simplex_lp` 求 `z_s*`（后悔值的基准；任一情景不可行 就无法定义后悔值，直接抛 ValueError）。 | 时间 O((K+1) 次单纯形迭代) / 空间 O((m + K) x (n + 1))。 | 返回值 `objective` 是**最坏情景值**，而内部最小化的是混合目标；β<1 时两者不等… |
| `chance_constrained_lp` | 约束 `P(a_i @ x <= b_i + ξ_i) >= alpha`，`ξ_i ~ N(0, sigma_i^2)` 独立。 | 时间 O(一次单纯形迭代) / 空间 O(m x n)（与 `simplex_lp` 相同）。 | 这只对**单侧**约束成立：`a@x >= b` 形式必须写成 `-a@x <= -b` 再减 z*sigma… |
| `facility_location` | 用 `itertools.combinations` 枚举全部 `C(n_sites, p)` 个设施组合 （按字典序，保证结果确定）。 无容量约束时，每个客户在所选设施中取成本最小者（并列取下标最小者）。 | 时间 O(C(n_sites, p) * n_customers * p)（有容量时多一个排序的 log 因子）… | **组合爆炸**：这是精确枚举，n_sites=30、p=5 时 C(30,5)=142506 还能勉强跑… |

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
| `prim_mst` | 堆优化 Prim：从下标 0 出发，用二叉堆维护"已入树集合的横切边"，每次取最小边并入 新节点并松弛其邻边。 | 时间 O(E log E)（E 为有效边数，等价 O(E log V)）/ 空间 O(V + E)。 | 与 `kruskal_mst` 的输入格式不同（这里是矩阵），但同一张图上两者总权重必须 |
| `degree_centrality` | 一次扫描邻接矩阵求每行权重和（或非零邻居计数），再做标量归一化； 排序是 O(V log V) 的稳定排序。 | 时间 O(V^2)（矩阵运算）/ 空间 O(V^2)（含输入的矩阵副本）。 | 方向被忽略（有向图的出/入度中心性请自己按行/列统计）。 |
| `closeness_centrality` | 以每个节点为源跑一次 Dijkstra，统计**可达**（有限距离）节点的个数与距离和， 中心性 = 可达点数 / 距离和；没有可达点时取 0。 | 时间 O(V * (E log V)) / 空间 O(V + E)。 | **不可达对既不计入分子也不计入分母**（不是"距离记 0"，也不是"记无穷大"）。 |
| `betweenness_centrality` | Brandes 2006：以每个节点为源做 BFS，记录最短路上溯节点与最短路条数，再逆序累加 依赖量；无向图每对点被数了两次，故除以 2，最后除以组合数 `C(n-1, 2)`。 | 时间 O(V * E) / 空间 O(V + E)。 | 只按**无权**（每条边算 1 跳）计最短路；带权图的介数（用边权求最短路）需要另写。 |
| `louvain_communities` | 初始每个节点自成一社区，按随机顺序遍历节点… | 时间 O(max_iter * E) / 空间 O(V + E)。 | 只有**一层**局部移动，没有把社区收缩成超点再迭代，所以在大图上质量低于完整 |
| `min_cost_flow` | SSP（successive shortest path）：每次在残量网络上用 Bellman-Ford 找单位费用最小的 增广路，沿路推进"剩余需求 与 路上最小残量"的瓶颈量… | 时间 O(A * V * E)（A 为增广次数，每次 Bellman-Ford 为 O(VE)）/ 空间 O(V^2)。 | 容量不足时 `raise ValueError`，**不会**静默返回"能满足多少算多少"的部分流… |
| `bipartite_matching` | 最大化收益等价于最小化 `-cost`：把矩阵取负后用 `_hungarian_min` 求小边侧的 完备匹配（行数多于列数时先转置），再还原原始下标。 | 时间 O(min(r, c)^2 * max(r, c)) / 空间 O(r * c)。 | 返回的是**小边侧的完备匹配**（`size = min(rows, cols)`），零收益甚至负收益的 |
| `a_star` | 标准 A*：`f = g + h`，用二叉堆取最小 f；节点出堆时才判定是否扩展（配合"同一节点 允许多次入堆、取最优 g"），遇到 goal 立即返回。 | 时间 O(E log V)（最坏退化为 Dijkstra）/ 空间 O(V + E)。 | 启发式必须**可采纳**（`h <= 真实剩余代价`）；不可采纳时返回的 cost 可能大于真实 |
| `vrp_clarke_wright` | 节约法：先给每个客户一条 `depot -> i -> depot` 的独立路线，计算节约值 `s(i, j) = d(depot, i) + d(depot, j) - d(i, j)`… | 时间 O(n^2 log n)（节约值排序主导）/ 空间 O(n^2)。 | 只做"端点合并"，不做 2-opt / Or-opt 改进，因此结果一般不是最优解（Clarke-Wright |
| `network_robustness` | 每轮在剩余图中用"非零邻接计数"选度最大的节点（同分取 `_weight_matrix` 顺序中 靠前的），标记删除；全部移除结束后用一次 BFS 全源跳数统计最大连通分量与全局效率。 | 时间 O(n_remove * V^2 + V * (V + E)) / 空间 O(V + E)。 | 移除顺序是**确定性贪心**（度最大 + 下标 tie-break），不是随机攻击；随机故障 |

**外部库**：`networkx/networkx`（图算法齐全，适合交叉验证）。

### 3.3 组合优化与元启发式 —— `examples/algorithms/heuristics.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `simulated_annealing` | Metropolis + 几何降温 | O(iters × (T_cost + T_neighbor)) | **约定最小化**，最大化请取负；`T0`/`alpha`/`iters` 三个参数必须写进论文，否则结果不可复现 |
| `genetic_algorithm` | 二进制编码 + 锦标赛 + 单点交叉 + 位翻转 | O(gens × pop × n_genes) | 必须报种群规模、代数、交叉/变异率、选择方式；不报等于没做 |
| `particle_swarm` | 惯性权重线性递减 PSO | O(iters × pop × (dim + T_obj)) | 必须报 `w` 递减区间、`c1`/`c2`、粒子数；边界处理方式会影响结果 |
| `ant_colony_tsp` | Ant System：概率构造 + 信息素挥发 + 加强 | O(iters × n_ants × n²) | 只对**对称** TSP；`alpha`/`beta`/`rho` 与信息素初值都要报 |
| `differential_evolution` | 初始种群在 `[lo, hi]` 内独立均匀采样，逐个体求值… | 时间 O(n_gen * pop_size * (n_dim + T_objective)) / 空间 O(pop_si… | **越界处理会决定成败**：本实现在变异后直接 `clip` 到边界。若最优解落在边界上… |
| `tabu_search` | 令 cur = init，清空禁忌表； 每步枚举 `neighbors_fn(cur)`，跳过非有限候选；候选若在禁忌表中且**不优于** 历史最优则丢弃（藐视准则 aspiration：优于历史最优的禁忌解仍可取用）… | 时间 O(max_iter * (\|N\| * (T_neighbor + T_objective) + \|禁忌表\|))… | **本实现禁忌的是"解"（属性禁忌）而不是"移动"**：对置换类问题，交换 (i,j) 与交换 |
| `grey_wolf_optimizer` | 在 `[lo, hi]` 内均匀初始化 `n_wolves` 匹狼并求值，按目标值升序取 alpha / beta / delta 三匹头狼（GWO 的"等级"结构… | 时间 O(n_gen * n_wolves * n_dim)（外加等量目标函数求值）/ 空间 O(n_wolves *… | **GWO 没有显式的个体记忆**（没有 pbest）：狼的位置可以变差，只有三匹头狼保留了 |
| `variable_neighborhood_search` | 令 cur = init，求值； 每步用 `rng` 打乱邻域顺序，对**每个**邻域各生成 1 个候选并求值… | 时间 O(max_iter * \|neighborhoods\| * (T_perturb + T_objective))… | **本实现是"下降式"VNS：只接受改进解**，因此它**不能**像禁忌搜索那样主动走差解。 |

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
| `acf` | 先中心化，再算 1/n 归一化自协方差 `r_k`，最后 `rho_k = r_k / r_0`。 | 时间 O(n * nlags) / 空间 O(nlags)。 | 归一化口径必须说清楚：本实现用 **1/n**（有偏），k 较大时自相关被系统性压低… |
| `pacf` | 先由 ACF 得到自协方差 `r_0..r_nlags`，再跑 Levinson-Durbin 递推； 第 k 阶递推的最后一个系数 `kappa_k` 就是滞后 k 的偏自相关。 | 时间 O(n * nlags + nlags^2) / 空间 O(nlags)。 | PACF 的截尾/拖尾判读（AR(p) 的 PACF 在 p 阶后截尾）是**渐近**结论；n 小于 |
| `mape` | 逐点算绝对百分比误差再取均值。 | 时间 O(n) / 空间 O(n)。 | **真值中出现 0 时必须报错**：除以 0 会得到 inf，均值变成 inf/NaN，图表全毁。 |
| `rmse` | 误差平方取均值后开方（分母为 n，是总体口径而非 n-1；这是预测误差的通用定义）。 | 时间 O(n) / 空间 O(n)。 | RMSE 对**大误差特别敏感**（平方放大），少数离群点就能主导数值；比较模型时 |
| `mae` | 绝对误差取均值。 | 时间 O(n) / 空间 O(n)。 | MAE 的最优预测是**条件中位数**而 RMSE 的最优预测是条件均值：两条指标同时报告 |
| `naive_forecast` | `"last"`：`f_h = x_{n-1}`； `"mean"`：`f_h = mean(x)`； `"drift"`：`f_h = x_{n-1} + h * (x_{n-1} - x_0) / (n - 1)`… | 时间 O(n + n_ahead) / 空间 O(n_ahead)。 | 朴素方法在基准对比里是**及格线而不是模型**：一个复杂模型若打不赢 |
| `seasonal_decompose` | 令 `seasonal` 初值为 0（加法）或 1（乘法），重复 `n_iter` 轮： `adjusted = x - seasonal`（加法）或 `x / seasonal`（乘法）… | 时间 O(n_iter * n * period) / 空间 O(n)。 | **移动平均趋势本身会被季节项污染**：序列长度不是周期的整数倍、或真正的季节形态 |
| `holt_winters_multiplicative` | 初始化与 `holt_winters(mode="multiplicative")` 完全一致：用前两个完整季节做经典分解 （`base = level0 + trend0 * t`… | 时间 O(n + n_ahead) / 空间 O(n + period)。 | **乘法模式要求数据严格为正**：含 0 或负值时 `y_t / s_{t-m}` 无意义，本实现直接 |

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
| `t_test_one_sample` | `t = (xbar - mu) / (s / sqrt(n))`，`s` 用无偏方差（ddof=1），df = n-1。 | 时间 O(n) / 空间 O(n)。 | 要求样本近似来自**正态总体**。n 很小且数据明显偏斜时，t 检验的名义水平不准… |
| `t_test_two_sample` | Student：`sp^2 = ((nx-1)sx^2 + (ny-1)sy^2)/(nx+ny-2)`， `t = (xbar-ybar)/sqrt(sp^2(1/nx+1/ny))`，df = nx+ny-2。 | 时间 O(nx+ny) / 空间 O(nx+ny)。 | **默认 equal_var=True 是 Student 原版假设**，两组方差差 2 倍以上时（很常见） |
| `pca` | 中心化（必要时再除以列标准差，标准差为 0 的列置 1）得到 Z； 协方差阵 `C = Z'Z / (n-1)`，用 `np.linalg.eigh`（对称阵专用）分解… | 时间 O(n p^2 + p^3) / 空间 O(np + p^2)。 | `standardize` 决定你在相关阵还是协方差阵上工作，两者结果**不可比**… |
| `factor_analysis` | 标准化得到 Z，算相关阵 `R = Z'Z/(n-1)`； 共性方差初值取"该变量与其余变量的最大绝对相关系数"（对角元不足的常用代理）… | 时间 O(max_iter * p^3) / 空间 O(np + p^2)。 | 主因子法不是极大似然：它把 h2 当已知反复代入，收敛到的是"约化相关阵" |
| `lasso_regression` | 先把 y 与 X 各列**中心化**（截距因此不参与惩罚），得到 `yc`、`Xc`… | 时间 O(max_iter * n p) / 空间 O(np)。 | **本实现不做列标准化**（只中心化），因此同一个 alpha 对不同量纲的列 |
| `poisson_regression` | 线性预测量 `eta = Xd b`，均值 `mu = exp(eta)`（eta 截断到 [-50, 50]， mu 下限 1e-10）… | 时间 O(max_iter * n p^2) / 空间 O(np)。 | **过散布（overdispersion）**：真实计数数据的方差常大于均值，Poisson 假设下 |
| `durbin_watson` | 直接按定义算相邻残差差分平方和与残差平方和之比；等价形式 `DW ≈ 2(1 - rho_1)`（rho_1 为一阶样本自相关，大样本下近似）。 | 时间 O(n) / 空间 O(n)。 | 残差必须按**原始观测顺序**传入：排序过的残差会算出接近 2 的 "正常" 值… |
| `breusch_pagan` | 取 `e2 = residual^2`； 用 `ols` 的同一套最小二乘把 e2 对 `[1, X]` 回归，得辅助 `R2`… | 时间 O(n p^2) / 空间 O(np)。 | 这是 **LM（拉格朗日乘数）版本**，不是 Koenker 的学生化版本：它对残差的 |
| `stepwise_selection` | 从空模型（只有截距）出发； 每一轮同时评估所有"加入一个未选变量"与"剔除一个已选变量"的候选模型， 取准则值最低且**严格优于**当前模型的动作执行（这就是"逐步"而非纯前向）… | 时间 O(max_steps * p * n p^2) / 空间 O(np)。 | **逐步回归后的 p 值不可信**：变量是被数据挑出来的，常规 t 检验的名义显著性 |

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
| `gmm_em` | 用 `kmeans` 的中心作为初始均值，簇内样本比例作为初始权重， 簇内协方差（也加 reg_covar）作为初始协方差——比随机初始化稳定得多… | 时间 O(n k d^3 * n_iter)（d^3 来自 Cholesky）/ 空间 O(n k + k d^2)。 | **必须加 reg_covar**：某一分量只剩一个点（或所有点共线）时经验协方差奇异… |
| `spectral_clustering` | 相似度： - `sigma` 给定时 W_ij = exp(-\|\|x_i-x_j\|\|^2 / (2 sigma^2))… | 时间 O(n^3)（稠密特征分解，与 k 无关）/ 空间 O(n^2)。 | **带宽是真正的超参**：太小则相似度矩阵退化成近邻图的单位阵（图不连通… |
| `fuzzy_cmeans` | 初始化：跑一次 `kmeans`，用它的中心作为初始 c_j（比随机中心稳）； 更新隶属度：u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1))，d_ij = \|\|x_i - c_j\|\|… | 时间 O(n k d * n_iter) / 空间 O(n k)。 | **d_ij = 0 会让 u_ij 出现 0/0**：必须显式处理重合点，否则整行变 NaN 并 |
| `davies_bouldin_score` | S_j = 簇 j 内样本到簇心 c_j 的**平均**欧氏距离（散度）； M_jl = \|\|c_j - c_l\|\|（簇心距离）； R_j = max_{l != j} (S_j + S_l) / M_jl… | 时间 O(n d + k^2 d) / 空间 O(n d)。 | 两个簇心完全重合（M_jl = 0）时比值发散：这里按约定返回 `inf`… |
| `calinski_harabasz_score` | BGSS = sum_j n_j \|\|c_j - c_bar\|\|^2（簇间平方和，自由度 k-1）； WGSS = sum_j sum_{i in C_j} \|\|x_i - c_j\|\|^2（簇内平方和，自由度 n-k）… | 时间 O(n d) / 空间 O(n d)。 | k = n 时 WGSS = 0、n-k = 0，公式 0/0：这里返回 0.0，不要当成最优。 |
| `gap_statistic` | 对 k = 1..k_max 在**原数据**上跑 `kmeans`，W_k = sum_j sum_{i in C_j}\|\|x_i-c_j\|\|^2 （用簇内距离**和**，与原论文一致，而不是方差或均值）… | 时间 O(n_refs * k_max * n k d * iters) / 空间 O(n d + n k_max)。 | **参考分布必须与原数据同尺度**：对每一维用观测的 [min, max] 做均匀采样… |

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
| `sir_rhs` | 标准 SIR 常微分方程组（Kermack-McKendrick 1927），此处不做任何离散化。 | 时间 O(1) / 空间 O(1)。 | 分母用当前状态的 S+I+R，如果状态被数值误差推到负值，N 会变小甚至为 0… |
| `simulate_sir` | 拼装 `sir_rhs`，用 `solve_ivp_rk4` 积分，然后 np.argmax 找峰值。 | 时间 O(T/h) / 空间 O(T/h)。 | **峰值是网格上的峰**：h=0.01 时峰值时间有约 ±0.01 的分辨率误差… |
| `seir_rhs` | 标准 SEIR 常微分方程组（Hethcote 2000 综述里的出生-死亡版本），不做任何离散化。 `mu=0` 时 N 严格守恒，是传染病建模最常用的封闭人口口径。 | 时间 O(1) / 空间 O(1)。 | 与 `sir_rhs` 的签名**不同**：`sir_rhs` 是"先给参数、返回闭包"… |
| `simulate_seir` | 把 `[S,E,I,R]` 交给 `seir_rhs` 组成右端项，用 `solve_ivp_rk4`（或 `solve_ivp_euler`）积分，再用 `np.argmax` 找 I 的峰值。 | 时间 O(t_end/dt) / 空间 O(t_end/dt)。 | **峰值是网格上的峰**：dt 决定峰值时间的分辨率，dt=0.1 时不要报"第 38.55 天"。 |
| `basic_reproduction_number` | 用下一代矩阵法对 `seir_rhs` 的感染仓室 [E, I] 求谱半径： K = [[beta*S/N, beta*S/N], [0, 0]] 型矩阵经 V = [[sigma+mu, 0], [-sigma… | 时间 O(1) / 空间 O(1)。 | 两个口径在 sigma 很大时几乎相等，但 sigma 与 gamma 同量级时能差一倍以上… |
| `implicit_euler` | 用显式 Euler 一步 `y_n + h f(t_n, y_n)` 作为预测子（比直接用 y_n 收敛快）… | 时间 O(n * max_iter * m^2)（m 次额外的右端求值 + 一个 m×m 线性方程组）/ 空间 O(n… | **无条件稳定不等于无条件准确**：y'=-20y、h=0.25 时隐式 Euler 有界且收敛… |
| `solve_ivp_rk45` | 7 级 Dormand-Prince 格式：k1..k7，五阶解 `y5 = y + h Σ b5_i k_i`， 嵌入四阶解 `y4 = y + h Σ b4_i k_i`，误差估计 `err = y5 - y4`… | 时间 O(7 * n_steps * m) / 空间 O(n_steps * m)。 | **不做稠密输出**：返回的 t 是非均匀的求解器步点，画图/比较前要用 |
| `jacobian_stability` | 逐分量中心差分 `J[:, j] = (f(y + eps e_j) - f(y - eps e_j)) / (2 eps)`； `np.linalg.eigvals(J)` 求特征值… | 时间 O(m^2)（含 2m 次右端求值）/ 空间 O(m^2)。 | 这里固定取 `t = 0` 求雅可比：**只对自治系统有意义**。非自治系统请把时刻 |
| `logistic_map` | 先迭代 n_transient 次丢弃暂态，再记录 n_steps 个点… | 时间 O(n_steps + n_transient) / 空间 O(n_steps)。 | **对初值和暂态长度敏感**：周期检测用的是"尾巴上逐点重合"的强判据… |

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
| `mmc_metrics` | 记提供负载 a = c*rho = lam/mu… | 时间 O(c) / 空间 O(1)。 | **rho 必须按 c*mu 归一**：写成 lam/mu 在 c>1 时会把利用率放大 c 倍… |
| `mg1_metrics` | Pollaczek-Khinchine 公式：Wq = lam*E[S^2] / (2(1-rho))， 其中 E[S^2] = Var[S] + E[S]^2（用二阶矩而不是方差，是最容易写错的一步）。 | 时间 O(1) / 空间 O(1)。 | 服务时间方差进的是**二阶矩** E[S^2]，写成 Var[S] 会把 Wq 系统性算小 |
| `discrete_event_simulation` | 事件堆按 (时刻, 序号) 排序，只有"到达"与"离开"两类事件； 服务台用空闲堆维护；到达时若无空台则进 FIFO 等待队列， 离开时把等待队列队首放到刚空出的台上（不抢占）… | 时间 O(n log n) / 空间 O(n)。 | arrival_fn 返回的是**间隔**而不是时刻，传成时刻会让所有顾客挤在同一时间到达。 |
| `mmc_simulate` | 把 Exp(1/lam) 的到达间隔与 Exp(1/mu) 的服务时长交给通用事件引擎， 再对结果做统计；时间平均队长由引擎内部按"队列长度 × 时长"积分得到 （PASTA 保证泊松到达下它与"顾客看到的队长"一致）。 | 时间 O(n log n) / 空间 O(n)。 | 单次仿真的随机误差约 O(1/sqrt(n))：c=2、lam=3、mu=2 时 n=2000 的 Wq |
| `metropolis_hastings` | 提议 x' = x + sd * N(0, I)（对称提议，接受比只剩密度比）… | 时间 O((n_samples + burn_in) * d) / 空间 O(n_samples * d)。 | **步长是唯一需要调的参数**：proposal_sd 太小接受率接近 1 但链混合极慢 |
| `gibbs_sampler_bivariate_normal` | 对二元正态，条件分布仍是正态： X1\|X2=x2 ~ N(mu1 + rho*s1/s2*(x2-mu2), s1^2(1-rho^2))， X2\|X1 对称。每次迭代先抽 X1\|X2 再抽 X2\|X1，预热后记录。 | 时间 O(n_samples) / 空间 O(n_samples)。 | **必须真的交替使用最新值**：用同一轮的旧 X2 去抽 X2 会退化成独立采样… |
| `gaussian_copula` | 对 U 每列取平均秩得到 Spearman 相关矩阵 ρ_s，换算成 Pearson 相关 ρ； Cholesky 分解 ρ = LLᵀ，抽 Z = L*N(0, I)，则 Z ~ N(0, ρ)… | 时间 O(n d log n + d^3 + n_draws * d^2) / 空间 O(n_draws * d)。 | **高斯 copula 没有尾部相依**（λ_U = λ_L = 0）：它无法刻画"极端事件同时 |
| `t_copula` | 同 `gaussian_copula` 估计 ρ 并做 Cholesky… | 时间 O(n d log n + d^3 + n_draws * d * max_iter) / 空间 O(n_draw… | λ 只在 ρ>0 时非零：ρ<=0 时 t copula 也没有下尾相依（公式会给出 |
| `Z95` | (常量) | — | — |

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
| `segment_intersection` | 记 r = p2-p1，s = p4-p3，qp = p3-p1，叉积 cross(u,v) = u_x v_y - u_y v_x。 | 时间 O(1) / 空间 O(1)。 | **"相交"必须区分"一个公共点"与"一段公共线段"**：平行且共线的两线段即使 |
| `minimum_enclosing_circle` | 先取前两个点构成直径圆；随后依次加入第 i 个点（i 从 2 到 n-1）。 若第 i 个点在当前圆内则跳过；否则第 i 个点必在新圆边界上 —— 重置为 "以第 i 个点为圆心、半径 0"的退化圆。 | 时间 O(n³) 最坏（本实现不做随机洗牌，因此没有"期望 O(n)"的保证）/ 空间 O(n)。 | **不洗牌 = 放弃期望线性时间**：Welzl 算法的 O(n) 期望复杂度依赖随机化… |
| `polygon_centroid` | 记 cross_i = x_i·y_{i+1} - x_{i+1}·y_i（下标按模 n 回绕）， A = 0.5·Σ cross_i… | 时间 O(n) / 空间 O(n)。 | **顶点平均 ≠ 质心**：把顶点坐标直接求平均只在正多边形等特殊情形下成立… |
| `point_to_segment_distance` | 设 ab = b-a。若 \|ab\|² = 0（线段退化为点）返回 \|p-a\|… | 时间 O(1) / 空间 O(1)。 | **不要漏掉 t 的截断**：直接算 \|(p-a)×ab\|/\|ab\| 得到的是到**直线**的距离… |
| `voronoi_nearest` | 构造 (m, n) 的距离矩阵 D_jk = ‖q_j - p_k‖。 每行取最小值下标。 | 时间 O(mn) / 空间 O(mn)。暴力实现，n、m 都上千时请改用 KD 树或 Delaunay 对偶。 | **Voronoi 胞元只由最近距离定义，不含任何障碍/路网约束**：把站点当设施… |
| `EARTH_RADIUS_KM` | (常量) | — | — |

**外部库**：`pysal/pysal`（空间统计与空间权重）、`Toblerity/Shapely`（几何运算与谓词）。

### 3.11 博弈与网络 —— `examples/algorithms/game.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `zero_sum_value_lp` | 零和博弈 LP | 一次 LP（建表 O(mn)） | **约定 `payoff[i,j]` 是行玩家 i 对列玩家 j 的收益**，列玩家是最小化者；转置或取负会得到相反结论 |
| `nash_support_enumeration` | 支撑集枚举求全部纳什均衡 | O(2ᵐ·2ⁿ·(mn + 解方程)) | **只对小规模可行**；零和请设 `B = -A`；能求出**全部**均衡是它的价值，但也意味着枚举爆炸 |
| `shapley_value` | Shapley 值 | O(n·2ⁿ) | n=20 约 2e7 次、n=25 约 8e8 就太慢；特征函数接受位掩码或联盟迭代两种口径，可混用；自检 Shapley 值求和为 1 |
| `gale_shapley` | 延迟接受稳定匹配 | O(n·m) | 结果**对求婚方最优**；换边求婚会得到不同的稳定匹配（稳定匹配一般不唯一） |
| `replicator_dynamics` | 复制者动态（RK4） | O(T·n²) | **约定 `A[i,j]` 是"我选 i、对手选 j"时我的收益**；演化稳定策略（ESS）与纳什均衡不是一回事，要区分 |
| `minimax_alpha_beta` | 带 (alpha, beta) 窗口的递归极小极大： `depth == 0` 或 `children_fn(node)` 为空 → `evaluate_fn(node)`，叶子计数 +1… | 时间：最坏（子节点顺序最差）O(b^d)，与不剪枝的极小极大相同… | `n_pruned` 是**剪枝事件次数**，不是"省下的求值次数"。被剪掉的子树只有真的展开才知道多大… |
| `stackelberg_lp` | 跟随者问题：给定 x，解 `max_y c_follower @ y s.t. A_follower y <= b_follower - A_leader x… | 时间 O(n_grid^m * 一次 LP)；空间 O(k n + n_grid)（不存储全部网格结果，只留当前最好）。 | **这是网格近似，不是精确解**。领导者收益作为 x 的函数是分片线性的（一般还不凹）… |
| `nash_bargaining_solution` | 离散点集：逐点算纳什积，取最大值（严格大于才替换，并列取输入顺序在前的点，保证确定性）… | 时间：离散 O(N)；多边形 O(k^2)（顶点枚举）+ O(k) 条边的二次求根 + 4 次辅助 LP（有界性检查）… | **纳什解要求可行集里有严格优于 d 的点**。若 d 本身就在帕累托前沿上（没有合作剩余）… |

**外部库**：`drvinceknight/Nashpy`（双人矩阵博弈纳什均衡）、`mesa/mesa`（ABM 演化）。

### 3.12 时间序列进阶（ARIMA/GARCH/卡尔曼/灰色） —— `examples/algorithms/timeseries.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `gm11` | 一次累加 `X1_k = sum_{i<=k} x_i`（k = 0..n-1，0 基下标）； 紧邻均值生成 `z_k = 0.5 (X1_k + X1_{k-1})`（k = 1..n-1）… | 时间 O(n)（含一次 2 列最小二乘，`lstsq` 走 SVD）/ 空间 O(n)。 | **"对纯指数序列精确"是模型形式层面的说法，不是估计层面的**：紧邻均值（梯形） |
| `gm11_posterior_check` | 残差 `e = x - fitted`； `S1 = std(x, ddof=1)`、`S2 = std(e, ddof=1)`（**样本标准差口径**）； `C = S2 / S1`… | 时间 O(n) / 空间 O(n)。 | **口径必须交代**：`S1`/`S2` 用 ddof=1 还是 ddof=0 会改变 `C` 的第四位小数… |
| `difference_series` | 用 `np.diff(x, n=d)` 在**有效段**上直接算 d 阶差分（不做任何边缘填充）， 再左对齐放入长度为 n 的数组并前置 `NaN`。 | 时间 O(d * n) / 空间 O(n)。 | 本函数是**对齐口径**（长度不变、前面补 NaN），与模型内部的缩短口径 |
| `arima_fit` | 做 d 阶差分得 `w`（长度 `m = n - d`），并记录每阶末值 `x_tail`； `w` 中心化后交给 `_arma_cls` 做迭代条件最小二乘… | 时间 O(n_iter * m * (p + q)^2) / 空间 O(m * (p + q))。 | **估计量是 CLS 不是 MLE**：AR(1) 情形下它等于"去掉第一个观测的 OLS"… |
| `arima_forecast` | 递推点预测：`w_hat_h = mu + sum_i phi_i (w_{m+h-i} - mu) + sum_j theta_j e_{m+h-j}`， 未来残差取 0、历史残差不足处取 0… | 时间 O(n_ahead^3 + m*(p+q)) / 空间 O(n_ahead^2)。 | **只支持 `D == 0`**：季节差分模型的反差分未实现（点预测的相位对齐需要保存每一阶 |
| `arima_order_select` | 三重循环遍历 `d = 0..d_max`、`p = 0..p_max`、`q = 0..q_max`，逐个调用 `arima_fit` 收集残差；拟合失败（样本不足、残差方差为 0）的组合被跳过… | 时间 O((p_max+1)(d_max+1)(q_max+1) * 单次拟合) / 空间 O(组合数)。 | **跨 `d` 比较 AIC 理论上不成立**：不同 d 下似然对应不同的数据（差分后序列的 |
| `sarima_fit` | 先做 D 次季节差分 `w = w[s:] - w[:-s]`，再做 d 次普通差分… | 时间 O(n_iter * m * (p+q+P+Q)^2) / 空间 O(m * (p+q+P+Q))。 | **这是简化版**：真正的 SARIMA 季节多项式与 AR/MA 多项式是**乘积**关系 |
| `garch11_fit` | 去均值 `r <- r - mean(r)`，取 `sigma2_0 = var(r, ddof=1)`… | 时间 O(n_starts * n_iter * T) / 空间 O(T)。 | 优化器是**模式搜索**（无导数、收敛慢），不是 BFGS；似然面在 `alpha`/`beta` |
| `garch11_forecast` | 对 GARCH(1,1) 有 `E[sigma2_{T+h} \| F_T] = LR + (alpha+beta)^h (sigma2_T - LR)`… | 时间 O(n_ahead) / 空间 O(n_ahead)。 | 该公式是**条件方差的期望**，不是"波动率的期望"：`E[sigma_{T+h}] <= sqrt(E[sigma2])` |
| `kalman_filter_local_level` | 初值 `x_0 = y[0]`、`P_0 = 0`（即"第一个观测毫无先验不确定性"），随后对每个 t： 预测 `P_pred = P_{t-1} + q`；增益 `K = P_pred/(P_pred + r)`… | 时间 O(n) / 空间 O(n)。 | **初值是"先验"而不是"估计"**：`x_0 = y[0]`、`P_0 = 0` 会让第一个观测的新息 |
| `kalman_filter_linear` | 标准 Kalman 递推（前向）： 预测 `a = F x`、`P = F P F' + Q`； 新息 `v = y_t - H a`、`S = H P H' + R`； 增益 `K = P H' S^{-1}`… | 时间 O(T (k^3 + m k^2 + m^3)) / 空间 O(T (k^2 + m))。 | `S` 用显式求逆（`np.linalg.inv`），数值上不如 Cholesky 稳定；`R` 接近奇异或 |
| `ljung_box` | `x = residual - mean(residual)`； `rho_k = (1/n) sum_{t=k+1..n} x_t x_{t-k} / rho_0`（统一用 1/n，等价于带均值修正）… | 时间 O(L n) / 空间 O(n)。 | **自由度口径**：严谨做法是 `df = L - (已估参数个数)`；本实现取 `df = L`… |


### 3.13 机器学习（树模型/降维/分类器） —— `examples/algorithms/ml.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `train_test_split` | 校验后把样本总数记作 n，算出测试集条数 n_test； 不分层时对 `0..n-1` 整体做一次 `permutation`，前 n_test 个作测试集… | 时间 O(n) / 空间 O(n)。 | 分层抽样下各类的测试条数分别取整，**总数可能与不分层的 n_test 差 ±K**… |
| `standardize_fit` | `mean = X.mean(axis=0)`； `std = X.std(axis=0)`（ddof=0，即除以 n 而不是 n-1）… | 时间 O(n d) / 空间 O(d)。 | 用 ddof=1 的样本标准差会让"标准化后整列标准差恰好为 1"不成立 |
| `standardize_apply` | 逐列做 `(x - mean) / std`，形状靠 numpy 广播对齐。 | 时间 O(n d) / 空间 O(n d)。 | std 里出现 0 时本函数**直接抛 ValueError**而不是悄悄返回 inf/NaN… |
| `confusion_matrix` | 确定类别列表 labels（升序去重）； 把标签映射成 0..K-1 的下标； 用 `np.bincount` 在 `true*K + pred` 上一次性计数，再 reshape 成 K×K。 | 时间 O(n + K^2) / 空间 O(K^2)。 | 当 `labels` 显式给出时，出现在数据里但不在 labels 中的样本会被**静默丢弃**… |
| `classification_metrics` | 以真实与预测标签的并集为类别集合； 对每个类 c：TP = 预测 c 且真实 c，FP = 预测 c，FN = 真实 c… | 时间 O(n K) / 空间 O(K)。 | 宏平均**不给类别加权**：在 1:100 的不平衡数据上，少数类的坏表现会被放大… |
| `roc_auc` | 类别升序排列后，**较大标签记为正类**（这是个需要写进论文的口径）； 对 scores 求平均秩 rank_i（并列取平均）… | 时间 O(n log n) / 空间 O(n)。 | 正类是"较大的那个标签"：若标签是 {1, 2}，2 是正类；若标签是 {-1, 1}… |
| `kfold_indices` | 对 `0..n-1` 做一次 `permutation`； `base = n // k`、`rem = n % k`：前 rem 折每折 base+1 个样本，其余每折 base 个； 逐折切出连续片段作为该折测试集。 | 时间 O(n) / 空间 O(n)。 | n 不能被 k 整除时各折大小相差 1，**不能假设所有折等大**… |
| `stratified_kfold_indices` | 对每个类别，把该类的样本下标独立打乱； 按类内顺序把该类样本轮流分给 k 折（第 j 个样本分到第 j % k 折）； 合并各类在同一折里的下标并升序排序。 | 时间 O(n log n) / 空间 O(n)。 | 类内样本数 m_c 不能被 k 整除时，**不同折拿到该类样本数会差 1**… |
| `knn_predict` | 用展开式一次算出 (n_test, n_train) 欧氏距离矩阵； 每个测试点取距离最小的 k 个训练样本（`argsort(kind="mergesort")` 保证并列时取下标较小的，结果可复现）… | 时间 O(n_test · n_train · d + n_test · n_train log n_train) /… | **未标准化时 kNN 基本失效**：量纲大的特征会独占距离，先 `standardize_fit`。 |
| `decision_tree_fit` | 对当前节点先算出预测值（分类取多数类，平票取最小标签；回归取均值）和不纯度； 若深度到顶、样本数不足、节点已纯（不纯度 0）则成为叶子… | 时间 O(depth · d · n log n)（每个节点对每个特征排序一次）/ 空间 O(n · depth)。 | **贪心且无回看**：XOR 这类"单个特征上不纯度完全不下降"的关系，树会直接停在 |
| `decision_tree_predict` | 对每个样本从根节点开始，按节点自带的 feature/threshold 二分下行， 直到遇到不含 `feature` 键的叶子，取叶子 `value`。 | 时间 O(n · depth) / 空间 O(n)。 | 这是**逐样本 Python 循环**，样本量大时很慢；生产环境请把树展开成数组运算。 |
| `random_forest_fit` | 每棵树用 `gen.integers(0, n, n)` 做一次有放回抽样（自助样本）， 未被抽中的样本就是该树的袋外（OOB）样本； 建树时每个节点从随机特征子集里挑最优切分（Bagging + 随机子空间）… | 时间 O(n_trees · depth · d · n log n) / 空间 O(n_trees · n)。 | 自助抽样下**每棵树大约只用 63.2% 的样本**，其余是袋外；OOB 得分因此天然 |
| `random_forest_predict` | 逐棵树调用 `decision_tree_predict`，把结果堆成 (n_trees, n_samples)… | 时间 O(n_trees · n · depth) / 空间 O(n_trees · n)。 | 分类投票是**硬投票**（每棵树一票），没有用叶子概率做软投票… |
| `random_forest_feature_importance` | 遍历每棵树的所有内部节点，累加 `n_node·impurity - n_left·impurity_left - n_right·impurity_right` 到该节点的切分特征（负贡献截断为 0），最后除以总和。 | 时间 O(n_trees · 节点数) / 空间 O(d)。 | 这是**训练集上的不纯度重要性**，对连续/高基数特征有系统性偏好… |
| `gradient_boosting_fit` | 初始化 `F_0`：回归取 `mean(y)`；分类取 `log(p/(1-p))`，p 为 y 的均值 （截断到 [1e-6, 1-1e-6] 以避免 ±inf）… | 时间 O(n_estimators · depth · d · n log n) / 空间 O(n_estimators… | 这是**函数空间上的最速下降的近似**：每棵树叶子的取值用的是回归树的均值… |
| `gradient_boosting_predict` | `F = init + learning_rate · Σ_t tree_t(X)`；分类再取 `1[F > 0]`。 | 时间 O(n_estimators · n · depth) / 空间 O(n)。 | 分类返回的是**硬标签**，阈值固定 0.5；需要概率请自己调 `_sigmoid` |
| `gaussian_nb_fit` | 类别取 `np.unique(y)` 的升序，先验用频率估计； 对每个类取子集，按列算均值与总体方差； 方差统一加 1e-9，避免某特征在某类上所有取值相同（方差 0）时 对数似然出现除零。 | 时间 O(n d) / 空间 O(K d)。 | **1e-9 是绝对量级平滑，不是相对量级**：若某特征量纲极小（如 1e-6）… |
| `gaussian_nb_predict` | 逐类算对数似然 `log p(x\|c) = -0.5·Σ[log(2π·var) + (x-mean)^2/var]`； 加上 `log(prior)` 得未归一化对数后验… | 时间 O(n K d) / 空间 O(n K)。 | `log_prob` 是**归一化后**的对数后验，不是 sklearn 的 |
| `lda_fit` | 类内散度 `Sw = Σ_c Σ_{i∈c} (x_i-μ_c)(x_i-μ_c)'`， 类间散度 `Sb = Σ_c n_c (μ_c-μ)(μ_c-μ)'`… | 时间 O(n d^2 + d^3) / 空间 O(d^2)。 | `Sw` 奇异（样本数少于特征数，或某特征在类内完全不变）时白化会放大噪声方向… |
| `lda_transform` | 先减去训练时的总体均值，再右乘判别方向矩阵（线性投影）。 | 时间 O(n d c) / 空间 O(n c)。 | **必须平移**：直接 `X @ scalings` 会保留全局均值，投影后的类质心位置 |
| `permutation_importance` | 算基准得分 `base = metric_fn(y, predict_fn(X))`… | 时间 O(n_features · n_repeats · (预测开销)) / 空间 O(n d)。 | 只打乱**一列**时，与该列强相关的其他列仍在，模型仍能部分恢复信息… |
| `smote` | 统计各类样本数 `n_c`，多数类为 `n_max`； 对每个 `n_c < n_max` 的类，在其自身样本间算欧氏距离， 取 `k' = min(k, n_c-1)` 个最近邻… | 时间 O(Σ_c n_c^2 d + 合成数·d) / 空间 O(Σ_c n_c^2 + n_total d)。 | **必须在划分训练/测试之后做**：先 SMOTE 再划分会让同一少数类样本的近邻 |
| `class_weight_balanced` | 统计每类样本数 n_c 与类别数 K； `w_c = n / (K · n_c)`。 | 时间 O(n) / 空间 O(K)。 | 这个口径下权重之**和不为 1**（等于 `(1/K)Σ n/n_c`）… |


### 3.14 多准则决策（PROMETHEE/ELECTRE/RSR/共识） —— `examples/algorithms/multicriteria.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `promethee_ii` | 正向化 `Z = X`（成本型列取负），使所有指标都变成"越大越好"。 | 时间 O(m² n) / 空间 O(m² + mn)。 | `p` 的默认值是该指标的极差，**带量纲**：若直接把不同量纲的指标混在一起且不显式 |
| `electre_i` | 正向化 `Z`（成本型取负），再做向量归一化 `r_ij = z_ij / sqrt(Σ_i z_ij²)`。 | 时间 O(m² n) / 空间 O(m² + mn)。 | 不一致性用**逐指标极差**做分母：某个指标若对所有方案几乎相同（极差≈0），它一旦 |
| `electre_iii` | 正向化 Z（成本型取负），对每对 (a, b) 算逐指标劣势 `d_j = Z[b,j] - Z[a,j]`。 | 时间 O(m³ + m² n) / 空间 O(m² + mn)。 | **λ 的取法没有唯一标准**：Roy 原文里 λ 由决策者给定，这里为了接口自洽取非对角 |
| `rank_sum_ratio` | 逐指标编秩：正向指标中最大值得秩 1、成本型指标中最小值得秩 1（并列取平均秩）。 设原始秩阵为 R（1 为最优），令经典秩 `R' = m + 1 - R`（此时 R' 是"高优指标 秩次大"的教材口径）。 | 时间 O(mn log m) / 空间 O(mn)。 | 秩和比只用到**序信息**，指标的量级差异被完全丢弃：两个方案在某指标上差 0.001 与 |
| `rsr_distribution` | 把 rsr 升序排序，第 pos 位（0 起）的中位秩累计频率取 `p = (pos + 1 - 0.5)/m`… | 时间 O(m log m) / 空间 O(m)。 | 分档切点用的是**回归拟合值**而不是原始 probit：这样即使某个方案的 probit 落在极端 |
| `borda_count` | 校验每份排名是互不重复的非负整数下标（重复下标无定义，直接报错）。 对每份排名，第 p 位方案加 `L - 1 - p` 分（L 为该排名的长度）；未出现的方案得 0 分。 累加总分，按降序编秩。 | 时间 O(Σ L) / 空间 O(m + Σ L)。 | 截断选票的口径不唯一：这里按"该排名自身的长度"给分（第 1 名得 L-1 分）… |
| `copeland_score` | 校验形状方阵、取值只含 {-1, 0, 1}、对角线为 0、反对称。 `score_i = Σ_j W[i, j]`（因为平局贡献 0，求和恰好等于胜场数减负场数）。 按得分降序编秩。 | 时间 O(m²) / 空间 O(m²)。 | 反对称是硬要求：很多"胜场矩阵"只有 0/1（不分平局与负），传进来会被拒绝——那种矩阵 |
| `rank_consensus` | 解析每份排名为下标列表（同一份排名内不允许重复下标）。 对每一对排名，取**共同出现**的方案，各自在原文里的位次（1 为最好）作为秩向量， 用秩向量的 Pearson 相关算 Spearman（并列名次不影响正确性）。 | 时间 O(K² m) / 空间 O(K² + Km)。 | 共同方案少于 2 个时直接抛 ValueError，而不是返回 0：两段几乎不相交的排名之间 |


### 3.15 多目标优化（Pareto/NSGA-II/MOEA-D） —— `examples/algorithms/multiobjective.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `pareto_dominates` | 用 `as_vector` 校验并拉平两个向量，要求长度一致； `dominates = np.all(f1 <= f2) and np.any(f1 < f2)`。 | 时间 O(m) / 空间 O(m)。 | 本函数**内建"全部最小化"的口径**。如果你要最大化某个目标，必须先取负再传进来… |
| `fast_non_dominated_sort` | 对每一对 (i, j) 调用支配判定，统计 `n_dom[j]`（支配 j 的解个数）并记录 `dominates[i]`（i 支配的解集合）； 所有 `n_dom == 0` 的解构成第 0 层前沿… | 时间 O(n^2 * m) / 空间 O(n^2)（最坏情况下的支配关系表）。 | 本实现返回的 `rank` 是 **0 基**的；有些教材把它记成 1 基的"秩"… |
| `crowding_distance` | 对每个目标 k，用 `argsort` 排序，边界点记 `inf`； 内部点累加 `(f[order[i+1], k] - f[order[i-1], k]) / (f_max - f_min)`… | 时间 O(m * n log n) / 空间 O(n)。 | **必须按前沿分别调用**。把整份种群的目标矩阵直接丢进来会在不同前沿之间 |
| `weighted_sum_pareto` | 采样一次 `objective_fn` 得到目标个数 m，用 `_simplex_grid` 生成权重网格… | 时间 O(n_grid * n_starts * n_iter * (m + T_objective))… | **加权和法只能取到凸前沿**。当前沿非凸（例如 ZDT 系列、或两个目标都是 |
| `epsilon_constraint_pareto` | 随机采样估计第 1 个目标在盒约束下的取值范围，生成 n_grid 个 ε… | 时间 O(n_grid * n_starts * n_iter * (m + T_objective) + n_prob… | 罚函数法会给出**近似可行**的解：罚系数 P 太小会停在不可行侧（`infeasible` |
| `nsga2` | 在盒约束内均匀随机初始化 `pop_size` 个个体，求目标值… | 时间 O(n_gen * (pop_size^2 * m + pop_size * m log pop_size + p… | `history` 记录的是**第一前沿规模**，它在这个实现里几乎总是单调不减… |
| `pareto_front` | 直接调用 `fast_non_dominated_sort` 并取 `fronts[0]`，保证两条代码路径 的口径完全一致（自测里会把两者对拍）。 | 时间 O(n^2 * m) / 空间 O(n^2)。 | 返回的下标是**原始矩阵中的位置**，不是排序后的位置；如果你先把 F 排序再调用… |
| `hypervolume_2d` | 丢掉 `F[i, 0] >= r[0]` 或 `F[i, 1] >= r[1]` 的点； 剩下的点按第 1 个目标升序排序… | 时间 O(n log n) / 空间 O(n)。 | **参考点必须固定**才能跨代比较超体积。换参考点会让超体积数值失去可比性… |
| `ideal_point_distance` | `ideal_j = min_i F[i, j]`，`anti_j = max_i F[i, j]`， `d_plus_i = sqrt(sum_j w_j (F[i,j] - ideal_j)^2)`… | 时间 O(n m) / 空间 O(n)。 | 理想点由**当前解集自身**决定，所以加入或删除一个解就会改变所有解的贴近度。 |


### 3.16 灵敏度分析与缺失数据 —— `examples/algorithms/sensitivity.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `oat_sensitivity` | 对第 i 个参数，步长 `step_i = rel_step * \|x0_i\|`；若 x0_i == 0 则取 `step_i = rel_step`（否则 low == high，斜率无定义）。 | 时间 O(d · C_fn) / 空间 O(d)。 | OAT 在**非可加**模型上会误导：f = x0·x1 时两个参数的斜率都依赖另一个参数的 |
| `elasticity` | `h = rel_step * \|x0_i\|`（x0_i == 0 时弹性本身就是 0，见陷阱）。 中心差分 `g = (f(x + h e_i) - f(x - h e_i)) / (2h)`。 | 时间 O(C_fn) / 空间 O(d)。 | `x0_i == 0` 时弹性恒为 0（乘了 x_i），但此时相对扰动无意义，本实现直接返回 0 |
| `morris_screening` | 把参数归一化到 [0,1]：`u_i = (x_i - lo_i)/(hi_i - lo_i)`；网格步长 `step = 1/(p-1)`，Morris 步长 `delta = p/(2(p-1))`。 | 时间 O(n_trajectories · (d + · C_fn) / 空间 O(n_trajectories · d… | `n_trajectories == 1` 时样本标准差无定义：本实现返回 0 而不是 NaN… |
| `sobol_first_order` | 生成独立样本矩阵 A、B（各 N 行），`AB_i` 为 A 的第 i 列换成 B 的第 i 列。 记 `V = Var(y)`（用 A∪B 合并样本的无偏估计）。 | 时间 O(N·d·C_fn) / 空间 O(Nd)。 | N 太小时 S1 可能为负（估计量有偏噪声），小负值**不代表**参数有害… |
| `sobol_total_effect` | 与 `sobol_first_order` 共用同一套 A/B/AB_i 设计（同一 seed 下抽样完全一致）。 | 时间 O(N·d·C_fn) / 空间 O(Nd)。 | 可加模型上应当有 ST_i ≈ S1_i；若差得远，说明 N 太小或 fn 里藏了交互。 |
| `impute_mean` | 对每列用 `nanmean`（忽略 nan）得到列均值。 用该均值填回该列所有 nan 位置。 | 时间 O(mn) / 空间 O(mn)。 | 均值插补会**压缩方差**（填进去的都是列中心），后续做回归/聚类时会让变量显得 |
| `impute_knn` | 对每个缺失位置 (i, j)： 候选集为第 j 列**已观测**的所有行 c（c != i）… | 时间 O(n_missing · n · d) / 空间 O(n · d)。 | 只在共同观测维度上比距离是**稀疏数据的关键**：若改用全维度并把 nan 当 0… |
| `impute_regression` | 初始化：所有 nan 用所在列的观测均值填充。 每一轮：对每个含缺失的列 j，取该列已观测的行，构造设计矩阵 `[1, 其余列]`，用 `np.linalg.lstsq` 解 OLS，再对缺失行预测并回填。 | 时间 O(max_iter · m · n²) / 空间 O(mn)。 | 这是**确定性迭代**而非多重插补：它给出单点估计，无法反映插补本身的不确定性… |
| `detect_outliers_zscore` | 标准化 `z = (x - mean) / std`（标准差用**总体**口径 ddof=0），取 \|z\| > threshold。 | 时间 O(n) / 空间 O(n)。 | 均值和标准差本身被离群点污染：单个极端值会把 std 抬高，使自己的 \|z\| 被压缩。 |
| `detect_outliers_iqr` | 分位数用 numpy 的线性插值口径（`np.percentile` 默认）， IQR = Q3 - Q1，栅栏外即为异常。 | 时间 O(n log n)（排序取分位数）/ 空间 O(n)。 | IQR 对**偏态**分布仍会误报：右偏数据的上栅栏往往被压得过低… |
| `detect_outliers_mad` | `med = median(x)`，`MAD = median(\|x - med\|)`… | 时间 O(n log n) / 空间 O(n)。 | MAD 用中位数而非均值，抗污染能力强，但**对 n 很敏感**：n 小于约 10 时 MAD |


### 3.17 空间与物理场建模（热传导/Poisson/元胞自动机/量纲） —— `examples/algorithms/spatial.py`

| 函数 | 算法 | 复杂度 | 关键陷阱 |
|---|---|---|---|
| `heat_equation_1d_explicit` | 网格点 i = 0..n-1 等间距，内部点用二阶中心差分： u_i^{k+1} = u_i^k + r (u_{i-1}^k - 2 u_i^k + u_{i+1}^k)，r = alpha*dt/dx^2。 | 时间 O(n_steps * n) / 空间 O(n)。 | **r > 0.5 必然发散**，而且前几步看起来还正常，等发现时已经溢出；本函数直接报错 |
| `heat_equation_1d_implicit` | Crank-Nicolson（时间二阶、空间二阶）：对内部点 (1 + r) u_i^{k+1} - (r/2)(u_{i-1}^{k+1} + u_{i+1}^{k+1}) = (1 - r) u_i^k + (r/2… | 时间 O(n_steps * n^3)（每步一次稠密解方程… | 无条件稳定**不等于**无条件精确：dt 很大时时间精度退化为 O(1)，解会"过度平滑" |
| `poisson_2d_sor` | 五点差分：-(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_ij)/h^2 = f_ij， 整理成 u_ij = (四邻域和 + h^2 f_ij)/4。 | 时间 O(n_iter * n^2) / 空间 O(n^2)。 | **ω 不是越大越快**：ω 超过最优值后收敛变慢甚至发散；最优值随网格变细趋近 2… |
| `forest_fire_ca` | 状态 0 = 空地、1 = 树、2 = 燃烧。每一步**同步**执行： 所有燃烧格变为空地（燃烧只持续一步）。 树格若 8 邻域（或 4 邻域）内存在燃烧格，则被点燃；否则以 p_light 的概率被雷击点燃。 | 时间 O(n_steps * n^2) / 空间 O(n^2)。 | **必须同步更新**：如果就地更新（先烧的格子立刻去点燃邻居），火势会在一"步"内 |
| `traffic_ca_nagel_schreckenberg` | 每步对全部车辆**同步**执行 NaSch 四规则（顺序不能换）： 加速：v <- min(v + 1, v_max)； 减速：v <- min(v, gap)，gap 为到前车的空格数… | 时间 O(n_steps * n_cars) / 空间 O(n_cars)。 | **规则顺序不能变**：先随机慢化再加速会得到完全不同的基本图（"慢化"变成了 |
| `buckingham_pi` | 把 dims 组装成量纲矩阵 D（行 = 基本量纲，列 = 物理变量）。 | 时间 O(k * n^2)（k 为基本量纲数，n 为变量数，精确有理数运算） / 空间 O(k * n)。 | **基本量纲必须线性无关且完整**：若把 M 和"力"同时当基本量纲（力本身 = MLT^-2）… |
| `scaling_similarity` | factor_v = λ^{e_v}（量纲幂次换算：长度量纲按 λ、面积按 λ^2、速度按 λ^{1/2} …）。 scaled_v = measurements_v * factor_v。 | 时间 O(k) / 空间 O(k)（k 为物理量个数）。 | **相似比的方向**：λ 定义为"原型 / 模型"。若误用"模型 / 原型"，所有因子会整体取 |


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
