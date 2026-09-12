# GitHub 模型资源索引（已核验）

> **核验范围**：以下条目来自 GitHub 仓库主页或仓库 README 的一手信息。这里记录“仓库明确提供什么”，不虚构 stars、性能排名或维护活跃度。访问时应重新检查 commit、版本、许可证和依赖。
>
> **许可原则**：本索引只提供链接和用途说明，不复制第三方代码。仓库许可证、示例代码许可证、数据集许可证可能不同，使用前逐项核对。

## 1. 优化、规划、调度、路由

| 仓库 | README/项目明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [google/or-tools](https://github.com/google/or-tools) | CP-SAT/CP、Glop/PDLP 线性规划、MPSolver/MathOpt、背包/装箱/集合覆盖、TSP/VRP、最短路、流与指派；C++ 核心并提供 Python/C#/Java 接口 | 生产计划、排班、配送、选址、组合优化 | CP-SAT 主要面向整数建模；连续非线性要另选工具。锁定版本，商用 solver 另看授权 |
| [Pyomo/pyomo](https://github.com/Pyomo/pyomo) | Python 代数建模，覆盖 LP/QP/NLP/MILP/MIQP/MINLP、随机规划、广义析取、DAE、MPEC | 需要把文字约束写成可审计模型、切换 solver、做敏感性 | Pyomo 通常不自带 solver；求解器安装、结果和许可证要单独记录 |
| [Eurus-Holmes/Mathematical_Modeling](https://github.com/Eurus-Holmes/Mathematical_Modeling) | 目录包含线性/整数/非线性规划、网络、插值拟合、微分方程、统计、预测、现代优化、综合评价等 MATLAB 示例 | 备赛时按算法族找案例 | 资料型仓库；示例假设和代码质量要逐例复核，不能直接当通用软件 |

## 2. 时间序列、预测与统计

| 仓库 | README/用户指南明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [sktime/sktime](https://github.com/sktime/sktime) | forecasting、时间序列分类/回归、聚类、异常/变点、转换、距离/核和时间序列 splitters；统一 sklearn 风格接口 | 统一 backtesting、调参、预测与时间序列任务基线 | 部分 detection/clustering/proba 功能仍可能处于 maturing/experimental；软依赖需另装 |
| [unit8co/darts](https://github.com/unit8co/darts) | 朴素/漂移、ARIMA、VARIMA、ETS、Theta、Prophet、Kalman、TBATS、Croston，以及 RNN/LSTM/GRU、N-BEATS/N-HiTS、TCN、Transformer、TFT、XGBoost 等；支持 backtest、概率预测、协变量和层级 reconciliation | 快速横向比较经典、机器学习和深度预测模型 | 深度模型依赖 PyTorch；窗口、GPU、随机种子都会影响结果；不要把仓库 benchmark 当普遍保证 |
| [Nixtla/statsforecast](https://github.com/Nixtla/statsforecast) | AutoARIMA、AutoETS、AutoTheta、AutoTBATS、MSTL、GARCH/ARCH、朴素法、Holt-Winters、Croston/ADIDA/IMAPA/TSB 等，并支持区间和 cross-validation | 大量单变量序列、季节性、间歇需求 | 主要是单变量统计模型；正确设置频率和 season_length；速度比较只适用于其 benchmark |
| [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels) | OLS/GLS/WLS/Quantile、GLM/GEE、Logit/Probit/Poisson/NegBin、MixedLM、稳健回归、状态空间、SARIMA/VAR/VECM、检验、MICE、PCA 等 | 需要参数解释、置信区间、残差诊断和统计检验 | 不要用同一数据既定阶又评估；先查平稳性、缺失、季节性和误差结构 |

## 3. 回归、分类与机器学习

| 仓库 | README/API 明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | 用户指南/API 提供线性/岭/Lasso/ElasticNet、逻辑回归、树/随机森林/ExtraTrees/梯度提升、SVM、kNN、朴素 Bayes、聚类、降维、pipeline、model_selection 与 metrics | 竞赛中的可解释基线、统一预处理和交叉验证 | 普通 K-fold 不适合时间序列；按时间/组切分，防止泄漏。许可证见 [COPYING](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| [dmlc/xgboost](https://github.com/dmlc/xgboost) | gradient boosting/parallel tree boosting，支持多语言接口和分布式环境 | 表格数据非线性、特征交互、强基线 | 深度、学习率、早停、类别编码和时间/组泄漏需要单独验证；GPU/分布式安装复杂 |

## 4. ODE、PDE、物理与动力学

| 仓库 | README 明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [SciML/DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) | ODE、SDE、DDE、DAE 等微分方程及科学机器学习组件 | 疫情、种群、振动、随机动力学、参数扫描 | Julia/SciML 生态较大；竞赛需固定 Julia 与包版本，许可证按仓库 LICENSE 核对 |
| [usnistgov/FiPy](https://github.com/usnistgov/fipy) | Python 有限体积 PDE 求解器，含扩散、对流、相变等示例/方程 | 热传导、扩散、反应、流动原型 | 网格、边界条件、时间步长和稳定性必须自行验证 |
| [FEniCS/dolfinx](https://github.com/FEniCS/dolfinx) | Python/C++ 有限元 PDE 建模、变分形式、网格与边界值问题 | 力学、热、流体高保真模型 | 安装、网格和版本成本高，不宜临赛首次上手 |

## 5. 仿真、概率与随机过程

| 仓库 | README/项目明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [simpx/simpy](https://github.com/simpx/simpy) | Python 离散事件仿真：资源、过程、事件和队列/服务系统 | M/M/1、多服务台、库存、交通 | 这是 fork；优先核对上游、LICENSE、版本和维护情况；热身期、重复实验、CI 需自行设计 |
| [salabim/estebanangelm](https://github.com/estebanangelm/salabim) | Python 离散事件仿真，流程/资源/队列和可视化 | 复杂服务流程、生产系统 | 核对具体分支版本和许可证；不要把可视化结果当统计验证 |
| [openmc-dev/openmc](https://github.com/openmc-dev/openmc) | C++/Python Monte Carlo 中子/光子输运、几何、材料、核数据、临界/燃耗 | 物理空间与随机输运题 | 不是通用排队库；核数据与计算成本高，许可证/数据授权需分别核对 |

## 6. 图论、空间与几何

| 仓库 | README/项目明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [networkx/networkx](https://github.com/networkx/networkx) | Python 图构造/操作、连通、中心性、最短路、流、社区、图谱算法和生成器；README 明确 3-clause BSD | 路网、传播、任务依赖、关键节点 | 大图性能有限；必要时与 igraph/稀疏算法比较 |
| [PySAL/pysal](https://github.com/pysal/pysal) | 空间权重、空间自相关、区域化、网络/空间计量等生态 | 地理网络、空间相关和区域规划 | 生态庞大，按子包核对安装、API 和许可证 |
| [Toblerity/Shapely](https://github.com/shapely/shapely) | Python 几何对象与 GEOS 计算几何，含点线面、缓冲、相交/并集、距离、仿射变换 | 覆盖、选址、空间约束、GIS 预处理 | 平面几何而非球面几何；坐标系和数值鲁棒性要明确。版本/许可证看仓库页 |

## 7. 评价、模糊与博弈/ABM

| 仓库 | README/项目明确内容 | 适合题目 | 注意 |
|---|---|---|---|
| [pyMCDM/pymcdm](https://github.com/pyMCDM/pymcdm) | Python 多指标决策工具，包含 TOPSIS、多种归一化、权重和排序组件 | 方案排序、权重敏感性 | 方法实现不等于指标合理；检查正负向、归一化、权重来源与秩反转 |
| [scikit-fuzzy/scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) | SciPy 的模糊逻辑工具包：模糊集合/隶属函数、规则系统、控制器和聚类 | 模糊综合评价、规则控制 | 隶属函数和规则有主观性；模糊分数不是统计置信度 |
| [mesa/mesa](https://github.com/mesa/mesa) | Python ABM：Agent、Model、调度/激活、空间网格/网络空间、数据收集、可视化、批量运行和示例模型 | 扩散、拥堵、生态、市场主体仿真 | README 标明 Mesa 4 active development、Mesa 3 stable；竞赛固定 Mesa 3 更稳，种子/调度/校准必须记录 |
| [drvinceknight/Nashpy](https://github.com/drvinceknight/Nashpy) | Python 双人矩阵博弈、纳什均衡计算、枚举/支撑集等算法 | 双人竞争策略、零和/非零和题 | 核心是双人有限矩阵，不能替代大规模多智能体模型 |

## 8. 中文竞赛资料索引

- [zhanwen/MathModel](https://github.com/zhanwen/MathModel)：README 列出 2004–2025 竞赛题、优秀论文和算法目录（仿真、蒙特卡洛、马尔可夫链、PSO、GA 等），适合查题型、论文表达和验证思路。
- [HuangCongQing/Algorithms_MathModels](https://github.com/HuangCongQing/Algorithms_MathModels)：README 列出 MATLAB 的规划、图论、灰色系统、神经网络、插值、回归、时间序列等算法目录。

这些资料库不是经过统一测试的软件包，部分内容来自互联网整理；论文、代码和数据的许可证可能不同，应逐文件核验。不要把获奖论文结果当成某个算法在新题上的有效性证明。

## 9. 资源选择和引用记录模板

```text
仓库：owner/name
URL：固定仓库页 + 具体路径
用途：对应题目类别和使用的 API/示例
许可证：仓库 LICENSE / 示例 / 数据分别记录
版本：tag/release 或 commit SHA
访问日期：YYYY-MM-DD
本地验证：输入、参数、随机种子、运行环境、输出摘要
独立检查：基线、消融、敏感性、外部/滚动测试
```

## 10. 明确排除

- `janditzen/DEApy` 搜索结果对应 URL 已返回 404，因此不列为可核验 DEA 专用资源。
- 不报告未经核验的 stars、下载量、准确率或“最强算法”排名。
- 不把仓库 README 中的 benchmark、示例结果或作者宣传语改写成普遍结论。
