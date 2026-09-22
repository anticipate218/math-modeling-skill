"""数学建模常用算法的透明实现（仅依赖 Python 标准库与 numpy）。

设计目标
--------
1. **可读优先**：这些实现是为了让人看懂模型内部在做什么，而不是追求工业级性能。
   生产环境中应当优先使用成熟库（见 ``references/algorithm-implementations.md``）。
2. **可复现**：所有随机过程都接受 ``seed``，默认使用 ``DEFAULT_SEED``。
3. **可自测**：每个模块提供 ``_self_test()``，由 ``examples/run_algorithms.py`` 统一调用并断言。

为什么不直接调用现成库
----------------------
论文里写 "我们调用 sklearn 的 KMeans"，评委无法判断你是否理解算法。
本目录给出**自己实现**的版本用于教学、对照和消融实验；正式解题时可以用成熟库的结果
与这里的实现相互印证（两者不一致往往意味着预处理或参数理解有偏差）。

模块清单
--------
共 17 个算法模块（按"解题时的问题类型"排序）：

- ``_common``         公共工具（校验、归一化、随机数、加权求和）
- ``optimization``    线性/整数规划（单纯形、内点法、对偶与 RHS 灵敏度）、分支定界、
                      指派、运输、背包、目标规划、鲁棒 LP、机会约束 LP、离散选址
- ``evaluation``      层次分析、熵权、CRITIC、组合赋权、TOPSIS（含排序稳健性）、VIKOR、
                      灰关联、秩和比（RSR）与概率单位分档、PROMETHEE II、DEA（CCR/BCC）、
                      模糊综合评价、Kendall 协调系数
- ``forecasting``     移动平均、指数平滑、Holt 线性/季节、AR 最小二乘、
                      ACF/PACF/ADF（含趋势与漂移型临界值）、误差指标、滚动原点回测、季节分解
- ``timeseries``      GM(1,1) 与后验差检验、ARIMA/SARIMA、GARCH(1,1)、
                      卡尔曼滤波（局部水平与一般线性）与卡尔曼平滑、Ljung-Box 白噪声检验
- ``statistics``      最小二乘、岭回归、lasso、logistic、Poisson、PCA、因子分析、
                      Bootstrap（含 BCa）、置换检验、非参数检验（Mann-Whitney/Wilcoxon/
                      Kruskal-Wallis/单因素方差分析）、HAC(Newey-West) 标准误、
                      卡方、逐步回归、异方差/自相关诊断
- ``ml``              数据划分与分层抽样、标准化、混淆矩阵/宏平均指标/ROC-AUC/PR 曲线与
                      平均精度、kNN、CART（Gini 或熵）、随机森林、梯度提升、高斯朴素贝叶斯、
                      LDA、PCA（拟合/变换）、置换重要性、SMOTE、类别权重
- ``clustering``      K-means++、K-medoids、层次聚类、DBSCAN、GMM-EM、谱聚类、模糊 C 均值、
                      轮廓系数、DB/CH 指数、肘部曲线、Gap 统计量
- ``multicriteria``   PROMETHEE II、ELECTRE I/II/III、秩和比（RSR）与概率单位分档、
                      Borda、Copeland、排序一致性
- ``multiobjective``  支配关系、快速非支配排序、拥挤距离、加权和法、ε 约束法、
                      NSGA-II、MOEA/D、二维超体积、理想点距离、IGD、Spacing、拐点识别
- ``sensitivity``     单因素（OAT）敏感性与弹性、Morris 筛选、Sobol 一阶/总效应/二阶、
                      缺失值插补（均值/kNN/回归/MICE）、离群点检测（Z/IQR/MAD）
- ``differential``    Euler/RK4/RK45、SIR/SEIR、R0、隐式欧拉、logistic 增长与混沌、
                      Lotka-Volterra、参数拟合、平衡点稳定性、Euler-Maruyama（SDE）
- ``stochastic``      蒙特卡洛积分与置信区间、M/M/1 与 M/M/c、M/M/c/K、M/G/1 排队、
                      离散事件仿真、马尔可夫稳态与吸收、随机游走、MCMC、Copula、
                      几何布朗运动
- ``graphs``          Dijkstra、Bellman-Ford、Floyd、Prim/Kruskal、最大流、最小费用流、
                      二分图匹配、A*、VRP、PageRank、中心性、社区发现、网络鲁棒性、
                      拓扑排序与关键路径
- ``game``            零和博弈 LP（行/列双方口径）、双矩阵纳什、Shapley 值、
                      Gale-Shapley 稳定匹配、复制子动态、α-β 剪枝、Stackelberg 主从博弈、
                      纳什谈判解、迭代剔除、演化稳定策略（ESS）、相关均衡 LP
- ``heuristics``      模拟退火、遗传算法、粒子群、差分进化、禁忌搜索、
                      灰狼优化、变邻域搜索、人工蜂群、标准基准函数与优化器对比
- ``geometry``        凸包、多边形面积/质心/点在多边形内、球面距离、IDW、克里金、
                      变差函数、线段相交、最小包围圆、点到线段距离、最近站点查询、
                      Sutherland-Hodgman 多边形裁剪
- ``spatial``         一维热传导（显式/隐式）、二维泊松 SOR、森林火灾元胞自动机、
                      NaSch 交通流、Buckingham π 定理、相似缩放、空间自相关 Moran's I
"""

from __future__ import annotations

__all__ = ["DEFAULT_SEED"]

#: 全库统一默认随机种子，保证示例输出可复现。
DEFAULT_SEED = 20240101
