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
- ``_common``      公共工具（校验、归一化、随机数）
- ``optimization`` 线性/整数规划、指派、运输、背包
- ``evaluation``   层次分析、熵权、CRITIC、TOPSIS、VIKOR、灰关联、DEA、模糊综合
- ``forecasting``  移动平均、指数平滑、Holt-Winters、GM(1,1)、AR 最小二乘、回测
- ``statistics``   最小二乘、岭回归、logistic、PCA、Bootstrap、置换检验、卡方
- ``clustering``   K-means++、层次聚类、DBSCAN、轮廓系数
- ``differential`` Euler/RK4、SIR、logistic 增长、Lotka-Volterra、参数拟合
- ``stochastic``   蒙特卡洛积分与置信区间、M/M/1 排队、马尔可夫稳态、随机游走
- ``graphs``       Dijkstra、Floyd、Kruskal、最大流、TSP 启发式、PageRank
- ``game``         零和博弈 LP、双矩阵纳什、Shapley 值、Gale-Shapley 稳定匹配
- ``heuristics``   模拟退火、遗传算法、粒子群
- ``geometry``     凸包、点在多边形内、IDW 插值、球面距离
"""

from __future__ import annotations

__all__ = ["DEFAULT_SEED"]

#: 全库统一默认随机种子，保证示例输出可复现。
DEFAULT_SEED = 20240101
