"""竞赛建模模式示例：每个函数都只演示“透明基线”的结构。

本文件不是完整解题器，也不绑定任何具体赛题数据。复制前先替换数据、单位、约束与验证设计。
依赖：numpy；可选 scipy。运行：python examples/modeling_patterns.py
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def normalize_minmax(values: Sequence[float]) -> list[float]:
    """把同一指标线性缩放到 [0, 1]；常用于评价模型的第一步。

    改进提示：指标有异常值时先做稳健缩放；指标是成本型时使用 1 - scaled。
    验证提示：记录 min/max，检查常数列和正负向方向，避免单位混用。
    """
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [1.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def topsis(scores: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """最小可用 TOPSIS（假定所有指标都是正向指标）。

    逐步对应论文：向量归一化 → 加权 → 理想/负理想点 → 距离 → 贴近度。
    生产使用前必须补：成本型指标方向、零列处理、权重来源和敏感性分析。
    """
    if not scores or len(scores[0]) != len(weights):
        raise ValueError("scores 列数必须等于 weights 长度")
    cols = list(zip(*scores))
    denom = [math.sqrt(sum(x * x for x in col)) for col in cols]
    norm = [[x / d if d else 0.0 for x, d in zip(row, denom)] for row in scores]
    weighted = [[x * w for x, w in zip(row, weights)] for row in norm]
    ideal = [max(row[j] for row in weighted) for j in range(len(weights))]
    nadir = [min(row[j] for row in weighted) for j in range(len(weights))]
    result = []
    for row in weighted:
        dp = math.sqrt(sum((x - y) ** 2 for x, y in zip(row, ideal)))
        dm = math.sqrt(sum((x - y) ** 2 for x, y in zip(row, nadir)))
        result.append(dm / (dp + dm) if dp + dm else 0.0)
    return result


def rolling_mean_forecast(values: Sequence[float], window: int) -> list[float]:
    """滚动均值基线；故意不读取未来值，适合先建立预测下限。"""
    if window <= 0 or len(values) <= window:
        raise ValueError("window 必须为正，且序列长度必须大于 window")
    return [sum(values[i - window:i]) / window for i in range(window, len(values))]


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """平均绝对误差；预测题优先同时报告 MAE、RMSE 和朴素基线。"""
    pairs = list(zip(actual, predicted))
    if not pairs:
        raise ValueError("不能为空")
    return sum(abs(a - p) for a, p in pairs) / len(pairs)


def dijkstra(graph: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:
    """教学版 Dijkstra，展示最短路基线与非负边前提。

    改进提示：大图使用 heapq；时间依赖路网需改变状态；VRP 不能把逐点最短路
    直接当成全局配送最优。验证时回放每条边、容量、时间窗和路径闭合。
    """
    distance = {node: math.inf for node in graph}
    distance[source] = 0.0
    unseen = set(graph)
    while unseen:
        node = min(unseen, key=lambda x: distance[x])
        unseen.remove(node)
        if distance[node] == math.inf:
            break
        for nxt, cost in graph.get(node, []):
            if cost < 0:
                raise ValueError("Dijkstra 要求边权非负")
            candidate = distance[node] + cost
            if candidate < distance.get(nxt, math.inf):
                distance[nxt] = candidate
    return distance


def monte_carlo_probability(samples: Sequence[bool]) -> tuple[float, float]:
    """从 Bernoulli 样本估计概率并返回近似标准误。

    真实竞赛中还需报告随机种子、样本量、置信区间和随样本量的收敛曲线；
    小概率事件可能需要重要抽样，而不是盲目增加普通样本。
    """
    if not samples:
        raise ValueError("samples 不能为空")
    p = sum(samples) / len(samples)
    se = math.sqrt(p * (1 - p) / len(samples))
    return p, se


def main() -> None:
    print("TOPSIS scores:", topsis([[8, 7], [6, 9], [9, 6]], [0.5, 0.5]))
    print("MAE:", mae([10, 12, 9], [9, 13, 10]))
    print("Dijkstra:", dijkstra({"A": [("B", 2), ("C", 5)], "B": [("C", 1)], "C": []}, "A"))
    print("Monte Carlo:", monte_carlo_probability([True, False, True, True]))


if __name__ == "__main__":
    main()
