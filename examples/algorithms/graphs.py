"""图与网络算法：最短路、最小生成树、最大流、TSP 启发式、PageRank、连通分量。

本模块共 20 个公开函数，按用途分成六组：
- 最短路：``dijkstra`` / ``floyd_warshall`` / ``a_star`` / ``reconstruct_path``；
- 最小生成树：``kruskal_mst`` / ``prim_mst``；
- 流与匹配：``max_flow_edmonds_karp`` / ``min_cut_edges`` / ``min_cost_flow`` / ``bipartite_matching``；
- 路径启发式与配送：``tsp_nearest_neighbor`` / ``tsp_two_opt`` / ``vrp_clarke_wright``；
- 中心性与社区：``pagerank`` / ``degree_centrality`` / ``closeness_centrality``
  / ``betweenness_centrality`` / ``louvain_communities``；
- 连通性与可靠性：``connected_components`` / ``network_robustness``。

这一组是数学建模里出现频率最高的一类"结构模型"：
- 最短路（Dijkstra / Floyd）用于路网、管网、换乘、依赖排序；
- 最小生成树用于通信网/管网铺设、聚类骨架；
- 最大流/最小割用于运力、匹配、拦截、可靠性（并可用 max-flow min-cut 定理互相验证）；
- TSP 启发式用于巡检、配送、遍历；
- PageRank 用于影响力/重要性排序（本质是马尔可夫链稳态）。

实现口径提醒（写论文时要交代）：
- ``adj`` 统一是**出邻接字典** ``{u: {v: w}}``，不用邻接矩阵，因为稀疏路网用它更省；
- 无穷远统一用 ``float("inf")`` 表示，不要用 -1 或 1e18 之类的哨兵；
- PageRank 返回的是**概率分布**（和为 1），阻尼默认 0.85。

依赖：仅 numpy + 标准库 + ``._common``。
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ._common import as_matrix, check_square, rng as make_rng

Node = Hashable

__all__ = [
    "dijkstra",
    "floyd_warshall",
    "reconstruct_path",
    "kruskal_mst",
    "max_flow_edmonds_karp",
    "min_cut_edges",
    "tsp_nearest_neighbor",
    "tsp_two_opt",
    "pagerank",
    "connected_components",
    "prim_mst",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "louvain_communities",
    "min_cost_flow",
    "bipartite_matching",
    "a_star",
    "vrp_clarke_wright",
    "network_robustness",
]


# --------------------------------------------------------------------------- #
# 最短路
# --------------------------------------------------------------------------- #
def dijkstra(adj: Mapping[Node, Mapping[Node, float]], src: Node) -> dict:
    """单源最短路（Dijkstra + 二叉堆），边权必须非负。

    参数:
        adj: 出邻接字典 ``{u: {v: w}}``。``w`` 是 u->v 的边权，必须 >= 0。
             边的两个端点都会自动纳入节点全集，即使某个节点没有出边。
        src: 源点。若 src 不在 adj 中，它仍会被当作一个孤立源点处理。

    返回:
        ``{"dist": {node: float}, "prev": {node: Optional[node]}}``。
        ``dist[node]`` 为源点到 node 的最短距离，不可达为 ``float("inf")``；
        ``prev[node]`` 为最短路上的前驱，源点与不可达点均为 ``None``。

    算法:
        贪心 + 松弛：每次从堆中弹出当前距离最小的未定型节点，对其出边做松弛。
        堆中允许存在同一节点的多条过期记录（惰性删除），靠弹出的距离与 ``dist`` 比对跳过。

    复杂度:
        时间 O((V + E) log V)（惰性删除使堆操作常数略大）/ 空间 O(V + E)。

    陷阱:
        1. **负权会静默给出错误答案**，所以这里显式抛 ValueError；有负权请改用
           Bellman-Ford 或 Johnson。零权是允许的。
        2. 返回的 ``prev`` 只能还原**一棵最短路径树**；当存在多条等长最短路时，
           拿到哪一条取决于堆的弹出顺序，不要在论文里声称"the"唯一路径。
        3. 键是**节点对象本身**，不是下标。若节点是 numpy 整数，请先转成 int，
           否则 ``0`` 与 ``np.int64(0)`` 混用会导致查不到键。

    参考:
        Dijkstra 1959, "A note on two problems in connexion with graphs"。
    """
    if not isinstance(adj, Mapping):
        raise ValueError("adj 必须是 {u: {v: w}} 形式的字典")

    # 节点全集：键 + 所有出现过的邻居（邻居可能没有出边）
    nodes: List[Node] = []
    seen = set()
    for u, nbrs in adj.items():
        if u not in seen:
            seen.add(u)
            nodes.append(u)
        if not isinstance(nbrs, Mapping):
            raise ValueError(f"adj[{u!r}] 必须是 {{v: w}} 形式的字典")
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                nodes.append(v)

    # 先统一校验非负，避免跑到一半才发现（也避免"部分正确"的误导性结果）
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            try:
                wf = float(w)
            except (TypeError, ValueError):
                raise ValueError(f"边 {u!r}->{v!r} 的权重不是数值：{w!r}")
            if not np.isfinite(wf):
                raise ValueError(f"边 {u!r}->{v!r} 的权重必须是有限值，得到 {wf}")
            if wf < 0:
                raise ValueError(
                    f"边 {u!r}->{v!r} 权重为负（{wf}）；Dijkstra 要求非负权重，"
                    "请改用 Bellman-Ford / Johnson"
                )

    dist: Dict[Node, float] = {n: float("inf") for n in nodes}
    prev: Dict[Node, Optional[Node]] = {n: None for n in nodes}
    if src not in dist:
        dist[src] = 0.0
        prev[src] = None
    else:
        dist[src] = 0.0

    heap: List[Tuple[float, int, Node]] = [(0.0, 0, src)]
    counter = 1  # 节点可能不可比较，用自增序号打破堆的平局比较
    done = set()
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in done:
            continue
        if d > dist[u]:
            continue  # 过期记录
        done.add(u)
        nbrs = adj.get(u)
        if not nbrs:
            continue
        for v, w in nbrs.items():
            nd = d + float(w)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, counter, v))
                counter += 1
    return {"dist": dist, "prev": prev}


def floyd_warshall(W) -> dict:
    """全源最短路（Floyd-Warshall 动态规划）。

    参数:
        W: (n, n) 距离矩阵。``W[i, j]`` 是 i->j 的直接距离；无边用 ``np.inf``。
           对角线应为 0（若给出非 0 对角元，本实现会按给定值处理，不做覆盖）。

    返回:
        ``{"dist": np.ndarray(n, n), "next_node": np.ndarray(n, n)}``。
        ``next_node[i, j]`` 是 i->j 最短路上**从 i 出发的第一步**（"下一跳矩阵"，直接可达时
        就是 j 本身），不可达或 i==j 时为 ``-1``。它可直接喂给 :func:`reconstruct_path`。

    算法:
        对每个中间点 k，做 ``dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])``。
        只用 "k 之前" 的距离更新，因此允许就地覆盖而不必开三维数组。

    复杂度:
        时间 O(n^3) / 空间 O(n^2)。

    陷阱:
        1. **负环**会让 ``dist[i, i] < 0``，此时"最短路"无下界。本实现会在返回前
           检查对角线并抛 ValueError，而不是返回一个看似正常的矩阵。
        2. 用 0 表示"无边"是错的：0 权边和无穷远必须区分开。填矩阵时无边请填 ``np.inf``。
        3. 浮点加法下的 ``inf`` 会传播：``inf + (-inf)`` 是 NaN，所以不允许出现 -inf。
        4. 本函数在工作副本上**原地更新**（``dist`` 由 ``W.astype(float).copy()`` 得到），
           因此**不会**改动调用方传入的数组。

    参考:
        Floyd 1962 / Warshall 1962；《算法导论》第 25 章。
    """
    # 注意：不能直接用 _common.as_matrix —— 它禁止 inf，而本函数的输入约定正是用 inf 表示无边。
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError(f"W 必须是二维数组，得到 ndim={W.ndim}")
    check_square(W, "W")
    if W.size == 0:
        raise ValueError("W 不能为空")
    if np.any(np.isnan(W)):
        raise ValueError("W 含 NaN；无边请用 np.inf 表示")
    if np.any(W == -np.inf):
        raise ValueError("W 含 -inf；本实现不支持负无穷边")

    n = W.shape[0]
    dist = W.astype(float).copy()  # 不改调用方数组

    next_node = np.full((n, n), -1, dtype=int)
    finite = np.isfinite(W)
    ii, jj = np.nonzero(finite)
    next_node[ii, jj] = jj  # 直接可达：下一跳就是终点
    np.fill_diagonal(next_node, -1)

    for k in range(n):
        # (n,) 广播：dk_j 是 k 到各点的距离，di_k 是各点到 k 的距离
        dk = dist[k, :]
        through = dist[:, k][:, None] + dk[None, :]
        better = through < dist
        if not np.any(better):
            continue
        dist = np.where(better, through, dist)
        # 只有真正被 k 改进的 (i, j) 才把下一跳改成"i 走 k 的第一步"
        k_first = next_node[:, k]
        cand = np.broadcast_to(k_first[:, None], next_node.shape)
        next_node = np.where(better, cand, next_node)
        np.fill_diagonal(next_node, -1)

    if np.any(np.diag(dist) < 0):
        bad = int(np.argmin(np.diag(dist)))
        raise ValueError(f"检测到负环（dist[{bad}, {bad}]={np.diag(dist)[bad]}），最短路无下界")
    return {"dist": dist, "next_node": next_node}


def reconstruct_path(prev_or_next, src, dst) -> list:
    """从 Dijkstra 的 ``prev`` 或 Floyd 的 ``next_node`` 还原一条路径。

    参数:
        prev_or_next: 两种口径二选一：
            - dict（Dijkstra 的 ``prev``）：``node -> 前驱``，从 dst 反向回溯；
            - np.ndarray（Floyd 的 ``next_node``）：``(n, n)`` 下一跳矩阵，从 src 正向推进。
        src: 起点（下标或节点对象）。
        dst: 终点（下标或节点对象）。

    返回:
        list，形如 ``[src, ..., dst]``。不可达时返回 ``[]``；``src == dst`` 时返回 ``[src]``。

    算法:
        - dict 口径：不断取前驱直到 src，再反转；
        - 矩阵口径：从 src 开始反复取 ``next_node[cur, dst]``，直到 dst。
        两种口径都会检测环（最多走 V 步），防止脏数据导致死循环。

    复杂度:
        时间 O(V) / 空间 O(V)。

    陷阱:
        1. 两种口径**不能混用**：把 ``next_node`` 当 ``prev`` 传进来会得到反向乱序结果
           （本函数靠类型判断区分，不会报错，所以更要小心）。
        2. Floyd 的 ``next_node`` 口径要求 src/dst 是 **0..n-1 的整数下标**；
           若你的节点是字符串，请自己维护 ``name -> index`` 映射（或改用 dict 的 prev 口径）。
        3. 不可达返回空列表而不是抛异常；调用处必须显式判断 ``len(path) == 0``，
           否则下游把空路径当成合法解会静默出错。

    参考:
        《算法导论》第 22/25 章路径还原通例。
    """
    if src == dst:
        return [src]

    if isinstance(prev_or_next, np.ndarray):
        nxt = prev_or_next
        n = nxt.shape[0]
        if not (isinstance(src, (int, np.integer)) and isinstance(dst, (int, np.integer))):
            raise ValueError("next_node 口径要求 src/dst 是整数下标")
        cur = int(src)
        target = int(dst)
        if not (0 <= cur < n and 0 <= target < n):
            raise ValueError("下标越界")
        path = [cur]
        for _ in range(n + 1):
            step = int(nxt[cur, target])
            if step < 0:
                return []  # 不可达
            path.append(step)
            if step == target:
                return path
            cur = step
        raise ValueError("next_node 含环，路径无法还原")

    # dict 口径：prev
    if src not in prev_or_next:
        raise ValueError(f"prev 中缺少起点 {src!r}")
    path = [dst]
    cur = dst
    for _ in range(len(prev_or_next) + 1):
        if cur == src:
            path.reverse()
            return path
        p = prev_or_next.get(cur)
        if p is None:
            return []  # 回溯中断 = 不可达
        path.append(p)
        cur = p
    raise ValueError("prev 含环，路径无法还原")


# --------------------------------------------------------------------------- #
# 最小生成树
# --------------------------------------------------------------------------- #
def _find(parent: List[int], i: int) -> int:
    """并查集查找（带路径压缩）。"""
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def kruskal_mst(n_nodes: int, edges: Sequence[Tuple[int, int, float]]) -> dict:
    """Kruskal 最小生成树（并查集 + 按权排序）。

    参数:
        n_nodes: 节点数，节点编号为 ``0..n_nodes-1``。
        edges: 边列表 ``[(u, v, w), ...]``，无向、权重须有限（可为负）。

    返回:
        ``{"total_weight": float, "edges": [(u, v, w), ...]}``，``edges`` 按权重升序，
        ``u < v`` 归一化。图不连通时返回的是**最小生成森林**，``edges`` 长度 < n_nodes-1；
        请用 ``len(edges) == n_nodes - 1`` 判断是否连通。

    算法:
        按权重升序扫描边，若两端点不在同一并查集分量中则选入并合并；选满 n-1 条即停。

    复杂度:
        时间 O(E log E)（排序主导，并查集近似 O(E α(V))）/ 空间 O(V + E)。

    陷阱:
        1. **自环 (u, u, w)** 必须跳过，否则会被并查集判为"已在同一分量"而静默丢弃——
           行为没错，但如果你据此认为"选了自环"就错了；这里显式忽略更清楚。
        2. **多重边**（同 u,v 不同 w）是允许的，算法会自动只取较小的那条。
        3. 不连通图不会报错，只返回森林。很多建模代码忘了这一步，直接拿 ``total_weight``
           当"连通成本"，得到偏小的错误结论。
        4. 权重为负时 MST 仍然良定义（会优先选负权边），不要误以为"权重必须为正"。

    参考:
        Kruskal 1956；并查集（Union-Find，Tarjan 1975 路径压缩分析）。
    """
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes <= 0:
        raise ValueError("n_nodes 必须是正整数")
    n_nodes = int(n_nodes)

    # 无向边两端顺序无关，统一成 (min, max, w) 让结果可复现
    normalized: List[Tuple[int, int, float]] = []
    for item in edges:
        if len(item) != 3:
            raise ValueError(f"每条边必须是 (u, v, w)，得到 {item!r}")
        u, v, w = item
        u = int(u)
        v = int(v)
        w = float(w)
        if not (0 <= u < n_nodes and 0 <= v < n_nodes):
            raise ValueError(f"边 ({u}, {v}) 的端点超出 0..{n_nodes - 1}")
        if not np.isfinite(w):
            raise ValueError(f"边 ({u}, {v}) 的权重必须是有限值，得到 {w}")
        if u == v:
            continue  # 自环对生成树无意义
        if u > v:
            u, v = v, u
        normalized.append((u, v, w))

    normalized.sort(key=lambda e: (e[2], e[0], e[1]))

    parent = list(range(n_nodes))
    rank = [0] * n_nodes
    chosen: List[Tuple[int, int, float]] = []
    total = 0.0
    for u, v, w in normalized:
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            continue
        if rank[ru] < rank[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        if rank[ru] == rank[rv]:
            rank[ru] += 1
        chosen.append((u, v, w))
        total += w
        if len(chosen) == n_nodes - 1:
            break
    return {"total_weight": float(total), "edges": chosen}


# --------------------------------------------------------------------------- #
# 最大流 / 最小割
# --------------------------------------------------------------------------- #
def _validate_capacity(capacity: Mapping[Node, Mapping[Node, float]]) -> List[Node]:
    """把容量字典规范化成方形矩阵用的节点列表，并校验非负。"""
    if not isinstance(capacity, Mapping):
        raise ValueError("capacity 必须是 {u: {v: cap}} 形式的字典")
    nodes: List[Node] = []
    seen = set()
    for u, nbrs in capacity.items():
        if u not in seen:
            seen.add(u)
            nodes.append(u)
        if not isinstance(nbrs, Mapping):
            raise ValueError(f"capacity[{u!r}] 必须是 {{v: cap}} 形式的字典")
        for v, cap in nbrs.items():
            if v not in seen:
                seen.add(v)
                nodes.append(v)
            c = float(cap)
            if not np.isfinite(c):
                raise ValueError(f"容量 capacity[{u!r}][{v!r}] 必须是有限值，得到 {c}")
            if c < 0:
                raise ValueError(f"容量不能为负：capacity[{u!r}][{v!r}]={c}")
    return nodes


def max_flow_edmonds_karp(
    capacity: Mapping[Node, Mapping[Node, float]], source: Node, sink: Node
) -> dict:
    """Edmonds-Karp 最大流（BFS 找增广路的 Ford-Fulkerson 实现）。

    参数:
        capacity: 有向容量字典 ``{u: {v: cap}}``。只写 u->v 就是单向边；
            要建无向边请**两个方向都写同样的容量**（这不是自动的）。
        source: 源点。
        sink: 汇点，必须与 source 不同。

    返回:
        ``{"max_flow": float, "flow": {u: {v: 净流量}}}``。``flow`` 按**原始容量字典的
        方向**汇总净流量 ``cap[u][v] - res[u][v]``，只列出该值严格大于 0 的有序对；
        因此它**不含**残量网络中的反向抵消边（那些边原始容量为 0，净流量必不超过 0，
        会被过滤掉）。对任意中间点，流入量之和等于流出量之和。

    算法:
        反复用 BFS 在**残量网络**中找一条源到汇的最短（边数最少）增广路，沿路推
        瓶颈容量，直到不存在增广路。BFS 保证增广路条数 O(VE)，因此不会像 DFS 版
        Ford-Fulkerson 那样在有理数容量上退化甚至不收敛。

    复杂度:
        时间 O(V E^2)（BFS 增广次数 O(VE)，每次 O(E)）/ 空间 O(V + E)。

    陷阱:
        1. **反向边**必须有：残量网络里 ``res[v][u] += pushed`` 是算法正确性的关键，
           只写正向边会得到偏小的流量却不报错。
        2. 容量可以为 0（表示不允许），但**不能为负**；负容量会让残量网络出现可无限
           增广的假象。
        3. 平行边要自己合并成一条（字典天然合并），否则后写的会覆盖先写的。
        4. 浮点容量下"无增广路"的判断用的是严格 ``> 0``；若容量是 1e-16 级的小量，
           数值噪声可能被当成可增广。竞赛数据请先做单位统一和量级缩放。
        5. source == sink 无意义，这里直接抛 ValueError（有些实现会返回 inf）。

    参考:
        Edmonds & Karp 1972；《算法导论》第 26 章。
    """
    if source == sink:
        raise ValueError("source 与 sink 不能相同")
    nodes = _validate_capacity(capacity)
    if source not in nodes:
        raise ValueError(f"source {source!r} 不在容量字典中")
    if sink not in nodes:
        raise ValueError(f"sink {sink!r} 不在容量字典中")

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    cap = np.zeros((n, n), dtype=float)
    for u, nbrs in capacity.items():
        for v, c in nbrs.items():
            cap[idx[u], idx[v]] += float(c)  # 平行边累加
    res = cap.copy()  # 残量矩阵

    s, t = idx[source], idx[sink]
    max_flow = 0.0
    while True:
        # BFS：记录前驱与到达该点的瓶颈
        parent = np.full(n, -1, dtype=int)
        parent[s] = s
        queue = [s]
        head = 0
        while head < len(queue) and parent[t] == -1:
            u = queue[head]
            head += 1
            for v in np.nonzero(res[u] > 0)[0]:
                if parent[v] == -1:
                    parent[v] = u
                    queue.append(int(v))
                    if v == t:
                        break
        if parent[t] == -1:
            break  # 无增广路

        # 回推瓶颈
        bottleneck = float("inf")
        v = t
        while v != s:
            u = int(parent[v])
            bottleneck = min(bottleneck, res[u, v])
            v = u
        # 更新残量
        v = t
        while v != s:
            u = int(parent[v])
            res[u, v] -= bottleneck
            res[v, u] += bottleneck
            v = u
        max_flow += bottleneck

    flow: Dict[Node, Dict[Node, float]] = {}
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            f = cap[i, j] - res[i, j]  # 净流量 = 容量 - 剩余容量
            if f > 0:
                flow.setdefault(u, {})[v] = float(f)
    return {"max_flow": float(max_flow), "flow": flow}


def min_cut_edges(
    capacity: Mapping[Node, Mapping[Node, float]], source: Node, sink: Node
) -> list:
    """由残量网络可达性求**最小割边集**（max-flow min-cut 定理的构造侧）。

    参数:
        capacity: 同 :func:`max_flow_edmonds_karp`。
        source: 源点。
        sink: 汇点。

    返回:
        list of ``(u, v, cap)``：所有满足 "u 在源侧可达集 S 中、v 不在 S 中" 的**原始
        有向边**。其容量之和恰好等于最大流。若 source 与 sink 之间无路，返回 ``[]``。

    算法:
        先跑一次最大流，然后在残量网络（本实现取 ``res > 1e-12`` 的边，比
        :func:`max_flow_edmonds_karp` 判增广路时的严格 ``> 0`` 略松，用于吸收浮点噪声）
        上从 source 做 DFS，得到源侧集合 S；割边 = 从 S 指向 V\\S 且**原始容量 > 0**
        的边。由 max-flow min-cut 定理，这些边必然全部饱和，且容量和 = 最大流。

    复杂度:
        时间 O(V E^2)（含最大流）/ 空间 O(V + E)。

    陷阱:
        1. 返回的是**有向边** (S -> V\\S)，双向图里会只列出从 S 出去的那个方向，
           这正是定理要的那个割，不要"补全"成两条。
        2. 最小割**不唯一**：这里给出的是"离源点最近"的那个最小割（源侧集合最小）。
           论文里若断言割边集合唯一，会被质疑。
        3. 割容量必须用**原始容量**求和，不要用残量；用残量求和会得到 0。
        4. 若原图有平行边，容量已合并，返回的 cap 是合并后的值，和输入逐条对不上。
        5. 可达性判断用的是 ``res > 1e-12``（而非 ``max_flow_edmonds_karp`` 里的
           ``> 0``）：容量量级远小于 1e-12 的题（例如先把单位换成"亿元"再取微小系数）
           可能因此判错可达集合，请先把数据缩放到 O(1) 量级。

    参考:
        Ford & Fulkerson 1956（最大流最小割定理）。
    """
    if source == sink:
        raise ValueError("source 与 sink 不能相同")
    nodes = _validate_capacity(capacity)
    if source not in nodes or sink not in nodes:
        raise ValueError("source / sink 必须在容量字典中")

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    cap = np.zeros((n, n), dtype=float)
    for u, nbrs in capacity.items():
        for v, c in nbrs.items():
            cap[idx[u], idx[v]] += float(c)

    mf = max_flow_edmonds_karp(capacity, source, sink)
    # 重算残量矩阵（用返回的流量），避免重复实现
    res = cap.copy()
    for u, row in mf["flow"].items():
        for v, f in row.items():
            res[idx[u], idx[v]] -= f
            res[idx[v], idx[u]] += f

    s = idx[source]
    reachable = {s}
    stack = [s]
    while stack:
        u = stack.pop()
        for v in np.nonzero(res[u] > 1e-12)[0]:
            if int(v) not in reachable:
                reachable.add(int(v))
                stack.append(int(v))

    cut: List[Tuple[Node, Node, float]] = []
    for i, u in enumerate(nodes):
        if i not in reachable:
            continue
        for j, v in enumerate(nodes):
            if j in reachable:
                continue
            if cap[i, j] > 0:
                cut.append((u, v, float(cap[i, j])))
    cut.sort(key=lambda e: (str(e[0]), str(e[1])))
    return cut


# --------------------------------------------------------------------------- #
# TSP 启发式
# --------------------------------------------------------------------------- #
def _validate_dist(dist) -> np.ndarray:
    """校验距离矩阵：方阵、无 NaN、非负。**不做**对称性检查，也不发任何警告。"""
    D = as_matrix(dist, "dist")
    check_square(D, "dist")
    if np.any(D < 0):
        raise ValueError("dist 不能有负距离")
    if np.any(np.isnan(D)):
        raise ValueError("dist 含 NaN")
    return D.astype(float)


def tsp_nearest_neighbor(dist, start: int = 0) -> dict:
    """最近邻构造 TSP 初始回路（贪心，不保证最优）。

    参数:
        dist: (n, n) 距离矩阵，``dist[i, j]`` 为 i->j 距离。非对称矩阵也支持。
        start: 起始城市下标，默认 0。

    返回:
        ``{"tour": [i0, i1, ..., i_{n-1}], "length": float}``。
        ``tour`` **不含重复的起点**（闭环由首尾隐含），长度按闭环计算。

    算法:
        从 start 出发，每次跳到最近的未访问城市，最后回到 start 闭合回路。

    复杂度:
        时间 O(n^2) / 空间 O(n)。

    陷阱:
        1. 最近邻可能给出比最优解差 20%~25% 的回路（n 大时）；必须再接 2-opt 或
           Or-opt 改进，不要直接把它的长度当"TSP 最优值"。
        2. **起点不同结果不同**，竞赛中应报告"对每个起点各跑一次取最好"的结论，
           而不是随手取 start=0。
        3. 非对称矩阵（如单向道路）下闭合回路方向有意义；本函数沿行进方向累加，
           不做对称化处理。
        4. n == 0 抛 ValueError；n == 1 返回 ``[0]``、长度 0。

    参考:
        最近邻启发式（greedy nearest neighbor），见 Gutin & Punnen《The TSP and Its Variations》。
    """
    D = _validate_dist(dist)
    n = D.shape[0]
    if n == 0:
        raise ValueError("dist 不能为空")
    start = int(start)
    if not (0 <= start < n):
        raise ValueError(f"start={start} 越界，应为 0..{n - 1}")

    unvisited = set(range(n))
    tour = [start]
    unvisited.discard(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda c: (D[cur, c], c))
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    length = _tour_length(tour, D)
    return {"tour": tour, "length": float(length)}


def _tour_length(tour: Sequence[int], D: np.ndarray) -> float:
    """按闭环计算回路总长度（tour 不含重复起点）。"""
    if len(tour) <= 1:
        return 0.0
    total = 0.0
    for k in range(len(tour)):
        total += D[tour[k], tour[(k + 1) % len(tour)]]
    return float(total)


def tsp_two_opt(tour, dist, max_pass: int = 100) -> dict:
    """2-opt 局部搜索改进 TSP 回路。

    参数:
        tour: 初始回路（不含重复起点），如 :func:`tsp_nearest_neighbor` 的输出。
        dist: (n, n) 距离矩阵，与 tour 的规模一致。
        max_pass: 最大扫描轮数上限（每轮尝试所有 (i, j) 对），默认 100。

    返回:
        ``{"tour": list, "length": float, "improved": bool}``。``improved`` 表示相比
        输入回路长度**严格变小**（相等时为 False）。

    算法:
        反复扫描所有 0 <= i < j < n，尝试把边 (i, i+1) 与 (j, j+1) 换成
        (i, j) 与 (i+1, j+1)，即**反转 tour[i+1..j]**；只接受严格改进（容差 1e-12）。
        一轮扫描中每遇到一个严格改进就**立即接受并沿当前解继续扫下去**（不回到本轮开头
        重扫），一轮扫完无改进或达到 max_pass 才停。

    复杂度:
        时间 O(n^2) 每次扫描，最多 max_pass 轮；实践上常几轮就停 / 空间 O(n)。

    陷阱:
        1. 2-opt 只是**局部最优**：一轮扫描内不重扫，因此本轮接受的改进不会立刻被
           后续交换重新评估，最终解依赖初始回路。本模块自带的 8 城反例就说明了这点——
           最近邻构造后 2-opt 仍**达不到**穷举最优值 26.484006136241458
           （见 ``_self_test`` 的 ``tsp8_2opt_len`` 与 ``tsp8_brute_opt_len``）。
           论文里只能报"改进幅度"，不能宣称最优；城市多了还需多起点重启。
        2. 赚量算法（只算增量 delta）在**非对称**矩阵上不成立：反转一段会改变两端的
           方向。本实现直接重算整条回路长度，因此对非对称矩阵也正确但更慢。
        3. 传入的 tour 若含重复点或缺点，本函数不校验，会静默给出无意义结果；
           请先用 :func:`tsp_nearest_neighbor` 或自己保证它是 0..n-1 的一个排列。
        4. 浮点比较用严格小于，等长回路的交换不会发生，因此相同输入必得相同输出。

    参考:
        Croes 1958 / Lin & Kernighan 1973（2-opt / k-opt 邻域）。
    """
    D = _validate_dist(dist)
    n = D.shape[0]
    if n <= 3:
        tour_list = [int(x) for x in tour]
        return {"tour": tour_list, "length": _tour_length(tour_list, D), "improved": False}
    if len(tour) != n:
        raise ValueError(f"tour 长度 {len(tour)} 与 dist 规模 {n} 不一致")

    cur = [int(x) for x in tour]
    best_len = _tour_length(cur, D)
    best = list(cur)
    improved = False

    for _ in range(int(max_pass)):
        changed = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  # 相邻交换不改变回路
                cand = best[:i + 1] + best[i + 1:j + 1][::-1] + best[j + 1:]
                cand_len = _tour_length(cand, D)
                if cand_len < best_len - 1e-12:
                    best = cand
                    best_len = cand_len
                    changed = True
                    improved = True
        if not changed:
            break
    return {"tour": best, "length": float(best_len), "improved": bool(improved)}


# --------------------------------------------------------------------------- #
# 节点对齐顺序（返回数组类结果共用）
# --------------------------------------------------------------------------- #
def _ordered_nodes(adj: Mapping[Node, Mapping[Node, float]]) -> List[Node]:
    """收集 ``adj`` 中出现的全部节点，给出**确定性**的数组对齐顺序。

    参数:
        adj: 邻接字典 ``{u: {v: w}}``。

    返回:
        list，包含所有作为键或作为邻居出现过的节点，无重复。
        优先返回 ``sorted(nodes)``；若节点之间不可比较（如 int 与 str 混用，
        ``sorted`` 抛 ``TypeError``），退化为"扫描顺序"：先按键的迭代顺序，
        再按邻居首次出现的顺序。

    算法:
        单次遍历收集出现顺序，然后尝试排序；排序失败则回退。

    复杂度:
        时间 O(V log V) / 空间 O(V)。

    陷阱:
        这个顺序是 :func:`pagerank` 与 :func:`connected_components` 返回值的
        下标约定。**不排序**的话，返回的下标会随"先遇到谁"而变，同一个图换个
        字典构造写法就得到不同下标——这类 bug 只表现为数值对不上号，极难排查。
        键类型不可比较时才退回扫描顺序，此时必须依赖调用方自己记录键顺序。

    参考:
        本项目约定：返回数组型结果一律给出可复现的节点顺序。
    """
    nodes: List[Node] = []
    seen = set()
    for u, nbrs in adj.items():
        if u not in seen:
            seen.add(u)
            nodes.append(u)
        for v in nbrs:
            if v not in seen:
                seen.add(v)
                nodes.append(v)
    try:
        return sorted(nodes)
    except TypeError:
        return nodes


# --------------------------------------------------------------------------- #
# PageRank
# --------------------------------------------------------------------------- #
def pagerank(
    adj: Mapping[Node, Mapping[Node, float]],
    damping: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """PageRank 重要性打分（带悬挂点处理的幂迭代）。

    参数:
        adj: 出邻接字典 ``{u: {v: w}}``。``w`` 是**转移强度**（非负，不必归一化，
             本函数按行归一化），用 1.0 表示普通无权有向边。
        damping: 阻尼系数 d，默认 0.85。
        tol: L1 收敛阈值。
        max_iter: 最大迭代次数。

    返回:
        np.ndarray，形状 (n,)。**下标对齐 :func:`_ordered_nodes` 的顺序**：
        优先是 ``sorted(adj 中所有节点)``（节点为纯 str 或纯 int 时即字典序/数值序），
        节点不可比较时退化为扫描顺序。想稳妥地把分数和节点对应起来，请显式写
        ``nodes = sorted(adj)`` 或用返回向量与 ``nodes`` 一起打包。
        向量非负且严格和为 1。

    算法:
        ``r = d * M^T r + (1 - d) / n + d * (悬挂点总权重) / n``，其中 M 按行归一化。
        悬挂点（无出边）的概率质量被均匀重新分配，避免收敛到全 0。

    复杂度:
        时间 O(max_iter * E) / 空间 O(V + E)。

    陷阱:
        1. **节点顺序**：``r[i]`` 对应 ``sorted(adj)`` 中的第 i 个节点（节点不可比较时
           为扫描顺序，见 :func:`_ordered_nodes`）。只依赖"和为 1"最安全；要和节点名
           对应就必须自己按同样的顺序取节点。混用 int/str 键时排序会失败并静默退回
           扫描顺序，此时务必显式记录键顺序。
        2. 悬挂点不处理会漏概率质量，迭代结果和不为 1（这是最常见的 bug）。
        3. 阻尼系数过小（如 0.1）时结果接近均匀分布，看起来"没区分度"；论文里要报告 d。
        4. 达到 max_iter 仍未收敛**不会报错**，只返回当前向量。收敛对竞赛小图很快，
           但大规模稀疏图建议同时打印残差。
        5. 每条边权重必须非负；负权重会让幂迭代失去概率意义。

    参考:
        Page & Brin 1998, "The PageRank Citation Ranking"；幂迭代 = 马尔可夫链稳态分布。
    """
    if not isinstance(adj, Mapping):
        raise ValueError("adj 必须是 {u: {v: w}} 形式的字典")
    d = float(damping)
    if not (0.0 <= d < 1.0):
        raise ValueError(f"damping 应在 [0, 1) 内，得到 {d}")
    if tol <= 0:
        raise ValueError("tol 必须为正")

    nodes: List[Node] = _ordered_nodes(adj)
    n = len(nodes)
    if n == 0:
        return np.zeros(0, dtype=float)

    idx = {node: i for i, node in enumerate(nodes)}
    out_w = np.zeros(n, dtype=float)
    M = np.zeros((n, n), dtype=float)  # M[i, j] = i->j 的转移概率
    for u, nbrs in adj.items():
        i = idx[u]
        for v, w in nbrs.items():
            wf = float(w)
            if not np.isfinite(wf) or wf < 0:
                raise ValueError(f"边权必须是非负有限值：{u!r}->{v!r}={w!r}")
            M[i, idx[v]] += wf
            out_w[i] += wf
    for i in range(n):
        if out_w[i] > 0:
            M[i] /= out_w[i]

    dangling = out_w <= 0
    r = np.full(n, 1.0 / n)
    for _ in range(int(max_iter)):
        dangling_mass = float(r[dangling].sum())
        new_r = d * (M.T @ r) + d * dangling_mass / n + (1.0 - d) / n
        if np.abs(new_r - r).sum() < tol:
            r = new_r
            break
        r = new_r
    total = r.sum()
    if total > 0:
        r = r / total  # 抹掉浮点累积误差，保证严格和为 1
    return r


def connected_components(adj: Mapping[Node, Mapping[Node, float]]) -> list:
    """无向化后的连通分量标签（0, 1, 2, ... 按首次出现顺序编号）。

    参数:
        adj: 邻接字典 ``{u: {v: w}}``。方向被**忽略**（当作无向图），边权也忽略。

    返回:
        list of int，长度 = 节点数，``labels[i]`` 是第 i 个节点（按 :func:`_ordered_nodes`
        的顺序，即优先 ``sorted(adj 中所有节点)``）的分量编号。编号从 0 开始、
        按分量中最早出现的节点排序。

    算法:
        把 adj 无向化后做 BFS/DFS；为避免递归深度问题，用显式栈迭代。

    复杂度:
        时间 O(V + E) / 空间 O(V + E)。

    陷阱:
        1. 参数名虽然是 ``adj``，但这里按**无向**处理。"强连通分量"是另一个概念
           （Tarjan / Kosaraju），不要混用结论。
        2. 只出现在别人邻接表里的节点也会被分配标签，别以为"孤立点"会消失。
        3. 返回的是标签数组而不是"分量列表"，因为大多数下游（如判断两点是否连通）
           用标签查表更方便；需要分组时用 ``collections.defaultdict(list)`` 自己聚合。
        4. **下标顺序**：``labels[i]`` 对应 ``sorted(adj 中所有节点)`` 的第 i 个
           （节点不可比较时退回扫描顺序，见 :func:`_ordered_nodes`）。用
           ``dict(zip(sorted(adj), labels))`` 才是"名字 -> 分量"的可靠映射；直接把
           ``labels[i]`` 当"第 i 号节点"用，在键不是 0..n-1 时会静默错位。

    参考:
        图连通性的 BFS/DFS 通例；强连通分量见 Tarjan 1972。
    """
    if not isinstance(adj, Mapping):
        raise ValueError("adj 必须是 {u: {v: w}} 形式的字典")

    nodes: List[Node] = _ordered_nodes(adj)

    undirected: Dict[Node, List[Node]] = {n: [] for n in nodes}
    for u, nbrs in adj.items():
        for v in nbrs:
            undirected[u].append(v)
            undirected[v].append(u)

    labels = [-1] * len(nodes)
    index_of = {node: i for i, node in enumerate(nodes)}
    comp = 0
    for i, start in enumerate(nodes):
        if labels[i] != -1:
            continue
        stack = [start]
        labels[i] = comp
        while stack:
            u = stack.pop()
            for v in undirected[u]:
                j = index_of[v]
                if labels[j] == -1:
                    labels[j] = comp
                    stack.append(v)
        comp += 1
    return labels


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _weight_matrix(
    adj: Union[np.ndarray, Sequence[Sequence[float]], Mapping[Node, Mapping[Node, float]]],
) -> Tuple[np.ndarray, List[Node]]:
    """把邻接资料统一成（对称权重矩阵, 节点顺序）。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典，或 ``n x n`` 邻接矩阵（``inf`` 表示无边）。

    返回:
        ``(W, nodes)``：``W`` 是 ``n x n`` float 矩阵（对角线置 0、无边处为 ``inf``、
        **已对称化**）；``nodes`` 是下标 -> 原节点 的列表（矩阵输入时为 ``range(n)``）。

    算法:
        字典输入先按 :func:`_ordered_nodes` 定序再填矩阵；随后取 ``min(W, W.T)`` 完成
        无向化（双向权重不同时取较小值）。

    复杂度:
        时间 O(V^2) / 空间 O(V^2)。

    陷阱:
        1. 本函数**丢掉方向**：有向图请直接用邻接字典，不要经过这里。
        2. 双向权重不同时取**较小值**，这是本模块的约定，不是"真实无向权重"。
        3. 平行边无法表示：字典输入取较小权重，矩阵输入以矩阵为准。
        4. 矩阵输入不会检查对称性，非对称矩阵会被静默对称化。

    参考:
        图的无向化处理见任意图论教材；本模块内部约定。
    """
    if isinstance(adj, Mapping):
        nodes = _ordered_nodes(adj)
        n = len(nodes)
        W = np.full((n, n), np.inf)
        idx = {node: i for i, node in enumerate(nodes)}
        for u, nbrs in adj.items():
            for v, w in nbrs.items():
                if u == v:
                    continue
                iu, iv = idx[u], idx[v]
                W[iu, iv] = min(W[iu, iv], float(w))
        np.fill_diagonal(W, 0.0)
        W = np.minimum(W, W.T)
    else:
        W = np.asarray(adj, dtype=float)
        if W.ndim == 1:
            W = W.reshape(1, -1)
        if W.ndim != 2:
            raise ValueError(f"adj 必须是二维数组，得到 ndim={W.ndim}")
        check_square(W, "adj")
        if np.isnan(W).any():
            raise ValueError("adj 含 NaN：无边请用 inf 表示，不要用 NaN")
        W = np.array(W, dtype=float, copy=True)
        np.fill_diagonal(W, 0.0)
        W = np.minimum(W, W.T)
        nodes = list(range(W.shape[0]))
    return W, nodes


# --------------------------------------------------------------------------- #
# 生成树：Prim（邻接矩阵版本）
# --------------------------------------------------------------------------- #
def prim_mst(
    adj: Union[np.ndarray, Sequence[Sequence[float]], Mapping[Node, Mapping[Node, float]]],
) -> dict:
    """邻接矩阵（``inf`` 表示无边）上的 Prim 最小生成树。

    参数:
        adj: ``n x n`` 邻接矩阵，``inf`` 表示无边（对角线视为 0）；也接受 ``{u: {v: w}}``
        邻接字典，此时节点按下标 0,1,2,... 对齐（顺序由 :func:`_ordered_nodes` 决定：
        可比较时取 ``sorted``，否则退化为扫描顺序），并按**无向**处理（见
        :func:`_weight_matrix`）。

    返回:
        dict，键 ``"edges"``：list，每项是 ``[u, v, w]``，端点是 :func:`_weight_matrix`
        给出的**整数下标**（字典输入时请用同一顺序解读）；``"total_weight"``：float，总权重。

    算法:
        堆优化 Prim：从下标 0 出发，用二叉堆维护"已入树集合的横切边"，每次取最小边并入
        新节点并松弛其邻边。

    复杂度:
        时间 O(E log E)（E 为有效边数，等价 O(E log V)）/ 空间 O(V + E)。

    陷阱:
        1. 与 :func:`kruskal_mst` 的输入格式不同（这里是矩阵），但同一张图上两者总权重必须
           一致；本模块自测做了这项交叉验证。
        2. 图不连通时 ``raise ValueError``，不会返回"最小生成森林"。
        3. 返回的端点是下标而非原节点名；矩阵输入时下标即原下标。
        4. 负权边不会被拒绝（Prim 对负权仍然正确，只是"最小"含义要自己确认）。

    参考:
        Prim 1957；堆优化实现见 CLRS 第 23 章。
    """
    W, _nodes = _weight_matrix(adj)
    n = W.shape[0]
    if n == 0:
        return {"edges": [], "total_weight": 0.0}

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    heap: List[Tuple[float, int, int]] = []
    for v in range(1, n):
        w = float(W[0, v])
        if np.isfinite(w):
            heapq.heappush(heap, (w, 0, v))

    edges: List[List[float]] = []
    total = 0.0
    while heap and len(edges) < n - 1:
        w, u, v = heapq.heappop(heap)
        if in_tree[v]:
            continue
        in_tree[v] = True
        edges.append([int(u), int(v), float(w)])
        total += float(w)
        for x in range(n):
            if in_tree[x]:
                continue
            wx = float(W[v, x])
            if np.isfinite(wx):
                heapq.heappush(heap, (wx, int(v), int(x)))
    if len(edges) != n - 1:
        raise ValueError("图不连通：无法构造生成树（请检查 inf 的位置）")
    return {"edges": edges, "total_weight": float(total)}


# --------------------------------------------------------------------------- #
# 中心性
# --------------------------------------------------------------------------- #
def degree_centrality(
    adj: Union[np.ndarray, Sequence[Sequence[float]], Mapping[Node, Mapping[Node, float]]],
    weighted: bool = True,
) -> dict:
    """度中心性（按**无向**图计算），带权时用强度归一化。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典或 ``n x n`` 邻接矩阵（``inf`` 表示无边）。
        weighted: ``False`` 用"不同邻居个数 / (n-1)"；``True`` 用"相邻权重之和 / 最大权重和"。

    返回:
        dict，键 ``"centrality"``：``{节点: float}``（取值 ``[0, 1]``）；
        ``"ranking"``：list，节点按中心性**降序**排列，同分时按 :func:`_weight_matrix`
        的节点顺序稳定排列。

    算法:
        一次扫描邻接矩阵求每行权重和（或非零邻居计数），再做标量归一化；
        排序是 O(V log V) 的稳定排序。

    复杂度:
        时间 O(V^2)（矩阵运算）/ 空间 O(V^2)（含输入的矩阵副本）。

    陷阱:
        1. 方向被忽略（有向图的出/入度中心性请自己按行/列统计）。
        2. ``weighted=True`` 归一化用的是**最大强度**，不是 ``(n-1)*max_w``；因此中心点
           在星形图上恰好为 1，而"绝对强度"没有单位意义。
        3. 权重为负时可能让强度为 0 甚至为负，此时分母取最大值仍可运行，但解释失效。
        4. 返回字典的键是原节点，若节点不可 JSON 序列化（如 tuple）不能直接写进黄金值。

    参考:
        Freeman 1978《Centrality in social networks》。
    """
    W, nodes = _weight_matrix(adj)
    n = len(nodes)
    if n == 0:
        return {"centrality": {}, "ranking": []}
    finite = np.where(np.isfinite(W), W, 0.0)
    if weighted:
        vals = finite.sum(axis=1).astype(float)
        mx = float(vals.max()) if n > 0 else 0.0
        cent = vals / mx if mx > 0 else np.zeros(n)
    else:
        cnt = (finite != 0).sum(axis=1).astype(float)
        cent = cnt / (n - 1) if n > 1 else np.zeros(n)
    centrality = {nodes[i]: float(cent[i]) for i in range(n)}
    order = sorted(range(n), key=lambda i: (-float(cent[i]), i))
    return {"centrality": centrality, "ranking": [nodes[i] for i in order]}


def closeness_centrality(adj: Mapping[Node, Mapping[Node, float]]) -> dict:
    """紧密中心性：``可达点数 / 可达最短路长度之和``（Dijkstra 全源）。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典，边权必须非负（直接交给 :func:`dijkstra`）。

    返回:
        dict，键 ``"centrality"``：``{节点: float}``；``"ranking"``：list，按中心性降序、
        同分时按 :func:`_ordered_nodes` 顺序稳定排列。

    算法:
        以每个节点为源跑一次 Dijkstra，统计**可达**（有限距离）节点的个数与距离和，
        中心性 = 可达点数 / 距离和；没有可达点时取 0。

    复杂度:
        时间 O(V * (E log V)) / 空间 O(V + E)。

    陷阱:
        1. **不可达对既不计入分子也不计入分母**（不是"距离记 0"，也不是"记无穷大"）。
           因此孤立点中心性为 0，而"只能到达很少但很近的点"的节点中心性可能偏高
           ——这是本实现的约定，与"只用最大连通分量计算"的教科书写法不同。
        2. 有向图会得到非对称结果：谁都能到的节点中心性高，别当成无向图的结论。
        3. 中心性的量纲是 1/距离，不要跨算例比较绝对值。

    参考:
        Bavelas 1950；Freeman 1978；不可达处理见 Wasserman & Faust 1994。
    """
    if not isinstance(adj, Mapping):
        raise ValueError("adj 必须是 {u: {v: w}} 形式的字典")
    nodes = _ordered_nodes(adj)
    n = len(nodes)
    if n == 0:
        return {"centrality": {}, "ranking": []}

    cent: Dict[Node, float] = {}
    for s in nodes:
        dist = dijkstra(adj, s)["dist"]
        reach = 0
        total = 0.0
        for t in nodes:
            if t == s:
                continue
            d = float(dist.get(t, float("inf")))
            if np.isfinite(d):
                reach += 1
                total += d
        cent[s] = float(reach / total) if total > 0 else 0.0
    order = sorted(nodes, key=lambda v: (-cent[v], nodes.index(v)))
    return {"centrality": cent, "ranking": order}


def betweenness_centrality(adj: Mapping[Node, Mapping[Node, float]]) -> dict:
    """介数中心性（Brandes 算法，按**无权**最短路计数，无向图口径）。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典；边权和方向都**被忽略**（自环忽略，重复边去重）。

    返回:
        dict，键 ``"centrality"``：``{节点: float}``，已用 ``C(n-1, 2)`` 归一化，
        取值 ``[0, 1]``；``"ranking"``：list，按介数降序、同分按 :func:`_ordered_nodes`
        顺序稳定排列。

    算法:
        Brandes 2001：以每个节点为源做 BFS，记录最短路上溯节点与最短路条数，再逆序累加
        依赖量；无向图每对点被数了两次，故除以 2，最后除以组合数 ``C(n-1, 2)``。

    复杂度:
        时间 O(V * E) / 空间 O(V + E)。

    陷阱:
        1. 只按**无权**（每条边算 1 跳）计最短路；带权图的介数（用边权求最短路）需要另写。
        2. 归一化分母是 ``C(n-1, 2) = (n-1)(n-2)/2``，n < 3 时全部取 0（不报错）。
        3. "只有中间点非零"是常见误记：路径图 P5 上两端点为 0 但次中间点也非零（各 0.5）。
        4. 有向图必须另版实现（无向口径会把反向路径也算进依赖量）。

    参考:
        Brandes 2001《A faster algorithm for betweenness centrality》。
    """
    if not isinstance(adj, Mapping):
        raise ValueError("adj 必须是 {u: {v: w}} 形式的字典")
    nodes = _ordered_nodes(adj)
    n = len(nodes)
    if n == 0:
        return {"centrality": {}, "ranking": []}
    idx = {node: i for i, node in enumerate(nodes)}
    nb: List[List[int]] = [[] for _ in range(n)]
    seen = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u == v:
                continue
            a, b = idx[u], idx[v]
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            nb[a].append(int(b))
            nb[b].append(int(a))

    bc = np.zeros(n)
    for s in range(n):
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        depth = np.full(n, -1, dtype=int)
        depth[s] = 0
        queue = deque([s])
        while queue:
            v = int(queue.popleft())
            stack.append(v)
            for w in nb[v]:
                if depth[w] < 0:
                    depth[w] = depth[v] + 1
                    queue.append(w)
                if depth[w] == depth[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    bc = bc / 2.0  # 无向图：每对无序点被 BFS 数了两次
    if n > 2:
        bc = bc / ((n - 1) * (n - 2) / 2.0)
    else:
        bc = np.zeros(n)
    centrality = {nodes[i]: float(bc[i]) for i in range(n)}
    order = sorted(range(n), key=lambda i: (-float(bc[i]), i))
    return {"centrality": centrality, "ranking": [nodes[i] for i in order]}


# --------------------------------------------------------------------------- #
# 社区发现：单层 Louvain
# --------------------------------------------------------------------------- #
def louvain_communities(
    adj: Union[np.ndarray, Sequence[Sequence[float]], Mapping[Node, Mapping[Node, float]]],
    resolution: float = 1.0,
    seed: Optional[int] = None,
    max_iter: int = 20,
) -> dict:
    """单层 Louvain：模块度增益贪心 + 局部移动（不做社区聚合的递归）。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典或 ``n x n`` 邻接矩阵（``inf`` 表示无边）；
        边权即无向权重（负权按 0 处理）。
        resolution: 分辨率参数 ``gamma > 0``，越大社区越多。
        seed: 随机种子；``None`` 时用 ``_common.rng(None)`` 的库默认种子。
        max_iter: 局部移动的最大轮数（至少 1）。

    返回:
        dict，键 ``"labels"``：list of int，长度 = 节点数，按 :func:`_weight_matrix` 的
        节点顺序排列，社区编号从 0 开始且按首次出现顺序紧凑编号；``"modularity"``：
        float，最终划分在给定分辨率下的模块度；``"n_communities"``：int，社区个数。

    算法:
        初始每个节点自成一社区，按随机顺序遍历节点，对每个节点试算"移动到某个邻居社区"
        的模块度增量 ``(k_i_in_D - k_i_in_C)/m - gamma * k_i * (d_D - d_C') / (2 m^2)``，
        取增益最大的正增益移动；一轮无移动或达到 ``max_iter`` 后停止。模块度按闭式
        ``sum_C [ l_C/m - gamma * (d_C/(2m))^2 ]`` 独立重算（不是把增量累加）。

    复杂度:
        时间 O(max_iter * E) / 空间 O(V + E)。

    陷阱:
        1. 只有**一层**局部移动，没有把社区收缩成超点再迭代，所以在大图上质量低于完整
           Louvain（这也是自测里只保证"两个团 + 桥"这类小算例分对的原因）。
        2. 结果依赖节点遍历顺序：换 ``seed`` 可能得到不同划分（模块度也可能不同），
           自测只断言同 seed 可复现。
        3. 单层贪心**不保证全局最优**，模块度可能停在局部极大（例如先吞下桥端点）。
        4. 无向、无权重的自环与负权不做支持：自环丢弃，负权截断为 0。

    参考:
        Blondel et al. 2008《Fast unfolding of communities in large networks》。
    """
    if resolution <= 0:
        raise ValueError("resolution 必须为正数")
    if max_iter < 1:
        raise ValueError("max_iter 必须 >= 1")
    W, nodes = _weight_matrix(adj)
    n = len(nodes)
    if n == 0:
        return {"labels": [], "modularity": 0.0, "n_communities": 0}

    Wf = np.where(np.isfinite(W), W, 0.0)
    Wf = np.where(Wf > 0, Wf, 0.0)
    np.fill_diagonal(Wf, 0.0)
    deg = Wf.sum(axis=1)
    m2 = float(Wf.sum())  # = 2m
    m = m2 / 2.0
    if m <= 0:
        return {"labels": list(range(n)), "modularity": 0.0, "n_communities": n}

    labels = np.arange(n, dtype=int)
    tot = deg.astype(float).copy()
    gen = make_rng(seed)
    for _round in range(max_iter):
        order = gen.permutation(n)
        moved = 0
        for i_raw in order:
            i = int(i_raw)
            ci = int(labels[i])
            k_i = float(deg[i])
            nbr_com: Dict[int, float] = {}
            row = Wf[i]
            for j_raw in np.nonzero(row)[0]:
                j = int(j_raw)
                cj = int(labels[j])
                nbr_com[cj] = nbr_com.get(cj, 0.0) + float(row[j])
            tot[ci] -= k_i
            d_ci_after = float(tot[ci])
            k_in_ci = nbr_com.get(ci, 0.0)
            best_c = ci
            best_gain = 0.0
            for c, k_in_c in nbr_com.items():
                if c == ci:
                    continue
                d_c = float(tot[c])
                gain = (k_in_c - k_in_ci) / m - resolution * k_i * (d_c - d_ci_after) / (2.0 * m * m)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_c = int(c)
            if best_c != ci:
                labels[i] = best_c
                tot[best_c] += k_i
                moved += 1
            else:
                tot[ci] += k_i
        if moved == 0:
            break

    remap: Dict[int, int] = {}
    out_labels: List[int] = []
    for i in range(n):
        c = int(labels[i])
        if c not in remap:
            remap[c] = len(remap)
        out_labels.append(int(remap[c]))

    modularity = 0.0
    for c in range(len(remap)):
        members = np.nonzero(np.asarray(out_labels) == c)[0]
        sub = Wf[np.ix_(members, members)]
        l_c = float(sub.sum()) / 2.0
        d_c = float(deg[members].sum())
        modularity += l_c / m - resolution * (d_c / m2) ** 2
    return {
        "labels": [int(x) for x in out_labels],
        "modularity": float(modularity),
        "n_communities": int(len(remap)),
    }


# --------------------------------------------------------------------------- #
# 最小费用流
# --------------------------------------------------------------------------- #
def min_cost_flow(
    cost: Union[np.ndarray, Sequence[Sequence[float]]],
    capacity: Union[np.ndarray, Sequence[Sequence[float]]],
    source: int,
    sink: int,
    demand: float,
) -> dict:
    """连续最短路（SSP）增广的最小费用流。

    参数:
        cost: ``n x n`` 费用矩阵，``cost[u][v]`` 是 ``u -> v`` 的单位费用。
        capacity: ``n x n`` 容量矩阵，``capacity[u][v]`` 是 ``u -> v`` 的容量上限。
        source: 源点下标；sink: 汇点下标（二者不能相同）。
        demand: 需要从 source 送到 sink 的总流量（正数）。

    返回:
        dict，键 ``"flow"``：float，实际送达流量（一定等于 ``demand``，否则报错）；
        ``"cost"``：float，总费用；``"edge_flow"``：list，每项 ``[u, v, f]``，
        只列出 ``f > 0`` 的**原始方向**边，按 ``(u, v)`` 升序。

    算法:
        SSP（successive shortest path）：每次在残量网络上用 Bellman-Ford 找单位费用最小的
        增广路，沿路推进"剩余需求 与 路上最小残量"的瓶颈量；残量反向边的费用取 ``-cost``
        （用"流量矩阵可为负"的技巧表示反向流）。

    复杂度:
        时间 O(A * V * E)（A 为增广次数，每次 Bellman-Ford 为 O(VE)）/ 空间 O(V^2)。

    陷阱:
        1. 容量不足时 ``raise ValueError``，**不会**静默返回"能满足多少算多少"的部分流；
           只想要最大流请用 :func:`max_flow_edmonds_karp`。
        2. 依赖 `capacity - flow` 表示残量，因此 ``u->v`` 与 ``v->u`` 不能同时有独立容量，
           否则两条管道会被混成同一条（请把双向边拆成虚拟节点）。
        3. 原图存在负费用环时 SSP 不保证最优（Bellman-Ford 只跑 V-1 轮，也不报错）。
        4. 费用与容量都按 float 处理，大整数需求会有浮点误差；比较时请留容差。

    参考:
        Ahuja, Magnanti & Orlin 1993《Network Flows》第 9 章；
        Bellman-Ford 增广的实现即 SSP。
    """
    C = np.array(as_matrix(cost, "cost"), dtype=float, copy=True)
    K = np.array(as_matrix(capacity, "capacity"), dtype=float, copy=True)
    check_square(C, "cost")
    check_square(K, "capacity")
    if C.shape != K.shape:
        raise ValueError("cost 与 capacity 的形状必须一致")
    n = C.shape[0]
    if not (0 <= int(source) < n) or not (0 <= int(sink) < n):
        raise ValueError("source / sink 下标越界")
    if int(source) == int(sink):
        raise ValueError("source 与 sink 不能是同一个节点")
    if not np.isfinite(demand) or demand <= 0:
        raise ValueError("demand 必须是正的有限数")
    if np.any(K < 0):
        raise ValueError("capacity 不能为负")
    if not np.all(np.isfinite(C)):
        raise ValueError("cost 必须全部有限（无边请用容量 0 表示）")

    s = int(source)
    t = int(sink)
    flow = np.zeros((n, n))
    total_cost = 0.0
    total_flow = 0.0
    while total_flow < demand - 1e-12:
        dist = np.full(n, np.inf)
        dist[s] = 0.0
        prev = np.full(n, -1, dtype=int)
        for _ in range(n - 1):
            updated = False
            for u in range(n):
                if not np.isfinite(dist[u]):
                    continue
                for v in range(n):
                    if u == v:
                        continue
                    if K[u, v] - flow[u, v] > 1e-12:
                        nd = dist[u] + C[u, v]
                        if nd < dist[v] - 1e-12:
                            dist[v] = nd
                            prev[v] = u
                            updated = True
            if not updated:
                break
        if not np.isfinite(dist[t]):
            raise ValueError("容量不足：在给定 capacity 下无法满足 demand")
        add = float(demand - total_flow)
        v = t
        while v != s:
            u = int(prev[v])
            add = min(add, float(K[u, v] - flow[u, v]))
            v = u
        if add <= 1e-12:
            raise ValueError("增广瓶颈为 0，残量网络异常（请检查 cost/capacity）")
        v = t
        while v != s:
            u = int(prev[v])
            flow[u, v] += add
            flow[v, u] -= add
            total_cost += add * float(C[u, v])
            v = u
        total_flow += add

    edge_flow = [
        [int(u), int(v), float(flow[u, v])]
        for u in range(n)
        for v in range(n)
        if flow[u, v] > 1e-12
    ]
    return {"flow": float(total_flow), "cost": float(total_cost), "edge_flow": edge_flow}


# --------------------------------------------------------------------------- #
# 二分图最大权匹配
# --------------------------------------------------------------------------- #
def _hungarian_min(a: np.ndarray) -> Tuple[float, List[int]]:
    """匈牙利算法（e-maxx 的 O(n^2 m) 势函数版本）求最小费用**完备**匹配。

    参数:
        a: ``n x m`` 有限实数矩阵，且必须 ``n <= m``（每行都能配上列）。

    返回:
        ``(总费用, assign)``：``assign[i]`` 是第 i 行匹配到的列下标（长度为 n）。

    算法:
        逐行加入，维护行势 ``u`` 与列势 ``v``，在"未用列"上用松弛量 ``minv`` 找增广列，
        沿 ``way`` 回溯改写匹配。

    复杂度:
        时间 O(n^2 * m) / 空间 O(n + m)。

    陷阱:
        1. 要求 ``n <= m`` 且矩阵有限，否则行为未定义（调用方需自行转置 / 检查）。
        2. 只求"行数那么多个匹配"（小边侧的完备匹配），不做"允许空匹配"的松弛。

    参考:
        Kuhn 1955；Jonker-Volgenant 1987；e-maxx "Hungarian algorithm" 实现。
    """
    n, m = a.shape
    if n > m:
        raise ValueError("匈牙利算法要求行数 <= 列数")
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = float(a[i0 - 1, j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == 0:
                raise ValueError("匈牙利算法失败：矩阵含无效值")
            for j in range(m + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    assign = [0] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assign[int(p[j]) - 1] = j - 1
    total = sum(float(a[i, assign[i]]) for i in range(n))
    return float(total), assign


def bipartite_matching(
    cost: Union[np.ndarray, Sequence[Sequence[float]]],
) -> dict:
    """二分图最大权匹配（直接实现增广路 / 匈牙利，不依赖外部优化库）。

    参数:
        cost: ``rows x cols`` 权重矩阵，``cost[i][j]`` 是左部 i 与右部 j 匹配的收益
        （可正可负，但必须全部有限）。

    返回:
        dict，键 ``"matching"``：list，每项 ``[i, j]``（按 i 升序）；``"total_cost"``：
        float，匹配对的收益之和；``"size"``：int，匹配边数 ``= min(rows, cols)``。

    算法:
        最大化收益等价于最小化 ``-cost``：把矩阵取负后用 :func:`_hungarian_min` 求小边侧的
        完备匹配（行数多于列数时先转置），再还原原始下标。

    复杂度:
        时间 O(min(r, c)^2 * max(r, c)) / 空间 O(r * c)。

    陷阱:
        1. 返回的是**小边侧的完备匹配**（``size = min(rows, cols)``），零收益甚至负收益的
           边也会被选上；若只想保留正收益边，请自己按 ``total_cost`` 过滤。
        2. 权重必须有限；"禁止匹配"不能写 ``inf``，请用足够小的负数（如 ``-1e9``）。
        3. 收益矩阵被当作**权重**而不是费用；若你的矩阵是成本，请传 ``-cost`` 并读
           ``-total_cost``。
        4. 不支持一对多/多对一（那不是匹配问题，要建流网络）。

    参考:
        Kuhn 1955；Munkres 1957；最大权匹配的理论见 Schrijver 2003。
    """
    M = np.array(as_matrix(cost, "cost"), dtype=float, copy=True)
    if M.ndim != 2:
        raise ValueError("cost 必须是二维矩阵")
    if M.size == 0:
        return {"matching": [], "total_cost": 0.0, "size": 0}
    if not np.all(np.isfinite(M)):
        raise ValueError("cost 必须全部有限（禁止匹配请用很小的负数）")
    rows, cols = M.shape
    if rows <= cols:
        neg_total, assign = _hungarian_min(-M)
        pairs = [[int(i), int(assign[i])] for i in range(rows)]
    else:
        neg_total, assign = _hungarian_min(-M.T)
        pairs = [[int(assign[k]), int(k)] for k in range(cols)]
    pairs.sort()
    total = sum(float(M[i, j]) for i, j in pairs)
    return {
        "matching": pairs,
        "total_cost": float(total),
        "size": int(len(pairs)),
    }


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #
def a_star(
    adj: Mapping[Node, Mapping[Node, float]],
    start: Node,
    goal: Node,
    heuristic: Union[Mapping[Node, float], Callable[[Node], float]],
) -> dict:
    """A* 最短路：启发式可采纳时与 Dijkstra 结果相同。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典，边权必须非负。
        start: 起点；goal: 终点。
        heuristic: ``{节点: 估计值}`` 字典或 ``f(节点) -> float`` 可调用对象；
        缺省节点按 0 处理（即退化为 Dijkstra）。

    返回:
        dict，键 ``"path"``：list，从 start 到 goal 的节点序列（不可达时为 ``[]``）；
        ``"cost"``：float，路径总代价（不可达时为 ``inf``）；``"expanded"``：int，
        从优先队列中真正弹出的节点个数（衡量搜索规模）。

    算法:
        标准 A*：``f = g + h``，用二叉堆取最小 f；节点出堆时才判定是否扩展（配合"同一节点
        允许多次入堆、取最优 g"），遇到 goal 立即返回。

    复杂度:
        时间 O(E log V)（最坏退化为 Dijkstra）/ 空间 O(V + E)。

    陷阱:
        1. 启发式必须**可采纳**（``h <= 真实剩余代价``）；不可采纳时返回的 cost 可能大于真实
           最短路，且不会报错。这里只检查"非负且有限"，查不出可采纳性。
        2. 用"出堆即闭合"的写法，配合**不一致**（inconsistent）启发式时可能失去最优性；
           要保证最优请用一致启发式，或允许重新打开已闭合节点。
        3. 只支持非负边权（负权请用 Bellman-Ford）；一旦在扩展中读到负权边就抛 ValueError，
           但**只遍历到的边**会被检查，不可达部分的负权边不会被发现。
        4. ``heuristic`` 用可调用对象时会对每个入堆邻居求值，代价高请先做成字典。
        5. ``start`` 或 ``goal`` **不在图中**时直接抛 ``ValueError``（不会返回
           "不可达"）；只有两者都在图中但确实无路时才返回 ``path=[]``、``cost=inf``。
        6. ``heuristic`` 若对任一节点给出负值或非有限值，会在搜索前就抛 ``ValueError``
           （本实现会先对全部节点求一遍 ``h`` 做校验）。

    参考:
        Hart, Nilsson & Raphael 1968《A Formal Basis for the Heuristic Determination of
        Minimum Cost Paths》。
    """
    if not isinstance(adj, Mapping):
        raise ValueError("adj 必须是 {u: {v: w}} 形式的字典")
    nodes = _ordered_nodes(adj)
    if start not in set(nodes):
        raise ValueError("start 不在图中")
    if goal not in set(nodes):
        raise ValueError("goal 不在图中")
    if not isinstance(heuristic, Mapping) and not callable(heuristic):
        raise ValueError("heuristic 必须是字典或可调用对象")

    def h(v: Node) -> float:
        val = heuristic(v) if callable(heuristic) else heuristic.get(v, 0.0)
        out = float(val)
        if not np.isfinite(out) or out < 0:
            raise ValueError("启发式必须是非负有限值")
        return out

    for node in nodes:
        h(node)

    counter = itertools.count()
    g: Dict[Node, float] = {start: 0.0}
    prev: Dict[Node, Optional[Node]] = {start: None}
    heap: List[Tuple[float, int, Node]] = [(h(start), next(counter), start)]
    closed = set()
    expanded = 0
    while heap:
        _f, _tie, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == goal:
            break
        for v, w in adj.get(u, {}).items():
            wv = float(w)
            if wv < 0:
                raise ValueError("A* 只支持非负边权")
            ng = g[u] + wv
            if ng < g.get(v, float("inf")) - 1e-15:
                g[v] = ng
                prev[v] = u
                heapq.heappush(heap, (ng + h(v), next(counter), v))

    if goal not in g:
        return {"path": [], "cost": float("inf"), "expanded": int(expanded)}
    if goal == start:
        return {"path": [start], "cost": 0.0, "expanded": int(expanded)}
    path = reconstruct_path(prev, start, goal)
    return {"path": path, "cost": float(g[goal]), "expanded": int(expanded)}


# --------------------------------------------------------------------------- #
# 车辆路径：Clarke-Wright 节约算法
# --------------------------------------------------------------------------- #
def vrp_clarke_wright(
    distance: Union[np.ndarray, Sequence[Sequence[float]]],
    demand: Sequence[float],
    capacity: float,
    depot: int = 0,
) -> dict:
    """Clarke-Wright 节约算法求解带容量约束的车辆路径问题（CVRP，送货型）。

    参数:
        distance: ``n x n`` 距离矩阵（**假设对称**，对角线视为 0）。
        demand: 长度 n 的需求序列，``demand[depot]`` 被忽略。
        capacity: 单车容量上限（正数）。
        depot: 车场下标，默认 0。

    返回:
        dict，键 ``"routes"``：list，每条路线是 ``[depot, 客户..., depot]``（元素为下标，
        路线之间按首个客户下标升序排列）；``"total_distance"``：float，所有路线长度之和；
        ``"n_routes"``：int，路线条数。

    算法:
        节约法：先给每个客户一条 ``depot -> i -> depot`` 的独立路线，计算节约值
        ``s(i, j) = d(depot, i) + d(depot, j) - d(i, j)``，按节约值降序尝试把两条路线的
        **端点**客户 i、j 合并（要求容量之和不超过 capacity，且 i、j 都在各自路线端点），
        直到没有可合并的节约。

    复杂度:
        时间 O(n^2 log n)（节约值排序主导）/ 空间 O(n^2)。

    陷阱:
        1. 只做"端点合并"，不做 2-opt / Or-opt 改进，因此结果一般不是最优解（Clarke-Wright
           本身是启发式，只保证可行）。
        2. 距离矩阵假定对称；非对称矩阵上路径长度计算会静默偏小（回程按正向元素取）。
        3. 单个客户需求超过 capacity 时直接 ``raise ValueError``，不返回不可行路线。
        4. 节约值相同的手工比较顺序由 ``(-s, i, j)`` 决定，是有意为之的确定性 tie-break；
           不同实现（不同 tie-break）会给出不同但同样可行的解。

    参考:
        Clarke & Wright 1964《Scheduling of vehicles from a central depot to a number of
        delivery points》。
    """
    D = np.array(as_matrix(distance, "distance"), dtype=float, copy=True)
    check_square(D, "distance")
    n = D.shape[0]
    if not (0 <= int(depot) < n):
        raise ValueError("depot 下标越界")
    dem = np.asarray(demand, dtype=float).reshape(-1)
    if dem.size != n:
        raise ValueError("demand 的长度必须等于距离矩阵的阶数")
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity 必须是正的有限数")
    if not np.all(np.isfinite(D)):
        raise ValueError("distance 必须全部有限")

    dep = int(depot)
    customers = [i for i in range(n) if i != dep]
    for i in customers:
        if dem[i] > capacity + 1e-9:
            raise ValueError("存在客户需求超过单车容量，问题不可行")
        if dem[i] < 0:
            raise ValueError("demand 不能为负")

    routes: Dict[int, List[int]] = {i: [dep, i, dep] for i in customers}
    load: Dict[int, float] = {i: float(dem[i]) for i in customers}
    route_of: Dict[int, int] = {i: i for i in customers}

    savings: List[Tuple[float, int, int]] = []
    for a in range(len(customers)):
        for b in range(a + 1, len(customers)):
            i, j = customers[a], customers[b]
            s = float(D[dep, i]) + float(D[dep, j]) - float(D[i, j])
            savings.append((-s, int(i), int(j)))
    savings.sort()

    for _neg_s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        if load[ri] + load[rj] > capacity + 1e-9:
            continue
        seq_i = routes[ri][1:-1]
        seq_j = routes[rj][1:-1]
        if not seq_i or not seq_j:
            continue
        if i != seq_i[0] and i != seq_i[-1]:
            continue
        if j != seq_j[0] and j != seq_j[-1]:
            continue
        a_seq = list(seq_i)
        b_seq = list(seq_j)
        if a_seq[0] == i:  # 让 i 落在 a_seq 末端（对称距离下反转不改变长度）
            a_seq.reverse()
        if b_seq[-1] == j:  # 让 j 落在 b_seq 首端
            b_seq.reverse()
        routes[ri] = [dep] + a_seq + b_seq + [dep]
        load[ri] = load[ri] + load[rj]
        for c in b_seq:
            route_of[c] = ri
        del routes[rj]
        del load[rj]

    out_routes = sorted(routes.values(), key=lambda r: r[1] if len(r) > 2 else -1)
    total = 0.0
    for r in out_routes:
        for k in range(len(r) - 1):
            total += float(D[r[k], r[k + 1]])
    return {
        "routes": [[int(x) for x in r] for r in out_routes],
        "total_distance": float(total),
        "n_routes": int(len(out_routes)),
    }


# --------------------------------------------------------------------------- #
# 网络鲁棒性
# --------------------------------------------------------------------------- #
def network_robustness(
    adj: Union[np.ndarray, Sequence[Sequence[float]], Mapping[Node, Mapping[Node, float]]],
    n_remove: int = 3,
) -> dict:
    """依次移除度最大的节点，评估剩余网络的连通性与效率。

    参数:
        adj: ``{u: {v: w}}`` 邻接字典或 ``n x n`` 邻接矩阵（``inf`` 表示无边）；
        按**无向无权**图处理（只关心"有没有边"）。
        n_remove: 要移除的节点个数（非负；最多移除到只剩 1 个节点）。

    返回:
        dict，键 ``"largest_component"``：int，**移除结束后**剩余图的最大连通分量规模
        （单个孤立点算 1；没有剩余节点时为 0）；``"efficiency"``：float，剩余图的全局效率
        ``1/(m(m-1)) * sum_{i != j} 1/d(i,j)``（d 为**跳数**，不可达对贡献 0）；
        ``"removed"``：list，按移除顺序排列的节点。

    算法:
        每轮在剩余图中用"非零邻接计数"选度最大的节点（同分取 :func:`_weight_matrix` 顺序中
        靠前的），标记删除；全部移除结束后用一次 BFS 全源跳数统计最大连通分量与全局效率。

    复杂度:
        时间 O(n_remove * V^2 + V * (V + E)) / 空间 O(V + E)。

    陷阱:
        1. 移除顺序是**确定性贪心**（度最大 + 下标 tie-break），不是随机攻击；随机故障
           鲁棒性请自行打乱顺序重复实验。
        2. 效率用的是**跳数**而非边权，带权图上的结论可能完全不同。
        3. 返回的是"移除完毕之后"的指标，不含每步演化曲线；需要曲线请循环调用本函数。
        4. ``n_remove`` 过大时只会移除到剩 1 个节点，``removed`` 长度可能小于 n_remove。

    参考:
        Albert, Jeong & Barabási 2000《Error and attack tolerance of complex networks》；
        Latora & Marchiori 2001（全局效率定义）。
    """
    W, nodes = _weight_matrix(adj)
    n = len(nodes)
    if n_remove < 0:
        raise ValueError("n_remove 不能为负")
    if n == 0:
        return {"largest_component": 0, "efficiency": 0.0, "removed": []}

    adjm = np.isfinite(W) & (W != 0)
    act = np.ones(n, dtype=bool)
    removed_idx: List[int] = []
    while len(removed_idx) < int(n_remove) and int(act.sum()) > 1:
        deg = ((adjm & act[None, :]).sum(axis=1)).astype(float)
        deg = np.where(act, deg, -1.0)
        i = int(np.argmax(deg))
        act[i] = False
        removed_idx.append(i)

    idx = np.nonzero(act)[0]
    m = int(idx.size)
    if m == 0:
        return {"largest_component": 0, "efficiency": 0.0, "removed": [nodes[i] for i in removed_idx]}

    largest = 0
    reach_sum = 0.0
    for s_raw in idx:
        s = int(s_raw)
        dist = {s: 0}
        queue = deque([s])
        while queue:
            u = int(queue.popleft())
            for v_raw in np.nonzero(adjm[u] & act)[0]:
                v = int(v_raw)
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        largest = max(largest, len(dist))
        for t, d in dist.items():
            if t != s:
                reach_sum += 1.0 / float(d)
    efficiency = reach_sum / (m * (m - 1)) if m > 1 else 0.0
    return {
        "largest_component": int(largest),
        "efficiency": float(efficiency),
        "removed": [nodes[i] for i in removed_idx],
    }


def _self_test() -> dict:
    """跑一组小规模确定性算例，返回关键数值供 examples/run_algorithms.py 断言。

    返回:
        dict，键全部为简短 ASCII，值可复现（本模块无随机性，两次调用完全一致）。

    算法:
        混合覆盖：带权有向图最短路、Floyd 全源距离、Kruskal MST、
        最大流/最小割定理、TSP 两阶段、PageRank、连通分量。

    复杂度:
        时间 O(1)（固定小算例）/ 空间 O(1)。

    陷阱:
        本函数只用固定算例，**不含随机性**；若你改了算法实现而这里仍通过，
        请检查算例是否过于简单（例如 4 城 TSP 的最优解和最近邻解恰好相同）。

    参考:
        契约第 3 节"每个模块结尾提供 _self_test()"。
    """
    result: Dict[str, object] = {}

    # --- 最短路：5 节点带权有向图 ---
    adj = {
        "A": {"B": 1.0, "C": 4.0},
        "B": {"C": 2.0, "D": 5.0},
        "C": {"D": 1.0},
        "D": {"E": 3.0},
        "E": {},
    }
    dj = dijkstra(adj, "A")
    result["dij_a_e"] = float(dj["dist"]["E"])
    result["dij_path_a_e"] = reconstruct_path(dj["prev"], "A", "E")
    result["dij_unreach"] = reconstruct_path(dj["prev"], "E", "A")

    # --- Floyd：转成矩阵，验证与 Dijkstra 一致 ---
    names = ["A", "B", "C", "D", "E"]
    W = np.full((5, 5), np.inf)
    np.fill_diagonal(W, 0.0)
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            W[names.index(u), names.index(v)] = w
    fw = floyd_warshall(W)
    result["floyd_0_4"] = float(fw["dist"][0, 4])
    result["floyd_path_0_4"] = reconstruct_path(fw["next_node"], 0, 4)
    # 交叉验证：Floyd 全点对距离 vs 以每个点为源的 Dijkstra（同一张图），取最大绝对偏差
    max_dev = 0.0
    for si, s in enumerate(names):
        dj_s = dijkstra(adj, s)["dist"]
        for ti, t in enumerate(names):
            a, b = fw["dist"][si, ti], dj_s[t]
            both_finite = np.isfinite(a) and np.isfinite(b)
            if both_finite:
                max_dev = max(max_dev, abs(float(a) - float(b)))
            elif not (np.isinf(a) and np.isinf(b)):
                max_dev = float("inf")
    result["floyd_vs_dijkstra_max_dev"] = float(max_dev)

    # --- MST：教材图（节点 A..G），已知最小生成树权重 39 ---
    # 边集：(A,B,7)(A,D,5)(B,C,8)(B,D,9)(B,E,7)(C,E,5)(D,E,15)(D,F,6)(E,F,8)(E,G,9)(F,G,11)
    edges = [
        (0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7),
        (2, 4, 5), (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11),
    ]
    mst = kruskal_mst(7, edges)
    result["mst_weight"] = float(mst["total_weight"])
    result["mst_n_edges"] = len(mst["edges"])
    result["mst_edge0"] = [int(mst["edges"][0][0]), int(mst["edges"][0][1]), float(mst["edges"][0][2])]

    # --- 最大流 / 最小割：CLRS 经典例，最大流 23 ---
    cap = {
        "s": {"v1": 16.0, "v2": 13.0},
        "v1": {"v2": 10.0, "v3": 12.0},
        "v2": {"v1": 4.0, "v4": 14.0},
        "v3": {"v2": 9.0, "t": 20.0},
        "v4": {"v3": 7.0, "t": 4.0},
        "t": {},
    }
    mf = max_flow_edmonds_karp(cap, "s", "t")
    cut = min_cut_edges(cap, "s", "t")
    result["max_flow"] = float(mf["max_flow"])
    result["min_cut_cap"] = float(sum(c for _, _, c in cut))
    result["min_cut_n"] = len(cut)
    result["flow_s_v1"] = float(mf["flow"]["s"]["v1"])

    # --- TSP：10 城圆上均匀分布，最优 = 10 * 2*sin(pi/10)（最近邻恰好命中）---
    n_city = 10
    ang = 2 * np.pi * np.arange(n_city) / n_city
    pts = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    nn = tsp_nearest_neighbor(D, start=0)
    two = tsp_two_opt(nn["tour"], D)
    result["tsp_nn_len"] = float(nn["length"])
    result["tsp_2opt_len"] = float(two["length"])
    result["tsp_optimal_len"] = float(2 * n_city * np.sin(np.pi / n_city))

    # --- TSP 反例算例：8 城（坐标已四舍五入到 2 位小数，穷举最优值 26.484006）---
    # 该例同时体现"最近邻次优"和"2-opt 改进后仍非最优"两个真实现象。
    coords8 = np.array(
        [[0.82, 7.53], [5.79, 3.00], [0.78, 7.63], [1.31, 1.33],
         [1.31, 0.81], [9.06, 2.69], [3.06, 8.33], [6.20, 1.87]]
    )
    D8 = np.sqrt(((coords8[:, None, :] - coords8[None, :, :]) ** 2).sum(-1))
    nn8 = tsp_nearest_neighbor(D8, start=0)
    two8 = tsp_two_opt(nn8["tour"], D8)
    result["tsp8_nn_len"] = float(nn8["length"])
    result["tsp8_2opt_len"] = float(two8["length"])
    result["tsp8_brute_opt_len"] = 26.484006136241458
    result["tsp8_2opt_improved"] = bool(two8["improved"])

    # --- PageRank：3 节点环形图 + 悬挂点，无悬挂时对称图应均匀 ---
    web = {"a": {"b": 1.0}, "b": {"c": 1.0}, "c": {"a": 1.0}}
    pr = pagerank(web, damping=0.85, tol=1e-12, max_iter=500)
    result["pr_cycle_max_dev"] = float(np.abs(pr - 1.0 / 3.0).max())
    result["pr_sum"] = float(pr.sum())

    web2 = {"a": {"b": 1.0, "c": 1.0}, "b": {"c": 1.0}, "c": {}}
    pr2 = pagerank(web2, damping=0.85, tol=1e-12, max_iter=500)
    result["pr_dangling_sum"] = float(pr2.sum())
    result["pr_b_gt_a"] = bool(pr2[1] > pr2[0])

    # --- 连通分量：两条独立链 + 一个孤立点 ---
    g = {"0": {"1": 1.0}, "1": {"2": 1.0}, "2": {}, "3": {"4": 1.0}, "4": {}, "5": {}}
    labels = connected_components(g)
    result["cc_labels"] = [int(x) for x in labels]
    result["cc_count"] = int(max(labels) + 1)

    # ======================= 新增：10 个图算法的独立判据 ======================= #
    # --- Prim：与 Kruskal 在同一张 7 节点图上比较总权重（交叉验证）---
    W7 = np.full((7, 7), np.inf)
    np.fill_diagonal(W7, 0.0)
    for u, v, w in edges:
        W7[u, v] = min(W7[u, v], float(w))
        W7[v, u] = min(W7[v, u], float(w))
    prim = prim_mst(W7)
    result["graphs_prim_total_weight"] = float(prim["total_weight"])
    result["graphs_prim_n_edges"] = int(len(prim["edges"]))
    result["graphs_prim_vs_kruskal_dev"] = float(
        abs(float(prim["total_weight"]) - float(mst["total_weight"]))
    )
    if result["graphs_prim_vs_kruskal_dev"] > 1e-9:
        raise AssertionError("Prim 与 Kruskal 的总权重不一致，两者至少有一个错")
    if result["graphs_prim_n_edges"] != 6:
        raise AssertionError("生成树边数应为 V-1 = 6")
    # 树边总权重 = 逐边求和（不复用 total_weight，独立重算一遍）
    if abs(sum(float(e[2]) for e in prim["edges"]) - 39.0) > 1e-9:
        raise AssertionError("Prim 生成树逐边求和应等于教材值 39")

    # --- 度中心性：星形图闭式解（中心 1、叶子 1/(n-1)），带权按最大强度归一化 ---
    star = {0: {1: 2.0, 2: 4.0, 3: 6.0}, 1: {0: 2.0}, 2: {0: 4.0}, 3: {0: 6.0}}
    deg_u = degree_centrality(star, weighted=False)
    deg_w = degree_centrality(star, weighted=True)
    result["graphs_deg_star_center"] = float(deg_u["centrality"][0])
    result["graphs_deg_star_leaf"] = float(deg_u["centrality"][1])
    if abs(result["graphs_deg_star_center"] - 1.0) > 1e-12:
        raise AssertionError("无权星形图中心点度中心性应为 1")
    if abs(result["graphs_deg_star_leaf"] - 1.0 / 3.0) > 1e-12:
        raise AssertionError("无权星形图叶子度中心性应为 1/(n-1)=1/3")
    if abs(float(deg_w["centrality"][0]) - 1.0) > 1e-12:
        raise AssertionError("带权星形图中心点（强度 12）应为 1")
    if abs(float(deg_w["centrality"][3]) - 0.5) > 1e-12:
        raise AssertionError("带权星形图叶子强度 6/12 应为 0.5")
    if deg_u["ranking"][0] != 0:
        raise AssertionError("星形图 ranking 首位应是中心点")

    # --- 紧密中心性：路径图 P4 手算 3/(1+1+2)=0.75 与 3/(1+2+3)=0.5 ---
    p4 = {
        "0": {"1": 1.0},
        "1": {"0": 1.0, "2": 1.0},
        "2": {"1": 1.0, "3": 1.0},
        "3": {"2": 1.0},
    }
    clo = closeness_centrality(p4)
    result["graphs_closeness_p4_mid"] = float(clo["centrality"]["1"])
    result["graphs_closeness_p4_end"] = float(clo["centrality"]["0"])
    if abs(result["graphs_closeness_p4_mid"] - 0.75) > 1e-12:
        raise AssertionError("P4 中间点紧密中心性手算应为 0.75")
    if abs(result["graphs_closeness_p4_end"] - 0.5) > 1e-12:
        raise AssertionError("P4 端点紧密中心性手算应为 0.5")
    if not clo["centrality"]["1"] > clo["centrality"]["0"]:
        raise AssertionError("P4 上中间点中心性应大于端点")

    # --- 介数中心性：路径图 P5，中间点 = 2*2/C(4,2) = 2/3，两端点为 0 ---
    p5 = {str(i): {str(j): 1.0 for j in (i - 1, i + 1) if 0 <= j <= 4} for i in range(5)}
    btw = betweenness_centrality(p5)
    result["graphs_btw_p5_mid"] = float(btw["centrality"]["2"])
    ends = [float(btw["centrality"]["0"]), float(btw["centrality"]["4"])]
    result["graphs_btw_p5_ends_zero"] = bool(max(abs(x) for x in ends) < 1e-12)
    if abs(result["graphs_btw_p5_mid"] - 2.0 / 3.0) > 1e-12:
        raise AssertionError("P5 中间点介数应为 (2*2)/C(4,2)=2/3")
    if not result["graphs_btw_p5_ends_zero"]:
        raise AssertionError("P5 两端点介数应为 0")
    if not all(
        float(btw["centrality"]["2"]) > float(btw["centrality"][k]) for k in ("0", "1", "3", "4")
    ):
        raise AssertionError("P5 中间点应是唯一介数最大点")

    # --- Louvain：两个三角形 + 一条桥必须分到两个社区，且同 seed 可复现 ---
    tri = {i: {} for i in range(6)}
    for a_, b_ in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        tri[a_][b_] = 1.0
        tri[b_][a_] = 1.0
    lv = louvain_communities(tri, resolution=1.0, seed=0, max_iter=20)
    lv2 = louvain_communities(tri, resolution=1.0, seed=0, max_iter=20)
    result["graphs_louvain_n_communities"] = int(lv["n_communities"])
    result["graphs_louvain_modularity"] = float(lv["modularity"])
    result["graphs_louvain_same_seed_equal"] = bool(list(lv["labels"]) == list(lv2["labels"]))
    if result["graphs_louvain_n_communities"] != 2:
        raise AssertionError("两个团 + 一条桥应划分为 2 个社区")
    if not result["graphs_louvain_modularity"] > 0.3:
        raise AssertionError("该算例的模块度应大于 0.3")
    if not result["graphs_louvain_same_seed_equal"]:
        raise AssertionError("同一 seed 两次调用结果必须一致")
    lab = list(lv["labels"])
    if not (lab[0] == lab[1] == lab[2]) or not (lab[3] == lab[4] == lab[5]):
        raise AssertionError("每个团内部应同社区")
    if lab[0] == lab[3]:
        raise AssertionError("两个团不应被合并成一个社区")

    # --- 最小费用流：0->2 直达费用 5，0->1->2 费用 1+1，容量各 4，手算 4*2=8 ---
    mcf_cost = np.array([[0.0, 1.0, 5.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mcf_cap = np.array([[0.0, 4.0, 4.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    mcf = min_cost_flow(mcf_cost, mcf_cap, 0, 2, 4.0)
    result["graphs_mcf_flow"] = float(mcf["flow"])
    result["graphs_mcf_cost"] = float(mcf["cost"])
    if abs(result["graphs_mcf_flow"] - 4.0) > 1e-9:
        raise AssertionError("送达流量应等于 demand = 4")
    if abs(result["graphs_mcf_cost"] - 8.0) > 1e-9:
        raise AssertionError("手算最小费用应为 4*(1+1)=8（走 0->1->2）")
    net: Dict[int, float] = {}
    for u_, v_, f_ in mcf["edge_flow"]:
        net[u_] = net.get(u_, 0.0) + float(f_)
        net[v_] = net.get(v_, 0.0) - float(f_)
    result["graphs_mcf_conservation"] = bool(
        all(abs(val) < 1e-9 for key, val in net.items() if key not in (0, 2))
    )
    if not result["graphs_mcf_conservation"]:
        raise AssertionError("中间节点必须满足流量守恒（净流出为 0）")
    if abs(net.get(0, 0.0) - 4.0) > 1e-9:
        raise AssertionError("源点净流出应等于总流量 4")

    # --- 二分匹配：3x3 单位矩阵 + 穷举对照 + 2x2 手算 ---
    unit3 = np.ones((3, 3))
    bm = bipartite_matching(unit3)
    result["graphs_bip_unit_size"] = int(bm["size"])
    result["graphs_bip_unit_cost"] = float(bm["total_cost"])
    if result["graphs_bip_unit_size"] != 3 or abs(result["graphs_bip_unit_cost"] - 3.0) > 1e-12:
        raise AssertionError("3x3 单位权矩阵应给出 size=3, total_cost=3")
    # 2x2 反例：贪心先取最大元素 (0,0)=10 只能得 10+1=11，最优是 (0,1)+(1,0)=9+9=18
    m2 = np.array([[10.0, 9.0], [9.0, 1.0]])
    bm2 = bipartite_matching(m2)
    result["graphs_bip_2x2_cost"] = float(bm2["total_cost"])
    if abs(result["graphs_bip_2x2_cost"] - 18.0) > 1e-12:
        raise AssertionError("2x2 手算最大权应为 9+9=18（贪心取 10 只能得 11）")
    hard = np.array([[10.0, 2.0, 8.0], [9.0, 7.0, 5.0], [6.0, 4.0, 3.0]])
    bmh = bipartite_matching(hard)
    brute_max = max(
        sum(float(hard[i, perm[i]]) for i in range(3))
        for perm in itertools.permutations(range(3))
    )
    result["graphs_bip_brute_equal"] = bool(abs(float(bmh["total_cost"]) - brute_max) < 1e-9)
    if not result["graphs_bip_brute_equal"]:
        raise AssertionError("二分匹配结果应等于 3x3 穷举最优值")

    # --- A*：5x5 栅格（挖掉中心格），可采纳的曼哈顿启发式应与 Dijkstra 完全一致 ---
    side = 5
    grid: Dict[int, Dict[int, float]] = {r * side + c: {} for r in range(side) for c in range(side)}
    for r in range(side):
        for c in range(side):
            if (r, c) == (2, 2):
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) != (2, 2):
                    grid[r * side + c][nr * side + nc] = 1.0
    manhattan = {node: float(abs(node // side - 4) + abs(node % side - 4)) for node in grid}
    ast = a_star(grid, 0, 24, manhattan)
    ast_zero = a_star(grid, 0, 24, {})
    djg = dijkstra(grid, 0)
    ref = float(djg["dist"][24])
    result["graphs_astar_cost_dev"] = float(abs(float(ast["cost"]) - ref))
    result["graphs_astar_zero_h_dev"] = float(abs(float(ast_zero["cost"]) - ref))
    if result["graphs_astar_cost_dev"] > 1e-9:
        raise AssertionError("可采纳启发式下 A* 代价必须等于 Dijkstra 最短路")
    if result["graphs_astar_zero_h_dev"] > 1e-9:
        raise AssertionError("h=0 时 A* 应退化为 Dijkstra（代价相同）")
    if abs(ref - 8.0) > 1e-9:
        raise AssertionError("5x5 栅格 (0,0)->(4,4) 的最短路应为 8 跳")
    if int(ast["expanded"]) > 25 or int(ast_zero["expanded"]) > 25:
        raise AssertionError("A* 扩展节点数不可能超过节点总数")
    if int(ast["expanded"]) > int(ast_zero["expanded"]):
        raise AssertionError("曼哈顿启发式不应比 h=0 扩展更多节点")
    path = list(ast["path"])
    if path[0] != 0 or path[-1] != 24:
        raise AssertionError("A* 返回路径的起点/终点不正确")
    if abs(len(path) - 1 - 8.0) > 1e-9:
        raise AssertionError("单位权栅格上路径应恰好 8 条边")

    # --- VRP：直线上 depot 居中、需求各 1、容量 2 的手算算例 ---
    pos = np.array([0.0, -1.0, -2.0, 1.0, 2.0])
    Dv = np.abs(pos[:, None] - pos[None, :])
    vrp = vrp_clarke_wright(Dv, [0.0, 1.0, 1.0, 1.0, 1.0], capacity=2.0, depot=0)
    result["graphs_vrp_total_distance"] = float(vrp["total_distance"])
    result["graphs_vrp_n_routes"] = int(vrp["n_routes"])
    if abs(result["graphs_vrp_total_distance"] - 8.0) > 1e-9:
        raise AssertionError("手算最优应为 2 条路线各 1+1+2=4，合计 8")
    if result["graphs_vrp_n_routes"] != 2:
        raise AssertionError("总需求 4 / 容量 2 恰好需要 2 辆车")
    seen_customers: List[int] = []
    manual_total = 0.0
    for route in vrp["routes"]:
        if route[0] != 0 or route[-1] != 0:
            raise AssertionError("每条路线都必须从 depot 出发并回到 depot")
        load_sum = sum(float(dem_v) for dem_v in [1.0] * (len(route) - 2))
        if load_sum > 2.0 + 1e-9:
            raise AssertionError("路线需求不得超过 capacity")
        seen_customers.extend(route[1:-1])
        for k in range(len(route) - 1):
            manual_total += float(Dv[route[k], route[k + 1]])
    if sorted(seen_customers) != [1, 2, 3, 4]:
        raise AssertionError("每个客户必须恰好被服务一次")
    if abs(manual_total - result["graphs_vrp_total_distance"]) > 1e-9:
        raise AssertionError("total_distance 应等于逐路线距离求和")

    # --- 网络鲁棒性：K5 移除 1 点剩 K4；星形图移除中心点只剩孤立点 ---
    K5 = np.ones((5, 5))
    np.fill_diagonal(K5, 0.0)
    rob = network_robustness(K5, n_remove=1)
    result["graphs_robu_k5_largest"] = int(rob["largest_component"])
    star_adj = {0: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, 1: {0: 1.0}, 2: {0: 1.0}, 3: {0: 1.0}, 4: {0: 1.0}}
    rob2 = network_robustness(star_adj, n_remove=1)
    result["graphs_robu_star_largest"] = int(rob2["largest_component"])
    if result["graphs_robu_k5_largest"] != 4:
        raise AssertionError("K5 移除 1 个点后最大连通分量应为 4")
    if abs(float(rob["efficiency"]) - 1.0) > 1e-12:
        raise AssertionError("K4 的全局效率应为 1")
    if list(rob["removed"]) != [0]:
        raise AssertionError("完全图上应移除 tie-break 最靠前的节点 0")
    if result["graphs_robu_star_largest"] != 1:
        raise AssertionError("星形图移除中心点后只剩孤立点，最大分量应为 1")
    if list(rob2["removed"]) != [0]:
        raise AssertionError("星形图上度最大的点应是中心点 0")
    if abs(float(rob2["efficiency"]) - 0.0) > 1e-12:
        raise AssertionError("只剩孤立点时全局效率应为 0")

    return result
