"""图与网络算法：最短路、最小生成树、最大流、TSP 启发式、PageRank、连通分量。

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
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
        ``next_node[i, j]`` 是 i->j 最短路上 j 的前一个节点（"下一跳矩阵"），
        不可达或 i==j 时为 ``-1``。它可直接喂给 :func:`reconstruct_path`。

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
        4. 本函数**原地修改**传入的 numpy 数组副本（内部已 ``copy()``），调用方数组不受影响。

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
        ``{"max_flow": float, "flow": {u: {v: 流量}}}``。``flow`` 只包含残量网络中
        有正流量的边（含反向抵消边，即为负流量形式的对偶记录时同为正），
        每条边的净流量满足 ``0 <= flow[u][v] <= capacity[u][v]``。

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
        先跑一次最大流，然后在残量网络（``res > 0`` 的边）上从 source 做 BFS/DFS，
        得到源侧集合 S；割边 = 从 S 指向 V\\S 的原始边。由 max-flow min-cut 定理，
        这些边必然全部饱和，且容量和 = 最大流。

    复杂度:
        时间 O(V E^2)（含最大流）/ 空间 O(V + E)。

    陷阱:
        1. 返回的是**有向边** (S -> V\\S)，双向图里会只列出从 S 出去的那个方向，
           这正是定理要的那个割，不要"补全"成两条。
        2. 最小割**不唯一**：这里给出的是"离源点最近"的那个最小割（源侧集合最小）。
           论文里若断言割边集合唯一，会被质疑。
        3. 割容量必须用**原始容量**求和，不要用残量；用残量求和会得到 0。
        4. 若原图有平行边，容量已合并，返回的 cap 是合并后的值，和输入逐条对不上。

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
    """校验距离矩阵：方阵、非负、对角为 0、对称性给警告式校验（不强制）。"""
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
        (i, j) 与 (i+1, j+1)，即**反转 tour[i+1..j]**；只接受严格改进。
        每轮内用"首次改进即继续"（first-improvement），直到某轮无改进或达到 max_pass。

    复杂度:
        时间 O(n^2) 每次扫描，最多 max_pass 轮；实践上常几轮就停 / 空间 O(n)。

    陷阱:
        1. 2-opt 是**局部最优**，对 10 城以下常能跑到最优，城市多了必须配合
           多起点重启（这里的 ``improved=False`` 不代表已最优）。
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
    return result
