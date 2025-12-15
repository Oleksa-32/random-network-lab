from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx
import numpy as np


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    return G.subgraph(components[0]).copy()


def summarize_vector(values: Dict[Any, float]) -> Dict[str, float]:
    arr = list(values.values())
    if len(arr) == 0:
        return {"mean": None, "max": None}
    return {"mean": float(np.mean(arr)), "max": float(np.max(arr))}


def pearson_corr(xs: List[float | None], ys: List[float | None]) -> float | None:
    x_f: List[float] = []
    y_f: List[float] = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        xv = float(x)
        yv = float(y)
        if np.isnan(xv) or np.isnan(yv):
            continue
        x_f.append(xv)
        y_f.append(yv)

    if len(x_f) < 2:
        return None

    x_arr = np.asarray(x_f, dtype=float)
    y_arr = np.asarray(y_f, dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None

    try:
        return float(np.corrcoef(x_arr, y_arr)[0, 1])
    except Exception:
        return None


def long_range_correlations(
    G: nx.Graph | nx.DiGraph,
    max_nodes: int = 600,
    max_pairs: int = 250_000,
    seed: int = 42,
    directed_mode: str = "out_in",
) -> Dict[str, Any]:
    nodes = list(G.nodes())
    n_total = len(nodes)
    if n_total == 0:
        return {"used_nodes": 0, "total_nodes": 0, "pairs": 0, "by_distance": []}

    if n_total > max_nodes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_nodes, replace=False)
        nodes = [nodes[i] for i in idx]

    node_to_i = {node: i for i, node in enumerate(nodes)}
    used_set = set(nodes)

    if isinstance(G, nx.DiGraph):
        if directed_mode == "out_in":
            deg_u = {u: float(G.out_degree(u)) for u in used_set}
            deg_v = {v: float(G.in_degree(v)) for v in used_set}
        elif directed_mode == "out_out":
            deg_u = {u: float(G.out_degree(u)) for u in used_set}
            deg_v = {v: float(G.out_degree(v)) for v in used_set}
        elif directed_mode == "in_in":
            deg_u = {u: float(G.in_degree(u)) for u in used_set}
            deg_v = {v: float(G.in_degree(v)) for v in used_set}
        else:
            deg_u = {u: float(G.degree(u)) for u in used_set}
            deg_v = {v: float(G.degree(v)) for v in used_set}
    else:
        deg_u = {u: float(G.degree(u)) for u in used_set}
        deg_v = deg_u

    buckets_u: Dict[int, List[float]] = defaultdict(list)
    buckets_v: Dict[int, List[float]] = defaultdict(list)

    pair_count = 0
    undirected = not isinstance(G, nx.DiGraph)

    for u in nodes:
        dist = nx.single_source_shortest_path_length(G, u)
        iu = node_to_i[u]
        for v, d in dist.items():
            if v == u or v not in used_set:
                continue
            if undirected:
                iv = node_to_i[v]
                if iv <= iu:
                    continue

            buckets_u[int(d)].append(deg_u[u])
            buckets_v[int(d)].append(deg_v[v])

            pair_count += 1
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break

    by_distance: List[Dict[str, Any]] = []
    for d in sorted(buckets_u.keys()):
        xs = buckets_u[d]
        ys = buckets_v[d]
        corr = pearson_corr(xs, ys)
        by_distance.append(
            {
                "distance": int(d),
                "samples": int(len(xs)),
                "degree_corr": corr,
                "u_degree_mean": float(np.mean(xs)) if xs else None,
                "v_degree_mean": float(np.mean(ys)) if ys else None,
            }
        )

    return {
        "total_nodes": int(n_total),
        "used_nodes": int(len(nodes)),
        "pairs": int(pair_count),
        "directed_mode": directed_mode if isinstance(G, nx.DiGraph) else "undirected",
        "by_distance": by_distance,
    }


def compute_metrics_single(G: nx.Graph, metrics: List[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}

    H = largest_connected_component(G)
    n = G.number_of_nodes()

    SMALL = 1500
    MEDIUM = 5000
    LARGE = 10000

    for m in metrics:
        if m == "avg_clustering":
            res[m] = float(nx.average_clustering(G))

        elif m == "transitivity":
            res[m] = float(nx.transitivity(G))

        elif m == "avg_shortest_path_length":
            val = None
            if H.number_of_nodes() > 1 and n <= MEDIUM:
                try:
                    val = float(nx.average_shortest_path_length(H))
                except Exception:
                    val = None
            res[m] = val

        elif m == "diameter":
            val = None
            if H.number_of_nodes() > 1 and n <= SMALL:
                try:
                    val = float(nx.diameter(H))
                except Exception:
                    val = None
            res[m] = val

        elif m == "degree_centrality":
            res[m] = summarize_vector(nx.degree_centrality(G))

        elif m == "betweenness_centrality":
            if n <= SMALL:
                try:
                    bet = nx.betweenness_centrality(G, normalized=True)
                    res[m] = summarize_vector(bet)
                except Exception:
                    res[m] = {"mean": None, "max": None}
            elif n <= MEDIUM:
                try:
                    k = min(200, n)
                    bet = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
                    res[m] = summarize_vector(bet)
                except Exception:
                    res[m] = {"mean": None, "max": None}
            else:
                res[m] = {"mean": None, "max": None, "note": "skipped (graph too large)"}

        elif m == "closeness_centrality":
            if n <= LARGE:
                try:
                    clo = nx.closeness_centrality(G)
                    res[m] = summarize_vector(clo)
                except Exception:
                    res[m] = {"mean": None, "max": None}
            else:
                res[m] = {"mean": None, "max": None, "note": "skipped (graph too large)"}

        elif m == "eigenvector_centrality":
            if n <= MEDIUM:
                try:
                    eig = nx.eigenvector_centrality(G, max_iter=1000)
                    res[m] = summarize_vector(eig)
                except Exception:
                    res[m] = {"mean": None, "max": None}
            else:
                res[m] = {"mean": None, "max": None, "note": "skipped (graph too large)"}

        elif m == "assortativity_degree":
            try:
                res[m] = float(nx.degree_assortativity_coefficient(G))
            except Exception:
                res[m] = None

        elif m == "long_range_correlations":
            if n <= LARGE:
                res[m] = long_range_correlations(G)
            else:
                res[m] = {"note": "skipped (graph too large)"}

        else:
            res[m] = None

    res["_meta"] = {
        "n": n,
        "m": G.number_of_edges(),
        "density": float(nx.density(G)),
        "is_connected": bool(nx.is_connected(G)) if n > 0 else False,
        "largest_cc_size": H.number_of_nodes(),
    }
    return res


def flatten_metric_for_group(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("mean", None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def graph_to_d3(G: nx.Graph) -> Dict[str, Any]:
    return {
        "nodes": [{"id": str(n), "degree": int(G.degree(n))} for n in G.nodes()],
        "links": [{"source": str(u), "target": str(v)} for u, v in G.edges()],
    }


def centrality_vectors(G: nx.Graph) -> Dict[str, List[float] | None]:
    nodes = list(G.nodes())
    out: Dict[str, List[float] | None] = {}
    n = len(nodes)

    SMALL = 1500
    MEDIUM = 5000
    LARGE = 10000

    deg = nx.degree_centrality(G)
    out["degree_centrality"] = [float(deg[n_]) for n_ in nodes] if nodes else []

    if n <= SMALL:
        try:
            btw = nx.betweenness_centrality(G, normalized=True)
            out["betweenness_centrality"] = [float(btw[n_]) for n_ in nodes]
        except Exception:
            out["betweenness_centrality"] = None
    elif n <= MEDIUM:
        try:
            k = min(200, n)
            btw = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
            out["betweenness_centrality"] = [float(btw[n_]) for n_ in nodes]
        except Exception:
            out["betweenness_centrality"] = None
    else:
        out["betweenness_centrality"] = None

    if n <= LARGE:
        try:
            clo = nx.closeness_centrality(G)
            out["closeness_centrality"] = [float(clo[n_]) for n_ in nodes]
        except Exception:
            out["closeness_centrality"] = None
    else:
        out["closeness_centrality"] = None

    if n <= MEDIUM:
        try:
            eig = nx.eigenvector_centrality(G, max_iter=1000)
            out["eigenvector_centrality"] = [float(eig[n_]) for n_ in nodes]
        except Exception:
            out["eigenvector_centrality"] = None
    else:
        out["eigenvector_centrality"] = None

    out["_order"] = nodes
    return out


def corr_matrix_nodewise(cv: Dict[str, List[float] | None]) -> Dict[str, Dict[str, float | None]]:
    keys = [
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
    ]
    available = [k for k in keys if isinstance(cv.get(k), list)]
    mat: Dict[str, Dict[str, float | None]] = {}

    for i, a in enumerate(available):
        mat[a] = {}
        for j, b in enumerate(available):
            if j < i:
                mat[a][b] = mat[b][a]
            else:
                if a == b:
                    mat[a][b] = 1.0
                else:
                    xa = cv[a]
                    xb = cv[b]
                    mat[a][b] = pearson_corr(xa, xb) if (xa is not None and xb is not None) else None
    return mat


def corr_matrix_groupwise(agg_buckets: Dict[str, List[float | None]]) -> Dict[str, Dict[str, float | None]]:
    metrics = list(agg_buckets.keys())
    mat: Dict[str, Dict[str, float | None]] = {}

    for i, a in enumerate(metrics):
        mat[a] = {}
        for j, b in enumerate(metrics):
            if j < i:
                mat[a][b] = mat[b][a]
            else:
                if a == b:
                    mat[a][b] = 1.0
                else:
                    xs = agg_buckets[a]
                    ys = agg_buckets[b]
                    mat[a][b] = pearson_corr(xs, ys)
    return mat
