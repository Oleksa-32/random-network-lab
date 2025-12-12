from __future__ import annotations

from typing import Dict, Any, List

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
    keys = ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"]
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
