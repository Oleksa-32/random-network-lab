from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from metrics import graph_to_d3


def save_graph_to_mongo(coll, study_id: int, G: nx.Graph) -> None:
    doc = {
        "study_id": study_id,
        "nodes": [str(n) for n in G.nodes()],
        "edges": [[str(u), str(v)] for u, v in G.edges()],
    }
    coll.update_one({"study_id": study_id}, {"$set": doc}, upsert=True)


def load_graph_from_mongo(coll, study_id: int) -> nx.Graph | None:
    doc = coll.find_one({"study_id": study_id})
    if not doc:
        return None
    G = nx.Graph()
    for n in doc.get("nodes", []):
        G.add_node(n)
    for u, v in doc.get("edges", []):
        G.add_edge(u, v)
    return G


def load_graph_d3_from_mongo(coll, study_id: int) -> Dict[str, Any] | None:
    G = load_graph_from_mongo(coll, study_id)
    if G is None:
        return None
    return graph_to_d3(G)
