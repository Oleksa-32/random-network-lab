from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np


def parse_txt_network(text: str) -> nx.Graph:
    raw_lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith(("#", "%", "//"))
    ]
    if not raw_lines:
        raise ValueError("No data lines found in txt file")

    sample = raw_lines[:10]
    tokenized = [ln.split() for ln in sample]

    def mostly_k(k: int) -> bool:
        non_empty = [t for t in tokenized if t]
        if not non_empty:
            return False
        count_k = sum(1 for t in non_empty if len(t) == k)
        return count_k / len(non_empty) >= 0.8

    first_len = len(tokenized[0])

    if all(len(t) == first_len for t in tokenized) and len(raw_lines) == first_len:
        data = []
        for ln in raw_lines:
            row = [float(x) for x in ln.split()]
            data.append(row)
        mat = np.array(data)
        return nx.from_numpy_array(mat)

    if mostly_k(2) or mostly_k(3):
        G = nx.Graph()

        def is_int_pair(tokens: List[str]) -> bool:
            if len(tokens) != 2:
                return False
            try:
                int(tokens[0])
                int(tokens[1])
                return True
            except ValueError:
                return False

        skip_first = False
        if len(tokenized) >= 2 and is_int_pair(tokenized[0]) and is_int_pair(tokenized[1]):
            n1, m1 = map(int, tokenized[0])
            try:
                u, v = map(int, tokenized[1])
                if max(u, v) < n1 and m1 >= len(raw_lines) - 1:
                    skip_first = True
            except ValueError:
                pass

        for idx, ln in enumerate(raw_lines):
            if skip_first and idx == 0:
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            G.add_edge(u, v)
        return G

    raise ValueError("Could not detect txt network format (not adjacency matrix, not edge list)")
