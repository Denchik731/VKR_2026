# build_grap_similarity.py
# -------------------------------------------------
# Построение графа схожести пользователей по Cosine (для sparse)
# + поддержка show_progress=True
# -------------------------------------------------

from __future__ import annotations

import networkx as nx
from sklearn.neighbors import NearestNeighbors

from create_ug_matrix import UserCommunityData


def _tqdm(iterable, enabled: bool, **kwargs):
    """Безопасный tqdm: если tqdm не установлен — возвращаем iterable."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def build_similarity_graph(
    data: UserCommunityData,
    threshold: float = 0.20,
    k_neighbors: int = 40,
    show_progress: bool = True,
    **kwargs,  # совместимость на будущее
) -> nx.Graph:
    """
    Строит граф схожести пользователей на sparse-матрице user×community.

    """

    X = data.csr
    n_users = X.shape[0]
    if n_users < 2:
        raise ValueError("Нужно минимум 2 пользователя для построения графа.")

    n_neighbors = min(k_neighbors + 1, n_users)

    # Cosine работает на sparse
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )

    print(" Считаю ближайших соседей (kNN, метрика Cosine)...")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    print("kNN готово. Строю рёбра графа...")

    G = nx.Graph()

    # Добавляем узлы заранее
    for uid in data.user_ids:
        G.add_node(uid, type="user")

    iterator = _tqdm(range(n_users), enabled=show_progress, desc="Построение рёбер", unit="user")

    for i in iterator:
        u = data.user_ids[i]

        # 0-й сосед — сам пользователь (distance=0), пропускаем
        neigh_idx = indices[i][1:]
        neigh_dist = distances[i][1:]

        for j, dist in zip(neigh_idx, neigh_dist):
            sim = 1.0 - float(dist)
            if sim >= threshold:
                v = data.user_ids[int(j)]
                if u == v:
                    continue

                # Если ребро уже есть — оставим максимальный вес
                if G.has_edge(u, v):
                    if sim > G[u][v].get("weight", 0):
                        G[u][v]["weight"] = sim
                else:
                    G.add_edge(u, v, weight=sim)

    return G
