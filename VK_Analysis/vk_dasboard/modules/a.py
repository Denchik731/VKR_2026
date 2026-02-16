

"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from create_ug_matrix import create_user_community_matrix_from_edges  # <-- создаём data корректно из edges
from build_grap_similarity import build_similarity_graph              # <-- ВАЖНО: импорт отсюда!
from e import visualize_network_advanced
from create_ug_matrix import create_user_community_matrix_from_edges




def load_edges_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    required = {"user_id", "community_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"Ожидались колонки {required}, но есть: {list(df.columns)}")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["community_id"] = df["community_id"].astype(str).str.strip()
    return df


if __name__ == "__main__":
    # ---------------------------
    # Пути к файлам (лежать рядом с a.py)
    # ---------------------------
    edges_csv = Path("users_communities_edges.csv")
    topics_csv = Path("community_topics.csv")

    if not edges_csv.exists():
        raise FileNotFoundError(f"Не найден файл: {edges_csv.resolve()}")
    if not topics_csv.exists():
        raise FileNotFoundError(f"Не найден файл: {topics_csv.resolve()}")

    # ---------------------------
    # 1) Загружаем edges
    # ---------------------------
    df_edges = load_edges_csv(str(edges_csv))
    print(f"Пользователей (уникальных): {df_edges['user_id'].nunique()} | Рёбер: {df_edges.shape[0]}")

    # ---------------------------
    # 2) Создаём структуру данных (sparse user×community)
    # ---------------------------
    #data = create_user_community_matrix_from_edges(df_edges)
    data = create_user_community_matrix_from_edges(df_edges)

    # ---------------------------
    # 3) Строим граф схожести
    # ---------------------------
    G = build_similarity_graph(   # это параметры плотности графа
        data=data,
        threshold=0.15,
        k_neighbors=50,
        show_progress=True
    )

    print(f"Граф: узлов={G.number_of_nodes()}, рёбер={G.number_of_edges()}")

    # ---------------------------
    # 4) Визуализация + анализ скрытых сообществ
    # ---------------------------
    partition, summary_rows, cluster_info, fig = visualize_network_advanced(
        G=G,
        edges_df=df_edges,                    # <-- исправлено: было edges_df (несуществовало)
        topics_csv_path=str(topics_csv),
        title="Анализ скрытых сообществ ВКонтакте",
        show=True,
        max_nodes_plot=2000
    )

    print("\nТОП-5 скрытых сообществ (по значимости):")
    for r in summary_rows[:5]:
        print(
            f"ID {r['hidden_comm_id']}: score={r['score']} | size={r['size_users']} | "
            f"тематики: {r['top_topics']} | признак: {r['обобщающий_признак']}"
        )
"""

