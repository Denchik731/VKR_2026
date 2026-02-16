import streamlit as st
import pandas as pd
import networkx as nx
from e import visualize_network_advanced
from create_ug_matrix import UserCommunityData
from build_grap_similarity import build_similarity_graph
from collections import Counter
from pathlib import Path
import tempfile
import os
# TODO:  теперь загрузка осуществляется через элемент управления Streamlit — компонент file_uploader, позволяя пользователям выбирать файлы вручную.



# Функция загрузки данных
def load_data():
    # Пользователи загружают CSV с ребрами и тематическими метками сообществ
    edges_csv = st.file_uploader("Выберите файл с ребрами (User-Community)", type=["csv"])
    topics_csv = st.file_uploader("Выберите файл с темой сообществ", type=["csv"])

    if edges_csv is not None and topics_csv is not None:
        edges_df = pd.read_csv(edges_csv, sep=";", encoding="utf-8-sig", dtype=str)
        topics_df = pd.read_csv(topics_csv, sep=";", encoding="utf-8-sig", dtype=str)

        # Формируем структуру данных
        user_community_data = UserCommunityData.from_edges_df(edges_df)
        return edges_df, topics_df, user_community_data
    else:
        return None, None, None

# Анализ и визуализация данных
def analyze_and_visualize():
    edges_df, topics_df, user_community_data = load_data()

    if edges_df is not None and topics_df is not None:
        # Создаем временный CSV файл из DataFrame
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig') as tmp:
            topics_df.to_csv(tmp.name, sep=';', index=False)
            topics_csv_path = tmp.name

        G = build_similarity_graph(user_community_data, threshold=0.15, k_neighbors=50)

        partition, summary_rows, cluster_info, fig = visualize_network_advanced(
            G=G, edges_df=edges_df, topics_csv_path=topics_csv_path,
            title="Анализ скрытых сообществ ВКонтакте", show=True, max_nodes_plot=2000
        )

        os.unlink(topics_csv_path)  # Cleanup
# Структура страницы
def page(card):
    st.markdown("## 🕵️ Латентные интересы и группы")
    st.write("Этот инструмент помогает выявить скрытые группы и сообщества, основываясь на взаимодействии пользователей.")

    analyze_and_visualize()