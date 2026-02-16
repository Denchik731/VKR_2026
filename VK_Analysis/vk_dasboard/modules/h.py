

import networkx as nx
def generate_report(df, graph, top_communities, suspicious_communities):
    """Генерация полного отчета"""

    print("=" * 60)
    print("АНАЛИТИЧЕСКИЙ ОТЧЕТ ПО СООБЩЕСТВАМ ВК")
    print("=" * 60)

    print(f"\nОбщая статистика:")
    print(f"   Всего пользователей: {len(df)}")
    print \
        (f"   Всего уникальных сообществ: {len(set([item for sublist in df['communities_list'] for item in sublist]))}")
    print(f"   Связей в графе: {graph.number_of_edges()}")
    print(f"   Плотность графа: {nx.density(graph):.4f}")

    print(f"\n Топ-5 самых популярных сообществ:")
    for i, (comm, count) in enumerate(top_communities[:5], 1):
        print(f"   {i}. {comm}: {count} подписчиков")

    if suspicious_communities:
        print(f"\n Обнаружено подозрительных сообществ: {len(suspicious_communities)}")
        for comm, count in suspicious_communities.most_common(5):
            print(f" {comm}: {count} подписчиков")

    # Анализ компонент связности
    components = list(nx.connected_components(graph))
    print(f"\n Компоненты связности: {len(components)}")
    print(f"   Размер самой большой компоненты: {len(max(components, key=len))}")

