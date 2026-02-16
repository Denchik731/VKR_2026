
# анализ вовлеченности
import networkx as nx

def analyze_engagement(df, graph):
    """Анализ вовлеченности пользователей"""

    # Степень центральности
    degree_centrality = nx.degree_centrality(graph) # подается граф и расчитвыется степь его ценральности

    # Междуness centrality (важность узла как моста)
    betweenness = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes())))

    # Наиболее связанные пользователи
    top_connected = sorted(degree_centrality.items(),
                           key=lambda x: x[1], reverse=True)[:10]

    print("\nСамые связанные пользователи:")
    for user, centrality in top_connected:
        user_communities = df[df['user_id'] == user]['communities_list'].iloc[0]
        print(f" {user}: центральность {centrality:.4f}")
        print(f"   Сообщества: {user_communities[:5]}...")

    return degree_centrality, betweenness


