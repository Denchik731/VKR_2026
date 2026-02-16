

from collections import Counter
# Анализ наиболее влиятельных сообществ

def find_most_common_communities(df, top_n=10):
    # top_n - сколько топовых сообществ выводим
    #extend() - добавляет все элементы списка в общий список (в отличие от append(), который добавил бы весь список как один элемент)


    # Собираем все сообщества
    all_communities = []
    for communities in df['communities_list']:
        all_communities.extend(communities)

    # Считаем частоты
    community_counts = Counter(all_communities)

    # Топ-N самых популярных
    top_communities = community_counts.most_common(top_n)

    # most_common(top_n) - метод Counter, который возвращает: список кортежей (сообщество, количество) отсортированный по убыванию частоты

    return top_communities



