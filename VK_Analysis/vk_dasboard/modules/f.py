

# ищим подозрительные сообщества

from collections import Counter


def detect_suspicious_patterns(df, graph):
    """Анализ на предмет деструктивных/радикальных сообществ"""

    suspicious_keywords = [
        'радикал', 'экстремизм', 'ненависть', 'противостояние',
        'революция', 'сопротивление', 'подполье', 'анархия',
        'независимость', 'протест', 'свержение'
    ]

    suspicious_communities = []

    for communities in df['communities_list']:
        for community in communities:
            # Преобразуем в строку, если это число
            if isinstance(community, (int, float)):
                community_str = str(community)
            else:
                community_str = str(community)

            community_lower = community_str.lower()

            if any(keyword in community_lower for keyword in suspicious_keywords):
                suspicious_communities.append(community_str)

    suspicious_counts = Counter(suspicious_communities)

    return suspicious_counts
