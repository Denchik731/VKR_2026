import random
import pandas as pd


# ============================
# 1) Тематики "как в ВК" (примерно)
# ============================
VK_TOPICS = [
    "Политика", "Новости", "Юмор", "Музыка", "Кино", "Игры", "Спорт", "ЗОЖ",
    "Путешествия", "Наука", "Образование", "IT", "Финансы", "Бизнес",
    "Авто", "Мода", "Кулинария", "Психология", "Искусство", "Книги"
]

# ============================
# 2) Сегменты пользователей (для выраженных скрытых сообществ)
# ============================
SEGMENTS = {
    "A_надежные": {
        "topics": {"Образование": 5, "Наука": 4, "IT": 4, "Книги": 3, "ЗОЖ": 3, "Финансы": 2},
        "noise": 0.15
    },
    "B_карьера": {
        "topics": {"Бизнес": 5, "Финансы": 4, "IT": 3, "Новости": 2, "Образование": 2},
        "noise": 0.20
    },
    "C_развлечения": {
        "topics": {"Юмор": 5, "Музыка": 4, "Кино": 4, "Игры": 3, "Новости": 1},
        "noise": 0.25
    },
    "D_спорт": {
        "topics": {"Спорт": 5, "ЗОЖ": 4, "Путешествия": 2, "Юмор": 1},
        "noise": 0.20
    },
    "E_политика": {
        "topics": {"Политика": 5, "Новости": 4, "Психология": 1},
        "noise": 0.25
    },
    "F_лайфстайл": {
        "topics": {"Мода": 4, "Кулинария": 4, "Путешествия": 3, "Психология": 2, "Искусство": 2},
        "noise": 0.25
    },
}

SEGMENT_WEIGHTS = [
    ("A_надежные", 0.18),
    ("B_карьера", 0.17),
    ("C_развлечения", 0.20),
    ("D_спорт", 0.15),
    ("E_политика", 0.15),
    ("F_лайфстайл", 0.15),
]


# ============================
# 3) Генерация каталога сообществ: community_id -> topic -> name
# ============================
def generate_communities_catalog(total_communities: int = 1200, seed: int = 42):
    """
    total_communities:
      - для дипломной демонстрации хорошо 800..1500
      - меньше 400 — часто портит качество кластеров
    """
    random.seed(seed)

    base = 10_000_000
    community_ids = [-(base + i) for i in range(total_communities)]
    topics = [random.choice(VK_TOPICS) for _ in range(total_communities)]

    df_topics = pd.DataFrame({
        "community_id": [str(x) for x in community_ids],
        "topic": topics,
        "name": [f"{t} • паблик #{i+1}" for i, t in enumerate(topics)]
    })

    topic_to_ids = {
        t: df_topics.loc[df_topics["topic"] == t, "community_id"].tolist()
        for t in VK_TOPICS
    }
    return df_topics, topic_to_ids


def choose_segment():
    r = random.random()
    cum = 0.0
    for seg, w in SEGMENT_WEIGHTS:
        cum += w
        if r <= cum:
            return seg
    return SEGMENT_WEIGHTS[-1][0]


# ============================
# 4) Генерация "рёбер" user_id -> community_id (25..60 на пользователя)
# ============================
def generate_edges(n_users: int, topic_to_ids: dict, seed: int = 123):
    random.seed(seed)

    edges = []
    used_ids = set()

    # ---- Ядро для сегментов: чтобы у людей сегмента были устойчивые пересечения ----
    segment_core = {}
    for seg_name, cfg in SEGMENTS.items():
        core_pool = []
        for t in cfg["topics"].keys():
            # берем небольшой стабильный набор групп по каждой теме
            pool = topic_to_ids[t]
            take = min(25, len(pool))
            core_pool += random.sample(pool, k=take)

        # итоговое ядро сегмента (чтобы пересечения были сильные)
        segment_core[seg_name] = random.sample(core_pool, k=min(50, len(core_pool)))

    start_id = 100000000
    for i in range(n_users):
        user_id = str(start_id + i)
        seg = choose_segment()
        cfg = SEGMENTS[seg]

        # сколько сообществ у пользователя
        k = random.randint(25, 60)

        communities = set()

        # 1) ядро сегмента (12..25)
        core_take = random.randint(12, 25)
        core_list = segment_core[seg]
        communities.update(random.sample(core_list, k=min(core_take, len(core_list))))

        # 2) добор по тематике + шум
        topics = list(cfg["topics"].keys())
        weights = list(cfg["topics"].values())

        while len(communities) < k:
            if random.random() < cfg["noise"]:
                t = random.choice(VK_TOPICS)
            else:
                t = random.choices(topics, weights=weights, k=1)[0]

            communities.add(random.choice(topic_to_ids[t]))

        # записываем ребра
        for cid in communities:
            cid_str = str(cid)
            edges.append({"user_id": user_id, "community_id": cid_str})
            used_ids.add(cid_str)

    return pd.DataFrame(edges), used_ids


# ============================
# 5) MAIN
# ============================
if __name__ == "__main__":
    # -------- Настройки для диплома --------
    N_USERS = 3000                 # сколько пользователей
    TOTAL_COMMUNITIES = 1200       # сколько уникальных сообществ всего (лучше 800..1500)

    # 1) Справочник сообществ
    topics_df, topic_to_ids = generate_communities_catalog(
        total_communities=TOTAL_COMMUNITIES,
        seed=42
    )

    # 2) Рёбра user->community
    edges_df, used_ids = generate_edges(
        n_users=N_USERS,
        topic_to_ids=topic_to_ids,
        seed=123
    )

    # 3) Оставляем только реально используемые community_id
    topics_filtered = topics_df[topics_df["community_id"].isin(used_ids)].copy()

    # Проверка целостности
    missing = set(edges_df["community_id"]) - set(topics_filtered["community_id"])
    if missing:
        raise ValueError(f"В edges есть community_id, которых нет в topics: {list(missing)[:10]}")

    # 4) Сохраняем
    edges_df.to_csv("users_communities_edges.csv", index=False, encoding="utf-8-sig", sep=";")
    topics_filtered.to_csv("community_topics.csv", index=False, encoding="utf-8-sig", sep=";")

    # -------- Мини-отчёт качества  --------
    print("Готово:")
    print("- users_communities_edges.csv (user_id;community_id)")
    print("- community_topics.csv (community_id;topic;name)")
    print(f"Пользователей: {N_USERS}")
    print(f"Уникальных сообществ у пользователей: {len(used_ids)}")
    print(f"Всего ребер user->community: {len(edges_df)}")

    # Распределение тематик
    topic_counts = topics_filtered["topic"].value_counts().head(10)
    print("\nТОП-10 тематик по количеству сообществ:")
    print(topic_counts.to_string())
