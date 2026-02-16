
import numpy as np # Математика, массивы
import pandas as pd  # таблицы красивые
import streamlit as st # библа для веб-интерфеса
import plotly.express as px # Интерактивные графики

from pathlib import Path # пути к файлам
from io import BytesIO# Чтение CSV из байтов

# Sklearn: предобработка + кластеризация
from sklearn.compose import ColumnTransformer #
from sklearn.preprocessing import OneHotEncoder, StandardScaler #
from sklearn.cluster import MiniBatchKMeans, DBSCAN #
from sklearn.metrics import silhouette_score # силует для поиска кол-ва кластеров к-means

import umap


# ============================================================
# CONFIG
# ============================================================

# константы

DEFAULT_DATA_PATHS = [              # ищу откуда подтянуть csv с данными
    Path("vk_users_10000.csv"),
    Path("data/vk_users_10000.csv"),
    Path("datasets/vk_users_10000.csv"),
]

DROP_COLS = {"id", "synthetic_cluster", "cluster_kmeans", "cluster_dbscan"}
NUM_COLS_CANDIDATES = ["age"]


# ============================================================
# загрузка и предообработка
# ============================================================

# функция поиска пути к файлу дадасету
def find_default_csv() -> Path | None:
    for p in DEFAULT_DATA_PATHS:
        if p.exists():
            return p
    return None


# совместимость с разными версиями библиотеки sklearn --> экземпляр OneHot создастся в любом случае
def safe_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# декоратор нужен для быстрой загруки в Streamlit
@st.cache_data(show_spinner=False) # ← Декоратор кэширования в Streamlit
def read_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") # возвращает DataFrame


@st.cache_data(show_spinner=False)
def read_csv_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(b), encoding="utf-8-sig")


# разделяет датасет на числовые и категориальные колонки
def detect_columns(df: pd.DataFrame):
    num_cols = [c for c in NUM_COLS_CANDIDATES if c in df.columns] # тут числовые
    cat_cols = [c for c in df.columns if c not in DROP_COLS and c not in num_cols]
    cat_cols = [c for c in cat_cols if df[c].dtype == "object"] # тут лежат категориальные
    return num_cols, cat_cols

# кешируем объект - результат функции
#@st.cache_resource(show_spinner=False)
def fit_preprocessor(df: pd.DataFrame, num_cols, cat_cols) -> ColumnTransformer:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols), # страндартизиреум признаки
            ("cat", safe_onehot(), cat_cols), # ван хот для категориальных
        ],
        remainder="drop",
    )
    pre.fit(df) # запуск методов
    return pre


# предобработка данных (ванхот, скалер и тд)
#@st.cache_resource(show_spinner=False)
def transform_features(_pre: ColumnTransformer, df: pd.DataFrame):
    return _pre.transform(df)


# ============================================================
# Risk / Explanation helpers
# ============================================================
def share_positive(series: pd.Series) -> float:
    """Доля 'положительного' отношения."""
    if series is None or series.empty:
        return 0.0
    s = series.fillna("").astype(str).str.lower()
    return float(s.str.contains("полож").mean())


def share_is(series: pd.Series, keywords: list[str]) -> float:
    """Доля строк, где значение содержит хотя бы одно ключевое слово."""
    if series is None or series.empty:
        return 0.0
    s = series.fillna("").astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | s.str.contains(kw)
    return float(mask.mean())


def ideological_risk_share(series: pd.Series) -> float:
    """
    Идеологический риск (для режимного предприятия):
    либеральные / либертарианские / индифферентные.
    """
    if series is None or series.empty:
        return 0.0
    s = series.fillna("").astype(str).str.lower()
    return float(
        (s.str.contains("либерал") | s.str.contains("либертариан") | s.str.contains("индиффер")).mean()
    )


def top_value(series: pd.Series) -> str:
    if series is None or series.empty:
        return "—"
    vc = series.fillna("—").astype(str).value_counts()
    return str(vc.index[0])


def top_n(series: pd.Series, n=3) -> str:
    if series is None or series.empty:
        return "—"
    vc = series.fillna("").astype(str).value_counts(normalize=True).head(n)
    items = []
    for name, share in vc.items():
        if name == "":
            continue
        items.append(f"{name} ({share*100:.0f}%)")
    return ", ".join(items) if items else "—"


def risk_drivers(part: pd.DataFrame) -> dict:
    """Считаем доли факторов риска в кластере."""
    alc_pos = share_positive(part["alcohol"]) if "alcohol" in part.columns else 0.0
    smk_pos = share_positive(part["smoking"]) if "smoking" in part.columns else 0.0

    edu_low = share_is(part["education_level"], ["нет", "среднее"]) if "education_level" in part.columns else 0.0
    life_hed = share_is(part["main_in_life"], ["развлеч", "слава", "влияние"]) if "main_in_life" in part.columns else 0.0
    ppl_money = share_is(part["main_in_people"], ["власть", "богат"]) if "main_in_people" in part.columns else 0.0

    pol_liberal = ideological_risk_share(part["political"]) if "political" in part.columns else 0.0

    return {
        "alc_pos": alc_pos,
        "smk_pos": smk_pos,
        "edu_low": edu_low,
        "life_hed": life_hed,
        "ppl_money": ppl_money,
        "pol_liberal": pol_liberal,
    }

#   тут можно настривать чувсвительность системы 
def risk_score_0_100(alc, smk, pol, edu):
    """
    Итоговый риск кластера (0–100).
    Приоритет:
    - алкоголь (45%)
    - либеральные/либертарианские/индифферентные (25%)
    - курение (20%)
    - низкое образование (10%)
    """
    return float(
        round(
            100 * (
                0.45 * alc +
                0.20 * smk +
                0.25 * pol +
                0.10 * edu
            ),
            1
        )
    )


def risk_level_ru(score: float) -> str:
    if score >= 60:
        return "ВЫСОКИЙ"
    if score >= 30:
        return "СРЕДНИЙ"
    return "НИЗКИЙ"


def main_risk_factor(dr: dict) -> str:
    """
    Главный фактор риска — выбираем фактор с максимальным вкладом в итоговый score.
    """
    factors = {
        "Положительное отношение к алкоголю": dr["alc_pos"] * 0.45,
        "Либеральные политические взгляды": dr["pol_liberal"] * 0.25,
        "Положительное отношение к курению": dr["smk_pos"] * 0.20,
        "Низкий уровень образования": dr["edu_low"] * 0.10,
    }
    return max(factors, key=factors.get)


def why_danger_ru(alcohol_pos: float, smoking_pos: float, pol_liberal: float, edu_low: float, top_life: str) -> str:
    reasons = []
    if alcohol_pos >= 0.45:
        reasons.append("высокая доля положительного отношения к алкоголю")
    if smoking_pos >= 0.45:
        reasons.append("высокая доля положительного отношения к курению")
    if pol_liberal >= 0.45:
        reasons.append("преобладание либеральных/либертарианских/индифферентных взглядов")
    if edu_low >= 0.45:
        reasons.append("высокая доля низкого уровня образования")
    if isinstance(top_life, str) and (("развлеч" in top_life.lower()) or ("слава" in top_life.lower())):
        reasons.append("ценности смещены в сторону развлечений/влияния")
    return "; ".join(reasons) if reasons else "выраженных риск-факторов не обнаружено"


def cluster_type_ru(dr: dict, top_edu: str, top_life: str) -> str:
    if dr["alc_pos"] > 0.50 and dr["pol_liberal"] > 0.40:
        return "Рисковый: алкоголь + идеология"
    if dr["alc_pos"] > 0.50:
        return "Рисковый: вредные привычки"
    if dr["pol_liberal"] > 0.50:
        return "Рисковый: идеологический профиль"
    if dr["edu_low"] > 0.45:
        return "Рисковый: низкое образование"
    if "высш" in str(top_edu).lower() and ("семья" in str(top_life).lower() or "саморазвит" in str(top_life).lower()):
        return "Надёжный: социально устойчивый"
    return "Смешанный: требует внимания"


def recommendation_ru(level: str) -> str:
    if level == "ВЫСОКИЙ":
        return "Рекомендуется углублённая проверка (сообщества/контент/окружение)."
    if level == "СРЕДНИЙ":
        return "Рекомендуется точечная проверка (аномалии, окружение 1–2 уровня)."
    return "Фоновый контроль (без приоритета)."


def build_text_report(summary_df: pd.DataFrame, total_n: int) -> str:
    lines = []
    lines.append("ОТЧЁТ ПО КЛАСТЕРИЗАЦИИ ОКРУЖЕНИЯ ВК")
    lines.append("=" * 70)
    lines.append(f"Всего профилей: {total_n}")
    lines.append("")

    for _, r in summary_df.iterrows():
        lines.append(f"Кластер {r['Кластер']} — {r['Тип кластера']}")
        lines.append(f"  Уровень риска: {r['Уровень риска']} | Риск: {r['Риск, % (0-100)']} / 100")
        lines.append(f"  Главный фактор риска: {r['Главный фактор риска']}")
        lines.append(f"  Размер: {r['Количество']} ({r['Доля, %']}%)")
        lines.append(f"  Ключевые признаки: {r['Ключевые признаки']}")
        lines.append(f"  Почему важен: {r['Почему важен']}")
        lines.append(f"  Рекомендация: {r['Рекомендация']}")
        lines.append(f"  Основной город: {r['Основной город']}")
        lines.append(f"  Основной вуз: {r['Основной вуз']}")
        lines.append("-" * 70)

    return "\n".join(lines)


# ============================================================
# стримлит страница
# ============================================================
def page(card=None):
    st.markdown("## 🧩 Кластеризация пользователей ВКонтакте")

    # -------------------------
    # 1) Автозагрузка CSV (без лишнего file_uploader, если файл найден)
    # -------------------------
    default_path = find_default_csv()
    df = None

    if default_path:
        df = read_csv_from_path(str(default_path))
        st.caption(f"Датасет загружен автоматически: **{default_path.as_posix()}**")
    else:
        st.warning("Файл vk_users_10000.csv не найден. Загрузите CSV вручную:")
        uploaded = st.file_uploader("CSV файл", type=["csv"])
        if uploaded is None:
            return
        df = read_csv_from_bytes(uploaded.getvalue())

    # Сырые данные НЕ показываем

    # -------------------------
    # 2) Признаки и X
    # -------------------------
    num_cols, cat_cols = detect_columns(df)
    df_proc = df.copy()
    for c in cat_cols:
        df_proc[c] = df_proc[c].fillna("")

    pre = fit_preprocessor(df_proc, num_cols, cat_cols)
    X = transform_features(pre, df_proc)

    st.markdown("### Настройки")
    total_n = len(df_proc)
    st.write(f"Всего профилей: **{total_n:,}**")

    k = st.slider("Количество кластеров", 2, 10, 4)

    # -------------------------
    # 3) KMeans
    # -------------------------
    with st.spinner("Выполняю K-Means кластеризацию..."):
        km = MiniBatchKMeans(n_clusters=int(k), random_state=42, batch_size=1024)
        df_out = df.copy()
        df_out["cluster_kmeans"] = km.fit_predict(X).astype(int)

    # -------------------------
    # 4) UMAP
    # -------------------------
    with st.spinner("Строю UMAP-проекцию..."):
        reducer = umap.UMAP(
            n_neighbors=25,
            min_dist=0.10,
            n_components=2,
            metric="cosine",
            random_state=42
        )
        emb = reducer.fit_transform(X)

    # -------------------------
    # 5) Интеллектуальная сводка
    # -------------------------
    summary_rows = []
    for cl in sorted(df_out["cluster_kmeans"].unique()):
        part = df_out[df_out["cluster_kmeans"] == cl]
        dr = risk_drivers(part)

        score = risk_score_0_100(dr["alc_pos"], dr["smk_pos"], dr["pol_liberal"], dr["edu_low"])
        lvl = risk_level_ru(score)

        top_city = top_value(part["city"]) if "city" in part.columns else "—"
        top_uni = top_value(part["university"]) if "university" in part.columns else "—"
        top_life = top_value(part["main_in_life"]) if "main_in_life" in part.columns else "—"
        top_people = top_value(part["main_in_people"]) if "main_in_people" in part.columns else "—"
        top_edu = top_value(part["education_level"]) if "education_level" in part.columns else "—"
        top_pol = top_value(part["political"]) if "political" in part.columns else "—"

        size = int(len(part))
        share_pct = (size / total_n) * 100.0

        # ключевые признаки (коротко)
        key_facts = []
        key_facts.append(f"алк+ {dr['alc_pos']*100:.0f}%")
        key_facts.append(f"кур+ {dr['smk_pos']*100:.0f}%")
        key_facts.append(f"либ/индиф {dr['pol_liberal']*100:.0f}%")
        key_facts.append(f"низк.обр {dr['edu_low']*100:.0f}%")
        key_facts = ", ".join(key_facts)

        ctype = cluster_type_ru(dr, str(top_edu), str(top_life))
        main_factor = main_risk_factor(dr)
        why = why_danger_ru(dr["alc_pos"], dr["smk_pos"], dr["pol_liberal"], dr["edu_low"], str(top_life))

        summary_rows.append({
            "Кластер": int(cl),
            "Тип кластера": ctype,
            "Уровень риска": lvl,
            "Риск, % (0-100)": round(score, 1),
            "Доля, %": round(share_pct, 2),
            "Количество": size,
            "Главный фактор риска": main_factor,
            "Ключевые признаки": key_facts,
            "Почему важен": why,
            "Рекомендация": recommendation_ru(lvl),
            "Основной город": str(top_city),
            "Основной вуз": str(top_uni),
            "Ценности (топ)": str(top_life),
            "В людях (топ)": str(top_people),
            "Образование (топ)": str(top_edu),
            "Политика (топ)": str(top_pol),
        })

    summary_df = pd.DataFrame(summary_rows)

    order = {"ВЫСОКИЙ": 2, "СРЕДНИЙ": 1, "НИЗКИЙ": 0}
    summary_df["_ord"] = summary_df["Уровень риска"].map(order).fillna(0).astype(int)
    summary_df = summary_df.sort_values(["_ord", "Риск, % (0-100)", "Количество"], ascending=[False, False, False]).drop(columns=["_ord"])

    st.markdown("### Оперативная сводка по кластерам")

    def style_summary(df: pd.DataFrame):
        def risk_color(val):
            if val == "ВЫСОКИЙ":
                return "color:#ef4444; font-weight:800;"
            if val == "СРЕДНИЙ":
                return "color:#eab308; font-weight:800;"
            return "color:#22c55e; font-weight:800;"

        sty = (
            df.style
            # базовый фон таблицы — чистый чёрный
            .set_properties(**{
                "background-color": "#0b0f14",
                "color": "#e6edf3",
                "border-color": "#6d28d9",
                "font-size": "13px",
            })
            # рамки и заголовки
            .set_table_styles([
                # Вся таблица
                {
                    "selector": "",
                    "props": [
                        ("border", "1px solid #6d28d9"),
                        ("border-radius", "12px"),
                    ]
                },
                # Заголовки
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#0b0f14"),
                        ("color", "#ffffff"),
                        ("border", "1px solid #6d28d9"),
                        ("font-weight", "800"),
                    ]
                },
                # Ячейки
                {
                    "selector": "td",
                    "props": [
                        ("background-color", "#0b0f14"),
                        ("border", "1px solid rgba(109,40,217,0.55)"),
                    ]
                },
                # Подсветка строк при наведении
                {
                    "selector": "tbody tr:hover",
                    "props": [
                        ("background-color", "rgba(109,40,217,0.08)")
                    ]
                }
            ])
            # Цвет уровня риска
            .map(risk_color, subset=["Уровень риска"])
            # Полоса риска (без мутного фона)
            .bar(subset=["Риск, % (0-100)"], color="#ef4444", vmin=0, vmax=100)
        )

        return sty

    st.dataframe(
        style_summary(summary_df[[
            "Кластер", "Тип кластера",
            "Уровень риска", "Риск, % (0-100)",
            "Доля, %", "Количество",
            "Главный фактор риска",
            "Ключевые признаки",
            "Почему важен",
            "Рекомендация",
            "Основной город", "Основной вуз"
        ]]),
        use_container_width=True
    )

    # -------------------------
    # 6) UMAP график (SOC colors + белая легенда + скрытые оси)
    # -------------------------
    risk_level_map = dict(zip(summary_df["Кластер"], summary_df["Уровень риска"]))
    risk_score_map = dict(zip(summary_df["Кластер"], summary_df["Риск, % (0-100)"]))
    main_factor_map = dict(zip(summary_df["Кластер"], summary_df["Главный фактор риска"]))

    vis = df_out.copy()
    vis["umap_x"] = emb[:, 0]
    vis["umap_y"] = emb[:, 1]
    vis["Уровень риска"] = vis["cluster_kmeans"].map(risk_level_map)
    vis["Риск, %"] = vis["cluster_kmeans"].map(risk_score_map)
    vis["Главный фактор риска"] = vis["cluster_kmeans"].map(main_factor_map)

    color_map = {"НИЗКИЙ": "#22c55e", "СРЕДНИЙ": "#eab308", "ВЫСОКИЙ": "#ef4444"}

    fig = px.scatter(
        vis,
        x="umap_x",
        y="umap_y",
        color="Уровень риска",
        color_discrete_map=color_map,
        symbol="cluster_kmeans",
        opacity=0.88,
        hover_data=[c for c in [
            "cluster_kmeans", "Уровень риска", "Риск, %", "Главный фактор риска",
            "sex", "age", "city", "education_level", "university",
            "main_in_life", "main_in_people", "alcohol", "smoking", "political"
        ] if c in vis.columns],
        title="UMAP-проекция кластеров (визуализация сходства профилей)"
    )

    fig.update_layout(
        template="plotly_dark",
        height=760,
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(color="#e6edf3", size=14),
        legend=dict(
            title=dict(text="Уровень риска", font=dict(color="#ffffff", size=14)),
            font=dict(color="#ffffff", size=13),
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.12)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # скрываю "непонятные цифры" координат
    fig.update_xaxes(title="UMAP-проекция X", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title="UMAP-проекция Y", showgrid=False, zeroline=False, showticklabels=False)

    st.plotly_chart(fig, use_container_width=True)
    st.caption("UMAP — метод визуализации. Координаты X/Y не имеют физического смысла и показывают только относительную близость профилей.")

    # -------------------------
    # 7) DBSCAN (скрыт по умолчанию)
    # -------------------------
    show_dbscan = st.checkbox("Показать поиск нетипичных профилей (DBSCAN)", value=False)
    if show_dbscan:
        st.markdown("### DBSCAN: поиск аномальных/нетипичных профилей")

        with st.expander("Параметры DBSCAN", expanded=False):
            eps = st.slider("eps", 0.05, 5.0, 0.60, 0.05)
            min_samples = st.slider("min_samples", 3, 30, 10)

        with st.spinner("Запускаю DBSCAN на UMAP..."):
            db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
            labels_db = db.fit_predict(emb)

        noise_share = float((labels_db == -1).mean()) * 100.0

        if noise_share < 0.1:
            st.info("Нетипичные профили (шум) не выявлены при текущих параметрах.")
        else:
            st.write(f"Аномалии / шум (label=-1): **{noise_share:.1f}%**")
            st.dataframe(pd.Series(labels_db).value_counts().rename("Количество").to_frame(), use_container_width=True)

    # -------------------------
    # 8) Экспорт (TXT отчёт + опционально CSV)
    # -------------------------
    st.markdown("### Экспорт результата")

    report_text = build_text_report(summary_df, total_n)

    st.markdown(
        """
        <div style="
            background:#0f1622;
            color:#e6edf3;
            border-radius:16px;
            padding:16px 18px;
            border:1px solid rgba(109,40,217,0.55);
            box-shadow: 0 10px 26px rgba(0,0,0,0.45);
            ">
          <div style="font-weight:800; font-size:14px; margin-bottom:8px; color:#ffffff;">
            📄 Экспорт текстового отчёта по кластерам
          </div>
          <div style="font-size:12px; opacity:.9; margin-bottom:12px;">
            Содержит описание кластеров, доли и количество, главный фактор риска, ключевые признаки и рекомендации для ИБ.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        "Скачать отчёт (.txt)",
        data=report_text.encode("utf-8"),
        file_name="vk_clusters_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    # ===============================
    # Экспорт CSV
    # ===============================

    df_export = df_out.copy()
    df_export["risk_score_0_100"] = df_export["cluster_kmeans"].map(risk_score_map)
    df_export["risk_level_ru"] = df_export["cluster_kmeans"].map(risk_level_map)
    df_export["main_risk_factor"] = df_export["cluster_kmeans"].map(main_factor_map)

    csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.markdown(
        """
        <div class="card accent-purple" style="margin-top:14px;">
            <div style="font-size:15px; font-weight:800; margin-bottom:8px;">
                 Экспорт результатов кластеризации
            </div>
            <div style="font-size:12px; opacity:.9; line-height:1.5; margin-bottom:12px;">
                CSV содержит всех пользователей с присвоенными кластерами, 
                уровнем риска и главным фактором угрозы. 
                Файл можно использовать для отчётов, расследований и архивации.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        "Скачать CSV с метками кластеров",
        data=csv_bytes,
        file_name="vk_users_10000_clustered.csv",
        mime="text/csv",
        key="export_clusters",
        use_container_width=True
    )


# Для запуска отдельно:
# streamlit run vk_dasboard/modules/clustering.py
if __name__ == "__main__":
    st.set_page_config(
        page_title="Кластеризация VK — тестовый запуск",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    page()
