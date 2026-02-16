
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))


import streamlit as st
from pathlib import Path
from datetime import datetime
import psutil

from modules.clustering import page as clustering_page
from modules.profile_completion import page as profile_completion_page

from modules.comments_analysis import page as comments_analysis_page


from modules.hidden_groups import page as hidden_groups_page

# ---------- Page config ----------
st.set_page_config(
    page_title="Интеллектуальная система анализа рисков профилей ВКонтакте",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Load CSS ----------
css_path = Path("assets/style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------- State ----------
if "module" not in st.session_state:
    st.session_state["module"] = "🏠 Обзор"

# Demo risk (потом заменим на логику)
st.session_state.setdefault("risk_100", 35)
st.session_state.setdefault("risk_note", "Риск установлен вручную (демо). После кластеризации будет рассчитываться автоматически.")

def risk_level(v: int) -> str:
    if v >= 70:
        return "HIGH"
    if v >= 40:
        return "MEDIUM"
    return "LOW"

def level_css(level: str) -> str:
    return {"LOW": "risk-low", "MEDIUM": "risk-med", "HIGH": "risk-high"}.get(level, "risk-low")

# ---------- Sidebar ----------
st.sidebar.markdown("## 🧭 Меню")

items = [
    "🏠 Обзор",
    "🧩 Сегментация окружения",
    "🧠 Восстановление профиля",
    "🕵️ Латентные интересы",
    "💬 Контент-анализ (6 месяцев)",
]

module = st.sidebar.radio(
    "Разделы",
    items,
    index=items.index(st.session_state["module"])
)
st.session_state["module"] = module

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Настройки")

# Demo slider
risk_100 = st.sidebar.slider("Risk Score (демо)", 0, 100, int(st.session_state["risk_100"]), 1)
st.session_state["risk_100"] = risk_100

st.sidebar.checkbox("Показывать отладочную информацию", value=False, key="debug")

# ---------- Topbar metrics ----------
now = datetime.now().strftime("%d.%m.%Y • %H:%M:%S")

level = risk_level(risk_100)
marker_left = max(0, min(100, risk_100))

# ---------- TOP BAR ----------
st.markdown(
    f"""
    <div class="topbar">
      <div class="left">
        <span class="apptitle">🛡️ Интеллектуальная система анализа рисков профилей ВКонтакте</span>
        <span class="badge">РЕЖИМ: АНАЛИЗ</span>
        <span class="badge">ВРЕМЯ: {now}</span>
      </div>

      <div class="right">
        <div class="risk-wrap">
          <span class="risk-title">RISK</span>
          <div class="riskbar">
            <div class="riskfill" style="width:{risk_100}%"></div>
            <div class="riskmarker" style="left:calc({marker_left}% - 6px)"></div>
          </div>
          <span class="risknum">{risk_100}/100</span>
          <span class="risklevel {level_css(level)}">{level}</span>
        </div>
      </div>
    </div>

    <div class="topnote">{st.session_state.get("risk_note","")}</div>
    """,
    unsafe_allow_html=True
)

# ---------- Helpers ----------
def card(title: str, body_html: str, accent: str = "accent-blue"):
    st.markdown(
        f"""
        <div class="card {accent}">
          <div style="font-size:14px; opacity:.9; margin-bottom:6px;">{title}</div>
          <div style="font-size:13px; line-height:1.45;">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def go(target: str):
    st.session_state["module"] = target
    st.rerun()

# ---------- Pages ----------
if module == "🏠 Обзор":
    st.markdown("## 📌 Обзор")
    st.markdown(
        "Информационно-аналитическая платформа для выявления, сегментации и оценки "
        "социально-поведенческих рисков пользователей социальной сети ВКонтакте."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Статус системы", "<span class='pill'>Экспериментальная</span><span class='pill'>Модульная</span>", "accent-green")
    with c2:
        card("Источники данных", "Граф друзей 1–2 уровня, сообщества, активность и комментарии.", "accent-blue")
    with c3:
        card("Интерпретация", "Каждому кластеру автоматически присваивается профиль риска и семантическое описание.", "accent-yellow")
    with c4:
        card("Архитектура", "Новые модули добавляются без переписывания ядра.", "accent-purple")

    st.markdown("### Быстрые действия")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown(
            """
            <div class="tile tile-blue">
              <div class="tile-title">🧩 Сегментация окружения</div>
              <div class="tile-sub">Кластеризация друзей 1–2 уровня и профилирование групп.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Открыть", key="btn_blue", use_container_width=True):
            go("🧩 Сегментация окружения")

    with t2:
        st.markdown(
            """
            <div class="tile tile-purple">
              <div class="tile-title">🧠 Восстановление профиля</div>
              <div class="tile-sub">Предсказание недостающих атрибутов профиля (демо-страница).</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Открыть", key="btn_purple", use_container_width=True):
            go("🧠 Восстановление профиля")

    with t3:
        st.markdown(
            """
            <div class="tile tile-yellow">
              <div class="tile-title">🕵️ Латентные интересы</div>
              <div class="tile-sub">Поиск скрытых интересов и сообществ (демо-страница).</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Открыть", key="btn_yellow", use_container_width=True):
            go("🕵️ Латентные интересы")

    with t4:
        st.markdown(
            """
            <div class="tile tile-green tile-big">
              <div class="tile-title">💬 Контент-анализ</div>
              <div class="tile-sub">Анализ комментариев (демо-страница).</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Открыть", key="btn_green", use_container_width=True):
            go("💬 Контент-анализ (6 месяцев)")

elif module == "🧩 Сегментация окружения":
    clustering_page(card)

elif module == "🧠 Восстановление профиля":
    profile_completion_page(card)

elif module == "🕵️ Латентные интересы":
    hidden_groups_page(card)

elif module == "💬 Контент-анализ (6 месяцев)":
    comments_analysis_page(card)
