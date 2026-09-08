import os

from dotenv import load_dotenv

load_dotenv()

from utils.logging import configure_logging

configure_logging()

import streamlit as st

# Must happen before any langchain_google_genai import (occurs inside src modules).
from src.infra.llm_factory import apply_ipv6_patch_for_googleapis

apply_ipv6_patch_for_googleapis()

from config.settings import settings
from src.ui_helpers import find_proof_images
from src.v7.feedback import default_feedback_writer
from src.v7.bridge import init_v7_pipeline
from utils.logging import logger

# V7 Graph
try:
    from src.v7.graph import build_graph as build_v7_graph
    from src.v7.runner import default_writer, run_query as run_v7_query

    V7_AVAILABLE = True
except Exception as e:
    logger.warning(f"V7 Graph is not available: {e}")
    V7_AVAILABLE = False

# On-demand reindex button
try:
    import index as index_module

    INDEX_AVAILABLE = True
except Exception as e:
    logger.warning(f"Index module not importable: {e}")
    INDEX_AVAILABLE = False

# =========================
#     PAGE CONFIG & UI
# =========================
st.set_page_config(page_title="Regulatory RAG", page_icon="📚", layout="wide")

st.title("📚 Regulatory RAG")
st.caption("Поиск по нормативной базе: ГОСТ, СНиП, СП, ТК РФ и другие документы.")


# =========================
#     RESOURCE LOADING
# =========================
@st.cache_resource(show_spinner=False)
def get_telemetry_writer():
    """One writer per Streamlit process — monitoring module 05, issue #17."""
    return default_writer()


@st.cache_resource(show_spinner=False)
def get_feedback_writer():
    """One writer per Streamlit process; None when votes have nowhere to go (#20)."""
    return default_feedback_writer()


@st.cache_resource(show_spinner=False)
def load_resources():
    if not os.path.exists(settings.CHROMA_DB_PATH) or not os.listdir(
        settings.CHROMA_DB_PATH
    ):
        st.error("База данных не найдена. Запустите 'python index.py' для её создания.")
        return None

    if not V7_AVAILABLE:
        st.error("V7 Graph недоступен. Проверьте конфигурацию.")
        return None

    try:
        from src.backends.vector_store import get_vector_store_backend

        vector_store = get_vector_store_backend(load_existing=True)
        init_v7_pipeline(vector_store)
        v7_app = build_v7_graph().compile()
        return v7_app
    except Exception as e:
        st.error(f"Ошибка инициализации V7 Graph: {e}")
        logger.warning(f"Failed to init V7 Graph: {e}")
        return None


# =========================
#        SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:8px 10px;border-radius:8px;background:#eef2ff;
                    border:1px solid #c7d2fe;font-size:12px;margin-bottom:4px;">
            Simple: <b>{settings.SIMPLE_LLM_PROVIDER}</b> · <b>{settings.SIMPLE_MODEL_NAME}</b>
        </div>
        <div style="padding:8px 10px;border-radius:8px;background:#ecfeff;
                    border:1px solid #a5f3fc;font-size:12px;margin-bottom:4px;">
            Complex: <b>{settings.COMPLEX_LLM_PROVIDER}</b> · <b>{settings.COMPLEX_MODEL_NAME}</b>
        </div>
        <div style="padding:8px 10px;border-radius:8px;background:#f0fdf4;
                    border:1px solid #bbf7d0;font-size:12px;">
            Embeddings: <b>{settings.EMBEDDING_PROVIDER}</b> · <b>{settings.EMBEDDING_MODEL_NAME}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("📄 Библиотека документов")
    st.caption(f"БД: `{settings.CHROMA_DB_PATH}`")

    if INDEX_AVAILABLE:
        if st.button("♻️ Переиндексировать библиотеку", use_container_width=True):
            with st.spinner("Индексация… это может занять несколько минут"):
                try:
                    index_module.main()
                    load_resources.clear()
                    st.success("Готово: библиотека переиндексирована.")
                except Exception as e:
                    st.error(f"Ошибка индексации: {e}")
    else:
        st.caption("Модуль индексации недоступен.")

    st.divider()

    st.subheader("🔧 Параметры отображения")
    show_sources_n = st.slider("Сколько источников показать", 3, 20, 8, 1)

    st.divider()
    if st.button("🧹 Очистить чат", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


v7_app = load_resources()
if v7_app is None:
    st.warning("Приложение не может быть запущено…")
    st.stop()


# =========================
#     FEEDBACK (issue #20)
# =========================
def _save_vote(query_id: str, verdict: int, comment: str | None = None) -> bool:
    """Write the vote; a dead database must not take the answer down with it."""
    writer = get_feedback_writer()
    if writer is None:
        return False
    try:
        writer.record(query_id, verdict, comment)
        return True
    except Exception as e:  # noqa: BLE001 — monitoring never breaks the answer
        logger.warning(f"Feedback not saved for {query_id}: {e}")
        st.caption("⚠️ Оценка не сохранилась — журнал недоступен.")
        return False


def render_feedback(query_id: str) -> None:
    """👍/👎 under an answer, with an optional comment on 👎.

    Streamlit reruns the script on a click, but the answer is already in
    ``session_state`` — nothing is recomputed and the text does not move. The
    vote itself is upserted by ``query_id``, so a changed mind replaces the row
    instead of adding one.
    """
    if not query_id or get_feedback_writer() is None:
        return

    votes = st.session_state.setdefault("votes", {})
    up, down, _ = st.columns([1, 1, 10])
    if up.button("👍", key=f"vote_up_{query_id}", help="Ответ помог"):
        if _save_vote(query_id, 1):
            votes[query_id] = 1
    if down.button("👎", key=f"vote_down_{query_id}", help="Ответ не помог"):
        if _save_vote(query_id, -1):
            votes[query_id] = -1

    vote = votes.get(query_id)
    if vote == 1:
        st.caption("Спасибо — засчитано как 👍.")
    elif vote == -1:
        st.caption("Засчитано как 👎.")
        with st.form(key=f"vote_note_{query_id}", clear_on_submit=False):
            comment = st.text_input(
                "Что не так с ответом? (необязательно)",
                key=f"vote_text_{query_id}",
            )
            if st.form_submit_button("Отправить") and _save_vote(query_id, -1, comment):
                st.caption("Комментарий сохранён.")


# =========================
#     CHAT HISTORY INIT
# =========================
if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Привет! Задай вопрос — найду ответ по ГОСТ, СНиП, СП, ТК РФ.",
        }
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            for img_path in find_proof_images(m["content"]):
                st.image(img_path, caption="Визуальное доказательство", width=600)
            render_feedback(m.get("query_id", ""))

# =========================
#       CHAT INPUT
# =========================
user_query = st.chat_input(
    "Спросите, например: «Какие требования к ширине эвакуационных путей?»"
)
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Ищу в нормативных документах..."):
            result, query_id = run_v7_query(
                v7_app, user_query, source="ui", writer=get_telemetry_writer()
            )

        if result.get("clarify_message"):
            answer = result["clarify_message"]
        elif result.get("abstain_reason"):
            answer = f"Не могу ответить: {result['abstain_reason']}"
        elif result.get("answer"):
            answer = result["answer"]
            passages = result.get("final_passages", [])
            if passages:
                with st.expander(f"🔎 Источники ({len(passages)})", expanded=False):
                    for i, p in enumerate(passages[:show_sources_n], 1):
                        src = p.get("metadata", {}).get("source", "N/A")
                        score = p.get("score", 0.0)
                        preview = p.get("text", "")[:500].strip().replace("\n", " ")
                        st.markdown(f"**{i}.** `{src}` · 🎯 {score:.2f}")
                        st.code(preview, language="markdown")
                        st.divider()
        elif result.get("final_passages"):
            passages = result["final_passages"]
            answer = "\n\n---\n\n".join(p.get("text", "") for p in passages[:10])
            with st.expander(f"🔎 Источники ({len(passages)})", expanded=False):
                for i, p in enumerate(passages[:show_sources_n], 1):
                    src = p.get("metadata", {}).get("source", "N/A")
                    score = p.get("score", 0.0)
                    preview = p.get("text", "")[:500].strip().replace("\n", " ")
                    st.markdown(f"**{i}.** `{src}` · 🎯 {score:.2f}")
                    st.code(preview, language="markdown")
                    st.divider()
        elif result.get("intent") == "noise":
            answer = "Задайте вопрос по нормативной документации."
        else:
            answer = "Не удалось получить ответ."

        st.markdown(answer)
        render_feedback(query_id)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "query_id": query_id}
    )
