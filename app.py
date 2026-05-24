import os

import streamlit as st
from dotenv import load_dotenv

# Локальные импорты
from config.settings import settings
from src.final_chain import create_final_hybrid_chain
from src.ui_helpers import find_proof_images
from utils.logging import logger

# Multi-Agent RAG Workflow
try:
    from agents.multiagent_rag import MultiAgentRAGWorkflow

    MAS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Multi-Agent RAG is not available: {e}")
    MAS_AVAILABLE = False

# V7 Graph
try:
    from src.v7.bridge import init_v7_from_chroma
    from src.v7.graph import build_graph as build_v7_graph

    V7_AVAILABLE = True
except Exception as e:
    logger.warning(f"V7 Graph is not available: {e}")
    V7_AVAILABLE = False

# Индексация «по кнопке»
try:
    import index as index_module  # index.py должен содержать main()

    INDEX_AVAILABLE = True
except Exception as e:
    logger.warning(f"Index module not importable: {e}")
    INDEX_AVAILABLE = False

load_dotenv()

# =========================
#     PAGE CONFIG & UI
# =========================
st.set_page_config(
    page_title="AI Safety Compliance Assistant", page_icon="🤖", layout="wide"
)

# Верхний заголовок
left, right = st.columns([0.8, 0.2], vertical_alignment="center")
with left:
    st.title("🤖 AI Safety Compliance Assistant")
    st.caption("Ваш ИИ-помощник по нормативной документации (СНиП, ГОСТ, ОТ, ПБ).")
with right:
    st.markdown(
        f"""
        <div style="text-align:right">
            <div style="display:inline-block;padding:6px 10px;border-radius:8px;
                        background:#eef2ff;border:1px solid #c7d2fe;font-size:12px;">
                LLM: <b>{settings.LLM_PROVIDER}</b> · <b>{settings.MODEL_NAME}</b>
            </div><br/>
            <div style="display:inline-block;margin-top:6px;padding:6px 10px;border-radius:8px;
                        background:#ecfeff;border:1px solid #a5f3fc;font-size:12px;">
                Embeddings: <b>{getattr(settings, "EMBEDDING_PROVIDER", "?")}</b>
                · <b>{getattr(settings, "EMBEDDING_MODEL_NAME", "?")}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
#        SIDEBAR
# =========================
with st.sidebar:
    st.subheader("⚙️ Режим работы")

    # V7 — основной режим
    if V7_AVAILABLE and settings.USE_V7_GRAPH:
        v7_mode = st.toggle("🔬 V7 Graph (основной)", value=True)
    else:
        v7_mode = False
        st.warning("V7 Graph недоступен.")

    # MAS — только если V7 выключен
    mas_mode = False
    if not v7_mode and MAS_AVAILABLE:
        mas_mode = st.toggle("🧠 Multi-Agent RAG (устаревший)", value=False)

    if v7_mode:
        st.caption("Hybrid RAG → LangGraph → Gemini")
    elif mas_mode:
        st.caption("Router → RAG Agent → Verifier")
    else:
        st.caption("Classic RAG (legacy)")

    st.divider()

    st.subheader("📚 Библиотека документов")
    st.caption(f"БД: `{settings.CHROMA_DB_PATH}`")

    # Кнопка пересоздания индекса
    if INDEX_AVAILABLE:
        if st.button("♻️ Переиндексировать библиотеку", use_container_width=True):
            with st.spinner("Индексация… это может занять несколько минут"):
                try:
                    index_module.main()
                    # Обновить кэш ресурсов после индексации
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
        st.session_state.pop("last_answer", None)
        st.rerun()


# =========================
#     RESOURCE LOADING
# =========================
@st.cache_resource(show_spinner=False)
def load_resources():
    """
    Грузим один раз:
      - гибридную RAG-цепочку (retriever + LLM)
      - Multi-Agent RAG Workflow
      - V7 Graph (if enabled)
    """
    # Проверяем наличие БД
    if not os.path.exists(settings.CHROMA_DB_PATH) or not os.listdir(
        settings.CHROMA_DB_PATH
    ):
        st.error("База данных не найдена. Запустите 'python index.py' для её создания.")
        return None, None, None, None

    try:
        chain, retriever, agent_retriever = create_final_hybrid_chain()
    except Exception as e:
        st.error(f"Произошла ошибка при подготовке RAG-цепочки: {e}")
        return None, None, None, None

    # Multi-Agent RAG Workflow (использует OpenAI по умолчанию)
    agent = None
    if MAS_AVAILABLE:
        try:
            agent = MultiAgentRAGWorkflow(agent_retriever, llm_provider="gemini")
        except Exception as e:
            logger.warning(f"Failed to init MultiAgentRAGWorkflow: {e}")

    # V7 Graph
    v7_app = None
    if V7_AVAILABLE and settings.USE_V7_GRAPH:
        try:
            from src.backends.vector_store import get_vector_store_backend

            vector_store = get_vector_store_backend(load_existing=True)
            init_v7_from_chroma(vector_store)
            v7_app = build_v7_graph().compile()
        except Exception as e:
            logger.warning(f"Failed to init V7 Graph: {e}")

    return (chain, retriever, agent, v7_app)


loaded = load_resources()
if not loaded or loaded[0] is None:
    st.warning("Приложение не может быть запущено…")
    st.stop()

rag_chain, hybrid_retriever, agent, v7_app = loaded

if mas_mode and agent is None:
    st.sidebar.warning(
        "Multi-Agent RAG не удалось инициализировать. Используется Classic RAG."
    )

# =========================
#     CHAT HISTORY INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Здравствуйте! Задайте вопрос по охране труда, и я найду ответ в нормативных документах (приказы, ТК РФ, СанПиН, ГОСТы).",
        }
    ]

# Рендерим историю
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Рендеринг картинок в истории сложнее, так как st.image создает отдельный блок
        # Проверим, есть ли картинки в контенте сообщения (если это сохраненный ответ)
        if m["role"] == "assistant":
            for img_path in find_proof_images(m["content"]):
                st.image(img_path, caption="Визуальное доказательство", width=600)

# =========================
#       CHAT INPUT
# =========================
user_query = st.chat_input(
    "Спросите, например: «Какие требования к ширине эвакуационных путей?»"
)
if user_query:
    answer = ""  # Initialize answer to prevent NameError
    # Показываем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Ветка ответа
    with st.chat_message("assistant"):
        if v7_mode and v7_app:
            # --- V7 GRAPH MODE ---
            with st.spinner("Ищу в нормативных документах..."):
                result = v7_app.invoke({"query": user_query})

            # Determine answer
            if result.get("clarify_message"):
                answer = result["clarify_message"]
            elif result.get("abstain_reason"):
                answer = f"Не могу ответить: {result['abstain_reason']}"
            elif result.get("answer"):
                answer = result["answer"]
                # Источники к синтезированному ответу
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
                texts = [p.get("text", "") for p in passages[:10]]
                answer = "\n\n---\n\n".join(texts)
                with st.expander(f"🔎 Источники ({len(passages)})", expanded=False):
                    for i, p in enumerate(passages[:show_sources_n], 1):
                        src = p.get("metadata", {}).get("source", "N/A")
                        score = p.get("score", 0.0)
                        preview = p.get("text", "")[:500].strip().replace("\n", " ")
                        st.markdown(f"**{i}.** `{src}` · 🎯 {score:.2f}")
                        st.code(preview, language="markdown")
                        st.divider()
            elif result.get("intent") == "noise":
                answer = "Задайте вопрос по охране труда."
            else:
                answer = "Не удалось получить ответ."

            st.markdown(answer)

        elif mas_mode and agent:
            # --- MULTI-AGENT RAG MODE ---
            status_container = st.empty()
            answer_container = st.empty()

            statuses = []
            final_answer = ""
            final_chunks = []
            images_from_state = []

            for event in agent.stream_events(user_query):
                if not isinstance(event, dict):
                    continue

                if event.get("type") == "status":
                    statuses.append(event["text"])
                    with status_container:
                        for s in statuses[-3:]:
                            st.caption(s)
                elif event.get("type") == "final":
                    final_answer = event.get("answer", "")
                    final_chunks = event.get("chunks_found", [])
                    images_from_state = event.get("image_paths", [])
                    answer_container.markdown(final_answer)

            # After streaming finishes
            status_container.empty()

            # Combine and deduplicate images
            images_from_text = find_proof_images(
                final_answer
            )  # Check for images mentioned in final answer content
            all_images = list(set(images_from_state + images_from_text))

            for img_path in all_images:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Визуальное доказательство", width=600)

            st.session_state.last_answer = final_answer

            # Sources
            if final_chunks:
                with st.expander(
                    f"🔎 Показать источники ({len(final_chunks)})", expanded=False
                ):
                    for i, chunk in enumerate(final_chunks, start=1):
                        src = chunk.get("source", "N/A")
                        pg = chunk.get("page_no", "")
                        sim = chunk.get("similarity", 0.0)
                        content_txt = chunk.get("content", "")
                        preview = (
                            content_txt[:500].strip().replace("\n", " ")
                            if content_txt
                            else ""
                        )

                        header_parts = [f"**{i}. Источник:** `{src}`"]
                        if pg:
                            header_parts.append(f"стр. {pg}")
                        if sim and isinstance(sim, (int, float)) and sim > 0:
                            header_parts.append(f"🎯 {sim:.2f}")

                        st.markdown(" · ".join(header_parts))
                        st.code(preview, language="markdown")
                        st.divider()
            else:
                st.caption("Источники не найдены.")

            answer = final_answer  # Assign to answer for history

        else:
            # --- RAG MODE (Legacy) ---
            try:
                with st.spinner("Готовлю ответ (Classic RAG)..."):
                    response_text = st.write_stream(
                        rag_chain.stream(
                            {
                                "question": user_query,
                                "chat_history": st.session_state.get("messages", []),
                            }
                        )
                    )
                st.session_state.last_answer = response_text
                answer = response_text

                # Источники Legacy RAG
                try:
                    retrieved_docs = hybrid_retriever.invoke(user_query)[
                        :show_sources_n
                    ]
                except Exception:
                    retrieved_docs = []

                if retrieved_docs:
                    with st.expander(
                        f"🔎 Показать источники (топ-{len(retrieved_docs)})",
                        expanded=False,
                    ):
                        for i, doc in enumerate(retrieved_docs, start=1):
                            src = (
                                doc.metadata.get("source")
                                or doc.metadata.get("file_path")
                                or "N/A"
                            )
                            section = (
                                doc.metadata.get("section")
                                or doc.metadata.get("header")
                                or ""
                            )
                            preview = doc.page_content[:500].strip().replace("\n", " ")
                            st.markdown(
                                f"**{i}. Источник:** `{src}`"
                                + (f" · *{section}*" if section else "")
                            )
                            st.code(preview, language="markdown")
                            st.divider()
                else:
                    st.caption("Не удалось получить источники для отображения.")

            except Exception as e:
                st.error(f"Ошибка RAG-цепочки: {e}")
                answer = f"Извините, произошла ошибка: {e}"

    # Сохраняем сообщение ассистента в историю
    st.session_state.messages.append({"role": "assistant", "content": answer})
