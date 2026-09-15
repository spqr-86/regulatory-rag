"""Department Q&A screen: unit selector, law and LNA bases, checked sources (spec §4, §10)."""

import os

import streamlit as st
from dotenv import load_dotenv

from config.settings import settings
from src.department_qa.service import answer_question
from src.department_qa.view import basis_lines, status_banner
from src.department_qa.wiring import make_hybrid_search_fn, make_model_fn
from src.indexing.manifest import load_manifest

load_dotenv()
st.set_page_config(page_title="Q&A подразделений", layout="wide")


@st.cache_resource(show_spinner=False)
def load_department_resources():
    from src.backends.vector_store import get_vector_store_backend
    from src.infra.llm_factory import get_simple_llm
    from src.v7.bridge import init_v7_pipeline

    manifest = load_manifest(settings.CORPUS_MANIFEST_PATH)
    store = get_vector_store_backend(load_existing=True)
    init_v7_pipeline(store)  # builds the BM25 index over the same collection
    return manifest, make_hybrid_search_fn(store), make_model_fn(get_simple_llm())


if not settings.CORPUS_MANIFEST_PATH or not os.path.exists(
    settings.CORPUS_MANIFEST_PATH
):
    st.error(
        "Не задан корпус подразделений: укажите CORPUS_MANIFEST_PATH и переиндексируйте."
    )
    st.stop()

manifest, search_fn, model_fn = load_department_resources()
units = sorted({m["unit_id"] for m in manifest.documents.values() if m.get("unit_id")})

with st.sidebar:
    unit = st.selectbox(
        "Подразделение",
        [None, *units],
        format_func=lambda u: (
            "Не выбрано (только общекорпоративные ЛНА)" if u is None else u
        ),
    )
    st.caption(
        f"Корпус: `{manifest.snapshot_id}` · проверен {manifest.checked_at or '—'}"
    )
    st.caption("Демонстрационный обезличенный корпус. Не юридическая консультация.")

st.title("Вопрос по пожарной безопасности")
question = st.text_input("Вопрос", placeholder="Как часто осматривать огнетушители?")

if st.button("Ответить", type="primary", disabled=not question):
    with st.spinner("Ищу в законодательстве и ЛНА…"):
        response = answer_question(
            question, unit, search_fn, model_fn, snapshot_id=manifest.snapshot_id
        )

    kind, text = status_banner(response)
    getattr(st, kind)(text)

    if response.answer:
        st.markdown(response.answer)
    for q in response.clarifying_questions:
        st.markdown(f"❓ {q}")

    law, lna = st.columns(2)
    with law:
        st.subheader("⚖️ Законодательство")
        lines = basis_lines(response.external_basis, response.evidence)
        st.markdown(
            "\n".join(f"- {line}" for line in lines) or "_Подтверждения не найдено._"
        )
    with lna:
        st.subheader("🏢 ЛНА компании")
        lines = basis_lines(response.internal_basis, response.evidence)
        st.markdown(
            "\n".join(f"- {line}" for line in lines) or "_Подтверждения не найдено._"
        )

    if response.next_step:
        st.caption(f"Следующий шаг: {response.next_step}")

    with st.expander(f"🔎 Источники ({len(response.evidence)})"):
        for e in response.evidence:
            place = f", {e.locator}" if e.locator else ""
            st.markdown(f"**[{e.id}]** {e.title}{place} · `{e.source}`")
            st.code(e.text[:1500], language="markdown")

    st.caption(f"trace_id: `{response.trace_id}`")
