"""Department Q&A screen: unit selector, law, LNA and object facts, checked sources (spec §4, §10)."""

import os

import streamlit as st
from dotenv import load_dotenv

from config.settings import settings
from src.department_qa.service import answer_question
from src.department_qa.view import basis_lines, profile_caption, status_banner
from src.department_qa.wiring import build_department_stack

load_dotenv()
st.set_page_config(page_title="Q&A подразделений", layout="wide")


@st.cache_resource(show_spinner=False)
def load_department_stack():
    return build_department_stack()


if not settings.CORPUS_MANIFEST_PATH or not os.path.exists(
    settings.CORPUS_MANIFEST_PATH
):
    st.error(
        "Не задан корпус подразделений: укажите CORPUS_MANIFEST_PATH и переиндексируйте."
    )
    st.stop()

stack = load_department_stack()
manifest, config = stack.manifest, stack.config
units = sorted({m["unit_id"] for m in manifest.documents.values() if m.get("unit_id")})


def _bullets(lines: list[str], empty: str) -> str:
    return "\n".join(f"- {line}" for line in lines) or empty


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
    st.caption(f"Режим: `{config.mode}` · коллекция `{config.collection}`")
    st.caption("Демонстрационный обезличенный корпус. Не юридическая консультация.")

st.title("Вопрос по пожарной безопасности")
question = st.text_input("Вопрос", placeholder="Как часто осматривать огнетушители?")

if st.button("Ответить", type="primary", disabled=not question):
    with st.spinner("Ищу в законодательстве и ЛНА…"):
        response = answer_question(
            question,
            unit,
            stack.search_fn,
            stack.model_fn,
            snapshot_id=manifest.snapshot_id,
            profile=config.profiles.get(unit) if unit else None,
            prompt_version=config.prompt_version,
        )

    kind, text = status_banner(response)
    getattr(st, kind)(text)
    for q in response.clarifying_questions:
        st.markdown(f"❓ {q}")

    law, lna = st.columns(2)
    with law:
        st.subheader("⚖️ Законодательство")
        st.markdown(
            _bullets(
                basis_lines(response.external_basis, response.evidence),
                "_Подтверждения не найдено._",
            )
        )
    with lna:
        st.subheader("🏢 ЛНА компании")
        st.markdown(
            _bullets(
                basis_lines(response.internal_basis, response.evidence),
                "_Подтверждения не найдено._",
            )
        )

    if config.mode == "v2":
        facts, applied = st.columns(2)
        with facts:
            st.subheader("🏠 Сведения объекта")
            st.markdown(
                _bullets(
                    basis_lines(response.object_facts, response.evidence),
                    "_Сведения объекта не использованы._",
                )
            )
            if caption := profile_caption(response):
                st.caption(caption)
        with applied:
            st.subheader("🔗 Применение к объекту")
            st.markdown(
                _bullets(
                    basis_lines(response.applied_conclusions, response.evidence),
                    "_Выводов о применении нет._",
                )
            )

    if response.answer:
        st.caption("Краткий черновик")
        st.markdown(response.answer)

    if response.next_step:
        st.caption(f"Следующий шаг: {response.next_step}")

    with st.expander(f"🔎 Источники ({len(response.evidence)})"):
        for e in response.evidence:
            place = f", {e.locator}" if e.locator else ""
            st.markdown(f"**[{e.id}]** {e.title}{place} · `{e.source}`")
            st.code(e.text[:1500], language="markdown")

    st.caption(f"trace_id: `{response.trace_id}`")
