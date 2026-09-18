"""Regulatory Compliance Assistant — department Q&A portfolio screen.

Spec streamlit-portfolio-demo §5–§18: the root screen is the Department Q&A
first-contact view (header, unit selector, compact profile, examples, one input)
and, after a request, the structured result screen (answer, reasoning chain,
bases, sources, technical details). The generic Regulatory RAG chat is the
secondary line on ``pages/2_Общий_поиск.py``.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

from utils.logging import configure_logging  # noqa: E402

configure_logging()

import streamlit as st  # noqa: E402

# Must happen before any langchain_google_genai import (occurs inside src modules).
from src.infra.llm_factory import apply_ipv6_patch_for_googleapis  # noqa: E402

apply_ipv6_patch_for_googleapis()

from config.settings import settings  # noqa: E402
from src.department_qa.object_profile import (  # noqa: E402
    ObjectProfileError,
    typed_fields_prompt_lines,
)
from src.department_qa.service import answer_question  # noqa: E402
from src.department_qa.view import (  # noqa: E402
    basis_cards,
    build_reasoning_chains,
    clarification_fallback,
    compact_profile,
    evidence_cards,
    presentation_status,
    technical_details,
    unit_name,
)
from src.department_qa.wiring import (  # noqa: E402
    build_department_stack,
    stack_cache_key,
)
from src.ui_feedback import render_feedback  # noqa: E402

st.set_page_config(
    page_title="Regulatory Compliance Assistant", page_icon="🧭", layout="wide"
)

# Spec §25/§7: light, calm, generous whitespace, one accent, consistent radius.
_CSS = """
<style>
  .block-container { max-width: 980px; padding-top: 2.5rem; padding-bottom: 4rem; }
  div[data-testid="stAlert"] { border-radius: 8px; }
  div[data-testid="stExpander"] {
    border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 8px;
  }
  h1, h2, h3 { letter-spacing: -0.01em; }
  hr { margin: 1.8rem 0; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

_EXAMPLES = [
    "Какие требования применимы?",
    "Есть ли расхождения?",
    "Как часто требуется проверка?",
]

_TONE = {
    "success": st.success,
    "info": st.info,
    "warning": st.warning,
    "error": st.error,
}

_SOURCE_TYPE = {
    "external": "внешний",
    "internal": "локальный",
    "object": "профиль объекта",
}

# Spec §11: only stages that really run, mapped to user-facing labels.
_PROGRESS_LABELS = {
    "retrieval_started": "Поиск требований",
    "generation_started": "Формирование ответа по найденным основаниям",
    "verification_started": "Проверка доказательств",
}


@st.cache_resource(show_spinner=False)
def load_department_stack(cache_key):
    # cache_key is stack_cache_key(): it changes when Streamlit hot-reloads local
    # modules, so the stale cached stack is not reused across a reload.
    return build_department_stack()


def _unit_names(manifest) -> dict[str, str]:
    return {
        meta["unit_id"]: unit_name(meta.get("title", ""), meta["unit_id"])
        for meta in manifest.documents.values()
        if meta.get("unit_id")
    }


def _render_profile_detail(profile) -> None:
    with st.expander("Подробнее об объекте"):
        for field in typed_fields_prompt_lines(profile):
            st.markdown(f"**{field.label}:** {field.value}")


def _render_reasoning(response) -> None:
    chains = build_reasoning_chains(response)
    if not chains:
        return
    st.subheader("Почему такой вывод")
    for i, chain in enumerate(chains):
        if i:
            st.divider()
        blocks = []
        if chain.fact:
            blocks.append(f"**Факт**\n\n{chain.fact}")
        if chain.requirement:
            blocks.append(f"**Требование**\n\n{chain.requirement}")
        blocks.append(f"**Вывод**\n\n{chain.conclusion}")
        st.markdown("\n\n↓\n\n".join(blocks))


def _render_bases(response) -> None:
    groups = (
        ("Внешнее требование", response.external_basis),
        ("Локальный акт", response.internal_basis),
        ("Факт подразделения", response.object_facts),
    )
    groups = [(label, items) for label, items in groups if items]
    if not groups:
        return
    st.subheader("На чём основан ответ")
    for label, items in groups:
        st.markdown(f"**{label}**")
        for card in basis_cards(items, response.evidence):
            st.markdown(f"- {card.statement}")
            meta = " · ".join(part for part in (card.title, card.locator) if part)
            if meta:
                st.caption(meta)


def _render_sources(response) -> None:
    cards = evidence_cards(response)
    if not cards:
        return
    with st.expander(f"Источники, на которые опирается ответ ({len(cards)})"):
        for card in cards:
            st.markdown(f"**{card.title}**")
            meta = [f"Тип: {_SOURCE_TYPE[card.source_type]}"]
            if card.locator:
                meta.append(card.locator)
            if card.retrieval_score is not None:
                meta.append(f"Retrieval score: {card.retrieval_score}")
            st.caption(" · ".join(meta))
            st.markdown(card.excerpt)


def _render_technical(response, latency_s) -> None:
    rows = technical_details(
        response, model_name=settings.SIMPLE_MODEL_NAME, latency_s=latency_s
    )
    with st.expander("Технические детали"):
        for label, value in rows:
            st.markdown(f"**{label}:** {value}")


def _render_clarification(response) -> None:
    """P0a (spec §13.2): show the clarifying question(s); continuation is out of P0."""
    questions = response.clarifying_questions
    if not questions:
        return
    for i, question in enumerate(questions):
        st.markdown(question)
        for column, label in zip(st.columns(3), ("Да", "Нет", "Не знаю")):
            column.button(
                label,
                key=f"clarify_{i}_{label}",
                disabled=True,
                use_container_width=True,
            )
    st.caption(
        "Уточнение и повторный запрос появятся в следующей версии: снимок объекта "
        "остаётся неизменным."
    )


def _render_cards(cards) -> None:
    for card in cards:
        st.markdown(f"- {card.statement}")
        meta = " · ".join(part for part in (card.title, card.locator) if part)
        if meta:
            st.caption(meta)


def _render_clarification_context(response) -> None:
    """Spec §13.2: name what the norms say and what still has to be clarified.

    The model draft and applied conclusions stay hidden: they may rest on facts
    the object sheet marks unknown.
    """
    fallback = clarification_fallback(response)
    if fallback.requirements:
        st.subheader("Что говорят требования")
        _render_cards(fallback.requirements)
    if fallback.facts:
        st.subheader("Что известно об объекте")
        _render_cards(fallback.facts)


def _render_conflict(response) -> None:
    """P1 (spec §16): show the clashing bases; only called on a real conflict."""
    groups = (
        ("Внешнее требование", response.external_basis),
        ("Локальный акт", response.internal_basis),
    )
    groups = [(label, items) for label, items in groups if items]
    if not groups:
        return
    st.subheader("Обнаружено расхождение")
    for label, items in groups:
        st.markdown(f"**{label}**")
        for card in basis_cards(items, response.evidence):
            st.markdown(f"- {card.statement}")
            meta = " · ".join(part for part in (card.title, card.locator) if part)
            if meta:
                st.caption(meta)
    st.markdown("**Как обработано**")
    st.markdown("Требуется проверка специалистом.")


def _render_result(entry: dict) -> None:
    response = entry["response"]
    st.divider()
    st.subheader("Ваш вопрос")
    st.markdown(entry["question"])

    status = presentation_status(response)
    _TONE[status.tone](f"**{status.title}**\n\n{status.detail}")

    if status.code == "clarification_required":
        _render_clarification_context(response)
        _render_clarification(response)
    if status.code == "conflict":
        _render_conflict(response)

    # Spec §13.3/§32.5: never show a confident answer when evidence is lacking.
    if status.code == "sufficient":
        if response.answer:
            st.subheader("Ответ")
            st.markdown(response.answer)
        _render_reasoning(response)
        _render_bases(response)

    _render_sources(response)
    _render_technical(response, entry.get("latency_s"))
    render_feedback(response.trace_id)


if not settings.CORPUS_MANIFEST_PATH or not os.path.exists(
    settings.CORPUS_MANIFEST_PATH
):
    st.error(
        "Не задан корпус подразделений: укажите CORPUS_MANIFEST_PATH и переиндексируйте."
    )
    st.stop()

try:
    stack = load_department_stack(stack_cache_key())
except (RuntimeError, ObjectProfileError) as exc:  # fail-fast config errors, shown
    st.error(str(exc))
    st.stop()

manifest, config = stack.manifest, stack.config
units = sorted({m["unit_id"] for m in manifest.documents.values() if m.get("unit_id")})
names = _unit_names(manifest)

st.title("Regulatory Compliance Assistant")
st.caption(
    "Ответы с учётом нормативных документов, локальных актов и данных подразделения."
)

st.subheader("Контекст запроса")
unit = st.selectbox(
    "Подразделение",
    [None, *units],
    index=1 if units else 0,
    format_func=lambda u: (
        "Без подразделения — общие требования" if u is None else names.get(u, u)
    ),
)

if st.session_state.get("last_unit") != unit:
    st.session_state.pop("last_answer", None)
    st.session_state["last_unit"] = unit

profile = config.profiles.get(unit) if unit else None
if parts := compact_profile(profile):
    st.markdown(" · ".join(parts))
if profile is not None:
    if profile.as_of_date is not None:
        st.caption(f"Данные на {profile.as_of_date:%d.%m.%Y}")
    else:
        st.caption("Данные: дата заполнения не указана")
    _render_profile_detail(profile)

st.subheader("Что хотите проверить?")
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

example_clicked = False
for column, example in zip(st.columns(len(_EXAMPLES)), _EXAMPLES):
    if column.button(example, use_container_width=True):
        st.session_state.question_input = example
        example_clicked = True

question = st.text_input(
    "Вопрос",
    key="question_input",
    placeholder="Задайте вопрос по требованиям к выбранному подразделению",
)
submit = st.button("Спросить", type="primary")

if (submit or example_clicked) and question.strip():
    with st.status("Обработка запроса…", expanded=True) as progress:

        def on_progress(stage: str) -> None:
            label = _PROGRESS_LABELS.get(stage)
            if label:
                progress.write(label)

        started = time.perf_counter()
        response = answer_question(
            question.strip(),
            unit,
            stack.search_fn,
            stack.model_fn,
            snapshot_id=manifest.snapshot_id,
            profile=profile,
            prompt_version=config.prompt_version,
            progress_fn=on_progress,
        )
        latency_s = time.perf_counter() - started
        progress.update(label="Готово", state="complete", expanded=False)
    st.session_state["last_answer"] = {
        "question": question.strip(),
        "unit_id": unit,
        "response": response,
        "latency_s": latency_s,
    }

if last := st.session_state.get("last_answer"):
    _render_result(last)
