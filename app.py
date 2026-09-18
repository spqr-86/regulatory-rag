"""Regulatory Compliance Assistant — department Q&A portfolio screen.

Spec streamlit-portfolio-demo §5–§10: the root screen is the Department Q&A
first-contact view (header, unit selector, compact profile, examples, one input).
The generic Regulatory RAG chat is the secondary line on ``pages/2_Общий_поиск.py``.
"""

import os

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
    compact_profile,
    presentation_status,
    unit_name,
)
from src.department_qa.wiring import build_department_stack  # noqa: E402

st.set_page_config(
    page_title="Regulatory Compliance Assistant", page_icon="🧭", layout="wide"
)

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


@st.cache_resource(show_spinner=False)
def load_department_stack():
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


def _render_result(entry: dict) -> None:
    response = entry["response"]
    st.divider()
    st.subheader("Ваш вопрос")
    st.markdown(entry["question"])

    status = presentation_status(response)
    _TONE[status.tone](f"**{status.title}**\n\n{status.detail}")

    if status.code == "sufficient" and response.answer:
        st.subheader("Ответ")
        st.markdown(response.answer)


if not settings.CORPUS_MANIFEST_PATH or not os.path.exists(
    settings.CORPUS_MANIFEST_PATH
):
    st.error(
        "Не задан корпус подразделений: укажите CORPUS_MANIFEST_PATH и переиндексируйте."
    )
    st.stop()

try:
    stack = load_department_stack()
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
    with st.spinner("Поиск требований…"):
        response = answer_question(
            question.strip(),
            unit,
            stack.search_fn,
            stack.model_fn,
            snapshot_id=manifest.snapshot_id,
            profile=profile,
            prompt_version=config.prompt_version,
        )
    st.session_state["last_answer"] = {
        "question": question.strip(),
        "unit_id": unit,
        "response": response,
    }

if last := st.session_state.get("last_answer"):
    _render_result(last)
