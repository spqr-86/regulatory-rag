"""Shared 👍/👎 feedback block for Streamlit pages (issue #20, spec §18).

Extracted from ``pages/2_Общий_поиск.py`` so the department Q&A result screen can
reuse it. Votes live next to the telemetry events, so they are offered only when
events go to Postgres; otherwise the buttons are hidden rather than failing.
"""

from __future__ import annotations

import streamlit as st

from src.v7.feedback import default_feedback_writer
from utils.logging import logger


@st.cache_resource(show_spinner=False)
def get_feedback_writer():
    """One writer per Streamlit process; None when votes have nowhere to go (#20)."""
    return default_feedback_writer()


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
