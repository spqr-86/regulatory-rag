"""Unit tests for eval/ls/run.py helpers — no network."""

from __future__ import annotations

import pytest

from eval.ls.run import with_current_references

pytestmark = pytest.mark.unit


def test_rejudge_scores_against_current_references():
    # LangSmith pins an existing experiment to its dataset version: a rejudge after
    # editing references saw the old ones (16.10). The wrapper swaps them in.
    seen = []

    def judge(inputs, outputs, reference_outputs):
        seen.append(reference_outputs)
        return {"key": "correctness", "score": 1}

    current = {"Вопрос?": {"ground_truth": "новый", "ground_truth_extra": "доп"}}
    (wrapped,) = with_current_references([judge], current)
    wrapped({"question": " Вопрос? "}, {"answer": "a"}, {"ground_truth": "старый"})
    assert seen == [{"ground_truth": "новый", "ground_truth_extra": "доп"}]


def test_question_missing_from_current_set_fails_loudly():
    (wrapped,) = with_current_references(
        [lambda inputs, outputs, reference_outputs: 1], {}
    )
    with pytest.raises(KeyError):
        wrapped({"question": "нет такого"}, {}, {})


@pytest.mark.parametrize(
    "override,effective", [(None, "low"), ("none", "none"), ("medium", "medium")]
)
def test_experiment_metadata_records_effective_simple_effort(
    monkeypatch, tmp_path, override, effective
):
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    import dotenv
    import langsmith
    from langsmith.schemas import Example

    import eval.ls.run as run_module
    from config.settings import settings
    from eval.ls.dataset import DEFAULT_CSV, DEFAULT_LABELS, build_examples

    results = MagicMock()
    results.experiment_name = "offline"
    client = MagicMock()
    client.evaluate.return_value = results
    now = datetime.now(timezone.utc)
    client.list_examples.return_value = [
        Example(
            **e,
            created_at=now,
            modified_at=now,
            dataset_id="00000000-0000-0000-0000-000000000001",
        )
        for e in build_examples(DEFAULT_CSV, DEFAULT_LABELS)
    ]
    monkeypatch.setattr(langsmith, "Client", MagicMock(return_value=client))
    monkeypatch.setattr(dotenv, "load_dotenv", MagicMock())
    monkeypatch.setattr(run_module, "make_target", MagicMock())
    monkeypatch.setattr(run_module, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(settings, "OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setattr(settings, "SIMPLE_REASONING_EFFORT", override)
    snapshot = tmp_path / "build-j7ceu4r4"
    snapshot.mkdir()
    (snapshot / "snapshot-report.json").write_text("{}")
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(snapshot))
    assert (
        run_module.run_experiment(
            "verified", "offline", True, None, output_dir=tmp_path / "artifacts"
        )
        == "offline"
    )
    metadata = client.evaluate.call_args.kwargs["metadata"]
    assert metadata["effort"] == "low"
    assert metadata["simple_effort"] == effective
