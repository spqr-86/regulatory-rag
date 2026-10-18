"""Tests for the provisioned Grafana dashboard (#19).

The dashboard is a file in the repository, not a thing assembled by clicks, so it
is checkable without Grafana: the provider must point at where the file is
mounted, every panel must name the provisioned data source, and every column a
panel reads must exist in the migration. What needs a live stack — that the
panels actually draw — is checked by hand against the running stack.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "grafana" / "dashboards" / "regulatory-rag.json"
PROVIDER = ROOT / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
COMPOSE = ROOT / "docker-compose.yml"
MIGRATION = ROOT / "db" / "migrations" / "001_init.sql"

DATASOURCE_UID = "regrag-postgres"


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panels(dashboard) -> list:
    return [p for p in dashboard["panels"] if p.get("type") != "row"]


@pytest.fixture(scope="module")
def queries(panels) -> list:
    return [t["rawSql"] for p in panels for t in p.get("targets", [])]


def test_provider_path_is_where_compose_mounts_the_dashboard():
    """A provider pointing elsewhere provisions an empty folder, silently."""
    provider = yaml.safe_load(PROVIDER.read_text(encoding="utf-8"))
    path = provider["providers"][0]["options"]["path"]
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    mounts = compose["services"]["grafana"]["volumes"]
    assert any(m.startswith(f"./grafana/dashboards:{path}") for m in mounts)


def test_every_panel_reads_the_provisioned_datasource(panels):
    """A panel on another uid stays empty however good its SQL is."""
    assert panels
    for panel in panels:
        assert panel["datasource"]["uid"] == DATASOURCE_UID, panel["title"]
        for target in panel["targets"]:
            assert target["datasource"]["uid"] == DATASOURCE_UID, panel["title"]


def test_the_spec_panels_are_all_there(panels):
    """The panels the issues ask for, matched by what they measure."""
    titles = " ".join(p["title"].lower() for p in panels)
    for expected in (
        "запрос",
        "цена за запрос",
        "латентност",
        "маршрут",
        "👎",
        "итого",
        "последние 👎",
    ):
        assert expected in titles, expected


def test_the_last_thumbs_down_panel_shows_what_a_miss_needs(panels):
    """A 👎 is only actionable with the question, the route and the comment (#20)."""
    matches = [p for p in panels if "последние 👎" in p["title"].lower()]
    assert len(matches) == 1
    panel = matches[0]
    assert panel["type"] == "table", panel["type"]
    sql = " ".join(t["rawSql"] for t in panel["targets"]).lower()
    for column in ("question", "path", "comment", "query_id"):
        assert column in sql, column
    assert "verdict = -1" in sql, sql
    assert "feedback" in sql and "join queries" in sql, sql


def test_period_switch_reaches_every_query(queries):
    """Without the macro a panel ignores the time picker and shows all history."""
    assert queries
    for sql in queries:
        assert "$__timeFilter" in sql or "$__timeGroup" in sql, sql


def test_price_and_latency_are_percentiles_not_averages(panels):
    """The issue asks for p50/p95; an average hides the tail it exists to show."""
    for panel in panels:
        if (
            "цена за запрос" in panel["title"].lower()
            or "латентност" in panel["title"].lower()
        ):
            sql = " ".join(t["rawSql"] for t in panel["targets"])
            assert "percentile_cont" in sql, panel["title"]
            assert not re.search(r"\bavg\s*\(", sql, re.IGNORECASE), panel["title"]


def test_panels_only_read_columns_the_migration_creates(queries):
    """A renamed column breaks the panel at query time; catch it at test time."""
    schema = MIGRATION.read_text(encoding="utf-8")
    known = set(re.findall(r"^\s{4}(\w+)\s+\w", schema, re.MULTILINE))
    assert "cost_usd" in known  # the scrape found real columns, not nothing

    reserved = {
        "select",
        "from",
        "where",
        "group",
        "order",
        "by",
        "as",
        "and",
        "or",
        "not",
        "count",
        "sum",
        "date_trunc",
        "percentile_cont",
        "within",
        "filter",
        "join",
        "using",
        "null",
        "nullif",
        "desc",
        "asc",
        "day",
        "time",
        "metric",
        "value",
        "queries",
        "feedback",
        "on",
        "case",
        "when",
        "then",
        "else",
        "end",
        "coalesce",
        "q",
        "f",
        "p50",
        "p95",
        "interval",
        "distinct",
        "left",
        "is",
        "cast",
        "float8",
        "limit",
        "in",
    }
    for sql in queries:
        sql = re.sub(r"\$\{\w+:sqlstring\}", "'fixture'", sql)
        body = re.sub(r"\$__\w+\([^)]*\)", " ", sql)
        body = re.sub(r"'[^']*'", " ", body)
        words = {
            w.lower() for w in re.findall(r"[a-z_][a-z_0-9]*", body, re.IGNORECASE)
        }
        unknown = words - known - reserved
        assert not unknown, f"{sql}\nunknown identifiers: {sorted(unknown)}"


def test_default_sources_exclude_eval_and_run_selector_includes_null(dashboard):
    variables = {v["name"]: v for v in dashboard["templating"]["list"]}
    assert set(variables["source"]["current"]["value"]) == {"ui", "api", "mcp"}
    assert variables["run_id"]["includeAll"] is False
    assert variables["run_id"]["current"]["value"] == ["__all__"]
    assert "'__all__' AS __value" in variables["run_id"]["query"]


@pytest.mark.parametrize(
    "source,run_id,expected",
    [
        ("ui", "__all__", 1),
        ("eval", "__all__", 2),
        ("eval", "batch-a", 1),
        ("ui", "batch-a", 0),
    ],
)
def test_filters_select_rows_including_feedback_and_live_nulls(
    queries, source, run_id, expected
):
    # Execute each exact filter expression from the provisioned SQL. SQLite shares
    # the IN/OR/COALESCE semantics used here; live PostgreSQL panels remain a manual check.
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE queries(source TEXT, run_id TEXT); INSERT INTO queries VALUES ('ui', NULL), ('eval', 'batch-a'), ('eval', 'batch-b');"
    )
    try:
        for sql in queries:
            alias = "q." if "JOIN queries q" in sql else ""
            predicate = f"{alias}source IN (${{source:sqlstring}}) AND ('__all__' IN (${{run_id:sqlstring}}) OR COALESCE({alias}run_id, '__live__') IN (${{run_id:sqlstring}}))"
            assert predicate in sql
            rendered = predicate.replace(
                "${source:sqlstring}", "'" + source + "'"
            ).replace("${run_id:sqlstring}", "'" + run_id + "'")
            actual = db.execute(
                "SELECT count(*) FROM queries q WHERE " + rendered
            ).fetchone()[0]
            assert actual == expected
    finally:
        db.close()
