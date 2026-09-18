"""Compact profile: the few typed fields that explain a unit at a glance (spec §9)."""

from __future__ import annotations

from datetime import date

import pytest

from src.department_qa.object_profile import ObjectProfile, TypedObjectFields
from src.department_qa.view import compact_profile


def _profile(**overrides) -> ObjectProfile:
    fields = dict(
        people_in_object_zone=24,
        people_on_floor_total="unknown",
        people_in_building_total="unknown",
        permanent_workplaces_on_floor="unknown",
        evacuation_plan_present=True,
        room_categories=["В2"],
        aupt_present=False,
        extinguishers_total=4,
        outside_ladder_last_test_date="not_applicable",
    )
    fields.update(overrides)
    return ObjectProfile(
        unit_id="unit_office",
        document_id="d",
        source="s.md",
        content_sha256="sha",
        title="Лист",
        as_of_date=date(2026, 9, 12),
        sections={},
        typed_fields=TypedObjectFields(**fields),
    )


@pytest.mark.unit
def test_known_fields_become_compact_phrases():
    lines = compact_profile(_profile())
    assert "24 сотрудника" in lines
    assert "план эвакуации есть" in lines
    assert "АУПТ отсутствует" in lines
    assert "4 огнетушителя" in lines


@pytest.mark.unit
def test_absent_state_is_spelled_out():
    lines = compact_profile(_profile(evacuation_plan_present=False))
    assert "план эвакуации не разработан" in lines


@pytest.mark.unit
def test_unknown_and_not_applicable_fields_are_omitted():
    text = " · ".join(compact_profile(_profile()))
    assert "неизвестно" not in text
    assert "не применимо" not in text
    assert "лестниц" not in text.lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    "count, phrase",
    [
        (1, "1 сотрудник"),
        (2, "2 сотрудника"),
        (5, "5 сотрудников"),
        (21, "21 сотрудник"),
    ],
)
def test_russian_plural_forms(count, phrase):
    assert phrase in compact_profile(_profile(people_in_object_zone=count))


@pytest.mark.unit
def test_none_profile_is_empty():
    assert compact_profile(None) == []
