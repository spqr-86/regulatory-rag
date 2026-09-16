"""Object profile sheet grammar and loading (spec object-profile §2.1, §5.1)."""

from __future__ import annotations

import hashlib
from datetime import date

import pytest

from src.department_qa.object_profile import (
    SECTION_TITLES,
    ObjectProfileError,
    load_profiles,
    parse_profile,
    profile_evidence,
    typed_fields_prompt_lines,
)
from src.indexing.manifest import load_manifest

FULL = {n: f"Текст раздела {n}." for n in SECTION_TITLES}

FIELDS = {
    2: ["- Категории помещений: В2; Д"],
    3: ["- АУПТ на объекте: нет"],
    4: ["- Огнетушителей всего: 8"],
    7: [
        "- Людей в зоне объекта: 25",
        "- Людей на этаже всего: неизвестно",
        "- Людей в здании всего: 25",
        "- Постоянных рабочих мест на этаже: неизвестно",
        "- План эвакуации разработан: да",
    ],
    8: ["- Последнее испытание наружной пожарной лестницы: неизвестно"],
}


def _sheet(
    bodies, *, date_line="Дата заполнения: 15.09.2026", headings=None, fields=FIELDS
):
    parts = ["## Лист особенностей объекта защиты: тестовый объект", "Преамбула листа."]
    for n, body in bodies.items():
        heading = (headings or {}).get(n, f"## {n} {SECTION_TITLES[n]}")
        typed = ["### Типизированные поля", *fields.get(n, [])] if n in fields else []
        parts += [heading, *typed, "", body]
    if date_line is not None:
        parts.append(date_line)
    return "\n".join(parts) + "\n"


def _parse(text):
    return parse_profile(
        text,
        unit_id="unit_1",
        document_id="int_list",
        source="list.md",
        content_sha256="abc",
    )


@pytest.mark.unit
def test_full_sheet_has_nine_present_sections():
    profile = _parse(_sheet(FULL))
    assert list(profile.sections) == [f"obj_s{n}" for n in range(1, 10)]
    assert {s.presence for s in profile.sections.values()} == {"present"}
    assert profile.sections["obj_s3"].title == SECTION_TITLES[3]
    assert profile.sections["obj_s3"].number == 3
    assert profile.title == "Лист особенностей объекта защиты: тестовый объект"
    assert "Преамбула" not in "".join(s.text for s in profile.sections.values())
    assert profile.typed_fields.people_in_object_zone == 25
    assert profile.typed_fields.people_on_floor_total == "unknown"
    assert profile.typed_fields.evacuation_plan_present is True
    assert profile.typed_fields.room_categories == ["В2", "Д"]
    assert "Типизированные поля" not in profile.sections["obj_s7"].text


@pytest.mark.unit
def test_typed_fields_have_stable_semantic_evidence_ids():
    profile = _parse(_sheet(FULL))

    lines = typed_fields_prompt_lines(profile)

    assert [line.id for line in lines] == [
        "obj_f_people_in_object_zone",
        "obj_f_people_on_floor_total",
        "obj_f_people_in_building_total",
        "obj_f_permanent_workplaces_on_floor",
        "obj_f_evacuation_plan_present",
        "obj_f_room_categories",
        "obj_f_aupt_present",
        "obj_f_extinguishers_total",
        "obj_f_outside_ladder_last_test_date",
    ]


@pytest.mark.unit
def test_typed_fields_are_individually_citable_evidence():
    profile = _parse(_sheet(FULL))

    evidence = {item.id: item for item in profile_evidence(profile)}

    field = evidence["obj_f_people_in_building_total"]
    assert field.level == "object"
    assert field.text == "Людей в здании всего: 25"
    assert field.locator == "Типизированное поле: Людей в здании всего"


@pytest.mark.unit
def test_empty_and_missing_sections():
    bodies = {**FULL, 4: "   "}
    del bodies[5]
    profile = _parse(_sheet(bodies))
    assert profile.sections["obj_s4"].presence == "empty"
    assert profile.sections["obj_s5"].presence == "missing"
    assert profile.sections["obj_s5"].text == ""


@pytest.mark.unit
def test_unknown_fact_text_is_present():
    profile = _parse(
        _sheet({**FULL, 8: "Огнезащитная обработка: информация уточняется."})
    )
    assert profile.sections["obj_s8"].presence == "present"


@pytest.mark.unit
@pytest.mark.parametrize(
    "fields",
    [
        {**FIELDS, 7: FIELDS[7][1:]},
        {**FIELDS, 4: ["- Огнетушителей всего: -1"]},
        {**FIELDS, 3: ["- АУПТ на объекте: возможно"]},
        {**FIELDS, 2: ["- Категории помещений: неизвестно", "- Лишнее поле: 1"]},
        {**FIELDS, 8: ["- Последнее испытание наружной пожарной лестницы: [дата]"]},
    ],
    ids=["missing", "negative", "bad-bool", "extra", "placeholder"],
)
def test_typed_field_grammar_violations_raise(fields):
    with pytest.raises(ObjectProfileError):
        _parse(_sheet(FULL, fields=fields))


@pytest.mark.unit
def test_ladder_date_not_applicable_is_distinct_from_unknown():
    fields = {
        **FIELDS,
        8: ["- Последнее испытание наружной пожарной лестницы: не применимо"],
    }
    profile = _parse(_sheet(FULL, fields=fields))
    assert profile.typed_fields.outside_ladder_last_test_date == "not_applicable"


@pytest.mark.unit
@pytest.mark.parametrize(
    "date_line, expected",
    [
        ("Дата заполнения: 15.09.2026", date(2026, 9, 15)),
        ("Дата заполнения: 15.09.2026.", date(2026, 9, 15)),
        ("Дата заполнения: [дата]", None),
        (None, None),
    ],
    ids=["plain", "trailing-dot", "placeholder", "no-line"],
)
def test_as_of_date(date_line, expected):
    assert _parse(_sheet(FULL, date_line=date_line)).as_of_date == expected


@pytest.mark.unit
def test_date_line_is_cut_from_section_text():
    profile = _parse(_sheet(FULL))
    assert profile.sections["obj_s9"].text == "Текст раздела 9."


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        _sheet(FULL, date_line="Дата заполнения: 32.13.2026"),
        _sheet(FULL, date_line="Дата заполнения: сентябрь 2026"),
        _sheet(
            FULL, date_line="Дата заполнения: 01.09.2026\nДата заполнения: 02.09.2026"
        ),
        _sheet(FULL) + "## 10 Примечания\nтекст\n",
        _sheet(FULL) + "## Приложение\nтекст\n",
        _sheet(FULL, headings={3: "## 3 Системы пожаротушения"}),
        _sheet(FULL, headings={4: f"## 3 {SECTION_TITLES[3]}"}),
        "## Лист особенностей объекта защиты: пустой\nПреамбула.\n",
        "Нет заголовка листа\n## 1 Ответственный за пожарную безопасность на объекте\nx\n",
    ],
    ids=[
        "invalid-date",
        "month-word",
        "two-date-lines",
        "section-10",
        "appendix-heading",
        "wrong-title",
        "duplicate-number",
        "no-sections",
        "no-sheet-title",
    ],
)
def test_grammar_violations_raise(text):
    with pytest.raises(ObjectProfileError):
        _parse(text)


@pytest.mark.unit
def test_title_match_ignores_case_and_spacing():
    heading = "## 3 системы   ПРОТИВОПОЖАРНОЙ защиты объекта"
    profile = _parse(_sheet(FULL, headings={3: heading}))
    assert profile.sections["obj_s3"].presence == "present"


@pytest.mark.unit
def test_load_profiles_takes_unit_from_manifest_and_hash_from_file(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    raw = _sheet(FULL).encode("utf-8")
    (docs / "office_list.md").write_bytes(raw)
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "snapshot_id: s\norganization_id: o\ndocuments:\n"
        "  - {file: office_list.md, document_id: int_office_list, source_type: internal,"
        " scope: unit, unit_id: unit_office, role: object_profile, title: t}\n",
        encoding="utf-8",
    )

    profiles = load_profiles(load_manifest(manifest_path), docs)

    profile = profiles["unit_office"]
    assert profile.unit_id == "unit_office"
    assert profile.document_id == "int_office_list"
    assert profile.source == "office_list.md"
    assert profile.content_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.unit
def test_load_profiles_missing_file_raises(tmp_path):
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "snapshot_id: s\norganization_id: o\ndocuments:\n"
        "  - {file: absent.md, document_id: d, source_type: internal,"
        " scope: unit, unit_id: u, role: object_profile, title: t}\n",
        encoding="utf-8",
    )
    with pytest.raises(ObjectProfileError):
        load_profiles(load_manifest(manifest_path), tmp_path)
