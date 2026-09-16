"""Unit object profile: the object sheet parsed into template sections.

# ANCHOR: object profile
# Role: pass the unit object sheet to the answer as structure, not as search hits
#   (spec object-profile §1, §2.1, §2.4).
# Input: manifest.object_profiles (unit_id -> file) + SOURCE_DOCS_PATH; one Markdown
#   template: sheet title "## Лист особенностей объекта защиты…", sections "## N Title"
#   (N 1–9, title checked against SECTION_TITLES), optional "Дата заполнения: ДД.ММ.ГГГГ".
# Output: ObjectProfile with all nine sections (present / empty / missing).
#   Typed fields are exposed as individually citable obj_f_* evidence.
# Failure modes: any grammar violation → ObjectProfileError at startup (data error,
#   not degradation). The parser does not read meaning: "информация уточняется" is present.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from src.department_qa.contract import Evidence, ObjectSection, TypedFieldLine
from src.indexing.manifest import Manifest

SECTION_TITLES: dict[int, str] = {
    1: "Ответственный за пожарную безопасность на объекте",
    2: "Категории помещений по взрывопожарной и пожарной опасности",
    3: "Системы противопожарной защиты объекта",
    4: "Первичные средства пожаротушения",
    5: "Дежурный персонал",
    6: "Контактные телефоны объекта",
    7: "Планировка и эвакуация",
    8: "Технические особенности",
    9: "Складские и подсобные помещения",
}
CONTACTS_SECTION = 6
SHEET_TITLE_PREFIX = "## Лист особенностей объекта защиты"

_SECTION_HEADING = re.compile(r"^## (\d+) (.+?)\s*$")
_DATE_LINE = re.compile(r"^\s*Дата заполнения:\s*(.*?)\s*$")
_DATE_VALUE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})\.?$")
_DATE_PLACEHOLDER = "[дата]"
_TYPED_HEADING = "### Типизированные поля"
_FIELD_LINE = re.compile(r"^- ([^:]+):\s*(.*?)\s*$")
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"

FIELD_SPECS = {
    "people_in_object_zone": (7, "Людей в зоне объекта", "int"),
    "people_on_floor_total": (7, "Людей на этаже всего", "int"),
    "people_in_building_total": (7, "Людей в здании всего", "int"),
    "permanent_workplaces_on_floor": (7, "Постоянных рабочих мест на этаже", "int"),
    "evacuation_plan_present": (7, "План эвакуации разработан", "bool"),
    "room_categories": (2, "Категории помещений", "list"),
    "aupt_present": (3, "АУПТ на объекте", "bool"),
    "extinguishers_total": (4, "Огнетушителей всего", "int"),
    "outside_ladder_last_test_date": (
        8,
        "Последнее испытание наружной пожарной лестницы",
        "date",
    ),
}


class TypedObjectFields(BaseModel):
    people_in_object_zone: int | Literal["unknown"]
    people_on_floor_total: int | Literal["unknown"]
    people_in_building_total: int | Literal["unknown"]
    permanent_workplaces_on_floor: int | Literal["unknown"]
    evacuation_plan_present: bool | Literal["unknown"]
    room_categories: list[str] | Literal["unknown"]
    aupt_present: bool | Literal["unknown"]
    extinguishers_total: int | Literal["unknown"]
    outside_ladder_last_test_date: date | Literal["unknown", "not_applicable"]


class ObjectProfileError(ValueError):
    """Object sheet does not follow the supported template."""


class ObjectProfile(BaseModel):
    unit_id: str
    document_id: str
    source: str
    content_sha256: str
    title: str
    as_of_date: Optional[date]
    sections: dict[str, ObjectSection]
    typed_fields: TypedObjectFields


def _parse_field_value(kind: str, raw: str, source: str, label: str):
    if raw == "неизвестно":
        return UNKNOWN
    if kind == "int":
        if not raw.isdigit():
            raise ObjectProfileError(f"{source}: bad value for {label!r}: {raw!r}")
        return int(raw)
    if kind == "bool":
        if raw not in {"да", "нет"}:
            raise ObjectProfileError(f"{source}: bad value for {label!r}: {raw!r}")
        return raw == "да"
    if kind == "list":
        values = [item.strip() for item in raw.split(";")]
        if not values or any(not item for item in values):
            raise ObjectProfileError(f"{source}: bad value for {label!r}: {raw!r}")
        return values
    if raw == "не применимо":
        return NOT_APPLICABLE
    if not _DATE_VALUE.match(raw):
        raise ObjectProfileError(f"{source}: bad value for {label!r}: {raw!r}")
    return _parse_date(raw, source)


def _extract_typed_fields(
    bodies: dict[int, list[str]], source: str
) -> TypedObjectFields:
    by_label = {
        label: (key, section, kind)
        for key, (section, label, kind) in FIELD_SPECS.items()
    }
    parsed = {}
    for section, lines in bodies.items():
        heading_rows = [i for i, line in enumerate(lines) if line == _TYPED_HEADING]
        expected_here = any(spec[0] == section for spec in FIELD_SPECS.values())
        if len(heading_rows) != (1 if expected_here else 0):
            raise ObjectProfileError(
                f"{source}: section {section} must contain {1 if expected_here else 0} typed field block(s)"
            )
        if not heading_rows:
            continue
        start = heading_rows[0]
        end = start + 1
        while end < len(lines) and lines[end].startswith("- "):
            match = _FIELD_LINE.match(lines[end])
            if not match:
                raise ObjectProfileError(
                    f"{source}: malformed typed field {lines[end]!r}"
                )
            label, raw = match.groups()
            if label not in by_label:
                raise ObjectProfileError(f"{source}: unknown typed field {label!r}")
            key, required_section, kind = by_label[label]
            if required_section != section:
                raise ObjectProfileError(
                    f"{source}: field {label!r} belongs to section {required_section}"
                )
            if key in parsed:
                raise ObjectProfileError(f"{source}: typed field {label!r} repeated")
            parsed[key] = _parse_field_value(kind, raw, source, label)
            end += 1
        del lines[start:end]
        if start < len(lines) and not lines[start].strip():
            del lines[start]
    missing = [key for key in FIELD_SPECS if key not in parsed]
    if missing:
        raise ObjectProfileError(
            f"{source}: missing typed fields: {', '.join(missing)}"
        )
    return TypedObjectFields(**parsed)


def _normalize(title: str) -> str:
    return " ".join(title.split()).casefold()


def _parse_date(value: str, source: str) -> Optional[date]:
    if value == _DATE_PLACEHOLDER:
        return None
    match = _DATE_VALUE.match(value)
    if not match:
        raise ObjectProfileError(f"{source}: bad fill date {value!r}")
    day, month, year = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ObjectProfileError(f"{source}: bad fill date {value!r}") from exc


def parse_profile(
    text: str, *, unit_id: str, document_id: str, source: str, content_sha256: str
) -> ObjectProfile:
    lines = text.splitlines()

    date_rows = [i for i, line in enumerate(lines) if _DATE_LINE.match(line)]
    if len(date_rows) > 1:
        raise ObjectProfileError(f"{source}: more than one fill date line")
    as_of_date = None
    if date_rows:
        as_of_date = _parse_date(_DATE_LINE.match(lines[date_rows[0]]).group(1), source)
        del lines[date_rows[0]]

    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None or not lines[first].startswith(SHEET_TITLE_PREFIX):
        raise ObjectProfileError(f"{source}: first line must be the sheet title")
    title = lines[first][3:].strip()

    bodies: dict[int, list[str]] = {}
    current: Optional[int] = None
    for line in lines[first + 1 :]:
        if line.startswith("## "):
            match = _SECTION_HEADING.match(line)
            number = int(match.group(1)) if match else None
            if number not in SECTION_TITLES:
                raise ObjectProfileError(f"{source}: unexpected heading {line!r}")
            if _normalize(match.group(2)) != _normalize(SECTION_TITLES[number]):
                raise ObjectProfileError(f"{source}: section {number} title {line!r}")
            if number in bodies:
                raise ObjectProfileError(f"{source}: section {number} repeated")
            bodies[number] = []
            current = number
        elif current is not None:
            bodies[current].append(line)
    if not bodies:
        raise ObjectProfileError(f"{source}: no sections")

    typed_fields = _extract_typed_fields(bodies, source)

    sections: dict[str, ObjectSection] = {}
    for number, section_title in SECTION_TITLES.items():
        section_id = f"obj_s{number}"
        if number not in bodies:
            body, presence = "", "missing"
        else:
            body = "\n".join(bodies[number]).strip()
            presence = "present" if body else "empty"
        sections[section_id] = ObjectSection(
            id=section_id,
            number=number,
            title=section_title,
            text=body,
            presence=presence,
        )

    return ObjectProfile(
        unit_id=unit_id,
        document_id=document_id,
        source=source,
        content_sha256=content_sha256,
        title=title,
        as_of_date=as_of_date,
        sections=sections,
        typed_fields=typed_fields,
    )


def typed_fields_prompt_lines(profile: ObjectProfile) -> list[TypedFieldLine]:
    return [
        TypedFieldLine(
            id=f"obj_f_{key}",
            section_id=f"obj_s{section}",
            label=label,
            value=_display_field_value(getattr(profile.typed_fields, key)),
        )
        for key, (section, label, _kind) in FIELD_SPECS.items()
    ]


def _display_field_value(value) -> str:
    if value == UNKNOWN:
        return "неизвестно"
    if value == NOT_APPLICABLE:
        return "не применимо"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, list):
        return "; ".join(value)
    return str(value)


def profile_evidence(profile: ObjectProfile) -> list[Evidence]:
    """Citable raw sections and individually addressable typed fields."""
    sections = [
        Evidence(
            id=s.id,
            level="object",
            text=s.text,
            source=profile.source,
            title=profile.title,
            locator=f"{s.number} {s.title}",
            document_id=profile.document_id,
        )
        for s in profile.sections.values()
        if s.presence == "present" and s.number != CONTACTS_SECTION
    ]
    fields = [
        Evidence(
            id=field.id,
            level="object",
            text=f"{field.label}: {field.value}",
            source=profile.source,
            title=profile.title,
            locator=f"Типизированное поле: {field.label}",
            document_id=profile.document_id,
        )
        for field in typed_fields_prompt_lines(profile)
    ]
    return sections + fields


def profile_prompt_block(
    unit_id: Optional[str], profile: Optional[ObjectProfile]
) -> tuple[str, list[ObjectSection]]:
    if unit_id is None:
        return "Объект не выбран: сведений об объекте нет.", []
    if profile is None:
        return (
            "Лист не предоставлен: сведений об объекте нет, факты объекта не придумывай.",
            [],
        )
    filled = (
        profile.as_of_date.strftime("%d.%m.%Y") if profile.as_of_date else "не указана"
    )
    sections = [s for s in profile.sections.values() if s.number != CONTACTS_SECTION]
    return f"{profile.title}. Дата заполнения: {filled}.", sections


def load_profiles(
    manifest: Manifest, source_dir: str | Path
) -> dict[str, ObjectProfile]:
    profiles: dict[str, ObjectProfile] = {}
    for unit_id, name in manifest.object_profiles.items():
        matches = sorted(Path(source_dir).rglob(name))
        if len(matches) != 1:
            raise ObjectProfileError(
                f"{name}: expected one file under {source_dir}, found {len(matches)}"
            )
        raw = matches[0].read_bytes()
        profiles[unit_id] = parse_profile(
            raw.decode("utf-8"),
            unit_id=unit_id,
            document_id=manifest.documents[name]["document_id"],
            source=name,
            content_sha256=hashlib.sha256(raw).hexdigest(),
        )
    return profiles
