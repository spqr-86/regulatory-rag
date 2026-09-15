"""Unit object profile: the object sheet parsed into template sections.

# ANCHOR: object profile
# Role: pass the unit object sheet to the answer as structure, not as search hits
#   (spec object-profile §1, §2.1, §2.4).
# Input: manifest.object_profiles (unit_id -> file) + SOURCE_DOCS_PATH; one Markdown
#   template: sheet title "## Лист особенностей объекта защиты…", sections "## N Title"
#   (N 1–9, title checked against SECTION_TITLES), optional "Дата заполнения: ДД.ММ.ГГГГ".
# Output: ObjectProfile with all nine sections (present / empty / missing).
# Failure modes: any grammar violation → ObjectProfileError at startup (data error,
#   not degradation). The parser does not read meaning: "информация уточняется" is present.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.department_qa.contract import ObjectSection
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
    )


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
