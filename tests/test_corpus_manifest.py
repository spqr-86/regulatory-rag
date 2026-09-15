"""Tests for corpus manifest: per-document metadata for the department Q&A (issue #36)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.indexing.manifest import ManifestError, apply_manifest, load_manifest

MANIFEST = """
snapshot_id: pb_demo_2026_09
checked_at: 2026-09-15
organization_id: org_example
documents:
  - file: ppr_1479.pdf
    document_id: ext_ppr_1479
    source_type: external
    title: Правила противопожарного режима
  - file: prikaz_instruktazhi.md
    document_id: int_prikaz_instr
    source_type: internal
    scope: company
    title: Приказ о порядке инструктажей
  - file: to_ognetushiteli_unit.md
    document_id: int_to_unit1
    source_type: internal
    scope: unit
    unit_id: unit_1
    title: Записка о ТО огнетушителей
"""


@pytest.fixture
def manifest(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST, encoding="utf-8")
    return load_manifest(p)


def _write(tmp_path, text):
    p = tmp_path / "manifest.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.unit
def test_chunks_get_document_metadata(manifest):
    chunks = [
        Document(page_content="a", metadata={"source": "ppr_1479.pdf"}),
        Document(page_content="b", metadata={"source": "to_ognetushiteli_unit.md"}),
    ]
    out = apply_manifest(chunks, manifest)

    ext, unit = out
    assert ext.metadata["source_type"] == "external"
    assert ext.metadata["document_id"] == "ext_ppr_1479"
    assert ext.metadata["version_id"] == "pb_demo_2026_09"
    assert "scope" not in ext.metadata
    assert unit.metadata["source_type"] == "internal"
    assert unit.metadata["organization_id"] == "org_example"
    assert unit.metadata["scope"] == "unit"
    assert unit.metadata["unit_id"] == "unit_1"
    assert unit.metadata["snapshot_id"] == "pb_demo_2026_09"
    # Scalar scope key: Chroma metadata cannot hold lists, $in filters on it.
    assert unit.metadata["audience"] == "unit_1"
    assert "audience" not in ext.metadata


@pytest.mark.unit
def test_company_scope_audience(manifest):
    chunk = Document(page_content="c", metadata={"source": "prikaz_instruktazhi.md"})
    assert apply_manifest([chunk], manifest)[0].metadata["audience"] == "company"


@pytest.mark.unit
def test_source_matched_by_basename(manifest):
    chunks = [Document(page_content="a", metadata={"source": "sub/dir/ppr_1479.pdf"})]
    assert apply_manifest(chunks, manifest)[0].metadata["document_id"] == "ext_ppr_1479"


@pytest.mark.unit
def test_file_absent_from_manifest_is_excluded(manifest):
    # Spec §5: a document with unknown applicability does not enter the snapshot.
    chunks = [Document(page_content="a", metadata={"source": "unknown.md"})]
    assert apply_manifest(chunks, manifest) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        "{file: x.md, document_id: d, source_type: internal, title: t}",
        "{file: x.md, document_id: d, source_type: internal, scope: unit, title: t}",
        "{file: x.md, document_id: d, source_type: other, title: t}",
        "{file: x.md, source_type: external, title: t}",
    ],
    ids=["internal-no-scope", "unit-no-unit_id", "bad-source_type", "no-document_id"],
)
def test_invalid_entry_rejected(tmp_path, entry):
    text = f"snapshot_id: s\norganization_id: o\ndocuments:\n  - {entry}\n"
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, text))


@pytest.mark.unit
def test_duplicate_file_or_document_id_rejected(tmp_path):
    text = (
        "snapshot_id: s\norganization_id: o\ndocuments:\n"
        "  - {file: a.md, document_id: d, source_type: external, title: t}\n"
        "  - {file: b.md, document_id: d, source_type: external, title: t}\n"
    )
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, text))


PROFILE_ENTRY = """  - file: office_list.md
    document_id: int_office_list
    source_type: internal
    scope: unit
    unit_id: unit_1
    role: object_profile
    title: Лист особенностей объекта
"""


@pytest.mark.unit
def test_object_profile_listed_and_kept_out_of_index(tmp_path):
    manifest = load_manifest(_write(tmp_path, MANIFEST + PROFILE_ENTRY))
    assert manifest.object_profiles == {"unit_1": "office_list.md"}
    assert "role" not in manifest.documents["office_list.md"]
    assert manifest.documents["office_list.md"]["document_id"] == "int_office_list"

    chunks = [
        Document(page_content="лист", metadata={"source": "office_list.md"}),
        Document(page_content="закон", metadata={"source": "ppr_1479.pdf"}),
    ]
    kept = apply_manifest(chunks, manifest)
    assert [c.metadata["source"] for c in kept] == ["ppr_1479.pdf"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry",
    [
        "{file: x.md, document_id: d, source_type: external, role: object_profile, title: t}",
        "{file: x.md, document_id: d, source_type: internal, scope: company, role: object_profile, title: t}",
        "{file: x.md, document_id: d, source_type: internal, scope: unit, unit_id: u, role: sheet, title: t}",
    ],
    ids=["profile-external", "profile-company-scope", "unknown-role"],
)
def test_invalid_role_rejected(tmp_path, entry):
    text = f"snapshot_id: s\norganization_id: o\ndocuments:\n  - {entry}\n"
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, text))


@pytest.mark.unit
def test_two_profiles_for_one_unit_rejected(tmp_path):
    text = (
        "snapshot_id: s\norganization_id: o\ndocuments:\n"
        "  - {file: a.md, document_id: a, source_type: internal, scope: unit, unit_id: u, role: object_profile, title: t}\n"
        "  - {file: b.md, document_id: b, source_type: internal, scope: unit, unit_id: u, role: object_profile, title: t}\n"
    )
    with pytest.raises(ManifestError):
        load_manifest(_write(tmp_path, text))


@pytest.mark.unit
def test_manifest_without_profiles_has_empty_mapping(manifest):
    assert manifest.object_profiles == {}
