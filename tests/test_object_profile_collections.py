"""Presence check of object sheets in a department collection (spec object-profile §5.2 step 4)."""

from __future__ import annotations

import pytest

from eval.object_profile_collections import sheet_sources


@pytest.mark.unit
def test_sheet_sources_matches_by_basename():
    passages = [
        {"metadata": {"source": "source_docs_dept/int_unit_office_list.md"}},
        {"metadata": {"source": "ext_ppr_1479.pdf"}},
        {"metadata": {}},
    ]
    assert sheet_sources(
        passages, {"int_unit_office_list.md", "int_unit_depot_list.md"}
    ) == {"int_unit_office_list.md"}
