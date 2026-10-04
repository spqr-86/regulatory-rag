from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from config.settings import settings
from src.indexing.table_rows import explicit_table_chunks
from utils.logging import logger

FileLike = Union[str, os.PathLike, io.BufferedIOBase, io.BytesIO, io.StringIO]

# Chunk-cache version changes when normalization or occurrence identity changes.
PIPELINE_VERSION = "v3.3-table-rows"
PARSER_VERSION = "v1-complete-docling"
CHUNKER_MAX_TOKENS = 400
CHUNKER_MERGE_PEERS = True


# ANCHOR: parse/cache structural chunks; bind provenance per input before dedup.
# Input: files/streams. Output: documents and a per-file processing_report.
# Strict mode rejects any failed/empty input; cache keys fingerprint the pipeline.
@dataclass(frozen=True)
class ProcessingResult:
    source: str
    chunk_count: int
    error: str | None = None


class ProcessingError(ValueError):
    """One or more required inputs failed to produce chunks."""


# --- Constants for filtering and grouping ---
# bbox height threshold: below this the bbox is considered noise (visual artifacts, footer).
# Text is kept — bbox is just nulled (visual_proof won't work,
# but retrieval stays complete). Previously the item was dropped entirely — this
# caused short single-line regulatory clauses to be lost (see PPRF 2464).
MIN_BBOX_HEIGHT = 7
MAX_CHUNK_SIZE = settings.CHUNK_SIZE

# Whole-line boilerplate only: never remove arbitrary inline URLs or legal IDs.
_NOISE_PATTERNS = re.compile(
    r"^(?:Скачано с[^\n]*|Премиальная версия[^\n]*|"
    r"Страница\s+\d+(?:\s+(?:из|/)\s*\d+)?|"
    r"https?://(?:www\.)?1otruda\.ru/\S*|"
    r"\d{2}\.\d{2}\.\d{4},?\s+\d{2}:\d{2})$",
    re.IGNORECASE,
)


def _clean_noise(text: str) -> str:
    """Remove recognized boilerplate lines; preserve identifiers and inline URLs."""
    text = unicodedata.normalize("NFC", text).replace("–", "—")
    lines = [
        line
        for line in text.splitlines()
        if not _NOISE_PATTERNS.fullmatch(line.strip())
    ]
    return re.sub(r" {2,}", " ", "\n".join(lines)).strip()


def pipeline_fingerprint() -> dict:
    """Configuration/version identity for reusable parser output."""
    return {
        "pipeline": PIPELINE_VERSION,
        "parser": PARSER_VERSION,
        "packages": {
            p: version(p)
            for p in ("docling", "docling-core", "transformers", "tokenizers")
        },
        "chunker": {
            "max_tokens": CHUNKER_MAX_TOKENS,
            "merge_peers": CHUNKER_MERGE_PEERS,
            "tokenizer": "docling-core default (version pinned in fingerprint)",
        },
        "cleaning": {
            "noise": _NOISE_PATTERNS.pattern,
            "policy": "whole-line-boilerplate-only",
            "bbox": MIN_BBOX_HEIGHT,
        },
    }


def _dict_to_document(d: dict) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata") or {})


class DocumentProcessor:
    """
    File processor with BBox coordinate extraction for visualisation.
    Uses Docling for structural parsing.
    """

    def __init__(
        self,
        headers: Optional[
            List[Tuple[str, str]]
        ] = None,  # Deprecated, kept for interface compat
        chunk_size: Optional[int] = None,  # Deprecated
    ):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy Docling initialisation
        self._docling = DocumentConverter()
        from docling_core.transforms.chunker import HybridChunker

        self._chunker = HybridChunker(
            max_tokens=CHUNKER_MAX_TOKENS, merge_peers=CHUNKER_MERGE_PEERS
        )

    # ---------- public methods ----------

    def validate_files(self, files: Iterable[FileLike]) -> None:
        """Check total size of files being processed."""
        total = 0
        for f in files:
            size = self._safe_sizeof(f)
            if size is None:
                continue
            total += size

        if total and total > settings.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {settings.MAX_TOTAL_SIZE // 1024 // 1024}MB limit "
                f"({total // 1024 // 1024}MB provided)."
            )

    def process(
        self, files: Iterable[FileLike], *, strict: bool = False
    ) -> List[Document]:
        """Process files with caching."""
        files = list(files)
        self.validate_files(files)
        self.processing_report: list[ProcessingResult] = []

        all_chunks: List[Document] = []
        # Dedup identity includes source, content and structural/page location.
        seen_chunk_keys: set[tuple[str, str, str]] = set()
        # Per-source 0-based sequential counter. Assigned AFTER dedup so ids are
        # contiguous and stable per source (no gaps from dropped duplicates).
        # Powers RRF fusion / dedup (passage_identity) and range queries
        # (query_chunks_by_range). Kept an int.
        source_chunk_counter: dict[str, int] = {}

        for file_obj in files:
            display_name = getattr(file_obj, "name", str(file_obj))
            try:
                stream, display_name = self._get_stream_and_name(file_obj)

                # File hash for cache key
                file_hash = self._hash_bytes_stream(stream)
                cache_path = self._cache_path_for(
                    f"{file_hash}:{self._suffix_from_name(display_name)}"
                )

                if self._is_cache_valid(cache_path):
                    logger.info(f"[cache] {display_name}")
                    try:
                        chunks = self._load_from_cache(cache_path)
                    except (ValueError, KeyError, TypeError, OSError):
                        chunks = []
                else:
                    chunks = []
                if strict and any(
                    ch.metadata.get("type") == "fallback_lxml" for ch in chunks
                ):
                    # Older/non-strict caches must not prevent parser recovery.
                    chunks = []
                if not chunks:
                    logger.info(f"[process] {display_name}")
                    stream.seek(0)
                    chunks = self._convert_and_extract(stream, display_name, file_hash)
                    # Do not cache an empty result: it most likely came from a
                    # Docling conversion error (_convert_and_extract returns []
                    # on exception). Caching [] for CACHE_EXPIRE_DAYS would make
                    # a reindex silently skip the document until the cache
                    # expires. A genuinely empty document is harmless to re-parse.
                    if chunks and not any(
                        ch.metadata.get("type") == "fallback_lxml" for ch in chunks
                    ):
                        self._save_to_cache(chunks, cache_path)
                    else:
                        logger.warning(
                            f"[no-cache] empty or degraded result for {display_name}"
                        )

                if not chunks:
                    raise ProcessingError(f"{display_name}: no chunks produced")
                if strict and any(
                    ch.metadata.get("type") == "fallback_lxml" for ch in chunks
                ):
                    raise ProcessingError(
                        f"{display_name}: degraded DOCX fallback is not permitted in strict indexing"
                    )
                retained = 0

                # Bind source for fresh AND cached chunks before deduplication.
                for ch in chunks:
                    ch.metadata["source"] = display_name
                    source = ch.metadata.get("source", display_name)
                    key = (
                        source,
                        hashlib.sha256(ch.page_content.encode("utf-8")).hexdigest(),
                        json.dumps(
                            {
                                k: ch.metadata[k]
                                for k in ("page_no", "bbox", "doc_item_refs")
                                if k in ch.metadata
                            },
                            sort_keys=True,
                        ),
                    )
                    if key not in seen_chunk_keys:
                        cid = source_chunk_counter.get(source, 0)
                        ch.metadata["chunk_id"] = cid
                        source_chunk_counter[source] = cid + 1
                        all_chunks.append(ch)
                        seen_chunk_keys.add(key)
                        retained += 1
                self.processing_report.append(ProcessingResult(display_name, retained))

            except Exception as e:
                self.processing_report.append(
                    ProcessingResult(str(display_name), 0, str(e))
                )
                logger.error(
                    f"Failed to process '{getattr(file_obj, 'name', str(file_obj))}': {e}",
                    exc_info=True,
                )
                continue

        if strict:
            failures = [r for r in self.processing_report if r.error]
            if failures:
                raise ProcessingError(
                    "Required documents failed: "
                    + "; ".join(f"{r.source}: {r.error}" for r in failures)
                )
        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # ---------- conversion and extraction ----------

    def _convert_and_extract(
        self, stream: io.BufferedIOBase, source_name: str, file_hash: str
    ) -> List[Document]:
        """Convert via Docling and extract structural chunks."""
        import tempfile

        # Parser IR has a separate identity from chunker/cleaning configuration.
        from docling_core.types.doc import DoclingDocument

        suffix = self._suffix_from_name(source_name)
        ir_key = hashlib.sha256(
            json.dumps(
                {
                    "input": file_hash,
                    "format": suffix,
                    "parser_schema": PARSER_VERSION,
                    "docling": version("docling"),
                    "docling_core": version("docling-core"),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        ir_dir = self.cache_dir / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        ir_path = ir_dir / f"{ir_key}.json"
        if ir_path.exists():
            try:
                parsed = DoclingDocument.model_validate_json(
                    ir_path.read_text(encoding="utf-8")
                )
            except (ValueError, OSError):
                logger.warning(
                    f"Invalid structured cache for {source_name}; parsing again"
                )
            else:
                return self._process_docling_document(parsed, source_name)

        # Docling requires a file on disk
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            stream.seek(0)
            tmp.write(stream.read())
            tmp.flush()

            # Convert
            try:
                res = self._docling.convert(tmp.name)
            except Exception as e:
                logger.error(f"Docling conversion failed for {source_name}: {e}")
                if source_name.lower().endswith(".docx"):
                    return self._fallback_docx(stream, source_name)
                return []

            status = getattr(res.status, "value", res.status)
            if status != "success" or res.errors:
                raise ProcessingError(
                    f"{source_name}: conversion status={status}, errors={len(res.errors)}"
                )
            # Only complete conversion enters the reusable structural cache.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=ir_dir, delete=False
            ) as cached:
                pending = Path(cached.name)
                cached.write(res.document.model_dump_json())
            try:
                pending.replace(ir_path)
            finally:
                pending.unlink(missing_ok=True)
            return self._process_docling_document(res.document, source_name)

    def _fallback_docx(
        self, stream: io.BufferedIOBase, source_name: str
    ) -> List[Document]:
        """Extract text directly from DOCX via lxml when Docling fails (broken .rels paths)."""
        import zipfile

        from lxml import etree

        logger.info(f"[fallback] lxml extraction for {source_name}")
        try:
            stream.seek(0)
            z = zipfile.ZipFile(stream)
            tree = etree.fromstring(z.read("word/document.xml"))
            ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            texts = [
                el.text for el in tree.iter(f"{ns}t") if el.text and el.text.strip()
            ]
            full_text = " ".join(texts)
        except Exception as e:
            logger.error(f"Fallback lxml extraction failed for {source_name}: {e}")
            return []

        if not full_text.strip():
            return []

        chunk_size = 800
        words = full_text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            text = _clean_noise(" ".join(words[i : i + chunk_size]))
            if not text:
                continue
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "type": "fallback_lxml",
                        "parent_section": "",
                    },
                )
            )
        logger.info(f"[fallback] {source_name}: {len(chunks)} chunks")
        return chunks

    def _process_docling_document(self, doc: Any, source: str) -> List[Document]:
        chunks = []
        for chunk in self._chunker.chunk(doc):
            text = _clean_noise(chunk.text.strip())
            if not text:
                continue

            headings = chunk.meta.headings or []
            parent_section = headings[-1] if headings else "Document start"
            heading_path = " > ".join(headings) if headings else ""

            meta: dict = {
                "source": source,
                "type": "hybrid_chunk",
                "parent_section": parent_section,
                "heading_path": heading_path,
            }
            if chunk.meta.doc_items:
                refs = [getattr(di, "self_ref", None) for di in chunk.meta.doc_items]
                if all(isinstance(ref, str) for ref in refs):
                    meta["doc_item_refs"] = json.dumps(refs)
                # element_type: propagate Docling structural label (table/text/...)
                # so visual_enrichment can route table chunks (e.g. 29н periodicity,
                # 817н classifier) through the VLM. "table" wins if any item is a table.
                labels = [
                    str(getattr(di, "label", "")).lower() for di in chunk.meta.doc_items
                ]
                if any("table" in lbl for lbl in labels):
                    meta["element_type"] = "table"
                elif labels and labels[0]:
                    meta["element_type"] = labels[0].split(".")[-1]

                item = chunk.meta.doc_items[0]
                if hasattr(item, "prov") and item.prov:
                    prov = item.prov[0]
                    if hasattr(prov, "page_no"):
                        meta["page_no"] = prov.page_no
                    if hasattr(prov, "bbox") and prov.bbox:
                        bbox = (
                            prov.bbox.as_tuple()
                            if hasattr(prov.bbox, "as_tuple")
                            else prov.bbox
                        )
                        if abs(bbox[3] - bbox[1]) >= MIN_BBOX_HEIGHT:
                            meta["bbox"] = json.dumps(bbox)

            # Contextual embedding: prepend the section/article title to the
            # embedded text so retrieval can disambiguate near-duplicate wording
            # across different norms (e.g. ст.228.1 «Порядок извещения» vs 223н
            # «Сообщение о последствиях»). Lightweight, LLM-free variant of
            # Contextual Retrieval — the NPA article title IS the disambiguator.
            # Skip when the chunk text already opens with the heading (HybridChunker
            # often inlines it) to avoid duplicating the title.
            if (
                parent_section
                and parent_section != "Document start"
                and not text.startswith(parent_section)
            ):
                embed_text = f"{parent_section}\n{text}"
            else:
                embed_text = text
            chunks.append(Document(page_content=embed_text, metadata=meta))
        # Numbered tables (29н-style) -> one chunk per row with its parent row.
        return explicit_table_chunks(getattr(doc, "tables", None), chunks)

    # ---------- cache and utilities ----------

    def _cache_path_for(self, file_hash: str) -> Path:
        key = hashlib.sha256(
            (
                file_hash + ":" + json.dumps(pipeline_fingerprint(), sort_keys=True)
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _save_to_cache(self, chunks: List[Document], cache_path: Path) -> None:
        payload = {
            "schema_version": 2,
            "timestamp": datetime.now().timestamp(),
            "chunks": [
                {
                    "page_content": c.page_content,
                    "metadata": {
                        k: v
                        for k, v in c.metadata.items()
                        if k not in {"source", "chunk_id"}
                    },
                }
                for c in chunks
            ],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False))

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        raw = json.loads(cache_path.read_text())
        if raw.get("schema_version") != 2:
            raise ValueError(f"unsupported cache schema: {raw.get('schema_version')}")
        return [_dict_to_document(d) for d in raw["chunks"]]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        max_age = timedelta(days=settings.CACHE_EXPIRE_DAYS)
        return cache_age < max_age

    def _safe_sizeof(self, f: FileLike) -> Optional[int]:
        try:
            if isinstance(f, (str, os.PathLike)):
                return Path(f).stat().st_size
            if hasattr(f, "seek") and hasattr(f, "tell"):
                cur = f.tell()
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(cur, os.SEEK_SET)
                return size
        except Exception:
            return None
        return None

    def _get_stream_and_name(self, f: FileLike) -> Tuple[io.BytesIO, str]:
        if isinstance(f, (str, os.PathLike)):
            p = Path(f)
            with open(p, "rb") as fh:
                data = fh.read()
            return io.BytesIO(data), p.name
        if hasattr(f, "read"):
            raw = f.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            return io.BytesIO(raw), getattr(f, "name", "uploaded_file")
        raise TypeError(f"Unsupported file type: {type(f)}")

    def _hash_bytes_stream(self, stream: io.BytesIO) -> str:
        stream.seek(0)
        h = hashlib.sha256()
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        stream.seek(0)
        return h.hexdigest()

    def _suffix_from_name(self, name: str) -> str:
        suf = Path(name).suffix.lower()
        return suf if suf else ".bin"
