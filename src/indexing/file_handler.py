from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Any

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from config.settings import settings
from utils.logging import logger

FileLike = Union[str, os.PathLike, io.BufferedIOBase, io.BytesIO, io.StringIO]

# ⚙️ Bump version: storage format has changed significantly.
# v2.2-grouped: bbox filter no longer drops text (only nulls bbox);
# MAX_CHUNK_SIZE from settings; page change flushes the chunk.
PIPELINE_VERSION = "v3.0-hybrid"

# --- Constants for filtering and grouping ---
# bbox height threshold: below this the bbox is considered noise (visual artifacts, footer).
# Text is kept — bbox is just nulled (visual_proof won't work,
# but retrieval stays complete). Previously the item was dropped entirely — this
# caused short single-line regulatory clauses to be lost (see PPRF 2464).
MIN_BBOX_HEIGHT = 7
BLACKLIST_PHRASES = ["Премиальная версия", "Скачано с", "Страница"]
MAX_CHUNK_SIZE = settings.CHUNK_SIZE

# Noise patterns — removed from chunk text before indexing.
# URL watermarks (e.g. https://1otruda.ru/#/document/99/727688582),
# page markers (14/34), timestamps (25.01.2026, 20:10).
_NOISE_PATTERNS = re.compile(
    r"https?://\S+"  # URL
    r"|(?<!\d)\d{1,2}/\d{2,3}(?!\d)"  # n/nn page marker (14/34) — not fractions in text
    r"|\d{2}\.\d{2}\.\d{4},?\s+\d{2}:\d{2}",  # date+time 25.01.2026, 20:10
    re.UNICODE,
)


def _clean_noise(text: str) -> str:
    """Remove noise: normalise Cyrillic, strip URLs/page markers/timestamps."""
    text = unicodedata.normalize("NFC", text)
    cleaned = _NOISE_PATTERNS.sub("", text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _document_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}


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

        self._chunker = HybridChunker(max_tokens=400, merge_peers=True)

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

    def process(self, files: Iterable[FileLike]) -> List[Document]:
        """Process files with caching."""
        self.validate_files(files)

        all_chunks: List[Document] = []
        seen_chunk_hashes: set[str] = set()

        for file_obj in files:
            try:
                stream, display_name = self._get_stream_and_name(file_obj)

                # File hash for cache key
                file_hash = self._hash_bytes_stream(stream)
                cache_path = self._cache_path_for(file_hash)

                if self._is_cache_valid(cache_path):
                    logger.info(f"[cache] {display_name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"[process] {display_name}")
                    stream.seek(0)
                    chunks = self._convert_and_extract(stream, display_name, file_hash)
                    self._save_to_cache(chunks, cache_path)

                # Deduplication
                for ch in chunks:
                    # Uniqueness by text + coordinates (if present);
                    # for simplicity we use text only, though different bboxes may share text
                    content_hash = hashlib.sha256(
                        ch.page_content.encode("utf-8")
                    ).hexdigest()
                    if content_hash not in seen_chunk_hashes:
                        all_chunks.append(ch)
                        seen_chunk_hashes.add(content_hash)

            except Exception as e:
                logger.error(
                    f"Failed to process '{getattr(file_obj, 'name', str(file_obj))}': {e}",
                    exc_info=True,
                )
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    # ---------- conversion and extraction ----------

    def _convert_and_extract(
        self, stream: io.BufferedIOBase, source_name: str, file_hash: str
    ) -> List[Document]:
        """Convert via Docling and extract structural chunks."""
        import tempfile

        # Docling requires a file on disk
        suffix = self._suffix_from_name(source_name)
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            stream.seek(0)
            tmp.write(stream.read())
            tmp.flush()

            # Convert
            try:
                res = self._docling.convert(tmp.name)
            except Exception as e:
                logger.error(f"Docling conversion failed for {source_name}: {e}")
                return []

            return self._process_docling_document(res.document, source_name)

    def _process_docling_document(self, doc: Any, source: str) -> List[Document]:
        chunks = []
        for chunk in self._chunker.chunk(doc):
            text = _clean_noise(chunk.text.strip())
            if not text:
                continue
            if any(p in text for p in BLACKLIST_PHRASES):
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
        return chunks

    # ---------- cache and utilities (logic unchanged) ----------

    def _cache_path_for(self, file_hash: str) -> Path:
        key = hashlib.sha256(
            f"{file_hash}:{PIPELINE_VERSION}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _save_to_cache(self, chunks: List[Document], cache_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "timestamp": datetime.now().timestamp(),
            "chunks": [_document_to_dict(c) for c in chunks],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False))

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        raw = json.loads(cache_path.read_text())
        if raw.get("schema_version") != 1:
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
