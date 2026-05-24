from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Any

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    SectionHeaderItem,
    TextItem,
    ListItem,
)
from langchain_core.documents import Document

from config import constants
from config.settings import settings
from utils.logging import logger

FileLike = Union[str, os.PathLike, io.BufferedIOBase, io.BytesIO, io.StringIO]

# ⚙️ Обновляем версию, так как формат хранения кардинально меняется
# v2.2-grouped: bbox-фильтр больше не отбрасывает текст (только зануляет bbox);
# MAX_CHUNK_SIZE из settings; смена страницы flushит чанк.
PIPELINE_VERSION = "v2.3-noise-clean"

# --- Константы для фильтрации и группировки ---
# Порог высоты bbox: ниже него bbox считается мусором (визуальные артефакты, footer).
# Текст при этом сохраняется — bbox просто зануляется (visual_proof не сработает,
# но retrieval остаётся полным). Раньше item выбрасывался целиком — это роняло
# короткие однострочные пункты норм (см. ППРФ 2464).
MIN_BBOX_HEIGHT = 7
BLACKLIST_PHRASES = ["Премиальная версия", "Скачано с", "Страница"]
MAX_CHUNK_SIZE = settings.CHUNK_SIZE
# Parent-context chunking: parent (~1500 chars) stored as metadata on each child.
# Children (~400 chars, 50-char overlap) are what gets embedded and indexed.
CHILD_CHUNK_SIZE = 400
CHILD_OVERLAP = 50

# Паттерны шума — удаляются из текста чанка перед индексацией.
# URL-водяные знаки (напр. https://1otruda.ru/#/document/99/727688582),
# маркеры страниц (14/34), временны́е штампы (25.01.2026, 20:10).
_NOISE_PATTERNS = re.compile(
    r"https?://\S+"  # URL
    r"|(?<!\d)\d{1,2}/\d{2,3}(?!\d)"  # n/nn страница (14/34) — не дроби в тексте
    r"|\d{2}\.\d{2}\.\d{4},?\s+\d{2}:\d{2}",  # дата+время 25.01.2026, 20:10
    re.UNICODE,
)


def _clean_noise(text: str) -> str:
    """Удаляет URL-водяные знаки, маркеры страниц и временны́е штампы."""
    cleaned = _NOISE_PATTERNS.sub("", text)
    # Убираем лишние пробелы/переносы, оставшиеся после удаления
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _split_into_children(parent_text: str, parent_meta: dict) -> List[Document]:
    """Разбивает parent-чанк на child-чанки для индексации.

    Child-чанки (~400 chars) используются для embedding/поиска.
    Каждый child несёт parent_text в метаданных — именно он передаётся LLM.
    """
    if len(parent_text) <= CHILD_CHUNK_SIZE:
        meta = {**parent_meta, "parent_text": parent_text}
        return [Document(page_content=parent_text, metadata=meta)]

    children: List[Document] = []
    start = 0
    child_idx = 0
    while start < len(parent_text):
        end = min(start + CHILD_CHUNK_SIZE, len(parent_text))
        child_text = parent_text[start:end]
        meta = {**parent_meta, "parent_text": parent_text, "child_idx": child_idx}
        children.append(Document(page_content=child_text, metadata=meta))
        if end == len(parent_text):
            break
        start += CHILD_CHUNK_SIZE - CHILD_OVERLAP
        child_idx += 1
    return children


def _document_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}


def _dict_to_document(d: dict) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata") or {})


class DocumentProcessor:
    """
    Обработчик файлов с поддержкой извлечения координат (BBox) для визуализации.
    Использует Docling для структурного парсинга.
    """

    def __init__(
        self,
        headers: Optional[
            List[Tuple[str, str]]
        ] = None,  # Deprecated, kept for interface compat
        chunk_size: Optional[int] = None,  # Deprecated
        chunk_overlap: Optional[int] = None,  # Deprecated
    ):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ленивая инициализация Docling
        self._docling = DocumentConverter()

    # ---------- публичные методы ----------

    def validate_files(self, files: Iterable[FileLike]) -> None:
        """Проверяет суммарный размер загружаемых файлов."""
        total = 0
        for f in files:
            size = self._safe_sizeof(f)
            if size is None:
                continue
            total += size

        if total and total > constants.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total size exceeds {constants.MAX_TOTAL_SIZE // 1024 // 1024}MB limit "
                f"({total // 1024 // 1024}MB provided)."
            )

    def process(self, files: Iterable[FileLike]) -> List[Document]:
        """Обработка файлов с кэшированием."""
        self.validate_files(files)

        all_chunks: List[Document] = []
        seen_chunk_hashes: set[str] = set()

        for file_obj in files:
            try:
                stream, display_name = self._get_stream_and_name(file_obj)

                # Хэш файла для кэша
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

                # Дедупликация
                for ch in chunks:
                    # Уникальность определяем по тексту + координатам (если есть)
                    # Но для простоты пока по тексту, хотя разные bbox могут иметь один текст
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

    # ---------- конвертация и извлечение ----------

    def _convert_and_extract(
        self, stream: io.BufferedIOBase, source_name: str, file_hash: str
    ) -> List[Document]:
        """Конвертация через Docling и извлечение структурных чанков."""
        import tempfile

        # Docling требует файл на диске
        suffix = self._suffix_from_name(source_name)
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            stream.seek(0)
            tmp.write(stream.read())
            tmp.flush()

            # Конвертация
            try:
                res = self._docling.convert(tmp.name)
            except Exception as e:
                logger.error(f"Docling conversion failed for {source_name}: {e}")
                return []

            return self._process_docling_document(res.document, source_name)

    def _process_docling_document(self, doc: Any, source: str) -> List[Document]:
        """
        Итерация по структуре документа Docling.
        Сохраняем каждый элемент как Document с metadata.
        Реализована группировка (Grouping) мелких элементов.
        """
        chunks = []

        # Безопасное получение списка элементов
        items = []
        if hasattr(doc, "texts"):
            if callable(doc.texts):
                items = list(doc.texts())
            else:
                items = list(doc.texts)
        elif hasattr(doc, "body") and hasattr(doc.body, "children"):
            items = self._flatten_items(doc.body.children)

        current_section = "Начало документа"

        # Буфер для группировки
        current_chunk_text = []
        current_chunk_bbox = None  # [l, t, r, b]
        current_chunk_page = None

        def finalize_chunk(idx_for_id):
            nonlocal current_chunk_text, current_chunk_bbox, current_chunk_page
            if not current_chunk_text:
                return

            # Собираем текст
            full_text = "\n".join(current_chunk_text)

            meta = {
                "source": source,
                "type": "grouped_text",
                "chunk_id": idx_for_id,
                "parent_section": current_section,  # В метаданные, не в контент
            }

            if current_chunk_bbox:
                meta["bbox"] = json.dumps(current_chunk_bbox)
            if current_chunk_page:
                meta["page_no"] = current_chunk_page

            # Контент БЕЗ заголовка (чистый текст)
            chunks.append(Document(page_content=full_text, metadata=meta))

            # Сброс
            current_chunk_text = []
            current_chunk_bbox = None
            current_chunk_page = None

        def update_bbox(new_bbox, new_page):
            nonlocal current_chunk_bbox, current_chunk_page
            if not new_bbox:
                return

            # Принимаем только bbox с той же страницы. Если страница сменилась — это сложный кейс.
            # Для простоты: если страница сменилась, мы, возможно, захотим закрыть чанк.
            # Но пока будем обновлять страницу на последнюю актуальную.
            if current_chunk_page is not None and current_chunk_page != new_page:
                # Если страница поменялась, это сигнал закрыть чанк,
                # иначе bbox будет некорректным (координаты с разных страниц).
                return False

            current_chunk_page = new_page

            if current_chunk_bbox is None:
                current_chunk_bbox = list(new_bbox)
            else:
                # Merge: [min_l, min_t, max_r, max_b]
                # Docling bbox обычно: left, bottom, right, top (или top-left origin? Проверим ниже)
                # Обычно Docling возвращает [l, b, r, t] (bottom-up) или [l, t, r, b] (top-down)
                # В коде visual_proof мы уже обрабатываем это. Здесь просто берем min/max.

                # Предполодим [x0, y0, x1, y1] где x0<x1. Порядок Y зависит от системы координат.
                # Просто берем min для 0,1 и max для 2,3 — это безопасно для охвата.
                current_chunk_bbox[0] = min(current_chunk_bbox[0], new_bbox[0])
                current_chunk_bbox[1] = min(current_chunk_bbox[1], new_bbox[1])
                current_chunk_bbox[2] = max(current_chunk_bbox[2], new_bbox[2])
                current_chunk_bbox[3] = max(current_chunk_bbox[3], new_bbox[3])
            return True

        for i, item in enumerate(items):
            text = item.text.strip()
            if not text:
                continue

            # 1. Blacklist Filter
            if any(phrase in text for phrase in BLACKLIST_PHRASES):
                continue

            # 1b. Noise Cleanup — URL-watermarks, page markers, timestamps
            text = _clean_noise(text)
            if not text:
                continue

            # 2. BBox Extraction & Height Filter
            item_bbox = None
            item_page = None
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0]
                if hasattr(prov, "bbox") and prov.bbox:
                    bbox_tuple = (
                        prov.bbox.as_tuple()
                        if hasattr(prov.bbox, "as_tuple")
                        else prov.bbox
                    )
                    # Проверка высоты: bbox мусорный → зануляем, текст оставляем.
                    height = abs(bbox_tuple[3] - bbox_tuple[1])
                    if height >= MIN_BBOX_HEIGHT:
                        item_bbox = bbox_tuple

                if hasattr(prov, "page_no"):
                    item_page = prov.page_no

            # 3. Handling Headers (Explicit Break)
            if isinstance(item, SectionHeaderItem):
                # Закрываем предыдущий чанк
                finalize_chunk(i)
                current_section = text

                # Заголовок сам по себе тоже может быть чанком, или началом нового.
                # Обычно заголовок полезно иметь как отдельный короткий чанк или начало.
                # Давайте добавим его как отдельный чанк для навигации,
                # ИЛИ просто обновим контекст.
                # Лучше: Заголовок — это контекст. Мы его не добавляем как текст,
                # если только он не содержит полезной инфы.
                # Но часто заголовок — это и есть инфа.
                # Добавим заголовок как отдельный чанк.
                chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "type": "header",
                            "chunk_id": i,
                            "parent_section": current_section,
                            "bbox": json.dumps(item_bbox) if item_bbox else None,
                            "page_no": item_page,
                        },
                    )
                )
                continue

            # 4. Grouping Logic
            # Если страница сменилась — закрываем чанк
            if (
                current_chunk_page is not None
                and item_page is not None
                and current_chunk_page != item_page
            ):
                finalize_chunk(i)

            # Если размер превышен — закрываем чанк
            current_len = sum(len(t) for t in current_chunk_text)
            if current_len + len(text) > MAX_CHUNK_SIZE:
                finalize_chunk(i)

            # Добавляем в буфер
            if item_bbox:
                # Смена страницы внутри bbox-обновления: flush + new chunk.
                if not update_bbox(item_bbox, item_page):
                    finalize_chunk(i)
                    update_bbox(item_bbox, item_page)
            elif item_page and current_chunk_page is None:
                current_chunk_page = item_page
            current_chunk_text.append(text)

        # Finalize last chunk
        finalize_chunk(len(items))

        return chunks

    def _flatten_items(self, children: List[Any]) -> List[Any]:
        """Рекурсивно собирает текстовые элементы."""
        result = []
        for child in children:
            if isinstance(child, (TextItem, ListItem, SectionHeaderItem)):
                result.append(child)
            if hasattr(child, "children"):
                result.extend(self._flatten_items(child.children))
        return result

    def _get_item_type(self, item: Any) -> str:
        if isinstance(item, SectionHeaderItem):
            return "header"
        if isinstance(item, ListItem):
            return "list_item"
        return "text"

    # ---------- кэш и утилиты (без изменений логики) ----------

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
        max_age = timedelta(days=getattr(settings, "CACHE_EXPIRE_DAYS", 7))
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
