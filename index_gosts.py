"""Индексация ГОСТов и СНиП ЕРС-Инжиниринг в отдельную ChromaDB-коллекцию.

Не трогает основную коллекцию SIA (chroma_db_openai).
Результат: chroma_db_gosts/ с коллекцией wta_gosts.

Файлы ГОСТов содержат баг: word/_rels/document.xml.rels использует абсолютные пути
(Target="word/settings.xml") вместо относительных ("settings.xml"). Функция
_fix_docx_rels() патчит файл в памяти перед передачей в Docling.

Запуск:
    source venv/bin/activate
    python index_gosts.py
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import zipfile

from dotenv import load_dotenv

load_dotenv()

GOSTS_DOCS_PATH = "../water-treatment-analyzer/data/gosts/gosts_unpacked/ГОСТЫ и СНИП"
GOSTS_CHROMA_PATH = "./chroma_db_gosts"
GOSTS_COLLECTION_NAME = "wta_gosts"
ALLOWED_EXTS = {".docx", ".pdf"}
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400


def _collect_paths(root_dir: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in ALLOWED_EXTS:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def _fix_docx_rels(data: bytes) -> bytes:
    """Исправляет абсолютные Target-пути в word/_rels/document.xml.rels.

    Некоторые DOCX имеют Target="word/settings.xml" вместо Target="settings.xml".
    Docling/python-docx резолвит их как word/word/settings.xml → KeyError.
    """
    try:
        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf, "r") as zin:
            if "word/_rels/document.xml.rels" not in zin.namelist():
                return data
            rels = zin.read("word/_rels/document.xml.rels").decode("utf-8-sig")

        fixed = re.sub(r'Target="word/([^"]+)"', r'Target="\1"', rels)
        if fixed == rels:
            return data  # патч не нужен

        out = io.BytesIO()
        buf.seek(0)
        with (
            zipfile.ZipFile(buf, "r") as zin,
            zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout,
        ):
            for item in zin.infolist():
                if item.filename == "word/_rels/document.xml.rels":
                    zout.writestr(item, fixed.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))
        return out.getvalue()
    except Exception:
        return data  # если что-то пошло не так — возвращаем оригинал


def _extract_chunks_docling(path: str) -> list[dict]:
    """Извлечь чанки из DOCX через Docling (с патчем rels для битых файлов)."""
    from src.indexing.file_handler import DocumentProcessor

    ext = os.path.splitext(path)[1].lower()
    if ext != ".docx":
        # PDF и прочее — через стандартный DocumentProcessor
        proc = DocumentProcessor()
        chunks = proc.process([path])
        return [{"text": c.page_content, "metadata": c.metadata} for c in chunks]

    with open(path, "rb") as f:
        data = f.read()

    fixed_data = _fix_docx_rels(data)

    # Записываем во временный файл чтобы Docling мог читать с диска
    suffix = os.path.basename(path)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(fixed_data)
        tmp_path = tmp.name

    try:
        proc = DocumentProcessor()
        chunks = proc.process([tmp_path])
        # Восстанавливаем правильный source в метаданных
        doc_name = os.path.basename(path)
        result = []
        for c in chunks:
            meta = dict(c.metadata)
            meta["source"] = doc_name
            result.append({"text": c.page_content, "metadata": meta})
        return result
    finally:
        os.unlink(tmp_path)


def _chunk_text(
    text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Разбить текст на чанки с перекрытием."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # Очищаем markdown-артефакты (таблицы, ссылки), но сохраняем текст
    clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) → text
    clean_text = re.sub(
        r"^\|.*\|$", "", clean_text, flags=re.MULTILINE
    )  # markdown tables
    clean_text = re.sub(r"^#+\s*", "", clean_text, flags=re.MULTILINE)  # headings
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

    doc_name = os.path.basename(source)
    docs = splitter.create_documents([clean_text], metadatas=[{"source": doc_name}])
    return [
        {"text": d.page_content, "metadata": d.metadata}
        for d in docs
        if d.page_content.strip()
    ]


def main() -> None:
    from langchain_chroma import Chroma
    from src.infra.llm_factory import get_embedding_model
    from src.indexing.vector_store import _batches_by_tokens, _sanitize_metadata
    from utils.logging import logger

    logger.info("index_gosts: старт", docs_path=GOSTS_DOCS_PATH)

    if not os.path.isdir(GOSTS_DOCS_PATH):
        raise FileNotFoundError(f"Папка с ГОСТами не найдена: {GOSTS_DOCS_PATH}")

    if os.path.exists(GOSTS_CHROMA_PATH):
        logger.info("index_gosts: удаляем старую БД", path=GOSTS_CHROMA_PATH)
        shutil.rmtree(GOSTS_CHROMA_PATH, ignore_errors=True)

    file_paths = _collect_paths(GOSTS_DOCS_PATH)
    if not file_paths:
        logger.warning("index_gosts: файлы не найдены", path=GOSTS_DOCS_PATH)
        return

    logger.info("index_gosts: найдено файлов", count=len(file_paths))

    all_chunks: list[dict] = []
    errors = 0
    for i, path in enumerate(file_paths):
        try:
            chunks = _extract_chunks_docling(path)
            if not chunks:
                logger.warning(
                    "index_gosts: пустой результат", file=os.path.basename(path)
                )
                continue
            all_chunks.extend(chunks)
            logger.info(
                "index_gosts: файл обработан",
                i=i + 1,
                total=len(file_paths),
                chunks=len(chunks),
                file=os.path.basename(path)[:60],
            )
        except Exception as exc:
            errors += 1
            logger.error(
                "index_gosts: ошибка файла",
                file=os.path.basename(path)[:60],
                error=str(exc),
            )

    if not all_chunks:
        logger.warning("index_gosts: чанки не получены", errors=errors)
        return

    logger.info("index_gosts: всего чанков", count=len(all_chunks), errors=errors)

    os.makedirs(GOSTS_CHROMA_PATH, exist_ok=True)
    embeddings = get_embedding_model()
    is_openai = embeddings.__class__.__name__ in {
        "OpenAIEmbeddings",
        "AzureOpenAIEmbeddings",
    }

    vs = Chroma(
        collection_name=GOSTS_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=GOSTS_CHROMA_PATH,
    )

    from langchain_core.documents import Document

    docs = [
        Document(page_content=c["text"], metadata=c["metadata"]) for c in all_chunks
    ]
    total, done = len(docs), 0
    for batch in _batches_by_tokens(
        docs, max_tokens_per_batch=280_000, hard_batch_cap=128, is_openai=is_openai
    ):
        texts = [d.page_content for d in batch]
        metas = [_sanitize_metadata(d.metadata or {}) for d in batch]
        vs.add_texts(texts=texts, metadatas=metas)
        done += len(batch)
        logger.info("index_gosts: embedding прогресс", done=done, total=total)

    logger.info(
        "index_gosts: готово",
        collection=GOSTS_COLLECTION_NAME,
        path=GOSTS_CHROMA_PATH,
        chunks=total,
        errors=errors,
    )


if __name__ == "__main__":
    main()
