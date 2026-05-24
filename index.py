# index.py
import os
import shutil

from dotenv import load_dotenv

from config.settings import settings
from src.backends.vector_store import get_vector_store_backend
from src.indexing.file_handler import DocumentProcessor
from utils.logging import logger

load_dotenv()


def _collect_paths(root_dir: str, allowed_exts: list[str]) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                paths.append(os.path.join(dirpath, name))
    return paths


def main():
    logger.info("Запуск процесса индексации...")

    if os.path.exists(settings.CHROMA_DB_PATH):
        logger.info(f"Удаление старой базы данных из: {settings.CHROMA_DB_PATH}...")
        shutil.rmtree(settings.CHROMA_DB_PATH, ignore_errors=True)

    # Инвалидация кэшей, привязанных к содержимому индекса.
    # Без этого BM25/Docling-pkl переживают destructive reindex и поиск идёт
    # по «призракам» удалённых чанков.
    if os.path.exists(settings.CACHE_DIR):
        logger.info(f"Очистка Docling-cache: {settings.CACHE_DIR}")
        shutil.rmtree(settings.CACHE_DIR, ignore_errors=True)
    # .bm25_cache.pkl no longer produced (CARD-2.1b: BM25 rebuilt on startup).
    # Kept here as a no-op cleanup for any stale files from older deployments.
    bm25_cache = ".bm25_cache.pkl"
    if os.path.exists(bm25_cache):
        logger.info(f"Удаление устаревшего BM25-cache: {bm25_cache}")
        os.remove(bm25_cache)

    # Собираем все файлы допустимых типов
    file_paths = _collect_paths(settings.SOURCE_DOCS_PATH, settings.ALLOWED_TYPES)
    if not file_paths:
        logger.warning(
            f"В папке {settings.SOURCE_DOCS_PATH} не найдено подходящих файлов."
        )
        return

    # Обработка через новый DocumentProcessor
    processor = DocumentProcessor()
    chunks = processor.process(file_paths)

    if chunks:
        logger.info(f"Всего будет проиндексировано {len(chunks)} чанков.")
        get_vector_store_backend(load_existing=False).create(chunks)
        logger.info("Индексация успешно завершена.")
    else:
        logger.warning("Чанки не получены. Проверьте документы/конвертацию.")


if __name__ == "__main__":
    main()
