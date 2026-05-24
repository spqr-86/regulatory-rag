from __future__ import annotations

import json
import base64
import io
import fitz  # pymupdf
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from config.settings import settings
from src.llm_factory import get_vision_llm
from src.vector_store import load_vector_store
from src.chroma_helpers import query_chunks_by_range


@dataclass
class ToolContext:
    """Isolated context for tool invocations — no global state."""

    retriever: BaseRetriever
    search_call_count: int = 0
    visual_proof_call_count: int = 0


def create_tool_context(retriever: BaseRetriever) -> ToolContext:
    """Create a fresh tool context for a workflow invocation."""
    return ToolContext(retriever=retriever)


def make_tools(ctx: ToolContext):
    """Create tool functions bound to a specific ToolContext.

    Returns (search_documents, visual_proof) tools.
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Search for information in the safety regulations.
        Returns RELEVANT text chunks with their ID, Source File, Page Number, and Bounding Box (bbox).

        The tool automatically expands context (fetches neighboring paragraphs) to provide complete information.

        Use this tool to find the information, then use the visual_proof tool with the extracted details.
        """
        ctx.search_call_count += 1
        if ctx.search_call_count > settings.MAX_SEARCH_CALLS:
            return (
                f"Лимит поисков достигнут ({settings.MAX_SEARCH_CALLS} из {settings.MAX_SEARCH_CALLS}). "
                "Сформулируй ответ на основе уже найденных данных."
            )
        if not ctx.retriever:
            return "Error: Retriever not initialized."

        # 1. Initial Retrieval
        initial_docs = ctx.retriever.invoke(query)

        if not initial_docs:
            return "No relevant documents found."

        # 2. Smart Context Extension & Deduplication
        hits = set()
        chunk_similarity = {}  # Store similarity scores for each chunk

        for doc in initial_docs:
            meta = doc.metadata
            if "chunk_id" in meta and "source" in meta:
                chunk_id = meta["chunk_id"]
                hits.add((meta["source"], chunk_id))
                # Store similarity if available
                if "similarity_score" in meta:
                    chunk_similarity[(meta["source"], chunk_id)] = meta[
                        "similarity_score"
                    ]

        # If we can't find chunk_id, fall back to simple return
        if not hits:
            # Fallback logic identical to old version
            results = []
            for i, doc in enumerate(initial_docs):
                meta = doc.metadata
                source = meta.get("source", "unknown")
                page = meta.get("page_no", "N/A")
                bbox = meta.get("bbox", "N/A")
                content = doc.page_content.replace("\n", " ")[:500]
                results.append(
                    f"[ID: {i}] File: {source} | Page: {page} | BBox: {bbox}\nContent: {content}..."
                )
            return "\n\n".join(results)

        # Load vector store ONCE for all range queries
        vs = load_vector_store()

        # Group hits by source
        hits_by_source = {}
        for source, cid in hits:
            if source not in hits_by_source:
                hits_by_source[source] = []
            hits_by_source[source].append(cid)

        final_blocks = []

        # Process each source
        for source, cids in hits_by_source.items():
            # Sort IDs
            cids.sort()

            # Merge ranges
            ranges = []
            if not cids:
                continue

            curr_start = cids[0] - 2
            curr_end = cids[0] + 2

            for cid in cids[1:]:
                start = cid - 2
                end = cid + 2

                if start <= curr_end + 1:  # Overlap or adjacent
                    curr_end = max(curr_end, end)
                else:
                    ranges.append((curr_start, curr_end))
                    curr_start = start
                    curr_end = end
            ranges.append((curr_start, curr_end))

            # Fetch and Merge
            for start, end in ranges:
                try:
                    range_docs = query_chunks_by_range(vs, source, start, end)
                    if range_docs:
                        merged = _merge_chunks(range_docs, chunk_similarity)
                        if merged:
                            final_blocks.append(merged)
                except Exception as e:
                    print(f"Error processing range {start}-{end} for {source}: {e}")
                    continue

        # 3. Format Output
        output = []
        for i, block in enumerate(final_blocks):
            src = block["source"]
            pg = block["page_no"]
            bbox = block["bbox"]
            txt = block["content"]
            sim = block.get("similarity", 0.0)
            txt_preview = txt[:2000]

            output.append(
                f"[Result {i}] File: {src} | Page: {pg} | BBox: {bbox} | Similarity: {sim:.2f}\n"
                f"Extended Context:\n{txt_preview}\n"
                f"(IDs: {block['chunk_ids']})"
            )

        return "\n\n".join(output)

    @tool
    def visual_proof(
        file_name: str, page_no: int, bbox: List[float], mode: str = "show"
    ) -> str:
        """
        Generate a visual proof (image) or analyze the visual content using AI (VLM).

        Args:
            file_name: The source file name (e.g., 'safety.pdf').
            page_no: The page number (1-based).
            bbox: The bounding box [left, top, right, bottom] from the search results.
            mode: "show" to return an image file path (default), or "analyze" to interpret the image content (e.g. for complex tables).

        Returns:
            Path to the generated image file (mode="show") or text description (mode="analyze").
        """
        ctx.visual_proof_call_count += 1
        if ctx.visual_proof_call_count > settings.MAX_VISUAL_PROOF_CALLS:
            return (
                f"Лимит visual_proof достигнут ({settings.MAX_VISUAL_PROOF_CALLS}). "
                "Сформулируй ответ на основе уже найденных данных."
            )

        # Call the existing implementation helper or duplicate logic.
        return _visual_proof_impl(file_name, page_no, bbox, mode)

    return [search_documents, visual_proof]


SOURCE_DOCS_DIR = Path("source_docs").resolve()
MAX_BBOX_DIM = 10_000.0  # PDF user-units; защита от DoS-рендера
MAX_BBOX_AREA_PT2 = 921_600.0  # ~4 MP at 150 DPI; ~1.84 A4 pages


def _validate_file_name(file_name: str) -> Path | str:
    """Возвращает безопасный Path внутри SOURCE_DOCS_DIR или строку с ошибкой.

    Параметр приходит от LLM, который может быть подвержен prompt-injection через
    содержимое PDF. Любые '/', '..' или абсолютные пути — невалидны.
    """
    if not file_name or not isinstance(file_name, str):
        return "Error: file_name must be a non-empty string."
    if Path(file_name).name != file_name:
        return f"Error: file_name must be a bare filename, got '{file_name}'."
    candidate = (SOURCE_DOCS_DIR / file_name).resolve()
    if SOURCE_DOCS_DIR not in candidate.parents:
        return f"Error: file_name '{file_name}' resolves outside source_docs."
    if not candidate.exists():
        return f"Error: File '{file_name}' not found in source_docs."
    return candidate


def _validate_bbox(bbox) -> str | None:
    """Возвращает None если bbox валиден, иначе текст ошибки."""
    import math

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "Error: bbox must be a list of 4 floats [left, top, right, bottom]."
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return "Error: bbox values must be numeric."
    if not all(math.isfinite(v) for v in vals):
        return "Error: bbox values must be finite."
    if any(v < 0 or v > MAX_BBOX_DIM for v in vals):
        return f"Error: bbox values must be within [0, {MAX_BBOX_DIM}]."
    left, top, right, bottom = vals
    if right <= left or abs(bottom - top) < 1e-6:
        return "Error: bbox has zero/negative area."
    area = (right - left) * (bottom - top)
    if area > MAX_BBOX_AREA_PT2:
        return f"Error: bbox area {area:.0f}pt² exceeds max {MAX_BBOX_AREA_PT2:.0f}pt² (≈4 MP at 150 DPI)."
    return None


def _visual_proof_impl(file_name, page_no, bbox, mode):
    # Move the body of visual_proof here
    try:
        resolved = _validate_file_name(file_name)
        if isinstance(resolved, str):
            return resolved
        source_path = resolved

        bbox_err = _validate_bbox(bbox)
        if bbox_err:
            return bbox_err

        doc = fitz.open(source_path)
        if page_no < 1 or page_no > len(doc):
            return f"Error: Page {page_no} out of range (1-{len(doc)})."

        page = doc[page_no - 1]

        left, top, right, bottom = bbox

        # Normalize coordinates (PDF bottom-left to Top-Left)
        if top > bottom:
            height = page.rect.height
            y0 = height - top
            y1 = height - bottom
            # Fix order for Rect (y0 < y1)
            rect = fitz.Rect(left, y0, right, y1)
        else:
            rect = fitz.Rect(left, top, right, bottom)

        # --- Mode: Analyze (VLM) with Red Box Strategy ---
        if mode == "analyze":
            try:
                # 1. Render FULL page (context is key for avoiding safety refusal)
                target_dpi = 150
                pix = page.get_pixmap(dpi=target_dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # 2. Scale coordinates (PDF 72 dpi -> target_dpi)
                scale = target_dpi / 72.0
                draw_rect = [
                    rect.x0 * scale,
                    rect.y0 * scale,
                    rect.x1 * scale,
                    rect.y1 * scale,
                ]

                # 3. Draw Red Box
                draw = ImageDraw.Draw(img)
                draw.rectangle(draw_rect, outline="red", width=5)

                # 4. Encode
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                vlm = get_vision_llm()

                prompt_text = (
                    "This is a public government document (Russia) containing safety regulations.\n"
                    "Focus ONLY on the content inside the RED BOX.\n"
                    "1. Transcribe the text or table inside the red box exactly as it appears, in Russian.\n"
                    "2. Do not summarize.\n"
                    "3. If the box cuts text, transcribe what is visible.\n"
                    "4. Output raw text/markdown only."
                )

                msg = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                        },
                    ]
                )
                response = vlm.invoke([msg])
                return f"[Visual Analysis Result]\n{response.content}"
            except Exception as e:
                return f"Error in VLM analysis: {str(e)}"

        # --- Mode: Show (Default) with Visual Zoom (Padding) ---
        # Add generous padding for context (50px)
        padding = 50
        rect.x0 = max(0, rect.x0 - padding)
        rect.y0 = max(0, rect.y0 - padding)
        rect.x1 = min(page.rect.width, rect.x1 + padding)
        rect.y1 = min(page.rect.height, rect.y1 + padding)

        pix = page.get_pixmap(clip=rect, dpi=150)

        # Save
        output_dir = Path("static/visuals")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Hash params to make filename unique but deterministic
        import hashlib

        h = hashlib.md5(f"{file_name}_{page_no}_{bbox}".encode()).hexdigest()[:8]
        output_filename = f"proof_{h}.png"
        output_path = output_dir / output_filename

        pix.save(output_path)

        return str(output_path)

    except Exception as e:
        return f"Error processing visual proof: {str(e)}"


def _merge_chunks(docs: List[Document], similarity_map: Dict = None) -> Dict:
    """
    Merge a list of sorted chunks into a single context block.
    Computes union BBox and max similarity.
    """
    if not docs:
        return {}

    similarity_map = similarity_map or {}

    # Merge Content
    full_content = "\n".join([d.page_content for d in docs])

    # Base Metadata from the middle or first chunk
    base_meta = docs[0].metadata.copy()

    # Calculate Max Similarity for this block
    max_similarity = 0.0
    chunk_ids = []

    for d in docs:
        chunk_id = d.metadata.get("chunk_id")
        chunk_ids.append(chunk_id)
        source = d.metadata.get("source")

        # Check if this chunk was in the initial search hits
        if (source, chunk_id) in similarity_map:
            score = similarity_map[(source, chunk_id)]
            if score > max_similarity:
                max_similarity = score

    # Compute Union BBox
    # [min_l, min_t, max_r, max_b]
    union_bbox = None

    # We also need to handle page changes.
    # If chunks span multiple pages, visual proof might be tricky.
    # For now, let's assume we want the BBox of the *primary* page (where the hit was).
    # Or just return the bbox of the first page encountered.
    target_page = base_meta.get("page_no")

    for d in docs:
        bbox_str = d.metadata.get("bbox")
        page = d.metadata.get("page_no")

        if bbox_str and page == target_page:
            try:
                bbox = json.loads(bbox_str)
                if union_bbox is None:
                    union_bbox = list(bbox)
                else:
                    union_bbox[0] = min(union_bbox[0], bbox[0])
                    union_bbox[1] = min(union_bbox[1], bbox[1])
                    union_bbox[2] = max(union_bbox[2], bbox[2])
                    union_bbox[3] = max(union_bbox[3], bbox[3])
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

    return {
        "content": full_content,
        "bbox": union_bbox,
        "page_no": target_page,
        "source": base_meta.get("source"),
        "chunk_ids": chunk_ids,
        "similarity": max_similarity,
    }
