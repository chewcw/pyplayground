#!/usr/bin/env python3
"""Shared helpers for the vector DB indexer examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from docling.document_converter import DocumentConverter

try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None


# Reduce noisy httpx/transformers logs when available
try:
    logging.getLogger("httpx").setLevel(logging.WARNING)
except Exception:
    pass
try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass


# chunk_text chunk the text by splitting text into fixed-size chunks with some overlap
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    if length == 0:
        return []
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if end == length:
            break
    return chunks

# Recursive splitting
# https://docs.trychroma.com/guides/build/chunking#chunking-strategies
def recursive_chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: Sequence[str] = ("\n\n", "\n", ". ", " ", ""),
) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        if sep:
            parts = text.split(sep)
            sep_len = len(sep)
        else:
            parts = list(text)
            sep_len = 0

        chunks = []
        current_chunk = ""
        for part in parts:
            if current_chunk and len(current_chunk) + len(part) + sep_len > chunk_size:
                chunks.append(current_chunk)
                current_chunk = ""
            current_chunk += part + (sep if sep else "")
        if current_chunk:
            chunks.append(current_chunk)

        if all(len(c) <= chunk_size for c in chunks):
            # Add overlap
            final_chunks = []
            for i, chunk in enumerate(chunks):
                final_chunks.append(chunk)
                if i < len(chunks) - 1 and overlap > 0:
                    final_chunks[-1] += chunk[-overlap:]
            return final_chunks

    # If we couldn't split properly, just truncate with overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


CHUNK_STRATEGIES = ("overlap", "recursive")
DEFAULT_CHUNK_STRATEGY = CHUNK_STRATEGIES[0]


def _select_chunker(chunk_strategy: str) -> Callable[[str, int, int], List[str]]:
    if chunk_strategy == "overlap":
        return chunk_text
    if chunk_strategy == "recursive":
        return recursive_chunk_text
    raise ValueError(
        f"Unknown chunk strategy: {chunk_strategy}. Expected one of: {', '.join(CHUNK_STRATEGIES)}"
    )

def _sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _coerce_metadata_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    try:
        parsed = json.loads(value)
    except Exception:
        return value
    return _sanitize_metadata_value(parsed)


def _parse_metadata_overrides(metadata_args: Optional[List[str]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not metadata_args:
        return overrides

    for raw_entry in metadata_args:
        if raw_entry is None:
            continue

        entry = raw_entry.strip()
        if not entry:
            continue

        if entry.startswith("{"):
            try:
                parsed = json.loads(entry)
            except Exception as exc:
                raise ValueError(f"Invalid JSON metadata value: {raw_entry}") from exc

            if not isinstance(parsed, dict):
                raise ValueError("--metadata JSON input must decode to an object")

            for key, value in parsed.items():
                overrides[str(key)] = _sanitize_metadata_value(value)
            continue

        if "=" not in entry:
            raise ValueError(
                f"Metadata entries must use key=value or JSON object syntax: {raw_entry}"
            )

        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Metadata key cannot be empty: {raw_entry}")

        overrides[key] = _coerce_metadata_value(raw_value)

    return overrides


def _build_document_record(
    path: Path,
    *,
    doc_id_suffix: str,
    doc_text: str,
    chunk_index: int,
    mime_type: str,
    page: Optional[int] = None,
    converted_with: Optional[str] = None,
    embed_item: Any = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    metadata: Dict[str, Any] = {
        "source": str(path),
        "filename": path.name,
        "chunk": chunk_index,
        "mime_type": mime_type,
        "text_preview": (doc_text or "")[:200],
    }
    if page is not None:
        metadata["page"] = page
    if converted_with is not None:
        metadata["converted_with"] = converted_with
    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata.setdefault(key, _sanitize_metadata_value(value))

    return f"{path.stem}::{doc_id_suffix}", doc_text, metadata, embed_item


def _to_text_inputs(embed_items_batch: List) -> List[str]:
    text_inputs: List[str] = []
    try:
        from PIL import Image as _PILImage
        import pytesseract as _pytesseract

        _has_tesseract = True
    except Exception:
        _has_tesseract = False

    from pathlib import Path as _Path

    for itm in embed_items_batch:
        if isinstance(itm, str):
            text_inputs.append(itm)
        elif isinstance(itm, dict) and itm.get("text"):
            text_inputs.append(itm.get("text"))
        else:
            try:
                if isinstance(itm, (_Path, str)):
                    text_inputs.append(extract_text_from_image(_Path(itm)))
                else:
                    if _has_tesseract:
                        try:
                            text_inputs.append(_pytesseract.image_to_string(itm))
                        except Exception:
                            text_inputs.append("")
                    else:
                        text_inputs.append("")
            except Exception:
                text_inputs.append("")
    return text_inputs


def _docling_result_to_text(result) -> str:
    if hasattr(result, "text_content"):
        return result.text_content or ""
    if hasattr(result, "text"):
        return result.text or ""
    doc = getattr(result, "document", None)
    if doc is not None:
        if hasattr(doc, "export_to_markdown"):
            return doc.export_to_markdown() or ""
        if hasattr(doc, "export_to_text"):
            return doc.export_to_text() or ""
    return ""


def _docling_convert_file_to_text(converter: DocumentConverter, path: Path) -> str:
    try:
        result = converter.convert(str(path))
        return _docling_result_to_text(result)
    except Exception:
        return ""


def _is_multimodal_eligible(suffix: str) -> bool:
    return suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


def _embed_items(
    embed_client,
    embed_items_batch: List,
    use_multimodal: bool,
) -> List[List[float]]:
    try:
        if use_multimodal and embed_client.provider == "sentence-transformers":
            return embed_client.embed_multimodal(embed_items_batch)
        return embed_client.embed_texts(_to_text_inputs(embed_items_batch))
    except Exception as e:
        logging.warning(
            "Embedding failed for batch: %s; falling back to text-only embedding",
            e,
        )
        return embed_client.embed_texts(_to_text_inputs(embed_items_batch))


def _handle_pdf_multimodal(path: Path, extra_metadata: Optional[Dict[str, Any]] = None):
    """Return list of (doc_id, doc_text, meta, embed_item) for PDF pages when using multimodal flow.

    If PDF->images conversion fails or yields no pages, returns an empty list.
    """
    try:
        from pdf2image import convert_from_path

        pages_imgs = convert_from_path(str(path), dpi=300)
    except Exception:
        pages_imgs = None

    out = []
    if not pages_imgs:
        return out

    for page_idx, img in enumerate(pages_imgs, start=1):
        preview = ""
        try:
            import pytesseract

            preview = pytesseract.image_to_string(img)
        except Exception:
            preview = ""

        doc_id, doc_text, meta, embed_item = _build_document_record(
            path,
            doc_id_suffix=f"p{page_idx}::img",
            doc_text=preview or f"[image page {page_idx}]",
            chunk_index=1,
            mime_type="application/pdf",
            page=page_idx,
            converted_with="multimodal",
            embed_item={"image": img, "text": preview},
            extra_metadata=extra_metadata,
        )
        out.append((doc_id, doc_text, meta, embed_item))

    return out


def _handle_image_multimodal(
    path: Path, extra_metadata: Optional[Dict[str, Any]] = None
):
    """Return list of (doc_id, doc_text, meta, embed_item) for a single image file.

    If the image cannot be opened, returns an empty list.
    """
    try:
        from PIL import Image as _PILImage

        img_obj = _PILImage.open(path).convert("RGB")
    except Exception:
        img_obj = None

    if img_obj is None:
        return []

    preview = ""
    try:
        import pytesseract

        preview = pytesseract.image_to_string(img_obj)
    except Exception:
        preview = ""

    doc_id, doc_text, meta, embed_item = _build_document_record(
        path,
        doc_id_suffix="img::c1",
        doc_text=preview or f"[image:{path.name}]",
        chunk_index=1,
        mime_type="image/*",
        converted_with="multimodal",
        embed_item={"image": img_obj, "text": preview},
        extra_metadata=extra_metadata,
    )
    return [(doc_id, doc_text, meta, embed_item)]


def extract_text_from_pdf(path: Path, use_ocr: bool = True) -> List[str]:
    """Return list of page texts. Falls back to OCR (pdf->images->tesseract) if no text found."""
    texts: List[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                texts.append(txt)
    except Exception:
        texts = []

    if use_ocr and (not any(t.strip() for t in texts)):
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(str(path), dpi=300)
            texts = [pytesseract.image_to_string(img) for img in images]
        except Exception:
            pass

    if len(texts) == 0:
        return [""]
    return texts


def extract_text_from_image(path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


class EmbeddingClient:
    """Simple wrapper to support different embedding providers."""

    def __init__(self, provider: str = "sentence-transformers", model_name: str = None):
        self.provider = provider
        self.model_name = model_name
        if provider == "openai":
            try:
                import openai

                self._client = openai
            except Exception as e:
                raise ImportError("openai package required for provider=openai") from e
            self.model = model_name or "text-embedding-3-small"
        elif provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")
                self._fallback = False
            except Exception:
                logging.warning(
                    "sentence-transformers not available; using deterministic fallback embeddings"
                )
                self._model = None
                self._fallback = True
                self._dim = 384
        elif provider == "ollama":
            if OllamaEmbeddings is None:
                raise ImportError(
                    "langchain-ollama package required for provider=ollama"
                )
            self._model = OllamaEmbeddings(model=model_name or "embeddinggemma:latest")
        else:
            raise ValueError("Unsupported provider: %s" % provider)

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if self.provider == "openai":
            out = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = self._client.Embedding.create(model=self.model, input=batch)
                out.extend([item["embedding"] for item in resp["data"]])
            return out
        elif self.provider == "ollama":
            out = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                out.extend(self._model.embed_documents(batch))
            return out
        else:
            if getattr(self, "_fallback", False):
                out = []
                for t in texts:
                    h = hashlib.sha256(t.encode("utf-8")).digest()
                    vec = []
                    for i in range(self._dim):
                        b = h[i % len(h)]
                        vec.append(b / 255.0)
                    out.append(vec)
                return out
            embs = self._model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            return embs.tolist()

    def embed_multimodal(self, items: List, batch_size: int = 8) -> List[List[float]]:
        """Embed a list of multimodal inputs using a sentence-transformers model.

        items may be a mix of:
          - text strings
          - Path/str pointing to image files
          - PIL.Image.Image objects
          - dicts like {"image": <Path|PIL.Image>, "text": <str>}

        This function will attempt to call the underlying SentenceTransformer
        encode API with the mixed inputs and fall back to OCR->text
        embeddings if multimodal encoding is not available or fails.
        """
        if self.provider != "sentence-transformers":
            raise ValueError(
                "Multimodal embeddings require provider=sentence-transformers"
            )

        if getattr(self, "_fallback", False):
            logging.warning(
                "Multimodal requested but sentence-transformers unavailable; falling back to OCR+text embeddings"
            )
            texts = []
            for it in items:
                if isinstance(it, dict) and it.get("text"):
                    texts.append(it.get("text"))
                else:
                    try:
                        if isinstance(it, (str, Path)):
                            texts.append(extract_text_from_image(Path(it)))
                        else:
                            try:
                                import pytesseract

                                texts.append(pytesseract.image_to_string(it))
                            except Exception:
                                texts.append("")
                    except Exception:
                        texts.append("")
            return self.embed_texts(texts, batch_size=batch_size)

        inputs = []
        try:
            from PIL import Image
            import numpy as _np
        except Exception:
            Image = None
            _np = None

        for it in items:
            if isinstance(it, str):
                inputs.append(it)
                continue

            if isinstance(it, dict):
                img = it.get("image")
                txt = it.get("text")
                if isinstance(img, (str, Path)) and Image is not None:
                    try:
                        img_obj = Image.open(str(img)).convert("RGB")
                        if _np is not None:
                            img_arr = _np.array(img_obj)
                            inputs.append(
                                {"image": img_arr, "text": txt} if txt else img_arr
                            )
                        else:
                            inputs.append(img_obj)
                    except Exception:
                        inputs.append(txt or "")
                else:
                    if Image is not None and isinstance(img, Image.Image):
                        try:
                            img_arr = _np.array(img) if _np is not None else img
                            inputs.append(
                                {"image": img_arr, "text": txt} if txt else img_arr
                            )
                        except Exception:
                            inputs.append(txt or "")
                    else:
                        inputs.append(txt or "")
                continue

            if isinstance(it, (str, Path)):
                if Image is not None:
                    try:
                        img_obj = Image.open(str(it)).convert("RGB")
                        if _np is not None:
                            inputs.append(_np.array(img_obj))
                        else:
                            inputs.append(img_obj)
                        continue
                    except Exception:
                        inputs.append(str(it))
                        continue

            try:
                import PIL

                if isinstance(it, PIL.Image.Image):
                    if _np is not None:
                        inputs.append(_np.array(it))
                    else:
                        inputs.append(it)
                    continue
            except Exception:
                pass

            inputs.append(str(it))

        try:
            embs = self._model.encode(
                inputs,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=batch_size,
            )
            return embs.tolist()
        except Exception as e:
            logging.warning(
                "Multimodal model encoding failed (%s); falling back to OCR->text embeddings",
                e,
            )
            texts = []
            try:
                import pytesseract
            except Exception:
                pytesseract = None

            for it in items:
                if isinstance(it, dict) and it.get("text"):
                    texts.append(it.get("text"))
                    continue
                try:
                    if isinstance(it, (str, Path)):
                        texts.append(extract_text_from_image(Path(it)))
                    else:
                        if pytesseract is not None:
                            try:
                                texts.append(pytesseract.image_to_string(it))
                            except Exception:
                                texts.append("")
                        else:
                            texts.append("")
                except Exception:
                    texts.append("")

            return self.embed_texts(texts, batch_size=batch_size)


def embed_with_qwen3_vl_hf(
    input_dir: Union[Path, List[Path]],
    texts: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 4,
    pdf_dpi: int = 150,
    max_pages: Optional[int] = None,
):
    """Embed files with the public Qwen3-VL model loaded locally via SentenceTransformer.

    The Qwen3-VL embedding model is documented to work through the local
    SentenceTransformer interface for text, image, and mixed-modal inputs.
    """
    return embed_with_qwen3_vl_st(
        paths=input_dir,
        texts=texts,
        model=model,
        batch_size=batch_size,
        pdf_dpi=pdf_dpi,
        max_pages=max_pages,
    )


def embed_with_qwen3_vl_st(
    paths: Union[Path, List[Path]],
    texts: Optional[List[str]] = None,
    model: str = "Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 4,
    pdf_dpi: int = 150,
    max_pages: Optional[int] = None,
):
    """Embed a list of files (PDFs or images) using a local
    `sentence-transformers` `SentenceTransformer` model.

    - PDFs are converted to images (one image per page) using `pdf2image`.
    - Each image is prepared as a dict: {"image": numpy_array, "text": str}
      and passed to `SentenceTransformer.encode`.

    NOTE: This function requires `sentence-transformers` and `pillow` to be
    installed locally.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from PIL import Image
        import numpy as _np
    except Exception as e:
        raise ImportError(
            "sentence-transformers and pillow are required for local Qwen3-VL embedding."
            " Install with: pip install 'sentence-transformers' pillow pdf2image"
        ) from e

    try:
        st = SentenceTransformer(model)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SentenceTransformer model '{model}': {e}"
        ) from e

    inputs = []

    try:
        from pdf2image import convert_from_path
    except Exception:
        convert_from_path = None

    if isinstance(paths, (str, Path)):
        input_path = Path(paths).expanduser()
        try:
            input_path = input_path.resolve(strict=False)
        except Exception:
            input_path = Path(input_path)

        if input_path.is_dir():
            files_to_process = [p for p in sorted(input_path.rglob("*")) if p.is_file()]
        elif input_path.is_file():
            files_to_process = [input_path]
        else:
            raise ValueError(
                "paths must be an existing directory, a file path, or an iterable of file paths"
            )
    else:
        files_to_process = list(paths)

    for idx, p in enumerate(files_to_process):
        p = Path(p)
        suffix = p.suffix.lower()

        if suffix == ".pdf":
            if convert_from_path is None:
                raise ImportError(
                    "pdf2image is required to convert PDFs to images (pip install pdf2image)"
                )
            try:
                pages = convert_from_path(str(p), dpi=pdf_dpi)
            except Exception:
                pages = []

            for pi, img in enumerate(pages):
                if max_pages is not None and pi >= max_pages:
                    break
                try:
                    arr = _np.array(img.convert("RGB"))
                    txt = texts[idx] if texts and idx < len(texts) else ""
                    inputs.append({"image": arr, "text": txt})
                except Exception:
                    inputs.append(img.convert("RGB"))

        elif suffix in (".png", ".jpg", ".jpeg", ".tiff"):
            try:
                img = Image.open(p).convert("RGB")
                arr = _np.array(img)
                txt = texts[idx] if texts and idx < len(texts) else ""
                inputs.append({"image": arr, "text": txt})
            except Exception:
                continue

        else:
            continue

    if not inputs:
        return []

    try:
        embs = st.encode(
            inputs,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
    except Exception as e:
        raise RuntimeError(f"SentenceTransformer.encode failed: {e}") from e

    return [list(e) for e in embs]


def embed_query(embed_client: EmbeddingClient, query: str) -> List[float]:
    """Return a single embedding vector for the query string."""
    embs = embed_client.embed_texts([query])
    if not embs:
        return []
    return embs[0]


def retrieve_documents(
    collection,
    embedding: List[float],
    top_k: int = 10,
    filters: Dict = None,
    include: List[str] = None,
):
    """Query the collection using a precomputed embedding.

    Returns a list of result dicts with keys: id, source, text, metadata, score
    """
    include = include or [
        "documents",
        "metadatas",
        "distances",
        "embeddings",
        "uris",
        "data",
    ]

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters,
            include=include,
        )
    except TypeError:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters,
            include=include,
        )

    hits = []
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if "ids" in results:
        ids = results.get("ids", [[]])[0]
    elif "data" in results:
        ids = results.get("data", [[]])[0]
    elif "uris" in results:
        ids = results.get("uris", [[]])[0]
    else:
        ids = [None] * len(docs)

    distances = results.get("distances", [[]])[0]

    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        raw_id = ids[i] if i < len(ids) else None
        if isinstance(raw_id, dict):
            doc_id = raw_id.get("id") or raw_id.get("uri") or str(raw_id)
        else:
            doc_id = raw_id
        distance = distances[i] if i < len(distances) else None
        hits.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "score": float(distance) if distance is not None else None,
            }
        )
    return hits


def rerank_results(results: List[Dict], reranker=None) -> List[Dict]:
    """Stub for a reranker. By default returns results unchanged."""
    return results


def _resolve_collection(
    chroma_dir: Path,
    collection_name: str,
    collection: Any,
    get_collection_fn: Optional[Callable[..., tuple]],
):
    if collection is not None:
        return None, collection
    if get_collection_fn is None:
        raise ValueError(
            "get_collection_fn is required when collection is not provided"
        )
    return get_collection_fn(chroma_dir, collection_name=collection_name)


def list_ids(
    chroma_dir: Path,
    collection_name: str = "receipts",
    collection: Any = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
) -> List[str]:
    """Return a list of ids stored in the collection."""
    if collection is None:
        _, collection = _resolve_collection(
            chroma_dir, collection_name, collection, get_collection_fn
        )
        results = collection.get()
        ids = results.get("ids") or results.get("data") or results.get("uris")
        return ids

    results = collection.get()
    ids = results.get("ids") or results.get("data") or results.get("uris")
    return ids


def metadata_by_id(
    chroma_dir: Path,
    collection_name: str = "receipts",
    doc_id: Optional[str] = None,
    collection: Any = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
) -> Optional[Dict]:
    """Return the metadata for a given document id."""
    if collection is None:
        _, collection = _resolve_collection(
            chroma_dir, collection_name, collection, get_collection_fn
        )
    try:
        results = collection.get(ids=[doc_id], include=["metadatas"])
        metadatas = results.get("metadatas", [])
        if metadatas and len(metadatas) > 0:
            return metadatas[0]
        return None
    except Exception:
        return None


def _unwrap_list_field(field: Any) -> Any:
    """Unwrap nested list-of-list responses used by some collection results."""
    if isinstance(field, list) and len(field) == 1 and isinstance(field[0], list):
        return field[0]
    return field


def _normalize_id(raw_id: Any) -> Any:
    if isinstance(raw_id, dict):
        return raw_id.get("id") or raw_id.get("uri") or str(raw_id)
    return raw_id


def get_vectors_by_ids(
    chroma_dir: Optional[Path] = None,
    collection_name: str = "receipts",
    ids: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    collection: Any = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
) -> List[Dict[str, Any]]:
    """Return stored documents, metadata and embeddings for the requested ids."""
    if collection is None:
        if chroma_dir is None:
            raise ValueError("Either `chroma_dir` or `collection` must be provided")
        _, collection = _resolve_collection(
            chroma_dir, collection_name, collection, get_collection_fn
        )

    include = include or [
        "documents",
        "metadatas",
        "embeddings",
        "distances",
        "ids",
        "uris",
        "data",
    ]

    try:
        results = collection.get(ids=ids, include=include)
    except TypeError:
        results = collection.get(ids=ids, include=include)
    except Exception as e:
        raise RuntimeError(f"collection.get failed: {e}") from e

    docs = _unwrap_list_field(results.get("documents", []))
    metadatas = _unwrap_list_field(results.get("metadatas", []))
    embeddings = _unwrap_list_field(results.get("embeddings", []))
    ids_res = (
        _unwrap_list_field(results.get("ids", []))
        or _unwrap_list_field(results.get("data", []))
        or _unwrap_list_field(results.get("uris", []))
    )
    distances = _unwrap_list_field(results.get("distances", []))
    uris = _unwrap_list_field(results.get("uris", []))
    data = _unwrap_list_field(results.get("data", []))

    max_len = max(
        len(ids_res) if ids_res else 0,
        len(docs or []),
        len(embeddings or []),
        len(metadatas or []),
    )
    out: List[Dict[str, Any]] = []
    for i in range(max_len):
        raw_id = ids_res[i] if ids_res and i < len(ids_res) else None
        doc_id = _normalize_id(raw_id)
        out.append(
            {
                "id": doc_id,
                "document": docs[i] if i < len(docs or []) else None,
                "metadata": metadatas[i] if i < len(metadatas or []) else None,
                "embedding": embeddings[i] if i < len(embeddings or []) else None,
                "distance": float(distances[i]) if i < len(distances or []) else None,
                "uri": uris[i] if i < len(uris or []) else None,
                "data": data[i] if i < len(data or []) else None,
            }
        )
    return out


def dump_collection_sample(
    chroma_dir: Path,
    collection_name: str = "receipts",
    limit: int = 10,
    include: Optional[List[str]] = None,
    collection: Any = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
) -> List[Dict[str, Any]]:
    """Return a sample of documents from the collection."""
    if collection is None:
        _, collection = _resolve_collection(
            chroma_dir, collection_name, collection, get_collection_fn
        )

    include = include or [
        "documents",
        "metadatas",
        "embeddings",
        "distances",
        "ids",
        "uris",
        "data",
    ]

    try:
        results = collection.query(query_texts=[""], n_results=limit, include=include)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        embeddings = (
            results.get("embeddings", [[]])[0] if "embeddings" in results else []
        )
        ids_res = (
            results.get("ids", [[]])[0]
            if "ids" in results
            else results.get("data", [[]])[0] if "data" in results else []
        )
        distances = results.get("distances", [[]])[0] if "distances" in results else []
        out: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            raw_id = ids_res[i] if i < len(ids_res) else None
            doc_id = _normalize_id(raw_id)
            out.append(
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else None,
                    "embedding": embeddings[i] if i < len(embeddings) else None,
                    "distance": float(distances[i]) if i < len(distances) else None,
                }
            )
        return out
    except Exception:
        try:
            results = collection.get(include=include)
            ids_res = (
                _unwrap_list_field(results.get("ids", []))
                or _unwrap_list_field(results.get("data", []))
                or []
            )
            if not ids_res:
                docs = _unwrap_list_field(results.get("documents", []))
                metadatas = _unwrap_list_field(results.get("metadatas", []))
                embeddings = _unwrap_list_field(results.get("embeddings", []))
                out = []
                for i, doc in enumerate(docs[:limit]):
                    out.append(
                        {
                            "id": None,
                            "document": doc,
                            "metadata": metadatas[i] if i < len(metadatas) else None,
                            "embedding": embeddings[i] if i < len(embeddings) else None,
                            "distance": None,
                        }
                    )
                return out
            return get_vectors_by_ids(collection=collection, ids=ids_res[:limit])
        except Exception:
            return []


def ingest_directory(
    input_dir: Path,
    chroma_dir: Path,
    embed_client: EmbeddingClient,
    collection_name: str = "receipts",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    use_multimodal: bool = False,
    record_metadata: Optional[Dict[str, Any]] = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
):
    input_dir = Path(input_dir).expanduser()
    try:
        input_dir = input_dir.resolve(strict=False)
    except Exception:
        input_dir = Path(input_dir)

    logging.debug(
        "Normalized input_dir=%s (exists=%s, is_file=%s, is_dir=%s)",
        input_dir,
        input_dir.exists(),
        input_dir.is_file(),
        input_dir.is_dir(),
    )

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("--input_dir must be an existing directory (got: %s)", input_dir)
        raise ValueError(f"--input_dir must be an existing directory: {input_dir}")

    chunker = _select_chunker(chunk_strategy)

    skip_docling_for_multimodal = (
        use_multimodal and embed_client.provider == "sentence-transformers"
    )

    client, collection = _resolve_collection(
        chroma_dir, collection_name, None, get_collection_fn
    )

    docs_batch: List[str] = []
    metas_batch: List[Dict] = []
    ids_batch: List[str] = []
    embed_items_batch: List = []

    def _flush_batch():
        if not docs_batch:
            return
        logging.info("Embedding batch of %d chunks", len(docs_batch))
        embs = _embed_items(embed_client, embed_items_batch, use_multimodal)

        if collection is not None:
            collection.add(
                documents=docs_batch,
                metadatas=metas_batch,
                embeddings=embs,
                ids=ids_batch,
            )
        else:
            logging.info(
                "Dry-run: would add %d documents to collection", len(docs_batch)
            )

        docs_batch.clear()
        metas_batch.clear()
        ids_batch.clear()
        embed_items_batch.clear()

    _docling_converter = DocumentConverter()
    logging.info("Docling converter initialized: %s", bool(_docling_converter))

    files_iter = [p for p in sorted(input_dir.rglob("*")) if p.is_file()]
    logging.info(
        "Found %d files to process in input_dir: %s", len(files_iter), input_dir
    )

    for p in files_iter:
        logging.info("Processing file: %s", p)
        suffix = p.suffix.lower()
        docling_text = ""
        if _docling_converter is not None and not (
            skip_docling_for_multimodal and _is_multimodal_eligible(suffix)
        ):
            docling_text = _docling_convert_file_to_text(_docling_converter, p)
        elif skip_docling_for_multimodal and _is_multimodal_eligible(suffix):
            logging.debug(
                "Skipping docling conversion for multimodal-eligible file: %s",
                p,
            )

        logging.info(
            "Docling conversion for %s: %d chars",
            p.name,
            len(docling_text) if docling_text else 0,
        )

        if docling_text and docling_text.strip():
            logging.info("Docling converted %s (%d chars)", p.name, len(docling_text))
            chunks = chunker(docling_text, chunk_size=chunk_size, overlap=chunk_overlap)
            for ci, chunk in enumerate(chunks, start=1):
                doc_id, doc_text, meta, embed_item = _build_document_record(
                    p,
                    doc_id_suffix=f"docling::c{ci}",
                    doc_text=chunk,
                    chunk_index=ci,
                    mime_type=suffix or "",
                    converted_with="docling",
                    embed_item=chunk,
                    extra_metadata=record_metadata,
                )
                docs_batch.append(doc_text)
                metas_batch.append(meta)
                ids_batch.append(doc_id)
                embed_items_batch.append(embed_item)
            if len(docs_batch) >= 256:
                _flush_batch()
            continue

        if suffix == ".pdf":
            if use_multimodal and embed_client.provider == "sentence-transformers":
                items = _handle_pdf_multimodal(p, extra_metadata=record_metadata)
                if items:
                    for doc_id, doc_text, meta, embed_item in items:
                        docs_batch.append(doc_text)
                        metas_batch.append(meta)
                        ids_batch.append(doc_id)
                        embed_items_batch.append(embed_item)
                    if len(docs_batch) >= 256:
                        _flush_batch()
                    continue

            pages = extract_text_from_pdf(p)
            for page_idx, page_text in enumerate(pages, start=1):
                if not page_text or not page_text.strip():
                    continue
                chunks = chunker(page_text, chunk_size=chunk_size, overlap=chunk_overlap)
                for ci, chunk in enumerate(chunks, start=1):
                    doc_id, doc_text, meta, embed_item = _build_document_record(
                        p,
                        doc_id_suffix=f"p{page_idx}::c{ci}",
                        doc_text=chunk,
                        chunk_index=ci,
                        mime_type="application/pdf",
                        page=page_idx,
                        embed_item=chunk,
                        extra_metadata=record_metadata,
                    )
                    docs_batch.append(doc_text)
                    metas_batch.append(meta)
                    ids_batch.append(doc_id)
                    embed_items_batch.append(embed_item)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff"):
            if use_multimodal and embed_client.provider == "sentence-transformers":
                items = _handle_image_multimodal(p, extra_metadata=record_metadata)
                if items:
                    for doc_id, doc_text, meta, embed_item in items:
                        docs_batch.append(doc_text)
                        metas_batch.append(meta)
                        ids_batch.append(doc_id)
                        embed_items_batch.append(embed_item)
                    if len(docs_batch) >= 256:
                        _flush_batch()
                    continue

            text = extract_text_from_image(p)
            if not text or not text.strip():
                continue
            chunks = chunker(text, chunk_size=chunk_size, overlap=chunk_overlap)
            for ci, chunk in enumerate(chunks, start=1):
                doc_id, doc_text, meta, embed_item = _build_document_record(
                    p,
                    doc_id_suffix=f"img::c{ci}",
                    doc_text=chunk,
                    chunk_index=ci,
                    mime_type="image/*",
                    embed_item=chunk,
                    extra_metadata=record_metadata,
                )
                docs_batch.append(doc_text)
                metas_batch.append(meta)
                ids_batch.append(doc_id)
                embed_items_batch.append(embed_item)
        else:
            continue

        if len(docs_batch) >= 256:
            _flush_batch()

    _flush_batch()

    try:
        if client is not None:
            client.persist()
    except Exception:
        pass

    logging.info("Ingestion complete. Collection '%s' updated.", collection_name)


def retrieve(
    query: str,
    embed_client: EmbeddingClient,
    chroma_dir: Path,
    collection_name: str = "receipts",
    top_k: int = 10,
    filters: Dict = None,
    rerank: bool = False,
    include: List[str] = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
):
    """High-level retrieval orchestrator."""
    start = time.time()
    _, collection = _resolve_collection(
        chroma_dir, collection_name, None, get_collection_fn
    )
    logging.info("Performing retrieval for query: '%s'", query)
    emb = embed_query(embed_client, query)
    logging.info("Query embedding computed (length=%d)", len(emb))
    if not emb:
        return {
            "query": query,
            "top_k": top_k,
            "retrieval_time_ms": int((time.time() - start) * 1000),
            "results": [],
        }

    hits = retrieve_documents(
        collection, emb, top_k=top_k, filters=filters, include=include
    )
    if rerank:
        hits = rerank_results(hits)

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "query": query,
        "top_k": top_k,
        "retrieval_time_ms": elapsed_ms,
        "results": hits,
    }


def search_tool(
    query: str,
    chroma_dir: Path,
    embed_client: EmbeddingClient,
    collection_name: str = "receipts",
    top_k: int = 5,
    filters: Dict = None,
    include: List[str] = None,
    get_collection_fn: Optional[Callable[..., tuple]] = None,
):
    """Agent-facing helper: perform a retrieval and return top_k results."""
    return retrieve(
        query=query,
        embed_client=embed_client,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        top_k=top_k,
        filters=filters,
        include=include,
        get_collection_fn=get_collection_fn,
    )


def run_retrieve_command(args, *, collection_target, get_collection_fn) -> None:
    embed_client = EmbeddingClient(
        provider=args.embed_provider, model_name=args.embed_model
    )
    results = retrieve(
        query=args.retrieve_query,
        embed_client=embed_client,
        chroma_dir=collection_target,
        collection_name=args.collection_name,
        top_k=5,
        filters=None,
        rerank=True,
        get_collection_fn=get_collection_fn,
    )
    print(results)


def run_embed_command(args) -> None:
    if args.query_string is None:
        raise SystemExit("--query_string is required for the embed command")

    embed_client = EmbeddingClient(
        provider=args.embed_provider, model_name=args.embed_model
    )
    vector = embed_query(embed_client, args.query_string)
    print(json.dumps({"query_string": args.query_string, "vector": vector}, indent=2))


def run_ingest_command(args, *, collection_target, get_collection_fn) -> None:
    input_path = Path(args.input_dir).expanduser()
    try:
        input_path = input_path.resolve(strict=False)
    except Exception:
        input_path = Path(input_path)

    if not input_path.exists() or not input_path.is_dir():
        raise SystemExit(f"--input_dir must be an existing directory: {input_path}")

    try:
        metadata_overrides = _parse_metadata_overrides(args.metadata)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    embed_client = EmbeddingClient(
        provider=args.embed_provider, model_name=args.embed_model
    )
    ingest_directory(
        input_dir=input_path,
        chroma_dir=collection_target,
        embed_client=embed_client,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_strategy=getattr(args, "chunk_strategy", DEFAULT_CHUNK_STRATEGY),
        use_multimodal=args.use_multimodal,
        record_metadata=metadata_overrides,
        get_collection_fn=get_collection_fn,
    )


def run_inspect_command(args, *, collection_target, get_collection_fn) -> None:
    action = getattr(args, "inspect_action", "list_ids") or "list_ids"
    if action == "list_ids":
        ids_list = list_ids(
            collection_target,
            collection_name=args.collection_name,
            get_collection_fn=get_collection_fn,
        )
        print(json.dumps({"ids": ids_list}, indent=2))
        return

    if action == "metadata":
        metadata = None
        if args.doc_id:
            metadata = metadata_by_id(
                chroma_dir=collection_target,
                collection_name=args.collection_name,
                doc_id=args.doc_id,
                get_collection_fn=get_collection_fn,
            )
        print(
            json.dumps({"action": action, "results": metadata}, indent=2, default=str)
        )
        return

    raise SystemExit(f"Unknown inspect action: {action}")


def run_qwen_hf_command(args, *, collection_target, get_collection_fn) -> None:
    import sys

    input_path = Path(args.input_dir).expanduser()
    try:
        input_path = input_path.resolve(strict=False)
    except Exception:
        input_path = Path(input_path)

    if not input_path.exists() or not input_path.is_dir():
        logging.error(
            "--input_dir must be an existing directory. Got: %s", args.input_dir
        )
        sys.exit(1)

    logging.info(
        "Embedding files in %s using Qwen3-VL-HF model '%s'",
        input_path,
        args.embed_model or "Qwen/Qwen3-VL-Embedding-2B",
    )

    embeddings = embed_with_qwen3_vl_hf(
        input_dir=input_path,
        model=args.embed_model or "Qwen/Qwen3-VL-Embedding-2B",
        batch_size=4,
        pdf_dpi=150,
        max_pages=5,
    )
    client, collection = get_collection_fn(
        collection_target, collection_name=args.collection_name
    )
    for idx, emb in enumerate(embeddings):
        doc_id = f"qwen-hf::doc{idx+1}"
        collection.add(
            documents=[f"Document {idx+1}"],
            metadatas=[{"source": "qwen-hf", "index": idx + 1}],
            embeddings=[emb],
            ids=[doc_id],
        )
    try:
        client.persist()
    except Exception:
        pass
