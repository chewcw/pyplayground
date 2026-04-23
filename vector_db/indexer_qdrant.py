#!/usr/bin/env python3
"""Indexer example: OCR -> extract -> chunk -> embed -> Qdrant

Usage example:
    python indexer.py --input_dir ./sample_docs --qdrant_url http://localhost:6333 \
        --embed_provider sentence-transformers --embed_model all-MiniLM-L6-v2

Ollama example (local):
    python indexer.py --input_dir ./sample_docs --qdrant_url http://localhost:6333 \
        --embed_provider ollama --embed_model nomic-embed-text

Multimodal example (opt-in):
    python indexer.py ingest --input_dir ./sample_docs --qdrant_url http://localhost:6333 \
        --embed_provider sentence-transformers --embed_model <multimodal-model> --use_multimodal

Requirements (examples):
    pip install qdrant-client sentence-transformers pdfplumber pytesseract pdf2image pillow openai
System deps for OCR/PDF to image:
    - tesseract (for pytesseract)
    - poppler (for pdf2image.convert_from_path)

This script is an example only — adapt models, batching, and error handling to your needs.
"""
import argparse
import json
import logging
import uuid
from functools import partial
from pathlib import Path
from typing import Any, List, Optional

import indexer_core as core
from qdrant_client import QdrantClient, models as qmodels


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


QDRANT_DEFAULT_URL = "http://localhost:6333"


def _normalize_point_vector(vector: Any) -> List[float]:
    if vector is None:
        return []
    if isinstance(vector, dict):
        if "" in vector:
            vector = vector[""]
        elif len(vector) == 1:
            vector = next(iter(vector.values()))
        else:
            return []

    try:
        return [float(value) for value in list(vector)]
    except Exception:
        return []


def _qdrant_point_id(document_id: Any):
    if isinstance(document_id, int) and document_id >= 0:
        return document_id

    text_id = str(document_id)
    try:
        return str(uuid.UUID(text_id))
    except Exception:
        pass

    if text_id.isdigit():
        try:
            return int(text_id)
        except Exception:
            pass

    return str(uuid.uuid5(uuid.NAMESPACE_URL, text_id))


def _build_qdrant_filter(where: Any):
    if not where:
        return None
    if isinstance(where, qmodels.Filter):
        return where

    must = []
    for key, value in where.items():
        if isinstance(value, dict):
            match_value = value.get("match", value.get("value", value))
        else:
            match_value = value
        must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=match_value)))

    if not must:
        return None
    return qmodels.Filter(must=must)


def _point_to_record(point: Any):
    payload = dict(getattr(point, "payload", None) or {})
    document = payload.pop("document", getattr(point, "document", None))
    document_id = payload.pop("document_id", getattr(point, "id", None))
    metadata = payload
    vector = _normalize_point_vector(getattr(point, "vector", None))
    score = getattr(point, "score", None)
    return document_id, document, metadata, vector, score


def _points_to_result(points: Optional[List[Any]], *, nested: bool) -> dict[str, Any]:
    documents: List[str] = []
    metadatas: List[dict[str, Any]] = []
    embeddings: List[List[float]] = []
    ids: List[Any] = []
    distances: List[Optional[float]] = []
    uris: List[None] = []
    data: List[Any] = []

    for point in points or []:
        point_id, document, metadata, vector, score = _point_to_record(point)
        ids.append(point_id)
        documents.append(document)
        metadatas.append(metadata)
        embeddings.append(vector)
        distances.append(score)
        uris.append(None)
        data.append(point_id)

    if nested:
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "embeddings": [embeddings],
            "ids": [ids],
            "distances": [distances],
            "uris": [uris],
            "data": [data],
        }

    return {
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings,
        "ids": ids,
        "distances": distances,
        "uris": uris,
        "data": data,
    }


def _empty_query_result() -> dict[str, Any]:
    return {
        "documents": [[]],
        "metadatas": [[]],
        "embeddings": [[]],
        "ids": [[]],
        "distances": [[]],
        "uris": [[]],
        "data": [[]],
    }


def _empty_get_result() -> dict[str, Any]:
    return {
        "documents": [],
        "metadatas": [],
        "embeddings": [],
        "ids": [],
        "distances": [],
        "uris": [],
        "data": [],
    }


class QdrantCollectionAdapter:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(collection_name=self.collection_name)
            return True
        except Exception:
            return False

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def _scroll_all(self, limit: int = 256) -> List[Any]:
        points: List[Any] = []
        offset = None
        while True:
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception:
                break

            if isinstance(scroll_result, tuple) and len(scroll_result) == 2:
                batch, offset = scroll_result
            else:
                batch = getattr(scroll_result, "points", [])
                offset = getattr(scroll_result, "next_page_offset", None)

            if not batch:
                break

            points.extend(batch)
            if offset is None:
                break

        return points

    def add(self, documents=None, metadatas=None, embeddings=None, ids=None, **kwargs):
        if not documents:
            return None
        if not embeddings:
            raise ValueError("Qdrant upsert requires embeddings")
        if ids is None:
            raise ValueError("Qdrant upsert requires ids")

        vector_size = len(_normalize_point_vector(embeddings[0]))
        if vector_size == 0:
            raise ValueError("Embeddings must be non-empty vectors")

        self._ensure_collection(vector_size)

        points = []
        for index, point_id in enumerate(ids):
            payload = {
                key: core._sanitize_metadata_value(value)
                for key, value in ((metadatas[index] or {}) if metadatas and index < len(metadatas) else {}).items()
            }
            payload["document_id"] = str(point_id)
            payload["document"] = documents[index]
            points.append(
                qmodels.PointStruct(
                    id=_qdrant_point_id(point_id),
                    vector=_normalize_point_vector(embeddings[index]),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return None

    def get(self, ids=None, include=None, **kwargs):
        if not self._collection_exists():
            return _empty_get_result()

        if ids:
            try:
                storage_ids = [_qdrant_point_id(point_id) for point_id in ids]
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=storage_ids,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception:
                points = []
            return _points_to_result(points, nested=False)

        return _points_to_result(self._scroll_all(), nested=False)

    def query(self, query_embeddings=None, query_texts=None, n_results=10, where=None, include=None, **kwargs):
        if not self._collection_exists():
            return _empty_query_result()

        qfilter = _build_qdrant_filter(where)

        if query_embeddings:
            query_vector = query_embeddings[0]
            if isinstance(query_vector, list) and query_vector and isinstance(query_vector[0], list):
                query_vector = query_vector[0]
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=qfilter,
                    limit=n_results,
                    with_payload=True,
                    with_vectors=True,
                )
                points = getattr(response, "points", response)
            except Exception:
                points = []
            return _points_to_result(points, nested=True)

        if query_texts is not None:
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qfilter,
                    limit=n_results,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception:
                scroll_result = ([], None)

            if isinstance(scroll_result, tuple) and len(scroll_result) == 2:
                points, _offset = scroll_result
            else:
                points = getattr(scroll_result, "points", [])

            return _points_to_result(points, nested=True)

        return _empty_query_result()


def _build_qdrant_client(qdrant_url: Path):
    url = str(qdrant_url)
    if "://" not in url:
        url = QDRANT_DEFAULT_URL
    return QdrantClient(url=url)


def get_collection(chroma_dir: Path, collection_name: str = "receipts") -> tuple:
    client = _build_qdrant_client(chroma_dir)
    logging.info("Connected to Qdrant at %s", chroma_dir)
    collection = QdrantCollectionAdapter(client, collection_name=collection_name)
    logging.info("Using collection '%s'", collection_name)
    return client, collection

get_vectors_by_ids = partial(core.get_vectors_by_ids, get_collection_fn=get_collection)
dump_collection_sample = partial(core.dump_collection_sample, get_collection_fn=get_collection)
list_ids = partial(core.list_ids, get_collection_fn=get_collection)
metadata_by_id = partial(core.metadata_by_id, get_collection_fn=get_collection)
ingest_directory = partial(core.ingest_directory, get_collection_fn=get_collection)
retrieve = partial(core.retrieve, get_collection_fn=get_collection)
search_tool = partial(core.search_tool, get_collection_fn=get_collection)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["ingest", "retrieve", "inspect", "qwen-hf", "embed"], help="Command to execute"
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=False,
        help="Directory containing files to ingest (required for 'ingest' command)",
    )
    p.add_argument(
        "--collection_name",
        dest="collection_name",
        type=str,
        default="receipts",
        help="Qdrant collection to use",
    )
    p.add_argument(
        "--metadata",
        dest="metadata",
        action="append",
        default=None,
        metavar="KEY=VALUE|JSON",
        help=(
            "Additional metadata applied to every ingested record. "
            "Use repeated key=value pairs or a JSON object string."
        ),
    )
    p.add_argument(
        "--qdrant_url",
        type=str,
        default=QDRANT_DEFAULT_URL,
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    p.add_argument("--chroma_dir", dest="qdrant_url", type=str, help=argparse.SUPPRESS)
    p.add_argument(
        "--embed_provider",
        choices=["openai", "sentence-transformers", "ollama"],
        default="sentence-transformers",
    )
    p.add_argument("--embed_model", type=str, default=None)
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--chunk_overlap", type=int, default=200)
    p.add_argument(
        "--chunk_strategy",
        choices=list(core.CHUNK_STRATEGIES),
        default=core.DEFAULT_CHUNK_STRATEGY,
        help="Chunking strategy to use for ingesting text",
    )
    p.add_argument("--retrieve_query", type=str, default="example query for retrieval")
    p.add_argument(
        "--query_string",
        type=str,
        default=None,
        help="Query string to embed when running the 'embed' command",
    )
    p.add_argument(
        "--use_multimodal",
        action="store_true",
        help=(
            "Use multimodal sentence-transformers embeddings for images and PDFs. "
            "When enabled, PDF/image files bypass docling conversion."
        ),
    )
    p.add_argument(
        "inspect_action",
        nargs="?",
        default="list_ids",
        choices=["list_ids", "metadata"],
        help="Action for 'inspect' command (default: list_ids)",
    )
    p.add_argument(
        "--doc-id",
        dest="doc_id",
        type=str,
        default=None,
        help="Document id to inspect (used with 'metadata')",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.command == "retrieve":
        core.run_retrieve_command(args, collection_target=args.qdrant_url, get_collection_fn=get_collection)
    elif args.command == "embed":
        core.run_embed_command(args)
    elif args.command == "ingest":
        core.run_ingest_command(args, collection_target=args.qdrant_url, get_collection_fn=get_collection)
    elif args.command == "inspect":
        core.run_inspect_command(args, collection_target=args.qdrant_url, get_collection_fn=get_collection)
    elif args.command == "qwen-hf":
        core.run_qwen_hf_command(args, collection_target=args.qdrant_url, get_collection_fn=get_collection)

if __name__ == "__main__":
    main()
