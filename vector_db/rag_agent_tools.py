"""Utilities for the DeepAgents vector-store playground.

This module keeps the agent entrypoint thin while centralizing the local RAG
integration for Qdrant and Chroma.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import chromadb
from langchain_core.tools import tool
from qdrant_client import QdrantClient


DEFAULT_BACKEND = "qdrant"
DEFAULT_COLLECTION_NAME = "handbook"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_CHROMA_PATH = "./.chroma"
DEFAULT_EMBED_PROVIDER = "sentence-transformers"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_EMBED_MODEL = "embeddinggemma:latest"
DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class RetrievedDocument:
    id: str | None
    text: str
    metadata: dict[str, Any]
    score: float | None = None


class Embedder:
    def __init__(self, provider: str = DEFAULT_EMBED_PROVIDER, model_name: str | None = None):
        self.provider = (provider or DEFAULT_EMBED_PROVIDER).strip().lower()
        self.model_name = model_name or get_embed_model_name(self.provider)

        if self.provider == "openai":
            try:
                import openai

                self._client = openai
            except Exception as exc:
                raise ImportError("openai package required for provider=openai") from exc
        elif self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._fallback_dim = None
            except Exception:
                self._model = None
                self._fallback_dim = 384
        elif self.provider == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
            except Exception as exc:
                raise ImportError(
                    "langchain-ollama package required for provider=ollama"
                ) from exc

            self._model = OllamaEmbeddings(model=self.model_name)
        else:
            raise ValueError("Unsupported provider: %s" % self.provider)

    def embed(self, text: str) -> list[float]:
        if self.provider == "openai":
            response = self._client.Embedding.create(model=self.model_name, input=[text])
            return response["data"][0]["embedding"]

        if self.provider == "ollama":
            return self._model.embed_documents([text])[0]

        if self._model is None:
            return _hash_embedding(text, dim=self._fallback_dim or 384)

        vector = self._model.encode([text])
        first_vector = vector[0]
        if hasattr(first_vector, "tolist"):
            return first_vector.tolist()
        return list(first_vector)


def _hash_embedding(text: str, dim: int = 384) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[index % len(digest)] / 255.0 for index in range(dim)]


def _parse_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def get_backend_name() -> str:
    return os.environ.get("VECTOR_BACKEND", DEFAULT_BACKEND).strip().lower()


def get_collection_name() -> str:
    return os.environ.get("VECTOR_COLLECTION", DEFAULT_COLLECTION_NAME)


def get_embed_provider() -> str:
    return os.environ.get("EMBED_PROVIDER", DEFAULT_EMBED_PROVIDER).strip().lower()


def get_qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_chroma_path() -> Path:
    return Path(os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)).expanduser()


def get_chroma_url() -> str | None:
    raw_value = os.environ.get("CHROMA_URL")
    if raw_value is None:
        return None
    value = raw_value.strip()
    return value or None


def _default_embed_model_name(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_OPENAI_EMBED_MODEL
    if provider == "ollama":
        return DEFAULT_OLLAMA_EMBED_MODEL
    return DEFAULT_EMBED_MODEL


def get_embed_model_name(provider: str | None = None) -> str:
    raw_value = os.environ.get("EMBED_MODEL")
    if raw_value is not None and raw_value.strip():
        return raw_value

    resolved_provider = (provider or get_embed_provider()).strip().lower()
    return _default_embed_model_name(resolved_provider)


def get_top_k() -> int:
    return _parse_int_env("RAG_TOP_K", DEFAULT_TOP_K)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def format_retrieved_documents(
    documents: Sequence[RetrievedDocument],
    *,
    backend_name: str,
    collection_name: str,
    max_excerpt_chars: int = 1200,
) -> str:
    if not documents:
        return f"No relevant context found in {backend_name} collection '{collection_name}'."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = (
            document.metadata.get("source")
            or document.metadata.get("filename")
            or document.id
            or "unknown"
        )
        excerpt = document.text.strip()
        if max_excerpt_chars and len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[:max_excerpt_chars].rstrip() + " ..."

        block_lines = [f"[{index}] id={document.id or 'unknown'} source={source}"]
        if document.metadata:
            block_lines.append(f"metadata={_safe_json(document.metadata)}")
        block_lines.append(excerpt)
        blocks.append("\n".join(block_lines))

    header = f"Retrieved {len(documents)} document(s) from {backend_name} collection '{collection_name}':"
    return f"{header}\n\n" + "\n\n".join(blocks)


def _parse_chroma_url(url: str) -> tuple[str, int, bool]:
    candidate = url if "://" in url else f"http://{url}"
    parsed = urlparse(candidate)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8000)
    ssl = parsed.scheme == "https"
    return host, port, ssl


def _result_list(result: Any, key: str) -> list[Any]:
    if isinstance(result, dict):
        raw_value = result.get(key, [[]])
    else:
        raw_value = getattr(result, key, [[]])

    if not raw_value:
        return []
    if isinstance(raw_value, list) and raw_value and isinstance(raw_value[0], list):
        return list(raw_value[0])
    if isinstance(raw_value, tuple) and raw_value and isinstance(raw_value[0], list):
        return list(raw_value[0])
    return list(raw_value)


class QdrantBackend:
    backend_name = "qdrant"

    def __init__(self, url: str, collection_name: str, embedder: Embedder):
        self.url = url
        self.collection_name = collection_name
        self.embedder = embedder
        self.client = QdrantClient(url=url)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        embedding = self.embedder.embed(query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(result, "points", result)
        if points is None:
            return []

        documents: list[RetrievedDocument] = []
        for point in points:
            payload = getattr(point, "payload", None) or {}
            text = (
                payload.get("document")
                or payload.get("text")
                or payload.get("content")
                or ""
            )
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"document", "text", "content"}
            }
            point_id = getattr(point, "id", None)
            score = getattr(point, "score", None)
            documents.append(
                RetrievedDocument(
                    id=str(point_id) if point_id is not None else None,
                    text=str(text),
                    metadata=metadata,
                    score=float(score) if score is not None else None,
                )
            )
        return documents


class ChromaBackend:
    backend_name = "chroma"

    def __init__(
        self,
        *,
        path: Path | None,
        url: str | None,
        collection_name: str,
        embedder: Embedder,
    ):
        self.collection_name = collection_name
        self.embedder = embedder
        self.path = path
        self.url = url

        if url:
            host, port, ssl = _parse_chroma_url(url)
            self.client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
        else:
            resolved_path = path or Path(DEFAULT_CHROMA_PATH)
            self.client = chromadb.PersistentClient(path=str(resolved_path))

        self.collection = self.client.get_or_create_collection(name=collection_name)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        embedding = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings", "uris", "data"],
        )

        documents = _result_list(results, "documents")
        metadatas = _result_list(results, "metadatas")
        ids = _result_list(results, "ids")
        distances = _result_list(results, "distances")

        retrieved: list[RetrievedDocument] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            raw_id = ids[index] if index < len(ids) else None
            distance = distances[index] if index < len(distances) else None
            retrieved.append(
                RetrievedDocument(
                    id=str(raw_id) if raw_id is not None else None,
                    text=str(text or ""),
                    metadata=metadata,
                    score=float(distance) if distance is not None else None,
                )
            )
        return retrieved


def load_vector_backend(
    *,
    backend_name: str | None = None,
    collection_name: str | None = None,
    embed_provider: str | None = None,
    embed_model_name: str | None = None,
) -> QdrantBackend | ChromaBackend:
    resolved_backend = (backend_name or get_backend_name()).strip().lower()
    resolved_collection = collection_name or get_collection_name()
    resolved_provider = (embed_provider or get_embed_provider()).strip().lower()
    resolved_model_name = embed_model_name or get_embed_model_name(resolved_provider)
    embedder = Embedder(
        provider=resolved_provider,
        model_name=resolved_model_name,
    )

    if resolved_backend == "qdrant":
        return QdrantBackend(
            url=get_qdrant_url(),
            collection_name=resolved_collection,
            embedder=embedder,
        )
    if resolved_backend == "chroma":
        return ChromaBackend(
            path=get_chroma_path(),
            url=get_chroma_url(),
            collection_name=resolved_collection,
            embedder=embedder,
        )

    raise ValueError("VECTOR_BACKEND must be 'qdrant' or 'chroma'")


@lru_cache(maxsize=1)
def get_vector_backend() -> QdrantBackend | ChromaBackend:
    return load_vector_backend()


def retrieve_context_from_backend(
    backend: QdrantBackend | ChromaBackend,
    query: str,
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    documents = backend.retrieve(query, top_k=top_k)
    serialized = format_retrieved_documents(
        documents,
        backend_name=backend.backend_name,
        collection_name=backend.collection_name,
    )
    artifacts = [asdict(document) for document in documents]
    return serialized, artifacts


def build_retrieve_context_tool(
    *,
    backend: QdrantBackend | ChromaBackend | None = None,
    backend_name: str | None = None,
    collection_name: str | None = None,
    embed_provider: str | None = None,
    embed_model_name: str | None = None,
    top_k: int | None = None,
):
    if backend is not None:
        resolved_backend = backend
    elif any(
        value is not None
        for value in (backend_name, collection_name, embed_provider, embed_model_name)
    ):
        resolved_backend = load_vector_backend(
            backend_name=backend_name,
            collection_name=collection_name,
            embed_provider=embed_provider,
            embed_model_name=embed_model_name,
        )
    else:
        resolved_backend = get_vector_backend()
    resolved_top_k = top_k or get_top_k()

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Search the configured local vector store for relevant context."""
        return retrieve_context_from_backend(resolved_backend, query, resolved_top_k)

    return retrieve_context
