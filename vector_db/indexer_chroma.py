#!/usr/bin/env python3
"""Indexer example: OCR -> extract -> chunk -> embed -> Chroma

Usage example:
    python indexer.py --input_dir ./sample_docs --chroma_dir ./chroma_data \
        --embed_provider sentence-transformers --embed_model all-MiniLM-L6-v2

Ollama example (local):
    python indexer.py --input_dir ./sample_docs --chroma_dir ./chroma_data \
        --embed_provider ollama --embed_model nomic-embed-text

Multimodal example (opt-in):
    python indexer.py ingest --input_dir ./sample_docs --chroma_dir ./chroma_data \
        --embed_provider sentence-transformers --embed_model <multimodal-model> --use_multimodal

Requirements (examples):
    pip install chromadb sentence-transformers pdfplumber pytesseract pdf2image pillow openai
System deps for OCR/PDF to image:
    - tesseract (for pytesseract)
    - poppler (for pdf2image.convert_from_path)

This script is an example only — adapt models, batching, and error handling to your needs.
"""
import argparse
import logging
from functools import partial
from pathlib import Path

import chromadb

import indexer_core as core


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


def get_collection(chroma_dir: Path, collection_name: str = "receipts") -> tuple:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)
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
        help="Chroma collection to use",
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
    p.add_argument("--chroma_dir", type=Path, required=True)
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
        core.run_retrieve_command(args, collection_target=args.chroma_dir, get_collection_fn=get_collection)
    elif args.command == "embed":
        core.run_embed_command(args)
    elif args.command == "ingest":
        core.run_ingest_command(args, collection_target=args.chroma_dir, get_collection_fn=get_collection)
    elif args.command == "inspect":
        core.run_inspect_command(args, collection_target=args.chroma_dir, get_collection_fn=get_collection)
    elif args.command == "qwen-hf":
        core.run_qwen_hf_command(args, collection_target=args.chroma_dir, get_collection_fn=get_collection)

if __name__ == "__main__":
    main()
