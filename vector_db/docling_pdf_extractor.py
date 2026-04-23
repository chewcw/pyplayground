#!/usr/bin/env python3
"""Extract PDF content into markdown files using Docling strategies.

The helper supports:
- hierarchical chunking via Docling's HierarchicalChunker
- hybrid chunking via Docling's HybridChunker
- page-by-page extraction
- bookmark/outline extraction
- full-document markdown export

Each extracted unit is written to its own markdown file and a manifest is
saved alongside the output for traceability.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from docling.chunking import HierarchicalChunker, HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)


STRATEGIES = ("hierarchical", "hybrid", "pages", "toc", "full")
DEFAULT_HYBRID_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HYBRID_MAX_TOKENS = 1000


@dataclass(frozen=True)
class ExtractionRecord:
    kind: str
    output_file: str
    title: str
    title_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    chunk_index: int | None = None


@dataclass(frozen=True)
class OutlineEntry:
    level: int
    title: str
    page_index: int
    title_path: tuple[str, ...]


@dataclass(frozen=True)
class OutlineSection:
    title_path: tuple[str, ...]
    page_start: int
    page_end: int


def _slugify(value: str, *, fallback: str = "section") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or fallback


def _render_markdown(heading_lines: Sequence[str], body: str) -> str:
    lines = [line.strip() for line in heading_lines if line and line.strip()]
    body = body.strip()
    if body:
        lines.append(body)
    if not lines:
        return ""
    return "\n\n".join(lines).rstrip() + "\n"


def _render_chunk_markdown(chunk_index: int, headings: Sequence[str], body: str) -> str:
    cleaned_headings = [heading.strip() for heading in headings if heading and heading.strip()]
    if cleaned_headings:
        heading_lines = [f"{'#' * (level + 1)} {heading}" for level, heading in enumerate(cleaned_headings)]
    else:
        heading_lines = [f"# Chunk {chunk_index}"]
    return _render_markdown(heading_lines, body)


def _render_page_markdown(page_number: int, body: str) -> str:
    return _render_markdown([f"# Page {page_number}"], body)


def _render_outline_markdown(title_path: Sequence[str], body: str) -> str:
    cleaned_title_path = [title.strip() for title in title_path if title and title.strip()]
    if cleaned_title_path:
        heading_lines = [f"{'#' * (level + 1)} {title}" for level, title in enumerate(cleaned_title_path)]
    else:
        heading_lines = ["# Section"]
    return _render_markdown(heading_lines, body)


def _build_output_file_name(
    source_stem: str,
    strategy: str,
    discriminator: str,
    *,
    title_parts: Sequence[str] = (),
) -> str:
    parts = [_slugify(source_stem), strategy, _slugify(discriminator)]
    parts.extend(_slugify(part) for part in title_parts if part and part.strip())
    return "__".join(parts) + ".md"


def parse_page_selection(page_spec: str, *, total_pages: int | None = None) -> list[int]:
    if not page_spec or not page_spec.strip():
        raise ValueError("Page selection cannot be empty")

    selected_pages: set[int] = set()
    for raw_part in page_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start_page = int(start_text)
            end_page = int(end_text)
            if start_page < 1 or end_page < start_page:
                raise ValueError(f"Invalid page range: {part}")
            page_range = range(start_page - 1, end_page)
        else:
            page_number = int(part)
            if page_number < 1:
                raise ValueError(f"Invalid page number: {part}")
            page_range = [page_number - 1]

        for page_index in page_range:
            if total_pages is not None and page_index >= total_pages:
                raise ValueError(
                    f"Page selection {page_spec!r} exceeds the document page count ({total_pages})."
                )
            selected_pages.add(page_index)

    if not selected_pages:
        raise ValueError("Page selection did not resolve to any pages")

    return sorted(selected_pages)


def _load_pypdf():
    from pypdf import PdfReader, PdfWriter

    return PdfReader, PdfWriter


def _get_total_page_count(source_pdf: Path) -> int:
    pdf_reader, _ = _load_pypdf()
    reader = pdf_reader(str(source_pdf))
    return len(reader.pages)


def _write_pdf_subset(source_pdf: Path, page_numbers: Sequence[int]) -> Path:
    pdf_reader, pdf_writer = _load_pypdf()
    reader = pdf_reader(str(source_pdf))
    writer = pdf_writer()
    for page_number in page_numbers:
        writer.add_page(reader.pages[page_number])

    temp_dir = Path(tempfile.mkdtemp(prefix="docling-pdf-extract-"))
    subset_path = temp_dir / f"{source_pdf.stem}__subset.pdf"
    with subset_path.open("wb") as handle:
        writer.write(handle)
    return subset_path


def _convert_source_to_doc(
    source_pdf: Path,
    converter: DocumentConverter,
    page_numbers: Sequence[int] | None = None,
):
    if page_numbers is None:
        result = converter.convert(str(source_pdf))
        return result.document

    subset_path = _write_pdf_subset(source_pdf, page_numbers)
    result = converter.convert(str(subset_path))
    return result.document


def _resolve_page_selection(source_pdf: Path, page_spec: str) -> list[int]:
    total_pages = _get_total_page_count(source_pdf)
    return parse_page_selection(page_spec, total_pages=total_pages)


def _iter_outline_entries(outline, reader, *, level: int = 1, parent_path: tuple[str, ...] = ()):
    previous_entry: OutlineEntry | None = None
    for item in outline:
        if isinstance(item, list):
            if previous_entry is not None:
                yield from _iter_outline_entries(
                    item,
                    reader,
                    level=previous_entry.level + 1,
                    parent_path=previous_entry.title_path,
                )
            continue

        page_index = reader.get_destination_page_number(item)
        if page_index is None or page_index < 0:
            continue

        title = str(getattr(item, "title", item)).strip() or "untitled"
        entry = OutlineEntry(
            level=level,
            title=title,
            page_index=page_index,
            title_path=parent_path + (title,),
        )
        yield entry
        previous_entry = entry


def _outline_sections(source_pdf: Path) -> list[OutlineSection]:
    pdf_reader, _ = _load_pypdf()
    reader = pdf_reader(str(source_pdf))
    outline = getattr(reader, "outline", []) or []
    outline_entries = list(_iter_outline_entries(outline, reader))
    if not outline_entries:
        return []

    total_pages = len(reader.pages)
    sections: list[OutlineSection] = []
    for index, entry in enumerate(outline_entries):
        page_end = total_pages - 1
        for next_entry in outline_entries[index + 1 :]:
            if next_entry.level <= entry.level:
                page_end = max(entry.page_index, next_entry.page_index - 1)
                break

        sections.append(
            OutlineSection(
                title_path=entry.title_path,
                page_start=entry.page_index,
                page_end=page_end,
            )
        )

    return sections


def _write_manifest(
    output_dir: Path,
    source_pdf: Path,
    strategy: str,
    page_spec: str | None,
    records: Sequence[ExtractionRecord],
) -> Path:
    manifest = {
        "source_pdf": str(source_pdf),
        "strategy": strategy,
        "page_spec": page_spec,
        "record_count": len(records),
        "records": [asdict(record) for record in records],
    }
    manifest_path = output_dir / f"{_slugify(source_pdf.stem)}__manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _extract_chunk_strategy(
    source_pdf: Path,
    output_dir: Path,
    converter: DocumentConverter,
    chunker,
    strategy: str,
    page_numbers: Sequence[int] | None = None,
) -> list[ExtractionRecord]:
    doc = _convert_source_to_doc(source_pdf, converter, page_numbers)
    records: list[ExtractionRecord] = []

    for chunk_index, chunk in enumerate(chunker.chunk(doc), start=1):
        headings = list(getattr(getattr(chunk, "meta", None), "headings", None) or [])
        body = str(getattr(chunk, "text", "") or "").strip()
        markdown = _render_chunk_markdown(chunk_index, headings, body)
        title_parts = [heading.strip() for heading in headings if heading and heading.strip()]
        output_file = _build_output_file_name(
            source_pdf.stem,
            strategy,
            f"chunk-{chunk_index:03d}",
            title_parts=title_parts,
        )
        output_path = output_dir / output_file
        output_path.write_text(markdown, encoding="utf-8")
        records.append(
            ExtractionRecord(
                kind="chunk",
                output_file=output_file,
                title=" > ".join(title_parts) if title_parts else f"Chunk {chunk_index}",
                title_path=tuple(title_parts),
                chunk_index=chunk_index,
            )
        )

    return records


def _extract_page_strategy(
    source_pdf: Path,
    output_dir: Path,
    converter: DocumentConverter,
    page_numbers: Sequence[int],
) -> list[ExtractionRecord]:
    records: list[ExtractionRecord] = []
    for page_number in page_numbers:
        doc = _convert_source_to_doc(source_pdf, converter, [page_number])
        body = str(doc.export_to_markdown() or "").strip()
        display_page = page_number + 1
        markdown = _render_page_markdown(display_page, body)
        output_file = _build_output_file_name(
            source_pdf.stem,
            "pages",
            f"page-{display_page:03d}",
            title_parts=[f"Page {display_page}"],
        )
        output_path = output_dir / output_file
        output_path.write_text(markdown, encoding="utf-8")
        records.append(
            ExtractionRecord(
                kind="page",
                output_file=output_file,
                title=f"Page {display_page}",
                title_path=(f"Page {display_page}",),
                page_start=display_page,
                page_end=display_page,
            )
        )

    return records


def _extract_outline_strategy(
    source_pdf: Path,
    output_dir: Path,
    converter: DocumentConverter,
) -> list[ExtractionRecord]:
    sections = _outline_sections(source_pdf)
    if not sections:
        raise ValueError("No bookmark outline entries were found in the PDF")

    records: list[ExtractionRecord] = []
    for section_index, section in enumerate(sections, start=1):
        page_numbers = list(range(section.page_start, section.page_end + 1))
        doc = _convert_source_to_doc(source_pdf, converter, page_numbers)
        body = str(doc.export_to_markdown() or "").strip()
        markdown = _render_outline_markdown(section.title_path, body)
        output_file = _build_output_file_name(
            source_pdf.stem,
            "toc",
            f"section-{section_index:03d}-pp{section.page_start + 1:03d}-{section.page_end + 1:03d}",
            title_parts=section.title_path,
        )
        output_path = output_dir / output_file
        output_path.write_text(markdown, encoding="utf-8")
        records.append(
            ExtractionRecord(
                kind="section",
                output_file=output_file,
                title=" > ".join(section.title_path),
                title_path=section.title_path,
                page_start=section.page_start + 1,
                page_end=section.page_end + 1,
            )
        )

    return records


def _extract_full_strategy(
    source_pdf: Path,
    output_dir: Path,
    converter: DocumentConverter,
    page_numbers: Sequence[int] | None = None,
) -> list[ExtractionRecord]:
    doc = _convert_source_to_doc(source_pdf, converter, page_numbers)
    body = str(doc.export_to_markdown() or "").strip()
    markdown = _render_markdown([f"# {source_pdf.stem}"], body)
    output_file = _build_output_file_name(source_pdf.stem, "full", "full")
    output_path = output_dir / output_file
    output_path.write_text(markdown, encoding="utf-8")
    return [
        ExtractionRecord(
            kind="full",
            output_file=output_file,
            title=source_pdf.stem,
            title_path=(source_pdf.stem,),
        )
    ]


def extract_pdf(
    source_pdf: str | Path,
    output_dir: str | Path,
    strategy: str,
    *,
    pages_spec: str | None = None,
    converter: DocumentConverter | None = None,
    hybrid_model: str = DEFAULT_HYBRID_MODEL,
    hybrid_max_tokens: int = DEFAULT_HYBRID_MAX_TOKENS,
) -> list[ExtractionRecord]:
    source_pdf = Path(source_pdf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = converter or DocumentConverter()

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Expected one of: {', '.join(STRATEGIES)}")

    if strategy == "toc" and pages_spec is not None:
        raise ValueError("Page selection is not supported together with the toc strategy")

    if strategy == "pages":
        if pages_spec is None:
            page_numbers = list(range(_get_total_page_count(source_pdf)))
        else:
            page_numbers = _resolve_page_selection(source_pdf, pages_spec)
        records = _extract_page_strategy(source_pdf, output_dir, converter, page_numbers)
    else:
        page_numbers = None
        if pages_spec is not None:
            page_numbers = _resolve_page_selection(source_pdf, pages_spec)

        if strategy == "hierarchical":
            chunker = HierarchicalChunker(always_emit_headings=True)
            records = _extract_chunk_strategy(
                source_pdf,
                output_dir,
                converter,
                chunker,
                "hierarchical",
                page_numbers=page_numbers,
            )
        elif strategy == "hybrid":
            tokenizer = HuggingFaceTokenizer.from_pretrained(
                model_name=hybrid_model,
                max_tokens=hybrid_max_tokens,
            )
            chunker = HybridChunker(tokenizer=tokenizer, always_emit_headings=True)
            records = _extract_chunk_strategy(
                source_pdf,
                output_dir,
                converter,
                chunker,
                "hybrid",
                page_numbers=page_numbers,
            )
        elif strategy == "toc":
            records = _extract_outline_strategy(source_pdf, output_dir, converter)
        else:
            records = _extract_full_strategy(
                source_pdf,
                output_dir,
                converter,
                page_numbers=page_numbers,
            )

    _write_manifest(output_dir, source_pdf, strategy, pages_spec, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract PDF sections with Docling")
    parser.add_argument("pdf", type=Path, help="Input PDF file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for extracted markdown files")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="hierarchical",
        help="Extraction strategy to use",
    )
    parser.add_argument(
        "--pages",
        dest="pages_spec",
        default=None,
        help="Optional 1-based page selection such as 1-3,5,8",
    )
    parser.add_argument(
        "--hybrid-model",
        default=DEFAULT_HYBRID_MODEL,
        help="Tokenizer model used by the hybrid chunker",
    )
    parser.add_argument(
        "--hybrid-max-tokens",
        type=int,
        default=DEFAULT_HYBRID_MAX_TOKENS,
        help="Maximum tokens per hybrid chunk",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    records = extract_pdf(
        args.pdf,
        args.output_dir,
        args.strategy,
        pages_spec=args.pages_spec,
        hybrid_model=args.hybrid_model,
        hybrid_max_tokens=args.hybrid_max_tokens,
    )

    manifest_path = args.output_dir / f"{_slugify(args.pdf.stem)}__manifest.json"
    print(f"Wrote {len(records)} file(s) to {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())