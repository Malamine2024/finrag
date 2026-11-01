"""Extract text and tables from annual-report PDFs and split them into chunks.

PDF files must be named  <company>_<year>.pdf, e.g. banque-nationale_2024.pdf
Output: data/processed/chunks.jsonl (one chunk per line, with metadata).

Two extractors (FINRAG_EXTRACTOR or --extractor):
  docling  (default) Docling layout model + TableFormer, on the GPU when there is one. Text comes in reading
           order, without page headers and footers, and every table is rebuilt row by row (one figure, one
           label). See eval/table_extraction/report.md: 18/18 verified figures against 3/18 for pymupdf.
           Each converted window of pages is cached in data/processed/docling/, so an interrupted run
           resumes where it stopped. If Docling finds much less text on a page than PyMuPDF (a paragraph
           taken for a picture...), that page keeps the PyMuPDF text.
  pymupdf  the original extraction: page text + find_tables(). Fast, kept to compare before / after.

    python -m finrag.ingest                              # whole corpus -> chunks.jsonl
    python -m finrag.ingest --only aircanada_2021        # quick test   -> chunks_test.jsonl
    python -m finrag.ingest --extractor pymupdf          # original extraction
"""
import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows cache warning

import pymupdf as fitz
from tqdm import tqdm

from .config import (CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE, MIN_CHUNK_CHARS,
                     PROCESSED_DIR, RAW_DIR)

EXTRACTOR = os.getenv("FINRAG_EXTRACTOR", "docling").strip().lower()
DOCLING_DIR = PROCESSED_DIR / "docling"   # cache of converted pages: delete it to convert again
WINDOW = 40             # pages converted at once: bounds memory, and the unit of the cache
FALLBACK_RATIO = 0.7    # Docling text < 70 % of PyMuPDF text on a page -> keep PyMuPDF text for that page
MIN_PAGE_CHARS = 200    # ...only checked on pages with real content
MAX_TITLE_CHARS = 160

FILENAME_RE = re.compile(r"^(?P<company>[a-z0-9-]+)_(?P<year>\d{4})$")


@dataclass
class Chunk:
    id: str
    text: str
    company: str
    year: int
    page: int
    kind: str      # "text" or "table"
    source: str    # PDF file name


def parse_filename(path: Path) -> tuple[str, int]:
    match = FILENAME_RE.match(path.stem.lower())
    if not match:
        raise ValueError(f"Expected '<company>_<year>.pdf', got '{path.name}'")
    return match["company"], int(match["year"])


def clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_chars(text: str) -> int:
    """Characters that carry content (no spaces, no table pipes or rules): to compare extractors."""
    return len(re.sub(r"[\s|:\-]", "", text))


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, cutting on paragraph or sentence ends when possible."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def is_section_row(row: str) -> bool:
    """A row that only names a group of rows ("| Couvertures des flux de trésorerie | | |"): a label and no
    value. Docling repeats the text of a spanning cell in every column, so equal cells count as empty."""
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    label, rest = cells[0], cells[1:]
    return bool(label) and bool(rest) and all(c in ("", label) for c in rest)


def split_table(markdown: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split a markdown table by rows, repeating in every part the header and the group rows (section
    labels such as "Couvertures des flux de trésorerie > Risque de taux d'intérêt") the part falls under:
    without them, a part reads "Payeur fixe 2,1 % 1,9 %..." with no way to tell which hedges it is about."""
    lines = markdown.strip().splitlines()
    if len(markdown) <= size or len(lines) <= 3:
        return [markdown.strip()]
    header, rows = lines[:2], lines[2:]
    parts, current, groups, current_groups = [], [], [], []
    previous_was_group = False
    for row in rows:
        if current and len("\n".join(header + current_groups + current + [row])) > size:
            parts.append("\n".join(header + current_groups + current))
            current, current_groups = [], list(groups)
        if is_section_row(row):
            groups = (groups + [row])[-3:] if previous_was_group else [row]   # nested labels form a path
            previous_was_group = True
        else:
            previous_was_group = False
        current.append(row)
    if current:
        parts.append("\n".join(header + current_groups + current))
    return parts


# Lines that state the period or the unit of a table, not its subject: "Au 31 octobre", "Pour les exercices
# terminés le 31 décembre", "(en millions de dollars canadiens)", "Year ended December 31", "(In US $ millions)".
PERIOD_OR_UNIT_RE = re.compile(
    r"^\(?\s*(au|aux|pour l'exercice|pour les exercices|exercices? (clos|terminés?)|trimestres? (clos|terminés?)|"
    r"en (millions|milliers|milliards)|as at|as of|for the (years?|three|six|nine|quarters?)|years? ended|"
    r"three months|in (thousands|millions|us \$|canadian))(?!\w)", re.I)
TABLE_NUMBER_RE = re.compile(r"^(tableau|table|figure)\s+\d+", re.I)


def table_title(recent: list[str], section: str | None, repeated: set[str] = frozenset()) -> str | None:
    """Title of a table from the short lines just above it (most recent last).

    A numbered title ("Tableau 8 - Revenu net d'intérêts...") wins; a period or unit line is kept only as a
    complement ("... | Pour les exercices terminés le 31 décembre"); without a subject line, the section.
    Lines repeated on many pages of the report (running headers such as "États financiers consolidés
    audités", missed by the layout model) are never a subject.
    """
    recent = [t for t in recent if t not in repeated or PERIOD_OR_UNIT_RE.match(t)]
    numbered = [t for t in recent if TABLE_NUMBER_RE.match(t)]
    # A subject line has words: not a footnote mark ("1") or a lone year left above the table.
    subjects = [t for t in recent if not PERIOD_OR_UNIT_RE.match(t) and len(re.findall(r"[^\W\d_]", t)) >= 4]
    periods = [t for t in recent if PERIOD_OR_UNIT_RE.match(t)]
    subject = (numbered or subjects or [None])[-1] or section
    if subject and periods and periods[-1] != subject:
        return f"{subject} | {periods[-1]}"
    return subject or (periods[-1] if periods else None)


def short_title(title: str | None) -> str | None:
    if not title:
        return None
    title = re.sub(r"[\s|\[\]]+", " ", title).strip()
    return title[:MAX_TITLE_CHARS] or None


def context_header(company: str, year: int, page: int, kind: str, title: str | None = None) -> str:
    # Prepended to every chunk so the embedding and the LLM both know where it comes from.
    label = "Table" if kind == "table" else "Text"
    header = f"[{label} | Company: {company} | Fiscal year: {year} | Page: {page}"
    if title := short_title(title):
        header += f" | {'Title' if kind == 'table' else 'Section'}: {title}"
    return header + "]"


def make_chunks(company: str, year: int, page: int, source: str, text: str,
                tables: list[tuple[str | None, str]], section: str | None = None) -> list[Chunk]:
    """Chunks of one page: its text, then each table (title, markdown) split by rows."""
    chunks = []
    for i, part in enumerate(split_text(text) if text else []):
        if len(part) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                id=f"{company}_{year}_p{page}_t{i}",
                text=f"{context_header(company, year, page, 'text', section)}\n{part}",
                company=company, year=year, page=page, kind="text", source=source))
    for j, (title, markdown) in enumerate(tables):
        for k, part in enumerate(split_table(markdown)):
            if len(part) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(
                    id=f"{company}_{year}_p{page}_tab{j}_{k}",
                    text=f"{context_header(company, year, page, 'table', title)}\n{part}",
                    company=company, year=year, page=page, kind="table", source=source))
    return chunks


# -------------------------------------------------------------- pymupdf ---
def pymupdf_tables(page) -> list[tuple[None, str]]:
    try:
        return [(None, t.to_markdown()) for t in page.find_tables().tables]
    except Exception:  # table detection can fail on unusual layouts
        return []


def ingest_pdf_pymupdf(path: Path, extract_tables: bool = True, stats: dict | None = None) -> list[Chunk]:
    company, year = parse_filename(path)
    chunks: list[Chunk] = []
    with fitz.open(path) as doc:
        for page in doc:
            tables = pymupdf_tables(page) if extract_tables else []
            chunks += make_chunks(company, year, page.number + 1, path.name,
                                  clean(page.get_text("text")), tables)
    return chunks


# -------------------------------------------------------------- docling ---
_CONVERTER = None
SKIPPED_LABELS = {"page_header", "page_footer", "document_index", "picture", "chart"}
HEADING_LABELS = {"title", "section_header"}


def docling_converter():
    """Docling without OCR (the annual reports have a text layer) and on the GPU when there is one."""
    global _CONVERTER
    if _CONVERTER is None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        try:  # recent Docling versions
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        except ImportError:  # older ones
            from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
        import torch

        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = True
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        options.accelerator_options = AcceleratorOptions(device=device)
        tqdm.write(f"docling: OCR off, device = {device.value}")
        _CONVERTER = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return _CONVERTER


def load_document(path: Path):
    from docling_core.types.doc import DoclingDocument
    try:
        return DoclingDocument.load_from_json(path)
    except AttributeError:  # older docling-core
        return DoclingDocument.model_validate_json(path.read_text(encoding="utf-8"))


def docling_window(doc, path: Path, first: int, last: int):
    """DoclingDocument of pages first..last (1-based), from the cache or converted now."""
    cache = DOCLING_DIR / path.stem / f"p{first:04d}-{last:04d}.json"
    if cache.exists():
        return load_document(cache)
    # A crash in Docling's native code (no Python error, the process just stops) leaves this marker behind:
    # the next run then skips Docling for these pages instead of crashing again.
    marker = cache.with_name(cache.stem + ".started")
    if marker.exists():
        raise RuntimeError("the previous run crashed on these pages (delete the .started file to retry)")
    from docling.datamodel.base_models import DocumentStream

    sub = fitz.open()
    sub.insert_pdf(doc, from_page=first - 1, to_page=last - 1)
    data = sub.tobytes()
    sub.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    try:
        result = docling_converter().convert(DocumentStream(name=f"{path.stem}_p{first}.pdf", stream=BytesIO(data)))
    finally:
        marker.unlink(missing_ok=True)  # reached on a Python error, not on a hard crash
    status = getattr(result.status, "name", str(result.status))
    if status not in {"SUCCESS", "PARTIAL_SUCCESS"}:
        raise RuntimeError(f"conversion status {status}")
    if status == "SUCCESS":  # a partial result is used but not cached: the next run tries again
        tmp = cache.with_name(cache.stem + ".tmp.json")
        result.document.save_as_json(tmp)
        tmp.replace(cache)
    return result.document


def _label(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def table_markdown(item, document) -> str:
    try:
        markdown = item.export_to_markdown(doc=document)
    except TypeError:  # older Docling versions
        markdown = item.export_to_markdown()
    lines = []
    for line in markdown.splitlines():
        line = re.sub(r" {2,}", " ", line.replace(" ", " ")).rstrip()
        if re.fullmatch(r"[|\-: ]+", line):
            line = re.sub(r"-{3,}", "---", line)
        if line:
            lines.append(line)
    # Financial tables end with a total that has no label ("|  | 2 836,3 | 2 804,9 |"): name it, so the
    # LLM does not take it for the last line item.
    if len(lines) > 3 and re.match(r"^\|\s*\|", lines[-1]) and re.search(r"\d", lines[-1]):
        lines[-1] = re.sub(r"^\|\s*\|", "| Total |", lines[-1])
    return "\n".join(lines)


REPEATED_MIN_PAGES = 5   # a short line found on at least this many pages of a report is a running header


def repeated_lines(documents: list[tuple[int, object]]) -> set[str]:
    """Short text lines that appear on many different pages of a report."""
    pages_by_text: dict[str, set[int]] = {}
    for first, document in documents:
        for item, _level in document.iterate_items():
            prov = getattr(item, "prov", None)
            text = clean(getattr(item, "text", "") or "")
            if prov and text and len(text) <= 150 and _label(item) not in HEADING_LABELS:
                pages_by_text.setdefault(text, set()).add(prov[0].page_no + first - 1)
    return {t for t, pages in pages_by_text.items() if len(pages) >= REPEATED_MIN_PAGES}


def parse_document(document, first: int, state: dict) -> dict[int, dict]:
    """Group the items of a Docling document by page (numbered as in the original PDF).

    state["section"] carries the current section title from one window to the next.
    """
    pages: dict[int, dict] = {}
    for item, _level in document.iterate_items():
        label = _label(item)
        prov = getattr(item, "prov", None)
        if label in SKIPPED_LABELS or not prov:
            continue
        page = prov[0].page_no + first - 1
        entry = pages.setdefault(page, {"blocks": [], "tables": [], "chars": 0, "section": state["section"],
                                        "heading_seen": False, "recent_short": []})
        if label == "table":
            markdown = table_markdown(item, document)
            if not markdown:
                continue
            caption = item.caption_text(document) if hasattr(item, "caption_text") else ""
            # Annual reports rarely use real captions: the title is usually the short line just above.
            title = caption or table_title(entry["recent_short"], state["section"], state.get("repeated", set()))
            entry["tables"].append((title, markdown))
            entry["chars"] += content_chars(markdown)
            entry["recent_short"] = []
            continue
        text = clean(getattr(item, "text", "") or "")
        if not text:
            continue
        if label in HEADING_LABELS:
            state["section"] = short_title(text)
            if not entry["heading_seen"]:
                entry["section"], entry["heading_seen"] = state["section"], True
        entry["blocks"].append(text)
        entry["chars"] += content_chars(text)
        # Short lines since the previous table, at most 3: candidates for the next table's title.
        entry["recent_short"] = (entry["recent_short"] + [text])[-3:] if len(text) <= 150 else []
    return pages


def ingest_pdf_docling(path: Path, stats: dict, progress=None) -> list[Chunk]:
    company, year = parse_filename(path)
    chunks: list[Chunk] = []
    state = {"section": None}
    with fitz.open(path) as doc:
        # First pass: convert (or load) every window, to find the running headers of the whole report.
        windows = []
        for first in range(1, doc.page_count + 1, WINDOW):
            last = min(first + WINDOW - 1, doc.page_count)
            try:
                windows.append((first, last, docling_window(doc, path, first, last), None))
            except Exception as exc:
                windows.append((first, last, None, exc))
        state["repeated"] = repeated_lines([(f, d) for f, _, d, _ in windows if d is not None])
        for first, last, document, error in windows:
            pages = None
            if document is not None:
                try:
                    pages = parse_document(document, first, state)
                except Exception as exc:
                    error = exc
            if pages is None:
                tqdm.write(f"{path.name} p{first}-{last}: Docling failed ({error}), PyMuPDF used for these pages")
                stats["failed_pages"] += last - first + 1
            for p in range(first, last + 1):
                page = doc[p - 1]
                raw = clean(page.get_text("text"))
                if pages is None:
                    chunks += make_chunks(company, year, p, path.name, raw, pymupdf_tables(page))
                    continue
                entry = pages.get(p, {"blocks": [], "tables": [], "chars": 0, "section": state["section"]})
                text = "\n\n".join(entry["blocks"])
                if content_chars(raw) >= MIN_PAGE_CHARS and entry["chars"] < FALLBACK_RATIO * content_chars(raw):
                    text = raw                       # Docling missed text on this page: keep PyMuPDF's
                    stats["fallback_pages"] += 1
                chunks += make_chunks(company, year, p, path.name, text, entry["tables"], entry["section"])
            if progress is not None:
                progress.update(last - first + 1)
    return chunks


# ----------------------------------------------------------------- main ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and chunk annual-report PDFs.")
    parser.add_argument("--extractor", choices=["docling", "pymupdf"], default=EXTRACTOR)
    parser.add_argument("--no-tables", action="store_true", help="pymupdf only: skip table extraction")
    parser.add_argument("--only", nargs="*", metavar="NAME",
                        help="only these PDFs (e.g. aircanada_2021), written to chunks_test.jsonl")
    args = parser.parse_args()

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if args.only:
        wanted = {Path(n).stem.lower() for n in args.only}
        pdfs = [p for p in pdfs if p.stem.lower() in wanted]
    if not pdfs:
        raise SystemExit(f"No PDF found in {RAW_DIR}" + (f" matching {args.only}" if args.only else ""))
    if args.extractor == "docling":
        try:
            import docling.document_converter  # noqa: F401
        except ImportError:
            raise SystemExit("Docling is not installed: pip install docling (or use --extractor pymupdf)")

    out_file = PROCESSED_DIR / "chunks_test.jsonl" if args.only else CHUNKS_FILE
    tmp_file = out_file.with_name(out_file.stem + ".tmp.jsonl")  # the old file stays until the run completes
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"failed_pages": 0, "fallback_pages": 0}
    n_pages = 0
    for p in pdfs:
        with fitz.open(p) as d:
            n_pages += d.page_count
    start = time.perf_counter()
    total = n_tables_total = 0
    with tmp_file.open("w", encoding="utf-8") as out, tqdm(total=n_pages, desc="pages", unit="p") as progress:
        for pdf in pdfs:
            if args.extractor == "docling":
                chunks = ingest_pdf_docling(pdf, stats, progress)
            else:
                chunks = ingest_pdf_pymupdf(pdf, extract_tables=not args.no_tables)
                with fitz.open(pdf) as d:
                    progress.update(d.page_count)
            for chunk in chunks:
                out.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            n_tables = sum(c.kind == "table" for c in chunks)
            tqdm.write(f"{pdf.name}: {len(chunks)} chunks ({n_tables} from tables)")
            total += len(chunks)
            n_tables_total += n_tables
    tmp_file.replace(out_file)

    minutes = (time.perf_counter() - start) / 60
    print(f"Done in {minutes:.0f} min ({args.extractor}): {total} chunks, {n_tables_total} from tables, "
          f"{n_pages} pages -> {out_file}")
    if args.extractor == "docling":
        print(f"Pages that kept the PyMuPDF text (Docling found < {FALLBACK_RATIO:.0%} of it): "
              f"{stats['fallback_pages']}  |  pages where Docling failed: {stats['failed_pages']}")


if __name__ == "__main__":
    main()
