"""Compare table-extraction methods on pages whose tables were verified by hand.

    python -m finrag.compare_tables                          # all available methods
    python -m finrag.compare_tables --methods pymupdf_lines pymupdf_text docling

Each test case is (PDF, page, row label, figure): the figure must appear in the same table row as the
label, in a clean row (no "<br>", i.e. no merged cells). That is exactly what the LLM needs to read a
table correctly. The page and its two neighbours are searched, because tables often span pages.

Test cases are read from eval/table_cases.json (created with defaults on the first run: edit it freely).
Outputs, in eval/table_extraction/:
  <pdf>_p<page>/<method>.md   every table each method extracted, to inspect by eye
  report.md                   scores per method and the detail per test case
"""
import argparse
import json
import re
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import pymupdf

from .config import RAW_DIR, ROOT
from .run_eval import close, numbers_in, parse_number

CASES_FILE = ROOT / "eval" / "table_cases.json"
OUT_DIR = ROOT / "eval" / "table_extraction"

# Figures verified in the PDFs during the review of the evaluation set.
DEFAULT_CASES = [
    {"pdf": "banquenationale_2019.pdf", "page": 57, "label": "solde d'ouverture", "figure": "8 608"},
    {"pdf": "banquenationale_2017.pdf", "page": 85, "label": "certificats de dépôt", "figure": "9 623"},
    {"pdf": "banquenationale_2017.pdf", "page": 85, "label": "financement non garanti", "figure": "33 021"},
    {"pdf": "banquenationale_2020.pdf", "page": 191, "label": "payeur fixe", "figure": "1,7"},
    {"pdf": "aircanada_2021.pdf", "page": 133, "label": "au 31 décembre 2021", "figure": "638"},
    {"pdf": "aircanada_2021.pdf", "page": 133, "label": "amortissement", "figure": "(104)"},
    {"pdf": "couchetard_2020.pdf", "page": 114, "label": "salaires", "figure": "2 394,0"},
    {"pdf": "couchetard_2020.pdf", "page": 114, "label": "coût des ventes", "figure": "44 397,7"},
    {"pdf": "couchetard_2018.pdf", "page": 50, "label": "frais de restructuration", "figure": "51,7"},
    {"pdf": "couchetard_2018.pdf", "page": 50, "label": "baiia ajusté", "figure": "2 979,4"},
    {"pdf": "couchetard_2025.pdf", "page": 121, "label": "taux d'actualisation", "figure": "4,8"},
    {"pdf": "desjardins_2020.pdf", "page": 205, "label": "groupe d'assurances générales", "figure": "90"},
    {"pdf": "desjardins_2021.pdf", "page": 120, "label": "à la fin de l'exercice", "figure": "1 088"},
    {"pdf": "desjardins_2024.pdf", "page": 238, "label": "sinistres en suspens", "figure": "(240)"},
    {"pdf": "desjardins_2025.pdf", "page": 224, "label": "solde net à la fin", "figure": "952"},
    {"pdf": "cgi_2017.pdf", "page": 100, "label": "en pourcentage des revenus", "figure": "14,7"},
    {"pdf": "shopify_2025.pdf", "page": 101, "label": "deferral of revenue", "figure": "224"},
    {"pdf": "shopify_2025.pdf", "page": 101, "label": "balance, end of the year", "figure": "398"},
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("–", "-").lower()
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------- extractors ---
# Each extractor returns {page_number: [markdown_table, ...]} for the requested pages.

def _pymupdf(path: Path, pages: list[int], **settings) -> dict[int, list[str]]:
    out = {}
    with pymupdf.open(path) as doc:
        for p in pages:
            if 1 <= p <= doc.page_count:
                try:
                    tables = doc[p - 1].find_tables(**settings).tables
                except Exception:
                    tables = []
                out[p] = [t.to_markdown() for t in tables]
    return out


def plain_text(path: Path, pages: list[int]) -> dict[int, list[str]]:
    """What the text chunks contain today: one 'row' per line of page text."""
    out = {}
    with pymupdf.open(path) as doc:
        for p in pages:
            if 1 <= p <= doc.page_count:
                lines = [ln for ln in doc[p - 1].get_text("text").splitlines() if ln.strip()]
                out[p] = ["\n".join(f"| {ln} |" for ln in lines)]
    return out


def docling(path: Path, pages: list[int]) -> dict[int, list[str]]:
    """Docling (IBM): layout model + TableFormer. Converts only the requested pages."""
    import docling.document_converter  # noqa: F401  (raises ImportError early: pip install docling)

    first, last = min(pages), max(pages)
    out = {p: [] for p in pages}
    with pymupdf.open(path) as src, tempfile.TemporaryDirectory() as tmp:
        last = min(last, src.page_count)
        sub = pymupdf.open()
        sub.insert_pdf(src, from_page=first - 1, to_page=last - 1)
        sub_path = Path(tmp) / "pages.pdf"
        sub.save(sub_path)
        sub.close()
        result = _docling_converter().convert(str(sub_path))
    document = result.document
    for table in document.tables:
        page_no = table.prov[0].page_no + first - 1 if table.prov else first
        try:
            md = table.export_to_markdown(doc=document)
        except TypeError:  # older Docling versions
            md = table.export_to_markdown()
        out.setdefault(page_no, []).append(md)
    return out


_CONVERTER = None


def _docling_converter():
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
        options.do_ocr = False              # text is read from the PDF itself: OCR is slow and useless here
        options.do_table_structure = True   # TableFormer: the part we want
        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        options.accelerator_options = AcceleratorOptions(device=device)
        print(f"docling: OCR off, device = {device.value}")
        _CONVERTER = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    return _CONVERTER


METHODS = {
    "plain_text": plain_text,                                                       # no table detection
    "pymupdf_lines": lambda path, pages: _pymupdf(path, pages),                      # current ingestion
    "pymupdf_text": lambda path, pages: _pymupdf(path, pages, strategy="text"),      # alignment only
    "pymupdf_mixed": lambda path, pages: _pymupdf(                                   # columns from lines,
        path, pages, vertical_strategy="lines", horizontal_strategy="text"),         # rows from text
    "docling": docling,
}


# ------------------------------------------------------------- scoring ---
def rows_of(markdown: str) -> list[str]:
    return [ln for ln in markdown.splitlines() if ln.strip().startswith("|") and not re.fullmatch(r"[|\-: ]+", ln.strip())]


def check_case(case: dict, tables_by_page: dict[int, list[str]]) -> str:
    """'ok' (clean row with label and figure), 'merged' (only in a row with merged cells), or 'missing'."""
    label = normalize(case["label"])
    target = abs(parse_number(re.sub(r"[()]", "", case["figure"])) or 0)
    merged = False
    for page in (case["page"], case["page"] - 1, case["page"] + 1):
        for md in tables_by_page.get(page, []):
            for row in rows_of(md):
                text = normalize(row)
                if label in text and any(close(target, v) for v in numbers_in(row)):
                    if "<br>" in row:
                        merged = True
                    else:
                        return "ok"
    return "merged" if merged else "missing"


def cells_of(row: str) -> list[str]:
    row = row.strip()
    row = row[1:] if row.startswith("|") else row      # remove one outer pipe on each side only:
    row = row[:-1] if row.endswith("|") else row       # an empty first cell must survive
    return [c.strip() for c in row.split("|")]


def table_stats(tables: list[str]) -> dict:
    rows = [r for md in tables for r in rows_of(md)]
    cells = [c for r in rows for c in cells_of(r)]
    unlabelled = sum(1 for r in rows
                     if (parts := cells_of(r)) and not parts[0] and any(numbers_in(c) for c in parts[1:]))
    return {"tables": len(tables), "rows": len(rows),
            "merged_cells": sum("<br>" in c for c in cells), "unlabelled_rows": unlabelled}


# ---------------------------------------------------------------- main ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare table-extraction methods on verified pages.")
    parser.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    args = parser.parse_args()

    if not CASES_FILE.exists():
        CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
        CASES_FILE.write_text(json.dumps(DEFAULT_CASES, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Test cases written to {CASES_FILE} (edit them to add your own).")
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = [c for c in cases if (RAW_DIR / c["pdf"]).exists() or print(f"skipped, file not found: {c['pdf']}")]

    pages_by_pdf = defaultdict(set)
    for c in cases:
        pages_by_pdf[c["pdf"]].update({c["page"] - 1, c["page"], c["page"] + 1})

    results = defaultdict(dict)        # method -> case index -> status
    stats = defaultdict(lambda: defaultdict(int))
    timing = defaultdict(float)
    for method in args.methods:
        extractor = METHODS[method]
        for pdf, pages in sorted(pages_by_pdf.items()):
            start = time.perf_counter()
            try:
                tables_by_page = extractor(RAW_DIR / pdf, sorted(p for p in pages if p >= 1))
            except ImportError:
                print(f"{method}: not installed, skipped (pip install docling)")
                break
            except Exception as exc:
                print(f"{method} failed on {pdf}: {exc}")
                tables_by_page = {}
            timing[method] += time.perf_counter() - start
            for page, tables in tables_by_page.items():
                folder = OUT_DIR / f"{Path(pdf).stem}_p{page}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{method}.md").write_text("\n\n".join(tables) or "(no table)", encoding="utf-8")
                if method != "plain_text":
                    for key, value in table_stats(tables).items():
                        stats[method][key] += value
            for i, c in enumerate(cases):
                if c["pdf"] == pdf:
                    results[method][i] = check_case(c, tables_by_page)

    methods = [m for m in args.methods if results[m]]
    n_pages = sum(len(p) for p in pages_by_pdf.values())
    lines = ["# Table extraction comparison", "",
             f"{len(cases)} verified figures, {n_pages} pages.", "",
             "| Method | Figures in a clean row | Only in merged cells | Missing | Tables | Merged cells | "
             "Unlabelled rows | Seconds / page |",
             "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        status = list(results[m].values())
        s = stats[m]
        lines.append(f"| {m} | {status.count('ok')}/{len(status)} | {status.count('merged')} | "
                     f"{status.count('missing')} | {s['tables'] if m != 'plain_text' else '-'} | "
                     f"{s['merged_cells'] if m != 'plain_text' else '-'} | "
                     f"{s['unlabelled_rows'] if m != 'plain_text' else '-'} | {timing[m] / n_pages:.2f} |")
    lines += ["", "## Detail", "", "| PDF | Page | Label | Figure | " + " | ".join(methods) + " |",
              "|---|---|---|---|" + "---|" * len(methods)]
    for i, c in enumerate(cases):
        lines.append(f"| {c['pdf']} | {c['page']} | {c['label']} | {c['figure']} | "
                     + " | ".join(results[m].get(i, '-') for m in methods) + " |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nTables extracted by each method: {OUT_DIR}")


if __name__ == "__main__":
    main()
