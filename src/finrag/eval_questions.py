"""Build the evaluation set: draft questions from real passages, review them, keep the good ones.

Step 1 - generate drafts (uses your local LLM through Ollama):
    python -m finrag.eval_questions generate --n 70
  -> eval/questions_draft.jsonl  (full records)
  -> eval/review.csv             (open in Excel: one row per question, with file, page and evidence)

Step 2 - review by hand: open each PDF at the given page, check the answer, fix the question or
the answer if needed, and put "o" in the "garder" column for every question to keep.

Step 3 - finalize:
    python -m finrag.eval_questions finalize
  -> eval/questions.jsonl  (the reference set used by the evaluation script)
"""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import ollama
from tqdm import tqdm

from .config import CHUNKS_FILE, LLM_MODEL, OLLAMA_HOST, ROOT

EVAL_DIR = ROOT / "eval"
DRAFT_FILE = EVAL_DIR / "questions_draft.jsonl"
REVIEW_FILE = EVAL_DIR / "review.csv"
FINAL_FILE = EVAL_DIR / "questions.jsonl"

COMPANY_NAMES = {
    "aircanada": "Air Canada",
    "banquenationale": "Banque Nationale du Canada",
    "cgi": "CGI",
    "couchetard": "Alimentation Couche-Tard",
    "desjardins": "Mouvement Desjardins",
    "shopify": "Shopify",
}

SYSTEM_PROMPT = """You write evaluation questions for a question-answering system over company annual reports.
You receive ONE passage. Write ONE question that this passage alone answers, and its answer.
Rules:
- The question must be self-contained: name the company and the fiscal year explicitly.
  Never refer to "this passage", "this table", "the document" or "the report above".
- The answer must be fully supported by the passage. For a figure, copy it exactly, with its unit.
- "evidence" must be an exact, verbatim excerpt of the passage (one sentence or one table row)
  that contains the answer. Do not paraphrase it.
- Ask what a financial analyst would ask. Avoid trivial questions (page numbers, headings,
  addresses, publication dates).
- Return only JSON: {"question": "...", "answer": "...", "evidence": "..."}"""

QUESTION_TYPES = {
    "numeric": "a specific figure (an amount, a percentage, a ratio or a count)",
    "qualitative": "an explanation, a cause, a strategy or a description, not a single figure",
}
LANGUAGES = {"fr": "French", "en": "English"}

UNANSWERABLE = [
    ("fr", "Quel était le résultat net de l'entreprise {company} pour l'exercice 2030 ?"),
    ("en", "What was {company}'s total revenue for fiscal 2030?"),
    ("fr", "Quel était le résultat net de Bombardier pour l'exercice 2021 ?"),
    ("en", "What was Rogers Communications' total revenue in fiscal 2022?"),
]

HEADER_RE = re.compile(r"^\[(Text|Table) \|[^\]]*\]\n")
NUMBER_RE = re.compile(r"\d[\d\s.,]*\d|\d")
VAGUE_RE = re.compile(r"\b(passage|ce tableau|this table|the table above|ci-dessus|above)\b", re.I)


def strip_header(text: str) -> str:
    return HEADER_RE.sub("", text, count=1)


def normalize(text: str) -> str:
    text = text.replace(" ", " ").replace(" ", " ").replace("|", " ").lower()
    return re.sub(r"\s+", " ", text).strip()


def numbers(text: str) -> set[str]:
    return {re.sub(r"\D", "", n) for n in NUMBER_RE.findall(text.replace(" ", " "))}


def is_informative(body: str, kind: str) -> bool:
    """Skip passages that rarely yield good questions: short pages, tables of contents, legal boilerplate."""
    if len(body) < (150 if kind == "table" else 500):
        return False
    if body.count("....") >= 3:                      # dot leaders: table of contents
        return False
    digits = sum(ch.isdigit() for ch in body)
    if kind == "table" and digits < 15:
        return False
    if kind == "text" and digits / len(body) > 0.25:  # number soup, usually a badly parsed table
        return False
    return True


def load_candidates() -> dict[tuple[str, str], dict[int, list[dict]]]:
    """Group usable chunks by (company, kind), then by fiscal year."""
    groups: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            body = strip_header(c["text"])
            if is_informative(body, c["kind"]):
                groups[(c["company"], c["kind"])][c["year"]].append(
                    {k: c[k] for k in ("id", "company", "year", "page", "kind", "source")} | {"body": body}
                )
    return groups


def extract_json(content: str) -> dict:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError("no JSON object in the model output")
    return json.loads(match.group(0))


def ask_model(client, model: str, chunk: dict, qtype: str, lang: str, seed: int) -> dict:
    company = COMPANY_NAMES.get(chunk["company"], chunk["company"])
    user = (
        f"Company: {company}\nFiscal year: {chunk['year']}\n"
        f"Question type: {QUESTION_TYPES[qtype]}\n"
        f"Write the question and the answer in {LANGUAGES[lang]}.\n\n"
        f"Passage:\n{chunk['body']}"
    )
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        format="json",
        options={"temperature": 0.3, "num_ctx": 8192, "seed": seed},
    )
    return extract_json(response.message.content)


def check(item: dict, chunk: dict) -> tuple[bool, list[str]]:
    """Return (accept, review notes). Rejects drafts whose answer cannot be found in the passage."""
    question, answer, evidence = (str(item.get(k, "")).strip() for k in ("question", "answer", "evidence"))
    if not question or not answer:
        return False, ["empty question or answer"]
    body = normalize(chunk["body"])
    evidence_found = bool(evidence) and normalize(evidence) in body
    answer_numbers = {n for n in numbers(answer) if n}
    numbers_found = answer_numbers <= numbers(chunk["body"])
    years = {str(y) for y in range(chunk["year"] - 10, chunk["year"] + 2)}
    if answer_numbers and answer_numbers <= years:   # "2019" is not a real answer
        return False, ["answer is only a year"]
    if not evidence_found and not (answer_numbers and numbers_found):
        return False, ["answer not traceable to the passage"]
    notes = []
    if not evidence_found:
        notes.append("extrait reformulé : retrouver la phrase exacte dans le PDF")
    if answer_numbers and not numbers_found:
        notes.append("un chiffre de la réponse est absent du passage")
    if str(chunk["year"]) not in question:
        notes.append("l'année n'est pas dans la question")
    if VAGUE_RE.search(question):
        notes.append("question pas autonome (« ce tableau », « passage »...)")
    return True, notes


def generate(n: int, tables_share: float, n_unanswerable: int, model: str, seed: int) -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"{CHUNKS_FILE} not found. Run `python -m finrag.ingest` first.")
    rng = random.Random(seed)
    groups = load_candidates()
    companies = sorted({company for company, _ in groups})
    if not companies:
        raise SystemExit("No usable passage found in chunks.jsonl.")

    client = ollama.Client(host=OLLAMA_HOST)
    used_pages: set[tuple[str, int, int]] = set()
    drafts: list[dict] = []
    attempts = 0
    progress = tqdm(total=n, desc="Questions")
    while len(drafts) < n and attempts < 4 * n:
        i = attempts
        attempts += 1
        company = companies[i % len(companies)]
        kind = "table" if rng.random() < tables_share else "text"
        if (company, kind) not in groups:
            kind = "text" if kind == "table" else "table"
        years = groups[(company, kind)]
        year = rng.choice(sorted(years))
        chunk = rng.choice(years[year])
        page_key = (chunk["company"], chunk["year"], chunk["page"])
        if page_key in used_pages:
            continue
        qtype = "numeric" if kind == "table" or i % 3 else "qualitative"
        lang = "fr" if i % 2 == 0 else "en"
        try:
            item = ask_model(client, model, chunk, qtype, lang, seed + i)
        except Exception as exc:  # bad JSON, timeout: skip this passage
            tqdm.write(f"skipped {chunk['id']}: {exc}")
            continue
        accept, notes = check(item, chunk)
        if not accept:
            continue
        used_pages.add(page_key)
        drafts.append({
            "id": f"q{len(drafts) + 1:03d}",
            "question": item["question"].strip(),
            "expected_answer": item["answer"].strip(),
            "evidence": str(item.get("evidence", "")).strip(),
            "answerable": True,
            "type": qtype,
            "lang": lang,
            "company": chunk["company"],
            "year": chunk["year"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["id"],
            "kind": chunk["kind"],
            "notes": notes,
        })
        progress.update(1)
    progress.close()
    if len(drafts) < n:
        print(f"Only {len(drafts)} questions after {attempts} attempts.")

    for j in range(n_unanswerable):
        lang, template = UNANSWERABLE[j % len(UNANSWERABLE)]
        company = COMPANY_NAMES.get(companies[j % len(companies)], companies[j % len(companies)])
        drafts.append({
            "id": f"u{j + 1:03d}", "question": template.format(company=company),
            "expected_answer": "Information absente des documents",
            "evidence": "", "answerable": False, "type": "unanswerable", "lang": lang,
            "company": None, "year": None, "source": None, "page": None,
            "chunk_id": None, "kind": None, "notes": [],
        })

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with DRAFT_FILE.open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    # Semicolon + UTF-8 BOM: opens correctly in Excel with French regional settings.
    with REVIEW_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "question", "reponse_attendue", "fichier", "page", "type",
                         "langue", "extrait", "remarques", "garder"])
        for d in drafts:
            writer.writerow([d["id"], d["question"], d["expected_answer"], d["source"] or "",
                             d["page"] or "", d["type"], d["lang"], d["evidence"],
                             " / ".join(d["notes"]), ""])
    flagged = sum(bool(d["notes"]) for d in drafts)
    print(f"{len(drafts)} drafts ({flagged} flagged for review) -> {REVIEW_FILE}")


def finalize() -> None:
    if not (REVIEW_FILE.exists() and DRAFT_FILE.exists()):
        raise SystemExit("Run `generate` first, then review eval/review.csv.")
    drafts = {}
    with DRAFT_FILE.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            drafts[d["id"]] = d
    kept = []
    with REVIEW_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row.get("garder", "").strip().lower() not in {"o", "oui", "y", "yes", "x", "1"}:
                continue
            d = dict(drafts.get(row["id"], {"id": row["id"]}))
            # Edits made in the spreadsheet win over the draft.
            d["question"] = row["question"].strip()
            d["expected_answer"] = row["reponse_attendue"].strip()
            d["source"] = row["fichier"].strip() or None
            d["page"] = int(row["page"]) if row["page"].strip().isdigit() else None
            if row.get("type", "").strip():
                d["type"] = row["type"].strip()
            if row.get("langue", "").strip():
                d["lang"] = row["langue"].strip()
            d["answerable"] = d.get("type") != "unanswerable"
            if d["source"]:   # the file is the reference: keep company/year consistent with it
                match = re.match(r"^([a-z0-9-]+)_(\d{4})", d["source"].lower())
                if match:
                    d["company"], d["year"] = match[1], int(match[2])
            d.pop("notes", None)
            d.pop("evidence", None)
            kept.append(d)
    with FINAL_FILE.open("w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(kept)} questions kept -> {FINAL_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FinRAG evaluation set.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="draft questions from real passages")
    gen.add_argument("--n", type=int, default=70, help="number of answerable drafts")
    gen.add_argument("--tables-share", type=float, default=0.35, help="share of questions from tables")
    gen.add_argument("--unanswerable", type=int, default=4, help="number of questions with no answer")
    gen.add_argument("--model", default=LLM_MODEL)
    gen.add_argument("--seed", type=int, default=42)
    sub.add_parser("finalize", help="keep the reviewed questions")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.n, args.tables_share, args.unanswerable, args.model, args.seed)
    else:
        finalize()


if __name__ == "__main__":
    main()
