"""Phase D1: check every figure of an answer against the sources it cites.

The answer model can read the wrong line of a table (q033: 9 385, the interest income, given for the net
interest income 6 330), take a figure from another year, or write a figure found in no source at all. The
verifier reads each sentence of the answer, finds its figures and its citations [S2], and classifies every
figure:

  supported        found in a source the sentence cites
  other_source     found in another retrieved source, not the cited one (wrong citation)
  computed         a difference, sum, ratio or growth rate of figures of the cited sources
  unsupported      found nowhere: the figure has no basis in the passages given to the model

Matching accepts the formats and scales of the reports: "1 476", "1,476", "1 476,0", and the same figure in
thousands or in millions ("5 599 864" thousand = "5 599,9" million). Years, dates, citation numbers and the
figures of the question itself ("the last 3 years") are not checked.

This is a deterministic check: it says whether a figure exists in the sources, not whether it was read
from the right row or column. A figure that is in the source but on another line is still "supported";
an LLM check of each claim could catch those, at the cost of one call per claim.

    python -m finrag.verify eval/results/<run>.jsonl         # re-run the search, verify every saved answer
    python -m finrag.verify eval/results/<run>.jsonl --llm   # also check the meaning of each sentence
"""
import re

from .run_eval import CITATION_RE, NUMBER_RE, REL_TOL, is_year, parse_number

SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6)
# Dates: "31 décembre 2021", "December 31, 2021", "le 6 septembre 2017", "Q4", "T2" are not figures to check.
MONTHS = (r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|"
          r"decembre|january|february|march|april|may|june|july|august|september|october|november|december")
DATE_RE = re.compile(rf"\b\d{{1,2}}(?:er)?\s+(?:{MONTHS})\b|\b(?:{MONTHS})\s+\d{{1,2}}\b|\b[QT][1-4]\b", re.I)
CITED_RE = re.compile(r"\[([^\]]*S\d+[^\]]*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def figures(text: str) -> list[tuple[str, float]]:
    """(token, value) of the figures of a text, without citations, dates and years."""
    text = DATE_RE.sub(" ", CITATION_RE.sub(" ", CITED_RE.sub(" ", text or "")))
    text = re.sub(r"\bS\d+\b", " ", text)
    out = []
    for token in NUMBER_RE.findall(text):
        value = parse_number(token)
        if value is not None and not is_year(abs(value)):
            out.append((token.strip(), abs(value)))
    return out


def _close(a: float, b: float, tol: float = REL_TOL) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def found_in(value: float, numbers: list[float]) -> bool:
    return any(_close(value, n * s) for n in numbers for s in SCALES)


def computed_from(value: float, numbers: list[float]) -> str | None:
    """A difference, sum, ratio or growth rate of two source figures, or None."""
    values = sorted(set(numbers))[:150]          # bound the pairs on very long tables
    for a in values:
        for b in values:
            if a == b or b == 0:
                continue
            for label, result, tol in (("difference", a - b, REL_TOL), ("sum", a + b, REL_TOL),
                                       ("growth %", (a - b) / b * 100, 0.02), ("ratio %", a / b * 100, 0.02)):
                if result > 0 and any(_close(value, result * s, tol) for s in (1.0, 1e3, 1e-3)):
                    return f"{label} of {a:g} and {b:g}"
    return None


def cited_indexes(sentence: str) -> set[int]:
    return {int(n) for group in CITED_RE.findall(sentence) for n in re.findall(r"S(\d+)", group)}


def verify(answer: str, sources: list[dict], question: str = "") -> dict:
    """Classify every figure of the answer. Sources are the passages given to the model, [S1] first."""
    source_numbers = [[v for _, v in figures(s["text"])] for s in sources]
    question_values = [v for _, v in figures(question)]
    checks = []
    last_cited: set[int] = set()
    for sentence in SENTENCE_RE.split(answer or ""):
        # A sentence without a citation inherits the previous one (a table row under a cited heading).
        cited = cited_indexes(sentence) or last_cited
        last_cited = cited
        cited_numbers = [n for i in cited if 1 <= i <= len(sources) for n in source_numbers[i - 1]]
        all_numbers = [n for numbers in source_numbers for n in numbers]
        for token, value in figures(sentence):
            if found_in(value, question_values) and not found_in(value, all_numbers):
                continue                                          # taken from the question itself
            if found_in(value, cited_numbers):
                status, detail = "supported", None
            elif found_in(value, all_numbers):
                where = [f"S{i}" for i, numbers in enumerate(source_numbers, start=1) if found_in(value, numbers)]
                status, detail = "other_source", ", ".join(where)
            elif detail := computed_from(value, cited_numbers or all_numbers):
                status = "computed"
            else:
                status, detail = "unsupported", None
            checks.append({"figure": token, "status": status, "cited": sorted(cited), "detail": detail,
                           "sentence": sentence.strip()[:200]})
    unsupported = [c for c in checks if c["status"] == "unsupported"]
    return {"checks": checks, "n_figures": len(checks), "unsupported": [c["figure"] for c in unsupported],
            "other_source": [c["figure"] for c in checks if c["status"] == "other_source"],
            "ok": not unsupported}


CLAIM_PROMPT = """You check one sentence of an answer about annual reports against the passages it cites.
The figure may be written in another unit (thousands vs millions) or format (1 476 / 1,476): that is fine.
Check that each figure of the sentence is given in the passages FOR THE SAME metric (same table line),
the same company and the same period (fiscal year, quarter, column) as the sentence says.
Fiscal years do not always match calendar years: Couche-Tard's ends in late April (a date in September
2017 is in its fiscal 2018), CGI's on September 30, National Bank's on October 31; the others on
December 31. Judge periods by fiscal year.
Return only JSON: {"verdict": "supported" | "contradicted" | "not_found",
 "source_line": "<the line or phrase of the passage the figure comes from, or empty>",
 "reason": "<one short sentence>"}
- supported: the passages give this figure for this metric and period.
- contradicted: the figure is in the passages but for another metric, line, company or period, or the
  passages give another figure for this metric and period.
- not_found: the passages do not give this metric for this period."""


def llm_check(sentence: str, question: str, cited: list[tuple[int, dict]]) -> dict:
    """Ask the LLM whether the figures of a sentence match the metric and period of the cited passages."""
    import ollama
    from .config import LLM_MODEL, OLLAMA_HOST
    from .run_eval import ask_json
    per_passage = 2500 if len(cited) <= 3 else 900          # stay inside the context window with 10 passages
    passages = "\n\n".join(f"[S{i}] {h['text'][:per_passage]}" for i, h in cited)
    user = f"QUESTION:\n{question}\n\nSENTENCE:\n{sentence}\n\nPASSAGES:\n{passages}"
    try:
        data = ask_json(ollama.Client(host=OLLAMA_HOST), LLM_MODEL, CLAIM_PROMPT, user)
        verdict = str(data.get("verdict", "")).lower()
        return {"verdict": verdict if verdict in {"supported", "contradicted", "not_found"} else "error",
                "source_line": str(data.get("source_line", ""))[:200], "reason": str(data.get("reason", ""))[:200]}
    except Exception as exc:
        return {"verdict": "error", "source_line": "", "reason": f"check failed: {exc}"[:200]}


def verify_claims(answer: str, sources: list[dict], question: str = "") -> list[dict]:
    """LLM check of every sentence that contains a figure, against the passages it cites."""
    claims, last_cited = [], set()
    for sentence in SENTENCE_RE.split(answer or ""):
        cited = cited_indexes(sentence) or last_cited
        last_cited = cited
        if not figures(sentence):
            continue
        cited_hits = [(i, sources[i - 1]) for i in sorted(cited) if 1 <= i <= len(sources)]
        uncited = not cited_hits
        if uncited:
            # The first sentence of an answer, or the first line of a list, often comes before any citation:
            # check it against all the passages rather than calling it unsupported.
            cited_hits = list(enumerate(sources, start=1))
        claims.append({"sentence": sentence.strip()[:200], "uncited": uncited,
                       **llm_check(sentence, question, cited_hits)})
    return claims


def feedback(report: dict) -> str:
    """Instructions for a revision of the answer, from a verification report."""
    lines = [f"- \"{c['sentence']}\": {c['verdict']}. {c['reason']}" + (f" (source: {c['source_line']})" if c['source_line'] else "")
             for c in report.get("claims") or [] if c["verdict"] in {"contradicted", "not_found"}]
    for c in report["checks"]:
        if c["status"] == "unsupported":
            lines.append(f"- The figure {c['figure']} does not appear in any source (sentence: \"{c['sentence']}\").")
        elif c["status"] == "other_source":
            lines.append(f"- The figure {c['figure']} is not in the source you cited for it; it appears in {c['detail']}.")
    return "\n".join(lines)


def main() -> None:
    """Verify the saved answers of an evaluation run (the search is run again to get the passages)."""
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    from .generate import retrieve_for_plan
    parser = argparse.ArgumentParser(description="Verify the figures of the answers of an evaluation run.")
    parser.add_argument("run", type=Path)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--llm", action="store_true", help="also check each sentence with the LLM (slower)")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.run.open(encoding="utf-8") if line.strip()]
    statuses, verdicts = Counter(), Counter()
    flagged = {"correct": 0, "partial": 0, "incorrect": 0}
    answers = {"correct": 0, "partial": 0, "incorrect": 0}
    for r in rows:
        verdict = r["grade"].get("verdict")
        if not r.get("answerable", True) or r["grade"].get("refused") or r.get("error") or verdict not in answers:
            continue
        filters = r.get("filters") or {}
        if "companies" not in filters:
            continue
        hits, _ = retrieve_for_plan(r["question"], filters, args.k)
        report = verify(r["answer"], hits, r["question"])
        statuses.update(c["status"] for c in report["checks"])
        problems = list(report["unsupported"])
        claims = verify_claims(r["answer"], hits, r["question"]) if args.llm else []
        verdicts.update(c["verdict"] for c in claims)
        problems += [c["sentence"][:80] for c in claims if c["verdict"] in {"contradicted", "not_found"}]
        answers[verdict] += 1
        if problems:
            flagged[verdict] += 1
            print(f"- {r['id']} [{verdict}] {problems}")
            for c in claims:
                if c["verdict"] in {"contradicted", "not_found"}:
                    print(f"    {c['verdict']}: {c['reason']} | source: {c['source_line'][:100]}")
    print("\nFigures by status:", dict(statuses))
    if args.llm:
        print("Sentences by LLM verdict:", dict(verdicts))
    print("Answers flagged / answers:", ", ".join(f"{k} {flagged[k]}/{answers[k]}" for k in answers))
    print("A good verifier flags most incorrect and partial answers, and few correct ones.")


if __name__ == "__main__":
    main()
