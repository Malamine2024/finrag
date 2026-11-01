"""Compare evaluation runs side by side:  python -m finrag.compare_runs  [--markdown]

Reads every eval/results/*.summary.json, oldest first. --markdown prints a table ready for the README.
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"

COLUMNS = [
    ("Date", lambda s: f"{s.get('date', '')}{' (re)' if s.get('regraded') else ''}"),
    ("Model", lambda s: s.get("model", "")),
    ("Prompt", lambda s: s.get("prompt", "v1")),
    ("num_ctx", lambda s: s.get("num_ctx", "-")),
    ("Think", lambda s: s.get("think", "-")),
    ("k", lambda s: s.get("k", "")),
    ("Filters", lambda s: (s.get("filters") or ("oracle" if s.get("use_filters") else "none"))
                          + (f"+{s['year_window']}y" if s.get("year_window") else "")
                          + (f"/{s['parser']}" if s.get("parser") not in (None, "regex") else "")),
    ("Search", lambda s: (s.get("retrieval") or "vector") + ("+rerank" if s.get("rerank") else "")),
    ("Verify", lambda s: s.get("verify") or "off"),
    ("N", lambda s: s.get("n_questions", "")),
    ("Accuracy", lambda s: pct(s.get("overall", {}).get("accuracy"))),
    ("Score", lambda s: pct(s.get("overall", {}).get("score"))),
    ("Retrieval", lambda s: pct(s.get("overall", {}).get("retrieval_hit"))),
    ("Tables", lambda s: pct(s.get("by_kind", {}).get("table", {}).get("accuracy"))),
    ("Declined", lambda s: pct(s.get("unanswerable_declined"))),
    ("False ref.", lambda s: s.get("false_refusals", "")),
    ("Wrong", lambda s: s.get("confident_errors", "-")),
    ("Prec. ans.", lambda s: pct(s.get("precision_when_answering"))),
    ("Lang ok", lambda s: pct(s.get("language_ok"))),
    ("s/question", lambda s: s.get("avg_seconds", "")),
]


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FinRAG evaluation runs.")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    files = sorted(RESULTS_DIR.glob("*.summary.json"))
    if not files:
        raise SystemExit(f"No summary in {RESULTS_DIR}. Run `python -m finrag.run_eval` first.")
    rows = [[str(get(json.loads(f.read_text(encoding="utf-8")))) for _, get in COLUMNS] for f in files]
    headers = [name for name, _ in COLUMNS]

    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        return
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
