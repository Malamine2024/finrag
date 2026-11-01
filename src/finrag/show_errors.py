"""Explain the wrong answers of a run whose retrieval succeeded: where was the right figure?

    python -m finrag.show_errors 20261028_1015 > eval\\errors.txt

For each answerable question graded partial or incorrect although the right page was retrieved, the search
is run again with the same settings (no answer generation: fast) and every passage given to the model is
listed with the expected figures it contains. That tells the error type:
  - figure in a passage, model gave another number    -> reading error (wrong line, column or table)
  - figure in a passage, model said "not found"         -> false refusal
  - figure in no passage (page retrieved by tolerance)  -> the right passage was not among the k
"""
import argparse
import json

from .diff_runs import find_run
from .run_eval import key_numbers, match_figures, numbers_in


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain the wrong answers of an evaluation run.")
    parser.add_argument("run", help="start of the run file name (its date stamp)")
    parser.add_argument("--all", action="store_true", help="also the questions whose retrieval failed")
    args = parser.parse_args()
    run_file = find_run(args.run)
    summary_file = run_file.with_name(run_file.name.replace("_regraded", "")).with_suffix(".summary.json")
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    rows = [json.loads(line) for line in run_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    from .generate import PERIOD_YEAR_WINDOW, retrieve_for_plan
    from .retrieve import retrieve, retrieve_period
    mode, use_rerank, k = summary.get("retrieval") or "vector", bool(summary.get("rerank")), summary.get("k") or 6
    print(f"run: {run_file.name} | search {mode}{' + rerank' if use_rerank else ''} | k={k}\n")

    wrong = [r for r in rows if r.get("answerable", True) and r["grade"].get("verdict") in {"partial", "incorrect"}
             and (args.all or r["grade"].get("retrieval_ok"))]
    counts = {"reading error": 0, "false refusal": 0, "figure not in the passages": 0, "qualitative": 0}
    for r in wrong:
        filters = r.get("filters") or {}
        if "companies" in filters:  # search plan (query_plan.py): same search as generate.answer
            hits, _ = retrieve_for_plan(r["question"], filters, k)
        elif filters.get("years"):   # period question: one search per year, as in generate.answer
            hits = retrieve_period(r["question"], filters["years"], filters.get("company"),
                                   window=PERIOD_YEAR_WINDOW, mode=mode, use_rerank=use_rerank)
        else:
            hits = retrieve(r["question"], k=k, company=filters.get("company"), year=filters.get("year"),
                            mode=mode, use_rerank=use_rerank)
        expected = key_numbers(r.get("expected_answer") or "")
        holders = []
        lines = []
        for i, h in enumerate(hits, start=1):
            m = h["metadata"]
            found = match_figures(expected, numbers_in(h["text"]))[0] if expected else 0
            if expected and found == len(expected):
                holders.append(i)
            right_page = m.get("source") == r.get("source") and abs((m.get("page") or -99) - (r.get("page") or -1)) <= 1
            first_line = h["text"].split("\n", 1)[0]
            lines.append(f"    [S{i}] {m.get('source')} p.{m.get('page')} {m.get('kind')}"
                         f"{'  <- expected page' if right_page else ''}"
                         f"{f'  figures {found}/{len(expected)}' if expected else ''}  {first_line[:110]}")
        if r.get("type") != "numeric" or not expected:
            kind = "qualitative"
        elif not holders:
            kind = "figure not in the passages"
        elif r["grade"].get("refused"):
            kind = "false refusal"
        else:
            kind = "reading error"
        counts[kind] += 1
        print(f"## {r['id']} [{kind}] {r['grade'].get('verdict')} | {r.get('company')} {r.get('year')} | "
              f"{r.get('source')} p.{r.get('page')}")
        print(f"  Q: {r['question']}")
        print(f"  expected: {r.get('expected_answer')}")
        print(f"  answer:   {(r.get('answer') or '').strip()[:400]}")
        if holders:
            print(f"  expected figures found in: {', '.join(f'S{i}' for i in holders)}")
        print("\n".join(lines))
        print()
    print("Summary:", ", ".join(f"{key} {value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
