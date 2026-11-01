"""List the evaluation questions whose reference answer may be wrong.

    python -m finrag.suspects > eval\\suspects.txt
    python -m finrag.suspects 20261028_0926 20261028_1015 20261028_1050    # only these runs

A question is suspect when several runs gave a confident answer (not a refusal) that was graded wrong:
the model keeps disagreeing with the reference, often with the same figure. Either the model misreads
the same table every time, or the reference is wrong (q010: the reference held the total revenue instead
of the merchant solutions line). Check each one in the PDF before changing anything.

By default, all runs with filters (auto or oracle) are used: without filters, the search often returned
another year, and those wrong answers say nothing about the reference.
"""
import argparse
import json
from collections import Counter, defaultdict

from .diff_runs import RESULTS_DIR, find_run, load


def main() -> None:
    parser = argparse.ArgumentParser(description="List evaluation questions whose reference may be wrong.")
    parser.add_argument("runs", nargs="*", help="start of run file names (date stamps); default: runs with filters")
    parser.add_argument("--min-runs", type=int, default=2, help="minimum number of confident wrong answers")
    args = parser.parse_args()

    if args.runs:
        files = [find_run(prefix) for prefix in args.runs]
    else:
        files = []
        for summary in sorted(RESULTS_DIR.glob("*.summary.json")):
            meta = json.loads(summary.read_text(encoding="utf-8"))
            if (meta.get("filters") or ("oracle" if meta.get("use_filters") else "none")) != "none":
                run = summary.with_name(summary.name.replace(".summary.json", ".jsonl"))
                if run.exists():
                    files.append(run)
    if not files:
        raise SystemExit("No run with filters found: pass the run date stamps.")
    print(f"{len(files)} runs: {', '.join(f.name[:13] for f in files)}\n")

    rows_by_id = defaultdict(list)
    for f in files:
        for qid, row in load(f).items():
            rows_by_id[qid].append(row)

    suspects = []
    for qid, rows in rows_by_id.items():
        if not rows[0].get("answerable", True):
            continue
        graded = [r for r in rows if r["grade"].get("verdict") in {"correct", "partial", "incorrect"}]
        confident_wrong = [r for r in graded if r["grade"]["verdict"] != "correct" and not r["grade"].get("refused")]
        if len(confident_wrong) >= args.min_runs:
            suspects.append((len(confident_wrong), len(graded), qid, confident_wrong))

    for n_wrong, n_graded, qid, wrong in sorted(suspects, key=lambda s: (-s[0] / s[1], s[2])):
        r = wrong[0]
        print(f"## {qid}  wrong and confident in {n_wrong}/{n_graded} runs | {r.get('source')} p.{r.get('page')}")
        print(f"  Q:         {r['question']}")
        print(f"  reference: {r.get('expected_answer')}")
        answers = Counter((w.get("answer") or "").strip().replace("\n", " ")[:220] for w in wrong)
        for text, count in answers.most_common(3):
            print(f"  model x{count}: {text}")
        print()
    print(f"{len(suspects)} suspect questions. For each: check the PDF page, then either fix expected_answer in "
          "eval/questions.jsonl (the reference was wrong) or leave it (the model misreads).")


if __name__ == "__main__":
    main()
