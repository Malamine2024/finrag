"""Compare two evaluation runs question by question:

    python -m finrag.diff_runs 20261027_1724 20261028_0040

Each argument is the start of a run file name in eval/results (its date stamp is enough); the regraded file
is preferred when there is one. Prints the questions whose retrieval or answer changed, with what each
run retrieved, to see why.
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"
GOOD = {"correct": 2, "partial": 1, "incorrect": 0}


def find_run(prefix: str) -> Path:
    files = sorted(f for f in RESULTS_DIR.glob(f"{prefix}*.jsonl"))
    if not files:
        raise SystemExit(f"No run file starting with '{prefix}' in {RESULTS_DIR}")
    regraded = [f for f in files if f.stem.endswith("_regraded")]
    return (regraded or files)[-1]


def load(path: Path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["id"]: r for r in rows}


def describe(r: dict) -> str:
    return f"{r['id']} [{r.get('kind') or '-'}|{r.get('type') or '-'}|{r.get('lang') or '-'}|{r.get('company')}]"


def show(title: str, ids: list[str], old: dict, new: dict, details: bool) -> None:
    print(f"\n## {title}: {len(ids)}")
    for i in ids:
        o, n = old[i], new[i]
        print(f"- {describe(n)} {n['question'][:110]}")
        if details:
            print(f"    expected: {n.get('source')} p.{n.get('page')}  ->  {str(n.get('expected_answer'))[:100]}")
            print(f"    before:   {', '.join(o.get('retrieved') or [])}")
            print(f"    after:    {', '.join(n.get('retrieved') or [])}")
            print(f"    answer:   {o['grade'].get('verdict')} -> {n['grade'].get('verdict')}: {(n.get('answer') or '')[:160]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation runs question by question.")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    before, after = find_run(args.before), find_run(args.after)
    old, new = load(before), load(after)
    print(f"before: {before.name}\nafter:  {after.name}")
    common = [i for i in new if i in old]

    def ret(r):
        return r["grade"].get("retrieval_ok")

    def good(r):
        return GOOD.get(r["grade"].get("verdict"))

    lost_ret = [i for i in common if ret(old[i]) is True and ret(new[i]) is False]
    won_ret = [i for i in common if ret(old[i]) is False and ret(new[i]) is True]
    lost_ans = [i for i in common if good(old[i]) is not None and good(new[i]) is not None and good(new[i]) < good(old[i])]
    won_ans = [i for i in common if good(old[i]) is not None and good(new[i]) is not None and good(new[i]) > good(old[i])]
    errors = [i for i in common if new[i]["grade"].get("verdict") == "error" or old[i]["grade"].get("verdict") == "error"]

    show("Retrieval lost", lost_ret, old, new, details=True)
    show("Retrieval gained", won_ret, old, new, details=False)
    show("Answer worse", lost_ans, old, new, details=True)
    show("Answer better", won_ans, old, new, details=False)
    show("Errors (excluded from the scores)", errors, old, new, details=False)
    for i in errors:
        for name, r in (("before", old[i]), ("after", new[i])):
            if r["grade"].get("verdict") == "error":
                print(f"    {i} {name}: {r.get('error') or r['grade'].get('reason')}")


if __name__ == "__main__":
    main()
