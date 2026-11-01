"""Add the completed questions of a draft file to eval/questions.jsonl.

    python -m finrag.add_questions eval/new_questions_draft.jsonl
    python -m finrag.add_questions eval/new_questions_draft.jsonl --set-category q039=comparative

A question is added only when it is complete: no "TODO" left in its question or expected answer, and an id
not already used. Questions still to complete are listed and left for later; run the command again once
they are done.
"""
import argparse
import json
from pathlib import Path

from .config import ROOT

QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
CATEGORIES = {"single", "period", "multi", "paraphrase", "comparative"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add completed draft questions to the reference set.")
    parser.add_argument("draft", type=Path)
    parser.add_argument("--set-category", nargs="*", default=[], metavar="ID=CATEGORY",
                        help="change the category of existing questions, e.g. q039=comparative")
    args = parser.parse_args()

    questions = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8") if line.strip()]
    ids = {q["id"] for q in questions}
    draft = [json.loads(line) for line in args.draft.open(encoding="utf-8") if line.strip()]

    added, pending = [], []
    for q in draft:
        if q["id"] in ids:
            continue                                   # already added by a previous run
        if "TODO" in q["question"] or "TODO" in str(q.get("expected_answer")):
            pending.append(q["id"])
            continue
        if q.get("category", "single") not in CATEGORIES:
            raise SystemExit(f"{q['id']}: unknown category {q.get('category')!r} (one of {sorted(CATEGORIES)})")
        questions.append(q)
        added.append(q["id"])

    changed = []
    for item in args.set_category:
        qid, _, category = item.partition("=")
        if category not in CATEGORIES:
            raise SystemExit(f"{item}: unknown category (one of {sorted(CATEGORIES)})")
        for q in questions:
            if q["id"] == qid:
                q["category"] = category
                changed.append(qid)

    QUESTIONS_FILE.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8")
    print(f"{len(questions)} questions in {QUESTIONS_FILE.name}")
    print(f"  added:    {', '.join(added) or '-'}")
    print(f"  pending:  {', '.join(pending) or '-'}   (complete the TODO fields, then run again)")
    if args.set_category:
        print(f"  category: {', '.join(changed) or '-'}")


if __name__ == "__main__":
    main()
