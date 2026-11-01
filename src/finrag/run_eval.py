"""Evaluate the RAG pipeline on the reference set eval/questions.jsonl.

    python -m finrag.run_eval                      # full run, current LLM
    python -m finrag.run_eval --limit 5            # quick smoke test
    python -m finrag.run_eval --model qwen3.5:9b   # compare another generation model
    python -m finrag.run_eval --filters auto       # company/year detected in each question
    python -m finrag.run_eval --filters oracle     # company/year of the evaluation set (upper bound)
    python -m finrag.run_eval --retrieval vector   # search: vector, bm25 or hybrid (default)
    python -m finrag.run_eval --rerank on          # rerank the 30 best candidates (bge-reranker-v2-m3)
    python -m finrag.run_eval --regrade eval/results/<run>.jsonl   # grade saved answers again, no LLM call

For every question it measures two things separately:
  - retrieval: is the answer among the k retrieved passages? (right page, or the expected figures present)
  - answer:    numeric questions are graded by comparing figures (French and English formats);
               qualitative ones by an LLM judge; unanswerable ones must be declined.
Results go to eval/results/: one .jsonl with every question, one .summary.json with the scores.
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"

# ---------------------------------------------------------------- figures ---
NUMBER_RE = re.compile(
    r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?!\d)"   # 8 608 / 2 836,3 / 1 134 554
    r"|\d+(?:[.,]\d+)*"                                     # 14,119 / 45,7 / 2,836.3 / 7.31
)
REL_TOL = 0.005  # 0.5 %: accepts 2 836 for 2 836,3 or 61,0 G$ for 61 038 M$


def parse_number(token: str) -> float | None:
    s = re.sub(r"[   ]", "", token)
    if "," in s and "." in s:                      # the last separator is the decimal one
        dec, thou = (",", ".") if s.rfind(",") > s.rfind(".") else (".", ",")
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if all(len(p) == 3 for p in parts[1:]) else s.replace(",", ".", 1).replace(",", "")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


CITATION_RE = re.compile(r"\[\s*(?:S\d+\s*(?:[,;&]|and|et)?\s*)+\]")   # [S1], [S1, S2], [S1][S2]
SOURCE_TAG_RE = re.compile(r"\bS\d+\b")                                 # "according to S4", "[From S1, S2]"


def numbers_in(text: str) -> list[float]:
    # Remove citation markers only: an answer can be written entirely between brackets.
    text = SOURCE_TAG_RE.sub(" ", CITATION_RE.sub(" ", text or ""))
    values = []
    for token in NUMBER_RE.findall(text):
        value = parse_number(token)
        if value is not None:
            values.append(abs(value))                # (240) and "a decrease of 240" are the same figure
    return values


def is_year(value: float) -> bool:
    return value.is_integer() and 1990 <= value <= 2035


def key_numbers(expected: str) -> list[float]:
    """Figures the answer must contain. Years are context, unless the answer is only a year."""
    values = numbers_in(expected)
    non_years = [v for v in values if not is_year(v)]
    return non_years or values


def close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b), 1e-9)


def match_figures(expected: list[float], found: list[float]) -> tuple[int, bool]:
    """Return (number of expected figures found at the right scale, any found only at another scale)."""
    hits, scale_issue = 0, False
    for v in expected:
        if any(close(v, a) for a in found):
            hits += 1
        elif any(close(v, a * f) for a in found for f in (1e3, 1e-3, 1e6, 1e-6)):
            scale_issue = True
    return hits, scale_issue


# -------------------------------------------------------------- refusals ---
REFUSAL_MARKERS = [
    "pas trouvé", "introuvable", "ne contiennent pas", "ne contient pas", "aucune information",
    "pas d'information", "pas mentionn", "ne mentionnent pas", "ne permettent pas", "pas disponible",
    "n'ai pas pu", "je ne peux pas", "impossible de trouver", "pas en mesure",
    "not found", "could not find", "cannot find", "can't find", "do not contain", "does not contain",
    "no information", "not mentioned", "not provided", "not available", "unable to find", "unable to answer",
    "do not mention", "does not mention", "don't mention", "doesn't mention", "not covered", "no data",
    "aucune donnée", "aucune mention", "ne mentionne", "n'apparaît pas", "ne figure pas", "pas couvert",
    "not applicable", "ne s'applique pas",
    "找不到", "没有找到", "未找到", "无法找到", "无法提供", "没有提及", "未提及", "没有涉及",   # Qwen sometimes answers in Chinese
]


def is_refusal(answer: str) -> bool:
    text = (answer or "").lower().replace("’", "'")
    return any(marker in text for marker in REFUSAL_MARKERS)


# -------------------------------------------------------------- language ---
FR_WORDS = {"le", "la", "les", "de", "des", "du", "un", "une", "est", "et", "pour", "dans", "qui", "que", "pas",
            "sur", "au", "aux", "était", "été", "selon", "l", "d", "qu", "il", "elle", "ce", "cette", "sont",
            "ont", "avec", "en", "par", "plus", "ses", "son", "sa", "aucune", "trouvé"}
EN_WORDS = {"the", "of", "and", "is", "was", "for", "in", "to", "that", "with", "not", "as", "by", "according",
            "were", "are", "this", "from", "which", "its", "an", "be", "has", "have", "could", "find", "found"}


def detect_language(text: str) -> str:
    """Rough detection, enough to tell French, English and Chinese apart: 'fr', 'en', 'zh' or 'unknown'."""
    text = CITATION_RE.sub(" ", text or "")
    letters = sum(ch.isalpha() for ch in text)
    if letters and sum("\u4e00" <= ch <= "\u9fff" for ch in text) / letters > 0.2:
        return "zh"
    words = re.findall(r"[a-zà-ÿœ]+", text.lower())
    fr, en = sum(w in FR_WORDS for w in words), sum(w in EN_WORDS for w in words)
    if fr == en:
        return "unknown"
    return "fr" if fr > en else "en"


# ----------------------------------------------------------------- judge ---
JUDGE_PROMPT = """You grade answers produced by a question-answering system over company annual reports.
Compare the SYSTEM ANSWER with the REFERENCE ANSWER for the QUESTION.
Ignore the language (French or English), the wording and citation markers such as [S1].
Extra details, longer explanations or additional context are NOT errors: they must never
lower the verdict, as long as they do not contradict the reference.
- "correct": contains the key facts of the reference and nothing that contradicts them.
- "partial": at least one key fact of the reference is missing, and nothing contradicts it.
- "incorrect": contradicts the reference, misses its main fact, or says the answer was not found.
Return only JSON: {"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}"""

CONTEXT_PROMPT = """You check the retrieval step of a question-answering system over company annual reports.
You receive a QUESTION, its REFERENCE ANSWER and numbered PASSAGES (French or English).
Decide whether at least one passage contains the information needed to give the reference answer.
The wording and the language may differ; the facts must be the same.
Return only JSON: {"contains_answer": true | false, "passage": <number or null>}"""


def ask_json(client, model: str, system: str, user: str, attempts: int = 2) -> dict:
    """Ask for a JSON verdict. Thinking is turned off (a reasoning model may otherwise spend its answer on
    thinking and return no JSON), and the call is retried once when the reply contains no JSON."""
    kwargs = {"model": model, "format": "json", "options": {"temperature": 0, "num_ctx": 8192},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    content = ""
    for _ in range(attempts):
        try:
            response = client.chat(think=False, **kwargs)
        except Exception:  # model without a thinking mode (qwen2.5), or ollama package too old for `think`
            response = client.chat(**kwargs)
        content = re.sub(r"<think>.*?</think>", "", response.message.content or "", flags=re.S)
        match = re.search(r"\{.*\}", content, flags=re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"no JSON in the reply: {content[:80]!r}")


def judge(client, model: str, question: str, reference: str, answer: str) -> tuple[str, str]:
    user = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nSYSTEM ANSWER:\n{answer}"
    try:
        data = ask_json(client, model, JUDGE_PROMPT, user)
        verdict = str(data.get("verdict", "")).lower()
        if verdict not in {"correct", "partial", "incorrect"}:
            return "error", f"unexpected verdict: {verdict}"
        return verdict, str(data.get("reason", ""))
    except Exception as exc:
        return "error", f"judge failed: {exc}"


def context_has_answer(client, model: str, question: str, reference: str, hits: list[dict]) -> bool | None:
    """Used when neither the page nor the figures settle retrieval (qualitative questions)."""
    passages = "\n\n".join(f"[{i}] {h['text'][:1500]}" for i, h in enumerate(hits, start=1))
    user = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nPASSAGES:\n{passages}"
    try:
        return bool(ask_json(client, model, CONTEXT_PROMPT, user).get("contains_answer"))
    except Exception:
        return None


# --------------------------------------------------------------- grading ---
# Question categories (field "category" in questions.jsonl; "single" when absent):
#   single       one company, one fiscal year, named in full (the original reference set)
#   period       several fiscal years ("de 2019 à 2021", "depuis 2022")
#   multi        several companies, or all of them ("Air Canada et CGI", "quelles entreprises...")
#   paraphrase   company or year not written in full ("la BN", "l'an dernier", "pendant la pandémie")
#   comparative  a figure of year Y found in the report of year Y+1
MULTI_PASSAGE_CATEGORIES = {"period", "multi"}


def grade(q: dict, result: dict, judge_fn, context_fn=None) -> dict:
    answer = result["answer"]
    hits = result["sources"]
    out: dict = {"refused": is_refusal(answer)}
    out["answer_lang"] = detect_language(answer)
    out["lang_ok"] = (out["answer_lang"] == q["lang"]) if q.get("lang") and out["answer_lang"] != "unknown" else None

    if not q.get("answerable", True):
        out.update(verdict="correct" if out["refused"] else "incorrect",
                   method="refusal", reason="declined" if out["refused"] else "answered a question with no answer")
        out["retrieval_ok"] = None
        return out

    # Retrieval succeeds if: the right page (or an adjacent one: tables often span two pages) is
    # retrieved, or a passage contains every expected figure, or (qualitative questions, whose text is
    # often repeated in other pages and other years) an LLM confirms a passage contains the answer.
    expected = key_numbers(q["expected_answer"])
    # One expected page ("source" + "page"), or several ("sources": [{"source": ..., "page": ...}], for a
    # period or several companies): every one of them must be among the passages.
    expected_pages = q.get("sources") or (
        [{"source": q["source"], "page": q["page"]}] if q.get("source") and q.get("page") is not None else [])
    expected_pages = [p for p in expected_pages if p.get("source") and p.get("page") is not None]
    page_hit = bool(expected_pages) and all(
        any(h["metadata"].get("source") == p["source"] and abs((h["metadata"].get("page") or -99) - p["page"]) <= 1
            for h in hits)
        for p in expected_pages
    )
    figures_in_context = bool(expected) and any(
        match_figures(expected, numbers_in(h["text"]))[0] == len(expected) for h in hits
    )
    if not figures_in_context and expected and q.get("category") in MULTI_PASSAGE_CATEGORIES:
        # A period or a comparison spreads its figures over several passages (one per year or company):
        # retrieval succeeds when every expected figure is in at least one of them.
        all_numbers = [n for h in hits for n in numbers_in(h["text"])]
        figures_in_context = match_figures(expected, all_numbers)[0] == len(expected)
    context_ok = None
    if not (page_hit or figures_in_context) and context_fn and hits and q.get("type") != "numeric":
        context_ok = context_fn(q["question"], q["expected_answer"], hits)
    out.update(page_hit=page_hit, figures_in_context=figures_in_context, context_judge=context_ok,
               retrieval_ok=page_hit or figures_in_context or bool(context_ok))

    if q.get("type") == "numeric" and expected:
        found, scale_issue = match_figures(expected, numbers_in(answer))
        verdict = "correct" if found == len(expected) else "partial" if found else "incorrect"
        reason = f"{found}/{len(expected)} expected figures found"
        if scale_issue:
            reason += " (some only at another scale: unit error?)"
        out.update(verdict=verdict, method="figures", reason=reason, scale_issue=scale_issue)
    else:
        verdict, reason = judge_fn(q["question"], q["expected_answer"], answer)
        out.update(verdict=verdict, method="judge", reason=reason)
    return out


# --------------------------------------------------------------- summary ---
def score(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["grade"]["verdict"] in {"correct", "partial", "incorrect"}]
    n = len(graded)
    if not n:
        return {"n": 0}
    correct = sum(r["grade"]["verdict"] == "correct" for r in graded)
    partial = sum(r["grade"]["verdict"] == "partial" for r in graded)
    retrieval = [r["grade"]["retrieval_ok"] for r in graded if r["grade"].get("retrieval_ok") is not None]
    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "partial": round(partial / n, 3),
        "score": round((correct + 0.5 * partial) / n, 3),
        "retrieval_hit": round(sum(retrieval) / len(retrieval), 3) if retrieval else None,
    }


def _group(rows: list[dict], key) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return groups


def summarize(rows: list[dict], meta: dict) -> dict:
    ok_rows = [r for r in rows if r.get("error") is None]   # failed calls are reported apart
    answerable = [r for r in ok_rows if r["answerable"]]
    unanswerable = [r for r in ok_rows if not r["answerable"]]
    groups = {
        "by_type": lambda r: r["type"] or "?",
        "by_lang": lambda r: r["lang"] or "?",
        "by_company": lambda r: r["company"] or "?",
        "by_kind": lambda r: r["kind"] or "manual",
        "by_category": lambda r: r.get("category") or "single",
    }
    breakdown = {}
    for name, key in groups.items():
        buckets = defaultdict(list)
        for r in answerable:
            buckets[key(r)].append(r)
        breakdown[name] = {k: score(v) for k, v in sorted(buckets.items())}

    # Where do the errors come from? Retrieval failure vs generation failure.
    diagnosis = defaultdict(int)
    for r in answerable:
        g = r["grade"]
        if g["verdict"] not in {"correct", "partial", "incorrect"}:
            continue
        ok = g["verdict"] == "correct"
        diagnosis[("retrieval_ok" if g["retrieval_ok"] else "retrieval_miss") + ("+correct" if ok else "+wrong")] += 1

    lang_checked = [r for r in ok_rows if r["grade"].get("lang_ok") is not None]
    # A refusal is not a wrong answer: separate what the system answers from what it declines.
    graded = [r for r in answerable if r["grade"]["verdict"] in {"correct", "partial", "incorrect"}]
    attempted = [r for r in graded if not r["grade"]["refused"]]
    confident_errors = [r["id"] for r in attempted if r["grade"]["verdict"] == "incorrect"]
    return {
        **meta,
        "overall": score(answerable),
        "language_ok": round(sum(r["grade"]["lang_ok"] for r in lang_checked) / len(lang_checked), 3)
                       if lang_checked else None,
        "wrong_language": [f"{r['id']}:{r['grade']['answer_lang']}" for r in lang_checked if not r["grade"]["lang_ok"]],
        "unanswerable_declined": round(sum(r["grade"]["verdict"] == "correct" for r in unanswerable)
                                       / len(unanswerable), 3) if unanswerable else None,
        "declined_by_category": {
            c: f"{sum(r['grade']['verdict'] == 'correct' for r in rs)}/{len(rs)}"
            for c, rs in sorted(_group(unanswerable, lambda r: r.get("category") or "single").items())},
        "false_refusals": sum(r["grade"]["refused"] for r in answerable),
        "answered": len(attempted),
        "precision_when_answering": round(sum(r["grade"]["verdict"] == "correct" for r in attempted)
                                          / len(attempted), 3) if attempted else None,
        "confident_errors": len(confident_errors),
        "confident_error_ids": confident_errors,
        "scale_issues": sum(bool(r["grade"].get("scale_issue")) for r in answerable),
        "judge_errors": sum(r["grade"].get("method") == "judge" and r["grade"]["verdict"] == "error" for r in rows),
        "errors": sum(r.get("error") is not None for r in rows),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None,
        "diagnosis": dict(diagnosis),
        # Answers whose verification found a figure in no source (only when FINRAG_VERIFY is on).
        "flagged_answers": sum(bool((r.get("verification") or {}).get("unsupported")) for r in ok_rows),
        "revised_answers": sum(bool((r.get("verification") or {}).get("revised")) for r in ok_rows),
        **breakdown,
    }


def print_summary(s: dict) -> None:
    o = s["overall"]
    pct = lambda x: "  n/a" if x is None else f"{100 * x:5.1f}%"
    print(f"\n{'[regraded] ' if s.get('regraded') else ''}Model {s['model']} | prompt {s['prompt']} | num_ctx {s['num_ctx']} | think {s.get('think')} | k={s['k']} | "
          f"filters={s.get('filters') or ('oracle' if s.get('use_filters') else 'none')} | "
          f"search {s.get('retrieval') or 'vector'}{' + rerank' if s.get('rerank') else ''} | judge {s['judge_model']} | {s['n_questions']} questions")
    print(f"  Answer accuracy      {pct(o.get('accuracy'))}  (+{pct(o.get('partial')).strip()} partial, score {pct(o.get('score')).strip()})")
    print(f"  Retrieval hit@{str(s.get('k') or '?'):<2}      {pct(o.get('retrieval_hit'))}")
    print(f"  Unanswerable declined{pct(s['unanswerable_declined'])}"
          + (f"  ({', '.join(f'{c} {v}' for c, v in s['declined_by_category'].items())})"
             if len(s.get("declined_by_category") or {}) > 1 else ""))
    print(f"  False refusals       {s['false_refusals']:>5}")
    if s.get("answered") is not None:
        print(f"  Answered             {s['answered']:>5}  (precision when answering "
              f"{pct(s.get('precision_when_answering')).strip()}, confident errors {s.get('confident_errors')})")
    wrong = s.get("wrong_language") or []
    print(f"  Language respected   {pct(s.get('language_ok'))}" + (f"  (wrong: {', '.join(wrong)})" if wrong else ""))
    print(f"  Unit/scale errors    {s['scale_issues']:>5}")
    print(f"  Avg time / question  {s['avg_seconds'] if s['avg_seconds'] is not None else '?':>5}s")
    if s["errors"] or s["judge_errors"]:
        print(f"  Failed questions     {s['errors']:>5}  (judge errors: {s['judge_errors']}) -> see the .jsonl")
    print("  Error sources:", ", ".join(f"{k}={v}" for k, v in sorted(s["diagnosis"].items())))
    for name in ("by_type", "by_kind", "by_lang", "by_company", "by_category"):
        print(f"  {name}:")
        for key, v in (s.get(name) or {}).items():
            print(f"    {key:<16} n={v['n']:<3} accuracy {pct(v.get('accuracy'))}  retrieval {pct(v.get('retrieval_hit'))}")


# --------------------------------------------------------------- regrade ---
def regrade(run_file: Path, judge_model: str | None) -> None:
    """Grade the answers saved in a run file again (after a fix in the grader), without re-running the RAG.
    Retrieval results are kept as they were: only the answer grading and the language check are redone."""
    import ollama
    from .config import OLLAMA_HOST

    summary_file = run_file.with_suffix(".summary.json")
    old_meta = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    meta_keys = ("model", "judge_model", "prompt", "num_ctx", "think", "k", "use_filters", "filters", "retrieval",
                 "rerank", "year_window", "parser", "verify", "date")
    meta = {key: old_meta.get(key) for key in meta_keys}
    meta["judge_model"] = judge_model or meta.get("judge_model") or "qwen2.5:7b"
    client = ollama.Client(host=OLLAMA_HOST)
    judge_fn = lambda q, ref, ans: judge(client, meta["judge_model"], q, ref, ans)

    rows = [json.loads(line) for line in run_file.open(encoding="utf-8") if line.strip()]
    for row in rows:
        if row.get("error"):
            continue
        old = row["grade"]
        new = grade(row, {"answer": row["answer"], "sources": []}, judge_fn)
        for key in ("page_hit", "figures_in_context", "context_judge", "retrieval_ok"):
            if key in old:
                new[key] = old[key]
        row["grade"] = new

    out_file = run_file.with_name(run_file.stem.replace("_regraded", "") + "_regraded.jsonl")
    with out_file.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta.update(n_questions=len(rows), run_file=out_file.name, regraded=True)
    summary = summarize(rows, meta)
    out_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print_summary(summary)
    print(f"\nDetails: {out_file}")


# ------------------------------------------------------------------ main ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinRAG on eval/questions.jsonl.")
    parser.add_argument("-k", type=int, default=None, help="passages retrieved per question (default: TOP_K)")
    parser.add_argument("--model", help="generation model (default: FINRAG_LLM_MODEL / config)")
    parser.add_argument("--prompt", help="system prompt version: v1 or v2 (default: FINRAG_PROMPT / config)")
    parser.add_argument("--num-ctx", type=int, help="Ollama context window; 0 = Ollama default")
    parser.add_argument("--think", choices=["on", "off"], help="reasoning mode, for models that have one")
    parser.add_argument("--judge-model", help="model used to grade qualitative answers (default: generation model)")
    parser.add_argument("--filters", choices=["none", "oracle", "auto"], default="none",
                        help="none: search the whole corpus; oracle: each question's company/year from the "
                             "evaluation set (upper bound); auto: detected in the question, as in production")
    parser.add_argument("--use-filters", action="store_true", help="same as --filters oracle")
    parser.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"],
                        help="search method (default: FINRAG_RETRIEVAL, hybrid)")
    parser.add_argument("--rerank", choices=["on", "off"], help="rerank the candidates (default: FINRAG_RERANK, off)")
    parser.add_argument("--parser", choices=["regex", "llm", "hybrid"],
                        help="auto filters: how companies and years are found (default: FINRAG_QUERY_PARSER, hybrid)")
    parser.add_argument("--verify", choices=["off", "flag", "revise"],
                        help="check the figures of each answer (default: FINRAG_VERIFY, off)")
    parser.add_argument("--year-window", type=int, choices=[0, 1],
                        help="auto filters: 1 = also search the next year's report (default: FINRAG_YEAR_WINDOW, 1)")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--ids", nargs="*", help="evaluate only these question ids")
    parser.add_argument("--category", nargs="*", metavar="NAME",
                        help="evaluate only these categories (single, period, multi, paraphrase, comparative)")
    parser.add_argument("--regrade", type=Path, help="grade the saved answers of a previous run again")
    args = parser.parse_args()
    if args.regrade:
        return regrade(args.regrade, args.judge_model)

    # Must be set before finrag.config is imported.
    if args.model:
        os.environ["FINRAG_LLM_MODEL"] = args.model
    if args.prompt:
        os.environ["FINRAG_PROMPT"] = args.prompt
    if args.num_ctx is not None:
        os.environ["FINRAG_NUM_CTX"] = str(args.num_ctx)
    if args.think:
        os.environ["FINRAG_THINK"] = args.think
    if args.retrieval:
        os.environ["FINRAG_RETRIEVAL"] = args.retrieval
    if args.rerank:
        os.environ["FINRAG_RERANK"] = args.rerank
    if args.year_window is not None:
        os.environ["FINRAG_YEAR_WINDOW"] = str(args.year_window)
    if args.parser:
        os.environ["FINRAG_QUERY_PARSER"] = args.parser
    if args.verify:
        os.environ["FINRAG_VERIFY"] = args.verify
    import ollama
    from tqdm import tqdm
    from .config import LLM_MODEL, NUM_CTX, OLLAMA_HOST, PROMPT_VERSION, THINK, TOP_K
    from .generate import VERIFY, YEAR_WINDOW, answer as rag_answer
    from .query_plan import PARSER
    from .retrieve import RERANK, RETRIEVAL

    k = args.k or TOP_K
    mode = "oracle" if args.use_filters else args.filters
    judge_model = args.judge_model or LLM_MODEL
    client = ollama.Client(host=OLLAMA_HOST)
    judge_fn = lambda q, ref, ans: judge(client, judge_model, q, ref, ans)
    context_fn = lambda q, ref, hits: context_has_answer(client, judge_model, q, ref, hits)

    questions = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8") if line.strip()]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.category:
        questions = [q for q in questions if (q.get("category") or "single") in set(args.category)]
    if args.limit:
        questions = questions[: args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    tag = re.sub(r"[^\w.-]", "-", LLM_MODEL)
    run_file = RESULTS_DIR / (f"{stamp}_{tag}_{PROMPT_VERSION}_ctx{NUM_CTX}_k{k}"
                              f"{'_think-' + THINK if THINK else ''}"
                              f"{'' if mode == 'none' else '_filters' if mode == 'oracle' else '_autofilters'}"
                              f"{f'_yw{YEAR_WINDOW}' if mode == 'auto' and YEAR_WINDOW else ''}"
                              f"{f'_{PARSER}parse' if mode == 'auto' and PARSER != 'regex' else ''}"
                              f"{f'_verify-{VERIFY}' if VERIFY != 'off' else ''}"
                              f"{'' if RETRIEVAL == 'vector' else '_' + RETRIEVAL}{'_rerank' if RERANK else ''}.jsonl")

    rows = []
    with run_file.open("w", encoding="utf-8") as out:
        for q in tqdm(questions, desc="Evaluating"):
            start = time.perf_counter()
            row = {k_: q.get(k_) for k_ in ("id", "question", "expected_answer", "type", "lang",
                                            "company", "year", "source", "page", "sources", "kind")}
            row["category"] = q.get("category") or "single"
            row["answerable"] = q.get("answerable", True)
            try:
                filters = {"company": q.get("company"), "year": q.get("year")} if mode == "oracle" else {}
                result = rag_answer(q["question"], k=k, auto_filters=(mode == "auto"), **filters)
                row["error"] = None
            except Exception as exc:
                result = {"answer": "", "sources": []}
                row["error"] = str(exc)
            row["seconds"] = round(time.perf_counter() - start, 2)
            row["answer"] = result["answer"]
            row["filters"] = result.get("filters")
            row["verification"] = result.get("verification")
            row["retrieved"] = [f"{h['metadata'].get('source')} p.{h['metadata'].get('page')}" for h in result["sources"]]
            row["grade"] = grade(q, result, judge_fn, context_fn) if row["error"] is None else \
                {"verdict": "error", "reason": row["error"], "refused": False, "retrieval_ok": None}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()  # partial results survive a crash

    meta = {"model": LLM_MODEL, "judge_model": judge_model, "prompt": PROMPT_VERSION, "num_ctx": NUM_CTX, "think": THINK or "default",
            "k": k, "use_filters": mode != "none", "filters": mode, "retrieval": RETRIEVAL,
            "rerank": RERANK, "year_window": YEAR_WINDOW if mode == "auto" else 0,
            "parser": PARSER if mode == "auto" else None, "verify": VERIFY,
            "n_questions": len(rows), "run_file": run_file.name, "date": stamp}
    summary = summarize(rows, meta)
    run_file.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print_summary(summary)
    print(f"\nDetails: {run_file}")


if __name__ == "__main__":
    main()
