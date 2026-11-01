"""Answer a question from retrieved chunks, with numbered citations.

Two settings can be changed without editing code, to compare them with the evaluation:
  FINRAG_PROMPT   v1 = original prompt; v2 to v5 = variants tested in the evaluation (see PROMPTS)
  FINRAG_NUM_CTX  context window given to Ollama (0 = Ollama's default, which may truncate the prompt)
  FINRAG_THINK    on / off: reasoning mode of models that have one (empty = Ollama's default)
  FINRAG_AUTO_FILTERS  on (default) / off: search only the company and fiscal year named in the question
  FINRAG_YEAR_WINDOW   1 (default): also search the next report (comparatives); 0: only that year's report
  FINRAG_PERIOD_YEAR_WINDOW  1 (default): in a period question, each year is also searched in the next report
  FINRAG_QUERY_PARSER  hybrid (default) / regex / llm: how companies and years are found (see query_plan.py)
  FINRAG_MAX_SLOTS     7 (default): above this many (company, year) pairs, answer company by company
  FINRAG_VERIFY        off (default) / flag / revise: check the figures of the answer against its sources
                       (verify.py); flag adds a warning, revise asks the model to correct the answer once
"""
import os
import re

from .config import LLM_MODEL, NUM_CTX, PROMPT_VERSION, THINK, TOP_K
from .index import get_ollama
from .retrieve import retrieve, retrieve_slots

AUTO_FILTERS = os.getenv("FINRAG_AUTO_FILTERS", "on").strip().lower() != "off"
# Figures of year Y are in the report of year Y and again, as comparatives (sometimes restated), in the
# report of year Y+1. With FINRAG_YEAR_WINDOW=1, a detected year Y searches both reports, unless the
# question names a report ("selon le rapport annuel 2023"). 0 = only the report of year Y.
# Default 1: best evaluation (eval 20261029_2217: 98.3 % retrieval, 3 confident errors, 90.6 % precision).
YEAR_WINDOW = int(os.getenv("FINRAG_YEAR_WINDOW", "1"))
# Period questions ("de 2020 à 2024"): each year is searched in its own report and in the next one(s).
PERIOD_YEAR_WINDOW = int(os.getenv("FINRAG_PERIOD_YEAR_WINDOW", "1"))
VERIFY = os.getenv("FINRAG_VERIFY", "off").strip().lower()

BASE_PROMPT = """You are a careful financial analyst assistant.
Answer ONLY with information found in the numbered sources provided.
Rules:
- Cite every fact with its source number, e.g. [S2]. Never cite a source that does not contain the fact.
- Report figures exactly as written, with their units (thousands, millions, %) and the fiscal year.
- Fiscal years may not match calendar years; say so when it matters.
- If the sources do not contain the answer, say clearly that you could not find it. Never guess.
- Answer in the same language as the question, concisely."""

# Rules added after the first evaluation; each one targets an error category observed in eval/results.
RULE_ENTITY = """
- Check the company and the fiscal year in each source header. Use only sources about the company and
  the period asked. If no source concerns the company asked about, say you could not find it."""
RULE_METRIC = """
- Answer exactly the metric asked. Do not substitute an adjusted (non-GAAP) figure for a reported one,
  a consolidated figure for a segment figure, or a closing balance for an opening balance."""
RULE_TABLES = """
- In tables, check the row label and the column header (year, quarter, segment) before taking a figure.
- If several figures could match, give the one that fits the question and say which line it comes from."""
RULE_LANGUAGE = """
- Write the whole answer in the language of the question (French or English), even when the sources are
  in another language. Never answer in any other language."""
# Added after eval/errors.txt: with the right passage in hand, the model still refused when the source used
# another language or another wording (q061: "Particuliers et Entreprises" for "Personal and Commercial").
RULE_MATCHING = """
- Sources may be in another language than the question: segment and line-item names are translated
  (e.g. "Particuliers et Entreprises" = "Personal and Commercial"). Match them by meaning.
- If a source gives the figure asked under a slightly different wording, or the facts from which the answer
  follows directly, answer with it and quote the source's wording. Say that you could not find the answer
  only when no source contains it."""

# Added only for questions about several fiscal years (the other questions keep their prompt unchanged).
RULE_PERIOD = """
- The question covers several fiscal years. Give the figure for each year asked, in chronological order
  (a short list or table), each with its citation. Read each year in the column of that year.
- A report also shows the previous year as a comparative: use it for a year whose own report is not
  among the sources, and say so. If the two reports differ (restated figure), give the later one and say
  it was restated. If a year is missing from all sources, say which one; never estimate it."""

PROMPTS = {
    "v1": BASE_PROMPT,                                            # original prompt
    "v2": BASE_PROMPT + RULE_ENTITY + RULE_METRIC + RULE_TABLES,  # all rules from the first error analysis
    "v3": BASE_PROMPT + RULE_ENTITY + RULE_METRIC,                # v2 without the table rules (ablation)
    "v4": BASE_PROMPT + RULE_ENTITY + RULE_METRIC + RULE_LANGUAGE,  # v3 + language rule
    "v5": BASE_PROMPT + RULE_ENTITY + RULE_METRIC + RULE_LANGUAGE + RULE_MATCHING,  # v4 + fewer false refusals
}


def system_prompt(version: str = PROMPT_VERSION) -> str:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version '{version}'. Choose among: {', '.join(PROMPTS)}")
    return PROMPTS[version]


def format_sources(hits: list[dict], prefix: str = "S") -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        m = hit["metadata"]
        blocks.append(f"[{prefix}{i}] {m['company']}, fiscal {m['year']}, page {m['page']}\n{hit['text']}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ plan ---
# More (company, year) slots than this: the passages per slot get too few, so the question is answered
# company by company, then the answers are combined (decomposition).
MAX_SLOTS = int(os.getenv("FINRAG_MAX_SLOTS", "7"))

RULE_COMPANIES = """
- The question covers several companies. Answer for each company asked (a short table or list), each
  figure with its citation. Use for each company only the sources about that company. If a company has no
  source for what is asked, say so.
- When you compare or rank them: their fiscal years end on different dates, and some report in US dollars
  (Shopify, Couche-Tard) while others report in Canadian dollars. Say so; never rank amounts in different
  currencies as if they were comparable (growth rates and ratios are). Say which fiscal year each figure is for."""

SYNTHESIS_PROMPT = """You combine answers about several companies into one answer to the question.
Rules:
- Use only the facts in the answers below; keep every figure exactly as written, with its unit, its fiscal
  year and its citation (e.g. [cgi S2]).
- Present one line or table row per company. Say when a company's answer could not find the figure.
- Fiscal years end on different dates depending on the company: say so when you compare them.
- Answer in the same language as the question, concisely."""


def make_plan(question: str, company: str | None = None, year=None, auto_filters: bool | None = None,
              parser: str | None = None) -> dict:
    """The search plan: given filters (evaluation oracle, API) or detected in the question."""
    if company or year:
        return {"companies": [company] if company else [], "all": False, "year": year, "years": None,
                "report": True, "parser": "given"}       # given filters are used as is: no widening
    if not (AUTO_FILTERS if auto_filters is None else auto_filters):
        return {"companies": [], "all": False, "year": None, "years": None, "report": False, "parser": "none"}
    from .query_plan import plan_question
    return plan_question(question, parser)


def plan_slots(plan: dict) -> list[tuple[str | None, int | None]]:
    companies = plan.get("companies") or [None]
    years = plan.get("years") or [plan.get("year")]
    return [(c, y) for c in companies for y in years]


def retrieve_for_plan(question: str, plan: dict, k: int = TOP_K) -> tuple[list[dict], str]:
    """Passages for a plan, and the mode used: "single", "slots", "decompose" (no passages) or "relaxed"."""
    companies, years = plan.get("companies") or [], plan.get("years")
    if len(companies) <= 1 and not years:
        company = companies[0] if companies else None
        year = plan.get("year")
        if YEAR_WINDOW and isinstance(year, int) and not plan.get("report"):
            year = list(range(year, year + YEAR_WINDOW + 1))
        hits = retrieve(question, k=k, company=company, year=year)
        if len(hits) < k and (company or year):
            # Too narrow (a report missing from the index...): search the whole corpus rather than answer from less.
            return retrieve(question, k=k), "relaxed"
        return hits, "single"
    if len(companies) > 1 and not years and plan.get("year") is None:
        # Comparing companies without a year: take each one's latest report, not a mix of years.
        from .query_plan import LATEST_YEAR
        plan["year"], plan["default_year"] = LATEST_YEAR, True
    slots = plan_slots(plan)
    if len(companies) > 1 and len(slots) > MAX_SLOTS:
        return [], "decompose"
    window = 0 if plan.get("report") else (PERIOD_YEAR_WINDOW if years else YEAR_WINDOW)
    return retrieve_slots(question, slots, window=window, k=k), "slots"


# ---------------------------------------------------------------- answer ---
def chat(system: str, user: str) -> str:
    options = {"temperature": 0}
    if NUM_CTX:
        options["num_ctx"] = NUM_CTX
    kwargs = {}
    if THINK in {"on", "off"}:
        kwargs["think"] = THINK == "on"
    try:
        response = get_ollama().chat(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options=options,
            **kwargs,
        )
    except TypeError as exc:  # ollama Python package too old to know the `think` argument
        raise RuntimeError("FINRAG_THINK needs a recent ollama package: pip install -U ollama") from exc
    # Reasoning models (e.g. Qwen 3.x) may write their thinking between <think> tags: keep only the answer.
    return re.sub(r"<think>.*?</think>", "", response.message.content, flags=re.S).strip()


def answer(question: str, k: int = TOP_K, company: str | None = None, year=None,
           auto_filters: bool | None = None, plan: dict | None = None, parser: str | None = None) -> dict:
    """Answer from the best passages. Without an explicit company or year, the companies and fiscal years
    are detected in the question (see query_plan.py) so that only the right reports are searched."""
    plan = dict(plan or make_plan(question, company, year, auto_filters, parser))
    if plan.get("outside") and not plan.get("companies"):
        return {"answer": outside_corpus_answer(question), "sources": [], "filters": {**plan, "mode": "outside"}}
    hits, mode = retrieve_for_plan(question, plan, k)
    plan = {**plan, "mode": mode}
    if mode == "decompose":
        return answer_by_company(question, plan, k)
    if not hits:
        return {"answer": "No matching documents in the index.", "sources": [], "filters": plan}
    rules = (RULE_PERIOD if plan.get("years") else "") + (RULE_COMPANIES if len(plan.get("companies") or []) > 1 else "")
    if plan.get("default_year"):
        rules += (f"\n- The question gives no year: the sources are from each company's latest annual report "
                  f"(fiscal {plan['year']}). Say so at the start of the answer.")
    system = system_prompt() + rules
    user = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    content = chat(system, user)
    result = {"answer": content, "sources": hits, "filters": plan}
    if VERIFY in {"flag", "revise"}:
        result.update(verify_answer(question, content, hits, system, user))
    return result


REVISION_PROMPT = """A verification of your answer against the sources found these problems:
{problems}

Rewrite the answer. Keep only figures that appear in the sources (or that you compute from them, saying
so), cite for each figure the source that contains it, and say that you could not find a figure rather
than give one that is not in the sources. Reply with the corrected answer only."""


def verify_answer(question: str, content: str, hits: list[dict], system: str, user: str) -> dict:
    """Check the figures of an answer; with FINRAG_VERIFY=revise, ask for one correction if some are wrong."""
    from .retrieve import detect_lang
    from .verify import feedback, verify
    report = verify(content, hits, question)
    revised = False
    if not report["ok"] or report["other_source"]:
        if VERIFY == "revise":
            fixed = chat(system, f"{user}\n\nYour answer:\n{content}\n\n"
                                 + REVISION_PROMPT.format(problems=feedback(report)))
            fixed_report = verify(fixed, hits, question)
            if len(fixed_report["unsupported"]) <= len(report["unsupported"]):
                content, report, revised = fixed, fixed_report, True
        if not report["ok"]:
            figures = ", ".join(report["unsupported"])
            # Worded so that the evaluation does not take the warning for a refusal ("not found"...).
            content += ("\n\n⚠ To check: these figures appear in none of the sources: " if detect_lang(question) == "en"
                        else "\n\n⚠ À vérifier : ces chiffres n'apparaissent dans aucune des sources : ") + figures
    return {"answer": content, "verification": {"ok": report["ok"], "n_figures": report["n_figures"],
                                                 "unsupported": report["unsupported"],
                                                 "other_source": report["other_source"], "revised": revised}}


CORPUS_NAMES = "Air Canada, Banque Nationale du Canada, CGI, Alimentation Couche-Tard, Mouvement Desjardins, Shopify"


def outside_corpus_answer(question: str) -> str:
    """Fixed refusal for a company outside the corpus, in the language of the question."""
    from .retrieve import detect_lang
    if detect_lang(question) == "en":
        return ("I could not find the answer: this company is not covered by the annual reports I have. "
                f"They cover {CORPUS_NAMES.replace('Banque Nationale du Canada', 'National Bank of Canada')} "
                "(fiscal years 2017 to 2025).")
    return ("Je n'ai pas trouvé la réponse : cette entreprise ne fait pas partie des rapports annuels dont je "
            f"dispose. Ils couvrent {CORPUS_NAMES} (exercices 2017 à 2025).")


def answer_by_company(question: str, plan: dict, k: int = TOP_K) -> dict:
    """Too many (company, year) slots for one prompt: answer for each company, then combine."""
    parts, sources = [], []
    for company in plan["companies"]:
        sub = answer(question, k=k, plan={**plan, "companies": [company], "all": False})
        # Citations become [company S2]: the per-company source numbers would collide otherwise.
        text = re.sub(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]",
                      lambda m: "[" + ", ".join(f"{company} {s.strip()}" for s in re.split(r"[,;]", m.group(1))) + "]",
                      sub["answer"])
        parts.append(f"## {company}\n{text}")
        sources += [{**h, "for_company": company} for h in sub["sources"]]
    content = chat(SYNTHESIS_PROMPT, "Answers by company:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}")
    return {"answer": content, "sources": sources, "filters": plan, "parts": parts}
