"""Turn a question into a search plan: which companies, which fiscal years, which report.

    {"companies": ["aircanada", "cgi"], "all": False, "year": None, "years": [2021, 2022], "report": False,
     "parser": "regex"}

Three parsers (FINRAG_QUERY_PARSER, or the `parser` argument):
  regex   query_filters.py: instant and predictable, but only knows the wordings it was written for.
  llm     the LLM reads the question and fills a JSON schema (companies, years, named report). It
          understands paraphrases ("l'an dernier", "the pandemic years", "la BN", "chaque entreprise").
  hybrid  regex first; the LLM only when the regex found no company, or no period although the question
          talks about time. Most questions cost nothing extra.

The LLM's answer is never trusted as is: a company must be in the corpus AND be designated in the question
(by one of its loose aliases, or by a question about all companies); years must be in the corpus. The
check matters for trap questions: asked about WestJet, an LLM may "helpfully" pick Air Canada, and the
search would then return Air Canada figures for a WestJet question.

    python -m finrag.query_plan "Quel était le résultat net de la BN l'an dernier ?"
"""
import json
import os
import re
import sys
from datetime import date

from .query_filters import (COMPANY_PATTERNS, YEARS, _norm, detect_companies, detect_period, detect_year,
                            mentions_all_companies, names_report)

# hybrid: chosen with eval/bench_parser.txt (98/98 companies, no wrong company; 0.9 s per question on average)
PARSER = os.getenv("FINRAG_QUERY_PARSER", "hybrid").strip().lower()
LATEST_YEAR = max(YEARS)

COMPANY_INFO = {
    "aircanada": "Air Canada (airline; fiscal year ends December 31)",
    "banquenationale": "National Bank of Canada / Banque Nationale du Canada, BNC, BN (bank; fiscal year ends October 31)",
    "cgi": "CGI Inc. (IT services; fiscal year ends September 30)",
    "couchetard": "Alimentation Couche-Tard, Circle K (convenience stores; fiscal year ends in late April)",
    "desjardins": "Mouvement Desjardins / Desjardins Group (cooperative financial group; fiscal year ends December 31)",
    "shopify": "Shopify Inc. (e-commerce platform; fiscal year ends December 31)",
}
# Looser than query_filters.COMPANY_PATTERNS: used only to confirm a company chosen by the LLM.
LOOSE_ALIASES = {
    "aircanada": [r"\bair[- ]?canada\b", r"\ba[ée]roplan\b", r"\bac\b"],
    "banquenationale": [r"\bbanque[- ]?nationale\b", r"\bnational[- ]?bank\b", r"\bbnc\b", r"\bbn\b", r"\bnbc\b"],
    "cgi": [r"\bcgi\b"],
    "couchetard": [r"\bcouche[- ]?tard\b", r"\bcircle ?k\b", r"\balimentation\b", r"\batd\b"],
    "desjardins": [r"\bdesjardins\b", r"\b(?:le|du|au) mouvement\b", r"\bcaisses? populaires?\b"],
    "shopify": [r"\bshopify\b"],
}
# Every strict pattern is also a loose alias: the LLM's choice is never rejected for a name the regex knows.
for _company, _patterns in COMPANY_PATTERNS.items():
    LOOSE_ALIASES[_company] = list(dict.fromkeys(LOOSE_ALIASES.get(_company, []) + _patterns))
ALL_WORDS_RE = re.compile(r"\b(?:toutes|tous|chaque|chacune|laquelle|lesquelles|parmi|each|every|all|which|"
                          r"among|quelle|quelles)\b")
REPORT_WORDS_RE = re.compile(r"\b(?:rapport|report)\b")

SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {"type": "array", "items": {"type": "string", "enum": [*COMPANY_INFO, "other"]}},
        "all_companies": {"type": "boolean"},
        "years": {"type": "array", "items": {"type": "integer"}},
        "report_year": {"type": ["integer", "null"]},
    },
    "required": ["companies", "all_companies", "years", "report_year"],
}

PROMPT = """You turn a question about annual reports into search filters. Reply with JSON only.

Companies in the corpus:
{companies}
Annual reports available: fiscal years {first} to {latest}. Today is {today}; the latest report is fiscal {latest}.

Fields:
- companies: the corpus companies the question is about, named or clearly designated (abbreviation, brand,
  "le Mouvement"...). Use "other" for a company that is not in the list. NEVER replace a company outside
  the corpus by a similar corpus company (a question about WestJet is not about Air Canada).
- all_companies: true only if the question asks about all / each / every company, or which company.
- years: the fiscal years the answer needs. One year -> [year]. A period -> every year of it ("2019 to 2021"
  -> [2019, 2020, 2021]; "the last three years" -> the last three fiscal years up to {latest}). Relative
  expressions are relative to the latest report ("last year", "l'an dernier", "le dernier exercice" ->
  [{latest}]). The most recent / latest / up-to-date figures ("les plus récents", "les derniers chiffres",
  "à jour") -> [{latest}]; the oldest / first ones ("les plus anciens", "le premier rapport") -> [{first}].
  An event -> its fiscal year(s) (the COVID-19 pandemic: 2020 and 2021). A comparison of a year
  with the previous one ("2022 compared with 2021") -> [the later year]. No time mentioned -> [], even when
  the question is in the past tense ("Comment la marge a-t-elle évolué ?") or names a quarter without a year
  ("in Q2"): never guess the latest year when the question does not ask for the latest figures.
- report_year: the year of a report the question explicitly names ("selon le rapport annuel 2023" -> 2023),
  otherwise null.

Question: {question}"""


def _has_alias(company: str, text: str) -> bool:
    return any(re.search(p, text) for p in LOOSE_ALIASES.get(company, []))


def _time_fields(report: bool, year: int | None, years: list[int] | None) -> dict:
    return {"year": year, "years": years, "report": report}


def _normalize_years(years: list[int]) -> tuple[int | None, list[int] | None]:
    """One year, or a period. The prompt already turns "2022 compared with 2021" into [2022]: several years
    from the LLM are a period (the pandemic years 2020 and 2021)."""
    years = sorted({y for y in years if y in YEARS})
    if not years or len(years) == len(YEARS):
        # Every year of the corpus ("in which year did...?"): no year filter, rather than a 9-year period
        # with one passage per year.
        return None, None
    if len(years) == 1:
        return years[0], None
    return None, years


def regex_plan(question: str) -> dict:
    companies = detect_companies(question)
    all_companies = mentions_all_companies(question, companies)
    plan = {"companies": list(COMPANY_PATTERNS) if all_companies else companies, "all": all_companies,
            "parser": "regex"}
    if not plan["companies"]:            # no corpus company: no year filter either (often a trap question)
        return {**plan, **_time_fields(False, None, None)}
    if names_report(question):
        return {**plan, **_time_fields(True, detect_year(question), None)}
    if period := detect_period(question):
        return {**plan, **_time_fields(False, None, period)}
    return {**plan, **_time_fields(False, detect_year(question), None)}


def ask_llm(question: str) -> dict | None:
    """Raw JSON from the LLM, or None if the call or the parsing fails."""
    from .config import LLM_MODEL, NUM_CTX
    from .index import get_ollama
    prompt = PROMPT.format(companies="\n".join(f"- {c}: {d}" for c, d in COMPANY_INFO.items()),
                           first=min(YEARS), latest=LATEST_YEAR, today=date.today().isoformat(), question=question)
    kwargs = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
              # Same num_ctx as the answer: with another value, Ollama reloads the model at every switch.
              "options": {"temperature": 0, **({"num_ctx": NUM_CTX} if NUM_CTX else {})}}
    for fmt in (SCHEMA, "json"):     # a JSON schema needs a recent Ollama; plain JSON mode otherwise
        try:
            try:
                response = get_ollama().chat(format=fmt, think=False, **kwargs)
            except TypeError:  # ollama package too old for `think`
                response = get_ollama().chat(format=fmt, **kwargs)
            content = re.sub(r"<think>.*?</think>", "", response.message.content or "", flags=re.S)
            match = re.search(r"\{.*\}", content, flags=re.S)
            if match:
                return json.loads(match.group(0))
        except Exception:
            continue
    return None


def llm_plan(question: str) -> dict | None:
    data = ask_llm(question)
    if not isinstance(data, dict):
        return None
    text = _norm(question)
    proposed = [c for c in data.get("companies") or [] if c in COMPANY_INFO]
    companies = [c for c in proposed if _has_alias(c, text)]          # designated in the question
    # "all" needs an all-companies wording; a "which" alone does not count when a company is named.
    all_companies = bool(data.get("all_companies")) and bool(ALL_WORDS_RE.search(text)) and (
        not companies or mentions_all_companies(question, companies))
    if all_companies:
        companies = list(COMPANY_INFO)
    plan = {"companies": companies, "all": all_companies, "parser": "llm",
            "rejected": [c for c in proposed if c not in companies] or None,
            # Only about companies outside the corpus (RBC, WestJet...): answered without any search, since
            # the closest passages would be about a corpus company (National Bank for "Royal Bank of Canada").
            "outside": "other" in (data.get("companies") or []) and not companies and not all_companies}
    if not companies:
        return {**plan, **_time_fields(False, None, None)}
    report_year = data.get("report_year")
    if isinstance(report_year, int) and report_year in YEARS and REPORT_WORDS_RE.search(text):
        return {**plan, **_time_fields(True, report_year, None)}
    year, years = _normalize_years([y for y in data.get("years") or [] if isinstance(y, int)])
    return {**plan, **_time_fields(False, year, years)}


# A year given relative to another one ("l'exercice qui a suivi 2019", "the year before 2021"): the regex
# would take the year written, not the year meant.
RELATIVE_YEAR_RE = re.compile(
    r"\b(?:suivi|suivant|suivante|suivre|pr[ée]c[ée]d(?:ant|ent|ente|é)|avant|apr[eè]s|after|following|"
    r"before|prior to|preceding)\s+(?:l'exercice\s+|l'ann[ée]e\s+|fiscal\s+|the year\s+)?20\d\d")


def hybrid_plan(question: str) -> dict:
    """Regex first; the LLM only to fill what the regex could not find.

    The LLM is called when the regex found no company; or a company alias the regex does not know ("la BN"
    next to "Air Canada"); or no year in digits (the LLM reads "les plus récents", "le premier exercice",
    "pendant la pandémie" without a list of wordings, and returns no year when the question has none); or a
    year given relative to another one. Companies are merged (regex + validated LLM); the LLM's years replace the regex's only when
    the regex found none or the year is relative.
    """
    plan = regex_plan(question)
    text = _norm(question)
    no_company = not plan["companies"]
    unmatched_alias = any(_has_alias(c, text) for c in COMPANY_INFO if c not in plan["companies"])
    # Every question without a year in digits: listing the time words ("récent", "récents", "dernier",
    # "à jour", "plus ancien"...) always misses one, and a missed one means a search across nine years.
    no_time = plan["year"] is None and plan["years"] is None
    relative_year = bool(RELATIVE_YEAR_RE.search(text))
    if not (no_company or unmatched_alias or no_time or relative_year):
        return plan
    llm = llm_plan(question)
    if llm is None:
        return plan
    merged = dict(plan)
    # Union of the companies, in corpus order: every LLM company was already checked against the question.
    companies = set(plan["companies"]) | set(llm["companies"])
    merged.update(companies=[c for c in COMPANY_INFO if c in companies], all=plan["all"] or llm["all"])
    regex_time_found = plan["year"] is not None or plan["years"] is not None
    if merged["companies"] and (not regex_time_found or relative_year or not plan["companies"]):
        merged.update(year=llm["year"], years=llm["years"], report=llm["report"])
    merged["parser"] = "hybrid+llm"
    merged["rejected"] = llm.get("rejected")
    merged["outside"] = bool(llm.get("outside")) and not merged["companies"]
    return merged


def plan_question(question: str, parser: str | None = None) -> dict:
    parser = (parser or PARSER).lower()
    if parser == "regex":
        return regex_plan(question)
    if parser == "llm":
        return llm_plan(question) or regex_plan(question)
    if parser == "hybrid":
        return hybrid_plan(question)
    raise ValueError(f"Unknown query parser '{parser}': regex, llm or hybrid")


if __name__ == "__main__":
    for q in sys.argv[1:]:
        for name in ("regex", "hybrid", "llm"):
            print(f"{name:<7}", plan_question(q, name))
        print()
