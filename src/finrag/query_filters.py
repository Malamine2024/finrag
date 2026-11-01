"""Detect the company and the fiscal year a question is about, to filter the search.

Annual reports repeat the same paragraphs from one year to the next, and the search cannot tell those
copies apart: it often returns the right page of the wrong year (see eval/diff_docling.txt). With the
company and the year of each question given as filters, accuracy rose from 53 % to 69 %. This module finds
them in the question itself, with simple rules (no LLM: fast, predictable, testable).

Questions about a period ("de 2020 à 2024", "between 2019 and 2022", "since 2020", "over the last three
years", "in 2019, 2020 and 2021") get a list of years instead: each year is then searched on its own (see
retrieve.retrieve_period), so that every year asked is represented among the passages.

    python -m finrag.query_filters      # compare the detection with the metadata of eval/questions.jsonl
    python -m finrag.query_filters --test "How did Shopify's revenue change from 2019 to 2022?"
"""
import json
import re
import unicodedata

from .config import ROOT

# Names as they appear in questions, in French and English. Order matters only for readability: every
# pattern is searched. Competitors and other banks (WestJet, RBC...) must NOT match: questions about them
# are unanswerable traps, and filtering them onto a corpus company would invite a substitution.
# Multi-word names also match written together or with a hyphen ("aircanada", "Air-Canada", "banquenationale").
COMPANY_PATTERNS = {
    "aircanada": [r"\bair[- ]?canada\b", r"\ba[ée]roplan\b"],
    "banquenationale": [r"\bbanque[- ]?nationale\b", r"\bnational[- ]?bank\b", r"\bbnc\b"],
    "cgi": [r"\bcgi\b"],
    "couchetard": [r"\bcouche[- ]?tard\b", r"\bcircle k\b"],
    "desjardins": [r"\bdesjardins\b"],
    "shopify": [r"\bshopify\b"],
}
YEARS = range(2017, 2026)        # fiscal years in the corpus

# "rapport annuel 2023", "2023 annual report": the report is named, take it as is.
REPORT_RE = re.compile(r"(?:rapport annuel|annual report)\s+(?:de\s+)?(20\d\d)|(20\d\d)\s+annual report")
YEAR_RE = re.compile(r"(?<![\d.,])(20\d\d)(?![\d.,]\d)")
# A target year ("500 more stores by 2018", "d'ici 2025") is a plan, not the period asked about.
TARGET_RE = re.compile(r"\b(?:by|until|d'ici|jusqu'en|jusqu'à)\s+(?:the end of\s+|la fin de\s+)?(?:fiscal\s+|l'exercice\s+)?20\d\d")


_YEAR_WORD = r"(?:l'exercice\s+|l'annee\s+|l'année\s+|fiscal\s+|the year\s+|year\s+)?"
# "de 2020 à 2024", "from 2020 to 2024", "2020-2024", "fiscal 2019 through fiscal 2022", "2020 jusqu'à 2024"
RANGE_RE = re.compile(rf"{_YEAR_WORD}(20\d\d)\s*(?:-|à|au|to|through|thru|until|jusqu'à|jusqu'en)\s*{_YEAR_WORD}(20\d\d)\b")
# "entre 2020 et 2024", "between 2019 and 2022" ("2020 and 2024" alone names two years, not a period)
BETWEEN_RE = re.compile(rf"\b(?:entre|between)\s+(?:les\s+exercices\s+)?{_YEAR_WORD}(20\d\d)\s+(?:et|and)\s+{_YEAR_WORD}(20\d\d)\b")
# "depuis 2020", "since fiscal 2019": up to the latest report
SINCE_RE = re.compile(rf"\b(?:depuis|since)\s+{_YEAR_WORD}(20\d\d)\b")
# "les trois derniers exercices", "over the last five years", "the past 3 fiscal years"
NUMBERS = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
           "two": 2, "three": 3, "four": 4, "five": 5, "six ": 6, "seven": 7, "eight": 8}
LAST_N_RE = re.compile(r"\b(?:(?:les|des|aux)\s+(\d|deux|trois|quatre|cinq|six|sept|huit)\s+derni[eè]r(?:es|s)\s+"
                       r"(?:exercices|années|annees|ans)|(?:last|past)\s+(\d|two|three|four|five|six|seven|eight)\s+"
                       r"(?:fiscal\s+)?years)\b")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("’", "'").replace("–", "-").replace("—", "-").lower()
    return re.sub(r"\s+", " ", text)


# "pour toutes les entreprises", "chaque société", "for each company", "across all companies", "which company..."
_FIRM = r"(?:entreprises?|soci[ée]t[ée]s?|compagnies?|[ée]metteurs?|groupes?)"
# Universal wordings always mean every company: "toutes les entreprises", "chaque société", "all companies".
ALL_COMPANIES_RE = re.compile(
    rf"\b(?:toutes les {_FIRM}|tous les (?:[ée]metteurs|groupes)|chaque {_FIRM}|chacune des {_FIRM}|"
    rf"les (?:six|6) {_FIRM}|les {_FIRM} (?:du corpus|disponibles|analys[ée]es)|"
    r"all (?:the |six |6 )?(?:companies|issuers|firms)|each (?:of the )?(?:compan(?:y|ies)|firms?)|"
    r"every (?:company|firm)|across (?:all )?(?:the )?companies)\b")
# Interrogative wordings mean every company only when none is named: "Quelle entreprise a le plus
# d'employés ?" compares them all, but "Quelle entreprise Air Canada a-t-elle acquise ?" is about Air Canada.
WHICH_COMPANY_RE = re.compile(
    rf"\b(?:quelles? {_FIRM}|lesquelles? (?:des|de ces) {_FIRM}|laquelle des {_FIRM}|parmi les {_FIRM}|"
    r"which (?:of the )?(?:compan(?:y|ies)|firms?)|among the companies)\b")


def detect_companies(question: str) -> list[str]:
    """Every corpus company named in the question, in corpus order."""
    text = _norm(question)
    return [c for c, patterns in COMPANY_PATTERNS.items() if any(re.search(p, text) for p in patterns)]


def mentions_all_companies(question: str, named: list[str] | None = None) -> bool:
    """True for a question about every company: a universal wording, or an interrogative one ("which
    company...?") when no company is named."""
    text = _norm(question)
    if named is None:
        named = detect_companies(question)
    return bool(ALL_COMPANIES_RE.search(text)) or (not named and bool(WHICH_COMPANY_RE.search(text)))


def detect_company(question: str) -> str | None:
    text = _norm(question)
    found = [c for c, patterns in COMPANY_PATTERNS.items() if any(re.search(p, text) for p in patterns)]
    return found[0] if len(found) == 1 else None   # none or several: do not filter on the company


def detect_year(question: str) -> int | None:
    """The fiscal year of the report to search.

    A named report wins. Otherwise the latest year mentioned: a question comparing 2022 with 2021 is
    answered by the 2022 report, which shows both years. Years outside the corpus are ignored.
    """
    text = _norm(question)
    if (m := REPORT_RE.search(text)) and int(m.group(1) or m.group(2)) in YEARS:
        return int(m.group(1) or m.group(2))
    years = [int(y) for y in YEAR_RE.findall(TARGET_RE.sub(" ", text)) if int(y) in YEARS]
    return max(years) if years else None


def detect_period(question: str) -> list[int] | None:
    """The fiscal years of a question about several years, or None for a single-year question.

    "2022 compared with 2021" stays a single-year question: the 2022 report shows both years. A period is
    a range ("from 2020 to 2024", "entre 2019 et 2022", "since 2020", "the last three years") or at least two
    years that are not consecutive ("2019 and 2024") or three years or more ("2019, 2020 and 2021").
    """
    text = TARGET_RE.sub(" ", _norm(question))
    latest = max(YEARS)
    for pattern in (BETWEEN_RE, RANGE_RE):
        if m := pattern.search(text):
            start, end = sorted((int(m.group(1)), int(m.group(2))))
            if start != end:
                years = [y for y in range(start, end + 1) if y in YEARS]
                return years if len(years) >= 2 else None
    if m := SINCE_RE.search(text):
        start = int(m.group(1))
        years = [y for y in range(start, latest + 1) if y in YEARS]
        return years if len(years) >= 2 else None
    if m := LAST_N_RE.search(text):
        word = m.group(1) or m.group(2)
        n = int(word) if word.isdigit() else NUMBERS.get(word, 0)
        mentioned = [int(y) for y in YEAR_RE.findall(text) if int(y) in YEARS]
        end = max(mentioned) if mentioned else latest       # "the last three years" of the corpus
        years = [y for y in range(end - n + 1, end + 1) if y in YEARS]
        return years if len(years) >= 2 else None
    mentioned = sorted({int(y) for y in YEAR_RE.findall(text) if int(y) in YEARS})
    if len(mentioned) >= 3 or (len(mentioned) == 2 and mentioned[1] - mentioned[0] > 1):
        return mentioned
    return None


def names_report(question: str) -> bool:
    """True when the question names the report to use ("selon le rapport annuel 2023")."""
    m = REPORT_RE.search(_norm(question))
    return bool(m) and int(m.group(1) or m.group(2)) in YEARS


def detect_filters(question: str) -> dict:
    """{"company", "year", "years", "report"}; a value is None when it could not be detected reliably.

    "years" is the list of fiscal years of a period question (then "year" is None); "report" is True when
    the question names the report: its year is then used as is, even for a period inside that report.
    """
    company = detect_company(question)
    # A year without a company would mix all companies of that year: it still narrows the search, but a
    # question without a company is more often a trap (a company outside the corpus): keep it unfiltered.
    if not company:
        return {"company": None, "year": None, "years": None, "report": False}
    if names_report(question):
        return {"company": company, "year": detect_year(question), "years": None, "report": True}
    if period := detect_period(question):
        return {"company": company, "year": None, "years": period, "report": False}
    return {"company": company, "year": detect_year(question), "years": None, "report": False}


def main() -> None:
    """Compare the detection with the company and year stored with each evaluation question."""
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        for question in sys.argv[2:]:
            print(detect_filters(question), "|", question)
        return
    path = ROOT / "eval" / "questions.jsonl"
    questions = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    counts = {"company_ok": 0, "company_missed": 0, "company_wrong": 0,
              "year_ok": 0, "year_missed": 0, "year_wrong": 0, "trap_unfiltered": 0, "trap_filtered": 0}
    problems = []
    for q in questions:
        got = detect_filters(q["question"])
        if not q.get("answerable", True):
            key = "trap_filtered" if got["company"] else "trap_unfiltered"
            counts[key] += 1
            if got["company"]:  # fine when the question names a corpus company (a metric it does not report)
                problems.append((q["id"], "unanswerable, filtered (check the company is named)", got, q))
            continue
        for field in ("company", "year"):
            expected, value = q.get(field), got[field]
            if value is None:
                counts[f"{field}_missed"] += 1
                problems.append((q["id"], f"{field} missed", got, q))
            elif value == expected:
                counts[f"{field}_ok"] += 1
            else:
                counts[f"{field}_wrong"] += 1
                problems.append((q["id"], f"{field} WRONG", got, q))
    n = sum(q.get("answerable", True) for q in questions)
    print(f"{n} answerable questions, {len(questions) - n} unanswerable")
    for key, value in counts.items():
        print(f"  {key:<16} {value}")
    print("\nA missed value only means no filter (the search stays as today); a WRONG one hides the answer."
          "\nUnanswerable questions: a company outside the corpus (WestJet, RBC...) must stay unfiltered.")
    for qid, issue, got, q in problems:
        print(f"- {qid} {issue}: detected {got['company']}/{got['year']}, expected "
              f"{q.get('company')}/{q.get('year')} | {q['question'][:120]}")


if __name__ == "__main__":
    main()
