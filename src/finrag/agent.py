"""Phase E: the conversational agent. A router in front of two answer paths, and a memory of the conversation.

Router
    A question about an indicator of the table of phase D2 (revenue, net income, diluted EPS, total assets,
    CET1 ratio) for given companies and fiscal years is answered from the table: values checked against
    their passage, cross-checked with the next report or verified by hand, with their source and page.
    No search, no LLM, under a second. Everything else goes to the RAG (generate.answer): explanations,
    figures that are not in the table, quarters, segments, adjusted measures, rankings. When a value is
    missing from the table or has a low confidence, the question goes to the RAG as well.

    The indicator is found with a short list of patterns. Unlike the time words of the query planner, a
    missed wording costs nothing: the question goes to the RAG, as before. The patterns only decide when
    the shortcut is safe.

Memory
    Each turn keeps its plan (companies, years) and indicator. A follow-up question takes from the previous
    turn what it does not say:
        "Résultat net de CGI en 2024 ?"  ->  CGI, 2024, net income        (table)
        "Et en 2023 ?"                   ->  CGI, 2023, net income        (table)
        "Et pour Desjardins ?"           ->  Desjardins, 2023, net income (table)
        "Pourquoi a-t-il baissé ?"       ->  Desjardins, 2023             (RAG, question rewritten)
    A follow-up is a question that starts like one ("et", "and", "what about", "qu'en est-il") or names no
    company. A new question that names a company and no follow-up word starts a new topic.
    For the RAG, the follow-up is rewritten as a standalone question by the LLM (one short call), since the
    search and the answer model only see the question.

    python -m finrag.agent       # chat in the terminal
"""
import json
import re
from dataclasses import dataclass, field

from .config import LLM_MODEL, NUM_CTX, PROCESSED_DIR
from .query_filters import _norm, detect_companies
from .query_plan import LATEST_YEAR, plan_question

KPIS_FILE = PROCESSED_DIR / "kpis.jsonl"
TRUSTED = {"manual", "high", "medium"}
NAMES = {"aircanada": "Air Canada", "banquenationale": "Banque Nationale", "cgi": "CGI",
         "couchetard": "Couche-Tard", "desjardins": "Desjardins", "shopify": "Shopify"}
NAMES_EN = {**NAMES, "banquenationale": "National Bank", "desjardins": "Desjardins Group"}
FY_END = {"aircanada": ("le 31 décembre", "December 31"), "banquenationale": ("le 31 octobre", "October 31"),
          "cgi": ("le 30 septembre", "September 30"), "couchetard": ("fin avril", "in late April"),
          "desjardins": ("le 31 décembre", "December 31"), "shopify": ("le 31 décembre", "December 31")}
CURRENCY = {"couchetard": "USD", "shopify": "USD"}
KPI_LABELS = {"revenue": ("Revenus", "Revenue"), "net_income": ("Résultat net", "Net income"),
              "diluted_eps": ("Résultat par action dilué", "Diluted earnings per share"),
              "total_assets": ("Total de l'actif", "Total assets"), "cet1_ratio": ("Ratio CET1", "CET1 ratio")}
CONFIDENCE_TEXT = {"manual": ("vérifiée à la main", "verified by hand"),
                   "high": ("vérifiée et confirmée par le rapport suivant", "verified and confirmed by the next report"),
                   "medium": ("vérifiée dans le rapport", "verified in the report")}

# Earnings per share first: "résultat net par action" also contains "résultat net".
KPI_PATTERNS = [
    ("diluted_eps", r"par action|per share|\beps\b|\bbpa\b"),
    ("cet1_ratio", r"\bcet ?1\b|common equity tier 1|cat[ée]gorie 1a?\b|tier 1a?\b"),
    ("total_assets", r"total de l.actif|total des actifs|actifs? totaux|actif total|total assets"),
    ("net_income", r"r[ée]sultats? nets?|b[ée]n[ée]fices? nets?|pertes? nettes?|net (?:income|earnings|loss|profit)|"
                   r"profit net|exc[ée]dents?|surplus"),
    # Not "ventes" / "sales": Couche-Tard's "ventes de carburant" are one line of its revenue, not the total.
    ("revenue", r"\brevenus?\b|\brevenues?\b|chiffre d.affaires|total des produits|produits d.exploitation|"
                r"\bturnover\b"),
]
# Measures the table does not hold, even though they contain an indicator's words: the RAG answers them.
NOT_IN_TABLE_RE = re.compile(
    r"net d.int[ée]r[eê]ts|net interest|trimestr|quarter|\b[qt][1-4]\b|trois mois|three months|secteur|segment|"
    r"r[ée]gion|g[ée]ograph|pays|country|[ée]tats-unis|united states|ajust|adjusted|non conforme|non-gaap|"
    r"de base|basic|avant (?:imp[oô]ts|int[ée]r[eê]ts)|before (?:tax|interest)|baiia?|ebitda?|marge|margin|"
    r"dollars? (?:am[ée]ricains?|canadiens?|us)|us dollars?|canadian dollars?|en euros?|convert|"
    r"global|comprehensive|par employ|per employee|moyen|average|pr[ée]vision|forecast|objectif|target|guidance|"
    # one kind of revenue or one share of net income, not the total
    r"placement|investment|commissions?|honoraires|fees|assurance|insurance|primes|premiums|passagers?|passenger|"
    r"cargo|fret|carburant|fuel|marchandises|merchandise|abonnement|subscription|solutions|transactions?|"
    r"location|rental|autres revenus|other revenue|diff[ée]r[ée]s|deferred|unearned|attribuable|attributable|"
    r"actionnaires ordinaires|common shareholders|participations ne donnant|non-controlling|minoritaires")
# Questions that need reading and reasoning, not a value: explanation, ranking, judgment.
EXPLAIN_RE = re.compile(
    r"\bpourquoi\b|\bwhy\b|expliqu|explain|\bcauses?\b|raisons?|reasons?|\bcomment\b|\bhow (?:did|does|do|has|have|is|was)\b|"
    r"facteurs?|factors?|drivers?|impact|effet|effect|plus [ée]lev[ée]|plus faible|highest|lowest|largest|"
    r"smallest|class|rank|meilleur|best|worst|pire|risques?|risks?|strat[ée]gie|strategy|perspectives?|outlook")
FOLLOW_UP_RE = re.compile(r"^\s*(?:et|and|or|ou|what about|how about|same|m[eê]me chose|idem|pareil|"
                          r"qu.en est-il|aussi|also|puis|then)\b")


# Words of a short follow-up that ask nothing new: "2023 ?", "Et pour Desjardins ?", "Same for CGI?",
# "Et les plus récents ?". A question with other words ("Qui est le PDG ?") asks something else.
FILLER = set("""et en pour and in for of the what about how same même meme chose idem pareil aussi also puis then
alors ou or qu en est-il qu'en celui celle ceux celles son sa ses leur leurs its their it à a au aux le la les l
de du des d un une exercice année annee fiscal year fy plus most le la récent récente récents récentes recent
latest dernier dernière derniere derniers dernières dernieres last précédent precedent previous ?""".split())


def is_short_follow_up(question: str) -> bool:
    """Only years, company names and filler words: the question repeats the previous one with a change."""
    words = re.findall(r"[a-zà-ÿ0-9]+(?:[-'][a-zà-ÿ]+)*", _without_names(question))
    return all(w in FILLER for w in words)


def _without_names(question: str) -> str:
    """The question without company names and years, normalised."""
    from .query_filters import COMPANY_PATTERNS
    text = _norm(question)
    for patterns in COMPANY_PATTERNS.values():
        for p in patterns:
            text = re.sub(p, " ", text)
    return re.sub(r"\b(?:fy)?\d{2,4}\b", " ", text)


BARE_RESULT_RE = re.compile(r"\b(?:r[ée]sultats?|b[ée]n[ée]fices?|profits?|earnings)\b(.*)")


def bare_result(question: str) -> bool:
    """ "Le résultat de Desjardins de 2021 à 2023", "CGI's earnings in 2022": a result with nothing after it but
    filler words, once company names and years are removed. "Résultat d'exploitation", "résultat global",
    "bénéfice avant impôts" name another measure and do not count."""
    m = BARE_RESULT_RE.search(_without_names(question))
    if not m:
        return False
    rest = [w for w in re.findall(r"[a-zà-ÿ]+", m.group(1)) if w not in {"d", "l"}]
    return all(w in FILLER or w in {"entre", "sur", "depuis", "compare", "comparer", "to", "from"} for w in rest)


def detect_kpi(question: str) -> str | None:
    """The indicator of the table the question asks for, or None (none, or several)."""
    text = _norm(question)
    found = [k for k, p in KPI_PATTERNS if re.search(p, text)]
    if "diluted_eps" in found:                       # "résultat net par action": one indicator, the EPS
        found = [k for k in found if k != "net_income"]
    if not found and bare_result(question):          # "le résultat", "le bénéfice" alone: the net income
        found = ["net_income"]
    return found[0] if len(found) == 1 else None


def lang_of(question: str) -> str:
    try:
        from .retrieve import detect_lang
        return detect_lang(question) or "fr"
    except Exception:                    # retrieve.py needs the index; the language is a detail
        return "en" if re.search(r"\b(?:what|the|of|in|and|how|was|is)\b", question.lower()) else "fr"


# ------------------------------------------------------------------ table ---

_table: dict = {"mtime": None, "values": {}}


def table_values() -> dict:
    """{(company, year, kpi): row} of kpis.jsonl, read again when the file changes."""
    if not KPIS_FILE.exists():
        return {}
    mtime = KPIS_FILE.stat().st_mtime
    if _table["mtime"] != mtime:
        rows = [json.loads(line) for line in KPIS_FILE.open(encoding="utf-8") if line.strip()]
        _table["values"] = {(r["company"], int(r["year"]), r["kpi"]): r for r in rows}
        _table["mtime"] = mtime
    return _table["values"]


def fmt_number(v: float, digits: int, lang: str) -> str:
    s = f"{v:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",") if lang == "fr" else s


def fmt_value(row: dict, kpi: str, lang: str) -> str:
    v, cur = float(row["value_m"]), row.get("currency") or CURRENCY.get(row["company"], "CAD")
    sign = "−" if v < 0 and lang == "fr" else ("-" if v < 0 else "")
    if kpi == "cet1_ratio":
        return f"{fmt_number(v, 1, lang)} %" if lang == "fr" else f"{fmt_number(v, 1, lang)}%"
    if kpi == "diluted_eps":
        return (f"{sign}{fmt_number(abs(v), 2, lang)} $ {'US' if cur == 'USD' else 'CA'} par action" if lang == "fr"
                else f"{sign}{'US' if cur == 'USD' else 'C'}${fmt_number(abs(v), 2, lang)} per share")
    digits = 0 if abs(v) >= 100 and v == int(v) else 1
    return (f"{sign}{fmt_number(abs(v), digits, lang)} M$ {'US' if cur == 'USD' else 'CA'}" if lang == "fr"
            else f"{sign}{'US' if cur == 'USD' else 'C'}${fmt_number(abs(v), digits, lang)} million")


def table_answer(plan: dict, kpi: str, lang: str) -> tuple[dict | None, str]:
    """The answer from the table, or (None, reason) when a value is missing or not trusted."""
    companies = plan.get("companies") or []
    years = plan.get("years") or ([plan["year"]] if plan.get("year") else [])
    latest = not years
    years = years or [LATEST_YEAR]
    values = table_values()
    rows, missing = [], []
    for c in companies:
        for y in years:
            r = values.get((c, y, kpi))
            if r and r.get("value_m") is not None and r.get("confidence") in TRUSTED:
                rows.append(r)
            else:
                missing.append(f"{c} {y}")
    if not companies or missing:
        return None, ("no company" if not companies else "not in the table or not verified: " + ", ".join(missing))
    i = 0 if lang == "fr" else 1
    label = KPI_LABELS[kpi][i]
    lines, sources = [], []
    for n, r in enumerate(rows, start=1):
        c, y = r["company"], int(r["year"])
        name = (NAMES if lang == "fr" else NAMES_EN)[c]
        fy = (f"exercice {y} (clos {FY_END[c][0]})" if lang == "fr" else f"fiscal {y} (year ended {FY_END[c][1]})")
        lines.append((c, y, f"{name}, {fy} : {fmt_value(r, kpi, lang)} [T{n}]" if lang == "fr"
                      else f"{name}, {fy}: {fmt_value(r, kpi, lang)} [T{n}]"))
        sources.append({"n": n, "label": f"T{n}", "company": c, "year": y, "page": r.get("page"),
                        "document": r.get("source"), "kind": "kpi table",
                        "excerpt": f"{r.get('line') or label} : {r['value_m']} ({CONFIDENCE_TEXT[r['confidence']][i]})"})
    head = []
    if latest:
        head.append(f"Exercice le plus récent du corpus : {LATEST_YEAR}." if lang == "fr"
                    else f"Latest fiscal year in the corpus: {LATEST_YEAR}.")
    if len(lines) == 1:
        body = [f"{label} — {lines[0][2]}"]
    else:
        body = [f"{label} :" if lang == "fr" else f"{label}:"] + [f"- {text}" for _, _, text in lines]
    # A series: the change from the first to the last year, per company (not across currencies).
    if kpi in {"revenue", "net_income", "total_assets"} and len(years) > 1:
        for c in companies:
            series = sorted((int(r["year"]), float(r["value_m"])) for r in rows if r["company"] == c)
            (y0, v0), (y1, v1) = series[0], series[-1]
            if v0 > 0:
                pct = (v1 - v0) / v0 * 100
                body.append(f"Variation {y0} → {y1} ({(NAMES if lang == 'fr' else NAMES_EN)[c]}) : "
                            f"{'+' if pct > 0 else ''}{fmt_number(pct, 1, lang)} %" if lang == "fr" else
                            f"Change {y0} → {y1} ({NAMES_EN[c]}): {'+' if pct > 0 else ''}{fmt_number(pct, 1, lang)}%")
    if "desjardins" in companies and kpi in {"revenue", "net_income"} and min(years) <= 2021 < max(years):
        body.append("Attention : Desjardins applique IFRS 17 (contrats d'assurance) depuis 2023, avec 2022 retraité. "
                    "Ses revenus et ses excédents d'avant 2022 ne se comparent pas à ceux d'après."
                    if lang == "fr" else "Note: Desjardins applies IFRS 17 (insurance contracts) since 2023, with 2022 "
                    "restated; its figures before 2022 are not comparable with those after.")
    if len({CURRENCY.get(c, "CAD") for c in companies}) > 1:
        body.append("Attention : Couche-Tard et Shopify publient en dollars américains, les autres en dollars canadiens."
                    if lang == "fr" else "Note: Couche-Tard and Shopify report in US dollars, the others in Canadian dollars.")
    body.append("Source : tableau des indicateurs vérifiés (montants en millions, devise de publication)."
                if lang == "fr" else "Source: verified indicator table (amounts in millions, reporting currency).")
    return {"answer": "\n".join(head + body), "sources": sources}, ""


# ----------------------------------------------------------------- memory ---

@dataclass
class Turn:
    question: str
    standalone: str
    plan: dict
    kpi: str | None
    route: str
    answer: str = ""


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 10

    @property
    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None


# A name in the question (a capitalised word after the first one) that is not a place: "la Banque Royale
# du Canada", "WestJet". Places and acronyms ("États-Unis", "EPS", "CET1") are not company names.
NAME_RE = re.compile(r"(?<!^)\b[A-ZÀ-Ý][a-zà-ÿ]+(?:[- ][A-ZÀ-Ý][a-zà-ÿ]+)*")
PLACES = {"Canada", "Québec", "Quebec", "Ontario", "États-Unis", "United States", "États", "Unis", "United",
          "States", "Europe", "Amérique", "America", "Asie", "Asia", "Covid", "Royaume-Uni", "France"}


def names_a_company(question: str) -> bool:
    """A name that could be a company outside the corpus, other than the first word of the question."""
    body = question.strip().split(" ", 1)[1] if " " in question.strip() else ""
    return any(m not in PLACES for m in NAME_RE.findall(body))


def resolve(question: str, previous: Turn | None, parser: str | None = None) -> tuple[dict, str | None, list[str]]:
    """The plan and indicator of a question, completed from the previous turn for a follow-up.

    A question that names no company of the corpus and no other name ("Et en 2023 ?", "Pourquoi ?") is a
    follow-up: it is planned with the companies of the previous turn written in, so that its years are read
    as for a full question ("Et ses revenus les plus récents ?" -> 2025). Planned alone, the LLM of the
    planner would see a question without a company and could answer "a company outside the corpus".
    A question that names a corpus company is a follow-up only if it starts like one ("Et pour Desjardins ?").
    A name outside the corpus ("Et celui de la Banque Royale ?") is never completed from memory.
    """
    text = _norm(question)
    # "Revenu net d'intérêts", "revenus du quatrième trimestre": words of an indicator, another measure.
    kpi = None if NOT_IN_TABLE_RE.search(text) else detect_kpi(question)
    inherited = []
    previous_companies = list(previous.plan.get("companies") or []) if previous else []
    if previous_companies and not detect_companies(question) and not names_a_company(question):
        names = ", ".join(NAMES[c] for c in previous_companies if c in NAMES)
        plan = dict(plan_question(f"{question} ({names})", parser))
        plan.update(companies=previous_companies, all=previous.plan.get("all", False), outside=False)
        inherited.append("companies")
    else:
        plan = dict(plan_question(question, parser))
        if plan.get("companies") or not previous_companies:
            if not (previous and (FOLLOW_UP_RE.search(text) or is_short_follow_up(question))):
                return plan, kpi, inherited                     # a new question
        else:
            # A name the corpus does not have: answered as such (declined), never completed from memory.
            plan.update(outside=True)
            return plan, kpi, inherited
    if plan.get("year") is None and not plan.get("years"):
        plan.update(year=previous.plan.get("year"), years=previous.plan.get("years"),
                    report=previous.plan.get("report", False))
        if plan["year"] is not None or plan["years"]:
            inherited.append("years")
    # The indicator carries over to a short follow-up ("2023 ?") and to a question about it ("Pourquoi ?"), not
    # to another question about the same company ("Qui est le PDG ?").
    if (kpi is None and previous.kpi and not NOT_IN_TABLE_RE.search(text)
            and (is_short_follow_up(question) or EXPLAIN_RE.search(text))):
        kpi = previous.kpi
        inherited.append("kpi")
    plan["parser"] = f"{plan.get('parser', '')}+memory"
    return plan, kpi, inherited


def choose_route(question: str, plan: dict, kpi: str | None, inherited: list[str] | None = None) -> tuple[str, str]:
    text = _norm(question)
    if plan.get("outside"):
        return "rag", "company outside the corpus"
    if NOT_IN_TABLE_RE.search(text):
        return "rag", "a measure the table does not hold"
    if kpi is None:
        return "rag", "no indicator of the table"
    if EXPLAIN_RE.search(text):
        return "rag", "an explanation or a ranking"
    # An indicator taken from the previous turn is the question only for a short follow-up ("Et en 2023 ?",
    # "Et pour Desjardins ?"). "Quelles sont les exigences minimales pour ce ratio ?" keeps the ratio as
    # context for the search, but asks something else.
    if "kpi" in (inherited or []) and not (FOLLOW_UP_RE.search(text) or is_short_follow_up(question)):
        return "rag", "the indicator of the previous question, but another question"
    if not plan.get("companies"):
        return "rag", "no company"
    return "table", ""


REWRITE_PROMPT = """Rewrite the user's last question as one standalone question, in the same language. What it
leaves implicit (company, fiscal year, metric) is given as "Context": use exactly that context, and the previous
questions only to understand the wording. Do not bring back companies or years of older questions that are not
in the context, do not add a comparison the user did not ask for, put no figure in it the user did not write,
and do not answer it. Reply with the question only."""

BARE_WHY_RE = re.compile(r"^\s*(?:et\s+|and\s+)?(?:pourquoi|why|comment (?:ça|ca|cela)|how come|explique|explain)"
                         r"(?:\s+(?:donc|ça|ca|cela|so|that|it))?\s*[?.!]*\s*$", re.I)


def short_why(question: str) -> bool:
    """ "Pourquoi ?", "pourquoi il a diminué ?", "Why did it fall so much?": a short question about the value
    just given, with no company and no year of its own."""
    text = _norm(question)
    return (bool(BARE_WHY_RE.search(text)) or (
        bool(re.search(r"\b(?:pourquoi|why|how come|comment (?:ça|ca|cela) se fait)\b", text))
        and not detect_companies(question) and not re.search(r"\b20\d\d\b", text)
        and len(re.findall(r"[a-zà-ÿ]+", text)) <= 7))


def compared_years(conversation: "Conversation", plan: dict, kpi: str) -> tuple[int, int] | None:
    """(year, base year) of a why-question. After "2020 ?" then "et en 2019 ?", "pourquoi il a diminué ?"
    asks about the change between those two years: the later one against the earlier one, whatever the
    order of the questions. Otherwise the year of the plan against the year before."""
    companies = plan.get("companies") or []
    recent = [t for t in conversation.turns[-2:]
              if t.kpi == kpi and (t.plan.get("companies") or []) == companies and t.plan.get("year")]
    years = sorted({t.plan["year"] for t in recent})
    if len(years) == 2:
        return years[1], years[0]
    year = plan.get("year") or (plan.get("years") or [None])[-1]
    return (year, year - 1) if year else None


def why_question(plan: dict, kpi: str, lang: str, base: int | None = None) -> str | None:
    """A why-question after a value of the table: the question that worked best in the COVID study, why the
    indicator changed from one year to another (by default, the year before)."""
    companies = plan.get("companies") or []
    years = plan.get("years") or ([plan["year"]] if plan.get("year") else [])
    if len(companies) != 1 or len(years) != 1 or companies[0] not in NAMES:
        return None
    base = base or years[0] - 1
    if lang == "fr":
        subject, verb = {"revenue": ("les revenus", "ont-ils"), "net_income": ("le résultat net", "a-t-il"),
                         "diluted_eps": ("le résultat par action dilué", "a-t-il"),
                         "total_assets": ("le total de l'actif", "a-t-il"), "cet1_ratio": ("le ratio CET1", "a-t-il")}[kpi]
        return (f"Pourquoi {subject} de {NAMES[companies[0]]} {verb} varié pendant l'exercice {years[0]} par rapport "
                f"à l'exercice {base} ? Explique les principaux écarts.")
    return (f"Why did {NAMES_EN[companies[0]]}'s {KPI_LABELS[kpi][1].lower()} change in fiscal {years[0]} compared "
            f"with fiscal {base}? Explain the main differences.")


def rewrite(question: str, conversation: Conversation, plan: dict, kpi: str | None) -> str:
    """A standalone question for the RAG; without the LLM, the question with its context in brackets."""
    names = ", ".join(NAMES[c] for c in plan.get("companies") or [] if c in NAMES)
    years = plan.get("years") or ([plan["year"]] if plan.get("year") else [])
    context = "; ".join(x for x in (names, ", ".join(map(str, years)),
                                    KPI_LABELS[kpi][0] if kpi else "") if x)
    fallback = f"{question} ({context})" if context else question
    if kpi and short_why(question) and (q := why_question(plan, kpi, lang_of(question), plan.get("base_year"))):
        return q
    # The previous questions only, not the answers: a wrong figure of an answer would otherwise become part
    # of the next question ("Pourquoi le résultat net était-il de 14 660 M$ ?") and of its search.
    history = "\n".join(f"User: {t.standalone}" for t in conversation.turns[-3:])
    try:
        from .index import get_ollama
        response = get_ollama().chat(model=LLM_MODEL, think=False, messages=[
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Previous questions:\n{history}\n\nLast question: {question}\n"
                                        f"(Context found: {context or 'none'})"}],
            options={"temperature": 0, **({"num_ctx": NUM_CTX} if NUM_CTX else {})})
        text = re.sub(r"<think>.*?</think>", "", response.message.content or "", flags=re.S).strip().strip('"')
        return text if 5 <= len(text) <= 400 else fallback
    except Exception:
        return fallback


# ------------------------------------------------------------------ agent ---

# The words of the reports for an indicator, when they differ from the usual ones: a cooperative has
# "excédents", not a "résultat net"; without them, the search finds "revenu net total" (Desjardins's revenue).
REPORT_TERMS = {
    ("cgi", "net_income"): ("bénéfice net", "net earnings"),
    ("couchetard", "net_income"): ("bénéfice net", "net earnings"),
    ("desjardins", "net_income"): ("excédents avant ristournes aux membres", "surplus earnings before member dividends"),
    ("desjardins", "revenue"): ("revenu net total, revenu total", "total net income, total income"),
    ("banquenationale", "revenue"): ("revenu total", "total revenues"),
    ("couchetard", "revenue"): ("chiffre d'affaires, total des ventes", "revenues, total sales"),
    ("desjardins", "cet1_ratio"): ("ratio des fonds propres de la catégorie 1A", "Tier 1A capital ratio"),
}


def report_terms(plan: dict, kpi: str | None, lang: str) -> str:
    companies = plan.get("companies") or []
    if kpi is None or len(companies) != 1:
        return ""
    terms = REPORT_TERMS.get((companies[0], kpi))
    return terms[0 if lang == "fr" else 1] if terms else ""


def table_check(answer_text: str, plan: dict, kpi: str | None) -> dict | None:
    """For an indicator answered by the RAG: the value of the table, even one still to review, and whether the
    answer gives it. A different figure is worth a warning: one of the two is wrong."""
    companies = plan.get("companies") or []
    years = plan.get("years") or ([plan["year"]] if plan.get("year") else [])
    if kpi is None or len(companies) != 1 or len(years) != 1:
        return None
    row = table_values().get((companies[0], years[0], kpi))
    if not row or row.get("value_m") is None or row.get("confidence") == "low":
        return None
    from .verify import figures, found_in
    numbers = [v for _, v in figures(answer_text)]
    value = abs(float(row["value_m"]))
    return {"value_m": row["value_m"], "currency": row.get("currency") or CURRENCY.get(companies[0], "CAD"),
            "confidence": row.get("confidence"), "page": row.get("page"), "document": row.get("source"),
            "match": found_in(value, numbers)}


def chat(conversation: Conversation, question: str, k: int = 10, parser: str | None = None) -> dict:
    """Answer one turn of the conversation; the turn is added to it."""
    previous = conversation.last
    plan, kpi, inherited = resolve(question, previous, parser)
    route, reason = choose_route(question, plan, kpi, inherited)
    lang = lang_of(question)
    result = None
    if route == "table":
        result, why = table_answer(plan, kpi, lang)
        if result is None:
            route, reason = "rag", why
    standalone = question
    if route == "rag":
        from .generate import answer
        if inherited and previous is not None:
            if kpi and short_why(question) and (pair := compared_years(conversation, plan, kpi)):
                # Search the later year (its report compares it with the year before) and say which change.
                plan.update(year=pair[0], years=None, base_year=pair[1])
            standalone = rewrite(question, conversation, plan, kpi)
        if (terms := report_terms(plan, kpi, lang)) and terms.lower() not in standalone.lower():
            standalone = f"{standalone} ({terms})"
        result = answer(standalone, k=k, plan=plan)          # the plan is already made: no second planner call
        plan = result.get("filters") or plan
        # Only when the question asks for the value: an explanation need not repeat it.
        result["table_check"] = None if EXPLAIN_RE.search(_norm(question)) else table_check(result["answer"], plan, kpi)
    conversation.turns.append(Turn(question, standalone, plan, kpi, route, result["answer"]))
    del conversation.turns[:-conversation.max_turns]
    return {**result, "filters": plan, "route": route, "route_reason": reason, "kpi": kpi,
            "inherited": inherited, "standalone": standalone}


def main() -> None:
    conversation = Conversation()
    print("FinRAG — conversation (Ctrl+C pour quitter)")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        res = chat(conversation, question)
        print(f"\n[{res['route']}{' — ' + res['route_reason'] if res['route_reason'] else ''}]"
              + (f" hérité : {', '.join(res['inherited'])}" if res["inherited"] else "")
              + (f"\nQuestion comprise : {res['standalone']}" if res["standalone"] != question else ""))
        print(res["answer"])


if __name__ == "__main__":
    main()
