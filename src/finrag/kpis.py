"""Phase D2: extract key indicators into a table: company x fiscal year x indicator, with provenance.

    python -m finrag.kpis                                   # every company, year and indicator
    python -m finrag.kpis --companies aircanada --years 2019 2020 2021
    python -m finrag.kpis --crosscheck                      # also read each value in the next year's report
    python -m finrag.kpis --gold                            # compare with eval/kpi_gold.jsonl only

For each (company, year, indicator): a targeted search in the report of that year (filters: company and
year; the indicator's words, then the name of the financial statement; then the comparative column of the
next report), and the LLM extracts the value as JSON (value, unit, currency, source, row label). A value
is "verified" only if two deterministic checks pass: the figure is written in the passage the LLM cites,
and the row label names the indicator (not income before taxes for revenue, not comprehensive income for
net income). Values are normalised to millions of the reporting currency (per-share values and ratios as is).

With --crosscheck, the same indicator is read in the comparative column of the next report: equal values
are "consistent", different ones "restated" (a change of accounting standard, IFRS 17 at Desjardins).

Output: data/processed/kpis.csv (one row per value) and kpis.jsonl (with the passages' lines).
"""
import argparse
import csv
import json
import re
import time
import unicodedata

from .config import PROCESSED_DIR, ROOT
from .query_filters import YEARS

OUT_CSV = PROCESSED_DIR / "kpis.csv"
OUT_JSONL = PROCESSED_DIR / "kpis.jsonl"
GOLD_FILE = ROOT / "eval" / "kpi_gold.jsonl"

COMPANY_NAMES = {"aircanada": "Air Canada", "banquenationale": "Banque Nationale du Canada", "cgi": "CGI",
                 "couchetard": "Alimentation Couche-Tard", "desjardins": "Mouvement Desjardins", "shopify": "Shopify"}
CURRENCY = {"couchetard": "USD", "shopify": "USD"}           # the others report in CAD

# label: what the LLM looks for; query: words for the search, in French and English.
KPIS = {
    "revenue": {"label": "total revenues of the fiscal year for the whole company, not a business segment "
                         "(total des produits / revenus totaux)",
                "fr": "total des produits revenus état consolidé des résultats", "en": "total revenues consolidated statements of operations",
                "kind": "amount"},
    "net_income": {"label": "net income or net loss of the fiscal year for the whole company, not a business "
                            "segment (résultat net, bénéfice net ou perte nette; "
                            "for Desjardins: excédents avant ristournes aux membres / surplus earnings before member "
                            "dividends, not the adjusted line and not the line after member dividends); not comprehensive income",
                   "fr": "résultat net bénéfice net perte nette de l'exercice", "en": "net income net loss for the year",
                   "kind": "amount"},
    "diluted_eps": {"label": "diluted earnings (loss) per share (résultat net par action dilué)",
                    "fr": "résultat par action dilué", "en": "diluted earnings per share", "kind": "per_share"},
    "total_assets": {"label": "total assets at the end of the fiscal year (total de l'actif)",
                     "fr": "total de l'actif bilan état de la situation financière", "en": "total assets balance sheet",
                     "kind": "amount"},
    "cet1_ratio": {"label": "Common Equity Tier 1 (CET1) capital ratio at the end of the fiscal year (ratio des fonds propres CET1; "
                            "at Desjardins: ratio des fonds propres de la catégorie 1A)",
                   "fr": "ratio des fonds propres de catégorie 1 sous forme d'actions ordinaires CET1",
                   "en": "CET1 capital ratio", "kind": "percent"},
}
KPIS_BY_COMPANY = {
    "aircanada": ["revenue", "net_income", "diluted_eps", "total_assets"],
    "banquenationale": ["revenue", "net_income", "diluted_eps", "total_assets", "cet1_ratio"],
    "cgi": ["revenue", "net_income", "diluted_eps", "total_assets"],
    "couchetard": ["revenue", "net_income", "diluted_eps", "total_assets"],
    "desjardins": ["revenue", "net_income", "total_assets", "cet1_ratio"],      # a cooperative: no shares
    "shopify": ["revenue", "net_income", "diluted_eps", "total_assets"],
}
SCALE_TO_MILLIONS = {"units": 1e-6, "thousands": 1e-3, "millions": 1.0, "billions": 1e3}

# The row label the value is read from must be one of these, and none of the excluded ones: a figure of the
# right table but the wrong line (income before taxes for revenue) is rejected, even if it is in the source.
LINES = {
    "revenue": (r"total des produits|produits d.exploitation|revenus?|revenu total|chiffre d.affaires|total revenues?|"
                r"revenues?|total du revenu", r"par action|per share|net d.int[ée]r[eê]ts|co[uû]ts?|costs?|avant|before|"
                r"diff[ée]r[ée]s?|deferred|ajust[ée]|adjusted|imp[oô]ts|tax"),
    "net_income": (r"r[ée]sultat net|b[ée]n[ée]fice net|perte nette|net income|net loss|net earnings|excédents?|surplus",
                   # Desjardins: "excédents avant ristournes aux membres" is its net income (member dividends are a
                   # distribution, like a dividend); the line after them and the adjusted line are other measures.
                   r"global|comprehensive|avant(?! ristournes)|before(?! member dividends)|ajust[ée]|adjusted|"
                   r"apr[eè]s ristournes|after member dividends|par action|per share|attribuable aux participations"),
    "diluted_eps": (r"dilu[ée]|diluted|par action|per share", r"ajust[ée]|adjusted|dividende|dividend"),
    "total_assets": (r"total de l.actif|total des actifs|total assets|total actif", r"courant|current|net"),
    # Desjardins, a cooperative, calls it "catégorie 1A" (Tier 1A).
    "cet1_ratio": (r"cet1|cat[ée]gorie 1 sous forme d.actions ordinaires|common equity tier 1|cat[ée]gorie 1a|tier 1a",
                   r"montant|amount|\$"),
}
STATEMENT_QUERIES = {
    "revenue": {"fr": "comptes consolidés de résultat état consolidé des résultats produits", "en": "consolidated statements of operations revenues"},
    "net_income": {"fr": "comptes consolidés de résultat état consolidé des résultats", "en": "consolidated statements of operations net income"},
    "diluted_eps": {"fr": "résultat par action de base et dilué", "en": "net income per share basic and diluted"},
    "total_assets": {"fr": "état consolidé de la situation financière bilan consolidé", "en": "consolidated balance sheets total assets"},
    "cet1_ratio": {"fr": "ratios de fonds propres réglementaires CET1 catégorie 1 total", "en": "regulatory capital ratios CET1"},
}


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).replace("’", "'").lower()).strip()


def line_ok(kpi: str, line: str, passage: str) -> tuple[bool, str]:
    """The row label must name the indicator, not an excluded one, and be written in the passage."""
    accepted, excluded = LINES[kpi]
    label = _plain(line)
    if not label or not re.search(accepted, label):
        return False, f"line '{line}' does not name the indicator"
    if re.search(excluded, label):
        return False, f"line '{line}' is another measure"
    words = [w for w in re.findall(r"[^\W\d_]{3,}", label)]
    text = _plain(passage)
    if words and sum(w in text for w in words) < max(1, len(words) * 2 // 3):
        return False, f"line '{line}' is not in the passage"
    return True, ""


# A full fiscal year only: Couche-Tard publishes 12- and 16-week quarters next to its 52-week year.
PARTIAL_PERIOD_RE = re.compile(r"\b(?:12|16|13|24|28|36|40) (?:semaines|weeks)\b|trimestre|quarter|three months|"
                               r"trois mois|six mois|six months|nine months|neuf mois|\b[QT][1-4]\b", re.I)
# The whole company, not one of its business segments: Desjardins and National Bank publish a table per
# segment ("Particuliers et Entreprises") with the same row labels as the consolidated table (Desjardins 2020:
# 1 314, the segment's surplus, read for the 2 419 of the group). The title of the table says which it is.
SEGMENT_RE = re.compile(r"\bsecteurs? d.activit|\b(?:par|du|des) secteurs?\b|\bby segment|\bsegments?\b|particuliers et entreprises|gestion de patrimoine|"
                        r"assurance de dommages|assurance de personnes|march[ée]s financiers|personal and (?:commercial|business)|"
                        r"wealth management|financial markets|property and casualty|\bus specialty finance", re.I)


def table_title(text: str) -> str:
    """The header line of a chunk ("[Table | company | year | page | title]") or its first line."""
    return (text or "").strip().split("\n", 1)[0]


THOUSANDS_RE = re.compile(r"en milliers|in thousands|\(milliers|thousands of", re.I)
MILLIONS_RE = re.compile(r"en millions|in millions|millions de dollars|millions of", re.I)


def table_unit(text: str) -> str | None:
    """The unit stated in a passage (its title or header), when it states exactly one."""
    thousands, millions = bool(THOUSANDS_RE.search(text)), bool(MILLIONS_RE.search(text))
    return "thousands" if thousands and not millions else "millions" if millions and not thousands else None


def written_sign(value: float, text: str) -> int:
    """-1 if the figure is written in parentheses or with a minus in the passage, 1 if written plainly, 0 if absent."""
    from .run_eval import NUMBER_RE, parse_number
    signs = set()
    for m in NUMBER_RE.finditer(text):
        number = parse_number(m.group(0))
        if number is None or abs(abs(number) - abs(value)) > 1e-6 * max(1.0, abs(value)):
            continue
        before = text[max(0, m.start() - 2):m.start()]
        signs.add(-1 if "(" in before or before.rstrip().endswith(("-", "\u2212")) else 1)
    return -1 if signs == {-1} else 1 if signs == {1} else 0


def parse_value(raw) -> float | None:
    """-4 647 / (4 647) / "(10,25)" / "1,476.0" / "5 599 864" -> float, negative for parentheses or a minus."""
    if isinstance(raw, (int, float)):
        return float(raw)
    from .run_eval import parse_number
    text = str(raw).strip().replace("\u2212", "-").replace("$", "").replace("%", "")
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    token = re.search(r"\d[\d\s.,\u00a0\u202f]*", text)
    value = parse_number(token.group(0).strip()) if token else None
    return None if value is None else (-abs(value) if negative else value)

PROMPT = """You extract one indicator from passages of {company}'s annual report for fiscal {year}.
Indicator: {label}.
Take the consolidated figure FOR FISCAL {target} (the column of that year{comparative}). Read the unit
of the table ("en millions de dollars", "in thousands of US dollars"...). A loss is negative.
Return only JSON:
{{"found": true | false, "value": <number as written, negative for a loss, or null>,
  "unit": "units" | "thousands" | "millions" | "billions" | "percent" | "per_share",
  "currency": "CAD" | "USD" | null, "source": "S<number>", "line": "<row label as written>",
  "period": "<the period of the column, e.g. year ended December 31, 2021>"}}
If the passages do not give this indicator for fiscal {target}, return {{"found": false}}."""

# Default reference values, verified by hand in the PDFs (millions of the reporting currency).
DEFAULT_GOLD = [
    {"company": "aircanada", "year": 2019, "kpi": "net_income", "value_m": 1476},
    {"company": "aircanada", "year": 2020, "kpi": "net_income", "value_m": -4647},
    {"company": "aircanada", "year": 2021, "kpi": "net_income", "value_m": -3602},
    {"company": "aircanada", "year": 2020, "kpi": "revenue", "value_m": 5833},
    {"company": "aircanada", "year": 2021, "kpi": "revenue", "value_m": 6400},
    {"company": "aircanada", "year": 2021, "kpi": "total_assets", "value_m": 30614},
    {"company": "aircanada", "year": 2021, "kpi": "diluted_eps", "value_m": -10.25},
    {"company": "shopify", "year": 2017, "kpi": "revenue", "value_m": 673.304},
    {"company": "shopify", "year": 2018, "kpi": "revenue", "value_m": 1073.229},
    {"company": "shopify", "year": 2019, "kpi": "revenue", "value_m": 1578.173},
    {"company": "shopify", "year": 2020, "kpi": "revenue", "value_m": 2929.491},
    {"company": "shopify", "year": 2021, "kpi": "revenue", "value_m": 4611.856},
    {"company": "shopify", "year": 2022, "kpi": "revenue", "value_m": 5599.864},
]


def normalise(value: float, unit: str, kind: str) -> float | None:
    if kind == "amount":
        scale = SCALE_TO_MILLIONS.get(unit)
        return None if scale is None else value * scale
    return value                                     # per share, percent: as written


def _ask(company: str, year: int, kpi: str, report_year: int, hits: list[dict]) -> dict:
    """One LLM extraction on a set of passages, checked against them."""
    import ollama
    from .config import LLM_MODEL, OLLAMA_HOST
    from .run_eval import ask_json, numbers_in

    spec = KPIS[kpi]
    passages = "\n\n".join(f"[S{i}] {h['text'][:2000]}" for i, h in enumerate(hits, start=1))
    comparative = (f"; this is the {report_year} report, so fiscal {year} is the comparative column"
                   if report_year != year else "")
    prompt = PROMPT.format(company=COMPANY_NAMES[company], year=report_year, target=year, label=spec["label"],
                           comparative=comparative)
    row = {"company": company, "year": year, "kpi": kpi, "report_year": report_year, "status": "not_found"}
    try:
        data = ask_json(ollama.Client(host=OLLAMA_HOST), LLM_MODEL, prompt, f"PASSAGES:\n{passages}")
    except Exception as exc:
        return {**row, "status": "error", "detail": str(exc)[:120]}
    if not data.get("found") or data.get("value") is None:
        return row
    value = parse_value(data["value"])
    if value is None:
        return {**row, "status": "unparsed", "detail": str(data.get("value"))[:60]}
    source = str(data.get("source") or "")
    index = int(source[1:]) if source[1:].isdigit() else 0
    hit = hits[index - 1] if 1 <= index <= len(hits) else None
    line = str(data.get("line") or "")
    unit = str(data.get("unit") or "")
    period = str(data.get("period") or "")
    text = hit["text"] if hit else ""
    corrections = []
    # The sign and the unit are read from the passage itself when it states them: the LLM loses the
    # parentheses of a loss ((16,47) -> 16.47) and takes thousands for millions (CGI).
    sign = written_sign(value, text)
    if sign == -1 and value > 0:
        value, _ = -value, corrections.append("sign from the passage")
    stated = table_unit(text) if spec["kind"] == "amount" else None
    if stated and stated != unit and unit in SCALE_TO_MILLIONS:
        unit, _ = stated, corrections.append(f"unit from the passage: {stated}")
    value_m = normalise(value, unit, spec["kind"])
    problems = []
    # The figure must be written in the cited passage (years included: Shopify's 2024 net income is 2 019),
    # exactly, not only as a rounded figure of the text ("1,7 G$" for 1 673,2).
    numbers = numbers_in(text)
    if hit is None or not any(abs(abs(value) - n) <= 1e-6 * max(1.0, n) for n in numbers):
        problems.append("value not written as such in the cited passage")
    good_line, why = line_ok(kpi, line, text)
    if not good_line:
        problems.append(why)
    if PARTIAL_PERIOD_RE.search(f"{period} {line}"):
        problems.append(f"partial period '{period}'")
    if spec["kind"] == "amount" and SEGMENT_RE.search(f"{table_title(text)} {line}"):
        problems.append(f"a business segment, not the whole company ('{table_title(text)[:80]}')")
    if value_m is None:
        problems.append(f"unknown unit '{unit}'")
    return {**row, "status": "unverified" if problems else "verified",
            "detail": "; ".join(problems + corrections) or None,
            "value": value, "unit": unit, "value_m": value_m,
            "currency": data.get("currency") or CURRENCY.get(company, "CAD"),
            "line": line[:120], "period": str(data.get("period") or "")[:80],
            "source": hit["metadata"].get("source") if hit else None,
            "page": hit["metadata"].get("page") if hit else None}


RANK = {"verified": 3, "unverified": 2, "unparsed": 1, "not_found": 0, "error": 0}


def extract(company: str, year: int, kpi: str, report_year: int | None = None, k: int = 8,
            fallback: bool = True) -> dict:
    """Read one indicator for fiscal `year`: in the report of that year with two queries (the indicator,
    then the financial statement), then, if still not verified, in the comparative column of the next report."""
    from .retrieve import get_bm25, retrieve

    attempts = [(report_year or year, q) for q in ("indicator", "statement")]
    if fallback and report_year is None and year + 1 in YEARS:
        attempts += [(year + 1, "statement")]
    best = {"company": company, "year": year, "kpi": kpi, "report_year": report_year or year, "status": "not_found"}
    for rep_year, kind in attempts:
        lang = get_bm25().language(company, rep_year) or "fr"
        words = KPIS[kpi][lang] if kind == "indicator" else STATEMENT_QUERIES[kpi][lang]
        hits = retrieve(f"{words} {COMPANY_NAMES[company]} {year}", k=k, company=company, year=rep_year)
        if not hits:
            continue
        row = _ask(company, year, kpi, rep_year, hits)
        row["attempt"] = f"{rep_year}:{kind}"
        if RANK[row["status"]] > RANK[best["status"]]:
            best = row
        if best["status"] == "verified":
            break
    return best


def confidence(row: dict) -> str:
    """high: verified and read the same in the next report; medium: verified, nothing to compare with (the
    latest year) or not found in the next report; review: the two reports differ (a restatement, a stock
    split, or a reading error: check the PDF); low: not verified."""
    if row["status"] != "verified":
        return "low"
    if row.get("crosscheck") == "consistent":
        return "high"
    if row.get("crosscheck") == "differs":
        return "review"
    return "medium"


def series_checks(rows: list[dict]) -> None:
    """Consistency of each company's series of years, for amounts (in place).

    - scale: a value about 1 000 times the median of the other years was read in thousands as millions
      (CGI: 1 141 402 for 1 141.4): divided by 1 000.
    - outlier: a revenue or total assets more than 3 times away from both its neighbouring years
      (Couche-Tard revenue 14 666 between years of 50 000+), or, for the first or latest year, more than 5
      times away from its only neighbour. A real shock (Air Canada 2020) stays close to one neighbour.
    Both lower the confidence to "review".
    """
    import math
    series: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r.get("value_m") and KPIS[r["kpi"]]["kind"] == "amount" and r.get("status") != "manual":
            series.setdefault((r["company"], r["kpi"]), []).append(r)
    for items in series.values():
        items.sort(key=lambda r: r["year"])
        sizes = sorted(math.log10(abs(r["value_m"])) for r in items)
        if len(sizes) < 3:
            continue
        median = sizes[len(sizes) // 2]
        for r in items:
            if 2.5 <= math.log10(abs(r["value_m"])) - median <= 3.5:
                r["value_m_raw"] = r["value_m"]
                r["value_m"] /= 1000
                r["detail"] = "; ".join(filter(None, [r.get("detail"), "rescaled /1000 (series)"]))
                r["confidence"] = "review" if r.get("confidence") in {"high", "medium"} else r.get("confidence")
        for i, r in enumerate(items):
            if r["kpi"] not in {"revenue", "total_assets"}:
                continue
            ratios = [abs(r["value_m"]) / abs(items[j]["value_m"]) for j in (i - 1, i + 1) if 0 <= j < len(items)]
            # Inside the series: away from both neighbours; at an end (first or latest year): very far from
            # its only neighbour (Couche-Tard 2025 revenue 1 040.5 after 69 263.5).
            if (len(ratios) == 2 and all(q > 3 or q < 1 / 3 for q in ratios)) or \
                    (len(ratios) == 1 and (ratios[0] > 5 or ratios[0] < 1 / 5)):
                r["detail"] = "; ".join(filter(None, [r.get("detail"), "outlier vs neighbouring years"]))
                r["confidence"] = "review" if r.get("confidence") in {"high", "medium"} else r.get("confidence")


def apply_gold(rows: list[dict], gold: list[dict]) -> None:
    """Values verified by hand replace the extraction (confidence "manual"); the extracted value is kept."""
    by_key = {(r["company"], r["year"], r["kpi"]): r for r in rows}
    for g in gold:
        key = (g["company"], g["year"], g["kpi"])
        r = by_key.get(key)
        if r is None:
            r = {"company": g["company"], "year": g["year"], "kpi": g["kpi"], "status": "not_found"}
            rows.append(r)
            by_key[key] = r
        r["value_m_extracted"] = r.get("value_m")
        r.update(value_m=g["value_m"], status="manual", confidence="manual",
                 currency=g.get("currency") or CURRENCY.get(g["company"], "CAD"),
                 source=g.get("source") or r.get("source"), page=g.get("page") or r.get("page"))


def close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def load_gold() -> list[dict]:
    if not GOLD_FILE.exists():
        GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOLD_FILE.write_text("".join(json.dumps(g) + "\n" for g in DEFAULT_GOLD), encoding="utf-8")
        print(f"Reference values written to {GOLD_FILE} (add the values you verify).")
    return [json.loads(line) for line in GOLD_FILE.open(encoding="utf-8") if line.strip()]


def score_gold(rows: list[dict], gold: list[dict]) -> None:
    by_key = {(r["company"], r["year"], r["kpi"]): r for r in rows}
    counts = {"correct": 0, "wrong": 0, "unverified": 0, "missing": 0}
    for g in gold:
        r = by_key.get((g["company"], g["year"], g["kpi"]))
        if r is None or r["status"] in {"not_found", "error", "unparsed"}:
            outcome = "missing"
        elif r.get("value_m") is not None and close(r["value_m"], g["value_m"]):
            outcome = "correct" if r["status"] == "verified" else "unverified"
        else:
            outcome = "wrong"
        counts[outcome] += 1
        if outcome != "correct":
            got = None if r is None else r.get("value_m")
            print(f"  {outcome:<10} {g['company']} {g['year']} {g['kpi']}: got {got}, expected {g['value_m']}"
                  + (f" | {r.get('line')} ({r.get('source')} p.{r.get('page')}) {r.get('detail') or ''}" if r else ""))
    n = len(gold)
    print(f"Reference values: {counts['correct']}/{n} correct and verified, {counts['unverified']} correct but "
          f"not found in the cited passage, {counts['wrong']} wrong, {counts['missing']} missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract key indicators from the annual reports.")
    parser.add_argument("--companies", nargs="*", default=list(KPIS_BY_COMPANY))
    parser.add_argument("--years", nargs="*", type=int, default=list(YEARS))
    parser.add_argument("--kpis", nargs="*", choices=list(KPIS))
    parser.add_argument("--crosscheck", action="store_true", help="also read each value in the next report")
    parser.add_argument("--gold", action="store_true", help="extract and score only the reference values")
    parser.add_argument("--finalize", action="store_true",
                        help="no extraction: re-apply the series checks and the hand-verified values to kpis.jsonl")
    args = parser.parse_args()

    gold = load_gold()
    previous = ([json.loads(line) for line in OUT_JSONL.open(encoding="utf-8") if line.strip()]
                if OUT_JSONL.exists() else [])
    if args.finalize:
        finish([], gold, previous=previous)
        return
    if args.gold:
        tasks = [(g["company"], g["year"], g["kpi"]) for g in gold]
    else:
        tasks = [(c, y, kpi) for c in args.companies for y in args.years
                 for kpi in KPIS_BY_COMPANY[c] if not args.kpis or kpi in args.kpis]
    rows, start = [], time.perf_counter()
    for i, (company, year, kpi) in enumerate(tasks, start=1):
        row = extract(company, year, kpi)
        if args.crosscheck and year + 1 in YEARS and row.get("value_m") is not None:
            later = extract(company, year, kpi, report_year=year + 1, fallback=False)
            if later.get("value_m") is None:
                row["crosscheck"] = "missing"
            else:
                row["crosscheck"] = "consistent" if close(row["value_m"], later["value_m"]) else "differs"
                row["value_m_next_report"] = later["value_m"]
        row["confidence"] = confidence(row)
        rows.append(row)
        print(f"[{i}/{len(tasks)}] {company} {year} {kpi}: {row['status']} {row.get('value_m')} "
              f"{row.get('crosscheck', '')} -> {row.get('confidence', '')}", flush=True)

    print(f"\nExtracted in {time.perf_counter() - start:.0f} s")
    # Score the raw extraction against the reference values, before they replace it.
    done = {(r["company"], r["year"], r["kpi"]) for r in rows}
    in_scope = [g for g in gold if (g["company"], g["year"], g["kpi"]) in done]
    if in_scope:
        score_gold(rows, in_scope)
    finish(rows, gold, previous=previous)


FIELDS = ["company", "year", "kpi", "confidence", "status", "value_m", "currency", "value", "unit", "line", "period",
          "source", "page", "report_year", "attempt", "detail", "crosscheck", "value_m_next_report",
          "value_m_extracted"]


def finish(rows: list[dict], gold: list[dict], previous: list[dict]) -> None:
    """Merge with the previous extraction (a partial run replaces only its own cells), check the series of
    years, apply the hand-verified values, and write the table and the list of values to check."""
    keys = {(r["company"], r["year"], r["kpi"]) for r in rows}
    rows = [r for r in previous if (r["company"], r["year"], r["kpi"]) not in keys] + rows
    for r in rows:
        if r.get("status") == "manual":                # back to the extraction before re-applying the checks
            if r.get("status_extracted") is None:
                r["status"] = "not_found"
            else:
                r["value_m"], r["status"] = r.get("value_m_extracted"), r["status_extracted"]
        if r.get("value_m_raw") is not None:           # undo a previous series rescaling
            r["value_m"] = r.pop("value_m_raw")
        r["status_extracted"] = r.get("status")
        r["detail"] = "; ".join(d for d in (r.get("detail") or "").split("; ")
                                if d and d not in {"rescaled /1000 (series)", "outlier vs neighbouring years"}) or None
        r["confidence"] = confidence(r)
    series_checks(rows)
    apply_gold(rows, gold)
    rows.sort(key=lambda r: (r["company"], r["year"], r["kpi"]))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    levels = {c: sum(r.get("confidence") == c for r in rows) for c in ("manual", "high", "medium", "review", "low")}
    print(f"{len(rows)} values -> {OUT_CSV}")
    print(f"Confidence: {levels}")
    review = [r for r in rows if r.get("confidence") in {"review", "low"}]
    with (PROCESSED_DIR / "kpis_review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", delimiter=";")
        writer.writeheader()
        writer.writerows(review)
    print(f"{len(review)} values to check by hand -> {PROCESSED_DIR / 'kpis_review.csv'} "
          "(add the verified values to eval/kpi_gold.jsonl, then run --finalize)")


if __name__ == "__main__":
    main()
