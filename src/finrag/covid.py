"""Phase D3: the COVID-19 shock (fiscal 2019 -> 2020 -> 2021) across the six companies.

The figures come from the KPI table of phase D2 (data/processed/kpis.jsonl): verified, cross-checked or
entered by hand. Values with a "low" confidence are left out, and values to review are marked "?".
The causes come from the RAG itself (--explain): one question per company, filtered on the company and its
COVID fiscal year, answered with citations, then checked by the verifier of phase D1.

Fiscal years do not cover the same months. The pandemic starts in March 2020:
  - Air Canada, Desjardins, Shopify (December 31): fiscal 2020 has about 10 months of COVID;
  - National Bank (October 31): 8 months; CGI (September 30): 7 months;
  - Couche-Tard (late April): fiscal 2020 ends on April 26, 2020 with only 6 weeks of COVID, so its COVID
    year is fiscal 2021 (May 2020 - April 2021).
Each company is compared on its own COVID year against fiscal 2019, the last year before the pandemic.

    python -m finrag.covid               # table, CSV and HTML report from kpis.jsonl
    python -m finrag.covid --explain     # also ask the RAG why (6 questions, about 1-2 minutes)

Output: data/processed/covid_study.csv, covid_study.html, covid_explanations.jsonl (kept between runs).
"""
import argparse
import csv
import html
import json

from .config import PROCESSED_DIR

KPIS_FILE = PROCESSED_DIR / "kpis.jsonl"
OUT_CSV = PROCESSED_DIR / "covid_study.csv"
OUT_HTML = PROCESSED_DIR / "covid_study.html"
EXPLANATIONS = PROCESSED_DIR / "covid_explanations.jsonl"

COMPANIES = {          # name, fiscal year end, months of COVID in fiscal 2020, COVID year, currency
    "aircanada": ("Air Canada", "31 décembre", 10, 2020, "CAD"),
    "banquenationale": ("Banque Nationale", "31 octobre", 8, 2020, "CAD"),
    "cgi": ("CGI", "30 septembre", 7, 2020, "CAD"),
    "couchetard": ("Couche-Tard", "fin avril", 1.5, 2021, "USD"),
    "desjardins": ("Desjardins", "31 décembre", 10, 2020, "CAD"),
    "shopify": ("Shopify", "31 décembre", 10, 2020, "USD"),
}
COLORS = {"aircanada": "#d62728", "banquenationale": "#1f77b4", "cgi": "#9467bd",
          "couchetard": "#ff7f0e", "desjardins": "#2ca02c", "shopify": "#17becf"}
BASE_YEAR = 2019
CHART_YEARS = list(range(2018, 2023))
KPIS = ("revenue", "net_income", "diluted_eps", "total_assets", "cet1_ratio")


# ------------------------------------------------------------------ data ---

def load_values() -> dict:
    """{(company, year, kpi): {"value": float, "confidence": str, ...}}; low-confidence values are left out."""
    values = {}
    for line in KPIS_FILE.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("value_m") in (None, "") or r.get("confidence") in (None, "low"):
            continue
        values[(r["company"], int(r["year"]), r["kpi"])] = {
            "value": float(r["value_m"]), "confidence": r["confidence"],
            "source": r.get("source"), "page": r.get("page")}
    return values


def get(values, company, year, kpi):
    v = values.get((company, year, kpi))
    return v["value"] if v else None


def pct(new, old):
    if new is None or old is None or old <= 0:      # a change from a loss has no meaningful percentage
        return None
    return (new - old) / old * 100


def margin(values, company, year):
    rev, ni = get(values, company, year, "revenue"), get(values, company, year, "net_income")
    return ni / rev * 100 if rev and ni is not None else None


def shock(change):
    if change is None:
        return "n.d."
    if change <= -30:
        return "choc sévère"
    if change <= -5:
        return "recul"
    if change < 5:
        return "stable"
    return "croissance"


def recovery_year(values, company, covid_year):
    """First fiscal year from the COVID year on whose revenue is back at the 2019 level."""
    base = get(values, company, BASE_YEAR, "revenue")
    if base is None:
        return None
    for year in range(covid_year, 2026):
        rev = get(values, company, year, "revenue")
        if rev is not None and rev >= base:
            return year
    return None


def study(values) -> list[dict]:
    rows = []
    for c, (name, fy_end, months, covid_year, currency) in COMPANIES.items():
        rev = {y: get(values, c, y, "revenue") for y in (BASE_YEAR, 2020, 2021, covid_year + 1)}
        ni = {y: get(values, c, y, "net_income") for y in (BASE_YEAR, 2020, 2021, covid_year + 1)}
        rev_change = pct(rev[covid_year], rev[BASE_YEAR])
        flags = sorted({f"{y} {k}" for y in (BASE_YEAR, 2020, 2021, covid_year + 1)
                        for k in ("revenue", "net_income")
                        if (values.get((c, y, k)) or {}).get("confidence") == "review"})
        back = recovery_year(values, c, covid_year)
        rows.append({
            "company": c, "name": name, "currency": currency, "fiscal_year_end": fy_end,
            "covid_months_fy2020": months, "covid_year": covid_year,
            "revenue_2019": rev[BASE_YEAR], "revenue_2020": rev[2020], "revenue_2021": rev[2021],
            "revenue_change_covid_year_pct": rev_change,
            "revenue_change_next_year_pct": pct(rev[covid_year + 1], rev[covid_year]),
            "net_income_2019": ni[BASE_YEAR], "net_income_2020": ni[2020], "net_income_2021": ni[2021],
            "net_income_change_covid_year_pct": pct(ni[covid_year], ni[BASE_YEAR]),
            "net_margin_2019": margin(values, c, BASE_YEAR), "net_margin_2020": margin(values, c, 2020),
            "net_margin_2021": margin(values, c, 2021),
            "shock": shock(rev_change),
            "revenue_back_to_2019": ("pas de recul" if rev_change is not None and rev_change >= 0
                                     else back or "pas encore"),
            "to_review": ", ".join(flags),
        })
    return rows


# ---------------------------------------------------------------- report ---

def fmt(v, digits=0, suffix=""):
    if v is None:
        return "—"
    s = f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−") + suffix


def fmt_pct(v):
    return "—" if v is None else ("+" if v > 0 else "") + fmt(v, 1, " %")


def print_table(rows):
    print(f"{'Entreprise':<17}{'Année COVID':>12}{'Revenus':>10}{'Rés. net':>10}{'Marge 19':>10}"
          f"{'Marge COVID':>12}  Choc / retour au niveau 2019")
    for r in rows:
        print(f"{r['name']:<17}{r['covid_year']:>12}{fmt_pct(r['revenue_change_covid_year_pct']):>10}"
              f"{fmt_pct(r['net_income_change_covid_year_pct']):>10}{fmt(r['net_margin_2019'], 1, ' %'):>10}"
              f"{fmt(r['net_margin_%d' % r['covid_year']], 1, ' %'):>12}  {r['shock']} / {r['revenue_back_to_2019']}"
              + (f"  (à revoir : {r['to_review']})" if r["to_review"] else ""))


def svg_revenue_index(values) -> str:
    """Revenue indexed to fiscal 2019 = 100, one line per company."""
    W, H, L, R, T, B = 640, 300, 44, 16, 14, 30
    series = {}
    for c in COMPANIES:
        base = get(values, c, BASE_YEAR, "revenue")
        if base:
            series[c] = [(y, get(values, c, y, "revenue") / base * 100) for y in CHART_YEARS
                         if get(values, c, y, "revenue") is not None]
    top = max([v for s in series.values() for _, v in s] + [120])
    top = (int(top // 50) + 1) * 50
    x = lambda y: L + (y - CHART_YEARS[0]) / (CHART_YEARS[-1] - CHART_YEARS[0]) * (W - L - R)
    yy = lambda v: T + (1 - v / top) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Revenus indexés, 2019 = 100">']
    for v in range(0, top + 1, 50):
        cls = "base" if v == 100 else "grid"
        out.append(f'<line class="{cls}" x1="{L}" x2="{W - R}" y1="{yy(v):.1f}" y2="{yy(v):.1f}"/>'
                   f'<text class="tick" x="{L - 6}" y="{yy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for y in CHART_YEARS:
        out.append(f'<text class="tick" x="{x(y):.1f}" y="{H - 8}" text-anchor="middle">{y}</text>')
    out.append(f'<rect class="covid" x="{x(2020) - 14:.1f}" y="{T}" width="{x(2021) - x(2020) + 28:.1f}" '
               f'height="{H - T - B}"/>')
    for c, pts in series.items():
        d = " ".join(f"{'M' if i == 0 else 'L'}{x(y):.1f},{yy(v):.1f}" for i, (y, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{COLORS[c]}" stroke-width="2.5"/>')
        for y, v in pts:
            out.append(f'<circle cx="{x(y):.1f}" cy="{yy(v):.1f}" r="3.5" fill="{COLORS[c]}">'
                       f'<title>{COMPANIES[c][0]} {y} : {v:.0f}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def svg_margins(values) -> str:
    """Net margin of fiscal 2019, 2020 and 2021, grouped by company."""
    years = (2019, 2020, 2021)
    W, H, L, R, T, B = 640, 300, 44, 16, 14, 30
    data = {c: [margin(values, c, y) for y in years] for c in COMPANIES}
    vals = [v for vs in data.values() for v in vs if v is not None] + [0, 10]
    lo, hi = (int(min(vals) // 10) - 1) * 10, (int(max(vals) // 10) + 1) * 10
    yy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    group = (W - L - R) / len(COMPANIES)
    bw = group * 0.8 / len(years)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Marge nette par exercice">']
    step = 20 if hi - lo > 80 else 10
    for v in range((lo // step) * step, hi + 1, step):
        if v < lo:
            continue
        out.append(f'<line class="{"base" if v == 0 else "grid"}" x1="{L}" x2="{W - R}" y1="{yy(v):.1f}" '
                   f'y2="{yy(v):.1f}"/><text class="tick" x="{L - 6}" y="{yy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for i, (c, vs) in enumerate(data.items()):
        x0 = L + i * group + group * 0.1
        for j, v in enumerate(vs):
            if v is None:
                continue
            top, bottom = (yy(v), yy(0)) if v >= 0 else (yy(0), yy(v))
            out.append(f'<rect x="{x0 + j * bw:.1f}" y="{top:.1f}" width="{bw - 2:.1f}" height="{max(bottom - top, 1):.1f}" '
                       f'fill="{COLORS[c]}" opacity="{(0.45, 1, 0.7)[j]}"><title>{COMPANIES[c][0]} {years[j]} : '
                       f'{fmt(v, 1, " %")}</title></rect>')
        out.append(f'<text class="tick" x="{x0 + group * 0.4:.1f}" y="{H - 8}" text-anchor="middle">{COMPANIES[c][0]}</text>')
    out.append("</svg>")
    return "".join(out)


def amount(v, currency, start=False):
    if v is None:
        return "?"
    word = "une perte de" if v < 0 else "un bénéfice de"
    word = ("d'" if start else "à ") + word
    return f"{word} {fmt(abs(v), 1)} M$ {'US' if currency == 'USD' else ''}".strip()


def questions(r, values) -> list[tuple[str, str]]:
    """Two questions per company:
    impact   the general question: which passages talk about COVID in the COVID year;
    drivers  a question built from the table: why did net income move like this? The year-over-year change
             is what the MD&A of the report explains, so the COVID year is compared with the year before.
    The general question finds the passages that name the pandemic, but the causes of the figures (fuel
    margins, provisions, restructuring) are often written without the word COVID: the second question
    searches for them with the figures of the table."""
    c, cy, cur = r["company"], r["covid_year"], r["currency"]
    impact = (f"Quel a été l'impact de la pandémie de COVID-19 sur les activités et les résultats de {r['name']} "
              f"au cours de l'exercice {cy} ? Donne les principales causes.")
    ni_before, ni_now = get(values, c, cy - 1, "net_income"), get(values, c, cy, "net_income")
    rev_change = pct(get(values, c, cy, "revenue"), get(values, c, cy - 1, "revenue"))
    if ni_before is None or ni_now is None:
        return [("impact", impact)]
    drivers = (f"Pourquoi le résultat net de {r['name']} est-il passé {amount(ni_before, cur, True)} pour l'exercice "
               f"{cy - 1} {amount(ni_now, cur)} pour l'exercice {cy}"
               + (f", alors que les revenus ont varié de {fmt_pct(rev_change)}" if rev_change is not None else "")
               + " ? Explique les principaux écarts entre les deux exercices.")
    return [("impact", impact), ("drivers", drivers)]


def explain(rows, values, k: int = 10) -> list[dict]:
    """Ask the RAG why, for each company and its COVID year; check the figures of each answer."""
    from .generate import answer
    from .verify import verify
    out = []
    for r in rows:
        for kind, q in questions(r, values):
            print(f"  {r['name']} ({kind}) ...", flush=True)
            res = answer(q, k=k, company=r["company"], year=r["covid_year"])
            report = verify(res["answer"], res["sources"], q)
            out.append({"company": r["company"], "kind": kind, "question": q, "answer": res["answer"],
                        "unsupported": report["unsupported"], "other_source": report["other_source"],
                        "sources": [{"n": i, "source": h["metadata"].get("source"), "page": h["metadata"].get("page"),
                                     "year": h["metadata"].get("year")} for i, h in enumerate(res["sources"], start=1)]})
    with EXPLANATIONS.open("w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return out


def load_explanations() -> list[dict]:
    if not EXPLANATIONS.exists():
        return []
    return [json.loads(line) for line in EXPLANATIONS.open(encoding="utf-8") if line.strip()]


def findings(rows) -> list[str]:
    """Deterministic summary sentences, from the table only."""
    out = []
    by = {r["company"]: r for r in rows}
    groups = {}
    for r in rows:
        groups.setdefault(r["shock"], []).append(f"{r['name']} ({fmt_pct(r['revenue_change_covid_year_pct'])})")
    for label in ("choc sévère", "recul", "stable", "croissance"):
        if label in groups:
            out.append(f"<b>{label.capitalize()}</b> des revenus pendant l'année COVID : {', '.join(groups[label])}.")
    losses = [r["name"] for r in rows if (r[f"net_income_{r['covid_year']}"] or 0) < 0 <= (r["net_income_2019"] or 0)]
    if losses:
        out.append(f"Passage d'un bénéfice à une perte : {', '.join(losses)}.")
    gains = [r["name"] for r in rows if (r[f"net_income_{r['covid_year']}"] or 0) > 0 > (r["net_income_2019"] or 0)]
    if gains:
        out.append(f"Passage d'une perte à un bénéfice : {', '.join(gains)}.")
    ct = by.get("couchetard")
    if ct and ct["revenue_change_covid_year_pct"] is not None and (ct["net_income_change_covid_year_pct"] or 0) > 0:
        out.append("Couche-Tard : revenus en baisse mais bénéfice en hausse. Les revenus suivent le prix du "
                   "carburant à la pompe ; la marge par litre, elle, a augmenté (voir l'explication).")
    return out


def write_html(rows, values, explanations) -> None:
    head = ("<tr><th>Entreprise</th><th>Fin d'exercice</th><th>Mois COVID<br>en 2020</th><th>Année COVID</th>"
            "<th>Revenus 2019</th><th>Revenus année COVID</th><th>Variation</th><th>Résultat net 2019</th>"
            "<th>Résultat net année COVID</th><th>Marge 2019</th><th>Marge COVID</th><th>Revenus de retour au niveau 2019</th></tr>")
    body = []
    for r in rows:
        cy = r["covid_year"]
        mark = lambda kpi, y: " ?" if (values.get((r["company"], y, kpi)) or {}).get("confidence") == "review" else ""
        cls = "neg" if (r["revenue_change_covid_year_pct"] or 0) < 0 else "pos"
        body.append(
            f"<tr><td><span class='dot' style='background:{COLORS[r['company']]}'></span>{r['name']}"
            f"<small> ({r['currency']})</small></td><td>{r['fiscal_year_end']}</td>"
            f"<td class='n'>{fmt(r['covid_months_fy2020'], 1 if r['covid_months_fy2020'] % 1 else 0)}</td><td class='n'>{cy}</td>"
            f"<td class='n'>{fmt(r['revenue_2019'])}{mark('revenue', 2019)}</td>"
            f"<td class='n'>{fmt(r['revenue_%d' % cy])}{mark('revenue', cy)}</td>"
            f"<td class='n {cls}'>{fmt_pct(r['revenue_change_covid_year_pct'])}</td>"
            f"<td class='n'>{fmt(r['net_income_2019'])}{mark('net_income', 2019)}</td>"
            f"<td class='n'>{fmt(r['net_income_%d' % cy])}{mark('net_income', cy)}</td>"
            f"<td class='n'>{fmt(r['net_margin_2019'], 1, ' %')}</td><td class='n'>{fmt(r['net_margin_%d' % cy], 1, ' %')}</td>"
            f"<td class='n'>{r['revenue_back_to_2019']}</td></tr>")
    legend = "".join(f"<span><i style='background:{COLORS[c]}'></i>{n[0]}</span>" for c, n in COMPANIES.items())
    expl = []
    titles = {"impact": "Ce que le rapport dit de la pandémie", "drivers": "Ce qui explique la variation du résultat net"}
    for r in rows:
        parts = []
        for e in (e for e in explanations if e["company"] == r["company"]):
            refs = ", ".join(f"S{s['n']} {s['source']} p.{s['page']}" for s in e["sources"])
            warn = (f"<p class='warn'>Chiffres absents des sources : {', '.join(map(html.escape, e['unsupported']))}</p>"
                    if e["unsupported"] else "<p class='ok'>Tous les chiffres cités se trouvent dans les sources.</p>")
            parts.append(f"<h4>{titles.get(e.get('kind', 'impact'))}</h4><p class='refs'>{html.escape(e['question'])}</p>"
                         f"<div class='answer'>{html.escape(e['answer']).replace(chr(10), '<br>')}</div>{warn}"
                         f"<p class='refs'>{html.escape(refs)}</p>")
        if parts:
            expl.append(f"<details><summary><span class='dot' style='background:{COLORS[r['company']]}'></span>"
                        f"{r['name']} — exercice {r['covid_year']}</summary>{''.join(parts)}</details>")
    expl_html = ("".join(expl) if expl else
                 "<p class='muted'>Lancez <code>python -m finrag.covid --explain</code> pour ajouter les causes, "
                 "tirées des rapports avec citations.</p>")
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FinRAG — étude COVID</title>
<style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e3e3e3;--card:#f7f7f5;--neg:#b42318;--pos:#067647}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#eee;--muted:#9a9a9a;--line:#333;--card:#1f1f1f;--neg:#f97066;--pos:#47cd89}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1040px;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:32px 0 8px}} .muted,small{{color:var(--muted)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}} th{{font-weight:600;font-size:12.5px;color:var(--muted)}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} .neg{{color:var(--neg)}} .pos{{color:var(--pos)}}
.dot,.legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;margin:6px 0}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}
.card{{background:var(--card);border-radius:10px;padding:14px}} .card h3{{margin:0 0 6px;font-size:14px}}
svg{{width:100%;height:auto}} .grid{{stroke:var(--line)}} .base{{stroke:var(--muted);stroke-dasharray:3 3}}
.tick{{fill:var(--muted);font-size:11px}} .covid{{fill:#ff000012}}
ul.find li{{margin:4px 0}} details{{background:var(--card);border-radius:10px;padding:10px 14px;margin:8px 0}}
summary{{cursor:pointer;font-weight:600}} details h4{{margin:14px 0 2px;font-size:14px}} .answer{{margin:10px 0}} .refs{{font-size:12px;color:var(--muted)}}
.warn{{color:var(--neg);font-size:13px}} .ok{{color:var(--pos);font-size:13px}}
</style></head><body>
<h1>Le choc de la COVID-19 sur six entreprises canadiennes</h1>
<p class="muted">Exercices 2019 à 2021, tirés des rapports annuels par FinRAG (phase D2 : valeurs vérifiées,
recoupées avec le rapport suivant ou saisies à la main). Montants en millions, dans la devise de publication.
« ? » : valeur à revoir.</p>
<h2>Constats</h2><ul class="find">{''.join(f'<li>{f}</li>' for f in findings(rows))}</ul>
<div class="charts">
<div class="card"><h3>Revenus, indice 2019 = 100</h3><div class="legend">{legend}</div>{svg_revenue_index(values)}
<p class="muted" style="font-size:12px">Bande rouge : exercices 2020-2021.</p></div>
<div class="card"><h3>Marge nette (résultat net / revenus), %</h3><div class="legend">{legend}</div>{svg_margins(values)}
<p class="muted" style="font-size:12px">Par entreprise : 2019 (clair), 2020 (plein), 2021 (moyen).</p></div>
</div>
<h2>Chaque entreprise sur son année COVID</h2><div class="wrap"><table>{head}{''.join(body)}</table></div>
<h2>Pourquoi : ce qu'en disent les rapports</h2>{expl_html}
<h2>Méthode et limites</h2><ul>
<li>L'année COVID de chaque entreprise est l'exercice qui contient le plus de mois de pandémie (à partir de mars 2020) :
2020, sauf Couche-Tard, dont l'exercice 2020 se termine le 26 avril 2020 (2021 pour elle).</li>
<li>Le résultat net est celui de l'état des résultats, part des actionnaires minoritaires comprise.</li>
<li>Pour les banques et Desjardins, les « revenus » sont le revenu total (ou le total des produits) ; ceux de Desjardins
incluent les primes d'assurance et les revenus de placement, qui varient avec les marchés.</li>
<li>Couche-Tard et Shopify publient en dollars américains ; les autres en dollars canadiens. Les variations en %
se comparent, les montants non.</li>
<li>Une variation en % n'a pas de sens à partir d'une perte (Shopify 2019) : elle est laissée vide.</li>
</ul></body></html>"""
    OUT_HTML.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="COVID-19 study from the KPI table.")
    parser.add_argument("--explain", action="store_true", help="ask the RAG for the causes (one question per company)")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    values = load_values()
    rows = study(values)
    print_table(rows)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    if args.explain:
        print("\nCauses, from the reports:")
        explanations = explain(rows, values, args.k)
    else:
        explanations = load_explanations()
    write_html(rows, values, explanations)
    print(f"\n-> {OUT_CSV}\n-> {OUT_HTML}" + (f"\n-> {EXPLANATIONS}" if args.explain else ""))


if __name__ == "__main__":
    main()
