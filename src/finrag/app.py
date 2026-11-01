"""Phase E: the FinRAG demo (Streamlit). It only calls the API: start the API first.

    pip install "streamlit>=1.50"           # once
    python -m finrag.api                    # terminal 1
    streamlit run src/finrag/app.py         # terminal 2 -> http://localhost:8501

FINRAG_API_URL changes the address of the API (default http://localhost:8000).
"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("FINRAG_API_URL", "http://localhost:8000").rstrip("/")
NAMES = {"aircanada": "Air Canada", "banquenationale": "Banque Nationale", "cgi": "CGI",
         "couchetard": "Couche-Tard", "desjardins": "Desjardins", "shopify": "Shopify"}
CURRENCY = {"couchetard": "USD", "shopify": "USD"}
KPI_LABELS = {"revenue": "Revenus", "net_income": "Résultat net", "diluted_eps": "Résultat par action dilué",
              "total_assets": "Total de l'actif", "cet1_ratio": "Ratio CET1"}
KPI_UNITS = {"diluted_eps": "$ par action", "cet1_ratio": "%"}
CONFIDENCE = ["manual", "high", "medium", "review", "low"]
EXAMPLES = ["Quel était le résultat net de la Banque Nationale en 2024 ?",
            "How did Air Canada's operating revenues change from 2019 to 2021?",
            "Quel est le ratio CET1 de Desjardins et de la Banque Nationale en 2023 ?",
            "What was Royal Bank of Canada's net income in 2022?"]

st.set_page_config(page_title="FinRAG", page_icon="📊", layout="wide")


# ------------------------------------------------------------------- API ---

def api_get(path: str, **params):
    r = requests.get(f"{API}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API}{path}", json=payload, timeout=600)
    if r.status_code != 200:
        detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def api_ask(question: str, k: int, company, year, verify: bool) -> dict:
    """One question, no memory, RAG only (company or year forced in the sidebar)."""
    return api_post("/ask", {"question": question, "k": k, "company": company, "year": year, "verify": verify})


def api_chat(question: str, k: int, verify: bool) -> dict:
    """One turn of the conversation: the agent routes it and remembers the session."""
    res = api_post("/chat", {"question": question, "k": k, "verify": verify,
                             "session_id": st.session_state.get("session_id")})
    st.session_state.session_id = res["session_id"]
    return res


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict | None:
    try:
        return api_get("/health")
    except requests.RequestException:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def kpis(min_confidence: str) -> pd.DataFrame:
    return pd.DataFrame(api_get("/kpis", min_confidence=min_confidence)["rows"])


@st.cache_data(ttl=300, show_spinner=False)
def covid() -> dict:
    return api_get("/covid")


def md(text: str) -> str:
    """Markdown without LaTeX: Streamlit reads "1 902 $ et 1 734 $" as a formula between two dollar signs."""
    return (text or "").replace("$", "\\$")


def fmt(v, digits=1, suffix="", sign=False):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return ("+" if sign and v > 0 else "") + f"{v:,.{digits}f}".replace(",", " ").replace(".", ",") + suffix


# --------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.title("📊 FinRAG")
    st.caption("Rapports annuels de six entreprises canadiennes, 2017-2025. Modèle local, réponses citées.")
    h = health()
    if h is None:
        st.error(f"API injoignable à {API}. Lancez : `python -m finrag.api`")
        st.stop()
    if h["status"] == "ok":
        st.success(f"Système prêt · {h['llm_model']} · {h['passages']:,} passages".replace(",", " "))
    else:
        problems = []
        if not h.get("ollama_ok"):
            problems.append("Ollama ne répond pas")
        if h.get("models_missing"):
            problems.append("modèles absents : " + ", ".join(h["models_missing"]))
        if not h.get("passages"):
            problems.append("index vide")
        st.warning("Système dégradé : " + "; ".join(problems))

    st.subheader("Recherche")
    k = st.slider("Passages donnés au modèle", 3, 20, 10)
    company = st.selectbox("Entreprise", [None, *NAMES], format_func=lambda c: "Détectée dans la question" if c is None else NAMES[c])
    year = st.selectbox("Exercice", [None, *range(2017, 2026)], format_func=lambda y: "Détecté dans la question" if y is None else str(y))
    verify = st.checkbox("Vérifier les chiffres de la réponse", value=True)
    if company or year:
        st.caption("Entreprise ou exercice imposé : chaque question est traitée seule, par la recherche dans "
                   "les rapports, sans mémoire de la conversation.")
    if st.button("Nouvelle conversation", width="stretch"):
        if sid := st.session_state.pop("session_id", None):
            try:
                requests.delete(f"{API}/chat/{sid}", timeout=10)
            except requests.RequestException:
                pass
        st.session_state.messages = []
        st.rerun()

tab_ask, tab_kpis, tab_covid = st.tabs(["💬 Questions", "📈 Indicateurs", "🦠 Étude COVID-19"])


# ------------------------------------------------------------- questions ---

INHERITED = {"companies": "l'entreprise", "years": "l'exercice", "kpi": "l'indicateur"}
REASONS = {"company outside the corpus": "entreprise hors du corpus",
           "a measure the table does not hold": "mesure absente du tableau des indicateurs",
           "no indicator of the table": "pas un indicateur du tableau",
           "an explanation or a ranking": "explication ou classement",
           "the indicator of the previous question, but another question": "autre question sur le même indicateur",
           "no company": "aucune entreprise"}
CONFIDENCE_FR = {"manual": "vérifiée à la main", "high": "confirmée", "medium": "vérifiée", "review": "à revoir"}


def reason_fr(reason: str) -> str:
    if reason.startswith("not in the table or not verified: "):
        return "valeur absente du tableau ou à revoir : " + reason.split(": ", 1)[1]
    return REASONS.get(reason, reason)


def show_answer(res: dict) -> None:
    st.markdown(md(res["answer"]))
    v = res.get("verification")
    plan = res.get("plan") or {}
    badges = []
    if res.get("route") == "table":
        badges.append("📋 Lu dans le tableau des indicateurs vérifiés (sans recherche ni modèle)")
    elif res.get("route") == "rag":
        badges.append("📄 Recherche dans les rapports" + (f" ({reason_fr(res['route_reason'])})" if res.get("route_reason") else ""))
    if tc := res.get("table_check"):
        value = fmt(tc["value_m"], 1, f" M$ {'US' if tc['currency'] == 'USD' else 'CA'}")
        if tc["match"]:
            badges.append(f"🟰 Concorde avec le tableau des indicateurs ({value})")
        else:
            badges.append(f"⚠️ Le tableau des indicateurs donne {value} (valeur {CONFIDENCE_FR.get(tc['confidence'], tc['confidence'])}, "
                          f"{tc.get('document')}" + (f" p. {tc['page']}" if tc.get("page") else "") + ") : l'un des deux chiffres est faux, "
                          "vérifiez dans le rapport.")
    if res.get("inherited"):
        badges.append("↪ Repris de la question précédente : " + ", ".join(INHERITED.get(x, x) for x in res["inherited"]))
    if res.get("standalone") and res.get("route") == "rag" and res["inherited"]:
        badges.append(f"💬 Question comprise : {md(res['standalone'])}")
    if v is not None:
        if v["ok"]:
            badges.append(f"✅ {v['n_figures']} chiffre(s), tous présents dans les sources")
        else:
            badges.append(f"⚠️ Chiffre(s) absent(s) des sources : {', '.join(v['unsupported'])}")
    companies = ", ".join(NAMES.get(c, c) for c in plan.get("companies") or []) or "toutes"
    years = plan.get("years") or ([plan["year"]] if plan.get("year") else [])
    badges.append(f"🔎 {companies} · {', '.join(map(str, years)) or 'toutes années'} · {res['seconds']:.1f} s")
    for b in badges:
        st.caption(b)
    if res["sources"]:
        with st.expander(f"Sources ({len(res['sources'])})"):
            for s in res["sources"]:
                st.markdown(f"**[{s['label']}]** {NAMES.get(s['company'], s['company'])} {s['year']} · "
                            f"`{s['document']}` p. {s['page']} · {s['kind']}")
                st.caption(s["excerpt"])


with tab_ask:
    st.session_state.setdefault("messages", [])
    if not st.session_state.messages:
        st.markdown("##### Exemples")
        st.caption("Puis posez une question de suivi : « Et en 2023 ? », « Et pour Desjardins ? », « Pourquoi ? »")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
                st.session_state.pending = ex
                st.rerun()
    history = st.container()          # the conversation above, the input box below it
    for m in st.session_state.messages:
        with history.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(md(m["content"]))
            elif "error" in m:
                st.error(m["error"])
            else:
                show_answer(m["content"])

    question = st.chat_input("Posez une question sur les rapports annuels…") or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with history.chat_message("user"):
            st.markdown(md(question))
        with history.chat_message("assistant"):
            try:
                with st.spinner("Recherche dans les rapports et rédaction de la réponse…"):
                    res = (api_ask(question, k, company, year, verify) if company or year
                           else api_chat(question, k, verify))
                show_answer(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"La question n'a pas abouti : {exc}")
                st.session_state.messages.append({"role": "assistant", "error": f"La question n'a pas abouti : {exc}"})


# ------------------------------------------------------------ indicators ---

with tab_kpis:
    c1, c2, c3 = st.columns([2, 3, 2])
    kpi = c1.selectbox("Indicateur", list(KPI_LABELS), format_func=KPI_LABELS.get)
    chosen = c2.multiselect("Entreprises", list(NAMES), default=list(NAMES), format_func=NAMES.get)
    min_conf = c3.selectbox("Confiance minimale", CONFIDENCE, index=2,
                            help="manual : vérifiée à la main · high : vérifiée et confirmée par le rapport suivant · "
                                 "medium : vérifiée · review : à revoir · low : non vérifiée")
    try:
        df = kpis(min_conf)
    except requests.RequestException as exc:
        st.error(f"Tableau des indicateurs indisponible : {exc}")
        df = pd.DataFrame()
    if not df.empty:
        df = df[(df["kpi"] == kpi) & (df["company"].isin(chosen))]
        if df.empty:
            st.info("Aucune valeur pour ce choix : élargissez la confiance minimale ou les entreprises.")
        else:
            df = df.assign(entreprise=df["company"].map(NAMES), year=df["year"].astype(int))
            wide = df.pivot(index="year", columns="entreprise", values="value_m").sort_index()
            wide.index = wide.index.astype(str)              # years as labels, not as 2,018
            unit = KPI_UNITS.get(kpi, "millions, devise de publication")
            st.markdown(f"**{KPI_LABELS[kpi]}** ({unit})")
            if kpi not in KPI_UNITS and {"couchetard", "shopify"} & set(chosen):
                st.caption("Couche-Tard et Shopify publient en dollars américains, les autres en dollars canadiens.")
            st.line_chart(wide, height=360)
            conf = df.pivot(index="year", columns="entreprise", values="confidence").sort_index()
            conf.index = conf.index.astype(str)
            table = wide.apply(lambda col: col.map(lambda v: fmt(v, 2 if kpi == "diluted_eps" else 1)))
            marks = conf.apply(lambda col: col.map(lambda c: {"manual": " ✋", "review": " ?"}.get(c, "")))
            st.dataframe((table + marks.reindex_like(table).fillna("")).T, width="stretch")
            st.caption("✋ vérifiée à la main · ? à revoir · — absente au niveau de confiance choisi")
            st.download_button("Télécharger (CSV)", df.drop(columns=["entreprise"]).to_csv(index=False, sep=";"),
                               f"finrag_{kpi}.csv", "text/csv")


# ----------------------------------------------------------------- covid ---

with tab_covid:
    try:
        data = covid()
    except requests.RequestException as exc:
        st.error(f"Étude indisponible : {exc}")
        data = None
    if data:
        st.markdown("Chaque entreprise est comparée sur **son année COVID** (l'exercice qui contient le plus de mois "
                    "de pandémie) à l'exercice 2019. Pour Couche-Tard, dont l'exercice 2020 se termine le 26 avril "
                    "2020, c'est 2021.")
        rows = []
        for r in data["rows"]:
            cy = r["covid_year"]
            rows.append({"Entreprise": f"{r['name']} ({r['currency']})", "Année COVID": cy,
                         "Revenus 2019": fmt(r["revenue_2019"], 0), f"Revenus année COVID": fmt(r[f"revenue_{cy}"], 0),
                         "Variation des revenus": fmt(r["revenue_change_covid_year_pct"], 1, " %", sign=True),
                         "Résultat net 2019": fmt(r["net_income_2019"], 0),
                         "Résultat net année COVID": fmt(r[f"net_income_{cy}"], 0),
                         "Marge 2019": fmt(r["net_margin_2019"], 1, " %"),
                         "Marge année COVID": fmt(r[f"net_margin_{cy}"], 1, " %"),
                         "Choc": r["shock"], "Revenus au niveau de 2019": str(r["revenue_back_to_2019"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        change = pd.DataFrame({"Variation des revenus (%)": {r["name"]: r["revenue_change_covid_year_pct"]
                                                             for r in data["rows"]}})
        c1.markdown("**Revenus, année COVID vs 2019 (%)**")
        c1.bar_chart(change, horizontal=True, height=300)
        margins = pd.DataFrame({str(y): {r["name"]: r[f"net_margin_{y}"] for r in data["rows"]} for y in (2019, 2020, 2021)})
        c2.markdown("**Marge nette (%)**")
        c2.bar_chart(margins, stack=False, height=300)

        st.markdown("#### Pourquoi : ce qu'en disent les rapports")
        if not data["explanations"]:
            st.info("Aucune explication : lancez `python -m finrag.covid --explain`.")
        titles = {"impact": "Ce que le rapport dit de la pandémie", "drivers": "Ce qui explique la variation du résultat net"}
        for r in data["rows"]:
            parts = [e for e in data["explanations"] if e["company"] == r["company"]]
            if not parts:
                continue
            with st.expander(f"{r['name']} — exercice {r['covid_year']}"):
                for e in parts:
                    st.markdown(f"**{titles.get(e.get('kind', 'impact'))}**")
                    st.caption(e["question"])
                    st.markdown(md(e["answer"]))
                    if e.get("unsupported"):
                        st.caption("⚠️ Chiffres absents des sources : " + ", ".join(e["unsupported"]))
                    st.caption("Sources : " + ", ".join(f"S{s['n']} {s['source']} p.{s['page']}" for s in e["sources"]))
