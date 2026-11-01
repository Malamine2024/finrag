"""Multi-turn benchmark of the conversational agent: what each turn is understood to ask, without answering.

For every turn of every conversation: the companies, the fiscal years and the indicator after the memory
of the previous turns, and the route chosen (the indicator table or the RAG). The real query planner is
used (regex + LLM, about one second per turn); no search and no answer, so the whole run takes a minute.

    python -m finrag.bench_agent

Conversations come from eval/conversations.jsonl (written with the defaults below on the first run; add
your own). A turn's expected route is the route chosen from the question; at answer time, a value missing
from the table or not verified still sends a "table" question to the RAG.
"""
import json
import time

from .agent import Turn, choose_route, resolve
from .config import ROOT

CASES_FILE = ROOT / "eval" / "conversations.jsonl"


def t(q, companies, years, kpi, route):
    return {"q": q, "companies": companies, "years": years, "kpi": kpi, "route": route}


DEFAULT_CONVERSATIONS = [
    {"id": "c01", "turns": [
        t("Quel était le résultat net de CGI en 2024 ?", ["cgi"], [2024], "net_income", "table"),
        t("Et en 2023 ?", ["cgi"], [2023], "net_income", "table"),
        t("Et pour Desjardins ?", ["desjardins"], [2023], "net_income", "table"),
        t("Pourquoi a-t-il varié ?", ["desjardins"], [2023], "net_income", "rag")]},
    {"id": "c02", "turns": [
        t("What was Air Canada's revenue in 2019?", ["aircanada"], [2019], "revenue", "table"),
        t("And in 2020?", ["aircanada"], [2020], "revenue", "table"),
        t("Why did it fall so much?", ["aircanada"], [2020], "revenue", "rag"),
        t("What about Shopify?", ["shopify"], [2020], "revenue", "table")]},
    {"id": "c03", "turns": [
        t("Revenus de Shopify de 2019 à 2021", ["shopify"], [2019, 2020, 2021], "revenue", "table"),
        t("Et son résultat net sur la même période ?", ["shopify"], [2019, 2020, 2021], "net_income", "table"),
        t("Et le total de l'actif ?", ["shopify"], [2019, 2020, 2021], "total_assets", "table")]},
    {"id": "c04", "turns": [
        t("Ratio CET1 de la Banque Nationale en 2022", ["banquenationale"], [2022], "cet1_ratio", "table"),
        t("Et celui de Desjardins ?", ["desjardins"], [2022], "cet1_ratio", "table"),
        t("Quelles sont les exigences réglementaires minimales pour ce ratio ?", ["desjardins"], [2022], "cet1_ratio", "rag")]},
    {"id": "c05", "turns": [   # a new question that names a company starts a new topic
        t("Résultat net de Couche-Tard en 2021 ?", ["couchetard"], [2021], "net_income", "table"),
        t("Qui est le président et chef de la direction de CGI ?", ["cgi"], [], None, "rag"),
        t("Et ses revenus les plus récents ?", ["cgi"], [2025], "revenue", "table")]},
    {"id": "c06", "turns": [   # a company outside the corpus is not completed from memory
        t("Quel était le résultat net de la Banque Nationale en 2022 ?", ["banquenationale"], [2022], "net_income", "table"),
        t("Et celui de la Banque Royale du Canada ?", [], [], None, "rag")]},
    {"id": "c07", "turns": [   # measures the table does not hold
        t("Revenu net d'intérêts de la Banque Nationale en 2023", ["banquenationale"], [2023], None, "rag"),
        t("Et son résultat net ?", ["banquenationale"], [2023], "net_income", "table"),
        t("Et au quatrième trimestre ?", ["banquenationale"], [2023], None, "rag")]},
    {"id": "c08", "turns": [
        t("How much revenue did CGI make in fiscal 2022?", ["cgi"], [2022], "revenue", "table"),
        t("How much of it came from the United States?", ["cgi"], [2022], None, "rag"),
        t("And the diluted EPS?", ["cgi"], [2022], "diluted_eps", "table")]},
    {"id": "c09", "turns": [
        t("Comment Air Canada a-t-elle traversé la pandémie ?", ["aircanada"], [2020, 2021], None, "rag"),
        t("Et sa perte nette en 2021 ?", ["aircanada"], [2021], "net_income", "table")]},
    {"id": "c10", "turns": [
        t("Résultat net de toutes les entreprises en 2024", "all", [2024], "net_income", "table"),
        t("Laquelle a le plus élevé ?", "all", [2024], "net_income", "rag"),
        t("Et en 2023 ?", "all", [2023], "net_income", "table")]},
]


def load_conversations() -> list[dict]:
    if not CASES_FILE.exists():
        CASES_FILE.parent.mkdir(parents=True, exist_ok=True)
        CASES_FILE.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in DEFAULT_CONVERSATIONS),
                              encoding="utf-8")
        print(f"Conversations written to {CASES_FILE} (add your own).")
    return [json.loads(line) for line in CASES_FILE.open(encoding="utf-8") if line.strip()]


def main() -> None:
    from .query_plan import COMPANY_INFO
    conversations = load_conversations()
    checks = {"companies": 0, "years": 0, "kpi": 0, "route": 0}
    turns_ok = conv_ok = n_turns = 0
    errors = []
    start = time.perf_counter()
    for conv in conversations:
        previous, all_ok = None, True
        for i, turn in enumerate(conv["turns"], start=1):
            plan, kpi, inherited = resolve(turn["q"], previous)
            route, _ = choose_route(turn["q"], plan, kpi, inherited)
            got = {"companies": sorted(plan.get("companies") or []),
                   "years": sorted(plan.get("years") or ([plan["year"]] if plan.get("year") else [])),
                   "kpi": kpi, "route": route}
            expected = {"companies": sorted(COMPANY_INFO) if turn["companies"] == "all" else sorted(turn["companies"]),
                        "years": sorted(turn["years"]), "kpi": turn["kpi"], "route": turn["route"]}
            wrong = [f for f in checks if got[f] != expected[f]]
            for f in checks:
                checks[f] += f not in wrong
            n_turns += 1
            turns_ok += not wrong
            all_ok &= not wrong
            if wrong:
                errors.append(f"{conv['id']} turn {i} \"{turn['q']}\" (inherited: {', '.join(inherited) or '-'})\n"
                              + "\n".join(f"      {f}: got {got[f]}, expected {expected[f]}" for f in wrong))
            # The next turn sees what this one understood, right or wrong, as in a real conversation.
            previous = Turn(turn["q"], turn["q"], plan, kpi, route)
        conv_ok += all_ok
    seconds = time.perf_counter() - start
    print(f"{len(conversations)} conversations, {n_turns} turns, {seconds / n_turns:.2f} s per turn\n")
    print("| Check | Correct |\n|---|---|")
    for f, n in checks.items():
        print(f"| {f} | {n}/{n_turns} |")
    print(f"| **turn fully right** | **{turns_ok}/{n_turns}** |")
    print(f"| **conversation fully right** | **{conv_ok}/{len(conversations)}** |")
    if errors:
        print("\nErrors:")
        print("\n".join(f"  - {e}" for e in errors))


if __name__ == "__main__":
    main()
