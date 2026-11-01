"""Tests of the conversational agent: indicator detection, router, memory, answers from the table.
No Ollama, no index: the planner is the regex one, the RAG and the rewriting are fakes.

    pytest tests/test_agent.py -q
"""
import json

import pytest

from finrag import agent
from finrag.query_plan import regex_plan

ROWS = [
    {"company": "cgi", "year": 2020, "kpi": "net_income", "value_m": 1117.862, "confidence": "manual",
     "currency": "CAD", "source": "cgi_2020.pdf", "page": 6, "line": "Bénéfice net"},
    {"company": "cgi", "year": 2021, "kpi": "net_income", "value_m": 1369.072, "confidence": "high",
     "currency": "CAD", "source": "cgi_2021.pdf", "page": 6, "line": "Bénéfice net"},
    {"company": "couchetard", "year": 2021, "kpi": "net_income", "value_m": 2705.5, "confidence": "manual",
     "currency": "USD", "source": "couchetard_2021.pdf", "page": 81, "line": "Bénéfice net"},
    {"company": "shopify", "year": 2019, "kpi": "revenue", "value_m": 1578.173, "confidence": "manual", "currency": "USD"},
    {"company": "shopify", "year": 2020, "kpi": "revenue", "value_m": 2929.491, "confidence": "manual", "currency": "USD"},
    {"company": "shopify", "year": 2021, "kpi": "revenue", "value_m": 4611.856, "confidence": "manual", "currency": "USD"},
    {"company": "desjardins", "year": 2023, "kpi": "net_income", "value_m": 22.0, "confidence": "review", "currency": "CAD"},
    {"company": "aircanada", "year": 2020, "kpi": "net_income", "value_m": -4647, "confidence": "manual", "currency": "CAD"},
]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    path = tmp_path / "kpis.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    monkeypatch.setattr(agent, "KPIS_FILE", path)
    monkeypatch.setattr(agent, "_table", {"mtime": None, "values": {}})
    monkeypatch.setattr(agent, "plan_question", lambda q, parser=None: regex_plan(q))
    monkeypatch.setattr(agent, "rewrite", lambda q, conv, plan, kpi: f"REWRITTEN {q}")
    calls = []
    import finrag.generate as generate
    monkeypatch.setattr(generate, "answer", lambda q, k=10, plan=None, **_: calls.append((q, plan)) or
                        {"answer": "RAG", "sources": [], "filters": plan or {}})
    return calls


@pytest.mark.parametrize("question, kpi", [
    ("Quel était le résultat net de CGI en 2021 ?", "net_income"),
    ("What was Shopify's net loss in 2019?", "net_income"),
    ("revenus de CGI les plus récents", "revenue"),
    ("Chiffre d'affaires de Couche-Tard en 2022", "revenue"),
    ("Résultat net par action dilué de la Banque Nationale", "diluted_eps"),
    ("Total de l'actif d'Air Canada en 2021", "total_assets"),
    ("Ratio CET1 de Desjardins en 2023", "cet1_ratio"),
    ("Excédents de Desjardins en 2020", "net_income"),
    ("Qui est le PDG de CGI ?", None),
    ("Revenus et résultat net de CGI en 2021", None),      # two indicators: the RAG answers
])
def test_detect_kpi(question, kpi):
    assert agent.detect_kpi(question) == kpi


@pytest.mark.parametrize("question, route", [
    ("Résultat net de CGI en 2021 ?", "table"),
    ("Pourquoi le résultat net de CGI a-t-il baissé en 2020 ?", "rag"),
    ("Revenu net d'intérêts de la Banque Nationale en 2021", "rag"),
    ("Revenus de CGI au quatrième trimestre 2021", "rag"),
    ("Revenus de placement de Desjardins en 2020", "rag"),
    ("Revenus de CGI aux États-Unis en 2021", "rag"),
    ("Résultat net ajusté de la Banque Nationale en 2021", "rag"),
    ("Quelle entreprise a le résultat net le plus élevé en 2021 ?", "rag"),
    ("Résultat net de Royal Bank of Canada en 2021", "rag"),
])
def test_choose_route(question, route):
    plan = regex_plan(question)
    assert agent.choose_route(question, plan, agent.detect_kpi(question))[0] == route


def test_conversation_inherits_and_routes(offline):
    conv = agent.Conversation()
    r = agent.chat(conv, "Résultat net de CGI en 2020 ?")
    assert (r["route"], r["inherited"]) == ("table", [])
    assert "1 117,9 M$ CA" in r["answer"] and "[T1]" in r["answer"]
    assert r["sources"][0]["page"] == 6

    r = agent.chat(conv, "Et en 2021 ?")
    assert r["route"] == "table" and set(r["inherited"]) == {"companies", "kpi"}
    assert r["filters"]["companies"] == ["cgi"] and r["filters"]["year"] == 2021
    assert "1 369,1 M$ CA" in r["answer"]

    r = agent.chat(conv, "Et pour Couche-Tard ?")
    assert r["filters"]["companies"] == ["couchetard"] and r["filters"]["year"] == 2021
    assert "2 705,5 M$ US" in r["answer"] and "fin avril" in r["answer"]

    r = agent.chat(conv, "Pourquoi a-t-il augmenté ?")
    assert r["route"] == "rag" and r["standalone"] == "REWRITTEN Pourquoi a-t-il augmenté ?"
    question, plan = offline[-1]
    assert plan["companies"] == ["couchetard"] and plan["year"] == 2021     # the search keeps the context


def test_new_topic_does_not_inherit():
    conv = agent.Conversation()
    agent.chat(conv, "Résultat net de CGI en 2020 ?")
    r = agent.chat(conv, "Qui est le PDG de Shopify ?")
    assert r["inherited"] == [] and r["filters"]["companies"] == ["shopify"] and r["filters"].get("year") is None


def test_outside_company_is_not_completed_from_memory(monkeypatch):
    monkeypatch.setattr(agent, "plan_question", lambda q, parser=None:
                        {**regex_plan(q), "outside": "royal" in q.lower()})
    conv = agent.Conversation()
    agent.chat(conv, "Résultat net de CGI en 2020 ?")
    r = agent.chat(conv, "Et celui de Royal Bank of Canada ?")
    assert r["route"] == "rag" and r["inherited"] == []


def test_period_with_change_and_currency():
    r = agent.chat(agent.Conversation(), "Revenus de Shopify de 2019 à 2021")
    assert r["route"] == "table"
    assert "1 578,2 M$ US" in r["answer"] and "4 611,9 M$ US" in r["answer"]
    assert "Variation 2019 → 2021 (Shopify) : +192,2 %" in r["answer"]


def test_loss_and_english():
    r = agent.chat(agent.Conversation(), "What was Air Canada's net income in 2020?")
    assert r["route"] == "table" and "-C$4,647 million" in r["answer"] and "fiscal 2020" in r["answer"]


def test_untrusted_or_missing_value_goes_to_rag():
    r = agent.chat(agent.Conversation(), "Résultat net de Desjardins en 2023 ?")      # confidence "review"
    assert r["route"] == "rag" and "desjardins 2023" in r["route_reason"]
    r = agent.chat(agent.Conversation(), "Résultat net de CGI en 2017 ?")             # not in the table
    assert r["route"] == "rag"


def test_memory_is_bounded():
    conv = agent.Conversation(max_turns=3)
    for y in (2020, 2021, 2020, 2021):
        agent.chat(conv, f"Résultat net de CGI en {y} ?")
    assert len(conv.turns) == 3
