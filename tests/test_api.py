"""Tests of the HTTP API, without Ollama or the index: the answer function is replaced by a fake.

    pytest tests/test_api.py -q
"""
import json

import pytest
from fastapi.testclient import TestClient

from finrag import api
from finrag import covid as study

PASSAGE = ("[Table | Company: banquenationale | Fiscal year: 2024 | Page: 120 | Title: États consolidés des résultats]\n"
           "Résultat net 3 816 3 335\nRevenu total 11 400 10 170")


def hit(company="banquenationale", year=2024, page=120, text=PASSAGE):
    return {"id": f"{company}-{page}", "text": text, "distance": 0.1,
            "metadata": {"company": company, "year": year, "page": page, "kind": "table",
                         "source": f"{company}_{year}.pdf"}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    rows = [
        {"company": "banquenationale", "year": 2024, "kpi": "net_income", "value_m": 3816, "confidence": "high"},
        {"company": "banquenationale", "year": 2023, "kpi": "net_income", "value_m": 3335, "confidence": "review"},
        {"company": "cgi", "year": 2020, "kpi": "revenue", "value_m": 12164.115, "confidence": "manual"},
        {"company": "cgi", "year": 2019, "kpi": "revenue", "value_m": None, "confidence": "low"},
    ]
    kpis_file = tmp_path / "kpis.jsonl"
    kpis_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(api, "KPIS_FILE", kpis_file)
    monkeypatch.setattr(api, "_kpi_cache", {"mtime": None, "rows": []})
    monkeypatch.setattr(study, "KPIS_FILE", kpis_file)
    monkeypatch.setattr(study, "EXPLANATIONS", tmp_path / "covid_explanations.jsonl")
    return TestClient(api.app)


def fake_answer(text, sources=None, mode="single"):
    def _answer(question, k=10, company=None, year=None, **_):
        return {"answer": text, "sources": [hit()] if sources is None else sources,
                "filters": {"companies": ["banquenationale"], "years": [2024], "mode": mode}}
    return _answer


def test_companies(client):
    data = client.get("/companies").json()
    assert len(data["companies"]) == 6
    assert data["fiscal_years"] == [2017, 2025]


def test_ask_supported_answer(client, monkeypatch):
    monkeypatch.setattr(api, "answer", fake_answer("Le résultat net de 2024 est de 3 816 M$ [S1]."))
    r = client.post("/ask", json={"question": "Résultat net de la Banque Nationale en 2024 ?"})
    assert r.status_code == 200
    data = r.json()
    assert data["verification"] == {"ok": True, "n_figures": 1, "unsupported": [], "other_source": []}
    src = data["sources"][0]
    assert (src["label"], src["company"], src["page"], src["document"]) == ("S1", "banquenationale", 120,
                                                                           "banquenationale_2024.pdf")
    assert not src["excerpt"].startswith("[Table"), "the context header is removed from the excerpt"


def test_ask_flags_figure_absent_from_sources(client, monkeypatch):
    monkeypatch.setattr(api, "answer", fake_answer("Le résultat net de 2024 est de 4 125 M$ [S1]."))
    data = client.post("/ask", json={"question": "Résultat net de la Banque Nationale en 2024 ?"}).json()
    assert data["verification"]["ok"] is False
    assert data["verification"]["unsupported"] == ["4 125"]


def test_ask_without_verification(client, monkeypatch):
    monkeypatch.setattr(api, "answer", fake_answer("3 816 M$ [S1]."))
    data = client.post("/ask", json={"question": "Résultat net 2024 ?", "verify": False}).json()
    assert data["verification"] is None


def test_ask_by_company_labels_and_no_verification(client, monkeypatch):
    sources = [{**hit("cgi", 2020), "for_company": "cgi"}, {**hit("cgi", 2020, 121), "for_company": "cgi"},
               {**hit("shopify", 2020), "for_company": "shopify"}]
    monkeypatch.setattr(api, "answer", fake_answer("CGI [cgi S2], Shopify [shopify S1].", sources, mode="decompose"))
    data = client.post("/ask", json={"question": "Revenus de toutes les entreprises de 2017 à 2025 ?"}).json()
    assert [s["label"] for s in data["sources"]] == ["cgi S1", "cgi S2", "shopify S1"]
    assert data["verification"] is None


@pytest.mark.parametrize("payload", [
    {"question": "Résultat net ?", "company": "rbc"},
    {"question": "Résultat net ?", "year": 2012},
    {"question": "?"},
    {"question": "Résultat net ?", "k": 50},
])
def test_ask_rejects_bad_requests(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_llm_unreachable(client, monkeypatch):
    def down(*_, **__):
        raise ConnectionError("Failed to connect to Ollama")
    monkeypatch.setattr(api, "answer", down)
    r = client.post("/ask", json={"question": "Résultat net de CGI en 2020 ?"})
    assert r.status_code == 503


def test_kpis_filters_and_confidence(client):
    data = client.get("/kpis", params={"company": "banquenationale"}).json()
    assert [r["year"] for r in data["rows"]] == [2024]             # 2023 is "review": below "medium"
    data = client.get("/kpis", params={"company": "banquenationale", "min_confidence": "review"}).json()
    assert [r["year"] for r in data["rows"]] == [2023, 2024]
    data = client.get("/kpis", params={"min_confidence": "low"}).json()
    assert all(r["value_m"] is not None for r in data["rows"])      # values not found are left out


def test_kpis_bad_parameters(client):
    assert client.get("/kpis", params={"kpi": "ebitda"}).status_code == 422
    assert client.get("/kpis", params={"min_confidence": "sure"}).status_code == 422


def test_kpis_missing_table(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "KPIS_FILE", tmp_path / "absent.jsonl")
    assert client.get("/kpis").status_code == 404


def test_covid(client):
    data = client.get("/covid").json()
    by = {r["company"]: r for r in data["rows"]}
    assert set(by) == set(study.COMPANIES)
    assert by["couchetard"]["covid_year"] == 2021
    assert data["explanations"] == []


def test_chat_keeps_the_session(client, monkeypatch):
    seen = []

    def fake_chat(conversation, question, k=10):
        seen.append(len(conversation.turns))
        conversation.turns.append(question)
        return {"answer": "Résultat net — CGI : 1 369,1 M$ CA [T1]", "route": "table", "route_reason": "",
                "kpi": "net_income", "inherited": ["companies"] if seen[-1] else [], "standalone": question,
                "filters": {"companies": ["cgi"], "year": 2021},
                "sources": [{"n": 1, "label": "T1", "company": "cgi", "year": 2021, "page": 6,
                             "document": "cgi_2021.pdf", "kind": "kpi table", "excerpt": "Bénéfice net : 1369.072"}]}
    monkeypatch.setattr(api, "agent_chat", fake_chat)
    first = client.post("/chat", json={"question": "Résultat net de CGI en 2021 ?"}).json()
    second = client.post("/chat", json={"question": "Et en 2020 ?", "session_id": first["session_id"]}).json()
    other = client.post("/chat", json={"question": "Et en 2020 ?"}).json()
    assert seen == [0, 1, 0]                                   # same session: the second turn sees the first
    assert second["session_id"] == first["session_id"] != other["session_id"]
    assert first["route"] == "table" and first["verification"] is None and first["sources"][0]["label"] == "T1"
    assert client.delete(f"/chat/{first['session_id']}").json() == {"deleted": True}
