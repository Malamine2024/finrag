"""Phase E: the HTTP API of FinRAG (FastAPI).

    uvicorn finrag.api:app --port 8000          # or: python -m finrag.api
    http://localhost:8000/docs                  # interactive documentation (Swagger)

Endpoints
    GET  /health      Ollama reachable, models present, number of indexed passages
    GET  /companies   the companies and fiscal years of the corpus
    POST /ask         a question -> cited answer, sources, search plan, figure check (RAG only, no memory)
    POST /chat        a turn of a conversation: router (indicator table or RAG) and memory of the session
    GET  /kpis        the indicator table of phase D2 (filters: company, kpi, year, confidence)
    GET  /covid       the COVID-19 study of phase D3 and its explanations

One GPU serves the answer model: questions are answered one at a time (a lock), the others wait. The
indicator and COVID endpoints only read files and answer at once, even while a question is running.
"""
import json
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import EMBED_MODEL, LLM_MODEL, OLLAMA_HOST, PROCESSED_DIR
from .agent import Conversation
from .agent import chat as agent_chat
from .generate import answer
from .query_filters import YEARS
from .query_plan import COMPANY_INFO
from .verify import verify

KPIS_FILE = PROCESSED_DIR / "kpis.jsonl"
CONFIDENCE_ORDER = ["manual", "high", "medium", "review", "low"]
KPI_NAMES = ["revenue", "net_income", "diluted_eps", "total_assets", "cet1_ratio"]
DEFAULT_K = 10

app = FastAPI(title="FinRAG", version="1.0",
              description="Questions over the annual reports of six Canadian companies (2017-2025), "
                          "answered by a local LLM with cited sources.")
_gpu = threading.Lock()


# ----------------------------------------------------------------- models ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["Quel était le résultat net de la Banque Nationale en 2024 ?"])
    k: int = Field(DEFAULT_K, ge=1, le=20, description="passages given to the model")
    company: str | None = Field(None, description="force a company (otherwise detected in the question)")
    year: int | None = Field(None, description="force a fiscal year (otherwise detected in the question)")
    verify: bool = Field(True, description="check that every figure of the answer is in the sources")


class Source(BaseModel):
    n: int
    label: str                    # how the answer cites it: "S2", or "cgi S2" for an answer by company
    company: str | None
    year: int | None
    page: int | None
    document: str | None
    kind: str | None              # "table" or "text"
    excerpt: str


class Verification(BaseModel):
    ok: bool
    n_figures: int
    unsupported: list[str]
    other_source: list[str]


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    plan: dict
    verification: Verification | None
    seconds: float


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, examples=["Résultat net de CGI en 2024 ?"])
    session_id: str | None = Field(None, description="returned by the first turn; omit it to start a conversation")
    k: int = Field(DEFAULT_K, ge=1, le=20)
    verify: bool = True


class ChatResponse(AskResponse):
    session_id: str
    route: str                    # "table" (indicator table) or "rag" (search in the reports)
    route_reason: str             # why the RAG, when the table was not used
    kpi: str | None
    inherited: list[str]          # what the turn took from the previous one: companies, years, kpi
    standalone: str               # the question as understood (rewritten for a follow-up sent to the RAG)
    table_check: dict | None = None   # an indicator answered by the RAG: the table's value, and whether it matches


# ---------------------------------------------------------------- helpers ---

def _excerpt(text: str, size: int = 400) -> str:
    """The passage without its context header ("[Table | Company: ... ]"), shortened."""
    body = text.split("]", 1)[1] if text.startswith("[") and "]" in text[:300] else text
    body = " ".join(body.split())
    return body[:size] + ("…" if len(body) > size else "")


def _sources(hits: list[dict]) -> list[Source]:
    out, counters = [], {}
    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        group = h.get("for_company")
        if group:                                  # answer by company: numbering restarts per company
            counters[group] = counters.get(group, 0) + 1
            label = f"{group} S{counters[group]}"
        else:
            label = f"S{i}"
        out.append(Source(n=i, label=label, company=meta.get("company"), year=meta.get("year"),
                          page=meta.get("page"), document=meta.get("source"), kind=meta.get("kind"),
                          excerpt=_excerpt(h.get("text", ""))))
    return out


def _is_connection_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return "connect" in name or "timeout" in name or "connection" in str(exc).lower()


MAX_SESSIONS = 200
_sessions: "OrderedDict[str, Conversation]" = OrderedDict()


def get_session(session_id: str | None) -> tuple[str, Conversation]:
    """The conversation of a session, or a new one; the oldest sessions are dropped beyond MAX_SESSIONS."""
    if session_id and session_id in _sessions:
        _sessions.move_to_end(session_id)
        return session_id, _sessions[session_id]
    session_id = session_id or uuid.uuid4().hex
    _sessions[session_id] = Conversation()
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)
    return session_id, _sessions[session_id]


_kpi_cache: dict = {"mtime": None, "rows": []}


def load_kpis() -> list[dict]:
    """kpis.jsonl, read again only when the file changes."""
    if not KPIS_FILE.exists():
        return []
    mtime = KPIS_FILE.stat().st_mtime
    if _kpi_cache["mtime"] != mtime:
        _kpi_cache["rows"] = [json.loads(line) for line in KPIS_FILE.open(encoding="utf-8") if line.strip()]
        _kpi_cache["mtime"] = mtime
    return _kpi_cache["rows"]


# -------------------------------------------------------------- endpoints ---

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    status = {"llm_model": LLM_MODEL, "embed_model": EMBED_MODEL, "ollama": OLLAMA_HOST}
    try:
        import ollama
        names = {m.model for m in ollama.Client(host=OLLAMA_HOST).list().models}
        status["ollama_ok"] = True
        status["models_missing"] = [m for m in (LLM_MODEL, EMBED_MODEL)
                                    if m not in names and f"{m}:latest" not in names]
    except Exception as exc:
        status.update(ollama_ok=False, ollama_error=str(exc)[:200], models_missing=[])
    try:
        from .index import get_collection
        status["passages"] = get_collection().count()
    except Exception as exc:
        status.update(passages=0, index_error=str(exc)[:200])
    status["kpis"] = len(load_kpis())
    status["status"] = ("ok" if status["ollama_ok"] and not status["models_missing"] and status["passages"]
                        else "degraded")
    return status


@app.get("/companies")
def companies() -> dict:
    return {"companies": [{"id": c, "description": d} for c, d in COMPANY_INFO.items()],
            "fiscal_years": [YEARS[0], YEARS[-1]]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.company and req.company not in COMPANY_INFO:
        raise HTTPException(422, f"unknown company '{req.company}'; one of {sorted(COMPANY_INFO)}")
    if req.year is not None and req.year not in YEARS:
        raise HTTPException(422, f"fiscal year {req.year} is not in the corpus ({YEARS[0]}-{YEARS[-1]})")
    start = time.perf_counter()
    try:
        with _gpu:                                 # one question at a time on the GPU
            res = answer(req.question, k=req.k, company=req.company, year=req.year)
    except Exception as exc:
        if _is_connection_error(exc):
            raise HTTPException(503, f"the language model is not reachable at {OLLAMA_HOST}: {exc}")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    hits = res.get("sources") or []
    plan = res.get("filters") or {}
    verification = None
    # An answer by company cites "[cgi S2]": its numbers do not point into the combined list of sources.
    if req.verify and hits and plan.get("mode") != "decompose":
        report = verify(res["answer"], hits, req.question)
        verification = Verification(ok=report["ok"], n_figures=report["n_figures"],
                                    unsupported=report["unsupported"], other_source=report["other_source"])
    return AskResponse(answer=res["answer"], sources=_sources(hits), plan=plan, verification=verification,
                       seconds=round(time.perf_counter() - start, 2))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id, conversation = get_session(req.session_id)
    start = time.perf_counter()
    try:
        with _gpu:
            res = agent_chat(conversation, req.question, k=req.k)
    except Exception as exc:
        if _is_connection_error(exc):
            raise HTTPException(503, f"the language model is not reachable at {OLLAMA_HOST}: {exc}")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    plan = res.get("filters") or {}
    if res["route"] == "table":
        sources = [Source(**s) for s in res["sources"]]
        verification = None                   # values of the table: already checked and cross-checked
    else:
        hits = res.get("sources") or []
        sources = _sources(hits)
        verification = None
        if req.verify and hits and plan.get("mode") != "decompose":
            report = verify(res["answer"], hits, res["standalone"])
            verification = Verification(ok=report["ok"], n_figures=report["n_figures"],
                                        unsupported=report["unsupported"], other_source=report["other_source"])
    return ChatResponse(answer=res["answer"], sources=sources, plan=plan, verification=verification,
                        seconds=round(time.perf_counter() - start, 2), session_id=session_id, route=res["route"],
                        route_reason=res.get("route_reason") or "", kpi=res.get("kpi"),
                        inherited=res.get("inherited") or [], standalone=res.get("standalone") or req.question,
                        table_check=res.get("table_check"))


@app.delete("/chat/{session_id}")
def end_chat(session_id: str) -> dict:
    return {"deleted": _sessions.pop(session_id, None) is not None}


@app.get("/kpis")
def kpis(company: str | None = None, kpi: str | None = None, year: int | None = None,
         min_confidence: str = Query("medium", description="lowest level kept: " + ", ".join(CONFIDENCE_ORDER))) -> dict:
    if min_confidence not in CONFIDENCE_ORDER:
        raise HTTPException(422, f"min_confidence must be one of {CONFIDENCE_ORDER}")
    if kpi and kpi not in KPI_NAMES:
        raise HTTPException(422, f"kpi must be one of {KPI_NAMES}")
    rows = load_kpis()
    if not rows:
        raise HTTPException(404, "no indicator table: run python -m finrag.kpis --crosscheck")
    keep = set(CONFIDENCE_ORDER[:CONFIDENCE_ORDER.index(min_confidence) + 1])
    fields = ("company", "year", "kpi", "value_m", "currency", "confidence", "source", "page", "line")
    out = [{f: r.get(f) for f in fields} for r in rows
           if r.get("confidence") in keep and r.get("value_m") is not None
           and (company is None or r["company"] == company) and (kpi is None or r["kpi"] == kpi)
           and (year is None or int(r["year"]) == year)]
    out.sort(key=lambda r: (r["company"], r["kpi"], int(r["year"])))
    return {"count": len(out), "unit": "millions of the reporting currency (per share for diluted_eps, % for cet1_ratio)",
            "rows": out}


@app.get("/covid")
def covid() -> dict:
    from . import covid as study
    if not KPIS_FILE.exists():
        raise HTTPException(404, "no indicator table: run python -m finrag.kpis --crosscheck")
    rows = study.study(study.load_values())
    return {"rows": rows, "explanations": study.load_explanations()}


def main() -> None:
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="Run the FinRAG API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
