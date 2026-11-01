"""Retrieval with optional company/year filters: vector search, BM25 keyword search, or both (hybrid).

FINRAG_RETRIEVAL (or the `mode` argument):
  vector  embeddings only (bge-m3 in Chroma): finds passages that mean the same thing as the question.
  bm25    keywords only: finds exact terms the embeddings may miss ("bail-in", "NSFR", "BAII ajusté").
  hybrid  (default) both lists merged with Reciprocal Rank Fusion: a passage ranked high by either one,
          or fairly high by both, comes first. RRF uses ranks only, so the two scores need no calibration.

The BM25 index is built from data/processed/chunks.jsonl on first use (about a minute) and cached in
data/processed/bm25.pkl; it is rebuilt automatically when chunks.jsonl changes.

BM25 matches words, so it cannot match an English question with a French report (Couche-Tard,
Desjardins...). When the question and the filtered report are in different languages, the question is
translated for BM25 only (FINRAG_TRANSLATE=on, default; one short LLM call). The vector search keeps the
original question: bge-m3 is multilingual.

FINRAG_RERANK=on adds a cross-encoder (bge-reranker-v2-m3): it reads the question and each of the 30 best
candidates together and scores their relevance, then the k best are kept. More precise than the first
search, which compares separate vectors or word counts, but too slow to run on the whole corpus.
  pip install sentence-transformers      (the model, about 2 GB, is downloaded on first use)
"""
import math
import os
import pickle
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from .config import CHUNKS_FILE, PROCESSED_DIR, TOP_K
from .index import embed, get_collection

RETRIEVAL = os.getenv("FINRAG_RETRIEVAL", "hybrid").strip().lower()
CANDIDATES = 30        # passages taken from each list before fusion
RERANK = os.getenv("FINRAG_RERANK", "off").strip().lower() == "on"
RERANK_MODEL = os.getenv("FINRAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("FINRAG_RERANK_DEVICE", "").strip().lower()   # cuda / cpu; empty = cuda if available
RERANK_CANDIDATES = 30  # passages scored by the reranker
RRF_K = 60             # usual RRF constant: 1 / (60 + rank)
BM25_FILE = PROCESSED_DIR / "bm25.pkl"
BM25_K1, BM25_B = 1.5, 0.75
# A BM25 hit is kept only if it matches this share of the question's information (sum of the idf of its
# words). An English question on a French report matches almost nothing: its BM25 list would be noise,
# and RRF would rank that noise as high as the good vector results.
MIN_COVERAGE = 0.3
TRANSLATE = os.getenv("FINRAG_TRANSLATE", "on").strip().lower() != "off"
INDEX_VERSION = 2      # bump when the index structure changes: the cache is then rebuilt

STOPWORDS = set("""
a au aux avec ce ces dans de des du elle en et est il ils la le les leur leurs lors mais ne ni nos notre ou
par pas pour qu que quel quelle quels quelles qui sa se ses son sont sur ta te tes ton un une vos votre
été était étaient être avait ont fait selon comment combien
an and are as at be been by for from had has have how in is it its of on or that the their this to was were
what when which who why with did does do according during
""".split())
# Words of the context header repeated in every chunk: no information for BM25.
STOPWORDS |= {"text", "table", "company", "fiscal", "year", "page", "title", "section"}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


STOPWORDS = {strip_accents(w) for w in STOPWORDS}   # compared with accent-free tokens

# Company names: the filters already select the company, and these words appear in most of its passages.
# Left in the question, they would match every passage of the company and drown the useful words.
COMPANY_WORDS = {"air", "canada", "aeroplan", "banque", "nationale", "national", "bank", "bnc", "cgi",
                 "alimentation", "couche", "tard", "circle", "desjardins", "mouvement", "shopify", "inc"}


def tokenize(text: str) -> list[str]:
    """Lowercase words without accents (réassurance = reassurance), without stopwords and 1-letter bits."""
    words = re.findall(r"[a-z0-9]+", strip_accents(text.lower()))
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


FR_MARKERS = {"le", "la", "les", "des", "du", "de", "et", "est", "une", "pour", "dans", "aux", "sur", "par", "au",
              "quel", "quelle", "quels", "quelles", "combien", "selon", "comment"}
EN_MARKERS = {"the", "of", "and", "to", "is", "are", "was", "for", "with", "on", "by", "from", "as", "were",
              "what", "which", "how", "did", "does", "in", "its"}


def detect_lang(text: str) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    fr = sum(w in FR_MARKERS for w in words)
    en = sum(w in EN_MARKERS for w in words)
    if fr == en:
        return None
    return "fr" if fr > en else "en"


_TRANSLATIONS: dict[tuple[str, str], str] = {}


def translate(question: str, lang: str) -> str:
    """Translate a question for the keyword search; on any failure, keep the original."""
    key = (question, lang)
    if key not in _TRANSLATIONS:
        from .config import LLM_MODEL, NUM_CTX
        from .index import get_ollama
        target = "French" if lang == "fr" else "English"
        prompt = (f"Translate this question into {target}, as it would be worded in a {target} annual report. "
                  "Keep company names, acronyms, numbers and dates. Reply with the translation only.\n\n"
                  f"{question}")
        kwargs = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                  # Same num_ctx as the answer: with another value, Ollama reloads the model at every switch.
                  "options": {"temperature": 0, **({"num_ctx": NUM_CTX} if NUM_CTX else {})}}
        try:
            try:
                response = get_ollama().chat(think=False, **kwargs)
            except Exception:  # model without a thinking mode, or old ollama package
                response = get_ollama().chat(**kwargs)
            text = re.sub(r"<think>.*?</think>", "", response.message.content or "", flags=re.S).strip()
            _TRANSLATIONS[key] = text or question
        except Exception:
            _TRANSLATIONS[key] = question
    return _TRANSLATIONS[key]


def as_years(year) -> list[int]:
    """A year filter may be one year or several (the report of year Y and the next one)."""
    if not year:
        return []
    return [int(y) for y in year] if isinstance(year, (list, tuple, set)) else [int(year)]


def build_filter(company: str | None, year) -> dict | None:
    conditions = []
    if company:
        conditions.append({"company": company.lower()})
    years = as_years(year)
    if len(years) == 1:
        conditions.append({"year": years[0]})
    elif years:
        conditions.append({"year": {"$in": years}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


# ----------------------------------------------------------------- vector ---
def vector_search(question: str, n: int, company: str | None, year: int | None) -> list[dict]:
    result = get_collection().query(
        query_embeddings=embed([question]),
        n_results=n,
        where=build_filter(company, year),
    )
    return [
        {"id": id_, "text": doc, "metadata": meta, "distance": dist}
        for id_, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0],
            result["metadatas"][0], result["distances"][0],
        )
    ]


# ------------------------------------------------------------------- bm25 ---
class BM25Index:
    """Inverted index over chunks.jsonl, scored with Okapi BM25 (numpy, no extra dependency)."""

    def __init__(self, chunks: list[dict]):
        self.ids = [c["id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]
        self.metadata = [{k: c[k] for k in ("company", "year", "page", "kind", "source")} for c in chunks]
        self.companies = np.array([c["company"] for c in chunks])
        self.years = np.array([int(c["year"]) for c in chunks])
        self.langs = np.array([detect_lang(c["text"]) or "" for c in chunks])
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, text in enumerate(self.texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for word, tf in counts.items():
                postings[word].append((i, tf))
        self.lengths = np.array(lengths, dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if lengths else 0.0
        n = len(chunks)
        self.postings = {}
        for word, plist in postings.items():
            docs = np.array([d for d, _ in plist], dtype=np.int32)
            tfs = np.array([t for _, t in plist], dtype=np.float32)
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            self.postings[word] = (docs, tfs, idf)

    def search(self, question: str, n: int, company: str | None = None, year: int | None = None) -> list[dict]:
        scores = np.zeros(len(self.ids), dtype=np.float32)
        matched = np.zeros(len(self.ids), dtype=np.float32)   # idf of the question words each passage has
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths / self.avg_length)
        n_docs = len(self.ids)
        max_idf = math.log(1 + (n_docs + 0.5) / 0.5)          # a word absent from the corpus: maximal idf
        # The filters already select the company and the year: their words would match the whole report.
        words = {w for w in tokenize(question) if not re.fullmatch(r"20\d\d", w)} - COMPANY_WORDS
        total_idf = 0.0
        for word in words:
            if word in self.postings:
                docs, tfs, idf = self.postings[word]
                scores[docs] += idf * tfs * (BM25_K1 + 1) / (tfs + norm[docs])
                matched[docs] += idf
                total_idf += idf
            else:
                total_idf += max_idf
        mask = (scores > 0) & (matched >= MIN_COVERAGE * total_idf)
        if company:
            mask &= self.companies == company.lower()
        if year:
            mask &= np.isin(self.years, as_years(year))
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            return []
        top = candidates[np.argsort(-scores[candidates])[:n]]
        return [{"id": self.ids[i], "text": self.texts[i], "metadata": self.metadata[i],
                 "bm25": float(scores[i])} for i in top]


    def language(self, company: str | None, year: int | None) -> str | None:
        """Main language of the filtered passages (None without a company filter: mixed corpus)."""
        if not company:
            return None
        mask = self.companies == company.lower()
        if year:
            mask &= np.isin(self.years, as_years(year))
        langs = self.langs[mask]
        langs = langs[langs != ""]
        if not len(langs):
            return None
        values, counts = np.unique(langs, return_counts=True)
        return str(values[np.argmax(counts)])


_BM25: BM25Index | None = None


def get_bm25() -> BM25Index:
    """Load the BM25 index from its cache, or build it when chunks.jsonl is newer or different."""
    global _BM25
    if _BM25 is not None:
        return _BM25
    import json
    stat = CHUNKS_FILE.stat()
    key = (INDEX_VERSION, stat.st_size, int(stat.st_mtime))
    if BM25_FILE.exists():
        try:
            with BM25_FILE.open("rb") as f:
                cached_key, index = pickle.load(f)
            if cached_key == key:
                _BM25 = index
                return _BM25
        except Exception:  # unreadable or old cache: rebuild it
            pass
    print("Building the BM25 index from chunks.jsonl (once, about a minute)...")
    chunks = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8") if line.strip()]
    _BM25 = BM25Index(chunks)
    tmp = BM25_FILE.with_name(BM25_FILE.stem + ".tmp.pkl")
    with tmp.open("wb") as f:
        pickle.dump((key, _BM25), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(BM25_FILE)
    return _BM25


def bm25_search(question: str, n: int, company: str | None, year: int | None) -> list[dict]:
    index = get_bm25()
    if TRANSLATE:
        doc_lang = index.language(company, year)
        if doc_lang and detect_lang(question) not in (None, doc_lang):
            question = translate(question, doc_lang)
    return index.search(question, n, company, year)


# ----------------------------------------------------------------- fusion ---
def rrf(*ranked_lists: list[dict], k: int) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (RRF_K + rank)."""
    scores: dict[str, float] = defaultdict(float)
    hits: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit["id"]] += 1 / (RRF_K + rank)
            hits.setdefault(hit["id"], {}).update(hit)   # keeps the vector distance and the bm25 score
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**hits[i], "rrf": round(scores[i], 5)} for i in best]


# --------------------------------------------------------------- reranker ---
_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("FINRAG_RERANK=on needs: pip install sentence-transformers") from exc
        import torch
        device = RERANK_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"max_length": 512, "device": device}
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}   # half the memory, same ranking
        try:
            _RERANKER = CrossEncoder(RERANK_MODEL, **kwargs)
        except TypeError:  # older sentence-transformers without model_kwargs
            kwargs.pop("model_kwargs", None)
            _RERANKER = CrossEncoder(RERANK_MODEL, **kwargs)
        print(f"reranker: {RERANK_MODEL} on {device}")
    return _RERANKER


def rerank(question: str, hits: list[dict], k: int) -> list[dict]:
    if len(hits) <= 1:
        return hits[:k]
    scores = get_reranker().predict([(question, h["text"]) for h in hits], batch_size=16, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores, dtype=np.float32))[:k]
    return [{**hits[i], "rerank": round(float(scores[i]), 4)} for i in order]


def retrieve(question: str, k: int = TOP_K, company: str | None = None,
             year: int | None = None, mode: str | None = None, use_rerank: bool | None = None) -> list[dict]:
    mode = (mode or RETRIEVAL).lower()
    use_rerank = RERANK if use_rerank is None else use_rerank
    pool = max(RERANK_CANDIDATES, k) if use_rerank else k     # candidates to return before reranking
    if mode == "vector":
        hits = vector_search(question, pool, company, year)
    elif mode == "bm25":
        hits = bm25_search(question, pool, company, year)
    elif mode == "hybrid":
        n = max(CANDIDATES, pool)
        hits = rrf(vector_search(question, n, company, year), bm25_search(question, n, company, year), k=pool)
    else:
        raise ValueError(f"Unknown retrieval mode '{mode}': vector, bm25 or hybrid")
    return rerank(question, hits, k) if use_rerank else hits


# ----------------------------------------------------------------- period ---
PERIOD_MAX_PASSAGES = 15   # all years together: about 5 000 tokens of sources, inside num_ctx 8192
PERIOD_PER_YEAR = 3        # at most, per year asked


def retrieve_slots(question: str, slots: list[tuple[str | None, int | None]], window: int = 1,
                   max_passages: int = PERIOD_MAX_PASSAGES, k: int = TOP_K, **kwargs) -> list[dict]:
    """One search per (company, fiscal year) slot, so that every company and year asked is represented.

    The figures of year Y are in the report of year Y and again, as comparatives (sometimes restated), in
    the reports of the next `window` years: each year is searched in those reports. A passage found for two
    slots is kept once. Each hit records its slot ("for_company", "for_year"); results follow the order of
    the slots (company, then chronological), the best passages of each slot first.
    """
    # The k passages of a single question shared among the slots, at least PERIOD_PER_YEAR each, within
    # max_passages in all: 2 companies -> 5 each; 5 years -> 3 each; 9 years -> 1 each.
    per_slot = max(1, min(max(PERIOD_PER_YEAR, k // len(slots)), max_passages // len(slots)))
    seen, hits = set(), []
    for company, y in slots:
        reports = (list(range(y, y + window + 1)) if window else y) if y else None
        taken = 0
        for hit in retrieve(question, k=per_slot + 2, company=company, year=reports, **kwargs):
            if taken == per_slot:
                break
            if hit["id"] not in seen:
                seen.add(hit["id"])
                hits.append({**hit, "for_company": company, "for_year": y})
                taken += 1
    return hits


def retrieve_period(question: str, years: list[int], company: str | None, window: int = 1,
                    max_passages: int = PERIOD_MAX_PASSAGES, **kwargs) -> list[dict]:
    """One search per fiscal year of a period question (see retrieve_slots)."""
    return retrieve_slots(question, [(company, y) for y in years], window, max_passages, **kwargs)
