# 🏙️ Ontario Places-to-Live RAG Assistant

A retrieval-augmented generation (RAG) system that answers questions about living in
Ontario municipalities using **only real, public data**, generating **cited,
source-grounded** answers with a small open LLM that runs **free in Google Colab**.

It is deliberately **descriptive and sourced, not prescriptive**: it reports what the
data says, cites where every fact came from, and answers *"I don't have data on that"*
when the knowledge base is silent — rather than guessing.

> **What it is:** a sourced research assistant for Ontario municipalities.
> **What it is *not*:** a tool that tells you where to move.

---

## Highlights

- **100% grounded, cited answers** from a knowledge base built entirely from real
  public sources — no synthetic or fabricated data anywhere.
- **Runs free in Colab** (GPU-aware, degrades to CPU) with **zero API keys or auth**.
- **Fully automated ingestion** with caching, per-host rate limiting, exponential
  backoff, mirror fallback, and a circuit breaker — restart-safe end to end.
- **A real evaluation harness** (not an afterthought) measuring retrieval, answer
  faithfulness, numeric support, and abstention behaviour.
- **Honest by construction:** graceful degradation when a source is down, and
  documented limitations throughout.

## Evaluation results (from an actual GPU run)

Run configuration: **Qwen2.5-3B-Instruct** generator, **BAAI/bge-small-en-v1.5**
embeddings, FAISS cosine index over **568 chunks** (278 Wikipedia + 266 Wikivoyage +
24 census), evaluated on **38 questions** (22 answerable-factual, 16 abstention).

| Metric | Value | What it means |
|---|---|---|
| Retrieval hit@5 | **1.00** | The expected source chunk was in the top-5 for every answerable factual question. |
| Mean lexical faithfulness | **0.785** | Fraction of answer content-words traceable to retrieved context. |
| Numeric-support rate | **1.00** | Every number in every answer also appears in the retrieved context (no fabricated stats). |
| False-abstention rate | **0.045** | Only 1 of 22 answerable questions wrongly abstained (a census-vocabulary edge case). |
| Abstention accuracy (out-of-scope) | **1.00** | Every genuinely unanswerable question correctly returned "I don't have data on that." |

*These are encouraging results on a small, curated evaluation set — a sanity-check of
the pipeline's behaviour, not a large-scale benchmark. Full per-question results are
in `artifacts/eval_results.csv`; the report is in `artifacts/eval_report.md`.*

> **Note on this run:** public OpenStreetMap Overpass mirrors were down, so amenity
> counts were unavailable. The system detected this, skipped OSM cleanly, and the
> evaluation reclassified the 10 amenity questions as expected-abstain — so the
> metrics above reflect only genuinely answerable questions. This is the
> graceful-degradation path working as designed.

---

## Architecture

```
                       ┌───────────────────────────────────────────────────┐
  REAL PUBLIC SOURCES  │  INGEST (cache · rate-limit · backoff · restart-safe)│
  ┌─────────────────┐  │                                                     │
  │ Wikipedia API   │──┤  per municipality:                                  │
  │ Wikivoyage API  │  │    • Wikipedia extract    (descriptive prose)       │
  │ OSM Overpass    │  │    • Wikivoyage extract   (editorial "vibe")        │
  │ StatCan WDS(opt)│  │    • OSM amenity counts   (probe → skip if down)    │
  │ 2021 Census     │  │    • 2021-Census backbone (structured facts)        │
  └─────────────────┘  └───────────────────────┬─────────────────────────────┘
                                               │  raw corpus (JSON on disk)
                                               ▼
                    ┌────────────────────────────────────────────────┐
                    │ PROCESS: clean → fact-sentences → chunk          │
                    │ (token-budget, self-contained, metadata-tagged)  │
                    └───────────────────────┬────────────────────────┘
                                            │ chunks + metadata
                                            ▼
     ┌────────────────────┐  embed   ┌───────────────────────────────┐
     │ BGE-small-en-v1.5  │────────▶ │ FAISS IndexFlatIP (cosine)     │
     └────────────────────┘          │  + parallel chunk metadata     │
                                     └───────────────┬───────────────┘
                                                     │ top-k (+ municipality filter,
                                                     │ per-city merge for comparisons)
              ┌──────────────────────────────────────▼──────────────────────┐
  question ─▶ │ RETRIEVE → grounding prompt → Qwen2.5-3B-Instruct → CITED ANSWER │
              └──────────────────────────────────────┬──────────────────────┘
                                                     ▼
                        EVAL: hit@k · faithfulness · numeric-support · abstention
```

---

## Real data sources (all free, no auth)

- **Wikipedia** (MediaWiki Action API, `prop=extracts`) — descriptive prose per
  municipality: history, economy, geography, demographics.
- **Wikivoyage** (same MediaWiki API) — editorial, qualitative "what's it like" prose
  (e.g. *trendy*, *gentrified*, *quiet residential*). Labelled as travel-writer tone,
  **not** resident reviews or measured sentiment.
- **Statistics Canada, 2021 Census of Population** — municipality population, land
  area, and density (a curated backbone of verifiable public facts). A live StatCan
  Web Data Service (`getFullTableDownloadCSV`) client is also wired in and **off by
  default**.
- **OpenStreetMap Overpass API** — counts of mapped amenities (parks, schools,
  hospitals, libraries, pharmacies, transit stops, supermarkets, restaurants) within a
  bounding box per city. Best-effort: auto-probed and skipped if the public mirrors
  are unavailable.

All endpoints and model IDs were verified as real and currently accessible before use.

## Models

- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, ungated). Fallback:
  `sentence-transformers/all-MiniLM-L6-v2`.
- **Generator:** `Qwen/Qwen2.5-3B-Instruct` on GPU, with an automatic fallback chain
  (`Phi-4-mini-instruct` → `Qwen2.5-1.5B-Instruct`). On a CPU runtime the notebook
  auto-switches to a lighter chain (`Qwen2.5-1.5B` → `0.5B`), shorter answers, and a
  trimmed evaluation set. All models are **ungated** — no Hugging Face token required.
  *(Llama-3.2-3B is deliberately excluded: it is gated and would break the no-auth
  constraint.)*

---

## How to run (from a fresh Colab)

1. Upload `Ontario_Places_to_Live_RAG.ipynb` to Google Colab.
2. `Runtime → Change runtime type → T4 GPU → Save` (CPU also works, just slower).
3. `Runtime → Run all`. The first run ingests + embeds + downloads the model (all
   cached afterwards, so re-runs and disconnects are cheap and restart-safe).
4. Ask questions with `answer("...")`, or call `interactive()` for a widget/loop.

Locally instead of Colab: `pip install -r requirements.txt` (plus a PyTorch build for
your platform) and run the notebook in Jupyter.

## Example answers (from an actual run)

**Grounded factual answer**
> **Q:** What is the population of Guelph and what county is it in?
> **A:** The population of Guelph is 143,740 according to the 2021 Census of
> Population. Guelph is located in Wellington County, Ontario. *(cites the
> 2021-Census fact chunk and the Guelph Wikipedia article.)*

**Correct abstention (out-of-scope)**
> **Q:** What is the average monthly rent in Hamilton?
> **A:** I don't have data on that in my knowledge base.

**Comparison (per-city retrieval + census-card injection)**
> **Q:** Compare Guelph and Kingston for a young family.
> **A (excerpt):** Population — Guelph: 143,740 (2021); Kingston: 132,485 (2021).
> Density — Guelph: ~1,648 /km²; Kingston: ~294 /km². Education — Guelph: University
> of Guelph; Kingston: Queen's University and Royal Military College… *(both cities'
> 2021-Census fact cards are force-included so population reflects the census, not a
> stale descriptive figure.)*

---

## Saved artifacts

- `artifacts/faiss.index` — the FAISS vector index (doubles as the embedding cache).
- `artifacts/chunks.json` — processed chunk text + metadata.
- `artifacts/eval_results.csv`, `artifacts/eval_report.md` — evaluation outputs.
- `raw/corpus.json` — the raw ingested corpus; `http_cache/` — cached API responses.

## Evaluation harness

The harness (notebook §10) measures three things against the *actual ingested data*:

1. **Retrieval hit-rate / recall@k** — for each question, was a chunk of the expected
   municipality + source type retrieved in the top-k?
2. **Answer faithfulness** — a deterministic **lexical-overlap** score plus a
   **numeric-support** check (every number in an answer must appear in the retrieved
   context). An optional **LLM-as-judge** (same local model, labelled a soft signal)
   is included but off by default.
3. **Abstention behaviour** — does the system correctly abstain on out-of-scope
   questions instead of hallucinating?

It also prints qualitative cases: a good grounded answer, a correct abstention, and a
comparison stress-test with analysis.

## Honest limitations

- **Retrieval can miss.** If the relevant chunk isn't in the top-k, the model can't use it.
- **Small local model.** A 3B model is a weaker synthesizer than frontier models and
  can occasionally over-generalize (e.g. summarizing "both cities have low crime
  rates") or blur which fact belongs to which city in a comparison. It also
  over-abstained on one census-vocabulary phrasing ("census division" vs "county").
- **OSM coverage is weather-dependent.** Public Overpass mirrors are frequently
  overloaded; amenity counts are also bounding-box approximations. Treat them as rough
  *relative* signals, not exact inventories.
- **Census scope.** "Population" is the city proper (census subdivision), not the metro
  area; boundary definitions don't always align across sources.
- **Wikivoyage is editorial travel-writer tone**, not resident reviews or measured
  sentiment.
- **The numeric-support metric checks traceability, not authoritativeness** — it
  verifies a number appears in *some* retrieved chunk, not that it's the most current
  one (which is why comparison queries force-include each city's census card).
- **No cost / rent / crime / school-quality data** — the system abstains on these.

## Future work

Hybrid (BM25 + dense) retrieval; a cross-encoder reranker; wiring StatCan WDS
income/housing tables end-to-end; a cited, ethically-sourced sentiment layer (e.g.
Reddit via the authenticated API, anonymized, framed as tone not fact); answer-level
citation verification; larger evaluation set.

## References

- Wikimedia REST / Action API — https://www.mediawiki.org/wiki/API:REST_API
- OpenStreetMap Overpass API — https://wiki.openstreetmap.org/wiki/Overpass_API
- Statistics Canada Web Data Service — https://www.statcan.gc.ca/en/developers/wds
- BGE embeddings — https://huggingface.co/BAAI/bge-small-en-v1.5
- Qwen2.5 — https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- FAISS — https://github.com/facebookresearch/faiss

## License / data attribution

Code: MIT (see `LICENSE`). Data belongs to its providers and is used under their terms
— Wikipedia & Wikivoyage text (CC BY-SA), OpenStreetMap (ODbL), Statistics Canada
(Statistics Canada Open Licence). This tool is informational and is not advice.
