# Hybrid RAG: ask the papers

A retrieval console over 24 foundational machine-learning papers, with an
upload path for your own documents. Hybrid BM25 + dense retrieval fused with
Reciprocal Rank Fusion, cross-encoder reranking, a groundedness gate that
declines rather than guesses, and citations validated server-side against what
was actually retrieved.



https://github.com/user-attachments/assets/b9313ac3-9b63-4e1e-a28d-a23c6ebca56d



**Run it yourself in three commands, about five minutes:**

```bash
pip install -r requirements.txt
python -m ragcore.ingest all      # fetches 24 papers, chunks, embeds, indexes
uvicorn app.main:app --port 7860
```

Then open <http://localhost:7860>. Without an API key it runs retrieval-only:
search, ranking, sources and the trace panel all work, and only answer
synthesis is stubbed. A free Groq key enables generation; see
[`.env.example`](.env.example).

```
┌─────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────┐
│ question│──▶│  condense    │──▶│  BM25  ×  │──▶│     RRF      │──▶│  cross-  │
│         │   │ (follow-ups) │   │  dense    │   │  fusion k=60 │   │ encoder  │
└─────────┘   └──────────────┘   │  30 each  │   └──────────────┘   │ rerank→6 │
                                 └───────────┘                      └────┬─────┘
                                                                         │
                          ┌──────────────────┐   ┌─────────────────┐     ▼
                          │ validated        │◀──│  stream answer  │◀── grounded?
                          │ citations [1][2] │   │  (SSE)          │    else refuse
                          └──────────────────┘   └─────────────────┘
```

---

## Why it is built this way

Most of the interesting decisions in a RAG system are about what to do when the
obvious approach fails. Each of these is a failure I designed around rather
than a feature I added.

### Hybrid retrieval, not pure vector search

A bi-encoder maps text into a semantic space where **exact tokens are not
preserved**. Ask it about `[CLS]`, `175 billion`, `RoPE` or `k=60` and it will
happily return passages that are *about the right topic* while missing the one
that contains the literal answer. BM25 nails those and fails at paraphrase.
Ask "why replace recurrence with attention?" and lexical search finds nothing,
because the source paper never uses the word "replace".

The two failure modes are close to independent, which is precisely the
condition under which fusion helps. `evaluation/` measures this: the `lexical`
and `semantic` question splits exist to show each retriever winning where the
other loses.

### RRF, not weighted score blending

Dense scores are cosine similarities in a narrow, corpus-dependent band. BM25
scores are unbounded sums of IDF weights whose scale depends on query length
and term rarity. They are not comparable, and normalising them (min-max,
z-score) makes the blend depend on the score distribution of whichever
candidates happened to come back, and that changes per query, so a weight tuned
on one query set silently mis-generalises.

RRF sidesteps calibration entirely by throwing scores away and using ranks:

```
RRF(d) = Σ_retrievers  1 / (k + rank_r(d))
```

A closed-form consequence, [pinned by a test](tests/test_fusion.py): one first
place scores `1/(k+1)`, two second places score `2/(k+2)`, and the latter is
larger for **every** k > 0. So agreement between retrievers always beats a lone
confident hit, regardless of tuning. What `k` actually controls is how *deep*
that agreement may be: at k=60, two rank-5 hits beat one rank-1 hit; at k=1
they do not.

### A cross-encoder on top of fusion

A bi-encoder compresses a passage into a vector **before it has seen the
query**, a hard information bottleneck, since one vector must serve every
possible question. A cross-encoder reads query and passage together and can
attend to the exact terms asked about, so it resolves negation and
near-miss-but-factually-wrong passages that fusion ranks confidently.

The cost is that it cannot be precomputed: it is one forward pass per
candidate. Hence the standard two-stage shape: cheap recall-oriented retrieval
to 30, expensive precision-oriented reranking to 6.

### A groundedness gate

Reranking will happily rank the least-bad of thirty irrelevant passages first.
That is exactly how a RAG system ends up answering a question its corpus cannot
support, fluently and with citations. So if nothing clears
`RAG_MIN_RERANK_SCORE`, the pipeline **refuses before calling the model**,
which also means a refusal costs no tokens. The `unanswerable` split in the
eval set exists to measure this, and it is the metric I would look at first.

### Citations validated server-side

A model asked to cite will sometimes cite passage `[7]` when six were
retrieved. The answer still reads perfectly. So every marker is checked against
the retrieved set, invalid ones are stripped, and survivors are renumbered
densely so the reader never sees a citation list starting at `[2]`. The
streamed text is replaced by the validated version when the stream closes.

### Structure-aware chunking

Fixed-width chunking slices mid-sentence and severs claims from the heading
that gives them meaning: "we use a batch size of 512" is useless without
knowing which experiment it belongs to. Here, sections are detected first and
never crossed, whole paragraphs are packed into a token budget, and **every
chunk is prefixed with its document title and section**, so the embedded text
is self-describing. Runt fragments are merged rather than indexed, because a
40-token fragment matches everything weakly and pollutes BM25.

### Exact search, not an ANN index

`IndexFlatIP`, brute force. At ~1,500 chunks an exact scan is well under a
millisecond, while HNSW or IVF would add a build step, tuning parameters and a
recall ceiling below 100% in exchange for nothing. Approximate search starts
earning its complexity somewhere north of ~10⁶ vectors. `VectorIndex` is the
seam where you would swap it.

Choosing the simple thing and being able to say exactly when it stops working
is the point.

---

## Running it

```bash
git clone <your-repo-url> && cd rag-chatbot
python -m venv .venv && .venv/Scripts/activate     # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt

cp .env.example .env          # then add a free Groq key (optional)

python -m ragcore.ingest all  # download 24 papers, chunk, embed, index (~3-5 min)
uvicorn app.main:app --reload --port 7860
```

Open <http://localhost:7860>.

Without an API key the service runs in retrieval-only mode: retrieval, ranking,
sources and the trace panel all work, and only synthesis is stubbed. Get a free
key at [console.groq.com/keys](https://console.groq.com/keys). No card required.

**Tests:**

```bash
pip install -r requirements-dev.txt
pytest                # 205 tests, ~2s, no network and no model downloads
ruff check .
```

The suite runs entirely on stubs: a hashed bag-of-words embedder, an identity
reranker, a scripted LLM. A test suite that needs 500MB of weights and an API
key is a test suite nobody runs.

**Evaluation:**

```bash
python -m evaluation.run_eval               # retrieval ablation, no API key
python -m evaluation.run_eval --generate    # + answer metrics
```

See [`evaluation/README.md`](evaluation/README.md) for what the numbers mean
and, more importantly, what they do not.

### Measured results

24 papers, 2,665 chunks, 40 gold questions (35 answerable, 5 out-of-corpus).
`bge-small-en-v1.5` + `ms-marco-MiniLM-L-6-v2`, k=6, on a 4-core laptop CPU.

| config | hit rate | recall@6 | MRR | nDCG@6 | refusal | p50 latency |
|---|---|---|---|---|---|---|
| dense only | 0.971 | 0.957 | 0.914 | 0.916 | n/a | 17 ms |
| sparse only (BM25) | 0.971 | 0.943 | 0.895 | 0.898 | n/a | **0.3 ms** |
| hybrid (RRF) | **1.000** | 0.971 | 0.919 | 0.918 | n/a | 18 ms |
| hybrid + rerank | **1.000** | **0.986** | **0.938** | **0.941** | **1.000** | 2,498 ms |

Two results behave exactly as the design predicts:

- **Fusion buys recall.** Either retriever alone misses one question in 35;
  fused, neither does. The two failure modes are genuinely different.
- **Reranking reorders rather than retrieves.** Hit rate does not move
  (1.000 → 1.000) while MRR climbs 0.919 → 0.938 and recall@6 0.971 → 0.986.
  That is the signature of a working cross-encoder: it is not finding new
  documents, it is promoting the right passage within the ones fusion already
  found. The cost is a 140× latency increase, which is the honest price of
  30 cross-encoder forward passes on CPU.

**The refusal column only means something in the last row.** The groundedness
gate keys off cross-encoder scores, so it cannot fire at all when reranking is
disabled, so the `n/a` entries are structural, not a comparison. Where the gate
does run it refused all 5 out-of-corpus questions with **zero false refusals**
on the 35 answerable ones.

### What this evaluation does *not* show

Reported deliberately, because the table above looks better than the system has
earned:

**The question set is near its ceiling and is not discriminative.** Every
configuration scores ≥ 0.94 on hit rate. The `lexical` split, written
specifically to make dense retrieval fail on rare literal tokens like `[CLS]`
and `175 billion`, is answered perfectly by *dense retrieval alone*. The
prediction that motivated the split was wrong.

The reason is that **gold labels are document-level**. Answering "what does the
`[CLS]` token do in BERT?" only requires identifying the BERT paper among 24,
and the word "BERT" makes that trivial for any retriever. The eval never tests
the thing that actually distinguishes these methods: finding the right
*passage* inside the right paper.

So the case for hybrid retrieval in this README rests on the mechanism and on
the published literature, not on this table, which is too easy to separate the
approaches. Making it discriminative means passage-level labels and questions
whose answers do not name their source document. That is the first thing I
would fix, and it is more valuable than any further tuning.

**Latency is a real constraint, not a footnote.** 2.5 s of retrieval before
generation even begins is slow for a chat interface, and the free-tier host has
fewer cores than the machine these numbers came from. The obvious lever is
`RAG_RERANK_CANDIDATES` (30 today); the harness can measure what quality costs
what milliseconds, and that experiment has not been run yet.

---|---|---|---|---|---|---|
> | dense_only | | | | | | |
> | sparse_only | | | | | | |
> | hybrid | | | | | | |
> | hybrid_rerank | | | | | | |

---

## Deploying

Not currently hosted, and that is a deliberate call rather than an omission.

The service needs roughly **1 GB resident**: PyTorch is ~250 MB before a model
loads, and the bi-encoder and cross-encoder add ~220 MB more. That does not fit
the free tiers that remain genuinely free:

| host | why not |
|---|---|
| Hugging Face Spaces | Docker/Gradio Spaces now require PRO ($9/mo); only Static Spaces are free |
| Render free tier | 512 MB; PyTorch alone does not fit |
| Google Cloud Run | free tier covers demo traffic comfortably, but requires a card on file |
| Vercel / Netlify | serverless bundle limits rule out PyTorch |

The interesting question is what you would change to fit. **Exporting both
models to ONNX Runtime** would cut memory to roughly 300-400 MB, enough for a
512 MB host, at the cost of a numerics re-validation against the eval harness
in `evaluation/`. That is the first thing I would do given a reason to host it,
and the `Embedder` and `Reranker` protocols in `ragcore/` are the seams where
that swap belongs: neither the retriever nor the pipeline would change.

`docs/DEPLOY.md` has working step-by-step instructions for **Google Cloud Run**
(the Dockerfile already honours `$PORT`) if you want to stand it up yourself.

## Layout

```
ragcore/            retrieval and generation, no web framework
  chunking.py         structure-aware chunking
  sparse.py           BM25, implemented directly (see the module docstring for why)
  fusion.py           Reciprocal Rank Fusion
  embedder.py         bi-encoder + a dependency-free stub
  reranker.py         cross-encoder + a no-op for ablations
  store.py            vectors + BM25 + metadata, persisted as one unit
  retriever.py        the hybrid pipeline
  llm.py              provider-agnostic streaming (Groq / Gemini / OpenAI-compatible)
  citations.py        marker validation and renumbering
  pipeline.py         question in, streamed grounded answer out
  ingest.py           the ingestion CLI

app/                FastAPI service, SSE transport, per-visitor upload sessions
  static/             the UI: one HTML file, one CSS file, one JS file, no build step

evaluation/         gold questions, metrics, ablation runner
tests/              205 tests, all offline
```

## API

| endpoint | what it does |
|---|---|
| `POST /api/chat` | SSE stream: `query` → `sources` → `token`… → `done` |
| `POST /api/upload` | index a document into the caller's session |
| `GET /api/documents` | what is in the preloaded corpus |
| `GET /api/health` | readiness and index statistics |
| `GET /api/debug/retrieve` | retrieval only, with per-stage ablation flags |

`/api/debug/retrieve?q=...&dense=false&rerank=false` is how the system was
tuned: it makes the effect of each stage directly observable, and it produced
the ablation numbers.

## Notes on the demo

Uploaded documents are chunked and indexed **in memory**, scoped to an opaque
session id, never written to disk, and evicted after an hour. They are searched
by the same code path as the preloaded corpus and compete with it on equal
terms rather than being appended as a second-class source. [A test asserts that
one visitor's upload cannot appear in another's
retrieval.](tests/test_api.py)

## Known limitations

- **Citation coverage cannot detect a *wrong* citation**, only a missing one.
  Faithfulness checking needs entailment or a judge model.
- **No incremental indexing.** A full rebuild takes under a minute at this
  scale, and an incremental path that can leave the vector and BM25 indexes
  disagreeing about document identity is a bug factory. `ChunkStore` is where
  it would go once rebuild time actually hurts.
- **Single worker.** Each worker loads its own ~600MB of model weights;
  concurrency comes from the in-process thread pool. Scaling out means moving
  the models behind their own service.
- **The condense step costs an extra LLM round-trip** on follow-ups that
  actually need it. It is skipped by a cheap heuristic when the question is
  already self-contained.
- **PDF extraction is imperfect.** Two-column layouts, tables and equations are
  handled with targeted cleanup, not a layout model. Tables in particular
  extract as line noise.

## Licence

MIT. The papers in the corpus belong to their authors and are downloaded from
arXiv at ingest time rather than redistributed here.
