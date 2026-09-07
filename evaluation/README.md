# Evaluation

## Running it

```bash
# Retrieval ablation. No API key, no LLM calls.
python -m evaluation.run_eval

# Add answer-quality metrics (needs RAG_LLM_API_KEY).
python -m evaluation.run_eval --generate --out evaluation/results.json

# One configuration only.
python -m evaluation.run_eval --config hybrid_rerank
```

## The question set

`questions.jsonl` holds 40 hand-written questions over the preloaded corpus,
labelled with the document(s) that contain the answer. They are split
deliberately:

| type | n | what it isolates |
|---|---|---|
| `semantic` | 23 | Paraphrased questions sharing little vocabulary with the source. Dense retrieval should carry these; BM25 should struggle. |
| `lexical` | 10 | Rare literal tokens — `[CLS]`, `175 billion`, `10000`, `RoPE`. BM25 should carry these; dense retrieval should struggle. |
| `multi-source` | 2 | The answer needs two papers. Tests whether the per-document cap actually lets a second source through. |
| `unanswerable` | 5 | Out-of-corpus questions. The system must refuse. |

35 of the 40 are answerable, and the gold labels cover **all 24 papers** — no
document in the corpus is unreachable by the eval set, so a retrieval
regression cannot hide in an unmeasured corner. A test asserts that every gold
filename is a real corpus entry, which catches the commonest gold-set bug: a
typo'd filename that silently makes a question unscorable and drags every
metric down.

The split is the point. An aggregate hit rate hides the fact that a hybrid
system's whole justification is winning on *both* halves — so the report breaks
results down by type, and a configuration that gains 3 points overall while
losing 15 on `lexical` is visibly a bad trade.

## Metrics

**Retrieval** (objective, computed against the gold labels):

- `hit_rate` — any gold document anywhere in the top k. Predicts whether the
  generator *can* answer at all.
- `recall@k` — fraction of gold documents retrieved. Differs from hit rate only
  on multi-source questions.
- `MRR` — 1/rank of the first gold document. The metric that tracks answer
  quality most closely, because a generator's attention is front-loaded: rank 1
  is worth much more than rank 5.
- `nDCG@k` — rewards retrieving *several* gold documents, not just the easiest.
- `refusal_rate_unanswerable` — how often the groundedness gate fires on
  out-of-corpus questions. Should be high.
- `false_refusal_rate` — how often it fires on answerable ones. Should be near
  zero. These two move in opposite directions and together set
  `RAG_MIN_RERANK_SCORE`.

**Generation** (proxies — read the caveats):

- `citation_coverage` — fraction of substantive sentences carrying a citation.
- `keyword_recall` — fraction of expected terms present in the answer.
- `uncited_answer_rate` — answers with no citations at all. Should be zero.
- `correct_refusal_rate` / `false_refusal_rate` — end-to-end refusal behaviour.

## What these numbers do not measure

Worth stating plainly, because a portfolio project that overclaims its
evaluation is worse than one that evaluates less and says so:

- **Citation coverage detects missing evidence, not wrong evidence.** An answer
  that cites passage [3] for a claim passage [3] does not support scores 1.0.
  Catching that needs entailment checking or a judge model.
- **Keyword recall rewards saying the right words**, not being right. It is a
  smoke test for catastrophic failure, not a quality measure.
- **n = 35 answerable questions is small.** The harness prints a 95% confidence
  interval on hit rate for exactly this reason — differences smaller than that
  interval are noise, and tuning against them is over-fitting to the eval set.
- **The gold labels are document-level, not passage-level.** Retrieving the
  right paper's wrong section counts as a hit. Passage-level labels would be
  stricter but would need re-annotating on every chunking change.

The honest next step is an LLM-as-judge pass for faithfulness. It is not here
because a judge from the same model family as the generator is a biased judge,
and because a metric that cannot be reproduced without an API key is worse than
a transparent proxy.

## Reading the ablation

The table's purpose is to make each stage justify itself. What to look for:

- **`sparse_only` should beat `dense_only` on the `lexical` split** and lose
  badly on `semantic`. If it does not, the BM25 tokenizer is probably
  over-normalising and destroying the rare tokens it exists to preserve.
- **`hybrid` should beat both on aggregate hit rate** while sitting between
  them on each individual split. Fusion buys recall, not precision.
- **`hybrid_rerank` should barely move hit rate but should move MRR
  substantially.** That is the signature of a working reranker: it does not
  find new documents, it reorders the ones fusion already found. If reranking
  changes hit rate much, something upstream is wrong.
- **Latency should rise sharply at `hybrid_rerank`** — that is 30 cross-encoder
  forward passes, and it is the price of the MRR gain. If it is not visible in
  `latency_p50_ms`, the reranker silently failed to load and fell back to
  `IdentityReranker`.

Record the numbers you actually get in the main README rather than the ones you
hoped for. A result that contradicts the reasoning above is the interesting
one — it means an assumption in the pipeline is wrong, and finding out which is
worth more than a clean table.
