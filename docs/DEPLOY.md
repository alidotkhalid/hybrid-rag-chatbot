# Deploying to Hugging Face Spaces

Written for Windows, since that is where you are running. Everything below is
free — no card, no trial.

---

## 0. Before you start

You need Python 3.10+ and git on your machine:

```powershell
python --version
git --version
```

If Python is missing, install from [python.org/downloads](https://www.python.org/downloads/)
and **tick "Add python.exe to PATH"** on the first screen. If git is missing,
get it from [git-scm.com/download/win](https://git-scm.com/download/win).

---

## 1. Build the index locally

The repository ships without the corpus (the PDFs are the authors' work) and
without the index (it is generated). Both are one command.

```powershell
cd "C:\Users\user\Desktop\RAG chatbot"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> If PowerShell blocks the activate script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window
> and try again. It applies only to that window.

Then:

```powershell
python -m ragcore.ingest all
```

This downloads 24 papers from arXiv (~60MB, rate-limited to one per second so
it takes about a minute), then chunks, embeds and indexes them. The first run
also downloads the embedding model (~130MB). Expect 3–5 minutes total.

You should see something like:

```
Index written to ...\data\index
  chunks               1487
  documents            24
  vector_backend       faiss-IndexFlatIP
  vector_dimension     384
```

Sanity-check it:

```powershell
python -m ragcore.ingest inspect
```

---

## 2. Get a free LLM key

[console.groq.com/keys](https://console.groq.com/keys) → sign in with Google →
**Create API Key** → copy it.

Groq's free tier is generous and fast, and no card is required. Then:

```powershell
copy .env.example .env
notepad .env
```

Set `RAG_LLM_API_KEY=gsk_...` and save.

---

## 3. Run it locally

```powershell
uvicorn app.main:app --reload --port 7860
```

Open <http://localhost:7860> and ask something like *"What is the
compute-optimal ratio of tokens to parameters?"*

Tick **Show retrieval trace** — you will see which retriever found each
passage. That panel is worth understanding before an interview; it is the
clearest possible demonstration that you know what your own system is doing.

---

## 4. Choose a host

**On Hugging Face Spaces:** as of late 2026, Docker and Gradio Spaces on the
free CPU-basic tier require a **PRO subscription** ($9/month). Only Static
Spaces — pure client-side HTML with no server — remain free, and this app needs
a Python process. Hugging Face's pricing page still advertises CPU Basic as
free; the Space creation form is the accurate source. If you have PRO, the
Spaces route still works and the YAML header in `README.md` is already correct
for it.

These instructions use **Google Cloud Run**, whose free tier genuinely covers
demo traffic: 2 million requests, 180,000 vCPU-seconds and 360,000 GiB-seconds
per month, with scale-to-zero so an idle demo costs nothing.

**Be honest with yourself about the caveats:**

- Google requires a billing account (a card on file) before Cloud Run can be
  enabled, even though the free tier costs nothing. Set a budget alert.
- Container images live in Artifact Registry, which gives 0.5 GB free. This
  image is roughly 1 GB, so expect **a few cents a month** in storage unless
  you delete old revisions. It is not literally zero.
- Cold starts take 20–40 s while the models load. Scale-to-zero is what keeps
  it free; that latency is the price.

---

## 5. Install the Google Cloud CLI

Download from [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install),
run the installer, then **close and reopen PowerShell**.

```powershell
gcloud --version
gcloud auth login
```

## 6. Create a project and enable billing

```powershell
gcloud projects create rag-chatbot-demo --name="RAG Chatbot"
gcloud config set project rag-chatbot-demo
```

Then in the console, link a billing account to the project
([console.cloud.google.com/billing](https://console.cloud.google.com/billing))
and set a budget alert at $1 so any charge is immediately visible.

Enable the APIs the deploy needs:

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 7. Deploy

**On the region.** These commands use `us-central1` rather than a region closer
to you. Cloud Run's Always Free tier includes 1 GiB of outbound transfer per
month *from North America only*; egress from other regions is billed from the
first byte. The amounts here are trivial either way — an answer is a few KB —
but `us-central1` keeps the deployment cleanly inside the documented free tier.
The added latency is ~200 ms, which is noise next to the ~2.5 s the
cross-encoder already takes.

From the project root:

```powershell
gcloud run deploy hybrid-rag-papers `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --cpu-boost `
  --timeout 300 `
  --concurrency 8 `
  --min-instances 0 `
  --max-instances 2
```

What each flag is doing, since these are the ones that matter:

| flag | why |
|---|---|
| `--source .` | builds from the Dockerfile with Cloud Build; no local Docker needed |
| `--memory 2Gi` | torch plus both models sit around 1 GB resident; 512 Mi will OOM |
| `--cpu 2` | reranking is CPU-bound — 30 cross-encoder passes per query |
| `--cpu-boost` | extra CPU during startup, which cuts cold-start model loading |
| `--timeout 300` | answers stream over SSE; the default 60 s can cut long ones off |
| `--concurrency 8` | one process, one model copy — do not let it be swamped |
| `--min-instances 0` | scale to zero when idle. This is what keeps it free. |
| `--max-instances 2` | a hard ceiling, so a traffic spike cannot run up a bill |

The first build takes ~10 minutes. It prints a service URL when done.

## 8. Add your API key

Set it as an environment variable on the deployed service, so it is never in
your shell history or the image:

```powershell
gcloud run services update hybrid-rag-papers `
  --region us-central1 `
  --update-env-vars RAG_LLM_API_KEY=your_groq_key_here
```

Or through the console: **Cloud Run → hybrid-rag-papers → Edit & deploy new
revision → Variables & Secrets**.

For a longer-lived deployment, Secret Manager is the better home for this
(`--update-secrets`), and its free tier covers a single secret comfortably.

## 9. Verify

```powershell
gcloud run services describe hybrid-rag-papers --region us-central1 --format="value(status.url)"
```

Visit `<that URL>/api/health` and confirm `"index_ready": true`. The URL without
the path is your live demo.

## 10. Redeploying after a change

```powershell
git add -A ; git commit -m "..." ; git push origin main
gcloud run deploy hybrid-rag-papers --source . --region us-central1
```

The service keeps its URL and its environment variables across deploys.

## 11. Keeping the cost at zero

```powershell
# what has actually been billed
gcloud billing accounts list

# delete old container images (the one real cost)
gcloud artifacts docker images list us-central1-docker.pkg.dev/rag-chatbot-demo/cloud-run-source-deploy
```

Delete superseded images after a few redeploys and storage stays inside the
free allowance.

## Troubleshooting

**Build fails on `pip install torch`** — check the Dockerfile still has
`--extra-index-url https://download.pytorch.org/whl/cpu`. Without it pip pulls
~5GB of CUDA wheels and the build runs out of disk.

**Space builds but `index_ready` is false** — `data/index/` is missing from the
commit. `.gitignore` excludes `data/corpus/` but *not* `data/index/`; if you
changed that, change it back.

**Answers stream in one lump instead of token by token** — a proxy is buffering
the response. The service already sends `X-Accel-Buffering: no`; if you moved
it behind something else, that header needs to survive.

**`model_not_found` / 404 from the provider** — your API key cannot access the
configured model. Groq's public model table lists Enterprise-only models
alongside free ones, so the docs are not a reliable guide to what your key may
call. Ask the provider directly:

```powershell
python scripts/list_models.py
```

Then set `RAG_LLM_MODEL` in `.env` to something from that list. The symptom is
distinctive: retrieval works and sources render normally, but the answer area
shows "The language model could not be reached" — the failure is entirely in
the generation stage.

**"Could not reach the language model"** — the secret is missing or misnamed.
It must be exactly `RAG_LLM_API_KEY`. The service degrades to retrieval-only
rather than erroring, so the sources panel still working is the tell.

**Space sleeps after inactivity** — free Spaces do. First request after a sleep
takes ~30s to wake. Mention this next to the link so nobody thinks it is broken.
