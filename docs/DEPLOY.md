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

## 4. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) (sign up
   first if needed — free).
2. **Space name:** something like `hybrid-rag-papers`.
3. **License:** MIT.
4. **SDK:** **Docker** → **Blank**.
5. **Hardware:** CPU basic (free).
6. **Visibility:** Public — the point is that people can open it.
7. Create.

---

## 5. Add your API key as a secret

In the Space: **Settings → Variables and secrets → New secret**.

- Name: `RAG_LLM_API_KEY`
- Value: your Groq key

Use **secret**, not variable — variables are visible to anyone who can see the
Space.

---

## 6. The Space header — already done

Spaces needs a YAML block at the very top of `README.md` telling it which SDK
to use and which port to expose. **This is already in your `README.md`** — you
do not need to add it. For reference, it is:

```yaml
---
title: Hybrid RAG — Ask The Papers
emoji: 📚
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

Leave `sdk: docker` and `app_port: 7860` alone — they must match the Dockerfile.
The title and emoji are yours to change.

## 7. Push

```powershell
git init
git add -A
git commit -m "Hybrid RAG chatbot"

git remote add space https://huggingface.co/spaces/<your-username>/<your-space>
git push space main
```

When prompted for a password, use an **access token**, not your account
password: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
→ New token → type **Write** → copy → paste at the prompt.

> **On the index size.** The committed index is a few MB, which is fine for
> normal git. If you later index a much larger corpus and a push is rejected
> for file size, that is when to set up git-lfs — not before.

---

## 8. Watch the build

The Space shows a build log. First build takes ~10 minutes: it installs
CPU-only torch and bakes the two models into the image so cold starts do not
stall on a download.

When it finishes, check `https://<your-space>.hf.space/api/health`. You want:

```json
{"status": "ok", "index_ready": true, "stats": {"chunks": 1487, ...}}
```

If `status` is `degraded`, the message in `stats.error` says why. The usual
cause is that `data/index/` was not committed — check with
`git ls-files data/index`.

---

## 9. Put it on your CV

Once it is live:

- Add the Space URL to the top of `README.md` where the placeholder is.
- Run the evaluation and **fill in the ablation table** in `README.md` with
  your real numbers. An empty table is worse than no table; a filled one is the
  thing that makes the project credible.
- Push the repo to GitHub too — recruiters look at commit history and READMEs,
  and Spaces' git view is not where they will look.

---

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
