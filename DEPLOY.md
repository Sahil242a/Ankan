# Putting Ankan online

Ankan needs roughly 2 to 3 GB of memory once the handwriting model is loaded (the model alone
is 1.3 GB), so it does not fit on hosts that give 512 MB for free. Plan for 4 GB.

Three ways to make it reachable by others:

| | Cost | Setup | Link | Best for |
| --- | --- | --- | --- | --- |
| **1. Share from your computer** | Free | 5 minutes | Changes each time | Interviews and demos |
| **2. Google Cloud Run** | Free for light use (card needed) | 20 minutes | Permanent | A demo link on your resume |
| **3. Any Docker host** | Paid (about 4 GB of memory) | Depends on host | Permanent | Real use |

**Always set a password before sharing.** Without one, anyone with the link can see and
delete everything. Set `ANKAN_PASSWORD`; the browser then asks for it (any username works).
Use made-up answers for demos, never real students' work.

---

## 1. Share from your computer (Cloudflare Tunnel)

Your computer runs Ankan as usual, and Cloudflare gives it a temporary public `https://`
address. Nothing to sign up for. The link works only while your computer and both
terminals stay on.

1. Install cloudflared once (Windows):
   ```powershell
   winget install --id Cloudflare.cloudflared
   ```
   Close and reopen the terminal afterwards. On Mac: `brew install cloudflared`.

2. Terminal 1, in the `ankan` folder with the virtual environment active:
   ```powershell
   $env:ANKAN_PASSWORD="choose-a-password"
   ankan serve
   ```
   (Mac/Linux: `export ANKAN_PASSWORD="choose-a-password"` then `ankan serve`.)

3. Terminal 2:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```
   It prints a link like `https://random-words.trycloudflare.com`. Share that link and the
   password.

Stop with **Ctrl + C** in both terminals. Next time you get a new link.

---

## 2. Google Cloud Run (permanent link)

Cloud Run builds the app from the `Dockerfile`, starts it when someone visits, and stops it
when idle, so you pay only for time in use. The monthly free allowance (request-based
billing) is 180,000 vCPU-seconds and 360,000 GiB-seconds: with the settings below that is
roughly 25 hours of active use a month. A billing account with a card is required, and
storing the built image (a few GB) may cost a little; set a budget alert.

1. Create a Google Cloud project with billing turned on, and install the
   [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
2. In a terminal:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
3. From the `ankan` folder:
   ```bash
   gcloud run deploy ankan --source . --region us-central1 \
     --memory 4Gi --cpu 2 --timeout 300 --max-instances 1 \
     --allow-unauthenticated \
     --set-env-vars ANKAN_PASSWORD=choose-a-password,ANKAN_FAST_OCR=1
   ```
   On Windows PowerShell, put it on one line or end lines with a backtick (`` ` ``) instead
   of `\`. The first build takes 10 to 20 minutes because it bakes the models into the
   image. When it finishes it prints the app's link.

`ANKAN_FAST_OCR=1` reads handwriting about twice as fast on Cloud Run's small CPUs with very
close accuracy; remove it to use the full-precision model.
`--allow-unauthenticated` only lets the page load; Ankan's own password still protects it.
`--max-instances 1` keeps costs predictable and keeps all answers on one instance.

Things to know:
- **Data is temporary.** Questions and answers are lost when Cloud Run stops the app after
  a quiet spell. Export marks before you leave. Keeping data needs a database (a next step).
- **The first visit after a quiet spell is slow** (up to a minute) while the app starts and
  loads the handwriting model.
- To update after changing code, run the same `gcloud run deploy` command again.

---

## 3. Any Docker host

The `Dockerfile` works on any host that runs containers and gives it about 4 GB of memory,
such as a cloud VM or a container host's 4 GB plan. Plans with 2 GB or less are likely too small.

```bash
docker build -t ankan .
docker run -p 8080:8080 -e ANKAN_PASSWORD=choose-a-password -v ankan-data:/app/data ankan
```

Open `http://localhost:8080`. The `-v ankan-data:/app/data` part keeps questions and
answers when the container restarts. The host must pass the port in `$PORT` or use 8080.

---

## Settings for deployments

| Variable | Purpose |
| --- | --- |
| `ANKAN_PASSWORD` | Password the browser asks for. Strongly recommended for any shared link. |
| `ANKAN_GRADER` | `lexical` (default), `embedding`, `llm` or `hybrid` |
| `ANKAN_FAST_OCR` | `1` for about 2x faster handwriting reading on CPU |
| `ANTHROPIC_API_KEY` | Needed only for the `llm` and `hybrid` graders |
| `ANKAN_DATA_DIR` | Where questions, answers and page images are stored |

All settings are listed in the README's Configuration section.
