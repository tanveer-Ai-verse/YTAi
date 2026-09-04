# 🎬 ShortsCraft AI

> **Transform any YouTube video into viral 9:16 Shorts — powered by Groq AI (100% free models)**

ShortsCraft AI is a production-ready Streamlit application that:
1. Downloads a YouTube video via `yt-dlp`
2. Transcribes audio using **Groq Whisper** (`whisper-large-v3-turbo`)
3. Detects the top 10 best clip moments using **Groq Llama** (`llama-3.3-70b-versatile`)
4. Auto-crops each clip to 9:16 vertical format
5. Burns dynamic captions and optional sound effects onto each clip
6. Delivers production-ready MP4 files ready for YouTube Shorts, Reels, and TikTok

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎯 AI Scene Detection | Llama 3.3 70B analyzes full transcripts to find viral-worthy moments |
| 🎙️ Whisper Transcription | `whisper-large-v3-turbo` — fast, accurate, timestamp-exact |
| 📱 9:16 Vertical Output | Auto center-crop from any 16:9 source, 1080×1920 |
| 🎨 5 Caption Presets | Bold Yellow, Hormozi Style, Minimal White, Cyber Neon, Red Impact |
| 🔊 Sound Effects | Synthesized whoosh/impact SFX injected at clip transitions |
| 🗂️ 8 Scene Categories | Funny, Action, Drama, Educational, Fights, Inspiration, Twists, Peaks |
| ⚡ Free API Tier | Zero cost — all Groq models are on the free tier |
| 🔒 Secrets-Safe | API key stored in Streamlit Secrets — never hardcoded |

---

## 🚀 Quickstart — Local Development

### 1. Prerequisites

#### Python
Python **3.10 or higher** is required.

```bash
python --version   # should print 3.10.x, 3.11.x, or 3.12.x
```

#### FFmpeg (required — must be installed at system level)

**Ubuntu / Debian / WSL:**
```bash
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version   # confirm installation
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
ffmpeg -version
```

**Windows:**
1. Download the latest build from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system `PATH`
4. Open a new terminal and run `ffmpeg -version`

### 2. Clone the repository

```bash
git clone https://github.com/your-username/shortscraftai.git
cd shortscraftai
```

### 3. Create a virtual environment

```bash
python -m venv .venv

# Activate:
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell
```

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add your Groq API key (local secrets)

Create the directory and file:

```bash
mkdir -p .streamlit
```

Create `.streamlit/secrets.toml` with the following content:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
```

> **⚠️ Important:** Add `.streamlit/secrets.toml` to your `.gitignore` — never commit API keys.

```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

Get a **free** Groq API key at 👉 https://console.groq.com

### 6. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## ☁️ Deploy to Streamlit Cloud

### Step 1 — Push your code to GitHub

Make sure your repository does **not** contain `.streamlit/secrets.toml`.

```bash
git add app.py requirements.txt README.md .gitignore
git commit -m "feat: initial ShortsCraft AI"
git push origin main
```

### Step 2 — Create a new Streamlit Cloud app

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **"Deploy"**.

### Step 3 — Add GROQ_API_KEY to Streamlit Cloud Secrets

> This is the most important step — the app will not work without it.

1. In your Streamlit Cloud dashboard, find your deployed app.
2. Click the **"⋮" (three-dot menu)** → **"Settings"**.
3. Navigate to the **"Secrets"** tab.
4. Paste the following into the secrets text area:

```toml
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
```

5. Click **"Save"**.
6. Your app will automatically restart and pick up the new secret.

> Streamlit encrypts secrets at rest and injects them securely at runtime. They are never exposed in your code or logs.

---

## 🏗️ Project Structure

```
shortscraftai/
├── app.py                  # Main Streamlit application (all logic)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── secrets.toml        # Local secrets — DO NOT commit this file
```

The app uses `tempfile.gettempdir()` for intermediate files (raw clips, audio chunks, ASS subtitle files). All final Shorts are rendered in memory and served via Streamlit's download button — no permanent storage is required.

---

## 🧠 AI Models Used

| Task | Model | Notes |
|---|---|---|
| Audio transcription | `whisper-large-v3-turbo` | Chunked for long videos (5-min chunks) |
| Scene analysis | `llama-3.3-70b-versatile` | Primary model |
| Scene analysis fallback | `llama-3.1-8b-instant` | Automatic fallback on rate limit |

All models are on Groq's **free tier** — no billing required.

---

## 🎨 Caption Presets

| Preset | Font | Color | Style |
|---|---|---|---|
| Bold Yellow Highlight | Impact | Yellow on black | Classic meme-style |
| Hormozi Style | Arial Bold | White on black | Alex Hormozi signature thick text |
| Minimal White | Helvetica | White outline | Clean and professional |
| Cyber Neon | Courier | Cyan / Magenta | Futuristic vibe |
| Red Impact | Impact | Red on white | High-energy alert |

---

## 🔊 Sound Effects

Sound effects are **synthesized via FFmpeg** (no external audio files needed):

| Effect | Used At |
|---|---|
| Whoosh | Clip intro (0s) |
| Impact Boom | Clip outro (last 0.5s) |

---

## ⚠️ Troubleshooting

### "FFmpeg not found"
Install FFmpeg at the **system level** — `pip install ffmpeg` does not install the binary. See the [prerequisites](#prerequisites) section.

### "GROQ_API_KEY not found"
For local dev: ensure `.streamlit/secrets.toml` exists and contains your key.  
For Streamlit Cloud: add the key via **Settings → Secrets** (see [Step 3](#step-3--add-groq_api_key-to-streamlit-cloud-secrets)).

### "Video download failed"
- Verify the YouTube URL is public (not age-gated or private).
- Update yt-dlp: `pip install --upgrade yt-dlp`

### "Scene detection returned fallback clips"
- The Groq model returned invalid JSON or was rate-limited.
- The app automatically falls back to evenly-distributed 30s clips.
- Try again — Groq free tier has per-minute limits that reset quickly.

### Subtitle burn errors on Windows
FFmpeg ASS filter path handling on Windows can occasionally conflict with drive letters. The app gracefully falls back to the un-captioned cropped clip in this case. Running inside WSL avoids this entirely.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙌 Credits

Built with:
- [Streamlit](https://streamlit.io) — UI framework
- [Groq](https://groq.com) — LLM inference (Whisper + Llama)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloader
- [FFmpeg](https://ffmpeg.org) — Video processing engine
- [MoviePy](https://zulko.github.io/moviepy/) — Python video utilities
