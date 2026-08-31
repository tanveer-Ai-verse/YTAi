# 🎬 YTAi — Script-to-YouTube Video in One Click

> Your own Fliki.ai — no watermark, no subscription, no limits.

YTAi takes a plain narration script and produces a ready-to-upload YouTube video:
it fetches **real, royalty-free stock clips** from Pexels that match each sentence,
generates a **premium AI voiceover** via Groq PlayAI TTS, burns **animated word-highlight
captions** (exactly like the Fliki.ai reference style), auto-ducks background music during
speech, and exports a clean MP4 + SRT subtitle file.

---

## What it produces

| Feature | Detail |
|---|---|
| Stock footage | Real human footage from Pexels — HD landscape clips, no AI-generated imagery |
| Voiceover | Groq PlayAI TTS — 18 English voices, natural prosody |
| Captions | Word-highlight style (active word highlighted + underlined, others white on dark pill) |
| Output | 1280×720 MP4 (YouTube-ready) + `.srt` subtitle file |
| Music | Optional background track with auto-ducking (drops under speech, rises in pauses) |

---

## APIs required (both free)

| API | What it does | Get it |
|---|---|---|
| **Groq API** | TTS voiceover (PlayAI) + LLM keyword extraction + optional cloud Whisper | [console.groq.com](https://console.groq.com) — free tier |
| **Pexels API** | Real stock video search & download | [pexels.com/api](https://www.pexels.com/api/) — free, 200 req/hr |

---

## System prerequisites

### 1 — FFmpeg (required)

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**Windows** — download a static build from [ffmpeg.org](https://ffmpeg.org/download.html)
and add the `bin/` folder to your `PATH`.

Verify:
```bash
ffmpeg -version && ffprobe -version
```

### 2 — Python 3.10 – 3.12

Python 3.13+ is not yet supported by PyTorch (required by local Whisper fallback).

---

## Local installation

```bash
# 1. Clone
git clone https://github.com/your-username/ytai.git
cd ytai

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Secrets
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
GROQ_API_KEY   = "gsk_your_groq_key_here"
PEXELS_API_KEY = "your_pexels_key_here"
EOF

# 5. Run
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## Streamlit Community Cloud deployment

1. Push the repo (with `app.py` and `requirements.txt` at root) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select repo, branch, `app.py` as entry point.
4. Open **Settings → Secrets** and add:
   ```toml
   GROQ_API_KEY   = "gsk_your_groq_key_here"
   PEXELS_API_KEY = "your_pexels_key_here"
   ```
5. Deploy.

> **Note on size:** `openai-whisper` + `torch` ≈ 800 MB. Streamlit Community Cloud
> has a ~1 GB limit. To stay under it, remove `openai-whisper` and `torch` from
> `requirements.txt` — YTAi will use Groq Whisper (cloud) for transcription instead,
> which is faster anyway.

---

## How it works (the pipeline)

```
Script text
    │
    ▼
① Sentence splitting  ──────────────────────────────────── re / punctuation
    │
    ▼
② AI keyword extraction  ───────────────────────────────── Groq LLM
    │  "What if the Earth stopped spinning?" → "earth from space"
    │
    ▼
③ Pexels stock clip search + download  ─────────────────── Pexels API (HD, landscape)
    │
    ▼
④ Groq PlayAI TTS voiceover (per sentence)  ────────────── groq.audio.speech.create
    │
    ▼
⑤ Word-level timestamps  ───────────────────────────────── Groq Whisper (or local / fallback)
    │
    ▼
⑥ Per-frame render  ────────────────────────────────────── OpenCV filter + PIL caption burn
    │  • Video colour filter (Cinematic Dark, Color Boost, …)
    │  • Word-highlight captions (active word highlighted + underlined)
    │
    ▼
⑦ Scene clip assembly  (FFmpeg mux: processed video + TTS audio)
    │
    ▼
⑧ Crossfade concatenation  ─────────────────────────────── FFmpeg xfade filter
    │
    ▼
⑨ Background music mix + auto-duck  ────────────────────── pydub / FFmpeg amix
    │
    ▼
⑩ Final MP4 (1280×720, H.264/AAC) + SRT subtitles
```

---

## Project structure

```
ytai/
├── app.py            # Full application — all logic + UI in one file
├── requirements.txt  # Production Python packages
└── README.md         # This file
```

`app.py` is split into clearly labelled modules:

| Module | Responsibility |
|---|---|
| A — Script → Scenes | Sentence splitter + Groq LLM keyword extractor |
| B — Pexels | HD stock clip search + stream download |
| C — Groq TTS | PlayAI voiceover generation |
| D — Captions | Word timestamps (Groq Whisper → local Whisper → fallback) + SRT |
| E — Frame rendering | PIL word-highlight caption burn (Fliki-style) |
| F — Video filters | OpenCV per-frame colour grades |
| G — Clip assembly | Per-scene: trim clip → burn captions → mux TTS |
| H — Final assembly | FFmpeg xfade concat + BGM auto-duck |
| UI | Streamlit 3-tab interface: Create / Settings / Output |

---

## Caption styles

| Style | Active word | Background |
|---|---|---|
| **Fliki Classic** | Yellow, underlined | Black pill, 82% opacity |
| Neon Pop | Bright green, underlined | Black pill, 76% opacity |
| Cinematic White | Amber, no underline | Black pill, lower-third |
| Bold Impact | Red, underlined | Near-opaque black pill |

---

## TTS voices (Groq PlayAI)

Recommended for documentary / YouTube narration:

- **Chip** — clear, authoritative, neutral American
- **Thunder** — deep, dramatic (great for "What If" style videos)
- **Atlas** — calm and measured
- **Briggs** — warm, storytelling tone
- **Ethan** — upbeat, conversational

All 18 available voices are selectable in the Settings tab.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not found` | Add to `.streamlit/secrets.toml` or Streamlit Cloud Secrets |
| `PEXELS_API_KEY not found` | Same — or enter in the Create tab key fields |
| Pexels returns no results | Try a broader keyword; Pexels has 1M+ videos |
| TTS returns silence | Check your Groq key has TTS access (PlayAI model) |
| Clip assembly is slow | Normal — each scene runs FFmpeg trim + frame pipeline; ~8–20s per scene |
| `ffmpeg not found` | Install FFmpeg and confirm it's on your `PATH` |
| Video is black frames only | Stock clip download failed; check network and Pexels quota |
| Local Whisper very slow | Switch to Groq Whisper (cloud) in Settings, or use `tiny` model |

---

## Groq model note

YTAi uses:
- **`openai/gpt-oss-120b`** for LLM keyword extraction (Groq's current recommended model — replaced the deprecated `llama-3.3-70b-versatile` on August 16, 2026)
- **`playai-tts`** for TTS voiceover
- **`whisper-large-v3`** for cloud transcription

Both model names are single constants at the top of `app.py` and trivial to swap.

---

## License

Free for personal and educational use. Pexels videos are royalty-free for commercial use
(see [pexels.com/license](https://www.pexels.com/license/)). Always attribute photographers
when required.
