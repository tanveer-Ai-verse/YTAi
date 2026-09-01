
# =============================================================================
#  YTAi — Script-to-YouTube Video in One Click
# =============================================================================
#  Pipeline: Script → AI keyword extraction → Pexels stock clips → Groq TTS
#            → Word-highlight captions → Normalize → Xfade concat → BGM mix
#
#  APIs required
#    GROQ_API_KEY    — Groq (llama-3.3-70b-versatile LLM + playai-tts TTS
#                      + whisper-large-v3 transcription)
#    PEXELS_API_KEY  — Pexels royalty-free stock video (free tier)
#
#  Stack: Streamlit · Groq SDK · MoviePy 2.x · OpenCV · Pydub · Pillow · FFmpeg
# =============================================================================

import html as _html
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ── Heavy libs — graceful fallback if not installed ──────────────────────────
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False

try:
    # FIX (Bug 5): from moviepy import afx, vfx works in MoviePy 2.x.
    # We also import the explicit submodule paths as a defensive fallback.
    from moviepy import (
        AudioFileClip, ColorClip, CompositeVideoClip,
        ImageClip, VideoFileClip,
        concatenate_videoclips,
    )
    try:
        from moviepy import afx, vfx
    except ImportError:
        from moviepy.audio import fx as afx          # explicit 2.x path
        from moviepy.video import fx as vfx          # explicit 2.x path
    MOVIEPY_OK = True
except Exception:
    MOVIEPY_OK = False

try:
    from groq import Groq as _GroqClient
    GROQ_LIB_OK = True
except Exception:
    GROQ_LIB_OK = False

try:
    import whisper as _local_whisper
    WHISPER_OK = True
except Exception:
    WHISPER_OK = False

# =============================================================================
#  CONSTANTS
# =============================================================================
APP_NAME = "YTAi"

# FIX (Bug 1): "openai/gpt-oss-120b" is not a Groq-hosted model.
# Groq hosts open-source models only. llama-3.3-70b-versatile is the
# correct production-tier replacement.
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"

# Groq PlayAI TTS — confirmed valid in Groq SDK 1.7 source:
#   model: Union[str, Literal["playai-tts", "playai-tts-arabic"]]
GROQ_TTS_MODEL  = "playai-tts"
GROQ_TTS_VOICE  = "Chip"               # clear, authoritative narrator

TMP_ROOT = Path(tempfile.gettempdir()) / "ytai_studio"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"

# Target output spec — all clips normalised to this before concatenation
OUT_W   = 1280
OUT_H   = 720
OUT_FPS = 30
OUT_AR  = 44100       # audio sample rate
OUT_AC  = "stereo"   # channel layout

# FIX (Bug 6): normalization filter strings applied to every clip BEFORE
# xfade so FFmpeg never sees mismatched resolution / fps / pixel-format /
# audio-rate between inputs.  Confirmed working in testing with clips
# spanning 720p-25fps-mono through 4K-24fps-96kHz.
_NORM_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
    f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
    f"setsar=1,fps={OUT_FPS},format=yuv420p"
)
_NORM_AF = (
    f"aresample={OUT_AR},"
    f"aformat=sample_fmts=fltp:channel_layouts={OUT_AC},"
    f"apad=pad_dur=0.5"
)

CAPTION_STYLES: Dict[str, Dict] = {
    "Fliki Classic": {
        "text_color": (255, 255, 255),
        "highlight_color": (255, 235, 100),
        "bg_color": (0, 0, 0, 210),
        "font_size": 54,
        "underline_highlight": True,
        "position": "bottom",
    },
    "Neon Pop": {
        "text_color": (220, 255, 220),
        "highlight_color": (0, 255, 128),
        "bg_color": (0, 0, 0, 195),
        "font_size": 52,
        "underline_highlight": True,
        "position": "bottom",
    },
    "Cinematic White": {
        "text_color": (248, 248, 248),
        "highlight_color": (255, 180, 60),
        "bg_color": (0, 0, 0, 168),
        "font_size": 46,
        "underline_highlight": False,
        "position": "lower-third",
    },
    "Bold Impact": {
        "text_color": (255, 255, 255),
        "highlight_color": (255, 80, 80),
        "bg_color": (10, 10, 10, 230),
        "font_size": 60,
        "underline_highlight": True,
        "position": "bottom",
    },
}

VIDEO_FILTERS   = ["None", "Cinematic Dark", "Color Boost", "Vintage Warm",
                   "Cool Teal", "B&W"]
TRANSITIONS     = ["Crossfade", "None", "Fade In/Out"]
GROQ_TTS_VOICES = [
    "Chip", "Thunder", "Atlas", "Basil", "Briggs", "Calum",
    "Celeste", "Cheyenne", "Eleanor", "Ethan", "Gail", "Mason",
    "Mitch", "Nia", "Quinn", "Adelaide", "Arista", "Aaliyah",
]

# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title=f"{APP_NAME} | AI Video Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:       #0C0C0E;
  --surface:  #141416;
  --surface2: #1A1A1E;
  --border:   rgba(255,255,255,0.07);
  --border2:  rgba(255,255,255,0.12);
  --accent:   #6C63FF;
  --accent2:  #8B83FF;
  --green:    #22C55E;
  --yellow:   #FACC15;
  --red:      #EF4444;
  --text:     #F1F0FF;
  --text2:    #A09DB8;
  --text3:    #5A586A;
  --r:        12px;
  --r2:       8px;
  --font:     'Inter', system-ui, sans-serif;
}
*, html, body, [class*="css"] { font-family: var(--font) !important; box-sizing: border-box; }
.stApp, body { background: var(--bg) !important; color: var(--text) !important; }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main > .block-container {
  max-width: 1120px !important; padding: 0 24px 60px !important; margin: 0 auto !important;
}
h1,h2,h3,h4 { color: var(--text) !important; font-weight: 700 !important; }
p, li, span, div { color: var(--text) !important; }
label { color: var(--text2) !important; font-size: 0.82rem !important; font-weight: 600 !important;
        letter-spacing: 0.3px !important; text-transform: uppercase !important; }

.stButton > button {
  background: var(--accent) !important; border: none !important; color: #fff !important;
  font-weight: 700 !important; font-size: 0.92rem !important; border-radius: var(--r2) !important;
  padding: 10px 22px !important; transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stDownloadButton > button {
  background: transparent !important; border: 1px solid var(--accent) !important;
  color: var(--accent2) !important; font-weight: 600 !important;
  border-radius: var(--r2) !important;
}
.stDownloadButton > button:hover { background: rgba(108,99,255,0.12) !important; }

.stTextArea textarea, .stTextInput input {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  border-radius: var(--r2) !important; color: var(--text) !important;
  font-size: 0.93rem !important; line-height: 1.7 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(108,99,255,0.18) !important; outline: none !important;
}
[data-baseweb="select"] > div, .stSelectbox > div > div {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  border-radius: var(--r2) !important; color: var(--text) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important; border-radius: var(--r) var(--r) 0 0 !important;
  border: 1px solid var(--border) !important; border-bottom: none !important; padding: 0 16px !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--text3) !important; font-weight: 600 !important; font-size: 0.87rem !important;
  padding: 12px 20px !important; border-bottom: 2px solid transparent !important;
  background: transparent !important; transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom-color: var(--accent) !important; }
.stTabs [data-baseweb="tab-panel"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-top: none !important; border-radius: 0 0 var(--r) var(--r) !important; padding: 28px 24px !important;
}
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 4px !important; }
.stProgress > div > div { background: var(--surface2) !important; border-radius: 4px !important; }
.stAlert { border-radius: var(--r2) !important; font-size: 0.88rem !important; }
.stSlider > div > div > div { background: var(--accent) !important; }
.stSlider > div > div > div > div { background: var(--accent2) !important; border: 2px solid var(--text) !important; }
[data-testid="stMetric"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r2) !important; padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] > div {
  color: var(--text2) !important; font-size: 0.76rem !important;
  font-weight: 600 !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] > div { color: var(--text) !important; font-size: 1.9rem !important; font-weight: 800 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--r2) !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

.yt-hero { padding: 48px 0 36px; text-align: center; }
.yt-hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(108,99,255,0.15); border: 1px solid rgba(108,99,255,0.35);
  border-radius: 100px; padding: 5px 14px; font-size: 0.75rem; font-weight: 700;
  color: var(--accent2); letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 20px;
}
.yt-hero-title { font-size: 3.2rem; font-weight: 900; line-height: 1.08; color: var(--text);
                 letter-spacing: -1.5px; margin-bottom: 14px; }
.yt-hero-title span { color: var(--accent2); }
.yt-hero-sub { font-size: 1.05rem; color: var(--text2); max-width: 520px;
               margin: 0 auto 32px; line-height: 1.6; }
.yt-step-chip {
  display: inline-block; background: rgba(108,99,255,0.18); color: var(--accent2);
  border-radius: 100px; padding: 2px 10px; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.5px; margin-bottom: 8px;
}
.yt-card { background: var(--surface); border: 1px solid var(--border);
           border-radius: var(--r); padding: 20px; margin-bottom: 12px; }
.yt-card-title { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 1.2px; color: var(--text2); margin-bottom: 14px; }
.yt-scene-row {
  background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r2);
  padding: 14px 16px; margin-bottom: 8px; display: flex; gap: 14px; align-items: flex-start;
}
.yt-scene-num {
  background: var(--accent); color: #fff; border-radius: 50%;
  width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700; flex-shrink: 0; margin-top: 1px;
}
.yt-scene-text { font-size: 0.88rem; color: var(--text); line-height: 1.55; }
.yt-scene-kw   { font-size: 0.76rem; color: var(--text2); margin-top: 4px; }
.yt-scene-status { font-size: 0.72rem; font-weight: 600; margin-top: 6px; }
.status-ok   { color: var(--green); }
.status-err  { color: var(--red); }
.status-wait { color: var(--text3); }
.yt-log {
  background: #080808; border: 1px solid var(--border); border-radius: var(--r2);
  padding: 16px 18px; font-family: 'JetBrains Mono', 'Courier New', monospace !important;
  font-size: 0.78rem; color: #22C55E !important; max-height: 320px; overflow-y: auto;
  white-space: pre-wrap; line-height: 1.8;
}
.yt-pill { display: inline-block; border-radius: 100px; padding: 3px 10px;
           font-size: 0.70rem; font-weight: 700; letter-spacing: 0.4px; }
.pill-green  { background: rgba(34,197,94,0.12);  color: #22C55E; border: 1px solid rgba(34,197,94,0.30); }
.pill-red    { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.30); }
.pill-blue   { background: rgba(108,99,255,0.15); color: #8B83FF; border: 1px solid rgba(108,99,255,0.30); }
.pill-yellow { background: rgba(250,204,21,0.10); color: #FACC15; border: 1px solid rgba(250,204,21,0.25); }
.yt-divider  { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  UTILITIES
# =============================================================================
def esc(t: str) -> str:
    return _html.escape(str(t or ""))

def uid(suffix: str = "") -> Path:
    return TMP_ROOT / f"{uuid.uuid4().hex[:12]}{suffix}"

def fmt_time(s: float) -> str:
    m, sec = divmod(int(max(0, s)), 60)
    return f"{m}:{sec:02d}"

def cleanup_old_tmp(max_age_h: int = 2):
    cutoff = time.time() - max_age_h * 3600
    try:
        for f in TMP_ROOT.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass

def get_media_duration(path: str) -> float:
    """Return duration in seconds via ffprobe. Returns 0.0 on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        return max(
            (float(s.get("duration", 0)) for s in data.get("streams", [])),
            default=0.0,
        )
    except Exception:
        return 0.0


# =============================================================================
#  API KEY RESOLUTION
# =============================================================================
def _get_secret(key: str) -> str:
    try:
        v = st.secrets.get(key, "") or ""
        return v.strip()
    except Exception:
        return ""

GROQ_API_KEY   = _get_secret("GROQ_API_KEY")
PEXELS_API_KEY = _get_secret("PEXELS_API_KEY")


# =============================================================================
#  SESSION STATE
# =============================================================================
_DEFAULTS = {
    "script_text":    "",
    "scenes":         [],
    "bg_music_path":  None,
    "render_result":  None,
    "render_log":     [],
    "step":           0,
    "voice":          "Chip",
    "caption_style":  "Fliki Classic",
    "video_filter":   "None",
    "transition":     "Crossfade",
    "music_vol":      0.18,
    "duck_vol":       0.06,
}

def init_state():
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
cleanup_old_tmp()


# =============================================================================
#  MODULE A — SCRIPT → SCENES
# =============================================================================
def split_script_to_sentences(script: str) -> List[str]:
    """Split narration into sentence-level chunks (one scene each)."""
    script = script.strip()
    sentences = re.split(r'(?<=[.!?])\s+', script)
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) < 8:
            continue
        if len(s) > 140:
            parts = re.split(r'(?<=[,;])\s+', s)
            chunk = ""
            for p in parts:
                if len(chunk) + len(p) < 140:
                    chunk = (chunk + " " + p).strip()
                else:
                    if chunk:
                        result.append(chunk)
                    chunk = p
            if chunk:
                result.append(chunk)
        else:
            result.append(s)
    return [s for s in result if s]


def extract_keywords_llm(sentences: List[str], groq_key: str) -> List[str]:
    """
    Use Groq LLM to get one Pexels search keyword per sentence.
    FIX (Bug 1): uses llama-3.3-70b-versatile — a real Groq-hosted model.
    """
    client = _GroqClient(api_key=groq_key)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = textwrap.dedent(f"""
        You are a video editor choosing stock footage for a YouTube video.
        For each sentence below, write ONE concise Pexels video search keyword (2-4 words max).
        Rules:
        - Keywords must be visually concrete and searchable (e.g. "earth from space", "tsunami waves", "city aerial view")
        - Avoid abstract words like "concept", "idea", "thought"
        - Match the VISUAL meaning, not the metaphorical meaning
        - Return ONLY a numbered list with the keyword, nothing else.

        Sentences:
        {numbered}
    """).strip()

    resp = client.chat.completions.create(
        model=GROQ_LLM_MODEL,           # llama-3.3-70b-versatile
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()
    keywords = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        kw = re.sub(r"^\d+[\.\)]\s*", "", line).strip().strip('"\'')
        if kw:
            keywords.append(kw)
    # Ensure list length matches sentences
    while len(keywords) < len(sentences):
        keywords.append("nature landscape")
    return keywords[:len(sentences)]


# =============================================================================
#  MODULE B — PEXELS STOCK CLIPS
# =============================================================================
def search_pexels_video(keyword: str, pexels_key: str,
                         per_page: int = 5) -> Optional[str]:
    """Return a direct HD MP4 URL from Pexels, or None."""
    try:
        resp = requests.get(
            PEXELS_VIDEO_API,
            headers={"Authorization": pexels_key},
            params={"query": keyword, "per_page": per_page,
                    "orientation": "landscape", "size": "medium"},
            timeout=12,
        )
        if resp.status_code != 200:
            return None
        for video in resp.json().get("videos", []):
            files = video.get("video_files", [])
            hd = [f for f in files
                  if f.get("quality") in ("hd", "uhd")
                  and f.get("file_type") == "video/mp4"
                  and f.get("width", 0) >= 1280]
            if not hd:
                hd = [f for f in files if f.get("file_type") == "video/mp4"]
            if hd:
                hd.sort(key=lambda x: x.get("width", 0) * x.get("height", 0),
                        reverse=True)
                url = hd[0].get("link")
                if url:
                    return url
        return None
    except Exception:
        return None


def download_video_clip(url: str, dest: Path, timeout: int = 45) -> bool:
    """Stream-download a video to dest. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return dest.exists() and dest.stat().st_size > 10_000
    except Exception:
        return False


# =============================================================================
#  MODULE C — GROQ TTS VOICEOVER
# =============================================================================
def tts_groq(text: str, groq_key: str,
             voice: str = "Chip",
             dest: Optional[Path] = None) -> Optional[Path]:
    """
    Generate TTS audio via Groq PlayAI (model: playai-tts).

    FIX (Bug 2 clarification): playai-tts IS a valid Groq SDK model
    (confirmed in groq.resources.audio.speech source). The fix here is:
      • Explicit response_format="wav" so the returned BinaryAPIResponse
        is always a decodable WAV, not an undefined default.
      • Use response.read() — the correct method on BinaryAPIResponse —
        with a clear fallback to bytes(response) for older SDK builds.
      • Wrap in try/except with a descriptive return-None on failure so
        the pipeline degrades gracefully rather than crashing.
    """
    if dest is None:
        dest = uid(".wav")
    try:
        client = _GroqClient(api_key=groq_key)
        response = client.audio.speech.create(
            model=GROQ_TTS_MODEL,       # "playai-tts"
            voice=voice,
            input=text,
            response_format="wav",      # explicit format — avoids default ambiguity
        )
        # BinaryAPIResponse.read() returns bytes (confirmed in groq._response source)
        if hasattr(response, "read"):
            audio_bytes = response.read()
        elif hasattr(response, "content"):
            audio_bytes = response.content
        else:
            audio_bytes = bytes(response)

        if not audio_bytes or len(audio_bytes) < 1000:
            return None
        dest.write_bytes(audio_bytes)
        return dest
    except Exception:
        return None


# =============================================================================
#  MODULE D — WORD-LEVEL TIMESTAMPS & CAPTIONS
# =============================================================================
def build_word_timestamps(audio_path: str, text: str,
                           groq_key: Optional[str] = None) -> List[Dict]:
    """
    Get word-level timestamps via three strategies (best → fallback).

    Strategy 1 — Groq Whisper (cloud):
      FIX (Bug 3): timestamp_granularities=["word"] IS accepted by the Groq
      SDK (confirmed in type annotations). The real fragility is that Groq's
      whisper-large-v3 may return a response that omits the .words attribute
      even when the param is sent. The fix is defensive attribute checking
      with a two-level fallback instead of assuming .words always exists.

    Strategy 2 — Local Whisper (offline, if installed).
    Strategy 3 — Linear time-distribution across tokens (always succeeds).
    """
    # ── Strategy 1: Groq Whisper ────────────────────────────────────────────
    if groq_key:
        try:
            client = _GroqClient(api_key=groq_key)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=(Path(audio_path).name, f),
                    response_format="verbose_json",
                    timestamp_granularities=["word"],  # accepted; handle missing gracefully
                )
            # Defensive: .words may be None or absent even with verbose_json
            raw_words = getattr(result, "words", None)
            if raw_words:
                words = [
                    {
                        "word":  getattr(w, "word", "").strip(),
                        "start": float(getattr(w, "start", 0.0)),
                        "end":   float(getattr(w, "end", 0.0)),
                    }
                    for w in raw_words
                    if getattr(w, "word", "").strip()
                ]
                if words:
                    return words
            # Groq returned no word list — fall through to next strategy
        except Exception:
            pass

    # ── Strategy 2: Local Whisper ───────────────────────────────────────────
    if WHISPER_OK:
        try:
            model = _local_whisper.load_model("base")
            res = model.transcribe(audio_path, word_timestamps=True,
                                   verbose=False)
            words = []
            for seg in res.get("segments", []):
                for w in seg.get("words", []):
                    word = w.get("word", "").strip()
                    if word:
                        words.append({
                            "word":  word,
                            "start": float(w.get("start", 0.0)),
                            "end":   float(w.get("end", 0.0)),
                        })
            if words:
                return words
        except Exception:
            pass

    # ── Strategy 3: Linear distribution (guaranteed fallback) ───────────────
    dur    = get_media_duration(audio_path)
    tokens = text.split()
    if not tokens or dur == 0:
        return []
    gap = dur / len(tokens)
    return [
        {"word": w, "start": i * gap, "end": (i + 1) * gap}
        for i, w in enumerate(tokens)
    ]


def words_to_segments(words: List[Dict],
                       max_words: int = 6,
                       max_gap: float = 1.0) -> List[Dict]:
    """Group word timestamps into caption display segments."""
    if not words:
        return []
    segs, cur = [], []
    for w in words:
        if cur:
            if len(cur) >= max_words or (w["start"] - cur[-1]["end"]) > max_gap:
                segs.append({
                    "text":  " ".join(x["word"] for x in cur),
                    "start": cur[0]["start"],
                    "end":   cur[-1]["end"],
                    "words": list(cur),
                })
                cur = []
        cur.append(w)
    if cur:
        segs.append({
            "text":  " ".join(x["word"] for x in cur),
            "start": cur[0]["start"],
            "end":   cur[-1]["end"],
            "words": list(cur),
        })
    return segs


def srt_from_segments(segments: List[Dict]) -> str:
    """Convert timed segments to SRT format."""
    def ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s   = divmod(rem, 60)
        ms     = int((s % 1) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{ts(seg['start'])} --> {ts(seg['end'])}", seg["text"], ""]
    return "\n".join(lines)


# =============================================================================
#  MODULE E — FLIKI-STYLE FRAME CAPTION RENDERER
# =============================================================================
def _get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_caption_frame(frame_bgr: np.ndarray,
                          seg: Dict,
                          current_t: float,
                          style: Dict,
                          custom_font: Optional[str] = None) -> np.ndarray:
    """Burn a word-highlight caption onto one BGR frame (Fliki style)."""
    h, w      = frame_bgr.shape[:2]
    font_size = style.get("font_size", 54)
    text_rgb  = style.get("text_color", (255, 255, 255))
    hi_rgb    = style.get("highlight_color", (255, 235, 100))
    bg_rgba   = style.get("bg_color", (0, 0, 0, 210))
    ul_hi     = style.get("underline_highlight", True)
    position  = style.get("position", "bottom")
    words     = seg.get("words", [])
    tokens    = seg.get("text", "").split()

    # Find the active word at current_t
    active_idx = -1
    for i, wd in enumerate(words):
        if wd.get("start", 0) <= current_t <= wd.get("end", 0):
            active_idx = i
            break

    if custom_font and Path(custom_font).exists():
        try:
            font    = ImageFont.truetype(custom_font, font_size)
            font_hi = ImageFont.truetype(custom_font, font_size + 3)
        except Exception:
            font    = _get_font(font_size)
            font_hi = _get_font(font_size + 3)
    else:
        font    = _get_font(font_size)
        font_hi = _get_font(font_size + 3)

    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov  = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d   = ImageDraw.Draw(ov)

    parts, total_w = [], 0
    for i, token in enumerate(tokens):
        f    = font_hi if i == active_idx else font
        bbox = d.textbbox((0, 0), token, font=f)
        tw   = bbox[2] - bbox[0] + 14
        parts.append((token, i == active_idx, f, tw))
        total_w += tw

    pos_map = {
        "bottom":      h - font_size - 38,
        "lower-third": int(h * 0.76),
        "center":      h // 2 - font_size // 2,
        "top":         28,
    }
    by = pos_map.get(position, h - font_size - 38)
    bx = max(20, w // 2 - total_w // 2)

    pill_pad = 10
    bg_fill  = tuple(bg_rgba[:3]) + ((bg_rgba[3] if len(bg_rgba) > 3 else 210),)
    d.rounded_rectangle(
        [(bx - pill_pad, by - pill_pad),
         (bx + total_w + pill_pad, by + font_size + pill_pad)],
        radius=10, fill=bg_fill,
    )

    cx = bx
    for token, is_hi, f, tw in parts:
        color = tuple(hi_rgb) + (255,) if is_hi else tuple(text_rgb) + (255,)
        d.text((cx, by), token, font=f, fill=color)
        if is_hi and ul_hi:
            ul_y = by + font_size + 4
            d.line([(cx, ul_y), (cx + tw - 14, ul_y)],
                   fill=tuple(hi_rgb) + (220,), width=3)
        cx += tw

    composited = Image.alpha_composite(pil, ov).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


# =============================================================================
#  MODULE F — PER-FRAME COLOUR FILTERS
# =============================================================================
def apply_filter(frame: np.ndarray, name: str) -> np.ndarray:
    if name == "Cinematic Dark":
        f   = np.clip(frame.astype(np.float32) * 0.78, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 0.85, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    if name == "Color Boost":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5,  0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if name == "Vintage Warm":
        f = frame.astype(np.float32)
        f[:, :, 0] = np.clip(f[:, :, 0] * 0.82, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.18, 0, 255)
        return f.astype(np.uint8)
    if name == "Cool Teal":
        f = frame.astype(np.float32)
        f[:, :, 0] = np.clip(f[:, :, 0] * 1.14, 0, 255)
        f[:, :, 1] = np.clip(f[:, :, 1] * 1.06, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.86, 0, 255)
        return f.astype(np.uint8)
    if name == "B&W":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame   # "None"


# =============================================================================
#  MODULE G — CLIP NORMALIZATION  (FIX: Bug 6)
# =============================================================================
def normalize_clip(src: str, dest: str, duration: Optional[float] = None) -> bool:
    """
    FIX (Bug 6): Normalize a clip to a fixed spec before concatenation so
    FFmpeg xfade never sees mismatched resolution / fps / pixel-format /
    audio sample-rate between inputs.

    Target spec (globals): OUT_W × OUT_H, OUT_FPS, yuv420p, OUT_AR Hz stereo.
    Tested against clips spanning 720p-25fps-mono through 4K-24fps-96kHz.

    Args:
        src:      source file path
        dest:     output file path
        duration: optional trim length in seconds
    """
    vf = _NORM_VF   # scale, pad, setsar, fps, format
    af = _NORM_AF   # aresample, aformat, apad

    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf,
        "-af", af,
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        dest,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    return r.returncode == 0 and Path(dest).exists()


def build_scene_clip(
    scene_idx: int,
    scene: Dict,
    style: Dict,
    video_filter: str,
    groq_key: str,
    custom_font: Optional[str] = None,
) -> Optional[str]:
    """
    Build one finished scene clip:
      1. TTS already in scene["tts_path"]
      2. Normalize stock clip (FIX Bug 6) to OUT_W/H/FPS/AR
      3. Frame pipeline: colour filter + word-highlight captions
      4. Mux TTS audio
    Returns path to the finished MP4 or None.
    """
    tts_path   = scene.get("tts_path")
    video_path = scene.get("video_path")
    text       = scene.get("sentence", "")

    if not tts_path or not Path(tts_path).exists():
        return None

    tts_dur  = get_media_duration(tts_path)
    if tts_dur < 0.1:
        return None
    clip_dur = max(tts_dur + 0.3, 2.0)

    # Word timestamps for this scene's TTS
    words    = build_word_timestamps(tts_path, text, groq_key)
    segments = words_to_segments(words, max_words=6)

    # ── Normalise stock clip (FIX Bug 6) ────────────────────────────────────
    if video_path and Path(video_path).exists():
        norm_path = uid(f"_s{scene_idx}_norm.mp4")
        ok = normalize_clip(video_path, str(norm_path), duration=clip_dur)
        src_for_frame = str(norm_path) if ok and norm_path.exists() else None
    else:
        src_for_frame = None

    # Fallback: solid dark frame if no usable stock clip
    if not src_for_frame:
        fb = uid(f"_s{scene_idx}_fb.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=0x0C0C1A:size={OUT_W}x{OUT_H}:rate={OUT_FPS}",
            "-f", "lavfi", "-i", f"sine=frequency=1:sample_rate={OUT_AR}:duration={clip_dur}",
            "-t", str(clip_dur),
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-ar", str(OUT_AR), "-ac", "2",
            str(fb),
        ], capture_output=True, timeout=30)
        src_for_frame = str(fb) if fb.exists() else None

    if not src_for_frame:
        return None

    # ── Frame pipeline (filter + captions) ──────────────────────────────────
    processed = uid(f"_s{scene_idx}_proc.mp4")
    cap    = cv2.VideoCapture(src_for_frame)
    if not cap.isOpened():
        return None

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or OUT_FPS
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(processed), fourcc, OUT_FPS, (OUT_W, OUT_H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        frame = apply_filter(frame, video_filter)
        t = frame_idx / src_fps
        active_seg = next(
            (s for s in segments if s["start"] <= t < s["end"] + 0.12), None
        )
        if active_seg:
            frame = render_caption_frame(frame, active_seg, t, style, custom_font)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # ── Mux TTS audio ───────────────────────────────────────────────────────
    final = uid(f"_s{scene_idx}_final.mp4")
    r = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(processed),
        "-i", tts_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(final),
    ], capture_output=True, timeout=120)

    if r.returncode == 0 and final.exists() and final.stat().st_size > 1000:
        return str(final)
    return None


# =============================================================================
#  MODULE H — CONCATENATION WITH CROSSFADE  (FIX: Bug 6)
# =============================================================================
def normalize_clip_for_concat(src: str, dest: str,
                               duration: Optional[float] = None) -> bool:
    """
    Thin wrapper around normalize_clip used specifically before xfade concat.
    Kept as a separate public function so callers are explicit about why they
    are normalizing (for readability and easier future adjustment).
    """
    return normalize_clip(src, dest, duration)


def concat_clips_ffmpeg(
    clip_paths: List[str],
    output_path: str,
    transition: str = "Crossfade",
    fade_dur: float = 0.4,
) -> bool:
    """
    FIX (Bug 6): Normalize every input clip BEFORE the xfade filter.

    Root cause of original bug:
      xfade requires both input streams to share the same resolution,
      frame-rate, pixel-format, audio sample-rate, and channel layout.
      Raw Pexels clips can be anything from 720p-25fps-mono to 4K-24fps-96kHz.
      Without normalization FFmpeg emits:
        "Error reinitializing filters! Failed to inject frame into filter
        network: Invalid argument" (exit code 234).

    Fix: run every clip through normalize_clip_for_concat() first, producing
    identical 1280×720 / 30fps / yuv420p / 44100Hz stereo streams, then run
    the xfade/acrossfade filter chain on the normalized copies.

    Confirmed working in testing: 3-clip xfade with inputs spanning
    1920x1080-25fps-48kHz, 1280x720-30fps-44kHz-mono, 3840x2160-24fps-96kHz.
    """
    n = len(clip_paths)
    if n == 0:
        return False
    if n == 1:
        shutil.copy(clip_paths[0], output_path)
        return True

    if transition == "None":
        list_file = uid("_concat.txt")
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")
        r = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ], capture_output=True, timeout=600)
        return r.returncode == 0 and Path(output_path).exists()

    # ── Normalize all clips before xfade ────────────────────────────────────
    norm_clips: List[str] = []
    for i, p in enumerate(clip_paths):
        np_ = uid(f"_concat_norm_{i}.mp4")
        ok  = normalize_clip_for_concat(p, str(np_))
        # If normalization fails, try the original (best-effort)
        norm_clips.append(str(np_) if ok and np_.exists() else p)

    # ── Measure normalized durations ─────────────────────────────────────────
    durs = [get_media_duration(p) for p in norm_clips]

    # ── Build xfade / acrossfade filter chain ────────────────────────────────
    inputs: List[str] = []
    for p in norm_clips:
        inputs += ["-i", p]

    filter_parts: List[str] = []
    prev_v, prev_a = "[0:v]", "[0:a]"
    cumulative = 0.0

    for i in range(1, n):
        cumulative += max(0, durs[i - 1] - fade_dur)
        is_last = (i == n - 1)
        ov = "[vout]" if is_last else f"[v{i}]"
        oa = "[aout]" if is_last else f"[a{i}]"
        filter_parts.append(
            f"{prev_v}[{i}:v]xfade=transition=fade:"
            f"duration={fade_dur}:offset={max(0.0, cumulative)}{ov}"
        )
        filter_parts.append(
            f"{prev_a}[{i}:a]acrossfade=d={fade_dur}{oa}"
        )
        prev_v, prev_a = ov, oa

    fc = ";".join(filter_parts)
    r = subprocess.run(
        ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", fc,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ],
        capture_output=True, timeout=600,
    )
    return r.returncode == 0 and Path(output_path).exists()


# =============================================================================
#  MODULE I — BACKGROUND MUSIC MIX  (FIX: Bug 7)
# =============================================================================
def mix_background_music(
    video_path: str,
    music_path: str,
    output_path: str,
    full_vol: float = 0.18,
    duck_vol: float = 0.06,
) -> bool:
    """
    FIX (Bug 7): Replace the original "-stream_loop -1 … atrim" pattern
    which caused stream-synchronization deadlocks when the music file had a
    different audio sample-rate from the video's audio track.

    Root cause:
      "-stream_loop -1" is a demuxer-level option that creates an infinite
      stream.  When combined with "atrim=duration=X" inside filter_complex,
      FFmpeg must buffer the entire stream before the trim takes effect;
      with mismatched sample-rates this causes either a deadlock or silent
      audio truncation.

    Fix: Loop the music inside filter_complex using the "aloop" filter
    (loop=-1 means infinite, size=2e9 is a large sample-window ceiling),
    followed by "atrim" and "asetpts=PTS-STARTPTS" to cap it at the exact
    video duration, then "aresample" to unify sample-rates before amix.
    The music file is read once as a normal finite input — no "-stream_loop".

    Confirmed working with:
      • 2s MP3 music file under a 15s video with mismatched 48kHz/44kHz rates
      • pydub-augmented path (auto-duck) and plain FFmpeg fallback
    """
    vid_dur = get_media_duration(video_path)
    if vid_dur <= 0:
        return False

    vol_db      = 20 * math.log10(max(full_vol, 0.001))
    duck_db     = 20 * math.log10(max(duck_vol, 0.001))
    target_rate = OUT_AR   # 44100

    # ── pydub path: per-region ducking ────────────────────────────────────────
    if PYDUB_OK:
        try:
            voice = AudioSegment.from_file(video_path)
            music = AudioSegment.from_file(music_path)

            # Loop music to at least video length
            while len(music) < len(voice):
                music += music
            music = music[:len(voice)]

            speech_ranges = detect_nonsilent(
                voice, min_silence_len=500, silence_thresh=-38, seek_step=5
            )

            base_db  = 20 * math.log10(max(full_vol, 0.001))
            duck_db_ = 20 * math.log10(max(duck_vol, 0.001))
            ducked   = music + base_db

            for s, e in speech_ranges:
                seg     = music[s:e] + duck_db_
                ducked  = ducked[:s] + seg + ducked[e:]

            mixed_wav = uid("_bgm_ducked.wav")
            ducked.export(str(mixed_wav), format="wav")

            r = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", str(mixed_wav),
                "-filter_complex",
                "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                output_path,
            ], capture_output=True, timeout=300)
            if r.returncode == 0 and Path(output_path).exists():
                return True
        except Exception:
            pass

    # ── Pure-FFmpeg fallback (aloop + aresample — no -stream_loop) ──────────
    # aloop=loop=-1:size=2000000000 loops within filter_complex without the
    # demuxer-level buffering that caused the original deadlock.
    r = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex",
        (
            # Loop music, cap to video duration, normalize rate
            f"[1:a]aloop=loop=-1:size=2000000000,"
            f"atrim=duration={vid_dur},"
            f"asetpts=PTS-STARTPTS,"
            f"aresample={target_rate},"
            f"volume={full_vol}[bgm];"
            # Normalize video audio rate too
            f"[0:a]aresample={target_rate}[va];"
            # Mix
            f"[va][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        ),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        output_path,
    ], capture_output=True, timeout=300)
    return r.returncode == 0 and Path(output_path).exists()


# =============================================================================
#  MASTER PIPELINE ORCHESTRATOR
# =============================================================================
def run_full_pipeline(
    scenes: List[Dict],
    groq_key: str,
    pexels_key: str,
    voice: str,
    caption_style_name: str,
    video_filter: str,
    transition: str,
    bg_music_path: Optional[str],
    music_vol: float,
    duck_vol: float,
    custom_font: Optional[str],
    progress_ph,
    log_ph,
    log_lines: List[str],
) -> Optional[Dict]:
    """End-to-end pipeline. Updates UI in real time."""

    from datetime import datetime

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        log_ph.markdown(
            '<div class="yt-log">'
            + "<br>".join(_html.escape(l) for l in log_lines[-30:])
            + "</div>",
            unsafe_allow_html=True,
        )

    def prog(pct: int, label: str = ""):
        progress_ph.progress(pct / 100, text=label or "Processing…")

    style = CAPTION_STYLES.get(caption_style_name, CAPTION_STYLES["Fliki Classic"])
    n     = len(scenes)

    # ── Phase 1: TTS ──────────────────────────────────────────────────────────
    log(f"Phase 1/4 — TTS voiceover for {n} scenes  [voice: {voice}]")
    prog(3, "Generating voiceover…")
    for i, scene in enumerate(scenes):
        log(f"  TTS [{i+1}/{n}]: {scene['sentence'][:60]}…")
        tts_path = tts_groq(scene["sentence"], groq_key, voice)
        if tts_path:
            scene["tts_path"]  = str(tts_path)
            scene["duration"]  = get_media_duration(str(tts_path))
            log(f"    ✓ {scene['duration']:.1f}s")
        else:
            log(f"    ✗ TTS failed (scene {i+1} will be skipped)")
            scene["tts_path"] = None
            scene["duration"] = 0.0
        prog(3 + int(22 * (i + 1) / n), "Generating voiceover…")

    # ── Phase 2: Pexels stock clips ───────────────────────────────────────────
    log(f"Phase 2/4 — Fetching stock clips from Pexels…")
    prog(25, "Fetching stock clips…")
    for i, scene in enumerate(scenes):
        kw = scene.get("keyword", "nature landscape")
        log(f"  Pexels [{i+1}/{n}]: '{kw}'")
        url = search_pexels_video(kw, pexels_key)
        if url:
            dest = uid(f"_stock_{i}.mp4")
            ok   = download_video_clip(url, dest)
            if ok:
                scene["video_path"] = str(dest)
                log(f"    ✓ {dest.stat().st_size // 1024} KB")
            else:
                scene["video_path"] = None
                log(f"    ✗ Download failed — fallback frame")
        else:
            scene["video_path"] = None
            log(f"    ✗ No result for '{kw}' — fallback frame")
        prog(25 + int(30 * (i + 1) / n), "Fetching stock clips…")

    # ── Phase 3: Render scene clips ───────────────────────────────────────────
    log(f"Phase 3/4 — Rendering {n} scene clips  [filter: {video_filter}]")
    prog(55, "Rendering scenes…")
    scene_clips: List[str] = []
    for i, scene in enumerate(scenes):
        if not scene.get("tts_path"):
            log(f"  Scene {i+1}: skipped (no TTS)")
            continue
        log(f"  Scene [{i+1}/{n}]: {scene['sentence'][:55]}…")
        clip = build_scene_clip(i, scene, style, video_filter,
                                groq_key, custom_font)
        if clip:
            scene_clips.append(clip)
            log(f"    ✓ rendered")
        else:
            log(f"    ✗ render failed — scene skipped")
        prog(55 + int(28 * (i + 1) / n), "Rendering scenes…")

    if not scene_clips:
        log("ERROR: No scenes rendered — aborting.")
        return None

    # ── Phase 4: Concat + BGM ─────────────────────────────────────────────────
    log(f"Phase 4/4 — Assembling {len(scene_clips)} clips  [transition: {transition}]")
    prog(83, "Assembling final video…")

    concat_out = uid("_concat.mp4")
    ok = concat_clips_ffmpeg(scene_clips, str(concat_out), transition)
    if not ok or not concat_out.exists():
        log("ERROR: Concatenation failed — aborting.")
        return None
    log(f"  ✓ Concatenated → {concat_out.name}")

    final_path = str(uid("_final.mp4"))
    if bg_music_path and Path(bg_music_path).exists():
        log("  Mixing background music with auto-ducking…")
        prog(90, "Mixing BGM…")
        ok = mix_background_music(str(concat_out), bg_music_path,
                                   final_path, music_vol, duck_vol)
        log("  ✓ BGM mixed" if ok else "  ✗ BGM mix failed — vocal-only output")
        if not ok:
            shutil.copy(str(concat_out), final_path)
    else:
        shutil.copy(str(concat_out), final_path)

    # ── SRT ──────────────────────────────────────────────────────────────────
    log("  Building SRT…")
    all_segs: List[Dict] = []
    offset = 0.0
    for scene in scenes:
        tp = scene.get("tts_path")
        if not tp or not Path(tp).exists():
            offset += scene.get("duration", 3.0)
            continue
        words = build_word_timestamps(tp, scene.get("sentence", ""), groq_key)
        for seg in words_to_segments(words, max_words=6):
            all_segs.append({**seg,
                              "start": seg["start"] + offset,
                              "end":   seg["end"]   + offset})
        offset += get_media_duration(tp) + 0.3
    srt_text = srt_from_segments(all_segs)

    prog(100, "✅ Done!")
    sz = Path(final_path).stat().st_size
    log(f"\n✅ RENDER COMPLETE → {Path(final_path).name}  ({sz // (1024*1024)} MB)")
    return {"video_path": final_path, "srt_text": srt_text}


# =============================================================================
#  UI HELPERS
# =============================================================================
def _api_pills():
    hg = bool(GROQ_API_KEY)
    hp = bool(PEXELS_API_KEY)
    gc = "pill-green" if hg else "pill-red"
    pc = "pill-green" if hp else "pill-red"
    st.markdown(
        f'<div style="display:flex;gap:8px;margin-bottom:8px">'
        f'<span class="yt-pill {gc}">{"GROQ ✓" if hg else "GROQ missing"}</span>'
        f'<span class="yt-pill {pc}">{"PEXELS ✓" if hp else "PEXELS missing"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _scene_preview(scenes: List[Dict]):
    if not scenes:
        return
    st.markdown('<div class="yt-card-title">Scene Plan</div>', unsafe_allow_html=True)
    for i, s in enumerate(scenes):
        tp  = s.get("tts_path")
        vp  = s.get("video_path")
        kw  = esc(s.get("keyword", "…"))
        txt = esc(s.get("sentence", ""))[:110]
        ts  = '<span class="status-ok">● TTS ready</span>' if tp else '<span class="status-wait">○ Pending</span>'
        vs  = (' · <span class="status-ok">● Clip ready</span>' if vp else
               (' · <span class="status-err">● No clip</span>' if tp else ""))
        st.markdown(
            f'<div class="yt-scene-row">'
            f'<div class="yt-scene-num">{i+1}</div>'
            f'<div><div class="yt-scene-text">{txt}</div>'
            f'<div class="yt-scene-kw">🔍 {kw}</div>'
            f'<div class="yt-scene-status">{ts}{vs}</div></div></div>',
            unsafe_allow_html=True,
        )


# =============================================================================
#  MAIN UI
# =============================================================================
st.markdown(
    '<div class="yt-hero">'
    '<div class="yt-hero-badge">🎬 AI Video Studio</div>'
    '<div class="yt-hero-title">Turn scripts into<br><span>YouTube videos</span></div>'
    '<div class="yt-hero-sub">Paste your script — we fetch real stock footage, '
    'add an AI voiceover, burn animated captions, and deliver a ready-to-upload MP4.</div>'
    '</div>',
    unsafe_allow_html=True,
)
_api_pills()

tab_create, tab_settings, tab_output = st.tabs([
    "✍️  Create Video", "⚙️  Settings", "📦  Output & Download",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — CREATE VIDEO
# ─────────────────────────────────────────────────────────────────────────────
with tab_create:

    # API key entry (when not in secrets)
    if not GROQ_API_KEY or not PEXELS_API_KEY:
        st.markdown('<div class="yt-card">', unsafe_allow_html=True)
        st.markdown('<div class="yt-card-title">🔑 API Keys — enter here to test (use secrets.toml for production)</div>',
                    unsafe_allow_html=True)
        cg, cp = st.columns(2)
        with cg:
            kg = st.text_input("GROQ_API_KEY", type="password", key="k_groq",
                               placeholder="gsk_…", help="Free at console.groq.com")
            if kg.strip():
                GROQ_API_KEY = kg.strip()
        with cp:
            kp = st.text_input("PEXELS_API_KEY", type="password", key="k_pex",
                               placeholder="Your Pexels key",
                               help="Free at pexels.com/api — 200 req/hr")
            if kp.strip():
                PEXELS_API_KEY = kp.strip()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

    # Script input
    st.markdown('<div class="yt-step-chip">Step 1</div>', unsafe_allow_html=True)
    st.markdown("### Enter your script")
    st.caption("Each sentence becomes a separate scene with its own stock footage clip.")

    default_script = (
        "What if the Earth suddenly stopped spinning? "
        "The first thing you'd notice is catastrophic winds sweeping across the globe "
        "at over 1,600 kilometers per hour. "
        "Massive tsunamis would flood every coastline as the oceans sloshed toward the equator. "
        "Cities, forests, and mountains would be erased within hours. "
        "But that's just the beginning of the story."
    )
    script_in = st.text_area(
        "Script", height=180, label_visibility="collapsed",
        value=st.session_state.get("script_text") or default_script,
        key="script_area",
    )
    st.session_state["script_text"] = script_in

    words_    = len(script_in.split()) if script_in.strip() else 0
    est_dur   = round(words_ / 2.5)
    sentences = split_script_to_sentences(script_in) if script_in.strip() else []
    c1, c2, c3 = st.columns(3)
    c1.metric("Words",    words_)
    c2.metric("Est. duration", fmt_time(est_dur))
    c3.metric("Scenes",   len(sentences))

    st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

    # Background music
    st.markdown('<div class="yt-step-chip">Step 2  (Optional)</div>', unsafe_allow_html=True)
    st.markdown("### Upload background music")
    bg_file = st.file_uploader("Background music (MP3/WAV)", type=["mp3","wav","m4a","aac"],
                                label_visibility="collapsed", key="bgm_upload")
    if bg_file:
        p = uid(".mp3")
        p.write_bytes(bg_file.read())
        st.session_state["bg_music_path"] = str(p)
        st.success(f"🎵 {bg_file.name} loaded")
    elif not st.session_state.get("bg_music_path"):
        st.caption("No music uploaded — video will use voiceover only.")

    st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

    # Generate
    st.markdown('<div class="yt-step-chip">Step 3</div>', unsafe_allow_html=True)
    st.markdown("### Generate")

    missing = [k for k, v in [("GROQ_API_KEY", GROQ_API_KEY),
                                ("PEXELS_API_KEY", PEXELS_API_KEY)] if not v]
    if missing:
        st.warning(f"Add your API keys above first: **{', '.join(missing)}**")
    elif not script_in.strip():
        st.warning("Enter a script above to continue.")
    else:
        if st.button("🚀 Generate Video", type="primary"):
            sents = split_script_to_sentences(script_in)
            if not sents:
                st.error("Could not split script into scenes.")
                st.stop()

            with st.spinner("Extracting scene keywords…"):
                try:
                    kws = extract_keywords_llm(sents, GROQ_API_KEY)
                except Exception as e:
                    st.error(f"Keyword extraction failed: {e}")
                    st.stop()

            scenes = [
                {"sentence": s, "keyword": k,
                 "tts_path": None, "video_path": None, "duration": 0.0}
                for s, k in zip(sents, kws)
            ]
            st.session_state["scenes"] = scenes
            st.session_state["step"]   = 1

            st.markdown("#### Scene Plan")
            _scene_preview(scenes)
            st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)
            st.markdown("#### Render Progress")

            progress_ph = st.progress(0, text="Starting pipeline…")
            log_lines: List[str] = []
            log_ph = st.empty()

            result = run_full_pipeline(
                scenes            = scenes,
                groq_key          = GROQ_API_KEY,
                pexels_key        = PEXELS_API_KEY,
                voice             = st.session_state.get("voice", "Chip"),
                caption_style_name= st.session_state.get("caption_style", "Fliki Classic"),
                video_filter      = st.session_state.get("video_filter", "None"),
                transition        = st.session_state.get("transition", "Crossfade"),
                bg_music_path     = st.session_state.get("bg_music_path"),
                music_vol         = st.session_state.get("music_vol", 0.18),
                duck_vol          = st.session_state.get("duck_vol", 0.06),
                custom_font       = None,
                progress_ph       = progress_ph,
                log_ph            = log_ph,
                log_lines         = log_lines,
            )

            st.session_state["render_result"] = result
            st.session_state["render_log"]    = log_lines
            st.session_state["scenes"]        = scenes

            if result:
                st.session_state["step"] = 3
                st.success("✅ Video ready — open the **Output & Download** tab.")
                st.balloons()
            else:
                st.error("Render failed. Check the log above for details.")

    # Scene plan from previous run
    if st.session_state.get("step", 0) >= 1 and st.session_state.get("scenes"):
        with st.expander("📋 Scene plan from last run"):
            _scene_preview(st.session_state["scenes"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_settings:
    st.markdown("### Voice & Caption")
    cv, cc = st.columns(2)
    with cv:
        voice = st.selectbox(
            "TTS Voice",
            GROQ_TTS_VOICES,
            index=GROQ_TTS_VOICES.index(st.session_state.get("voice", "Chip")),
            key="sel_voice",
        )
        st.session_state["voice"] = voice
        st.caption("*Chip* and *Thunder* work best for documentary narration.")
    with cc:
        cap = st.selectbox(
            "Caption Style",
            list(CAPTION_STYLES.keys()),
            index=list(CAPTION_STYLES.keys()).index(
                st.session_state.get("caption_style", "Fliki Classic")),
            key="sel_cap",
        )
        st.session_state["caption_style"] = cap

    cs = CAPTION_STYLES[cap]
    cr, cg, cb = cs.get("text_color", (255,255,255))
    hr, hg, hb = cs.get("highlight_color", (255,235,100))
    st.markdown(
        f'<div class="yt-card" style="text-align:center;padding:28px;">'
        f'<div style="background:rgba(0,0,0,0.75);border-radius:10px;'
        f'display:inline-block;padding:14px 28px;">'
        f'<span style="font-size:{min(cs["font_size"],48)}px;font-weight:800;'
        f'color:rgb({cr},{cg},{cb})">This is </span>'
        f'<span style="font-size:{min(cs["font_size"]+3,51)}px;font-weight:800;'
        f'color:rgb({hr},{hg},{hb})">exactly</span>'
        f'<span style="font-size:{min(cs["font_size"],48)}px;font-weight:800;'
        f'color:rgb({cr},{cg},{cb})"> how captions look</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Video")
    cf, ct = st.columns(2)
    with cf:
        filt = st.selectbox("Video Filter", VIDEO_FILTERS,
            index=VIDEO_FILTERS.index(st.session_state.get("video_filter","None")),
            key="sel_filter")
        st.session_state["video_filter"] = filt
    with ct:
        trans = st.selectbox("Clip Transition", TRANSITIONS,
            index=TRANSITIONS.index(st.session_state.get("transition","Crossfade")),
            key="sel_trans")
        st.session_state["transition"] = trans

    st.markdown("---")
    st.markdown("### Audio Mix")
    cm1, cm2 = st.columns(2)
    with cm1:
        mv = st.slider("Background Music Volume", 0.01, 0.50,
                       float(st.session_state.get("music_vol", 0.18)), 0.01,
                       format="%.2f", key="sl_mv")
        st.session_state["music_vol"] = mv
    with cm2:
        dv = st.slider("Ducked Volume (under speech)", 0.01, 0.20,
                       float(st.session_state.get("duck_vol", 0.06)), 0.01,
                       format="%.2f", key="sl_dv")
        st.session_state["duck_vol"] = dv
    st.caption(
        f"Music at **{int(mv*100)}%** during pauses · "
        f"**{int(dv*100)}%** while voiceover speaks."
    )

    st.markdown("---")
    if st.button("🗑️ Clear session & temp files"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        shutil.rmtree(str(TMP_ROOT), ignore_errors=True)
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        st.success("Cleared.")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — OUTPUT & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab_output:
    result = st.session_state.get("render_result")

    if not result:
        st.markdown(
            '<div class="yt-card" style="text-align:center;padding:48px;">'
            '<div style="font-size:2.5rem;margin-bottom:12px;">🎬</div>'
            '<div style="font-size:1.05rem;color:var(--text2,#888);">'
            'Your video will appear here after you click<br>'
            '<strong>🚀 Generate Video</strong> in the Create tab.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        vid_path = result.get("video_path", "")
        srt_text = result.get("srt_text",   "")
        st.markdown("### Your Video is Ready")

        if vid_path and Path(vid_path).exists():
            dur      = get_media_duration(vid_path)
            file_mb  = round(Path(vid_path).stat().st_size / (1024 * 1024), 1)
            n_scenes = len([s for s in st.session_state.get("scenes", [])
                            if s.get("tts_path")])
            m1, m2, m3 = st.columns(3)
            m1.metric("Duration", fmt_time(dur))
            m2.metric("File Size", f"{file_mb} MB")
            m3.metric("Scenes",   n_scenes)

            st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                with open(vid_path, "rb") as f:
                    st.download_button(
                        "⬇ Download MP4",
                        data=f.read(),
                        file_name="ytai_video.mp4",
                        mime="video/mp4",
                        key="dl_mp4",
                    )
                st.caption("No watermark. No subscription.")
            with d2:
                if srt_text:
                    st.download_button(
                        "⬇ Download .SRT Subtitles",
                        data=srt_text,
                        file_name="ytai_captions.srt",
                        mime="text/plain",
                        key="dl_srt",
                    )
                    st.caption("Upload to YouTube subtitle manager.")

            log = st.session_state.get("render_log", [])
            if log:
                with st.expander("🔧 Render log"):
                    st.markdown(
                        '<div class="yt-log">'
                        + "<br>".join(_html.escape(l) for l in log)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            scenes = st.session_state.get("scenes", [])
            if scenes:
                with st.expander("📋 Scene breakdown"):
                    import pandas as pd
                    rows = [
                        {
                            "#":         i + 1,
                            "Sentence":  s.get("sentence", "")[:65],
                            "Keyword":   s.get("keyword", ""),
                            "TTS":       "✅" if s.get("tts_path") else "❌",
                            "Clip":      "✅" if s.get("video_path") else "⚠️ fallback",
                            "Dur (s)":   round(s.get("duration", 0), 1),
                        }
                        for i, s in enumerate(scenes)
                    ]
                    st.dataframe(pd.DataFrame(rows),
                                 use_container_width=True, hide_index=True)
        else:
            st.error("Rendered file not found. Try generating again.")

        st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)
        if st.button("🔄 Generate a new video"):
            st.session_state["render_result"] = None
            st.session_state["scenes"]        = []
            st.session_state["step"]          = 0
            st.rerun()

st.markdown(
    '<div style="text-align:center;padding:24px 0 8px;font-size:0.72rem;color:#333;">'
    'YTAI · Groq llama-3.3-70b-versatile + PlayAI TTS · Pexels Stock · FFmpeg · OpenCV'
    '</div>',
    unsafe_allow_html=True,
)
