# =============================================================================
#  YTAi — Script-to-YouTube-Video in One Click
# =============================================================================
#  Fliki-style workflow:
#    Script → AI keyword extraction → Pexels stock clips → Groq TTS voiceover
#    → Word-highlight captions → Auto-duck BGM → Final render
#
#  APIs required
#    GROQ_API_KEY    — Groq (TTS via PlayAI + LLM keyword extraction)
#    PEXELS_API_KEY  — Pexels (royalty-free real stock video, free tier)
#
#  Stack: Streamlit · Groq · MoviePy 2.x · OpenCV · Pydub · Pillow · FFmpeg
# =============================================================================

import asyncio
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

# ── Heavy libs with graceful fallback ────────────────────────────────────────
try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    PYDUB_OK = True
except Exception:
    PYDUB_OK = False

try:
    from moviepy import (
        AudioFileClip, ColorClip, CompositeVideoClip,
        ImageClip, VideoFileClip, afx, vfx,
        concatenate_videoclips,
    )
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
APP_NAME       = "YTAi"
GROQ_LLM_MODEL = "openai/gpt-oss-120b"   # Groq LLM for keyword extraction
GROQ_TTS_MODEL = "playai-tts"            # Groq TTS (English, high quality)
GROQ_TTS_VOICE = "Chip"                  # Clear, authoritative narrator voice

TMP_ROOT = Path(tempfile.gettempdir()) / "ytai_studio"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

PEXELS_VIDEO_API = "https://api.pexels.com/videos/search"
PEXELS_IMAGE_API = "https://api.pexels.com/v1/search"

# Caption visual presets (matching reference Fliki output)
CAPTION_STYLES = {
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

VIDEO_FILTERS = ["None", "Cinematic Dark", "Color Boost", "Vintage Warm", "Cool Teal", "B&W"]
TRANSITIONS   = ["Crossfade", "None", "Fade In/Out"]

GROQ_TTS_VOICES = [
    "Chip", "Thunder", "Atlas", "Basil", "Briggs", "Calum",
    "Celeste", "Cheyenne", "Eleanor", "Ethan", "Gail", "Mason",
    "Mitch", "Nia", "Quinn", "Adelaide", "Arista", "Aaliyah",
]

# =============================================================================
#  PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title=f"{APP_NAME} | AI Video Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
#  CSS — Clean modern dark UI (fresh design, not the previous cluttered one)
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
  --bg:        #0C0C0E;
  --surface:   #141416;
  --surface2:  #1A1A1E;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.12);
  --accent:    #6C63FF;
  --accent2:   #8B83FF;
  --green:     #22C55E;
  --yellow:    #FACC15;
  --red:       #EF4444;
  --text:      #F1F0FF;
  --text2:     #A09DB8;
  --text3:     #5A586A;
  --r:         12px;
  --r2:        8px;
  --font:      'Inter', 'Space Grotesk', system-ui, sans-serif;
}

*, html, body, [class*="css"] {
  font-family: var(--font) !important;
  box-sizing: border-box;
}
.stApp, body { background: var(--bg) !important; color: var(--text) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }

/* ── Main container ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
  max-width: 1120px !important;
  padding: 0 24px 60px !important;
  margin: 0 auto !important;
}

/* ── Typography ── */
h1,h2,h3,h4 { color: var(--text) !important; font-weight: 700 !important; }
p, li, span, div { color: var(--text) !important; }
label { color: var(--text2) !important; font-size: 0.82rem !important; font-weight: 600 !important; letter-spacing: 0.3px !important; text-transform: uppercase !important; }

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 700 !important;
  font-size: 0.92rem !important;
  border-radius: var(--r2) !important;
  padding: 10px 22px !important;
  transition: opacity 0.15s, transform 0.1s !important;
  letter-spacing: 0.2px !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.stDownloadButton > button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent2) !important;
  font-weight: 600 !important;
  border-radius: var(--r2) !important;
  transition: background 0.15s !important;
}
.stDownloadButton > button:hover { background: rgba(108,99,255,0.12) !important; }

/* ── Inputs & Selects ── */
.stTextArea textarea, .stTextInput input, div[data-baseweb="input"] input {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r2) !important;
  color: var(--text) !important;
  font-size: 0.92rem !important;
  transition: border-color 0.15s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(108,99,255,0.18) !important;
  outline: none !important;
}
.stTextArea textarea { font-size: 0.93rem !important; line-height: 1.7 !important; min-height: 160px; }

[data-baseweb="select"] > div, .stSelectbox > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r2) !important;
  color: var(--text) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: var(--r) var(--r) 0 0 !important;
  border: 1px solid var(--border) !important;
  border-bottom: none !important;
  padding: 0 16px !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--text3) !important;
  font-weight: 600 !important;
  font-size: 0.87rem !important;
  padding: 12px 20px !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--r) var(--r) !important;
  padding: 28px 24px !important;
}

/* ── Progress ── */
.stProgress > div > div > div { background: var(--accent) !important; border-radius: 4px !important; }
.stProgress > div > div { background: var(--surface2) !important; border-radius: 4px !important; }

/* ── Alerts ── */
.stAlert { border-radius: var(--r2) !important; font-size: 0.88rem !important; }
[data-testid="stAlertContainer"][data-baseweb="notification"] { border-radius: var(--r2) !important; }

/* ── Slider ── */
.stSlider > div > div > div { background: var(--accent) !important; }
.stSlider > div > div > div > div { background: var(--accent2) !important; border: 2px solid var(--text) !important; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: var(--text3) !important; }

/* ── Toggle ── */
.stToggle span[data-checked="true"] { background: var(--accent) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--r2) !important; padding: 16px 20px !important; }
[data-testid="stMetricLabel"] > div { color: var(--text2) !important; font-size: 0.76rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
[data-testid="stMetricValue"] > div { color: var(--text) !important; font-size: 1.9rem !important; font-weight: 800 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--r2) !important; }

/* ── Custom components ── */
.yt-hero {
  padding: 48px 0 36px;
  text-align: center;
}
.yt-hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(108,99,255,0.15);
  border: 1px solid rgba(108,99,255,0.35);
  border-radius: 100px;
  padding: 5px 14px;
  font-size: 0.75rem; font-weight: 700;
  color: var(--accent2);
  letter-spacing: 0.8px; text-transform: uppercase;
  margin-bottom: 20px;
}
.yt-hero-title {
  font-size: 3.2rem; font-weight: 900;
  line-height: 1.08;
  color: var(--text);
  letter-spacing: -1.5px;
  margin-bottom: 14px;
}
.yt-hero-title span { color: var(--accent2); }
.yt-hero-sub {
  font-size: 1.05rem; color: var(--text2);
  max-width: 520px; margin: 0 auto 32px;
  line-height: 1.6;
}
.yt-step-chip {
  display: inline-block;
  background: rgba(108,99,255,0.18);
  color: var(--accent2);
  border-radius: 100px;
  padding: 2px 10px;
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}
.yt-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 20px;
  margin-bottom: 12px;
}
.yt-card-title {
  font-size: 0.72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--text2); margin-bottom: 14px;
}
.yt-scene-row {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--r2);
  padding: 14px 16px;
  margin-bottom: 8px;
  display: flex; gap: 14px; align-items: flex-start;
}
.yt-scene-num {
  background: var(--accent);
  color: #fff;
  border-radius: 50%;
  width: 26px; height: 26px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700;
  flex-shrink: 0; margin-top: 1px;
}
.yt-scene-text { font-size: 0.88rem; color: var(--text); line-height: 1.55; }
.yt-scene-kw { font-size: 0.76rem; color: var(--text2); margin-top: 4px; }
.yt-scene-status { font-size: 0.72rem; font-weight: 600; margin-top: 6px; }
.status-ok  { color: var(--green); }
.status-err { color: var(--red); }
.status-wait{ color: var(--text3); }

.yt-log {
  background: #080808;
  border: 1px solid var(--border);
  border-radius: var(--r2);
  padding: 16px 18px;
  font-family: 'JetBrains Mono', 'Courier New', monospace !important;
  font-size: 0.78rem;
  color: #22C55E !important;
  max-height: 320px;
  overflow-y: auto;
  white-space: pre-wrap;
  line-height: 1.8;
}
.yt-pill {
  display: inline-block;
  border-radius: 100px;
  padding: 3px 10px;
  font-size: 0.70rem; font-weight: 700;
  letter-spacing: 0.4px;
}
.pill-green { background: rgba(34,197,94,0.12); color: #22C55E; border: 1px solid rgba(34,197,94,0.30); }
.pill-red   { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.30); }
.pill-blue  { background: rgba(108,99,255,0.15); color: #8B83FF; border: 1px solid rgba(108,99,255,0.30); }
.pill-yellow{ background: rgba(250,204,21,0.10); color: #FACC15; border: 1px solid rgba(250,204,21,0.25); }

.yt-divider { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
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


# =============================================================================
#  API KEY MANAGEMENT
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
_STATE_DEFAULTS = {
    "script_text": "",
    "scenes": [],            # List[{sentence, keyword, video_url, video_path, tts_path, duration}]
    "bg_music_path": None,
    "render_result": None,   # {video_path, srt_text}
    "render_log": [],
    "step": 0,               # 0=script, 1=preview, 2=render, 3=done
    "voice": "Chip",
    "caption_style": "Fliki Classic",
    "video_filter": "None",
    "transition": "Crossfade",
    "target_duration": 60,
    "music_vol": 0.18,
    "duck_vol": 0.06,
    "clips_per_sentence": 1,
}

def init_state():
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
cleanup_old_tmp()


# =============================================================================
#  MODULE A — GROQ LLM: SCRIPT → SCENES
# =============================================================================
def split_script_to_sentences(script: str) -> List[str]:
    """Split script into clean sentence-level chunks for scene mapping."""
    script = script.strip()
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', script)
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) < 8:
            continue
        # If very long, split further at commas or semicolons
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
    Use Groq LLM to extract the best Pexels search keyword for each sentence.
    Returns a list of keywords, one per sentence.
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
        model=GROQ_LLM_MODEL,
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
        # Remove leading "1." "1)" etc.
        kw = re.sub(r"^\d+[\.\)]\s*", "", line).strip().strip('"').strip("'")
        if kw:
            keywords.append(kw)
    # Pad or trim to match sentence count
    while len(keywords) < len(sentences):
        keywords.append("nature landscape")
    return keywords[:len(sentences)]


# =============================================================================
#  MODULE B — PEXELS: KEYWORD → STOCK VIDEO
# =============================================================================
def search_pexels_video(keyword: str, pexels_key: str,
                         per_page: int = 5) -> Optional[str]:
    """
    Search Pexels for a landscape HD stock video clip.
    Returns a direct MP4 download URL or None.
    Priority: HD landscape, longest clip first (up to 60s).
    """
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
        data = resp.json()
        videos = data.get("videos", [])
        if not videos:
            return None
        # Pick the video with the most HD files
        best_url = None
        for video in videos:
            files = video.get("video_files", [])
            # Prefer HD (1280x720 or higher)
            hd_files = [f for f in files
                        if f.get("quality") in ("hd", "uhd") and f.get("file_type") == "video/mp4"
                        and f.get("width", 0) >= 1280]
            if not hd_files:
                hd_files = [f for f in files if f.get("file_type") == "video/mp4"]
            if hd_files:
                # Pick highest resolution
                hd_files.sort(key=lambda x: x.get("width", 0) * x.get("height", 0), reverse=True)
                best_url = hd_files[0].get("link")
                if best_url:
                    break
        return best_url
    except Exception:
        return None


def download_video_clip(url: str, dest: Path,
                         timeout: int = 45) -> bool:
    """Stream-download a video file to dest."""
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


def trim_video_to_duration(src: str, dest: str,
                            duration: float, out_w: int = 1280, out_h: int = 720) -> bool:
    """Use FFmpeg to trim + scale a stock clip to the required duration."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", src,
            "-t", str(duration),
            "-vf", (f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"setsar=1"),
            "-r", "30",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-an",   # strip audio; we'll add TTS separately
            dest,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=90)
        return r.returncode == 0 and Path(dest).exists()
    except Exception:
        return False


# =============================================================================
#  MODULE C — GROQ TTS: TEXT → VOICEOVER
# =============================================================================
def tts_groq(text: str, groq_key: str,
             voice: str = "Chip",
             dest: Optional[Path] = None) -> Optional[Path]:
    """
    Generate high-quality TTS audio via Groq PlayAI.
    Returns path to WAV file.
    """
    if dest is None:
        dest = uid(".wav")
    try:
        client = _GroqClient(api_key=groq_key)
        response = client.audio.speech.create(
            model=GROQ_TTS_MODEL,
            voice=voice,
            input=text,
            response_format="wav",
        )
        # response is the raw audio bytes
        audio_bytes = response.read() if hasattr(response, "read") else bytes(response)
        dest.write_bytes(audio_bytes)
        return dest if dest.stat().st_size > 1000 else None
    except Exception as e:
        return None


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds via FFprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            d = float(s.get("duration", 0))
            if d > 0:
                return d
        return 0.0
    except Exception:
        return 0.0


# =============================================================================
#  MODULE D — CAPTIONS: WORD-HIGHLIGHT (Fliki style)
# =============================================================================
def build_word_timestamps(audio_path: str,
                          text: str,
                          groq_key: Optional[str] = None) -> List[Dict]:
    """
    Generate word-level timestamps.
    Strategy:
      1. Try Groq Whisper if key provided (fastest, cloud)
      2. Try local Whisper if installed
      3. Fallback: estimate by splitting evenly across audio duration
    Returns list of {word, start, end}.
    """
    # Strategy 1: Groq Whisper
    if groq_key:
        try:
            client = _GroqClient(api_key=groq_key)
            with open(audio_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=(Path(audio_path).name, f),
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            if hasattr(result, "words") and result.words:
                return [
                    {"word": getattr(w, "word", "").strip(),
                     "start": float(getattr(w, "start", 0)),
                     "end": float(getattr(w, "end", 0))}
                    for w in result.words if getattr(w, "word", "").strip()
                ]
        except Exception:
            pass

    # Strategy 2: Local Whisper
    if WHISPER_OK:
        try:
            model = _local_whisper.load_model("base")
            res = model.transcribe(audio_path, word_timestamps=True, verbose=False)
            words = []
            for seg in res.get("segments", []):
                for w in seg.get("words", []):
                    words.append({"word": w["word"].strip(),
                                  "start": float(w["start"]),
                                  "end": float(w["end"])})
            if words:
                return words
        except Exception:
            pass

    # Strategy 3: Even distribution fallback
    dur = get_audio_duration(audio_path)
    tokens = text.split()
    if not tokens or dur == 0:
        return []
    gap = dur / len(tokens)
    return [{"word": w, "start": i * gap, "end": (i + 1) * gap}
            for i, w in enumerate(tokens)]


def words_to_segments(words: List[Dict],
                       max_words: int = 6,
                       max_gap: float = 1.0) -> List[Dict]:
    """Group word timestamps into caption display segments."""
    if not words:
        return []
    segs, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= max_words or gap > max_gap:
                segs.append({
                    "text": " ".join(x["word"] for x in cur),
                    "start": cur[0]["start"],
                    "end": cur[-1]["end"],
                    "words": list(cur),
                })
                cur = []
        cur.append(w)
    if cur:
        segs.append({
            "text": " ".join(x["word"] for x in cur),
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "words": list(cur),
        })
    return segs


def srt_from_segments(segments: List[Dict]) -> str:
    """Convert segments to SRT subtitle format."""
    def ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        ms = int((s % 1) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{ts(seg['start'])} --> {ts(seg['end'])}", seg["text"], ""]
    return "\n".join(lines)


# =============================================================================
#  MODULE E — FRAME RENDERING (Fliki-style captions)
# =============================================================================
def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the best available bold system font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_frame_with_caption(frame_bgr: np.ndarray,
                               seg: Dict,
                               current_t: float,
                               style: Dict,
                               custom_font: Optional[str] = None) -> np.ndarray:
    """
    Burn a Fliki-style word-highlight caption onto a frame.
    Active word is highlighted; others are white on semi-transparent dark pill.
    """
    h, w = frame_bgr.shape[:2]
    font_size = style.get("font_size", 54)
    text_rgb  = style.get("text_color", (255, 255, 255))
    hi_rgb    = style.get("highlight_color", (255, 235, 100))
    bg_rgba   = style.get("bg_color", (0, 0, 0, 210))
    ul_hi     = style.get("underline_highlight", True)
    position  = style.get("position", "bottom")

    words  = seg.get("words", [])
    tokens = seg.get("text", "").split()

    # Find active word index
    active_idx = -1
    for i, wd in enumerate(words):
        if wd.get("start", 0) <= current_t <= wd.get("end", 0):
            active_idx = i
            break

    # Load fonts
    if custom_font and Path(custom_font).exists():
        try:
            font = ImageFont.truetype(custom_font, font_size)
        except Exception:
            font = _get_font(font_size)
    else:
        font = _get_font(font_size)
    font_hi = _get_font(font_size + 3)

    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov  = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d   = ImageDraw.Draw(ov)

    # Measure token widths
    parts = []
    total_w = 0
    for i, token in enumerate(tokens):
        is_hi = (i == active_idx)
        f = font_hi if is_hi else font
        bbox = d.textbbox((0, 0), token, font=f)
        tw = bbox[2] - bbox[0] + 14  # 7px side padding each
        parts.append((token, is_hi, f, tw))
        total_w += tw

    # Y position
    pad_y = 14
    line_h = font_size + 10
    pos_map = {
        "bottom":      h - line_h - pad_y * 2 - 4,
        "lower-third": int(h * 0.76),
        "center":      h // 2 - line_h // 2,
        "top":         pad_y * 2,
    }
    by = pos_map.get(position, h - line_h - pad_y * 2 - 4)
    bx = max(20, w // 2 - total_w // 2)

    # Draw background pill
    pill_pad = 10
    d.rounded_rectangle(
        [(bx - pill_pad, by - pill_pad),
         (bx + total_w + pill_pad, by + line_h + pill_pad)],
        radius=10,
        fill=tuple(bg_rgba[:3]) + ((bg_rgba[3] if len(bg_rgba) > 3 else 210),),
    )

    # Draw words
    cx = bx
    for i, (token, is_hi, f, tw) in enumerate(parts):
        color = tuple(hi_rgb) + (255,) if is_hi else tuple(text_rgb) + (255,)
        d.text((cx, by), token, font=f, fill=color)
        if is_hi and ul_hi:
            ul_y = by + font_size + 4
            d.line([(cx, ul_y), (cx + tw - 14, ul_y)], fill=tuple(hi_rgb) + (220,), width=3)
        cx += tw

    result = Image.alpha_composite(pil, ov).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


# =============================================================================
#  MODULE F — VIDEO COLOUR FILTERS (per-frame OpenCV)
# =============================================================================
def apply_filter(frame: np.ndarray, name: str) -> np.ndarray:
    if name == "Cinematic Dark":
        f = np.clip(frame.astype(np.float32) * 0.78, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 0.85, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    elif name == "Color Boost":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif name == "Vintage Warm":
        f = frame.astype(np.float32)
        f[:, :, 0] = np.clip(f[:, :, 0] * 0.82, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.18, 0, 255)
        return f.astype(np.uint8)
    elif name == "Cool Teal":
        f = frame.astype(np.float32)
        f[:, :, 0] = np.clip(f[:, :, 0] * 1.14, 0, 255)
        f[:, :, 1] = np.clip(f[:, :, 1] * 1.06, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.86, 0, 255)
        return f.astype(np.uint8)
    elif name == "B&W":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame


# =============================================================================
#  MODULE G — CLIP ASSEMBLY (per-scene video + audio → one clip MP4)
# =============================================================================
def build_scene_clip(
    scene_idx: int,
    scene: Dict,
    style: Dict,
    video_filter: str,
    groq_key: str,
    out_w: int = 1280,
    out_h: int = 720,
    fps: float = 30.0,
    custom_font: Optional[str] = None,
) -> Optional[str]:
    """
    For one scene:
      1. Get TTS audio duration
      2. Trim stock clip to match
      3. Frame-by-frame: apply filter + burn word-highlight caption
      4. Mux TTS audio
    Returns path to the finished scene MP4, or None on failure.
    """
    tts_path   = scene.get("tts_path")
    video_path = scene.get("video_path")
    text       = scene.get("sentence", "")

    if not tts_path or not Path(tts_path).exists():
        return None
    if not video_path or not Path(video_path).exists():
        # If no stock clip, use a black frame
        video_path = None

    tts_dur = get_audio_duration(tts_path)
    if tts_dur < 0.1:
        return None

    clip_dur = max(tts_dur + 0.3, 2.0)  # slight tail after TTS

    # Get word timestamps
    words    = build_word_timestamps(tts_path, text, groq_key)
    segments = words_to_segments(words, max_words=6)

    # Stage 1: trim source clip
    if video_path:
        trimmed = uid(f"_s{scene_idx}_raw.mp4")
        ok = trim_video_to_duration(video_path, str(trimmed), clip_dur, out_w, out_h)
        if not ok or not trimmed.exists():
            video_path = None
        else:
            video_path = str(trimmed)

    if not video_path:
        # Generate a dark gradient fallback frame
        fb_path = uid(f"_s{scene_idx}_fb.mp4")
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=0x0C0C1A:size={out_w}x{out_h}:rate={int(fps)}",
            "-t", str(clip_dur),
            "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-an",
            str(fb_path),
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        video_path = str(fb_path) if fb_path.exists() else None

    if not video_path:
        return None

    # Stage 2: frame-by-frame processing (filter + captions)
    processed = uid(f"_s{scene_idx}_proc.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or fps
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(str(processed), fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        frame = apply_filter(frame, video_filter)
        t = frame_idx / src_fps

        # Find active segment
        active_seg = next(
            (s for s in segments if s["start"] <= t < s["end"] + 0.12),
            None,
        )
        if active_seg:
            frame = render_frame_with_caption(frame, active_seg, t, style, custom_font)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Stage 3: mux TTS audio
    final = uid(f"_s{scene_idx}_final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(processed),
        "-i", tts_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(final),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode == 0 and final.exists() and final.stat().st_size > 1000:
        return str(final)
    return None


# =============================================================================
#  MODULE H — FINAL ASSEMBLY
# =============================================================================
def concat_clips_ffmpeg(clip_paths: List[str],
                         output_path: str,
                         transition: str = "Crossfade",
                         fade_dur: float = 0.4) -> bool:
    """
    Concatenate scene clips via FFmpeg filter_complex with optional crossfade.
    """
    n = len(clip_paths)
    if n == 0:
        return False
    if n == 1:
        shutil.copy(clip_paths[0], output_path)
        return True

    if transition == "None":
        # Simple concat
        list_file = uid("_concat.txt")
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", str(list_file), "-c:v", "libx264", "-preset", "fast",
               "-crf", "18", "-c:a", "aac", "-b:a", "192k", output_path]
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        return r.returncode == 0

    # Crossfade via xfade filter
    # Build complex filter chain
    inputs = []
    for p in clip_paths:
        inputs += ["-i", p]

    # Get durations
    durs = [get_audio_duration(p) for p in clip_paths]

    filter_parts = []
    av_parts = []
    cumulative = 0.0

    # Video xfade chain
    prev_v = "[0:v]"
    prev_a = "[0:a]"
    for i in range(1, n):
        cumulative += durs[i - 1] - fade_dur
        out_v = f"[v{i}]"
        out_a = f"[a{i}]"
        filter_parts.append(
            f"{prev_v}[{i}:v]xfade=transition=fade:duration={fade_dur}:"
            f"offset={max(0, cumulative)}{out_v}"
        )
        filter_parts.append(
            f"{prev_a}[{i}:a]acrossfade=d={fade_dur}{out_a}"
        )
        prev_v = out_v
        prev_a = out_a

    fc = ";".join(filter_parts)
    cmd = (["ffmpeg", "-y"] + inputs +
           ["-filter_complex", fc,
            "-map", prev_v, "-map", prev_a,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path])
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    return r.returncode == 0 and Path(output_path).exists()


def mix_background_music(video_path: str,
                          music_path: str,
                          output_path: str,
                          full_vol: float = 0.18,
                          duck_vol: float = 0.06) -> bool:
    """
    Mix background music under the voiceover with auto-ducking via FFmpeg.
    Simpler approach: lower music to duck_vol across the whole video,
    bring up slightly during pauses (detected from speech track).
    """
    if not PYDUB_OK:
        # Fallback: fixed-volume mix via ffmpeg
        dur_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        try:
            r = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=10)
            vid_dur = float(r.stdout.strip())
        except Exception:
            vid_dur = 60.0

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={duck_vol},atrim=duration={vid_dur}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0.5[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        return r.returncode == 0 and Path(output_path).exists()

    # Full ducking with pydub
    try:
        voice = AudioSegment.from_file(video_path)
        music = AudioSegment.from_file(music_path)

        # Loop music
        while len(music) < len(voice):
            music += music
        music = music[:len(voice)]

        # Detect speech ranges
        speech = detect_nonsilent(voice, min_silence_len=500,
                                   silence_thresh=-38, seek_step=5)

        full_db = 20 * math.log10(max(full_vol, 0.001))
        duck_db = 20 * math.log10(max(duck_vol, 0.001))
        ducked  = music + full_db  # base: full_vol

        for s, e in speech:
            seg = (music[s:e] + duck_db)
            ducked = ducked[:s] + seg + ducked[e:]

        # Re-mux into the video
        mixed_audio = uid("_mixed.wav")
        ducked.export(str(mixed_audio), format="wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", str(mixed_audio),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=1[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        return r.returncode == 0 and Path(output_path).exists()
    except Exception:
        return False


def build_srt_from_scenes(scenes: List[Dict],
                            groq_key: str) -> str:
    """Aggregate per-scene SRT segments into a single video-level SRT."""
    all_segs = []
    offset = 0.0
    for scene in scenes:
        tts = scene.get("tts_path")
        text = scene.get("sentence", "")
        if not tts or not Path(tts).exists():
            offset += scene.get("duration", 3.0)
            continue
        words = build_word_timestamps(tts, text, groq_key)
        segs = words_to_segments(words, max_words=6)
        for seg in segs:
            all_segs.append({
                **seg,
                "start": seg["start"] + offset,
                "end":   seg["end"] + offset,
            })
        offset += get_audio_duration(tts) + 0.3
    return srt_from_segments(all_segs)


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
    out_w: int = 1280,
    out_h: int = 720,
) -> Optional[Dict]:
    """
    End-to-end pipeline.  Updates progress_ph and log_ph in real time.
    Returns {video_path, srt_text} or None.
    """

    def log(msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        log_ph.markdown(
            '<div class="yt-log">' + "<br>".join(_html.escape(l) for l in log_lines[-30:]) + "</div>",
            unsafe_allow_html=True,
        )

    def prog(pct: int, label: str = ""):
        progress_ph.progress(pct / 100, text=label or "Processing…")

    style = CAPTION_STYLES.get(caption_style_name, CAPTION_STYLES["Fliki Classic"])
    n = len(scenes)

    # ── Phase 1: TTS for all scenes ───────────────────────────────────────────
    log(f"Phase 1/4 — Generating TTS voiceover for {n} scenes…")
    prog(3, "Generating voiceover…")
    for i, scene in enumerate(scenes):
        log(f"  TTS [{i+1}/{n}]: {scene['sentence'][:60]}…")
        tts_path = tts_groq(scene["sentence"], groq_key, voice)
        if tts_path:
            scene["tts_path"] = str(tts_path)
            scene["duration"] = get_audio_duration(str(tts_path))
            log(f"    ✓ {scene['duration']:.1f}s")
        else:
            log(f"    ✗ TTS failed for scene {i+1}")
            scene["tts_path"] = None
            scene["duration"] = 0.0
        prog(3 + int(22 * (i + 1) / n), "Generating voiceover…")

    # ── Phase 2: Download Pexels clips ────────────────────────────────────────
    log(f"Phase 2/4 — Fetching {n} stock video clips from Pexels…")
    prog(25, "Fetching stock clips…")
    for i, scene in enumerate(scenes):
        kw = scene.get("keyword", "nature")
        log(f"  Pexels [{i+1}/{n}]: '{kw}'")
        url = search_pexels_video(kw, pexels_key)
        if url:
            dest = uid(f"_stock_{i}.mp4")
            ok = download_video_clip(url, dest)
            if ok:
                scene["video_path"] = str(dest)
                log(f"    ✓ Downloaded {dest.stat().st_size // 1024} KB")
            else:
                scene["video_path"] = None
                log(f"    ✗ Download failed (will use fallback)")
        else:
            scene["video_path"] = None
            log(f"    ✗ No Pexels result for '{kw}' (fallback)")
        prog(25 + int(30 * (i + 1) / n), "Fetching stock clips…")

    # ── Phase 3: Build per-scene clips ────────────────────────────────────────
    log(f"Phase 3/4 — Rendering {n} scene clips…")
    prog(55, "Rendering scenes…")
    scene_clips: List[str] = []
    for i, scene in enumerate(scenes):
        if not scene.get("tts_path"):
            log(f"  Scene {i+1}: skipped (no TTS)")
            continue
        log(f"  Scene [{i+1}/{n}]: {scene['sentence'][:55]}…")
        clip_path = build_scene_clip(
            i, scene, style, video_filter, groq_key,
            out_w, out_h, 30.0, custom_font,
        )
        if clip_path:
            scene_clips.append(clip_path)
            log(f"    ✓ Rendered")
        else:
            log(f"    ✗ Render failed — scene skipped")
        prog(55 + int(28 * (i + 1) / n), "Rendering scenes…")

    if not scene_clips:
        log("ERROR: No scene clips rendered. Aborting.")
        return None

    # ── Phase 4: Concatenate + BGM ────────────────────────────────────────────
    log(f"Phase 4/4 — Assembling {len(scene_clips)} clips…")
    prog(83, "Assembling final video…")

    concat_out = uid("_concat.mp4")
    ok = concat_clips_ffmpeg(scene_clips, str(concat_out), transition)
    if not ok or not concat_out.exists():
        log("ERROR: Concatenation failed. Aborting.")
        return None
    log(f"  ✓ Concatenated {len(scene_clips)} clips → {concat_out.name}")

    final_path = str(uid("_final.mp4"))

    if bg_music_path and Path(bg_music_path).exists():
        log("  Mixing background music with auto-ducking…")
        prog(90, "Mixing background music…")
        ok = mix_background_music(
            str(concat_out), bg_music_path, final_path, music_vol, duck_vol
        )
        if ok:
            log("  ✓ BGM mixed")
        else:
            log("  ✗ BGM mix failed — using vocal-only output")
            shutil.copy(str(concat_out), final_path)
    else:
        shutil.copy(str(concat_out), final_path)

    # Generate combined SRT
    log("  Building SRT subtitle file…")
    srt_text = build_srt_from_scenes(scenes, groq_key)

    prog(100, "✅ Done!")
    log(f"\n✅ RENDER COMPLETE → {Path(final_path).name}")
    log(f"   Size: {Path(final_path).stat().st_size // (1024*1024)} MB")

    return {"video_path": final_path, "srt_text": srt_text}


# =============================================================================
#  UI HELPERS
# =============================================================================
def _api_status_bar():
    """Show small API status pills at top."""
    has_groq   = bool(GROQ_API_KEY)
    has_pexels = bool(PEXELS_API_KEY)
    g_cls = "pill-green" if has_groq   else "pill-red"
    p_cls = "pill-green" if has_pexels else "pill-red"
    g_txt = "GROQ ✓"     if has_groq   else "GROQ missing"
    p_txt = "PEXELS ✓"   if has_pexels else "PEXELS missing"
    st.markdown(
        f'<div style="display:flex;gap:8px;margin-bottom:8px">'
        f'<span class="yt-pill {g_cls}">{g_txt}</span>'
        f'<span class="yt-pill {p_cls}">{p_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _scene_preview(scenes: List[Dict]):
    """Render the scene plan as a visual card list."""
    if not scenes:
        return
    st.markdown('<div class="yt-card-title">Scene Plan</div>', unsafe_allow_html=True)
    for i, s in enumerate(scenes):
        kw   = esc(s.get("keyword", "…"))
        text = esc(s.get("sentence", ""))[:120]
        tp   = s.get("tts_path")
        vp   = s.get("video_path")
        if tp:
            status = '<span class="status-ok">● TTS ready</span>'
        else:
            status = '<span class="status-wait">○ Pending</span>'
        if vp:
            vstatus = ' · <span class="status-ok">● Clip ready</span>'
        elif tp is not None:
            vstatus = ' · <span class="status-err">● No clip</span>'
        else:
            vstatus = ""
        st.markdown(
            f'<div class="yt-scene-row">'
            f'  <div class="yt-scene-num">{i+1}</div>'
            f'  <div>'
            f'    <div class="yt-scene-text">{text}</div>'
            f'    <div class="yt-scene-kw">🔍 {kw}</div>'
            f'    <div class="yt-scene-status">{status}{vstatus}</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
#  MAIN UI
# =============================================================================

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="yt-hero">'
    '  <div class="yt-hero-badge">🎬 AI Video Studio</div>'
    '  <div class="yt-hero-title">Turn scripts into<br><span>YouTube videos</span></div>'
    '  <div class="yt-hero-sub">Paste your script. We fetch real stock footage, add a premium AI voiceover, '
    'burn animated captions, and deliver a ready-to-upload MP4.</div>'
    '</div>',
    unsafe_allow_html=True,
)
_api_status_bar()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_create, tab_settings, tab_output = st.tabs([
    "✍️  Create Video", "⚙️  Settings", "📦  Output & Download",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CREATE VIDEO
# ─────────────────────────────────────────────────────────────────────────────
with tab_create:

    # ── API key quick-entry (shown when not in secrets) ──
    if not GROQ_API_KEY or not PEXELS_API_KEY:
        st.markdown('<div class="yt-card">', unsafe_allow_html=True)
        st.markdown('<div class="yt-card-title">🔑 API Keys (stored in st.secrets — enter here to test)</div>', unsafe_allow_html=True)
        col_g, col_p = st.columns(2)
        with col_g:
            k_groq = st.text_input("GROQ_API_KEY", type="password", key="k_groq",
                                    placeholder="gsk_…",
                                    help="Get free at console.groq.com")
            if k_groq.strip():
                GROQ_API_KEY = k_groq.strip()
        with col_p:
            k_pex = st.text_input("PEXELS_API_KEY", type="password", key="k_pex",
                                   placeholder="Your Pexels key",
                                   help="Free at pexels.com/api — 200 req/hr")
            if k_pex.strip():
                PEXELS_API_KEY = k_pex.strip()
        st.markdown(
            '<div style="font-size:0.78rem;color:var(--text2,#888);margin-top:8px">'
            'For production: add these to <code>.streamlit/secrets.toml</code> so they persist.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

    # ── Script input ──────────────────────────────────────────────────────────
    st.markdown('<div class="yt-step-chip">Step 1</div>', unsafe_allow_html=True)
    st.markdown("### Enter your script")
    st.caption("Write or paste the narration for your video. Each sentence becomes a separate scene with its own stock clip.")

    default_script = (
        "What if the Earth suddenly stopped spinning? "
        "The first thing you'd notice is catastrophic winds sweeping across the globe at over 1,600 kilometers per hour. "
        "Massive tsunamis would flood every coastline as the oceans sloshed toward the equator. "
        "Cities, forests, and mountains would be erased within hours. "
        "But that's just the beginning of the story."
    )
    script_in = st.text_area(
        "Script",
        value=st.session_state.get("script_text") or default_script,
        height=180,
        label_visibility="collapsed",
        placeholder="What if the Earth suddenly stopped spinning?…",
        key="script_area",
    )
    st.session_state["script_text"] = script_in

    # Word count + estimated duration
    words = len(script_in.split()) if script_in.strip() else 0
    est_dur = round(words / 2.5)  # ~150 wpm
    col_wc1, col_wc2, col_wc3 = st.columns(3)
    col_wc1.metric("Words", words)
    col_wc2.metric("Est. Duration", f"{fmt_time(est_dur)}")
    col_wc3.metric("Scenes", len(split_script_to_sentences(script_in)) if script_in.strip() else 0)

    st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

    # ── Background music ──────────────────────────────────────────────────────
    st.markdown('<div class="yt-step-chip">Step 2  (Optional)</div>', unsafe_allow_html=True)
    st.markdown("### Upload background music")
    bg_music_file = st.file_uploader(
        "Background music",
        type=["mp3", "wav", "m4a", "aac"],
        label_visibility="collapsed",
        key="bgm_upload",
    )
    if bg_music_file:
        p = uid(".mp3")
        p.write_bytes(bg_music_file.read())
        st.session_state["bg_music_path"] = str(p)
        st.success(f"🎵 {bg_music_file.name} loaded")
    elif not st.session_state.get("bg_music_path"):
        st.caption("No music uploaded — video will use voiceover audio only.")

    st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

    # ── Generate ──────────────────────────────────────────────────────────────
    st.markdown('<div class="yt-step-chip">Step 3</div>', unsafe_allow_html=True)
    st.markdown("### Generate")

    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not PEXELS_API_KEY:
        missing_keys.append("PEXELS_API_KEY")

    if missing_keys:
        st.warning(f"Add your API keys above first: **{', '.join(missing_keys)}**")
    elif not script_in.strip():
        st.warning("Enter a script above to continue.")
    else:
        gen_btn = st.button("🚀 Generate Video", type="primary")

        if gen_btn:
            sentences = split_script_to_sentences(script_in)
            if not sentences:
                st.error("Could not split script into scenes.")
                st.stop()

            # Extract keywords via LLM
            with st.spinner("Extracting scene keywords via AI…"):
                try:
                    keywords = extract_keywords_llm(sentences, GROQ_API_KEY)
                except Exception as e:
                    st.error(f"Keyword extraction failed: {e}")
                    st.stop()

            scenes = [
                {"sentence": s, "keyword": k, "tts_path": None,
                 "video_path": None, "duration": 0.0}
                for s, k in zip(sentences, keywords)
            ]
            st.session_state["scenes"] = scenes
            st.session_state["step"]   = 1

            # Show scene plan
            st.markdown("#### Scene Plan")
            _scene_preview(scenes)
            st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

            # Live render
            st.markdown("#### Render Progress")
            progress_ph = st.progress(0, text="Starting pipeline…")
            log_lines: List[str] = []
            log_ph = st.empty()

            cap_style  = st.session_state.get("caption_style",  "Fliki Classic")
            vid_filter = st.session_state.get("video_filter",   "None")
            transition = st.session_state.get("transition",     "Crossfade")
            voice      = st.session_state.get("voice",          "Chip")
            music_vol  = st.session_state.get("music_vol",      0.18)
            duck_vol   = st.session_state.get("duck_vol",       0.06)
            bgm_path   = st.session_state.get("bg_music_path")

            result = run_full_pipeline(
                scenes       = scenes,
                groq_key     = GROQ_API_KEY,
                pexels_key   = PEXELS_API_KEY,
                voice        = voice,
                caption_style_name = cap_style,
                video_filter = vid_filter,
                transition   = transition,
                bg_music_path= bgm_path,
                music_vol    = music_vol,
                duck_vol     = duck_vol,
                custom_font  = None,
                progress_ph  = progress_ph,
                log_ph       = log_ph,
                log_lines    = log_lines,
            )

            st.session_state["render_result"] = result
            st.session_state["render_log"]    = log_lines
            st.session_state["scenes"]        = scenes  # updated with paths

            if result:
                st.session_state["step"] = 3
                st.success("✅ Video generated! Go to the **Output & Download** tab to get your file.")
                st.balloons()
            else:
                st.error("Render failed. Check the log below.")

    # Show existing scene plan if available
    if st.session_state.get("scenes") and not (
        script_in and st.session_state.get("step", 0) == 0
    ):
        if st.session_state.get("step", 0) >= 1:
            with st.expander("📋 Scene plan from last run"):
                _scene_preview(st.session_state["scenes"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab_settings:
    st.markdown("### Voice & Caption Settings")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown('<div class="yt-card-title">🎙️ TTS Voice</div>', unsafe_allow_html=True)
        voice = st.selectbox(
            "Voice",
            GROQ_TTS_VOICES,
            index=GROQ_TTS_VOICES.index(st.session_state.get("voice", "Chip")),
            label_visibility="collapsed",
            key="sel_voice",
        )
        st.session_state["voice"] = voice
        st.caption("Powered by Groq PlayAI TTS. *Chip* and *Thunder* work great for documentary narration.")

    with col_v2:
        st.markdown('<div class="yt-card-title">📝 Caption Style</div>', unsafe_allow_html=True)
        cap = st.selectbox(
            "Caption style",
            list(CAPTION_STYLES.keys()),
            index=list(CAPTION_STYLES.keys()).index(st.session_state.get("caption_style", "Fliki Classic")),
            label_visibility="collapsed",
            key="sel_cap",
        )
        st.session_state["caption_style"] = cap

    # Caption preview
    cs = CAPTION_STYLES[cap]
    cr, cg, cb = cs.get("text_color", (255,255,255))
    hr, hg, hb = cs.get("highlight_color", (255,235,100))
    st.markdown(
        f'<div class="yt-card" style="text-align:center;padding:28px;">'
        f'<div style="background:rgba(0,0,0,0.75);border-radius:10px;display:inline-block;padding:14px 28px;">'
        f'<span style="font-size:{min(cs["font_size"],48)}px;font-weight:800;'
        f'color:rgb({cr},{cg},{cb})">This is </span>'
        f'<span style="font-size:{min(cs["font_size"]+3,51)}px;font-weight:800;'
        f'color:rgb({hr},{hg},{hb});border-bottom:3px solid rgb({hr},{hg},{hb}22)">exactly</span>'
        f'<span style="font-size:{min(cs["font_size"],48)}px;font-weight:800;'
        f'color:rgb({cr},{cg},{cb})"> how it looks</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Video Settings")

    col_f, col_t = st.columns(2)
    with col_f:
        filt = st.selectbox(
            "Video Filter",
            VIDEO_FILTERS,
            index=VIDEO_FILTERS.index(st.session_state.get("video_filter", "None")),
            key="sel_filter",
        )
        st.session_state["video_filter"] = filt

    with col_t:
        trans = st.selectbox(
            "Clip Transition",
            TRANSITIONS,
            index=TRANSITIONS.index(st.session_state.get("transition", "Crossfade")),
            key="sel_trans",
        )
        st.session_state["transition"] = trans

    st.markdown("---")
    st.markdown("### Audio Mix")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mv = st.slider(
            "Background Music Volume",
            0.01, 0.50, float(st.session_state.get("music_vol", 0.18)), 0.01,
            format="%.2f", key="sl_mv",
        )
        st.session_state["music_vol"] = mv
    with col_m2:
        dv = st.slider(
            "Ducked Volume (under speech)",
            0.01, 0.20, float(st.session_state.get("duck_vol", 0.06)), 0.01,
            format="%.2f", key="sl_dv",
        )
        st.session_state["duck_vol"] = dv

    st.caption(f"Music plays at **{int(mv*100)}%** during pauses and drops to **{int(dv*100)}%** while the voiceover speaks.")

    st.markdown("---")
    st.markdown("### Advanced")

    if st.button("🗑️ Clear all temp files & reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        try:
            shutil.rmtree(str(TMP_ROOT), ignore_errors=True)
            TMP_ROOT.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        st.success("Session cleared.")
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — OUTPUT & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab_output:
    result = st.session_state.get("render_result")

    if not result:
        st.markdown(
            '<div class="yt-card" style="text-align:center;padding:48px;">'
            '<div style="font-size:2.5rem;margin-bottom:12px;">🎬</div>'
            '<div style="font-size:1.05rem;color:var(--text2,#888);">'
            'Your rendered video will appear here.<br>'
            'Go to <strong>Create Video</strong> and click <strong>Generate Video</strong>.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        vid_path = result.get("video_path", "")
        srt_text = result.get("srt_text", "")

        st.markdown("### Your Video is Ready")

        if vid_path and Path(vid_path).exists():
            file_mb = round(Path(vid_path).stat().st_size / (1024 * 1024), 1)

            # Duration
            dur = get_audio_duration(vid_path)
            n_scenes = len([s for s in st.session_state.get("scenes", []) if s.get("tts_path")])

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Duration", fmt_time(dur))
            col_m2.metric("File Size", f"{file_mb} MB")
            col_m3.metric("Scenes", n_scenes)

            st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)

            # Downloads
            dl1, dl2 = st.columns(2)
            with dl1:
                with open(vid_path, "rb") as f:
                    st.download_button(
                        "⬇ Download MP4 Video",
                        data=f.read(),
                        file_name="ytai_video.mp4",
                        mime="video/mp4",
                        key="dl_mp4",
                    )
                st.caption("Ready to upload to YouTube — no watermark, no subscription required.")

            with dl2:
                if srt_text:
                    st.download_button(
                        "⬇ Download .SRT Subtitles",
                        data=srt_text,
                        file_name="ytai_captions.srt",
                        mime="text/plain",
                        key="dl_srt",
                    )
                    st.caption("Upload this to YouTube's subtitle manager for auto-captions.")

            # Render log
            log = st.session_state.get("render_log", [])
            if log:
                with st.expander("🔧 Render log"):
                    st.markdown(
                        '<div class="yt-log">' +
                        "<br>".join(_html.escape(l) for l in log) +
                        "</div>",
                        unsafe_allow_html=True,
                    )

            # Scene breakdown table
            scenes = st.session_state.get("scenes", [])
            if scenes:
                with st.expander("📋 Scene breakdown"):
                    rows = []
                    for i, s in enumerate(scenes):
                        rows.append({
                            "#": i + 1,
                            "Sentence": s.get("sentence", "")[:70],
                            "Keyword": s.get("keyword", ""),
                            "TTS": "✅" if s.get("tts_path") else "❌",
                            "Clip": "✅" if s.get("video_path") else "⚠️ fallback",
                            "Dur (s)": round(s.get("duration", 0), 1),
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        else:
            st.error("Rendered file not found on disk. Try generating again.")

        # Generate new video button
        st.markdown('<hr class="yt-divider">', unsafe_allow_html=True)
        if st.button("🔄 Generate a New Video"):
            st.session_state["render_result"] = None
            st.session_state["scenes"] = []
            st.session_state["step"] = 0
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;padding:24px 0 8px;font-size:0.72rem;color:#333;">'
    'YTAI · Built on Groq PlayAI TTS + Pexels Stock + FFmpeg + OpenCV'
    '</div>',
    unsafe_allow_html=True,
)
