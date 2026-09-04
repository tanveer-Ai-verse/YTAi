"""
ShortsCraft AI — Automated Long-Video-to-Shorts Conversion
app.py — Main Streamlit application
"""

import os
import sys
import json
import time
import math
import shutil
import hashlib
import logging
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Optional

import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShortsCraft AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0d0d0f; color: #e8e8ed; }

    /* ── Hero banner ── */
    .sc-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .sc-hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 .5rem 0;
    }
    .sc-hero p { color: #94a3b8; font-size: 1rem; margin: 0; }

    /* ── Cards ── */
    .sc-card {
        background: #12121a;
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Step pills ── */
    .sc-step {
        display: inline-block;
        background: rgba(99,102,241,.15);
        border: 1px solid rgba(99,102,241,.35);
        border-radius: 999px;
        padding: .25rem .8rem;
        font-size: .75rem;
        font-weight: 600;
        color: #818cf8;
        letter-spacing: .04em;
        margin-bottom: .6rem;
    }

    /* ── Progress text ── */
    .sc-progress { color: #94a3b8; font-size: .88rem; }

    /* ── Clip card ── */
    .clip-card {
        background: #1a1a2e;
        border: 1px solid rgba(99,102,241,.2);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: .75rem;
    }
    .clip-title { font-weight: 600; color: #c7d2fe; }
    .clip-meta  { color: #64748b; font-size: .82rem; }

    /* ── Streamlit overrides ── */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: .55rem 1.4rem;
        transition: opacity .2s;
    }
    .stButton>button:hover { opacity: .85; }
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div {
        background: #12121a !important;
        border-color: rgba(99,102,241,.3) !important;
        color: #e8e8ed !important;
    }
    hr { border-color: rgba(99,102,241,.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ShortsCraftAI")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
WORK_DIR        = Path(tempfile.gettempdir()) / "shortscraftai"
AUDIO_DIR       = WORK_DIR / "audio"
VIDEO_DIR       = WORK_DIR / "video"
CLIPS_DIR       = WORK_DIR / "clips"
FINAL_DIR       = WORK_DIR / "final"
SFX_DIR         = WORK_DIR / "sfx"

for _d in (AUDIO_DIR, VIDEO_DIR, CLIPS_DIR, FINAL_DIR, SFX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MIN_CLIP_SEC  = 15
MAX_CLIP_SEC  = 60
MAX_CLIPS     = 10
TARGET_W      = 1080
TARGET_H      = 1920

SCENE_CATEGORIES = [
    "Funny Clips",
    "Fight Scenes",
    "High Drama",
    "Educational Takes",
    "Action Cues",
    "Inspirational Moments",
    "Plot Twists",
    "Emotional Peaks",
]

CAPTION_PRESETS = {
    "Bold Yellow Highlight": {
        "fontsize": 72,
        "color":    "yellow",
        "stroke_color": "black",
        "stroke_width": 3,
        "font":     "Impact",
        "bg_color": None,
        "position": ("center", 0.75),
    },
    "Hormozi Style": {
        "fontsize": 80,
        "color":    "white",
        "stroke_color": "black",
        "stroke_width": 4,
        "font":     "Arial-Bold",
        "bg_color": None,
        "position": ("center", 0.78),
    },
    "Minimal White": {
        "fontsize": 58,
        "color":    "white",
        "stroke_color": "black",
        "stroke_width": 2,
        "font":     "Helvetica",
        "bg_color": None,
        "position": ("center", 0.80),
    },
    "Cyber Neon": {
        "fontsize": 68,
        "color":    "#00ffff",
        "stroke_color": "#ff00ff",
        "stroke_width": 3,
        "font":     "Courier",
        "bg_color": None,
        "position": ("center", 0.76),
    },
    "Red Impact": {
        "fontsize": 76,
        "color":    "#ff1a1a",
        "stroke_color": "white",
        "stroke_width": 3,
        "font":     "Impact",
        "bg_color": None,
        "position": ("center", 0.77),
    },
}

SFX_PROFILES = {
    "whoosh":    {"freq": 800,  "duration": 0.4, "sweep_end": 200},
    "impact":    {"freq": 60,   "duration": 0.5, "sweep_end": 40},
    "transition":{"freq": 1200, "duration": 0.3, "sweep_end": 600},
    "dramatic":  {"freq": 150,  "duration": 0.8, "sweep_end": 80},
}

PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
WHISPER_MODEL  = "whisper-large-v3-turbo"


# ─────────────────────────────────────────────
# SECRETS / API KEY
# ─────────────────────────────────────────────
def get_groq_api_key() -> Optional[str]:
    """Return GROQ_API_KEY from Streamlit Secrets or env, or None."""
    try:
        key = st.secrets.get("GROQ_API_KEY", None)
        if key:
            return key.strip()
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY", None)
    if key:
        return key.strip()
    return None


# ─────────────────────────────────────────────
# DEPENDENCY CHECKS
# ─────────────────────────────────────────────
def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_yt_dlp() -> bool:
    try:
        import yt_dlp  # noqa
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────
# YT-DLP DOWNLOAD
# ─────────────────────────────────────────────
def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]


def download_video(url: str, progress_cb=None) -> dict:
    """
    Download video + audio from *url* using yt-dlp.
    Returns dict with keys: video_path, audio_path, title, duration_sec
    """
    import yt_dlp

    uid      = _url_hash(url)
    vid_path = VIDEO_DIR / f"{uid}_video.mp4"
    aud_path = AUDIO_DIR / f"{uid}_audio.m4a"

    info_result = {}

    # ── probe metadata ──
    ydl_opts_info = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(url, download=False)
        info_result["title"]        = info.get("title", "Unknown Title")
        info_result["duration_sec"] = float(info.get("duration", 0) or 0)

    # ── download video (best mp4 up to 1080p) ──
    if not vid_path.exists():
        def _video_hook(d):
            if progress_cb and d.get("status") == "downloading":
                pct = d.get("_percent_str", "?%")
                progress_cb(f"Downloading video … {pct}")

        ydl_opts_v = {
            "format":            "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl":           str(vid_path),
            "quiet":             True,
            "no_warnings":       True,
            "merge_output_format": "mp4",
            "progress_hooks":    [_video_hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts_v) as ydl:
            ydl.download([url])
    else:
        log.info("Video already cached: %s", vid_path)

    # ── download audio ──
    if not aud_path.exists():
        def _audio_hook(d):
            if progress_cb and d.get("status") == "downloading":
                pct = d.get("_percent_str", "?%")
                progress_cb(f"Downloading audio … {pct}")

        ydl_opts_a = {
            "format":          "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl":         str(aud_path),
            "quiet":           True,
            "no_warnings":     True,
            "progress_hooks":  [_audio_hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts_a) as ydl:
            ydl.download([url])
    else:
        log.info("Audio already cached: %s", aud_path)

    # ── fallback: extract audio from video if m4a still missing ──
    if not aud_path.exists() and vid_path.exists():
        log.warning("m4a not found — extracting audio from video with ffmpeg")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(vid_path), "-vn",
             "-acodec", "aac", "-b:a", "192k", str(aud_path)],
            capture_output=True, check=True,
        )

    if not vid_path.exists():
        raise FileNotFoundError(f"Video download failed — file not found: {vid_path}")
    if not aud_path.exists():
        raise FileNotFoundError(f"Audio download failed — file not found: {aud_path}")

    info_result["video_path"] = str(vid_path)
    info_result["audio_path"] = str(aud_path)
    return info_result


# ─────────────────────────────────────────────
# AUDIO CHUNKING & TRANSCRIPTION
# ─────────────────────────────────────────────
def split_audio_chunks(audio_path: str, chunk_sec: int = 300) -> list[str]:
    """Split *audio_path* into ≤chunk_sec chunks and return list of chunk paths."""
    import subprocess

    src  = Path(audio_path)
    base = AUDIO_DIR / f"{src.stem}_chunk"

    # Probe duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         str(src)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        total_sec = float(result.stdout.strip())
    except ValueError:
        total_sec = 3600.0

    n_chunks = math.ceil(total_sec / chunk_sec)
    chunks   = []

    for i in range(n_chunks):
        start    = i * chunk_sec
        out_path = Path(f"{base}_{i:03d}.m4a")
        if not out_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-ss", str(start), "-t", str(chunk_sec),
                 "-vn", "-acodec", "copy", str(out_path)],
                capture_output=True, check=True,
            )
        chunks.append(str(out_path))

    return chunks


def transcribe_audio(audio_path: str, api_key: str, progress_cb=None) -> dict:
    """
    Transcribe *audio_path* via Groq Whisper.
    Returns {"text": str, "segments": [...], "duration": float}
    """
    from groq import Groq

    client  = Groq(api_key=api_key)
    chunks  = split_audio_chunks(audio_path)
    all_seg = []
    all_txt = []
    offset  = 0.0

    for idx, chunk_path in enumerate(chunks):
        if progress_cb:
            progress_cb(f"Transcribing chunk {idx + 1}/{len(chunks)} …")

        chunk_file = Path(chunk_path)
        if not chunk_file.exists() or chunk_file.stat().st_size == 0:
            log.warning("Chunk %s is missing or empty — skipping", chunk_path)
            continue

        for attempt in range(3):
            try:
                with open(chunk_path, "rb") as fh:
                    resp = client.audio.transcriptions.create(
                        model       = WHISPER_MODEL,
                        file        = fh,
                        response_format = "verbose_json",
                        timestamp_granularities = ["segment"],
                    )
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Whisper transcription failed after 3 attempts: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)

        raw_text = getattr(resp, "text", "") or ""
        all_txt.append(raw_text)

        segments = getattr(resp, "segments", None) or []
        for seg in segments:
            seg_dict = {
                "id":    getattr(seg, "id",    len(all_seg)),
                "start": float(getattr(seg, "start", 0)) + offset,
                "end":   float(getattr(seg, "end",   0)) + offset,
                "text":  getattr(seg, "text",  ""),
            }
            all_seg.append(seg_dict)

        # Advance offset by the duration of this chunk
        if segments:
            last_end = float(getattr(segments[-1], "end", 0))
            offset  += last_end
        else:
            # Fallback: probe chunk duration
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", chunk_path],
                capture_output=True, text=True, timeout=20,
            )
            try:
                offset += float(result.stdout.strip())
            except ValueError:
                pass

    return {
        "text":     " ".join(all_txt),
        "segments": all_seg,
        "duration": offset,
    }


# ─────────────────────────────────────────────
# SCENE DETECTION VIA GROQ LLAMA
# ─────────────────────────────────────────────
_SCENE_SYSTEM = """\
You are a world-class video editor specializing in viral social media shorts.
Your task is to analyze a video transcript and identify the best clip segments.

Rules:
- Each clip MUST be between {min_sec} and {max_sec} seconds long (end - start).
- Return EXACTLY {max_clips} clips (or fewer if the video is short).
- Clips must NOT overlap.
- Clips must match the requested category: {category}.
- Return ONLY a valid JSON array — no markdown, no commentary, no code fences.
- Each element must have exactly these keys:
    "start"   : float  (seconds from video start)
    "end"     : float  (seconds from video start)
    "title"   : string (catchy 5–8 word title)
    "reason"  : string (1-sentence engagement hook explanation)
    "score"   : int    (1–10 virality score)
"""

_SCENE_USER = """\
Video title: {title}
Category requested: {category}
Total duration: {duration:.1f} seconds

Full transcript with timestamps:
{transcript_json}

Identify the top {max_clips} clips that best fit "{category}".
Return only a JSON array of clip objects.
"""


def _build_transcript_json(segments: list, max_chars: int = 12000) -> str:
    """Convert segment list to a compact JSON string, truncating if needed."""
    compact = [
        {"s": round(seg["start"], 1), "e": round(seg["end"], 1), "t": seg["text"].strip()}
        for seg in segments
        if seg.get("text", "").strip()
    ]
    raw = json.dumps(compact, ensure_ascii=False)
    if len(raw) > max_chars:
        # Keep first + last portion
        half   = max_chars // 2 - 50
        raw    = raw[:half] + " ... [truncated] ... " + raw[-half:]
    return raw


def detect_scenes(
    transcript: dict,
    title: str,
    category: str,
    api_key: str,
    progress_cb=None,
) -> list[dict]:
    """
    Call Groq Llama to detect the top scenes.
    Returns list of scene dicts: {start, end, title, reason, score}.
    """
    from groq import Groq

    client     = Groq(api_key=api_key)
    duration   = transcript.get("duration", 0)
    segments   = transcript.get("segments", [])
    trans_json = _build_transcript_json(segments)

    system_msg = _SCENE_SYSTEM.format(
        min_sec   = MIN_CLIP_SEC,
        max_sec   = MAX_CLIP_SEC,
        max_clips = MAX_CLIPS,
        category  = category,
    )
    user_msg = _SCENE_USER.format(
        title         = title,
        category      = category,
        duration      = duration,
        transcript_json = trans_json,
        max_clips     = MAX_CLIPS,
    )

    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    last_error    = None

    for model in models_to_try:
        if progress_cb:
            progress_cb(f"Analyzing scenes with {model} …")
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model    = model,
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature = 0.3,
                    max_tokens  = 2048,
                )
                raw_content = resp.choices[0].message.content or ""
                scenes      = _parse_scenes_json(raw_content, duration)
                if scenes:
                    return scenes
            except Exception as exc:
                last_error = exc
                log.warning("Scene detection attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

    # Hard fallback: generate evenly-spaced clips
    log.error("All scene detection attempts failed (%s) — using fallback clips", last_error)
    return _fallback_scenes(duration)


def _parse_scenes_json(raw: str, max_duration: float) -> list[dict]:
    """Extract and validate the JSON array from LLM response."""
    raw = raw.strip()

    # Strip markdown fences if present
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    if raw.endswith("```"):
        raw = raw[:-3]

    # Find the JSON array boundaries
    start_idx = raw.find("[")
    end_idx   = raw.rfind("]")
    if start_idx == -1 or end_idx == -1:
        log.warning("No JSON array found in LLM response: %s", raw[:200])
        return []

    try:
        scenes = json.loads(raw[start_idx : end_idx + 1])
    except json.JSONDecodeError as exc:
        log.warning("JSON parse error: %s — raw: %s", exc, raw[:200])
        return []

    validated = []
    for s in scenes:
        if not isinstance(s, dict):
            continue
        try:
            start = float(s.get("start", 0))
            end   = float(s.get("end",   0))
        except (TypeError, ValueError):
            continue

        duration = end - start
        if duration < MIN_CLIP_SEC:
            end = start + MIN_CLIP_SEC
        if duration > MAX_CLIP_SEC:
            end = start + MAX_CLIP_SEC
        if end > max_duration and max_duration > 0:
            end   = max_duration
            start = max(0, end - MAX_CLIP_SEC)

        validated.append({
            "start":  round(max(0.0, start), 2),
            "end":    round(end, 2),
            "title":  str(s.get("title",  f"Clip {len(validated)+1}")),
            "reason": str(s.get("reason", "High-engagement moment")),
            "score":  max(1, min(10, int(s.get("score", 7)))),
        })

    # Deduplicate / remove overlaps
    validated.sort(key=lambda x: x["start"])
    deduped = []
    for sc in validated:
        if deduped and sc["start"] < deduped[-1]["end"] - 1.0:
            continue
        deduped.append(sc)

    return deduped[:MAX_CLIPS]


def _fallback_scenes(duration: float) -> list[dict]:
    """Generate evenly distributed 30-second fallback clips."""
    clips     = []
    clip_dur  = 30.0
    n_clips   = min(MAX_CLIPS, max(1, int(duration // clip_dur)))
    step      = duration / n_clips if n_clips else duration

    for i in range(n_clips):
        start = i * step
        end   = min(start + clip_dur, duration)
        clips.append({
            "start":  round(start, 2),
            "end":    round(end,   2),
            "title":  f"Scene {i + 1}",
            "reason": "Auto-generated clip (scene detection fallback).",
            "score":  5,
        })
    return clips


# ─────────────────────────────────────────────
# SYNTHETIC SFX GENERATOR
# ─────────────────────────────────────────────
def generate_sfx(sfx_type: str) -> Optional[str]:
    """
    Generate a simple synthesized sound-effect WAV using ffmpeg.
    Returns path to the WAV file or None on failure.
    """
    profile   = SFX_PROFILES.get(sfx_type, SFX_PROFILES["whoosh"])
    out_path  = SFX_DIR / f"{sfx_type}.wav"

    if out_path.exists():
        return str(out_path)

    try:
        freq_start = profile["freq"]
        freq_end   = profile["sweep_end"]
        dur        = profile["duration"]

        # sine sweep: freq=start:end
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq_start}:end_frequency={freq_end}:duration={dur}",
            "-af", "afade=t=out:st=0:d=" + str(dur),
            str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=15)
        return str(out_path)
    except Exception as exc:
        log.warning("SFX generation failed for %s: %s", sfx_type, exc)
        return None


# ─────────────────────────────────────────────
# VIDEO RENDERING — CROP + CAPTION + SFX
# ─────────────────────────────────────────────
def seconds_to_ass_time(secs: float) -> str:
    """Convert float seconds to ASS subtitle timestamp H:MM:SS.cc"""
    h   = int(secs // 3600)
    m   = int((secs % 3600) // 60)
    s   = int(secs % 60)
    cs  = int((secs - int(secs)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass_subtitles(
    segments:   list,
    clip_start: float,
    clip_end:   float,
    preset:     dict,
) -> str:
    """
    Generate an ASS subtitle file content string for the given clip window.
    """
    fontname    = preset.get("font",         "Arial")
    fontsize    = preset.get("fontsize",     72)
    color_hex   = preset.get("color",        "white")
    stroke_hex  = preset.get("stroke_color", "black")
    stroke_w    = preset.get("stroke_width", 3)

    def _hex_to_ass(c: str) -> str:
        """Convert #RRGGBB or named color to ASS &HAABBGGRR format."""
        named = {
            "white":   "FFFFFF", "black":   "000000",
            "yellow":  "00FFFF", "red":     "0000FF",
            "#ff1a1a": "0000FF", "#00ffff": "FFFF00",
            "#ff00ff": "FF00FF",
        }
        c = c.strip()
        if c in named:
            hex6 = named[c]
        elif c.startswith("#"):
            hex6 = c[1:].upper().zfill(6)
        else:
            hex6 = "FFFFFF"
        r, g, b = hex6[0:2], hex6[2:4], hex6[4:6]
        return f"&H00{b}{g}{r}"

    primary  = _hex_to_ass(str(color_hex))
    outline  = _hex_to_ass(str(stroke_hex))

    ass_header = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {TARGET_W}
PlayResY: {TARGET_H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{primary},&H000000FF,{outline},&H80000000,-1,0,0,0,100,100,0,0,1,{stroke_w},0,2,20,20,{int(TARGET_H * 0.12)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end   = float(seg.get("end",   0))

        # Intersect with clip window
        rel_start = seg_start - clip_start
        rel_end   = seg_end   - clip_start
        clip_dur  = clip_end  - clip_start

        if rel_end <= 0 or rel_start >= clip_dur:
            continue

        rel_start = max(0.0, rel_start)
        rel_end   = min(clip_dur, rel_end)

        text = seg.get("text", "").strip().upper()
        if not text:
            continue

        # Wrap long lines at ~30 chars
        words    = text.split()
        current  = ""
        wrapped  = []
        for w in words:
            if len(current) + len(w) + 1 > 30 and current:
                wrapped.append(current.strip())
                current = w
            else:
                current += (" " if current else "") + w
        if current:
            wrapped.append(current.strip())

        ass_text = r"\N".join(wrapped)
        lines.append(
            f"Dialogue: 0,{seconds_to_ass_time(rel_start)},{seconds_to_ass_time(rel_end)},"
            f"Default,,0,0,0,,{ass_text}"
        )

    return ass_header + "\n".join(lines)


def render_short(
    video_path:    str,
    scene:         dict,
    caption_preset_name: str,
    add_sfx:       bool,
    clip_index:    int,
    progress_cb    = None,
    segments:      list = None,
) -> str:
    """
    Render a single 9:16 Short clip with captions and optional SFX.
    Returns the path to the final MP4.
    """
    start      = scene["start"]
    end        = scene["end"]
    duration   = end - start
    preset     = CAPTION_PRESETS.get(caption_preset_name, list(CAPTION_PRESETS.values())[0])
    uid        = hashlib.md5(f"{video_path}{start}{end}".encode()).hexdigest()[:8]
    out_path   = FINAL_DIR / f"short_{clip_index:02d}_{uid}.mp4"

    if out_path.exists():
        return str(out_path)

    if progress_cb:
        progress_cb(f"Rendering clip {clip_index+1}: {scene['title'][:40]} …")

    # ── Step 1: Extract raw clip ──
    raw_clip = CLIPS_DIR / f"raw_{uid}.mp4"
    cmd_extract = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(raw_clip),
    ]
    _run_ffmpeg(cmd_extract, "clip extraction")

    # ── Step 2: Probe source dimensions ──
    probe_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "json", str(raw_clip)],
        capture_output=True, text=True, timeout=20,
    )
    try:
        vinfo     = json.loads(probe_result.stdout)
        src_w     = int(vinfo["streams"][0]["width"])
        src_h     = int(vinfo["streams"][0]["height"])
    except (KeyError, IndexError, json.JSONDecodeError):
        src_w, src_h = 1920, 1080

    # ── Step 3: Crop to 9:16 (center crop) ──
    target_ratio = TARGET_W / TARGET_H        # 0.5625
    src_ratio    = src_w / src_h

    if src_ratio > target_ratio:
        # wider than 9:16 → crop width
        crop_h = src_h
        crop_w = int(src_h * target_ratio)
    else:
        # taller or equal → crop height
        crop_w = src_w
        crop_h = int(src_w / target_ratio)

    crop_x = (src_w - crop_w) // 2
    crop_y = (src_h - crop_h) // 2

    cropped_clip = CLIPS_DIR / f"cropped_{uid}.mp4"
    cmd_crop = [
        "ffmpeg", "-y", "-i", str(raw_clip),
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={TARGET_W}:{TARGET_H}",
        "-c:v", "libx264", "-c:a", "aac",
        str(cropped_clip),
    ]
    _run_ffmpeg(cmd_crop, "crop/scale")

    # ── Step 4: Build ASS subtitles ──
    ass_path = CLIPS_DIR / f"subs_{uid}.ass"
    segs_for_clip = segments or []
    ass_content   = build_ass_subtitles(segs_for_clip, start, end, preset)
    ass_path.write_text(ass_content, encoding="utf-8")

    # ── Step 5: Burn subtitles ──
    captioned_clip = CLIPS_DIR / f"captioned_{uid}.mp4"
    # Escape path for ffmpeg filter
    ass_escaped    = str(ass_path).replace("\\", "/").replace(":", r"\:")
    cmd_subs = [
        "ffmpeg", "-y", "-i", str(cropped_clip),
        "-vf", f"ass='{ass_escaped}'",
        "-c:v", "libx264", "-c:a", "copy",
        str(captioned_clip),
    ]
    try:
        _run_ffmpeg(cmd_subs, "subtitle burn")
        working_clip = str(captioned_clip)
    except RuntimeError as exc:
        log.warning("Subtitle burn failed (%s) — using clip without captions", exc)
        working_clip = str(cropped_clip)

    # ── Step 6: Add SFX (optional) ──
    if add_sfx:
        sfx_intro = generate_sfx("whoosh")
        sfx_outro = generate_sfx("impact")

        if sfx_intro and sfx_outro:
            mixed_clip = CLIPS_DIR / f"mixed_{uid}.mp4"
            try:
                # Overlay SFX at clip start and end
                sfx_intro_escaped = str(sfx_intro).replace("\\", "/")
                sfx_outro_escaped = str(sfx_outro).replace("\\", "/")
                offset_outro      = max(0.0, duration - 0.5)

                cmd_sfx = [
                    "ffmpeg", "-y",
                    "-i", working_clip,
                    "-i", sfx_intro_escaped,
                    "-i", sfx_outro_escaped,
                    "-filter_complex",
                    (
                        f"[1:a]adelay=0|0[sfx1];"
                        f"[2:a]adelay={int(offset_outro*1000)}|{int(offset_outro*1000)}[sfx2];"
                        f"[0:a][sfx1][sfx2]amix=inputs=3:duration=first:normalize=0[aout]"
                    ),
                    "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac",
                    str(mixed_clip),
                ]
                _run_ffmpeg(cmd_sfx, "SFX overlay")
                working_clip = str(mixed_clip)
            except RuntimeError as exc:
                log.warning("SFX overlay failed (%s) — skipping SFX", exc)

    # ── Step 7: Final output ──
    shutil.copy2(working_clip, str(out_path))

    # Cleanup intermediates
    for p in (raw_clip, cropped_clip, captioned_clip):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    return str(out_path)


def _run_ffmpeg(cmd: list, step_name: str):
    """Run an ffmpeg command and raise RuntimeError on failure."""
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        err = result.stderr.decode(errors="ignore")[-1000:]
        raise RuntimeError(f"FFmpeg {step_name} failed (rc={result.returncode}): {err}")


# ─────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "step":           0,
        "video_info":     None,
        "transcript":     None,
        "scenes":         [],
        "rendered_clips": {},
        "last_url":       "",
        "processing":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset():
    keys = [
        "step", "video_info", "transcript",
        "scenes", "rendered_clips", "last_url", "processing",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    _init_state()


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def render_hero():
    st.markdown(
        """
        <div class="sc-hero">
            <h1>🎬 ShortsCraft AI</h1>
            <p>Transform any YouTube video into viral 9:16 Shorts — powered by Groq AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_badge(label: str):
    st.markdown(f'<div class="sc-step">{label}</div>', unsafe_allow_html=True)


def render_clip_card(idx: int, scene: dict):
    duration = scene["end"] - scene["start"]
    score    = scene.get("score", 7)
    stars    = "⭐" * min(5, max(1, score // 2))
    st.markdown(
        f"""
        <div class="clip-card">
            <div class="clip-title">#{idx+1} — {scene['title']}</div>
            <div class="clip-meta">
                ⏱ {scene['start']:.1f}s – {scene['end']:.1f}s &nbsp;|&nbsp;
                {duration:.1f}s long &nbsp;|&nbsp;
                Virality {stars} ({score}/10)
            </div>
            <div style="color:#94a3b8;font-size:.85rem;margin-top:.4rem">
                {scene.get('reason', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    _init_state()
    render_hero()

    # ── API Key check ──
    api_key = get_groq_api_key()
    if not api_key:
        st.error(
            "🔑 **GROQ_API_KEY not found.**\n\n"
            "Add your key to Streamlit Secrets:\n"
            "1. Go to your Streamlit Cloud app → **Settings → Secrets**\n"
            "2. Add:  `GROQ_API_KEY = \"gsk_your_key_here\"`\n"
            "3. For local dev, create `.streamlit/secrets.toml` with the same line.\n\n"
            "Get a free key at [console.groq.com](https://console.groq.com)."
        )
        return

    # ── Dependency check ──
    if not check_ffmpeg():
        st.error(
            "⚠️ **FFmpeg not found.**  "
            "Install it: `sudo apt install ffmpeg` (Ubuntu) "
            "or `brew install ffmpeg` (macOS)."
        )
        return

    if not check_yt_dlp():
        st.error("⚠️ **yt-dlp not installed.** Run `pip install yt-dlp`.")
        return

    # ═══════════════════════════════════════════
    # SIDEBAR — Settings
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        scene_category = st.selectbox(
            "Scene Category",
            SCENE_CATEGORIES,
            help="AI will detect clips that best match this category.",
        )

        caption_style = st.selectbox(
            "Caption Style",
            list(CAPTION_PRESETS.keys()),
            help="Caption preset burned onto every Short.",
        )

        add_sfx = st.toggle("Add Sound Effects", value=True)

        st.divider()
        st.markdown("### 🎬 About ShortsCraft AI")
        st.caption(
            "Uses **Groq Whisper** for transcription and "
            "**Llama 3.3 70B** for intelligent scene detection. "
            "All free-tier Groq models."
        )

        if st.button("🔄 Reset / New Video", use_container_width=True):
            _reset()
            st.rerun()

    # ═══════════════════════════════════════════
    # STEP 0 — URL Input
    # ═══════════════════════════════════════════
    col_main, col_info = st.columns([2, 1])

    with col_main:
        render_step_badge("STEP 1 — Input")
        url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            value=st.session_state.get("last_url", ""),
            help="Paste any public YouTube URL.",
        )

        process_btn = st.button(
            "🚀 Analyze & Generate Shorts",
            disabled=st.session_state.processing,
            use_container_width=True,
        )

    with col_info:
        st.markdown(
            """
            <div class="sc-card">
                <b style="color:#818cf8">How it works</b><br><br>
                <span class="sc-progress">
                1️⃣ Download video<br>
                2️⃣ Transcribe with Whisper<br>
                3️⃣ AI detects best scenes<br>
                4️⃣ Crop → 9:16 vertical<br>
                5️⃣ Burn captions + SFX<br>
                6️⃣ Download your Shorts
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ═══════════════════════════════════════════
    # PROCESSING PIPELINE
    # ═══════════════════════════════════════════
    if process_btn and url.strip():
        if not (url.startswith("http://") or url.startswith("https://")):
            st.warning("Please enter a valid URL starting with http:// or https://")
            return

        st.session_state.processing = True
        st.session_state.last_url   = url.strip()
        st.session_state.step       = 1

        status_box = st.empty()
        progress   = st.progress(0)

        def update_status(msg: str, pct: float = None):
            status_box.info(f"⏳ {msg}")
            if pct is not None:
                progress.progress(min(1.0, max(0.0, pct)))
            log.info(msg)

        try:
            # ── Download ──
            update_status("Downloading video from YouTube …", 0.05)
            video_info = download_video(
                url.strip(),
                progress_cb=lambda m: update_status(m, 0.1),
            )
            st.session_state.video_info = video_info
            update_status(
                f"Downloaded: **{video_info['title']}** "
                f"({video_info['duration_sec']:.0f}s)",
                0.25,
            )

            # ── Transcribe ──
            update_status("Transcribing audio with Whisper …", 0.30)
            transcript = transcribe_audio(
                video_info["audio_path"],
                api_key,
                progress_cb=lambda m: update_status(m, 0.40),
            )
            st.session_state.transcript = transcript
            n_segs = len(transcript.get("segments", []))
            update_status(
                f"Transcription complete — {n_segs} segments, "
                f"{len(transcript['text'].split())} words.",
                0.50,
            )

            # ── Scene detection ──
            update_status(
                f"Detecting top scenes for '{scene_category}' …", 0.55
            )
            scenes = detect_scenes(
                transcript,
                video_info["title"],
                scene_category,
                api_key,
                progress_cb=lambda m: update_status(m, 0.60),
            )
            st.session_state.scenes = scenes
            update_status(
                f"Found {len(scenes)} scenes. Starting rendering …", 0.65
            )

            # ── Render each clip ──
            rendered = {}
            n = len(scenes)
            for i, scene in enumerate(scenes):
                pct = 0.65 + (i / max(1, n)) * 0.33
                update_status(
                    f"Rendering clip {i+1}/{n}: {scene['title'][:40]} …", pct
                )
                try:
                    out = render_short(
                        video_path          = video_info["video_path"],
                        scene               = scene,
                        caption_preset_name = caption_style,
                        add_sfx             = add_sfx,
                        clip_index          = i,
                        progress_cb         = lambda m: update_status(m, pct),
                        segments            = transcript.get("segments", []),
                    )
                    rendered[i] = out
                except Exception as exc:
                    log.error("Clip %d render failed: %s\n%s", i, exc, traceback.format_exc())
                    rendered[i] = None

            st.session_state.rendered_clips = rendered
            st.session_state.step           = 2
            st.session_state.processing     = False

            progress.progress(1.0)
            status_box.success(
                f"✅ Done! {sum(1 for v in rendered.values() if v)} / {n} clips rendered."
            )

        except Exception as exc:
            st.session_state.processing = False
            st.error(f"❌ Processing failed: {exc}")
            log.error("Pipeline error: %s\n%s", exc, traceback.format_exc())
            return

    # ═══════════════════════════════════════════
    # STEP 2 — Results & Downloads
    # ═══════════════════════════════════════════
    if st.session_state.step >= 2 and st.session_state.scenes:
        st.divider()
        render_step_badge("STEP 2 — Your Shorts")

        video_info = st.session_state.video_info or {}
        st.markdown(
            f"**Source:** {video_info.get('title','Unknown')}  "
            f"— {video_info.get('duration_sec', 0):.0f}s"
        )

        scenes   = st.session_state.scenes
        rendered = st.session_state.rendered_clips

        # Sort by virality score descending
        sorted_scenes = sorted(
            enumerate(scenes), key=lambda x: x[1].get("score", 0), reverse=True
        )

        for rank, (orig_idx, scene) in enumerate(sorted_scenes):
            render_clip_card(rank, scene)
            clip_path = rendered.get(orig_idx)

            col_dl, col_preview = st.columns([1, 2])

            with col_dl:
                if clip_path and Path(clip_path).exists():
                    with open(clip_path, "rb") as fh:
                        st.download_button(
                            label    = f"⬇️ Download Short #{rank+1}",
                            data     = fh,
                            file_name= f"short_{rank+1:02d}_{scene['title'][:30].replace(' ','_')}.mp4",
                            mime     = "video/mp4",
                            key      = f"dl_{orig_idx}",
                            use_container_width=True,
                        )
                else:
                    st.warning(f"Clip #{rank+1} failed to render.")

            with col_preview:
                if clip_path and Path(clip_path).exists():
                    st.video(clip_path)

            st.markdown("")  # spacer

        # ── Batch download note ──
        st.markdown(
            """
            <div class="sc-card">
            💡 <b>Tip:</b> Download all clips, then upload directly to YouTube Shorts,
            Instagram Reels, or TikTok. Each clip is production-ready 1080×1920 at 30fps.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
