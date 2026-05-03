import streamlit as st
from openai import OpenAI
from groq import Groq
from pydub import AudioSegment
import json
import tempfile
import os
import re
import math
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motion Forge Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide sidebar, Streamlit deploy button, and footer
st.markdown("""
<style>
  [data-testid="stSidebar"]        { display: none; }
  [data-testid="collapsedControl"] { display: none; }
  [data-testid="stToolbar"]        { display: none; }
  footer                           { display: none; }
  .hero { text-align: center; padding: 2rem 0 0.75rem; }
  .hero h1 { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem; }
  .hero p  { color: #999; font-size: 1.05rem; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎬 Motion Forge Script Generator</h1>
</div>
""", unsafe_allow_html=True)

# ── Load API keys from Streamlit secrets ──────────────────────────────────────
try:
    groq_key       = st.secrets["GROQ_API_KEY"]
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# ── Model config ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "anthropic/claude-sonnet-4-5"   # via OpenRouter

# ── Helpers ───────────────────────────────────────────────────────────────────

GROQ_CHUNK_LIMIT_MB = 23   # Stay under Groq's 25 MB limit per request

def shift_srt_timestamps(srt_text: str, offset_ms: int) -> str:
    """Shift all timestamps in an SRT string by offset_ms milliseconds."""
    def shift(t: str) -> str:
        h, m, rest = t.split(":")
        s, ms = rest.split(",")
        total = (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms) + offset_ms
        total = max(0, total)
        h2, rem = divmod(total, 3_600_000)
        m2, rem = divmod(rem, 60_000)
        s2, ms2 = divmod(rem, 1_000)
        return f"{h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}"
    return re.sub(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
        lambda m: f"{shift(m.group(1))} --> {shift(m.group(2))}",
        srt_text,
    )


def renumber_srt(srt_text: str, start: int) -> tuple[str, int]:
    """Renumber SRT blocks starting from `start`. Returns (new_srt, next_index)."""
    blocks = [b.strip() for b in srt_text.strip().split("\n\n") if b.strip()]
    out = []
    for i, block in enumerate(blocks):
        lines = block.splitlines()
        lines[0] = str(start + i)
        out.append("\n".join(lines))
    return "\n\n".join(out), start + len(blocks)


def transcribe_audio(groq_client: Groq, file_bytes: bytes, ext: str, language: str) -> str:
    """Transcribe audio bytes, chunking automatically if > GROQ_CHUNK_LIMIT_MB."""
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb <= GROQ_CHUNK_LIMIT_MB:
        # Single request
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=af,
                response_format="srt",
                language=language,
            )
        os.unlink(tmp_path)
        return result if isinstance(result, str) else result.text

    # Large file — split into chunks
    audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
    total_ms   = len(audio)
    # Target chunk duration based on size ratio
    chunk_ms   = int(total_ms * (GROQ_CHUNK_LIMIT_MB / size_mb) * 0.9)
    chunk_ms   = min(chunk_ms, 10 * 60 * 1000)   # cap at 10 minutes

    srt_parts      = []
    offset_ms      = 0
    next_idx       = 1

    n_chunks = math.ceil(total_ms / chunk_ms)
    for i in range(n_chunks):
        chunk   = audio[i * chunk_ms : (i + 1) * chunk_ms]
        buf     = io.BytesIO()
        chunk.export(buf, format="mp3")
        buf.seek(0)
        chunk_bytes = buf.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(chunk_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=af,
                response_format="srt",
                language=language,
            )
        os.unlink(tmp_path)

        chunk_srt = result if isinstance(result, str) else result.text
        if offset_ms > 0:
            chunk_srt = shift_srt_timestamps(chunk_srt, offset_ms)
        chunk_srt, next_idx = renumber_srt(chunk_srt, next_idx)
        srt_parts.append(chunk_srt)
        offset_ms += chunk_ms   # accumulate offset for next chunk

    return "\n\n".join(srt_parts)


def parse_srt(srt_text: str) -> list[dict]:
    segments = []
    for block in srt_text.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            segments.append({
                "index":     lines[0].strip(),
                "timestamp": lines[1].strip(),
                "text":      " ".join(lines[2:]).strip(),
            })
    return segments


def enrich_srt_in_batches(
    or_client: OpenAI,
    segments: list[dict],
    progress_bar,
    batch_size: int = 25,
) -> str:
    annotated_parts = []
    total_batches = math.ceil(len(segments) / batch_size)

    for batch_num, start in enumerate(range(0, len(segments), batch_size)):
        batch = segments[start : start + batch_size]
        raw_srt = "\n\n".join(
            f"{s['index']}\n{s['timestamp']}\n{s['text']}" for s in batch
        )

        prompt = f"""You are a cinematic AI script supervisor for an episodic audio drama.

Add a [MORPHIC] line directly after each subtitle line in this SRT batch.

Each [MORPHIC] prompt must be 1–2 sentences and include:
- Visual description of what is happening (characters, objects, setting)
- Camera movement — one of: slow dolly in, slow dolly out, wide crane shot, handheld follow, static close-up, rack focus, low angle push in, aerial drone pull back, whip pan, slow pan left/right
- Lighting style — e.g. golden hour, harsh shadows, soft diffused, moonlit blue, candlelight, neon, overcast grey
- Mood keywords — e.g. melancholic, tense, joyful, mysterious, epic, intimate, ominous, hopeful
- If a character is mentioned, describe their appearance and action

Return ONLY the annotated SRT text. No JSON, no headers, no extra explanation.

SRT:
{raw_srt}"""

        resp = or_client.chat.completions.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        annotated_parts.append(resp.choices[0].message.content.strip())
        progress_bar.progress((batch_num + 1) / total_batches)

    return "\n\n".join(annotated_parts)


def extract_characters_and_locations(or_client: OpenAI, srt_text: str) -> dict:
    excerpt = srt_text[:12_000]

    prompt = f"""You are a story analyst and art director. Analyse this SRT transcript from an audio drama episode.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:

{{
  "characters": [
    {{
      "name": "Character Name",
      "role": "protagonist | antagonist | supporting | narrator",
      "physical_description": "Height, build, hair, eye colour, skin tone, distinguishing features",
      "costume": "Full clothing — fabric, colour, accessories, footwear, headwear",
      "personality_notes": "2–3 adjectives capturing their personality",
      "visual_style_prompt": "Ready-to-paste 1–2 sentence Morphic image prompt for this character"
    }}
  ],
  "locations": [
    {{
      "name": "Location Name",
      "description": "What this place looks like in detail",
      "time_of_day": "morning | afternoon | evening | night | varies",
      "atmosphere": "Overall mood or feel",
      "lighting": "Typical lighting conditions",
      "visual_style_prompt": "Ready-to-paste 1–2 sentence Morphic image prompt for this location"
    }}
  ]
}}

TRANSCRIPT:
{excerpt}"""

    resp = or_client.chat.completions.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def format_character_sheet(characters: list[dict], episode_name: str) -> str:
    lines = ["=" * 60, "CHARACTER & COSTUME SHEET"]
    if episode_name:
        lines.append(f"Episode: {episode_name}")
    lines += ["=" * 60, ""]
    for c in characters:
        lines += [
            f"CHARACTER: {c.get('name', 'Unknown').upper()}",
            "─" * 40,
            f"Role          : {c.get('role', '')}",
            f"Physical      : {c.get('physical_description', '')}",
            f"Costume       : {c.get('costume', '')}",
            f"Personality   : {c.get('personality_notes', '')}",
            "",
            "[MORPHIC PROMPT FRAGMENT]",
            c.get("visual_style_prompt", ""),
            "", "",
        ]
    return "\n".join(lines)


def format_locations_sheet(locations: list[dict], episode_name: str) -> str:
    lines = ["=" * 60, "LOCATIONS SHEET"]
    if episode_name:
        lines.append(f"Episode: {episode_name}")
    lines += ["=" * 60, ""]
    for loc in locations:
        lines += [
            f"LOCATION: {loc.get('name', 'Unknown').upper()}",
            "─" * 40,
            f"Description   : {loc.get('description', '')}",
            f"Time of Day   : {loc.get('time_of_day', '')}",
            f"Atmosphere    : {loc.get('atmosphere', '')}",
            f"Lighting      : {loc.get('lighting', '')}",
            "",
            "[MORPHIC PROMPT FRAGMENT]",
            loc.get("visual_style_prompt", ""),
            "", "",
        ]
    return "\n".join(lines)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("---")
col_upload, col_meta = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "📁 Upload Episode Audio",
        type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
        help="Up to 300 MB. Large files are automatically split and processed in chunks.",
    )
    if uploaded_file:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > 300:
            st.warning(f"⚠️ File is {size_mb:.1f} MB — maximum is 300 MB.")
        elif size_mb > GROQ_CHUNK_LIMIT_MB:
            st.info(f"📦 {uploaded_file.name} ({size_mb:.1f} MB) — will be processed in chunks automatically.")
        else:
            st.success(f"✅ {uploaded_file.name}  ({size_mb:.1f} MB) — ready")

with col_meta:
    episode_name = st.text_input(
        "Episode name (used in filenames)",
        placeholder="Episode 01 – The Beginning",
    )
    language = st.selectbox(
        "Audio language",
        options=["en", "hi", "ta", "te", "es", "fr", "de", "ja", "ko", "zh"],
        format_func=lambda x: {
            "en": "English", "hi": "Hindi", "ta": "Tamil",  "te": "Telugu",
            "es": "Spanish", "fr": "French", "de": "German",
            "ja": "Japanese","ko": "Korean","zh": "Chinese",
        }[x],
    )

    st.markdown("")  # spacer

    file_ok = bool(uploaded_file and (uploaded_file.size / (1024 * 1024)) <= 300)

    generate_clicked = False
    if file_ok:
        generate_clicked = st.button(
            "🚀 Generate Morphic Script", type="primary", use_container_width=True
        )
    else:
        st.button("🚀 Generate Morphic Script", disabled=True, use_container_width=True)
        if not uploaded_file:
            st.caption("Upload an audio file to get started")

# ── Processing ────────────────────────────────────────────────────────────────

if file_ok and generate_clicked:

    safe_name = re.sub(r"[^\w\-]", "_", episode_name) if episode_name else "episode"

    st.markdown("---")
    st.markdown("#### ⏳ Processing")

    # STEP 1 — Transcribe with Groq Whisper
    step1 = st.status("🎙️ Step 1 — Transcribing audio with Whisper...", expanded=False)
    srt_content = ""
    with step1:
        try:
            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            file_bytes = uploaded_file.read()
            size_mb = len(file_bytes) / (1024 * 1024)
            groq_client = Groq(api_key=groq_key)

            if size_mb > GROQ_CHUNK_LIMIT_MB:
                n_chunks = math.ceil(size_mb / GROQ_CHUNK_LIMIT_MB)
                st.write(f"Large file ({size_mb:.1f} MB) — splitting into ~{n_chunks} chunks…")

            srt_content = transcribe_audio(groq_client, file_bytes, ext, language)
            step1.update(label="✅ Step 1 — Transcription complete!", state="complete")
        except Exception as e:
            step1.update(label="❌ Step 1 — Transcription failed", state="error")
            st.error(f"Whisper error: {e}")
            st.stop()

    # STEP 2 — Enrich with Claude via OpenRouter
    step2 = st.status("🤖 Step 2 — Generating Morphic prompts with Claude...", expanded=True)
    annotated_srt = ""
    with step2:
        try:
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            segments = parse_srt(srt_content)
            total = len(segments)
            st.write(f"Found **{total} subtitle segments** — enriching in batches…")
            bar = st.progress(0)
            annotated_srt = enrich_srt_in_batches(or_client, segments, bar)
            step2.update(
                label=f"✅ Step 2 — Morphic prompts added to {total} segments!",
                state="complete",
            )
        except Exception as e:
            step2.update(label="❌ Step 2 — Enrichment failed", state="error")
            st.error(f"Claude/OpenRouter error: {e}")
            st.stop()

    # STEP 3 — Extract characters & locations
    step3 = st.status("🎭 Step 3 — Extracting characters, costumes & locations...", expanded=False)
    char_sheet = loc_sheet = ""
    with step3:
        try:
            data = extract_characters_and_locations(or_client, srt_content)
            char_sheet = format_character_sheet(data.get("characters", []), episode_name)
            loc_sheet  = format_locations_sheet(data.get("locations", []),  episode_name)
            n_chars = len(data.get("characters", []))
            n_locs  = len(data.get("locations",  []))
            step3.update(
                label=f"✅ Step 3 — Found {n_chars} character(s) and {n_locs} location(s)!",
                state="complete",
            )
        except json.JSONDecodeError:
            step3.update(label="⚠️ Step 3 — Could not parse JSON", state="error")
            st.warning("Character & location extraction returned unexpected output. Annotated SRT is still available.")
        except Exception as e:
            step3.update(label="❌ Step 3 — Extraction failed", state="error")
            st.error(f"Error: {e}")

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Your Files")

    dcol1, dcol2, dcol3 = st.columns(3)

    with dcol1:
        st.download_button(
            "📄 Annotated SRT",
            data=annotated_srt,
            file_name=f"{safe_name}_morphic.srt",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption("Time-coded script with [MORPHIC] camera prompts per line")

    with dcol2:
        if char_sheet:
            st.download_button(
                "👤 Character & Costume Sheet",
                data=char_sheet,
                file_name=f"{safe_name}_characters.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.caption("Characters with costume & Morphic prompt fragments")
        else:
            st.button("👤 Character Sheet", disabled=True, use_container_width=True)
            st.caption("Not available")

    with dcol3:
        if loc_sheet:
            st.download_button(
                "📍 Locations Sheet",
                data=loc_sheet,
                file_name=f"{safe_name}_locations.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.caption("Locations with atmosphere & Morphic prompt fragments")
        else:
            st.button("📍 Locations Sheet", disabled=True, use_container_width=True)
            st.caption("Not available")

    # ── Preview ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 👁️ Preview")

    tab_srt, tab_chars, tab_locs = st.tabs(["Annotated SRT", "Characters & Costumes", "Locations"])

    with tab_srt:
        preview = annotated_srt[:4000]
        if len(annotated_srt) > 4000:
            preview += "\n\n… (truncated — download for full file)"
        st.text_area("", value=preview, height=420, label_visibility="collapsed")

    with tab_chars:
        st.text_area("", value=char_sheet or "No data", height=420, label_visibility="collapsed")

    with tab_locs:
        st.text_area("", value=loc_sheet or "No data", height=420, label_visibility="collapsed")
