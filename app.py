import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
import json
import tempfile
import os
import re
import math
import io
import subprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motion Forge Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

# ── Load API keys ──────────────────────────────────────────────────────────────
try:
    openai_key     = st.secrets["OPENAI_API_KEY"]
    openrouter_key = st.secrets["OPENROUTER_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

CLAUDE_MODEL       = "anthropic/claude-sonnet-4-5"
CHUNK_SECS         = 300   # 5-minute chunks → each finishes in ~30-60 sec


# ── Helpers ───────────────────────────────────────────────────────────────────

def seconds_to_srt_time(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    s  = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def verbose_json_to_srt(result, index_start: int = 1, offset_sec: float = 0.0):
    lines = []
    segs  = getattr(result, "segments", None) or []
    for i, seg in enumerate(segs, start=index_start):
        start = seconds_to_srt_time(seg.start + offset_sec)
        end   = seconds_to_srt_time(seg.end   + offset_sec)
        text  = seg.text.strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(lines), index_start + len(segs)


def transcribe_audio(openai_client: OpenAI, file_bytes: bytes, ext: str,
                     language: str, progress_cb=None) -> str:
    """Split audio into 5-min chunks via ffmpeg (disk-only, no Python RAM),
    transcribe each with OpenAI Whisper, show progress via progress_cb."""

    def call_whisper(path: str, offset_sec: float = 0.0, idx_start: int = 1):
        with open(path, "rb") as af:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1", file=af,
                response_format="verbose_json", language=language,
            )
        return verbose_json_to_srt(result, index_start=idx_start, offset_sec=offset_sec)

    # Write source to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_in:
        tmp_in.write(file_bytes)
        src_path = tmp_in.name

    # Get duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", src_path],
        capture_output=True, text=True,
    )
    try:
        total_secs = float(probe.stdout.strip())
    except Exception:
        total_secs = 7200  # fallback: 2 hours

    n_chunks  = max(1, math.ceil(total_secs / CHUNK_SECS))
    srt_parts = []
    next_idx  = 1

    for i in range(n_chunks):
        start_sec  = i * CHUNK_SECS
        chunk_path = src_path + f"_chunk{i}.mp3"

        # Extract + compress chunk with ffmpeg
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", src_path,
             "-ss", str(start_sec), "-t", str(CHUNK_SECS),
             "-ac", "1", "-ab", "64k", "-f", "mp3", chunk_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # Stop if chunk is missing or too small (past end of file)
        if r.returncode != 0 or not os.path.exists(chunk_path) \
                or os.path.getsize(chunk_path) < 2000:
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
            break

        if progress_cb:
            progress_cb(i + 1, n_chunks)

        chunk_srt, next_idx = call_whisper(chunk_path, offset_sec=start_sec,
                                           idx_start=next_idx)
        os.unlink(chunk_path)
        if chunk_srt:
            srt_parts.append(chunk_srt)

    os.unlink(src_path)
    return "\n\n".join(srt_parts)


def parse_srt(srt_text: str) -> list:
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


def enrich_srt_in_batches(or_client, segments, progress_bar, batch_size=25):
    annotated_parts = []
    total_batches   = math.ceil(len(segments) / batch_size)

    for batch_num, start in enumerate(range(0, len(segments), batch_size)):
        batch   = segments[start : start + batch_size]
        raw_srt = "\n\n".join(
            f"{s['index']}\n{s['timestamp']}\n{s['text']}" for s in batch
        )
        prompt = f"""You are a cinematic AI script supervisor for an episodic audio drama.

Add a [MORPHIC] line directly after each subtitle line in this SRT batch.

Each [MORPHIC] prompt must be 1-2 sentences and include:
- Visual description of what is happening (characters, objects, setting)
- Camera movement: slow dolly in, slow dolly out, wide crane shot, handheld follow, static close-up, rack focus, low angle push in, aerial drone pull back, whip pan, slow pan left/right
- Lighting style: golden hour, harsh shadows, soft diffused, moonlit blue, candlelight, neon, overcast grey
- Mood keywords: melancholic, tense, joyful, mysterious, epic, intimate, ominous, hopeful
- If a character is mentioned, describe their appearance and action

Return ONLY the annotated SRT text. No JSON, no headers, no extra explanation.

SRT:
{raw_srt}"""

        resp = or_client.chat.completions.create(
            model=CLAUDE_MODEL, max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        annotated_parts.append(resp.choices[0].message.content.strip())
        progress_bar.progress((batch_num + 1) / total_batches)

    return "\n\n".join(annotated_parts)


def extract_characters_and_locations(or_client, srt_text: str) -> dict:
    excerpt = srt_text[:12_000]
    prompt  = f"""You are a story analyst and art director. Analyse this SRT transcript from an audio drama episode.

Return ONLY valid JSON (no markdown fences, no extra text) with this exact structure:

{{
  "characters": [
    {{
      "name": "Character Name",
      "role": "protagonist | antagonist | supporting | narrator",
      "physical_description": "Height, build, hair, eye colour, skin tone, distinguishing features",
      "costume": "Full clothing - fabric, colour, accessories, footwear, headwear",
      "personality_notes": "2-3 adjectives capturing their personality",
      "visual_style_prompt": "Ready-to-paste 1-2 sentence Morphic image prompt for this character"
    }}
  ],
  "locations": [
    {{
      "name": "Location Name",
      "description": "What this place looks like in detail",
      "time_of_day": "morning | afternoon | evening | night | varies",
      "atmosphere": "Overall mood or feel",
      "lighting": "Typical lighting conditions",
      "visual_style_prompt": "Ready-to-paste 1-2 sentence Morphic image prompt for this location"
    }}
  ]
}}

TRANSCRIPT:
{excerpt}"""

    resp = or_client.chat.completions.create(
        model=CLAUDE_MODEL, max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def format_character_sheet(characters, episode_name):
    lines = ["=" * 60, "CHARACTER & COSTUME SHEET"]
    if episode_name:
        lines.append(f"Episode: {episode_name}")
    lines += ["=" * 60, ""]
    for c in characters:
        lines += [
            f"CHARACTER: {c.get('name', 'Unknown').upper()}", "-" * 40,
            f"Role          : {c.get('role', '')}",
            f"Physical      : {c.get('physical_description', '')}",
            f"Costume       : {c.get('costume', '')}",
            f"Personality   : {c.get('personality_notes', '')}",
            "", "[MORPHIC PROMPT FRAGMENT]", c.get("visual_style_prompt", ""), "", "",
        ]
    return "\n".join(lines)


def format_locations_sheet(locations, episode_name):
    lines = ["=" * 60, "LOCATIONS SHEET"]
    if episode_name:
        lines.append(f"Episode: {episode_name}")
    lines += ["=" * 60, ""]
    for loc in locations:
        lines += [
            f"LOCATION: {loc.get('name', 'Unknown').upper()}", "-" * 40,
            f"Description   : {loc.get('description', '')}",
            f"Time of Day   : {loc.get('time_of_day', '')}",
            f"Atmosphere    : {loc.get('atmosphere', '')}",
            f"Lighting      : {loc.get('lighting', '')}",
            "", "[MORPHIC PROMPT FRAGMENT]", loc.get("visual_style_prompt", ""), "", "",
        ]
    return "\n".join(lines)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("---")
col_upload, col_meta = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Episode Audio",
        type=["mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"],
        help="Up to 300 MB. Automatically split into 5-min chunks for fast transcription.",
    )
    if uploaded_file:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > 300:
            st.warning(f"File is {size_mb:.1f} MB - maximum is 300 MB.")
        else:
            st.success(f"{uploaded_file.name}  ({size_mb:.1f} MB) - ready")

with col_meta:
    episode_name = st.text_input(
        "Episode name (used in filenames)",
        placeholder="Episode 01 - The Beginning",
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
    st.markdown("")
    file_ok = bool(uploaded_file and (uploaded_file.size / (1024 * 1024)) <= 300)
    generate_clicked = False
    if file_ok:
        generate_clicked = st.button(
            "Generate Morphic Script", type="primary", use_container_width=True
        )
    else:
        st.button("Generate Morphic Script", disabled=True, use_container_width=True)
        if not uploaded_file:
            st.caption("Upload an audio file to get started")

# ── Processing ────────────────────────────────────────────────────────────────
if file_ok and generate_clicked:

    safe_name = re.sub(r"[^\w\-]", "_", episode_name) if episode_name else "episode"
    st.markdown("---")
    st.markdown("#### Processing")

    # STEP 1 - Transcribe
    step1 = st.status("Step 1 - Transcribing audio with OpenAI Whisper...", expanded=True)
    srt_content = ""
    with step1:
        try:
            ext        = uploaded_file.name.rsplit(".", 1)[-1].lower()
            file_bytes = uploaded_file.read()
            size_mb    = len(file_bytes) / (1024 * 1024)
            openai_client = OpenAI(api_key=openai_key)

            est_chunks = max(1, math.ceil(size_mb / 1.2))
            msg_slot   = st.empty()
            msg_slot.write(f"Splitting into ~{est_chunks} chunks of 5 min each...")

            def on_chunk(done, total):
                msg_slot.write(f"Chunk {done}/{total} transcribed...")

            srt_content = transcribe_audio(
                openai_client, file_bytes, ext, language, progress_cb=on_chunk
            )
            step1.update(label="Step 1 - Transcription complete!", state="complete")
        except Exception as e:
            step1.update(label="Step 1 - Transcription failed", state="error")
            st.error(f"Whisper error: {e}")
            st.stop()

    # STEP 2 - Enrich with Claude
    step2 = st.status("Step 2 - Generating Morphic prompts with Claude...", expanded=True)
    annotated_srt = ""
    with step2:
        try:
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=openrouter_key
            )
            segments = parse_srt(srt_content)
            total    = len(segments)
            st.write(f"Found {total} subtitle segments - enriching in batches...")
            bar = st.progress(0)
            annotated_srt = enrich_srt_in_batches(or_client, segments, bar)
            step2.update(
                label=f"Step 2 - Morphic prompts added to {total} segments!",
                state="complete",
            )
        except Exception as e:
            step2.update(label="Step 2 - Enrichment failed", state="error")
            st.error(f"Claude/OpenRouter error: {e}")
            st.stop()

    # STEP 3 - Extract characters & locations
    step3 = st.status("Step 3 - Extracting characters, costumes & locations...", expanded=False)
    char_sheet = loc_sheet = ""
    with step3:
        try:
            data       = extract_characters_and_locations(or_client, srt_content)
            char_sheet = format_character_sheet(data.get("characters", []), episode_name)
            loc_sheet  = format_locations_sheet(data.get("locations",  []), episode_name)
            n_chars    = len(data.get("characters", []))
            n_locs     = len(data.get("locations",  []))
            step3.update(
                label=f"Step 3 - Found {n_chars} character(s) and {n_locs} location(s)!",
                state="complete",
            )
        except json.JSONDecodeError:
            step3.update(label="Step 3 - Could not parse JSON", state="error")
            st.warning("Character & location extraction returned unexpected output. SRT is still available.")
        except Exception as e:
            step3.update(label="Step 3 - Extraction failed", state="error")
            st.error(f"Error: {e}")

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download Your Files")
    dcol1, dcol2, dcol3 = st.columns(3)

    with dcol1:
        st.download_button(
            "Annotated SRT", data=annotated_srt,
            file_name=f"{safe_name}_morphic.srt", mime="text/plain",
            use_container_width=True,
        )
        st.caption("Time-coded script with [MORPHIC] camera prompts per line")

    with dcol2:
        if char_sheet:
            st.download_button(
                "Character & Costume Sheet", data=char_sheet,
                file_name=f"{safe_name}_characters.txt", mime="text/plain",
                use_container_width=True,
            )
            st.caption("Characters with costume & Morphic prompt fragments")
        else:
            st.button("Character Sheet", disabled=True, use_container_width=True)
            st.caption("Not available")

    with dcol3:
        if loc_sheet:
            st.download_button(
                "Locations Sheet", data=loc_sheet,
                file_name=f"{safe_name}_locations.txt", mime="text/plain",
                use_container_width=True,
            )
            st.caption("Locations with atmosphere & Morphic prompt fragments")
        else:
            st.button("Locations Sheet", disabled=True, use_container_width=True)
            st.caption("Not available")

    # ── Preview ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Preview")
    tab_srt, tab_chars, tab_locs = st.tabs(["Annotated SRT", "Characters & Costumes", "Locations"])

    with tab_srt:
        preview = annotated_srt[:4000]
        if len(annotated_srt) > 4000:
            preview += "\n\n... (truncated - download for full file)"
        st.text_area("", value=preview, height=420, label_visibility="collapsed")

    with tab_chars:
        st.text_area("", value=char_sheet or "No data", height=420, label_visibility="collapsed")

    with tab_locs:
        st.text_area("", value=loc_sheet or "No data", height=420, label_visibility="collapsed")
