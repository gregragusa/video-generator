# app.py
# -------------------------------------------------------
# Streamlit app: IMMAGINI / AUDIO con Replicate + FishAudio.
# ❌ NESSUN RESUME: ogni run riparte da zero e pulisce le cartelle output
# ✅ Audio spezzato ~N caratteri (500/1000/custom), sempre a fine frase
# ✅ Anti-taglio: punteggiatura + spazio finale e ~0.25s di silenzio in coda ad ogni clip
# ✅ NON UNISCE i clip: crea ZIP con tutti gli audio generati
# ✅ Voce FishAudio FORZATA: 80e34d5e0b2b4577a486f3a77e357261
# -------------------------------------------------------

import os
import re
import time
import zipfile
import shutil
import subprocess
import streamlit as st

# opzionale
try:
    from scripts.config_loader import load_config  # type: ignore
except Exception:  # pragma: no cover
    load_config = None

# utils base (richiesti)
from scripts.utils import (  # type: ignore
    chunk_by_sentences_count,
    generate_audio,
    generate_images,
    mp3_duration_seconds,
)

# -------------------------------------------------------
# Costanti / Utility
# -------------------------------------------------------
AUDIO_EXTS = ("mp3", "wav", "m4a")
IMAGE_EXTS = ("png", "jpg", "jpeg")

TAIL_SILENCE_SECS = 0.25  # silenzio aggiunto alla FINE di OGNI clip

# 👉 VOCE FISSA FishAudio
DEFAULT_FISHAUDIO_VOICE_ID = "80e34d5e0b2b4577a486f3a77e357261"


def sanitize(title: str) -> str:
    s = (title or "").lower()
    tr = [(" ", "_"), ("ù", "u"), ("à", "a"), ("è", "e"), ("ì", "i"), ("ò", "o"), ("é", "e")]
    for a, b in tr:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum() or ch == "_") or "video"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def empty_dir(path: str):
    """Svuota la directory (senza cancellarla)."""
    ensure_dir(path)
    for n in os.listdir(path):
        fp = os.path.join(path, n)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception:
            pass


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except Exception:
        return -1


def move_single_output(src_dir: str, dst_fullpath_no_ext: str) -> str | None:
    """
    Sposta il file più recente da src_dir in dst_fullpath_no_ext.ext
    Aspetta che la dimensione si stabilizzi per evitare file in scrittura.
    """
    files = [n for n in os.listdir(src_dir) if not n.startswith(".")]
    if not files:
        return None
    files.sort(key=lambda n: os.path.getmtime(os.path.join(src_dir, n)))
    src = os.path.join(src_dir, files[-1])
    # attendo stabilizzazione dimensione
    last = _file_size(src)
    for _ in range(12):  # fino a ~1.2s
        time.sleep(0.1)
        cur = _file_size(src)
        if cur == last and cur > 0:
            break
        last = cur
    ext = files[-1].split(".")[-1].lower()
    dst = f"{dst_fullpath_no_ext}.{ext}"
    try:
        os.replace(src, dst)
    except Exception:
        shutil.copy2(src, dst)
        try:
            os.remove(src)
        except Exception:
            pass
    # pulizia residui
    for n in os.listdir(src_dir):
        try:
            os.remove(os.path.join(src_dir, n))
        except Exception:
            pass
    return dst


def sentences_from_script(script: str) -> list[str]:
    text = (script or "").strip()
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def build_audio_chunks(script: str, target_chars: int = 1000) -> list[str]:
    """
    Split greedy per frasi con target ≈N char per chunk.
    - Non taglia MAI parole o frasi.
    - Se una singola frase supera target, la teniamo intera.
    - Chiude con punteggiatura .?! e aggiunge spazio finale (aiuta TTS).
    """
    sents = sentences_from_script(script)
    chunks: list[str] = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        candidate = (f"{buf} {s}".strip()) if buf else s
        if buf and len(candidate) > target_chars:
            if buf[-1] not in ".?!":
                buf += "."
            if not buf.endswith(" "):
                buf += " "
            chunks.append(buf)
            buf = s
        else:
            buf = candidate

    if buf:
        if buf[-1] not in ".?!":
            buf += "."
        if not buf.endswith(" "):
            buf += " "
        chunks.append(buf)

    return chunks


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _append_tail_silence_mp3(in_path: str, out_path: str, tail_secs: float = TAIL_SILENCE_SECS) -> bool:
    """
    Aggiunge tail_secs di silenzio alla fine di un MP3.
    Implementazione: genera clip di silenzio e concatena con filter_complex.
    """
    ff = _ffmpeg_exe()
    if not ff:
        return False
    tmp_dir = os.path.dirname(out_path)
    os.makedirs(tmp_dir, exist_ok=True)
    sil = os.path.join(tmp_dir, "_tail_silence.mp3")
    try:
        r_sil = subprocess.run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", f"{tail_secs}",
            "-c:a", "libmp3lame", "-b:a", "192k",
            sil
        ], capture_output=True, text=True)
        if r_sil.returncode != 0 or not os.path.exists(sil):
            return False

        r = subprocess.run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-i", sil,
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
            "-map", "[a]",
            "-ar", "44100", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "192k",
            out_path
        ], capture_output=True, text=True)

        ok = r.returncode == 0 and os.path.exists(out_path)
        try:
            os.remove(sil)
        except Exception:
            pass
        return ok
    except Exception:
        return False


def zip_audios(src_dir: str, zip_path: str) -> str | None:
    """Crea ZIP con tutti i file part_*.mp3/.wav/.m4a presenti in src_dir."""
    files = []
    for n in sorted(os.listdir(src_dir)):
        if not n.startswith("part_"):
            continue
        ext = n.split(".")[-1].lower()
        if ext in AUDIO_EXTS:
            files.append(os.path.join(src_dir, n))
    if not files:
        return None
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in files:
            zf.write(f, arcname=os.path.basename(f))
    return zip_path


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Generatore Contenuti", page_icon="🎬", layout="wide")
st.title("🎬 Generatore di Immagini e Audio (clip separati in ZIP)")

base_cfg = {}
if load_config:
    try:
        loaded = load_config()
        if isinstance(loaded, dict):
            base_cfg = loaded
    except Exception:
        base_cfg = {}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔐 API Keys")
    rep_prefill = st.session_state.get("replicate_api_key", "")
    fish_prefill = st.session_state.get("fish_audio_api_key", "")
    with st.form("api_keys_form_main", clear_on_submit=False):
        replicate_key = st.text_input("Replicate API key", type="password", value=rep_prefill)
        fish_key = st.text_input("FishAudio API key", type="password", value=fish_prefill)
        save_keys = st.form_submit_button("💾 Salva")
    if save_keys:
        st.session_state["replicate_api_key"] = replicate_key.strip()
        st.session_state["fish_audio_api_key"] = fish_key.strip()
        st.success("Chiavi salvate!")

    st.divider()
    st.header("⚙️ Parametri")
    st.info(f"🎙️ Voice fissa: `{DEFAULT_FISHAUDIO_VOICE_ID}`")

    model_presets = [
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-dev",
        "stability-ai/stable-diffusion-xl-base-1.0",
        "bytedance/sdxl-lightning-4step",
        "playgroundai/playground-v2.5-1024px-aesthetic",
        "Custom (digita sotto)",
    ]
    preset_selected = st.selectbox("Modello Replicate", model_presets, index=0)
    custom_prefill = st.session_state.get("replicate_model_custom", "")
    custom_model = st.text_input("Custom model (owner/name:tag)", value=custom_prefill)
    if custom_model != custom_prefill:
        st.session_state["replicate_model_custom"] = custom_model.strip()
    effective_model = (
        st.session_state.get("replicate_model_custom", "").strip()
        if preset_selected == "Custom (digita sotto)"
        else preset_selected
    )
    st.session_state["replicate_model"] = effective_model

    st.divider()
    st.subheader("⚡ Velocità")
    speed_mode = st.selectbox("Modalità", ["🐌 Lenta", "⚡ Veloce", "🚀 Turbo"], index=1)
    if speed_mode == "⚡ Veloce":
        st.session_state["sleep_time"] = 5
    elif speed_mode == "🚀 Turbo":
        st.session_state["sleep_time"] = 2
    else:
        st.session_state["sleep_time"] = 11

st.write(
    f"🔎 **Stato API** → Replicate: {'✅' if (st.session_state.get('replicate_api_key') or os.environ.get('REPLICATE_API_TOKEN')) else '⚠️'} · "
    f"FishAudio: {'✅' if (st.session_state.get('fish_audio_api_key') or os.environ.get('FISHAUDIO_API_KEY')) else '⚠️'} · "
    f"Model: `{st.session_state.get('replicate_model') or '—'}` · Voice: `{DEFAULT_FISHAUDIO_VOICE_ID}`"
)

# ---------------- Main UI ----------------
col_main, col_side = st.columns([2, 1])
with col_main:
    st.subheader("📝 Input")
    title = st.text_input("Titolo del progetto")
    script = st.text_area("Testo per generare immagini/audio", height=260)
    mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio (clip separati)", "Entrambi"], index=1)

    # nuova opzione: target per clip audio
    target_chars = st.number_input(
        "Target caratteri per clip audio (≈, chiude sempre a fine frase)",
        min_value=300, max_value=2000, value=1000, step=50,
        help="Esempi: 500 o 1000. Se una singola frase supera il target, viene tenuta intera."
    )

    if mode in ["Audio (clip separati)", "Entrambi"]:
        st.caption("I clip audio NON verranno uniti; potrai scaricarli tutti in uno ZIP.")
    else:
        sentences_per_image = st.number_input("Quante frasi per immagine?", min_value=1, value=2, step=1)
        st.session_state["sentences_per_image"] = sentences_per_image

    generate = st.button("🚀 Genera", use_container_width=True)

with col_side:
    st.subheader("📊 Avanzamento")
    audio_prog = st.progress(0.0, text="Audio 0%")
    img_prog = st.progress(0.0, text="Immagini 0%")

# -------------------------------------------------------
# Driver di generazione (sempre da zero)
# -------------------------------------------------------
if generate and title.strip() and script.strip():
    safe = sanitize(title)
    base = os.path.join("data", "outputs", safe)
    aud_dir = os.path.join(base, "audio")
    img_dir = os.path.join(base, "images")
    ensure_dir(base); ensure_dir(aud_dir); ensure_dir(img_dir)

    # ❌ niente resume: pulisco tutto
    empty_dir(aud_dir)
    empty_dir(img_dir)
    try:
        os.remove(os.path.join(base, "output.zip"))
    except Exception:
        pass
    try:
        os.remove(os.path.join(base, "audio_clips.zip"))
    except Exception:
        pass

    # API / Modello
    rep_key = (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN", "")).strip()
    fish_key = (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY", "")).strip()
    model = (st.session_state.get("replicate_model") or "").strip()

    if mode in ["Audio (clip separati)", "Entrambi"]:
        if not fish_key:
            st.error("❌ FishAudio API key mancante!")
            st.stop()
    if mode in ["Immagini", "Entrambi"]:
        if not rep_key:
            st.error("❌ Replicate API key mancante!")
            st.stop()
        if not model:
            st.error("❌ Modello Replicate mancante!")
            st.stop()

    # runtime cfg
    runtime_cfg = dict(base_cfg)
    if rep_key:
        os.environ["REPLICATE_API_TOKEN"] = rep_key
        runtime_cfg["replicate_api_key"] = rep_key
        runtime_cfg["replicate_api_token"] = rep_key
    if fish_key:
        os.environ["FISHAUDIO_API_KEY"] = fish_key
        runtime_cfg["fishaudio_api_key"] = fish_key
    if model:
        runtime_cfg["replicate_model"] = model

    # 👉 forza la voce ovunque (hardcoded)
    voice = DEFAULT_FISHAUDIO_VOICE_ID
    for k in [
        "FISHAUDIO_VOICE_ID", "FISHAUDIO_VOICE", "FISHAUDIO_SPEAKER", "FISHAUDIO_SPEAKER_ID",
        "VOICE_MODEL_ID", "VOICE_ID", "VOICE", "SPEAKER", "SPEAKER_ID"
    ]:
        os.environ[k] = voice
    for k in [
        "fishaudio_voice_id", "fishaudio_voice", "fishaudio_speaker", "fishaudio_speaker_id",
        "voice_model_id", "voice_id", "voice", "speaker", "speaker_id"
    ]:
        runtime_cfg[k] = voice

    # ----------------- AUDIO (clip separati) -----------------
    audio_zip_ready = False
    audio_clips_count = 0
    if mode in ["Audio (clip separati)", "Entrambi"]:
        # prepara chunk testuali
        chunks = build_audio_chunks(script, target_chars=int(target_chars))
        total = len(chunks)
        st.info(f"🎧 Generazione di {total} clip audio separati… (voice: {voice})")

        for i, ch in enumerate(chunks):
            tmp = os.path.join(aud_dir, "_tmp")
            empty_dir(tmp)
            # genera UN SOLO clip alla volta per evitare concatenazione interna
            try:
                generate_audio([ch], runtime_cfg, tmp)  # type: ignore
            except TypeError:
                generate_audio([ch], runtime_cfg, tmp)

            target_noext = os.path.join(aud_dir, f"part_{i:03d}")
            out_path = move_single_output(tmp, target_noext)
            if not out_path or not os.path.exists(out_path):
                st.error(f"❌ Errore nel generare clip {i+1}/{total}")
                st.stop()

            # se è mp3, aggiungo tail di silenzio anti-taglio
            if out_path.lower().endswith(".mp3"):
                tailed = f"{target_noext}.tail.mp3"
                if _append_tail_silence_mp3(out_path, tailed, TAIL_SILENCE_SECS):
                    try: os.remove(out_path)
                    except Exception: pass
                    try: os.replace(tailed, f"{target_noext}.mp3")
                    except Exception: shutil.copy2(tailed, f"{target_noext}.mp3")
                    try: os.remove(tailed)
                    except Exception: pass

            audio_prog.progress((i + 1) / total, text=f"Audio {int((i+1)/total*100)}%")

        # crea ZIP con tutti i clip
        audio_zip = os.path.join(base, "audio_clips.zip")
        z = zip_audios(aud_dir, audio_zip)
        if z and os.path.exists(z):
            audio_zip_ready = True
            audio_clips_count = len([n for n in os.listdir(aud_dir) if n.startswith("part_") and n.split(".")[-1].lower() in AUDIO_EXTS])
            st.success(f"🗜️ ZIP audio pronto: {audio_clips_count} clip.")

    # ----------------- IMMAGINI -----------------
    zip_images_ready = False
    if mode in ["Immagini", "Entrambi"]:
        if mode == "Immagini":
            img_chunks_plan = chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2)))
        else:
            # se vuoi legare il numero di immagini alla durata totale, puoi farlo solo quando hai un audio unico.
            # qui i clip sono separati: generiamo immagini per frasi (semplice).
            img_chunks_plan = chunk_by_sentences_count(script, 2)

        total_i = len(img_chunks_plan)
        st.info(f"🖼️ Generazione di {total_i} immagini…")
        for i, ptxt in enumerate(img_chunks_plan):
            tmp = os.path.join(img_dir, "_tmp")
            empty_dir(tmp)
            try:
                generate_images([ptxt], runtime_cfg, tmp)  # type: ignore
            except TypeError:
                generate_images([ptxt], runtime_cfg, tmp)
            target_noext = os.path.join(img_dir, f"img_{i:03d}")
            out = move_single_output(tmp, target_noext)
            if not out or not os.path.exists(out):
                st.error(f"❌ Errore nel generare immagine {i+1}/{total_i}")
                st.stop()
            img_prog.progress((i + 1) / total_i, text=f"Immagini {int((i+1)/total_i*100)}%")

        # ZIP immagini
        img_zip = os.path.join(base, "output.zip")
        if os.path.exists(img_zip):
            try:
                os.remove(img_zip)
            except Exception:
                pass
        with zipfile.ZipFile(img_zip, "w") as zf:
            for fn in os.listdir(img_dir):
                if fn.startswith("_tmp"):
                    continue
                full = os.path.join(img_dir, fn)
                if os.path.isfile(full) and full.split(".")[-1].lower() in IMAGE_EXTS:
                    zf.write(full, arcname=os.path.join("images", fn))
        zip_images_ready = os.path.exists(img_zip)
        if zip_images_ready:
            st.success("🗜️ ZIP immagini pronto.")

    # ----------------- DOWNLOAD -----------------
    st.divider()
    st.subheader("📥 Download")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎧 Clip Audio (ZIP)")
        audio_zip = os.path.join(base, "audio_clips.zip")
        if audio_zip_ready and os.path.exists(audio_zip):
            size_mb = os.path.getsize(audio_zip) / (1024 * 1024)
            st.info(f"{audio_clips_count} clip · {size_mb:.1f} MB")
            with open(audio_zip, "rb") as f:
                st.download_button(
                    "Scarica ZIP Audio",
                    f.read(),
                    file_name=f"{sanitize(title)}_audio_clips.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.info("⏳ Nessun ZIP audio pronto")
    with c2:
        st.markdown("### 🖼️ Immagini (ZIP)")
        img_zip = os.path.join(base, "output.zip")
        if zip_images_ready and os.path.exists(img_zip):
            with open(img_zip, "rb") as f:
                st.download_button(
                    "Scarica ZIP Immagini",
                    f.read(),
                    file_name=f"{sanitize(title)}_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.info("⏳ Nessun ZIP immagini pronto")
