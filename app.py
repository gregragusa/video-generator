# app.py
# -------------------------------------------------------
# Streamlit app: IMMAGINI / AUDIO con Replicate + FishAudio.
# ❌ NESSUN RESUME: ogni run riparte da zero e pulisce le cartelle output
# ✅ Audio spezzato ~1000 caratteri, sempre a fine frase
# ✅ Pausa ~0.8s tra gli spezzoni nel merge finale
# ✅ Voce FishAudio FORZATA: 80e34d5e0b2b4577a486f3a77e357261
# -------------------------------------------------------

import os
import re
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
SILENCE_BETWEEN_PARTS_SECS = 0.8  # pausa tra spezzoni audio

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


def move_single_output(src_dir: str, dst_fullpath_no_ext: str, preferred_exts) -> str | None:
    files = [n for n in os.listdir(src_dir) if not n.startswith(".")]
    if not files:
        return None
    files.sort(key=lambda n: os.path.getmtime(os.path.join(src_dir, n)))
    src = os.path.join(src_dir, files[-1])  # prendi il PIÙ recente
    ext = files[-1].split(".")[-1].lower()
    # salviamo col suo ext; eventuale transcodifica avverrà nel combine
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


def zip_images(base_dir: str) -> str | None:
    import zipfile
    zip_path = os.path.join(base_dir, "output.zip")
    img_dir = os.path.join(base_dir, "images")
    if not os.path.exists(img_dir):
        return None
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fn in os.listdir(img_dir):
            if fn.startswith("_tmp"):
                continue
            full = os.path.join(img_dir, fn)
            if os.path.isfile(full) and full.split(".")[-1].lower() in IMAGE_EXTS:
                zf.write(full, arcname=os.path.join("images", fn))
    return zip_path


# -------------------------------------------------------
# SPLIT AUDIO: ~1000 caratteri, sempre a fine frase
# -------------------------------------------------------
def sentences_from_script(script: str) -> list[str]:
    # split robusto per . ? ! seguiti da spazio/line-break
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", (script or "").strip()) if s.strip()]


def build_audio_chunks(script: str, target_chars: int = 1000) -> list[str]:
    """
    Split greedy per frasi con target ≈1000 char per chunk.
    - Non taglia MAI parole o frasi.
    - Se una singola frase supera target, la teniamo intera.
    - Aggiunge un punto finale se il chunk non termina con .?!
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
            if buf and buf[-1] not in ".?!":
                buf += "."
            chunks.append(buf.strip())
            buf = s
        else:
            buf = candidate

    if buf:
        if buf[-1] not in ".?!":
            buf += "."
        chunks.append(buf.strip())

    return chunks


# -------------------------------------------------------
# Combine Audio (formati misti) + pausa tra parti
# -------------------------------------------------------
def list_audio_parts(aud_dir: str) -> list[str]:
    files = []
    for n in sorted(os.listdir(aud_dir)):
        if not n.startswith("part_"):
            continue
        if n.split(".")[-1].lower() in AUDIO_EXTS:
            files.append(os.path.join(aud_dir, n))
    return files


def combine_parts_to_mp3(aud_dir: str, out_path: str) -> bool:
    """
    Concatena i part_*.audio in un unico MP3.
    1) Normalizza ogni parte in mp3 44.1kHz stereo 192kbps
    2) Inserisce ~0.8s di silenzio dopo ogni parte (tranne l'ultima)
    3) Concat demuxer; fallback: filter_complex
    """
    parts = list_audio_parts(aud_dir)
    if not parts:
        return False

    if len(parts) == 1:
        try:
            r = subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", parts[0],
                "-ar", "44100", "-ac", "2",
                "-codec:a", "libmp3lame", "-b:a", "192k",
                out_path,
            ], capture_output=True, text=True)
            return r.returncode == 0 and os.path.exists(out_path)
        except Exception:
            return False

    tmp_dir = os.path.join(aud_dir, "_concat_tmp")
    empty_dir(tmp_dir)

    # 1) normalizza
    mp3_parts: list[str] = []
    for i, p in enumerate(parts):
        tmp_mp3 = os.path.join(tmp_dir, f"p_{i:03d}.mp3")
        try:
            r = subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", p,
                "-ar", "44100", "-ac", "2",
                "-codec:a", "libmp3lame", "-b:a", "192k",
                tmp_mp3,
            ], capture_output=True, text=True)
            if r.returncode != 0:
                return False
            mp3_parts.append(tmp_mp3)
        except Exception:
            return False

    # 2) genera silenzio
    silence_mp3 = os.path.join(tmp_dir, "silence.mp3")
    try:
        r_sil = subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", f"{SILENCE_BETWEEN_PARTS_SECS}",
            "-c:a", "libmp3lame", "-b:a", "192k",
            silence_mp3
        ], capture_output=True, text=True)
        if r_sil.returncode != 0 or not os.path.exists(silence_mp3):
            silence_mp3 = None
    except Exception:
        silence_mp3 = None

    # 3) pezzo + silenzio (tranne ultimo)
    mp3_with_silence: list[str] = []
    for i, p in enumerate(mp3_parts):
        if silence_mp3 and i < len(mp3_parts) - 1:
            out_p = os.path.join(tmp_dir, f"ps_{i:03d}.mp3")
            r = subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", p, "-i", silence_mp3,
                "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                "-map", "[a]", "-c:a", "libmp3lame", "-b:a", "192k",
                out_p
            ], capture_output=True, text=True)
            mp3_with_silence.append(out_p if r.returncode == 0 and os.path.exists(out_p) else p)
        else:
            mp3_with_silence.append(p)

    # 4) concat demuxer
    def _posix(pth: str) -> str:
        return os.path.abspath(pth).replace("\\", "/")

    filelist = os.path.join(tmp_dir, "list.txt")
    try:
        with open(filelist, "w", encoding="utf-8") as f:
            for p in mp3_with_silence:
                f.write(f"file '{_posix(p)}'\n")

        r = subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", filelist,
            "-c:a", "libmp3lame", "-b:a", "192k",
            out_path,
        ], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out_path):
            empty_dir(tmp_dir)
            return True
    except Exception:
        pass

    # 5) fallback filter_complex
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for p in mp3_with_silence:
            cmd += ["-i", p]
        n = len(mp3_with_silence)
        inputs = "".join([f"[{i}:a]" for i in range(n)])
        filter_complex = f"{inputs}concat=n={n}:v=0:a=1[a]"
        cmd += ["-filter_complex", filter_complex, "-map", "[a]", "-c:a", "libmp3lame", "-b:a", "192k", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = r.returncode == 0 and os.path.exists(out_path)
        return ok
    except Exception:
        return False
    finally:
        empty_dir(tmp_dir)


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="wide")
st.title("🎬 Generatore di Video con Immagini e Audio")

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
        st.session_state["chunk_size"] = 3500
        st.session_state["sleep_time"] = 5
    elif speed_mode == "🚀 Turbo":
        st.session_state["chunk_size"] = 5000
        st.session_state["sleep_time"] = 2
    else:
        st.session_state["chunk_size"] = 2000
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
    title = st.text_input("Titolo del video")
    script = st.text_area("Testo per generare immagini/audio", height=240)
    mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio", "Entrambi"], index=2)

    if mode in ["Audio", "Entrambi"]:
        seconds_per_img = st.number_input(
            "Ogni quanti secondi creare un'immagine?",
            min_value=1, value=8, step=1,
            help="Usato per pianificare il numero di immagini in base alla durata dell'audio."
        )
        st.session_state["seconds_per_img"] = seconds_per_img
    else:
        sentences_per_image = st.number_input("Quante frasi per immagine?", min_value=1, value=2, step=1)
        st.session_state["sentences_per_image"] = sentences_per_image

    generate = st.button("🚀 Genera contenuti", use_container_width=True)

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
        os.remove(os.path.join(aud_dir, "combined_audio.mp3"))
    except Exception:
        pass
    try:
        os.remove(os.path.join(base, "output.zip"))
    except Exception:
        pass

    # API / Modello
    rep_key = (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN", "")).strip()
    fish_key = (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY", "")).strip()
    model = (st.session_state.get("replicate_model") or "").strip()

    if mode in ["Audio", "Entrambi"]:
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
    for k in ["FISHAUDIO_VOICE_ID", "FISHAUDIO_VOICE", "FISHAUDIO_SPEAKER", "VOICE_ID", "VOICE", "SPEAKER"]:
        os.environ[k] = voice
    for k in ["fishaudio_voice_id", "fishaudio_voice", "fishaudio_speaker", "voice_id", "voice", "speaker"]:
        runtime_cfg[k] = voice

    # ----------------- AUDIO -----------------
    audio_ready = False
    combined_audio_path = os.path.join(aud_dir, "combined_audio.mp3")
    if mode in ["Audio", "Entrambi"]:
        chunks = build_audio_chunks(script, target_chars=1000)
        total = len(chunks)
        st.info(f"🎧 Generazione audio in {total} spezzoni…")
        for i, ch in enumerate(chunks):
            tmp = os.path.join(aud_dir, "_tmp")
            empty_dir(tmp)
            try:
                generate_audio([ch], runtime_cfg, tmp)  # type: ignore
            except TypeError:
                generate_audio([ch], runtime_cfg, tmp)
            target_noext = os.path.join(aud_dir, f"part_{i:03d}")
            out = move_single_output(tmp, target_noext, preferred_exts=AUDIO_EXTS)
            if not out or not os.path.exists(out):
                st.error(f"❌ Errore nel generare chunk audio {i+1}/{total}")
                st.stop()
            audio_prog.progress((i + 1) / total, text=f"Audio {int((i+1)/total*100)}%")
        ok = combine_parts_to_mp3(aud_dir, combined_audio_path)
        if not ok:
            st.warning("ℹ️ ffmpeg non disponibile o merge fallito: restano i file part_*.audio")
        else:
            audio_ready = True
            st.success("🎵 Audio combinato pronto.")

    # ----------------- IMMAGINI -----------------
    zip_ready = False
    if mode in ["Immagini", "Entrambi"]:
        if mode == "Entrambi" and audio_ready and os.path.exists(combined_audio_path):
            # pianifica in base alla durata dell'audio
            duration = mp3_duration_seconds(combined_audio_path) or 60
            seconds_per_img = st.session_state.get("seconds_per_img", 8)
            num_images = max(1, int(duration // seconds_per_img))
            sents = sentences_from_script(script)
            if num_images == 1:
                img_chunks_plan = [script]
            else:
                per_img = max(1, len(sents) // max(1, num_images))
                img_chunks_plan = [" ".join(sents[i:i + per_img]) for i in range(0, len(sents), per_img)]
        else:
            # solo immagini: per frasi
            img_chunks_plan = chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2)))

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
            out = move_single_output(tmp, target_noext, preferred_exts=IMAGE_EXTS)
            if not out or not os.path.exists(out):
                st.error(f"❌ Errore nel generare immagine {i+1}/{total_i}")
                st.stop()
            img_prog.progress((i + 1) / total_i, text=f"Immagini {int((i+1)/total_i*100)}%")

        # ZIP
        zip_path = zip_images(base)
        if zip_path and os.path.exists(zip_path):
            zip_ready = True
            st.success("🗜️ ZIP immagini pronto.")

    # ----------------- DOWNLOAD -----------------
    st.divider()
    st.subheader("📥 Download Files")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎧 Audio")
        if audio_ready and os.path.exists(combined_audio_path):
            try:
                size_mb = os.path.getsize(combined_audio_path) / (1024 * 1024)
                dur = mp3_duration_seconds(combined_audio_path)
                bitrate = (size_mb * 8 * 1024) / dur if dur and dur > 0 else 0
                st.info(f"Durata: {dur:.1f}s · Dimensione: {size_mb:.1f} MB · ~{bitrate:.0f} kbps")
            except Exception:
                pass
            with open(combined_audio_path, "rb") as f:
                st.download_button(
                    "Scarica MP3",
                    f.read(),
                    file_name=f"{sanitize(title)}.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                )
        else:
            st.info("⏳ Nessun audio pronto")
    with c2:
        st.markdown("### 🖼️ Immagini")
        zip_path = os.path.join(base, "output.zip")
        if zip_ready and os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    "Scarica ZIP Immagini",
                    f.read(),
                    file_name=f"{sanitize(title)}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.info("⏳ Nessuna immagine pronta")
