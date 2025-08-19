# app.py
# -------------------------------------------------------
# Streamlit app: API e parametri (modello/voce), genera IMMAGINI / AUDIO.
# Compatibile con Python 3.13: niente pydub; usiamo mutagen + ffmpeg via imageio-ffmpeg.
# Con ripresa progressi AUDIO (resume): salva manifest + chunk MP3 numerati.
# Concat robusta: fallback via WAV -> concat -> MP3 finale.
# -------------------------------------------------------

import os
import re
import json
import hashlib
import shutil
import textwrap
import subprocess
import requests
import streamlit as st

# se hai questo loader lo usiamo, altrimenti proseguiamo senza
try:
    from scripts.config_loader import load_config  # opzionale
except Exception:
    load_config = None

from scripts.utils import (
    chunk_text,
    chunk_by_sentences_count,
    generate_audio,       # riusato per generare un singolo chunk alla volta
    generate_images,
    mp3_duration_seconds, # util per leggere durata MP3
)

# imageio-ffmpeg per trovare ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"  # fallback: deve essere nel PATH

# ---------------------------
# Utility di base
# ---------------------------
def sanitize(title: str) -> str:
    s = (title or "").lower()
    for a, b in [(" ", "_"), ("ù", "u"), ("à", "a"), ("è", "e"),
                 ("ì", "i"), ("ò", "o"), ("é", "e")]:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum() or ch == "_") or "video"

def zip_images(base_dir: str):
    import zipfile
    zip_path = os.path.join(base_dir, "output.zip")
    img_dir = os.path.join(base_dir, "images")
    if not os.path.exists(img_dir):
        return None
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for filename in sorted(os.listdir(img_dir)):
            full_path = os.path.join(img_dir, filename)
            if os.path.isfile(full_path):
                zipf.write(full_path, arcname=os.path.join("images", filename))
    return zip_path

def _clean_token(tok: str) -> str:
    return re.sub(r"\s+", "", (tok or ""))

def _mask(tok: str) -> str:
    t = (tok or "").strip()
    return t[:3] + "…" + t[-4:] if len(t) > 8 else "—"

def script_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

# ---------------------------
# Split testo per AUDIO a ~2000 caratteri spezzando sui punti
# ---------------------------
def split_text_into_sentence_chunks(text: str, max_chars: int = 2000):
    """
    Divide il testo in blocchi di circa max_chars caratteri,
    spezzando dopo . ! ? dove possibile.
    Se una singola frase supera max_chars, la spezza duramente.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []

    sentences = re.split(r"(?<=[\.\!\?])\s+", t)
    chunks, acc = [], ""

    def flush_acc():
        nonlocal acc
        if acc.strip():
            chunks.append(acc.strip())
        acc = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) > max_chars:
            flush_acc()
            parts = textwrap.wrap(s, width=max_chars, break_long_words=True, break_on_hyphens=False)
            chunks.extend([p.strip() for p in parts if p.strip()])
            continue

        new_len = len(s) if not acc else len(acc) + 1 + len(s)
        if new_len <= max_chars:
            acc = s if not acc else f"{acc} {s}"
        else:
            flush_acc()
            acc = s

    flush_acc()
    return [c for c in chunks if c]

# ---------------------------
# Manifest + concat MP3
# ---------------------------
def manifest_path(aud_dir: str) -> str:
    return os.path.join(aud_dir, "audio_manifest.json")

def load_manifest(aud_dir: str) -> dict | None:
    p = manifest_path(aud_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_manifest(aud_dir: str, data: dict) -> None:
    p = manifest_path(aud_dir)
    tmp = p + ".tmp"
    os.makedirs(aud_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def chunk_filename(aud_dir: str, idx: int) -> str:
    return os.path.join(aud_dir, f"chunk_{idx:04d}.mp3")

def _ffconcat_escape(path: str) -> str:
    """Escapa backslash e apici singoli per il file di lista ffmpeg (-f concat)."""
    return path.replace("\\", "\\\\").replace("'", "\\'")

def _run_ffmpeg(cmd: list[str]) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, (p.stderr.decode("utf-8", "ignore") or p.stdout.decode("utf-8", "ignore"))
    except subprocess.CalledProcessError as e:
        return False, (e.stderr.decode("utf-8", "ignore") if e.stderr else str(e))

def _concat_try_copy(aud_dir: str, out_path: str, total_chunks: int) -> tuple[bool, str]:
    list_file = os.path.join(aud_dir, "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for i in range(total_chunks):
            esc = _ffconcat_escape(chunk_filename(aud_dir, i))
            f.write(f"file '{esc}'\n")
    cmd = [FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_path]
    return _run_ffmpeg(cmd)

def _concat_try_reencode_from_list(aud_dir: str, out_path: str) -> tuple[bool, str]:
    """Usa lo stesso list file, ma forza ricodifica a MP3 (potrebbe comunque fallire se il demuxer non accetta stream eterogenei)."""
    list_file = os.path.join(aud_dir, "concat_list.txt")
    # proviamo libmp3lame, poi mp3
    cmd1 = [FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-vn", "-c:a", "libmp3lame", "-q:a", "2", out_path]
    ok, log = _run_ffmpeg(cmd1)
    if ok:
        return ok, log
    cmd2 = [FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-vn", "-c:a", "mp3", "-q:a", "2", out_path]
    return _run_ffmpeg(cmd2)

def _concat_via_wav(aud_dir: str, out_path: str, total_chunks: int) -> tuple[bool, str]:
    """
    Fallback robusto:
      1) Ricodifica ciascun chunk in WAV coerente (stereo, 44.1kHz, s16).
      2) Concatena i WAV (copy) in combined.wav
      3) Ricodifica il WAV finale in MP3 (libmp3lame o mp3)
    """
    tmp_dir = os.path.join(aud_dir, "_concat_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1) mp3 -> wav coerenti
    wav_paths = []
    for i in range(total_chunks):
        src = chunk_filename(aud_dir, i)
        if not os.path.exists(src):
            return False, f"Chunk mancante: {src}"
        dst = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        cmd = [FFMPEG_EXE, "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", dst]
        ok, log = _run_ffmpeg(cmd)
        if not ok:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, f"WAV transcode fallita al chunk {i}: {log}"
        wav_paths.append(dst)

    # 2) concatena WAV (copy)
    wav_list = os.path.join(tmp_dir, "wav_list.txt")
    with open(wav_list, "w", encoding="utf-8") as f:
        for p in wav_paths:
            esc = _ffconcat_escape(p)
            f.write(f"file '{esc}'\n")
    combined_wav = os.path.join(tmp_dir, "combined.wav")
    ok, log = _run_ffmpeg([FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", wav_list, "-c", "copy", combined_wav])
    if not ok:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, f"Concat WAV fallita: {log}"

    # 3) WAV -> MP3 finale
    cmd1 = [FFMPEG_EXE, "-y", "-i", combined_wav, "-vn", "-c:a", "libmp3lame", "-q:a", "2", out_path]
    ok, log = _run_ffmpeg(cmd1)
    if not ok:
        cmd2 = [FFMPEG_EXE, "-y", "-i", combined_wav, "-vn", "-c:a", "mp3", "-q:a", "2", out_path]
        ok, log = _run_ffmpeg(cmd2)

    # pulizia temp
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return ok, log

def concat_mp3_chunks(aud_dir: str, out_path: str, total_chunks: int) -> bool:
    """
    Strategia:
      A) concat demuxer + copy (veloce)
      B) concat demuxer con ricodifica diretta a mp3
      C) Fallback robusto: WAV coerenti -> concat -> MP3
    """
    if total_chunks <= 0:
        return False

    # Se esiste già un MP3 finale non vuoto, riusa
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    # A) veloce
    ok, log = _concat_try_copy(aud_dir, out_path, total_chunks)
    if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    # B) ricodifica diretta (potrebbe comunque fallire)
    ok, log_b = _concat_try_reencode_from_list(aud_dir, out_path)
    if ok and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    # C) WAV robusto
    ok, log_c = _concat_via_wav(aud_dir, out_path, total_chunks)
    return ok and os.path.exists(out_path) and os.path.getsize(out_path) > 0

# ---------------------------
# Sintesi di UN chunk riusando scripts.utils.generate_audio
# ---------------------------
def synthesize_single_chunk(text_chunk: str, runtime_cfg: dict, aud_dir: str, idx: int) -> str | None:
    """
    Genera un singolo chunk audio riusando generate_audio su una cartella temporanea,
    poi rinomina il risultato a chunk_{idx:04d}.mp3 nell'aud_dir.
    """
    tmp_out = os.path.join(aud_dir, f"_tmp_chunk_{idx:04d}")
    os.makedirs(tmp_out, exist_ok=True)
    try:
        result_path = generate_audio([text_chunk], runtime_cfg, tmp_out)
        if not result_path or not os.path.exists(result_path):
            return None
        dest = chunk_filename(aud_dir, idx)
        os.replace(result_path, dest)  # atomic move
        try:
            shutil.rmtree(tmp_out, ignore_errors=True)
        except Exception:
            pass
        return dest
    except Exception:
        try:
            shutil.rmtree(tmp_out, ignore_errors=True)
        except Exception:
            pass
        return None

# ---------------------------
# Generazione AUDIO con resume
# ---------------------------
def generate_audio_with_resume(script_text: str, runtime_cfg: dict, aud_dir: str, max_chars: int = 2000) -> str | None:
    """
    1) Crea i chunk del testo (~2000 char).
    2) Salva/legge un manifest per il resume.
    3) Genera ogni chunk come MP3 separato (chunk_0000.mp3, ...).
    4) Concatena tutto in combined_audio.mp3 (con fallback robusto).
    """
    os.makedirs(aud_dir, exist_ok=True)
    chunks = split_text_into_sentence_chunks(script_text, max_chars=max_chars)
    total = len(chunks)
    if total == 0:
        return None

    m = load_manifest(aud_dir) or {}
    cur_hash = script_hash(script_text)
    expected = {
        "version": 2,  # bump versione manifest per distinguere vecchi tentativi
        "script_hash": cur_hash,
        "max_chars": max_chars,
        "total_chunks": total,
        "completed": [],
    }

    if not m or (
        m.get("script_hash") != cur_hash or
        int(m.get("max_chars", -1)) != max_chars or
        int(m.get("total_chunks", -1)) != total
    ):
        m = expected
        save_manifest(aud_dir, m)
    else:
        m["completed"] = sorted(set(int(i) for i in m.get("completed", []) if 0 <= int(i) < total))
        save_manifest(aud_dir, m)

    completed = set(m.get("completed", []))
    # marca come completati i chunk mp3 già presenti
    for i in range(total):
        f = chunk_filename(aud_dir, i)
        if os.path.exists(f) and os.path.getsize(f) > 0:
            completed.add(i)
    m["completed"] = sorted(completed)
    save_manifest(aud_dir, m)

    st.caption(f"🎧 Audio: {total} chunk da generare (~{max_chars} caratteri ciascuno).")
    prog = st.progress(len(completed) / total if total else 0.0)
    status = st.empty()

    # genera solo i mancanti
    for i, piece in enumerate(chunks):
        if i in completed:
            status.write(f"✅ Chunk {i+1}/{total} già presente, salto.")
            prog.progress((len(completed)) / total)
            continue

        status.write(f"🎙️ Genero chunk {i+1}/{total} …")
        out = synthesize_single_chunk(piece, runtime_cfg, aud_dir, i)
        if out and os.path.exists(out):
            completed.add(i)
            m["completed"] = sorted(completed)
            save_manifest(aud_dir, m)
            prog.progress(len(completed) / total)
        else:
            st.error(f"❌ Errore nella generazione del chunk {i+1}. Puoi ripremere 'Genera contenuti' per riprendere.")
            return None

    status.write("🔗 Concateno i chunk MP3 in un unico file…")
    final_path = os.path.join(aud_dir, "combined_audio.mp3")
    ok = concat_mp3_chunks(aud_dir, final_path, total_chunks=total)
    if not ok:
        st.error("❌ Concatenazione fallita anche con fallback WAV. I chunk singoli sono comunque salvati.")
        return None

    try:
        dur = mp3_duration_seconds(final_path)
        st.caption(f"⏱️ Durata audio: ~{int(dur)}s")
    except Exception:
        pass

    status.write("✅ Audio completo!")
    return final_path

# ---------------------------
# Pagina
# ---------------------------
st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="centered")
st.title("🎬 Generatore di Video con Immagini e Audio")

# Carica config opzionale
base_cfg = {}
if load_config:
    try:
        loaded = load_config()
        if isinstance(loaded, dict):
            base_cfg = loaded
    except Exception:
        base_cfg = {}

# ===========================
# 🔐 & ⚙️ Sidebar: API + Parametri
# ===========================
with st.sidebar:
    st.header("🔐 API Keys")
    st.caption("Le chiavi valgono solo per *questa sessione* del browser.")

    rep_prefill = st.session_state.get("replicate_api_key", "")
    fish_prefill = st.session_state.get("fish_audio_api_key", "")

    with st.form("api_keys_form", clear_on_submit=False):
        replicate_key = st.text_input(
            "Replicate API key",
            type="password",
            value=rep_prefill,
            placeholder="r8_********",
            help="Necessaria per generare IMMAGINI (Replicate)"
        )
        fish_key = st.text_input(
            "FishAudio API key",
            type="password",
            value=fish_prefill,
            placeholder="fa_********",
            help="Necessaria per generare AUDIO (FishAudio)"
        )
        save_keys = st.form_submit_button("💾 Save")

    if save_keys:
        st.session_state["replicate_api_key"] = replicate_key.strip()
        st.session_state["fish_audio_api_key"] = fish_key.strip()
        st.success("Chiavi salvate nella sessione!")

    st.subheader("🔎 Verifica token Replicate")
    if st.button("Verifica token"):
        tok = _clean_token(st.session_state.get("replicate_api_key", ""))
        if not tok:
            st.error("Nessun token Replicate inserito.")
        else:
            try:
                r = requests.get(
                    "https://api.replicate.com/v1/account",
                    headers={"Authorization": f"Bearer {tok}"},
                    timeout=15,
                )
                if r.status_code == 200:
                    who = r.json()
                    st.success(f"✅ Token valido. Utente: {who.get('username','?')}")
                else:
                    st.error(f"❌ Token NON valido. HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                st.error(f"❌ Errore chiamando l’API: {e}")

    st.divider()
    st.header("⚙️ Parametri generazione")

    # FishAudio Voice ID
    voice_prefill = st.session_state.get("fishaudio_voice_id", "")
    fish_voice_id = st.text_input(
        "FishAudio Voice ID",
        value=voice_prefill,
        placeholder="es. voice_123abc...",
        help="ID della voce TTS da usare in FishAudio"
    )
    if fish_voice_id != voice_prefill:
        st.session_state["fishaudio_voice_id"] = fish_voice_id.strip()

    # Modello Replicate (preset + custom)
    model_presets = [
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-1.1",
        "stability-ai/sdxl",
        "scenario/anything-v4.5",
        "Custom (digita sotto)",
    ]
    preset_selected = st.selectbox(
        "Modello Replicate (image generator)",
        model_presets,
        index=0,
        help="Scegli un preset oppure 'Custom' e inserisci il nome esatto del modello sotto."
    )

    custom_prefill = st.session_state.get("replicate_model_custom", "")
    custom_model = st.text_input(
        "Custom model (owner/name:tag)",
        value=custom_prefill,
        placeholder="es. puccincolli/super-image:latest",
        help="Inserisci il nome completo del modello se usi 'Custom'"
    )
    if custom_model != custom_prefill:
        st.session_state["replicate_model_custom"] = custom_model.strip()

    effective_model = (
        st.session_state.get("replicate_model_custom", "").strip()
        if preset_selected == "Custom (digita sotto)"
        else preset_selected
    )
    st.session_state["replicate_model"] = effective_model

# Funzioni per recuperare stati
def get_replicate_key() -> str:
    return (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN", "")).strip()

def get_fishaudio_key() -> str:
    return (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY", "")).strip()

def get_fishaudio_voice_id() -> str:
    return (st.session_state.get("fishaudio_voice_id", "")).strip()

def get_replicate_model() -> str:
    return (st.session_state.get("replicate_model", "")).strip()

# Badge di stato
rep_ok = bool(get_replicate_key())
fish_ok = bool(get_fishaudio_key())
rep_model = get_replicate_model() or "—"
voice_id = get_fishaudio_voice_id() or "—"
st.write(
    f"🔎 Stato API → Replicate: {'✅' if rep_ok else '⚠️'} · FishAudio: {'✅' if fish_ok else '⚠️'} · "
    f"Model(Immagini): {rep_model} · VoiceID(Audio): {voice_id}"
)

# ===========================
# 🎛️ Parametri generazione (centrale)
# ===========================
title = st.text_input("Titolo del video")
script = st.text_area("Inserisci il testo da usare per generare immagini/audio", height=300)
mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio", "Entrambi"])

# Config chunking audio esposto per comodità
max_chars_audio = st.number_input(
    "Massimo caratteri per chunk audio",
    min_value=500, max_value=8000, value=2000, step=100,
    help="I chunk vengono spezzati dopo i punti dove possibile."
)

# Input condizionali
if mode in ["Audio", "Entrambi"]:
    seconds_per_img = st.number_input(
        "Ogni quanti secondi di audio creare un'immagine?",
        min_value=1, value=8, step=1
    )
else:  # Solo Immagini
    sentences_per_image = st.number_input(
        "Ogni quante frasi creare un'immagine?",
        min_value=1, value=2, step=1
    )

generate = st.button("🚀 Genera contenuti")

# ===========================
# 🚀 Avvio generazione
# ===========================
if generate and title.strip() and script.strip():
    safe = sanitize(title)
    base = os.path.join("data", "outputs", safe)
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    audio_path = os.path.join(aud_dir, "combined_audio.mp3")

    st.subheader("🔄 Generazione in corso…")

    # Config runtime passata ai metodi utils
    runtime_cfg = dict(base_cfg)  # copia

    # Inietta chiavi/parametri scelti dall'utente (puliti)
    replicate_from_ui = _clean_token(get_replicate_key())
    fishaudio_from_ui = _clean_token(get_fishaudio_key())

    if replicate_from_ui:
        os.environ["REPLICATE_API_TOKEN"] = replicate_from_ui
        runtime_cfg["replicate_api_key"] = replicate_from_ui
        runtime_cfg["replicate_api_token"] = replicate_from_ui  # compat
    if fishaudio_from_ui:
        os.environ["FISHAUDIO_API_KEY"] = fishaudio_from_ui
        runtime_cfg["fishaudio_api_key"] = fishaudio_from_ui

    # Parametri specifici
    replicate_model = get_replicate_model()
    if replicate_model:
        runtime_cfg["replicate_model"] = replicate_model  # usato in generate_images
    fish_voice = get_fishaudio_voice_id()
    if fish_voice:
        runtime_cfg["fishaudio_voice_id"] = fish_voice   # usato in generate_audio

    # Debug (token mascherato + modello)
    st.write(
        "🔐 Replicate token: "
        + _mask(runtime_cfg.get("replicate_api_key") or runtime_cfg.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN"))
        + " · Modello: `"
        + (runtime_cfg.get("replicate_model") or runtime_cfg.get("image_model") or "—")
        + "`"
    )

    # ---- AUDIO (con resume) ----
    if mode in ["Audio", "Entrambi"]:
        if not fish_ok:
            st.error("❌ FishAudio API key mancante. Inseriscila nella sidebar.")
        elif not get_fishaudio_voice_id():
            st.error("❌ FishAudio Voice ID mancante. Inseriscilo nella sidebar.")
        else:
            st.text(f"🎧 Generazione audio con voce: {get_fishaudio_voice_id()} …")
            final_audio = generate_audio_with_resume(script, runtime_cfg, aud_dir, max_chars=int(max_chars_audio))
            if final_audio:
                audio_path = final_audio
            else:
                st.error("⚠️ Audio non completato: premi di nuovo 'Genera contenuti' per riprovare la sola concatenazione.")
                st.stop()

    # ---- IMMAGINI ----
    if mode in ["Immagini", "Entrambi"]:
        if not rep_ok:
            st.error("❌ Replicate API key mancante. Inseriscila nella sidebar.")
        elif not get_replicate_model():
            st.error("❌ Modello Replicate mancante. Seleziona un preset o inserisci un Custom model.")
        else:
            if mode == "Entrambi":
                if not os.path.exists(audio_path):
                    st.error("❌ Audio non trovato per calcolare le immagini. Genera prima l’audio.")
                else:
                    st.text(f"🖼️ Generazione immagini con modello: {get_replicate_model()} (tempo audio)…")
                    try:
                        duration_sec = mp3_duration_seconds(audio_path)
                    except Exception:
                        duration_sec = 0
                    if not duration_sec:
                        duration_sec = 60  # fallback
                    num_images = max(1, int(duration_sec // seconds_per_img))
                    approx_chars = max(1, len(script) // max(1, num_images))
                    img_chunks = chunk_text(script, approx_chars)
                    st.text(f"🖼️ Generazione di {len(img_chunks)} immagini…")
                    generate_images(img_chunks, runtime_cfg, img_dir)
                    zip_images(base)
            else:
                st.text(f"🖼️ Generazione immagini con modello: {get_replicate_model()} (per frasi)…")
                groups = chunk_by_sentences_count(script, int(sentences_per_image))
                st.text(f"🖼️ Generazione di {len(groups)} immagini (1 ogni {int(sentences_per_image)} frasi)…")
                generate_images(groups, runtime_cfg, img_dir)
                zip_images(base)

    st.success("✅ Generazione completata!")

    # salva percorsi in sessione per i download
    st.session_state["audio_path"] = audio_path if os.path.exists(audio_path) else None
    zip_path = os.path.join(base, "output.zip")
    st.session_state["zip_path"] = zip_path if os.path.exists(zip_path) else None

# ---- Download (chiavi uniche per evitare DuplicateElementId) ----
if st.session_state.get("audio_path") and os.path.exists(st.session_state["audio_path"]):
    with open(st.session_state["audio_path"], "rb") as f:
        st.download_button("🎧 Scarica Audio MP3", f, file_name="audio.mp3", mime="audio/mpeg", key="dl-audio")

if st.session_state.get("zip_path") and os.path.exists(st.session_state["zip_path"]):
    with open(st.session_state["zip_path"], "rb") as f:
        st.download_button("🖼️ Scarica ZIP Immagini", f, file_name="output.zip", mime="application/zip", key="dl-zip")
