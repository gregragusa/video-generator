# app.py
# -------------------------------------------------------
# Generatore di Video con Immagini e Audio (Streamlit)
# - Resume AUDIO con manifest + chunk numerati
# - Concat MP3 robusta (WAV fallback) + validazione chunk
# - Risoluzione automatica modello Replicate -> owner/name:version_id
# - Download audio immediato appena pronto
# Compatibile con Python >=3.10
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
from typing import Optional, Tuple, Dict, Any, List

# se hai questo loader lo usiamo, altrimenti proseguiamo senza
try:
    from scripts.config_loader import load_config  # opzionale
except Exception:
    load_config = None

from scripts.utils import (
    chunk_text,
    chunk_by_sentences_count,
    generate_audio,        # riusato per generare un singolo chunk alla volta
    generate_images,
    mp3_duration_seconds,  # util per leggere durata MP3
)

# imageio-ffmpeg per trovare ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"  # fallback: deve essere nel PATH

FFPROBE_EXE = os.environ.get("FFPROBE_EXE", "ffprobe")  # usa ffprobe di sistema

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
def split_text_into_sentence_chunks(text: str, max_chars: int = 2000) -> List[str]:
    """
    Divide il testo in blocchi di circa max_chars caratteri,
    spezzando dopo . ! ? dove possibile. Se una singola frase supera max_chars, la spezza duramente.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []

    sentences = re.split(r"(?<=[\.\!\?])\s+", t)
    chunks: List[str] = []
    acc = ""

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
# Manifest
# ---------------------------
def manifest_path(aud_dir: str) -> str:
    return os.path.join(aud_dir, "audio_manifest.json")

def load_manifest(aud_dir: str) -> Optional[Dict[str, Any]]:
    p = manifest_path(aud_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_manifest(aud_dir: str, data: Dict[str, Any]) -> None:
    p = manifest_path(aud_dir)
    tmp = p + ".tmp"
    os.makedirs(aud_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def chunk_filename(aud_dir: str, idx: int) -> str:
    return os.path.join(aud_dir, f"chunk_{idx:04d}.mp3")

# ---------------------------
# FFmpeg helpers
# ---------------------------
def _run(cmd: List[str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = (p.stderr.decode("utf-8", "ignore") or p.stdout.decode("utf-8", "ignore"))
        return True, out
    except subprocess.CalledProcessError as e:
        msg = (e.stderr.decode("utf-8", "ignore") if e.stderr else str(e))
        return False, msg

def _ffconcat_escape(path: str) -> str:
    """Escapa backslash e apici singoli per il file lista di ffmpeg (-f concat)."""
    return path.replace("\\", "\\\\").replace("'", "\\'")

def ffprobe_has_audio(path: str) -> bool:
    """True se il file ha almeno uno stream audio. Se ffprobe non esiste, fallback su size>0."""
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    # prova ffprobe
    try:
        cmd = [
            FFPROBE_EXE, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=nk=1:nw=1",
            path
        ]
        ok, out = _run(cmd)
        if not ok:
            return False
        return any(line.strip() == "audio" for line in out.splitlines())
    except Exception:
        return os.path.getsize(path) > 0

def mp3_to_wav(src_mp3: str, dst_wav: str) -> Tuple[bool, str]:
    """Ricodifica in WAV coerente (stereo, 44.1kHz, s16)."""
    cmd = [FFMPEG_EXE, "-y", "-i", src_mp3, "-vn", "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", dst_wav]
    return _run(cmd)

def concat_wavs_demuxer(wav_paths: List[str], combined_wav: str) -> Tuple[bool, str]:
    """Concatena WAV via demuxer -f concat -c copy."""
    tmp_list = combined_wav + ".list.txt"
    with open(tmp_list, "w", encoding="utf-8") as f:
        for p in wav_paths:
            f.write(f"file '{_ffconcat_escape(p)}'\n")
    cmd = [FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", tmp_list, "-c", "copy", combined_wav]
    ok, log = _run(cmd)
    try:
        os.remove(tmp_list)
    except Exception:
        pass
    return ok, log

def concat_wavs_filter_complex_to_mp3(wav_paths: List[str], out_mp3: str) -> Tuple[bool, str]:
    """Concatena WAV usando filter_complex concat e produce MP3 finale."""
    if not wav_paths:
        return False, "No WAV inputs"
    cmd = [FFMPEG_EXE, "-y"]
    for p in wav_paths:
        cmd += ["-i", p]
    filter_str = f"concat=n={len(wav_paths)}:v=0:a=1"
    cmd_lame = cmd + ["-filter_complex", filter_str, "-vn", "-c:a", "libmp3lame", "-q:a", "2", out_mp3]
    ok, log = _run(cmd_lame)
    if ok:
        return True, log
    cmd_mp3 = cmd + ["-filter_complex", filter_str, "-vn", "-c:a", "mp3", "-q:a", "2", out_mp3]
    return _run(cmd_mp3)

# ---------------------------
# Sintesi di UN chunk riusando scripts.utils.generate_audio
# ---------------------------
def synthesize_single_chunk(text_chunk: str, runtime_cfg: Dict[str, Any], aud_dir: str, idx: int) -> Optional[str]:
    """
    Genera un singolo chunk audio riusando generate_audio su una cartella temporanea,
    poi rinomina il risultato a chunk_{idx:04d}.mp3 nell'aud_dir.
    """
    tmp_out = os.path.join(aud_dir, f"_tmp_chunk_{idx:04d}")
    os.makedirs(tmp_out, exist_ok=True)
    try:
        result_path = generate_audio([text_chunk], runtime_cfg, tmp_out)
        if not result_path or not os.path.exists(result_path) or os.path.getsize(result_path) <= 0:
            shutil.rmtree(tmp_out, ignore_errors=True)
            return None
        dest = chunk_filename(aud_dir, idx)
        os.replace(result_path, dest)  # atomic move
        shutil.rmtree(tmp_out, ignore_errors=True)
        return dest
    except Exception:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

# ---------------------------
# Concat MP3 (robusta, diagnostica)
# ---------------------------
def concat_mp3_chunks_robust(script_chunks: List[str], runtime_cfg: Dict[str, Any], aud_dir: str) -> Tuple[bool, str]:
    """
    1) Convalida ogni chunk mp3 con ffprobe.
    2) Ricodifica tutti i chunk validi in WAV coerenti (rigenera una volta i chunk difettosi).
    3) Concat demuxer WAV -> combined.wav (copy), oppure filter_complex -> MP3 diretto.
    Ritorna (ok, path_mp3) oppure (False, errore).
    """
    total_chunks = len(script_chunks)
    if total_chunks == 0:
        return False, "Nessun chunk"

    # 1) validazione + autocorrezione base
    invalid = []
    for i in range(total_chunks):
        f = chunk_filename(aud_dir, i)
        if not ffprobe_has_audio(f):
            invalid.append(i)

    if invalid:
        st.warning(f"🔁 Rilevati {len(invalid)} chunk MP3 non validi: {invalid[:10]}{'…' if len(invalid)>10 else ''}. Rigenero…")
        for i in invalid:
            try:
                os.remove(chunk_filename(aud_dir, i))
            except Exception:
                pass
            out = synthesize_single_chunk(script_chunks[i], runtime_cfg, aud_dir, i)
            if not (out and ffprobe_has_audio(out)):
                return False, f"Chunk {i} non rigenerabile (MP3 invalido dopo rigenerazione)."

    # 2) normalizza in WAV
    tmp_dir = os.path.join(aud_dir, "_wav_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    wav_paths: List[str] = []
    bad_after_wav: List[Tuple[int, str]] = []
    for i in range(total_chunks):
        src = chunk_filename(aud_dir, i)
        wav = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        ok, _ = mp3_to_wav(src, wav)
        if not ok or (not os.path.exists(wav)) or os.path.getsize(wav) <= 0:
            out = synthesize_single_chunk(script_chunks[i], runtime_cfg, aud_dir, i)
            if not (out and ffprobe_has_audio(out)):
                bad_after_wav.append((i, "rigenerazione MP3 fallita"))
                continue
            ok2, _ = mp3_to_wav(out, wav)
            if not ok2 or (not os.path.exists(wav)) or os.path.getsize(wav) <= 0:
                bad_after_wav.append((i, "transcode WAV fallita dopo rigenerazione"))
                continue
        wav_paths.append(wav)

    if bad_after_wav:
        log_path = os.path.join(aud_dir, "_concat_error.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Chunk problematici (fase WAV):\n")
            for idx, why in bad_after_wav:
                f.write(f"- chunk {idx}: {why}\n")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, f"Alcuni chunk non sono convertibili in WAV: {bad_after_wav}. Vedi log {log_path}"

    # 3) concatena i WAV
    combined_wav = os.path.join(tmp_dir, "combined.wav")
    ok, log = concat_wavs_demuxer(wav_paths, combined_wav)
    out_mp3 = os.path.join(aud_dir, "combined_audio.mp3")

    if ok and os.path.exists(combined_wav) and os.path.getsize(combined_wav) > 0:
        ok2, log2 = _run([FFMPEG_EXE, "-y", "-i", combined_wav, "-vn", "-c:a", "libmp3lame", "-q:a", "2", out_mp3])
        if not ok2:
            ok3, log3 = _run([FFMPEG_EXE, "-y", "-i", combined_wav, "-vn", "-c:a", "mp3", "-q:a", "2", out_mp3])
            if not ok3:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False, f"Encoding MP3 dal WAV fallita: {log2}\n{log3}"
    else:
        ok_fc, log_fc = concat_wavs_filter_complex_to_mp3(wav_paths, out_mp3)
        if not ok_fc:
            log_path = os.path.join(aud_dir, "_concat_error.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("Concat fallita.\n")
                f.write("Demuxer log:\n")
                f.write((log or "") + "\n")
                f.write("Filter_complex log:\n")
                f.write((log_fc or "") + "\n")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, f"Concat fallita (demuxer e filter_complex). Log in {log_path}"

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
        return True, out_mp3
    return False, "MP3 finale assente o vuoto dopo concat"

# ---------------------------
# Generazione AUDIO con resume + validazione
# ---------------------------
def generate_audio_with_resume(script_text: str, runtime_cfg: Dict[str, Any], aud_dir: str, max_chars: int = 2000) -> Optional[str]:
    """
    1) Crea i chunk del testo (~2000 char).
    2) Manifest per resume (completed = solo chunk con stream audio valido).
    3) Genera i mancanti.
    4) Concat ultra-robusta con rigenerazione automatica dei chunk difettosi.
    """
    os.makedirs(aud_dir, exist_ok=True)
    chunks = split_text_into_sentence_chunks(script_text, max_chars=max_chars)
    total = len(chunks)
    if total == 0:
        return None

    m = load_manifest(aud_dir) or {}
    cur_hash = script_hash(script_text)
    expected = {
        "version": 3,  # bump
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

    completed = set()
    for i in range(total):
        f = chunk_filename(aud_dir, i)
        if ffprobe_has_audio(f):
            completed.add(i)
    m["completed"] = sorted(completed)
    save_manifest(aud_dir, m)

    st.caption(f"🎧 Audio: {total} chunk da generare (~{max_chars} caratteri ciascuno).")
    prog = st.progress(len(completed) / total if total else 0.0)
    status = st.empty()

    for i, piece in enumerate(chunks):
        if i in completed:
            status.write(f"✅ Chunk {i+1}/{total} ok, salto.")
            prog.progress((len(completed)) / total)
            continue

        status.write(f"🎙️ Genero chunk {i+1}/{total} …")
        out = synthesize_single_chunk(piece, runtime_cfg, aud_dir, i)
        if out and ffprobe_has_audio(out):
            completed.add(i)
            m["completed"] = sorted(completed)
            save_manifest(aud_dir, m)
            prog.progress(len(completed) / total)
        else:
            status.write(f"❌ Chunk {i+1} non valido. Ripremi 'Genera contenuti' per riprovare.")
            return None

    status.write("🔗 Concateno i chunk MP3…")
    ok, result = concat_mp3_chunks_robust(chunks, runtime_cfg, aud_dir)
    if not ok:
        st.error(f"❌ {result}")
        return None

    final_path = result
    try:
        dur = mp3_duration_seconds(final_path)
        st.caption(f"⏱️ Durata audio: ~{int(dur)}s")
    except Exception:
        pass
    status.write("✅ Audio completo!")
    return final_path

# ---------------------------
# Risoluzione modello Replicate -> owner/name:version_id
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def resolve_replicate_model_identifier(model_input: str, token: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Accetta:
      - "owner/name"
      - "owner/name:version_or_tag"
    Se manca la parte dopo ":", interroga l'API Replicate per ottenere default_version (o latest_version). Ritorna:
      ( "owner/name:version_id", None )  in caso di successo
      ( None, "messaggio di errore" )    in caso di errore
    """
    mi = (model_input or "").strip()
    if not mi:
        return None, "Modello vuoto."
    # se già include ":", si presume risolto
    if ":" in mi:
        parts = mi.split(":")
        if len(parts) == 2 and parts[0].count("/") == 1 and parts[1]:
            return mi, None
        # formato strano
        return None, f"Formato modello non valido: {mi}"

    # deve essere owner/name
    if mi.count("/") != 1:
        return None, f"Atteso 'owner/name' (facoltativo ':version'), ricevuto: {mi}"

    if not token:
        return None, "Replicate API token assente: impossibile risolvere la versione."

    owner, name = mi.split("/", 1)
    url = f"https://api.replicate.com/v1/models/{owner}/{name}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code == 200:
            data = r.json() or {}
            # Replicate in alcuni casi espone default_version, in altri latest_version
            ver = (data.get("default_version") or data.get("latest_version") or {}).get("id")
            if not ver:
                return None, f"Modello trovato ma senza version id: {mi}"
            return f"{owner}/{name}:{ver}", None
        elif r.status_code in (401, 403):
            return None, "Token non autorizzato a leggere il modello (401/403)."
        elif r.status_code == 404:
            return None, "Modello inesistente (404)."
        else:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"Errore chiamando Replicate: {e}"

# ---------------------------
# Pagina
# ---------------------------
st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="centered")
st.title("🎬 Generatore di Video con Immagini e Audio")

# Carica config opzionale
base_cfg: Dict[str, Any] = {}
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
        "Custom model (owner/name:tag oppure owner/name)",
        value=custom_prefill,
        placeholder="es. black-forest-labs/flux-1.1  (verrà risolto a owner/name:version_id)",
        help="Se ometti ':version', risolvo automaticamente la versione di default."
    )
    if custom_model != custom_prefill:
        st.session_state["replicate_model_custom"] = custom_model.strip()

    effective_model = (
        st.session_state.get("replicate_model_custom", "").strip()
        if preset_selected == "Custom (digita sotto)"
        else preset_selected
    )
    st.session_state["replicate_model"] = effective_model

    # Verifica modello Replicate (risolve :version)
    if st.button("Verifica modello Replicate"):
        tok = _clean_token(st.session_state.get("replicate_api_key", ""))
        resolved, err = resolve_replicate_model_identifier(effective_model, tok)
        if resolved:
            st.success(f"✅ Modello utilizzabile: `{resolved}`")
        else:
            st.error(f"❌ Modello non utilizzabile: {err}")

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
    runtime_cfg: Dict[str, Any] = dict(base_cfg)

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

    # Parametri specifici (IMMAGINI): risolviamo il modello -> owner/name:version_id
    effective = get_replicate_model()
    if effective:
        resolved_model, err = resolve_replicate_model_identifier(effective, replicate_from_ui)
        if resolved_model:
            runtime_cfg["replicate_model"] = resolved_model  # usato da generate_images
            runtime_cfg["replicate_model_resolved"] = resolved_model
            if ":" not in effective:
                st.caption(f"🔁 Modello risolto automaticamente: `{resolved_model}`")
        else:
            # Non blocco: passo comunque il testo originale, ma avviso (così vedi l'errore originale se persistente)
            runtime_cfg["replicate_model"] = effective
            st.warning(f"⚠️ Modello non risolto: {err}")

    # Parametri specifici (AUDIO)
    fish_voice = get_fishaudio_voice_id()
    if fish_voice:
        runtime_cfg["fishaudio_voice_id"] = fish_voice

    # Debug (token mascherato + modello)
    st.write(
        "🔐 Replicate token: "
        + _mask(runtime_cfg.get("replicate_api_key") or runtime_cfg.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN"))
        + " · Modello risolto: `"
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
                # ✅ Download immediato appena l'audio è pronto
                st.success("🎉 Audio pronto. Puoi scaricarlo subito mentre genero (eventualmente) le immagini.")
                st.session_state["audio_path"] = audio_path
                try:
                    with open(audio_path, "rb") as f:
                        st.download_button("🎧 Scarica Audio MP3 (subito)", f, file_name="audio.mp3",
                                           mime="audio/mpeg", key="dl-audio-early")
                except Exception:
                    pass
            else:
                st.error("⚠️ Audio non completato. Premi di nuovo 'Genera contenuti' per rigenerare eventuali chunk rotti e riprovare la concat.")
                st.stop()

    # ---- IMMAGINI ----
    if mode in ["Immagini", "Entrambi"]:
        if not rep_ok:
            st.error("❌ Replicate API key mancante. Inseriscila nella sidebar.")
        elif not runtime_cfg.get("replicate_model"):
            st.error("❌ Modello Replicate mancante o non risolto.")
        else:
            if mode == "Entrambi":
                if not os.path.exists(audio_path):
                    st.error("❌ Audio non trovato per calcolare le immagini. Genera prima l’audio.")
                else:
                    st.text(f"🖼️ Generazione immagini con modello: {runtime_cfg.get('replicate_model')} (tempo audio)…")
                    try:
                        duration_sec = mp3_duration_seconds(audio_path)
                    except Exception:
                        duration_sec = 0
                    if not duration_sec:
                        duration_sec = 60  # fallback
                    seconds_per_img = 8  # valore di default per 'Entrambi' se non impostato da UI precedente
                    # Nel vecchio UI questo veniva letto prima; qui fissiamo 8 se non esiste in sessione
                    try:
                        seconds_per_img = int(st.session_state.get("seconds_per_img", 8))
                    except Exception:
                        pass
                    num_images = max(1, int(duration_sec // max(1, seconds_per_img)))
                    approx_chars = max(1, len(script) // max(1, num_images))
                    img_chunks = chunk_text(script, approx_chars)
                    st.text(f"🖼️ Generazione di {len(img_chunks)} immagini…")
                    try:
                        generate_images(img_chunks, runtime_cfg, img_dir)
                        zip_images(base)
                    except Exception as e:
                        st.error(f"❌ Errore generazione immagini: {e}")
            else:
                st.text(f"🖼️ Generazione immagini con modello: {runtime_cfg.get('replicate_model')} (per frasi)…")
                # default 2 frasi per immagine, se non definito precedentemente
                sentences_per_image = int(st.session_state.get("sentences_per_image", 2)) or 2
                groups = chunk_by_sentences_count(script, sentences_per_image)
                st.text(f"🖼️ Generazione di {len(groups)} immagini (1 ogni {int(sentences_per_image)} frasi)…")
                try:
                    generate_images(groups, runtime_cfg, img_dir)
                    zip_images(base)
                except Exception as e:
                    st.error(f"❌ Errore generazione immagini: {e}")

    st.success("✅ Generazione completata!")

    # salva percorsi in sessione per i download (anche finali)
    st.session_state["audio_path"] = audio_path if os.path.exists(audio_path) else st.session_state.get("audio_path")
    zip_path = os.path.join(base, "output.zip")
    st.session_state["zip_path"] = zip_path if os.path.exists(zip_path) else None

# ---- Download (chiavi uniche per evitare DuplicateElementId) ----
if st.session_state.get("audio_path") and os.path.exists(st.session_state["audio_path"]):
    try:
        with open(st.session_state["audio_path"], "rb") as f:
            st.download_button("🎧 Scarica Audio MP3", f, file_name="audio.mp3", mime="audio/mpeg", key="dl-audio-final")
    except Exception:
        pass

if st.session_state.get("zip_path") and os.path.exists(st.session_state["zip_path"]):
    try:
        with open(st.session_state["zip_path"], "rb") as f:
            st.download_button("🖼️ Scarica ZIP Immagini", f, file_name="output.zip", mime="application/zip", key="dl-zip")
    except Exception:
        pass
