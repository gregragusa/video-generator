# app.py
# -------------------------------------------------------
# Generatore di Video con Immagini e Audio (Streamlit)
# - Resume AUDIO + concat robusta (ffmpeg/ffprobe)
# - Resume IMMAGINI + prompt fisso (before/after)
# - Download MP3 immediato
# - Risoluzione modello Replicate -> owner/name:version_id
# - 🔧 Compat shims per vecchie versioni di Streamlit (cache_data, radio horizontal)
# - 🛡️ "Gabbia di sicurezza": l'app non crasha all'avvio, mostra lo stacktrace in pagina
# -------------------------------------------------------

import os
import re
import json
import hashlib
import shutil
import textwrap
import subprocess
import requests
import traceback
import inspect
import streamlit as st
from typing import Optional, Tuple, Dict, Any, List

# ============== Streamlit compatibility shims ==============
# cache_data -> fallback a experimental_memo o no-op
try:
    cache_data = st.cache_data  # Streamlit >= 1.18
except AttributeError:
    try:
        cache_data = st.experimental_memo  # vecchie versioni
    except AttributeError:
        def cache_data(*args, **kwargs):
            def decorator(fn):  # no-op
                return fn
            return decorator

# radio(horizontal=...) compat
_SIG_RADIO = None
try:
    _SIG_RADIO = inspect.signature(st.radio)
except Exception:
    _SIG_RADIO = None
HAS_RADIO_HORIZONTAL = _SIG_RADIO and ("horizontal" in _SIG_RADIO.parameters)

def radio_compat(label: str, options: List[str], index: int = 0, horizontal_default: bool = False):
    if HAS_RADIO_HORIZONTAL:
        return st.radio(label, options=options, index=index, horizontal=horizontal_default)
    return st.radio(label, options=options, index=index)

# ============== Opzionale config loader ==============
try:
    from scripts.config_loader import load_config  # opzionale
except Exception:
    load_config = None

# ============== utils import con guardia ==============
# Definiamo stub sicuri in caso di import fallito: l'app si apre comunque.
chunk_text = None
chunk_by_sentences_count = None
generate_audio = None
generate_images = None
mp3_duration_seconds = None
_utils_import_error: Optional[str] = None

try:
    from scripts.utils import (
        chunk_text as _chunk_text,
        chunk_by_sentences_count as _chunk_by_sentences_count,
        generate_audio as _generate_audio,
        generate_images as _generate_images,
        mp3_duration_seconds as _mp3_duration_seconds,
    )
    chunk_text = _chunk_text
    chunk_by_sentences_count = _chunk_by_sentences_count
    generate_audio = _generate_audio
    generate_images = _generate_images
    mp3_duration_seconds = _mp3_duration_seconds
except Exception as e:
    _utils_import_error = f"Errore import scripts.utils: {e}\n{traceback.format_exc()}"

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
            flush_acc(); acc = s
    flush_acc()
    return [c for c in chunks if c]

# ---------------------------
# Manifest path helpers
# ---------------------------
def manifest_path_audio(aud_dir: str) -> str:
    return os.path.join(aud_dir, "audio_manifest.json")

def manifest_path_images(img_dir: str) -> str:
    return os.path.join(img_dir, "images_manifest.json")

def load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_manifest(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------------------------
# AUDIO helpers
# ---------------------------
def chunk_filename(aud_dir: str, idx: int) -> str:
    return os.path.join(aud_dir, f"chunk_{idx:04d}.mp3")

def _run(cmd: List[str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = (p.stderr.decode("utf-8", "ignore") or p.stdout.decode("utf-8", "ignore"))
        return True, out
    except subprocess.CalledProcessError as e:
        msg = (e.stderr.decode("utf-8", "ignore") if e.stderr else str(e))
        return False, msg

def _ffconcat_escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace("'", "\\'")

def ffprobe_has_audio(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        cmd = [FFPROBE_EXE, "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=codec_type", "-of", "default=nk=1:nw=1", path]
        ok, out = _run(cmd)
        if not ok:
            return False
        return any(line.strip() == "audio" for line in out.splitlines())
    except Exception:
        return os.path.getsize(path) > 0

def mp3_to_wav(src_mp3: str, dst_wav: str) -> Tuple[bool, str]:
    cmd = [FFMPEG_EXE, "-y", "-i", src_mp3, "-vn", "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", dst_wav]
    return _run(cmd)

def concat_wavs_demuxer(wav_paths: List[str], combined_wav: str) -> Tuple[bool, str]:
    tmp_list = combined_wav + ".list.txt"
    with open(tmp_list, "w", encoding="utf-8") as f:
        for p in wav_paths:
            f.write(f"file '{_ffconcat_escape(p)}'\n")
    cmd = [FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0", "-i", tmp_list, "-c", "copy", combined_wav]
    ok, log = _run(cmd)
    try: os.remove(tmp_list)
    except Exception: pass
    return ok, log

def concat_wavs_filter_complex_to_mp3(wav_paths: List[str], out_mp3: str) -> Tuple[bool, str]:
    if not wav_paths:
        return False, "No WAV inputs"
    cmd = [FFMPEG_EXE, "-y"]
    for p in wav_paths:
        cmd += ["-i", p]
    filter_str = f"concat=n={len(wav_paths)}:v=0:a=1"
    cmd_lame = cmd + ["-filter_complex", filter_str, "-vn", "-c:a", "libmp3lame", "-q:a", "2", out_mp3]
    ok, log = _run(cmd_lame)
    if ok: return True, log
    cmd_mp3 = cmd + ["-filter_complex", filter_str, "-vn", "-c:a", "mp3", "-q:a", "2", out_mp3]
    return _run(cmd_mp3)

def synthesize_single_chunk(text_chunk: str, runtime_cfg: Dict[str, Any], aud_dir: str, idx: int) -> Optional[str]:
    if generate_audio is None:
        raise RuntimeError(_utils_import_error or "generate_audio non disponibile.")
    tmp_out = os.path.join(aud_dir, f"_tmp_chunk_{idx:04d}")
    os.makedirs(tmp_out, exist_ok=True)
    try:
        result_path = generate_audio([text_chunk], runtime_cfg, tmp_out)
        if not result_path or not os.path.exists(result_path) or os.path.getsize(result_path) <= 0:
            shutil.rmtree(tmp_out, ignore_errors=True); return None
        dest = chunk_filename(aud_dir, idx)
        os.replace(result_path, dest)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return dest
    except Exception:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

def concat_mp3_chunks_robust(script_chunks: List[str], runtime_cfg: Dict[str, Any], aud_dir: str) -> Tuple[bool, str]:
    total_chunks = len(script_chunks)
    if total_chunks == 0:
        return False, "Nessun chunk"
    invalid = []
    for i in range(total_chunks):
        if not ffprobe_has_audio(chunk_filename(aud_dir, i)):
            invalid.append(i)
    if invalid:
        st.warning(f"🔁 Chunk MP3 non validi: {invalid[:10]}{'…' if len(invalid)>10 else ''}. Rigenero…")
        for i in invalid:
            try: os.remove(chunk_filename(aud_dir, i))
            except Exception: pass
            out = synthesize_single_chunk(script_chunks[i], runtime_cfg, aud_dir, i)
            if not (out and ffprobe_has_audio(out)):
                return False, f"Chunk {i} non rigenerabile."

    tmp_dir = os.path.join(aud_dir, "_wav_tmp"); os.makedirs(tmp_dir, exist_ok=True)
    wav_paths: List[str] = []; bad_after_wav: List[Tuple[int, str]] = []
    for i in range(total_chunks):
        src = chunk_filename(aud_dir, i); wav = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        ok, _ = mp3_to_wav(src, wav)
        if not ok or (not os.path.exists(wav)) or os.path.getsize(wav) <= 0:
            out = synthesize_single_chunk(script_chunks[i], runtime_cfg, aud_dir, i)
            if not (out and ffprobe_has_audio(out)):
                bad_after_wav.append((i, "rigenerazione MP3 fallita")); continue
            ok2, _ = mp3_to_wav(out, wav)
            if not ok2 or (not os.path.exists(wav)) or os.path.getsize(wav) <= 0:
                bad_after_wav.append((i, "transcode WAV fallita")); continue
        wav_paths.append(wav)

    if bad_after_wav:
        log_path = os.path.join(aud_dir, "_concat_error.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Chunk problematici (fase WAV):\n")
            for idx, why in bad_after_wav: f.write(f"- chunk {idx}: {why}\n")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False, f"Alcuni chunk non convertibili in WAV: {bad_after_wav}. Log {log_path}"

    combined_wav = os.path.join(tmp_dir, "combined.wav")
    ok, log = concat_wavs_demuxer(wav_paths, combined_wav)
    out_mp3 = os.path.join(aud_dir, "combined_audio.mp3")
    if ok and os.path.exists(combined_wav) and os.path.getsize(combined_wav) > 0:
        ok2, log2 = _run([FFMPEG_EXE, "-y", "-i", combined_wav, "-vn", "-c:a", "libmp3lame", "-q:a", "2", out_mp3])
        if not ok2:
            ok3, log3 = _run([FFMPEG_EXE, "-y", "-i", combined_wav, "-vn", "-c:a", "mp3", "-q:a", "2", out_mp3])
            if not ok3:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False, f"Encoding MP3 fallita: {log2}\n{log3}"
    else:
        ok_fc, log_fc = concat_wavs_filter_complex_to_mp3(wav_paths, out_mp3)
        if not ok_fc:
            log_path = os.path.join(aud_dir, "_concat_error.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("Concat fallita.\nDemuxer:\n"); f.write((log or "")+"\n")
                f.write("Filter_complex:\n"); f.write((log_fc or "")+"\n")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False, f"Concat fallita (demuxer e filter_complex). Log {log_path}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 0:
        return True, out_mp3
    return False, "MP3 finale assente o vuoto"

def generate_audio_with_resume(script_text: str, runtime_cfg: Dict[str, Any], aud_dir: str, max_chars: int = 2000) -> Optional[str]:
    chunks = split_text_into_sentence_chunks(script_text, max_chars=max_chars)
    total = len(chunks)
    if total == 0: return None
    m_path = manifest_path_audio(aud_dir)
    m = load_manifest(m_path) or {}
    cur_hash = script_hash(script_text)
    expected = {"version": 3, "script_hash": cur_hash, "max_chars": max_chars, "total_chunks": total, "completed": []}
    if (not m or m.get("script_hash") != cur_hash or int(m.get("max_chars", -1)) != max_chars or int(m.get("total_chunks", -1)) != total):
        m = expected; save_manifest(m_path, m)
    else:
        m["completed"] = sorted(set(int(i) for i in m.get("completed", []) if 0 <= int(i) < total)); save_manifest(m_path, m)

    completed = set()
    for i in range(total):
        if ffprobe_has_audio(chunk_filename(aud_dir, i)): completed.add(i)
    m["completed"] = sorted(completed); save_manifest(m_path, m)

    st.caption(f"🎧 Audio: {total} chunk da generare (~{max_chars} char).")
    prog = st.progress(len(completed)/total if total else 0.0); status = st.empty()
    for i, piece in enumerate(chunks):
        if i in completed:
            status.write(f"✅ Chunk {i+1}/{total} ok, salto."); prog.progress(len(completed)/total); continue
        status.write(f"🎙️ Genero chunk {i+1}/{total} …")
        out = synthesize_single_chunk(piece, runtime_cfg, aud_dir, i)
        if out and ffprobe_has_audio(out):
            completed.add(i); m["completed"] = sorted(completed); save_manifest(m_path, m)
            prog.progress(len(completed)/total)
        else:
            status.write(f"❌ Chunk {i+1} non valido. Ripremi 'Genera contenuti'."); return None

    status.write("🔗 Concateno i chunk MP3…")
    ok, result = concat_mp3_chunks_robust(chunks, runtime_cfg, aud_dir)
    if not ok: st.error(f"❌ {result}"); return None
    return result

# ---------------------------
# IMMAGINI helpers + resume + prompt fisso
# ---------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def _find_first_image_file(root: str) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                full = os.path.join(dirpath, fn)
                try: mtime = os.path.getmtime(full)
                except Exception: mtime = 0.0
                candidates.append((mtime, full))
    if not candidates: return None
    candidates.sort(reverse=True); return candidates[0][1]

def _image_existing_path(img_dir: str, idx: int) -> Optional[str]:
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, f"img_{idx:04d}{ext}")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None

def combined_image_prompt(content: str, fixed: str, position: str) -> str:
    fixed = (fixed or "").strip(); content = (content or "").strip()
    if not fixed: return content
    return (f"{fixed}\n\n{content}" if (position or "after").lower().startswith("before") else f"{content}\n\n{fixed}").strip()

def synthesize_single_image(text_chunk: str, runtime_cfg: Dict[str, Any], img_dir: str, idx: int) -> Optional[str]:
    if generate_images is None:
        raise RuntimeError(_utils_import_error or "generate_images non disponibile.")
    tmp_out = os.path.join(img_dir, f"_tmp_img_{idx:04d}")
    os.makedirs(tmp_out, exist_ok=True)
    try:
        generate_images([text_chunk], runtime_cfg, tmp_out)
        produced = _find_first_image_file(tmp_out)
        if not produced:
            shutil.rmtree(tmp_out, ignore_errors=True); return None
        ext = os.path.splitext(produced)[1].lower()
        if ext not in IMG_EXTS: ext = ".png"
        dest = os.path.join(img_dir, f"img_{idx:04d}{ext}")
        os.replace(produced, dest)
        shutil.rmtree(tmp_out, ignore_errors=True)
        return dest
    except Exception:
        shutil.rmtree(tmp_out, ignore_errors=True)
        return None

def generate_images_with_resume(img_chunks: List[str], img_dir: str, resume_key: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> bool:
    total = len(img_chunks)
    if total == 0: return True
    m_path = manifest_path_images(img_dir)
    m = load_manifest(m_path) or {}
    expected = {
        "version": 2,
        "script_hash": resume_key.get("script_hash"),
        "strategy": resume_key.get("strategy"),
        "param_value": int(resume_key.get("param_value", 0)),
        "total_images": total,
        "fixed_prompt_hash": resume_key.get("fixed_prompt_hash") or "",
        "fixed_prompt_position": resume_key.get("fixed_prompt_position") or "after",
        "completed": [],
    }
    if (not m or m.get("script_hash") != expected["script_hash"] or m.get("strategy") != expected["strategy"] or
        (m.get("fixed_prompt_hash") or "") != expected["fixed_prompt_hash"] or
        (m.get("fixed_prompt_position") or "after") != expected["fixed_prompt_position"] or
        int(m.get("param_value", -1)) != expected["param_value"] or int(m.get("total_images", -1)) != total):
        m = expected; save_manifest(m_path, m)
    else:
        m["completed"] = sorted(set(int(i) for i in m.get("completed", []) if 0 <= int(i) < total)); save_manifest(m_path, m)

    completed = set(m.get("completed", []))
    for i in range(total):
        if _image_existing_path(img_dir, i): completed.add(i)
    m["completed"] = sorted(completed); save_manifest(m_path, m)

    st.caption(f"🖼️ Immagini: {total} da generare (resume attivo).")
    prog = st.progress(len(completed)/total if total else 0.0); status = st.empty()

    for i, piece in enumerate(img_chunks):
        if i in completed:
            status.write(f"✅ Immagine {i+1}/{total} già presente, salto."); prog.progress(len(completed)/total); continue
        status.write(f"🎨 Genero immagine {i+1}/{total} …")
        out = synthesize_single_image(piece, runtime_cfg, img_dir, i)
        if out and os.path.exists(out) and os.path.getsize(out) > 0:
            completed.add(i); m["completed"] = sorted(completed); save_manifest(m_path, m)
            prog.progress(len(completed)/total)
        else:
            st.error(f"❌ Errore immagine {i+1}. Ripremi 'Genera contenuti'.")
            return False
    status.write("✅ Immagini completate!")
    return True

# ---------------------------
# Risoluzione modello Replicate -> owner/name:version_id
# ---------------------------
@cache_data(ttl=3600, show_spinner=False)
def resolve_replicate_model_identifier(model_input: str, token: str) -> Tuple[Optional[str], Optional[str]]:
    mi = (model_input or "").strip()
    if not mi: return None, "Modello vuoto."
    if ":" in mi:
        parts = mi.split(":")
        if len(parts) == 2 and parts[0].count("/") == 1 and parts[1]:
            return mi, None
        return None, f"Formato modello non valido: {mi}"
    if mi.count("/") != 1: return None, f"Atteso 'owner/name' (facoltativo ':version'), ricevuto: {mi}"
    if not token: return None, "Replicate API token assente."
    owner, name = mi.split("/", 1)
    url = f"https://api.replicate.com/v1/models/{owner}/{name}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code == 200:
            data = r.json() or {}
            ver = (data.get("default_version") or data.get("latest_version") or {}).get("id")
            if not ver: return None, f"Modello trovato ma senza version id: {mi}"
            return f"{owner}/{name}:{ver}", None
        elif r.status_code in (401, 403): return None, "Token non autorizzato (401/403)."
        elif r.status_code == 404: return None, "Modello inesistente (404)."
        else: return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"Errore chiamando Replicate: {e}"

# ---------------------------
# MAIN APP (gabbia di sicurezza)
# ---------------------------
def main():
    st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="centered")
    st.title("🎬 Generatore di Video con Immagini e Audio")

    # Se utils non è importato, avvisa ma non bloccare l'app
    if _utils_import_error:
        st.warning("⚠️ Problema nel caricare `scripts.utils`. L'app si apre comunque; gli errori compariranno al momento della generazione.")
        with st.expander("Dettagli import error"):
            st.code(_utils_import_error, language="text")

    # Carica config opzionale
    base_cfg: Dict[str, Any] = {}
    if load_config:
        try:
            loaded = load_config()
            if isinstance(loaded, dict):
                base_cfg = loaded
        except Exception as e:
            st.warning(f"Config opzionale non caricata: {e}")

    # ===========================
    # 🔐 & ⚙️ Sidebar: API + Parametri
    # ===========================
    with st.sidebar:
        st.header("🔐 API Keys")
        st.caption("Le chiavi valgono solo per *questa sessione* del browser.")

        rep_prefill = st.session_state.get("replicate_api_key", "")
        fish_prefill = st.session_state.get("fish_audio_api_key", "")

        with st.form("api_keys_form", clear_on_submit=False):
            replicate_key = st.text_input("Replicate API key", type="password", value=rep_prefill, placeholder="r8_********")
            fish_key = st.text_input("FishAudio API key", type="password", value=fish_prefill, placeholder="fa_********")
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
                    r = requests.get("https://api.replicate.com/v1/account",
                                     headers={"Authorization": f"Bearer {tok}"}, timeout=15)
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
        fish_voice_id = st.text_input("FishAudio Voice ID", value=voice_prefill, placeholder="es. voice_123abc...")
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
        preset_selected = st.selectbox("Modello Replicate (image generator)", model_presets, index=0)

        custom_prefill = st.session_state.get("replicate_model_custom", "")
        custom_model = st.text_input("Custom model (owner/name:tag oppure owner/name)",
                                     value=custom_prefill,
                                     placeholder="es. black-forest-labs/flux-1.1")
        if custom_model != custom_prefill:
            st.session_state["replicate_model_custom"] = custom_model.strip()

        effective_model = (st.session_state.get("replicate_model_custom", "").strip()
                           if preset_selected == "Custom (digita sotto)" else preset_selected)
        st.session_state["replicate_model"] = effective_model

        if st.button("Verifica modello Replicate"):
            tok = _clean_token(st.session_state.get("replicate_api_key", ""))
            resolved, err = resolve_replicate_model_identifier(effective_model, tok)
            if resolved: st.success(f"✅ Modello utilizzabile: `{resolved}`")
            else: st.error(f"❌ Modello non utilizzabile: {err}")

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
    st.write(f"🔎 Stato API → Replicate: {'✅' if rep_ok else '⚠️'} · FishAudio: {'✅' if fish_ok else '⚠️'} · "
             f"Model(Immagini): {rep_model} · VoiceID(Audio): {voice_id}")

    # ===========================
    # 🎛️ Parametri generazione (centrale)
    # ===========================
    title = st.text_input("Titolo del video")
    script = st.text_area("Inserisci il testo da usare per generare immagini/audio", height=300)
    mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio", "Entrambi"])

    if mode in ["Audio", "Entrambi"]:
        seconds_per_img = st.number_input("Ogni quanti secondi di audio creare un'immagine?",
                                          min_value=1, value=int(st.session_state.get("seconds_per_img", 8)), step=1)
        st.session_state["seconds_per_img"] = int(seconds_per_img)
    else:
        sentences_per_image = st.number_input("Ogni quante frasi creare un'immagine?",
                                              min_value=1, value=int(st.session_state.get("sentences_per_image", 2)), step=1)
        st.session_state["sentences_per_image"] = int(sentences_per_image)

    # Prompt fisso per ogni immagine
    fixed_image_prompt = st.text_area(
        "Prompt fisso per ogni immagine (opzionale)",
        value=st.session_state.get("fixed_image_prompt", ""),
        help="Esempio: 'Epoca dell'antica Roma nell'anno 200 d.C. Usa uno stile ultra realistico.'"
    )
    st.session_state["fixed_image_prompt"] = fixed_image_prompt

    fixed_prompt_position_label = radio_compat(
        "Dove inserire il prompt fisso?",
        options=["Dopo il testo del chunk", "Prima del testo del chunk"],
        index=0 if st.session_state.get("fixed_image_prompt_position", "after") == "after" else 1,
        horizontal_default=True
    )
    fixed_pos_norm = "after" if fixed_prompt_position_label.startswith("Dopo") else "before"
    st.session_state["fixed_image_prompt_position"] = fixed_pos_norm

    generate = st.button("🚀 Genera contenuti")

    # ===========================
    # 🚀 Avvio generazione
    # ===========================
    if generate and title.strip() and script.strip():
        # cartelle output
        safe = sanitize(title)
        base = os.path.join("data", "outputs", safe)
        img_dir = os.path.join(base, "images")
        aud_dir = os.path.join(base, "audio")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(aud_dir, exist_ok=True)

        audio_path = os.path.join(aud_dir, "combined_audio.mp3")

        st.subheader("🔄 Generazione in corso…")

        # runtime cfg
        runtime_cfg: Dict[str, Any] = dict(base_cfg)
        replicate_from_ui = _clean_token(get_replicate_key())
        fishaudio_from_ui = _clean_token(get_fishaudio_key())
        if replicate_from_ui:
            os.environ["REPLICATE_API_TOKEN"] = replicate_from_ui
            runtime_cfg["replicate_api_key"] = replicate_from_ui
            runtime_cfg["replicate_api_token"] = replicate_from_ui
        if fishaudio_from_ui:
            os.environ["FISHAUDIO_API_KEY"] = fishaudio_from_ui
            runtime_cfg["fishaudio_api_key"] = fishaudio_from_ui

        # risolvi modello Replicate
        effective = get_replicate_model()
        if effective:
            resolved_model, err = resolve_replicate_model_identifier(effective, replicate_from_ui)
            if resolved_model:
                runtime_cfg["replicate_model"] = resolved_model
                runtime_cfg["replicate_model_resolved"] = resolved_model
                if ":" not in effective:
                    st.caption(f"🔁 Modello risolto automaticamente: `{resolved_model}`")
            else:
                runtime_cfg["replicate_model"] = effective
                st.warning(f"⚠️ Modello non risolto: {err}")

        # audio params
        fish_voice = get_fishaudio_voice_id()
        if fish_voice:
            runtime_cfg["fishaudio_voice_id"] = fish_voice

        st.write("🔐 Replicate token: "
                 + _mask(runtime_cfg.get("replicate_api_key") or runtime_cfg.get("replicate_api_token") or os.getenv("REPLICATE_API_TOKEN"))
                 + " · Modello: `" + (runtime_cfg.get("replicate_model") or runtime_cfg.get("image_model") or "—") + "`")

        # ---- AUDIO (con resume) ----
        if mode in ["Audio", "Entrambi"]:
            if not fish_ok:
                st.error("❌ FishAudio API key mancante.")
                st.stop()
            if not get_fishaudio_voice_id():
                st.error("❌ FishAudio Voice ID mancante.")
                st.stop()
            st.text(f"🎧 Generazione audio con voce: {get_fishaudio_voice_id()} …")
            try:
                final_audio = generate_audio_with_resume(script, runtime_cfg, aud_dir, max_chars=2000)
            except Exception as e:
                st.error("❌ Errore durante la generazione audio.")
                st.exception(e)
                st.stop()
            if final_audio:
                audio_path = final_audio
                st.success("🎉 Audio pronto. Puoi scaricarlo subito mentre genero (eventualmente) le immagini.")
                st.session_state["audio_path"] = audio_path
                try:
                    with open(audio_path, "rb") as f:
                        st.download_button("🎧 Scarica Audio MP3 (subito)", f, file_name="audio.mp3",
                                           mime="audio/mpeg", key="dl-audio-early")
                except Exception:
                    pass
            else:
                st.error("⚠️ Audio non completato.")
                st.stop()

        # ---- IMMAGINI (con resume + prompt fisso) ----
        if mode in ["Immagini", "Entrambi"]:
            if not rep_ok:
                st.error("❌ Replicate API key mancante.")
                st.stop()
            if not runtime_cfg.get("replicate_model"):
                st.error("❌ Modello Replicate mancante o non risolto.")
                st.stop()

            fixed = st.session_state.get("fixed_image_prompt", "") or ""
            fixed_pos = st.session_state.get("fixed_image_prompt_position", "after")

            try:
                if mode == "Entrambi":
                    if not os.path.exists(audio_path):
                        st.error("❌ Audio non trovato per calcolare le immagini.")
                    else:
                        secs = int(st.session_state.get("seconds_per_img", 8))
                        st.text(f"🖼️ Generazione immagini (1 ogni {secs}s)…")
                        try:
                            duration_sec = mp3_duration_seconds(audio_path) if mp3_duration_seconds else 0
                        except Exception:
                            duration_sec = 0
                        if not duration_sec: duration_sec = 60
                        num_images = max(1, int(duration_sec // max(1, secs)))
                        approx_chars = max(1, len(script) // max(1, num_images))
                        base_chunks = chunk_text(script, approx_chars) if chunk_text else [script]
                        img_chunks = [combined_image_prompt(c, fixed, fixed_pos) for c in base_chunks]
                        resume_key = {
                            "script_hash": script_hash(script + "||" + fixed + "||" + fixed_pos),
                            "strategy": "by_seconds",
                            "param_value": secs,
                            "fixed_prompt_hash": script_hash(fixed),
                            "fixed_prompt_position": fixed_pos,
                        }
                        ok_imgs = generate_images_with_resume(img_chunks, img_dir, resume_key, runtime_cfg)
                        if ok_imgs: zip_images(base)
                else:
                    spi = int(st.session_state.get("sentences_per_image", 2)) or 2
                    st.text(f"🖼️ Generazione immagini (1 ogni {spi} frasi)…")
                    base_groups = (chunk_by_sentences_count(script, spi) if chunk_by_sentences_count else [script])
                    img_chunks = [combined_image_prompt(c, fixed, fixed_pos) for c in base_groups]
                    resume_key = {
                        "script_hash": script_hash(script + "||" + fixed + "||" + fixed_pos),
                        "strategy": "by_sentences",
                        "param_value": spi,
                        "fixed_prompt_hash": script_hash(fixed),
                        "fixed_prompt_position": fixed_pos,
                    }
                    ok_imgs = generate_images_with_resume(img_chunks, img_dir, resume_key, runtime_cfg)
                    if ok_imgs: zip_images(base)
            except Exception as e:
                st.error("❌ Errore durante la generazione immagini.")
                st.exception(e)
                st.stop()

        st.success("✅ Generazione completata!")
        st.session_state["audio_path"] = audio_path if os.path.exists(audio_path) else st.session_state.get("audio_path")
        zip_path = os.path.join(base, "output.zip")
        st.session_state["zip_path"] = zip_path if os.path.exists(zip_path) else None

    # ---- Download (finale)
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

# Avvio protetto: mostra lo stacktrace in pagina invece del generico "Oh no"
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 L'app è andata in errore all'avvio (import-time). Ecco lo stacktrace:")
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
