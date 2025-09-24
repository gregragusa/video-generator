# app.py
# -------------------------------------------------------
# Streamlit app: IMMAGINI / AUDIO con Replicate + FishAudio.
# ✅ Resume reale e robusto per AUDIO/IMMAGINI con checkpoint atomici
# ✅ Pulsanti "Continua da chunk N" + Auto-Restore con "Genera contenuti"
# ✅ Combine audio affidabile anche con formati misti (mp3/wav/m4a)
# ✅ Nessuna dipendenza da vecchie funzioni di checkpoint esterne
# ✅ (MOD) Chunk audio ~1000 char a fine frase + pausa tra chunk (~0.8s)
# -------------------------------------------------------

import os
import re
import json
import time
import hashlib
import subprocess
from datetime import datetime
import streamlit as st

# opzionale
try:
    from scripts.config_loader import load_config  # type: ignore
except Exception:  # pragma: no cover
    load_config = None

# utils base (richiesti)
from scripts.utils import (  # type: ignore
    chunk_by_sentences_count,
    chunk_text_for_audio,
    generate_audio,
    generate_images,
    mp3_duration_seconds,
)

# -------------------------------------------------------
# Costanti / Utility
# -------------------------------------------------------
AUDIO_EXTS = ("mp3", "wav", "m4a")
IMAGE_EXTS = ("png", "jpg", "jpeg")
STATE_FILENAME = "state.json"  # checkpoint unificato per questo progetto
GEN_TIMEOUT_SECS = 10 * 60
SILENCE_BETWEEN_PARTS_SECS = 0.8  # ✨ pausa tra un chunk audio e il successivo (0.5–1.0s ok)


def sanitize(title: str) -> str:
    s = (title or "").lower()
    for a, b in [(" ", "_"), ("ù", "u"), ("à", "a"), ("è", "e"), ("ì", "i"), ("ò", "o"), ("é", "e")]:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum() or ch == "_") or "video"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# -------------------------------------------------------
# Stato/Checkpoint (atomico, senza dipendere da utils vecchi)
# -------------------------------------------------------
class StateStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, STATE_FILENAME)
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {}

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def update(self, **kwargs):
        self.state.update(kwargs)
        self.save()

    def get(self, key: str, default=None):
        return self.state.get(key, default)


# -------------------------------------------------------
# Lock anti-stallo
# -------------------------------------------------------

def _lock_path(base_dir: str) -> str:
    return os.path.join(base_dir, ".generation.lock")


def write_lock(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    data = {"start_ts": time.time(), "last_progress_ts": time.time()}
    with open(_lock_path(base_dir), "w", encoding="utf-8") as f:
        json.dump(data, f)


def touch_progress(base_dir: str):
    try:
        p = _lock_path(base_dir)
        data = {"start_ts": time.time(), "last_progress_ts": time.time()}
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                old = json.load(f)
                data["start_ts"] = old.get("start_ts", data["start_ts"])
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def is_lock_stale(base_dir: str, timeout_secs: int = GEN_TIMEOUT_SECS) -> bool:
    p = _lock_path(base_dir)
    try:
        if not os.path.exists(p):
            return False
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        last = float(d.get("last_progress_ts") or d.get("start_ts") or 0)
        return (time.time() - last) > timeout_secs
    except Exception:
        return True


def clear_lock(base_dir: str):
    try:
        os.remove(_lock_path(base_dir))
    except Exception:
        pass


# -------------------------------------------------------
# File helpers
# -------------------------------------------------------

def ensure_empty_dir(path: str):
    os.makedirs(path, exist_ok=True)
    for n in os.listdir(path):
        fp = os.path.join(path, n)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception:
            pass


def move_single_output(src_dir: str, dst_fullpath_no_ext: str, preferred_exts) -> str | None:
    files = [n for n in os.listdir(src_dir) if not n.startswith(".")]
    if not files:
        return None
    # nella cartella temporanea ci deve essere un solo file di output
    files.sort(key=lambda n: os.path.getmtime(os.path.join(src_dir, n)))
    src = os.path.join(src_dir, files[-1])  # prendi il PIÙ recente
    ext = files[-1].split(".")[-1].lower()
    if ext not in preferred_exts:
        pass  # salviamo col suo ext; eventuale transcodifica avverrà nel combine
    dst = f"{dst_fullpath_no_ext}.{ext}"
    try:
        os.replace(src, dst)
    except Exception:
        import shutil
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


def existing_part_indices(dir_path: str, prefix: str, exts) -> list[int]:
    inds = []
    for n in os.listdir(dir_path):
        if not n.startswith(f"{prefix}_"):
            continue
        try:
            name, ext = n.rsplit(".", 1)
        except ValueError:
            continue
        if ext.lower() not in exts:
            continue
        m = re.match(rf"{re.escape(prefix)}_(\d+)$", name)
        if m:
            inds.append(int(m.group(1)))
    return sorted(inds)


def contiguous_from_zero(indices: list[int]) -> int:
    s = set(indices)
    i = 0
    while i in s:
        i += 1
    return i


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
# CHUNK PERSISTENCE (determinismo)
# -------------------------------------------------------

def json_list_save(path: str, items: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def json_list_load(path: str) -> list | None:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return None
    return None


# -------------------------------------------------------
# SPLIT AUDIO: ~1000 caratteri, sempre a fine frase
# -------------------------------------------------------

def sentences_from_script(script: str) -> list:
    # split robusto per . ? ! e spazi successivi
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", script.strip()) if s.strip()]


def build_or_load_audio_chunks(base: str, script: str, chunk_size: int) -> list:
    """
    (MOD) Split greedy per frasi con target ≈1000 char per chunk.
    - Non taglia MAI parole o frasi.
    - Se una singola frase supera i 1000, la teniamo intera (come richiesto).
    - Aggiunge un punto finale se il chunk non termina con .?!
    - Cache su file; invalida se cambia lo script.
    """
    path = os.path.join(base, "audio_chunks.json")
    meta_path = os.path.join(base, "audio_chunks_meta.json")
    cur_sha = sha1(script)

    if os.path.exists(path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            if meta.get("script_sha") == cur_sha:
                cached = json_list_load(path)
                if cached:
                    return cached
        except Exception:
            pass

    TARGET = 1000  # soft limit (priorità: fine frase)
    sents = sentences_from_script(script)

    chunks: list[str] = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        candidate = (f"{buf} {s}".strip()) if buf else s
        if buf and len(candidate) > TARGET:
            # chiudi buf a fine frase
            if buf[-1] not in ".?!":
                buf += "."
            chunks.append(buf.strip())
            buf = s  # nuova frase nel prossimo chunk
        else:
            buf = candidate
    if buf:
        if buf[-1] not in ".?!":
            buf += "."
        chunks.append(buf.strip())

    json_list_save(path, chunks)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"script_sha": cur_sha, "target": TARGET}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return chunks


# -------------------------------------------------------
# Combine Audio robusto (gestisce formati misti) + pausa tra parti
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
    ensure_empty_dir(tmp_dir)

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

    # 3) appende silenzio a ogni parte (tranne l'ultima)
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
            # cleanup
            for n in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, n))
                except Exception:
                    pass
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
        for n in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, n))
            except Exception:
                pass


# -------------------------------------------------------
# Timeline (uguale alla tua)
# -------------------------------------------------------
class ProgressTracker:
    def __init__(self):
        self.start_time = None
        self.steps = []
        self.current_step = None
        self.estimated_total_seconds = 0

    def start(self, total_audio_chunks: int, total_images: int):
        self.start_time = datetime.now()
        self.steps = []
        audio_est = (total_audio_chunks or 0) * 9
        img_est = (total_images or 0) * 15
        self.estimated_total_seconds = audio_est + img_est

    def add_step(self, step_type, description, status="running"):
        step = {
            "type": step_type,
            "description": description,
            "status": status,
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None,
            "substeps": [],
        }
        self.steps.append(step)
        self.current_step = len(self.steps) - 1
        return self.current_step

    def add_substep(self, idx, description, status="completed"):
        if 0 <= idx < len(self.steps):
            self.steps[idx]["substeps"].append(
                {"description": description, "status": status, "timestamp": datetime.now()}
            )

    def complete_step(self, idx, status="completed"):
        if 0 <= idx < len(self.steps):
            self.steps[idx]["end_time"] = datetime.now()
            self.steps[idx]["status"] = status
            if self.steps[idx]["start_time"]:
                dt = self.steps[idx]["end_time"] - self.steps[idx]["start_time"]
                self.steps[idx]["duration"] = dt.total_seconds()

    def get_elapsed_time(self):
        return (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

    def get_eta(self):
        elapsed = self.get_elapsed_time()
        if elapsed < 30:
            return max(0, self.estimated_total_seconds - elapsed)
        completed = len([s for s in self.steps if s["status"] == "completed"])
        total = len(self.steps)
        if completed > 0:
            avg = elapsed / completed
            remaining = total - completed
            return max(0, remaining * avg)
        return max(0, self.estimated_total_seconds - elapsed)

    def get_completion_percentage(self):
        if not self.steps:
            return 0
        completed = len([s for s in self.steps if s["status"] == "completed"])
        return min(100, (completed / len(self.steps)) * 100)


def display_timeline(tracker: ProgressTracker, container):
    """Due barre di avanzamento (Audio/Immagini)."""
    try:
        container.empty()
    except Exception:
        pass

    title_cur = st.session_state.get("title", "")
    a_done = a_total = i_done = i_total = 0
    if title_cur:
        base = os.path.join("data", "outputs", sanitize(title_cur))
        aud_dir = os.path.join(base, "audio")
        img_dir = os.path.join(base, "images")

        if os.path.exists(aud_dir):
            a_done = contiguous_from_zero(existing_part_indices(aud_dir, "part", AUDIO_EXTS))
        if os.path.exists(img_dir):
            i_done = contiguous_from_zero(existing_part_indices(img_dir, "img", IMAGE_EXTS))

        acp = os.path.join(base, "audio_chunks.json")
        icp = os.path.join(base, "image_chunks.json")
        try:
            if os.path.exists(acp):
                a_total = len(json_list_load(acp) or [])
        except Exception:
            a_total = 0
        try:
            if os.path.exists(icp):
                i_total = len(json_list_load(icp) or [])
        except Exception:
            i_total = 0

    with container.container():
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎧 Audio")
            st.metric("Completati", f"{a_done}/{a_total}" if a_total else "—")
            st.progress((a_done / a_total) if a_total else 0.0)
        with c2:
            st.markdown("#### 🖼️ Immagini")
            st.metric("Completate", f"{i_done}/{i_total}" if i_total else "—")
            st.progress((i_done / i_total) if i_total else 0.0)
        st.caption("La barra si aggiorna ad ogni chunk completato.")


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="wide")
st.title("🎬 Generatore di Video con Immagini e Audio")

# auto-sblocco se lock stantio
if st.session_state.get("is_generating", False):
    t = st.session_state.get("title", "")
    if t:
        base_check = os.path.join("data", "outputs", sanitize(t))
        if is_lock_stale(base_check):
            clear_lock(base_check)
            st.session_state["is_generating"] = False
            st.info("🔧 Sessione precedente bloccata: sbloccata automaticamente.")

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
    voice_prefill = st.session_state.get("fishaudio_voice_id", "")
    fish_voice_id = st.text_input("FishAudio Voice ID", value=voice_prefill)
    if fish_voice_id != voice_prefill:
        st.session_state["fishaudio_voice_id"] = fish_voice_id.strip()

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
    st.subheader("🎯 Prompt fisso immagini")
    st.checkbox("Usa prompt fisso per immagini", value=True, key="use_fixed_prompt_images")

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

    # ✅ Opzione per avere chunk ~durata target (stima)
    use_dur = st.checkbox(
        "Imposta chunk per durata target",
        value=True,
        help="Calcola automaticamente il chunk_size in base ai secondi desiderati e a una velocità di parlato stimata."
    )
    if use_dur:
        target_secs = st.number_input("Durata target chunk (s)", min_value=30, max_value=600, value=120, step=10, key="target_chunk_secs")
        cps_est = st.number_input("Parlato stimato (caratteri/s)", min_value=8.0, max_value=25.0, value=16.0, step=0.5, key="cps_est")
        st.session_state["chunk_size"] = int(target_secs * cps_est)
        st.caption(f"Chunk stimato ≈ {st.session_state['chunk_size']} caratteri (~{st.session_state['chunk_size']/(cps_est*60):.1f} min)")

    st.divider()
    st.header("🔄 Resume & Sblocco")
    if st.button("🔓 Sblocca progetto corrente"):
        t = st.session_state.get("title", "")
        if t:
            clear_lock(os.path.join("data", "outputs", sanitize(t)))
        st.session_state["is_generating"] = False
        st.success("Sbloccato. Puoi riprendere.")
        st.experimental_rerun()
    if st.button("🧹 Sblocca tutti i progetti"):
        base_root = "data/outputs"
        n = 0
        if os.path.exists(base_root):
            for name in os.listdir(base_root):
                p = os.path.join(base_root, name, ".generation.lock")
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        n += 1
                    except Exception:
                        pass
        st.session_state["is_generating"] = False
        st.success(f"Sbloccati {n} progetti.")
        st.experimental_rerun()

# Stato API

def get_replicate_key() -> str:
    return (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN", "")).strip()


def get_fishaudio_key() -> str:
    return (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY", "")).strip()


def get_fishaudio_voice_id() -> str:
    return st.session_state.get("fishaudio_voice_id", "").strip()


def get_replicate_model() -> str:
    return st.session_state.get("replicate_model", "").strip()


def get_chunk_size() -> int:
    return st.session_state.get("chunk_size", 2000)


def get_sleep_time() -> float:
    return st.session_state.get("sleep_time", 11.0)


st.write(
    f"🔎 **Stato API** → Replicate: {'✅' if get_replicate_key() else '⚠️'} · "
    f"FishAudio: {'✅' if get_fishaudio_key() else '⚠️'} · "
    f"Model: `{get_replicate_model() or '—'}` · Voice: `{get_fishaudio_voice_id() or '—'}` · "
    f"Chunk: ~{get_chunk_size()/(st.session_state.get('cps_est',16)*60):.1f} min (≈{get_chunk_size()} char)"
)

# ---------------- Main UI ----------------
col_main, col_timeline = st.columns([2, 3])
with col_main:
    st.subheader("📝 Input")
    title = st.text_input("Titolo del video")
    script = st.text_area("Testo per generare immagini/audio", height=220)
    mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio", "Entrambi"], index=2)

    if mode in ["Audio", "Entrambi"]:
        seconds_per_img = st.number_input("Ogni quanti secondi creare un'immagine?", min_value=1, value=8, step=1)
        st.session_state["seconds_per_img"] = seconds_per_img
    else:
        sentences_per_image = st.number_input("Quante frasi per immagine?", min_value=1, value=2, step=1)
        st.session_state["sentences_per_image"] = sentences_per_image

    # ----- Stato resume + pulsanti -----
    resume_audio_btn_clicked = False
    resume_images_btn_clicked = False

    generate = st.button("🚀 Genera contenuti", use_container_width=True)

    if title.strip() and script.strip():
        safe = sanitize(title)
        base = os.path.join("data", "outputs", safe)
        aud_dir = os.path.join(base, "audio")
        img_dir = os.path.join(base, "images")
        os.makedirs(aud_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        state = StateStore(base)
        state.update(project_title=title, script_sha=sha1(script), chunk_size=get_chunk_size())

        # AUDIO: preview stato
        audio_chunks_preview = build_or_load_audio_chunks(base, script, get_chunk_size())
        total_a = len(audio_chunks_preview)
        a_indices = existing_part_indices(aud_dir, "part", AUDIO_EXTS)
        a_done = contiguous_from_zero(a_indices)
        a_done = max(a_done, int(state.get("audio_completed", 0) or 0))
        st.info(f"🎧 Audio: **{a_done} su {total_a}** chunk creati")
        if a_done < total_a:
            if st.button(f"▶ Continua generazione AUDIO da chunk {a_done+1}", key="resume_audio_btn"):
                st.session_state["resume_mode"] = "audio"
                st.session_state["resume_start_audio_idx"] = a_done
                resume_audio_btn_clicked = True

        # IMMAGINI: preview stato
        img_chunks_path = os.path.join(base, "image_chunks.json")
        img_chunks_preview = json_list_load(img_chunks_path)
        total_i = len(img_chunks_preview) if img_chunks_preview else None
        i_indices = existing_part_indices(img_dir, "img", IMAGE_EXTS)
        i_done = contiguous_from_zero(i_indices)
        i_done = max(i_done, int(state.get("images_completed", 0) or 0))
        if total_i is not None:
            st.info(f"🖼️ Immagini: **{i_done} su {total_i}** create")
            if i_done < total_i:
                if st.button(f"▶ Continua generazione IMMAGINI da chunk {i_done+1}", key="resume_images_btn"):
                    st.session_state["resume_mode"] = "images"
                    st.session_state["resume_start_image_idx"] = i_done
                    resume_images_btn_clicked = True
        else:
            st.info(f"🖼️ Immagini: **{i_done}** create (totale sconosciuto finché non si pianifica)")

with col_timeline:
    st.subheader("📊 Avanzamento")
    timeline_container = st.empty()
    if not st.session_state.get("is_generating", False):
        with timeline_container.container():
            st.info("⏳ Premi 'Genera contenuti' o un pulsante 'Continua…' per iniziare / riprendere")

trigger_generate = generate or resume_audio_btn_clicked or resume_images_btn_clicked

# -------------------------------------------------------
# Driver di generazione (resume automatico + bottoni dedicati)
# -------------------------------------------------------
if trigger_generate and title.strip() and script.strip():
    # evita doppia generazione
    if st.session_state.get("is_generating", False):
        base_try = os.path.join("data", "outputs", sanitize(st.session_state.get("title", "")))
        if base_try and is_lock_stale(base_try):
            clear_lock(base_try)
            st.session_state["is_generating"] = False
        else:
            st.warning("⏳ Generazione già in corso...")
            st.stop()

    safe = sanitize(title)
    base = os.path.join("data", "outputs", safe)
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    state = StateStore(base)

    # API / Modello
    rep_key = get_replicate_key()
    fish_key = get_fishaudio_key()
    model = get_replicate_model()
    voice = get_fishaudio_voice_id()

    forced_resume_mode = st.session_state.get("resume_mode")  # "audio" | "images" | None
    effective_mode = mode
    if forced_resume_mode == "audio":
        effective_mode = "Audio"
    elif forced_resume_mode == "images":
        effective_mode = "Immagini"

    if effective_mode in ["Audio", "Entrambi"]:
        if not fish_key:
            st.error("❌ FishAudio API key mancante!")
            st.stop()
        if not voice:
            st.error("❌ FishAudio Voice ID mancante!")
            st.stop()
    if effective_mode in ["Immagini", "Entrambi"]:
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
    if voice:
        runtime_cfg["fishaudio_voice_id"] = voice
    runtime_cfg["chunk_size"] = get_chunk_size()
    runtime_cfg["sleep_time"] = get_sleep_time()

    # Flag + lock
    st.session_state["is_generating"] = True
    st.session_state["title"] = title
    write_lock(base)

    tracker = ProgressTracker()

    # Chunk deterministici per AUDIO
    audio_chunks = (
        build_or_load_audio_chunks(base, script, runtime_cfg["chunk_size"]) if effective_mode in ["Audio", "Entrambi"] else []
    )

    # Pianificazione immagini
    if effective_mode == "Immagini":
        img_groups = chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2)))
        img_chunks_plan = img_groups
        json_list_save(os.path.join(base, "image_chunks.json"), img_chunks_plan)
    else:
        img_chunks_plan = json_list_load(os.path.join(base, "image_chunks.json")) or []

    est_images = len(img_chunks_plan) if effective_mode == "Immagini" else (len(img_chunks_plan) if img_chunks_plan else 0)
    tracker.start(len(audio_chunks), est_images)
    display_timeline(tracker, timeline_container)
    st.success(f"🎯 Avvio: {len(audio_chunks)} chunk audio previsti • {est_images} immagini pianificate")

    def _progress(msg: str):
        touch_progress(base)
        if tracker.current_step is not None:
            tracker.add_substep(tracker.current_step, msg, "completed")
            display_timeline(tracker, timeline_container)

    try:
        # ----------------- AUDIO -----------------
        if effective_mode in ["Audio", "Entrambi"]:
            step = tracker.add_step("audio", "🎧 Generazione Audio (resume a chunk)")
            display_timeline(tracker, timeline_container)

            # indice di ripartenza
            if st.session_state.get("resume_mode") == "audio" and st.session_state.get("resume_start_audio_idx") is not None:
                start_idx = int(st.session_state["resume_start_audio_idx"])
            else:
                completed_cp = int(state.get("audio_completed", 0) or 0)
                indices = existing_part_indices(aud_dir, "part", AUDIO_EXTS)
                leading_files = contiguous_from_zero(indices)
                start_idx = max(completed_cp, leading_files)

            total = len(audio_chunks)
            tracker.add_substep(step, f"📦 Ripartenza audio: {start_idx} su {total} (prossimo: chunk {start_idx+1})", "completed")
            display_timeline(tracker, timeline_container)

            for i in range(start_idx, total):
                tmp = os.path.join(aud_dir, "_tmp")
                ensure_empty_dir(tmp)

                # genera 1 chunk alla volta
                try:
                    generate_audio([audio_chunks[i]], runtime_cfg, tmp, progress_cb=_progress)  # type: ignore
                except TypeError:
                    generate_audio([audio_chunks[i]], runtime_cfg, tmp)  # compat vecchie utils

                target_noext = os.path.join(aud_dir, f"part_{i:03d}")
                out = move_single_output(tmp, target_noext, preferred_exts=AUDIO_EXTS)
                if not out or not os.path.exists(out):
                    tracker.add_substep(step, f"❌ Chunk {i} non prodotto", "failed")
                    state.update(audio_completed=i)
                    display_timeline(tracker, timeline_container)
                    st.error(f"Errore nel generare chunk audio {i}")
                    st.stop()

                state.update(audio_completed=i + 1)
                tracker.add_substep(step, f"✅ {i+1} su {total} creati", "completed")
                display_timeline(tracker, timeline_container)
                touch_progress(base)

            # combine sempre in mp3 (anche se part_* sono wav/m4a)
            combined = os.path.join(aud_dir, "combined_audio.mp3")
            ok = combine_parts_to_mp3(aud_dir, combined)
            if ok:
                st.session_state["audio_path"] = combined
                st.session_state["audio_ready"] = True
                try:
                    tracker.add_substep(step, f"🎵 Combinato: {mp3_duration_seconds(combined):.1f}s", "completed")
                except Exception:
                    tracker.add_substep(step, f"🎵 Combinato", "completed")
            else:
                tracker.add_substep(step, "ℹ️ ffmpeg non disponibile o combine fallito: restano i part_*.audio", "completed")

            tracker.complete_step(step, "completed")
            display_timeline(tracker, timeline_container)

        # ----------------- IMMAGINI -----------------
        if effective_mode in ["Immagini", "Entrambi"]:
            step = tracker.add_step("images", "🖼️ Generazione Immagini (resume a chunk)")
            display_timeline(tracker, timeline_container)

            if effective_mode == "Entrambi":
                # richiede audio combinato per calcolare planning immagini
                audio_path = os.path.join(aud_dir, "combined_audio.mp3")
                if not os.path.exists(audio_path):
                    st.error("❌ Audio combinato non trovato: completa l'audio prima di creare le immagini.")
                    tracker.complete_step(step, "failed")
                    display_timeline(tracker, timeline_container)
                    st.stop()
                duration = mp3_duration_seconds(audio_path) or 60
                seconds_per_img = st.session_state.get("seconds_per_img", 8)
                num_images = max(1, int(duration // seconds_per_img))
                sents = sentences_from_script(script)
                if num_images == 1:
                    img_chunks_plan = [script]
                else:
                    per_img = max(1, len(sents) // num_images)
                    img_chunks_plan = [" ".join(sents[i: i + per_img]) for i in range(0, len(sents), per_img)]
                json_list_save(os.path.join(base, "image_chunks.json"), img_chunks_plan)
            else:
                if not img_chunks_plan:
                    img_chunks_plan = json_list_load(os.path.join(base, "image_chunks.json")) or []
                    if not img_chunks_plan:
                        img_chunks_plan = chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2)))
                        json_list_save(os.path.join(base, "image_chunks.json"), img_chunks_plan)

            total_imgs = len(img_chunks_plan)

            # indice di ripartenza
            if st.session_state.get("resume_mode") == "images" and st.session_state.get("resume_start_image_idx") is not None:
                start_img = int(st.session_state["resume_start_image_idx"])
            else:
                completed_imgs_cp = int(state.get("images_completed", 0) or 0)
                indices = existing_part_indices(img_dir, "img", IMAGE_EXTS)
                leading_files = contiguous_from_zero(indices)
                start_img = max(completed_imgs_cp, leading_files)

            tracker.add_substep(step, f"📦 Ripartenza immagini: {start_img} su {total_imgs} (prossima: {start_img+1})", "completed")
            display_timeline(tracker, timeline_container)

            for i in range(start_img, total_imgs):
                tmp = os.path.join(img_dir, "_tmp")
                ensure_empty_dir(tmp)

                try:
                    generate_images([img_chunks_plan[i]], runtime_cfg, tmp, progress_cb=_progress)  # type: ignore
                except TypeError:
                    generate_images([img_chunks_plan[i]], runtime_cfg, tmp)

                target_noext = os.path.join(img_dir, f"img_{i:03d}")
                out = move_single_output(tmp, target_noext, preferred_exts=IMAGE_EXTS)
                if not out or not os.path.exists(out):
                    tracker.add_substep(step, f"❌ Immagine {i} non prodotta", "failed")
                    state.update(images_completed=i)
                    display_timeline(tracker, timeline_container)
                    st.error(f"Errore nel generare immagine {i}")
                    st.stop()

                state.update(images_completed=i + 1)
                tracker.add_substep(step, f"✅ {i+1} su {total_imgs} create", "completed")
                display_timeline(tracker, timeline_container)
                touch_progress(base)

            # ZIP
            zip_images(base)
            st.session_state["zip_path"] = os.path.join(base, "output.zip")
            st.session_state["zip_ready"] = os.path.exists(st.session_state["zip_path"])

            tracker.complete_step(step, "completed")
            display_timeline(tracker, timeline_container)

        # ----------------- FINAL -----------------
        final = tracker.add_step("finalize", "🎉 Finalizzazione")
        display_timeline(tracker, timeline_container)
        files = []
        ap = os.path.join(aud_dir, "combined_audio.mp3")
        if os.path.exists(ap):
            st.session_state["audio_path"] = ap
            st.session_state["audio_ready"] = True
            files.append("Audio MP3")
        zp = os.path.join(base, "output.zip")
        if os.path.exists(zp):
            st.session_state["zip_path"] = zp
            st.session_state["zip_ready"] = True
            files.append("ZIP Immagini")
        tracker.add_substep(final, f"📦 Files: {', '.join(files) if files else '—'}", "completed")
        tracker.complete_step(final, "completed")
        display_timeline(tracker, timeline_container)
        st.balloons()
        st.success("✅ Generazione completata (o stato aggiornato).")

    except Exception as e:
        st.error(f"💥 ERRORE: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        clear_lock(base)
        st.session_state["is_generating"] = False
        for k in ["resume_mode", "resume_start_audio_idx", "resume_start_image_idx"]:
            if k in st.session_state:
                del st.session_state[k]

# ----------------- DOWNLOAD -----------------
st.divider()
st.subheader("📥 Download Files")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 🎧 Audio")
    if st.session_state.get("audio_ready") and st.session_state.get("audio_path") and os.path.exists(st.session_state["audio_path"]):
        ap = st.session_state["audio_path"]
        try:
            size_mb = os.path.getsize(ap) / (1024 * 1024)
            dur = mp3_duration_seconds(ap)
            bitrate = (size_mb * 8 * 1024) / dur if dur and dur > 0 else 0
            st.info(f"Durata: {dur:.1f}s · Dimensione: {size_mb:.1f} MB · ~{bitrate:.0f} kbps")
        except Exception:
            pass
        with open(ap, "rb") as f:
            st.download_button(
                "Scarica MP3",
                f.read(),
                file_name=f"{sanitize(st.session_state.get('title') or 'audio')}.mp3",
                mime="audio/mpeg",
                use_container_width=True,
            )
    else:
        st.info("⏳ Nessun audio pronto")
with c2:
    st.markdown("### 🖼️ Immagini")
    if st.session_state.get("zip_ready") and st.session_state.get("zip_path") and os.path.exists(st.session_state["zip_path"]):
        zp = st.session_state["zip_path"]
        with open(zp, "rb") as f:
            st.download_button(
                "Scarica ZIP Immagini",
                f.read(),
                file_name=f"{sanitize(st.session_state.get('title') or 'images')}.zip",
                mime="application/zip",
                use_container_width=True,
            )
    else:
        st.info("⏳ Nessuna immagine pronta")
