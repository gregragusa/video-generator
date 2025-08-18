# app.py
# -------------------------------------------------------
# Streamlit app: IMMAGINI / AUDIO con Replicate + FishAudio.
# Resume reale per chunk + pulsanti "Continua da chunk N" + contatori "X su Y".
# Compatibile con utils vecchio/nuovo (progress_cb opzionale).
# -------------------------------------------------------

import os
import re
import json
import time
import subprocess
import requests
from datetime import datetime
import streamlit as st

# opzionale
try:
    from scripts.config_loader import load_config
except Exception:
    load_config = None

# utils base
from scripts.utils import (
    chunk_by_sentences_count,
    chunk_text_for_audio,
    generate_audio,
    generate_images,
    mp3_duration_seconds,
)

# -------------------------------------------------------
# Checkpoint minimi (fallback se non presenti in utils)
# -------------------------------------------------------
try:
    from scripts.utils import load_checkpoint, save_checkpoint
except Exception:
    def load_checkpoint(base_dir: str):
        path = os.path.join(base_dir, "checkpoint.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    def save_checkpoint(base_dir: str, updates: dict, merge: bool = True):
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, "checkpoint.json")
        state = {}
        if merge and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                state = {}
        state.update(updates or {})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------
# Wrapper compatibilità (progress_cb opzionale)
# -------------------------------------------------------
def _call_generate_audio(chunks, cfg, out_dir, progress_cb=None):
    try:
        return generate_audio(chunks, cfg, out_dir, progress_cb=progress_cb)
    except TypeError:
        return generate_audio(chunks, cfg, out_dir)

def _call_generate_images(chunks, cfg, out_dir, progress_cb=None):
    try:
        return generate_images(chunks, cfg, out_dir, progress_cb=progress_cb)
    except TypeError:
        return generate_images(chunks, cfg, out_dir)

# -------------------------------------------------------
# Timeline
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
            "substeps": []
        }
        self.steps.append(step)
        self.current_step = len(self.steps) - 1
        return self.current_step

    def add_substep(self, idx, description, status="completed"):
        if 0 <= idx < len(self.steps):
            self.steps[idx]["substeps"].append({
                "description": description,
                "status": status,
                "timestamp": datetime.now()
            })

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
    if not tracker.start_time:
        return
    with container:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("⏱️ Trascorso", f"{tracker.get_elapsed_time()/60:.1f} min")
        with c2:
            st.metric("🎯 ETA", f"{tracker.get_eta()/60:.1f} min")
        with c3:
            total = (tracker.get_elapsed_time() + tracker.get_eta()) / 60
            st.metric("📊 Totale Stimato", f"{total:.1f} min")
        with c4:
            completed = len([s for s in tracker.steps if s["status"] == "completed"])
            st.metric("✅ Completati", f"{completed}/{len(tracker.steps)}")
        st.progress(tracker.get_completion_percentage()/100, text=f"{tracker.get_completion_percentage():.1f}%")
        st.markdown("### 📋 Timeline Dettagliata")
        for s in tracker.steps:
            icon = "🔄" if s["status"] == "running" else ("✅" if s["status"] == "completed" else "❌")
            style = "**" if s["status"] == "running" else ""
            if s["duration"]:
                tstr = f"({s['duration']:.1f}s)"
            elif s["status"] == "running":
                tstr = f"({(datetime.now()-s['start_time']).total_seconds():.1f}s...)"
            else:
                tstr = ""
            st.markdown(f"{icon} {style}{s['description']}{style} {tstr}")
            if s["substeps"]:
                show = s["substeps"][-3:] if s["status"] == "running" else s["substeps"][-1:]
                for sub in show:
                    sub_icon = "✅" if sub["status"] == "completed" else "❌"
                    st.markdown(f"   └ {sub_icon} {sub['description']}")

# -------------------------------------------------------
# Utility + Lock anti-stallo
# -------------------------------------------------------
def sanitize(title: str) -> str:
    s = (title or "").lower()
    for a, b in [(" ", "_"), ("ù", "u"), ("à", "a"), ("è", "e"),
                 ("ì", "i"), ("ò", "o"), ("é", "e")]:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum() or ch == "_") or "video"

def _clean_token(tok: str) -> str:
    return re.sub(r"\s+", "", (tok or ""))

GEN_TIMEOUT_SECS = 10 * 60
def _lock_path(base_dir: str) -> str: return os.path.join(base_dir, ".generation.lock")
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
        if not os.path.exists(p): return False
        with open(p, "r", encoding="utf-8") as f: d = json.load(f)
        last = float(d.get("last_progress_ts") or d.get("start_ts") or 0)
        return (time.time() - last) > timeout_secs
    except Exception:
        return True
def clear_lock(base_dir: str):
    try: os.remove(_lock_path(base_dir))
    except Exception: pass

def zip_images(base_dir: str):
    import zipfile
    zip_path = os.path.join(base_dir, "output.zip")
    img_dir = os.path.join(base_dir, "images")
    if not os.path.exists(img_dir): return None
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fn in os.listdir(img_dir):
            if fn.startswith("_tmp"): continue
            full = os.path.join(img_dir, fn)
            if os.path.isfile(full):
                zf.write(full, arcname=os.path.join("images", fn))
    return zip_path

# -------------------------------------------------------
# CHUNK PERSISTENCE (determinismo)
# -------------------------------------------------------
def json_list_save(path: str, items: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def json_list_load(path: str) -> list | None:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def build_or_load_audio_chunks(base: str, script: str, chunk_size: int) -> list:
    path = os.path.join(base, "audio_chunks.json")
    cached = json_list_load(path)
    if cached and isinstance(cached, list) and cached:
        return cached
    chunks = chunk_text_for_audio(script, target_chars=chunk_size)
    json_list_save(path, chunks)
    return chunks

def sentences_from_script(script: str) -> list:
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', script.strip()) if s.strip()]

# -------------------------------------------------------
# Driver a chunk singolo + helpers resume
# -------------------------------------------------------
def ensure_empty_dir(path: str):
    os.makedirs(path, exist_ok=True)
    for n in os.listdir(path):
        try: os.remove(os.path.join(path, n))
        except Exception: pass

def move_single_output(src_dir: str, dst_fullpath_no_ext: str, preferred_exts=("mp3","wav","m4a","png","jpg","jpeg")) -> str | None:
    files = [n for n in os.listdir(src_dir) if not n.startswith(".")]
    if not files: return None
    files.sort(key=lambda n: os.path.getmtime(os.path.join(src_dir, n)))
    src = os.path.join(src_dir, files[0])
    ext = files[0].split(".")[-1].lower()
    if ext not in preferred_exts: ext = preferred_exts[0]
    dst = f"{dst_fullpath_no_ext}.{ext}"
    try:
        os.replace(src, dst)
    except Exception:
        import shutil
        shutil.copy2(src, dst)
        try: os.remove(src)
        except Exception: pass
    # pulizia eventuali residui
    for n in os.listdir(src_dir):
        try: os.remove(os.path.join(src_dir, n))
        except Exception: pass
    return dst

def contiguous_from_zero(indices: list[int]) -> int:
    s = set(indices)
    i = 0
    while i in s: i += 1
    return i

def existing_part_indices(dir_path: str, prefix: str, exts=("mp3","wav","m4a","png","jpg","jpeg")) -> list[int]:
    inds = []
    for n in os.listdir(dir_path):
        if not n.startswith(f"{prefix}_"): continue
        try: name, ext = n.rsplit(".", 1)
        except ValueError: continue
        if ext.lower() not in exts: continue
        m = re.match(rf"{re.escape(prefix)}_(\d+)$", name)
        if m: inds.append(int(m.group(1)))
    return sorted(inds)

# -------------------------------------------------------
# App UI
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
    st.subheader("⚡ Velocità")
    speed_mode = st.selectbox("Modalità", ["🐌 Lenta", "⚡ Veloce", "🚀 Turbo"], index=1)
    if speed_mode == "⚡ Veloce":
        st.session_state["chunk_size"] = 3500; st.session_state["sleep_time"] = 5
    elif speed_mode == "🚀 Turbo":
        st.session_state["chunk_size"] = 5000; st.session_state["sleep_time"] = 2
    else:
        st.session_state["chunk_size"] = 2000; st.session_state["sleep_time"] = 11

    st.divider()
    st.header("🔄 Resume & Sblocco")
    if st.button("🔓 Sblocca progetto corrente"):
        t = st.session_state.get("title", "")
        if t:
            clear_lock(os.path.join("data", "outputs", sanitize(t)))
        st.session_state["is_generating"] = False
        st.success("Sbloccato. Puoi riprendere.")
        st.rerun()

    if st.button("🧹 Sblocca tutti i progetti"):
        base_root = "data/outputs"; n = 0
        if os.path.exists(base_root):
            for name in os.listdir(base_root):
                p = os.path.join(base_root, name, ".generation.lock")
                if os.path.exists(p):
                    try: os.remove(p); n += 1
                    except Exception: pass
        st.session_state["is_generating"] = False
        st.success(f"Sbloccati {n} progetti.")
        st.rerun()

# Stato API
def get_replicate_key() -> str: return (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN","")).strip()
def get_fishaudio_key() -> str: return (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY","")).strip()
def get_fishaudio_voice_id() -> str: return st.session_state.get("fishaudio_voice_id","").strip()
def get_replicate_model() -> str: return st.session_state.get("replicate_model","").strip()
def get_chunk_size() -> int: return st.session_state.get("chunk_size", 2000)
def get_sleep_time() -> float: return st.session_state.get("sleep_time", 11.0)

st.write(
    f"🔎 **Stato API** → Replicate: {'✅' if get_replicate_key() else '⚠️'} · "
    f"FishAudio: {'✅' if get_fishaudio_key() else '⚠️'} · "
    f"Model: `{get_replicate_model() or '—'}` · Voice: `{get_fishaudio_voice_id() or '—'}`"
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

    # ----- Se sono disponibili titolo+script, mostra stato resume + pulsanti DEDICATI -----
    resume_audio_btn_clicked = False
    resume_images_btn_clicked = False
    audio_resume_from = None
    images_resume_from = None

    if title.strip() and script.strip():
        safe = sanitize(title)
        base = os.path.join("data", "outputs", safe)
        aud_dir = os.path.join(base, "audio")
        img_dir = os.path.join(base, "images")
        os.makedirs(aud_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        # AUDIO: carica/crea chunk deterministici e calcola progresso
        audio_chunks_preview = build_or_load_audio_chunks(base, script, get_chunk_size())
        total_a = len(audio_chunks_preview)
        a_indices = existing_part_indices(aud_dir, "part", ("mp3","wav","m4a"))
        a_done = contiguous_from_zero(a_indices)  # 0-based: se vale 55 -> pross. è 55 (== 56 in 1-based)
        st.info(f"🎧 Audio: **{a_done} su {total_a}** chunk creati")
        if a_done < total_a:
            if st.button(f"▶ Continua generazione AUDIO da chunk {a_done+1}", key="resume_audio_btn"):
                st.session_state["resume_mode"] = "audio"
                st.session_state["resume_start_audio_idx"] = a_done
                resume_audio_btn_clicked = True

        # IMMAGINI: se abbiamo image_chunks.json mostriamo stato, altrimenti contiamo i file
        img_chunks_path = os.path.join(base, "image_chunks.json")
        img_chunks_preview = json_list_load(img_chunks_path)
        total_i = len(img_chunks_preview) if img_chunks_preview else None
        i_indices = existing_part_indices(img_dir, "img", ("png","jpg","jpeg"))
        i_done = contiguous_from_zero(i_indices)
        if total_i is not None:
            st.info(f"🖼️ Immagini: **{i_done} su {total_i}** create")
            if i_done < total_i:
                if st.button(f"▶ Continua generazione IMMAGINI da chunk {i_done+1}", key="resume_images_btn"):
                    st.session_state["resume_mode"] = "images"
                    st.session_state["resume_start_image_idx"] = i_done
                    resume_images_btn_clicked = True
        else:
            # non ancora calcolato il planning immagini (es. modalità Entrambi in cui serve l'audio)
            st.info(f"🖼️ Immagini: **{i_done}** create (totale non ancora determinato)")

    generate = st.button("🚀 Genera contenuti", type="primary", use_container_width=True)

with col_timeline:
    st.subheader("📊 Timeline Generazione")
    timeline_container = st.container()
    if not st.session_state.get("is_generating", False):
        with timeline_container: st.info("⏳ Premi 'Genera contenuti' o un pulsante 'Continua…' per iniziare / riprendere")

# -------------------------------------------------------
# TRIGGER: genera se premi "Genera contenuti" o un pulsante di resume
# -------------------------------------------------------
trigger_generate = generate or resume_audio_btn_clicked or resume_images_btn_clicked

# -------------------------------------------------------
# AVVIO GENERAZIONE (driver a chunk singolo + resume + pulsanti dedicati)
# -------------------------------------------------------
if trigger_generate and title.strip() and script.strip():
    # evita doppia generazione
    if st.session_state.get("is_generating", False):
        base_try = os.path.join("data", "outputs", sanitize(st.session_state.get("title","")))
        if base_try and is_lock_stale(base_try):
            clear_lock(base_try); st.session_state["is_generating"] = False
        else:
            st.warning("⏳ Generazione già in corso..."); st.stop()

    # setup e controlli PRIMA del flag
    safe = sanitize(title)
    base = os.path.join("data", "outputs", safe)
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    rep_key = _clean_token(get_replicate_key())
    fish_key = _clean_token(get_fishaudio_key())
    model = get_replicate_model()
    voice = get_fishaudio_voice_id()

    # cosa generare? se pulsante resume forza il tipo
    forced_resume_mode = st.session_state.get("resume_mode")  # "audio" | "images" | None
    effective_mode = mode
    if forced_resume_mode == "audio":
        effective_mode = "Audio"
    elif forced_resume_mode == "images":
        effective_mode = "Immagini"

    if effective_mode in ["Audio", "Entrambi"]:
        if not fish_key: st.error("❌ FishAudio API key mancante!"); st.stop()
        if not voice: st.error("❌ FishAudio Voice ID mancante!"); st.stop()
    if effective_mode in ["Immagini", "Entrambi"]:
        if not rep_key: st.error("❌ Replicate API key mancante!"); st.stop()
        if not model: st.error("❌ Modello Replicate mancante!"); st.stop()

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

    # ORA attiva flag + lock
    st.session_state["is_generating"] = True
    st.session_state["title"] = title
    write_lock(base)

    debug = st.container()
    tracker = ProgressTracker()

    # Costruisci o carica chunk audio deterministici
    audio_chunks = build_or_load_audio_chunks(base, script, runtime_cfg["chunk_size"]) if effective_mode in ["Audio", "Entrambi"] else []
    # stime immagini se solo immagini
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
        # ----------------- AUDIO (chunk singolo con resume & pulsante) -----------------
        if effective_mode in ["Audio", "Entrambi"]:
            step = tracker.add_step("audio", "🎧 Generazione Audio (resume a chunk)")
            display_timeline(tracker, timeline_container)

            # indice di ripartenza: se premuto pulsante resume_audio usa quello, altrimenti autodetect
            if st.session_state.get("resume_mode") == "audio" and st.session_state.get("resume_start_audio_idx") is not None:
                start_idx = int(st.session_state["resume_start_audio_idx"])
            else:
                cp = load_checkpoint(base)
                completed_cp = int(cp.get("audio_completed", 0))
                indices = existing_part_indices(aud_dir, "part", ("mp3","wav","m4a"))
                leading_files = contiguous_from_zero(indices)
                start_idx = max(completed_cp, leading_files)

            total = len(audio_chunks)
            tracker.add_substep(step, f"📦 Ripartenza audio: {start_idx} su {total} (prossimo: chunk {start_idx+1})", "completed")
            display_timeline(tracker, timeline_container)

            for i in range(start_idx, total):
                tmp = os.path.join(aud_dir, "_tmp")
                ensure_empty_dir(tmp)

                _call_generate_audio([audio_chunks[i]], runtime_cfg, tmp, progress_cb=_progress)

                target_noext = os.path.join(aud_dir, f"part_{i:03d}")
                out = move_single_output(tmp, target_noext, preferred_exts=("mp3","wav","m4a"))
                if not out or not os.path.exists(out):
                    tracker.add_substep(step, f"❌ Chunk {i} non prodotto", "failed")
                    save_checkpoint(base, {"audio_completed": i}, merge=True)
                    display_timeline(tracker, timeline_container)
                    st.error(f"Errore nel generare chunk audio {i}")
                    st.stop()

                save_checkpoint(base, {"audio_completed": i+1}, merge=True)
                tracker.add_substep(step, f"✅ {i+1} su {total} creati", "completed")
                display_timeline(tracker, timeline_container)
                touch_progress(base)

            # combine se presenti tutti
            try:
                if len([n for n in os.listdir(aud_dir) if n.startswith("part_") and n.endswith(".mp3")]) == total:
                    combined = os.path.join(aud_dir, "combined_audio.mp3")
                    filelist = os.path.join(aud_dir, "_concat.txt")
                    with open(filelist, "w", encoding="utf-8") as f:
                        for i in range(total):
                            f.write(f"file '{os.path.join(aud_dir, f'part_{i:03d}.mp3')}'\n")
                    try:
                        subprocess.run(
                            ["ffmpeg","-y","-f","concat","-safe","0","-i",filelist,"-c:a","libmp3lame","-b:a","192k",combined],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        if os.path.exists(combined):
                            st.session_state["audio_path"] = combined
                            st.session_state["audio_ready"] = True
                            tracker.add_substep(step, f"🎵 Combinato: {mp3_duration_seconds(combined):.1f}s", "completed")
                    except Exception:
                        tracker.add_substep(step, "ℹ️ ffmpeg non disponibile o combine fallito: restano i part_*.mp3", "completed")
                    finally:
                        try: os.remove(filelist)
                        except Exception: pass
            except Exception as e:
                tracker.add_substep(step, f"⚠️ Combine errore: {e}", "failed")

            tracker.complete_step(step, "completed")
            display_timeline(tracker, timeline_container)

        # ----------------- IMMAGINI (chunk singolo con resume & pulsante) -----------------
        if effective_mode in ["Immagini", "Entrambi"]:
            step = tracker.add_step("images", "🖼️ Generazione Immagini (resume a chunk)")
            display_timeline(tracker, timeline_container)

            if effective_mode == "Entrambi":
                # richiede audio combinato per calcolare il planning immagini
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
                    per_img = max(1, len(sents)//num_images)
                    img_chunks_plan = [" ".join(sents[i:i+per_img]) for i in range(0, len(sents), per_img)]
                json_list_save(os.path.join(base, "image_chunks.json"), img_chunks_plan)
            else:
                if not img_chunks_plan:
                    img_chunks_plan = json_list_load(os.path.join(base, "image_chunks.json")) or []
                    if not img_chunks_plan:
                        # fallback se non presente
                        img_chunks_plan = chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2)))
                        json_list_save(os.path.join(base, "image_chunks.json"), img_chunks_plan)

            total_imgs = len(img_chunks_plan)

            # indice di ripartenza: se premuto pulsante resume_images usa quello, altrimenti autodetect
            if st.session_state.get("resume_mode") == "images" and st.session_state.get("resume_start_image_idx") is not None:
                start_img = int(st.session_state["resume_start_image_idx"])
            else:
                cp = load_checkpoint(base)
                completed_imgs_cp = int(cp.get("images_completed", 0))
                indices = existing_part_indices(img_dir, "img", ("png","jpg","jpeg"))
                leading_files = contiguous_from_zero(indices)
                start_img = max(completed_imgs_cp, leading_files)

            tracker.add_substep(step, f"📦 Ripartenza immagini: {start_img} su {total_imgs} (prossima: {start_img+1})", "completed")
            display_timeline(tracker, timeline_container)

            for i in range(start_img, total_imgs):
                tmp = os.path.join(img_dir, "_tmp")
                ensure_empty_dir(tmp)

                _call_generate_images([img_chunks_plan[i]], runtime_cfg, tmp, progress_cb=_progress)

                target_noext = os.path.join(img_dir, f"img_{i:03d}")
                out = move_single_output(tmp, target_noext, preferred_exts=("png","jpg","jpeg"))
                if not out or not os.path.exists(out):
                    tracker.add_substep(step, f"❌ Immagine {i} non prodotta", "failed")
                    save_checkpoint(base, {"images_completed": i}, merge=True)
                    display_timeline(tracker, timeline_container)
                    st.error(f"Errore nel generare immagine {i}")
                    st.stop()

                save_checkpoint(base, {"images_completed": i+1}, merge=True)
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
        import traceback; st.code(traceback.format_exc())
    finally:
        clear_lock(base)
        st.session_state["is_generating"] = False
        # reset pulsanti resume
        for k in ["resume_mode","resume_start_audio_idx","resume_start_image_idx"]:
            if k in st.session_state: del st.session_state[k]

# ----------------- DOWNLOAD -----------------
st.divider(); st.subheader("📥 Download Files")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### 🎧 Audio")
    if st.session_state.get("audio_ready") and st.session_state.get("audio_path") and os.path.exists(st.session_state["audio_path"]):
        ap = st.session_state["audio_path"]
        try:
            size_mb = os.path.getsize(ap)/(1024*1024)
            dur = mp3_duration_seconds(ap)
            bitrate = (size_mb*8*1024)/dur if dur>0 else 0
            st.info(f"Durata: {dur:.1f}s · Dimensione: {size_mb:.1f} MB · ~{bitrate:.0f} kbps")
        except Exception:
            pass
        with open(ap, "rb") as f:
            st.download_button("Scarica MP3", f.read(), file_name=f"{sanitize(st.session_state.get('title') or 'audio')}.mp3", mime="audio/mpeg", use_container_width=True)
    else:
        st.info("⏳ Nessun audio pronto")
with c2:
    st.markdown("### 🖼️ Immagini")
    if st.session_state.get("zip_ready") and st.session_state.get("zip_path") and os.path.exists(st.session_state["zip_path"]):
        zp = st.session_state["zip_path"]
        with open(zp, "rb") as f:
            st.download_button("Scarica ZIP Immagini", f.read(), file_name=f"{sanitize(st.session_state.get('title') or 'images')}.zip", mime="application/zip", use_container_width=True)
    else:
        st.info("⏳ Nessuna immagine pronta")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;padding:12px'>🎬 Generatore Video AI — Resume a prova di crash • Pulsanti 'Continua da chunk N'</div>", unsafe_allow_html=True)
