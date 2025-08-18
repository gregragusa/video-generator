# app.py
# -------------------------------------------------------
# Streamlit app: genera IMMAGINI / AUDIO con Replicate + FishAudio.
# Compatibile con Python 3.13: niente pydub; ffmpeg via imageio-ffmpeg lato utils.
# VERSIONE con TIMELINE, CHECKPOINT e RESUME ROBUSTI + ANTI-STALLO
# (Compatibile con utils.py vecchio o nuovo tramite wrapper)
# -------------------------------------------------------

import os
import re
import json
import time
import requests
from datetime import datetime
import streamlit as st

# se hai questo loader lo usiamo, altrimenti proseguiamo senza
try:
    from scripts.config_loader import load_config  # opzionale
except Exception:
    load_config = None

# Utils (funzionano sia vecchi che nuovi; i wrapper sotto gestiscono progress_cb)
from scripts.utils import (
    chunk_text,
    chunk_by_sentences_count,
    chunk_text_for_audio,   # chunking specifico per audio
    generate_audio,
    generate_images,
    mp3_duration_seconds,   # util per leggere durata MP3
)

# Proviamo a importare load_checkpoint/save_checkpoint se esistono (nuova versione)
try:
    from scripts.utils import load_checkpoint, save_checkpoint
except Exception:
    # fallback minimi per non rompere se il tuo utils è vecchio
    def load_checkpoint(base_dir: str):
        try:
            path = os.path.join(base_dir, "checkpoint.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    def save_checkpoint(base_dir: str, updates: dict, merge: bool = True):
        try:
            os.makedirs(base_dir, exist_ok=True)
            path = os.path.join(base_dir, "checkpoint.json")
            if merge and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            else:
                state = {}
            state.update(updates or {})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

# ---- COMPAT WRAPPERS: accettano sia utils vecchi (senza progress_cb) che nuovi ----
def _call_generate_audio(chunks, cfg, out_dir, progress_cb):
    try:
        # utils nuovo
        return generate_audio(chunks, cfg, out_dir, progress_cb=progress_cb)
    except TypeError:
        # utils vecchio (senza progress_cb)
        return generate_audio(chunks, cfg, out_dir)

def _call_generate_images(chunks, cfg, out_dir, progress_cb):
    try:
        # utils nuovo
        return generate_images(chunks, cfg, out_dir, progress_cb=progress_cb)
    except TypeError:
        # utils vecchio (senza progress_cb)
        return generate_images(chunks, cfg, out_dir)

# ---------------------------
# Timeline Tracker System
# ---------------------------

class ProgressTracker:
    """Sistema di tracking timeline per generazione"""
    def __init__(self):
        self.start_time = None
        self.steps = []
        self.current_step = None
        self.estimated_total_seconds = 0

    def start(self, total_audio_chunks: int, total_images: int):
        self.start_time = datetime.now()
        # Stime basate su esperienza reale
        audio_estimate = (total_audio_chunks or 0) * 12  # ~12s per chunk audio
        image_estimate = (total_images or 0) * 18        # ~18s per immagine
        self.estimated_total_seconds = audio_estimate + image_estimate
        self.steps = []

    def add_step(self, step_type: str, description: str, status: str = "running"):
        step = {
            "type": step_type,
            "description": description,
            "status": status,  # running, completed, failed
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None,
            "substeps": []
        }
        self.steps.append(step)
        self.current_step = len(self.steps) - 1
        return self.current_step

    def add_substep(self, step_index: int, description: str, status: str = "completed"):
        if 0 <= step_index < len(self.steps):
            substep = {
                "description": description,
                "status": status,
                "timestamp": datetime.now()
            }
            self.steps[step_index]["substeps"].append(substep)

    def complete_step(self, step_index: int, status: str = "completed"):
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]["end_time"] = datetime.now()
            self.steps[step_index]["status"] = status
            if self.steps[step_index]["start_time"]:
                duration = self.steps[step_index]["end_time"] - self.steps[step_index]["start_time"]
                self.steps[step_index]["duration"] = duration.total_seconds()

    def get_elapsed_time(self):
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0

    def get_eta(self):
        elapsed = self.get_elapsed_time()
        if elapsed < 30:
            return max(0, self.estimated_total_seconds - elapsed)
        completed_steps = len([s for s in self.steps if s["status"] == "completed"])
        total_steps = len(self.steps)
        if completed_steps > 0 and total_steps > 0:
            avg_time_per_step = elapsed / completed_steps
            remaining_steps = total_steps - completed_steps
            return max(0, remaining_steps * avg_time_per_step)
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            elapsed = tracker.get_elapsed_time()
            st.metric("⏱️ Trascorso", f"{elapsed/60:.1f} min")
        with col2:
            eta = tracker.get_eta()
            st.metric("🎯 ETA", f"{eta/60:.1f} min")
        with col3:
            total_estimate = (elapsed + eta) / 60
            st.metric("📊 Totale Stimato", f"{total_estimate:.1f} min")
        with col4:
            completed = len([s for s in tracker.steps if s["status"] == "completed"])
            st.metric("✅ Completati", f"{completed}/{len(tracker.steps)}")  # FIX corretto

        progress = tracker.get_completion_percentage() / 100
        st.progress(progress, text=f"Progresso generale: {progress*100:.1f}%")

        st.markdown("### 📋 Timeline Dettagliata")
        for i, step in enumerate(tracker.steps):
            if step["status"] == "completed":
                icon = "✅"; style = ""
            elif step["status"] == "failed":
                icon = "❌"; style = ""
            elif step["status"] == "running":
                icon = "🔄"; style = "**"
            else:
                icon = "⏳"; style = ""

            if step["duration"]:
                time_str = f"({step['duration']:.1f}s)"
            elif step["status"] == "running":
                running_time = (datetime.now() - step["start_time"]).total_seconds()
                time_str = f"({running_time:.1f}s...)"
            else:
                time_str = ""

            step_text = f"{icon} {style}{step['description']}{style} {time_str}"
            st.markdown(step_text)

            if step["substeps"]:
                substeps_to_show = step["substeps"][-3:] if step["status"] == "running" else step["substeps"][-1:]
                for substep in substeps_to_show:
                    substep_icon = "✅" if substep["status"] == "completed" else "❌"
                    st.markdown(f"   └ {substep_icon} {substep['description']}")

# ---------------------------
# Utility / Lock Helpers
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
        for filename in os.listdir(img_dir):
            full_path = os.path.join(img_dir, filename)
            if os.path.isfile(full_path):
                zipf.write(full_path, arcname=os.path.join("images", filename))
    return zip_path

def _clean_token(tok: str) -> str:
    return re.sub(r"\s+", "", (tok or ""))

def _mask(tok: str) -> str:
    t = (tok or "").strip()
    return t[:3] + "…" + t[-4:] if len(t) > 8 else "—"

def validate_replicate_model(model_name: str, api_key: str) -> bool:
    if not model_name or not api_key:
        return False
    try:
        if ":" not in model_name:
            model_name += ":latest"
        owner, name_version = model_name.split("/", 1)
        name = name_version.split(":")[0]
        resp = requests.get(
            f"https://api.replicate.com/v1/models/{owner}/{name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if resp.status_code == 200:
            model_info = resp.json()
            st.success(f"✅ Modello '{model_name}' trovato e accessibile")
            st.info(f"📝 Descrizione: {model_info.get('description', 'N/A')[:100]}...")
            return True
        elif resp.status_code == 404:
            st.error(f"❌ Modello '{model_name}' NON TROVATO (404)")
            st.markdown("### 🔄 Modelli Verificati Funzionanti:")
            st.code("black-forest-labs/flux-schnell")
            st.code("stability-ai/stable-diffusion-xl-base-1.0")
            st.code("bytedance/sdxl-lightning-4step")
            return False
        else:
            st.warning(f"⚠️ Modello '{model_name}' - Status: {resp.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ Errore validazione modello: {e}")
        return False

# ---- LOCK SYSTEM (anti-stallo) ----
GEN_TIMEOUT_SECS = 10 * 60  # 10 minuti; regola a piacere

def _lock_path(base_dir: str) -> str:
    return os.path.join(base_dir, ".generation.lock")

def write_lock(base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    data = {"start_ts": time.time(), "last_progress_ts": time.time()}
    with open(_lock_path(base_dir), "w", encoding="utf-8") as f:
        json.dump(data, f)

def touch_progress(base_dir: str):
    try:
        path = _lock_path(base_dir)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"start_ts": time.time()}
        data["last_progress_ts"] = time.time()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def is_lock_stale(base_dir: str, timeout_secs: int = GEN_TIMEOUT_SECS) -> bool:
    path = _lock_path(base_dir)
    try:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        last = float(data.get("last_progress_ts") or data.get("start_ts") or 0.0)
        return (time.time() - last) > timeout_secs
    except Exception:
        # se non riesco a leggere, considero stantio
        return True

def clear_lock(base_dir: str):
    try:
        os.remove(_lock_path(base_dir))
    except Exception:
        pass

# ---------------------------
# Streamlit App Setup
# ---------------------------

st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="wide")
st.title("🎬 Generatore di Video con Immagini e Audio")

# Se la sessione risulta "in corso", prova a sbloccare se il lock è stantio
if st.session_state.get("is_generating", False):
    t = st.session_state.get("title", "")
    if t:
        base_check = os.path.join("data", "outputs", sanitize(t))
        if is_lock_stale(base_check):
            clear_lock(base_check)
            st.session_state["is_generating"] = False
            st.info("🔧 Sessione precedente risultava bloccata: sbloccata automaticamente.")

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
    st.caption("Le chiavi valgono solo per **questa sessione** del browser.")

    rep_prefill = st.session_state.get("replicate_api_key", "")
    fish_prefill = st.session_state.get("fish_audio_api_key", "")

    with st.form("api_keys_form_main", clear_on_submit=False):
        replicate_key = st.text_input(
            "Replicate API key",
            type="password",
            value=rep_prefill,
            placeholder="r8_************************",
            help="Necessaria per generare IMMAGINI (Replicate)"
        )
        fish_key = st.text_input(
            "FishAudio API key",
            type="password",
            value=fish_prefill,
            placeholder="fa_************************",
            help="Necessaria per generare AUDIO (FishAudio)"
        )
        save_keys = st.form_submit_button("💾 Salva API Keys")

    if save_keys:
        st.session_state["replicate_api_key"] = replicate_key.strip()
        st.session_state["fish_audio_api_key"] = fish_key.strip()
        st.success("Chiavi salvate nella sessione!")

    st.subheader("🔎 Verifica token Replicate")
    if st.button("🔍 Verifica Token Replicate", key="verify_token_btn"):
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
                st.error(f"❌ Errore chiamando l'API: {e}")

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

    # Modello Replicate con modelli verificati
    model_presets = [
        "black-forest-labs/flux-schnell",
        "black-forest-labs/flux-dev",
        "stability-ai/stable-diffusion-xl-base-1.0",
        "bytedance/sdxl-lightning-4step",
        "playgroundai/playground-v2.5-1024px-aesthetic",
        "Custom (digita sotto)",
    ]
    preset_selected = st.selectbox(
        "Modello Replicate (image generator)",
        model_presets,
        index=0,
        help="Scegli un preset VERIFICATO oppure 'Custom' e inserisci il nome esatto del modello sotto."
    )

    custom_prefill = st.session_state.get("replicate_model_custom", "")
    custom_model = st.text_input(
        "Custom model (owner/name:tag)",
        value=custom_prefill,
        placeholder="es. owner/model-name:latest",
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

    # Test modello Replicate
    st.subheader("🧪 Test Modello Replicate")
    current_model = effective_model
    if current_model and st.button("🧪 Test Modello", key="test_model_btn"):
        rep_key = st.session_state.get("replicate_api_key", "").strip()
        if not rep_key:
            st.error("❌ API key Replicate mancante")
        else:
            st.write(f"🔍 Testing: `{current_model}`")
            validate_replicate_model(current_model, rep_key)

    # Ottimizzazioni velocità
    st.divider()
    st.subheader("⚡ Ottimizzazioni Velocità")
    speed_mode = st.selectbox("Modalità velocità", [
        "🐌 Lenta ma sicura (default)",
        "⚡ Veloce (raccomandato)",
        "🚀 Turbo (sperimentale)"
    ])
    if speed_mode == "⚡ Veloce (raccomandato)":
        st.session_state["chunk_size"] = 3500
        st.session_state["sleep_time"] = 5
    elif speed_mode == "🚀 Turbo (sperimentale)":
        st.session_state["chunk_size"] = 5000
        st.session_state["sleep_time"] = 2
    else:
        st.session_state["chunk_size"] = 2000
        st.session_state["sleep_time"] = 11

    # Gestione Resume/Checkpoint e Sblocco
    st.divider()
    st.header("🔄 Resume & Sblocco")

    # Bottone sblocco per il titolo corrente
    if st.button("🔓 Sblocca generazione bloccata (titolo corrente)"):
        t = st.session_state.get("title", "")
        if t:
            base_unlock = os.path.join("data", "outputs", sanitize(t))
            clear_lock(base_unlock)
        st.session_state["is_generating"] = False
        st.success("Sblocco eseguito. Puoi riprendere la generazione.")
        st.rerun()

    # Bottone sblocco globale (tutti i lock)
    if st.button("🧹 Sblocca tutti i progetti bloccati"):
        base_root = "data/outputs"
        cleared = 0
        if os.path.exists(base_root):
            for name in os.listdir(base_root):
                p = os.path.join(base_root, name, ".generation.lock")
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        cleared += 1
                    except Exception:
                        pass
        st.session_state["is_generating"] = False
        st.success(f"Sbloccati {cleared} progetti.")
        st.rerun()

    # Mostra checkpoint esistenti
    data_dir = "data/outputs"
    if os.path.exists(data_dir):
        import glob
        checkpoints = glob.glob(os.path.join(data_dir, "*/checkpoint.json"))
        if checkpoints:
            st.write(f"📁 Checkpoint attivi: {len(checkpoints)}")
            with st.expander("Dettagli checkpoint"):
                shown = 0
                for cp in checkpoints:
                    if shown >= 5:
                        st.caption("…")
                        break
                    try:
                        with open(cp, 'r', encoding="utf-8") as f:
                            data = json.load(f)
                        project_name = os.path.basename(os.path.dirname(cp))
                        audio_progress = data.get('audio_completed', 0)
                        images_progress = data.get('images_completed', 0)
                        st.write(f"• **{project_name}**: Audio {audio_progress}, Immagini {images_progress}")
                        shown += 1
                    except Exception:
                        pass

# Funzioni per recuperare stati
def get_replicate_key() -> str:
    return (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN", "")).strip()

def get_fishaudio_key() -> str:
    return (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY", "")).strip()

def get_fishaudio_voice_id() -> str:
    return (st.session_state.get("fishaudio_voice_id", "")).strip()

def get_replicate_model() -> str:
    return (st.session_state.get("replicate_model", "")).strip()

def get_chunk_size() -> int:
    return st.session_state.get("chunk_size", 2000)

def get_sleep_time() -> float:
    return st.session_state.get("sleep_time", 11.0)

# Badge di stato
rep_ok = bool(get_replicate_key())
fish_ok = bool(get_fishaudio_key())
rep_model = get_replicate_model() or "—"
voice_id = get_fishaudio_voice_id() or "—"
st.write(
    f"🔎 **Stato API** → Replicate: {'✅' if rep_ok else '⚠️'} · FishAudio: {'✅' if fish_ok else '⚠️'} · "
    f"Model: `{rep_model}` · Voice: `{voice_id}`"
)

# ===========================
# 🎛️ Main Interface
# ===========================
col_main, col_timeline = st.columns([2, 3])

with col_main:
    st.subheader("📝 Input")

    title = st.text_input("Titolo del video")
    script = st.text_area("Inserisci il testo da usare per generare immagini/audio", height=200)
    mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio", "Entrambi"])

    if mode in ["Audio", "Entrambi"]:
        seconds_per_img = st.number_input(
            "Ogni quanti secondi di audio creare un'immagine?",
            min_value=1, value=8, step=1
        )
        st.session_state["seconds_per_img"] = seconds_per_img
    else:  # Solo Immagini
        sentences_per_image = st.number_input(
            "Ogni quante frasi creare un'immagine?",
            min_value=1, value=2, step=1
        )
        st.session_state["sentences_per_image"] = sentences_per_image

    # Info script con controllo resume
    if script:
        char_count = len(script)
        word_count = len(script.split())
        if title.strip():
            safe = sanitize(title)
            base = os.path.join("data", "outputs", safe)
            checkpoint = load_checkpoint(base)
            # Mostra stato parziale se presente
            if checkpoint and (checkpoint.get("audio_completed") or checkpoint.get("images_completed")):
                st.warning(f"""
                🔄 **LAVORO IN CORSO RILEVATO**
                - Audio: {checkpoint.get('audio_completed', 0)} chunk completati
                - Immagini: {checkpoint.get('images_completed', 0)} completate  
                - Premi 'Genera' per **continuare da dove interrotto**
                """)

        st.info(f"📊 Script: {char_count:,} caratteri | {word_count:,} parole")

        if char_count > 100000:
            st.warning(f"⚠️ Script molto lungo! Generazione stimata: {char_count/10000:.1f}-{char_count/5000:.1f} minuti")
            st.info("💡 **Tip**: Il sistema salva automaticamente i progressi. Se si interrompe, riavvia per continuare.")

    generate = st.button("🚀 Genera contenuti", type="primary", use_container_width=True, key="generate_btn")

with col_timeline:
    st.subheader("📊 Timeline Generazione")
    timeline_container = st.container()
    if not st.session_state.get("is_generating", False):
        with timeline_container:
            st.info("⏳ Premi 'Genera contenuti' per iniziare la timeline")

# ===========================
# 🚀 Avvio generazione (con anti-stallo)
# ===========================
if generate and title.strip() and script.strip():
    # Se un'altra sessione è rimasta appesa ma lo stop è vecchio, sblocca
    if st.session_state.get("is_generating", False):
        base_try = os.path.join("data", "outputs", sanitize(st.session_state.get("title","")))
        if base_try and is_lock_stale(base_try):
            clear_lock(base_try)
            st.session_state["is_generating"] = False
        else:
            st.warning("⏳ Generazione già in corso...")
            st.stop()

    # --- Setup e CONTROLLI PRIMA del flag ---
    safe = sanitize(title)
    base = os.path.join("data", "outputs", safe)
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)
    audio_path = os.path.join(aud_dir, "combined_audio.mp3")

    runtime_cfg = dict(base_cfg)
    replicate_from_ui = _clean_token(get_replicate_key())
    fishaudio_from_ui = _clean_token(get_fishaudio_key())
    replicate_model  = get_replicate_model()
    fish_voice       = get_fishaudio_voice_id()

    # prerequisiti
    if mode in ["Audio", "Entrambi"]:
        if not fishaudio_from_ui:
            st.error("❌ FishAudio API key mancante!")
            st.stop()
        if not fish_voice:
            st.error("❌ FishAudio Voice ID mancante!")
            st.stop()
    if mode in ["Immagini", "Entrambi"]:
        if not replicate_from_ui:
            st.error("❌ Replicate API key mancante!")
            st.stop()
        if not replicate_model:
            st.error("❌ Modello Replicate mancante!")
            st.stop()

    # Config in env
    if replicate_from_ui:
        os.environ["REPLICATE_API_TOKEN"] = replicate_from_ui
        runtime_cfg["replicate_api_key"] = replicate_from_ui
        runtime_cfg["replicate_api_token"] = replicate_from_ui
    if fishaudio_from_ui:
        os.environ["FISHAUDIO_API_KEY"] = fishaudio_from_ui
        runtime_cfg["fishaudio_api_key"] = fishaudio_from_ui
    if replicate_model:
        runtime_cfg["replicate_model"] = replicate_model
    if fish_voice:
        runtime_cfg["fishaudio_voice_id"] = fish_voice
    runtime_cfg["chunk_size"] = st.session_state.get("chunk_size", 2000)
    runtime_cfg["sleep_time"] = st.session_state.get("sleep_time", 11.0)

    # --- SOLO ORA attiva il flag e scrivi il lock ---
    st.session_state["is_generating"] = True
    st.session_state["title"] = title
    write_lock(base)

    # avvia tracker
    tracker = ProgressTracker()
    chunk_size = runtime_cfg["chunk_size"]
    aud_chunks = chunk_text_for_audio(script, target_chars=chunk_size) if mode in ["Audio", "Entrambi"] else []

    if mode == "Entrambi":
        estimated_audio_duration = (len(script) / 5) / 150 * 60
        estimated_images = max(1, int(estimated_audio_duration // st.session_state.get("seconds_per_img", 8)))
    elif mode == "Immagini":
        estimated_images = len(chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2))))
    else:
        estimated_images = 0

    tracker.start(len(aud_chunks), estimated_images)
    display_timeline(tracker, timeline_container)
    st.success(f"🎯 **Generazione Iniziata!** Stimati {len(aud_chunks)} chunk audio + {estimated_images} immagini")

    # Progress callback per aggiornare timeline + lock
    def _progress(msg: str):
        touch_progress(base)
        if tracker.current_step is not None:
            tracker.add_substep(tracker.current_step, msg, "completed")
            display_timeline(tracker, timeline_container)

    debug_container = st.container()

    try:
        # ---- AUDIO ----
        if mode in ["Audio", "Entrambi"]:
            with debug_container:
                st.write("🎧 **Iniziando generazione AUDIO...**")
            step_idx = tracker.add_step("audio", f"🎧 Generazione Audio ({len(aud_chunks)} segmenti)")
            display_timeline(tracker, timeline_container)
            tracker.add_substep(step_idx, f"📝 Creati {len(aud_chunks)} chunk da ~{chunk_size} caratteri", "completed")
            display_timeline(tracker, timeline_container)

            final_audio = _call_generate_audio(aud_chunks, runtime_cfg, aud_dir, _progress)
            if final_audio and os.path.exists(final_audio):
                audio_path = final_audio
                duration = mp3_duration_seconds(audio_path)
                tracker.complete_step(step_idx, "completed")
                tracker.steps[step_idx]["description"] = f"🎵 Audio Completato ({duration:.1f}s = {duration/60:.1f}min)"
                tracker.add_substep(step_idx, f"🔊 Audio finale: {duration:.1f}s", "completed")
            else:
                tracker.complete_step(step_idx, "failed")
                tracker.add_substep(step_idx, "❌ Generazione audio fallita", "failed")
                display_timeline(tracker, timeline_container)
                st.error("⚠️ Audio non generato.")
                st.stop()

            display_timeline(tracker, timeline_container)

        # ---- IMMAGINI ----
        if mode in ["Immagini", "Entrambi"]:
            with debug_container:
                st.write("🖼️ **Iniziando generazione IMMAGINI...**")
            if mode == "Entrambi":
                step_idx = tracker.add_step("images", f"🖼️ Generazione Immagini (basata su durata audio)")
            else:
                step_idx = tracker.add_step("images", f"🖼️ Generazione Immagini ({estimated_images} immagini)")
            display_timeline(tracker, timeline_container)

            if mode == "Entrambi":
                if not os.path.exists(audio_path):
                    tracker.complete_step(step_idx, "failed")
                    tracker.add_substep(step_idx, "❌ Audio non trovato", "failed")
                    display_timeline(tracker, timeline_container)
                    st.error("❌ Audio non trovato per calcolare le immagini.")
                    st.stop()

                seconds_per_img = st.session_state.get("seconds_per_img", 8)
                duration_sec = mp3_duration_seconds(audio_path) or 60
                num_images = max(1, int(duration_sec // seconds_per_img))
                tracker.add_substep(step_idx, f"📊 Audio {duration_sec:.1f}s → {num_images} immagini", "completed")

                if num_images == 1:
                    img_chunks = [script]
                else:
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', script.strip()) if s.strip()]
                    sentences_per_image_calc = max(1, len(sentences) // num_images)
                    img_chunks = []
                    for i in range(0, len(sentences), sentences_per_image_calc):
                        chunk_sentences = sentences[i:i + sentences_per_image_calc]
                        img_chunks.append(" ".join(chunk_sentences))

                tracker.steps[step_idx]["description"] = f"🖼️ Generazione {len(img_chunks)} Immagini"
                display_timeline(tracker, timeline_container)
                _call_generate_images(img_chunks, runtime_cfg, img_dir, _progress)
                zip_images(base)
                tracker.complete_step(step_idx, "completed")
            else:
                groups = chunk_by_sentences_count(script, int(st.session_state.get("sentences_per_image", 2)))
                tracker.add_substep(step_idx, f"📝 Creati {len(groups)} gruppi di {int(st.session_state.get('sentences_per_image', 2))} frasi", "completed")
                tracker.steps[step_idx]["description"] = f"🖼️ Generazione {len(groups)} Immagini"
                display_timeline(tracker, timeline_container)
                _call_generate_images(groups, runtime_cfg, img_dir, _progress)
                zip_images(base)
                tracker.complete_step(step_idx, "completed")

            display_timeline(tracker, timeline_container)

        # Finalizzazione
        final_step = tracker.add_step("finalize", "🎉 Finalizzazione e Packaging")
        display_timeline(tracker, timeline_container)

        files_created = []
        if os.path.exists(audio_path):
            st.session_state["audio_path"] = audio_path
            st.session_state["audio_ready"] = True
            files_created.append("Audio MP3")
            tracker.add_substep(final_step, "💾 Audio MP3 salvato", "completed")

        zip_path = os.path.join(base, "output.zip")
        if os.path.exists(zip_path):
            st.session_state["zip_path"] = zip_path
            st.session_state["zip_ready"] = True
            files_created.append("ZIP Immagini")
            tracker.add_substep(final_step, "📦 ZIP Immagini creato", "completed")

        tracker.complete_step(final_step, "completed")
        tracker.steps[final_step]["description"] = f"🎉 Completato! Files: {', '.join(files_created)}"
        display_timeline(tracker, timeline_container)

        total_time = tracker.get_elapsed_time()
        st.balloons()
        st.success(f"✅ **Generazione completata in {total_time/60:.1f} minuti!**")

        # Summary
        with st.expander("📊 Statistiche Dettagliate", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Tempo Totale", f"{total_time/60:.1f} min")
                audio_steps = [s for s in tracker.steps if s["type"] == "audio" and s.get("duration")]
                if audio_steps:
                    st.metric("🎧 Tempo Audio", f"{sum(s['duration'] for s in audio_steps)/60:.1f} min")
            with col2:
                completed_steps = len([s for s in tracker.steps if s["status"] == "completed"])
                st.metric("✅ Step Completati", f"{completed_steps}/{len(tracker.steps)}")
                image_steps = [s for s in tracker.steps if s["type"] == "images" and s.get("duration")]
                if image_steps:
                    st.metric("🖼️ Tempo Immagini", f"{sum(s['duration'] for s in image_steps)/60:.1f} min")
            with col3:
                if mode in ["Audio", "Entrambi"]:
                    st.metric("🎵 Chunk Audio", len(aud_chunks))
                if mode in ["Immagini", "Entrambi"]:
                    try:
                        import zipfile
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            files = [f for f in zf.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]
                            st.metric("🎨 Immagini", len(files))
                    except Exception:
                        pass

    except Exception as e:
        st.error(f"💥 **ERRORE PRINCIPALE**: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # sempre ripulisci lock e flag
        clear_lock(base)
        st.session_state["is_generating"] = False

# ===========================
# 📥 DOWNLOAD SECTION (Sempre visibile)
# ===========================
st.divider()
st.subheader("📥 Download Files")

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.markdown("### 🎧 Audio")
    if st.session_state.get("audio_ready") and st.session_state.get("audio_path"):
        audio_path = st.session_state["audio_path"]
        if os.path.exists(audio_path):
            try:
                size_mb = os.path.getsize(audio_path) / (1024*1024)
                duration = mp3_duration_seconds(audio_path)
                bitrate = (size_mb * 8 * 1024) / duration if duration > 0 else 0
                st.info(f"""
                📊 **Dettagli Audio:**
                - Durata: {duration:.1f}s ({duration/60:.1f} min)
                - Dimensione: {size_mb:.1f} MB
                - Bitrate: ~{bitrate:.0f} kbps
                """)
            except Exception:
                st.info("📊 File audio disponibile")
            with open(audio_path, "rb") as f:
                st.download_button(
                    "🎧 Scarica Audio MP3",
                    f.read(),
                    file_name=f"{sanitize(st.session_state.get('title') or 'audio')}.mp3",
                    mime="audio/mpeg",
                    key="download-audio-main",
                    use_container_width=True
                )
        else:
            st.session_state["audio_ready"] = False
            st.error("❌ File audio non trovato")
    else:
        st.info("⏳ Nessun audio generato ancora")

with download_col2:
    st.markdown("### 🖼️ Immagini")
    if st.session_state.get("zip_ready") and st.session_state.get("zip_path"):
        zip_path = st.session_state["zip_path"]
        if os.path.exists(zip_path):
            try:
                size_mb = os.path.getsize(zip_path) / (1024*1024)
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    files = [f for f in zf.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]
                    img_count = len(files)
                    avg_size_mb = size_mb / img_count if img_count > 0 else 0
                st.info(f"""
                📊 **Dettagli Immagini:**
                - Numero: {img_count} immagini
                - Dimensione totale: {size_mb:.1f} MB
                - Dimensione media: {avg_size_mb:.2f} MB/img
                """)
            except Exception:
                st.info("📊 ZIP immagini disponibile")
            with open(zip_path, "rb") as f:
                st.download_button(
                    "🖼️ Scarica ZIP Immagini",
                    f.read(),
                    file_name=f"{sanitize(st.session_state.get('title') or 'images')}.zip",
                    mime="application/zip",
                    key="download-zip-main",
                    use_container_width=True
                )
        else:
            st.session_state["zip_ready"] = False
            st.error("❌ File ZIP non trovato")
    else:
        st.info("⏳ Nessuna immagine generata ancora")

# Controlli download
st.markdown("---")
col_clear, col_info = st.columns([1, 2])
with col_clear:
    if st.session_state.get("audio_ready") or st.session_state.get("zip_ready"):
        if st.button("🗑️ Pulisci Download", help="Rimuove i file dalla lista download", use_container_width=True, key="clear_downloads_btn"):
            st.session_state["audio_ready"] = False
            st.session_state["zip_ready"] = False
            st.session_state.pop("audio_path", None)
            st.session_state.pop("zip_path", None)
            st.success("🗑️ Download puliti!")
            st.rerun()
with col_info:
    if st.session_state.get("audio_ready") or st.session_state.get("zip_ready"):
        st.info("💡 I file rimangono disponibili fino a quando non premi 'Pulisci Download' o riavvii l'app")

# ===========================
# 🏠 Footer
# ===========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    🎬 <strong>Generatore Video AI</strong> |
    Powered by Replicate + FishAudio |
    Timeline real-time + Sistema Resume + Anti-stallo
</div>
""", unsafe_allow_html=True)

