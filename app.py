# app.py
# -------------------------------------------------------
# 2 step reali (Streamlit-friendly):
# 1) AUDIO (a passate, resume) → download SUBITO
# 2) Start images (a passate, resume, 1 per volta con retry 429)
# -------------------------------------------------------

import os
import re
import time
import glob
import requests
import streamlit as st

try:
    from scripts.config_loader import load_config  # opzionale
except Exception:
    load_config = None

from scripts.utils import (
    chunk_text,
    chunk_by_sentences_count,
    generate_audio,
    generate_images,
    mp3_duration_seconds,
)

# ---------------------------
# Utility
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

# ---------------------------
# Pagina
# ---------------------------
st.set_page_config(page_title="Generatore Video", page_icon="🎬", layout="centered")
st.title("🎬 Generatore di Video con Immagini e Audio")

# Config opzionale
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

    with st.form("api_keys_form", clear_on_submit=False):
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

# Helper per stati
def get_replicate_key() -> str:
    return (st.session_state.get("replicate_api_key") or os.environ.get("REPLICATE_API_TOKEN", "")).strip()

def get_fishaudio_key() -> str:
    return (st.session_state.get("fish_audio_api_key") or os.environ.get("FISHAUDIO_API_KEY", "")).strip()

def get_fishaudio_voice_id() -> str:
    return (st.session_state.get("fishaudio_voice_id", "")).strip()

def get_replicate_model() -> str:
    return (st.session_state.get("replicate_model", "")).strip()

rep_ok = bool(get_replicate_key())
fish_ok = bool(get_fishaudio_key())
rep_model = get_replicate_model() or "—"
voice_id = get_fishaudio_voice_id() or "—"
st.write(
    f"🔎 Stato API → Replicate: {'✅' if rep_ok else '⚠️'} · FishAudio: {'✅' if fish_ok else '⚠️'} · "
    f"Model(Immagini): `{rep_model}` · VoiceID(Audio): `{voice_id}`"
)

# ===========================
# 🎛️ Parametri centrali
# ===========================
title = st.text_input("Titolo del video")
script = st.text_area("Inserisci il testo da usare per generare immagini/audio", height=300)

# Audio: quanti chunk in questa passata (Streamlit Cloud friendly)
audio_parts_this_run = st.number_input(
    "Quanti chunk audio generare in questa passata?",
    min_value=1, value=12, step=1
)

# Immagini: quanti file in questa passata
seconds_per_img = st.number_input("Ogni quanti secondi di audio creare un'immagine?", min_value=1, value=30, step=1)
images_this_run = st.number_input(
    "Quante immagini generare in questa passata?",
    min_value=1, value=80, step=1
)

# Bottoni separati (2 step)
btn_audio = st.button("🎧 Generate audio (this run)")
btn_images = st.button("🖼️ Start images (this run)")

# ===========================
# Path base
# ===========================
safe = sanitize(title)
base = os.path.join("data", "outputs", safe) if safe else None
if safe:
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    audio_path = os.path.join(aud_dir, "combined_audio.mp3")

# ===========================
# 🔑 Prepara cfg runtime
# ===========================
def make_runtime_cfg():
    runtime_cfg = dict(base_cfg)
    rep = _clean_token(get_replicate_key())
    fish = _clean_token(get_fishaudio_key())
    if rep:
        os.environ["REPLICATE_API_TOKEN"] = rep
        runtime_cfg["replicate_api_key"] = rep
        runtime_cfg["replicate_api_token"] = rep
    if fish:
        os.environ["FISHAUDIO_API_KEY"] = fish
        runtime_cfg["fishaudio_api_key"] = fish

    model = get_replicate_model()
    if model:
        runtime_cfg["replicate_model"] = model
    voice = get_fishaudio_voice_id()
    if voice:
        runtime_cfg["fishaudio_voice_id"] = voice
    return runtime_cfg

# ===========================
# 🎧 AUDIO (step 1)
# ===========================
if btn_audio:
    if not (title.strip() and script.strip()):
        st.error("Titolo e testo sono obbligatori.")
    elif not fish_ok:
        st.error("❌ FishAudio API key mancante (sidebar).")
    elif not get_fishaudio_voice_id():
        st.error("❌ FishAudio Voice ID mancante (sidebar).")
    else:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(aud_dir, exist_ok=True)

        runtime_cfg = make_runtime_cfg()

        st.subheader("🎧 Generazione audio…")
        aud_chunks = chunk_text(script, 3000)  # più robusto per lavori lunghi
        st.text(f"Chunk totali: {len(aud_chunks)}  ·  Questa passata: {audio_parts_this_run}")

        # Genera SOLO un certo numero di parti nuove (resume automatico dentro)
        generate_audio(
            aud_chunks,
            runtime_cfg,
            aud_dir,
            retries_per_chunk=6,
            base_backoff=3.0,
            sleep_between_chunks=2.0,
            max_parts_this_run=int(audio_parts_this_run),
            combine=True
        )

        # Download immediato (parziale o totale)
        if os.path.exists(audio_path):
            st.session_state["audio_path"] = audio_path
            with open(audio_path, "rb") as f:
                st.download_button("🎧 Scarica Audio MP3 (subito)", f, file_name="audio.mp3", mime="audio/mpeg")
        else:
            st.warning("⚠️ Audio non trovato (verifica la cartella output).")

# ===========================
# 🖼️ IMMAGINI (step 2)
# ===========================
if btn_images:
    if not (title.strip() and script.strip()):
        st.error("Titolo e testo sono obbligatori.")
    elif not rep_ok:
        st.error("❌ Replicate API key mancante (sidebar).")
    elif not get_replicate_model():
        st.error("❌ Modello Replicate mancante (sidebar).")
    else:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(aud_dir, exist_ok=True)

        runtime_cfg = make_runtime_cfg()

        # Calcola i chunk testuali per le immagini
        if os.path.exists(audio_path):
            st.text("Calcolo numero immagini dalla durata dell'audio…")
            try:
                duration_sec = mp3_duration_seconds(audio_path)
            except Exception:
                duration_sec = 0
            if not duration_sec:
                duration_sec = 60.0
            num_images = max(1, int(round(duration_sec / float(seconds_per_img))))
            approx_chars = max(50, len(script) // max(1, num_images))
            all_chunks = chunk_text(script, approx_chars)
        else:
            # fallback (se audio non c'è): 1 img ogni ~2 frasi
            all_chunks = chunk_by_sentences_count(script, 2)

        existing = len([f for f in os.listdir(img_dir) if f.startswith("img_") and f.endswith(".png")])
        start_index = existing + 1
        to_generate = all_chunks[existing : existing + int(images_this_run)]

        if not to_generate:
            st.info("✅ Non ci sono immagini da generare in questa passata (sei già al totale).")
        else:
            st.subheader("🖼️ Generazione immagini… (passata)")
            st.text(f"Esistenti: {existing}  ·  Nuove in questa passata: {len(to_generate)}  ·  Indici: {start_index:03d}–{start_index+len(to_generate)-1:03d}")

            generate_images(
                to_generate,
                runtime_cfg,
                img_dir,
                start_index=start_index,
                sleep_between_calls=0.0,   # gestione 429 dinamica
                retries=7,
                base_backoff=2.0
            )

            zip_images(base)
            zip_path = os.path.join(base, "output.zip")
            if os.path.exists(zip_path):
                st.session_state["zip_path"] = zip_path
                with open(zip_path, "rb") as f:
                    st.download_button("🖼️ Scarica ZIP Immagini (parziale)", f, file_name="output.zip", mime="application/zip")

# ---- Download (riappare a fondo pagina) ----
if safe:
    if os.path.exists(os.path.join(aud_dir, "combined_audio.mp3")):
        st.session_state["audio_path"] = os.path.join(aud_dir, "combined_audio.mp3")
    if os.path.exists(os.path.join(base, "output.zip")):
        st.session_state["zip_path"] = os.path.join(base, "output.zip")

if st.session_state.get("audio_path") and os.path.exists(st.session_state["audio_path"]):
    with open(st.session_state["audio_path"], "rb") as f:
        st.download_button("🎧 Scarica Audio MP3", f, file_name="audio.mp3", mime="audio/mpeg")

if st.session_state.get("zip_path") and os.path.exists(st.session_state["zip_path"]):
    with open(st.session_state["zip_path"], "rb") as f:
        st.download_button("🖼️ Scarica ZIP Immagini", f, file_name="output.zip", mime="application/zip")

