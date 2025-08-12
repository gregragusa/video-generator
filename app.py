# app.py
# -------------------------------------------------------
# Streamlit app: API e parametri (modello/voce), genera IMMAGINI / AUDIO.
# Niente pydub: usiamo durata MP3 via scripts.utils.mp3_duration_seconds (mutagen)
# + concat MP3 con ffmpeg se generiamo chunk audio lunghi.
# Inoltre: generazione IMMAGINI in batch con retry e progress bar.
# -------------------------------------------------------

import os
import re
import time
import glob
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
    generate_audio,
    generate_images,
    mp3_duration_seconds,  # legge durata MP3 senza caricarlo in RAM
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
    # rimuove whitespace/newline invisibili
    return re.sub(r"\s+", "", (tok or ""))

def _mask(tok: str) -> str:
    t = (tok or "").strip()
    return t[:3] + "…" + t[-4:] if len(t) > 8 else "—"

def ffmpeg_concat_mp3(aud_dir: str, out_path: str) -> bool:
    """Se troviamo part_*.mp3 li concateniamo senza ricodifica."""
    parts = sorted(glob.glob(os.path.join(aud_dir, "part_*.mp3")))
    if not parts:
        return False
    list_path = os.path.join(aud_dir, "concat_list.txt")
    with open(list_path, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", out_path
    ]
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return os.path.exists(out_path)
    except Exception as e:
        # Mostra un messaggio non bloccante
        st.warning(f"ffmpeg concat fallita: {e}")
        return False

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
    f"Model(Immagini): `{rep_model}` · VoiceID(Audio): `{voice_id}`"
)

# ===========================
# 🎛️ Parametri generazione (centrale)
# ===========================
title = st.text_input("Titolo del video")
script = st.text_area("Inserisci il testo da usare per generare immagini/audio", height=300)
mode = st.selectbox("Cosa vuoi generare?", ["Immagini", "Audio", "Entrambi"])

# Input condizionali
if mode in ["Audio", "Entrambi"]:
    seconds_per_img = st.number_input(
        "Ogni quanti secondi di audio creare un'immagine?",
        min_value=1, value=30, step=1  # default 30s per la tua richiesta
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

    # ---- AUDIO ----
    if mode in ["Audio", "Entrambi"]:
        if not fish_ok:
            st.error("❌ FishAudio API key mancante. Inseriscila nella sidebar.")
        elif not get_fishaudio_voice_id():
            st.error("❌ FishAudio Voice ID mancante. Inseriscilo nella sidebar.")
        else:
            st.text(f"🎧 Generazione audio con voce: {get_fishaudio_voice_id()} …")
            # Chunk più piccoli per evitare timeout su audio lunghi
            # (se il tuo TTS consente 30k caratteri puoi rialzare)
            aud_chunks = chunk_text(script, 8000)
            # La tua funzione genera i file. Idealmente: part_0001.mp3, part_0002.mp3, ...
            generate_audio(aud_chunks, runtime_cfg, aud_dir)

            # Se non esiste il file combinato, proviamo a concatenare eventuali part_*.mp3
            if not os.path.exists(audio_path):
                concatenated = ffmpeg_concat_mp3(aud_dir, audio_path)
                if concatenated:
                    st.success("🔗 Audio concatenato con ffmpeg.")
                else:
                    # come fallback resterà duration=0 e gestiremo sotto
                    st.warning("⚠️ Non ho trovato file da concatenare. Assicurati che generate_audio produca part_*.mp3 o combined_audio.mp3.")

    # ---- IMMAGINI ----
    if mode in ["Immagini", "Entrambi"]:
        if not rep_ok:
            st.error("❌ Replicate API key mancante. Inseriscila nella sidebar.")
        elif not get_replicate_model():
            st.error("❌ Modello Replicate mancante. Seleziona un preset o inserisci un Custom model.")
        else:
            if mode == "Entrambi":
                # serve l'audio per calcolare le immagini in base ai secondi
                if not os.path.exists(audio_path):
                    # se ci sono part_*.mp3, prova a concatenare adesso
                    if ffmpeg_concat_mp3(aud_dir, audio_path):
                        st.info("🔗 Ho creato l'audio combinato ora per calcolare la durata.")
                if not os.path.exists(audio_path):
                    st.error("❌ Audio non trovato per calcolare il numero di immagini. Genera prima l’audio.")
                else:
                    st.text(f"🖼️ Generazione immagini con modello: {get_replicate_model()} (in base alla durata audio)…")
                    try:
                        duration_sec = mp3_duration_seconds(audio_path)
                    except Exception:
                        duration_sec = 0
                    if not duration_sec:
                        # fallback prudente
                        duration_sec = 60.0

                    # Numero immagini senza limiti (es: 3h / 30s = 360 immagini)
                    num_images = max(1, int(round(duration_sec / float(seconds_per_img))))

                    # Spezzetta lo script in blocchi testuali bilanciati
                    approx_chars = max(50, len(script) // max(1, num_images))
                    chunks = chunk_text(script, approx_chars)

                    st.text(f"🖼️ Devo generare {len(chunks)} immagini totali…")
                    # Batch + retry + progress
                    batch_size = 8  # invia 8 alla volta per non saturare l'API
                    total = len(chunks)
                    done = 0
                    prog = st.progress(0.0)

                    for i in range(0, total, batch_size):
                        batch = chunks[i:i+batch_size]
                        for attempt in range(3):
                            try:
                                generate_images(batch, runtime_cfg, img_dir)
                                break
                            except Exception as e:
                                if attempt == 2:
                                    st.error(f"Errore batch {i//batch_size+1}: {e}")
                                time.sleep(2 + attempt * 3)  # piccoli backoff
                        done += len(batch)
                        prog.progress(min(1.0, done/total))
                        time.sleep(0.2)

                    zip_images(base)
            else:
                # SOLO IMMAGINI → raggruppo per frasi e faccio batch + progress
                st.text(f"🖼️ Generazione immagini con modello: {get_replicate_model()} (per frasi)…")
                groups = chunk_by_sentences_count(script, int(sentences_per_image))
                st.text(f"🖼️ Devo generare {len(groups)} immagini (1 ogni {int(sentences_per_image)} frasi)…")

                batch_size = 8
                total = len(groups)
                done = 0
                prog = st.progress(0.0)

                for i in range(0, total, batch_size):
                    batch = groups[i:i+batch_size]
                    for attempt in range(3):
                        try:
                            generate_images(batch, runtime_cfg, img_dir)
                            break
                        except Exception as e:
                            if attempt == 2:
                                st.error(f"Errore batch {i//batch_size+1}: {e}")
                            time.sleep(2 + attempt * 3)
                    done += len(batch)
                    prog.progress(min(1.0, done/total))
                    time.sleep(0.2)

                zip_images(base)

    st.success("✅ Generazione completata!")

    # salva percorsi in sessione per i download
    st.session_state["audio_path"] = audio_path if os.path.exists(audio_path) else None
    zip_path = os.path.join(base, "output.zip")
    st.session_state["zip_path"] = zip_path if os.path.exists(zip_path) else None

# ---- Download ----
if st.session_state.get("audio_path") and os.path.exists(st.session_state["audio_path"]):
    with open(st.session_state["audio_path"], "rb") as f:
        st.download_button("🎧 Scarica Audio MP3", f, file_name="audio.mp3", mime="audio/mpeg")

if st.session_state.get("zip_path") and os.path.exists(st.session_state["zip_path"]):
    with open(st.session_state["zip_path"], "rb") as f:
        st.download_button("🖼️ Scarica ZIP Immagini", f, file_name="output.zip", mime="application/zip")
