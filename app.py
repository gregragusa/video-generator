
import os
import streamlit as st
from pydub import AudioSegment

from scripts.config_loader import load_config
from scripts.utils import (
    chunk_text,
    chunk_by_sentence,
    chunk_by_sentences_count,
    generate_audio,
    generate_images
)

# --- Autenticazione semplice ---
PASSWORD = "odioprogrammare88"  # Password condivisa con l'utente autorizzato

st.sidebar.title("🔒 Login")
password_input = st.sidebar.text_input("Inserisci la password", type="password")
if password_input != PASSWORD:
    st.sidebar.error("🔑 Password errata")
    st.stop()

# --- Funzioni di utilità ---
def sanitize(title: str) -> str:
    s = title.lower()
    for a, b in [(" ", "_"), ("ù", "u"), ("à", "a"),
                 ("è", "e"), ("ì", "i"), ("ò", "o"), ("é", "e")]:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum() or ch == "_")


def zip_images(base_dir: str) -> str:
    import zipfile
    zip_path = os.path.join(base_dir, "output.zip")
    img_dir = os.path.join(base_dir, "images")
    if not os.path.isdir(img_dir):
        return None
    with zipfile.ZipFile(zip_path, "w") as z:
        for fn in os.listdir(img_dir):
            z.write(os.path.join(img_dir, fn), arcname=os.path.join("images", fn))
    return zip_path

# --- Streamlit UI ---
st.set_page_config(page_title="Generatore Video", layout="centered")
st.title("🎬 Generatore di Video con Immagini e Audio")

# Carica configurazione (API keys lette da variabili d'ambiente)
cfg = load_config()

# Override API keys dalla UI
custom_fish = st.text_input("FishAudio API Key (lascia vuoto per default)", type="password")
if custom_fish:
    cfg["fishaudio_api_key"] = custom_fish

custom_repl = st.text_input("Replicate API Token (lascia vuoto per default)", type="password")
if custom_repl:
    cfg["replicate_api_token"] = custom_repl

custom_model = st.text_input("Replicate model (es. black-forest-labs/flux-schnell)", value=cfg.get("image_model", ""))
if custom_model:
    cfg["image_model"] = custom_model

# Input principali
title = st.text_input("Titolo del video")
script = st.text_area("Testo da usare", height=300)
mode = st.selectbox("Modalità", ["Immagini", "Audio", "Entrambi"])

# Parametri dinamici
if mode == "Immagini":
    sentences_per_img = st.number_input("Quante frasi per immagine?", 1, 10, 1)
elif mode == "Entrambi":
    seconds_per_img = st.number_input("Ogni quanti secondi cambiare immagine?", 1, 60, 8)

# Generazione contenuti
if st.button("🚀 Genera contenuti") and title and script:
    safe = sanitize(title)
    base = os.path.join("data", "outputs", safe)
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    audio_path = os.path.join(aud_dir, "combined_audio.mp3")
    zip_path = os.path.join(base, "output.zip")

    st.subheader("🔄 Generazione in corso…")

    # Audio\    
    if mode in ["Audio", "Entrambi"]:
        st.text("🎧 Generazione audio…")
        aud_chunks = chunk_by_sentence(script, 2000)
        generate_audio(aud_chunks, cfg, aud_dir)

    # Immagini
    if mode in ["Immagini", "Entrambi"]:
        st.text("🖼️ Generazione immagini…")
        if mode == "Entrambi":
            audio = AudioSegment.from_file(audio_path)
            duration_sec = len(audio) / 1000.0
            num_images = max(1, int(duration_sec // seconds_per_img))
            img_chunks = chunk_text(script, max(1, len(script) // num_images))
        else:
            img_chunks = chunk_by_sentences_count(script, sentences_per_img)
        generate_images(img_chunks, cfg, img_dir)
        zip_images(base)

    st.success("✅ Generazione completata!")

    # Download
    if os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            st.download_button("🎧 Scarica Audio MP3", f, file_name="combined_audio.mp3", mime="audio/mpeg")
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button("🖼️ Scarica ZIP Immagini", f, file_name="output.zip", mime="application/zip")

