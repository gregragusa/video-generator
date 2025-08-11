#!/usr/bin/env python3
import os
import zipfile
import time
from config_loader import load_config
from utils import chunk_text, generate_images, generate_audio

def sanitize(title):
    s = title.lower()
    for a, b in [(" ", "_"), ("ù", "u"), ("à", "a"), ("è", "e"),
                 ("ì", "i"), ("ò", "o"), ("é", "e")]:
        s = s.replace(a, b)
    return "".join(ch for ch in s if ch.isalnum() or ch == "_")

def zip_output(base_dir):
    zip_path = os.path.join(base_dir, "output.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        print(f"📦 Creo ZIP in: {zip_path}")

        # Aggiungi immagini
        img_dir = os.path.join(base_dir, "images")
        if os.path.exists(img_dir):
            for filename in os.listdir(img_dir):
                full_path = os.path.join(img_dir, filename)
                print(f"➕ Aggiungo immagine: {filename}")
                zipf.write(full_path, arcname=os.path.join("images", filename))
        else:
            print("❌ Cartella immagini NON trovata")

        # Aggiungi TUTTI i file audio
        aud_dir = os.path.join(base_dir, "audio")
        if os.path.exists(aud_dir):
            for filename in os.listdir(aud_dir):
                full_path = os.path.join(aud_dir, filename)
                print(f"➕ Aggiungo audio: {filename}")
                zipf.write(full_path, arcname=os.path.join("audio", filename))
        else:
            print("❌ Cartella audio NON trovata!")

        # Script opzionale
        if os.path.exists("script.txt"):
            print("➕ Aggiungo script.txt")
            zipf.write("script.txt", arcname="script.txt")

    print(f"✅ ZIP finale creato: {zip_path}")

def main():
    cfg = load_config()
    title = input("🎬 Titolo del video: ").strip()
    safe = sanitize(title)

    mode = input("🔧 Cosa vuoi generare? (1=solo immagini, 2=solo audio, 3=entrambi): ").strip()
    if mode not in {"1", "2", "3"}:
        print("❌ Scelta non valida. Esco.")
        return

    img_freq = input("🖼️ Ogni quanti secondi vuoi un'immagine? (default 8): ").strip()
    try:
        seconds_per_img = int(img_freq)
    except:
        seconds_per_img = 8

    with open("script.txt", "r", encoding="utf-8") as f:
        script = f.read()

    base = os.path.join("data", "outputs", safe)
    img_dir = os.path.join(base, "images")
    aud_dir = os.path.join(base, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    if mode in {"1", "3"}:
        print("📷 Generazione immagini in corso…")
        est_wpm = 160
        total_words = len(script.split())
        duration_sec = total_words / (est_wpm / 60)
        img_count = max(1, int(duration_sec // seconds_per_img))
        img_chunks = chunk_text(script, len(script) // img_count)
        generate_images(img_chunks, cfg, img_dir)

    if mode in {"2", "3"}:
        print("🎧 Generazione audio in corso…")
        aud_chunks = chunk_text(script, 30000)
        generate_audio(aud_chunks, cfg, aud_dir)

    time.sleep(2)
    zip_output(base)

    print(f"\n✅ Completato. Trovi tutto in: {base}")

if __name__ == "__main__":
    main()
