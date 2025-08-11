#!/usr/bin/env python3
import os, time, replicate
from .config_loader import load_config
from .utils         import load_blocks, save_image

def main():
    cfg = load_config()
    replicate.api_token = cfg["replicate_api_token"]

    title = input("🎬 Titolo del video (per i blocchi): ").strip()
    # genera il nome "safe"
    safe = (title
        .lower()
        .replace(" ", "_")
        .replace("ù","u").replace("à","a").replace("è","e")
        .replace("ì","i").replace("ò","o").replace("é","e")
        .replace("’","").replace("'","")
    )

    blocks_dir = os.path.join("data","outputs","blocks", safe)
    images_dir = os.path.join("data","outputs","images", safe)
    os.makedirs(images_dir, exist_ok=True)

    blocks = load_blocks(blocks_dir)
    for i, b in enumerate(blocks, 1):
        print(f"🎨 Genero immagine per BLOCCO {i}…")
        out = replicate.run(cfg["image_model"], input={"prompt": b})
        url = out[0]
        time.sleep(1)  # evita rate‐limit
        save_image(url, os.path.join(images_dir, f"block_{i}.png"))

if __name__=="__main__":
    main()
