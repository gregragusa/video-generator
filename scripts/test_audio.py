#!/usr/bin/env python3
from scripts.config_loader import load_config
from scripts.utils import chunk_text, generate_audio

def main():
    cfg = load_config()
    text = "Questo è un test per verificare la voce."
    chunks = chunk_text(text, 30000)
    outdir = "data/outputs/test_audio_only"
    generate_audio(chunks, cfg, outdir)
    print("✅ Test audio salvato in:", outdir)

if __name__ == "__main__":
    main()

