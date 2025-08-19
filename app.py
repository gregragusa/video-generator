# app.py
# -------------------------------------------------------
# Streamlit app: API e parametri (modello/voce), genera IMMAGINI / AUDIO.
# Compatibile con Python 3.13: niente pydub; usiamo mutagen + ffmpeg via imageio-ffmpeg.
# Con ripresa progressi AUDIO (resume): salva manifest + chunk MP3 numerati.
# -------------------------------------------------------

import os
import re
import time
import json
import hashlib
import shutil
import textwrap
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
    generate_audio,       # lo riusiamo per generare un singolo chunk alla volta
    generate_images,
    mp3_duration_seconds, # util per leggere durata MP3
)

# imageio-ffmpeg per trovare ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = "ffmpeg"  # fallback: deve essere nel PATH

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
def split_text_into_sentence_chunks(text: str, max_chars: int = 2000):
    """
    Divide il testo in blocchi di circa max_chars caratteri,
    spezzando dopo . ! ? dove possibile.
    Se una singola frase supera max_chars, la spezza duramente.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []

    sentences = re.split(r"(?<=[\.\!\?])\s+", t)
    chunks, acc = [], ""

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
            flush_acc()
            acc = s

    flush_acc()
    return [c for c in chunks if c]

# ---------------------------
# Manifest + concat MP3
# ---------------------------
def manifest_path(aud_dir: str) -> str:
    return os.path.join(aud_dir, "audio_manifest.json")

def load_manifest(aud_dir: str) -> dict | None:
    p = manifest_path(aud_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_manifest(aud_dir: str, data: dict) -> None:
    p = manifest_path(aud_dir)
    tmp = p + ".tmp"
    os.makedirs(aud_dir, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def chunk_filename(aud_dir: str, idx: int) -> str:
    return os.path.join(aud_dir, f"chunk_{idx:04d}.mp3")

def _ffconcat_escape(path: str) -> str:
    """
    Escapa backslash e apici singoli per il file di lista ffmpeg (-f concat).
    Scriveremo: file '<percorso-escapato>'
    """
    # raddoppia backslash e escapa l'apice singolo come \'
    return path.replace("\\", "\\\\").replace("'", "\\'")

def concat_mp3_chunks(aud_dir: str, out_path: str, total_chunks: int) -> bool:
    """
    Concatena con ffmpeg (demuxer concat).
