# scripts/utils.py
# -------------------------------------------------------
# Utility: chunking testo + IMMAGINI (Replicate) + AUDIO (FishAudio)
# Niente pydub: usiamo mutagen (durate) e ffmpeg (via imageio-ffmpeg) per concatenare.
# Con ritocchi "a prova di blocco": retry/backoff, resume, skip chunk già generati.
# -------------------------------------------------------

import os
import re
import time
import glob
import subprocess
from io import BytesIO
from typing import List

import requests
from PIL import Image

# ============== Chunking ==============
def chunk_text(text: str, max_chars: int):
    """Divide text in blocchi di max_chars, spezzando tra le parole."""
    words = (text or "").split()
    parts, curr, length = [], [], 0
    for w in words:
        add = len(w) + (1 if curr else 0)
        if length + add > max_chars and curr:
            parts.append(" ".join(curr))
            curr, length = [], 0
        curr.append(w)
        length += add
    if curr:
        parts.append(" ".join(curr))
    return parts


def chunk_by_sentence(text: str, max_chars: int):
    """Divide il testo in blocchi di max_chars, spezzando sempre a fine frase (.?!)."""
    sentences = re.split(r'(?<=[.?!])\s+', (text or "").strip())
    parts, curr = [], ""
    for sent in sentences:
        candidate = (curr + " " + sent).strip() if curr else sent.strip()
        if len(candidate) <= max_chars:
            curr = candidate
        else:
            if curr:
                parts.append(curr)
            curr = sent.strip()
    if curr:
        parts.append(curr)
    return parts


def chunk_by_sentences_count(text: str, sentences_per_chunk: int):
    """Divide il testo in blocchi di N frasi."""
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', (text or "").strip()) if s.strip()]
    N = max(1, int(sentences_per_chunk or 1))
    return [" ".join(sentences[i:i + N]) for i in range(0, len(sentences), N)]

# ============== Streamlit helper ==============
def _st():
    try:
        import streamlit as st  # type: ignore
        return st
    except Exception:
        return None

# ============== Immagini ==============
def save_image_from_url(url: str, path: str, timeout: int = 30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    img.save(path)

def _download_first(urls, dest_path: str):
    if not urls:
        raise ValueError("Nessuna URL immagine restituita dal modello.")
    save_image_from_url(urls[0], dest_path)
    return dest_path

def generate_images(chunks, cfg: dict, outdir: str, sleep_between_calls: float = 11.0):
    """
    Genera 1 immagine per ogni elemento di `chunks` usando Replicate.

    Config letta da:
      - API: cfg['replicate_api_token'] | cfg['replicate_api_key'] | env REPLICATE_API_TOKEN
      - Modello: cfg['image_model'] | cfg['replicate_model']
      - Extra input: cfg['replicate_input'] (dict, opzionale)
      - aspect_ratio default: cfg['aspect_ratio'] o '16:9'
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    api_key = (
        (cfg or {}).get("replicate_api_token")
        or (cfg or {}).get("replicate_api_key")
        or os.getenv("REPLICATE_API_TOKEN")
    )
    model = (cfg or {}).get("image_model") or (cfg or {}).get("replicate_model")
    extra_input = (cfg or {}).get("replicate_input", {})

    if not api_key:
        msg = "Replicate API token assente. Imposta 'replicate_api_token' o 'replicate_api_key' in cfg, o REPLICATE_API_TOKEN in env."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not model:
        msg = "Modello Replicate assente. Imposta 'image_model' o 'replicate_model' (es. 'owner/name:tag')."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    import replicate
    client = replicate.Client(api_token=api_key)

    masked = (api_key[:3] + "…" + api_key[-4:]) if len(api_key) > 8 else "—"
    if st:
        st.write(f"🔐 Token Replicate in uso: {masked}")
        st.write(f"🧩 Modello Replicate: `{model}`")
    else:
        print(f"[INFO] Replicate token: {masked}")
        print(f"[INFO] Using Replicate model: {model}")

    results = []
    for idx, prompt in enumerate(chunks, start=1):
        model_input = {"prompt": prompt}
        model_input.setdefault("aspect_ratio", (cfg or {}).get("aspect_ratio", "16:9"))
        if isinstance(extra_input, dict):
            model_input.update(extra_input)

        try:
            output = client.run(model, input=model_input)

            urls = []
            if isinstance(output, str):
                urls = [output]
            elif isinstance(output, list):
                urls = output
            elif isinstance(output, dict):
                for k in ("image", "images", "output", "url", "urls"):
                    v = output.get(k)
                    if isinstance(v, str):
                        urls = [v]; break
                    if isinstance(v, list) and v:
                        urls = v; break

            outpath = os.path.join(outdir, f"img_{idx:03d}.png")
            _download_first(urls, outpath)
            results.append(outpath)

            if st:
                st.write(f"✅ Immagine {idx} generata: `{os.path.basename(outpath)}`")
            else:
                print(f"[OK] Saved {outpath}")

        except Exception as e:
            try:
                ReplicateError = replicate.exceptions.ReplicateError
            except Exception:
                ReplicateError = Exception
            msg = f"ReplicateError su chunk {idx}: {e}" if isinstance(e, ReplicateError) else f"Errore su chunk {idx}: {e}"
            if st: st.error("❌ " + msg)
            else: print("[ERROR]", msg)
            raise

        if sleep_between_calls and idx < len(chunks):
            time.sleep(sleep_between_calls)

    return results

# ============== Audio (FishAudio) senza pydub ==============

# Durata MP3
def mp3_duration_seconds(path: str) -> float:
    """Ritorna la durata in secondi usando mutagen."""
    try:
        from mutagen.mp3 import MP3
        return float(MP3(path).info.length)
    except Exception:
        # fallback: 0 se non disponibile
        return 0.0

# Concatenazione MP3 con ffmpeg (portabile via imageio-ffmpeg)
def concat_mp3s(paths: List[str], out_path: str, bitrate_kbps: int = 128):
    """
    Concatena MP3 usando ffmpeg. Per robustezza, ricodifichiamo a libmp3lame.
    """
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        raise RuntimeError("Nessun file MP3 da concatenare.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # prepara file di lista per concat demuxer
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            abspath = os.path.abspath(p)
            f.write(f"file '{abspath}'\n")

    # trova ffmpeg portabile
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"  # prova sistema

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c:a", "libmp3lame",
        "-b:a", f"{bitrate_kbps}k",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {proc.stderr.decode(errors='ignore')[:500]}")

    try:
        os.remove(list_path)
    except Exception:
        pass

def _download_with_retry(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            # backoff con jitter
            time.sleep(1.5 + attempt * 1.0)
    raise last_exc or RuntimeError("Download fallito")

def _pad_width(n: int) -> int:
    """Larghezza zero-pad in base al numero di chunk (99->2, 1000->4, ecc.)."""
    return max(2, len(str(max(1, n))))

def generate_audio(
    chunks: List[str],
    cfg: dict,
    outdir: str,
    tts_endpoint: str = "https://api.fish.audio/v1/tts",
    retries_per_chunk: int = 4,
    base_backoff: float = 2.0,
    sleep_between_chunks: float = 1.2,
):
    """
    Genera audio da ogni blocco di testo in `chunks` con FishAudio e li concatena.

    Config:
      - cfg['fishaudio_api_key']               (obbligatoria)
      - cfg['fishaudio_voice'] | cfg['fishaudio_voice_id']   (obbligatoria)
      - cfg['fishaudio_model']                 (opzionale; se c'è la inviamo)
      - cfg['fishaudio_extra']                 (dict opzionale; es. format/bitrate)
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    api_key = (cfg or {}).get("fishaudio_api_key")
    voice_id = (cfg or {}).get("fishaudio_voice") or (cfg or {}).get("fishaudio_voice_id")
    model = (cfg or {}).get("fishaudio_model")
    extra = (cfg or {}).get("fishaudio_extra", {})

    if not api_key:
        msg = "FishAudio API key assente. Imposta 'fishaudio_api_key' in cfg."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not voice_id:
        msg = "FishAudio Voice ID assente. Imposta 'fishaudio_voice' o 'fishaudio_voice_id' in cfg."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    session = requests.Session()  # connessioni riutilizzate

    pad = _pad_width(len(chunks))
    audio_paths: List[str] = []

    for i, text in enumerate(chunks, 1):
        # path del pezzo corrente (resume/skip)
        part_path = os.path.join(outdir, f"audio_{i:0{pad}d}.mp3")

        # se esiste e ha durata > 0.5s, salta (resume automatico)
        if os.path.exists(part_path) and mp3_duration_seconds(part_path) > 0.5:
            if st: st.write(f"⏭️ Skip chunk {i} (già presente)")
            audio_paths.append(part_path)
            continue

        if st:
            st.write(f"🎧 Audio {i}/{len(chunks)}…")
        else:
            print(f" • Audio {i}/{len(chunks)}…", end=" ")

        payload = {
            "text": text,
            "reference_id": voice_id,
            "format": "mp3",
            "mp3_bitrate": 128,
        }
        if model:
            payload["model"] = model
        if isinstance(extra, dict):
            payload.update(extra)

        success = False
        last_err = None

        for attempt in range(retries_per_chunk):
            try:
                resp = session.post(tts_endpoint, headers=headers, json=payload, timeout=90)
                # Gestione status non 2xx
                if resp.status_code == 413:
                    raise RuntimeError("Payload troppo grande (413). Riduci i chunk (es. 3000 caratteri).")
                if resp.status_code in (429, 502, 503, 504):
                    raise RuntimeError(f"Rate limit/temporaneo ({resp.status_code}).")
                resp.raise_for_status()

                ct = resp.headers.get("Content-Type", "")
                if "application/json" in ct:
                    data = resp.json()
                    audio_url = data.get("audio_url") or data.get("url")
                    if not audio_url:
                        raise RuntimeError(f"Risposta JSON inattesa: {str(data)[:200]}")
                    audio_bytes = _download_with_retry(audio_url, retries=3, timeout=60)
                else:
                    audio_bytes = resp.content

                # salva su disco
                with open(part_path, "wb") as f:
                    f.write(audio_bytes)

                # verifica durata > 0
                dur_s = mp3_duration_seconds(part_path)
                if dur_s <= 0.5:
                    raise RuntimeError("File MP3 generato ma con durata nulla (possibile errore).")

                if st:
                    st.write(f"✅ TTS chunk {i:0{pad}d} — durata: {int(dur_s*1000)} ms")
                else:
                    print(f"[OK {i}] {int(dur_s*1000)} ms")

                success = True
                break

            except Exception as e:
                last_err = e
                # backoff progressivo con un po’ di jitter
                sleep_s = base_backoff * (attempt + 1) + (0.3 * attempt)
                if st:
                    st.warning(f"⚠️ Retry {attempt+1}/{retries_per_chunk} per chunk {i}: {e} (sleep {sleep_s:.1f}s)")
                time.sleep(sleep_s)

        if not success:
            # se fallisce, NON blocchiamo tutto: passiamo al prossimo
            if st:
                st.error(f"❌ Fallito chunk {i}: {last_err}")
            else:
                print(f"[FAIL {i}] {last_err}")
        else:
            audio_paths.append(part_path)

        # piccola pausa tra i chunk per non saturare l'API
        time.sleep(sleep_between_chunks)

    # Se non abbiamo generato nulla, errore
    audio_paths = [p for p in audio_paths if os.path.exists(p)]
    if not audio_paths:
        raise RuntimeError("Nessun audio generato. (Controlla API key/voice ID e dimensione chunk).")

    # Concatena TUTTI i pezzi trovati (anche se qualcuno è saltato)
    out_path = os.path.join(outdir, "combined_audio.mp3")
    # ordina per numero
    audio_paths_sorted = sorted(audio_paths)
    concat_mp3s(audio_paths_sorted, out_path, bitrate_kbps=128)

    total_ms = int(mp3_duration_seconds(out_path) * 1000)
    if st:
        st.write(f"🔊 Durata totale audio: {total_ms} ms")
        st.success("🔊 Audio finale creato.")
    else:
        print(f"[TTS] combined duration: {total_ms} ms")
        print("🔊 Audio finale creato.")

    return out_path
