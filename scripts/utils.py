# scripts/utils.py
# -------------------------------------------------------
# Utility: chunking testo + IMMAGINI (Replicate) + AUDIO (FishAudio)
# Senza pydub: durate via mutagen, concatenazione via ffmpeg (imageio-ffmpeg).
# Robusto: retry/backoff, resume, skip di pezzi già fatti.
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

# ============== Immagini (Replicate) ==============
def save_image_from_url(url: str, path: str, timeout: int = 45):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    img.save(path, format="PNG")

def _download_first(urls, dest_path: str):
    if not urls:
        raise ValueError("Nessuna URL immagine restituita dal modello.")
    save_image_from_url(urls[0], dest_path)
    return dest_path

def generate_images(
    chunks,
    cfg: dict,
    outdir: str,
    start_index: int = 1,
    sleep_between_calls: float = 0.0,  # usiamo attesa dinamica su 429
    retries: int = 7,
    base_backoff: float = 2.0,
):
    """
    Genera 1 immagine per elemento in `chunks` usando Replicate.
    - Resume: se il file esiste, salta
    - 429-aware: estrae "~Xs" dal messaggio e attende quel tempo (+margine), poi riprova
    - Numerazione stabile con start_index
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
        msg = "Replicate API token assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not model:
        msg = "Modello Replicate assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    import replicate, re
    client = replicate.Client(api_token=api_key)

    results = []
    for j, prompt in enumerate(chunks, start=start_index):
        outpath = os.path.join(outdir, f"img_{j:03d}.png")
        if os.path.exists(outpath):
            if st: st.write(f"⏭️ Skip image {j:03d} (già presente)")
            results.append(outpath)
            continue

        model_input = {"prompt": prompt}
        model_input.setdefault("aspect_ratio", (cfg or {}).get("aspect_ratio", "16:9"))
        if isinstance(extra_input, dict):
            model_input.update(extra_input)

        success, last_err = False, None
        attempt = 0
        while attempt < retries:
            try:
                output = client.run(model, input=model_input)

                # normalizza output in lista URL
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

                _download_first(urls, outpath)
                if st: st.write(f"✅ Immagine {j:03d} salvata")
                results.append(outpath)
                success = True
                break

            except Exception as e:
                last_err = e
                msg = str(e)

                # 429: estrai "~9s" e dormi quel tempo + 2s
                if "429" in msg:
                    wait = 12
                    m = re.search(r"~(\d+)s", msg)
                    if m:
                        try:
                            wait = int(m.group(1)) + 2
                        except Exception:
                            wait = 12
                    if st: st.warning(f"⚠️ Rate limit: aspetto {wait}s e riprovo (img {j:03d})")
                    time.sleep(wait)
                    # NON contiamo questo come tentativo “vero”
                    continue

                # altri errori → backoff crescente
                attempt += 1
                sleep_s = base_backoff * attempt
                if st: st.warning(f"⚠️ Retry {attempt}/{retries} img {j:03d}: {e} (sleep {sleep_s:.1f}s)")
                time.sleep(sleep_s)

        if not success:
            if st: st.error(f"❌ Fallita img {j:03d}: {last_err}")

        if sleep_between_calls:
            time.sleep(sleep_between_calls)

    return results

# ============== Audio (FishAudio) ==============

def mp3_duration_seconds(path: str) -> float:
    """Durata MP3 (secondi) via mutagen."""
    try:
        from mutagen.mp3 import MP3
        return float(MP3(path).info.length)
    except Exception:
        return 0.0

def concat_mp3s(paths: List[str], out_path: str, bitrate_kbps: int = 128):
    """Concatena MP3 con ffmpeg (ricodifica libmp3lame per robustezza)."""
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        raise RuntimeError("Nessun file MP3 da concatenare.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            abspath = os.path.abspath(p)
            f.write(f"file '{abspath}'\n")

    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {proc.stderr.decode(errors='ignore')[:500]}")
    try:
        os.remove(list_path)
    except Exception:
        pass

def _download_with_retry(url: str, retries: int = 3, timeout: int = 45) -> bytes:
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(1.5 + attempt * 1.0)
    raise last_exc or RuntimeError("Download fallito")

def _pad_width(n: int) -> int:
    """Zero-pad per i nomi file (99->2, 1000->4, ecc.)."""
    return max(2, len(str(max(1, n))))

def generate_audio(
    chunks: List[str],
    cfg: dict,
    outdir: str,
    tts_endpoint: str = "https://api.fish.audio/v1/tts",
    retries_per_chunk: int = 6,
    base_backoff: float = 3.0,
    sleep_between_chunks: float = 2.0,
    start_index: int | None = None,
    max_parts_this_run: int | None = None,
    combine: bool = True,
):
    """
    Genera audio in parti numerate (part_0001.mp3, …) con resume automatico.
    - Se il job si interrompe, basta rilanciare: salta i part già presenti.
    - max_parts_this_run: quante NUOVE parti creare in questa passata (Streamlit Cloud friendly).
    - combine: se True, crea/aggiorna combined_audio.mp3 concatenando TUTTE le part presenti.
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

    session = requests.Session()

    # Resume: trova il prossimo indice
    existing_parts = sorted(glob.glob(os.path.join(outdir, "part_*.mp3")))
    next_seq = len(existing_parts) + 1
    if start_index and start_index > next_seq:
        next_seq = start_index

    # calcola da quale chunk partire (0-based)
    start_chunk_idx = next_seq - 1
    remaining_chunks = chunks[start_chunk_idx:]
    pad = _pad_width(len(chunks) + len(existing_parts) + 5)

    if st:
        st.write(f"▶️ Resume audio: parti esistenti {len(existing_parts)}; genero da part {next_seq:0{pad}d}")

    new_generated = 0
    for text in remaining_chunks:
        if max_parts_this_run and new_generated >= max_parts_this_run:
            break

        part_path = os.path.join(outdir, f"part_{next_seq:0{pad}d}.mp3")

        # se esiste ed è valido, salta
        if os.path.exists(part_path) and mp3_duration_seconds(part_path) > 0.5:
            if st: st.write(f"⏭️ Skip part {next_seq:0{pad}d} (già presente)")
            next_seq += 1
            continue

        payload = {
            "text": text,
            "reference_id": voice_id,
            "format": "mp3",
            "mp3_bitrate": 128,
        }
        if model: payload["model"] = model
        if isinstance(extra, dict): payload.update(extra)

        success, last_err = False, None
        for attempt in range(retries_per_chunk):
            try:
                resp = session.post(tts_endpoint, headers=headers, json=payload, timeout=60)
                if resp.status_code in (429, 502, 503, 504):
                    raise RuntimeError(f"Temporaneo/rate-limit ({resp.status_code})")
                if resp.status_code == 413:
                    raise RuntimeError("Payload troppo grande (413). Riduci i chunk: 2500–3000.")
                resp.raise_for_status()

                ct = resp.headers.get("Content-Type", "")
                if "application/json" in ct:
                    data = resp.json()
                    audio_url = data.get("audio_url") or data.get("url")
                    if not audio_url:
                        raise RuntimeError(f"Risposta JSON inattesa: {str(data)[:200]}")
                    audio_bytes = _download_with_retry(audio_url, retries=3, timeout=45)
                else:
                    audio_bytes = resp.content

                with open(part_path, "wb") as f:
                    f.write(audio_bytes)

                dur_s = mp3_duration_seconds(part_path)
                if dur_s <= 0.5:
                    raise RuntimeError("MP3 generato con durata nulla.")

                if st: st.write(f"✅ Salvato {os.path.basename(part_path)} — {int(dur_s*1000)} ms")
                success = True
                break

            except Exception as e:
                last_err = e
                wait_s = base_backoff * (attempt + 1)
                if st: st.warning(f"⚠️ Retry {attempt+1}/{retries_per_chunk} part {next_seq:0{pad}d}: {e} (sleep {wait_s:.1f}s)")
                time.sleep(wait_s)

        if not success:
            if st: st.error(f"❌ Fallito part {next_seq:0{pad}d}: {last_err}")
        else:
            new_generated += 1

        next_seq += 1
        time.sleep(sleep_between_chunks)

    # Concatena TUTTE le part_*.mp3 (anche di run precedenti)
    if combine:
        parts = sorted(glob.glob(os.path.join(outdir, "part_*.mp3")))
        if not parts:
            raise RuntimeError("Nessun audio generato.")
        out_path = os.path.join(outdir, "combined_audio.mp3")
        concat_mp3s(parts, out_path, bitrate_kbps=128)
        total_ms = int(mp3_duration_seconds(out_path) * 1000)
        if st:
            st.write(f"🔊 Durata totale audio attuale: {total_ms} ms")
            st.success("🔊 Audio combinato aggiornato.")
        return out_path

    return None
