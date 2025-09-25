# scripts/utils.py
# -------------------------------------------------------
# Utility: chunking testo + IMMAGINI (Replicate) + AUDIO (FishAudio)
# Versione robusta anti-taglio + supporto voce forzata
# -------------------------------------------------------

import os
import re
import time
import subprocess
from io import BytesIO
import base64
import json
import glob
import shutil
from typing import List, Optional

import requests
from PIL import Image

# ======================= Costanti =======================

AUDIO_EXTS = ("mp3", "wav", "m4a")
IMAGE_EXTS = ("png", "jpg", "jpeg")

# Silenzi per migliorare il TTS
TAIL_SILENCE_SECS = 0.25            # silenzio alla fine di ogni clip
BETWEEN_SILENCE_SECS = 0.8          # silenzio tra clip nel merge finale

# Voce di default (se non arriva da cfg/env)
DEFAULT_FISHAUDIO_VOICE_ID = "80e34d5e0b2b4577a486f3a77e357261"

# ================== Checkpoint (compat) =================

def save_checkpoint(base_dir: str, data: dict):
    cp = os.path.join(base_dir, "checkpoint.json")
    os.makedirs(base_dir, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_checkpoint(base_dir: str) -> dict:
    cp = os.path.join(base_dir, "checkpoint.json")
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def clear_checkpoint(base_dir: str):
    cp = os.path.join(base_dir, "checkpoint.json")
    try:
        if os.path.exists(cp):
            os.remove(cp)
    except Exception:
        pass

def get_completed_files(directory: str, pattern: str) -> list:
    if not os.path.exists(directory):
        return []
    return sorted(glob.glob(os.path.join(directory, pattern)))

# ================== Streamlit helper ====================

def _st():
    try:
        import streamlit as st
        return st
    except Exception:
        return None

# =================== Chunking testo =====================

def _normalize_trailing_for_tts(text: str) -> str:
    """Assicura terminazione con .?! e spazio finale (aiuta alcuni TTS a non troncare)."""
    s = (text or "").rstrip()
    if s and s[-1] not in ".?!":
        s += "."
    if not s.endswith(" "):
        s += " "
    return s

def chunk_text(text: str, max_chars: int):
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
    sentences = re.split(r'(?<=[.?!])\s+', (text or "").strip())
    parts, curr = [], ""
    for sent in sentences:
        cand = (curr + " " + sent).strip() if curr else sent.strip()
        if len(cand) <= max_chars:
            curr = cand
        else:
            if curr:
                parts.append(curr)
            curr = sent.strip()
    if curr:
        parts.append(curr)
    return parts

def chunk_by_sentences_count(text: str, sentences_per_chunk: int):
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', (text or "").strip()) if s.strip()]
    N = max(1, int(sentences_per_chunk or 1))
    return [" ".join(sentences[i:i + N]) for i in range(0, len(sentences), N)]

def chunk_text_for_audio(text: str, target_chars: int = 2000, max_chars: int = 4000):
    """Chunking per audio: accumula frasi fino al target, mai spezzare parole; spezza parole solo oltre max_chars."""
    if not text:
        return []
    if len(text) <= target_chars:
        return [_normalize_trailing_for_tts(text)]

    sentences = re.split(r'(?<=[.?!;])\s+', text.strip())
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        test = (current + " " + s).strip() if current else s
        if len(test) <= target_chars:
            current = test
        else:
            if len(s) > max_chars:
                if current:
                    chunks.append(_normalize_trailing_for_tts(current))
                    current = ""
                # spezza per parole ma senza spezzare parole stesse
                words = s.split()
                buf = ""
                for w in words:
                    cand = (buf + " " + w).strip() if buf else w
                    if len(cand) <= max_chars:
                        buf = cand
                    else:
                        if buf:
                            chunks.append(_normalize_trailing_for_tts(buf))
                        buf = w
                if buf:
                    current = buf
            else:
                if current:
                    chunks.append(_normalize_trailing_for_tts(current))
                current = s
    if current:
        chunks.append(_normalize_trailing_for_tts(current))
    return chunks

# ================== IMMAGINI (Replicate) =================

def save_image_from_url(url: str, path: str, timeout: int = 60):
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

def generate_images(chunks, cfg: dict, outdir: str, sleep_between_calls: float = None):
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    api_key = (cfg or {}).get("replicate_api_token") or (cfg or {}).get("replicate_api_key") or os.getenv("REPLICATE_API_TOKEN")
    model = (cfg or {}).get("image_model") or (cfg or {}).get("replicate_model")
    extra_input = (cfg or {}).get("replicate_input", {})

    if sleep_between_calls is None:
        sleep_between_calls = (cfg or {}).get("sleep_time", 11.0)

    if not api_key:
        msg = "Replicate API token assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not model:
        msg = "Modello Replicate assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    if ":" not in model:
        model = f"{model}:latest"
        if st:
            st.info(f"🔧 Aggiunto tag :latest → `{model}`")

    import replicate
    client = replicate.Client(api_token=api_key)

    results = []
    failed = []
    t0 = time.time()

    progress_bar = st.progress(0) if st and len(chunks) > 5 else None
    status_text = st.empty() if progress_bar else None

    for i, prompt in enumerate(chunks, start=1):
        if progress_bar and status_text:
            progress_bar.progress((i - 1) / len(chunks))
            if i > 1:
                avg = (time.time() - t0) / (i - 1)
                eta = avg * (len(chunks) - i + 1)
                status_text.write(f"🎨 {i}/{len(chunks)} · ETA {eta/60:.1f} min")
        elif st:
            st.write(f"🎨 Immagine {i}/{len(chunks)}…")

        inp = {"prompt": prompt}
        inp.setdefault("aspect_ratio", (cfg or {}).get("aspect_ratio", "16:9"))
        if isinstance(extra_input, dict):
            inp.update(extra_input)

        try:
            output = client.run(model, input=inp)
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

            outpath = os.path.join(outdir, f"img_{i:03d}.png")
            _download_first(urls, outpath)
            results.append(outpath)

            if st:
                st.write(f"✅ Immagine {i}: `{os.path.basename(outpath)}`")

        except Exception as e:
            failed.append(i)
            if st:
                st.error(f"❌ Errore immagine {i}: {e}")

        if sleep_between_calls and i < len(chunks):
            time.sleep(sleep_between_calls)

    if progress_bar:
        progress_bar.progress(1.0)
        if status_text:
            status_text.write(f"✅ {len(results)} immagini completate")
        time.sleep(0.5)
        progress_bar.empty(); status_text.empty()

    if failed and st:
        st.warning(f"⚠️ {len(failed)} fallite su {len(chunks)}")

    if not results:
        raise RuntimeError("Nessuna immagine generata con successo.")
    return results

# ==================== AUDIO helpers =====================

def mp3_duration_seconds(path: str) -> float:
    try:
        from mutagen.mp3 import MP3
        return float(MP3(path).info.length)
    except Exception:
        return 0.0

def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")

def _make_silence_wav(path: str, secs: float) -> bool:
    ff = _ffmpeg_exe()
    if not ff: return False
    try:
        r = subprocess.run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(secs), "-c:a", "pcm_s16le", path
        ], capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(path)
    except Exception:
        return False

def _to_wav_with_tail(in_path: str, out_path: str, tail_secs: float = TAIL_SILENCE_SECS) -> bool:
    """
    Converte qualsiasi input in WAV 44.1kHz, aggiungendo tail di silenzio.
    """
    ff = _ffmpeg_exe()
    if not ff:  # niente ffmpeg → semplice copia in WAV possibile? no → falla in MP3 diretto
        return False
    tmp_dir = os.path.dirname(out_path)
    os.makedirs(tmp_dir, exist_ok=True)
    sil = os.path.join(tmp_dir, "_tail.wav")
    if not _make_silence_wav(sil, tail_secs):
        # solo transcodifica a WAV
        r = subprocess.run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", out_path
        ], capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(out_path)

    r = subprocess.run([
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path, "-i", sil,
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
        "-map", "[a]", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", out_path
    ], capture_output=True, text=True)
    try:
        os.remove(sil)
    except Exception:
        pass
    return r.returncode == 0 and os.path.exists(out_path)

def concat_mp3s(paths: List[str], out_path: str, bitrate_kbps: int = 192, between_silence: float = BETWEEN_SILENCE_SECS):
    """
    Concat robusto: decodifica a PCM, inserisce silenzio tra clip, ricodifica in MP3.
    Evita il demuxer MP3 (padding/delay).
    """
    if not paths:
        raise RuntimeError("Nessun file audio da concatenare.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ff = _ffmpeg_exe()
    if not ff:
        raise RuntimeError("ffmpeg non trovato")

    tmp = out_path + ".__tmp"
    os.makedirs(tmp, exist_ok=True)

    # converti tutto a WAV con tail
    wavs = []
    for i, p in enumerate(paths):
        w = os.path.join(tmp, f"p_{i:03d}.wav")
        if not _to_wav_with_tail(p, w, TAIL_SILENCE_SECS):
            raise RuntimeError(f"ffmpeg fallito su {p}")
        wavs.append(w)

    # silenzio tra parti
    sil = os.path.join(tmp, "_between.wav")
    if between_silence > 0 and not _make_silence_wav(sil, between_silence):
        sil = None

    # costruisci cmd: [p1,(sil),p2,(sil),...,pn]
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    nin = 0
    for i, w in enumerate(wavs):
        cmd += ["-i", w]; nin += 1
        if sil and i < len(wavs) - 1:
            cmd += ["-i", sil]; nin += 1

    labels = "".join([f"[{i}:a]" for i in range(nin)])
    filt = f"{labels}concat=n={nin}:v=0:a=1[a]"
    cmd += [
        "-filter_complex", filt, "-map", "[a]",
        "-ar", "44100", "-ac", "2",
        "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k",
        out_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    # cleanup
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg concat fallito: {r.stderr[:200]}")

# ==================== AUDIO (FishAudio) ==================

def _download_with_retry(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc or RuntimeError("Download fallito")

def _resolve_voice_id(cfg: dict) -> str:
    """Prende voice id da cfg o ENV (molte chiavi comuni), fallback a DEFAULT."""
    # cfg keys
    for k in [
        "fishaudio_voice_id", "fishaudio_voice", "fishaudio_speaker",
        "voice_model_id", "voice_id", "voice", "speaker", "speaker_id"
    ]:
        v = (cfg or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # env keys
    for k in [
        "FISHAUDIO_VOICE_ID", "FISHAUDIO_VOICE", "FISHAUDIO_SPEAKER",
        "VOICE_MODEL_ID", "VOICE_ID", "VOICE", "SPEAKER", "SPEAKER_ID"
    ]:
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    return DEFAULT_FISHAUDIO_VOICE_ID

def generate_audio(
    chunks: List[str],
    cfg: dict,
    outdir: str,
    tts_endpoint: str = "https://api.fish.audio/v1/tts",
    progress_cb=None,
):
    """
    Genera audio usando FishAudio TTS.
    - Normalizza i testi (punteggiatura + spazio finale).
    - Forza voice_id se non presente in cfg (fallback alla voce di default).
    - Salva MP3; se ffmpeg disponibile, crea anche WAV con tail di silenzio.
    - Se più chunk, concat finale robusto (PCM) con silenzio tra clip.
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    # Config
    api_key = (cfg or {}).get("fishaudio_api_key") or os.getenv("FISHAUDIO_API_KEY")
    if not api_key:
        msg = "FishAudio API key assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    voice_id = _resolve_voice_id(cfg)
    model = (cfg or {}).get("fishaudio_model") or os.getenv("FISHAUDIO_MODEL") or None
    extra = (cfg or {}).get("fishaudio_extra", {})

    # Debug voce
    if st:
        st.caption(f"🎙️ Voice ID in uso: `{voice_id}`")

    # Normalize chunk texts
    norm_chunks = [_normalize_trailing_for_tts(c) for c in (chunks or []) if (c or "").strip()]
    if not norm_chunks:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # il sito mostra spesso un header "model: speech-1.5"
    if model:
        headers["model"] = model
    else:
        # usa un default ragionevole se non specificato
        headers["model"] = "speech-1.5"

    audio_paths: List[str] = []
    t0 = time.time()
    pb = st.progress(0.0) if st and len(norm_chunks) > 5 else None
    stat = st.empty() if pb else None

    for i, text in enumerate(norm_chunks, start=1):
        if pb and stat:
            pb.progress((i - 1) / len(norm_chunks))
            if i > 1:
                avg = (time.time() - t0) / (i - 1)
                eta = avg * (len(norm_chunks) - i + 1)
                stat.write(f"🎧 {i}/{len(norm_chunks)} · ETA {eta/60:.1f} min")
        elif st:
            st.write(f"🎧 Generando audio {i}/{len(norm_chunks)}… ({len(text)} char)")

        payload = {
            "text": text,
            "reference_id": voice_id,   # chiave corretta per FishAudio
            "format": "mp3",
            "mp3_bitrate": 192,
            # Flag “sicuri” spesso supportati (ignorati se non previsti):
            "normalize": True,
            "trim_silence": False,
        }
        if model:
            payload["model"] = model
        if isinstance(extra, dict):
            payload.update(extra)

        try:
            resp = requests.post(tts_endpoint, headers=headers, json=payload, timeout=180)
            if resp.status_code >= 400:
                if st: st.error(f"❌ HTTP {resp.status_code} al chunk {i}: {resp.text[:200]}")
                continue

            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                data = resp.json()
                audio_url = data.get("audio_url") or data.get("url")
                audio_b64 = data.get("audio_base64") or data.get("audio")
                if audio_url:
                    audio_bytes = _download_with_retry(audio_url)
                elif audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                else:
                    if st: st.error(f"❌ Nessun audio nella risposta (chunk {i})")
                    continue
            else:
                audio_bytes = resp.content

            # Salva MP3
            mp3_path = os.path.join(outdir, f"audio_{i:03d}.mp3")
            with open(mp3_path, "wb") as f:
                f.write(audio_bytes)
            audio_paths.append(mp3_path)

            # se posso, creo subito un WAV con tail di silenzio (anti-taglio per uso stand-alone)
            ff = _ffmpeg_exe()
            if ff:
                wav_fixed = os.path.join(outdir, f"audio_{i:03d}.wav")
                if _to_wav_with_tail(mp3_path, wav_fixed, TAIL_SILENCE_SECS):
                    # teniamo entrambe le versioni; il chiamante sceglie cosa usare
                    pass

            if progress_cb:
                try:
                    progress_cb(f"Chunk {i:03d} ok")
                except Exception:
                    pass

            if st:
                dur = mp3_duration_seconds(mp3_path)
                st.write(f"✅ Chunk {i:03d} ({dur:.1f}s)")

        except Exception as e:
            if st:
                st.error(f"❌ Errore chunk {i}: {e}")
            continue

    if pb:
        pb.progress(1.0); time.sleep(0.2); pb.empty()
        if stat: stat.empty()

    if not audio_paths:
        return None

    # Se più chunk: concat robusto (PCM + silenzio) → outdir/combined_audio.mp3
    if len(audio_paths) > 1:
        out_path = os.path.join(outdir, "combined_audio.mp3")
        try:
            concat_mp3s(audio_paths, out_path, bitrate_kbps=192, between_silence=BETWEEN_SILENCE_SECS)
            if st:
                final_dur = mp3_duration_seconds(out_path)
                st.success(f"🔊 Audio finale: {final_dur:.1f}s")
            return out_path
        except Exception as e:
            if st:
                st.error(f"❌ Errore concatenazione: {e}")
            # ritorna comunque lista delle clip
            return audio_paths

    # Un solo chunk → ritorna il file prodotto
    return audio_paths[0]

