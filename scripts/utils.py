# scripts/utils.py
# -------------------------------------------------------
# Utility: chunking testo + IMMAGINI (Replicate) + AUDIO (FishAudio)
# Compatibile con Python 3.13: niente pydub; mutagen per durate, imageio-ffmpeg per concat.
# VERSIONE PULITA SENZA TIMELINE - RICREATA DA ZERO
# -------------------------------------------------------

import os
import re
import time
import subprocess
from io import BytesIO
import base64

import requests
from PIL import Image

# ============== Helper Functions ==============

def _st():
    """Helper per importare Streamlit solo se disponibile"""
    try:
        import streamlit as st
        return st
    except Exception:
        return None

# ============== Chunking Functions ==============

def chunk_text(text: str, max_chars: int):
    """Chunking base per caratteri"""
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
    """Chunking per frasi con limite caratteri"""
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
    """Chunking per numero di frasi"""
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', (text or "").strip()) if s.strip()]
    N = max(1, int(sentences_per_chunk or 1))
    return [" ".join(sentences[i:i + N]) for i in range(0, len(sentences), N)]

def chunk_text_for_audio(text: str, target_chars: int = 2000, max_chars: int = 4000):
    """
    Chunking specifico per audio - gestisce script lunghi
    """
    if not text:
        return []
    
    original_length = len(text)
    if original_length <= target_chars:
        return [text]
    
    # Dividi per frasi (supporta . ? ! ;)
    sentences = re.split(r'(?<=[.?!;])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Test se aggiungendo questa frase supero il target
        test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        
        if len(test_chunk) <= target_chars:
            current_chunk = test_chunk
        else:
            # Se la frase è troppo lunga, spezzala intelligentemente
            if len(sentence) > max_chars:
                # Salva chunk corrente se non vuoto
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Spezza la frase lunga per parole
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    test_word = (temp_chunk + " " + word).strip() if temp_chunk else word
                    if len(test_word) <= max_chars:
                        temp_chunk = test_word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        if len(word) > max_chars:
                            # Spezza parole troppo lunghe
                            for j in range(0, len(word), max_chars):
                                chunks.append(word[j:j + max_chars])
                            temp_chunk = ""
                        else:
                            temp_chunk = word
                
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                # Salva chunk corrente e inizia nuovo
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
    
    # Aggiungi ultimo chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# ============== Image Generation ==============

def save_image_from_url(url: str, path: str, timeout: int = 60):
    """Scarica e salva immagine da URL"""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    img.save(path)

def _download_first(urls, dest_path: str):
    """Scarica la prima URL disponibile"""
    if not urls:
        raise ValueError("Nessuna URL immagine restituita dal modello.")
    save_image_from_url(urls[0], dest_path)
    return dest_path

def generate_images(chunks, cfg: dict, outdir: str, sleep_between_calls: float = None):
    """
    Genera immagini usando Replicate
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    # Estrai configurazione
    api_key = (cfg or {}).get("replicate_api_token") or (cfg or {}).get("replicate_api_key") or os.getenv("REPLICATE_API_TOKEN")
    model = (cfg or {}).get("image_model") or (cfg or {}).get("replicate_model")
    extra_input = (cfg or {}).get("replicate_input", {})
    
    # Sleep dinamico
    if sleep_between_calls is None:
        sleep_between_calls = cfg.get("sleep_time", 11.0)

    if not api_key:
        msg = "Replicate API token assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not model:
        msg = "Modello Replicate assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    # Normalizza nome modello
    if ":" not in model:
        model = f"{model}:latest"
        if st:
            st.info(f"🔧 Aggiunto tag :latest → `{model}`")

    # Inizializza client Replicate
    import replicate
    client = replicate.Client(api_token=api_key)

    # Info debug
    masked = (api_key[:3] + "…" + api_key[-4:]) if len(api_key) > 8 else "—"
    if st:
        st.write(f"🔐 Token: {masked} | 🧩 Modello: `{model}`")

    results = []
    failed_chunks = []
    start_time = time.time()

    # Progress bar per molte immagini
    if st and len(chunks) > 5:
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        progress_bar = None
        status_text = None
    
    for idx, prompt in enumerate(chunks, start=1):
        # Aggiorna progress bar
        if progress_bar and status_text:
            progress = (idx - 1) / len(chunks)
            progress_bar.progress(progress)
            elapsed = time.time() - start_time
            if idx > 1:
                avg_time = elapsed / (idx - 1)
                eta = avg_time * (len(chunks) - idx + 1)
                status_text.write(f"🎨 Generando immagine {idx}/{len(chunks)} - ETA: {eta/60:.1f} min")
            else:
                status_text.write(f"🎨 Generando immagine {idx}/{len(chunks)}...")
        elif st:
            st.write(f"🎨 Generando immagine {idx}/{len(chunks)}...")

        model_input = {"prompt": prompt}
        model_input.setdefault("aspect_ratio", (cfg or {}).get("aspect_ratio", "16:9"))
        if isinstance(extra_input, dict):
            model_input.update(extra_input)

        try:
            output = client.run(model, input=model_input)
            
            # Estrai URLs
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

            # Successo
            if st:
                elapsed = time.time() - start_time
                st.write(f"✅ Immagine {idx} generata: `{os.path.basename(outpath)}` (tempo: {elapsed/60:.1f} min)")

        except Exception as e:
            failed_chunks.append(idx)
            if st:
                st.error(f"❌ Errore immagine {idx}: {e}")

        # Sleep tra chiamate
        if sleep_between_calls > 0 and idx < len(chunks):
            time.sleep(sleep_between_calls)

    # Cleanup progress bar
    if progress_bar:
        progress_bar.progress(1.0)
        if status_text:
            status_text.write(f"✅ Completate {len(results)} immagini")
        time.sleep(1)
        progress_bar.empty()
        if status_text:
            status_text.empty()

    # Report finale
    if failed_chunks and st:
        st.warning(f"⚠️ {len(failed_chunks)} immagini fallite su {len(chunks)} totali")
    
    if not results:
        raise RuntimeError("Nessuna immagine generata con successo.")

    return results

# ============== Audio Generation ==============

def mp3_duration_seconds(path: str) -> float:
    """Ritorna la durata in secondi usando mutagen"""
    try:
        from mutagen.mp3 import MP3
        return float(MP3(path).info.length)
    except Exception:
        return 0.0

def concat_mp3s(paths, out_path: str, bitrate_kbps: int = 128):
    """Concatena MP3 usando ffmpeg con fallback"""
    if not paths:
        raise RuntimeError("Nessun file MP3 da concatenare.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Prova ffmpeg
    ffmpeg_bin = None
    
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        ffmpeg_bin = shutil.which("ffmpeg")
    
    if not ffmpeg_bin:
        return _concat_mp3s_alternative(paths, out_path)

    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            abspath = os.path.abspath(p)
            f.write(f"file '{abspath}'\n")

    cmd = [
        ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:200]}")
    
    try:
        os.remove(list_path)
    except Exception:
        pass

def _concat_mp3s_alternative(paths, out_path: str):
    """Fallback concatenazione senza ffmpeg"""
    with open(out_path, 'wb') as outfile:
        for i, path in enumerate(paths):
            with open(path, 'rb') as infile:
                if i == 0:
                    outfile.write(infile.read())
                else:
                    data = infile.read()
                    # Cerca sync frame MP3
                    for j in range(min(1000, len(data) - 1)):
                        if data[j] == 0xFF and (data[j+1] & 0xE0) == 0xE0:
                            outfile.write(data[j:])
                            break
                    else:
                        outfile.write(data)
    return out_path

def _download_with_retry(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    """Download con retry"""
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

def generate_audio(chunks, cfg: dict, outdir: str, tts_endpoint: str = "https://api.fish.audio/v1/tts"):
    """
    Genera audio usando FishAudio TTS
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)
    
    # Estrai configurazione
    api_key = (cfg or {}).get("fishaudio_api_key")
    voice_id = (cfg or {}).get("fishaudio_voice") or (cfg or {}).get("fishaudio_voice_id")
    model = (cfg or {}).get("fishaudio_model")
    extra = (cfg or {}).get("fishaudio_extra", {})

    if not api_key:
        msg = "FishAudio API key assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not voice_id:
        msg = "FishAudio Voice ID assente."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    # Info chunking
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    
    if st:
        st.info(f"📊 Audio: {len(chunks)} segmenti | {total_chars:,} caratteri | Media: {avg_chars:.0f}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if model:
        headers["model"] = model

    audio_paths = []
    failed_count = 0
    start_time = time.time()

    # Progress bar per molti chunk
    if st and len(chunks) > 10:
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        progress_bar = None
        status_text = None

    for i, text in enumerate(chunks, 1):
        # Aggiorna progress
        if progress_bar and status_text:
            progress = i / len(chunks)
            progress_bar.progress(progress)
            elapsed = time.time() - start_time
            if i > 1:
                avg_time = elapsed / (i - 1)
                eta = avg_time * (len(chunks) - i)
                status_text.write(f"🎧 Audio {i}/{len(chunks)} - ETA: {eta/60:.1f} min")
            else:
                status_text.write(f"🎧 Audio {i}/{len(chunks)}...")
        elif st:
            st.write(f"🎧 Generando audio {i}/{len(chunks)} ({len(text)} caratteri)...")

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

        try:
            resp = requests.post(tts_endpoint, headers=headers, json=payload, timeout=180)
            
            if resp.status_code >= 400:
                failed_count += 1
                if st:
                    st.error(f"❌ HTTP {resp.status_code} chunk {i}")
                continue

            # Processa risposta
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
                    failed_count += 1
                    continue
            else:
                audio_bytes = resp.content

            # Salva file
            path = os.path.join(outdir, f"audio_{i:03d}.mp3")
            with open(path, "wb") as f:
                f.write(audio_bytes)

            dur_sec = mp3_duration_seconds(path)
            audio_paths.append(path)
            
            if st:
                st.write(f"✅ Chunk {i:03d}: {dur_sec:.1f}s")

        except Exception as e:
            failed_count += 1
            if st:
                st.error(f"❌ Errore chunk {i}: {e}")

    # Cleanup progress
    if progress_bar:
        progress_bar.progress(1.0)
        if status_text:
            status_text.write(f"✅ Completati {len(audio_paths)} chunk")
        time.sleep(1)
        progress_bar.empty()
        if status_text:
            status_text.empty()

    if failed_count > 0 and st:
        st.warning(f"⚠️ {failed_count} chunk falliti")

    if not audio_paths:
        return None

    # Concatenazione
    if st:
        st.write(f"🔗 Concatenando {len(audio_paths)} file...")

    try:
        out_path = os.path.join(outdir, "combined_audio.mp3")
        concat_mp3s(audio_paths, out_path)
        
        final_duration = mp3_duration_seconds(out_path)
        total_time = time.time() - start_time
        
        if st:
            st.success(f"🔊 Audio finale: {final_duration:.1f}s ({final_duration/60:.1f} min)")
            st.info(f"⏱️ Tempo totale: {total_time/60:.1f} min")

        return out_path
        
    except Exception as e:
        if st:
            st.error(f"❌ Errore concatenazione: {e}")
        raise
