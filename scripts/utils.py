# scripts/utils.py
# -------------------------------------------------------
# Utility: chunking testo + IMMAGINI (Replicate) + AUDIO (FishAudio)
# Compatibile con Python 3.13: niente pydub; mutagen per durate, imageio-ffmpeg per concat.
# VERSIONE CORRETTA - Chunking audio migliorato + gestione errori 404
# -------------------------------------------------------

import os
import re
import time
import subprocess
from io import BytesIO
import base64

import requests
from PIL import Image

# ============== Chunking ==============
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
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', (text or "").strip()) if s.strip()]
    N = max(1, int(sentences_per_chunk or 1))
    return [" ".join(sentences[i:i + N]) for i in range(0, len(sentences), N)]

# NUOVO: Chunking specifico per audio - gestisce script lunghi fino a 500k caratteri
def chunk_text_for_audio(text: str, target_chars: int = 2000, max_chars: int = 3000):
    """
    Chunking specifico per audio - mantiene frasi complete e mira a ~2000 caratteri
    Ottimizzato per script molto lunghi (200k+ caratteri)
    
    Args:
        text: Testo da dividere
        target_chars: Caratteri target per chunk (default 2000)
        max_chars: Caratteri massimi per chunk prima di forzare split (default 3000)
    """
    if not text:
        return []
    
    # Per script molto lunghi, mostra progress
    original_length = len(text)
    if original_length <= target_chars:
        return [text]
    
    # Dividi per frasi (supporta anche ;)
    sentences = re.split(r'(?<=[.?!;])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Test se aggiungendo questa frase supero il target
        test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        
        if len(test_chunk) <= target_chars:
            current_chunk = test_chunk
        else:
            # Se la frase corrente è troppo lunga da sola, spezzala
            if len(sentence) > max_chars:
                # Salva chunk corrente se non vuoto
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Spezza la frase lunga per parole, poi per caratteri se necessario
                words = sentence.split()
                temp_chunk = ""
                
                for word in words:
                    test_word_chunk = (temp_chunk + " " + word).strip() if temp_chunk else word
                    
                    if len(test_word_chunk) <= max_chars:
                        temp_chunk = test_word_chunk
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        
                        # Se una singola parola è troppo lunga, spezzala per caratteri
                        if len(word) > max_chars:
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
    
    # Log per script lunghi
    if original_length > 50000:
        print(f"[CHUNKING] Script {original_length:,} chars → {len(chunks)} chunks (avg: {original_length//len(chunks):,} chars)")
    
    return chunks

# ============== Streamlit helper ==============
def _st():
    try:
        import streamlit as st  # type: ignore
        return st
    except Exception:
        return None

# ============== Immagini ==============
def save_image_from_url(url: str, path: str, timeout: int = 60):  # AUMENTATO timeout
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
    VERSIONE CORRETTA con gestione errori 404 migliorata
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

    # NORMALIZZA IL NOME DEL MODELLO
    if ":" not in model:
        model = f"{model}:latest"
        if st: st.info(f"🔧 Aggiunto tag :latest → `{model}`")

    import replicate
    client = replicate.Client(api_token=api_key)

    # VERIFICA PRELIMINARE DEL MODELLO
    try:
        owner, name_version = model.split("/", 1)
        name, version = name_version.split(":", 1) if ":" in name_version else (name_version, "latest")
        
        # Test call per verificare esistenza
        resp = requests.get(
            f"https://api.replicate.com/v1/models/{owner}/{name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        
        if resp.status_code == 404:
            if st:
                st.error(f"❌ MODELLO NON TROVATO: `{model}`")
                st.markdown("### 🔄 SOLUZIONI:")
                st.markdown("1. **Verifica il nome** - Controlla su replicate.com")
                st.markdown("2. **Prova un modello alternativo:**")
                st.code("black-forest-labs/flux-schnell")
                st.code("stability-ai/stable-diffusion-xl-base-1.0")
                st.markdown("3. **Controlla i permessi** del tuo account Replicate")
            raise ValueError(f"Modello '{model}' non trovato (404). Verifica il nome su replicate.com")
        
    except requests.exceptions.RequestException as e:
        if st: st.warning(f"⚠️ Impossibile verificare modello: {e} - Procedendo...")

    masked = (api_key[:3] + "…" + api_key[-4:]) if len(api_key) > 8 else "—"
    if st:
        st.write(f"🔐 Token: {masked}")
        st.write(f"🧩 Modello: `{model}`")
    else:
        print(f"[INFO] Token: {masked}, Model: {model}")

    results = []
    failed_chunks = []
    
    for idx, prompt in enumerate(chunks, start=1):
        model_input = {"prompt": prompt}
        model_input.setdefault("aspect_ratio", (cfg or {}).get("aspect_ratio", "16:9"))
        if isinstance(extra_input, dict):
            model_input.update(extra_input)

        try:
            if st: st.write(f"🎨 Generando immagine {idx}/{len(chunks)}...")
            
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

        except replicate.exceptions.ReplicateError as e:
            error_msg = str(e)
            if "404" in error_msg:
                if st:
                    st.error(f"❌ MODELLO INESISTENTE: `{model}`")
                    st.markdown("### 🆘 AZIONE RICHIESTA:")
                    st.markdown("1. Vai su **replicate.com** e cerca un modello funzionante")
                    st.markdown("2. Copia il nome esatto (es: `owner/model-name:version`)")
                    st.markdown("3. Incollalo nel campo 'Custom model' nella sidebar")
                raise ValueError(f"Modello '{model}' non trovato. Vai su replicate.com per trovare modelli disponibili.")
            else:
                failed_chunks.append(idx)
                if st: st.error(f"❌ Errore Replicate chunk {idx}: {error_msg}")
                else: print(f"[ERROR] Replicate chunk {idx}: {error_msg}")
                
        except Exception as e:
            failed_chunks.append(idx)
            if st: st.error(f"❌ Errore generico chunk {idx}: {e}")
            else: print(f"[ERROR] Generic chunk {idx}: {e}")

        if sleep_between_calls and idx < len(chunks):
            time.sleep(sleep_between_calls)

    # REPORT FINALE
    if failed_chunks:
        if st:
            st.warning(f"⚠️ {len(failed_chunks)} immagini fallite su {len(chunks)} totali")
            st.write(f"Chunks falliti: {failed_chunks}")
        else:
            print(f"[WARNING] {len(failed_chunks)} failed chunks: {failed_chunks}")
    
    if not results:
        raise RuntimeError("Nessuna immagine generata con successo. Controlla modello e API key.")

    return results

# ============== Audio (FishAudio) senza pydub ==============

def mp3_duration_seconds(path: str) -> float:
    """Ritorna la durata in secondi usando mutagen."""
    try:
        from mutagen.mp3 import MP3
        return float(MP3(path).info.length)
    except Exception:
        return 0.0

def concat_mp3s(paths, out_path: str, bitrate_kbps: int = 128):
    """
    Concatena MP3 usando ffmpeg (imageio-ffmpeg). Ricodifica a libmp3lame.
    VERSIONE CORRETTA con fallback alternativo
    """
    if not paths:
        raise RuntimeError("Nessun file MP3 da concatenare.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Prova multiple fonti per FFmpeg
    ffmpeg_bin = None
    
    # 1. Prova imageio-ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[DEBUG] Using imageio-ffmpeg: {ffmpeg_bin}")
    except Exception as e:
        print(f"[DEBUG] imageio-ffmpeg failed: {e}")
    
    # 2. Prova ffmpeg system
    if not ffmpeg_bin:
        import shutil
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            print(f"[DEBUG] Using system ffmpeg: {ffmpeg_bin}")
    
    # 3. Fallback alternativo senza ffmpeg
    if not ffmpeg_bin:
        print("[WARNING] FFmpeg not found, using alternative method")
        return _concat_mp3s_alternative(paths, out_path)

    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            abspath = os.path.abspath(p)
            f.write(f"file '{abspath}'\n")

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

def _concat_mp3s_alternative(paths, out_path: str):
    """Fallback: concatena usando solo Python quando FFmpeg non disponibile"""
    print("[INFO] Using Python-only MP3 concat (no ffmpeg)")
    
    with open(out_path, 'wb') as outfile:
        for i, path in enumerate(paths):
            with open(path, 'rb') as infile:
                if i == 0:
                    # Primo file: copia tutto
                    outfile.write(infile.read())
                else:
                    # File successivi: salta header MP3 (approssimativo)
                    data = infile.read()
                    # Cerca sync frame MP3 (0xFF FB o 0xFF FA)
                    for j in range(min(1000, len(data) - 1)):
                        if data[j] == 0xFF and (data[j+1] & 0xE0) == 0xE0:
                            outfile.write(data[j:])
                            break
                    else:
                        # Fallback: scrivi tutto
                        outfile.write(data)
    return out_path

def _download_with_retry(url: str, retries: int = 5, timeout: int = 60) -> bytes:  # AUMENTATI retry e timeout
    last_exc = None
    for _ in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(1.5)
    raise last_exc or RuntimeError("Download fallito")

def generate_audio(chunks, cfg: dict, outdir: str,
                   tts_endpoint: str = "https://api.fish.audio/v1/tts"):
    """
    Genera audio da ogni blocco di testo in `chunks` con FishAudio e li concatena.
    VERSIONE CORRETTA con logging chunking e timeout migliorati
    Ottimizzata per script lunghi (200k+ caratteri)
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

    # LOG CHUNKING INFO (migliorato per script lunghi)
    if st:
        total_chars = sum(len(chunk) for chunk in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        min_chars = min(len(chunk) for chunk in chunks) if chunks else 0
        max_chars = max(len(chunk) for chunk in chunks) if chunks else 0
        
        st.write(f"📊 Audio chunking: {len(chunks)} segmenti")
        st.write(f"📈 Caratteri: Totale {total_chars:,} | Media {avg_chars:.0f} | Min {min_chars:,} | Max {max_chars:,}")
        
        # Stima durata (approssimativa: ~150 parole/minuto, ~5 caratteri/parola)
        estimated_minutes = (total_chars / 5) / 150
        st.write(f"⏱️ Durata stimata audio: ~{estimated_minutes:.1f} minuti")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if model:
        headers["model"] = model

    audio_paths = []
    failed_count = 0

    # Progress bar per script lunghi
    if st and len(chunks) > 10:
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        progress_bar = None
        status_text = None

    for i, text in enumerate(chunks, 1):
        if progress_bar:
            progress_bar.progress(i / len(chunks))
            status_text.text(f"🎧 Generando audio {i}/{len(chunks)} ({len(text)} caratteri)...")
        elif st:
            st.write(f"🎧 Audio {i}/{len(chunks)} ({len(text)} caratteri)...")
        else:
            print(f" • Audio {i}/{len(chunks)} ({len(text)} chars)…", end=" ")

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
            resp = requests.post(tts_endpoint, headers=headers, json=payload, timeout=180)  # AUMENTATO timeout per chunk lunghi
            if resp.status_code >= 400:
                detail = resp.text[:500]
                if st: st.error(f"❌ HTTP {resp.status_code} FishAudio chunk {i}: {detail}")
                else: print(f"HTTP {resp.status_code} FishAudio chunk {i}: {detail}")
                failed_count += 1
                continue

            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                data = resp.json()
                audio_url = data.get("audio_url") or data.get("url")
                audio_b64 = data.get("audio_base64") or data.get("audio")
                if audio_url:
                    audio_bytes = _download_with_retry(audio_url, retries=5, timeout=90)  # AUMENTATO timeout download
                elif audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                else:
                    if st: st.error(f"❌ Risposta JSON inattesa chunk {i}: {data}")
                    else: print(f"JSON inatteso chunk {i}: {data}")
                    failed_count += 1
                    continue
            else:
                audio_bytes = resp.content

            path = os.path.join(outdir, f"audio_{i:03d}.mp3")  # 3 cifre per supportare 999+ chunk
            with open(path, "wb") as f:
                f.write(audio_bytes)

            dur_sec = mp3_duration_seconds(path)
            if progress_bar:
                # Non loggare ogni singolo chunk per script lunghi
                pass
            elif st:
                st.write(f"✅ TTS chunk {i:03d} durata: {dur_sec:.1f}s")
            else:
                print(f"[TTS] chunk {i:03d} duration: {dur_sec:.1f}s — ✅")

            audio_paths.append(path)

        except Exception as e:
            failed_count += 1
            if st: st.error(f"❌ Errore TTS sul chunk {i}: {e}")
            else: print("❌", f"Errore TTS sul chunk {i}: {e}")
            continue

    # Cleanup progress bar
    if progress_bar:
        progress_bar.empty()
        status_text.empty()

    # Report finale
    if failed_count > 0:
        if st:
            st.warning(f"⚠️ {failed_count} chunk audio falliti su {len(chunks)} totali")
        else:
            print(f"[WARNING] {failed_count} failed audio chunks out of {len(chunks)}")

    if not audio_paths:
        return None

    # Concatenazione con progress per script lunghi
    if st and len(audio_paths) > 20:
        st.write(f"🔗 Concatenando {len(audio_paths)} file audio...")

    out_path = os.path.join(outdir, "combined_audio.mp3")
    concat_mp3s(audio_paths, out_path, bitrate_kbps=128)

    final_duration = mp3_duration_seconds(out_path)
    if st:
        st.write(f"🔊 Durata totale audio finale: {final_duration:.1f} secondi ({final_duration/60:.1f} minuti)")
        st.success("🔊 Audio finale creato.")
    else:
        print(f"[TTS] Final audio: {final_duration:.2f}s ({final_duration/60:.1f}min)")
        print("🔊 Audio finale creato.")

    return out_path
