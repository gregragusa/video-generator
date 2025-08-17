# scripts/utils.py
# -------------------------------------------------------
# Utility: chunking testo + IMMAGINI (Replicate) + AUDIO (FishAudio)
# Compatibile con Python 3.13: niente pydub; mutagen per durate, imageio-ffmpeg per concat.
# VERSIONE COMPLETA CON TIMELINE INTEGRATA
# -------------------------------------------------------

import os
import re
import time
import subprocess
from io import BytesIO
import base64
from datetime import datetime

import requests
from PIL import Image

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
    Chunking specifico per audio - gestisce script lunghi fino a 500k caratteri
    Ottimizzato per qualità TTS e performance
    
    Args:
        text: Testo da dividere
        target_chars: Caratteri target per chunk (default 2000)
        max_chars: Caratteri massimi per chunk prima di forzare split (default 4000)
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
    
    for i, sentence in enumerate(sentences):
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
                
                # Spezza la frase lunga per virgole, poi per parole
                parts = sentence.split(',')
                temp_chunk = ""
                
                for part in parts:
                    part = part.strip()
                    test_part = (temp_chunk + ", " + part).strip() if temp_chunk else part
                    
                    if len(test_part) <= max_chars:
                        temp_chunk = test_part
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        
                        # Se anche la parte è troppo lunga, spezza per parole
                        if len(part) > max_chars:
                            words = part.split()
                            word_chunk = ""
                            for word in words:
                                test_word = (word_chunk + " " + word).strip() if word_chunk else word
                                if len(test_word) <= max_chars:
                                    word_chunk = test_word
                                else:
                                    if word_chunk:
                                        chunks.append(word_chunk)
                                    # Se una singola parola è troppo lunga, spezzala brutalmente
                                    if len(word) > max_chars:
                                        for j in range(0, len(word), max_chars):
                                            chunks.append(word[j:j + max_chars])
                                        word_chunk = ""
                                    else:
                                        word_chunk = word
                            if word_chunk:
                                temp_chunk = word_chunk
                        else:
                            temp_chunk = part
                
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

# ============== Streamlit Helper ==============

def _st():
    """Helper per importare Streamlit solo se disponibile"""
    try:
        import streamlit as st
        return st
    except Exception:
        return None

# ============== Image Generation ==============

def save_image_from_url(url: str, path: str, timeout: int = 60):
    """Scarica e salva immagine da URL con timeout esteso"""
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
    Genera immagini con timeline integrata e gestione errori avanzata
    
    Args:
        chunks: Lista di prompt per le immagini
        cfg: Configurazione con API keys e parametri
        outdir: Directory di output
        sleep_between_calls: Secondi di pausa tra chiamate (auto se None)
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    # Estrai configurazione
    api_key = (cfg or {}).get("replicate_api_token") or (cfg or {}).get("replicate_api_key") or os.getenv("REPLICATE_API_TOKEN")
    model = (cfg or {}).get("image_model") or (cfg or {}).get("replicate_model")
    extra_input = (cfg or {}).get("replicate_input", {})
    
    # Timeline tracker
    tracker = cfg.get("progress_tracker")
    timeline_container = cfg.get("timeline_container")
    display_timeline_func = cfg.get("display_timeline_func")  # Funzione passata da app.py
    display_timeline_func = cfg.get("display_timeline_func")  # Funzione passata da app.py
    
    # Sleep dinamico basato su modalità velocità
    if sleep_between_calls is None:
        sleep_between_calls = cfg.get("sleep_time", 11.0)

    if not api_key:
        msg = "Replicate API token assente. Configura le API keys nella sidebar."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not model:
        msg = "Modello Replicate assente. Seleziona un modello nella sidebar."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    # Normalizza nome modello
    if ":" not in model:
        model = f"{model}:latest"
        if tracker:
            tracker.add_substep(tracker.current_step, f"🔧 Modello normalizzato: {model}", "completed")
            _update_timeline(tracker, timeline_container, display_timeline_func)
        elif st:
            st.info(f"🔧 Aggiunto tag :latest → `{model}`")

    # Inizializza client Replicate
    import replicate
    client = replicate.Client(api_token=api_key)

    # Verifica modello
    try:
        owner, name_version = model.split("/", 1)
        name, version = name_version.split(":", 1) if ":" in name_version else (name_version, "latest")
        
        resp = requests.get(
            f"https://api.replicate.com/v1/models/{owner}/{name}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        
        if resp.status_code == 404:
            error_msg = f"Modello '{model}' non trovato (404). Verifica il nome su replicate.com"
            if tracker:
                tracker.add_substep(tracker.current_step, f"❌ {error_msg}", "failed")
                if timeline_container:
                    # Aggiorna timeline per mostrare l'errore
                    with timeline_container:
                        st.error(f"❌ MODELLO NON TROVATO: `{model}`")
                        st.markdown("### 🔄 MODELLI VERIFICATI:")
                        st.code("black-forest-labs/flux-schnell")
                        st.code("stability-ai/stable-diffusion-xl-base-1.0")
            elif st:
                st.error(f"❌ MODELLO NON TROVATO: `{model}`")
            raise ValueError(error_msg)
        elif resp.status_code == 200 and tracker:
            tracker.add_substep(tracker.current_step, f"✅ Modello verificato: {model}", "completed")
            _update_timeline(tracker, timeline_container, display_timeline_func)
        
    except requests.exceptions.RequestException as e:
        warning_msg = f"Impossibile verificare modello: {e} - Procedendo..."
        if tracker:
            tracker.add_substep(tracker.current_step, f"⚠️ {warning_msg}", "completed")
            _update_timeline(tracker, timeline_container, display_timeline_func)
        elif st:
            st.warning(f"⚠️ {warning_msg}")

    # Info debug
    masked = (api_key[:3] + "…" + api_key[-4:]) if len(api_key) > 8 else "—"
    if tracker:
        tracker.add_substep(tracker.current_step, f"🔐 Token: {masked} | 🧩 Modello: {model}", "completed")
        # Update timeline per mostrare info iniziali
        _update_timeline(tracker, timeline_container, display_timeline_func)
    elif st:
        st.write(f"🔐 Token: {masked} | 🧩 Modello: `{model}`")

    results = []
    failed_chunks = []
    start_time = time.time()
    
    for idx, prompt in enumerate(chunks, start=1):
        # Timeline per ogni immagine
        step_start = time.time()
        
        if tracker:
            # Aggiungi substep per questa immagine
            img_substep = f"🎨 Immagine {idx}/{len(chunks)}: Generazione..."
            tracker.add_substep(tracker.current_step, img_substep, "running")
            _update_timeline(tracker, timeline_container, display_timeline_func)
        elif st:
            st.write(f"🎨 Generando immagine {idx}/{len(chunks)}...")

        model_input = {"prompt": prompt}
        model_input.setdefault("aspect_ratio", (cfg or {}).get("aspect_ratio", "16:9"))
        if isinstance(extra_input, dict):
            model_input.update(extra_input)

        try:
            output = client.run(model, input=model_input)
            
            # Estrai URLs dalle diverse strutture di risposta
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

            # Successo - aggiorna timeline
            step_duration = time.time() - step_start
            elapsed_total = time.time() - start_time
            
            if tracker:
                # Aggiorna substep completato
                success_msg = f"✅ Immagine {idx}/{len(chunks)}: {os.path.basename(outpath)} ({step_duration:.1f}s)"
                tracker.steps[tracker.current_step]["substeps"][-1]["description"] = success_msg
                tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "completed"
                
                # Aggiungi ETA se abbiamo abbastanza dati
                if idx >= 2:
                    avg_time = elapsed_total / idx
                    remaining = len(chunks) - idx
                    eta_minutes = (remaining * avg_time) / 60
                    tracker.add_substep(tracker.current_step, f"⏱️ ETA rimanente: {eta_minutes:.1f} min", "completed")
                
                _update_timeline(tracker, timeline_container, display_timeline_func)
            elif st:
                st.write(f"✅ Immagine {idx} generata: `{os.path.basename(outpath)}` ({step_duration:.1f}s)")

        except replicate.exceptions.ReplicateError as e:
            error_msg = str(e)
            if "404" in error_msg:
                final_error = f"Modello '{model}' non trovato. Vai su replicate.com per trovare modelli disponibili."
                if tracker:
                    tracker.add_substep(tracker.current_step, f"❌ Immagine {idx}: Modello non trovato", "failed")
                    if timeline_container:
                        display_timeline(tracker, timeline_container)
                elif st:
                    st.error(f"❌ MODELLO INESISTENTE: `{model}`")
                raise ValueError(final_error)
            else:
                failed_chunks.append(idx)
                fail_msg = f"❌ Errore Replicate immagine {idx}: {error_msg[:100]}"
                if tracker:
                    tracker.steps[tracker.current_step]["substeps"][-1]["description"] = f"❌ Immagine {idx}: Errore API"
                    tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "failed"
                    if timeline_container:
                        display_timeline(tracker, timeline_container)
                elif st:
                    st.error(fail_msg)
                else:
                    print(f"[ERROR] {fail_msg}")
                
        except Exception as e:
            failed_chunks.append(idx)
            fail_msg = f"❌ Errore generico immagine {idx}: {e}"
            if tracker:
                tracker.steps[tracker.current_step]["substeps"][-1]["description"] = f"❌ Immagine {idx}: Errore generico"
                tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "failed"
                if timeline_container:
                    display_timeline(tracker, timeline_container)
            elif st:
                st.error(fail_msg)
            else:
                print(f"[ERROR] {fail_msg}")

        # Sleep tra chiamate (rate limiting)
        if sleep_between_calls > 0 and idx < len(chunks):
            if tracker:
                tracker.add_substep(tracker.current_step, f"⏳ Pausa {sleep_between_calls}s (rate limiting)", "completed")
            time.sleep(sleep_between_calls)

    # Report finale
    total_time = time.time() - start_time
    
    if tracker:
        if failed_chunks:
            tracker.add_substep(tracker.current_step, f"⚠️ {len(failed_chunks)} fallite su {len(chunks)}", "completed")
        tracker.add_substep(tracker.current_step, f"🏁 Totale: {len(results)} immagini in {total_time:.1f}s", "completed")
    elif failed_chunks and st:
        st.warning(f"⚠️ {len(failed_chunks)} immagini fallite su {len(chunks)} totali")
    
    if not results:
        error_msg = "Nessuna immagine generata con successo. Controlla modello e API key."
        if tracker:
            tracker.add_substep(tracker.current_step, f"❌ {error_msg}", "failed")
        raise RuntimeError(error_msg)

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
    """
    Concatena MP3 usando ffmpeg con fallback per sistemi senza ffmpeg
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

def _download_with_retry(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    """Download con retry automatico e timeout esteso"""
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:  # Non aspettare nell'ultimo tentativo
                time.sleep(1.5 * (attempt + 1))  # Backoff progressivo
    raise last_exc or RuntimeError("Download fallito dopo tutti i retry")

def generate_audio(chunks, cfg: dict, outdir: str, tts_endpoint: str = "https://api.fish.audio/v1/tts"):
    """
    Genera audio con timeline integrata e ottimizzazioni per script lunghi
    
    Args:
        chunks: Lista di testi da convertire in audio
        cfg: Configurazione con API keys e parametri
        outdir: Directory di output
        tts_endpoint: Endpoint API FishAudio
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)
    
    # Estrai configurazione
    api_key = (cfg or {}).get("fishaudio_api_key")
    voice_id = (cfg or {}).get("fishaudio_voice") or (cfg or {}).get("fishaudio_voice_id")
    model = (cfg or {}).get("fishaudio_model")
    extra = (cfg or {}).get("fishaudio_extra", {})
    
    # Timeline tracker
    tracker = cfg.get("progress_tracker")
    timeline_container = cfg.get("timeline_container")

    if not api_key:
        msg = "FishAudio API key assente. Configura le API keys nella sidebar."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)
    if not voice_id:
        msg = "FishAudio Voice ID assente. Configura Voice ID nella sidebar."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    # Log informazioni chunking
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    min_chars = min(len(chunk) for chunk in chunks) if chunks else 0
    max_chars = max(len(chunk) for chunk in chunks) if chunks else 0
    
    if tracker:
        tracker.add_substep(tracker.current_step, f"📊 {len(chunks)} chunk | Totale: {total_chars:,} char", "completed")
        tracker.add_substep(tracker.current_step, f"📈 Media: {avg_chars:.0f} | Min: {min_chars:,} | Max: {max_chars:,}", "completed")
        
        # Stima durata
        estimated_minutes = (total_chars / 5) / 150  # ~150 parole/min, ~5 char/parola
        tracker.add_substep(tracker.current_step, f"⏱️ Durata stimata: ~{estimated_minutes:.1f} minuti", "completed")
    elif st:
        st.info(f"📊 Audio chunking: {len(chunks)} segmenti | {total_chars:,} caratteri totali")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if model:
        headers["model"] = model

    audio_paths = []
    failed_count = 0
    start_time = time.time()

    for i, text in enumerate(chunks, 1):
        chunk_start_time = time.time()
        
        if tracker:
            # Aggiungi substep per questo chunk
            chunk_substep = f"🎵 Chunk {i}/{len(chunks)} ({len(text)} char): Generazione..."
            tracker.add_substep(tracker.current_step, chunk_substep, "running")
            if timeline_container:
                display_timeline(tracker, timeline_container)
        elif st:
            st.write(f"🎧 Generando audio {i}/{len(chunks)} ({len(text)} caratteri)...")
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
            resp = requests.post(tts_endpoint, headers=headers, json=payload, timeout=180)
            
            if resp.status_code >= 400:
                detail = resp.text[:200]
                failed_count += 1
                
                if tracker:
                    error_msg = f"❌ Chunk {i}: HTTP {resp.status_code}"
                    tracker.steps[tracker.current_step]["substeps"][-1]["description"] = error_msg
                    tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "failed"
                    tracker.add_substep(tracker.current_step, f"🔍 Dettaglio errore: {detail[:50]}...", "failed")
                    if timeline_container:
                        display_timeline(tracker, timeline_container)
                elif st:
                    st.error(f"❌ HTTP {resp.status_code} FishAudio chunk {i}: {detail}")
                else:
                    print(f"HTTP {resp.status_code} FishAudio chunk {i}: {detail}")
                continue

            # Processa risposta
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                data = resp.json()
                audio_url = data.get("audio_url") or data.get("url")
                audio_b64 = data.get("audio_base64") or data.get("audio")
                
                if audio_url:
                    audio_bytes = _download_with_retry(audio_url, retries=5, timeout=90)
                elif audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                else:
                    failed_count += 1
                    if tracker:
                        tracker.steps[tracker.current_step]["substeps"][-1]["description"] = f"❌ Chunk {i}: Risposta JSON invalida"
                        tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "failed"
                    elif st:
                        st.error(f"❌ Risposta JSON inattesa chunk {i}: {data}")
                    else:
                        print(f"JSON inatteso chunk {i}: {data}")
                    continue
            else:
                audio_bytes = resp.content

            # Salva file audio
            path = os.path.join(outdir, f"audio_{i:03d}.mp3")
            with open(path, "wb") as f:
                f.write(audio_bytes)

            # Calcola durata e aggiorna timeline
            dur_sec = mp3_duration_seconds(path)
            chunk_duration = time.time() - chunk_start_time
            elapsed_total = time.time() - start_time
            
            audio_paths.append(path)
            
            if tracker:
                # Aggiorna substep completato
                success_msg = f"✅ Chunk {i}/{len(chunks)}: {dur_sec:.1f}s audio ({chunk_duration:.1f}s gen)"
                tracker.steps[tracker.current_step]["substeps"][-1]["description"] = success_msg
                tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "completed"
                
                # Aggiungi ETA se abbiamo abbastanza dati
                if i >= 3:
                    avg_time_per_chunk = elapsed_total / i
                    remaining_chunks = len(chunks) - i
                    eta_minutes = (remaining_chunks * avg_time_per_chunk) / 60
                    tracker.add_substep(tracker.current_step, f"⏱️ ETA audio: {eta_minutes:.1f} min", "completed")
                
                if timeline_container:
                    display_timeline(tracker, timeline_container)
            elif st:
                st.write(f"✅ TTS chunk {i:03d} durata: {dur_sec:.1f}s (gen: {chunk_duration:.1f}s)")
            else:
                print(f"[TTS] chunk {i:03d} duration: {dur_sec:.1f}s — ✅")

        except Exception as e:
            failed_count += 1
            if tracker:
                tracker.steps[tracker.current_step]["substeps"][-1]["description"] = f"❌ Chunk {i}: Errore connessione"
                tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "failed"
                tracker.add_substep(tracker.current_step, f"🔍 Errore: {str(e)[:50]}...", "failed")
                if timeline_container:
                    display_timeline(tracker, timeline_container)
            elif st:
                st.error(f"❌ Errore TTS sul chunk {i}: {e}")
            else:
                print("❌", f"Errore TTS sul chunk {i}: {e}")
            continue

    # Report intermedio sui fallimenti
    if failed_count > 0:
        if tracker:
            tracker.add_substep(tracker.current_step, f"⚠️ {failed_count} chunk falliti su {len(chunks)}", "completed")
        elif st:
            st.warning(f"⚠️ {failed_count} chunk audio falliti su {len(chunks)} totali")
        else:
            print(f"[WARNING] {failed_count} failed audio chunks out of {len(chunks)}")

    if not audio_paths:
        error_msg = "Nessun chunk audio generato con successo. Controlla API key e voice ID."
        if tracker:
            tracker.add_substep(tracker.current_step, f"❌ {error_msg}", "failed")
        return None

    # Concatenazione con timeline
    if tracker:
        concat_substep = f"🔗 Concatenazione {len(audio_paths)} file audio..."
        tracker.add_substep(tracker.current_step, concat_substep, "running")
        if timeline_container:
            from app import display_timeline
            display_timeline(tracker, timeline_container)
    elif st and len(audio_paths) > 10:
        st.write(f"🔗 Concatenando {len(audio_paths)} file audio...")

    try:
        out_path = os.path.join(outdir, "combined_audio.mp3")
        concat_mp3s(audio_paths, out_path, bitrate_kbps=128)
        
        # Risultato finale
        final_duration = mp3_duration_seconds(out_path)
        total_generation_time = time.time() - start_time
        
        if tracker:
            # Aggiorna substep concatenazione
            tracker.steps[tracker.current_step]["substeps"][-1]["description"] = f"✅ Concatenazione completata"
            tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "completed"
            
            # Aggiungi statistiche finali
            tracker.add_substep(tracker.current_step, f"🎵 Audio finale: {final_duration:.1f}s ({final_duration/60:.1f} min)", "completed")
            tracker.add_substep(tracker.current_step, f"⏱️ Tempo generazione: {total_generation_time/60:.1f} min", "completed")
            
            # Calcola efficienza
            if final_duration > 0:
                efficiency_ratio = final_duration / total_generation_time
                tracker.add_substep(tracker.current_step, f"⚡ Efficienza: {efficiency_ratio:.2f}x (audio/tempo)", "completed")
            
            if timeline_container:
                display_timeline(tracker, timeline_container)
        elif st:
            st.success(f"🔊 Audio finale: {final_duration:.1f}s ({final_duration/60:.1f} min)")
            st.info(f"⏱️ Tempo generazione totale: {total_generation_time/60:.1f} minuti")
        else:
            print(f"[TTS] Final audio: {final_duration:.2f}s ({final_duration/60:.1f}min)")
            print("🔊 Audio finale creato.")

        return out_path
        
    except Exception as e:
        error_msg = f"Errore durante concatenazione: {e}"
        if tracker:
            tracker.steps[tracker.current_step]["substeps"][-1]["description"] = f"❌ Concatenazione fallita"
            tracker.steps[tracker.current_step]["substeps"][-1]["status"] = "failed"
            tracker.add_substep(tracker.current_step, f"🔍 Errore concat: {str(e)[:50]}...", "failed")
        elif st:
            st.error(f"❌ {error_msg}")
        
        raise RuntimeError(error_msg)
