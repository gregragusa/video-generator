# scripts/utils.py
# -------------------------------------------------------
# Utility per chunking + generazione IMMAGINI (Replicate) e AUDIO (FishAudio)
# con gestione errori, normalizzazione output e logging "amichevole".
# -------------------------------------------------------

import os
import re
import time
from io import BytesIO

import requests
from PIL import Image
from pydub import AudioSegment

# ===========================
# Chunking helpers
# ===========================
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


# ===========================
# Helpers vari
# ===========================
def _st():
    """Ritorna streamlit se disponibile, altrimenti None."""
    try:
        import streamlit as st  # type: ignore
        return st
    except Exception:
        return None


def save_image_from_url(url: str, path: str, timeout: int = 30):
    """Scarica immagine da URL e salva su disco (converte in PNG se serve)."""
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.mode in ("P", "RGBA"):
        img = img.convert("RGB")
    img.save(path)


def _download_first(urls, dest_path: str):
    """Salva la prima URL valida da una lista in dest_path."""
    if not urls:
        raise ValueError("Nessuna URL immagine restituita dal modello.")
    save_image_from_url(urls[0], dest_path)
    return dest_path


# ===========================
# IMMAGINI (Replicate)
# ===========================
def generate_images(
    chunks,
    cfg: dict,
    outdir: str,
    sleep_between_calls: float = 11.0,  # anti rate-limit
):
    """
    Genera 1 immagine per ogni elemento di `chunks` usando Replicate.

    Config letta da:
      - API: cfg['replicate_api_token'] | cfg['replicate_api_key'] | env REPLICATE_API_TOKEN
      - Modello: cfg['image_model'] | cfg['replicate_model'] (es. 'owner/name:tag')
      - Extra input: cfg['replicate_input'] (dict) opzionale (width/height/steps/guidance...)
      - aspect_ratio default da cfg['aspect_ratio'] o '16:9'
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
        msg = "Modello Replicate assente. Imposta 'image_model' o 'replicate_model' in cfg (es. 'owner/name:tag')."
        if st: st.error("❌ " + msg)
        raise ValueError(msg)

    # Import qui per evitare dipendenza se non serve
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

            # Normalizza output in lista di URL
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
            # Identifica ReplicateError se disponibile
            try:
                ReplicateError = replicate.exceptions.ReplicateError
            except Exception:
                ReplicateError = Exception

            if isinstance(e, ReplicateError):
                msg = f"ReplicateError su chunk {idx}: {e}"
            else:
                msg = f"Errore su chunk {idx}: {e}"

            if st: st.error("❌ " + msg)
            else: print("[ERROR]", msg)
            raise

        if sleep_between_calls and idx < len(chunks):
            time.sleep(sleep_between_calls)

    return results


# ===========================
# AUDIO (FishAudio)
# ===========================
def _download_with_retry(url: str, retries: int = 3, timeout: int = 30) -> bytes:
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


def generate_audio(
    chunks,
    cfg: dict,
    outdir: str,
    tts_endpoint: str = "https://api.fish.audio/v1/tts",
):
    """
    Genera audio da ogni blocco di testo in `chunks` con FishAudio e li concatena.

    Config usata:
      - cfg['fishaudio_api_key']          (obbligatoria)
      - cfg['fishaudio_voice'] | cfg['fishaudio_voice_id']  (obbligatoria)
      - cfg['fishaudio_model']            (opzionale: se presente la inviamo)
      - opzionale: cfg['fishaudio_extra'] (dict) per altri campi (format, bitrate, ecc.)
    """
    st = _st()
    os.makedirs(outdir, exist_ok=True)

    api_key = (cfg or {}).get("fishaudio_api_key")
    voice_id = (cfg or {}).get("fishaudio_voice") or (cfg or {}).get("fishaudio_voice_id")
    model = (cfg or {}).get("fishaudio_model")  # opzionale
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
    # compat: se vuoi inviarlo in header come nel tuo codice originale
    if model:
        headers["model"] = model

    audio_paths = []
    for i, text in enumerate(chunks, 1):
        if st:
            st.write(f"🎧 Audio {i}/{len(chunks)}…")
        else:
            print(f" • Audio {i}/{len(chunks)}…", end=" ")

        payload = {
            "text": text,
            "reference_id": voice_id,
            # default comuni
            "format": "mp3",
            "mp3_bitrate": 128,
        }
        if model:
            payload["model"] = model
        if isinstance(extra, dict):
            payload.update(extra)

        try:
            resp = requests.post(tts_endpoint, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()

            # Alcune API restituiscono JSON con 'audio_url'; altre restituiscono binario
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = resp.json()
                audio_url = data.get("audio_url") or data.get("url")
                if not audio_url:
                    raise RuntimeError(f"Risposta JSON inattesa: {data}")
                audio_bytes = _download_with_retry(audio_url, retries=3, timeout=60)
            else:
                audio_bytes = resp.content

            path = os.path.join(outdir, f"audio_{i:02d}.mp3")
            with open(path, "wb") as f:
                f.write(audio_bytes)

            seg = AudioSegment.from_file(path)
            if st:
                st.write(f"✅ TTS chunk {i:02d} durata: {len(seg)} ms")
            else:
                print(f"[TTS] chunk {i:02d} duration: {len(seg)} ms — ✅")

            audio_paths.append(path)

        except Exception as e:
            msg = f"Errore TTS sul chunk {i}: {e}"
            if st: st.error("❌ " + msg)
            else: print("❌", msg)
            # continua col chunk successivo

    if not audio_paths:
        raise RuntimeError("Nessun audio generato.")

    combined = AudioSegment.empty()
    for p in audio_paths:
        combined += AudioSegment.from_mp3(p)

    if st:
        st.write(f"🔊 Durata totale audio: {len(combined)} ms")
    else:
        print(f"[TTS] combined duration: {len(combined)} ms")

    out_path = os.path.join(outdir, "combined_audio.mp3")
    combined.export(out_path, format="mp3")

    if st:
        st.success("🔊 Audio finale creato.")
    else:
        print("🔊 Audio finale creato.")

    return out_path
