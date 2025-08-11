#!/usr/bin/env python3
import os
import requests
from config_loader import load_config

def main():
    cfg = load_config()
    headers = {"Authorization": f"Bearer {cfg['fishaudio_api_key']}"}
    url = "https://api.fish.audio/v1/tts"  # endpoint che stiamo testando

    payload = {"text": "Questo è un test di debug per capire la risposta."}
    print("🚧 Invio richiesta a Fish Audio...")
    resp = requests.post(url, headers=headers, json=payload, timeout=30)

    # Stampa status code e body grezzo
    print("Status code:", resp.status_code)
    print("Response text:")
    print(resp.text)

    # Prova a fare il JSON parse (facoltativo)
    try:
        data = resp.json()
        print("JSON decoded:", data)
    except Exception as e:
        print("❌ Errore parsing JSON:", e)

if __name__ == "__main__":
    main()

