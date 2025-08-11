from config_loader import load_config

config = load_config()

print("Claude API Key:", config["claude_api_key"])
print("Modello immagini:", config["model"]["image_model"])

