import yaml, os

def load_config():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(base, 'config', 'settings.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

