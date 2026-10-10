"""Load the project configuration and resolve paths relative to the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config.yaml'


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve(path):
    """Return an absolute path; relative paths are interpreted from the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def ttc_threshold(cfg):
    """TTC (seconds) below which a vehicle is considered high risk."""
    return cfg['warning']['reaction_time_s'] * cfg['warning']['ttc_ratio']
