"""
Load application configuration from app.yml.
This module provides a simple interface for accessing config values.
"""

import yaml
import os


def load_config() -> dict:
    """
    Load the YAML configuration file and return it as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.yml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load once at import time
CONFIG = load_config()
