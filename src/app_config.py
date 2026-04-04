"""Hydra-backed application configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from src.logging_utils import get_logger


logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "conf"


def load_settings(overrides: Optional[list[str]] = None) -> DictConfig:
    """
    Load the Hydra application configuration.

    :param overrides: Optional list of Hydra overrides to apply when loading the configuration.
    :type overrides: Optional[list[str]]
    :return: The loaded Hydra configuration as a DictConfig object.
    :rtype: DictConfig
    """
    load_dotenv()

    deployment = os.getenv("APP_ENV", "local").strip() or "local"
    selected_overrides = [f"deployment={deployment}"]
    if overrides:
        selected_overrides.extend(overrides)

    logger.info(
        "Loading Hydra configuration from '%s' with deployment '%s'",
        _CONFIG_DIR,
        deployment,
    )

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        return compose(config_name="config", overrides=selected_overrides)
