from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Tuple

def load_config(repo_root: Path) -> DictConfig:
    cfg_path = repo_root / "config.yml"

    cfg = OmegaConf.load(str(cfg_path))
    return cfg  # type: ignore

def load_secrets(repo_root: Path) -> DictConfig:
    sec_path = repo_root / "secrets.yml"

    sec = OmegaConf.load(str(sec_path))
    return sec  # type: ignore


def load_ui_config(repo_root: Path) -> DictConfig:
    ui_path = repo_root / "ui.yml"
    return OmegaConf.load(str(ui_path)) # type: ignore