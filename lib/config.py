from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Tuple

def load_configs(repo_root: Path) -> Tuple[DictConfig, DictConfig]:
    cfg_path = repo_root / "config.yml"
    sec_path = repo_root / "secrets.yml"

    cfg = OmegaConf.load(str(cfg_path))
    sec = OmegaConf.load(str(sec_path))
    return cfg, sec  # type: ignore
