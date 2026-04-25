from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from investment_backtest_lab.models import BacktestConfig


def load_config_dict(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(text)
    elif config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping.")
    return loaded


def load_backtest_config(path: str | Path) -> BacktestConfig:
    return BacktestConfig.from_dict(load_config_dict(path))
