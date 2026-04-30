from __future__ import annotations

import importlib
import os
from pathlib import Path

cache_dir = Path(".cache/matplotlib")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "vectorbt",
    "bt",
    "quantstats",
    "yfinance",
    "FinMind",
    "investment_backtest_lab",
]


def main() -> None:
    failures: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package}: {version}")
        except Exception as exc:
            failures.append(f"{package}: {exc}")
            print(f"[FAIL] {package}: {exc}")

    if failures:
        raise SystemExit("Import smoke test failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
