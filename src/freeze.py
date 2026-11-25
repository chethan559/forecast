"""Freeze the final config before Phase 1:   python -m src.freeze

Writes outputs/phase1/frozen_config.json (full config + sha256 of every src/*.py).
src.run_phase1 refuses to run if anything differs from this snapshot.
"""
from src.config import get_config
from src.run_phase1 import freeze

if __name__ == "__main__":
    print(f"frozen: {freeze(get_config())}")
