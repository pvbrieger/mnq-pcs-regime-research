"""Validate the local MNQ PCS research environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

REQUIRED_SHARED_FOLDERS = (
    "NDX",
    "QQQ",
    "SPY",
    "VIX",
    "economic-events",
)

LOCAL_FOLDERS = (
    PROJECT_ROOT / "data" / "cache",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "database",
    PROJECT_ROOT / "results",
)


def main() -> int:
    print("MNQ PCS Regime Research — Environment Check")
    print("=" * 47)
    print(f"Project root: {PROJECT_ROOT}")

    if not ENV_FILE.exists():
        print(f"ERROR: Missing environment file: {ENV_FILE}")
        print("Copy .env.example to .env and enter this computer's Google Drive path.")
        return 1

    load_dotenv(dotenv_path=ENV_FILE)

    shared_path_text = os.getenv("SHARED_MARKET_DATA")
    if not shared_path_text:
        print("ERROR: SHARED_MARKET_DATA is not defined in .env")
        return 1

    shared_path = Path(shared_path_text).expanduser()
    print(f"Shared data: {shared_path}")

    if not shared_path.exists():
        print("ERROR: Shared market-data folder does not exist.")
        return 1

    missing_folders = [
        folder
        for folder in REQUIRED_SHARED_FOLDERS
        if not (shared_path / folder).is_dir()
    ]

    if missing_folders:
        print("ERROR: Missing required shared-data folders:")
        for folder in missing_folders:
            print(f" - {folder}")
        return 1

    for folder in LOCAL_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nShared folders:")
    for folder in REQUIRED_SHARED_FOLDERS:
        folder_path = shared_path / folder
        file_count = sum(1 for item in folder_path.iterdir() if item.is_file())
        print(f" - {folder}: {file_count} file(s)")

    print("\nLocal working folders:")
    for folder in LOCAL_FOLDERS:
        print(f" - {folder.relative_to(PROJECT_ROOT)}")

    print("\nEnvironment check completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
