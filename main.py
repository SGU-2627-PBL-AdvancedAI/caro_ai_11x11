"""Điểm khởi chạy chính của ứng dụng Caro AI 11x11 (Người vs Máy & Máy vs Máy)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui_app import launch_ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Khởi chạy giao diện Caro AI 11x11 (7 Cấp độ AI — Người vs Máy & Máy vs Máy)."
    )
    parser.parse_args()
    launch_ui(project_root=PROJECT_ROOT)


if __name__ == "__main__":
    main()
