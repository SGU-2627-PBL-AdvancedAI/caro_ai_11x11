"""Điểm khởi chạy chính của ứng dụng web Streamlit Caro AI 11×11 (7 Cấp độ AI)."""

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
        description="Khởi chạy giao diện web Streamlit Caro AI 11×11 (7 Cấp độ AI — Người vs Máy, Máy vs Máy & Replay)."
    )
    parser.add_argument("--port", type=int, default=8501, help="Cổng chạy máy chủ Streamlit (mặc định: 8501).")
    parser.add_argument("--headless", action="store_true", help="Chạy ở chế độ headless (không tự mở trình duyệt).")
    args = parser.parse_args()
    launch_ui(project_root=PROJECT_ROOT, port=args.port, headless=args.headless)


if __name__ == "__main__":
    main()
