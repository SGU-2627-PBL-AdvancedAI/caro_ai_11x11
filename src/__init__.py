"""Gói nguồn chính cho dự án Caro AI 11x11."""

from __future__ import annotations

from .caro_env import CaroEnv
from .heuristic import (
    MAX_HEURISTIC_SCORE,
    TERMINAL_WIN_SCORE,
    evaluate_board,
    evaluate_cell_threat,
    extract_board_patterns,
    get_candidate_actions,
)

__all__ = [
    "CaroEnv",
    "TERMINAL_WIN_SCORE",
    "MAX_HEURISTIC_SCORE",
    "get_candidate_actions",
    "evaluate_cell_threat",
    "evaluate_board",
    "extract_board_patterns",
]
