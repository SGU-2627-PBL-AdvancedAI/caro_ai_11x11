"""Kiểm thử đơn vị cho ứng dụng giao diện Caro 11x11 (`src/ui_app.py`)."""

from __future__ import annotations

from pathlib import Path
import threading

import pytest

from src.caro_env import CaroEnv
from src.ui_app import (
    AGENT_CATALOG,
    UIAgentFactory,
    export_ai_vs_ai_analytics_chart,
    find_winning_cells,
    load_match_logs,
    run_ai_vs_ai_series,
)


def test_agent_catalog_clean_naming():
    assert [item["level"] for item in AGENT_CATALOG] == [1, 2, 3, 4, 5, 6, 7]
    for item in AGENT_CATALOG:
        for forbidden in ("5B", "6A", "7A", "7B"):
            assert forbidden not in item["label"]
            assert forbidden not in item["id"]


def test_ui_agent_factory_all_seven_levels():
    factory = UIAgentFactory()
    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)  # X trung tâm
    env.step(49)  # O lân cận

    valid = set(env.legal_actions())
    for lv in range(1, 8):
        agent = factory.create(level=lv, seed=42, fast_mode=True)
        act = int(agent.select_action(env))
        assert act in valid, f"Level {lv} trả về nước không hợp lệ: {act}"

    with pytest.raises(ValueError):
        factory.create(level=9)


def test_find_winning_cells():
    env = CaroEnv(board_size=11, win_length=5)
    # X đánh ngang hàng 5: (5,2)..(5,6), O đánh hàng 0
    for i in range(5):
        env.step(5 * 11 + (2 + i))
        if not env.is_terminal():
            env.step(i)
    cells = find_winning_cells(env)
    assert len(cells) == 5
    assert cells == {(5, 2), (5, 3), (5, 4), (5, 5), (5, 6)}


def test_ai_vs_ai_series_logging_and_chart_export(tmp_path: Path):
    factory = UIAgentFactory()
    log_path = tmp_path / "ai_vs_ai_L2_vs_L1.jsonl"
    chart_path = tmp_path / "analytics.png"
    csv_path = tmp_path / "analytics.csv"

    recs = run_ai_vs_ai_series(
        factory=factory,
        level_a=2,
        level_b=1,
        num_games=4,
        swap_sides=True,
        diverse_openings=True,
        fast_mode=True,
        base_seed=42,
        log_path=log_path,
    )
    assert len(recs) == 4
    assert log_path.exists()

    loaded = load_match_logs(log_path)
    assert len(loaded) == 4
    assert sum(1 for r in loaded if r["a_side"] == "X") == 2
    assert sum(1 for r in loaded if r["a_side"] == "O") == 2
    assert all(r["num_moves"] == len(r["history"]) >= 9 for r in loaded)

    out_png = export_ai_vs_ai_analytics_chart(loaded, chart_path, csv_summary_path=csv_path)
    assert out_png.exists() and out_png.stat().st_size > 10_000
    assert csv_path.exists()

    # Kiểm thử dừng sớm bằng stop_event
    ev = threading.Event()
    ev.set()
    stopped = run_ai_vs_ai_series(
        factory=factory,
        level_a=1,
        level_b=1,
        num_games=4,
        stop_event=ev,
    )
    assert stopped == []


def test_ui_app_window_smoke_construct():
    """Kiểm tra khởi tạo & chuyển chế độ cửa sổ Tkinter không lỗi (tự bỏ qua nếu không có display)."""
    import tkinter as tk
    from src.ui_app import CaroUIApp

    try:
        app = CaroUIApp()
    except tk.TclError:
        pytest.skip("Môi trường headless không có display Tkinter.")
        return

    try:
        app.root.withdraw()
        app.mode_var.set("ai_vs_ai")
        app._on_mode_change()
        app.mode_var.set("human_vs_ai")
        app._on_mode_change()
        app.root.update_idletasks()
    finally:
        app.root.destroy()
