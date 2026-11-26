"""Kiểm thử đơn vị cho ứng dụng web Streamlit Caro 11×11 (`src/ui_app.py`)."""

from __future__ import annotations

from pathlib import Path
import threading

import pytest

from src.caro_env import CaroEnv
from src.ui_app import (
    AGENT_CATALOG,
    UIAgentFactory,
    action_to_coord,
    export_ai_vs_ai_analytics_chart,
    extract_ai_explainability,
    find_winning_cells,
    load_match_logs,
    render_analytics_chart_bytes,
    render_board_svg,
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
    for i in range(5):
        env.step(5 * 11 + (2 + i))
        if not env.is_terminal():
            env.step(i)
    cells = find_winning_cells(env)
    assert len(cells) == 5
    assert cells == {(5, 2), (5, 3), (5, 4), (5, 5), (5, 6)}


def test_render_board_svg_and_coords():
    assert action_to_coord(0) == "A1"
    assert action_to_coord(60) == "F6"
    assert action_to_coord(120) == "K11"

    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)
    env.step(49)
    svg = render_board_svg(env, show_move_numbers=True, hint_action=71)
    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>")
    assert "hinokiWood" in svg
    assert ">1<" in svg and ">2<" in svg


def test_explainability_for_neural_levels():
    factory = UIAgentFactory()
    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)
    env.step(49)

    for lv in (5, 6, 7):
        ag = factory.create(level=lv, seed=42, fast_mode=True)
        ag.select_action(env)
        exp = extract_ai_explainability(ag, env, lv, top_k=3)
        assert exp is not None
        assert len(exp["rows"]) >= 1


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

    png_bytes = render_analytics_chart_bytes(loaded)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 10_000

    out_png = export_ai_vs_ai_analytics_chart(loaded, chart_path, csv_summary_path=csv_path)
    assert out_png.exists() and out_png.stat().st_size > 10_000
    assert csv_path.exists()

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


def test_streamlit_app_smoke_and_navigation():
    """Kiểm tra khởi chạy ứng dụng web Streamlit không lỗi qua AppTest headless."""
    streamlit = pytest.importorskip("streamlit")
    _ = streamlit
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src" / "ui_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Lỗi khi khởi chạy Streamlit UI: {at.exception}"

    # Có đủ 121 ô cờ tương tác (11x11) + nút điều khiển ở tab Người vs Máy
    assert len(at.button) >= 121

    # Chuyển sang tab Máy vs Máy và Lịch sử & Replay không lỗi
    at.sidebar.radio(key="studio_nav_mode").set_value("⚔️ Máy vs Máy").run()
    assert not at.exception

    at.sidebar.radio(key="studio_nav_mode").set_value("📜 Lịch sử & Replay").run()
    assert not at.exception
