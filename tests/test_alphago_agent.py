from __future__ import annotations

import numpy as np
import pytest

from src.agents.alphago_agent import AlphaGoAgent
from src.caro_env import CaroEnv


def _midgame_env() -> CaroEnv:
    env = CaroEnv(board_size=11, win_length=5)
    for m in [60, 49, 61, 38, 59, 71]:
        env.step(m)
    return env


@pytest.mark.parametrize("bad_sims", [0, -5, True, "50"])
def test_alphago_invalid_num_simulations(bad_sims):
    with pytest.raises(ValueError):
        AlphaGoAgent(auto_load_default=False, num_simulations=bad_sims, time_limit_sec=None)


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.2, True, float("nan")])
def test_alphago_invalid_blend_alpha(bad_alpha):
    with pytest.raises(ValueError):
        AlphaGoAgent(auto_load_default=False, num_simulations=10, time_limit_sec=None, value_blend_alpha=bad_alpha)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_alphago_blend_modes_and_explainability(alpha: float):
    """Kiểm tra cả 3 chế độ pha trộn (Heuristic / Hybrid / Pure Neural) và API giải thích."""
    agent = AlphaGoAgent(
        auto_load_default=False,
        num_simulations=20,
        time_limit_sec=None,
        value_blend_alpha=alpha,
        use_tactical_shortcut=False,
        device="cpu",
        seed=42,
    )
    env = _midgame_env()
    board_before = env.board.copy()
    hist_before = env.history.copy()

    action = agent.select_action(env)
    assert isinstance(action, int)
    assert action in env.legal_actions()
    assert np.array_equal(env.board, board_before)
    assert env.history == hist_before

    assert agent.last_search_count == 20
    assert agent.last_root is not None
    assert agent.last_root.visits == 20
    assert agent.last_move_time_ms > 0.0

    exp = agent.explain_decision(top_k=5)
    assert exp["simulations"] == 20
    assert exp["value_blend_alpha"] == alpha
    assert len(exp["top_moves"]) >= 1
    assert exp["top_moves"][0]["action"] == action
    for item in exp["top_moves"]:
        assert 0.0 <= item["q_mcts"] <= 1.0
        assert 0.0 <= item["prior"] <= 1.0


def test_alphago_shortcuts_and_terminal_error():
    agent = AlphaGoAgent(
        auto_load_default=False,
        num_simulations=15,
        time_limit_sec=None,
        use_tactical_shortcut=True,
        device="cpu",
        seed=42,
    )

    # Bàn cờ trống -> single candidate center 60
    empty_env = CaroEnv()
    assert agent.select_action(empty_env) == 60
    assert agent.last_search_count == 0
    assert agent.last_root is None

    # Thắng ngay 5 quân
    win_env = CaroEnv()
    for c in range(4):
        win_env.step(5 * 11 + c)
        win_env.step(2 * 11 + c)
    assert agent.select_action(win_env) == 5 * 11 + 4
    assert agent.last_decision_stage == "Shortcut: immediate win"

    # Bàn đã kết thúc -> ValueError
    win_env.step(5 * 11 + 4)
    assert win_env.is_terminal()
    with pytest.raises(ValueError):
        agent.select_action(win_env)


def test_alphago_determinism_with_same_seed():
    env = _midgame_env()
    a1 = AlphaGoAgent(auto_load_default=False, num_simulations=24, time_limit_sec=None, device="cpu", seed=77)
    a2 = AlphaGoAgent(auto_load_default=False, num_simulations=24, time_limit_sec=None, device="cpu", seed=77)
    m1 = a1.select_action(env)
    m2 = a2.select_action(env)
    assert m1 == m2
    assert a1.get_root_stats() == a2.get_root_stats()
