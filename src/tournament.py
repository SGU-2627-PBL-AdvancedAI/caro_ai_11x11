
"""Utilities for playing and evaluating Caro agents."""

from __future__ import annotations

import time
from typing import Callable

from src.caro_env import CaroEnv


def play_game(
    agent_x,
    agent_o,
    board_size=11,
    win_length=5,
    start_player=1,
    record=False
):
    """
    Play one Caro game.

    Returns:
        winner: 1 (X), -1 (O), None (Draw)
        moves: Total number of moves
        timings: Average action selection time
        history: Board states if record=True
    """

    if start_player not in (1, -1):
        raise ValueError("start_player phải là 1 hoặc -1.")

    env = CaroEnv(board_size, win_length)
    env.current_player = int(start_player)

    history = [env.board.copy()] if record else None

    timings = {
        1: [],
        -1: []
    }

    agents = {
        1: agent_x,
        -1: agent_o
    }

    # Main game loop
    while not env.is_terminal():

        player = env.current_player
        agent = agents[player]

        # Measure agent decision time
        t0 = time.perf_counter()
        action = agent.select_action(env)
        elapsed = (time.perf_counter() - t0) * 1000.0

        timings[player].append(elapsed)

        # Execute action
        _, _, done, info = env.step(action)

        # Record board state
        if record:
            history.append(env.board.copy())

    # Normalize winner: 0 -> None (Draw)
    winner = env.winner if env.winner != 0 else None

    return {
        "winner": winner,
        "moves": len(env.history),

        "x_agent": agent_x.name,
        "o_agent": agent_o.name,

        "x_avg_ms": (
            sum(timings[1]) / len(timings[1])
            if timings[1] else 0.0
        ),

        "o_avg_ms": (
            sum(timings[-1]) / len(timings[-1])
            if timings[-1] else 0.0
        ),

        "history": history,
    }


def run_matchup(
    agent_factory_a: Callable,
    agent_factory_b: Callable,
    games=20,
    board_size=11,
    win_length=5,
    seed=42
):
    """
    Run multiple games between Agent A and Agent B.

    A/B alternate the first-player (X) seat.
    """

    rows = []

    for i in range(int(games)):

        agent_a = agent_factory_a(seed + i)
        agent_b = agent_factory_b(seed + 10_000 + i)

        # Alternate X/O positions
        if i % 2 == 0:

            # A plays X, B plays O
            res = play_game(
                agent_a,
                agent_b,
                board_size,
                win_length,
                start_player=1
            )

            winner = (
                "A" if res["winner"] == 1
                else "B" if res["winner"] == -1
                else "Draw"
            )

            x_agent = agent_a
            o_agent = agent_b

        else:

            # B plays X, A plays O
            res = play_game(
                agent_b,
                agent_a,
                board_size,
                win_length,
                start_player=1
            )

            winner = (
                "A" if res["winner"] == -1
                else "B" if res["winner"] == 1
                else "Draw"
            )

            x_agent = agent_b
            o_agent = agent_a

        rows.append({
            "game": i + 1,
            "winner": winner,
            "moves": res["moves"],

            "x_avg_ms": res["x_avg_ms"],
            "o_avg_ms": res["o_avg_ms"],

            "first_player": "A" if i % 2 == 0 else "B",

            "x_agent": x_agent.name,
            "o_agent": o_agent.name,
        })

    return rows


__all__ = [
    "play_game",
    "run_matchup"
]
