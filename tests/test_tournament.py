"""Kiểm thử đơn vị cho hệ thống giải đấu Round-Robin & Ablation (`src/tournament.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents import RandomAgent, RuleBasedAgent
from src.tournament import (
    DEFAULT_SEEDS,
    GAME_SPEC,
    OPENING_BOOK_11X11,
    AgentSpec,
    bootstrap_ci95,
    build_agent_specs,
    collect_hardware_info,
    compute_elo_ratings,
    get_process_rss_mb,
    percentile_p95,
    play_single_game,
    run_pair_match,
    run_round_robin,
    wilson_ci95,
)


def test_wilson_and_bootstrap_ci95():
    lo, hi = wilson_ci95(50.0, 100)
    assert 39.0 < lo < 41.0
    assert 59.0 < hi < 61.0
    assert wilson_ci95(0.0, 0) == (0.0, 0.0)

    # All wins -> Wilson hi = 100, lo < 100; Bootstrap = (100, 100)
    w_lo, w_hi = wilson_ci95(10.0, 10)
    assert w_hi == 100.0 and w_lo < 100.0
    assert bootstrap_ci95([1.0] * 10, n_boot=500) == (100.0, 100.0)

    b_lo, b_hi = bootstrap_ci95([1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0] * 5, n_boot=500, seed=42)
    assert 0.0 <= b_lo < b_hi <= 100.0


def test_percentile_p95_and_elo_ratings():
    vals = [float(i) for i in range(1, 101)]
    p95 = percentile_p95(vals)
    assert 94.0 <= p95 <= 96.5
    assert percentile_p95([]) == 0.0

    # Elo Bradley-Terry: A beats B and C, B beats C -> Elo(A) > Elo(B) > Elo(C), finite even at 100%/0%
    matchups = [
        {"agent_a": "A", "agent_b": "B", "games": 20, "wins_a": 18, "draws": 0},
        {"agent_a": "A", "agent_b": "C", "games": 20, "wins_a": 20, "draws": 0},
        {"agent_a": "B", "agent_b": "C", "games": 20, "wins_a": 17, "draws": 1},
    ]
    elos = compute_elo_ratings(matchups, ["A", "B", "C"])
    assert elos["A"] > elos["B"] > elos["C"]
    assert abs(sum(elos.values()) / 3.0 - 1500.0) < 0.5


def test_hardware_and_game_spec():
    hw = collect_hardware_info()
    assert hw["cpu_cores_logical"] >= 1
    assert "python_version" in hw
    assert get_process_rss_mb() >= 0.0
    assert GAME_SPEC["board_size"] == 11
    assert GAME_SPEC["win_length"] == 5
    assert len(OPENING_BOOK_11X11) == 20
    assert OPENING_BOOK_11X11[0] == ()


def test_play_single_game_schema():
    a = RuleBasedAgent(name="L2_RuleBased", candidate_radius=2, root_width=8, seed=42)
    b = RandomAgent(seed=99, name="L1_Random")
    rec = play_single_game(
        agent_a=a,
        agent_b=b,
        agent_a_id="L2_RuleBased",
        agent_b_id="L1_Random",
        a_side=1,
        opening_moves=(),
        opening_index=0,
        seed=42,
        mode="default_config",
        game_index=1,
    )
    required_keys = {
        "mode",
        "matchup",
        "game_index",
        "agent_a",
        "agent_b",
        "seed",
        "opening_index",
        "opening_type",
        "opening_moves",
        "a_side",
        "x_agent",
        "o_agent",
        "winner_side",
        "winner_agent",
        "result_a",
        "outcome_a",
        "num_moves",
        "duration_sec",
        "avg_ms_a",
        "p95_ms_a",
        "avg_search_a",
        "avg_ms_b",
        "p95_ms_b",
        "avg_search_b",
        "move_times_ms_a",
        "move_times_ms_b",
        "history",
    }
    assert required_keys.issubset(rec.keys())
    assert rec["a_side"] == "X"
    assert rec["opening_type"] == "standard"
    assert rec["num_moves"] == len(rec["history"]) >= 9
    assert rec["result_a"] in (0.0, 0.5, 1.0)


def test_run_pair_match_balance_and_crash_safe_resume(tmp_path: Path):
    spec_a = AgentSpec(
        agent_id="L2_Fast",
        display_name="L2 Fast",
        level=2,
        mode="default_config",
        factory=lambda seed: RuleBasedAgent(name="L2_Fast", candidate_radius=2, root_width=6, seed=seed),
        config_summary="Fast L2",
        budget_note="test",
    )
    spec_b = AgentSpec(
        agent_id="L1_Rand",
        display_name="L1 Rand",
        level=1,
        mode="default_config",
        factory=lambda seed: RandomAgent(seed=seed, name="L1_Rand"),
        config_summary="Rand",
        budget_note="test",
    )

    log_file = tmp_path / "L2_Fast_vs_L1_Rand.jsonl"

    # Odd games_per_pair must be rejected
    with pytest.raises(ValueError):
        run_pair_match(spec_a, spec_b, games_per_pair=3, log_path=log_file, verbose=False)

    # Run 2 games first
    summ2, recs2 = run_pair_match(
        spec_a,
        spec_b,
        games_per_pair=2,
        seeds=DEFAULT_SEEDS,
        log_path=log_file,
        resume=False,
        bootstrap_samples=200,
        verbose=False,
    )
    assert summ2["games"] == 2
    assert len(recs2) == 2
    assert log_file.exists()

    # Resume to 4 games -> should append exactly 2 more games
    summ4, recs4 = run_pair_match(
        spec_a,
        spec_b,
        games_per_pair=4,
        seeds=DEFAULT_SEEDS,
        log_path=log_file,
        resume=True,
        bootstrap_samples=200,
        verbose=False,
    )
    assert summ4["games"] == 4
    lines = [json.loads(ln) for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 4
    assert [ln["game_index"] for ln in lines] == [1, 2, 3, 4]
    assert sum(1 for ln in lines if ln["a_side"] == "X") == 2
    assert sum(1 for ln in lines if ln["a_side"] == "O") == 2
    # Paired openings: games (1,2) share opening_index 0; games (3,4) share opening_index 1
    assert lines[0]["opening_index"] == lines[1]["opening_index"] == 0
    assert lines[2]["opening_index"] == lines[3]["opening_index"] == 1


def test_build_agent_specs_both_modes():
    specs_tb = build_agent_specs(mode="time_budget", time_budget_sec=0.20)
    specs_def = build_agent_specs(mode="default_config")
    assert [s.agent_id for s in specs_tb] == [
        "L1_Random",
        "L2_RuleBased",
        "L3_Minimax",
        "L4_MCTS",
        "L5_CNN",
        "L6_RL",
        "L7_AlphaGo",
    ]
    assert len(specs_def) == 7
    assert "200 ms" in specs_tb[2].budget_note
    with pytest.raises(ValueError):
        build_agent_specs(mode="invalid_mode")


def test_run_round_robin_end_to_end_subset(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    tables_dir = tmp_path / "tables"

    out = run_round_robin(
        mode="default_config",
        games_per_pair=2,
        include_levels=(1, 2, 5, 6),
        output_logs_dir=logs_dir,
        output_tables_dir=tables_dir,
        resume=False,
        bootstrap_samples=200,
        verbose=False,
    )

    # 4 agents -> C(4, 2) = 6 pairs -> 6 separate per-pair JSONL files
    jsonl_files = sorted(logs_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 6
    for jf in jsonl_files:
        lines = [ln for ln in jf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2

    # Verify all 6 CSV tables + metadata JSON exist
    for fname in (
        "matchup_summary.csv",
        "score_matrix.csv",
        "ci95_matrix.csv",
        "leaderboard.csv",
        "first_player_analysis.csv",
        "seed_sensitivity.csv",
        "tournament_metadata.json",
    ):
        assert (tables_dir / fname).exists()

    # Check exact symmetry in score matrix: M[a, b] + M[b, a] == 100.0
    sm = out["score_matrix"]
    agents = list(sm.index)
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i != j:
                assert abs((sm.loc[a, b] + sm.loc[b, a]) - 100.0) < 1e-6

    # Check overall first-player row exists
    assert "ALL_GAMES_OVERALL" in set(out["first_player"]["agent_id"])
    assert out["metadata"]["total_pairs"] == 6
    assert out["metadata"]["total_games"] == 12
