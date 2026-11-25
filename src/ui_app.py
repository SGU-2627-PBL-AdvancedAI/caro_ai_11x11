"""Giao diện đồ họa Caro 11x11 (Tkinter + Matplotlib) cho cả 7 Cấp độ AI.

Tính năng chính:
- Chế độ **Người vs Máy (Human vs AI)**: chọn cấp AI (Cấp 1 -> Cấp 7), chọn phe X/O,
  đánh trực tiếp trên bàn cờ 11x11, hỗ trợ Undo, Ván mới, lưu log lịch sử `.jsonl`.
- Chế độ **Máy vs Máy (AI vs AI)**: chọn độc lập 2 cấp AI bất kỳ (Cấp 1 -> Cấp 7),
  nhập số trận muốn đấu (`1..500`), tùy chọn hoán đổi X/O cân bằng 50/50,
  xem trực tiếp hoặc chạy nhanh, tự động lưu log lịch sử từng trận (`.jsonl`),
  bấm xem lại (Replay) bất kỳ ván nào trong bảng lịch sử, và có nút
  **Xuất biểu đồ phân tích** (Matplotlib 180 DPI) cho chuỗi trận Máy vs Máy.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import statistics
import sys
import threading
import time
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np
import pandas as pd

try:
    from src.caro_env import CaroEnv
    from src.agents import (
        AlphaGoAgent,
        CNNAgent,
        MCTSAgent,
        MinimaxAgent,
        RandomAgent,
        RLAgent,
        RuleBasedAgent,
    )
    from src.tournament import (
        OPENING_BOOK_11X11,
        percentile_p95,
        wilson_ci95,
    )
except ImportError:  # pragma: no cover
    from caro_env import CaroEnv
    from agents import (
        AlphaGoAgent,
        CNNAgent,
        MCTSAgent,
        MinimaxAgent,
        RandomAgent,
        RLAgent,
        RuleBasedAgent,
    )
    from tournament import (
        OPENING_BOOK_11X11,
        percentile_p95,
        wilson_ci95,
    )


# ---------------------------------------------------------------------------
# 1. Danh mục 7 Cấp độ AI chuẩn (tên gọi thống nhất Cấp 1 -> Cấp 7)
# ---------------------------------------------------------------------------

AGENT_CATALOG: list[dict[str, Any]] = [
    {"level": 1, "id": "L1_Random", "label": "Cấp 1 — Random"},
    {"level": 2, "id": "L2_RuleBased", "label": "Cấp 2 — Rule-Based & Heuristic"},
    {"level": 3, "id": "L3_Minimax", "label": "Cấp 3 — Minimax Alpha-Beta"},
    {"level": 4, "id": "L4_MCTS", "label": "Cấp 4 — MCTS (UCT)"},
    {"level": 5, "id": "L5_CNN", "label": "Cấp 5 — CNN Policy Network"},
    {"level": 6, "id": "L6_RL", "label": "Cấp 6 — RL Dueling DQN"},
    {"level": 7, "id": "L7_AlphaGo", "label": "Cấp 7 — AlphaGo PUCT (CNN + RL)"},
]

LABEL_TO_LEVEL: dict[str, int] = {item["label"]: item["level"] for item in AGENT_CATALOG}
LEVEL_TO_ITEM: dict[int, dict[str, Any]] = {item["level"]: item for item in AGENT_CATALOG}


class _SharedModelAdapter:
    """Bọc mỏng tái sử dụng mô hình PyTorch đã nạp sẵn để đổi cấp / chơi nhiều ván tức thì."""

    def __init__(self, base_agent: Any, name: str) -> None:
        self._base = base_agent
        self.name = name
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 1

    def select_action(self, env: CaroEnv) -> int:
        action = int(self._base.select_action(env))
        self.last_move_time_ms = float(getattr(self._base, "last_move_time_ms", 0.0))
        self.last_search_count = int(getattr(self._base, "last_search_count", 1))
        return action


class UIAgentFactory:
    """Khởi tạo tác tử Cấp 1–7 cho giao diện, tự động cache trọng số mạng Cấp 5 & Cấp 6."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = (
            Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        )
        self._shared_cnn: CNNAgent | None = None
        self._shared_rl: RLAgent | None = None
        self._lock = threading.Lock()

    def _ensure_neural_models(self) -> tuple[CNNAgent, RLAgent]:
        with self._lock:
            if self._shared_cnn is None:
                cnn_path = self.project_root / "checkpoints" / "05_cnn_policy_best.pt"
                if not cnn_path.exists():
                    raise FileNotFoundError(f"Không tìm thấy checkpoint Cấp 5: {cnn_path}")
                self._shared_cnn = CNNAgent(
                    name="UI_CNN_L5",
                    model_path=cnn_path,
                    use_tactical_shortcut=True,
                    use_mix_mcts=False,
                    deterministic=True,
                    seed=42,
                )
            if self._shared_rl is None:
                rl_path = self.project_root / "checkpoints" / "06_rl_best.pt"
                if not rl_path.exists():
                    raise FileNotFoundError(f"Không tìm thấy checkpoint Cấp 6: {rl_path}")
                self._shared_rl = RLAgent(
                    checkpoint_path=rl_path,
                    name="UI_RL_L6",
                )
            return self._shared_cnn, self._shared_rl

    def create(self, level: int, seed: int = 42, fast_mode: bool = False) -> Any:
        """Tạo một thể hiện tác tử cho cấp độ `level` (1..7)."""
        if level not in LEVEL_TO_ITEM:
            raise ValueError(f"Cấp độ không hợp lệ: {level}. Chọn từ 1 đến 7.")
        aid = LEVEL_TO_ITEM[level]["id"]

        if level == 1:
            return RandomAgent(seed=seed, name=aid)
        if level == 2:
            return RuleBasedAgent(
                name=aid,
                candidate_radius=2,
                root_width=14,
                seed=seed,
                tie_break_eps=1.5,
            )
        if level == 3:
            return MinimaxAgent(
                name=aid,
                max_depth=3,
                use_alpha_beta=True,
                use_move_ordering=True,
                use_tt=True,
                max_branch_root=14 if fast_mode else 16,
                max_branch_inner=10 if fast_mode else 12,
                time_limit_sec=0.35 if fast_mode else 0.8,
                seed=seed,
                tie_break_eps=1.5,
            )
        if level == 4:
            return MCTSAgent(
                num_rollouts=200 if fast_mode else 300,
                exploration_c=1.414,
                time_limit_sec=0.45 if fast_mode else 0.8,
                rollout_policy="heuristic",
                use_heuristic_leaf=True,
                use_progressive_widening=True,
                seed=seed,
                name=aid,
            )
        if level == 5:
            cnn, _ = self._ensure_neural_models()
            return _SharedModelAdapter(cnn, name=aid)
        if level == 6:
            _, rl = self._ensure_neural_models()
            return _SharedModelAdapter(rl, name=aid)
        if level == 7:
            cnn, rl = self._ensure_neural_models()
            return AlphaGoAgent(
                name=aid,
                policy_agent=cnn,
                value_agent=rl,
                auto_load_default=False,
                num_simulations=140 if fast_mode else 200,
                time_limit_sec=0.35 if fast_mode else 0.6,
                value_blend_alpha=0.25,
                rl_prior_weight=0.35,
                c_puct=1.6,
                use_tactical_shortcut=True,
                seed=seed,
            )
        raise ValueError(f"Không hỗ trợ cấp độ: {level}")


# ---------------------------------------------------------------------------
# 2. Hỗ trợ xử lý ván đấu, lưu log JSONL & xuất biểu đồ phân tích Máy vs Máy
# ---------------------------------------------------------------------------

def find_winning_cells(env: CaroEnv) -> set[tuple[int, int]]:
    """Trả về tập tọa độ `(r, c)` thuộc chuỗi >= 5 quân thắng cuộc (nếu có)."""
    if env.winner not in (1, -1):
        return set()
    n = env.board_size
    b = env.board
    win_cells: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if b[r, c] != env.winner:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                coords = []
                rr, cc = r, c
                while 0 <= rr < n and 0 <= cc < n and b[rr, cc] == env.winner:
                    coords.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(coords) >= env.win_length:
                    win_cells.update(coords)
    return win_cells


def append_match_log(log_path: Path, record: dict[str, Any]) -> None:
    """Ghi nối tiếp 1 bản ghi ván đấu vào file `.jsonl`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_match_logs(log_path: Path) -> list[dict[str, Any]]:
    """Đọc toàn bộ bản ghi ván đấu từ file `.jsonl`."""
    if not log_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_match_record(
    *,
    mode: str,
    game_index: int,
    agent_a_id: str,
    agent_b_id: str,
    a_side: int,
    env: CaroEnv,
    times_a: Sequence[float],
    times_b: Sequence[float],
    searches_a: Sequence[float],
    searches_b: Sequence[float],
    duration_sec: float,
    seed: int = 42,
    opening_moves: Sequence[int] = (),
) -> dict[str, Any]:
    """Đóng gói bản ghi log chuẩn hóa cho 1 ván đấu đã kết thúc."""
    if env.winner == a_side:
        result_a = 1.0
        outcome_a = "win"
    elif env.winner == -a_side:
        result_a = 0.0
        outcome_a = "loss"
    else:
        result_a = 0.5
        outcome_a = "draw"

    x_agent = agent_a_id if a_side == 1 else agent_b_id
    o_agent = agent_b_id if a_side == 1 else agent_a_id
    winner_side = "X" if env.winner == 1 else ("O" if env.winner == -1 else "Draw")
    winner_agent = x_agent if env.winner == 1 else (o_agent if env.winner == -1 else "Draw")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "game_index": int(game_index),
        "agent_a": agent_a_id,
        "agent_b": agent_b_id,
        "a_side": "X" if a_side == 1 else "O",
        "x_agent": x_agent,
        "o_agent": o_agent,
        "winner_side": winner_side,
        "winner_agent": winner_agent,
        "result_a": result_a,
        "outcome_a": outcome_a,
        "seed": int(seed),
        "opening_moves": [int(m) for m in opening_moves],
        "num_moves": len(env.history),
        "duration_sec": round(float(duration_sec), 3),
        "avg_ms_a": round(statistics.mean(times_a), 3) if times_a else 0.0,
        "p95_ms_a": round(percentile_p95(times_a), 3),
        "avg_search_a": round(statistics.mean(searches_a), 2) if searches_a else 0.0,
        "avg_ms_b": round(statistics.mean(times_b), 3) if times_b else 0.0,
        "p95_ms_b": round(percentile_p95(times_b), 3),
        "avg_search_b": round(statistics.mean(searches_b), 2) if searches_b else 0.0,
        "move_times_ms_a": [round(float(x), 3) for x in times_a],
        "move_times_ms_b": [round(float(x), 3) for x in times_b],
        "history": [int(m) for m in env.history],
    }


def run_ai_vs_ai_series(
    factory: UIAgentFactory,
    level_a: int,
    level_b: int,
    num_games: int,
    *,
    swap_sides: bool = True,
    diverse_openings: bool = True,
    fast_mode: bool = True,
    base_seed: int = 42,
    log_path: Path | None = None,
    stop_event: threading.Event | None = None,
    on_move: Callable[[CaroEnv, int, int, float, float], None] | None = None,
    on_game_done: Callable[[dict[str, Any], int, int], None] | None = None,
    move_delay_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Chạy chuỗi `num_games` trận Máy vs Máy, lưu log từng trận và gọi callback cập nhật UI."""
    if num_games <= 0:
        raise ValueError("Số trận phải >= 1.")

    agent_a_id = LEVEL_TO_ITEM[level_a]["id"]
    agent_b_id = LEVEL_TO_ITEM[level_b]["id"]
    records: list[dict[str, Any]] = []

    for g in range(num_games):
        if stop_event is not None and stop_event.is_set():
            break

        pair_k = g // 2
        a_side = 1 if (not swap_sides or g % 2 == 0) else -1
        seed_val = base_seed + pair_k * 17
        seed_a = seed_val * 100 + g * 2
        seed_b = seed_val * 100 + g * 2 + 1

        # Khai cuộc đa dạng (khi chạy >= 2 trận) giúp các cặp tất định không lặp lại đúng 1 ván
        opening: tuple[int, ...] = ()
        if diverse_openings and num_games >= 2:
            opening = OPENING_BOOK_11X11[pair_k % len(OPENING_BOOK_11X11)]

        agent_a = factory.create(level_a, seed=seed_a, fast_mode=fast_mode)
        agent_b = factory.create(level_b, seed=seed_b, fast_mode=fast_mode)

        env = CaroEnv(board_size=11, win_length=5)
        for m in opening:
            env.step(int(m))
            if on_move is not None:
                on_move(env.clone(), int(m), g + 1, 0.0, 0.0)

        times_a: list[float] = []
        times_b: list[float] = []
        searches_a: list[float] = []
        searches_b: list[float] = []
        t0 = time.perf_counter()

        aborted = False
        while not env.is_terminal():
            if stop_event is not None and stop_event.is_set():
                aborted = True
                break
            is_a_turn = env.current_player == a_side
            actor = agent_a if is_a_turn else agent_b
            mt0 = time.perf_counter()
            action = int(actor.select_action(env))
            wall_ms = (time.perf_counter() - mt0) * 1000.0
            rep_ms = float(getattr(actor, "last_move_time_ms", 0.0))
            dt_ms = rep_ms if rep_ms > 0.0 else wall_ms
            sc = float(getattr(actor, "last_search_count", 0.0))

            if is_a_turn:
                times_a.append(dt_ms)
                searches_a.append(sc)
            else:
                times_b.append(dt_ms)
                searches_b.append(sc)

            env.step(action)
            if on_move is not None:
                on_move(env.clone(), action, g + 1, dt_ms, sc)
            if move_delay_sec > 0.0:
                time.sleep(move_delay_sec)

        if aborted:
            break

        rec = build_match_record(
            mode="ai_vs_ai",
            game_index=g + 1,
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            a_side=a_side,
            env=env,
            times_a=times_a,
            times_b=times_b,
            searches_a=searches_a,
            searches_b=searches_b,
            duration_sec=time.perf_counter() - t0,
            seed=seed_val,
            opening_moves=opening,
        )
        records.append(rec)
        if log_path is not None:
            append_match_log(log_path, rec)
        if on_game_done is not None:
            on_game_done(rec, g + 1, num_games)

    return records


def export_ai_vs_ai_analytics_chart(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    csv_summary_path: str | Path | None = None,
) -> Path:
    """Xuất biểu đồ phân tích 4 khung (PNG 180 DPI) cho chuỗi trận Máy vs Máy."""
    if not records:
        raise ValueError("Chưa có dữ liệu trận đấu Máy vs Máy để xuất biểu đồ.")

    # Sử dụng Figure OO API để an toàn khi gọi từ luồng nền hoặc không có GUI
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent_a = str(records[0]["agent_a"])
    agent_b = str(records[0]["agent_b"])
    n = len(records)

    scores_a = [float(r["result_a"]) for r in records]
    wins_a = sum(1 for s in scores_a if s == 1.0)
    draws = sum(1 for s in scores_a if s == 0.5)
    losses_a = sum(1 for s in scores_a if s == 0.0)
    pts_a = wins_a + 0.5 * draws
    score_a_pct = 100.0 * pts_a / n
    score_b_pct = 100.0 - score_a_pct
    w_lo, w_hi = wilson_ci95(pts_a, n)

    # Tách theo lượt đi trước (X) và đi sau (O)
    x_games = [r for r in records if r["a_side"] == "X"]
    o_games = [r for r in records if r["a_side"] == "O"]
    sx = 100.0 * sum(float(r["result_a"]) for r in x_games) / len(x_games) if x_games else 0.0
    so = 100.0 * sum(float(r["result_a"]) for r in o_games) / len(o_games) if o_games else 0.0

    # Độ trễ & số nút
    all_ta: list[float] = []
    all_tb: list[float] = []
    for r in records:
        if r.get("move_times_ms_a"):
            all_ta.extend(float(v) for v in r["move_times_ms_a"])
        elif r.get("avg_ms_a", 0) > 0:
            all_ta.append(float(r["avg_ms_a"]))
        if r.get("move_times_ms_b"):
            all_tb.extend(float(v) for v in r["move_times_ms_b"])
        elif r.get("avg_ms_b", 0) > 0:
            all_tb.append(float(r["avg_ms_b"]))

    avg_ms_a = statistics.mean(all_ta) if all_ta else 0.0
    p95_ms_a = percentile_p95(all_ta)
    avg_ms_b = statistics.mean(all_tb) if all_tb else 0.0
    p95_ms_b = percentile_p95(all_tb)
    avg_sc_a = statistics.mean(float(r.get("avg_search_a", 0.0)) for r in records)
    avg_sc_b = statistics.mean(float(r.get("avg_search_b", 0.0)) for r in records)
    num_moves = [int(r["num_moves"]) for r in records]
    cum_score_a = [100.0 * sum(scores_a[: i + 1]) / (i + 1) for i in range(n)]

    fig = Figure(figsize=(13.5, 9.2), dpi=180)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)

    # Panel 1: W/D/L & Score (%) ± Wilson CI95
    ax = axes[0, 0]
    cats = [f"{agent_a}\nthắng", "Hòa", f"{agent_b}\nthắng"]
    counts = [wins_a, draws, losses_a]
    colors = ["#2563EB", "#9CA3AF", "#DC2626"]
    bars = ax.bar(cats, counts, color=colors, edgecolor="#1F2937", width=0.52)
    for b_rect, cnt in zip(bars, counts):
        pct = 100.0 * cnt / n
        ax.text(
            b_rect.get_x() + b_rect.get_width() / 2,
            cnt + max(0.2, n * 0.03),
            f"{cnt} ({pct:.1f}%)",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_ylim(0, max(counts + [1]) * 1.24)
    ax.set_ylabel("Số trận")
    ax.set_title(
        f"(1) Kết quả Đối kháng ({n} trận)\n"
        f"Score {agent_a}: {score_a_pct:.1f}% [CI95: {w_lo:.1f}–{w_hi:.1f}%] | {agent_b}: {score_b_pct:.1f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: Diễn biến Score tích lũy & Lợi thế X/O
    ax = axes[0, 1]
    xs = list(range(1, n + 1))
    ax.plot(xs, cum_score_a, "-o", color="#2563EB", linewidth=2.0, markersize=4 if n <= 40 else 2, label=f"Score tích lũy {agent_a}")
    ax.axhline(50.0, color="#6B7280", linestyle="--", linewidth=1.1, label="Ngưỡng cân bằng 50%")
    ax.fill_between(xs, 50.0, cum_score_a, where=[v >= 50.0 for v in cum_score_a], alpha=0.14, color="#2563EB")
    ax.fill_between(xs, 50.0, cum_score_a, where=[v < 50.0 for v in cum_score_a], alpha=0.14, color="#DC2626")
    ax.set_ylim(-5, 108)
    ax.set_xlabel("Thứ tự trận đấu")
    ax.set_ylabel(f"Score tích lũy của {agent_a} (%)")
    ax.set_title(
        f"(2) Diễn biến Score & Lợi thế Đi trước\n"
        f"{agent_a} cầm X: {sx:.1f}% ({len(x_games)}g) | cầm O: {so:.1f}% ({len(o_games)}g)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    # Panel 3: Độ trễ trung bình & p95 (ms/nước)
    ax = axes[1, 0]
    x_pos = np.arange(2)
    w = 0.34
    b1 = ax.bar(x_pos - w / 2, [avg_ms_a, avg_ms_b], width=w, color="#10B981", edgecolor="#065F46", label="avg_ms")
    b2 = ax.bar(x_pos + w / 2, [p95_ms_a, p95_ms_b], width=w, color="#F59E0B", edgecolor="#92400E", label="p95_ms")
    max_ms = max(avg_ms_a, avg_ms_b, p95_ms_a, p95_ms_b, 1.0)
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + max_ms * 0.025, f"{h:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{agent_a}\n(search={avg_sc_a:.1f})", f"{agent_b}\n(search={avg_sc_b:.1f})"], fontsize=9.5)
    ax.set_ylim(0, max_ms * 1.24)
    ax.set_ylabel("Thời gian suy nghĩ (ms / nước)")
    ax.set_title("(3) So sánh Độ trễ (avg_ms & p95_ms) & Không gian tìm kiếm", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel 4: Độ dài ván đấu theo từng trận
    ax = axes[1, 1]
    bar_colors = [
        "#2563EB" if r["result_a"] == 1.0 else ("#DC2626" if r["result_a"] == 0.0 else "#9CA3AF")
        for r in records
    ]
    ax.bar(xs, num_moves, color=bar_colors, alpha=0.88, edgecolor="#374151", linewidth=0.5)
    mean_mv = statistics.mean(num_moves)
    ax.axhline(mean_mv, color="#111827", linestyle="--", linewidth=1.3, label=f"Trung bình = {mean_mv:.1f} nước")
    ax.set_xlabel("Thứ tự trận đấu (Xanh: A thắng | Đỏ: B thắng | Xám: Hòa)")
    ax.set_ylabel("Tổng số nước đi")
    ax.set_title(
        f"(4) Độ dài từng Ván đấu (min={min(num_moves)}, max={max(num_moves)}, avg={mean_mv:.1f})",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Phân tích Đối kháng Máy vs Máy — Caro 11×11: {agent_a} vs {agent_b} ({n} trận)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)

    if csv_summary_path is not None:
        csv_p = Path(csv_summary_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df_rec = pd.DataFrame(
            [
                {
                    "game_index": r["game_index"],
                    "x_agent": r["x_agent"],
                    "o_agent": r["o_agent"],
                    "winner_agent": r["winner_agent"],
                    "result_a": r["result_a"],
                    "num_moves": r["num_moves"],
                    "avg_ms_a": r["avg_ms_a"],
                    "p95_ms_a": r["p95_ms_a"],
                    "avg_ms_b": r["avg_ms_b"],
                    "p95_ms_b": r["p95_ms_b"],
                    "duration_sec": r["duration_sec"],
                }
                for r in records
            ]
        )
        df_rec.to_csv(csv_p, index=False, encoding="utf-8-sig")

    return out_path


# ---------------------------------------------------------------------------
# 3. Ứng dụng Desktop Tkinter (`CaroUIApp`)
# ---------------------------------------------------------------------------

class CaroUIApp:
    """Giao diện đồ họa Caro 11x11 hỗ trợ Người vs Máy và Máy vs Máy (7 cấp độ AI)."""

    BOARD_SIZE = 11
    CELL_PX = 48
    MARGIN_PX = 36

    def __init__(self, project_root: str | Path | None = None) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.filedialog = filedialog

        self.project_root = (
            Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
        )
        self.logs_dir = self.project_root / "results" / "logs" / "ui_matches"
        self.figures_dir = self.project_root / "results" / "figures"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.factory = UIAgentFactory(self.project_root)

        # Trạng thái ván đấu hiện tại
        self.env = CaroEnv(board_size=self.BOARD_SIZE, win_length=5)
        self.human_side: int = 1  # 1 = X, -1 = O
        self.human_ai_agent: Any = None
        self.human_game_t0: float = time.perf_counter()
        self.human_times: list[float] = []
        self.ai_times: list[float] = []
        self.ai_searches: list[float] = []
        self.human_move_start: float = time.perf_counter()
        self.human_game_counter: int = 0
        self.ai_thinking: bool = False
        self.hover_cell: tuple[int, int] | None = None

        # Trạng thái chuỗi trận Máy vs Máy
        self.ai_series_running: bool = False
        self.stop_event = threading.Event()
        self.ai_series_records: list[dict[str, Any]] = []
        self.current_log_path: Path | None = None
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        # Cửa sổ chính
        self.root = tk.Tk()
        self.root.title("Caro AI 11×11 — 7 Cấp độ (Người vs Máy & Máy vs Máy)")
        self.root.configure(bg="#F3F4F6")
        self.root.minsize(1140, 690)

        self._build_layout()
        self._on_mode_change()
        self._draw_board()
        self._poll_queue()

    # ---------------------- Xây dựng giao diện ----------------------

    def _build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#F3F4F6")
        style.configure("Card.TLabelframe", background="#FFFFFF", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background="#FFFFFF", foreground="#111827", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#F3F4F6", foreground="#111827", font=("Segoe UI", 9))
        style.configure("Card.TLabel", background="#FFFFFF", foreground="#111827", font=("Segoe UI", 9))
        style.configure("Header.TLabel", background="#1E293B", foreground="#FFFFFF", font=("Segoe UI", 13, "bold"))
        style.configure("SubHeader.TLabel", background="#1E293B", foreground="#CBD5E1", font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"))

        # Thanh tiêu đề trên cùng
        header = tk.Frame(self.root, bg="#1E293B", padx=16, pady=10)
        header.pack(fill="x")
        ttk.Label(header, text="CARO AI 11×11 — HỆ THỐNG 7 CẤP ĐỘ TÁC TỬ", style="Header.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="Free Gomoku (>=5 liên tiếp) | Người vs Máy & Máy vs Máy | Log .jsonl & Xuất biểu đồ phân tích",
            style="SubHeader.TLabel",
        ).pack(side="right")

        # Khung nội dung 2 cột
        body = tk.Frame(self.root, bg="#F3F4F6", padx=12, pady=10)
        body.pack(fill="both", expand=True)

        # CỘT TRÁI: Bàn cờ 11x11 + thanh trạng thái dưới bàn cờ
        left_col = tk.Frame(body, bg="#F3F4F6")
        left_col.pack(side="left", fill="y", padx=(0, 12))

        canvas_size = self.MARGIN_PX * 2 + self.CELL_PX * (self.BOARD_SIZE - 1)
        board_card = tk.Frame(left_col, bg="#FFFFFF", bd=1, relief="solid", padx=8, pady=8)
        board_card.pack()

        self.canvas = tk.Canvas(
            board_card,
            width=canvas_size,
            height=canvas_size,
            bg="#F4DFB6",
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

        # Thanh trạng thái dưới bàn cờ
        status_frame = tk.Frame(left_col, bg="#FFFFFF", bd=1, relief="solid", padx=10, pady=8)
        status_frame.pack(fill="x", pady=(8, 0))
        self.status_var = tk.StringVar(value="Sẵn sàng. Chọn chế độ và bắt đầu ván đấu!")
        self.telemetry_var = tk.StringVar(value="Lượt: X | Số nước: 0 | Độ trễ gần nhất: -- ms")
        tk.Label(status_frame, textvariable=self.status_var, bg="#FFFFFF", fg="#111827", font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        tk.Label(status_frame, textvariable=self.telemetry_var, bg="#FFFFFF", fg="#4B5563", font=("Segoe UI", 9), anchor="w").pack(fill="x", pady=(2, 0))

        # CỘT PHẢI: Điều khiển chế độ + Thống kê + Bảng lịch sử trận đấu
        right_col = tk.Frame(body, bg="#F3F4F6")
        right_col.pack(side="left", fill="both", expand=True)

        # 1. Chọn chế độ chơi
        mode_box = ttk.LabelFrame(right_col, text="1. Chế độ thi đấu", style="Card.TLabelframe", padding=8)
        mode_box.pack(fill="x")
        self.mode_box = mode_box
        self.mode_var = tk.StringVar(value="human_vs_ai")
        rb_row = tk.Frame(mode_box, bg="#FFFFFF")
        rb_row.pack(fill="x")
        tk.Radiobutton(
            rb_row,
            text="Người vs Máy (Human vs AI)",
            variable=self.mode_var,
            value="human_vs_ai",
            command=self._on_mode_change,
            bg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 18))
        tk.Radiobutton(
            rb_row,
            text="Máy vs Máy (AI vs AI — nhiều trận)",
            variable=self.mode_var,
            value="ai_vs_ai",
            command=self._on_mode_change,
            bg="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        agent_labels = [item["label"] for item in AGENT_CATALOG]

        # 2A. Khung cấu hình Người vs Máy
        self.human_box = ttk.LabelFrame(right_col, text="2. Thiết lập Người vs Máy", style="Card.TLabelframe", padding=8)
        h_grid = tk.Frame(self.human_box, bg="#FFFFFF")
        h_grid.pack(fill="x")

        tk.Label(h_grid, text="Đối thủ AI:", bg="#FFFFFF", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=3)
        self.human_ai_var = tk.StringVar(value=agent_labels[6])  # Mặc định Cấp 7
        ttk.Combobox(h_grid, textvariable=self.human_ai_var, values=agent_labels, state="readonly", width=34).grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)

        tk.Label(h_grid, text="Phe của bạn:", bg="#FFFFFF", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=3)
        side_row = tk.Frame(h_grid, bg="#FFFFFF")
        side_row.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=3)
        self.human_side_var = tk.StringVar(value="X")
        tk.Radiobutton(side_row, text="Đi trước (X — Quân Đen)", variable=self.human_side_var, value="X", bg="#FFFFFF").pack(side="left", padx=(0, 12))
        tk.Radiobutton(side_row, text="Đi sau (O — Quân Trắng)", variable=self.human_side_var, value="O", bg="#FFFFFF").pack(side="left")
        h_grid.columnconfigure(1, weight=1)

        h_btns = tk.Frame(self.human_box, bg="#FFFFFF")
        h_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(h_btns, text="Ván mới", style="Primary.TButton", command=self._start_human_game).pack(side="left", padx=(0, 6))
        ttk.Button(h_btns, text="Đi lại (Undo)", command=self._undo_human_move).pack(side="left", padx=(0, 6))
        ttk.Button(h_btns, text="Gợi ý (Cấp 7)", command=self._hint_move).pack(side="left")

        # 2B. Khung cấu hình Máy vs Máy
        self.ai_box = ttk.LabelFrame(right_col, text="2. Thiết lập Máy vs Máy (AI vs AI)", style="Card.TLabelframe", padding=8)
        a_grid = tk.Frame(self.ai_box, bg="#FFFFFF")
        a_grid.pack(fill="x")

        tk.Label(a_grid, text="AI A (X mặc định):", bg="#FFFFFF", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", pady=3)
        self.ai_a_var = tk.StringVar(value=agent_labels[6])  # Cấp 7
        ttk.Combobox(a_grid, textvariable=self.ai_a_var, values=agent_labels, state="readonly", width=34).grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=3)

        tk.Label(a_grid, text="AI B (O mặc định):", bg="#FFFFFF", font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", pady=3)
        self.ai_b_var = tk.StringVar(value=agent_labels[2])  # Cấp 3
        ttk.Combobox(a_grid, textvariable=self.ai_b_var, values=agent_labels, state="readonly", width=34).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=3)

        tk.Label(a_grid, text="Số trận muốn đấu:", bg="#FFFFFF", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w", pady=4)
        self.num_games_var = tk.StringVar(value="10")
        tk.Spinbox(a_grid, from_=1, to=500, textvariable=self.num_games_var, width=8, font=("Segoe UI", 9, "bold")).grid(row=2, column=1, sticky="w", padx=(8, 12), pady=4)

        self.swap_sides_var = tk.BooleanVar(value=True)
        tk.Checkbutton(a_grid, text="Đổi lượt X/O (50/50)", variable=self.swap_sides_var, bg="#FFFFFF").grid(row=2, column=2, sticky="w", pady=4)

        self.fast_mode_var = tk.BooleanVar(value=True)
        tk.Checkbutton(a_grid, text="Chạy nhanh", variable=self.fast_mode_var, bg="#FFFFFF").grid(row=2, column=3, sticky="w", padx=(8, 0), pady=4)
        a_grid.columnconfigure(3, weight=1)

        a_btns = tk.Frame(self.ai_box, bg="#FFFFFF")
        a_btns.pack(fill="x", pady=(8, 0))
        self.btn_start_ai = ttk.Button(a_btns, text="Bắt đầu đấu", style="Primary.TButton", command=self._start_ai_vs_ai)
        self.btn_start_ai.pack(side="left", padx=(0, 6))
        self.btn_stop_ai = ttk.Button(a_btns, text="Dừng", command=self._stop_ai_vs_ai, state="disabled")
        self.btn_stop_ai.pack(side="left", padx=(0, 6))
        self.btn_export_chart = ttk.Button(
            a_btns,
            text="Xuất biểu đồ phân tích",
            style="Primary.TButton",
            command=self._export_chart_clicked,
            state="disabled",
        )
        self.btn_export_chart.pack(side="left", padx=(0, 6))
        ttk.Button(a_btns, text="Mở log (.jsonl)", command=self._load_existing_log_clicked).pack(side="left")

        # 3. Thống kê tổng hợp & đường dẫn log
        stats_box = ttk.LabelFrame(right_col, text="3. Thống kê & File Log", style="Card.TLabelframe", padding=8)
        stats_box.pack(fill="x", pady=(8, 0))
        self.series_summary_var = tk.StringVar(value="Chưa có trận đấu hoàn tất.")
        self.log_path_var = tk.StringVar(value=f"Thư mục log: {self.logs_dir}")
        tk.Label(stats_box, textvariable=self.series_summary_var, bg="#FFFFFF", fg="#1D4ED8", font=("Segoe UI", 9, "bold"), anchor="w", justify="left").pack(fill="x")
        tk.Label(stats_box, textvariable=self.log_path_var, bg="#FFFFFF", fg="#6B7280", font=("Segoe UI", 8), anchor="w").pack(fill="x", pady=(3, 0))

        # 4. Bảng Lịch sử các trận đấu (bấm để xem lại Replay trên bàn cờ)
        hist_box = ttk.LabelFrame(right_col, text="4. Lịch sử các trận đấu (Bấm vào dòng để xem lại bàn cờ)", style="Card.TLabelframe", padding=6)
        hist_box.pack(fill="both", expand=True, pady=(8, 0))

        cols = ("idx", "x_agent", "o_agent", "winner", "moves", "ms")
        self.tree = ttk.Treeview(hist_box, columns=cols, show="headings", height=11)
        self.tree.heading("idx", text="#")
        self.tree.heading("x_agent", text="Bên X (Đi trước)")
        self.tree.heading("o_agent", text="Bên O (Đi sau)")
        self.tree.heading("winner", text="Người thắng")
        self.tree.heading("moves", text="Số nước")
        self.tree.heading("ms", text="Avg ms (A/B)")

        self.tree.column("idx", width=36, anchor="center")
        self.tree.column("x_agent", width=115, anchor="w")
        self.tree.column("o_agent", width=115, anchor="w")
        self.tree.column("winner", width=105, anchor="center")
        self.tree.column("moves", width=62, anchor="center")
        self.tree.column("ms", width=100, anchor="center")

        sb = ttk.Scrollbar(hist_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_history_select)

    # ---------------------- Vẽ bàn cờ Caro 11x11 ----------------------

    def _cell_to_xy(self, r: int, c: int) -> tuple[int, int]:
        return (self.MARGIN_PX + c * self.CELL_PX, self.MARGIN_PX + r * self.CELL_PX)

    def _xy_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        c = round((x - self.MARGIN_PX) / self.CELL_PX)
        r = round((y - self.MARGIN_PX) / self.CELL_PX)
        if 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
            cx, cy = self._cell_to_xy(r, c)
            if abs(x - cx) <= self.CELL_PX * 0.45 and abs(y - cy) <= self.CELL_PX * 0.45:
                return (r, c)
        return None

    def _draw_board(self, hint_action: int | None = None) -> None:
        c = self.canvas
        c.delete("all")
        m = self.MARGIN_PX
        s = self.CELL_PX
        n = self.BOARD_SIZE
        end = m + (n - 1) * s

        # Lưới + tọa độ A-K / 1-11
        col_letters = "ABCDEFGHIJK"
        for i in range(n):
            pos = m + i * s
            c.create_line(m, pos, end, pos, fill="#7C5C36", width=1.1)
            c.create_line(pos, m, pos, end, fill="#7C5C36", width=1.1)
            c.create_text(pos, 14, text=col_letters[i], fill="#5B4224", font=("Segoe UI", 9, "bold"))
            c.create_text(14, pos, text=str(i + 1), fill="#5B4224", font=("Segoe UI", 9, "bold"))

        # Điểm sao (hoshi)
        for hr, hc in ((5, 5), (2, 2), (2, 8), (8, 2), (8, 8)):
            hx, hy = self._cell_to_xy(hr, hc)
            c.create_oval(hx - 3.5, hy - 3.5, hx + 3.5, hy + 3.5, fill="#5B4224", outline="")

        # Hover preview khi đến lượt người chơi
        if (
            self.hover_cell is not None
            and self.mode_var.get() == "human_vs_ai"
            and not self.ai_thinking
            and not self.env.is_terminal()
            and self.env.current_player == self.human_side
        ):
            hr, hc = self.hover_cell
            if self.env.board[hr, hc] == 0:
                hx, hy = self._cell_to_xy(hr, hc)
                rad = s * 0.40
                c.create_oval(
                    hx - rad, hy - rad, hx + rad, hy + rad,
                    outline="#2563EB", width=2, dash=(3, 2),
                )

        # Ô gợi ý (Hint)
        if hint_action is not None:
            hr, hc = divmod(int(hint_action), n)
            hx, hy = self._cell_to_xy(hr, hc)
            rad = s * 0.42
            c.create_oval(hx - rad, hy - rad, hx + rad, hy + rad, outline="#10B981", width=3)
            c.create_text(hx, hy, text="★", fill="#059669", font=("Segoe UI", 14, "bold"))

        # Quân cờ kèm số thứ tự nước đi + đánh dấu nước cuối + chuỗi thắng 5 quân
        win_cells = find_winning_cells(self.env)
        last_move = self.env.history[-1] if self.env.history else None
        rad = s * 0.41

        for move_num, action in enumerate(self.env.history, start=1):
            r, col = divmod(int(action), n)
            x, y = self._cell_to_xy(r, col)
            is_x = (move_num % 2 == 1)
            fill_col = "#111827" if is_x else "#FFFFFF"
            if (r, col) in win_cells:
                outline_col = "#EF4444"
                width_px = 3.2
            elif action == last_move:
                outline_col = "#2563EB"
                width_px = 2.8
            else:
                outline_col = "#111827" if is_x else "#4B5563"
                width_px = 1.4

            c.create_oval(x - rad, y - rad, x + rad, y + rad, fill=fill_col, outline=outline_col, width=width_px)
            txt_col = "#FFFFFF" if is_x else "#111827"
            c.create_text(x, y, text=str(move_num), fill=txt_col, font=("Segoe UI", 8, "bold"))

    # ---------------------- Chuyển chế độ ----------------------

    def _on_mode_change(self) -> None:
        if self.ai_series_running:
            self._stop_ai_vs_ai()

        if self.mode_var.get() == "human_vs_ai":
            self.ai_box.pack_forget()
            self.human_box.pack(fill="x", pady=(8, 0), after=self.mode_box)
            self._start_human_game()
        else:
            self.human_box.pack_forget()
            self.ai_box.pack(fill="x", pady=(8, 0), after=self.mode_box)
            self.status_var.set("Chế độ Máy vs Máy: chọn 2 cấp AI, nhập số trận và bấm 'Bắt đầu đấu'.")

    # ---------------------- Chế độ Người vs Máy ----------------------

    def _start_human_game(self) -> None:
        if self.ai_series_running:
            return
        level = LABEL_TO_LEVEL[self.human_ai_var.get()]
        self.human_side = 1 if self.human_side_var.get() == "X" else -1
        self.env = CaroEnv(board_size=self.BOARD_SIZE, win_length=5)
        self.human_times = []
        self.ai_times = []
        self.ai_searches = []
        self.ai_thinking = False
        self.human_game_t0 = time.perf_counter()
        self.human_move_start = time.perf_counter()

        try:
            self.human_ai_agent = self.factory.create(level=level, seed=42 + self.human_game_counter, fast_mode=False)
        except Exception as exc:
            self.messagebox.showerror("Lỗi nạp tác tử", str(exc))
            return

        self._draw_board()
        if self.human_side == 1:
            self.status_var.set(f"Người (X) vs {self.human_ai_var.get()} (O) — Đến lượt bạn đi trước!")
            self.telemetry_var.set("Lượt: X (Người) | Số nước: 0")
        else:
            self.status_var.set(f"{self.human_ai_var.get()} (X) đang suy nghĩ nước khai cuộc...")
            self._trigger_ai_move_for_human_game()

    def _on_canvas_motion(self, event: Any) -> None:
        cell = self._xy_to_cell(event.x, event.y)
        if cell != self.hover_cell:
            self.hover_cell = cell
            if self.mode_var.get() == "human_vs_ai" and not self.ai_thinking and not self.env.is_terminal():
                self._draw_board()

    def _on_canvas_leave(self, _event: Any) -> None:
        if self.hover_cell is not None:
            self.hover_cell = None
            if self.mode_var.get() == "human_vs_ai":
                self._draw_board()

    def _on_canvas_click(self, event: Any) -> None:
        if self.mode_var.get() != "human_vs_ai" or self.ai_thinking or self.env.is_terminal():
            return
        if self.env.current_player != self.human_side:
            return
        cell = self._xy_to_cell(event.x, event.y)
        if cell is None:
            return
        r, c = cell
        if self.env.board[r, c] != 0:
            return

        dt_ms = (time.perf_counter() - self.human_move_start) * 1000.0
        self.human_times.append(dt_ms)
        action = r * self.BOARD_SIZE + c
        self.env.step(action)
        self._draw_board()

        if self.env.is_terminal():
            self._finish_human_game()
        else:
            self._trigger_ai_move_for_human_game()

    def _trigger_ai_move_for_human_game(self) -> None:
        if self.human_ai_agent is None:
            return
        self.ai_thinking = True
        self.status_var.set(f"AI ({self.human_ai_agent.name}) đang suy nghĩ...")
        env_snap = self.env.clone()
        agent = self.human_ai_agent

        def _worker() -> None:
            t0 = time.perf_counter()
            act = int(agent.select_action(env_snap))
            wall_ms = (time.perf_counter() - t0) * 1000.0
            rep_ms = float(getattr(agent, "last_move_time_ms", 0.0))
            dt_ms = rep_ms if rep_ms > 0.0 else wall_ms
            sc = float(getattr(agent, "last_search_count", 0.0))
            self.ui_queue.put(("human_ai_move", (act, dt_ms, sc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _undo_human_move(self) -> None:
        if self.mode_var.get() != "human_vs_ai" or self.ai_thinking:
            return
        if len(self.env.history) < 2:
            return
        # Lùi 2 nửa nước (nước của AI và nước trước đó của người)
        keep = self.env.history[:-2]
        self.env = CaroEnv(board_size=self.BOARD_SIZE, win_length=5)
        for m in keep:
            self.env.step(int(m))
        self.human_move_start = time.perf_counter()
        self._draw_board()
        self.status_var.set("Đã lùi 1 lượt đi. Mời bạn chọn nước mới.")

    def _hint_move(self) -> None:
        if self.mode_var.get() != "human_vs_ai" or self.ai_thinking or self.env.is_terminal():
            return
        self.ai_thinking = True
        self.status_var.set("Đang tính gợi ý nước đi tốt nhất bằng Cấp 7...")
        env_snap = self.env.clone()

        def _worker() -> None:
            try:
                helper = self.factory.create(level=7, seed=99, fast_mode=True)
                act = int(helper.select_action(env_snap))
                self.ui_queue.put(("hint_done", act))
            except Exception as exc:
                self.ui_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_human_game(self) -> None:
        self.human_game_counter += 1
        ai_id = self.human_ai_agent.name if self.human_ai_agent else "AI"
        rec = build_match_record(
            mode="human_vs_ai",
            game_index=self.human_game_counter,
            agent_a_id="Human",
            agent_b_id=ai_id,
            a_side=self.human_side,
            env=self.env,
            times_a=self.human_times,
            times_b=self.ai_times,
            searches_a=[1.0] * len(self.human_times),
            searches_b=self.ai_searches,
            duration_sec=time.perf_counter() - self.human_game_t0,
        )
        log_file = self.logs_dir / "human_vs_ai_history.jsonl"
        append_match_log(log_file, rec)
        self.current_log_path = log_file
        self.log_path_var.set(f"Log đã lưu: {log_file}")

        if rec["winner_agent"] == "Human":
            msg = f"CHÚC MỪNG! Bạn đã thắng {ai_id} sau {rec['num_moves']} nước!"
        elif rec["winner_agent"] == "Draw":
            msg = f"HÒA sau {rec['num_moves']} nước đấu với {ai_id}!"
        else:
            msg = f"{ai_id} đã chiến thắng sau {rec['num_moves']} nước!"
        self.status_var.set(msg)
        self._append_history_row(rec)
        self.ai_series_records.append(rec)

    # ---------------------- Chế độ Máy vs Máy ----------------------

    def _start_ai_vs_ai(self) -> None:
        if self.ai_series_running:
            return
        try:
            num_games = int(self.num_games_var.get())
            if num_games < 1 or num_games > 500:
                raise ValueError
        except ValueError:
            self.messagebox.showwarning("Số trận không hợp lệ", "Vui lòng nhập số nguyên từ 1 đến 500.")
            return

        level_a = LABEL_TO_LEVEL[self.ai_a_var.get()]
        level_b = LABEL_TO_LEVEL[self.ai_b_var.get()]
        id_a = LEVEL_TO_ITEM[level_a]["id"]
        id_b = LEVEL_TO_ITEM[level_b]["id"]
        swap_sides = bool(self.swap_sides_var.get())
        fast_mode = bool(self.fast_mode_var.get())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"ai_vs_ai_{id_a}_vs_{id_b}_{ts}.jsonl"
        self.current_log_path = log_file
        self.ai_series_records = []
        for row_id in self.tree.get_children():
            self.tree.delete(row_id)

        self.stop_event.clear()
        self.ai_series_running = True
        self.btn_start_ai.configure(state="disabled")
        self.btn_stop_ai.configure(state="normal")
        self.btn_export_chart.configure(state="disabled")
        self.log_path_var.set(f"Đang ghi log: {log_file}")
        self.status_var.set(f"Đang chạy chuỗi {num_games} trận: {id_a} vs {id_b}...")

        def _worker() -> None:
            try:
                def _on_move(env_snap: CaroEnv, action: int, game_idx: int, dt_ms: float, sc: float) -> None:
                    self.ui_queue.put(("ai_move", (env_snap, action, game_idx, num_games, dt_ms, sc)))

                def _on_done(rec: dict[str, Any], done_n: int, total_n: int) -> None:
                    self.ui_queue.put(("ai_game_done", (rec, done_n, total_n)))

                run_ai_vs_ai_series(
                    factory=self.factory,
                    level_a=level_a,
                    level_b=level_b,
                    num_games=num_games,
                    swap_sides=swap_sides,
                    diverse_openings=True,
                    fast_mode=fast_mode,
                    base_seed=42,
                    log_path=log_file,
                    stop_event=self.stop_event,
                    on_move=None if (fast_mode and num_games >= 10) else _on_move,
                    on_game_done=_on_done,
                    move_delay_sec=0.0 if fast_mode else 0.12,
                )
                self.ui_queue.put(("ai_series_finished", None))
            except Exception as exc:
                self.ui_queue.put(("error", str(exc)))
                self.ui_queue.put(("ai_series_finished", None))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_ai_vs_ai(self) -> None:
        self.stop_event.set()
        self.status_var.set("Đang dừng chuỗi trận an toàn sau nước hiện tại...")

    def _update_series_summary_label(self) -> None:
        ai_recs = [r for r in self.ai_series_records if r.get("mode") == "ai_vs_ai"]
        if not ai_recs:
            self.series_summary_var.set("Chưa có trận Máy vs Máy hoàn tất.")
            return
        a_id = ai_recs[0]["agent_a"]
        b_id = ai_recs[0]["agent_b"]
        n = len(ai_recs)
        wa = sum(1 for r in ai_recs if r["result_a"] == 1.0)
        dr = sum(1 for r in ai_recs if r["result_a"] == 0.5)
        la = sum(1 for r in ai_recs if r["result_a"] == 0.0)
        pts = wa + 0.5 * dr
        score_pct = 100.0 * pts / n
        lo, hi = wilson_ci95(pts, n)
        avg_a = statistics.mean(float(r["avg_ms_a"]) for r in ai_recs)
        avg_b = statistics.mean(float(r["avg_ms_b"]) for r in ai_recs)
        self.series_summary_var.set(
            f"{a_id} vs {b_id} ({n} trận): W{wa} - D{dr} - L{la} | "
            f"Score {a_id} = {score_pct:.1f}% [CI95: {lo:.1f}–{hi:.1f}%]\n"
            f"Độ trễ TB: {a_id} = {avg_a:.1f} ms | {b_id} = {avg_b:.1f} ms"
        )

    def _export_chart_clicked(self) -> None:
        ai_recs = [r for r in self.ai_series_records if r.get("mode") == "ai_vs_ai"]
        if not ai_recs:
            self.messagebox.showinfo("Chưa có dữ liệu", "Hãy chạy ít nhất 1 trận Máy vs Máy (hoặc mở file log `.jsonl`) trước khi xuất biểu đồ.")
            return
        a_id = ai_recs[0]["agent_a"]
        b_id = ai_recs[0]["agent_b"]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_png = self.figures_dir / f"ui_ai_vs_ai_{a_id}_vs_{b_id}_{ts}.png"
        out_csv = self.project_root / "results" / "tables" / f"ui_ai_vs_ai_{a_id}_vs_{b_id}_{ts}.csv"

        try:
            saved_path = export_ai_vs_ai_analytics_chart(ai_recs, out_png, csv_summary_path=out_csv)
            self.status_var.set(f"Đã xuất biểu đồ phân tích: {saved_path.name}")
            self._show_chart_preview(saved_path)
            self.messagebox.showinfo(
                "Xuất biểu đồ thành công",
                f"Đã lưu biểu đồ phân tích Máy vs Máy tại:\n{saved_path}\n\nBảng CSV chi tiết:\n{out_csv}",
            )
        except Exception as exc:
            self.messagebox.showerror("Lỗi xuất biểu đồ", str(exc))

    def _show_chart_preview(self, png_path: Path) -> None:
        try:
            win = self.tk.Toplevel(self.root)
            win.title(f"Biểu đồ Phân tích — {png_path.name}")
            img = self.tk.PhotoImage(file=str(png_path))
            # Thu nhỏ nếu quá lớn so với màn hình
            if img.width() > 1100:
                img = img.subsample(2, 2)
            lbl = self.tk.Label(win, image=img, bg="#FFFFFF")
            lbl.image = img  # giữ tham chiếu tránh GC
            lbl.pack(padx=8, pady=8)
        except Exception:
            pass

    def _load_existing_log_clicked(self) -> None:
        chosen = self.filedialog.askopenfilename(
            title="Chọn file log lịch sử trận đấu (.jsonl)",
            initialdir=str(self.logs_dir),
            filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")],
        )
        if not chosen:
            return
        p = Path(chosen)
        recs = load_match_logs(p)
        if not recs:
            self.messagebox.showwarning("File rỗng", "File log không có bản ghi hợp lệ.")
            return
        self.ai_series_records = recs
        self.current_log_path = p
        for row_id in self.tree.get_children():
            self.tree.delete(row_id)
        for rec in recs:
            self._append_history_row(rec)
        self._update_series_summary_label()
        self.log_path_var.set(f"Đã nạp log: {p} ({len(recs)} trận)")
        if any(r.get("mode") == "ai_vs_ai" for r in recs):
            self.btn_export_chart.configure(state="normal")
        self._replay_record(recs[-1])

    # ---------------------- Bảng lịch sử & Replay ----------------------

    def _append_history_row(self, rec: dict[str, Any]) -> None:
        idx = len(self.tree.get_children())
        self.tree.insert(
            "",
            "end",
            iid=str(idx),
            values=(
                rec.get("game_index", idx + 1),
                rec.get("x_agent", ""),
                rec.get("o_agent", ""),
                rec.get("winner_agent", ""),
                rec.get("num_moves", 0),
                f"{rec.get('avg_ms_a', 0.0):.1f} / {rec.get('avg_ms_b', 0.0):.1f}",
            ),
        )
        self.tree.see(str(idx))

    def _on_history_select(self, _event: Any) -> None:
        if self.ai_series_running:
            return
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.ai_series_records):
            self._replay_record(self.ai_series_records[idx])

    def _replay_record(self, rec: dict[str, Any]) -> None:
        env = CaroEnv(board_size=self.BOARD_SIZE, win_length=5)
        for mv in rec.get("history", []):
            env.step(int(mv))
        self.env = env
        self._draw_board()
        self.status_var.set(
            f"[Replay Trận #{rec.get('game_index')}] X: {rec.get('x_agent')} vs O: {rec.get('o_agent')} "
            f"| Thắng: {rec.get('winner_agent')} ({rec.get('num_moves')} nước)"
        )
        self.telemetry_var.set(
            f"Avg ms: {rec.get('agent_a')}={rec.get('avg_ms_a', 0):.1f} ms | "
            f"{rec.get('agent_b')}={rec.get('avg_ms_b', 0):.1f} ms | "
            f"Search: {rec.get('avg_search_a', 0):.1f} / {rec.get('avg_search_b', 0):.1f}"
        )

    # ---------------------- Vòng xử lý sự kiện đa luồng ----------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "human_ai_move":
                    act, dt_ms, sc = payload
                    self.ai_thinking = False
                    if self.mode_var.get() == "human_vs_ai" and not self.env.is_terminal():
                        self.ai_times.append(dt_ms)
                        self.ai_searches.append(sc)
                        self.env.step(act)
                        self.human_move_start = time.perf_counter()
                        self._draw_board()
                        self.telemetry_var.set(
                            f"Số nước: {len(self.env.history)} | AI vừa đi mất {dt_ms:.1f} ms "
                            f"(search={int(sc)})"
                        )
                        if self.env.is_terminal():
                            self._finish_human_game()
                        else:
                            side_txt = "X" if self.human_side == 1 else "O"
                            self.status_var.set(f"Đến lượt bạn ({side_txt})!")
                elif kind == "hint_done":
                    self.ai_thinking = False
                    act = int(payload)
                    r, col = divmod(act, self.BOARD_SIZE)
                    self._draw_board(hint_action=act)
                    self.status_var.set(f"Gợi ý từ Cấp 7: ô {'ABCDEFGHIJK'[col]}{r + 1} (hàng {r+1}, cột {col+1})")
                elif kind == "ai_move":
                    env_snap, _act, g_idx, total_n, dt_ms, sc = payload
                    self.env = env_snap
                    self._draw_board()
                    self.telemetry_var.set(
                        f"Trận {g_idx}/{total_n} | Nước #{len(env_snap.history)} | "
                        f"Thời gian nước vừa rồi: {dt_ms:.1f} ms (search={int(sc)})"
                    )
                elif kind == "ai_game_done":
                    rec, done_n, total_n = payload
                    self.ai_series_records.append(rec)
                    self._append_history_row(rec)
                    self._replay_record(rec)
                    self._update_series_summary_label()
                    self.status_var.set(f"Đã xong {done_n}/{total_n} trận — Thắng trận #{done_n}: {rec['winner_agent']}")
                    self.btn_export_chart.configure(state="normal")
                elif kind == "ai_series_finished":
                    self.ai_series_running = False
                    self.btn_start_ai.configure(state="normal")
                    self.btn_stop_ai.configure(state="disabled")
                    if self.ai_series_records:
                        self.btn_export_chart.configure(state="normal")
                        self._update_series_summary_label()
                        self.status_var.set(
                            f"Hoàn tất {len(self.ai_series_records)} trận Máy vs Máy! "
                            f"Bấm 'Xuất biểu đồ phân tích' để xem biểu đồ."
                        )
                elif kind == "error":
                    self.ai_thinking = False
                    self.messagebox.showerror("Lỗi thực thi AI", str(payload))
        except queue.Empty:
            pass
        self.root.after(40, self._poll_queue)

    def run(self) -> None:
        self.root.mainloop()


def launch_ui(project_root: str | Path | None = None) -> None:
    """Khởi chạy cửa sổ ứng dụng Caro 11x11."""
    app = CaroUIApp(project_root=project_root)
    app.run()


if __name__ == "__main__":
    launch_ui()
