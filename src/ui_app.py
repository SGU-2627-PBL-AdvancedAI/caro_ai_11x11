"""Giao diện Web Streamlit cho Caro AI 11×11 (7 Cấp độ AI — Người vs Máy, Máy vs Máy & Replay).

Tính năng chính:
- **Người vs Máy (Human vs AI):** Chọn cấp AI (Cấp 1 → Cấp 7), chọn phe X/O,
  đánh trực tiếp trên bàn cờ gỗ 11×11 tương tác, hỗ trợ Đi lại (Undo), Ván mới,
  Gợi ý nước đi từ Cấp 7 (AlphaGo PUCT), bảng giải thích Top nước ứng viên (Cấp 5–7),
  và tự động lưu lịch sử `.jsonl`.
- **Máy vs Máy (AI vs AI Arena):** Chọn độc lập 2 cấp AI bất kỳ (Cấp 1 → Cấp 7),
  chạy 1–200 trận với tùy chọn hoán đổi X/O 50/50, khai cuộc đa dạng, xem trực tiếp
  từng nước trên bàn cờ SVG sắc nét, lưu log `.jsonl`, hiển thị biểu đồ phân tích
  4 khung ngay trên web và tải về PNG (180 DPI) / CSV / JSONL.
- **Lịch sử & Replay:** Mở bất kỳ file log `.jsonl` (từ UI hoặc Giải đấu chính thức),
  tua lại từng nước đi, bật/tắt số thứ tự nước trên quân cờ và xem thống kê chi tiết.

Chạy trực tiếp:
    streamlit run src/ui_app.py
hoặc:
    python main.py
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

COL_LABELS = "ABCDEFGHIJK"
HOSHI_POINTS = {(2, 2), (2, 8), (5, 5), (8, 2), (8, 8)}

AGENT_CATALOG: list[dict[str, Any]] = [
    {
        "level": 1,
        "id": "L1_Random",
        "label": "Cấp 1 — Random",
        "short": "Random",
        "family": "Baseline ngẫu nhiên",
        "speed": "< 1 ms/nước",
        "desc": "Chọn nước đi hợp lệ ngẫu nhiên đồng đều để làm mốc kiểm chứng.",
    },
    {
        "level": 2,
        "id": "L2_RuleBased",
        "label": "Cấp 2 — Rule-Based & Heuristic",
        "short": "Rule-Based",
        "family": "Luật 7 tầng + 3-ply",
        "speed": "~5–15 ms/nước",
        "desc": "Phát hiện thắng 1 bước, chặn 4, đòn kép và đánh giá mối đe dọa cục bộ.",
    },
    {
        "level": 3,
        "id": "L3_Minimax",
        "label": "Cấp 3 — Minimax Alpha-Beta",
        "short": "Minimax AB",
        "family": "Tìm kiếm cây + Zobrist TT",
        "speed": "~80–400 ms/nước",
        "desc": "Tìm kiếm Minimax cắt tỉa Alpha-Beta kết hợp bảng chuyển vị và Killer Moves.",
    },
    {
        "level": 4,
        "id": "L4_MCTS",
        "label": "Cấp 4 — MCTS (UCT)",
        "short": "MCTS UCT",
        "family": "Monte Carlo Tree Search",
        "speed": "~250–600 ms/nước",
        "desc": "MCTS nâng cấp với mở rộng dần (Progressive Widening) và đánh giá lá heuristic.",
    },
    {
        "level": 5,
        "id": "L5_CNN",
        "label": "Cấp 5 — CNN Policy Network",
        "short": "CNN ResNet",
        "family": "Học sâu PyTorch ResNet",
        "speed": "~3–12 ms/nước",
        "desc": "Mạng Policy ResNet 4 kênh dự đoán phân bố xác suất nước đi từ dữ liệu chuyên gia.",
    },
    {
        "level": 6,
        "id": "L6_RL",
        "label": "Cấp 6 — RL Dueling DQN",
        "short": "Dueling DQN",
        "family": "Học tăng cường (DQfD + Self-Play)",
        "speed": "~2–8 ms/nước",
        "desc": "Dueling Double-DQN Residual + GroupNorm học giá trị trạng thái V(s) và Q(s,a).",
    },
    {
        "level": 7,
        "id": "L7_AlphaGo",
        "label": "Cấp 7 — AlphaGo PUCT (CNN + RL)",
        "short": "AlphaGo PUCT",
        "family": "Lai Policy + Value + PUCT",
        "speed": "~250–550 ms/nước",
        "desc": "Kết hợp Policy Cấp 5, Value/Q Cấp 6 và tìm kiếm PUCT-MCTS + Zobrist Eval Cache.",
    },
]

LABEL_TO_LEVEL: dict[str, int] = {item["label"]: item["level"] for item in AGENT_CATALOG}
LEVEL_TO_ITEM: dict[int, dict[str, Any]] = {item["level"]: item for item in AGENT_CATALOG}


def action_to_coord(action: int, board_size: int = 11) -> str:
    """Chuyển chỉ số ô `0..120` sang tọa độ chuẩn dạng `F6`."""
    r, c = divmod(int(action), board_size)
    return f"{COL_LABELS[c]}{r + 1}"


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
# 2. Hỗ trợ xử lý ván đấu, lưu log JSONL, Explainability & Biểu đồ phân tích
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


def extract_ai_explainability(
    agent: Any,
    env_before_move: CaroEnv,
    level: int,
    top_k: int = 5,
) -> dict[str, Any] | None:
    """Trích xuất Top nước ứng viên để giải thích quyết định của AI (Cấp 5, 6, 7)."""
    try:
        base = getattr(agent, "_base", agent)
        if level == 7 and hasattr(base, "explain_decision"):
            exp = base.explain_decision(top_k=top_k)
            top_moves = exp.get("top_moves") or []
            rows = []
            for item in top_moves[:top_k]:
                act = int(item.get("action", -1))
                rows.append(
                    {
                        "Ô": action_to_coord(act) if act >= 0 else "-",
                        "Thăm": int(item.get("visits", 0)),
                        "Prior": f"{float(item.get('prior', 0.0)) * 100.0:.1f}%",
                        "Q": f"{float(item.get('q_value', 0.0)):+.2f}",
                    }
                )
            return {
                "title": "Giải thích AlphaGo PUCT (Cấp 7)",
                "summary": (
                    f"Mô phỏng: {exp.get('simulations', 0)} | "
                    f"Cache hits: {exp.get('cache_hits', 0)} | "
                    f"Root V(s): {exp.get('root_value_rl', 'N/A')}"
                ),
                "rows": rows,
            }
        if level == 5 and hasattr(base, "predict_probs"):
            probs = np.asarray(base.predict_probs(env_before_move, restrict_to_candidates=False), dtype=float)
            legal = env_before_move.legal_actions()
            ranked = sorted(legal, key=lambda a: float(probs[a]), reverse=True)[:top_k]
            rows = [
                {"Ô": action_to_coord(a), "Xác suất P(s,a)": f"{probs[a] * 100:.2f}%"}
                for a in ranked
            ]
            return {
                "title": "Phân bố xác suất Policy CNN (Cấp 5)",
                "summary": f"Top-{len(rows)} ô được mạng ResNet ưu tiên nhất trên trạng thái vừa đánh.",
                "rows": rows,
            }
        if level == 6 and hasattr(base, "get_masked_q_values"):
            q_vals = np.asarray(base.get_masked_q_values(env_before_move), dtype=float)
            legal = env_before_move.legal_actions()
            ranked = sorted(legal, key=lambda a: float(q_vals[a]), reverse=True)[:top_k]
            rows = [
                {"Ô": action_to_coord(a), "Giá trị Q(s,a)": round(float(q_vals[a]), 4)}
                for a in ranked
            ]
            return {
                "title": "Đánh giá Q-value Dueling DQN (Cấp 6)",
                "summary": f"Top-{len(rows)} ô có kỳ vọng Q(s,a) cao nhất trên trạng thái vừa đánh.",
                "rows": rows,
            }
    except Exception:
        return None
    return None


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


def _build_analytics_figure(records: Sequence[dict[str, Any]]) -> Any:
    """Dựng đối tượng Matplotlib Figure 4 khung phân tích chuỗi trận Máy vs Máy."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

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

    x_games = [r for r in records if r.get("a_side") == "X"]
    o_games = [r for r in records if r.get("a_side") == "O"]
    sx = 100.0 * sum(float(r["result_a"]) for r in x_games) / len(x_games) if x_games else 0.0
    so = 100.0 * sum(float(r["result_a"]) for r in o_games) / len(o_games) if o_games else 0.0

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

    fig = Figure(figsize=(13.5, 9.2), dpi=180, facecolor="#FAF8F4")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for row_ax in axes:
        for ax_i in row_ax:
            ax_i.set_facecolor("#FFFFFF")

    # Panel 1: W/D/L & Score (%) ± Wilson CI95
    ax = axes[0, 0]
    cats = [f"{agent_a}\nthắng", "Hòa", f"{agent_b}\nthắng"]
    counts = [wins_a, draws, losses_a]
    colors = ["#1D4ED8", "#9CA3AF", "#B91C1C"]
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
    ax.plot(
        xs,
        cum_score_a,
        "-o",
        color="#1D4ED8",
        linewidth=2.0,
        markersize=4 if n <= 40 else 2,
        label=f"Score tích lũy {agent_a}",
    )
    ax.axhline(50.0, color="#6B7280", linestyle="--", linewidth=1.1, label="Ngưỡng cân bằng 50%")
    ax.fill_between(xs, 50.0, cum_score_a, where=[v >= 50.0 for v in cum_score_a], alpha=0.14, color="#1D4ED8")
    ax.fill_between(xs, 50.0, cum_score_a, where=[v < 50.0 for v in cum_score_a], alpha=0.14, color="#B91C1C")
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
    b1 = ax.bar(x_pos - w / 2, [avg_ms_a, avg_ms_b], width=w, color="#059669", edgecolor="#065F46", label="avg_ms")
    b2 = ax.bar(x_pos + w / 2, [p95_ms_a, p95_ms_b], width=w, color="#D97706", edgecolor="#92400E", label="p95_ms")
    max_ms = max(avg_ms_a, avg_ms_b, p95_ms_a, p95_ms_b, 1.0)
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + max_ms * 0.025,
            f"{h:.1f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
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
        "#1D4ED8" if r["result_a"] == 1.0 else ("#B91C1C" if r["result_a"] == 0.0 else "#9CA3AF")
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
    return fig


def render_analytics_chart_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    """Kết xuất biểu đồ 4 khung thành dữ liệu PNG (180 DPI) trong bộ nhớ."""
    if not records:
        raise ValueError("Chưa có dữ liệu trận đấu Máy vs Máy để xuất biểu đồ.")
    fig = _build_analytics_figure(records)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    return buf.getvalue()


def export_ai_vs_ai_analytics_chart(
    records: Sequence[dict[str, Any]],
    output_path: str | Path,
    csv_summary_path: str | Path | None = None,
) -> Path:
    """Xuất biểu đồ phân tích 4 khung (PNG 180 DPI) cho chuỗi trận Máy vs Máy."""
    if not records:
        raise ValueError("Chưa có dữ liệu trận đấu Máy vs Máy để xuất biểu đồ.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = _build_analytics_figure(records)
    fig.savefig(out_path, dpi=180)

    if csv_summary_path is not None:
        csv_p = Path(csv_summary_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df_rec = build_records_summary_df(records)
        df_rec.to_csv(csv_p, index=False, encoding="utf-8-sig")

    return out_path


def build_records_summary_df(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Chuyển danh sách bản ghi log trận đấu thành DataFrame gọn để hiển thị/tải CSV."""
    return pd.DataFrame(
        [
            {
                "game_index": r.get("game_index", idx + 1),
                "x_agent": r.get("x_agent", ""),
                "o_agent": r.get("o_agent", ""),
                "winner_agent": r.get("winner_agent", ""),
                "result_a": r.get("result_a", 0.5),
                "num_moves": r.get("num_moves", len(r.get("history", []))),
                "avg_ms_a": r.get("avg_ms_a", 0.0),
                "p95_ms_a": r.get("p95_ms_a", 0.0),
                "avg_ms_b": r.get("avg_ms_b", 0.0),
                "p95_ms_b": r.get("p95_ms_b", 0.0),
                "duration_sec": r.get("duration_sec", 0.0),
            }
            for idx, r in enumerate(records)
        ]
    )


# ---------------------------------------------------------------------------
# 3. Trình vẽ bàn cờ SVG sắc nét (cho Live AI vs AI, Replay & Xem trước)
# ---------------------------------------------------------------------------

def render_board_svg(
    env: CaroEnv,
    *,
    show_move_numbers: bool = True,
    hint_action: int | None = None,
    highlight_win: bool = True,
    size_px: int = 540,
) -> str:
    """Tạo chuỗi SVG độ phân giải sắc nét mô phỏng bàn cờ gỗ Hinoki 11×11."""
    n = env.board_size
    pad = 38
    cell = 42
    board_px = n * cell
    total = board_px + 2 * pad

    win_cells = find_winning_cells(env) if highlight_win else set()
    last_action = env.history[-1] if env.history else None
    move_num_map: dict[tuple[int, int], int] = {}
    for idx, act in enumerate(env.history, start=1):
        rr, cc = divmod(int(act), n)
        move_num_map[(rr, cc)] = idx

    parts: list[str] = [
        f'<svg viewBox="0 0 {total} {total}" width="100%" '
        f'style="max-width:{size_px}px; display:block; margin:0 auto; '
        f'border-radius:16px; box-shadow:0 14px 32px rgba(68,42,16,0.18), '
        f'0 2px 6px rgba(68,42,16,0.12);" xmlns="http://www.w3.org/2000/svg">',
        "<defs>",
        '<linearGradient id="hinokiWood" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#E7C48B"/>',
        '<stop offset="50%" stop-color="#DBAE6C"/>',
        '<stop offset="100%" stop-color="#CCA05A"/>',
        "</linearGradient>",
        '<radialGradient id="stoneX" cx="35%" cy="30%" r="70%">',
        '<stop offset="0%" stop-color="#DC2626"/>',
        '<stop offset="65%" stop-color="#991B1B"/>',
        '<stop offset="100%" stop-color="#6B1010"/>',
        "</radialGradient>",
        '<radialGradient id="stoneO" cx="35%" cy="30%" r="70%">',
        '<stop offset="0%" stop-color="#3B82F6"/>',
        '<stop offset="65%" stop-color="#1D4ED8"/>',
        '<stop offset="100%" stop-color="#1E3A8A"/>',
        "</radialGradient>",
        "</defs>",
        f'<rect x="0" y="0" width="{total}" height="{total}" rx="16" fill="url(#hinokiWood)"/>',
        f'<rect x="{pad}" y="{pad}" width="{board_px}" height="{board_px}" '
        f'fill="rgba(255,250,240,0.22)" stroke="#5C3A18" stroke-width="2"/>',
    ]

    # Đường kẻ lưới ô vuông 11x11
    for i in range(1, n):
        x = pad + i * cell
        y = pad + i * cell
        parts.append(
            f'<line x1="{x}" y1="{pad}" x2="{x}" y2="{pad + board_px}" '
            f'stroke="#6E4720" stroke-opacity="0.48" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{pad}" y1="{y}" x2="{pad + board_px}" y2="{y}" '
            f'stroke="#6E4720" stroke-opacity="0.48" stroke-width="1"/>'
        )

    # Nhãn tọa độ A-K & 1-11
    for i in range(n):
        cx = pad + i * cell + cell / 2
        cy = pad + i * cell + cell / 2
        lbl_col = COL_LABELS[i]
        lbl_row = str(i + 1)
        parts.append(
            f'<text x="{cx}" y="{pad - 13}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="12" font-weight="700" fill="#4A2E12">{lbl_col}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{pad + board_px + 22}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="12" font-weight="700" fill="#4A2E12">{lbl_col}</text>'
        )
        parts.append(
            f'<text x="{pad - 14}" y="{cy + 4}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="12" font-weight="700" fill="#4A2E12">{lbl_row}</text>'
        )
        parts.append(
            f'<text x="{pad + board_px + 16}" y="{cy + 4}" text-anchor="middle" '
            f'font-family="JetBrains Mono, monospace" font-size="12" font-weight="700" fill="#4A2E12">{lbl_row}</text>'
        )

    # Các điểm sao (hoshi) trên ô trống
    for hr, hc in HOSHI_POINTS:
        if env.board[hr, hc] == 0:
            hx = pad + hc * cell + cell / 2
            hy = pad + hr * cell + cell / 2
            parts.append(f'<circle cx="{hx}" cy="{hy}" r="3.2" fill="#5C3A18" fill-opacity="0.58"/>')

    # Tô nền ô thắng & nước vừa đánh
    for r in range(n):
        for c in range(n):
            x0 = pad + c * cell
            y0 = pad + r * cell
            act = r * n + c
            if (r, c) in win_cells:
                parts.append(
                    f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{cell - 2}" height="{cell - 2}" '
                    f'fill="#FDE047" fill-opacity="0.62" stroke="#CA8A04" stroke-width="2"/>'
                )
            elif last_action is not None and act == int(last_action):
                parts.append(
                    f'<rect x="{x0 + 1.5}" y="{y0 + 1.5}" width="{cell - 3}" height="{cell - 3}" '
                    f'fill="#FEF3C7" fill-opacity="0.55" stroke="#D97706" stroke-width="2.2"/>'
                )
            elif hint_action is not None and act == int(hint_action):
                parts.append(
                    f'<rect x="{x0 + 1.5}" y="{y0 + 1.5}" width="{cell - 3}" height="{cell - 3}" '
                    f'fill="#34D399" fill-opacity="0.35" stroke="#059669" stroke-width="2.4" stroke-dasharray="4 2"/>'
                )

    # Vẽ quân cờ X / O
    for r in range(n):
        for c in range(n):
            val = int(env.board[r, c])
            if val == 0:
                continue
            cx = pad + c * cell + cell / 2
            cy = pad + r * cell + cell / 2
            mv_no = move_num_map.get((r, c), 0)

            if val == 1:
                parts.append(
                    f'<circle cx="{cx}" cy="{cy + 1.2}" r="15.8" fill="rgba(0,0,0,0.22)"/>'
                    f'<circle cx="{cx}" cy="{cy}" r="15.5" fill="url(#stoneX)" stroke="#450A0A" stroke-width="1"/>'
                )
                if show_move_numbers and mv_no > 0:
                    parts.append(
                        f'<text x="{cx}" y="{cy + 4.5}" text-anchor="middle" '
                        f'font-family="JetBrains Mono, monospace" font-size="11.5" font-weight="700" fill="#FFF7ED">{mv_no}</text>'
                    )
                else:
                    parts.append(
                        f'<line x1="{cx - 6.5}" y1="{cy - 6.5}" x2="{cx + 6.5}" y2="{cy + 6.5}" '
                        f'stroke="#FFF7ED" stroke-width="2.8" stroke-linecap="round"/>'
                        f'<line x1="{cx + 6.5}" y1="{cy - 6.5}" x2="{cx - 6.5}" y2="{cy + 6.5}" '
                        f'stroke="#FFF7ED" stroke-width="2.8" stroke-linecap="round"/>'
                    )
            else:
                parts.append(
                    f'<circle cx="{cx}" cy="{cy + 1.2}" r="15.8" fill="rgba(0,0,0,0.22)"/>'
                    f'<circle cx="{cx}" cy="{cy}" r="15.5" fill="url(#stoneO)" stroke="#172554" stroke-width="1"/>'
                )
                if show_move_numbers and mv_no > 0:
                    parts.append(
                        f'<text x="{cx}" y="{cy + 4.5}" text-anchor="middle" '
                        f'font-family="JetBrains Mono, monospace" font-size="11.5" font-weight="700" fill="#EFF6FF">{mv_no}</text>'
                    )
                else:
                    parts.append(
                        f'<circle cx="{cx}" cy="{cy}" r="7.5" fill="none" stroke="#EFF6FF" stroke-width="2.7"/>'
                    )

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 4. Ứng dụng Web Streamlit — Caro AI 11×11 Studio
# ---------------------------------------------------------------------------

def _inject_studio_css() -> None:
    """Chèn hệ thống CSS tinh chỉnh giao diện Hinoki & Obsidian Studio cho Streamlit."""
    import streamlit as st

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Gọn khoảng trắng đầu trang nhưng không chạm thanh header của Streamlit */
        .block-container {
            padding-top: 2.85rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.6rem !important;
            padding-right: 1.6rem !important;
            max-width: 1420px !important;
        }

        /* Header Hero Banner */
        .caro-hero {
            background: linear-gradient(125deg, #292524 0%, #44403C 52%, #78350F 100%);
            color: #FAF8F4;
            padding: 0.95rem 1.35rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 26px rgba(41, 37, 36, 0.16);
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.65rem;
            border: 1px solid rgba(251, 191, 36, 0.24);
        }
        .caro-hero-title {
            font-size: 1.38rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            color: #FFFBEB;
            line-height: 1.2;
        }
        .caro-hero-sub {
            font-size: 0.84rem;
            color: #E7E5E4;
            margin: 0.22rem 0 0 0;
            line-height: 1.35;
        }
        .caro-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(255, 251, 235, 0.12);
            border: 1px solid rgba(253, 224, 71, 0.35);
            color: #FEF3C7;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }

        /* Thẻ trạng thái lượt đi */
        .status-banner {
            border-radius: 12px;
            padding: 0.62rem 0.9rem;
            margin-bottom: 0.7rem;
            font-weight: 600;
            font-size: 0.88rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.35rem;
            border: 1px solid transparent;
        }
        .status-turn-x {
            background: #FEF2F2;
            border-color: #FECACA;
            color: #991B1B;
        }
        .status-turn-o {
            background: #EFF6FF;
            border-color: #BFDBFE;
            color: #1E40AF;
        }
        .status-win {
            background: linear-gradient(90deg, #FEF9C3 0%, #FDE68A 100%);
            border-color: #F59E0B;
            color: #78350F;
        }
        .status-draw {
            background: #F5F5F4;
            border-color: #D6D3D1;
            color: #44403C;
        }

        /* Thẻ hồ sơ AI & Telemetry */
        .studio-card {
            background: #FFFFFF;
            border: 1px solid #E5DFD3;
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 2px 10px rgba(28, 25, 23, 0.04);
            margin-bottom: 0.7rem;
            overflow-wrap: anywhere;
            word-break: break-word;
            font-size: 0.86rem;
        }
        .studio-card-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 700;
            color: #78716C;
            margin-bottom: 0.35rem;
        }
        .ai-badge {
            display: inline-block;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            background: #FEF3C7;
            color: #92400E;
            margin-right: 0.3rem;
            margin-bottom: 0.25rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin-bottom: 0.7rem;
            width: 100%;
            box-sizing: border-box;
        }
        .metric-box {
            background: #FFFFFF;
            border: 1px solid #E5DFD3;
            border-radius: 11px;
            padding: 0.5rem 0.35rem;
            text-align: center;
            min-width: 0;
        }
        .metric-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: #78716C;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.96rem;
            font-weight: 700;
            color: #1C1917;
            margin-top: 0.12rem;
            white-space: nowrap;
        }

        /* Bảng Explainability gọn đẹp không thanh cuộn ngang */
        .explain-table-wrap {
            width: 100%;
            overflow-x: auto;
        }
        .explain-table {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            font-size: 0.76rem;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.35rem;
        }
        .explain-table th {
            text-align: left;
            padding: 0.28rem 0.25rem;
            background: #F6F3EC;
            color: #57534E;
            font-weight: 700;
            border-bottom: 1px solid #E5DFD3;
            white-space: nowrap;
        }
        .explain-table td {
            padding: 0.26rem 0.25rem;
            border-bottom: 1px solid #F0ECE3;
            color: #1C1917;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .explain-table tr:first-child td {
            font-weight: 700;
            background: #FFFBEB;
            color: #92400E;
        }

        /* Khung bàn cờ gỗ Hinoki 11x11 tương tác — ô vuông khít 1:1 */
        .st-key-caro_board_frame {
            background: linear-gradient(145deg, #E7C48B 0%, #D9AC69 52%, #CA9C57 100%);
            border-radius: 16px;
            padding: 12px 14px 14px 8px;
            box-shadow: 0 14px 30px rgba(68, 42, 16, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.45);
            border: 2px solid #9A6B32;
            max-width: 560px;
            width: 100%;
            margin: 0 auto 0.7rem auto;
            box-sizing: border-box;
            gap: 0 !important;
            row-gap: 0 !important;
        }
        .st-key-caro_board_frame > div,
        .st-key-caro_board_frame [data-testid="stVerticalBlock"],
        .st-key-caro_board_frame [data-testid="stVerticalBlockBorderWrapper"] > div {
            gap: 0 !important;
            row-gap: 0 !important;
        }
        .st-key-caro_board_frame [data-testid="stElementContainer"],
        .st-key-caro_board_frame .stElementContainer,
        .st-key-caro_board_frame .element-container {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
        }
        .st-key-caro_board_frame [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            column-gap: 0 !important;
            flex-wrap: nowrap !important;
            align-items: stretch !important;
        }
        .st-key-caro_board_frame [data-testid="stColumn"] {
            min-width: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .st-key-caro_board_frame [data-testid="stMarkdownContainer"] {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-caro_board_frame [data-testid="stMarkdownContainer"] p {
            margin: 0 !important;
            line-height: 1 !important;
        }
        .st-key-caro_board_frame .coord-col {
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            color: #4A2E12;
            padding-bottom: 4px;
            user-select: none;
            white-space: nowrap;
        }
        .st-key-caro_board_frame .coord-row {
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            color: #4A2E12;
            line-height: 1;
            user-select: none;
            white-space: nowrap;
            padding-right: 5px;
        }
        .st-key-caro_board_frame .stButton {
            margin: 0 !important;
            padding: 0 !important;
            width: 100% !important;
            display: block !important;
        }
        .st-key-caro_board_frame .stButton > button,
        .st-key-caro_board_frame .stButton > button:disabled {
            width: 100% !important;
            min-width: 0 !important;
            height: auto !important;
            min-height: 0 !important;
            aspect-ratio: 1 / 1 !important;
            padding: 0 !important;
            margin: 0 !important;
            border-radius: 0 !important;
            border: 0.5px solid rgba(92, 58, 22, 0.45) !important;
            background: rgba(255, 250, 240, 0.16) !important;
            color: rgba(92, 58, 22, 0.6) !important;
            -webkit-text-fill-color: rgba(92, 58, 22, 0.6) !important;
            font-size: 0.95rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
            box-shadow: none !important;
            opacity: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: background 0.1s ease, box-shadow 0.1s ease;
        }
        .st-key-caro_board_frame .stButton > button:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.38) !important;
            box-shadow: inset 0 0 0 2px rgba(180, 83, 9, 0.75) !important;
            color: #78350F !important;
            -webkit-text-fill-color: #78350F !important;
        }
        .st-key-caro_board_frame .stButton > button:disabled {
            cursor: default !important;
        }
        .st-key-caro_board_frame .stButton > button p {
            font-size: 0.92rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Chữ ký nước đi dạng chips */
        .move-history-box {
            max-height: 175px;
            overflow-y: auto;
            background: #FAF8F4;
            border: 1px solid #E5DFD3;
            border-radius: 10px;
            padding: 0.5rem 0.65rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            line-height: 1.65;
            color: #292524;
        }
        .mv-x { color: #B91C1C; font-weight: 700; }
        .mv-o { color: #1D4ED8; font-weight: 700; }

        /* Tinh gọn sidebar để mọi nút điều khiển nằm gọn trong tầm mắt */
        section[data-testid="stSidebar"] {
            border-right: 1px solid #DDD6C7;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.6rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_dynamic_cell_css(
    env: CaroEnv,
    *,
    win_cells: set[tuple[int, int]],
    last_action: int | None,
    hint_action: int | None,
) -> None:
    """Tô màu từng ô quân X/O dạng đá 3D nổi bật, ô thắng 5 quân, nước vừa đánh và gợi ý."""
    import streamlit as st

    n = env.board_size
    rules: list[str] = []

    for r in range(n):
        for c in range(n):
            val = int(env.board[r, c])
            act = r * n + c
            btn_sel = f".st-key-caro_board_frame .st-key-cell_{r}_{c} button"
            btn_dis = f".st-key-caro_board_frame .st-key-cell_{r}_{c} button:disabled"
            p_sel = f".st-key-caro_board_frame .st-key-cell_{r}_{c} button p"
            is_win = (r, c) in win_cells
            is_last = last_action is not None and act == int(last_action)
            is_hint = hint_action is not None and act == int(hint_action) and val == 0

            if val == 1:
                outer = (
                    "rgba(253,224,71,0.88)"
                    if is_win
                    else ("rgba(254,240,138,0.85)" if is_last else "rgba(255,250,240,0.16)")
                )
                bg = (
                    f"radial-gradient(circle at 36% 32%, #EF4444 0%, #B91C1C 52%, #7F1D1D 66%, "
                    f"{outer} 69%, {outer} 100%)"
                )
                ring = (
                    "inset 0 0 0 2.5px #CA8A04"
                    if is_win
                    else ("inset 0 0 0 2.5px #D97706" if is_last else "none")
                )
                rules.append(
                    f"{btn_sel}, {btn_dis} {{ background: {bg} !important; color: #FFFFFF !important; "
                    f"-webkit-text-fill-color: #FFFFFF !important; box-shadow: {ring} !important; opacity: 1 !important; }}"
                    f"{p_sel} {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }}"
                )
            elif val == -1:
                outer = (
                    "rgba(253,224,71,0.88)"
                    if is_win
                    else ("rgba(254,240,138,0.85)" if is_last else "rgba(255,250,240,0.16)")
                )
                bg = (
                    f"radial-gradient(circle at 36% 32%, #60A5FA 0%, #2563EB 52%, #1E3A8A 66%, "
                    f"{outer} 69%, {outer} 100%)"
                )
                ring = (
                    "inset 0 0 0 2.5px #CA8A04"
                    if is_win
                    else ("inset 0 0 0 2.5px #D97706" if is_last else "none")
                )
                rules.append(
                    f"{btn_sel}, {btn_dis} {{ background: {bg} !important; color: #FFFFFF !important; "
                    f"-webkit-text-fill-color: #FFFFFF !important; box-shadow: {ring} !important; opacity: 1 !important; }}"
                    f"{p_sel} {{ color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }}"
                )
            elif is_hint:
                rules.append(
                    f"{btn_sel}, {btn_dis} {{ background: rgba(16, 185, 129, 0.34) !important; "
                    f"color: #065F46 !important; -webkit-text-fill-color: #065F46 !important; "
                    f"box-shadow: inset 0 0 0 2.5px #059669 !important; }}"
                    f"{p_sel} {{ color: #065F46 !important; -webkit-text-fill-color: #065F46 !important; }}"
                )

    if rules:
        st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)


def _get_cached_factory(project_root: Path) -> UIAgentFactory:
    """Cache `UIAgentFactory` xuyên suốt các lần rerun của Streamlit (nạp PyTorch 1 lần duy nhất)."""
    import streamlit as st

    @st.cache_resource(show_spinner="Đang nạp mô hình AI Cấp 5 & Cấp 6...")
    def _build_factory(root_str: str) -> UIAgentFactory:
        return UIAgentFactory(project_root=Path(root_str))

    return _build_factory(str(project_root.resolve()))


def _init_session_state() -> None:
    import streamlit as st

    defaults: dict[str, Any] = {
        "hv_env": CaroEnv(board_size=11, win_length=5),
        "hv_ai_level": 7,
        "hv_human_side": 1,  # 1 = X, -1 = O
        "hv_fast_mode": False,
        "hv_game_count": 1,
        "hv_start_ts": time.perf_counter(),
        "hv_human_move_start": time.perf_counter(),
        "hv_human_times": [],
        "hv_ai_times": [],
        "hv_ai_searches": [],
        "hv_last_ai_ms": 0.0,
        "hv_last_ai_search": 0,
        "hv_hint_action": None,
        "hv_explain": None,
        "hv_saved_terminal": False,
        "hv_need_ai_move": False,
        "hv_scoreboard": {"win": 0, "draw": 0, "loss": 0},
        # AI vs AI state
        "ava_records": [],
        "ava_log_path": None,
        "ava_chart_bytes": None,
        "ava_last_pair": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_human_game(ai_level: int, human_side: int, fast_mode: bool) -> None:
    import streamlit as st

    st.session_state["hv_env"] = CaroEnv(board_size=11, win_length=5)
    st.session_state["hv_ai_level"] = ai_level
    st.session_state["hv_human_side"] = human_side
    st.session_state["hv_fast_mode"] = fast_mode
    st.session_state["hv_start_ts"] = time.perf_counter()
    st.session_state["hv_human_move_start"] = time.perf_counter()
    st.session_state["hv_human_times"] = []
    st.session_state["hv_ai_times"] = []
    st.session_state["hv_ai_searches"] = []
    st.session_state["hv_last_ai_ms"] = 0.0
    st.session_state["hv_last_ai_search"] = 0
    st.session_state["hv_hint_action"] = None
    st.session_state["hv_explain"] = None
    st.session_state["hv_saved_terminal"] = False
    st.session_state["hv_need_ai_move"] = human_side == -1


def _finalize_human_game(logs_dir: Path) -> None:
    import streamlit as st

    if st.session_state["hv_saved_terminal"]:
        return
    env: CaroEnv = st.session_state["hv_env"]
    if not env.is_terminal():
        return

    human_side = int(st.session_state["hv_human_side"])
    ai_level = int(st.session_state["hv_ai_level"])
    ai_id = LEVEL_TO_ITEM[ai_level]["id"]
    rec = build_match_record(
        mode="human_vs_ai",
        game_index=int(st.session_state["hv_game_count"]),
        agent_a_id="Human",
        agent_b_id=ai_id,
        a_side=human_side,
        env=env,
        times_a=st.session_state["hv_human_times"],
        times_b=st.session_state["hv_ai_times"],
        searches_a=[],
        searches_b=st.session_state["hv_ai_searches"],
        duration_sec=time.perf_counter() - float(st.session_state["hv_start_ts"]),
        seed=42 + int(st.session_state["hv_game_count"]),
    )
    append_match_log(logs_dir / "human_vs_ai_history.jsonl", rec)
    st.session_state["hv_game_count"] += 1
    st.session_state["hv_saved_terminal"] = True

    sb = st.session_state["hv_scoreboard"]
    if rec["outcome_a"] == "win":
        sb["win"] += 1
    elif rec["outcome_a"] == "loss":
        sb["loss"] += 1
    else:
        sb["draw"] += 1


def _perform_ai_turn(factory: UIAgentFactory, logs_dir: Path, show_explain: bool) -> None:
    import streamlit as st

    env: CaroEnv = st.session_state["hv_env"]
    if env.is_terminal():
        st.session_state["hv_need_ai_move"] = False
        return

    ai_level = int(st.session_state["hv_ai_level"])
    fast_mode = bool(st.session_state["hv_fast_mode"])
    seed = 42 + int(st.session_state["hv_game_count"]) * 31 + len(env.history)

    env_before = env.clone()
    agent = factory.create(ai_level, seed=seed, fast_mode=fast_mode)
    t0 = time.perf_counter()
    action = int(agent.select_action(env))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    rep_ms = float(getattr(agent, "last_move_time_ms", 0.0))
    dt_ms = rep_ms if rep_ms > 0.0 else wall_ms
    sc = int(getattr(agent, "last_search_count", 1))

    env.step(action)
    st.session_state["hv_ai_times"].append(dt_ms)
    st.session_state["hv_ai_searches"].append(float(sc))
    st.session_state["hv_last_ai_ms"] = dt_ms
    st.session_state["hv_last_ai_search"] = sc
    st.session_state["hv_human_move_start"] = time.perf_counter()
    st.session_state["hv_hint_action"] = None
    st.session_state["hv_need_ai_move"] = False

    if ai_level in (5, 6, 7):
        st.session_state["hv_explain"] = extract_ai_explainability(agent, env_before, ai_level, top_k=5)
    else:
        st.session_state["hv_explain"] = None

    if env.is_terminal():
        _finalize_human_game(logs_dir)


def _handle_cell_click(action: int, logs_dir: Path) -> None:
    import streamlit as st

    env: CaroEnv = st.session_state["hv_env"]
    human_side = int(st.session_state["hv_human_side"])
    if env.is_terminal() or env.current_player != human_side:
        return
    r, c = divmod(action, env.board_size)
    if env.board[r, c] != 0:
        return

    dt_ms = (time.perf_counter() - float(st.session_state["hv_human_move_start"])) * 1000.0
    st.session_state["hv_human_times"].append(dt_ms)
    st.session_state["hv_hint_action"] = None
    env.step(action)

    if env.is_terminal():
        _finalize_human_game(logs_dir)
    else:
        st.session_state["hv_need_ai_move"] = True


def _undo_human_turn() -> None:
    import streamlit as st

    env: CaroEnv = st.session_state["hv_env"]
    human_side = int(st.session_state["hv_human_side"])
    if not env.history:
        return

    # Hoàn tác sao cho quay lại đúng lượt người chơi
    if env.current_player == human_side and len(env.history) >= 2:
        steps_back = 2
    else:
        steps_back = 1
    new_hist = env.history[:-steps_back]

    fresh = CaroEnv(board_size=11, win_length=5)
    for m in new_hist:
        fresh.step(int(m))
    st.session_state["hv_env"] = fresh
    st.session_state["hv_hint_action"] = None
    st.session_state["hv_explain"] = None
    st.session_state["hv_saved_terminal"] = False
    st.session_state["hv_need_ai_move"] = (not fresh.is_terminal()) and (fresh.current_player != human_side)
    st.session_state["hv_human_move_start"] = time.perf_counter()


def _render_human_vs_ai_tab(factory: UIAgentFactory, logs_dir: Path) -> None:
    import streamlit as st

    # Sidebar thiết lập cho Người vs Máy
    with st.sidebar:
        st.markdown("### ⚙️ Cấu hình Người vs Máy")
        labels = [item["label"] for item in AGENT_CATALOG]
        cur_lv = int(st.session_state["hv_ai_level"])
        selected_label = st.selectbox(
            "Chọn cấp độ AI đối thủ",
            options=labels,
            index=cur_lv - 1,
            key="sb_hv_level",
        )
        new_lv = LABEL_TO_LEVEL[selected_label]

        side_choice = st.radio(
            "Chọn quân cờ của bạn",
            options=["X (Đi trước)", "O (Đi sau)"],
            index=0 if st.session_state["hv_human_side"] == 1 else 1,
            horizontal=True,
            key="sb_hv_side",
        )
        new_side = 1 if side_choice.startswith("X") else -1

        fast_mode = st.toggle(
            "Chế độ phản hồi nhanh",
            value=bool(st.session_state["hv_fast_mode"]),
            help="Giảm nhẹ ngân sách tìm kiếm cho Minimax / MCTS / AlphaGo để phản hồi nhanh hơn.",
            key="sb_hv_fast",
        )
        show_explain = st.toggle(
            "Hiển thị bảng phân bố xác suất / giải thích AI (Cấp 5–7)",
            value=True,
            help="Bật/tắt tức thì bảng phân bố xác suất Policy CNN (Cấp 5), Q-value DQN (Cấp 6) hoặc AlphaGo PUCT (Cấp 7).",
            key="sb_hv_explain",
        )
        show_move_log = st.toggle(
            "Hiển thị biên bản nước đi",
            value=False,
            help="Mặc định ẩn. Bật để xem danh sách tọa độ từng nước đi và tải file biên bản JSON.",
            key="sb_hv_move_log",
        )

        # Tự động tạo ván mới nếu đổi cấp AI hoặc đổi phe
        if new_lv != st.session_state["hv_ai_level"] or new_side != st.session_state["hv_human_side"]:
            _reset_human_game(new_lv, new_side, fast_mode)
            st.rerun()
        st.session_state["hv_fast_mode"] = fast_mode

        st.divider()
        c_act1, c_act2 = st.columns(2)
        with c_act1:
            if st.button("🔄 Ván mới", use_container_width=True, type="primary"):
                _reset_human_game(new_lv, new_side, fast_mode)
                st.rerun()
        with c_act2:
            if st.button("↩️ Đi lại", use_container_width=True):
                _undo_human_turn()
                st.rerun()

        if st.button("💡 Gợi ý nước đi (Cấp 7 AlphaGo)", use_container_width=True):
            env_cur: CaroEnv = st.session_state["hv_env"]
            if not env_cur.is_terminal() and env_cur.current_player == new_side:
                with st.spinner("AlphaGo Cấp 7 đang phân tích bàn cờ..."):
                    advisor = factory.create(7, seed=777, fast_mode=True)
                    hint_act = int(advisor.select_action(env_cur))
                    st.session_state["hv_hint_action"] = hint_act
                    st.session_state["hv_explain"] = extract_ai_explainability(advisor, env_cur, 7, top_k=5)
                st.rerun()

        # Hồ sơ AI đang đối đầu
        ai_meta = LEVEL_TO_ITEM[new_lv]
        st.markdown(
            f"""
            <div class="studio-card" style="margin-top:0.85rem;">
                <div class="studio-card-title">Hồ sơ đối thủ</div>
                <div style="font-weight:700; font-size:0.96rem; margin-bottom:0.35rem;">{ai_meta['label']}</div>
                <div style="margin-bottom:0.4rem;">
                    <span class="ai-badge">{ai_meta['family']}</span>
                    <span class="ai-badge">{ai_meta['speed']}</span>
                </div>
                <div style="font-size:0.83rem; color:#57534E; line-height:1.45;">{ai_meta['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Nếu đến lượt AI đánh, thực hiện ngay
    if st.session_state["hv_need_ai_move"]:
        with st.spinner(f"{LEVEL_TO_ITEM[new_lv]['label']} đang suy nghĩ..."):
            _perform_ai_turn(factory, logs_dir, show_explain=show_explain)

    env: CaroEnv = st.session_state["hv_env"]
    human_side = int(st.session_state["hv_human_side"])
    win_cells = find_winning_cells(env)
    last_action = env.history[-1] if env.history else None
    hint_action = st.session_state["hv_hint_action"]

    col_board, col_info = st.columns([1.35, 1.0], gap="large")

    with col_board:
        # Banner trạng thái ván đấu
        if env.is_terminal():
            if env.winner == human_side:
                st.markdown(
                    '<div class="status-banner status-win">🏆 Chúc mừng! Bạn đã chiến thắng! '
                    f'<span>Tổng {len(env.history)} nước</span></div>',
                    unsafe_allow_html=True,
                )
            elif env.winner == -human_side:
                st.markdown(
                    f'<div class="status-banner status-turn-x">🤖 {LEVEL_TO_ITEM[new_lv]["short"]} đã thắng ván này! '
                    f'<span>Tổng {len(env.history)} nước</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="status-banner status-draw">🤝 Ván đấu hòa! '
                    f'<span>Tổng {len(env.history)} nước</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            side_txt = "X (Đỏ)" if env.current_player == 1 else "O (Xanh)"
            cls = "status-turn-x" if env.current_player == 1 else "status-turn-o"
            who = "Đến lượt bạn" if env.current_player == human_side else f"Đến lượt {LEVEL_TO_ITEM[new_lv]['short']}"
            extra = (
                f"Gợi ý: {action_to_coord(hint_action)}"
                if hint_action is not None
                else (f"Nước cuối: {action_to_coord(last_action)}" if last_action is not None else "Bàn cờ trống")
            )
            st.markdown(
                f'<div class="status-banner {cls}">🎯 {who} — quân {side_txt} <span>{extra}</span></div>',
                unsafe_allow_html=True,
            )

        # CSS động cho các ô có quân / ô thắng / ô gợi ý
        _inject_dynamic_cell_css(
            env,
            win_cells=win_cells,
            last_action=last_action,
            hint_action=hint_action,
        )

        # Bàn cờ tương tác 11x11 bằng lưới nút gốc Streamlit (gap=None để 11 hàng khít nhau tuyệt đối)
        n = env.board_size
        can_play = (not env.is_terminal()) and (env.current_player == human_side)
        with st.container(key="caro_board_frame", gap=None):
            # Hàng tiêu đề cột A..K (tỷ lệ cột số hàng đủ rộng để 10, 11 luôn nằm 1 dòng)
            hdr_cols = st.columns([0.82] + [1.0] * n, gap=None)
            hdr_cols[0].markdown('<div class="coord-col">&nbsp;</div>', unsafe_allow_html=True)
            for c_idx in range(n):
                hdr_cols[c_idx + 1].markdown(
                    f'<div class="coord-col">{COL_LABELS[c_idx]}</div>',
                    unsafe_allow_html=True,
                )

            # 11 hàng ô cờ
            for r in range(n):
                row_cols = st.columns([0.82] + [1.0] * n, gap=None, vertical_alignment="center")
                row_cols[0].markdown(f'<div class="coord-row">{r + 1}</div>', unsafe_allow_html=True)
                for c in range(n):
                    val = int(env.board[r, c])
                    act = r * n + c
                    if val == 1:
                        label = "✕"
                    elif val == -1:
                        label = "◯"
                    elif hint_action is not None and act == int(hint_action):
                        label = "★"
                    elif (r, c) in HOSHI_POINTS:
                        label = "·"
                    else:
                        label = "\u00A0"

                    row_cols[c + 1].button(
                        label,
                        key=f"cell_{r}_{c}",
                        disabled=(val != 0) or (not can_play),
                        on_click=_handle_cell_click,
                        args=(act, logs_dir),
                        use_container_width=True,
                    )

    with col_info:
        # Tỷ số phiên chơi & Telemetry gọn 1 dòng không tràn
        sb = st.session_state["hv_scoreboard"]
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Nước đi</div>
                    <div class="metric-value">{len(env.history)}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">AI (ms)</div>
                    <div class="metric-value">{st.session_state['hv_last_ai_ms']:.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Tỷ số</div>
                    <div class="metric-value">{sb['win']}-{sb['draw']}-{sb['loss']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Bảng phân bố xác suất / giải thích AI (bật/tắt tức thì theo công tắc sidebar)
        exp_data = st.session_state.get("hv_explain")
        if show_explain and exp_data and exp_data.get("rows"):
            cols_exp = list(exp_data["rows"][0].keys())
            th_html = "".join(f"<th>{k}</th>" for k in cols_exp)
            tr_html = "".join(
                "<tr>" + "".join(f"<td>{row[k]}</td>" for k in cols_exp) + "</tr>"
                for row in exp_data["rows"]
            )
            st.markdown(
                f"""
                <div class="studio-card">
                    <div class="studio-card-title">🧠 {exp_data['title']}</div>
                    <div style="font-size:0.78rem; color:#57534E; margin-bottom:0.35rem;">{exp_data['summary']}</div>
                    <div class="explain-table-wrap">
                        <table class="explain-table">
                            <thead><tr>{th_html}</tr></thead>
                            <tbody>{tr_html}</tbody>
                        </table>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Biên bản nước đi (tùy chọn, mặc định ẩn)
        if show_move_log:
            st.markdown('<div class="studio-card-title" style="margin-top:0.35rem;">📝 Biên bản nước đi</div>', unsafe_allow_html=True)
            if env.history:
                chips = []
                for idx, mv in enumerate(env.history, start=1):
                    is_x = (idx % 2) == 1
                    cls = "mv-x" if is_x else "mv-o"
                    sym = "X" if is_x else "O"
                    chips.append(f'<span class="{cls}">#{idx:02d} {sym}:{action_to_coord(mv)}</span>')
                rows_html = []
                for i in range(0, len(chips), 2):
                    pair = " &nbsp;│&nbsp; ".join(chips[i : i + 2])
                    rows_html.append(f"<div>{pair}</div>")
                st.markdown(f'<div class="move-history-box">{"".join(rows_html)}</div>', unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Tải biên bản (.json)",
                    data=json.dumps(
                        {
                            "mode": "human_vs_ai",
                            "human_side": "X" if human_side == 1 else "O",
                            "ai_level": new_lv,
                            "ai_id": LEVEL_TO_ITEM[new_lv]["id"],
                            "winner": "X" if env.winner == 1 else ("O" if env.winner == -1 else "ongoing_or_draw"),
                            "moves": [action_to_coord(m) for m in env.history],
                            "history": [int(m) for m in env.history],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    file_name=f"caro_human_vs_L{new_lv}_{datetime.now().strftime('%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.caption("Chưa có nước đi nào.")


def _render_ai_vs_ai_tab(factory: UIAgentFactory, logs_dir: Path, figures_dir: Path) -> None:
    import streamlit as st

    labels = [item["label"] for item in AGENT_CATALOG]

    with st.sidebar:
        st.markdown("### ⚔️ Cấu hình Máy vs Máy")
        lbl_a = st.selectbox("Tác tử A (Màu xanh trên biểu đồ)", options=labels, index=6, key="sb_ava_a")
        lbl_b = st.selectbox("Tác tử B (Màu đỏ trên biểu đồ)", options=labels, index=3, key="sb_ava_b")
        level_a = LABEL_TO_LEVEL[lbl_a]
        level_b = LABEL_TO_LEVEL[lbl_b]

        num_games = st.number_input("Số trận muốn đấu", min_value=1, max_value=200, value=6, step=2, key="sb_ava_n")
        swap_sides = st.checkbox("Hoán đổi X/O cân bằng 50/50", value=True, key="sb_ava_swap")
        diverse_openings = st.checkbox("Khai cuộc đa dạng (Opening Book)", value=True, key="sb_ava_open")
        fast_mode = st.checkbox("Chế độ nhanh (fast_mode)", value=True, key="sb_ava_fast")
        live_watch = st.checkbox("Xem bàn cờ trực tiếp từng nước", value=True, key="sb_ava_live")
        move_delay = st.slider(
            "Độ trễ mỗi nước khi xem trực tiếp (giây)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.05,
            disabled=not live_watch,
            key="sb_ava_delay",
        )
        base_seed = st.number_input("Seed tái lập", min_value=0, max_value=999999, value=42, step=1, key="sb_ava_seed")

        run_btn = st.button("▶️ Bắt đầu chuỗi trận", type="primary", use_container_width=True)

    col_arena, col_stats = st.columns([1.2, 1.0], gap="large")
    with col_arena:
        board_slot = st.empty()
        live_caption = st.empty()
    with col_stats:
        progress_slot = st.empty()
        score_slot = st.empty()

    if run_btn:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        a_id = LEVEL_TO_ITEM[level_a]["id"]
        b_id = LEVEL_TO_ITEM[level_b]["id"]
        log_path = logs_dir / f"ai_vs_ai_{a_id}_vs_{b_id}_{ts}.jsonl"

        prog = progress_slot.progress(0, text=f"Đang chuẩn bị chuỗi {num_games} trận: {a_id} vs {b_id}...")
        live_records: list[dict[str, Any]] = []

        def _on_move(env_snap: CaroEnv, _act: int, g_idx: int, dt_ms: float, sc: float) -> None:
            if live_watch:
                board_slot.markdown(
                    render_board_svg(env_snap, show_move_numbers=True, size_px=500),
                    unsafe_allow_html=True,
                )
                live_caption.caption(
                    f"Trận {g_idx}/{num_games} • Nước #{len(env_snap.history)} • "
                    f"Thời gian nước vừa rồi: {dt_ms:.1f} ms (search={int(sc)})"
                )

        def _on_game_done(rec: dict[str, Any], done_n: int, total_n: int) -> None:
            live_records.append(rec)
            prog.progress(
                done_n / total_n,
                text=f"Hoàn tất trận {done_n}/{total_n} — Thắng: {rec['winner_agent']} ({rec['num_moves']} nước)",
            )
            wins_a = sum(1 for r in live_records if r["result_a"] == 1.0)
            draws = sum(1 for r in live_records if r["result_a"] == 0.5)
            losses_a = sum(1 for r in live_records if r["result_a"] == 0.0)
            pct_a = 100.0 * (wins_a + 0.5 * draws) / len(live_records)
            score_slot.markdown(
                f"""
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">{a_id} thắng</div>
                        <div class="metric-value" style="color:#1D4ED8;">{wins_a}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Hòa</div>
                        <div class="metric-value" style="color:#57534E;">{draws}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">{b_id} thắng</div>
                        <div class="metric-value" style="color:#B91C1C;">{losses_a}</div>
                    </div>
                </div>
                <div class="studio-card">
                    <div class="studio-card-title">Tỷ lệ điểm hiện tại</div>
                    <div style="font-weight:700; font-size:1.05rem;">{a_id}: {pct_a:.1f}% &nbsp;vs&nbsp; {b_id}: {100.0 - pct_a:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not live_watch:
                env_end = CaroEnv(board_size=11, win_length=5)
                for m in rec["history"]:
                    env_end.step(int(m))
                board_slot.markdown(
                    render_board_svg(env_end, show_move_numbers=True, size_px=500),
                    unsafe_allow_html=True,
                )

        records = run_ai_vs_ai_series(
            factory=factory,
            level_a=level_a,
            level_b=level_b,
            num_games=int(num_games),
            swap_sides=bool(swap_sides),
            diverse_openings=bool(diverse_openings),
            fast_mode=bool(fast_mode),
            base_seed=int(base_seed),
            log_path=log_path,
            on_move=_on_move,
            on_game_done=_on_game_done,
            move_delay_sec=float(move_delay) if live_watch else 0.0,
        )
        st.session_state["ava_records"] = records
        st.session_state["ava_log_path"] = str(log_path)
        st.session_state["ava_chart_bytes"] = render_analytics_chart_bytes(records) if records else None
        st.session_state["ava_last_pair"] = (a_id, b_id)
        st.success(f"Hoàn tất {len(records)} trận! Log đã tự động lưu tại `{log_path.name}`.")

    # Hiển thị kết quả chuỗi trận hiện có
    records: list[dict[str, Any]] = st.session_state.get("ava_records") or []
    if not records:
        board_slot.markdown(
            render_board_svg(CaroEnv(board_size=11, win_length=5), show_move_numbers=False, size_px=490),
            unsafe_allow_html=True,
        )
        score_slot.info(
            "Chọn 2 cấp độ AI ở thanh bên trái và bấm **▶️ Bắt đầu chuỗi trận** để chạy đối kháng, "
            "xem trực tiếp từng nước và xuất biểu đồ phân tích 4 khung."
        )
        return

    # Cập nhật KPI tổng kết & trình xem lại ngay trên trang
    a_id = str(records[0]["agent_a"])
    b_id = str(records[0]["agent_b"])
    n_tot = len(records)
    wins_a = sum(1 for r in records if r["result_a"] == 1.0)
    draws = sum(1 for r in records if r["result_a"] == 0.5)
    losses_a = sum(1 for r in records if r["result_a"] == 0.0)
    pts_a = wins_a + 0.5 * draws
    pct_a = 100.0 * pts_a / n_tot
    w_lo, w_hi = wilson_ci95(pts_a, n_tot)

    score_slot.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">{a_id} thắng</div>
                <div class="metric-value" style="color:#1D4ED8;">{wins_a}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Hòa</div>
                <div class="metric-value" style="color:#57534E;">{draws}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{b_id} thắng</div>
                <div class="metric-value" style="color:#B91C1C;">{losses_a}</div>
            </div>
        </div>
        <div class="studio-card">
            <div class="studio-card-title">Tổng kết chuỗi {n_tot} trận</div>
            <div style="font-weight:700; font-size:1.05rem;">
                {a_id}: {pct_a:.1f}% <span style="font-size:0.82rem; color:#78716C;">[CI95: {w_lo:.1f}–{w_hi:.1f}%]</span>
            </div>
            <div style="font-size:0.84rem; color:#57534E; margin-top:0.2rem;">
                {b_id}: {100.0 - pct_a:.1f}% • Độ dài TB: {statistics.mean(r['num_moves'] for r in records):.1f} nước
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with col_stats:
        game_opts = [
            f"Trận #{r['game_index']} — X:{r['x_agent']} vs O:{r['o_agent']} → Thắng: {r['winner_agent']} ({r['num_moves']} nước)"
            for r in records
        ]
        sel_idx = st.selectbox("Chọn trận để xem lại (Replay)", range(len(records)), format_func=lambda i: game_opts[i])
        rec_sel = records[sel_idx]
        hist = rec_sel.get("history", [])
        step_k = st.slider("Tua đến nước số", 0, len(hist), len(hist), key="ava_replay_step")
        show_nums = st.checkbox("Hiện số thứ tự trên quân cờ", value=True, key="ava_replay_nums")

        env_rep = CaroEnv(board_size=11, win_length=5)
        for m in hist[:step_k]:
            env_rep.step(int(m))
        board_slot.markdown(
            render_board_svg(env_rep, show_move_numbers=show_nums, size_px=500),
            unsafe_allow_html=True,
        )
        live_caption.caption(
            f"Replay Trận #{rec_sel['game_index']} • Nước {step_k}/{len(hist)} • "
            f"X={rec_sel['x_agent']} | O={rec_sel['o_agent']} | Thắng={rec_sel['winner_agent']}"
        )

    st.divider()
    st.markdown("### 📊 Biểu đồ phân tích 4 khung & Xuất dữ liệu")
    chart_bytes = st.session_state.get("ava_chart_bytes")
    if chart_bytes:
        st.image(chart_bytes, use_container_width=True)

    df_summary = build_records_summary_df(records)
    b1, b2, b3 = st.columns(3)
    with b1:
        if chart_bytes:
            st.download_button(
                "🖼️ Tải biểu đồ PNG (180 DPI)",
                data=chart_bytes,
                file_name=f"ai_vs_ai_{a_id}_vs_{b_id}_{n_tot}g.png",
                mime="image/png",
                use_container_width=True,
            )
    with b2:
        st.download_button(
            "📄 Tải bảng tổng hợp (.csv)",
            data=df_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"ai_vs_ai_{a_id}_vs_{b_id}_{n_tot}g.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with b3:
        if st.button("💾 Lưu biểu đồ PNG vào results/figures/", use_container_width=True):
            ts_save = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_png = figures_dir / f"ui_ai_vs_ai_{a_id}_vs_{b_id}_{ts_save}.png"
            out_csv = figures_dir.parent / "tables" / f"ui_ai_vs_ai_{a_id}_vs_{b_id}_{ts_save}.csv"
            export_ai_vs_ai_analytics_chart(records, out_png, csv_summary_path=out_csv)
            st.success(f"Đã lưu tại `{out_png.relative_to(figures_dir.parents[1])}`")

    with st.expander("📋 Xem bảng chi tiết từng trận đấu", expanded=False):
        st.dataframe(df_summary, use_container_width=True, hide_index=True)


def _render_history_replay_tab(project_root: Path) -> None:
    import streamlit as st

    logs_root = project_root / "results" / "logs"
    jsonl_files = sorted(logs_root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if logs_root.exists() else []

    if not jsonl_files:
        st.info("Chưa tìm thấy file log `.jsonl` nào trong `results/logs/`. Hãy chơi 1 ván ở tab Người vs Máy hoặc Máy vs Máy!")
        return

    with st.sidebar:
        st.markdown("### 📜 Nguồn dữ liệu Log")
        rel_labels = [str(p.relative_to(logs_root)).replace("\\", "/") for p in jsonl_files]
        sel_file_idx = st.selectbox("Chọn file log `.jsonl`", range(len(jsonl_files)), format_func=lambda i: rel_labels[i])
        show_nums = st.checkbox("Hiện số thứ tự trên quân cờ", value=True, key="rep_show_nums")
        show_rep_move_log = st.checkbox("Hiển thị biên bản nước đi", value=False, key="rep_show_move_log")

    chosen_path = jsonl_files[sel_file_idx]
    records = load_match_logs(chosen_path)
    if not records:
        st.warning(f"File `{chosen_path.name}` không có bản ghi hợp lệ.")
        return

    # Chuẩn hóa trường hiển thị cho cả log UI lẫn log Tournament
    def _fmt_rec(idx: int, r: dict[str, Any]) -> str:
        x_ag = r.get("x_agent") or r.get("player_x") or r.get("agent_a", "X")
        o_ag = r.get("o_agent") or r.get("player_o") or r.get("agent_b", "O")
        win = r.get("winner_agent") or r.get("winner_name") or r.get("winner_side") or r.get("winner", "?")
        n_mv = r.get("num_moves") or len(r.get("history") or r.get("moves") or [])
        return f"#{idx + 1} | X: {x_ag} vs O: {o_ag} → Thắng: {win} ({n_mv} nước)"

    col_left, col_right = st.columns([1.25, 1.0], gap="large")
    with col_right:
        sel_game = st.selectbox(
            f"Chọn ván đấu ({len(records)} ván trong file)",
            range(len(records)),
            format_func=lambda i: _fmt_rec(i, records[i]),
            key="rep_game_idx",
        )
        rec = records[sel_game]
        raw_moves = rec.get("history") or rec.get("moves") or []
        moves = [int(m) for m in raw_moves if isinstance(m, (int, np.integer))]

        step = st.slider("Tua ván đấu theo từng nước", 0, len(moves), len(moves), key=f"rep_step_{sel_file_idx}_{sel_game}")

        x_ag = rec.get("x_agent") or rec.get("player_x") or rec.get("agent_a", "X")
        o_ag = rec.get("o_agent") or rec.get("player_o") or rec.get("agent_b", "O")
        win = rec.get("winner_agent") or rec.get("winner_name") or rec.get("winner_side") or rec.get("winner", "?")

        st.markdown(
            f"""
            <div class="studio-card">
                <div class="studio-card-title">Thông tin ván đấu</div>
                <div><b>File:</b> <code>{chosen_path.name}</code></div>
                <div><b>Quân X (Đỏ):</b> {x_ag}</div>
                <div><b>Quân O (Xanh):</b> {o_ag}</div>
                <div><b>Kết quả:</b> {win} ({len(moves)} nước)</div>
                <div><b>Độ trễ TB:</b> A={rec.get('avg_ms_a', '-')} ms | B={rec.get('avg_ms_b', '-')} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_rep_move_log and moves:
            shown = moves[:step]
            chips = [
                f'<span class="{"mv-x" if (i % 2) == 1 else "mv-o"}">#{i:02d} {"X" if (i % 2) == 1 else "O"}:{action_to_coord(m)}</span>'
                for i, m in enumerate(shown, start=1)
            ]
            rows_html = [f"<div>{' &nbsp;│&nbsp; '.join(chips[i:i+2])}</div>" for i in range(0, len(chips), 2)]
            st.markdown('<div class="studio-card-title">📝 Biên bản nước đi</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="move-history-box">{"".join(rows_html) or "<i>Chưa có nước đi</i>"}</div>', unsafe_allow_html=True)

    with col_left:
        env_rep = CaroEnv(board_size=11, win_length=5)
        for m in moves[:step]:
            env_rep.step(m)
        st.markdown(
            render_board_svg(env_rep, show_move_numbers=show_nums, size_px=540),
            unsafe_allow_html=True,
        )
        st.caption(f"Đang xem nước {step}/{len(moves)} — X: {x_ag} vs O: {o_ag}")

    # Nếu file là log AI vs AI có đủ trường `result_a`, cho phép xuất biểu đồ ngay
    if all("result_a" in r and "agent_a" in r and "agent_b" in r for r in records) and len(records) >= 1:
        with st.expander("📊 Xem & tải biểu đồ phân tích cho file log này", expanded=False):
            png_bytes = render_analytics_chart_bytes(records)
            st.image(png_bytes, use_container_width=True)
            st.download_button(
                "🖼️ Tải biểu đồ PNG (180 DPI)",
                data=png_bytes,
                file_name=f"{chosen_path.stem}_analytics.png",
                mime="image/png",
            )


def run_streamlit_app(project_root: str | Path | None = None) -> None:
    """Khởi chạy toàn bộ ứng dụng web Streamlit Caro AI 11×11 Studio."""
    import streamlit as st

    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    logs_dir = root / "results" / "logs" / "ui_matches"
    figures_dir = root / "results" / "figures"
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    st.set_page_config(
        page_title="Caro AI 11×11 Studio",
        page_icon="♟️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_studio_css()
    _init_session_state()

    with st.sidebar:
        st.markdown("## ♟️ Caro AI 11×11")
        mode = st.radio(
            "Chế độ làm việc",
            options=["🎮 Người vs Máy", "⚔️ Máy vs Máy", "📜 Lịch sử & Replay"],
            index=0,
            key="studio_nav_mode",
        )
        st.divider()

    st.markdown(
        f"""
        <div class="caro-hero">
            <div>
                <h1 class="caro-hero-title">Caro AI 11×11 Studio</h1>
                <p class="caro-hero-sub">
                    Nền tảng tương tác web & thực nghiệm 7 cấp độ AI (Random → Heuristic → Minimax → MCTS → CNN → DQN → AlphaGo PUCT)
                </p>
            </div>
            <div>
                <span class="caro-pill">11×11 • 5-in-a-row</span>
                <span class="caro-pill">{mode}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    factory = _get_cached_factory(root)

    if mode == "🎮 Người vs Máy":
        _render_human_vs_ai_tab(factory, logs_dir)
    elif mode == "⚔️ Máy vs Máy":
        _render_ai_vs_ai_tab(factory, logs_dir, figures_dir)
    else:
        _render_history_replay_tab(root)


def launch_ui(
    project_root: str | Path | None = None,
    *,
    port: int = 8501,
    headless: bool = False,
) -> None:
    """Khởi chạy máy chủ web Streamlit cho ứng dụng Caro 11×11."""
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true" if headless else "false",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


if __name__ == "__main__":
    if _is_running_in_streamlit():
        run_streamlit_app(project_root=_PROJECT_ROOT)
    else:
        launch_ui(project_root=_PROJECT_ROOT)
