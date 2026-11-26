"""Hệ thống giải đấu Round-Robin, độ đo thống kê & Ablation cho Caro 11x11 (Mục 3.5).

Đáp ứng đầy đủ các yêu cầu thực nghiệm đối kháng (Mục 3.5 đề tài TTNTNC & sách
*AlphaGo Simplified* - Mark Liu):
- Giải đấu vòng tròn (round-robin) đầy đủ giữa tất cả 7 cấp độ AI (Cấp 1–7).
- Cân bằng tuyệt đối 50/50 lượt đi trước (X) và đi sau (O) theo từng cặp khai cuộc đối xứng.
- Đa seed cho các tác tử có tính ngẫu nhiên / phá hòa (`seeds = [42, 102, 202, 302, 402]`).
- Báo cáo tối thiểu 2 chế độ minh bạch:
  1. `time_budget`: Ngân sách thời gian cố định trên mỗi nước đi (ví dụ `0.25s/nước`).
  2. `default_config`: Cấu hình mặc định chuẩn của từng cấp độ tác tử.
- Độ đo đầy đủ: W/D/L, Score = (W + 0.5*D)/N, Wilson CI95, Bootstrap CI95,
  avg_ms, p95_ms, avg_search, RAM (RSS), VRAM (nếu có GPU), thời gian huấn luyện,
  kích thước checkpoint, Elo/Bradley-Terry, phân tích First-Player Advantage & Seed Variance.
- Lưu log thô từng trận đấu cho **từng cặp đấu** dưới dạng `.jsonl` (hỗ trợ `resume=True`).
- Hỗ trợ 03 thí nghiệm Ablation / Sensitivity chuẩn (Alpha-Beta & Depth, MCTS Budget &
  Heuristic Playout, Policy-only vs Policy+PUCT vs AlphaGo Cấp 7).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
import json
import math
import multiprocessing
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

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
except ImportError:  # pragma: no cover - fallback when executed inside src/
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


# ---------------------------------------------------------------------------
# 1. Cố định phiên bản trò chơi, luật và bộ khai cuộc đối xứng 11x11
# ---------------------------------------------------------------------------

GAME_SPEC: dict[str, Any] = {
    "game_name": "Caro / Free Gomoku 11x11",
    "engine_version": "CaroEnv v1.0",
    "board_size": 11,
    "win_length": 5,
    "first_player": "X (1)",
    "second_player": "O (-1)",
    "overline_rule": "Free Gomoku (>= 5 liên tiếp là thắng, không cấm quá 5, không luật chặn 2 đầu)",
    "observability": "Full observability, zero-sum, turn-based",
    "draw_condition": "Bàn cờ đầy 121 nước không có bên nào đủ 5 quân liên tiếp",
}

DEFAULT_SEEDS: tuple[int, ...] = (42, 102, 202, 302, 402)

# Bộ 20 khai cuộc 11x11 cân bằng: chỉ số 0 là bàn trống chuẩn (standard),
# 19 khai cuộc còn lại là thế cờ 2 nước (X tại vùng trung tâm, O phản hồi lân cận)
# giúp các cặp đấu tất định (như Lv2 vs Lv3) không lặp lại cùng 1 ván duy nhất.
# Mọi khai cuộc đều được ghép cặp đối xứng (A đi X 1 ván, A đi O 1 ván) để loại trừ thiên lệch.
OPENING_BOOK_11X11: tuple[tuple[int, ...], ...] = (
    (),          # 0: Standard empty board
    (60, 49),    # 1: (5,5) vs (4,5) trực giao
    (60, 48),    # 2: (5,5) vs (4,4) chéo gần
    (60, 61),    # 3: (5,5) vs (5,6) ngang
    (60, 72),    # 4: (5,5) vs (6,6) chéo dưới
    (60, 50),    # 5: (5,5) vs (4,6) chéo trên phải
    (60, 70),    # 6: (5,5) vs (6,4) chéo dưới trái
    (60, 71),    # 7: (5,5) vs (6,5) dọc dưới
    (60, 59),    # 8: (5,5) vs (5,4) ngang trái
    (60, 38),    # 9: (5,5) vs (3,5) nhảy 2 ô dọc
    (60, 37),    # 10: (5,5) vs (3,4) mã
    (60, 39),    # 11: (5,5) vs (3,6) mã phải
    (60, 51),    # 12: (5,5) vs (4,7) mã ngang
    (60, 62),    # 13: (5,5) vs (5,7) nhảy 2 ô ngang
    (60, 83),    # 14: (5,5) vs (7,6) mã dưới
    (48, 60),    # 15: (4,4) vs (5,5) lệch tâm
    (49, 60),    # 16: (4,5) vs (5,5) lệch tâm dọc
    (59, 60),    # 17: (5,4) vs (5,5) lệch tâm ngang
    (50, 60),    # 18: (4,6) vs (5,5) lệch tâm chéo
    (72, 60),    # 19: (6,6) vs (5,5) đối xứng tâm
)

# Thông tin thời gian huấn luyện thực tế đã đo (để báo cáo minh bạch theo Mục 3.4 & 3.5)
DEFAULT_TRAINING_METADATA: dict[str, dict[str, Any]] = {
    "L1_Random": {"train_time_sec": 0.0, "checkpoint_rel": None, "method_family": "Baseline"},
    "L2_RuleBased": {"train_time_sec": 0.0, "checkpoint_rel": None, "method_family": "Rule/Heuristic"},
    "L3_Minimax": {"train_time_sec": 0.0, "checkpoint_rel": None, "method_family": "Adversarial Search"},
    "L4_MCTS": {"train_time_sec": 0.0, "checkpoint_rel": None, "method_family": "Monte Carlo Search"},
    "L5_CNN": {
        "train_time_sec": 185.0,  # ~15 epochs ResNet-4x64 trên CPU (15,472 mẫu train)
        "checkpoint_rel": "checkpoints/05_cnn_policy_best.pt",
        "method_family": "Supervised Deep Learning",
    },
    "L6_RL": {
        "train_time_sec": 449.3,  # Thực đo từ results/tables/06_rl_dqn6a_training_log.csv
        "checkpoint_rel": "checkpoints/06_rl_best.pt",
        "method_family": "Deep Reinforcement Learning",
    },
    "L7_AlphaGo": {
        "train_time_sec": 634.3,  # Tổng thời gian huấn luyện Cấp 5 (185.0s) + Cấp 6 (449.3s)
        "checkpoint_rel": "checkpoints/05_cnn_policy_best.pt+checkpoints/06_rl_best.pt",
        "method_family": "AlphaGo-Inspired (Neural+PUCT)",
    },
}


# ---------------------------------------------------------------------------
# 2. Phần cứng, bộ nhớ RAM/VRAM & Độ đo thống kê (Wilson, Bootstrap, Elo, p95)
# ---------------------------------------------------------------------------

def get_process_rss_mb() -> float:
    """Đo RAM hiện tại (RSS, MB) của tiến trình Python (hỗ trợ Windows & POSIX)."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    if sys.platform == "win32":
        try:
            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            h_proc = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                h_proc, ctypes.byref(counters), counters.cb
            ):
                return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass
    else:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux returns KB, macOS returns bytes
            if sys.platform == "darwin":
                return float(rss) / (1024.0 * 1024.0)
            return float(rss) / 1024.0
        except Exception:
            pass
    return 0.0


def get_gpu_vram_info() -> dict[str, Any]:
    """Trả về thông tin GPU/VRAM nếu PyTorch có CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            alloc_mb = float(torch.cuda.memory_allocated(idx)) / (1024.0 * 1024.0)
            return {
                "device": f"CUDA ({props.name})",
                "vram_total_mb": round(float(props.total_memory) / (1024.0 * 1024.0), 1),
                "vram_allocated_mb": round(alloc_mb, 2),
            }
    except Exception:
        pass
    return {"device": "CPU-only", "vram_total_mb": 0.0, "vram_allocated_mb": 0.0}


def collect_hardware_info() -> dict[str, Any]:
    """Thu thập cấu hình phần cứng & phần mềm cố định cho báo cáo (Mục 3.5)."""
    torch_ver = "N/A"
    try:
        import torch

        torch_ver = str(torch.__version__)
    except Exception:
        pass

    gpu = get_gpu_vram_info()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor() or "Unknown CPU",
        "cpu_cores_logical": os.cpu_count() or 1,
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "torch_version": torch_ver,
        "compute_device": gpu["device"],
        "vram_total_mb": gpu["vram_total_mb"],
        "process_rss_mb": round(get_process_rss_mb(), 2),
    }


def wilson_ci95(successes: float, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Khoảng tin cậy 95% Wilson (đơn vị %) cho điểm số trung bình `successes / total`."""
    if total <= 0:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, float(successes) / float(total)))
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return (round(100.0 * max(0.0, center - half), 2), round(100.0 * min(1.0, center + half), 2))


def bootstrap_ci95(
    scores_01: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Khoảng tin cậy 95% Bootstrap phi tham số (đơn vị %) trên điểm từng ván `{0.0, 0.5, 1.0}`."""
    n = len(scores_01)
    if n == 0:
        return (0.0, 0.0)
    arr = np.asarray(scores_01, dtype=np.float64)
    if np.all(arr == arr[0]):
        val = round(float(arr[0]) * 100.0, 2)
        return (val, val)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (round(float(lo), 2), round(float(hi), 2))


def percentile_p95(values: Sequence[float]) -> float:
    """Tính phân vị thứ 95 (p95) của danh sách độ trễ (ms)."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def compute_elo_ratings(
    matchup_rows: Sequence[dict[str, Any]],
    agent_ids: Sequence[str],
    base_elo: float = 1500.0,
    max_iters: int = 400,
) -> dict[str, float]:
    """Tính bảng xếp hạng Elo (Bradley-Terry MLE ổn định với prior Laplace nhẹ 0.5 ván)."""
    ids = list(agent_ids)
    n = len(ids)
    if n == 0:
        return {}
    idx_map = {aid: i for i, aid in enumerate(ids)}
    wins_mat = np.zeros((n, n), dtype=np.float64)
    games_mat = np.zeros((n, n), dtype=np.float64)

    for row in matchup_rows:
        a = row["agent_a"]
        b = row["agent_b"]
        if a not in idx_map or b not in idx_map:
            continue
        ia, ib = idx_map[a], idx_map[b]
        g = float(row["games"])
        sa = float(row["wins_a"]) + 0.5 * float(row["draws"])
        sb = g - sa
        wins_mat[ia, ib] += sa
        wins_mat[ib, ia] += sb
        games_mat[ia, ib] += g
        games_mat[ib, ia] += g

    # Laplace smoothing nhẹ (0.25 điểm / 0.5 ván) để Elo hội tụ hữu hạn ngay cả khi có đội 100% hoặc 0%
    eps_w = 0.25
    eps_g = 0.50
    strengths = np.ones(n, dtype=np.float64)
    for _ in range(max_iters):
        new_s = np.copy(strengths)
        for i in range(n):
            num = eps_w * (n - 1)
            den = 0.0
            for j in range(n):
                if i == j:
                    continue
                g_ij = games_mat[i, j] + eps_g
                w_ij = wins_mat[i, j]
                num += w_ij
                den += g_ij / (strengths[i] + strengths[j])
            new_s[i] = max(1e-9, num / max(den, 1e-9))
        # Chuẩn hóa tích hình học = 1
        geo = float(np.exp(np.mean(np.log(new_s))))
        new_s /= geo
        if np.max(np.abs(new_s - strengths)) < 1e-8:
            strengths = new_s
            break
        strengths = new_s

    elos = base_elo + 400.0 * np.log10(strengths)
    elos = elos - float(np.mean(elos)) + base_elo
    return {aid: round(float(elos[i]), 1) for i, aid in enumerate(ids)}


# ---------------------------------------------------------------------------
# 3. Định nghĩa cấu hình 7 tác tử cho 2 chế độ ngân sách (Mục 3.5)
# ---------------------------------------------------------------------------

def recommended_workers(max_cap: int = 8) -> int:
    """Số tiến trình song song an toàn (<= nửa số luồng logic, trần max_cap) để tránh nghẽn CPU."""
    cores = os.cpu_count() or 2
    return max(1, min(int(max_cap), cores // 2))


@dataclass
class AgentSpec:
    """Đặc tả một tác tử tham gia giải đấu."""

    agent_id: str
    display_name: str
    level: int
    mode: str
    factory: Callable[[int], Any]
    config_summary: str
    budget_note: str
    train_time_sec: float = 0.0
    checkpoint_size_mb: float = 0.0
    extra_meta: dict[str, Any] = field(default_factory=dict)
    descriptor: dict[str, Any] | None = None


def _checkpoint_size_mb(root: Path, rel_spec: str | None) -> float:
    if not rel_spec:
        return 0.0
    total_bytes = 0
    for part in rel_spec.split("+"):
        p = root / part.strip()
        if p.exists():
            total_bytes += p.stat().st_size
    return round(total_bytes / (1024.0 * 1024.0), 3)


def _load_shared_neural_models(
    project_root: Path,
    cnn_ckpt: Path | None = None,
    rl_ckpt: Path | None = None,
) -> tuple[CNNAgent, RLAgent]:
    """Nạp sẵn mô hình mạng nơ-ron Cấp 5 và Cấp 6 một lần duy nhất để chia sẻ bộ nhớ."""
    cnn_path = cnn_ckpt or (project_root / "checkpoints" / "05_cnn_policy_best.pt")
    rl_path = rl_ckpt or (project_root / "checkpoints" / "06_rl_best.pt")

    if not cnn_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Cấp 5: {cnn_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint Cấp 6: {rl_path}")

    shared_cnn = CNNAgent(
        name="Shared_CNN_L5",
        model_path=cnn_path,
        use_tactical_shortcut=True,
        use_mix_mcts=False,
        deterministic=True,
        seed=42,
    )
    shared_rl = RLAgent(
        checkpoint_path=rl_path,
        name="Shared_RL_L6",
    )
    return shared_cnn, shared_rl


class _SharedPolicyWrapper:
    """Bọc mỏng quanh `CNNAgent` đã nạp trọng số để tránh load lại checkpoint mỗi ván."""

    def __init__(self, base_cnn: CNNAgent, name: str, seed: int) -> None:
        self._base = base_cnn
        self.name = name
        self.seed = seed
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 1

    def select_action(self, env: CaroEnv) -> int:
        act = self._base.select_action(env)
        self.last_move_time_ms = float(getattr(self._base, "last_move_time_ms", 0.0))
        self.last_search_count = int(getattr(self._base, "last_search_count", 1))
        return act


class _SharedRLWrapper:
    """Bọc mỏng quanh `RLAgent` đã nạp trọng số để tránh load lại checkpoint mỗi ván."""

    def __init__(self, base_rl: RLAgent, name: str, seed: int) -> None:
        self._base = base_rl
        self.name = name
        self.seed = seed
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 1

    def select_action(self, env: CaroEnv) -> int:
        act = self._base.select_action(env)
        self.last_move_time_ms = float(getattr(self._base, "last_move_time_ms", 0.0))
        self.last_search_count = int(getattr(self._base, "last_search_count", 1))
        return act


def build_agent_specs(
    mode: str = "default_config",
    time_budget_sec: float = 0.25,
    project_root: str | Path | None = None,
    cnn_ckpt: str | Path | None = None,
    rl_ckpt: str | Path | None = None,
    include_levels: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
) -> list[AgentSpec]:
    """Khởi tạo danh sách `AgentSpec` cho chế độ `time_budget` hoặc `default_config`.

    Lưu ý khoa học (Mục 3.5):
    - Ở chế độ `time_budget`: mọi tác tử tìm kiếm (`L3_Minimax`, `L4_MCTS`, `L7_AlphaGo`)
      đều bị giới hạn bởi cùng một trần thời gian tường `time_budget_sec` trên mỗi nước đi.
      Các tác tử không tìm kiếm cây (`L1`, `L2`, `L5`, `L6`) vận hành ở độ trễ suy luận tự nhiên
      (`< 15 ms`) và luôn được báo cáo song song `avg_ms`, `p95_ms`, `avg_search`.
    - Ở chế độ `default_config`: mỗi tác tử dùng đúng cấu hình mặc định chuẩn của cấp độ đó.
    """
    if mode not in ("default_config", "time_budget"):
        raise ValueError(f"mode không hợp lệ: {mode!r}. Chỉ hỗ trợ 'default_config' hoặc 'time_budget'.")

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    need_nn = any(lv in include_levels for lv in (5, 6, 7))
    shared_cnn: CNNAgent | None = None
    shared_rl: RLAgent | None = None
    if need_nn:
        shared_cnn, shared_rl = _load_shared_neural_models(
            root,
            Path(cnn_ckpt) if cnn_ckpt else None,
            Path(rl_ckpt) if rl_ckpt else None,
        )

    tb = float(time_budget_sec)
    tb_ms = int(round(tb * 1000))
    specs: list[AgentSpec] = []

    def _meta(aid: str) -> tuple[float, float]:
        info = DEFAULT_TRAINING_METADATA[aid]
        return float(info["train_time_sec"]), _checkpoint_size_mb(root, info["checkpoint_rel"])

    if 1 in include_levels:
        t_sec, ck_mb = _meta("L1_Random")
        specs.append(
            AgentSpec(
                agent_id="L1_Random",
                display_name="L1 Random",
                level=1,
                mode=mode,
                factory=lambda seed: RandomAgent(seed=seed, name="L1_Random"),
                config_summary="Uniform random over all legal moves",
                budget_note="O(1) single-step (<0.05 ms)",
                train_time_sec=t_sec,
                checkpoint_size_mb=ck_mb,
            )
        )

    if 2 in include_levels:
        t_sec, ck_mb = _meta("L2_RuleBased")
        specs.append(
            AgentSpec(
                agent_id="L2_RuleBased",
                display_name="L2 RuleBased",
                level=2,
                mode=mode,
                factory=lambda seed: RuleBasedAgent(
                    name="L2_RuleBased",
                    candidate_radius=2,
                    root_width=14,
                    seed=seed,
                    tie_break_eps=1.5,
                ),
                config_summary="7-tier rules + 3-ply look-ahead (root_width=14, eps=1.5)",
                budget_note="Fixed 3-ply shallow look-ahead (~5–15 ms)",
                train_time_sec=t_sec,
                checkpoint_size_mb=ck_mb,
            )
        )

    if 3 in include_levels:
        t_sec, ck_mb = _meta("L3_Minimax")
        if mode == "time_budget":
            specs.append(
                AgentSpec(
                    agent_id="L3_Minimax",
                    display_name="L3 Minimax-AB",
                    level=3,
                    mode=mode,
                    factory=lambda seed, _tb=tb: MinimaxAgent(
                        name="L3_Minimax",
                        max_depth=4,
                        use_alpha_beta=True,
                        use_move_ordering=True,
                        use_tt=True,
                        max_branch_root=14,
                        max_branch_inner=10,
                        time_limit_sec=_tb,
                        seed=seed,
                        tie_break_eps=1.5,
                    ),
                    config_summary=f"ID Alpha-Beta + TT + Killer (cap d=4, K=14/10, time={tb_ms}ms)",
                    budget_note=f"Time budget = {tb_ms} ms/move",
                    train_time_sec=t_sec,
                    checkpoint_size_mb=ck_mb,
                )
            )
        else:
            specs.append(
                AgentSpec(
                    agent_id="L3_Minimax",
                    display_name="L3 Minimax-AB",
                    level=3,
                    mode=mode,
                    factory=lambda seed: MinimaxAgent(
                        name="L3_Minimax",
                        max_depth=3,
                        use_alpha_beta=True,
                        use_move_ordering=True,
                        use_tt=True,
                        max_branch_root=16,
                        max_branch_inner=12,
                        time_limit_sec=1.0,
                        seed=seed,
                        tie_break_eps=1.5,
                    ),
                    config_summary="Default Alpha-Beta + TT + Killer (d=3, K=16/12, cap=1.0s)",
                    budget_note="Default depth=3 (K=16/12)",
                    train_time_sec=t_sec,
                    checkpoint_size_mb=ck_mb,
                )
            )

    if 4 in include_levels:
        t_sec, ck_mb = _meta("L4_MCTS")
        if mode == "time_budget":
            specs.append(
                AgentSpec(
                    agent_id="L4_MCTS",
                    display_name="L4 MCTS-UCT",
                    level=4,
                    mode=mode,
                    factory=lambda seed, _tb=tb: MCTSAgent(
                        num_rollouts=5000,
                        exploration_c=1.414,
                        time_limit_sec=_tb,
                        rollout_policy="heuristic",
                        use_heuristic_leaf=True,
                        use_progressive_widening=True,
                        seed=seed,
                        name="L4_MCTS",
                    ),
                    config_summary=f"UCT + Heavy Playout + PW (C=1.414, time={tb_ms}ms)",
                    budget_note=f"Time budget = {tb_ms} ms/move",
                    train_time_sec=t_sec,
                    checkpoint_size_mb=ck_mb,
                )
            )
        else:
            specs.append(
                AgentSpec(
                    agent_id="L4_MCTS",
                    display_name="L4 MCTS-UCT",
                    level=4,
                    mode=mode,
                    factory=lambda seed: MCTSAgent(
                        num_rollouts=300,
                        exploration_c=1.414,
                        time_limit_sec=None,
                        rollout_policy="heuristic",
                        use_heuristic_leaf=True,
                        use_progressive_widening=True,
                        seed=seed,
                        name="L4_MCTS",
                    ),
                    config_summary="Default UCT + Heavy Playout + PW (300 rollouts, C=1.414)",
                    budget_note="Default 300 rollouts/move",
                    train_time_sec=t_sec,
                    checkpoint_size_mb=ck_mb,
                )
            )

    if 5 in include_levels and shared_cnn is not None:
        t_sec, ck_mb = _meta("L5_CNN")
        specs.append(
            AgentSpec(
                agent_id="L5_CNN",
                display_name="L5 CNN",
                level=5,
                mode=mode,
                factory=lambda seed, _cnn=shared_cnn: _SharedPolicyWrapper(
                    _cnn, name="L5_CNN", seed=seed
                ),
                config_summary="4-ch ResNet-4x64 Policy + Tactical Shortcut (no tree search)",
                budget_note="1 forward pass (~1.5 ms/move)",
                train_time_sec=t_sec,
                checkpoint_size_mb=ck_mb,
            )
        )

    if 6 in include_levels and shared_rl is not None:
        t_sec, ck_mb = _meta("L6_RL")
        specs.append(
            AgentSpec(
                agent_id="L6_RL",
                display_name="L6 RL-DQN",
                level=6,
                mode=mode,
                factory=lambda seed, _rl=shared_rl: _SharedRLWrapper(
                    _rl, name="L6_RL", seed=seed
                ),
                config_summary="4-ch ResGN-3x48 Masked Dueling DQN (Boltzmann-DQfD, eps=0)",
                budget_note="1 forward pass (~1.0 ms/move)",
                train_time_sec=t_sec,
                checkpoint_size_mb=ck_mb,
            )
        )

    if 7 in include_levels and shared_cnn is not None and shared_rl is not None:
        t_sec, ck_mb = _meta("L7_AlphaGo")
        if mode == "time_budget":
            specs.append(
                AgentSpec(
                    agent_id="L7_AlphaGo",
                    display_name="L7 AlphaGo",
                    level=7,
                    mode=mode,
                    factory=lambda seed, _cnn=shared_cnn, _rl=shared_rl, _tb=tb: AlphaGoAgent(
                        name="L7_AlphaGo",
                        policy_agent=_cnn,
                        value_agent=_rl,
                        auto_load_default=False,
                        num_simulations=2000,
                        time_limit_sec=_tb,
                        value_blend_alpha=0.25,
                        rl_prior_weight=0.35,
                        c_puct=1.6,
                        use_tactical_shortcut=True,
                        seed=seed,
                    ),
                    config_summary=f"Ensemble Prior (L5+L6) + PUCT + Value Blend 0.25 (time={tb_ms}ms)",
                    budget_note=f"Time budget = {tb_ms} ms/move",
                    train_time_sec=t_sec,
                    checkpoint_size_mb=ck_mb,
                )
            )
        else:
            specs.append(
                AgentSpec(
                    agent_id="L7_AlphaGo",
                    display_name="L7 AlphaGo",
                    level=7,
                    mode=mode,
                    factory=lambda seed, _cnn=shared_cnn, _rl=shared_rl: AlphaGoAgent(
                        name="L7_AlphaGo",
                        policy_agent=_cnn,
                        value_agent=_rl,
                        auto_load_default=False,
                        num_simulations=200,
                        time_limit_sec=0.5,
                        value_blend_alpha=0.25,
                        rl_prior_weight=0.35,
                        c_puct=1.6,
                        use_tactical_shortcut=True,
                        seed=seed,
                    ),
                    config_summary="Default Ensemble Prior (L5+L6) + PUCT (200 sims, cap=0.5s)",
                    budget_note="Default 200 sims / 0.5s cap",
                    train_time_sec=t_sec,
                    checkpoint_size_mb=ck_mb,
                )
            )

    for sp in specs:
        sp.descriptor = {
            "kind": "standard",
            "mode": mode,
            "time_budget_sec": tb,
            "agent_id": sp.agent_id,
            "level": sp.level,
        }

    return specs


# ---------------------------------------------------------------------------
# 3.5 Worker đa tiến trình (tương thích Windows spawn & Jupyter Notebook)
# ---------------------------------------------------------------------------

_WORKER_SPEC_CACHE: dict[tuple[str, float, str, int], AgentSpec] = {}
_WORKER_ABL3_MODELS: dict[str, dict[str, Any]] = {}


def _ensure_pythonpath(root: Path) -> None:
    """Đảm bảo thư mục gốc dự án nằm trong PYTHONPATH để tiến trình con (Windows spawn) từ notebooks/ luôn import được `src`."""
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if root_str not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root_str] + parts) if parts else root_str


def _worker_init() -> None:
    """Giới hạn mỗi worker dùng 1 luồng BLAS/PyTorch để tránh tranh chấp CPU khi chạy song song."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _create_worker_pool(max_workers: int) -> ProcessPoolExecutor:
    """Khởi tạo ProcessPoolExecutor bằng phương thức `spawn` để an toàn với PyTorch trên cả Linux và Windows."""
    return ProcessPoolExecutor(
        max_workers=int(max_workers),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
    )


def _build_agent_from_descriptor(desc: dict[str, Any], seed: int, project_root: str) -> Any:
    """Khởi tạo tác tử trong tiến trình worker từ đặc tả tuần tự hóa được (picklable)."""
    kind = desc["kind"]
    root = Path(project_root)

    if kind == "standard":
        mode = str(desc["mode"])
        tb = float(desc["time_budget_sec"])
        lv = int(desc["level"])
        key = (mode, tb, project_root, lv)
        if key not in _WORKER_SPEC_CACHE:
            built = build_agent_specs(
                mode=mode,
                time_budget_sec=tb,
                project_root=root,
                include_levels=(lv,),
            )
            for s in built:
                _WORKER_SPEC_CACHE[(mode, tb, project_root, s.level)] = s
        return _WORKER_SPEC_CACHE[key].factory(seed)

    if kind == "ablation_minimax_depth":
        d_int = int(desc["depth"])
        return MinimaxAgent(
            name=f"Minimax_d{d_int}",
            max_depth=d_int,
            use_alpha_beta=True,
            use_move_ordering=True,
            use_tt=True,
            max_branch_root=14 if d_int <= 3 else 10,
            max_branch_inner=10 if d_int <= 3 else 7,
            time_limit_sec=0.8,
            seed=seed,
            tie_break_eps=1.5,
        )

    if kind == "ablation_mcts":
        use_h = bool(desc["use_heuristic_leaf"])
        return MCTSAgent(
            num_rollouts=int(desc["num_rollouts"]),
            exploration_c=1.414,
            time_limit_sec=None,
            rollout_policy=str(desc["rollout_policy"]),
            use_heuristic_leaf=use_h,
            use_prior_ordering=use_h,
            use_progressive_widening=use_h,
            seed=seed,
            name=str(desc["agent_id"]),
        )

    if kind == "ablation_policy_variant":
        var_id = str(desc["variant_id"])
        if project_root not in _WORKER_ABL3_MODELS:
            shared_cnn, shared_rl = _load_shared_neural_models(root)
            cnn_path = root / "checkpoints" / "05_cnn_policy_best.pt"
            cnn_no_shortcut = CNNAgent(
                name="L5_PolicyOnly_NoShortcut",
                model_path=cnn_path,
                use_tactical_shortcut=False,
                use_mix_mcts=False,
                deterministic=True,
                seed=42,
            )
            cnn_plus_puct = CNNAgent(
                name="L5_Policy_Plus_PUCT",
                model_path=cnn_path,
                use_tactical_shortcut=True,
                use_mix_mcts=True,
                num_rollouts=150,
                time_limit_sec=0.5,
                seed=42,
            )
            _WORKER_ABL3_MODELS[project_root] = {
                "shared_cnn": shared_cnn,
                "shared_rl": shared_rl,
                "cnn_no_shortcut": cnn_no_shortcut,
                "cnn_plus_puct": cnn_plus_puct,
            }
        models = _WORKER_ABL3_MODELS[project_root]
        if var_id == "L5_PolicyOnly_NoShortcut":
            base = models["cnn_no_shortcut"]
            base.seed = seed
            if hasattr(base, "rng"):
                base.rng = random.Random(seed)
            return _SharedPolicyWrapper(base, name=var_id, seed=seed)
        if var_id == "L5_PolicyOnly_Shortcut":
            base = models["shared_cnn"]
            base.seed = seed
            if hasattr(base, "rng"):
                base.rng = random.Random(seed)
            return _SharedPolicyWrapper(base, name=var_id, seed=seed)
        if var_id == "L5_Policy_Plus_PUCT":
            base = models["cnn_plus_puct"]
            base.seed = seed
            if hasattr(base, "rng"):
                base.rng = random.Random(seed)
            return _SharedPolicyWrapper(base, name=var_id, seed=seed)
        if var_id == "L7_AlphaGo_EnsemblePUCT":
            return AlphaGoAgent(
                name=var_id,
                policy_agent=models["shared_cnn"],
                value_agent=models["shared_rl"],
                auto_load_default=False,
                num_simulations=200,
                time_limit_sec=0.5,
                value_blend_alpha=0.25,
                rl_prior_weight=0.35,
                seed=seed,
            )

    raise ValueError(f"Không nhận diện được descriptor: {desc!r}")


def _play_game_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Hàm top-level cho ProcessPoolExecutor chơi trọn 1 ván đấu và trả về log dict."""
    agent_a = _build_agent_from_descriptor(task["desc_a"], int(task["game_seed_a"]), str(task["project_root"]))
    agent_b = _build_agent_from_descriptor(task["desc_b"], int(task["game_seed_b"]), str(task["project_root"]))
    return play_single_game(
        agent_a=agent_a,
        agent_b=agent_b,
        agent_a_id=str(task["agent_a_id"]),
        agent_b_id=str(task["agent_b_id"]),
        a_side=int(task["a_side"]),
        opening_moves=task["opening_moves"],
        opening_index=int(task["opening_index"]),
        seed=int(task["seed"]),
        mode=str(task["mode"]),
        game_index=int(task["game_index"]),
    )


# ---------------------------------------------------------------------------
# 4. Chơi 1 ván & Chạy cặp đấu với lưu log JSONL riêng từng cặp + Resume
# ---------------------------------------------------------------------------

def play_single_game(
    agent_a: Any,
    agent_b: Any,
    agent_a_id: str,
    agent_b_id: str,
    a_side: int,
    opening_moves: Sequence[int] = (),
    opening_index: int = 0,
    seed: int = 42,
    mode: str = "default_config",
    game_index: int = 1,
    board_size: int = 11,
    win_length: int = 5,
) -> dict[str, Any]:
    """Chơi trọn vẹn 1 ván Caro 11x11 và trả về bản ghi log thô đầy đủ."""
    if a_side not in (1, -1):
        raise ValueError("a_side phải là 1 (X) hoặc -1 (O).")

    env = CaroEnv(board_size=board_size, win_length=win_length)
    opening_list = [int(m) for m in opening_moves]
    for m in opening_list:
        env.step(m)

    times_a: list[float] = []
    times_b: list[float] = []
    searches_a: list[float] = []
    searches_b: list[float] = []

    t0 = time.perf_counter()
    while not env.is_terminal():
        is_a_turn = env.current_player == a_side
        actor = agent_a if is_a_turn else agent_b
        mv_t0 = time.perf_counter()
        action = int(actor.select_action(env))
        wall_ms = (time.perf_counter() - mv_t0) * 1000.0
        reported_ms = float(getattr(actor, "last_move_time_ms", 0.0))
        dt_ms = reported_ms if reported_ms > 0.0 else wall_ms
        sc = float(getattr(actor, "last_search_count", 0.0))
        if is_a_turn:
            times_a.append(dt_ms)
            searches_a.append(sc)
        else:
            times_b.append(dt_ms)
            searches_b.append(sc)
        env.step(action)

    duration_sec = time.perf_counter() - t0
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
        "mode": mode,
        "matchup": f"{agent_a_id}_vs_{agent_b_id}",
        "game_index": int(game_index),
        "agent_a": agent_a_id,
        "agent_b": agent_b_id,
        "seed": int(seed),
        "opening_index": int(opening_index),
        "opening_type": "standard" if len(opening_list) == 0 else "diverse",
        "opening_moves": opening_list,
        "a_side": "X" if a_side == 1 else "O",
        "x_agent": x_agent,
        "o_agent": o_agent,
        "winner_side": winner_side,
        "winner_agent": winner_agent,
        "result_a": result_a,
        "outcome_a": outcome_a,
        "num_moves": len(env.history),
        "duration_sec": round(duration_sec, 4),
        "moves_a": len(times_a),
        "moves_b": len(times_b),
        "avg_ms_a": round(statistics.mean(times_a), 3) if times_a else 0.0,
        "p95_ms_a": round(percentile_p95(times_a), 3),
        "avg_search_a": round(statistics.mean(searches_a), 2) if searches_a else 0.0,
        "avg_ms_b": round(statistics.mean(times_b), 3) if times_b else 0.0,
        "p95_ms_b": round(percentile_p95(times_b), 3),
        "avg_search_b": round(statistics.mean(searches_b), 2) if searches_b else 0.0,
        "move_times_ms_a": [round(x, 3) for x in times_a],
        "move_times_ms_b": [round(x, 3) for x in times_b],
        "history": [int(m) for m in env.history],
    }


def _load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_pair_logs(
    records: Sequence[dict[str, Any]],
    agent_a_id: str,
    agent_b_id: str,
    mode: str,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Tổng hợp đầy đủ thống kê Mục 3.5 từ danh sách log thô của 1 cặp đấu."""
    n = len(records)
    if n == 0:
        raise ValueError(f"Không có ván đấu nào để tổng hợp cho {agent_a_id} vs {agent_b_id}")

    scores_a = [float(r["result_a"]) for r in records]
    wins_a = sum(1 for s in scores_a if s == 1.0)
    draws = sum(1 for s in scores_a if s == 0.5)
    losses_a = sum(1 for s in scores_a if s == 0.0)
    pts_a = wins_a + 0.5 * draws
    score_pct = 100.0 * pts_a / n
    w_lo, w_hi = wilson_ci95(pts_a, n)
    b_lo, b_hi = bootstrap_ci95(scores_a, n_boot=bootstrap_samples, seed=42)

    # Phân tách lượt đi trước (X) / đi sau (O)
    x_games = [r for r in records if r["a_side"] == "X"]
    o_games = [r for r in records if r["a_side"] == "O"]
    score_as_x = (
        100.0 * sum(float(r["result_a"]) for r in x_games) / len(x_games) if x_games else 0.0
    )
    score_as_o = (
        100.0 * sum(float(r["result_a"]) for r in o_games) / len(o_games) if o_games else 0.0
    )
    x_wins_overall = sum(1 for r in records if r["winner_side"] == "X")
    o_wins_overall = sum(1 for r in records if r["winner_side"] == "O")

    # Độ trễ chính xác trên toàn bộ từng nước đi (nếu có move_times_ms_a/b, fallback về avg_ms_a/b)
    all_t_a: list[float] = []
    all_t_b: list[float] = []
    for r in records:
        if "move_times_ms_a" in r and r["move_times_ms_a"]:
            all_t_a.extend(float(x) for x in r["move_times_ms_a"])
        elif r.get("moves_a", 0) > 0:
            all_t_a.append(float(r["avg_ms_a"]))
        if "move_times_ms_b" in r and r["move_times_ms_b"]:
            all_t_b.extend(float(x) for x in r["move_times_ms_b"])
        elif r.get("moves_b", 0) > 0:
            all_t_b.append(float(r["avg_ms_b"]))

    # Độ lệch chuẩn điểm giữa các seed khác nhau
    by_seed: dict[int, list[float]] = {}
    for r in records:
        by_seed.setdefault(int(r["seed"]), []).append(float(r["result_a"]))
    seed_scores = [100.0 * sum(v) / len(v) for v in by_seed.values() if v]
    seed_std_pp = float(statistics.stdev(seed_scores)) if len(seed_scores) >= 2 else 0.0

    return {
        "mode": mode,
        "matchup": f"{agent_a_id}_vs_{agent_b_id}",
        "agent_a": agent_a_id,
        "agent_b": agent_b_id,
        "games": n,
        "wins_a": wins_a,
        "draws": draws,
        "losses_a": losses_a,
        "win_rate_a_pct": round(100.0 * wins_a / n, 2),
        "draw_rate_pct": round(100.0 * draws / n, 2),
        "loss_rate_a_pct": round(100.0 * losses_a / n, 2),
        "score_a_pct": round(score_pct, 2),
        "wilson_ci95_low": w_lo,
        "wilson_ci95_high": w_hi,
        "bootstrap_ci95_low": b_lo,
        "bootstrap_ci95_high": b_hi,
        "score_a_as_X_pct": round(score_as_x, 2),
        "score_a_as_O_pct": round(score_as_o, 2),
        "x_win_overall_pct": round(100.0 * x_wins_overall / n, 2),
        "o_win_overall_pct": round(100.0 * o_wins_overall / n, 2),
        "seed_count": len(by_seed),
        "seed_std_pp": round(seed_std_pp, 2),
        "avg_moves": round(statistics.mean(float(r["num_moves"]) for r in records), 2),
        "avg_ms_a": round(statistics.mean(all_t_a), 3) if all_t_a else 0.0,
        "p95_ms_a": round(percentile_p95(all_t_a), 3),
        "avg_search_a": round(statistics.mean(float(r["avg_search_a"]) for r in records), 2),
        "avg_ms_b": round(statistics.mean(all_t_b), 3) if all_t_b else 0.0,
        "p95_ms_b": round(percentile_p95(all_t_b), 3),
        "avg_search_b": round(statistics.mean(float(r["avg_search_b"]) for r in records), 2),
        "total_duration_sec": round(sum(float(r.get("duration_sec", 0.0)) for r in records), 2),
    }


def run_pair_match(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    games_per_pair: int = 100,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    opening_book: Sequence[Sequence[int]] = OPENING_BOOK_11X11,
    log_path: Path | None = None,
    resume: bool = True,
    bootstrap_samples: int = 2000,
    verbose: bool = True,
    max_workers: int = 1,
    executor: ProcessPoolExecutor | None = None,
    project_root: str | Path | None = None,
    pair_label: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Chạy `games_per_pair` ván đối kháng cân bằng 50/50 X-O giữa 2 tác tử và lưu `.jsonl` riêng.

    Hỗ trợ chạy song song đa tiến trình (`max_workers > 1` hoặc `executor` dùng chung)
    và khôi phục an toàn theo `game_index` (`resume=True`).
    """
    if games_per_pair <= 0 or games_per_pair % 2 != 0:
        raise ValueError("games_per_pair phải là số nguyên dương chẵn để chia đều 50/50 X và O.")
    if not seeds:
        raise ValueError("seeds không được rỗng.")

    records_by_idx: dict[int, dict[str, Any]] = {}
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and log_path.exists():
            for r in _load_existing_jsonl(log_path):
                gi = int(r.get("game_index", 0))
                if 1 <= gi <= games_per_pair and gi not in records_by_idx:
                    records_by_idx[gi] = r
        elif not resume and log_path.exists():
            log_path.unlink()

    existing_count = len(records_by_idx)
    pending_g = [g for g in range(games_per_pair) if (g + 1) not in records_by_idx]
    prefix = f"{pair_label} " if pair_label else ""

    if verbose and existing_count > 0:
        print(
            f"  {prefix}[Resume] {spec_a.agent_id} vs {spec_b.agent_id}: "
            f"đã có {existing_count}/{games_per_pair} ván (còn {len(pending_g)} ván).",
            flush=True,
        )

    n_seeds = len(seeds)
    n_openings = max(1, len(opening_book))
    root_str = str(Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1])

    def _make_task(g: int) -> dict[str, Any]:
        pair_k = g // 2
        a_side = 1 if (g % 2 == 0) else -1
        seed_val = int(seeds[pair_k % n_seeds])
        game_seed_a = seed_val * 1000 + pair_k * 2 + (0 if a_side == 1 else 1)
        game_seed_b = seed_val * 1000 + pair_k * 2 + (1 if a_side == 1 else 0) + 500
        opening_idx = pair_k % n_openings
        opening = tuple(int(x) for x in opening_book[opening_idx]) if opening_book else ()
        return {
            "g": g,
            "game_index": g + 1,
            "a_side": a_side,
            "seed": seed_val,
            "game_seed_a": game_seed_a,
            "game_seed_b": game_seed_b,
            "opening_index": opening_idx,
            "opening_moves": opening,
            "agent_a_id": spec_a.agent_id,
            "agent_b_id": spec_b.agent_id,
            "mode": spec_a.mode,
            "desc_a": spec_a.descriptor,
            "desc_b": spec_b.descriptor,
            "project_root": root_str,
        }

    can_parallel = (
        len(pending_g) > 1
        and spec_a.descriptor is not None
        and spec_b.descriptor is not None
        and (executor is not None or int(max_workers) > 1)
    )

    if pending_g:
        file_handle = log_path.open("a", encoding="utf-8") if log_path is not None else None
        try:
            if can_parallel:
                tasks = [_make_task(g) for g in pending_g]
                own_pool: ProcessPoolExecutor | None = None
                pool = executor
                if pool is None:
                    _ensure_pythonpath(Path(root_str))
                    own_pool = _create_worker_pool(min(int(max_workers), len(tasks)))
                    pool = own_pool
                try:
                    futures = [pool.submit(_play_game_worker, t) for t in tasks]
                    for fut in as_completed(futures):
                        rec = fut.result()
                        records_by_idx[int(rec["game_index"])] = rec
                        if file_handle is not None:
                            file_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            file_handle.flush()
                finally:
                    if own_pool is not None:
                        own_pool.shutdown(wait=True)
            else:
                for g in pending_g:
                    t = _make_task(g)
                    agent_a = spec_a.factory(t["game_seed_a"])
                    agent_b = spec_b.factory(t["game_seed_b"])
                    rec = play_single_game(
                        agent_a=agent_a,
                        agent_b=agent_b,
                        agent_a_id=spec_a.agent_id,
                        agent_b_id=spec_b.agent_id,
                        a_side=t["a_side"],
                        opening_moves=t["opening_moves"],
                        opening_index=t["opening_index"],
                        seed=t["seed"],
                        mode=spec_a.mode,
                        game_index=t["game_index"],
                    )
                    records_by_idx[int(rec["game_index"])] = rec
                    if file_handle is not None:
                        file_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        file_handle.flush()
        finally:
            if file_handle is not None:
                file_handle.close()

    records = [records_by_idx[idx] for idx in sorted(records_by_idx)]
    summary = summarize_pair_logs(
        records=records,
        agent_a_id=spec_a.agent_id,
        agent_b_id=spec_b.agent_id,
        mode=spec_a.mode,
        bootstrap_samples=bootstrap_samples,
    )
    if verbose:
        print(
            f"  {prefix}[Done] {spec_a.agent_id:12s} vs {spec_b.agent_id:12s} ({games_per_pair:3d}g) | "
            f"Score={summary['score_a_pct']:5.1f}% "
            f"(W{summary['wins_a']}-D{summary['draws']}-L{summary['losses_a']}) | "
            f"CI95=[{summary['wilson_ci95_low']:.1f}, {summary['wilson_ci95_high']:.1f}] | "
            f"ms={summary['avg_ms_a']:.1f}/{summary['avg_ms_b']:.1f}",
            flush=True,
        )
    return summary, records


# ---------------------------------------------------------------------------
# 5. Giải đấu Round-Robin toàn diện (lưu log từng cặp, ma trận, bảng xếp hạng)
# ---------------------------------------------------------------------------

def run_round_robin(
    mode: str = "default_config",
    games_per_pair: int = 100,
    time_budget_sec: float = 0.25,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    include_levels: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
    project_root: str | Path | None = None,
    output_logs_dir: str | Path | None = None,
    output_tables_dir: str | Path | None = None,
    resume: bool = True,
    bootstrap_samples: int = 2000,
    verbose: bool = True,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Chạy giải đấu Round-Robin đầy đủ cho chế độ `mode`, lưu toàn bộ log từng cặp & CSV."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    logs_dir = (
        Path(output_logs_dir)
        if output_logs_dir is not None
        else (root / "results" / "logs" / "tournament" / mode)
    )
    tables_dir = (
        Path(output_tables_dir)
        if output_tables_dir is not None
        else (root / "results" / "tables" / "tournament" / mode)
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    hw_info = collect_hardware_info()
    specs = build_agent_specs(
        mode=mode,
        time_budget_sec=time_budget_sec,
        project_root=root,
        include_levels=include_levels,
    )
    spec_map = {s.agent_id: s for s in specs}
    agent_ids = [s.agent_id for s in specs]
    pairs = list(combinations(specs, 2))
    total_pairs = len(pairs)
    workers = max(1, int(max_workers))

    if verbose:
        print(
            f"\n=== ROUND-ROBIN TOURNAMENT [{mode}] | {len(specs)} agents | "
            f"{total_pairs} pairs x {games_per_pair} games | workers={workers} ===",
            flush=True,
        )

    matchup_rows: list[dict[str, Any]] = []
    all_games: list[dict[str, Any]] = []
    pair_log_files: dict[str, str] = {}
    peak_rss_mb = get_process_rss_mb()

    # Kiểm tra trước xem còn ván nào chưa hoàn thành để chỉ khởi tạo pool khi cần
    needs_work = False
    if workers > 1:
        for spec_a, spec_b in pairs:
            pf = logs_dir / f"{spec_a.agent_id}_vs_{spec_b.agent_id}.jsonl"
            if not (resume and pf.exists() and len(_load_existing_jsonl(pf)) >= games_per_pair):
                needs_work = True
                break

    if workers > 1 and needs_work:
        _ensure_pythonpath(root)
    shared_pool: ProcessPoolExecutor | None = (
        _create_worker_pool(workers)
        if (workers > 1 and needs_work)
        else None
    )
    try:
        for pair_idx, (spec_a, spec_b) in enumerate(pairs, start=1):
            pair_file = logs_dir / f"{spec_a.agent_id}_vs_{spec_b.agent_id}.jsonl"
            pair_label = f"[{pair_idx:2d}/{total_pairs}]"
            summary, recs = run_pair_match(
                spec_a=spec_a,
                spec_b=spec_b,
                games_per_pair=games_per_pair,
                seeds=seeds,
                opening_book=OPENING_BOOK_11X11,
                log_path=pair_file,
                resume=resume,
                bootstrap_samples=bootstrap_samples,
                verbose=verbose,
                max_workers=workers,
                executor=shared_pool,
                project_root=root,
                pair_label=pair_label,
            )
            summary["log_file"] = str(pair_file.relative_to(root)) if pair_file.is_relative_to(root) else str(pair_file)
            matchup_rows.append(summary)
            all_games.extend(recs)
            pair_log_files[summary["matchup"]] = summary["log_file"]
            peak_rss_mb = max(peak_rss_mb, get_process_rss_mb())
            if verbose:
                elapsed = time.perf_counter() - t0
                eta = (elapsed / pair_idx) * (total_pairs - pair_idx)
                print(
                    f"         -> Tiến độ [{mode}]: {pair_idx}/{total_pairs} cặp | "
                    f"Đã chạy {elapsed:.1f}s | Ước tính còn ~{eta:.0f}s",
                    flush=True,
                )
    finally:
        if shared_pool is not None:
            shared_pool.shutdown(wait=True)

    # 5.1 Ma trận đối đầu Score (%) & CI95
    score_mat = pd.DataFrame(np.nan, index=agent_ids, columns=agent_ids, dtype=float)
    ci_mat = pd.DataFrame("--", index=agent_ids, columns=agent_ids, dtype=object)
    for r in matchup_rows:
        a, b = r["agent_a"], r["agent_b"]
        sa = float(r["score_a_pct"])
        sb = round(100.0 - sa, 2)
        score_mat.loc[a, b] = sa
        score_mat.loc[b, a] = sb
        ci_mat.loc[a, b] = f"{sa:.1f}% [{r['wilson_ci95_low']:.1f}, {r['wilson_ci95_high']:.1f}]"
        # Wilson CI của B là đối xứng [100 - hi, 100 - lo]
        ci_mat.loc[b, a] = (
            f"{sb:.1f}% [{round(100.0 - r['wilson_ci95_high'], 1):.1f}, "
            f"{round(100.0 - r['wilson_ci95_low'], 1):.1f}]"
        )

    # 5.2 Bảng xếp hạng tổng hợp (Leaderboard) & First-Player Advantage & Seed Sensitivity
    elos = compute_elo_ratings(matchup_rows, agent_ids)
    gpu_info = get_gpu_vram_info()

    leaderboard_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for aid in agent_ids:
        scores_01: list[float] = []
        scores_x: list[float] = []
        scores_o: list[float] = []
        move_times: list[float] = []
        search_counts: list[float] = []
        by_seed: dict[int, list[float]] = {}
        wins = draws = losses = 0

        for g in all_games:
            if g["agent_a"] == aid:
                s = float(g["result_a"])
                side = g["a_side"]
                mts = g.get("move_times_ms_a") or ([float(g["avg_ms_a"])] if g.get("moves_a", 0) else [])
                sc = float(g["avg_search_a"])
            elif g["agent_b"] == aid:
                s = 1.0 - float(g["result_a"])
                side = "O" if g["a_side"] == "X" else "X"
                mts = g.get("move_times_ms_b") or ([float(g["avg_ms_b"])] if g.get("moves_b", 0) else [])
                sc = float(g["avg_search_b"])
            else:
                continue

            scores_01.append(s)
            if s == 1.0:
                wins += 1
            elif s == 0.5:
                draws += 1
            else:
                losses += 1

            if side == "X":
                scores_x.append(s)
            else:
                scores_o.append(s)
            move_times.extend(float(x) for x in mts)
            search_counts.append(sc)
            by_seed.setdefault(int(g["seed"]), []).append(s)

        total_g = len(scores_01)
        pts = wins + 0.5 * draws
        score_pct = 100.0 * pts / max(1, total_g)
        w_lo, w_hi = wilson_ci95(pts, total_g)
        b_lo, b_hi = bootstrap_ci95(scores_01, n_boot=bootstrap_samples, seed=42)
        sx = 100.0 * sum(scores_x) / max(1, len(scores_x))
        so = 100.0 * sum(scores_o) / max(1, len(scores_o))

        seed_pcts = [100.0 * sum(v) / len(v) for v in by_seed.values() if v]
        seed_std = float(statistics.stdev(seed_pcts)) if len(seed_pcts) >= 2 else 0.0
        for sv, vals in sorted(by_seed.items()):
            seed_rows.append({
                "mode": mode,
                "agent_id": aid,
                "seed": sv,
                "games": len(vals),
                "score_pct": round(100.0 * sum(vals) / len(vals), 2),
            })

        sp = spec_map[aid]
        leaderboard_rows.append({
            "mode": mode,
            "level": sp.level,
            "agent_id": aid,
            "display_name": sp.display_name,
            "games": total_g,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate_pct": round(100.0 * wins / max(1, total_g), 2),
            "draw_rate_pct": round(100.0 * draws / max(1, total_g), 2),
            "loss_rate_pct": round(100.0 * losses / max(1, total_g), 2),
            "score_pct": round(score_pct, 2),
            "wilson_ci95_low": w_lo,
            "wilson_ci95_high": w_hi,
            "bootstrap_ci95_low": b_lo,
            "bootstrap_ci95_high": b_hi,
            "elo_rating": elos.get(aid, 1500.0),
            "score_as_X_pct": round(sx, 2),
            "score_as_O_pct": round(so, 2),
            "first_player_delta_pp": round(sx - so, 2),
            "seed_std_pp": round(seed_std, 2),
            "avg_ms": round(statistics.mean(move_times), 3) if move_times else 0.0,
            "p95_ms": round(percentile_p95(move_times), 3),
            "avg_search": round(statistics.mean(search_counts), 2) if search_counts else 0.0,
            "peak_rss_mb": round(peak_rss_mb, 1),
            "vram_mb": gpu_info["vram_allocated_mb"],
            "checkpoint_mb": sp.checkpoint_size_mb,
            "train_time_sec": sp.train_time_sec,
            "budget_note": sp.budget_note,
            "config_summary": sp.config_summary,
        })
        fp_rows.append({
            "mode": mode,
            "agent_id": aid,
            "games_as_X": len(scores_x),
            "score_as_X_pct": round(sx, 2),
            "games_as_O": len(scores_o),
            "score_as_O_pct": round(so, 2),
            "delta_X_minus_O_pp": round(sx - so, 2),
        })

    # Dòng tổng thể First-Player Advantage trên toàn bộ giải đấu
    total_all = len(all_games)
    all_x_wins = sum(1 for g in all_games if g["winner_side"] == "X")
    all_o_wins = sum(1 for g in all_games if g["winner_side"] == "O")
    all_draws = sum(1 for g in all_games if g["winner_side"] == "Draw")
    x_pts = all_x_wins + 0.5 * all_draws
    x_ci_lo, x_ci_hi = wilson_ci95(x_pts, total_all)
    fp_rows.append({
        "mode": mode,
        "agent_id": "ALL_GAMES_OVERALL",
        "games_as_X": total_all,
        "score_as_X_pct": round(100.0 * x_pts / max(1, total_all), 2),
        "games_as_O": total_all,
        "score_as_O_pct": round(100.0 * (all_o_wins + 0.5 * all_draws) / max(1, total_all), 2),
        "delta_X_minus_O_pp": round(100.0 * (all_x_wins - all_o_wins) / max(1, total_all), 2),
    })

    df_matchups = pd.DataFrame(matchup_rows)
    df_leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        by=["score_pct", "elo_rating"], ascending=[False, False]
    ).reset_index(drop=True)
    df_leaderboard.insert(0, "rank", range(1, len(df_leaderboard) + 1))
    df_fp = pd.DataFrame(fp_rows)
    df_seeds = pd.DataFrame(seed_rows)

    # Lưu toàn bộ CSV & metadata JSON
    df_matchups.to_csv(tables_dir / "matchup_summary.csv", index=False, encoding="utf-8-sig")
    score_mat.to_csv(tables_dir / "score_matrix.csv", encoding="utf-8-sig")
    ci_mat.to_csv(tables_dir / "ci95_matrix.csv", encoding="utf-8-sig")
    df_leaderboard.to_csv(tables_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    df_fp.to_csv(tables_dir / "first_player_analysis.csv", index=False, encoding="utf-8-sig")
    df_seeds.to_csv(tables_dir / "seed_sensitivity.csv", index=False, encoding="utf-8-sig")

    elapsed_total = round(time.perf_counter() - t0, 2)
    metadata = {
        "mode": mode,
        "games_per_pair": int(games_per_pair),
        "total_pairs": len(matchup_rows),
        "total_games": total_all,
        "time_budget_sec": float(time_budget_sec) if mode == "time_budget" else None,
        "seeds": [int(s) for s in seeds],
        "opening_book_size": len(OPENING_BOOK_11X11),
        "max_workers": workers,
        "elapsed_sec": elapsed_total,
        "peak_rss_mb": round(peak_rss_mb, 2),
        "overall_first_player": {
            "x_wins": all_x_wins,
            "o_wins": all_o_wins,
            "draws": all_draws,
            "x_score_pct": round(100.0 * x_pts / max(1, total_all), 2),
            "x_score_wilson_ci95": [x_ci_lo, x_ci_hi],
            "x_minus_o_win_pp": round(100.0 * (all_x_wins - all_o_wins) / max(1, total_all), 2),
        },
        "game_spec": GAME_SPEC,
        "hardware": hw_info,
        "agents": [
            {
                "agent_id": s.agent_id,
                "level": s.level,
                "config_summary": s.config_summary,
                "budget_note": s.budget_note,
                "checkpoint_mb": s.checkpoint_size_mb,
                "train_time_sec": s.train_time_sec,
            }
            for s in specs
        ],
        "pair_log_files": pair_log_files,
    }
    with (tables_dir / "tournament_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if verbose:
        print(
            f"=== Finished [{mode}] in {elapsed_total:.1f}s | Peak RSS={peak_rss_mb:.1f} MB | "
            f"Logs -> {logs_dir} ===\n",
            flush=True,
        )

    return {
        "metadata": metadata,
        "matchups": df_matchups,
        "score_matrix": score_mat,
        "ci95_matrix": ci_mat,
        "leaderboard": df_leaderboard,
        "first_player": df_fp,
        "seed_sensitivity": df_seeds,
        "logs_dir": logs_dir,
        "tables_dir": tables_dir,
    }


# ---------------------------------------------------------------------------
# 6. Bộ 03 Thí nghiệm Ablation / Sensitivity (Bắt buộc Mục 3.5 & sách Mark Liu)
# ---------------------------------------------------------------------------

def run_ablation_alphabeta_depth(
    games_per_depth: int = 10,
    depths: Sequence[int] = (2, 3, 4),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    project_root: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    max_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    """Ablation 1: Alpha-Beta Bật/Tắt (Mark Liu Ch. 6) & Độ nhạy Search Depth d=2..4."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    logs_dir = root / "results" / "logs" / "tournament" / "ablations"
    tables_dir = root / "results" / "tables" / "tournament" / "ablations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Phần A: So sánh trực tiếp Alpha-Beta ON vs OFF trên cùng 5 thế cờ trung cuộc (d=2)
    # để kiểm chứng tính tương đương chính xác + tỷ lệ cắt tỉa nút & tăng tốc
    test_positions = [
        ("Midgame-Center", [60, 49, 61, 48, 59, 71]),
        ("Midgame-Diagonal", [60, 48, 72, 49, 50, 70]),
        ("Open3-Threat", [60, 25, 61, 26, 70, 27, 81, 110]),
        ("Fork-Setup", [60, 49, 61, 50, 71, 48, 70, 38]),
        ("Dense-Cluster", [60, 49, 61, 59, 71, 48, 72, 50, 82, 70]),
    ]
    pruning_rows: list[dict[str, Any]] = []
    for pos_name, moves in test_positions:
        env = CaroEnv()
        for m in moves:
            env.step(m)
        mm_off = MinimaxAgent(
            name="Minimax_NoAB",
            max_depth=2,
            use_alpha_beta=False,
            use_move_ordering=True,
            use_tt=False,
            time_limit_sec=None,
            max_branch_root=12,
            max_branch_inner=10,
            use_tactical_shortcut=False,
        )
        mm_on = MinimaxAgent(
            name="Minimax_AB_TT",
            max_depth=2,
            use_alpha_beta=True,
            use_move_ordering=True,
            use_tt=True,
            time_limit_sec=None,
            max_branch_root=12,
            max_branch_inner=10,
            use_tactical_shortcut=False,
        )
        act_off = mm_off.select_action(env)
        nodes_off = int(mm_off.last_search_count)
        ms_off = float(mm_off.last_move_time_ms)

        act_on = mm_on.select_action(env)
        nodes_on = int(mm_on.last_search_count)
        ms_on = float(mm_on.last_move_time_ms)

        pruning_rows.append({
            "position": pos_name,
            "depth": 2,
            "action_no_ab": act_off,
            "action_ab_tt": act_on,
            "exact_same_move": bool(act_off == act_on),
            "nodes_no_ab": nodes_off,
            "nodes_ab_tt": nodes_on,
            "node_reduction_pct": round(100.0 * (1.0 - nodes_on / max(1, nodes_off)), 2),
            "ms_no_ab": round(ms_off, 2),
            "ms_ab_tt": round(ms_on, 2),
            "speedup_x": round(ms_off / max(1e-6, ms_on), 2),
        })

    # Phần B: Độ nhạy Search Depth d ∈ depths đối đầu L2_RuleBased
    opp_spec = build_agent_specs(mode="default_config", project_root=root, include_levels=(2,))[0]
    depth_rows: list[dict[str, Any]] = []
    workers = max(1, int(max_workers))
    if workers > 1:
        _ensure_pythonpath(root)
    pool: ProcessPoolExecutor | None = (
        _create_worker_pool(workers) if workers > 1 else None
    )
    try:
        for d in depths:
            d_int = int(d)
            mm_spec = AgentSpec(
                agent_id=f"Minimax_d{d_int}",
                display_name=f"Minimax d={d_int}",
                level=3,
                mode="ablation",
                factory=lambda seed, _d=d_int: MinimaxAgent(
                    name=f"Minimax_d{_d}",
                    max_depth=_d,
                    use_alpha_beta=True,
                    use_move_ordering=True,
                    use_tt=True,
                    max_branch_root=14 if _d <= 3 else 10,
                    max_branch_inner=10 if _d <= 3 else 7,
                    time_limit_sec=0.8,
                    seed=seed,
                    tie_break_eps=1.5,
                ),
                config_summary=f"Alpha-Beta+TT depth={d_int}",
                budget_note=f"depth={d_int}",
                descriptor={"kind": "ablation_minimax_depth", "depth": d_int},
            )
            log_file = logs_dir / f"ablation1_minimax_d{d_int}_vs_L2_RuleBased.jsonl"
            summ, _ = run_pair_match(
                spec_a=mm_spec,
                spec_b=opp_spec,
                games_per_pair=games_per_depth,
                seeds=seeds,
                log_path=log_file,
                resume=resume,
                verbose=verbose,
                max_workers=workers,
                executor=pool,
                project_root=root,
            )
            summ["search_depth"] = d_int
            depth_rows.append(summ)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    df_pruning = pd.DataFrame(pruning_rows)
    df_depth = pd.DataFrame(depth_rows)
    df_pruning.to_csv(tables_dir / "ablation1_alphabeta_pruning.csv", index=False, encoding="utf-8-sig")
    df_depth.to_csv(tables_dir / "ablation1_depth_sensitivity.csv", index=False, encoding="utf-8-sig")
    return {"pruning": df_pruning, "depth": df_depth}


def run_ablation_mcts_budget_heuristic(
    games_per_config: int = 10,
    rollout_budgets: Sequence[int] = (50, 150, 300, 600),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    project_root: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Ablation 2: MCTS Simulation Budget & Heuristic Playout ON vs OFF (Mark Liu Ch. 8)."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    logs_dir = root / "results" / "logs" / "tournament" / "ablations"
    tables_dir = root / "results" / "tables" / "tournament" / "ablations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    opp_spec = build_agent_specs(mode="default_config", project_root=root, include_levels=(2,))[0]
    rows: list[dict[str, Any]] = []

    # 1. Bật/tắt Heuristic trong MCTS tại cùng ngân sách 300 rollouts (Plain UCT vs Heavy Playout)
    configs: list[tuple[str, int, str, bool]] = [
        ("MCTS_RandomPlayout_NoHeur_300", 300, "random", False),
    ]
    for r_budget in rollout_budgets:
        configs.append((f"MCTS_Heuristic_r{int(r_budget)}", int(r_budget), "heuristic", True))

    workers = max(1, int(max_workers))
    if workers > 1:
        _ensure_pythonpath(root)
    pool: ProcessPoolExecutor | None = (
        _create_worker_pool(workers) if workers > 1 else None
    )
    try:
        for cfg_id, n_roll, r_pol, use_h_leaf in configs:
            mcts_spec = AgentSpec(
                agent_id=cfg_id,
                display_name=cfg_id,
                level=4,
                mode="ablation",
                factory=lambda seed, _n=n_roll, _rp=r_pol, _hl=use_h_leaf, _id=cfg_id: MCTSAgent(
                    num_rollouts=_n,
                    exploration_c=1.414,
                    time_limit_sec=None,
                    rollout_policy=_rp,
                    use_heuristic_leaf=_hl,
                    use_prior_ordering=_hl,
                    use_progressive_widening=_hl,
                    seed=seed,
                    name=_id,
                ),
                config_summary=f"rollouts={n_roll}, rollout_policy={r_pol}, heur_leaf={use_h_leaf}",
                budget_note=f"{n_roll} rollouts",
                descriptor={
                    "kind": "ablation_mcts",
                    "agent_id": cfg_id,
                    "num_rollouts": n_roll,
                    "rollout_policy": r_pol,
                    "use_heuristic_leaf": use_h_leaf,
                },
            )
            log_file = logs_dir / f"ablation2_{cfg_id}_vs_L2_RuleBased.jsonl"
            summ, _ = run_pair_match(
                spec_a=mcts_spec,
                spec_b=opp_spec,
                games_per_pair=games_per_config,
                seeds=seeds,
                log_path=log_file,
                resume=resume,
                verbose=verbose,
                max_workers=workers,
                executor=pool,
                project_root=root,
            )
            summ["rollouts_budget"] = n_roll
            summ["heuristic_enabled"] = bool(use_h_leaf)
            rows.append(summ)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "ablation2_mcts_budget_heuristic.csv", index=False, encoding="utf-8-sig")
    return df


def run_ablation_policy_vs_search(
    games_per_config: int = 10,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    project_root: str | Path | None = None,
    resume: bool = True,
    verbose: bool = True,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Ablation 3: Policy-Only (CNN Cấp 5) vs Policy+PUCT (Cấp 5 + MCTS) vs AlphaGo Cấp 7 (Mark Liu Ch. 10–18)."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    logs_dir = root / "results" / "logs" / "tournament" / "ablations"
    tables_dir = root / "results" / "tables" / "tournament" / "ablations"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    shared_cnn, shared_rl = _load_shared_neural_models(root)
    cnn_path = root / "checkpoints" / "05_cnn_policy_best.pt"
    opp_specs = build_agent_specs(mode="default_config", project_root=root, include_levels=(2, 4))

    cnn_no_shortcut = CNNAgent(
        name="L5_PolicyOnly_NoShortcut",
        model_path=cnn_path,
        use_tactical_shortcut=False,
        use_mix_mcts=False,
        deterministic=True,
        seed=42,
    )
    cnn_plus_puct = CNNAgent(
        name="L5_Policy_Plus_PUCT",
        model_path=cnn_path,
        use_tactical_shortcut=True,
        use_mix_mcts=True,
        num_rollouts=150,
        time_limit_sec=0.5,
        seed=42,
    )

    def _reseed_policy_wrapper(base_cnn: CNNAgent, name: str, seed: int) -> _SharedPolicyWrapper:
        base_cnn.seed = seed
        if hasattr(base_cnn, "rng"):
            base_cnn.rng = random.Random(seed)
        return _SharedPolicyWrapper(base_cnn, name=name, seed=seed)

    variants = [
        AgentSpec(
            agent_id="L5_PolicyOnly_NoShortcut",
            display_name="CNN Policy-Only (Raw)",
            level=5,
            mode="ablation",
            factory=lambda seed, _c=cnn_no_shortcut: _reseed_policy_wrapper(
                _c, name="L5_PolicyOnly_NoShortcut", seed=seed
            ),
            config_summary="Pure ResNet Policy (no tactical shortcut, no search)",
            budget_note="1 forward pass",
            descriptor={"kind": "ablation_policy_variant", "variant_id": "L5_PolicyOnly_NoShortcut"},
        ),
        AgentSpec(
            agent_id="L5_PolicyOnly_Shortcut",
            display_name="CNN Policy + Shortcut",
            level=5,
            mode="ablation",
            factory=lambda seed, _cnn=shared_cnn: _reseed_policy_wrapper(
                _cnn, name="L5_PolicyOnly_Shortcut", seed=seed
            ),
            config_summary="ResNet Policy + Tactical Shortcut (no tree search)",
            budget_note="1 forward pass",
            descriptor={"kind": "ablation_policy_variant", "variant_id": "L5_PolicyOnly_Shortcut"},
        ),
        AgentSpec(
            agent_id="L5_Policy_Plus_PUCT",
            display_name="CNN Policy + PUCT(150)",
            level=5,
            mode="ablation",
            factory=lambda seed, _c=cnn_plus_puct: _reseed_policy_wrapper(
                _c, name="L5_Policy_Plus_PUCT", seed=seed
            ),
            config_summary="ResNet Policy + PUCT-MCTS (150 sims, cap=0.5s)",
            budget_note="150 sims / 0.5s cap",
            descriptor={"kind": "ablation_policy_variant", "variant_id": "L5_Policy_Plus_PUCT"},
        ),
        AgentSpec(
            agent_id="L7_AlphaGo_EnsemblePUCT",
            display_name="AlphaGo Cấp 7 (L5+L6+PUCT)",
            level=7,
            mode="ablation",
            factory=lambda seed, _c=shared_cnn, _r=shared_rl: AlphaGoAgent(
                name="L7_AlphaGo_EnsemblePUCT",
                policy_agent=_c,
                value_agent=_r,
                auto_load_default=False,
                num_simulations=200,
                time_limit_sec=0.5,
                value_blend_alpha=0.25,
                rl_prior_weight=0.35,
                seed=seed,
            ),
            config_summary="Ensemble Prior (L5+L6) + Value Blend + PUCT (200 sims, cap=0.5s)",
            budget_note="200 sims / 0.5s cap",
            descriptor={"kind": "ablation_policy_variant", "variant_id": "L7_AlphaGo_EnsemblePUCT"},
        ),
    ]

    workers = max(1, int(max_workers))
    if workers > 1:
        _ensure_pythonpath(root)
    pool: ProcessPoolExecutor | None = (
        _create_worker_pool(workers) if workers > 1 else None
    )
    rows: list[dict[str, Any]] = []
    try:
        for var_spec in variants:
            for opp in opp_specs:
                log_file = logs_dir / f"ablation3_{var_spec.agent_id}_vs_{opp.agent_id}.jsonl"
                summ, _ = run_pair_match(
                    spec_a=var_spec,
                    spec_b=opp,
                    games_per_pair=games_per_config,
                    seeds=seeds,
                    log_path=log_file,
                    resume=resume,
                    verbose=verbose,
                    max_workers=workers,
                    executor=pool,
                    project_root=root,
                )
                summ["variant"] = var_spec.display_name
                summ["opponent"] = opp.agent_id
                rows.append(summ)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "ablation3_policy_vs_puct_alphago.csv", index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# 7. CLI Entrypoint (`python -m src.tournament`)
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Giải đấu Round-Robin đa chế độ & Ablation cho Caro AI 11x11 (Mục 3.5)."
    )
    parser.add_argument(
        "--mode",
        choices=("time_budget", "default_config", "both"),
        default="both",
        help="Chế độ ngân sách: time_budget, default_config, hoặc both.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Số ván/cặp đấu (chẵn; Mục 3.5 yêu cầu tối thiểu 100, khuyến nghị 200).",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=0.25,
        help="Ngân sách thời gian/nước (giây) cho chế độ time_budget.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Danh sách seed cho các tác tử ngẫu nhiên.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Danh sách các cấp AI tham gia (1..7).",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Chạy thêm đầy đủ 03 thí nghiệm Ablation/Sensitivity (Mục 3.5).",
    )
    parser.add_argument(
        "--ablation-games",
        type=int,
        default=20,
        help="Số ván mỗi cấu hình trong các thí nghiệm ablation (chẵn).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ghi đè log cũ thay vì tiếp tục từ file .jsonl đã lưu.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    modes = ("time_budget", "default_config") if args.mode == "both" else (args.mode,)
    resume = not args.no_resume

    for m in modes:
        res = run_round_robin(
            mode=m,
            games_per_pair=args.games,
            time_budget_sec=args.time_budget,
            seeds=args.seeds,
            include_levels=args.levels,
            resume=resume,
            verbose=True,
        )
        print(f"\n--- LEADERBOARD [{m}] ---")
        cols = [
            "rank", "agent_id", "score_pct", "wilson_ci95_low", "wilson_ci95_high",
            "elo_rating", "score_as_X_pct", "score_as_O_pct", "avg_ms", "p95_ms", "avg_search",
        ]
        print(res["leaderboard"][cols].to_string(index=False))

    if args.ablations:
        print("\n=== RUNNING ABLATION 1: Alpha-Beta ON/OFF & Depth ===")
        run_ablation_alphabeta_depth(
            games_per_depth=args.ablation_games, seeds=args.seeds, resume=resume, verbose=True
        )
        print("\n=== RUNNING ABLATION 2: MCTS Budget & Heuristic ON/OFF ===")
        run_ablation_mcts_budget_heuristic(
            games_per_config=args.ablation_games, seeds=args.seeds, resume=resume, verbose=True
        )
        print("\n=== RUNNING ABLATION 3: Policy-Only vs Policy+PUCT vs AlphaGo Cấp 7 ===")
        run_ablation_policy_vs_search(
            games_per_config=args.ablation_games, seeds=args.seeds, resume=resume, verbose=True
        )


if __name__ == "__main__":
    main()
