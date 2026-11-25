from __future__ import annotations

import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

try:
    from .base_agent import BaseAgent
    from .cnn_agent import CNNAgent
    from .rl_agent import RLAgent, resolve_device
    from ..caro_env import CaroEnv
    from ..heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions
except ImportError:
    from agents.base_agent import BaseAgent
    from agents.cnn_agent import CNNAgent
    from agents.rl_agent import RLAgent, resolve_device
    from caro_env import CaroEnv
    from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions


DEFAULT_POLICY_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "05_cnn_policy_best.pt"
DEFAULT_VALUE_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "06_rl_best.pt"

# Cache bảng Zobrist 64-bit cho Eval Cache theo board_size
_AG_ZOBRIST_CACHE: dict[int, tuple[tuple[int, ...], tuple[int, ...], int]] = {}


def _get_zobrist_keys(board_size: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    cached = _AG_ZOBRIST_CACHE.get(board_size)
    if cached is not None:
        return cached
    rng = random.Random(20260707 + board_size * 997)
    num_cells = board_size * board_size
    keys_x = tuple(rng.getrandbits(64) for _ in range(num_cells))
    keys_o = tuple(rng.getrandbits(64) for _ in range(num_cells))
    turn_key = rng.getrandbits(64)
    res = (keys_x, keys_o, turn_key)
    _AG_ZOBRIST_CACHE[board_size] = res
    return res


def _compute_state_hash(
    env: CaroEnv,
    keys_x: tuple[int, ...],
    keys_o: tuple[int, ...],
    turn_key: int,
) -> int:
    h = 0
    flat = env.board.ravel()
    for idx_np in np.flatnonzero(flat):
        idx = int(idx_np)
        h ^= keys_x[idx] if flat[idx] == 1 else keys_o[idx]
    if env.current_player == -1:
        h ^= turn_key
    return h


class AlphaGoNode:
    """Nút trong cây tìm kiếm PUCT-MCTS của tác tử Cấp 7A (Hybrid AlphaGo)."""

    __slots__ = (
        "parent",
        "action_from_parent",
        "player_just_moved",
        "prior",
        "visits",
        "wins",
        "neural_value",
        "children",
        "is_expanded",
    )

    def __init__(
        self,
        parent: AlphaGoNode | None = None,
        action_from_parent: int | None = None,
        player_just_moved: int = 0,
        prior: float = 0.0,
    ) -> None:
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.player_just_moved = player_just_moved
        self.prior: float = float(prior)
        self.visits: int = 0
        self.wins: float = 0.0
        self.neural_value: float = 0.0  # Giá trị ước lượng lá theo góc nhìn player_just_moved trong [-1, +1]
        self.children: dict[int, AlphaGoNode] = {}
        self.is_expanded: bool = False

    def expand(self, actions_and_priors: list[tuple[int, float]], next_player: int) -> None:
        for act, prior in actions_and_priors:
            self.children[act] = AlphaGoNode(
                parent=self,
                action_from_parent=act,
                player_just_moved=next_player,
                prior=prior,
            )
        self.is_expanded = True

    def select_child_puct(self, c_puct: float, fpu_reduction: float = 0.15) -> AlphaGoNode:
        """Chọn nút con tối đa hóa công thức AlphaGo PUCT kèm First Play Urgency (FPU)."""
        sqrt_n = math.sqrt(self.visits + 1.0)
        parent_q = (self.wins / self.visits) if self.visits > 0 else 0.5
        fpu = max(0.0, parent_q - fpu_reduction) if self.visits > 0 else 0.5

        best_score = -math.inf
        best_child: AlphaGoNode | None = None
        for child in self.children.values():
            q_val = (child.wins / child.visits) if child.visits > 0 else fpu
            u_val = c_puct * child.prior * (sqrt_n / (1.0 + child.visits))
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child


class AlphaGoAgent(BaseAgent):
    """
    Tác tử Cấp 7A (Bonus — AlphaGo-Inspired Hybrid PUCT-MCTS) cho Caro 11x11.

    Kết hợp 3 trụ cột theo đúng kiến trúc AlphaGo (Silver et al., Nature 2016):
      1. **Policy Network (Cấp 5B — Residual Policy CNN)**: cung cấp phân phối tiên nghiệm
         `P(s, a)` dẫn đường cây tìm kiếm PUCT và lọc top-K nhánh chiến thuật mạnh nhất.
      2. **Value / Q Network (Cấp 6A — Dueling Double-DQN)**: cung cấp ước lượng giá trị lá
         học từ tăng cường `v_RL(s) = clip(max_{a in legal} Q(s, a), -1, 1)`.
      3. **PUCT-MCTS + Zobrist Eval Cache + Pha trộn giá trị lá (`value_blend_alpha`)**:
         `v_leaf = alpha * v_RL(s) + (1 - alpha) * tanh(evaluate_board / scale)`
         hỗ trợ linh hoạt từ **Pure Neural (`alpha=1.0`)** đến **Hybrid (`alpha=0.5`)**
         cho cả thực chiến lẫn Ablation Study và Explainability trên UI Demo.
    """

    def __init__(
        self,
        name: str = "Level7_AlphaGo_PUCT",
        policy_checkpoint: str | Path | None = None,
        value_checkpoint: str | Path | None = None,
        policy_agent: CNNAgent | None = None,
        value_agent: RLAgent | None = None,
        auto_load_default: bool = True,
        num_simulations: int | None = None,
        time_limit_sec: float | None = 0.8,
        c_puct: float = 1.6,
        fpu_reduction: float = 0.15,
        value_blend_alpha: float = 0.25,
        rl_prior_weight: float = 0.35,
        rl_prior_temp: float = 0.20,
        heuristic_scale: float = 12000.0,
        candidate_radius: int = 2,
        max_branch_root: int = 14,
        max_branch_inner: int = 8,
        use_tactical_shortcut: bool = True,
        use_eval_cache: bool = True,
        device: str = "auto",
        board_size: int = 11,
        seed: int | None = 42,
    ) -> None:
        super().__init__(name=name)

        if num_simulations is None and time_limit_sec is None:
            num_simulations = 250

        if num_simulations is not None:
            if isinstance(num_simulations, bool) or not isinstance(num_simulations, int) or num_simulations <= 0:
                raise ValueError("num_simulations phải là số nguyên dương (> 0) hoặc None.")
        if time_limit_sec is not None:
            if (
                isinstance(time_limit_sec, bool)
                or not isinstance(time_limit_sec, (int, float))
                or not math.isfinite(float(time_limit_sec))
                or float(time_limit_sec) <= 0.0
            ):
                raise ValueError("time_limit_sec phải là số thực hữu hạn dương hoặc None.")
        if isinstance(c_puct, bool) or not math.isfinite(float(c_puct)) or float(c_puct) < 0.0:
            raise ValueError("c_puct phải là số thực hữu hạn >= 0.")
        if isinstance(fpu_reduction, bool) or not math.isfinite(float(fpu_reduction)) or float(fpu_reduction) < 0.0:
            raise ValueError("fpu_reduction phải là số thực hữu hạn >= 0.")
        if (
            isinstance(value_blend_alpha, bool)
            or not math.isfinite(float(value_blend_alpha))
            or not (0.0 <= float(value_blend_alpha) <= 1.0)
        ):
            raise ValueError("value_blend_alpha phải nằm trong [0.0, 1.0].")
        if (
            isinstance(rl_prior_weight, bool)
            or not math.isfinite(float(rl_prior_weight))
            or not (0.0 <= float(rl_prior_weight) <= 1.0)
        ):
            raise ValueError("rl_prior_weight phải nằm trong [0.0, 1.0].")
        if isinstance(rl_prior_temp, bool) or float(rl_prior_temp) <= 0.0:
            raise ValueError("rl_prior_temp phải > 0.")
        if isinstance(heuristic_scale, bool) or float(heuristic_scale) <= 0.0:
            raise ValueError("heuristic_scale phải > 0.")
        if isinstance(candidate_radius, bool) or not isinstance(candidate_radius, int) or candidate_radius < 1:
            raise ValueError("candidate_radius phải là số nguyên >= 1.")
        if isinstance(max_branch_root, bool) or not isinstance(max_branch_root, int) or max_branch_root < 1:
            raise ValueError("max_branch_root phải >= 1.")
        if isinstance(max_branch_inner, bool) or not isinstance(max_branch_inner, int) or max_branch_inner < 1:
            raise ValueError("max_branch_inner phải >= 1.")

        self.device = resolve_device(device)
        self.board_size = int(board_size)
        self.num_simulations: int | None = num_simulations
        self.time_limit_sec: float | None = float(time_limit_sec) if time_limit_sec is not None else None
        self.c_puct: float = float(c_puct)
        self.fpu_reduction: float = float(fpu_reduction)
        self.value_blend_alpha: float = float(value_blend_alpha)
        self.rl_prior_weight: float = float(rl_prior_weight)
        self.rl_prior_temp: float = float(rl_prior_temp)
        self.heuristic_scale: float = float(heuristic_scale)
        self.candidate_radius: int = int(candidate_radius)
        self.max_branch_root: int = int(max_branch_root)
        self.max_branch_inner: int = int(max_branch_inner)
        self.use_tactical_shortcut: bool = bool(use_tactical_shortcut)
        self.use_eval_cache: bool = bool(use_eval_cache)
        self.seed = seed
        self.rng = random.Random(seed)

        # 1. Nạp Policy Network (Cấp 5B)
        if policy_agent is not None:
            self.policy_agent = policy_agent
        else:
            p_path = Path(policy_checkpoint) if policy_checkpoint is not None else None
            if p_path is None and auto_load_default and DEFAULT_POLICY_CHECKPOINT.exists():
                p_path = DEFAULT_POLICY_CHECKPOINT
            self.policy_agent = CNNAgent(
                name=f"{name}_Policy5B",
                model_type="resnet",
                backend="pytorch",
                model_path=p_path,
                auto_load_default=False,
                board_size=self.board_size,
                use_tactical_shortcut=False,
                use_mix_mcts=False,
                device=str(self.device),
                seed=seed,
            )

        # 2. Nạp Value / Q Network (Cấp 6A)
        if value_agent is not None:
            self.value_agent = value_agent
        else:
            v_path = Path(value_checkpoint) if value_checkpoint is not None else None
            if v_path is None and auto_load_default and DEFAULT_VALUE_CHECKPOINT.exists():
                v_path = DEFAULT_VALUE_CHECKPOINT
            self.value_agent = RLAgent(
                checkpoint_path=v_path,
                in_channels=4,
                conv_channels=(48,),
                board_size=self.board_size,
                device=str(self.device),
                name=f"{name}_Value6A",
                use_residual=True,
                num_res_blocks=3,
                norm_type="group",
                auto_load_default=False,
            )

        # Telemetry & Explainability
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_cache_hits: int = 0
        self.last_decision_stage: str = ""
        self.last_root: AlphaGoNode | None = None
        self.last_root_value_rl: float = 0.0

    def _check_root_tactical_shortcut(self, env: CaroEnv, candidates: list[int]) -> tuple[int | None, str]:
        """Kiểm tra nước thắng ngay 5 quân hoặc chặn đối thủ thắng 5 quân tại gốc."""
        player = env.current_player
        opponent = -player

        for act in candidates:
            if evaluate_cell_threat(env, act, player)["is_win_5"]:
                return int(act), "Shortcut: immediate win"

        block_moves = [int(a) for a in candidates if evaluate_cell_threat(env, a, opponent)["is_win_5"]]
        if block_moves:
            best_block = max(
                block_moves,
                key=lambda a: evaluate_cell_threat(env, a, player)["score"]
                + evaluate_cell_threat(env, a, opponent)["score"],
            )
            return int(best_block), "Shortcut: block immediate win"

        return None, ""

    def _get_policy_priors(
        self,
        env: CaroEnv,
        candidates: list[int],
        max_branch: int,
        cache_key: int,
        policy_cache: dict[int, list[tuple[int, float]]],
    ) -> list[tuple[int, float]]:
        """
        Trả về top-K `(action, prior)` kết hợp Ensemble Neural Prior từ cả hai mạng:
          `P_ens(s, a) = (1 - w_rl) * P_5B(s, a) + w_rl * softmax(Q_6A(s, a) / T)`
        (có Zobrist Eval Cache).
        """
        if self.use_eval_cache:
            cached = policy_cache.get(cache_key)
            if cached is not None:
                self.last_cache_hits += 1
                return cached

        probs_5b = self.policy_agent.predict_probs(env, restrict_to_candidates=False)
        p5 = np.array([max(1e-6, float(probs_5b[a])) for a in candidates], dtype=np.float64)
        p5 /= float(np.sum(p5))

        w_rl = self.rl_prior_weight
        if w_rl > 0.0:
            masked_q = self.value_agent.get_masked_q_values(env)
            q_cands = np.array([float(masked_q[a]) if np.isfinite(masked_q[a]) else -10.0 for a in candidates], dtype=np.float64)
            q_scaled = (q_cands - float(np.max(q_cands))) / self.rl_prior_temp
            q_exp = np.exp(np.clip(q_scaled, -40.0, 0.0))
            p6 = q_exp / max(1e-12, float(np.sum(q_exp)))
            blended = (1.0 - w_rl) * p5 + w_rl * p6
        else:
            blended = p5

        k = min(len(candidates), max_branch)
        ranked_idx = np.argsort(blended)[::-1][:k]
        top_actions = [int(candidates[i]) for i in ranked_idx]
        raw_p = np.array([max(1e-6, float(blended[i])) for i in ranked_idx], dtype=np.float64)
        raw_p /= float(np.sum(raw_p))
        result = [(act, float(p)) for act, p in zip(top_actions, raw_p)]

        if self.use_eval_cache:
            policy_cache[cache_key] = result
        return result

    def _evaluate_leaf_p1(
        self,
        env: CaroEnv,
        cache_key: int,
        value_cache: dict[int, tuple[float, float]],
    ) -> tuple[float, float]:
        """
        Đánh giá nút lá theo góc nhìn người chơi 1 (X) trong `[-1.0, +1.0]`.
        Trả về `(blended_value_p1, neural_value_p1)`.
        Khi heuristic đã phát hiện thế thắng cưỡng bức tuyệt đối (`|score| >= 40_000`,
        vd: thắng ngay hoặc 4 mở không thể chặn) và `alpha < 1.0`, bảo toàn trực tiếp
        tín hiệu cưỡng bức thay vì để ước lượng 1-ply làm nhiễu.
        """
        if env.is_terminal():
            res = float(env.result(perspective=1))
            return res, res

        if self.use_eval_cache:
            cached = value_cache.get(cache_key)
            if cached is not None:
                self.last_cache_hits += 1
                return cached

        alpha = self.value_blend_alpha
        v_rl_p1 = 0.0
        v_heur_p1 = 0.0
        heur_forced = False

        if alpha < 1.0:
            heur_score = evaluate_board(env, perspective=1, consider_turn=True)
            v_heur_p1 = float(math.tanh(heur_score / self.heuristic_scale))
            if abs(heur_score) >= 40000.0:
                heur_forced = True

        if alpha > 0.0:
            if heur_forced:
                # Thế cờ đã ngã ngũ cưỡng bức -> dùng luôn tín hiệu chính xác, tiết kiệm 1 lượt forward RL
                v_rl_p1 = v_heur_p1
            else:
                # Dùng độ chênh V_head đã chuẩn hóa kết hợp lợi thế top-Q từ Dueling DQN 6A
                v_head = self.value_agent.predict_value(env)
                v_to_move = float(math.tanh(v_head + 0.5))
                v_rl_p1 = v_to_move * float(env.current_player)

        blended_p1 = v_heur_p1 if heur_forced else (alpha * v_rl_p1 + (1.0 - alpha) * v_heur_p1)
        out = (float(blended_p1), float(v_rl_p1))

        if self.use_eval_cache:
            value_cache[cache_key] = out
        return out

    def select_action(self, env: CaroEnv) -> int:
        """Chọn nước đi tối ưu nhất bằng cây tìm kiếm PUCT-MCTS kết hợp Policy 5B + Value/Q 6A."""
        t0 = time.perf_counter()
        self.last_search_count = 0
        self.last_cache_hits = 0
        self.last_decision_stage = ""
        self.last_root = None
        self.last_root_value_rl = 0.0

        try:
            legal = env.legal_actions()
            if not legal:
                raise ValueError("Không còn nước đi hợp lệ nào trên bàn cờ.")

            candidates = get_candidate_actions(env, radius=self.candidate_radius)
            if not candidates:
                candidates = legal

            if len(candidates) == 1:
                self.last_decision_stage = "Shortcut: single candidate"
                return int(candidates[0])

            if self.use_tactical_shortcut:
                urgent, stage = self._check_root_tactical_shortcut(env, candidates)
                if urgent is not None:
                    self.last_decision_stage = stage
                    return int(urgent)

            keys_x, keys_o, turn_key = _get_zobrist_keys(env.board_size)
            root_hash = _compute_state_hash(env, keys_x, keys_o, turn_key) if self.use_eval_cache else 0
            policy_cache: dict[int, list[tuple[int, float]]] = {}
            value_cache: dict[int, tuple[float, float]] = {}

            # Đánh giá giá trị gốc từ mạng RL 6A phục vụ giải thích (Explainability)
            if self.value_blend_alpha > 0.0:
                _, root_v_p1 = self._evaluate_leaf_p1(env, root_hash, value_cache)
                self.last_root_value_rl = root_v_p1 * float(env.current_player)

            root = AlphaGoNode(
                parent=None,
                action_from_parent=None,
                player_just_moved=-env.current_player,
                prior=1.0,
            )
            root_priors = self._get_policy_priors(
                env, candidates, self.max_branch_root, root_hash, policy_cache
            )
            root.expand(root_priors, next_player=env.current_player)

            c_puct = self.c_puct
            fpu_red = self.fpu_reduction
            sims_done = 0

            while True:
                if self.num_simulations is not None and sims_done >= self.num_simulations:
                    break
                if (
                    self.time_limit_sec is not None
                    and sims_done > 0
                    and (time.perf_counter() - t0) >= self.time_limit_sec
                ):
                    break

                node = root
                sim_env = env.clone()
                cur_hash = root_hash

                # 1. Selection theo AlphaGo PUCT + FPU
                while node.is_expanded and node.children and not sim_env.is_terminal():
                    mover = sim_env.current_player
                    node = node.select_child_puct(c_puct, fpu_red)
                    act = int(node.action_from_parent)
                    sim_env.step(act)
                    if self.use_eval_cache:
                        cur_hash ^= (keys_x[act] if mover == 1 else keys_o[act]) ^ turn_key
                    if node.visits == 0:
                        break

                # 2. Lazy Expansion bằng Policy Network 5B khi nút được thăm lại lần >= 2
                if node.visits > 0 and not node.is_expanded and not sim_env.is_terminal():
                    sub_cands = get_candidate_actions(sim_env, radius=self.candidate_radius)
                    if not sub_cands:
                        sub_cands = sim_env.legal_actions()
                    sub_priors = self._get_policy_priors(
                        sim_env, sub_cands, self.max_branch_inner, cur_hash, policy_cache
                    )
                    if sub_priors:
                        mover = sim_env.current_player
                        node.expand(sub_priors, next_player=mover)
                        node = node.select_child_puct(c_puct, fpu_red)
                        act = int(node.action_from_parent)
                        sim_env.step(act)
                        if self.use_eval_cache:
                            cur_hash ^= (keys_x[act] if mover == 1 else keys_o[act]) ^ turn_key

                # 3. Evaluation bằng Value/Q 6A (+ pha trộn Heuristic theo value_blend_alpha)
                outcome_p1, neural_p1 = self._evaluate_leaf_p1(sim_env, cur_hash, value_cache)
                node.neural_value = neural_p1 * float(node.player_just_moved)

                # 4. Backpropagation theo thang thưởng chuẩn [0.0, 1.0]
                reward_p1 = 0.5 * (outcome_p1 + 1.0)
                reward_p2 = 1.0 - reward_p1
                curr: AlphaGoNode | None = node
                while curr is not None:
                    curr.visits += 1
                    curr.wins += reward_p1 if curr.player_just_moved == 1 else reward_p2
                    curr = curr.parent

                sims_done += 1

            self.last_root = root
            self.last_search_count = sims_done
            self.last_decision_stage = f"AlphaGo-PUCT: {sims_done} sims (alpha={self.value_blend_alpha:.2f})"

            best_child = max(
                root.children.values(),
                key=lambda c: (c.visits, (c.wins / c.visits) if c.visits > 0 else 0.0, c.prior),
            )
            return int(best_child.action_from_parent)
        finally:
            self.last_move_time_ms = (time.perf_counter() - t0) * 1000.0

    def get_root_stats(self, top_k: int | None = None) -> list[dict[str, Any]]:
        """Trả về thống kê giải thích (Explainability) của các nhánh gốc theo thứ tự ưu tiên."""
        if self.last_root is None:
            return []
        total_visits = max(1, self.last_root.visits)
        n = self.board_size
        stats: list[dict[str, Any]] = []
        for action, child in self.last_root.children.items():
            q_mcts = (child.wins / child.visits) if child.visits > 0 else 0.0
            stats.append(
                {
                    "action": int(action),
                    "row": int(action // n),
                    "col": int(action % n),
                    "visits": int(child.visits),
                    "visit_pct": round(100.0 * child.visits / total_visits, 2),
                    "value": float(q_mcts),
                    "q_mcts": round(float(q_mcts), 4),
                    "prior_5b": round(float(child.prior), 4),
                    "value_rl": round(float(child.neural_value), 4),
                }
            )
        stats.sort(key=lambda item: (item["visits"], item["value"], item["prior_5b"]), reverse=True)
        return stats[:top_k] if top_k is not None else stats

    def explain_decision(self, top_k: int = 5) -> dict[str, Any]:
        """Xuất báo cáo giải thích nước đi vừa chọn phục vụ UI Demo và báo cáo định tính."""
        return {
            "agent": self.name,
            "stage": self.last_decision_stage,
            "simulations": self.last_search_count,
            "cache_hits": self.last_cache_hits,
            "move_time_ms": round(self.last_move_time_ms, 2),
            "root_value_rl": round(self.last_root_value_rl, 4),
            "value_blend_alpha": self.value_blend_alpha,
            "top_moves": self.get_root_stats(top_k=top_k),
        }
