from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from .base_agent import BaseAgent
    from ..heuristic import (
        evaluate_board,
        evaluate_cell_threat,
        get_candidate_actions,
    )
except ImportError:
    from agents.base_agent import BaseAgent
    from heuristic import (
        evaluate_board,
        evaluate_cell_threat,
        get_candidate_actions,
    )

if TYPE_CHECKING:
    from caro_env import CaroEnv


class _SearchTimeout(Exception):
    """Ngoại lệ nội bộ dùng để ngắt ngay lập tức khi vượt quá time_limit_sec."""


# Tách biệt hoàn toàn miền điểm thắng/thua tuyệt đối với heuristic [-80_000, +80_000]
_WIN_SCORE: float = 10_000_000.0
_WIN_THRESHOLD: float = 9_000_000.0
_MATE_PLY_STEP: float = 1_000.0

# Cờ trạng thái Transposition Table
_TT_EXACT: int = 0
_TT_LOWERBOUND: int = 1
_TT_UPPERBOUND: int = 2

# Cache bảng Zobrist theo board_size (tạo cố định theo seed để tái lập hoàn toàn)
_ZOBRIST_CACHE: dict[int, tuple[tuple[int, ...], tuple[int, ...], int]] = {}


def _get_zobrist_keys(board_size: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Trả về `(keys_x, keys_o, turn_key)` 64-bit cho Zobrist Hashing."""
    cached = _ZOBRIST_CACHE.get(board_size)
    if cached is not None:
        return cached

    rng = random.Random(20260303 + board_size * 1009)
    num_cells = board_size * board_size
    keys_x = tuple(rng.getrandbits(64) for _ in range(num_cells))
    keys_o = tuple(rng.getrandbits(64) for _ in range(num_cells))
    turn_key = rng.getrandbits(64)
    result = (keys_x, keys_o, turn_key)
    _ZOBRIST_CACHE[board_size] = result
    return result


def _compute_zobrist_hash(
    env: CaroEnv,
    keys_x: tuple[int, ...],
    keys_o: tuple[int, ...],
    turn_key: int,
) -> int:
    """Tính mã băm Zobrist từ trạng thái bàn cờ hiện tại (an toàn cả khi board sửa trực tiếp)."""
    h = 0
    flat = env.board.ravel()
    occupied = np.flatnonzero(flat)
    for idx_np in occupied:
        idx = int(idx_np)
        if flat[idx] == 1:
            h ^= keys_x[idx]
        else:
            h ^= keys_o[idx]
    if env.current_player == -1:
        h ^= turn_key
    return h


def _from_child_score(score: float) -> float:
    """Giảm điểm chiếu hết 1 nấc theo khoảng cách ply để ưu tiên thắng sớm / thua muộn."""
    if score > _WIN_THRESHOLD:
        return score - _MATE_PLY_STEP
    if score < -_WIN_THRESHOLD:
        return score + _MATE_PLY_STEP
    return score


def _to_child_bound(bound: float) -> float:
    """Biến đổi ngược ngưỡng alpha/beta khi đi xuống node con để bảo toàn cắt tỉa tuyệt đối."""
    if math.isfinite(bound):
        if bound > _WIN_THRESHOLD:
            return bound + _MATE_PLY_STEP
        if bound < -_WIN_THRESHOLD:
            return bound - _MATE_PLY_STEP
    return bound


class MinimaxAgent(BaseAgent):
    """
    Tác tử Cấp 3: Minimax tích hợp Alpha-Beta Pruning, Move Ordering chuẩn ưu tiên,
    Zobrist Transposition Table, Candidate Action Pruning, Iterative Deepening,
    Tactical Shortcuts và Kiểm soát thời gian.
    """

    def __init__(
        self,
        name: str = "MinimaxAgent",
        max_depth: int = 3,
        use_alpha_beta: bool = True,
        use_move_ordering: bool = True,
        candidate_radius: int | None = 2,
        time_limit_sec: float | None = 1.0,  # Truyền None để chạy cố định ở max_depth
        use_tt: bool = True,
        tt_max_entries: int = 100_000,
        max_branch_root: int | None = None,
        max_branch_inner: int | None = None,
        use_tactical_shortcut: bool = True,
        seed: int | None = None,
        tie_break_eps: float = 0.0,
    ):
        super().__init__(name=name)

        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth phải là số nguyên >= 1.")
        if candidate_radius is not None:
            if isinstance(candidate_radius, bool) or not isinstance(candidate_radius, int) or candidate_radius < 1:
                raise ValueError("candidate_radius phải là số nguyên >= 1 hoặc None.")
        if time_limit_sec is not None and (isinstance(time_limit_sec, bool) or time_limit_sec <= 0):
            raise ValueError("time_limit_sec phải > 0 hoặc None.")
        if isinstance(tt_max_entries, bool) or not isinstance(tt_max_entries, int) or tt_max_entries < 1:
            raise ValueError("tt_max_entries phải là số nguyên >= 1.")
        if max_branch_root is not None:
            if isinstance(max_branch_root, bool) or not isinstance(max_branch_root, int) or max_branch_root < 1:
                raise ValueError("max_branch_root phải là số nguyên >= 1 hoặc None.")
        if max_branch_inner is not None:
            if isinstance(max_branch_inner, bool) or not isinstance(max_branch_inner, int) or max_branch_inner < 1:
                raise ValueError("max_branch_inner phải là số nguyên >= 1 hoặc None.")
        if isinstance(tie_break_eps, bool) or tie_break_eps < 0:
            raise ValueError("tie_break_eps phải >= 0.")

        self.max_depth = max_depth
        self.use_alpha_beta = bool(use_alpha_beta)
        self.use_move_ordering = bool(use_move_ordering)
        self.candidate_radius = candidate_radius
        self.time_limit_sec = float(time_limit_sec) if time_limit_sec is not None else None
        self.use_tt = bool(use_tt)
        self.tt_max_entries = tt_max_entries
        self.max_branch_root = max_branch_root
        self.max_branch_inner = max_branch_inner
        self.use_tactical_shortcut = bool(use_tactical_shortcut)
        self.seed = seed
        self.tie_break_eps = float(tie_break_eps)
        self.rng = random.Random(seed)

        # Thuộc tính đo lường hiệu năng phục vụ Ablation Study và Tournament
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_completed_depth: int = 0
        self.last_best_score: float = 0.0
        self.last_tt_hits: int = 0
        self.last_decision_stage: str = ""

    def select_action(self, env: CaroEnv) -> int:
        """
        Lựa chọn nước đi tối ưu nhất từ môi trường `env` hiện tại.
        Đảm bảo không làm biến đổi trạng thái của môi trường gốc.
        """
        start_time = time.perf_counter()
        self.last_search_count = 0
        self.last_completed_depth = 0
        self.last_best_score = 0.0
        self.last_tt_hits = 0
        self.last_decision_stage = ""

        try:
            legal_actions = env.legal_actions()
            if not legal_actions:
                raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ.")

            # Làm việc trên bản sao để bảo đảm tuyệt đối không biến đổi env gốc
            sim_env = env.clone()

            if self.candidate_radius is None:
                candidates = legal_actions
            else:
                candidates = get_candidate_actions(sim_env, radius=self.candidate_radius)
                if not candidates:
                    candidates = legal_actions

            # Nếu chỉ còn đúng 1 nước ứng viên (vd: ô tâm bàn cờ trống hoặc ô cuối cùng)
            if len(candidates) == 1:
                self.last_search_count = 1
                self.last_completed_depth = 1
                self.last_decision_stage = "Shortcut: single candidate"
                return int(candidates[0])

            perspective = sim_env.current_player
            opponent = -perspective

            # Đánh giá mối đe dọa 1 lần tại gốc (tái sử dụng cho cả Shortcut và Move Ordering)
            root_my_threats: dict[int, dict[str, Any]] | None = None
            root_opp_threats: dict[int, dict[str, Any]] | None = None

            if self.use_tactical_shortcut or self.use_move_ordering:
                root_my_threats = {}
                root_opp_threats = {}
                for action in candidates:
                    act = int(action)
                    root_my_threats[act] = evaluate_cell_threat(sim_env, act, perspective)
                    root_opp_threats[act] = evaluate_cell_threat(sim_env, act, opponent)

            if self.use_tactical_shortcut and root_my_threats is not None and root_opp_threats is not None:
                # 1. Immediate Win Shortcut: Thắng ngay lập tức tại gốc
                for action in candidates:
                    if root_my_threats[int(action)]["is_win_5"]:
                        self.last_search_count = 1
                        self.last_completed_depth = 1
                        self.last_best_score = _WIN_SCORE
                        self.last_decision_stage = "Shortcut: immediate win"
                        return int(action)

                # 2. Forced Single Block Shortcut: Đối thủ có đúng 1 nước thắng ngay -> bắt buộc chặn
                opp_wins = [int(a) for a in candidates if root_opp_threats[int(a)]["is_win_5"]]
                if len(opp_wins) == 1:
                    self.last_search_count = 1
                    self.last_completed_depth = 1
                    self.last_decision_stage = "Shortcut: forced block"
                    return opp_wins[0]

            # Sắp xếp danh sách ứng viên tại gốc nếu bật Move Ordering
            if self.use_move_ordering and root_my_threats is not None and root_opp_threats is not None:
                candidates = self._order_moves_from_threats(
                    sim_env,
                    candidates,
                    root_my_threats,
                    root_opp_threats,
                )
                if self.max_branch_root is not None and len(candidates) > self.max_branch_root:
                    candidates = candidates[: self.max_branch_root]

            best_action = int(candidates[0])
            best_score = 0.0

            keys_x, keys_o, turn_key = _get_zobrist_keys(sim_env.board_size)
            root_hash = _compute_zobrist_hash(sim_env, keys_x, keys_o, turn_key) if self.use_tt else 0
            tt: dict[int, tuple[int, int, float, int | None]] = {}
            killers: dict[int, list[int]] = {}

            # Nếu tắt giới hạn thời gian (dùng cho Ablation ở độ sâu cố định), chạy trực tiếp tại max_depth.
            # Nếu có giới hạn thời gian, chạy Iterative Deepening từ depth = 1 -> max_depth.
            start_depth = self.max_depth if self.time_limit_sec is None else 1

            for current_depth in range(start_depth, self.max_depth + 1):
                if current_depth > start_depth and self.time_limit_sec is not None:
                    if time.perf_counter() - start_time >= self.time_limit_sec:
                        break

                # Đưa Principal Variation (best_action ở độ sâu trước) lên đầu để tối ưu cắt tỉa Alpha-Beta
                if current_depth > start_depth and best_action in candidates:
                    ordered_candidates = [best_action] + [a for a in candidates if a != best_action]
                else:
                    ordered_candidates = candidates

                try:
                    depth_best_action, depth_best_val = self._search_root(
                        env=sim_env,
                        candidates=ordered_candidates,
                        depth=current_depth,
                        perspective=perspective,
                        start_time=start_time,
                        board_hash=root_hash,
                        keys_x=keys_x,
                        keys_o=keys_o,
                        turn_key=turn_key,
                        tt=tt,
                        killers=killers,
                    )
                    best_action = depth_best_action
                    best_score = depth_best_val
                    self.last_completed_depth = current_depth
                    self.last_best_score = depth_best_val

                    # Đã tìm thấy chuỗi thắng cưỡng bức -> dừng sớm Iterative Deepening
                    if depth_best_val > _WIN_THRESHOLD:
                        self.last_decision_stage = f"Search: forced win at depth {current_depth}"
                        break
                except _SearchTimeout:
                    # Hủy kết quả dang dở ở độ sâu bị quá giờ, giữ best_action của độ sâu hoàn tất gần nhất
                    break

            if not self.last_decision_stage:
                self.last_decision_stage = f"Search: depth {self.last_completed_depth}"
            self.last_best_score = best_score
            return int(best_action)
        finally:
            self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0

    def _search_root(
        self,
        env: CaroEnv,
        candidates: list[int],
        depth: int,
        perspective: int,
        start_time: float,
        board_hash: int,
        keys_x: tuple[int, ...],
        keys_o: tuple[int, ...],
        turn_key: int,
        tt: dict[int, tuple[int, int, float, int | None]],
        killers: dict[int, list[int]],
    ) -> tuple[int, float]:
        """Thực hiện tìm kiếm ở tầng gốc cho một độ sâu `depth` cụ thể."""
        best_action = int(candidates[0])
        best_val = -math.inf
        best_ties: list[int] = []
        alpha = -math.inf
        beta = math.inf
        eps = self.tie_break_eps
        player_keys = keys_x if perspective == 1 else keys_o

        for action in candidates:
            if self.time_limit_sec is not None and (time.perf_counter() - start_time >= self.time_limit_sec):
                raise _SearchTimeout()

            act = int(action)
            next_hash = board_hash ^ player_keys[act] ^ turn_key if self.use_tt else 0

            env.step(act)
            self.last_search_count += 1
            try:
                child_val = self._minimax(
                    env=env,
                    depth=depth - 1,
                    is_maximizing=False,
                    alpha=_to_child_bound(alpha),
                    beta=_to_child_bound(beta),
                    perspective=perspective,
                    start_time=start_time,
                    board_hash=next_hash,
                    keys_x=keys_x,
                    keys_o=keys_o,
                    turn_key=turn_key,
                    tt=tt,
                    killers=killers,
                )
            finally:
                env.undo()

            val = _from_child_score(child_val)

            if eps > 0.0:
                if val > best_val + eps:
                    best_val = val
                    best_action = act
                    best_ties = [act]
                elif val >= best_val - eps:
                    if val > best_val:
                        best_val = val
                        best_action = act
                    best_ties.append(act)
            else:
                if val > best_val:
                    best_val = val
                    best_action = act

            if self.use_alpha_beta:
                eff_alpha = best_val - eps if eps > 0.0 else best_val
                if eff_alpha > alpha:
                    alpha = eff_alpha
                if beta <= alpha:
                    break

        if eps > 0.0 and len(best_ties) > 1:
            best_action = int(self.rng.choice(best_ties))

        return best_action, best_val

    @staticmethod
    def _record_killer(killers: dict[int, list[int]], depth: int, action: int) -> None:
        """Lưu tối đa 2 killer moves gần nhất gây cắt tỉa ở độ sâu `depth`."""
        slot = killers.get(depth)
        if slot is None:
            killers[depth] = [action]
        elif action not in slot:
            slot.insert(0, action)
            if len(slot) > 2:
                slot.pop()

    def _minimax(
        self,
        env: CaroEnv,
        depth: int,
        is_maximizing: bool,
        alpha: float,
        beta: float,
        perspective: int,
        start_time: float,
        board_hash: int,
        keys_x: tuple[int, ...],
        keys_o: tuple[int, ...],
        turn_key: int,
        tt: dict[int, tuple[int, int, float, int | None]],
        killers: dict[int, list[int]],
    ) -> float:
        """
        Hàm đệ quy Minimax dùng chung cho cả chế độ BẬT/TẮT Alpha-Beta và Transposition Table.
        """
        if (
            self.time_limit_sec is not None
            and (self.last_search_count & 31) == 0
            and (time.perf_counter() - start_time >= self.time_limit_sec)
        ):
            raise _SearchTimeout()

        # 1. Trạng thái kết thúc: tách biệt tuyệt đối với heuristic
        if env.is_terminal():
            res = env.result(perspective)
            if res > 0:
                return _WIN_SCORE
            if res < 0:
                return -_WIN_SCORE
            return 0.0

        # 2. Tra cứu Transposition Table (nếu bật)
        orig_alpha = alpha
        orig_beta = beta
        tt_best_move: int | None = None

        if self.use_tt:
            entry = tt.get(board_hash)
            if entry is not None:
                entry_depth, entry_flag, entry_val, entry_move = entry
                tt_best_move = entry_move
                if entry_depth >= depth:
                    if entry_flag == _TT_EXACT:
                        self.last_tt_hits += 1
                        return entry_val
                    if self.use_alpha_beta:
                        if entry_flag == _TT_LOWERBOUND:
                            if entry_val > alpha:
                                alpha = entry_val
                        elif entry_flag == _TT_UPPERBOUND:
                            if entry_val < beta:
                                beta = entry_val
                        if alpha >= beta:
                            self.last_tt_hits += 1
                            return entry_val

        # 3. Nút lá: đánh giá bàn cờ có xét quyền đi trước (consider_turn=True)
        if depth <= 0:
            leaf_val = evaluate_board(env, perspective, consider_turn=True)
            if self.use_tt and board_hash not in tt and len(tt) < self.tt_max_entries:
                tt[board_hash] = (0, _TT_EXACT, leaf_val, None)
            return leaf_val

        # 4. Sinh tập ứng viên
        if self.candidate_radius is None:
            candidates = env.legal_actions()
        else:
            candidates = get_candidate_actions(env, radius=self.candidate_radius)
            if not candidates:
                candidates = env.legal_actions()

        if not candidates:
            return evaluate_board(env, perspective, consider_turn=True)

        current_player = perspective if is_maximizing else -perspective

        if self.use_move_ordering:
            killer_moves = tuple(killers.get(depth, ()))
            candidates = self._order_moves(
                env,
                candidates,
                current_player,
                tt_best_move=tt_best_move,
                killer_moves=killer_moves,
            )
            if self.max_branch_inner is not None and len(candidates) > self.max_branch_inner:
                candidates = candidates[: self.max_branch_inner]

        player_keys = keys_x if current_player == 1 else keys_o
        best_move: int | None = None

        if is_maximizing:
            best_eval = -math.inf
            for action in candidates:
                act = int(action)
                next_hash = board_hash ^ player_keys[act] ^ turn_key if self.use_tt else 0

                env.step(act)
                self.last_search_count += 1
                try:
                    child_score = self._minimax(
                        env=env,
                        depth=depth - 1,
                        is_maximizing=False,
                        alpha=_to_child_bound(alpha),
                        beta=_to_child_bound(beta),
                        perspective=perspective,
                        start_time=start_time,
                        board_hash=next_hash,
                        keys_x=keys_x,
                        keys_o=keys_o,
                        turn_key=turn_key,
                        tt=tt,
                        killers=killers,
                    )
                finally:
                    env.undo()

                eval_score = _from_child_score(child_score)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = act

                if self.use_alpha_beta:
                    if eval_score > alpha:
                        alpha = eval_score
                    if beta <= alpha:
                        self._record_killer(killers, depth, act)
                        break
        else:
            best_eval = math.inf
            for action in candidates:
                act = int(action)
                next_hash = board_hash ^ player_keys[act] ^ turn_key if self.use_tt else 0

                env.step(act)
                self.last_search_count += 1
                try:
                    child_score = self._minimax(
                        env=env,
                        depth=depth - 1,
                        is_maximizing=True,
                        alpha=_to_child_bound(alpha),
                        beta=_to_child_bound(beta),
                        perspective=perspective,
                        start_time=start_time,
                        board_hash=next_hash,
                        keys_x=keys_x,
                        keys_o=keys_o,
                        turn_key=turn_key,
                        tt=tt,
                        killers=killers,
                    )
                finally:
                    env.undo()

                eval_score = _from_child_score(child_score)
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = act

                if self.use_alpha_beta:
                    if eval_score < beta:
                        beta = eval_score
                    if beta <= alpha:
                        self._record_killer(killers, depth, act)
                        break

        # 5. Lưu kết quả vào Transposition Table
        if self.use_tt:
            if not self.use_alpha_beta:
                flag = _TT_EXACT
            elif best_eval <= orig_alpha:
                flag = _TT_UPPERBOUND
            elif best_eval >= orig_beta:
                flag = _TT_LOWERBOUND
            else:
                flag = _TT_EXACT

            existing = tt.get(board_hash)
            if existing is not None:
                if depth >= existing[0]:
                    tt[board_hash] = (depth, flag, best_eval, best_move)
            elif len(tt) < self.tt_max_entries:
                tt[board_hash] = (depth, flag, best_eval, best_move)

        return best_eval

    @staticmethod
    def _move_priority(
        my_threat: dict[str, Any],
        opp_threat: dict[str, Any],
        center_bonus: float,
        is_tt_move: bool = False,
    ) -> float:
        """
        Tính điểm ưu tiên cho Move Ordering theo phân tầng chiến thuật:
          1. Thắng ngay của bản thân (`is_win_5`) luôn đứng đầu tuyệt đối.
          2. Nước đi tốt nhất từ Transposition Table (nếu có).
          3. Chặn thắng ngay của đối thủ.
          4. Tạo 4 mở / đòn kép thắng chắc (`open_4`, `double_four`, `four_three`).
          5. Chặn 4 mở / đòn kép thắng chắc của đối thủ.
          6. Tạo / chặn đòn kép đôi-3 (`double_three`).
          7. Tổng hợp điểm công + thủ + độ gần tâm bàn cờ.
        """
        if my_threat["is_win_5"]:
            return 10_000_000.0 + float(my_threat["score"])

        if is_tt_move:
            return 8_000_000.0 + float(my_threat["score"])

        my_score = float(my_threat["score"])
        opp_score = float(opp_threat["score"])

        if opp_threat["is_win_5"]:
            return 5_000_000.0 + opp_score + 0.5 * my_score

        if my_threat["open_4"] > 0 or my_threat.get("fork_type") in ("double_four", "four_three"):
            return 2_000_000.0 + my_score + 0.5 * opp_score

        if opp_threat["open_4"] > 0 or opp_threat.get("fork_type") in ("double_four", "four_three"):
            return 1_000_000.0 + opp_score + 0.5 * my_score

        if my_threat.get("fork_type") == "double_three":
            return 500_000.0 + my_score + 0.5 * opp_score

        if opp_threat.get("fork_type") == "double_three":
            return 300_000.0 + opp_score + 0.5 * my_score

        return my_score + 0.95 * opp_score + center_bonus

    def _order_moves_from_threats(
        self,
        env: CaroEnv,
        candidates: list[int],
        my_threats: dict[int, dict[str, Any]],
        opp_threats: dict[int, dict[str, Any]],
        tt_best_move: int | None = None,
    ) -> list[int]:
        """Sắp xếp danh sách ứng viên từ bảng threat đã tính sẵn."""
        board_size = env.board_size
        center = (board_size - 1) / 2.0
        scored_moves: list[tuple[float, int]] = []

        for action in candidates:
            act = int(action)
            r, c = divmod(act, board_size)
            center_bonus = 5.0 - 0.5 * (abs(r - center) + abs(c - center))
            priority = self._move_priority(
                my_threats[act],
                opp_threats[act],
                center_bonus,
                is_tt_move=(tt_best_move is not None and act == tt_best_move),
            )
            scored_moves.append((priority, act))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [act for _, act in scored_moves]

    def _order_moves(
        self,
        env: CaroEnv,
        candidates: list[int],
        current_player: int,
        tt_best_move: int | None = None,
        killer_moves: tuple[int, ...] = (),
    ) -> list[int]:
        """
        Sắp xếp nhanh danh sách nước đi ứng viên theo phân tầng chiến thuật
        kết hợp Tấn công, Phòng thủ, TT Best Move và Killer Moves (đọc trực tiếp
        từ mảng phẳng Python để giảm chi phí cấp phát ở các tầng sâu).
        """
        opponent = -current_player
        size = env.board_size
        win_len = env.win_length
        center = (size - 1) / 2.0
        flat = env.board.ravel().tolist()
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        scored_moves: list[tuple[float, int]] = []

        for action in candidates:
            act = int(action)
            r, c = divmod(act, size)

            my_win = False
            opp_win = False
            my_o4 = my_h4 = my_b4 = my_o3 = my_bo3 = my_h3 = my_o2 = 0
            opp_o4 = opp_h4 = opp_b4 = opp_o3 = opp_bo3 = opp_h3 = opp_o2 = 0

            for dr, dc in directions:
                # --- Quét nhanh phía + và - cho cả 2 màu ---
                # Kiểm tra trước ô kề đầu tiên để bỏ qua ngay các phương trống hoàn toàn
                nr1, nc1 = r + dr, c + dc
                pr1, pc1 = r - dr, c - dc
                v_pos1 = flat[nr1 * size + nc1] if (0 <= nr1 < size and 0 <= nc1 < size) else 2
                v_neg1 = flat[pr1 * size + pc1] if (0 <= pr1 < size and 0 <= pc1 < size) else 2

                # Kiểm tra nhanh xem có ít nhất 1 quân trong khoảng cách 2 trên phương này không
                if v_pos1 == 0:
                    nr2, nc2 = nr1 + dr, nc1 + dc
                    v_pos2 = flat[nr2 * size + nc2] if (0 <= nr2 < size and 0 <= nc2 < size) else 2
                else:
                    v_pos2 = 2
                if v_neg1 == 0:
                    pr2, pc2 = pr1 - dr, pc1 - dc
                    v_neg2 = flat[pr2 * size + pc2] if (0 <= pr2 < size and 0 <= pc2 < size) else 2
                else:
                    v_neg2 = 2

                # Đánh giá cho `current_player` nếu có quân gần trên phương này
                if v_pos1 == current_player or v_neg1 == current_player or v_pos2 == current_player or v_neg2 == current_player:
                    pos = 0
                    tr, tc = nr1, nc1
                    while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == current_player:
                        pos += 1
                        tr += dr
                        tc += dc
                    pos_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0
                    pos_gap = 0
                    pos_gap_open = False
                    if pos_open:
                        tr += dr
                        tc += dc
                        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == current_player:
                            pos_gap += 1
                            tr += dr
                            tc += dc
                        pos_gap_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

                    neg = 0
                    tr, tc = pr1, pc1
                    while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == current_player:
                        neg += 1
                        tr -= dr
                        tc -= dc
                    neg_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0
                    neg_gap = 0
                    neg_gap_open = False
                    if neg_open:
                        tr -= dr
                        tc -= dc
                        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == current_player:
                            neg_gap += 1
                            tr -= dr
                            tc -= dc
                        neg_gap_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

                    total = 1 + pos + neg
                    if total >= win_len:
                        my_win = True
                        break
                    open_ends = (1 if pos_open else 0) + (1 if neg_open else 0)
                    if total == win_len - 1:
                        if open_ends == 2:
                            my_o4 += 1
                        elif open_ends == 1:
                            my_h4 += 1
                    elif total == win_len - 2:
                        if open_ends == 2:
                            my_o3 += 1
                        elif open_ends == 1:
                            my_h3 += 1
                    elif total == win_len - 3 and open_ends == 2:
                        my_o2 += 1

                    if pos_gap > 0 or neg_gap > 0:
                        best_g = pos_gap if pos_gap >= neg_gap else neg_gap
                        if total + best_g >= win_len - 1:
                            my_b4 += 1
                        elif total + best_g == win_len - 2:
                            if (pos_gap > 0 and pos_gap_open and neg_open) or (neg_gap > 0 and neg_gap_open and pos_open):
                                my_bo3 += 1

                # Đánh giá cho `opponent` nếu có quân gần trên phương này
                if v_pos1 == opponent or v_neg1 == opponent or v_pos2 == opponent or v_neg2 == opponent:
                    pos = 0
                    tr, tc = nr1, nc1
                    while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == opponent:
                        pos += 1
                        tr += dr
                        tc += dc
                    pos_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0
                    pos_gap = 0
                    pos_gap_open = False
                    if pos_open:
                        tr += dr
                        tc += dc
                        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == opponent:
                            pos_gap += 1
                            tr += dr
                            tc += dc
                        pos_gap_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

                    neg = 0
                    tr, tc = pr1, pc1
                    while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == opponent:
                        neg += 1
                        tr -= dr
                        tc -= dc
                    neg_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0
                    neg_gap = 0
                    neg_gap_open = False
                    if neg_open:
                        tr -= dr
                        tc -= dc
                        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == opponent:
                            neg_gap += 1
                            tr -= dr
                            tc -= dc
                        neg_gap_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

                    total = 1 + pos + neg
                    if total >= win_len:
                        opp_win = True
                    else:
                        open_ends = (1 if pos_open else 0) + (1 if neg_open else 0)
                        if total == win_len - 1:
                            if open_ends == 2:
                                opp_o4 += 1
                            elif open_ends == 1:
                                opp_h4 += 1
                        elif total == win_len - 2:
                            if open_ends == 2:
                                opp_o3 += 1
                            elif open_ends == 1:
                                opp_h3 += 1
                        elif total == win_len - 3 and open_ends == 2:
                            opp_o2 += 1

                        if pos_gap > 0 or neg_gap > 0:
                            best_g = pos_gap if pos_gap >= neg_gap else neg_gap
                            if total + best_g >= win_len - 1:
                                opp_b4 += 1
                            elif total + best_g == win_len - 2:
                                if (pos_gap > 0 and pos_gap_open and neg_open) or (neg_gap > 0 and neg_gap_open and pos_open):
                                    opp_bo3 += 1

            my_score = (
                my_o4 * 50000.0
                + my_h4 * 8000.0
                + my_b4 * 7000.0
                + my_o3 * 5000.0
                + my_bo3 * 3600.0
                + my_h3 * 600.0
                + my_o2 * 300.0
            )
            opp_score = (
                opp_o4 * 50000.0
                + opp_h4 * 8000.0
                + opp_b4 * 7000.0
                + opp_o3 * 5000.0
                + opp_bo3 * 3600.0
                + opp_h3 * 600.0
                + opp_o2 * 300.0
            )

            if my_win:
                priority = 10_000_000.0 + my_score
            elif tt_best_move is not None and act == tt_best_move:
                priority = 8_000_000.0 + my_score
            elif opp_win:
                priority = 5_000_000.0 + opp_score + 0.5 * my_score
            else:
                my_f4 = my_o4 + my_h4 + my_b4
                my_l3 = my_o3 + my_bo3
                opp_f4 = opp_o4 + opp_h4 + opp_b4
                opp_l3 = opp_o3 + opp_bo3

                if my_o4 > 0 or my_f4 >= 2 or (my_f4 >= 1 and my_l3 >= 1):
                    priority = 2_000_000.0 + my_score + 0.5 * opp_score
                elif opp_o4 > 0 or opp_f4 >= 2 or (opp_f4 >= 1 and opp_l3 >= 1):
                    priority = 1_000_000.0 + opp_score + 0.5 * my_score
                elif my_l3 >= 2:
                    priority = 500_000.0 + my_score + 0.5 * opp_score
                elif opp_l3 >= 2:
                    priority = 300_000.0 + opp_score + 0.5 * my_score
                else:
                    center_bonus = 5.0 - 0.5 * (abs(r - center) + abs(c - center))
                    killer_bonus = 40_000.0 if (killer_moves and act in killer_moves) else 0.0
                    priority = my_score + 0.95 * opp_score + center_bonus + killer_bonus

            scored_moves.append((priority, act))

        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return [act for _, act in scored_moves]
