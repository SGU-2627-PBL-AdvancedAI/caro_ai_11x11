from __future__ import annotations
import math
import time
from typing import TYPE_CHECKING, Tuple

from agents.base_agent import BaseAgent
from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions

if TYPE_CHECKING:
    from caro_env import CaroEnv


class _SearchTimeout(Exception):
    """Ngoại lệ nội bộ dùng để ngắt ngay lập tức khi vượt quá time_limit_sec."""
    pass


class MinimaxAgent(BaseAgent):
    """
    Tác tử Cấp 3: Sử dụng thuật toán Minimax tích hợp Alpha-Beta Pruning,
    Move Ordering, Candidate Action Pruning, Iterative Deepening và Kiểm soát thời gian.
    """

    def __init__(
        self,
        name: str = "MinimaxAgent",
        max_depth: int = 3,
        use_alpha_beta: bool = True,
        use_move_ordering: bool = True,
        candidate_radius: int | None = 2,
        time_limit_sec: float | None = 1.0,  # Truyền None để chạy cố định ở max_depth
    ):
        super().__init__(name=name)
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.use_move_ordering = use_move_ordering
        self.candidate_radius = candidate_radius
        self.time_limit_sec = time_limit_sec

        # Thuộc tính đo lường hiệu năng phục vụ thực nghiệm Ablation Study
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_completed_depth: int = 0

    def select_action(self, env: CaroEnv) -> int:
        """
        Lựa chọn nước đi tối ưu nhất từ môi trường env hiện tại.
        Đảm bảo không biến đổi trạng thái của môi trường gốc.
        """
        start_time = time.perf_counter()
        self.last_search_count = 0
        self.last_completed_depth = 0

        legal_actions = env.legal_actions()
        if not legal_actions:
            raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ.")

        candidates = get_candidate_actions(env, radius=self.candidate_radius)
        if not candidates:
            candidates = legal_actions

        perspective = env.current_player

        # 1. Immediate Win Shortcut: Kiểm tra nước thắng ngay lập tức tại gốc
        for action in candidates:
            if evaluate_cell_threat(env, action, perspective)["is_win_5"]:
                self.last_search_count = 1
                self.last_completed_depth = 1
                self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0
                return action

        # Sắp xếp danh sách candidates chuẩn từ gốc nếu use_move_ordering được bật
        if self.use_move_ordering:
            candidates = self._order_moves(env, candidates, perspective)

        best_action = candidates[0]

        # Nếu tắt giới hạn thời gian (dùng cho Ablation so sánh số nút duyệt ở độ sâu cố định),
        # chạy trực tiếp 1 lượt tại max_depth. Nếu có giới hạn thời gian, dùng Iterative Deepening.
        start_depth = self.max_depth if self.time_limit_sec is None else 1

        for current_depth in range(start_depth, self.max_depth + 1):
            # Đưa best_action của độ sâu trước lên đầu để tối ưu cắt tỉa Alpha-Beta
            if current_depth > start_depth and best_action in candidates:
                ordered_candidates = [best_action] + [a for a in candidates if a != best_action]
            else:
                ordered_candidates = candidates

            try:
                depth_best_action = self._search_root(
                    env=env,
                    candidates=ordered_candidates,
                    depth=current_depth,
                    perspective=perspective,
                    start_time=start_time,
                )
                best_action = depth_best_action
                self.last_completed_depth = current_depth
            except _SearchTimeout:
                # Hủy bỏ kết quả dang dở của độ sâu bị quá giờ, giữ lại best_action của độ sâu trước
                break

        self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0
        return best_action

    def _search_root(self, env: CaroEnv, candidates: list[int], depth: int, perspective: int, start_time: float) -> int:
        """Thực hiện tìm kiếm ở tầng gốc cho một độ sâu `depth` cụ thể."""
        best_action = candidates[0]
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in candidates:
            env.step(action)
            self.last_search_count += 1
            try:
                val = self._minimax(
                    env=env,
                    depth=depth - 1,
                    is_maximizing=False,
                    alpha=alpha,
                    beta=beta,
                    perspective=perspective,
                    start_time=start_time,
                )
            finally:
                # Luôn hoàn tác bàn cờ kể cả khi bị ngắt bởi _SearchTimeout
                env.undo()

            if val > best_val:
                best_val = val
                best_action = action

            if self.use_alpha_beta:
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break

        return best_action

    def _minimax(self, env: CaroEnv, depth: int, is_maximizing: bool, alpha: float, beta: float, perspective: int, start_time: float) -> float:
        """
        Hàm đệ quy Minimax dùng chung cho cả 2 chế độ BẬT/TẮT Alpha-Beta.
        """
        if self.time_limit_sec is not None and (time.perf_counter() - start_time >= self.time_limit_sec):
            raise _SearchTimeout()

        # Điều kiện dừng đệ quy: ưu tiên thắng sớm / thua muộn bằng cách cộng/trừ theo `depth`
        if env.is_terminal():
            res = env.result(perspective)
            if res > 0:
                return 100000.0 + depth * 100.0
            elif res < 0:
                return -100000.0 - depth * 100.0
            return 0.0

        if depth <= 0:
            return evaluate_board(env, perspective)

        candidates = get_candidate_actions(env, radius=self.candidate_radius)
        if not candidates:
            return evaluate_board(env, perspective)

        current_player = perspective if is_maximizing else -perspective

        if self.use_move_ordering:
            candidates = self._order_moves(env, candidates, current_player)

        if is_maximizing:
            max_eval = -math.inf
            for action in candidates:
                env.step(action)
                self.last_search_count += 1
                try:
                    eval_score = self._minimax(
                        env=env,
                        depth=depth - 1,
                        is_maximizing=False,
                        alpha=alpha,
                        beta=beta,
                        perspective=perspective,
                        start_time=start_time,
                    )
                finally:
                    env.undo()

                if eval_score > max_eval:
                    max_eval = eval_score

                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = math.inf
            for action in candidates:
                env.step(action)
                self.last_search_count += 1
                try:
                    eval_score = self._minimax(
                        env=env,
                        depth=depth - 1,
                        is_maximizing=True,
                        alpha=alpha,
                        beta=beta,
                        perspective=perspective,
                        start_time=start_time,
                    )
                finally:
                    env.undo()

                if eval_score < min_eval:
                    min_eval = eval_score

                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            return min_eval

    def _order_moves(self, env: CaroEnv, candidates: list[int], current_player: int) -> list[int]:
        """
        Sắp xếp danh sách nước đi ứng viên theo điểm kết hợp Tấn công và Phòng thủ.
        """
        opponent = -current_player
        scored_moves: list[Tuple[float, int]] = []

        for action in candidates:
            my_threat = evaluate_cell_threat(env, action, current_player)
            opp_threat = evaluate_cell_threat(env, action, opponent)
            combined_score = my_threat["score"] + 1.2 * opp_threat["score"]
            scored_moves.append((combined_score, action))

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [action for _, action in scored_moves]