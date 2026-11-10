from __future__ import annotations
import math
import time
from typing import TYPE_CHECKING, Tuple

from agents.base_agent import BaseAgent
from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions

if TYPE_CHECKING:
    from caro_env import CaroEnv


class MinimaxAgent(BaseAgent):
    """
    Tác tử Cấp 3: Sử dụng thuật toán Minimax tích hợp Alpha-Beta Pruning,
    Move Ordering, Candidate Action Pruning và Kiểm soát thời gian suy nghĩ.
    """

    def __init__(
        self,
        name: str = "MinimaxAgent",
        max_depth: int = 3,
        depth: int | None = None,  # Alias hỗ trợ gọi depth hoặc max_depth
        use_alpha_beta: bool = True,
        use_move_ordering: bool = True,
        candidate_radius: int | None = 2,
        time_limit_sec: float | None = 1.0,  # Truyền None để TẮT ngắt thời gian
    ):
        super().__init__(name=name)
        self.max_depth = depth if depth is not None else max_depth
        self.use_alpha_beta = use_alpha_beta
        self.use_move_ordering = use_move_ordering
        self.candidate_radius = candidate_radius
        self.time_limit_sec = time_limit_sec

        # Thuộc tính đo lường hiệu năng phục vụ thực nghiệm Ablation Study
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0

    def select_action(self, env: CaroEnv) -> int:
        """
        Lựa chọn nước đi tối ưu nhất từ môi trường env hiện tại.
        Đảm bảo không biến đổi trạng thái của môi trường gốc.
        """
        start_time = time.perf_counter()
        self.last_search_count = 0

        legal_actions = env.legal_actions()
        if not legal_actions:
            raise ValueError("Không có nước đi hợp lệ nào trên bàn cờ.")

        candidates = get_candidate_actions(env, radius=self.candidate_radius)
        if not candidates:
            candidates = legal_actions

        # Sắp xếp danh sách candidates chuẩn từ gốc nếu use_move_ordering được bật
        if self.use_move_ordering:
            candidates = self._order_moves(env, candidates, env.current_player)

        best_action = candidates[0]
        perspective = env.current_player
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in candidates:
            # 1. Thử đánh nước đi
            env.step(action)
            self.last_search_count += 1

            # 2. Duyệt đệ quy sang lượt của đối thủ (is_maximizing=False)
            val = self._minimax(
                env=env,
                depth=self.max_depth - 1,
                is_maximizing=False,
                alpha=alpha,
                beta=beta,
                perspective=perspective,
                start_time=start_time,
            )

            # 3. Hoàn tác nước đi (backtrack)
            env.undo()

            # Giữ nguyên tiêu chí chọn nước đi đầu tiên nếu điểm hòa nhau (tie-breaking)
            if val > best_val:
                best_val = val
                best_action = action

            # Nếu BẬT Alpha-Beta thì thực hiện cập nhật alpha và kiểm tra cắt tỉa
            if self.use_alpha_beta:
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break

        # Ghi nhận thời gian suy nghĩ (miligiây)
        self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0
        return best_action

    def _minimax(self, env: CaroEnv, depth: int, is_maximizing: bool, alpha: float, beta: float, perspective: int, start_time: float) -> float:
        """
        Hàm đệ quy Minimax dùng chung cho cả 2 chế độ BẬT/TẮT Alpha-Beta,
        đảm bảo 100% nhất quán về nước đi thu được.
        """
        # Kiểm tra điều kiện giới hạn thời gian (chỉ kiểm tra nếu time_limit_sec được thiết lập)
        is_timeout = (
            self.time_limit_sec is not None
            and (time.perf_counter() - start_time >= self.time_limit_sec)
        )

        # Điều kiện dừng đệ quy
        if depth == 0 or env.is_terminal() or is_timeout:
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

                eval_score = self._minimax(
                    env=env,
                    depth=depth - 1,
                    is_maximizing=False,
                    alpha=alpha,
                    beta=beta,
                    perspective=perspective,
                    start_time=start_time,
                )
                env.undo()

                if eval_score > max_eval:
                    max_eval = eval_score

                # Chỉ cắt tỉa khi cờ use_alpha_beta = True
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Cắt nhánh Beta
            return max_eval
        else:
            min_eval = math.inf
            for action in candidates:
                env.step(action)
                self.last_search_count += 1

                eval_score = self._minimax(
                    env=env,
                    depth=depth - 1,
                    is_maximizing=True,
                    alpha=alpha,
                    beta=beta,
                    perspective=perspective,
                    start_time=start_time,
                )
                env.undo()

                if eval_score < min_eval:
                    min_eval = eval_score

                # Chỉ cắt tỉa khi cờ use_alpha_beta = True
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Cắt nhánh Alpha
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
            # Ưu tiên kết hợp giữa tấn công của mình và phòng thủ chặn đối thủ
            combined_score = my_threat["score"] + 1.2 * opp_threat["score"]
            scored_moves.append((combined_score, action))

        # Sắp xếp giảm dần theo điểm số đe dọa
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [action for _, action in scored_moves]