from __future__ import annotations

import random
import time
from typing import Any

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


class RuleBasedAgent(BaseAgent):
    """
    Tác tử Cấp 2: Hybrid Rule-Based / Heuristic / Greedy Look-Ahead Agent cho Caro 11x11.

    Thứ tự quyết định 7 tầng:
      1. Thắng ngay nếu tạo được chuỗi >= 5 quân (`is_win_5`).
      2. Chặn nước thắng ngay của đối thủ.
      3. Tạo chuỗi 4 mở hai đầu (`open_4`) hoặc đòn kép thắng chắc (`double_four`, `four_three`).
      4. Chặn đối thủ tạo chuỗi 4 mở hai đầu hoặc đòn kép thắng chắc (`double_four`, `four_three`).
      5. Tạo đòn kép đôi-3 (`double_three`).
      6. Chặn đối thủ tạo đòn kép đôi-3 (`double_three`).
      7. Greedy Heuristic Look-Ahead 3 ply: AI (m1) -> Đối thủ (m2) -> AI (m3),
         kết hợp lượng giá bàn cờ có xét quyền đi trước (`consider_turn=True`).

    Thuật toán vẫn giữ bản chất Heuristic Look-Ahead tham lam (không sử dụng cây Minimax / Alpha-Beta).
    """

    def __init__(
        self,
        name: str = "Rule-Based Heuristic Greedy Look-Ahead",
        candidate_radius: int = 2,
        root_width: int = 8,
        defense_weight: float = 0.95,
        center_weight: float = 2.0,
        seed: int | None = None,
        tie_break_eps: float = 1e-6,
    ):
        super().__init__(name)

        self.candidate_radius = int(candidate_radius)
        self.root_width = max(1, int(root_width))
        self.defense_weight = float(defense_weight)
        self.center_weight = float(center_weight)
        self.seed = seed
        self.tie_break_eps = float(tie_break_eps)
        self.rng = random.Random(seed)

        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_search_depth: int = 0
        self.last_decision_stage: str = ""

    def _candidates(self, env, radius: int | None = None) -> list[int]:
        """Sinh các nước đi ứng viên quanh những quân đã xuất hiện."""
        use_radius = self.candidate_radius if radius is None else radius
        return get_candidate_actions(env, radius=use_radius)

    def _evaluate_candidates(
        self,
        env,
        actions: list[int],
        player: int,
    ) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        """
        Đánh giá một lần duy nhất tiềm năng công (`player`) và thủ (`-player`)
        cho toàn bộ tập `actions` để tránh lặp lại `evaluate_cell_threat` qua nhiều tầng.
        """
        my_threats: dict[int, dict[str, Any]] = {}
        opp_threats: dict[int, dict[str, Any]] = {}
        opponent = -player

        for action in actions:
            act = int(action)
            self.last_search_count += 1
            my_threats[act] = evaluate_cell_threat(env, act, player)
            opp_threats[act] = evaluate_cell_threat(env, act, opponent)

        return my_threats, opp_threats

    def _score_action(
        self,
        action: int,
        board_size: int,
        my_threat: dict[str, Any],
        opp_threat: dict[str, Any],
    ) -> float:
        """Tính điểm Heuristic tổng hợp: Công + defense_weight * Thủ + Thưởng tâm bàn cờ."""
        center = board_size // 2
        r, c = divmod(action, board_size)
        distance = abs(r - center) + abs(c - center)
        center_bonus = max(0, board_size - distance) * self.center_weight
        return (
            float(my_threat["score"])
            + self.defense_weight * float(opp_threat["score"])
            + center_bonus
        )

    def _pick_best_from_scored(self, scored: list[tuple[float, int]]) -> int:
        """
        Chọn nước đi điểm cao nhất từ danh sách `(score, action)` đã sắp giảm dần,
        phá hòa ngẫu nhiên có kiểm soát (theo `self.rng`) trong biên độ `tie_break_eps`.
        """
        best_score = scored[0][0]
        tied = [
            action
            for score, action in scored
            if best_score - score <= self.tie_break_eps
        ]
        if len(tied) == 1:
            return int(tied[0])
        return int(self.rng.choice(tied))

    def _rank(
        self,
        env,
        actions: list[int],
        player: int,
        limit: int | None = None,
        my_threats: dict[int, dict[str, Any]] | None = None,
        opp_threats: dict[int, dict[str, Any]] | None = None,
    ) -> list[tuple[float, int]]:
        """
        Xếp hạng nước đi theo công thức:
            H(a) = Attack(a) + defense_weight * Defense(a) + CenterBonus(a)
        Tái sử dụng bảng `my_threats` / `opp_threats` nếu đã tính trước.
        """
        n = env.board_size
        scored: list[tuple[float, int]] = []

        for action in actions:
            act = int(action)
            if my_threats is not None and act in my_threats:
                mt = my_threats[act]
            else:
                self.last_search_count += 1
                mt = evaluate_cell_threat(env, act, player)

            if opp_threats is not None and act in opp_threats:
                ot = opp_threats[act]
            else:
                ot = evaluate_cell_threat(env, act, -player)

            total_score = self._score_action(act, n, mt, ot)
            scored.append((total_score, act))

        # Sắp giảm dần theo điểm; xáo nhẹ nhóm bằng điểm qua khóa ngẫu nhiên có kiểm soát
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        if limit is not None:
            scored = scored[:limit]

        return scored

    def _select_greedy_reply(self, env, player: int) -> int | None:
        """
        Chọn 1 nước đáp trả tham lam hợp lý nhất cho `player` trong mô phỏng Look-Ahead:
          1. Thắng ngay (`is_win_5`).
          2. Chặn đối thủ thắng ngay.
          3. Tạo chuỗi 4 mở hai đầu hoặc đòn kép `double_four`/`four_three`.
          4. Nước có điểm Heuristic `_rank` cao nhất.
        """
        actions = self._candidates(env)
        if not actions:
            return None

        my_threats, opp_threats = self._evaluate_candidates(env, actions, player)

        win_moves = [a for a, mt in my_threats.items() if mt["is_win_5"]]
        if win_moves:
            return int(win_moves[0])

        block_wins = [a for a, ot in opp_threats.items() if ot["is_win_5"]]
        if block_wins:
            ranked = self._rank(env, block_wins, player, my_threats=my_threats, opp_threats=opp_threats)
            return self._pick_best_from_scored(ranked)

        lethal_fours = [
            a
            for a, mt in my_threats.items()
            if mt["open_4"] > 0 or mt["fork_type"] in ("double_four", "four_three")
        ]
        if lethal_fours:
            ranked = self._rank(env, lethal_fours, player, my_threats=my_threats, opp_threats=opp_threats)
            return self._pick_best_from_scored(ranked)

        ranked = self._rank(env, actions, player, my_threats=my_threats, opp_threats=opp_threats)
        return self._pick_best_from_scored(ranked)

    @staticmethod
    def _terminal_value(env, perspective: int, ply: int) -> float | None:
        """Trả về điểm kết thúc có ưu tiên thắng sớm / thua muộn theo `ply`."""
        if not env.is_terminal():
            return None

        winner = getattr(env, "winner", 0)
        if winner == perspective:
            return 10_000_000.0 - ply * 1_000.0
        if winner == -perspective:
            return -10_000_000.0 + ply * 1_000.0
        return 0.0

    def _lookahead_three_ply(
        self,
        env,
        player: int,
        actions: list[int],
        my_threats: dict[int, dict[str, Any]],
        opp_threats: dict[int, dict[str, Any]],
    ) -> int:
        """
        Greedy Heuristic-Guided Look-Ahead 3 ply:
            AI (m1) -> Đối thủ (m2) -> AI (m3).

        Cải tiến so với bản gốc:
        - Chỉ `clone()` môi trường gốc đúng 1 lần rồi dùng `step()` / `undo()` trong suốt look-ahead.
        - Phản ứng `m2` và `m3` tuân thủ ưu tiên thắng/chặn ngay trước khi chọn greedy.
        - Đánh giá trạng thái cuối bằng `evaluate_board(..., consider_turn=True)` để nhận diện
          nguy cơ đối thủ có nước thắng/đe dọa ở ply 4.
        - Phá hòa ngẫu nhiên có kiểm soát khi nhiều nhánh gốc đạt cùng giá trị.
        """
        if not actions:
            raise ValueError("Không có nước đi ứng viên cho Look-Ahead.")

        root_ranked = self._rank(
            env,
            actions,
            player,
            limit=self.root_width,
            my_threats=my_threats,
            opp_threats=opp_threats,
        )
        if not root_ranked:
            raise ValueError("Không thể xếp hạng nước đi ứng viên.")

        sim_env = env.clone()
        evaluated_roots: list[tuple[float, float, int]] = []

        for root_h_score, m1 in root_ranked:
            sim_env.step(m1)
            try:
                term_1 = self._terminal_value(sim_env, player, ply=1)
                if term_1 is not None:
                    value = term_1
                else:
                    # Ply 2: Đối thủ chọn nước phản ứng tham lam tốt nhất
                    m2 = self._select_greedy_reply(sim_env, -player)
                    if m2 is None:
                        value = evaluate_board(sim_env, player, consider_turn=True)
                    else:
                        sim_env.step(m2)
                        try:
                            term_2 = self._terminal_value(sim_env, player, ply=2)
                            if term_2 is not None:
                                value = term_2
                            else:
                                # Ply 3: AI chọn nước đáp trả tham lam tốt nhất
                                m3 = self._select_greedy_reply(sim_env, player)
                                if m3 is None:
                                    value = evaluate_board(sim_env, player, consider_turn=True)
                                else:
                                    sim_env.step(m3)
                                    try:
                                        term_3 = self._terminal_value(sim_env, player, ply=3)
                                        if term_3 is not None:
                                            value = term_3
                                        else:
                                            value = evaluate_board(sim_env, player, consider_turn=True)
                                    finally:
                                        sim_env.undo()
                        finally:
                            sim_env.undo()
            finally:
                sim_env.undo()

            evaluated_roots.append((value, root_h_score, m1))

        # Ưu tiên giá trị sau 3 ply; nếu bằng nhau, ưu tiên điểm heuristic gốc cao hơn
        evaluated_roots.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_val, best_root_h, _ = evaluated_roots[0]
        tied_best = [
            action
            for val, rh, action in evaluated_roots
            if (best_val - val <= self.tie_break_eps)
            and (best_root_h - rh <= self.tie_break_eps)
        ]
        if len(tied_best) == 1:
            return int(tied_best[0])
        return int(self.rng.choice(tied_best))

    def select_action(self, env) -> int:
        """Điều phối 7 tầng ưu tiên chiến thuật và trả về nước đi hợp lệ (0..N*N-1)."""
        started = time.perf_counter()

        self.last_search_count = 0
        self.last_search_depth = 0
        self.last_decision_stage = ""

        try:
            legal = env.legal_actions()
            if not legal:
                raise ValueError("Không còn nước đi hợp lệ.")

            player = env.current_player
            candidates = self._candidates(env)
            if not candidates:
                candidates = legal

            if len(candidates) == 1:
                self.last_decision_stage = "Rule-0: forced or opening center"
                self.last_search_depth = 1
                self.last_search_count = 1
                return int(candidates[0])

            # Tính 1 lần duy nhất toàn bộ điểm đe dọa công/thủ trên tập ứng viên
            my_threats, opp_threats = self._evaluate_candidates(env, candidates, player)

            # ---------------------------------------------------------
            # Tầng 1: AI thắng ngay nếu có thể (chuỗi >= 5 quân).
            # ---------------------------------------------------------
            own_wins = [a for a, mt in my_threats.items() if mt["is_win_5"]]
            if own_wins:
                ranked = self._rank(env, own_wins, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-1: immediate win"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            # ---------------------------------------------------------
            # Tầng 2: Chặn nước thắng ngay của đối thủ.
            # ---------------------------------------------------------
            opponent_wins = [a for a, ot in opp_threats.items() if ot["is_win_5"]]
            if opponent_wins:
                ranked = self._rank(env, opponent_wins, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-2: block immediate win"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            # ---------------------------------------------------------
            # Tầng 3: Tạo chuỗi 4 mở hai đầu hoặc đòn kép thắng chắc (double_four, four_three).
            # ---------------------------------------------------------
            own_open_four = [a for a, mt in my_threats.items() if mt["open_4"] > 0]
            if own_open_four:
                ranked = self._rank(env, own_open_four, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-3: create open four"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            own_winning_fork = [
                a for a, mt in my_threats.items() if mt["fork_type"] in ("double_four", "four_three")
            ]
            if own_winning_fork:
                ranked = self._rank(env, own_winning_fork, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-3b: create winning fork"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            # ---------------------------------------------------------
            # Tầng 4: Chặn đối thủ tạo chuỗi 4 mở hai đầu hoặc đòn kép thắng chắc.
            # ---------------------------------------------------------
            opp_lethal_four = [
                a
                for a, ot in opp_threats.items()
                if ot["open_4"] > 0 or ot["fork_type"] in ("double_four", "four_three")
            ]
            if opp_lethal_four:
                ranked = self._rank(env, opp_lethal_four, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-4: block open four or winning fork"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            # ---------------------------------------------------------
            # Tầng 5: Tạo đòn kép đôi-3 (double_three).
            # ---------------------------------------------------------
            own_double_three = [a for a, mt in my_threats.items() if mt["fork_type"] == "double_three"]
            if own_double_three:
                ranked = self._rank(env, own_double_three, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-5: create double three"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            # ---------------------------------------------------------
            # Tầng 6: Chặn đối thủ tạo đòn kép đôi-3 (double_three).
            # ---------------------------------------------------------
            opp_double_three = [a for a, ot in opp_threats.items() if ot["fork_type"] == "double_three"]
            if opp_double_three:
                ranked = self._rank(env, opp_double_three, player, my_threats=my_threats, opp_threats=opp_threats)
                self.last_decision_stage = "Rule-6: block double three"
                self.last_search_depth = 1
                return self._pick_best_from_scored(ranked)

            # ---------------------------------------------------------
            # Tầng 7: Greedy Heuristic-Guided Look-Ahead 3 ply.
            # ---------------------------------------------------------
            action = self._lookahead_three_ply(
                env,
                player,
                candidates,
                my_threats=my_threats,
                opp_threats=opp_threats,
            )
            self.last_decision_stage = "Rule-7: greedy heuristic look-ahead"
            self.last_search_depth = 3
            return int(action)
        finally:
            self.last_move_time_ms = (time.perf_counter() - started) * 1000.0
