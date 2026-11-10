from __future__ import annotations

import time


from agents.base_agent import BaseAgent

from heuristic import (
    get_candidate_actions,
    evaluate_cell_threat,
    evaluate_board,
)


class RuleBasedAgent(BaseAgent):
    """
    Hybrid Rule-Based / Heuristic / Greedy Look-Ahead Agent cho Caro 11x11.

    Thứ tự quyết định:
      1. Thắng ngay nếu có thể.
      2. Chặn nước thắng ngay của đối thủ.
      3. Tạo chuỗi 4 mở hai đầu.
      4. Chặn cơ hội tạo chuỗi 4 mở hai đầu của đối thủ.
      5. Nếu không có tình huống ưu tiên, sử dụng Greedy Look-Ahead 3 ply.

    Look-Ahead mô phỏng:
      AI (m1) -> Đối thủ (m2) -> AI (m3).

    Ở mỗi ply, người chơi chọn nước đi có điểm Heuristic cao nhất.
    Thuật toán KHÔNG sử dụng Minimax hoặc phép luân phiên Max-Min.
    """

    def __init__(
        self,
        name="Rule-Based Heuristic Greedy Look-Ahead",
        candidate_radius=2,
        root_width=8,
        defense_weight=0.9,
        center_weight=2.0,
    ):
        super().__init__(name)

        self.candidate_radius = candidate_radius
        self.root_width = root_width
        self.defense_weight = defense_weight
        self.center_weight = center_weight

        self.last_move_time_ms = 0.0
        self.last_search_count = 0
        self.last_search_depth = 0
        self.last_decision_stage = ""

    def _candidates(self, env):
        """Sinh các nước đi ứng viên quanh những quân đã xuất hiện."""
        return get_candidate_actions(
            env,
            radius=self.candidate_radius,
        )

    def _rank(self, env, actions, player, limit=None):
        """
        Xếp hạng nước đi theo công thức:

            H(a) = Attack(a)
                   + defense_weight * Defense(a)
                   + CenterBonus(a)

        Attack: tiềm năng tấn công của player.
        Defense: tiềm năng đe dọa của đối thủ tại cùng ô trống.
        CenterBonus: điểm thưởng cho vị trí gần tâm bàn cờ.
        """
        scored = []

        n = env.board_size
        center = n // 2

        for action in actions:
            self.last_search_count += 1

            r, c = divmod(int(action), n)

            # 1. Điểm tấn công của người chơi đang được đánh giá.
            attack = evaluate_cell_threat(
                env,
                int(action),
                player,
            )["score"]

            # 2. Điểm phòng thủ: đánh giá mối đe dọa của đối thủ
            # tại chính ô mà player có thể chiếm.
            defense = evaluate_cell_threat(
                env,
                int(action),
                -player,
            )["score"]

            # 3. Điểm ưu tiên tâm bàn cờ.
            distance = abs(r - center) + abs(c - center)
            center_bonus = (
                max(0, n - distance) * self.center_weight
            )

            # 4. Tổng điểm Heuristic.
            total_score = (
                attack
                + self.defense_weight * defense
                + center_bonus
            )

            scored.append((total_score, int(action)))

        # Điểm cao hơn được ưu tiên.
        # Dùng action làm tiêu chí phụ để kết quả ổn định khi bằng điểm.
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        if limit is not None:
            scored = scored[:limit]

        return scored

    @staticmethod
    def _simulate(env, action):
        """Tạo bản sao môi trường và mô phỏng một nước đi."""
        child = env.clone()
        child.step(int(action))
        return child

    def _find_immediate_win(self, env, player, actions):
        """Tìm nước đi giúp player thắng ngay."""
        for action in actions:
            self.last_search_count += 1

            threat = evaluate_cell_threat(
                env,
                int(action),
                player,
            )

            if threat["is_win_5"]:
                return int(action)

        return None

    def _find_open_four(self, env, player, actions):
        """Tìm các nước đi tạo được ít nhất một chuỗi 4 mở hai đầu."""
        found = []

        for action in actions:
            self.last_search_count += 1

            threat = evaluate_cell_threat(
                env,
                int(action),
                player,
            )

            if threat["open_4"] > 0:
                found.append(int(action))

        return found

    def _best_greedy_action(self, env, player, actions):
        """
        Chọn nước đi có điểm Heuristic cao nhất.
        Đây là lựa chọn tham lam, không phải bước Minimax.
        """
        if not actions:
            return None

        ranked = self._rank(
            env,
            actions,
            player,
            limit=1,
        )

        return ranked[0][1] if ranked else None

    def _terminal_value(self, env, perspective):
        """Trả về điểm đánh giá khi trạng thái đã kết thúc."""
        if not env.is_terminal():
            return None

        winner = getattr(env, "winner", None)

        if winner == perspective:
            return 10_000_000.0

        if winner == -perspective:
            return -10_000_000.0

        return 0.0

    def _lookahead_three_ply(self, env, player, actions):
        """
        Greedy Heuristic-Guided Look-Ahead 3 ply:

            AI (m1) -> Đối thủ (m2) -> AI (m3).

        Với mỗi nước đi gốc m1:
          1. Mô phỏng m1.
          2. Đối thủ chọn phản ứng có Heuristic cao nhất theo góc nhìn
             của chính đối thủ.
          3. AI chọn nước đáp trả có Heuristic cao nhất.
          4. Đánh giá trạng thái sau m3 bằng evaluate_board().

        Cuối cùng chọn m1 có giá trị trạng thái cuối cao nhất.

        Không sử dụng Minimax, Min, Max luân phiên hoặc tìm kiếm vét cạn.
        """
        if not actions:
            raise RuntimeError("Không có nước đi ứng viên cho Look-Ahead.")

        # Chỉ giữ lại các nước đi gốc tốt nhất theo Heuristic.
        root_ranked = self._rank(
            env,
            actions,
            player,
            limit=self.root_width,
        )

        if not root_ranked:
            raise RuntimeError("Không thể xếp hạng nước đi ứng viên.")

        best_action = root_ranked[0][1]
        best_value = float("-inf")

        for _, m1 in root_ranked:
            # Ply 1: AI mô phỏng nước đi m1.
            after_m1 = self._simulate(env, m1)

            terminal_value = self._terminal_value(after_m1, player)
            if terminal_value is not None:
                value = terminal_value
            else:
                # Ply 2: Đối thủ chọn phản ứng Greedy tốt nhất.
                opponent_actions = self._candidates(after_m1)

                m2 = self._best_greedy_action(
                    after_m1,
                    -player,
                    opponent_actions,
                )

                if m2 is None:
                    value = evaluate_board(after_m1, player)
                else:
                    after_m2 = self._simulate(after_m1, m2)

                    terminal_value = self._terminal_value(
                        after_m2,
                        player,
                    )

                    if terminal_value is not None:
                        value = terminal_value
                    else:
                        # Ply 3: AI chọn nước đáp trả Greedy tốt nhất.
                        ai_actions = self._candidates(after_m2)

                        m3 = self._best_greedy_action(
                            after_m2,
                            player,
                            ai_actions,
                        )

                        if m3 is None:
                            value = evaluate_board(after_m2, player)
                        else:
                            after_m3 = self._simulate(after_m2, m3)

                            terminal_value = self._terminal_value(
                                after_m3,
                                player,
                            )

                            if terminal_value is not None:
                                value = terminal_value
                            else:
                                value = evaluate_board(
                                    after_m3,
                                    player,
                                )

            # Chọn nước đi gốc có giá trị trạng thái cuối cao nhất.
            if value > best_value:
                best_value = value
                best_action = m1

        return int(best_action)

    def select_action(self, env):
        """Điều phối 5 tầng ưu tiên và trả về nước đi của AI."""
        started = time.perf_counter()

        self.last_search_count = 0
        self.last_search_depth = 0
        self.last_decision_stage = ""

        legal = env.legal_actions()

        if not legal:
            raise RuntimeError("Không còn nước đi hợp lệ.")

        player = env.current_player
        candidates = self._candidates(env)

        # ---------------------------------------------------------
        # Tầng 1: AI thắng ngay nếu có thể.
        # ---------------------------------------------------------
        action = self._find_immediate_win(
            env,
            player,
            legal,
        )

        if action is not None:
            self.last_decision_stage = "Rule-1: immediate win"
            self.last_search_depth = 1
            self.last_move_time_ms = (
                time.perf_counter() - started
            ) * 1000
            return int(action)

        # ---------------------------------------------------------
        # Tầng 2: Chặn nước thắng ngay của đối thủ.
        # ---------------------------------------------------------
        opponent_wins = []

        for action_candidate in legal:
            self.last_search_count += 1

            threat = evaluate_cell_threat(
                env,
                int(action_candidate),
                -player,
            )

            if threat["is_win_5"]:
                opponent_wins.append(int(action_candidate))

        if opponent_wins:
            ranked_blocks = self._rank(
                env,
                opponent_wins,
                player,
                limit=1,
            )

            action = ranked_blocks[0][1]

            self.last_decision_stage = "Rule-2: block immediate win"
            self.last_search_depth = 2
            self.last_move_time_ms = (
                time.perf_counter() - started
            ) * 1000
            return int(action)

        # ---------------------------------------------------------
        # Tầng 3: Ưu tiên tạo chuỗi 4 mở hai đầu của AI.
        # ---------------------------------------------------------
        own_open_four = self._find_open_four(
            env,
            player,
            candidates,
        )

        if own_open_four:
            ranked_actions = self._rank(
                env,
                own_open_four,
                player,
                limit=1,
            )

            action = ranked_actions[0][1]

            self.last_decision_stage = "Rule-3: create open four"
            self.last_search_depth = 1
            self.last_move_time_ms = (
                time.perf_counter() - started
            ) * 1000
            return int(action)

        # ---------------------------------------------------------
        # Tầng 4: Chặn khả năng tạo chuỗi 4 mở hai đầu của đối thủ.
        # ---------------------------------------------------------
        opponent_open_four = self._find_open_four(
            env,
            -player,
            candidates,
        )

        if opponent_open_four:
            ranked_actions = self._rank(
                env,
                opponent_open_four,
                player,
                limit=1,
            )

            action = ranked_actions[0][1]

            self.last_decision_stage = "Rule-4: block open four"
            self.last_search_depth = 1
            self.last_move_time_ms = (
                time.perf_counter() - started
            ) * 1000
            return int(action)

        # ---------------------------------------------------------
        # Tầng 5: Greedy Heuristic-Guided Look-Ahead 3 ply.
        # ---------------------------------------------------------
        action = self._lookahead_three_ply(
            env,
            player,
            candidates,
        )

        self.last_decision_stage = (
            "Rule-5: greedy heuristic look-ahead"
        )
        self.last_search_depth = 3
        self.last_move_time_ms = (
            time.perf_counter() - started
        ) * 1000

        return int(action)
