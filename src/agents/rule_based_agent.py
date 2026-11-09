import numpy as np
from .base_agent import BaseAgent

class RuleBasedAgent(BaseAgent):
    """Heuristic theo cửa sổ 5 ô: thắng ngay > chặn thắng > tạo thế mạnh."""
    def __init__(self, name="Rule-Based"):
        super().__init__(name)
        self.weights = {5: 10**9, 4: 200000, 3: 12000, 2: 700, 1: 30, 0: 0}

    def _score_move(self, board, r, c, player):
        n = board.shape[0]
        opponent = -player
        total_score = 0
        # Chỉ xét các đoạn 5 ô có chứa vị trí đang đánh.
        for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
            for offset in range(5):
                sr, sc = r - offset*dr, c - offset*dc
                er, ec = sr + 4*dr, sc + 4*dc
                if not (0 <= sr < n and 0 <= sc < n and 0 <= er < n and 0 <= ec < n):
                    continue
                cells = [board[sr+i*dr, sc+i*dc] for i in range(5)]
                own = int(sum(v == player for v in cells))
                opp = int(sum(v == opponent for v in cells))
                if own and opp:
                    continue
                if own:
                    total_score += self.weights[own]
                elif opp:
                    # Chặn mối đe dọa của đối thủ, ưu tiên chặn 4 hơn tự tạo 3.
                    total_score += self.weights[opp] * 1.08
        # Ưu tiên vùng trung tâm khi điểm chiến thuật tương đương.
        center = (n - 1) / 2
        total_score += max(0, 10 - (abs(r-center) + abs(c-center)))
        return total_score

    def select_action(self, env):
        legal = env.legal_actions()
        if not legal:
            raise RuntimeError("Không còn nước đi hợp lệ.")
        board = env.board
        player = env.current_player
        # Các nước cách quân đã có quá 4 ô không thể tạo/chặn một chuỗi 5
        # ở lượt hiện tại; bỏ qua để giảm chi phí đánh giá.
        occupied = np.argwhere(board != 0)
        if len(occupied):
            candidates = []
            for action in legal:
                r0, c0 = divmod(action, env.board_size)
                if any(max(abs(r0-r), abs(c0-c)) <= 4 for r, c in occupied):
                    candidates.append(action)
            legal = candidates or legal
        else:
            center = (env.board_size // 2) * env.board_size + env.board_size // 2
            legal = [center]
        best_action, best_score = legal[0], float("-inf")
        for action in legal:
            r, c = divmod(action, env.board_size)
            board[r, c] = player
            # Nếu thắng ngay thì chọn tức thì.
            won = env._check_win_from_cell(r, c, player)
            if won:
                board[r, c] = 0
                return action
            attack = self._score_move(board, r, c, player)
            # Giả lập đối thủ chiếm ô này để đo mức độ cần phòng thủ.
            board[r, c] = -player
            defense = self._score_move(board, r, c, -player)
            board[r, c] = 0
            score = attack + 0.98 * defense
            if score > best_score:
                best_action, best_score = action, score
        return best_action
