from __future__ import annotations
import numpy as np


class CaroEnv:
    """
    Môi trường trò chơi Caro (Gomoku) 11x11 chuẩn hóa theo hợp đồng API Mục 3.1 đề cương.
    Quy ước trạng thái bàn cờ:
      -  1: Quân của người chơi 1 (X / Đen - đi trước)
      - -1: Quân của người chơi 2 (O / Trắng - đi sau)
      -  0: Ô trống
    """

    def __init__(self, board_size: int = 11, win_length: int = 5):
        if board_size < win_length or win_length < 3:
            raise ValueError("Kích thước bàn cờ và độ dài chuỗi thắng không hợp lệ.")
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self) -> np.ndarray:
        """Đưa bàn cờ về trạng thái rỗng ban đầu và trả về bản sao mảng trạng thái."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player: int = 1
        self.history: list[int] = []
        self._done: bool = False
        self.winner: int = 0
        self.last_player: int = 0
        return self.board.copy()

    def legal_actions(self) -> list[int]:
        """Trả về danh sách tăng dần các hành động hợp lệ. Trả về [] nếu ván đã kết thúc."""
        if self._done:
            return []
        return np.flatnonzero(self.board == 0).tolist()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Thực hiện một nước đi hợp lệ trên bàn cờ.
        Chỉ đổi current_player sang đối thủ khi ván đấu chưa kết thúc.
        """
        if self._done:
            raise ValueError("Ván đấu đã kết thúc, không thể thực hiện thêm nước đi.")
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
            raise ValueError(f"Hành động phải là số nguyên int, nhận được: {type(action)}.")

        max_action = self.board_size * self.board_size
        action_int = int(action)
        if not (0 <= action_int < max_action):
            raise ValueError(f"Hành động {action_int} nằm ngoài phạm vi [0, {max_action - 1}].")

        r, c = divmod(action_int, self.board_size)
        if self.board[r, c] != 0:
            raise ValueError(f"Ô ({r}, {c}) ứng với action {action_int} đã có quân cờ.")

        player = self.current_player
        self.board[r, c] = player
        self.history.append(action_int)
        self.last_player = player

        if self._check_win_from_cell(r, c, player):
            self._done = True
            self.winner = player
            reward = 1.0
        elif len(self.history) == max_action:
            self._done = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0
            self.current_player = -self.current_player

        info = {
            "last_action": action_int,
            "last_player": player,
            "winner": self.winner,
            "move_count": len(self.history),
        }
        return self.board.copy(), reward, self._done, info

    def _check_win_from_cell(self, r: int, c: int, player: int) -> bool:
        """
        Kiểm tra chuỗi >= win_length quân liên tiếp đi qua ô (r, c) theo 4 phương.
        Độ phức tạp: O(K) với K = win_length (hoặc O(N) trong trường hợp tổng quát).
        """
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1
            for step in range(1, self.win_length):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            for step in range(1, self.win_length):
                nr, nc = r - dr * step, c - dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            if count >= self.win_length:
                return True
        return False

    def is_terminal(self) -> bool:
        """Kiểm tra ván đấu đã kết thúc hay chưa."""
        return self._done

    def result(self, perspective: int = 1) -> float:
        """
        Trả về kết quả ván đấu theo góc nhìn của người chơi `perspective` (1 hoặc -1):
        1.0 nếu thắng, -1.0 nếu thua, 0.0 nếu hòa hoặc chưa kết thúc.
        """
        if perspective not in (1, -1):
            raise ValueError("Tham số perspective phải là 1 hoặc -1.")
        if not self._done or self.winner == 0:
            return 0.0
        return 1.0 if self.winner == perspective else -1.0

    def undo(self) -> int:
        """Hoàn tác nước đi cuối cùng và khôi phục chính xác trạng thái trước đó."""
        if not self.history:
            raise ValueError("Không có nước đi nào trong lịch sử để hoàn tác.")

        last_action = self.history.pop()
        r, c = divmod(last_action, self.board_size)
        undone_player = int(self.board[r, c])
        self.board[r, c] = 0

        self.current_player = undone_player
        self._done = False
        self.winner = 0
        if self.history:
            prev_r, prev_c = divmod(self.history[-1], self.board_size)
            self.last_player = int(self.board[prev_r, prev_c])
        else:
            self.last_player = 0
        return last_action

    def clone(self) -> CaroEnv:
        """Tạo bản sao độc lập hoàn toàn với môi trường gốc."""
        cloned = CaroEnv(self.board_size, self.win_length)
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.history = self.history.copy()
        cloned._done = self._done
        cloned.winner = self.winner
        cloned.last_player = self.last_player
        return cloned

    def render(self) -> str:
        """Hiển thị bàn cờ ra màn hình và trả về chuỗi biểu diễn."""
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = ["   " + " ".join(f"{c:2d}" for c in range(self.board_size))]
        for r in range(self.board_size):
            row_str = "  ".join(symbols[int(self.board[r, c])] for c in range(self.board_size))
            lines.append(f"{r:2d}  {row_str}")
        output = "\n".join(lines)
        print(output)
        return output