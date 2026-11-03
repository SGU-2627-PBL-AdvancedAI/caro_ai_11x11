from __future__ import annotations
import numpy as np
from caro_env import CaroEnv


def get_candidate_actions(env: CaroEnv, radius: int | None = 2) -> list[int]:
    """
    Lọc danh sách các ô trống ứng viên để thu hẹp không gian tìm kiếm:
    - Nếu radius is None: Trả về toàn bộ env.legal_actions() (dùng khi tắt Candidate Pruning).
    - Nếu bàn cờ trống: Trả về ô chính giữa bàn cờ (5, 5) ứng với action = 60.
    - Ngược lại: Chỉ lấy các ô trống nằm trong bán kính `radius` quanh các quân cờ đã đánh.
    """
    if env.is_terminal():
        return []
    if radius is None:
        return env.legal_actions()
    if not env.history:
        center = env.board_size // 2
        return [center * env.board_size + center]

    occupied_rows, occupied_cols = np.where(env.board != 0)
    n = env.board_size
    mask = np.zeros((n, n), dtype=bool)

    for r, c in zip(occupied_rows, occupied_cols):
        r_min, r_max = max(0, int(r) - radius), min(n - 1, int(r) + radius)
        c_min, c_max = max(0, int(c) - radius), min(n - 1, int(c) + radius)
        mask[r_min : r_max + 1, c_min : c_max + 1] = True

    # Chỉ giữ lại các ô đang trống (board == 0) nằm trong vùng lân cận
    candidate_mask = mask & (env.board == 0)
    candidates = np.flatnonzero(candidate_mask).tolist()
    return candidates if candidates else env.legal_actions()


def evaluate_cell_threat(env: CaroEnv, action: int, player: int) -> dict:
    """
    Đánh giá tiềm năng chiến thuật cục bộ khi đặt thử quân của `player` vào ô `action`.
    Trả về từ điển chứa thông tin thắng ngay (5 quân), chuỗi 4 mở 2 đầu, 4 mở 1 đầu, 3 mở 2 đầu và tổng điểm.
    """
    n = env.board_size
    r, c = divmod(action, n)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    is_win_5 = False
    open_4 = 0
    half_open_4 = 0
    open_3 = 0
    score = 0.0

    for dr, dc in directions:
        contiguous = 1
        open_ends = 0

        # Đếm theo chiều thuận
        for step in range(1, env.win_length):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < n and 0 <= nc < n:
                if env.board[nr, nc] == player:
                    contiguous += 1
                elif env.board[nr, nc] == 0:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break

        # Đếm theo chiều ngược
        for step in range(1, env.win_length):
            nr, nc = r - dr * step, c - dc * step
            if 0 <= nr < n and 0 <= nc < n:
                if env.board[nr, nc] == player:
                    contiguous += 1
                elif env.board[nr, nc] == 0:
                    open_ends += 1
                    break
                else:
                    break
            else:
                break

        if contiguous >= env.win_length:
            is_win_5 = True
            score += 100000.0
        elif contiguous == 4:
            if open_ends == 2:
                open_4 += 1
                score += 15000.0
            elif open_ends == 1:
                half_open_4 += 1
                score += 3000.0
        elif contiguous == 3:
            if open_ends == 2:
                open_3 += 1
                score += 800.0
            elif open_ends == 1:
                score += 150.0
        elif contiguous == 2 and open_ends >= 1:
            score += 25.0 * open_ends

    return {
        "is_win_5": is_win_5,
        "open_4": open_4,
        "half_open_4": half_open_4,
        "open_3": open_3,
        "score": score,
    }


def evaluate_board(env: CaroEnv, perspective: int) -> float:
    """
    Hàm lượng giá toàn bàn cờ theo góc nhìn của `perspective` (1 hoặc -1).
    Dùng tại độ sâu giới hạn (depth == 0) của thuật toán Minimax (Cấp 3).
    """
    if env.is_terminal():
        return 100000.0 * env.result(perspective)

    candidates = get_candidate_actions(env, radius=1)
    my_score = 0.0
    opp_score = 0.0
    opponent = -perspective

    for action in candidates:
        my_threat = evaluate_cell_threat(env, action, perspective)
        opp_threat = evaluate_cell_threat(env, action, opponent)
        my_score += my_threat["score"]
        opp_score += opp_threat["score"]

    return my_score - 1.05 * opp_score