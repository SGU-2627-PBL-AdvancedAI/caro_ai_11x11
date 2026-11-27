from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

try:
    from .caro_env import CaroEnv
except ImportError:
    from caro_env import CaroEnv


# ---------------------------------------------------------------------------
# Hằng số thang điểm chuẩn hóa (đảm bảo Heuristic KHÔNG BAO GIỜ vượt điểm thắng thật)
# ---------------------------------------------------------------------------
TERMINAL_WIN_SCORE: float = 100_000.0
MAX_HEURISTIC_SCORE: float = 80_000.0
FORCED_WIN_NEXT_MOVE_SCORE: float = 60_000.0
FORCED_UNSTOPPABLE_FOUR_SCORE: float = 50_000.0
TEMPO_OPEN_THREE_BONUS: float = 22_000.0

# Trọng số cho evaluate_cell_threat (đánh giá tiềm năng chiến thuật cục bộ của 1 nước đi)
_CELL_WEIGHTS = {
    "win_5": 100_000.0,
    "open_4": 20_000.0,
    "double_four_bonus": 18_000.0,
    "four_three_bonus": 14_000.0,
    "double_three_bonus": 7_000.0,
    "half_open_4": 3_500.0,
    "broken_4": 3_200.0,
    "open_3": 1_200.0,
    "broken_open_3": 1_050.0,
    "half_open_3": 220.0,
    "open_2": 90.0,
    "broken_open_2": 65.0,
    "half_open_2": 20.0,
    "single_live": 5.0,
}

# Trọng số cho evaluate_board (quét toàn bàn theo từng đường thẳng, không đếm trùng)
_BOARD_WEIGHTS = {
    "five": 50_000.0,
    "open_4": 18_000.0,
    "half_open_4": 3_200.0,
    "broken_4": 2_900.0,
    "open_3": 1_100.0,
    "broken_open_3": 950.0,
    "half_open_3": 180.0,
    "open_2": 75.0,
    "broken_open_2": 55.0,
    "half_open_2": 18.0,
    "double_four_bonus": 12_000.0,
    "four_three_bonus": 8_000.0,
    "double_three_bonus": 4_500.0,
    "center_scale": 1.5,
}

_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


# ---------------------------------------------------------------------------
# Bảng tra cứu cache theo kích thước bàn cờ (tăng tốc tính toán trên Python thuần)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _get_neighbor_bitmasks(board_size: int, radius: int) -> tuple[int, ...]:
    """Tính trước bitmask các ô lân cận trong bán kính Chebyshev `radius` cho mỗi ô 0..N*N-1."""
    masks: list[int] = []
    for r in range(board_size):
        r_min = max(0, r - radius)
        r_max = min(board_size - 1, r + radius)
        for c in range(board_size):
            c_min = max(0, c - radius)
            c_max = min(board_size - 1, c + radius)
            mask = 0
            for nr in range(r_min, r_max + 1):
                row_off = nr * board_size
                for nc in range(c_min, c_max + 1):
                    mask |= 1 << (row_off + nc)
            masks.append(mask)
    return tuple(masks)


@lru_cache(maxsize=16)
def _get_board_lines(board_size: int, win_length: int) -> tuple[tuple[int, ...], ...]:
    """
    Liệt kê tất cả các đường thẳng (hàng, cột, chéo chính, chéo phụ) có độ dài >= win_length
    dưới dạng tuple các chỉ số phẳng (flat index 0..N*N-1).
    """
    lines: list[tuple[int, ...]] = []
    n = board_size

    # 1. Hàng ngang
    for r in range(n):
        row_off = r * n
        lines.append(tuple(row_off + c for c in range(n)))

    # 2. Cột dọc
    for c in range(n):
        lines.append(tuple(r * n + c for r in range(n)))

    # 3. Đường chéo chính (\): dr = +1, dc = +1
    for start_c in range(n):
        diag = []
        r, c = 0, start_c
        while r < n and c < n:
            diag.append(r * n + c)
            r += 1
            c += 1
        if len(diag) >= win_length:
            lines.append(tuple(diag))
    for start_r in range(1, n):
        diag = []
        r, c = start_r, 0
        while r < n and c < n:
            diag.append(r * n + c)
            r += 1
            c += 1
        if len(diag) >= win_length:
            lines.append(tuple(diag))

    # 4. Đường chéo phụ (/): dr = +1, dc = -1
    for start_c in range(n):
        diag = []
        r, c = 0, start_c
        while r < n and c >= 0:
            diag.append(r * n + c)
            r += 1
            c -= 1
        if len(diag) >= win_length:
            lines.append(tuple(diag))
    for start_r in range(1, n):
        diag = []
        r, c = start_r, n - 1
        while r < n and c >= 0:
            diag.append(r * n + c)
            r += 1
            c -= 1
        if len(diag) >= win_length:
            lines.append(tuple(diag))

    return tuple(lines)


@lru_cache(maxsize=16)
def _get_center_weights(board_size: int) -> tuple[float, ...]:
    """Trọng số kiểm soát trung tâm nhẹ cho từng ô (dùng làm tiêu chí phụ khi thế cờ cân bằng)."""
    center = (board_size - 1) / 2.0
    max_dist = max(1.0, center * 2.0)
    weights: list[float] = []
    for r in range(board_size):
        for c in range(board_size):
            dist = abs(r - center) + abs(c - center)
            weights.append((max_dist - dist) / max_dist)
    return tuple(weights)


# ---------------------------------------------------------------------------
# 1. Sinh nước đi ứng viên (Candidate Action Pruning)
# ---------------------------------------------------------------------------
def get_candidate_actions(env: CaroEnv, radius: int | None = 2) -> list[int]:
    """
    Lọc danh sách các ô trống ứng viên để thu hẹp không gian tìm kiếm:
    - Nếu bàn cờ đã kết thúc: trả về [].
    - Nếu radius is None: trả về toàn bộ env.legal_actions().
    - Nếu bàn cờ trống: trả về ô chính giữa bàn cờ (ví dụ (5, 5) -> action = 60).
    - Ngược lại: chỉ lấy các ô trống nằm trong bán kính Chebyshev `radius` quanh các quân cờ đã đánh.
    """
    if env.is_terminal():
        return []
    if radius is None:
        return env.legal_actions()

    n = env.board_size
    if not env.history and not np.any(env.board):
        center = n // 2
        return [center * n + center]

    if radius <= 0:
        return env.legal_actions()

    nb_masks = _get_neighbor_bitmasks(n, int(radius))
    flat = env.board.ravel().tolist()

    neighbor_union = 0
    occupied_mask = 0
    for idx, val in enumerate(flat):
        if val != 0:
            occupied_mask |= 1 << idx
            neighbor_union |= nb_masks[idx]

    if occupied_mask == 0:
        center = n // 2
        return [center * n + center]

    cand_mask = neighbor_union & ~occupied_mask
    if cand_mask == 0:
        return env.legal_actions()

    candidates: list[int] = []
    while cand_mask:
        lsb = cand_mask & -cand_mask
        candidates.append(lsb.bit_length() - 1)
        cand_mask ^= lsb
    return candidates


# ---------------------------------------------------------------------------
# 2. Đánh giá tiềm năng chiến thuật cục bộ cho 1 ô (Local Cell Threat)
# ---------------------------------------------------------------------------
def evaluate_cell_threat(env: CaroEnv, action: int, player: int) -> dict[str, Any]:
    """
    Đánh giá tiềm năng chiến thuật cục bộ khi đặt quân của `player` vào ô `action`.

    Hỗ trợ đầy đủ:
    - Chuỗi liền kề: win_5, open_4, half_open_4, open_3 (có kiểm tra đủ 6 ô), half_open_3, open_2, half_open_2.
    - Chuỗi ngắt quãng 1 ô trống: broken_4 (XX_XX, XXX_X), broken_open_3 (_X_XX_), broken_open_2 (_X_X_).
    - Loại bỏ đường chết (max_span < win_length).
    - Phát hiện đòn kép (Forks): double_four, four_three, double_three.

    Giữ nguyên tương thích ngược với tất cả khóa cũ:
      `is_win_5`, `open_4`, `half_open_4`, `open_3`, `score`
    và bổ sung thêm:
      `broken_4`, `broken_open_3`, `half_open_3`, `open_2`, `is_fork`, `fork_type`.
    """
    n = env.board_size
    win_len = env.win_length
    r, c = divmod(int(action), n)
    board = env.board

    is_win_5 = False
    open_4 = 0
    half_open_4 = 0
    broken_4 = 0
    open_3 = 0
    broken_open_3 = 0
    half_open_3 = 0
    open_2 = 0
    score = 0.0

    # Cửa sổ 1D độ dài 2 * win_len + 1 quanh ô (r, c), tâm tại vị trí `win_len`
    # Quy ước: 1 = quân player, 0 = ô trống, -1 = quân đối thủ hoặc biên bàn cờ
    win_size = 2 * win_len + 1
    center_idx = win_len
    line = [-1] * win_size
    line[center_idx] = 1

    for dr, dc in _DIRECTIONS:
        # Nạp chiều ngược (-1 .. -win_len)
        for step in range(1, win_len + 1):
            nr = r - dr * step
            nc = c - dc * step
            idx = center_idx - step
            if 0 <= nr < n and 0 <= nc < n:
                cell = board.item(nr, nc)
                if cell == player:
                    line[idx] = 1
                elif cell == 0:
                    line[idx] = 0
                else:
                    line[idx] = -1
            else:
                line[idx] = -1

        # Nạp chiều thuận (+1 .. +win_len)
        for step in range(1, win_len + 1):
            nr = r + dr * step
            nc = c + dc * step
            idx = center_idx + step
            if 0 <= nr < n and 0 <= nc < n:
                cell = board.item(nr, nc)
                if cell == player:
                    line[idx] = 1
                elif cell == 0:
                    line[idx] = 0
                else:
                    line[idx] = -1
            else:
                line[idx] = -1

        # 1. Đếm chuỗi quân liền kề đi qua tâm
        left_end = center_idx - 1
        while left_end >= 0 and line[left_end] == 1:
            left_end -= 1
        right_end = center_idx + 1
        while right_end < win_size and line[right_end] == 1:
            right_end += 1

        contiguous = right_end - left_end - 1

        if contiguous >= win_len:
            is_win_5 = True
            score += _CELL_WEIGHTS["win_5"]
            continue

        # 2. Kiểm tra khoảng trống tối đa không bị chặn (loại bỏ đường chết)
        left_wall = left_end
        while left_wall >= 0 and line[left_wall] != -1:
            left_wall -= 1
        right_wall = right_end
        while right_wall < win_size and line[right_wall] != -1:
            right_wall += 1

        max_span = right_wall - left_wall - 1
        if max_span < win_len:
            continue

        left_open = left_end >= 0 and line[left_end] == 0
        right_open = right_end < win_size and line[right_end] == 0
        open_ends = int(left_open) + int(right_open)

        # 3. Chuỗi 4 liền kề
        if contiguous == win_len - 1:
            if open_ends == 2:
                open_4 += 1
                score += _CELL_WEIGHTS["open_4"]
            elif open_ends == 1:
                half_open_4 += 1
                score += _CELL_WEIGHTS["half_open_4"]
            continue

        # 4. Đếm quân nối tiếp qua 1 ô trống ở mỗi phía để phát hiện mẫu ngắt quãng (broken patterns)
        left_gap_cnt = 0
        left_outer_open = False
        if left_open:
            idx = left_end - 1
            while idx >= 0 and line[idx] == 1:
                left_gap_cnt += 1
                idx -= 1
            left_outer_open = idx >= 0 and line[idx] == 0

        right_gap_cnt = 0
        right_outer_open = False
        if right_open:
            idx = right_end + 1
            while idx < win_size and line[idx] == 1:
                right_gap_cnt += 1
                idx += 1
            right_outer_open = idx < win_size and line[idx] == 0

        # 4a. Kiểm tra Broken-4 (ví dụ XX_XX, XXX_X, X_XXX)
        left_is_b4 = left_gap_cnt > 0 and (contiguous + left_gap_cnt >= win_len - 1)
        right_is_b4 = right_gap_cnt > 0 and (contiguous + right_gap_cnt >= win_len - 1)
        if left_is_b4 or right_is_b4:
            num_b4 = int(left_is_b4) + int(right_is_b4)
            broken_4 += num_b4
            score += _CELL_WEIGHTS["broken_4"] * num_b4
            continue

        # 5. Chuỗi 3 liền kề
        if contiguous == win_len - 2:
            if open_ends == 2 and max_span >= win_len + 1:
                open_3 += 1
                score += _CELL_WEIGHTS["open_3"]
            elif open_ends >= 1:
                half_open_3 += 1
                score += _CELL_WEIGHTS["half_open_3"]
            continue

        # 6. Chuỗi 3 ngắt quãng 1 ô (ví dụ _X_XX_, _XX_X_)
        left_is_b3 = left_gap_cnt > 0 and (contiguous + left_gap_cnt == win_len - 2)
        right_is_b3 = right_gap_cnt > 0 and (contiguous + right_gap_cnt == win_len - 2)
        if left_is_b3 or right_is_b3:
            is_live_b3 = (left_is_b3 and left_outer_open and right_open) or (
                right_is_b3 and right_outer_open and left_open
            )
            if is_live_b3:
                broken_open_3 += 1
                score += _CELL_WEIGHTS["broken_open_3"]
            else:
                half_open_3 += 1
                score += _CELL_WEIGHTS["half_open_3"]
            continue

        # 7. Chuỗi 2 liền kề hoặc ngắt quãng 1 ô (khi win_len >= 4)
        if win_len >= 4 and contiguous == win_len - 3:
            if open_ends == 2 and max_span >= win_len + 1:
                open_2 += 1
                score += _CELL_WEIGHTS["open_2"]
            elif open_ends >= 1:
                score += _CELL_WEIGHTS["half_open_2"]
            continue

        if win_len >= 4:
            left_is_b2 = left_gap_cnt > 0 and (contiguous + left_gap_cnt == win_len - 3)
            right_is_b2 = right_gap_cnt > 0 and (contiguous + right_gap_cnt == win_len - 3)
            if left_is_b2 or right_is_b2:
                is_live_b2 = (left_is_b2 and left_outer_open and right_open) or (
                    right_is_b2 and right_outer_open and left_open
                )
                if is_live_b2:
                    open_2 += 1
                    score += _CELL_WEIGHTS["broken_open_2"]
                else:
                    score += _CELL_WEIGHTS["half_open_2"]
                continue

        if open_ends == 2:
            score += _CELL_WEIGHTS["single_live"]

    # 8. Đánh giá đòn kép (Forks) trên nhiều hướng
    effective_fours = half_open_4 + broken_4
    effective_live_threes = open_3 + broken_open_3
    is_fork = False
    fork_type = ""

    if not is_win_5:
        if open_4 >= 1:
            is_fork = True
            fork_type = "open_four"
        elif effective_fours >= 2:
            is_fork = True
            fork_type = "double_four"
            score += _CELL_WEIGHTS["double_four_bonus"]
        elif effective_fours >= 1 and effective_live_threes >= 1:
            is_fork = True
            fork_type = "four_three"
            score += _CELL_WEIGHTS["four_three_bonus"]
        elif effective_live_threes >= 2:
            is_fork = True
            fork_type = "double_three"
            score += _CELL_WEIGHTS["double_three_bonus"]

    return {
        "is_win_5": is_win_5,
        "open_4": open_4,
        "half_open_4": half_open_4,
        "broken_4": broken_4,
        "open_3": open_3,
        "broken_open_3": broken_open_3,
        "half_open_3": half_open_3,
        "open_2": open_2,
        "is_fork": is_fork,
        "fork_type": fork_type,
        "score": score,
    }


# ---------------------------------------------------------------------------
# 3. Quét & thống kê mẫu toàn bàn cờ theo đường thẳng (Line-Based Pattern Scanner)
# ---------------------------------------------------------------------------
def _scan_line_for_player(
    flat_board: list[int],
    line_indices: tuple[int, ...],
    player: int,
    win_len: int,
    counts: dict[str, int],
) -> None:
    """
    Quét 1 đường thẳng cho người chơi `player` theo từng phân đoạn không bị đối thủ chặn.
    - Loại bỏ hoàn toàn phân đoạn có độ dài < win_len (đường chết).
    - Không đếm trùng giữa mẫu bậc cao (broken_4, broken_open_3) và mẫu con bên trong.
    """
    line_len = len(line_indices)
    i = 0

    while i < line_len:
        # Bỏ qua quân đối thủ
        if flat_board[line_indices[i]] == -player:
            i += 1
            continue

        seg_start = i
        runs: list[tuple[int, int]] = []  # (run_start, run_len)
        while i < line_len and flat_board[line_indices[i]] != -player:
            if flat_board[line_indices[i]] == player:
                run_start = i
                while i < line_len and flat_board[line_indices[i]] == player:
                    i += 1
                runs.append((run_start, i - run_start))
            else:
                i += 1
        seg_end = i
        seg_len = seg_end - seg_start

        # Bỏ qua nếu đoạn trống hoặc không đủ độ dài tạo chuỗi thắng
        if not runs or seg_len < win_len:
            continue

        num_runs = len(runs)
        used = [False] * num_runs

        # Bước 1: Ghi nhận các chuỗi >= win_len và chuỗi (win_len - 1) liền kề
        for idx, (rs, rl) in enumerate(runs):
            if rl >= win_len:
                counts["five"] += 1
                used[idx] = True
            elif rl == win_len - 1:
                left_open = rs > seg_start
                right_open = (rs + rl) < seg_end
                if left_open and right_open:
                    counts["open_4"] += 1
                else:
                    counts["half_open_4"] += 1
                used[idx] = True

        # Bước 2: Phát hiện Broken-4 (2 cụm quân cách nhau đúng 1 ô trống có tổng >= win_len - 1)
        # Dùng cờ riêng để bắt được cả trường hợp 2 khoảng trống thắng trên cùng 1 đường (vd: X_XXX_X, XX_XX_XX)
        b4_used = [False] * num_runs
        for idx in range(num_runs - 1):
            if used[idx] or used[idx + 1]:
                continue
            rs1, rl1 = runs[idx]
            rs2, rl2 = runs[idx + 1]
            if rs2 == rs1 + rl1 + 1 and (rl1 + rl2 >= win_len - 1):
                counts["broken_4"] += 1
                b4_used[idx] = True
                b4_used[idx + 1] = True
        for idx in range(num_runs):
            if b4_used[idx]:
                used[idx] = True

        # Bước 3: Chuỗi (win_len - 2) liền kề (Open-3 / Half-Open-3)
        if win_len >= 4:
            for idx, (rs, rl) in enumerate(runs):
                if used[idx]:
                    continue
                if rl == win_len - 2:
                    left_open = rs > seg_start
                    right_open = (rs + rl) < seg_end
                    if left_open and right_open and seg_len >= win_len + 1:
                        counts["open_3"] += 1
                    else:
                        counts["half_open_3"] += 1
                    used[idx] = True

            # Bước 4: Broken-3 (2 cụm quân cách nhau đúng 1 ô trống có tổng == win_len - 2)
            for idx in range(num_runs - 1):
                if used[idx] or used[idx + 1]:
                    continue
                rs1, rl1 = runs[idx]
                rs2, rl2 = runs[idx + 1]
                if rs2 == rs1 + rl1 + 1 and (rl1 + rl2 == win_len - 2):
                    left_open = rs1 > seg_start
                    right_open = (rs2 + rl2) < seg_end
                    if left_open and right_open and seg_len >= win_len + 1:
                        counts["broken_open_3"] += 1
                    else:
                        counts["half_open_3"] += 1
                    used[idx] = True
                    used[idx + 1] = True

        # Bước 5: Chuỗi (win_len - 3) liền kề và ngắt quãng (Open-2 / Half-Open-2 với win_len >= 5)
        if win_len >= 5:
            for idx, (rs, rl) in enumerate(runs):
                if used[idx]:
                    continue
                if rl == win_len - 3:
                    left_open = rs > seg_start
                    right_open = (rs + rl) < seg_end
                    if left_open and right_open and seg_len >= win_len + 1:
                        counts["open_2"] += 1
                    else:
                        counts["half_open_2"] += 1
                    used[idx] = True

            for idx in range(num_runs - 1):
                if used[idx] or used[idx + 1]:
                    continue
                rs1, rl1 = runs[idx]
                rs2, rl2 = runs[idx + 1]
                if rs2 == rs1 + rl1 + 1 and (rl1 + rl2 == win_len - 3):
                    left_open = rs1 > seg_start
                    right_open = (rs2 + rl2) < seg_end
                    if left_open and right_open and seg_len >= win_len + 1:
                        counts["broken_open_2"] += 1
                    else:
                        counts["half_open_2"] += 1
                    used[idx] = True
                    used[idx + 1] = True


def extract_board_patterns(env: CaroEnv, player: int) -> dict[str, int]:
    """
    Trích xuất số lượng từng loại mẫu hình chiến thuật hiện có của `player` trên toàn bàn cờ
    (không đếm trùng và đã loại trừ đường chết).
    """
    counts: dict[str, int] = {
        "five": 0,
        "open_4": 0,
        "half_open_4": 0,
        "broken_4": 0,
        "open_3": 0,
        "broken_open_3": 0,
        "half_open_3": 0,
        "open_2": 0,
        "broken_open_2": 0,
        "half_open_2": 0,
    }
    flat_board = env.board.ravel().tolist()
    lines = _get_board_lines(env.board_size, env.win_length)
    win_len = env.win_length

    for line_indices in lines:
        # Bỏ qua nhanh đường không có quân nào của `player`
        has_player = False
        for idx in line_indices:
            if flat_board[idx] == player:
                has_player = True
                break
        if has_player:
            _scan_line_for_player(flat_board, line_indices, player, win_len, counts)

    return counts


def _score_player_patterns(counts: dict[str, int], center_score: float) -> float:
    """Tính tổng điểm thế cờ từ bảng đếm mẫu hình + thưởng tổ hợp đòn kép."""
    score = (
        counts["five"] * _BOARD_WEIGHTS["five"]
        + counts["open_4"] * _BOARD_WEIGHTS["open_4"]
        + counts["half_open_4"] * _BOARD_WEIGHTS["half_open_4"]
        + counts["broken_4"] * _BOARD_WEIGHTS["broken_4"]
        + counts["open_3"] * _BOARD_WEIGHTS["open_3"]
        + counts["broken_open_3"] * _BOARD_WEIGHTS["broken_open_3"]
        + counts["half_open_3"] * _BOARD_WEIGHTS["half_open_3"]
        + counts["open_2"] * _BOARD_WEIGHTS["open_2"]
        + counts["broken_open_2"] * _BOARD_WEIGHTS["broken_open_2"]
        + counts["half_open_2"] * _BOARD_WEIGHTS["half_open_2"]
        + center_score * _BOARD_WEIGHTS["center_scale"]
    )

    fours = counts["half_open_4"] + counts["broken_4"]
    live_threes = counts["open_3"] + counts["broken_open_3"]

    if counts["open_4"] >= 1 or fours >= 2:
        score += _BOARD_WEIGHTS["double_four_bonus"]
    elif fours >= 1 and live_threes >= 1:
        score += _BOARD_WEIGHTS["four_three_bonus"]
    elif live_threes >= 2:
        score += _BOARD_WEIGHTS["double_three_bonus"]

    return score


def evaluate_board(
    env: CaroEnv,
    perspective: int,
    consider_turn: bool = False,
) -> float:
    """
    Hàm lượng giá toàn bàn cờ theo góc nhìn của `perspective` (1 hoặc -1), dựa trên quét
    mẫu hình theo đường thẳng (Line-Based Evaluation):
    - Không đếm trùng chuỗi trên nhiều ô trống.
    - Nhận diện đầy đủ chuỗi liền kề, chuỗi ngắt quãng (broken-4, broken-3), đòn kép và loại bỏ đường chết.
    - Giới hạn chặt điểm trạng thái chưa kết thúc trong [-MAX_HEURISTIC_SCORE, +MAX_HEURISTIC_SCORE]
      để KHÔNG BAO GIỜ vượt quá điểm thắng thật (±TERMINAL_WIN_SCORE).
    - Tham số `consider_turn=True` kích hoạt đánh giá theo quyền đi trước (Tempo / Turn-to-Move),
      phù hợp cho nút lá Minimax / Look-Ahead. Mặc định `False` để bảo toàn tính đối xứng màu cờ.
    """
    if env.is_terminal():
        return TERMINAL_WIN_SCORE * env.result(perspective)

    flat_board = env.board.ravel().tolist()
    lines = _get_board_lines(env.board_size, env.win_length)
    center_weights = _get_center_weights(env.board_size)
    win_len = env.win_length
    opponent = -perspective

    my_counts: dict[str, int] = {
        "five": 0,
        "open_4": 0,
        "half_open_4": 0,
        "broken_4": 0,
        "open_3": 0,
        "broken_open_3": 0,
        "half_open_3": 0,
        "open_2": 0,
        "broken_open_2": 0,
        "half_open_2": 0,
    }
    opp_counts: dict[str, int] = {
        "five": 0,
        "open_4": 0,
        "half_open_4": 0,
        "broken_4": 0,
        "open_3": 0,
        "broken_open_3": 0,
        "half_open_3": 0,
        "open_2": 0,
        "broken_open_2": 0,
        "half_open_2": 0,
    }

    my_center = 0.0
    opp_center = 0.0
    for idx, val in enumerate(flat_board):
        if val == perspective:
            my_center += center_weights[idx]
        elif val == opponent:
            opp_center += center_weights[idx]

    for line_indices in lines:
        has_me = False
        has_opp = False
        for idx in line_indices:
            v = flat_board[idx]
            if v == perspective:
                has_me = True
            elif v == opponent:
                has_opp = True
            if has_me and has_opp:
                break
        if has_me:
            _scan_line_for_player(flat_board, line_indices, perspective, win_len, my_counts)
        if has_opp:
            _scan_line_for_player(flat_board, line_indices, opponent, win_len, opp_counts)

    my_score = _score_player_patterns(my_counts, my_center)
    opp_score = _score_player_patterns(opp_counts, opp_center)

    if not consider_turn:
        raw_score = my_score - 1.05 * opp_score
    else:
        my_fours = my_counts["open_4"] + my_counts["half_open_4"] + my_counts["broken_4"]
        opp_fours = opp_counts["open_4"] + opp_counts["half_open_4"] + opp_counts["broken_4"]
        my_live_3 = my_counts["open_3"] + my_counts["broken_open_3"]
        opp_live_3 = opp_counts["open_3"] + opp_counts["broken_open_3"]
        my_unstoppable_4 = my_counts["open_4"] >= 1 or (my_counts["half_open_4"] + my_counts["broken_4"]) >= 2
        opp_unstoppable_4 = opp_counts["open_4"] >= 1 or (opp_counts["half_open_4"] + opp_counts["broken_4"]) >= 2

        if env.current_player == perspective:
            if my_fours >= 1:
                return FORCED_WIN_NEXT_MOVE_SCORE
            if opp_unstoppable_4:
                return -FORCED_UNSTOPPABLE_FOUR_SCORE
            tempo_bonus = TEMPO_OPEN_THREE_BONUS if (my_live_3 >= 1 and opp_fours == 0) else 0.0
            raw_score = 1.08 * my_score - opp_score + tempo_bonus
        else:
            if opp_fours >= 1:
                return -FORCED_WIN_NEXT_MOVE_SCORE
            if my_unstoppable_4:
                return FORCED_UNSTOPPABLE_FOUR_SCORE
            tempo_penalty = TEMPO_OPEN_THREE_BONUS if (opp_live_3 >= 1 and my_fours == 0) else 0.0
            raw_score = my_score - 1.08 * opp_score - tempo_penalty

    if raw_score > MAX_HEURISTIC_SCORE:
        return MAX_HEURISTIC_SCORE
    if raw_score < -MAX_HEURISTIC_SCORE:
        return -MAX_HEURISTIC_SCORE
    return float(raw_score)
