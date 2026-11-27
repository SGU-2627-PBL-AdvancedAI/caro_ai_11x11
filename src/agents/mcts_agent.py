from __future__ import annotations

import math
import random
import time

import numpy as np

try:
    from .base_agent import BaseAgent
    from ..caro_env import CaroEnv
    from ..heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions
except ImportError:
    from agents.base_agent import BaseAgent
    from caro_env import CaroEnv
    from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions


# Cache bảng láng giềng Chebyshev theo (board_size, radius) để cập nhật ứng viên O(1) trong rollout
_NEIGHBOR_CACHE: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


def _get_neighbors(board_size: int, radius: int) -> tuple[tuple[int, ...], ...]:
    """Trả về danh sách chỉ số ô lân cận trong bán kính Chebyshev `radius` cho mỗi ô."""
    key = (board_size, radius)
    cached = _NEIGHBOR_CACHE.get(key)
    if cached is not None:
        return cached

    table: list[tuple[int, ...]] = []
    for r in range(board_size):
        r_min = max(0, r - radius)
        r_max = min(board_size - 1, r + radius)
        for c in range(board_size):
            c_min = max(0, c - radius)
            c_max = min(board_size - 1, c + radius)
            nbs: list[int] = []
            for nr in range(r_min, r_max + 1):
                row_off = nr * board_size
                for nc in range(c_min, c_max + 1):
                    if nr == r and nc == c:
                        continue
                    nbs.append(row_off + nc)
            table.append(tuple(nbs))

    result = tuple(table)
    _NEIGHBOR_CACHE[key] = result
    return result


def _check_win_flat(
    flat: list[int],
    size: int,
    win_len: int,
    action: int,
    player: int,
) -> bool:
    """Kiểm tra nhanh chiến thắng (>= win_len quân liên tiếp) trực tiếp trên mảng phẳng."""
    r, c = divmod(action, size)
    for dr, dc in _DIRECTIONS:
        count = 1
        tr, tc = r + dr, c + dc
        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == player:
            count += 1
            if count >= win_len:
                return True
            tr += dr
            tc += dc
        tr, tc = r - dr, c - dc
        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == player:
            count += 1
            if count >= win_len:
                return True
            tr -= dr
            tc -= dc
    return False


def _scan_stone_threats_flat(
    flat: list[int],
    size: int,
    win_len: int,
    stone_idx: int,
) -> tuple[int, int]:
    """
    Từ ô vừa đặt quân `stone_idx`, quét đúng 4 phương trên mảng phẳng để tìm:
      - `win_cell`: ô trống hoàn thành >= win_len quân (gồm cả chuỗi 4 liền và 4 đứt quãng).
      - `open4_cell`: ô trống tạo chuỗi 4 mở hai đầu (`open_4`, gồm cả 3 mở liền và 3 mở đứt quãng).
    Trả về `(-1, -1)` nếu không có đe dọa cấp bách.
    """
    color = flat[stone_idx]
    if color == 0:
        return -1, -1

    r, c = divmod(stone_idx, size)
    open4_cell = -1

    for dr, dc in _DIRECTIONS:
        # Quét phía +
        pos = 0
        tr, tc = r + dr, c + dc
        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == color:
            pos += 1
            tr += dr
            tc += dc
        pos_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0
        pos_end = tr * size + tc if pos_open else -1
        pos_after = 0
        pos_after_open = False
        if pos_open:
            tr += dr
            tc += dc
            while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == color:
                pos_after += 1
                tr += dr
                tc += dc
            pos_after_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

        # Quét phía -
        neg = 0
        tr, tc = r - dr, c - dc
        while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == color:
            neg += 1
            tr -= dr
            tc -= dc
        neg_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0
        neg_end = tr * size + tc if neg_open else -1
        neg_after = 0
        neg_after_open = False
        if neg_open:
            tr -= dr
            tc -= dc
            while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == color:
                neg_after += 1
                tr -= dr
                tc -= dc
            neg_after_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

        run = 1 + pos + neg
        if pos_open and (run + pos_after >= win_len - 1):
            return pos_end, open4_cell
        if neg_open and (run + neg_after >= win_len - 1):
            return neg_end, open4_cell

        if open4_cell < 0 and pos_open and neg_open:
            if run + pos_after == win_len - 2 and pos_after_open:
                open4_cell = pos_end
            elif run + neg_after == win_len - 2 and neg_after_open:
                open4_cell = neg_end

    return -1, open4_cell


def _quick_cell_score_flat(
    flat: list[int],
    size: int,
    win_len: int,
    action: int,
    my_player: int,
    opp_player: int,
) -> tuple[bool, bool, float]:
    """
    Chấm điểm chiến thuật nhanh cho 1 ô trống trên mảng phẳng Python (không cấp phát dict).
    Trả về `(my_win, opp_win, priority_score)`.
    """
    r, c = divmod(action, size)
    my_win = False
    opp_win = False
    my_o4 = my_h4 = my_o3 = my_h3 = my_o2 = 0
    opp_o4 = opp_h4 = opp_o3 = opp_h3 = opp_o2 = 0

    for dr, dc in _DIRECTIONS:
        nr1, nc1 = r + dr, c + dc
        pr1, pc1 = r - dr, c - dc
        v_pos1 = flat[nr1 * size + nc1] if (0 <= nr1 < size and 0 <= nc1 < size) else 2
        v_neg1 = flat[pr1 * size + pc1] if (0 <= pr1 < size and 0 <= pc1 < size) else 2

        if v_pos1 == my_player or v_neg1 == my_player:
            pos = 0
            tr, tc = nr1, nc1
            while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == my_player:
                pos += 1
                tr += dr
                tc += dc
            pos_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

            neg = 0
            tr, tc = pr1, pc1
            while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == my_player:
                neg += 1
                tr -= dr
                tc -= dc
            neg_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

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

        if v_pos1 == opp_player or v_neg1 == opp_player:
            pos = 0
            tr, tc = nr1, nc1
            while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == opp_player:
                pos += 1
                tr += dr
                tc += dc
            pos_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

            neg = 0
            tr, tc = pr1, pc1
            while 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == opp_player:
                neg += 1
                tr -= dr
                tc -= dc
            neg_open = 0 <= tr < size and 0 <= tc < size and flat[tr * size + tc] == 0

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

    if my_win:
        return True, opp_win, 10_000_000.0

    my_score = my_o4 * 50000.0 + my_h4 * 8000.0 + my_o3 * 5000.0 + my_h3 * 600.0 + my_o2 * 300.0
    opp_score = opp_o4 * 50000.0 + opp_h4 * 8000.0 + opp_o3 * 5000.0 + opp_h3 * 600.0 + opp_o2 * 300.0

    if opp_win:
        return False, True, 5_000_000.0 + opp_score + 0.5 * my_score

    my_f4 = my_o4 + my_h4
    opp_f4 = opp_o4 + opp_h4
    if my_o4 > 0 or my_f4 >= 2 or (my_f4 >= 1 and my_o3 >= 1):
        priority = 2_000_000.0 + my_score + 0.5 * opp_score
    elif opp_o4 > 0 or opp_f4 >= 2 or (opp_f4 >= 1 and opp_o3 >= 1):
        priority = 1_000_000.0 + opp_score + 0.5 * my_score
    elif my_o3 >= 2:
        priority = 500_000.0 + my_score + 0.5 * opp_score
    elif opp_o3 >= 2:
        priority = 300_000.0 + opp_score + 0.5 * my_score
    else:
        center = (size - 1) / 2.0
        center_bonus = 5.0 - 0.5 * (abs(r - center) + abs(c - center))
        priority = my_score + 0.95 * opp_score + center_bonus

    return False, False, priority


class MCTSNode:
    """Nút trong cây tìm kiếm Monte Carlo (MCTS)."""

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action_from_parent: int | None = None,
        player_just_moved: int = 0,
        untried_actions: list[int] | None = None,
        prior: float = 0.0,
        untried_priors: dict[int, float] | None = None,
    ):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.player_just_moved = player_just_moved
        self.children: dict[int, MCTSNode] = {}
        self._actions_initialized: bool = untried_actions is not None
        self.untried_actions: list[int] = untried_actions if untried_actions is not None else []
        self.untried_priors: dict[int, float] = untried_priors if untried_priors is not None else {}
        self.prior: float = float(prior)
        self.visits: int = 0
        self.wins: float = 0.0

    def can_expand(
        self,
        use_progressive_widening: bool = False,
        pw_c: float = 1.2,
        pw_alpha: float = 0.38,
        pw_min_children: int = 4,
    ) -> bool:
        """Kiểm tra nút hiện tại có thể mở rộng thêm nhánh con mới hay không."""
        if not self._actions_initialized:
            return True
        if not self.untried_actions:
            return False
        if not use_progressive_widening:
            return True
        max_allowed = max(pw_min_children, int(pw_c * (self.visits ** pw_alpha)))
        return len(self.children) < max_allowed

    def select_child_uct(
        self,
        exploration_c: float,
        progressive_bias_weight: float = 0.0,
    ) -> MCTSNode:
        """Chọn nút con tối đa hóa công thức UCT (kèm Progressive Bias tùy chọn)."""
        log_parent_visits = math.log(max(1, self.visits))
        best_score = -math.inf
        best_child: MCTSNode | None = None

        if progressive_bias_weight > 0.0:
            for child in self.children.values():
                inv_visits = 1.0 / child.visits
                exploitation = child.wins * inv_visits
                exploration = exploration_c * math.sqrt(log_parent_visits * inv_visits)
                bias = progressive_bias_weight * child.prior / (child.visits + 1.0)
                uct_score = exploitation + exploration + bias
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child
        else:
            for child in self.children.values():
                inv_visits = 1.0 / child.visits
                exploitation = child.wins * inv_visits
                exploration = exploration_c * math.sqrt(log_parent_visits * inv_visits)
                uct_score = exploitation + exploration
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child

        assert best_child is not None
        return best_child


class MCTSAgent(BaseAgent):
    """
    Tác tử Cấp 4: Monte Carlo Tree Search (UCT) nâng cấp cho Caro 11x11.

    Các thành phần cải tiến (đều có công tắc bật/tắt phục vụ Ablation Study):
      - Chuẩn hóa thang thưởng UCT về [0, 1] tương khớp hằng số `exploration_c = sqrt(2)`.
      - Khởi tạo lười (lazy expansion) cho `untried_actions` để tránh chấm điểm ứng viên
        trên các nút lá chỉ được ghé thăm 1 lần.
      - Mô phỏng tăng tốc trên mảng phẳng + tập ứng viên cập nhật gia tăng O(1).
      - Heavy playout chiến thuật (`rollout_policy="heuristic"`): phát hiện thắng 5,
        chặn 5, tạo/chặn 4 mở (`open_4`) quanh vùng giao tranh + tournament selection k ô.
      - Đánh giá lá bằng heuristic (`use_heuristic_leaf=True`): khi rollout đạt
        `max_rollout_steps` chưa kết thúc, dùng `tanh(evaluate_board(..., consider_turn=True) / scale)`.
      - Sắp xếp prior (`use_prior_ordering=True`) + Progressive Widening (`use_progressive_widening=True`)
        + Progressive Bias (`progressive_bias_weight`) để tập trung độ sâu vào các nhánh
        chiến thuật trọng yếu mà vẫn giữ tính đầy đủ tiệm cận của UCT.
    """

    def __init__(
        self,
        num_rollouts: int | None = None,
        exploration_c: float = 1.414,
        time_limit_sec: float | None = None,
        candidate_radius: int = 2,
        rollout_radius: int = 1,
        max_rollout_steps: int = 30,
        use_tactical_shortcut: bool = True,
        seed: int | None = 42,
        name: str = "Level4_MCTS_UCT",
        rollout_policy: str = "heuristic",
        use_heuristic_leaf: bool = True,
        use_prior_ordering: bool = True,
        use_progressive_widening: bool = True,
        pw_c: float = 1.2,
        pw_alpha: float = 0.38,
        pw_min_children: int = 4,
        progressive_bias_weight: float = 0.6,
        heuristic_scale: float = 12000.0,
        rollout_sample_k: int = 5,
    ):
        super().__init__(name=name)
        if num_rollouts is None and time_limit_sec is None:
            num_rollouts = 300

        if num_rollouts is not None:
            if (
                isinstance(num_rollouts, bool)
                or not isinstance(num_rollouts, (int, np.integer))
                or int(num_rollouts) <= 0
            ):
                raise ValueError("num_rollouts phải là số nguyên dương (> 0).")
            num_rollouts = int(num_rollouts)

        if time_limit_sec is not None:
            if (
                isinstance(time_limit_sec, bool)
                or not isinstance(time_limit_sec, (int, float, np.number))
                or not math.isfinite(float(time_limit_sec))
                or float(time_limit_sec) <= 0.0
            ):
                raise ValueError("time_limit_sec phải là số thực hữu hạn lớn hơn 0.")
            time_limit_sec = float(time_limit_sec)

        if (
            isinstance(exploration_c, bool)
            or not isinstance(exploration_c, (int, float, np.number))
            or not math.isfinite(float(exploration_c))
            or float(exploration_c) < 0.0
        ):
            raise ValueError("exploration_c phải là số thực hữu hạn và >= 0.")

        if (
            isinstance(candidate_radius, bool)
            or not isinstance(candidate_radius, (int, np.integer))
            or int(candidate_radius) < 0
        ):
            raise ValueError("candidate_radius phải là số nguyên không âm (>= 0).")

        if (
            isinstance(rollout_radius, bool)
            or not isinstance(rollout_radius, (int, np.integer))
            or int(rollout_radius) < 0
        ):
            raise ValueError("rollout_radius phải là số nguyên không âm (>= 0).")

        if (
            isinstance(max_rollout_steps, bool)
            or not isinstance(max_rollout_steps, (int, np.integer))
            or int(max_rollout_steps) <= 0
        ):
            raise ValueError("max_rollout_steps phải là số nguyên dương (> 0).")

        if rollout_policy not in ("heuristic", "random"):
            raise ValueError("rollout_policy phải là 'heuristic' hoặc 'random'.")

        if (
            isinstance(pw_c, bool)
            or not isinstance(pw_c, (int, float, np.number))
            or not math.isfinite(float(pw_c))
            or float(pw_c) <= 0.0
        ):
            raise ValueError("pw_c phải là số thực hữu hạn dương.")

        if (
            isinstance(pw_alpha, bool)
            or not isinstance(pw_alpha, (int, float, np.number))
            or not math.isfinite(float(pw_alpha))
            or not (0.0 < float(pw_alpha) <= 1.0)
        ):
            raise ValueError("pw_alpha phải nằm trong (0, 1].")

        if (
            isinstance(pw_min_children, bool)
            or not isinstance(pw_min_children, (int, np.integer))
            or int(pw_min_children) < 1
        ):
            raise ValueError("pw_min_children phải là số nguyên >= 1.")

        if (
            isinstance(progressive_bias_weight, bool)
            or not isinstance(progressive_bias_weight, (int, float, np.number))
            or not math.isfinite(float(progressive_bias_weight))
            or float(progressive_bias_weight) < 0.0
        ):
            raise ValueError("progressive_bias_weight phải là số thực hữu hạn >= 0.")

        if (
            isinstance(heuristic_scale, bool)
            or not isinstance(heuristic_scale, (int, float, np.number))
            or not math.isfinite(float(heuristic_scale))
            or float(heuristic_scale) <= 0.0
        ):
            raise ValueError("heuristic_scale phải là số thực hữu hạn dương.")

        if (
            isinstance(rollout_sample_k, bool)
            or not isinstance(rollout_sample_k, (int, np.integer))
            or int(rollout_sample_k) < 1
        ):
            raise ValueError("rollout_sample_k phải là số nguyên >= 1.")

        self.num_rollouts: int | None = num_rollouts
        self.exploration_c: float = float(exploration_c)
        self.time_limit_sec: float | None = time_limit_sec
        self.candidate_radius: int = int(candidate_radius)
        self.rollout_radius: int = int(rollout_radius)
        self.max_rollout_steps: int = int(max_rollout_steps)
        self.use_tactical_shortcut: bool = bool(use_tactical_shortcut)
        self.seed: int | None = seed

        self.rollout_policy: str = rollout_policy
        self.use_heuristic_leaf: bool = bool(use_heuristic_leaf)
        self.use_prior_ordering: bool = bool(use_prior_ordering)
        self.use_progressive_widening: bool = bool(use_progressive_widening)
        self.pw_c: float = float(pw_c)
        self.pw_alpha: float = float(pw_alpha)
        self.pw_min_children: int = int(pw_min_children)
        self.progressive_bias_weight: float = float(progressive_bias_weight)
        self.heuristic_scale: float = float(heuristic_scale)
        self.rollout_sample_k: int = int(rollout_sample_k)

        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(int(self.rng.integers(0, (1 << 63) - 1)))

        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_root: MCTSNode | None = None
        self.last_decision_stage: str = ""

    def get_root_stats(self) -> list[dict]:
        """Trả về danh sách thống kê các nhánh con tại nút gốc ([] nếu last_root là None)."""
        if self.last_root is None:
            return []
        stats = []
        for action, child in self.last_root.children.items():
            value = child.wins / child.visits if child.visits > 0 else 0.0
            stats.append(
                {
                    "action": int(action),
                    "visits": int(child.visits),
                    "wins": float(child.wins),
                    "value": float(value),
                }
            )
        stats.sort(key=lambda item: (item["visits"], item["value"]), reverse=True)
        return stats

    def _check_urgent_tactical_move(self, env: CaroEnv, candidates: list[int]) -> tuple[int | None, str]:
        """Kiểm tra nước thắng ngay 5 quân hoặc chặn đối thủ thắng 5 quân tại gốc."""
        my_player = env.current_player
        opponent = -my_player

        for action in candidates:
            if evaluate_cell_threat(env, action, my_player)["is_win_5"]:
                return int(action), "Shortcut: immediate win"
        for action in candidates:
            if evaluate_cell_threat(env, action, opponent)["is_win_5"]:
                return int(action), "Shortcut: block immediate win"
        return None, ""

    def _prepare_untried_actions(
        self,
        sim_env: CaroEnv,
        candidates: list[int] | None = None,
    ) -> tuple[list[int], dict[int, float]]:
        """
        Chuẩn bị danh sách `untried_actions` và bảng `priors` trong [0, 1] cho một nút MCTS.
        Nếu bật `use_prior_ordering`, sắp xếp TĂNG DẦN theo điểm ưu tiên chiến thuật
        sao cho `.pop()` mở rộng nước mạnh nhất trước.
        """
        if candidates is None:
            if sim_env.is_terminal():
                return [], {}
            candidates = get_candidate_actions(sim_env, radius=self.candidate_radius)

        if len(candidates) <= 1:
            acts = [int(a) for a in candidates]
            return acts, {a: 1.0 for a in acts}

        if not self.use_prior_ordering:
            shuffled = [int(a) for a in candidates]
            self._py_rng.shuffle(shuffled)
            return shuffled, {}

        flat = sim_env.board.ravel().tolist()
        size = sim_env.board_size
        win_len = sim_env.win_length
        my_player = int(sim_env.current_player)
        opp_player = -my_player
        rng_rand = self._py_rng.random

        scored: list[tuple[float, int]] = []
        max_log = 1e-9
        raw_logs: list[tuple[int, float]] = []

        for action in candidates:
            act = int(action)
            _, _, prio = _quick_cell_score_flat(flat, size, win_len, act, my_player, opp_player)
            scored.append((prio + rng_rand() * 1e-3, act))
            lg = math.log1p(max(0.0, prio))
            if lg > max_log:
                max_log = lg
            raw_logs.append((act, lg))

        # Sắp tăng dần để `untried_actions.pop()` luôn lấy nước có độ ưu tiên cao nhất trước
        scored.sort(key=lambda item: item[0])
        priors = {act: lg / max_log for act, lg in raw_logs}
        return [act for _, act in scored], priors

    def _rollout_policy(self, sim_env: CaroEnv) -> float:
        """
        Bước 3 (Simulation): Mô phỏng tăng tốc trực tiếp trên mảng phẳng + tập ứng viên
        cập nhật gia tăng O(1). Trả về điểm trong [-1.0, +1.0] theo góc nhìn người chơi 1 (X).
        """
        if sim_env.is_terminal():
            return float(sim_env.result(perspective=1))

        size = sim_env.board_size
        win_len = sim_env.win_length
        num_cells = size * size
        flat = sim_env.board.ravel().tolist()
        current_player = int(sim_env.current_player)
        py_rng = self._py_rng
        use_heuristic_rollout = self.rollout_policy == "heuristic"
        sample_k = self.rollout_sample_k

        # Khởi tạo tập ứng viên rollout gia tăng (cand_list + cand_pos cho xóa/thêm O(1))
        occupied = [i for i, v in enumerate(flat) if v != 0]
        moves_played = len(occupied)
        cand_list: list[int] = []
        cand_pos: dict[int, int] = {}

        if self.rollout_radius == 0:
            neighbors: tuple[tuple[int, ...], ...] | None = None
            for i, v in enumerate(flat):
                if v == 0:
                    cand_pos[i] = len(cand_list)
                    cand_list.append(i)
        else:
            neighbors = _get_neighbors(size, self.rollout_radius)
            if not occupied:
                center = (size // 2) * size + (size // 2)
                cand_pos[center] = 0
                cand_list.append(center)
            else:
                for occ in occupied:
                    for nb in neighbors[occ]:
                        if flat[nb] == 0 and nb not in cand_pos:
                            cand_pos[nb] = len(cand_list)
                            cand_list.append(nb)
                if not cand_list:
                    for i, v in enumerate(flat):
                        if v == 0:
                            cand_pos[i] = len(cand_list)
                            cand_list.append(i)

        # Theo dõi 3 nước gần nhất của mỗi bên để phát hiện tức thì thắng 5 / chặn 5 / tạo 4 mở / chặn 4 mở
        last_by_p: dict[int, list[int]] = {1: [], -1: []}
        if use_heuristic_rollout:
            for mv in sim_env.history[-6:]:
                c_mv = flat[mv]
                if c_mv in (1, -1):
                    lst = last_by_p[c_mv]
                    lst.append(mv)
                    if len(lst) > 3:
                        lst.pop(0)

        steps = 0
        max_steps = self.max_rollout_steps

        while steps < max_steps and cand_list:
            n_cand = len(cand_list)
            if not use_heuristic_rollout or n_cand == 1:
                action = cand_list[py_rng.randrange(n_cand)]
            else:
                opp_player = -current_player
                chosen = -1
                block_win = -1
                my_open4 = -1
                block_open4 = -1

                # 1. Quét đe dọa cấp bách trực tiếp theo 4 phương từ các quân gần nhất của mỗi bên
                for st in last_by_p[current_player]:
                    w_cell, o4_cell = _scan_stone_threats_flat(flat, size, win_len, st)
                    if w_cell >= 0:
                        chosen = w_cell
                        break
                    if o4_cell >= 0 and my_open4 < 0:
                        my_open4 = o4_cell

                if chosen < 0:
                    for st in last_by_p[opp_player]:
                        w_cell, o4_cell = _scan_stone_threats_flat(flat, size, win_len, st)
                        if w_cell >= 0 and block_win < 0:
                            block_win = w_cell
                        if o4_cell >= 0 and block_open4 < 0:
                            block_open4 = o4_cell

                if chosen < 0:
                    if block_win >= 0:
                        chosen = block_win
                    elif my_open4 >= 0:
                        # Không có đe dọa thắng ngay từ đối thủ và ta tạo được open-4 -> thắng cưỡng bức
                        return 0.98 if current_player == 1 else -0.98
                    elif block_open4 >= 0:
                        chosen = block_open4
                    else:
                        # 2. Tournament selection k ứng viên ngẫu nhiên: O(k) mỗi bước rollout
                        k = sample_k if sample_k < n_cand else n_cand
                        best_act = cand_list[py_rng.randrange(n_cand)]
                        if k > 1:
                            _, _, best_p = _quick_cell_score_flat(
                                flat, size, win_len, best_act, current_player, opp_player
                            )
                            for _ in range(k - 1):
                                cand = cand_list[py_rng.randrange(n_cand)]
                                if cand == best_act:
                                    continue
                                my_w, _, prio = _quick_cell_score_flat(
                                    flat, size, win_len, cand, current_player, opp_player
                                )
                                if my_w:
                                    best_act = cand
                                    break
                                if prio > best_p:
                                    best_p = prio
                                    best_act = cand
                        chosen = best_act

                action = chosen

            # Đặt quân trực tiếp lên mảng phẳng
            flat[action] = current_player
            moves_played += 1
            steps += 1

            # Kiểm tra kết thúc ngay lập tức
            if _check_win_flat(flat, size, win_len, action, current_player):
                return 1.0 if current_player == 1 else -1.0
            if moves_played >= num_cells:
                return 0.0

            # Cập nhật tập ứng viên gia tăng O(1)
            pos = cand_pos.pop(action, -1)
            if pos >= 0:
                last_elem = cand_list.pop()
                if pos < len(cand_list):
                    cand_list[pos] = last_elem
                    cand_pos[last_elem] = pos

            if neighbors is not None:
                for nb in neighbors[action]:
                    if flat[nb] == 0 and nb not in cand_pos:
                        cand_pos[nb] = len(cand_list)
                        cand_list.append(nb)

            if use_heuristic_rollout:
                lst = last_by_p[current_player]
                lst.append(action)
                if len(lst) > 3:
                    lst.pop(0)
            current_player = -current_player

        # Rollout chạm max_rollout_steps nhưng chưa kết thúc: đánh giá heuristic lá thay vì trả 0.0 nhiễu
        if self.use_heuristic_leaf:
            sim_env.board = np.asarray(flat, dtype=np.int8).reshape((size, size))
            sim_env.current_player = current_player
            leaf_score = evaluate_board(sim_env, perspective=1, consider_turn=True)
            return math.tanh(leaf_score / self.heuristic_scale)

        return 0.0

    def select_action(self, env: CaroEnv) -> int:
        t0 = time.perf_counter()
        self.last_search_count = 0
        self.last_root = None
        self.last_decision_stage = ""

        try:
            candidates = get_candidate_actions(env, radius=self.candidate_radius)
            if not candidates:
                raise ValueError("Không còn hành động hợp lệ trên bàn cờ.")

            if len(candidates) == 1:
                self.last_decision_stage = "Shortcut: single candidate"
                return int(candidates[0])

            if self.use_tactical_shortcut:
                urgent_action, stage = self._check_urgent_tactical_move(env, candidates)
                if urgent_action is not None:
                    self.last_decision_stage = stage
                    return int(urgent_action)

            root_untried, root_priors = self._prepare_untried_actions(env, candidates=candidates)
            root = MCTSNode(
                parent=None,
                action_from_parent=None,
                player_just_moved=-env.current_player,
                untried_actions=root_untried,
                untried_priors=root_priors,
            )

            use_pw = self.use_progressive_widening
            pw_c = self.pw_c
            pw_alpha = self.pw_alpha
            pw_min = self.pw_min_children
            c_param = self.exploration_c
            pb_weight = self.progressive_bias_weight if self.use_prior_ordering else 0.0

            rollouts_done = 0
            while True:
                if self.num_rollouts is not None and rollouts_done >= self.num_rollouts:
                    break
                if (
                    self.time_limit_sec is not None
                    and rollouts_done > 0
                    and (time.perf_counter() - t0) >= self.time_limit_sec
                ):
                    break

                node = root
                sim_env = env.clone()

                # 1. Selection (UCT + Progressive Bias + Progressive Widening)
                while (
                    not node.can_expand(use_pw, pw_c, pw_alpha, pw_min)
                    and node.children
                    and not sim_env.is_terminal()
                ):
                    node = node.select_child_uct(c_param, pb_weight)
                    sim_env.step(node.action_from_parent)

                # 2. Expansion (khởi tạo lười untried_actions khi nút thực sự được mở rộng)
                if not sim_env.is_terminal():
                    if not node._actions_initialized:
                        untried, priors = self._prepare_untried_actions(sim_env)
                        node.untried_actions = untried
                        node.untried_priors = priors
                        node._actions_initialized = True

                    if node.can_expand(use_pw, pw_c, pw_alpha, pw_min):
                        action = node.untried_actions.pop()
                        mover = sim_env.current_player
                        sim_env.step(action)
                        child_prior = node.untried_priors.get(action, 0.0)
                        child_node = MCTSNode(
                            parent=node,
                            action_from_parent=action,
                            player_just_moved=mover,
                            untried_actions=None,  # Lazy: chỉ khởi tạo nếu nút này được thăm lại
                            prior=child_prior,
                        )
                        node.children[action] = child_node
                        node = child_node

                # 3. Simulation
                outcome_for_p1 = self._rollout_policy(sim_env)

                # 4. Backpropagation (chuẩn hóa thang thưởng về [0.0, 1.0])
                reward_p1 = 0.5 * (outcome_for_p1 + 1.0)
                reward_p2 = 1.0 - reward_p1
                curr: MCTSNode | None = node
                while curr is not None:
                    curr.visits += 1
                    curr.wins += reward_p1 if curr.player_just_moved == 1 else reward_p2
                    curr = curr.parent

                rollouts_done += 1

            self.last_root = root
            self.last_search_count = rollouts_done
            self.last_decision_stage = f"MCTS-UCT: {rollouts_done} rollouts"

            best_child = max(
                root.children.values(),
                key=lambda c: (c.visits, c.wins / max(1, c.visits)),
            )
            return int(best_child.action_from_parent)
        finally:
            self.last_move_time_ms = (time.perf_counter() - t0) * 1000.0
