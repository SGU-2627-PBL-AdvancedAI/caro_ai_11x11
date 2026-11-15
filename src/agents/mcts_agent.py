from __future__ import annotations
import math
import time
import numpy as np
from agents.base_agent import BaseAgent
from caro_env import CaroEnv
from heuristic import get_candidate_actions, evaluate_cell_threat


class MCTSNode:
    """Nút trong cây tìm kiếm Monte Carlo (MCTS)."""

    def __init__(
        self,
        parent: MCTSNode | None = None,
        action_from_parent: int | None = None,
        player_just_moved: int = 0,
        untried_actions: list[int] | None = None,
    ):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.player_just_moved = player_just_moved
        self.children: dict[int, MCTSNode] = {}
        self.untried_actions: list[int] = untried_actions if untried_actions is not None else []
        self.visits: int = 0
        self.wins: float = 0.0

    def select_child_uct(self, exploration_c: float) -> MCTSNode:
        """Chọn nút con tối đa hóa công thức UCT."""
        log_parent_visits = math.log(max(1, self.visits))
        best_score = -math.inf
        best_child = None

        for child in self.children.values():
            exploitation = child.wins / child.visits
            exploration = exploration_c * math.sqrt(log_parent_visits / child.visits)
            uct_score = exploitation + exploration
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child


class MCTSAgent(BaseAgent):
    """
    Tác tử Cấp 4: Monte Carlo Tree Search (UCT) cho bàn cờ Caro 11x11.
    Hỗ trợ giới hạn theo num_rollouts, time_limit_sec hoặc kết hợp cả hai.
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

        self.num_rollouts: int | None = num_rollouts
        self.exploration_c: float = float(exploration_c)
        self.time_limit_sec: float | None = time_limit_sec
        self.candidate_radius: int = int(candidate_radius)
        self.rollout_radius: int = int(rollout_radius)
        self.max_rollout_steps: int = int(max_rollout_steps)
        self.use_tactical_shortcut = use_tactical_shortcut
        self.rng = np.random.default_rng(seed)

        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_root: MCTSNode | None = None

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

    def _check_urgent_tactical_move(self, env: CaroEnv, candidates: list[int]) -> int | None:
        """Kiểm tra nước thắng ngay 5 quân hoặc chặn đối thủ thắng 5 quân tại gốc."""
        my_player = env.current_player
        opponent = -my_player

        for action in candidates:
            if evaluate_cell_threat(env, action, my_player)["is_win_5"]:
                return action
        for action in candidates:
            if evaluate_cell_threat(env, action, opponent)["is_win_5"]:
                return action
        return None

    def _rollout_policy(self, sim_env: CaroEnv) -> float:
        """Bước 3 (Simulation): Mô phỏng nhanh trong vùng lân cận tới khi kết thúc hoặc đạt giới hạn bước."""
        steps = 0
        while not sim_env.is_terminal() and steps < self.max_rollout_steps:
            candidates = get_candidate_actions(sim_env, radius=self.rollout_radius)
            if not candidates:
                break
            action = int(self.rng.choice(candidates))
            sim_env.step(action)
            steps += 1

        if sim_env.is_terminal():
            return sim_env.result(perspective=1)
        return 0.0

    def select_action(self, env: CaroEnv) -> int:
        t0 = time.perf_counter()
        self.last_search_count = 0
        self.last_root = None

        try:
            candidates = get_candidate_actions(env, radius=self.candidate_radius)
            if not candidates:
                raise ValueError("Không còn hành động hợp lệ trên bàn cờ.")

            if len(candidates) == 1:
                return int(candidates[0])

            if self.use_tactical_shortcut:
                urgent_action = self._check_urgent_tactical_move(env, candidates)
                if urgent_action is not None:
                    return int(urgent_action)

            shuffled_candidates = candidates.copy()
            self.rng.shuffle(shuffled_candidates)
            root = MCTSNode(
                parent=None,
                action_from_parent=None,
                player_just_moved=-env.current_player,
                untried_actions=shuffled_candidates,
            )

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

                # 1. Selection
                while not node.untried_actions and node.children and not sim_env.is_terminal():
                    node = node.select_child_uct(self.exploration_c)
                    sim_env.step(node.action_from_parent)

                # 2. Expansion
                if node.untried_actions and not sim_env.is_terminal():
                    action = node.untried_actions.pop()
                    mover = sim_env.current_player
                    sim_env.step(action)
                    next_candidates = get_candidate_actions(sim_env, radius=self.candidate_radius)
                    self.rng.shuffle(next_candidates)
                    child_node = MCTSNode(
                        parent=node,
                        action_from_parent=action,
                        player_just_moved=mover,
                        untried_actions=next_candidates,
                    )
                    node.children[action] = child_node
                    node = child_node

                # 3. Simulation
                outcome_for_p1 = self._rollout_policy(sim_env)

                # 4. Backpropagation
                curr = node
                while curr is not None:
                    curr.visits += 1
                    if outcome_for_p1 != 0.0:
                        if (curr.player_just_moved == 1 and outcome_for_p1 > 0) or (
                            curr.player_just_moved == -1 and outcome_for_p1 < 0
                        ):
                            curr.wins += 1.0
                        else:
                            curr.wins -= 1.0
                    curr = curr.parent

                rollouts_done += 1

            self.last_root = root
            self.last_search_count = rollouts_done

            best_child = max(
                root.children.values(),
                key=lambda c: (c.visits, c.wins / max(1, c.visits)),
            )
            return int(best_child.action_from_parent)
        finally:
            self.last_move_time_ms = (time.perf_counter() - t0) * 1000.0