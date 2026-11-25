from __future__ import annotations

import random
import time

try:
    from .base_agent import BaseAgent
    from ..heuristic import get_candidate_actions
except ImportError:
    from agents.base_agent import BaseAgent
    from heuristic import get_candidate_actions


class RandomAgent(BaseAgent):
    """
    Tác tử Cấp 1: Random Baseline cho Caro 11x11.

    Mặc định (`candidate_radius=None`) chọn ngẫu nhiên đều (uniform) trên toàn bộ
    các nước đi hợp lệ `env.legal_actions()`, có hỗ trợ `seed` để tái lập kết quả
    thực nghiệm và ghi nhận telemetry (`last_move_time_ms`, `last_search_count`,
    `last_decision_stage`) đồng bộ với các tác tử cấp cao hơn.

    Nếu truyền `candidate_radius >= 1`, tác tử sẽ lấy mẫu ngẫu nhiên trong vùng
    ứng viên cục bộ quanh các quân đã đặt (`get_candidate_actions`) phục vụ
    thử nghiệm mở rộng / ablation.
    """

    def __init__(
        self,
        seed: int | None = None,
        name: str = "Random",
        candidate_radius: int | None = None,
    ):
        super().__init__(name)

        if candidate_radius is not None:
            if isinstance(candidate_radius, bool) or not isinstance(candidate_radius, int):
                raise ValueError("candidate_radius phải là số nguyên dương hoặc None.")
            if candidate_radius < 1:
                raise ValueError("candidate_radius phải >= 1.")

        self.seed: int | None = seed
        self.candidate_radius: int | None = candidate_radius
        self.rng = random.Random(seed)

        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_decision_stage: str = ""

    def select_action(self, env) -> int:
        """Chọn ngẫu nhiên một hành động hợp lệ từ trạng thái `env` hiện tại."""
        started = time.perf_counter()
        self.last_search_count = 0
        self.last_decision_stage = ""

        try:
            if self.candidate_radius is None:
                actions = env.legal_actions()
                stage = "uniform_random"
            else:
                actions = get_candidate_actions(env, radius=self.candidate_radius)
                stage = f"local_random_r{self.candidate_radius}"

            if not actions:
                raise ValueError("Không còn nước đi hợp lệ.")

            self.last_search_count = 1
            self.last_decision_stage = stage
            return int(self.rng.choice(actions))
        finally:
            self.last_move_time_ms = (time.perf_counter() - started) * 1000.0
