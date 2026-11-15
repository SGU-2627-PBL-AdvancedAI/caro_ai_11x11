
import random
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed=None, name="Random"):
        super().__init__(name)
        self.rng = random.Random(seed)

    def select_action(self, env):
        legal = env.legal_actions()

        if not legal:
            raise RuntimeError("Không còn nước đi hợp lệ.")

        return self.rng.choice(legal)
