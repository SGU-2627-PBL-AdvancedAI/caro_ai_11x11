from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Lớp cơ sở trừu tượng cho mọi tác tử AI (Cấp 1 -> Cấp 7)."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, env) -> int:
        """Nhận môi trường CaroEnv hiện tại và trả về hành động hợp lệ (0..120)."""
        raise NotImplementedError