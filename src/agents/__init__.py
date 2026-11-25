"""Gói chứa lớp cơ sở và đầy đủ 7 cấp tác tử AI cho Caro 11x11 (Cấp 1–6 + Cấp 7 Bonus AlphaGo)."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from .alphago_agent import AlphaGoAgent
    from .cnn_agent import CNNAgent
    from .mcts_agent import MCTSAgent
    from .minimax_agent import MinimaxAgent
    from .random_agent import RandomAgent
    from .rl_agent import RLAgent
    from .rule_based_agent import RuleBasedAgent

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "RandomAgent": (".random_agent", "RandomAgent"),
    "RuleBasedAgent": (".rule_based_agent", "RuleBasedAgent"),
    "MinimaxAgent": (".minimax_agent", "MinimaxAgent"),
    "MCTSAgent": (".mcts_agent", "MCTSAgent"),
    "CNNAgent": (".cnn_agent", "CNNAgent"),
    "RLAgent": (".rl_agent", "RLAgent"),
    "AlphaGoAgent": (".alphago_agent", "AlphaGoAgent"),
}

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "RuleBasedAgent",
    "MinimaxAgent",
    "MCTSAgent",
    "CNNAgent",
]
if find_spec("torch") is not None:
    __all__.extend(["RLAgent", "AlphaGoAgent"])


def __getattr__(name: str) -> Any:
    """Nạp lười (PEP 562) từng agent để tránh kéo theo PyTorch/TensorFlow khi không cần."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        try:
            module = import_module(module_name, package=__name__)
        except ModuleNotFoundError as exc:
            if name in ("RLAgent", "AlphaGoAgent") and exc.name == "torch":
                raise ImportError(
                    f"{name} yêu cầu PyTorch (`torch`). Hãy cài đặt PyTorch trước khi sử dụng."
                ) from exc
            raise
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
