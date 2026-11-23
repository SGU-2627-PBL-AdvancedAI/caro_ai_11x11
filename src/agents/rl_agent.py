from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from .base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent


def encode_board_state(board: np.ndarray, current_player: int) -> np.ndarray:
    """
    Mã hóa bàn cờ theo góc nhìn chuẩn hóa (Canonical State) của người đang tới lượt.
    Đầu ra có shape (3, H, W), dtype float32:
      - Kênh 0: 1.0 tại các ô có quân của mình (current_player)
      - Kênh 1: 1.0 tại các ô có quân đối thủ (-current_player)
      - Kênh 2: 1.0 tại các ô trống (0)
    """
    my_pieces = (board == current_player).astype(np.float32)
    opp_pieces = (board == -current_player).astype(np.float32)
    empty_cells = (board == 0).astype(np.float32)
    return np.stack([my_pieces, opp_pieces, empty_cells], axis=0)


class DuelingCaroDQN(nn.Module):
    """
    Mạng Dueling CNN DQN cho bàn cờ Caro N x N:
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, .))).
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (64, 64, 128),
        board_size: int = 11,
    ) -> None:
        super().__init__()
        self.board_size = board_size

        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in conv_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.backbone = nn.Sequential(*layers)

        # Nhánh Value V(s) -> (B, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(c_in, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * board_size * board_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Nhánh Advantage A(s, a) -> (B, N*N)
        self.advantage_head = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        value = self.value_head(feat)  # (B, 1)
        advantage = self.advantage_head(feat).flatten(start_dim=1)  # (B, N*N)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RLAgent(BaseAgent):
    """
    Tác tử Cấp 6: Deep Reinforcement Learning (Dueling Double DQN) cho Caro 11x11.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (64, 64, 128),
        board_size: int = 11,
        device: str = "auto",
        name: str = "Level6_RL_DQN",
    ) -> None:
        super().__init__(name=name)
        self.device = resolve_device(device)
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else None
        self.in_channels = in_channels
        self.conv_channels = tuple(conv_channels)
        self.board_size = board_size

        state_dict = None
        if checkpoint_path is not None:
            ckpt_file = Path(checkpoint_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.in_channels = ckpt.get("in_channels", self.in_channels)
                self.conv_channels = tuple(ckpt.get("conv_channels", self.conv_channels))
                self.board_size = ckpt.get("board_size", self.board_size)
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

        self.model = DuelingCaroDQN(
            in_channels=self.in_channels,
            conv_channels=self.conv_channels,
            board_size=self.board_size,
        ).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()

        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0

    def _predict_q_values(self, env) -> np.ndarray:
        state = encode_board_state(env.board, env.current_player)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    def get_masked_q_values(self, env) -> np.ndarray:
        legal_actions = env.legal_actions()
        q_values = self._predict_q_values(env)
        masked_q = np.full_like(q_values, -np.inf, dtype=np.float32)
        if legal_actions:
            masked_q[legal_actions] = q_values[legal_actions]
        return masked_q

    def select_action(self, env) -> int:
        start_time = time.perf_counter()
        legal_actions = env.legal_actions()
        if not legal_actions:
            raise ValueError("Không còn nước đi hợp lệ nào trên bàn cờ.")

        masked_q = self.get_masked_q_values(env)
        best_action = int(np.argmax(masked_q))

        self.last_search_count = 1
        self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0
        return best_action

    def get_top_actions(self, env, top_k: int = 5) -> list[dict]:
        legal_actions = env.legal_actions()
        if not legal_actions:
            return []

        q_values = self._predict_q_values(env)
        n = env.board_size
        ranked = sorted(legal_actions, key=lambda a: float(q_values[a]), reverse=True)[:top_k]
        return [
            {
                "action": int(a),
                "row": int(a // n),
                "col": int(a % n),
                "q_value": float(q_values[a]),
            }
            for a in ranked
        ]