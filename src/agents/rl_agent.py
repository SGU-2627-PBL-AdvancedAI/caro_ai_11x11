from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .base_agent import BaseAgent
    from ..heuristic import get_candidate_actions
except ImportError:
    from agents.base_agent import BaseAgent
    from heuristic import get_candidate_actions


DEFAULT_RL_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "06_rl_best.pt"


def encode_board_state(
    board: np.ndarray,
    current_player: int,
    last_action: int | None = None,
    in_channels: int = 3,
) -> np.ndarray:
    """
    Mã hóa bàn cờ theo góc nhìn chuẩn hóa (Canonical State) của người đang tới lượt.
    Đầu ra có shape `(in_channels, H, W)`, dtype `float32`:
      - Kênh 0: 1.0 tại các ô có quân của mình (`current_player`)
      - Kênh 1: 1.0 tại các ô có quân đối thủ (`-current_player`)
      - Kênh 2: 1.0 tại các ô trống (`0`)
      - Kênh 3 (nếu `in_channels == 4`): 1.0 tại ô nước đi gần nhất (`last_action`)
    """
    if in_channels not in (3, 4):
        raise ValueError(f"in_channels phải là 3 hoặc 4, nhận được: {in_channels}")

    arr = np.asarray(board, dtype=np.int8)
    n = arr.shape[0]
    my_pieces = (arr == current_player).astype(np.float32)
    opp_pieces = (arr == -current_player).astype(np.float32)
    empty_cells = (arr == 0).astype(np.float32)

    if in_channels == 3:
        return np.stack([my_pieces, opp_pieces, empty_cells], axis=0)

    last_plane = np.zeros((n, n), dtype=np.float32)
    if last_action is not None and 0 <= int(last_action) < n * n:
        lr, lc = divmod(int(last_action), n)
        last_plane[lr, lc] = 1.0
    return np.stack([my_pieces, opp_pieces, empty_cells, last_plane], axis=0)


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        groups = 8
        while groups > 1 and (num_channels % groups != 0):
            groups //= 2
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"norm_type không hợp lệ: {norm_type!r} (chọn 'group', 'batch', hoặc 'none').")


class _ResGNBlock(nn.Module):
    """Khối Residual 3x3 dùng GroupNorm (ổn định tuyệt đối dưới Polyak target update trong DQN)."""

    def __init__(self, channels: int, norm_type: str = "group") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _make_norm(norm_type, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _make_norm(norm_type, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class DuelingCaroDQN(nn.Module):
    """
    Mạng Dueling CNN DQN cho bàn cờ Caro N x N (Phương án 6A):
      - Hỗ trợ kiến trúc nâng cấp Residual + GroupNorm (`use_residual=True`, `norm_type="group"`)
        với trường nhìn phủ trọn bàn cờ 11x11 và ổn định dưới Soft Target Update (Polyak).
      - Hỗ trợ tương thích ngược với cấu trúc Conv tuần tự cũ (`use_residual=False`).
      - **Masked Dueling Mean**: chỉ trừ trung bình nhánh Advantage trên các ô trống hợp lệ
        (tránh các ô đã có quân làm méo V(s) và Q(s, a)).
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (64, 64, 128),
        board_size: int = 11,
        use_residual: bool = False,
        num_res_blocks: int = 3,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.board_size = int(board_size)
        self.use_residual = bool(use_residual)
        self.num_res_blocks = int(num_res_blocks)
        self.norm_type = str(norm_type)

        if self.use_residual:
            res_ch = self.conv_channels[0] if self.conv_channels else 48
            layers: list[nn.Module] = [
                nn.Conv2d(self.in_channels, res_ch, kernel_size=3, padding=1, bias=False),
                _make_norm(self.norm_type, res_ch),
                nn.ReLU(inplace=True),
            ]
            for _ in range(self.num_res_blocks):
                layers.append(_ResGNBlock(res_ch, norm_type=self.norm_type))
            self.backbone = nn.Sequential(*layers)
            c_in = res_ch
        else:
            layers = []
            c_in = self.in_channels
            for c_out in self.conv_channels:
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
                layers.append(_make_norm(self.norm_type, c_out))
                layers.append(nn.ReLU(inplace=True))
                c_in = c_out
            self.backbone = nn.Sequential(*layers)

        # Nhánh Value V(s) -> (B, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(c_in, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * self.board_size * self.board_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Nhánh Advantage A(s, a) -> (B, N*N)
        self.advantage_head = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward_value_and_q(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về đồng thời `(V(s), Q(s, .))` với Masked Dueling Mean trên các ô hợp lệ."""
        feat = self.backbone(x)
        value = self.value_head(feat)  # (B, 1)
        advantage = self.advantage_head(feat).flatten(start_dim=1)  # (B, N*N)

        if legal_mask is not None:
            mask = legal_mask.to(dtype=advantage.dtype, device=advantage.device)
        elif x.shape[1] >= 3:
            # Kênh 2 luôn biểu diễn ô trống (1.0 tại các ô hợp lệ, 0.0 tại ô đã có quân)
            mask = (x[:, 2].flatten(start_dim=1) > 0.5).to(dtype=advantage.dtype)
        else:
            mask = None

        if mask is not None:
            mask_sum = mask.sum(dim=1, keepdim=True)
            has_valid = mask_sum > 0.5
            masked_mean = (advantage * mask).sum(dim=1, keepdim=True) / mask_sum.clamp(min=1.0)
            adv_mean = torch.where(has_valid, masked_mean, advantage.mean(dim=1, keepdim=True))
        else:
            adv_mean = advantage.mean(dim=1, keepdim=True)

        q_values = value + (advantage - adv_mean)
        return value, q_values

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, q_values = self.forward_value_and_q(x, legal_mask=legal_mask)
        return q_values


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RLAgent(BaseAgent):
    """
    Tác tử Cấp 6: Deep Reinforcement Learning (Dueling Double-DQN nâng cấp 6A) cho Caro 11x11.
    Tự động nhận diện cả checkpoint mới (Residual + GroupNorm + 4 kênh) lẫn checkpoint cũ
    (BatchNorm + 3 kênh) khi nạp từ đĩa.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (64, 64, 128),
        board_size: int = 11,
        device: str = "auto",
        name: str = "Level6_RL_DQN",
        use_residual: bool = False,
        num_res_blocks: int = 3,
        norm_type: str = "group",
        candidate_radius: int | None = None,
        auto_load_default: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.device = resolve_device(device)
        self.in_channels = int(in_channels)
        self.conv_channels = tuple(int(c) for c in conv_channels)
        self.board_size = int(board_size)
        self.use_residual = bool(use_residual)
        self.num_res_blocks = int(num_res_blocks)
        self.norm_type = str(norm_type)
        self.candidate_radius = candidate_radius

        resolved_ckpt: Path | None = None
        if checkpoint_path is not None:
            resolved_ckpt = Path(checkpoint_path)
        elif auto_load_default and DEFAULT_RL_CHECKPOINT.exists():
            resolved_ckpt = DEFAULT_RL_CHECKPOINT

        self.checkpoint_path: str | None = str(resolved_ckpt) if resolved_ckpt is not None else None

        state_dict = None
        if resolved_ckpt is not None:
            if not resolved_ckpt.exists():
                raise FileNotFoundError(f"Không tìm thấy file checkpoint: {resolved_ckpt}")
            ckpt = torch.load(resolved_ckpt, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.in_channels = int(ckpt.get("in_channels", self.in_channels))
                self.conv_channels = tuple(int(c) for c in ckpt.get("conv_channels", self.conv_channels))
                self.board_size = int(ckpt.get("board_size", self.board_size))
                self.use_residual = bool(ckpt.get("use_residual", False))
                self.num_res_blocks = int(ckpt.get("num_res_blocks", self.num_res_blocks))
                state_dict = ckpt["model_state_dict"]
                # Tự nhận diện checkpoint legacy dùng BatchNorm2d nếu không ghi rõ norm_type
                if "norm_type" in ckpt:
                    self.norm_type = str(ckpt["norm_type"])
                elif any("running_mean" in k for k in state_dict.keys()):
                    self.norm_type = "batch"
            else:
                state_dict = ckpt
                if isinstance(state_dict, dict) and any("running_mean" in k for k in state_dict.keys()):
                    self.norm_type = "batch"

        self.model = DuelingCaroDQN(
            in_channels=self.in_channels,
            conv_channels=self.conv_channels,
            board_size=self.board_size,
            use_residual=self.use_residual,
            num_res_blocks=self.num_res_blocks,
            norm_type=self.norm_type,
        ).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()

        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_decision_stage: str = ""
        self.last_q_value: float = 0.0

    def _encode_env(self, env) -> np.ndarray:
        last_action = env.history[-1] if getattr(env, "history", None) else None
        return encode_board_state(
            env.board,
            env.current_player,
            last_action=last_action,
            in_channels=self.in_channels,
        )

    def _predict_q_values(self, env) -> np.ndarray:
        state = self._encode_env(env)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    def predict_value(self, env) -> float:
        """Trả về giá trị trạng thái V(s) ước lượng từ nhánh Value của Dueling DQN."""
        state = self._encode_env(env)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, _ = self.model.forward_value_and_q(state_tensor)
        return float(value.item())

    def get_masked_q_values(self, env) -> np.ndarray:
        legal_actions = env.legal_actions()
        if self.candidate_radius is not None and legal_actions:
            valid = get_candidate_actions(env, radius=self.candidate_radius)
            if valid:
                legal_actions = valid
        q_values = self._predict_q_values(env)
        masked_q = np.full_like(q_values, -np.inf, dtype=np.float32)
        if legal_actions:
            masked_q[legal_actions] = q_values[legal_actions]
        return masked_q

    def select_action(self, env) -> int:
        start_time = time.perf_counter()
        self.last_search_count = 0
        self.last_decision_stage = ""
        try:
            legal_actions = env.legal_actions()
            if not legal_actions:
                raise ValueError("Không còn nước đi hợp lệ nào trên bàn cờ.")

            masked_q = self.get_masked_q_values(env)
            best_action = int(np.argmax(masked_q))

            self.last_search_count = 1
            self.last_q_value = float(masked_q[best_action])
            self.last_decision_stage = "dqn_greedy_q"
            return best_action
        finally:
            self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0

    def get_top_actions(self, env, top_k: int = 5) -> list[dict[str, Any]]:
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
