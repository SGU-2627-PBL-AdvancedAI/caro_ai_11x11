from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent
from caro_env import CaroEnv
from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions

# Thử nạp TensorFlow/Keras (nếu môi trường đã cài đặt như trong sách AlphaGo Simplified Chương 10).
# Nếu chưa cài đặt (ví dụ môi trường chỉ có NumPy), hệ thống tự động chuyển sang backend NumPy CNN
# có cùng kiến trúc toán học và định dạng trọng số tương thích hoàn toàn.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.utils import to_categorical as keras_to_categorical

    HAS_TENSORFLOW = True
except Exception:
    tf = None
    layers = None
    models = None
    optimizers = None
    keras_to_categorical = None
    HAS_TENSORFLOW = False


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


# ==============================================================================
# 1. MÃ HÓA TRẠNG THÁI ĐỘC LẬP NGƯỜI CHƠI & TĂNG CƯỜNG DỮ LIỆU ĐỐI XỨNG
#    (Tham khảo Mục 10.1.2 - "A Player-Independent Setup for the Game Data")
# ==============================================================================


def encode_board(board: np.ndarray, current_player: int = 1) -> np.ndarray:
    """
    Mã hóa trạng thái bàn cờ theo góc nhìn chuẩn hóa (canonical perspective) của người chơi hiện tại.
    Theo Chương 10 sách AlphaGo Simplified (Mark Liu):
      - Quân của người chơi đang tới lượt luôn được biểu diễn là +1.0
      - Quân của đối thủ luôn được biểu diễn là -1.0
      - Ô trống được biểu diễn là 0.0
    Nhờ đó chỉ cần huấn luyện 1 mạng Policy Network duy nhất cho cả người chơi đi trước (X = 1)
    và người chơi đi sau (O = -1).

    Trả về mảng float32 có kích thước (board_size, board_size, 1).
    """
    if current_player not in (1, -1):
        raise ValueError(f"current_player phải là 1 hoặc -1, nhận được: {current_player}")
    canonical = np.asarray(board, dtype=np.float32) * float(current_player)
    n = canonical.shape[0]
    return canonical.reshape(n, n, 1)


def to_one_hot(actions: np.ndarray | list[int], num_classes: int = 121) -> np.ndarray:
    """
    Chuyển danh sách chỉ số hành động (0 .. num_classes-1) sang dạng vector one-hot
    (tương đương hàm `to_categorical(y, num_classes=121)` trong Mục 10.2.1 của sách).
    """
    actions_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    if keras_to_categorical is not None:
        return np.asarray(keras_to_categorical(actions_arr, num_classes=num_classes), dtype=np.float32)
    one_hot = np.zeros((actions_arr.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(actions_arr.shape[0]), actions_arr] = 1.0
    return one_hot


def augment_state_action(
    state_3d: np.ndarray,
    action: int,
    board_size: int = 11,
) -> list[tuple[np.ndarray, int]]:
    """
    Tăng cường dữ liệu 8 phép biến đổi đối xứng dihedral (nhóm D4: 4 góc quay x 2 phép lật)
    trên bàn cờ vuông 11x11. Giúp nhân 8 lần số mẫu huấn luyện hợp lệ từ mỗi nước đi của Expert.
    """
    board_2d = np.asarray(state_3d, dtype=np.float32).reshape(board_size, board_size)
    action_grid = np.zeros((board_size, board_size), dtype=np.int8)
    r, c = divmod(int(action), board_size)
    action_grid[r, c] = 1

    samples: list[tuple[np.ndarray, int]] = []
    for k in range(4):
        rot_board = np.rot90(board_2d, k=k)
        rot_action_grid = np.rot90(action_grid, k=k)
        rot_action = int(np.flatnonzero(rot_action_grid.reshape(-1))[0])
        samples.append((rot_board.reshape(board_size, board_size, 1).copy(), rot_action))

        flip_board = np.fliplr(rot_board)
        flip_action_grid = np.fliplr(rot_action_grid)
        flip_action = int(np.flatnonzero(flip_action_grid.reshape(-1))[0])
        samples.append((flip_board.reshape(board_size, board_size, 1).copy(), flip_action))

    return samples


# ==============================================================================
# 2. SINH DỮ LIỆU VÁN ĐẤU TỪ EXPERT VÀ NON-EXPERT PLAYER
#    (Tham khảo Mục 10.1.1 - "Simulate Games with Expert and Non-Expert Players")
# ==============================================================================


def expert_select_action(
    env: CaroEnv,
    candidate_radius: int = 2,
    defense_weight: float = 0.95,
    center_weight: float = 2.0,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Hàm chọn nước đi của người chơi Chuyên gia (Expert Player) cho Caro 11x11.
    Tương ứng với hàm `expert()` trong Chương 10 sách AlphaGo Simplified, được thiết kế
    tối ưu cho bàn cờ Caro 11x11 dựa trên hệ thống luật chiến thuật và lượng giá Heuristic:
      1. Thắng ngay lập tức nếu tạo được chuỗi 5 quân (`is_win_5`).
      2. Chặn ngay lập tức nếu đối thủ có nước đi tạo chuỗi 5 quân.
      3. Tạo chuỗi 4 mở hai đầu (`open_4 >= 1`).
      4. Chặn nước tạo chuỗi 4 mở hai đầu của đối thủ.
      5. Đánh giá tổng hợp công - thủ - vị trí trung tâm kết hợp kiểm tra phản đòn 1 bước.
    """
    legal = env.legal_actions()
    if not legal:
        raise ValueError("Không còn nước đi hợp lệ trên bàn cờ.")

    candidates = get_candidate_actions(env, radius=candidate_radius)
    if not candidates:
        candidates = legal

    player = env.current_player
    opponent = -player
    n = env.board_size
    center = n // 2

    my_threats: dict[int, dict[str, Any]] = {}
    opp_threats: dict[int, dict[str, Any]] = {}

    # Ưu tiên 1: Thắng ngay (5 quân liên tiếp)
    win_moves: list[int] = []
    for action in candidates:
        mt = evaluate_cell_threat(env, action, player)
        my_threats[action] = mt
        if mt["is_win_5"]:
            win_moves.append(action)
    if win_moves:
        return int(win_moves[0])

    # Ưu tiên 2: Chặn đối thủ thắng ngay
    block_win_moves: list[int] = []
    for action in candidates:
        ot = evaluate_cell_threat(env, action, opponent)
        opp_threats[action] = ot
        if ot["is_win_5"]:
            block_win_moves.append(action)
    if block_win_moves:
        # Nếu có nhiều ô chặn, chọn ô vừa chặn vừa có điểm tấn công cao nhất
        return int(max(block_win_moves, key=lambda a: my_threats[a]["score"] + opp_threats[a]["score"]))

    # Ưu tiên 3: Tạo chuỗi 4 mở 2 đầu chắc chắn thắng
    open_4_moves = [a for a in candidates if my_threats[a]["open_4"] >= 1]
    if open_4_moves:
        return int(max(open_4_moves, key=lambda a: my_threats[a]["score"]))

    # Ưu tiên 4: Chặn chuỗi 4 mở 2 đầu hoặc đe dọa kép nguy hiểm của đối thủ
    block_open_4_moves = [
        a
        for a in candidates
        if opp_threats[a]["open_4"] >= 1
        or (opp_threats[a]["half_open_4"] >= 1 and opp_threats[a]["open_3"] >= 1)
        or opp_threats[a]["open_3"] >= 2
    ]
    if block_open_4_moves:
        return int(
            max(
                block_open_4_moves,
                key=lambda a: opp_threats[a]["score"] * 1.1 + my_threats[a]["score"],
            )
        )

    # Ưu tiên 5: Xếp hạng theo tổng điểm Công + Thủ + Thưởng vị trí gần tâm bàn cờ
    best_score = -math.inf
    best_actions: list[int] = []
    for action in candidates:
        r, c = divmod(int(action), n)
        dist = abs(r - center) + abs(c - center)
        center_bonus = max(0.0, float(n - dist)) * center_weight
        total_score = (
            float(my_threats[action]["score"])
            + defense_weight * float(opp_threats[action]["score"])
            + center_bonus
        )
        if total_score > best_score + 1e-6:
            best_score = total_score
            best_actions = [int(action)]
        elif abs(total_score - best_score) <= 1e-6:
            best_actions.append(int(action))

    if len(best_actions) == 1 or rng is None:
        return best_actions[0]
    return int(rng.choice(best_actions))


def non_expert_select_action(
    env: CaroEnv,
    random_prob: float = 0.5,
    candidate_radius: int = 2,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Hàm chọn nước đi của người chơi Bán chuyên (Non-Expert Player) theo Mục 10.1.1 sách AlphaGo Simplified:
      - Với xác suất `random_prob` (mặc định 50%), chọn ngẫu nhiên một nước đi trong vùng ứng viên
        để tạo sự đa dạng cho không gian trạng thái bàn cờ.
      - Với xác suất `1 - random_prob` (50% còn lại), chọn nước đi tối ưu từ `expert_select_action`.
    """
    if rng is None:
        rng = np.random.default_rng()

    if float(rng.random()) < random_prob:
        candidates = get_candidate_actions(env, radius=candidate_radius)
        if not candidates:
            candidates = env.legal_actions()
        return int(rng.choice(candidates))

    return expert_select_action(env, candidate_radius=candidate_radius, rng=rng)


def simulate_expert_vs_non_expert_game(
    board_size: int = 11,
    win_length: int = 5,
    random_prob: float = 0.5,
    candidate_radius: int = 2,
    augment_symmetry: bool = True,
    expert_agent: BaseAgent | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[int], int]:
    """
    Mô phỏng 1 ván đấu giữa Expert và Non-Expert theo đúng quy trình Mục 10.1.2 của sách:
      - Phân công ngẫu nhiên 50/50 lượt đi trước (X = 1) hoặc đi sau (O = -1) cho Expert.
      - Tại mỗi lượt của Expert:
          + Mã hóa trạng thái bàn cờ theo góc nhìn hiện tại: `state = board * current_player`.
          + Ghi nhận cặp `(state, action)` của Expert vào tập dữ liệu huấn luyện.
      - Tại mỗi lượt của Non-Expert:
          + Thực hiện nước đi pha trộn (50% ngẫu nhiên, 50% expert) để mở rộng không gian trạng thái,
            nhưng KHÔNG ghi vào dữ liệu huấn luyện.
    """
    if rng is None:
        rng = np.random.default_rng()

    env = CaroEnv(board_size=board_size, win_length=win_length)
    env.reset()

    # Chọn ngẫu nhiên Expert đóng vai người chơi 1 (X) hoặc -1 (O)
    expert_player = 1 if int(rng.integers(0, 2)) == 1 else -1

    states: list[np.ndarray] = []
    actions: list[int] = []

    while not env.is_terminal():
        if env.current_player == expert_player:
            canonical_state = encode_board(env.board, env.current_player)
            if expert_agent is not None:
                action = int(expert_agent.select_action(env))
            else:
                action = expert_select_action(env, candidate_radius=candidate_radius, rng=rng)

            if augment_symmetry:
                for aug_state, aug_action in augment_state_action(canonical_state, action, board_size):
                    states.append(aug_state)
                    actions.append(aug_action)
            else:
                states.append(canonical_state)
                actions.append(action)
        else:
            action = non_expert_select_action(
                env,
                random_prob=random_prob,
                candidate_radius=candidate_radius,
                rng=rng,
            )

        env.step(action)

    return states, actions, env.winner


def generate_expert_dataset(
    num_games: int = 200,
    board_size: int = 11,
    win_length: int = 5,
    random_prob: float = 0.5,
    candidate_radius: int = 2,
    augment_symmetry: bool = True,
    expert_agent: BaseAgent | None = None,
    seed: int | None = 42,
    save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinh tập dữ liệu huấn luyện có giám sát `(X, y)` từ `num_games` ván đấu tự chơi
    giữa Expert và Non-Expert (Mục 10.1 sách AlphaGo Simplified).

    Trả về:
      - `X`: Mảng float32 kích thước `(N, board_size, board_size, 1)`
      - `y`: Mảng int64 kích thước `(N,)` chứa nhãn hành động `0 .. board_size^2 - 1`
    """
    if num_games <= 0:
        raise ValueError("num_games phải là số nguyên dương (> 0).")

    rng = np.random.default_rng(seed)
    all_states: list[np.ndarray] = []
    all_actions: list[int] = []

    for _ in range(num_games):
        game_states, game_actions, _ = simulate_expert_vs_non_expert_game(
            board_size=board_size,
            win_length=win_length,
            random_prob=random_prob,
            candidate_radius=candidate_radius,
            augment_symmetry=augment_symmetry,
            expert_agent=expert_agent,
            rng=rng,
        )
        all_states.extend(game_states)
        all_actions.extend(game_actions)

    X = np.asarray(all_states, dtype=np.float32)
    y = np.asarray(all_actions, dtype=np.int64)

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, X=X, y=y)

    return X, y


# ==============================================================================
# 3. KIẾN TRÚC MẠNG POLICY NETWORK (FAST & STRONG POLICY NETWORKS)
#    (Tham khảo Mục 10.2.1 - "Train Two Policy Networks" trong sách AlphaGo Simplified)
#
#    - Fast Policy Network (`model_type="fast"`):
#        Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(11,11,1))
#        -> Flatten()
#        -> Dense(64, activation="relu")
#        -> Dense(64, activation="relu")
#        -> Dense(121, activation="softmax")
#
#    - Strong Policy Network (`model_type="strong"`):
#        Conv2D(128, (3,3), padding="same", activation="relu", input_shape=(11,11,1))
#        -> Flatten()
#        -> Dense(64, activation="relu")
#        -> Dense(64, activation="relu")
#        -> Dense(64, activation="relu")
#        -> Dense(121, activation="softmax")
# ==============================================================================


class NumpyCNNPolicyNetwork:
    """
    Triển khai mạng CNN Policy Network bằng NumPy thuần túy (Vectorized Conv2D + ReLU + Flatten
    + Dense layers + Softmax + Categorical Cross-Entropy + Adam Optimizer).
    Đảm bảo kiến trúc và chiều trọng số giống hệt 100% với mô hình `tensorflow.keras.Sequential`
    trong Chương 10 sách AlphaGo Simplified:
      - `conv_w`: (kernel_size, kernel_size, 1, conv_filters)
      - `conv_b`: (conv_filters,)
      - `dense_w_i`: (in_features, out_features)
      - `dense_b_i`: (out_features,)
    Có thể đọc/ghi trực tiếp file trọng số `.npz` dùng chung với mô hình Keras.
    """

    def __init__(
        self,
        board_size: int = 11,
        model_type: str = "strong",
        conv_filters: int = 128,
        kernel_size: int = 3,
        dense_units: int = 64,
        learning_rate: float = 1e-3,
        seed: int | None = 42,
    ):
        if model_type not in ("fast", "strong"):
            raise ValueError("model_type phải là 'fast' (2 lớp ẩn Dense) hoặc 'strong' (3 lớp ẩn Dense).")
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size phải là số lẻ dương (ví dụ: 3 hoặc 5).")

        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size
        self.model_type = model_type
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)
        self.dense_units = int(dense_units)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)

        # Khởi tạo trọng số He (Kaiming) cho các lớp ReLU và Glorot (Xavier) cho lớp Softmax
        fan_in_conv = self.kernel_size * self.kernel_size * 1
        self.conv_w = self.rng.normal(
            0.0,
            math.sqrt(2.0 / fan_in_conv),
            size=(self.kernel_size, self.kernel_size, 1, self.conv_filters),
        ).astype(np.float32)
        self.conv_b = np.zeros((self.conv_filters,), dtype=np.float32)

        flat_dim = self.board_size * self.board_size * self.conv_filters
        num_hidden_dense = 2 if self.model_type == "fast" else 3

        layer_dims = [flat_dim] + [self.dense_units] * num_hidden_dense + [self.num_actions]
        self.dense_weights: list[np.ndarray] = []
        self.dense_biases: list[np.ndarray] = []

        for i in range(len(layer_dims) - 1):
            in_d, out_d = layer_dims[i], layer_dims[i + 1]
            is_last = i == len(layer_dims) - 2
            scale = math.sqrt(2.0 / (in_d + out_d)) if is_last else math.sqrt(2.0 / in_d)
            w = self.rng.normal(0.0, scale, size=(in_d, out_d)).astype(np.float32)
            b = np.zeros((out_d,), dtype=np.float32)
            self.dense_weights.append(w)
            self.dense_biases.append(b)

        self._init_adam_state()

    def _init_adam_state(self) -> None:
        """Khởi tạo bộ nhớ moment bậc 1 (m) và bậc 2 (v) cho thuật toán tối ưu Adam."""
        self.adam_t = 0
        self.m_conv_w = np.zeros_like(self.conv_w)
        self.v_conv_w = np.zeros_like(self.conv_w)
        self.m_conv_b = np.zeros_like(self.conv_b)
        self.v_conv_b = np.zeros_like(self.conv_b)

        self.m_dense_w = [np.zeros_like(w) for w in self.dense_weights]
        self.v_dense_w = [np.zeros_like(w) for w in self.dense_weights]
        self.m_dense_b = [np.zeros_like(b) for b in self.dense_biases]
        self.v_dense_b = [np.zeros_like(b) for b in self.dense_biases]

    def _extract_patches(self, x: np.ndarray) -> np.ndarray:
        """
        Biến đổi im2col (trích xuất các cửa sổ trượt 2D với padding='same') cho đầu vào
        `x` có kích thước `(batch_size, H, W, 1)`.
        Trả về tensor `(batch_size, H, W, kernel_size * kernel_size)`.
        """
        pad = self.kernel_size // 2
        x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
        bsz, h, w, _ = x.shape
        k = self.kernel_size
        patches = np.empty((bsz, h, w, k * k), dtype=np.float32)
        idx = 0
        for kr in range(k):
            for kc in range(k):
                patches[:, :, :, idx] = x_pad[:, kr : kr + h, kc : kc + w, 0]
                idx += 1
        return patches

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray, return_cache: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """
        Lan truyền tiến (Forward Pass) qua mạng CNN Policy Network:
          Input (B, 11, 11, 1) -> Conv2D(128, 3x3, 'same', ReLU) -> Flatten
          -> Dense(64, ReLU) x (2 hoặc 3) -> Dense(121, Softmax).
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 3:
            x_arr = x_arr.reshape(1, self.board_size, self.board_size, 1)

        bsz = x_arr.shape[0]
        patches = self._extract_patches(x_arr)  # (B, H, W, K*K)
        w_col = self.conv_w.reshape(-1, self.conv_filters)  # (K*K, F)
        conv_pre = patches @ w_col + self.conv_b  # (B, H, W, F)
        conv_act = np.maximum(conv_pre, 0.0)
        flat = conv_act.reshape(bsz, -1)

        hidden_inputs: list[np.ndarray] = []
        hidden_pres: list[np.ndarray] = []
        curr = flat

        for i in range(len(self.dense_weights) - 1):
            hidden_inputs.append(curr)
            pre = curr @ self.dense_weights[i] + self.dense_biases[i]
            hidden_pres.append(pre)
            curr = np.maximum(pre, 0.0)

        logits_in = curr
        logits = logits_in @ self.dense_weights[-1] + self.dense_biases[-1]
        probs = self._softmax(logits)

        if not return_cache:
            return probs

        cache = {
            "patches": patches,
            "conv_pre": conv_pre,
            "hidden_inputs": hidden_inputs,
            "hidden_pres": hidden_pres,
            "logits_in": logits_in,
            "probs": probs,
        }
        return probs, cache

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Dự đoán phân phối xác suất 121 nước đi cho batch trạng thái `x`."""
        return self.forward(x, return_cache=False)

    def _adam_step(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-7,
    ) -> None:
        m *= beta1
        m += (1.0 - beta1) * grad
        v *= beta2
        v += (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**self.adam_t)
        v_hat = v / (1.0 - beta2**self.adam_t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_on_batch(self, x_batch: np.ndarray, y_onehot_batch: np.ndarray, lr: float | None = None) -> tuple[float, float]:
        """
        Thực hiện 1 bước lan truyền tiến + lan truyền ngược (Backpropagation) và cập nhật trọng số Adam
        với hàm mất mát Categorical Cross-Entropy:
            L = - (1 / B) * sum(y * log(p))
        Trả về `(loss, accuracy)` trên batch.
        """
        use_lr = self.learning_rate if lr is None else float(lr)
        bsz = x_batch.shape[0]
        probs, cache = self.forward(x_batch, return_cache=True)

        # Categorical Cross-Entropy Loss & Accuracy
        clipped_probs = np.clip(probs, 1e-7, 1.0)
        loss = float(-np.sum(y_onehot_batch * np.log(clipped_probs)) / bsz)
        preds = np.argmax(probs, axis=-1)
        targets = np.argmax(y_onehot_batch, axis=-1)
        acc = float(np.mean(preds == targets))

        # Backprop qua lớp đầu ra Softmax + Cross-Entropy: dL/dlogits = (probs - y_onehot) / B
        d_logits = (probs - y_onehot_batch) / float(bsz)

        grad_dense_w: list[np.ndarray] = [None] * len(self.dense_weights)  # type: ignore[list-item]
        grad_dense_b: list[np.ndarray] = [None] * len(self.dense_biases)  # type: ignore[list-item]

        # Lớp Dense cuối (Softmax output)
        logits_in = cache["logits_in"]
        grad_dense_w[-1] = logits_in.T @ d_logits
        grad_dense_b[-1] = np.sum(d_logits, axis=0)
        d_curr = d_logits @ self.dense_weights[-1].T

        # Các lớp Dense ẩn (ReLU)
        for i in range(len(self.dense_weights) - 2, -1, -1):
            pre = cache["hidden_pres"][i]
            inp = cache["hidden_inputs"][i]
            d_pre = d_curr * (pre > 0.0)
            grad_dense_w[i] = inp.T @ d_pre
            grad_dense_b[i] = np.sum(d_pre, axis=0)
            d_curr = d_pre @ self.dense_weights[i].T

        # Backprop qua Flatten và Conv2D (ReLU)
        conv_pre = cache["conv_pre"]
        d_conv_act = d_curr.reshape(conv_pre.shape)
        d_conv_pre = d_conv_act * (conv_pre > 0.0)  # (B, H, W, F)

        patches_flat = cache["patches"].reshape(-1, self.kernel_size * self.kernel_size)
        d_conv_pre_flat = d_conv_pre.reshape(-1, self.conv_filters)

        grad_conv_w = (patches_flat.T @ d_conv_pre_flat).reshape(self.conv_w.shape)
        grad_conv_b = np.sum(d_conv_pre_flat, axis=0)

        # Cập nhật trọng số bằng Adam
        self.adam_t += 1
        self._adam_step(self.conv_w, grad_conv_w, self.m_conv_w, self.v_conv_w, use_lr)
        self._adam_step(self.conv_b, grad_conv_b, self.m_conv_b, self.v_conv_b, use_lr)
        for i in range(len(self.dense_weights)):
            self._adam_step(
                self.dense_weights[i],
                grad_dense_w[i],
                self.m_dense_w[i],
                self.v_dense_w[i],
                use_lr,
            )
            self._adam_step(
                self.dense_biases[i],
                grad_dense_b[i],
                self.m_dense_b[i],
                self.v_dense_b[i],
                use_lr,
            )

        return loss, acc

    def fit(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        learning_rate: float | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Huấn luyện mạng CNN Policy Network theo mini-batch (tương đương `model.fit(...)` trong Keras).
        Trả về từ điển lịch sử huấn luyện `{"loss": [...], "accuracy": [...], "val_loss": [...], "val_accuracy": [...]}`.
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Tập dữ liệu huấn luyện X không được rỗng.")

        if 0.0 < validation_split < 1.0 and n_samples >= 10:
            val_count = max(1, int(round(n_samples * validation_split)))
            train_count = n_samples - val_count
            X_train, X_val = X[:train_count], X[train_count:]
            y_train, y_val = y_onehot[:train_count], y_onehot[train_count:]
        else:
            X_train, y_train = X, y_onehot
            X_val, y_val = None, None

        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
        }
        if X_val is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        n_train = X_train.shape[0]
        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n_train)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            batch_losses: list[float] = []
            batch_accs: list[float] = []

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                b_loss, b_acc = self.train_on_batch(
                    X_shuf[start:end],
                    y_shuf[start:end],
                    lr=learning_rate,
                )
                batch_losses.append(b_loss)
                batch_accs.append(b_acc)

            epoch_loss = float(np.mean(batch_losses))
            epoch_acc = float(np.mean(batch_accs))
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            if X_val is not None and y_val is not None:
                val_probs = self.predict(X_val)
                val_clipped = np.clip(val_probs, 1e-7, 1.0)
                v_loss = float(-np.sum(y_val * np.log(val_clipped)) / X_val.shape[0])
                v_acc = float(np.mean(np.argmax(val_probs, axis=-1) == np.argmax(y_val, axis=-1)))
                history["val_loss"].append(v_loss)
                history["val_accuracy"].append(v_acc)
                if verbose:
                    print(
                        f"Epoch {epoch:02d}/{epochs:02d} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f} "
                        f"- val_loss: {v_loss:.4f} - val_accuracy: {v_acc:.4f}"
                    )
            elif verbose:
                print(f"Epoch {epoch:02d}/{epochs:02d} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}")

        return history

    def get_weights_dict(self) -> dict[str, np.ndarray]:
        """Xuất toàn bộ trọng số của mạng thành dictionary để lưu trữ `.npz`."""
        data: dict[str, np.ndarray] = {
            "board_size": np.array(self.board_size, dtype=np.int64),
            "model_type": np.array(self.model_type),
            "conv_filters": np.array(self.conv_filters, dtype=np.int64),
            "kernel_size": np.array(self.kernel_size, dtype=np.int64),
            "dense_units": np.array(self.dense_units, dtype=np.int64),
            "num_dense": np.array(len(self.dense_weights), dtype=np.int64),
            "conv_w": self.conv_w,
            "conv_b": self.conv_b,
        }
        for i, (w, b) in enumerate(zip(self.dense_weights, self.dense_biases)):
            data[f"dense_w_{i}"] = w
            data[f"dense_b_{i}"] = b
        return data

    def set_weights_from_dict(self, data: dict[str, Any]) -> None:
        """Nạp trọng số từ dictionary `.npz` vào mạng."""
        self.conv_w = np.asarray(data["conv_w"], dtype=np.float32)
        self.conv_b = np.asarray(data["conv_b"], dtype=np.float32)
        num_dense = int(data["num_dense"])
        self.dense_weights = [np.asarray(data[f"dense_w_{i}"], dtype=np.float32) for i in range(num_dense)]
        self.dense_biases = [np.asarray(data[f"dense_b_{i}"], dtype=np.float32) for i in range(num_dense)]
        self._init_adam_state()


def build_keras_policy_network(
    board_size: int = 11,
    model_type: str = "strong",
    conv_filters: int = 128,
    kernel_size: int = 3,
    dense_units: int = 64,
    learning_rate: float = 1e-3,
) -> Any:
    """
    Khởi tạo mô hình CNN Policy Network bằng `tensorflow.keras.Sequential` theo đúng Mục 10.2.1
    sách AlphaGo Simplified (Mark Liu):
      - `fast`: 1 lớp Conv2D(128, 3x3, padding='same', relu) -> Flatten -> 2 lớp Dense(64, relu) -> Dense(121, softmax)
      - `strong`: 1 lớp Conv2D(128, 3x3, padding='same', relu) -> Flatten -> 3 lớp Dense(64, relu) -> Dense(121, softmax)
    """
    if not HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow/Keras chưa được cài đặt trong môi trường hiện tại.")
    if model_type not in ("fast", "strong"):
        raise ValueError("model_type phải là 'fast' hoặc 'strong'.")

    num_actions = board_size * board_size
    model = models.Sequential(name=f"caro_{board_size}x{board_size}_{model_type}_policy_net")
    model.add(layers.Input(shape=(board_size, board_size, 1)))
    model.add(
        layers.Conv2D(
            filters=conv_filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            activation="relu",
            name="conv2d_feature_extractor",
        )
    )
    model.add(layers.Flatten(name="flatten_board"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense_hidden_1"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense_hidden_2"))
    if model_type == "strong":
        model.add(layers.Dense(dense_units, activation="relu", name="dense_hidden_3"))
    model.add(layers.Dense(num_actions, activation="softmax", name="policy_head_softmax"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


# ==============================================================================
# 4. TÁC TỬ CẤP 5: CNN POLICY NETWORK AGENT & POLICY-GUIDED MIXED MCTS
#    (Tham khảo Mục 10.2, 10.3 & 10.4 sách AlphaGo Simplified)
# ==============================================================================


class CNNAgent(BaseAgent):
    """
    Tác tử Cấp 5: Mạng Nơ-ron Tích chập (Convolutional Policy Network) cho Caro 11x11.
    Xây dựng dựa trên kiến trúc Deep Learning Chương 10 ("Policy Networks in Tic Tac Toe")
    trong sách *AlphaGo Simplified: Rule-Based AI and Deep Learning in Everyday Games* (Mark Liu):

    1. **Biểu diễn trạng thái độc lập người chơi (Player-Independent Encoding)**:
       Nhân ma trận bàn cờ `env.board` với `env.current_player` để quân của mình luôn là `+1`,
       quân đối phương là `-1`, ô trống là `0`, đưa về tensor 4 chiều `(1, 11, 11, 1)`.
    2. **Kiến trúc Fast & Strong Policy Network**:
       - `Conv2D(128, kernel_size=(3,3), padding="same", activation="relu")`
       - `Flatten()`
       - 2 lớp `Dense(64, activation="relu")` (`model_type="fast"`) hoặc
         3 lớp `Dense(64, activation="relu")` (`model_type="strong"`)
       - `Dense(121, activation="softmax")` dự đoán phân phối xác suất trên 121 ô cờ.
    3. **Hai chế độ ra quyết định**:
       - **Pure Policy Network (`use_mix_mcts=False`)**: Dự đoán xác suất từ mạng CNN, lọc các
         nước đi hợp lệ, chuẩn hóa tổng xác suất về 1.0 và chọn `argmax` (hoặc lấy mẫu xác suất).
       - **Policy-Guided Mixed MCTS (`use_mix_mcts=True`)**: Kết hợp tìm kiếm cây Monte Carlo (UCT)
         với xác suất tiên nghiệm $p_i$ từ mạng Policy Network theo công thức Mục 10.4.2 của sách:
         $$\\text{mix\\_score}_i = v_i + C \\sqrt{\\frac{\\ln N}{n_i}} + \\gamma \\frac{p_i}{1 + n_i}$$
    """

    def __init__(
        self,
        name: str = "Level5_CNN_Policy",
        model_type: str = "strong",
        backend: str = "auto",
        model_path: str | Path | None = None,
        board_size: int = 11,
        conv_filters: int = 128,
        kernel_size: int = 3,
        dense_units: int = 64,
        learning_rate: float = 1e-3,
        deterministic: bool = True,
        temperature: float = 1.0,
        candidate_radius: int | None = 2,
        use_tactical_shortcut: bool = True,
        use_mix_mcts: bool = False,
        num_rollouts: int = 100,
        exploration_c: float = 1.414,
        gamma: float = 10.0,
        max_rollout_steps: int = 24,
        time_limit_sec: float | None = None,
        seed: int | None = 42,
    ):
        super().__init__(name=name)

        if model_type not in ("fast", "strong"):
            raise ValueError("model_type phải là 'fast' hoặc 'strong'.")
        if backend not in ("auto", "keras", "numpy"):
            raise ValueError("backend phải là 'auto', 'keras' hoặc 'numpy'.")
        if board_size < 3:
            raise ValueError("board_size phải >= 3.")
        if temperature <= 0.0:
            raise ValueError("temperature phải > 0.")
        if candidate_radius is not None and candidate_radius < 0:
            raise ValueError("candidate_radius phải >= 0 hoặc None.")
        if num_rollouts <= 0:
            raise ValueError("num_rollouts phải > 0.")

        self.model_type = model_type
        if backend == "auto":
            self.backend = "keras" if HAS_TENSORFLOW else "numpy"
        elif backend == "keras" and not HAS_TENSORFLOW:
            raise RuntimeError("Yêu cầu backend='keras' nhưng TensorFlow chưa được cài đặt.")
        else:
            self.backend = backend

        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)
        self.dense_units = int(dense_units)
        self.learning_rate = float(learning_rate)
        self.deterministic = bool(deterministic)
        self.temperature = float(temperature)
        self.candidate_radius = candidate_radius
        self.use_tactical_shortcut = bool(use_tactical_shortcut)

        # Cấu hình chế độ Mixed MCTS (Mục 10.4 sách AlphaGo Simplified)
        self.use_mix_mcts = bool(use_mix_mcts)
        self.num_rollouts = int(num_rollouts)
        self.exploration_c = float(exploration_c)
        self.gamma = float(gamma)
        self.max_rollout_steps = int(max_rollout_steps)
        self.time_limit_sec = float(time_limit_sec) if time_limit_sec is not None else None

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Khởi tạo mạng nơ-ron tích chập (duy trì đồng thời numpy_net để suy luận 1 mẫu cực nhanh < 1ms)
        self.numpy_net = NumpyCNNPolicyNetwork(
            board_size=self.board_size,
            model_type=self.model_type,
            conv_filters=self.conv_filters,
            kernel_size=self.kernel_size,
            dense_units=self.dense_units,
            learning_rate=self.learning_rate,
            seed=seed,
        )
        self.keras_model: Any = None
        if self.backend == "keras":
            self.keras_model = build_keras_policy_network(
                board_size=self.board_size,
                model_type=self.model_type,
                conv_filters=self.conv_filters,
                kernel_size=self.kernel_size,
                dense_units=self.dense_units,
                learning_rate=self.learning_rate,
            )
            self._sync_numpy_to_keras()

        # Chỉ số đo đạc phục vụ Giải đấu (Tournament) và Ablation Study
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_decision_stage: str = ""
        self.last_action_probs: np.ndarray | None = None
        self.training_history: dict[str, list[float]] = {}

        # Tự động nạp trọng số nếu truyền model_path hoặc nếu có sẵn checkpoint mặc định
        if model_path is not None:
            self.load_model(model_path)
        else:
            default_ckpt = DEFAULT_CHECKPOINT_DIR / f"cnn_{self.model_type}_{self.board_size}x{self.board_size}.npz"
            if default_ckpt.exists():
                try:
                    self.load_model(default_ckpt)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Đồng bộ trọng số giữa Keras Model và NumpyCNNPolicyNetwork
    # ------------------------------------------------------------------

    def _sync_numpy_to_keras(self) -> None:
        """Đồng bộ trọng số từ `self.numpy_net` sang `self.keras_model`."""
        if self.keras_model is None:
            return
        weights_list: list[np.ndarray] = [self.numpy_net.conv_w, self.numpy_net.conv_b]
        for w, b in zip(self.numpy_net.dense_weights, self.numpy_net.dense_biases):
            weights_list.append(w)
            weights_list.append(b)
        self.keras_model.set_weights(weights_list)

    def _sync_keras_to_numpy(self) -> None:
        """Đồng bộ trọng số từ `self.keras_model` sang `self.numpy_net`."""
        if self.keras_model is None:
            return
        weights_list = self.keras_model.get_weights()
        self.numpy_net.conv_w = np.asarray(weights_list[0], dtype=np.float32)
        self.numpy_net.conv_b = np.asarray(weights_list[1], dtype=np.float32)
        num_dense = (len(weights_list) - 2) // 2
        self.numpy_net.dense_weights = [
            np.asarray(weights_list[2 + 2 * i], dtype=np.float32) for i in range(num_dense)
        ]
        self.numpy_net.dense_biases = [
            np.asarray(weights_list[2 + 2 * i + 1], dtype=np.float32) for i in range(num_dense)
        ]

    # ------------------------------------------------------------------
    # Dự đoán phân phối xác suất nước đi (Mục 10.2.2 sách AlphaGo Simplified)
    # ------------------------------------------------------------------

    def encode_env_state(self, env: CaroEnv) -> np.ndarray:
        """
        Biểu diễn trạng thái bàn cờ `env` thành tensor 4 chiều `(1, board_size, board_size, 1)`
        theo góc nhìn của `env.current_player`.
        """
        if env.board_size != self.board_size:
            raise ValueError(
                f"Kích thước bàn cờ của env ({env.board_size}) không khớp với CNNAgent ({self.board_size})."
            )
        return encode_board(env.board, env.current_player).reshape(1, self.board_size, self.board_size, 1)

    def predict_probs(
        self,
        env: CaroEnv,
        restrict_to_candidates: bool = False,
    ) -> np.ndarray:
        """
        Dự đoán vector phân phối xác suất có độ dài `121` cho các nước đi trên bàn cờ hiện tại.
        Tuân thủ quy trình tại Mục 10.2.2 ("Test the Policy Network") của sách:
          1. Chuẩn hóa dấu bàn cờ theo người chơi hiện tại và đưa qua mạng CNN.
          2. Loại bỏ các ô đã có quân (chỉ giữ lại các ô hợp lệ `legal_actions` hoặc `candidates`).
          3. Chuẩn hóa lại xác suất trên các ô hợp lệ sao cho tổng bằng `1.0`.
        """
        legal = env.legal_actions()
        if not legal:
            return np.zeros((self.num_actions,), dtype=np.float32)

        valid_actions = (
            get_candidate_actions(env, radius=self.candidate_radius)
            if restrict_to_candidates and self.candidate_radius is not None
            else legal
        )
        if not valid_actions:
            valid_actions = legal

        state_4d = self.encode_env_state(env)
        raw_probs = self.numpy_net.forward(state_4d, return_cache=False).reshape(-1)

        # Áp dụng nhiệt độ (temperature scaling) nếu temperature != 1.0
        if abs(self.temperature - 1.0) > 1e-6:
            clipped = np.clip(raw_probs[valid_actions], 1e-12, 1.0)
            scaled = np.power(clipped, 1.0 / self.temperature)
        else:
            scaled = raw_probs[valid_actions].astype(np.float64)

        masked_probs = np.zeros((self.num_actions,), dtype=np.float32)
        total = float(np.sum(scaled))
        if total > 1e-12 and np.isfinite(total):
            masked_probs[valid_actions] = (scaled / total).astype(np.float32)
        else:
            # Fallback phân phối đều trên tập hợp lệ nếu tổng xác suất quá nhỏ
            masked_probs[valid_actions] = np.float32(1.0 / len(valid_actions))

        self.last_action_probs = masked_probs
        return masked_probs

    # ------------------------------------------------------------------
    # Kiểm tra chiến thuật khẩn cấp (Tactical Shortcut)
    # ------------------------------------------------------------------

    def _check_tactical_move(self, env: CaroEnv, candidates: list[int]) -> tuple[int | None, str]:
        """
        Kiểm tra các tình huống chiến thuật bắt buộc (Thắng ngay 5 quân, Chặn thắng ngay 5 quân,
        Tạo chuỗi 4 mở 2 đầu, Chặn chuỗi 4 mở 2 đầu) trước khi gọi mạng CNN.
        """
        player = env.current_player
        opponent = -player

        my_threats: dict[int, dict[str, Any]] = {}
        opp_threats: dict[int, dict[str, Any]] = {}

        # 1. Thắng ngay lập tức (5 quân liên tiếp)
        for action in candidates:
            mt = evaluate_cell_threat(env, action, player)
            my_threats[action] = mt
            if mt["is_win_5"]:
                return int(action), "win_5"

        # 2. Chặn nước thắng ngay của đối thủ
        block_wins: list[int] = []
        for action in candidates:
            ot = evaluate_cell_threat(env, action, opponent)
            opp_threats[action] = ot
            if ot["is_win_5"]:
                block_wins.append(action)
        if block_wins:
            best_block = max(block_wins, key=lambda a: my_threats[a]["score"] + opp_threats[a]["score"])
            return int(best_block), "block_win_5"

        # 3. Tạo chuỗi 4 mở hai đầu
        open_4_moves = [a for a in candidates if my_threats[a]["open_4"] >= 1]
        if open_4_moves:
            best_o4 = max(open_4_moves, key=lambda a: my_threats[a]["score"])
            return int(best_o4), "create_open_4"

        # 4. Chặn chuỗi 4 mở hai đầu của đối thủ
        block_open_4 = [a for a in candidates if opp_threats[a]["open_4"] >= 1]
        if block_open_4:
            best_bo4 = max(block_open_4, key=lambda a: opp_threats[a]["score"] + my_threats[a]["score"])
            return int(best_bo4), "block_open_4"

        return None, ""

    # ------------------------------------------------------------------
    # Chế độ Policy-Guided Mixed MCTS (Mục 10.4 sách AlphaGo Simplified)
    # ------------------------------------------------------------------

    def _rollout_policy_game(self, sim_env: CaroEnv, root_player: int) -> float:
        """
        Mô phỏng 1 ván đấu ngẫu nhiên có định hướng từ trạng thái sau khi chọn nhánh con tại gốc
        (tương ứng hàm `one_game()` trong Mục 10.4.2 sách AlphaGo Simplified).
        Trả về điểm kết quả theo góc nhìn của `root_player` trong đoạn `[-1.0, 1.0]`.
        """
        steps = 0
        while not sim_env.is_terminal() and steps < self.max_rollout_steps:
            cands = get_candidate_actions(sim_env, radius=1)
            if not cands:
                cands = sim_env.legal_actions()

            # Kiểm tra thắng ngay trong rollout để kết thúc sớm chính xác
            cur_p = sim_env.current_player
            win_now = None
            block_now = None
            for a in cands:
                if evaluate_cell_threat(sim_env, a, cur_p)["is_win_5"]:
                    win_now = a
                    break
                if block_now is None and evaluate_cell_threat(sim_env, a, -cur_p)["is_win_5"]:
                    block_now = a

            if win_now is not None:
                move = win_now
            elif block_now is not None:
                move = block_now
            else:
                move = int(self.rng.choice(cands))

            sim_env.step(move)
            steps += 1

        if sim_env.is_terminal():
            return float(sim_env.result(root_player))

        # Nếu chạm giới hạn bước rollout mà chưa kết thúc, dùng hàm lượng giá bàn cờ chuẩn hóa tanh
        heuristic_val = evaluate_board(sim_env, perspective=root_player)
        return float(math.tanh(heuristic_val / 5000.0))

    def _select_action_mix_mcts(
        self,
        env: CaroEnv,
        candidates: list[int],
        prior_probs: np.ndarray,
        start_time: float,
    ) -> int:
        """
        Thuật toán Mixed MCTS kết hợp Policy Network với UCT theo đúng Mục 10.4.2
        ("Implement the Policy Network with MCTS") trong sách AlphaGo Simplified:

        Công thức chọn nhánh trong mỗi lượt mô phỏng:
            mix_score_i = v_i + C * sqrt(ln(N) / n_i) + gamma * p_i / (1 + n_i)
        Trong đó:
            - v_i = (wins_i - losses_i) / n_i là giá trị trung bình của nước đi i
            - p_i là xác suất tiên nghiệm dự đoán bởi mạng Strong Policy Network cho nước đi i
            - gamma là hệ số trọng số của Policy Network (mặc định gamma = 10.0 theo sách)
        Sau khi hoàn tất các lượt rollout, chọn nước đi tối đa hóa:
            final_score_i = v_i + gamma * p_i / (1 + n_i)
        """
        num_cands = len(candidates)
        if num_cands == 1:
            self.last_search_count = 1
            return int(candidates[0])

        # Giới hạn tập ứng viên hàng đầu theo xác suất CNN để tối ưu tốc độ MCTS trên bàn 11x11
        max_branch = min(num_cands, 12)
        ranked_indices = np.argsort(prior_probs[candidates])[::-1][:max_branch]
        top_candidates = [int(candidates[idx]) for idx in ranked_indices]
        priors = np.array([float(prior_probs[a]) for a in top_candidates], dtype=np.float64)
        prior_sum = float(np.sum(priors))
        if prior_sum > 1e-12:
            priors /= prior_sum
        else:
            priors = np.full(len(top_candidates), 1.0 / len(top_candidates), dtype=np.float64)

        k = len(top_candidates)
        visits = np.zeros(k, dtype=np.float64)
        value_sums = np.zeros(k, dtype=np.float64)
        root_player = env.current_player

        rollouts_done = 0
        for rollout_idx in range(self.num_rollouts):
            if (
                self.time_limit_sec is not None
                and rollout_idx > 0
                and (time.perf_counter() - start_time) >= self.time_limit_sec
            ):
                break

            if rollout_idx < k:
                # Khám phá mỗi ứng viên hàng đầu ít nhất 1 lần
                choice_idx = rollout_idx
            else:
                total_n = float(np.sum(visits))
                v_i = value_sums / np.maximum(1.0, visits)
                exploration = self.exploration_c * np.sqrt(math.log(max(1.0, total_n)) / np.maximum(1.0, visits))
                policy_bonus = self.gamma * priors / (1.0 + visits)
                mix_scores = v_i + exploration + policy_bonus
                choice_idx = int(np.argmax(mix_scores))

            chosen_action = top_candidates[choice_idx]
            sim_env = env.clone()
            sim_env.step(chosen_action)
            reward = self._rollout_policy_game(sim_env, root_player=root_player)

            visits[choice_idx] += 1.0
            value_sums[choice_idx] += reward
            rollouts_done += 1

        self.last_search_count = rollouts_done
        v_final = value_sums / np.maximum(1.0, visits)
        final_scores = v_final + self.gamma * priors / (1.0 + visits)
        best_idx = int(np.argmax(final_scores))
        return int(top_candidates[best_idx])

    # ------------------------------------------------------------------
    # Giao diện chính: select_action(env) -> int
    # ------------------------------------------------------------------

    def select_action(self, env: CaroEnv) -> int:
        """
        Nhận môi trường `CaroEnv` hiện tại và trả về nước đi hợp lệ `(0..120)`.
        Đảm bảo không làm thay đổi trạng thái của `env` gốc.
        """
        start_time = time.perf_counter()
        self.last_search_count = 0
        self.last_decision_stage = ""

        try:
            legal = env.legal_actions()
            if not legal:
                raise ValueError("Không còn nước đi hợp lệ trên bàn cờ.")

            # Khai cuộc nước đầu tiên trên bàn cờ trống: đánh vào tâm (5, 5) -> action 60
            if not env.history:
                center = env.board_size // 2
                self.last_search_count = 1
                self.last_decision_stage = "opening_center"
                return int(center * env.board_size + center)

            candidates = (
                get_candidate_actions(env, radius=self.candidate_radius)
                if self.candidate_radius is not None
                else legal
            )
            if not candidates:
                candidates = legal

            # Bước 1: Kiểm tra chiến thuật bắt buộc nếu bật use_tactical_shortcut
            if self.use_tactical_shortcut:
                tactical_move, stage = self._check_tactical_move(env, candidates)
                if tactical_move is not None:
                    self.last_search_count = len(candidates)
                    self.last_decision_stage = f"tactical_{stage}"
                    return int(tactical_move)

            # Bước 2: Tính phân phối xác suất từ mạng CNN Policy Network
            probs = self.predict_probs(env, restrict_to_candidates=(self.candidate_radius is not None))
            self.last_search_count = 1

            # Bước 3: Nếu bật chế độ Policy-Guided Mixed MCTS (Mục 10.4 sách AlphaGo Simplified)
            if self.use_mix_mcts:
                self.last_decision_stage = "policy_mix_mcts"
                return self._select_action_mix_mcts(
                    env=env,
                    candidates=candidates,
                    prior_probs=probs,
                    start_time=start_time,
                )

            # Bước 4: Chế độ Pure Policy Network (Mục 10.2.2 & 10.3 sách AlphaGo Simplified)
            self.last_decision_stage = "cnn_policy_argmax" if self.deterministic else "cnn_policy_sample"
            valid_actions = candidates if self.candidate_radius is not None else legal
            valid_probs = probs[valid_actions].astype(np.float64)
            prob_sum = float(np.sum(valid_probs))
            if prob_sum > 1e-12:
                valid_probs /= prob_sum
            else:
                valid_probs = np.full(len(valid_actions), 1.0 / len(valid_actions), dtype=np.float64)

            if self.deterministic:
                best_idx = int(np.argmax(valid_probs))
                return int(valid_actions[best_idx])

            chosen = int(self.rng.choice(valid_actions, p=valid_probs))
            return chosen
        finally:
            self.last_move_time_ms = (time.perf_counter() - start_time) * 1000.0

    # ------------------------------------------------------------------
    # Huấn luyện & Lưu / Nạp Trọng số Mô hình
    # ------------------------------------------------------------------

    def train_from_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        learning_rate: float | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Huấn luyện mạng CNN Policy Network trên tập dữ liệu `(X, y)` theo Mục 10.2.1 của sách:
          - `X`: Tensor trạng thái bàn cờ kích thước `(N, 11, 11, 1)`
          - `y`: Mảng nhãn hành động `(N,)` hoặc ma trận one-hot `(N, 121)`
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 3:
            X_arr = X_arr.reshape(-1, self.board_size, self.board_size, 1)

        y_arr = np.asarray(y)
        if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
            y_onehot = to_one_hot(y_arr.reshape(-1), num_classes=self.num_actions)
        elif y_arr.ndim == 2 and y_arr.shape[1] == self.num_actions:
            y_onehot = y_arr.astype(np.float32)
        else:
            raise ValueError(f"Kích thước nhãn y không hợp lệ: {y_arr.shape}.")

        if self.backend == "keras" and self.keras_model is not None:
            if learning_rate is not None:
                self.keras_model.optimizer.learning_rate.assign(float(learning_rate))
            fit_res = self.keras_model.fit(
                X_arr,
                y_onehot,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if X_arr.shape[0] >= 10 else 0.0,
                verbose=verbose,
            )
            self._sync_keras_to_numpy()
            self.training_history = {k: [float(v) for v in vals] for k, vals in fit_res.history.items()}
            return self.training_history

        self.training_history = self.numpy_net.fit(
            X=X_arr,
            y_onehot=y_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            learning_rate=learning_rate,
            verbose=verbose,
        )
        return self.training_history

    def train_self_play(
        self,
        num_games: int = 200,
        random_prob: float = 0.5,
        candidate_radius: int = 2,
        augment_symmetry: bool = True,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        learning_rate: float | None = None,
        save_path: str | Path | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Quy trình huấn luyện khép kín (End-to-End Pipeline) theo Chương 10 sách AlphaGo Simplified:
          1. Sinh dữ liệu từ `num_games` ván đấu giữa Expert và Non-Expert (Mục 10.1).
          2. Mã hóa trạng thái độc lập người chơi `(N, 11, 11, 1)` và one-hot target `(N, 121)`.
          3. Huấn luyện mạng CNN Policy Network (Mục 10.2).
          4. Lưu trọng số vào `save_path` (nếu được cung cấp).
        """
        X, y = generate_expert_dataset(
            num_games=num_games,
            board_size=self.board_size,
            win_length=5 if self.board_size >= 5 else self.board_size,
            random_prob=random_prob,
            candidate_radius=candidate_radius,
            augment_symmetry=augment_symmetry,
            seed=self.seed,
        )
        history = self.train_from_dataset(
            X=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            learning_rate=learning_rate,
            verbose=verbose,
        )
        if save_path is not None:
            self.save_model(save_path)
        return history

    def save_model(self, path: str | Path) -> Path:
        """
        Lưu mô hình ra đĩa.
        - Hỗ trợ định dạng chuẩn `.npz` (tương thích đa nền tảng cho cả Keras và NumPy).
        - Nếu đường dẫn có đuôi `.keras` hoặc `.h5` và có TensorFlow, lưu cả file Keras model
          kèm bản sao `.npz` cùng tên.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "keras" and self.keras_model is not None:
            self._sync_keras_to_numpy()

        weights_dict = self.numpy_net.get_weights_dict()
        if out_path.suffix.lower() in (".keras", ".h5"):
            if self.keras_model is not None:
                self.keras_model.save(out_path)
            npz_companion = out_path.with_suffix(".npz")
            np.savez_compressed(npz_companion, **weights_dict)
            return out_path

        if out_path.suffix.lower() != ".npz":
            out_path = out_path.with_suffix(".npz")
        np.savez_compressed(out_path, **weights_dict)
        return out_path

    def load_model(self, path: str | Path) -> None:
        """
        Nạp trọng số mô hình từ file `.npz`, `.keras`, hoặc `.h5`.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình tại: {in_path}")

        if in_path.suffix.lower() in (".keras", ".h5"):
            npz_companion = in_path.with_suffix(".npz")
            if HAS_TENSORFLOW and models is not None:
                self.keras_model = models.load_model(in_path)
                self._sync_keras_to_numpy()
                return
            if npz_companion.exists():
                in_path = npz_companion
            else:
                raise RuntimeError(
                    f"Không thể nạp file {in_path} vì môi trường chưa cài TensorFlow và không có file {npz_companion}."
                )

        with np.load(in_path, allow_pickle=False) as data:
            self.numpy_net.set_weights_from_dict(data)

        if self.keras_model is not None:
            self._sync_numpy_to_keras()


def main() -> None:
    """Giao diện dòng lệnh (CLI) hỗ trợ sinh dữ liệu, huấn luyện và kiểm thử nhanh CNNAgent."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá CNN Policy Network Agent (Cấp 5) cho Caro 11x11."
    )
    parser.add_argument("--model-type", choices=["fast", "strong"], default="strong", help="Kiến trúc mạng ('fast' hoặc 'strong').")
    parser.add_argument("--backend", choices=["auto", "keras", "numpy"], default="auto", help="Backend tính toán ('auto', 'keras', 'numpy').")
    parser.add_argument("--games", type=int, default=150, help="Số ván đấu tự chơi Expert vs Non-Expert để sinh dữ liệu.")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch huấn luyện.")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước mini-batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học (learning rate) cho Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để tái lập thực nghiệm.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Đường dẫn lưu trọng số mô hình (mặc định: checkpoints/cnn_<model_type>_11x11.npz).",
    )
    args = parser.parse_args()

    save_target = (
        Path(args.save_path)
        if args.save_path
        else DEFAULT_CHECKPOINT_DIR / f"cnn_{args.model_type}_11x11.npz"
    )

    print(f"[1/3] Khởi tạo CNNAgent (model_type={args.model_type}, backend={args.backend})...")
    agent = CNNAgent(
        name=f"Level5_CNN_{args.model_type.capitalize()}",
        model_type=args.model_type,
        backend=args.backend,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print(f"[2/3] Sinh dữ liệu từ {args.games} ván đấu Expert vs Non-Expert và huấn luyện ({args.epochs} epochs)...")
    history = agent.train_self_play(
        num_games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=save_target,
        verbose=1,
    )
    final_acc = history["accuracy"][-1] if history.get("accuracy") else 0.0
    print(f"[3/3] Hoàn tất! Độ chính xác tập huấn luyện cuối cùng: {final_acc:.4f}. Đã lưu tại: {save_target}")


if __name__ == "__main__":
    main()
