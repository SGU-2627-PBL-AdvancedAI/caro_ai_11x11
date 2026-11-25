from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

try:
    from .base_agent import BaseAgent
    from ..caro_env import CaroEnv
    from ..heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions
except ImportError:
    from agents.base_agent import BaseAgent
    from caro_env import CaroEnv
    from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions

# Thử nạp PyTorch cho kiến trúc Residual Policy CNN đa kênh (Cấp 5)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_PYTORCH = True
except Exception:
    torch = None
    nn = None
    F = None
    HAS_PYTORCH = False

# Thử nạp TensorFlow/Keras (nếu môi trường đã cài đặt như trong sách AlphaGo Simplified Chương 10).
# Nếu chưa cài đặt hoặc lỗi phụ thuộc (ví dụ protobuf), hệ thống tự động dùng backend NumPy CNN
# có cùng kiến trúc toán học và định dạng trọng số tương thích hoàn toàn.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers
    from tensorflow.keras.utils import to_categorical as keras_to_categorical

    HAS_TENSORFLOW = True
except Exception:
    tf = None
    layers = None
    models = None
    optimizers = None
    regularizers = None
    keras_to_categorical = None
    HAS_TENSORFLOW = False


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


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


def encode_board_multichannel(
    board: np.ndarray,
    current_player: int = 1,
    last_action: int | None = None,
    in_channels: int = 4,
) -> np.ndarray:
    """
    Mã hóa bàn cờ đa kênh chuẩn hóa góc nhìn người chơi hiện tại ở định dạng NCHW `(C, H, W)`
    dành cho Residual Policy CNN (Cấp 5):
      - Kênh 0: Quân của người chơi đang tới lượt (`board == current_player`)
      - Kênh 1: Quân của đối thủ (`board == -current_player`)
      - Kênh 2: Các ô trống (`board == 0`) — loại bỏ trùng lặp giá trị 0 với zero-padding biên
      - Kênh 3 (nếu `in_channels >= 4`): One-hot nước đi vừa đánh (`last_action`), giúp mạng nhận diện
        ngay trọng tâm chiến thuật mới nhất trên bàn cờ.
    """
    if current_player not in (1, -1):
        raise ValueError(f"current_player phải là 1 hoặc -1, nhận được: {current_player}")
    if in_channels not in (3, 4):
        raise ValueError(f"in_channels phải là 3 hoặc 4, nhận được: {in_channels}")

    arr = np.asarray(board, dtype=np.int8)
    n = arr.shape[0]
    state = np.zeros((in_channels, n, n), dtype=np.float32)
    state[0] = (arr == current_player).astype(np.float32)
    state[1] = (arr == -current_player).astype(np.float32)
    state[2] = (arr == 0).astype(np.float32)

    if in_channels == 4 and last_action is not None and 0 <= int(last_action) < n * n:
        lr, lc = divmod(int(last_action), n)
        state[3, lr, lc] = 1.0

    return state


def augment_multichannel_state_action(
    state_chw: np.ndarray,
    action: int,
    board_size: int = 11,
) -> list[tuple[np.ndarray, int]]:
    """
    Tăng cường 8 phép đối xứng D4 (4 góc quay x 2 phép lật ngang) cho tensor đa kênh `(C, H, W)`
    và nhãn hành động tương ứng.
    """
    arr = np.asarray(state_chw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != board_size or arr.shape[2] != board_size:
        raise ValueError(f"state_chw phải có shape (C, {board_size}, {board_size}), nhận {arr.shape}.")

    action_grid = np.zeros((board_size, board_size), dtype=np.int8)
    r, c = divmod(int(action), board_size)
    action_grid[r, c] = 1

    samples: list[tuple[np.ndarray, int]] = []
    for k in range(4):
        rot_state = np.rot90(arr, k=k, axes=(1, 2))
        rot_act_grid = np.rot90(action_grid, k=k)
        rot_act = int(np.flatnonzero(rot_act_grid.reshape(-1))[0])
        samples.append((rot_state.copy(), rot_act))

        flip_state = rot_state[:, :, ::-1]
        flip_act_grid = np.fliplr(rot_act_grid)
        flip_act = int(np.flatnonzero(flip_act_grid.reshape(-1))[0])
        samples.append((flip_state.copy(), flip_act))

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
      5. Đánh giá tổng hợp công - thủ - vị trí trung tâm.
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
          + Thực hiện nước đi pha trộn để mở rộng không gian trạng thái, nhưng KHÔNG ghi vào dữ liệu.
    """
    if rng is None:
        rng = np.random.default_rng()

    env = CaroEnv(board_size=board_size, win_length=win_length)
    env.reset()

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
    """
    if num_games <= 0:
        raise ValueError("num_games phải là số nguyên dương (> 0).")

    rng = np.random.default_rng(seed)
    all_states: list[np.ndarray] = []
    all_actions: list[int] = []

    # Đa dạng hóa tỷ lệ ngẫu nhiên của Non-Expert quanh mức random_prob để Expert gặp cả thế cờ công và thủ
    prob_choices = [max(0.15, random_prob - 0.2), random_prob, min(0.85, random_prob + 0.2)]

    for idx in range(num_games):
        game_rprob = prob_choices[idx % len(prob_choices)]
        game_states, game_actions, _ = simulate_expert_vs_non_expert_game(
            board_size=board_size,
            win_length=win_length,
            random_prob=game_rprob,
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


def generate_expert_train_val_datasets(
    num_train_games: int = 170,
    num_val_games: int = 30,
    board_size: int = 11,
    win_length: int = 5,
    random_prob: float = 0.5,
    candidate_radius: int = 2,
    augment_train_symmetry: bool = True,
    augment_val_symmetry: bool = False,
    seed: int = 42,
    save_path: str | Path | None = None,
    load_if_exists: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sinh (hoặc nạp lại nếu `load_if_exists=True`) tập dữ liệu Train và Validation
    TÁCH BIỆT THEO VÁN ĐẤU (Game-Level Split) trước khi áp dụng phép biến đổi đối xứng D4.
    Đảm bảo tuyệt đối KHÔNG rò rỉ các biến thể xoay/lật của cùng một thế cờ giữa Train và Val.

    Định dạng lưu trữ khi truyền `save_path` (ví dụ `data/05_cnn_expert_train_val_11x11.npz`):
      - `X_train`: float32, kích thước `(N_train, 11, 11, 1)` — trạng thái bàn cờ góc nhìn chuẩn hóa (+1/-1/0).
      - `y_train`: int64, kích thước `(N_train,)` — nhãn hành động `0..120` của Expert trên tập Train.
      - `X_val`: float32, kích thước `(N_val, 11, 11, 1)` — trạng thái bàn cờ từ các ván Validation độc lập.
      - `y_val`: int64, kích thước `(N_val,)` — nhãn hành động `0..120` của Expert trên tập Validation.
      - Siêu dữ liệu đi kèm: `board_size`, `win_length`, `num_train_games`, `num_val_games`,
        `random_prob`, `augment_train_symmetry`, `augment_val_symmetry`, `seed`.

    Trả về: `(X_train, y_train, X_val, y_val)`
    """
    if save_path is not None and load_if_exists and Path(save_path).exists():
        X_train, y_train, X_val, y_val, _ = load_expert_train_val_datasets(save_path)
        return X_train, y_train, X_val, y_val

    X_train, y_train = generate_expert_dataset(
        num_games=num_train_games,
        board_size=board_size,
        win_length=win_length,
        random_prob=random_prob,
        candidate_radius=candidate_radius,
        augment_symmetry=augment_train_symmetry,
        seed=seed,
    )
    X_val, y_val = generate_expert_dataset(
        num_games=num_val_games,
        board_size=board_size,
        win_length=win_length,
        random_prob=random_prob,
        candidate_radius=candidate_radius,
        augment_symmetry=augment_val_symmetry,
        seed=seed + 100000,
    )

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            board_size=np.array(board_size, dtype=np.int64),
            win_length=np.array(win_length, dtype=np.int64),
            num_train_games=np.array(num_train_games, dtype=np.int64),
            num_val_games=np.array(num_val_games, dtype=np.int64),
            random_prob=np.array(random_prob, dtype=np.float32),
            augment_train_symmetry=np.array(augment_train_symmetry, dtype=np.bool_),
            augment_val_symmetry=np.array(augment_val_symmetry, dtype=np.bool_),
            seed=np.array(seed, dtype=np.int64),
        )

    return X_train, y_train, X_val, y_val


def load_expert_train_val_datasets(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Nạp lại tập dữ liệu tự sinh `(X_train, y_train, X_val, y_val, metadata)` từ file `.npz` trong thư mục `data/`.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {in_path}")
    with np.load(in_path, allow_pickle=False) as d:
        X_train = np.asarray(d["X_train"], dtype=np.float32)
        y_train = np.asarray(d["y_train"], dtype=np.int64)
        X_val = np.asarray(d["X_val"], dtype=np.float32)
        y_val = np.asarray(d["y_val"], dtype=np.int64)
        meta = {
            k: d[k].item()
            for k in d.files
            if k not in ("X_train", "y_train", "X_val", "y_val")
        }
    return X_train, y_train, X_val, y_val, meta


def _canonical_d4_key(
    board: np.ndarray,
    current_player: int,
    last_action: int | None,
) -> bytes:
    """
    Tính khóa bất biến theo nhóm đối xứng D4 (8 phép quay/lật) cho cặp `(bàn cờ chuẩn hóa, last_action)`.
    Dùng để khử trùng lặp tuyệt đối và đảm bảo 0% rò rỉ dữ liệu giữa train / val / test
    ngay cả sau khi áp dụng tăng cường đối xứng D4 trên tập train.
    """
    canonical = (np.asarray(board, dtype=np.int8) * int(current_player)).astype(np.int8)
    n = canonical.shape[0]
    lm_grid = np.zeros((n, n), dtype=np.int8)
    if last_action is not None and 0 <= int(last_action) < n * n:
        r, c = divmod(int(last_action), n)
        lm_grid[r, c] = 1

    best_bytes: bytes | None = None
    for k in range(4):
        rb = np.rot90(canonical, k=k)
        rl = np.rot90(lm_grid, k=k)
        cand1 = rb.tobytes() + rl.tobytes()
        if best_bytes is None or cand1 < best_bytes:
            best_bytes = cand1

        fb = np.fliplr(rb)
        fl = np.fliplr(rl)
        cand2 = fb.tobytes() + fl.tobytes()
        if cand2 < best_bytes:
            best_bytes = cand2

    assert best_bytes is not None
    return best_bytes


def _build_teacher_agent(teacher: str, seed: int, candidate_radius: int):
    """Khởi tạo tác tử chuyên gia (teacher) phục vụ sinh dữ liệu học có giám sát."""
    if teacher == "legacy":
        return None
    if teacher == "rule_based":
        try:
            from .rule_based_agent import RuleBasedAgent
        except ImportError:
            from agents.rule_based_agent import RuleBasedAgent
        return RuleBasedAgent(
            name="Teacher_RuleBased",
            candidate_radius=candidate_radius,
            seed=seed,
            tie_break_eps=1e-6,
        )
    if teacher == "minimax":
        try:
            from .minimax_agent import MinimaxAgent
        except ImportError:
            from agents.minimax_agent import MinimaxAgent
        return MinimaxAgent(
            name="Teacher_Minimax",
            max_depth=3,
            candidate_radius=candidate_radius,
            time_limit_sec=0.5,
            max_branch_root=14,
            max_branch_inner=10,
            seed=seed,
            tie_break_eps=1e-6,
        )
    raise ValueError(f"teacher không hợp lệ: {teacher!r}. Chọn 'rule_based', 'minimax', 'mixed', hoặc 'legacy'.")


def _collect_teacher_games(
    num_games: int,
    teacher: str,
    board_size: int,
    win_length: int,
    in_channels: int,
    candidate_radius: int,
    opening_random_plies: int,
    opponent_noise_prob: float,
    seen_d4_keys: set[bytes],
    seed: int,
) -> tuple[list[np.ndarray], list[int], int, int]:
    """
    Sinh các mẫu `(state_chw, expert_action)` từ `num_games` ván đấu độc lập,
    khử trùng lặp theo khóa bất biến D4 `seen_d4_keys`.
    Trả về `(states, actions, raw_recorded, dedup_skipped)`.
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    actions: list[int] = []
    raw_recorded = 0
    dedup_skipped = 0

    center = board_size // 2
    opening_zone = [
        r * board_size + c
        for r in range(max(0, center - 2), min(board_size, center + 3))
        for c in range(max(0, center - 2), min(board_size, center + 3))
    ]

    rb_teacher = _build_teacher_agent("rule_based", seed, candidate_radius) if teacher in ("rule_based", "mixed") else None
    mm_teacher = _build_teacher_agent("minimax", seed + 777, candidate_radius) if teacher in ("minimax", "mixed") else None

    for g in range(num_games):
        env = CaroEnv(board_size=board_size, win_length=win_length)
        # Chọn chuyên gia cho ván này (nếu teacher="mixed": 70% RuleBased + 30% Minimax)
        if teacher == "mixed":
            use_mm = (g % 10) < 3
            active_teacher = mm_teacher if use_mm else rb_teacher
        elif teacher == "minimax":
            active_teacher = mm_teacher
        elif teacher == "rule_based":
            active_teacher = rb_teacher
        else:
            active_teacher = None

        # Số bước khai cuộc ngẫu nhiên quanh trung tâm (1..opening_random_plies) để đa dạng hóa thế cờ
        rand_plies = int(rng.integers(1, opening_random_plies + 1)) if opening_random_plies > 0 else 0
        # Một bên được gắn nhãn chuyên gia hoàn toàn; bên kia thỉnh thoảng đi nhiễu để tạo thế công/thủ thực tế
        noisy_side = 1 if (g % 2 == 0) else -1

        while not env.is_terminal():
            ply = len(env.history)
            if ply < rand_plies:
                legal_open = [a for a in opening_zone if env.board[a // board_size, a % board_size] == 0]
                move = int(rng.choice(legal_open)) if legal_open else int(rng.choice(env.legal_actions()))
                env.step(move)
                continue

            # Nếu bên nhiễu tới lượt và tung trúng xác suất nhiễu -> đánh ngẫu nhiên trong vùng ứng viên (không ghi nhãn)
            if env.current_player == noisy_side and opponent_noise_prob > 0.0 and float(rng.random()) < opponent_noise_prob:
                cands = get_candidate_actions(env, radius=candidate_radius)
                move = int(rng.choice(cands)) if cands else int(rng.choice(env.legal_actions()))
                env.step(move)
                continue

            # Lấy nước đi chuẩn từ chuyên gia
            if active_teacher is not None:
                move = int(active_teacher.select_action(env))
            else:
                move = expert_select_action(env, candidate_radius=candidate_radius, rng=rng)

            last_act = env.history[-1] if env.history else None
            raw_recorded += 1
            d4_key = _canonical_d4_key(env.board, env.current_player, last_act)
            if d4_key in seen_d4_keys:
                dedup_skipped += 1
            else:
                seen_d4_keys.add(d4_key)
                state_chw = encode_board_multichannel(
                    env.board,
                    current_player=env.current_player,
                    last_action=last_act,
                    in_channels=in_channels,
                )
                states.append(state_chw)
                actions.append(int(move))

            env.step(move)

    return states, actions, raw_recorded, dedup_skipped


def generate_expert_train_val_test_datasets(
    num_train_games: int = 120,
    num_val_games: int = 20,
    num_test_games: int = 20,
    teacher: str = "mixed",
    board_size: int = 11,
    win_length: int = 5,
    in_channels: int = 4,
    candidate_radius: int = 2,
    opening_random_plies: int = 3,
    opponent_noise_prob: float = 0.25,
    augment_train_symmetry: bool = True,
    seed: int = 42,
    save_path: str | Path | None = None,
    load_if_exists: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Sinh bộ dữ liệu học có giám sát chuẩn hóa 3 tập độc lập `(Train / Val / Test)` theo cấp độ
    ván đấu (Game-Level Split) kèm khử trùng lặp bất biến D4 (D4-Invariant Deduplication):
      - Tập `Val` và `Test` được sinh từ các ván đấu hoàn toàn tách biệt, giữ nguyên mẫu gốc
        (không tăng cường D4) để đánh giá trung thực khả năng tổng quát hóa.
      - Khóa D4 toàn cục đảm bảo 0% trạng thái (kể cả biến thể quay/lật) của `Val` hoặc `Test`
        xuất hiện trong `Train`.
      - Dữ liệu đa kênh `(N, in_channels, board_size, board_size)` phục vụ PyTorch ResNet (Cấp 5).
    """
    if save_path is not None and load_if_exists and Path(save_path).exists():
        return load_expert_train_val_test_datasets(save_path)

    seen_d4_keys: set[bytes] = set()

    # Sinh Test và Val trước để bảo đảm các mẫu đánh giá độc lập không bị loại bởi tập Train
    test_states, test_actions, raw_test, dedup_test = _collect_teacher_games(
        num_games=num_test_games,
        teacher=teacher,
        board_size=board_size,
        win_length=win_length,
        in_channels=in_channels,
        candidate_radius=candidate_radius,
        opening_random_plies=opening_random_plies,
        opponent_noise_prob=opponent_noise_prob,
        seen_d4_keys=seen_d4_keys,
        seed=seed + 200000,
    )
    val_states, val_actions, raw_val, dedup_val = _collect_teacher_games(
        num_games=num_val_games,
        teacher=teacher,
        board_size=board_size,
        win_length=win_length,
        in_channels=in_channels,
        candidate_radius=candidate_radius,
        opening_random_plies=opening_random_plies,
        opponent_noise_prob=opponent_noise_prob,
        seen_d4_keys=seen_d4_keys,
        seed=seed + 100000,
    )
    train_states_raw, train_actions_raw, raw_train, dedup_train = _collect_teacher_games(
        num_games=num_train_games,
        teacher=teacher,
        board_size=board_size,
        win_length=win_length,
        in_channels=in_channels,
        candidate_radius=candidate_radius,
        opening_random_plies=opening_random_plies,
        opponent_noise_prob=opponent_noise_prob,
        seen_d4_keys=seen_d4_keys,
        seed=seed,
    )

    if augment_train_symmetry:
        train_states: list[np.ndarray] = []
        train_actions: list[int] = []
        for s_chw, act in zip(train_states_raw, train_actions_raw):
            for aug_s, aug_a in augment_multichannel_state_action(s_chw, act, board_size=board_size):
                train_states.append(aug_s)
                train_actions.append(aug_a)
    else:
        train_states = train_states_raw
        train_actions = train_actions_raw

    empty_shape = (0, in_channels, board_size, board_size)
    X_train = np.asarray(train_states, dtype=np.float32) if train_states else np.empty(empty_shape, dtype=np.float32)
    y_train = np.asarray(train_actions, dtype=np.int64)
    X_val = np.asarray(val_states, dtype=np.float32) if val_states else np.empty(empty_shape, dtype=np.float32)
    y_val = np.asarray(val_actions, dtype=np.int64)
    X_test = np.asarray(test_states, dtype=np.float32) if test_states else np.empty(empty_shape, dtype=np.float32)
    y_test = np.asarray(test_actions, dtype=np.int64)

    metadata: dict[str, Any] = {
        "board_size": int(board_size),
        "win_length": int(win_length),
        "in_channels": int(in_channels),
        "teacher": str(teacher),
        "num_train_games": int(num_train_games),
        "num_val_games": int(num_val_games),
        "num_test_games": int(num_test_games),
        "raw_train_samples": int(raw_train),
        "raw_val_samples": int(raw_val),
        "raw_test_samples": int(raw_test),
        "dedup_removed_total": int(dedup_train + dedup_val + dedup_test),
        "train_unique_samples": int(len(train_states_raw)),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "augment_train_symmetry": bool(augment_train_symmetry),
        "opening_random_plies": int(opening_random_plies),
        "opponent_noise_prob": float(opponent_noise_prob),
        "seed": int(seed),
    }

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_payload: dict[str, Any] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }
        for k, v in metadata.items():
            save_payload[k] = np.array(v)
        np.savez_compressed(out_path, **save_payload)

    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


def load_expert_train_val_test_datasets(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Nạp lại bộ dữ liệu 3 tập `(X_train, y_train, X_val, y_val, X_test, y_test, metadata)` từ `.npz`."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {in_path}")
    with np.load(in_path, allow_pickle=False) as d:
        X_train = np.asarray(d["X_train"], dtype=np.float32)
        y_train = np.asarray(d["y_train"], dtype=np.int64)
        X_val = np.asarray(d["X_val"], dtype=np.float32)
        y_val = np.asarray(d["y_val"], dtype=np.int64)
        X_test = np.asarray(d["X_test"], dtype=np.float32)
        y_test = np.asarray(d["y_test"], dtype=np.int64)
        meta = {
            k: d[k].item()
            for k in d.files
            if k not in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
        }
    return X_train, y_train, X_val, y_val, X_test, y_test, meta


def export_dataset_to_csv(
    X: np.ndarray,
    y: np.ndarray,
    csv_path: str | Path,
    split_label: str = "train",
    board_size: int = 11,
    append: bool = False,
) -> Path:
    """
    Xuất tập dữ liệu tự sinh `(X, y)` sang định dạng bảng `.csv` (UTF-8-SIG) tại `csv_path`
    (ví dụ: `results/tables/04_cnn_dataset_train_val_11x11.csv` hoặc `04_cnn_dataset_train_noaug_11x11.csv`).

    Cấu trúc các cột trong bảng CSV:
      - `sample_id`: Thứ tự mẫu (1..N)
      - `split`: Nhãn phân tập (`train_aug`, `val_unseen`, hoặc `train_noaug`)
      - `action`: Nhãn nước đi của Expert (`0..120`)
      - `row`: Tọa độ hàng (`action // 11`)
      - `col`: Tọa độ cột (`action % 11`)
      - `my_stones`: Số quân của người chơi hiện tại (`+1`) trên bàn cờ
      - `opp_stones`: Số quân của đối thủ (`-1`) trên bàn cờ
      - `board_flat`: Chuỗi 121 giá trị trạng thái bàn cờ góc nhìn chuẩn hóa (`-1`, `0`, `1`) cách nhau bởi dấu cách
    """
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and out_path.exists() else "w"
    fieldnames = [
        "sample_id",
        "split",
        "action",
        "row",
        "col",
        "my_stones",
        "opp_stones",
        "board_flat",
    ]
    X_int = np.asarray(X, dtype=np.int8).reshape(-1, board_size * board_size)
    y_int = np.asarray(y, dtype=np.int64).reshape(-1)

    with out_path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for idx in range(X_int.shape[0]):
            flat = X_int[idx]
            act = int(y_int[idx])
            r, c = divmod(act, board_size)
            writer.writerow({
                "sample_id": idx + 1,
                "split": split_label,
                "action": act,
                "row": r,
                "col": c,
                "my_stones": int(np.sum(flat == 1)),
                "opp_stones": int(np.sum(flat == -1)),
                "board_flat": " ".join(str(int(v)) for v in flat),
            })
    return out_path


# ==============================================================================
# 3. KIẾN TRÚC MẠNG POLICY NETWORK (FAST & STRONG POLICY NETWORKS)
#    (Tham khảo Mục 10.2.1 - "Train Two Policy Networks" trong sách AlphaGo Simplified)
# ==============================================================================


class NumpyCNNPolicyNetwork:
    """
    Triển khai mạng CNN Policy Network bằng NumPy thuần túy (Vectorized Conv2D + ReLU + Flatten
    + Dense layers + Softmax + Categorical Cross-Entropy + L2 Regularization + Adam Optimizer).
    Đảm bảo kiến trúc và chiều trọng số giống hệt 100% với mô hình `tensorflow.keras.Sequential`.
    """

    def __init__(
        self,
        board_size: int = 11,
        model_type: str = "strong",
        conv_filters: int = 128,
        kernel_size: int = 5,
        dense_units: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
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
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)

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
        """Lan truyền tiến (Forward Pass) qua mạng CNN Policy Network."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 3:
            x_arr = x_arr.reshape(1, self.board_size, self.board_size, 1)

        bsz = x_arr.shape[0]
        patches = self._extract_patches(x_arr)
        w_col = self.conv_w.reshape(-1, self.conv_filters)
        conv_pre = patches @ w_col + self.conv_b
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

    def train_on_batch(
        self,
        x_batch: np.ndarray,
        y_onehot_batch: np.ndarray,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """Thực hiện 1 bước lan truyền tiến + ngược và cập nhật Adam với L2 weight decay."""
        use_lr = self.learning_rate if lr is None else float(lr)
        bsz = x_batch.shape[0]
        probs, cache = self.forward(x_batch, return_cache=True)

        clipped_probs = np.clip(probs, 1e-7, 1.0)
        loss = float(-np.sum(y_onehot_batch * np.log(clipped_probs)) / bsz)
        preds = np.argmax(probs, axis=-1)
        targets = np.argmax(y_onehot_batch, axis=-1)
        acc = float(np.mean(preds == targets))

        d_logits = (probs - y_onehot_batch) / float(bsz)

        grad_dense_w: list[np.ndarray] = [None] * len(self.dense_weights)  # type: ignore[list-item]
        grad_dense_b: list[np.ndarray] = [None] * len(self.dense_biases)  # type: ignore[list-item]

        logits_in = cache["logits_in"]
        grad_dense_w[-1] = logits_in.T @ d_logits + self.weight_decay * self.dense_weights[-1]
        grad_dense_b[-1] = np.sum(d_logits, axis=0)
        d_curr = d_logits @ self.dense_weights[-1].T

        for i in range(len(self.dense_weights) - 2, -1, -1):
            pre = cache["hidden_pres"][i]
            inp = cache["hidden_inputs"][i]
            d_pre = d_curr * (pre > 0.0)
            grad_dense_w[i] = inp.T @ d_pre + self.weight_decay * self.dense_weights[i]
            grad_dense_b[i] = np.sum(d_pre, axis=0)
            d_curr = d_pre @ self.dense_weights[i].T

        conv_pre = cache["conv_pre"]
        d_conv_act = d_curr.reshape(conv_pre.shape)
        d_conv_pre = d_conv_act * (conv_pre > 0.0)

        patches_flat = cache["patches"].reshape(-1, self.kernel_size * self.kernel_size)
        d_conv_pre_flat = d_conv_pre.reshape(-1, self.conv_filters)

        grad_conv_w = (patches_flat.T @ d_conv_pre_flat).reshape(self.conv_w.shape) + self.weight_decay * self.conv_w
        grad_conv_b = np.sum(d_conv_pre_flat, axis=0)

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
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        learning_rate: float | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Huấn luyện mạng CNN Policy Network theo mini-batch.
        Hỗ trợ truyền `validation_data=(X_val, y_val_onehot)` tách biệt ở cấp độ ván đấu
        để tránh rò rỉ dữ liệu đối xứng D4 giữa Train và Validation.
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Tập dữ liệu huấn luyện X không được rỗng.")

        if validation_data is not None:
            X_train, y_train = X, y_onehot
            X_val = np.asarray(validation_data[0], dtype=np.float32)
            if X_val.ndim == 3:
                X_val = X_val.reshape(-1, self.board_size, self.board_size, 1)
            raw_y_val = np.asarray(validation_data[1])
            if raw_y_val.ndim == 1 or (raw_y_val.ndim == 2 and raw_y_val.shape[1] == 1):
                y_val = to_one_hot(raw_y_val.reshape(-1), num_classes=self.num_actions)
            else:
                y_val = raw_y_val.astype(np.float32)
        elif 0.0 < validation_split < 1.0 and n_samples >= 10:
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
        self.kernel_size = int(self.conv_w.shape[0])
        self.conv_filters = int(self.conv_w.shape[3])
        num_dense = int(data["num_dense"])
        self.dense_weights = [np.asarray(data[f"dense_w_{i}"], dtype=np.float32) for i in range(num_dense)]
        self.dense_biases = [np.asarray(data[f"dense_b_{i}"], dtype=np.float32) for i in range(num_dense)]
        self._init_adam_state()


def build_keras_policy_network(
    board_size: int = 11,
    model_type: str = "strong",
    conv_filters: int = 128,
    kernel_size: int = 5,
    dense_units: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Any:
    """
    Khởi tạo mô hình CNN Policy Network bằng `tensorflow.keras.Sequential` theo Mục 10.2.1
    sách AlphaGo Simplified (Mark Liu).
    """
    if not HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow/Keras chưa được cài đặt trong môi trường hiện tại.")
    if model_type not in ("fast", "strong"):
        raise ValueError("model_type phải là 'fast' hoặc 'strong'.")

    num_actions = board_size * board_size
    reg = regularizers.l2(weight_decay) if regularizers is not None and weight_decay > 0 else None
    model = models.Sequential(name=f"caro_{board_size}x{board_size}_{model_type}_policy_net")
    model.add(layers.Input(shape=(board_size, board_size, 1)))
    model.add(
        layers.Conv2D(
            filters=conv_filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name="conv2d_feature_extractor",
        )
    )
    model.add(layers.Flatten(name="flatten_board"))
    model.add(layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense_hidden_1"))
    model.add(layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense_hidden_2"))
    if model_type == "strong":
        model.add(layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense_hidden_3"))
    model.add(layers.Dense(num_actions, activation="softmax", kernel_regularizer=reg, name="policy_head_softmax"))

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


def _fast_check_five(board: np.ndarray, r: int, c: int, player: int, win_length: int = 5) -> bool:
    """
    Kiểm tra siêu nhanh O(win_length) xem việc đặt quân `player` tại ô trống `(r, c)`
    có tạo thành chuỗi >= `win_length` quân liên tiếp hay không.
    Dùng trong bước Rollout của Mixed MCTS để tăng tốc độ mô phỏng gấp ~30 lần.
    """
    n = board.shape[0]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for step in range(1, win_length):
            nr, nc = r + dr * step, c + dc * step
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == player:
                count += 1
            else:
                break
        for step in range(1, win_length):
            nr, nc = r - dr * step, c - dc * step
            if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == player:
                count += 1
            else:
                break
        if count >= win_length:
            return True
    return False


if HAS_PYTORCH:

    class TorchResidualBlock(nn.Module):
        """Khối Residual 3x3 chuẩn cho bàn cờ Caro (bảo toàn kích thước không gian HxW)."""

        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            return F.relu(out + x, inplace=True)

    class TorchPolicyResNet(nn.Module):
        """
        Mạng Residual Policy CNN thuần tích chập (Fully Convolutional Policy ResNet) cho Caro 11x11
        (Cấp 5):
          - Đầu vào: `(B, in_channels, H, W)` với `in_channels = 4` (`my`, `opp`, `empty`, `last_move`).
          - Stem: `Conv3x3(in_channels -> channels) + BN + ReLU`.
          - Residual Tower: `num_res_blocks` khối Residual 3x3 (với 4 block, trường nhìn đạt 19x19,
            phủ trọn bàn cờ 11x11 và giữ tính bất biến tịnh tiến trên toàn bàn).
          - Policy Head không gian: `Conv1x1(channels -> 16) + BN + ReLU + Conv1x1(16 -> 1) -> Flatten(B, H*W)`.
            Loại bỏ hoàn toàn nút cổ chai `Flatten(15488) -> Dense(64)` gây overfit ở bản cũ.
        """

        def __init__(
            self,
            board_size: int = 11,
            in_channels: int = 4,
            channels: int = 64,
            num_res_blocks: int = 4,
        ):
            super().__init__()
            self.board_size = int(board_size)
            self.in_channels = int(in_channels)
            self.channels = int(channels)
            self.num_res_blocks = int(num_res_blocks)

            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(*[TorchResidualBlock(self.channels) for _ in range(self.num_res_blocks)])
            self.policy_head = nn.Sequential(
                nn.Conv2d(self.channels, 16, kernel_size=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, bias=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.stem(x)
            h = self.blocks(h)
            logits = self.policy_head(h)
            return logits.flatten(start_dim=1)

else:

    class TorchResidualBlock:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cần cài đặt PyTorch để sử dụng TorchResidualBlock.")

    class TorchPolicyResNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cần cài đặt PyTorch để sử dụng TorchPolicyResNet.")


class _PUCTNode:
    """Nút trong cây tìm kiếm PUCT-MCTS chuẩn AlphaGo định hướng bởi mạng CNN Policy."""

    __slots__ = (
        "parent",
        "action_from_parent",
        "player_just_moved",
        "prior",
        "visits",
        "wins",
        "children",
        "is_expanded",
    )

    def __init__(
        self,
        parent: _PUCTNode | None = None,
        action_from_parent: int | None = None,
        player_just_moved: int = 0,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.player_just_moved = player_just_moved
        self.prior = float(prior)
        self.visits: int = 0
        self.wins: float = 0.0
        self.children: dict[int, _PUCTNode] = {}
        self.is_expanded: bool = False

    def expand(self, actions_and_priors: list[tuple[int, float]], next_player: int) -> None:
        for act, prior in actions_and_priors:
            self.children[act] = _PUCTNode(
                parent=self,
                action_from_parent=act,
                player_just_moved=next_player,
                prior=prior,
            )
        self.is_expanded = True

    def select_child_puct(self, c_puct: float, fpu_reduction: float = 0.15) -> _PUCTNode:
        sqrt_n = math.sqrt(self.visits + 1.0)
        parent_q = (self.wins / self.visits) if self.visits > 0 else 0.5
        fpu = max(0.0, parent_q - fpu_reduction) if self.visits > 0 else 0.5

        best_score = -math.inf
        best_child: _PUCTNode | None = None
        for child in self.children.values():
            q_val = (child.wins / child.visits) if child.visits > 0 else fpu
            u_val = c_puct * child.prior * (sqrt_n / (1.0 + child.visits))
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child


class CNNAgent(BaseAgent):
    """
    Tác tử Cấp 5: Mạng Nơ-ron Tích chập (Convolutional Policy Network) cho Caro 11x11.
    Hỗ trợ đồng thời:
      - Kiến trúc nâng cấp 5 (`model_type="resnet"`, `backend="pytorch"`): Residual Policy CNN
        4 kênh `(my, opp, empty, last_move)`, lưu/nạp `.pt`, tích hợp PUCT-MCTS nhiều tầng.
      - Kiến trúc tham chiếu cũ (`model_type="fast"|"strong"`, `backend="numpy"|"keras"`):
        tương thích ngược hoàn toàn với các checkpoint `.npz` và kiểm thử cũ.
    """

    def __init__(
        self,
        name: str = "Level5_CNN_Policy",
        model_type: str = "strong",
        backend: str = "auto",
        model_path: str | Path | None = None,
        auto_load_default: bool = False,
        board_size: int = 11,
        conv_filters: int = 128,
        kernel_size: int = 5,
        dense_units: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        deterministic: bool = True,
        temperature: float = 1.0,
        candidate_radius: int | None = 2,
        use_tactical_shortcut: bool = True,
        use_mix_mcts: bool = False,
        num_rollouts: int = 64,
        exploration_c: float = 1.414,
        gamma: float = 10.0,
        max_rollout_steps: int = 20,
        time_limit_sec: float | None = 1.0,
        seed: int | None = 42,
        in_channels: int = 4,
        res_channels: int = 64,
        num_res_blocks: int = 4,
        device: str = "auto",
    ):
        super().__init__(name=name)

        # Tự nhận diện checkpoint PyTorch (.pt / .pth) nếu người dùng chỉ truyền model_path
        if model_path is not None and Path(model_path).suffix.lower() in (".pt", ".pth"):
            if backend == "auto":
                backend = "pytorch"
            if model_type in ("fast", "strong"):
                model_type = "resnet"

        if model_type not in ("fast", "strong", "resnet"):
            raise ValueError("model_type phải là 'fast', 'strong' hoặc 'resnet'.")
        if backend not in ("auto", "pytorch", "keras", "numpy"):
            raise ValueError("backend phải là 'auto', 'pytorch', 'keras' hoặc 'numpy'.")
        if board_size < 3:
            raise ValueError("board_size phải >= 3.")
        if temperature <= 0.0:
            raise ValueError("temperature phải > 0.")
        if candidate_radius is not None and candidate_radius < 0:
            raise ValueError("candidate_radius phải >= 0 hoặc None.")
        if num_rollouts <= 0:
            raise ValueError("num_rollouts phải > 0.")
        if in_channels not in (3, 4):
            raise ValueError("in_channels phải là 3 hoặc 4.")
        if res_channels < 8 or num_res_blocks < 1:
            raise ValueError("res_channels phải >= 8 và num_res_blocks phải >= 1.")

        self.model_type = model_type
        if backend == "auto":
            if model_type == "resnet":
                if not HAS_PYTORCH:
                    raise RuntimeError("model_type='resnet' yêu cầu cài đặt PyTorch.")
                self.backend = "pytorch"
            else:
                self.backend = "keras" if HAS_TENSORFLOW else "numpy"
        elif backend == "pytorch":
            if not HAS_PYTORCH:
                raise RuntimeError("Yêu cầu backend='pytorch' nhưng PyTorch chưa được cài đặt.")
            self.backend = "pytorch"
            self.model_type = "resnet"
        elif backend == "keras" and not HAS_TENSORFLOW:
            raise RuntimeError("Yêu cầu backend='keras' nhưng TensorFlow chưa được cài đặt.")
        else:
            self.backend = backend

        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size
        self.conv_filters = int(conv_filters)
        self.kernel_size = int(kernel_size)
        self.dense_units = int(dense_units)
        self.in_channels = int(in_channels)
        self.res_channels = int(res_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.deterministic = bool(deterministic)
        self.temperature = float(temperature)
        self.candidate_radius = candidate_radius
        self.use_tactical_shortcut = bool(use_tactical_shortcut)

        # Cấu hình chế độ Policy-Guided PUCT-MCTS
        self.use_mix_mcts = bool(use_mix_mcts)
        self.num_rollouts = int(num_rollouts)
        self.exploration_c = float(exploration_c)
        self.gamma = float(gamma)
        self.max_rollout_steps = int(max_rollout_steps)
        self.time_limit_sec = float(time_limit_sec) if time_limit_sec is not None else None

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(int(self.rng.integers(0, (1 << 63) - 1)))

        self.device_str = device
        self.device: Any = None
        self.torch_model: Any = None
        self.numpy_net: Any = None
        self.keras_model: Any = None

        if self.backend == "pytorch":
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            if seed is not None:
                torch.manual_seed(int(seed))
            self.torch_model = TorchPolicyResNet(
                board_size=self.board_size,
                in_channels=self.in_channels,
                channels=self.res_channels,
                num_res_blocks=self.num_res_blocks,
            ).to(self.device)
            self.torch_model.eval()
        else:
            self.numpy_net = NumpyCNNPolicyNetwork(
                board_size=self.board_size,
                model_type=self.model_type,
                conv_filters=self.conv_filters,
                kernel_size=self.kernel_size,
                dense_units=self.dense_units,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                seed=seed,
            )
            if self.backend == "keras":
                self.keras_model = build_keras_policy_network(
                    board_size=self.board_size,
                    model_type=self.model_type,
                    conv_filters=self.conv_filters,
                    kernel_size=self.kernel_size,
                    dense_units=self.dense_units,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
                self._sync_numpy_to_keras()

        # Chỉ số đo đạc phục vụ Giải đấu và Ablation Study
        self.last_move_time_ms: float = 0.0
        self.last_search_count: int = 0
        self.last_decision_stage: str = ""
        self.last_action_probs: np.ndarray | None = None
        self.training_history: dict[str, list[float]] = {}

        # Chỉ nạp trọng số khi người dùng truyền rõ `model_path` hoặc bật `auto_load_default=True`
        if model_path is not None:
            self.load_model(model_path)
        elif auto_load_default:
            if self.backend == "pytorch":
                default_ckpt = DEFAULT_CHECKPOINT_DIR / "05_cnn_policy_best.pt"
            else:
                default_ckpt = DEFAULT_CHECKPOINT_DIR / f"cnn_{self.model_type}_{self.board_size}x{self.board_size}.npz"
            if default_ckpt.exists():
                self.load_model(default_ckpt)

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
        self.numpy_net.kernel_size = int(self.numpy_net.conv_w.shape[0])
        self.numpy_net.conv_filters = int(self.numpy_net.conv_w.shape[3])
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

    def encode_env_state_multichannel(self, env: CaroEnv) -> np.ndarray:
        """
        Biểu diễn trạng thái bàn cờ `env` thành tensor đa kênh `(1, in_channels, H, W)`
        cho mạng PyTorch Residual Policy CNN (Cấp 5).
        """
        if env.board_size != self.board_size:
            raise ValueError(
                f"Kích thước bàn cờ của env ({env.board_size}) không khớp với CNNAgent ({self.board_size})."
            )
        last_act = env.history[-1] if env.history else None
        return encode_board_multichannel(
            env.board,
            current_player=env.current_player,
            last_action=last_act,
            in_channels=self.in_channels,
        )[np.newaxis, ...]

    def predict_probs(
        self,
        env: CaroEnv,
        restrict_to_candidates: bool = False,
    ) -> np.ndarray:
        """
        Dự đoán vector phân phối xác suất có độ dài `121` cho các nước đi trên bàn cờ hiện tại.
        Loại bỏ hoàn toàn các ô đã có quân (hoặc ngoài vùng ứng viên nếu `restrict_to_candidates=True`)
        và chuẩn hóa tổng xác suất về `1.0`.
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

        if self.backend == "pytorch" and self.torch_model is not None:
            state_nchw = self.encode_env_state_multichannel(env)
            self.torch_model.eval()
            with torch.no_grad():
                tensor_in = torch.from_numpy(state_nchw).to(self.device)
                logits = self.torch_model(tensor_in).squeeze(0).detach().cpu().numpy().astype(np.float64)
            valid_logits = logits[valid_actions] / float(self.temperature)
            valid_logits -= float(np.max(valid_logits))
            exp_vals = np.exp(valid_logits)
            total = float(np.sum(exp_vals))
            masked_probs = np.zeros((self.num_actions,), dtype=np.float32)
            if total > 1e-12 and np.isfinite(total):
                masked_probs[valid_actions] = (exp_vals / total).astype(np.float32)
            else:
                masked_probs[valid_actions] = np.float32(1.0 / len(valid_actions))
            self.last_action_probs = masked_probs
            return masked_probs

        state_4d = self.encode_env_state(env)
        raw_probs = self.numpy_net.forward(state_4d, return_cache=False).reshape(-1)

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

        for action in candidates:
            mt = evaluate_cell_threat(env, action, player)
            my_threats[action] = mt
            if mt["is_win_5"]:
                return int(action), "win_5"

        block_wins: list[int] = []
        for action in candidates:
            ot = evaluate_cell_threat(env, action, opponent)
            opp_threats[action] = ot
            if ot["is_win_5"]:
                block_wins.append(action)
        if block_wins:
            best_block = max(block_wins, key=lambda a: my_threats[a]["score"] + opp_threats[a]["score"])
            return int(best_block), "block_win_5"

        open_4_moves = [a for a in candidates if my_threats[a]["open_4"] >= 1]
        if open_4_moves:
            best_o4 = max(open_4_moves, key=lambda a: my_threats[a]["score"])
            return int(best_o4), "create_open_4"

        block_open_4 = [a for a in candidates if opp_threats[a]["open_4"] >= 1]
        if block_open_4:
            best_bo4 = max(block_open_4, key=lambda a: opp_threats[a]["score"] + my_threats[a]["score"])
            return int(best_bo4), "block_open_4"

        return None, ""

    # ------------------------------------------------------------------
    # Chế độ Policy-Guided Mixed MCTS (Mục 10.4 sách AlphaGo Simplified)
    # ------------------------------------------------------------------

    def _rollout_policy_game(self, sim_env: CaroEnv, root_player: int, root_priors: np.ndarray) -> float:
        """
        Mô phỏng 1 ván đấu tốc độ cao từ trạng thái nút con (tương ứng `one_game()` trong Mục 10.4.2):
          - Cập nhật tập ô trống ứng viên tăng dần (incremental candidate set) quanh các quân vừa đánh.
          - Kiểm tra thắng/chặn 5 quân cục bộ quanh 2 nước đi gần nhất bằng `_fast_check_five`.
          - Lấy mẫu hành động có trọng số theo `root_priors` (chính sách rollout định hướng bởi CNN).
        """
        steps = 0
        n = sim_env.board_size
        win_len = sim_env.win_length
        board = sim_env.board

        cands_list = get_candidate_actions(sim_env, radius=1)
        cand_set = set(cands_list)

        while not sim_env.is_terminal() and steps < self.max_rollout_steps:
            if not cand_set:
                cand_set = set(sim_env.legal_actions())
                if not cand_set:
                    break

            cur_p = sim_env.current_player
            opp_p = -cur_p
            win_now = None
            block_now = None

            # Chỉ kiểm tra thắng/chặn 5 quân trên các ô trống quanh 2 nước đi cuối cùng để đạt tốc độ O(1)
            recent_moves = sim_env.history[-2:]
            check_cells: list[int] = []
            for rm in recent_moves:
                rr, rc = divmod(rm, n)
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = rr + dr, rc + dc
                        if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                            check_cells.append(nr * n + nc)

            for a in check_cells:
                r, c = divmod(a, n)
                if _fast_check_five(board, r, c, cur_p, win_len):
                    win_now = a
                    break
                if block_now is None and _fast_check_five(board, r, c, opp_p, win_len):
                    block_now = a

            if win_now is not None:
                move = win_now
            elif block_now is not None:
                move = block_now
            else:
                c_arr = np.fromiter(cand_set, dtype=np.int64, count=len(cand_set))
                cand_weights = root_priors[c_arr].astype(np.float64) + 0.02
                cand_weights /= np.sum(cand_weights)
                move = int(self.rng.choice(c_arr, p=cand_weights))

            sim_env.step(move)
            cand_set.discard(move)
            mr, mc = divmod(move, n)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = mr + dr, mc + dc
                    if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                        cand_set.add(nr * n + nc)
            steps += 1

        if sim_env.is_terminal():
            return float(sim_env.result(perspective=1))

        leaf_score = evaluate_board(sim_env, perspective=1, consider_turn=True)
        return float(math.tanh(leaf_score / 12000.0))

    def _build_puct_priors(
        self,
        env: CaroEnv,
        candidates: list[int],
        prior_probs: np.ndarray | None = None,
        max_branch: int = 12,
    ) -> list[tuple[int, float]]:
        """Trả về danh sách `(action, prior)` của top-K ứng viên mạnh nhất theo mạng CNN."""
        if not candidates:
            return []
        probs = (
            prior_probs
            if prior_probs is not None
            else self.predict_probs(env, restrict_to_candidates=(self.candidate_radius is not None))
        )
        k = min(len(candidates), max_branch)
        ranked_indices = np.argsort(probs[candidates])[::-1][:k]
        top_actions = [int(candidates[idx]) for idx in ranked_indices]
        raw_p = np.array([max(1e-6, float(probs[a])) for a in top_actions], dtype=np.float64)
        raw_p /= float(np.sum(raw_p))
        return [(act, float(p)) for act, p in zip(top_actions, raw_p)]

    def _select_action_mix_mcts(
        self,
        env: CaroEnv,
        candidates: list[int],
        prior_probs: np.ndarray,
        start_time: float,
    ) -> int:
        """
        Cây tìm kiếm PUCT-MCTS chuẩn AlphaGo nhiều tầng định hướng bởi mạng CNN Policy:
          - Selection: `PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_s + 1) / (1 + N_a)`
            kèm First Play Urgency (FPU), chỉ thăm sâu các nhánh có prior cao từ mạng CNN.
          - Expansion (Lazy): mở rộng top-K ứng viên theo `predict_probs` khi nút được thăm lại.
          - Leaf Evaluation: đánh giá trực tiếp nút lá bằng `tanh(evaluate_board(..., consider_turn=True) / 12000)`
            (không nhiễu rollout ngẫu nhiên, tốc độ cao).
          - Backpropagation: lan truyền ngược trên toàn đường đi theo thang thưởng `[0, 1]`.
        """
        if len(candidates) == 1:
            self.last_search_count = 1
            return int(candidates[0])

        root = _PUCTNode(
            parent=None,
            action_from_parent=None,
            player_just_moved=-env.current_player,
            prior=1.0,
        )
        root_actions_priors = self._build_puct_priors(
            env, candidates=candidates, prior_probs=prior_probs, max_branch=12
        )
        root.expand(root_actions_priors, next_player=env.current_player)

        c_puct = max(1.0, float(self.exploration_c))
        inner_branch = 8
        rollouts_done = 0

        for rollout_idx in range(self.num_rollouts):
            if (
                self.time_limit_sec is not None
                and rollout_idx > 0
                and (time.perf_counter() - start_time) >= self.time_limit_sec
            ):
                break

            node = root
            sim_env = env.clone()

            # 1. Selection: đi sâu theo PUCT cho tới nút chưa mở rộng hoặc nút kết thúc
            while node.is_expanded and node.children and not sim_env.is_terminal():
                node = node.select_child_puct(c_puct)
                sim_env.step(node.action_from_parent)
                if node.visits == 0:
                    break

            # 2. Lazy Expansion: nếu nút này đã từng thăm (>= 1 lần) và chưa mở rộng -> mở rộng bằng CNN prior
            if node.visits > 0 and not node.is_expanded and not sim_env.is_terminal():
                sub_cands = (
                    get_candidate_actions(sim_env, radius=self.candidate_radius)
                    if self.candidate_radius is not None
                    else sim_env.legal_actions()
                )
                if not sub_cands:
                    sub_cands = sim_env.legal_actions()
                sub_priors = self._build_puct_priors(
                    sim_env, candidates=sub_cands, prior_probs=None, max_branch=inner_branch
                )
                if sub_priors:
                    node.expand(sub_priors, next_player=sim_env.current_player)
                    node = node.select_child_puct(c_puct)
                    sim_env.step(node.action_from_parent)

            # 3. Leaf Evaluation (chính xác và không nhiễu rollout ngẫu nhiên)
            if sim_env.is_terminal():
                outcome_p1 = float(sim_env.result(perspective=1))
            else:
                leaf_score = evaluate_board(sim_env, perspective=1, consider_turn=True)
                outcome_p1 = float(math.tanh(leaf_score / 12000.0))

            # 4. Backpropagation (thang thưởng [0, 1])
            reward_p1 = 0.5 * (outcome_p1 + 1.0)
            reward_p2 = 1.0 - reward_p1
            curr: _PUCTNode | None = node
            while curr is not None:
                curr.visits += 1
                curr.wins += reward_p1 if curr.player_just_moved == 1 else reward_p2
                curr = curr.parent

            rollouts_done += 1

        self.last_search_count = rollouts_done
        best_child = max(
            root.children.values(),
            key=lambda c: (c.visits, (c.wins / c.visits) if c.visits > 0 else 0.0, c.prior),
        )
        return int(best_child.action_from_parent)

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

    def _normalize_torch_inputs(self, X: np.ndarray) -> np.ndarray:
        """Chuẩn hóa mảng đầu vào về dạng `(N, in_channels, H, W)` float32 cho PyTorch ResNet."""
        arr = np.asarray(X, dtype=np.float32)
        n = self.board_size
        if arr.ndim == 4 and arr.shape[1] == self.in_channels and arr.shape[2] == n and arr.shape[3] == n:
            return arr
        if arr.ndim == 4 and arr.shape[1] == n and arr.shape[2] == n and arr.shape[3] == 1:
            # Tương thích nếu truyền dữ liệu cũ (N, H, W, 1) gồm +1/-1/0
            raw = arr[:, :, :, 0]
            out = np.zeros((arr.shape[0], self.in_channels, n, n), dtype=np.float32)
            out[:, 0] = (raw > 0.5).astype(np.float32)
            out[:, 1] = (raw < -0.5).astype(np.float32)
            out[:, 2] = (np.abs(raw) < 0.5).astype(np.float32)
            return out
        if arr.ndim == 3 and arr.shape[1] == n and arr.shape[2] == n:
            out = np.zeros((arr.shape[0], self.in_channels, n, n), dtype=np.float32)
            out[:, 0] = (arr > 0.5).astype(np.float32)
            out[:, 1] = (arr < -0.5).astype(np.float32)
            out[:, 2] = (np.abs(arr) < 0.5).astype(np.float32)
            return out
        raise ValueError(
            f"Kích thước X không tương thích với PyTorch ResNet (in_channels={self.in_channels}, H=W={n}): {arr.shape}"
        )

    @staticmethod
    def _normalize_class_targets(y: np.ndarray, num_classes: int) -> np.ndarray:
        """Chuẩn hóa nhãn mục tiêu về mảng chỉ số lớp nguyên 1 chiều `(N,)`."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            return y_arr.astype(np.int64)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            return y_arr.reshape(-1).astype(np.int64)
        if y_arr.ndim == 2 and y_arr.shape[1] == num_classes:
            return np.argmax(y_arr, axis=1).astype(np.int64)
        raise ValueError(f"Kích thước nhãn y không hợp lệ: {y_arr.shape}.")

    def _evaluate_torch_arrays(
        self,
        X_nchw: np.ndarray,
        y_idx: np.ndarray,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Đánh giá loss, top-1, top-3 và top-5 accuracy trên tập `(X_nchw, y_idx)`."""
        assert self.torch_model is not None
        self.torch_model.eval()
        n_samples = int(X_nchw.shape[0])
        if n_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0}

        total_loss = 0.0
        top1 = 0
        top3 = 0
        top5 = 0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb = torch.from_numpy(X_nchw[start:end]).to(self.device)
                yb = torch.from_numpy(y_idx[start:end]).to(self.device)
                logits = self.torch_model(xb)
                total_loss += float(criterion(logits, yb).item())
                k_max = min(5, logits.shape[1])
                topk_idx = torch.topk(logits, k=k_max, dim=1).indices
                yb_col = yb.unsqueeze(1)
                top1 += int((topk_idx[:, :1] == yb_col).any(dim=1).sum().item())
                top3 += int((topk_idx[:, : min(3, k_max)] == yb_col).any(dim=1).sum().item())
                top5 += int((topk_idx[:, :k_max] == yb_col).any(dim=1).sum().item())

        return {
            "loss": total_loss / n_samples,
            "accuracy": top1 / n_samples,
            "top3_accuracy": top3 / n_samples,
            "top5_accuracy": top5 / n_samples,
        }

    def evaluate_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Đánh giá chất lượng dự đoán Policy Network trên tập dữ liệu độc lập (ví dụ tập Test)."""
        y_idx = self._normalize_class_targets(y, self.num_actions)
        if self.backend == "pytorch" and self.torch_model is not None:
            X_nchw = self._normalize_torch_inputs(X)
            return self._evaluate_torch_arrays(X_nchw, y_idx, batch_size=batch_size)

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 3:
            X_arr = X_arr.reshape(-1, self.board_size, self.board_size, 1)
        probs = self.numpy_net.predict(X_arr, batch_size=batch_size)
        n_samples = len(y_idx)
        if n_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0, "top5_accuracy": 0.0}
        eps = 1e-12
        loss = float(-np.mean(np.log(np.clip(probs[np.arange(n_samples), y_idx], eps, 1.0))))
        ranked = np.argsort(probs, axis=1)[:, ::-1]
        top1 = float(np.mean(ranked[:, 0] == y_idx))
        top3 = float(np.mean(np.any(ranked[:, :3] == y_idx[:, None], axis=1)))
        top5 = float(np.mean(np.any(ranked[:, :5] == y_idx[:, None], axis=1)))
        return {"loss": loss, "accuracy": top1, "top3_accuracy": top3, "top5_accuracy": top5}

    def evaluate_tactical_suite(self) -> dict[str, Any]:
        """
        Đánh giá năng lực chiến thuật THUẦN của mạng CNN (tắt `use_tactical_shortcut`) trên
        bộ thế cờ chuẩn: Thắng-1 (ngang/dọc/chéo/đứt quãng), Chặn-1 (ngang/dọc/chéo/đứt quãng),
        Tạo 4 mở từ 3 mở, và Chặn 3 mở của đối thủ.
        """
        cases: list[tuple[str, list[int], set[int]]] = [
            # Win-in-1 (X to move)
            ("win1_horizontal", [56, 11, 57, 12, 58, 13, 59, 14], {55, 60}),
            ("win1_vertical", [27, 22, 38, 23, 49, 24, 60, 25], {16, 71}),
            ("win1_diagonal", [24, 10, 36, 21, 48, 32, 60, 43], {12, 72}),
            ("win1_gapped", [56, 11, 57, 12, 59, 13, 60, 14], {58}),
            # Block-in-1 (O to move after X's 4th stone)
            ("block1_horizontal", [56, 11, 57, 12, 58, 13, 59], {55, 60}),
            ("block1_vertical", [27, 22, 38, 23, 49, 24, 60], {16, 71}),
            ("block1_diagonal", [24, 10, 36, 21, 48, 32, 60], {12, 72}),
            ("block1_gapped", [56, 11, 57, 12, 59, 13, 60], {58}),
            # Create open-4 from open-3 (X to move)
            ("open3_attack_h", [59, 22, 60, 23, 61, 34], {58, 62}),
            ("open3_attack_d", [48, 22, 60, 23, 72, 34], {36, 84}),
            # Block opponent open-3 (O to move)
            ("open3_block_h", [59, 22, 60, 23, 61], {58, 62}),
            ("open3_block_d", [48, 22, 60, 23, 72], {36, 84}),
        ]

        details: list[dict[str, Any]] = []
        passed = 0
        prob_sum = 0.0
        for name, moves, expected_set in cases:
            env = CaroEnv(board_size=self.board_size, win_length=5)
            for mv in moves:
                env.step(mv)
            probs = self.predict_probs(env, restrict_to_candidates=False)
            top_act = int(np.argmax(probs))
            target_prob = float(sum(probs[a] for a in expected_set))
            ok = top_act in expected_set
            if ok:
                passed += 1
            prob_sum += target_prob
            details.append(
                {
                    "case": name,
                    "top_action": top_act,
                    "expected": sorted(expected_set),
                    "passed": ok,
                    "target_prob": round(target_prob, 4),
                    "top_prob": round(float(probs[top_act]), 4),
                }
            )

        return {
            "passed": passed,
            "total": len(cases),
            "pass_rate": passed / len(cases),
            "mean_target_prob": prob_sum / len(cases),
            "details": details,
        }

    def _train_pytorch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        validation_split: float = 0.1,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        learning_rate: float | None = None,
        label_smoothing: float = 0.02,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """Huấn luyện Residual Policy CNN bằng PyTorch với AdamW + CosineAnnealingLR + Best-Val restore."""
        assert self.torch_model is not None
        X_all = self._normalize_torch_inputs(X)
        y_all = self._normalize_class_targets(y, self.num_actions)

        if validation_data is not None:
            X_train, y_train = X_all, y_all
            X_val = self._normalize_torch_inputs(validation_data[0])
            y_val = self._normalize_class_targets(validation_data[1], self.num_actions)
        elif 0.0 < validation_split < 1.0 and X_all.shape[0] >= 10:
            n_val = max(1, int(round(X_all.shape[0] * validation_split)))
            perm = self.rng.permutation(X_all.shape[0])
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            X_train, y_train = X_all[tr_idx], y_all[tr_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]
        else:
            X_train, y_train = X_all, y_all
            X_val, y_val = None, None

        lr = float(learning_rate) if learning_rate is not None else self.learning_rate
        optimizer = torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
            eta_min=lr * 0.05,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "top3_accuracy": [],
        }
        if X_val is not None and len(y_val) > 0:
            history["val_loss"] = []
            history["val_accuracy"] = []
            history["val_top3_accuracy"] = []

        best_val_loss = math.inf
        best_state: dict[str, Any] | None = None
        n_train = int(X_train.shape[0])

        for epoch in range(1, int(epochs) + 1):
            self.torch_model.train()
            perm = self.rng.permutation(n_train)
            tr_loss_sum = 0.0
            tr_top1 = 0
            tr_top3 = 0

            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                xb = torch.from_numpy(X_train[idx]).to(self.device)
                yb = torch.from_numpy(y_train[idx]).to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.torch_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.torch_model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = int(yb.shape[0])
                tr_loss_sum += float(loss.item()) * bs
                with torch.no_grad():
                    topk = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
                    yb_col = yb.unsqueeze(1)
                    tr_top1 += int((topk[:, :1] == yb_col).any(dim=1).sum().item())
                    tr_top3 += int((topk == yb_col).any(dim=1).sum().item())

            scheduler.step()
            ep_loss = tr_loss_sum / max(1, n_train)
            ep_acc = tr_top1 / max(1, n_train)
            ep_top3 = tr_top3 / max(1, n_train)
            history["loss"].append(ep_loss)
            history["accuracy"].append(ep_acc)
            history["top3_accuracy"].append(ep_top3)

            if X_val is not None and y_val is not None and len(y_val) > 0:
                val_metrics = self._evaluate_torch_arrays(X_val, y_val, batch_size=max(256, batch_size))
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_top3_accuracy"].append(val_metrics["top3_accuracy"])
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_state = copy.deepcopy(self.torch_model.state_dict())
                if verbose:
                    print(
                        f"Epoch {epoch:2d}/{epochs} | "
                        f"loss={ep_loss:.4f} acc={ep_acc:.4f} top3={ep_top3:.4f} | "
                        f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                        f"val_top3={val_metrics['top3_accuracy']:.4f}"
                    )
            elif verbose:
                print(f"Epoch {epoch:2d}/{epochs} | loss={ep_loss:.4f} acc={ep_acc:.4f} top3={ep_top3:.4f}")

        if best_state is not None:
            self.torch_model.load_state_dict(best_state)
        self.torch_model.eval()
        self.training_history = history
        return history

    def train_from_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        validation_split: float = 0.1,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        learning_rate: float | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Huấn luyện mạng CNN Policy Network trên tập dữ liệu `(X, y)`.
        Hỗ trợ đồng thời backend PyTorch (ResNet đa kênh 5), Keras và NumPy.
        """
        if self.backend == "pytorch" and self.torch_model is not None:
            return self._train_pytorch(
                X=X,
                y=y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                learning_rate=learning_rate,
                verbose=verbose,
            )

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
            keras_val = None
            if validation_data is not None:
                xv = np.asarray(validation_data[0], dtype=np.float32).reshape(-1, self.board_size, self.board_size, 1)
                yv_raw = np.asarray(validation_data[1])
                yv = (
                    to_one_hot(yv_raw.reshape(-1), num_classes=self.num_actions)
                    if yv_raw.ndim == 1
                    else yv_raw.astype(np.float32)
                )
                keras_val = (xv, yv)
            fit_res = self.keras_model.fit(
                X_arr,
                y_onehot,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.0 if keras_val is not None else (validation_split if X_arr.shape[0] >= 10 else 0.0),
                validation_data=keras_val,
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
            validation_data=validation_data,
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
        validation_split: float = 0.12,
        learning_rate: float | None = None,
        save_path: str | Path | None = None,
        verbose: int = 1,
    ) -> dict[str, list[float]]:
        """
        Quy trình huấn luyện khép kín (End-to-End Pipeline) tách biệt tập Train và Validation
        theo cấp độ ván đấu (Game-Level Split) trước khi tăng cường đối xứng D4.
        """
        if 0.0 < validation_split < 1.0 and num_games >= 5:
            val_games = max(1, int(round(num_games * validation_split)))
            train_games = max(1, num_games - val_games)
            X_train, y_train, X_val, y_val = generate_expert_train_val_datasets(
                num_train_games=train_games,
                num_val_games=val_games,
                board_size=self.board_size,
                win_length=5 if self.board_size >= 5 else self.board_size,
                random_prob=random_prob,
                candidate_radius=candidate_radius,
                augment_train_symmetry=augment_symmetry,
                augment_val_symmetry=False,
                seed=self.seed if self.seed is not None else 42,
            )
            val_data: tuple[np.ndarray, np.ndarray] | None = (X_val, y_val)
        else:
            X_train, y_train = generate_expert_dataset(
                num_games=num_games,
                board_size=self.board_size,
                win_length=5 if self.board_size >= 5 else self.board_size,
                random_prob=random_prob,
                candidate_radius=candidate_radius,
                augment_symmetry=augment_symmetry,
                seed=self.seed,
            )
            val_data = None

        history = self.train_from_dataset(
            X=X_train,
            y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0,
            validation_data=val_data,
            learning_rate=learning_rate,
            verbose=verbose,
        )
        if save_path is not None:
            self.save_model(save_path)
        return history

    def save_model(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Lưu mô hình ra đĩa (`.pt` cho PyTorch ResNet Cấp 5, `.npz`/`.keras` cho NumPy/Keras)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "pytorch" and self.torch_model is not None:
            if out_path.suffix.lower() not in (".pt", ".pth"):
                out_path = out_path.with_suffix(".pt")
            payload: dict[str, Any] = {
                "format": "caro_cnn_policy_resnet_v1",
                "model_type": self.model_type,
                "board_size": self.board_size,
                "in_channels": self.in_channels,
                "res_channels": self.res_channels,
                "num_res_blocks": self.num_res_blocks,
                "state_dict": self.torch_model.state_dict(),
                "training_history": self.training_history,
                "metadata": metadata or {},
            }
            torch.save(payload, out_path)
            return out_path

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
        """Nạp trọng số mô hình từ file `.pt`/`.pth` (PyTorch), `.npz` (NumPy), hoặc `.keras`/`.h5`."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình tại: {in_path}")

        if in_path.suffix.lower() in (".pt", ".pth"):
            if not HAS_PYTORCH:
                raise RuntimeError("Cần cài đặt PyTorch để nạp checkpoint .pt/.pth.")
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(in_path, map_location=self.device, weights_only=False)
            self.backend = "pytorch"
            self.model_type = str(ckpt.get("model_type", "resnet"))
            self.board_size = int(ckpt.get("board_size", self.board_size))
            self.num_actions = self.board_size * self.board_size
            self.in_channels = int(ckpt.get("in_channels", self.in_channels))
            self.res_channels = int(ckpt.get("res_channels", self.res_channels))
            self.num_res_blocks = int(ckpt.get("num_res_blocks", self.num_res_blocks))
            self.torch_model = TorchPolicyResNet(
                board_size=self.board_size,
                in_channels=self.in_channels,
                channels=self.res_channels,
                num_res_blocks=self.num_res_blocks,
            ).to(self.device)
            self.torch_model.load_state_dict(ckpt["state_dict"])
            self.torch_model.eval()
            if isinstance(ckpt.get("training_history"), dict):
                self.training_history = ckpt["training_history"]
            return

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

        if self.numpy_net is None:
            self.numpy_net = NumpyCNNPolicyNetwork(
                board_size=self.board_size,
                model_type="strong" if self.model_type == "resnet" else self.model_type,
                conv_filters=self.conv_filters,
                kernel_size=self.kernel_size,
                dense_units=self.dense_units,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                seed=self.seed,
            )
            self.backend = "numpy"

        with np.load(in_path, allow_pickle=False) as data:
            self.numpy_net.set_weights_from_dict(data)
            self.kernel_size = self.numpy_net.kernel_size
            self.conv_filters = self.numpy_net.conv_filters

        if self.keras_model is not None:
            self._sync_numpy_to_keras()


def main() -> None:
    """Giao diện dòng lệnh (CLI) hỗ trợ sinh dữ liệu, huấn luyện và kiểm thử nhanh CNNAgent."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá CNN Policy Network Agent (Cấp 5) cho Caro 11x11."
    )
    parser.add_argument("--model-type", choices=["fast", "strong", "resnet"], default="resnet", help="Kiến trúc mạng ('fast', 'strong' hoặc 'resnet').")
    parser.add_argument("--backend", choices=["auto", "pytorch", "keras", "numpy"], default="auto", help="Backend tính toán ('auto', 'pytorch', 'keras', 'numpy').")
    parser.add_argument("--games", type=int, default=200, help="Số ván đấu tự chơi để sinh dữ liệu.")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch huấn luyện.")
    parser.add_argument("--batch-size", type=int, default=128, help="Kích thước mini-batch.")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Tốc độ học (learning rate) cho Adam/AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để tái lập thực nghiệm.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Đường dẫn lưu trọng số mô hình (mặc định: checkpoints/05_cnn_policy_best.pt hoặc .npz).",
    )
    args = parser.parse_args()

    default_ckpt = (
        DEFAULT_CHECKPOINT_DIR / "05_cnn_policy_best.pt"
        if args.model_type == "resnet" or args.backend == "pytorch"
        else DEFAULT_CHECKPOINT_DIR / f"cnn_{args.model_type}_11x11.npz"
    )
    save_target = Path(args.save_path) if args.save_path else default_ckpt

    print(f"[1/3] Khởi tạo CNNAgent (model_type={args.model_type}, backend={args.backend})...")
    agent = CNNAgent(
        name=f"Level5_CNN_{args.model_type.capitalize()}",
        model_type=args.model_type,
        backend=args.backend,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print(f"[2/3] Sinh dữ liệu từ {args.games} ván đấu (Game-level Train/Val split) và huấn luyện ({args.epochs} epochs)...")
    history = agent.train_self_play(
        num_games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=save_target,
        verbose=1,
    )
    final_acc = history["accuracy"][-1] if history.get("accuracy") else 0.0
    val_acc = history["val_accuracy"][-1] if history.get("val_accuracy") else 0.0
    print(f"[3/3] Hoàn tất! Train Acc: {final_acc:.4f} | Val Acc (Unseen Games): {val_acc:.4f}. Đã lưu tại: {save_target}")


if __name__ == "__main__":
    main()
