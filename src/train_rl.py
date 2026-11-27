from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.random_agent import RandomAgent
from agents.rl_agent import DuelingCaroDQN, RLAgent, encode_board_state, resolve_device
from agents.rule_based_agent import RuleBasedAgent
from caro_env import CaroEnv
from heuristic import evaluate_board, evaluate_cell_threat, get_candidate_actions


# ---------------------------------------------------------------------------
# D4 Symmetry Helper (8 phép quay/lật trên tensor (C, H, W) và action index)
# ---------------------------------------------------------------------------
def _apply_d4_transform(
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    k_rot: int,
    do_flip: bool,
    board_size: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    if k_rot == 0 and not do_flip:
        return state, action, next_state

    s_out = np.rot90(state, k=k_rot, axes=(1, 2)) if k_rot else state
    ns_out = np.rot90(next_state, k=k_rot, axes=(1, 2)) if k_rot else next_state

    r, c = divmod(action, board_size)
    # Tương ứng chính xác với np.rot90(arr, k=k_rot) (quay ngược chiều kim đồng hồ)
    for _ in range(k_rot):
        r, c = board_size - 1 - c, r

    if do_flip:
        s_out = s_out[:, :, ::-1]
        ns_out = ns_out[:, :, ::-1]
        c = board_size - 1 - c

    return np.ascontiguousarray(s_out), r * board_size + c, np.ascontiguousarray(ns_out)


# ---------------------------------------------------------------------------
# Replay Buffer gọn nhẹ + D4 Augmentation on-the-fly + Terminal Priority
# ---------------------------------------------------------------------------
class D4ReplayBuffer:
    """Replay buffer dạng mảng liên tục, hỗ trợ D4 augmentation on-the-fly và ưu tiên mẫu kết thúc."""

    def __init__(
        self,
        capacity: int,
        in_channels: int,
        board_size: int,
        seed: int = 42,
    ) -> None:
        self.capacity = int(capacity)
        self.in_channels = int(in_channels)
        self.board_size = int(board_size)
        self.rng = np.random.default_rng(seed)

        self.states = np.zeros((capacity, in_channels, board_size, board_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, in_channels, board_size, board_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.ptr = 0
        self.size = 0
        self.terminal_mask = np.zeros((capacity,), dtype=np.bool_)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)
        self.terminal_mask[idx] = bool(done)

        self.ptr = (idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(
        self,
        batch_size: int,
        terminal_ratio: float = 0.2,
        use_d4_aug: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        term_indices = np.flatnonzero(self.terminal_mask[: self.size])
        n_term = 0
        if len(term_indices) > 0 and terminal_ratio > 0.0:
            n_term = min(len(term_indices), int(round(batch_size * terminal_ratio)))
        n_reg = batch_size - n_term

        reg_idx = self.rng.integers(0, self.size, size=n_reg)
        if n_term > 0:
            t_idx = term_indices[self.rng.integers(0, len(term_indices), size=n_term)]
            indices = np.concatenate([reg_idx, t_idx])
        else:
            indices = reg_idx

        s_batch = self.states[indices].copy()
        ns_batch = self.next_states[indices].copy()
        a_batch = self.actions[indices].copy()
        r_batch = self.rewards[indices].copy()
        d_batch = self.dones[indices].copy()

        if use_d4_aug:
            k_rots = self.rng.integers(0, 4, size=batch_size)
            flips = self.rng.integers(0, 2, size=batch_size)
            for i in range(batch_size):
                k = int(k_rots[i])
                fl = bool(flips[i])
                if k != 0 or fl:
                    s_t, a_t, ns_t = _apply_d4_transform(
                        s_batch[i],
                        int(a_batch[i]),
                        ns_batch[i],
                        k,
                        fl,
                        self.board_size,
                    )
                    s_batch[i] = s_t
                    a_batch[i] = a_t
                    ns_batch[i] = ns_t

        return s_batch, a_batch, r_batch, ns_batch, d_batch


# ---------------------------------------------------------------------------
# Đối thủ phụ trợ cho Curriculum (nhanh trên CPU, có độ khó tăng dần mượt)
# ---------------------------------------------------------------------------
class _FastTacticalOpponent:
    """
    Đối thủ chiến thuật tốc độ cao (< 0.3 ms/nước) cho Stage 2 curriculum:
      - Luôn thắng ngay (5 quân) và chặn thắng ngay (5 quân) với xác suất `1 - noise_prob`.
      - Ưu tiên tạo/chặn chuỗi 4 mở hoặc đòn mạnh, còn lại chọn nước ứng viên có điểm cao.
    Giúp RL gặp đầy đủ tình huống công/thủ thực chiến mà vẫn giữ được tỷ lệ thắng > 0 để học.
    """

    def __init__(self, noise_prob: float = 0.35, candidate_radius: int = 2, seed: int = 42) -> None:
        self.noise_prob = float(noise_prob)
        self.candidate_radius = int(candidate_radius)
        self.rng = random.Random(seed)

    def select_action(self, env: CaroEnv) -> int:
        cands = get_candidate_actions(env, radius=self.candidate_radius)
        if not cands:
            cands = env.legal_actions()
        if len(cands) == 1:
            return int(cands[0])

        if self.rng.random() < self.noise_prob:
            return int(self.rng.choice(cands))

        p = env.current_player
        opp = -p
        best_act = cands[0]
        best_score = -1e18

        for a in cands:
            mt = evaluate_cell_threat(env, a, p)
            if mt["is_win_5"]:
                return int(a)
            ot = evaluate_cell_threat(env, a, opp)
            if ot["is_win_5"]:
                score = 5_000_000.0 + float(ot["score"])
            elif mt["open_4"] > 0:
                score = 2_000_000.0 + float(mt["score"])
            elif ot["open_4"] > 0:
                score = 1_000_000.0 + float(ot["score"])
            else:
                score = float(mt["score"]) + 0.95 * float(ot["score"]) + self.rng.random() * 5.0
            if score > best_score:
                best_score = score
                best_act = a

        return int(best_act)


# ---------------------------------------------------------------------------
# Hàm tiện ích chọn hành động epsilon-greedy, đánh giá và cập nhật Double-DQN
# ---------------------------------------------------------------------------
def _encode_env(env: CaroEnv, in_channels: int) -> np.ndarray:
    last_action = env.history[-1] if env.history else None
    return encode_board_state(env.board, env.current_player, last_action=last_action, in_channels=in_channels)


def _select_epsilon_greedy(
    model: DuelingCaroDQN,
    env: CaroEnv,
    in_channels: int,
    epsilon: float,
    device: torch.device,
    py_rng: random.Random,
    candidate_radius: int = 2,
    full_board_explore_prob: float = 0.1,
) -> tuple[int, float]:
    legal = env.legal_actions()
    if len(legal) == 1:
        return int(legal[0]), 0.0

    if py_rng.random() < epsilon:
        if py_rng.random() < full_board_explore_prob:
            return int(py_rng.choice(legal)), 0.0
        cands = get_candidate_actions(env, radius=candidate_radius)
        return int(py_rng.choice(cands if cands else legal)), 0.0

    state = _encode_env(env, in_channels)
    with torch.no_grad():
        q_vals = model(torch.from_numpy(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    best_act = max(legal, key=lambda a: float(q_vals[a]))
    return int(best_act), float(q_vals[best_act])


def _select_greedy(
    model: DuelingCaroDQN,
    env: CaroEnv,
    in_channels: int,
    device: torch.device,
) -> int:
    legal = env.legal_actions()
    if len(legal) == 1:
        return int(legal[0])
    state = _encode_env(env, in_channels)
    with torch.no_grad():
        q_vals = model(torch.from_numpy(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    return int(max(legal, key=lambda a: float(q_vals[a])))


def _potential_phi(env: CaroEnv, perspective: int, weight: float) -> float:
    """Hàm thế năng Phi(s) theo định lý Ng-Harada-Russell (1999): bảo toàn chính sách tối ưu."""
    if weight <= 0.0 or env.is_terminal():
        return 0.0
    score = evaluate_board(env, perspective=perspective, consider_turn=True)
    return float(weight * np.tanh(score / 15000.0))


def _dqfd_pretrain_step(
    online_net: DuelingCaroDQN,
    target_net: DuelingCaroDQN,
    optimizer: torch.optim.Optimizer,
    demo_replay: D4ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    margin: float,
    margin_weight: float,
    grad_clip_norm: float,
    device: torch.device,
) -> tuple[float, float]:
    """
    Một bước cập nhật DQfD (Hester et al., AAAI 2018):
      Loss = SmoothL1(Q(s, a_E), y_TD) + margin_weight * (max_{a in legal} [Q(s, a) + l(a_E, a)] - Q(s, a_E))
    Giúp mạng Dueling Double-DQN vượt qua nút thắt khám phá thưa 1/121 ở giai đoạn khởi đầu
    mà vẫn giữ nguyên 100% suy luận thuần `argmax Q(s, a)` khi thi đấu.
    """
    s_b, a_b, r_b, ns_b, d_b = demo_replay.sample(
        batch_size=batch_size,
        terminal_ratio=0.25,
        use_d4_aug=True,
    )
    states_t = torch.from_numpy(s_b).to(device)
    next_states_t = torch.from_numpy(ns_b).to(device)
    actions_t = torch.from_numpy(a_b).to(device)
    rewards_t = torch.from_numpy(r_b).to(device)
    dones_t = torch.from_numpy(d_b.astype(np.float32)).to(device)

    online_net.train()
    q_all = online_net(states_t)
    q_taken = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_legal_mask = next_states_t[:, 2].flatten(start_dim=1) > 0.5
        has_next_legal = next_legal_mask.any(dim=1)
        online_next_q = online_net(next_states_t).masked_fill(~next_legal_mask, -1e9)
        next_best_actions = online_next_q.argmax(dim=1, keepdim=True)
        target_next_q = target_net(next_states_t).gather(1, next_best_actions).squeeze(1)
        target_next_q = torch.where(has_next_legal, target_next_q, torch.zeros_like(target_next_q))
        td_target = (rewards_t + (1.0 - dones_t) * gamma * target_next_q).clamp(-1.0, 1.0)

    td_loss = nn.functional.smooth_l1_loss(q_taken, td_target)

    # Boltzmann-DQfD Masked Margin Loss: đẩy Q(s, a_E) lên và đồng thời ép toàn bộ
    # các hành động hợp lệ khác xuống (khắc phục gradient thưa 1/121 của hinge max đơn).
    legal_mask = states_t[:, 2].flatten(start_dim=1) > 0.5
    q_masked = q_all.masked_fill(~legal_mask, -1e9)
    boltzmann_temp = max(0.05, float(margin))
    margin_loss = nn.functional.cross_entropy(q_masked / boltzmann_temp, actions_t)

    total_loss = td_loss + margin_weight * margin_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=grad_clip_norm)
    optimizer.step()

    with torch.no_grad():
        for p_target, p_online in zip(target_net.parameters(), online_net.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    return float(total_loss.item()), float(q_taken.detach().abs().mean().item())


def _collect_dqfd_demo_buffer(
    num_games: int,
    in_channels: int,
    board_size: int,
    win_length: int,
    gamma: float,
    step_penalty: float,
    potential_weight: float,
    seed: int,
) -> D4ReplayBuffer:
    """Thu thập tập quỹ đạo mẫu (demonstrations) cho DQfD từ các ván đấu chiến thuật."""
    demo_buf = D4ReplayBuffer(capacity=max(8000, num_games * 40), in_channels=in_channels, board_size=board_size, seed=seed)
    env = CaroEnv(board_size=board_size, win_length=win_length)
    py_rng = random.Random(seed + 1337)
    center = board_size // 2
    opening_zone = [
        r * board_size + c
        for r in range(max(0, center - 2), min(board_size, center + 3))
        for c in range(max(0, center - 2), min(board_size, center + 3))
    ]
    zero_state = np.zeros((in_channels, board_size, board_size), dtype=np.float32)

    expert = RuleBasedAgent(name="DQfD_Teacher", root_width=6, seed=seed, tie_break_eps=1e-6)

    for g in range(num_games):
        env.reset()
        expert_side = 1 if (g % 2 == 0) else -1
        # Đối thủ đa dạng: 50% RuleBased, 50% FastTactical có nhiễu để tạo đủ thế công & thủ
        if g % 2 == 0:
            opp: Any = RuleBasedAgent(name="DQfD_Opp_RB", root_width=5, seed=seed + 1000 + g, tie_break_eps=1e-6)
            record_both = True
        else:
            opp = _FastTacticalOpponent(noise_prob=0.25, candidate_radius=2, seed=seed + 2000 + g)
            record_both = False

        # Mở màn ngẫu nhiên 1..3 nước quanh trung tâm
        k_open = py_rng.randint(1, 3)
        for _ in range(k_open):
            if env.is_terminal():
                break
            open_legal = [a for a in opening_zone if env.board[a // board_size, a % board_size] == 0]
            mv = py_rng.choice(open_legal) if open_legal else py_rng.choice(env.legal_actions())
            env.step(int(mv))

        pending: dict[int, tuple[np.ndarray, int, float] | None] = {1: None, -1: None}
        while not env.is_terminal():
            cur_p = env.current_player
            record_side = record_both or (cur_p == expert_side)

            if record_side and pending[cur_p] is not None:
                prev_s, prev_a, prev_phi = pending[cur_p]
                curr_s = _encode_env(env, in_channels)
                curr_phi = _potential_phi(env, cur_p, potential_weight)
                r_step = step_penalty + (gamma * curr_phi - prev_phi)
                demo_buf.push(prev_s, prev_a, r_step, curr_s, False)
                pending[cur_p] = None

            state_now = _encode_env(env, in_channels) if record_side else None
            phi_now = _potential_phi(env, cur_p, potential_weight) if record_side else 0.0
            action = int(expert.select_action(env)) if cur_p == expert_side else int(opp.select_action(env))
            env.step(action)

            if env.is_terminal():
                winner = env.winner
                if record_side and state_now is not None:
                    r_end = (1.0 if winner == cur_p else (0.0 if winner == 0 else -1.0)) - phi_now
                    demo_buf.push(state_now, action, r_end, zero_state, True)
                opp_p = -cur_p
                if (record_both or opp_p == expert_side) and pending[opp_p] is not None:
                    prev_s, prev_a, prev_phi = pending[opp_p]
                    r_opp = (1.0 if winner == opp_p else (0.0 if winner == 0 else -1.0)) - prev_phi
                    demo_buf.push(prev_s, prev_a, r_opp, zero_state, True)
                    pending[opp_p] = None
            elif record_side and state_now is not None:
                pending[cur_p] = (state_now, action, phi_now)

    return demo_buf


def _dqn_train_step(
    online_net: DuelingCaroDQN,
    target_net: DuelingCaroDQN,
    optimizer: torch.optim.Optimizer,
    replay: D4ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    terminal_ratio: float,
    use_d4_aug: bool,
    grad_clip_norm: float,
    device: torch.device,
) -> tuple[float, float]:
    s_b, a_b, r_b, ns_b, d_b = replay.sample(
        batch_size=batch_size,
        terminal_ratio=terminal_ratio,
        use_d4_aug=use_d4_aug,
    )

    states_t = torch.from_numpy(s_b).to(device)
    next_states_t = torch.from_numpy(ns_b).to(device)
    actions_t = torch.from_numpy(a_b).to(device)
    rewards_t = torch.from_numpy(r_b).to(device)
    dones_t = torch.from_numpy(d_b.astype(np.float32)).to(device)

    online_net.train()
    q_all = online_net(states_t)
    q_taken = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double-DQN với Action Masking chuẩn theo kênh 2 (ô trống) trên trạng thái kế tiếp s'
        next_legal_mask = next_states_t[:, 2].flatten(start_dim=1) > 0.5
        has_next_legal = next_legal_mask.any(dim=1)

        online_next_q = online_net(next_states_t)
        masked_online_next_q = online_next_q.masked_fill(~next_legal_mask, -1e9)
        next_best_actions = masked_online_next_q.argmax(dim=1, keepdim=True)

        target_next_q_all = target_net(next_states_t)
        target_next_q = target_next_q_all.gather(1, next_best_actions).squeeze(1)
        target_next_q = torch.where(has_next_legal, target_next_q, torch.zeros_like(target_next_q))

        td_target = rewards_t + (1.0 - dones_t) * gamma * target_next_q
        # Kẹp TD target trong [-1, +1] để chống khuếch đại sai số Q-value
        td_target = td_target.clamp(-1.0, 1.0)

    loss = nn.functional.smooth_l1_loss(q_taken, td_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=grad_clip_norm)
    optimizer.step()

    # Polyak Soft Target Update (ổn định hoàn toàn với GroupNorm vì không có running stats)
    with torch.no_grad():
        for p_target, p_online in zip(target_net.parameters(), online_net.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

    return float(loss.item()), float(q_taken.detach().abs().mean().item())


def _evaluate_model(
    model: DuelingCaroDQN,
    in_channels: int,
    device: torch.device,
    make_opponent,
    n_games: int = 20,
    board_size: int = 11,
    win_length: int = 5,
) -> dict[str, float]:
    model.eval()
    wins = draws = losses = 0
    moves_list: list[int] = []
    for g in range(n_games):
        env = CaroEnv(board_size=board_size, win_length=win_length)
        opp = make_opponent(g)
        rl_side = 1 if g % 2 == 0 else -1
        while not env.is_terminal():
            if env.current_player == rl_side:
                act = _select_greedy(model, env, in_channels, device)
            else:
                act = opp.select_action(env)
            env.step(act)
        moves_list.append(len(env.history))
        if env.winner == rl_side:
            wins += 1
        elif env.winner == -rl_side:
            losses += 1
        else:
            draws += 1
    score_pct = (wins + 0.5 * draws) / max(1, n_games) * 100.0
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": score_pct,
        "avg_moves": statistics.mean(moves_list) if moves_list else 0.0,
    }


def _save_rl_checkpoint(
    path: Path,
    model: DuelingCaroDQN,
    metadata: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "caro_dueling_dqn_6a_v2",
        "model_state_dict": model.state_dict(),
        "in_channels": model.in_channels,
        "conv_channels": model.conv_channels,
        "board_size": model.board_size,
        "use_residual": model.use_residual,
        "num_res_blocks": model.num_res_blocks,
        "norm_type": model.norm_type,
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def _save_training_log(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _play_training_episode(
    env: CaroEnv,
    online_net: DuelingCaroDQN,
    opponent: Any,
    is_selfplay: bool,
    rl_side: int,
    in_channels: int,
    epsilon: float,
    gamma: float,
    step_penalty: float,
    potential_weight: float,
    opening_random_plies: int,
    replay: D4ReplayBuffer,
    device: torch.device,
    py_rng: random.Random,
) -> tuple[int, int]:
    """
    Chạy 1 ván huấn luyện và ghi đầy đủ chuyển trạng thái 2 người chơi (zero-sum) vào ReplayBuffer.
    Trong chế độ `is_selfplay=True`, ghi chuyển trạng thái của cả 2 phía để tận dụng gấp đôi dữ liệu.
    Trả về `(winner, num_steps_pushed)`.
    """
    env.reset()
    n = env.board_size
    center = n // 2
    opening_zone = [
        r * n + c
        for r in range(max(0, center - 2), min(n, center + 3))
        for c in range(max(0, center - 2), min(n, center + 3))
    ]

    # Mở màn ngẫu nhiên 0..opening_random_plies bước quanh trung tâm (không ghi nhãn)
    if opening_random_plies > 0 and py_rng.random() < 0.6:
        k_open = py_rng.randint(1, opening_random_plies)
        for _ in range(k_open):
            if env.is_terminal():
                break
            open_legal = [a for a in opening_zone if env.board[a // n, a % n] == 0]
            mv = py_rng.choice(open_legal) if open_legal else py_rng.choice(env.legal_actions())
            env.step(int(mv))

    # Lưu nước đi đang chờ kết quả cho từng bên được ghi nhận (pending transition)
    pending: dict[int, tuple[np.ndarray, int, float] | None] = {1: None, -1: None}
    pushed = 0
    zero_state = np.zeros((in_channels, n, n), dtype=np.float32)

    while not env.is_terminal():
        cur_p = env.current_player
        record_this_side = is_selfplay or (cur_p == rl_side)

        # Nếu bên `cur_p` có nước đang chờ từ lượt trước -> nay đã tới lượt `cur_p` lại (chưa thua)
        if record_this_side and pending[cur_p] is not None:
            prev_s, prev_a, prev_phi = pending[cur_p]
            curr_s = _encode_env(env, in_channels)
            curr_phi = _potential_phi(env, cur_p, potential_weight)
            r_step = step_penalty + (gamma * curr_phi - prev_phi)
            replay.push(prev_s, prev_a, r_step, curr_s, False)
            pushed += 1
            pending[cur_p] = None

        # Chọn hành động cho `cur_p`
        if cur_p == rl_side:
            state_now = _encode_env(env, in_channels)
            phi_now = _potential_phi(env, cur_p, potential_weight) if record_this_side else 0.0
            action, _ = _select_epsilon_greedy(
                online_net, env, in_channels, epsilon, device, py_rng,
                candidate_radius=2, full_board_explore_prob=0.1,
            )
        elif is_selfplay:
            state_now = _encode_env(env, in_channels)
            phi_now = _potential_phi(env, cur_p, potential_weight)
            if isinstance(opponent, DuelingCaroDQN):
                # Đối thủ snapshot tự chơi với một chút epsilon nhỏ để đa dạng hóa nhánh
                action, _ = _select_epsilon_greedy(
                    opponent, env, in_channels, max(0.05, epsilon * 0.5), device, py_rng,
                    candidate_radius=2, full_board_explore_prob=0.05,
                )
            else:
                action = int(opponent.select_action(env))
        else:
            state_now = None
            phi_now = 0.0
            action = int(opponent.select_action(env))

        env.step(action)

        if env.is_terminal():
            winner = env.winner
            # Kết thúc ván: ghi ngay kết quả cho bên vừa đánh `cur_p`
            if record_this_side and state_now is not None:
                r_end = 1.0 if winner == cur_p else (0.0 if winner == 0 else -1.0)
                r_end += -phi_now  # Phi(terminal) = 0
                replay.push(state_now, action, r_end, zero_state, True)
                pushed += 1

            # Đồng thời đóng transition đang chờ của đối thủ `opp_p` (vì nước đi trước của họ dẫn tới thua/hòa)
            opp_p = -cur_p
            if (is_selfplay or opp_p == rl_side) and pending[opp_p] is not None:
                prev_s, prev_a, prev_phi = pending[opp_p]
                r_opp = 1.0 if winner == opp_p else (0.0 if winner == 0 else -1.0)
                r_opp += -prev_phi
                replay.push(prev_s, prev_a, r_opp, zero_state, True)
                pushed += 1
                pending[opp_p] = None
        else:
            if record_this_side and state_now is not None:
                pending[cur_p] = (state_now, action, phi_now)

    return int(env.winner), pushed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Huấn luyện Cấp 6: Dueling Double-DQN nâng cấp (Residual + GroupNorm + Masked Mean + Curriculum Self-Play)."
    )
    parser.add_argument("--stage1-episodes", type=int, default=900, help="Số ván chặng 1 (vs Lv1 Random/Local-Random).")
    parser.add_argument("--stage2-episodes", type=int, default=1200, help="Số ván chặng 2 (vs Tactical Ladder mềm).")
    parser.add_argument("--stage3-episodes", type=int, default=1200, help="Số ván chặng 3 (Self-Play Pool + Anchor).")
    parser.add_argument("--eval-every", type=int, default=300, help="Chu kỳ sát hạch định kỳ (episodes).")
    parser.add_argument("--eval-games", type=int, default=16, help="Số ván mỗi bài sát hạch (50/50 X-O).")
    parser.add_argument("--res-channels", type=int, default=48, help="Số kênh của các khối Residual GroupNorm.")
    parser.add_argument("--num-res-blocks", type=int, default=3, help="Số khối Residual 3x3 (3 blocks -> trường nhìn 15x15).")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước mini-batch Double-DQN.")
    parser.add_argument("--train-freq", type=int, default=4, help="Cập nhật mạng sau mỗi N nước ghi vào buffer.")
    parser.add_argument("--replay-capacity", type=int, default=60000, help="Dung lượng Replay Buffer.")
    parser.add_argument("--gamma", type=float, default=0.98, help="Hệ số chiết khấu gamma.")
    parser.add_argument("--tau", type=float, default=0.005, help="Hệ số Polyak soft target update.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate chặng 1 (tự giảm dần ở chặng 2, 3).")
    parser.add_argument(
        "--shaping-mode",
        choices=["none", "potential"],
        default="potential",
        help="Chế độ reward: 'none' (sparse +1/0/-1 thuần) hoặc 'potential' (Ng-Harada-Russell potential-based).",
    )
    parser.add_argument("--potential-weight", type=float, default=0.15, help="Trọng số Phi(s) khi shaping-mode='potential'.")
    parser.add_argument("--dqfd-demo-games", type=int, default=80, help="Số ván mẫu cho pha khởi động DQfD (0 để tắt -> RL thuần tabula-rasa).")
    parser.add_argument("--dqfd-pretrain-steps", type=int, default=600, help="Số bước cập nhật pre-train DQfD (TD + Large-Margin).")
    parser.add_argument("--dqfd-margin", type=float, default=0.25, help="Biên margin l(a_E, a) của DQfD.")
    parser.add_argument("--dqfd-margin-weight", type=float, default=0.5, help="Trọng số thành phần margin loss trong DQfD.")
    parser.add_argument("--dqfd-online-ratio", type=float, default=0.20, help="Tỷ lệ cập nhật giữ lại từ demo buffer trong pha online RL để chống quên.")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị tính toán ('auto', 'cpu', 'cuda').")
    parser.add_argument("--seed", type=int, default=42, help="Seed tái lập.")
    parser.add_argument(
        "--init-weights",
        type=str,
        default="",
        help="Tùy chọn: đường dẫn checkpoint .pt để nạp trọng số khởi tạo / tiếp tục huấn luyện.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "06_rl_best.pt"),
        help="Đường dẫn lưu checkpoint tốt nhất (.pt).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "tables" / "06_rl_dqn6a_training_log.csv"),
        help="Đường dẫn lưu bảng log quá trình huấn luyện (.csv).",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    py_rng = random.Random(args.seed)

    device = resolve_device(args.device)
    in_channels = 4
    board_size = 11
    win_length = 5

    online_net = DuelingCaroDQN(
        in_channels=in_channels,
        conv_channels=(args.res_channels,),
        board_size=board_size,
        use_residual=True,
        num_res_blocks=args.num_res_blocks,
        norm_type="group",
    ).to(device)
    if args.init_weights:
        init_path = Path(args.init_weights)
        if not init_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint khởi tạo: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        sd = init_ckpt["model_state_dict"] if isinstance(init_ckpt, dict) and "model_state_dict" in init_ckpt else init_ckpt
        online_net.load_state_dict(sd)
        print(f"Đã nạp trọng số khởi tạo từ: {init_path}")
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()

    n_params = sum(p.numel() for p in online_net.parameters())
    print(
        f"=== Level 6 Dueling Double-DQN (ResGN blocks={args.num_res_blocks}, ch={args.res_channels}, "
        f"in={in_channels}, params={n_params:,}) on {device} | shaping={args.shaping_mode} ==="
    )

    replay = D4ReplayBuffer(
        capacity=args.replay_capacity,
        in_channels=in_channels,
        board_size=board_size,
        seed=args.seed,
    )
    env = CaroEnv(board_size=board_size, win_length=win_length)

    pot_weight = args.potential_weight if args.shaping_mode == "potential" else 0.0
    demo_replay: D4ReplayBuffer | None = None
    best_composite = -1.0
    best_state_dict: dict[str, Any] | None = None
    best_eval_summary: dict[str, float] = {}
    log_rows: list[dict[str, Any]] = []
    global_episode = 0
    total_steps = 0

    save_path = Path(args.save_path)
    log_path = Path(args.log_path)

    # Pha 0 (Tùy chọn): Khởi động DQfD (Deep Q-learning from Demonstrations) để vượt nút thắt 1/121
    if args.dqfd_demo_games > 0 and args.dqfd_pretrain_steps > 0:
        t_demo = time.perf_counter()
        print(f"\n--- [stage0_dqfd_pretrain] thu thập {args.dqfd_demo_games} ván mẫu + {args.dqfd_pretrain_steps} bước cập nhật DQfD ---")
        demo_replay = _collect_dqfd_demo_buffer(
            num_games=args.dqfd_demo_games,
            in_channels=in_channels,
            board_size=board_size,
            win_length=win_length,
            gamma=args.gamma,
            step_penalty=-0.002,
            potential_weight=pot_weight,
            seed=args.seed,
        )
        print(f"  Đã thu thập {len(demo_replay)} chuyển trạng thái mẫu (x8 D4 = {len(demo_replay)*8:,}) trong {time.perf_counter()-t_demo:.1f}s")
        pre_opt = torch.optim.AdamW(online_net.parameters(), lr=args.lr * 1.5, weight_decay=1e-4)
        pre_losses: list[float] = []
        pre_qs: list[float] = []
        for step_i in range(1, args.dqfd_pretrain_steps + 1):
            lv, qv = _dqfd_pretrain_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=pre_opt,
                demo_replay=demo_replay,
                batch_size=args.batch_size,
                gamma=args.gamma,
                tau=0.01,
                margin=args.dqfd_margin,
                margin_weight=args.dqfd_margin_weight,
                grad_clip_norm=1.0,
                device=device,
            )
            pre_losses.append(lv)
            pre_qs.append(qv)
            if step_i % max(1, args.dqfd_pretrain_steps // 2) == 0 or step_i == args.dqfd_pretrain_steps:
                ev1 = _evaluate_model(online_net, in_channels, device, lambda g: RandomAgent(seed=5000 + g), n_games=args.eval_games)
                evs = _evaluate_model(online_net, in_channels, device, lambda g: _FastTacticalOpponent(noise_prob=0.25, seed=6000 + g), n_games=args.eval_games)
                ev2 = _evaluate_model(online_net, in_channels, device, lambda g: RuleBasedAgent(seed=7000 + g), n_games=args.eval_games)
                comp = 0.25 * ev1["score_pct"] + 0.35 * evs["score_pct"] + 0.40 * ev2["score_pct"]
                elapsed = time.perf_counter() - t_start
                is_best = comp >= best_composite
                if is_best:
                    best_composite = comp
                    best_state_dict = copy.deepcopy(online_net.state_dict())
                    best_eval_summary = {
                        "lv1_pct": ev1["score_pct"],
                        "soft_lv2_pct": evs["score_pct"],
                        "lv2_pct": ev2["score_pct"],
                        "composite": comp,
                        "global_episode": 0,
                        "stage": "stage0_dqfd_pretrain",
                    }
                    _save_rl_checkpoint(save_path, online_net, metadata=best_eval_summary)
                log_rows.append(
                    {
                        "global_episode": 0,
                        "stage": "stage0_dqfd_pretrain",
                        "stage_ep": step_i,
                        "epsilon": 0.0,
                        "replay_size": len(demo_replay),
                        "avg_loss": round(statistics.mean(pre_losses[-100:]), 5),
                        "mean_abs_q": round(statistics.mean(pre_qs[-100:]), 4),
                        "eval_lv1_pct": round(ev1["score_pct"], 2),
                        "eval_soft_lv2_pct": round(evs["score_pct"], 2),
                        "eval_lv2_pct": round(ev2["score_pct"], 2),
                        "composite_score": round(comp, 2),
                        "elapsed_sec": round(elapsed, 1),
                    }
                )
                _save_training_log(log_rows, log_path)
                tag = " [BEST]" if is_best else ""
                print(
                    f"  pretrain {step_i:4d}/{args.dqfd_pretrain_steps} | loss={statistics.mean(pre_losses[-100:]):.4f} "
                    f"|Q|={statistics.mean(pre_qs[-100:]):.3f} | vsLv1={ev1['score_pct']:5.1f}% "
                    f"vsSoftTac={evs['score_pct']:5.1f}% vsLv2={ev2['score_pct']:5.1f}% | comp={comp:5.1f} | {elapsed:5.1f}s{tag}"
                )
        # Đồng bộ target_net sau pretrain
        target_net.load_state_dict(online_net.state_dict())

    stages = [
        ("stage1_lv1", args.stage1_episodes, 0.35 if demo_replay is not None else 1.0, 0.08, args.lr),
        ("stage2_ladder", args.stage2_episodes, 0.30 if demo_replay is not None else 0.45, 0.06, args.lr * 0.6),
        ("stage3_selfplay", args.stage3_episodes, 0.20, 0.04, args.lr * 0.35),
    ]

    opponent_pool: list[DuelingCaroDQN] = []
    pool_max_size = 6

    for stage_name, n_eps, eps_start, eps_end, stage_lr in stages:
        if n_eps <= 0:
            continue

        optimizer = torch.optim.AdamW(online_net.parameters(), lr=stage_lr, weight_decay=1e-4)
        print(f"\n--- [{stage_name}] episodes={n_eps} | eps={eps_start:.2f}->{eps_end:.2f} | lr={stage_lr:.5f} ---")

        if stage_name == "stage3_selfplay" and not opponent_pool:
            snap = copy.deepcopy(online_net).to(device)
            snap.eval()
            opponent_pool.append(snap)

        recent_losses: list[float] = []
        recent_qs: list[float] = []
        stage_wins = stage_losses = stage_draws = 0

        for ep in range(1, n_eps + 1):
            global_episode += 1
            frac = min(1.0, ep / max(1.0, 0.85 * n_eps))
            epsilon = eps_start + frac * (eps_end - eps_start)
            rl_side = 1 if (global_episode % 2 == 1) else -1
            is_selfplay = False

            if stage_name == "stage1_lv1":
                # 50% Uniform Random + 50% Local Random (radius=2) để sớm gặp đe dọa thực tế
                opp_radius = None if (ep % 2 == 0) else 2
                opponent: Any = RandomAgent(seed=args.seed + global_episode, candidate_radius=opp_radius)
            elif stage_name == "stage2_ladder":
                # Noise giảm mượt từ 0.70 -> 0.15 theo tiến độ stage 2 để tạo cầu nối học mượt
                noise_p = 0.70 - 0.55 * frac
                opponent = _FastTacticalOpponent(noise_prob=noise_p, candidate_radius=2, seed=args.seed + global_episode)
            else:
                roll = py_rng.random()
                if roll < 0.65 and opponent_pool:
                    is_selfplay = True
                    opponent = py_rng.choice(opponent_pool)
                elif roll < 0.85:
                    opponent = _FastTacticalOpponent(noise_prob=0.18, candidate_radius=2, seed=args.seed + global_episode)
                elif roll < 0.95:
                    opponent = RuleBasedAgent(root_width=4, seed=args.seed + global_episode)
                else:
                    opponent = RandomAgent(seed=args.seed + global_episode)

            winner, pushed = _play_training_episode(
                env=env,
                online_net=online_net,
                opponent=opponent,
                is_selfplay=is_selfplay,
                rl_side=rl_side,
                in_channels=in_channels,
                epsilon=epsilon,
                gamma=args.gamma,
                step_penalty=-0.002,
                potential_weight=pot_weight,
                opening_random_plies=2,
                replay=replay,
                device=device,
                py_rng=py_rng,
            )
            if winner == rl_side:
                stage_wins += 1
            elif winner == -rl_side:
                stage_losses += 1
            else:
                stage_draws += 1

            total_steps += pushed
            n_updates = max(1, pushed // max(1, args.train_freq))
            if len(replay) >= max(512, args.batch_size * 4):
                for _ in range(n_updates):
                    if demo_replay is not None and args.dqfd_online_ratio > 0.0 and py_rng.random() < args.dqfd_online_ratio:
                        loss_v, q_abs = _dqfd_pretrain_step(
                            online_net=online_net,
                            target_net=target_net,
                            optimizer=optimizer,
                            demo_replay=demo_replay,
                            batch_size=args.batch_size,
                            gamma=args.gamma,
                            tau=args.tau,
                            margin=args.dqfd_margin,
                            margin_weight=args.dqfd_margin_weight * 0.5,
                            grad_clip_norm=1.0,
                            device=device,
                        )
                    else:
                        loss_v, q_abs = _dqn_train_step(
                            online_net=online_net,
                            target_net=target_net,
                            optimizer=optimizer,
                            replay=replay,
                            batch_size=args.batch_size,
                            gamma=args.gamma,
                            tau=args.tau,
                            terminal_ratio=0.20,
                            use_d4_aug=True,
                            grad_clip_norm=1.0,
                            device=device,
                        )
                    recent_losses.append(loss_v)
                    recent_qs.append(q_abs)

            # Cập nhật Opponent Pool trong Stage 3
            if stage_name == "stage3_selfplay" and ep % 200 == 0:
                snap = copy.deepcopy(online_net).to(device)
                snap.eval()
                opponent_pool.append(snap)
                if len(opponent_pool) > pool_max_size:
                    opponent_pool.pop(0)

            # Sát hạch định kỳ
            if ep % args.eval_every == 0 or ep == n_eps:
                eval_lv1 = _evaluate_model(
                    online_net, in_channels, device,
                    make_opponent=lambda g: RandomAgent(seed=5000 + g),
                    n_games=args.eval_games,
                )
                eval_soft = _evaluate_model(
                    online_net, in_channels, device,
                    make_opponent=lambda g: _FastTacticalOpponent(noise_prob=0.25, seed=6000 + g),
                    n_games=args.eval_games,
                )
                eval_lv2 = _evaluate_model(
                    online_net, in_channels, device,
                    make_opponent=lambda g: RuleBasedAgent(seed=7000 + g),
                    n_games=args.eval_games,
                )
                composite = (
                    0.25 * eval_lv1["score_pct"]
                    + 0.35 * eval_soft["score_pct"]
                    + 0.40 * eval_lv2["score_pct"]
                )
                avg_loss = statistics.mean(recent_losses[-200:]) if recent_losses else 0.0
                avg_q = statistics.mean(recent_qs[-200:]) if recent_qs else 0.0
                elapsed = time.perf_counter() - t_start

                is_best = composite >= best_composite
                if is_best:
                    best_composite = composite
                    best_state_dict = copy.deepcopy(online_net.state_dict())
                    best_eval_summary = {
                        "lv1_pct": eval_lv1["score_pct"],
                        "soft_lv2_pct": eval_soft["score_pct"],
                        "lv2_pct": eval_lv2["score_pct"],
                        "composite": composite,
                        "global_episode": global_episode,
                        "stage": stage_name,
                    }
                    _save_rl_checkpoint(save_path, online_net, metadata=best_eval_summary)

                row = {
                    "global_episode": global_episode,
                    "stage": stage_name,
                    "stage_ep": ep,
                    "epsilon": round(epsilon, 4),
                    "replay_size": len(replay),
                    "avg_loss": round(avg_loss, 5),
                    "mean_abs_q": round(avg_q, 4),
                    "eval_lv1_pct": round(eval_lv1["score_pct"], 2),
                    "eval_soft_lv2_pct": round(eval_soft["score_pct"], 2),
                    "eval_lv2_pct": round(eval_lv2["score_pct"], 2),
                    "composite_score": round(composite, 2),
                    "elapsed_sec": round(elapsed, 1),
                }
                log_rows.append(row)
                _save_training_log(log_rows, log_path)

                tag = " [BEST]" if is_best else ""
                print(
                    f"  ep {ep:4d}/{n_eps} (tot {global_episode:4d}) | eps={epsilon:.2f} | "
                    f"loss={avg_loss:.4f} |Q|={avg_q:.3f} | "
                    f"vsLv1={eval_lv1['score_pct']:5.1f}% vsSoftTac={eval_soft['score_pct']:5.1f}% "
                    f"vsLv2={eval_lv2['score_pct']:5.1f}% | comp={composite:5.1f} | {elapsed:5.1f}s{tag}"
                )

    if best_state_dict is not None:
        online_net.load_state_dict(best_state_dict)
        _save_rl_checkpoint(save_path, online_net, metadata=best_eval_summary)

    # Kiểm chứng cuối cùng qua giao diện RLAgent chính thức
    print("\n=== Đánh giá cuối cùng checkpoint chính thức qua RLAgent ===")
    final_agent = RLAgent(checkpoint_path=save_path, device=args.device)
    final_lv1 = _evaluate_model(
        final_agent.model, final_agent.in_channels, final_agent.device,
        make_opponent=lambda g: RandomAgent(seed=9000 + g),
        n_games=20,
    )
    final_soft = _evaluate_model(
        final_agent.model, final_agent.in_channels, final_agent.device,
        make_opponent=lambda g: _FastTacticalOpponent(noise_prob=0.25, seed=9100 + g),
        n_games=20,
    )
    final_lv2 = _evaluate_model(
        final_agent.model, final_agent.in_channels, final_agent.device,
        make_opponent=lambda g: RuleBasedAgent(seed=9200 + g),
        n_games=20,
    )
    print(f"Checkpoint: {save_path}")
    print(f"  vs Random (20g):        {final_lv1['wins']}W-{final_lv1['draws']}D-{final_lv1['losses']}L ({final_lv1['score_pct']:.1f}%)")
    print(f"  vs Soft-Tactical (20g): {final_soft['wins']}W-{final_soft['draws']}D-{final_soft['losses']}L ({final_soft['score_pct']:.1f}%)")
    print(f"  vs Lv2 RuleBased (20g): {final_lv2['wins']}W-{final_lv2['draws']}D-{final_lv2['losses']}L ({final_lv2['score_pct']:.1f}%)")
    print(f"  Log CSV: {log_path} | Total elapsed: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
