import numpy as np
import pytest
import torch

from src.agents.rl_agent import DuelingCaroDQN, RLAgent, encode_board_state
from src.caro_env import CaroEnv


def _build_midgame_env() -> CaroEnv:
    """Tạo bàn cờ trung cuộc có cả quân X (1) và O (-1)."""
    env = CaroEnv(board_size=11, win_length=5)
    for move in [60, 61, 49, 38, 71, 72]:
        env.step(move)
    return env


def test_encode_board_state_canonical_perspective():
    """Kiểm tra mã hóa 3 kênh chuẩn hóa cho cả góc nhìn X (1) và O (-1)."""
    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)  # X đánh (5, 5)
    env.step(61)  # O đánh (5, 6)

    # Góc nhìn của X (1)
    state_x = encode_board_state(env.board, current_player=1)
    assert state_x.shape == (3, 11, 11)
    assert state_x.dtype == np.float32
    assert state_x[0, 5, 5] == 1.0 and state_x[1, 5, 6] == 1.0
    assert state_x[2, 5, 5] == 0.0 and state_x[2, 0, 0] == 1.0
    # Tại mỗi ô, tổng 3 kênh phải luôn bằng 1 (chỉ thuộc 1 trong 3 trạng thái)
    assert np.all(state_x.sum(axis=0) == 1.0)

    # Góc nhìn của O (-1): kênh 0 và kênh 1 phải đảo ngược so với X
    state_o = encode_board_state(env.board, current_player=-1)
    assert np.array_equal(state_o[0], state_x[1])
    assert np.array_equal(state_o[1], state_x[0])
    assert np.array_equal(state_o[2], state_x[2])


def test_dueling_dqn_forward_shape_and_finite():
    """Kiểm tra kích thước đầu ra (B, 121) và không có NaN/Inf."""
    model = DuelingCaroDQN(in_channels=3, conv_channels=(32, 32, 64), board_size=11)
    model.eval()
    dummy_input = torch.zeros((4, 3, 11, 11), dtype=torch.float32)
    with torch.no_grad():
        q_values = model(dummy_input)
    assert q_values.shape == (4, 121)
    assert torch.isfinite(q_values).all()


def test_rl_agent_select_action_both_sides_and_no_env_mutation():
    """Kiểm tra RLAgent chơi hợp lệ cho cả X và O, không làm biến đổi env gốc."""
    agent = RLAgent(checkpoint_path=None, device="cpu")
    env = _build_midgame_env()

    for _ in range(2):  # Thử 1 lượt của X (1) và 1 lượt của O (-1)
        board_before = env.board.copy()
        history_before = env.history.copy()
        player_before = env.current_player

        action = agent.select_action(env)

        assert isinstance(action, int)
        assert action in env.legal_actions()
        assert agent.last_search_count == 1
        assert agent.last_move_time_ms > 0.0

        # Đảm bảo env gốc không bị thay đổi sau khi gọi select_action
        assert np.array_equal(env.board, board_before)
        assert env.history == history_before
        assert env.current_player == player_before

        env.step(action)


def test_rl_agent_action_masking_single_empty_cell():
    """Kiểm chứng Action Masking: khi chỉ còn duy nhất 1 ô trống, bắt buộc chọn đúng ô đó."""
    agent = RLAgent(checkpoint_path=None, device="cpu")
    env = CaroEnv(board_size=11, win_length=5)

    # Điền kín bàn cờ trừ ô cuối cùng (10, 10) -> action = 120
    env.board[:, :] = 1
    env.board[0::2, 1::2] = -1
    env.board[10, 10] = 0
    env._done = False

    masked_q = agent.get_masked_q_values(env)
    assert np.isneginf(masked_q[:120]).all()
    assert np.isfinite(masked_q[120])
    assert agent.select_action(env) == 120


def test_rl_agent_raises_on_terminal_env():
    """Kiểm tra ném ValueError khi bàn cờ đã kết thúc."""
    agent = RLAgent(checkpoint_path=None, device="cpu")
    env = CaroEnv(board_size=11, win_length=5)
    for i in range(4):
        env.step(5 * 11 + i)
        env.step(0 * 11 + i)
    env.step(5 * 11 + 4)  # X thắng
    assert env.is_terminal()

    with pytest.raises(ValueError):
        agent.select_action(env)


def test_rl_agent_get_top_actions():
    """Kiểm tra get_top_actions trả về đúng số lượng, hợp lệ và giảm dần theo Q-value."""
    agent = RLAgent(checkpoint_path=None, device="cpu")
    env = _build_midgame_env()
    legal = set(env.legal_actions())

    top_actions = agent.get_top_actions(env, top_k=5)
    assert len(top_actions) == 5
    for item in top_actions:
        assert item["action"] in legal
        assert item["row"] * 11 + item["col"] == item["action"]
    q_list = [item["q_value"] for item in top_actions]
    assert q_list == sorted(q_list, reverse=True)


def test_rl_agent_checkpoint_save_and_load_deterministic(tmp_path):
    """Kiểm tra lưu và nạp checkpoint cho ra cùng Q-value và cùng nước đi."""
    torch.manual_seed(42)
    agent1 = RLAgent(checkpoint_path=None, conv_channels=(32, 32, 64), device="cpu")
    env = _build_midgame_env()

    q1 = agent1.get_masked_q_values(env)
    action1 = agent1.select_action(env)

    ckpt_path = tmp_path / "test_rl_best.pt"
    torch.save(
        {
            "model_state_dict": agent1.model.state_dict(),
            "in_channels": agent1.in_channels,
            "conv_channels": agent1.conv_channels,
            "board_size": agent1.board_size,
        },
        ckpt_path,
    )

    agent2 = RLAgent(checkpoint_path=ckpt_path, device="cpu")
    q2 = agent2.get_masked_q_values(env)
    action2 = agent2.select_action(env)

    assert action2 == action1
    assert np.allclose(q1[np.isfinite(q1)], q2[np.isfinite(q2)], atol=1e-6)


def test_rl_agent_missing_checkpoint_raises():
    """Kiểm tra ném FileNotFoundError khi truyền đường dẫn checkpoint không tồn tại."""
    with pytest.raises(FileNotFoundError):
        RLAgent(checkpoint_path="checkpoints/khong_ton_tai.pt", device="cpu")