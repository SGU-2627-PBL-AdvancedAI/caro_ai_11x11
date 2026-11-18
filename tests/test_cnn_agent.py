import numpy as np
import pytest

from src.caro_env import CaroEnv
from src.agents.cnn_agent import (
    CNNAgent,
    augment_state_action,
    encode_board,
    expert_select_action,
    generate_expert_dataset,
    non_expert_select_action,
    to_one_hot,
)


def _build_midgame_env() -> CaroEnv:
    """Tạo bàn cờ trung cuộc cân bằng (không có nước thắng/chặn 5 quân tức thì)."""
    env = CaroEnv(board_size=11, win_length=5)
    for move in [60, 61, 49, 38, 71, 72]:
        env.step(move)
    return env


def _build_four_in_a_row_env_for_x() -> CaroEnv:
    """
    Tạo bàn cờ X có 4 quân liên tiếp tại (5,0), (5,1), (5,2), (5,3).
    Sau 8 nước đi, current_player == 1 (X) và ô (5,4) = 59 là nước thắng ngay.
    """
    env = CaroEnv(board_size=11, win_length=5)
    x_moves = [5 * 11 + 0, 5 * 11 + 1, 5 * 11 + 2, 5 * 11 + 3]
    o_moves = [0 * 11 + 0, 0 * 11 + 2, 2 * 11 + 0, 2 * 11 + 2]
    for xm, om in zip(x_moves, o_moves):
        env.step(xm)
        env.step(om)
    assert env.current_player == 1
    return env


def _build_opponent_four_in_a_row_env() -> CaroEnv:
    """
    Tạo bàn cờ đến lượt O (-1), trong đó X (1) đã có 4 quân liên tiếp
    tại (5,0), (5,1), (5,2), (5,3) và ô (5,4) = 59 cần phải chặn ngay.
    """
    env = CaroEnv(board_size=11, win_length=5)
    x_moves = [5 * 11 + 0, 5 * 11 + 1, 5 * 11 + 2, 5 * 11 + 3]
    o_moves = [0 * 11 + 0, 0 * 11 + 2, 2 * 11 + 0]
    for i, xm in enumerate(x_moves):
        env.step(xm)
        if i < len(o_moves):
            env.step(o_moves[i])
    assert env.current_player == -1
    return env


@pytest.mark.parametrize("bad_model_type", ["medium", "", None])
def test_cnn_invalid_model_type(bad_model_type):
    with pytest.raises(ValueError):
        CNNAgent(model_type=bad_model_type, backend="numpy")


@pytest.mark.parametrize("bad_backend", ["torch", "", None])
def test_cnn_invalid_backend(bad_backend):
    with pytest.raises(ValueError):
        CNNAgent(backend=bad_backend)


@pytest.mark.parametrize("bad_board_size", [0, 1, 2, -5])
def test_cnn_invalid_board_size(bad_board_size):
    with pytest.raises(ValueError):
        CNNAgent(board_size=bad_board_size, backend="numpy")


@pytest.mark.parametrize("bad_temp", [0.0, -0.5, -1.0])
def test_cnn_invalid_temperature(bad_temp):
    with pytest.raises(ValueError):
        CNNAgent(temperature=bad_temp, backend="numpy")


def test_cnn_invalid_candidate_radius_and_rollouts():
    with pytest.raises(ValueError):
        CNNAgent(candidate_radius=-1, backend="numpy")
    with pytest.raises(ValueError):
        CNNAgent(num_rollouts=0, backend="numpy")


def test_encode_board_player_independent():
    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)  # X (+1) tại tâm (5, 5)
    # Đến lượt O (-1): góc nhìn của O phải thấy quân mình là +1, quân X là -1
    encoded_o = encode_board(env.board, current_player=-1)
    assert encoded_o.shape == (11, 11, 1)
    assert encoded_o.dtype == np.float32
    assert encoded_o[5, 5, 0] == -1.0

    # Nếu giả lập góc nhìn X (+1): thấy quân X là +1
    encoded_x = encode_board(env.board, current_player=1)
    assert encoded_x[5, 5, 0] == 1.0


def test_to_one_hot_and_symmetry_augmentation():
    actions = [0, 60, 120]
    onehot = to_one_hot(actions, num_classes=121)
    assert onehot.shape == (3, 121)
    assert np.all(onehot.sum(axis=1) == 1.0)
    assert onehot[0, 0] == 1.0 and onehot[1, 60] == 1.0 and onehot[2, 120] == 1.0

    state = np.zeros((11, 11, 1), dtype=np.float32)
    state[2, 3, 0] = 1.0
    action = 4 * 11 + 7
    augmented = augment_state_action(state, action, board_size=11)
    assert len(augmented) == 8
    for aug_state, aug_action in augmented:
        assert aug_state.shape == (11, 11, 1)
        assert 0 <= aug_action < 121
        assert float(aug_state.sum()) == 1.0


def test_expert_and_non_expert_tactical_behavior():
    win_env = _build_four_in_a_row_env_for_x()
    assert expert_select_action(win_env) == 59

    block_env = _build_opponent_four_in_a_row_env()
    assert expert_select_action(block_env) == 59

    # random_prob=0.0 => hành xử giống hệt expert
    assert non_expert_select_action(win_env, random_prob=0.0) == 59
    # random_prob=1.0 => luôn chọn nước hợp lệ
    rng = np.random.default_rng(42)
    rand_action = non_expert_select_action(win_env, random_prob=1.0, rng=rng)
    assert rand_action in win_env.legal_actions()


def test_generate_expert_dataset_shapes():
    X, y = generate_expert_dataset(
        num_games=2,
        board_size=11,
        win_length=5,
        random_prob=0.5,
        augment_symmetry=False,
        seed=42,
    )
    assert X.ndim == 4 and X.shape[1:] == (11, 11, 1)
    assert y.ndim == 1 and y.shape[0] == X.shape[0]
    assert X.shape[0] > 0
    assert np.all((y >= 0) & (y < 121))


def test_cnn_predict_probs_valid_distribution():
    env = _build_midgame_env()
    agent = CNNAgent(model_type="fast", backend="numpy", conv_filters=16, dense_units=16, seed=42)
    probs = agent.predict_probs(env)

    assert probs.shape == (121,)
    assert np.all(probs >= 0.0)
    assert np.isclose(float(probs.sum()), 1.0, atol=1e-5)

    # Các ô đã đánh phải có xác suất = 0
    occupied = [60, 61, 49, 38, 71, 72]
    assert np.all(probs[occupied] == 0.0)


def test_cnn_select_action_pure_policy_and_tactical_shortcut():
    agent = CNNAgent(
        model_type="fast",
        backend="numpy",
        conv_filters=16,
        dense_units=16,
        use_tactical_shortcut=True,
        use_mix_mcts=False,
        seed=42,
    )

    win_env = _build_four_in_a_row_env_for_x()
    assert agent.select_action(win_env) == 59

    block_env = _build_opponent_four_in_a_row_env()
    assert agent.select_action(block_env) == 59

    mid_env = _build_midgame_env()
    move = agent.select_action(mid_env)
    assert move in mid_env.legal_actions()


def test_cnn_select_action_mix_mcts():
    env = _build_midgame_env()
    agent = CNNAgent(
        model_type="fast",
        backend="numpy",
        conv_filters=16,
        dense_units=16,
        use_mix_mcts=True,
        num_rollouts=10,
        gamma=10.0,
        max_rollout_steps=12,
        seed=42,
    )
    move = agent.select_action(env)
    assert move in env.legal_actions()
    assert agent.last_search_count == 10


def test_cnn_train_and_save_load_roundtrip(tmp_path):
    X, y = generate_expert_dataset(
        num_games=2,
        board_size=11,
        win_length=5,
        random_prob=0.5,
        augment_symmetry=False,
        seed=42,
    )
    agent = CNNAgent(
        model_type="fast",
        backend="numpy",
        conv_filters=8,
        dense_units=16,
        seed=42,
    )
    history = agent.train_from_dataset(
        X[:32],
        y[:32],
        epochs=2,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
    )
    assert "loss" in history and len(history["loss"]) == 2

    env = _build_midgame_env()
    probs_before = agent.predict_probs(env).copy()

    ckpt_path = tmp_path / "cnn_test_weights.npz"
    saved_path = agent.save_model(ckpt_path)
    assert saved_path.exists()

    reloaded_agent = CNNAgent(
        model_type="fast",
        backend="numpy",
        conv_filters=8,
        dense_units=16,
        seed=99,
    )
    reloaded_agent.load_model(saved_path)
    probs_after = reloaded_agent.predict_probs(env)
    assert np.allclose(probs_before, probs_after, atol=1e-6)
