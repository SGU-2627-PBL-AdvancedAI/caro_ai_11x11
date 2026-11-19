from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from agents.base_agent import BaseAgent
from agents.cnn_agent import (
    CNNAgent,
    HAS_TENSORFLOW,
    augment_state_action,
    encode_board,
    expert_select_action,
    export_dataset_to_csv,
    generate_expert_dataset,
    generate_expert_train_val_datasets,
    load_expert_train_val_datasets,
    to_one_hot,
)
from caro_env import CaroEnv


def test_encode_board_and_symmetry_augmentation():
    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)  # X (1) đánh tại tâm (5, 5)
    env.step(48)  # O (-1) đánh tại (4, 4)

    state_p1 = encode_board(env.board, current_player=1)
    assert state_p1.shape == (11, 11, 1)
    assert state_p1[5, 5, 0] == 1.0
    assert state_p1[4, 4, 0] == -1.0

    state_p2 = encode_board(env.board, current_player=-1)
    assert state_p2.shape == (11, 11, 1)
    assert state_p2[5, 5, 0] == -1.0
    assert state_p2[4, 4, 0] == 1.0

    aug_samples = augment_state_action(state_p1, action=2 * 11 + 3, board_size=11)
    assert len(aug_samples) == 8
    for aug_s, aug_a in aug_samples:
        assert aug_s.shape == (11, 11, 1)
        assert 0 <= aug_a < 121
        assert aug_s[5, 5, 0] == 1.0


def test_to_one_hot():
    y = np.array([0, 60, 120], dtype=np.int64)
    oh = to_one_hot(y, num_classes=121)
    assert oh.shape == (3, 121)
    assert np.allclose(oh.sum(axis=1), 1.0)
    assert oh[0, 0] == 1.0 and oh[1, 60] == 1.0 and oh[2, 120] == 1.0


def test_expert_and_game_level_train_val_dataset_generation():
    env = CaroEnv(board_size=11, win_length=5)
    for col in (1, 2, 3, 4):
        env.step(5 * 11 + col)
        env.step(2 * 11 + col)

    action = expert_select_action(env)
    assert action in (5 * 11 + 0, 5 * 11 + 5)

    X_train, y_train, X_val, y_val = generate_expert_train_val_datasets(
        num_train_games=2,
        num_val_games=1,
        board_size=11,
        win_length=5,
        augment_train_symmetry=True,
        augment_val_symmetry=False,
        seed=42,
    )
    assert X_train.ndim == 4 and X_train.shape[1:] == (11, 11, 1)
    assert X_val.ndim == 4 and X_val.shape[1:] == (11, 11, 1)
    assert X_train.shape[0] == y_train.shape[0] and X_train.shape[0] >= 16
    assert X_val.shape[0] == y_val.shape[0] and X_val.shape[0] >= 1
    assert np.all((y_train >= 0) & (y_train < 121))
    assert np.all((y_val >= 0) & (y_val < 121))


@pytest.mark.parametrize("model_type", ["fast", "strong"])
def test_cnn_agent_architecture_and_inference(model_type: str):
    agent = CNNAgent(
        name=f"Test_{model_type}",
        model_type=model_type,
        backend="numpy",
        deterministic=True,
        seed=42,
    )
    assert isinstance(agent, BaseAgent)
    assert agent.time_limit_sec == 1.0

    expected_dense_layers = 3 if model_type == "fast" else 4
    assert len(agent.numpy_net.dense_weights) == expected_dense_layers

    env = CaroEnv(board_size=11, win_length=5)
    first_move = agent.select_action(env)
    assert first_move == 60
    assert len(env.history) == 0

    env.step(60)
    env.step(61)

    probs = agent.predict_probs(env, restrict_to_candidates=False)
    assert probs.shape == (121,)
    assert probs[60] == 0.0
    assert probs[61] == 0.0
    assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-5)

    board_snapshot = env.board.copy()
    move = agent.select_action(env)
    assert move in env.legal_actions()
    assert np.array_equal(env.board, board_snapshot)
    assert agent.last_move_time_ms >= 0.0


def test_cnn_agent_tactical_win_and_block():
    agent = CNNAgent(model_type="fast", backend="numpy", use_tactical_shortcut=True, seed=42)
    env = CaroEnv(board_size=11, win_length=5)

    for c in range(3, 7):
        env.step(5 * 11 + c)
        env.step(3 * 11 + c)
    win_move = agent.select_action(env)
    assert win_move in (5 * 11 + 2, 5 * 11 + 7)
    assert agent.last_decision_stage == "tactical_win_5"

    env2 = CaroEnv(board_size=11, win_length=5)
    for c in range(0, 3):
        env2.step(5 * 11 + c)
        env2.step(3 * 11 + c)
    env2.step(5 * 11 + 3)
    block_move = agent.select_action(env2)
    assert block_move == 5 * 11 + 4
    assert agent.last_decision_stage == "tactical_block_win_5"


def test_cnn_agent_mix_mcts_mode():
    agent = CNNAgent(
        model_type="strong",
        backend="numpy",
        use_mix_mcts=True,
        num_rollouts=16,
        gamma=10.0,
        use_tactical_shortcut=False,
        seed=42,
    )
    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)
    env.step(48)

    board_before = env.board.copy()
    action = agent.select_action(env)
    assert action in env.legal_actions()
    assert np.array_equal(env.board, board_before)
    assert agent.last_decision_stage == "policy_mix_mcts"
    assert agent.last_search_count == 16


def test_cnn_training_and_save_load_checkpoint(tmp_path: Path):
    # Không tự nạp checkpoint mặc định khi model_path=None
    agent = CNNAgent(
        model_type="fast",
        backend="numpy",
        model_path=None,
        learning_rate=5e-3,
        seed=123,
    )

    X, y = generate_expert_dataset(
        num_games=2,
        board_size=11,
        win_length=5,
        augment_symmetry=False,
        seed=123,
    )
    history = agent.train_from_dataset(
        X=X,
        y=y,
        epochs=5,
        batch_size=16,
        validation_split=0.0,
        verbose=0,
    )
    assert len(history["loss"]) == 5
    assert history["loss"][-1] < history["loss"][0]

    ckpt_path = tmp_path / "test_cnn_fast.npz"
    saved_file = agent.save_model(ckpt_path)
    assert saved_file.exists()

    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)
    probs_before = agent.predict_probs(env).copy()

    loaded_agent = CNNAgent(
        model_type="fast",
        backend="numpy",
        model_path=saved_file,
        seed=999,
    )
    probs_after = loaded_agent.predict_probs(env)
    assert np.allclose(probs_before, probs_after, atol=1e-6)


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow/Keras không được cài đặt trong môi trường này.")
def test_keras_backend_and_numpy_weight_equivalence(tmp_path: Path):
    agent = CNNAgent(
        model_type="strong",
        backend="keras",
        seed=42,
    )
    assert agent.keras_model is not None

    env = CaroEnv(board_size=11, win_length=5)
    env.step(60)
    env.step(48)
    state_4d = agent.encode_env_state(env)

    keras_probs = agent.keras_model.predict(state_4d, verbose=0).reshape(-1)
    numpy_probs = agent.numpy_net.predict(state_4d).reshape(-1)
    assert np.allclose(keras_probs, numpy_probs, atol=1e-5)


def test_dataset_save_load_npz_and_export_csv(tmp_path: Path):
    npz_file = tmp_path / "test_train_val.npz"
    csv_file = tmp_path / "test_train_val.csv"

    X_tr, y_tr, X_val, y_val = generate_expert_train_val_datasets(
        num_train_games=1,
        num_val_games=1,
        augment_train_symmetry=True,
        augment_val_symmetry=False,
        seed=42,
        save_path=npz_file,
        load_if_exists=False,
    )
    assert npz_file.exists()

    # Nạp lại từ .npz và kiểm tra khớp tuyệt đối
    X_tr2, y_tr2, X_val2, y_val2, meta = load_expert_train_val_datasets(npz_file)
    assert np.array_equal(X_tr, X_tr2)
    assert np.array_equal(y_tr, y_tr2)
    assert np.array_equal(X_val, X_val2)
    assert np.array_equal(y_val, y_val2)
    assert meta["num_train_games"] == 1 and meta["num_val_games"] == 1

    # Kiểm tra xuất CSV (train + append val)
    export_dataset_to_csv(X_tr, y_tr, csv_file, split_label="train_aug", append=False)
    export_dataset_to_csv(X_val, y_val, csv_file, split_label="val_unseen", append=True)
    lines = csv_file.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 1 + len(y_tr) + len(y_val)
    assert lines[0].startswith("sample_id,split,action,row,col")
