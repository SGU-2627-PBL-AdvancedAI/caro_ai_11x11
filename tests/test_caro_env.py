import pytest
from src.caro_env import CaroEnv
from src.heuristic import get_candidate_actions, evaluate_cell_threat, evaluate_board
from src.agents.base_agent import BaseAgent


def test_initial_state_and_reset():
    env = CaroEnv(board_size=11, win_length=5)
    state = env.reset()
    assert state.shape == (11, 11)
    assert len(env.legal_actions()) == 121
    assert not env.is_terminal()
    assert env.current_player == 1
    assert env.result(1) == 0.0
    assert env.result(-1) == 0.0


@pytest.mark.parametrize("bad_action", [-1, 121, 999, "60", 3.5, None, True])
def test_invalid_action_types_and_bounds(bad_action):
    env = CaroEnv()
    with pytest.raises(ValueError):
        env.step(bad_action)


def test_occupied_cell_raises_error():
    env = CaroEnv()
    env.step(60)
    with pytest.raises(ValueError):
        env.step(60)


def test_horizontal_vertical_diagonal_wins():
    env = CaroEnv()
    for i in range(4):
        env.step(5 * 11 + i)
        env.step(0 * 11 + i)
    _, reward, done, _ = env.step(5 * 11 + 4)
    assert done is True and reward == 1.0 and env.winner == 1

    env.reset()
    for i in range(4):
        env.step(0 * 11 + i)
        env.step(i * 11 + 10)
    env.step(9 * 11 + 0)
    _, reward, done, _ = env.step(4 * 11 + 10)
    assert done is True and reward == 1.0 and env.winner == -1
    assert env.result(perspective=-1) == 1.0
    assert env.result(perspective=1) == -1.0

    env.reset()
    for i in range(4):
        env.step(i * 11 + i)
        env.step(i * 11 + (i + 1))
    _, reward, done, _ = env.step(4 * 11 + 4)
    assert done is True and reward == 1.0 and env.winner == 1


def test_edge_win_and_no_moves_after_terminal():
    env = CaroEnv()
    for i in range(4):
        env.step(10 * 11 + (6 + i))
        env.step(0 * 11 + i)

    _, reward, done, _ = env.step(10 * 11 + 10)
    assert done is True
    assert reward == 1.0
    assert env.is_terminal() is True
    assert env.legal_actions() == []
    assert env.current_player == 1

    with pytest.raises(ValueError):
        env.step(50)


def test_clone_independence_and_undo_restoration():
    env = CaroEnv()
    for i in range(4):
        env.step(5 * 11 + i)
        env.step(6 * 11 + i)

    cloned_env = env.clone()
    env.step(5 * 11 + 4)
    assert env.is_terminal() is True
    assert cloned_env.is_terminal() is False
    assert cloned_env.board[5, 4] == 0

    undone_action = env.undo()
    assert undone_action == 5 * 11 + 4
    assert env.is_terminal() is False
    assert env.winner == 0
    assert env.current_player == 1
    assert env.board[5, 4] == 0
    assert env.result(1) == 0.0


def test_full_board_draw():
    env = CaroEnv(board_size=4, win_length=4)
    pattern = [
         1,  1, -1,  1,
        -1, -1,  1, -1,
         1,  1, -1,  1,
        -1, -1,  1, -1,
    ]
    x_moves = [idx for idx, val in enumerate(pattern) if val == 1]
    o_moves = [idx for idx, val in enumerate(pattern) if val == -1]
    for xm, om in zip(x_moves, o_moves):
        env.step(xm)
        env.step(om)
    assert env.is_terminal() is True
    assert env.winner == 0
    assert env.result(1) == 0.0
    assert env.result(-1) == 0.0
    assert env.legal_actions() == []


def test_heuristic_attacking_advantage_and_symmetry():
    assert hasattr(BaseAgent, "select_action")
    env = CaroEnv()
    assert get_candidate_actions(env, radius=2) == [60]

    # 1. Thế cờ X chiếm ưu thế tấn công (X có 4 quân liên tiếp, O có 4 quân rời rạc)
    x_moves = [5 * 11 + 0, 5 * 11 + 1, 5 * 11 + 2, 5 * 11 + 3]
    o_moves = [0 * 11 + 0, 0 * 11 + 2, 2 * 11 + 0, 2 * 11 + 2]
    for xm, om in zip(x_moves, o_moves):
        env.step(xm)
        env.step(om)

    threat = evaluate_cell_threat(env, 5 * 11 + 4, player=1)
    assert threat["is_win_5"] is True
    assert evaluate_board(env, perspective=1) > 0.0
    assert evaluate_board(env, perspective=-1) < 0.0

    # 2. Tính đối xứng khi hoán đổi vị trí quân của X và O
    env_swapped = CaroEnv()
    for om, xm in zip(o_moves, x_moves):
        env_swapped.step(om)
        env_swapped.step(xm)

    assert evaluate_board(env, perspective=1) == pytest.approx(evaluate_board(env_swapped, perspective=-1))
    assert evaluate_board(env, perspective=-1) == pytest.approx(evaluate_board(env_swapped, perspective=1))