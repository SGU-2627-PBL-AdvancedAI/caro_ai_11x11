import pytest
import numpy as np
from src.caro_env import CaroEnv
from src.agents.mcts_agent import MCTSAgent


def _build_midgame_env() -> CaroEnv:
    """Tạo bàn cờ trung cuộc cân bằng (không có nước thắng/chặn 5 quân tức thì)."""
    env = CaroEnv(board_size=11, win_length=5)
    for move in [60, 61, 49, 38, 71, 72]:
        env.step(move)
    return env


def _build_four_in_a_row_env_for_x() -> CaroEnv:
    """
    Tạo bàn cờ X có 4 quân liên tiếp tại (5,0), (5,1), (5,2), (5,3) và O có 4 quân rời rạc.
    Sau 8 nước đi, current_player == 1 (lượt của X) và ô (5,4) = 59 là nước thắng ngay.
    """
    env = CaroEnv(board_size=11, win_length=5)
    x_moves = [5 * 11 + 0, 5 * 11 + 1, 5 * 11 + 2, 5 * 11 + 3]
    o_moves = [0 * 11 + 0, 0 * 11 + 2, 2 * 11 + 0, 2 * 11 + 2]
    for xm, om in zip(x_moves, o_moves):
        env.step(xm)
        env.step(om)
    assert env.current_player == 1
    return env


@pytest.mark.parametrize("bad_rollouts", [0, -1, -100, True, False, 12.5, "100"])
def test_mcts_invalid_num_rollouts_validation(bad_rollouts):
    """Kiểm chứng MF-1: Từ chối num_rollouts không phải số nguyên dương hoặc là kiểu bool."""
    with pytest.raises(ValueError):
        MCTSAgent(num_rollouts=bad_rollouts)


@pytest.mark.parametrize("bad_time", [0, 0.0, -0.5, True, False, float("inf"), float("-inf"), float("nan"), "0.5"])
def test_mcts_invalid_time_limit_validation(bad_time):
    """Kiểm chứng MF-1: Từ chối time_limit_sec không phải số thực hữu hạn dương hoặc là kiểu bool."""
    with pytest.raises(ValueError):
        MCTSAgent(time_limit_sec=bad_time)


@pytest.mark.parametrize("bad_c", [-0.1, -5, True, False, float("inf"), float("-inf"), float("nan"), "1.4"])
def test_mcts_invalid_exploration_c_validation(bad_c):
    """Kiểm tra exploration_c phải là số thực hữu hạn >= 0 và không phải bool."""
    with pytest.raises(ValueError):
        MCTSAgent(exploration_c=bad_c)


@pytest.mark.parametrize("bad_radius", [-1, -5, True, False, 1.5, "2", None])
def test_mcts_invalid_radius_validation(bad_radius):
    """Kiểm tra candidate_radius và rollout_radius phải là số nguyên không âm và không phải bool."""
    with pytest.raises(ValueError):
        MCTSAgent(candidate_radius=bad_radius)
    with pytest.raises(ValueError):
        MCTSAgent(rollout_radius=bad_radius)


@pytest.mark.parametrize("bad_steps", [0, -1, -10, True, False, 5.5, "30", None])
def test_mcts_invalid_max_rollout_steps_validation(bad_steps):
    """Kiểm tra max_rollout_steps phải là số nguyên dương và không phải bool."""
    with pytest.raises(ValueError):
        MCTSAgent(max_rollout_steps=bad_steps)


def test_mcts_budget_modes():
    """
    Kiểm chứng MF-1:
    1. Mặc định khi không truyền cả hai -> num_rollouts = 300, time_limit_sec = None.
    2. Chỉ đặt num_rollouts -> dừng đúng số rollout.
    3. Chỉ đặt time_limit_sec -> num_rollouts là None, dừng theo thời gian.
    4. Đặt cả hai -> dừng khi chạm giới hạn nào đến trước.
    """
    env = _build_midgame_env()

    default_agent = MCTSAgent(seed=42)
    assert default_agent.num_rollouts == 300
    assert default_agent.time_limit_sec is None

    agent_rollout_only = MCTSAgent(num_rollouts=25, seed=42)
    assert agent_rollout_only.time_limit_sec is None
    action = agent_rollout_only.select_action(env)
    assert action in env.legal_actions()
    assert agent_rollout_only.last_search_count == 25

    agent_time_only = MCTSAgent(time_limit_sec=0.04, seed=42)
    assert agent_time_only.num_rollouts is None
    action = agent_time_only.select_action(env)
    assert action in env.legal_actions()
    assert agent_time_only.last_search_count >= 1

    agent_both_rollout_cap = MCTSAgent(num_rollouts=12, time_limit_sec=5.0, seed=42)
    agent_both_rollout_cap.select_action(env)
    assert agent_both_rollout_cap.last_search_count == 12

    agent_both_time_cap = MCTSAgent(num_rollouts=100000, time_limit_sec=0.03, seed=42)
    agent_both_time_cap.select_action(env)
    assert 1 <= agent_both_time_cap.last_search_count < 100000


def test_mcts_zero_rollout_on_forced_and_shortcut():
    """
    Kiểm chứng MF-2 & SF-1:
    - Khi forced move hoặc tactical shortcut kích hoạt, không chạy rollout (last_search_count == 0),
      không tạo root giả (last_root is None), và vẫn cập nhật last_move_time_ms > 0.0.
    - Reset đúng trạng thái từ lượt tìm kiếm trước đó.
    """
    agent = MCTSAgent(num_rollouts=20, use_tactical_shortcut=True, seed=42)

    mid_env = _build_midgame_env()
    agent.select_action(mid_env)
    assert agent.last_search_count == 20
    assert agent.last_root is not None

    empty_env = CaroEnv()
    forced_action = agent.select_action(empty_env)
    assert forced_action == 60
    assert agent.last_search_count == 0
    assert agent.last_root is None
    assert agent.last_move_time_ms > 0.0

    win_env = _build_four_in_a_row_env_for_x()
    board_before = win_env.board.copy()
    urgent_action = agent.select_action(win_env)
    assert urgent_action == 5 * 11 + 4
    assert agent.last_search_count == 0
    assert agent.last_root is None
    assert agent.last_move_time_ms > 0.0
    assert np.array_equal(win_env.board, board_before)


def test_mcts_last_root_none_safety_and_visit_invariants():
    """
    Kiểm chứng MF-3 & OPT-1:
    - Khi last_root là None, get_root_stats() trả về [] an toàn.
    - Khi chạy MCTS bình thường với num_rollouts = N, last_root.visits == N và tổng visits các con == N.
    """
    agent = MCTSAgent(num_rollouts=30, seed=42)

    empty_env = CaroEnv()
    agent.select_action(empty_env)
    assert agent.last_root is None
    assert agent.get_root_stats() == []

    mid_env = _build_midgame_env()
    board_snapshot = mid_env.board.copy()
    history_snapshot = mid_env.history.copy()

    action = agent.select_action(mid_env)
    assert isinstance(action, int)
    assert action in mid_env.legal_actions()
    assert np.array_equal(mid_env.board, board_snapshot)
    assert mid_env.history == history_snapshot

    assert agent.last_root is not None
    assert agent.last_search_count == 30
    assert agent.last_root.visits == 30
    children_visits = sum(child.visits for child in agent.last_root.children.values())
    assert children_visits == 30

    stats = agent.get_root_stats()
    assert len(stats) == len(agent.last_root.children)
    assert sum(item["visits"] for item in stats) == 30
    assert stats[0]["action"] == action


def test_mcts_tactical_shortcut_toggle_and_rollout_radius():
    """
    Kiểm chứng SF-2 & Công tắc use_tactical_shortcut:
    - Lưu đúng tham số rollout_radius.
    - Khi tắt use_tactical_shortcut=False trên thế cờ X có 4 quân liên tiếp (current_player == 1),
      MCTS không thoát sớm mà chạy đủ num_rollouts và tạo cây last_root.
    """
    win_env = _build_four_in_a_row_env_for_x()
    agent_no_shortcut = MCTSAgent(
        num_rollouts=35,
        rollout_radius=2,
        use_tactical_shortcut=False,
        seed=42,
    )
    assert agent_no_shortcut.rollout_radius == 2

    action = agent_no_shortcut.select_action(win_env)
    assert action in win_env.legal_actions()
    assert agent_no_shortcut.last_search_count == 35
    assert agent_no_shortcut.last_root is not None
    assert agent_no_shortcut.last_root.visits == 35
    assert agent_no_shortcut.last_move_time_ms > 0.0