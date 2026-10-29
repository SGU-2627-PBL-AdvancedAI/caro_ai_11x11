from agents.base_agent import BaseAgent

def test_base_agent_interface():
    assert hasattr(BaseAgent, "select_action")