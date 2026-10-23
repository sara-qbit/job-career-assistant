import pytest

from turing_projects_v2.config import load_settings
from turing_projects_v2.errors import ConfigError


@pytest.fixture(autouse=True)
def no_key_in_environment(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_settings_are_read_from_the_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-test\nMAX_TOOL_CALLS=3\n")

    settings = load_settings(env_file)

    assert settings.openrouter_api_key.get_secret_value() == "sk-test"
    assert settings.max_tool_calls == 3
    assert settings.agent_model == "openai/gpt-6-luna"
    assert settings.judge_model == "anthropic/claude-haiku-5.5"
    assert settings.history_window == 20


def test_missing_key_fails_with_a_clear_config_error(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("AGENT_MODEL=some/free-model\n")

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_settings(env_file)


def test_key_is_not_revealed_when_settings_are_printed(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-secret\n")

    assert "sk-secret" not in repr(load_settings(env_file))


def test_empty_key_copied_from_the_example_file_fails_clearly(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=\n")

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        load_settings(env_file)
