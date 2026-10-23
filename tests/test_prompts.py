import pytest

from turing_projects_v2.errors import ConfigError
from turing_projects_v2.prompts import load_prompt


def test_prompt_is_loaded_by_name_from_its_own_file(tmp_path):
    (tmp_path / "greeting.md").write_text("You help with job applications.\n", encoding="utf-8")

    assert load_prompt("greeting", tmp_path) == "You help with job applications.\n"


def test_system_prompt_ships_with_the_app():
    assert "job" in load_prompt("system").lower()


def test_unknown_prompt_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="no_such_prompt"):
        load_prompt("no_such_prompt", tmp_path)


@pytest.mark.parametrize("name", ["../secrets", "sub/system", ""])
def test_prompt_names_cannot_point_outside_the_prompt_folder(tmp_path, name):
    with pytest.raises(ConfigError):
        load_prompt(name, tmp_path)
