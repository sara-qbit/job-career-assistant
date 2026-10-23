"""Prompts live as one Markdown file each in this folder and are loaded by name."""

import re
from pathlib import Path

from turing_projects_v2.errors import ConfigError

PROMPTS_DIR = Path(__file__).parent
PROMPT_NAME = re.compile(r"\w+")


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Return the text of `<prompts_dir>/<name>.md`."""
    if not PROMPT_NAME.fullmatch(name):
        raise ConfigError(f"Invalid prompt name '{name}': use letters, digits and underscores.")
    path = prompts_dir / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"Prompt '{name}' not found: {path}") from error
