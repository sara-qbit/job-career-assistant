"""Typed settings read from the environment file (.env)."""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from turing_projects_v2.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    agent_model: str = "openai/gpt-6-luna"
    judge_model: str = "anthropic/claude-haiku-5.5"
    embedding_model: str = "gemini"

    data_dir: Path = PROJECT_ROOT / "data"
    kaggle_dir: Path = PROJECT_ROOT / "data" / "raw" / "linkedin"
    database_path: Path = PROJECT_ROOT / "data" / "jobs.sqlite"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    conversations_path: Path = PROJECT_ROOT / "data" / "conversations.sqlite"
    candidate_dir: Path = PROJECT_ROOT / "candidate"
    title_variants_path: Path = PROJECT_ROOT / "candidate" / "title_variants.toml"

    max_tool_calls: int = 6
    history_window: int = 20

    @field_validator("openrouter_api_key")
    @classmethod
    def key_is_not_empty(cls, key: SecretStr) -> SecretStr:
        if not key.get_secret_value().strip():
            raise ValueError("is empty")
        return key


def load_settings(env_file: Path = PROJECT_ROOT / ".env") -> Settings:
    """Read settings from the environment and `env_file`; fail clearly if the key is missing."""
    try:
        return Settings(_env_file=env_file)
    except PydanticValidationError as error:
        problems = "; ".join(
            f"{str(problem['loc'][0]).upper()} {problem['msg'].lower()}"
            for problem in error.errors()
        )
        raise ConfigError(
            f"Invalid settings ({problems}). Copy .env.example to .env and fill it in."
        ) from error
