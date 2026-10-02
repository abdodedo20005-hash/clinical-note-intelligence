"""Runtime configuration, read from environment variables / a .env file."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    llm_concurrency: int = 5
    note_chars: int = 3000
    model_dir: Path = ROOT / "models"
    max_batch_size: int = 50
    max_note_length: int = 100_000


settings = Settings()
