"""Configuration management using pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM Configuration
    llm_mode: str = "local"  # local or hosted
    local_model_path: str = "./models/qwen2.5-7b-instruct.Q4_K_M.gguf"
    openai_api_key: str = ""

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval Settings
    top_k: int = 3
    chunk_size: int = 500
    chunk_overlap: int = 50

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths
    kb_path: Path = Path("./data/kb")
    index_path: Path = Path("./data/index")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure paths exist
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()