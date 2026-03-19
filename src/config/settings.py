from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "AI Agent Evaluation Pipeline"
    app_env: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # MongoDB
    mongo_uri: str
    mongo_db_name: str = "eval_pipeline"

    # Kafka
    kafka_bootstrap_servers: str
    kafka_topic_conversations: str = "conversations.raw"
    kafka_topic_feedback: str = "feedback.raw"
    kafka_topic_evaluations: str = "evaluations.trigger"
    kafka_consumer_group: str = "eval-pipeline-consumers"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
