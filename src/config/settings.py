from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # App
    app_name: str = "AI Agent Evaluation Pipeline"
    app_env: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # MongoDB
    mongo_uri: str
    mongo_db_name: str = "eval_pipeline"

    # LLM Judge
    llm_judge_model: str = "claude-sonnet-4-6"

    # Kafka
    kafka_bootstrap_servers: str
    kafka_topic_conversations: str = "conversations.raw"
    kafka_topic_feedback: str = "feedback.raw"
    kafka_topic_feedback_processed: str = "feedback.processed"
    kafka_topic_evaluations: str = "evaluations.trigger"
    kafka_consumer_group: str = "eval-pipeline-consumers"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
