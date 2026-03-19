import json
import logging

from aiokafka import AIOKafkaProducer

from src.config.settings import get_settings
from src.utils.exceptions import KafkaPublishError

logger = logging.getLogger(__name__)
settings = get_settings()


class KafkaProducerService:
    _producer: AIOKafkaProducer | None = None

    @classmethod
    async def start(cls) -> None:
        cls._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        await cls._producer.start()
        logger.info("Kafka producer started")

    @classmethod
    async def stop(cls) -> None:
        if cls._producer:
            await cls._producer.stop()
            logger.info("Kafka producer stopped")

    @classmethod
    async def publish(cls, topic: str, value: dict, key: str | None = None) -> None:
        if cls._producer is None:
            raise KafkaPublishError("Kafka producer is not initialized")
        try:
            await cls._producer.send_and_wait(
                topic,
                value=value,
                key=key.encode("utf-8") if key else None,
            )
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            raise KafkaPublishError(str(e)) from e
