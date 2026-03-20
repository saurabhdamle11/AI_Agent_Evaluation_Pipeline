import asyncio
import json
import logging
from abc import ABC, abstractmethod

from aiokafka import AIOKafkaConsumer

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseWorker(ABC):
    def __init__(self, topics: list[str]) -> None:
        self.topics = topics
        self._consumer: AIOKafkaConsumer | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_consumer_group,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        try:
            await self._consumer.start()
            logger.info(f"{self.__class__.__name__} consuming: {self.topics}")
            async for msg in self._consumer:
                try:
                    await self.handle(msg.value)
                except Exception as e:
                    logger.error(f"{self.__class__.__name__} failed to handle message: {e}")
        except asyncio.CancelledError:
            logger.info(f"{self.__class__.__name__} shutting down")
        finally:
            await self._consumer.stop()

    @abstractmethod
    async def handle(self, payload: dict) -> None: ...
