from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from src.data.repositories.base import BaseRepository


class ConversationRepository(BaseRepository):
    collection_name = "conversations"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        super().__init__(db)

    async def upsert(self, data: dict) -> dict:
        data["ingested_at"] = datetime.now(timezone.utc)
        return await self.collection.find_one_and_replace(
            {"conversation_id": data["conversation_id"]},
            data,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    async def get_by_id(self, conversation_id: str) -> dict | None:
        return await self.collection.find_one(
            {"conversation_id": conversation_id},
            {"_id": 0},
        )
