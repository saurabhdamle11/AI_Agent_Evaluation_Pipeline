from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from src.data.repositories.base import BaseRepository


class FeedbackRepository(BaseRepository):
    collection_name = "feedback"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        super().__init__(db)

    async def upsert(self, data: dict) -> dict:
        now = datetime.now(timezone.utc)
        data["updated_at"] = now
        data.setdefault("created_at", now)
        return await self.collection.find_one_and_replace(
            {"conversation_id": data["conversation_id"]},
            data,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    async def get_by_conversation_id(self, conversation_id: str) -> dict | None:
        return await self.collection.find_one(
            {"conversation_id": conversation_id},
            {"_id": 0},
        )

    async def get_all_with_aggregated(self, limit: int = 1000) -> list[dict]:
        cursor = self.collection.find(
            {"aggregated": {"$exists": True}},
            {"_id": 0},
        )
        return await cursor.to_list(length=limit)

    async def update_aggregated(self, conversation_id: str, aggregated: dict) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"aggregated": aggregated, "updated_at": now}},
        )
