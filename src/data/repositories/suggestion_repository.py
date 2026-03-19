from datetime import datetime, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data.repositories.base import BaseRepository


class SuggestionRepository(BaseRepository):
    collection_name = "suggestions"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        super().__init__(db)

    async def get_by_id(self, suggestion_id: str) -> dict | None:
        return await self.collection.find_one(
            {"suggestion_id": suggestion_id},
            {"_id": 0},
        )

    async def list_all(
        self,
        status: Optional[str] = None,
        type: Optional[str] = None,
        agent_version: Optional[str] = None,
    ) -> list[dict]:
        query: dict = {}
        if status:
            query["status"] = status
        if type:
            query["type"] = type
        if agent_version:
            query["agent_version"] = agent_version
        cursor = self.collection.find(query, {"_id": 0}).sort("created_at", -1)
        return await cursor.to_list(length=100)

    async def list_by_conversation_id(self, conversation_id: str) -> list[dict]:
        cursor = self.collection.find(
            {"conversation_ids": conversation_id},
            {"_id": 0},
        ).sort("created_at", -1)
        return await cursor.to_list(length=100)

    async def update_status(self, suggestion_id: str, status: str) -> dict | None:
        resolved_at = datetime.now(timezone.utc) if status in ("applied", "rejected", "expired") else None
        update: dict = {"$set": {"status": status}}
        if resolved_at:
            update["$set"]["resolved_at"] = resolved_at
        result = await self.collection.find_one_and_update(
            {"suggestion_id": suggestion_id},
            update,
            return_document=True,
            projection={"_id": 0},
        )
        return result
