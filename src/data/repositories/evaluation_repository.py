from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data.repositories.base import BaseRepository


class EvaluationRepository(BaseRepository):
    collection_name = "evaluations"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        super().__init__(db)

    async def insert(self, data: dict) -> dict:
        await self.collection.insert_one(data)
        data.pop("_id", None)
        return data

    async def get_by_id(self, evaluation_id: str) -> dict | None:
        return await self.collection.find_one(
            {"evaluation_id": evaluation_id},
            {"_id": 0},
        )

    async def get_by_conversation_id(self, conversation_id: str) -> list[dict]:
        cursor = self.collection.find(
            {"conversation_id": conversation_id},
            {"_id": 0},
        )
        return await cursor.to_list(length=100)

    async def get_all(self, limit: int = 1000) -> list[dict]:
        cursor = self.collection.find({}, {"_id": 0}).sort("created_at", -1)
        return await cursor.to_list(length=limit)
