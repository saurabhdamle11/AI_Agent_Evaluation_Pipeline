from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data.repositories.base import BaseRepository


class MetaEvaluationRepository(BaseRepository):
    collection_name = "meta_evaluations"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        super().__init__(db)

    async def insert(self, data: dict) -> dict:
        await self.collection.insert_one(data)
        data.pop("_id", None)
        return data

    async def get_by_id(self, meta_evaluation_id: str) -> dict | None:
        return await self.collection.find_one(
            {"meta_evaluation_id": meta_evaluation_id},
            {"_id": 0},
        )

    async def get_latest(self) -> dict | None:
        cursor = self.collection.find({}, {"_id": 0}).sort("created_at", -1).limit(1)
        results = await cursor.to_list(length=1)
        return results[0] if results else None
