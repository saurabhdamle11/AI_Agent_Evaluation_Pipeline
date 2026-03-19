from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data.database import get_db
from src.data.repositories.conversation_repository import ConversationRepository
from src.data.repositories.feedback_repository import FeedbackRepository


def get_conversation_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(db)


def get_feedback_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> FeedbackRepository:
    return FeedbackRepository(db)
