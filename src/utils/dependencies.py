from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.data.database import get_db
from src.data.repositories.conversation_repository import ConversationRepository
from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.repositories.feedback_repository import FeedbackRepository
from src.data.repositories.meta_evaluation_repository import MetaEvaluationRepository
from src.data.repositories.suggestion_repository import SuggestionRepository


def get_conversation_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(db)


def get_feedback_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> FeedbackRepository:
    return FeedbackRepository(db)


def get_evaluation_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> EvaluationRepository:
    return EvaluationRepository(db)


def get_meta_evaluation_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> MetaEvaluationRepository:
    return MetaEvaluationRepository(db)


def get_suggestion_repo(db: AsyncIOMotorDatabase = Depends(get_db)) -> SuggestionRepository:
    return SuggestionRepository(db)
