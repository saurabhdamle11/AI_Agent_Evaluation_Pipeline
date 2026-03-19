from fastapi import APIRouter, status

from src.data.schemas.ingestion import (
    BatchIngestionRequest,
    BatchIngestionResponse,
    ConversationIngestionRequest,
    FeedbackIngestionRequest,
    IngestionResponse,
)
from src.services.ingestion_service import IngestionService

router = APIRouter()


@router.post(
    "/conversation",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a single conversation",
)
async def ingest_conversation(payload: ConversationIngestionRequest) -> IngestionResponse:
    return await IngestionService.ingest_conversation(payload)


@router.post(
    "/conversations/batch",
    response_model=BatchIngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a batch of conversations",
)
async def ingest_batch(payload: BatchIngestionRequest) -> BatchIngestionResponse:
    return await IngestionService.ingest_batch(payload)


@router.post(
    "/feedback",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest feedback for a conversation",
)
async def ingest_feedback(payload: FeedbackIngestionRequest) -> IngestionResponse:
    return await IngestionService.ingest_feedback(payload)
