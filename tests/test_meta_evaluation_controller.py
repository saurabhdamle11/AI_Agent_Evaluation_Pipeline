import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.schemas.meta_evaluation import (
    BlindSpot,
    CalibrationAdjustment,
    EvaluatorMetrics,
    MetaEvaluationResult,
)
from src.utils.dependencies import (
    get_evaluation_repo,
    get_feedback_repo,
    get_meta_evaluation_repo,
)

META_EVAL_ID = "meta-eval-abc-123"
NOW = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def make_result(**overrides) -> MetaEvaluationResult:
    base = dict(
        meta_evaluation_id=META_EVAL_ID,
        evaluator_metrics=[
            EvaluatorMetrics(
                evaluator_type="llm_judge",
                precision=0.8,
                recall=0.75,
                f1_score=0.77,
                correlation_with_human=0.85,
                sample_size=50,
            )
        ],
        blind_spots=[
            BlindSpot(annotation_type="accuracy", miss_rate=0.4, example_count=8)
        ],
        calibration_adjustments=[
            CalibrationAdjustment(
                evaluator_type="llm_judge",
                weight_multiplier=1.2,
                reason="F1 above 0.7 — evaluator reliable",
            )
        ],
        sample_size=50,
        created_at=NOW,
    )
    base.update(overrides)
    return MetaEvaluationResult(**base)


@pytest.fixture
def mock_meta_repo():
    repo = MagicMock()
    repo.get_latest = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.insert = AsyncMock(return_value={})
    return repo


@pytest.fixture
def mock_eval_repo():
    repo = MagicMock()
    repo.get_all = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_feedback_repo():
    repo = MagicMock()
    repo.get_all_with_aggregated = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def client_with_repos(client, mock_meta_repo, mock_eval_repo, mock_feedback_repo):
    from src.main import app
    app.dependency_overrides[get_meta_evaluation_repo] = lambda: mock_meta_repo
    app.dependency_overrides[get_evaluation_repo] = lambda: mock_eval_repo
    app.dependency_overrides[get_feedback_repo] = lambda: mock_feedback_repo
    yield client, mock_meta_repo, mock_eval_repo, mock_feedback_repo
    app.dependency_overrides.pop(get_meta_evaluation_repo, None)
    app.dependency_overrides.pop(get_evaluation_repo, None)
    app.dependency_overrides.pop(get_feedback_repo, None)


# ── POST /api/v1/meta-evaluations/run ────────────────────────────────────────

class TestRunMetaEvaluation:
    async def test_returns_200_with_result(self, client_with_repos):
        client, _, _, _ = client_with_repos
        with patch(
            "src.controllers.meta_evaluation_controller.MetaEvaluationService.run",
            new_callable=AsyncMock,
            return_value=make_result(),
        ):
            resp = await client.post("/api/v1/meta-evaluations/run")
        assert resp.status_code == 200
        assert resp.json()["meta_evaluation_id"] == META_EVAL_ID

    async def test_response_includes_evaluator_metrics(self, client_with_repos):
        client, _, _, _ = client_with_repos
        with patch(
            "src.controllers.meta_evaluation_controller.MetaEvaluationService.run",
            new_callable=AsyncMock,
            return_value=make_result(),
        ):
            resp = await client.post("/api/v1/meta-evaluations/run")
        metrics = resp.json()["evaluator_metrics"]
        assert len(metrics) == 1
        assert metrics[0]["evaluator_type"] == "llm_judge"
        assert metrics[0]["f1_score"] == pytest.approx(0.77)

    async def test_response_includes_blind_spots(self, client_with_repos):
        client, _, _, _ = client_with_repos
        with patch(
            "src.controllers.meta_evaluation_controller.MetaEvaluationService.run",
            new_callable=AsyncMock,
            return_value=make_result(),
        ):
            resp = await client.post("/api/v1/meta-evaluations/run")
        blind_spots = resp.json()["blind_spots"]
        assert len(blind_spots) == 1
        assert blind_spots[0]["annotation_type"] == "accuracy"

    async def test_response_includes_calibration_adjustments(self, client_with_repos):
        client, _, _, _ = client_with_repos
        with patch(
            "src.controllers.meta_evaluation_controller.MetaEvaluationService.run",
            new_callable=AsyncMock,
            return_value=make_result(),
        ):
            resp = await client.post("/api/v1/meta-evaluations/run")
        adj = resp.json()["calibration_adjustments"]
        assert adj[0]["weight_multiplier"] == pytest.approx(1.2)

    async def test_response_includes_sample_size(self, client_with_repos):
        client, _, _, _ = client_with_repos
        with patch(
            "src.controllers.meta_evaluation_controller.MetaEvaluationService.run",
            new_callable=AsyncMock,
            return_value=make_result(sample_size=42),
        ):
            resp = await client.post("/api/v1/meta-evaluations/run")
        assert resp.json()["sample_size"] == 42

    async def test_empty_result_when_no_data(self, client_with_repos):
        client, _, _, _ = client_with_repos
        with patch(
            "src.controllers.meta_evaluation_controller.MetaEvaluationService.run",
            new_callable=AsyncMock,
            return_value=make_result(
                evaluator_metrics=[], blind_spots=[], calibration_adjustments=[], sample_size=0
            ),
        ):
            resp = await client.post("/api/v1/meta-evaluations/run")
        assert resp.status_code == 200
        assert resp.json()["sample_size"] == 0


# ── GET /api/v1/meta-evaluations/latest ──────────────────────────────────────

class TestGetLatestMetaEvaluation:
    async def test_returns_200_with_latest(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_latest.return_value = make_result().model_dump()
        resp = await client.get("/api/v1/meta-evaluations/latest")
        assert resp.status_code == 200
        assert resp.json()["meta_evaluation_id"] == META_EVAL_ID

    async def test_returns_404_when_none_exist(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_latest.return_value = None
        resp = await client.get("/api/v1/meta-evaluations/latest")
        assert resp.status_code == 404
        assert "been run" in resp.json()["detail"]

    async def test_queries_repo_for_latest(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_latest.return_value = make_result().model_dump()
        await client.get("/api/v1/meta-evaluations/latest")
        meta_repo.get_latest.assert_awaited_once()


# ── GET /api/v1/meta-evaluations/{id} ────────────────────────────────────────

class TestGetMetaEvaluationById:
    async def test_returns_200_with_result(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_by_id.return_value = make_result().model_dump()
        resp = await client.get(f"/api/v1/meta-evaluations/{META_EVAL_ID}")
        assert resp.status_code == 200
        assert resp.json()["meta_evaluation_id"] == META_EVAL_ID

    async def test_returns_404_when_not_found(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_by_id.return_value = None
        resp = await client.get(f"/api/v1/meta-evaluations/{META_EVAL_ID}")
        assert resp.status_code == 404

    async def test_queries_repo_with_correct_id(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_by_id.return_value = make_result().model_dump()
        await client.get(f"/api/v1/meta-evaluations/{META_EVAL_ID}")
        meta_repo.get_by_id.assert_awaited_once_with(META_EVAL_ID)

    async def test_response_includes_all_sections(self, client_with_repos):
        client, meta_repo, _, _ = client_with_repos
        meta_repo.get_by_id.return_value = make_result().model_dump()
        resp = await client.get(f"/api/v1/meta-evaluations/{META_EVAL_ID}")
        body = resp.json()
        for key in ("evaluator_metrics", "blind_spots", "calibration_adjustments", "sample_size"):
            assert key in body
