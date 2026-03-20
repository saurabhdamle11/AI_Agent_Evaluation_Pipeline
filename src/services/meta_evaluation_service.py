"""
Meta-evaluation service: evaluates the evaluators.

For every conversation that has both an automated EvaluationResult and a
processed AggregatedFeedback (with human signals), this service:

  1. Pairs them on conversation_id.
  2. Derives a human ground-truth score from user_rating and ops_review.quality.
  3. Computes per-evaluator precision, recall, F1 (binary: good vs. bad) and
     Pearson correlation against the human score.
  4. Detects blind spots — annotation types that appear in conversations the
     evaluators rated as good but humans rated as bad.
  5. Generates calibration adjustments — weight multipliers based on F1.

Terminology
-----------
"missed" conversation:
    overall automated score >= GOOD_THRESHOLD  (evaluator says good)
    human score             <  GOOD_THRESHOLD  (human says bad)

Evaluator-score → evaluator-type mapping
-----------------------------------------
    response_quality  →  llm_judge
    tool_accuracy     →  tool_evaluator
    coherence         →  coherence
    heuristic         →  heuristic
"""
import logging
import uuid
from datetime import datetime, timezone
from collections import defaultdict

from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.repositories.feedback_repository import FeedbackRepository
from src.data.repositories.meta_evaluation_repository import MetaEvaluationRepository
from src.data.schemas.meta_evaluation import (
    BlindSpot,
    CalibrationAdjustment,
    EvaluatorMetrics,
    MetaEvaluationResult,
)

logger = logging.getLogger(__name__)

GOOD_THRESHOLD = 0.5   # scores >= this → "good"; < this → "bad"
LIMIT = 1000           # max conversations to pull per run

_QUALITY_TO_SCORE: dict[str, float] = {
    "poor": 0.0,
    "fair": 0.33,
    "good": 0.67,
    "excellent": 1.0,
}

# Maps score-dict key → evaluator type label
_SCORE_KEY_TO_EVALUATOR: dict[str, str] = {
    "response_quality": "llm_judge",
    "tool_accuracy": "tool_evaluator",
    "coherence": "coherence",
    "heuristic": "heuristic",
}

# Calibration: F1 band → weight multiplier
_CALIBRATION_BANDS: list[tuple[float, float, str]] = [
    (0.0,  0.5,  "0.7",   "F1 below 0.5 — evaluator unreliable"),
    (0.5,  0.7,  "1.0",   "F1 between 0.5 and 0.7 — evaluator adequate"),
    (0.7,  1.01, "1.2",   "F1 above 0.7 — evaluator reliable"),
]


# ── pure maths ────────────────────────────────────────────────────────────────

def _human_score(feedback_doc: dict) -> float | None:
    """
    Derive a 0-1 ground-truth score from human feedback signals.
    Returns None when no human signal is available.
    """
    parts: list[float] = []

    user_rating = feedback_doc.get("user_rating")
    if user_rating is not None:
        parts.append((float(user_rating) - 1) / 4.0)

    aggregated = feedback_doc.get("aggregated") or {}
    ops_review = aggregated.get("ops_review") or {}
    quality = ops_review.get("quality") if isinstance(ops_review, dict) else None
    if quality and quality in _QUALITY_TO_SCORE:
        parts.append(_QUALITY_TO_SCORE[quality])

    return round(sum(parts) / len(parts), 4) if parts else None


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) ** 0.5) * (sum((y - my) ** 2 for y in ys) ** 0.5)
    return round(num / den, 4) if den else 0.0


def _prf(preds: list[float], truths: list[float]) -> tuple[float, float, float]:
    """
    Binary precision / recall / F1 where the positive class is "bad" (score < GOOD_THRESHOLD).
    """
    tp = fp = fn = 0
    for p, t in zip(preds, truths):
        pred_bad, true_bad = p < GOOD_THRESHOLD, t < GOOD_THRESHOLD
        if pred_bad and true_bad:
            tp += 1
        elif pred_bad and not true_bad:
            fp += 1
        elif not pred_bad and true_bad:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _weight_multiplier(f1: float) -> tuple[float, str]:
    for lo, hi, mult, reason in _CALIBRATION_BANDS:
        if lo <= f1 < hi:
            return float(mult), reason
    return 1.0, "unknown"


# ── service ───────────────────────────────────────────────────────────────────

class MetaEvaluationService:
    @staticmethod
    async def run(
        evaluation_repo: EvaluationRepository,
        feedback_repo: FeedbackRepository,
        meta_eval_repo: MetaEvaluationRepository,
    ) -> MetaEvaluationResult:
        evaluations = await evaluation_repo.get_all(limit=LIMIT)
        feedbacks_with_aggregated = await feedback_repo.get_all_with_aggregated(limit=LIMIT)

        # Index feedback by conversation_id for O(1) lookup
        feedback_by_conv: dict[str, dict] = {
            f["conversation_id"]: f for f in feedbacks_with_aggregated
        }

        # Pair evaluations with human feedback
        pairs: list[tuple[dict, float]] = []  # (eval_doc, human_score)
        for ev in evaluations:
            cid = ev.get("conversation_id")
            fb = feedback_by_conv.get(cid)
            if not fb:
                continue
            hs = _human_score(fb)
            if hs is None:
                continue
            pairs.append((ev, hs))

        logger.info(f"MetaEvaluationService: {len(pairs)} paired conversations")

        evaluator_metrics = MetaEvaluationService._compute_evaluator_metrics(pairs)
        blind_spots = MetaEvaluationService._detect_blind_spots(pairs, feedback_by_conv)
        calibration = MetaEvaluationService._calibration_adjustments(evaluator_metrics)

        result = MetaEvaluationResult(
            meta_evaluation_id=str(uuid.uuid4()),
            evaluator_metrics=evaluator_metrics,
            blind_spots=blind_spots,
            calibration_adjustments=calibration,
            sample_size=len(pairs),
            created_at=datetime.now(timezone.utc),
        )
        await meta_eval_repo.insert(result.model_dump())
        return result

    @staticmethod
    def _compute_evaluator_metrics(
        pairs: list[tuple[dict, float]],
    ) -> list[EvaluatorMetrics]:
        # Bucket (evaluator_score, human_score) by evaluator type
        buckets: dict[str, tuple[list[float], list[float]]] = {
            ev_type: ([], []) for ev_type in _SCORE_KEY_TO_EVALUATOR.values()
        }
        for ev, hs in pairs:
            scores: dict = (ev.get("scores") or {})
            for score_key, ev_type in _SCORE_KEY_TO_EVALUATOR.items():
                value = scores.get(score_key)
                if value is not None:
                    buckets[ev_type][0].append(float(value))
                    buckets[ev_type][1].append(hs)

        metrics: list[EvaluatorMetrics] = []
        for ev_type, (preds, truths) in buckets.items():
            if not preds:
                continue
            precision, recall, f1 = _prf(preds, truths)
            corr = _pearson(preds, truths)
            metrics.append(EvaluatorMetrics(
                evaluator_type=ev_type,
                precision=precision,
                recall=recall,
                f1_score=f1,
                correlation_with_human=corr,
                sample_size=len(preds),
            ))
        return metrics

    @staticmethod
    def _detect_blind_spots(
        pairs: list[tuple[dict, float]],
        feedback_by_conv: dict[str, dict],
    ) -> list[BlindSpot]:
        # "missed" = evaluator overall >= GOOD_THRESHOLD but human < GOOD_THRESHOLD
        annotation_total: dict[str, int] = defaultdict(int)
        annotation_missed: dict[str, int] = defaultdict(int)

        for ev, hs in pairs:
            cid = ev.get("conversation_id")
            fb = feedback_by_conv.get(cid, {})
            annotations = fb.get("annotations") or []
            overall = (ev.get("scores") or {}).get("overall")
            is_missed = (
                overall is not None
                and overall >= GOOD_THRESHOLD
                and hs < GOOD_THRESHOLD
            )
            for ann in annotations:
                ann_type = ann.get("type")
                if ann_type:
                    annotation_total[ann_type] += 1
                    if is_missed:
                        annotation_missed[ann_type] += 1

        blind_spots: list[BlindSpot] = []
        for ann_type, missed_count in annotation_missed.items():
            total = annotation_total[ann_type]
            miss_rate = round(missed_count / total, 4) if total else 0.0
            if miss_rate > 0:
                blind_spots.append(BlindSpot(
                    annotation_type=ann_type,
                    miss_rate=miss_rate,
                    example_count=missed_count,
                ))

        return sorted(blind_spots, key=lambda b: b.miss_rate, reverse=True)

    @staticmethod
    def _calibration_adjustments(
        metrics: list[EvaluatorMetrics],
    ) -> list[CalibrationAdjustment]:
        adjustments: list[CalibrationAdjustment] = []
        for m in metrics:
            multiplier, reason = _weight_multiplier(m.f1_score)
            adjustments.append(CalibrationAdjustment(
                evaluator_type=m.evaluator_type,
                weight_multiplier=multiplier,
                reason=reason,
            ))
        return adjustments
