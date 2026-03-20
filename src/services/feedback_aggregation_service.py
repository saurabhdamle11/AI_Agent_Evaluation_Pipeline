"""
Feedback aggregation: inter-annotator agreement, confidence weighting, routing.

For each annotation type on a conversation, multiple annotators may submit
different labels. This service:
  1. Finds the consensus label via confidence-weighted majority vote.
  2. Computes pairwise Cohen's kappa as an agreement score.
  3. Routes the conversation to "auto_label" (high agreement) or
     "human_review" (low agreement).
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Optional

from src.data.schemas.feedback import AggregatedFeedback, AnnotationConsensus


# Conversations where any annotation type falls below this kappa threshold
# will be sent to the human review queue.
KAPPA_THRESHOLD = 0.6


def _pairwise_kappa(labels: list[str]) -> float:
    """
    Cohen's kappa extended to multiple raters via pairwise averaging.

    With n raters and their labels:
      - Po (observed agreement): fraction of all pairs that chose the same label.
      - Pe (expected agreement by chance): sum of squared proportions per category.
      - kappa = (Po - Pe) / (1 - Pe)

    Returns 1.0 when there is only one rater (perfect agreement by default).
    """
    n = len(labels)
    if n <= 1:
        return 1.0

    # Observed agreement across all annotator pairs
    total_pairs = n * (n - 1) / 2
    agreeing_pairs = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if labels[i] == labels[j]
    )
    po = agreeing_pairs / total_pairs

    # Expected agreement by chance
    counts = Counter(labels)
    pe = sum((c / n) ** 2 for c in counts.values())

    if pe >= 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def _weighted_majority_label(
    labels: list[str],
    confidences: list[Optional[float]],
) -> tuple[str, float]:
    """
    Choose the label with the highest sum of confidence weights.

    Annotators without a confidence value are assigned a neutral weight of 0.5.
    Returns (consensus_label, weighted_confidence) where weighted_confidence is
    the mean confidence across all annotators who chose the consensus label.
    """
    weight_sums: dict[str, float] = defaultdict(float)
    weight_counts: dict[str, int] = defaultdict(int)

    for label, conf in zip(labels, confidences):
        w = conf if conf is not None else 0.5
        weight_sums[label] += w
        weight_counts[label] += 1

    consensus_label = max(weight_sums, key=weight_sums.__getitem__)
    mean_conf = weight_sums[consensus_label] / weight_counts[consensus_label]
    return consensus_label, round(mean_conf, 4)


class FeedbackAggregationService:
    @staticmethod
    def aggregate(feedback: dict) -> AggregatedFeedback:
        """
        Aggregate raw feedback into consensus labels + routing decision.

        Parameters
        ----------
        feedback:
            A raw feedback document as stored in MongoDB (dict form of
            FeedbackIngestionRequest plus timestamps).

        Returns
        -------
        AggregatedFeedback
        """
        annotations: list[dict] = feedback.get("annotations") or []

        # Group annotations by type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for ann in annotations:
            by_type[ann["type"]].append(ann)

        consensus_list: list[AnnotationConsensus] = []
        kappa_values: list[float] = []

        for ann_type, anns in by_type.items():
            labels = [a["label"] for a in anns]
            confidences = [a.get("confidence") for a in anns]

            kappa = _pairwise_kappa(labels)
            agreement_score = round(
                Counter(labels).most_common(1)[0][1] / len(labels), 4
            )
            consensus_label, weighted_confidence = _weighted_majority_label(
                labels, confidences
            )

            consensus_list.append(
                AnnotationConsensus(
                    type=ann_type,
                    consensus_label=consensus_label,
                    agreement_score=agreement_score,
                    kappa=round(kappa, 4),
                    weighted_confidence=weighted_confidence,
                    annotator_count=len(anns),
                )
            )
            kappa_values.append(kappa)

        overall_kappa = round(
            sum(kappa_values) / len(kappa_values) if kappa_values else 1.0, 4
        )
        routing = (
            "auto_label"
            if overall_kappa >= KAPPA_THRESHOLD
            else "human_review"
        )

        ops_review = feedback.get("ops_review")

        return AggregatedFeedback(
            conversation_id=feedback["conversation_id"],
            user_rating=feedback.get("user_rating"),
            ops_review=ops_review,
            annotation_consensus=consensus_list,
            overall_kappa=overall_kappa,
            routing=routing,
            aggregated_at=datetime.now(timezone.utc),
        )
