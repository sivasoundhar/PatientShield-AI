"""Compares PHI Agent's detections against user-labeled ground truth — Days 8-9 system testing.

**Ground truth file format** (what you produce when labeling MIMIC-III
discharge summaries — see CLAUDE.md section 16 Days 8-9 and the division of
labor in section 15: this labeling is the user's responsibility, not
something this module or a test can generate):

One JSON file per document in `settings.GROUND_TRUTH_PATH`, named to match
its source document's stem exactly — `data/test_documents/mimic_001.txt`
pairs with `data/ground_truth/mimic_001.json`. Each file:

```json
{
  "source_document": "mimic_001.txt",
  "phi_entities": [
    {"entity_type": "PERSON", "text": "John Smith"},
    {"entity_type": "DATE_TIME", "text": "03/15/1965"},
    {"entity_type": "MRN", "text": "MRN: 00512334"}
  ]
}
```

`entity_type` should match one of PHIAgent's target types (PERSON,
PHONE_NUMBER, EMAIL_ADDRESS, US_SSN, DATE_TIME, LOCATION, MRN — see
phi_agent.py's `_TARGET_ENTITIES`), but this module doesn't enforce that;
an unrecognized type just can't ever match a detection and shows up as a
missed entity, which is still an honest (if unhelpful) result.

**Matching is by (entity_type, normalized text), not text position.** A
human labeling ground truth by reading a document naturally works this way —
asking a labeler to also record exact character offsets would be real
friction for no accuracy benefit at this project's scale. Known limitation
(documented, not hidden, per CLAUDE.md section 17): if the same text appears
twice in one document with different PHI status (rare in practice for names/
SSNs/MRNs), this matching can't distinguish the two occurrences.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src.models import PHIEntity
from src.utils.metrics import f1_score, precision, recall


@dataclass
class GroundTruthEntity:
    """One human-labeled PHI entity, as loaded from a ground truth JSON file."""

    entity_type: str
    text: str


@dataclass
class PHIValidationResult:
    """Precision/recall/F1 for one document's PHI detection against its ground truth."""

    document_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    missed_entities: list[str]  # ground truth entities the agent never found (false negatives)
    extra_entities: list[str]  # agent detections with no matching ground truth entry (false positives)


def _normalize(entity_type: str, text: str) -> tuple[str, str]:
    """Case/whitespace-insensitive match key — a labeler typing "mrn" vs PHIAgent's "MRN" shouldn't count as a miss."""
    return entity_type.strip().upper(), " ".join(text.split()).lower()


def load_ground_truth(path: Path) -> list[GroundTruthEntity]:
    """Load one document's hand-labeled ground truth entities from its JSON file.

    Args:
        path: Path to a ground truth JSON file (see module docstring for format).

    Returns:
        List of GroundTruthEntity, in the order they appear in the file.

    Raises:
        FileNotFoundError: `path` doesn't exist — an edge condition (rule 8):
            a system test asking to validate a specific document must know
            immediately if its ground truth is missing, not silently score
            against zero entities.

    Use when: called once per document by tests/test_system.py, paired with
    that document's text from settings.TEST_DATA_PATH.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return [GroundTruthEntity(entity_type=e["entity_type"], text=e["text"]) for e in data.get("phi_entities", [])]


def validate_phi_detection(document_id: str, detected: list[PHIEntity], ground_truth: list[GroundTruthEntity]) -> PHIValidationResult:
    """Score PHIAgent's detected entities against one document's ground truth.

    Args:
        document_id: Identifies which document this result is for (surfaced
            in the returned result, not used for matching logic).
        detected: PHIAgent.process()'s output entities for this document.
        ground_truth: This document's hand-labeled entities (load_ground_truth()).

    Returns:
        PHIValidationResult with TP/FP/FN counts, precision/recall/F1, and
        the actual missed/extra entity text (not just counts) — an aggregate
        number alone doesn't tell a reviewer *which* entity the agent missed;
        that's the whole point of doing this against real labeled data.

    Use when: called once per document by tests/test_system.py's PHI
    validation test, then aggregated (sum TP/FP/FN across documents, not
    average each document's own precision/recall) into one project-wide
    metric per CLAUDE.md's "PHI precision >= 95%, recall >= 90%" target.
    """
    detected_keys = {_normalize(e.entity_type, e.text) for e in detected}
    truth_keys = {_normalize(e.entity_type, e.text) for e in ground_truth}

    true_positive_keys = detected_keys & truth_keys
    false_positive_keys = detected_keys - truth_keys
    false_negative_keys = truth_keys - detected_keys

    tp, fp, fn = len(true_positive_keys), len(false_positive_keys), len(false_negative_keys)
    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)

    return PHIValidationResult(
        document_id=document_id,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision_value,
        recall=recall_value,
        f1=f1_score(precision_value, recall_value),
        missed_entities=[text for _type, text in false_negative_keys],
        extra_entities=[text for _type, text in false_positive_keys],
    )


def aggregate_validation_results(results: list[PHIValidationResult]) -> dict:
    """Combine multiple documents' PHIValidationResults into one project-wide precision/recall/F1.

    Args:
        results: One PHIValidationResult per document (validate_phi_detection()'s output).

    Returns:
        Dict with aggregate tp/fp/fn/precision/recall/f1 plus a
        `per_document` breakdown — suitable for direct
        src.utils.metrics.export_metrics() output.

    Use when: called once after validating every document in a system-test
    run. Sums raw TP/FP/FN across documents before computing precision/
    recall (micro-averaging) rather than averaging each document's own
    precision/recall (macro-averaging) — a document with zero PHI would
    otherwise contribute a misleading "perfect" 1.0 score with equal weight
    to a document with twenty entities, per precision()/recall()'s own
    "nothing to grade" convention.
    """
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    aggregate_precision = precision(total_tp, total_fp)
    aggregate_recall = recall(total_tp, total_fn)

    return {
        "documents_evaluated": len(results),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": aggregate_precision,
        "recall": aggregate_recall,
        "f1": f1_score(aggregate_precision, aggregate_recall),
        "per_document": [
            {
                "document_id": r.document_id,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "missed_entities": r.missed_entities,
                "extra_entities": r.extra_entities,
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    import tempfile

    # Self-test: no external dependency — synthetic detected/ground-truth
    # entities, plus one throwaway ground truth file to exercise
    # load_ground_truth()'s real file-reading path.
    ground_truth = [
        GroundTruthEntity(entity_type="PERSON", text="John Smith"),
        GroundTruthEntity(entity_type="US_SSN", text="456-78-1234"),
        GroundTruthEntity(entity_type="MRN", text="MRN: 00512334"),
    ]
    detected = [
        PHIEntity(entity_type="PERSON", text="John Smith", confidence=0.95, start_pos=0, end_pos=10, reasoning="test"),
        PHIEntity(entity_type="US_SSN", text="456-78-1234", confidence=0.9, start_pos=20, end_pos=31, reasoning="test"),
        # False positive: detected but not in ground truth.
        PHIEntity(entity_type="LOCATION", text="Chicago", confidence=0.8, start_pos=40, end_pos=47, reasoning="test"),
        # MRN entirely missed -> false negative.
    ]

    result = validate_phi_detection("doc-1", detected, ground_truth)
    assert result.true_positives == 2
    assert result.false_positives == 1 and "chicago" in result.extra_entities
    assert result.false_negatives == 1 and "mrn: 00512334" in result.missed_entities
    assert abs(result.precision - (2 / 3)) < 1e-9
    assert abs(result.recall - (2 / 3)) < 1e-9

    perfect_result = validate_phi_detection("doc-2", detected[:2], ground_truth[:2])
    assert perfect_result.precision == 1.0 and perfect_result.recall == 1.0

    aggregate = aggregate_validation_results([result, perfect_result])
    assert aggregate["documents_evaluated"] == 2
    assert aggregate["true_positives"] == 4  # 2 + 2
    assert aggregate["false_positives"] == 1  # 1 + 0
    assert aggregate["false_negatives"] == 1  # 1 + 0
    assert len(aggregate["per_document"]) == 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_path = Path(tmp_dir) / "sample_doc.json"
        gt_path.write_text(
            json.dumps({"source_document": "sample_doc.txt", "phi_entities": [{"entity_type": "PERSON", "text": "Jane Doe"}]}),
            encoding="utf-8",
        )
        loaded = load_ground_truth(gt_path)
        assert len(loaded) == 1 and loaded[0].entity_type == "PERSON" and loaded[0].text == "Jane Doe"

        missing_path = Path(tmp_dir) / "does_not_exist.json"
        try:
            load_ground_truth(missing_path)
            raise AssertionError("expected FileNotFoundError for a missing ground truth file")
        except FileNotFoundError:
            pass

    print(
        "phi_validator.py self-test passed: TP/FP/FN matching, precision/recall/F1 scoring, "
        "aggregation, and ground truth file loading verified without any network dependency."
    )
