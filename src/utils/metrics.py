"""Precision/recall/F1/accuracy calculators + results export — Days 8-9 system testing infrastructure.

Pure functions only (no I/O beyond `export_metrics`'s single JSON write) —
every agent's own aggregate test (test_phi_agent.py's precision/recall,
test_clinical_agent.py's accuracy, test_knowledge_agent.py's accuracy)
already computed these inline with the same formulas; this module exists so
Days 8-9's system tests (tests/test_system.py, src/validation/phi_validator.py)
share one implementation instead of a fourth copy-pasted version, and so
results can be written to disk for docs/TEST_RESULTS.md (user-authored) to
reference.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings


def precision(true_positives: int, false_positives: int) -> float:
    """TP / (TP + FP) — of everything flagged, what fraction was actually correct.

    Returns:
        1.0 when nothing was flagged at all (TP=FP=0) — an empty prediction
        set has no false positives to be wrong about, so treating it as
        "perfectly precise" is the standard convention (matches the identical
        choice already made independently in test_phi_agent.py).

    Use when: computing PHI detection precision against ground truth, or any
    other TP/FP-shaped classification result.
    """
    if true_positives + false_positives == 0:
        return 1.0
    return true_positives / (true_positives + false_positives)


def recall(true_positives: int, false_negatives: int) -> float:
    """TP / (TP + FN) — of everything that should have been flagged, what fraction was.

    Returns:
        1.0 when there was nothing to find at all (TP=FN=0) — same
        empty-set convention as precision() above.

    Use when: computing PHI detection recall against ground truth.
    """
    if true_positives + false_negatives == 0:
        return 1.0
    return true_positives / (true_positives + false_negatives)


def f1_score(precision_value: float, recall_value: float) -> float:
    """Harmonic mean of precision and recall — penalizes a lopsided precision/recall split more than a plain average would.

    Returns:
        0.0 when both inputs are 0 (avoids a ZeroDivisionError on the
        harmonic mean's denominator).

    Use when: a single number is needed to compare two detection runs that
    might trade precision for recall differently.
    """
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def accuracy(correct: int, total: int) -> float:
    """correct / total, with the same "nothing to grade => perfect score" convention as precision/recall.

    Use when: any straightforward correct-vs-total ratio — clinical
    extraction accuracy, Q&A accuracy, etc.
    """
    if total == 0:
        return 1.0
    return correct / total


def hallucination_rate(hallucinated: int, total_answered: int) -> float:
    """Fraction of *answered* (not refused) questions that were ungrounded/incorrect.

    Args:
        hallucinated: Count of answers that were wrong or unsupported by the
            source document.
        total_answered: Count of questions the system actually attempted to
            answer (excludes ones it correctly refused — a refusal is the
            opposite of a hallucination, not a zero-length one).

    Returns:
        0.0 when nothing was answered at all — no answers means no
        hallucinations to measure, not an undefined rate.

    Use when: scoring the Knowledge Agent's Q&A against
    CLAUDE.md's "hallucination rate = 0%" target (section 12).
    """
    if total_answered == 0:
        return 0.0
    return hallucinated / total_answered


def export_metrics(results: dict, filename: str = "metrics.json") -> Path:
    """Write `results` to settings.RESULTS_PATH/filename as timestamped JSON.

    Args:
        results: Any JSON-serializable dict — callers (tests/test_system.py)
            decide the shape (per-agent metrics, per-document breakdowns, etc).
        filename: Output filename within RESULTS_PATH.

    Returns:
        The path written to, so callers/tests can assert on it directly.

    Use when: called once at the end of a system-test run to persist metrics
    for docs/TEST_RESULTS.md (user-authored) to reference — not called
    per-document, only for the final aggregate.
    """
    output_dir = Path(settings.RESULTS_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **results}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    import tempfile

    # Self-test: no external dependency — pure arithmetic plus one file
    # write to a throwaway temp directory (never touches the real
    # settings.RESULTS_PATH, so this can't pollute Day 8-9's actual output).
    assert precision(8, 2) == 0.8
    assert precision(0, 0) == 1.0
    assert recall(8, 2) == 0.8
    assert recall(0, 0) == 1.0
    assert abs(f1_score(0.8, 0.8) - 0.8) < 1e-9
    assert f1_score(0.0, 0.0) == 0.0
    assert accuracy(9, 10) == 0.9
    assert accuracy(0, 0) == 1.0
    assert hallucination_rate(1, 10) == 0.1
    assert hallucination_rate(0, 0) == 0.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        original_results_path = settings.RESULTS_PATH
        settings.RESULTS_PATH = tmp_dir
        try:
            written_path = export_metrics({"phi_precision": 0.95, "phi_recall": 0.9})
            assert written_path.exists()
            loaded = json.loads(written_path.read_text(encoding="utf-8"))
            assert loaded["phi_precision"] == 0.95
            assert "generated_at" in loaded
        finally:
            settings.RESULTS_PATH = original_results_path

    print("metrics.py self-test passed: precision/recall/F1/accuracy/hallucination_rate formulas and JSON export verified.")
