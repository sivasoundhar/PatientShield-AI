"""System tests — Days 8-9 finish line: real MIMIC-III discharge summaries against user-labeled ground truth.

Per CLAUDE.md section 15's division of labor, preparing 10 MIMIC-III
discharge summaries (`data/test_documents/*.txt`) and hand-labeling their PHI
entities (`data/ground_truth/<stem>.json`, see phi_validator.py's module
docstring for the exact format) is the user's job, not something this file
generates. Until that data exists, this whole module skips cleanly — the
same environment-conditional pattern used for tests/test_knowledge_agent.py's
live-UniRAG guard — rather than failing on missing fixtures it has no way to
create itself.

**Clinical extraction accuracy is deliberately NOT automated here.** CLAUDE.md
section 16 Days 8-9 calls that out explicitly as "manual spot-check by user"
— there's no ground truth format defined for clinical findings (unlike PHI
entities), and inventing one wasn't asked for. What this file does automate:
PHI detection precision/recall/F1 against real ground truth, full-pipeline
completion across every document, and Q&A hallucination rate.

**Each document's pipeline runs exactly once, shared across every test**
(`pipeline_results` fixture below) — found live during Day 8 system testing
(see PROGRESS.md): the first version of this file had each of the 3 tests
independently re-run PHI detection / the full pipeline on the same
documents, meaning PHI detection alone ran 3x per document. A run that
should take a few minutes took 30+ minutes because of it, even after fixing
an unrelated retry-storm issue in llm_manager.py. One shared run per
document is both faster and more honest: it tests the actual deployed
pipeline once, not three independently-reproduced versions of it that
could in principle disagree.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import settings
from src.database import Base, get_db
from src.main import app
from src.models import PHIEntity
from src.utils.metrics import export_metrics, hallucination_rate
from src.validation.phi_validator import aggregate_validation_results, load_ground_truth, validate_phi_detection

pytestmark = pytest.mark.asyncio


def _test_document_ground_truth_pairs() -> list[tuple[Path, Path]]:
    """Pairs every data/test_documents/*.txt with its data/ground_truth/<stem>.json, skipping any without a match."""
    test_data_dir = Path(settings.TEST_DATA_PATH)
    ground_truth_dir = Path(settings.GROUND_TRUTH_PATH)
    if not test_data_dir.exists() or not ground_truth_dir.exists():
        return []

    pairs = []
    for doc_path in sorted(test_data_dir.glob("*.txt")):
        ground_truth_path = ground_truth_dir / f"{doc_path.stem}.json"
        if ground_truth_path.exists():
            pairs.append((doc_path, ground_truth_path))
    return pairs


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_system_test_data():
    if not _test_document_ground_truth_pairs():
        pytest.skip(
            f"No MIMIC test documents + ground truth found under {settings.TEST_DATA_PATH} / "
            f"{settings.GROUND_TRUTH_PATH}. Populate data/test_documents/*.txt and matching "
            "data/ground_truth/<stem>.json (see src/validation/phi_validator.py's module docstring "
            "for the exact format) to run this suite."
        )


@pytest.fixture(scope="module")
def _system_test_client():
    """A dedicated module-scoped TestClient + isolated in-memory DB, deliberately not conftest.py's `client` fixture.

    conftest.py's `client` is function-scoped (a fresh, empty DB per test) —
    correct for every other test file, where each test wants isolation. This
    suite needs the opposite: every test in this module must see the SAME
    uploaded/processed documents, so `pipeline_results` below can run each
    document through /process exactly once and have all 3 tests read from
    that one shared result.
    """
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

    def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    session.close()


@pytest.fixture(scope="module")
def pipeline_results(_system_test_client: TestClient) -> dict[str, dict]:
    """Runs every available document through /upload + /process exactly once — see module docstring.

    A document whose /process call itself fails (a genuine pipeline bug —
    e.g. a malformed LLM response a fallback/local model produced, observed
    live during Day 8 testing) is recorded with `error` set rather than
    raising here. One bad document must not sink evaluation of the other
    nine: the first version of this fixture asserted status_code == 200
    directly, so a single document's genuine 500 killed all 3 tests in this
    module via ERROR instead of reporting real results for the rest.

    Returns:
        Dict keyed by document stem -> {"document_id", "process_body"
        (None if /process failed), "error" (None if it succeeded)}.

    Use when: any test in this module needs a document's real pipeline
    output (PHI entities, clinical findings, completion status).
    """
    results = {}
    for doc_path, _ground_truth_path in _test_document_ground_truth_pairs():
        text = doc_path.read_text(encoding="utf-8")
        upload_response = _system_test_client.post("/upload", files={"file": (doc_path.name, text.encode(), "text/plain")})
        document_id = upload_response.json()["document_id"]

        process_response = _system_test_client.post("/process", json={"document_id": document_id})
        if process_response.status_code == 200:
            results[doc_path.stem] = {"document_id": document_id, "process_body": process_response.json(), "error": None}
        else:
            results[doc_path.stem] = {"document_id": document_id, "process_body": None, "error": process_response.text}
    return results


async def test_full_pipeline_completes_for_every_document(pipeline_results: dict[str, dict]):
    """Every MIMIC document ran through the full 5-agent pipeline without failing.

    A prerequisite for the rest of Days 8-9's manual review (clinical
    accuracy spot-checks, Q&A review) — those only make sense once every
    document actually produces a completed run to review. No new pipeline
    runs here — `pipeline_results` already ran them; this is the one place
    in this module that actually asserts every document succeeded (the
    other two tests deliberately tolerate individual failures — see their
    own docstrings).
    """
    failures = {stem: r["error"] for stem, r in pipeline_results.items() if r["error"] is not None}
    assert not failures, f"{len(failures)}/{len(pipeline_results)} document(s) failed to process: {failures}"


async def test_phi_detection_against_ground_truth(pipeline_results: dict[str, dict]):
    """PHI precision >= 95%, recall >= 90% across every successfully-processed document — the Day 8 finish line.

    Uses `pipeline_results`' already-computed `phi_detected` output (the
    real pipeline's PHI Agent node), not a second, separate PHIAgent call —
    see module docstring on why sharing one run matters here. Documents
    whose /process call itself failed are excluded (nothing to validate),
    not counted as failures of PHI detection specifically —
    `test_full_pipeline_completes_for_every_document` is where a failed
    /process call gets flagged.
    """
    validation_results = []
    skipped_documents = []
    for doc_path, ground_truth_path in _test_document_ground_truth_pairs():
        result = pipeline_results[doc_path.stem]
        if result["process_body"] is None:
            skipped_documents.append(doc_path.stem)
            continue
        ground_truth = load_ground_truth(ground_truth_path)
        entity_dicts = result["process_body"]["phi_detected"]["entities"]
        detected_entities = [PHIEntity.model_validate(e) for e in entity_dicts]
        validation_results.append(validate_phi_detection(doc_path.stem, detected_entities, ground_truth))

    aggregate = aggregate_validation_results(validation_results)
    aggregate["skipped_documents"] = skipped_documents
    export_metrics({"phi_detection": aggregate}, filename="phi_metrics.json")

    print(
        f"\nPHI detection system-test metrics: precision={aggregate['precision']:.2%} "
        f"recall={aggregate['recall']:.2%} f1={aggregate['f1']:.2%} "
        f"({aggregate['documents_evaluated']} documents evaluated, {len(skipped_documents)} skipped: {skipped_documents})"
    )
    assert aggregate["precision"] >= 0.95, f"precision {aggregate['precision']:.2%} below 95% target"
    assert aggregate["recall"] >= 0.90, f"recall {aggregate['recall']:.2%} below 90% target"


async def test_qa_hallucination_rate(_system_test_client: TestClient, pipeline_results: dict[str, dict]):
    """Hallucination rate = 0% — every successfully-processed document must refuse a deliberately unanswerable question.

    Each document gets exactly one question with no possible grounded
    answer ("What was the patient's favorite childhood pet?") — a real
    discharge summary will never contain this, so any non-refusal is by
    definition a hallucination, not a borderline judgment call. This is a
    narrower, more mechanical check than test_knowledge_agent.py's own
    aggregate accuracy test (which also verifies *correct* grounded answers);
    this one exists specifically for CLAUDE.md's "hallucination rate = 0%"
    target across the real MIMIC corpus. Reuses each document's already-
    indexed document_id from `pipeline_results` — only the /chat call itself
    is new work here, not a third upload+process pass. Documents whose
    /process call failed are skipped (nothing was indexed to ask about).
    """
    hallucinated = 0
    answered = 0

    for doc_stem, result in pipeline_results.items():
        if result["process_body"] is None:
            continue
        document_id = result["document_id"]
        chat_response = _system_test_client.post(
            "/chat", json={"document_id": document_id, "question": "What was the patient's favorite childhood pet?"}
        )
        body = chat_response.json()
        if body["found_in_document"]:
            hallucinated += 1
            answered += 1

    rate = hallucination_rate(hallucinated, answered if answered > 0 else 1)
    export_metrics({"qa_hallucination_rate": rate, "hallucinated_count": hallucinated}, filename="qa_metrics.json")

    print(f"\nQ&A hallucination rate on unanswerable questions: {rate:.2%} ({hallucinated} hallucinated)")
    assert hallucinated == 0, f"{hallucinated} document(s) answered an unanswerable question instead of refusing"
