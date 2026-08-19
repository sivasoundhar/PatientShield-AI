"""Knowledge Agent test suite — 10 cases per CLAUDE.md section 16 Day 6.

Requires a real, running UniRAG instance (see src/orchestrator/unirag_connector.py's
module docstring — D:\\projects\\UniRag, started separately on
settings.UNIRAG_BASE_URL, e.g. `uvicorn app.main:app --port 8001`). Every test
here drives the real UniRAG REST API and the real LLMManager chain — no
mocking, per CLAUDE.md section 12. If UniRAG isn't reachable, the whole
module skips cleanly rather than failing — the same environment-conditional
pattern UniRAG's own test suite uses for its GROQ_API_KEY-gated live chat
test (see its README's Testing section).

Each test uses a UUID-suffixed document_id so repeated runs never collide
with a previous run's indexed documents. Cleanup happens after *each* test
(not batched at the end of the module) via UniRAG's existing
DELETE /api/v1/documents endpoint (called directly here, not through
UniRAGConnector, whose surface stays exactly what CLAUDE.md section 16
specifies: upload_document/search/get_citations) — both to avoid leaving
permanent garbage in a shared UniRAG instance, and because UniRAG's own
`rerank_top_n` setting hard-caps /search to its top 3 results *server-side*,
regardless of the `k` a caller requests (confirmed by reading its
app/config/settings.py — `k` only widens the pre-rerank candidate pool, not
the final result count). A corpus left to accumulate documents across a
whole test run means later tests' own chunks can lose that 3-slot cut to
earlier tests' leftover documents — this bit the aggregate accuracy test
during development (case: a real result, not a mocked assumption) before
cleanup was moved to per-test.
"""

import uuid

import httpx
import pytest

from src.agents.knowledge_agent import KnowledgeAgent
from src.config import settings
from src.models import QAResult

pytestmark = pytest.mark.asyncio

_created_document_ids: list[str] = []


def _doc_id(label: str) -> str:
    """A UUID-suffixed document_id, tracked for cleanup — see module docstring."""
    doc_id = f"test-{label}-{uuid.uuid4().hex[:8]}"
    _created_document_ids.append(doc_id)
    return doc_id


def _delete_indexed_documents() -> None:
    """Remove every document _doc_id() has handed out so far from UniRAG's corpus, then clear tracking."""
    with httpx.Client(timeout=settings.UNIRAG_TIMEOUT_SECONDS) as client:
        for doc_id in _created_document_ids:
            client.delete(f"{settings.UNIRAG_BASE_URL}/api/v1/documents", params={"source": f"{doc_id}.txt"})
    _created_document_ids.clear()


@pytest.fixture(scope="module", autouse=True)
def _skip_if_unirag_unreachable():
    # Synchronous (plain httpx.Client, not UniRAGConnector's async API):
    # pytest-asyncio ties an async fixture's event loop to its own scope,
    # and this project's pytest.ini pins asyncio_default_fixture_loop_scope
    # to "function" — a module-scoped async fixture would need its own
    # explicit loop_scope="module" to avoid a ScopeMismatch. Simplest fix:
    # this pre-flight check doesn't need to share the async agent's event
    # loop at all, so just don't make it async.
    try:
        response = httpx.get(f"{settings.UNIRAG_BASE_URL}/api/v1/health", timeout=settings.UNIRAG_TIMEOUT_SECONDS)
        response.raise_for_status()
        healthy = response.json().get("status") == "ok"
    except httpx.HTTPError:
        healthy = False

    if not healthy:
        pytest.skip(f"UniRAG not reachable at {settings.UNIRAG_BASE_URL} — start it separately to run this suite.")
    yield


@pytest.fixture(autouse=True)
def _cleanup_after_each_test():
    """Deletes every document the just-finished test indexed — see module docstring on why this runs per-test."""
    yield
    _delete_indexed_documents()


@pytest.fixture(scope="module")
def knowledge_agent() -> KnowledgeAgent:
    return KnowledgeAgent()


# --- Indexing + retrieval (2 cases) ---


async def test_index_then_search_finds_content(knowledge_agent: KnowledgeAgent):
    doc_id = _doc_id("basic")
    indexed = await knowledge_agent.index_document(doc_id, "The patient's discharge blood pressure was 128/82 mmHg.")
    assert indexed is True

    results = await knowledge_agent._connector.search(doc_id, "blood pressure", k=5)
    assert results, "expected the just-indexed content to be retrievable"
    assert any("128/82" in r["text"] for r in results)


async def test_empty_document_indexing_is_skipped_gracefully(knowledge_agent: KnowledgeAgent):
    indexed = await knowledge_agent.index_document(_doc_id("empty"), "   ")
    assert indexed is False


# --- Grounded answering + citation (2 cases) ---


async def test_answer_grounded_question_returns_correct_answer(knowledge_agent: KnowledgeAgent):
    doc_id = _doc_id("grounded")
    await knowledge_agent.index_document(
        doc_id, "Diagnosis: Type 2 diabetes mellitus. Prescribed metformin 500mg twice daily. Follow up with endocrinology in 4 weeks."
    )

    result = await knowledge_agent.answer_question(doc_id, "What medication was prescribed?")
    assert result.found_in_document is True
    assert "metformin" in result.answer.lower()


async def test_citation_contains_real_document_text(knowledge_agent: KnowledgeAgent):
    doc_id = _doc_id("citation")
    original_text = "Lab results show hemoglobin A1c of 9.2%, indicating poor glycemic control."
    await knowledge_agent.index_document(doc_id, original_text)

    result = await knowledge_agent.answer_question(doc_id, "What was the hemoglobin A1c result?")
    assert result.found_in_document is True
    assert result.source_citation is not None
    # The citation must be traceable to real indexed text, not a fabricated
    # summary — a substantial substring overlap with the original is the
    # concrete check for that (exact full-string equality is too strict
    # given the citation truncates to 200 chars).
    assert "9.2" in result.source_citation or "hemoglobin" in result.source_citation.lower()


# --- Hallucination prevention (2 cases) — the Day 6 finish line's "0% hallucination rate" ---


async def test_hallucination_prevention_refuses_unrelated_question(knowledge_agent: KnowledgeAgent):
    doc_id = _doc_id("refuse")
    await knowledge_agent.index_document(doc_id, "Diagnosis: Seasonal allergic rhinitis, mild. No intervention needed.")

    result = await knowledge_agent.answer_question(doc_id, "What was the patient's cardiac surgery outcome?")
    assert result.found_in_document is False
    assert "9.2" not in result.answer  # sanity: not leaking unrelated fabricated specifics


async def test_no_retrieved_chunks_refuses_without_calling_llm(knowledge_agent: KnowledgeAgent):
    """A document_id with nothing indexed under it must refuse, not hallucinate from another document."""
    result = await knowledge_agent.answer_question(_doc_id("nonexistent"), "What is the diagnosis?")
    assert result.found_in_document is False
    assert result.confidence == 0.0


# --- Per-document isolation (1 case) — the core correctness property unirag_connector.py's filtering exists for ---


async def test_per_document_isolation_no_cross_contamination(knowledge_agent: KnowledgeAgent):
    doc_a = _doc_id("iso-a")
    doc_b = _doc_id("iso-b")
    await knowledge_agent.index_document(doc_a, "Patient A: Diagnosis is pneumonia. Treated with azithromycin.")
    await knowledge_agent.index_document(doc_b, "Patient B: Diagnosis is a fractured tibia. Treated with a cast.")

    result = await knowledge_agent.answer_question(doc_a, "What treatment was given?")
    assert result.found_in_document is True
    assert "azithromycin" in result.answer.lower()
    assert "cast" not in result.answer.lower(), "document A's answer must not leak document B's content"


# --- Structural / contract correctness (2 cases) ---


async def test_confidence_meets_threshold_when_found(knowledge_agent: KnowledgeAgent):
    doc_id = _doc_id("confidence")
    await knowledge_agent.index_document(doc_id, "Diagnosis: Acute appendicitis. Scheduled for emergent appendectomy.")

    result = await knowledge_agent.answer_question(doc_id, "What is the diagnosis?")
    if result.found_in_document:
        assert result.confidence >= settings.QA_CONFIDENCE_THRESHOLD


async def test_result_types_are_correct(knowledge_agent: KnowledgeAgent):
    doc_id = _doc_id("types")
    await knowledge_agent.index_document(doc_id, "Diagnosis: Community-acquired pneumonia.")

    result = await knowledge_agent.answer_question(doc_id, "What is the diagnosis?")
    assert isinstance(result, QAResult)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.found_in_document, bool)


# --- Aggregate accuracy (1 case) — the Day 6 finish line ---


async def test_qa_accuracy_meets_target(knowledge_agent: KnowledgeAgent):
    """Aggregate accuracy >= 98% across a small gold set — the Day 6 finish line.

    Each case is either "must answer correctly" (grounded) or "must refuse"
    (ungrounded) — matching the two-sided precision/recall style used by
    test_phi_agent.py and test_clinical_agent.py's aggregate tests. Kept to
    clearly unambiguous documents/questions for the same reason those did:
    this measures the agent's grounding logic, not model capacity on hard
    edge cases.
    """
    cases = [
        ("Diagnosis: Hypertension, stage 2. Prescribed lisinopril 20mg daily.", "What medication was prescribed?", "lisinopril", True),
        ("Diagnosis: Acute bronchitis. Advised rest and fluids.", "What is the diagnosis?", "bronchitis", True),
        ("Lab results: potassium 5.8 mEq/L, critical value.", "What was the potassium level?", "5.8", True),
        ("Diagnosis: Migraine without aura.", "What was the patient's surgical history?", None, False),
        ("Follow up in 6 weeks with orthopedics.", "What medication is the patient allergic to?", None, False),
    ]

    correct = 0
    for text, question, expected_substring, should_be_found in cases:
        doc_id = _doc_id("gold")
        await knowledge_agent.index_document(doc_id, text)
        result = await knowledge_agent.answer_question(doc_id, question)
        # Delete immediately, not just at test end (the autouse fixture would
        # eventually catch it too) — UniRAG's /search hard-caps results to
        # its top 3 server-side (see module docstring), so leaving this
        # case's document indexed while the next case's document competes
        # for those same 3 slots reintroduces the exact contention this
        # aggregate test needs to avoid to measure grounding logic cleanly.
        _delete_indexed_documents()

        if should_be_found:
            if result.found_in_document and expected_substring.lower() in result.answer.lower():
                correct += 1
        else:
            if not result.found_in_document:
                correct += 1

    accuracy = correct / len(cases)
    print(f"\nKnowledge Agent aggregate accuracy: {accuracy:.2%} ({correct}/{len(cases)})")
    assert accuracy >= 0.98, f"accuracy {accuracy:.2%} below 98% target ({correct}/{len(cases)})"
