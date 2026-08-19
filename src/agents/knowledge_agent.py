"""Knowledge Agent — indexes de-identified text into UniRAG, then answers questions grounded in it.

Two distinct operations, not one, unlike PHI/Clinical's single `process()`
call: **indexing** happens once per document during the pipeline (Planner ->
PHI -> Clinical -> Knowledge -> Audit), while **answering** happens on demand,
once per `/chat` request, potentially long after indexing finished and with
no relationship to the pipeline run at all. `process()` still exists to
satisfy BaseAgent's contract (it's what the pipeline's knowledge_agent node
calls) and simply delegates to `index_document` — but `index_document` and
`answer_question` are this class's real public API.

Strict grounding, not implicit: the answer-generation prompt instructs the
LLM to refuse (`NOT_FOUND`) rather than answer outside the retrieved context,
and this agent independently enforces `QA_CONFIDENCE_THRESHOLD` on top of
that self-report — a low-confidence "answer" is treated the same as an
explicit refusal (`found_in_document=False`, with the returned `answer` text
overridden to a fixed, honest refusal rather than whatever text the LLM
wrote). This mirrors the real UniRAG project's own README note that a soft
"answer using only the context" instruction let its model hedge and then
answer from training data anyway — the same failure mode this project's
hallucination-prevention target (section 12: "Q&A accuracy >= 98%,
hallucination rate = 0%") exists to catch.
"""

import logging
import re

from src.agents.base_agent import BaseAgent
from src.config import settings
from src.models import QAResult
from src.orchestrator.llm_manager import LLMManager
from src.orchestrator.unirag_connector import UniRAGConnector, UniRAGUnavailableError

logger = logging.getLogger(__name__)

_ANSWER_PROMPT = """You are answering a question about a clinical document using ONLY the excerpts below. \
Do not use any outside knowledge, and do not guess.

Excerpts from the document:
{context}

Question: {question}

Rule 1 — if the excerpts do not contain enough information to answer the question, respond with exactly \
ANSWER: NOT_FOUND — do not attempt a partial or inferred answer.
Rule 2 — if the excerpts do contain the answer, state it directly and concisely, citing only what the \
excerpts actually say.
Rule 3 — confidence reflects how directly and completely the excerpts answer the question, not how \
plausible the answer sounds.

Respond in exactly this format, two lines, nothing else:
ANSWER: your answer, or NOT_FOUND
CONFIDENCE: a number from 0.00 to 1.00"""

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)(?:\nCONFIDENCE:|$)", re.IGNORECASE | re.DOTALL)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([\d.]+)", re.IGNORECASE)

_NO_EVIDENCE_ANSWER = "I don't have information about that in this document."
_LOW_CONFIDENCE_ANSWER = "I don't have enough confidence in this document's content to answer that question."
_UNAVAILABLE_ANSWER = "Unable to answer — the knowledge index is currently unavailable."


class KnowledgeAgent(BaseAgent):
    """Indexes de-identified text into UniRAG and answers questions grounded only in that document.

    Use when: `index_document()` is called once per document by the
    LangGraph pipeline's knowledge_agent node; `answer_question()` is called
    once per `/chat` request, independently of any pipeline run.
    """

    def __init__(self, connector: UniRAGConnector | None = None, llm_manager: LLMManager | None = None) -> None:
        super().__init__(agent_name="knowledge_agent")
        self._connector = connector or UniRAGConnector()
        self._llm = llm_manager or LLMManager()

    async def process(self, input_data: tuple[str, str]) -> bool:
        """BaseAgent contract — delegates to index_document, since that's the pipeline node's job.

        Args:
            input_data: (document_id, de_identified_text) tuple. Unlike
                PHI/Clinical's process(text), this agent genuinely needs
                document_id too — it's the handle UniRAG search later uses
                to isolate this document's chunks (see unirag_connector.py).

        Returns:
            Whether indexing succeeded.

        Use when: called once per document by pipeline.py's
        `_node_knowledge_agent`. Q&A goes through `answer_question()`
        directly, not through this method — see class docstring.
        """
        document_id, text = input_data
        return await self.index_document(document_id, text)

    async def index_document(self, document_id: str, de_identified_text: str) -> bool:
        """Index de-identified text into UniRAG so it becomes queryable via answer_question().

        Args:
            document_id: This document's id.
            de_identified_text: MUST already be de-identified — see
                unirag_connector.py's upload_document() docstring for where
                that guarantee actually lives (the pipeline's node order,
                not a check in this method).

        Returns:
            True if indexing succeeded, False if skipped (empty text) or
            UniRAG was unavailable — a middle-of-pipeline degradation per
            rule 8, not a reason to fail the whole document's processing
            (PHI and Clinical results are still useful without Q&A).

        Use when: called once per document, after PHI de-identification.
        """
        if not de_identified_text.strip():
            self.log_decision("indexing_skipped", status="skipped", reasoning="empty de-identified text")
            return False

        try:
            chunks_indexed = await self._connector.upload_document(document_id, de_identified_text)
        except UniRAGUnavailableError as exc:
            logger.info("Knowledge indexing failed for document %s: %s", document_id, exc)
            self.log_decision("indexing_failed", status="error", reasoning=str(exc))
            return False

        self.log_decision("document_indexed", details={"chunks_indexed": chunks_indexed})
        return True

    async def answer_question(self, document_id: str, question: str) -> QAResult:
        """Answer `question` grounded only in `document_id`'s indexed chunks, refusing if unsure.

        Args:
            document_id: Which document to search and answer from.
            question: The user's question.

        Returns:
            QAResult. `found_in_document=False` covers three distinct cases,
            all of which return the same honest-refusal shape rather than a
            partial/hallucinated answer: no matching chunks retrieved, the
            LLM itself reported NOT_FOUND, or the LLM answered but with
            confidence below settings.QA_CONFIDENCE_THRESHOLD.

        Use when: called once per POST /chat request.
        """
        try:
            chunks = await self._connector.search(document_id, question, k=settings.QA_TOP_K)
        except UniRAGUnavailableError as exc:
            logger.info("Knowledge search failed for document %s: %s", document_id, exc)
            self.log_decision("qa_failed", status="error", reasoning=str(exc))
            return QAResult(answer=_UNAVAILABLE_ANSWER, confidence=0.0, found_in_document=False)

        if not chunks:
            self.log_decision("qa_no_evidence", status="skipped", reasoning="no matching chunks retrieved for this document")
            return QAResult(answer=_NO_EVIDENCE_ANSWER, confidence=0.0, found_in_document=False)

        context = "\n\n".join(f"[Excerpt {i + 1}]: {chunk['text']}" for i, chunk in enumerate(chunks))
        prompt = _ANSWER_PROMPT.format(context=context, question=question)

        try:
            result = await self._llm.get_response(prompt, max_tokens=300)
        except Exception as exc:  # noqa: BLE001 - LLM unavailable is an edge condition here, not a middle-of-pipeline one: no answer can be produced at all
            logger.info("Knowledge answer-generation LLM call failed for document %s: %s", document_id, exc)
            self.log_decision("qa_failed", status="error", reasoning=str(exc))
            return QAResult(answer=_UNAVAILABLE_ANSWER, confidence=0.0, found_in_document=False)

        answer_text, confidence = self._parse_response(result.text)
        not_found = answer_text.strip().upper() == "NOT_FOUND"

        if not_found or confidence < settings.QA_CONFIDENCE_THRESHOLD:
            self.log_decision(
                "qa_refused",
                status="skipped",
                confidence=confidence,
                reasoning="LLM reported NOT_FOUND" if not_found else "confidence below QA_CONFIDENCE_THRESHOLD",
            )
            return QAResult(answer=_LOW_CONFIDENCE_ANSWER if not not_found else _NO_EVIDENCE_ANSWER, confidence=confidence, found_in_document=False)

        citations = self._connector.get_citations(chunks[:1])
        self.log_decision("qa_answered", confidence=confidence, reasoning=answer_text, details={"chunks_used": len(chunks)})
        return QAResult(answer=answer_text, confidence=confidence, source_citation=citations[0] if citations else None, found_in_document=True)

    def _parse_response(self, raw_text: str) -> tuple[str, float]:
        """Parse the LLM's ANSWER/CONFIDENCE response, falling back to a safe refusal if unparseable.

        An unparseable response degrades to NOT_FOUND-equivalent behavior
        (via confidence=0.0, which fails the threshold check in
        answer_question) rather than risk treating malformed text as a
        confident answer — hallucination prevention takes priority over
        recovering a possibly-valid answer from a malformed response.
        """
        answer_match = _ANSWER_RE.search(raw_text)
        confidence_match = _CONFIDENCE_RE.search(raw_text)

        if not answer_match:
            logger.info("Could not parse Q&A response: %r", raw_text)
            return "NOT_FOUND", 0.0

        answer = answer_match.group(1).strip()
        try:
            confidence = max(0.0, min(1.0, float(confidence_match.group(1)))) if confidence_match else 0.0
        except ValueError:
            confidence = 0.0

        return answer, confidence


if __name__ == "__main__":
    import asyncio
    from unittest.mock import AsyncMock

    from src.orchestrator.llm_manager import LLMResult

    async def _run_self_test() -> None:
        # Self-test per rule 6: no real network/LLM/UniRAG call. Both the
        # connector and the LLM manager are monkeypatched with canned,
        # deterministic responses.
        connector = UniRAGConnector()
        connector.upload_document = AsyncMock(return_value=4)
        connector.search = AsyncMock(
            return_value=[
                {"source": "doc-1.txt", "text": "Blood pressure was 128/82 at discharge.", "score": 0.9},
            ]
        )
        connector.get_citations = lambda chunks: [f'"{c["text"]}" (from {c["source"]})' for c in chunks]

        llm = LLMManager()

        def _canned(prompt: str, max_tokens: int | None = None) -> LLMResult:
            # Match on the "Question:" line specifically, not the whole
            # prompt — the context excerpt always contains "blood pressure",
            # so substring-matching the full prompt would answer every
            # question the same way regardless of what was actually asked.
            question_match = re.search(r"Question: (.+)", prompt)
            question = question_match.group(1) if question_match else ""
            if "blood pressure" in question.lower():
                return LLMResult(text="ANSWER: 128/82 at discharge\nCONFIDENCE: 0.92", provider="fake")
            return LLMResult(text="ANSWER: NOT_FOUND\nCONFIDENCE: 0.10", provider="fake")

        llm.get_response = AsyncMock(side_effect=lambda prompt, max_tokens=None: _canned(prompt, max_tokens))

        agent = KnowledgeAgent(connector=connector, llm_manager=llm)

        # index_document via the BaseAgent process() contract.
        indexed = await agent.process(("doc-1", "Blood pressure was 128/82 at discharge."))
        assert indexed is True

        # Grounded question -> confident, cited answer.
        result = await agent.answer_question("doc-1", "What was the patient's blood pressure?")
        assert result.found_in_document is True
        assert "128/82" in result.answer
        assert result.confidence >= settings.QA_CONFIDENCE_THRESHOLD
        assert result.source_citation is not None and "doc-1.txt" in result.source_citation

        # Ungrounded question -> refusal, not a guess.
        result2 = await agent.answer_question("doc-1", "What medication was prescribed?")
        assert result2.found_in_document is False
        assert result2.confidence < settings.QA_CONFIDENCE_THRESHOLD

        # No retrieved chunks at all -> refusal without ever calling the LLM.
        connector.search = AsyncMock(return_value=[])
        llm.get_response = AsyncMock(side_effect=AssertionError("should not call LLM with zero retrieved chunks"))
        result3 = await agent.answer_question("doc-1", "Anything at all?")
        assert result3.found_in_document is False

        trail = agent.get_audit_trail()
        assert any(e.action == "document_indexed" for e in trail)
        assert any(e.action == "qa_answered" for e in trail)
        assert any(e.action == "qa_refused" for e in trail)
        assert any(e.action == "qa_no_evidence" for e in trail)

        print(
            "knowledge_agent.py self-test passed: indexing, grounded answering, and "
            "hallucination-prevention refusal paths verified without any network dependency."
        )

    asyncio.run(_run_self_test())
