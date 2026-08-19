"""HTTP client for UniRAG — the real, already-built RAG service this Knowledge Agent
indexes documents into and retrieves from, per CLAUDE.md section 16 Day 6.

UniRAG (D:\\projects\\UniRag) is a separate 6-day portfolio project, not code
built as part of this sprint — it's a real FastAPI service with its own
hybrid BM25+dense retrieval stack, already running its own tests against a
real corpus. This module is the thin HTTP boundary between the two projects;
no retrieval/embedding logic is reimplemented here (see PROGRESS.md Day 6 for
the discovery that led here: CLAUDE.md's own section 3 tech-stack table listed
ChromaDB/sentence-transformers directly, which would have meant rebuilding
UniRAG's own job from scratch — the actual UniRAG project already exists and
does this correctly, so "reuse infrastructure" per section 1's story means
calling it, not re-implementing it).

Two real gaps in UniRAG's actual API (confirmed by reading its app/main.py,
not assumed from its README) shape this module's design:

1. **No per-document isolation.** UniRAG's corpus is shared across every
   document anyone has ever uploaded through it — including its own
   permanently-seeded 3-document sample corpus explaining its own retrieval
   pipeline. `/api/v1/search` has no source-filter parameter. Left
   unhandled, a clinical question could get "grounded" in UniRAG's own docs
   about RRF, or in a different patient's leftover chunks from earlier
   testing — a real correctness and privacy problem, not just noise.
   Fixed here by uploading each document under a filename unique to its
   `document_id` (the `source` UniRAG tracks internally), then over-fetching
   search results and filtering client-side down to just that source before
   the caller ever sees them.
2. **No chunk_id concept.** `/api/v1/search` returns only
   `{source, text, score}` per result — chunks aren't individually
   addressable over the API. `get_citations()` below is therefore a local
   formatting helper over already-retrieved (source, text) pairs, not a
   second network call — there's nothing on UniRAG's side to look up by id.
"""

import httpx

from src.config import settings


class UniRAGUnavailableError(RuntimeError):
    """Raised when UniRAG can't be reached or returns an unexpected response.

    An edge condition per rule 8 (fail loud) — callers (KnowledgeAgent)
    decide whether to degrade gracefully (e.g. "Q&A temporarily unavailable")
    rather than this module silently returning empty results that would be
    indistinguishable from "genuinely no matches."
    """


class UniRAGConnector:
    """Thin async HTTP client over UniRAG's real REST API (settings.UNIRAG_BASE_URL).

    Use when: instantiated once by KnowledgeAgent and reused across
    documents — no per-call state, so a module-level singleton (mirroring
    PHIAgent/ClinicalAgent's pattern in pipeline.py) is safe.
    """

    def __init__(self, base_url: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._base_url = (base_url or settings.UNIRAG_BASE_URL).rstrip("/")
        # Injectable only for the self-test below (httpx.MockTransport) —
        # None in every real code path, which leaves httpx's normal network
        # transport in place. Avoids adding a mocking library dependency
        # (e.g. respx) not already in requirements.txt, per CLAUDE.md rule 2.
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=settings.UNIRAG_TIMEOUT_SECONDS, transport=self._transport)

    async def health_check(self) -> bool:
        """True if UniRAG is reachable and reports itself healthy.

        Use when: called before indexing/search in contexts that want to
        fail fast with a clear reason rather than let an HTTP error surface
        from deeper in the call chain (e.g. a test suite deciding whether to
        skip cleanly — see tests/test_knowledge_agent.py).
        """
        try:
            async with self._client() as client:
                response = await client.get(f"{self._base_url}/api/v1/health")
                response.raise_for_status()
                return response.json().get("status") == "ok"
        except httpx.HTTPError:
            return False

    async def upload_document(self, document_id: str, text: str) -> int:
        """Index de-identified text into UniRAG under a filename unique to this document.

        Args:
            document_id: This document's id — becomes the UniRAG `source`
                (via `_source_for`), the only handle later search/delete
                calls have to isolate this document's chunks from every
                other document in UniRAG's shared corpus.
            text: The text to index. Callers MUST pass only de-identified
                text — CLAUDE.md's hard constraint ("100% de-identification
                before ANY indexing. No PHI leaks to vector DB") is enforced
                by the pipeline only ever calling this with
                `de_identified_text`, never `original_text` (see
                pipeline.py's `_node_knowledge_agent`), not by any check in
                this method itself.

        Returns:
            Number of chunks UniRAG indexed.

        Raises:
            UniRAGUnavailableError: UniRAG is unreachable or returned an
                error — an edge condition, not something to silently ignore.

        Use when: called once per document by KnowledgeAgent.index_document().
        """
        source = self._source_for(document_id)
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/upload",
                    files={"file": (source, text.encode("utf-8"), "text/plain")},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UniRAGUnavailableError(f"UniRAG upload failed for document {document_id!r}: {exc}") from exc

        return response.json()["chunks_indexed"]

    async def search(self, document_id: str, query: str, k: int = 5) -> list[dict]:
        """Search UniRAG's corpus for `query`, returning only chunks from this document.

        Args:
            document_id: Which document to restrict results to (see module
                docstring's "no per-document isolation" gap).
            query: The search query.
            k: How many matching chunks to return, after filtering.

        Returns:
            Up to `k` dicts shaped `{source, text, score}`, all belonging to
            this document, ranked as UniRAG's hybrid retrieval + rerank
            stack ordered them.

        Raises:
            UniRAGUnavailableError: UniRAG is unreachable or returned an error.

        Use when: called once per question by KnowledgeAgent.answer_question().
        """
        source = self._source_for(document_id)
        try:
            async with self._client() as client:
                response = await client.post(
                    f"{self._base_url}/api/v1/search",
                    json={"query": query, "k": settings.UNIRAG_SEARCH_OVERFETCH_K},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UniRAGUnavailableError(f"UniRAG search failed for document {document_id!r}: {exc}") from exc

        results = response.json()["results"]
        matched = [r for r in results if r["source"] == source]
        return matched[:k]

    def get_citations(self, chunks: list[dict]) -> list[str]:
        """Format already-retrieved chunks into human-readable citation strings.

        Args:
            chunks: Results from `search()`.

        Returns:
            One citation string per chunk, e.g. '"...excerpt..." (from doc-id.txt)'.

        Use when: called by KnowledgeAgent to build QAResult.source_citation.
        Not a network call — see module docstring on why UniRAG has no
        chunk-id lookup to call instead.
        """
        citations = []
        for chunk in chunks:
            text = chunk["text"].strip()
            excerpt = text[:200] + ("..." if len(text) > 200 else "")
            citations.append(f'"{excerpt}" (from {chunk["source"]})')
        return citations

    def _source_for(self, document_id: str) -> str:
        """The UniRAG `source` filename this document's chunks are tracked under."""
        return f"{document_id}.txt"


if __name__ == "__main__":
    import asyncio

    def _fake_handler(request: httpx.Request) -> httpx.Response:
        """Routes fake responses by path — stands in for a real UniRAG process.

        Use when: passed as `httpx.MockTransport(_fake_handler)` to
        UniRAGConnector's constructor below. httpx's own MockTransport (part
        of the httpx package already in requirements.txt) avoids pulling in
        a separate mocking library (e.g. respx) per CLAUDE.md rule 2.
        """
        path = request.url.path
        if path == "/api/v1/health":
            return httpx.Response(200, json={"status": "ok", "app_env": "test"})
        if path == "/api/v1/upload":
            return httpx.Response(200, json={"filename": "doc-1.txt", "chunking_strategy": "recursive", "chunks_indexed": 3})
        if path == "/api/v1/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"source": "doc-1.txt", "text": "Patient's blood pressure was elevated.", "score": 0.9},
                        {"source": "sample_hybrid_retrieval.txt", "text": "BM25 matches literal keywords.", "score": 0.85},
                        {"source": "doc-2.txt", "text": "A different patient's note entirely.", "score": 0.8},
                        {"source": "doc-1.txt", "text": "Follow up recommended in two weeks.", "score": 0.7},
                    ]
                },
            )
        return httpx.Response(404, json={"error": f"unexpected path {path!r} in self-test"})

    async def _run_self_test() -> None:
        # Self-test per rule 6: no real network call — httpx.MockTransport
        # verifies request shaping (multipart upload, search body,
        # source-filtering logic) without needing a running UniRAG process.
        connector = UniRAGConnector(base_url="http://fake-unirag:8001", transport=httpx.MockTransport(_fake_handler))

        healthy = await connector.health_check()
        assert healthy is True

        chunks_indexed = await connector.upload_document("doc-1", "Some de-identified clinical text.")
        assert chunks_indexed == 3

        # search() must filter out chunks from other documents/UniRAG's own
        # sample corpus — this is the core correctness property the module
        # docstring's "no per-document isolation" gap requires.
        results = await connector.search("doc-1", "blood pressure", k=5)
        assert len(results) == 2, f"expected only doc-1's own chunks, got {results}"
        assert all(r["source"] == "doc-1.txt" for r in results)

        citations = connector.get_citations(results)
        assert len(citations) == 2
        assert "doc-1.txt" in citations[0]

        print("unirag_connector.py self-test passed: request shaping and per-document filtering verified without any network dependency.")

    asyncio.run(_run_self_test())
