"""LangGraph 5-node orchestrator: planner -> phi -> clinical -> knowledge -> audit.

CLAUDE.md section 10 specifies this exact node order and says each node is a
placeholder until its agent lands on its own day (planner + PHI: Day 4,
clinical: Day 5, knowledge: Day 6, audit: Day 7). phi_agent is real as of
Day 4 (src/agents/phi_agent.py); the other three remain placeholders until
their own days — the graph topology itself does not change.

State management: PipelineState is a TypedDict per section 10. `audit_events`
uses LangGraph's reducer pattern (Annotated[..., operator.add]) so each node
returns only *its own* new events rather than the full accumulated list —
LangGraph concatenates them automatically. No side effects happen in nodes
(no DB writes): per section 10, "database writes happen in main.py after
pipeline completes".
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.audit_agent import AuditAgent
from src.agents.clinical_agent import ClinicalAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.phi_agent import PHIAgent, mask_text
from src.models import AuditEvent


class PipelineState(TypedDict):
    """Shared state threaded through all 5 pipeline nodes.

    Every field here mirrors CLAUDE.md section 10's field list exactly, so
    main.py can map this straight onto ProcessResponse without translation.
    """

    document_id: str
    original_text: str
    execution_plan: list[str] | None
    phi_results: dict | None
    de_identified_text: str | None
    clinical_results: dict | None
    knowledge_indexed: bool
    qa_results: dict | None
    audit_events: Annotated[list[AuditEvent], operator.add]
    error: str | None
    processing_time: float | None


def _placeholder_event(agent_name: str, note: str) -> AuditEvent:
    """Build the one audit event a placeholder node emits.

    Use when: called by each `_node_*` function below until that agent's
    real implementation lands (see the day noted in `note`). Matches the
    same "skipped" placeholder pattern main.py's /process endpoint already
    uses (rule: honest placeholders, never a fabricated success).
    """
    return AuditEvent(agent_name=agent_name, action="placeholder_response", status="skipped", details={"note": note})


def _with_timing(agent_name: str, elapsed_seconds: float, node_result: dict) -> dict:
    """Appends a `node_timing` AuditEvent recording how long this node's real work took (Day 9 perf instrumentation).

    Args:
        agent_name: Which node this timing belongs to.
        elapsed_seconds: Wall-clock time the node's real work took, measured
            by the caller around everything except this function itself.
        node_result: The dict the node is about to return — mutated in
            place (its `audit_events` list gets one more entry) and returned
            for a convenient `return _with_timing(...)` at each call site.

    Returns:
        `node_result`, with the timing event appended to `audit_events`.

    Use when: called once by every `_node_*` function, right before
    returning. Lives on the audit trail (AuditEvent.details is free-form
    JSON) rather than as a new PipelineState field — section 10's field list
    is fixed, and every existing /audit/{id} and /pipeline-status/{id}
    response already surfaces AuditEvent.details with no schema change
    needed. This is what Day 9's "which agent is slow?" analysis is
    measured from — see PROGRESS.md.
    """
    timing_event = AuditEvent(agent_name=agent_name, action="node_timing", details={"duration_seconds": round(elapsed_seconds, 3)})
    node_result["audit_events"] = list(node_result.get("audit_events", [])) + [timing_event]
    return node_result


async def _node_planner(state: PipelineState) -> dict:
    """Placeholder for the Planner Agent (Day 4). Emits a fixed, non-LLM execution plan.

    Use when: invoked by LangGraph as the graph's entry node. The plan here
    is the pipeline's own fixed node order, not an LLM-generated plan — real
    LLM-based planning is Day 4's job (src/agents/planner_agent.py).
    """
    start = time.perf_counter()
    event = _placeholder_event("planner", "Planner Agent lands Day 4 — see CLAUDE.md section 16")
    return _with_timing(
        "planner",
        time.perf_counter() - start,
        {
            "audit_events": [event],
            "execution_plan": ["phi_agent", "clinical_agent", "knowledge_agent", "audit_agent"],
        },
    )


# Module-level singleton: PHIAgent's __init__ loads a spaCy model, expensive
# enough to build once rather than per document (mirrors PatientShieldPipeline
# itself being constructed once in main.py). Its BaseAgent audit trail
# accumulates for the agent's entire lifetime though — see _node_phi_agent's
# before/after slicing below, which is what stops document A's PHI decisions
# from leaking into document B's returned audit trail (and, worse, into
# document B's /audit/{id} response) on a long-running server process.
_phi_agent = PHIAgent()


async def _node_phi_agent(state: PipelineState) -> dict:
    """Detect PHI in `original_text` and produce the de-identified text.

    Use when: invoked by LangGraph as the second pipeline node. Real as of
    Day 4 — see src/agents/phi_agent.py for the Presidio + LLM design.
    """
    start = time.perf_counter()
    events_before = len(_phi_agent.get_audit_trail())
    result = await _phi_agent.process(state["original_text"])
    new_events = _phi_agent.get_audit_trail()[events_before:]

    return _with_timing(
        "phi_agent",
        time.perf_counter() - start,
        {
            "audit_events": new_events,
            "phi_results": result.model_dump(mode="json"),
            "de_identified_text": mask_text(state["original_text"], result.entities),
        },
    )


# Module-level singleton, same rationale as _phi_agent above: no expensive
# per-document setup here (unlike PHIAgent's spaCy load), but keeping the
# pattern identical matters more than the marginal cost — see the Day 4
# PROGRESS.md note flagging that every later agent hits the same
# audit-trail-accumulates-across-documents issue if built as a singleton.
_clinical_agent = ClinicalAgent()


async def _node_clinical_agent(state: PipelineState) -> dict:
    """Extract prioritized clinical findings from the de-identified text.

    Use when: invoked by LangGraph as the third pipeline node. Real as of
    Day 5 — see src/agents/clinical_agent.py. Runs on `de_identified_text`,
    never `original_text` — the Clinical Agent must not see raw PHI.
    """
    start = time.perf_counter()
    events_before = len(_clinical_agent.get_audit_trail())
    result = await _clinical_agent.process(state["de_identified_text"] or "")
    new_events = _clinical_agent.get_audit_trail()[events_before:]

    return _with_timing(
        "clinical_agent",
        time.perf_counter() - start,
        {
            "audit_events": new_events,
            "clinical_results": result.model_dump(mode="json"),
        },
    )


# Module-level singleton, same rationale as _phi_agent/_clinical_agent above.
_knowledge_agent = KnowledgeAgent()


async def _node_knowledge_agent(state: PipelineState) -> dict:
    """Index the de-identified text into UniRAG, making it queryable via /chat.

    Use when: invoked by LangGraph as the fourth pipeline node. Real as of
    Day 6 — see src/agents/knowledge_agent.py. Indexing failure (UniRAG
    unreachable) degrades knowledge_indexed to False rather than failing the
    whole pipeline — PHI/Clinical results are still useful without Q&A.
    """
    start = time.perf_counter()
    events_before = len(_knowledge_agent.get_audit_trail())
    indexed = await _knowledge_agent.index_document(state["document_id"], state["de_identified_text"] or "")
    new_events = _knowledge_agent.get_audit_trail()[events_before:]

    return _with_timing(
        "knowledge_agent",
        time.perf_counter() - start,
        {
            "audit_events": new_events,
            "knowledge_indexed": indexed,
        },
    )


# Module-level singleton, same rationale as the other agents above.
_audit_agent = AuditAgent()


async def _node_audit_agent(state: PipelineState) -> dict:
    """Consolidate every prior node's audit events into one summary, flagging any that produced none.

    Use when: invoked by LangGraph as the fifth and final pipeline node. Real
    as of Day 7 — see src/agents/audit_agent.py. Runs on the full
    `audit_events` list accumulated so far in `state`, not a fresh slice —
    this is the one node whose job is literally to look at every prior
    node's output, not just to run in isolation and append to it.
    """
    start = time.perf_counter()
    events_before = len(_audit_agent.get_audit_trail())
    await _audit_agent.process(state["audit_events"])
    new_events = _audit_agent.get_audit_trail()[events_before:]

    return _with_timing("audit_agent", time.perf_counter() - start, {"audit_events": new_events})


def _build_graph():
    """Wire the 5 nodes into the fixed linear pipeline from CLAUDE.md section 10.

    Returns:
        A compiled LangGraph graph, ready for `.ainvoke()`.

    Use when: called once by PatientShieldPipeline.__init__ — compilation is
    the relatively expensive step, so it happens once, not per document.
    """
    graph = StateGraph(PipelineState)
    graph.add_node("planner", _node_planner)
    graph.add_node("phi_agent", _node_phi_agent)
    graph.add_node("clinical_agent", _node_clinical_agent)
    graph.add_node("knowledge_agent", _node_knowledge_agent)
    graph.add_node("audit_agent", _node_audit_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "phi_agent")
    graph.add_edge("phi_agent", "clinical_agent")
    graph.add_edge("clinical_agent", "knowledge_agent")
    graph.add_edge("knowledge_agent", "audit_agent")
    graph.add_edge("audit_agent", END)

    return graph.compile()


class PatientShieldPipeline:
    """Orchestrates the 5-agent LangGraph pipeline for one document at a time.

    Use when: instantiated once (module-level singleton is fine — the
    compiled graph holds no per-document state) and called via
    `run_pipeline()` from the /process endpoint.
    """

    def __init__(self) -> None:
        self._graph = _build_graph()

    async def run_pipeline(self, document_id: str, text: str) -> PipelineState:
        """Run all 5 nodes in order on one document and return the final state.

        Args:
            document_id: The document's DB id — threaded through every node
                so audit events and results can be correlated back to it.
            text: The document's original (not yet de-identified) text.

        Returns:
            Final PipelineState: results from every node plus the full
            accumulated audit trail and total wall-clock processing_time.

        Use when: called once per POST /process request.
        """
        start = time.perf_counter()
        initial_state: PipelineState = {
            "document_id": document_id,
            "original_text": text,
            "execution_plan": None,
            "phi_results": None,
            "de_identified_text": None,
            "clinical_results": None,
            "knowledge_indexed": False,
            "qa_results": None,
            "audit_events": [],
            "error": None,
            "processing_time": None,
        }
        result: PipelineState = await self._graph.ainvoke(initial_state)
        result["processing_time"] = time.perf_counter() - start
        return result


if __name__ == "__main__":
    import asyncio
    import json
    from unittest.mock import AsyncMock

    from src.orchestrator.llm_manager import LLMResult

    async def _run_self_test() -> None:
        pipeline = PatientShieldPipeline()

        # Clinical Agent (Day 5) makes a real LLM call for any non-empty
        # input, unlike phi_agent's per-candidate design — there's no
        # zero-candidate text that skips the call the way there is for PHI.
        # Monkeypatch the module-level singleton's LLM manager with a canned,
        # zero-findings response so this self-test stays network-free (rule
        # 6) while still exercising the real (not placeholder) node.
        _clinical_agent._llm.get_response = AsyncMock(
            return_value=LLMResult(text=json.dumps({"findings": [], "summary": "No actionable findings."}), provider="fake")
        )

        # Knowledge Agent (Day 6) makes a real HTTP call to UniRAG for any
        # non-empty input. Monkeypatch the module-level singleton's
        # connector so this self-test stays network-free (rule 6) regardless
        # of whether a real UniRAG process happens to be running on this
        # host at test time.
        _knowledge_agent._connector.upload_document = AsyncMock(return_value=1)

        # Deliberately chosen with zero PHI candidates (verified empirically
        # against the real Presidio analyzer): PHIAgent.process() then never
        # calls the LLM, keeping this self-test network-free per rule 6,
        # while still exercising the real (not placeholder) phi_agent node.
        # phi_agent's own "scan_completed" summary event (logged even at
        # zero candidates — see phi_agent.py), plus the node_timing event
        # _with_timing appends (Day 9), makes it exactly two events per node.
        text = "Vitals stable. No acute distress noted. Continue current medications."
        result = await pipeline.run_pipeline("doc-test-1", text)

        assert result["document_id"] == "doc-test-1"
        assert len(result["audit_events"]) == 10, f"expected 10 audit events (2 per node x 5 nodes), got {len(result['audit_events'])}"

        agent_order = list(dict.fromkeys(e.agent_name for e in result["audit_events"]))
        assert agent_order == ["planner", "phi_agent", "clinical_agent", "knowledge_agent", "audit_agent"], agent_order

        # Every node's own real event must be immediately followed by its
        # node_timing event — the ordering _with_timing is supposed to produce.
        timing_events = [e for e in result["audit_events"] if e.action == "node_timing"]
        assert len(timing_events) == 5, f"expected one node_timing event per node, got {len(timing_events)}"
        assert all(e.details["duration_seconds"] >= 0 for e in timing_events)

        assert result["phi_results"] is not None and result["phi_results"]["total_count"] == 0
        assert result["de_identified_text"] == text, "zero PHI candidates means nothing should be masked"
        assert result["clinical_results"] is not None and result["clinical_results"]["findings"] == []
        assert result["clinical_results"]["summary"] == "No actionable findings."
        assert result["knowledge_indexed"] is True
        assert result["processing_time"] is not None and result["processing_time"] >= 0

        # Audit Agent (Day 7): consolidates the other 4 nodes' events. All 4
        # produced two events each above (real + node_timing), so the summary
        # embedded in its own event's details should report a complete run.
        # Index -2, not -1: audit_agent's own node_timing event (added by
        # _with_timing, same as every other node) is what's actually last.
        audit_event = result["audit_events"][-2]
        assert audit_event.agent_name == "audit_agent" and audit_event.action == "audit_consolidated"
        assert audit_event.status == "success", "expected no missing agents in a full 5-node run"
        assert audit_event.details["missing_agents"] == []
        assert audit_event.details["total_events"] == 8  # 2 events each from planner + phi + clinical + knowledge, not counting itself

        print(
            "pipeline.py self-test passed: 5-node graph ran end-to-end in order, "
            "audit trail has 10 events (real + node_timing per node), no external dependency required."
        )

    asyncio.run(_run_self_test())
