"""Audit Agent — consolidates the pipeline's audit trail into one HIPAA-facing summary.

Runs last in the pipeline (Planner -> PHI -> Clinical -> Knowledge -> Audit),
receiving every event the four prior agents logged. Unlike them, it makes no
LLM/HTTP call and has no external dependency — its whole job is arithmetic
and one completeness check over data that already exists.

**Deviation from CLAUDE.md section 16 Day 7's literal text (flagged per
section 2's rule):** that text says this agent "logs to database:
AuditLogRecord table." Day 2's architecture — established when the pipeline
was first built and followed consistently through Days 4-6 — is explicit
that pipeline nodes have no side effects; every event a node returns flows
through `PipelineState` back to `main.py`, which is the only place that
writes to the database, after the full pipeline finishes. Giving this one
agent a direct DB write would both contradict that established pattern and
duplicate work `main.py` already does. This agent instead *consolidates* the
trail (counts, per-agent/per-status breakdown, a completeness check) and logs
one `audit_consolidated` decision carrying that summary — `main.py` persists
it exactly like every other agent's events, no different code path needed.

**Completeness, not just consolidation:** a HIPAA-facing audit trail's real
value is catching the failure mode where a node silently produced zero
events — indistinguishable, from the trail alone, from a node that never ran
at all, unless something explicitly checks for an expected agent's absence.
That check is this agent's one genuinely load-bearing piece of logic, not
just a summary for a UI.
"""

from src.agents.base_agent import BaseAgent
from src.models import AuditEvent, AuditSummary

# The four agents that run before this one in the pipeline (planner -> phi ->
# clinical -> knowledge -> audit) — audit_agent itself is deliberately
# excluded, since it's the one producing this consolidation and hasn't
# logged its own event yet at the point process() computes this.
_EXPECTED_PRIOR_AGENTS = ["planner", "phi_agent", "clinical_agent", "knowledge_agent"]


class AuditAgent(BaseAgent):
    """Consolidates every prior agent's audit events into one summary, flagging any that reported nothing.

    Use when: invoked once per document by the LangGraph pipeline's
    audit_agent node, with the full accumulated `audit_events` list from
    every prior node.
    """

    def __init__(self) -> None:
        super().__init__(agent_name="audit_agent")

    async def process(self, input_data: list[AuditEvent]) -> AuditSummary:
        """Consolidate `input_data` into an AuditSummary and log one completeness-checked decision.

        Args:
            input_data: Every AuditEvent logged by planner/phi_agent/
                clinical_agent/knowledge_agent so far in this pipeline run.

        Returns:
            AuditSummary with total/per-agent/per-status counts and any
            expected agent that produced zero events.

        Use when: called once per document by the pipeline's audit_agent
        node, as the final step before the graph reaches END.
        """
        events = input_data

        events_by_agent: dict[str, int] = {}
        events_by_status: dict[str, int] = {}
        for event in events:
            events_by_agent[event.agent_name] = events_by_agent.get(event.agent_name, 0) + 1
            events_by_status[event.status] = events_by_status.get(event.status, 0) + 1

        reported_agents = set(events_by_agent.keys())
        missing_agents = [agent for agent in _EXPECTED_PRIOR_AGENTS if agent not in reported_agents]

        summary = AuditSummary(
            total_events=len(events),
            events_by_agent=events_by_agent,
            events_by_status=events_by_status,
            missing_agents=missing_agents,
        )

        # A missing agent is an edge condition per rule 8 (fail loud, not
        # quiet): status="error" surfaces it in /audit/{id} and
        # /pipeline-status/{id} rather than the consolidation silently
        # reporting a clean-looking trail that's missing a whole node's work.
        self.log_decision(
            "audit_consolidated",
            status="error" if missing_agents else "success",
            reasoning=(
                f"{summary.total_events} events from {len(reported_agents)}/{len(_EXPECTED_PRIOR_AGENTS)} expected agents"
                + (f"; missing: {missing_agents}" if missing_agents else "")
            ),
            details=summary.model_dump(),
        )

        return summary


if __name__ == "__main__":
    import asyncio

    async def _run_self_test() -> None:
        # No mocking needed (rule 6 trivially satisfied) — this agent makes
        # no external call at all.
        agent = AuditAgent()

        complete_events = [
            AuditEvent(agent_name="planner", action="plan_created", status="success"),
            AuditEvent(agent_name="phi_agent", action="entity_detected", status="success"),
            AuditEvent(agent_name="phi_agent", action="scan_completed", status="success"),
            AuditEvent(agent_name="clinical_agent", action="extraction_completed", status="success"),
            AuditEvent(agent_name="knowledge_agent", action="document_indexed", status="success"),
        ]
        summary = await agent.process(complete_events)
        assert summary.total_events == 5
        assert summary.events_by_agent == {"planner": 1, "phi_agent": 2, "clinical_agent": 1, "knowledge_agent": 1}
        assert summary.events_by_status == {"success": 5}
        assert summary.missing_agents == []

        trail = agent.get_audit_trail()
        assert trail[-1].action == "audit_consolidated" and trail[-1].status == "success"

        # Missing agent -> flagged, not silently dropped.
        incomplete_events = [
            AuditEvent(agent_name="planner", action="plan_created", status="success"),
            AuditEvent(agent_name="phi_agent", action="scan_completed", status="success"),
            AuditEvent(agent_name="clinical_agent", action="extraction_completed", status="error"),
            # knowledge_agent produced nothing this run (e.g. UniRAG was down).
        ]
        incomplete_summary = await agent.process(incomplete_events)
        assert incomplete_summary.missing_agents == ["knowledge_agent"]
        assert incomplete_summary.events_by_status == {"success": 2, "error": 1}
        assert agent.get_audit_trail()[-1].status == "error"

        # Zero events -> every expected agent reported missing, no crash.
        empty_summary = await agent.process([])
        assert empty_summary.total_events == 0
        assert set(empty_summary.missing_agents) == set(_EXPECTED_PRIOR_AGENTS)

        print(
            "audit_agent.py self-test passed: consolidation, per-agent/per-status counts, and "
            "missing-agent completeness checking verified (no external dependency to mock)."
        )

    asyncio.run(_run_self_test())
