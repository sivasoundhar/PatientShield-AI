"""Audit Agent test suite — 5 cases per CLAUDE.md section 16 Day 7 ("event logging completeness").

No mocking needed, unlike every other agent's test file — AuditAgent makes no
LLM/HTTP call at all, so every test here is fully deterministic.
"""

import pytest

# Imports the module's private _EXPECTED_PRIOR_AGENTS constant deliberately
# (rather than a hardcoded duplicate list here) so this test can never drift
# out of sync with the real list audit_agent.py checks against.
from src.agents.audit_agent import AuditAgent, _EXPECTED_PRIOR_AGENTS
from src.models import AuditEvent


@pytest.fixture()
def audit_agent() -> AuditAgent:
    return AuditAgent()


async def test_consolidates_events_from_all_agents(audit_agent: AuditAgent):
    events = [
        AuditEvent(agent_name="planner", action="plan_created"),
        AuditEvent(agent_name="phi_agent", action="entity_detected"),
        AuditEvent(agent_name="phi_agent", action="scan_completed"),
        AuditEvent(agent_name="clinical_agent", action="extraction_completed"),
        AuditEvent(agent_name="knowledge_agent", action="document_indexed"),
    ]
    summary = await audit_agent.process(events)

    assert summary.events_by_agent == {"planner": 1, "phi_agent": 2, "clinical_agent": 1, "knowledge_agent": 1}
    assert summary.missing_agents == []


async def test_detects_missing_agent(audit_agent: AuditAgent):
    """A node that reported zero events must be flagged, not silently absorbed into a clean-looking summary."""
    events = [
        AuditEvent(agent_name="planner", action="plan_created"),
        AuditEvent(agent_name="phi_agent", action="scan_completed"),
        AuditEvent(agent_name="clinical_agent", action="extraction_completed"),
        # knowledge_agent produced nothing this run.
    ]
    summary = await audit_agent.process(events)

    assert summary.missing_agents == ["knowledge_agent"]
    trail = audit_agent.get_audit_trail()
    assert trail[-1].status == "error", "a missing agent must fail loud (rule 8), not log as a clean success"


async def test_total_event_count_correct(audit_agent: AuditAgent):
    events = [AuditEvent(agent_name="planner", action="plan_created")] * 3 + [AuditEvent(agent_name="phi_agent", action="entity_detected")] * 2
    summary = await audit_agent.process(events)
    assert summary.total_events == 5


async def test_events_by_status_breakdown_correct(audit_agent: AuditAgent):
    events = [
        AuditEvent(agent_name="planner", action="plan_created", status="success"),
        AuditEvent(agent_name="phi_agent", action="entity_rejected", status="skipped"),
        AuditEvent(agent_name="clinical_agent", action="extraction_completed", status="error"),
        AuditEvent(agent_name="knowledge_agent", action="document_indexed", status="success"),
    ]
    summary = await audit_agent.process(events)
    assert summary.events_by_status == {"success": 2, "skipped": 1, "error": 1}


async def test_empty_events_list_reports_all_agents_missing(audit_agent: AuditAgent):
    summary = await audit_agent.process([])
    assert summary.total_events == 0
    assert set(summary.missing_agents) == set(_EXPECTED_PRIOR_AGENTS)
