"""Abstract base class every pipeline agent inherits from.

Each of the 5 agents (Planner, PHI, Clinical, Knowledge, Audit) does very
different work, but they share two obligations: implement `process()`, and
call `log_decision()` for anything worth explaining later. Centralizing
`log_decision()` here — rather than letting each agent format its own audit
entries — is what makes the audit trail's shape consistent enough for the
Audit Agent (Day 7) and the AuditTrail UI page to rely on.

Agents do not write to the database themselves (see pipeline.py, Day 2: "no
side effects — database writes happen in main.py after pipeline completes").
`log_decision()` only appends to an in-memory list; the pipeline collects it
via `get_audit_trail()` after each node runs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.models import AuditEvent

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Shared contract for the 5 pipeline agents.

    Use when: subclassed by PlannerAgent, PHIAgent, ClinicalAgent,
    KnowledgeAgent, AuditAgent — never instantiated directly.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._audit_events: list[AuditEvent] = []

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Run this agent's core logic and return its typed result.

        Args:
            input_data: Agent-specific input (e.g. raw text for PHIAgent,
                de-identified text for ClinicalAgent). Subclasses narrow this
                type in their own signature.

        Returns:
            Agent-specific Pydantic result (e.g. PHIDetectionResult).

        Use when: called once per document by a LangGraph pipeline node.
        """
        raise NotImplementedError

    def log_decision(
        self,
        action: str,
        status: str = "success",
        confidence: float | None = None,
        reasoning: str | None = None,
        details: dict | None = None,
    ) -> AuditEvent:
        """Record one traceable decision to this agent's in-memory audit trail.

        Args:
            action: Short verb phrase, e.g. "entity_detected", "plan_created".
            status: "success" | "error" | "skipped".
            confidence: Optional 0.0-1.0 score behind this decision.
            reasoning: Optional human-readable justification — this is what
                turns a log line into something a HIPAA auditor can act on.
            details: Any extra structured context (entity spans, thresholds
                applied, etc).

        Returns:
            The AuditEvent that was recorded, in case the caller needs it
            (e.g. to attach to a specific finding).

        Use when: every time an agent makes a decision worth explaining —
        not for routine control flow. Per rule 8, log the middle quietly:
        confidence variance is INFO, not an audit-worthy decision by itself.
        """
        merged_details = dict(details or {})
        if confidence is not None:
            merged_details["confidence"] = confidence
        if reasoning is not None:
            merged_details["reasoning"] = reasoning

        event = AuditEvent(agent_name=self.agent_name, action=action, status=status, details=merged_details or None)
        self._audit_events.append(event)
        logger.info("[%s] %s (%s)", self.agent_name, action, status)
        return event

    def get_audit_trail(self) -> list[AuditEvent]:
        """Return a copy of every decision logged so far this run.

        Returns:
            List of AuditEvent, oldest first.

        Use when: called by the pipeline node after `process()` returns, to
        merge this agent's events into the shared PipelineState audit trail.
        """
        return list(self._audit_events)


if __name__ == "__main__":
    import asyncio

    class _EchoAgent(BaseAgent):
        """Minimal concrete agent used only to exercise BaseAgent's shared behavior."""

        async def process(self, input_data: Any) -> Any:
            self.log_decision("echoed_input", confidence=1.0, reasoning="Self-test: no real logic.")
            return input_data

    async def _run_self_test() -> None:
        agent = _EchoAgent(agent_name="echo")
        result = await agent.process("hello")
        assert result == "hello"
        trail = agent.get_audit_trail()
        assert len(trail) == 1
        assert trail[0].agent_name == "echo"
        assert trail[0].details["confidence"] == 1.0
        print("base_agent.py self-test passed: process() ran and log_decision() recorded one event.")

    asyncio.run(_run_self_test())
