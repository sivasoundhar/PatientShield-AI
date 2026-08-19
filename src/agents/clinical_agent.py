"""Clinical Analyzer Agent — extracts diagnoses, medications, findings, and follow-ups.

Per CLAUDE.md section 16 Day 5, this agent's input is already de-identified
text (PHI Agent runs first in the pipeline) — it never sees raw PHI, only the
masked note. Unlike PHI Agent's Presidio-candidates-then-LLM-verify design,
there's no cheap regex pre-filter for "is this a diagnosis" the way there is
for "is this shaped like an SSN" — clinical extraction is read-the-note work,
so this agent makes one structured-JSON LLM call per document rather than
many small per-candidate calls.

Four categories (FindingCategory in src/models.py): DIAGNOSIS, MEDICATION,
FINDING, FOLLOW_UP. Each finding carries two independent scores per CLAUDE.md
section 16: `confidence` (how clearly the text states this) and
`priority_score` (clinical urgency/severity) — a clearly-stated but minor
finding and a vaguely-implied but life-threatening one land in opposite
corners of that 2D space, and conflating them into one score would lose
exactly the distinction the priority filter needs.
"""

import logging

from src.agents.base_agent import BaseAgent
from src.config import settings
from src.models import ClinicalAnalysisResult, ClinicalFinding, FindingCategory
from src.orchestrator.llm_manager import LLMManager
from src.utils.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {c.value for c in FindingCategory}

_EXTRACTION_PROMPT = """You are a clinical documentation analyst extracting structured findings from a \
de-identified medical note. Extract every diagnosis, medication, clinical finding, and follow-up \
instruction that is ACTUALLY PRESENT in the note.

Categories:
- DIAGNOSIS: a condition the patient has or is being treated for.
- MEDICATION: a drug the patient is taking or was prescribed, including dosage when stated.
- FINDING: a lab result, vital sign, or exam observation.
- FOLLOW_UP: a next step, referral, or scheduled action, including its timing when stated.

Rule 1 — negation excludes a finding entirely. "Patient denies chest pain", "no history of diabetes", \
"ruled out sepsis" describe the ABSENCE of something — do not extract these, not even with low confidence.
Rule 2 — priority_score reflects clinical urgency/severity, not how prominently it's mentioned. A \
diagnosis of sepsis or active GI bleeding is high priority (0.85-1.00); a stable chronic condition being \
routinely monitored (e.g. well-controlled hypertension) is lower (0.40-0.60); an incidental or minor \
finding is lower still.
Rule 3 — confidence reflects how clearly the text states this, not clinical importance. An explicit \
statement ("Diagnosis: Type 2 diabetes") is high confidence; an inferred or ambiguous mention is lower.

Worked examples:
- "Patient denies chest pain." -> no finding extracted (negated)
- "Diagnosis: Sepsis secondary to UTI." -> DIAGNOSIS "Sepsis secondary to UTI", priority_score 0.95 (life-threatening)
- "Continue metformin 500mg twice daily." -> MEDICATION "Metformin 500mg twice daily", priority_score 0.55
- "Blood pressure 210/120, hypertensive urgency noted." -> FINDING "Blood pressure 210/120 (hypertensive urgency)", priority_score 0.90
- "Follow up with cardiology in 2 weeks." -> FOLLOW_UP "Follow up with cardiology in 2 weeks", priority_score 0.65

Respond with ONLY a JSON object, no markdown code fences, no commentary, in exactly this shape:
{{"findings": [{{"category": "DIAGNOSIS", "value": "...", "priority_score": 0.0, "confidence": 0.0}}], \
"summary": "one or two sentence clinical summary of this note"}}

De-identified clinical note:
\"\"\"
{text}
\"\"\""""


class ClinicalAgent(BaseAgent):
    """Extracts prioritized clinical findings from de-identified text via one structured LLM call.

    Use when: invoked once per document by the LangGraph pipeline's
    clinical_agent node, with the PHI Agent's de-identified text (never the
    original, PHI-containing text).
    """

    def __init__(self, llm_manager: LLMManager | None = None) -> None:
        super().__init__(agent_name="clinical_agent")
        self._llm = llm_manager or LLMManager()

    async def process(self, input_data: str) -> ClinicalAnalysisResult:
        """Extract findings and a summary from `input_data`, keeping only high-priority findings.

        Args:
            input_data: De-identified clinical note text.

        Returns:
            ClinicalAnalysisResult with only findings whose priority_score
            meets settings.CLINICAL_PRIORITY_THRESHOLD, plus a summary.

        Raises:
            RuntimeError: the LLM's response couldn't be parsed as the
                expected JSON shape after extraction. This is an edge
                condition (rule 8) — a whole-document extraction failure is
                not the same as "found nothing," so it must not be silently
                swallowed into an empty result.

        Use when: called once per document by the pipeline's clinical_agent node.
        """
        text = input_data
        if not text.strip():
            self.log_decision("extraction_completed", details={"findings_extracted": 0, "findings_kept": 0, "note": "empty document"})
            return ClinicalAnalysisResult(findings=[], summary="")

        prompt = _EXTRACTION_PROMPT.format(text=text)
        # LLM's structured JSON output can run longer than a short PHI
        # verification response (many findings + a summary), so this call
        # asks for more headroom than the default token cap.
        result = await self._llm.get_response(prompt, max_tokens=max(settings.LLM_MAX_TOKENS, 1500))

        parsed = self._parse_response(result.text)

        findings: list[ClinicalFinding] = []
        for raw in parsed.get("findings", []):
            finding = self._to_finding(raw)
            if finding is None:
                continue  # malformed individual entry — log and skip, not a whole-document failure

            if finding.priority_score >= settings.CLINICAL_PRIORITY_THRESHOLD:
                findings.append(finding)
                self.log_decision(
                    "finding_extracted",
                    confidence=finding.confidence,
                    reasoning=f"{finding.category.value}: {finding.value}",
                    details={"priority_score": finding.priority_score},
                )
            else:
                # Quiet middle-of-pipeline path (rule 8): below-threshold
                # findings are the priority filter doing its job, not an error.
                self.log_decision(
                    "finding_below_threshold",
                    status="skipped",
                    confidence=finding.confidence,
                    reasoning=f"{finding.category.value}: {finding.value}",
                    details={"priority_score": finding.priority_score},
                )

        summary = str(parsed.get("summary", "")).strip()

        # Always emit a summary decision, even at zero findings — mirrors
        # PHIAgent's scan_completed pattern (Day 4): an audit trail with no
        # clinical_agent entry at all is indistinguishable from the node
        # never running, which is worse than an entry that honestly says
        # "found nothing above threshold."
        self.log_decision(
            "extraction_completed",
            details={"findings_extracted": len(parsed.get("findings", [])), "findings_kept": len(findings)},
        )

        return ClinicalAnalysisResult(findings=findings, summary=summary)

    def _parse_response(self, raw_text: str) -> dict:
        """Parse the LLM's JSON response via the shared llm_json repair helper.

        Raises:
            RuntimeError: response isn't valid JSON even after
                parse_llm_json's repairs, or isn't shaped like
                {"findings": [...], "summary": "..."} — an unparseable
                whole-document response is an edge condition (rule 8), not a
                middle-of-pipeline degradation to silently paper over.

        Use when: called once per process() call, immediately after the
        extraction LLM call returns.
        """
        try:
            parsed = parse_llm_json(raw_text)
        except ValueError as exc:
            # ValueError here already means parse_llm_json's own
            # trailing-comma repair (Day 5's original fix — the single most
            # common malformation smaller/weaker models produce, confirmed
            # live during Day 7 integration testing) didn't save it either.
            raise RuntimeError(f"Clinical extraction LLM response was not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict) or "findings" not in parsed:
            raise RuntimeError(f"Clinical extraction LLM response missing expected 'findings' key. Raw response: {raw_text!r}")

        return parsed

    def _to_finding(self, raw: dict) -> ClinicalFinding | None:
        """Convert one raw JSON finding into a ClinicalFinding, or None if it's malformed.

        A single malformed entry (missing field, unknown category, unparseable
        score) is a middle-of-pipeline event per rule 8 — log and skip that
        one entry rather than failing the entire document's extraction over it.

        Use when: called once per entry in the LLM response's "findings" list.
        """
        try:
            category = str(raw["category"]).strip().upper()
            if category not in _VALID_CATEGORIES:
                raise ValueError(f"unknown category {category!r}")
            value = str(raw["value"]).strip()
            if not value:
                raise ValueError("empty value")
            priority_score = max(0.0, min(1.0, float(raw["priority_score"])))
            confidence = max(0.0, min(1.0, float(raw["confidence"])))
        except (KeyError, ValueError, TypeError) as exc:
            logger.info("Skipping malformed clinical finding entry %r: %s", raw, exc)
            self.log_decision(
                "finding_parse_error",
                status="skipped",
                reasoning=str(exc),
                details={"raw_entry": raw if isinstance(raw, dict) else str(raw)},
            )
            return None

        return ClinicalFinding(category=FindingCategory(category), value=value, priority_score=priority_score, confidence=confidence)


if __name__ == "__main__":
    import asyncio
    import json
    from unittest.mock import AsyncMock

    from src.orchestrator.llm_manager import LLMResult

    async def _run_self_test() -> None:
        # Self-test per rule 6: no real network/LLM call. Monkeypatch the LLM
        # manager to return a canned, deterministic JSON response so this
        # exercises parse -> filter -> result without needing GROQ_API_KEY or
        # a running Ollama.
        llm = LLMManager()
        canned = json.dumps(
            {
                "findings": [
                    {"category": "DIAGNOSIS", "value": "Sepsis secondary to UTI", "priority_score": 0.95, "confidence": 0.9},
                    {"category": "MEDICATION", "value": "Metformin 500mg twice daily", "priority_score": 0.75, "confidence": 0.85},
                    {"category": "FINDING", "value": "Incidental note, not clinically significant", "priority_score": 0.2, "confidence": 0.6},
                ],
                "summary": "Patient treated for sepsis secondary to UTI; continued on metformin for diabetes.",
            }
        )
        llm.get_response = AsyncMock(return_value=LLMResult(text=f"```json\n{canned}\n```", provider="fake"))

        agent = ClinicalAgent(llm_manager=llm)
        result = await agent.process("De-identified note text (content irrelevant — response is canned).")

        assert any(f.category == FindingCategory.DIAGNOSIS and "Sepsis" in f.value for f in result.findings)
        assert any(f.category == FindingCategory.MEDICATION and "Metformin" in f.value for f in result.findings)
        # priority_score 0.2 is below CLINICAL_PRIORITY_THRESHOLD (0.70 default) -> filtered out
        assert not any("Incidental" in f.value for f in result.findings)
        assert result.summary.startswith("Patient treated for sepsis")

        trail = agent.get_audit_trail()
        assert any(e.action == "finding_extracted" for e in trail)
        assert any(e.action == "finding_below_threshold" for e in trail)
        assert any(e.action == "extraction_completed" for e in trail)

        # Empty document short-circuits without an LLM call.
        empty_llm = LLMManager()
        empty_llm.get_response = AsyncMock(side_effect=AssertionError("should not call LLM for empty input"))
        empty_agent = ClinicalAgent(llm_manager=empty_llm)
        empty_result = await empty_agent.process("   ")
        assert empty_result.findings == [] and empty_result.summary == ""

        # Trailing-comma JSON repair (found live during Day 7 integration
        # testing against GROQ_MODEL_FALLBACK's real output — see
        # _TRAILING_COMMA_RE's comment): a response with a comma before the
        # closing bracket must still parse, not raise.
        malformed_llm = LLMManager()
        malformed_json = (
            '{"findings": [{"category": "DIAGNOSIS", "value": "Pneumonia", "priority_score": 0.85, "confidence": 1.0},],'
            ' "summary": "Patient with pneumonia."}'
        )
        malformed_llm.get_response = AsyncMock(return_value=LLMResult(text=malformed_json, provider="fake"))
        malformed_agent = ClinicalAgent(llm_manager=malformed_llm)
        malformed_result = await malformed_agent.process("De-identified note text.")
        assert any("Pneumonia" in f.value for f in malformed_result.findings)

        print(
            "clinical_agent.py self-test passed: JSON extraction + priority filtering + "
            "audit logging verified without any network dependency."
        )

    asyncio.run(_run_self_test())
