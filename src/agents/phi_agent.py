"""PHI Reasoner Agent — context-aware protected health information detection.

Two-stage design, per CLAUDE.md section 16 Day 4:

1. **Presidio** (regex + spaCy NER) finds *candidate* spans fast and cheaply
   — names, dates, phone numbers, SSNs, addresses, plus a custom MRN
   pattern Presidio doesn't ship. This stage is deliberately over-inclusive;
   Presidio alone can't tell "Dr. Smith" (a clinician, not PHI) from
   "Patient Smith" (PHI), or a specific date-of-birth from an incidental
   date mention like "follow up in March".

2. **LLM context verification** judges each candidate against the sentence
   it actually sits in, and returns its own confidence + reasoning. Only
   candidates whose *final* confidence clears PHI_CONFIDENCE_THRESHOLD make
   it into the result — this is the "context-aware" half of the agent's
   job, and it's what the Presidio-only regex stage cannot do by itself.

Why not skip Presidio and ask the LLM to find PHI directly? Recall: an LLM
asked to freeform-scan a document for PHI will miss low-salience spans
(a phone number buried mid-paragraph) that a regex never misses. Presidio
provides recall; the LLM provides precision. Neither stage alone is enough.
"""

import logging
import re
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from src.agents.base_agent import BaseAgent
from src.config import settings
from src.models import PHIDetectionResult, PHIEntity
from src.orchestrator.llm_manager import LLMManager
from src.utils.llm_json import parse_llm_json

logger = logging.getLogger(__name__)

# Presidio entity types worth surfacing in a US clinical document. Restricting
# to this list (rather than Presidio's full ~30-entity default set) skips
# entity types irrelevant here (AU_ABN, IBAN_CODE, CRYPTO, ...) that would
# otherwise burn LLM verification calls on candidates that can never be
# clinically relevant.
_TARGET_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "DATE_TIME",
    "LOCATION",
    "MRN",  # custom recognizer — see _build_analyzer()
]


def _build_analyzer() -> AnalyzerEngine:
    """Construct Presidio's AnalyzerEngine with the project's spaCy model and a custom MRN recognizer.

    Returns:
        A ready-to-use AnalyzerEngine.

    Use when: called once per PHIAgent instance (spaCy model loading is the
    expensive part — never rebuild this per document).
    """
    # Presidio defaults to en_core_web_lg if not told otherwise; we only
    # install the small model (PHI_SPACY_MODEL) to keep the footprint down.
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": settings.PHI_SPACY_MODEL}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine())

    # Medical Record Number isn't one of Presidio's built-in entities.
    # Matches "MRN: 00482913", "MRN-458921", "MRN 00482913" — the formats
    # seen in the sample fixtures and UI reference mockups.
    mrn_recognizer = PatternRecognizer(
        supported_entity="MRN",
        patterns=[Pattern(name="MRN pattern", regex=r"\bMRN[:\-#\s]*\d{5,10}\b", score=0.75)],
        context=["mrn", "medical record", "record number", "chart number"],
    )
    analyzer.registry.add_recognizer(mrn_recognizer)

    return analyzer


@dataclass
class _Verification:
    """Result of asking the LLM whether one candidate span is genuinely identifying PHI."""

    is_phi: bool
    confidence: float
    reasoning: str


# Batched, not one call per candidate — Day 9 optimization. CLAUDE.md section
# 16 Day 9 names this exact fix for a slow PHI Agent ("reduce number of
# context verification calls"), and real profiling during this project's own
# Day 8 system testing confirmed PHI verification was the dominant cost: a
# document with N Presidio candidates made N separate LLM round-trips before
# this change, vs. exactly 1 now (still 0 when there are zero candidates to
# verify — see process()). Each candidate still gets its own independent
# judgment in the response; batching changes how many HTTP round-trips that
# takes, not the judgment logic itself — same three rules, same worked
# examples, same per-candidate confidence/reasoning.
#
# Candidate list placement is deliberate, not incidental: first draft put it
# right after the intro line (worked examples came last, right before the
# response instruction) and a real live run against the Groq fallback model
# exposed the failure mode directly — its JSON response's `reasoning` fields
# were near-paraphrases of the WORKED EXAMPLES below, not judgments about the
# actual candidates sent, and the returned array length didn't match the
# real candidate count either. Recency: the worked examples, being the last
# candidate-shaped content before "Respond with...", got mistaken for the
# real input by a smaller model. Fix: the real candidate list is now the
# last thing before the response instruction, and the worked examples are
# explicitly labeled illustrative-only, positioned earlier.
_BATCH_VERIFICATION_PROMPT = """You are reviewing candidate PHI (Protected Health Information) entities \
detected in a clinical document. For EACH candidate, judge whether it genuinely identifies the PATIENT.

Rule 1 — a name is PHI only if it identifies the PATIENT, not a clinician. If the name is preceded \
by a professional title (Dr., Nurse, RN, MD, NP, PA, Prof.) and nothing in the context marks that \
same person as the patient, it is a clinician's name and is NOT PHI — the title is evidence AGAINST \
it being PHI, not evidence for it, regardless of how specific-sounding the name is.
Rule 2 — a date is PHI only if it identifies a specific point tied to the patient (date of birth, \
admission date, a specific visit date with a year). A relative or partial date used only to schedule \
a future action (e.g. "follow up in March", "recheck in 2 weeks") is NOT PHI.
Rule 3 — anything else that could reasonably be used, alone or combined with other information in \
this document, to identify the specific patient IS PHI (SSN, MRN, DOB, phone, email, home address, \
full patient name).

The following worked examples are for illustration ONLY — they are not real candidates, do not \
include them in your response, and do not let their wording influence your reasoning text for the \
real candidates below:
- Candidate "Smith", context "seen by Dr. Smith for a checkup" -> is_phi: false (title marks this as the clinician)
- Candidate "Smith", context "Patient Smith reports no chest pain" -> is_phi: true (this names the patient)
- Candidate "March 15", context "follow up scheduled for March 15" -> is_phi: false (scheduling reference, no year, not tied to the patient's own history)
- Candidate "03/15/1965", context "Patient DOB: 03/15/1965" -> is_phi: true (date of birth)

Here are the {count} REAL candidates from the actual document — respond about these and ONLY these:

{candidate_list}

Respond with ONLY a JSON array of exactly {count} objects, one per real candidate listed immediately \
above (not the worked examples), IN THE SAME ORDER LISTED, nothing else — no markdown fences, no \
commentary, no text before or after the array:
[{{"is_phi": true or false, "confidence": 0.00 to 1.00, "reasoning": "one sentence"}}, ...]"""

# Deterministic pre-check for CLAUDE.md's flagship "Dr. Smith is not PHI"
# case. This one is a clear, well-defined pattern — a title immediately
# before a name — so it's handled with a rule rather than left entirely to
# LLM judgment. A free-tier local model (this project's Ollama fallback) is
# good but not perfectly consistent on this exact judgment call every
# single time; a regex is. Reserve the LLM's judgment for cases that
# actually require reading comprehension, not pattern matching.
#
# No separate "unless nearby text calls them the patient" override: a title
# and a "Patient " qualifier can't both be the words immediately before the
# same name at once, so there's nothing for such an override to catch — an
# earlier version scanned a wider window for the word "patient" as an
# override signal and broke on "Patient was seen by Dr. Smith", where
# "Patient" is the sentence's unrelated grammatical subject, not a qualifier
# on "Smith" at all.
_CLINICIAN_TITLE_RE = re.compile(r"\b(Dr|Doctor|Nurse|RN|MD|NP|PA|Prof)\.?\s*$", re.IGNORECASE)


def _clinician_title_precedes(text: str, start: int) -> bool:
    """True if a professional title immediately precedes this span (e.g. "Dr. Smith")."""
    immediately_before = text[max(0, start - 20) : start]
    return bool(_CLINICIAN_TITLE_RE.search(immediately_before))


def mask_text(text: str, entities: list[PHIEntity]) -> str:
    """Replace every detected PHI span with a `[ENTITY_TYPE]` placeholder.

    Args:
        text: The original document text the entities' positions refer to.
        entities: PHIEntity list from PHIAgent.process() — masks the
            positions exactly as detected, so this must be called with the
            same text the entities were detected against.

    Returns:
        De-identified text. Per CLAUDE.md section 17, this masks with a
        placeholder (e.g. "[PERSON]") rather than generating a fake
        replacement name/date — safer for privacy, not intended to look
        like realistic synthetic data.

    Use when: called once per document by the pipeline's phi_agent node,
    immediately after PHIAgent.process() returns its entities for that text.
    """
    masked = text
    # Replace highest start_pos first so an earlier replacement's changed
    # length never shifts the offsets of a not-yet-processed span.
    for entity in sorted(entities, key=lambda e: e.start_pos, reverse=True):
        masked = masked[: entity.start_pos] + f"[{entity.entity_type}]" + masked[entity.end_pos :]
    return masked


class PHIAgent(BaseAgent):
    """Detects PHI with Presidio, then verifies each candidate against its context with an LLM.

    Use when: invoked once per document by the LangGraph pipeline's
    phi_agent node, with the document's raw (not yet de-identified) text.
    """

    def __init__(self, llm_manager: LLMManager | None = None) -> None:
        super().__init__(agent_name="phi_agent")
        self._analyzer = _build_analyzer()
        self._llm = llm_manager or LLMManager()

    async def process(self, input_data: str) -> PHIDetectionResult:
        """Detect and context-verify PHI entities in `input_data`.

        Args:
            input_data: The document's raw text (before any de-identification).

        Returns:
            PHIDetectionResult containing only entities whose final
            confidence (Presidio candidate + LLM verification) meets
            settings.PHI_CONFIDENCE_THRESHOLD. `total_count` and
            `high_confidence_count` are equal by construction: every
            entity that clears the threshold to be included is, by
            definition, high-confidence — there's no second, laxer bar
            for "included but not high-confidence."

        Use when: called once per document by the pipeline's phi_agent node.
        """
        text = input_data
        candidates = self._analyzer.analyze(text=text, language="en", entities=_TARGET_ENTITIES)

        # Split off what the deterministic clinician-title pre-check can
        # already resolve for free — only the rest goes into the one batch
        # LLM call below (Day 9: see _BATCH_VERIFICATION_PROMPT's comment).
        needs_verification = []
        for candidate in candidates:
            if candidate.entity_type == "PERSON" and _clinician_title_precedes(text, candidate.start):
                self.log_decision(
                    "entity_rejected",
                    status="skipped",
                    confidence=0.0,
                    reasoning="Preceded by a professional title (Dr./Nurse/RN/MD/...) with no "
                    "nearby 'patient' qualifier — treated as a clinician's name, not the patient.",
                    details={"entity_type": candidate.entity_type, "presidio_score": candidate.score, "rule": "clinician_title_prefix"},
                )
                continue
            needs_verification.append(candidate)

        verifications = await self._verify_batch(text, needs_verification)

        entities: list[PHIEntity] = []
        for candidate, verification in zip(needs_verification, verifications):
            candidate_text = text[candidate.start : candidate.end]
            if verification.confidence >= settings.PHI_CONFIDENCE_THRESHOLD and verification.is_phi:
                entities.append(
                    PHIEntity(
                        entity_type=candidate.entity_type,
                        text=candidate_text,
                        confidence=verification.confidence,
                        start_pos=candidate.start,
                        end_pos=candidate.end,
                        reasoning=verification.reasoning,
                    )
                )
                self.log_decision(
                    "entity_detected",
                    confidence=verification.confidence,
                    reasoning=verification.reasoning,
                    details={"entity_type": candidate.entity_type, "presidio_score": candidate.score},
                )
            else:
                # Quiet middle-of-pipeline path (rule 8): a rejected candidate
                # isn't a failure, it's the context check doing its job.
                self.log_decision(
                    "entity_rejected",
                    status="skipped",
                    confidence=verification.confidence,
                    reasoning=verification.reasoning,
                    details={"entity_type": candidate.entity_type, "presidio_score": candidate.score},
                )

        # Always emit a summary decision, even when zero candidates were
        # found — an audit trail with literally no phi_agent entry for a
        # document (as opposed to an entry saying "found nothing") is a
        # worse compliance record, and per-node presence in the pipeline's
        # audit trail shouldn't depend on whether this particular document
        # happened to contain any PHI-shaped text.
        self.log_decision(
            "scan_completed",
            details={"candidates_evaluated": len(candidates), "entities_kept": len(entities)},
        )

        return PHIDetectionResult(
            entities=entities,
            total_count=len(entities),
            high_confidence_count=len(entities),
            precision_estimate=None,  # populated Days 8-9 against ground truth
        )

    def _context_window(self, text: str, start: int, end: int) -> str:
        """Return the text around [start:end], bounded by PHI_CONTEXT_WINDOW_CHARS on each side."""
        window = settings.PHI_CONTEXT_WINDOW_CHARS
        return text[max(0, start - window) : min(len(text), end + window)]

    async def _verify_batch(self, text: str, candidates: list) -> list[_Verification]:
        """Ask the LLM to judge every remaining candidate in ONE call — see _BATCH_VERIFICATION_PROMPT's comment.

        Args:
            text: The document's full text (candidates' start/end are
                offsets into this).
            candidates: Presidio RecognizerResult objects that survived the
                deterministic clinician-title pre-check.

        Returns:
            One _Verification per candidate, in the same order. Falls back
            to Presidio's own raw score for every candidate — uniformly,
            not per-candidate — if the batch call fails outright, the
            response can't be parsed, or the response isn't shaped as
            exactly one judgment per candidate. A parsing/count mismatch is
            a middle-of-pipeline degradation (rule 8) applied across the
            whole batch, not a reason to drop candidates or crash the
            document — the same fallback semantics the pre-Day-9
            per-candidate design had, just applied once instead of N times.

        Use when: called once per process() call (not once per candidate)
        with every candidate needing genuine LLM judgment.
        """
        if not candidates:
            return []

        candidate_list = "\n".join(
            f'Candidate {i + 1}: entity_type={c.entity_type}, text="{text[c.start:c.end]}", '
            f'context="...{self._context_window(text, c.start, c.end)}..."'
            for i, c in enumerate(candidates)
        )
        prompt = _BATCH_VERIFICATION_PROMPT.format(count=len(candidates), candidate_list=candidate_list)

        def _fallback(reason: str) -> list[_Verification]:
            return [_Verification(is_phi=True, confidence=c.score, reasoning=reason) for c in candidates]

        try:
            # Response scales with candidate count now that they're batched
            # — generous enough for a realistically-sized document's worth
            # of candidates without capping the JSON array mid-way.
            result = await self._llm.get_response(prompt, max_tokens=max(settings.LLM_MAX_TOKENS, 150 * len(candidates)))
        except Exception as exc:  # noqa: BLE001 - LLM unavailable is a middle-of-pipeline event here
            logger.info("PHI batch verification LLM call failed for %d candidates, falling back to Presidio scores: %s", len(candidates), exc)
            return _fallback("LLM unavailable; used Presidio's own detection confidence.")

        try:
            parsed = parse_llm_json(result.text)
        except ValueError as exc:
            logger.info("Could not parse PHI batch verification response for %d candidates: %s", len(candidates), exc)
            return _fallback("LLM response unparseable; used Presidio's own detection confidence.")

        if not isinstance(parsed, list) or len(parsed) != len(candidates):
            logger.info(
                "PHI batch verification response shape mismatch (expected %d items, got %s)",
                len(candidates),
                parsed if isinstance(parsed, list) else type(parsed).__name__,
            )
            return _fallback("LLM response shape mismatch; used Presidio's own detection confidence.")

        return [self._parse_single_verification(item, candidate.score) for candidate, item in zip(candidates, parsed)]

    def _parse_single_verification(self, item: object, presidio_score: float) -> _Verification:
        """Convert one raw JSON judgment (an element of the batch response array) into a _Verification.

        A single malformed entry (missing field, wrong type) falls back to
        Presidio's own score for just that one candidate — a middle-of-
        pipeline event, same as the batch-level fallbacks in
        _verify_batch(), just scoped to one item instead of the whole batch.

        Use when: called once per candidate by _verify_batch(), after the
        response has already been confirmed to be a list of the right length.
        """
        try:
            is_phi = bool(item["is_phi"])
            confidence = max(0.0, min(1.0, float(item["confidence"])))
            reasoning = str(item.get("reasoning", "")).strip() or "No reasoning provided."
        except (KeyError, ValueError, TypeError):
            logger.info("Malformed PHI verification entry %r; falling back to Presidio score.", item)
            return _Verification(is_phi=True, confidence=presidio_score, reasoning="Malformed judgment entry; used Presidio's own detection confidence.")

        # A "false" judgment means "not PHI" — its stated confidence
        # describes confidence in *that* judgment, not confidence that the
        # candidate is PHI, so it must never itself clear
        # PHI_CONFIDENCE_THRESHOLD.
        if not is_phi:
            confidence = 0.0

        return _Verification(is_phi=is_phi, confidence=confidence, reasoning=reasoning)


if __name__ == "__main__":
    import asyncio
    import json
    from unittest.mock import AsyncMock

    from src.orchestrator.llm_manager import LLMResult

    async def _run_self_test() -> None:
        # Self-test per rule 6: no real network/LLM call. Monkeypatch the LLM
        # manager to return one canned, deterministic batch-verification
        # response — Day 9's batching means a single call now covers every
        # candidate needing LLM judgment, so this exercises the Presidio ->
        # batch-verify -> filter -> result pipeline without needing
        # GROQ_API_KEY or a running Ollama.
        llm = LLMManager()

        def _canned_response(prompt: str, max_tokens: int | None = None) -> LLMResult:
            # Read back which candidates the agent actually put in the batch
            # prompt, in the order it listed them (Presidio's own candidate
            # order isn't guaranteed) — "Dr. Johnson" never reaches here at
            # all, since the deterministic clinician-title pre-check filters
            # it out before any LLM call is made.
            candidate_texts = re.findall(r'text="([^"]*)"', prompt)
            judgments = [
                {"is_phi": True, "confidence": 0.95, "reasoning": "Full patient name."}
                if candidate_text == "John Smith"
                else {"is_phi": True, "confidence": 0.85, "reasoning": "Plausible identifier."}
                for candidate_text in candidate_texts
            ]
            return LLMResult(text=json.dumps(judgments), provider="fake")

        llm.get_response = AsyncMock(side_effect=lambda prompt, max_tokens=None: _canned_response(prompt, max_tokens))

        agent = PHIAgent(llm_manager=llm)
        text = "Patient: John Smith. Seen by Dr. Johnson. MRN: 00482913."
        result = await agent.process(text)

        assert result.total_count == result.high_confidence_count
        assert any(e.text == "John Smith" and e.entity_type == "PERSON" for e in result.entities)
        assert not any(e.entity_type == "PERSON" and "Johnson" in e.text for e in result.entities), (
            "Dr. Johnson should have been rejected by the deterministic clinician-title pre-check"
        )
        assert any(e.entity_type == "MRN" for e in result.entities)

        # Day 9: exactly one LLM call for the whole document, not one per
        # candidate — the actual point of batching.
        assert llm.get_response.call_count == 1, f"expected exactly 1 batched LLM call, got {llm.get_response.call_count}"

        trail = agent.get_audit_trail()
        assert any(e.action == "entity_detected" for e in trail)
        assert any(e.action == "entity_rejected" for e in trail)
        assert any(e.action == "scan_completed" for e in trail)

        masked = mask_text(text, result.entities)
        assert "John Smith" not in masked
        assert "[PERSON]" in masked
        assert "Dr. Johnson" in masked, "Dr. Johnson was correctly excluded from entities, so it must stay unmasked"

        print(
            "phi_agent.py self-test passed: Presidio detection + batched LLM context filtering "
            "(1 call, not N) + audit logging verified without any network dependency."
        )

    asyncio.run(_run_self_test())
