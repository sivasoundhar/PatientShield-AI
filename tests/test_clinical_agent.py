"""Clinical Analyzer Agent test suite — 15 cases per CLAUDE.md section 16 Day 5.

No mocking (CLAUDE.md section 12: "Tests should use real LLM calls"). Every
test here drives the real LLMManager fallback chain — Groq if GROQ_API_KEY
is set, else local Ollama. See tests/test_phi_agent.py's module docstring for
why this matters for aggregate-metric tests against a small local model; the
gold set below is kept similarly short and unambiguous for the same reason.

One shared ClinicalAgent (module-scoped) — no expensive per-document setup to
avoid re-doing here (unlike PHIAgent's spaCy load), but matching the fixture
pattern keeps the two test files easy to read side by side.
"""

import pytest

from src.agents.clinical_agent import ClinicalAgent
from src.config import settings
from src.models import FindingCategory


@pytest.fixture(scope="module")
def clinical_agent() -> ClinicalAgent:
    return ClinicalAgent()


def _values_by_category(findings, category: FindingCategory) -> list[str]:
    return [f.value for f in findings if f.category == category]


# --- Category extraction (4 cases) ---


async def test_extracts_diagnosis(clinical_agent: ClinicalAgent):
    result = await clinical_agent.process("Diagnosis: Community-acquired pneumonia, right lower lobe.")
    assert any("pneumonia" in v.lower() for v in _values_by_category(result.findings, FindingCategory.DIAGNOSIS))


async def test_extracts_medication_with_dosage(clinical_agent: ClinicalAgent):
    # Deliberately emergent (not a routine chronic med): a routine antibiotic
    # scores below CLINICAL_PRIORITY_THRESHOLD and gets filtered by design
    # (see process()'s threshold gate) — this would test the filter, not
    # extraction. Same lesson as test_phi_agent.py's gold-set design: keep
    # test documents unambiguous for the thing actually under test.
    result = await clinical_agent.process("Started emergent IV vancomycin 1g every 8 hours for MRSA bacteremia with septic shock.")
    meds = _values_by_category(result.findings, FindingCategory.MEDICATION)
    assert any("vancomycin" in v.lower() for v in meds)
    assert any("1g" in v.lower() or "1 g" in v.lower() for v in meds), f"expected dosage captured in medication value, got {meds}"


async def test_extracts_finding_vital_sign(clinical_agent: ClinicalAgent):
    result = await clinical_agent.process("Blood pressure 210/120 on arrival, hypertensive urgency noted by attending.")
    assert any("210" in v or "hypertensive" in v.lower() for v in _values_by_category(result.findings, FindingCategory.FINDING))


async def test_extracts_follow_up_with_timing(clinical_agent: ClinicalAgent):
    # Same reasoning as test_extracts_medication_with_dosage: a routine
    # "2 weeks" follow-up scores below threshold and gets filtered by design.
    # An urgent, time-boxed referral clears the bar reliably instead.
    result = await clinical_agent.process("Urgent oncology referral within 48 hours for suspected malignant mass on imaging.")
    follow_ups = _values_by_category(result.findings, FindingCategory.FOLLOW_UP)
    assert any("oncology" in v.lower() or "48 hours" in v.lower() for v in follow_ups)


# --- Negation handling (2 cases) — the agent's core clinical-reasoning value proposition ---


async def test_negation_denies_excluded(clinical_agent: ClinicalAgent):
    """'Patient denies chest pain' describes an absence — must not become a finding."""
    result = await clinical_agent.process("Patient denies chest pain, shortness of breath, or palpitations.")
    assert not any("chest pain" in f.value.lower() for f in result.findings)


async def test_negation_ruled_out_excluded(clinical_agent: ClinicalAgent):
    """'Ruled out' explicitly negates a diagnosis — must not become a DIAGNOSIS finding."""
    result = await clinical_agent.process("Sepsis was ruled out after negative blood cultures and stable vitals.")
    assert not any(f.category == FindingCategory.DIAGNOSIS and "sepsis" in f.value.lower() for f in result.findings)


# --- Priority scoring (1 case) ---


async def test_critical_finding_higher_priority_than_minor(clinical_agent: ClinicalAgent):
    """A life-threatening diagnosis must score higher priority than a stable, routine one."""
    critical = await clinical_agent.process("Diagnosis: Septic shock, patient in ICU on vasopressors.")
    routine = await clinical_agent.process("Diagnosis: Well-controlled seasonal allergic rhinitis, no intervention needed.")

    critical_max = max((f.priority_score for f in critical.findings), default=0.0)
    routine_max = max((f.priority_score for f in routine.findings), default=0.0)
    assert critical_max > routine_max, f"critical={critical_max} should exceed routine={routine_max}"


# --- Multiple findings / empty document (2 cases) ---


async def test_multiple_categories_in_one_document(clinical_agent: ClinicalAgent):
    # A high-acuity case throughout (DKA) so every category clears the
    # priority threshold, not just the diagnosis — a mixed routine/urgent
    # document would test the filter rather than multi-category extraction.
    doc = (
        "Diagnosis: Diabetic ketoacidosis, blood glucose 580 mg/dL. "
        "Started emergent IV regular insulin drip per DKA protocol. "
        "Anion gap 28, indicating severe metabolic acidosis. "
        "Patient transferred to ICU for continuous monitoring."
    )
    result = await clinical_agent.process(doc)
    categories_present = {f.category for f in result.findings}
    assert len(categories_present) >= 2, f"expected findings spanning 2+ categories, got {categories_present}"


async def test_empty_document_returns_no_findings(clinical_agent: ClinicalAgent):
    result = await clinical_agent.process("")
    assert result.findings == []
    assert result.summary == ""


# --- Structural / contract correctness (5 cases) ---


async def test_summary_is_nonempty_for_nonempty_document(clinical_agent: ClinicalAgent):
    result = await clinical_agent.process("Diagnosis: Type 2 diabetes mellitus. Continue metformin 500mg twice daily.")
    assert result.summary.strip(), "expected a non-empty clinical summary"


async def test_all_findings_meet_priority_threshold(clinical_agent: ClinicalAgent):
    result = await clinical_agent.process("Diagnosis: Acute myocardial infarction, STEMI, emergent cath lab activation.")
    assert result.findings, "test document should produce at least one finding"
    assert all(f.priority_score >= settings.CLINICAL_PRIORITY_THRESHOLD for f in result.findings)


async def test_all_findings_confidence_in_valid_range(clinical_agent: ClinicalAgent):
    result = await clinical_agent.process("Diagnosis: Acute appendicitis. Scheduled for emergent appendectomy.")
    assert result.findings, "test document should produce at least one finding"
    assert all(0.0 <= f.confidence <= 1.0 for f in result.findings)
    assert all(0.0 <= f.priority_score <= 1.0 for f in result.findings)


async def test_result_types_are_correct(clinical_agent: ClinicalAgent):
    from src.models import ClinicalAnalysisResult, ClinicalFinding

    result = await clinical_agent.process("Diagnosis: Type 2 diabetes mellitus. Continue metformin 500mg twice daily.")
    assert isinstance(result, ClinicalAnalysisResult)
    assert all(isinstance(f, ClinicalFinding) for f in result.findings)
    assert all(isinstance(f.category, FindingCategory) for f in result.findings)


async def test_audit_trail_records_extraction_completed(clinical_agent: ClinicalAgent):
    before = len(clinical_agent.get_audit_trail())
    await clinical_agent.process("Diagnosis: Hypertension, well-controlled on current regimen.")
    new_events = clinical_agent.get_audit_trail()[before:]
    actions = {e.action for e in new_events}
    assert "extraction_completed" in actions, "expected extraction_completed to be logged every run"


# --- Aggregate accuracy (1 case) ---

# Gold-labeled documents for the aggregate metric. Each `expected` entry is
# (category, substring expected somewhere in that category's extracted
# values) — substring matching rather than exact text, since a small local
# LLM (Ollama fallback) won't reliably reproduce the note's exact phrasing.
# Kept to clearly high-acuity or clearly negated documents, same reasoning as
# the individual extraction tests above: this metric is measuring extraction
# correctness, not re-measuring the priority threshold's own calibration.
_GOLD_SET: list[tuple[str, list[tuple[FindingCategory, str]]]] = [
    ("Diagnosis: Septic shock secondary to urinary tract infection, patient on vasopressors in ICU.", [(FindingCategory.DIAGNOSIS, "septic shock")]),
    ("Started emergent IV vancomycin 1g every 8 hours for MRSA bacteremia.", [(FindingCategory.MEDICATION, "vancomycin")]),
    ("Lab results show hemoglobin of 5.8, a critical value requiring emergent blood transfusion.", [(FindingCategory.FINDING, "5.8")]),
    ("Urgent oncology referral within 48 hours for suspected malignant mass on imaging.", [(FindingCategory.FOLLOW_UP, "oncology")]),
    ("Patient denies any history of seizures.", []),
    ("No evidence of pneumonia on chest X-ray.", []),
]


async def test_extraction_accuracy_meets_target(clinical_agent: ClinicalAgent):
    """Aggregate accuracy >= 90% across the gold set — the Day 5 finish line.

    Uses the real, currently-configured LLM (Groq if GROQ_API_KEY is set,
    else Ollama). "Accuracy" here is the fraction of gold-set expectations
    (both "this finding must appear" and "this document must produce no
    matching finding") that the agent satisfies — the same spirit as
    test_phi_agent.py's precision/recall test, adapted for clinical extraction
    where a document can also validly expect zero findings.
    """
    correct = 0
    total = 0
    for text, expected in _GOLD_SET:
        result = await clinical_agent.process(text)
        found_values = [f.value.lower() for f in result.findings]

        if not expected:
            total += 1
            if not found_values:
                correct += 1
            continue

        for category, substring in expected:
            total += 1
            category_values = _values_by_category(result.findings, category)
            if any(substring.lower() in v.lower() for v in category_values):
                correct += 1

    accuracy = correct / total if total > 0 else 1.0
    print(f"\nClinical Agent aggregate accuracy: {accuracy:.2%} ({correct}/{total})")

    assert accuracy >= 0.90, f"accuracy {accuracy:.2%} below 90% target ({correct}/{total})"
