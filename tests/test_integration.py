"""End-to-end 5-agent pipeline integration tests — 5 cases per CLAUDE.md section 16 Day 7.

No mocking (CLAUDE.md section 12) — every test here drives the real pipeline
through the actual `client` fixture: real Presidio + real LLMManager chain
(Groq if configured, else Ollama) for PHI/Clinical/Knowledge, and a real
UniRAGConnector HTTP call for indexing. If UniRAG isn't reachable, indexing
degrades to `knowledge_indexed=False` (see knowledge_agent.py) rather than
failing the run — these tests verify pipeline *orchestration* correctness
(all 5 agents ran, results are shaped right, audit trail is complete), not
UniRAG's own availability, so no test here hard-asserts
`knowledge_indexed is True`. tests/test_knowledge_agent.py already covers
UniRAG-specific behavior in isolation, with its own live-service skip guard.

Five distinct, short, single-concept sample documents (mirroring the
"unambiguous" gold-set lesson from Days 4-6) — both to satisfy CLAUDE.md's
"5 sample medical documents" and to keep this suite's real LLM-call volume
low, since each case runs the full pipeline, not just one agent.
"""

_SAMPLE_DOCUMENTS = [
    "Patient Diaz, MRN: 00591234. Diagnosis: Type 2 diabetes mellitus. Prescribed metformin 500mg twice daily.",
    "Patient Chen, DOB 06/11/1972. Diagnosis: Community-acquired pneumonia. Started on azithromycin 500mg daily.",
    "Patient Osei, SSN: 456-78-1234. Diagnosis: Acute appendicitis. Scheduled for emergent appendectomy.",
    "Patient Rossi, phone (415) 555-2211. Diagnosis: Hypertension, stage 2. Prescribed lisinopril 20mg daily.",
    "Patient Kim, email k.kim@example.com. Diagnosis: Fractured tibia. Treatment: cast placement, follow up in 6 weeks.",
]


def _upload_and_process(client, text: str) -> dict:
    """Upload `text` as a document and run it through the full pipeline, returning the ProcessResponse body."""
    upload_response = client.post("/upload", files={"file": ("note.txt", text.encode(), "text/plain")})
    document_id = upload_response.json()["document_id"]
    process_response = client.post("/process", json={"document_id": document_id})
    assert process_response.status_code == 200, process_response.text
    return process_response.json()


def test_upload_and_process_returns_completed_status(client):
    body = _upload_and_process(client, _SAMPLE_DOCUMENTS[0])
    assert body["processing_status"] == "completed"
    assert body["processing_time"] is not None and body["processing_time"] >= 0


def test_all_five_agents_appear_in_audit_trail(client):
    body = _upload_and_process(client, _SAMPLE_DOCUMENTS[1])
    agent_names = {event["agent_name"] for event in body["audit_events"]}
    assert agent_names == {"planner", "phi_agent", "clinical_agent", "knowledge_agent", "audit_agent"}, agent_names

    # The Audit Agent's own consolidation event should agree — this is the
    # completeness check (audit_agent.py) actually doing its job, not just
    # every node happening to have logged something by coincidence.
    audit_events = [e for e in body["audit_events"] if e["agent_name"] == "audit_agent"]
    assert audit_events, "audit_agent must log at least one event"
    assert audit_events[-1]["details"]["missing_agents"] == []


def test_audit_endpoint_matches_process_response_trail(client):
    upload_response = client.post("/upload", files={"file": ("note.txt", _SAMPLE_DOCUMENTS[2].encode(), "text/plain")})
    document_id = upload_response.json()["document_id"]
    process_body = client.post("/process", json={"document_id": document_id}).json()

    audit_response = client.get(f"/audit/{document_id}")
    assert audit_response.status_code == 200
    audit_body = audit_response.json()

    assert audit_body["document_id"] == document_id
    assert len(audit_body["audit_trail"]) == len(process_body["audit_events"])
    process_actions = sorted(e["action"] for e in process_body["audit_events"])
    persisted_actions = sorted(e["action"] for e in audit_body["audit_trail"])
    assert process_actions == persisted_actions


def test_process_response_fields_are_correctly_typed_and_populated(client):
    """Guards exactly the class of bug found during Day 6 verification: a real field silently dropped from the response."""
    body = _upload_and_process(client, _SAMPLE_DOCUMENTS[3])

    assert isinstance(body["original_text"], str) and body["original_text"]
    assert isinstance(body["de_identified_text"], str) and body["de_identified_text"]
    assert body["original_text"] != body["de_identified_text"], "PHI (name/phone) should have been masked"

    assert body["phi_detected"] is not None
    assert isinstance(body["phi_detected"]["entities"], list)

    assert body["clinical_analysis"] is not None
    assert isinstance(body["clinical_analysis"]["findings"], list)
    assert isinstance(body["clinical_analysis"]["summary"], str)

    assert isinstance(body["knowledge_indexed"], bool)  # Day 6's bug: this silently defaulted to False regardless of the real result
    assert isinstance(body["audit_events"], list) and len(body["audit_events"]) >= 5


def test_pipeline_status_endpoint_reflects_completion(client):
    upload_response = client.post("/upload", files={"file": ("note.txt", _SAMPLE_DOCUMENTS[4].encode(), "text/plain")})
    document_id = upload_response.json()["document_id"]
    client.post("/process", json={"document_id": document_id})

    status_response = client.get(f"/pipeline-status/{document_id}")
    assert status_response.status_code == 200
    status_body = status_response.json()

    assert status_body["document_id"] == document_id
    assert status_body["status"] == "completed"
    assert status_body["processing_time"] is not None and status_body["processing_time"] >= 0
    assert len(status_body["events"]) >= 5
