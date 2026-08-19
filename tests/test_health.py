"""Day 1 finish line: health, upload, and get_document all work end-to-end."""


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"]
    assert "timestamp" in body


def test_upload(client):
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"Patient John Doe reports headache.", "text/plain")},
        data={"patient_id": "PT-001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.txt"
    assert body["status"] == "uploaded"
    assert body["document_id"]


def test_get_document(client):
    upload_response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"Patient John Doe reports headache.", "text/plain")},
    )
    document_id = upload_response.json()["document_id"]

    response = client.get(f"/documents/{document_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == document_id
    assert body["filename"] == "sample.txt"
    assert body["processing_status"] == "uploaded"


def test_get_document_not_found(client):
    response = client.get("/documents/does-not-exist")
    assert response.status_code == 404
