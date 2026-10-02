import pytest
from fastapi.testclient import TestClient

from app.llm import PipelineError
from app.main import app
from app.schemas import NoteClassification, NoteExtraction

SURGERY = ("PREOPERATIVE DIAGNOSIS: Right knee medial meniscus tear. PROCEDURE: Arthroscopic partial medial "
           "meniscectomy. The patient was brought to the operating room, general anesthesia was induced, and the "
           "knee was prepped and draped in the usual sterile fashion.")
RADIOLOGY = "CT ABDOMEN AND PELVIS WITH CONTRAST. FINDINGS: Liver and spleen unremarkable. IMPRESSION: No acute process."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        c.app.state.pipeline.llm = None  # tests never call the network
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["local_model_loaded"] and not body["llm_enabled"]


def test_classify_local(client):
    r = client.post("/classify", json={"note": SURGERY}).json()
    assert r["source"] == "local"
    assert "Surgery (Operative Notes)" in r["groups"]
    assert len(r["scores"]) == 11


def test_classify_radiology(client):
    assert client.post("/classify", json={"note": RADIOLOGY}).json()["groups"][0] == "Radiology"


def test_batch(client):
    r = client.post("/classify/batch", json={"notes": [SURGERY, RADIOLOGY]}).json()
    assert r["count"] == 2 and r["results"][1]["groups"][0] == "Radiology"


@pytest.mark.parametrize("bad", ["", "   ", "### 12345 !!!"])
def test_rejects_unreadable_input(client, bad):
    assert client.post("/classify", json={"note": bad}).status_code == 422


def test_batch_limit(client):
    assert client.post("/classify/batch", json={"notes": [SURGERY] * 51}).status_code == 413


def test_llm_endpoints_503_without_key(client):
    assert client.post("/extract", json={"note": SURGERY}).status_code == 503
    assert client.post("/classify?mode=llm", json={"note": SURGERY}).status_code == 503


def test_analyze_without_llm_degrades_gracefully(client):
    r = client.post("/analyze", json={"note": SURGERY}).json()
    assert r["extraction_used_fallback"] is True and r["extraction"]["patient_sex"] == "Unknown"


class FakeLLM:
    model = "fake"

    def __init__(self, fail=False):
        self.fail = fail

    async def classify(self, note):
        if self.fail:
            raise PipelineError("boom")
        return NoteClassification(groups=["radiology"], confidence=0.9)

    async def extract(self, note):
        if self.fail:
            raise PipelineError("boom")
        return NoteExtraction(patient_age=52, patient_sex="Male", note_type="Operative")


def test_llm_path_and_fallback(client):
    pipe = client.app.state.pipeline
    pipe.llm = FakeLLM()
    r = client.post("/classify?mode=llm", json={"note": SURGERY}).json()
    assert r["source"] == "llm" and r["groups"] == ["Radiology"]  # casing normalised by the schema
    assert client.post("/extract", json={"note": SURGERY}).json()["extraction"]["patient_age"] == 52
    pipe.llm = FakeLLM(fail=True)
    r = client.post("/classify?mode=llm", json={"note": SURGERY}).json()
    assert r["source"] == "fallback" and "Surgery (Operative Notes)" in r["groups"]
    assert client.post("/extract", json={"note": SURGERY}).json()["used_fallback"] is True
    pipe.llm = None


def test_schema_guardrails():
    with pytest.raises(Exception):
        NoteClassification(groups=["Veterinary"], confidence=1.0)
    with pytest.raises(Exception):
        NoteClassification(groups=["Radiology"], confidence=1.7)
