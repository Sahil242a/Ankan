from __future__ import annotations

import csv
import io

from fastapi.testclient import TestClient

from ankan.api import create_app
from ankan.config import Settings
from ankan.errors import BackendUnavailableError
from ankan.grading import KeyPointGrader, LexicalMatcher
from ankan.ocr.preprocess import encode_png
from ankan.pipeline import GradingPipeline
from ankan.schemas import OCRResult
from ankan.store import ReviewStore
from ankan.synthetic import render_sheet

from .conftest import FULL_ANSWER


def _put_rubric(client: TestClient, rubric_data: dict[str, object]) -> None:
    response = client.put(f"/api/rubrics/{rubric_data['question_id']}", json=rubric_data)
    assert response.status_code == 200, response.text


def _png() -> bytes:
    return encode_png(render_sheet(["Photosynthesis happens in chloroplasts."], angle=2))


def test_index_and_health(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Ankan" in page.text
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["ocr_engine"] == "fake"
    assert health["low_ocr_confidence"] == 0.6
    assert health["max_upload_mb"] == 1


def test_rubric_crud(client: TestClient, rubric_data: dict[str, object]) -> None:
    _put_rubric(client, rubric_data)
    assert client.get("/api/rubrics/bio-q1").json()["version"] == 1
    _put_rubric(client, rubric_data)
    assert client.get("/api/rubrics/bio-q1").json()["version"] == 2
    assert [r["question_id"] for r in client.get("/api/rubrics").json()] == ["bio-q1"]
    assert client.get("/api/rubrics/nope").status_code == 404

    mismatch = client.put("/api/rubrics/other", json=rubric_data)
    assert mismatch.status_code == 422
    bad = dict(rubric_data, max_marks=9)
    assert client.put("/api/rubrics/bio-q1", json=bad).status_code == 422


def test_full_review_flow_with_image(client: TestClient, rubric_data: dict[str, object]) -> None:
    _put_rubric(client, rubric_data)
    created = client.post(
        "/api/submissions",
        data={"question_id": "bio-q1", "student_ref": "=HYPERLINK(evil)"},
        files={"file": ("sheet.png", _png(), "image/png")},
    )
    assert created.status_code == 201, created.text
    sub = created.json()
    assert sub["status"] == "pending"
    assert sub["has_image"] is True
    assert sub["proposed"]["score"] == 5
    assert abs(sub["ocr"]["skew_angle"] + 2) < 0.3

    page = client.get(f"/api/submissions/{sub['id']}/page.png")
    assert page.status_code == 200
    assert page.content[:4] == b"\x89PNG"

    queue = client.get("/api/submissions", params={"status": "pending"}).json()
    assert [s["id"] for s in queue] == [sub["id"]]

    decided = client.post(
        f"/api/submissions/{sub['id']}/decision",
        json={"action": "override", "reviewer": "Ms Rao", "final_score": 4.5, "comment": "messy"},
    )
    assert decided.status_code == 200, decided.text
    assert decided.json()["final_score"] == 4.5
    again = client.post(
        f"/api/submissions/{sub['id']}/decision", json={"action": "approve", "reviewer": "x"}
    )
    assert again.status_code == 409
    assert client.get("/api/submissions", params={"status": "pending"}).json() == []

    exported = client.get("/api/export.csv")
    assert exported.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(exported.text)))
    assert len(rows) == 1
    assert rows[0]["final_score"] == "4.5"
    assert rows[0]["student_ref"] == "'=HYPERLINK(evil)"  # formula injection neutralised


def test_typed_answer_and_regrade(client: TestClient, rubric_data: dict[str, object]) -> None:
    _put_rubric(client, rubric_data)
    sub = client.post(
        "/api/submissions", data={"question_id": "bio-q1", "text": "Plants need water."}
    ).json()
    assert sub["has_image"] is False
    assert sub["proposed"]["score"] < 5

    regraded = client.post(
        f"/api/submissions/{sub['id']}/regrade", json={"text": FULL_ANSWER, "actor": "Ms Rao"}
    )
    assert regraded.status_code == 200
    body = regraded.json()
    assert body["proposed"]["score"] == 5
    assert [a["action"] for a in body["audit"]] == ["graded", "regraded"]
    assert client.get(f"/api/submissions/{sub['id']}/page.png").status_code == 404


def test_submission_validation(client: TestClient, rubric_data: dict[str, object]) -> None:
    _put_rubric(client, rubric_data)
    neither = client.post("/api/submissions", data={"question_id": "bio-q1"})
    assert neither.status_code == 422
    both = client.post(
        "/api/submissions",
        data={"question_id": "bio-q1", "text": "typed"},
        files={"file": ("sheet.png", _png(), "image/png")},
    )
    assert both.status_code == 422
    unknown = client.post("/api/submissions", data={"question_id": "nope", "text": "x"})
    assert unknown.status_code == 404
    garbage = client.post(
        "/api/submissions",
        data={"question_id": "bio-q1"},
        files={"file": ("sheet.png", b"not an image", "image/png")},
    )
    assert garbage.status_code == 400
    huge = client.post(
        "/api/submissions",
        data={"question_id": "bio-q1"},
        files={"file": ("sheet.png", b"0" * (1024 * 1024 + 1), "image/png")},
    )
    assert huge.status_code == 413
    bad_status = client.get("/api/submissions", params={"status": "finished"})
    assert bad_status.status_code == 422


def test_decision_validation(client: TestClient, rubric_data: dict[str, object]) -> None:
    _put_rubric(client, rubric_data)
    sub = client.post("/api/submissions", data={"question_id": "bio-q1", "text": "x"}).json()
    url = f"/api/submissions/{sub['id']}/decision"
    no_reason = {"action": "override", "reviewer": "t", "final_score": 1}
    assert client.post(url, json=no_reason).status_code == 422
    too_high = {"action": "override", "reviewer": "t", "final_score": 50, "comment": "x"}
    response = client.post(url, json=too_high)
    assert response.status_code == 422
    assert "exceeds max marks" in response.json()["detail"]
    assert (
        client.post(
            "/api/submissions/" + "0" * 32 + "/decision",
            json={"action": "approve", "reviewer": "t"},
        ).status_code
        == 404
    )


def test_photo_without_ocr_packages_explains_the_fix(
    settings: Settings, store: ReviewStore, rubric_data: dict[str, object]
) -> None:
    class MissingOCR:
        name = "trocr"

        def read(self, page: object) -> OCRResult:
            raise BackendUnavailableError("trocr", "trocr")

    app = create_app(
        settings,
        pipeline=GradingPipeline(MissingOCR(), KeyPointGrader(LexicalMatcher())),
        store=store,
    )
    with TestClient(app) as client:
        _put_rubric(client, rubric_data)
        response = client.post(
            "/api/submissions",
            data={"question_id": "bio-q1"},
            files={"file": ("sheet.png", _png(), "image/png")},
        )
        assert response.status_code == 503
        assert 'pip install -e ".[trocr]"' in response.json()["detail"]
        assert client.get("/api/submissions").json() == []  # nothing half-saved
        typed = client.post("/api/submissions", data={"question_id": "bio-q1", "text": "x"})
        assert typed.status_code == 201  # typed answers still work


def test_delete_question_and_optional_model_answer(
    client: TestClient, rubric_data: dict[str, object]
) -> None:
    rubric_data = {k: v for k, v in rubric_data.items() if k != "model_answer"}
    _put_rubric(client, rubric_data)  # a question needs no model answer
    client.post("/api/submissions", data={"question_id": "bio-q1", "text": FULL_ANSWER})
    response = client.delete("/api/rubrics/bio-q1")
    assert response.status_code == 200
    assert response.json() == {"deleted_answers": 1}
    assert client.get("/api/rubrics").json() == []
    assert client.delete("/api/rubrics/bio-q1").status_code == 404


def test_password_protects_every_page(
    settings: Settings, pipeline: GradingPipeline, store: ReviewStore
) -> None:
    import base64
    import dataclasses

    locked = dataclasses.replace(settings, password="s3cret pass")
    app = create_app(locked, pipeline=pipeline, store=store)

    def basic(user: str, password: str) -> dict[str, str]:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    with TestClient(app) as client:
        for path in ("/", "/api/health", "/api/rubrics", "/api/export.csv", "/docs"):
            response = client.get(path)
            assert response.status_code == 401, path
            assert response.headers["www-authenticate"].startswith("Basic")
        assert client.get("/", headers=basic("teacher", "wrong")).status_code == 401
        assert client.get("/", headers={"Authorization": "Basic !!notbase64"}).status_code == 401
        assert client.get("/", headers=basic("anyone", "s3cret pass")).status_code == 200
        assert client.get("/api/health", headers=basic("", "s3cret pass")).status_code == 200


def test_password_is_not_shown_in_settings_repr() -> None:
    assert "hunter2" not in repr(Settings(password="hunter2"))


def test_models_load_in_the_background_at_startup(settings: Settings, store: ReviewStore) -> None:
    import threading

    loaded = threading.Event()

    class WarmableOCR:
        name = "warm"

        def warm_up(self) -> None:
            loaded.set()

        def read(self, page: object) -> OCRResult:
            return OCRResult(engine=self.name)

    pipeline = GradingPipeline(WarmableOCR(), KeyPointGrader(LexicalMatcher()))
    with TestClient(create_app(settings, pipeline=pipeline, store=store)) as client:
        assert loaded.wait(5)
        assert client.get("/api/health").status_code == 200

    loaded.clear()
    quiet = create_app(settings, pipeline=pipeline, store=store, warm_up=False)
    with TestClient(quiet):
        assert not loaded.wait(0.2)


def test_failed_warm_up_does_not_stop_the_server(settings: Settings, store: ReviewStore) -> None:
    class BrokenOCR:
        name = "broken"

        def warm_up(self) -> None:
            raise BackendUnavailableError("trocr", "trocr")

        def read(self, page: object) -> OCRResult:
            raise BackendUnavailableError("trocr", "trocr")

    pipeline = GradingPipeline(BrokenOCR(), KeyPointGrader(LexicalMatcher()))
    with TestClient(create_app(settings, pipeline=pipeline, store=store)) as client:
        assert client.get("/api/health").status_code == 200
