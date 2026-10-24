"""HTTP API and teacher review UI.

Every grade starts as a *proposal* in the review queue. Nothing becomes a final mark until
a named teacher approves or overrides it, and every step is written to the audit log.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import logging
import secrets
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ankan import __version__
from ankan.config import Settings
from ankan.errors import (
    BackendUnavailableError,
    ConflictError,
    GradingError,
    InvalidImageError,
    NotFoundError,
    OCRError,
)
from ankan.ocr.preprocess import encode_png
from ankan.pipeline import GradingPipeline, build_pipeline
from ankan.schemas import (
    OCRResult,
    Rubric,
    Submission,
    SubmissionStatus,
    TeacherDecision,
)
from ankan.store import ReviewStore

MAX_TEXT_CHARS = 20_000
log = logging.getLogger("uvicorn.error")  # shows in the same terminal as the server
# Plain codes: Starlette renamed these constants, and the numbers work on every version.
HTTP_UNPROCESSABLE = 422
HTTP_TOO_LARGE = 413


class RubricEnvelope(BaseModel):
    rubric: Rubric
    version: int


class RegradeRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    actor: str = Field(min_length=1, max_length=100)


def _csv_cell(value: object) -> str:
    """Neutralise spreadsheet formula injection in exported cells."""
    text = "" if value is None else str(value)
    return f"'{text}" if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: GradingPipeline | None = None,
    store: ReviewStore | None = None,
    warm_up: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or ReviewStore(settings.db_path, settings.image_dir)
    pipeline = pipeline or build_pipeline(settings)
    max_upload_bytes = int(settings.max_upload_mb * 1024 * 1024)

    def load_models() -> None:
        log.info("Loading models in the background so the first answer is quick...")
        try:
            pipeline.warm_up()
        except Exception as exc:  # reported again, clearly, when an answer needs the model
            log.warning("Could not load models in advance: %s", exc)
        else:
            log.info("Models ready.")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if warm_up:
            threading.Thread(target=load_models, name="ankan-warm-up", daemon=True).start()
        yield

    app = FastAPI(title="Ankan", version=__version__, lifespan=lifespan)

    # ------------------------------------------------------------------ password

    if settings.password:
        expected = settings.password.encode("utf-8")

        @app.middleware("http")
        async def require_password(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            """HTTP Basic auth: the browser shows its own password prompt (any username)."""
            scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() == "basic":
                try:
                    _, _, given = base64.b64decode(encoded, validate=True).partition(b":")
                except (binascii.Error, ValueError):
                    given = b""
                if secrets.compare_digest(given, expected):
                    return await call_next(request)
            return Response(
                "Password required",
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": 'Basic realm="Ankan", charset="UTF-8"'},
            )

    # ------------------------------------------------------------------ errors

    def _handler(code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
        async def handle(_: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=code, content={"detail": str(exc)})

        return handle

    app.add_exception_handler(NotFoundError, _handler(status.HTTP_404_NOT_FOUND))
    app.add_exception_handler(ConflictError, _handler(status.HTTP_409_CONFLICT))
    app.add_exception_handler(InvalidImageError, _handler(status.HTTP_400_BAD_REQUEST))
    app.add_exception_handler(
        BackendUnavailableError, _handler(status.HTTP_503_SERVICE_UNAVAILABLE)
    )
    app.add_exception_handler(OCRError, _handler(status.HTTP_502_BAD_GATEWAY))
    app.add_exception_handler(GradingError, _handler(status.HTTP_502_BAD_GATEWAY))

    # ------------------------------------------------------------------ meta

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return resources.files("ankan").joinpath("web/index.html").read_text("utf-8")

    @app.get("/api/health")
    def health() -> dict[str, str | float]:
        return {
            "status": "ok",
            "version": __version__,
            "ocr_engine": pipeline.ocr.name,
            "grader": pipeline.grader.name,
            "low_ocr_confidence": pipeline.low_ocr_confidence,
            "max_upload_mb": settings.max_upload_mb,
        }

    # ------------------------------------------------------------------ rubrics

    @app.get("/api/rubrics")
    def list_rubrics() -> list[Rubric]:
        return store.list_rubrics()

    @app.get("/api/rubrics/{question_id}")
    def get_rubric(question_id: str) -> RubricEnvelope:
        rubric, version = store.get_rubric(question_id)
        return RubricEnvelope(rubric=rubric, version=version)

    @app.put("/api/rubrics/{question_id}")
    def put_rubric(question_id: str, rubric: Rubric) -> RubricEnvelope:
        if rubric.question_id != question_id:
            raise HTTPException(
                HTTP_UNPROCESSABLE,
                "question_id in the body must match the URL",
            )
        return RubricEnvelope(rubric=rubric, version=store.save_rubric(rubric))

    @app.delete("/api/rubrics/{question_id}")
    def delete_rubric(question_id: str) -> dict[str, int]:
        """Delete a question together with all of its answers."""
        return {"deleted_answers": store.delete_rubric(question_id)}

    # ------------------------------------------------------------------ submissions

    @app.post("/api/submissions", status_code=status.HTTP_201_CREATED)
    def create_submission(
        question_id: Annotated[str, Form(min_length=1, max_length=64)],
        student_ref: Annotated[str | None, Form(max_length=64)] = None,
        text: Annotated[str | None, Form(max_length=MAX_TEXT_CHARS)] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ) -> Submission:
        """Grade one answer: upload a page image, or send already-typed text."""
        upload = file if file is not None and file.filename else None
        has_text = text is not None and (upload is None or text.strip() != "")
        if (upload is not None) == has_text:
            raise HTTPException(
                HTTP_UNPROCESSABLE,
                "send exactly one of 'file' (a page image) or 'text'",
            )
        rubric, version = store.get_rubric(question_id)
        ref = (student_ref or "").strip() or None

        if upload is not None:
            data = upload.file.read(max_upload_bytes + 1)
            if len(data) > max_upload_bytes:
                raise HTTPException(
                    HTTP_TOO_LARGE,
                    f"image is larger than {settings.max_upload_mb:g} MB",
                )
            graded = pipeline.grade_image(data, rubric)
            return store.create_submission(
                rubric=rubric,
                rubric_version=version,
                student_ref=ref,
                ocr=graded.ocr,
                answer_text=graded.ocr.text,
                proposed=graded.grade,
                page_png=encode_png(graded.page),
            )

        answer = text or ""
        return store.create_submission(
            rubric=rubric,
            rubric_version=version,
            student_ref=ref,
            ocr=OCRResult(engine="typed"),
            answer_text=answer,
            proposed=pipeline.grade_text(answer, rubric),
        )

    @app.get("/api/submissions")
    def list_submissions(
        status_: Annotated[SubmissionStatus | None, Query(alias="status")] = None,
        question_id: str | None = None,
    ) -> list[Submission]:
        """The review queue: pending first, most-flagged first, then oldest first."""
        return store.list_submissions(status=status_, question_id=question_id)

    @app.get("/api/submissions/{submission_id}")
    def get_submission(submission_id: str) -> Submission:
        return store.get_submission(submission_id)

    @app.get("/api/submissions/{submission_id}/page.png", include_in_schema=False)
    def get_page(submission_id: str) -> FileResponse:
        return FileResponse(store.image_path(submission_id), media_type="image/png")

    @app.post("/api/submissions/{submission_id}/regrade")
    def regrade(submission_id: str, body: RegradeRequest) -> Submission:
        """Re-grade with corrected text (e.g. after fixing an OCR mistake) and the latest rubric."""
        submission = store.get_submission(submission_id)
        if submission.status is not SubmissionStatus.PENDING:
            raise ConflictError(
                f"submission '{submission_id}' was already {submission.status.value}"
            )
        rubric, version = store.get_rubric(submission.question_id)
        proposed = pipeline.grade_text(body.text, rubric, ocr=None)
        return store.regrade(
            submission_id,
            answer_text=body.text,
            proposed=proposed,
            rubric=rubric,
            rubric_version=version,
            actor=body.actor,
        )

    @app.post("/api/submissions/{submission_id}/decision")
    def decide(submission_id: str, decision: TeacherDecision) -> Submission:
        try:
            return store.decide(submission_id, decision)
        except ValueError as exc:
            raise HTTPException(HTTP_UNPROCESSABLE, str(exc)) from exc

    @app.get("/api/export.csv")
    def export_csv(question_id: str | None = None) -> Response:
        """Final marks for decided submissions, ready for a spreadsheet."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "submission_id",
                "question_id",
                "student_ref",
                "status",
                "proposed_score",
                "final_score",
                "max_marks",
                "reviewer",
                "decided_at",
                "flags",
            ]
        )
        for sub in store.list_submissions(question_id=question_id):
            if sub.status is SubmissionStatus.PENDING:
                continue
            writer.writerow(
                [
                    _csv_cell(v)
                    for v in (
                        sub.id,
                        sub.question_id,
                        sub.student_ref,
                        sub.status.value,
                        sub.proposed.score,
                        sub.final_score,
                        sub.proposed.max_marks,
                        sub.reviewer,
                        sub.decided_at.isoformat() if sub.decided_at else "",
                        ";".join(sorted({f.code.value for f in sub.proposed.flags})),
                    )
                ]
            )
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ankan-marks.csv"'},
        )

    return app
