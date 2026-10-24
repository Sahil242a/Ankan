"""SQLite persistence: rubrics, submissions, teacher decisions and an append-only audit log.

Every submission keeps a snapshot of the rubric it was graded with, so a later rubric edit
can never silently change the meaning of a past grade.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ankan.errors import ConflictError, NotFoundError
from ankan.schemas import (
    AuditEntry,
    GradeResult,
    OCRResult,
    Rubric,
    Submission,
    SubmissionStatus,
    TeacherDecision,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS rubrics (
    question_id TEXT PRIMARY KEY,
    version     INTEGER NOT NULL,
    body        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    id              TEXT PRIMARY KEY,
    question_id     TEXT NOT NULL REFERENCES rubrics(question_id),
    student_ref     TEXT,
    status          TEXT NOT NULL,
    rubric_version  INTEGER NOT NULL,
    rubric          TEXT NOT NULL,
    ocr             TEXT NOT NULL,
    answer_text     TEXT NOT NULL,
    proposed        TEXT NOT NULL,
    flag_count      INTEGER NOT NULL,
    final_score     REAL,
    reviewer        TEXT,
    comment         TEXT,
    has_image       INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    decided_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_submissions_queue
    ON submissions (status, flag_count DESC, created_at);
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    action        TEXT NOT NULL,
    actor         TEXT NOT NULL,
    at            TEXT NOT NULL,
    details       TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    def __init__(self, db_path: Path, image_dir: Path) -> None:
        self.db_path = db_path
        self.image_dir = image_dir
        db_path.parent.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation: safe across server worker threads."""
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------ rubrics

    def save_rubric(self, rubric: Rubric) -> int:
        """Insert or update a rubric atomically. Returns its new version number."""
        with self._connect() as db:
            row = db.execute(
                "INSERT INTO rubrics (question_id, version, body, updated_at) VALUES (?, 1, ?, ?) "
                "ON CONFLICT(question_id) DO UPDATE SET version = rubrics.version + 1, "
                "body = excluded.body, updated_at = excluded.updated_at RETURNING version",
                (rubric.question_id, rubric.model_dump_json(), _now()),
            ).fetchone()
        return int(row["version"])

    def get_rubric(self, question_id: str) -> tuple[Rubric, int]:
        with self._connect() as db:
            row = db.execute(
                "SELECT body, version FROM rubrics WHERE question_id = ?", (question_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"no rubric for question '{question_id}'")
        return Rubric.model_validate_json(row["body"]), int(row["version"])

    def delete_rubric(self, question_id: str) -> int:
        """Delete a question with all its answers, their history and page images.

        Returns how many answers were removed.
        """
        with self._connect() as db:
            exists = db.execute(
                "SELECT 1 FROM rubrics WHERE question_id = ?", (question_id,)
            ).fetchone()
            if exists is None:
                raise NotFoundError(f"no question '{question_id}'")
            ids = [
                row["id"]
                for row in db.execute(
                    "SELECT id FROM submissions WHERE question_id = ?", (question_id,)
                )
            ]
            db.execute(
                "DELETE FROM audit_log WHERE submission_id IN "
                "(SELECT id FROM submissions WHERE question_id = ?)",
                (question_id,),
            )
            db.execute("DELETE FROM submissions WHERE question_id = ?", (question_id,))
            db.execute("DELETE FROM rubrics WHERE question_id = ?", (question_id,))
        for submission_id in ids:  # only after the rows are gone for good
            self._image_path(submission_id).unlink(missing_ok=True)
        return len(ids)

    def list_rubrics(self) -> list[Rubric]:
        with self._connect() as db:
            rows = db.execute("SELECT body FROM rubrics ORDER BY question_id").fetchall()
        return [Rubric.model_validate_json(row["body"]) for row in rows]

    # ------------------------------------------------------------------ submissions

    def create_submission(
        self,
        *,
        rubric: Rubric,
        rubric_version: int,
        student_ref: str | None,
        ocr: OCRResult,
        answer_text: str,
        proposed: GradeResult,
        page_png: bytes | None = None,
    ) -> Submission:
        submission_id = uuid.uuid4().hex
        image_path = self._image_path(submission_id)
        if page_png is not None:
            image_path.write_bytes(page_png)
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO submissions (id, question_id, student_ref, status, rubric_version,"
                    " rubric, ocr, answer_text, proposed, flag_count, has_image, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        submission_id,
                        rubric.question_id,
                        student_ref,
                        SubmissionStatus.PENDING.value,
                        rubric_version,
                        rubric.model_dump_json(),
                        ocr.model_dump_json(),
                        answer_text,
                        proposed.model_dump_json(),
                        len(proposed.flags),
                        int(page_png is not None),
                        _now(),
                    ),
                )
                self._audit(
                    db,
                    submission_id,
                    "graded",
                    "system",
                    {
                        "grader": proposed.grader,
                        "score": proposed.score,
                        "rubric_version": rubric_version,
                    },
                )
        except BaseException:
            image_path.unlink(missing_ok=True)
            raise
        return self.get_submission(submission_id)

    def get_submission(self, submission_id: str) -> Submission:
        with self._connect() as db:
            row = db.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"no submission '{submission_id}'")
            audit = db.execute(
                "SELECT action, actor, at, details FROM audit_log WHERE submission_id = ? "
                "ORDER BY id",
                (submission_id,),
            ).fetchall()
        return self._to_submission(row, audit)

    def list_submissions(
        self, *, status: SubmissionStatus | None = None, question_id: str | None = None
    ) -> list[Submission]:
        """Review queue order: pending first, most flags first, then oldest first."""
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if question_id is not None:
            clauses.append("question_id = ?")
            params.append(question_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            # `where` is built only from the fixed strings above; values are bound parameters.
            f"SELECT * FROM submissions {where} "
            "ORDER BY status != 'pending', flag_count DESC, created_at"
        )
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [self._to_submission(row, []) for row in rows]

    def regrade(
        self,
        submission_id: str,
        *,
        answer_text: str,
        proposed: GradeResult,
        rubric: Rubric,
        rubric_version: int,
        actor: str,
    ) -> Submission:
        """Replace the proposal (e.g. after the teacher corrected the OCR text)."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE submissions SET answer_text = ?, proposed = ?, flag_count = ?, rubric = ?,"
                " rubric_version = ? WHERE id = ? AND status = ?",
                (
                    answer_text,
                    proposed.model_dump_json(),
                    len(proposed.flags),
                    rubric.model_dump_json(),
                    rubric_version,
                    submission_id,
                    SubmissionStatus.PENDING.value,
                ),
            )
            if cursor.rowcount == 0:
                self._raise_missing_or_decided(db, submission_id)
            self._audit(
                db,
                submission_id,
                "regraded",
                actor,
                {
                    "grader": proposed.grader,
                    "score": proposed.score,
                    "rubric_version": rubric_version,
                },
            )
        return self.get_submission(submission_id)

    def decide(self, submission_id: str, decision: TeacherDecision) -> Submission:
        """Record the teacher's final decision. Each submission is decided exactly once."""
        current = self.get_submission(submission_id)
        if current.status is not SubmissionStatus.PENDING:
            raise ConflictError(f"submission '{submission_id}' was already {current.status.value}")
        if decision.action == "approve":
            status, score = SubmissionStatus.APPROVED, current.proposed.score
        else:
            if decision.final_score is None:  # also enforced by TeacherDecision itself
                raise ValueError("an override needs a final_score")
            if decision.final_score > current.proposed.max_marks:
                raise ValueError(
                    f"final_score {decision.final_score:g} exceeds max marks "
                    f"{current.proposed.max_marks:g}"
                )
            status, score = SubmissionStatus.OVERRIDDEN, decision.final_score
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE submissions SET status = ?, final_score = ?, reviewer = ?, comment = ?,"
                " decided_at = ? WHERE id = ? AND status = ?",
                (
                    status.value,
                    score,
                    decision.reviewer,
                    decision.comment or None,
                    _now(),
                    submission_id,
                    SubmissionStatus.PENDING.value,
                ),
            )
            if cursor.rowcount == 0:  # someone else decided it between our read and write
                self._raise_missing_or_decided(db, submission_id)
            self._audit(
                db,
                submission_id,
                status.value,
                decision.reviewer,
                {
                    "proposed_score": current.proposed.score,
                    "final_score": score,
                    "comment": decision.comment,
                },
            )
        return self.get_submission(submission_id)

    def image_path(self, submission_id: str) -> Path:
        submission = self.get_submission(submission_id)
        path = self._image_path(submission_id)
        if not submission.has_image or not path.is_file():
            raise NotFoundError(f"submission '{submission_id}' has no page image")
        return path

    # ------------------------------------------------------------------ helpers

    def _image_path(self, submission_id: str) -> Path:
        if not submission_id.isalnum():  # ids are uuid hex; never build paths from other input
            raise NotFoundError(f"no submission '{submission_id}'")
        return self.image_dir / f"{submission_id}.png"

    @staticmethod
    def _raise_missing_or_decided(db: sqlite3.Connection, submission_id: str) -> None:
        row = db.execute("SELECT status FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"no submission '{submission_id}'")
        raise ConflictError(f"submission '{submission_id}' was already {row['status']}")

    @staticmethod
    def _audit(
        db: sqlite3.Connection, submission_id: str, action: str, actor: str, details: dict[str, Any]
    ) -> None:
        db.execute(
            "INSERT INTO audit_log (submission_id, action, actor, at, details)"
            " VALUES (?, ?, ?, ?, ?)",
            (submission_id, action, actor, _now(), json.dumps(details)),
        )

    @staticmethod
    def _to_submission(row: sqlite3.Row, audit: list[sqlite3.Row]) -> Submission:
        return Submission(
            id=row["id"],
            question_id=row["question_id"],
            student_ref=row["student_ref"],
            status=SubmissionStatus(row["status"]),
            ocr=OCRResult.model_validate_json(row["ocr"]),
            answer_text=row["answer_text"],
            proposed=GradeResult.model_validate_json(row["proposed"]),
            final_score=row["final_score"],
            reviewer=row["reviewer"],
            comment=row["comment"],
            has_image=bool(row["has_image"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
            audit=[
                AuditEntry(
                    action=entry["action"],
                    actor=entry["actor"],
                    at=datetime.fromisoformat(entry["at"]),
                    details=json.loads(entry["details"]),
                )
                for entry in audit
            ],
        )
