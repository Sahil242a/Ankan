"""Command line: grade one sheet, calibrate thresholds, render a demo sheet, run the server."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ankan.config import GRADERS, OCR_ENGINES, Settings
from ankan.errors import AnkanError
from ankan.schemas import Coverage, GradeResult, OCRResult, Rubric

_MARK = {Coverage.COVERED: "[x]", Coverage.PARTIAL: "[~]", Coverage.MISSED: "[ ]"}


def _settings(args: argparse.Namespace) -> Settings:
    """Environment settings, overridden by --grader / --ocr when given."""
    settings = Settings.from_env()
    if args.grader:
        settings = dataclasses.replace(settings, grader=args.grader)
    if args.ocr:
        settings = dataclasses.replace(settings, ocr_engine=args.ocr)
    if getattr(args, "fast", False):
        settings = dataclasses.replace(settings, fast_ocr=True)
    return settings


def _report(grade: GradeResult, ocr: OCRResult | None) -> str:
    lines = []
    if ocr is not None:
        confidence = ocr.mean_confidence
        shown = f"{confidence:.0%}" if confidence is not None else "n/a"
        lines += [f"OCR ({ocr.engine}, mean confidence {shown}, deskew {ocr.skew_angle:+.1f} deg):"]
        lines += [f"  {line.text}" for line in ocr.lines] or ["  (no text found)"]
        lines.append("")
    lines.append(
        f"Proposed score: {grade.score:g} / {grade.max_marks:g}   (grader: {grade.grader})"
    )
    for kp in grade.key_points:
        similarity = f"  sim={kp.similarity:.2f}" if kp.similarity is not None else ""
        lines.append(
            f"  {_MARK[kp.status]} {kp.key_point_id}: {kp.marks_awarded:g}/{kp.marks_possible:g}"
            f"{similarity}  {kp.description}"
        )
        if kp.evidence:
            lines.append(f'        evidence: "{kp.evidence}"')
        if kp.rationale:
            lines.append(f"        why: {kp.rationale}")
    if grade.flags:
        lines.append("Needs a closer look:")
        lines += [f"  ! {flag.code.value}: {flag.message}" for flag in grade.flags]
    lines.append("This is a proposal. A teacher makes the final decision.")
    return "\n".join(lines)


def _cmd_grade(args: argparse.Namespace) -> int:
    from ankan.pipeline import build_pipeline

    rubric = Rubric.model_validate_json(Path(args.rubric).read_text(encoding="utf-8"))
    pipeline = build_pipeline(_settings(args))
    ocr: OCRResult | None = None
    if args.image:
        result = pipeline.grade_image(Path(args.image).read_bytes(), rubric)
        grade, ocr = result.grade, result.ocr
    else:
        grade = pipeline.grade_text(Path(args.text_file).read_text(encoding="utf-8"), rubric)
    if args.json:
        payload = {"grade": grade.model_dump(mode="json")}
        if ocr is not None:
            payload["ocr"] = ocr.model_dump(mode="json")
        print(json.dumps(payload, indent=2))
    else:
        print(_report(grade, ocr))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from ankan.grading.calibration import calibrate, load_examples
    from ankan.grading.matchers import EmbeddingMatcher, LexicalMatcher, Matcher

    settings = Settings.from_env()
    matcher: Matcher = (
        LexicalMatcher()
        if args.matcher == "lexical"
        else EmbeddingMatcher(settings.embedding_model, device=settings.device)
    )
    report = calibrate(matcher, load_examples(Path(args.examples)))
    print(
        f"{report.examples} examples, accuracy {report.accuracy:.1%} with "
        f"partial >= {report.thresholds.partial:.2f}, covered >= {report.thresholds.covered:.2f}"
    )
    print("confusion (rows = teacher, columns = predicted):")
    labels = list(report.confusion)
    print("            " + "".join(f"{label:>10}" for label in labels))
    for truth in labels:
        print(f"{truth:>12}" + "".join(f"{report.confusion[truth][p]:>10}" for p in labels))
    return 0


DEMO_ANSWER = [
    "The Earth is tilted on its axis.",
    "As it orbits the Sun, the half tilted",
    "towards the Sun gets more direct",
    "sunlight, so it has summer.",
]


def _cmd_demo_sheet(args: argparse.Namespace) -> int:
    from ankan.ocr.preprocess import encode_png
    from ankan.synthetic import render_sheet

    image = render_sheet(DEMO_ANSWER, angle=args.angle, noise=5)
    Path(args.output).write_bytes(encode_png(image))
    print(f"wrote {args.output}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ankan.api import create_app

    settings = _settings(args)
    if args.host not in ("127.0.0.1", "localhost") and not settings.password:
        print(
            "Warning: Ankan is reachable from other computers without a password. "
            "Set ANKAN_PASSWORD before sharing it.",
            file=sys.stderr,
        )
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ankan", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    grade = commands.add_parser("grade", help="grade one answer and print the proposal")
    grade.add_argument("--rubric", required=True, help="rubric JSON file")
    source = grade.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="photo or scan of the answer")
    source.add_argument("--text-file", help="plain-text answer (skips OCR)")
    grade.add_argument("--grader", choices=GRADERS)
    grade.add_argument("--ocr", choices=OCR_ENGINES)
    grade.add_argument("--json", action="store_true", help="print machine-readable JSON")
    grade.add_argument("--fast", action="store_true", help="faster handwriting reading (8-bit)")
    grade.set_defaults(handler=_cmd_grade)

    calibrate = commands.add_parser("calibrate", help="fit thresholds to teacher labels")
    calibrate.add_argument("examples", help="JSON lines: key_point, answer, label")
    calibrate.add_argument("--matcher", choices=("lexical", "embedding"), default="embedding")
    calibrate.set_defaults(handler=_cmd_calibrate)

    demo = commands.add_parser("demo-sheet", help="render a synthetic answer sheet image")
    demo.add_argument("output", help="where to write the PNG")
    demo.add_argument("--angle", type=float, default=2.5, help="page rotation in degrees")
    demo.set_defaults(handler=_cmd_demo_sheet)

    serve = commands.add_parser("serve", help="run the API and the teacher review UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--grader", choices=GRADERS)
    serve.add_argument("--ocr", choices=OCR_ENGINES)
    serve.add_argument("--fast", action="store_true", help="faster handwriting reading (8-bit)")
    serve.set_defaults(handler=_cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.handler(args)
    except (AnkanError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
