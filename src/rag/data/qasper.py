"""QASPER v0.3 papers, questions, answers, and gold evidence paragraphs.

Figure and table evidence (``FLOAT SELECTED``) is dropped. Questions with no
paragraph evidence left are skipped. ``link_qasper_evidence`` matches the
remaining paragraphs to chunk ids by exact string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from rag.data.papers import Paper

_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


@dataclass(frozen=True)
class QasperQuestion:
    question_id: str
    paper_id: str
    question: str
    answers: tuple[str, ...]
    evidence: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...] = ()


def load_qasper_papers(split: str, data_dir: Path) -> list[Paper]:
    papers = []
    for paper_id, raw in _load(split, data_dir).items():
        sections = _sections(raw)
        if sections:
            papers.append(Paper(paper_id, str(raw.get("title") or ""), tuple(sections)))
    return papers


def load_qasper_questions(split: str, data_dir: Path, *, require_evidence: bool = True) -> list[QasperQuestion]:
    questions = []
    skipped = 0
    for paper_id, raw in _load(split, data_dir).items():
        for record in raw.get("qas") or []:
            if not isinstance(record, dict):
                continue
            answers, evidence = _annotations(record.get("answers") or [])
            if require_evidence and not evidence:
                skipped += 1
                continue
            questions.append(
                QasperQuestion(
                    question_id=str(record.get("question_id") or ""),
                    paper_id=paper_id,
                    question=str(record.get("question") or "").strip(),
                    answers=tuple(answers),
                    evidence=tuple(evidence),
                )
            )
    if skipped:
        print(f"QASPER {split}: skipped {skipped} questions with no text evidence")
    return questions


def link_qasper_evidence(questions: list[QasperQuestion], chunks) -> list[QasperQuestion]:
    by_paper: dict[str, dict[str, list[str]]] = {}
    for chunk in chunks:
        by_paper.setdefault(chunk.doc_id, {}).setdefault(chunk.body, []).append(chunk.chunk_id)

    linked = []
    unmatched = 0
    for question in questions:
        chunk_ids = []
        for paragraph in question.evidence:
            found = by_paper.get(question.paper_id, {}).get(paragraph)
            if not found:
                unmatched += 1
                continue
            chunk_ids.extend(found)
        linked.append(
            replace(question, evidence_chunk_ids=tuple(dict.fromkeys(chunk_ids)))
            if chunk_ids
            else question
        )
    print(f"QASPER evidence: {unmatched} paragraph(s) did not match a chunk")
    return linked


def _load(split: str, data_dir: Path) -> dict[str, dict]:
    if split not in _FILES:
        raise ValueError(f"Unknown QASPER split {split!r}. Expected train, validation, or test.")
    path = data_dir / "qasper" / _FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_data.py --dataset qasper`."
        )
    with path.open() as handle:
        raw = json.load(handle)
    return {str(paper_id): paper for paper_id, paper in raw.items()}


def _sections(paper: dict) -> list[tuple[str, tuple[str, ...]]]:
    sections = []
    abstract = paper.get("abstract") or ""
    if isinstance(abstract, str) and abstract.strip():
        sections.append(("Abstract", (abstract.strip(),)))
    for section in paper.get("full_text") or []:
        if not isinstance(section, dict):
            continue
        paragraphs = tuple(
            paragraph.strip()
            for paragraph in section.get("paragraphs") or []
            if isinstance(paragraph, str) and paragraph.strip()
        )
        if paragraphs:
            sections.append((str(section.get("section_name") or ""), paragraphs))
    return sections


def _annotations(raw) -> tuple[list[str], list[str]]:
    answers = []
    evidence = []
    for annotation in raw:
        answer = annotation.get("answer") if isinstance(annotation, dict) else None
        if not isinstance(answer, dict):
            continue
        text = _answer_text(answer)
        if text:
            answers.append(text)
        for paragraph in answer.get("evidence") or []:
            if not isinstance(paragraph, str):
                continue
            paragraph = paragraph.strip()
            if not paragraph or paragraph.startswith("FLOAT SELECTED") or paragraph in evidence:
                continue
            evidence.append(paragraph)
    return answers, evidence


def _answer_text(answer: dict) -> str | None:
    if answer.get("unanswerable"):
        return "Unanswerable"
    spans = [span.strip() for span in answer.get("extractive_spans") or [] if isinstance(span, str) and span.strip()]
    if spans:
        return " ".join(spans)
    free_form = answer.get("free_form_answer") or ""
    if isinstance(free_form, str) and free_form.strip():
        return free_form.strip()
    if answer.get("yes_no") is True:
        return "Yes"
    if answer.get("yes_no") is False:
        return "No"
    return None
