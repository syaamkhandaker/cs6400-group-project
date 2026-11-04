"""
Answer a question from retrieved chunks and cite the chunks it used.
Each passage in the prompt carries a short label ([P1], [P2], ...) that maps
back to its chunk id, so every citation in the answer resolves to a chunk with
its paper, section, and text. Short labels are easier for a small local model
to copy than raw ids. The model is asked for two lines, ``Answer:`` and ``Citations:``. Only the
answer line is scored; the citations are the provenance record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You answer questions about scientific papers using only the passages you are given. "
    "Never use outside knowledge."
)

# _ANSWER: text after "Answer:" up to the Citations line; _CITATIONS: text after
# "Citations:"; _LABEL: one label such as P3; _INLINE_CITE: "[P2]" or "(P1, P3)".
_ANSWER = re.compile(r"answer\s*:\s*(.*?)(?:\n\s*citations?\s*:|\Z)", re.IGNORECASE | re.DOTALL)
_CITATIONS = re.compile(r"citations?\s*:(.*)", re.IGNORECASE | re.DOTALL)
_LABEL = re.compile(r"\bP(\d+)\b")
_INLINE_CITE = re.compile(r"[\[(]\s*P\d+(?:\s*[,;]\s*P\d+)*\s*[\])]")


@dataclass(frozen=True)
class GeneratedAnswer:
    """The scored answer, the cited chunk ids (provenance), and the raw reply."""

    answer: str
    cited_chunk_ids: tuple[str, ...]
    raw: str


def build_prompt(question: str, hits) -> str:
    """Labelled passages, then the question, the answer rules, and the two-line reply format."""
    passages = "\n\n".join(
        f"[P{i}] ({hit.chunk.section or 'Body'}) {hit.chunk.body}" for i, hit in enumerate(hits, start=1)
    )
    return (
        f"Passages:\n{passages}\n\n"
        f"Question: {question}\n\n"
        "Rules:\n"
        "- If it is a yes/no question, the answer starts with Yes or No.\n"
        "- If the passages do not contain the answer, the answer is exactly: Unanswerable\n"
        "- Otherwise answer with a short phrase, reusing the passage wording where possible.\n\n"
        "Reply with exactly two lines:\n"
        "Answer: <answer>\n"
        "Citations: <labels of the passages you used, e.g. P1, P3; leave empty if Unanswerable>"
    )


def parse_answer(raw: str, hits) -> GeneratedAnswer:
    """Split the reply into the answer line and the cited chunk ids."""
    # No "Answer:" line: use everything before the Citations line.
    match = _ANSWER.search(raw)
    answer = match.group(1) if match else re.split(r"\n\s*citations?\s*:", raw, flags=re.IGNORECASE)[0]
    # Drop inline labels so they are not scored as answer words, then tidy spacing.
    answer = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_INLINE_CITE.sub("", answer).split()))
    cited = _CITATIONS.search(raw)
    # Without a Citations line, fall back to labels written inline in the answer.
    labels = [int(x) for x in _LABEL.findall(cited.group(1) if cited else raw)]
    # Labels are 1-based; ones beyond the passages given are ignored.
    chunk_ids = tuple(dict.fromkeys(hits[i - 1].chunk.chunk_id for i in labels if 1 <= i <= len(hits)))
    return GeneratedAnswer(answer=answer, cited_chunk_ids=chunk_ids, raw=raw)


def answer_with_citations(question: str, hits, llm) -> GeneratedAnswer:
    """Ask the LLM about ``hits`` and parse its reply."""
    raw = llm.generate(build_prompt(question, hits), system_instruction=SYSTEM_PROMPT)
    return parse_answer(raw, hits)
