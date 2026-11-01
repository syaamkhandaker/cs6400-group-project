"""Chunk papers to the MiniLM token cap, prefixed with title and section.

Each paragraph becomes one chunk unless it exceeds the token budget, in which
case it is split into windows. Unsplit paragraphs keep their original text so
QASPER evidence can still be matched exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rag.data.papers import Paper

ENCODER_LIMIT = 256
MAX_PREFIX_TOKENS = 48
SPECIAL_TOKENS = 2


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    body: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Chunk:
        return cls(
            chunk_id=raw["chunk_id"],
            doc_id=raw["doc_id"],
            title=raw["title"],
            section=raw["section"],
            text=raw["text"],
            body=raw["body"],
        )


class Chunker:
    def __init__(self, tokenizer, max_tokens: int = 200):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        # Counting tokens on a long paragraph is how we split it. The embedding
        # model never sees that full paragraph, so the tokenizer's 512 warning
        # does not apply here.
        if hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = max(int(tokenizer.model_max_length), 1_000_000)

    @classmethod
    def from_config(cls, config: dict) -> Chunker:
        from transformers import AutoTokenizer

        chunking = config["chunking"]
        tokenizer = AutoTokenizer.from_pretrained(chunking["tokenizer"])
        return cls(tokenizer, max_tokens=int(chunking["max_tokens"]))

    def chunk_paper(self, paper: Paper) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section_idx, (section, paragraphs) in enumerate(paper.sections):
            prefix, budget = self._prefix_and_budget(paper.title, section)
            for para_idx, paragraph in enumerate(paragraphs):
                for part, body in enumerate(self._split(paragraph, budget)):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{paper.doc_id}:{section_idx}:{para_idx}:{part}",
                            doc_id=paper.doc_id,
                            title=paper.title,
                            section=section,
                            text=f"{prefix}{body}",
                            body=body,
                        )
                    )
        return chunks

    def _prefix_and_budget(self, title: str, section: str) -> tuple[str, int]:
        label = section.strip() or "Body"
        prefix = f"{title.strip()} [{label}]\n"
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        if len(prefix_ids) > MAX_PREFIX_TOKENS:
            prefix_ids = prefix_ids[:MAX_PREFIX_TOKENS]
            prefix = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            if not prefix.endswith("\n"):
                prefix += "\n"
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        room = ENCODER_LIMIT - SPECIAL_TOKENS - len(prefix_ids)
        return prefix, max(1, min(self.max_tokens, room))

    def _split(self, paragraph: str, budget: int) -> list[str]:
        token_ids = self.tokenizer.encode(paragraph, add_special_tokens=False)
        if not token_ids:
            return []
        if len(token_ids) <= budget:
            body = paragraph.strip()
            return [body] if body else []
        pieces = []
        for start in range(0, len(token_ids), budget):
            piece = self.tokenizer.decode(
                token_ids[start : start + budget],
                skip_special_tokens=True,
            ).strip()
            if piece:
                pieces.append(piece)
        return pieces
