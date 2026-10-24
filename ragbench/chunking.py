"""
RAGScaleBench Token-Aware, Sentence-Respecting Chunker.
Accurately preserves character offsets, token boundaries, and stable chunk identifiers.
"""

import re

import tiktoken
from pydantic import BaseModel

from ragbench.ids import make_chunk_id


class Chunk(BaseModel):
    """Schema for a processed document chunk."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    text: str
    title: str = ""
    source: str = ""


class SentenceSpan(BaseModel):
    text: str
    start_char: int
    end_char: int
    tokens: list[int]


class DocumentChunker:
    """
    Sentence-respecting, token-aware chunker.
    Preserves exact character offsets in original text for dynamic QA evaluation.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ):
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be 0 <= overlap < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _split_into_sentences(self, text: str) -> list[SentenceSpan]:
        """Split text into sentence spans with exact character offsets."""
        sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\n+")
        spans: list[SentenceSpan] = []
        last_idx = 0

        for match in sentence_end.finditer(text):
            split_idx = match.start()
            s_text = text[last_idx:split_idx].strip()
            if s_text:
                start_c = text.find(s_text, last_idx)
                end_c = start_c + len(s_text)
                tokens = self.tokenizer.encode(s_text)
                spans.append(
                    SentenceSpan(text=s_text, start_char=start_c, end_char=end_c, tokens=tokens)
                )
            last_idx = match.end()

        if last_idx < len(text):
            s_text = text[last_idx:].strip()
            if s_text:
                start_c = text.find(s_text, last_idx)
                end_c = start_c + len(s_text)
                tokens = self.tokenizer.encode(s_text)
                spans.append(
                    SentenceSpan(text=s_text, start_char=start_c, end_char=end_c, tokens=tokens)
                )

        if not spans and text.strip():
            s_text = text.strip()
            start_c = text.find(s_text)
            tokens = self.tokenizer.encode(s_text)
            spans.append(
                SentenceSpan(
                    text=s_text, start_char=start_c, end_char=start_c + len(s_text), tokens=tokens
                )
            )

        return spans

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        title: str = "",
        source: str = "",
    ) -> list[Chunk]:
        """Split a document into token-aware, sentence-respecting chunks."""
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        chunk_idx = 0
        start = 0
        curr_token_offset = 0

        while start < len(sentences):
            sent = sentences[start]
            sent_token_len = len(sent.tokens)

            # If single sentence exceeds chunk_size
            if sent_token_len > self.chunk_size:
                step = self.chunk_size - self.overlap
                for sub_start in range(0, sent_token_len, step):
                    sub_end = min(sub_start + self.chunk_size, sent_token_len)
                    sub_tokens = sent.tokens[sub_start:sub_end]
                    sub_text = self.tokenizer.decode(sub_tokens)
                    start_c = sent.text.find(sub_text[: min(20, len(sub_text))])
                    if start_c == -1:
                        start_c = 0
                    start_char = sent.start_char + start_c
                    end_char = min(start_char + len(sub_text), sent.end_char)
                    chunks.append(
                        Chunk(
                            chunk_id=make_chunk_id(
                                doc_id, self.chunk_size, self.overlap, chunk_idx
                            ),
                            doc_id=doc_id,
                            chunk_index=chunk_idx,
                            start_char=start_char,
                            end_char=end_char,
                            start_token=curr_token_offset + sub_start,
                            end_token=curr_token_offset + sub_end,
                            text=sub_text,
                            title=title,
                            source=source,
                        )
                    )
                    chunk_idx += 1
                    if sub_end == sent_token_len:
                        break
                curr_token_offset += sent_token_len
                start += 1
                continue

            # Accumulate sentences up to chunk_size
            end = start
            total_tokens = 0
            while end < len(sentences) and total_tokens + len(sentences[end].tokens) <= self.chunk_size:
                total_tokens += len(sentences[end].tokens)
                end += 1

            c_start_char = sentences[start].start_char
            c_end_char = sentences[end - 1].end_char
            c_text = text[c_start_char:c_end_char]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, self.chunk_size, self.overlap, chunk_idx),
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    start_char=c_start_char,
                    end_char=c_end_char,
                    start_token=curr_token_offset,
                    end_token=curr_token_offset + total_tokens,
                    text=c_text,
                    title=title,
                    source=source,
                )
            )
            chunk_idx += 1

            if end >= len(sentences):
                break

            # Next chunk start with overlap
            if self.overlap > 0:
                overlap_acc = 0
                s = end - 1
                while s > start:
                    overlap_acc += len(sentences[s].tokens)
                    if overlap_acc >= self.overlap:
                        break
                    s -= 1
                next_start = max(s, start + 1)
                tokens_advanced = sum(len(sentences[k].tokens) for k in range(start, next_start))
                curr_token_offset += tokens_advanced
                start = next_start
            else:
                curr_token_offset += total_tokens
                start = end

        return chunks
