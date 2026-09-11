"""Phase 6 (Chunking) engine.

Reads Phase 5's cleaned per-video entries (``data/cleaned/sadhguru_cleaned.jsonl``,
now including Phase 5-b's ``genre`` field) and splits each video's cleaned
text into semantically coherent chunks via LlamaIndex's
``SemanticSplitterNodeParser``, for stage 7 (embeddings) to index into
Qdrant.

**Switched from token-based ``SentenceSplitter`` to semantic chunking**
(2026-09-11): a fixed ~512-token window can sever a single continuously
developing thought mid-argument just because the token budget ran out, even
though it lands on a clean sentence boundary -- the 64-token overlap only
partially compensates, it doesn't repair a severed argument. Semantic
chunking instead embeds each sentence (grouped via ``buffer_size`` to smooth
over brief tangents) and only cuts where the cosine dissimilarity to the
next sentence group crosses ``breakpoint_percentile_threshold`` -- i.e.
where the meaning actually shifts, not where an arbitrary token count is
hit. Sentence embeddings for this boundary detection come from
``hf_embeddings.HFInferenceAPIEmbedding`` (HuggingFace's free Serverless
Inference API, ``Qwen/Qwen3-Embedding-0.6B``) -- see that module's docstring
for the model-selection rationale. This is a separate, throwaway use of an
embedding model purely to find chunk boundaries; it is not Phase 7's final
RAG embedding step.

Each video is chunked independently -- chunks never span two videos, since
the ``topic``/``genre`` metadata needs to stay attached to the text it
actually describes. ``genre`` is carried through unchanged from its source
video onto every chunk of that video -- it's whole-video metadata, not
something chunk boundaries are computed from.

**Post-process size safeguard** (added 2026-09-11 after the first real run):
semantic chunking has no built-in size bound, and the first full run
produced real outliers -- 66/1176 chunks (5.6%) under 10 words (often a
single isolated interjection like "Yeah." that got cut off as its own node
because a strong dissimilarity break landed right around it), and 11 chunks
over 1000 words (one at 3005). A chunk that small carries no usable context
alone; one that large may be unwieldy downstream. ``_merge_tiny_chunks``
and ``_split_oversized_chunks`` run as a pure text post-process on each
video's node texts (no re-embedding, so no extra HF API calls) to bound
both tails while keeping the semantic boundaries as the primary split
points everywhere else.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer

from src.chunking.hf_embeddings import HFInferenceAPIEmbedding

logger = logging.getLogger(__name__)

_sentence_tokenize = split_by_sentence_tokenizer()


@dataclass(slots=True)
class CleanedEntry:
    id: str
    topic: str
    text: str
    genre: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source_id: str
    topic: str
    genre: str
    chunk_index: int
    text: str
    word_count: int


@dataclass(slots=True)
class ChunkingSummary:
    total_videos: int
    total_chunks: int
    would_chunk_videos: int
    skipped_empty: int


def _merge_tiny_chunks(texts: List[str], min_words: int) -> List[str]:
    """Merge any chunk under ``min_words`` into its previous neighbor (or,
    for a tiny first chunk with nothing before it, into the next one).
    Loops until stable, so consecutive tiny chunks are merged into one.
    """
    texts = list(texts)
    changed = True
    while changed and len(texts) > 1:
        changed = False
        for i, t in enumerate(texts):
            if len(t.split()) < min_words:
                if i == 0:
                    texts[0:2] = [texts[0].rstrip() + " " + texts[1].lstrip()]
                else:
                    texts[i - 1 : i + 1] = [texts[i - 1].rstrip() + " " + texts[i].lstrip()]
                changed = True
                break
    return texts


def _split_oversized_chunks(texts: List[str], max_words: int) -> List[str]:
    """Split any chunk over ``max_words`` back down via a plain sentence-
    greedy fill -- a fallback only for the rare oversized node, not the
    primary splitting method.
    """
    result: List[str] = []
    for t in texts:
        if len(t.split()) <= max_words:
            result.append(t)
            continue
        sentences = _sentence_tokenize(t)
        current: List[str] = []
        current_words = 0
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current and current_words + sentence_words > max_words:
                result.append("".join(current).strip())
                current, current_words = [], 0
            current.append(sentence)
            current_words += sentence_words
        if current:
            result.append("".join(current).strip())
    return result


def load_cleaned_entries(cleaned_file: Path) -> List[CleanedEntry]:
    """Load Phase 5's cleaned JSONL, one entry per video."""
    entries: List[CleanedEntry] = []
    with cleaned_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            entries.append(
                CleanedEntry(
                    id=raw["id"],
                    topic=raw["topic"],
                    text=raw["text"],
                    genre=raw["genre"],
                )
            )
    return entries


class Chunker:
    def __init__(self, config) -> None:  # config: ChunkingConfig
        self.config = config
        self._splitter: Optional[SemanticSplitterNodeParser] = None

    def _get_splitter(self) -> SemanticSplitterNodeParser:
        # Lazily constructed: building the embed model requires HF_TOKEN and
        # a network call, so --dry-run never needs either.
        if self._splitter is None:
            load_dotenv()
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise RuntimeError(
                    "HF_TOKEN not set -- required to call HuggingFace's Inference API "
                    "for semantic chunking's sentence embeddings. Set it in .env."
                )
            embed_model = HFInferenceAPIEmbedding(
                hf_token=hf_token,
                model_name=self.config.chunking.embedding_model,
                embed_batch_size=self.config.chunking.embed_batch_size,
            )
            self._splitter = SemanticSplitterNodeParser.from_defaults(
                embed_model=embed_model,
                buffer_size=self.config.chunking.buffer_size,
                breakpoint_percentile_threshold=self.config.chunking.breakpoint_percentile_threshold,
            )
        return self._splitter

    def chunk_all(self, dry_run: bool = False) -> ChunkingSummary:
        cleaned_file = self.config.input.cleaned_file
        if not cleaned_file.exists():
            raise FileNotFoundError(f"Cleaned file not found: {cleaned_file}")

        entries = load_cleaned_entries(cleaned_file)
        logger.info("Loaded %d cleaned video entries from %s", len(entries), cleaned_file)

        if dry_run:
            logger.info(
                "[dry-run] would chunk %d videos (semantic, embedding_model=%s, "
                "buffer_size=%d, breakpoint_percentile_threshold=%d, "
                "min_chunk_words=%d, max_chunk_words=%d) -> %s",
                len(entries),
                self.config.chunking.embedding_model,
                self.config.chunking.buffer_size,
                self.config.chunking.breakpoint_percentile_threshold,
                self.config.chunking.min_chunk_words,
                self.config.chunking.max_chunk_words,
                self.config.output.chunks_file,
            )
            return ChunkingSummary(
                total_videos=len(entries), total_chunks=0,
                would_chunk_videos=len(entries), skipped_empty=0,
            )

        splitter = self._get_splitter()
        all_chunks: List[Chunk] = []
        skipped_empty = 0
        for entry in entries:
            if not entry.text.strip():
                logger.warning("Skipping video %s: empty cleaned text", entry.id)
                skipped_empty += 1
                continue
            document = Document(text=entry.text)
            nodes = splitter.get_nodes_from_documents([document])
            texts = [node.get_content() for node in nodes]
            texts = _merge_tiny_chunks(texts, self.config.chunking.min_chunk_words)
            texts = _split_oversized_chunks(texts, self.config.chunking.max_chunk_words)
            for i, chunk_text in enumerate(texts):
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{entry.id}-{i:03d}",
                        source_id=entry.id,
                        topic=entry.topic,
                        genre=entry.genre,
                        chunk_index=i,
                        text=chunk_text,
                        word_count=len(chunk_text.split()),
                    )
                )
            logger.info(
                "Chunked video %s (%s): %d chunk(s) (%d before size safeguard)",
                entry.id, entry.genre, len(texts), len(nodes),
            )

        output_file = self.config.output.chunks_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

        logger.info(
            "Wrote %d chunks from %d videos (%d skipped empty) to %s",
            len(all_chunks), len(entries), skipped_empty, output_file,
        )
        return ChunkingSummary(
            total_videos=len(entries), total_chunks=len(all_chunks),
            would_chunk_videos=0, skipped_empty=skipped_empty,
        )
