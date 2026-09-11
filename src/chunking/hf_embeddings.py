"""LlamaIndex-compatible embedding wrapper around HuggingFace's free
``hf-inference`` Serverless Inference API, used only to compute
sentence-group embeddings for semantic chunk-boundary detection
(``SemanticSplitterNodeParser``) -- not Phase 7's final RAG embeddings.

Model: ``BAAI/bge-m3``, called via ``huggingface_hub``'s
``InferenceClient.feature_extraction``.

**Not Qwen3-Embedding** (tried first): it benchmarks higher than BGE-M3 on
MTEB, but ``huggingface_hub``'s ``InferenceClient`` only serves it through
paid third-party marketplace partners (it auto-routed to "deepinfra"),
which burned through the account's $0.10/month Inference Providers credit
within ~115/199 videos of the first real run (repeated ``402 Payment
Required``) -- see PROJECT_UPDATES.md's "Phase 6 (chunking)" section for the
full story. Verified directly that ``BAAI/bge-m3`` (unlike Qwen3-Embedding)
*is* served by HF's own free ``hf-inference`` provider -- the client below
pins ``provider="hf-inference"`` explicitly so it can never silently
fall back to a paid partner again. This also gives BGE-M3 for both chunk
boundaries here and Phase 7's final embeddings, for free, with one model.

Requires ``HF_TOKEN`` in ``.env`` (already set from Phase 4's diarization
setup).
"""
from __future__ import annotations

import logging
import time
from typing import Any, List

import numpy as np
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_PROVIDER = "hf-inference"

# The free hf-inference tier can return a transient error while a model is
# cold-starting (observed as a 503) or under a short-lived rate limit --
# worth a short backoff-and-retry rather than failing the whole chunking
# run. A 402 (payment required -- i.e. the model isn't actually free on
# this provider) is NOT retried: it will never succeed, so failing fast
# with a clear message is better than burning 5 retries on every call.
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 10


class HFInferenceAPIEmbedding(BaseEmbedding):
    """Embeds text via HuggingFace's free hf-inference Serverless API."""

    _client: InferenceClient = PrivateAttr()

    def __init__(
        self,
        hf_token: str,
        model_name: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._client = InferenceClient(model=model_name, provider=provider, token=hf_token)

    @classmethod
    def class_name(cls) -> str:
        return "HFInferenceAPIEmbedding"

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._client.feature_extraction(texts)
                return np.asarray(result, dtype=np.float32).tolist()
            except HfHubHTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if status_code == 402:
                    raise RuntimeError(
                        f"HF Inference API returned 402 Payment Required for "
                        f"model={self.model_name!r} provider={DEFAULT_PROVIDER!r} -- this "
                        "model is not actually free on this provider; pick a different one."
                    ) from exc
                if attempt == _MAX_RETRIES:
                    raise
                wait = _BACKOFF_SECONDS * attempt
                logger.warning(
                    "HF Inference API call failed (attempt %d/%d): %s -- retrying in %ds",
                    attempt, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_batch([query])[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
