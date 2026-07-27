# persona-clone

Text-to-text persona/style replication pipeline for a public figure, grounded
in their real public statements. Combines a RAG-grounded tier with a
fine-tuning tier, plus evaluation and serving.

## Pipeline stages

1. **acquisition** — download source video/audio via yt-dlp.
2. **audio** — extract audio tracks from raw video via ffmpeg.
3. **transcription** — transcribe audio with faster-whisper.
4. **diarization** — separate speakers with pyannote.audio and isolate the target speaker.
5. **cleaning** — normalize and clean diarized transcripts.
6. **chunking** — split cleaned text into semantic chunks with LlamaIndex.
7. **embeddings** — embed chunks (BGE-M3) and index them in Qdrant for RAG.
8. **dataset_gen** — synthesize instruction-response pairs for fine-tuning.
9. **finetuning** — LoRA/QLoRA fine-tune a base LLM on the generated pairs.
10. **rag** — retrieve relevant context from Qdrant at inference time.
11. **serving** — serve the persona clone via FastAPI (+ vLLM).

Evaluation (BERTScore, ROUGE, BLEU, RAGAS) lives in `eval/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in secrets
```
