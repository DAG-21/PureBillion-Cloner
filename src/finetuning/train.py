"""
Fine-tunes a base LLM on persona instruction-response pairs using PEFT
LoRA/QLoRA.

Input: data/processed/finetune_pairs/*.jsonl
Output: checkpoints/<run_name>/
"""
