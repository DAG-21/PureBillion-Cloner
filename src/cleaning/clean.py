"""
Cleans and normalizes diarized transcripts: strips filler words, fixes casing/
punctuation, and isolates the target speaker's turns.

Input: data/diarized/*.json
Output: data/cleaned/<id>.jsonl
"""
