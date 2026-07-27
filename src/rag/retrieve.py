"""
Indexes embedded chunks in Qdrant and retrieves relevant context for a
given query at inference time.

Input: query string; Qdrant collection populated by src/embeddings/embed.py
Output: ranked list of retrieved chunk texts
"""
