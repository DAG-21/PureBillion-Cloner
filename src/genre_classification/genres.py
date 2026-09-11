"""Phase 5-b genre taxonomy and the id -> genre mapping.

The 8 genres below and the assignment of each of Phase 5's 199 topics to one
of them were produced by direct semantic classification -- reading all 199
real video titles (and, where a title was ambiguous, the underlying topic)
and judging what each is actually about -- not by an unsupervised clustering
algorithm (e.g. TF-IDF + KMeans) and not by a runtime LLM API call. See
PROJECT_UPDATES.md's "Phase 5-b (genre classification)" section for the full
rationale on why this method was chosen over those alternatives.

This mapping is static and keyed by Phase 5's sequential ``id`` field
(``"0000"``-``"0198"``), not by ``topic`` text, since ``id`` is the stable
identifier already used elsewhere in the pipeline (e.g. chunking's
``source_id``). It intentionally does not auto-generate: if Phase 5 is ever
rerun against a different/larger export file, this mapping must be
re-derived by the same direct-classification process for the new id set --
``classifier.py`` fails loudly (``KeyError``) on any id it doesn't cover,
rather than silently leaving some entries genre-less or guessing a default.
"""
from __future__ import annotations

from typing import Dict, List

MIND_CONSCIOUSNESS = "Mind & Consciousness"
MENTAL_HEALTH = "Mental Health & Emotional Wellbeing"
RELATIONSHIPS_LOVE = "Relationships & Love"
PHYSICAL_HEALTH_NUTRITION = "Physical Health & Nutrition"
MYSTICISM_OCCULT = "Mysticism & Occult"
SPIRITUAL_PRACTICE = "Spiritual Practice & Inner Engineering"
LIFE_PHILOSOPHY = "Life Philosophy & Existential Inquiry"
PERSONAL_GROWTH = "Personal Growth & Self-Mastery"

GENRES: List[str] = [
    MIND_CONSCIOUSNESS,
    MENTAL_HEALTH,
    RELATIONSHIPS_LOVE,
    PHYSICAL_HEALTH_NUTRITION,
    MYSTICISM_OCCULT,
    SPIRITUAL_PRACTICE,
    LIFE_PHILOSOPHY,
    PERSONAL_GROWTH,
]

# id lists per genre, grouped for readability/auditability -- flattened into
# GENRE_BY_ID below. Every id 0000-0198 appears in exactly one list.

_MIND_CONSCIOUSNESS_IDS = [
    "0001", "0003", "0008", "0009", "0017", "0027", "0029", "0031", "0035", "0037",
    "0042", "0044", "0053", "0056", "0058", "0060", "0063", "0072", "0073", "0076",
    "0091", "0092", "0100", "0106", "0108", "0110", "0112", "0115", "0119", "0121",
    "0129", "0132", "0136", "0139", "0152", "0153", "0155", "0163", "0165", "0171",
    "0175", "0178", "0183", "0195",
]

_MENTAL_HEALTH_IDS = [
    "0005", "0010", "0028", "0046", "0065", "0071", "0074", "0077", "0082", "0084",
    "0085", "0086", "0088", "0096", "0101", "0144", "0149", "0158", "0162", "0169",
    "0179", "0190",
]

_RELATIONSHIPS_LOVE_IDS = [
    "0012", "0020", "0024", "0030", "0049", "0057", "0059", "0062", "0066", "0068",
    "0079", "0093", "0094", "0097", "0102", "0113", "0117", "0123", "0130", "0138",
    "0140", "0147", "0150", "0151", "0154", "0167", "0176", "0177",
]

_PHYSICAL_HEALTH_NUTRITION_IDS = [
    "0002", "0016", "0023", "0033", "0043", "0064", "0083", "0090", "0099", "0109",
    "0118", "0131", "0148", "0197",
]

_MYSTICISM_OCCULT_IDS = [
    "0004", "0013", "0015", "0022", "0026", "0032", "0047", "0051", "0052", "0061",
    "0075", "0095", "0168", "0173", "0180", "0188", "0189", "0198",
]

_SPIRITUAL_PRACTICE_IDS = [
    "0025", "0041", "0078", "0114", "0127", "0133", "0145", "0170", "0184", "0185",
    "0192",
]

_LIFE_PHILOSOPHY_IDS = [
    "0011", "0021", "0034", "0039", "0040", "0045", "0054", "0067", "0069", "0070",
    "0080", "0081", "0087", "0089", "0098", "0104", "0105", "0111", "0116", "0120",
    "0122", "0124", "0128", "0137", "0142", "0143", "0164", "0166", "0172", "0174",
    "0181", "0186", "0187", "0191", "0193", "0194",
]

_PERSONAL_GROWTH_IDS = [
    "0000", "0006", "0007", "0014", "0018", "0019", "0036", "0038", "0048", "0050",
    "0055", "0103", "0107", "0125", "0126", "0134", "0135", "0141", "0146", "0156",
    "0157", "0159", "0160", "0161", "0182", "0196",
]

GENRE_BY_ID: Dict[str, str] = {}
for _genre, _ids in (
    (MIND_CONSCIOUSNESS, _MIND_CONSCIOUSNESS_IDS),
    (MENTAL_HEALTH, _MENTAL_HEALTH_IDS),
    (RELATIONSHIPS_LOVE, _RELATIONSHIPS_LOVE_IDS),
    (PHYSICAL_HEALTH_NUTRITION, _PHYSICAL_HEALTH_NUTRITION_IDS),
    (MYSTICISM_OCCULT, _MYSTICISM_OCCULT_IDS),
    (SPIRITUAL_PRACTICE, _SPIRITUAL_PRACTICE_IDS),
    (LIFE_PHILOSOPHY, _LIFE_PHILOSOPHY_IDS),
    (PERSONAL_GROWTH, _PERSONAL_GROWTH_IDS),
):
    for _id in _ids:
        if _id in GENRE_BY_ID:
            raise ValueError(f"id {_id!r} assigned to more than one genre")
        GENRE_BY_ID[_id] = _genre
