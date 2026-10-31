"""Golden evaluation dataset (§5.8, Stage 45).

Each line is a newline-delimited JSON record carrying the minimum fields
required by `metrics.retrieval_metrics_for_query()` and the evaluation
runner: ``query`` (str), ``relevant_chunks`` (list of chunk ids), and an
optional ``graded_relevance`` (map chunk id -> float). The file is read
line-by-line so it can grow without loading everything into memory.

No model answers are stored here: this is the retrieval-evaluation
corpus, not the answer-grounding golden set. The citation-verification
tests (`tests/unit/test_citations.py`, `tests/unit/test_answer_pipeline.py`)
already cover the grounding layer; this dataset covers retrieval recall.
"""

from pathlib import Path

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
