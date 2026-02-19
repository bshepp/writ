"""
GPU-aware CrossEncoder reranker to improve result ordering.

Requires: pip install sentence-transformers
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # type: ignore


def _torch_device() -> str:
    """Return preferred torch device."""
    try:
        import torch  # type: ignore
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_reranker(model_name: str = "BAAI/bge-reranker-base") -> Any:
    if CrossEncoder is None:
        raise RuntimeError(
            "sentence-transformers not installed. pip install sentence-transformers"
        )
    return CrossEncoder(model_name, device=_torch_device())


def rerank(
    query: str,
    candidates: List[Tuple[str, Dict[str, Any]]],
    model: Optional[Any] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Re-rank candidates by relevance to query.

    ``candidates`` is a list of ``(text, meta)`` tuples.
    Returns top_k dicts sorted by score descending: ``{text, meta, score}``.
    """
    owned = False
    if model is None:
        model = load_reranker()
        owned = True
    try:
        pairs = [(query, c[0]) for c in candidates]
        scores: Sequence[float] = model.predict(pairs)  # type: ignore[attr-defined]
        items: List[Dict[str, Any]] = []
        for (t, m), s in zip(candidates, scores):
            try:
                score_f = float(s)
            except Exception:
                score_f = 0.0
            items.append({"text": t, "meta": m, "score": score_f})
        ranked = sorted(items, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return ranked[:top_k]
    finally:
        if owned:
            try:
                if hasattr(model, "model"):
                    model.model.cpu()  # type: ignore[attr-defined]
            except Exception:
                pass
