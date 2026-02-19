"""
Validation framework for Writ knowledge graph queries.

Provides data structures, scoring, fuzzy matching, and checkpoint/resume
for measuring NLP engine accuracy across regulatory domains.

Adapted from the MedGemma validation harness pattern: direct engine
invocation (no HTTP server), incremental checkpointing (JSONL), and
aggregate scoring with pretty-print summaries.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidationCase:
    """A single validation test case."""
    case_id: str
    domain: str
    question: str
    expected_entity_type: str = ""
    expected_keywords: List[str] = field(default_factory=list)
    expected_min_results: int = 0
    expected_strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of running one case through the engine + scoring."""
    case_id: str
    domain: str
    success: bool
    scores: Dict[str, float] = field(default_factory=dict)
    engine_confidence: float = 0.0
    strategy_used: str = ""
    execution_time_ms: int = 0
    results_count: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Aggregate metrics for a domain validation run."""
    domain: str
    total_cases: int
    passed: int
    failed: int
    metrics: Dict[str, float] = field(default_factory=dict)
    per_case: List[ValidationResult] = field(default_factory=list)
    run_duration_sec: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── Fuzzy matching ────────────────────────────────────────────────────────────

_REGULATORY_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "are", "was",
    "were", "be", "been", "with", "for", "on", "at", "by", "from", "this",
    "that", "these", "those", "it", "its", "has", "have", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "which", "what", "all", "any", "each", "every", "some", "no",
    "must", "shall", "required", "following", "section",
})


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _content_tokens(text: str) -> set:
    """Extract meaningful content tokens, removing regulatory stopwords."""
    return set(normalize_text(text).split()) - _REGULATORY_STOPWORDS


def fuzzy_match(candidate: str, target: str, threshold: float = 0.6) -> bool:
    """Check if candidate text is a fuzzy match for target.

    Strategy (checked in order, first match wins):
      1. Normalized substring containment (either direction)
      2. All content tokens of target appear in candidate
      3. Token overlap ratio >= threshold
    """
    c_norm = normalize_text(candidate)
    t_norm = normalize_text(target)

    if not t_norm:
        return False

    if t_norm in c_norm or c_norm in t_norm:
        return True

    t_content = _content_tokens(target)
    c_content = _content_tokens(candidate)

    if t_content and t_content.issubset(c_content):
        return True

    if not t_content or not c_content:
        return False

    overlap = len(t_content & c_content)
    recall = overlap / len(t_content)
    return recall >= threshold


# ── Strategy inference ────────────────────────────────────────────────────────

def infer_strategy(explanation: str, confidence: float) -> str:
    """Infer which cascade strategy produced results from the explanation."""
    if not explanation:
        return "unknown"
    e = explanation.lower()
    if "pattern" in e:
        return "pattern_match"
    if "langchain" in e:
        return "langchain"
    if "ai" in e and "natural language" in e:
        return "direct_ai"
    if "full-text" in e or "fulltext" in e:
        return "fulltext"
    if confidence >= 0.85:
        return "pattern_match"
    if confidence >= 0.80:
        return "langchain"
    if confidence >= 0.75:
        return "direct_ai"
    if confidence >= 0.50:
        return "fulltext"
    return "no_match"


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_result(query_result, case: ValidationCase) -> Dict[str, float]:
    """Score a QueryResult against a ValidationCase.

    Metrics returned:
      - keyword_recall: fraction of expected_keywords found in results
      - result_count_met: 1.0 if result count >= expected_min_results
      - confidence_met: 1.0 if engine confidence >= 0.5
      - strategy_match: 1.0 if the inferred strategy matches expected
    """
    scores: Dict[str, float] = {}

    results = query_result.results or []

    if case.expected_keywords:
        result_text = " ".join(
            str(v)
            for r in results
            for v in r.values()
            if v is not None
        ).lower()
        found = sum(
            1 for kw in case.expected_keywords
            if fuzzy_match(result_text, kw, threshold=0.5)
        )
        scores["keyword_recall"] = found / len(case.expected_keywords)
    else:
        scores["keyword_recall"] = 1.0

    if case.expected_min_results > 0:
        scores["result_count_met"] = 1.0 if len(results) >= case.expected_min_results else 0.0
    else:
        scores["result_count_met"] = 1.0

    scores["confidence_met"] = 1.0 if query_result.confidence >= 0.5 else 0.0

    if case.expected_strategy:
        actual = infer_strategy(query_result.explanation, query_result.confidence)
        scores["strategy_match"] = 1.0 if actual == case.expected_strategy else 0.0

    return scores


def run_query_validation(engine, case: ValidationCase) -> ValidationResult:
    """Run a single validation case through the engine and score it."""
    try:
        t0 = time.time()
        result = engine.query(case.question)
        elapsed_ms = round((time.time() - t0) * 1000)

        scores = score_result(result, case)
        strategy = infer_strategy(result.explanation, result.confidence)

        passing = all(v >= 0.5 for v in scores.values())

        return ValidationResult(
            case_id=case.case_id,
            domain=case.domain,
            success=passing,
            scores=scores,
            engine_confidence=result.confidence,
            strategy_used=strategy,
            execution_time_ms=elapsed_ms,
            results_count=len(result.results),
        )
    except Exception as e:
        return ValidationResult(
            case_id=case.case_id,
            domain=case.domain,
            success=False,
            error=str(e),
        )


def compute_summary(
    domain: str,
    results: List[ValidationResult],
    run_duration_sec: float,
) -> ValidationSummary:
    """Compute aggregate metrics from a list of ValidationResults."""
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    metrics: Dict[str, float] = {}

    successful = [r for r in results if r.error is None]
    if successful:
        all_score_keys = set()
        for r in successful:
            all_score_keys.update(r.scores.keys())
        for key in sorted(all_score_keys):
            vals = [r.scores.get(key, 0.0) for r in successful]
            metrics[f"avg_{key}"] = sum(vals) / len(vals)

        metrics["avg_confidence"] = (
            sum(r.engine_confidence for r in successful) / len(successful)
        )
        metrics["avg_execution_time_ms"] = (
            sum(r.execution_time_ms for r in successful) / len(successful)
        )

    strategy_counts: Dict[str, int] = {}
    for r in successful:
        s = r.strategy_used or "unknown"
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    for s, c in strategy_counts.items():
        metrics[f"strategy_{s}_count"] = float(c)

    return ValidationSummary(
        domain=domain,
        total_cases=len(results),
        passed=passed,
        failed=failed,
        metrics=metrics,
        per_case=results,
        run_duration_sec=run_duration_sec,
    )


# ── Case loading ──────────────────────────────────────────────────────────────

CASES_DIR = Path(__file__).resolve().parent / "cases"


def load_cases(domain: str) -> List[ValidationCase]:
    """Load validation cases from a JSON file for the given domain."""
    path = CASES_DIR / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(f"No validation cases found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        ValidationCase(
            case_id=d["case_id"],
            domain=d.get("domain", domain),
            question=d["question"],
            expected_entity_type=d.get("expected_entity_type", ""),
            expected_keywords=d.get("expected_keywords", []),
            expected_min_results=d.get("expected_min_results", 0),
            expected_strategy=d.get("expected_strategy", ""),
            metadata=d.get("metadata", {}),
        )
        for d in data
    ]


def list_domains() -> List[str]:
    """List all available validation domains (JSON files in cases/)."""
    if not CASES_DIR.exists():
        return []
    return sorted(p.stem for p in CASES_DIR.glob("*.json"))


# ── Checkpoint I/O (JSONL) ────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _result_to_dict(r: ValidationResult) -> dict:
    """Convert a ValidationResult to a serialisable dict."""
    return {
        "case_id": r.case_id,
        "domain": r.domain,
        "success": r.success,
        "scores": r.scores,
        "engine_confidence": r.engine_confidence,
        "strategy_used": r.strategy_used,
        "execution_time_ms": r.execution_time_ms,
        "results_count": r.results_count,
        "error": r.error,
        "details": r.details,
    }


def checkpoint_path(domain: str) -> Path:
    """Return the path to the checkpoint JSONL for a domain."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{domain}_checkpoint.jsonl"


def save_incremental(result: ValidationResult, domain: str) -> None:
    """Append a single case result to the checkpoint JSONL file."""
    path = checkpoint_path(domain)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_result_to_dict(result), default=str) + "\n")


def load_checkpoint(domain: str) -> List[ValidationResult]:
    """Load previously-completed results from the checkpoint file."""
    path = checkpoint_path(domain)
    if not path.exists():
        return []
    results: List[ValidationResult] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        results.append(ValidationResult(
            case_id=d["case_id"],
            domain=d.get("domain", domain),
            success=d["success"],
            scores=d.get("scores", {}),
            engine_confidence=d.get("engine_confidence", 0.0),
            strategy_used=d.get("strategy_used", ""),
            execution_time_ms=d.get("execution_time_ms", 0),
            results_count=d.get("results_count", 0),
            error=d.get("error"),
            details=d.get("details", {}),
        ))
    return results


def clear_checkpoint(domain: str) -> None:
    """Delete checkpoint file for a fresh run."""
    path = checkpoint_path(domain)
    if path.exists():
        path.unlink()


# ── Results save & print ──────────────────────────────────────────────────────

def save_results(summary: ValidationSummary, filename: Optional[str] = None) -> Path:
    """Save validation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{summary.domain}_{ts}.json"
    path = RESULTS_DIR / filename
    data = {
        "domain": summary.domain,
        "total_cases": summary.total_cases,
        "passed": summary.passed,
        "failed": summary.failed,
        "metrics": summary.metrics,
        "run_duration_sec": summary.run_duration_sec,
        "timestamp": summary.timestamp,
        "per_case": [_result_to_dict(r) for r in summary.per_case],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def print_summary(summary: ValidationSummary) -> None:
    """Pretty-print validation results."""
    print(f"\n{'=' * 60}")
    print(f"  Validation Results: {summary.domain.upper()}")
    print(f"{'=' * 60}")
    print(f"  Total cases:    {summary.total_cases}")
    print(f"  Passed:         {summary.passed}")
    print(f"  Failed:         {summary.failed}")
    print(f"  Duration:       {summary.run_duration_sec:.1f}s")

    print(f"\n  Metrics:")
    for metric, value in sorted(summary.metrics.items()):
        if "time" in metric and isinstance(value, (int, float)):
            print(f"    {metric:35s} {value:.0f}ms")
        elif isinstance(value, float) and value <= 1.0 and "count" not in metric:
            print(f"    {metric:35s} {value:.1%}")
        else:
            print(f"    {metric:35s} {value:.1f}")

    if summary.per_case:
        failed_cases = [r for r in summary.per_case if not r.success]
        if failed_cases:
            print(f"\n  Failed Cases:")
            for r in failed_cases:
                err = f" error={r.error}" if r.error else ""
                scores_str = ", ".join(f"{k}={v:.2f}" for k, v in r.scores.items())
                print(f"    {r.case_id}: {scores_str}{err}")

    print(f"{'=' * 60}\n")
