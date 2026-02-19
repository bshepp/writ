"""
Validation runner for Writ knowledge graph queries.

Runs validation cases against a live NLP engine and produces scored
results with checkpoint/resume support.

Usage:
    python -m validation.run_validation --domain generic
    python -m validation.run_validation --domain generic --max-cases 5
    python -m validation.run_validation --all
    python -m validation.run_validation --domain epa --resume
    python -m validation.run_validation --list-domains
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.base import (
    ValidationResult,
    compute_summary,
    load_cases,
    load_checkpoint,
    clear_checkpoint,
    list_domains,
    print_summary,
    run_query_validation,
    save_incremental,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_engine(neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str = None):
    """Create a WritEngine for validation (no HTTP server)."""
    from query.nlp_engine_template import WritEngine

    return WritEngine(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
    )


def run_domain(
    engine,
    domain: str,
    max_cases: int = None,
    resume: bool = False,
):
    """Run validation for a single domain and return the summary."""
    cases = load_cases(domain)
    if max_cases:
        cases = cases[:max_cases]

    completed: list[ValidationResult] = []
    completed_ids: set[str] = set()

    if resume:
        completed = load_checkpoint(domain)
        completed_ids = {r.case_id for r in completed}
        if completed:
            logger.info(
                "Resumed %d/%d cases for domain '%s'",
                len(completed), len(cases), domain,
            )
    else:
        clear_checkpoint(domain)

    remaining = [c for c in cases if c.case_id not in completed_ids]
    logger.info(
        "Running %d cases for domain '%s' (%d already done)",
        len(remaining), domain, len(completed),
    )

    t0 = time.time()
    for i, case in enumerate(remaining):
        logger.info(
            "  [%d/%d] %s: %s",
            len(completed) + i + 1, len(cases), case.case_id, case.question[:60],
        )
        result = run_query_validation(engine, case)
        completed.append(result)
        save_incremental(result, domain)

        status = "PASS" if result.success else "FAIL"
        extra = f" error={result.error}" if result.error else ""
        logger.info(
            "    -> %s  conf=%.2f  strategy=%s  results=%d  %dms%s",
            status, result.engine_confidence, result.strategy_used,
            result.results_count, result.execution_time_ms, extra,
        )

    duration = time.time() - t0
    summary = compute_summary(domain, completed, duration)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run validation against Writ NLP engine",
    )
    parser.add_argument("--domain", type=str, default=None, help="Domain to validate (e.g., generic, epa)")
    parser.add_argument("--all", action="store_true", help="Run all available domains")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases per domain")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--list-domains", action="store_true", help="List available domains and exit")
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--openai-key", default=None)
    args = parser.parse_args()

    if args.list_domains:
        domains = list_domains()
        if domains:
            print("Available domains:")
            for d in domains:
                cases = load_cases(d)
                print(f"  {d}: {len(cases)} cases")
        else:
            print("No validation case files found in validation/cases/")
        return

    if not args.domain and not args.all:
        parser.error("Specify --domain <name> or --all")

    neo4j_uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")

    if not neo4j_password:
        logger.error("Neo4j password required. Set NEO4J_PASSWORD or use --neo4j-password")
        sys.exit(1)

    logger.info("Connecting to Neo4j at %s", neo4j_uri)
    engine = create_engine(neo4j_uri, neo4j_user, neo4j_password, openai_key)

    try:
        domains_to_run = []
        if args.all:
            domains_to_run = list_domains()
            if not domains_to_run:
                logger.error("No validation case files found")
                sys.exit(1)
        else:
            domains_to_run = [args.domain]

        for domain in domains_to_run:
            summary = run_domain(engine, domain, args.max_cases, args.resume)
            print_summary(summary)
            result_path = save_results(summary)
            logger.info("Results saved to %s", result_path)

    finally:
        engine.close()


if __name__ == "__main__":
    main()
