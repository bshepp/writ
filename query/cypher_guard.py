"""
Cypher Guard -- read-only validation for LLM-generated Cypher queries.

Prevents LLM-generated Cypher from executing write operations (CREATE, DELETE,
SET, MERGE, REMOVE, DROP) against Neo4j.  Use in any Writ knowledge graph
project that generates Cypher from natural language.

Usage:
    from cypher_guard import SafeGraphCypherQAChain, is_read_only_cypher

    # Standalone check
    if not is_read_only_cypher(generated_cypher):
        raise CypherWriteBlockedError(generated_cypher, find_write_violations(generated_cypher))

    # Drop-in replacement for GraphCypherQAChain (when LangChain available)
    chain = SafeGraphCypherQAChain.from_llm(...)
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Cypher write-operation keywords ──────────────────────────────────────────
_WRITE_KEYWORDS = [
    "CREATE",
    "MERGE",
    "DELETE",
    "DETACH",
    "SET",
    "REMOVE",
    "DROP",
    "LOAD CSV",
    "FOREACH",
]

_WRITE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _WRITE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
_STRING_LITERAL_PATTERN = re.compile(r"'[^']*'|\"[^\"]*\"")
_COMMENT_PATTERN = re.compile(r"//[^\n]*")


def _strip_literals_and_comments(cypher: str) -> str:
    """Remove string literals and comments so keyword checks don't false-positive."""
    cleaned = _COMMENT_PATTERN.sub("", cypher)
    cleaned = _STRING_LITERAL_PATTERN.sub("''", cleaned)
    return cleaned


def is_read_only_cypher(cypher: str) -> bool:
    """Return True if the Cypher query contains no write operations."""
    if not cypher or not cypher.strip():
        return True
    cleaned = _strip_literals_and_comments(cypher)
    return _WRITE_PATTERN.search(cleaned) is None


def find_write_violations(cypher: str) -> List[str]:
    """Return list of write keywords found in the Cypher query."""
    if not cypher or not cypher.strip():
        return []
    cleaned = _strip_literals_and_comments(cypher)
    return [m.group(0).upper() for m in _WRITE_PATTERN.finditer(cleaned)]


class CypherWriteBlockedError(Exception):
    """Raised when a generated Cypher query contains write operations."""

    def __init__(self, cypher: str, violations: List[str]):
        self.cypher = cypher
        self.violations = violations
        super().__init__(
            f"Blocked Cypher write operation: {', '.join(violations)}"
        )


# ── Safe chain wrapper (requires LangChain) ───────────────────────────────────
try:
    from langchain_neo4j import GraphCypherQAChain
    from langchain_neo4j.chains.graph_qa.cypher import extract_cypher
    _HAS_CHAIN = True
except ImportError:
    try:
        from langchain_community.chains.graph_qa.cypher import (
            GraphCypherQAChain,
            extract_cypher,
        )
        _HAS_CHAIN = True
    except ImportError:
        _HAS_CHAIN = False

if _HAS_CHAIN:
    from langchain_core.callbacks import CallbackManagerForChainRun

    class SafeGraphCypherQAChain(GraphCypherQAChain):
        """GraphCypherQAChain with a read-only Cypher guard."""

        def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Dict[str, Any]:
            _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
            callbacks = _run_manager.get_child()
            question = inputs[self.input_key]
            args = {"question": question, "schema": self.graph_schema}
            args.update(inputs)

            intermediate_steps: List = []
            generated_cypher = self.cypher_generation_chain.run(
                args, callbacks=callbacks
            )
            generated_cypher = extract_cypher(generated_cypher)

            if self.cypher_query_corrector:
                generated_cypher = self.cypher_query_corrector(generated_cypher)

            violations = find_write_violations(generated_cypher)
            if violations:
                logger.warning(
                    "Blocked LLM-generated Cypher with write operations: %s | query=%s",
                    violations,
                    generated_cypher[:200],
                )
                intermediate_steps.append({"query": generated_cypher})
                intermediate_steps.append({"context": []})
                chain_result: Dict[str, Any] = {
                    self.output_key: (
                        "I can only answer questions about the regulations. "
                        "I cannot modify the database."
                    )
                }
                if self.return_intermediate_steps:
                    chain_result["intermediate_steps"] = intermediate_steps
                return chain_result

            intermediate_steps.append({"query": generated_cypher})
            if generated_cypher:
                context = self.graph.query(generated_cypher)[: self.top_k]
            else:
                context = []

            if self.return_direct:
                final_result = context
            else:
                intermediate_steps.append({"context": context})
                result = self.qa_chain.invoke(
                    {"question": question, "context": context},
                    callbacks=callbacks,
                )
                final_result = result[self.qa_chain.output_key]

            chain_result = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result["intermediate_steps"] = intermediate_steps

            return chain_result
