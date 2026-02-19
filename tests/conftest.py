"""Shared pytest fixtures for Writ knowledge graph tests."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_lambda_path = str(PROJECT_ROOT / "lambda" / "query_handler")
if _lambda_path not in sys.path:
    sys.path.insert(0, _lambda_path)


# ---------------------------------------------------------------------------
# Module loaders (avoids name collisions between lambda_function.py files)
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: Path):
    """Load a Python module from an absolute path."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def query_handler_module():
    """Import the query handler Lambda function module."""
    with patch("boto3.client") as mock_client:
        mock_client.return_value = MagicMock()
        return _load_module(
            "query_handler_lambda",
            PROJECT_ROOT / "lambda" / "query_handler" / "lambda_function.py",
        )


# ---------------------------------------------------------------------------
# Unit-test fixtures (no external services)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client():
    """Return a mocked OpenAI client with canned responses."""
    client = MagicMock()
    chat_response = MagicMock()
    chat_response.choices = [MagicMock()]
    chat_response.choices[0].message.content = "This is a test AI summary."
    client.chat.completions.create.return_value = chat_response

    embedding_response = MagicMock()
    embedding_response.data = [MagicMock()]
    embedding_response.data[0].embedding = [0.1] * 1536
    client.embeddings.create.return_value = embedding_response
    return client


@pytest.fixture
def mock_neo4j_driver():
    """Return a mocked Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


@pytest.fixture
def mock_lambda_context():
    """Return a mocked Lambda context object."""
    ctx = MagicMock()
    ctx.aws_request_id = "test-request-id-12345"
    ctx.function_name = "test-function"
    ctx.memory_limit_in_mb = 512
    return ctx


@pytest.fixture
def sample_query_event():
    """Return a sample API Gateway POST event for /api/query."""
    return {
        "httpMethod": "POST",
        "path": "/api/query",
        "body": json.dumps({"query": "What are the main requirements?"}),
        "headers": {"Content-Type": "application/json"},
    }


@pytest.fixture
def sample_options_event():
    """Return a sample CORS preflight event."""
    return {"httpMethod": "OPTIONS", "path": "/api/query", "headers": {}}


# ---------------------------------------------------------------------------
# Integration-test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neo4j_connection_params():
    """Return Neo4j connection parameters from environment or defaults."""
    return {
        "uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.environ.get("NEO4J_USER", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password"),
    }


@pytest.fixture
def live_neo4j_driver(neo4j_connection_params):
    """Return a real Neo4j driver (skips if not available)."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            neo4j_connection_params["uri"],
            auth=(neo4j_connection_params["user"], neo4j_connection_params["password"]),
        )
        driver.verify_connectivity()
        yield driver
        driver.close()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")
