"""Tests for Lambda handler routing and response structure."""

import json
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_query_path = str(PROJECT_ROOT / "query")
if _query_path not in sys.path:
    sys.path.insert(0, _query_path)

from nlp_engine_template import QueryResult


class TestQueryHandler:

    @pytest.mark.unit
    def test_options_preflight(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "OPTIONS"}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 200
        assert resp["body"] == ""

    @pytest.mark.unit
    def test_successful_query_response_structure(self, query_handler_module, mock_lambda_context):
        mock_result = QueryResult(
            question="test question",
            cypher_query="MATCH (n) RETURN n",
            results=[{"name": "Test Result"}],
            explanation="test explanation",
            confidence=0.85,
            execution_time=1.5,
            ai_summary="Test summary",
            strategy="pattern_match",
        )

        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_result

        with patch.object(query_handler_module, "get_engine", return_value=mock_engine):
            event = {
                "httpMethod": "POST",
                "body": json.dumps({"query": "What are the main requirements?"}),
            }
            resp = query_handler_module.lambda_handler(event, mock_lambda_context)

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["success"] is True
        assert "result" in body
        assert "confidence" in body["result"]
        assert "ai_summary" in body["result"]
        assert body["result"]["results_count"] == 1
        assert body["result"]["strategy"] == "pattern_match"

    @pytest.mark.unit
    def test_internal_error_returns_500(self, query_handler_module, mock_lambda_context):
        mock_engine = MagicMock()
        mock_engine.query.side_effect = RuntimeError("database exploded")

        with patch.object(query_handler_module, "get_engine", return_value=mock_engine):
            event = {
                "httpMethod": "POST",
                "body": json.dumps({"query": "test"}),
            }
            resp = query_handler_module.lambda_handler(event, mock_lambda_context)

        assert resp["statusCode"] == 500
        body = json.loads(resp["body"])
        assert body["success"] is False
        assert "Internal server error" in body["error"]
        assert "database exploded" not in body["error"]
