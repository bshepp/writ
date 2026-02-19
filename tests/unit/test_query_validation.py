"""Tests for query input validation in the Lambda handler."""

import json

import pytest


class TestQueryValidation:

    @pytest.mark.unit
    def test_missing_body(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "POST"}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert body["success"] is False
        assert "Missing request body" in body["error"]

    @pytest.mark.unit
    def test_missing_query_field(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "POST", "body": json.dumps({"text": "hello"})}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "query" in body["error"].lower()

    @pytest.mark.unit
    def test_empty_query(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "POST", "body": json.dumps({"query": "   "})}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "empty" in body["error"].lower()

    @pytest.mark.unit
    def test_query_too_long(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "POST", "body": json.dumps({"query": "a" * 501})}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "500" in body["error"]

    @pytest.mark.unit
    def test_invalid_json_body(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "POST", "body": "not valid json{"}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "JSON" in body["error"]

    @pytest.mark.unit
    def test_options_returns_200(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "OPTIONS"}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert resp["statusCode"] == 200

    @pytest.mark.unit
    def test_cors_headers_present(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "OPTIONS"}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        assert "Access-Control-Allow-Origin" in resp["headers"]
        assert "Access-Control-Allow-Methods" in resp["headers"]

    @pytest.mark.unit
    def test_query_at_max_length_accepted(self, query_handler_module, mock_lambda_context):
        event = {"httpMethod": "POST", "body": json.dumps({"query": "a" * 500})}
        resp = query_handler_module.lambda_handler(event, mock_lambda_context)
        if resp["statusCode"] == 400:
            body = json.loads(resp["body"])
            assert "500" not in body.get("error", ""), "500-char query should not be rejected"
