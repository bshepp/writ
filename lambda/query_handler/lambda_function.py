"""
AWS Lambda Function: AI Query Handler (Template)

Processes natural language queries against a Writ knowledge graph.

CUSTOMIZATION:
  - Update SECRET_PREFIX to match your AWS Secrets Manager secret names
  - Update the engine import in get_engine()
"""

import json
import os
import logging
import time
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Structured JSON logging for CloudWatch Logs Insights
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "function": record.funcName,
        }
        for key in ("request_id", "query", "duration_ms", "error_type",
                     "confidence", "results_count", "method"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.handlers:
    for handler in root_logger.handlers:
        handler.setFormatter(JsonFormatter())

logger = logging.getLogger("query_handler")

# ---------------------------------------------------------------------------
# Module-level cache & constants
# ---------------------------------------------------------------------------

_engine_cache = None
secrets_client = boto3.client("secretsmanager")

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
MAX_QUERY_LENGTH = 500

# DOMAIN-SPECIFIC: Change this prefix to match your secrets
SECRET_PREFIX = os.environ.get("SECRET_PREFIX", "regulatory-ai")

_ALLOWED_ORIGINS: set = set()
if ALLOWED_ORIGIN and ALLOWED_ORIGIN != "*":
    _ALLOWED_ORIGINS.add(ALLOWED_ORIGIN)
    if ALLOWED_ORIGIN.startswith("https://www."):
        _ALLOWED_ORIGINS.add(ALLOWED_ORIGIN.replace("https://www.", "https://", 1))
    elif ALLOWED_ORIGIN.startswith("https://"):
        _ALLOWED_ORIGINS.add(ALLOWED_ORIGIN.replace("https://", "https://www.", 1))


def cors_headers(request_origin: str = "") -> Dict[str, str]:
    """Return CORS headers, dynamically matching the request Origin."""
    if ALLOWED_ORIGIN == "*":
        origin = "*"
    elif request_origin in _ALLOWED_ORIGINS:
        origin = request_origin
    else:
        origin = ALLOWED_ORIGIN
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }


def get_secret(secret_name: str) -> str:
    """Retrieve a secret value from AWS Secrets Manager."""
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        return response["SecretString"]
    except ClientError as e:
        logger.error("Failed to retrieve secret %s: %s", secret_name, e,
                      extra={"error_type": "secrets_manager"})
        raise


def _parse_secret(raw: str, json_key: str) -> str:
    """Return a plain string secret, or extract json_key if value is JSON."""
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw)[json_key]
        except (json.JSONDecodeError, KeyError):
            pass
    return raw


def get_engine():
    """Return (and cache) the NLP engine singleton.

    CUSTOMIZATION: Change the import and class name to match your engine.
    """
    global _engine_cache
    if _engine_cache is not None:
        return _engine_cache

    from nlp_engine_template import WritEngine

    neo4j_password = _parse_secret(
        get_secret(f"{SECRET_PREFIX}/neo4j-password"), "password"
    )
    openai_api_key = _parse_secret(
        get_secret(f"{SECRET_PREFIX}/openai-api-key"), "key"
    )

    _engine_cache = WritEngine(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://neo4j-instance:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
    )
    logger.info("AI engine initialised and cached")
    return _engine_cache


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda entry-point. Expected POST body: {"query": "..."}"""
    request_id = getattr(context, "aws_request_id", "local")
    start = time.time()

    request_headers = event.get("headers") or {}
    request_origin = request_headers.get("origin") or request_headers.get("Origin") or ""
    headers = cors_headers(request_origin)

    if event.get("httpMethod") == "OPTIONS":
        logger.info("CORS preflight", extra={"request_id": request_id, "method": "OPTIONS"})
        return {"statusCode": 200, "headers": headers, "body": ""}

    try:
        if "body" not in event:
            raise ValueError("Missing request body")

        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]

        if "query" not in body:
            raise ValueError("Missing 'query' field in request body")

        query = body["query"].strip()
        if not query:
            raise ValueError("Empty query provided")

        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds {MAX_QUERY_LENGTH} character limit")

        logger.info("Processing query (%d chars)", len(query),
                     extra={"request_id": request_id, "query": query[:200], "method": "POST"})

        engine = get_engine()
        result = engine.query(query)

        duration_ms = round((time.time() - start) * 1000)

        result_data = result.model_dump()
        result_data["results"] = result_data["results"][:10]
        result_data["results_count"] = len(result_data["results"])

        response_data = {
            "success": True,
            "query": query,
            "result": result_data,
        }

        logger.info(
            "Query OK",
            extra={
                "request_id": request_id,
                "confidence": result.confidence,
                "results_count": len(result.results),
                "duration_ms": duration_ms,
            },
        )

        return {
            "statusCode": 200,
            "headers": {**headers, "Content-Type": "application/json"},
            "body": json.dumps(response_data),
        }

    except json.JSONDecodeError:
        logger.warning("JSON parse error",
                        extra={"request_id": request_id, "error_type": "json_parse"})
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"success": False, "error": "Invalid JSON in request body"}),
        }

    except ValueError as exc:
        logger.warning("Validation error: %s", exc,
                        extra={"request_id": request_id, "error_type": "validation"})
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"success": False, "error": str(exc)}),
        }

    except Exception:
        duration_ms = round((time.time() - start) * 1000)
        logger.error("Unexpected error",
                      extra={"request_id": request_id, "error_type": "internal",
                             "duration_ms": duration_ms},
                      exc_info=True)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"success": False, "error": "Internal server error"}),
        }
