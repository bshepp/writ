"""
Local development web server for Writ knowledge graph projects.

Serves the static frontend (index.html, styles.css) and provides API
endpoints that mirror the Lambda backend, so the same frontend works
locally and in production.

CUSTOMIZATION: Set your NLP engine class in ``run_server()``.

Usage:
    python server.py
    # Open http://localhost:8080
"""

import json
import os
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
_ALLOWED_ORIGINS: set = set()
if ALLOWED_ORIGIN and ALLOWED_ORIGIN != "*":
    _ALLOWED_ORIGINS.add(ALLOWED_ORIGIN)
    if ALLOWED_ORIGIN.startswith("https://www."):
        _ALLOWED_ORIGINS.add(ALLOWED_ORIGIN.replace("https://www.", "https://", 1))
    elif ALLOWED_ORIGIN.startswith("https://"):
        _ALLOWED_ORIGINS.add(ALLOWED_ORIGIN.replace("https://", "https://www.", 1))


def _cors_origin(request_origin: str = "") -> str:
    """Return the Access-Control-Allow-Origin value for a request."""
    if ALLOWED_ORIGIN == "*":
        return "*"
    if request_origin in _ALLOWED_ORIGINS:
        return request_origin
    return ALLOWED_ORIGIN


class LocalAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves static files and API endpoints."""

    engine: Any = None  # set once at server start

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/index.html"):
            self._serve_file("index.html", "text/html")
        elif path == "/styles.css":
            self._serve_file("styles.css", "text/css")
        elif path in ("/health", "/api/health"):
            self._handle_health()
        elif path in ("/status", "/api/status"):
            self._handle_status()
        elif path in ("/query", "/api/query"):
            query_params = urllib.parse.parse_qs(parsed.query)
            if "q" in query_params:
                self._process_query(query_params["q"][0].strip())
            else:
                self._json_error(400, "Missing query parameter q")
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/query", "/api/query"):
            self._handle_query_post()
        elif path in ("/query/stream", "/api/query/stream"):
            self._handle_query_stream_post()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self._send_cors_preflight()

    # ------------------------------------------------------------------ #
    # Static file serving
    # ------------------------------------------------------------------ #

    def _serve_file(self, filename: str, content_type: str):
        filepath = PROJECT_ROOT / filename
        if not filepath.exists():
            self.send_error(404, f"{filename} not found")
            return
        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------ #
    # API: /query
    # ------------------------------------------------------------------ #

    def _handle_query_post(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body = json.loads(raw) if raw else {}
            question = body.get("query", "").strip()
        except (json.JSONDecodeError, ValueError):
            self._json_error(400, "Invalid JSON")
            return

        if not question:
            self._json_error(400, "Missing or empty query")
            return
        if len(question) > 500:
            self._json_error(400, "Query exceeds 500 character limit")
            return

        self._process_query(question)

    def _process_query(self, question: str):
        try:
            result = self.engine.query(question)
            result_data = result.model_dump()

            results_list = []
            for res in (result_data.get("results") or [])[:10]:
                item: Dict[str, Any] = {}
                for key, value in res.items():
                    if hasattr(value, "labels") and hasattr(value, "items"):
                        for pk, pv in dict(value).items():
                            item[f"{key}.{pk}"] = pv
                    elif isinstance(value, list):
                        item[key] = [str(v) for v in value]
                    else:
                        item[key] = value
                results_list.append(item)
            result_data["results"] = results_list
            result_data["results_count"] = len(results_list)

            self._json_response(200, {
                "success": True,
                "query": question,
                "result": result_data,
            })

        except Exception as e:
            print(f"Query error: {e}")
            self._json_error(500, f"Query processing failed: {e}")

    # ------------------------------------------------------------------ #
    # API: /query/stream  (NDJSON streaming, local-dev only)
    # ------------------------------------------------------------------ #

    def _handle_query_stream_post(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body = json.loads(raw) if raw else {}
            question = body.get("query", "").strip()
        except (json.JSONDecodeError, ValueError):
            self._json_error(400, "Invalid JSON")
            return

        if not question:
            self._json_error(400, "Missing or empty query")
            return
        if len(question) > 500:
            self._json_error(400, "Query exceeds 500 character limit")
            return

        origin = _cors_origin(self.headers.get("Origin", ""))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()

        try:
            for step in self.engine.query_streaming(question):
                if step.get("type") == "result":
                    data = step["data"].model_dump()
                    data["results"] = data.get("results", [])[:10]
                    data["results_count"] = len(data["results"])
                    line = json.dumps({"type": "result", "data": data}, default=str)
                else:
                    line = json.dumps(step, default=str)
                chunk = f"{len(line) + 2:X}\r\n{line}\r\n\r\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()

            end_chunk = "0\r\n\r\n"
            self.wfile.write(end_chunk.encode())
            self.wfile.flush()
        except Exception as e:
            print(f"Stream error: {e}")
            err_line = json.dumps({"type": "error", "error": str(e)})
            chunk = f"{len(err_line) + 2:X}\r\n{err_line}\r\n\r\n"
            self.wfile.write(chunk.encode())
            self.wfile.write("0\r\n\r\n".encode())
            self.wfile.flush()

    # ------------------------------------------------------------------ #
    # API: /health
    # ------------------------------------------------------------------ #

    def _handle_health(self):
        db_connected = False
        total_nodes = total_rels = total_docs = 0

        try:
            if self.engine and self.engine.neo4j_driver:
                with self.engine.neo4j_driver.session() as session:
                    row = session.run(
                        "MATCH (n) WITH count(n) AS nodes "
                        "OPTIONAL MATCH ()-[r]->() WITH nodes, count(r) AS rels "
                        "OPTIONAL MATCH (d:Document) "
                        "RETURN nodes, rels, count(d) AS docs"
                    ).single()
                    if row:
                        db_connected = True
                        total_nodes = row["nodes"]
                        total_rels = row["rels"]
                        total_docs = row["docs"]
        except Exception as e:
            print(f"Health check DB error: {e}")

        self._json_response(200, {
            "status": "healthy" if db_connected else "degraded",
            "components": {
                "database": {
                    "connected": db_connected,
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "total_documents": total_docs,
                },
            },
        })

    # ------------------------------------------------------------------ #
    # API: /status
    # ------------------------------------------------------------------ #

    def _handle_status(self):
        self._json_response(200, {
            "status": "operational",
            "version": "1.0.0-local",
            "api": {"health": "available", "query": "available", "status": "available"},
            "environment": "local",
        })

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _json_response(self, status: int, data: Any):
        body = json.dumps(data, default=str).encode()
        origin = _cors_origin(self.headers.get("Origin", ""))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str):
        self._json_response(status, {"success": False, "error": message})

    def _send_cors_preflight(self):
        origin = _cors_origin(self.headers.get("Origin", ""))
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if len(args) >= 2 and "200" in str(args[1]):
            return
        super().log_message(format, *args)


def run_server(port: int = 8080):
    """Start the local development server.

    CUSTOMIZATION: Replace WritEngine with your subclass or
    import your domain-specific engine here.
    """
    from query.nlp_engine_template import WritEngine

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    LocalAPIHandler.engine = WritEngine(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_api_key=openai_api_key,
    )

    server = HTTPServer(("", port), LocalAPIHandler)
    print(f"Writ -- local dev server")
    print(f"  http://localhost:{port}")
    print(f"  CORS origin: {ALLOWED_ORIGIN}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", "8080"))
    run_server(port)
