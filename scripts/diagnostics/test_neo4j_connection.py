"""
Test Neo4j connectivity and report basic stats.

Usage:
    python scripts/diagnostics/test_neo4j_connection.py
    python scripts/diagnostics/test_neo4j_connection.py --neo4j-uri bolt://host:7687
"""

import argparse
import os
import sys
from neo4j import GraphDatabase


def main():
    parser = argparse.ArgumentParser(description="Test Neo4j connection")
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    args = parser.parse_args()

    uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        print("NEO4J_PASSWORD required")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as s:
            ping = s.run("RETURN 1 AS ok").single()
            count = s.run("MATCH (n) RETURN count(n) AS c").single()
            print(f"[ok] ping: {ping['ok']}; nodes: {count['c']}")
    finally:
        driver.close()
    print("[done] connection test complete")


if __name__ == "__main__":
    main()
