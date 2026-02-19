"""
Quick data-quality checks: verify Document nodes have paths/URLs and
sample MENTIONED_IN links are present.

Usage:
    python scripts/validate_data_quality.py
"""

import argparse
import os
import sys
from neo4j import GraphDatabase


def check_documents(session):
    q = """
    MATCH (d:Document)
    RETURN count(d) AS total,
           count(CASE WHEN d.url IS NOT NULL OR d.file_path IS NOT NULL
                           OR d.filename IS NOT NULL THEN 1 END) AS with_paths,
           count(CASE WHEN d.url IS NOT NULL THEN 1 END) AS with_url
    """
    return session.run(q).single().data()


def sample_mentions(session, limit=10):
    q = """
    MATCH (e)-[r:MENTIONED_IN]->(d:Document)
    RETURN labels(e)[0] AS type, e.name AS name,
           coalesce(d.url, d.file_path, d.filename, d.name) AS cite
    LIMIT $limit
    """
    return session.run(q, {"limit": limit}).data()


def main():
    parser = argparse.ArgumentParser(description="Validate data quality")
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
    with driver.session() as s:
        docs = check_documents(s)
        print("[docs]", docs)
        mentions = sample_mentions(s)
        for i, m in enumerate(mentions, 1):
            print(f"[{i}] {m['type']}: {m['name']} -> {m['cite']}")
    driver.close()


if __name__ == "__main__":
    main()
