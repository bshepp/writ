"""
Inspect the knowledge graph: list labels, relationship types, and sample paths.

Usage:
    python scripts/diagnostics/inspect_graph.py
    python scripts/diagnostics/inspect_graph.py --neo4j-uri bolt://host:7687
"""

import argparse
import os
import sys
from neo4j import GraphDatabase


def main():
    parser = argparse.ArgumentParser(description="Inspect Neo4j graph")
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
            labels = [r["label"] for r in s.run(
                "CALL db.labels() YIELD label RETURN label ORDER BY label"
            ).data()]
            rels = [r["relationshipType"] for r in s.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            ).data()]
            print(f"[info] Labels({len(labels)}): {labels}")
            print(f"[info] RelationshipTypes({len(rels)}): {rels}")

            rows = s.run(
                "MATCH p=(d:Document)-[r]-(n) "
                "RETURN d.name AS doc, type(r) AS rel, "
                "labels(n) AS nlabels, n.name AS nname LIMIT 10"
            ).data()
            for i, row in enumerate(rows, 1):
                print(f" {i}. {row['doc']} --{row['rel']}--> {row['nlabels']}:{row['nname']}")
    finally:
        driver.close()
    print("[done] inspection complete")


if __name__ == "__main__":
    main()
