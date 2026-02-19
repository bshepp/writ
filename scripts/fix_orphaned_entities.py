"""
Repair orphaned nodes in the knowledge graph.

Finds nodes with no relationships and optionally attaches them to a
placeholder Document node so they remain reachable.

Usage:
    python scripts/fix_orphaned_entities.py --list
    python scripts/fix_orphaned_entities.py --fix
    python scripts/fix_orphaned_entities.py --fix --placeholder-name "Unknown Document"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_orphaned_nodes(tx) -> List[Dict[str, object]]:
    """Return all nodes with no relationships."""
    query = """
    MATCH (n)
    WHERE COUNT { (n)--() } = 0
    RETURN id(n) AS id, labels(n) AS labels, n.name AS name
    """
    return [dict(record) for record in tx.run(query)]


def attach_orphans_to_placeholder(tx, placeholder_name: str) -> int:
    """Attach orphaned nodes to a placeholder Document node.

    Creates the placeholder Document if it does not already exist.
    Returns the number of nodes attached.
    """
    query = """
    MERGE (placeholder:Document {name: $placeholder_name})
      ON CREATE SET placeholder.description = 'Placeholder document for orphaned entities'
    WITH placeholder
    MATCH (n)
    WHERE COUNT { (n)--() } = 0
    MERGE (n)-[:MENTIONED_IN]->(placeholder)
    RETURN count(n) AS attached
    """
    result = tx.run(query, placeholder_name=placeholder_name).single()
    return result["attached"] if result else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix or list orphaned entities in Neo4j.")
    parser.add_argument("--list", action="store_true", help="List orphaned nodes.")
    parser.add_argument("--fix", action="store_true", help="Attach orphans to placeholder Document.")
    parser.add_argument("--placeholder-name", default="Unknown Document")
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    args = parser.parse_args()

    if not (args.list or args.fix):
        parser.error("Please specify either --list or --fix.")

    neo4j_uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")
    if not neo4j_password:
        logger.error("Neo4j password required. Set NEO4J_PASSWORD or use --neo4j-password")
        sys.exit(1)

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        if args.list:
            orphans = session.execute_read(get_orphaned_nodes)
            if not orphans:
                logger.info("No orphaned nodes found.")
            else:
                logger.info("Found %d orphaned nodes:", len(orphans))
                for node in orphans:
                    labels_str = ",".join(node["labels"]) if node["labels"] else "(no labels)"
                    name_str = node["name"] or "(no name)"
                    logger.info("  id=%s labels=%s name=%s", node["id"], labels_str, name_str)
        elif args.fix:
            attached = session.execute_write(
                attach_orphans_to_placeholder, args.placeholder_name
            )
            logger.info("Attached %d orphaned nodes to '%s'.", attached, args.placeholder_name)

    driver.close()


if __name__ == "__main__":
    main()
