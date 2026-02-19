"""
Knowledge Graph Ingestion for Regulatory Entities.

Loads extracted_entities.json into Neo4j: creates nodes, relationships,
Document nodes, MENTIONED_IN links, uniqueness constraints, and full-text
indexes.

CUSTOMIZATION REQUIRED:
  1. Update VALID_LABELS to match your entity types
  2. Update VALID_REL_TYPES to match your relationship types
  3. Adjust full-text index labels in create_constraints()

Usage:
    python scripts/ingest_template.py
    python scripts/ingest_template.py --neo4j-uri bolt://localhost:7687
    python scripts/ingest_template.py --no-clear
"""

import json
import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

INPUT_PATH = PROJECT_ROOT / "data" / "extracted_entities.json"

# DOMAIN-SPECIFIC: Update these to match your entity types from schema.example.yaml
VALID_LABELS = {
    "Regulation", "Requirement", "Personnel", "Control",
    "Timeline", "Document",
}

VALID_REL_TYPES = {
    "REQUIRES", "APPLIES_TO", "PERFORMED_BY", "IMPLEMENTS",
    "HAS_DEADLINE", "MENTIONED_IN", "REFERENCES", "RELATED_TO",
}


def clear_graph(session):
    """Delete all nodes and relationships."""
    logger.info("Clearing existing graph data...")
    session.run("MATCH (n) DETACH DELETE n")
    logger.info("Graph cleared.")


def create_constraints(session):
    """Create uniqueness constraints and full-text indexes."""
    logger.info("Creating constraints and indexes...")
    for label in VALID_LABELS:
        try:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )
        except Exception as e:
            logger.warning("Constraint for %s: %s", label, e)

    try:
        session.run("DROP INDEX entityContentSearch IF EXISTS")
    except Exception:
        pass

    # DOMAIN-SPECIFIC: Adjust the labels in the full-text index to match yours.
    entity_labels = "|".join(l for l in VALID_LABELS if l != "Document")
    session.run(
        f"CREATE FULLTEXT INDEX entityContentSearch IF NOT EXISTS "
        f"FOR (n:{entity_labels}) "
        f"ON EACH [n.name, n.description]"
    )
    logger.info("Constraints and indexes created.")


def ingest_entities(session, entities: List[Dict]) -> int:
    """Create entity nodes."""
    count = 0
    for ent in entities:
        etype = ent.get("entity_type", "")
        if etype not in VALID_LABELS or etype == "Document":
            continue

        eid = ent.get("id", "")
        if not eid:
            continue

        props: Dict[str, Any] = {
            "id": eid,
            "name": ent.get("name", ""),
            "description": ent.get("description", ""),
            "section_reference": ent.get("section_reference"),
        }

        extra = ent.get("properties") or {}
        for k, v in extra.items():
            if v is None or k == "document_source":
                continue
            if isinstance(v, dict):
                props[k] = json.dumps(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                props[k] = json.dumps(v)
            else:
                props[k] = v

        doc_sources = ent.get("document_sources", [])
        if not doc_sources:
            ds = extra.get("document_source")
            if ds:
                doc_sources = [ds]
        props["document_sources"] = doc_sources

        query = f"MERGE (n:{etype} {{id: $id}}) SET n += $props"
        try:
            session.run(query, {"id": eid, "props": props})
            count += 1
        except Exception as e:
            logger.error("Failed to create %s %s: %s", etype, eid, e)

    return count


def ingest_documents(session, documents: List[Dict]) -> int:
    """Create Document nodes."""
    count = 0
    for doc in documents:
        doc_id = doc.get("document_id", "")
        if not doc_id:
            continue

        name = doc_id.replace("-", " ").replace("_", " ").title()
        props = {
            "id": doc_id,
            "name": name,
            "text_length": doc.get("text_length", 0),
            "chunks_processed": doc.get("chunks", 0),
            "raw_entities_extracted": doc.get("raw_entities", 0),
            "document_type": "Regulatory Document",
        }

        session.run(
            "MERGE (d:Document {id: $id}) SET d += $props",
            {"id": doc_id, "props": props},
        )
        count += 1

    return count


def create_mentioned_in(session, entities: List[Dict]) -> int:
    """Link entities to their source documents via MENTIONED_IN."""
    count = 0
    for ent in entities:
        eid = ent.get("id", "")
        sources = ent.get("document_sources", [])
        if not sources:
            ds = (ent.get("properties") or {}).get("document_source")
            if ds:
                sources = [ds]
        for doc_id in sources:
            try:
                result = session.run(
                    "MATCH (e {id: $eid}), (d:Document {id: $did}) "
                    "MERGE (e)-[:MENTIONED_IN]->(d) "
                    "RETURN count(*) AS c",
                    {"eid": eid, "did": doc_id},
                )
                if result.single()["c"] > 0:
                    count += 1
            except Exception:
                pass
    return count


def ingest_relationships(session, relationships: List[Dict], entities: List[Dict]) -> int:
    """Create relationships between entities using name-based matching."""
    name_to_id: Dict[str, str] = {}
    for ent in entities:
        key = (ent.get("name") or "").strip().lower()
        if key:
            name_to_id[key] = ent.get("id", "")

    count = 0
    skipped = 0

    for rel in relationships:
        from_name = (rel.get("from_name") or "").strip().lower()
        to_name = (rel.get("to_name") or "").strip().lower()
        rel_type = (rel.get("relationship") or "RELATED_TO").strip().upper().replace(" ", "_")

        if rel_type not in VALID_REL_TYPES:
            rel_type = "RELATED_TO"

        from_id = name_to_id.get(from_name)
        to_id = name_to_id.get(to_name)

        if not from_id or not to_id:
            skipped += 1
            continue

        try:
            result = session.run(
                f"MATCH (a {{id: $fid}}), (b {{id: $tid}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                "RETURN count(*) AS c",
                {"fid": from_id, "tid": to_id},
            )
            if result.single()["c"] > 0:
                count += 1
        except Exception as e:
            logger.error("Relationship error: %s", e)

    if skipped:
        logger.info("  Skipped %d relationships (unresolved names)", skipped)

    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest entities into Neo4j")
    parser.add_argument("--neo4j-uri", default=None, help="bolt://host:7687")
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--no-clear", action="store_true", help="Don't clear graph first")
    args = parser.parse_args()

    neo4j_uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")

    if not neo4j_password:
        logger.error("Neo4j password required. Set NEO4J_PASSWORD or use --neo4j-password")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    documents = data.get("documents", [])

    logger.info(
        "Loaded: %d entities, %d relationships, %d documents",
        len(entities), len(relationships), len(documents),
    )

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    driver.verify_connectivity()
    logger.info("Connected to Neo4j at %s", neo4j_uri)

    with driver.session() as session:
        if not args.no_clear:
            clear_graph(session)

        create_constraints(session)

        logger.info("Ingesting entities...")
        n_ent = ingest_entities(session, entities)
        logger.info("  Created %d entity nodes", n_ent)

        logger.info("Ingesting document nodes...")
        n_doc = ingest_documents(session, documents)
        logger.info("  Created %d document nodes", n_doc)

        logger.info("Creating MENTIONED_IN relationships...")
        n_mentioned = create_mentioned_in(session, entities)
        logger.info("  Created %d MENTIONED_IN links", n_mentioned)

        logger.info("Creating extracted relationships...")
        n_rel = ingest_relationships(session, relationships, entities)
        logger.info("  Created %d extracted relationships", n_rel)

        logger.info("--- INGESTION COMPLETE ---")
        for row in session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC"
        ).data():
            logger.info("  %-25s %d", row["label"], row["cnt"])

        total_rels = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
        logger.info("  Total relationships:    %d", total_rels)

    driver.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
