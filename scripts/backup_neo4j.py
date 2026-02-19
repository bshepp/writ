"""
Neo4j Backup Script.

Exports all nodes and relationships as Cypher CREATE statements.
Optionally uploads to S3 with timestamp and retention policy.

Usage:
    python scripts/backup_neo4j.py --local-only
    python scripts/backup_neo4j.py --bucket my-backup-bucket
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BACKUP_RETENTION_DAYS = int(os.environ.get("BACKUP_RETENTION_DAYS", "7"))


def export_graph(neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> str:
    """Export all nodes and relationships as portable Cypher MERGE statements.

    Uses each node's ``id`` property (which has uniqueness constraints) for
    matching instead of Neo4j's internal ``id(n)`` — making the export safe
    to restore on a different Neo4j instance.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    lines = []

    with driver.session() as session:
        logger.info("Exporting nodes...")
        result = session.run(
            "MATCH (n) "
            "RETURN labels(n) AS labels, properties(n) AS props"
        )
        node_count = 0
        for record in result:
            labels_str = ":".join(record["labels"])
            props = record["props"]
            ent_id = props.get("id") or props.get("chunk_id") or props.get("name")
            if ent_id:
                id_key = "id" if "id" in props else ("chunk_id" if "chunk_id" in props else "name")
                escaped_id = str(ent_id).replace("\\", "\\\\").replace("'", "\\'")
                props_str = _props_to_cypher(props)
                lines.append(
                    f"MERGE (n:{labels_str} {{{id_key}: '{escaped_id}'}}) SET n += {props_str};"
                )
            else:
                props_str = _props_to_cypher(props)
                lines.append(f"CREATE (n:{labels_str} {props_str});")
            node_count += 1
        logger.info("  Exported %d nodes", node_count)

        logger.info("Exporting relationships...")
        result = session.run(
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a) AS a_labels, properties(a) AS a_props, "
            "       labels(b) AS b_labels, properties(b) AS b_props, "
            "       type(r) AS rtype, properties(r) AS r_props"
        )
        rel_count = 0
        for record in result:
            a_props, b_props = record["a_props"], record["b_props"]
            a_id = a_props.get("id") or a_props.get("chunk_id") or a_props.get("name")
            b_id = b_props.get("id") or b_props.get("chunk_id") or b_props.get("name")
            if not a_id or not b_id:
                continue
            a_key = "id" if "id" in a_props else ("chunk_id" if "chunk_id" in a_props else "name")
            b_key = "id" if "id" in b_props else ("chunk_id" if "chunk_id" in b_props else "name")
            a_esc = str(a_id).replace("\\", "\\\\").replace("'", "\\'")
            b_esc = str(b_id).replace("\\", "\\\\").replace("'", "\\'")
            rtype = record["rtype"]
            r_props = record["r_props"]
            props_str = " " + _props_to_cypher(r_props) if r_props else ""
            lines.append(
                f"MATCH (a {{{a_key}: '{a_esc}'}}), (b {{{b_key}: '{b_esc}'}}) "
                f"MERGE (a)-[:{rtype}{props_str}]->(b);"
            )
            rel_count += 1
        logger.info("  Exported %d relationships", rel_count)

    driver.close()

    header = (
        f"// Neo4j backup generated at {datetime.now(timezone.utc).isoformat()}\n"
        f"// Nodes: {node_count}, Relationships: {rel_count}\n"
        f"// Uses MERGE on entity id properties for portability\n\n"
    )
    return header + "\n".join(lines)


def _props_to_cypher(props: dict) -> str:
    if not props:
        return "{}"
    parts = []
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
            parts.append(f"{k}: '{escaped}'")
        elif isinstance(v, bool):
            parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        else:
            serialized = json.dumps(v, default=str)
            escaped = serialized.replace("\\", "\\\\").replace("'", "\\'")
            parts.append(f"{k}: '{escaped}'")
    return "{" + ", ".join(parts) + "}"


def save_local(cypher_content: str, output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    filepath = output_dir / f"neo4j-backup-{timestamp}.cypher"
    filepath.write_text(cypher_content, encoding="utf-8")
    logger.info("Saved local backup to %s (%d bytes)", filepath, len(cypher_content))
    return filepath


def upload_to_s3(cypher_content: str, bucket: str, prefix: str) -> str:
    import boto3
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    key = f"{prefix}neo4j-backup-{timestamp}.cypher"
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=cypher_content.encode("utf-8"), ContentType="text/plain")
    logger.info("Uploaded to s3://%s/%s", bucket, key)
    return key


def cleanup_old_backups(bucket: str, prefix: str, retention_days: int):
    import boto3
    s3 = boto3.client("s3")
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        return
    deleted = 0
    for obj in response["Contents"]:
        if obj["LastModified"].replace(tzinfo=timezone.utc) < cutoff:
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
            deleted += 1
    if deleted:
        logger.info("Cleaned up %d old backups", deleted)


def main():
    parser = argparse.ArgumentParser(description="Backup Neo4j graph")
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/backups"))
    parser.add_argument("--bucket", default=os.environ.get("BACKUP_S3_BUCKET", ""))
    parser.add_argument("--prefix", default=os.environ.get("BACKUP_S3_PREFIX", "backups/"))
    args = parser.parse_args()

    uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        logger.error("NEO4J_PASSWORD required")
        sys.exit(1)

    logger.info("Starting backup from %s", uri)
    cypher_content = export_graph(uri, user, password)

    if args.local_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_local(cypher_content, args.output_dir)
    elif args.bucket:
        upload_to_s3(cypher_content, args.bucket, args.prefix)
        cleanup_old_backups(args.bucket, args.prefix, BACKUP_RETENTION_DAYS)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_local(cypher_content, args.output_dir)

    logger.info("Backup complete!")


if __name__ == "__main__":
    main()
