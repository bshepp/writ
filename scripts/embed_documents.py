"""
Embed document chunks and store in Neo4j vector index for RAG retrieval.

Reads all PDFs, chunks them into ~500-token overlapping segments, embeds
each chunk with OpenAI text-embedding-3-small, and stores the embeddings
as DocumentChunk nodes in Neo4j with a vector index.

Usage:
    python scripts/embed_documents.py
    python scripts/embed_documents.py --limit 5
"""

import logging
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.document_loader import read_pdf_pages, chunk_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESOURCES_DIR = PROJECT_ROOT / "resources"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(client, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using OpenAI."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Neo4j storage
# ---------------------------------------------------------------------------

def setup_vector_index(session):
    """Create vector index and uniqueness constraint for document chunks."""
    try:
        session.run("DROP INDEX chunk_embeddings IF EXISTS")
    except Exception:
        pass

    session.run(
        "CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS "
        "FOR (c:DocumentChunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {EMBEDDING_DIM}, "
        "`vector.similarity_function`: 'cosine'"
        "}}"
    )
    logger.info("Vector index 'chunk_embeddings' created")

    try:
        session.run(
            "CREATE CONSTRAINT IF NOT EXISTS "
            "FOR (c:DocumentChunk) REQUIRE c.chunk_id IS UNIQUE"
        )
    except Exception:
        pass


def store_chunk(session, chunk: Dict, embedding: List[float]):
    """Store a single chunk with its embedding in Neo4j."""
    session.run(
        "MERGE (c:DocumentChunk {chunk_id: $chunk_id}) "
        "SET c.document_id = $document_id, "
        "    c.text = $text, "
        "    c.page_number = $page_number, "
        "    c.char_offset = $char_offset, "
        "    c.embedding = $embedding",
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "char_offset": chunk["char_offset"],
            "embedding": embedding,
        },
    )


def link_chunks_to_documents(session):
    """Link DocumentChunk nodes to their parent Document nodes."""
    result = session.run(
        "MATCH (c:DocumentChunk), (d:Document) "
        "WHERE c.document_id = d.id "
        "MERGE (c)-[:PART_OF]->(d) "
        "RETURN count(*) AS linked"
    )
    count = result.single()["linked"]
    logger.info("Linked %d chunks to document nodes", count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed document chunks for RAG")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N PDFs")
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--key", type=str, default=None, help="OpenAI API key")
    args = parser.parse_args()

    neo4j_uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")
    api_key = args.key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.error("No OpenAI API key. Set OPENAI_API_KEY or use --key")
        sys.exit(1)
    if not neo4j_password:
        logger.error("No Neo4j password. Set NEO4J_PASSWORD or use --neo4j-password")
        sys.exit(1)

    from openai import OpenAI
    from neo4j import GraphDatabase

    oai_client = OpenAI(api_key=api_key)
    neo4j_driver = GraphDatabase.driver(
        neo4j_uri, auth=(neo4j_user, neo4j_password)
    )
    neo4j_driver.verify_connectivity()
    logger.info("Connected to Neo4j")

    with neo4j_driver.session() as session:
        setup_vector_index(session)

    pdf_files = sorted(RESOURCES_DIR.glob("*.pdf"))
    if args.limit:
        pdf_files = pdf_files[: args.limit]

    all_chunks: List[Dict] = []
    for pi, pdf_path in enumerate(pdf_files):
        doc_id = pdf_path.stem
        logger.info("[%d/%d] Chunking %s", pi + 1, len(pdf_files), pdf_path.name)

        pages = read_pdf_pages(pdf_path)
        if not pages:
            continue

        chunks = chunk_document(pages, doc_id, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info("  %d chunks", len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total chunks: %d", len(all_chunks))
    logger.info("Embedding chunks (batch size %d)...", BATCH_SIZE)
    total_embedded = 0

    with neo4j_driver.session() as session:
        for batch_start in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
            texts = [c["text"] for c in batch]

            try:
                embeddings = embed_texts(oai_client, texts)
            except Exception as e:
                logger.error("Embedding batch failed: %s", e)
                time.sleep(5)
                try:
                    embeddings = embed_texts(oai_client, texts)
                except Exception:
                    logger.error("Retry failed, skipping batch")
                    continue

            for chunk, emb in zip(batch, embeddings):
                store_chunk(session, chunk, emb)

            total_embedded += len(batch)
            if total_embedded % 200 == 0 or total_embedded == len(all_chunks):
                logger.info("  Embedded %d / %d", total_embedded, len(all_chunks))

        link_chunks_to_documents(session)

    neo4j_driver.close()
    logger.info("Done. %d chunks embedded and stored.", total_embedded)


if __name__ == "__main__":
    main()
