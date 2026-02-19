"""
LLM-based Entity Extraction for Regulatory Documents.

Reads PDFs from resources/, chunks them, sends each chunk to GPT-4o-mini
with a structured output schema, deduplicates entities, and writes
data/extracted_entities.json.

CUSTOMIZATION REQUIRED:
  1. Update ENTITY_TYPES and RELATIONSHIP_TYPES for your domain
  2. Customize EXTRACTION_PROMPT for your regulatory documents
  3. See schema.example.yaml for the full schema definition

Usage:
    python scripts/entity_extractor_template.py
    python scripts/entity_extractor_template.py --limit 5
    python scripts/entity_extractor_template.py --resume
"""

import json
import logging
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.document_loader import read_pdf, chunk_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config -- CUSTOMIZE THESE FOR YOUR DOMAIN
# ---------------------------------------------------------------------------

RESOURCES_DIR = PROJECT_ROOT / "resources"
OUTPUT_PATH = PROJECT_ROOT / "data" / "extracted_entities.json"

CHUNK_SIZE = 1500       # tokens (approx 4 chars/token)
CHUNK_OVERLAP = 200     # tokens

# DOMAIN-SPECIFIC: Update these to match schema.example.yaml
ENTITY_TYPES = [
    "Regulation",
    "Requirement",
    "Personnel",
    "Control",
    "Timeline",
]

RELATIONSHIP_TYPES = [
    "REQUIRES",
    "APPLIES_TO",
    "PERFORMED_BY",
    "IMPLEMENTS",
    "HAS_DEADLINE",
    "MENTIONED_IN",
]

# DOMAIN-SPECIFIC: Rewrite this prompt for your regulatory domain.
# The structure (entity types, rules, JSON format) should stay the same;
# replace the descriptions with your domain's specifics.
EXTRACTION_PROMPT = """You are an expert at reading regulatory documents and extracting structured knowledge graph entities.

Given a chunk of text from a regulatory document, extract all meaningful entities and relationships.

## Entity Types and What to Extract

- **Regulation**: Governing rules, permits, orders, statutes. Include identifiers, effective dates, applicability.
- **Requirement**: Mandatory compliance items (look for "shall", "must", "required to"). Include the full requirement text, what category it falls under (monitoring, reporting, planning, implementation, compliance, notification).
- **Personnel**: Roles, qualifications, responsibilities (inspector, certifier, responsible person, etc.).
- **Control**: Best practices, technical measures, safety controls. Include type, when required, maintenance requirements.
- **Timeline**: Deadlines, timeframes, reporting periods. Include duration, what triggers the deadline, and consequences.

## Rules

1. Write REAL descriptions (2-4 sentences) explaining what the entity IS, why it matters, and key details. Do NOT just restate the name.
2. Include section references when visible (e.g., "Section 4.3", "Appendix B").
3. Extract relationships between entities found in this chunk.
4. Be specific -- detailed names are better than generic ones.
5. Do not invent information not present in the text.
6. Consolidate related sentences about the same entity into one entity with a rich description.

## Output Format (JSON)

{
  "entities": [
    {
      "name": "string (concise, descriptive name)",
      "entity_type": "Regulation|Requirement|Personnel|Control|Timeline",
      "description": "string (2-4 sentences)",
      "section_reference": "string or null",
      "properties": {}
    }
  ],
  "relationships": [
    {
      "from_name": "string (must match an entity name above or a well-known entity)",
      "relationship": "REQUIRES|APPLIES_TO|PERFORMED_BY|IMPLEMENTS|HAS_DEADLINE|MENTIONED_IN",
      "to_name": "string (must match an entity name above or a well-known entity)"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def extract_from_chunk(
    client,
    chunk: str,
    document_id: str,
    chunk_index: int,
    total_chunks: int,
) -> Dict[str, Any]:
    """Send a single chunk to GPT-4o-mini and parse the structured response."""
    user_message = (
        f"Document: {document_id} (chunk {chunk_index + 1}/{total_chunks})\n\n"
        f"---BEGIN TEXT---\n{chunk}\n---END TEXT---"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000,
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            for ent in entities:
                if "properties" not in ent or ent["properties"] is None:
                    ent["properties"] = {}
                ent["properties"]["document_source"] = document_id

            return {"entities": entities, "relationships": relationships}

        except json.JSONDecodeError:
            logger.warning(
                "  Chunk %d: JSON parse failed (attempt %d), retrying...",
                chunk_index, attempt + 1,
            )
            time.sleep(1)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** (attempt + 1)
                logger.warning("  Rate limited, waiting %ds...", wait)
                time.sleep(wait)
            else:
                logger.error("  Chunk %d error: %s", chunk_index, e)
                break

    return {"entities": [], "relationships": []}


# ---------------------------------------------------------------------------
# Deduplication / merging
# ---------------------------------------------------------------------------

def _entity_key(ent: Dict) -> str:
    """Create a normalised key for deduplication."""
    name = ent.get("name", "").strip().lower()
    etype = ent.get("entity_type", "").strip()
    return f"{etype}::{name}"


def merge_entities(all_entities: List[Dict]) -> List[Dict]:
    """Deduplicate entities, keeping the richest description."""
    merged: Dict[str, Dict] = {}

    for ent in all_entities:
        key = _entity_key(ent)
        if not key or key == "::":
            continue

        if key not in merged:
            merged[key] = ent.copy()
            merged[key]["_sources"] = {
                ent.get("properties", {}).get("document_source", "")
            }
        else:
            existing = merged[key]
            new_desc = (ent.get("description") or "").strip()
            old_desc = (existing.get("description") or "").strip()
            if len(new_desc) > len(old_desc):
                existing["description"] = new_desc

            if not existing.get("section_reference") and ent.get("section_reference"):
                existing["section_reference"] = ent["section_reference"]

            new_props = ent.get("properties", {}) or {}
            old_props = existing.get("properties", {}) or {}
            for k, v in new_props.items():
                if v and not old_props.get(k):
                    old_props[k] = v
            existing["properties"] = old_props

            existing["_sources"].add(new_props.get("document_source", ""))

    results = []
    for ent in merged.values():
        sources = ent.pop("_sources", set())
        ent["document_sources"] = sorted(s for s in sources if s)
        results.append(ent)

    return results


def merge_relationships(all_rels: List[Dict], entity_names: set) -> List[Dict]:
    """Deduplicate relationships and validate endpoint names exist."""
    seen: set = set()
    merged: List[Dict] = []
    for rel in all_rels:
        fn = (rel.get("from_name") or "").strip()
        tn = (rel.get("to_name") or "").strip()
        rt = (rel.get("relationship") or "").strip()
        if not fn or not tn or not rt:
            continue
        key = f"{fn.lower()}|{rt}|{tn.lower()}"
        if key in seen:
            continue
        seen.add(key)
        merged.append({"from_name": fn, "relationship": rt, "to_name": tn})
    return merged


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def generate_id(entity: Dict, index: int) -> str:
    """Create a stable ID from entity type and name."""
    etype = (entity.get("entity_type") or "unknown").lower()
    name = (entity.get("name") or f"entity_{index}").strip()
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
    return f"{etype}_{slug}"


# ---------------------------------------------------------------------------
# Intermediate save / resume
# ---------------------------------------------------------------------------

def _save_intermediate(
    path: Path,
    raw_entities: List[Dict],
    raw_relationships: List[Dict],
    documents_meta: List[Dict],
    total_chunks: int,
):
    """Save intermediate extraction state for resume."""
    data = {
        "_raw_entities": raw_entities,
        "_raw_relationships": raw_relationships,
        "documents": documents_meta,
        "total_chunks": total_chunks,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info("  Intermediate results saved (%d entities so far)", len(raw_entities))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_all_pdfs(
    api_key: str,
    limit: Optional[int] = None,
    skip: int = 0,
    resume_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Process all PDFs and return the complete extraction result."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    pdf_files = sorted(RESOURCES_DIR.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]

    logger.info("Processing %d PDFs from %s", len(pdf_files), RESOURCES_DIR)

    all_entities: List[Dict] = []
    all_relationships: List[Dict] = []
    documents_meta: List[Dict] = []
    total_chunks_processed = 0

    if resume_file and resume_file.exists():
        with open(resume_file, "r", encoding="utf-8") as f:
            partial = json.load(f)
        all_entities = partial.get("_raw_entities", [])
        all_relationships = partial.get("_raw_relationships", [])
        documents_meta = partial.get("documents", [])
        total_chunks_processed = partial.get("total_chunks", 0)
        logger.info(
            "Resumed: %d entities, %d rels from %d docs",
            len(all_entities), len(all_relationships), len(documents_meta),
        )

    intermediate_path = OUTPUT_PATH.parent / "extraction_intermediate.json"

    for pdf_idx, pdf_path in enumerate(pdf_files):
        if pdf_idx < skip:
            continue

        doc_id = pdf_path.stem
        logger.info(
            "[%d/%d] Processing %s ...",
            pdf_idx + 1, len(pdf_files), pdf_path.name,
        )

        text = read_pdf(pdf_path)
        if not text.strip():
            logger.warning("  Empty text, skipping")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info("  %d chars -> %d chunks", len(text), len(chunks))

        doc_entities: List[Dict] = []
        doc_relationships: List[Dict] = []

        for ci, chunk in enumerate(chunks):
            if len(chunk.strip()) < 100:
                continue
            result = extract_from_chunk(client, chunk, doc_id, ci, len(chunks))
            doc_entities.extend(result["entities"])
            doc_relationships.extend(result["relationships"])
            total_chunks_processed += 1

            if total_chunks_processed % 20 == 0:
                time.sleep(0.5)

        all_entities.extend(doc_entities)
        all_relationships.extend(doc_relationships)

        documents_meta.append({
            "document_id": doc_id,
            "document_path": str(pdf_path),
            "text_length": len(text),
            "chunks": len(chunks),
            "raw_entities": len(doc_entities),
            "raw_relationships": len(doc_relationships),
        })

        logger.info(
            "  -> %d entities, %d relationships extracted",
            len(doc_entities), len(doc_relationships),
        )

        if (pdf_idx + 1) % 5 == 0:
            _save_intermediate(
                intermediate_path, all_entities, all_relationships,
                documents_meta, total_chunks_processed,
            )

    logger.info(
        "Merging: %d raw entities, %d raw relationships",
        len(all_entities), len(all_relationships),
    )
    merged_entities = merge_entities(all_entities)

    for i, ent in enumerate(merged_entities):
        ent["id"] = generate_id(ent, i)

    entity_names = {e["name"].strip().lower() for e in merged_entities}
    merged_rels = merge_relationships(all_relationships, entity_names)

    logger.info(
        "Final: %d unique entities, %d unique relationships",
        len(merged_entities), len(merged_rels),
    )

    return {
        "version": "2.0",
        "extraction_model": "gpt-4o-mini",
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "documents_processed": len(documents_meta),
        "total_chunks": total_chunks_processed,
        "entity_count": len(merged_entities),
        "relationship_count": len(merged_rels),
        "documents": documents_meta,
        "entities": merged_entities,
        "relationships": merged_rels,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Entity Extraction")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N PDFs")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N PDFs (for resume)")
    parser.add_argument("--resume", action="store_true", help="Resume from intermediate file")
    parser.add_argument("--key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key. Set OPENAI_API_KEY or use --key")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    resume_file = None
    if args.resume:
        resume_file = OUTPUT_PATH.parent / "extraction_intermediate.json"
        if not resume_file.exists():
            logger.warning("No intermediate file found, starting fresh")
            resume_file = None

    result = process_all_pdfs(
        api_key, limit=args.limit, skip=args.skip, resume_file=resume_file,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Output written to %s", OUTPUT_PATH)

    type_counts: Dict[str, int] = {}
    for ent in result["entities"]:
        t = ent.get("entity_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info("Entity type breakdown:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-25s %d", t, c)
    logger.info("Relationships: %d", result["relationship_count"])


if __name__ == "__main__":
    main()
