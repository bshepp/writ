# Methodology: Writ Knowledge Graphs

This document describes a generalized process for building transparent AI knowledge graphs from regulatory documents. The methodology is domain-agnostic: you adapt entity types, relationship types, and prompts to your regulatory domain (EPA, OSHA, building codes, financial compliance, constitutional law, etc.).

---

## Phase 1: Domain Modeling

**Goal:** Define the semantic structure of your regulatory domain.

### 1.1 Entity Types

List the kinds of things you want to extract from documents. Typical regulatory entities:

| Entity Type | Description | Example |
|-------------|-------------|---------|
| Regulation | Governing rules, orders, statutes | NPDES permit, OSHA standard |
| Requirement | Mandatory compliance items | "Shall submit SWPPP within 7 days" |
| Procedure | Step-by-step compliance procedures | Inspection checklist |
| Personnel | Roles and qualifications | QSD, Site Inspector |
| Control | Best practices, technical measures | BMP, safety control |
| Timeline | Deadlines, reporting periods | 30-day notice, annual report |
| Document | Source reference | PDF name, section reference |
| Location | Geographic or site-specific scope | Watershed, facility |

**Your task:** Add domain-specific types. E.g., environmental: Permit, BMP, MonitoringRequirement; workplace: Hazard, PPE, Training.

### 1.2 Relationship Types

Define how entities connect:

| Relationship | Example |
|--------------|---------|
| REQUIRES | (Regulation)-[:REQUIRES]->(Requirement) |
| APPLIES_TO | (Requirement)-[:APPLIES_TO]->(Personnel) |
| IMPLEMENTS | (Requirement)-[:IMPLEMENTS]->(Control) |
| HAS_DEADLINE | (Requirement)-[:HAS_DEADLINE]->(Timeline) |
| MENTIONED_IN | (Entity)-[:MENTIONED_IN]->(Document) |
| REFERENCES | (Regulation)-[:REFERENCES]->(Document) |
| PERFORMED_BY | (Procedure)-[:PERFORMED_BY]->(Personnel) |
| SUPERSEDES | (Regulation)-[:SUPERSEDES]->(Regulation) |

### 1.3 Provenance

Every extracted entity should reference its source:
- Document name (e.g., CGP 2022, 29 CFR 1910)
- Section reference (e.g., Section IV.K.3, Attachment D)
- Page number (optional)

Store these as properties or as `Document` nodes with `MENTIONED_IN` relationships.

---

## Phase 2: Document Processing

**Goal:** Convert regulatory PDFs into text chunks suitable for LLM extraction.

### 2.1 Acquisition

- Collect authoritative PDFs (regulations, permits, guidance documents)
- Store in `resources/` (or equivalent)
- Prefer machine-readable PDFs; scanned documents need OCR

### 2.2 Text Extraction

**Options:**
1. **pypdf** — Fast, free; quality varies with PDF structure
2. **LlamaParse** — Higher quality for complex layouts; requires API key
3. **PyMuPDF / pdfplumber** — Good middle ground

### 2.3 Chunking

- **Chunk size:** 1000–2000 tokens (roughly 4000–8000 characters)
- **Overlap:** 100–300 tokens to preserve context across boundaries
- **Boundary rules:** Prefer splitting at paragraph or section boundaries

---

## Phase 3: Entity Extraction

**Goal:** Extract structured entities and relationships from text chunks.

### 3.1 Option A: Rule-Based

- Regex or pattern matching for known structures (section numbers, "shall" clauses, dates)
- Fast and deterministic; limited to predefined patterns
- Use when structure is highly consistent

### 3.2 Option B: LLM-Based (Recommended)

1. Define an extraction prompt with:
   - Entity types and their properties
   - Relationship types
   - Output format (JSON schema)
   - Rules (e.g., "Do not invent information not in the text")

2. Send each chunk to GPT-4o-mini (or equivalent) with structured output (JSON mode)

3. Parse responses, deduplicate entities (merge by name or ID), resolve relationships (match `from_name`/`to_name` to entity IDs)

4. Write output to `data/extracted_entities.json`

### 3.3 Deduplication

- Canonicalize names (lowercase, strip extra whitespace)
- Merge entities with same name and type; merge properties
- Use stable IDs (e.g., `type_name_hash`) for relationships

---

## Phase 4: Knowledge Graph Ingestion

**Goal:** Load entities and relationships into Neo4j.

### 4.1 Schema Creation

- Create nodes per entity type with labels (e.g., `:Requirement`, `:Personnel`)
- Create indexes on `name`, `id` for lookups
- Create relationships per relationship types

### 4.2 Ingestion Logic

1. For each entity: `MERGE` or `CREATE` node with properties
2. For each relationship: `MATCH` source and target by ID/name, then `CREATE` relationship
3. Link entities to `Document` nodes for provenance

### 4.3 Optional: Vector Embeddings (RAG)

- Chunk documents, embed with OpenAI (or similar)
- Store as `DocumentChunk` nodes with vector property
- Use Neo4j vector index for semantic search as fallback

---

## Phase 5: Query Strategy

**Goal:** Convert natural language questions into graph queries.

### 5.1 Priority Order

1. **AI-generated Cypher** — LLM (GPT-4o-mini) converts question → Cypher. Most flexible.
2. **Rule-based patterns** — Keyword match → predefined Cypher template. Fast, predictable.
3. **Full-text / vector search** — Fallback when no pattern matches and AI fails.

### 5.2 Security

- **Parameterisation:** Never interpolate user input into Cypher. Use parameters.
- **Read-only guard:** Block CREATE, DELETE, MERGE, SET, REMOVE, DROP, LOAD CSV, FOREACH before execution.
- **Length limit:** Cap query input (e.g., 500 chars).

Use the provided `cypher_guard.py` or equivalent in your query handler.

### 5.3 Query Result Format

Return:
- `question` — Original user question
- `cypher_query` — Generated or matched Cypher
- `results` — List of records
- `explanation` — Human-readable explanation
- `confidence` — 0.0–1.0 score
- `ai_summary` — Optional LLM-generated summary of results

---

## Phase 6: Interface

**Goal:** Expose the system to users.

### 6.1 Local Development

- HTTP server (e.g., Python `http.server` or Flask) serving:
  - Static frontend (HTML/CSS/JS)
  - POST `/api/query` — Accept `{"query": "..."}`, return structured result
  - GET `/api/health` — Health check
  - GET `/api/status` — Sanitised status (no internal URIs)

### 6.2 Production

- **Frontend:** S3 + CloudFront (static hosting)
- **Backend:** Lambda (or equivalent) + API Gateway
- **Database:** Neo4j on EC2 or Neo4j Aura

### 6.3 Deployment

- CI/CD (e.g., GitHub Actions) on push to main:
  - Run tests
  - Package and deploy Lambda
  - Sync frontend to S3, invalidate CloudFront cache

---

## Domain-Specific vs Reusable

| Component | Reusable | Domain-Specific |
|-----------|----------|-----------------|
| Entity extraction schema (structure) | ✅ | Entity types, properties |
| Cypher guard | ✅ | — |
| Query strategy (AI → rules → search) | ✅ | Rule patterns, templates |
| Document processing pipeline | ✅ | Chunk size, extraction method |
| Infrastructure (S3, Lambda, Neo4j) | ✅ | — |
| Entity types, relationship types | — | ✅ |
| Extraction prompts | — | ✅ |
| Sample questions for UI | — | ✅ |

---

## Reference Implementation

The **California Storm Water AI** project (`cal_storm_water`, sibling repo) implements this methodology for California Construction General Permit (CGP 2022) regulations. See the project README for location and structure.
