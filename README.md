# Writ

Transparent AI knowledge graphs from regulatory documents. Extract entities from PDFs, load them into a graph database (Neo4j), and serve natural-language queries via a secure API -- with full visibility into how every answer was produced.

**Use cases:** EPA regulations, OSHA standards, building codes, environmental permits, financial compliance, healthcare rules, constitutional law -- any domain where structured regulatory text needs AI interpretation.

## Overview

This project provides:

1. **METHODOLOGY.md** -- Step-by-step process (domain modeling, document processing, entity extraction, graph ingestion, query layer, security, deployment)
2. **Working pipeline** -- Entity extraction, ingestion, vector embeddings, NLP query engine, local dev server, Lambda handler, CI/CD, and CloudFormation infrastructure
3. **Domain-agnostic** -- Clone this repo, customize the marked sections for your regulatory domain, and deploy

## Quick Start

1. **Clone or copy** this repository into a new project directory.

2. **Define your domain** -- edit `schema.example.yaml` with your entity types (e.g., Regulation, Requirement, Personnel, Control) and relationship types (REQUIRES, APPLIES_TO, PERFORMED_BY).

3. **Add source documents** -- place regulatory PDFs in `resources/`.

4. **Configure** -- copy `env.example` to `.env` and set your Neo4j and OpenAI credentials.

5. **Run the pipeline:**
   ```bash
   python scripts/entity_extractor_template.py   # Extract entities from PDFs
   python scripts/ingest_template.py              # Load into Neo4j
   python scripts/embed_documents.py              # Create vector embeddings for RAG
   python server.py                               # Start local dev server at :8080
   ```

## Reference Implementation

The [California Storm Water AI](../cal_storm_water) project (`cal_storm_water`, sibling repo) is a full implementation of this methodology for California Construction General Permit (CGP 2022) regulations.

## Files to Customize

These files contain `DOMAIN-SPECIFIC` or `CUSTOMIZATION` markers that you need to update:

| File | What to Change |
|------|---------------|
| `schema.example.yaml` | Define your entity types, relationships, and properties |
| `env.example` | Set your Neo4j URI, credentials, and OpenAI key |
| `config.example.py` | Update project name, paths, and schema constants |
| `scripts/entity_extractor_template.py` | Update `ENTITY_TYPES`, `RELATIONSHIP_TYPES`, and `EXTRACTION_PROMPT` |
| `scripts/ingest_template.py` | Update `VALID_LABELS` and `VALID_REL_TYPES` |
| `query/nlp_engine_template.py` | Add patterns in `_build_patterns()`, update schema in `_generate_cypher_with_ai()` |
| `lambda/query_handler/lambda_function.py` | Update `SECRET_PREFIX`, engine import |
| `infrastructure/stack-template.yaml` | Update `Description`, `DomainName`, `SecretPrefix` |
| `.github/workflows/deploy.yml` | Update `STACK_NAME` |

## Files to Use As-Is

These are domain-agnostic and typically require no changes:

| File | Purpose |
|------|---------|
| `lib/document_loader.py` | PDF text extraction and chunking |
| `query/cypher_guard.py` | Read-only Cypher query validation |
| `utils/rerank.py` | GPU cross-encoder result reranking |
| `scripts/fix_orphaned_entities.py` | Graph repair for orphaned nodes |
| `scripts/backup_neo4j.py` | Graph export and S3 backup |
| `scripts/embed_documents.py` | Vector embedding pipeline |
| `scripts/validate_data_quality.py` | Data quality checks |
| `scripts/diagnostics/*` | Connection and graph inspection |
| `tests/*` | Unit tests (Cypher guard, validation, Lambda) |
| `server.py` | Local development server |

## Pipeline Execution Order

```
1. Place PDFs in resources/

2. Extract entities:
   python scripts/entity_extractor_template.py
   -> outputs data/extracted_entities.json

3. Ingest into Neo4j:
   python scripts/ingest_template.py
   -> creates nodes, relationships, constraints, full-text index

4. Create vector embeddings (for RAG):
   python scripts/embed_documents.py
   -> creates DocumentChunk nodes with vector index

5. (Optional) Fix orphaned nodes:
   python scripts/fix_orphaned_entities.py --list
   python scripts/fix_orphaned_entities.py --fix

6. Run locally:
   python server.py
   -> http://localhost:8080

7. Deploy to AWS:
   - Create CloudFormation stack from infrastructure/stack-template.yaml
   - Set up GitHub secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
   - Push to main branch -> CI/CD deploys automatically
```

## Architecture Overview

```
 PDFs -> entity_extractor -> extracted_entities.json -> ingest -> Neo4j
                                                                   |
                          embed_documents -> DocumentChunk nodes ---+
                                                                   |
           User Question -> NLP Engine (4-strategy cascade) -------+
                               |                                   |
                          AI Summary <--- RAG context <-- vector search
                               |
                          JSON response
                               |
                    server.py (local) / Lambda (prod)
                               |
                     index.html frontend
```

## Query Strategy (Priority Order)

1. **Pattern matching** (confidence 0.85) -- keyword-based, fastest
2. **LangChain QA** (confidence 0.80) -- schema-aware Cypher generation
3. **Direct AI** (confidence 0.75) -- OpenAI Cypher generation fallback
4. **Full-text search** (confidence 0.50) -- broadest catch-all

## Project Structure

```
writ/
├── README.md                        # This file
├── METHODOLOGY.md                   # Detailed process (wiki-style)
├── requirements.txt                 # Python dependencies
├── .gitignore
├── config.example.py                # Configuration template
├── schema.example.yaml              # Entity/relationship schema
├── env.example                      # Environment variables
├── server.py                        # Local dev server (http://localhost:8080)
├── resources/                       # Place regulatory PDFs here
├── data/                            # Extraction output, intermediate files
├── lib/
│   └── document_loader.py           # Shared PDF reading and chunking
├── scripts/
│   ├── entity_extractor_template.py # LLM entity extraction
│   ├── ingest_template.py           # Neo4j ingestion
│   ├── embed_documents.py           # Vector embeddings for RAG
│   ├── fix_orphaned_entities.py     # Graph repair utility
│   ├── backup_neo4j.py              # Graph export + S3 upload
│   ├── validate_data_quality.py     # Quick quality checks
│   └── diagnostics/
│       ├── inspect_graph.py
│       └── test_neo4j_connection.py
├── query/
│   ├── cypher_guard.py              # Read-only Cypher validator
│   └── nlp_engine_template.py       # Multi-strategy NLP engine
├── lambda/
│   └── query_handler/
│       └── lambda_function.py       # AWS Lambda handler
├── utils/
│   └── rerank.py                    # GPU cross-encoder reranker
├── tests/
│   ├── conftest.py
│   └── unit/
│       ├── test_cypher_guard.py
│       ├── test_query_validation.py
│       └── test_lambda_handler.py
├── infrastructure/
│   └── stack-template.yaml          # CloudFormation (VPC, EC2, Lambda, API GW, S3, CF)
└── .github/
    └── workflows/
        └── deploy.yml               # CI/CD pipeline
```

## Environment Variables

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=sk-...
ALLOWED_ORIGIN=*              # or https://yourdomain.com for production
SECRET_PREFIX=regulatory-ai   # for AWS Secrets Manager
```

## Requirements

- Python 3.11+
- Neo4j 5.x (Docker recommended)
- OpenAI API key (for LLM extraction, Cypher generation, and embeddings)
- AWS account (for production deployment)
- Optional: `sentence-transformers` (for GPU reranking)
- Optional: LangChain (for schema-aware Cypher QA)
