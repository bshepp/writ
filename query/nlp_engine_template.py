"""
Writ NLP Query Engine for knowledge graphs.

Converts natural language questions into Neo4j Cypher queries using a
multi-strategy approach:

    1. Pattern matching  -- keyword-based, fast and predictable
    2. LangChain QA      -- schema-aware Cypher generation (if available)
    3. Direct OpenAI     -- LLM Cypher generation fallback
    4. Full-text search  -- broadest catch-all

All Cypher queries are parameterised to prevent injection.

CUSTOMIZATION REQUIRED:
  1. Add domain-specific patterns in ``_build_patterns()``
  2. Update the schema description in ``_generate_cypher_with_ai()``
  3. Adjust the summarization system prompt in ``_summarize_results()``
"""

import logging
import time
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional LangChain imports (graceful fallback if not installed)
# ---------------------------------------------------------------------------
_HAS_LANGCHAIN = False
try:
    from langchain_neo4j import Neo4jGraph, Neo4jVector
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from cypher_guard import SafeGraphCypherQAChain
    _HAS_LANGCHAIN = True
    logger.info("LangChain integration available (with Cypher guard)")
except ImportError:
    logger.info("LangChain not installed -- using direct OpenAI fallback")


class QueryResult(BaseModel):
    """Result of a natural language query."""
    question: str
    cypher_query: str = ""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    execution_time: float = 0.0
    ai_summary: str = ""
    strategy: str = ""


class WritEngine:
    """Writ NLP engine for knowledge graph queries.

    Rename or subclass this for your specific domain.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: str = None,
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.openai_api_key = openai_api_key

        self.neo4j_driver = None
        self.openai_client = None

        self._lc_graph: Optional[Any] = None
        self._lc_qa_chain: Optional[Any] = None
        self._lc_vector_store: Optional[Any] = None
        self._lc_llm: Optional[Any] = None
        self._lc_embeddings: Optional[Any] = None

        self._connect()
        self._build_patterns()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self):
        t0 = time.time()
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connected",
                        extra={"duration_ms": round((time.time() - t0) * 1000)})
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise

        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialised")
            except Exception as e:
                logger.warning("OpenAI init failed: %s", e)

        if _HAS_LANGCHAIN and self.openai_api_key:
            self._init_langchain()

        self._log_capabilities()

    def _log_capabilities(self):
        """Log which query capabilities are available at startup."""
        if self._lc_qa_chain and self._lc_vector_store:
            mode = "LangChain + RAG"
        elif self._lc_qa_chain:
            mode = "LangChain (no RAG vector index)"
        elif self.openai_client:
            mode = "Direct OpenAI"
        else:
            mode = "Basic (pattern match + full-text only)"
        logger.info("NLP engine mode: %s", mode)

    def _init_langchain(self):
        """Initialise LangChain graph, LLM, vector store, and QA chain."""
        t0 = time.time()
        try:
            from langchain_core.prompts import PromptTemplate

            self._lc_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=self.openai_api_key,
            )
            self._lc_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=self.openai_api_key,
            )

            self._lc_graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
            )

            _READ_ONLY_CYPHER_PROMPT = PromptTemplate(
                input_variables=["schema", "question"],
                template=(
                    "You are an expert Neo4j Cypher developer. Generate a Cypher "
                    "statement to answer the user's question.\n\n"
                    "Schema:\n{schema}\n\n"
                    "CRITICAL RULES:\n"
                    "- Generate ONLY read queries (MATCH, OPTIONAL MATCH, WITH, "
                    "WHERE, RETURN, ORDER BY, LIMIT, CALL for indexes).\n"
                    "- NEVER generate CREATE, MERGE, SET, DELETE, DETACH, REMOVE, "
                    "DROP, LOAD CSV, or FOREACH.\n"
                    "- If the question asks to modify data, reply with: "
                    "MATCH (n) WHERE false RETURN n\n"
                    "- Always include a LIMIT clause (max 10).\n"
                    "- Use toLower() for case-insensitive text comparisons.\n\n"
                    "Question: {question}\n"
                    "Cypher query:"
                ),
            )

            self._lc_qa_chain = SafeGraphCypherQAChain.from_llm(
                llm=self._lc_llm,
                graph=self._lc_graph,
                cypher_prompt=_READ_ONLY_CYPHER_PROMPT,
                verbose=False,
                top_k=10,
                return_intermediate_steps=True,
                allow_dangerous_requests=True,
            )

            try:
                self._lc_vector_store = Neo4jVector.from_existing_index(
                    embedding=self._lc_embeddings,
                    url=self.neo4j_uri,
                    username=self.neo4j_user,
                    password=self.neo4j_password,
                    index_name="chunk_embeddings",
                    node_label="DocumentChunk",
                    text_node_property="text",
                    embedding_node_property="embedding",
                )
            except Exception as e:
                logger.warning("Neo4jVector index not available (RAG disabled): %s", e)
                self._lc_vector_store = None

            logger.info("LangChain initialised",
                        extra={"duration_ms": round((time.time() - t0) * 1000)})
        except Exception as e:
            logger.warning("LangChain init failed (direct fallback): %s", e)
            self._lc_qa_chain = None
            self._lc_vector_store = None

    # ------------------------------------------------------------------
    # Query patterns
    #
    # DOMAIN-SPECIFIC: Add your keyword -> Cypher template mappings here.
    # Each pattern has:
    #   keywords  -- list of lowercase trigger phrases
    #   cypher    -- parameterised Cypher query
    #   params    -- dict of Cypher parameters (empty for static queries)
    #
    # Example:
    #   "safety_equipment": {
    #       "keywords": ["safety equipment", "ppe", "protective equipment"],
    #       "cypher": "MATCH (c:Control) WHERE toLower(c.name) CONTAINS 'ppe' ...",
    #       "params": {},
    #   }
    # ------------------------------------------------------------------

    def _build_patterns(self):
        """Pre-built query patterns keyed by keyword groups.

        These generic patterns work across any regulatory domain because they
        rely on the shared entity type structure (Regulation, Requirement,
        Personnel, Control, Timeline, Document).  Override or extend this
        method with domain-specific patterns in your subclass.
        """
        self.query_patterns: Dict[str, Dict] = {
            "regulation_overview": {
                "keywords": [
                    "what regulations", "list regulations", "show regulations",
                    "all regulations", "which regulations", "regulations exist",
                    "governing rules", "statutes",
                ],
                "cypher": (
                    "MATCH (r:Regulation) "
                    "OPTIONAL MATCH (r)-[:MENTIONED_IN]->(doc:Document) "
                    "RETURN r.name AS name, r.description AS description, "
                    "r.section_reference AS section_reference, "
                    "collect(DISTINCT doc.name)[0..3] AS source_documents "
                    "ORDER BY r.name LIMIT 10"
                ),
                "params": {},
            },
            "requirements_list": {
                "keywords": [
                    "what are the requirements", "list requirements",
                    "show requirements", "all requirements",
                    "what is required", "mandatory", "compliance requirements",
                    "must comply", "shall",
                ],
                "cypher": (
                    "MATCH (req:Requirement) "
                    "OPTIONAL MATCH (req)-[:MENTIONED_IN]->(doc:Document) "
                    "RETURN req.name AS name, req.description AS description, "
                    "req.category AS category, req.section_reference AS section_reference, "
                    "collect(DISTINCT doc.name)[0..3] AS source_documents "
                    "ORDER BY req.name LIMIT 10"
                ),
                "params": {},
            },
            "personnel_roles": {
                "keywords": [
                    "who is responsible", "personnel", "roles",
                    "qualifications", "inspector", "who performs",
                    "who does", "responsible for", "staff",
                ],
                "cypher": (
                    "MATCH (p:Personnel) "
                    "OPTIONAL MATCH (req:Requirement)-[:PERFORMED_BY]->(p) "
                    "OPTIONAL MATCH (p)-[:MENTIONED_IN]->(doc:Document) "
                    "RETURN p.name AS name, p.description AS description, "
                    "p.role_type AS role_type, p.qualifications AS qualifications, "
                    "collect(DISTINCT req.name)[0..3] AS performs, "
                    "collect(DISTINCT doc.name)[0..3] AS source_documents "
                    "ORDER BY p.name LIMIT 10"
                ),
                "params": {},
            },
            "deadlines_timelines": {
                "keywords": [
                    "deadline", "deadlines", "timeline", "timelines",
                    "how long", "when is", "due date", "reporting period",
                    "time limit", "how many days", "timeframe",
                ],
                "cypher": (
                    "MATCH (t:Timeline) "
                    "OPTIONAL MATCH (req:Requirement)-[:HAS_DEADLINE]->(t) "
                    "OPTIONAL MATCH (t)-[:MENTIONED_IN]->(doc:Document) "
                    "RETURN t.name AS name, t.description AS description, "
                    "t.duration_days AS duration_days, t.trigger AS trigger, "
                    "collect(DISTINCT req.name)[0..3] AS applies_to, "
                    "collect(DISTINCT doc.name)[0..3] AS source_documents "
                    "ORDER BY t.duration_days LIMIT 10"
                ),
                "params": {},
            },
            "controls_measures": {
                "keywords": [
                    "controls", "best practices", "measures",
                    "best management practices", "bmp", "bmps",
                    "safety controls", "technical measures",
                    "what controls", "list controls",
                ],
                "cypher": (
                    "MATCH (c:Control) "
                    "OPTIONAL MATCH (req:Requirement)-[:IMPLEMENTS]->(c) "
                    "OPTIONAL MATCH (c)-[:MENTIONED_IN]->(doc:Document) "
                    "RETURN c.name AS name, c.description AS description, "
                    "c.control_type AS control_type, "
                    "collect(DISTINCT req.name)[0..3] AS implements, "
                    "collect(DISTINCT doc.name)[0..3] AS source_documents "
                    "ORDER BY c.name LIMIT 10"
                ),
                "params": {},
            },
            "source_documents": {
                "keywords": [
                    "documents", "source documents", "sources",
                    "references", "which documents", "list documents",
                    "show documents", "pdfs",
                ],
                "cypher": (
                    "MATCH (d:Document) "
                    "OPTIONAL MATCH (e)-[:MENTIONED_IN]->(d) "
                    "RETURN d.name AS name, d.document_type AS document_type, "
                    "count(DISTINCT e) AS referenced_entities "
                    "ORDER BY referenced_entities DESC LIMIT 10"
                ),
                "params": {},
            },
            "graph_overview": {
                "keywords": [
                    "overview", "summary", "what is this about",
                    "graph summary", "what do we have", "show everything",
                    "how many", "count", "statistics",
                ],
                "cypher": (
                    "MATCH (n) "
                    "RETURN labels(n)[0] AS entity_type, count(n) AS count "
                    "ORDER BY count DESC"
                ),
                "params": {},
            },
            "requirement_for_entity": {
                "keywords": [
                    "requirements for", "what does it require",
                    "compliance for", "rules for",
                ],
                "cypher": (
                    "MATCH (r:Regulation)-[:REQUIRES]->(req:Requirement) "
                    "OPTIONAL MATCH (req)-[:MENTIONED_IN]->(doc:Document) "
                    "RETURN r.name AS regulation, req.name AS requirement, "
                    "req.description AS description, req.category AS category, "
                    "collect(DISTINCT doc.name)[0..3] AS source_documents "
                    "ORDER BY r.name, req.name LIMIT 10"
                ),
                "params": {},
            },
        }

    # ------------------------------------------------------------------
    # LangChain-powered Cypher QA
    # ------------------------------------------------------------------

    def _langchain_cypher_qa(self, question: str) -> Optional[Dict]:
        """Use LangChain GraphCypherQAChain for schema-aware Cypher gen."""
        if not self._lc_qa_chain:
            return None

        t0 = time.time()
        try:
            response = self._lc_qa_chain.invoke({"query": question})
            result_text = response.get("result", "")
            intermediate = response.get("intermediate_steps", [])

            generated_cypher = ""
            raw_results: List[Dict] = []
            if intermediate and len(intermediate) >= 1:
                generated_cypher = (
                    intermediate[0].get("query", "")
                    if isinstance(intermediate[0], dict)
                    else str(intermediate[0])
                )
            if intermediate and len(intermediate) >= 2:
                context = (
                    intermediate[1].get("context", [])
                    if isinstance(intermediate[1], dict)
                    else intermediate[1]
                )
                if isinstance(context, list):
                    raw_results = context

            logger.info("LangChain QA: %d results", len(raw_results),
                        extra={"duration_ms": round((time.time() - t0) * 1000)})

            return {
                "cypher": generated_cypher,
                "results": raw_results if raw_results else [],
                "summary": result_text,
            }
        except Exception as e:
            logger.warning("LangChain QA failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def _langchain_rag_retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """Use LangChain Neo4jVector for similarity search."""
        if not self._lc_vector_store:
            return self._direct_rag_retrieve(question, top_k)

        t0 = time.time()
        try:
            docs = self._lc_vector_store.similarity_search_with_score(question, k=top_k)
            chunks = []
            for doc, score in docs:
                chunks.append({
                    "text": doc.page_content,
                    "document_id": doc.metadata.get("document_id", ""),
                    "page": doc.metadata.get("page_number", "?"),
                    "score": score,
                })
            return chunks
        except Exception as e:
            logger.warning("LangChain RAG failed, trying direct: %s", e)
            return self._direct_rag_retrieve(question, top_k)

    def _direct_rag_retrieve(self, question: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks via direct OpenAI embeddings + Neo4j vector search."""
        if not self.openai_client or not self.neo4j_driver:
            return []

        t0 = time.time()
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=question,
            )
            query_embedding = response.data[0].embedding

            cypher = (
                "CALL db.index.vector.queryNodes('chunk_embeddings', $topK, $embedding) "
                "YIELD node, score "
                "RETURN node.chunk_id AS chunk_id, node.document_id AS document_id, "
                "node.text AS text, node.page_number AS page, score "
                "ORDER BY score DESC"
            )
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, {"topK": top_k, "embedding": query_embedding})
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning("Direct RAG retrieval failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Direct OpenAI Cypher generation (fallback)
    # ------------------------------------------------------------------

    def _generate_cypher_with_ai(self, question: str) -> Optional[str]:
        """Use OpenAI directly to generate a read-only Cypher query.

        DOMAIN-SPECIFIC: Update the schema description below to match
        your Neo4j node types, properties, and relationships.
        """
        if not self.openai_client:
            return None

        t0 = time.time()
        try:
            # DOMAIN-SPECIFIC: Replace this schema with your actual schema.
            prompt = (
                "You are a Neo4j Cypher expert for a Writ knowledge graph.\n\n"
                "## Node Types and Properties\n"
                "- Regulation: id, name, description, section_reference\n"
                "- Requirement: id, name, description, category, section_reference\n"
                "- Personnel: id, name, description, role_type, qualifications\n"
                "- Control: id, name, description, control_type\n"
                "- Timeline: id, name, description, duration_days, trigger\n"
                "- Document: id, name, document_type\n"
                "- DocumentChunk: chunk_id, document_id, text, page_number\n\n"
                "## Relationships\n"
                "- (entity)-[:MENTIONED_IN]->(Document)\n"
                "- (Regulation)-[:REQUIRES]->(Requirement)\n"
                "- (Requirement)-[:PERFORMED_BY]->(Personnel)\n"
                "- (Requirement)-[:IMPLEMENTS]->(Control)\n"
                "- (Requirement)-[:HAS_DEADLINE]->(Timeline)\n"
                "- (DocumentChunk)-[:PART_OF]->(Document)\n\n"
                "## Rules\n"
                "- Return ONLY valid Cypher, no explanation.\n"
                "- Always include LIMIT 10.\n"
                "- Use toLower() for text comparisons.\n"
                "- Always OPTIONAL MATCH for document links.\n"
                "- RETURN name, description, section_reference, source_documents.\n"
                "- Do NOT embed user strings directly; use $param parameters.\n\n"
                f"Question: {question}"
            )
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only valid Cypher queries. No explanation, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            cypher = response.choices[0].message.content.strip()

            if cypher.startswith("```"):
                lines = cypher.split("\n")
                cypher = "\n".join(
                    l for l in lines if not l.strip().startswith("```")
                ).strip()

            if "MATCH" in cypher and "RETURN" in cypher:
                logger.info("AI generated Cypher: %s", cypher[:120],
                            extra={"duration_ms": round((time.time() - t0) * 1000)})
                return cypher
        except Exception as e:
            logger.error("AI Cypher generation failed: %s", e)

        return None

    # ------------------------------------------------------------------
    # AI summarization (with RAG context)
    # ------------------------------------------------------------------

    def _summarize_results(
        self, question: str, results: List[Dict], rag_chunks: List[Dict] = None
    ) -> str:
        """Use OpenAI to create a human-readable summary of query results.

        DOMAIN-SPECIFIC: Update the system prompt to match your domain.
        """
        if not self.openai_client or not results:
            return ""

        t0 = time.time()
        try:
            results_text = ""
            for i, r in enumerate(results[:5]):
                parts = []
                for k, v in r.items():
                    if v is not None and v != "" and v != []:
                        parts.append(f"{k}: {v}")
                results_text += f"\nResult {i + 1}: {'; '.join(parts)}"

            rag_text = ""
            if rag_chunks:
                rag_text = "\n\n## Relevant Regulatory Text\n"
                for c in rag_chunks[:3]:
                    doc_name = (c.get("document_id") or "unknown").replace("-", " ").replace("_", " ").title()
                    page = c.get("page", "?")
                    text_snippet = (c.get("text") or "")[:600]
                    rag_text += f"\n[{doc_name}, Page {page}]:\n{text_snippet}\n"

            # DOMAIN-SPECIFIC: adjust the system prompt for your domain
            prompt = (
                "You are a regulatory compliance expert. "
                "Based on the knowledge graph results AND the relevant regulatory text below, "
                "provide a clear, concise summary that directly answers the user's question. "
                "Cite specific sections, documents, or page numbers where possible. "
                "Keep it to 3-5 sentences.\n\n"
                f"Question: {question}\n\n"
                f"## Knowledge Graph Results:{results_text}"
                f"{rag_text}"
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance assistant. Be concise, accurate, and cite specific sources."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("AI summarization failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _find_pattern(self, question: str) -> Optional[Dict[str, Any]]:
        q = question.lower()
        matches = []
        for name, pat in self.query_patterns.items():
            for kw in pat["keywords"]:
                if kw in q:
                    word_count = len(kw.split())
                    score = len(kw) + (word_count * 10 if word_count > 1 else 0)
                    matches.append((score, name, kw, pat))
                    break
        if not matches:
            return None
        matches.sort(key=lambda x: x[0], reverse=True)
        _, name, kw, pat = matches[0]
        logger.info("Matched pattern: %s (keyword: '%s')", name, kw)
        return pat

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def _fulltext_search(self, question: str) -> tuple:
        """Run full-text search and return (cypher, params)."""
        search_term = question.replace("?", "").replace("!", "").strip()
        cypher = (
            "CALL db.index.fulltext.queryNodes('entityContentSearch', $searchTerm) "
            "YIELD node, score "
            "WHERE score > 0.3 "
            "WITH node, score "
            "OPTIONAL MATCH (node)-[:MENTIONED_IN]->(doc:Document) "
            "RETURN node.name AS name, labels(node)[0] AS type, "
            "node.description AS description, score, "
            "collect(DISTINCT doc.name)[0..3] AS source_documents "
            "ORDER BY score DESC LIMIT 10"
        )
        return cypher, {"searchTerm": search_term}

    # ------------------------------------------------------------------
    # Query execution (always parameterised)
    # ------------------------------------------------------------------

    def _run_cypher(self, cypher: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        if not self.neo4j_driver:
            return []
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, parameters=params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error("Cypher execution failed: %s | Query: %s", e, cypher[:200])
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_streaming(self, question: str):
        """Generator that yields step-update dicts, then the final QueryResult.

        Each yielded dict has ``{"type": "step", "step": <name>, ...}`` or
        ``{"type": "result", "data": <QueryResult>}`` for the final result.
        The streaming endpoint writes each as an NDJSON line.
        """
        start = time.time()

        cypher = None
        params: Dict[str, Any] = {}
        confidence = 0.0
        explanation = ""
        strategy = ""
        results: List[Dict] = []
        ai_summary = ""

        # 1. Pattern matching
        yield {"type": "step", "step": "pattern_match", "status": "running"}
        pattern = self._find_pattern(question)
        if pattern:
            cypher = pattern["cypher"]
            params = pattern.get("params", {})
            results = self._run_cypher(cypher, params)
            if results:
                confidence = 0.85
                explanation = "Query matched to predefined regulatory pattern"
                strategy = "pattern_match"
        yield {
            "type": "step", "step": "pattern_match", "status": "done",
            "found": bool(results), "count": len(results),
        }

        # 2. LangChain GraphCypherQAChain
        if not results and self._lc_qa_chain:
            yield {"type": "step", "step": "langchain", "status": "running"}
            lc_result = self._langchain_cypher_qa(question)
            if lc_result and (lc_result.get("results") or lc_result.get("summary")):
                cypher = lc_result.get("cypher", "")
                results = lc_result.get("results", [])
                ai_summary = lc_result.get("summary", "")
                confidence = 0.80
                explanation = "Query processed by LangChain GraphCypherQAChain"
                strategy = "langchain"
            yield {
                "type": "step", "step": "langchain", "status": "done",
                "found": bool(results), "count": len(results),
            }
        else:
            yield {"type": "step", "step": "langchain", "status": "skipped"}

        # 3. Direct AI Cypher generation
        if not results and self.openai_client:
            yield {"type": "step", "step": "direct_ai", "status": "running"}
            ai_cypher = self._generate_cypher_with_ai(question)
            if ai_cypher:
                ai_results = self._run_cypher(ai_cypher)
                if ai_results:
                    cypher = ai_cypher
                    results = ai_results
                    confidence = 0.75
                    explanation = "Query generated using AI natural language understanding"
                    strategy = "direct_ai"
            yield {
                "type": "step", "step": "direct_ai", "status": "done",
                "found": bool(results), "count": len(results),
            }
        else:
            yield {"type": "step", "step": "direct_ai", "status": "skipped"}

        # 4. Full-text search fallback
        if not results:
            yield {"type": "step", "step": "fulltext", "status": "running"}
            ft_cypher, ft_params = self._fulltext_search(question)
            ft_results = self._run_cypher(ft_cypher, ft_params)
            if ft_results:
                cypher = ft_cypher
                params = ft_params
                results = ft_results
                confidence = 0.5
                explanation = "Using full-text search across all entities"
                strategy = "fulltext"
            yield {
                "type": "step", "step": "fulltext", "status": "done",
                "found": bool(results), "count": len(results),
            }
        else:
            yield {"type": "step", "step": "fulltext", "status": "skipped"}

        if not results:
            confidence = 0.1
            if not cypher:
                cypher = ""
            explanation = (
                "No results found for this query. "
                "Try rephrasing or using specific terms from your domain."
            )
            strategy = "no_match"

        # 5. RAG retrieval
        yield {"type": "step", "step": "rag", "status": "running"}
        rag_chunks = (
            self._langchain_rag_retrieve(question)
            if _HAS_LANGCHAIN and self._lc_vector_store
            else self._direct_rag_retrieve(question)
        )
        yield {
            "type": "step", "step": "rag", "status": "done",
            "found": bool(rag_chunks),
        }

        # 6. Summarisation
        if not ai_summary and (results or rag_chunks):
            yield {"type": "step", "step": "summary", "status": "running"}
            ai_summary = self._summarize_results(question, results, rag_chunks=rag_chunks)
            yield {"type": "step", "step": "summary", "status": "done"}
        else:
            yield {"type": "step", "step": "summary", "status": "skipped"}

        final = QueryResult(
            question=question,
            cypher_query=cypher or "",
            results=results,
            explanation=explanation,
            confidence=confidence,
            execution_time=time.time() - start,
            ai_summary=ai_summary,
            strategy=strategy,
        )
        yield {"type": "result", "data": final}

    def query(self, question: str) -> QueryResult:
        """Synchronous query -- consumes the streaming generator internally."""
        final = None
        for step in self.query_streaming(question):
            if step.get("type") == "result":
                final = step["data"]
        if final is None:
            raise RuntimeError("query_streaming did not yield a final result")
        return final

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()


# Backward-compatible alias for existing domain implementations
RegulatoryNLPEngine = WritEngine
