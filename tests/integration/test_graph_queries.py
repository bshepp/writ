"""Integration tests for graph queries against a live Neo4j instance.

Requires: docker-compose up (or a running Neo4j on bolt://localhost:7687).
Run with: pytest tests/integration -m integration
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_query_path = str(PROJECT_ROOT / "query")
if _query_path not in sys.path:
    sys.path.insert(0, _query_path)

from cypher_guard import is_read_only_cypher, CypherWriteBlockedError


@pytest.fixture(scope="module")
def neo4j_session(live_neo4j_driver):
    """Provide a Neo4j session for the entire test module."""
    with live_neo4j_driver.session() as session:
        yield session


@pytest.fixture(autouse=True, scope="module")
def seed_test_data(neo4j_session):
    """Create a small test graph and tear it down after the module."""
    neo4j_session.run(
        "CREATE (r:Regulation {id: 'test_reg_1', name: 'Test Regulation Alpha', "
        "description: 'A test regulation for integration tests'})"
    )
    neo4j_session.run(
        "CREATE (req:Requirement {id: 'test_req_1', name: 'Annual Reporting', "
        "description: 'Must submit annual compliance report', category: 'reporting'})"
    )
    neo4j_session.run(
        "CREATE (p:Personnel {id: 'test_pers_1', name: 'Site Inspector', "
        "description: 'Responsible for on-site inspections', role_type: 'Inspector'})"
    )
    neo4j_session.run(
        "CREATE (c:Control {id: 'test_ctrl_1', name: 'Erosion Barrier', "
        "description: 'Physical barrier to prevent soil erosion', control_type: 'structural'})"
    )
    neo4j_session.run(
        "CREATE (t:Timeline {id: 'test_time_1', name: '30-Day Deadline', "
        "description: 'Must be completed within 30 days', duration_days: 30})"
    )
    neo4j_session.run(
        "CREATE (d:Document {id: 'test_doc_1', name: 'Test Source Document', "
        "document_type: 'Regulatory Document'})"
    )
    neo4j_session.run(
        "MATCH (r:Regulation {id: 'test_reg_1'}), (req:Requirement {id: 'test_req_1'}) "
        "CREATE (r)-[:REQUIRES]->(req)"
    )
    neo4j_session.run(
        "MATCH (req:Requirement {id: 'test_req_1'}), (p:Personnel {id: 'test_pers_1'}) "
        "CREATE (req)-[:PERFORMED_BY]->(p)"
    )
    neo4j_session.run(
        "MATCH (req:Requirement {id: 'test_req_1'}), (c:Control {id: 'test_ctrl_1'}) "
        "CREATE (req)-[:IMPLEMENTS]->(c)"
    )
    neo4j_session.run(
        "MATCH (req:Requirement {id: 'test_req_1'}), (t:Timeline {id: 'test_time_1'}) "
        "CREATE (req)-[:HAS_DEADLINE]->(t)"
    )
    neo4j_session.run(
        "MATCH (r:Regulation {id: 'test_reg_1'}), (d:Document {id: 'test_doc_1'}) "
        "CREATE (r)-[:MENTIONED_IN]->(d)"
    )

    yield

    neo4j_session.run(
        "MATCH (n) WHERE n.id STARTS WITH 'test_' DETACH DELETE n"
    )


class TestBasicQueries:

    @pytest.mark.integration
    def test_match_regulation_by_name(self, neo4j_session):
        result = neo4j_session.run(
            "MATCH (r:Regulation) WHERE r.name = $name RETURN r.id AS id",
            {"name": "Test Regulation Alpha"},
        ).single()
        assert result is not None
        assert result["id"] == "test_reg_1"

    @pytest.mark.integration
    def test_traverse_requires_relationship(self, neo4j_session):
        results = neo4j_session.run(
            "MATCH (r:Regulation {id: 'test_reg_1'})-[:REQUIRES]->(req:Requirement) "
            "RETURN req.name AS name"
        ).data()
        assert len(results) == 1
        assert results[0]["name"] == "Annual Reporting"

    @pytest.mark.integration
    def test_traverse_full_path(self, neo4j_session):
        results = neo4j_session.run(
            "MATCH (r:Regulation {id: 'test_reg_1'})-[:REQUIRES]->(req)-[:PERFORMED_BY]->(p) "
            "RETURN r.name AS regulation, req.name AS requirement, p.name AS personnel"
        ).data()
        assert len(results) == 1
        assert results[0]["personnel"] == "Site Inspector"

    @pytest.mark.integration
    def test_mentioned_in_document(self, neo4j_session):
        results = neo4j_session.run(
            "MATCH (r:Regulation {id: 'test_reg_1'})-[:MENTIONED_IN]->(d:Document) "
            "RETURN d.name AS doc"
        ).data()
        assert len(results) == 1
        assert results[0]["doc"] == "Test Source Document"

    @pytest.mark.integration
    def test_optional_match_missing_link(self, neo4j_session):
        results = neo4j_session.run(
            "MATCH (p:Personnel {id: 'test_pers_1'}) "
            "OPTIONAL MATCH (p)-[:MENTIONED_IN]->(d:Document) "
            "RETURN p.name AS name, d.name AS doc"
        ).data()
        assert len(results) == 1
        assert results[0]["doc"] is None


class TestFullTextSearch:

    @pytest.fixture(autouse=True, scope="class")
    def _ensure_fulltext_index(self, neo4j_session):
        try:
            neo4j_session.run("DROP INDEX entityContentSearch IF EXISTS")
        except Exception:
            pass
        neo4j_session.run(
            "CREATE FULLTEXT INDEX entityContentSearch IF NOT EXISTS "
            "FOR (n:Regulation|Requirement|Personnel|Control|Timeline) "
            "ON EACH [n.name, n.description]"
        )
        import time
        time.sleep(2)

    @pytest.mark.integration
    def test_fulltext_finds_regulation(self, neo4j_session):
        results = neo4j_session.run(
            "CALL db.index.fulltext.queryNodes('entityContentSearch', $term) "
            "YIELD node, score "
            "WHERE score > 0.1 "
            "RETURN node.name AS name, labels(node)[0] AS label, score "
            "ORDER BY score DESC LIMIT 5",
            {"term": "regulation alpha"},
        ).data()
        assert len(results) >= 1
        names = [r["name"] for r in results]
        assert "Test Regulation Alpha" in names

    @pytest.mark.integration
    def test_fulltext_finds_requirement(self, neo4j_session):
        results = neo4j_session.run(
            "CALL db.index.fulltext.queryNodes('entityContentSearch', $term) "
            "YIELD node, score "
            "WHERE score > 0.1 "
            "RETURN node.name AS name, score "
            "ORDER BY score DESC LIMIT 5",
            {"term": "annual reporting compliance"},
        ).data()
        assert len(results) >= 1


class TestCypherGuardLive:
    """Verify that the Cypher guard prevents writes against a real database."""

    @pytest.mark.integration
    def test_read_only_allowed(self, neo4j_session):
        cypher = "MATCH (n) RETURN count(n) AS cnt"
        assert is_read_only_cypher(cypher) is True
        result = neo4j_session.run(cypher).single()
        assert result["cnt"] >= 0

    @pytest.mark.integration
    def test_write_blocked_by_guard(self, neo4j_session):
        cypher = "CREATE (n:EvilNode {name: 'should not exist'})"
        assert is_read_only_cypher(cypher) is False

    @pytest.mark.integration
    def test_no_evil_nodes_exist(self, neo4j_session):
        result = neo4j_session.run(
            "MATCH (n:EvilNode) RETURN count(n) AS cnt"
        ).single()
        assert result["cnt"] == 0

    @pytest.mark.integration
    def test_detach_delete_blocked(self, neo4j_session):
        cypher = "MATCH (n) DETACH DELETE n"
        assert is_read_only_cypher(cypher) is False


class TestVectorIndex:

    @pytest.mark.integration
    def test_create_and_query_vector_index(self, neo4j_session):
        try:
            neo4j_session.run("DROP INDEX test_vector_idx IF EXISTS")
        except Exception:
            pass

        neo4j_session.run(
            "CREATE VECTOR INDEX test_vector_idx IF NOT EXISTS "
            "FOR (c:TestChunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: 8, "
            "`vector.similarity_function`: 'cosine'"
            "}}"
        )

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        neo4j_session.run(
            "CREATE (c:TestChunk {chunk_id: 'test_vc_1', text: 'sample text', "
            "embedding: $emb})",
            {"emb": embedding},
        )

        import time
        time.sleep(2)

        results = neo4j_session.run(
            "CALL db.index.vector.queryNodes('test_vector_idx', 1, $emb) "
            "YIELD node, score "
            "RETURN node.chunk_id AS id, score",
            {"emb": embedding},
        ).data()
        assert len(results) >= 1
        assert results[0]["id"] == "test_vc_1"

        neo4j_session.run("MATCH (c:TestChunk) WHERE c.chunk_id = 'test_vc_1' DELETE c")
        try:
            neo4j_session.run("DROP INDEX test_vector_idx IF EXISTS")
        except Exception:
            pass
