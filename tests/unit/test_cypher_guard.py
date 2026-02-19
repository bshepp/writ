"""Tests for the Cypher read-only guard.

Verifies that the guard correctly identifies write operations and allows
read-only queries through.  No Neo4j or LangChain connection needed.
"""

import sys
from pathlib import Path

import pytest

_query_path = str(Path(__file__).resolve().parent.parent.parent / "query")
if _query_path not in sys.path:
    sys.path.insert(0, _query_path)

from cypher_guard import is_read_only_cypher, find_write_violations


class TestReadOnlyQueries:

    @pytest.mark.unit
    def test_simple_match(self):
        assert is_read_only_cypher("MATCH (n) RETURN n") is True

    @pytest.mark.unit
    def test_match_with_where(self):
        assert is_read_only_cypher(
            "MATCH (p:Personnel) WHERE p.role_type = 'Inspector' RETURN p.name"
        ) is True

    @pytest.mark.unit
    def test_optional_match(self):
        cypher = (
            "MATCH (b:Control) "
            "OPTIONAL MATCH (b)-[:MENTIONED_IN]->(d:Document) "
            "RETURN b.name, collect(d.name)"
        )
        assert is_read_only_cypher(cypher) is True

    @pytest.mark.unit
    def test_fulltext_search(self):
        cypher = (
            "CALL db.index.fulltext.queryNodes('entityContentSearch', $term) "
            "YIELD node, score RETURN node.name, score ORDER BY score DESC LIMIT 10"
        )
        assert is_read_only_cypher(cypher) is True

    @pytest.mark.unit
    def test_vector_search(self):
        cypher = (
            "CALL db.index.vector.queryNodes('chunk_embeddings', $topK, $embedding) "
            "YIELD node, score RETURN node.text, score"
        )
        assert is_read_only_cypher(cypher) is True

    @pytest.mark.unit
    def test_empty_query(self):
        assert is_read_only_cypher("") is True
        assert is_read_only_cypher("   ") is True

    @pytest.mark.unit
    def test_none_input(self):
        assert is_read_only_cypher(None) is True


class TestWriteOperations:

    @pytest.mark.unit
    def test_create_node(self):
        cypher = "CREATE (n:Person {name: 'Evil'})"
        assert is_read_only_cypher(cypher) is False
        assert "CREATE" in find_write_violations(cypher)

    @pytest.mark.unit
    def test_merge_node(self):
        assert is_read_only_cypher("MERGE (n:Person {name: 'Sneaky'})") is False

    @pytest.mark.unit
    def test_delete_node(self):
        assert is_read_only_cypher("MATCH (n) DELETE n") is False

    @pytest.mark.unit
    def test_detach_delete(self):
        cypher = "MATCH (n) DETACH DELETE n"
        violations = find_write_violations(cypher)
        assert "DETACH" in violations
        assert "DELETE" in violations

    @pytest.mark.unit
    def test_set_property(self):
        assert is_read_only_cypher("MATCH (n) SET n.name = 'Hacked'") is False

    @pytest.mark.unit
    def test_remove_property(self):
        assert is_read_only_cypher("MATCH (n) REMOVE n.name") is False

    @pytest.mark.unit
    def test_drop_index(self):
        assert is_read_only_cypher("DROP INDEX my_index") is False

    @pytest.mark.unit
    def test_load_csv(self):
        cypher = "LOAD CSV FROM 'http://evil.com/data.csv' AS row CREATE (n {name: row[0]})"
        assert "LOAD CSV" in find_write_violations(cypher)

    @pytest.mark.unit
    def test_foreach(self):
        cypher = "MATCH (n) FOREACH (x IN [1,2] | SET n.count = x)"
        assert "FOREACH" in find_write_violations(cypher)

    @pytest.mark.unit
    def test_case_insensitive(self):
        assert is_read_only_cypher("match (n) delete n") is False
        assert is_read_only_cypher("Match (n) SET n.x = 1") is False


class TestFalsePositives:

    @pytest.mark.unit
    def test_delete_in_string_literal(self):
        assert is_read_only_cypher(
            "MATCH (n) WHERE n.name = 'DELETE this record' RETURN n"
        ) is True

    @pytest.mark.unit
    def test_create_in_property_name(self):
        assert is_read_only_cypher("MATCH (n) RETURN n.created_at") is True

    @pytest.mark.unit
    def test_set_in_property_name(self):
        assert is_read_only_cypher("MATCH (n) WHERE n.dataset = 'test' RETURN n") is True

    @pytest.mark.unit
    def test_merge_in_property_name(self):
        assert is_read_only_cypher("MATCH (n) WHERE n.emerged = true RETURN n") is True

    @pytest.mark.unit
    def test_remove_in_comment(self):
        assert is_read_only_cypher("MATCH (n) // REMOVE this later\nRETURN n") is True


class TestPromptInjection:

    @pytest.mark.unit
    def test_injection_detach_delete_all(self):
        assert is_read_only_cypher("MATCH (n) DETACH DELETE n") is False

    @pytest.mark.unit
    def test_injection_create_admin(self):
        assert is_read_only_cypher(
            "CREATE (u:User {name: 'admin', role: 'superadmin'})"
        ) is False

    @pytest.mark.unit
    def test_injection_mixed_read_write(self):
        cypher = (
            "MATCH (n:Document) WITH n LIMIT 1 "
            "SET n.compromised = true RETURN n"
        )
        assert is_read_only_cypher(cypher) is False

    @pytest.mark.unit
    def test_injection_drop_constraint(self):
        assert is_read_only_cypher("DROP CONSTRAINT unique_name IF EXISTS") is False
