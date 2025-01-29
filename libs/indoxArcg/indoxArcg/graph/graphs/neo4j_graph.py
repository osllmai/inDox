from typing import List, Dict
from indoxArcg.core import Document


# Define the Neo4jGraph class for structured Cypher queries
class Neo4jGraph:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initializes the connection to the Neo4j database.

        Args:
            uri (str): URI of the Neo4j instance (e.g., "bolt://localhost:7687").
            username (str): Username for Neo4j.
            password (str): Password for Neo4j.
        """
        try:
            # Import neo4j GraphDatabase dynamically to handle environments where it might not be installed.
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Could not import the neo4j package. Please install it with `pip install neo4j`."
            )

        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Closes the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def add_graph_documents(
        self,
        graph_documents: List["GraphDocument"],
        base_entity_label: bool = True,
        include_source: bool = True,
    ):
        """
        Adds graph documents to the Neo4j database.

        Args:
            graph_documents (List[GraphDocument]): List of GraphDocument objects to be added to the Neo4j database.
            base_entity_label (bool): Whether to add a base entity label to the node. Defaults to True.
            include_source (bool): Whether to include the source document as part of the graph. Defaults to True.
        """
        try:
            with self.driver.session() as session:
                for graph_doc in graph_documents:
                    self._add_graph_document(
                        session, graph_doc, base_entity_label, include_source
                    )
        finally:
            self.close()

    def _add_graph_document(
        self,
        session,
        graph_doc: "GraphDocument",
        base_entity_label: bool,
        include_source: bool,
    ):
        """
        Adds a single GraphDocument to the Neo4j database.

        Args:
            session: Neo4j session.
            graph_doc (GraphDocument): The graph document to be added.
            base_entity_label (bool): Whether to add a base entity label to the node.
            include_source (bool): Whether to include the source document as part of the graph.
        """
        for node in graph_doc.nodes:
            self._add_node(session, node, base_entity_label)
        for relationship in graph_doc.relationships:
            self._add_relationship(session, relationship)
        if include_source:
            self._add_source(session, graph_doc)

    def _add_node(self, session, node: "Node", base_entity_label: bool):
        """
        Adds a node to the Neo4j database.

        Args:
            session: Neo4j session.
            node (Node): The node object to be added.
            base_entity_label (bool): Whether to add a base entity label to the node.
        """
        labels = f":{node.type}"
        if base_entity_label:
            labels += ":Entity"
        query = f"MERGE (n{labels} {{id: $id}})"
        params = {"id": node.id}
        if node.embedding:
            query += " SET n.embedding = $embedding"
            params["embedding"] = node.embedding
        if node.text:
            query += " SET n.text = $text"
            params["text"] = node.text
        session.run(query, params)

    def _add_relationship(self, session, relationship: "Relationship"):
        """
        Adds a relationship between two nodes in the Neo4j database.

        Args:
            session: Neo4j session.
            relationship (Relationship): The relationship to be added.
        """
        relationship_type = relationship.type.replace(" ", "_")
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        """
        session.run(
            query,
            {"source_id": relationship.source.id, "target_id": relationship.target.id},
        )

    def _add_source(self, session, graph_doc: "GraphDocument"):
        """
        Adds the source metadata to a special node in the Neo4j database.

        Args:
            session: Neo4j session.
            graph_doc (GraphDocument): The graph document containing the source information.
        """
        source = graph_doc.source
        query = """
        MERGE (s:Source {title: $title, page_content: $page_content})
        """
        params = {
            "title": source.metadata.get("metadata", {}).get("title", "Untitled"),
            "page_content": source.page_content,
        }
        session.run(query, params)

        for node in graph_doc.nodes:
            query = """
            MATCH (n {id: $node_id}), (s:Source {title: $title})
            MERGE (n)-[:HAS_SOURCE]->(s)
            """
            session.run(
                query,
                {
                    "node_id": node.id,
                    "title": source.metadata.get("metadata", {}).get(
                        "title", "Untitled"
                    ),
                },
            )

    def search_relationships_by_entity(self, entity_id: str, relationship_type: str):
        """
        Searches for relationships by entity ID and relationship type.

        Args:
            entity_id (str): The ID of the entity to search for.
            relationship_type (str): The type of relationship to search for.

        Returns:
            List[Dict]: A list of relationships including the source entity, relationship type, and target entity.
        """
        query = f"""
        MATCH (a {{id: $entity_id}})-[r:{relationship_type}]->(b)
        RETURN a, TYPE(r) AS rel_type, b
        """
        with self.driver.session() as session:
            result = session.run(query, {"entity_id": entity_id})
            return [record for record in result]
