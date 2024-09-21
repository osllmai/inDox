from neo4j import GraphDatabase
from typing import List, Dict, Optional
from indox.graph import GraphDocument, Node, Relationship

class MemgraphDB:
    def __init__(self, uri: str, username: Optional[str] = "", password: Optional[str] = ""):
        """
        Initializes the connection to the Memgraph database.

        Args:
            uri (str): URI of the Memgraph instance (e.g., "bolt://localhost:7687").
            username (Optional[str]): Username for Memgraph (optional, defaults to an empty string).
            password (Optional[str]): Password for Memgraph (optional, defaults to an empty string).
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Closes the Memgraph connection."""
        if self.driver:
            self.driver.close()

    def add_graph_documents(self, graph_documents: List[GraphDocument], base_entity_label: bool = True, include_source: bool = True):
        """
        Adds a list of graph documents to the Memgraph database.

        Args:
            graph_documents (List[GraphDocument]): A list of GraphDocument objects to be added to the Memgraph database.
            base_entity_label (bool): Whether to add a base entity label to each node. Defaults to True.
            include_source (bool): Whether to include the source document as part of the graph. Defaults to True.
        """
        try:
            with self.driver.session() as session:
                for graph_doc in graph_documents:
                    self._add_graph_document(session, graph_doc, base_entity_label, include_source)
        finally:
            self.close()

    def _add_graph_document(self, session, graph_doc: GraphDocument, base_entity_label: bool, include_source: bool):
        """
        Adds a single GraphDocument to the Memgraph database.

        Args:
            session: Memgraph session.
            graph_doc (GraphDocument): The graph document to be added.
            base_entity_label (bool): Whether to add a base entity label to each node.
            include_source (bool): Whether to include the source document as part of the graph.
        """
        for node in graph_doc.nodes:
            self._add_node(session, node, base_entity_label)
        for relationship in graph_doc.relationships:
            self._add_relationship(session, relationship)
        if include_source:
            self._add_source(session, graph_doc)

    def _add_node(self, session, node: Node, base_entity_label: bool):
        """
        Adds a node to the Memgraph database.

        Args:
            session: Memgraph session.
            node (Node): The node object to be added.
            base_entity_label (bool): Whether to add a base entity label to the node.
        """
        labels = [node.type]
        if base_entity_label:
            labels.append("Entity")
        query = f"MERGE (n:{':'.join(labels)} {{id: $id}}) SET n = $properties"
        properties = {"id": node.id}
        if node.embedding is not None:
            properties["embedding"] = node.embedding
        if node.text is not None:
            properties["text_data"] = node.text

        session.run(query, {"id": node.id, "properties": properties})

    def _add_relationship(self, session, relationship: Relationship):
        """
        Adds a relationship between two nodes in the Memgraph database.

        Args:
            session: Memgraph session.
            relationship (Relationship): The relationship to be added.
        """
        relationship_type = relationship.type.replace(" ", "_")
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        """
        session.run(query, {"source_id": relationship.source.id, "target_id": relationship.target.id})

    def _add_source(self, session, graph_doc: GraphDocument):
        """
        Adds the source metadata to a special node in the Memgraph database.

        Args:
            session: Memgraph session.
            graph_doc (GraphDocument): The graph document containing the source information.
        """
        source = graph_doc.source
        query = """
        MERGE (s:Source {title: $title})
        SET s.page_content = $page_content
        """
        params = {
            "title": source.metadata.get('title', 'Untitled'),
            "page_content": source.page_content
        }
        session.run(query, params)

        for node in graph_doc.nodes:
            query = """
            MATCH (n {id: $node_id}), (s:Source {title: $title})
            MERGE (n)-[:HAS_SOURCE]->(s)
            """
            session.run(query, {"node_id": node.id, "title": source.metadata.get("title", "Untitled")})

    def search_relationships_by_entity(self, entity_id: str, relationship_type: str) -> List[Dict]:
        """
        Searches for relationships by entity ID and relationship type.

        Args:
            entity_id (str): The ID of the entity to search for.
            relationship_type (str): The type of relationship to search for.

        Returns:
            List[Dict]: A list of dictionaries, each containing the source entity, relationship type, and target entity.
        """
        query = f"""
        MATCH (a {{id: $entity_id}})-[r:{relationship_type}]->(b)
        RETURN a, TYPE(r) AS rel_type, b
        """
        with self.driver.session() as session:
            result = session.run(query, {"entity_id": entity_id})
            return [record.data() for record in result]
