from neo4j import GraphDatabase
from typing import List, Dict

class CustomNeo4jGraph:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the CustomNeo4jGraph with connection details.

        Args:
            uri (str): The Neo4j connection URI (e.g., "bolt://localhost:7687").
            username (str): The username for authenticating with the Neo4j database.
            password (str): The password for authenticating with the Neo4j database.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j database: {e}")

    def close(self):
        """
        Close the connection to the Neo4j database.
        """
        try:
            self.driver.close()
        except Exception as e:
            raise RuntimeError(f"Failed to close Neo4j connection: {e}")

    def _add_node(self, tx, node: Dict[str, str]):
        """
        Add a single node to the Neo4j database.

        Args:
            tx: A Neo4j transaction object.
            node (Dict[str, str]): A dictionary representing the node with 'id' and 'type'.
        """
        query = (
            "MERGE (n:Entity {id: $id}) "
            "SET n.type = $type"
        )
        try:
            tx.run(query, id=node["id"], type=node["type"])
        except Exception as e:
            raise RuntimeError(f"Failed to add node: {e}")

    def _add_relationship(self, tx, relationship: Dict[str, str]):
        """
        Add a single relationship to the Neo4j database with a dynamic relationship type.

        Args:
            tx: A Neo4j transaction object.
            relationship (Dict[str, str]): A dictionary with 'source', 'target', and 'type' fields.

        Raises:
            RuntimeError: If adding the relationship to the database fails.
        """
        # Ensure relationship type does not contain spaces or invalid characters
        relationship_type = relationship['type'].replace(" ", "_").upper()

        query = (
            f"MATCH (a:Entity {{id: $source_id}}), (b:Entity {{id: $target_id}}) "
            f"MERGE (a)-[r:{relationship_type}]->(b)"
        )
        try:
            tx.run(query, source_id=relationship["source"], target_id=relationship["target"])
        except Exception as e:
            raise RuntimeError(f"Failed to add relationship: {e}")

    def add_graph_document(self, graph_document: Dict[str, List[Dict[str, str]]]):
        """
        Add all nodes and relationships from a graph document to the Neo4j database.

        Args:
            graph_document (Dict[str, List[Dict[str, str]]]): A dictionary containing nodes and relationships.

        Raises:
            RuntimeError: If adding the graph document to the database fails.
        """
        with self.driver.session() as session:
            try:
                # Add nodes
                for node in graph_document.get("nodes", []):
                    session.write_transaction(self._add_node, node)

                # Add relationships
                for relationship in graph_document.get("relationships", []):
                    session.write_transaction(self._add_relationship, relationship)
            except Exception as e:
                raise RuntimeError(f"Failed to add graph document: {e}")

    def add_graph_documents(self, graph_documents: List[Dict[str, List[Dict[str, str]]]]):
        """
        Add multiple graph documents to the Neo4j database.

        Args:
            graph_documents (List[Dict[str, List[Dict[str, str]]]]): A list of graph documents to be added.

        Raises:
            RuntimeError: If adding any of the graph documents to the database fails.
        """
        for graph_document in graph_documents:
            self.add_graph_document(graph_document)

    def query_graph(self, query: str) -> List[Dict[str, str]]:
        """
        Execute a custom Cypher query on the Neo4j database and return the result.

        Args:
            query (str): A Cypher query string.

        Returns:
            List[Dict[str, str]]: A list of records returned by the query.

        Raises:
            RuntimeError: If executing the query fails.
        """
        with self.driver.session() as session:
            try:
                result = session.run(query)
                return [record for record in result]
            except Exception as e:
                raise RuntimeError(f"Failed to execute query: {e}")
