from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional


class Neo4jGraph:
    def __init__(self, uri: str, username: str, password: str):
        """
        Initializes a connection to the Neo4j database.

        Args:
            uri (str): The URI of the Neo4j database.
            username (str): The username for the Neo4j database.
            password (str): The password for the Neo4j database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        """
        Closes the connection to the Neo4j database.
        """
        self.driver.close()

    def add_node(self, label: str, properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Adds a node to the Neo4j database with the specified label and properties.

        Args:
            label (str): The label for the node.
            properties (Dict[str, Any]): A dictionary of properties for the node.

        Returns:
            Optional[Dict[str, Any]]: The added node's properties if the node was created successfully, otherwise None.
        """
        with self.driver.session() as session:
            result = session.write_transaction(self._add_node, label, properties)
            return result

    @staticmethod
    def _add_node(tx, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Cypher query to add a node.

        Args:
            tx: The transaction object.
            label (str): The label for the node.
            properties (Dict[str, Any]): A dictionary of properties for the node.

        Returns:
            Dict[str, Any]: The added node's properties.
        """
        query = (
            f"MERGE (n:Nodes {{id: $id}}) "
            "ON CREATE SET n += $properties "
            "ON MATCH SET n += $properties "
            "RETURN n"
        )
        result = tx.run(query, id=properties['id'], properties=properties)
        return result.single()["n"]

    def create_relationship(self, start_node: Dict[str, Any], end_node: Dict[str, Any], relationship_type: str) -> Optional[str]:
        """
        Creates a relationship between two nodes in the Neo4j database.

        Args:
            start_node (Dict[str, Any]): The starting node's label and id.
            end_node (Dict[str, Any]): The ending node's label and id.
            relationship_type (str): The type of relationship to create.

        Returns:
            Optional[str]: The type of the created relationship if successful, otherwise None.
        """
        with self.driver.session() as session:
            result = session.write_transaction(self._create_relationship, start_node, end_node, relationship_type)
            return result

    @staticmethod
    def _create_relationship(tx, start_node: Dict[str, Any], end_node: Dict[str, Any], relationship_type: str) -> str:
        """
        Executes the Cypher query to create a relationship between two nodes.

        Args:
            tx: The transaction object.
            start_node (Dict[str, Any]): The starting node's label and id.
            end_node (Dict[str, Any]): The ending node's label and id.
            relationship_type (str): The type of relationship to create.

        Returns:
            str: The type of the created relationship.
        """
        query = (
            f"MATCH (a:Nodes {{id: $start_id}}), (b:Nodes {{id: $end_id}}) "
            f"MERGE (a)-[r:{relationship_type}]->(b) "
            "RETURN type(r)"
        )
        result = tx.run(query, start_id=start_node['id'], end_id=end_node['id'])
        return result.single()[0]

    def add_graph_documents(self, graph_documents: List[Any]) -> None:
        """
        Adds nodes and relationships from a list of graph documents.

        Args:
            graph_documents (List[Any]): A list of documents containing nodes and relationships to be added to the graph.
        """
        for doc in graph_documents:
            with self.driver.session() as session:
                for node in doc.nodes:
                    properties = {k: v for k, v in node.__dict__.items() if k != 'type'}
                    session.write_transaction(self._add_node, node.type, properties)

                for rel in doc.relationships:
                    start_node = {'label': rel.source.type, 'id': rel.source.id}
                    end_node = {'label': rel.target.type, 'id': rel.target.id}
                    session.write_transaction(self._create_relationship, start_node, end_node, rel.type)

    def print_graph_summary(self) -> None:
        """
        Prints a summary of the graph, including the total number of nodes and relationships.
        """
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"Total Nodes: {node_count}")
            print(f"Total Relationships: {rel_count}")

    def check_data_exists(self) -> bool:
        """
        Checks if there are any nodes or relationships in the graph.

        Returns:
            bool: True if there are nodes or relationships in the graph, False otherwise.
        """
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN n LIMIT 5").values()
            relationships = session.run("MATCH ()-[r]->() RETURN r LIMIT 5").values()
            print(f"Sample Nodes: {nodes}")
            print(f"Sample Relationships: {relationships}")
            return bool(nodes or relationships)
