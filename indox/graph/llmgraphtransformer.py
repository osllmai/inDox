from indox.llms import IndoxApi
from indox.core import Document
from indox.embeddings import HuggingFaceEmbedding
from typing import List, Dict, Any
import json


# Node class to represent each entity
class Node:
    def __init__(self, id: str, type: str, embedding: List[float] = None, text: str = None):
        """
        Initializes a Node object.

        Args:
            id (str): The unique identifier for the node (usually a name or title).
            type (str): The category or type of the entity (e.g., 'Person', 'Country').
            embedding (List[float], optional): Embedding vector for chunk nodes. Defaults to None.
            text (str, optional): Text for the chunk node. Defaults to None.
        """
        self.id = id
        self.type = type
        self.embedding = embedding
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Node object into a dictionary format for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the Node.
        """
        node_dict = {"id": self.id, "type": self.type}
        if self.embedding:
            node_dict["embedding"] = self.embedding
        if self.text:
            node_dict["text"] = self.text
        return node_dict


# Relationship class to represent relationships between nodes
class Relationship:
    def __init__(self, source: Node, target: Node, type: str):
        """
        Initializes a Relationship object between two nodes.

        Args:
            source (Node): The source node of the relationship.
            target (Node): The target node of the relationship.
            type (str): The type of relationship (e.g., 'PARENT', 'REIGNED').
        """
        self.source = source
        self.target = target
        self.type = type

    def to_dict(self) -> Dict[str, str]:
        """
        Converts the Relationship object into a dictionary format for serialization.

        Returns:
            Dict[str, str]: Dictionary representation of the Relationship.
        """
        return {
            "source": self.source.id,
            "target": self.target.id,
            "type": self.type
        }


# GraphDocument class to represent the overall graph structure (nodes and relationships)
class GraphDocument:
    def __init__(self, nodes: List[Node], relationships: List[Relationship], source: Document):
        """
        Initializes a GraphDocument that holds all nodes and relationships.

        Args:
            nodes (List[Node]): List of Node objects representing entities and chunks.
            relationships (List[Relationship]): List of Relationship objects representing connections between nodes.
            source (Document): A Document object representing the source document.
        """
        self.nodes = nodes
        self.relationships = relationships
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the GraphDocument into a dictionary format for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the GraphDocument.
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "source": {
                "page_content": self.source.page_content,
                "metadata": self.source.metadata
            }
        }

    def to_cypher(self) -> str:
        """
        Converts the GraphDocument into a Cypher query for Neo4j ingestion.

        Returns:
            str: A Cypher query string for creating nodes and relationships in Neo4j.
        """
        nodes_cypher = "\n".join(
            [f"CREATE (: {node.type} {{id: '{node.id}', text: '{node.text}'}})" for node in self.nodes if node.text])
        relationships_cypher = "\n".join(
            [f"MATCH (a {{id: '{rel.source.id}'}}), (b {{id: '{rel.target.id}'}}) CREATE (a)-[:{rel.type}]->(b)"
             for rel in self.relationships]
        )
        return nodes_cypher + "\n" + relationships_cypher


# LLMGraphTransformer class to interact with the LLM and convert text into a graph document
class LLMGraphTransformer:
    def __init__(self, api_key: str, embeddings_model: HuggingFaceEmbedding):
        """
        Initializes the LLMGraphTransformer with an API key and embedding model.

        Args:
            api_key (str): The API key to access the Indox LLM.
            embeddings_model (HuggingFaceEmbedding): An instance of the embedding model to generate text embeddings.
        """
        self.api_key = api_key
        self.model = IndoxApi(api_key=api_key)
        self.embeddings_model = embeddings_model

        self.system_prompt_template = (
            "You are an AI that specializes in transforming text into structured knowledge graphs. "
            "Your task is to extract key entities (as nodes) and their relationships from the text provided. "
            "Each entity should be identified by a unique ID and categorized by type (e.g., Person, Country, Monarchy, etc.). "
            "Relationships between entities should be clearly defined with a type that describes their connection (e.g., REIGNED, PARENT, SIBLING, etc.). "
            "The output should be a JSON object with two main lists: "
            "1. 'nodes': A list of objects where each object represents an entity with the following structure: "
            "- 'id': A unique identifier for the entity (usually a name or title). "
            "- 'type': The category of the entity (e.g., Person, Country, Monarchy). "
            "2. 'relationships': A list of objects where each object represents a relationship between two entities with the following structure: "
            "- 'source': The 'id' of the source entity. "
            "- 'target': The 'id' of the target entity. "
            "- 'type': The type of relationship (e.g., REIGNED, PARENT, SIBLING). "
            "Ensure that the JSON output is properly formatted and free of errors."
        )

    def create_prompts(self, text: str) -> Dict[str, str]:
        """
        Generates the system and human prompts to send to the LLM.

        Args:
            text (str): The input text from which entities and relationships should be extracted.

        Returns:
            Dict[str, str]: Dictionary containing the system prompt and human prompt.
        """
        human_prompt = (
            f"Convert the following text into a knowledge graph by extracting entities and their relationships. "
            f"Provide the output as a JSON object with 'nodes' and 'relationships' lists as described: {text}"
        )
        return {
            "system_prompt": self.system_prompt_template,
            "human_prompt": human_prompt
        }

    def _safe_parse_llm_output(self, raw_output: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parses the raw LLM output, ensuring it's a valid JSON object.

        Args:
            raw_output (str): The raw string output from the LLM.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The parsed JSON output.

        Raises:
            ValueError: If the LLM output is not valid JSON.
        """
        json_output = raw_output.strip('```json').strip('```').strip()
        try:
            return json.loads(json_output)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM output is not valid JSON: {e}")

    def call_llm_with_retry(self, text: str, retries: int = 3) -> str:
        """
        Calls the LLM with retry logic to ensure robustness.

        Args:
            text (str): The text to be sent to the LLM.
            retries (int): The number of retry attempts.

        Returns:
            str: The LLM response.
        """
        attempts = 0
        while attempts < retries:
            try:
                prompts = self.create_prompts(text)
                response = self.model.chat(
                    prompt=prompts["human_prompt"],
                    system_prompt=prompts["system_prompt"]
                )
                return response
            except Exception as e:
                print(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1
        raise ValueError(f"Failed after {retries} attempts.")

    def extract_graph_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities and relationships from text by interacting with the LLM.

        Args:
            text (str): The input text to be processed.

        Returns:
            Dict[str, Any]: Parsed JSON output containing nodes and relationships.
        """
        response = self.call_llm_with_retry(text)
        return self._safe_parse_llm_output(response)

    def generate_embeddings_for_text(self, text: str) -> List[float]:
        """
        Generates embedding for the given text using the embedding model.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector for the text.
        """
        return self.embeddings_model.embed_documents(text)

    def convert_to_graph_documents(self, text_chunks: List[str], metadata: Dict[str, Any] = None,
                                   embeddings: List[List[float]] = None) -> List[GraphDocument]:
        """
        Converts text chunks into graph documents using the LLM API and optionally generates embeddings.

        Args:
            text_chunks (List[str]): List of text chunks to be processed by the LLM.
            metadata (Dict[str, Any]): Metadata about the source of the text (optional).
            embeddings (List[List[float]]): List of embeddings corresponding to the text chunks.

        Returns:
            List[GraphDocument]: A list of GraphDocument objects containing extracted nodes and relationships.
        """
        if embeddings and len(embeddings) != len(text_chunks):
            raise ValueError("Number of embeddings must match the number of text chunks.")

        graph_documents = []
        source_document = Document(page_content=" ".join(text_chunks), metadata=metadata)

        for i, text in enumerate(text_chunks):
            # Extract graph structure from text via LLM
            response_json = self.extract_graph_from_text(text)

            # Create Chunk node with embedding and text (chunk text)
            chunk_embedding = embeddings[i] if embeddings else self.generate_embeddings_for_text(text)
            chunk_node = Node(id=f"Chunk_{i}", type="Chunk", embedding=chunk_embedding, text=text)

            # Extract nodes and relationships from the LLM output
            nodes = [Node(id=node['id'], type=node['type']) for node in response_json.get('nodes', [])]
            relationships = [
                Relationship(
                    source=Node(id=rel['source'], type=""),  # Keep using Node for relationships
                    target=Node(id=rel['target'], type=""),
                    type=rel['type']
                ) for rel in response_json.get('relationships', [])
            ]

            # Connect the chunk node to its extracted nodes (e.g., entities)
            for entity_node in nodes:
                relationships.append(Relationship(source=chunk_node, target=entity_node, type="CONTAINS_ENTITY"))

            # Add the chunk node and its relationships to the graph
            all_nodes = [chunk_node] + nodes
            graph_doc = GraphDocument(nodes=all_nodes, relationships=relationships, source=source_document)
            graph_documents.append(graph_doc)

        return graph_documents
