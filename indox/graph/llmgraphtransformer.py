from indox.core import Document
from typing import List, Dict, Any
import json


# Node class to represent each entity
class Node:
    def __init__(self, id: str, type: str, embedding: List[float] = None, text: str = None):
        self.id = id
        self.type = type
        self.embedding = embedding
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        node_dict = {"id": self.id, "type": self.type}
        if self.embedding:
            node_dict["embedding"] = self.embedding
        if self.text:
            node_dict["text"] = self.text
        return node_dict


# Relationship class to represent relationships between nodes
class Relationship:
    def __init__(self, source: Node, target: Node, type: str):
        self.source = source
        self.target = target
        self.type = type

    def to_dict(self) -> Dict[str, str]:
        return {
            "source": self.source.id,
            "target": self.target.id,
            "type": self.type
        }


# GraphDocument class to represent the overall graph structure (nodes and relationships)
class GraphDocument:
    def __init__(self, nodes: List[Node], relationships: List[Relationship], source: Document):
        self.nodes = nodes
        self.relationships = relationships
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "source": {
                "page_content": self.source.page_content,
                "metadata": self.source.metadata
            }
        }

    def to_cypher(self) -> str:
        nodes_cypher = "\n".join(
            [f"CREATE (: {node.type} {{id: '{node.id}', text: '{node.text}'}})" for node in self.nodes if node.text])
        relationships_cypher = "\n".join(
            [f"MATCH (a {{id: '{rel.source.id}'}}), (b {{id: '{rel.target.id}'}}) CREATE (a)-[:{rel.type}]->(b)"
             for rel in self.relationships]
        )
        return nodes_cypher + "\n" + relationships_cypher


# LLMGraphTransformer class to interact with any LLM API and convert text into a graph document
class LLMGraphTransformer:
    def __init__(self, llm_transformer: Any, embeddings_model: Any):
        """
        Initializes the LLMGraphTransformer with an LLM transformer and embedding model.

        Args:
            llm_transformer (Any): An instance of the LLM transformer (e.g., IndoxApi, OpenAI).
            embeddings_model (Any): An instance of the embedding model to generate text embeddings.
        """
        self.llm_transformer = llm_transformer  # The transformer must have a `chat` method
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
        human_prompt = (
            f"Convert the following text into a knowledge graph by extracting entities and their relationships. "
            f"Provide the output as a JSON object with 'nodes' and 'relationships' lists as described: {text}"
        )
        return {
            "system_prompt": self.system_prompt_template,
            "human_prompt": human_prompt
        }

    def _safe_parse_llm_output(self, raw_output: str) -> Dict[str, List[Dict[str, Any]]]:
        json_output = raw_output.strip('```json').strip('```').strip()
        try:
            return json.loads(json_output)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM output is not valid JSON: {e}")

    def call_llm_with_retry(self, text: str, retries: int = 3) -> str:
        attempts = 0
        while attempts < retries:
            try:
                prompts = self.create_prompts(text)
                response = self.llm_transformer.chat(
                    prompt=prompts["human_prompt"],
                    system_prompt=prompts["system_prompt"]
                )
                return response
            except Exception as e:
                print(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1
        raise ValueError(f"Failed after {retries} attempts.")

    def extract_graph_from_text(self, text: str) -> Dict[str, Any]:
        response = self.call_llm_with_retry(text)
        return self._safe_parse_llm_output(response)

    def generate_embeddings_for_text(self, text: str) -> List[float]:
        return self.embeddings_model.embed_documents(text)

    def convert_to_graph_documents(self, text_chunks: List[str], metadata: Dict[str, Any] = None,
                                   embeddings: List[List[float]] = None) -> List[GraphDocument]:
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
