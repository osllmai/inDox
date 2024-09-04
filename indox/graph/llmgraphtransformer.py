from indox.llms import IndoxApi
from typing import List, Dict, Any
import json

# Node class to represent each entity
class Node:
    def __init__(self, id: str, type: str):
        """
        Initializes a Node object.
        
        Args:
            id (str): The unique identifier for the node (usually a name or title).
            type (str): The category or type of the entity (e.g., 'Person', 'Country').
        """
        self.id = id
        self.type = type

    def to_dict(self) -> Dict[str, str]:
        """
        Converts the Node object into a dictionary format for serialization.

        Returns:
            Dict[str, str]: Dictionary representation of the Node.
        """
        return {"id": self.id, "type": self.type}

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
    def __init__(self, nodes: List[Node], relationships: List[Relationship], source: Dict[str, Any]):
        """
        Initializes a GraphDocument that holds all nodes and relationships.
        
        Args:
            nodes (List[Node]): List of Node objects representing entities.
            relationships (List[Relationship]): List of Relationship objects representing connections between nodes.
            source (Dict[str, Any]): Metadata about the source document (e.g., title, URL).
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
            "source": self.source
        }

# LLMGraphTransformer class to interact with the LLM and convert text into a graph document
class LLMGraphTransformer:
    def __init__(self, api_key: str):
        """
        Initializes the LLMGraphTransformer with an API key.

        Args:
            api_key (str): The API key to access the Indox LLM.
        """
        self.api_key = api_key
        self.model = IndoxApi(api_key=api_key)

    def create_prompts(self, text: str) -> Dict[str, str]:
        """
        Generates the prompts to send to the LLM.

        Args:
            text (str): The text from which entities and relationships should be extracted.

        Returns:
            Dict[str, str]: Dictionary containing the human prompt and system prompt.
        """
        system_prompt = ("You are an AI that specializes in transforming text into structured knowledge graphs. "
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
                         "Ensure that the JSON output is properly formatted and free of errors.")

        human_prompt = (f"Convert the following text into a knowledge graph by extracting entities and their relationships. "
                        f"Provide the output as a JSON object with 'nodes' and 'relationships' lists as described: {text}")

        return {"system_prompt": system_prompt, "human_prompt": human_prompt}

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
        # Clean up the output by stripping any markdown code block delimiters
        json_output = raw_output.strip('```json').strip('```').strip()

        try:
            # Attempt to load the string as JSON
            return json.loads(json_output)
        except json.JSONDecodeError as e:
            # Raise an error if the JSON parsing fails
            raise ValueError(f"LLM output is not valid JSON: {e}")

    def convert_to_graph_documents(self, text_chunks: List[str], metadata: Dict[str, Any] = {}) -> List[GraphDocument]:
        """
        Converts text chunks into graph documents using the LLM API.

        Args:
            text_chunks (List[str]): List of text chunks to be processed by the LLM.
            metadata (Dict[str, Any]): Metadata about the source of the text (optional).

        Returns:
            List[GraphDocument]: A list of GraphDocument objects containing extracted nodes and relationships.
        """
        graph_documents = []

        for text in text_chunks:
            # Create prompts for the LLM
            prompts = self.create_prompts(text)

            # Call the LLM API with the system and human prompts
            response = self.model.chat(
                prompt=prompts["human_prompt"], 
                system_prompt=prompts["system_prompt"]
            )

            # Parse the LLM's response safely
            try:
                response_json = self._safe_parse_llm_output(response)
            except ValueError as e:
                print(f"Error parsing LLM output: {e}")
                continue  # Skip this chunk if the output is not valid JSON

            # Extract nodes and relationships
            nodes = [Node(id=node['id'], type=node['type']) for node in response_json.get('nodes', [])]
            relationships = [
                Relationship(
                    source=Node(id=rel['source'], type=""),
                    target=Node(id=rel['target'], type=""),
                    type=rel['type']
                ) for rel in response_json.get('relationships', [])
            ]

            # Create the GraphDocument
            graph_doc = GraphDocument(nodes=nodes, relationships=relationships, source={"page_content": text, "metadata": metadata})
            graph_documents.append(graph_doc)

        return graph_documents
