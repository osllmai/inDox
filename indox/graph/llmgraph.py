import json
from typing import List, Dict, Any, Tuple, Sequence
from indox.llms import IndoxApi


class Node:
    def __init__(self, id: str, type: str):
        """
        Initialize a Node object.

        Args:
            id (str): A unique identifier for the node.
            type (str): The category/type of the node (e.g., Person, Country).
        """
        self.id = id
        self.type = type

    def __repr__(self):
        return f"Node(id='{self.id}', type='{self.type}')"


class Relationship:
    def __init__(self, source: Node, target: Node, type: str):
        """
        Initialize a Relationship object.

        Args:
            source (Node): The source node in the relationship.
            target (Node): The target node in the relationship.
            type (str): The type of relationship (e.g., REIGNED, PARENT).
        """
        self.source = source
        self.target = target
        self.type = type

    def __repr__(self):
        return f"Relationship(source={self.source}, target={self.target}, type='{self.type}')"


class GraphDocument:
    def __init__(self, nodes: List[Node], relationships: List[Relationship], source: str):
        """
        Initialize a GraphDocument object.

        Args:
            nodes (List[Node]): A list of Node objects representing entities.
            relationships (List[Relationship]): A list of Relationship objects between nodes.
            source (str): The source text from which the graph was derived.
        """
        self.nodes = nodes
        self.relationships = relationships
        self.source = source

    def __repr__(self):
        return f"GraphDocument(nodes={self.nodes}, relationships={self.relationships}, source={self.source})"


class LLMGraphTransformer:
    def __init__(self, api_key: str):
        """
        Initialize an LLMGraphTransformer object with an API key.

        Args:
            api_key (str): API key for accessing the language model service.
        """
        self.api_key = api_key

    def _generate_prompt(self, text: str) -> Tuple[str, str]:
        """
        Generate the system and human prompts for the language model.

        Args:
            text (str): The input text to be converted into a knowledge graph.

        Returns:
            Tuple[str, str]: The human and system prompts for the LLM.
        """
        system_prompt = """You are an AI that specializes in transforming text into structured knowledge graphs. Your task is to extract key entities (as nodes) and their relationships from the text provided. Each entity should be identified by a unique ID and categorized by type (e.g., Person, Country, Monarchy, etc.). Relationships between entities should be clearly defined with a type that describes their connection (e.g., REIGNED, PARENT, SIBLING, etc.). The output should be a JSON object with two main lists:
        1. `nodes`: A list of objects where each object represents an entity with the following structure:
           - `id`: A unique identifier for the entity (usually a name or title).
           - `type`: The category of the entity (e.g., Person, Country, Monarchy).
        2. `relationships`: A list of objects where each object represents a relationship between two entities with the following structure:
           - `source`: The `id` of the source entity.
           - `target`: The `id` of the target entity.
           - `type`: The type of relationship (e.g., REIGNED, PARENT, SIBLING).
        Ensure that the JSON output is properly formatted and free of errors.
        """
        human_prompt = f"Convert the following text into a knowledge graph by extracting entities and their relationships. Provide the output as a JSON object with 'nodes' and 'relationships' lists as described: {text}"
        return human_prompt, system_prompt

    def _call_openai_api(self, human_prompt: str, system_prompt: str) -> str:
        """
        Call the language model API with the given prompts.

        Args:
            human_prompt (str): The prompt for the language model.
            system_prompt (str): The system prompt with instructions for the model.

        Returns:
            str: The output from the language model.

        Raises:
            RuntimeError: If the API call fails or returns an error.
        """
        openai_model = IndoxApi(api_key=self.api_key)
        try:
            response = openai_model.chat(prompt=human_prompt, system_prompt=system_prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"Failed to call API: {e}")

    def _parse_llm_output(self, output: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse the JSON output from the language model.

        Args:
            output (str): The raw output from the language model.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The parsed JSON object containing nodes and relationships.

        Raises:
            ValueError: If the output is not valid JSON.
        """
        json_output = output.strip('```json').strip('```').strip()
        try:
            return json.loads(json_output)
        except json.JSONDecodeError:
            raise ValueError("LLM output is not valid JSON")

    def convert_to_graph(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert the input text into a graph representation.

        Args:
            text (str): The input text to be converted.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The graph representation with nodes and relationships.
        """
        human_prompt, system_prompt = self._generate_prompt(text)
        llm_output = self._call_openai_api(human_prompt, system_prompt)
        return self._parse_llm_output(llm_output)

    def process_document(self, text: str) -> GraphDocument:
        """
        Process a text document and create a GraphDocument object.

        Args:
            text (str): The input text document.

        Returns:
            GraphDocument: The resulting GraphDocument object.

        Raises:
            ValueError: If the output JSON does not match expected format.
        """
        graph_data = self.convert_to_graph(text)

        # Validate 'nodes' and 'relationships' in graph_data
        if not isinstance(graph_data, dict) or 'nodes' not in graph_data or 'relationships' not in graph_data:
            raise ValueError("Invalid format of LLM output. Expected 'nodes' and 'relationships' keys.")

        nodes_dict = {node['id']: Node(id=node['id'], type=node['type']) for node in graph_data['nodes']}

        relationships = [
            Relationship(
                source=nodes_dict[rel['source']],
                target=nodes_dict[rel['target']],
                type=rel['type']
            ) for rel in graph_data['relationships']
        ]

        return GraphDocument(nodes=list(nodes_dict.values()), relationships=relationships, source=text)

    def convert_to_graph_documents(
            self, documents: Sequence[str]
    ) -> List[GraphDocument]:
        """
        Convert a sequence of text documents into a list of GraphDocument objects.

        Args:
            documents (Sequence[str]): The original documents as strings.

        Returns:
            List[GraphDocument]: The transformed documents as GraphDocument objects.

        Raises:
            ValueError: If any document fails to be processed.
        """
        graph_documents = []
        for document in documents:
            try:
                graph_document = self.process_document(document)
                graph_documents.append(graph_document)
            except ValueError as e:
                print(f"Warning: Skipping document due to error: {e}")
        return graph_documents
