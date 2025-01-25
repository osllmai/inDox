# Neo4jGraph

The Neo4jGraph class allows you to use Neo4j as a graph database to store and query graph-structured data. This can be used in a Retrieval-Augmented Generation (RAG) system or other knowledge graph-related applications.

# Hyperparameters

- **uri** (str): The URI of the Neo4j instance (e.g., "bolt://localhost:7687").
- **username** (str): The username for authenticating to the Neo4j database.
- **password** (str): The password for authenticating to the Neo4j database.

# Installation

To use Neo4j with this package, you first need to install the `Neo4j` driver. You can install it using pip:

```python
pip install neo4j
```

You will also need a running Neo4j instance. Refer to the official Neo4j installation guide for instructions on setting up a Neo4j instance locally or in the cloud.

# Example Usage

## Initialize the Neo4jGraph

You can initialize the Neo4jGraph class by passing the Neo4j connection details.

```python
from indoxArcg.vector_stores import Neo4jGraph

# Initialize the Neo4jGraph with connection details
neo4j_graph = Neo4jGraph(uri="bolt://localhost:7687", username="neo4j", password="your_password")
```

## Add Graph Documents to Neo4j

To add graph documents (which contain nodes and relationships) to the Neo4j database, you can use the `add_graph_documents` method. This method allows you to store a list of graph documents with options to include source metadata and entity labels.

```python
# Assuming `graph_documents` is a list of GraphDocument objects you want to store
neo4j_graph.add_graph_documents(graph_documents, base_entity_label=True, include_source=True)
```

- **graph_documents**: A list of GraphDocument objects to store in the Neo4j database.
- **base_entity_label** (bool, optional): If True, adds a base "Entity" label to all nodes. Defaults to True.
- **include_source** (bool, optional): If True, includes the source document as part of the graph. Defaults to True.

```python
# Query Relationships by Entity
To query relationships in Neo4j, you can use the `search_relationships_by_entity` method. This method allows you to search for relationships by specifying the entity ID and the relationship type.

# Search for parent relationships for the entity with ID "Elizabeth_I"
relationships = neo4j_graph.search_relationships_by_entity(entity_id="Elizabeth_I", relationship_type="PARENT")

# Print the found relationships
for rel in relationships:
    print(f"{rel['a']['id']} is a {rel['rel_type']} of {rel['b']['id']}")
```

- **entity_id** (str): The ID of the entity for which you want to find relationships.
- **relationship_type** (str): The type of relationship to search for (e.g., "PARENT").

## Close the Connection

After performing your operations, always remember to close the Neo4j connection:

```python
neo4j_graph.close()
```

# Example Workflow

Here's an example of how to use `Neo4jGraph` to store and query relationships in Neo4j:

```python
from neo4j_graph import Neo4jGraph

# Initialize Neo4jGraph
neo4j_graph = Neo4jGraph(uri="bolt://localhost:7687", username="neo4j", password="your_password")

# Assuming you have a list of GraphDocument objects ready to store
graph_documents = [...]  # List of GraphDocument objects

# Add the graph documents to Neo4j
neo4j_graph.add_graph_documents(graph_documents, base_entity_label=True, include_source=True)

# Query relationships of an entity
relationships = neo4j_graph.search_relationships_by_entity(entity_id="Elizabeth_I", relationship_type="PARENT")

# Output the relationships
for rel in relationships:
    print(f"{rel['a']['id']} is a {rel['rel_type']} of {rel['b']['id']}")

# Close the Neo4j connection
neo4j_graph.close()
```

# FAQ

**Q : What do I need to use `Neo4jGraph`?**

- You need a running Neo4j database and the Neo4j Python driver installed.

**Q: Can I use this with an existing Neo4j database?**

- Yes, you can connect to any Neo4j instance by providing the correct URI, username, and password.
