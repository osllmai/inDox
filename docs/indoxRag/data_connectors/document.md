# Document

Document is a class that represents a data document from various sources. It provides a structured way to store and manage content along with its metadata.

To use the Document class:

```python
from indoxRag.data_connectors import Document

# Create a new document
doc = Document(source="Wikipedia", content="Wikipedia is a free online encyclopedia.")

# Access document attributes
print(doc.id_)
print(doc.source)
print(doc.content)
print(doc.metadata)

# Convert to dictionary
doc_dict = doc.to_dict()

# Create from dictionary
new_doc = Document.from_dict(doc_dict)
```

# Class Attributes

- **id\_** [str]: A unique identifier for the document (automatically generated).
- **source** [str]: The source of the document (e.g., YouTube, Wikipedia).
- **content** [str]: The actual content of the document.
- **metadata** [Dict[str, Any]]: Additional metadata associated with the document.

**init(source: str, content: str, metadata: Optional[Dict[str, Any]] = None):**

Initializes a new Document object.

**Parameters:**

- **source** [str]: The source of the document.
- - **content** [str]: The content of the document.
- **metadata** [Optional[Dict[str, Any]]]: Optional metadata associated with the document.

**Raises:**

- **TypeError:** If the source or content is not a string.
- **ValueError:** If the source or content is empty.

****str**() -> str:**

Returns a string representation of the document.

**to_dict() -> Dict[str, Any]:**

Converts the document to a dictionary representation.

**Returns:**

- **Dict[str, Any]**: A dictionary containing the document's attributes.

**from_dict(cls, data: Dict[str, Any]) -> Document:**

Creates a Document object from a dictionary.

**Parameters:**

- **data** [Dict[str, Any]]: A dictionary containing the document's attributes.

**Returns:**

- **Document**: A new Document object.

**Raises:**

- **KeyError**: If the dictionary is missing required keys.
- **TypeError**: If the dictionary values are not of the expected types.
- **ValueError**: If the source or content is empty.

## Usage

```python
from indoxRag.data_connectors import Document

# Create a new document
doc = Document(
    source="Wikipedia",
    content="Wikipedia is a free online encyclopedia.",
    metadata={"language": "English", "accessed_date": "2024-08-20"}
)

# Access document attributes
print(f"Document ID: {doc.id_}")
print(f"Source: {doc.source}")
print(f"Content: {doc.content}")
print(f"Metadata: {doc.metadata}")

# Convert to dictionary
doc_dict = doc.to_dict()
print("Document as dictionary:", doc_dict)

# Create a new document from dictionary
new_doc = Document.from_dict(doc_dict)
print("New document:", new_doc)

# String representation
print(str(doc))
```

This example demonstrates how to create Document objects, access their attributes, convert them to and from dictionaries, and get their string representations.
