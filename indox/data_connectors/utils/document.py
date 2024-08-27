import uuid
from typing import Dict, Any, Optional

class Document:
    """Represents a data document from various sources.

    Attributes:
        id_: A unique identifier for the document.
        source: The source of the document, e.g., YouTube, Wikipedia.
        content: The actual content of the document.
        metadata: Additional metadata associated with the document.
    """

    id_: str = uuid.uuid4().hex  # Unique ID for each document
    source: str  # The source of the document, e.g., YouTube, Wikipedia
    content: str  # The actual content of the document
    metadata: Dict[str, Any]  # Metadata related to the document

    def __init__(self, source: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Initializes a new Document object.

        Args:
            source: The source of the document.
            content: The content of the document.
            metadata: Optional metadata associated with the document.

        Raises:
            TypeError: If the source or content is not a string.
            ValueError: If the source or content is empty.
        """

        if not isinstance(source, str) or not source:
            raise ValueError("Source must be a non-empty string.")
        if not isinstance(content, str) or not content:
            raise ValueError("Content must be a non-empty string.")

        self.source = source
        self.content = content
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """Returns a string representation of the document."""
        return f"Doc ID: {self.id_}\nSource: {self.source}\nContent: {self.content}\n"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the document to a dictionary representation.

        Returns:
            A dictionary containing the document's attributes.
        """
        return {
            "doc_id": self.id_,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Creates a Document object from a dictionary.

        Args:
            data: A dictionary containing the document's attributes.

        Raises:
            KeyError: If the dictionary is missing required keys.
            TypeError: If the dictionary values are not of the expected types.
            ValueError: If the source or content is empty.

        Returns:
            A new Document object.
        """

        try:
            source = data["source"]
            content = data["content"]
            metadata = data.get("metadata", {})

            if not isinstance(source, str) or not source:
                raise ValueError("Source must be a non-empty string.")
            if not isinstance(content, str) or not content:
                raise ValueError("Content must be a non-empty string.")

            return cls(source=source, content=content, metadata=metadata)
        except KeyError as e:
            raise KeyError(f"Missing required key: {e.args[0]}")