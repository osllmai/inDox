from typing import Any, Dict, Literal


class Document:
    """Class for storing a piece of text and associated metadata."""

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Initialize the Document with page content and arbitrary metadata."""
        self.page_content = page_content
        self.metadata: Dict[str, Any] = kwargs
        self.type: Literal["Document"] = "Document"

    def __repr__(self) -> str:
        return f"Document(page_content={self.page_content}, metadata={self.metadata})"
