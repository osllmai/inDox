from indoxRag.data_connectors.utils import Document
from typing import Any, List, Optional


class ArxivReader:
    """
    A reader class to access papers from the arXiv repository.
    """

    def __init__(self) -> None:
        """Initializes the ArxivReader object.

        Raises:
            ImportError: If the `arxiv` library is not installed.
        """
        try:
            import arxiv
        except ImportError:
            raise ImportError(
                "`arxiv` package not found, please run `pip install arxiv`"
            )

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name ("ArxivReader")."""
        return "ArxivReader"

    def load_data(
        self, paper_ids: List[str], **load_kwargs: Any
    ) -> List[Document] | Document:
        """Loads data from arXiv for the provided paper IDs.

        Args:
            paper_ids (List[str]): A list of arXiv paper IDs to read.

        Returns:
            List[Document]: A list of Document objects containing paper content and metadata.

        Raises:
            ValueError: If a paper ID is not found in the arXiv search results.
        """
        import arxiv

        documents = []
        for paper_id in paper_ids:
            try:
                search = arxiv.Search(id_list=[paper_id])
                paper = next(search.results())
            except StopIteration:
                raise ValueError(
                    f"Paper ID '{paper_id}' not found in arXiv search results."
                )

            # Extract paper information
            title = paper.title
            abstract = paper.summary
            authors = ", ".join(author.name for author in paper.authors)

            # Combine information into content
            content = f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}"

            # Create metadata
            metadata = {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "published": paper.published,
                "arxiv_url": paper.entry_id,
            }

            # Create Document instance
            documents.append(
                Document(source="arXiv", content=content, metadata=metadata)
            )
        if len(documents) == 1:
            return documents[0]

        return documents

    def load_content(self, paper_ids: List[str], **load_kwargs: Any) -> List[str] | str:
        """Loads paper content from arXiv for the provided paper IDs.

        Args:
            paper_ids (List[str]): A list of arXiv paper IDs to read.

        Returns:
            List[str]: A list of strings containing paper content (title, authors, abstract).

        Raises:
            ValueError: If a paper ID is not found in the arXiv search results.
        """
        import arxiv

        contents = []
        for paper_id in paper_ids:
            try:
                search = arxiv.Search(id_list=[paper_id])
                paper = next(search.results())
            except StopIteration:
                raise ValueError(
                    f"Paper ID '{paper_id}' not found in arXiv search results."
                )

            # Extract paper information
            title = paper.title
            abstract = paper.summary
            authors = ", ".join(author.name for author in paper.authors)

            # Combine information into content
            content = f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}"

            # Append content to list
            contents.append(content)
        if len(contents) == 1:
            return contents[0]

        return contents
