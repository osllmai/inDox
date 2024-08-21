from indox.data_connector.utils import Document
from typing import Any, List


class WikipediaReader:
    """Wikipedia reader that reads and processes Wikipedia pages.

    This class uses the `wikipedia` package to fetch and process Wikipedia pages. To use this
    class, ensure that the `wikipedia` package is installed.
    """

    def __init__(self) -> None:
        """Initialize WikipediaReader by importing the `wikipedia` module.

        Raises:
            ImportError: If the `wikipedia` package is not installed.
        """
        try:
            import wikipedia
        except ImportError:
            raise ImportError(
                "`wikipedia` package not found. Please install it using `pip install wikipedia`."
            )

    @classmethod
    def class_name(cls) -> str:
        """Get the name of the class.

        Returns:
            str: The name of the class, "WikipediaReader".
        """
        return "WikipediaReader"

    def load_data(self, pages: List[str], **load_kwargs: Any) -> List[Document]:
        """Load data from specified Wikipedia pages.

        Args:
            pages (List[str]): List of Wikipedia page titles to read.
            **load_kwargs: Additional keyword arguments to pass to the `wikipedia.page` method.

        Returns:
            List[Document]: A list of Document instances, each containing the content and
            metadata of a Wikipedia page.

        Raises:
            wikipedia.exceptions.DisambiguationError: If a page title is ambiguous and requires
                disambiguation.
            wikipedia.exceptions.PageError: If a page title does not correspond to any page.
            Exception: For any other unexpected exceptions during data retrieval.
        """
        import wikipedia

        documents = []
        for page in pages:
            try:
                wiki_page = wikipedia.page(page, **load_kwargs)
                page_content = wiki_page.content
                page_id = wiki_page.pageid
                metadata = {
                    "page_id": page_id,
                    "title": wiki_page.title,
                    "url": wiki_page.url,
                    "summary": wiki_page.summary,
                }
                documents.append(Document(source="Wikipedia", content=page_content, metadata=metadata))
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"DisambiguationError for page '{page}': {e.options}")
            except wikipedia.exceptions.PageError:
                print(f"PageError: The page '{page}' does not exist.")
            except Exception as e:
                print(f"Unexpected error while processing page '{page}': {e}")

        return documents
