import requests
from bs4 import BeautifulSoup
import re
from typing import List, Optional
from indox.data_connector.utils import Document

class GutenbergReader:
    BASE_URL = "https://www.gutenberg.org/files"

    def __init__(self):
        """
        Initializes the GutenbergReader with a requests session.
        """
        self.session = requests.Session()

    def get_book(self, book_id: str) -> Optional[Document]:
        """
        Fetch a book from Project Gutenberg by its ID.

        :param book_id: The ID of the book on Project Gutenberg.
        :return: A Document instance containing the book's title, text content, and metadata, or None if the book cannot be fetched.
        """
        url = f"{self.BASE_URL}/{book_id}/{book_id}-0.txt"
        try:
            response = self.session.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx and 5xx)
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch book with ID {book_id}. Error: {str(e)}")
            return None

        content = response.text
        title = self._extract_title(content)
        text = self._extract_text(content)

        return Document(
            source="Project Gutenberg",
            content=text,
            metadata={
                "book_id": book_id,
                "title": title,
            }
        )

    def get_content(self, book_id: str) -> Optional[str]:
        """
        Fetch a book from Project Gutenberg by its ID.

        :param book_id: The ID of the book on Project Gutenberg.
        :return: The text content of the book, or None if the book cannot be fetched.
        """
        url = f"{self.BASE_URL}/{book_id}/{book_id}-0.txt"
        try:
            response = self.session.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx and 5xx)
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch book with ID {book_id}. Error: {str(e)}")
            return None

        content = response.text
        text = self._extract_text(content)

        return text

    def _extract_title(self, content: str) -> str:
        """
        Extract the title of the book from its content.

        :param content: The full text content of the book.
        :return: The extracted title, or "Unknown Title" if not found.
        """
        title_match = re.search(r"Title: (.+)\n", content)
        if title_match:
            return title_match.group(1).strip()
        return "Unknown Title"

    def _extract_text(self, content: str) -> str:
        """
        Extract the main text content of the book.

        :param content: The full text content of the book.
        :return: The extracted text content between the start and end markers, or the full content if markers are not found.
        """
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

        start_index = content.find(start_marker)
        end_index = content.find(end_marker)

        if start_index != -1 and end_index != -1:
            text = content[start_index:end_index].strip()
            text = text.split('\n', 1)[1] if '\n' in text else text
            return text
        else:
            return content

    def search_gutenberg(self, query: str) -> List[Document]:
        """
        Search for books on Project Gutenberg.

        :param query: Search query string.
        :return: List of Document instances containing book information (ID, title, author), or an empty list if the search fails.
        """
        search_url = f"https://www.gutenberg.org/ebooks/search/?query={query}"
        try:
            response = self.session.get(search_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Search failed. Error: {str(e)}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        documents = []
        for book in soup.select('li.booklink'):
            link = book.select_one('a')
            if not link or 'href' not in link.attrs:
                continue

            book_id = link['href'].split('/')[-1]
            title_elem = book.select_one('span.title')
            title = title_elem.text.strip() if title_elem else "Unknown Title"
            author_elem = book.select_one('span.subtitle')
            author = author_elem.text.strip() if author_elem else "Unknown Author"

            # Create a Document object for each search result
            documents.append(Document(
                source="Project Gutenberg",
                content=f"Title: {title}\nAuthor: {author}",
                metadata={
                    "book_id": book_id,
                    "title": title,
                    "author": author,
                }
            ))

        return documents
