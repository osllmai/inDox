# GutenbergReader

GutenbergReader is a data connector for fetching and searching books from Project Gutenberg. It provides functionality to retrieve full book content by ID and search for books based on a query.

**Note**: To use GutenbergReader, users need to install the `requests` and `beautifulsoup4` packages. You can install them using `pip install requests beautifulsoup4`.

To use GutenbergReader:

```python
from indoxArcg.data_connectors import GutenbergReader

reader = GutenbergReader()

# Fetch a book by ID
book = reader.get_book("11")

# Search for books
search_results = reader.search_gutenberg("Alice in Wonderland")
```

# Class Attributes

- **BASE_URL** [str]: The base URL for Project Gutenberg files.

## Methods

**init()**
Initializes the `GutenbergReader` and creates a requests session.

**get_book(book_id: str) -> Optional[Document]**

Fetches a book from Project Gutenberg by its ID.

**Parameters:**

- **book_id** [str]: The ID of the book on Project Gutenberg.

**Returns:**

- **Optional[Document]**: A Document instance containing the book's title, text content, and metadata, or None if the fetch fails.

**search_gutenberg(query: str) -> List[Document]**
Searches for books on Project Gutenberg based on a query string.
**Parameters:**

- **query [str]**: Search query string.

**Returns:**

- **List[Document]**: List of Document instances containing book info (id, title, author) for search results.

## Private Methods

**\_extract_title(content: str) -> str**
Extracts the title of the book from its content.
**\_extract_text(content: str) -> str**
Extracts the main text content of the book.

## Usage

### Setting Up the Python Environment

**Windows**

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
indoxArcg\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
source indoxArcg/bin/activate
```

## Get started

### Import Essential Libraries

```python
from indoxArcg.data_connectors import GutenbergReader

# Initialize the reader
reader = GutenbergReader()

# Fetch a specific book by ID
book_id = "11"  # Alice's Adventures in Wonderland
book = reader.get_book(book_id)

if book:
    print(f"Title: {book.metadata['title']}")
    print(f"Content preview: {book.content[:200]}...")
    print("---")

# Search for books
search_query = "Sherlock Holmes"
search_results = reader.search_gutenberg(search_query)

for result in search_results[:5]:  # Print first 5 results
    print(f"Book ID: {result.metadata['book_id']}")
    print(f"Title: {result.metadata['title']}")
    print(f"Author: {result.metadata['author']}")
    print("---")
```

This example demonstrates how to use GutenbergReader to fetch a specific book and search for books on Project Gutenberg.
