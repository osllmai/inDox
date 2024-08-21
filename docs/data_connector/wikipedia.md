# WikipediaReader

WikipediaReader is a data connector for loading content from Wikipedia pages. It retrieves page content, summaries, and metadata for specified Wikipedia pages.

**Note**: To use WikipediaReader, users need to install the `wikipedia` package. You can install it using `pip install wikipedia`.

To use WikipediaReader:

```python
from indox.data_connector import WikipediaReader

reader = WikipediaReader()
documents = reader.load_data(pages=["Python (programming language)", "Artificial intelligence"])
```

## Methods 

**__init__()**

Initializes the WikipediaReader and checks if the `wikipedia` package is installed.

**class_name()**

Returns the name of the class as a string.

**load_data(pages: List[str], load_kwargs: Any) -> List[Document]**

Loads data from the specified Wikipedia pages.

**Parameters:**
- **pages** [List[str]]: List of Wikipedia page titles to retrieve.
- **load_kwargs** [Any]: Additional keyword arguments passed to `wikipedia.page()`.

**Returns:**
- **List[Document]**: List of Document objects containing page content and metadata.

## Usage
### Setting Up the Python Environment
**Windows**
1. **Create the virtual environment:**
```bash
python -m venv indox
```
2. **Activate the virtual environment:**
```bash
indox\Scripts\activate
```
### macOS/Linux
1. **Create the virtual environment:**
```bash
python -m venv indox
```
2. **Activate the virtual environment:**
```bash
source indox/bin/activate
```

## Get Started
### Import Essential Libraries and Use WikipediaReader
```python
from indox.data_connector import WikipediaReader

# Initialize the reader
reader = WikipediaReader()

# Fetch content from specific Wikipedia pages
pages = ["Python (programming language)", "Artificial intelligence"]
documents = reader.load_data(pages)

# Process the retrieved documents
for doc in documents:
    print(f"Title: {doc.metadata['title']}")
    print(f"URL: {doc.metadata['url']}")
    print(f"Summary: {doc.metadata['summary'][:200]}...")
    print(f"Content preview: {doc.content[:200]}...")
    print("---")
```
This example demonstrates how to use WikipediaReader to retrieve content from specific Wikipedia pages and access their content and metadata.   