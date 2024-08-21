# ArxivReader

ArxivReader is a data connector for loading paper information from the arXiv repository. It retrieves paper details such as title, abstract, authors, and metadata for given arXiv paper IDs.

**Note**: To use ArxivReader, users need to install the `arxiv` package. You can install it using `pip install arxiv`.

To use ArxivReader:

```python
from indox.data_connector import ArxivReader

reader = ArxivReader()
documents = reader.load_data(paper_ids=["1234.56789"])
```
## Methods 
**init()**
Initializes the ArxivReader. Checks if the `arxiv` package is installed.
**class_name()**
Returns the name of the class as a string.

**load_data(paper_ids: List[str], load_kwargs: Any) -> List[Document]**
Loads paper data from arXiv for the given paper IDs.

**Parameters:**
- **paper_ids** [List[str]]: List of arXiv paper IDs to retrieve.
- **load_kwargs**[Any]:  Additional keyword arguments (not used in current implementation).

- **Returns:**
  - **List[Document]**: List of Document objects containing paper content and metadata.
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
## Get started
### Import Essential Libraries
```python
from indox.data_connector import ArxivReader

reader = ArxivReader()

paper_ids = ["2201.08239", "2203.02155"]
documents = reader.load_data(paper_ids)

for doc in documents:
    print(f"Title: {doc.metadata['title']}")
    print(f"Authors: {doc.metadata['authors']}")
    print(f"Abstract: {doc.content[:200]}...") 
    print(f"arXiv URL: {doc.metadata['arxiv_url']}")
    print("---")
```
This example demonstrates how to use ArxivReader to retrieve information about specific arXiv papers and access their content and metadata.


