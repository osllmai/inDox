# RecursiveCharacterTextSplitter

RecursiveCharacterTextSplitter is a Python class designed for splitting text into chunks recursively based on specified separators. It implements a recursive algorithm to split text into chunks of a specified size, with an optional overlap between chunks.

**Note**: This class is part of the `indox.splitter` module. Ensure you have the necessary dependencies installed before using this class.

To use RecursiveCharacterTextSplitter:

```python
from indox.splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
```

## Hyperparameters

- **chunk_size** [int]: The maximum size of each chunk (default: 400).
- **chunk_overlap** [int]: The number of characters to overlap between chunks (default: 50).
- **separators** [Optional[List[str]]]: A list of separators to use for splitting the text (default: ["\n\n", "\n", ". ", " ", ""]).

## Usage
### Setting Up the Python Environment
### Windows

1. **Create the virtual environment:**
```bash
python -m venv indox
```
2**Activate the virtual environment:**
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

### Import the RecursiveCharacterTextSplitter
```python
from indox.splitter import RecursiveCharacterTextSplitter
```
### Initialize RecursiveCharacterTextSplitter
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
```
### Split And Processing Chunks
```python
text = """
This is a long piece of text that needs to be split into smaller chunks.
It contains multiple sentences and paragraphs.

Here's another paragraph with some content.

And one more paragraph to demonstrate the splitting process.
"""

chunks = splitter.split_text(text)

print(f"The text has been split into {len(chunks)} chunks.")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(chunk)
    print("====")
```