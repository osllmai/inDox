# CharacterTextSplitter

CharacterTextSplitter is a Python class designed for splitting text into chunks based on a specified separator. It implements an algorithm to split text into chunks of a specified size, with an optional overlap between chunks.

**Note**: This class is part of a text splitting module. Ensure you have the necessary dependencies installed before using this class.

To use CharacterTextSplitter:

```python
from indoxRag.splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
```

## Hyperparameters

- **separator** [str]: The string used to split the text (default: "\n\n").
- **chunk_size** [int]: The maximum size of each chunk (default: 400).
- **chunk_overlap** [int]: The number of characters to overlap between chunks (default: 50).
- **length_function** [callable]: A function used to calculate the length of text (default: len).

## Usage

### Setting Up the Python Environment

### Windows

1. **Create the virtual environment:**

```bash
python -m venv indoxRag
```

2**Activate the virtual environment:**

```bash
indoxRag\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
python -m venv indoxRag
```

2. **Activate the virtual environment:**

```bash
source indoxRag/bin/activate
```

### Import the CharacterTextSplitter

```python
from indoxRag.splitter import CharacterTextSplitter
```

### Initialize CharacterTextSplitter

```python
splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
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
