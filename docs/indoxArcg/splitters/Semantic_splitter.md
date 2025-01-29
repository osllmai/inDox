# SemanticTextSplitter

SemanticTextSplitter is a Python class designed for splitting text into semantically meaningful chunks using a BERT tokenizer. It utilizes the semantic_text_splitter library to split input text into chunks that preserve semantic meaning while ensuring each chunk does not exceed a specified maximum number of tokens.

**Note**: This class requires the semantic_text_splitter and tokenizers libraries. Ensure you have these dependencies installed before using this class.

To use SemanticTextSplitter:

```python
from indoxArcg.splitter import SemanticTextSplitter

splitter = SemanticTextSplitter(chunk_size=400, model_name="bert-base-uncased")
```

## Hyperparameters

- **chunk_size** [int]: The maximum number of tokens allowed in each chunk (default: 400).
- **model_name** [str]: The name of the pre-trained model to use for the tokenizer (default: "bert-base-uncased").

## Usage

### Setting Up the Python Environment

### Windows

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2**Activate the virtual environment:**

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

### Install Required Libraries

```bash
pip install semantic-text-splitter tokenizers
```

### Import the SemanticTextSplitter

```python
from indoxArcg.splitter import SemanticTextSplitter
```

### Initialize SemanticTextSplitter

```python
splitter = SemanticTextSplitter(chunk_size=400, model_name="bert-base-uncased")
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
