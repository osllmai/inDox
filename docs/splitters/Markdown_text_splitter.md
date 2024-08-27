# MarkdownTextSplitter

MarkdownTextSplitter is a Python class designed for splitting Markdown-formatted text into chunks. It extends the RecursiveCharacterTextSplitter to specifically handle Markdown-formatted text, splitting along headings and other Markdown-specific separators.

**Note**: This class is part of the `indox.splitter` module and requires the `RecursiveCharacterTextSplitter` as a base class. Ensure you have the necessary dependencies installed before using this class.

To use MarkdownTextSplitter:

```python
from indox.splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50)

```

## Hyperparameters
MarkdownTextSplitter inherits all parameters from RecursiveCharacterTextSplitter. Key parameters include:

- **chunk_size** [int]: The maximum size of each text chunk.
- **chunk_overlap** [int]: The number of characters to overlap between chunks.
- **length_function** [Callable[[str],int]]: Function to measure the length of given text.

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

### Import the MarkdownTextSplitter
```python
from indox.splitter import MarkdownTextSplitter
```
### Initialize MarkdownTextSplitter
```python
splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50)
```
### Split And Processing Chunks
```python
markdown_text = """
# Main Heading

## Subheading 1
Content for subheading 1...

## Subheading 2
Content for subheading 2...

***

Some more content with horizontal rule above.
"""

chunks = splitter.split_text(markdown_text)

print(f"The text has been split into {len(chunks)} chunks.")
for chunk in chunks:
    print(chunk)
    print("====")
```