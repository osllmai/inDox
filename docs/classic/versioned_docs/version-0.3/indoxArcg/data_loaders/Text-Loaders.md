# Text Files

This guide covers basic text file processing capabilities in indoxArcg, focusing on simple text extraction and handling.

---

## Supported Loaders

### Txt Loader

**Best for**: Raw text file processing and ingestion

#### Features

- Encoding auto-detection
- Large file streaming
- Basic metadata tracking

```python
from indoxArcg.data_loaders import Txt

loader = Txt(encoding="utf-8", chunk_size=4096)
docs = loader.load("novel.txt")
```

---

## Key Capabilities

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| File Formats        | .txt, .log, .md               |
| Encoding Support    | UTF-8/16, ASCII, Latin-1      |
| Max File Size       | 10GB+ (with streaming)        |
| Metadata Extraction | File stats, basic headers     |
| Special Handling    | CR/LF normalization           |
| Performance         | High-speed sequential reading |

---

## Installation

```bash
pip install indoxArcg
```

---

## Basic Usage

### Simple Text Loading

```python
from indoxArcg.data_loaders import Txt

docs = Txt().load("document.txt")
```
