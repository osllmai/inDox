# Text Loaders in indoxArcg

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

| Feature               | Description                            |
|-----------------------|----------------------------------------|
| File Formats          | .txt, .log, .md                        |
| Encoding Support      | UTF-8/16, ASCII, Latin-1               |
| Max File Size         | 10GB+ (with streaming)                 |
| Metadata Extraction   | File stats, basic headers              |
| Special Handling      | CR/LF normalization                    |
| Performance           | High-speed sequential reading          |

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

### Large File Handling
```python
loader = Txt(buffer_size=1048576)  # 1MB chunks
docs = loader.load("large_log.log")
```

---

## Advanced Configuration

### Custom Metadata
```python
def text_metadata(file_path, content):
    return {
        "line_count": len(content.split('\n')),
        "language": detect_language(content)
    }

loader = Txt(metadata_fn=text_metadata)
```

### Pattern-based Processing
```python
loader = Txt(
    preprocess=lambda text: re.sub(r'\s+', ' ', text),  # Remove extra whitespace
    skip_lines_matching=r'^#.*'  # Skip comment lines
)
```

---

## Troubleshooting

### Common Issues
1. **Encoding Errors**
   ```python
   Txt(encoding="latin-1").load("legacy.txt")
   ```

2. **Memory Constraints**
   ```python
   Txt(stream=True, chunk_size=524288).load("huge_file.txt")
   ```

3. **Line Ending Conflicts**
   ```python
   Txt(normalize_newlines=True).load("mixed_endings.txt")
   ```

4. **Binary File Detection**
   ```python
   Txt(strict_text_check=True).load("possibly_binary.dat")
   ```

---
