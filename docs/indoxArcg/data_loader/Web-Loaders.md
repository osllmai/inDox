# Web & Rich Text Loaders in indoxArcg

This guide covers web content and rich text format processors supported in indoxArcg, organized by format type and extraction capability.

---

## Supported Loaders

### 1. HTML Loader (BeautifulSoup)
**Best for**: Web content extraction and DOM analysis

#### Features
- Tag-aware content extraction
- Automatic encoding detection
- Clean HTML formatting

```python
from indoxArcg.data_loaders import Bs4

loader = Bs4(exclude_tags=["script", "style"])
docs = loader.load("webpage.html")
```

---

### 2. RTF Loader
**Best for**: Legacy rich text document processing

#### Features
- Basic formatting preservation
- Font style recognition
- Simple metadata extraction

```python
from indoxArcg.data_loaders import Rtf

loader = Rtf(preserve_formatting=True)
docs = loader.load("document.rtf")
```

---

## Comparison Table

| Feature               | HTML Loader     | RTF Loader     |
|-----------------------|-----------------|----------------|
| Text Accuracy         | ★★★★★           | ★★★☆☆          |
| Format Preservation   | Semantic        | Basic Styles   |
| Encoding Handling     | Auto-detect     | Latin-1 Default|
| Metadata Extraction   | DOM Structure   | Limited        |
| Max File Size         | 100MB           | 50MB           |
| Complex Layouts       | Tables/CSS      | Simple Formatting |

---

## Installation

```bash
pip install beautifulsoup4 pyth indoxArcg
```

---

## Basic Usage

### HTML Processing
```python
from indoxArcg.data_loaders import Bs4

# Extract specific elements
loader = Bs4(include_selectors=[".article-content", "#main-heading"])
docs = loader.load("news.html")
```

### RTF Processing
```python
from indoxArcg.data_loaders import Rtf

# Preserve basic formatting
docs = Rtf(keep_bold=True, keep_italic=True).load("legacy.rtf")
```

---

## Advanced Configuration

### Custom HTML Metadata
```python
def html_metadata(soup):
    return {
        "title": soup.title.string,
        "word_count": len(soup.get_text().split())
    }

loader = Bs4(metadata_fn=html_metadata)
```

### RTF Format Conversion
```python
from indoxArcg.data_loaders import Rtf

loader = Rtf(
    conversion_mode="markdown",  # Convert RTF to Markdown
    image_dir="./rtf_images"
)
```

---

## Troubleshooting

### Common Issues
1. **HTML Encoding Errors**
   ```python
   Bs4(encoding="windows-1252").load("legacy_page.html")
   ```

2. **Malformed RTF Files**
   ```python
   Rtf(strict_parsing=False).load("corrupted.rtf")
   ```

3. **Missing Web Content**
   ```python
   Bs4(retry_count=3, timeout=10).load("https://example.com")
   ```

4. **Formatting Loss**
   ```python
   Rtf(preserve_layout=True).load("formatted.rtf")
   ```

---
