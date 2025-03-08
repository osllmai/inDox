# Structured Data Loaders in indoxArcg

This guide covers structured data format processors supported in indoxArcg, organized by file type and processing capability.

---

## Supported Loaders

### 1. CSV Loader
**Best for**: Tabular data processing and row-based analysis

#### Features
- Row-wise document generation
- Custom metadata injection
- Encoding auto-detection

```python
from indoxArcg.data_loaders import CSV

loader = CSV(delimiter=";", skip_rows=1)
docs = loader.load("data.csv")
```

---

### 2. JSON Loader
**Best for**: Nested data structures and API responses

#### Features
- Key-value pair extraction
- Nested field flattening
- Size-aware chunking

```python
from indoxArcg.data_loaders import Json

loader = Json(max_depth=3, flatten_nested=True)
docs = loader.load("api_response.json")
```

---

### 3. SQL Loader
**Best for**: Database query files and schema analysis

#### Features
- Query validation
- Parameterized query support
- Execution plan metadata

```python
from indoxArcg.data_loaders import Sql

loader = Sql(include_execution_plan=True)
docs = loader.load("query.sql")
```

---

### 4. MD Loader
**Best for**: Markdown documentation processing

#### Features
- Section-aware splitting
- Frontmatter extraction
- Code block preservation

```python
from indoxArcg.data_loaders import Md

loader = Md(extract_frontmatter=True)
docs = loader.load("documentation.md")
```

---

## Comparison Table

| Feature               | CSV          | JSON         | SQL          | MD           |
|-----------------------|--------------|--------------|--------------|--------------|
| Nested Data Support   | ❌           | ✅           | ❌           | ❌           |
| Large File Handling   | Streaming    | Chunking     | ❌           | ❌           |
| Metadata Extraction   | Column Headers | Key Paths   | Query Stats  | Frontmatter  |
| Max File Size         | 10GB         | 2GB          | 100MB        | 50MB         |
| Encoding Support      | Auto-detect  | UTF-8/16     | UTF-8        | UTF-8        |

---

## Installation

```bash
pip install indoxArcg[structured]
# Or individual packages
pip install pandas python-jsonlogger sqlparse markdown
```

---

## Basic Usage

### CSV Processing
```python
from indoxArcg.data_loaders import CSV

docs = CSV(include_headers=True).load("dataset.csv")
```

### JSON Handling
```python
from indoxArcg.data_loaders import Json

loader = Json(json_path="$.items[*]")  # JSONPath support
docs = loader.load("nested_data.json")
```

### SQL Analysis
```python
from indoxArcg.data_loaders import Sql

docs = Sql(highlight_syntax=True).load("schema.sql")
```

### Markdown Processing
```python
from indoxArcg.data_loaders import Md

docs = Md(split_sections=True).load("README.md")
```

---

## Advanced Configuration

### Custom CSV Metadata
```python
def csv_metadata(row):
    return {"row_id": row["id"], "source": "legacy_system"}

loader = CSV(metadata_fn=csv_metadata)
```

### JSON Streaming
```python
from indoxArcg.data_loaders import Json

loader = Json(
    stream=True,
    chunk_size=1000  # Process 1000 items at a time
)
```

### SQL Parameterization
```python
from indoxArcg.data_loaders import Sql

loader = Sql(
    parameters={"date": "2024-01-01"},
    dialect="postgresql"
)
```

---

## Troubleshooting

### Common Issues
1. **CSV Encoding Errors**
   ```python
   CSV(encoding="latin-1").load("legacy.csv")
   ```

2. **Malformed JSON**
   ```python
   Json(strict=False).load("partial_data.json")
   ```

3. **Large SQL Files**
   ```python
   Sql(chunk_queries=True).load("large_dump.sql")
   ```

4. **Markdown Formatting**
   ```python
   Md(clean_extra_spaces=True).load("messy.md")
   ```

---
