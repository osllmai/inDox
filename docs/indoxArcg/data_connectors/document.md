# Document Class Reference

The `Document` class is the fundamental data structure for representing content and metadata in indoxArcg's data pipeline. It serves as the standardized format for:
- Input from data connectors
- Processing in RAG/CAG workflows
- Storage in vector databases

```python
from indoxArcg.data_connectors import Document
```

---

## Core Attributes

| Attribute  | Type                | Description                                  | Example                      |
|------------|---------------------|----------------------------------------------|------------------------------|
| `id_`      | `str`               | Auto-generated UUIDv4 identifier             | "550e8400-e29b-41d4-a716-446655440000" |
| `source`   | `str`               | Origin system/format identifier              | "wikipedia", "youtube_transcript" |
| `content`  | `str`               | Primary textual content (UTF-8 encoded)      | "Large language models..."    |
| `metadata` | `Dict[str, Any]`    | Contextual information about the content     | `{"author": "John Doe", "timestamp": "2024-03-15"}` |

---

## Key Methods

### `__init__(source: str, content: str, metadata: Optional[Dict[str, Any]] = None)`
**Initializes a new Document instance**

Parameters:
- `source` (str): Origin identifier for tracking document provenance  
  - **Format**: `[system]_[type]` (e.g., `arxiv_paper`, `youtube_transcript`)
  - **Required**: Yes
- `content` (str): Main textual payload (minimum 10 characters)
  - **Required**: Yes
- `metadata` (dict): Additional context (default: empty dict)

```python
doc = Document(
    source="arxiv_paper",
    content="We present a novel approach...",
    metadata={
        "doi": "10.1234/abcd.56789",
        "authors": ["Smith, J.", "Lee, R."],
        "publication_date": "2024-03-01"
    }
)
```

---

### Serialization Methods

#### `to_dict() -> Dict[str, Any]`
Converts document to portable dictionary format

```python
doc_dict = doc.to_dict()
# {
#     "id_": "550e8400-e29b-41d4-a716-446655440000",
#     "source": "arxiv_paper",
#     "content": "We present...",
#     "metadata": {...}
# }
```

#### `from_dict(data: Dict[str, Any]) -> Document`
Reconstructs document from dictionary representation

```python
reconstructed_doc = Document.from_dict(doc_dict)
```

---

## Usage Guidelines

### Best Practices
1. **Source Identification**
   ```python
   # Good - specific source
   Document(source="wikipedia_llm_article", ...)
   
   # Avoid - vague source
   Document(source="website", ...)
   ```

2. **Metadata Standards**
   ```python
   recommended_metadata = {
       "author": str,          # Content creator
       "created_date": str,    # ISO 8601 format
       "source_url": str,      # Original location
       "language": str,        # BCP-47 code
       "confidence": float     # 0.0-1.0 for AI-generated content
   }
   ```

3. **Content Validation**
   ```python
   # Minimum viable content check
   if len(doc.content) < 10:
       raise ValueError("Content too short for meaningful processing")
   ```

---

## Advanced Usage

### Batch Processing
```python
def process_documents(docs: List[Document]) -> List[Document]:
    """Add processing timestamp to all documents"""
    return [
        Document(
            source=doc.source,
            content=doc.content,
            metadata={**doc.metadata, "processed_at": datetime.now().isoformat()}
        )
        for doc in docs
    ]
```

### Error Handling
```python
try:
    doc = Document(source="", content="Invalid document") 
except ValueError as e:
    print(f"Validation error: {e}")

invalid_dict = {"source": "test", "content": ""}
try:
    Document.from_dict(invalid_dict)
except KeyError as e:
    print(f"Missing required field: {e}")
```

---

## Integration Points

### With Data Connectors
```python
from indoxArcg.data_connectors import WikipediaReader

reader = WikipediaReader()
docs = reader.load_data(pages=["Large language models"])
# Returns List[Document] instances
```

### With Vector Stores
```python
from indoxArcg.vector_stores import PineconeVectorStore

vector_store = PineconeVectorStore()
vector_store.add(documents=docs)  # Accepts List[Document]
```

---

## Performance Considerations

1. **Content Size**
   - Optimal: 500-1500 characters per document
   - Maximum: 10,000 characters (split larger content)

2. **Metadata Efficiency**
   ```python
   # Preferred - flat structure
   {"author": "Alice", "page_count": 12}
   
   # Avoid - nested data
   {"details": {"author": {"name": "Alice"}}}
   ```

3. **Bulk Operations**
   ```python
   # Process 1000 documents at a time
   batch_size = 1000
   for i in range(0, len(docs), batch_size):
       process_batch(docs[i:i+batch_size])
   ```

---

## Security Notes

1. **Sensitive Data**
   ```python
   # Never store raw PII in content/metadata
   Document(
       source="customer_service_chat",
       content="[REDACTED]",
       metadata={"user_id": "hash_abc123"}
   )
   ```

2. **Validation Filters**
   ```python
   from html import escape

   sanitized_content = escape(raw_content)
   doc = Document(source="web", content=sanitized_content)
   ```
```
