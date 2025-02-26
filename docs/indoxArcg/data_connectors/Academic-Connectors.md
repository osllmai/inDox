# Academic & Research Connectors in indoxArcg

This guide covers integrations with academic databases and research repositories for scholarly content ingestion.

---

## Supported Connectors

### 1. ArxivReader
**arXiv preprint paper retrieval**

#### Features
- Paper metadata extraction
- Abstract/content fetching
- Citation context analysis

```python
from indoxArcg.data_connectors import ArxivReader

reader = ArxivReader()
docs = reader.load_data(
    paper_ids=["2201.08239", "2203.02155"],
    include_full_text=True
)
```

#### Installation
```bash
pip install arxiv
```

---

### 2. WikipediaReader
**Encyclopedic knowledge base access**

#### Features
- Page content extraction
- Disambiguation handling
- Multilingual support

```python
from indoxArcg.data_connectors import WikipediaReader

reader = WikipediaReader()
docs = reader.load_data(
    pages=["Large language model", "Transformer (machine learning model)"],
    lang="en"
)
```

#### Installation
```bash
pip install wikipedia
```

---

### 3. GutenbergReader
**Public domain literary works**

#### Features
- Full-text book retrieval
- Metadata-aware search
- Classic literature corpus

```python
from indoxArcg.data_connectors import GutenbergReader

reader = GutenbergReader()
# Get book by ID
alice = reader.get_book("11")
# Search by query
results = reader.search_gutenberg("Shakespeare")
```

#### Installation
```bash
pip install requests beautifulsoup4
```

---

## Comparison Table

| Connector     | Content Type       | Access Method     | Format          | Rate Limits       |
|---------------|--------------------|-------------------|-----------------|-------------------|
| ArxivReader   | Research Papers    | API (arXiv IDs)   | PDF/TeX         | 1 req/3 seconds   |
| WikipediaReader| Encyclopedia       | API (Page Titles) | HTML/Markdown   | 500 req/minute    |
| GutenbergReader| Books              | Web Scraping      | Plain Text      | 5 req/second      |

---

## Common Operations

### Bulk Paper Fetching
```python
# arXiv bulk download
paper_ids = [f"2305.{10000+i}" for i in range(50)]
chunk_size = 10
docs = []
for i in range(0, len(paper_ids), chunk_size):
    docs += reader.load_data(paper_ids[i:i+chunk_size])
    time.sleep(30)  # Respect rate limits
```

### Literature Search
```python
# Gutenberg advanced search
results = reader.search_gutenberg(
    query="19th century astronomy",
    language="en",
    file_type=["txt", "html"]
)
```

### Metadata Enrichment
```python
# Extract references from papers
for doc in docs:
    references = re.findall(r'\bdoi:\d+\.\d+/\S+', doc.content)
    doc.metadata['references'] = references
```

---

## Troubleshooting

1. **arXiv Rate Limits**
   ```python
   import time
   try:
       docs = reader.load_data(...)
   except HTTPError as e:
       if e.status == 403:
           time.sleep(300)  # Wait 5 minutes
   ```

2. **Wikipedia Disambiguation**
   ```python
   from wikipedia import DisambiguationError
   try:
       docs = reader.load_data(pages=["AI"])
   except DisambiguationError as e:
       print(f"Possible options: {e.options}")
   ```

3. **Gutenberg Parsing Issues**
   ```python
   # Raw content handling
   book = reader.get_book("11", raw=True)
   clean_text = re.sub(r'\n{3,}', '\n\n', book.content)
   ```

---

## Best Practices

1. **arXiv**
   - Use arXiv IDs instead of titles for precision
   - Cache responses to avoid redundant fetches
   - Combine with PDF parsers for full-text analysis

2. **Wikipedia**
   - Specify language codes for multilingual content
   - Use page sections for chunked processing
   - Handle redirects with `auto_suggest=False`

3. **Gutenberg**
   - Prefer TXT format over HTML for cleaner text
   - Use metadata filters for era/genre selection
   - Combine with NLP models for classical text analysis

---

## Security Considerations
- Avoid storing raw API responses with sensitive metadata
- Use caching mechanisms to reduce API calls
- Validate user inputs to prevent injection attacks
- Respect Project Gutenberg's terms of service
```
