Below is a **unified** overview of all the text splitters in **indoxArcg**. These splitters help you break down large documents into smaller, more manageable segments—often crucial for LLM processing, chunked embeddings, and context windows in retrieval-augmented generation (RAG) pipelines.

---

# Text Splitters in indoxArcg

When dealing with large documents, splitting them into coherent chunks improves:

1. **Embedding quality**: Smaller text chunks typically produce more focused embeddings.
2. **LLM performance**: Language models often have token limits; chunking ensures each piece stays within the model’s bounds.
3. **Search granularity**: In RAG pipelines, chunk-level search can yield more relevant context for queries.

indoxArcg provides various splitter classes, each tailored to different formats or splitting strategies. You’ll find their implementations in the `docs\indoxArcg\splitters` directory, each with its own `.md` file.

Below is a summary:

---

## 1. [AI21semantic_splitter.md](./AI21semantic_splitter.md)
- **Description**: Leverages AI21’s semantic understanding to split text into logical or semantic units.  
- **Use Case**: When you want to rely on AI21’s advanced NLP to identify sentence or paragraph boundaries more intelligently than simple token limits or punctuation-based splitting.  
- **Key Feature**: Can produce chunks that preserve context while adhering to a specified token or word threshold.

---

## 2. [Charachter_splitter.md](./Charachter_splitter.md)
- **Description**: Splits text purely based on character count.  
- **Use Case**: A simple, language-agnostic approach—quick to implement if you just need uniform chunk sizes.  
- **Key Parameter**: `chunk_size` in characters. Splitting can optionally also consider overlap to maintain context continuity between chunks.

---

## 3. [Markdown_text_splitter.md](./Markdown_text_splitter.md)
- **Description**: Splits documents that are in Markdown format, respecting Markdown-specific elements (e.g., headings, code blocks).  
- **Use Case**: Ideal for knowledge bases or technical docs stored in Markdown. Ensures that headings and code blocks remain intact, improving embedding coherence.  
- **Advanced**: May have toggles to handle code fences or skip certain front-matter sections.

---

## 4. [Recursively_splitter.md](./Recursively_splitter.md)
- **Description**: A hierarchical splitter that first attempts to split at the highest-level boundaries (e.g., paragraphs or sections), then recursively splits smaller sections if they exceed a certain token or character limit.  
- **Use Case**: Best when you want to preserve logical boundaries (e.g., paragraphs, then sentences) while ensuring each chunk meets token size constraints.  
- **Key Feature**: Reduces the risk of cutting sentences mid-way, often leading to more semantically meaningful chunks.

---

## 5. [Semantic_splitter.md](./Semantic_splitter.md)
- **Description**: Similar concept to the AI21-based splitter but typically uses a different semantic parsing or internal logic (e.g., a local embedding-based approach or simpler heuristics).  
- **Use Case**: For text where you want more nuanced splits than raw character or sentence boundaries, but you may not rely on an external API like AI21.  
- **Key Parameter**: Often has a `chunk_size` (in tokens or sentences) and a method for detecting semantic boundaries.

---

## Common Usage Pattern

Most splitters in **indoxArcg** share a similar interface. For instance:

```python
from indoxArcg.splitter import SemanticTextSplitter, CharacterSplitter

# Example: Using the SemanticTextSplitter
splitter = SemanticTextSplitter(chunk_size=200, overlap=20)
chunks = splitter.split_text("Your long text document here...")

for chunk in chunks:
    print(chunk)
```

Typical parameters:

- **`chunk_size`**: The desired length of each chunk (in tokens, words, or characters).  
- **`overlap`**: A small amount of text (e.g., 20 tokens or 50 characters) to repeat from the end of one chunk to the start of the next, to preserve context.  
- **`split_text`**: The main method that takes a string (or list of strings) and returns a list of chunked segments.

---

## Best Practices

1. **Choose a splitter that matches your data**  
   - **Markdown** for `.md` docs  
   - **AI21Semantic** or **Semantic** for well-structured semantic breaks  
   - **Character** or **Recursive** for general text with no strict semantic cues

2. **Be mindful of chunk length**  
   - Larger chunks can preserve more context but risk exceeding model token limits.  
   - Smaller chunks are easier for LLMs to handle but may reduce context continuity.

3. **Use overlap**  
   Overlapping text ensures a question referencing the boundary of two chunks won’t lose context.

4. **Experiment**  
   The ideal splitting strategy depends on your domain, data format, and LLM constraints. It’s often useful to try multiple approaches and see which yields the best retrieval or summarization results.

---

## Getting Started

1. **Install** `indoxArcg` if you haven’t already:
   ```bash
   pip install indoxArcg
   ```

2. **Import** the desired splitter:
   ```python
   from indoxArcg.splitter import CharacterSplitter
   ```

3. **Configure & Split**:
   ```python
   splitter = CharacterSplitter(chunk_size=500, overlap=50)
   document_chunks = splitter.split_text(long_document_text)
   ```

4. **Use in RAG**  
   These chunks can then be fed into embedding models or graph transformers for indexing or knowledge graph construction.

---

## Summary

Text splitters in **indoxArcg** cater to a variety of formats (Markdown, raw text, recursively structured docs) and approaches (simple character-based vs. semantic-based). Correctly splitting your documents is a critical step in any RAG or LLM-based pipeline, ensuring that context is both manageable and semantically coherent.

Use the links above for detailed usage instructions on each splitter. By choosing the right splitter and parameters, you’ll improve the quality of your embeddings, queries, and final outputs. Happy splitting!