# inDoxArcg

<div style="text-align: center;">
    <h1>inDoxArcg</h1>
    <a href="https://github.com/osllmai/inDox/libs/IndoxArcg">
        <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=16&duration=3000&pause=500&multiline=true&width=700&height=100&lines=inDoxArcg;Cache-Augmented+and+Retrieval-Augmented+Generative+%7C+Open+Source;Copyright+%C2%A9%EF%B8%8F+OSLLAM.ai" alt="Typing SVG" style="margin-top: 20px;"/>
    </a>
</div>

---

[![License](https://img.shields.io/github/license/osllmai/inDoxArcg)](https://github.com/osllmai/inDox/blob/master/LICENSE)
[![PyPI](https://badge.furyIndoxArcg.svg)](https://pypi.org/IndoxArcg/0.0.3/)
[![Python](https://img.shields.io/pypi/pyveIndoxArcg.svg)](https://pypi.org/pIndoxArcg/0.0.3/)
[![Downloads](https://static.pepy.techIndoxArcg)](https://pepy.tech/pIndoxArcg)

[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/inDoxArcg?style=social)](https://github.com/osllmai/inDoxArcg)

<p align="center">
  <a href="https://osllm.ai">Official Website</a> &bull; <a href="https://docs.osllm.ai/index.html">Documentation</a> &bull; <a href="https://discord.gg/qrCc56ZR">Discord</a>
</p>

<p align="center">
  <b>NEW:</b> <a href="https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill">Subscribe to our mailing list</a> for updates and news!
</p>

## Overview

**inDoxArcg** is a next-generation application designed for advanced document processing and retrieval augmentation. It offers two powerful pipelines:

1. **Cache-Augmented Generation (CAG)**: Enhances LLM responses by leveraging local caching, similarity search, and fallback mechanisms.
2. **Retrieval-Augmented Generation (RAG)**: Provides context-aware answers by retrieving relevant information from vector stores.

Key features include multi-query retrieval, smart validation, web search fallback, and customizable similarity search algorithms.

---

## Features

The **inDoxArcg** application offers two powerful pipelines designed to optimize the use of large language models (LLMs) and enhance document retrieval capabilities. These pipelines provide flexibility and adaptability to meet diverse use cases:

### Cache-Augmented Generation (CAG) Pipeline

- **Multi-query retrieval**: Expands single queries into multiple related queries.
- **Smart retrieval**: Validates context for relevance and hallucination.
- **Web search fallback**: Uses DuckDuckGo when local cache is insufficient.
- **Customizable similarity search**: Supports TF-IDF, BM25, and Jaccard similarity algorithms.

### Retrieval-Augmented Generation (RAG) Pipeline

The **Retrieval-Augmented Generation (RAG) Pipeline** is designed to provide highly accurate and contextually aware answers by retrieving relevant documents from a vector store. For example, if you want to answer the question, "What are the health benefits of green tea?", the pipeline will:

1. Search for relevant articles or documents in the vector store.
2. Validate the retrieved context for relevance and accuracy.
3. Generate a detailed answer using the Large Language Model (LLM) based on the retrieved context.

This makes RAG particularly suitable for scenarios requiring:

- **Research and Academia:** Retrieving specific scientific studies or historical data.
- **Customer Support:** Answering customer queries by extracting relevant data from a knowledge base.
- **Legal and Compliance:** Providing precise answers using legal documents or compliance guidelines.
- **Standard retrieval**: Uses vector similarity search.
- **Context clustering**: Organizes retrieved context for enhanced usability.
- **Advanced querying**: Offers options like multi-query expansion and smart validation.
- **Web fallback**: Ensures high-quality results with external web searches when needed.

---

## Roadmap

| Feature               | Implemented | Description                                           |
| --------------------- | ----------- | ----------------------------------------------------- |
| **Model Support**     |             |                                                       |
| Ollama (e.g., Llama3) | ✅          | Local Embedding and LLM Models powered by Ollama      |
| HuggingFace           | ✅          | Local Embedding and LLM Models powered by HuggingFace |
| Google (e.g., Gemini) | ✅          | Embedding and Generation Models by Google             |
| OpenAI (e.g., GPT4)   | ✅          | Embedding and Generation Models by OpenAI             |
| **API Model Support** |             |                                                       |
| OpenAI                | ✅          | Embedding and LLM Models from Indox API               |
| Mistral               | ✅          | Embedding and LLM Models from Indox API               |
| Anthropic             | ✅          | Embedding and LLM Models from Indox API               |

| Loader and Splitter      | Implemented | Description                                    |
| ------------------------ | ----------- | ---------------------------------------------- |
| Simple PDF               | ✅          | Import PDF files                               |
| UnstructuredIO           | ✅          | Import data through Unstructured               |
| Clustered Load And Split | ✅          | Adds a clustering layer to PDFs and text files |

| RAG Features          | Implemented | Description                                                  |
| --------------------- | ----------- | ------------------------------------------------------------ |
| Hybrid Search         | ✅          | Combines Semantic Search with Keyword Search                 |
| Semantic Caching      | ✅          | Saves and retrieves results based on semantic meaning        |
| Clustered Prompt      | ✅          | Retrieves smaller chunks and clusters for summarization      |
| Agentic RAG           | ✅          | Ranks context and performs web searches for reliable answers |
| Advanced Querying     | ✅          | Delegates tasks based on LLM evaluation                      |
| Reranking             | ✅          | Improves results by ranking based on context                 |
| Customizable Metadata | ❌          | Offers flexible control over metadata                        |

| Bonus Features        | Implemented | Description                 |
| --------------------- | ----------- | --------------------------- |
| Docker Support        | ❌          | Deployable via Docker       |
| Customizable Frontend | ❌          | Fully customizable frontend |

---

## Installation

Install the latest stable version:

```bash
pip install inDoxArcg
```

> **Note:** This package requires Python 3.9 or later. Please ensure you have the appropriate version installed before proceeding. Additionally, verify that you have `pip` updated to the latest version to avoid dependency issues.

### Setting Up Python Environment

1. Create a virtual environment:

   **Windows:**

   ```bash
   python -m venv indoxarcg_env
   ```

   **macOS/Linux:**

   ```bash
   python3 -m venv indoxarcg_env
   ```

2. Activate the virtual environment:

   **Windows:**

   ```bash
   indoxarcg_env\Scripts\activate
   ```

   **macOS/Linux:**

   ```bash
   source indoxarcg_env/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Examples

### Cache-Augmented Generation (CAG)

The `indoxArcg` package emphasizes a modular design to provide flexibility and ease of use. The imports are structured to clearly separate functionalities such as LLMs, vector stores, data loaders, and pipelines. Below is an example of using the Cache-Augmented Generation (CAG) pipeline:

**Initialization:**

```python
from indoxArcg.llms import OpenAi
from indoxArcg.data_loaders import Txt, DoclingReader
from indoxArcg.splitter import RecursiveCharacterTextSplitter, SemanticTextSplitter
from indoxArcg.pipelines.cag import CAG, KVCache

llm = OpenAi(api_key="your_openai_api_key")
embedding_model = DeepSeek()
cache = KVCache()

cag = CAG(llm, embedding_model, cache)
```

**Preload Documents:**

```python
documents = ["Document 1 text...", "Document 2 text..."]
cag.preload_documents(documents, cache_key="my_cache")
```

**Inference:**

```python
query = "What is the capital of France?"
response = cag.infer(
    query=query,
    cache_key="my_cache",
    context_strategy="recent",
    context_turns=5,
    top_k=5,
    similarity_threshold=0.5,
    web_search=True,
    smart_retrieval=True,
)
print(response)
```

**Initialization:**

```python
from indoxarcg import CAG, KVCache

llm = YourLLM()
embedding_model = YourEmbeddingModel()
cache = KVCache()

cag = CAG(llm, embedding_model, cache)
```

**Preload Documents:**

```python
documents = ["Document 1 text...", "Document 2 text..."]
cag.preload_documents(documents, cache_key="my_cache")
```

**Inference:**

```python
query = "What is the capital of France?"
response = cag.infer(
    query=query,
    cache_key="my_cache",
    context_strategy="recent",
    context_turns=5,
    top_k=5,
    similarity_threshold=0.5,
    web_search=True,
    smart_retrieval=True,
)
print(response)
```

### Retrieval-Augmented Generation (RAG)

**Initialization:**

```python
from indoxArcg.pipelines.rag import RAG
from indoxArcg.llms import OpenAi
from indoxArcg.vector_stores import Chroma
from indoxArcg.embeddings import OpenAiEmbedding

llm = OpenAi(api_key="your_openai_api_key",model="gpt-3.5-turbo")
embedding = OpenAiEmbedding(api_key="your_openai_api_key")
db = Chroma(collection_name="your_collection_name",embedding)
rag = RAG(llm, vector_store)
```

**Inference:**

```python
question = "What is the capital of France?"
response = rag.infer(
    question=question,
    top_k=5,
    use_clustering=True,
    use_multi_query=False,
    smart_retrieval=True,
)
print(response)
```

---

## Contribution

We welcome contributions to improve inDoxArcg. Please refer to our [Contribution Guidelines](https://github.com/osllmai/inDox/blob/master/CONTRIBUTING.md) for detailed instructions on how to get started. The guide includes information on setting up your development environment, submitting pull requests, and adhering to our code of conduct. Please refer to our [Contribution Guidelines](https://github.com/osllmai/inDox/blob/master/CONTRIBUTING.md).

---

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](https://github.com/osllmai/inDox/blob/master/LICENSE) file for details.

---

## Support

For questions, support, or feedback, join our [Discord](https://discord.com/invite/ossllmai) or contact us via [our website](https://osllm.ai).
