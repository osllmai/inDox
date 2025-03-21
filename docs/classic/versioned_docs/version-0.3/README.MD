# Welcome to Indox Documentation

Welcome to the official documentation for the Indox ecosystem. This comprehensive guide will help you understand and use all the components of the Indox ecosystem.

## 🌟 What is Indox?

Indox is an advanced search and retrieval technique that efficiently extracts data from diverse document types, including PDFs and HTML, using online or offline large language models such as OpenAI, Hugging Face, and more.

The Indox Ecosystem is a comprehensive suite of tools designed to revolutionize your AI and data workflows. Our ecosystem consists of four powerful components:

### 1. 🔍 IndoxArcg

Advanced **Retrieval-Augmented Generation (RAG)** and **Cache-Augmented Generation (CAG)** system for intelligent information extraction and processing.

**Key Features:**

- **Multi-format document support**: Handles PDF, HTML, Markdown, LaTeX, and more.
- **Intelligent clustering and chunk processing**: Organizes and processes documents for efficient retrieval.
- **Support for major LLM providers**: Compatible with OpenAI, Google, Mistral, HuggingFace, Ollama, and others.
- **Advanced RAG features**: Semantic caching, multi-query retrieval, reranking, and relevance scoring.
- **Cache-Augmented Generation (CAG)**: Preloading and caching of documents for faster inference.

### 2. ⛏️ IndoxMiner

Powerful data extraction and mining tool leveraging LLMs.

- Schema-based structured data extraction
- Multi-format support with OCR capabilities
- Flexible validation and type safety
- Async processing for scalability
- High-resolution PDF support

### 3. 📊 IndoxJudge

Comprehensive LLM and RAG evaluation framework.

- Multiple evaluation metrics (Faithfulness, Toxicity, BertScore, etc.)
- Safety and bias assessment
- Multi-model comparison capabilities
- RAG-specific evaluation metrics
- Extensible framework for custom metrics

### 4. 🔄 IndoxGen

Advanced synthetic data generation suite with three specialized components:

- **IndoxGen Core**: LLM-powered synthetic data generation
- **IndoxGen-Tensor**: TensorFlow-based GAN data generation
- **IndoxGen-Torch**: PyTorch-based GAN data generation

## 📦 Installation

You can install the entire Indox ecosystem:

```bash
pip install indoxArcg indoxminer indoxjudge indoxgen indoxgen-tensor indoxgen-torch
```

Or install components separately:

```bash
pip install indoxArcg    # Core RAG or CAG functionality
pip install indoxminer   # Data extraction
pip install indoxjudge   # LLM evaluation
pip install indoxgen     # Synthetic data generation
```

## 💡 Getting Started

Explore our detailed guides for each component:

- [IndoxArcg Documentation](/docs/category/indoxarcg)
- [IndoxMiner Documentation](/docs/category/indoxminer)
- [IndoxJudge Documentation](/docs/category/indoxjudge)
- [IndoxGen Documentation](/docs/category/indoxgen)
