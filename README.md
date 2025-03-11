<div align="center">
  <h1>Indox Ecosystem</h1>
  <a href="https://github.com/osllmai/indoxArcg">
    <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=16&duration=3000&pause=500&multiline=true&width=700&height=100&lines=Indox+Ecosystem;Advanced+Search+%7C+Data+Mining+%7C+LLM+Evaluation+%7C+Synthetic+Data;Copyright+©️+OSLLAM.ai" alt="Typing SVG"/>
  </a>
</div>

<div align="center">

<p align="center">
  <img src="https://github.com/osllmai/inDox/blob/master/docs/indoxArcg/assets/lite-logo%201.png" alt="inDox Lite Logo">
</p>
</br>

[![License](https://img.shields.io/github/license/osllmai/inDox)](https://github.com/osllmai/inDox/blob/master/LICENSE)
[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)

<!-- [![GitHub stars](https://img.shields.io/github/stars/osllmai/indoxArcg?style=social)](https://github.com/osllmai/inDox) -->

[Official Website](https://osllm.ai) • [Documentation](https://docs.osllm.ai/index.html) • [Discord](https://discord.gg/xGz5tQYaeq)

**NEW:** [Subscribe to our mailing list](https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill) for updates and news!

</div>

## 🌟 The Indox Ecosystem

The Indox Ecosystem is a comprehensive suite of tools designed to revolutionize your AI and data workflows. Our ecosystem consists of four powerful components:

### 1. 🔍 [IndoxArcg](https://github.com/osllmai/indoxArcg)

Advanced **Retrieval-Augmented Generation (RAG)** and **Cache-Augmented Generation (CAG)** system for intelligent information extraction and processing.

## Key Features:

- **Multi-format document support**: Handles PDF, HTML, Markdown, LaTeX, and more.
- **Intelligent clustering and chunk processing**: Organizes and processes documents for efficient retrieval.
- **Support for major LLM providers**: Compatible with OpenAI, Google, Mistral, HuggingFace, Ollama, and others.
- **Advanced RAG features**:
  - Semantic caching for faster retrieval.
  - Multi-query retrieval for improved context extraction.
  - Reranking and relevance scoring for high-quality results.
- **Cache-Augmented Generation (CAG)**:
  - Preloading and caching of documents for faster inference.
  - Smart retrieval with validation and hallucination detection.
  - Web search fallback for missing or insufficient context.
- **Customizable similarity search**: Supports TF-IDF, BM25, and Jaccard similarity algorithms.
- **Robust error handling**: Includes fallback mechanisms for retrieval failures and hallucination detection.

### 2. ⛏️ [IndoxMiner](https://github.com/osllmai/indoxMiner)

Powerful data extraction and mining tool leveraging LLMs.

- Schema-based structured data extraction
- Multi-format support with OCR capabilities
- Flexible validation and type safety
- Async processing for scalability
- High-resolution PDF support

### 3. 📊 [IndoxJudge](https://github.com/osllmai/indoxJudge)

Comprehensive LLM and RAG evaluation framework.

- Multiple evaluation metrics (Faithfulness, Toxicity, BertScore, etc.)
- Safety and bias assessment
- Multi-model comparison capabilities
- RAG-specific evaluation metrics
- Extensible framework for custom metrics

### 4. 🔄 [IndoxGen](https://github.com/osllmai/indoxGen)

Advanced synthetic data generation suite with three specialized components:

- **IndoxGen Core**: LLM-powered synthetic data generation
- **IndoxGen-Tensor**: TensorFlow-based GAN data generation
- **IndoxGen-Torch**: PyTorch-based GAN data generation

## 📦 Quick Installation

Install the entire ecosystem:

```bash
pip install indoxArcg indoxminer indoxjudge indoxgen indoxgen-tensor indoxgen-torch
```

Or install components separately:

```bash
pip install indoxArcg       # Core RAG or Cag functionality
pip install indoxminer     # Data extraction
pip install indoxjudge     # LLM evaluation
pip install indoxgen       # Synthetic data generation
```

## 🚀 Model Support

| Model Provider | indoxArcg | IndoxJudge | IndoxGen |
| -------------- | --------- | ---------- | -------- |
| OpenAI         | ✅        | ✅         | ✅       |
| Google         | ✅        | ✅         | ✅       |
| Mistral        | ✅        | ✅         | ✅       |
| HuggingFace    | ✅        | ✅         | ✅       |
| Ollama         | ✅        | ✅         | ❌       |
| Anthropic      | ❌        | ❌         | ❌       |

## 💡 Getting Started

Check out our example notebooks:

- [indoxArcg Pipeline](https://colab.research.google.com/github/osllmai/indoxArcg/blob/master/Demo/indox_api_openai.ipynb)
- [IndoxJudge Evaluation](https://colab.research.google.com/github/osllmai/indoxArcg/blob/master/Demo/indoxJudge_evaluation.ipynb)
- [IndoxMiner Extraction](examples/indoxminer_extraction.ipynb)
- [IndoxGen Data Generation](examples/indoxgen_synthetic.ipynb)

## 🛣️ Roadmap

- [ ] Unified web interface for all components
- [ ] Docker support across the ecosystem
- [ ] Enhanced integration between components
- [ ] Advanced privacy and security features
- [ ] Multi-language support expansion
- [ ] Additional model provider integrations

## 🤝 Contributing

We welcome contributions to any component of the Indox ecosystem! Please check our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the AGPL License - see the [LICENSE](https://github.com/osllmai/inDox/blob/master/LICENSE) file for details.

<!--
## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=osllmai/indoxArcg,osllmai/indoxMiner,osllmai/indoxJudge,osllmai/indoxGen&type=Date)](https://star-history.com/#osllmai/indoxArcg&osllmai/indoxMiner&osllmai/indoxJudge&osllmai/indoxGen) -->

---



```txt
  .----------------.  .-----------------. .----------------.  .----------------.  .----------------.
| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |     _____    | || | ____  _____  | || |  ________    | || |     ____     | || |  ____  ____  | |
| |    |_   _|   | || ||_   \|_   _| | || | |_   ___ `.  | || |   .'    `.   | || | |_  _||_  _| | |
| |      | |     | || |  |   \ | |   | || |   | |   `. \ | || |  /  .--.  \  | || |   \ \  / /   | |
| |      | |     | || |  | |\ \| |   | || |   | |    | | | || |  | |    | |  | || |    > `' <    | |
| |     _| |_    | || | _| |_\   |_  | || |  _| |___.' / | || |  \  `--'  /  | || |  _/ /'`\ \_  | |
| |    |_____|   | || ||_____|\____| | || | |________.'  | || |   `.____.'   | || | |____||____| | |
| |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'
```

<div align="center">
  Made with ❤️ by OSLLM.ai
</div>
