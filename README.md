<div align="center">
  <h1>Indox Ecosystem</h1>
  <a href="https://github.com/osllmai/indoxRag">
    <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=16&duration=3000&pause=500&multiline=true&width=700&height=100&lines=Indox+Ecosystem;Advanced+Search+%7C+Data+Mining+%7C+LLM+Evaluation+%7C+Synthetic+Data;Copyright+©️+OSLLAM.ai" alt="Typing SVG"/>
  </a>
</div>

<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/osllmai/inDox/blob/master/docs/indoxRag/assets/lite-logo%201.png" alt="inDox Lite Logo">
</p>
</br>

[![License](https://img.shields.io/github/license/osllmai/indoxRag)](https://github.com/osllmai/indoxRag/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/indoxRag?style=social)](https://github.com/osllmai/indoxRag)

[Official Website](https://osllm.ai) • [Documentation](https://docs.osllm.ai/index.html) • [Discord](https://discord.gg/qrCc56ZR)

**NEW:** [Subscribe to our mailing list](https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill) for updates and news!

</div>

## 🌟 The Indox Ecosystem

The Indox Ecosystem is a comprehensive suite of tools designed to revolutionize your AI and data workflows. Our ecosystem consists of four powerful components:

### 1. 🔍 [IndoxRag](https://github.com/osllmai/indoxRag)

Advanced Retrieval Augmentation Generation (RAG) system for intelligent information extraction and processing.

- Multi-format document support (PDF, HTML, Markdown, LaTeX)
- Intelligent clustering and chunk processing
- Support for major LLM providers (OpenAI, Google, Mistral, HuggingFace, Ollama)
- Advanced RAG features including semantic caching and reranking

### 2. ⛏️ [IndoxMiner](https://github.com/osllmai/indoxMiner)

Powerful data extraction and mining tool leveraging LLMs.

- Schema-based structured data extraction
- Multi-format support with OCR capabilities
- Flexible validation and type safety
- Async processing for scalability
- High-resolution PDF support

### 3. 📊 [IndoxJudge](https://github.com/osllmai/indoxJudge)

Comprehensive LLM and RAG evaluation framework.\
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
pip install indoxrag indoxminer indoxjudge indoxgen indoxgen-tensor indoxgen-torch
```

Or install components separately:

```bash
pip install indoxrag       # Core RAG functionality
pip install indoxminer     # Data extraction
pip install indoxjudge     # LLM evaluation
pip install indoxgen       # Synthetic data generation
```

## 🚀 Model Support

| Model Provider | IndoxRag | IndoxJudge | IndoxGen |
| -------------- | -------- | ---------- | -------- |
| OpenAI         | ✅       | ✅         | ✅       |
| Google         | ✅       | ✅         | ✅       |
| Mistral        | ✅       | ✅         | ✅       |
| HuggingFace    | ✅       | ✅         | ✅       |
| Ollama         | ✅       | ✅         | ❌       |
| Anthropic      | ❌       | ❌         | ❌       |

## 💡 Getting Started

Check out our example notebooks:

- [IndoxRag Pipeline](https://colab.research.google.com/github/osllmai/indoxRag/blob/master/Demo/indox_api_openai.ipynb)
- [IndoxJudge Evaluation](https://colab.research.google.com/github/osllmai/indoxRag/blob/master/Demo/indoxJudge_evaluation.ipynb)
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

This project is licensed under the AGPL License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=osllmai/indoxRag,osllmai/indoxMiner,osllmai/indoxJudge,osllmai/indoxGen&type=Date)](https://star-history.com/#osllmai/indoxRag&osllmai/indoxMiner&osllmai/indoxJudge&osllmai/indoxGen)

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
