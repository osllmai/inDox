# Large Language Models (LLMs)

indoxArcg provides unified access to state-of-the-art LLMs through a consistent interface. All models implement common methods for question answering, document grading, and hallucination checking.

## Table of Contents

- [Large Language Models (LLMs)](#large-language-models-llms)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [.env file with API keys:](#env-file-with-api-keys)
  - [Supported Models](#supported-models)
  - [Common Interface](#common-interface)
  - [Model Configuration Guides](#model-configuration-guides)
    - [1. OpenAI](#1-openai)
    - [2. Mistral](#2-mistral)
    - [3. Hugging Face](#3-hugging-face)
    - [4. Google AI](#4-google-ai)
    - [5. Ollama](#5-ollama)
    - [6. DeepSeek](#6-deepseek)
    - [7. NerdToken](#7-nerdtoken)
    - [8. Azure OpenAI](#8-azure-openai)
    - [9. Local Inference with HuggingFace](#9-local-inference-with-huggingface)
  - [Troubleshooting](#troubleshooting)
  - [Future Development](#future-development)

## Prerequisites

1. Python 3.8+ environment
2. Install base package:
   ```bash
   pip install python-dotenv
   ```

## .env file with API keys:

```ini
OPENAI_API_KEY=your_key
MISTRAL_API_KEY=your_key
GOOGLE_API_KEY=your_key
HF_API_KEY=your_key
```

## Supported Models

| #   | Provider     | Class Name          | Requirements                    |
| --- | ------------ | ------------------- | ------------------------------- |
| 1   | OpenAI       | OpenAi              | pip install openai              |
| 2   | Mistral      | Mistral             | pip install mistralai           |
| 3   | Hugging Face | HuggingFaceAPIModel | pip install requests            |
| 4   | Google AI    | GoogleAi            | pip install google-generativeai |
| 5   | Ollama       | Ollama              | Local Ollama server             |
| 6   | DeepSeek     | DeepSeek            | pip install openai              |
| 7   | NerdToken    | NerdToken           | API key required                |
| 8   | Azure OpenAI | AzureOpenAi         | Azure deployment                |

## Common Interface

All LLM classes implement these core methods:

```python
class BaseLLM:
    def answer_question(self, context: str, question: str) -> str: ...
    def get_summary(self, documentation: str) -> str: ...
    def grade_docs(self, context: list, question: str) -> list: ...
    def check_hallucination(self, context: str, answer: str) -> str: ...
    def chat(self, prompt: str, system_prompt: str) -> str: ...
```

## Model Configuration Guides

### 1. OpenAI

Recommended Models: gpt-4-turbo, GPT4-o

```python
from indoxArcg.llms import OpenAi
from dotenv import load_dotenv
import os

load_dotenv()

llm = OpenAi(
    api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-4-turbo",
)

response = llm.answer_question(
    context="Climate change refers to...",
    question="What are main causes of global warming?"
    temperature=0.3
)
```

### 2. Mistral

Recommended Models: mistral-large-latest, codestral-latest

```python
from indoxArcg.llms import Mistral

mistral_llm = Mistral(
    api_key=os.getenv('MISTRAL_API_KEY'),
    model="mistral-large-latest",
)
```

### 3. Hugging Face

```python
from indoxArcg.llms import HuggingFaceAPIModel

hf_llm = HuggingFaceAPIModel(
    api_key=os.getenv('HF_API_KEY'),
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    prompt_template="[INST] {context}\nQuestion: {question} [/INST]"
)
```

### 4. Google AI

```python
from indoxArcg.llms import GoogleAi

gemini = GoogleAi(
    api_key=os.getenv('GOOGLE_API_KEY'),
    model="gemini-1.5-flash",
)
```

### 5. Ollama

Setup:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
```

```python
from indoxArcg.llms import Ollama

local_llm = Ollama(
    model="llama2:13b",
)
```

### 6. DeepSeek

```python
from indoxArcg.llms import DeepSeek

ds_llm = DeepSeek(
    api_key="your_deepseek_key",
    model="deepseek-chat",
)
```

### 7. NerdToken

```python
from indoxArcg.llms import NerdToken

nt_llm = NerdToken(
    api_key=os.getenv('NERDTOKEN_API_KEY'),
    model="openai/gpt-4o-turbo",
)
```

### 8. Azure OpenAI

```python
from indoxArcg.llms import AzureOpenAi

azure_llm = AzureOpenAi(
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    endpoint="https://your-resource.openai.azure.com",
    deployment_name="gpt-4-turbo-deployment",
    api_version="2024-02-01"
)
```

### 9. Local Inference with HuggingFace

For private/offline use with 4-bit quantization:

```python
from indoxArcg.llms import HuggingFaceLocalModel

local_llm = HuggingFaceLocalModel(
    hf_model_id="BioMistral/BioMistral-7B",
    device_map="auto",
    bnb_4bit_quant_type="nf4",
    max_new_tokens=512
)

response = local_llm.answer_question(
    context="Mitochondria are...",
    question="What is the function of mitochondria?"
)
```

Requirements:

```bash
pip install torch transformers accelerate bitsandbytes
```

## Troubleshooting

Common Issues:

- APIError: Invalid API Key: Verify .env file loading and key permissions
- Timeout Errors: Increase timeout parameter for local models
- CUDA Out of Memory: Reduce max_new_tokens or use smaller model
- Formatting Issues: Adjust prompt_template for model-specific formats

## Future Development

Planned enhancements:

- Streaming response support for all models
- Automatic model fallback strategies
- Integrated token counting
- Advanced caching mechanisms
- Multi-modal capabilities (image+text)


Reviewed by: Ali Nemati - March, 22, 2025

