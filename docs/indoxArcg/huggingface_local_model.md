```markdown
# Using `HuggingFaceLocalModel` for Local Inference

This document explains how to load and run inference on large Hugging Face language models locally, in **4-bit precision**, with the `HuggingFaceLocalModel` class. Unlike the remote approach (which calls Hugging Face Inference API and has model size limits), local inference runs on your machine’s GPU or CPU—avoiding size constraints but requiring **sufficient hardware**.

---

## 1. Prerequisites

1. **Python Environment**  
   Use a Python environment that can install and run PyTorch, `transformers`, and `bitsandbytes`.

2. **Install Required Packages**  
   ```bash
   pip install torch transformers accelerate bitsandbytes
   ```
   - `bitsandbytes` allows 4-bit quantization, drastically reducing VRAM usage.  
   - `accelerate` helps with device mapping in `transformers`.

3. **GPU with Enough VRAM**  
   - Even in 4-bit precision, you need enough GPU memory for your chosen model.  
   - A 7B parameter model typically requires ~8–10GB of VRAM in 4-bit quantization.

---

## 2. Class Overview

`HuggingFaceLocalModel` is an **indoxArcg** LLM class that implements the `BaseLLM` interface. Internally, it:

- Loads a Hugging Face *causal language model* (`AutoModelForCausalLM`) in **4-bit quantization** using a `BitsAndBytesConfig`.
- Performs **local** GPU (or CPU) inference, **not** via HF Inference API.
- Supports these key methods:
  - `answer_question`
  - `get_summary`
  - `grade_docs`
  - `check_hallucination`
  - `chat`

### Fields in `HuggingFaceLocalModel`

- **`hf_model_id`** (`str`): The Hugging Face model repository identifier (default: `"BioMistral/BioMistral-7B"`).  
- **`prompt_template`** (`str`): A prompt format string (default: `"Context: {context}\nQuestion: {question}\nAnswer:"`).  
- **`bnb_4bit_use_double_quant`** (`bool`): Whether to enable double quant in 4-bit.  
- **`bnb_4bit_quant_type`** (`str`): The 4-bit quantization type (`"nf4"`, etc.).  
- **`bnb_4bit_compute_dtype`** (`torch.dtype`): Typically `torch.bfloat16` for 4-bit compute.  
- **`device_map`** (`str` or dict): For device placement (e.g. `"auto"`, `"cuda"`, or a dictionary).

---

## 3. Basic Usage

Here’s a minimal code example showing how to instantiate the model **locally**:

```python
from indoxArcg.llms import HuggingFaceLocalModel

# Initialize the local model
local_model = HuggingFaceLocalModel(
    hf_model_id="BioMistral/BioMistral-7B",  
    prompt_template="Context: {context}\nQuestion: {question}\nAnswer:",
    device_map="auto"
)

# Generate an answer
context = "The p53 protein is often called the guardian of the genome..."
question = "What role does p53 play in the cell cycle?"
answer = local_model.answer_question(context=context, question=question)

print("Answer:", answer)
```

**Notes**:
- Change `hf_model_id` to any other 4-bit-compatible Hugging Face model (e.g., `"meta-llama/Llama-2-7b-chat-hf"`).
- Ensure you have enough VRAM to load your chosen model in 4-bit.

---

## 4. Integration with indoxArcg Pipelines

Below is a short example using `HuggingFaceLocalModel` with the **indoxArcg** pipeline components (like `CAG`, `KVCache`, `RecursiveCharacterTextSplitter`, etc.):

```python
import torch
from indoxArcg.embeddings import HuggingFaceEmbedding
from indoxArcg.data_loaders import Txt
from indoxArcg.splitter import RecursiveCharacterTextSplitter
from indoxArcg.pipelines.cag import CAG, KVCache
from indoxArcg.llms import HuggingFaceLocalModel

# 1. Initialize the local model in 4-bit
local_model = HuggingFaceLocalModel(
    hf_model_id="BioMistral/BioMistral-7B"
)

# 2. Create an embedding model (for retrieval)
embedding_model = HuggingFaceEmbedding(
    api_key="YOUR_HF_API_KEY",   # if needed
    model="multi-qa-mpnet-base-cos-v1"
)

# 3. Load text data from sample.txt
data_loader = Txt(txt_path="sample.txt")
raw_text = data_loader.load()

# 4. Split the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_text(text=raw_text)

# 5. Create a pipeline with caching
pipeline_with_embedding = CAG(
    llm=local_model,
    embedding_model=embedding_model,
    cache=KVCache(),
)

# 6. Preload (embed) documents
cache_key = "embed_cache"
pipeline_with_embedding.preload_documents(docs, cache_key=cache_key)

# 7. Inference: ask a question
query = "What are the functions of the p53 protein in cellular processes?"
response = pipeline_with_embedding.infer(query, cache_key=cache_key)
print("Query:", query)
print("Answer:", response)
```

### Explanation

- **`HuggingFaceLocalModel`** loads the HF model locally.  
- **`HuggingFaceEmbedding`** queries the HF embeddings API for vector embeddings (optional).  
- **`CAG`** (Context-Augmented Generation) retrieves relevant chunks via embedding similarity and generates an answer with the local LLM.  
- **`KVCache`** stores vector embeddings so they don’t need to be recomputed on every run.

---

## 5. Troubleshooting & Tips

1. **Out of Memory Errors (OOM)**  
   - Lower `max_new_tokens` or try `device_map="auto"` to split layers across multiple GPUs if available.
   - Ensure no other large models share the same GPU.

2. **Pydantic Warnings**  
   - This class sets `protected_namespaces=()` to silence warnings about fields like `_model`.
   - If you see additional warnings, update your Pydantic or rename conflicting fields.

3. **Version Conflicts**  
   - Check that `transformers`, `bitsandbytes`, and `accelerate` versions are compatible (usually best to keep them up to date).

4. **Slower Than Expected**  
   - 4-bit inference is typically slower than full-precision GPU usage but still faster than CPU. Ensure `torch.cuda.is_available()` is `True` if you have a GPU.

---

## 6. Conclusion

`HuggingFaceLocalModel` allows you to avoid model-size limits by loading and running models **locally** in 4-bit quantized form. This is ideal for **offline** or **private** inference when you have a suitable GPU. Combine it with **indoxArcg** pipeline components for an end-to-end RAG (Retrieval-Augmented Generation) workflow.