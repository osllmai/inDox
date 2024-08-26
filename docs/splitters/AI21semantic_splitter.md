# AI21SemanticTextSplitter

AI21SemanticTextSplitter is a Python class that utilizes the AI21 API for semantic text segmentation. It splits input text into meaningful segments and optionally merges these segments into larger chunks, maintaining semantic coherence. This tool is particularly useful for processing large texts while preserving context and meaning.

**Note**: To use AI21SemanticTextSplitter, users need to have an AI21 API key and the `requests` library installed. The AI21 API key should be provided either as an environment variable or when initializing the class.

To use AI21SemanticTextSplitter:

```python
from indox.splitter import AI21SemanticTextSplitter

splitter = AI21SemanticTextSplitter(
    chunk_size=4000,
    chunk_overlap=200,
    api_key="your_ai21_api_key"
)
```

## Hyperparameters

- **chunk_size** [int]: Maximum size of each chunk (default: 4000). Set to 0 to disable merging.
- **chunk_overlap** [int]: Number of characters to overlap between chunks (default: 200).
- **api_key** [Optional[str]]: AI21 API key (optional if set as an environment variable).
- **api_host** [str]: Base URL for the AI21 API (default: "https://api.ai21.com/studio/v1").
- **timeout_sec** [Optional[float]]: Timeout for API requests in seconds (optional).
- **num_retries** [int]: Number of times to retry failed API calls (default: 3).

## Usage
### Setting Up the Python Environment
### Windows

1. **Create the virtual environment:**
```bash
python -m venv indox
```
2**Activate the virtual environment:**
```bash
indox\Scripts\activate
```
### macOS/Linux
1. **Create the virtual environment:**
```bash
python -m venv indox
```
2. **Activate the virtual environment:**
```bash
source indox/bin/activate
```
### Get Started
### Set AI21 API Key as Environment Variable
**Import HuggingFace API Key**
```python
import os
from dotenv import load_dotenv

load_dotenv('api.env')
AI21_API_KEY = os.getenv('AI21_API_KEY')
```
### Import Essential Libraries
```python
from indox.splitter import AI21SemanticTextSplitter
```
### Initialize AI21SemanticTextSplitter
```python
splitter = AI21SemanticTextSplitter(
    chunk_size=4000,
    chunk_overlap=200
)
```
### Split And Processing Chunks
```python
TEXT = (
    "We’ve all experienced reading long, tedious, and boring pieces of text - financial reports, "
    "legal documents, or terms and conditions (though, who actually reads those terms and conditions to be honest?).\n"
    "Imagine a company that employs hundreds of thousands of employees. In today's information "
    "overload age, nearly 30% of the workday is spent dealing with documents. There's no surprise "
    "here, given that some of these documents are long and convoluted on purpose (did you know that "
    "reading through all your privacy policies would take almost a quarter of a year?). Aside from "
    "inefficiency, workers may simply refrain from reading some documents (for example, Only 16% of "
    "Employees Read Their Employment Contracts Entirely Before Signing!).\nThis is where AI-driven summarization "
    "tools can be helpful: instead of reading entire documents, which is tedious and time-consuming, "
    "users can (ideally) quickly extract relevant information from a text. With large language models, "
    "the development of those tools is easier than ever, and you can offer your users a summary that is "
    "specifically tailored to their preferences.\nLarge language models naturally follow patterns in input "
    "(prompt), and provide coherent completion that follows the same patterns. For that, we want to feed "
    'them with several examples in the input ("few-shot prompt"), so they can follow through. '
    "The process of creating the correct prompt for your problem is called prompt engineering, "
    "and you can read more about it here."
)

semantic_text_splitter = AI21SemanticTextSplitter()
chunks = semantic_text_splitter.split_text(TEXT)

print(f"The text has been split into {len(chunks)} chunks.")
for chunk in chunks:
    print(chunk)
    print("====")
```