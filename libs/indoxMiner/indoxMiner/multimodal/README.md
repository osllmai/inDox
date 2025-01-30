# Multimodal Models in IndoxMiner

## Overview
The `multimodal/` module in IndoxMiner provides support for **vision-language models** that can process both images and text. These models extend IndoxMiner beyond object detection and classification by enabling **natural language understanding of images**. 

Currently, the following multimodal models are supported:
- **LLaVA-NeXT (LLaVA + LLaMA 3 8B Instruct)**
- **BLIP-2 (Vision-Language Model for Captioning & Question Answering)**
- **GPT-4o Turbo (Multimodal Large Language Model)**

These models allow users to ask questions about images and receive **detailed natural language responses** or generate captions automatically.

---

## Installation (for LLaVA Model)
Before using the `multimodal` models, you need to install LLaVA-NeXT and its dependencies.

### **1️⃣ Clone and Install LLaVA-NeXT**
```bash
!git clone https://github.com/LLaVA-VL/LLaVA-NeXT
%cd LLaVA-NeXT
!pip install -e .
```

### **2️⃣ Modify LLaVA to Use LLaMA 3 (NousResearch/Meta-Llama-3-8B-Instruct)**
LLaVA uses a default model, but for better performance, update **`conversation.py`** to use **NousResearch/Meta-Llama-3-8B-Instruct** instead.

Modify this line in `LLaVA-NeXT/llava/conversation.py`:
```python
pretrained = "NousResearch/Meta-Llama-3-8B-Instruct"
```

After making the change, save the file and proceed.

---

## Usage
### **Loading Multimodal Models in IndoxMiner**
Once installed, you can use the LLaVA, BLIP-2, or GPT-4o Turbo models from IndoxMiner.

#### **Using LLaVA**
```python
from indoxminer.multimodal.llava import LLaVA

# Initialize the model
model = LLaVA()

# Provide a local image path and a question
image_path = "path/to/your/image.jpg"
question = "What is shown in this image?"

# Generate response
response = model.generate_response(image_path, question)
print("LLaVA Response:", response)
```

#### **Using BLIP-2**
```python
from indoxminer.multimodal.blip2 import BLIP2

# Initialize the model
model = BLIP2()

# Provide a local image path and a question
image_path = "path/to/your/image.jpg"

# Generate a caption
caption = model.generate_response(image_path)
print("Generated Caption:", caption)

# Ask a question about the image
question = "How many objects are there?"
answer = model.generate_response(image_path, question)
print("Model Answer:", answer)
```

#### **Using GPT-4o Turbo**
```python
from indoxminer.multimodal.gpt4o import GPT4o

# Initialize the model
api_key = "your-api-key-here"
model = GPT4o(api_key)

# Provide a local image path and a question
image_path = "path/to/your/image.jpg"
question = "What is happening in this image?"

# Generate response
response = model.generate_response(image_path, question)
print("GPT-4o Response:", response)
```

### **Expected Output**
```
Generated Caption: A person sitting on a bench with a dog.
Model Answer: There are two dogs in the image.
GPT-4o Response: The person appears to be sitting on a bench, observing something nearby.
```

---

## Future Work
The `multimodal/` category will expand to support:
- **Kosmos-2**
- **GPT-4V**
- **Other vision-language models**

This makes IndoxMiner a powerful **multi-task AI framework** combining **text, images, and structured data extraction**.

---

## Contact
For issues or contributions, please submit a pull request or open an issue in IndoxMiner’s GitHub repository.

🚀 **Happy Coding with IndoxMiner!** 🚀