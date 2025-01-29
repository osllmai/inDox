# Multimodal Models in IndoxMiner

## Overview
The `multimodal/` module in IndoxMiner provides support for **vision-language models** that can process both images and text. These models extend IndoxMiner beyond object detection and classification by enabling **natural language understanding of images**. 

Currently, the following multimodal models are supported:
- **LLaVA-NeXT (LLaVA + LLaMA 3 8B Instruct)**
- **BLIP-2 (Vision-Language Model for Captioning & Question Answering)**

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
Once installed, you can use the LLaVA or BLIP-2 models from IndoxMiner.

#### **Using LLaVA Model**
```python
from indoxminer.multimodal.llava_model import LLaVAModel

# Initialize the model
model = LLaVAModel()

# Provide a local image path and a question
image_path = "path/to/your/image.jpg"
question = "What is shown in this image?"

# Generate response
response = model.generate_response(image_path, question)
print("LLaVA Response:", response)
```

#### **Using BLIP-2 Model**
```python
from indoxminer.multimodal.blip2_model import BLIP2Model

# Initialize the model
model = BLIP2Model()

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

### **Expected Output**
```
Generated Caption: A person sitting on a bench with a dog.
Model Answer: There are two dogs in the image.
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

