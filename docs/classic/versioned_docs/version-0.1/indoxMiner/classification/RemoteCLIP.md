# Remote CLIP Classifier

The `RemoteCLIPClassifier` is a specialized classifier based on the CLIP framework, tailored for remote sensing and general image classification tasks. It supports various model variants, including `RN50`, `ViT-B-32`, and `ViT-L-14`.

---

## 🔧 Installation

Before using the `RemoteCLIPClassifier`, ensure you have the `open-clip-torch` library installed. This library provides the required framework for working with RemoteCLIP models.

### Install OpenCLIP

```bash
pip install open-clip-torch
```

---

## 📥 Downloading Pretrained Checkpoints

The pretrained checkpoints for RemoteCLIP are available on Hugging Face. You can download them programmatically using the `huggingface_hub` library:

### Example: Download Checkpoints

```python
from huggingface_hub import hf_hub_download

# Download checkpoints for all supported models
for model_name in ['RN50', 'ViT-B-32', 'ViT-L-14']:
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
    print(f'{model_name} is downloaded to {checkpoint_path}.')
```

Alternatively, you can clone the repository using Git LFS to manually download the files.

---

## 🧠 Using RemoteCLIPClassifier

The `RemoteCLIPClassifier` in the `IndoxMiner` library simplifies the process of using RemoteCLIP models for image classification.

### Initialization

```python
from indoxminer.classification import RemoteCLIPClassifier

# Initialize the classifier with a specific model and checkpoint
model_name = "ViT-L-14"  # Options: 'RN50', 'ViT-B-32', 'ViT-L-14'
checkpoint_path = "/path/to/RemoteCLIP-ViT-L-14.pt"

classifier = RemoteCLIPClassifier(model_name=model_name, checkpoint_path=checkpoint_path)
```

---

### Parameters

| Parameter         | Type  | Default      | Description                             |
| ----------------- | ----- | ------------ | --------------------------------------- |
| `model_name`      | `str` | `"ViT-L-14"` | Specifies the RemoteCLIP model to use.  |
| `checkpoint_path` | `str` | `None`       | Path to the pretrained checkpoint file. |

---

## 🚀 Usage

### Single Image Classification

```python
from PIL import Image

# Load an image
image = Image.open("/path/to/image.jpg")

# Define labels for classification
labels = ["An airport", "A university", "A stadium"]

# Classify the image
classifier.classify(image, labels, top=3)
```

### Batch Image Classification

```python
# Load multiple images
images = [Image.open("/path/to/image1.jpg"), Image.open("/path/to/image2.jpg")]

# Classify the batch of images
classifier.classify(images, labels, top=3)
```

### Custom Labels

```python
custom_labels = ["A tiger", "A mountain", "A river"]
classifier.classify(image, labels=custom_labels, top=3)
```

---

## 🔍 Visualization

The `classify` method automatically generates visualizations:

1. **Image**: Displays the input image.
2. **Bar Chart**: Shows the top `N` predictions and their probabilities.

Example:

```plaintext
Image 1 predictions:
[{'An airport': 85.3}, {'A stadium': 10.2}, {'A university': 4.5}]
```

---

## 🔬 Advanced Usage

For more control, use the `preprocess`, `predict`, and `visualize` methods directly:

```python
# Preprocess the image and labels
inputs = classifier.preprocess(images, labels)

# Predict probabilities
probs = classifier.predict(inputs)

# Visualize the results
classifier.visualize(images, labels, probs, top=3)
```

---

## 📜 Supported Models

| Model Name | Description                            |
| ---------- | -------------------------------------- |
| `RN50`     | ResNet-50-based RemoteCLIP model.      |
| `ViT-B-32` | Vision Transformer with 32x32 patches. |
| `ViT-L-14` | Larger Vision Transformer model.       |

---

## Example End-to-End Workflow

```python
from PIL import Image
from indoxminer.classification import RemoteCLIPClassifier

# Step 1: Initialize the classifier
model_name = "ViT-L-14"
checkpoint_path = "/path/to/RemoteCLIP-ViT-L-14.pt"
classifier = RemoteCLIPClassifier(model_name=model_name, checkpoint_path=checkpoint_path)

# Step 2: Load the image
image = Image.open("/path/to/airport.jpg")

# Step 3: Define labels
labels = ["An airport", "A university", "A stadium"]

# Step 4: Classify the image
classifier.classify(image, labels, top=3)
```

---

## 🛠️ Common Issues and Solutions

### 1. **Missing Checkpoints**

- Ensure the pretrained checkpoint is downloaded and accessible at the specified `checkpoint_path`.

### 2. **CUDA Errors**

- Check that your system supports CUDA and that `torch.cuda.is_available()` returns `True`.

### 3. **Installation Issues**

- Ensure `open-clip-torch` and all dependencies are installed.

---

## 🌟 Summary

The `RemoteCLIPClassifier` is a powerful tool for image classification, supporting multiple model variants and flexible input handling. It integrates seamlessly with `IndoxMiner`, offering:

- Batch processing.
- Custom label support.
- Visualization of results.

---
