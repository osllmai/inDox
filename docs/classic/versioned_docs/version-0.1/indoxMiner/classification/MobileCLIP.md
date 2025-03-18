# Mobile CLIP Classifier

The `MobileCLIPClassifier` provides a lightweight and efficient solution for image classification, optimized for mobile environments. Built on MobileCLIP, it supports flexible classification tasks with batch processing and custom labels.

---

## 🔧 Installation

Before using the `MobileCLIPClassifier`, ensure you have the required dependencies installed:

### Install MobileCLIP

```bash
pip install git+https://github.com/apple/ml-mobileclip.git --no-deps
```

### Install PyTorch and OpenCLIP

```bash
pip install torch==2.1.0 torchvision==0.16.0
pip install open-clip-torch
```

---

## 📥 Downloading Pretrained Checkpoints

You can download the pretrained MobileCLIP checkpoint from its official repository or another hosting location. Ensure the checkpoint is accessible at a known path.

For example:

- Download and save the `mobileclip_s0.pt` checkpoint to `/content/mobileclip_s0.pt`.

---

## 🧠 Using MobileCLIPClassifier

The `MobileCLIPClassifier` in the `IndoxMiner` library streamlines classification tasks by providing a simple API for single and batch image processing.

### Initialization

```python
from indoxminer.classification import MobileCLIPClassifier

# Initialize the classifier with the pretrained checkpoint
classifier = MobileCLIPClassifier(pretrained_path="/content/mobileclip_s0.pt")
```

---

### Parameters

| Parameter         | Type  | Default           | Description                           |
| ----------------- | ----- | ----------------- | ------------------------------------- |
| `model_name`      | `str` | `"mobileclip_s0"` | Name of the MobileCLIP model variant. |
| `pretrained_path` | `str` | `None`            | Path to the pretrained weights file.  |

---

## 🚀 Usage

### Single Image Classification

```python
from PIL import Image

# Load an image
image = Image.open("/path/to/image.jpg")

# Classify the image
classifier.classify(image, top=5)
```

### Batch Image Classification

```python
from pathlib import Path

# Load multiple images
image_paths = [Path("/path/to/image1.jpg"), Path("/path/to/image2.jpg")]
images = [Image.open(path) for path in image_paths]

# Classify the batch of images
classifier.classify(images, top=5)
```

### Custom Labels

```python
custom_labels = ["a tiger", "a mountain", "a river", "a boat", "a forest"]
classifier.classify(images, labels=custom_labels, top=5)
```

---

## 🔍 Visualization

The `classify` method generates visualizations for each image, including:

1. **Input Image**: Displays the input image.
2. **Bar Chart**: Shows the top predictions and their probabilities.

---

## 🔬 Advanced Usage

For more control, you can directly use the classifier's methods:

### Preprocessing

```python
inputs = classifier.preprocess(images, labels=["a cat", "a dog"])
```

### Prediction

```python
probs = classifier.predict(inputs)
```

### Visualization

```python
classifier.visualize(images, labels, probs, top=5)
```

---

## 🌐 Example Workflow

```python
from pathlib import Path
from PIL import Image
from indoxminer.classification import MobileCLIPClassifier

# Step 1: Load images
image_paths = [Path("/path/to/image1.jpg"), Path("/path/to/image2.jpg")]
images = [Image.open(path) for path in image_paths]

# Step 2: Initialize the classifier
classifier = MobileCLIPClassifier(pretrained_path="/path/to/mobileclip_s0.pt")

# Step 3: Classify the images with default labels
classifier.classify(images, top=5)

# Step 4: Classify the images with custom labels
custom_labels = ["a forest", "a river", "a mountain"]
classifier.classify(images, labels=custom_labels, top=5)
```

---

## 🔧 Supported Features

- **Custom Labels**: Define specific categories for image classification.
- **Batch Processing**: Efficiently classify multiple images in a single call.
- **Visualization**: Automatically generates bar plots for results.

---

## 🛠️ Troubleshooting

### 1. **Missing Checkpoint**

- Ensure the `pretrained_path` is correct and accessible.

### 2. **CUDA Not Available**

- Verify that your system supports CUDA and PyTorch is installed with CUDA capabilities.

### 3. **Dependency Errors**

- Ensure all required dependencies (`torch`, `open-clip-torch`, and `mobileclip`) are installed.

---

## 🌟 Summary

The `MobileCLIPClassifier` provides a powerful, lightweight solution for mobile-friendly image classification. Its features include:

- Support for custom labels and batch processing.
- Clear and intuitive visualizations.
- Seamless integration with `IndoxMiner` for broader use cases.

