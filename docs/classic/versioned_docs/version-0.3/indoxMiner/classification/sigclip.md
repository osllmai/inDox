# SigCLIP

## Overview

SigCLIP is a semantic image classification model integrated into IndoxMiner. It is designed to leverage CLIP's capabilities for classifying images based on textual labels, allowing users to define their own categories and achieve high adaptability across diverse classification tasks.

## Key Features

- **Semantic Classification**: Uses natural language labels to classify images, making it highly flexible.
- **Pretrained CLIP Model**: Built upon CLIP’s vision-language understanding for robust performance.
- **Custom Labels**: Define your own labels for domain-specific classifications.
- **Batch Processing**: Efficiently classifies multiple images in a single function call.
- **Visualization**: Generates bar plots for intuitive interpretation of classification results.

## Quick Start

### Single Image Classification

```python
from indoxminer.classification import SigCLIP
from PIL import Image

# Initialize classifier
classifier = SigCLIP()

# Load an image
image = Image.open("/path/to/image.jpg")

# Classify the image with default labels
classifier.classify(image)
```

### Batch Image Classification

```python
# Load multiple images
images = [Image.open("/path/to/image1.jpg"), Image.open("/path/to/image2.jpg")]

# Classify the batch of images
classifier.classify(images)
```

### Custom Labels

```python
# Define custom labels
labels = ["a cat", "a dog", "a bird"]

# Classify the image with custom labels
classifier.classify(image, labels=labels)
```

## Advanced Features

### Customizable Workflows

SigCLIP allows direct interaction with its processing pipeline for greater control.

```python
# Preprocess the image and labels
inputs = classifier.preprocess(image, labels)

# Predict probabilities
probs = classifier.predict(inputs)

# Visualize results
classifier.visualize(image, labels, probs, top=5)
```

### Model-Specific Capabilities

- Optimized tokenization for text-image understanding.
- Supports natural language-based categories for classification.
- Fine-tuned for high semantic accuracy.

## Example Use Cases

- **Content Moderation**: Automatically classify images for compliance checks.
- **E-commerce**: Categorize product images based on descriptions.
- **Medical Research**: Analyze biological images using semantic labels.

## Visualization

The `classify` method generates bar plots displaying predicted probabilities, helping users interpret model confidence.

## Why Use SigCLIP?

- Leverages CLIP’s semantic capabilities for flexible classification.
- Supports customizable workflows and labels.
- Provides intuitive visualizations for classification results.

For more details, refer to the main [IndoxMiner Classification Module Documentation](./Classification_Module.md).
