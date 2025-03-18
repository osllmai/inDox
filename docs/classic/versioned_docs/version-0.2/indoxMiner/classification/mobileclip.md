# MobileCLIP

## Overview

MobileCLIP is a lightweight version of the CLIP model optimized for mobile and edge devices. It enables efficient and fast image classification without compromising accuracy, making it ideal for mobile applications and real-time image recognition tasks.

## Key Features

- **Optimized for Mobile**: Lightweight architecture for efficient execution on mobile and embedded devices.
- **Fast Inference**: Reduced computational requirements for real-time classification.
- **Custom Labels**: Define label sets to suit application-specific needs.
- **Batch Processing**: Supports classification of multiple images in a single call.
- **Visualization**: Generates intuitive bar plots for classification results.

## Quick Start

### Single Image Classification

```python
from indoxminer.classification import MobileCLIP
from PIL import Image

# Initialize classifier
classifier = MobileCLIP()

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
labels = ["a tree", "a car", "a house"]

# Classify the image with custom labels
classifier.classify(image, labels=labels)
```

## Advanced Features

### Customizable Workflows

MobileCLIP allows users to interact directly with its internal components for flexible integration.

```python
# Preprocess the image and labels
inputs = classifier.preprocess(image, labels)

# Predict probabilities
probs = classifier.predict(inputs)

# Visualize results
classifier.visualize(image, labels, probs, top=5)
```

### Model-Specific Capabilities

- Optimized to run efficiently on mobile processors and low-power devices.
- Maintains high classification accuracy despite reduced model size.
- Tailored for real-time applications requiring fast inference.

## Example Use Cases

- **Mobile Apps**: Implement real-time image classification in smartphone applications.
- **Augmented Reality**: Enhance AR experiences by recognizing objects in real time.
- **Edge Computing**: Deploy AI-powered classification on embedded systems and IoT devices.

## Visualization

The `classify` method generates bar plots displaying predicted probabilities, allowing users to interpret results effectively.

## Why Use MobileCLIP?

- Efficient and lightweight for mobile and edge device deployment.
- Real-time classification with fast inference.
- Customizable for application-specific requirements.

For more details, refer to the main [IndoxMiner Classification Module Documentation](./Classification_Module.md).
