### Classification Module

The **classification module** in `IndoxMiner` provides a robust framework for image classification, leveraging advanced deep learning models to deliver high accuracy and flexibility. Designed for scalability and ease of use, it supports both single-image and batch processing, custom label definitions, and intuitive result visualizations.

---

## 🌟 Key Features

- **Multi-Model Support**: Includes a variety of cutting-edge models, such as:
  - **SigCLIP**: A semantic image classification model.
  - **ViT**: Vision Transformer for general image classification tasks.
  - **MetaCLIP**: Meta AI’s enhanced CLIP for diverse applications.
  - **MobileCLIP**: Lightweight CLIP optimized for mobile devices.
  - **BioCLIP**: Designed for biological image analysis.
  - **AltCLIP**: A powerful CLIP model alternative from BAAI.
  - **RemoteCLIP**: Specializes in remote sensing and satellite imagery classification.
  
- **Custom Labels**: Allows users to define custom label sets for tailored classification tasks.

- **Batch Processing**: Efficiently handles multiple images in a single call, enabling scalability for large datasets.

- **Visualization**: Automatically generates bar plots of predicted probabilities, providing an intuitive understanding of classification results.

- **Flexible Integration**: Seamlessly integrates into workflows requiring classification alongside data extraction or object detection.

---

## 📖 Supported Models

| Model          | Description                                      |
|-----------------|--------------------------------------------------|
| **SigCLIP**     | Semantic image classification model.             |
| **ViT**         | Vision Transformer for image classification.     |
| **MetaCLIP**    | Meta AI’s advanced CLIP model.                   |
| **MobileCLIP**  | Mobile-optimized CLIP.                           |
| **BioCLIP**     | Specialized for biological images.               |
| **AltCLIP**     | Alternative CLIP from BAAI.                      |
| **RemoteCLIP**  | Remote sensing-specific CLIP model.              |

Each model has unique strengths, enabling the classification module to cater to diverse use cases, from general-purpose image recognition to domain-specific tasks like biological image analysis or remote sensing.

---

## 🚀 Quick Start

Here’s how you can get started with the **SigCLIP** model for image classification:

### Single Image Classification

```python
from indoxminer.classification import SigCLIPClassifier
from PIL import Image

# Initialize SigCLIP classifier
classifier = SigCLIPClassifier()

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

---

## 🔍 Advanced Features

### Customizable Workflows
The module allows users to directly interact with its components (`preprocess`, `predict`, `visualize`) for complete control over the classification pipeline.

```python
# Preprocess the image and labels
inputs = classifier.preprocess(image, labels)

# Predict probabilities
probs = classifier.predict(inputs)

# Visualize the results
classifier.visualize(image, labels, probs, top=5)
```

### Model-Specific Capabilities
Each classifier is optimized for its respective model, ensuring that unique features like **specialized tokenization** or **domain-specific fine-tuning** are fully utilized.

---

## 🌐 Example Use Cases

1. **E-commerce**: Automatically classify product images into predefined categories.
2. **Healthcare**: Analyze biological images with BioCLIP for medical research.
3. **Remote Sensing**: Identify features in satellite images using RemoteCLIP.
4. **Content Moderation**: Classify images for compliance and moderation workflows.
5. **Mobile Applications**: Leverage MobileCLIP for lightweight classification on mobile devices.

---

## 📊 Visualization

The `classify` method generates bar plots that display the top predictions along with their probabilities. This feature aids in understanding the model’s confidence and results.

---

## 🛠️ Model-Specific Documentation

Refer to the individual documentation pages for detailed information about each classifier:

- [SigCLIPClassifier Documentation](./sigclip.md)
- [ViTClassifier Documentation](./vit.md)
- [MetaCLIPClassifier Documentation](./metaclip.md)
- [MobileCLIPClassifier Documentation](./mobileclip.md)
- [BioCLIPClassifier Documentation](./bioclip.md)
- [AltCLIPClassifier Documentation](./altclip.md)
- [RemoteCLIPClassifier Documentation](./remoteclip.md)

---

## 🌟 Why Use the Classification Module?

The classification module in `IndoxMiner` is more than just a collection of models. It’s a comprehensive solution for integrating image classification into your projects with features like:
- Seamless multi-model support.
- Intuitive APIs for rapid development.
- Customizability for domain-specific tasks.
- Scalable batch processing for large datasets.

---

Feel free to explore the [full documentation](https://indoxminer.readthedocs.io/) for additional details and advanced use cases.