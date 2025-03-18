# SigLIP

SigLIP is a vision-language model that uses a sigmoid-based contrastive learning objective to compute text-image similarity scores. Unlike traditional classifiers, it enables zero-shot image classification by matching images to user-defined text prompts (e.g., "a photo of a &#123;class&#125;"), achieving flexibility and strong performance without task-specific training.

## Initialization

```python
from indoxminer.classification import SigLIP

# Initialize the classifier
classifier = SigLIP()
```

## Parameters

| Parameter    | Type  | Default                            | Description                           |
| ------------ | ----- | ---------------------------------- | ------------------------------------- |
| `model_name` | `str` | `"google/siglip-base-patch16-224"` | Model name to use for classification. |

## Usage

### Single Image Classification

```python
from PIL import Image

# Load an image
image = Image.open("/path/to/image.jpg")

# Classify the image
classifier.classify(image)
```

### Batch Image Classification

```python
images = [Image.open("/path/to/image1.jpg"), Image.open("/path/to/image2.jpg")]

# Classify the images
classifier.classify(images)
```

### Custom Labels

```python
custom_labels = ["a cat", "a dog", "a bird"]
classifier.classify(image, labels=custom_labels)
```

## Visualization

The `classify` method automatically generates a bar plot of predicted probabilities.

---

## Advanced Usage

For advanced users, you can directly use the `preprocess`, `predict`, and `visualize` methods for greater control.
