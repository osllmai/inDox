# IndoxMiner: Extracting Structured Data from Images

IndoxMiner provides a powerful and flexible way to extract structured data from unstructured text within images. Using OCR (Optical Character Recognition) to convert image content to text and LLMs (Large Language Models) to interpret and extract specific fields, IndoxMiner simplifies data extraction from images like invoices, receipts, ID cards, and more.

## Key Features

- 📸 **Image to Structured Data**: Extract information from images and convert it into structured formats.
- 🔠 **OCR Integration**: Supports multiple OCR models, including PaddleOCR, EasyOCR, Tesseract, and Mistral OCR, for text extraction from images.
- 🔍 **Custom Extraction Schemas**: Define and validate the data fields you want to extract, tailored to document types like passports and invoices.
- ✅ **Built-in Validation Rules**: Ensures data accuracy with customizable validation options.
- 📊 **Easy Conversion to DataFrames**: Seamlessly convert results to pandas DataFrames for analysis and manipulation.
- 🤖 **LLM Integration**: Use OpenAI, IndoxApi, and other LLMs for advanced text interpretation and extraction quality enhancement.
- 🔄 **Async Support**: Process multiple images concurrently for optimized performance.

## Installation

To set up IndoxMiner, clone the repository and install dependencies:

```bash
pip install indoxminer
pip install paddlepaddle paddleocr  # or easyocr, tesseract depending on your choice
```

You will also need an OCR library to handle the image-to-text conversion.

## Quick Start

### Step 1: Set up the OCR Processor and LLM

```python
from indoxminer import OpenAi, DocumentProcessor, ProcessingConfig, Schema, Extractor

# Initialize OpenAi
openai_api = OpenAi(api_key="YOUR_API_KEY")  # Replace with your actual API key

# Initialize OCR processor with configuration
config = ProcessingConfig(ocr_for_images=True, ocr_model='paddle')  # Change to 'easyocr', 'tesseract', or 'mistral' as needed
```

### Step 2: Define Image Paths and Schema

Define the images to process and select a predefined schema or create a custom one.

```python
# Define the directory containing passport images
image_directory = 'data/passport_dataset_jpg/'
passport_images = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if the paths exist
for image_path in passport_images:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image file {image_path} does not exist.")

# Initialize the extractor with the OpenAi LLM and the passport schema
extractor = Extractor(llm=openai_api, schema=Schema.Passport)
```

### Step 3: Process Images and Extract Data

Process each image with OCR to extract the text, then use IndoxMiner to extract structured data according to the schema.

```python
processor = DocumentProcessor(passport_images)

# Process document to extract text using OCR
results = processor.process(config)

# Extract data from the documents
extracted_data = extractor.extract(results)
```

### Step 4: Handle Results and Convert to DataFrame

Process the extraction results and convert them into a DataFrame.

```python
# Convert extraction results to DataFrame
df = extractor.to_dataframe(extracted_data)
print(df)

# Display valid results or handle errors
for result in extraction_results:
    if result.is_valid:
        print("Extracted Data:", result)
    else:
        print("Extraction errors occurred:", result.validation_errors)
```

## Detailed Workflow

1. **OCR Processing**: Extracts raw text from images using the `DocumentProcessor`. This component utilizes OCR to convert image-based content into text.
2. **Schema Selection**: Choose from predefined schemas (like Passport, Invoice, Receipt) or create custom schemas to define the structure of data to be extracted.
3. **Data Extraction**: Using the selected schema, the LLM processes the extracted text and returns it as structured data according to the specified fields.
4. **Validation**: Each field undergoes validation based on rules defined in the schema, ensuring accuracy by checking constraints like minimum length, numeric range, and specific patterns.
5. **Output Formats**: Extracted results can be converted into multiple formats, including JSON, pandas DataFrames, or raw dictionaries, making further data processing seamless.

## Core Components for Image Extraction

### `OpenAi`

The `OpenAi` class serves as the primary interface for interacting with the OpenAI. This component handles authentication, manages API requests, and retrieves responses.

### `DocumentProcessor`

The `DocumentProcessor` class is responsible for managing the workflow of document processing, including reading the documents and applying OCR.

### `Schema`

Schemas define the structure of data to be extracted from the text, including fields and validation rules. 

### `Extractor`

The `Extractor` is the main class responsible for interacting with the LLM, validating extracted data, and formatting output.

### Validation Rules

Validation rules ensure data quality by setting constraints on each field within a schema.

## Supported OCR Models

- **PaddleOCR**: A deep learning-based OCR engine for recognizing text from images.
- **EasyOCR**: An OCR tool that supports more than 80 languages.
- **Tesseract**: An open-source OCR engine.
- **Mistral OCR**: A powerful OCR model for text extraction from images, integrated with API key input.

## Supported Output Formats

- **JSON**: Returns structured data in JSON format, suitable for further processing or storage.
- **DataFrame**: Converts the results to a pandas DataFrame for analysis and manipulation.
- **Dictionary**: Access the raw extraction results as dictionaries for flexible handling.
- **Markdown**: Display OCR results in Markdown format for easy reading.

## Conclusion

In this document, we outlined how to extract structured data from passport images using IndoxMiner. This process automates the retrieval of critical information typically found in passports, facilitating efficient data processing for various applications.

### Future Work

Consider enhancing this demo by adding error handling for OCR failures, integrating logging for better traceability, or extending the extraction schema for additional fields.

### Additional Features of IndoxMiner

- **Dynamic Schema Adaptation**: Users can define and adapt schemas dynamically, allowing for easy adjustments to the data extraction process as document formats change.
- **Comprehensive Documentation**: IndoxMiner comes with thorough documentation to help users implement features, troubleshoot issues, and optimize their extraction processes.
- **Community Support**: As an open-source project, IndoxMiner benefits from community contributions and support, enabling continuous improvement and feature enhancement.
