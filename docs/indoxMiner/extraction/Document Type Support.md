# Document Type Support in Indox Miner

Indox Miner is designed to process various document types, providing users with the flexibility to extract structured information from multiple sources, including text files, spreadsheets, and images. This guide details the supported document types and provides instructions for enabling image processing.

## Supported Document Types

Indox Miner can process the following document types:

| Document Type | Extensions                      | Description |
|---------------|---------------------------------|-------------|
| **PDF**       | `.pdf`                          | Portable Document Format, widely used for sharing formatted documents. |
| **Word**      | `.doc`, `.docx`                 | Microsoft Word files, common for reports and text-heavy documents. |
| **Excel**     | `.xls`, `.xlsx`                 | Microsoft Excel spreadsheets, often used for tabular data. |
| **PowerPoint**| `.ppt`, `.pptx`                 | Microsoft PowerPoint files, typically used for presentations. |
| **Image**     | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.heic` | Image files in various formats (supports OCR for text extraction). |
| **Text**      | `.txt`, `.csv`, `.tsv`          | Plain text and comma/tab-separated files, ideal for simple data. |
| **Markdown**  | `.md`                           | Markdown files for lightweight formatting. |
| **RTF**       | `.rtf`                          | Rich Text Format, allows basic text formatting. |
| **Email**     | `.eml`, `.msg`                  | Email files, useful for processing email threads and headers. |
| **Web Page**  | `.html`                         | HTML files, enabling processing of saved web pages. |
| **XML**       | `.xml`                          | XML files, common for structured data exchange. |
| **EPUB**      | `.epub`                         | E-book format for reading and processing digital publications. |

## Enabling Image Processing

Indox Miner includes support for Optical Character Recognition (OCR), allowing it to extract text from image files. To enable image processing:

1. **Set the OCR flag**: The `DocumentProcessor` class includes a configuration flag for OCR. To process images, set `ocr_for_images` to `True` in the configuration.

2. **Choose an OCR Model**: Indox Miner supports various OCR models, such as Tesseract, PaddleOCR, and EasyOCR. Specify the model in the `ocr_model` parameter.

### Example: Enabling Image Processing

```python
from indox_miner.processor import DocumentProcessor, ProcessingConfig

# Configure the document processor with OCR enabled
config = ProcessingConfig(
    ocr_for_images=True,       # Enable OCR for images
    ocr_model="tesseract"      # Choose an OCR model (tesseract, paddle, or easyocr)
)

# Initialize DocumentProcessor with sources and configuration
sources = ["path/to/document.pdf", "path/to/image.jpg"]
processor = DocumentProcessor(sources)

# Process documents, including images
processed_data = processor.process(config=config)
```

### OCR Models and Their Usage

| OCR Model   | Description                                       |
|-------------|---------------------------------------------------|
| **Tesseract** | A highly accurate, open-source OCR engine.     |
| **PaddleOCR** | Suitable for multilingual support, uses PaddlePaddle. |
| **EasyOCR**   | Quick and efficient, good for basic OCR tasks. |

## Handling Different Document Types in Extraction

1. **Initialize `DocumentProcessor`** with one or more document sources.
2. **Set configuration options**: Use `ProcessingConfig` to specify additional settings, such as chunk size, table inference, and OCR options.
3. **Run Processing**: The processor will automatically detect each document type and apply the appropriate extraction method.

### Example: Processing Multiple Document Types

```python
from indox_miner.processor import DocumentProcessor, ProcessingConfig

# Define document sources
sources = [
    "path/to/document.pdf",
    "path/to/spreadsheet.xlsx",
    "path/to/image.jpg"
]

# Configure processor for images and PDFs
config = ProcessingConfig(
    chunk_size=500,
    hi_res_pdf=True,
    ocr_for_images=True,
    ocr_model="tesseract"
)

# Initialize the DocumentProcessor and process
processor = DocumentProcessor(sources=sources)
processed_documents = processor.process(config=config)
```

## Advanced Configuration Options

The `ProcessingConfig` class provides additional settings to customize processing:

- **Chunk Size**: Set `chunk_size` to control the maximum size of text chunks (default is 4048 characters).
- **High-Resolution PDFs**: Set `hi_res_pdf` to `True` for high-quality PDF processing.
- **Infer Tables**: Set `infer_tables` to `True` to detect and process tables within documents.
- **Remove Headers/References**: Configure `remove_headers` and `remove_references` to exclude header and reference sections in structured documents.
- **Filter Empty Elements**: Set `filter_empty_elements` to remove any blank elements from the extracted content.

## Conclusion

Indox Miner’s document type support allows for efficient extraction from a wide range of formats. By enabling image processing with OCR, users can also extract information from scanned documents and images, making Indox Miner a versatile tool for handling complex document types.
