from typing import List, Optional, Union, Dict
from pathlib import Path
import importlib
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
from unstructured.documents.elements import Element
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby

from .ocr_processor import OCRProcessor


def convert_latex_to_md(latex_path):
    """
    Converts a LaTeX file to Markdown using the latex2markdown library.

    Args:
        latex_path (str): The path to the LaTeX file.

    Returns:
        str: The converted Markdown content, or None if there's an error.

    Example:
        markdown_content = convert_latex_to_md('example.tex')
        if markdown_content:
            print(markdown_content)
        else:
            print("Conversion failed.")
    """
    import latex2markdown

    try:
        with open(latex_path, "r") as f:
            latex_content = f.read()
            l2m = latex2markdown.LaTeX2Markdown(latex_content)
            markdown_content = l2m.to_markdown()
        return markdown_content
    except FileNotFoundError:
        print(f"Error: LaTeX file not found at {latex_path}")
        return None
    except Exception as e:
        print(f"Error during conversion: {e}")


def import_unstructured_partition(content_type):
    """
    Dynamically imports the appropriate partition function from the unstructured library.

    Args:
        content_type (str): The type of content to process (e.g., 'pdf', 'docx').

    Returns:
        callable: The partition function for the specified content type.

    Example:
        partition_func = import_unstructured_partition('pdf')
        elements = partition_func('document.pdf')
    """
    module_name = f"unstructured.partition.{content_type}"
    module = importlib.import_module(module_name)
    partition_function_name = f"partition_{content_type}"
    return getattr(module, partition_function_name)


@dataclass
class Document:
    """
    A dataclass representing a document with its content and metadata.

    Attributes:
        page_content (str): The textual content of the document page.
        metadata (dict): Associated metadata like filename, page number, etc.

    Example:
        doc = Document(page_content="This is a document page.", metadata={"filename": "doc1.pdf", "page_number": 1})
        print(doc.page_content)  # Output: This is a document page.
    """

    page_content: str
    metadata: dict


@dataclass
class ProcessingConfig:
    """
    Configuration settings for document processing.

    Attributes:
        chunk_size (int): Maximum size of text chunks (default: 500)
        hi_res_pdf (bool): Whether to use high-resolution PDF processing (default: True)
        infer_tables (bool): Whether to detect and process tables (default: False)
        custom_splitter (callable): Custom function for splitting text (default: None)
        max_workers (int): Maximum number of concurrent processing threads (default: 4)
        remove_headers (bool): Whether to remove header elements (default: False)
        remove_references (bool): Whether to remove reference sections (default: False)
        filter_empty_elements (bool): Whether to remove empty elements (default: True)
        ocr_for_images (bool): Whether to perform OCR on images (default: False)
        ocr_model (str): OCR model to use ('tesseract' or 'paddle') (default: 'tesseract')

    Example:
        config = ProcessingConfig(chunk_size=1000, hi_res_pdf=True)
        print(config.chunk_size)  # Output: 1000
    """

    chunk_size: int = 4048
    hi_res_pdf: bool = True
    infer_tables: bool = False
    custom_splitter: Optional[callable] = None
    max_workers: int = 4
    remove_headers: bool = False
    remove_references: bool = False
    filter_empty_elements: bool = True
    ocr_for_images: bool = False
    ocr_model: str = "tesseract"


class DocumentType(Enum):
    """
    Enumeration of supported document types with their corresponding file extensions.

    Example:
        doc_type = DocumentType.from_file('example.pdf')
        print(doc_type)  # Output: DocumentType.PDF
    """

    BMP = "bmp"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    EML = "eml"
    EPUB = "epub"
    HEIC = "heic"
    HTML = "html"
    JPEG = "jpeg"
    JPG = "jpg"
    MARKDOWN = "md"
    MSG = "msg"
    ODT = "odt"
    ORG = "org"
    P7S = "p7s"
    PDF = "pdf"
    PNG = "png"
    PPT = "ppt"
    PPTX = "pptx"
    RST = "rst"
    RTF = "rtf"
    TIFF = "tiff"
    TEXT = "txt"
    TSV = "tsv"
    XLS = "xls"
    XLSX = "xlsx"
    XML = "xml"

    @classmethod
    def from_file(cls, file_path: str) -> "DocumentType":
        """
        Determines the document type from a file path or URL.

        Args:
            file_path (str): Path or URL to the document.

        Returns:
            DocumentType: The determined document type.

        Raises:
            ValueError: If the file type is not supported.

        Example:
            doc_type = DocumentType.from_file('document.pdf')
            print(doc_type)  # Output: DocumentType.PDF
        """
        if file_path.lower().startswith(("http://", "https://", "www.")):
            return cls.HTML

        extension = Path(file_path).suffix.lower().lstrip(".")
        if extension == "jpg":
            extension = "jpeg"

        try:
            return cls(extension)
        except ValueError:
            raise ValueError(f"Unsupported file type: {extension}")


class UnstructuredProcessor:
    """
    A processor for extracting and structuring content from various document types.

    This class handles the extraction of text and metadata from different document formats,
    including PDFs, Office documents, images, and web content. It supports concurrent
    processing, content chunking, and various filtering options.

    Attributes:
        sources (List[str]): List of file paths or URLs to process.
        doc_types (Dict[str, DocumentType]): Mapping of sources to their document types.
        ocr_processor (Optional[OCRProcessor]): Processor for optical character recognition.

    Example:
        processor = DocumentProcessor(['document1.pdf', 'document2.pdf'])
        elements = processor._get_elements('document1.pdf')
    """

    def __init__(self, sources: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the DocumentProcessor with one or more sources.

        Args:
            sources: Single source or list of sources to process.

        Example:
            processor = DocumentProcessor('document.pdf')
        """
        self.sources = (
            [str(sources)]
            if isinstance(sources, (str, Path))
            else [str(s) for s in sources]
        )
        self.doc_types = {
            source: DocumentType.from_file(source) for source in self.sources
        }
        self.ocr_processor = None

    def _init_ocr_processor(self):
        """
        Initialize OCR processor if OCR processing is enabled.

        Example:
            processor._init_ocr_processor()
        """
        if self.config.ocr_for_images and not self.ocr_processor:
            self.ocr_processor = OCRProcessor(model=self.config.ocr_model)

    def _create_element_from_ocr(self, text: str, file_path: str) -> List[Element]:
        """
        Create Element objects from OCR-extracted text.

        Args:
            text (str): Extracted text from OCR.
            file_path (str): Path to the processed file.

        Returns:
            List[Element]: List containing the created Element object.

        Example:
            elements = processor._create_element_from_ocr("OCR text here", "document.png")
        """
        from unstructured.documents.elements import Text
        import datetime

        metadata = {
            "filename": Path(file_path).name,
            "file_directory": str(Path(file_path).parent),
            "filetype": self._get_filetype(file_path),
            "page_number": 1,
            "text_as_html": text,
            "last_modified": datetime.datetime.now().isoformat(),
        }

        element = Text(text=text)
        element.metadata = metadata
        return [element]

    def _filter_elements(self, elements: List[Element]) -> List[Element]:
        """
        Filter elements based on configuration settings.

        Args:
            elements (List[Element]): List of elements to filter

        Returns:
            List[Element]: Filtered list of elements
        """
        if not elements:
            return elements

        filtered = elements

        if self.config.filter_empty_elements:
            filtered = [
                el
                for el in filtered
                if hasattr(el, "text") and el.text and el.text.strip()
            ]

        if self.config.remove_headers:
            filtered = [
                el for el in filtered if getattr(el, "category", "") != "Header"
            ]

        if self.config.remove_references:
            try:
                reference_titles = [
                    el
                    for el in filtered
                    if el.text
                    and el.text.strip().lower() == "references"
                    and getattr(el, "category", "") == "Title"
                ]
                if reference_titles:
                    reference_id = reference_titles[0].id
                    filtered = [
                        el
                        for el in filtered
                        if getattr(el.metadata, "parent_id", None) != reference_id
                    ]
            except Exception as e:
                print(f"Warning: Could not process references: {e}")

        return filtered

    def _get_elements(self, file_path: str) -> List[Element]:
        """
        Extract elements from a document using appropriate partition function.

        Args:
            file_path (str): Path to the document to process

        Returns:
            List[Element]: Extracted elements from the document
        """
        try:
            if (
                file_path.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")
                )
                and self.config.ocr_for_images
            ):
                text = self.ocr_processor.extract_text(file_path)
                return self._create_element_from_ocr(text, file_path)
            elif file_path.lower().endswith(".pdf"):
                from unstructured.partition.pdf import partition_pdf

                elements = partition_pdf(
                    filename=file_path,
                    strategy="hi_res" if self.config.hi_res_pdf else "fast",
                    infer_table_structure=self.config.infer_tables,
                )

            elif file_path.lower().endswith(".xlsx"):
                from unstructured.partition.xlsx import partition_xlsx

                elements_ = partition_xlsx(filename=file_path)
                elements = [
                    el for el in elements_ if el.metadata.text_as_html is not None
                ]
            elif file_path.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")
            ):
                from unstructured.partition.image import partition_image

                elements = partition_image(filename=file_path, strategy="auto")
            elif file_path.lower().startswith("www") or file_path.lower().startswith(
                "http"
            ):
                from unstructured.partition.html import partition_html

                elements = partition_html(url=file_path)
            else:
                if file_path.lower().endswith(".tex"):
                    file_path = convert_latex_to_md(latex_path=file_path)
                content_type = file_path.lower().split(".")[-1]
                if content_type == "txt":
                    prt = import_unstructured_partition(content_type="text")
                else:
                    prt = import_unstructured_partition(content_type=content_type)
                elements = prt(filename=file_path)
            return elements
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _combine_elements_by_page(self, elements: List[Element]) -> List[Document]:
        """
        Combine elements on the same page into single documents.

        Args:
            elements (List[Element]): Elements to combine

        Returns:
            List[Document]: List of combined page documents
        """
        documents = []

        def get_page_number(element):
            return getattr(element.metadata, "page_number", 1)

        sorted_elements = sorted(elements, key=get_page_number)

        for page_num, page_elements in groupby(sorted_elements, key=get_page_number):
            page_content = " ".join(
                el.text for el in page_elements if hasattr(el, "text") and el.text
            )
            page_content = page_content.replace("\n", " ").strip()

            if page_content:
                documents.append(page_content)

        return documents

    def _process_elements_to_document(
        self, elements: List[Element], source: str
    ) -> List[Document]:
        """
        Convert elements to Document objects with appropriate metadata.

        Args:
            elements (List[Element]): Elements to process
            source (str): Source file path

        Returns:
            List[Document]: Processed document objects
        """
        page_contents = self._combine_elements_by_page(elements)
        documents = []

        for idx, content in enumerate(page_contents, 1):

            metadata = {
                "filename": Path(source).name,
                "filetype": self._get_filetype(source),
                "page_number": idx,
                "source": source,
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def _get_filetype(self, source: str) -> str:
        """Get MIME type for the file."""
        doc_type = self.doc_types[source]
        mime_types = {
            DocumentType.PDF: "application/pdf",
            DocumentType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            DocumentType.XLS: "application/vnd.ms-excel",
            DocumentType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DocumentType.DOC: "application/msword",
            DocumentType.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            DocumentType.PPT: "application/vnd.ms-powerpoint",
            DocumentType.HTML: "text/html",
            DocumentType.TEXT: "text/plain",
            DocumentType.MARKDOWN: "text/markdown",
            DocumentType.XML: "application/xml",
            DocumentType.CSV: "text/csv",
            DocumentType.TSV: "text/tab-separated-values",
            DocumentType.RTF: "application/rtf",
            DocumentType.EPUB: "application/epub+zip",
            DocumentType.MSG: "application/vnd.ms-outlook",
            DocumentType.EML: "message/rfc822",
            DocumentType.PNG: "image/png",
            DocumentType.JPEG: "image/jpeg",
            DocumentType.TIFF: "image/tiff",
            DocumentType.BMP: "image/bmp",
            DocumentType.HEIC: "image/heic",
        }
        return mime_types.get(doc_type, "application/octet-stream")

    def process(
        self, config: Optional[ProcessingConfig] = None
    ) -> Dict[str, List[Document]]:
        """Process all documents with the given configuration."""
        self.config = config or ProcessingConfig()

        self._init_ocr_processor()

        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_source = {
                executor.submit(self._get_elements, source): source
                for source in self.sources
            }

            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    elements = future.result()
                    filtered_elements = self._filter_elements(elements)
                    results[Path(source).name] = self._process_elements_to_document(
                        filtered_elements, source
                    )
                except Exception as e:
                    print(f"Failed to process {source}: {e}")
                    results[Path(source).name] = []

        return results


class DoclingProcessor:
    """
    A processor for extracting and structuring content from various document types using Docling.

    This class handles the extraction of text and metadata from different document formats
    using the Docling library, supporting various export formats and chunking capabilities.

    Attributes:
        sources (List[str]): List of file paths to process.
        config (ProcessingConfig): Configuration for document processing.

    Example:
        processor = DoclingProcessor(['document1.pdf', 'document2.pdf'])
        documents = processor.process()
    """

    def __init__(self, sources: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the DoclingProcessor with one or more sources.

        Args:
            sources: Single source or list of sources to process.
        """
        self.sources = (
            [str(sources)]
            if isinstance(sources, (str, Path))
            else [str(s) for s in sources]
        )
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration for the processor."""
        from loguru import logger
        import sys

        logger.remove()  # Remove default logger
        logger.add(
            sys.stdout,
            format="<green>{level}</green>: <level>{message}</level>",
            level="INFO",
        )
        logger.add(
            sys.stdout,
            format="<red>{level}</red>: <level>{message}</level>",
            level="ERROR",
        )
        self.logger = logger

    def _process_single_document(
        self,
        source: str,
        max_num_pages: Optional[int] = None,
        max_file_size: Optional[int] = None,
    ) -> List[Document]:
        """
        Process a single document using Docling.

        Args:
            source (str): Path to the document
            max_num_pages (Optional[int]): Maximum number of pages to process
            max_file_size (Optional[int]): Maximum file size to process

        Returns:
            List[Document]: List of processed Document objects
        """
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            kwargs = {}

            if max_num_pages is not None:
                kwargs["max_num_pages"] = max_num_pages
            if max_file_size is not None:
                kwargs["max_file_size"] = max_file_size

            result = converter.convert(source=source, **kwargs)

            # Create base metadata
            metadata = {
                "filename": Path(source).name,
                "filetype": self._get_filetype(source),
                "source": source,
            }

            # Create documents based on config
            if getattr(self.config, "use_chunking", False):
                return self._chunk_document(result, metadata)
            else:
                return self._create_full_document(result, metadata)

        except Exception as e:
            self.logger.error(f"Error processing {source}: {e}")
            return []

    def _chunk_document(self, docling_result, base_metadata: dict) -> List[Document]:
        """
        Chunk the document using Docling's HybridChunker.

        Args:
            docling_result: The result from Docling conversion
            base_metadata (dict): Base metadata for the document

        Returns:
            List[Document]: List of chunked Document objects
        """
        from docling.chunking import HybridChunker

        chunker = HybridChunker()
        chunk_iter = chunker.chunk(
            dl_doc=docling_result.document,
            max_tokens=getattr(self.config, "chunk_size", 512),
        )

        documents = []
        for i, chunk in enumerate(chunk_iter):
            enriched_text = chunker.serialize(chunk=chunk)
            metadata = {
                **base_metadata,
                "chunk_index": i,
                "enriched_text": enriched_text,
            }
            doc = Document(page_content=chunk.text, metadata=metadata)
            documents.append(doc)

        return documents

    def _create_full_document(self, docling_result, metadata: dict) -> List[Document]:
        """
        Create a single document without chunking.

        Args:
            docling_result: The result from Docling conversion
            metadata (dict): Metadata for the document

        Returns:
            List[Document]: List containing single Document object
        """
        export_format = getattr(self.config, "export_format", "text")

        if export_format == "markdown":
            content = docling_result.document.export_to_markdown()
        elif export_format == "html":
            content = docling_result.document.export_to_html()
        else:  # default to text
            content = docling_result.document.export_to_text()

        return [Document(page_content=content, metadata=metadata)]

    def _get_filetype(self, source: str) -> str:
        """Get MIME type for the file."""
        extension = Path(source).suffix.lower().lstrip(".")
        mime_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
            "txt": "text/plain",
            "md": "text/markdown",
            "html": "text/html",
        }
        return mime_types.get(extension, "application/octet-stream")

    def process(
        self,
        config: Optional[ProcessingConfig] = None,
        max_num_pages: Optional[int] = None,
        max_file_size: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Process all documents with the given configuration.

        Args:
            config (Optional[ProcessingConfig]): Processing configuration
            max_num_pages (Optional[int]): Maximum number of pages to process
            max_file_size (Optional[int]): Maximum file size to process

        Returns:
            Dict[str, List[Document]]: Dictionary mapping filenames to processed documents
        """
        self.config = config or ProcessingConfig()
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_source = {
                executor.submit(
                    self._process_single_document, source, max_num_pages, max_file_size
                ): source
                for source in self.sources
            }

            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    results[Path(source).name] = future.result()
                except Exception as e:
                    self.logger.error(f"Failed to process {source}: {e}")
                    results[Path(source).name] = []

        return results
