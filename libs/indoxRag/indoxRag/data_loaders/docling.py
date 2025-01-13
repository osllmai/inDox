from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class DoclingReader:
    def __init__(self, file_path: str):
        from docling.document_converter import DocumentConverter

        try:
            self.converter = DocumentConverter()
            self.file_path = file_path
        except Exception as e:
            logger.error(f"Error initializing DocumentConverter: {e}")
            raise

    def load(self, max_num_pages=None, max_file_size=None):
        kwargs = {}

        if max_num_pages is not None:
            kwargs["max_num_pages"] = max_num_pages

        if max_file_size is not None:
            kwargs["max_file_size"] = max_file_size

        self.result = self.converter.convert(source=self.file_path, **kwargs)

        # if output_format == "txt":
        #     return self.result.document.export_to_text
        # elif output_format == "markdown":
        #     return self.result.document.export_to_markdown
        # elif output_format == "html":
        #     return self.result.document.export_to_html
        return self.result

    # def load_and_split(self, max_tokens=512, tokenizer="BAAI/bge-small-en-v1.5"):
    #     from docling.chunking import HybridChunker

    #     self.chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens)
    #     chunk_iter = self.chunker.chunk(self.result.document)
    #     return chunk_iter
    def load_and_split(self, max_tokens=512, tokenizer="BAAI/bge-small-en-v1.5"):
        from docling.chunking import HybridChunker

        # Initialize the chunker with the tokenizer and max tokens
        self.chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens)

        # Chunk the document
        chunk_iter = self.chunker.chunk(self.result.document)

        return chunk_iter
