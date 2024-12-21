import cv2
import pandas as pd

class OCRProcessor:
    """
    A class for processing images and extracting text using different OCR models.

    This class supports Tesseract, EasyOCR, and PaddleOCR for Optical Character Recognition (OCR).
    It provides functionality to preprocess images and extract text from them using the selected OCR model.

    Attributes:
        model (str): The OCR model to use (options: 'tesseract', 'paddle', 'easyocr').
        ocr (object): The OCR instance for the selected model (PaddleOCR, EasyOCR, or None for Tesseract).

    Methods:
        __init__(model: str = 'tesseract'):
            Initializes the OCRProcessor with the selected OCR model.
        preprocess_image_for_tesseract(image_path: str):
            Preprocesses the image for Tesseract OCR (grayscale, blur, and thresholding).
        extract_text_with_tesseract(image_path: str) -> str:
            Extracts text from an image using Tesseract OCR.
        preprocess_image_for_easyocr(image_path: str):
            Preprocesses the image for EasyOCR (grayscale, thresholding, and dilation).
        extract_text_with_easyocr(image_path: str) -> str:
            Extracts text from an image using EasyOCR.
        extract_text_with_paddle(image_path: str) -> str:
            Extracts text from an image using PaddleOCR.
        extract_text(image_path: str) -> str:
            Extracts text from an image using the selected OCR model.
    
    Example:
        # Using Tesseract OCR:
        processor = OCRProcessor(model='tesseract')
        text = processor.extract_text('image_path.png')

        # Using EasyOCR:
        processor = OCRProcessor(model='easyocr')
        text = processor.extract_text('image_path.png')

        # Using PaddleOCR:
        processor = OCRProcessor(model='paddle')
        text = processor.extract_text('image_path.png')
    """

    def __init__(self, model: str = 'tesseract'):
        """
        Initializes the OCRProcessor with the selected OCR model.

        Args:
            model (str): The OCR model to use (options: 'tesseract', 'paddle', 'easyocr').
        
        Example:
            processor = OCRProcessor(model='easyocr')
        """

        self.model = model.lower()
        self.ocr = None

        if self.model == 'paddle':
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(lang='en')
            except ImportError:
                raise ImportError("Please install paddleocr package to use PaddleOCR")

        elif self.model == 'easyocr':
            try:
                import easyocr
                self.ocr = easyocr.Reader(['en'])
            except ImportError:
                raise ImportError("Please install easyocr package to use EasyOCR")

    def preprocess_image_for_tesseract(self, image_path: str):
        """
        Preprocesses the image for Tesseract OCR by resizing, converting to grayscale, 
        applying Gaussian blur, and thresholding.

        Args:
            image_path (str): The path to the image file.

        Returns:
            numpy.ndarray: The preprocessed binary image ready for Tesseract OCR.

        Example:
            processed_image = processor.preprocess_image_for_tesseract('image_path.png')
        """

        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        return binary_image

    def extract_text_with_tesseract(self, image_path: str) -> str:
        """
        Extracts text from an image using Tesseract OCR.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text.

        Example:
            text = processor.extract_text_with_tesseract('image_path.png')
        """

        try:
            import pytesseract
            processed_image = self.preprocess_image_for_tesseract(image_path)
            text = pytesseract.image_to_string(processed_image, config="--oem 3 --psm 6")
            return text.strip()
        except ImportError:
            raise ImportError("Please install pytesseract package to use Tesseract OCR")

    def preprocess_image_for_easyocr(self, image_path: str):
        """
        Preprocesses the image for EasyOCR by resizing, converting to grayscale, 
        applying thresholding, and performing dilation.

        Args:
            image_path (str): The path to the image file.

        Returns:
            numpy.ndarray: The preprocessed image ready for EasyOCR.

        Example:
            processed_image = processor.preprocess_image_for_easyocr('image_path.png')
        """

        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        return dilated_image

    def extract_text_with_easyocr(self, image_path: str) -> str:
        """
        Extracts text from an image using EasyOCR.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text.

        Example:
            text = processor.extract_text_with_easyocr('image_path.png')
        """

        processed_image = self.preprocess_image_for_easyocr(image_path)
        results = self.ocr.readtext(processed_image, text_threshold=0.6, low_text=0.3)
        text_data = [text for (_, text, confidence) in results]
        # Join detected text lines into a single string
        return "\n".join(text_data)

    def extract_text_with_paddle(self, image_path: str) -> str:
        """
        Extracts text from an image using PaddleOCR.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text.

        Example:
            text = processor.extract_text_with_paddle('image_path.png')
        """

        result = self.ocr.ocr(image_path, rec=True)
        text_lines = [line[1][0] for res in result for line in res if line[1][0].strip()]
        return "\n".join(text_lines)

    def extract_text(self, image_path: str):
        """
        Extracts text from an image using the selected OCR model.

        This method calls the appropriate OCR extraction method based on the chosen model 
        ('tesseract', 'easyocr', or 'paddle').

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text.

        Example:
            text = processor.extract_text('image_path.png')
        """

        if self.model == 'tesseract':
            return self.extract_text_with_tesseract(image_path)
        elif self.model == 'paddle':
            return self.extract_text_with_paddle(image_path)
        elif self.model == 'easyocr':
            return self.extract_text_with_easyocr(image_path)
        else:
            raise ValueError("Invalid OCR model selected. Choose 'tesseract', 'paddle', or 'easyocr'.")