import cv2
import pandas as pd

class OCRProcessor:
    """
    A class for processing images and extracting text using different OCR models.

    This class supports Tesseract, EasyOCR, PaddleOCR, and Surya OCR for Optical Character Recognition (OCR).
    It provides functionality to preprocess images and extract text from them using the selected OCR model.

    Attributes:
        model (str): The OCR model to use (options: 'tesseract', 'paddle', 'easyocr', 'surya').
        ocr (object): The OCR instance for the selected model (PaddleOCR, EasyOCR, or None for Tesseract and Surya).

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
        extract_text_with_surya(image_path: str) -> str:
            Extracts text from an image using Surya OCR.
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

        # Using Surya OCR:
        processor = OCRProcessor(model='surya')
        text = processor.extract_text('image_path.png')
    """

    def __init__(self, model: str = 'surya'):
        """
        Initializes the OCRProcessor with the selected OCR model.

        Args:
            model (str): The OCR model to use (options: 'tesseract', 'paddle', 'easyocr', 'surya').
        
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
        
        elif self.model == 'surya':
            try:
                from surya.ocr import run_ocr
                from surya.model.detection.model import load_model as load_det_model
                from surya.model.detection.processor import load_processor as load_det_processor
                from surya.model.recognition.model import load_model as load_rec_model
                from surya.model.recognition.processor import load_processor as load_rec_processor

                self.run_ocr = run_ocr
                self.det_processor, self.det_model = load_det_processor(), load_det_model()
                self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
            except ImportError:
                raise ImportError("Please install surya package to use Surya OCR")
            except Exception as e:
                raise RuntimeError(f"Error initializing Surya OCR: {e}")

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
    
    def extract_text_with_surya(self, image_path: str) -> str:
        """
        Extracts text from an image using Surya OCR.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text.

        Example:
            text = processor.extract_text_with_surya('image_path.png')
        """
        try:
            image = Image.open(image_path)
            langs = ["en"]  # Replace with supported languages as needed
            ocr_results = self.run_ocr(
                [image],
                [langs],
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor,
            )

            # Combine text from all OCRResult objects
            all_text = []
            for ocr_result in ocr_results:
                for line in ocr_result.text_lines:
                    all_text.append(line.text)

            return "\n".join(all_text).strip()
        except Exception as e:
            raise RuntimeError(f"Error during Surya OCR processing: {e}")

    def extract_text(self, image_path: str):
        """
        Extracts text from an image using the selected OCR model.

        This method calls the appropriate OCR extraction method based on the chosen model 
        ('tesseract', 'easyocr', 'paddle', or 'surya').

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
        elif self.model == 'surya':
            return self.extract_text_with_surya(image_path)
        else:
            raise ValueError("Invalid OCR model selected. Choose 'tesseract', 'paddle', 'easyocr', or 'surya'.")