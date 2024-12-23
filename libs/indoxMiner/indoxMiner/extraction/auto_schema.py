from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import re
from datetime import datetime


@dataclass
class AutoDetectedField:
    """
    Represents an automatically detected field with inferred type, description, and validation rules.

    Attributes:
        name (str): The name of the detected field.
        field_type (str): The inferred type of the field (e.g., 'string', 'number', 'date').
        description (str): A description of the detected field.
        required (bool): Whether the field is required (default is False).
        rules (Optional[Dict[str, Any]]): Validation rules for the field (e.g., min/max length, pattern).

    Example:
        field = AutoDetectedField(
            name="Age",
            field_type="number",
            description="Age of the person",
            required=True,
            rules={"min_value": 0, "max_value": 120}
        )
    """

    name: str
    field_type: str
    description: str
    required: bool = False
    rules: Optional[Dict[str, Any]] = None


@dataclass
class AutoExtractionRules:
    """
    Rules for auto-extraction and validation of field data.

    Attributes:
        min_length (Optional[int]): Minimum length for string fields.
        max_length (Optional[int]): Maximum length for string fields.
        pattern (Optional[str]): Regular expression pattern for string validation.
        min_value (Optional[float]): Minimum value for numeric fields.
        max_value (Optional[float]): Maximum value for numeric fields.
        allowed_values (Optional[List[Any]]): List of allowed values for the field.

    Example:
        rules = AutoExtractionRules(min_value=18, max_value=100)
    """

    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


@dataclass
class AutoSchema:
    """
    Represents a schema that automatically detects and adapts to the structure of a document.

    Attributes:
        fields (List[AutoDetectedField]): A list of fields detected within the document.
        detected_structure (Dict[str, Any]): A dictionary containing the inferred structure of the document.

    Methods:
        infer_structure(text: str) -> None:
            Analyzes the provided text to infer structure and detect fields.

        to_prompt(text: str) -> str:
            Generates an extraction prompt based on the detected structure of the document.

        validate_extraction(data: Dict[str, Any]) -> List[str]:
            Validates extracted data against the inferred rules.

    Example:
        schema = AutoSchema()
        schema.infer_structure("Name: John\nAge: 30\nDate: 2023-01-01")
        prompt = schema.to_prompt("Name: John\nAge: 30\nDate: 2023-01-01")
        errors = schema.validate_extraction({"Name": "John", "Age": 30})
    """

    fields: List[AutoDetectedField] = field(default_factory=list)
    detected_structure: Dict[str, Any] = field(default_factory=dict)

    def infer_structure(self, text: str) -> None:
        """
        Analyzes the provided text to infer structure and automatically detect fields.

        Args:
            text (str): The document text to analyze.

        Example:
            schema.infer_structure("Name: John\nAge: 30\nDate: 2023-01-01")
        """
        self.fields = []

        # Detect table-like structures
        if self._looks_like_table(text):
            headers = self._detect_headers(text)
            if headers:
                for header in headers:
                    field_type = self._infer_field_type(header, text)
                    self.fields.append(
                        AutoDetectedField(
                            name=header,
                            field_type=field_type,
                            description=f"Automatically detected {field_type} field from table column",
                            required=True,
                            rules=self._generate_rules(field_type),
                        )
                    )

        # Detect form-like structures
        form_fields = self._detect_form_fields(text)
        for label, sample_value in form_fields.items():
            field_type = self._infer_field_type(label, sample_value)
            self.fields.append(
                AutoDetectedField(
                    name=label,
                    field_type=field_type,
                    description=f"Automatically detected {field_type} field from form",
                    required=True,
                    rules=self._generate_rules(field_type),
                )
            )

    def to_prompt(self, text: str) -> str:
        """
        Generates an extraction prompt based on the detected structure of the document.

        Args:
            text (str): The document text to analyze.

        Returns:
            str: A structured prompt for data extraction.

        Example:
            prompt = schema.to_prompt("Name: John\nAge: 30\nDate: 2023-01-01")
        """
        # First analyze the text to detect structure if not already done
        if not self.fields:
            self.infer_structure(text)

        # Build field descriptions
        fields_desc = "\n".join(
            f"- {field.name} ({field.field_type}): {field.description}"
            for field in self.fields
        )

        # Determine if we're dealing with tabular data
        is_tabular = self._looks_like_table(text)
        table_instruction = (
            """
- Extract data in a tabular format
- Preserve column headers and row relationships
- Return as an array of objects"""
            if is_tabular
            else ""
        )

        return f"""Task: Extract structured information from the given text using automatic field detection.

Detected Fields:
{fields_desc}

Extraction Requirements:
1. Extract all detected fields maintaining their original names
2. Use appropriate data types for each field:
   - Dates in ISO format (YYYY-MM-DD)
   - Numbers as numeric values (not strings)
   - Boolean values as true/false
   - Lists as arrays
   - Nested data as objects
3. Preserve any detected relationships between fields
4. Return data in JSON format{table_instruction}

Text to analyze:
{text}"""

    def _looks_like_table(self, text: str) -> bool:
        """
        Detects if the text contains a table-like structure.

        Args:
            text (str): The document text to analyze.

        Returns:
            bool: True if the text resembles a table, otherwise False.

        Example:
            is_table = schema._looks_like_table("Name | Age\nJohn | 30")
        """
        lines = text.split("\n")
        if len(lines) < 2:
            return False

        # Check for common table indicators
        has_delimiter_row = any(
            line.count("|") > 1 or line.count("\t") > 1 for line in lines
        )
        has_consistent_spacing = self._check_consistent_spacing(lines)
        has_header_indicators = any(
            line.count("-") > 3 or line.count("=") > 3 for line in lines
        )

        return has_delimiter_row or has_consistent_spacing or has_header_indicators

    def _check_consistent_spacing(self, lines: List[str]) -> bool:
        """
        Checks if the lines in the document have consistent spacing, typically for table-like structures.

        Args:
            lines (List[str]): List of lines in the document.

        Returns:
            bool: True if spacing is consistent, otherwise False.

        Example:
            consistent = schema._check_consistent_spacing(["Name  Age", "John  30"])
        """
        if len(lines) < 2:
            return False

        # Get positions of whitespace chunks
        space_positions = []
        for line in lines[:3]:  # Check first few lines
            positions = [i for i, char in enumerate(line) if char.isspace()]
            if positions:
                space_positions.append(positions)

        # Check if space positions are consistent
        if len(space_positions) > 1:
            return any(
                abs(len(pos1) - len(pos2)) <= 1
                for pos1, pos2 in zip(space_positions[:-1], space_positions[1:])
            )

        return False

    def _detect_headers(self, text: str) -> List[str]:
        """
        Detects column headers from table-like text.

        Args:
            text (str): The document text to analyze.

        Returns:
            List[str]: A list of detected column headers.

        Example:
            headers = schema._detect_headers("Name | Age\nJohn | 30")
        """
        lines = text.split("\n")
        potential_headers = []

        # List of known non-field phrases to ignore
        ignore_phrases = {"Thanks for your", "First Class"}

        for i, line in enumerate(lines[:3]):  # Check first few lines
            # Split by common delimiters
            cells = re.split(r"\s{2,}|\t|\|", line.strip())
            cells = [cell.strip() for cell in cells if cell.strip() and cell not in ignore_phrases]

            # Header characteristics
            looks_like_header = all(
                word[0].isupper() for word in cells if word
            ) and not any(cell.replace(".", "").isdigit() for cell in cells)

            if looks_like_header:
                potential_headers = cells
                break

        return potential_headers

    def _detect_form_fields(self, text: str) -> Dict[str, str]:
        """
        Detects form-like field labels and sample values from text.

        Args:
            text (str): The document text to analyze.

        Returns:
            Dict[str, str]: A dictionary of field labels and their sample values.

        Example:
            fields = schema._detect_form_fields("Name: John\nAge: 30")
        """
        fields = {}
        lines = text.split("\n")

        # List of known non-field phrases to ignore
        ignore_phrases = {"Thanks for your"}

        for line in lines:
            # Skip lines with non-field phrases
            if any(phrase in line for phrase in ignore_phrases):
                continue

            # Look for label-value patterns
            matches = re.finditer(r"([A-Za-z][A-Za-z\s]+)[\s:]+([^:]+)(?=\s*|$)", line)
            for match in matches:
                label = match.group(1).strip()
                value = match.group(2).strip()
                if label and value:
                    fields[label] = value

        return fields

    def _infer_field_type(self, label: str, sample: str) -> str:
        """
        Infers the field type based on the label and sample value.

        Args:
            label (str): The field label.
            sample (str): A sample value to determine the field's type.

        Returns:
            str: The inferred field type (e.g., "string", "number", "date").

        Example:
            field_type = schema._infer_field_type("Age", "30")
        """
        # Check for date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}/\d{2}/\d{4}",
            r"\d{2}\.\d{2}\.\d{4}",
        ]
        if any(re.search(pattern, str(sample)) for pattern in date_patterns):
            return "date"

        # Check for numeric patterns
        if isinstance(sample, (int, float)) or (
            isinstance(sample, str)
            and re.match(r"^-?\d*\.?\d+$", sample.replace(",", ""))
        ):
            return "number"

        # Check for boolean indicators
        bool_values = {"true", "false", "yes", "no", "y", "n"}
        if str(sample).lower() in bool_values:
            return "boolean"

        # Check for list indicators
        if isinstance(sample, list) or (
            isinstance(sample, str) and ("," in sample or ";" in sample)
        ):
            return "list"

        # Default to string
        return "string"

    def _generate_rules(self, field_type: str) -> AutoExtractionRules:
        """
        Generates validation rules for a field based on its type.

        Args:
            field_type (str): The field type (e.g., "string", "number", "date").

        Returns:
            AutoExtractionRules: The generated validation rules for the field.

        Example:
            rules = schema._generate_rules("number")
        """

        rules = AutoExtractionRules()

        if field_type == "string":
            rules.min_length = 1
            rules.max_length = 1000
        elif field_type == "number":
            rules.min_value = float("-inf")
            rules.max_value = float("inf")
        elif field_type == "date":
            rules.pattern = r"^\d{4}-\d{2}-\d{2}$"
        elif field_type == "boolean":
            rules.allowed_values = [True, False]

        return rules

    def validate_extraction(self, data: Dict[str, Any]) -> List[str]:
        """
        Validates the extracted data against the inferred rules.

        Args:
            data (Dict[str, Any]): The extracted data to validate.

        Returns:
            List[str]: A list of validation error messages.

        Example:
            errors = schema.validate_extraction({"Name": "John", "Age": 30})
        """
        errors = []

        for field in self.fields:
            value = data.get(field.name)
            if field.required and value is None:
                errors.append(f"{field.name} is required but missing")
                continue

            if value is not None and field.rules:
                rules = field.rules
                if rules.min_length and len(str(value)) < rules.min_length:
                    errors.append(
                        f"{field.name} is shorter than minimum length {rules.min_length}"
                    )
                if rules.max_length and len(str(value)) > rules.max_length:
                    errors.append(
                        f"{field.name} exceeds maximum length {rules.max_length}"
                    )
                if rules.pattern and not re.match(rules.pattern, str(value)):
                    errors.append(f"{field.name} does not match expected pattern")
                if rules.min_value and value < rules.min_value:
                    errors.append(
                        f"{field.name} is below minimum value {rules.min_value}"
                    )
                if rules.max_value and value > rules.max_value:
                    errors.append(
                        f"{field.name} exceeds maximum value {rules.max_value}"
                    )
                if rules.allowed_values and value not in rules.allowed_values:
                    errors.append(f"{field.name} contains invalid value")

        return errors