from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any, Union
import re


@dataclass
class ValidationRule:
    """
    Represents a set of validation rules for a data field. It can enforce constraints such as minimum and maximum values,
    regex patterns, allowed values, and length limits for string fields.

    Attributes:
        min_value (float, optional): Minimum allowed numeric value (e.g., 0.0).
        max_value (float, optional): Maximum allowed numeric value (e.g., 100.0).
        pattern (str, optional): Regex pattern to validate string fields (e.g., r'^[A-Za-z]+$').
        allowed_values (List[Any], optional): List of valid values for the field (e.g., ['small', 'medium', 'large']).
        min_length (int, optional): Minimum length for string fields (e.g., 3).
        max_length (int, optional): Maximum length for string fields (e.g., 50).
    
    Example:
        >>> rule = ValidationRule(min_value=10, max_value=100, pattern=r'^[A-Za-z]+$')
        >>> rule.to_prompt_string()
        'minimum value: 10; maximum value: 100; must match pattern: ^[A-Za-z]+$'
    """


    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def to_prompt_string(self) -> str:
        """
        Converts the validation rules to a human-readable string format. This string can be used for documentation or 
        user prompts.

        Returns:
            str: A string representation of the validation rules.

        Example:
            >>> rule = ValidationRule(min_value=0, max_value=10, pattern=r'^[0-9]+$')
            >>> rule.to_prompt_string()
            'minimum value: 0; maximum value: 10; must match pattern: ^[0-9]+$'
        """

        rules = []
        if self.min_value is not None:
            rules.append(f"minimum value: {self.min_value}")
        if self.max_value is not None:
            rules.append(f"maximum value: {self.max_value}")
        if self.pattern is not None:
            rules.append(f"must match pattern: {self.pattern}")
        if self.allowed_values is not None:
            rules.append(f"must be one of: {', '.join(map(str, self.allowed_values))}")
        if self.min_length is not None:
            rules.append(f"minimum length: {self.min_length}")
        if self.max_length is not None:
            rules.append(f"maximum length: {self.max_length}")
        return "; ".join(rules)

    @staticmethod
    def validate_with_pattern(value: str, pattern: str) -> bool:
        """
        Validates a string value against a given regex pattern.

        Args:
            value (str): The string value to be validated.
            pattern (str): The regex pattern to validate against.

        Returns:
            bool: True if the value matches the pattern, False otherwise.

        Example:
            >>> ValidationRule.validate_with_pattern('test@example.com', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            True
            >>> ValidationRule.validate_with_pattern('12345', r'^[A-Za-z]+$')
            False
        """

        return bool(re.match(pattern, value))


# Common validation patterns
class ValidationPatterns:
    """
    A collection of predefined regular expression patterns for common data types like emails, phone numbers, credit cards, 
    and more. These patterns can be used in conjunction with `ValidationRule` to enforce data validation.

    Attributes:
        EMAIL (str): Regex pattern for validating email addresses.
        PHONE_INTERNATIONAL (str): Regex pattern for validating international phone numbers.
        NAME (str): Regex pattern for validating names.
        DATE_ISO (str): Regex pattern for validating dates in ISO format (YYYY-MM-DD).
        PASSPORT_NUMBER (dict): Dictionary containing regex patterns for validating passport numbers by country.
        ...
    
    Example:
        >>> ValidationPatterns.EMAIL
        '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    """

    # Personal Information
    EMAIL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    PHONE_INTERNATIONAL = r"^\+?[\d\s-]{10,20}$"
    NAME = r"^[A-Za-z\s\'-]{2,50}$"
    DATE_ISO = r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$"

    # Document Numbers
    PASSPORT_NUMBER = {
        "generic": r"^[A-Z0-9]{6,9}$",
        "usa": r"^[A-Z]\d{8}$",
        "uk": r"^[0-9]{9}$",
        "eu": r"^[A-Z0-9]{8,12}$",
    }

    ID_NUMBER = {
        "generic": r"^\d{5,12}$",
        "ssn": r"^\d{3}-\d{2}-\d{4}$",
        "ein": r"^\d{2}-\d{7}$",
    }

    # Financial
    CREDIT_CARD = r"^\d{4}(?:[ -]?\d{4}){3}$"
    BANK_ACCOUNT = {
        "generic": r"^\d{8,12}$",
        "iban": r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}$",
        "routing": r"^\d{9}$",
    }
    CURRENCY = r"^\$?\d+(\.\d{2})?$"

    # Location
    ZIP_CODE = {
        "usa": r"^\d{5}(-\d{4})?$",
        "canada": r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$",
        "uk": r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$",
    }

    # Business
    VAT_NUMBER = {"eu": r"^[A-Z]{2}\d{8,12}$", "uk": r"^GB\d{9}$"}
    COMPANY_NUMBER = r"^[A-Z0-9]{6,12}$"

    # Travel
    FLIGHT_NUMBER = r"^[A-Z]{2,3}\d{1,4}[A-Z]?$"
    BOOKING_REFERENCE = r"^[A-Z0-9]{6}$"
    IATA_AIRPORT = r"^[A-Z]{3}$"
    SEAT_NUMBER = r"^\d{1,2}[A-Z]$"

    # Medical
    MEDICAL_LICENSE = r"^[A-Z]\d{7}$"
    ICD_CODE = r"^[A-Z]\d{2}(\.\d{1,2})?$"

    # Vehicle
    VIN = r"^[A-HJ-NPR-Z0-9]{17}$"
    LICENSE_PLATE = {
        "generic": r"^[A-Z0-9]{1,8}$",
        "usa": r"^[A-Z0-9]{5,8}$",
        "eu": r"^[A-Z]{1,3}\s?[A-Z0-9]{1,4}$",
    }


class CommonValidationRules:
    """
    A set of predefined validation rules for commonly used data fields such as names, emails, phone numbers, and dates.
    This class offers convenience methods to create validation rules for specific field types.

    Example:
        >>> CommonValidationRules.NAME_RULE
        ValidationRule(pattern=r'^[A-Za-z\s\'-]{2,50}$', min_length=2, max_length=50)

        >>> CommonValidationRules.create_enum_rule(['small', 'medium', 'large'])
        ValidationRule(allowed_values=['small', 'medium', 'large'])
    """

    NAME_RULE = ValidationRule(
        pattern=ValidationPatterns.NAME, min_length=2, max_length=50
    )

    EMAIL_RULE = ValidationRule(pattern=ValidationPatterns.EMAIL, max_length=100)

    PHONE_RULE = ValidationRule(pattern=ValidationPatterns.PHONE_INTERNATIONAL)

    AMOUNT_RULE = ValidationRule(min_value=0.0, pattern=ValidationPatterns.CURRENCY)

    DATE_RULE = ValidationRule(pattern=ValidationPatterns.DATE_ISO)

    ADDRESS_RULE = ValidationRule(min_length=10, max_length=200)

    PASSPORT_RULE = ValidationRule(
        pattern=ValidationPatterns.PASSPORT_NUMBER["generic"],
        min_length=6,
        max_length=9,
    )

    ID_RULE = ValidationRule(
        pattern=ValidationPatterns.ID_NUMBER["generic"], min_length=5, max_length=12
    )

    FLIGHT_RULE = ValidationRule(pattern=ValidationPatterns.FLIGHT_NUMBER)

    BOOKING_RULE = ValidationRule(
        pattern=ValidationPatterns.BOOKING_REFERENCE, min_length=6, max_length=6
    )

    @staticmethod
    def create_enum_rule(values: List[str]) -> ValidationRule:
        """
        Creates a validation rule for enumerated values (e.g., a list of allowed options).

        Args:
            values (List[str]): List of valid values to be used for the rule.

        Returns:
            ValidationRule: A rule that only allows values from the given list.

        Example:
            >>> CommonValidationRules.create_enum_rule(['small', 'medium', 'large'])
            ValidationRule(allowed_values=['small', 'medium', 'large'])
        """
        return ValidationRule(allowed_values=values)

    @staticmethod
    def create_number_range_rule(min_val: float, max_val: float) -> ValidationRule:
        """
        Creates a validation rule for numeric fields that ensures the value is within a specified range.

        Args:
            min_val (float): The minimum valid value.
            max_val (float): The maximum valid value.

        Returns:
            ValidationRule: A rule that enforces the numeric range.

        Example:
            >>> CommonValidationRules.create_number_range_rule(10, 100)
            ValidationRule(min_value=10, max_value=100)
        """
        return ValidationRule(min_value=min_val, max_value=max_val)

    @staticmethod
    def create_text_length_rule(min_len: int, max_len: int) -> ValidationRule:
        """
        Creates a validation rule for string fields that ensures the text length is within a specified range.

        Args:
            min_len (int): The minimum allowed length of the string.
            max_len (int): The maximum allowed length of the string.

        Returns:
            ValidationRule: A rule that enforces the text length constraints.

        Example:
            >>> CommonValidationRules.create_text_length_rule(5, 20)
            ValidationRule(min_length=5, max_length=20)
        """
        return ValidationRule(min_length=min_len, max_length=max_len)


class FieldType(Enum):
    """
    Enum that defines the data types supported for field extraction and validation.

    Attributes:
        STRING (str): Represents a string field.
        INTEGER (str): Represents an integer field.
        FLOAT (str): Represents a floating-point number field.
        BOOLEAN (str): Represents a boolean field (True/False).
        DATE (str): Represents a date field in ISO format.
        LIST (str): Represents a list field.
        DICT (str): Represents a dictionary field.
        EMAIL (str): Represents an email field.
        PHONE (str): Represents a phone number field.
        URL (str): Represents a URL field.

    Example:
        >>> FieldType.STRING
        'string'

        >>> FieldType.DATE
        'date'
    """

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    LIST = "list"
    DICT = "dict"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"


@dataclass
class Field:
    """
    Represents a data field with associated validation rules and metadata. Fields can be simple types like strings and 
    integers, or more complex types like lists and dictionaries.

    Attributes:
        name (str): The name of the field.
        description (str): A description of the field.
        field_type (FieldType): The type of the field (e.g., STRING, INTEGER).
        required (bool): Whether the field is mandatory (default is True).
        rules (Optional[ValidationRule]): Validation rules associated with the field.
        array_item_type (Optional[FieldType]): The type of items in a list (if applicable).
        dict_fields (Optional[Dict[str, Union[FieldType, 'Field']]]): The structure of a dictionary field (if applicable).

    Example:
        >>> field = Field(name="email", description="User email address", field_type=FieldType.EMAIL, required=True)
        >>> field.to_prompt_string()
        'email (email*): User email address'
    """

    name: str
    description: str
    field_type: FieldType
    required: bool = True
    rules: Optional[ValidationRule] = None
    array_item_type: Optional[FieldType] = None
    dict_fields: Optional[Dict[str, Union[FieldType, "Field"]]] = None

    def __post_init__(self):
        """Validate field configuration after initialization."""
        if self.field_type == FieldType.LIST:
            if not self.array_item_type:
                raise ValueError(
                    "LIST field_type requires array_item_type to be specified"
                )
            if self.dict_fields and self.array_item_type != FieldType.DICT:
                raise ValueError(
                    "dict_fields can only be specified for LIST fields with DICT array_item_type"
                )

        if self.field_type == FieldType.DICT and not self.dict_fields:
            raise ValueError("DICT field_type requires dict_fields to be specified")

        if self.dict_fields:
            for field_name, field_type in self.dict_fields.items():
                if not isinstance(field_type, (FieldType, Field)):
                    raise ValueError(f"Invalid dict_fields type for {field_name}")

    def to_prompt_string(self) -> str:
        """
        Converts the field definition to a human-readable string format, which includes the field name, type, description, 
        and validation rules.

        Returns:
            str: A string representation of the field.

        Example:
            >>> field = Field(name="email", description="User email address", field_type=FieldType.EMAIL, rules=CommonValidationRules.EMAIL_RULE)
            >>> field.to_prompt_string()
            'email (email*): User email address\n    Validation: must match pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        """
        type_desc = self.field_type.value

        if self.field_type == FieldType.LIST:
            if self.array_item_type == FieldType.DICT and self.dict_fields:
                dict_fields_desc = self._format_dict_fields(self.dict_fields)
                type_desc = f"list of dictionaries containing {{{dict_fields_desc}}}"
            else:
                type_desc = f"list of {self.array_item_type.value}s"
        elif self.field_type == FieldType.DICT and self.dict_fields:
            dict_fields_desc = self._format_dict_fields(self.dict_fields)
            type_desc = f"dictionary containing {{{dict_fields_desc}}}"

        desc = f"{self.name} ({type_desc}{'*' if self.required else ''}): {self.description}"

        if self.rules:
            rules_str = self.rules.to_prompt_string()
            if rules_str:
                desc += f"\n    Validation: {rules_str}"

        return desc

    def _format_dict_fields(self, fields: Dict[str, Union[FieldType, "Field"]]) -> str:
        """Format dictionary fields for prompt string."""
        formatted = []
        for k, v in fields.items():
            if isinstance(v, Field):
                formatted.append(f"{k}: {v.to_prompt_string()}")
            else:
                formatted.append(f"{k}: {v.value}")
        return ", ".join(formatted)

    def validate_value(self, value: Any) -> bool:
        """
        Validates a value against the field's type and validation rules.

        Args:
            value (Any): The value to validate.

        Returns:
            bool: True if the value is valid, False otherwise.

        Example:
            >>> field = Field(name="age", description="User age", field_type=FieldType.INTEGER, rules=CommonValidationRules.create_number_range_rule(18, 99))
            >>> field.validate_value(25)
            True
            >>> field.validate_value(15)
            False
        """
        if value is None:
            return not self.required

        if self.field_type == FieldType.LIST:
            if not isinstance(value, (list, tuple)):
                return False
            if self.array_item_type == FieldType.DICT:
                return all(self.validate_dict_structure(item) for item in value)
            return all(
                self._validate_basic_type(item, self.array_item_type) for item in value
            )

        if self.field_type == FieldType.DICT:
            return self.validate_dict_structure(value)

        return self._validate_basic_type(value, self.field_type)

    def validate_dict_structure(self, value: Dict) -> bool:
        """Validate that a dictionary value matches the specified structure."""
        if not isinstance(value, dict) or not self.dict_fields:
            return False

        for field_name, field_type in self.dict_fields.items():
            if field_name not in value:
                return not self.required

            if isinstance(field_type, FieldType):
                if not self._validate_basic_type(value[field_name], field_type):
                    return False
            elif isinstance(field_type, Field):
                if not field_type.validate_value(value[field_name]):
                    return False

        return True

    def _validate_basic_type(self, value: Any, field_type: FieldType) -> bool:
        """Validate a value against a basic field type."""
        type_checks = {
            FieldType.STRING: lambda x: isinstance(x, str),
            FieldType.INTEGER: lambda x: isinstance(x, int),
            FieldType.FLOAT: lambda x: isinstance(x, (int, float)),
            FieldType.BOOLEAN: lambda x: isinstance(x, bool),
            FieldType.DATE: lambda x: isinstance(x, str)
            and bool(re.match(ValidationPatterns.DATE_ISO, x)),
            FieldType.LIST: lambda x: isinstance(x, (list, tuple)),
            FieldType.DICT: lambda x: isinstance(x, dict),
            FieldType.EMAIL: lambda x: isinstance(x, str)
            and bool(re.match(ValidationPatterns.EMAIL, x)),
            FieldType.PHONE: lambda x: isinstance(x, str)
            and bool(re.match(ValidationPatterns.PHONE_INTERNATIONAL, x)),
            FieldType.URL: lambda x: isinstance(x, str),
        }
        return type_checks.get(field_type, lambda x: True)(value)