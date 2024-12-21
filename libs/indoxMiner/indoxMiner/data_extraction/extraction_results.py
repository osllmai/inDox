from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExtractionResult:
    """
    Container for a single extraction result with validation.

    Attributes:
        data (Dict[str, Any]): The extracted data.
        raw_response (str): The original response from the LLM (Language Learning Model).
        validation_errors (List[str]): A list of validation errors, if any.

    Methods:
        is_valid (bool): Checks if the extraction passed validation. Returns `True` if no validation errors, `False` otherwise.

    Example:
        result = ExtractionResult(data={"key": "value"}, raw_response="LLM output", validation_errors=["Error 1"])
        print(result.is_valid)  # Output: False
        result.is_valid = True   # Output: True, no validation errors present.
    """


    data: Dict[str, Any]
    raw_response: str
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if extraction passed validation.

        Returns:
            bool: True if no validation errors, False otherwise
        """
        return len(self.validation_errors) == 0


@dataclass
class ExtractionResults:
    """
    Container for multiple extraction results with validation.

    Attributes:
        data (List[Dict[str, Any]]): A list of extracted data.
        raw_responses (List[str]): A list of original LLM responses.
        validation_errors (Dict[int, List[str]]): A dictionary mapping the index of each result to its validation errors.

    Methods:
        is_valid (bool): Checks if all extractions passed validation. Returns `True` if no validation errors across all results.
        get_valid_results (List[Dict[str, Any]]): Returns a list of results that passed validation.

    Example:
        results = ExtractionResults(
            data=[{"key1": "value1"}, {"key2": "value2"}],
            raw_responses=["Response 1", "Response 2"],
            validation_errors={0: [], 1: ["Error 1"]}
        )
        print(results.is_valid)  # Output: False (because result 1 has validation errors)
        valid_results = results.get_valid_results()  # Output: [{"key1": "value1"}]
    """


    data: List[Dict[str, Any]]
    raw_responses: List[str]
    validation_errors: Dict[int, List[str]]

    @property
    def is_valid(self) -> bool:
        """Check if all extractions passed validation.

        Returns:
            bool: True if no validation errors across all results
        """
        return all(not errors for errors in self.validation_errors.values())

    def get_valid_results(self) -> List[Dict[str, Any]]:
        """Get list of results that passed validation.

        Returns:
            List[Dict[str, Any]]: Valid extraction results
        """
        return [
            data
            for i, data in enumerate(self.data)
            if not self.validation_errors.get(i, [])
        ]