import asyncio
from typing import List, Dict, Any, Union, Tuple, Optional
from loguru import logger
import re
import pandas as pd
import json
from tabulate import tabulate
from decimal import Decimal
from .extraction_results import ExtractionResult, ExtractionResults
from .fields import Field, FieldType
from .utils import Document


class Extractor:
    """
    Data extractor using LLM with validation and concurrent processing.

    This class provides methods for extracting structured data from various input types using a language model (LLM). It supports both synchronous and asynchronous extraction, handles validation of extracted fields, and offers various formats for presenting the extracted data, including DataFrame, JSON, Markdown, and table.

    Attributes:
        llm (Any): The language model used for extraction.
        schema (Any): The schema that defines the expected data structure and validation rules.
        max_concurrent (int): Maximum number of concurrent extraction tasks for async processing.
        is_async (bool): Indicates whether the LLM supports asynchronous operations.
    """

    def __init__(
        self,
        llm: Any,
        schema: Any,
        max_concurrent: int = 3,
    ):
        self.llm = llm
        self.schema = schema
        self.max_concurrent = max_concurrent
        self.is_async = asyncio.iscoroutinefunction(self.llm.generate)

    def _sync_extract_chunk(
        self, text: str, chunk_index: int
    ) -> Tuple[int, ExtractionResult]:
        """
        Synchronous version of extract chunk.

        This method processes a single chunk of text and returns the extracted data along with validation errors, if any.

        Args:
            text (str): The text to be processed and extracted.
            chunk_index (int): The index of the chunk for reference in results.

        Returns:
            Tuple[int, ExtractionResult]: The chunk index and the corresponding ExtractionResult containing extracted data and any validation errors.
        """

        try:
            prompt = self.schema.to_prompt(text)
            response = self.llm.generate(prompt)
            return self._process_response(response, chunk_index)
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_index}: {e}")
            return chunk_index, ExtractionResult(
                data={},
                raw_response=str(e),
                validation_errors=[f"Extraction error: {str(e)}"],
            )

    async def _async_extract_chunk(
        self, text: str, chunk_index: int
    ) -> Tuple[int, ExtractionResult]:
        """
        Asynchronous version of extract chunk.

        This method processes a single chunk of text asynchronously and returns the extracted data along with validation errors, if any.

        Args:
            text (str): The text to be processed and extracted.
            chunk_index (int): The index of the chunk for reference in results.

        Returns:
            Tuple[int, ExtractionResult]: The chunk index and the corresponding ExtractionResult containing extracted data and any validation errors.
        """

        try:
            prompt = self.schema.to_prompt(text)
            response = await self.llm.generate(prompt)
            return self._process_response(response, chunk_index)
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_index}: {e}")
            return chunk_index, ExtractionResult(
                data={},
                raw_response=str(e),
                validation_errors=[f"Extraction error: {str(e)}"],
            )

    def _validate_field(self, field: Field, value: Any) -> List[str]:
        """
        Validate a single field value against its rules.

        This method validates a field value based on various rules such as minimum/maximum values, patterns, and required values defined in the schema.

        Args:
            field (Field): The field to be validated.
            value (Any): The value of the field to be validated.

        Returns:
            List[str]: A list of validation error messages, if any.
        """

        errors = []
        if value is None and field.required:
            errors.append(f"{field.name} is required but missing")
            return errors

        if field.rules:
            rules = field.rules
            if rules.min_value is not None and value < rules.min_value:
                errors.append(f"{field.name} is below minimum value {rules.min_value}")
            if rules.max_value is not None and value > rules.max_value:
                errors.append(f"{field.name} exceeds maximum value {rules.max_value}")
            if (
                rules.pattern is not None
                and isinstance(value, str)
                and not re.match(rules.pattern, value)
            ):
                errors.append(f"{field.name} does not match pattern {rules.pattern}")
            if rules.allowed_values is not None and value not in rules.allowed_values:
                errors.append(f"{field.name} contains invalid value")
            if rules.min_length is not None and len(str(value)) < rules.min_length:
                errors.append(
                    f"{field.name} is shorter than minimum length {rules.min_length}"
                )
            if rules.max_length is not None and len(str(value)) > rules.max_length:
                errors.append(f"{field.name} exceeds maximum length {rules.max_length}")
        return errors

    def _process_response(
        self, response: str, chunk_index: int
    ) -> Tuple[int, ExtractionResult]:
        """
        Process and validate the LLM response, assuming JSON.

        This method cleans, parses, and normalizes the LLM's response, followed by validation of the extracted data.

        Args:
            response (str): The raw response from the LLM.
            chunk_index (int): The index of the chunk for reference in results.

        Returns:
            Tuple[int, ExtractionResult]: The chunk index and the corresponding ExtractionResult containing extracted data and validation errors.
        """

        try:
            cleaned_response = self._clean_json_response(response)
            logger.debug(f"Cleaned JSON response: {cleaned_response}")

            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                fixed_json = self._fix_json(cleaned_response)
                data = json.loads(fixed_json)

            data = self._normalize_json_structure(data)
            validation_errors = self._validate_data(data)

            return chunk_index, ExtractionResult(
                data=data, raw_response=response, validation_errors=validation_errors
            )
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing error: {e}\nCleaned Response: {cleaned_response}"
            )
            return chunk_index, ExtractionResult(
                data={},
                raw_response=response,
                validation_errors=[f"JSON parsing error: {str(e)}"],
            )

    def extract(
        self,
        input_data: Union[str, Document, List[Document], Dict[str, List[Document]]],
    ) -> Union[ExtractionResult, ExtractionResults]:
        """
        Unified extraction method that handles both sync and async LLMs appropriately.

        This method handles different input data types (string, Document, list of Documents, or a dictionary of Document lists) and performs data extraction synchronously or asynchronously based on the LLM's capabilities.

        Args:
            input_data (Union[str, Document, List[Document], Dict[str, List[Document]]]): The input data to be extracted. Can be a string, a Document, a list of Documents, or a dictionary of Document lists.

        Returns:
            Union[ExtractionResult, ExtractionResults]: The extraction result(s), either a single ExtractionResult or a collection of ExtractionResults.

        Example:
            input_str = "Extract the names and ages of individuals."
            extractor = Extractor(llm_model, schema)
            result = extractor.extract(input_str)

        Raises:
            ValueError: If the input data type is unsupported.
        """

        if not self.is_async:
            if isinstance(input_data, str):
                return self._sync_extract_chunk(input_data, 0)[1]
            elif isinstance(input_data, Document):
                return self._sync_extract_chunk(input_data.page_content, 0)[1]
            elif isinstance(input_data, list):
                results = [
                    self._sync_extract_chunk(doc.page_content, i)
                    for i, doc in enumerate(input_data)
                ]
                results.sort(key=lambda x: x[0])
                return ExtractionResults(
                    data=[result.data for _, result in results],
                    raw_responses=[result.raw_response for _, result in results],
                    validation_errors={
                        i: result.validation_errors
                        for i, (_, result) in enumerate(results)
                        if result.validation_errors
                    },
                )
            elif isinstance(input_data, dict):
                all_documents = [doc for docs in input_data.values() for doc in docs]
                return self.extract(all_documents)
            else:
                raise ValueError("Unsupported input type")
        else:
            try:
                return asyncio.run(self._async_extract(input_data))
            except Exception as e:
                logger.error(f"Async extraction failed: {str(e)}")
                raise

    async def _async_extract(
        self,
        input_data: Union[str, Document, List[Document], Dict[str, List[Document]]],
    ) -> Union[ExtractionResult, ExtractionResults]:
        if isinstance(input_data, str):
            _, result = await self._async_extract_chunk(input_data, 0)
            return result
        elif isinstance(input_data, Document):
            _, result = await self._async_extract_chunk(input_data.page_content, 0)
            return result
        elif isinstance(input_data, list):
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def extract_with_semaphore(text: str, index: int):
                async with semaphore:
                    return await self._async_extract_chunk(text, index)

            tasks = [
                extract_with_semaphore(doc.page_content, i)
                for i, doc in enumerate(input_data)
            ]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])
            return ExtractionResults(
                data=[result.data for _, result in results],
                raw_responses=[result.raw_response for _, result in results],
                validation_errors={
                    i: result.validation_errors
                    for i, (_, result) in enumerate(results)
                    if result.validation_errors
                },
            )
        elif isinstance(input_data, dict):
            all_documents = [doc for docs in input_data.values() for doc in docs]
            return await self._async_extract(all_documents)
        else:
            raise ValueError("Unsupported input type")

    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean the raw JSON response by removing unnecessary characters.

        This method strips out code block delimiters and comments to clean up the raw JSON response from the LLM.

        Args:
            response_text (str): The raw response text to be cleaned.

        Returns:
            str: The cleaned response text.
        """

        response_text = re.sub(r"```json\s*|\s*```", "", response_text.strip())
        return "\n".join(
            re.sub(r"\s*//.*$", "", line.rstrip())
            for line in response_text.split("\n")
            if line
        )

    def _fix_json(self, json_str: str) -> str:
        """
        Fix common issues in malformed JSON strings.

        This method fixes common formatting issues in the JSON response, such as missing commas or improperly merged JSON objects.

        Args:
            json_str (str): The malformed JSON string to be fixed.

        Returns:
            str: The corrected JSON string.
        """

        fixed_json = re.sub(r",(\s*[}\]])", r"\1", json_str)
        return re.sub(r"}\s*{", "},{", fixed_json)

    def _normalize_json_structure(self, data):
        if isinstance(data, list):
            data = {"items": data}
        elif isinstance(data, dict) and not any(
            isinstance(v, list) for v in data.values()
        ):
            data = {"items": [data]}
        else:
            items_field_name = None
            items = []
            common_fields = {}

            for key, value in data.items():
                if isinstance(value, list) and all(
                    isinstance(item, dict) for item in value
                ):
                    items_field_name = key
                    items = value
                elif not isinstance(value, (list, dict)):
                    common_fields[key] = value

            if items_field_name and items:
                data = {"items": [{**common_fields, **{items_field_name: items}}]}
            else:
                data = {"items": [data]}

        return data

    def _validate_data(self, data: Dict) -> List[str]:
        """
        Validate the entire extracted data against the schema's rules.

        This method validates each field in the extracted data based on the rules defined in the schema.

        Args:
            data (Dict): The extracted data to be validated.

        Returns:
            List[str]: A list of validation error messages, if any.
        """

        validation_errors = []
        for i, item in enumerate(data["items"]):
            for field in self.schema.fields:
                value = item.get(field.name)
                errors = self._validate_field(field, value)
                if errors:
                    validation_errors.extend(
                        [f"Item {i + 1}: {error}" for error in errors]
                    )
        return validation_errors

    def to_dataframe(
        self,
        results: Union[
            ExtractionResult,
            ExtractionResults,
            List[Union[ExtractionResult, ExtractionResults]],
        ],
    ) -> Optional[pd.DataFrame]:
        """
        Convert one or multiple extraction results to a single pandas DataFrame.

        This method converts the extracted data into a pandas DataFrame, ensuring that the fields match the schema and numeric columns are properly handled.

        Args:
            results (Union[ExtractionResult, ExtractionResults, List[Union[ExtractionResult, ExtractionResults]]]): The extraction result(s) to be converted.

        Returns:
            Optional[pd.DataFrame]: A pandas DataFrame containing the extracted data, or None if the conversion fails.

        Example:
            result = extractor.extract(input_data)
            df = extractor.to_dataframe(result)
        """

        try:
            if not isinstance(results, list):
                results = [results]

            rows = []
            for result in results:
                if isinstance(result, ExtractionResults):
                    for res in result.data:
                        if "items" in res:
                            # Copy main invoice data, excluding the nested 'items'
                            main_data = res.copy()
                            line_items = main_data.pop("items", [])

                            # Flatten each item in 'items' and combine with main invoice fields
                            for item in line_items:
                                row = {**main_data, **item}  # Merge dictionaries
                                rows.append(row)
                                print(
                                    "Row added:", row
                                )  # Debug print to confirm structure
                        else:
                            rows.append(res)

            # Create a DataFrame from the list of flattened rows
            if rows:
                final_df = pd.DataFrame(rows)
            else:
                print("No rows to create DataFrame.")
                return pd.DataFrame()  # Return an empty DataFrame if no data is found

            # Return the DataFrame directly if it has columns
            return final_df if not final_df.empty else pd.DataFrame()

        except Exception as e:
            print(f"Failed to convert to DataFrame: {e}")
            return None

    def to_json(
        self,
        results: Union[
            ExtractionResult,
            ExtractionResults,
            List[Union[ExtractionResult, ExtractionResults]],
        ],
    ) -> Optional[str]:
        """
        Convert multiple extraction results to a single JSON string.

        This method converts the extracted data into a formatted JSON string.

        Args:
            results (Union[ExtractionResult, ExtractionResults, List[Union[ExtractionResult, ExtractionResults]]]): The extraction result(s) to be converted.

        Returns:
            Optional[str]: A JSON string containing the extracted data, or None if the conversion fails.
        """

        try:
            if not isinstance(results, list):
                results = [results]

            combined_data = [result.data for result in results]
            return json.dumps(combined_data, indent=2)
        except Exception as e:
            logger.error(f"Failed to convert to JSON: {e}")
            return None

    def to_markdown(
        self,
        results: Union[
            ExtractionResult,
            ExtractionResults,
            List[Union[ExtractionResult, ExtractionResults]],
        ],
    ) -> Optional[str]:
        """
        Convert one or multiple extraction results to a single Markdown formatted string.

        This method converts the extracted data into a Markdown formatted string for easy display.

        Args:
            results (Union[ExtractionResult, ExtractionResults, List[Union[ExtractionResult, ExtractionResults]]]): The extraction result(s) to be converted.

        Returns:
            Optional[str]: A Markdown string containing the extracted data, or None if the conversion fails.
        """

        try:
            if not isinstance(results, list):
                results = [results]

            markdown_output = "\n\n".join(
                self._extract_items_to_markdown(
                    result.data if isinstance(result, ExtractionResult) else item
                )
                for result in results
                for item in (
                    result.data
                    if isinstance(result, ExtractionResults)
                    else [result.data]
                )
            )
            return markdown_output
        except Exception as e:
            logger.error(f"Failed to convert to Markdown: {e}")
            return None

    def to_table(
        self,
        results: Union[
            ExtractionResult,
            ExtractionResults,
            List[Union[ExtractionResult, ExtractionResults]],
        ],
    ) -> Optional[str]:
        """
        Convert one or multiple extraction results to a table formatted string.

        This method converts the extracted data into a table format (using the `tabulate` library) for easy viewing.

        Args:
            results (Union[ExtractionResult, ExtractionResults, List[Union[ExtractionResult, ExtractionResults]]]): The extraction result(s) to be converted.

        Returns:
            Optional[str]: A table-formatted string containing the extracted data, or None if the conversion fails.
        """

        try:
            if not isinstance(results, list):
                results = [results]

            table_output = " \n\n ".join(
                self._extract_items_to_table(
                    result.data if isinstance(result, ExtractionResult) else item
                )
                for result in results
                for item in (
                    result.data
                    if isinstance(result, ExtractionResults)
                    else [result.data]
                )
            )
            return table_output
        except Exception as e:
            logger.error(f"Failed to convert to table: {e}")
            return None

    def _extract_items_to_markdown(self, data: Dict[str, Any]) -> str:
        """Convert data with nested 'items' into a more readable Markdown format."""
        markdown_sections = []
        headers = [key for key in data.keys() if key != "items"]
        values = [data[key] for key in headers]
        markdown_sections.append(tabulate([values], headers=headers, tablefmt="github"))

        if "items" in data and isinstance(data["items"], list):
            for item in data["items"]:
                markdown_sections.append(self._dict_to_markdown(item))

        return " \n\n ".join(markdown_sections)

    def _extract_items_to_table(self, data: Dict[str, Any]) -> str:
        """Convert data with nested 'items' into a more readable table format."""
        table_sections = []
        headers = [key for key in data.keys() if key != "items"]
        values = [data[key] for key in headers]
        table_sections.append(tabulate([values], headers=headers, tablefmt="grid"))

        if "items" in data and isinstance(data["items"], list):
            for item in data["items"]:
                table_sections.append(self._dict_to_table(item))

        return " \n\n ".join(table_sections)

    def _dict_to_markdown(self, data: Dict) -> str:
        headers = list(data.keys())
        rows = [[data[key] for key in headers]]
        return tabulate(rows, headers=headers, tablefmt="github")

    def _dict_to_table(self, data: Dict) -> str:
        headers = list(data.keys())
        rows = [[data[key] for key in headers]]
        return tabulate(rows, headers=headers, tablefmt="grid")
