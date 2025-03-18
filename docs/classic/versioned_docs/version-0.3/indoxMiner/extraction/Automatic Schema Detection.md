# Automatic Schema Detection

Indox Miner’s **Automatic Schema Detection** feature provides users with the ability to dynamically infer and extract structured information from documents without manually defining a schema. By leveraging the `AutoSchema` class, Indox Miner can automatically identify fields, detect data structures, and apply validation rules to ensure data quality.

## Purpose of Automatic Schema Detection

Automatic Schema Detection enables:

- **Dynamic extraction** of fields from unstructured documents, reducing the need for predefined schemas.
- **Flexibility** in handling a variety of document layouts, including tables, forms, and other structured data.
- **Automated validation** of extracted fields based on inferred types and rules, ensuring accurate and consistent data.

## How Automatic Schema Detection Works

The `AutoSchema` class in Indox Miner is designed to:

1. **Analyze Document Structure**: `AutoSchema` examines the text to identify patterns resembling tables, forms, and key-value pairs.
2. **Infer Field Types**: Using the `AutoDetectedField` class, it assigns data types (e.g., date, number, boolean) to detected fields.
3. **Generate Extraction Rules**: Validation rules are automatically generated based on field types, ensuring data consistency and quality.
4. **Validate Extraction Results**: After extraction, detected data is validated against inferred rules, and any inconsistencies are flagged.

## Components of Automatic Schema Detection

### 1. `AutoDetectedField`

Represents an automatically detected field with properties like:

- **Name**: Field name, inferred from the text.
- **Field Type**: Type of data (e.g., string, number, date, boolean).
- **Description**: Description based on inferred type and document structure.
- **Required**: Whether the field is essential for extraction.
- **Rules**: Validation rules for the field, generated based on type and context.

### 2. `AutoExtractionRules`

Contains rules for auto-extraction and validation, including:

- **Min/Max Length**: Ensures text fields have appropriate character length.
- **Pattern**: Regular expression patterns for formats like dates or specific identifiers.
- **Min/Max Value**: Specifies valid ranges for numeric fields.
- **Allowed Values**: List of valid values for categorical fields, such as booleans.

### 3. `AutoSchema`

Manages the overall automatic detection and extraction process, including:

- **Field Detection**: Identifies fields by analyzing the document for table structures and form-like fields.
- **Structure Inference**: Automatically detects document layout (e.g., table, form).
- **Prompt Generation**: Generates a prompt that instructs the extraction model on the detected fields and structures.

## Usage of Automatic Schema Detection

To use `AutoSchema` for automatic schema detection, initialize the class, pass the document text, and let `AutoSchema` infer the structure and fields.

### Example

```python
from indox_miner.autoschema import AutoSchema

# Initialize AutoSchema instance
auto_schema = AutoSchema()

# Document text to analyze
document_text = """
Name: John Doe
Date of Birth: 1990-01-01
Total Amount: $150.00

Item Description    Quantity   Unit Price   Total
Item 1              2          $20.00       $40.00
Item 2              5          $15.00       $75.00
"""

# Infer structure and fields
auto_schema.infer_structure(document_text)

# Generate prompt based on detected fields
prompt = auto_schema.to_prompt(document_text)
print(prompt)
```

### Output Prompt Example

```plaintext
Task: Extract structured information from the given text using automatic field detection.

Detected Fields:
- Name (string): Automatically detected string field from form
- Date of Birth (date): Automatically detected date field from form
- Total Amount (number): Automatically detected number field from form
- Item Description (string): Automatically detected string field from table column
- Quantity (number): Automatically detected number field from table column
- Unit Price (number): Automatically detected number field from table column
- Total (number): Automatically detected number field from table column

Extraction Requirements:
1. Extract all detected fields maintaining their original names
2. Use appropriate data types for each field:
   - Dates in ISO format (YYYY-MM-DD)
   - Numbers as numeric values (not strings)
   - Boolean values as true/false
   - Lists as arrays
   - Nested data as objects
3. Preserve any detected relationships between fields
4. Return data in JSON format

Text to analyze:
{Name: John Doe...}
```

## How `AutoSchema` Detects Structure

### Table Detection

The `_looks_like_table` method detects table-like structures by analyzing:

- **Delimiter rows**: Lines with multiple delimiters (e.g., tabs, pipes `|`).
- **Consistent Spacing**: Lines with uniform spacing patterns across columns.
- **Header Indicators**: Rows with indicators such as hyphens (`---`) or equals (`===`) which often denote headers.

### Form Field Detection

The `_detect_form_fields` method identifies fields with:

- **Label-Value Patterns**: Lines containing labels followed by values, separated by whitespace or punctuation (e.g., `Name: John Doe`).
- **Keyword Filtering**: Ignores known non-field phrases, focusing only on relevant label-value pairs.

### Field Type Inference

The `_infer_field_type` method determines field types based on:

- **Patterns**: Recognizes dates, numbers, and booleans through regular expressions.
- **Content Indicators**: Uses delimiters (e.g., commas, semicolons) to detect lists and categorizes remaining data as strings.

### Rule Generation

The `_generate_rules` method automatically creates validation rules according to the inferred field type, ensuring:

- **String Fields**: Min and max character length limits.
- **Numeric Fields**: Range limits (min and max values).
- **Date Fields**: Pattern matching (e.g., ISO format).
- **Boolean Fields**: Restriction to true/false values.

## Validating Extraction Results

After extracting data, the `validate_extraction` method validates each field according to its rules, identifying any inconsistencies or missing values. This ensures that the extracted data meets predefined quality standards.

### Example of Validation

```python
extracted_data = {
    "Name": "John Doe",
    "Date of Birth": "1990-01-01",
    "Total Amount": 150.0,
    "Item Description": ["Item 1", "Item 2"],
    "Quantity": [2, 5],
    "Unit Price": [20.0, 15.0],
    "Total": [40.0, 75.0]
}

# Validate extracted data
errors = auto_schema.validate_extraction(extracted_data)
if errors:
    print("Validation errors found:", errors)
else:
    print("All fields are valid.")
```

## Conclusion

The **Automatic Schema Detection** feature in Indox Miner simplifies data extraction from unstructured documents by dynamically detecting fields and validating data against inferred rules. This functionality reduces setup time, enhances flexibility, and maintains data quality, making Indox Miner a powerful tool for diverse document processing needs.
