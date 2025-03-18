# Output Types

Indox Miner provides multiple output formats, enabling users to view and export extracted data in the most convenient form for their needs. The available output types include structured formats such as **DataFrame**, **JSON**, **Markdown**, and **Table**. Each format offers unique benefits, depending on how the data will be used or shared.

## Available Output Types

| Output Type   | Description                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| **DataFrame** | A pandas DataFrame, useful for data analysis and manipulation in Python.    |
| **JSON**      | A JSON string, ideal for storing or transmitting structured data.           |
| **Markdown**  | Markdown format for easily readable text with structured tables.            |
| **Table**     | A formatted table string, suitable for displaying data in terminal outputs. |

## Using Output Types in Indox Miner

The `Extractor` class in Indox Miner includes methods for each output type, making it simple to format extracted data as needed.

### 1. **DataFrame Output**

The `to_dataframe()` method converts extraction results to a pandas DataFrame, which is particularly useful for data analysis, manipulation, and visualization.

#### Example

```python
from indox_miner.extractor import Extractor
from indox_miner.llms import OpenAi
from indox_miner.schema import Schema

# Set up the extractor with LLM and schema
llm = OpenAi(api_key="your-api-key", model="gpt-4")
extractor = Extractor(llm=llm, schema=Schema.Invoice)

# Extract data and convert to DataFrame
result = extractor.extract("Your document text here")
df = extractor.to_dataframe(result)

# Display or manipulate DataFrame
print(df)
```

### 2. **JSON Output**

The `to_json()` method converts results to a JSON string, which is ideal for storing or transmitting data in a structured and compact format.

#### Example

```python
json_output = extractor.to_json(result)
print(json_output)
```

### 3. **Markdown Output**

The `to_markdown()` method generates a Markdown-formatted string, which is highly readable and can be directly used in Markdown-compatible platforms like GitHub or documentation tools.

#### Example

```python
markdown_output = extractor.to_markdown(result)
print(markdown_output)
```

### 4. **Table Output**

The `to_table()` method creates a formatted table string, which is especially useful for displaying extracted data directly in the terminal or CLI environments.

#### Example

```python
table_output = extractor.to_table(result)
print(table_output)
```

## Choosing the Right Output Type

| Use Case                  | Recommended Output Type |
| ------------------------- | ----------------------- |
| Data Analysis             | **DataFrame**           |
| API Integration / Storage | **JSON**                |
| Documentation or Reports  | **Markdown**            |
| Command Line Display      | **Table**               |

## Customizing Output Structure

1. **Field Selection**: The output format includes all fields defined in the schema by default. To customize this, adjust the schema or filter fields in the resulting DataFrame or JSON.
2. **Nested and List Fields**: Indox Miner preserves nested structures and lists in output, allowing easy handling of complex data (e.g., items in an invoice or medications in a medical record).
3. **Validation and Formatting**: All output formats respect validation rules specified in the schema, ensuring data quality and consistency.

## Example: Extracting and Exporting Data

Here’s a complete example of extracting data from a document, then exporting it in different formats:

```python
# Perform extraction
result = extractor.extract("Sample document text here")

# Export to different formats
df = extractor.to_dataframe(result)
json_output = extractor.to_json(result)
markdown_output = extractor.to_markdown(result)
table_output = extractor.to_table(result)

# Display or save outputs
print("DataFrame:\n", df)
print("JSON:\n", json_output)
print("Markdown:\n", markdown_output)
print("Table:\n", table_output)
```

## Conclusion

Indox Miner’s flexible output options make it easy to integrate extracted data into various workflows, whether for analysis, documentation, or display. By selecting the appropriate output type, users can maximize the utility and accessibility of their data.
