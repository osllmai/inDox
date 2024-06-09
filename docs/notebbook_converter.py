import nbformat
from nbconvert import MarkdownExporter

# Define the paths
input_notebook = 'examples/unstructuredSplit.ipynb'  # Replace with your notebook filename
output_markdown = 'examples/unstructuredSplit.md'    # The output markdown filename

# Load the notebook
with open(input_notebook, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Convert the notebook to Markdown
exporter = MarkdownExporter()
body, _ = exporter.from_notebook_node(notebook)

# Write the markdown output to a file
with open(output_markdown, 'w', encoding='utf-8') as f:
    f.write(body)

print(f"Notebook {input_notebook} has been successfully converted to {output_markdown}.")
