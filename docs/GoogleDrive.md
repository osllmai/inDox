
---

# GoogleDrive 

## Overview

The `GoogleDrive` class provides functionality to authenticate with Google Drive using OAuth2 and read the content of various file types from Google Drive. It supports multiple file formats including plain text, Jupyter Notebooks, PDF, Excel, CSV, PowerPoint, Word documents, JSON, XML, and HTML.

## Installation

Ensure you have the following Python libraries installed:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pandas nbformat PyPDF2 python-pptx python-docx beautifulsoup4
```

## Quick Start

1. **Set Up Google API Credentials**

   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project (if you don't have one).
   - Enable the Google Drive API for your project.
   - Create OAuth 2.0 Client IDs and download the `credentials.json` file.

2. **Initialize the `GoogleDrive` Class**

   ```python
   from google_drive import GoogleDrive

   # Initialize GoogleDrive object
   drive = GoogleDrive(creds_file='tokenGoogleDrive.json', credentials_json='credentials.json')
   ```

3. **Read a File from Google Drive**

   - Call the `read` method with the file ID from Google Drive.

   ```python
   file_id = 'YOUR_FILE_ID_HERE'
   drive.read(file_id)
   ```

## Class `GoogleDrive`

### Methods

#### `__init__(self, creds_file='tokenGoogleDrive.json', credentials_json='credentials.json')`

Initializes the `GoogleDrive` object and handles OAuth2 authentication.

**Parameters:**
- `creds_file` (str): The path to the file where the credentials are stored. Default is `'tokenGoogleDrive.json'`.
- `credentials_json` (str): The path to the JSON file containing the client secrets. Default is `'credentials.json'`.



#### `read(self, file_id)`

Reads and prints the content of a file from Google Drive based on its MIME type.

**Parameters:**
- `file_id` (str): The ID of the file in Google Drive to be read.

**Raises:**
- `Exception`: For any errors encountered during file reading or processing.

### File Type Handling

- **Plain Text (`text/plain`)**: Reads and prints plain text files.
- **Jupyter Notebook (`application/x-ipynb+json`)**: Reads and prints the code cells from Jupyter Notebooks.
- **PDF (`application/pdf`)**: Extracts and prints text from PDF files.
- **Excel (`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`)**: Reads and prints content from Excel files.
- **CSV (`text/csv`)**: Reads and prints content from CSV files.
- **PowerPoint (`application/vnd.openxmlformats-officedocument.presentationml.presentation`)**: Extracts and prints text from PowerPoint slides.
- **Word Document (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`)**: Reads and prints content from Word documents.
- **JSON (`application/json`)**: Reads and prints JSON data.
- **XML (`application/xml`)**: Parses and prints XML data.
- **HTML (`text/html`)**: Parses and prints HTML content.

### Error Handling

Errors during file reading or processing are caught and reported. Temporary credentials file is removed after processing.

## Example Usage

```python
from google_drive import GoogleDrive

# Initialize GoogleDrive object
drive = GoogleDrive(creds_file='tokenGoogleDrive.json', credentials_json='credentials.json')

# Read content from a file with a specific file ID
file_id = 'YOUR_FILE_ID_HERE'
drive.read(file_id)
```

## Notes

- Ensure that `credentials.json` is properly set up with OAuth2 credentials.
- The `tokenGoogleDrive.json` file is automatically created and managed to store the user's access and refresh tokens.

---

