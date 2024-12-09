---

# GoogleDoc 

## Overview

The `GoogleDoc` class enables users to read the content of a Google Document using the Google Docs API. It handles OAuth2 authentication and provides methods to access and print the content of a specified Google Doc.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## Quick Start

1. **Initialize the `GoogleDoc` Class**

   ```python
   from indoxRag.data_connectors import GoogleDoc

   # Initialize GoogleDoc with custom paths for credentials files
   doc = GoogleDoc(creds_file='tokenGoogleDoc.json', credentials_json='./credentials.json')
   ```

2. **Read the Document Content**

   - Call the `read` method with the document ID you want to access.

   ```python
   document_id = 'YOUR DOCUMENT ID HERE'
   doc.read(document_id)
   ```

## Class `GoogleDoc`

### Methods

#### `__init__(self, creds_file='tokenGoogleDoc.json', credentials_json='./credentials.json')`

Initializes the `GoogleDoc` object and handles OAuth2 authentication.

**Parameters:**
- `creds_file` (str): The path to the file where the credentials are stored. Default is `'tokenGoogleDoc.json'`.
- `credentials_json` (str): The path to the JSON file containing the client secrets. Default is `'./credentials.json'`.


**Notes:**
- Calls `_authenticate` to handle the authentication process.

#### `def _authenticate(self)`

Handles user authentication and refreshes credentials if necessary.

**Notes:**
- Reads existing credentials from `creds_file` if available.
- If credentials are not valid or do not exist, it performs the OAuth2 flow to obtain new credentials and saves them to `creds_file`.

#### `read(self, document_id: str) -> None`

Reads the content of a Google Doc by its document ID.

**Parameters:**
- `document_id` (str): The ID of the Google Doc to be read.

**Notes:**
- Retrieves and prints the text content of the document.
- Iterates through the document’s content and prints text from paragraph elements.
- Deletes the `creds_file` after use.

### Error Handling

- Raises an exception if there is an issue with the Google API request or authentication process.

## Example Usage

```python
from indoxRag.data_connectors import GoogleDoc

# Initialize GoogleDoc object
doc = GoogleDoc(credentials_json='./credentials.json')

# Read the content of a document
document_id = 'YOUR DOCUMENT ID HERE'
doc.read(document_id)
```

## Notes

- Ensure that your `credentials.json` file contains valid OAuth2 credentials and that the `tokenGoogleDoc.json` file is writable.
- The `document_id` must be a valid Google Document ID that you have access to.

---
