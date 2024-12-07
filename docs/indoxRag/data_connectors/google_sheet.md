
---

# GoogleSheet 

## Overview

The `GoogleSheet` class provides functionality to authenticate with Google Sheets using OAuth2 and read data from a specific Google Sheet. It supports reading data from a defined range within the sheet and prints the contents to the console.

## Installation

Ensure you have the following Python libraries installed:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## Quick Start

1. **Set Up Google API Credentials**

   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project (if you don't have one).
   - Enable the Google Sheets API for your project.
   - Create OAuth 2.0 Client IDs and download the `credentials.json` file.

2. **Initialize the `GoogleSheet` Class**

   ```python
   from google_sheet import GoogleSheet

   # Initialize GoogleSheet object
   sheet = GoogleSheet(creds_file='tokenGoogleSheet.json', credentials_json='credentials.json')
   ```

3. **Read Data from a Google Sheet**

   - Call the `read` method with the spreadsheet ID and optionally specify a range.

   ```python
   spreadsheet_id = 'YOUR_SPREADSHEET_ID_HERE'
   sheet.read(spreadsheet_id, 'Sheet1!A1:Z')
   ```

## Class `GoogleSheet`

### Methods

#### `__init__(self, creds_file='tokenGoogleSheet.json', credentials_json='credentials.json')`

Initializes the `GoogleSheet` object and handles OAuth2 authentication.

**Parameters:**
- `creds_file` (str): The path to the file where the credentials are stored. Default is `'tokenGoogleSheet.json'`.
- `credentials_json` (str): The path to the JSON file containing the client secrets. Default is `'credentials.json'`.


#### `_authenticate(self)`

Authenticates the user and refreshes credentials if necessary.

#### `read(self, spreadsheet_id, range_name='A1:Z')`

Reads data from a Google Sheet by its spreadsheet ID and range.

**Parameters:**
- `spreadsheet_id` (str): The ID of the Google Sheet to be read.
- `range_name` (str): The A1 notation of the range to retrieve. Default is `'A1:Z'`.

**Raises:**
- `GoogleAPIError`: If there is an issue with the Google API request.

**Notes:**
- Prints the data to the console.
- Removes the temporary credentials file after reading.

## Example Usage

```python
from google_sheet import GoogleSheet

# Initialize GoogleSheet object
sheet = GoogleSheet(credentials_json='credentials.json')

# Read content from a Google Sheet with a specific ID and range
spreadsheet_id = '1f2qu3NGL-kU_RLSvN1O3rTXi-NpAZRtQ7B0trI5xH-U'
sheet.read(spreadsheet_id, 'Sheet1!A1:Z')
```

## Notes

- Ensure that `credentials.json` is properly set up with OAuth2 credentials.
- The `tokenGoogleSheet.json` file is automatically created and managed to store the user's access and refresh tokens.
- The `read` method prints the content of the specified range to the console. Adjust the range as needed for your specific use case.

---

