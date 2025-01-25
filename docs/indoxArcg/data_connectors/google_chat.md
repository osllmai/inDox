---

# GoogleChat 

## Overview

The `GoogleChat` class allows users to interact with the Google Chat API to list spaces (chat rooms) within Google Chat. It handles the authentication process using OAuth 2.0 and manages credentials, including refreshing them when necessary.

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client google-chat
```

## Quick Start

1. **Initialize the `GoogleChat` Class**

   To use the Google Chat API, you need to authenticate using OAuth 2.0. This requires a `credentials.json` file that contains your application's client ID and client secret.

   **Initialize the Class:**

   ```python
   from indoxArcg.data_connectors import GoogleChat

   # Initialize GoogleChat class
   chat = GoogleChat()
   ```

2. **List Google Chat Spaces**

   - Call the `list_spaces` method to list the spaces in Google Chat.

   ```python
   spaces = chat.list_spaces()
   ```

## Class `GoogleChat`

### Attributes

- `SCOPES`: The scopes required for the Google Chat API, specifically `https://www.googleapis.com/auth/chat.spaces.readonly`.
- `TOKEN_FILE`: The path to the file where the OAuth2 token is stored.
- `CREDENTIALS_FILE`: The path to the file containing the OAuth2 client secrets.

### Methods

#### `__init__(self)`

Initializes the `GoogleChat` object and handles the authentication process.

**Parameters:**
- None

**Returns:**
- None

**Notes:**
- Calls the `_authenticate` method to handle OAuth2 authentication and create the API client.

#### `_authenticate(self)`

Handles the OAuth2 authentication flow and creates the Google Chat API client.

**Notes:**
- Uses stored credentials if available and valid.
- Refreshes credentials if they have expired and a refresh token is available.
- Saves new credentials to a token file if a new authentication flow is initiated.
- Initializes the Google Chat API client.

#### `list_spaces(self)`

Lists the spaces in Google Chat and prints the details of each space.

**Notes:**
- This method filters for spaces of type "SPACE" in Google Chat.
- If an error occurs during the API call, the method returns `None`.

### Error Handling

- Errors during authentication, credential saving, and API client creation are caught, printed, and re-raised.
- Errors during space listing are caught and logged, and `None` is returned.

## Example Usage

```python
from indoxArcg.data_connectors import GoogleChat

# Initialize GoogleChat class
chat = GoogleChat()

# List spaces in Google Chat
spaces = chat.list_spaces()
```

## Notes

- Ensure you have a valid `credentials.json` file containing your OAuth2 client ID and client secret.
- The `token.json` file is created after successful authentication and is used to store credentials for future runs.

---
