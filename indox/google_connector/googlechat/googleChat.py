from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.apps import chat_v1 as google_chat


class GoogleChat:
    """A class to interact with the Google Chat API."""

    SCOPES = ['https://www.googleapis.com/auth/chat.spaces.readonly']
    TOKEN_FILE = 'token.json'
    CREDENTIALS_FILE = 'credentials.json'

    def __init__(self):
        """
        Initialize the GoogleChat class and authenticate the user.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - Calls the _authenticate method to handle authentication and set up the API client.
        """
        self.creds = None
        self.client = None
        self._authenticate()

    def _authenticate(self):
        """
        Authenticate and create the API client.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - Uses stored credentials if available and valid; otherwise, performs a new authentication flow.
        - Refreshes the credentials if expired and a refresh token is available.
        - Saves the new credentials to a token file.
        - Sets up the Google Chat API client using the authenticated credentials.
        """
        if os.path.exists(self.TOKEN_FILE):
            self.creds = Credentials.from_authorized_user_file(self.TOKEN_FILE, self.SCOPES)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.CREDENTIALS_FILE, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open(self.TOKEN_FILE, 'w') as token:
                token.write(self.creds.to_json())
        self.client = google_chat.ChatServiceClient(
            credentials=self.creds,
            client_options={"scopes": self.SCOPES}
        )

    def list_spaces(self):
        """
        List spaces in Google Chat.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - Prints the details of each space found in Google Chat.
        - Prints an error message if an exception occurs during the process.
        """
        try:
            request = google_chat.ListSpacesRequest(
                filter='space_type = "SPACE"'
            )
            page_result = self.client.list_spaces(request)
            for response in page_result:
                print(response)
        except Exception as error:
            print(f'An error occurred: {error}')
