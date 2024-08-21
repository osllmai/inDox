import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleSheet:
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

    def __init__(self, creds_file='tokenGoogleSheet.json', credentials_json='credentials.json'):
        """
        Initialize the GoogleSheet object and authenticate using OAuth2.

        Parameters:
        - creds_file (str): The path to the file where the credentials are stored.
        - credentials_json (str): The path to the JSON file containing the client secrets.

        Returns:
        - None
        """
        self.creds = None
        self.creds_file = creds_file
        self.credentials_json = credentials_json
        self._authenticate()

    def _authenticate(self):
        """
        Authenticate the user and refresh credentials if necessary.

        Returns:
        - None
        """
        try:
            if os.path.exists(self.creds_file):
                print("Loading credentials from file.")
                self.creds = Credentials.from_authorized_user_file(self.creds_file, self.SCOPES)
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    print("Refreshing expired credentials.")
                    self.creds.refresh(Request())
                else:
                    print("Authenticating new credentials.")
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_json, self.SCOPES)
                    self.creds = flow.run_local_server(port=0)
                with open(self.creds_file, 'w') as token:
                    token.write(self.creds.to_json())
                    print("Credentials saved to token file.")
        except Exception as e:
            print(f"An error occurred during authentication: {e}")
            raise

    def read(self, spreadsheet_id, range_name='A1:Z'):
        """
        Read data from a Google Sheet by its spreadsheet ID and range.

        Parameters:
        - spreadsheet_id (str): The ID of the Google Sheet to be read.
        - range_name (str): The A1 notation of the range to retrieve. Default is 'A1:Z'.

        Returns:
        - None
        """
        try:
            service = build('sheets', 'v4', credentials=self.creds)
            sheet = service.spreadsheets()
            print(f"Fetching data from spreadsheet ID: {spreadsheet_id}, Range: {range_name}")
            result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
            values = result.get('values', [])

            if not values:
                print('No data found.')
            else:
                print('Data retrieved successfully:')
                for row in values:
                    print(', '.join(row))
        except HttpError as error:
            print(f"An error occurred with the Google Sheets API: {error}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            if os.path.exists(self.creds_file):
                os.remove(self.creds_file)
                print(f"Credentials file {self.creds_file} deleted.")

