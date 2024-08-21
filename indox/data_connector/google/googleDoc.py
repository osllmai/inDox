import os.path



class GoogleDoc:
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']

    def __init__(self, creds_file='tokenGoogleDoc.json', credentials_json='./credentials.json'):
        """
        Initialize the GoogleDoc object and authenticate using OAuth2.

        Parameters:
        - creds_file (str): The path to the file where the credentials are stored.
        - credentials_json (str): The path to the JSON file containing the client secrets.

        Returns:
        - None
        """

        self.creds_file = creds_file
        self.credentials_json = credentials_json
        self.creds = None
        self._authenticate()

    def _authenticate(self):
        """
        Authenticate the user and refresh credentials if necessary.

        Returns:
        - None
        """
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        try:
            if os.path.exists(self.creds_file):
                self.creds = Credentials.from_authorized_user_file(self.creds_file, self.SCOPES)

            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_json, self.SCOPES)
                    self.creds = flow.run_local_server(port=0)

                with open(self.creds_file, 'w') as token:
                    token.write(self.creds.to_json())

        except Exception as e:
            raise Exception(f"Error during authentication: {str(e)}")

    def read(self, document_id):
        """
        Read the content of a Google Doc by its document ID.

        Parameters:
        - document_id (str): The ID of the Google Doc to be read.

        Returns:
        - None
        """
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        try:
            service = build('docs', 'v1', credentials=self.creds)
            document = service.documents().get(documentId=document_id).execute()
            content = document.get('body').get('content', [])

            for element in content:
                if 'paragraph' in element:
                    for para_element in element['paragraph']['elements']:
                        text_run = para_element.get('textRun')
                        if text_run:
                            print(text_run.get('content', ''))

        except HttpError as e:
            raise Exception(f"HTTP error occurred: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred while reading the document: {str(e)}")
        finally:
            if os.path.exists(self.creds_file):
                try:
                    os.remove(self.creds_file)
                except Exception as e:
                    raise Exception(f"Failed to remove credentials file: {str(e)}")


