# Google Workspace Connectors in indoxArcg

This guide covers integrations with Google Workspace services for document management and collaboration.

---

## Supported Connectors

### 1. GoogleDoc
**Google Docs content retrieval**

#### Features
- Full document text extraction
- Paragraph-level metadata
- Real-time collaboration awareness

```python
from indoxArcg.data_connectors import GoogleDoc

doc = GoogleDoc(creds_file='token.json')
content = doc.read("1aBcDeFgHiJkLmNoPqRsTuVwXyZ")
```

#### Installation
```bash
pip install google-api-python-client google-auth
```

---

### 2. GoogleDrive
**Cloud file storage integration**

#### Features
- Multi-format support (Docs, Sheets, PDF, etc.)
- File metadata extraction
- Version history access

```python
from indoxArcg.data_connectors import GoogleDrive

drive = GoogleDrive()
drive.read("0B9qHj-hJ5W-rdVJXRlZzVUJtWVE", mime_type="application/pdf")
```

#### Installation
```bash
pip install PyPDF2 python-pptx python-docx beautifulsoup4
```

---

### 3. GoogleSheet
**Spreadsheet data processing**

#### Features
- Range-based data extraction
- Formula resolution
- Tabular metadata preservation

```python
from indoxArcg.data_connectors import GoogleSheet

sheet = GoogleSheet()
data = sheet.read("1f2qu3NGL-kU_RLSvN1O3rTXi-NpAZRtQ7B0trI5xH-U", 
                range="Q2!A1:Z1000")
```

#### Installation
```bash
pip install pandas google-auth-oauthlib
```

---

## Comparison Table

| Feature          | GoogleDoc       | GoogleDrive     | GoogleSheet     |
|------------------|-----------------|-----------------|-----------------|
| Content Type     | Text Documents  | 100+ File Types | Spreadsheet Data|
| Auth Scope       | docs.readonly   | drive.readonly  | sheets.readonly |
| Rate Limits      | 300 req/min     | 1000 req/100s   | 500 req/100s    |
| Data Structure   | Hierarchical    | File Tree       | Tabular         |
| Version Control  | ✅              | ✅              | ❌              |

---

## Common Setup

### Authentication Workflow
1. Enable APIs in [Google Cloud Console](https://console.cloud.google.com/):
   - Google Docs API
   - Google Drive API
   - Google Sheets API
2. Download `credentials.json`
3. First-run OAuth flow:
```python
# Shared auth pattern
from google.oauth2.credentials import Credentials

creds = Credentials.from_authorized_user_file('token.json')
```

---

## Advanced Operations

### Batch Processing
```python
# Process multiple Google Docs
doc_ids = ["doc1_id", "doc2_id", "doc3_id"]
for doc_id in doc_ids:
    content = GoogleDoc().read(doc_id)
    process_content(content)
```

### Drive Search
```python
from googleapiclient.discovery import build

service = build('drive', 'v3', credentials=creds)
results = service.files().list(
    q="name contains 'report' and mimeType='application/pdf'",
    pageSize=10
).execute()
```

### Sheet Data Transformation
```python
# Convert to pandas DataFrame
import pandas as pd

values = sheet.read(spreadsheet_id, range="Sales!A1:Z1000")
df = pd.DataFrame(values[1:], columns=values[0])
```

---

## Troubleshooting

### Common Issues
1. **Authentication Errors**
   - Verify `credentials.json` exists
   - Check OAuth consent screen configuration
   - Ensure redirect URIs match

2. **Rate Limits**
```python
from time import sleep
from googleapiclient.errors import HttpError

try:
    doc.read(doc_id)
except HttpError as e:
    if e.resp.status == 429:
        sleep(60)  # Backoff 1 minute
```

3. **File Permissions**
   - Share documents with service account email
   - Enable domain-wide delegation for workspace accounts

---

## Security Best Practices
1. Use least-privilege scopes:
   ```python
   SCOPES = [
       'https://www.googleapis.com/auth/documents.readonly',
       'https://www.googleapis.com/auth/drive.metadata.readonly'
   ]
   ```
2. Store tokens encrypted
3. Rotate credentials quarterly
4. Monitor API usage in Cloud Console

---

## Performance Tips
- Enable batch processing for bulk operations
- Use fields parameter to limit response size
- Cache frequently accessed documents
- Prefer document IDs over title searches

---
