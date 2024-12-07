# YoutubeTranscriptReader

YoutubeTranscriptReader is a data connector for loading transcripts from YouTube videos. It retrieves transcript text and metadata for specified YouTube video links.

**Note**: To use YoutubeTranscriptReader, users need to install the `youtube_transcript_api` package. You can install it using `pip install youtube-transcript-api`.

To use YoutubeTranscriptReader:

```python
from indox.data_connectors import YoutubeTranscriptReader

reader = YoutubeTranscriptReader()
documents = reader.load_data(ytlinks=["https://www.youtube.com/watch?v=dN0lsF2cvm4&t=44s"])
```
# Class Attributes

- **languages** [tuple]: Tuple of language codes for transcript retrieval (default is ("en",) for English).

## Methods 

**class_name()**
Returns the name of the class as a string.

**load_data(ytlinks: List[str], load_kwargs: Any) -> List[Document]**

Loads transcript data from the specified YouTube video links.

**Parameters:**
- **ytlinks** [List[str]]: List of YouTube video URLs to retrieve transcripts from.
- **load_kwargs** [Any]: Additional keyword arguments (not used in current implementation).

**Returns:**
- **List[Document]**: List of Document objects containing transcript text and metadata.

## Usage
### Setting Up the Python Environment
**Windows**
1. **Create the virtual environment:**
```bash
python -m venv indox
```
2. **Activate the virtual environment:**
```bash
indox\Scripts\activate
```
### macOS/Linux
1. **Create the virtual environment:**
```bash
python -m venv indox
```
2. **Activate the virtual environment:**
```bash
source indox/bin/activate
```

## Get Started
### Import Essential Libraries and Use YoutubeTranscriptReader

```python
from indox.data_connectors import YoutubeTranscriptReader

# Initialize the reader
reader = YoutubeTranscriptReader()

# Fetch transcripts from specific YouTube videos
video_links = ["https://www.youtube.com/watch?v=dN0lsF2cvm4&t=44s"]

documents = reader.load_data(ytlinks=video_links)

# Process the retrieved documents
for doc in documents:
    print(f"Video ID: {doc.metadata['video_id']}")
    print(f"Video Link: {doc.metadata['link']}")
    print(f"Language: {doc.metadata['language']}")
    print(f"Transcript preview: {doc.content[:200]}...")
    print("---")
```
This example demonstrates how to use YoutubeTranscriptReader to retrieve transcripts from specific YouTube videos and access their content and metadata.