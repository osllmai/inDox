# Multimedia & Geospatial Connectors in indoxArcg

This guide covers integrations with multimedia platforms and location services for video and geospatial data processing.

---

## Supported Connectors

### 1. YoutubeTranscriptReader
**YouTube video transcript extraction**

#### Features
- Multi-language caption support
- Timestamp metadata
- Automatic translation handling

```python
from indoxArcg.data_connectors import YoutubeTranscriptReader

reader = YoutubeTranscriptReader(languages=("en", "es"))
docs = reader.load_data(
    ytlinks=["https://youtu.be/dQw4w9WgXcQ"],
    include_timestamps=True
)
```

#### Installation
```bash
pip install youtube-transcript-api
```

---

### 2. MapsTextSearch
**Geocoding and location data retrieval**

#### Features
- Address → coordinates conversion
- Reverse geocoding support
- Multiple result ranking

```python
from indoxArcg.data_connectors import MapsTextSearch

searcher = MapsTextSearch(user_agent="my-geo-app/1.0")
locations = searcher.search_address("Eiffel Tower, Paris")
```

#### Installation
```bash
pip install geopy
```

---

## Comparison Table

| Feature               | YoutubeTranscriptReader | MapsTextSearch      |
|-----------------------|--------------------------|---------------------|
| Data Type             | Video Captions           | Location Metadata   |
| API Source            | YouTube Data API         | OpenStreetMap       |
| Rate Limits           | 1M chars/day             | 1 req/second        |
| Authentication        | None                     | User Agent Required |
| Output Format         | Timed Text               | GeoJSON             |
| Precision Control     | Language Selection       | Result Thresholding |

---

## Common Operations

### Video Content Processing
```python
# Merge multiple video transcripts
combined = "\n".join([doc.content for doc in docs])
```

### Location Data Enrichment
```python
# Convert to GeoJSON format
import geojson

feature = geojson.Feature(
    geometry=geojson.Point((location.longitude, location.latitude)),
    properties={"address": location.address}
)
```

### Temporal Analysis
```python
# Extract timing data from YouTube transcripts
timed_content = [
    (entry['start'], entry['text']) 
    for entry in doc.metadata['timed_transcript']
]
```

---

## Troubleshooting

### Common Issues
1. **Missing Transcripts**
   ```python
   YoutubeTranscriptReader(
       fallback_ASR=True  # Use automatic speech recognition
   ).load_data(links)
   ```

2. **Geocoding Ambiguity**
   ```python
   MapsTextSearch(
       exactly_one=False  # Get multiple results
   ).search_address("Springfield")
   ```

3. **API Rate Limits**
   ```python
   from time import sleep

   for video in playlist:
       docs = reader.load_data([video])
       sleep(2)  # Respect YouTube API limits
   ```

4. **Encoding Issues**
   ```python
   YoutubeTranscriptReader(
       languages=("en",),
       preserve_formatting=False
   )
   ```

---

## Security Best Practices
- **YouTube**: Monitor API usage through Google Cloud Console
- **Maps**: 
  - Obfuscate exact coordinates in logs
  - Cache frequently queried locations
  - Comply with OSM's Tile Usage Policy
- General:
  - Rotate user agents regularly
  - Store API keys in encrypted secrets
  - Limit PII exposure in location data

---

## Performance Optimization

### Batch Processing
```python
# Bulk geocoding with rate limiting
addresses = ["Paris", "London", "New York"]
results = []
for addr in addresses:
    results.append(searcher.search_address(addr))
    time.sleep(1.1)  # Stay under OSM limits
```

### Selective Loading
```python
# YouTube partial transcript loading
reader.load_data(
    ytlinks=[video_url],
    start_time=60,  # Start at 1 minute
    end_time=300    # End at 5 minutes
)
```

---
