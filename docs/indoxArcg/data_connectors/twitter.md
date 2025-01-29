# TwitterTweetReader

TwitterTweetReader is a data connector for loading tweets from specified Twitter handles. It retrieves tweets and user metadata using the Twitter API.

**Note**: To use TwitterTweetReader, users need to install the `tweepy` package and have access to the Twitter API. You can install tweepy using `pip install tweepy`.

For API access, follow the instructions at 'https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api'.

To use TwitterTweetReader:

```python
from indoxArcg.data_connectors import TwitterTweetReader

reader = TwitterTweetReader(bearer_token="your_bearer_token")
documents = reader.load_data(twitterhandles=["username1", "username2"])
```

# Class Attributes

- **bearer_token** [str]: Bearer token for Twitter API authentication.
- **num_tweets** [Optional[int]]: Default number of tweets to fetch per user.

\***\*init**(bearer_token: str, num_tweets: Optional[int] = 100):\*\*

Initializes the TwitterTweetReader with the given bearer token and optional number of tweets.

**class_name():**

Returns the name of the class as a string.

**load_data(twitterhandles: List[str], num_tweets: Optional[int] = None, load_kwargs: Any) -> List[Document]**

Loads tweets from the specified Twitter handles.

**Parameters:**

- **twitterhandles** [List[str]]: List of Twitter usernames to fetch tweets from.
- **num_tweets** [Optional[int]]: Number of tweets to fetch per user (overrides the default if specified).
- **load_kwargs** [Any]: Additional keyword arguments (not used in current implementation).

**Returns:**

- **List[Document]**: List of Document objects containing tweets and metadata.

## Usage

### Setting Up the Python Environment

**Windows**

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
indoxArcg\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
source indoxArcg/bin/activate
```

## Get Started

### Import Essential Libraries and Use TwitterTweetReader

```python
from indoxArcg.data_connectors import TwitterTweetReader
import os
from dotenv import load_dotenv

load_dotenv('twitter.env')
# Initialize the reader with your bearer token
twitter_token = os.environ['twitter_token']
reader = TwitterTweetReader(bearer_token=twitter_token)

# Fetch tweets from specific Twitter handles
twitter_handles = ["OpenAI", "DeepMind"]
documents = reader.load_data(twitterhandles=twitter_handles, num_tweets=50)

# Process the retrieved documents
for doc in documents:
    print(f"Username: {doc.metadata['username']}")
    print(f"User ID: {doc.metadata['user_id']}")
    print(f"Number of tweets: {doc.metadata['num_tweets']}")
    print(f"Tweets preview: {doc.content[:200]}...")
    print("---")
```

This example demonstrates how to use TwitterTweetReader to retrieve tweets from specific Twitter handles and access their content and metadata.
