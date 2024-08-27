# DiscordChannelReader

DiscordChannelReader is a data connector for loading messages from specified Discord channels. It retrieves messages using a Discord bot.
**Important:**: To use DiscordChannelReader, you need to create a Discord bot and obtain its token. For more information on creating a bot and obtaining its token, visit https://discord.com/developers/docs/intro.
**Note:** To use DiscordChannelReader, users need to install the `discord.py` package. You can install it using `pip install discord.py.`

To use DiscordChannelReader:

```python
from indox.data_connectors import DiscordChannelReader

reader = DiscordChannelReader(bot_token="your_bot_token")
documents = reader.load_data(channel_ids=["channel_id1", "channel_id2"])
```
# Class Attributes

- **bot_token** [str]: The bot token for your Discord bot.
- **num_messages** [Optional[int]]: Default number of messages to retrieve from a channel.

**init(bot_token: str, num_messages: Optional[int] = 100):**

Initializes the DiscordChannelReader with the given bot token and optional number of messages.
**class_name():**

Returns the name of the class as a string ("DiscordChannelReader").

**load_data(channel_ids: List[str], num_messages: Optional[int] = None, load_kwargs: Any) -> List[Document]**

Loads messages from specified Discord channels.

**Parameters:**
- **channel_ids** [List[str]]: List of Twitter usernames to fetch tweets from.
- **num_messages** [Optional[int]]: The maximum number of messages to retrieve from each channel (overrides the default if specified).
- **load_kwargs**  [Any]: Additional keyword arguments (not used in current implementation).

**Returns:**
- **List[Document]**: List of Document objects containing channel messages and metadata.

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
### Import Essential Libraries and Use DiscordChannelReader

```python
from indox.data_connectors import DiscordChannelReader
import os
from dotenv import load_dotenv
import nest_asyncio

# Apply the nest_asyncio patch
nest_asyncio.apply()

load_dotenv('discord.env')
# Initialize the reader with your bearer token
discord_token = os.environ['discord_token']
reader = DiscordChannelReader(bot_token=discord_token)

# Fetch messages from specific Discord channels
channel_ids = ["channel_id1", "channel_id2"]
documents = reader.load_data(channel_ids=channel_ids, num_messages=50)

# Process the retrieved documents
for doc in documents:
    print(f"Channel ID: {doc.metadata['channel_id']}")
    print(f"Channel Name: {doc.metadata['channel_name']}")
    print(f"Number of messages: {doc.metadata['num_messages']}")
    print(f"Messages preview: {doc.content[:200]}...")
    print("---")
```
This example demonstrates how to use DiscordChannelReader to retrieve messages from specific Discord channels and access their content and metadata.