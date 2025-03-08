# Social & Communication Connectors in indoxArcg

This guide covers integrations with social platforms and team communication tools for real-time data ingestion.

---

## Supported Connectors

### 1. TwitterTweetReader
**Social media content ingestion from Twitter**

#### Features
- User timeline extraction
- Hashtag/mention tracking
- Tweet metadata preservation

```python
from indoxArcg.data_connectors import TwitterTweetReader

# Initialize with bearer token
reader = TwitterTweetReader(bearer_token=os.getenv("TWITTER_BEARER"))

# Load tweets from handles
docs = reader.load_data(
    twitterhandles=["OpenAI", "DeepMind"],
    num_tweets=200
)
```

#### Authentication
1. Create Twitter Developer account
2. Generate Bearer Token
3. Set environment variable:
```bash
export TWITTER_BEARER="your-api-token"
```

---

### 2. DiscordChannelReader
**Community chat data extraction**

#### Features
- Channel history retrieval
- Message threading support
- User reaction tracking

```python
from indoxArcg.data_connectors import DiscordChannelReader

# Initialize with bot token
reader = DiscordChannelReader(bot_token=os.getenv("DISCORD_TOKEN"))

# Load channel messages
docs = reader.load_data(
    channel_ids=["1234567890"],
    num_messages=500
)
```

#### Authentication
1. Create Discord Bot at [Developer Portal](https://discord.com/developers)
2. Add bot to server with `read_messages` permission
3. Set environment variable:
```bash
export DISCORD_TOKEN="your-bot-token"
```

---

### 3. GoogleChat
**Enterprise team communication integration**

#### Features
- Space/room listing
- Threaded conversation export
- Google Workspace integration

```python
from indoxArcg.data_connectors import GoogleChat

# OAuth2 authentication
chat = GoogleChat()

# List available spaces
spaces = chat.list_spaces()
```

#### Authentication
1. Enable Google Chat API in [Cloud Console](https://console.cloud.google.com)
2. Download OAuth2 `credentials.json`
3. First run initiates browser authentication flow

---

## Comparison Table

| Platform  | Data Type       | Rate Limits      | History Depth | Media Support | Setup Complexity |
|-----------|-----------------|------------------|---------------|---------------|-------------------|
| Twitter   | Public Posts    | 900 tweets/15min| 7 days        | Images/GIFs   | Medium            |
| Discord   | Private Chats   | 50 reqs/second   | Unlimited*    | Files/Embeds  | High              |
| GoogleChat| Team Spaces     | 600 reqs/minute  | 25k messages  | Google Docs    | Medium            |

*Discord history limited by server retention policies

---

## Common Operations

### Filtering Content
```python
# Twitter - Exclude retweets
docs = [doc for doc in docs if "RT @" not in doc.content]

# Discord - Filter bot messages
clean_docs = [doc for doc in docs if not doc.metadata.get('is_bot')]
```

### Handling Pagination
```python
# Twitter - Iterative loading
for page in range(0, 3):
    docs += reader.load_data(
        twitterhandles=["AI_Research"],
        num_tweets=100,
        start_page=page
    )
```

---

## Troubleshooting

1. **Twitter Rate Limits**
   ```python
   import time
   try:
       docs = reader.load_data(...)
   except RateLimitError:
       time.sleep(900)  # Wait 15 minutes
   ```

2. **Discord Permission Issues**
   - Ensure bot has `View Channel` and `Read Message History` permissions
   - Check server role hierarchy

3. **Google OAuth Errors**
   - Verify `credentials.json` exists
   - Ensure redirect URI matches in Google Cloud Console

---

## Security Best Practices
- Use environment variables for tokens
- Store credentials in encrypted vaults
- Limit Discord bot permissions
- Rotate Twitter bearer tokens quarterly
```

