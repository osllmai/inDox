from indox.data_connector.utils import Document
from typing import Any, List, Optional

class DiscordChannelReader:
    """Reads messages from a specified Discord channel using a Discord bot.

    **Important:** You need to create a Discord bot and get its token to use this class.
    See https://discord.com/developers/docs/intro for more information on creating a bot
    and obtaining its token.

    Attributes:
        bot_token: The bot token for your Discord bot.
        num_messages: The default number of messages to retrieve from a channel.
    """

    bot_token: str
    num_messages: Optional[int]

    def __init__(
        self,
        bot_token: str,
        num_messages: Optional[int] = 100,
    ) -> None:
        """Initializes the DiscordChannelReader object.

        Args:
            bot_token (str): The bot token for your Discord bot.
            num_messages (Optional[int]): The default number of messages to retrieve
                from a channel. Defaults to 100.

        Raises:
            ImportError: If the `discord.py` library is not installed.
        """
        try:
            import discord
        except ImportError:
            raise ImportError(
                "`discord` package not found, please run `pip install discord.py`"
            )

        self.bot_token = bot_token
        self.num_messages = num_messages

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name ("DiscordChannelReader")."""
        return "DiscordChannelReader"

    def load_data(
        self,
        channel_ids: List[str],
        num_messages: Optional[int] = None,
        **load_kwargs: Any
    ) -> List[Document]:
        """Loads messages from specified Discord channels.

        Args:
            channel_ids (List[str]): A list of Discord channel IDs to read messages from.
            num_messages (Optional[int]): The maximum number of messages to retrieve
                from each channel. Defaults to the default value set during initialization
                or 100 if not set.

        Returns:
            List[Document]: A list of Document instances containing channel messages
                and metadata.

        Raises:
            ValueError: If a channel ID is not found in the Discord server.
        """
        import discord
        client = discord.Client(intents=discord.Intents.default())
        documents = []

        @client.event
        async def on_ready():
            for channel_id in channel_ids:
                channel = client.get_channel(int(channel_id))
                if channel:
                    messages = []
                    async for message in channel.history(limit=num_messages or self.num_messages):
                        messages.append(message.content)
                    response = "\n".join(messages)
                    metadata = {
                        "channel_id": channel_id,
                        "channel_name": channel.name,
                        "num_messages": len(messages),
                    }
                    documents.append(Document(source="Discord", content=response, metadata=metadata))
                else:
                    raise ValueError(f"Channel ID '{channel_id}' not found in Discord server.")

            await client.close()

        client.run(self.bot_token)
        return documents