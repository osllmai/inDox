from indox.data_connectors.utils import Document
from typing import Any, List, Optional


class TwitterTweetReader:
    """Twitter tweets reader.

    This class reads tweets from a specified user's Twitter handle. To use this
    class, you must have access to the Twitter API. For more information on how to
    obtain access, refer to the following guide:
    'https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api'.

    Args:
        bearer_token (str): Bearer token obtained from the Twitter API.
        num_tweets (Optional[int]): The number of tweets to fetch for each user's Twitter
            handle. The default is 100 tweets.
    """

    is_remote: bool = True
    bearer_token: str
    num_tweets: Optional[int]

    def __init__(
            self,
            bearer_token: str,
            num_tweets: Optional[int] = 100,
    ) -> None:
        """Initialize TwitterTweetReader with the given parameters.

        Args:
            bearer_token (str): Bearer token obtained from the Twitter API.
            num_tweets (Optional[int], optional): The number of tweets to fetch for each
                user's Twitter handle. Defaults to 100.
        """
        self.bearer_token = bearer_token
        self.num_tweets = num_tweets

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name as a string.

        Returns:
            str: The name of the class, "TwitterTweetReader".
        """
        return "TwitterTweetReader"

    def load_data(
            self,
            twitterhandles: List[str],
            num_tweets: Optional[int] = None,
            **load_kwargs: Any
    ) -> List[Document] | Document:
        """Load tweets for specified Twitter handles.

        Args:
            twitterhandles (List[str]): List of Twitter handles to read tweets from.
            num_tweets (Optional[int], optional): The number of tweets to fetch for each handle.
                If not provided, the instance's `num_tweets` will be used.

        Returns:
            List[Document]: A list of Document instances containing the fetched tweets
            and their associated metadata.

        Raises:
            ImportError: If the `tweepy` package is not installed.
            tweepy.TweepError: If there is an error with the Twitter API request.
            ValueError: If a Twitter handle is invalid or the user does not exist.
        """
        try:
            import tweepy
        except ImportError:
            raise ImportError(
                "`tweepy` package not found. Please install it using `pip install tweepy`."
            )

        try:
            client = tweepy.Client(bearer_token=self.bearer_token)
            documents = []
            for username in twitterhandles:
                user = client.get_user(username=username)
                if not user.data:
                    raise ValueError(f"User '{username}' not found or is invalid.")

                tweets = client.get_users_tweets(
                    user.data.id, max_results=num_tweets or self.num_tweets
                )

                if not tweets.data:
                    raise ValueError(f"No tweets found for user '{username}'.")

                response = "\n".join(tweet.text for tweet in tweets.data)
                metadata = {
                    "username": username,
                    "user_id": user.data.id,
                    "num_tweets": len(tweets.data),
                }

                documents.append(Document(source="Twitter", content=response, metadata=metadata))
            if len(documents) == 1:
                return documents[0]
            return documents

        except tweepy.TweepyException as e:
            raise tweepy.TweepyException(f"An error occurred with the Twitter API: {e}")

    def load_content(
            self,
            twitterhandles: List[str],
            num_tweets: Optional[int] = None,
            **load_kwargs: Any
    ) -> List[str] | str:
        """Load tweets for specified Twitter handles.

        Args:
            twitterhandles (List[str]): List of Twitter handles to read tweets from.
            num_tweets (Optional[int], optional): The number of tweets to fetch for each handle.
                If not provided, the instance's `num_tweets` will be used.

        Returns:
            List[str]: A list of strings, each containing concatenated tweets from a user.

        Raises:
            ImportError: If the `tweepy` package is not installed.
            tweepy.TweepError: If there is an error with the Twitter API request.
            ValueError: If a Twitter handle is invalid or the user does not exist.
        """
        try:
            import tweepy
        except ImportError:
            raise ImportError(
                "`tweepy` package not found. Please install it using `pip install tweepy`."
            )

        try:
            client = tweepy.Client(bearer_token=self.bearer_token)
            tweets_contents = []
            for username in twitterhandles:
                user = client.get_user(username=username)
                if not user.data:
                    raise ValueError(f"User '{username}' not found or is invalid.")

                tweets = client.get_users_tweets(
                    user.data.id, max_results=num_tweets or self.num_tweets
                )

                if not tweets.data:
                    raise ValueError(f"No tweets found for user '{username}'.")

                response = "\n".join(tweet.text for tweet in tweets.data)
                tweets_contents.append(response)
            if len(tweets_contents) == 1:
                return tweets_contents[0]
            return tweets_contents

        except tweepy.TweepyException as e:
            raise tweepy.TweepyException(f"An error occurred with the Twitter API: {e}")
