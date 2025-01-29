from loguru import logger
import sys
import pickle
import os

# Set up logging
logger.remove()
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)
logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class KVCache:
    """
    Key-Value Cache for storing and managing preloaded knowledge.
    """

    def __init__(self, cache_dir="kv_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def save_cache(self, key, kv_data):
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(kv_data, f)
        logger.info(f"KV cache saved: {filepath}")

    def load_cache(self, key):
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                kv_data = pickle.load(f)
            logger.info(f"KV cache loaded: {filepath}")
            return kv_data
        else:
            logger.warning(f"KV cache not found for key: {key}")
            return None

    def reset_cache(self):
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            os.remove(filepath)
        logger.info("KV cache reset successfully")

    # def update_cache(self, key, new_data):
    #     """
    #     Update an existing cache with new data.

    #     Args:
    #         key (str): The cache key to update.
    #         new_data (Any): The new data to append or merge into the existing cache.
    #     """
    #     existing_data = self.load_cache(key)
    #     if existing_data is None:
    #         logger.info(f"Creating a new cache for key: {key}")
    #         existing_data = []

    #     # Combine the existing data with new data
    #     if isinstance(existing_data, list):
    #         if isinstance(new_data, list):
    #             existing_data.extend(new_data)
    #         else:
    #             existing_data.append(new_data)
    #     elif isinstance(existing_data, dict):
    #         if isinstance(new_data, dict):
    #             existing_data.update(new_data)
    #         else:
    #             logger.warning("Cannot merge non-dict data into dict cache")
    #     else:
    #         logger.warning("Unsupported cache data type for updating")

    #     # Save the updated cache
    #     self.save_cache(key, existing_data)
    #     logger.info(f"Cache updated successfully for key: {key}")
