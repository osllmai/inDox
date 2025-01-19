import os
from dotenv import load_dotenv
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


def generate_clustered_prompts(context, embeddings, summary_model):
    """
    Clusters the provided context using the given embeddings and generates clustered prompts.

    Parameters:
    context (list of str): A list of text strings to be clustered.
    embeddings: Embeddings function.

    Returns:
    list: A list of clustered document segments based on the provided context and embeddings.
    """
    try:
        logger.info("Loading environment variables from .env file")
        # Load environment variables from .env file
        load_dotenv()

        # Check if OpenAI API key is set
        # use_openai_summary = os.getenv("OPENAI_API_KEY") is not None
        # logger.info(f"OpenAI API key is {'set' if use_openai_summary else 'not set'}")

        texts = " ".join(context)
        from indoxArcg.data_loader_splitter import ClusteredSplit

        logger.info("Initializing ClusteredSplit")
        loader_splitter = ClusteredSplit(
            file_path=texts,
            embeddings=embeddings,
            chunk_size=50,
            threshold=0.1,
            dim=30,
            summary_model=summary_model,
        )
        loader_splitter.cluster_prompt = True

        logger.info("Retrieving all documents")
        docs = loader_splitter.load_and_chunk()

        logger.info("Clustered prompts generated successfully")
        return docs

    except Exception as e:
        logger.error(f"Error in generate_clustered_prompts: {e}")
        raise
