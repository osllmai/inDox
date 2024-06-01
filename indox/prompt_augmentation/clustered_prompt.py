import os
import logging
from dotenv import load_dotenv

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


def generate_clustered_prompts(context, embeddings):
    """
    Clusters the provided context using the given embeddings and generates clustered prompts.

    Parameters:
    context (list of str): A list of text strings to be clustered.
    embeddings: Embeddings function.

    Returns:
    list: A list of clustered document segments based on the provided context and embeddings.
    """
    try:
        logging.info("Loading environment variables from .env file")
        # Load environment variables from .env file
        load_dotenv()

        # Check if OpenAI API key is set
        use_openai_summary = os.getenv("OPENAI_API_KEY") is not None
        logging.info(f"OpenAI API key is {'set' if use_openai_summary else 'not set'}")

        texts = ' '.join(context)
        from indox.data_loader_splitter import ClusteredSplit

        logging.info("Initializing ClusteredSplit")
        loader_splitter = ClusteredSplit(file_path=texts, embeddings=embeddings, chunk_size=50, threshold=0.1, dim=30,
                                         use_openai_summary=use_openai_summary)
        loader_splitter.cluster_prompt = True

        logging.info("Retrieving all documents")
        docs = loader_splitter.load_and_chunk()

        logging.info("Clustered prompts generated successfully")
        return docs

    except Exception as e:
        logging.error(f"Error in generate_clustered_prompts: {e}")
        raise
