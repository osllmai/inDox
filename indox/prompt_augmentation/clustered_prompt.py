import os
from dotenv import load_dotenv


def generate_clustered_prompts(context, embeddings):
    """
    Clusters the provided context using the given embeddings and generates clustered prompts.

    Parameters:
    context (list of str): A list of text strings to be clustered.
    embeddings: Embeddings function.

    Returns:
    list: A list of clustered document segments based on the provided context and embeddings.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Check if OpenAI API key is set
    use_openai_summary = os.getenv("OPENAI_API_KEY") is not None

    texts = ' '.join(context)
    from indox.data_loader_splitter import ClusteredSplit

    loader_splitter = ClusteredSplit(file_path=texts, embeddings=embeddings, chunk_size=50, threshold=0.1, dim=30,
                                     use_openai_summary=use_openai_summary)
    loader_splitter.cluster_prompt = True
    docs = loader_splitter.get_all_docs()
    return docs
