# from ..utils import read_config, construct_postgres_connection_string
# import logging
#
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
# )
#







#
#
# def get_vector_store(embeddings, collection_name: str):
#     """
#     Returns the appropriate vector store based on the configuration.
#
#     Args:
#         embeddings (Embedding): The embeddings to be used.
#         collection_name (str): The name of the collection in the database.
#
#     Returns:
#         VectorStoreBase: The appropriate vector store.
#     """
#     config = read_config()
#     if config['vector_store'].lower() == 'pgvector':
#         conn_string = construct_postgres_connection_string()
#         return PGVectorStore(conn_string=conn_string, collection_name=collection_name,
#                              embedding=embeddings)
#     elif config['vector_store'].lower() == 'chroma':
#         return ChromaVectorStore(collection_name=collection_name, embedding=embeddings)
#     elif config['vector_store'].lower() == 'faiss':
#         db = FAISSVectorStore(embedding=embeddings)
#         return db
