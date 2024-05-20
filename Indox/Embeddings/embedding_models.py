import os


def OpenAiEmbedding(model, openai_api_key):
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=openai_api_key)
    return embeddings


def HuggingFaceEmbedding(model_name="multi-qa-mpnet-base-cos-v1"):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
