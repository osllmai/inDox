from .Splitter import get_chunks
from .Embedding import embedding_model
from .utils import (
    reconfig, get_user_input, get_metrics
)
from typing import List, Tuple, Optional, Any, Dict
from .QAModels import GPT3TurboQAModel
from .vectorstore import get_vector_store
from .utils import read_config
import warnings
import tiktoken

warnings.filterwarnings("ignore")


class IndoxRetrievalAugmentation:
    def __init__(
            self,
            qa_model: Optional[Any] = None,
            re_chunk: bool = False
    ):
        """
        Initialize the IndoxRetrievalAugmentation class with documents, embeddings object, an optional QA model, database connection, and maximum token count for text splitting.

        :param qa_model: Optional pre-initialized QA model
        """
        self.embeddings, self.embed_documents = embedding_model()
        self.qa_model = qa_model if qa_model is not None else GPT3TurboQAModel()
        self.input_tokens_all = 0
        self.embedding_tokens = 0
        self.output_tokens_all = 0
        self.db = None
        self.config = read_config()
        self.inputs = {}
        self.re_chunk = re_chunk

    def create_chunks_from_document(self, docs, max_chunk_size: Optional[int] = 512):
        """
        Retrieve all chunks from the documents, using the specified maximum number of tokens if provided.
        """
        do_clustering = True if get_user_input() == "y" else False
        all_chunks = None
        try:
            if do_clustering:
                all_chunks, input_tokens_all, output_tokens_all = \
                    get_chunks(docs=docs,
                               embeddings=self.embeddings,
                               chunk_size=max_chunk_size,
                               do_clustering=do_clustering,
                               re_chunk=self.re_chunk)
                encoding = tiktoken.get_encoding("cl100k_base")
                embedding_tokens = 0
                for chunk in all_chunks:
                    token_count = len(encoding.encode(chunk))
                    embedding_tokens = embedding_tokens + token_count
                self.input_tokens_all = input_tokens_all
                self.embedding_tokens = embedding_tokens
                self.output_tokens_all = output_tokens_all
            elif not do_clustering:
                all_chunks = get_chunks(docs=docs,
                                        embeddings=self.embeddings,
                                        chunk_size=max_chunk_size,
                                        do_clustering=do_clustering,
                                        re_chunk=self.re_chunk)
                encoding = tiktoken.get_encoding("cl100k_base")
                embedding_tokens = 0
                for chunk in all_chunks:
                    token_count = len(encoding.encode(chunk))
                    embedding_tokens = embedding_tokens + token_count
                self.embedding_tokens = embedding_tokens

            return all_chunks
        except Exception as e:
            print(f"Error while getting chunks: {e}")
            return []

    def connect_to_vectorstore(self, collection_name: str):
        """
        Establish a connection to the vector store database using configuration parameters.
        """
        try:
            self.db = get_vector_store(collection_name=collection_name,
                                       embeddings=self.embeddings)
            if self.db is None:
                raise RuntimeError('Failed to connect to the vector store database.')
            print("Connection established successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to the database: {e}")

    def store_in_vectorstore(self, all_chunks: List[str]) -> Any:
        """
        Store text chunks into a PostgreSQL database.
        """

        try:
            if self.db is not None:
                self.db.add_document(all_chunks)
            return self.db
        except Exception as e:
            print(f"Error while storing in PostgreSQL: {e}")
            return None

    def answer_question(self, query: str, top_k: int):
        """
        Answer a query using the QA model based on similar document chunks found in the database.
        """
        try:
            context, scores = self.db.retrieve(query, top_k=top_k)
            answer = self.qa_model.answer_question(context=context, question=query)
            self.inputs = {"answer": answer, "query": query, "context": context}
            return answer, scores, context
        except Exception as e:
            print(f"Error while answering question: {e}")
            return "", []

    def evaluate(self):
        if self.inputs:
            return get_metrics(self.inputs)
        else:
            print("You should make a query first!!!")

    def get_tokens_info(self):
        """
        prints out number of tokens used
        """
        if self.output_tokens_all > 0:
            print(
                f"""
                Overview of All Tokens Used:
                Input tokens sent to GPT-3.5 Turbo (Model ID: 0125) for summarizing: {self.input_tokens_all}
                Output tokens received from GPT-3.5 Turbo (Model ID: 0125): {self.output_tokens_all}
                Tokens used in the embedding section that were sent to the database: {self.embedding_tokens}
                """
            )
        else:
            print(
                f"""
                Overview of All Tokens Used:
                Tokens used in the embedding section that were sent to the database: {self.embedding_tokens}
                           """
            )

    def update_config(self):
        return reconfig(self.config)

    @classmethod
    def from_config(cls, config: dict,
                    qa_model: Optional[Any] = None,
                    re_chunk: bool = False):
        reconfig(config)
        return cls(qa_model, re_chunk)
