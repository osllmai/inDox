import pandas as pd
from typing import List, Dict, Any, Tuple
from indox.core import Document
import json
import uuid


def extract_title(text: str) -> str:
    """
    Extract the title from the text.

    Args:
        text (str): The input text.

    Returns:
        str: Title extracted from the text, which is the first sentence.
    """
    try:
        if '.' in text:
            title = text.split('.')[0] + '.'
        else:
            title = "Not mentioned"
        return title
    except Exception as e:
        print(f"Error extracting title: {e}")
        return "Not mentioned"


def generate_id() -> str:
    """
    Generate a unique identifier using UUID.

    Returns:
        str: A unique identifier in string format.
    """
    return str(uuid.uuid4())


def process_docs(input_list: List[str], embedding_function: Any) -> List[Dict[str, Any]]:
    """
    Process the list of input texts, extract titles, and generate embeddings.

    Args:
        input_list (List[str]): List of input documents in string format.
        embedding_function (Any): Function to generate embeddings.

    Returns:
        List[Dict[str, Any]]: Processed document objects including embeddings, titles, and ids.
    """
    docs = []
    try:
        for text in input_list:
            doc_id = generate_id()
            embedding = embedding_function.embed_query(text)
            
            print(f"Embedding shape for document {doc_id}: {len(embedding)}")

            output_dict = {
                "id": doc_id,
                "fields": {
                    "title": extract_title(text),
                    "body": text,
                    "embedding": embedding,
                    "id": doc_id
                }
            }
            docs.append(output_dict)
    except Exception as e:
        print(f"Error processing documents: {e}")
    return docs

from typing import Any

class VESPA:

    def __init__(self, app_name: str, embedding_function: Any):
        """
        Initialize the Vespa application with schema and embedding functionality.

        Args:
            app_name (str): Name of the Vespa application.
            embedding_function (Any): Embedding function to handle text embeddings.
        """
        try:
            from vespa.package import (
                ApplicationPackage,
                Field,
                Schema,
                Document as VespaDocument,
                HNSW,
                RankProfile,
                FieldSet,
                GlobalPhaseRanking,
                Function,
            )
            from vespa.deployment import VespaDocker
            from vespa.io import VespaResponse

            self.app_name = app_name
            self.embedding_function = embedding_function

            # Get the embedding for the test sentence to determine its dimensions
            test_sentence = "This is a test."
            test_embedding = self.embedding_function.embed_query(test_sentence)
            self.embedding_dim = len(test_embedding)

            print(f"Detected embedding dimension: {self.embedding_dim}")

            # Update schema to use dynamic embedding dimension
            self.package = ApplicationPackage(
                name=self.app_name,
                schema=[
                    Schema(
                        name="doc",
                        document=VespaDocument(
                            fields=[
                                Field(name="id", type="string", indexing=["summary"]),
                                Field(
                                    name="title",
                                    type="string",
                                    indexing=["index", "summary"],
                                    index="enable-bm25",
                                ),
                                Field(
                                    name="body",
                                    type="string",
                                    indexing=["index", "summary"],
                                    index="enable-bm25",
                                    bolding=True,
                                ),
                                Field(
                                    name="embedding",
                                    type=f"tensor<float>(x[{self.embedding_dim}])",
                                    indexing=["index", "attribute"],
                                    ann=HNSW(distance_metric="angular"),
                                ),
                            ]
                        ),
                        fieldsets=[FieldSet(name="default", fields=["title", "body"])],
                        rank_profiles=[
                            RankProfile(
                                name="bm25",
                                functions=[
                                    Function(name="bm25sum", expression="bm25(title) + bm25(body)")
                                ],
                                first_phase="bm25sum",
                            ),
                            RankProfile(
                                name="semantic",
                                inputs=[
                                    ("query(q)", f"tensor<float>(x[{self.embedding_dim}])"),
                                ],
                                first_phase="closeness(field, embedding)",
                            ),
                            RankProfile(
                                name="fusion",
                                inherits="bm25",
                                inputs=[("query(q)", f"tensor<float>(x[{self.embedding_dim}])")],
                                first_phase="closeness(field, embedding)",
                                global_phase=GlobalPhaseRanking(
                                    expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                                    rerank_count=1000,
                                ),
                            ),
                        ],
                    )
                ]
            )
            self.vespa_docker = VespaDocker()
            self.app = None
        except Exception as e:
            print(f"Error initializing VESPA application: {e}")

    def deploy(self):
        """
        Deploy the Vespa application using Docker.

        Raises:
            Exception: If deployment fails.
        """
        try:
            self.app = self.vespa_docker.deploy(application_package=self.package)
        except Exception as e:
            print(f"Error deploying VESPA application: {e}")
            raise

    def add(self, docs: List[Dict[str, Any]]):
        """
        Add documents to Vespa schema.

        Args:
            docs (List[Dict[str, Any]]): List of documents to add to the Vespa schema.
        """
        try:
            from vespa.io import VespaResponse
            docs = process_docs(docs, self.embedding_function)

            def callback(response: VespaResponse, id: str):
                if not response.is_successful():
                    print(f"Error when feeding document {id}: {response.get_json()}")

            self.app.feed_iterable(docs, schema="doc", namespace=self.app_name, callback=callback)
        except Exception as e:
            print(f"Error adding documents to VESPA: {e}")

    def _get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): Input query text.

        Returns:
            List[Dict[str, Any]]: List of relevant documents.

        Raises:
            ValueError: If the query fails.
        """
        try:
            embedded_query = self.embedding_function.embed_query(query)
            response = self.app.query(
                yql="select * from sources * where ({targetHits:1000}nearestNeighbor(embedding,q)) limit 5",
                query=query,
                ranking="semantic",
                body={"input.query(q)": embedded_query},
            )
            if not response.is_successful():
                raise ValueError(
                    f"Query failed with status code {response.status_code}, url={response.url} response={response.get_json()}"
                )
            return self._parse_response(response.get_json())
        except Exception as e:
            print(f"Error retrieving relevant documents: {e}")
            raise

    def _parse_response(self, response: Dict[str, Any]) -> List[Document]:
        """
        Parse the response from Vespa into a list of Document objects.

        Args:
            response (Dict[str, Any]): The JSON response from Vespa.

        Returns:
            List[Document]: Parsed list of document objects.
        """
        try:
            documents: List[Document] = []
            hits = response.get('root', {}).get('children', [])
            for hit in hits:
                fields = hit.get('fields', {})
                documents.append(Document(
                    page_content=fields.get('body', ''),
                    relevance=hit.get('relevance', 0)
                ))
            return documents
        except Exception as e:
            print(f"Error parsing response: {e}")
            return []

    def display_hits_as_df(self, documents: List[Document]) -> pd.DataFrame:
        """
        Display document hits as a pandas DataFrame.

        Args:
            documents (List[Document]): List of Document objects.

        Returns:
            pd.DataFrame: DataFrame containing document bodies and relevance scores.
        """
        try:
            hits_data = []
            for doc in documents:
                hit_data = {
                    "body": doc.page_content,
                    "score": doc.metadata.get("relevance", 0)
                }
                hits_data.append(hit_data)

            df = pd.DataFrame(hits_data)
            return df
        except Exception as e:
            print(f"Error displaying hits as DataFrame: {e}")
            return pd.DataFrame()

    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with score, retrieving the top k documents.

        Args:
            query (str): The query to search for.
            k (int): The number of top documents to return.

        Returns:
            List[Tuple[Document, float]]: A list of Document and score tuples, sorted by relevance.
        """
        try:
            relevant_documents = self._get_relevant_documents(query)

            df = self.display_hits_as_df(relevant_documents)

            results = []
            for _, row in df.iterrows():
                doc = Document(page_content=row["body"], relevance=row["score"])
                results.append((doc, row["score"]))

            return sorted(results, key=lambda x: x[1], reverse=True)[:k]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []


