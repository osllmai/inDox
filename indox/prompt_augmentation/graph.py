import logging
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import JsonOutputParser
from typing import List, TypedDict

logging.basicConfig(filename='indox.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

relevancy_prompt = """You are a grader assessing relevance of a retrieved
    document to a user question. If the document contains keywords related to the
    user question, grade it as relevant. It does not need to be a stringent test.
    The goal is to filter out erroneous retrievals.

    Give a binary score ''yes'' or ''no'' score to indicate whether the document is
    relevant to the question.

    Provide the binary score as a JSON with a single key ''score'' and no preamble
    or explanation.

    Here is the retrieved document:

    {document}

    Here is the user question: 

    {question}
    """


class GraphState(TypedDict):
    """
    Represents the state of the graph.
    """
    question: str
    generation: str
    relevance: str
    scores: List[float]
    documents: List[str]


class RAGGraph:

    def __init__(self, model_name='gpt-3.5-turbo-0125'):
        try:
            logging.info("Initializing RAGGraph with model: %s", model_name)
            llm = ChatOpenAI(model=model_name, temperature=0)
            document_relevancy_prompt = PromptTemplate(template=relevancy_prompt,
                                                       input_variables=["question", "document"])
            self.retrieval_grader = document_relevancy_prompt | llm | JsonOutputParser()
            logging.info("RAGGraph initialized successfully")
        except Exception as e:
            logging.error("Error initializing RAGGraph: %s", e)
            raise

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Filtered out irrelevant documents and updated web_search state
        """
        try:
            logging.info("Grading documents for relevance")
            question = state["question"]
            documents = state["documents"]
            scores = state["scores"]

            filtered_docs = []
            filtered_scores = []
            for i in range(len(documents)):
                score = self.retrieval_grader.invoke({"question": question, "document": documents[i]})
                grade = score['score']
                if grade.lower() == "yes":
                    logging.info("Document %d is relevant", i)
                    filtered_docs.append(documents[i])
                    filtered_scores.append(scores[i])
                else:
                    logging.info("Document %d is not relevant", i)
                    continue

            logging.info("Document grading completed")
            return {"documents": filtered_docs, "question": question, 'scores': filtered_scores}
        except Exception as e:
            logging.error("Error grading documents: %s", e)
            raise

    def run(self, inputs):
        try:
            logging.info("Running RAGGraph with inputs: %s", inputs)
            workflow = StateGraph(GraphState)

            workflow.add_node("grade_document", self.grade_documents)
            workflow.set_entry_point('grade_document')
            workflow.add_edge('grade_document', END)

            app = workflow.compile()

            output = app.invoke(inputs)

            logging.info("RAGGraph run completed with output: %s", output)
            return output
        except Exception as e:
            logging.error("Error running RAGGraph: %s", e)
            raise
