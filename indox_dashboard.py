from urllib.parse import urlparse
import pandas as pd
import psycopg2
import streamlit as st
import os
from indox import IndoxRetrievalAugmentation
from dotenv import load_dotenv
import time
from datetime import datetime
def get_database_connection(con_string):
    result = urlparse(con_string)

    # Extract connection parameters
    dbname = result.path[1:]  # remove leading '/'
    user = result.username
    password = result.password
    host = result.hostname
    port = result.port
    # Create a database connection
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    return conn


def fetch_data(query, con_string):
    conn = get_database_connection(con_string)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# def read_log_file(file_path):
#     with open(file_path, 'r') as file:
#         return file.read()

def display_logs_from_file(log_area, log_file_path):
    while True:
        with open(log_file_path, 'r') as file:
            log_data = file.read()
        log_area.markdown(f"```\n{log_data}\n```")  # Display logs using markdown
        time.sleep(1)  # Simulate processing time and update interval

# Set up the page configuration
st.set_page_config(
    page_title="Indox Application Dashboard",
    layout="centered",
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Main content
st.title("Indox Application Dashboard")

if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'use_existing_database' not in st.session_state:
    st.session_state.use_existing_database = False
if 'db_type' not in st.session_state:
    st.session_state.db_type = None
if 'conn_string' not in st.session_state:
    st.session_state.conn_string = None

if st.session_state.step == 1:
    # Selection section
    st.write("## Choose Data Source")
    data_source = st.radio("Select the data source:", ("Upload a file", "Use existing database"))

    if data_source == "Upload a file":
        st.session_state.use_existing_database = False
        # File upload section
        st.write("### Upload Your Data")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "csv", "docx", "xlsx", "html"])

        if uploaded_file is not None:
            try:
                # Ensure the tempDir directory exists
                if not os.path.exists("tempDir"):
                    os.makedirs("tempDir")

                # Save the uploaded file temporarily
                temp_file_path = os.path.join("tempDir", uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process the uploaded file
                file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type,
                                "filesize": uploaded_file.size}
                st.write(file_details)

                next_button_disabled = False
            except Exception as e:
                st.error(f"An error occurred while uploading the file: {e}")
                next_button_disabled = True
        else:
            next_button_disabled = True

    else:
        st.session_state.use_existing_database = True
        # Existing database section
        st.write("### Choose Database Type")
        db_type = st.selectbox("Database Type", ["Chroma", "Faiss", "Postgres(pgvector)"])
        st.session_state.db_type = db_type

        st.write("### Enter Database Address")
        if db_type == "Postgres(pgvector)":

            st.session_state.host = st.text_input("Host")
            st.session_state.port = st.text_input("Port")
            st.session_state.dbname = st.text_input("Database Name")
            st.session_state.user = st.text_input("User")
            st.session_state.password = st.text_input("Password")
            connection_string = f"postgresql+psycopg2://{st.session_state.user}:{st.session_state.password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.dbname}"
            st.session_state.connection_string = connection_string
        else:
            db_address = st.text_input("Database Address")
            st.session_state.connection_string = db_address

        next_button_disabled = False

    if st.button("Next", disabled=next_button_disabled):
        if not st.session_state.use_existing_database:
            st.session_state.uploaded_file_path = temp_file_path
            st.session_state.step = 2
        else:
            st.session_state.step = 4
        st.rerun()

# Step 2: Choose Target Vector Database for Uploaded File
if st.session_state.step == 2:
    st.write("## Choose Target Vector Database")
    db_type = st.selectbox("Vector Database Type", ["Chroma", "Faiss", "Postgres(pgvector)"])
    st.session_state.db_type = db_type

    if st.button("Next"):
        try:
            if db_type == "Postgres(pgvector)":
                # Indox.config["vector_store"] = "pgvector"
                # Indox.update_config()
                st.session_state.step = 3
            else:
                # Indox.config["vector_store"] = db_type.lower()
                # Indox.update_config()
                st.session_state.step = 4
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred while configuring the vector database: {e}")

    if st.button("Back"):
        st.session_state.step = 1
        st.rerun()

# Step 3: Enter Connection String for Postgres
if st.session_state.step == 3:
    st.write("## Enter Postgres Connection String")
    # connection_string = st.text_input("Connection String",
    #                             placeholder="postgresql+psycopg2://postgres:xxx@localhost:port/db_name")
    st.session_state.host = st.text_input("Host")
    st.session_state.port = st.text_input("Port")
    st.session_state.dbname = st.text_input("Database Name")
    st.session_state.user = st.text_input("User")
    st.session_state.password = st.text_input("Password")
    connection_string = f"postgresql+psycopg2://{st.session_state.user}:{st.session_state.password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.dbname}"
    if st.button("Next"):
        if connection_string:
            st.session_state.connection_string = connection_string

            st.session_state.step = 4
            st.rerun()

    if st.button("Back"):
        st.session_state.step = 2
        st.rerun()

if st.session_state.step == 4:
    if not st.session_state.use_existing_database:
        st.write(f"Uploaded file: {st.session_state.uploaded_file_path}")
    st.write(f"Target Vector Database: {st.session_state.db_type}")

    load_dotenv()

    # Retrieve API keys from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
    INDOX_API_KEY = os.getenv("INDOX_API_KEY")

    # Embedding model and QA model selection
    st.write("## Choose Your Embedding Model")

    # Embedding model selection
    openai_embedding_checked = st.checkbox("OpenAI Embedding Model")
    if openai_embedding_checked:
        openai_embedding_model = st.text_input("Enter OpenAI Embedding Model")

    hf_embedding_checked = st.checkbox("Hugging Face Embedding Model")
    if hf_embedding_checked:
        hf_embedding_model = st.text_input("Enter Hugging Face Embedding Model Name")

    indox_embedding_checked = st.checkbox("Indox API Embedding Model")
    if indox_embedding_checked:
        indox_embedding_model_name = st.text_input("Enter Indox API Embedding Model Name")

    if openai_embedding_checked + hf_embedding_checked + indox_embedding_checked > 1:
        st.error("Please select only one embedding model.")
        st.stop()

    st.write("## Choose Your QA Model")

    # QA model selection
    openai_qa_checked = st.checkbox("OpenAI Model")
    if openai_qa_checked:
        openai_qa_model = st.text_input("Enter OpenAI Model")

    mistral_qa_checked = st.checkbox("Mistral Model")
    if mistral_qa_checked:
        mistral_qa_model = st.text_input("Enter Mistral Model Name")

    indox_qa_checked = st.checkbox("Indox API Model")
    if indox_qa_checked:
        indox_qa_model_name = st.text_input("Enter Indox API Model Name")

    if openai_qa_checked + mistral_qa_checked + indox_qa_checked > 1:
        st.error("Please select only one QA model.")
        st.stop()

    # Check if all required configurations are set
    embedding_model_set = (
            (openai_embedding_checked and openai_embedding_model) or
            (hf_embedding_checked and hf_embedding_model) or
            (indox_embedding_checked and indox_embedding_model_name)
    )

    qa_model_set = (
            (openai_qa_checked and openai_qa_model) or
            (mistral_qa_checked and mistral_qa_model) or
            (indox_qa_checked and indox_qa_model_name)
    )

    next_button_disabled = not (embedding_model_set and qa_model_set)

    if st.button("Next", disabled=next_button_disabled):
        # Save configurations to session state
        if openai_embedding_checked:
            from indox.embeddings import OpenAiEmbedding

            embedding_model = OpenAiEmbedding(openai_api_key=OPENAI_API_KEY, model=openai_embedding_model)
        elif hf_embedding_checked:
            from indox.embeddings import HuggingFaceEmbedding

            embedding_model = HuggingFaceEmbedding(model=hf_embedding_model)
        elif indox_embedding_checked:
            from indox.embeddings import IndoxOpenAIEmbedding

            embedding_model = IndoxOpenAIEmbedding(model=indox_embedding_model_name, api_key=INDOX_API_KEY)

        if openai_qa_checked:
            from indox.qa_models import OpenAiQA

            qa_model = OpenAiQA(api_key=OPENAI_API_KEY, model=openai_qa_model)
        elif mistral_qa_checked:
            from indox.qa_models import MistralQA

            qa_model = MistralQA(api_key=HF_API_KEY, model=mistral_qa_model)
        elif indox_qa_checked:
            from indox.qa_models import IndoxApiOpenAiQa

            qa_model = IndoxApiOpenAiQa(api_key=INDOX_API_KEY)

        st.session_state.embedding_model = embedding_model
        st.session_state.qa_model = qa_model
        st.session_state.step = 5
        st.rerun()

    if st.button("Back"):
        st.session_state.step = 1
        st.rerun()

if st.session_state.step == 5:
    temp_file_path = st.session_state.uploaded_file_path
    embedding_model = st.session_state.embedding_model
    qa_model = st.session_state.qa_model

    if not st.session_state.use_existing_database:
        # Additional configurations

        # Splitter section with descriptions for each method
        st.write("## Choose Your Loader Splitter")
        splitter_method = st.radio(
            "Select a method to split your data:",
            ('Option 1: Load and cluster (TXT and PDF Only)',
             'Option 2: Load and Splitting with Unstructured Library (accept different types)')
        )

        collection_name = st.text_input("Enter Collection Name")

        # Check if required fields are filled
        if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)':
            re_chunk = st.checkbox("Re-chunk", value=False)
            remove_sword = st.checkbox("Remove Stop Words", value=False)
            chunk_size = st.number_input("Chunk Size", min_value=1, max_value=10000, value=100)
            overlap = st.number_input("Overlap", min_value=0, max_value=chunk_size - 1, value=0)
            threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.1)
            dim = st.number_input("Dimension", min_value=1, max_value=100, value=10)
            use_openai_summary = st.checkbox("Use OpenAI Summary", value=False)
            max_len_summary = st.number_input("Max Length of Summary", min_value=1, max_value=500, value=100)
            min_len_summary = st.number_input("Min Length of Summary", min_value=1, max_value=100, value=30)

            config_set = all([collection_name, chunk_size, threshold, dim, max_len_summary, min_len_summary])
        elif splitter_method == 'Option 2: Load and Splitting with Unstructured Library (accept different types)':
            remove_sword = st.checkbox("Remove Stop Words", value=False)
            max_chunk_size = st.number_input("Max Chunk Size", min_value=1, max_value=10000, value=500)

            config_set = all([collection_name, max_chunk_size])

        confirm_button_disabled = not config_set

        if st.button("Confirm", disabled=confirm_button_disabled):
            # Save splitter configurations to session state
            st.session_state.splitter_method = splitter_method
            st.session_state.collection_name = collection_name
            st.session_state.re_chunk = re_chunk if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.remove_sword = remove_sword
            st.session_state.chunk_size = chunk_size if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.overlap = overlap if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.threshold = threshold if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.dim = dim if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.use_openai_summary = use_openai_summary if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.max_len_summary = max_len_summary if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.min_len_summary = min_len_summary if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
            st.session_state.max_chunk_size = max_chunk_size if splitter_method == 'Option 2: Load and Splitting with Unstructured Library (accept different types)' else None

            if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)':
                from indox.data_loader_splitter import ClusteredSplit

                loader_splitter = ClusteredSplit(
                    temp_file_path,
                    embeddings=embedding_model,
                    re_chunk=st.session_state.re_chunk,
                    remove_sword=st.session_state.remove_sword,
                    chunk_size=st.session_state.chunk_size,
                    overlap=st.session_state.overlap,
                    threshold=st.session_state.threshold,
                    dim=st.session_state.dim,
                    use_openai_summary=st.session_state.use_openai_summary,
                    max_len_summary=st.session_state.max_len_summary,
                    min_len_summary=st.session_state.min_len_summary
                )
                data = loader_splitter.load_and_chunk()
            elif splitter_method == 'Option 2: Load and Splitting with Unstructured Library (accept different types)':
                from indox.data_loader_splitter import UnstructuredLoadAndSplit

                loader_splitter = UnstructuredLoadAndSplit(
                    temp_file_path,
                    remove_sword=st.session_state.remove_sword,
                    max_chunk_size=st.session_state.max_chunk_size
                )
                data = loader_splitter.load_and_chunk()
            # Save the data and move to the next step
            st.session_state.data = data
            st.session_state.step = 6
            st.rerun()
        if st.button("Back"):
            st.session_state.step = 4
            st.rerun()
    else:
        collection_name = st.text_input("Enter Collection Name")

        if st.button("Confirm"):
            st.session_state.collection_name = collection_name
            st.session_state.step = 6
            st.rerun()
Indox = IndoxRetrievalAugmentation()

if st.session_state.step == 6:

    if not st.session_state.use_existing_database:
        # Connect to vector store and store data
        data = st.session_state.data
        embedding_model = st.session_state.embedding_model
        collection_name = st.session_state.collection_name

        if st.session_state.db_type == "Postgres(pgvector)":
            from indox.vector_stores import PGVectorStore

            db = PGVectorStore(host=st.session_state.host,
                               password=st.session_state.password,
                               user=st.session_state.user,
                               port=st.session_state.port,
                               dbname=st.session_state.dbname,
                               embedding=st.session_state.embedding_model,
                               collection_name=st.session_state.collection_name)
        elif st.session_state.db_type == "Chroma":
            from indox.vector_stores import ChromaVectorStore

            db = ChromaVectorStore(collection_name=collection_name, embedding=embedding_model)
        elif st.session_state.db_type == "Faiss":
            from indox.vector_stores import FAISSVectorStore

            db = FAISSVectorStore(embedding=embedding_model)

        Indox.connect_to_vectorstore(vectorstore_database=db)
        database = Indox.store_in_vectorstore(data)

        st.session_state.vector_store_initialized = True
        st.session_state.database = database
    else:
        collection_name = st.session_state.collection_name
        embedding_model = st.session_state.embedding_model
        if st.session_state.db_type == "Postgres(pgvector)":
            from indox.vector_stores import PGVectorStore

            db = PGVectorStore(host=st.session_state.host,
                               password=st.session_state.password,
                               user=st.session_state.user,
                               port=st.session_state.port,
                               dbname=st.session_state.dbname,
                               embedding=st.session_state.embedding_model,
                               collection_name=st.session_state.collection_name)
        elif st.session_state.db_type == "Chroma":
            from indox.vector_stores import ChromaVectorStore

            db = ChromaVectorStore(collection_name=collection_name, embedding=embedding_model)
        elif st.session_state.db_type == "Faiss":
            from indox.vector_stores import FAISSVectorStore

            db = FAISSVectorStore(embedding=embedding_model)
        database = Indox.connect_to_vectorstore(vectorstore_database=db)
        st.session_state.vector_store_initialized = True
        st.session_state.database = database

    # Once the data is processed, move to the next step
    st.session_state.step = 7
    st.rerun()

if st.session_state.step == 7 and st.session_state.vector_store_initialized:
    database = st.session_state.database
    db_type = st.session_state.db_type

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Chat", "History", "Database", "Eval", "Prompt Augmentation", "Logs"])

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with tab1:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with tab1:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                response = Indox.answer_question(db=database, qa_model=st.session_state.qa_model, query=prompt,
                                                 document_relevancy_filter=st.session_state.doc_relevancy_filter,
                                                 generate_clustered_prompts=st.session_state.clustered_prompt)
                response_content = response[0]
                response_context = response[1][0] if response[1] else "No context available"
                st.write(response_content)
                # Save the context to session state for sidebar
                st.session_state.last_context = response_context
                st.session_state.qa_history.append((prompt, response_content))
                st.session_state.messages.append({"role": "assistant", "content": response_content})
    if st.session_state.vector_store_initialized:
        with st.sidebar:
            st.sidebar.title("Sources")
            if st.session_state.get("last_context"):
                st.sidebar.write(st.session_state["last_context"])

    with tab2:
        if 'qa_history' in st.session_state and st.session_state.qa_history:
            for q, a in st.session_state.qa_history:
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {a}")
        else:
            st.write("No chat history available.")

    with tab3:
        if db_type.lower() == "postgres(pgvector)":
            query = "SELECT * FROM public.langchain_pg_embedding"
            data = fetch_data(query, database.conn_string)
            st.dataframe(data)
            # st.bar_chart(data['document'])
        elif db_type.lower() == "chroma":
            docs = database.get_all_documents()
            df = pd.DataFrame(docs)
            st.dataframe(df)
        else:
            st.write("Doesn't support Faiss yet!")
    with tab4:
        st.write("Evaluation tab coming soon.")
    with tab5:
        doc_relevancy_filter = st.checkbox("Apply Document Relevancy Filter")
        st.session_state.doc_relevancy_filter = doc_relevancy_filter
        clustered_prompt = st.checkbox("Generate Clustered Prompt")
        st.session_state.clustered_prompt = clustered_prompt
    with tab6:
        if 'show_logs' not in st.session_state:
            st.session_state.show_logs = False

            # Button to toggle log visibility
        if st.button('Show logs' if not st.session_state.show_logs else 'Hide logs'):
            st.session_state.show_logs = not st.session_state.show_logs

            # Display logs if the state is set to show
        if st.session_state.show_logs:
            log_area = st.empty()  # Placeholder for text_area update

            while st.session_state.show_logs:
                with open("indox.log", 'r') as file:
                    log_data = file.read()
                log_area.markdown(f"```\n{log_data}\n```")  # Display logs using markdown
                time.sleep(1)  # Simulate processing time and update interval
                st.experimental_rerun()
        else:
            st.write("Logs are hidden. Click 'Show logs' to display them.")

if st.session_state.step > 7:
    st.write("Something went wrong. Please reset and try again.")
