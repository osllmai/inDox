import streamlit as st
import os
from Indox import IndoxRetrievalAugmentation
from streamlit_extras.customize_running import center_running
from dotenv import load_dotenv


Indox = IndoxRetrievalAugmentation()

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

# # Hidden reset button to be triggered by the footer button
# if st.button("Reset"):
#     st.session_state.step = 1
#     st.session_state.vector_store_initialized = False
#     st.session_state.chat_history = []
#     st.session_state.qa_history = []
#     st.rerun()

if st.session_state.step == 1:
    # File upload section
    st.write("## Upload Your Data")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "csv", "docx", "xlsx", "html"])

    if uploaded_file is not None:
        # Ensure the tempDir directory exists
        if not os.path.exists("tempDir"):
            os.makedirs("tempDir")

        # Save the uploaded file temporarily
        temp_file_path = os.path.join("tempDir", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the uploaded file
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(file_details)

        next_button_disabled = False
    else:
        next_button_disabled = True

    if st.button("Next", disabled=next_button_disabled):
        st.session_state.uploaded_file_path = temp_file_path
        st.session_state.step = 2
        st.rerun()

if st.session_state.step == 2:
    load_dotenv()
    st.write(
        "Note: API keys should be stored in a `.env` file in this directory.\nOPENAI_API_KEY or HF_API_KEY or INDOX_API_KEY")

    # Retrieve API keys from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
    INDOX_API_KEY = os.getenv("INDOX_API_KEY")
    # Embedding model and QA model selection
    st.write("## Choose Your Embedding Model")

    # Embedding model selection
    openai_embedding_checked = st.checkbox("OpenAI Embedding Model")
    if openai_embedding_checked:
        # openai_embedding_key = st.text_input("Enter OpenAI API Key", type="password", value=OPENAI_API_KEY)
        openai_embedding_model = st.text_input("Enter OpenAI Embedding Model")

    hf_embedding_checked = st.checkbox("Hugging Face Embedding Model")
    if hf_embedding_checked:
        hf_embedding_model = st.text_input("Enter Hugging Face Embedding Model Name")

    indox_embedding_checked = st.checkbox("Indox API Embedding Model")
    if indox_embedding_checked:
        # indox_embedding_api_key = st.text_input("Enter Indox API Key", type="password", value=INDOX_API_KEY)
        indox_embedding_model_name = st.text_input("Enter Indox API Embedding Model Name")

    st.write("## Choose Your QA Model")

    # QA model selection
    openai_qa_checked = st.checkbox("OpenAI Model")
    if openai_qa_checked:
        openai_qa_model = st.text_input("Enter OpenAI Model")
        # openai_qa_key = st.checkbox("Import From .env File", value=False)

    mistral_qa_checked = st.checkbox("Mistral Model")
    if mistral_qa_checked:
        # mistral_qa_key = st.checkbox("Import From .env File", value=False)
        mistral_qa_model = st.text_input("Enter Mistral Model Name")

    indox_qa_checked = st.checkbox("Indox API Model")
    if indox_qa_checked:
        # indox_qa_api_key = st.text_input("Enter Indox API Key", type="password", value=INDOX_API_KEY)
        indox_qa_model_name = st.text_input("Enter Indox API Model Name")

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

    if st.button("Next", disabled=next_button_disabled, on_click=center_running):

        # Save configurations to session state
        if openai_embedding_checked:
            from Indox.Embeddings import OpenAiEmbedding

            embedding_model = OpenAiEmbedding(openai_api_key=OPENAI_API_KEY, model=openai_embedding_model)
        elif hf_embedding_checked:
            from Indox.Embeddings import HuggingFaceEmbedding

            embedding_model = HuggingFaceEmbedding(model=hf_embedding_model)
        elif indox_embedding_checked:
            from Indox.Embeddings import IndoxOpenAIEmbedding

            embedding_model = IndoxOpenAIEmbedding(model=indox_embedding_model_name, api_key=INDOX_API_KEY)

        if openai_qa_checked:
            from Indox.QaModels import OpenAiQA

            qa_model = OpenAiQA(api_key=OPENAI_API_KEY, model=openai_qa_model)
        elif mistral_qa_checked:
            from Indox.QaModels import MistralQA

            qa_model = MistralQA(api_key=HF_API_KEY, model=mistral_qa_model)
        elif indox_qa_checked:
            from Indox.QaModels import IndoxApiOpenAiQa

            qa_model = IndoxApiOpenAiQa(api_key=INDOX_API_KEY)

        st.session_state.embedding_model = embedding_model
        st.session_state.qa_model = qa_model
        st.session_state.step = 3
        st.rerun()

    if st.button("Back"):
        st.session_state.step = 2
        st.rerun()
if st.session_state.step == 3:
    temp_file_path = st.session_state.uploaded_file_path
    embedding_model = st.session_state.embedding_model
    qa_model = st.session_state.qa_model

    # Additional configurations
    doc_relevancy_filter = st.checkbox("Apply Document Relevancy Filter")

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

        config_set = all([collection_name, chunk_size, threshold, dim])
    elif splitter_method == 'Option 2: Load and Splitting with Unstructured Library (accept different types)':
        remove_sword = st.checkbox("Remove Stop Words", value=False)
        max_chunk_size = st.number_input("Max Chunk Size", min_value=1, max_value=10000, value=500)

        config_set = all([collection_name, max_chunk_size])

    confirm_button_disabled = not config_set

    if st.button("Confirm", disabled=confirm_button_disabled):
        # Save splitter configurations to session state
        st.session_state.doc_relevancy_filter = doc_relevancy_filter
        st.session_state.splitter_method = splitter_method
        st.session_state.collection_name = collection_name
        st.session_state.re_chunk = re_chunk if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
        st.session_state.remove_sword = remove_sword
        st.session_state.chunk_size = chunk_size if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
        st.session_state.overlap = overlap if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
        st.session_state.threshold = threshold if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
        st.session_state.dim = dim if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)' else None
        st.session_state.max_chunk_size = max_chunk_size if splitter_method == 'Option 2: Load and Splitting with Unstructured Library (accept different types)' else None
        print(st.session_state.overlap)
        if splitter_method == 'Option 1: Load and cluster (TXT and PDF Only)':
            from Indox.DataLoaderSplitter import ClusteredSplit

            data = ClusteredSplit(
                temp_file_path,
                embeddings=embedding_model,
                re_chunk=st.session_state.re_chunk,
                remove_sword=st.session_state.remove_sword,
                chunk_size=st.session_state.chunk_size,
                overlap=st.session_state.overlap,
                threshold=st.session_state.threshold,
                dim=st.session_state.dim
            )
        elif splitter_method == 'Option 2: Load and Splitting with Unstructured Library (accept different types)':
            from Indox.DataLoaderSplitter import UnstructuredLoadAndSplit

            data = UnstructuredLoadAndSplit(
                temp_file_path,
                remove_sword=st.session_state.remove_sword,
                max_chunk_size=st.session_state.max_chunk_size
            )

        # Save the data and move to the next step
        st.session_state.data = data
        st.session_state.step = 4
        st.rerun()
    if st.button("Back"):
        st.session_state.step = 3
        st.rerun()
if st.session_state.step == 4:
    # Connect to vector store and store data
    data = st.session_state.data
    embedding_model = st.session_state.embedding_model
    collection_name = st.session_state.collection_name

    Indox.connect_to_vectorstore(embeddings=embedding_model, collection_name=collection_name)
    db = Indox.store_in_vectorstore(data)

    st.session_state.vector_store_initialized = True
    st.session_state.db = db
    # Once the data is processed, move to the next step
    st.session_state.step = 5
    st.rerun()

if st.session_state.step == 5 and st.session_state.vector_store_initialized:
    db = st.session_state.db
    tab1, tab2, tab3 = st.tabs(["Chat", "History", "Eval"])

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
                response = Indox.answer_question(db=db, qa_model=st.session_state.qa_model, query=prompt,
                                                 document_relevancy_filter=st.session_state.doc_relevancy_filter)
                response_content = response[0]
                response_context = response[1][0] if response[1] else "No context available"
                st.write(response_content)
                # for message in st.session_state.messages:
                #     with st.chat_message(message["role"]):
                #         st.markdown(message["content"])
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
        st.write("Evaluation tab coming soon.")
if st.session_state.step > 5:
    st.write("Something went wrong. Please reset and try again.")

