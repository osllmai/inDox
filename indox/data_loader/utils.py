from indox.core import Document


def load_and_process_input(loader, splitter, remove_stopwords=False):
    try:
        inputs = loader()

        if isinstance(inputs, str):
            # If inputs is a string, remove stopwords if requested
            if remove_stopwords:
                from indox.data_loader_splitter.utils.clean import remove_stopwords
                inputs = remove_stopwords(inputs)
            # Split the text
            chunks = splitter.split_text(inputs)

        elif isinstance(inputs, list) and all(isinstance(doc, Document) for doc in inputs):
            # If inputs is a list of Document objects
            text = ""
            for doc in inputs:
                text += doc.page_content
            # Remove stopwords if requested
            if remove_stopwords:
                from indox.data_loader_splitter.utils.clean import remove_stopwords
                text = remove_stopwords(text)
            # Split the concatenated text
            chunks = splitter.split_text(text)

        else:
            raise ValueError("Unsupported input type. Expected string or list of Document objects.")

        return chunks

    except Exception as e:
        raise RuntimeError(f"Error processing input: {e}")
