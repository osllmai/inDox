from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer


def semantic_text_splitter(text, max_tokens):
    # Maximum number of tokens in a chunk
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    chunks = splitter.chunks(text)
    return chunks
