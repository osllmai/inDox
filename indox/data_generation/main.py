import os
from dotenv import load_dotenv
from synthetic.synthetic_data import SyntheticData
from llms.openai import OpenAI
from prompt.data_from_prompt import DataFromPrompt

# Load environment variables
load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")

with SyntheticData("./output"):
    # Load GPT-4
    gpt_4 = OpenAI(api_key=INDOX_API_KEY, model_name="gpt-4o-mini")

    # Generate synthetic arXiv-style research paper abstracts with GPT-4
    arxiv_dataset = DataFromPrompt(
        "Generate Research Paper Abstracts",
        args={
            "llm": gpt_4,
            "n": 7,
            "temperature": 1.2,
            "instruction": (
                "Generate an arXiv abstract of an NLP research paper."
                " Return just the abstract, no titles."
            ),
        },
        outputs={"generations": "abstracts"},
    )

    # Print generated abstracts
    for abstract in arxiv_dataset.run():
        print(abstract)
