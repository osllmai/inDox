# LLMs


Indox supports three different types of question-answer (QA) models.
These models are:

1.  **OpenAI Model**
2.  **Mistral Model**
3.  **HuggingFace Models**
3.  **GoogleAi Models**


## Initial Setup

For all QA models, the initial setup is the same. Start by importing the
necessary Indox module and creating an instance of
`IndoxRetrievalAugmentation`:

``` python
from indox import IndoxRetrievalAugmentation
Indox = IndoxRetrievalAugmentation()
```

## Using OpenAI Model

To use the OpenAI QA model, follow these steps:
First install the OpenAI Python package:
```python
pip install openai
```

1.  Import necessary libraries and load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

1.  Import Indox modules and set the OpenAI model:

``` python
from indox.llms import OpenAi

openai_qa = OpenAi(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
retriever = indox.QuestionAnswer(vector_database=db,llm=openai_qa,top_k=5)
retriever.invoke(query)
```

## Using Mistral Model

First install the Hugging Face Python package:
```python
pip install mistralai
```

To use the Mistral model, follow these steps:

1.  Import necessary libraries and load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
```

1.  Import Indox modules and set the Mistral model:

``` python
from indox.llms import Mistral

mistral_qa = Mistral(api_key=MISTRAL_API_KEY, model="mistralai/Mistral-7B-Instruct-v0.2")
retriever = indox.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5)
retriever.invoke(query)
```

## Using GoogleAi Model

First install the Hugging Face Python package:
```python
pip install google-generativeai
```

To use the GoogleAi model, follow these steps:

1.  Import necessary libraries and load environment variables:

``` python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
```

1.  Import Indox modules and set the Mistral model:

``` python
from indox.llms import GoogleAi

mistral_qa = GoogleAi(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash-latest")
retriever = indox.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5)
retriever.invoke(query)
```

### Future Plans

We are committed to continuously improving Indox and will be adding
support for more QA models in the future.
