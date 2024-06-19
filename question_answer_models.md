# Question\_Answer\_Models

Indox supports three different types of question-answer (QA) models. These models are:

1. **OpenAI QA Model**
2. **Mistral QA Model from Hugging Face**
3. **OpenAI QA Model with Chain of Thought from the dspy Framework**

### Initial Setup

For all QA models, the initial setup is the same. Start by importing the necessary Indox module and creating an instance of `IndoxRetrievalAugmentation`:

```python
from indox import IndoxRetrievalAugmentation
Indox = IndoxRetrievalAugmentation()
```

### Using OpenAI QA Model

To use the OpenAI QA model, follow these steps:

1. Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

1. import Indox modules and set the OpenAI QA model:

```python
from indox.qa_models import OpenAiQA

openai_qa = OpenAiQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
```

### Using Mistral QA Model from Hugging Face

To use the Mistral QA model from Hugging Face, follow these steps:

1. Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv('HF_API_KEY')
```

1. Import Indox modules and set the Mistral QA model:

```python
from indox.qa_models import MistralQA

mistral_qa = MistralQA(api_key=HF_API_KEY, model="mistralai/Mistral-7B-Instruct-v0.2")
```

Note: Users can choose other models from Hugging Face as well, but we recommend the free Mistral model, which only requires a Hugging Face access token.

### Using OpenAI QA Model with Chain of Thought from the dspy Framework

To use the OpenAI QA model with Chain of Thought from the dspy framework, follow these steps:

1. Import necessary libraries and load environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
```

1. Import Indox modules and set the OpenAI QA model with Chain of Thought:

```python
from indox.qa_models import DspyCotQA

dspy_qa = DspyCotQA(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
```

#### Future Plans

We are committed to continuously improving Indox and will be adding support for more QA models in the future.

***

Previous: Embedding Models | Next: Evaluation