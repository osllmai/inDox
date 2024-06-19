# Evaluation

### Introduction

This notebook demonstrates how `InDox` uses various evaluation metrics to assess the quality of generated outputs. The available metrics are:

* **BertScore**: Evaluates the similarity between the generated text and reference text using BERT embeddings.
* **Toxicity**: Measures the level of toxicity in the generated text.
* **Similarity**: Assesses the similarity between different pieces of text.
* **Reliability**: Evaluates the factual accuracy and trustworthiness of the content.
* **Fairness**: Analyzes the text for potential biases and ensures equitable representation.
* **Readability**: Assesses how easy the text is to read and understand.

You can select the metrics you want to use and then provide the context and answer text for evaluation.

### Implementation

First, we need to import the `Evaluation` library to build our evaluator function:

```{python}
from Indox.Evaluation import Evaluation
```

### Quick Start

initiate your evaluator with the _default_ config :

```{python}
evaluator = Evaluation()
```

::: {.callout-note} You can choose a list of your disired metrics from `"BertScore"`, `"Toxicity"`, `"Similarity"`, `"Reliability"`, `"Fairness"` , `"Readibility"` :::

::: {.callout-tip} if you want to change models with your custom models you can define a `config` in `dict` and customize your `InDox` evaluator

```python
cfg = {"bert_toxic_tokenizer": "unitary/toxic-bert",
       "bert_toxic_model": "unitary/toxic-bert",
       "semantic_similarity": "sentence-transformers/bert-base-nli-mean-tokens",
       "bert_score_model": "bert-base-uncased",
       "reliability": 'vectara/hallucination_evaluation_model',
       "fairness": "wu981526092/Sentence-Level-Stereotype-Detector",

       }
```

:::

::: {.callout-caution collapse="true"} it's important to say `InDox` uses open source models and all models are scratched from **HuggingFace** :::

After initiate your evalutor check it with sample question answer response , each inputs should be have `question`, `answer`, `context` in a `dict` format , for example :

```{python}
sample = {
    'question' : "What is your Question?",
    'answer' : "It's your responsed answer from InDox",
    'context' : "this can be a list of context or a single context."
}
```

then use evaluator to calculate metrics, evaluator is callable function which in `__call__` function calculates metrics. InDox evaluator returns metrics as `dict` file :

```{python}
evaluator(sample)
```

```
{
"Precision": 0.46,
"Recall": 0.75,
"F1-score": 0.68,
"Toxicity": 0.01,
"BLEU": 0.01,
"Jaccard Similarity": 0.55,
"Cosine Similarity": 0.68,
"Semantic": 0.79,
"hallucination_score" : 0.11, 
"Perplexity": 11,
"ARI": 0.5,
"Flesch-Kincaid Grade Level": 0.91
}
```

### Evaluation Updater

The `inDox` evaluator provides an `update` function to store evaluation metrics in a data frame. To stack your results, use the `update` function as shown below:

```{python}
results = evaluator.update(inputs)
```

The output will be stored in the results data frame, which might look like this:

```{python}
results
```

::: {.callout-note} if you want to clean results use `reset` method to clean it. :::

#### Metric Explanations

* **Precision**: Measures the accuracy of positive predictions. It's the ratio of true positive results to the total predicted positives.
* **Recall**: Measures the ability to find all relevant instances. It's the ratio of true positive results to all actual positives.
* **F1-score**: The harmonic mean of precision and recall. It balances the two metrics, providing a single measure of a model's performance.
* **Toxicity**: Quantifies the level of harmful or abusive content in the text.
* **BLEU (Bilingual Evaluation Understudy)**: A metric for evaluating the quality of machine-translated text against one or more reference translations.
* **Jaccard Similarity**: Measures the similarity between two sets by dividing the size of the intersection by the size of the union of the sets.
* **Cosine Similarity**: Measures the cosine of the angle between two vectors in a multi-dimensional space. It’s used to determine the similarity between two text samples.
* **Semantic Similarity**: Evaluates the degree to which two pieces of text carry the same meaning.
* **Hallucination Score**: Indicates the extent to which a model generates information that is not present in the source data.
* **Perplexity**: Measures how well a probability model predicts a sample. Lower perplexity indicates a better predictive model.
* **ARI** (Automated Readability Index): An index that assesses the readability of a text based on sentence length and word complexity.
* **Flesch-Kincaid Grade Level**: An index that indicates the US school grade level required to understand the text.

#### Custom Evaluator Example Usage

Below code provide an example usage of `inDox` evaluation with custom config :

```{python}
from Indox.Evaluation import Evaluation

cfg = {
       "bert_score_model": "bert-base-uncased",
       }

dimanstions = ["BertScore"]

evaluator = Evaluation(cfg = cfg , dimanstions = dimanstions)


inputs1 = {
    'question': 'What is the capital of France?',
    'answer': 'The capital of France is Paris.',
    'context': 'France is a country in Western Europe. Paris is its capital.'
}

inputs2 = {
    'question': 'What is the largest planet in our solar system?',
    'answer': 'The largest planet in our solar system is Jupiter.',
    'context': 'Our solar system consists of eight planets. Jupiter is the largest among them.'
}
results = evaluator.update(inputs1)
results = evaluator.update(inputs2)
```

Display the results:

```{python}
print(results)
```

### **Expected Output**

The results data frame will contain evaluation metrics for each input:

```{python}
>>>
```

***

Previous: Question Answer Models | Next: Cluster Split Example
