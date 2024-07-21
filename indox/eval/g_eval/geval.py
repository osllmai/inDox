from openai import OpenAI
from template import GEvalTemplate
import json
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def g_eval(texts, parameters, model='gpt-3.5-turbo-0125'):
    """
    Evaluate the quality of NLG outputs using GPT-4 with the GEvalTemplate.

    Parameters:
    texts (list of str): List of generated texts to evaluate.
    parameters (str): The parameter to be evaluated (e.g., 'summary', 'dialogue').
    criteria (str): The criteria for evaluation.
    model (str): The model to use, 'gpt-4'.

    Returns:
    list of dict: Evaluation scores and reasons for each text.
    """
    criteria = """
    1. Retrieval Quality: The retrieved documents or snippets should be relevant and accurate.
    2. Integration: The retrieved information should be well integrated into the generated response.
    3. Coherence: The text should be logically structured and easy to follow.
    4. Relevance: The text should be relevant to the main topic and cover all key points.
    5. Accuracy: The text should be factually accurate and consistent with the source material.
    6. Fluency: The text should be easy to read and free from grammatical errors.
    7. Comprehensiveness: The text should cover all key points and provide a thorough response.
    8. Contextuality: The response should fit well within the context of the query.
    """
    eval_steps_prompt = GEvalTemplate.generate_evaluation_steps(parameters, criteria)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": eval_steps_prompt}]
    )
    eval_steps = json.loads(response.choices[0].message.content)["steps"]
    results = []
    for text in texts:
        eval_text = f"""
            Input: {text['Input']}
            Actual Output: {text['Actual Output']}
            Expected Output: {text['Expected Output']}
            Context: {text['Context']}
            Retrieval Context: {text['Retrieval Context']}
            """
        eval_results_prompt = GEvalTemplate.generate_evaluation_results(eval_steps, eval_text, parameters)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": eval_results_prompt}]
        )
        print(response.choices[0].message.content)
        result = json.loads(response.choices[0].message.content)
        results.append(result)

    return results