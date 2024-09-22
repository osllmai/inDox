from typing import List, Dict
import pandas as pd
import json


class Step:
    """Base class for steps in the synthetic data generation pipeline."""

    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def register_input(self, name: str, help: str):
        """Register an input for the step."""
        self.inputs[name] = help

    def register_output(self, name: str, help: str):
        """Register an output for the step."""
        self.outputs[name] = help

    def run(self):
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")


class FewShotPrompt(Step):
    """Generates outputs from a given LLM based on few-shot examples and a user prompt."""

    def __init__(self, prompt_name: str, args: dict, outputs: dict, examples: List[Dict[str, str]]):
        super().__init__()
        self.prompt_name = prompt_name
        self.args = args
        self.outputs = outputs
        self.examples = examples
        self.llm = args["llm"]
        self.n = args.get("n", 1)

    def prepare_prompt(self) -> str:
        """Prepares the full prompt including the examples."""
        few_shot_examples = ""
        for example in self.examples:
            input_example = example.get("input", "")
            output_example = example.get("output", "")
            few_shot_examples += f"Input: {input_example}\nOutput: {output_example}\n\n"

        full_prompt = f"{few_shot_examples}Input: {self.args['instruction']}\nOutput:"
        return full_prompt

    def run(self) -> pd.DataFrame:
        """Generates data based on the few-shot prompt and returns a DataFrame."""
        full_prompt = self.prepare_prompt()

        generations = self.llm.run(prompts=[full_prompt] * self.n, max_new_tokens=150)

        results = []
        for generation in generations:
            results.append(json.loads(generation))
            print(f"Generated Data in dictionary format: {results}")

        df = pd.DataFrame(results[0])

        return df


__all__ = ["FewShotPrompt"]
