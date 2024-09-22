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



class DataFromPrompt(Step):
    """Generates outputs from a given LLM based on a prompt and arguments."""

    def __init__(self, prompt_name: str, args: dict, outputs: dict, dataframe: pd.DataFrame = None):
        self.prompt_name = prompt_name
        self.args = args
        self.outputs = outputs
        self.llm = args["llm"]
        self.n = args.get("n", 1)
        self.dataframe = dataframe

    def run(self) -> pd.DataFrame:
        """Generates data based on the prompt and returns a DataFrame."""
        if self.dataframe is not None:
            sample_row = self.dataframe.iloc[0].to_dict()
            user_instruction = self.args["instruction"]

            prompt = f"{user_instruction} {self.prompt_name} : {json.dumps(sample_row)}"
            prompts = [prompt] * self.n

            # Get generated responses
            generations = self.llm.run(prompts=prompts, max_new_tokens=150)

            results = []
            for generation in generations:
                results.append(json.loads(generation))
                print(f"Generated Data in dictionary format: {results}")

            df = pd.DataFrame(results[0])

            return df
        else:
            # Default behavior if no dataframe is provided
            prompts = [self.args["instruction"]] * self.n

            # Get generated responses
            generations = self.llm.run(prompts=prompts, max_new_tokens=150)

            results = []
            for generation in generations:
                results.append(json.loads(generation))
                print(f"Generated Data in dictionary format: {results}")


            df = pd.DataFrame(results[0])

            return df

