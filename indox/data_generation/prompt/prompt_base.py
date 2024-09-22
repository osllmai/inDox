from typing import Any, Callable, Generator

class PromptBase:
    def _register_prompt_inputs(self, prompt_input_type: str = "prompt"):
        self._prompt_input_type = prompt_input_type
        self.register_input(f"{self._prompt_input_type}s", help=f"The {self._prompt_input_type}s to process.")

    def _register_prompt_args(self):
        self.register_arg("llm", help="The LLM to use.")

    def _register_prompt_optional_args(self):
        self.register_arg("post_process", required=False, help="Post-process function for the generations.")
        self.register_arg("lazy", required=False, default=False, help="Run lazily or not.")
        self.register_arg("**kwargs", required=False, help="Additional arguments for the LLM.")

    def _register_prompt_outputs(self):
        if self._prompt_input_type == "input":
            self.register_output("inputs", help="The inputs processed.")
        self.register_output("prompts", help="The prompts processed.")
        self.register_output("generations", help="The generations by the LLM.")

    def _run_prompts(self, args: dict[str, Any], prompts: Callable[[], Generator[str, None, None]] = None):
        llm = args.pop("llm")
        post_process = args.pop("post_process", None)
        prompts = prompts() if callable(prompts) else prompts

        generations = llm.run(prompts=prompts, **args)
        if post_process:
            generations = map(post_process, generations)

        return generations
