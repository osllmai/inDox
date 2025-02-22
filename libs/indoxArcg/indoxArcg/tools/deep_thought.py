from typing import Dict, List, Optional, Callable, Any
import html
import time


class DeepThought:
    """
    A DeepThought pipeline for structured problem-solving using a language model.

    This pipeline implements a step-by-step thinking process that breaks down complex problems
    into discrete analytical stages. It manages the flow of information between steps and
    produces a final synthesized response.

    Attributes:
        llm: Language model instance that must implement a 'chat' method
        thinking_steps (List[Dict]): Configuration for each thinking step, containing:
            - name: Step identifier
            - system_prompt: Instructions for the LLM's role
            - user_prompt_template: Template for formatting user input
            - temperature: Optional sampling temperature
            - stream: Optional boolean for streaming responses
            - max_tokens: Optional maximum response length
        final_answer_prompt (str): Template for generating the final response

    Example:
        >>> llm = SomeLLMImplementation()
        >>> # Using default thinking steps
        >>> pipeline = DeepThought(llm)
        >>> # Or with custom thinking steps
        >>> custom_steps = [
        ...     {
        ...         "name": "Data Analysis",
        ...         "system_prompt": "You are analyzing data patterns.",
        ...         "user_prompt_template": "Data: {input}\nWhat patterns do you see?",
        ...         "temperature": 0.5
        ...     }
        ... ]
        >>> pipeline = DeepThought(llm, thinking_steps=custom_steps)
    """

    DEFAULT_THINKING_STEPS = [
        {
            "name": "Problem Comprehension",
            "system_prompt": "You are breaking down the problem to understand its core components.",
            "user_prompt_template": "Original question: {input}\n\nFirst, let's ensure I fully grasp the problem. What are the key elements here?",
            "temperature": 0.7,
            "stream": True,
            "max_tokens": 400,
        },
        {
            "name": "Critical Analysis",
            "system_prompt": "You are identifying relationships and patterns in the problem components.",
            "user_prompt_template": "Problem breakdown:\n{previous_response}\n\nNow, let's analyze how these elements interact. What patterns emerge?",
            "temperature": 0.5,
            "stream": True,
        },
        {
            "name": "Solution Formulation",
            "system_prompt": "You are synthesizing insights to develop potential solutions.",
            "user_prompt_template": "Analysis results:\n{previous_response}\n\nBased on this, what solutions seem most promising? Let's evaluate options.",
            "temperature": 0.3,
            "stream": True,
        },
    ]

    def __init__(self, llm, thinking_steps: Optional[List[Dict]] = None):
        """
        Initialize the DeepThought pipeline with a language model and optional custom thinking steps.

        Args:
            llm: Language model instance that implements a 'chat' method
            thinking_steps: Optional list of custom thinking steps. If None, uses default steps.
                Each step should be a dictionary with at least a 'user_prompt_template' key.

        Raises:
            ValueError: If LLM doesn't have required methods or thinking steps are misconfigured
        """
        self.llm = llm
        self.thinking_steps = (
            thinking_steps
            if thinking_steps is not None
            else self.DEFAULT_THINKING_STEPS
        )
        self.final_answer_prompt = """Synthesis of Thought Process:

{thinking_process}

Consolidating all considerations, here is the structured response to:

Original Question: {input}

Final Answer:"""
        self._validate_structure()

    def _validate_structure(self):
        """
        Validate pipeline configuration during initialization.

        Checks if:
        - LLM has required chat method
        - All thinking steps have required configuration keys

        Raises:
            ValueError: If validation fails
        """
        if not hasattr(self.llm, "chat"):
            raise ValueError("LLM must have a chat method")

        required_keys = {"user_prompt_template"}
        for i, step in enumerate(self.thinking_steps):
            if not required_keys.issubset(step.keys()):
                missing = required_keys - step.keys()
                raise ValueError(f"Step {i+1} missing required keys: {missing}")

    def _sanitize_input(self, text: str) -> str:
        """
        Perform basic sanitization on input text to prevent injection attacks.

        Args:
            text (str): Raw input text

        Returns:
            str: Sanitized text with HTML entities escaped and whitespace stripped
        """
        return html.escape(text.strip())

    def _print_step_header(self, step_number: int, step_name: str):
        """
        Print formatted header for each pipeline step.

        Args:
            step_number (int): Current step number
            step_name (str): Name of the current step
        """
        header = f"🚀 STEP {step_number}: {step_name}"
        print(header)

    def _print_streaming_response(self, response, step_prefix: str):
        """
        Handle streaming response output with visual feedback.

        Args:
            response: Iterator of response chunks
            step_prefix (str): Prefix to display before the response

        Returns:
            str: Complete concatenated response
        """
        full_response = []
        start_time = time.time()

        print(f"{step_prefix} ", end="", flush=True)
        for chunk in response:
            print(chunk, end="", flush=True)
            full_response.append(chunk)
            if time.time() - start_time > 2 and not chunk.strip():
                print("...", end="", flush=True)
                start_time = time.time()

        print("\n")
        return "".join(full_response)

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Execute the complete pipeline on user input.

        Processes the input through each thinking step and generates a final synthesized response.

        Args:
            user_input (str): The problem or question to analyze

        Returns:
            Dict[str, Any]: Dictionary containing:
                - thinking_steps: List of intermediate step outputs
                - final_answer: Synthesized final response

        Raises:
            Exception: Propagates any errors that occur during execution
        """
        user_input = self._sanitize_input(user_input)
        context = {"input": user_input, "previous_response": None}
        thinking_process = []

        try:
            # Process thinking steps
            for step_idx, step in enumerate(self.thinking_steps):
                step_number = step_idx + 1
                self._print_step_header(step_number, step["name"])

                user_prompt = step["user_prompt_template"].format(**context)
                system_prompt = step.get(
                    "system_prompt",
                    "You are systematically working through this problem.",
                )

                llm_params = {
                    "prompt": user_prompt,
                    "system_prompt": system_prompt,
                    **{
                        k: v
                        for k, v in step.items()
                        if k in {"max_tokens", "temperature", "stream"}
                    },
                }

                response = self.llm.chat(**llm_params)
                full_response = ""

                if step.get("stream"):
                    full_response = self._print_streaming_response(
                        response, step_prefix=f"[Step {step_number} Thinking]"
                    )
                else:
                    full_response = response
                    print(f"[Step {step_number} Complete]:\n{full_response}\n")

                thinking_process.append(
                    {
                        "step": step_number,
                        "name": step["name"],
                        "content": full_response,
                    }
                )
                context["previous_response"] = full_response

                if step_idx < len(self.thinking_steps) - 1:
                    print("\n" + "⏳ Transitioning to next step..." + "\n")

            # Generate final answer
            self._print_step_header("FINAL", "Answer Synthesis")
            compiled_thoughts = "\n\n".join(
                f"STEP {t['step']} ({t['name']}):\n{t['content']}"
                for t in thinking_process
            )

            final_response = self.llm.chat(
                prompt=self.final_answer_prompt.format(
                    input=user_input, thinking_process=compiled_thoughts
                ),
                system_prompt="Integrate all analytical steps into a cohesive, professional response.",
                temperature=0.2,
            )

            print(f"\n📝 Comprehensive Answer:\n{'~' * 40}")
            print(final_response)
            print(f"{'~' * 40}\n")

            return {"thinking_steps": thinking_process, "final_answer": final_response}

        except Exception as e:
            raise
