import abc
import torch
from pydantic import BaseModel, ConfigDict
from typing import List
from loguru import logger
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from indoxArcg.core import BaseLLM


class HuggingFaceLocalModel(BaseLLM):
    """
    A class to load and run inference on a Hugging Face Causal Language Model locally
    in 4-bit precision using bitsandbytes.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        protected_namespaces=(),  # Disable protected namespace checking
    )

    # Fields with renamed attributes to avoid conflicts
    hf_model_id: str = "BioMistral/BioMistral-7B"  # Renamed from model_id
    prompt_template: str = "Context: {context}\nQuestion: {question}\nAnswer:"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"

    def __init__(
        self,
        hf_model_id: str = "BioMistral/BioMistral-7B",
        prompt_template: str = "Context: {context}\nQuestion: {question}\nAnswer:",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        """
        Initialize the local Hugging Face model.

        Args:
            hf_model_id (str): The Hugging Face model ID to load locally.
            prompt_template (str): A default prompt format string.
            bnb_4bit_use_double_quant (bool): 4-bit quantization parameter.
            bnb_4bit_quant_type (str): 4-bit quant type (e.g., "nf4").
            bnb_4bit_compute_dtype (torch.dtype): The compute dtype in 4-bit context.
            device_map (str): Passed to the model loading (e.g. "auto").
        """
        # Initialize pydantic fields
        super().__init__(
            hf_model_id=hf_model_id,
            prompt_template=prompt_template,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            device_map=device_map,
        )

        # Logger setup
        logger.remove()
        logger.add(
            sys.stdout,
            format="<green>{level}</green>: <level>{message}</level>",
            level="INFO",
        )
        logger.add(
            sys.stdout,
            format="<red>{level}</red>: <level>{message}</level>",
            level="ERROR",
        )

        logger.info(f"Initializing local Hugging Face model: {hf_model_id}")

        # Create BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )

        # Initialize tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()

        logger.info("Local Hugging Face model initialized successfully.")

    def _generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        repetition_penalty: float = 1.1,
        **generate_kwargs,
    ) -> str:
        """
        Internal method to generate text from the model given a prompt.
        """
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_tokens = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                **generate_kwargs,
            )

        return self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    def _attempt_answer_question(self, context: str, question: str) -> str:
        """
        Generates an answer to the given question, based on the provided context.
        """
        prompt = self.prompt_template.format(context=context, question=question)
        logger.info("Generating answer locally with Hugging Face model...")
        response = self._generate_text(prompt)
        if "Answer:" in response:
            return response.split("Answer:", 1)[-1].strip()
        return response

    def answer_question(
        self,
        context: str,
        question: str,
        max_tokens: int = 200,
        prompt_template: str = None,
    ) -> str:
        """
        Answer a question based on the given context using the local HF model.
        """
        logger.info("Answering question with local HF model.")
        if prompt_template:
            self.prompt_template = prompt_template

        try:
            return self._attempt_answer_question(context, question)
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return str(e)

    def get_summary(self, documentation: str) -> str:
        """
        Generates a detailed summary of the provided documentation.
        """
        logger.info("Generating summary locally with HF model...")
        prompt = (
            "You are a helpful assistant. Provide a detailed summary of the documentation below.\n\n"
            f"Documentation:\n{documentation}\n\nSummary:"
        )
        try:
            response = self._generate_text(prompt)
            if "Summary:" in response:
                return response.split("Summary:", 1)[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Error in get_summary: {e}")
            return str(e)

    def grade_docs(self, context: List[str], question: str) -> List[str]:
        """
        Grades documents for relevance to the question. Returns relevant docs only.
        """
        filtered_docs = []
        logger.info("Grading documents for relevance...")

        system_prompt = (
            "You are a grader assessing the relevance of a retrieved document to a user question.\n"
            "If the document contains keywords or information related to the user question, grade it as 'yes'.\n"
            "Otherwise, grade it as 'no'. Provide the answer with no explanation.\n"
        )

        for doc in context:
            user_prompt = (
                f"Document:\n{doc}\nUser question:\n{question}\nAnswer yes or no:\n"
            )
            try:
                response = self._generate_text(
                    system_prompt + user_prompt, max_new_tokens=20
                )
                final_answer = response.strip().lower()
                if final_answer.endswith("yes"):
                    logger.info("Relevant doc")
                    filtered_docs.append(doc)
                elif final_answer.endswith("no"):
                    logger.info("Not relevant doc")
                else:
                    logger.info("Unclear answer; skipping doc")
            except Exception as e:
                logger.error(f"Error grading doc: {e}")
        return filtered_docs

    def check_hallucination(self, context: str, answer: str) -> str:
        """
        Checks if an answer is grounded in the provided context.
        Returns 'yes' if grounded, 'no' otherwise.
        """
        logger.info("Checking for hallucination locally with HF model...")
        system_prompt = (
            "You are a grader assessing whether an answer is supported by the provided facts.\n"
            "Provide a 'yes' or 'no' response with no further explanation.\n"
        )
        user_prompt = f"Facts:\n{context}\nAnswer:\n{answer}\nIs the answer supported by these facts?\n"
        try:
            response = (
                self._generate_text(system_prompt + user_prompt, max_new_tokens=20)
                .strip()
                .lower()
            )
            if "yes" in response:
                return "yes"
            elif "no" in response:
                return "no"
            return "no"
        except Exception as e:
            logger.error(f"Error in check_hallucination: {e}")
            return str(e)

    def chat(self, prompt: str, system_prompt: str) -> str:
        """
        A simple chat-like method to combine a system prompt with user input.
        """
        logger.info("Chatting locally with HF model...")
        try:
            full_prompt = f"{system_prompt}\n{prompt}"
            return self._generate_text(full_prompt)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return str(e)
