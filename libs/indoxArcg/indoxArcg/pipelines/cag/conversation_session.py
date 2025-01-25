from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
import json


class ConversationSession:
    def __init__(self):
        self.history = []  # Stores the conversation history as a list of dictionaries

    def add_to_history(self, user_query: str, model_response: str):
        """
        Add a user query and the model's response to the conversation history.

        Args:
            user_query (str): User's input query.
            model_response (str): Model's response.
        """
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "model", "content": model_response})

    def get_relevant_context(
        self, query: str, fixed_last_n: int = 3, top_k: int = 10
    ) -> str:
        """
        Retrieve the most recent conversation turns and the most relevant
        parts of the conversation history using BM25.

        Args:
            query (str): The current user query.
            fixed_last_n (int): Number of most recent conversation turns to include.
            top_k (int): Number of most relevant entries to retrieve.

        Returns:
            str: Relevant parts of the conversation history, formatted as Q&A pairs.
        """
        if not self.history:
            return ""

        # Step 1: Include the fixed last N turns of the conversation
        fixed_context = self.history[
            -fixed_last_n * 2 :
        ]  # Include user and model turns

        # Format fixed context as Q&A pairs
        fixed_context_text = ""
        for i in range(0, len(fixed_context), 2):
            user_entry = fixed_context[i]
            model_entry = fixed_context[i + 1] if i + 1 < len(fixed_context) else None
            if model_entry:
                fixed_context_text += (
                    f"Last Question: {user_entry['content']}\n"
                    f"Last Answer: {model_entry['content']}\n\n"
                )

        # Step 2: Perform similarity search for additional context
        tokenized_history = [entry["content"].split() for entry in self.history]
        tokenized_query = query.split()

        bm25 = BM25Okapi(tokenized_history)
        scores = bm25.get_scores(tokenized_query)
        scored_history = list(zip(self.history, scores))
        scored_history.sort(key=lambda x: x[1], reverse=True)

        # Select the top-k most relevant entries, avoiding duplication with fixed context
        additional_entries = [
            entry
            for entry, score in scored_history[:top_k]
            if entry not in fixed_context
        ]

        # Format additional context as Q&A pairs
        additional_context_text = ""
        for i in range(0, len(additional_entries), 2):
            user_entry = additional_entries[i]
            model_entry = (
                additional_entries[i + 1] if i + 1 < len(additional_entries) else None
            )
            if model_entry:
                additional_context_text += (
                    f"Question: {user_entry['content']}\n"
                    f"Answer: {model_entry['content']}\n\n"
                )

        # Combine the fixed context and additional context
        combined_context = f"{fixed_context_text}{additional_context_text}".strip()
        return combined_context

    def get_full_conversation(self) -> str:
        """
        Retrieve the full conversation as a formatted string.

        Returns:
            str: The full conversation with roles annotated.
        """
        return "\n".join(
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in self.history
        )

    def get_recent_context(self, n_turns: int = 3) -> str:
        """
        Get the most recent n conversation turns, formatted as Q&A pairs.

        Args:
            n_turns (int): Number of recent turns to retrieve.

        Returns:
            str: Formatted recent context as Q&A pairs.
        """
        recent_history = self.history[-(n_turns * 2) :]
        formatted_context = ""
        for i in range(0, len(recent_history), 2):
            user_entry = recent_history[i]
            model_entry = recent_history[i + 1] if i + 1 < len(recent_history) else None
            if model_entry:
                formatted_context += (
                    f"Last Question: {user_entry['content']}\n"
                    f"Last Answer: {model_entry['content']}\n\n"
                )
        return formatted_context.strip()

    def generate_llm_input(
        self, query: str, fixed_last_n: int = 3, top_k: int = 10
    ) -> str:
        """
        Generate the JSON input string for the LLM, including relevant context.

        Args:
            query (str): The current user query.
            fixed_last_n (int): Number of recent conversation turns to include.
            top_k (int): Number of most relevant history entries to include.

        Returns:
            str: JSON string to be sent to the LLM.
        """
        relevant_context = self.get_relevant_context(query, fixed_last_n, top_k)
        llm_input = {
            "system_instructions": "You are a helpful assistant. Provide clear, concise, and accurate responses.",
            "conversation_history": self.history,
            "retrieved_context": relevant_context,
            "current_input": query,
        }
        return json.dumps(llm_input, indent=4)
