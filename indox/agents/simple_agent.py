import re
from typing import List, Dict, Tuple

from base import ToolInterface
from indox.llms import IndoxApiOpenAiQa
System_prompt = """
Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

{tool_description}

You will receive a message from the human, then you should start a loop and do one of two things

Option 1: You use a tool to answer the question.
For this, you should use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: "the input to the action, to be sent to the tool"

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: "your response to the human, summarizing what you did and what you learned"

Begin!
"""


class IndoxAgent:
    def __init__(self, llm, sys_prompt=System_prompt, tools=None, max_loops=None):
        self.llm = llm
        self.tools = tools
        self.max_loops = max_loops
        self.sys_prompt = sys_prompt.format(
            tool_description=self.tool_description,
            tool_names=self.tool_names,
        )

    @staticmethod
    def extract_action_and_input(text):
        action_pattern = r"Action: (.+?)\n"
        input_pattern = r"Action Input: \"(.+?)\""
        action = re.findall(action_pattern, text)
        action_input = re.findall(input_pattern, text)
        return action, action_input

    @property
    def tool_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, ToolInterface]:
        return {tool.name: tool for tool in self.tools}

    def run(self, prompt):
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ]
        while True:

            response_text = self.llm.answer_with_agent(question=messages,context=None,tool_names=None,tool_description=None)

            print(response_text)

            # To prevent the Rate Limit error for free-tier users, we need to decrease the number of requests/minute.
            action, action_input = self.extract_action_and_input(response_text)
            # print(action,action_input)
            if action[-1] in self.tool_names:
                observation = self.tool_by_names[action[-1]].use(action_input[-1])
                print("Observation: ", observation)

            elif action[-1] == "Response To Human":
                print(f"Response: {action_input[-1]}")
                break

            messages.extend([
                {"role": "system", "content": response_text},
                {"role": "user", "content": f"Observation: \"{observation}\""},
            ])



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from indox.agents.tools.repl import PythonREPLTool
    from indox.agents.tools.search import SerpAPITool

    print(load_dotenv())
    INDOX_OPENAI_API_KEY = os.getenv("INDOX_OPENAI_API_KEY")
    llm = IndoxApiOpenAiQa(api_key=INDOX_OPENAI_API_KEY)
    agent = IndoxAgent(llm=llm, tools=[PythonREPLTool(), SerpAPITool()])
    agent.run('which team did win champions league 2024? is Ebrahim Raisi alive')
