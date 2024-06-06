import re
from typing import List, Dict, Tuple

from indox.agents.base import ToolInterface
from indox.llms import IndoxApiOpenAiQa, IndoxApiOpenAiQaAgent

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
        self.messages = [
            {"role": "system", "content": self.sys_prompt}
        ]

    @staticmethod
    def extract_action_and_input(text):
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Output of LLM is not parsable for next tool use: `{text}`")
        tool = match.group(1).strip()
        tool_input = match.group(2)
        # print(tool_input)
        # breakpoint()
        return tool, tool_input.strip(" ").strip('"')

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
        self.messages.append({"role": "user", "content": prompt})
        loop_count = 0

        while self.max_loops is None or loop_count < self.max_loops:
            loop_count += 1

            response_text = self.llm._send_request(self.messages)

            # print(response_text)

            action, action_input = self.extract_action_and_input(response_text)
            if action in self.tool_names:
                print(f"Action: {action}\nAction Input : {action_input}")
                observation = self.tool_by_names[action].use(action_input)
                print(f"Observation: {observation}\n")
                self.messages.extend([
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": f"Observation: \"{observation}\""},
                ])
                # self.print_messages_markdown()
            elif action == "Response To Human":
                self.messages.append({"role": "assistant", "content": response_text})
                # self.print_messages_markdown()
                print(f"Response: {action_input}")
                break

    def print_messages_markdown(self):
        print("\n--- Chat Stream ---\n")
        for message in self.messages:
            role = message['role']
            content = message['content']
            if role == "system":
                print(f"**System:**\n```\n{content}\n```\n")
            elif role == "user":
                print(f"**User:** {content}\n")
            elif role == "assistant":
                print(f"**Assistant:** {content}\n")


class IndoxChatAgent:
    def __init__(self, sys_prompt=System_prompt, tools=None, max_loops=None):
        self.tools = tools
        self.max_loops = max_loops
        self.sys_prompt = sys_prompt.format(
            tool_description=self.tool_description,
            tool_names=self.tool_names,
        )
        self.messages = [
            {"role": "system", "content": self.sys_prompt}
        ]
        self.end_agent_process = False

    @staticmethod
    def extract_action_and_input(text):
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        if not match:
            raise ValueError(f"Output of LLM is not parsable for next tool use: `{text}`")
        tool = match.group(1).strip()
        tool_input = match.group(2)
        # print(tool_input)
        # breakpoint()
        return tool, tool_input.strip(" ").strip('"')

    @property
    def tool_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, ToolInterface]:
        return {tool.name: tool for tool in self.tools}

    def get_prompt(self, prompt):
        m = self.messages.copy()
        m.append({"role": "user", "content": prompt})
        print(f"Question : {prompt}")
        return m

    def run(self, response_text):

        action, action_input = self.extract_action_and_input(response_text)
        if action in self.tool_names:
            print(f"Action: {action}\nAction Input : {action_input}")
            observation = self.tool_by_names[action].use(action_input)
            print(f"Observation: {observation}\n")
            self.messages.extend([
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": f"Observation: \"{observation}\""},
            ])
            # self.print_messages_markdown()
            return self.messages
        elif action == "Response To Human":
            self.messages.append({"role": "assistant", "content": response_text})
            # self.print_messages_markdown()
            print(f"Response: {action_input}")
            self.end_agent_process = True
            return action_input

    def print_messages_markdown(self):
        print("\n--- Chat Stream ---\n")
        for message in self.messages:
            role = message['role']
            content = message['content']
            if role == "system":
                print(f"**System:**\n```\n{content}\n```\n")
            elif role == "user":
                print(f"**User:** {content}\n")
            elif role == "assistant":
                print(f"**Assistant:** {content}\n")


# if __name__ == "__main__":
#     import os
#     from dotenv import load_dotenv
#     from indox.agents.tools.repl import PythonREPLTool
#     from indox.agents.tools.search import SerpAPITool
#     from indox.agents.tools.wiki import WikipediaTool
#
#     print(load_dotenv())
#     INDOX_OPENAI_API_KEY = os.getenv("INDOX_OPENAI_API_KEY")
#     llm = IndoxApiOpenAiQaAgent(api_key=INDOX_OPENAI_API_KEY)
#     agent = IndoxAgent(llm=llm, tools=[PythonREPLTool(), WikipediaTool(), SerpAPITool()])
#     agent.run('how cinderella end her happy wedding? and who has written this her book?')
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    print(load_dotenv())
    INDOX_OPENAI_API_KEY = os.getenv("INDOX_OPENAI_API_KEY")
    llm = IndoxApiOpenAiQaAgent(api_key=INDOX_OPENAI_API_KEY)

    llm.answer_question('how cinderella end her happy wedding? and who has written this her book?')
