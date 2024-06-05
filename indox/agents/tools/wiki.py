import wikipedia
from indox.agents.base import ToolInterface


class WikipediaTool(ToolInterface):
    name: str = "WikipediaTool"
    description: str = "Fetches summaries from Wikipedia based on the given input text."

    language: str ="en"


    def use(self, input_text: str) -> str:
        search = wikipedia.search(input_text)

        if len(search) > 0:
            return wikipedia.summary(search[0])
        else:
            return f"No Wikipedia page found for '{input_text}'"


# Example usage
if __name__ == "__main__":
    tool = WikipediaTool('en')
    print(tool.use("bu`in Miandasht"))
