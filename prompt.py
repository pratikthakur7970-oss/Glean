
from langchain_core.prompts import ChatPromptTemplate


class PROMPT:

    PROMPT = """
    You answer questions using only the supplied excerpts from indexed PDF files.

    If the excerpts do not contain enough information, say so and do not use general
    knowledge. Cite supporting excerpts with their [Source number].
    Keep the answer concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(self.PROMPT)

    def get_prompt(self):
        return self.prompt


if __name__ == "__main__":
    prompt = PROMPT()
    print(prompt.get_prompt())
