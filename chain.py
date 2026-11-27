import os

from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings 
import os
from prompt import PROMPT
prompt = PROMPT().get_prompt()

class LLM:
    def __init__(self, api_key, model, embed_model, temperature=0):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=float(temperature)
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model
        )

if __name__ == "__main__":
    api_key =os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    embed=os.getenv("EMBED_MODEL")
    llm = LLM(api_key, model)



