#outputs=retriever.invoke('what is Attention')
from dotenv import load_dotenv
load_dotenv()   # sabse pehle

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from prompt import PROMPT
from models import LLM

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="sentence-vector-index", embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = PROMPT().get_prompt()      # class se asli prompt object nikalna


class RETRIEVER:
    def __init__(self, retriever):
        self.retriever = retriever


def doc_out(outputs):
    s = '\n\n'.join(f"[Source {i+1}] {out.page_content}" for i, out in enumerate(outputs))
    return s


chain = ({'context': retriever | doc_out,
          'question': lambda x: x}
         |
         prompt
         |
         LLM
         |
         StrOutputParser())


if __name__ == "__main__":
    r = chain.invoke('what is Attention')
    print(r)