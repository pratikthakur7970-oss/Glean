from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

PDF_PATH = r"C:\Users\prati\Downloads\attention_is_all_you_need.pdf"

documents = PyPDFLoader(PDF_PATH).load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print("Total chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name="sentence-vector-index", embedding=embeddings)

ids = [f"attention-{i}" for i in range(len(chunks))]
vector_store.add_documents(chunks, ids=ids)
print("Upload ho gaya")