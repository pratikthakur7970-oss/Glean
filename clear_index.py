import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.Index("sentence-vector-index").delete(delete_all=True)
print("Index clear ho gaya")