import os
import hashlib
from uuid import uuid4
 
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
 
load_dotenv()
 
 
class VECTOR_DB:
 
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = os.getenv("PINECONE_INDEX")
        self.vector_store = None
 
        # main.py ka /api/status inhe padhta hai (index ka naam aur vector count ke liye)
        self.index_name = self.pinecone_index
        self.pc = Pinecone(api_key=self.pinecone_api_key) if self.pinecone_api_key else None
 
    @staticmethod
    def _stable_id(doc):
        """Same content -> same id, taaki server restart par duplicate vectors na bane."""
        raw = f"{doc.metadata.get('source', '')}|{doc.metadata.get('page', '')}|{doc.page_content}"
        return "startup-" + hashlib.md5(raw.encode("utf-8")).hexdigest()
 
    def create_vector_store(self):
 
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is missing in .env")
 
        if not self.pinecone_index:
            raise ValueError("PINECONE_INDEX is missing in .env")
 
        pinecone = Pinecone(api_key=self.pinecone_api_key)
 
        print("INDEX NAME:", self.pinecone_index)
 
        if not pinecone.has_index(self.pinecone_index):
 
            print("Index does not exist. Creating new index...")
 
            pinecone.create_index(
                name=self.pinecone_index,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
 
        index_info = pinecone.describe_index(self.pinecone_index)
        print("INDEX INFO:", index_info)
 
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.pinecone_index,
            embedding=self.embeddings
        )
 
        if self.documents:
            # pehle yahan random ids bante the, isliye har restart par wahi chunks dobara jud jate the
            self.vector_store.add_documents(
                documents=self.documents,
                ids=[self._stable_id(d) for d in self.documents],
            )
 
        return self.vector_store
 
    def clear(self):
 
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized")
 
        try:
            self.vector_store.delete(delete_all=True)
        except Exception as e:
            # index pehle se khali ho to Pinecone 404 "Namespace not found" deta hai
            if "Namespace not found" not in str(e) and "404" not in str(e):
                raise
 
        print("Vector index cleared successfully")
 
    def add_documents(self, documents, prefix=None):
 
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized")
 
        ids = [
            f"{prefix or 'document'}-{uuid4().hex}"
            for _ in documents
        ]
 
        self.vector_store.add_documents(
            documents=documents,
            ids=ids
        )
 
        print(f"{len(documents)} documents added successfully")
 
 
if __name__ == "__main__":
 
    print("API Key:", bool(os.getenv("PINECONE_API_KEY")))
    print("Index:", os.getenv("PINECONE_INDEX"))
 
