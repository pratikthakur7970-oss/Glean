



from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_loader import PDFLoader

class SPLITTER:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def split_documents(self, documents):
        docs = self.splitter.split_documents(documents)
        return docs


if __name__ == "__main__":
    file_path = r"C:\Users\prati\Downloads\attention_is_all_you_need.pdf"

    
    loader = PDFLoader(file_path)
    documents = loader.load_pdf()