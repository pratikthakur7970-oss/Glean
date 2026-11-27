
from langchain_community.document_loaders import PyPDFLoader
import os


class PDFLoader:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_pdf(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(
                f"PDF file not found: {self.file_path}"
            )

        loader = PyPDFLoader(self.file_path)
        document = loader.load()
        return document


if __name__ == "__main__":
    file_path = os.getenv("PDF_PATH", "attention_is_all_you_need.pdf")
    loader = PDFLoader(file_path)
    documents = loader.load_pdf()
    print(f"Loaded {len(documents)} pages")
