import os
import uuid
import chromadb
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

ollama_embedding_model = "mxbai-embed-large"

class OllamaEmbeddingFunction:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, input):
        return [ollama.embeddings(model=self.model_name, prompt=text)["embedding"] for text in input]

    def name(self):
        return f"ollama-{self.model_name}"

def get_chroma():
    persist_path = os.path.join(os.path.dirname(__file__), "chromadb_data")
    client = chromadb.PersistentClient(path=persist_path)
    embedding_function = OllamaEmbeddingFunction(ollama_embedding_model)
    collection = client.get_or_create_collection(
        name="llm_docs",
        embedding_function=embedding_function
    )
    return collection

# dosyadan metin çıkarma
def extract_text(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext.lower() == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=[".", "\n", " ", "(", ")", "{", "}"]
    )
    return splitter.split_text(text)

def add_documents_from_folder(folder_path):
    collection = get_chroma()
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        text = extract_text(file_path)
        chunks = chunk_text(text)
        for chunk in chunks:
            collection.add(documents=[chunk], ids=[str(uuid.uuid4())])
        print(f"{file} -> {len(chunks)} parça eklendi.")


if __name__ == "__main__":
    add_documents_from_folder("pdf_files/egitim_ogrenci_isleri_pdf")
