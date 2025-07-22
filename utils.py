import os
import uuid
import chromadb
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

class OllamaEmbeddingFunction:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input):
        return [ollama.embeddings(model=self.model_name, prompt=text)["embedding"] for text in input]

    def name(self):
        return f"ollama-{self.model_name}"

def get_chroma():
    persist_path = os.path.join(os.path.dirname(__file__), "chromadb_data")
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name="llm_docs",
        embedding_function = OllamaEmbeddingFunction()

    )
    return collection

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
        chunk_size=1000,
        chunk_overlap=100,
        separators=[".", "\n", " ", "(", ")", "{", "}"]
    )
    return splitter.split_text(text)

# Belgeleri veritabanına ekleme ve chunk'ları dosyaya yazma
def add_documents_from_folder(folder_path):
    collection = get_chroma()
    all_chunks = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        text = extract_text(file_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        for chunk in chunks:
            collection.add(documents=[chunk], ids=[str(uuid.uuid4())])
        print(f"{file} -> {len(chunks)} parça eklendi.")

    with open("chunk_debug_output.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks, 1):
            f.write(f"[Chunk {i}]:\n{chunk}\n\n")

if __name__ == "__main__":
    add_documents_from_folder("pdf_files/egitim_ogrenci_isleri_pdf")
