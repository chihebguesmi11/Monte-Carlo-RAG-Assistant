import os
import shutil
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

os.environ["HUGGINGFACEHUB_API_TOKEN"] = apikey

# File paths
PDF_PATH = "cm_monte_carlo.pdf"
CHROMA_PATH = "chroma_db"

# === Custom Embedding class using Hugging Face Inference API ===
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-large", api_key=None):
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=api_key)

    def embed_documents(self, texts):
        """Return one embedding per input text (must match Chroma's expectations)."""
        embeddings = []
        for text in texts:
            response = self.client.feature_extraction(text)
            embeddings.append(response)  
        return embeddings

    def embed_query(self, text):
        return self.client.feature_extraction(text)

# === Load and clean PDF pages ===
def load_pdf(pdf_path):
    print(f" Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    documents = [
        Document(page_content=page.get_text(), metadata={"page": i})
        for i, page in enumerate(doc)
        if page.get_text().strip()
    ]
    print(f" Loaded {len(documents)} pages with text.")
    return documents

# === Split long text into chunks ===
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f" Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# === Embed and store into Chroma vector DB ===
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print(" Loading multilingual embedding model via Hugging Face Inference API...")
    embedding = HuggingFaceAPIEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        api_key=apikey
    )

    db = Chroma.from_documents(chunks, embedding=embedding, persist_directory=CHROMA_PATH)
    db.persist()
    print(f" Saved {len(chunks)} chunks to {CHROMA_PATH}")

# === Run the whole pipeline ===
def main():
    documents = load_pdf(PDF_PATH)
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
