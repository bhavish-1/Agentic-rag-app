from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os


def ingest_pdfs(
    pdf_dir: str,
    persist_dir: str = "data/chroma_db"
):
    """
    Load PDFs, split into chunks, embed, and store in ChromaDB.
    """

    # 1. Load all PDFs
    documents = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(
                os.path.join(pdf_dir, file)
            )
            documents.extend(loader.load())

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store in Chroma
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ Ingestion completed. PDFs embedded and stored.")
