import os

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest_pdfs(
    pdf_dir: str,
    persist_dir: str = "data/chroma_db"
):
    documents = []

    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            loader = UnstructuredPDFLoader(
                os.path.join(pdf_dir, file),
                mode="elements"   # IMPORTANT
            )
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No documents loaded from PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ Ingestion completed successfully")
