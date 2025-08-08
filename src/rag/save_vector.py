"""Vector store creation and saving functionality."""

import os
from typing import List, Optional

import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.utils.logger import get_logger
from src.utils.rag_utils import create_embeddings, load_vector_store, vectordb_path

logger = get_logger(__name__)


def load_pdf_documents(
    pdf_path: Optional[str] = None, extra_metadata: Optional[dict] = None
) -> Optional[List[Document]]:
    """Load a PDF and return a list of Documents with metadata."""
    extra_metadata = extra_metadata or {}

    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return None

    logger.info(f"Loading PDF from: {pdf_path}")
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                metadata = {"page": i + 1, "source": pdf_path, **extra_metadata}
                documents.append(Document(page_content=text, metadata=metadata))

    logger.info(f"Loaded {len(documents)} pages from PDF.")
    return documents


def split_documents(
    documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100
) -> List[Document]:
    """Split documents into chunks using token-aware splitter."""
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} text chunks.")
    return chunks


def save_vector_store(
    documents: Optional[List[Document]] = None,
    pdf_path: Optional[str] = None,
    chroma_dir: str = vectordb_path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    force_recreate: bool = False,
    extra_metadata: Optional[dict] = None,
) -> Optional[Chroma]:
    """Create (or load) and persist a Chroma vector store."""
    try:
        if not force_recreate:
            existing_store = load_vector_store(chroma_dir)
            if existing_store:
                logger.info("Using existing vector store.")
                return existing_store

        if documents is None:
            documents = load_pdf_documents(pdf_path, extra_metadata=extra_metadata)
            if documents is None:
                return None

        chunks = split_documents(documents, chunk_size, chunk_overlap)
        embedding = create_embeddings()

        logger.info(f"Creating Chroma vectorstore in: {chroma_dir}")
        vectorstore = Chroma.from_documents(
            chunks, embedding, persist_directory=chroma_dir
        )
        logger.info("Vector store created and saved successfully.")
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None


def add_documents_to_vector_store(
    documents: List[Document], chroma_dir: Optional[str] = None
) -> Optional[Chroma]:
    """Add new documents to an existing vector store."""
    try:
        vectorstore = load_vector_store(chroma_dir)
        if not vectorstore:
            logger.error("No existing vector store found to add documents to.")
            return None

        chunks = split_documents(documents)
        logger.info("Adding new documents to existing vector store...")
        vectorstore.add_documents(chunks)
        logger.info("Documents added successfully.")
        return vectorstore

    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
        return None


def main():
    logger.info("Creating vector store...")
    pdf_path = "data/BHASKAR_SAIKIA_LMLE.pdf"
    extra_metadata = {"type": "resume", "source": pdf_path}
    vectorstore = save_vector_store(pdf_path=pdf_path, extra_metadata=extra_metadata)
    if not vectorstore:
        logger.error("Failed to create vector store.")
    else:
        logger.info("Vector store created successfully!")


if __name__ == "__main__":
    main()
