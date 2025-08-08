import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.llm_config import get_llm_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_llm_config()
rag_config = config["rag"]
model_config = config["model"]
vectordb_path = rag_config.get("vectordb_path")


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create embedding model."""
    embedding = HuggingFaceEmbeddings(
        model_name=model_config.get("embedding_model_name")
    )
    return embedding


def load_vector_store(chroma_dir: str = vectordb_path) -> Chroma:
    """Load an existing vector store."""

    try:
        if not os.path.exists(chroma_dir):
            logger.error(f"Vector store not found at: {chroma_dir}")
            logger.error("Please create a vector store first using save_vector.py")
            return None

        logger.info(f"Loading vector store from: {chroma_dir}")
        embedding = create_embeddings()
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding)
        logger.info("Vector store loaded successfully")
        return vectorstore

    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None
