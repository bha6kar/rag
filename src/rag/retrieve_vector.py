"""Vector store retrieval and querying functionality."""

from typing import Optional

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_google_vertexai import ChatVertexAI

from src.utils.logger import get_logger
from src.utils.model import get_vertex_model
from src.utils.rag_utils import load_vector_store, vectordb_path

logger = get_logger(__name__)


def setup_rag_chain(
    llm: ChatVertexAI,
    vectorstore: Chroma,
    filters: Optional[dict] = None,
    top_k: int = 10,
) -> Optional[RetrievalQA]:
    """Setup the RAG chain."""
    try:
        if filters:
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k, "filter": filters}
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        logger.info("RAG chain setup successfully")
        return rag_chain
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {str(e)}")
        return None


def query_rag(rag_chain: RetrievalQA, query: str) -> Optional[str]:
    """Query the RAG system."""
    if not rag_chain:
        logger.error("No RAG chain available")
        return None

    try:
        logger.info(f"Q: {query}")
        answer = rag_chain.invoke({"query": query})

        if isinstance(answer, dict) and "result" in answer:
            result = answer["result"]
        else:
            result = str(answer)

        logger.info(f"A: {result}")
        return result
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return None


def main():
    logger.info("Loading vector store for retrieval...")

    vectorstore = load_vector_store(vectordb_path)
    if not vectorstore:
        logger.error("Failed to load vector store")
        return None

    rag_chain = setup_rag_chain(
        llm=get_vertex_model(),
        vectorstore=vectorstore,
        filters={"type": "resume"},
        top_k=10,
    )
    if not rag_chain:
        logger.error("Failed to setup RAG chain")
        return None

    query = "Give me details about the candidate Bhaskar."
    query_rag(rag_chain, query)


if __name__ == "__main__":
    main()
