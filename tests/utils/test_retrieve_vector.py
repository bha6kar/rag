from unittest.mock import MagicMock, patch

import pytest
from langchain_chroma import Chroma
from langchain_google_vertexai import ChatVertexAI

from src.rag import retrieve_vector


@pytest.mark.unit
def test_setup_rag_chain_with_filters():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = "mock_retriever"
    mock_llm = MagicMock()

    with patch("langchain.chains.RetrievalQA.from_chain_type") as mock_from_chain_type:
        mock_from_chain_type.return_value = "mock_chain"

        chain = retrieve_vector.setup_rag_chain(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            filters={"type": "resume"},
            top_k=5,
        )

    mock_vectorstore.as_retriever.assert_called_once_with(
        search_kwargs={"k": 5, "filter": {"type": "resume"}}
    )
    assert chain == "mock_chain"


@pytest.mark.unit
def test_setup_rag_chain_without_filters():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = "mock_retriever"
    mock_llm = MagicMock()

    with patch("langchain.chains.RetrievalQA.from_chain_type") as mock_from_chain_type:
        mock_from_chain_type.return_value = "mock_chain"

        chain = retrieve_vector.setup_rag_chain(
            llm=mock_llm,
            vectorstore=mock_vectorstore,
            filters=None,
            top_k=10,
        )

    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 10})
    assert chain == "mock_chain"


@pytest.mark.unit
def test_setup_rag_chain_exception():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.side_effect = Exception("fail")
    mock_llm = MagicMock()

    chain = retrieve_vector.setup_rag_chain(
        llm=mock_llm, vectorstore=mock_vectorstore, filters=None, top_k=10
    )
    assert chain is None


@pytest.mark.unit
def test_query_rag_success():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"result": "answer"}

    result = retrieve_vector.query_rag(mock_chain, "some query")
    assert result == "answer"


@pytest.mark.unit
def test_query_rag_returns_str_when_not_dict():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "plain string result"

    result = retrieve_vector.query_rag(mock_chain, "some query")
    assert result == "plain string result"


@pytest.mark.unit
def test_query_rag_no_chain():
    result = retrieve_vector.query_rag(None, "query")
    assert result is None


@pytest.mark.unit
def test_query_rag_exception():
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("fail")

    result = retrieve_vector.query_rag(mock_chain, "query")
    assert result is None


@pytest.mark.integration
@patch("src.rag.retrieve_vector.load_vector_store")
@patch("src.rag.retrieve_vector.logger")
def test_main_fails_to_load_vector_store(mock_logger, mock_load_store):
    mock_load_store.return_value = None

    retrieve_vector.main()

    mock_logger.error.assert_called_with("Failed to load vector store")


@pytest.mark.integration
@patch("src.rag.retrieve_vector.setup_rag_chain")
@patch("src.rag.retrieve_vector.get_vertex_model")
@patch("src.rag.retrieve_vector.load_vector_store")
@patch("src.rag.retrieve_vector.logger")
def test_main_fails_to_setup_rag_chain(
    mock_logger, mock_load_store, mock_get_model, mock_setup_chain
):
    mock_vectorstore = MagicMock(spec=Chroma)
    mock_load_store.return_value = mock_vectorstore

    mock_get_model.return_value = MagicMock(spec=ChatVertexAI)

    mock_setup_chain.return_value = None

    retrieve_vector.main()

    mock_logger.error.assert_called_with("Failed to setup RAG chain")
