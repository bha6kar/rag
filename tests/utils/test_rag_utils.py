import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import src.utils.rag_utils as rag_utils


@pytest.mark.unit
@patch("src.utils.rag_utils.HuggingFaceEmbeddings")
def test_create_embeddings_success(mock_hfemb):
    mock_instance = MagicMock()
    mock_hfemb.return_value = mock_instance
    result = rag_utils.create_embeddings()
    assert result == mock_instance
    mock_hfemb.assert_called_once_with(
        model_name=rag_utils.model_config.get("embedding_model_name")
    )


@pytest.mark.unit
@patch("src.utils.rag_utils.os.path.exists")
@patch("src.utils.rag_utils.Chroma")
@patch("src.utils.rag_utils.create_embeddings")
def test_load_vector_store_success(mock_create_emb, mock_chroma, mock_exists):
    mock_exists.return_value = True
    mock_emb = MagicMock()
    mock_create_emb.return_value = mock_emb
    mock_chroma_instance = MagicMock()
    mock_chroma.return_value = mock_chroma_instance
    result = rag_utils.load_vector_store("some_dir")
    assert result == mock_chroma_instance
    mock_chroma.assert_called_once_with(
        persist_directory="some_dir", embedding_function=mock_emb
    )


@pytest.mark.unit
@patch("src.utils.rag_utils.os.path.exists")
def test_load_vector_store_missing_dir(mock_exists, caplog):
    mock_exists.return_value = False
    with caplog.at_level("ERROR"):
        result = rag_utils.load_vector_store("missing_dir")
    assert result is None
    assert "Vector store not found at: missing_dir" in caplog.text


@pytest.mark.unit
@patch("src.utils.rag_utils.os.path.exists")
@patch("src.utils.rag_utils.create_embeddings")
@patch("src.utils.rag_utils.Chroma")
def test_load_vector_store_exception(mock_chroma, mock_create_emb, mock_exists, caplog):
    mock_exists.return_value = True
    mock_create_emb.return_value = MagicMock()
    mock_chroma.side_effect = Exception("fail")
    with caplog.at_level("ERROR"):
        result = rag_utils.load_vector_store("some_dir")
    assert result is None
    assert "Error loading vector store: fail" in caplog.text


@pytest.mark.integration
def test_vector_store_create_and_load_real():
    """End-to-end test creating and loading a real vector store."""
    temp_dir = tempfile.mkdtemp()

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(persist_directory=temp_dir, embedding_function=embeddings)
        vectorstore.add_documents(
            [
                Document(page_content="The Eiffel Tower is in Paris."),
                Document(page_content="The Tower of London is in London."),
            ]
        )

        loaded_store = rag_utils.load_vector_store(chroma_dir=temp_dir)
        assert loaded_store is not None

        results = loaded_store.similarity_search("Where is the Eiffel Tower?", k=1)
        assert results
        assert "Eiffel Tower" in results[0].page_content

    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.integration
def test_vector_store_with_default_path(tmp_path, monkeypatch):
    """Integration test for loading vector store using the default path from config."""
    monkeypatch.setattr(rag_utils, "vectordb_path", str(tmp_path))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    store = Chroma(persist_directory=str(tmp_path), embedding_function=embeddings)
    store.add_documents([Document(page_content="Big Ben is in London.")])

    loaded_store = rag_utils.load_vector_store(str(tmp_path))
    results = loaded_store.similarity_search("Where is Big Ben?", k=1)

    assert results
    assert "Big Ben" in results[0].page_content
