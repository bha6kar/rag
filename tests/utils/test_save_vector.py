from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from reportlab.pdfgen import canvas

import src.rag.save_vector as svs


@pytest.mark.unit
def test_load_pdf_documents_file_not_found(tmp_path):
    pdf_path = tmp_path / "missing.pdf"
    docs = svs.load_pdf_documents(str(pdf_path))
    assert docs is None


@pytest.mark.unit
@patch("pdfplumber.open")
def test_load_pdf_documents_success(mock_pdf_open, tmp_path):
    fake_page1 = MagicMock()
    fake_page1.extract_text.return_value = "Hello world"
    fake_page2 = MagicMock()
    fake_page2.extract_text.return_value = "Another page"

    mock_pdf = MagicMock()
    mock_pdf.pages = [fake_page1, fake_page2]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("fake data")

    docs = svs.load_pdf_documents(str(pdf_path), extra_metadata={"foo": "bar"})
    assert len(docs) == 2
    assert docs[0].metadata["foo"] == "bar"


@pytest.mark.unit
def test_split_documents_splits_chunks():
    docs = [Document(page_content="A " * 1000, metadata={})]
    chunks = svs.split_documents(docs, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)


@pytest.mark.unit
@patch("src.rag.save_vector.load_vector_store", return_value=MagicMock())
def test_save_vector_store_uses_existing(mock_load):
    mock_store = mock_load.return_value
    store = svs.save_vector_store(
        documents=[Document(page_content="Test", metadata={})]
    )
    assert store == mock_store


@pytest.mark.unit
@patch(
    "src.rag.save_vector.load_pdf_documents",
    return_value=[Document(page_content="Test", metadata={})],
)
@patch(
    "src.rag.save_vector.split_documents",
    return_value=[Document(page_content="chunk", metadata={})],
)
@patch("src.rag.save_vector.create_embeddings", return_value="fake-embedding")
@patch("src.rag.save_vector.Chroma")
def test_save_vector_store_creates_new(
    mock_chroma, mock_emb, mock_split, mock_load_pdf
):
    mock_chroma.from_documents.return_value = "fake-store"
    store = svs.save_vector_store(force_recreate=True, pdf_path="file.pdf")
    assert store == "fake-store"


@pytest.mark.unit
@patch("src.rag.save_vector.load_vector_store", return_value=None)
def test_add_documents_to_vector_store_no_store(mock_load):
    result = svs.add_documents_to_vector_store(
        [Document(page_content="Test", metadata={})]
    )
    assert result is None


@pytest.mark.unit
@patch("src.rag.save_vector.load_vector_store")
@patch(
    "src.rag.save_vector.split_documents",
    return_value=[Document(page_content="chunk", metadata={})],
)
def test_add_documents_to_vector_store_success(mock_split, mock_load):
    fake_store = MagicMock()
    mock_load.return_value = fake_store
    result = svs.add_documents_to_vector_store(
        [Document(page_content="Test", metadata={})]
    )
    assert result == fake_store
    fake_store.add_documents.assert_called_once()


def create_test_pdf(pdf_path: Path, text_pages):
    """Create a simple PDF with given list of page texts."""
    c = canvas.Canvas(str(pdf_path))
    for text in text_pages:
        c.drawString(100, 750, text)
        c.showPage()
    c.save()


@pytest.mark.integration
def test_full_pipeline_with_real_pdf(tmp_path):
    # 1. Create PDF file with 2 pages of text
    pdf_file = tmp_path / "sample.pdf"
    create_test_pdf(
        pdf_file, ["Big Ben is in London.", "The Eiffel Tower is in Paris."]
    )

    # 2. Load PDF into documents
    docs = svs.load_pdf_documents(str(pdf_file), extra_metadata={"source": "test-pdf"})
    assert docs and len(docs) == 2
    assert "source" in docs[0].metadata

    # 3. Save vector store
    store = svs.save_vector_store(
        documents=docs, chroma_dir=str(tmp_path / "vectordb"), force_recreate=True
    )
    assert isinstance(store, Chroma)

    # 4. Load it back
    loaded_store = svs.load_vector_store(str(tmp_path / "vectordb"))
    assert loaded_store is not None

    # 5. Search for something in the PDF
    results = loaded_store.similarity_search("Where is Big Ben?", k=1)
    assert results
    assert "Big Ben" in results[0].page_content


@pytest.mark.integration
def test_add_documents_to_existing_store_with_real_docs(tmp_path):
    # Create initial vector store
    docs = [Document(page_content="The Colosseum is in Rome.", metadata={})]
    svs.save_vector_store(
        documents=docs, chroma_dir=str(tmp_path / "vectordb"), force_recreate=True
    )

    # Add a new document
    new_docs = [Document(page_content="Statue of Liberty is in New York.", metadata={})]
    store = svs.add_documents_to_vector_store(
        new_docs, chroma_dir=str(tmp_path / "vectordb")
    )
    assert isinstance(store, Chroma)

    # Search for the new document
    results = store.similarity_search("Where is the Statue of Liberty?", k=1)
    assert any("Statue of Liberty" in r.page_content for r in results)
