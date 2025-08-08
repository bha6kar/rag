# RAG (Retrieval-Augmented Generation) Application

A Python-based Retrieval-Augmented Generation (RAG) system that enables intelligent document querying using vector embeddings and large language models. This application processes PDF documents, creates vector embeddings, and provides conversational access to document content through Google's Vertex AI.

## 🚀 Features

- **PDF Document Processing**: Extract and process text from PDF documents
- **Vector Embeddings**: Create and store document embeddings using ChromaDB
- **Intelligent Retrieval**: Semantic search and retrieval of relevant document chunks
- **LLM Integration**: Powered by Google Vertex AI (Gemini 2.0 Flash)
- **Configurable**: YAML-based configuration for model parameters
- **Testing Support**: Comprehensive test suite with unit and integration tests

## 📋 Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- Google Cloud Platform account with Vertex AI access
- Google Cloud credentials configured

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag
   ```

2. **Install dependencies**:
   ```bash
   make install
   # or
   poetry install
   ```

3. **Configure Google Cloud credentials**:
   ```bash
   # Set your GCP project ID
   export GCP_PROJECT="your-project-id"
   
   # Authenticate with Google Cloud
   gcloud auth application-default login
   ```

## ⚙️ Configuration

The application uses a YAML configuration file located at `src/config/llm-config.yml`:

```yaml
# Model config
project_id: "your-gcp-project-id"
location: "us-west1"
model_name: "gemini-2.0-flash-001"
embedding_model_name: "sentence-transformers/all-MiniLM-L6-v2"

# Generation config
temperature: 0.2
max_output_tokens: 1024
top_k: 40
top_p: 1

# RAG config
vectordb_path: "chroma_db"
```

### Environment Variables

- `GCP_PROJECT`: Your Google Cloud Project ID (optional, can be set in config)

## 📖 Usage

### 1. Processing Documents

To create vector embeddings from a PDF document:

```bash
# Using Makefile
make save

# Or directly with Python
poetry run python -m src.rag.save_vector
```

This will:
- Load the PDF from `data/BHASKAR_SAIKIA_LMLE.pdf`
- Split the document into chunks
- Create embeddings using sentence-transformers
- Store the vectors in ChromaDB

### 2. Querying Documents

To query the processed documents:

```bash
# Using Makefile
make retrieve

# Or directly with Python
poetry run python -m src.rag.retrieve_vector
```


## 🧪 Testing

Run the test suite using the provided Makefile commands:

```bash
# Run all tests
make test-all

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration
```

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Clean up cache files
make clean
```

### Available Make Commands

- `make install` - Install dependencies
- `make test-all` - Run all tests
- `make test-unit` - Run unit tests
- `make test-integration` - Run integration tests
- `make save` - Process documents and create vector store
- `make retrieve` - Query the vector store
- `make format` - Format code with Black and isort
- `make lint` - Check code formatting
- `make clean` - Clean up cache files

