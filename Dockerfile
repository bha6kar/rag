# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock poetry.toml ./


# Install dependencies
RUN poetry config virtualenvs.create false && poetry install


# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data chroma_db

# Expose port for retrieve service
EXPOSE 8000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    if [ "$1" = "save" ]; then\n\
    exec python -m src.rag.save_vector\n\
    elif [ "$1" = "retrieve" ]; then\n\
    exec python -m src.rag.retrieve_vector\n\
    else\n\
    echo "Usage: docker run <image> [save|retrieve]"\n\
    echo "  save     - Process documents and create vector store"\n\
    echo "  retrieve - Run RAG query service"\n\
    exit 1\n\
    fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (can be overridden)
CMD ["retrieve"]
