# Use official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Clone your repository and get latest code
RUN git clonehttps://github.com/Arqamansari23/Chat-With-Multiple-Books-With-Ensemble-Retriever-BM25-Search-Symentic-Search-.git /tmp/repo && \
    cp -r /tmp/repo/* /app/ && \
    rm -rf /tmp/repo

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for persistent data
RUN mkdir -p /app/vectorstores /app/temp_pdfs /app/data /app/static /app/templates

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
