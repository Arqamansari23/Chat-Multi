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

# Clone your repository
RUN git clone https://github.com/Arqamansari23/Chat-With-Multiple-Books-With-Ensemble-Retriever-BM25-Search-Symentic-Search-.git /app



# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
