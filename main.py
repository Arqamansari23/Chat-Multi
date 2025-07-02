import os
import shutil
import re
import unicodedata
import fitz  # PyMuPDF
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import asyncio

import logging


# === Config ===
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = "vectorstores"
TEMP_DIR = "temp_pdfs"
METADATA_FILE = "books_metadata.json"  # New: Store book metadata
MAX_FILE_SIZE_MB = 500
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, streaming=True)

# === Helper Functions ===
def sanitize_filename(filename):
    """
    Sanitize filename to remove problematic characters
    """
    # Remove file extension for processing
    name, ext = os.path.splitext(filename)
    
    # Normalize unicode characters (converts em dash to regular dash, etc.)
    name = unicodedata.normalize('NFKD', name)
    
    # Replace problematic characters with safe alternatives
    name = re.sub(r'[–—]', '-', name)  # Replace em dash and en dash with hyphen
    name = re.sub(r'[^\w\s\-_\.]', '', name)  # Remove special chars except word chars, spaces, hyphens, underscores, dots
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = name.strip('_-.')  # Remove leading/trailing separators
    
    # Ensure it's not empty
    if not name:
        name = "document"
    
    return name + ext

def save_books_metadata(books_metadata):
    """
    Save books metadata to JSON file
    """
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(books_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Books metadata saved to {METADATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving books metadata: {str(e)}")

def load_books_metadata():
    """
    Load books metadata from JSON file
    """
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded books metadata: {len(metadata)} books")
            return metadata
        else:
            logger.info("No existing books metadata found")
            return {}
    except Exception as e:
        logger.error(f"Error loading books metadata: {str(e)}")
        return {}

def update_book_metadata(title, original_filename, sanitized_filename, chunk_count):
    """
    Update metadata for a single book
    """
    metadata = load_books_metadata()
    metadata[title] = {
        "original_filename": original_filename,
        "sanitized_filename": sanitized_filename,
        "chunk_count": chunk_count,
        "upload_date": str(datetime.now()),
        "vector_store_path": os.path.join(VECTORSTORE_DIR, title),
        "temp_file_path": os.path.join(TEMP_DIR, sanitized_filename)
    }
    save_books_metadata(metadata)

def remove_book_metadata(title):
    """
    Remove metadata for a single book
    """
    metadata = load_books_metadata()
    if title in metadata:
        del metadata[title]
        save_books_metadata(metadata)

def load_pdf_with_pymupdf(path):
    """
    Load PDF using PyMuPDF with enhanced text extraction
    """
    try:
        logger.info(f"Loading PDF with PyMuPDF: {path}")
        documents = []
        
        # Open PDF document
        doc = fitz.open(path)
        
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text with better formatting preservation
            text = page.get_text("text")
            
            # Alternative: get text with layout information (slower but better quality)
            # text = page.get_text("blocks")  # Returns structured blocks
            # text = "\n".join([block[4] for block in text if block[6] == 0])  # Extract text from blocks
            
            if text and text.strip():  # Only add pages with actual text content
                # Clean up the text
                text = clean_extracted_text(text)
                
                document = Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "page": page_num + 1,
                        "total_pages": doc.page_count
                    }
                )
                documents.append(document)
        
        doc.close()  # Always close the document
        
        if not documents:
            raise ValueError("No text content found in PDF")
        
        logger.info(f"Successfully loaded PDF with PyMuPDF: {len(documents)} pages")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading PDF with PyMuPDF {path}: {str(e)}")
        # Ensure document is closed even if error occurs
        try:
            if 'doc' in locals():
                doc.close()
        except:
            pass
        raise

def clean_extracted_text(text):
    """
    Clean up extracted text from PyMuPDF
    """
    # Remove excessive whitespace while preserving paragraph structure
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
    
    # Remove trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()

def extract_pdf_with_layout_analysis(path):
    """
    Advanced PDF extraction with layout analysis using PyMuPDF
    This function provides better structure recognition
    """
    try:
        logger.info(f"Extracting PDF with layout analysis: {path}")
        documents = []
        
        doc = fitz.open(path)
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Get text blocks with position information
            blocks = page.get_text("dict")
            
            # Extract text maintaining structure
            page_text = ""
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        page_text += line_text + "\n"
                    page_text += "\n"  # Add spacing between blocks
            
            if page_text.strip():
                page_text = clean_extracted_text(page_text)
                
                document = Document(
                    page_content=page_text,
                    metadata={
                        "source": path,
                        "page": page_num + 1,
                        "total_pages": doc.page_count,
                        "extraction_method": "layout_analysis"
                    }
                )
                documents.append(document)
        
        doc.close()
        
        if not documents:
            raise ValueError("No text content found using layout analysis")
        
        logger.info(f"Successfully extracted PDF with layout analysis: {len(documents)} pages")
        return documents
        
    except Exception as e:
        logger.error(f"Layout analysis failed for {path}: {str(e)}")
        try:
            if 'doc' in locals():
                doc.close()
        except:
            pass
        raise

def recover_existing_vectorstores():
    """
    Recover existing vector stores on application startup
    """
    logger.info("Starting vector store recovery process...")
    
    if not os.path.exists(VECTORSTORE_DIR):
        logger.info("No vectorstore directory found")
        return {}
    
    recovered_retrievers = {}
    books_metadata = load_books_metadata()
    
    # Get all subdirectories in vectorstore directory
    try:
        vectorstore_dirs = [d for d in os.listdir(VECTORSTORE_DIR) 
                          if os.path.isdir(os.path.join(VECTORSTORE_DIR, d))]
        
        for title in vectorstore_dirs:
            try:
                logger.info(f"Attempting to recover vector store: {title}")
                
                # Check if FAISS index files exist
                vectorstore_path = os.path.join(VECTORSTORE_DIR, title)
                index_file = os.path.join(vectorstore_path, "index.faiss")
                pkl_file = os.path.join(vectorstore_path, "index.pkl")
                
                if os.path.exists(index_file) and os.path.exists(pkl_file):
                    # Load the existing FAISS vector store
                    vectorstore = FAISS.load_local(
                        vectorstore_path, 
                        EMBEDDING_MODEL,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Get document chunks from the vector store
                    # Note: We can't directly get chunks from FAISS, so we'll create a basic retriever
                    faiss_retriever = vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={'score_threshold': 0.8, 'k': 4}
                    )
                    
                    # For BM25, we need the original documents, but since we can't get them back,
                    # we'll use a fallback approach with just FAISS
                    recovered_retrievers[title] = faiss_retriever
                    
                    logger.info(f"Successfully recovered vector store: {title}")
                    
                    # Update metadata if not present
                    if title not in books_metadata:
                        books_metadata[title] = {
                            "original_filename": f"{title}.pdf",
                            "sanitized_filename": f"{title}.pdf",
                            "chunk_count": "unknown",
                            "upload_date": "recovered",
                            "vector_store_path": vectorstore_path,
                            "temp_file_path": "recovered"
                        }
                else:
                    logger.warning(f"Missing FAISS files for {title}, skipping recovery")
                    
            except Exception as e:
                logger.error(f"Failed to recover vector store {title}: {str(e)}")
                continue
        
        # Save updated metadata
        if books_metadata:
            save_books_metadata(books_metadata)
        
        logger.info(f"Recovery completed. Recovered {len(recovered_retrievers)} vector stores.")
        return recovered_retrievers
        
    except Exception as e:
        logger.error(f"Error during vector store recovery: {str(e)}")
        return {}

# === Prompt ===
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful assistant. Use only the provided context to answer the question.
        Format your answers in a clear, structured way using appropriate markdown headings (e.g., '### Heading').
        Please do not include any information that is not present in the context.
        If the provided context is empty or does not contain the answer, you MUST respond with "I don't know". Do not use your own knowledge.
        Context:
        {context}
        """
    ),
    ("human", "{input}")
])

llm_chain = create_stuff_documents_chain(LLM, prompt)

# === FastAPI Setup ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever map - will be populated on startup
retriever_map = {}

def load_and_split_pdf(path):
    """
    Load and split PDF using PyMuPDF with fallback to layout analysis
    """
    try:
        logger.info(f"Loading PDF: {path}")
        
        # First, try standard text extraction
        try:
            pages = load_pdf_with_pymupdf(path)
            logger.info(f"Successfully loaded PDF with standard PyMuPDF extraction")
            
        except Exception as standard_error:
            logger.warning(f"Standard PyMuPDF extraction failed for {path}: {str(standard_error)}")
            logger.info("Attempting layout analysis extraction...")
            
            # Fallback to layout analysis
            try:
                pages = extract_pdf_with_layout_analysis(path)
                logger.info(f"Successfully loaded PDF with layout analysis")
            except Exception as layout_error:
                logger.error(f"Layout analysis also failed: {str(layout_error)}")
                raise Exception(f"Both extraction methods failed. Standard: {str(standard_error)}, Layout: {str(layout_error)}")
        
        # Use sanitized filename for title
        original_filename = os.path.basename(path)
        title = sanitize_filename(original_filename).replace(".pdf", "")
        
        logger.info(f"Splitting PDF into chunks: {title}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separation for PyMuPDF extracted text
        )
        chunks = splitter.split_documents(pages)
        
        if not chunks:
            raise ValueError("No text content found in PDF after splitting")
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata["book_title"] = title
            chunk.metadata["extraction_tool"] = "pymupdf"
            
        logger.info(f"Successfully processed PDF: {title} ({len(chunks)} chunks)")
        return chunks, title
        
    except Exception as e:
        logger.error(f"Error processing PDF {path}: {str(e)}")
        raise

def save_faiss_index(chunks, title):
    """
    Save FAISS index with error handling
    """
    try:
        logger.info(f"Creating FAISS index for: {title}")
        vectorstore = FAISS.from_documents(chunks, EMBEDDING_MODEL)
        save_path = os.path.join(VECTORSTORE_DIR, title)
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        logger.info(f"FAISS index saved: {title}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error saving FAISS index for {title}: {str(e)}")
        raise

def create_ensemble_retriever(chunks, vectorstore):
    """
    Create ensemble retriever with error handling
    """
    try:
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 4

        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.8, 'k': 4}
        )

        retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25],
            weights=[0.6, 0.4]
        )
        return retriever
    except Exception as e:
        logger.error(f"Error creating ensemble retriever: {str(e)}")
        raise

# === Streaming Generator Function ===
async def generate_streaming_response(books: List[str], query: str):
    """
    Generate streaming response for the multi-book query
    """
    try:
        combined_docs = []

        # Collect documents from all selected books
        for book in books:
            retriever = retriever_map.get(book)
            if retriever:
                try:
                    docs = retriever.invoke(query)
                    combined_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Error querying book {book}: {str(e)}")
                    continue

        # If no relevant documents are found, return "I don't know"
        if not combined_docs:
            yield "data: " + json.dumps({"content": "I don't know.", "done": True}) + "\n\n"
            return

        # Stream the response from LLM
        try:
            # Get the streaming response from the chain
            response_stream = llm_chain.astream({"input": query, "context": combined_docs})
            
            async for chunk in response_stream:
                if chunk:
                    # Send each chunk as server-sent events
                    yield "data: " + json.dumps({"content": chunk, "done": False}) + "\n\n"
                    # Add a small delay to make streaming more visible
                    await asyncio.sleep(0.05)
            
            # Send completion signal
            yield "data: " + json.dumps({"content": "", "done": True}) + "\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield "data: " + json.dumps({"content": f"Error generating response: {str(e)}", "done": True}) + "\n\n"
            
    except Exception as e:
        logger.error(f"Error in generate_streaming_response: {str(e)}")
        yield "data: " + json.dumps({"content": f"Query processing failed: {str(e)}", "done": True}) + "\n\n"

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    """
    Initialize application and recover existing vector stores
    """
    logger.info("Application starting up...")
    
    # Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    # Recover existing vector stores
    global retriever_map
    retriever_map = recover_existing_vectorstores()
    
    logger.info(f"Application startup completed. {len(retriever_map)} books loaded.")

# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/upload_books/")
async def upload_books(files: List[UploadFile]):
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    added_books = []
    errors = []

    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith(".pdf"):
                errors.append(f"{file.filename} is not a PDF.")
                continue

            # Read file content
            content = await file.read()
            size_mb = len(content) / (1024 * 1024)
            
            # Validate file size
            if size_mb > MAX_FILE_SIZE_MB:
                errors.append(f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit.")
                continue

            # Sanitize filename before saving
            sanitized_filename = sanitize_filename(file.filename)
            file_path = os.path.join(TEMP_DIR, sanitized_filename)
            
            logger.info(f"Saving file: {file.filename} -> {sanitized_filename}")
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)

            # Process PDF
            try:
                chunks, title = load_and_split_pdf(file_path)
                
                # Check if book already exists
                if title in retriever_map:
                    errors.append(f"Book '{title}' already exists. Please delete it first or use a different filename.")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue
                
                vectorstore = save_faiss_index(chunks, title)
                retriever = create_ensemble_retriever(chunks, vectorstore)
                retriever_map[title] = retriever
                
                # Update metadata
                update_book_metadata(title, file.filename, sanitized_filename, len(chunks))
                
                added_books.append(title)
                
            except Exception as pdf_error:
                error_msg = f"Failed to process {file.filename}: {str(pdf_error)}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Clean up the saved file if processing failed
                if os.path.exists(file_path):
                    os.remove(file_path)
                continue
                
        except Exception as e:
            error_msg = f"Unexpected error with {file.filename}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue

    # Prepare response
    response_data = {
        "message": f"Processing completed. {len(added_books)} books added successfully.",
        "books": added_books
    }
    
    if errors:
        response_data["errors"] = errors
        response_data["message"] += f" {len(errors)} files had errors."
    
    # Return appropriate status code
    if not added_books and errors:
        # All files failed
        raise HTTPException(status_code=400, detail=response_data)
    
    return JSONResponse(content=response_data)

@app.get("/books/")
async def list_books():
    try:
        return {"books": list(retriever_map.keys())}
    except Exception as e:
        logger.error(f"Error listing books: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list books")

# Updated streaming endpoint
@app.post("/multi_query_stream/")
async def query_multiple_books_stream(books: List[str] = Body(...), query: str = Body(...)):
    """
    Streaming endpoint for multi-book queries
    """
    return StreamingResponse(
        generate_streaming_response(books, query),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

# Keep the original endpoint for backward compatibility
@app.post("/multi_query/")
async def query_multiple_books(books: List[str] = Body(...), query: str = Body(...)):
    try:
        combined_docs = []

        for book in books:
            retriever = retriever_map.get(book)
            if retriever:
                try:
                    docs = retriever.invoke(query)
                    combined_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Error querying book {book}: {str(e)}")
                    continue

        # If no relevant documents are found after searching, return "I don't know"
        if not combined_docs:
            return {"answer": "I don't know."}

        # Only call the LLM chain if we have context
        response = llm_chain.invoke({"input": query, "context": combined_docs})
        return {"answer": response}
        
    except Exception as e:
        logger.error(f"Error in multi_query: {str(e)}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.delete("/delete_book/")
async def delete_book(book: str = Form(...)):
    try:
        # Remove from retriever map
        if book in retriever_map:
            del retriever_map[book]
        
        # Remove vector store directory
        book_path = os.path.join(VECTORSTORE_DIR, book)
        if os.path.exists(book_path):
            shutil.rmtree(book_path, ignore_errors=True)
            logger.info(f"Deleted vector store directory: {book_path}")
        
        # Get metadata to find temp file
        books_metadata = load_books_metadata()
        if book in books_metadata:
            temp_file_path = books_metadata[book].get("temp_file_path")
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.info(f"Deleted temp file: {temp_file_path}")
        
        # Remove from metadata
        remove_book_metadata(book)
        
        logger.info(f"Successfully deleted book: {book}")
        return {"message": f"{book} deleted successfully."}
        
    except Exception as e:
        logger.error(f"Error deleting book {book}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete book")

@app.get("/health")
async def health_check():
    return {
        "Status": "Okay Api is Running Up Successfully",
        "books_loaded": len(retriever_map),
        "vectorstore_dir": VECTORSTORE_DIR,
        "temp_dir": TEMP_DIR
    }
