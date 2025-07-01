
import os
import shutil
import re
import unicodedata
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, Form, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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
import logging

# === Config ===
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = "vectorstores"
TEMP_DIR = "temp_pdfs"
MAX_FILE_SIZE_MB = 500
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

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
                vectorstore = save_faiss_index(chunks, title)
                retriever = create_ensemble_retriever(chunks, vectorstore)
                retriever_map[title] = retriever
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
        book_path = os.path.join(VECTORSTORE_DIR, book)
        if book in retriever_map and os.path.exists(book_path):
            del retriever_map[book]
            shutil.rmtree(book_path, ignore_errors=True)
            return {"message": f"{book} deleted successfully."}
        raise HTTPException(status_code=404, detail="Book not found.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting book {book}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete book")

@app.get("/health")
async def health_check():
    return {"Status": "Okay Api is Running Up Successfully"}