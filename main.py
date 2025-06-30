import os
import shutil
from fastapi import FastAPI, UploadFile, Form, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# === Config ===





load_dotenv()


VECTORSTORE_DIR = "vectorstores"
TEMP_DIR = "temp_pdfs"
MAX_FILE_SIZE_MB = 500
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")
LLM = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

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

# === Helper Functions ===
def load_and_split_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    title = os.path.basename(path).replace(".pdf", "")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["book_title"] = title
    return chunks, title

def save_faiss_index(chunks, title):
    vectorstore = FAISS.from_documents(chunks, EMBEDDING_MODEL)
    save_path = os.path.join(VECTORSTORE_DIR, title)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    return vectorstore

# <<< CHANGE START: Refined the retriever >>>
def create_ensemble_retriever(chunks, vectorstore):
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
# <<< CHANGE END >>>

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

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit.")

        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        chunks, title = load_and_split_pdf(file_path)
        vectorstore = save_faiss_index(chunks, title)
        retriever = create_ensemble_retriever(chunks, vectorstore)
        retriever_map[title] = retriever
        added_books.append(title)

    return {"message": "Books uploaded and indexed.", "books": added_books}

@app.get("/books/")
async def list_books():
    return {"books": list(retriever_map.keys())}

# <<< CHANGE START: Critical logic fix in the query endpoint >>>
@app.post("/multi_query/")
async def query_multiple_books(books: List[str] = Body(...), query: str = Body(...)):
    combined_docs = []

    for book in books:
        retriever = retriever_map.get(book)
        if retriever:
            docs = retriever.invoke(query)
            combined_docs.extend(docs)

    # CRITICAL: If no relevant documents are found after searching, do not proceed.
    if not combined_docs:
        return {"answer": "I don't know."}

    # Only call the LLM chain if we have context.
    response = llm_chain.invoke({"input": query, "context": combined_docs})
    return {"answer": response}
# <<< CHANGE END >>>

@app.delete("/delete_book/")
async def delete_book(book: str = Form(...)):
    book_path = os.path.join(VECTORSTORE_DIR, book)
    if book in retriever_map and os.path.exists(book_path):
        del retriever_map[book]
        shutil.rmtree(book_path, ignore_errors=True)
        return {"message": f"{book} deleted successfully."}
    raise HTTPException(status_code=404, detail="Book not found.")


@app.get("/health")
async def health_check():
    return {"Status": "Okay Api is Running Up Successfully"}