import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "documents/"
DB_PATH = "/data/db"

def main():
    print("Starting document ingestion...")
    # Check if API key is available
    if os.getenv("GOOGLE_API_KEY") is None:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    # Load documents
    print(f"Loading documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    if not documents:
        print("No documents found to ingest.")
        return
    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Create embeddings and vector store
    print("Creating embeddings and vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Create a new Chroma DB and persist it
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=DB_PATH
    )
    print(f"Successfully created and saved vector store to {DB_PATH}")

if __name__ == "__main__":
    main()
