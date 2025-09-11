import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

load_dotenv()

DATA_PATH = "documents/"
DB_PATH = "/app/db_v2" if os.getenv("RENDER") else "db_v2"

def main():
    print("Starting document ingestion...")
    # Check if API key is available
    if os.getenv("GOOGLE_API_KEY") is None:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    # Clear out the existing database directory
    if os.path.exists(DB_PATH):
        print(f"Clearing existing database at {DB_PATH}")
        shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH)

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {DATA_PATH}")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    for pdf_file in pdf_files:
        try:
            print(f"--- Processing {pdf_file} ---")
            file_path = os.path.join(DATA_PATH, pdf_file)
            
            # 1. Load the document
            print(f"Loading document: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                print("Could not load document.")
                continue

            # 2. Split documents into chunks
            print("Splitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks.")

            # 3. Create embeddings and vector store for the single document
            print("Creating embeddings and vector store...")
            # Sanitize filename to create a valid directory name
            db_name = os.path.splitext(pdf_file)[0]
            persist_directory = os.path.join(DB_PATH, db_name)
            
            db = Chroma.from_documents(
                chunks, embeddings, persist_directory=persist_directory
            )
            print(f"Successfully created and saved vector store to {persist_directory}")

        except Exception as e:
            print(f"Failed to process {pdf_file}. Error: {e}")

    print("\nAll documents processed.")

if __name__ == "__main__":
    main()