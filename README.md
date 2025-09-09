# Customer Support RAG Chatbot

## Overview

This project is a complete Retrieval-Augmented Generation (RAG) chatbot designed for customer support. It uses a Large Language Model (LLM) to answer questions, but its knowledge is augmented by a private collection of documents. This allows it to provide specific, accurate answers based on your own data (e.g., product manuals, FAQs, policy documents) rather than relying on the LLM's general knowledge.

The application has two main parts:
1.  **Ingestion Script (`ingest.py`):** A tool to process your local documents into a searchable knowledge base.
2.  **Chatbot Application (`app.py`):** A web-based interface that allows users to ask questions and get answers from the AI, which uses the knowledge base for context.

---

## How the Code Works

The project operates in two main phases:

### 1. Ingestion (Building the Knowledge Base)

This is handled by the `ingest.py` script.
- **Load Documents:** It scans the `documents/` directory for PDF files.
- **Chunking:** It splits the large documents into smaller, more manageable chunks of text. This is crucial because it's more efficient to find and feed small, relevant pieces of text to the AI than entire documents.
- **Embedding:** Each text chunk is converted into a numerical representation called a "vector embedding" using an AI model. These vectors capture the semantic meaning of the text.
- **Store in Vector Database:** The chunks and their corresponding embeddings are stored in a local vector database created using **ChromaDB**. This database is saved in the `db/` folder.

This process only needs to be run when you add or update your documents.

### 2. Chatting (Answering a Question)

This is handled by the `app.py` web application.
- **User Interface:** A simple web page (using Flask, HTML, and JavaScript) provides the chat window.
- **Receive Question:** When a user types a question and hits send, it's sent to the Flask backend.
- **Retrieve Context:** The backend takes the user's question, creates an embedding for it, and uses that to search the ChromaDB vector database. It finds the most relevant text chunks from your documents (similarity search).
- **Augment and Generate:** The original question and the retrieved text chunks are combined into a detailed prompt. This "augmented" prompt is then sent to the main LLM.
- **Return Answer:** The LLM generates a response based on the context provided. This response is sent back to the user in the chat window.

---

## AI Models Used

This project uses two different models from the Google Generative AI family:

1.  **`models/text-embedding-004`**: This is the **Embedding Model**. Its job is to convert pieces of text (both from your documents and the user's questions) into vector embeddings for similarity comparison.
2.  **`gemini-1.5-flash`**: This is the **Generative Model** (LLM). It's responsible for understanding the user's question and the retrieved context, and for generating the final, human-like answer.

---

## Project Structure

```
.
├── app.py                  # The main Flask web application and RAG logic.
├── ingest.py               # Script to process documents and build the vector DB.
├── requirements.txt        # A list of all the Python libraries needed.
├── .env                    # Configuration file to securely store your API key.
├── documents/              # Folder where you place your PDF knowledge files.
├── templates/
│   └── index.html          # The HTML for the chat interface.
└── static/
    ├── style.css           # CSS for styling the interface.
    └── script.js           # JavaScript to handle sending/receiving messages.
```

---

## Core Libraries and Skills

### Libraries (Tools)

- **`Flask`**: A web framework used to create the backend server and API for the chat application.
- **`LangChain`**: A powerful framework that simplifies the process of building AI applications. We use it to manage document loading (`PyPDFDirectoryLoader`), text splitting (`RecursiveCharacterTextSplitter`), and creating the RAG chain (`RetrievalQA`).
- **`google-generativeai` / `langchain-google-genai`**: Python libraries for interacting with the Google Generative AI models (Gemini and Embeddings).
- **`ChromaDB`**: A vector database that runs locally. It stores the embeddings and allows for very fast and efficient similarity searches.
- **`pypdf`**: A library used by LangChain to read and extract text from PDF files.
- **`python-dotenv`**: A utility to manage environment variables, allowing us to keep the `GOOGLE_API_KEY` out of the main source code.
- **`cryptography`**: A required dependency for `pypdf` to handle potentially encrypted PDF files.

### Skills Demonstrated

- **Python Programming**: The entire backend is written in Python.
- **AI/LLM Integration**: Connecting to and using third-party AI services (Google AI).
- **Retrieval-Augmented Generation (RAG)**: Implementing the full RAG pipeline to ground an LLM in custom data.
- **Vector Databases**: Using ChromaDB to store and query vector embeddings.
- **Web Development**: Building a functional frontend (HTML/CSS/JS) and backend (Flask) for a complete user experience.
- **Dependency Management & Debugging**: Using `requirements.txt` and systematically debugging issues related to dependencies, API keys, and model compatibility.
