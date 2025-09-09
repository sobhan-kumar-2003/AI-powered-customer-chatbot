from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")

app = Flask(__name__)

# --- RAG Setup ---
DB_PATH = "/data/db"
DATA_PATH = "documents/"

def load_rag_chain():
    """Loads the RAG chain for answering questions."""
    try:
        # Load embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # Load the vector store
        db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

        # Load the LLM
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        print(f"Error loading RAG chain: {e}")
        import traceback
        traceback.print_exc()
        # This can happen if the db doesn't exist yet. 
        # We'll return None and handle it in the chat endpoint.
        return None

rag_chain = load_rag_chain()
# --- End RAG Setup ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    
    if not rag_chain:
        # This happens if the DB is not yet created.
        bot_reply = "The knowledge base is not yet available. Please run the ingestion script first."
    elif not user_message:
        bot_reply = "Please ask a question."
    else:
        try:
            # Use the RAG chain to get a response
            response = rag_chain({"query": user_message})
            bot_reply = response["result"]
        except Exception as e:
            bot_reply = f"An error occurred: {e}"

    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
