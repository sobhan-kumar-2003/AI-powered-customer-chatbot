import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Document Chatbot",
    page_icon="🧠",
    layout="wide", # Use wide layout
)

# Load environment variables
load_dotenv()

# --- Constants ---
# Use an absolute path for Render's persistent disk
DB_BASE_PATH = "/app/db_v2" if os.getenv("RENDER") else "db_v2"

# --- Model and Cache Initialization ---
if 'rag_chains' not in st.session_state:
    st.session_state.rag_chains = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the router LLM
router_llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- Functions ---
@st.cache_data
def get_available_dbs():
    """Get a list of available knowledge bases."""
    try:
        return [d for d in os.listdir(DB_BASE_PATH) if os.path.isdir(os.path.join(DB_BASE_PATH, d))]
    except FileNotFoundError:
        return []

def route_query(query, available_dbs):
    """Uses an LLM to route the user's query to the most relevant knowledge base."""
    if not available_dbs:
        return None
    prompt = f"""You are an expert at routing user questions to the correct document knowledge base.
Based on the user's query, which of the following documents is the most relevant?

Query: '{query}'

Available Documents:
- {"\n- ".join(available_dbs)}

Respond with only the single, most relevant document name from the list. Do not add any other text."""
    try:
        response = router_llm.invoke(prompt)
        cleaned_response = response.strip()
        if cleaned_response in available_dbs:
            return cleaned_response
        else: return None
    except Exception as e:
        st.error(f"Router Error: {e}")
        return None

def load_rag_chain(db_name):
    """Loads a RAG chain for a specific database, with caching."""
    if db_name in st.session_state.rag_chains: return st.session_state.rag_chains[db_name]
    persist_directory = os.path.join(DB_BASE_PATH, db_name)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 3}))
        st.session_state.rag_chains[db_name] = qa_chain
        return qa_chain
    except Exception as e:
        st.error(f"Error loading RAG chain for '{db_name}': {e}")
        return None

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    available_dbs = get_available_dbs()
    if not available_dbs:
        st.warning("No knowledge bases found.")
    else:
        st.markdown("**Available Knowledge Bases:**")
        for db in available_dbs:
            st.markdown(f"- `{db}`")
    
    st.divider()
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
st.title("🧠 Intelligent Document Chatbot")
st.caption("Ask a question, and I'll find the right document to answer it.")

# Create a container for the chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input at the bottom
if prompt := st.chat_input("Ask about Hyundai, Mercedes, or Dr. Jekyll..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message immediately
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate and display bot response
    with st.spinner("Routing your question..."):
        selected_db = route_query(prompt, available_dbs)
    
    if not selected_db:
        response = "I couldn't determine which document is relevant. Please try rephrasing."
        with chat_container:
            with st.chat_message("assistant"):
                st.warning(response)
    else:
        with chat_container:
            st.info(f"Searching in: `{selected_db}`")
        with st.spinner("Searching for the answer..."):
            rag_chain = load_rag_chain(selected_db)
            if rag_chain:
                try:
                    rag_response = rag_chain({"query": prompt})
                    response = rag_response["result"]
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(response)
                except Exception as e:
                    response = f"An error occurred: {e}"
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.error(response)
            else:
                response = "Could not load the knowledge base."
                with chat_container:
                    with st.chat_message("assistant"):
                        st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Rerun to ensure the chat container is fully updated
    st.rerun()