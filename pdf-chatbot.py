import os
import tempfile
import streamlit as st
import requests
import json
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None

# Configuration variables
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama address
MODEL_NAME = "llama3.2:latest"  # Default model - using Llama 3.2

# Streamlit UI setup
st.title("PDF AI Chatbot")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    selected_model = st.text_input("Model Name", MODEL_NAME)
    ollama_url = st.text_input("Ollama URL", OLLAMA_BASE_URL)
    
    # PDF upload section
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Create temp file to store the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write uploaded file content to temp file
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
            st.session_state.temp_file_path = temp_file_path
        
        # Extract text from PDF and process it
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Extract text from PDF
                pdf_reader = PyPDF2.PdfReader(st.session_state.temp_file_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                
                # Check if we got any text
                if text.strip():
                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    st.session_state.document_chunks = chunks
                    
                    # Create vector store for semantic search
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    vector_store = FAISS.from_texts(chunks, embeddings)
                    st.session_state.vector_store = vector_store
                    
                    st.success(f"PDF processed successfully! Extracted {len(chunks)} chunks.")
                    # Clear previous conversation when loading a new PDF
                    st.session_state.messages = []
                else:
                    st.error("Could not extract text from the PDF. The file might be scanned or image-based.")

# Function to query Ollama
def query_ollama(prompt, model_name=selected_model, context=None):
    url = f"{ollama_url}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    if context:
        # Add relevant document context to the prompt
        context_prompt = "Based on the following information:\n\n"
        for ctx in context:
            context_prompt += f"{ctx}\n\n"
        context_prompt += f"Question: {prompt}\nPlease provide an accurate answer:"
        payload["prompt"] = context_prompt
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {str(e)}"

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user question
if prompt := st.chat_input("Ask a question about your PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        if st.session_state.vector_store is not None:
            # Get relevant context from the vector store
            with st.spinner("Searching document..."):
                # Create embeddings for the question
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                query_embedding = embeddings.embed_query(prompt)
                
                # Search for similar chunks
                similar_docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                relevant_contexts = [doc.page_content for doc in similar_docs]
                
                # Query LLM with context
                with st.spinner("Generating answer..."):
                    response = query_ollama(prompt, context=relevant_contexts)
                    response_placeholder.markdown(response)
        else:
            response = "Please upload and process a PDF document first."
            response_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Clean up temp file when the app is closed
def cleanup():
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        try:
            os.unlink(st.session_state.temp_file_path)
        except:
            pass

# Register the cleanup function to run when the app is closed
import atexit
atexit.register(cleanup)