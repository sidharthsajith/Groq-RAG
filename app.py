import streamlit as st
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import groq
from dotenv import load_dotenv
import tempfile
import pickle

# Load environment variables
load_dotenv()

# Initialize Groq client
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = get_embedding_model()

# Initialize FAISS index
vector_dimension = 384  # Dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(vector_dimension)

# Global storage for text chunks and their sources
text_chunks = []
chunk_sources = []

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=1000):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_text(text, source_name):
    """Process text and add to FAISS index"""
    global text_chunks, chunk_sources
    
    # Split text into chunks
    chunks = split_text(text)
    
    # Get embeddings for chunks
    embeddings = embedding_model.encode(chunks)
    
    # Add to FAISS index
    faiss.normalize_L2(embeddings)  # Normalize vectors before adding
    index.add(embeddings.astype('float32'))
    
    # Store text chunks and their sources
    text_chunks.extend(chunks)
    chunk_sources.extend([source_name] * len(chunks))
    
    return len(chunks)

def save_index(directory="./saved_index"):
    """Save FAISS index and related data"""
    os.makedirs(directory, exist_ok=True)
    
    # Save the FAISS index
    faiss.write_index(index, os.path.join(directory, "docs.index"))
    
    # Save the text chunks and sources
    with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
        pickle.dump((text_chunks, chunk_sources), f)

def load_index(directory="./saved_index"):
    """Load FAISS index and related data"""
    global index, text_chunks, chunk_sources
    
    if os.path.exists(os.path.join(directory, "docs.index")):
        index = faiss.read_index(os.path.join(directory, "docs.index"))
        
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            text_chunks, chunk_sources = pickle.load(f)
        
        return True
    return False

def query_documents(query, n_results=3):
    """Query the document collection"""
    # Get query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding.astype('float32'), n_results)
    
    # Get relevant chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    chunk_sources_info = [chunk_sources[i] for i in indices[0]]
    
    # Combine chunks into context
    context = "\n\n".join(relevant_chunks)
    
    # Use Groq to generate response
    prompt = f"""You are a helpful assistant to help users with their questions. Based on the following context, answer the question: {query}

Context:
{context}

Answer:"""
    
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="mixtral-8x7b-32768",
        temperature=0.1,
    )
    
    return completion.choices[0].message.content, list(zip(relevant_chunks, chunk_sources_info))

# Streamlit UI
st.title("Chat with your Pdfs")

# Try to load existing index
if not st.session_state.get("index_loaded"):
    if load_index():
        st.session_state.index_loaded = True
        st.success("Loaded existing knowledge base!")
    else:
        st.info("No existing knowledge base found. Please upload documents.")

# Sidebar for uploading files
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        total_chunks = 0
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                text = extract_text_from_pdf(tmp_file.name)
                chunks_added = process_text(text, uploaded_file.name)
                total_chunks += chunks_added
            os.unlink(tmp_file.name)
        
        # Save the updated index
        save_index()
        st.success(f"Added {total_chunks} chunks to knowledge base!")

    st.markdown("---")
    st.header("Or Process Directory")
    if st.button("Process PDF Directory"):
        directory_path = "./pdfs"  # Create this directory and add PDFs
        if os.path.exists(directory_path):
            total_chunks = 0
            for filename in os.listdir(directory_path):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(directory_path, filename)
                    with open(file_path, 'rb') as file:
                        text = extract_text_from_pdf(file)
                        chunks_added = process_text(text, filename)
                        total_chunks += chunks_added
            
            # Save the updated index
            save_index()
            st.success(f"Processed {total_chunks} chunks from PDF directory!")
        else:
            st.error("Directory './pdfs' not found! Please create it and add PDF files.")

# Main area for querying
st.header("Ask Questions")
query = st.text_input("Enter your question:")

if query:
    if len(text_chunks) == 0:
        st.warning("Please add some documents to the knowledge base first!")
    else:
        with st.spinner("Searching and generating response..."):
            answer, sources = query_documents(query)
            
            st.markdown("### Answer")
            st.write(answer)
            
            st.markdown("### Sources")
            for i, (chunk, source) in enumerate(sources, 1):
                with st.expander(f"Source {i} - {source}"):
                    st.write(chunk)
