# Groq-RAG
Ask questions, get answers from your documents in realtime using Groq and FAISS
## Documentation

This document provides a comprehensive explanation of the code, including setup, workflow, and the technologies involved.

### Setup

**Prerequisites:**

* Python 3.x
* Streamlit
* PyPDF2
* sentence-transformers
* faiss
* numpy
* groq
* dotenv
* pickle

**Installation:**

1. Install the required libraries using pip:

```bash
pip install streamlit pypdf2 sentence-transformers faiss numpy groq dotenv pickle
```

**Environment Variables:**

1. Create a file named `.env` in your project directory.
2. Add the following line to the `.env` file, replacing `<YOUR_GROQ_API_KEY>` with your actual Groq API key:

```
GROQ_API_KEY=<YOUR_GROQ_API_KEY>
```

### Workflow

This code follows a three-step workflow:

1. **Data Ingestion:** Users can upload PDF documents through the Streamlit interface. The code extracts text from these PDFs and splits them into smaller chunks.
2. **Indexing:** This program uses Sentence Transformers to generate embeddings for each text chunk. FAISS, a library for efficient similarity search, is used to create an index for these embeddings.
3. **Querying and Response Generation:** Users can ask questions through the interface. The code retrieves relevant text chunks from the index based on the query embedding. Finally, it uses Groq, a natural language processing platform, to generate an answer based on the retrieved context and the user's query.

### Technology Explanation

**Streamlit:**

This program leverages Streamlit to create a user-friendly web application for document upload and querying. Streamlit allows for building interactive UIs with minimal coding effort.

**PyPDF2:**

This library is used to extract text content from uploaded PDF documents.

**Sentence Transformers:**

Sentence Transformers provide pre-trained models to generate numerical representations (embeddings) for text. These embeddings capture the semantic similarity between different pieces of text. This program uses the `all-MiniLM-L6-v2` model for this purpose.

**FAISS:**

The Facebook AI Similarity Search (FAISS) library helps create an efficient index for the text embeddings. This allows for fast retrieval of relevant text chunks when a user submits a query.

**NumPy:**

NumPy provides essential numerical computing functionalities used throughout the code, especially for working with embeddings and distance calculations.

**Groq:**

The Groq platform is used for generating the final answer to the user's query. It takes the context retrieved from the index and the user's question as input and utilizes its natural language processing capabilities to provide a comprehensive response.

**dotenv:**

This library helps manage environment variables securely, such as storing your Groq API key in a separate file.

**pickle:**

Pickle is used to store the text chunks and their corresponding source information for faster loading during subsequent use.

### Conclusion

This program demonstrates a powerful combination of technologies for building a question-answering system. It extracts knowledge from documents, indexes them efficiently, and leverages NLP models to provide informative answers to user queries. `
