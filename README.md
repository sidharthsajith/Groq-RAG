# Groq RAG Application

A real-time Retrieval-Augmented Generation (RAG) application built with Streamlit and Groq. This application allows users to upload documents (PDF, DOCX, PPTX, TXT) and add website URLs to create a knowledge base, which can then be queried using natural language.

## Features

- Support for multiple document formats (PDF, DOCX, PPTX, TXT)
- Web crawling capability for adding online content
- Real-time document processing with queuing system
- Vector similarity search using FAISS
- Integration with Groq's LLM API
- Persistent storage of knowledge base
- Interactive web interface built with Streamlit

## Setup

### Prerequisites

- Python 3.8 or higher
- A Groq API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sidharthsajith/Groq-RAG.git
cd groq-rag
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Use the sidebar to:
   - Upload documents (PDF, DOCX, PPTX, TXT)
   - Add URLs for web crawling
   - Process documents
   - Clear the knowledge base if needed

4. Ask questions in the main interface to query your knowledge base

## How It Works

### Document Processing Pipeline

1. **Document Upload and URL Processing**
   - Documents are uploaded through the Streamlit interface
   - URLs are crawled to extract text content
   - All content is queued for processing with a 30-second delay between documents

2. **Text Extraction**
   - PDFs: Uses `pypdf` to extract text from each page
   - DOCX: Uses `python-docx` to extract text from paragraphs
   - PPTX: Uses `python-pptx` to extract text from slides
   - TXT: Direct text extraction
   - Web pages: Uses `BeautifulSoup` to extract cleaned text content

3. **Text Processing**
   - Content is split into chunks of approximately 1000 words
   - Each chunk is converted into a vector embedding using the `all-MiniLM-L6-v2` model
   - Embeddings are normalized and added to a FAISS index

4. **Knowledge Base Management**
   - Vector index and text chunks are saved to disk
   - Can be loaded on application restart
   - Clearable through the interface

### Query Processing

1. **Question Input**
   - User enters a natural language question
   - Question is converted to a vector embedding

2. **Retrieval**
   - FAISS index searches for most similar text chunks
   - Top 3 most relevant chunks are retrieved
   - Source information is preserved

3. **Response Generation**
   - Retrieved chunks are combined into context
   - Context and question are sent to Groq's LLM
   - Response is generated and displayed with source references

### Technical Components

- **DocumentProcessor**: Main class handling all document processing and querying
- **FAISS**: High-performance similarity search
- **Sentence Transformers**: Document and query embedding
- **Groq Integration**: LLM-based response generation
- **Threading**: Background processing of documents
- **Streamlit**: Web interface and user interaction

## Architecture

The application follows a modular architecture:
- Frontend: Streamlit web interface
- Processing Layer: Document handling and embedding generation
- Storage Layer: FAISS index and pickle storage
- API Layer: Groq LLM integration

## Limitations

- Processing large documents may take significant time
- Web crawling is rate-limited and basic
- Knowledge base size is limited by available memory
- Requires stable internet connection for Groq API access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
