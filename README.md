# RAG-Enabled Chat Application

A web-based application that implements three distinct chat modes using Retrieval-Augmented Generation (RAG) technology.

## Features

### 1. Direct AI Chat Mode
- Integrates with Ollama for local model hosting
- Allows users to select from available downloaded models
- Enables real-time conversation without document context

### 2. User PDF Chat Mode
- Implements PDF upload functionality with progress indicator
- Processes uploaded PDFs using a lightweight embedding model (BAAI/bge-small-en)
- Creates and maintains vector stores for uploaded documents
- Enables semantic search and context-aware responses

### 3. Pre-loaded PDF Chat Mode
- Uses a dedicated folder structure for pre-loaded PDFs
- Automatically indexes and embeds documents from the specified folder
- Allows users to chat with these documents

## Technical Implementation

- **Backend**: FastAPI
- **Vector Storage**: FAISS
- **Embedding Model**: BAAI/bge-small-en
- **LLM Integration**: Ollama
- **PDF Processing**: PyPDF

## Prerequisites

1. Python 3.8 or higher
2. [Ollama](https://ollama.ai/) installed and running with at least one model
3. Sufficient disk space for vector stores

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd bold_rag
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create the necessary directories (if they don't exist):
```bash
mkdir -p data/uploaded_pdfs data/preloaded_pdfs data/vector_stores
```

## Usage

1. Start the application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Choose a chat mode:
   - **Direct AI Chat**: Select a model and start chatting
   - **User PDF Chat**: Upload a PDF, wait for processing to complete, then chat
   - **Pre-loaded PDF Chat**: Place PDFs in the `data/preloaded_pdfs` directory, click "Load & Index", then chat

## Adding Pre-loaded PDFs

1. Place your PDF files in the `data/preloaded_pdfs` directory
2. In the application, select "Pre-loaded PDF Chat" mode
3. Click "Load & Index" to process the PDFs
4. Once processing is complete, you can start chatting with the documents

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running and accessible
- **PDF Processing Errors**: Check that PDFs are not corrupted and are readable
- **Memory Issues**: For very large PDFs, consider increasing your system's available memory

## License

[MIT License](LICENSE)
