import os
import asyncio
from typing import List, Dict, Any, Callable, Optional, Union
import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
import glob
import pickle
import shutil


async def get_ollama_models() -> List[str]:
    """Get a list of available models from Ollama."""
    try:
        # Run ollama list command directly
        process = await asyncio.create_subprocess_exec(
            'ollama', 'list',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if stderr:
            print(f"Ollama stderr: {stderr.decode()}")
        
        # Parse the output to get model names
        output = stdout.decode()
        print("Ollama output:", output)
        
        # Skip the header line and process each line
        lines = output.strip().split('\n')[1:]
        model_names = [line.split()[0] for line in lines if line.strip()]
        
        print("Found models:", model_names)
        return model_names
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return []


async def chat_with_ollama(model_name: str, message: str) -> str:
    """Chat directly with an Ollama model."""
    try:
        # Run in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': message}]
            )
        )
        
        # Extract the response content
        return response['message']['content']
    except Exception as e:
        print(f"Error chatting with Ollama: {e}")
        raise


async def stream_chat_with_ollama(model_name: str, message: str):
    """Stream chat response from an Ollama model."""
    try:
        # Create a generator that yields response chunks
        async def response_generator():
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def generate_chunks():
                return list(chunk['response'] 
                          for chunk in ollama.generate(
                              model=model_name,
                              prompt=message,
                              stream=True
                          ) if chunk.get('response'))
            
            # Get all chunks at once through the executor
            chunks = await loop.run_in_executor(None, generate_chunks)
            
            # Yield each chunk
            for chunk in chunks:
                yield chunk
        
        return response_generator()
    except Exception as e:
        print(f"Error streaming chat with Ollama: {e}")
        raise


class SimpleVectorStore:
    """A simple vector store using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.documents = []
    
    def add_documents(self, documents):
        """Add documents to the vector store."""
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        self.documents.extend(documents)
        
        # Vectorize the texts
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def similarity_search(self, query, k=4):
        """Search for similar documents."""
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get the top k most similar documents
        indices = similarities.argsort()[-k:][::-1]
        
        return [self.documents[i] for i in indices]
    
    def save(self, path):
        """Save the vector store to disk."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'vectorstore.pkl'), 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'vectors': self.vectors,
                'documents': self.documents
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load a vector store from disk."""
        with open(os.path.join(path, 'vectorstore.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        store = cls()
        store.vectorizer = data['vectorizer']
        store.vectors = data['vectors']
        store.documents = data['documents']
        
        return store

async def process_pdf(
    pdf_path: str, 
    vector_store_path: str, 
    embedding_model_name: str = None,  # Not used in this implementation
    progress_callback: Optional[Callable[[float], None]] = None
) -> None:
    """Process a PDF and create a vector store."""
    try:
        # Make sure the directory doesn't exist or is empty
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Update progress
        if progress_callback:
            progress_callback(10)
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Update progress
        if progress_callback:
            progress_callback(30)
        
        # Create the vector store
        vector_store = SimpleVectorStore()
        
        # Update progress
        if progress_callback:
            progress_callback(50)
        
        # Add documents to the vector store
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: vector_store.add_documents(chunks)
        )
        
        # Save the vector store
        await loop.run_in_executor(
            None,
            lambda: vector_store.save(vector_store_path)
        )
        
        # Update progress
        if progress_callback:
            progress_callback(100)
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise


async def chat_with_pdf(
    vector_store_path: str,
    query: str,
    model_name: str,
    k: int = 4
) -> str:
    """Chat with a PDF using RAG."""
    try:
        # Load the vector store
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(
            None,
            lambda: SimpleVectorStore.load(vector_store_path)
        )
        
        # Search for relevant documents
        docs = vector_store.similarity_search(query, k=k)
        
        # Extract the content from the documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt with the context
        prompt = f"""Answer the following question based on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question:
{query}

Answer:"""
        
        # Get the response from Ollama
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
        )
        
        return response['message']['content']
    
    except Exception as e:
        print(f"Error chatting with PDF: {e}")
        raise


async def stream_chat_with_pdf(
    vector_store_path: str,
    query: str,
    model_name: str,
    k: int = 4
):
    """Stream chat with a PDF using RAG."""
    try:
        # Load the vector store
        loop = asyncio.get_event_loop()
        vector_store = await loop.run_in_executor(
            None,
            lambda: SimpleVectorStore.load(vector_store_path)
        )
        
        # Search for relevant documents
        docs = vector_store.similarity_search(query, k=k)
        
        # Extract the content from the documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt with the context
        prompt = f"""Answer the following question based on the provided context. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question:
{query}

Answer:"""
        
        # Create a generator that yields response chunks
        async def response_generator():
            # Run in a thread pool to avoid blocking
            def generate_chunks():
                return list(chunk['response'] 
                          for chunk in ollama.generate(
                              model=model_name,
                              prompt=prompt,
                              stream=True
                          ) if chunk.get('response'))
            
            # Get all chunks at once through the executor
            chunks = await loop.run_in_executor(None, generate_chunks)
            
            # Yield each chunk
            for chunk in chunks:
                yield chunk
        
        return response_generator()
    
    except Exception as e:
        print(f"Error streaming chat with PDF: {e}")
        raise


async def get_preloaded_pdf_list(preloaded_dir: str) -> List[str]:
    """Get a list of preloaded PDFs."""
    try:
        pdf_files = glob.glob(os.path.join(preloaded_dir, "**", "*.pdf"), recursive=True)
        return [os.path.basename(pdf) for pdf in pdf_files]
    except Exception as e:
        print(f"Error getting preloaded PDFs: {e}")
        return []


async def load_preloaded_pdfs(
    preloaded_dir: str,
    vector_store_path: str,
    embedding_model_name: str = None,  # Not used in this implementation
    progress_callback: Optional[Callable[[float], None]] = None
) -> None:
    """Load and process preloaded PDFs."""
    try:
        # Make sure the directory doesn't exist or is empty
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Get all PDF files
        pdf_files = glob.glob(os.path.join(preloaded_dir, "**", "*.pdf"), recursive=True)
        
        if not pdf_files:
            if progress_callback:
                progress_callback(100)
            return
        
        all_documents = []
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files):
            # Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            
            # Update progress
            if progress_callback:
                progress = int((i + 1) / len(pdf_files) * 50)  # First 50% for loading
                progress_callback(progress)
        
        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Update progress
        if progress_callback:
            progress_callback(60)  # 60% after splitting
        
        # Create the vector store
        vector_store = SimpleVectorStore()
        
        # Update progress
        if progress_callback:
            progress_callback(70)  # 70% after initializing vectorizer
        
        # Add documents to the vector store
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: vector_store.add_documents(chunks)
        )
        
        # Save the vector store
        await loop.run_in_executor(
            None,
            lambda: vector_store.save(vector_store_path)
        )
        
        # Update progress
        if progress_callback:
            progress_callback(100)  # 100% after creating vector store
            
    except Exception as e:
        print(f"Error loading preloaded PDFs: {e}")
        raise
