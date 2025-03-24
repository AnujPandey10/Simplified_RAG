from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
import uuid
import time
from typing import List, Optional, AsyncIterator, Dict, Any
import json
import asyncio

# Import custom modules
from app.rag_engine import (
    get_ollama_models,
    chat_with_ollama,
    stream_chat_with_ollama,
    process_pdf,
    chat_with_pdf,
    stream_chat_with_pdf,
    load_preloaded_pdfs,
    get_preloaded_pdf_list
)

app = FastAPI(title="RAG-Enabled Chat Application")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Set up data directories
UPLOAD_DIR = "data/uploaded_pdfs"
PRELOADED_DIR = "data/preloaded_pdfs"
VECTOR_STORE_DIR = "data/vector_stores"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PRELOADED_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Store processing status
processing_tasks = {}

# Store chat histories
chat_histories = {}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/ollama-models")
async def get_models():
    try:
        models = await get_ollama_models()
        if not models:
            # If no models found, return a more informative response
            return {
                "models": [],
                "message": "No models found. Please ensure Ollama service is running and models are installed."
            }
        return {"models": models}
    except Exception as e:
        print(f"Error in get_models endpoint: {str(e)}")  # Debug print
        return HTTPException(
            status_code=500,
            detail=f"Failed to get models: {str(e)}. Please ensure Ollama service is running."
        )


@app.post("/api/chat-direct")
async def chat_direct(model: str = Form(...), message: str = Form(...), session_id: str = Form(None)):
    try:
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
        
        # Get response from the model
        response = await chat_with_ollama(model, message, history)
        
        # Update chat history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return {"response": response, "session_id": session_id}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/chat-direct")
async def chat_direct(
    model: str = Form(...),
    message: str = Form(...),
    session_id: str = Form(None)
):
    try:
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        # Get response from Ollama
        response = await chat_with_ollama(model, message, history)
        
        # Add response to history
        history.append({"role": "assistant", "content": response})
        
        # Return the response
        return {"response": response, "session_id": session_id}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/stream-chat-direct")
async def stream_chat_direct(
    model: str = Form(...),
    message: str = Form(...),
    session_id: str = Form(None)
):
    try:
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        # Get streaming response generator
        response_generator = await stream_chat_with_ollama(model, message, history)
        
        # Create a streaming response
        async def response_stream():
            full_response = ""
            async for chunk in response_generator:
                full_response += chunk
                # Send the chunk with proper SSE format
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
            
            # After streaming is complete, add the full response to history
            history.append({"role": "assistant", "content": full_response})
            # Send an end marker to signal the complete message
            yield f"data: {json.dumps({'complete': True, 'full_response': full_response, 'session_id': session_id})}\n\n"
        
        # Return a streaming response
        return StreamingResponse(
            content=response_stream(),
            media_type="text/event-stream"
        )
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")


@app.post("/api/upload-pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_model: str = Form("BAAI/bge-small-en")
):
    try:
        # Generate a unique ID for this upload
        task_id = str(uuid.uuid4())
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize the processing status
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "file_name": file.filename,
            "vector_store_path": None
        }
        
        # Process the PDF in the background
        background_tasks.add_task(
            process_pdf_background,
            task_id,
            file_path,
            embedding_model
        )
        
        return {"task_id": task_id, "message": "PDF upload started"}
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_pdf_background(task_id: str, file_path: str, embedding_model: str):
    try:
        # Update the vector store path
        vector_store_path = os.path.join(VECTOR_STORE_DIR, f"{os.path.basename(file_path)}_store")
        processing_tasks[task_id]["vector_store_path"] = vector_store_path
        
        # Process the PDF
        await process_pdf(
            file_path, 
            vector_store_path, 
            embedding_model,
            progress_callback=lambda progress: update_progress(task_id, progress)
        )
        
        # Update status to completed
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["progress"] = 100
    
    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


def update_progress(task_id: str, progress: float):
    if task_id in processing_tasks:
        processing_tasks[task_id]["progress"] = progress


@app.get("/api/pdf-processing-status/{task_id}")
async def get_processing_status(task_id: str):
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return processing_tasks[task_id]


@app.post("/api/chat-pdf")
async def chat_with_uploaded_pdf(
    task_id: str = Form(...),
    message: str = Form(...),
    model: str = Form(...),
    session_id: str = Form(None)
):
    try:
        if task_id not in processing_tasks:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        if processing_tasks[task_id]["status"] != "completed":
            raise HTTPException(status_code=400, detail="PDF is still processing")
        
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
        
        vector_store_path = processing_tasks[task_id]["vector_store_path"]
        response = await chat_with_pdf(vector_store_path, message, model, history)
        
        # Update chat history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return {"response": response, "session_id": session_id}
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/stream-chat-pdf")
async def stream_chat_with_uploaded_pdf(
    task_id: str = Form(...),
    message: str = Form(...),
    model: str = Form(...),
    session_id: str = Form(None)
):
    try:
        if task_id not in processing_tasks:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        if processing_tasks[task_id]["status"] != "completed":
            raise HTTPException(status_code=400, detail="PDF is still processing")
        
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        vector_store_path = processing_tasks[task_id]["vector_store_path"]
        response_generator = await stream_chat_with_pdf(vector_store_path, message, model, history)
        
        # Create a streaming response
        async def response_stream():
            full_response = ""
            async for chunk in response_generator:
                full_response += chunk
                # Send the chunk with proper SSE format
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
            
            # After streaming is complete, add the full response to history
            history.append({"role": "assistant", "content": full_response})
            # Send an end marker to signal the complete message
            yield f"data: {json.dumps({'complete': True, 'full_response': full_response, 'session_id': session_id})}\n\n"
        
        # Return a streaming response
        return StreamingResponse(
            content=response_stream(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")


@app.get("/api/preloaded-pdfs")
async def get_preloaded_pdfs():
    try:
        pdfs = await get_preloaded_pdf_list(PRELOADED_DIR)
        return {"pdfs": pdfs}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Failed to get preloaded PDFs: {str(e)}")


@app.post("/api/load-preloaded-pdfs")
async def load_preloaded_pdf_endpoint(background_tasks: BackgroundTasks):
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize the processing status
        processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "file_name": "preloaded_pdfs",
            "vector_store_path": os.path.join(VECTOR_STORE_DIR, "preloaded_store")
        }
        
        # Process preloaded PDFs in the background
        background_tasks.add_task(
            load_preloaded_pdfs_background,
            task_id,
            PRELOADED_DIR,
            VECTOR_STORE_DIR
        )
        
        return {"task_id": task_id, "message": "Preloaded PDF indexing started"}
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Loading failed: {str(e)}")


async def load_preloaded_pdfs_background(task_id: str, preloaded_dir: str, vector_store_dir: str):
    try:
        vector_store_path = os.path.join(vector_store_dir, "preloaded_store")
        
        # Load and process preloaded PDFs
        await load_preloaded_pdfs(
            preloaded_dir,
            vector_store_path,
            progress_callback=lambda progress: update_progress(task_id, progress)
        )
        
        # Update status to completed
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["progress"] = 100
    
    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


@app.post("/api/chat-preloaded")
async def chat_with_preloaded(
    message: str = Form(...),
    model: str = Form(...),
    session_id: str = Form(None)
):
    try:
        vector_store_path = os.path.join(VECTOR_STORE_DIR, "preloaded_store")
        
        if not os.path.exists(vector_store_path):
            raise HTTPException(status_code=400, detail="Preloaded PDFs not indexed yet")
        
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
            
        response = await chat_with_pdf(vector_store_path, message, model, history)
        
        # Update chat history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return {"response": response, "session_id": session_id}
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/stream-chat-preloaded")
async def stream_chat_with_preloaded(
    message: str = Form(...),
    model: str = Form(...),
    session_id: str = Form(None)
):
    try:
        vector_store_path = os.path.join(VECTOR_STORE_DIR, "preloaded_store")
        
        if not os.path.exists(vector_store_path):
            raise HTTPException(status_code=400, detail="Preloaded PDFs not indexed yet")
        
        # Create a new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            chat_histories[session_id] = []
        elif session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Get the chat history
        history = chat_histories[session_id]
        
        # Add user message to history
        history.append({"role": "user", "content": message})
        
        response_generator = await stream_chat_with_pdf(vector_store_path, message, model, history)
        
        # Create a streaming response
        async def response_stream():
            full_response = ""
            async for chunk in response_generator:
                full_response += chunk
                # Send the chunk with proper SSE format
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
            
            # After streaming is complete, add the full response to history
            history.append({"role": "assistant", "content": full_response})
            # Send an end marker to signal the complete message
            yield f"data: {json.dumps({'complete': True, 'full_response': full_response, 'session_id': session_id})}\n\n"
        
        # Return a streaming response
        return StreamingResponse(
            content=response_stream(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
