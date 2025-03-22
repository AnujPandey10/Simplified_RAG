document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatModeRadios = document.querySelectorAll('input[name="chatMode"]');
    const modelSelect = document.getElementById('modelSelect');
    const pdfUploadSection = document.getElementById('pdfUploadSection');
    const preloadedPdfsSection = document.getElementById('preloadedPdfsSection');
    const pdfFileInput = document.getElementById('pdfFileInput');
    const uploadPdfBtn = document.getElementById('uploadPdfBtn');
    const uploadProgressContainer = document.getElementById('uploadProgressContainer');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadStatus = document.getElementById('uploadStatus');
    const preloadedPdfsList = document.getElementById('preloadedPdfsList');
    const loadPreloadedBtn = document.getElementById('loadPreloadedBtn');
    const preloadedProgressContainer = document.getElementById('preloadedProgressContainer');
    const preloadedProgressBar = document.getElementById('preloadedProgressBar');
    const preloadedStatus = document.getElementById('preloadedStatus');
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendMessageBtn = document.getElementById('sendMessageBtn');

    // State variables
    let currentChatMode = 'direct';
    let currentTaskId = null;
    let processingStatusInterval = null;
    let sessionId = null; // Add session ID for chat history

    // Initialize the application
    init();

    function init() {
        // Load available models
        loadOllamaModels();
        
        // Set up event listeners
        setupEventListeners();
        
        // Load preloaded PDFs list
        loadPreloadedPdfsList();
    }

    function setupEventListeners() {
        // Chat mode selection
        chatModeRadios.forEach(radio => {
            radio.addEventListener('change', handleChatModeChange);
        });
        
        // PDF upload
        uploadPdfBtn.addEventListener('click', handlePdfUpload);
        
        // Load preloaded PDFs
        loadPreloadedBtn.addEventListener('click', handleLoadPreloadedPdfs);
        
        // Send message
        sendMessageBtn.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    function handleChatModeChange(e) {
        currentChatMode = e.target.value;
        
        // Show/hide relevant sections based on selected mode
        if (currentChatMode === 'direct') {
            pdfUploadSection.style.display = 'none';
            preloadedPdfsSection.style.display = 'none';
        } else if (currentChatMode === 'pdf') {
            pdfUploadSection.style.display = 'block';
            preloadedPdfsSection.style.display = 'none';
        } else if (currentChatMode === 'preloaded') {
            pdfUploadSection.style.display = 'none';
            preloadedPdfsSection.style.display = 'block';
        }
    }

    async function loadOllamaModels() {
        try {
            const response = await fetch('/api/ollama-models');
            const data = await response.json();
            
            // Clear existing options
            modelSelect.innerHTML = '';
            
            if (data.models && data.models.length > 0) {
                // Add models to select
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
            } else {
                // No models found
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No models available';
                option.disabled = true;
                option.selected = true;
                modelSelect.appendChild(option);
                
                addSystemMessage('No Ollama models found. Please install Ollama and download at least one model.');
            }
        } catch (error) {
            console.error('Error loading models:', error);
            addSystemMessage('Failed to load Ollama models. Please check if Ollama is running.');
        }
    }

    async function loadPreloadedPdfsList() {
        try {
            const response = await fetch('/api/preloaded-pdfs');
            const data = await response.json();
            
            // Clear existing list
            preloadedPdfsList.innerHTML = '';
            
            if (data.pdfs && data.pdfs.length > 0) {
                // Add PDFs to list
                data.pdfs.forEach(pdf => {
                    const div = document.createElement('div');
                    div.className = 'pdf-item';
                    div.textContent = pdf;
                    preloadedPdfsList.appendChild(div);
                });
            } else {
                // No PDFs found
                preloadedPdfsList.textContent = 'No preloaded PDFs found. Add PDFs to the preloaded_pdfs directory.';
            }
        } catch (error) {
            console.error('Error loading preloaded PDFs:', error);
            preloadedPdfsList.textContent = 'Failed to load preloaded PDFs.';
        }
    }

    async function handlePdfUpload() {
        // Check if a file is selected
        if (!pdfFileInput.files || pdfFileInput.files.length === 0) {
            addSystemMessage('Please select a PDF file to upload.');
            return;
        }
        
        const file = pdfFileInput.files[0];
        
        // Check if it's a PDF
        if (!file.type.includes('pdf')) {
            addSystemMessage('Please select a valid PDF file.');
            return;
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('embedding_model', 'BAAI/bge-small-en');
        
        try {
            // Show progress bar
            uploadProgressContainer.style.display = 'block';
            uploadProgressBar.style.width = '0%';
            uploadProgressBar.textContent = '0%';
            uploadStatus.textContent = 'Uploading...';
            
            // Upload the file
            const response = await fetch('/api/upload-pdf', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.task_id) {
                currentTaskId = data.task_id;
                uploadStatus.textContent = 'Processing PDF...';
                
                // Start checking the processing status
                startProcessingStatusCheck(currentTaskId, 'upload');
                
                addSystemMessage(`PDF "${file.name}" uploaded and processing started.`);
            } else {
                uploadStatus.textContent = 'Upload failed.';
                uploadProgressContainer.style.display = 'none';
                addSystemMessage('Failed to upload PDF.');
            }
        } catch (error) {
            console.error('Error uploading PDF:', error);
            uploadStatus.textContent = 'Upload failed.';
            uploadProgressContainer.style.display = 'none';
            addSystemMessage('Failed to upload PDF. Please try again.');
        }
    }

    async function handleLoadPreloadedPdfs() {
        try {
            // Show progress bar
            preloadedProgressContainer.style.display = 'block';
            preloadedProgressBar.style.width = '0%';
            preloadedProgressBar.textContent = '0%';
            preloadedStatus.textContent = 'Loading...';
            
            // Start loading preloaded PDFs
            const response = await fetch('/api/load-preloaded-pdfs', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.task_id) {
                currentTaskId = data.task_id;
                preloadedStatus.textContent = 'Processing PDFs...';
                
                // Start checking the processing status
                startProcessingStatusCheck(currentTaskId, 'preloaded');
                
                addSystemMessage('Preloaded PDFs indexing started.');
            } else {
                preloadedStatus.textContent = 'Loading failed.';
                preloadedProgressContainer.style.display = 'none';
                addSystemMessage('Failed to load preloaded PDFs.');
            }
        } catch (error) {
            console.error('Error loading preloaded PDFs:', error);
            preloadedStatus.textContent = 'Loading failed.';
            preloadedProgressContainer.style.display = 'none';
            addSystemMessage('Failed to load preloaded PDFs. Please try again.');
        }
    }

    function startProcessingStatusCheck(taskId, type) {
        // Clear any existing interval
        if (processingStatusInterval) {
            clearInterval(processingStatusInterval);
        }
        
        // Set up a new interval
        processingStatusInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/pdf-processing-status/${taskId}`);
                const data = await response.json();
                
                // Update progress bar
                const progressBar = type === 'upload' ? uploadProgressBar : preloadedProgressBar;
                const statusElement = type === 'upload' ? uploadStatus : preloadedStatus;
                
                progressBar.style.width = `${data.progress}%`;
                progressBar.textContent = `${data.progress}%`;
                
                // Check if processing is complete
                if (data.status === 'completed') {
                    clearInterval(processingStatusInterval);
                    statusElement.textContent = 'Processing completed.';
                    
                    if (type === 'upload') {
                        addSystemMessage(`PDF "${data.file_name}" is now ready for chat.`);
                    } else {
                        addSystemMessage('Preloaded PDFs are now ready for chat.');
                    }
                    
                    // Hide progress bar after a delay
                    setTimeout(() => {
                        if (type === 'upload') {
                            uploadProgressContainer.style.display = 'none';
                        } else {
                            preloadedProgressContainer.style.display = 'none';
                        }
                    }, 3000);
                } else if (data.status === 'failed') {
                    clearInterval(processingStatusInterval);
                    statusElement.textContent = 'Processing failed.';
                    addSystemMessage(`Processing failed: ${data.error || 'Unknown error'}`);
                }
            } catch (error) {
                console.error('Error checking processing status:', error);
            }
        }, 1000);
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        
        if (!message) {
            return;
        }
        
        // Check if a model is selected
        const selectedModel = modelSelect.value;
        if (!selectedModel) {
            addSystemMessage('Please select a model first.');
            return;
        }
        
        // Add user message to chat
        addUserMessage(message);
        
        // Clear input
        messageInput.value = '';
        
        try {
            // Add a loading message
            const loadingId = addLoadingMessage();
            
            let endpoint;
            let body;
            
            if (currentChatMode === 'direct') {
                endpoint = '/api/stream-chat-direct';
                body = new URLSearchParams({
                    model: selectedModel,
                    message: message,
                    session_id: sessionId
                });
            } else if (currentChatMode === 'pdf' && currentTaskId) {
                endpoint = '/api/stream-chat-pdf';
                body = new URLSearchParams({
                    task_id: currentTaskId,
                    message: message,
                    model: selectedModel,
                    session_id: sessionId
                });
            } else if (currentChatMode === 'preloaded') {
                endpoint = '/api/stream-chat-preloaded';
                body = new URLSearchParams({
                    message: message,
                    model: selectedModel,
                    session_id: sessionId
                });
            } else {
                if (currentChatMode === 'pdf' && !currentTaskId) {
                    addSystemMessage('Please upload a PDF first.');
                } else {
                    addSystemMessage('Please select a valid chat mode.');
                }
                removeLoadingMessage(loadingId);
                return;
            }
            
            // Use non-streaming fetch to avoid SSE display issues
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: body
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Remove the loading message
            removeLoadingMessage(loadingId);
            
            // Extract the final response from the streaming data
            const text = await extractFinalResponseFromStream(response);
            
            // Add the bot's response to the chat
            if (text) {
                addBotMessage(text);
            } else {
                addSystemMessage('No response received from the model.');
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            addSystemMessage('Failed to get a response. Please try again.');
        }
    }
    
    // New function to extract the final response from a streaming response
    async function extractFinalResponseFromStream(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalResponse = '';
        
        try {
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, {stream: true});
            }
            
            // Process the complete buffer to extract the final response
            const events = buffer.split('\n\n');
            
            // Look for the complete event with full_response
            for (const event of events) {
                if (event.startsWith('data: ')) {
                    try {
                        const jsonData = JSON.parse(event.slice(6));
                        if (jsonData.complete && jsonData.full_response) {
                            finalResponse = jsonData.full_response;
                            break;
                        }
                    } catch (e) {
                        console.error('Error parsing event:', e);
                    }
                }
            }
            
            // If we didn't find a complete event, try to reconstruct from chunks
            if (!finalResponse) {
                let reconstructed = '';
                for (const event of events) {
                    if (event.startsWith('data: ')) {
                        try {
                            const jsonData = JSON.parse(event.slice(6));
                            if (jsonData.chunk) {
                                reconstructed += jsonData.chunk;
                            }
                        } catch (e) {
                            // Ignore parsing errors
                        }
                    }
                }
                
                if (reconstructed) {
                    finalResponse = reconstructed;
                }
            }
            
            return finalResponse;
        } catch (error) {
            console.error('Error extracting final response:', error);
            return '';
        }
    }

    function addUserMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.textContent = message;
        
        const containerDiv = document.createElement('div');
        containerDiv.style.clear = 'both';
        containerDiv.style.overflow = 'hidden';
        containerDiv.style.marginBottom = '15px';
        containerDiv.appendChild(messageDiv);
        
        chatMessages.appendChild(containerDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addBotMessage(message) {
        // Create container for the message
        const containerDiv = document.createElement('div');
        containerDiv.style.clear = 'both';
        containerDiv.style.overflow = 'hidden';
        containerDiv.style.marginBottom = '15px';
        
        // Create the message element
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        // Clean the message text - remove any data: prefixes or JSON formatting
        let cleanMessage = message;
        
        // Check if the message contains raw SSE data
        if (message.includes('data: {"chunk":')) {
            // Extract the full_response from the complete event if available
            const completeMatch = message.match(/data: \{"complete":true,"full_response":"([^"]+)"/);
            if (completeMatch && completeMatch[1]) {
                cleanMessage = completeMatch[1].replace(/\\n/g, '\n').replace(/\\\//g, '/').replace(/\\"/g, '"');
            } else {
                // If no complete event, try to reconstruct from chunks
                cleanMessage = '';
                const chunkRegex = /data: \{"chunk":"([^"]+)"[^\}]+\}/g;
                let match;
                while ((match = chunkRegex.exec(message)) !== null) {
                    cleanMessage += match[1].replace(/\\n/g, '\n').replace(/\\\//g, '/').replace(/\\"/g, '"');
                }
            }
        }
        
        // Set the clean message text
        messageDiv.textContent = cleanMessage;
        
        // Add to the container and chat
        containerDiv.appendChild(messageDiv);
        chatMessages.appendChild(containerDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addSystemMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';
        messageDiv.textContent = message;
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addLoadingMessage() {
        const id = 'loading-' + Date.now();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'system-message';
        messageDiv.id = id;
        
        const spinner = document.createElement('div');
        spinner.className = 'spinner-border text-primary';
        spinner.setAttribute('role', 'status');
        
        const span = document.createElement('span');
        span.textContent = 'Thinking...';
        
        messageDiv.appendChild(spinner);
        messageDiv.appendChild(span);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return id;
    }

    function removeLoadingMessage(id) {
        const loadingDiv = document.getElementById(id);
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }
});
