# PDF AI Chatbot Setup Guide

This guide will help you set up and run the PDF AI Chatbot application.

## Prerequisites

1. Python 3.8+ installed on your system
2. Ollama installed and running (https://ollama.com)
3. A compatible LLM model pulled into Ollama (e.g., llama3)

## Installation Steps

1. **Clone or download the application files**
   - Save the Python script as `app.py`
   - Save the requirements file as `requirements.txt`

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama**
   - Make sure Ollama is running on your system (default: http://localhost:11434)
   - Pull a compatible model if you haven't already:
     ```bash
     ollama pull llama3
     ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your web browser and go to http://localhost:8501

## Usage

1. Upload a PDF document using the file uploader in the sidebar
2. Click "Process PDF" to extract and process the content
3. Ask questions in the chat interface
4. The AI will retrieve relevant context from the PDF and generate answers

## Customization

- You can change the default model in the settings sidebar
- Adjust the Ollama URL if running on a different machine or port
- The app automatically cleans up temporary files when closed

## Troubleshooting

- If you get connection errors, ensure Ollama is running
- For memory issues with large PDFs, try adjusting the chunk size in the code
- If text extraction fails, the PDF might be image-based and require OCR (not included in this version)
