# StudyAI Assistant

## Overview
StudyAI Assistant is an intelligent study companion application that helps users better understand and learn from documents. It combines document processing, AI-powered summarization, question answering, and more to create an enhanced learning experience.

## Features
- **Document Chat**: Chat with multiple PDF documents using AI
- **Summarization**: Generate concise summaries of uploaded documents
- **Question & Answer**: Ask questions about your documents with step-by-step reasoning
- **Smart Follow-ups**: Get suggested follow-up questions based on your queries
- **Multiple Document Support**: Process and analyze multiple documents at once

## Technologies Used
- Streamlit for the web interface
- LangChain for document processing and chains
- Google Generative AI (Gemini models) for AI capabilities
- FAISS and ChromaDB for vector storage and retrieval
- PyPDF2 for PDF processing

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
   ```
   git clone <your-repository-url>
   cd chatmultipledocuments
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv new_env
   # On Windows
   new_env\Scripts\activate
   # On Unix or MacOS
   source new_env/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google Generative AI API key:
   - Create a `.env` file in the root directory
   - Add your API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage
1. Start the application:
   ```
   streamlit run app.py
   ```

2. Upload your PDF documents through the UI
3. Use the various features:
   - Summarize documents
   - Chat with your documents
   - Generate MCQs (if available)
   - Search and learn from your knowledge base

## Project Structure
- `app.py`: Main Streamlit application entry point
- `Chat_UI.py`: Chat interface with step-by-step reasoning
- `1_summarizer.py`: Document summarization functionality
- `requirements.txt`: Required Python packages
- `Knowledge_Base/`: Directory for storing sample documents
- `Pages/`: Contains different functionality pages
- `faiss_index/`: Vector database indexes

## Acknowledgements
- Thanks to LangChain for document processing capabilities
- Thanks to Google for the Generative AI API
