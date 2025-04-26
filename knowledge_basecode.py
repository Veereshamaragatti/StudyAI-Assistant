import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs in a folder
def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to chunk the extracted text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS knowledge base from the text chunks
def create_knowledge_base(folder_path):
    raw_text = get_pdf_text_from_folder(folder_path)
    text_chunks = get_text_chunks(raw_text)
    
    # Generate embeddings for the text chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save the FAISS index
    faiss_index_path = os.path.join(folder_path, "faiss_index")
    vector_store.save_local(faiss_index_path)
    print(f"Knowledge base created and saved at {faiss_index_path}")

# Provide the path to your folder containing PDFs
folder_path = "E:/Github/Complete-Langchain-Tutorials/chatmultipledocuments/Knowledge_Base"

# Create the knowledge base
create_knowledge_base(folder_path)
print("Knowledge base creation process completed.")