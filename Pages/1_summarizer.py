import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModelForSeq2SeqLM
import torch

def show_summarizer():
    st.header("Study Material - Summarizer")
    st.write("Upload your documents to get AI-powered explanations.")
# Load model and tokenizer from local folder
@st.cache_resource
def load_model():
    # Load the base model and tokenizer
    base_model = AutoModelForSeq2SeqLM.from_pretrained("flan_t5_lora_summarization_local")
    tokenizer = AutoTokenizer.from_pretrained("flan_t5_lora_summarization_local")
    model = base_model  # Directly use the base model for summarization
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("📄 PDF Summarizer with Flan-T5 (Arxiv Summarization)")

uploaded_file = st.file_uploader("Upload a PDF file (Max 30MB)", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split long text into smaller chunks (adjusted for large PDFs)
def chunk_text(text, max_tokens=800):
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        # Combine sentences until max length is reached
        if len(current) + len(sentence) < max_tokens:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# Summarization function
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main flow
if uploaded_file:
    with st.spinner("📖 Reading and processing your PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("📃 Extracted Text")
    st.text_area("PDF Content", value=text, height=300)

    if st.button("✨ Summarize"):
        with st.spinner("🧠 Generating Summary..."):
            chunks = chunk_text(text)
            summaries = [summarize_text(chunk) for chunk in chunks]
            final_summary = "\n".join([f"- {s.strip().rstrip('.')}" for s in summaries if s.strip()])
        
        st.subheader("📝 Summary")
        st.markdown(f"🔹 **Key Points:**\n{final_summary}")
