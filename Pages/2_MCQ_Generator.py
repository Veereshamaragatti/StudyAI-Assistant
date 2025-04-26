import os
import streamlit as st
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def show_mcq_generator():
    st.header("Practice Papers - MCQ Generator")
    st.write("Generate MCQs from your documents for practice.")
# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Chat with PDF using FAISS", page_icon="📄")

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# 🌱 Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 🔍 Load FAISS vector store
def load_knowledge_base(faiss_index_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# 🤖 Create Gemini QA Chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say "answer is not available in the context".
    Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# 🧠 Answer user question
def get_answer_from_knowledge_base(user_question, faiss_index_path):
    new_db = load_knowledge_base(faiss_index_path)
    docs = new_db.similarity_search(user_question)

    st.subheader("🔍 Retrieved Context:")
    for i, doc in enumerate(docs[:3]):
        st.text(f"--- Document {i+1} ---")
        st.write(doc.page_content[:500])

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# 📝 Generate MCQs with improved formatting
def generate_mcqs_from_context(faiss_index_path, user_question="Generate 10 MCQs"):
    new_db = load_knowledge_base(faiss_index_path)
    docs = new_db.similarity_search(user_question)

    # Improved MCQ prompt template with clearer formatting instructions
    prompt_template = """
    From the following context, generate 10 important multiple-choice questions (MCQs).
    Each question should have 4 options labeled (A), (B), (C), and (D), and specify the correct answer.

    Format each MCQ like this:
    ### Question X:
    [Question text]
    
    - (A) [Option A]
    - (B) [Option B]
    - (C) [Option C]
    - (D) [Option D]
    
    **Correct Answer: ([Letter])**

    Make sure each question is numbered and properly separated with clear spacing.
    Ensure each option appears on a new line with proper indentation.

    Context:
    {context}

    MCQs:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"]

# Function to display MCQs in a structured format
def display_mcqs(mcqs_text):
    st.markdown("### 🧪 Multiple Choice Questions:")
    
    # Use st.markdown with custom styling to ensure proper alignment
    st.markdown(mcqs_text)
    
    # Add a CSS hack to improve spacing and alignment
    st.markdown("""
    <style>
    .stMarkdown ul {
        padding-left: 25px;
    }
    .stMarkdown ol {
        padding-left: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# 🎯 Main Streamlit App
def main():
    st.header("📄 Chat with your PDF Documents using FAISS + Gemini")

    with st.sidebar:
        st.title("📂 Menu")
        folder_path = st.text_input(
            "Enter the path to your knowledge base folder",
            value="E:/Github/Complete-Langchain-Tutorials/chatmultipledocuments/Knowledge_Base"
        )
        faiss_index_path = os.path.join(folder_path, "faiss_index")

        if os.path.exists(faiss_index_path):
            st.success("✅ FAISS index found!")
        else:
            st.warning("⚠️ FAISS index not found. Please ensure the index is created.")

    user_question = st.text_input("💬 Ask a question from your PDFs")

    col1, col2 = st.columns(2)

    if col1.button("💡 Get Answer"):
        if os.path.exists(faiss_index_path) and user_question:
            with st.spinner("Generating answer..."):
                try:
                    answer = get_answer_from_knowledge_base(user_question, faiss_index_path)
                    st.markdown("### 📘 Answer:")
                    st.markdown(answer)
                except Exception as e:
                    st.error("❌ Error while generating answer.")
                    st.exception(e)
        else:
            st.warning("🚫 Please enter a question and ensure FAISS index exists.")

    if col2.button("📝 Generate MCQs"):
        if os.path.exists(faiss_index_path):
            with st.spinner("Generating MCQs..."):
                try:
                    mcqs = generate_mcqs_from_context(faiss_index_path, user_question)
                    display_mcqs(mcqs)
                except Exception as e:
                    st.error("❌ Error while generating MCQs.")
                    st.exception(e)

if __name__ == "__main__":
    main()