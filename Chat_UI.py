import streamlit as st
import google.generativeai as genai
import os

# Set environment variable to fix torch.classes error with Streamlit file watcher
os.environ["STREAMLIT_WATCHDOG_IGNORE_MODULES"] = "torch.classes"

# Configure Gemini API key
genai.configure(api_key="AIzaSyCGLICJ7Usg_Q1-MV5F4nSDBE8GUAoKHM0")

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-pro-002")

# Function to get step-by-step answer + follow-up questions (CoT)
def ask_question_with_followups(user_question):
    prompt = f"""
You are a helpful assistant. When a user asks a complex question, use step-by-step reasoning to break down the answer. 
Think through the problem like a human expert would. Then suggest 2-3 relevant follow-up questions they might ask next.

Format your output exactly as:
Answer (with step-by-step reasoning):
<step-by-step answer here>

Follow-up Questions:
1. <follow-up>
2. <follow-up>
3. <follow-up>

Question: "{user_question}"
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Title
st.title("🧠 Smart Q&A with Chain-of-Thought Reasoning")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

# Input form for user question
with st.form("question_form"):
    user_question = st.text_input("Ask a question", "")
    submitted = st.form_submit_button("Ask")
    if submitted and user_question:
        st.session_state.current_question = user_question
        st.session_state.history.append(user_question)

# Fetch and cache response
current_q = st.session_state.current_question
if current_q:
    if current_q not in st.session_state.response_cache:
        response = ask_question_with_followups(current_q)
        st.session_state.response_cache[current_q] = response
    else:
        response = st.session_state.response_cache[current_q]

    # Parse response format
    if "Answer (with step-by-step reasoning):" in response and "Follow-up Questions:" in response:
        answer_part, followups_part = response.split("Follow-up Questions:")
        answer = answer_part.replace("Answer (with step-by-step reasoning):", "").strip()
        followups = followups_part.strip().split("\n")

        # Display answer
        st.markdown(f"### 🤖 Answer to: *{current_q}*")
        with st.expander("🧩 Step-by-Step Reasoning"):
            st.write(answer)

        # Display follow-up buttons
        st.markdown("### 🔁 Follow-Up Questions")
        for idx, q in enumerate(followups):
            if q.strip():
                btn_label = q.strip()
                if st.button(btn_label, key=f"followup_{idx}_{btn_label}"):
                    followup_q = btn_label[3:].strip()  # remove '1. ', etc.
                    st.session_state.current_question = followup_q
                    st.session_state.history.append(followup_q)
                    st.rerun()

# Show Q&A history
with st.expander("🕘 Show Q&A History"):
    for i, q in enumerate(st.session_state.history):
        st.markdown(f"**{i+1}.** {q}")
