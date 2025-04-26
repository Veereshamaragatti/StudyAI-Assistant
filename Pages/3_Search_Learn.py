import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import os

# Set environment variable to fix torch.classes error with Streamlit
os.environ["STREAMLIT_WATCHDOG_IGNORE_MODULES"] = "torch.classes"

def show_search_learn():
    st.header("Search & Learn")
    st.write("Ask questions from your notes and get smart answers.")
# Configure Gemini API key
genai.configure(api_key="AIzaSyCGLICJ7Usg_Q1-MV5F4nSDBE8GUAoKHM0")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-pro-002")

# Initialize recognizer
recognizer = sr.Recognizer()

# Title
st.title("🧠 Smart Q&A with Chain-of-Thought Reasoning")

# Session state setup
if "history" not in st.session_state:
    st.session_state.history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

# Step-by-step reasoning function
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

# Speech recognition function
def record_and_transcribe():
    try:
        with sr.Microphone() as source:
            st.info("🎧 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            with st.spinner("🎙️ Listening... Speak now!"):
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return text
    except sr.WaitTimeoutError:
        return "⌛ Timeout. Please try speaking sooner."
    except sr.UnknownValueError:
        return "❌ Sorry, I couldn't understand your speech."
    except sr.RequestError:
        return "⚠️ Couldn't reach Google speech recognition service."

# Input form
with st.form("question_form"):
    user_question = st.text_input("Ask a question (or use voice below)", "")
    submitted = st.form_submit_button("Ask")
    if submitted and user_question:
        st.session_state.current_question = user_question
        st.session_state.history.append(user_question)

# Voice input
if st.button("🎙️ Speak a Question"):
    spoken_question = record_and_transcribe()
    if spoken_question.startswith("❌") or spoken_question.startswith("⚠️") or spoken_question.startswith("⌛"):
        st.warning(spoken_question)
    else:
        st.success(f"🗣️ You said: {spoken_question}")
        st.session_state.current_question = spoken_question
        st.session_state.history.append(spoken_question)
        st.rerun()

# Get response if question is set
current_q = st.session_state.current_question
if current_q:
    if current_q not in st.session_state.response_cache:
        response = ask_question_with_followups(current_q)
        st.session_state.response_cache[current_q] = response
    else:
        response = st.session_state.response_cache[current_q]

    if "Answer (with step-by-step reasoning):" in response and "Follow-up Questions:" in response:
        answer_part, followups_part = response.split("Follow-up Questions:")
        answer = answer_part.replace("Answer (with step-by-step reasoning):", "").strip()
        followups = followups_part.strip().split("\n")

        # Display main answer
        st.markdown(f"### 🤖 Answer to: *{current_q}*")
        with st.expander("🧩 Step-by-Step Reasoning"):
            st.write(answer)

        # Follow-up buttons
        st.markdown("### 🔁 Follow-Up Questions")
        for idx, q in enumerate(followups):
            if q.strip():
                btn_label = q.strip()
                if st.button(btn_label, key=f"followup_{idx}_{btn_label}"):
                    followup_q = btn_label[3:].strip()  # Remove number prefix
                    st.session_state.current_question = followup_q
                    st.session_state.history.append(followup_q)
                    st.rerun()

# Q&A History
with st.expander("🕘 Show Q&A History"):
    for i, q in enumerate(st.session_state.history):
        st.markdown(f"**{i+1}.** {q}")
