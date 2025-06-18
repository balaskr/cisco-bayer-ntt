import json
import os

import streamlit as st
from core_agent import handle_user_request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Streamlit App Setup ---

st.set_page_config(page_title="Project Administrator Assistant", page_icon="🏢")
st.title("Assistant")

# --- Data Loading (Cached for efficiency) ---
@st.cache_data
def load_hidden_json():
    """Loads the hidden JSON data from knowledge/data.json."""
    try:
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, "knowledge", "data.json")
        with open(data_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Error: knowledge/data.json not found. Please ensure the file exists.")
        return {"data": []} # Return empty data to prevent further errors
    except json.JSONDecodeError:
        st.error("Error: Could not decode knowledge/data.json. Check file format.")
        return {"data": []}

context_json = load_hidden_json()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial greeting message from the assistant
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your Project Adminstrator Assistant. How can I help you today?"})

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---
if prompt := st.chat_input("Hello!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):        
        assistant_response = handle_user_request(prompt, context_json, st.session_state.messages)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    with st.chat_message("assistant"):
        st.markdown(assistant_response)