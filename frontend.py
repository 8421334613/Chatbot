# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step 1: Setup UI with streamlit
import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agents")
st.write("Create and Interact with AI Agents using LangGraph + FastAPI")

# Inputs
system_prompt = st.text_area("🧠 Define your AI Agent's Role:", height=70, placeholder="E.g. Act as a travel guide...")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENROUTER = ["openrouter-llama-3.3-70b", "mixtral-8x7b-32768"]

provider = st.radio("🛰️ Select AI Provider:", ("Groq", "OpenRouter"))

if provider == "Groq":
    selected_model = st.selectbox("📌 Select Groq Model:", MODEL_NAMES_GROQ)
else:
    selected_model = st.selectbox("📌 Select OpenRouter Model:", MODEL_NAMES_OPENROUTER)

allow_web_search = st.checkbox("🌐 Allow Web Search")

user_query = st.text_area("💬 Ask Something:", height=150, placeholder="Type your question here...")

API_URL = "http://127.0.0.1:9999/chat"

# Step 2: Send query to backend
if st.button("🚀 Ask Agent"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            payload = {
                "model_name": selected_model,
                "model_provider": provider,
                "system_prompt": system_prompt,
                "messages": [user_query],
                "allow_search": allow_web_search
            }

            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    if "error" in response_data:
                        st.error(f"❌ {response_data['error']}")
                    else:
                        st.success("✅ Agent Responded!")
                        st.markdown(f"**🧠 Response:** {response_data['response']}")
                else:
                    st.error("🚨 API Error: Backend did not return a 200 OK response.")
            except requests.exceptions.RequestException as e:
                st.error(f"🚨 Request Failed: {e}")
    else:
        st.warning("⚠️ Please type a query before clicking the button.")
