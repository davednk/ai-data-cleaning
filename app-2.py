import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --- 1. ACCESS CONTROL ---
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if not st.session_state.password_correct:
    st.title("🔒 Data Lab Login")
    pwd = st.text_input("Enter 4-Digit Access Code (4894)", type="password")
    if st.button("Unlock"):
        if pwd == st.secrets["4894"]:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Incorrect code.")
    st.stop()

# --- 2. THE DIAGNOSTIC TOOL ---
st.title("🤖 AI Data Agent: Troubleshooting Mode")

# Let's verify the key is even being read
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ CRITICAL: 'GEMINI_API_KEY' not found in Streamlit Secrets!")
    st.stop()

API_KEY = st.secrets["AIzaSyAv3GRjjJypWmC6Kg3JzUgSHrmjS-v9-cY"]

if st.button("🔎 Run Connection Test"):
    try:
        # Test basic Google Generative AI connection
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello! Are you working?")
        st.success(f"✅ API Key is Valid! Google responded: {response.text}")
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.info("If it says 'API_KEY_INVALID', go to AI Studio and get a fresh key.")

# --- 3. DATA & AGENT ---
st.divider()
uploaded_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.write("Preview:", df.head(3))
    
    # Setup LLM and Agent
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
    
    # We add a try-except here to catch the agent's specific failure
    try:
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        query = st.chat_input("What should the agent do?")
        
        if query:
            with st.spinner("Agent is working..."):
                res = agent.run(query)
                st.info(res)
    except Exception as e:
        st.error(f"⚠️ Agent Initialization Error: {e}")
