import streamlit as st
import pandas as pd
import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import io

# --- 1. SECURE GATEKEEPER ---
def check_password():
    if st.session_state.get("password_correct", False):
        return True
    
    st.title("🔒 Data Science Lab Login")
    pwd = st.text_input("Enter Access Key", type="password")
    if st.button("Unlock Agent"):
        # You can change 'data2025' to whatever password you want
        if pwd == "data2025": 
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    return False

if not check_password():
    st.stop()

# --- 2. CONFIGURATION ---
# Use st.secrets for Streamlit Cloud, or a string for Colab testing
API_KEY = "AIzaSyAv3GRjjJypWmC6Kg3JzUgSHrmjS-v9-cY" 
genai.configure(api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

# --- 3. UI LAYOUT ---
st.title("🕵️‍♂️ AI Data Science Agent")
st.markdown("Upload a file and ask the agent to clean, analyze, or model it.")

uploaded_file = st.file_uploader("Upload your messy dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Load data once and keep it in memory
    if 'df' not in st.session_state:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.df = df
    
    df = st.session_state.df
    st.subheader("📊 Current Data Preview")
    st.dataframe(df.head(10))

    # --- 4. THE AGENTIC CHAT ---
    st.divider()
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
    query = st.chat_input("Command the agent (e.g., 'Clean all nulls' or 'Normalize the Age column')")
    
    if query:
        with st.spinner("Agent is writing and executing Python code..."):
            # The agent modifies the dataframe
            result = agent.run(query)
            st.write("### Agent Response:")
            st.info(result)
            
            # Important: We tell the agent to update the session state if it changed the data
            st.session_state.df = df 

    # --- 5. DOWNLOAD SECTION ---
    st.divider()
    st.subheader("💾 Export Your Work")
    
    @st.cache_data
    def convert_df(df_to_save):
        return df_to_save.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(st.session_state.df)
    
    st.download_button(
        label="Download Cleaned Data (CSV)",
        data=csv_data,
        file_name="ai_cleaned_data.csv",
        mime="text/csv",
    )
