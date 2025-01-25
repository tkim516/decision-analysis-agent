import streamlit as st
from functions.check_api import check_openai_api_key

# Page configuration
st.set_page_config(page_title="Decision Analysis Agent", 
                   page_icon="",
                   layout="wide")

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Decision Analysis Agent</h1>", unsafe_allow_html=True)

# Check if API key is stored in session state
check_openai_api_key(st.session_state)

import os
from pinecone import Pinecone

pinecone_api_key = os.environ.get('PINECONE_API_KEY', '')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

st.header('About')
st.subheader('''
This retrieval-augmented generation (RAG) app is a research project aimed to answer decision analysis questions.
''')


