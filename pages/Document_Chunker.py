import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import functions.research_helper as lch
from check_api import check_openai_api_key
from io import BytesIO
from functions.generate_unique_id import generate_unique_id
import time
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

# Display chunked data after it is added to vector store

st.set_page_config(page_title="Document Chunker", 
                   layout="wide")

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Document Chunker</h1>", unsafe_allow_html=True)

check_openai_api_key(st.session_state)

embeddings = OpenAIEmbeddings(
      model="text-embedding-3-large",
      openai_api_key=os.environ.get("OPENAI_API_KEY")
)

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

index_name = st.selectbox('Index Name', options=['decision-analysis-embeddings', 'decision-analysis-test'], index=0)
index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

doc_name = st.text_input('Enter PDF name')
pdf_file = st.file_uploader('Upload PDF', type='pdf')
pdf_sumbit_button = st.button('Submit PDF')

if pdf_sumbit_button and pdf_file and doc_name:
  binary_data = pdf_file.read()
  id_pdf_file = generate_unique_id()
  #pdf_viewer(input=binary_data, width=1200)
  
  lch.add_pdf_to_db(pdf_file, id_pdf_file, doc_name, vector_store)





  


