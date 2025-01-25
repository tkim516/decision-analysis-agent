import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import functions.research_helper as lch
from functions.check_api import check_openai_api_key
from io import BytesIO
from functions.generate_unique_id import generate_unique_id
import time
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os

st.set_page_config(page_title="Research Paper Assistant", 
                   page_icon="📝",
                   layout="wide")

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Research Paper Assistant</h1>", unsafe_allow_html=True)

check_openai_api_key(st.session_state)

embeddings = OpenAIEmbeddings(
      model="text-embedding-3-large",
      openai_api_key=os.environ.get("OPENAI_API_KEY")
  )

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "decision-analysis-embeddings"
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

with st.sidebar:
    with st.form("input_form"):
      doc_name_filter = st.text_input('Document Name', value='text_minus_q_ex')
      query_input = st.text_area('Your Question', value='What is the paper about?')
      query_submit_button = st.form_submit_button('Submit')

if query_submit_button and doc_name_filter:
  
  list_doc_name_filter = []
  list_doc_name_filter.append(doc_name_filter)
  st.write(list_doc_name_filter)

  response = lch.get_response_from_query(vector_store, query_input, doc_name_filter)   
  st.header('Response')   
  st.write(response.content)


#filter={"document_name": {"$in": list_document_names}} if list_document_names else None
  #)





  


