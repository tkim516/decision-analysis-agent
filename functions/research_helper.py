from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile
from langchain.schema import Document  # Import the Document class
import os
import streamlit as st


def add_pdf_to_db(uploaded_file, upload_id, document_name: str, vector_store):
  # Add argument, document_name, which will be used as semantic identifier for the document

  with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

  try:
        # Use the temporary file path with PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)

        # Split the PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(pages)

        # Convert splits into Document objects
        documents = [
            Document(page_content=split.page_content, 
                     metadata={"upload_id": upload_id, "document_name": document_name, "chunk_index": i})
            for i, split in enumerate(all_splits)
        ]

        with st.expander("Show Chunked Document"):
          st.write(upload_id)
          st.write(documents)

        _ = vector_store.add_documents(documents=documents)

        return vector_store
  
  finally:
        # Clean up: delete the temporary file
        os.remove(temp_file_path)



# Need a way to search textbook for frameworks to apply based on the question.
# LLM used to interpret the question and generate a query that will search for applicable frameworks 
def get_response_from_query(vector_store, query, list_document_names, k=4):

  retrieved_docs = vector_store.similarity_search(
    query,
    k=k,
    filter={"document_name": list_document_names}
    )
    
  
  retrieved_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

  with st.expander("Show Retrieved Content"):
    #st.write(upload_id)
    st.write(retrieved_content)
  
  llm = ChatOpenAI(model="gpt-4o-mini")

  prompt = PromptTemplate(
    input_variables=['question', 'context'],
    template="""
    You are an assistant that answers questions based on the information in the uploaded PDF.

    Answer this question: {question}

    Use this segment of the uploaded PDF: {context}
    """
  )

  message = prompt.invoke({
    'question': query,
    'context': retrieved_content})
  
  response = llm.invoke(message)

  return response

  



