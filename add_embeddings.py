from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
from langchain_core.documents import Document


answers_pdf_path = "/Users/tyler/Downloads/ML/decision-analysis-agent/textbook-answers-b.pdf"
textbook_pdf_path = "/Users/tyler/Downloads/ML/decision-analysis-agent/textbook-b.pdf"

def extract_text_with_page_numbers(vector_store, pdf_path, chunk_size=1200, chunk_overlap=400):
   
    doc = fitz.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks_with_metadata = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        chunks = text_splitter.split_text(text)
        
        for chunk in chunks:
            chunks_with_metadata.append(
                {"text": chunk, "metadata": {"page": page_num + 1}}
            )  # Page numbers are 1-based

    # Convert dictionary chunks into Document objects
    documents = [
      Document(page_content=chunk["text"], metadata=chunk["metadata"])
      for chunk in chunks_with_metadata
    ]

    # Embed and upload the text chunks
    vector_store.add_documents(documents=documents)
    
    doc.close()
    
    return chunks_with_metadata