import sys
import io
import base64
import fitz 
import os
import re
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pdf2image import convert_from_path




def retrieve_similar_questions(target_question: str, index_name, k=1):
         # Initialize embeddings
    embeddings = OpenAIEmbeddings(
      model="text-embedding-3-large",
      openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if index_name not in [i["name"] for i in pc.list_indexes()]:
      raise ValueError(f"Pinecone index '{index_name}' not found. Make sure it exists.")

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)

    retrieved_docs = vector_store.similarity_search(target_question, k=k)
    
    results = []
    for doc in retrieved_docs:

        page_num = doc.metadata.get("page", "N/A")  
        similar_question_text = doc.page_content
      
    return similar_question_text, page_num


def convert_page_to_image(pdf_path, page_number):

  images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)

  # Save the image
  image_path = f"answer_page.png"
  images[0].save(image_path, "PNG")

  print(f"Page {page_number} saved as {image_path}")

  return image_path

def parse_question_numbers(text):
    """
    Parse question numbers in the format '15.3-3' from the given text.

    Args:
        text (str): The input text to search.

    Returns:
        list: A list of extracted question numbers.
    """
    pattern = r'\b\d+\.\d+-\d+\b'  # Matches numbers like "15.3-3"
    question_numbers = re.findall(pattern, text)
    return question_numbers


def find_text_in_pdf(pdf_path, search_text):
    
    doc = fitz.open(pdf_path)  # Open the PDF

    for page_num in range(len(doc)):  # Loop through pages
        page = doc[page_num]
        text = page.get_text("text")  # Extract text
        if search_text in text:  # Check if the string is in the text
            return page_num + 1  # Return 1-based page number

    return -1  # Return -1 if not found


def derive_python_logic(image_path, question):
  client = OpenAI()

  # Function to encode the image
  def encode_image(image_path):
      with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode("utf-8")

  # Getting the Base64 string
  base64_image = encode_image(image_path)

  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": f"The attached image contains diagrams, tables, and formulas used to solve this question: {question}. Please analyze the image, explain the steps taken to answer the question, and write those steps as Python code that can be used to solve a similar question.",
                  },
                  {
                      "type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                  },
              ],
          }
      ],
  )

  print(response.choices[0])

  return response.choices[0].message


def generate_python_code(target_question: str, python_logic: str, llm):

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["target_question", "python_logic"],
        template="""
        You are an assistant that answers decision analysis problems.

        Answer this question: {target_question}

        Use this logic and Python code as a starting point to generate Python code to answer the question: {python_logic}

        Make sure all variable names use singular forms: 'poor_risk', 'average_risk', 'good_risk'. Do not use plural forms.
        
        Only return the Python code, do not include any explanations or additional text.
        """
    )

    # Generate response from LLM
    message = prompt.invoke({
        'target_question': target_question,
        'python_logic': python_logic
    })

    response = llm.invoke(message)  # Assuming this returns Python code as a string

    return response

import re
import sys
import io

def clean_code(code: str):
    # Remove Markdown-style code blocks
    code = re.sub(r"```(?:python)?\n?", "", code)  # Remove ```python or ```
    code = code.strip("`")  # Remove trailing backticks
    return code.strip()

def run_python_code(code: str):
    local_vars = {}  # Dictionary to store variables after execution
    output_buffer = io.StringIO()  # Redirect stdout to capture print statements

    # Clean the code before execution
    code = clean_code(code)

    try:
        # Redirect stdout
        sys.stdout = output_buffer
        
        # Execute the Python code safely
        exec(code, globals(), local_vars)  # Use globals() to maintain scope
        
        # Capture printed output
        output = output_buffer.getvalue()
    
    except Exception as e:
        output = f"Error: {str(e)}"
    
    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__

    # Debugging: Check stored variables
    print("Local Variables:", local_vars)

    return {
        "code": code,
        "output": output.strip(),
        "variables": local_vars
    }