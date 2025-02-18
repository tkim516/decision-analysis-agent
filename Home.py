import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_openai import ChatOpenAI
from rag_functions import ( 
    retrieve_similar_questions, 
    convert_page_to_image, 
    parse_question_numbers, 
    find_text_in_pdf, 
    derive_python_logic, 
    generate_python_code,
    run_python_code)

# Page configuration
st.set_page_config(page_title="Decision Analysis Agent", 
                   page_icon="",
                   layout="wide")

st.markdown("<h1 style='text-align: center; color: #57cfff;'>Decision Analysis Agent</h1>", unsafe_allow_html=True)

target_question ="""An athletic league does drug testing of its athletes, 15 percent of whom use drugs. This test, however, is only 97 percent reliable. That is, a drug user will test positive with prob- ability 0.97 and negative with probability 0.03, and a nonuser will test negative with probability 0.97 and positive with prob- ability 0.03.
Develop a probability tree diagram to determine the poste- rior probability of each of the following outcomes of testing an athlete.
(a) The athlete is a drug user, given that the test is positive.
(b) The athlete is not a drug user, given that the test is positive. (c) The athlete is a drug user, given that the test is negative.
(d) The athlete is not a drug user, given that the test is negative."""

st.subheader("Target Question")
st.write(target_question)

pinecone_api_key = os.environ.get('PINECONE_API_KEY', '')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")


answers_pdf_path = "/Users/tyler/Downloads/ML/decision-analysis-agent/textbook-answers-b.pdf"
textbook_pdf_path = "/Users/tyler/Downloads/ML/decision-analysis-agent/textbook-b.pdf"

answers_index = "textbook-answers-b"
textbook_index = "textbook-b"

similar_question_text, page_num = retrieve_similar_questions(target_question, textbook_index, k=1)
similar_question_num = parse_question_numbers(similar_question_text)

st.subheader("Similar Question")
st.write(f'Question number {similar_question_num}')
st.write(similar_question_text)
st.write(f'From {textbook_index} at page {page_num}')

page_number = find_text_in_pdf(answers_pdf_path, "15.3-2")

st.subheader("Similar Question Answer")
st.write(f'Answer on page {page_number} of {answers_pdf_path}')

answers_img_path = convert_page_to_image(answers_pdf_path, page_number)

python_logic = derive_python_logic(answers_img_path, similar_question_text)

st.subheader("Derived Logic")
st.write(python_logic.content)

llm = ChatOpenAI(model="gpt-4o-mini")
python_code = generate_python_code(target_question, python_logic, llm)

st.subheader("Generated Python Code")
st.write(python_code)

# Run the generated Python code
result = run_python_code(python_code.content)

st.subheader("Output")
st.write(result["output"])