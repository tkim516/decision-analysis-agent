
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Initialize the LLM (e.g., GPT-4 via OpenAI API)
llm = ChatOpenAI(model="gpt-4")

# ========================================
# 1. Extract Text from PDFs
# ========================================

def extract_text_from_pdfs(folder_path):
    """
    Extracts text from all PDF files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        list: A list of strings, each containing the text of one PDF.
    """
    pdf_texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            reader = PdfReader(file_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()
            pdf_texts.append(pdf_text)
    return pdf_texts

# ========================================
# 2. Create a Knowledge Base
# ========================================

def create_knowledge_base_from_pdfs(folder_path):
    """
    Creates a searchable vector-based knowledge base from text extracted from PDFs.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        FAISS: A FAISS vector store containing embeddings of the PDF content.
    """
    # Extract text from all PDFs
    pdf_texts = extract_text_from_pdfs(folder_path)

    # Split the text into smaller chunks for better embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = []
    for pdf_text in pdf_texts:
        documents.extend(text_splitter.split_text(pdf_text))

    # Create embeddings and store them in a FAISS vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)

    return vector_store

# ========================================
# 3. Retrieve Knowledge from the Vector Store
# ========================================

def retrieve_knowledge_from_vector_store(vector_store, query):
    """
    Retrieves the top relevant knowledge entries for a query from the vector store.

    Args:
        vector_store (FAISS): The knowledge base (vector store).
        query (str): The user's query.

    Returns:
        list: Top relevant knowledge entries retrieved from the vector store.
    """
    results = vector_store.similarity_search(query, k=3)
    return [result.page_content for result in results]

# ========================================
# 4. Parse Query
# ========================================

def parse_query(query):
    """
    Parses the user's query to extract the task and relevant parameters.

    Args:
        query (str): The user's query.

    Returns:
        dict: A JSON-like dictionary containing the task and parameters.
    """
    prompt = f"""
    Analyze the following query and extract the required information in JSON format:
    Query: {query}
    
    Example Output:
    {{
        "task": "calculate_expected_value",
        "parameters": {{
            "probabilities": [0.7, 0.3],
            "payoffs": [500000, -200000]
        }}
    }}
    """
    response = llm.predict(prompt)
    return eval(response)  # Convert the JSON-like string to a Python dictionary

# ========================================
# 5. Generate Functions Dynamically
# ========================================

def generate_function_with_knowledge(task, parameters, knowledge):
    """
    Dynamically generates a Python function based on the task, parameters, and knowledge.

    Args:
        task (str): The task type (e.g., "calculate_expected_value").
        parameters (dict): The parameters required for the task.
        knowledge (str): Relevant knowledge to guide function generation.

    Returns:
        str: The generated Python function code as a string.
    """
    prompt = f"""
    Task: {task}
    Parameters: {parameters}

    Use the following knowledge to generate a Python function:
    {knowledge}

    Return only the Python function code.
    """
    generated_code = llm.predict(prompt)
    return generated_code

# ========================================
# 6. Execute the Generated Function
# ========================================

def execute_generated_code(generated_code):
    """
    Executes the dynamically generated Python code in a sandboxed environment.

    Args:
        generated_code (str): The dynamically generated Python function code.

    Returns:
        str or float: The result of executing the function, or an error message.
    """
    local_vars = {}
    try:
        exec(generated_code, {}, local_vars)  # Execute code in a safe environment
        return local_vars.get("result", "No result found.")
    except Exception as e:
        return f"Error in executing the code: {e}"

# ========================================
# 7. Coordinator Agent
# ========================================

def coordinator_agent_with_pdf_knowledge(query, vector_store):
    """
    Orchestrates the entire workflow: parsing the query, retrieving knowledge,
    generating computation functions, executing them, and returning the results.

    Args:
        query (str): The user's query.
        vector_store (FAISS): The knowledge base created from PDFs.

    Returns:
        str: A comprehensive response including parsed info, knowledge, code, and results.
    """
    # Step 1: Parse the query
    parsed_info = parse_query(query)
    task = parsed_info["task"]
    parameters = parsed_info["parameters"]

    # Step 2: Retrieve relevant knowledge from the knowledge base
    knowledge = retrieve_knowledge_from_vector_store(vector_store, query)

    # Step 3: Generate the required function using the retrieved knowledge
    generated_code = generate_function_with_knowledge(task, parameters, "\n".join(knowledge))

    # Step 4: Execute the generated function
    result = execute_generated_code(generated_code)

    # Step 5: Combine all results into a comprehensive response
    response = f"""
    Query: {query}

    Parsed Information:
    {parsed_info}

    Retrieved Knowledge:
    {knowledge}

    Generated Code:
    {generated_code}

    Result:
    {result}
    """
    return response

# ========================================
# Main Script
# ========================================

if __name__ == "__main__":
    # Path to the folder containing PDF documents
    pdf_folder_path = "/Users/tyler/Downloads/ML/decision_analysis_agent/pdfs"

    # Step 1: Create the knowledge base from PDFs
    vector_store = create_knowledge_base_from_pdfs(pdf_folder_path)

    # Step 2: Example user query
    query = "A company has two projects. Project A has a 70% chance of earning $500K and a 30% chance of losing $200K. Calculate the expected value."

    # Step 3: Run the coordinator agent
    response = coordinator_agent_with_pdf_knowledge(query, vector_store)
    print(response)
