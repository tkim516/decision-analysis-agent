from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

#from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

from langchain_core.messages import SystemMessage
#from langgraph.prebuilt import ToolNode

from langchain_core.tools import tool
import requests
import json

#graph_builder = StateGraph(MessagesState)

# Build a tool which can search the internet for gifts
# Generate a query to pass to search based on the user's interests


def generate_gift_search_query(recipient_hobbies, recipient_age, budget, api_key):
    """Generate gift search queries based on interests."""
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    prompt = PromptTemplate(
        template="""
        Using the details provided, generate a Google search shopping query for gifts that are well-suited to the individual’s interests, age, 
        the nature of your relationship with them, the occasion, and the given budget range.

        Person's interests: {recipient_hobbies}
        Person's age in years: {recipient_age}
        Budget range in USD: {budget}
        """,
        input_variables=[
            "recipient_hobbies", "recipient_age", "budget"
        ]
    )

    # Format the prompt with user inputs
    message = prompt.format(
        recipient_hobbies=recipient_hobbies,
        recipient_age=recipient_age,
        budget=budget,
    )
    
    # Get the response
    response = llm(message)
    return response

def serper_shopping_search(serper_api_key: str, query: str) -> dict:
  """Search internet for gifts related to query"""

  url = "https://google.serper.dev/shopping"

  payload = json.dumps({
    "q": query
  })
  headers = {
    'X-API-KEY': serper_api_key,
    'Content-Type': 'application/json'
  }
  response = requests.request("POST", url, headers=headers, data=payload)

  print(response.text)

  return response