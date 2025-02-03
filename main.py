import pandas as pd
from langchain_openai import ChatOpenAI
from helper_functions import (
    initialize_vector_store,
    retrieve_similar_questions,
    get_answer_text,
    parse_excel_data,
    extract_excel_logic,
    solve_problem
)

llm = ChatOpenAI(model="gpt-4o-mini")

df = pd.read_csv('QuestionSet.csv')

vector_store = initialize_vector_store()

target_question = """
Scenario: Buying a New Laptop vs. Keeping Your Old One

You currently have an older laptop that’s starting to show its age. You are considering whether to buy a new laptop or continue using your old one. The key trade-off is between certain monthly loan payments for a new model versus unpredictable repair costs for your current laptop.

New Laptop Option

Price of the new laptop: $1,200
You have $400 in savings that you could put toward a down payment, leaving $800 to be financed.
Your bank offers a 2-year (24-month), 8% loan on the $800. The monthly payment on this loan would be $36.00 for 24 months.
Because the new laptop is under warranty for the first year, you will have no repair costs during that period. However, starting in the second year, you expect minimal maintenance of about $80 per year (e.g., extended warranty or general upkeep).
Keeping Your Old Laptop

Current estimated resale value of your old laptop: $300 (if you were to sell it now).
Potential repair costs over the next 2 years include:
Immediate repairs (likely to be needed in the next few months): $50 for a new battery and possibly a small hardware fix.
Year 1 (after the immediate repairs): A 25% chance of additional repairs costing $0, 50% chance of $150, and 25% chance of $300.
Year 2: The same probabilities (25% chance of $0, 50% chance of $150, 25% chance of $300) apply for potential repairs.
There are no monthly loan payments for keeping your old laptop, but you face uncertain repair bills.
Resale (Salvage) Value at the End of 2 Years

New Laptop: If you buy the new laptop, at the end of 2 years, it will likely be worth about $700. At that point, you will have fully paid off your loan, so the net salvage value is $700.
Old Laptop: If you keep your old laptop for 2 more years, its resale value by then will be approximately $100—assuming it remains functional.
Assignment: Draw a 2-Year Decision Tree

Decisions and Chance Nodes:

The first decision node is whether to buy the new laptop or keep the old one.
If you buy the new laptop, your cash outflow consists of the $400 down payment plus monthly payments of $36.00 for 24 months, and an annual maintenance cost of $80 in the second year.
If you keep the old laptop, you have an immediate $50 repair cost plus the uncertain repair costs for Year 1 and Year 2.
For each year that you keep the old laptop, there is a chance node with the following probabilities for repair costs:
25% chance of $0
50% chance of $150
25% chance of $300
End-of-Year Values:

After Year 2, add the resale (salvage) value for whichever laptop you have.
For the new laptop: salvage value is $700.
For the old laptop: salvage value is $100.
Expected Net Value Calculations:

Label all cash flows (outflows for loan payments, maintenance, repairs; inflow for salvage) on your decision tree.
Compute the net value at each terminal branch, accounting for the probability distributions of the repair costs for the old laptop and the certain loan and minimal maintenance costs for the new one.
Compare Both Options:

Use your decision tree to compare the expected net values of buying the new laptop versus keeping (and repairing) your old one.
Consider whether the old laptop’s higher repair uncertainty outweighs the certain but steady loan payment for the new one.
"""

similar_question_id, similar_question_text = retrieve_similar_questions(target_question, vector_store)

answer_text = get_answer_text(similar_question_id, df)

excel_data = parse_excel_data(similar_question_id, df)

logic_framework = extract_excel_logic(similar_question_text, answer_text, excel_data, llm)

print(f'Question Id: {similar_question_id}')
print(f'Question Text: {similar_question_text}')
print(f'Answer Text: {answer_text}')
print(f'Excel Data: {excel_data}')

print('_________________________')
print(f'Logic Framework: {logic_framework}')




