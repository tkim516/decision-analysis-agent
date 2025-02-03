import pandas as pd
from helper_functions import add_problem_embeddings, initialize_vector_store

vector_store = initialize_vector_store()

df = pd.read_csv('QuestionSet.csv')
df_subset= df[df["questionId"].isin(["3", "3", "3"])]

for index, row in df_subset.iterrows():
    question_id = row['questionId']
    question_text = row['questionText']
    
    add_problem_embeddings(question_text, question_id, vector_store)

