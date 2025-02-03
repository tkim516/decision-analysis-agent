from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

def eval_faithfulness(target_question: str, retrieved_questions: list, target_solution: str):

  test_case=LLMTestCase(
    input=target_question, 
    actual_output=target_solution,
    retrieval_context=retrieved_questions
  )
  
  metric = FaithfulnessMetric(threshold=0.5)

  metric.measure(test_case)
  print(metric.score)
  print(metric.reason)
  print(metric.is_successful())