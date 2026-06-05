import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)
import json
import time

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric
)

from agent_engine import RetailAgent
from metrics_summary import print_summary


with open(os.path.join(BASE_DIR, "eval_questions.json"), "r") as f:
    questions = json.load(f)

agent = RetailAgent()

faithfulness = FaithfulnessMetric()
relevancy = AnswerRelevancyMetric()
hallucination = HallucinationMetric()

results = []


for q in questions:

    start = time.time()

    out = agent.chat_completion(q)

    latency = time.time() - start

    # DeepEval test case
    test_case = LLMTestCase(
    input=q,
    actual_output=out["answer"],
    context=out["context"],
    retrieval_context=out["context"])
    
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    hallucination.measure(test_case)

    print("\n==============================")
    print("Question:", q)
    print("Answer:", out["answer"])
    print("Latency:", round(latency, 2))

    print("Faithfulness:", round(faithfulness.score, 3))
    print("Relevancy:", round(relevancy.score, 3))
    print("Hallucination:", round(hallucination.score, 3))
    print("==============================\n")

    results.append({
        "question": q,
        "faithfulness": faithfulness.score,
        "relevancy": relevancy.score,
        "hallucination": hallucination.score,
        "latency": latency
    })


df = print_summary(results)