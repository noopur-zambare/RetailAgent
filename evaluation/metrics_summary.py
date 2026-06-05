import pandas as pd

def print_summary(results):
    df = pd.DataFrame(results)

    print("\n========== FINAL RESULTS ==========\n")
    print(df)

    print("\n========== AVERAGE METRICS ==========\n")
    print("Faithfulness:", round(df["faithfulness"].mean(), 3))
    print("Relevancy:", round(df["relevancy"].mean(), 3))
    print("Hallucination:", round(df["hallucination"].mean(), 3))
    print("Latency:", round(df["latency"].mean(), 3))

    return df