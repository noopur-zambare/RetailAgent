from typing import List, Dict, Any
import sqlite3
import os

import chromadb
from chromadb.config import Settings

from google import genai
from google.genai import types


class RetailRetriever:
    def __init__(self, db_path: str = "data/retail.db", chroma_path: str = ".chroma"):
        self.db_path = db_path

        self.chroma_client = chromadb.Client(
            Settings(
                persist_directory=chroma_path,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="retail_data"
        )

    def load_to_chroma(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        doc_id = 0

        for table in tables:
            table_name = table[0]

            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description]

            for row in rows:
                row_dict = dict(zip(columns, row))

                text = " | ".join(
                    f"{k}: {v}" for k, v in row_dict.items()
                )

                self.collection.add(
                    documents=[text],
                    ids=[f"{table_name}_{doc_id}"],
                    metadatas=[{"table": table_name}],
                )

                doc_id += 1

        conn.close()

    def search(self, query: str, k: int = 5) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
        )

        return results["documents"][0]


class RetailAgent:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash",
        db_path: str = "data/retail.db",
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        self.client = genai.Client(api_key=self.api_key)

        self.model_name = model_name

        self.retriever = RetailRetriever(db_path=db_path)

        self.retriever.load_to_chroma()

        self.system_prompt = """
You are a Canadian retail data analyst assistant working with structured retail sales datasets stored in SQLite and retrieved via context.

Your job is to analyze trends, compare industries, and extract insights ONLY from the provided context.

IMPORTANT RULES:
- Do NOT assume missing data
- Do NOT hallucinate industries, values, or trends
- If comparison data is missing, clearly say: "Not enough data to compare."
- Always base conclusions strictly on retrieved context
- Prefer quantitative reasoning over explanations
- Identify patterns such as growth, decline, or dominance when possible

HOW TO ANSWER:
1. Identify key industries or categories in the context
2. Compare values (average, total, or provided metrics)
3. Highlight highest and lowest performers if available
4. Summarize trend in 2-4 sentences max
5. Be precise and data-driven

OUTPUT STYLE:
- Concise
- Analytical
- Use numbers when available
- No SQL
- No database mention
"""

    def chat_completion(self, user_query: str) -> str:
        context_docs = self.retriever.search(user_query)

        context = "\n".join(context_docs)

        prompt = f"""
{self.system_prompt}

Context:
{context}

Question:
{user_query}

Answer:
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=64000,
            ),
        )

        return response.text