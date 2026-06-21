"""
query.py
--------
Interactive query interface for the Nuclear Energy Intelligence dataset.

Usage:
    python query.py
    python query.py --query "Which countries generated the most nuclear electricity in 2025?"

Prerequisites:
    python ingest.py must have run successfully.
"""

import argparse
import os
from dotenv import load_dotenv
import pymongo

from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DB = os.environ.get("MONGODB_DB", "energy_intelligence")
COLLECTION = "nuclear_energy"
INDEX_NAME = "vector_index"
VOYAGE_MODEL = "voyage-large-2"

SYSTEM_PROMPT = """You are a nuclear energy intelligence analyst.

Use ONLY the information retrieved from the dataset records provided to you.
Do not use outside knowledge or training data.

If the retrieved records do not contain sufficient information to answer the
question, say: "The dataset does not contain sufficient information to answer
this question."

Guidelines:
- Provide quantitative answers whenever possible.
- Use markdown tables for rankings and comparisons.
- For ranking questions, return a clean markdown table with columns that match the question.
- For the nuclear electricity 2025 ranking question, use these columns exactly:
    Rank, Country, Nuclear Electricity (TWh), Nuclear Share of Grid (%), Dependency Tier.
- Always cite the specific year and country from the data.
- When discussing dependency tiers, use the exact tier names from the data:
  Nuclear-Free, Nuclear-Minor, Nuclear-Moderate, Nuclear-Heavy, Nuclear-Dominant.
"""

QA_PROMPT = PromptTemplate(
    "Context from the nuclear energy dataset:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using only the context above, answer the following question.\n"
    "If the question asks for a ranking or comparison, answer with a markdown table.\n"
    "Question: {query_str}\n"
    "Answer: "
)

DEMO_QUERIES = [
    "Which countries generated the most nuclear electricity in 2025, and how dependent are they on it?",
    "How has France's nuclear dependency changed over time?",
    "Compare China and France in nuclear generation and grid dependence.",
    "Which countries have the highest decarbonisation scores?",
    "Which European countries are most dependent on nuclear energy?",
]


def build_query_engine():
    client = pymongo.MongoClient(MONGODB_URI)
    client.admin.command("ping")

    embed_model = VoyageEmbedding(
        model_name=VOYAGE_MODEL,
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
    )
    Settings.embed_model = embed_model
    Settings.llm = OpenAI(
        model="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt=SYSTEM_PROMPT,
    )

    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=MONGODB_DB,
        collection_name=COLLECTION,
        vector_index_name=INDEX_NAME,
        embedding_key="embedding",
        text_key="text",
    )

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index.as_query_engine(
        similarity_top_k=10,
        text_qa_template=QA_PROMPT,
    )


def run_query(engine, question: str):
    print(f"\nQuestion: {question}")
    print("-" * 60)
    response = engine.query(question)
    print(response)
    print()


def main():
    parser = argparse.ArgumentParser(description="Query the Nuclear Energy Intelligence dataset.")
    parser.add_argument("--query", "-q", type=str, help="Question to ask (skips interactive mode)")
    parser.add_argument("--demo", action="store_true", help="Run all demo queries")
    args = parser.parse_args()

    print("Connecting to MongoDB Atlas and loading index...")
    engine = build_query_engine()
    print("Ready.\n")

    if args.demo:
        for q in DEMO_QUERIES:
            run_query(engine, q)
        return

    if args.query:
        run_query(engine, args.query)
        return

    # Interactive mode
    print("Nuclear Energy Intelligence Analyst")
    print("Type your question and press Enter. Type 'exit' to quit.")
    print("=" * 60)
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        run_query(engine, question)


if __name__ == "__main__":
    main()
