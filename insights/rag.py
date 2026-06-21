"""
insights/rag.py — Singleton query engine shared across Django requests.
Lazy-initialised on first use so the server starts even without Atlas credentials.
"""

import os
from functools import lru_cache
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class RAGUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def get_query_engine():
    try:
        import pymongo
        from llama_index.core import VectorStoreIndex
        from llama_index.core.settings import Settings
        from llama_index.core.prompts import PromptTemplate
        from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
        from llama_index.embeddings.voyageai import VoyageEmbedding
        from llama_index.llms.openai import OpenAI
    except ImportError as e:
        raise RAGUnavailable(f"Missing dependency: {e}") from e

    mongodb_uri = os.environ.get("MONGODB_URI", "")
    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not all([mongodb_uri, voyage_key, openai_key]):
        raise RAGUnavailable(
            "Missing credentials. Set MONGODB_URI, VOYAGE_API_KEY, and OPENAI_API_KEY in .env"
        )

    SYSTEM_PROMPT = (
        "You are a nuclear energy intelligence analyst.\n"
        "Use ONLY information retrieved from the dataset records provided.\n"
        "Do not use outside knowledge.\n"
        "If the retrieved records do not contain sufficient information, say so explicitly.\n"
        "Provide quantitative answers whenever possible.\n"
        "Use markdown tables for rankings and comparisons.\n"
        "For ranking questions, return a clean markdown table with columns that match the question.\n"
        "For the nuclear electricity 2025 ranking question, use these columns exactly: "
        "Rank, Country, Nuclear Electricity (TWh), Nuclear Share of Grid (%), Dependency Tier.\n"
        "Always state that the information comes from retrieved dataset records.\n"
        "When discussing dependency tiers, use exact tier names from the data: "
        "Nuclear-Free, Nuclear-Minor, Nuclear-Moderate, Nuclear-Heavy, Nuclear-Dominant.\n\n"
        "Context:\n---------------------\n{context_str}\n---------------------\n"
        "Question: {query_str}\nAnswer:"
    )

    embed_model = VoyageEmbedding(
        model_name="voyage-large-2",
        voyage_api_key=voyage_key,
    )
    Settings.embed_model = embed_model
    Settings.llm = OpenAI(
        model="gpt-4",
        api_key=openai_key,
        system_prompt=(
            "You are a nuclear energy intelligence analyst. "
            "Answer only from the retrieved dataset records. "
            "Use markdown tables for rankings and comparisons."
        ),
    )

    client = pymongo.MongoClient(mongodb_uri)
    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=os.environ.get("MONGODB_DB", "energy_intelligence"),
        collection_name="nuclear_energy",
        vector_index_name="vector_index",
    )

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    return index.as_query_engine(
        similarity_top_k=10,
        text_qa_template=PromptTemplate(SYSTEM_PROMPT),
    )
