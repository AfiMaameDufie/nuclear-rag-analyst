"""
ingest.py
---------
Loads JSON documents from data/, generates Voyage AI embeddings, and stores
vectors in MongoDB Atlas for use with Atlas Vector Search.

Run once (or re-run to refresh):
    python ingest.py

Prerequisites:
    1. python convert_csv.py          # build data/ folder
    2. Set .env with MONGODB_URI, VOYAGE_API_KEY, OPENAI_API_KEY
     3. Create Atlas Vector Search index named 'vector_index' on the collection
         with 1536 dimensions for voyage-large-2.
"""

import os
from dotenv import load_dotenv
import pymongo
import certifi

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
MONGODB_DB = os.environ.get("MONGODB_DB", "energy_intelligence")
COLLECTION = "nuclear_energy"
INDEX_NAME = "vector_index"
DATA_DIR = "data"

# voyage-large-2 produces 1536-dimensional vectors
VOYAGE_MODEL = "voyage-large-2"
EMBED_DIM = 1536


def main():
    print("Connecting to MongoDB Atlas...")
    client = pymongo.MongoClient(MONGODB_URI, tlsCAFile=certifi.where())

    # Smoke-test the connection
    client.admin.command("ping")
    print(f"  Connected. Target: {MONGODB_DB}.{COLLECTION}")

    # Configure LlamaIndex global settings
    Settings.embed_model = VoyageEmbedding(
        model_name=VOYAGE_MODEL,
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
    )
    Settings.llm = OpenAI(
        model="gpt-4",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    print(f"Loading documents from {DATA_DIR}/...")
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"  Loaded {len(docs)} documents.")

    vector_store = MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=MONGODB_DB,
        collection_name=COLLECTION,
        vector_index_name=INDEX_NAME,
        embedding_key="embedding",
        text_key="text",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Generating embeddings and storing vectors in Atlas...")
    print("  This may take several minutes for the full dataset.")

    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )

    print("\nIngestion complete.")
    print(f"Vectors stored in: {MONGODB_DB}.{COLLECTION}")
    print(f"Vector index name: {INDEX_NAME}")
    print("\nNext: create the Atlas Vector Search index if you haven't already,")
    print("then run:  python query.py")


if __name__ == "__main__":
    main()
