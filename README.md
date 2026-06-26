# Nuclear RAG Analyst

Django + MongoDB Vector Search + LlamaIndex + Voyage AI + GPT-4 demo for asking natural-language questions over global nuclear energy data.

## What It Does

- Converts the Kaggle CSV into one JSON file per country-year row.
- Embeds the JSON records with Voyage AI.
- Stores vectors in MongoDB Atlas.
- Retrieves relevant records with MongoDB Vector Search.
- Uses GPT-4 through LlamaIndex to return grounded, quantitative answers.
- Includes a minimal Django UI for screen-recorded demos.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` at the project root and fill in your values:

```dotenv
DJANGO_SECRET_KEY=replace-with-a-local-dev-secret
DEBUG=True
MONGODB_URI=your-mongodb-atlas-connection-string
MONGODB_DB=nuclear_rag
OPENAI_API_KEY=your-openai-key
VOYAGE_API_KEY=your-voyage-key
```

For a quick install command in demos:

```bash
pip install llama-index pymongo llama-index-vector-stores-mongodb llama-index-embeddings-voyageai voyageai openai python-dotenv django-mongodb-backend certifi
```

## Dataset

Download `global_nuclear_energy_intelligence_1965_2025.csv` from Kaggle and place it at the project root. The CSV and generated `data/` folder are intentionally ignored by git.

Convert the CSV into JSON documents:

```bash
python convert.py
```

The converter currently exports rows from 2015 onward to keep ingestion fast for demos.

## MongoDB Atlas Vector Index

Create a Vector Search index named `vector_index` on database `nuclear_rag`, collection `nuclear_energy`:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

## Ingest

```bash
python ingest.py
```

## Query From Terminal

```bash
python query.py --query "Which countries generated the most nuclear electricity in 2025, and how dependent are they on it?"
```

The prompt is configured to return a ranked markdown table with these columns: `Rank`, `Country`, `Nuclear Electricity (TWh)`, `Nuclear Share of Grid (%)`, and `Dependency Tier`.

## Run Django UI

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'dotenv'`:

  ```bash
  pip install -r requirements.txt
  python query.py --query "test"
  ```

- If `python` points to the wrong interpreter, always run scripts via `.venv/bin/python`.
