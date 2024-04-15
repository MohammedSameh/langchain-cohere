import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

CHROMA_SETTINGS = Settings(
chroma_db_impl='duckdb+parquet',
persist_directory=os.environ.get('PERSIST_DIRECTORY'),
anonymized_telemetry=False,
allow_reset=True
)