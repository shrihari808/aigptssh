# /aigptcur/app_service/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Robust .env loading ---
# This is placed at the very top to ensure variables are loaded before any other code runs.
# It explicitly finds the .env file in the project root directory (one level up from 'app_service').
try:
    load_dotenv(override=True)
except Exception as e:
    print(f"ERROR: Could not load .env file in config.py: {e}")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore
import chromadb
import tiktoken
import numpy as np
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

# --- API Keys and Configuration variables ---
# These are now read directly after being loaded.
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "market-data-index")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")

# OpenAI Model Definitions
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

DB_POOL = None
ENABLE_CACHING = True  # Set to True to enable caching
# Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Brave Search Parameters
MAX_SCRAPED_SOURCES = 10
MAX_PAGES = 1
MAX_ITEMS_PER_DOMAIN = 1


# Other constants
MAX_WEBPAGE_CONTENT_TOKENS = 1000
MAX_EMBEDDING_TOKENS = 8000
MAX_RERANKED_CONTEXT_ITEMS = 10

# Pinecone Indexing Wait Constants
PINECONE_MAX_WAIT_TIME = 30
PINECONE_CHECK_INTERVAL = 1

CONTEXT_SUFFICIENCY_THRESHOLD = 0.9
MIN_CONTEXT_LENGTH = 200
MIN_RELEVANT_DOCS = 3
W_RELEVANCE = float(os.getenv("W_RELEVANCE", 0.5450))
W_SENTIMENT = float(os.getenv("W_SENTIMENT", 0.1248))
W_TIME_DECAY = float(os.getenv("W_TIME_DECAY", 0.2814))
W_IMPACT = float(os.getenv("W_IMPACT", 0.0488))

# Source Credibility Weights
SOURCE_CREDIBILITY_WEIGHTS = {
    "moneycontrol.com": 0.9,
    "economictimes.indiatimes.com": 0.9,
    "business-standard.com": 0.85,
    "livemint.com": 0.85,
    "cnbctv18.com": 0.8,
    "screener.in": 0.95,
    "trendlyne.com": 0.9,
    "bloomberg.com": 0.95,
    "reuters.com": 0.95,
    "financialexpress.com": 0.85,
    "thehindubusinessline.com": 0.8,
    "ndtv.com": 0.75,
    "zeebiz.com": 0.7,
    "businesstoday.in": 0.8,
    "default": 0.5
}

# Impact Keywords for Scoring
IMPACT_KEYWORDS = [
    "price change", "rating downgrade", "layoffs", "policy changes",
    "acquisition", "merger", "earnings surprise", "bankruptcy",
    "restructuring", "dividend cut", "share buyback", "new product launch",
    "regulatory approval", "legal dispute", "fraud", "scandal",
    "inflation", "recession", "interest rate", "gdp growth", "unemployment",
    "high", "spike", "surge", "plunge", "soar", "crash", "rally",
    "breakout", "resistance", "support", "bullish", "bearish"
]


# CHROMA_SERVER Configuration
chroma_server_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=9001,
)

# --- Conditional LLM and Embeddings Initialization ---

if OPENAI_API_TYPE == "azure":
    # Azure OpenAI Configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")
    azure_chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    if not all([azure_endpoint, api_key, api_version, azure_chat_deployment, azure_embedding_deployment]):
        raise ValueError("Azure OpenAI credentials are not fully set in the environment variables.")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=azure_endpoint,
        api_type='azure',
        api_version=api_version,
        model_name=azure_embedding_deployment
    )
    
    GPT4o_mini = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=azure_chat_deployment,
        temperature=1.0
    )
    llm_stream = AzureChatOpenAI(
        streaming=True,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=azure_chat_deployment,
        temperature=1.0
    )
    llm_date = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=azure_chat_deployment,
        temperature=1.0
    )
    llm_screener = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=azure_chat_deployment,
        temperature=1.0
    )
    GPT3_16k = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_GPT3_16K_DEPLOYMENT_NAME", "gpt-35-turbo-16k"),
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=1.0
    )

else:
    # Standard OpenAI Configuration
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    GPT3_16k = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    GPT4o_mini = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
    llm_stream = ChatOpenAI(temperature=0.5, model="gpt-4o-mini", stream_usage=True, streaming=True)
    llm_date = ChatOpenAI(temperature=0.3, model="gpt-4o-2024-05-13")
    llm_screener = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# --- Pinecone Vector Store Initialization ---
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable must be set")

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index_name = "market-data-index"  # change if desired

if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# This 'vs' object can be imported and used across the application.
vs = PineconeVectorStore.from_existing_index(
    index_name, 
    embeddings,
    pool_threads=4
)

print("INFO: Pinecone vector store initialized.")


# # --- ChromaDB Vector Store Initialization ---
# client = chroma_server_client
# vs = Chroma(
#     client=client,
#     collection_name="brave_scraped",
#     embedding_function=embeddings,
# )

vs_promoter = Chroma(
    client=chroma_server_client,
    collection_name="promoters_202409",
    embedding_function=embeddings,
)

default_ef = embedding_functions.DefaultEmbeddingFunction()