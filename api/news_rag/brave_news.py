# /aigptcur/app_service/api/news_rag/brave_news.py

import os
import json
import asyncio
import aiohttp
import trafilatura
import requests
import re
from urllib.parse import urlparse
import asyncpg
import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.docstore.document import Document
import psycopg2
import pandas as pd
from psycopg2 import sql
from datetime import datetime

from dotenv import load_dotenv
from config import (
    MAX_WEBPAGE_CONTENT_TOKENS,
    MAX_EMBEDDING_TOKENS,
    MAX_PAGES,
    MAX_SCRAPED_SOURCES,
    encoding,
    embeddings
)

load_dotenv()

pg_ip = os.getenv('PG_IP_ADDRESS')
psql_url = os.getenv('DATABASE_URL')

class BraveNews:
    """
    A refactored class to handle Brave Search API interactions, web scraping,
    and data processing in a structured way.
    """
    
    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveNewsSearcher.")
        self.brave_api_key = brave_api_key
        
    async def _fetch_and_parse_url_async(self, url: str) -> str:
        """Fetch content from URL using aiohttp and extract main text using trafilatura."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response.raise_for_status()
                    text_content = await response.text()

            extracted_text = trafilatura.extract(text_content, include_comments=False, include_tables=False)

            if extracted_text:
                tokens = encoding.encode(extracted_text)
                if len(tokens) > MAX_WEBPAGE_CONTENT_TOKENS:
                    extracted_text = encoding.decode(tokens[:MAX_WEBPAGE_CONTENT_TOKENS]) + "..."
                print(f"DEBUG: Successfully fetched and parsed URL: {url} ({len(extracted_text)} chars)")
                return extracted_text
            else:
                print(f"WARNING: Could not extract content from URL: {url}")
                return ""
        except Exception as e:
            print(f"WARNING: Failed to fetch/parse URL {url}: {str(e)}")
            return ""

    def _extract_relevant_text(self, brave_results: dict) -> list[dict]:
        """Extract relevant text snippets from Brave Search API results."""
        extracted_data = []
        web_results = brave_results.get('web', {}).get('results', [])
        news_results = brave_results.get('news', {}).get('results', [])
        
        all_results = web_results + news_results
        
        for item in all_results:
            pub_date = item.get("page_age")
            if pub_date:
                try:
                    # Ensure it's a valid ISO format date by handling the 'Z'
                    datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pub_date = None

            extracted_data.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", ""),
                "link": item.get("url", ""),
                "publication_date": pub_date
            })
        
        return extracted_data

    async def search_and_scrape(self, query_term: str) -> list[dict]:
        """Perform Brave search and scrape content for news articles."""
        all_extracted_content = []
        current_page_num = 1
        total_results_available = float('inf')
        links_encountered = set()

        while current_page_num <= MAX_PAGES:
            if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                print(f"DEBUG: Reached MAX_SCRAPED_SOURCES ({MAX_SCRAPED_SOURCES}). Stopping.")
                break

            offset = (current_page_num - 1) * 20
            if offset >= total_results_available:
                break

            print(f"DEBUG: Fetching Brave API results for query '{query_term}', page {current_page_num}")
            brave_params = {
                "q": query_term, "count": 20, "country": "in",
                "result_filter": "web,news", "freshness": "pm"
            }
            brave_headers = {
                "Accept": "application/json", "X-Subscription-Token": self.brave_api_key
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.BRAVE_API_BASE_URL, headers=brave_headers, params=brave_params,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as brave_response:
                        brave_response.raise_for_status()
                        brave_results = await brave_response.json()

                if 'query' in brave_results and 'total_results' in brave_results['query']:
                    total_results_available = brave_results['query']['total_results']

                page_extracted_content = self._extract_relevant_text(brave_results)
                if not page_extracted_content:
                    break

                for item in page_extracted_content:
                    link = item.get('link')
                    if link and link not in links_encountered and len(all_extracted_content) < MAX_SCRAPED_SOURCES:
                        all_extracted_content.append(item)
                        links_encountered.add(link)
                
                if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                    break
                
                if brave_results.get('query', {}).get('more_results_available', False):
                    current_page_num += 1
                    await asyncio.sleep(1)
                else:
                    break
            except Exception as e:
                print(f"ERROR: Brave API search failed: {str(e)}")
                break

        links_to_scrape = [item['link'] for item in all_extracted_content if item.get('link')]
        scrape_tasks = [self._fetch_and_parse_url_async(link) for link in links_to_scrape]
        scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        scraped_content_map = {links_to_scrape[i]: content for i, content in enumerate(scraped_contents)}

        processed_items = []
        for item in all_extracted_content:
            link = item.get('link')
            full_webpage_content = ""
            if link and link in scraped_content_map and not isinstance(scraped_content_map[link], Exception):
                full_webpage_content = scraped_content_map[link]

            text_to_embed = f"Title: {item['title']}\nSnippet: {item['snippet']}\nFull Content: {full_webpage_content}"
            tokens_to_embed = encoding.encode(text_to_embed)
            if len(tokens_to_embed) > MAX_EMBEDDING_TOKENS:
                text_to_embed = encoding.decode(tokens_to_embed[:MAX_EMBEDDING_TOKENS]) + "..."

            processed_items.append({
                "text_to_embed": text_to_embed,
                "original_item": item,
                "full_webpage_content": full_webpage_content
            })
        return processed_items

    def _process_for_dataframe(self, processed_items: list[dict]) -> pd.DataFrame:
        """Convert processed items to DataFrame format for PostgreSQL storage."""
        news_data = []
        for item in processed_items:
            original = item['original_item']
            heading = "Unknown Source"
            if 'link' in original and original['link']:
                parsed_url = urlparse(original['link'])
                heading = parsed_url.netloc.replace('www.', '').title()
            
            source_date = original.get('publication_date')
            try:
                date_obj = datetime.fromisoformat(source_date.replace('Z', '+00:00')) if source_date else datetime.now()
                formatted_date = date_obj.isoformat()
                date_published_int = int(date_obj.strftime('%Y%m%d'))
            except (ValueError, TypeError):
                date_obj = datetime.now()
                formatted_date = date_obj.isoformat()
                date_published_int = int(date_obj.strftime('%Y%m%d'))

            news_data.append({
                'source_url': original.get('link', ''),
                'image_url': None,
                'heading': heading,
                'title': str(original.get('title', '')).replace("'", ""),
                'description': str(original.get('snippet', '')).replace("'", ""),
                'source_date': formatted_date,
                'date_published': date_published_int
            })
        
        if news_data:
            df = pd.DataFrame(news_data)
            df['title'] = df['title'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
            df['description'] = df['description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
            df.dropna(subset=['source_url'], inplace=True)
            return df
        return pd.DataFrame()

# --- Standalone Functions ---

async def get_brave_results(query: str):
    """
    High-level function to search Brave, scrape results, and return articles and a DataFrame.
    This is the main entry point for other modules.
    """
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found in environment variables.")
        return None, None
    
    searcher = BraveNews(brave_api_key)
    try:
        processed_items = await searcher.search_and_scrape(query)
        if not processed_items:
            print(f"DEBUG: No processed items returned from search_and_scrape for query: '{query}'")
            return None, None
        
        df = searcher._process_for_dataframe(processed_items)
        if df.empty:
            print(f"DEBUG: DataFrame is empty after processing for query: '{query}'")
            return None, None
        
        # The 'articles' should be a list of dictionaries, which is what df.to_dict('records') provides
        articles = df.to_dict('records')
        return articles, df
    except Exception as e:
        print(f"Error in get_brave_results: {str(e)}")
        return None, None

import pandas as pd
from config import DB_POOL # Import the initialized DB_POOL

async def insert_post1(df: pd.DataFrame):
    """
    Asynchronously inserts a DataFrame into the source_data table using a connection pool.
    This is the non-blocking version of insert_post1.
    """
    # Check if the pool is initialized, if not, it's a critical startup error.
    if DB_POOL is None:
        print("CRITICAL ERROR: Database pool is not initialized. Cannot insert data.")
        return

    # Acquire a connection from the pool. This is very fast and doesn't create a new connection each time.
    async with DB_POOL.acquire() as conn:
        # Use a transaction to ensure all rows are inserted successfully or none are.
        async with conn.transaction():
            for index, row in df.iterrows():
                try:
                    # Check if the source_url already exists.
                    # fetchval is an efficient way to get a single value.
                    exists = await conn.fetchval(
                        "SELECT 1 FROM source_data WHERE source_url = $1",
                        row['source_url']
                    )

                    if not exists:
                        # Use parameterized queries ($1, $2, etc.) to prevent SQL injection.
                        await conn.execute("""
                            INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        row['source_url'],
                        row.get('image_url'), # Use .get() for optional columns
                        row['heading'],
                        row['title'],
                        row['description'],
                        row['source_date']
                        )
                except Exception as e:
                    # Log any errors that occur for a specific row without stopping the entire batch.
                    print(f"Error inserting row for URL {row['source_url']}: {e}")
    
    print(f"DEBUG: Asynchronous insert for {len(df)} rows completed.")



def initialize_pinecone():
    """Initializes and returns a Pinecone client and index name."""
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "market-data-index"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=1536, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc, index_name

def initialize_and_upsert_sync(documents, ids, index_name):
    """A single synchronous function to handle all blocking Pinecone logic."""
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' not found. Creating it now...")
        pc.create_index(
            name=index_name, dimension=1536, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created. Waiting for initialization...")
        # NOTE: A sleep here is still blocking, but it's now in a background thread
        # where it won't freeze the main app.
        import time
        time.sleep(10)

    embeddings = embeddings
    PineconeVectorStore.from_documents(
        documents=documents, embedding=embeddings, index_name=index_name,
        namespace="__default__", ids=ids
    )
    print("Upsert complete.")


async def data_into_pinecone(df):
    """Asynchronously prepares data and calls the blocking upsert function in a thread."""
    documents = []
    ids = []
    # (Your existing loop to prepare documents and ids goes here)
    for _, row in df.iterrows():
        combined_text = f"Title: {row['title']}\nDescription: {row['description']}"
        doc_id = re.sub(r'[^a-zA-Z0-9]', '', row["source_url"])
        documents.append(Document(
            page_content=combined_text,
            metadata={"url": row["source_url"], "date": row["date_published"], "title": row["title"]}
        ))
        ids.append(doc_id)

    if documents:
        # Run the entire synchronous process in a background thread
        await asyncio.to_thread(
            initialize_and_upsert_sync,
            documents,
            ids,
            "market-data-index"
        )
    return "Inserted!"
