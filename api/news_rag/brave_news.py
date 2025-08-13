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
import time

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
    Optimized class to handle Brave Search API interactions with concurrent web scraping,
    stricter timeouts, and performance improvements.
    """
    
    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveNewsSearcher.")
        self.brave_api_key = brave_api_key
        
        # Optimized session configuration for better performance
        self.session_config = {
            'timeout': aiohttp.ClientTimeout(total=7, connect=3),  # Stricter timeouts
            'connector': aiohttp.TCPConnector(
                limit=20,  # Increased concurrent connections
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
            ),
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }

    async def _fetch_and_parse_url_async(self, session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
        """
        Optimized fetch and parse with better error handling and performance.
        Returns (url, extracted_text) tuple for easier processing.
        """
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                # Quick status check before reading content
                if response.status != 200:
                    print(f"WARNING: HTTP {response.status} for URL: {url}")
                    return url, ""
                
                # Check content type to avoid processing non-text content
                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in ['text/html', 'application/xhtml']):
                    print(f"WARNING: Skipping non-HTML content for URL: {url}")
                    return url, ""
                
                text_content = await response.text()

            # Run trafilatura in thread pool to avoid blocking
            extracted_text = await asyncio.to_thread(
                trafilatura.extract, 
                text_content, 
                include_comments=False, 
                include_tables=False,
                favor_precision=True  # Optimize for speed over completeness
            )

            if extracted_text:
                tokens = encoding.encode(extracted_text)
                if len(tokens) > MAX_WEBPAGE_CONTENT_TOKENS:
                    extracted_text = encoding.decode(tokens[:MAX_WEBPAGE_CONTENT_TOKENS]) + "..."
                print(f"DEBUG: Successfully parsed URL: {url} ({len(extracted_text)} chars)")
                return url, extracted_text
            else:
                print(f"WARNING: Could not extract content from URL: {url}")
                return url, ""
                
        except asyncio.TimeoutError:
            print(f"WARNING: Timeout fetching URL: {url}")
            return url, ""
        except aiohttp.ClientError as e:
            print(f"WARNING: Client error for URL {url}: {str(e)}")
            return url, ""
        except Exception as e:
            print(f"WARNING: Unexpected error for URL {url}: {str(e)}")
            return url, ""

    def _extract_relevant_text(self, brave_results: dict) -> list[dict]:
        """
        Optimized text extraction with better data validation.
        """
        extracted_data = []
        web_results = brave_results.get('web', {}).get('results', [])
        news_results = brave_results.get('news', {}).get('results', [])
        
        all_results = web_results + news_results
        
        for item in all_results:
            # Skip items without essential fields
            if not item.get("url") or not item.get("title"):
                continue
                
            pub_date = item.get("page_age")
            if pub_date:
                try:
                    # Validate and normalize date format
                    datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    pub_date = None

            extracted_data.append({
                "title": item.get("title", "").strip(),
                "snippet": item.get("description", "").strip(),
                "link": item.get("url", "").strip(),
                "publication_date": pub_date
            })
        
        return extracted_data

    async def _fetch_brave_page(self, session: aiohttp.ClientSession, query_term: str, page_num: int) -> tuple[dict, bool]:
        """
        Optimized Brave API call with better error handling.
        Returns (results_dict, has_more_pages).
        """
        offset = (page_num - 1) * 20
        
        brave_params = {
            "q": query_term, 
            "count": 20, 
            "country": "in",
            "result_filter": "web,news", 
            "freshness": "pm",
            "offset": offset
        }
        brave_headers = {
            "Accept": "application/json", 
            "X-Subscription-Token": self.brave_api_key
        }

        try:
            async with session.get(
                self.BRAVE_API_BASE_URL, 
                headers=brave_headers, 
                params=brave_params,
                timeout=aiohttp.ClientTimeout(total=10)  # Stricter timeout for API
            ) as response:
                if response.status != 200:
                    print(f"ERROR: Brave API returned status {response.status}")
                    return {}, False
                    
                brave_results = await response.json()
                
                # Check if more results are available
                has_more = brave_results.get('query', {}).get('more_results_available', False)
                
                return brave_results, has_more
                
        except asyncio.TimeoutError:
            print(f"ERROR: Timeout calling Brave API for page {page_num}")
            return {}, False
        except Exception as e:
            print(f"ERROR: Brave API call failed for page {page_num}: {str(e)}")
            return {}, False

    async def search_and_scrape(self, query_term: str) -> list[dict]:
        """
        Optimized search and scrape with concurrent processing and better resource management.
        """
        start_time = time.time()
        all_extracted_content = []
        links_encountered = set()

        # Create optimized session
        async with aiohttp.ClientSession(**self.session_config) as session:
            
            # Phase 1: Collect all URLs from Brave API (sequential but faster)
            print(f"DEBUG: Starting Brave API search for: '{query_term}'")
            
            for current_page in range(1, MAX_PAGES + 1):
                if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                    print(f"DEBUG: Reached MAX_SCRAPED_SOURCES ({MAX_SCRAPED_SOURCES}). Stopping API calls.")
                    break

                brave_results, has_more = await self._fetch_brave_page(session, query_term, current_page)
                
                if not brave_results:
                    print(f"DEBUG: No results from Brave API for page {current_page}")
                    break

                page_extracted_content = self._extract_relevant_text(brave_results)
                if not page_extracted_content:
                    print(f"DEBUG: No extracted content for page {current_page}")
                    break

                # Add unique URLs only
                for item in page_extracted_content:
                    link = item.get('link')
                    if link and link not in links_encountered and len(all_extracted_content) < MAX_SCRAPED_SOURCES:
                        all_extracted_content.append(item)
                        links_encountered.add(link)
                
                if len(all_extracted_content) >= MAX_SCRAPED_SOURCES or not has_more:
                    break
                    
                # Reduced sleep time for better performance
                await asyncio.sleep(0.5)

            api_time = time.time() - start_time
            print(f"DEBUG: Brave API phase completed in {api_time:.2f}s, collected {len(all_extracted_content)} URLs")

            # Phase 2: Concurrent web scraping with batching
            scrape_start = time.time()
            links_to_scrape = [item['link'] for item in all_extracted_content if item.get('link')]
            
            if not links_to_scrape:
                print("WARNING: No links to scrape")
                return []

            print(f"DEBUG: Starting concurrent scraping of {len(links_to_scrape)} URLs")
            
            # Batch processing to avoid overwhelming servers
            batch_size = 10  # Process 10 URLs concurrently
            scraped_content_map = {}
            
            for i in range(0, len(links_to_scrape), batch_size):
                batch_links = links_to_scrape[i:i + batch_size]
                print(f"DEBUG: Processing batch {i//batch_size + 1}/{(len(links_to_scrape)-1)//batch_size + 1}")
                
                # Create tasks for current batch
                batch_tasks = [
                    self._fetch_and_parse_url_async(session, url) 
                    for url in batch_links
                ]
                
                # Execute batch concurrently
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process batch results
                for url, content in batch_results:
                    if not isinstance(content, Exception):
                        scraped_content_map[url] = content
                
                # Small delay between batches to be respectful to servers
                if i + batch_size < len(links_to_scrape):
                    await asyncio.sleep(0.2)

            scrape_time = time.time() - scrape_start
            successful_scrapes = len([c for c in scraped_content_map.values() if c])
            print(f"DEBUG: Scraping phase completed in {scrape_time:.2f}s, {successful_scrapes}/{len(links_to_scrape)} successful")

        # Phase 3: Process and prepare final items
        processed_items = []
        for item in all_extracted_content:
            link = item.get('link')
            full_webpage_content = scraped_content_map.get(link, "")

            # Create optimized text for embedding
            text_parts = []
            if item['title']:
                text_parts.append(f"Title: {item['title']}")
            if item['snippet']:
                text_parts.append(f"Snippet: {item['snippet']}")
            if full_webpage_content:
                text_parts.append(f"Content: {full_webpage_content}")
            
            text_to_embed = "\n".join(text_parts)
            
            # Optimize token usage
            tokens_to_embed = encoding.encode(text_to_embed)
            if len(tokens_to_embed) > MAX_EMBEDDING_TOKENS:
                text_to_embed = encoding.decode(tokens_to_embed[:MAX_EMBEDDING_TOKENS]) + "..."

            processed_items.append({
                "text_to_embed": text_to_embed,
                "original_item": item,
                "full_webpage_content": full_webpage_content
            })

        total_time = time.time() - start_time
        print(f"DEBUG: Total processing time: {total_time:.2f}s for {len(processed_items)} items")
        
        return processed_items

    def _process_for_dataframe(self, processed_items: list[dict]) -> pd.DataFrame:
        """
        Optimized DataFrame processing with better error handling.
        """
        if not processed_items:
            return pd.DataFrame()
            
        news_data = []
        current_time = datetime.now()
        
        for item in processed_items:
            original = item['original_item']
            
            # Optimize heading extraction
            heading = "Unknown Source"
            if 'link' in original and original['link']:
                try:
                    parsed_url = urlparse(original['link'])
                    netloc = parsed_url.netloc.replace('www.', '')
                    heading = netloc.title() if netloc else "Unknown Source"
                except Exception:
                    heading = "Unknown Source"
            
            # Optimize date processing
            source_date = original.get('publication_date')
            try:
                if source_date:
                    date_obj = datetime.fromisoformat(source_date.replace('Z', '+00:00'))
                else:
                    date_obj = current_time
                formatted_date = date_obj.isoformat()
                date_published_int = int(date_obj.strftime('%Y%m%d'))
            except (ValueError, TypeError):
                date_obj = current_time
                formatted_date = date_obj.isoformat()
                date_published_int = int(date_obj.strftime('%Y%m%d'))

            # Clean text more efficiently
            title = str(original.get('title', ''))
            description = str(original.get('snippet', ''))
            
            # Remove non-ASCII characters more efficiently
            title = re.sub(r'[^\x00-\x7F]+', ' ', title).replace("'", "")
            description = re.sub(r'[^\x00-\x7F]+', ' ', description).replace("'", "")

            news_data.append({
                'source_url': original.get('link', ''),
                'image_url': None,
                'heading': heading,
                'title': title,
                'description': description,
                'source_date': formatted_date,
                'date_published': date_published_int
            })
        
        if news_data:
            df = pd.DataFrame(news_data)
            # Remove rows with empty URLs
            df = df[df['source_url'].str.strip() != '']
            df.reset_index(drop=True, inplace=True)
            return df
        
        return pd.DataFrame()

# --- Standalone Functions ---

async def get_brave_results(query: str):
    """
    Optimized high-level function to search Brave, scrape results, and return articles and a DataFrame.
    """
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found in environment variables.")
        return None, None
    
    searcher = BraveNews(brave_api_key)
    try:
        start_time = time.time()
        processed_items = await searcher.search_and_scrape(query)
        
        if not processed_items:
            print(f"DEBUG: No processed items returned from search_and_scrape for query: '{query}'")
            return None, None
        
        df = searcher._process_for_dataframe(processed_items)
        if df.empty:
            print(f"DEBUG: DataFrame is empty after processing for query: '{query}'")
            return None, None
        
        articles = df.to_dict('records')
        total_time = time.time() - start_time
        print(f"DEBUG: get_brave_results completed in {total_time:.2f}s with {len(articles)} articles")
        
        return articles, df
        
    except Exception as e:
        print(f"ERROR in get_brave_results: {str(e)}")
        return None, None

# --- Database Functions (Optimized for bulk operations) ---

import pandas as pd
from config import DB_POOL

async def insert_post1(df: pd.DataFrame):
    """
    Optimized bulk insert with better error handling and performance.
    """
    if DB_POOL is None:
        print("CRITICAL ERROR: Database pool is not initialized. Cannot insert data.")
        return

    if df.empty:
        print("DEBUG: Empty DataFrame, skipping insert")
        return

    start_time = time.time()
    
    async with DB_POOL.acquire() as conn:
        async with conn.transaction():
            # Prepare data for bulk operations
            urls_to_check = [row['source_url'] for _, row in df.iterrows()]
            
            # Batch check for existing URLs
            existing_urls = await conn.fetch(
                "SELECT source_url FROM source_data WHERE source_url = ANY($1)",
                urls_to_check
            )
            existing_urls_set = {row['source_url'] for row in existing_urls}
            
            # Filter out existing URLs
            new_rows = []
            for _, row in df.iterrows():
                if row['source_url'] not in existing_urls_set:
                    new_rows.append((
                        row['source_url'],
                        row.get('image_url'),
                        row['heading'],
                        row['title'],
                        row['description'],
                        row['source_date']
                    ))
            
            if new_rows:
                # Bulk insert new rows
                await conn.executemany("""
                    INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, new_rows)
                
                insert_time = time.time() - start_time
                print(f"DEBUG: Bulk inserted {len(new_rows)} new rows in {insert_time:.2f}s")
            else:
                print("DEBUG: No new rows to insert (all URLs already exist)")

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
    """Optimized synchronous function for Pinecone operations."""
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' not found. Creating it now...")
        pc.create_index(
            name=index_name, dimension=1536, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index created. Waiting for initialization...")
        import time
        time.sleep(10)

    embeddings_model = embeddings
    PineconeVectorStore.from_documents(
        documents=documents, embedding=embeddings_model, index_name=index_name,
        namespace="__default__", ids=ids
    )
    print("Upsert complete.")

async def data_into_pinecone(df):
    """Optimized Pinecone data insertion."""
    if df.empty:
        print("DEBUG: Empty DataFrame, skipping Pinecone upsert")
        return "No data to insert"
        
    start_time = time.time()
    documents = []
    ids = []
    
    for _, row in df.iterrows():
        combined_text = f"Title: {row['title']}\nDescription: {row['description']}"
        # More efficient ID generation
        doc_id = re.sub(r'[^a-zA-Z0-9]', '', str(row["source_url"]))[:100]  # Limit ID length
        
        documents.append(Document(
            page_content=combined_text,
            metadata={
                "url": row["source_url"], 
                "date": row["date_published"], 
                "title": row["title"]
            }
        ))
        ids.append(doc_id)

    if documents:
        await asyncio.to_thread(
            initialize_and_upsert_sync,
            documents,
            ids,
            "market-data-index"
        )
        
        upsert_time = time.time() - start_time
        print(f"DEBUG: Pinecone upsert completed in {upsert_time:.2f}s for {len(documents)} documents")
    
    return "Inserted!"