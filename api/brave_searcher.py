# /aigptssh/api/brave_searcher.py

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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from dotenv import load_dotenv
from config import (
    MAX_WEBPAGE_CONTENT_TOKENS,
    MAX_EMBEDDING_TOKENS,
    MAX_PAGES,
    MAX_SCRAPED_SOURCES,
    SOURCE_CREDIBILITY_WEIGHTS,
    encoding,
    embeddings
)

load_dotenv()

pg_ip = os.getenv('PG_IP_ADDRESS')
psql_url = os.getenv('DATABASE_URL')

BLACKLISTED_DOMAINS = {
    'linkedin.com',
    'twitter.com',
    'x.com',
    'facebook.com',
    'instagram.com',
    'indmoney.com',
    'business-standard.com',
    'reuters.com',
    'en.wikipedia.org',
    'wikipedia.org'
}

class BraveNews:
    """
    Optimized class to handle Brave Search API interactions with concurrent web scraping,
    stricter timeouts, and performance improvements for web and news content.
    """
    
    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveNews.")
        self.brave_api_key = brave_api_key
        
        self.session_config = {
            'timeout': aiohttp.ClientTimeout(total=7, connect=3),
            'connector': aiohttp.TCPConnector(
                limit=20,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
            ),
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }

    async def get_snippets_only(self, query_term: str, max_results: int = 5) -> list[str]:
        """
        Ultra-fast function that gets only snippets from Brave API without any web scraping.
        Returns formatted snippets ready for LLM consumption.
        """
        async with aiohttp.ClientSession(**self.session_config) as session:
            brave_params = {
                "q": query_term, 
                "count": max_results, 
                "country": "in",
                "result_filter": "web,news", 
                "freshness": "pm"
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
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status != 200:
                        print(f"ERROR: Brave API returned status {response.status}")
                        return []
                        
                    brave_results = await response.json()
                    
                    snippets = []
                    web_results = brave_results.get('web', {}).get('results', [])
                    news_results = brave_results.get('news', {}).get('results', [])
                    
                    all_results = web_results + news_results
                    
                    for item in all_results[:max_results]:
                        if item.get("title") and item.get("description"):
                            snippet = f"**{item['title']}**\n{item['description']}\nSource: {item.get('url', 'N/A')}"
                            snippets.append(snippet)
                    
                    return snippets
                    
            except Exception as e:
                print(f"ERROR: Failed to get snippets: {str(e)}")
                return []

    async def _fetch_and_parse_url_async(self, session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
        """
        Optimized fetch and parse with a domain blacklist and better error handling.
        """
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc.replace('www.', '') in BLACKLISTED_DOMAINS:
                print(f"DEBUG: Skipping blacklisted domain: {url}")
                return url, ""
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as response:
                if response.status != 200:
                    print(f"WARNING: HTTP {response.status} for URL: {url}")
                    return url, ""

                content_type = response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in ['text/html', 'application/xhtml']):
                    print(f"WARNING: Skipping non-HTML content for URL: {url}")
                    return url, ""
                
                text_content = await response.text()

            extracted_text = await asyncio.to_thread(
                trafilatura.extract, 
                text_content, 
                include_comments=False, 
                include_tables=False,
                favor_precision=True
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
            if not item.get("url") or not item.get("title"):
                continue
                
            pub_date = item.get("page_age")
            if pub_date:
                try:
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
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    print(f"ERROR: Brave API returned status {response.status}")
                    return {}, False
                    
                brave_results = await response.json()
                
                has_more = brave_results.get('query', {}).get('more_results_available', False)
                
                return brave_results, has_more
                
        except asyncio.TimeoutError:
            print(f"ERROR: Timeout calling Brave API for page {page_num}")
            return {}, False
        except Exception as e:
            print(f"ERROR: Brave API call failed for page {page_num}: {str(e)}")
            return {}, False

    def _deduplicate_content(self, processed_items: list[dict], similarity_threshold: float = 0.9) -> list[dict]:
        """
        Deduplicates scraped content to avoid redundant information.
        """
        if len(processed_items) < 2:
            return processed_items

        contents = [item.get("full_webpage_content", "") for item in processed_items]
        
        valid_indices = [i for i, content in enumerate(contents) if content and len(content.split()) > 10]
        if len(valid_indices) < 2:
            return processed_items

        valid_contents = [contents[i] for i in valid_indices]

        try:
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
            tfidf_matrix = vectorizer.fit_transform(valid_contents)
            cosine_sim = cosine_similarity(tfidf_matrix)

            to_remove = set()
            for i in range(len(cosine_sim)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(cosine_sim)):
                    if j in to_remove:
                        continue
                    if cosine_sim[i, j] > similarity_threshold:
                        to_remove.add(j)
            
            deduplicated_items = []
            for i, item in enumerate(processed_items):
                if i not in valid_indices:
                    deduplicated_items.append(item)
                    continue
                
                try:
                    valid_content_index = valid_indices.index(i)
                    if valid_content_index not in to_remove:
                        deduplicated_items.append(item)
                except ValueError:
                    deduplicated_items.append(item)


            num_removed = len(processed_items) - len(deduplicated_items)
            if num_removed > 0:
                print(f"DEBUG: Deduplicated {num_removed} items based on content similarity.")
            
            return deduplicated_items

        except Exception as e:
            print(f"WARNING: Deduplication failed: {e}. Returning original items.")
            return processed_items

    async def search_and_scrape(self, session: aiohttp.ClientSession, query_term: str, max_pages: int = MAX_PAGES, max_sources: int = 30) -> list[dict]:
        """
        Accepts an active aiohttp session to prevent premature closing.
        """
        start_time = time.time()
        all_extracted_content = []
        links_encountered = set()

        print(f"DEBUG: Starting Broad Search for: '{query_term}' (max_sources={max_sources})")

        for current_page in range(1, max_pages + 1):
            if len(all_extracted_content) >= max_sources:
                print(f"DEBUG: Reached max_sources ({max_sources}). Stopping API calls.")
                break

            brave_results, has_more = await self._fetch_brave_page(session, query_term, current_page)

            if not brave_results:
                break

            page_extracted_content = self._extract_relevant_text(brave_results)
            if not page_extracted_content:
                break

            for item in page_extracted_content:
                link = item.get('link')
                if (link and link not in links_encountered and len(all_extracted_content) < max_sources and is_valid_news_url(link)):
                    all_extracted_content.append(item)
                    links_encountered.add(link)

            if len(all_extracted_content) >= max_sources or not has_more:
                break

            await asyncio.sleep(1.1)

        api_time = time.time() - start_time
        print(f"DEBUG: Broad Search phase completed in {api_time:.2f}s, collected {len(all_extracted_content)} potential sources")

        return all_extracted_content

    async def scrape_top_urls(self, session: aiohttp.ClientSession, sources: list[dict]) -> list[dict]:
        """
        Accepts an active aiohttp session.
        """
        scrape_start_time = time.time()
        links_to_scrape = [source['link'] for source in sources if source.get('link')]

        if not links_to_scrape:
            return []

        print(f"DEBUG: Starting Deep Scrape for {len(links_to_scrape)} top-ranked URLs.")
        
        scraped_content_map = {}
        batch_size = 10

        for i in range(0, len(links_to_scrape), batch_size):
            batch_links = links_to_scrape[i:i + batch_size]
            batch_tasks = [self._fetch_and_parse_url_async(session, url) for url in batch_links]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if not isinstance(result, Exception) and isinstance(result, tuple) and len(result) == 2:
                    url, content = result
                    scraped_content_map[url] = content
            
            if i + batch_size < len(links_to_scrape):
                await asyncio.sleep(0.2)
        
        for source in sources:
            source['full_webpage_content'] = scraped_content_map.get(source.get('link'), "")

        scrape_time = time.time() - scrape_start_time
        print(f"DEBUG: Deep Scrape phase completed in {scrape_time:.2f}s.")
        
        return sources

    def _process_for_dataframe(self, processed_items: list[dict]) -> pd.DataFrame:
        """
        Updated to handle the new, simpler data structure from our pipeline.
        """
        if not processed_items:
            return pd.DataFrame()
            
        news_data = []
        current_time = datetime.now()
        
        for item in processed_items:
            original = item 
            
            heading = "Unknown Source"
            if 'link' in original and original['link']:
                try:
                    parsed_url = urlparse(original['link'])
                    netloc = parsed_url.netloc.replace('www.', '')
                    heading = netloc.title() if netloc else "Unknown Source"
                except Exception:
                    heading = "Unknown Source"
            
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

            title = str(original.get('title', ''))
            description = str(original.get('snippet', ''))
            
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
            df = df[df['source_url'].str.strip() != '']
            df.reset_index(drop=True, inplace=True)
            return df
        
        return pd.DataFrame()

class BraveVideoSearch:
    """
    Class to handle Brave Video Search API interactions.
    """
    
    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/videos/search"
    
    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveVideoSearch.")
        self.brave_api_key = brave_api_key
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }
    
    async def search_detailed(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Performs a video search using the Brave API and returns detailed video data.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing detailed video information
        """
        print(f"\n--- DEBUG: BraveVideoSearch.search_detailed ---")
        print(f"Querying Brave Video API with: '{query}'")
        
        params = {
            "q": query,
            "count": max_results,
            "country": "in"
        }
        
        video_results = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BRAVE_API_BASE_URL, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Brave API Response Status: {response.status}")
                        
                        results = data.get("results", [])
                        print(f"Found {len(results)} video results")
                        
                        for item in results:
                            if 'url' in item and ('youtube.com' in item['url'] or 'youtu.be' in item['url']):
                                video_info = {
                                    'url': item['url'],
                                    'title': item.get('title', 'No title available'),
                                    'description': item.get('description', 'No description available'),
                                    'page_age': item.get('page_age', ''),
                                    'video': item.get('video', {}),
                                    'thumbnail': item.get('thumbnail', {})
                                }
                                video_results.append(video_info)
                                print(f"  -> Added: {video_info['title']}")
                            else:
                                print(f"  -> Skipped non-YouTube URL: {item.get('url', 'No URL')}")
                                
                    else:
                        print(f"Brave Video Search API error: {response.status}")
                        error_text = await response.text()
                        print(f"Response Body: {error_text}")
                        
        except Exception as e:
            print(f"An error occurred during Brave video search: {e}")

        print(f"Final list of {len(video_results)} YouTube videos from Brave API")
        print(f"--- END DEBUG: BraveVideoSearch.search_detailed ---\n")
        return video_results

class BraveRedditSearch:
    """
    Class to handle Brave Search API interactions for Reddit.
    """

    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveRedditSearch.")
        self.brave_api_key = brave_api_key
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key
        }

    async def search(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Performs a Reddit search using the Brave API.
        """
        print(f"DEBUG: Searching Brave for Reddit posts with query: '{query}'")
        params = {
            "q": f"site:reddit.com {query}",
            "count": max_results,
            "country": "in"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.BRAVE_API_BASE_URL, headers=self.headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("web", {}).get("results", [])
                    print(f"DEBUG: Brave search returned {len(results)} Reddit results.")
                    return results
                else:
                    print(f"Brave Reddit Search API error: {response.status}")
                    return []
                
                
# --- Standalone Functions ---

def is_valid_news_url(url: str) -> bool:
    """
    Check if URL is likely to contain useful financial news content.
    """
    if not url:
        return False
        
    skip_patterns = [
        '/topic/', '/tag/', 'careers', 'contact',
    ]
    
    url_lower = url.lower()
    for pattern in skip_patterns:
        if pattern in url_lower:
            return False
    
    return True

async def get_brave_snippets_only(query: str, max_results: int = 5) -> list[str]:
    """
    Standalone function to get only Brave API snippets without any scraping.
    """
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found.")
        return []
    
    searcher = BraveNews(brave_api_key)
    try:
        snippets = await searcher.get_snippets_only(query, max_results)
        print(f"DEBUG: Retrieved {len(snippets)} snippets in <3s")
        return snippets
    except Exception as e:
        print(f"ERROR in get_brave_snippets_only: {str(e)}")
        return []

async def get_brave_results(query: str, max_pages: int = MAX_PAGES, max_sources: int = MAX_SCRAPED_SOURCES):
    """
    High-level function to search Brave, controlled by max_pages and max_sources.
    """
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found.")
        return None, None
    
    searcher = BraveNews(brave_api_key)
    try:
        # This needs an aiohttp session to be passed
        async with aiohttp.ClientSession(**searcher.session_config) as session:
            processed_items = await searcher.search_and_scrape(session, query, max_pages=max_pages, max_sources=max_sources)
        
        if not processed_items:
            return None, None
        
        df = searcher._process_for_dataframe(processed_items)
        articles = [] if df.empty else df.to_dict('records')
        
        return articles, df
        
    except Exception as e:
        print(f"ERROR in get_brave_results: {str(e)}")
        return None, None