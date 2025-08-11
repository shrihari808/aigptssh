# services/brave_searcher.py

import asyncio
import aiohttp
import trafilatura
import json # Import json for pretty printing raw API responses
from datetime import datetime
from urllib.parse import urlparse # For parsing URLs in impact score calculation

# Import constants from the config file
from config import (
    MAX_WEBPAGE_CONTENT_TOKENS,
    MAX_EMBEDDING_TOKENS,
    MAX_PAGES, # MAX_PAGES is imported from config.py
    MAX_SCRAPED_SOURCES, # NEW: Import MAX_SCRAPED_SOURCES
    encoding # tiktoken encoding
)

class BraveSearcher:
    """Handles interactions with the Brave Search API and web scraping."""
    BRAVE_API_BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, brave_api_key: str):
        if not brave_api_key:
            raise ValueError("Brave API key is required for BraveSearcher.")
        self.brave_api_key = brave_api_key

    async def _fetch_and_parse_url_async(self, url: str) -> str:
        """
        Fetches content from a URL using aiohttp and extracts main text using trafilatura.
        Truncates content to MAX_WEBPAGE_CONTENT_TOKENS.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                    text_content = await response.text()

            extracted_text = trafilatura.extract(text_content, include_comments=False, include_tables=False)

            if extracted_text:
                # Truncate extracted text to avoid excessively long content
                tokens = encoding.encode(extracted_text)
                if len(tokens) > MAX_WEBPAGE_CONTENT_TOKENS:
                    extracted_text = encoding.decode(tokens[:MAX_WEBPAGE_CONTENT_TOKENS]) + "..."
                print(f"DEBUG: Successfully fetched and parsed URL with trafilatura: {url} ({len(extracted_text)} chars)")
                return extracted_text
            else:
                print(f"WARNING: Trafilatura could not extract content from URL: {url}. Returning empty string.")
                return ""
        except aiohttp.ClientError as e:
            print(f"WARNING: HTTP/Network error fetching URL {url} with aiohttp: {str(e)}")
            return ""
        except Exception as e:
            print(f"WARNING: Failed to parse URL {url} or other error with trafilatura: {str(e)}")
            return ""

    def _extract_relevant_text(self, brave_results: dict) -> list[dict]:
        """
        Extracts relevant text snippets (title, snippet, link, publication_date)
        from Brave Search API results.
        Prioritizes 'mixed' results if available, otherwise falls back to direct categories.
        """
        extracted_data = []

        web_results = brave_results.get('web', {}).get('results', [])
        news_results = brave_results.get('news', {}).get('results', [])
        video_results = brave_results.get('videos', {}).get('results', []) # Still present in Brave API response structure

        if 'mixed' in brave_results and 'main' in brave_results['mixed']:
            print("DEBUG: Processing 'mixed.main' results from Brave API response.")
            for item_spec in brave_results['mixed']['main']:
                item_type = item_spec.get('type')
                item_index = item_spec.get('index')

                item_to_add = None

                if item_type == 'web' and item_index is not None and item_index < len(web_results):
                    item_to_add = web_results[item_index]
                elif item_type == 'news' and item_index is not None and item_index < len(news_results):
                    item_to_add = news_results[item_index]

                if item_to_add:
                    # Extract page_age if available, otherwise default to None
                    pub_date = item_to_add.get("page_age")
                    if pub_date:
                        try:
                            datetime.fromisoformat(pub_date)
                            print(f"DEBUG: Using page_age as publication_date: {pub_date}")
                        except (ValueError):
                            pub_date = None # Fallback if not a valid ISO format

                    extracted_data.append({
                        "title": item_to_add.get("title", ""),
                        "snippet": item_to_add.get("description", ""),
                        "link": item_to_add.get("url", ""),
                        "publication_date": pub_date
                    })
        else:
            print("DEBUG: 'mixed.main' not found or empty. Falling back to direct 'web' and 'news' results.")
            for item in web_results:
                pub_date = item.get("page_age")
                print(f"DEBUG: Brave API raw page_age (fallback web): {pub_date} for link: {item.get('url', 'N/A')}")
                if pub_date:
                    try:
                        datetime.fromisoformat(pub_date)
                        print(f"DEBUG: Using page_age as publication_date (fallback web): {pub_date}")
                    except (ValueError):
                        pub_date = None
                        print(f"WARNING: page_age '{pub_date}' is not a valid ISO format (fallback web). Setting to None.")
                else:
                    print(f"DEBUG: page_age is None for fallback web link: {item.get('url', 'N/A')}")
                extracted_data.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "link": item.get("url", ""),
                    "publication_date": pub_date
                })
            for item in news_results:
                pub_date = item.get("page_age")
                print(f"DEBUG: Brave API raw page_age (fallback news): {pub_date} for link: {item.get('url', 'N/A')}")
                if pub_date:
                    try:
                        datetime.fromisoformat(pub_date)
                        print(f"DEBUG: Using page_age as publication_date (fallback news): {pub_date}")
                    except (ValueError):
                        pub_date = None
                        print(f"WARNING: page_age '{pub_date}' is not a valid ISO format (fallback news). Setting to None.")
                else:
                    print(f"DEBUG: page_age is None for fallback news link: {item.get('url', 'N/A')}")
                extracted_data.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "link": item.get("url", ""),
                    "publication_date": pub_date
                })
        return extracted_data

    async def search_and_scrape(self, query_term: str) -> list[dict]:
        """
        Performs Brave search for a query, scrapes linked content, and returns processed items.
        Each item includes text prepared for embedding and original metadata.
        Uses MAX_PAGES from config.py to determine the maximum number of pages to scrape.
        Also uses MAX_SCRAPED_SOURCES from config.py to limit the total number of unique sources scraped.
        """
        all_extracted_content = []
        current_page_num = 1
        total_results_available = float('inf')
        
        # Use a set to track unique links encountered to avoid scraping the same link twice
        links_encountered = set() 

        # Use MAX_PAGES directly from config.py
        while current_page_num <= MAX_PAGES:
            # NEW: Break if we've already collected enough unique sources
            if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                print(f"DEBUG: Reached MAX_SCRAPED_SOURCES ({MAX_SCRAPED_SOURCES}). Stopping Brave API calls.")
                break

            offset = (current_page_num - 1) * 10

            if offset >= total_results_available:
                print(f"DEBUG: Skipping page {current_page_num} as offset {offset} is beyond total available results {total_results_available}.")
                break

            print(f"DEBUG: Fetching Brave API results for query '{query_term}', page {current_page_num}, offset {offset}...")

            brave_params = {
                "q": query_term,
                "count": 20,
                # "offset": offset, # Offset is not consistently supported across Brave API versions for pagination
                "country": "in",
                "result_filter": "web,news"
            }
            brave_headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.BRAVE_API_BASE_URL, headers=brave_headers, params=brave_params, timeout=aiohttp.ClientTimeout(total=15)) as brave_response:
                        brave_response.raise_for_status()
                        brave_results = await brave_response.json()

                if 'query' in brave_results and 'total_results' in brave_results['query']:
                    total_results_available = brave_results['query']['total_results']
                    print(f"DEBUG: Brave API reported total_results: {total_results_available}")

                page_extracted_content = self._extract_relevant_text(brave_results)

                if not page_extracted_content:
                    print(f"DEBUG: No more news/web results found for query '{query_term}' on page {current_page_num} after extraction.")
                    break

                for item in page_extracted_content:
                    link = item.get('link')
                    if link:
                        # Check if the exact link has already been encountered AND if we are within the MAX_SCRAPED_SOURCES limit
                        if link not in links_encountered and len(all_extracted_content) < MAX_SCRAPED_SOURCES:
                            all_extracted_content.append(item)
                            links_encountered.add(link) # Add to set to mark as seen
                        elif link in links_encountered:
                            print(f"DEBUG: Skipping duplicate link: {link}")
                        else: # This means len(all_extracted_content) >= MAX_SCRAPED_SOURCES
                            print(f"DEBUG: Reached MAX_SCRAPED_SOURCES ({MAX_SCRAPED_SOURCES}). Skipping link: {link}")
                            break # Break from this inner loop as we've hit the limit
                    else:
                        print(f"WARNING: Item has no link: {item.get('title', 'No Title')}. Skipping.")

                # If the inner loop broke due to MAX_SCRAPED_SOURCES, break the outer loop as well
                if len(all_extracted_content) >= MAX_SCRAPED_SOURCES:
                    break


                # Brave API pagination is often handled by 'next_page' or 'more_results_available'
                # and doesn't always rely on an explicit 'offset' parameter in subsequent requests.
                # For simplicity and to avoid issues with Brave's pagination, we'll increment
                # current_page_num and rely on the API to give us the next set of results
                # implicitly if 'more_results_available' is true and we haven't hit MAX_PAGES.
                if brave_results.get('query', {}).get('more_results_available', False) and (current_page_num * 10 < total_results_available):
                    current_page_num += 1
                    await asyncio.sleep(1) # Be kind to the API
                else:
                    print(f"DEBUG: Brave API: No more results available or reaching total results for query '{query_term}'.")
                    break

            except aiohttp.ClientError as e:
                if e.status == 422:
                    print(f"ERROR: Brave API returned 422 Unprocessable Entity for query '{query_term}' on page {current_page_num}. This often means no more results are available for this offset or the request parameters are invalid for the current state.")
                else:
                    print(f"ERROR: HTTP/Network error fetching Brave API results for query '{query_term}' on page {current_page_num} with aiohttp: {str(e)}")
                break
            except Exception as e:
                print(f"ERROR: An unexpected error occurred processing Brave API results for query '{query_term}' on page {current_page_num}: {str(e)}")
                break

        # Ensure that links_to_scrape only contains links that were actually added to all_extracted_content
        # and thus are intended to be scraped.
        links_to_scrape = [item['link'] for item in all_extracted_content if item.get('link')]
        
        scrape_tasks = [self._fetch_and_parse_url_async(link) for link in links_to_scrape]
        scraped_contents = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        processed_items = []
        # Create a dictionary for quick lookup of scraped content by link.
        scraped_content_map = {links_to_scrape[i]: scraped_contents[i] for i in range(len(links_to_scrape))}


        for item in all_extracted_content: # Iterate through the items that were selected for scraping
            link = item.get('link')
            full_webpage_content = ""

            if link and link in scraped_content_map: # Check if the link was successfully scraped
                scraped_result = scraped_content_map[link]
                if not isinstance(scraped_result, Exception):
                    full_webpage_content = scraped_result
                else:
                    print(f"WARNING: Failed to scrape content for {link} due to: {scraped_result}")
            elif not link:
                print(f"WARNING: Item has no link: {item.get('title', 'No Title')}. Skipping content processing for embedding.")
            else:
                print(f"DEBUG: Link {link} from all_extracted_content was not found in scraped_content_map. Likely due to prior scraping error.")


            text_to_embed = f"Title: {item['title']}\nSnippet: {item['snippet']}\nFull Content: {full_webpage_content}"
            tokens_to_embed = encoding.encode(text_to_embed)
            if len(tokens_to_embed) > MAX_EMBEDDING_TOKENS:
                text_to_embed = encoding.decode(tokens[:MAX_EMBEDDING_TOKENS]) + "..."

            processed_items.append({
                "text_to_embed": text_to_embed,
                "original_item": item,
                "full_webpage_content": full_webpage_content
            })
        return processed_items
