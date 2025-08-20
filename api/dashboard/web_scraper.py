import asyncio
import aiohttp
import trafilatura
from concurrent.futures import ThreadPoolExecutor

# Use a ThreadPoolExecutor for trafilatura as it can be CPU-bound
executor = ThreadPoolExecutor()

async def fetch_and_extract(session, url):
    """
    Asynchronously fetches a URL and extracts the main content using trafilatura.
    """
    print(f"Scraping URL: {url}")
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                # Run trafilatura in a separate thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                extracted_text = await loop.run_in_executor(
                    executor, trafilatura.extract, html
                )
                return {"url": url, "content": extracted_text or ""}
            else:
                print(f"Failed to fetch {url}: Status {response.status}")
                return {"url": url, "content": ""}
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return {"url": url, "content": ""}

async def scrape_urls(urls):
    """
    Scrapes a list of URLs concurrently.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_extract(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Example of how to use the scraper
if __name__ == '__main__':
    test_urls = [
        "https://www.moneycontrol.com/news/business/markets/stock-market-live-sensex-nifty-50-share-price-gift-nifty-latest-updates-08-08-2024-12753231.html",
        "https://timesofindia.indiatimes.com/business/india-business/stock-market-today-live-updates-sensex-nifty-share-prices-bse-nse-august-8-2024/liveblog/112345678.cms"
    ]
    
    # To run an async function from a regular script
    scraped_data = asyncio.run(scrape_urls(test_urls))
    
    for item in scraped_data:
        print(f"\n--- URL: {item['url']} ---")
        # Print first 300 characters of the content
        print(item['content'][:300] + "...")
