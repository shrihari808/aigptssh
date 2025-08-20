import asyncio
import aiohttp
import trafilatura
from concurrent.futures import ThreadPoolExecutor

async def fetch_and_extract(session, article, executor):
    """
    Asynchronously fetches a URL from an article object and extracts the main content.
    The article's metadata is preserved.
    """
    url = article.get("url")
    if not url:
        print("Skipping article with no URL.")
        return article  # Return original article if no URL

    print(f"Scraping URL: {url}")
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                # Run trafilatura in the executor to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                extracted_text = await loop.run_in_executor(
                    executor, trafilatura.extract, html
                )
                article["content"] = extracted_text or ""
                return article
            else:
                print(f"Failed to fetch {url}: Status {response.status}")
                article["content"] = ""
                return article
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        article["content"] = ""
        return article

async def scrape_urls(articles):
    """
    Scrapes a list of URLs from article objects concurrently, managing the
    ThreadPoolExecutor's lifecycle properly.
    """
    # Create and manage the executor within the async function
    with ThreadPoolExecutor() as executor:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_and_extract(session, article, executor) for article in articles]
            results = await asyncio.gather(*tasks)
            return results

# Example of how to use the scraper
if __name__ == '__main__':
    test_articles = [
        {
            "url": "https://www.moneycontrol.com/news/business/markets/stock-market-live-sensex-nifty-50-share-price-gift-nifty-latest-updates-08-08-2024-12753231.html",
            "page_age": "2025-08-18T10:00:00Z"
        },
        {
            "url": "https://timesofindia.indiatimes.com/business/india-business/stock-market-today-live-updates-sensex-nifty-share-prices-bse-nse-august-8-2024/liveblog/112345678.cms",
            "page_age": "2025-08-19T12:00:00Z"
        }
    ]
    
    # To run an async function from a regular script
    scraped_data = asyncio.run(scrape_urls(test_articles))
    
    for item in scraped_data:
        print(f"\n--- URL: {item['url']} (Age: {item['page_age']}) ---")
        # Print first 300 characters of the content
        print(item.get('content', '')[:300] + "...")