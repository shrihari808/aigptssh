# aigptssh/api/dashboard/brave_search.py
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

class BraveDashboard:
    """
    A class to fetch financial data for the Indian market using the Brave News Search API,
    specifically tailored for a financial dashboard.
    """
    BASE_URL = "https://api.search.brave.com/res/v1/news/search"  # Switched to News API endpoint
    
    # Define specific queries for each data type
    QUERIES = {
        "latest_news": "latest indian stock market news",
        "standout_gainers": "top stock market gainers in india today",
        "standout_losers": "top stock market losers in india today"
    }

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key not provided or found in environment variables.")
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

    def _perform_search(self, query, count=20, freshness="pd"):
        """
        Performs a search request to the Brave API.
        Offset has been removed to simplify and align with the new fetching strategy.
        """
        params = {
            "q": query, 
            "count": count, 
            "country": "IN", 
            "text_decorations": False,
            "freshness": freshness
        }
            
        try:
            # Note: The Web Search API is used for standouts as it provides better general results
            url = self.BASE_URL if "news" in query else "https://api.search.brave.com/res/v1/web/search"
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            return None

    def get_latest_news(self, target_count=50):
        """
        Fetches up to a target number of unique news articles in a single API call.
        """
        print(f"Fetching up to {target_count} latest news articles from News API...")
        
        # Make a single call to fetch the desired number of articles
        results = self._perform_search(self.QUERIES["latest_news"], count=target_count, freshness="pd")
        
        if not results or not results.get("results"):
            print("No news results found or API error.")
            return []

        news_items = []
        urls_seen = set()
        for item in results["results"]:
            url = item.get("url")
            if url and url not in urls_seen:
                urls_seen.add(url)
                # Fixed mapping: preserve 'description' field as is
                news_items.append({
                    "title": item.get("title"),
                    "url": url,
                    "description": item.get("description"),  # Keep as 'description', don't rename to 'snippet'
                    "page_age": item.get("page_age"),
                    "age": item.get("age")  # Also preserve the human-readable age from Brave
                })

        print(f"Successfully fetched {len(news_items)} unique news articles.")
        return news_items

    def get_portfolio_data(self, portfolio: list[str]):
        """
        Fetches news and data for a specific portfolio of stocks.
        """
        print(f"Fetching data for portfolio: {portfolio}")
        all_news = []
        urls_seen = set()

        for stock in portfolio:
            query = f"{stock} stock news"
            results = self._perform_search(query, count=10)
            if results and results.get("results"):
                for item in results["results"]:
                    url = item.get("url")
                    if url and url not in urls_seen:
                        urls_seen.add(url)
                        all_news.append({
                            "title": item.get("title"),
                            "url": url,
                            "description": item.get("description"),
                            "page_age": item.get("page_age"),
                            "age": item.get("age")
                        })
            time.sleep(1) # Respect API rate limits
        
        return {"latest_news": all_news}


    def get_standouts(self):
        """
        Fetches standout gainers and losers using the Web Search API for broader context.
        """
        print("Fetching standout gainers and losers from Web API...")
        gainers_results = self._perform_search(self.QUERIES["standout_gainers"], count=5)
        time.sleep(1)  # Respect API rate limits
        losers_results = self._perform_search(self.QUERIES["standout_losers"], count=5)

        gainers = [item.get("title") for item in gainers_results.get("web", {}).get("results", [])] if gainers_results else []
        losers = [item.get("title") for item in losers_results.get("web", {}).get("results", [])] if losers_results else []

        return {"gainers": gainers, "losers": losers}

    def get_dashboard_data(self):
        """
        Orchestrator method to fetch all necessary data for the dashboard.
        """
        print("Starting data acquisition from Brave Search API...")
        
        news = self.get_latest_news()
        time.sleep(1)
        
        standouts = self.get_standouts()
        
        dashboard_data = {
            "latest_news": news,
            "standouts": standouts
        }
        print("Brave Search API data acquisition complete.")
        return dashboard_data

# Example of how to use the class
if __name__ == '__main__':
    dashboard_fetcher = BraveDashboard()
    all_data = dashboard_fetcher.get_dashboard_data()
    
    print(f"\n--- Fetched {len(all_data['latest_news'])} News Articles ---")
    for i, news_item in enumerate(all_data['latest_news'][:3]): # Print first 3
        print(f"{i+1}. Title: {news_item['title']}\n   Link: {news_item['url']}\n   Age: {news_item['page_age']}\n   Description: {news_item['description'][:100]}...\n")
        
    print("\n--- Market Standouts ---")
    print("Gainers:", all_data['standouts']['gainers'])
    print("Losers:", all_data['standouts']['losers'])