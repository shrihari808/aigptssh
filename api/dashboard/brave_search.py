import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

class BraveDashboard:
    """
    A class to fetch financial data for the Indian market using the Brave Search API,
    specifically tailored for a financial dashboard.
    """
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
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

    def _perform_search(self, query, count=20, offset=0, freshness=None):
        """
        Performs a search request to the Brave API with pagination support.
        """
        params = {
            "q": query, 
            "count": count, 
            "offset": offset,
            "country": "IN", 
            "text_decorations": False
        }
        if freshness:
            params["freshness"] = freshness
            
        try:
            response = requests.get(self.BASE_URL, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            return None

    def get_latest_news(self, target_count=50):
        """
        Fetches up to a target number of unique news articles by paginating through search results.
        """
        print(f"Fetching up to {target_count} latest news articles...")
        news_items = []
        urls_seen = set()
        offset = 0
        
        while len(news_items) < target_count:
            results = self._perform_search(self.QUERIES["latest_news"], count=20, offset=offset, freshness="pd")
            
            if not results or "web" not in results or not results["web"].get("results"):
                print("No more results found or API error.")
                break

            for item in results["web"]["results"]:
                url = item.get("url")
                if url and url not in urls_seen:
                    urls_seen.add(url)
                    news_items.append({
                        "title": item.get("title"),
                        "url": url,
                        "description": item.get("description"),
                        "page_age": item.get("page_age") # Keep page age for recency check
                    })
                    if len(news_items) >= target_count:
                        break
            
            # As per Brave docs, increment offset by 1 for next page
            offset += 1 
            
            # Maximum offset is 9, so we can make at most 10 calls (0-9)
            if offset > 9:
                print("Reached maximum API offset. Stopping search.")
                break
            
            time.sleep(1) # Be respectful to the API

        print(f"Successfully fetched {len(news_items)} unique news articles.")
        return news_items


    def get_standouts(self):
        """
        Fetches standout gainers and losers.
        """
        print("Fetching standout gainers and losers...")
        gainers_results = self._perform_search(self.QUERIES["standout_gainers"], count=5)
        losers_results = self._perform_search(self.QUERIES["standout_losers"], count=5)

        gainers = [item.get("title") for item in gainers_results.get("web", {}).get("results", [])] if gainers_results else []
        losers = [item.get("title") for item in losers_results.get("web", {}).get("results", [])] if losers_results else []

        return {"gainers": gainers, "losers": losers}

    def get_dashboard_data(self):
        """
        Orchestrator method to fetch all necessary data for the dashboard.
        """
        print("Starting data acquisition from Brave Search API...")
        
        # We no longer need a separate summary query; it will be derived from the news context.
        news = self.get_latest_news()
        time.sleep(1) # Delay before next set of calls
        
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
        print(f"{i+1}. Title: {news_item['title']}\n   Link: {news_item['url']}\n")
        
    print("\n--- Market Standouts ---")
    print("Gainers:", all_data['standouts']['gainers'])
    print("Losers:", all_data['standouts']['losers'])
