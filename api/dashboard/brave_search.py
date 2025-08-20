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
    
    # Define specific queries for each data type (SIMPLIFIED QUERY)
    QUERIES = {
        "market_summary": "indian stock market summary today", # Simplified this query
        "latest_news": "latest indian stock market news",
        "standout_gainers": "top stock market gainers in india today",
        "standout_losers": "top stock market losers in india today"
    }

    def __init__(self, api_key=None):
        # ... (init method remains the same)
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key not provided or found in environment variables.")
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

    def _perform_search(self, query, count=20, freshness=None):
        # ... (_perform_search method remains the same)
        params = {"q": query, "count": count, "country": "IN", "text_decorations": False}
        if freshness:
            params["freshness"] = freshness
            
        try:
            response = requests.get(self.BASE_URL, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            return None

    def get_market_summary_points(self):
        """
        Fetches a wide range of search results and extracts descriptions to form
        a market summary.
        """
        print("Fetching data for market summary...")
        # REDUCED COUNT to 20 to avoid 422 error
        results = self._perform_search(self.QUERIES["market_summary"], count=20)
        if not results or "web" not in results or "results" not in results["web"]:
            return ["Market summary data could not be retrieved."]
        
        summary_points = [
            item.get("description", "") 
            for item in results["web"]["results"] 
            if item.get("description")
        ]
        return list(dict.fromkeys(summary_points))[:7]

    def get_latest_news(self):
        # ... (get_latest_news method remains the same)
        print("Fetching latest news...")
        results = self._perform_search(self.QUERIES["latest_news"], count=10, freshness="pd")
        if not results or "web" not in results or "results" not in results["web"]:
            return [{"title": "Latest news could not be retrieved.", "url": "#", "description": ""}]
        
        news_items = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description")
            }
            for item in results["web"]["results"]
        ]
        return news_items

    def get_standouts(self):
        # ... (get_standouts method remains the same)
        print("Fetching standout gainers and losers...")
        gainers_results = self._perform_search(self.QUERIES["standout_gainers"], count=5)
        losers_results = self._perform_search(self.QUERIES["standout_losers"], count=5)

        gainers = [item.get("title") for item in gainers_results.get("web", {}).get("results", [])] if gainers_results else []
        losers = [item.get("title") for item in losers_results.get("web", {}).get("results", [])] if losers_results else []

        return {"gainers": gainers, "losers": losers}

    def get_dashboard_data(self):
        """
        Orchestrator method to fetch all necessary data for the dashboard.
        Includes a longer delay to prevent hitting API rate limits.
        """
        print("Starting data acquisition from Brave Search API...")
        
        summary = self.get_market_summary_points()
        time.sleep(2) 
        
        news = self.get_latest_news()
        time.sleep(2)
        
        standouts = self.get_standouts()
        
        dashboard_data = {
            "market_summary": summary,
            "latest_news": news,
            "standouts": standouts
        }
        print("Brave Search API data acquisition complete.")
        return dashboard_data

# Example of how to use the class
if __name__ == '__main__':
    # Make sure you have a .env file with your BRAVE_API_KEY
    # or pass it directly to the constructor.
    # For example: dashboard_fetcher = BraveDashboard(api_key="your_key_here")
    
    dashboard_fetcher = BraveDashboard()
    all_data = dashboard_fetcher.get_dashboard_data()
    
    print("\n--- Market Summary ---")
    for point in all_data['market_summary']:
        print(f"- {point}")
        
    print("\n--- Latest News ---")
    for news in all_data['latest_news']:
        print(f"Title: {news['title']}\n  Link: {news['url']}\n")
        
    print("\n--- Market Standouts ---")
    print("Gainers:", all_data['standouts']['gainers'])
    print("Losers:", all_data['standouts']['losers'])
