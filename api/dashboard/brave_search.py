# aigptssh/api/dashboard/brave_search.py
import requests
import os
from dotenv import load_dotenv
import time
import re
from playwright.sync_api import sync_playwright
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import GPT4o_mini

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

    def _get_reason_from_snippets(self, stock_name, search_results):
        """
        Uses an LLM to determine the reason for a stock's price movement from search result snippets.
        """
        if not search_results or not search_results.get("web", {}).get("results"):
            return {"reason": "Could not determine the reason from search results.", "source_url": ""}

        first_result = search_results["web"]["results"][0]
        title = first_result.get("title", "")
        description = first_result.get("description", "")
        extra_snippets = first_result.get("extra_snippets", [])
        source_url = first_result.get("url", "")

        # Combine all available text
        all_text = f"Title: {title}\nDescription: {description}\n" + "\n".join(extra_snippets)

        # Use an LLM to summarize the reason
        prompt = PromptTemplate(
            template="""
            Based on the following search snippets for '{stock_name}', what is the primary reason for its recent stock price movement?
            Provide a concise, one-sentence summary.

            Snippets:
            {snippets}

            Respond in a JSON format with two keys: "reason" and "source_url".
            """,
            input_variables=["stock_name", "snippets"],
        )
        
        parser = JsonOutputParser()
        chain = prompt | GPT4o_mini | parser

        try:
            response = chain.invoke({"stock_name": stock_name, "snippets": all_text})
            response['source_url'] = source_url
            return response
        except Exception as e:
            print(f"LLM reason extraction failed for {stock_name}: {e}")
            return {"reason": "Could not summarize the reason.", "source_url": source_url}

    def scrape_trending_stocks(self):
        """
        Scrapes trending stocks from StockEdge and then uses Brave Search and an LLM to find the reason for the trend.
        """
        print("Scraping trending stocks from StockEdge...")
        
        base_url = "https://web.stockedge.com/trending-stocks?filter-type=Major%20Stocks"
        urls = {
            "Gainer": f"{base_url}&indicator=Gainers",
            "Loser": f"{base_url}&indicator=Losers"
        }

        trending_stocks = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for indicator, url in urls.items():
                page.goto(url, timeout=60000)
                page.wait_for_selector("div#in\\.stockedge\\.app\\:id\\/pricemovers-stockname-div")

                name_els = page.query_selector_all("div#in\\.stockedge\\.app\\:id\\/pricemovers-stockname-div")
                change_selector = (
                    "ion-text#in\\.stockedge\\.app\\:id\\/pricemovers-stockchgpercentage-lbl-positive-txt, "
                    "ion-text#in\\.stockedge\\.app\\:id\\/pricemovers-stockchgpercentage-lbl-negative-txt"
                )
                change_els = page.query_selector_all(change_selector)

                for name_el, change_el in zip(name_els, change_els):
                    stock_name = name_el.inner_text().strip()
                    chg_text = change_el.inner_text().strip()
                    chg_clean = chg_text.replace("▲", "").replace("▼", "").replace("%", "")
                    
                    try:
                        chg_val = float(chg_clean)
                        if chg_val > 3:
                            # Find the reason using Brave Search and LLM
                            reason_query = f"why is {stock_name} stock price {'increasing' if indicator == 'Gainer' else 'decreasing'} today"
                            search_results = self._perform_search(reason_query, count=3)
                            time.sleep(1)
                            reason_data = self._get_reason_from_snippets(stock_name, search_results)

                            trending_stocks.append({
                                "stock": stock_name,
                                "percentage_change": f"+{chg_val}%" if indicator == "Gainer" else f"-{chg_val}%",
                                "reason": reason_data.get("reason", "Reason not found."),
                                "source": reason_data.get("source_url", "")
                            })
                    except ValueError:
                        continue
            
            browser.close()

        return {"trending_stocks": trending_stocks}


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