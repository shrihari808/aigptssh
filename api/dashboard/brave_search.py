# aigptssh/api/dashboard/brave_search.py
import requests
import os
from dotenv import load_dotenv
import asyncio
import re
from playwright.async_api import async_playwright
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import GPT4o_mini
import aiohttp
import sys # ADD THIS IMPORT

# Load environment variables from .env file
load_dotenv()

class BraveDashboard:
    """
    A class to fetch financial data for the Indian market using the Brave News Search API,
    specifically tailored for a financial dashboard.
    """
    BASE_URL = "https://api.search.brave.com/res/v1/news/search"  # Switched to News API endpoint
    
    # Define specific queries for each data type
    def get_queries(self, country_name="India"):
        return {
            "latest_news": f"latest {country_name} stock market news",
            "standout_gainers": f"top stock market gainers in {country_name} today",
            "standout_losers": f"top stock market losers in {country_name} today"
        }

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key not provided or found in environment variables.")
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }

    async def _perform_search_async(self, query, country="IN", count=20, freshness="pd"):
        """
        Performs an asynchronous search request to the Brave API using aiohttp.
        """
        params = {
            "q": query, 
            "count": count, 
            "country": country, 
            "text_decorations": "false",
            "freshness": freshness
        }
            
        try:
            url = self.BASE_URL if "news" in query else "https://api.search.brave.com/res/v1/news/search"
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            print(f"An error occurred during the API request: {e}")
            return None

    async def _get_reason_from_snippets_async(self, stock_name, search_results):
        """
        Uses an LLM to determine the reason for a stock's price movement from search result snippets.
        Now uses ainoke for non-blocking operation.
        """
        if not search_results or not search_results.get("web", {}).get("results"):
            return {"reason": "Could not determine the reason from search results.", "source_url": ""}

        first_result = search_results["web"]["results"][0]
        title = first_result.get("title", "")
        description = first_result.get("description", "")
        extra_snippets = first_result.get("extra_snippets", [])
        source_url = first_result.get("url", "")

        all_text = f"Title: {title}\nDescription: {description}\n" + "\n".join(extra_snippets)

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
            response = await chain.ainvoke({"stock_name": stock_name, "snippets": all_text})
            response['source_url'] = source_url
            return response
        except Exception as e:
            print(f"LLM reason extraction failed for {stock_name}: {e}")
            return {"reason": "Could not summarize the reason.", "source_url": source_url}

    async def scrape_trending_stocks(self, country_code="IN"):
        """
        Asynchronously scrapes trending stocks based on the country code.
        - For "IN", scrapes StockEdge.
        - For "US", scrapes Business Insider.
        """
        # --- ADD THIS SNIPPET ---
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # --- END OF SNIPPET ---
        if country_code == "IN":
            return await self._scrape_trending_stocks_in()
        elif country_code == "US":
            return await self._scrape_trending_stocks_us()
        else:
            print(f"Trending stocks for country '{country_code}' not supported.")
            return {"trending_stocks": []}

    async def _scrape_trending_stocks_us(self):
        """
        Scrapes trending stocks for the US from Business Insider.
        """
        print("Scraping trending US stocks from Business Insider...")
        url = "https://markets.businessinsider.com/index/market-movers/s&p_500"
        trending_stocks = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=60000)
                await page.wait_for_selector("h2.header-underline")

                # --- Scrape Gainers ---
                gainers_header = page.locator('h2:has-text("Top Gainers")')
                gainers_table = gainers_header.locator('xpath=./following-sibling::div//table')
                gainer_rows = await gainers_table.locator('tbody tr').all()
                
                for row in gainer_rows:
                    name = await row.locator("td:nth-child(1) a").inner_text()
                    percentage_change = await row.locator("td:nth-child(4) span").nth(1).inner_text()
                    trending_stocks.append({
                        "stock": name.strip(),
                        "percentage_change": percentage_change.strip(),
                        "reason": "N/A",
                        "source": ""
                    })

                # --- Scrape Losers ---
                losers_header = page.locator('h2:has-text("Top Losers")')
                losers_table = losers_header.locator('xpath=./following-sibling::div//table')
                loser_rows = await losers_table.locator('tbody tr').all()

                for row in loser_rows:
                    name = await row.locator("td:nth-child(1) a").inner_text()
                    percentage_change = await row.locator("td:nth-child(4) span").nth(1).inner_text()
                    trending_stocks.append({
                        "stock": name.strip(),
                        "percentage_change": percentage_change.strip(),
                        "reason": "N/A",
                        "source": ""
                    })

            except Exception as e:
                print(f"Error scraping Business Insider: {e}")
            finally:
                await browser.close()

        return {"trending_stocks": trending_stocks}


    async def _scrape_trending_stocks_in(self):
        """
        Asynchronously scrapes trending stocks for India from StockEdge using async playwright.
        """
        print("Scraping trending IN stocks from StockEdge...")
        
        base_url = "https://web.stockedge.com/trending-stocks?filter-type=Major%20Stocks"
        urls = {
            "Gainer": f"{base_url}&indicator=Gainers",
            "Loser": f"{base_url}&indicator=Losers"
        }

        trending_stocks = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            for indicator, url in urls.items():
                await page.goto(url, timeout=60000)
                await page.wait_for_selector("div#in\\.stockedge\\.app\\:id\\/pricemovers-stockname-div")

                name_els = await page.query_selector_all("div#in\\.stockedge\\.app\\:id\\/pricemovers-stockname-div")
                change_selector = (
                    "ion-text#in\\.stockedge\\.app\\:id\\/pricemovers-stockchgpercentage-lbl-positive-txt, "
                    "ion-text#in\\.stockedge\\.app\\:id\\/pricemovers-stockchgpercentage-lbl-negative-txt"
                )
                change_els = await page.query_selector_all(change_selector)

                for name_el, change_el in zip(name_els, change_els):
                    stock_name = (await name_el.inner_text()).strip()
                    chg_text = (await change_el.inner_text()).strip()
                    chg_clean = chg_text.replace("▲", "").replace("▼", "").replace("%", "")
                    
                    try:
                        chg_val = float(chg_clean)
                        if chg_val > 3:
                            reason_query = f"why is {stock_name} stock price {'increasing' if indicator == 'Gainer' else 'decreasing'} today"
                            search_results = await self._perform_search_async(reason_query, count=3)
                            await asyncio.sleep(1)
                            reason_data = await self._get_reason_from_snippets_async(stock_name, search_results)

                            trending_stocks.append({
                                "stock": stock_name,
                                "percentage_change": f"+{chg_val}%" if indicator == "Gainer" else f"-{chg_val}%",
                                "reason": reason_data.get("reason", "Reason not found."),
                                "source": reason_data.get("source_url", "")
                            })
                    except ValueError:
                        continue
            
            await browser.close()

        return {"trending_stocks": trending_stocks}
    # --- END OF MODIFICATION ---

    # The synchronous methods are kept for other parts of the app that might not be async yet.
    def _perform_search(self, query, count=20, freshness="pd", country="IN"):
        params = {
            "q": query, 
            "count": count, 
            "country": country, 
            "text_decorations": False,
            "freshness": freshness
        }
        try:
            url = self.BASE_URL if "news" in query else "https://api.search.brave.com/res/v1/web/search"
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the API request: {e}")
            return None
            
    async def get_latest_news(self, query, country_code, target_count=10):
        print(f"Fetching up to {target_count} latest news articles from News API for country {country_code}...")
        results = await self._perform_search_async(query, count=target_count, freshness="pd", country=country_code)
        if not results or not results.get("results"):
            print("No news results found or API error.")
            return []
        news_items = []
        urls_seen = set()
        for item in results["results"]:
            url = item.get("url")
            if url and url not in urls_seen:
                urls_seen.add(url)
                news_items.append({
                    "title": item.get("title"),
                    "url": url,
                    "description": item.get("description"),
                    "page_age": item.get("page_age"),
                    "age": item.get("age")
                })
        print(f"Successfully fetched {len(news_items)} unique news articles for {country_code}.")
        return news_items

    def get_portfolio_data(self, portfolio: list[str]):
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
            asyncio.sleep(1)
        return {"latest_news": all_news}

    async def get_dashboard_data(self, country_code="IN", country_name="India"):
        """
        Fetches dashboard data for a specific country.
        """
        print(f"Starting data acquisition for {country_name} from Brave Search API...")
        queries = self.get_queries(country_name)
        news = await self.get_latest_news(queries["latest_news"], country_code)
        dashboard_data = {
            "latest_news": news,
        }
        print(f"Brave Search API data acquisition for {country_name} complete.")
        return dashboard_data