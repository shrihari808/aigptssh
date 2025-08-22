# aigptssh/api/dashboard/trending.py
from fastapi import APIRouter
from api.dashboard.brave_search import BraveDashboard
import re
import time
import json # Add this import for your test script

router = APIRouter()

@router.get("/dashboard/trending")
async def get_trending_stocks():
    """
    Generates a list of trending stocks for the day.
    """
    print("--- Starting Trending Stocks Generation ---")
    brave_fetcher = BraveDashboard()
    trending_stocks = brave_fetcher.scrape_trending_stocks()
    print("\n--- Pipeline Complete: Trending stocks identified. ---")
    return trending_stocks

if __name__ == '__main__':
    trending_stocks = BraveDashboard().scrape_trending_stocks()
    print(f"\n--- Trending Stocks ---\n{trending_stocks}")
    # Save the output to a JSON file for analysis
    with open('trending_stocks.json', 'w', encoding='utf-8') as f:
        json.dump(trending_stocks, f, indent=4, ensure_ascii=False)