# aigptssh/api/dashboard/portfolio_snapshot.py
import json
import os
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Query
from typing import List

from api.dashboard.data_aggregator import aggregate_and_process_portfolio_data
from api.dashboard.llm_generator import LLMGenerator

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_DATA_JSON_PATH = os.path.join(OUTPUT_DIR, 'portfolio_data.json')
PORTFOLIO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'portfolio_output.json')

router = APIRouter()

@router.get("/dashboard/portfolio")
async def get_portfolio_snapshot(portfolio: List[str] = Query(..., description="A list of stock tickers in the user's portfolio.")):
    """
    Generates a market snapshot for a given portfolio of stocks.
    """
    print(f"--- Starting Portfolio Snapshot Generation for: {portfolio} ---")

    # --- Step 1 & 2: Fetch, Scrape, and Process Data for the Portfolio ---
    processed_data = await aggregate_and_process_portfolio_data(portfolio)

    if not processed_data:
        return {"error": "Could not process data for the given portfolio."}

    # Save the intermediate processed data
    with open(PORTFOLIO_DATA_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    # --- Step 3: LLM Generation ---
    llm_generator = LLMGenerator(input_path=PORTFOLIO_DATA_JSON_PATH)
    portfolio_dashboard_content = llm_generator.generate_dashboard_content()

    # --- Step 4: Save Final Output ---
    with open(PORTFOLIO_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(portfolio_dashboard_content, f, indent=4, ensure_ascii=False)

    print("\n--- Pipeline Complete: Final portfolio snapshot generated and saved. ---")
    return portfolio_dashboard_content