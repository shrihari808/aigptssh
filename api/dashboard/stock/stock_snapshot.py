# aigptssh/api/dashboard/stock/stock_snapshot.py
import json
import os
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Query
from typing import List

from api.dashboard.data_aggregator import aggregate_and_process_stock_data
from api.dashboard.llm_generator import LLMGenerator, PortfolioLLMGenerator

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_DATA_JSON_PATH = os.path.join(OUTPUT_DIR, 'stock_data.json')
STOCK_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'stock_output.json')

router = APIRouter()

@router.get("/stock")
async def get_stock_snapshot(stock_name: str = Query(..., description="The name of the stock to get a snapshot for.")):
    """
    Generates a market snapshot for a given stock.
    """
    print(f"--- Starting Stock Snapshot Generation for: {stock_name} ---")

    # --- Step 1 & 2: Fetch, Scrape, and Process Data for the Stock ---
    processed_data = await aggregate_and_process_stock_data(stock_name)

    if not processed_data:
        return {"error": "Could not process data for the given stock."}

    # Save the intermediate processed data
    with open(STOCK_DATA_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    # --- Step 3: LLM Generation ---
    llm_generator = PortfolioLLMGenerator(input_path=STOCK_DATA_JSON_PATH)
    stock_dashboard_content = llm_generator.generate_dashboard_content()

    # --- Step 4: Save Final Output ---
    with open(STOCK_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(stock_dashboard_content, f, indent=4, ensure_ascii=False)

    print("\n--- Pipeline Complete: Final stock snapshot generated and saved. ---")
    return stock_dashboard_content