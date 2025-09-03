# aigptssh/api/dashboard/trending.py
import json
import os
import time
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from api.security import api_key_auth
from api.dashboard.brave_search import BraveDashboard 

router = APIRouter()

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

@router.get("/dashboard/trending")
async def get_trending_stocks(
    country: str = Query("IN", description="The country code for trending stocks (e.g., IN, US)."),
    api_key: str = Depends(api_key_auth)
):
    """
    Retrieves trending stocks. If a fresh cached file exists, it's served.
    Otherwise, it scrapes for new data on-demand.
    """
    country_code = country.upper()
    trending_output_path = os.path.join(OUTPUT_DIR, f'trending_stocks_{country_code}.json')

    if os.path.exists(trending_output_path):
        file_mod_time = os.path.getmtime(trending_output_path)
        if (time.time() - file_mod_time) < 3600: # 1 hour
            return FileResponse(trending_output_path)

    # On-demand generation
    brave_fetcher = BraveDashboard()
    # This method needs to be updated to accept a country code
    trending_data = await brave_fetcher.scrape_trending_stocks(country_code) 

    with open(trending_output_path, 'w', encoding='utf-8') as f:
        json.dump(trending_data, f, indent=4, ensure_ascii=False)

    return FileResponse(trending_output_path)