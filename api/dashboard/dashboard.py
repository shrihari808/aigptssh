# aigptssh/api/dashboard/dashboard.py
import json
import os
import asyncio
import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse, StreamingResponse
from api.security import api_key_auth
from api.dashboard.data_aggregator import aggregate_and_process_data

router = APIRouter()

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# A simple mapping for country names
COUNTRY_NAMES = {
    "IN": "India",
    "US": "USA",
    "GB": "UK"
    # Add more as needed
}

def get_country_name(country_code):
    return COUNTRY_NAMES.get(country_code.upper(), country_code)

@router.get("/dashboard")
async def get_dashboard_snapshot(
    country: str = Query("IN", description="The country code for the dashboard (e.g., IN, US)."),
    api_key: str = Depends(api_key_auth)
):
    """
    Retrieves the market snapshot. If a pre-generated version exists and is fresh,
    it's returned instantly. Otherwise, it generates a new snapshot on-demand with streaming progress.
    """
    country_code = country.upper()
    country_name = get_country_name(country_code)
    final_output_path = os.path.join(OUTPUT_DIR, f'dashboard_output_{country_code}.json')
    
    # Check for a fresh, cached file
    if os.path.exists(final_output_path):
        # Check file age (e.g., if it's less than 15 minutes old)
        file_mod_time = os.path.getmtime(final_output_path)
        if (time.time() - file_mod_time) < 900: # 15 minutes
             return FileResponse(final_output_path)

    # --- On-demand generation with streaming ---
    async def stream_generator():
        yield "Starting on-demand generation...\n"
        try:
            # This function now needs to be adapted to yield progress,
            # but for now, we'll just call the main aggregation function
            # and stream the final result. A more advanced implementation
            # would have `aggregate_and_process_data` yield its progress.
            await aggregate_and_process_data(country_code, country_name)
            
            with open(final_output_path, 'r', encoding='utf-8') as f:
                final_json = f.read()
            
            yield f"data: {final_json}\n\n"
        except Exception as e:
            yield f"error: {str(e)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")