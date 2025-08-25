# aigptssh/api/dashboard/trending.py
import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRENDING_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'trending_stocks.json')

@router.get("/dashboard/trending")
async def get_trending_stocks():
    """
    Retrieves the pre-generated trending stocks from the trending_stocks.json file.
    """
    if not os.path.exists(TRENDING_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="Trending stocks data is not yet available. Please try again in a few minutes.")

    return FileResponse(TRENDING_OUTPUT_PATH)