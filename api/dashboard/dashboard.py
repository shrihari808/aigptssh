# aigptssh/api/dashboard/dashboard.py
import json
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'dashboard_output.json')

@router.get("/dashboard")
async def get_dashboard_snapshot():
    """
    Retrieves the pre-generated market snapshot from the dashboard_output.json file.
    """
    if not os.path.exists(FINAL_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="Dashboard data is not yet available. Please try again in a few minutes.")

    return FileResponse(FINAL_OUTPUT_PATH)