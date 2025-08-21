# aigptssh/api/dashboard/dashboard.py
import asyncio
from fastapi import APIRouter
from api.dashboard.data_aggregator import aggregate_and_process_data

router = APIRouter()

@router.get("/dashboard")
async def get_dashboard_snapshot():
    """
    Generates a market snapshot for the general dashboard.
    """
    print("--- Starting Dashboard Snapshot Generation ---")
    dashboard_content = await aggregate_and_process_data()
    print("\n--- Pipeline Complete: Final dashboard snapshot generated. ---")
    return dashboard_content