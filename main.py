# /aigptcur/app_service/main.py
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Add the project root to the Python path ---
# This ensures that imports work correctly from anywhere in the project.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import only the single, master API router ---
# All individual endpoints are now managed within api/router.py
from api.router import api_router

# Initialize the FastAPI application with explicit doc URLs for robustness
app = FastAPI(
    title="Your AI-GPT Service",
    description="A multi-functional API for financial data analysis and content processing.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Add CORS Middleware ---
# This allows web pages from any origin to access your API, which is
# useful for building a separate frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Include the master router ---
# The application now has a single, clean entry point for all API routes.
app.include_router(api_router)

# --- Add a Root Endpoint ---
# This function handles requests to the base URL (e.g., http://localhost:8000/)
@app.get("/", tags=["Root"])
async def read_root():
    """A welcome message for the API root."""
    return {"message": "Welcome to your AI-GPT Service. Visit /docs for API documentation."}


# --- Main entry point for running the application ---
if __name__ == "__main__":
    # Use uvicorn to run the app. The --reload flag is useful for development.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
