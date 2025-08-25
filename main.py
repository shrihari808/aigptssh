# /aigptcur/app_service/main.py
import os
import sys
import asyncpg
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import Response
import uvicorn

# --- Add the project root to the Python path ---
# This ensures that imports work correctly from anywhere in the project.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import the master API router ---
from api.router import api_router
from config import DB_POOL # Import the DB_POOL variable to make it accessible to other modules

# --- Database Connection Pool Management ---
async def init_db_pool():
    """Initializes the asyncpg connection pool."""
    global DB_POOL
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        print("INFO: Initializing database connection pool...")
        DB_POOL = await asyncpg.create_pool(
            dsn=db_url,
            min_size=1,
            max_size=10
        )
        print("INFO: Database connection pool initialized successfully.")
    else:
        print("ERROR: DATABASE_URL not set. Cannot initialize connection pool.")

async def close_db_pool():
    """Closes the asyncpg connection pool."""
    global DB_POOL
    if DB_POOL:
        print("INFO: Closing database connection pool...")
        await DB_POOL.close()
        print("INFO: Database connection pool closed.")

# --- Add Lifecycle Event Handlers for DB Pool using lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    Initializes the database pool on startup and closes it on shutdown.
    """
    print("INFO: Application startup...")
    db_url = os.getenv('DATABASE_URL')
    if db_url:
        print("INFO: Initializing database connection pool...")
        # Attach the pool to the app's state
        app.state.db_pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=1,
            max_size=10
        )
        print("INFO: Database connection pool initialized successfully.")
    else:
        app.state.db_pool = None
        print("ERROR: DATABASE_URL not set. Database pool not initialized.")

    yield # The application is now running

    print("INFO: Application shutdown...")
    if app.state.db_pool:
        print("INFO: Closing database connection pool...")
        await app.state.db_pool.close()
        print("INFO: Database connection pool closed.")

# Initialize the FastAPI application with the lifespan manager
app = FastAPI(
    title="Your AI-GPT Service",
    description="A multi-functional API for financial data analysis and content processing.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ["/web_rag", "/reddit_rag", "/yt_rag"]:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache" 
        response.headers["Expires"] = "0"
        response.headers["X-Accel-Buffering"] = "no"
    return response

# --- Include the master router ---
app.include_router(api_router)

# --- Add a Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    """A welcome message for the API root."""
    return {"message": "Welcome to your AI-GPT Service. Visit /docs for API documentation."}

# --- Main entry point for running the application ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
