# aigptssh/api/dashboard/stock/stock_snapshot.py
import json
import os
import asyncio
from fastapi import APIRouter, Query, Depends
from fastapi.responses import StreamingResponse
from api.dashboard.brave_search import BraveDashboard
from api.dashboard.web_scraper import scrape_urls
from api.dashboard.vector_store import DashboardVectorStore
from api.dashboard.scoring_service import DashboardScoringService
from api.dashboard.data_aggregator import select_latest_news_articles
from api.dashboard.llm_generator import PortfolioLLMGenerator
from datetime import datetime, timezone
from api.security import api_key_auth
# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCK_DATA_JSON_PATH = os.path.join(OUTPUT_DIR, 'stock_data.json')
STOCK_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'stock_output.json')

router = APIRouter()

def create_progress_bar_string(progress: int, message: str = "", length: int = 40):
    """Creates a visual progress bar string with a dynamic message."""
    filled_length = int(length * progress // 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # The '\r' moves the cursor to the start of the line to overwrite it.
    return f'\rProgress: |{bar}| {progress}%  {message:<50}\033[K'

@router.get("/stock")
async def get_stock_snapshot(stock_name: str = Query(..., description="The name of the stock to get a snapshot for."), api_key: str = Depends(api_key_auth)):
    """
    Generates and streams a market snapshot for a given stock, providing progress updates.
    """
    async def stream_generator():
        # ... (steps 1 and 2 remain the same) ...
        yield create_progress_bar_string(5, f"Initializing for {stock_name}...").encode("utf-8")
        brave_fetcher = BraveDashboard()
        stock_data = brave_fetcher.get_portfolio_data([stock_name])
        news_articles = stock_data.get("latest_news", [])

        yield create_progress_bar_string(20, f"Found {len(news_articles)} news articles...").encode("utf-8")
        scraped_articles = await scrape_urls(news_articles) if news_articles else []
        
        yield create_progress_bar_string(45, "Indexing and analyzing content...").encode("utf-8")
        vector_store = DashboardVectorStore(collection_name="stock_news_content")
        vector_store.add_documents(scraped_articles)
        scoring_service = DashboardScoringService(vector_store=vector_store)

        yield create_progress_bar_string(65, "Scoring relevant context...").encode("utf-8")
        stock_query = f"What is the latest news, analyst opinions, and performance data for the stock: {stock_name}?"
        context_queries = {
            "key_issues_context": stock_query,
            "indices_context": f"Provide a summary of the latest news and key events for the stock: {stock_name}.",
            "market_drivers_context": f"What were the main reasons and key driving factors for the stock: {stock_name}?"
        }
        llm_contexts = {key: scoring_service.get_enhanced_context(query, k=5 if 'key_issues' not in key else 15) for key, query in context_queries.items()}
        
        latest_news_articles = select_latest_news_articles(news_articles, count=3)
        processed_data = {
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "llm_contexts": llm_contexts,
            "latest_news_articles": latest_news_articles,
            "portfolio": [stock_name]
        }
        with open(STOCK_DATA_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)

        # --- Step 3: LLM Generation (Now fully async) ---
        yield create_progress_bar_string(85, "Generating insights with LLM...").encode("utf-8")
        llm_generator = PortfolioLLMGenerator(input_path=STOCK_DATA_JSON_PATH)
        stock_dashboard_content = await llm_generator.generate_dashboard_content()

        # --- Step 4: Save and Yield Final Output ---
        with open(STOCK_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(stock_dashboard_content, f, indent=4, ensure_ascii=False)

        yield create_progress_bar_string(100, "Done!").encode("utf-8")
        yield f"\n{json.dumps(stock_dashboard_content, indent=4)}".encode("utf-8")

    return StreamingResponse(
        stream_generator(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "X-Accel-Buffering": "no"
        }
    )