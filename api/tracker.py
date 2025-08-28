# /aigptssh/api/tracker.py

import asyncio
import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
import aiohttp

from api.brave_searcher import BraveNews
from api.dashboard.web_scraper import scrape_urls
from api.fundamentals_rag.fundamental_chat2 import get_daily_ratios_async
from config import GPT4o_mini as llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from api.security import api_key_auth
# REMOVE: from main import lifespan
# REMOVE: from contextlib import asynccontextmanager

router = APIRouter()

# Global variable for the database pool, will be set by main.py
DB_POOL = None

class Contract(BaseModel):
    id: int
    company_name: str
    contract_value: int
    market_cap: int
    relative_to_market_cap: float
    impact: str
    source_url: str
    created_at: datetime

async def create_contracts_table():
    if not DB_POOL:
        print("ERROR: Database connection pool not initialized for tracker.")
        return
    async with DB_POOL.acquire() as connection:
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS contracts (
                id SERIAL PRIMARY KEY,
                company_name VARCHAR(255),
                contract_value BIGINT,
                market_cap BIGINT,
                relative_to_market_cap FLOAT,
                impact VARCHAR(50),
                source_url VARCHAR(255) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

async def extract_contract_info(text: str):
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        """
        Extract the company name and contract value from the following text.
        The output should be a JSON object with "company_name" and "contract_value" in crores of rupees. If no information is found, return null values.

        Text: {text}

        {format_instructions}
        """,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        return await chain.ainvoke({"text": text})
    except Exception:
        return None

def classify_impact(relative_to_market_cap: float) -> str:
    if relative_to_market_cap > 10:
        return 'Major'
    elif 5 <= relative_to_market_cap <= 10:
        return 'Significant'
    elif 1 <= relative_to_market_cap < 5:
        return 'Noticeable'
    else:
        return 'Minor'

async def process_contracts():
    await asyncio.sleep(5)  # Initial delay to ensure DB pool is ready
    if not DB_POOL:
        print("Scheduler waiting for DB pool...")
        return

    print("--- Starting Contract Tracker Data Aggregation ---")
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found.")
        return

    searcher = BraveNews(brave_api_key)
    queries = ["bags order", "receives order", "wins contract"]
    all_articles = []

    async with aiohttp.ClientSession(**searcher.session_config) as session:
        for query in queries:
            articles = await searcher.search_and_scrape(session, query, max_pages=1, max_sources=5)
            all_articles.extend(articles)

    unique_articles = {article['link']: article for article in all_articles}.values()
    scraped_articles = await scrape_urls(list(unique_articles))

    for article in scraped_articles:
        if article.get("content"):
            try:
                contract_info = await extract_contract_info(article["content"])
                if contract_info and contract_info.get("company_name") and contract_info.get("contract_value"):
                    company_name = contract_info["company_name"]
                    contract_value = int(float(contract_info["contract_value"])) * 1_00_00_000

                    market_cap_data = await get_daily_ratios_async(company_name)
                    if market_cap_data and isinstance(market_cap_data, dict) and "latest_data" in market_cap_data and market_cap_data["latest_data"].get("data"):
                        market_cap = int(float(market_cap_data["latest_data"]["data"][0].get("MCAP", 0))) * 1_00_00_000

                        if market_cap > 0:
                            relative_to_market_cap = (contract_value / market_cap) * 100
                            impact = classify_impact(relative_to_market_cap)

                            async with DB_POOL.acquire() as connection:
                                await connection.execute(
                                    """
                                    INSERT INTO contracts (company_name, contract_value, market_cap, relative_to_market_cap, impact, source_url)
                                    VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (source_url) DO NOTHING
                                    """,
                                    company_name, contract_value, market_cap, relative_to_market_cap, impact, article["url"]
                                )
                                print(f"Processed contract for {company_name} from {article['url']}")
            except Exception as e:
                print(f"Skipping article due to processing error: {e}")
                continue

@router.get("/tracker", response_model=List[Contract])
async def get_tracker(api_key: str = Depends(api_key_auth)):
    if not DB_POOL:
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    async with DB_POOL.acquire() as connection:
        rows = await connection.fetch("SELECT * FROM contracts ORDER BY created_at DESC")
        return [Contract(**row) for row in rows]