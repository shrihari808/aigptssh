import os
import json
import tiktoken
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, APIRouter, Request, Depends, HTTPException, Query
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from pinecone import PodSpec, Pinecone as PineconeClient
import requests
from pydantic import BaseModel
from typing import Any
import re
import asyncpg
from langchain.retrievers import MergerRetriever
from langchain.docstore.document import Document
from langchain_core.documents import Document
from datetime import datetime, timedelta
from langchain_pinecone import Pinecone
import google.generativeai as genai
from langchain import PromptTemplate, LLMChain
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ContextualCompressionRetriever
from psycopg2 import sql
from urllib.parse import urlparse
from contextlib import contextmanager
import time
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse
import pandas as pd
import aiohttp

# --- Local Project Imports ---
from config import (
    chroma_server_client, llm_date, llm_stream, vs, GPT4o_mini,
    PINECONE_INDEX_NAME, CONTEXT_SUFFICIENCY_THRESHOLD, ENABLE_CACHING
)
from api.news_rag.caching_service import (
    query_session_cache,
    add_passages_to_cache
)
from langchain_core.documents import Document
from api.brave_searcher import BraveNews, get_brave_results
from api.reddit_scraper import RedditScraper
from api.reddit_rag.reddit_vector_store import create_reddit_vector_store_from_scraped_data

# --- Functions imported from other modules ---
from streaming.reddit_stream import fetch_search_red, process_search_red
from streaming.yt_stream import get_data, get_yt_data_async
from api.news_rag.scoring_service import scoring_service
from api.youtube_rag.youtube_vector_store import create_yt_vector_store_from_transcripts
from api.security import api_key_auth


from dotenv import load_dotenv
load_dotenv(override=True)

# --- Environment Variable Loading ---
openai_api_key = os.getenv('OPENAI_API_KEY')
pg_ip = os.getenv('PG_IP_ADDRESS')
pine_api = os.getenv('PINECONE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
psql_url = os.getenv('DATABASE_URL')
node_key = os.getenv('node_key')

# --- API Routers ---
cmots_rag = APIRouter()
web_rag = APIRouter()
red_rag = APIRouter()
yt_rag = APIRouter()

# --- Dependency for Database Pool ---
async def get_db_pool(request: Request) -> asyncpg.Pool:
    """FastAPI dependency to get the database pool from the application state."""
    if not hasattr(request.app.state, 'db_pool') or not request.app.state.db_pool:
        print("CRITICAL ERROR: Database pool is not available on app state.")
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    return request.app.state.db_pool

async def quick_scrape_and_process(query: str, db_pool: asyncpg.Pool, num_urls: int = 3):
    """
    Tier 1 Helper: Correctly performs a limited search for a few URLs.
    """
    print(f"DEBUG: Tier 1 - Starting quick scrape for {num_urls} URLs.")
    try:
        # FIX: Pass the limits to get_brave_results for a truly quick search
        articles, df = await get_brave_results(query, max_pages=1, max_sources=num_urls)
        if not articles:
            return []

        if df is not None and not df.empty:
            asyncio.create_task(insert_post1(df, db_pool))

        passages = [{
            "text": f"Title: {a.get('title', '')}\nDescription: {a.get('description', '')}",
            "metadata": {
                "title": a.get('title'),
                "link": a.get('source_url'),
                "publication_date": a.get('source_date'),
                "snippet": a.get('description')
            }
        } for a in articles]

        print(f"DEBUG: Tier 1 - Quick scrape completed with {len(passages)} passages.")
        return passages
    except Exception as e:
        print(f"ERROR in quick_scrape_and_process: {e}")
        return []

def diversify_results(passages: list[dict], max_per_source: int = 2) -> list[dict]:
    """
    Enhanced diversification that properly extracts domains from passage metadata.
    """
    source_counts = {}
    diversified_list = []

    # Sort passages by score to process the best ones first
    passages.sort(key=lambda x: x.get('final_combined_score', 0), reverse=True)

    for passage in passages:
        # Try multiple ways to get the source URL
        source_link = None
        metadata = passage.get("metadata", {})

        # Check different possible URL fields
        source_link = (metadata.get("link") or
                      metadata.get("source_url") or
                      metadata.get("url") or
                      "unknown")

        try:
            if source_link and source_link != "unknown":
                from urllib.parse import urlparse
                domain = urlparse(source_link).netloc.replace('www.', '')
            else:
                domain = "unknown"
        except:
            domain = "unknown"

        # Enhanced logic: prefer first chunks from each source
        chunk_position = passage.get('chunk_position', 0)

        current_count = source_counts.get(domain, 0)

        # Prioritize first chunks and limit per source
        should_include = False
        if current_count < max_per_source:
            if current_count == 0:  # Always include first item from each source
                should_include = True
            elif chunk_position <= 2:  # Only include early chunks for additional items
                should_include = True

        if should_include:
            diversified_list.append(passage)
            source_counts[domain] = current_count + 1

    print(f"DEBUG: Diversified passages from {len(passages)} to {len(diversified_list)}")
    print(f"DEBUG: Sources included: {list(source_counts.keys())}")
    return diversified_list

async def quick_brave_search_for_snippets(articles: list) -> list:
    """
    Processes a list of articles to create snippets for the preliminary summary.
    This function NO LONGER performs its own web search.
    """
    print("DEBUG: Generating snippets from existing search results.")
    if not articles:
        return []

    # Create snippets from the provided articles
    snippets = [
        f"Title: {a.get('title', '')}\nSnippet: {a.get('description', '')}\nSource: {a.get('source_url', '')}"
        for a in articles[:3]  # Use the top 3 for the preliminary summary
    ]
    return snippets

# --- Optimized Chat History Functions ---

async def get_chat_history_optimized(session_id: str, db_pool: asyncpg.Pool, limit: int = 3):
    """
    Optimized chat history retrieval with reduced context and better caching.
    Returns only the last 'limit' messages for efficiency.
    """
    try:
        async with db_pool.acquire() as conn:
            s_id = str(session_id)
            # Get only the last few messages, ordered by timestamp desc
            rows = await conn.fetch("""
                SELECT message FROM message_store
                WHERE session_id = $1
                ORDER BY id DESC
                LIMIT $2
            """, s_id, limit)

            if not rows:
                return []

            # Parse messages and reverse to get chronological order
            messages = []
            for row in reversed(rows):  # Reverse to get chronological order
                try:
                    message_data = json.loads(row['message'])
                    content = message_data.get('data', {}).get('content', '')
                    if content.strip():  # Only add non-empty messages
                        messages.append(content.strip())
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse message: {e}")
                    continue

            return messages

    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def extract_date_robust(query: str, today: str) -> tuple[str, str]:
    """
    Robust date extraction using regex patterns and logical rules.
    Returns (extracted_date, cleaned_query).
    """
    query_lower = query.lower().strip()
    cleaned_query = query

    try:
        today_dt = datetime.strptime(today, "%Y-%m-%d")
    except ValueError:
        today_dt = datetime.now()

    # Pattern matching with extraction
    date_patterns = [
        # Relative dates
        (r'\btoday\b', 0, 'today'),
        (r'\byesterday\b', -1, 'yesterday'),
        (r'\btomorrow\b', 1, 'tomorrow'),

        # Recent/latest patterns
        (r'\b(recent|latest|current)\b.*\bnews\b', -7, 'recent'),
        (r'\b(recent|latest|trending)\b', -1, 'trending'),

        # Specific time periods
        (r'\blast\s+(\d+)\s+days?\b', lambda m: -int(m.group(1)), 'last_days'),
        (r'\b(\d+)\s+days?\s+ago\b', lambda m: -int(m.group(1)), 'days_ago'),
        (r'\blast\s+week\b', -7, 'last_week'),
        (r'\blast\s+month\b', -30, 'last_month'),

        # Specific date formats
        (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'specific_date', 'iso_date'),
        (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 'specific_date', 'us_date'),
        (r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', 'specific_date', 'dash_date'),
    ]

    extracted_date = "None"

    for pattern, offset_or_handler, pattern_type in date_patterns:
        match = re.search(pattern, query_lower)
        if match:
            if pattern_type in ['specific_date']:
                # Handle specific date formats
                if pattern_type == 'iso_date':
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                elif pattern_type in ['us_date', 'dash_date']:
                    month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))

                try:
                    specific_date = datetime(year, month, day)
                    extracted_date = specific_date.strftime("%Y%m%d")
                except ValueError:
                    continue  # Invalid date, try next pattern

            elif callable(offset_or_handler):
                # Handle dynamic offsets (e.g., "last 5 days")
                days_offset = offset_or_handler(match)
                target_date = today_dt + timedelta(days=days_offset)
                extracted_date = target_date.strftime("%Y%m%d")

            else:
                # Handle fixed offsets
                target_date = today_dt + timedelta(days=offset_or_handler)
                extracted_date = target_date.strftime("%Y%m%d")

            # Clean the matched pattern from query
            cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query)  # Remove extra spaces

            break  # Use first match found

    # Special handling for quarterly/annual requests
    if re.search(r'\b(quarter|quarterly|q[1-4]|annual|yearly)\b', query_lower):
        extracted_date = "None"

    # If no date found but asking about "news" without time context, assume recent
    if extracted_date == "None" and re.search(r'\bnews\b', query_lower) and not re.search(r'\b(upcoming|future|next|will)\b', query_lower):
        target_date = today_dt + timedelta(days=-3)  # Last 3 days for general news
        extracted_date = target_date.strftime("%Y%m%d")

    return extracted_date, cleaned_query

def is_followup_question(query: str, chat_history: list[str]) -> bool:
    """
    Determine if the query is a follow-up question that needs memory context.
    """
    if not chat_history:
        return False

    query_lower = query.lower().strip()

    # Clear indicators of follow-up questions
    followup_indicators = [
        # Pronouns and references
        r'\b(it|this|that|they|them|its|their)\b',
        r'\b(the company|the stock|the news)\b',
        r'\b(what about|how about|tell me more)\b',

        # Continuation words
        r'\b(also|additionally|furthermore|moreover)\b',
        r'\b(and what|what else|anything else)\b',

        # Comparative references
        r'\b(compared to|versus|vs\.?|difference)\b',
        r'\b(better than|worse than|similar to)\b',

        # Context-dependent questions
        r'\b(why|how|when|where|who)\b.*\b(this|that|it)\b',
        r'\bwhat.*\b(impact|effect|result|outcome)\b',

        # Short questions that likely need context
        r'^\b(yes|no|ok|sure|thanks?|please)\b',
        r'^\w{1,3}\?$',  # Very short questions like "Why?", "How?"
    ]

    # Check for follow-up indicators
    for indicator in followup_indicators:
        if re.search(indicator, query_lower):
            return True

    # If query is very short and chat history exists, likely a follow-up
    if len(query_lower.split()) <= 3 and chat_history:
        return True

    # Check if query contains company/stock names mentioned in recent history
    recent_history = ' '.join(chat_history[-2:]).lower()  # Last 2 messages
    query_words = set(query_lower.split())
    history_words = set(recent_history.split())

    # If significant word overlap, might be follow-up
    overlap = query_words.intersection(history_words)
    if len(overlap) >= 2:  # At least 2 common words
        return True

    return False

# --- Combined LLM Processing Function ---

async def combined_preprocessing(query: str, chat_history: list[str], today: str) -> dict:
    """
    Combined LLM call that handles validation, memory contextualization, and processing.
    Only uses memory chain for follow-up questions.
    """

    # First, do robust date extraction (no LLM needed)
    extracted_date, cleaned_query = extract_date_robust(query, today)

    # Check if this is a follow-up question
    is_followup = is_followup_question(query, chat_history)

    # Prepare the combined prompt
    if is_followup and chat_history:
        # For follow-up questions, include memory contextualization
        combined_prompt = """
You are a financial assistant for the Indian stock market. Your primary function is to analyze user queries and prepare them for a financial search engine.

User Query: "{query}"
Today's Date: {today}

Tasks:
1. VALIDATE: Is this query related to the Indian stock market, business, or finance?
2. INTERPRET: Interpret ambiguous acronyms or terms (e.g., "HCC", "BEL") as company names or stock tickers within the Indian market context.
3. REFORMULATE: Create a clear, standalone search query suitable for a financial news search. For ambiguous tickers like "HCC," reformulate the query to "HCC Ltd. stock" to ensure financial context.

Return JSON format:
{{
    "valid": 1 or 0,
    "reformulated_query": "financially-focused search query",
    "is_followup": false,
    "needs_memory": false
}}

Guidelines:
- For validation: 1 = valid query related to Indian markets/business/finance, 0 = not related.
- If the user asks why a company is trending, the reformulated query should be specific, like "Why is [Company Name] stock price trending".
"""

        input_data = {
            "query": query,
            "chat_history": chat_history[-3:],  # Last 3 messages only
            "today": today
        }

    else:
        # For standalone questions, skip memory contextualization
        combined_prompt = """
You are a financial assistant. Analyze this user query and provide a JSON response.

User Query: "{query}"
Today's Date: {today}

Tasks:
1. VALIDATE: Is this query related to the Indian stock market, Indian business, or Indian finance?
2. PROCESS: Clean the query and ensure it's grammatically correct.

Return JSON format:
{{
    "valid": 1 or 0,
    "reformulated_query": "{query}",
    "is_followup": false,
    "needs_memory": false
}}

Guidelines:
- If asking for "latest news" about any company, consider it valid.
- If asking about "current news" or "trending news", consider it valid.
- For validation: 1 = valid query related to Indian markets/business/finance, 0 = not related.
- Keep reformulated_query the same as the original for standalone questions.
"""

        input_data = {
            "query": query,
            "today": today
        }

    # Create the LLM chain
    prompt_template = PromptTemplate(
        template=combined_prompt,
        input_variables=list(input_data.keys())
    )

    chain = prompt_template | GPT4o_mini | JsonOutputParser()

    try:
        # Single LLM call to handle everything
        with get_openai_callback() as cb:
            result = await chain.ainvoke(input_data)

        # Add our robust date extraction and token count
        result["extracted_date"] = extracted_date
        result["cleaned_query"] = cleaned_query
        result["tokens_used"] = cb.total_tokens

        print(f"DEBUG: Combined preprocessing completed in single LLM call")
        print(f"DEBUG: Valid: {result.get('valid')}, Follow-up: {result.get('is_followup')}")
        print(f"DEBUG: Date extracted: {extracted_date}")
        print(f"DEBUG: Tokens used: {cb.total_tokens}")

        return result

    except Exception as e:
        print(f"ERROR in combined_preprocessing: {e}")
        # Fallback response
        return {
            "valid": 1,  # Assume valid to be safe
            "reformulated_query": query,
            "is_followup": is_followup,
            "needs_memory": is_followup,
            "extracted_date": extracted_date,
            "cleaned_query": cleaned_query,
            "tokens_used": 0
        }

# --- Database Functions (Optimized) ---

async def insert_post1(df: pd.DataFrame, db_pool: asyncpg.Pool):
    """Optimized bulk insert with better error handling."""
    if df.empty:
        return

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            # Batch check for existing URLs
            urls_to_check = df['source_url'].tolist()
            existing_urls = await conn.fetch(
                "SELECT source_url FROM source_data WHERE source_url = ANY($1)",
                urls_to_check
            )
            existing_urls_set = {row['source_url'] for row in existing_urls}

            # Prepare new rows for bulk insert
            new_rows = []
            for _, row in df.iterrows():
                if row['source_url'] not in existing_urls_set:
                    new_rows.append((
                        row['source_url'], row.get('image_url'), row['heading'],
                        row['title'], row['description'], row['source_date']
                    ))

            if new_rows:
                await conn.executemany("""
                    INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, new_rows)
                print(f"DEBUG: Bulk inserted {len(new_rows)} new rows")

async def insert_red(df: pd.DataFrame, db_pool: asyncpg.Pool):
    """Optimized Reddit data insertion."""
    if df.empty:
        return

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            for _, row in df.iterrows():
                exists = await conn.fetchval("SELECT 1 FROM source_data WHERE source_url = $1", row['source_url'])
                if not exists:
                    await conn.execute("""
                        INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, row['source_url'], None, None, row['title'], row['description'], row['source_date'])

async def insert_credit_usage(user_id: int, plan_id: int, credit_used: float, db_pool: asyncpg.Pool):
    """Optimized credit usage insertion."""
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO credit_usage (user_id, plan_id, credit_used, "createdAt", "updatedAt")
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, plan_id, credit_used, current_time, current_time)

async def store_into_db(pid: int, ph_id: int, result_json: dict, db_pool: asyncpg.Pool):
    """Optimized data storage."""
    result_json_str = json.dumps(result_json)
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO "streamingData" (prompt_id, prompt_history_id, source_data)
            VALUES ($1, $2, $3)
        """, pid, ph_id, result_json_str)

# --- Helper Functions ---

def count_tokens(text, model_name="gpt-4o-mini"):
    """Optimized token counting."""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def split_input(input_string):
    """Legacy function - kept for compatibility."""
    parts = input_string.split(',', 1)
    date = parts[0].strip()
    general_user_query = parts[1].strip() if len(parts) > 1 else ""
    return date, general_user_query

def create_progress_bar_string(progress: int, message: str = "", length: int = 40):
    """Creates a visual progress bar string with a dynamic message."""
    filled_length = int(length * progress // 100)
    bar = '█' * filled_length + '-' * (length - filled_length)
    # The '\r' moves the cursor to the start of the line to overwrite it.
    # We also add padding and clear the rest of the line to prevent artifacts.
    return f'\rProgress: |{bar}| {progress}%  {message:<50}\033[K'

# --- API Endpoints ---

class InRequest(BaseModel):
    query: str

# In streaming/streaming.py
use_caching = ENABLE_CACHING
@web_rag.post("/web_rag")
async def web_rag_mix(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    api_key: str = Depends(api_key_auth)
):
    """
    REPLACED: Final version with corrected session management.
    Implements the full, multi-stage "search and re-rank" strategy.
    """
    original_query = request.query.strip()
    today = datetime.now().strftime("%Y-%m-%d")

    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        raise HTTPException(status_code=500, detail="Brave API key not configured.")

    brave_searcher = BraveNews(brave_api_key)

    async def tiered_stream_generator():
        # --- Preprocessing Step ---
        chat_history = await get_chat_history_optimized(str(session_id), db_pool, limit=3)
        preprocessing_result = await combined_preprocessing(original_query, chat_history, today)

        if preprocessing_result.get("valid", 0) == 0:
            yield "I am a financial markets search engine and can only answer questions related to Indian markets, business, and finance. Please ask a relevant question.".encode("utf-8")
            return

        query = preprocessing_result.get("reformulated_query", original_query)  # Use a new local variable for the potentially reformulated query

        # --- Caching Logic Start ---
        cached_passages = []
        if use_caching:
            if is_followup_question(query, chat_history):
                cached_docs_with_scores = await asyncio.to_thread(
                    query_session_cache, str(session_id), query
                )

                sufficiency_score = await asyncio.to_thread(
                    scoring_service.assess_context_sufficiency, query, cached_docs_with_scores
                )

                if sufficiency_score > 0.4:
                    print("DEBUG: Sufficient context found in cache. Bypassing web scrape.")
                    for doc, score in cached_docs_with_scores:
                        cached_passages.append({
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "final_combined_score": score
                        })

        if cached_passages:
            final_passages = cached_passages
        else:
            async with aiohttp.ClientSession(**brave_searcher.session_config) as session:
                initial_sources = await brave_searcher.search_and_scrape(session, query, max_sources=30)
                if not initial_sources:
                    yield "\nCould not find any initial sources.".encode("utf-8")
                    return

                yield f"& Reading sources | {len(initial_sources)} articles\n".encode("utf-8")
                for source in initial_sources:
                    yield f"{json.dumps({'title': source.get('title'), 'url': source.get('link')})}\n".encode("utf-8")

                yield "& Filtering Content ...\n".encode("utf-8")
                
                sources_to_scrape = initial_sources[:10]

                scraped_sources = []
                total_to_scrape = len(sources_to_scrape)

                for i, source in enumerate(sources_to_scrape):
                    scraped = await brave_searcher.scrape_top_urls(session, [source])
                    if scraped:
                        scraped_sources.extend(scraped)
                    await asyncio.sleep(0.1)

                yield "& Re-ranking context ...\n".encode("utf-8")

                final_passages = await scoring_service.rerank_content_chunks(query, scraped_sources, top_n=7)
                if not final_passages:
                    yield "\nCould not extract sufficient detailed information.".encode("utf-8")
                    return

                if use_caching:
                    await asyncio.to_thread(add_passages_to_cache, str(session_id), final_passages)

        yield "& Creating enhanced context ...\n".encode("utf-8")

        final_context = await asyncio.to_thread(scoring_service.create_enhanced_context, final_passages)
        final_links = list(set([p["metadata"].get("link") for p in final_passages if p.get("metadata", {}).get("link")]))

        final_prompt = PromptTemplate.from_template(
            """
            You are a financial markets expert. Today's date is {today}. Provide a detailed, well-structured final answer using the comprehensive context provided.
            Use markdown for readability and cite the source links where appropriate. Provide the source links with their citation numbers at the end of the response.

            **CRITICAL INSTRUCTION:** Focus exclusively on financial, startup, corporate, and stock market-related information.

            Comprehensive Context:
            {context}

            Chat History:
            {history}

            User Question: {input}

            Final Detailed Answer:
            """
        )
        final_chain = final_prompt | llm_stream

        yield "& Thinking ...\n".encode("utf-8")

        final_response_text = ""
        with get_openai_callback() as cb:
            async for chunk in final_chain.astream({"context": final_context, "history": chat_history, "input": query, "today": today}):
                if chunk.content:
                    final_response_text += chunk.content
                    yield chunk.content.encode("utf-8")

            if not cached_passages:
                df_to_insert = await asyncio.to_thread(brave_searcher._process_for_dataframe, scraped_sources)
                if not df_to_insert.empty:
                    asyncio.create_task(insert_post1(df_to_insert, db_pool))

            total_tokens = cb.total_tokens
            asyncio.create_task(insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool))
            asyncio.create_task(store_into_db(session_id, prompt_history_id, {"links": final_links}, db_pool))

            if final_response_text:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                await asyncio.to_thread(history_db.add_user_message, original_query)
                await asyncio.to_thread(history_db.add_ai_message, final_response_text)

    return StreamingResponse(
        tiered_stream_generator(),
        media_type="text/plain",  # Change from "text/event-stream" to "text/plain"
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@cmots_rag.post("/cmots_rag")
async def cmots_only(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    api_key: str = Depends(api_key_auth)
):
    """Optimized CMOTS RAG endpoint."""
    pass

@red_rag.post("/reddit_rag")
async def red_rag_bing(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    api_key: str = Depends(api_key_auth)
):
    """
    Handles Reddit-based RAG requests with a full pipeline:
    Search -> Scrape -> Chunk & Embed -> Retrieve -> Rerank -> Synthesize.
    """

    original_query = request.query.strip()

    validation_result = await validate_query_only(original_query)
    valid = validation_result.get("valid", 0)
    validation_tokens = validation_result.get("tokens_used", 0)

    async def generate_chat_res():
        """Generator function that streams the entire RAG process."""
        if valid == 0:
            error_message = "The search query is not related to financial markets, companies, or economics. Please ask a relevant question."
            yield error_message.encode("utf-8")
            return

        final_links = []

        try:
            # Step 1: Search for Reddit posts
            yield create_progress_bar_string(10, "Searching for relevant Reddit discussions...").encode("utf-8")
            brave_api_key = os.getenv('BRAVE_API_KEY')
            search_results = await fetch_search_red(original_query, brave_api_key)

            if not search_results:
                yield "\nCould not find any relevant Reddit discussions for your query.".encode("utf-8")
                return

            articles, df, links = await process_search_red(search_results)

            if not articles:
                yield "\nCould not find any relevant Reddit discussions for your query.".encode("utf-8")
                return

            top_articles = articles[:5] # Limit to top 5 articles for scraping
            final_links = [article['url'] for article in top_articles]

            yield create_progress_bar_string(25, f"Found {len(links)} discussions. Scraping top posts...").encode("utf-8")

            # Step 2: Scrape top Reddit posts
            scraper = RedditScraper()
            scraped_data = [await scraper.scrape_post(article) for article in top_articles]

            if not any(scraped_data):
                yield "\nFound Reddit discussions, but could not scrape their content.".encode("utf-8")
                return
            yield create_progress_bar_string(50, "Processing and embedding Reddit content...").encode("utf-8")

            # --- MODIFICATION START ---
            # Run the synchronous, CPU-bound function in a separate thread
            retriever = await asyncio.to_thread(
                create_reddit_vector_store_from_scraped_data,
                scraped_data
            )
            # --- MODIFICATION END ---

            if not retriever:
                yield "\nFailed to process Reddit content for analysis.".encode("utf-8")
                return
            yield create_progress_bar_string(75, "Finding the most relevant information...").encode("utf-8")

            # Step 4: Search Chunks for Relevance
            relevant_chunks: list[Document] = await retriever.ainvoke(original_query)
            if not relevant_chunks:
                yield "\nCould not find specific information related to your query in the Reddit posts.".encode("utf-8")
                return
            yield create_progress_bar_string(80, "Ranking and scoring relevant information...").encode("utf-8")

            # Step 5: Score and Rerank Chunks
            passages_to_score = [
                {"text": doc.page_content, "metadata": doc.metadata} for doc in relevant_chunks
            ]

            reranked_passages = await scoring_service.score_and_rerank_passages(original_query, passages_to_score)

            if not reranked_passages:
                yield "\nCould not determine the most relevant information from the Reddit posts.".encode("utf-8")
                return

            top_passages = reranked_passages[:7]
            final_context = scoring_service.create_enhanced_context(top_passages)
            yield create_progress_bar_string(90, "Synthesizing the final answer...").encode("utf-8")

        except Exception as e:
            print(f"ERROR: Reddit RAG pipeline failed: {e}")
            yield "\nAn error occurred while processing the Reddit discussions.".encode("utf-8")
            return

        # Step 6: Synthesize Answer
        res_prompt = """
        You are a financial information assistant specializing in Reddit discussions and community insights.
        Using the provided Reddit articles and chat history, respond to the user's inquiries with detailed analysis.

        Focus on:
        - Community sentiment and discussions
        - Popular opinions and debates
        - Emerging trends mentioned by users
        - Different perspectives from the Reddit community
        - Provide the source links with their citation numbers at the end of the response

        Use proper markdown formatting and cite relevant Reddit discussions.

        The user has asked the following question: {input}

        Context from Reddit:
        {context}
        """

        R_prompt = PromptTemplate(
            template=res_prompt,
            input_variables=["context", "input"]
        )
        ans_chain = R_prompt | llm_stream

        # Step 7: Stream the final response
        yield create_progress_bar_string(100, "Done!").encode("utf-8")
        yield "\n\n".encode("utf-8")
        yield "#Thinking ...\n".encode("utf-8")

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for chunk in ans_chain.astream({"context": final_context, "input": original_query}):
                    content = chunk.content
                    if content:
                        final_response += content
                        yield content.encode("utf-8")
                        await asyncio.sleep(0.01)

                total_tokens = validation_tokens + cb.total_tokens
                await insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool)

            links_data = {"links": final_links}
            await store_into_db(session_id, prompt_history_id, links_data, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(original_query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during final response streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(
        generate_chat_res(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "X-Accel-Buffering": "no"
        }
    )


async def validate_query_only(query: str) -> dict:
    """
    Performs a simple validation check on the user's query without reformulation.
    """
    prompt_template_str = """
    You are a financial query validator. Your task is to determine if the user's query is related to the Indian stock market, finance, economics, companies, or related news.

    User Query: "{query}"

    Return a single JSON object with one key, "valid", which should be 1 for a valid query and 0 for an invalid one.

    Example:
    User Query: "latest news on RBI monetary policy"
    {{
        "valid": 1
    }}

    User Query: "what is the best pizza topping"
    {{
        "valid": 0
    }}
    """
    input_data = {"query": query}

    prompt_template = PromptTemplate(template=prompt_template_str, input_variables=["query"])
    chain = prompt_template | GPT4o_mini | JsonOutputParser()

    try:
        with get_openai_callback() as cb:
            result = await chain.ainvoke(input_data)

        result["tokens_used"] = cb.total_tokens
        print(f"DEBUG: Validation-only check complete. Valid: {result.get('valid')}")
        return result

    except Exception as e:
        print(f"ERROR in validate_query_only: {e}")
        # Fallback to a safe default to allow the query to proceed
        return {
            "valid": 1,
            "tokens_used": 0
        }

async def condense_context_for_llm(query: str, passages: list[dict]) -> str:
    """
    Uses a non-streaming LLM to quickly extract key points from passages,
    creating a condensed context to reduce final streaming latency.
    """
    print("DEBUG: Condensing context for final answer synthesis...")
    context_text = "\n\n---\n\n".join(
        [f"Source: {p['metadata'].get('title', 'N/A')}\nContent: {p['text']}" for p in passages]
    )

    prompt = PromptTemplate.from_template(
        """
        You are a fact-extraction expert. Based on the user's query, extract the most critical facts, figures, and key points from the provided context into a concise bulleted list.

        User Query: {query}

        Context:
        {context}

        Key Points:
        """
    )

    # Use the non-streaming model for a fast, single-shot extraction
    chain = prompt | GPT4o_mini | StrOutputParser()

    try:
        condensed_context = await chain.ainvoke({"query": query, "context": context_text})
        print("DEBUG: Context condensed successfully.")
        return condensed_context
    except Exception as e:
        print(f"ERROR: Failed to condense context: {e}")
        # Fallback to the original, larger context if condensation fails
        return context_text

@red_rag.post("/reddit_rag")
async def red_rag_bing(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    api_key: str = Depends(api_key_auth)
):
    """
    Handles Reddit-based RAG requests with a full pipeline:
    Search -> Scrape -> Chunk & Embed -> Retrieve -> Rerank -> Synthesize.
    """

    original_query = request.query.strip()

    validation_result = await validate_query_only(original_query)
    valid = validation_result.get("valid", 0)
    validation_tokens = validation_result.get("tokens_used", 0)

    async def generate_chat_res():
        """Generator function that streams the entire RAG process."""
        if valid == 0:
            error_message = "The search query is not related to financial markets, companies, or economics. Please ask a relevant question."
            yield error_message.encode("utf-8")
            return

        final_links = []

        try:
            # Step 1: Search for Reddit posts
            brave_api_key = os.getenv('BRAVE_API_KEY')
            search_results = await fetch_search_red(original_query, brave_api_key)

            if not search_results:
                yield "\nCould not find any relevant Reddit discussions for your query.".encode("utf-8")
                return

            articles, df, links = await process_search_red(search_results)

            if not articles:
                yield "\nCould not find any relevant Reddit discussions for your query.".encode("utf-8")
                return
            
            yield f"# Reading sources | {len(articles)} articles\n".encode("utf-8")
            for article in articles:
                yield f"{json.dumps({'title': article.get('title'), 'url': article.get('url')})}\n".encode("utf-8")

            top_articles = articles[:5] # Limit to top 5 articles for scraping
            final_links = [article['url'] for article in top_articles]

            yield "# Filtering Content ...\n".encode("utf-8")

            # Step 2: Scrape top Reddit posts
            scraper = RedditScraper()
            scraped_data = [await scraper.scrape_post(article) for article in top_articles]

            if not any(scraped_data):
                yield "\nFound Reddit discussions, but could not scrape their content.".encode("utf-8")
                return

            # --- MODIFICATION START ---
            # Run the synchronous, CPU-bound function in a separate thread
            retriever = await asyncio.to_thread(
                create_reddit_vector_store_from_scraped_data,
                scraped_data
            )
            # --- MODIFICATION END ---

            if not retriever:
                yield "\nFailed to process Reddit content for analysis.".encode("utf-8")
                return
            
            yield "# Re-ranking context ...\n".encode("utf-8")

            # Step 4: Search Chunks for Relevance
            relevant_chunks: list[Document] = await retriever.ainvoke(original_query)
            if not relevant_chunks:
                yield "\nCould not find specific information related to your query in the Reddit posts.".encode("utf-8")
                return

            # Step 5: Score and Rerank Chunks
            passages_to_score = [
                {"text": doc.page_content, "metadata": doc.metadata} for doc in relevant_chunks
            ]

            reranked_passages = await scoring_service.score_and_rerank_passages(original_query, passages_to_score)

            if not reranked_passages:
                yield "\nCould not determine the most relevant information from the Reddit posts.".encode("utf-8")
                return

            top_passages = reranked_passages[:7]
            
            yield "# Creating enhanced context ...\n".encode("utf-8")
            
            final_context = scoring_service.create_enhanced_context(top_passages)

        except Exception as e:
            print(f"ERROR: Reddit RAG pipeline failed: {e}")
            yield "\nAn error occurred while processing the Reddit discussions.".encode("utf-8")
            return

        # Step 6: Synthesize Answer
        res_prompt = """
        You are a financial information assistant specializing in Reddit discussions and community insights.
        Using the provided Reddit articles and chat history, respond to the user's inquiries with detailed analysis.

        Focus on:
        - Community sentiment and discussions
        - Popular opinions and debates
        - Emerging trends mentioned by users
        - Different perspectives from the Reddit community
        - Provide the source links with their citation numbers at the end of the response

        Use proper markdown formatting and cite relevant Reddit discussions.

        The user has asked the following question: {input}

        Context from Reddit:
        {context}
        """

        R_prompt = PromptTemplate(
            template=res_prompt,
            input_variables=["context", "input"]
        )
        ans_chain = R_prompt | llm_stream

        # Step 7: Stream the final response
        yield "#Thinking ...\n".encode("utf-8")

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for chunk in ans_chain.astream({"context": final_context, "input": original_query}):
                    content = chunk.content
                    if content:
                        final_response += content
                        yield content.encode("utf-8")
                        await asyncio.sleep(0.01)

                total_tokens = validation_tokens + cb.total_tokens
                await insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool)

            links_data = {"links": final_links}
            await store_into_db(session_id, prompt_history_id, links_data, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(original_query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during final response streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(
        generate_chat_res(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "X-Accel-Buffering": "no"
        }
    )