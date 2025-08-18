# /aigptssh/streaming/streaming.py

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
from langchain_chroma import Chroma

# --- Functions imported from other modules ---
from streaming.reddit_stream import fetch_search_red, process_search_red
from streaming.yt_stream import get_data, get_yt_data_async
from api.news_rag.scoring_service import scoring_service
from api.youtube_rag.youtube_vector_store import create_yt_vector_store_from_transcripts


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
You are a financial assistant. Analyze this user query and provide a JSON response.

Context: This appears to be a follow-up question based on chat history.

User Query: "{query}"
Recent Chat History: {chat_history}
Today's Date: {today}

Tasks:
1. VALIDATE: Is this query related to Indian stock market, finance, economics, elections, or companies? 
2. REFORMULATE: Create a standalone question that incorporates relevant context from chat history
3. CLEAN: Ensure the reformulated query is complete and grammatically correct

Return JSON format:
{{
    "valid": 1 or 0,
    "reformulated_query": "standalone question with context",
    "is_followup": true,
    "needs_memory": true
}}

Guidelines:
- If user mentions "it", "this", "that", "the company", "the stock" - identify from context
- Include specific company/stock names from history if referenced
- Make the reformulated query understandable without chat history
- For validation: 1=valid financial query, 0=not related to finance/markets
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
1. VALIDATE: Is this query related to Indian stock market, finance, economics, elections, or companies?
2. PROCESS: Clean the query and ensure it's grammatically correct

Return JSON format:
{{
    "valid": 1 or 0,
    "reformulated_query": "{query}",
    "is_followup": false,
    "needs_memory": false
}}

Guidelines:
- If asking for "latest news" about any company, consider valid
- If asking about "current news" or "trending news", consider valid  
- For validation: 1=valid financial query, 0=not related to finance/markets
- Keep reformulated_query same as original for standalone questions
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
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    REPLACED: Final version with corrected session management.
    Implements the full, multi-stage "search and re-rank" strategy.
    """
    query = request.query.strip()
    today = datetime.now().strftime("%Y-%m-%d")
    chat_history = await get_chat_history_optimized(str(session_id), db_pool, limit=3)

    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        raise HTTPException(status_code=500, detail="Brave API key not configured.")
    
    brave_searcher = BraveNews(brave_api_key)

    # /aigptssh/streaming/streaming.py

    async def tiered_stream_generator():
        # --- Caching Logic Start ---
        cached_passages = []
        if use_caching:
            yield create_progress_bar_string(5, "Checking cache for context...").encode("utf-8")
            
            # Check if it's a follow-up question that could benefit from cache
            if is_followup_question(query, chat_history):
                # Query the session-specific cache
                cached_docs_with_scores = query_session_cache(str(session_id), query)
                
                # Assess sufficiency of cached documents
                sufficiency_score = scoring_service.assess_context_sufficiency(query, cached_docs_with_scores)
                
                # A threshold of 0.4 is a good starting point
                if sufficiency_score > 0.4:
                    print("DEBUG: Sufficient context found in cache. Bypassing web scrape.")
                    # Convert results back to the passage dictionary format for processing
                    for doc, score in cached_docs_with_scores:
                        cached_passages.append({
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "final_combined_score": score # Use similarity score as the ranking metric
                        })
        
        if cached_passages:
            # If cache was sufficient, use the cached passages directly
            final_passages = cached_passages
            yield create_progress_bar_string(95, "Generating answer from cache...").encode("utf-8")
        else:
            # --- Fallback to Web Scraping Logic (if cache is not used, empty, or insufficient) ---
            async with aiohttp.ClientSession(**brave_searcher.session_config) as session:
                yield create_progress_bar_string(10, "Searching for sources...").encode("utf-8")
                
                initial_sources = await brave_searcher.search_and_scrape(session, query, max_sources=30)
                if not initial_sources:
                    yield "\nCould not find any initial sources.".encode("utf-8")
                    return
                
                yield create_progress_bar_string(20, f"Found {len(initial_sources)} sources...").encode("utf-8")
                sources_to_scrape = initial_sources[:10]
                
                scraped_sources = []
                total_to_scrape = len(sources_to_scrape)
                start_progress, end_progress = 20, 70
                
                for i, source in enumerate(sources_to_scrape):
                    progress = start_progress + int(((i + 1) / total_to_scrape) * (end_progress - start_progress))
                    title = source.get('title', 'Untitled Source')[:45]
                    yield create_progress_bar_string(progress, f"Analyzing: {title}...").encode("utf-8")
                    
                    scraped = await brave_searcher.scrape_top_urls(session, [source])
                    if scraped:
                        scraped_sources.extend(scraped)
                    await asyncio.sleep(0.1)

                yield create_progress_bar_string(80, "Ranking context...").encode("utf-8")
                
                final_passages = await scoring_service.rerank_content_chunks(query, scraped_sources, top_n=7)
                if not final_passages:
                    yield "\nCould not extract sufficient detailed information.".encode("utf-8")
                    return

                # Add newly scraped passages to the cache if caching is enabled
                if use_caching:
                    add_passages_to_cache(str(session_id), final_passages)

                yield create_progress_bar_string(95, "Generating final answer...").encode("utf-8")

        # --- Final Answer Generation (common for both cached and scraped paths) ---
        final_context = scoring_service.create_enhanced_context(final_passages)
        final_links = list(set([p["metadata"].get("link") for p in final_passages if p.get("metadata", {}).get("link")]))

        final_prompt = PromptTemplate.from_template(
            """
            You are a financial markets expert. Provide a detailed, well-structured final answer using the comprehensive context provided.
            Use markdown for readability and cite the source links where appropriate. Provide the source links with their citation numbers at the end of the response.
            
            Comprehensive Context:
            {context}
            
            Chat History: 
            {history}
            
            User Question: {input}
            
            Final Detailed Answer:
            """
        )
        final_chain = final_prompt | llm_stream
        
        yield create_progress_bar_string(100, "Done!").encode("utf-8")
        yield "\n\n".encode("utf-8") 
        yield "#Thinking ...\n".encode("utf-8")
        
        final_response_text = ""
        with get_openai_callback() as cb:
            async for chunk in final_chain.astream({"context": final_context, "history": chat_history, "input": query}):
                if chunk.content:
                    final_response_text += chunk.content
                    yield chunk.content.encode("utf-8")
            
            # --- Final Database Operations ---
            if not cached_passages: # Only process for dataframe if it was a fresh scrape
                df_to_insert = brave_searcher._process_for_dataframe(scraped_sources)
                if not df_to_insert.empty:
                    asyncio.create_task(insert_post1(df_to_insert, db_pool))

            total_tokens = cb.total_tokens
            asyncio.create_task(insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool))
            asyncio.create_task(store_into_db(session_id, prompt_history_id, {"links": final_links}, db_pool))
            
            if final_response_text:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(query)
                history_db.add_ai_message(final_response_text)

    return StreamingResponse(tiered_stream_generator(), media_type="text/event-stream")

@cmots_rag.post("/cmots_rag")
async def cmots_only(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Optimized CMOTS RAG endpoint."""
    
    query = request.query.strip()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get chat history and do combined preprocessing
    chat_history = await get_chat_history_optimized(str(session_id), db_pool, limit=3)
    preprocessing_result = await combined_preprocessing(query, chat_history, today)
    
    # Extract results
    valid = preprocessing_result.get("valid", 0)
    reformulated_query = preprocessing_result.get("reformulated_query", query)
    extracted_date = preprocessing_result.get("extracted_date", "None")
    is_followup = preprocessing_result.get("is_followup", False)
    preprocessing_tokens = preprocessing_result.get("tokens_used", 0)
    
    docs = ""
    
    if valid != 0:
        try:
            # Apply date filter if extracted
            filter_dict = None
            if extracted_date != 'None':
                try:
                    date_int = int(extracted_date)
                    filter_dict = {"date": {"$gte": date_int}}
                    print(f"DEBUG: Applying date filter: {filter_dict}")
                except ValueError:
                    print(f"DEBUG: Invalid date format: {extracted_date}")
            
            # Search vector store with reformulated query
            results = vs.similarity_search_with_score(reformulated_query, k=10, filter=filter_dict)
            docs = "\n\n".join([doc[0].page_content for doc in results])
            print(f"DEBUG: Retrieved {len(results)} documents from vector store")
            
        except Exception as e:
            docs = ""
            print(f"ERROR: Vector search failed: {e}")

    res_prompt = """
    CMOTS news articles: {cmots} 
    Chat history: {history}
    Today date: {date}
    
    You are a stock news and stock market information bot. 
    Using only the provided News Articles and chat history, respond to the user's inquiries in detail without omitting any context. 
    Provide accurate, factual information based solely on the given articles.
    Use proper markdown formatting for better readability.
    
    The user has asked the following question: {input}
    """
    
    question_prompt = PromptTemplate(
        input_variables=["history", "cmots", "date", "input"], 
        template=res_prompt
    )
    ans_chain = question_prompt | llm_stream

    async def generate_chat_res(docs, history, input_query, date):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets, companies, economics, or finance. Please ask questions about stocks, companies, market trends, financial news, or economic developments."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in ans_chain.astream_events(
                    {"cmots": docs, "history": history, "input": input_query, "date": date}, 
                    version="v1"
                ):
                    if event["event"] == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)
                
                # Calculate total tokens
                total_tokens = preprocessing_tokens + cb.total_tokens
                await insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool)

            await store_into_db(session_id, prompt_history_id, {"links": []}, db_pool)

            # Store conversation in history
            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(reformulated_query if is_followup else query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(
        generate_chat_res(docs, chat_history, reformulated_query, today), 
        media_type="text/event-stream"
    )


@red_rag.post("/reddit_rag")
async def red_rag_bing(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Optimized Reddit RAG endpoint."""
    
    query = request.query.strip()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Get chat history and do combined preprocessing
    chat_history = await get_chat_history_optimized(str(session_id), db_pool, limit=3)
    preprocessing_result = await combined_preprocessing(query, chat_history, today)
    
    # Extract results
    valid = preprocessing_result.get("valid", 0)
    reformulated_query = preprocessing_result.get("reformulated_query", query)
    is_followup = preprocessing_result.get("is_followup", False)
    preprocessing_tokens = preprocessing_result.get("tokens_used", 0)
    
    docs = ""
    links = []
    
    if valid != 0:
        try:
            # Fetch and process Reddit data
            sr = await fetch_search_red(reformulated_query)
            docs, df, links = await process_search_red(sr)
            
            if df is not None and not df.empty:
                await insert_red(df, db_pool)
                print(f"DEBUG: Processed {len(df)} Reddit posts")
                
        except Exception as e:
            print(f"ERROR: Reddit processing failed: {e}")
            docs, links = "", []

    res_prompt = """
    Reddit articles: {context}
    Chat history: {history}
    Today date: {date}
    
    You are a financial information assistant specializing in Reddit discussions and community insights.
    Using the provided Reddit articles and chat history, respond to the user's inquiries with detailed analysis.
    
    Focus on:
    - Community sentiment and discussions
    - Popular opinions and debates
    - Emerging trends mentioned by users
    - Different perspectives from the Reddit community
    
    Use proper markdown formatting and cite relevant Reddit discussions.
    
    The user has asked the following question: {input}        
    """
    
    R_prompt = PromptTemplate(
        template=res_prompt, 
        input_variables=["history", "context", "input", "date"]
    )
    ans_chain = R_prompt | llm_stream

    async def generate_chat_res(history, docs, query, date):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets, companies, economics, or finance. Please ask questions about stocks, companies, market trends, financial news, or economic developments."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in ans_chain.astream_events(
                    {"history": history, "context": docs, "input": query, "date": date}, 
                    version="v1"
                ):
                    if event["event"] == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)

                # Calculate total tokens
                total_tokens = preprocessing_tokens + cb.total_tokens
                await insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool)

            # Store links and conversation
            links_data = {"links": links}
            await store_into_db(session_id, prompt_history_id, links_data, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(reformulated_query if is_followup else query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(
        generate_chat_res(chat_history, docs, reformulated_query, today), 
        media_type="text/event-stream"
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

@yt_rag.post("/yt_rag")
async def yt_rag_brave(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Handles YouTube-based RAG requests using an embedding-based approach.
    1. Searches for relevant videos.
    2. Fetches transcripts.
    3. Chunks and embeds transcripts into a vector store.
    4. Retrieves relevant chunks.
    5. Synthesizes an answer from the chunks.
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
            # Step 1: Search & Filter Videos
            yield create_progress_bar_string(10, "Searching for relevant videos...").encode("utf-8")
            video_urls = await get_yt_data_async(original_query)
            
            if not video_urls:
                yield "\nCould not find any relevant YouTube videos for your query.".encode("utf-8")
                return
            final_links = video_urls
            yield create_progress_bar_string(25, f"Found {len(video_urls)} videos. Fetching transcripts...").encode("utf-8")

            # Step 2: Fetch Transcripts asynchronously
            video_transcripts = []
            async for transcript_data in get_data(video_urls, db_pool):
                video_transcripts.append(transcript_data)
            if not video_transcripts:
                yield "\nFound videos, but could not retrieve their transcripts.".encode("utf-8")
                return
            yield create_progress_bar_string(50, "Processing and embedding video content...").encode("utf-8")

            # Step 3: Chunk, Embed & Store in Vector DB
            retriever = create_yt_vector_store_from_transcripts(video_transcripts)
            if not retriever:
                yield "\nFailed to process video content for analysis.".encode("utf-8")
                return
            yield create_progress_bar_string(75, "Finding the most relevant information...").encode("utf-8")
            
            # Step 4: Search Chunks for Relevance
            relevant_chunks: list[Document] = await retriever.aget_relevant_documents(original_query)
            if not relevant_chunks:
                yield "\nCould not find specific information related to your query in the videos.".encode("utf-8")
                return
            yield create_progress_bar_string(80, "Ranking and scoring relevant information...").encode("utf-8")

            # Step 5: Score and Rerank Chunks
            passages_to_score = [
                {"text": doc.page_content, "metadata": doc.metadata} for doc in relevant_chunks
            ]
            
            reranked_passages = await scoring_service.score_and_rerank_passages(original_query, passages_to_score)
            
            if not reranked_passages:
                yield "\nCould not determine the most relevant information from the videos.".encode("utf-8")
                return

            top_passages = reranked_passages[:7]
            yield create_progress_bar_string(85, "Creating final context...").encode("utf-8")

            # Create the full context directly without condensation
            final_context = scoring_service.create_enhanced_context(top_passages)
            
            # Create a mapping of source titles to URLs for the final prompt
            source_map = {p['metadata'].get('title'): p['metadata'].get('url') for p in top_passages if p.get('metadata')}
            
            yield create_progress_bar_string(90, "Synthesizing the final answer...").encode("utf-8")

        except Exception as e:
            print(f"ERROR: YouTube RAG pipeline failed: {e}")
            yield "\nAn error occurred while processing the videos.".encode("utf-8")
            return

        # Step 6: Synthesize Answer using the full context
        prompt = """
        You are a financial information assistant. Your task is to answer the user's question using the provided context from YouTube video transcripts.

        Guidelines:
        - Base your answer on the information within the transcripts. Do not invent or use outside knowledge.
        - Provide a comprehensive and detailed response covering all relevant aspects from the transcripts.
        - Cite your sources by adding a number like [1], [2], etc., for each video used in your answer.
        - At the end of your response, create a "Sources" section and list all the YouTube links with their corresponding citation numbers and video title. Do not list the same source multiple times.

        **User's question:** {query}

        **Context from Videos:**
        {context}
        
        **Source Map (Titles and URLs):**
        {source_map}
        """
        
        yt_prompt = PromptTemplate(template=prompt, input_variables=["query", "context", "source_map"])
        chain = yt_prompt | llm_stream

        # Step 6: Stream the final response
        yield create_progress_bar_string(100, "Done!").encode("utf-8")
        yield "\n\n".encode("utf-8") 
        yield "#Thinking ...\n".encode("utf-8")
        
        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for chunk in chain.astream({
                    "context": final_context, 
                    "query": original_query,
                    "source_map": source_map
                }):
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
            
    return StreamingResponse(generate_chat_res(), media_type="text/event-stream")