# /aigptcur/app_service/streaming/streaming.py

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
from pinecone import PodSpec, Pinecone as PineconeClient
import requests
from pydantic import BaseModel
from typing import Any
import re
import asyncpg
from langchain.retrievers import MergerRetriever
from langchain.docstore.document import Document
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
from contextlib import contextmanager
import time
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse
import pandas as pd

# --- Local Project Imports ---
from config import (
    chroma_server_client, llm_date, llm_stream, vs, GPT4o_mini,
    PINECONE_INDEX_NAME, CONTEXT_SUFFICIENCY_THRESHOLD
)
from langchain_chroma import Chroma

# --- Functions imported from other modules ---
from api.news_rag.brave_news import get_brave_results
from streaming.reddit_stream import fetch_search_red, process_search_red
from streaming.yt_stream import get_data, get_yt_data_async
from api.news_rag.scoring_service import scoring_service

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

def diversify_results(passages: list[dict], max_per_source: int = 3) -> list[dict]:
    """
    Post-processes a list of passages to ensure source diversity.
    It prioritizes keeping the highest-scoring passage from each source.
    """
    source_counts = {}
    diversified_list = []
    
    # Sort passages by score to process the best ones first
    passages.sort(key=lambda x: x.get('final_combined_score', 0), reverse=True)
    
    for passage in passages:
        source_link = passage.get("metadata", {}).get("link", "unknown")
        
        try:
            # Normalize the domain to treat www. and non-www. as the same
            domain = urlparse(source_link).netloc.replace('www.', '') if source_link != "unknown" else "unknown"
        except:
            domain = "unknown"

        # Add the passage if we haven't hit the limit for this source
        if source_counts.get(domain, 0) < max_per_source:
            diversified_list.append(passage)
            source_counts[domain] = source_counts.get(domain, 0) + 1
            
    print(f"DEBUG: Diversified passages from {len(passages)} to {len(diversified_list)}")
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

# --- API Endpoints ---

class InRequest(BaseModel):
    query: str

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
    A robust, tiered response system that performs a single web search to avoid rate limiting and errors.
    """
    query = request.query.strip()
    today = datetime.now().strftime("%Y-%m-%d")

    chat_history = await get_chat_history_optimized(str(session_id), db_pool, limit=3)
    preprocessing_result = await combined_preprocessing(query, chat_history, today)

    if preprocessing_result.get("valid", 0) == 0:
        async def invalid_query_stream():
            yield "The search query is not related to Indian financial markets...".encode("utf-8")
        return StreamingResponse(invalid_query_stream(), media_type="text/event-stream")

    reformulated_query = preprocessing_result.get("reformulated_query", query)

    async def tiered_stream_generator():
        preliminary_prompt = PromptTemplate.from_template(
            "You are a financial news assistant. Based on these initial snippets, provide a brief, 2-3 bullet point summary for the user's question. Mention that a more detailed analysis is being prepared.\n\nSnippets:\n{context}\n\nUser Question: {input}\n\nPreliminary Summary:"
        )
        preliminary_chain = preliminary_prompt | llm_stream

        final_prompt = PromptTemplate.from_template(
            "You are a financial markets super-assistant. Provide a detailed, well-structured final answer to the user's question using all available context. Use markdown, cite sources with links, and be thorough.\n\nComprehensive Context:\n{context}\n\nChat History: {history}\n\nUser Question: {input}\n\nFinal Detailed Answer:"
        )
        final_chain = final_prompt | llm_stream
        
        # --- Tier 1: Perform a Single Web Search and Start Vector Search ---
        web_search_task = asyncio.create_task(get_brave_results(reformulated_query, max_pages=1, max_sources=10))
        vector_search_task = asyncio.create_task(vs.asimilarity_search_with_score(reformulated_query, k=5))

        # Await ONLY the web search first, as it's needed for the preliminary summary
        web_articles, df = await web_search_task
        
        # --- Tier 2: Generate Preliminary Summary Immediately ---
        if web_articles:
            yield "### Preliminary Summary\n".encode("utf-8")
            # Generate snippets from the results of our single search
            initial_snippets = await quick_brave_search_for_snippets(web_articles)
            preliminary_context = "\n\n".join(initial_snippets)
            
            async for chunk in preliminary_chain.astream({"context": preliminary_context, "input": reformulated_query}):
                if chunk.content:
                    yield chunk.content.encode("utf-8")
        
        # --- Tier 3: Assemble and Stream Detailed Analysis ---
        yield "\n\n---\n### Detailed Analysis\n".encode("utf-8")
        
        # Now, await the vector search results
        initial_vector_results = await vector_search_task

        final_passages = []
        if initial_vector_results:
            final_passages.extend([{"text": doc.page_content, "metadata": {**doc.metadata, "link": doc.metadata.get("source_url") or doc.metadata.get("url")}} for doc, score in initial_vector_results])
        
        # Add the web articles we already fetched
        if web_articles:
            existing_links = {p['metadata'].get('link') for p in final_passages}
            new_passages = [
                {"text": f"Title: {a.get('title', '')}\nDescription: {a.get('description', '')}",
                 "metadata": {"title": a.get('title'), "link": a.get('source_url'), "publication_date": a.get('source_date'), "snippet": a.get('description')}}
                for a in web_articles if a.get('source_url') not in existing_links
            ]
            final_passages.extend(new_passages)

        if not final_passages:
            yield "Could not find sufficient information to provide a detailed analysis.".encode("utf-8")
            return

        reranked_passages = await scoring_service.score_and_rerank_passages(question=reformulated_query, passages=final_passages)
        diversified_passages = diversify_results(reranked_passages, max_per_source=3)
        final_context = scoring_service.create_enhanced_context(diversified_passages)
        final_links = [p["metadata"].get("link") for p in diversified_passages[:10] if p.get("metadata", {}).get("link")]

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for chunk in final_chain.astream({"context": final_context, "history": chat_history, "input": reformulated_query}):
                    if chunk.content:
                        content = chunk.content
                        final_response += content
                        yield content.encode("utf-8")

                # Final database updates
                if df is not None and not df.empty:
                    await insert_post1(df, db_pool)
                total_tokens = cb.total_tokens
                await insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool)
                await store_into_db(session_id, prompt_history_id, {"links": final_links}, db_pool)
                if final_response:
                    history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                    history_db.add_user_message(reformulated_query)
                    history_db.add_ai_message(final_response)
        except Exception as e:
            print(f"ERROR: An error occurred during final streaming: {e}")
            yield b"An error occurred while generating the final response."

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


@yt_rag.post("/yt_rag")
async def yt_rag_bing(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Optimized YouTube RAG endpoint."""
    
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
    
    data = ""
    links = []
    
    if valid != 0:
        try:
            # Get YouTube data with reformulated query
            links = await get_yt_data_async(reformulated_query)
            data = await get_data(links, db_pool)
            print(f"DEBUG: Retrieved {len(links)} YouTube videos")
            
        except Exception as e:
            print(f"ERROR: YouTube processing failed: {e}")
            data, links = "", []

    prompt = """
    Given YouTube transcripts: {summaries}
    Chat history: {history}
    Today date: {date}
    
    You are a financial information assistant specializing in video content analysis.
    Using the provided YouTube transcripts and chat history, respond to the user's inquiries with insights from video discussions.
    
    Focus on:
    - Key insights from financial experts and analysts
    - Market predictions and analysis from videos
    - Educational content and explanations
    - Different expert perspectives on financial topics
    
    Always mention the source videos when referencing specific information.
    Use proper markdown formatting for better readability.
    
    The user has asked the following question: {query}
    """
    
    yt_prompt = PromptTemplate(
        template=prompt, 
        input_variables=["history", "query", "summaries", "date"]
    )
    chain = yt_prompt | llm_stream

    async def generate_chat_res(history, data, query, date):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets, companies, economics, or finance. Please ask questions about stocks, companies, market trends, financial news, or economic developments."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in chain.astream_events(
                    {"history": history, "summaries": data, "query": query, "date": date}, 
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
        generate_chat_res(chat_history, data, reformulated_query, today), 
        media_type="text/event-stream"
    )