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
from datetime import datetime
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
# Import necessary components from other parts of your application
from config import (
    chroma_server_client, llm_date, llm_stream, vs, GPT4o_mini,
    PINECONE_INDEX_NAME, CONTEXT_SUFFICIENCY_THRESHOLD
)
from langchain_chroma import Chroma

# --- Functions imported from other modules that need refactoring for DB access ---
# Note: Ideally, these functions would be refactored in their original files.
# They are included here to provide a complete, working example.
from api.news_rag.brave_news import get_brave_results
from streaming.reddit_stream import fetch_search_red, process_search_red
from streaming.yt_stream import get_data, get_yt_data_async

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
    """
    FastAPI dependency to get the database pool from the application state.
    This ensures that the pool is initialized before being used by any route.
    """
    # Check if the pool exists on the app state
    if not hasattr(request.app.state, 'db_pool') or not request.app.state.db_pool:
        print("CRITICAL ERROR: Database pool is not available on app state.")
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    return request.app.state.db_pool

# --- Refactored Database Functions ---
# These functions now accept a `db_pool` argument instead of relying on a global variable.

async def insert_post1(df: pd.DataFrame, db_pool: asyncpg.Pool):
    """
    Asynchronously inserts a DataFrame into the source_data table using a connection pool.
    This is a non-blocking version that accepts an injected pool.
    """
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            for index, row in df.iterrows():
                try:
                    exists = await conn.fetchval(
                        "SELECT 1 FROM source_data WHERE source_url = $1",
                        row['source_url']
                    )
                    if not exists:
                        await conn.execute("""
                            INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        row['source_url'], row.get('image_url'), row['heading'],
                        row['title'], row['description'], row['source_date'])
                except Exception as e:
                    print(f"Error inserting row for URL {row['source_url']}: {e}")
    print(f"DEBUG: Asynchronous insert for {len(df)} rows completed.")


async def insert_red(df: pd.DataFrame, db_pool: asyncpg.Pool):
    """Asynchronously inserts Reddit data from a DataFrame into the source_data table."""
    async with db_pool.acquire() as conn:
        for index, row in df.iterrows():
            exists = await conn.fetchval("SELECT 1 FROM source_data WHERE source_url = $1", row['source_url'])
            if not exists:
                await conn.execute("""
                    INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, row['source_url'], None, None, row['title'], row['description'], row['source_date'])


async def insert_credit_usage(user_id: int, plan_id: int, credit_used: float, db_pool: asyncpg.Pool):
    """Asynchronously inserts credit usage data using the connection pool."""
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO credit_usage (user_id, plan_id, credit_used, "createdAt", "updatedAt")
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, plan_id, credit_used, current_time, current_time)
    print(f"SUCCESS: Token usage captured: {credit_used * 1000}")


async def store_into_db(pid: int, ph_id: int, result_json: dict, db_pool: asyncpg.Pool):
    """Asynchronously stores data into the streamingData table using the connection pool."""
    result_json_str = json.dumps(result_json)
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO "streamingData" (prompt_id, prompt_history_id, source_data)
            VALUES ($1, $2, $3)
        """, pid, ph_id, result_json_str)


async def query_validate(query: str, session_id: str, db_pool: asyncpg.Pool):
    """Validates a query against chat history using the provided DB pool."""
    res_prompt = """
    You are a highly skilled indian stock market investor and financial advisor. Your task is to validate whether a given question is related to the stock market or finance or elections or economics or general private listed companies. Additionally, if the new question is a follow-up then only use chat history to determine its validity.
    If question is asking about latest news about any company or current news or just company or trending news of any company consider it as valid question.
    Given question : {q}
    chat history : {list_qs}
    Output the result in JSON format:
    "valid": Return 1 if the question is valid, otherwise return 0.
    """
    R_prompt = PromptTemplate(template=res_prompt, input_variables=["list_qs", "q"])
    chain = R_prompt | GPT4o_mini | JsonOutputParser()

    messages = []
    async with db_pool.acquire() as conn:
        s_id = str(session_id)
        rows = await conn.fetch("SELECT message FROM message_store WHERE session_id = $1", s_id)
        # The 'message' column contains JSON strings, so we extract them.
        messages = [row['message'] for row in rows]

    # FIX: Parse the JSON string into a dictionary before accessing keys.
    chat = [json.loads(row)['data']['content'] for row in messages[-2:]]
    m_chat = [json.loads(row)['data']['content'] for row in messages[-4:]]
    h_chat = [json.loads(row)['data']['content'] for row in messages[-6:]]


    with get_openai_callback() as cb:
        input_data = {"list_qs": chat, "q": query}
        res = await chain.ainvoke(input_data)

    return res['valid'], cb.total_tokens, m_chat, h_chat

# --- Helper Functions (No DB interaction) ---

async def llm_get_date(user_query):
    today = datetime.now().strftime("%Y-%m-%d")
    date_prompt = """
        Today's date is {today}.
        Here's the user query: {user_query}
        Using the above given date for context, figure out which date the user wants data for.
        If the user query mentions "today" ,then use the above given today's date. Output the date in the format YYYYMMDD.
        If the user query mentions "yesterday"or "trending",output 1 day back date from todays date in YYYYMMDD.
        If the user is asking about recently/latest news in user query .output 7 days back date from todays date in YYYYMMDD.
        If the user is aksing about specifc time period in user query from past. output the start date the user mentioned in YYYYMMDD format.
        If the user doesnot mention any date in user query or asking about upcoming date outpute date as  "None" and If the user mention anything about quater and year output date as "None".

        Also, remove time-related references from the user query, ensuring it remains grammatically correct.

        Format your output as:
        YYYYMMDD,modified_user_query
        """
    D_prompt = PromptTemplate(template=date_prompt, input_variables=["today","user_query"])
    llm_chain_date= LLMChain(prompt=D_prompt, llm=llm_date)
    response=await llm_chain_date.arun(today=today,user_query=user_query)
    response = response.strip()
    date, general_user_query = split_input(response)
    return date,general_user_query,today


def split_input(input_string):
    parts = input_string.split(',', 1)
    date = parts[0].strip()
    general_user_query = parts[1].strip() if len(parts) > 1 else ""
    return date, general_user_query


def count_tokens(text, model_name="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


async def memory_chain(query,m_chat):
    contextualize_q_system_prompt = """Given a chat history and the user question \
    which might reference context in the chat history, formulate a standalone question if needed include time/date part also based on user previous question.\
    which can be understood without the chat history. Do NOT answer the question,
    If user question contains only stock name or stock ticker reformulate question as recent news of that stock.
    If user question contains current news / recent trends reformulate question as todays market news or trends.
    just reformulate it if needed and if the user question doesnt have any relevancy to chat history return as it is. 
    chat history:{chat_his}
    user question:{query}
    """
    c_q_prompt=PromptTemplate(template=contextualize_q_system_prompt,input_variables=['chat_his','query'])
    memory_chain=LLMChain(prompt=c_q_prompt,llm=llm_date)
    res=await memory_chain.arun(query=query,chat_his=m_chat)
    return res

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
    db_pool: asyncpg.Pool = Depends(get_db_pool)  # Dependency Injection
):
    query = request.query
    valid, v_tokens, m_chat, h_chat = await query_validate(query, str(session_id), db_pool)

    matched_docs, memory_query, t_day, his, links = '', '', '', '', []
    if valid != 0:
        original_query = query
        memory_query = await memory_chain(query, m_chat)

        print(f"DEBUG: Original User Query: '{original_query}'")
        print(f"DEBUG: Reformulated Memory Query: '{memory_query}'")

        date, user_q, t_day = await llm_get_date(memory_query)
        pinecone_results_with_scores = vs.similarity_search_with_score(original_query, k=15)

        from api.news_rag.scoring_service import scoring_service
        sufficiency_score = scoring_service.assess_context_sufficiency(original_query, pinecone_results_with_scores)

        web_passages = []
        if sufficiency_score < CONTEXT_SUFFICIENCY_THRESHOLD:
            print(f"DEBUG: Sufficiency score {sufficiency_score:.2f} is below threshold. Triggering Brave search.")
            articles, df = await get_brave_results(memory_query)
            if articles and df is not None and not df.empty:
                await insert_post1(df, db_pool) # Pass the pool
                # This section should be updated to properly handle document creation and upserting
                # For now, it's assumed to populate `web_passages` correctly.
        else:
            print(f"DEBUG: Sufficiency score {sufficiency_score:.2f} is sufficient. Skipping Brave search.")

        pinecone_passages = [{"text": doc.page_content, "metadata": {**doc.metadata, "score": score}} for doc, score in pinecone_results_with_scores]
        all_passages_map = {}
        for p in web_passages + pinecone_passages:
            link = p["metadata"].get("link") or f"nolink_{hash(p['text'])}"
            if link not in all_passages_map: all_passages_map[link] = p
        all_passages = list(all_passages_map.values())
        if all_passages:
            reranked_passages = await scoring_service.score_and_rerank_passages(question=memory_query, passages=all_passages)
            matched_docs = scoring_service.create_enhanced_context(reranked_passages)
            links = [p["metadata"]["link"] for p in reranked_passages if "link" in p.get("metadata", {})]
        else:
            matched_docs, links = "", []
        his = h_chat

    res_prompt = """
     News Articles : {bing}
    chat history : {history}
    Today date:{date}
    
    use the date provided in the metadata to answer the user query if the user is asking in specific time periods.
    If the same question {input} present in chat_history ignore that answer present in chat history dont consider that answer while generating final answer.
    give priority to latest date provided in metadata while answering user query.
    
    I am a financial markets super-assistant trained to function like Perplexity.ai — with enhanced domain intelligence and deep search comprehension.
    I am connected to a real-time web search + scraping engine that extracts live content from verified financial websites, regulatory portals, media publishers, and government sources.
    I serve as an intelligent financial answer engine, capable of understanding and resolving even the most **complex multi-part queries**, returning **accurate, structured, and sourced answers**.
    \n---\n
    PRIMARY MISSION:\n
    Deliver **bang-on**, complete, real-time financial answers about:\n
    - Companies (ownership, results, ratios, filings, news, insiders)\n
    - Stocks (live prices, historicals, volumes, charts, trends)\n
    - People (CEOs, founders, investors, economists, politicians)\n
    - Mutual Funds & ETFs (returns, risk, AUM, portfolio, comparisons)\n
    - Regulators & Agencies (SEBI, RBI, IRDAI, MCA, MoF, CBIC, etc.)\n
    - Government (policies, circulars, appointments, reforms, speeches)\n
    - Macro Indicators (GDP, repo rate, inflation, tax policy, liquidity)\n
    - Sectoral Data (FMCG, BFSI, Infra, IT, Auto, Pharma, Realty, etc.)\n
    - Financial Concepts (with real-world context and current examples)\n
    \n---\n
    COMPLEX QUERY UNDERSTANDING:\n
    You are optimized to handle **simple to deeply complex queries**.\n
    \n---\n
    INTELLIGENT BEHAVIOR GUIDELINES:\n
    1. **Bang-On Precision**: Always provide factual, up-to-date data from verified sources. Never hallucinate.\n
    2. **Break Down Complex Queries**: Decompose long or layered queries. Use intelligent reasoning to structure the answer.\n
    3. **Research Assistant Tone**: Neutral, professional, data-first. No assumptions, no opinions. Cite all key facts.\n
    4. **Source-Based**: Every metric or statement must include a credible source: (Source: [Link Title or Description](URL)).\n
    5. **Fresh + Archived Data**: Always prioritize today's/latest info. For long-term trends or legacy data, explicitly state the timeframe.\n
    6. **Answer Structuring**: Start with a concise summary. Use bullet points, tables, and subheadings.\n
    \n---\n
    STRICT LIMITATIONS:\n
    - Never make up data.\n
    - No financial advice, tips, or trading guidance.\n
    - No generic phrases like "As an AI, I…".\n
    - No filler or irrelevant content — answer only the query's intent.\n
    **DONT PROVIDE ANY EXTRA INFORMATION APART FROM USER QUESTION AND ANSWER SHOULD BE IN PROPER MARKDOWN FORMATTING**
    
    The user has asked the following question: {input}
    """
    R_prompt = PromptTemplate(template=res_prompt, input_variables=["bing","input","date","history"])
    ans_chain = R_prompt | llm_stream

    async def generate_chat_res(matched_docs, query, t_day, history):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets..."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            prompt_text = f"{matched_docs}\n{history}\n{query}\n{t_day}"
            prompt_tokens = count_tokens(prompt_text)

            async for event in ans_chain.astream_events({"bing": matched_docs, "history": history, "input": query, "date": t_day}, version="v1"):
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        final_response += content
                        yield content.encode("utf-8")
                        await asyncio.sleep(0.01)

            completion_tokens = count_tokens(final_response)
            total_tokens = prompt_tokens + completion_tokens
            await insert_credit_usage(user_id, plan_id, total_tokens / 1000, db_pool)
            await store_into_db(session_id, prompt_history_id, {"links": links}, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(memory_query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(matched_docs, memory_query, t_day, his), media_type="text/event-stream")


@cmots_rag.post("/cmots_rag")
async def cmots_only(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool) # Dependency Injection
):
    query = request.query
    valid, v_tokens, m_chat, h_chat = await query_validate(query, str(session_id), db_pool)

    docs, his, memory_query, date = '', '', '', ''
    if valid != 0:
        memory_query = await memory_chain(query, m_chat)
        date, user_q, t_day = await llm_get_date(memory_query)
        his = h_chat
        try:
            filter_dict = {"date": {"$gte": int(date)}} if date != 'None' else None
            results = vs.similarity_search_with_score(memory_query, k=10, filter=filter_dict)
            docs = [doc[0].page_content for doc in results]
        except Exception as e:
            docs = None
            print(f"An error occurred during vector search: {e}")

    res_prompt = """
    cmots news articles :{cmots} 
    chat history : {history}
    Today date:{date}
    You are a stock news and stock market information bot. 
    Using only the provided News Articles and chat history, respond to the user's inquiries in detail without omitting any context. 
    The user has asked the following question: {input}
    """
    question_prompt = PromptTemplate(input_variables=["history", "cmots", "date", "input"], template=res_prompt)
    ans_chain = question_prompt | llm_stream

    async def generate_chat_res(docs, history, input_query, date):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets..."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in ans_chain.astream_events({"cmots": docs, "history": history, "input": input_query, "date": date}, version="v1"):
                    if event["event"] == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)
                
                total_tokens = cb.total_tokens / 1000
                await insert_credit_usage(user_id, plan_id, total_tokens, db_pool)

            await store_into_db(session_id, prompt_history_id, {"links": []}, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(memory_query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(docs, his, memory_query, date), media_type="text/event-stream")


@red_rag.post("/reddit_rag")
async def red_rag_bing(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    query = request.query
    valid, v_tokens, m_chat, h_chat = await query_validate(query, str(session_id), db_pool)

    his, docs, memory_query, links = '', '', '', []
    if valid != 0:
        memory_query = await memory_chain(query, m_chat)
        sr = await fetch_search_red(memory_query)
        docs, df, links = await process_search_red(sr)
        if df is not None and not df.empty:
            await insert_red(df, db_pool) # Pass the pool
        his = h_chat

    res_prompt = """
    Reddit articles: {context}
    chat history : {history}
    ...
    The user has asked the following question: {input}        
    """
    R_prompt = PromptTemplate(template=res_prompt, input_variables=["history", "context", "input"])
    ans_chain = R_prompt | llm_stream

    async def generate_chat_res(his, docs, query):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets..."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in ans_chain.astream_events({"history": his, "context": docs, "input": query}, version="v1"):
                    if event["event"] == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)

                total_tokens = cb.total_tokens / 1000
                await insert_credit_usage(user_id, plan_id, total_tokens, db_pool)

            links_data = {"links": links}
            await store_into_db(session_id, prompt_history_id, links_data, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(his, docs, memory_query), media_type="text/event-stream")


@yt_rag.post("/yt_rag")
async def yt_rag_bing(
    request: InRequest,
    session_id: int = Query(...),
    prompt_history_id: int = Query(...),
    user_id: int = Query(...),
    plan_id: int = Query(...),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    query = request.query
    valid, v_tokens, m_chat, h_chat = await query_validate(query, str(session_id), db_pool)

    his, data, memory_query, links = '', '', '', []
    if valid != 0:
        memory_query = await memory_chain(query, m_chat)
        # NOTE: Assumes get_yt_data_async and get_data are refactored to accept the db_pool for caching
        links = await get_yt_data_async(memory_query) # Pass db_pool if needed
        data = await get_data(links, db_pool) # Pass db_pool if needed
        his = h_chat

    prompt = """
    Given youtube transcripts {summaries}
    chat_history {history}
    ...
    The user has asked the following question: {query}
    """
    yt_prompt = PromptTemplate(template=prompt, input_variables=["history", "query", "summaries"])
    chain = yt_prompt | llm_stream

    async def generate_chat_res(his, data, query):
        if valid == 0:
            error_message = "The search query is not related to Indian financial markets..."
            yield error_message.encode("utf-8")
            return

        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in chain.astream_events({"history": his, "summaries": data, "query": query}, version="v1"):
                    if event["event"] == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)

                total_tokens = cb.total_tokens / 1000
                await insert_credit_usage(user_id, plan_id, total_tokens, db_pool)

            links_data = {"links": links}
            await store_into_db(session_id, prompt_history_id, links_data, db_pool)

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(query)
                history_db.add_ai_message(final_response)

        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(his, data, memory_query), media_type="text/event-stream")
