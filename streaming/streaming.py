# /aigptcur/app_service/streaming/streaming.py
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import json
import tiktoken
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, APIRouter
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from pinecone import PodSpec, Pinecone as PineconeClient
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import re
import psycopg2
from langchain.retrievers import (
    MergerRetriever,
)
from langchain.docstore.document import Document
from datetime import datetime
from langchain_pinecone import Pinecone
import google.generativeai as genai
import requests
from datetime import datetime
import re
import os
#from create_pine import insert_into_pine
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
)
from langchain.memory import ConversationBufferWindowMemory

#from langchain_postgres import PostgresChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from psycopg2 import sql
from contextlib import contextmanager
from config import chroma_server_client,llm_date,llm_stream,vs,GPT4o_mini, PINECONE_INDEX_NAME
from langchain_chroma import Chroma
import time
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse




from api.news_rag.brave_news import get_brave_results, insert_post1
from streaming.reddit_stream import fetch_search_red,process_search_red,insert_red
from streaming.yt_stream import get_data,get_yt_data_async
# from fund import agent2




from dotenv import load_dotenv
load_dotenv(override=True)

openai_api_key=os.getenv('OPENAI_API_KEY')
pg_ip=os.getenv('PG_IP_ADDRESS')
pine_api=os.getenv('PINECONE_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')
psql_url=os.getenv('DATABASE_URL')
node_key=os.getenv('node_key')

# from langchain.globals import set_debug

# set_debug(True)



# index_name = "news"
# demo_namespace='newsrag'
# index_name = "newsrag11052024"
# demo_namespace='news'
# embeddings = OpenAIEmbeddings()

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)



# pc = PineconeClient(
#  api_key=pine_api
# )


# index = pc.Index(index_name)

# #demo_namespace='newsrag'
# docsearch1 = Pinecone(
#     index, embeddings, "text", namespace=demo_namespace
# )


#llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",stream_usage=True,streaming=True)
#gpt-3.5-turbo-1106,gpt-3.5-turbo-16k
#llm1=ChatOpenAI(temperature=0.5, model="gpt-4o-2024-05-13")
# #gpt-3.5-turbo-instruct,gpt-4-turbo,gpt-4o-2024-05-13
#llm1 = ChatGoogleGenerativeAI(model="gemini-1.0-pro")




async def llm_get_date(user_query):
    today = datetime.now().strftime("%Y-%m-%d")
    #print(today)
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
    #print(response)
    date, general_user_query = split_input(response)

    return date,general_user_query,today


def split_input(input_string):
    # Split the input string at the first comma
    parts = input_string.split(',', 1)
    # Assign the parts to date and general_user_query
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
    #llm=ChatOpenAI(model="gpt-4o-2024-05-13",temperature=0.5)
    memory_chain=LLMChain(prompt=c_q_prompt,llm=llm_date)
    res=await memory_chain.arun(query=query,chat_his=m_chat)
    #print(res)

    return res

async def query_validate(query,session_id):
    res_prompt = """
    You are a highly skilled indian stock market investor and financial advisor. Your task is to validate whether a given question is related to the stock market or finance or elections or economics or general private listed companies. Additionally, if the new question is a follow-up then only use chat history to determine its validity.
    If question is asking about latest news about any company or current news or just company or trending news of any company consider it as valid question.
    Given question : {q}
    chat history : {list_qs}
    Output the result in JSON format:
    "valid": Return 1 if the question is valid, otherwise return 0.
    """


    R_prompt = PromptTemplate(template=res_prompt, input_variables=["list_qs","q"])
    # llm_chain_res= LLMChain(prompt=R_prompt, llm=GPT4o_mini)
    chain = R_prompt | GPT4o_mini | JsonOutputParser()


    db_url=psql_url
    conn = psycopg2.connect(db_url)

    # Create a cursor object
    cur = conn.cursor()

    # Execute the SQL query with the session_id as a string
    s_id=str(session_id)
    cur.execute("SELECT message FROM message_store WHERE session_id = %s", (s_id,))
    messages = cur.fetchall()
    chat=[row[0]['data']['content'] for row in messages[-2:]]
    m_chat=[row[0]['data']['content'] for row in messages[-4:]]
    h_chat=[row[0]['data']['content'] for row in messages[-6:]]
    cur.close()
    conn.close()
    #print(query)
    with get_openai_callback() as cb:
        input_data = {
            "list_qs": chat,
            "q":query
        }
        #res=llm_chain_res.predict(query=q)
        res=await chain.ainvoke(input_data)
    return res['valid'],cb.total_tokens,m_chat,h_chat



# def set_ret():
#     embeddings = OpenAIEmbeddings()
#     index_name = "newsrag11052024"
#     demo_namespace='news'

#     index = pc.Index(index_name)

#     #demo_namespace='newsrag'
#     docsearch_cmots = Pinecone(
#         index, embeddings, "text", namespace=demo_namespace
#     )

#     index_name1 = "bing-news"
#     demo_namespace1='bing'

#     index1 = pc.Index(index_name1)
#     embeddings = OpenAIEmbeddings()

#     #demo_namespace='newsrag'
#     docsearch_bing = Pinecone(
#         index1, embeddings, "text",namespace=demo_namespace1
#     )

#     return docsearch_bing,docsearch_cmots

@contextmanager
def get_db_connection(db_url):
    conn = psycopg2.connect(db_url)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        conn.close()

@contextmanager
def get_db_cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


def store_into_db_no(pid,ph_id,result_json):
    #db_url = f"postgresql://postgresql:1234@{pg_ip}/frruitmicro"
    db_url=psql_url
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Update all existing entries to set isactive to false
 
            result_json_str = json.dumps(result_json)
            #print(result_json_str)

            cur.execute("""
                INSERT INTO "streamingData" (prompt_id, prompt_history_id, source_data)
                VALUES (%s, %s, %s)
            """, (pid, ph_id, result_json_str))

            # Commit the transaction
            conn.commit()
            #print(f"Data inserted successfully with prompt_id: {pid}")

async def store_into_db(pid, ph_id, result_json):
    # Database connection URL
    db_url = psql_url

    # Convert result_json to a JSON string
    result_json_str = json.dumps(result_json)

    # Establish an async connection to the database
    conn = await asyncpg.connect(db_url)
    try:
        # Insert into the streamingData table
        await conn.execute("""
            INSERT INTO "streamingData" (prompt_id, prompt_history_id, source_data)
            VALUES ($1, $2, $3)
        """, pid, ph_id, result_json_str)
    finally:
        # Close the connection
        await conn.close()

def get_user_credits(user_id):
    url = f"https://api.frruit.co/api/users/getUserCredits?user_id={user_id}"
    headers = {
        "x-api-key": node_key
    }

    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  # Convert response to JSON
        return data['data']['remainingCredits']
    else:
        print(f"Error: {response.status_code}")


def store_into_userplan(user_id, count):
    #db_url = f"postgresql://postgresql:1234@{pg_ip}/frruitmicro"
    db_url=psql_url
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Check if user_id exists and is active
            cur.execute("""
                SELECT credits_used FROM "user_plans"
                WHERE user_id = %s AND \"isActive\" = true
            """, (user_id,))
            result = cur.fetchone()

            if result:
                # If user exists and is active, update credits_used
                current_credits = result[0]
                new_credits = current_credits + count
                cur.execute("""
                    UPDATE "user_plans"
                    SET credits_used = %s, "updatedAt" = %s
                    WHERE user_id = %s AND \"isActive\" = true
                """, (new_credits, current_time, user_id))
                #print(f"Credits updated successfully for user_id: {user_id}")
            else:
                # Insert new record if user_id does not exist or is inactive
                cur.execute("""
                    INSERT INTO "user_plans" (user_id, credits_used, "is_active")
                    VALUES (%s, %s, true)
                """, (user_id, count))
                #print(f"Data inserted successfully for user_id: {user_id}")

            # Commit the transaction
            conn.commit()


def insert_credit_usage_no(user_id, plan_id, credit_used):
    db_url=psql_url
    with get_db_connection(db_url) as conn:
        with get_db_cursor(conn) as cur:
            # Get the current time in the desired format
            current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'

            # Insert into the credit_usage table
            cur.execute("""
                INSERT INTO credit_usage (user_id, plan_id, credit_used, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, plan_id, credit_used, current_time, current_time))

            # Commit the transaction
            conn.commit()
            #print(f"Data inserted successfully for user_id: {user_id}, plan_id: {plan_id}")

import asyncpg
async def insert_credit_usage(user_id, plan_id, credit_used):
    # Database connection URL
    db_url = psql_url

    # Get the current time in the desired format
    current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + '+00:00'

    # Create a connection pool for efficient resource management
    conn = await asyncpg.connect(db_url)
    try:
        # Insert into the credit_usage table
        await conn.execute("""
            INSERT INTO credit_usage (user_id, plan_id, credit_used, "createdAt", "updatedAt")
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, plan_id, credit_used, current_time, current_time)
    finally:
        # Close the connection
        await conn.close()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    history = PostgresChatMessageHistory(
        str(session_id), psql_url
    )
    return history.messages

def save_response_to_json(session_id: str, response_data: dict):
    """Saves the LLM response data to a JSON file."""
    try:
        # Create a unique filename using session_id and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_response_stream_{session_id}_{timestamp}.json"
        
        # Define a path to save the file, e.g., a 'response_logs' directory
        log_dir = "response_logs"
        os.makedirs(log_dir, exist_ok=True) # Ensure the directory exists
        filepath = os.path.join(log_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)
        print(f"INFO: Successfully saved response to {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save response to JSON file: {e}")

class InRequest(BaseModel):
    query: str


app = FastAPI()
cmots_rag = APIRouter()
web_rag = APIRouter()
red_rag=APIRouter()
yt_rag=APIRouter()


# client=chroma_server_client
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vs= Chroma(
#     client=client,
#     collection_name="cmots_news",
#     embedding_function=embeddings,)

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    pass
    # if x_api_key != OPENAI_API_KEY:
    #     raise HTTPException(
    #         status_code=HTTP_403_FORBIDDEN,
    #         detail="Invalid or missing API Key",
    #     )



@cmots_rag.post("/cmots_rag")
async def cmots_only(
    request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...)
):
    query = request.query
    valid, v_tokens,m_chat,h_chat=await query_validate(query, session_id)
    
    if valid == 0:
        docs, his, memory_query, date = '', '', '', ''
    else:
        memory_query=await memory_chain(query,m_chat)
        print(memory_query)
        date,user_q,t_day=await llm_get_date(memory_query)
        user_q=memory_query
        his =h_chat
        try:
            if date == 'None':
                results = vs.similarity_search_with_score(
                    memory_query,
                    k=10,
                )
                docs=[doc[0].page_content for doc in results]
            else:
                results = vs.similarity_search_with_score(
                    memory_query,
                    k=10,
                    filter={"date": {"$gte": int(date)}},
                )
                docs=[doc[0].page_content for doc in results]
        except Exception as e:
            docs = None
            print(f"An error occurred: {e}")

        res_prompt = """
        cmots news articles :{cmots} 
        chat history : {history}
        Today date:{date}
        You are a stock news and stock market information bot. 
        
        use the date provided in the metadata to answer the user query if the user is asking in specific time periods.
        If the same question {input} present in chat_history ignore that answer present in chat history dont consider that answer while generating final answer.
        give prority to latest date provided in metadata while answering user query.
        
        Using only the provided News Articles and chat history, respond to the user's inquiries in detail without omitting any context. 
        Provide relevant answers to the user's queries, adhering strictly to the content of the given articles and chat history.
        Dont start answer with based on . Dont provide extra information just provide answer what user asked.
        Answer should be very detailed in point format and preicise ,answer only based on user query and news articles.
        **If You cant find answer in provided articles dont make up answer on your own**
        *IF USER QUESTION IS ABOUT BUY/SELL THEN FORMULATE NICE ANSWER CONSISTS OF THIS -'Frruit is an AI-powered capital markets search engine designed to help you search , discover and analyze market data to make better-informed decisions. However, Frruit does not provide personalized investment advice or recommendations to buy or sell specific securities. For tailored investment guidance, please consult with a licensed financial advisor(USE THIS TEXT ONLY WHEN USER ASKING ABOUT BUY/SELL QUESTIONS).'AND PROVIDE DATA YOU HAVE SO THAT USER CAN DECIDE*

        **DONT PROVIDE ANY EXTRA INFORMATION APART FROM USER QUESTION AND ANSWER SHOULD BE IN PROPER MARKDOWN FORMATTING**
        
        The user has asked the following question: {input}
        """

        question_prompt = PromptTemplate(
            input_variables=["history", "cmots", "date","input"], template=res_prompt
        )
        ans_chain=question_prompt | llm_stream

    async def generate_chat_res(docs,history,input,date):
        if valid == 0:
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets. Please ensure your query focuses on topics like finance, investing, stocks, or other related subjects to maintain platform quality. If your query is relevant but isn’t yielding the desired results, consider refining it to improve accuracy and alignment with financial market topics"
            error_chunks = error_message.split('. ')
            
            for chunk in error_chunks:
                yield chunk.encode("utf-8")
                await asyncio.sleep(1)
                
            return
        
        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in ans_chain.astream_events(
                    {"cmots": docs, "history": history, "input": input,"date":date},
                    version="v1"
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)
                
                total_tokens = cb.total_tokens / 1000
                await insert_credit_usage(user_id, plan_id, total_tokens)
                print(f"SUCCESS: Token usage captured: {total_tokens * 1000}")

            links_data = {"links": []}
            await store_into_db(session_id,prompt_history_id,links_data)
        
            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(memory_query)
                history_db.add_ai_message(final_response)
        
        except asyncio.CancelledError:
            print("INFO: Client disconnected, streaming was cancelled.")
        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(docs,his,memory_query,date), media_type="text/event-stream")


@web_rag.post("/web_rag")
async def web_rag_mix(
    request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...)
):
    query = request.query 
    valid, v_tokens, m_chat, h_chat = await query_validate(query, session_id)
    
    if valid == 0:
        matched_docs, docs, memory_query, t_day, his, links = '', '', '', '', '', []
    else:
        # **APPROACH 1 & 2 FIX**: Use the original query for retrieval and the memory_query for generation.
        original_query = query
        memory_query = await memory_chain(query, m_chat)
        
        # **DIAGNOSTIC LOGGING**
        print(f"DEBUG: Original User Query: '{original_query}'")
        print(f"DEBUG: Reformulated Memory Query: '{memory_query}'")

        date, user_q, t_day = await llm_get_date(memory_query)
        
        # **FIX**: Use the 'original_query' for the vector search to ensure a direct match.
        pinecone_results_with_scores = vs.similarity_search_with_score(original_query, k=15)
        
        from api.news_rag.scoring_service import scoring_service
        # **FIX**: Pass the 'original_query' to the sufficiency check.
        sufficiency_score = scoring_service.assess_context_sufficiency(original_query, pinecone_results_with_scores)
        
        web_passages = []
        if sufficiency_score < 0.7: # Threshold can be adjusted
            print(f"DEBUG: Sufficiency score {sufficiency_score:.2f} is below threshold. Triggering Brave search.")
            articles, df = await get_brave_results(memory_query)
            
            if articles and df is not None and not df.empty:
                insert_post1(df)
                from api.news_rag.brave_news import BraveNews
                brave_api_key = os.getenv('BRAVE_API_KEY')
                searcher = BraveNews(brave_api_key)
                scrape_tasks = [searcher._fetch_and_parse_url_async(a.get('source_url')) for a in articles if a.get('source_url')]
                full_contents = await asyncio.gather(*scrape_tasks)

                try:
                    documents_to_add = []
                    for i, a in enumerate(articles):
                        full_content = full_contents[i] if i < len(full_contents) else ""
                        # **FIX**: Use the full scraped content for ingestion
                        page_content = f"Title: {a.get('title', '')}\nSnippet: {a.get('description', '')}\nFull Content: {full_content}"
                        
                        documents_to_add.append(Document(
                            page_content=page_content,
                            metadata={
                                "title": a.get('title', ''),
                                "link": a.get('source_url', ''),
                                "snippet": a.get('description', ''),
                                "publication_date": a.get('source_date', ''),
                                "date": a.get('date_published', ''),
                                "source": "brave_search"
                            }
                        ))

                    ids_to_add = [
                        f"brave_{hash(a.get('source_url', f'doc_{i}'))}"
                        for i, a in enumerate(articles)
                    ]
                    vs.add_documents(documents=documents_to_add, ids=ids_to_add)
                    print(f"DEBUG: Successfully upserted {len(documents_to_add)} documents to {PINECONE_INDEX_NAME}")
                except Exception as e:
                    print(f"WARNING: Failed to upsert documents to Pinecone: {e}")
                
                web_passages = [
                    {
                        "text": f"{a.get('title', '')} {a.get('description', '')}",
                        "metadata": {
                            "title": a.get('title', ''),
                            "link": a.get('source_url', ''),
                            "snippet": a.get('description', ''),
                            "publication_date": a.get('source_date', ''),
                            "date": a.get('date_published', ''),
                            "source": "brave_search"
                        }
                    }
                    for a in articles
                ]
        else:
            print(f"DEBUG: Sufficiency score {sufficiency_score:.2f} is sufficient. Skipping Brave search.")

        
        pinecone_passages = [
            {"text": doc.page_content, "metadata": {**doc.metadata, "score": score}}
            for doc, score in pinecone_results_with_scores
        ]
        
        all_passages_map = {}
        for p in web_passages + pinecone_passages:
            link = p["metadata"].get("link") or f"nolink_{hash(p['text'])}"
            if link not in all_passages_map:
                all_passages_map[link] = p
        all_passages = list(all_passages_map.values())

        if all_passages:
            reranked_passages = await scoring_service.score_and_rerank_passages(
                question=memory_query, 
                passages=all_passages
            )
            matched_docs = scoring_service.create_enhanced_context(reranked_passages)
            links = [p["metadata"]["link"] for p in reranked_passages if "link" in p.get("metadata", {})]
        else:
            matched_docs = ""
            links = []

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
            error_message = (
                "The search query you're trying to use does not appear to be related to the Indian financial markets..."
            )
            for chunk in error_message.split('. '):
                yield chunk.encode("utf-8")
                await asyncio.sleep(1)
            return

        final_response = ""
        try:
            # Stream the LLM output
            async for event in ans_chain.astream_events(
                {"bing": matched_docs, "history": history, "input": query, "date": t_day},
                version="v1"
            ):
                if event["event"] == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        final_response += content
                        yield content.encode("utf-8")
                        await asyncio.sleep(0.01)

            # --- Manual token counting ---
            prompt_text = f"{matched_docs}\n{history}\n{query}\n{t_day}"
            prompt_tokens = count_tokens(prompt_text)
            completion_tokens = count_tokens(final_response)
            total_tokens = prompt_tokens + completion_tokens

            await insert_credit_usage(user_id, plan_id, total_tokens / 1000)
            print(f"SUCCESS: Token usage captured: {total_tokens}")

            await store_into_db(session_id, prompt_history_id, {"links": links})

            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(memory_query)
                history_db.add_ai_message(final_response)

        except asyncio.CancelledError:
            print("INFO: Client disconnected, streaming was cancelled.")
        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(matched_docs, memory_query, t_day, his), media_type="text/event-stream")




@red_rag.post("/reddit_rag")
async def red_rag_bing(
    request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...)
):
    query = request.query 
    valid,v_tokens,m_chat,h_chat=await query_validate(query,session_id)
    if valid == 2:
        his,docs,query, links= '', '', '', []
    else:
        query=await memory_chain(query,m_chat)
        sr=await fetch_search_red(query)
        docs,df,links=await process_search_red(sr)
        if df is not None:
            await insert_red(df)
        his=h_chat

        res_prompt = """
        Reddit articles: {context}
        chat history : {history}
        ...
        The user has asked the following question: {input}        
        """
        R_prompt = PromptTemplate(template=res_prompt, input_variables=["history","context","input"])
        ans_chain=R_prompt | llm_stream
    
    async def generate_chat_res(his,docs,query):
        if valid == 2:
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets..."
            for chunk in error_message.split('. '):
                yield f"{chunk}.".encode("utf-8")
                await asyncio.sleep(1)
            return
        
        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in ans_chain.astream_events(
                    {"history":his,"context": docs,"input": query},
                    version="v1"
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)
                
                total_tokens = cb.total_tokens / 1000
                await insert_credit_usage(user_id, plan_id, total_tokens)
                print(f"SUCCESS: Token usage captured: {total_tokens * 1000}")

            links_data = {"links": links}
            await store_into_db(session_id,prompt_history_id,links_data)
        
            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(query)
                history_db.add_ai_message(final_response)
        
        except asyncio.CancelledError:
            print("INFO: Client disconnected, streaming was cancelled.")
        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(his,docs,query), media_type="text/event-stream")


@yt_rag.post("/yt_rag")
async def yt_rag_bing(request: InRequest, 
    session_id: int = Query(...), 
    prompt_history_id: int = Query(...), 
    user_id: int = Query(...), 
    plan_id: int = Query(...)
):
    query = request.query 
    valid,v_tokens,m_chat,h_chat=await query_validate(query,session_id)
    if valid == 2:
        his,data,query, links= '', '', '', []
    else:
        links =await get_yt_data_async(query)
        data = await get_data(links)
        his=h_chat

        prompt = """
        Given youtube transcripts {summaries}
        chat_history {history}
        ...
        The user has asked the following question: {query}
        """
        yt_prompt = PromptTemplate(template=prompt, input_variables=["history","query", "summaries"])
        chain = yt_prompt | llm_stream
    
    async def generate_chat_res(his,data,query):
        if valid == 2:
            error_message = "The search query you're trying to use does not appear to be related to the Indian financial markets..."
            for chunk in error_message.split('. '):
                yield f"{chunk}.".encode("utf-8")
                await asyncio.sleep(1)
            return
        
        final_response = ""
        try:
            with get_openai_callback() as cb:
                async for event in chain.astream_events(
                    {"history":his,"summaries": data, "query": query},
                    version="v1"
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            final_response += content
                            yield content.encode("utf-8")
                            await asyncio.sleep(0.01)

                total_tokens = cb.total_tokens / 1000
                await insert_credit_usage(user_id, plan_id, total_tokens)
                print(f"SUCCESS: Token usage captured: {total_tokens * 1000}")

            links_data = {"links": links}
            await store_into_db(session_id,prompt_history_id,links_data)
            
            if final_response:
                history_db = PostgresChatMessageHistory(str(session_id), psql_url)
                history_db.add_user_message(query)
                history_db.add_ai_message(final_response)

        except asyncio.CancelledError:
            print("INFO: Client disconnected, streaming was cancelled.")
        except Exception as e:
            print(f"ERROR: An error occurred during streaming: {e}")
            yield b"An error occurred while generating the response."

    return StreamingResponse(generate_chat_res(his,data,query), media_type="text/event-stream")