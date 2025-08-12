import os
import json
import asyncio
from langchain import PromptTemplate, LLMChain
from langchain_community.chat_message_histories import (
    PostgresChatMessageHistory,
)
from fastapi import FastAPI, HTTPException
import requests
import re
from urllib.parse import urlparse
import asyncpg
import pandas as pd
from fastapi.responses import StreamingResponse
from openai import OpenAI
from azure.cognitiveservices.search.websearch import WebSearchClient
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.docstore.document import Document
import psycopg2
import pandas as pd
from psycopg2 import sql
from config import DB_POOL # Import the DB_POOL



from dotenv import load_dotenv
load_dotenv(override=True)

pg_ip=os.getenv('PG_IP_ADDRESS')
psql_url=os.getenv('DATABASE_URL')



def fetch_search_red1(query):
    bing_api_key = os.getenv('BING_API_KEY')
    subscription_key = bing_api_key
    #print(subscription_key)
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    params = {
    "count": "10",
    "cc":'IND',
    #"freshness": "Month",
    "q": f"site:reddit.com {query}",
    #"textDecorations": True,
    "mkt": "en-IN",
    #"responseFilter": "News",
    "sortBy": "Date",
    }
        


    try:
        # Send the GET request to the API
        response = requests.get(url=endpoint, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for any HTTP error codes
        search_results = response.json()

        if search_results and isinstance(search_results, dict) and 'webPages' in search_results:
            web_pages = search_results.get('webPages', {})
            if 'value' in web_pages and web_pages['value']:
                values = web_pages['value']
                return search_results
            else:
                print("No 'value' found in 'webPages'")
                return None
        else:
            print("'webPages' not found in search_results or invalid structure")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")

    return None



def process_search_red1(search_results):
    # Safely get the 'value' key and check if it's not None and not empty
    if search_results is None:
        values=None
    else:
        values = search_results['webPages']['value']
    #print(len(values))
    
    if values:
        news = []
        for item in search_results['webPages']['value']:
            #print(item)

            news.append({
                    "title": item.get("name", None),
                    "source_url": item.get("url", None),
                    "image_url": None,  # Explicitly set to None
                    "description": item.get("snippet", None),
                    #"heading": item.get('siteName', None),
                    # "image": item.get("image", {}).get("thumbnail", {}).get("contentUrl", None),
                    "source_date": item.get("dateLastCrawled", None),
                    #"date_published": item.get("datePublished", "abc"),
                })
            
        if len(news):
        
            # Create a DataFrame from the news list
            df = pd.DataFrame(news)
            
            # Clean up the 'title' and 'description' columns
            #df['title'] = df['title'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))
            #df['description'] = df['description'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', x))    
            # Convert 'date_published' to a specific format and type
            #df["date_published"] = pd.to_datetime(df["date_published"]).dt.strftime('%Y%m%d').astype(int)
            
            # Drop rows with any NaN values
            #df.dropna(inplace=True)
            
            #print(df.columns)
            #print(df[['source_date', 'date_published']])
            filtered_articles = [article["title"] + article["description"] for article in news]
            links = [article["source_url"] for article in news]
            return filtered_articles,df,links
        else:
            return None,None,None
    else:
        return None,None,None
    


import httpx
async def fetch_search_red(query):
    bing_api_key = os.getenv('BING_API_KEY')
    subscription_key = bing_api_key
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    params = {
        "count": "10",
        "cc": 'IND',
        "q": f"site:reddit.com {query}",
        "mkt": "en-IN",
        "sortBy": "Date",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url=endpoint, headers=headers, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            search_results = response.json()

            if search_results and isinstance(search_results, dict) and 'webPages' in search_results:
                web_pages = search_results.get('webPages', {})
                if 'value' in web_pages and web_pages['value']:
                    return search_results
                else:
                    print("No 'value' found in 'webPages'")
                    return None
            else:
                print("'webPages' not found in search_results or invalid structure")
                return None
    except httpx.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None


async def process_search_red(search_results):
    if search_results is None:
        return None, None, None

    values = search_results.get('webPages', {}).get('value', [])
    if not values:
        return None, None, None

    news = [
        {
            "title": item.get("name"),
            "source_url": item.get("url"),
            "image_url": None,  # Explicitly set to None
            "description": item.get("snippet"),
            "source_date": item.get("dateLastCrawled"),
        }
        for item in values
    ]

    if news:
        df = pd.DataFrame(news)
        filtered_articles = [article["title"] + article["description"] for article in news]
        links = [article["source_url"] for article in news]
        return filtered_articles, df, links

    return None, None, None


async def insert_red(db):
    df = db
    if not DB_POOL:
        print("ERROR: Database pool not initialized.")
        return

    try:
        async with DB_POOL.acquire() as conn:
            for index, row in df.iterrows():
                source_url = row['source_url']
                image_url = None
                heading = None
                title = row['title']
                description = row['description']
                source_date = row['source_date']

                # Check if the row already exists
                exists = await conn.fetchval(
                    "SELECT 1 FROM source_data WHERE source_url = $1", source_url
                )
                if not exists:
                    # Prepare and execute the SQL query
                    insert_query = """
                        INSERT INTO source_data (source_url, image_url, heading, title, description, source_date)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """
                    await conn.execute(insert_query, source_url, image_url, heading, title, description, source_date)

    except Exception as e:
        print(f"Error in insert_red: {e}")




# def memory_chain(query,session_id):
#     connection = psycopg2.connect(f"postgresql://postgresql:1234@{pg_ip}/frruitmicro")

#     # Create a cursor
#     cursor = connection.cursor()

#     # View the table
#     table_name = 'message_store'
#     s_id=session_id
#     cursor.execute("SELECT message FROM message_store WHERE session_id = %s", (session_id,))
#     rows = cursor.fetchall()

#     # for row in rows:
#     #     print(row)

#     # Close the cursor and connection
#     cursor.close()
#     connection.close()


#     chat=rows[:6]

#     #print(type(chat))
#     # print(chat[0][0]['data']['content'])
#     # return chat
#     contextualize_q_system_prompt = """Given a chat history {chat_his}and the latest user question {query}\
#     which might reference context in the chat history, formulate a standalone question if needed include time/date part also based on user previous question.\
#     which can be understood without the chat history. Do NOT answer the question, \
#     if the latest question {query} doesnot have any relevancy to chat history return it as is with out changing user query: {query}.
    
#     """

#     c_q_prompt=PromptTemplate(template=contextualize_q_system_prompt,input_variables=['chat_his','query'])
#     llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0.5)
#     memory_chain=LLMChain(prompt=c_q_prompt,llm=llm)
#     res=memory_chain.predict(query=query,chat_his=chat)

#     return res


# app = FastAPI()

# @app.post("/reddit_rag")
# async def red_rag(query,session_id):
#     query=memory_chain(query,session_id)
#     sr=fetch_search_red(query)
#     docs,_=process_search_red(sr)


#     res_prompt = """
#     Reddit articles: {context}
#     You are a stock news and stock market information bot. 
    

#     Using only the provided Reddit Articles, respond to the user's inquiries in detail without omitting any context. 
#     Provide relevant answers to the user's queries, adhering strictly to the content of the given articles.
#     Dont start answer with based on . Dont provide extra information just provide answer what user asked.
#     Answer should be very detailed in point format and preicise and dont provide any links ,answer only based on user query and news articles#.IN PROPER markdown formating.
#     If You cant find answer in provided articles dont make up answer on your own.

#     The user has asked the following question: {input}


    
#     """

#     R_prompt = PromptTemplate(template=res_prompt, input_variables=["context","input"])
#     #llm_chain_res= LLMChain(prompt=R_prompt, llm=llm1)
#     llm1 = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",streaming=True,stream_usage=True)
   
#     ans_chain=R_prompt | llm1

#     #final_task = asyncio.to_thread(ans_chain.invoke, {"context": docs, "cmots": matched_docs, "input": query,"date":t_day})
    
    
#     async def generate_chat_res(docs,query):
#         aggregate = None
#         async for chunk in ans_chain.astream({"context": docs,"input": query}):
#             #return chunk
            
#             if chunk is not None:
#                 #print(type(chunk))
#                 answer = chunk.content
#                 aggregate = chunk if aggregate is None else aggregate + chunk
#                 if answer is not None:
#                     await asyncio.sleep(0.01) 
#                     yield answer.encode("utf-8")
#                 else:
#                     pass
#             else:
#                 print("Received None chunk")

#         #yield b"metadata: " + json.dumps(aggregate.usage_metadata).encode("utf-8")
#         #print(aggregate)
#         print(aggregate.usage_metadata)
    
#         #return aggregate.usage_metadata
#         # #background_tasks.add_task(handle_metadata, aggregate.usage_metadata if aggregate else {})
#         history = PostgresChatMessageHistory(
#         connection_string=f"postgresql://postgresql:1234@{pg_ip}/frruitmicro",
#         session_id=session_id,
#         )



#         history.add_user_message(query)
#         history.add_ai_message(aggregate)


#     #return "hello"
#     return StreamingResponse(generate_chat_res(docs,query), media_type="text/event-stream")