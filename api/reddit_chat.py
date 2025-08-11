import bs4
import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.callbacks import get_openai_callback
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import uuid
from langchain.memory import PostgresChatMessageHistory
import asyncpg
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from fastapi import FastAPI, HTTPException, APIRouter, Query
from pydantic import BaseModel
from typing import Dict, Any
import os
import pandas as pd
from psycopg2 import sql
from config import chroma_server_client, GPT4o_mini

from dotenv import load_dotenv

load_dotenv(override=True)

psql_url=os.getenv('DATABASE_URL')

client = chroma_server_client
embedding_function = SentenceTransformerEmbeddings(model_name="thenlper/gte-base")
em = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-base")
post_collection = client.get_or_create_collection(name="post_data", embedding_function=em)
comment_collection = client.get_or_create_collection(name="comments", embedding_function=em)
llm_openai = ChatOpenAI(model="gpt-3.5-turbo-16k")
#db3 = Chroma(persist_directory="./new_database", embedding_function=embedding_function, collection_name='post_data')
#db_c = Chroma(persist_directory="./new_database", embedding_function=embedding_function, collection_name='comments')

# This creates a new collection named 'post_data' on your central server
db3 = Chroma(
    client=chroma_server_client,
    collection_name="post_data",
    embedding_function=embedding_function,
)
# This creates another collection named 'comment_data' on your central server
db_c = Chroma(
    client=chroma_server_client,
    collection_name="comment_data",
    embedding_function=embedding_function,
)


llm_openai = GPT4o_mini


def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


def extract_content_and_metadata(documents):
    extracted_data = []
    for doc in documents:
        content = doc.page_content.strip()
        content = content.replace('{', '').replace('}', '')
        metadata = doc.metadata
        metadata_str = ', '.join(f'{key}: {value}' for key, value in metadata.items())
        metadata_str = metadata_str.replace('{', '').replace('}', '')
        extracted_data.append(content)
        extracted_data.append(metadata_str)
    return extracted_data


async def memory_chain(query, session_id):
    connection = await asyncpg.connect(psql_url)
    rows = await connection.fetch("SELECT message FROM message_store WHERE session_id = $1", session_id)
    await connection.close()

    chat = [row['message'] for row in rows][:6]
    contextualize_q_system_prompt = """Given a chat history {chat_his} and the latest user question {query} which might reference context in the chat history, formulate a standalone question if needed include time/date part also based on user previous question. Which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and if the latest question doesn't have any relevancy to chat history return it as is."""

    c_q_prompt = PromptTemplate(template=contextualize_q_system_prompt, input_variables=['chat_his', 'query'])
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2)
    memory_chain = LLMChain(prompt=c_q_prompt, llm=llm)
    res = memory_chain.predict(query=query, chat_his=chat)

    return res


def relevant_posts(user_query):
    relevant_post_ids = post_collection.query(query_texts=user_query, n_results=30)["ids"][0]
    filter_dict = {"post_id": {"$in": relevant_post_ids}}

    k1 = 15
    k2 = 15

    while True:
        docs = db3.similarity_search(user_query, k=k2)
        c = db_c.similarity_search(user_query, k=k1, filter=filter_dict)

        docs_str = "\n".join([doc.page_content for doc in docs])
        c_str = "\n".join([doc.page_content for doc in c])

        num_tokens = count_tokens(user_query + docs_str + c_str)
        print(f"numtok{num_tokens}")
        print(num_tokens)

        if num_tokens <= 16173:
            break
        k1 -= 3
        k2 -= 1

    if k1 <= 0:
        c = db_c.similarity_search(user_query, k=5, filter=filter_dict)

    docs = extract_content_and_metadata(docs)
    c = extract_content_and_metadata(c)

    return docs, c


def extract_info_from_url(url):
    document_id = url.split('/')[-2]
    docs = db3.similarity_search(document_id, k=1)

    if docs:
        doc = docs[0]
        try:
            content = doc.page_content.strip()
            title_start = content.find('"Title":') + len('"Title":')
            title_end = content.find('"Content":', title_start)
            title = content[title_start:title_end].strip().strip('"')

            content_start = content.find('"Content":', title_end) + len('"Content":')
            content_end = content.find('"', content_start)
            content = content[content_start:content_end].strip().strip('"')

            return {'title': title, 'content': content, 'url': url}
        except Exception as e:
            print(f"Error processing document: {e}")
            return None
    else:
        return None


def process_response(response_data):
    extracted_data = []
    for link in response_data['links']:
        info = extract_info_from_url(link)
        if info:
            extracted_data.append(info)
    return extracted_data


async def insert_post(sources):
    conn = await asyncpg.connect(psql_url)
    for source in sources:
        await conn.execute("""
            INSERT INTO source_data (title, source_url)
            VALUES ($1, $2)
        """, source['title'], source['url'])
    await conn.close()


import time

async def reddit_rag(user_query, session_id):
    start_time = time.time()
    try:
        # Measure memory_chain time
        memory_chain_start = time.time()
        user_query = await memory_chain(user_query, session_id)
        memory_chain_end = time.time()
        print(f"memory_chain took: {memory_chain_end - memory_chain_start} seconds")

        # Measure relevant_posts time
        relevant_posts_start = time.time()
        docs, c = relevant_posts(user_query)
        relevant_posts_end = time.time()
        print(f"relevant_posts took: {relevant_posts_end - relevant_posts_start} seconds")

        # Measure LLMChain time
        llm_chain_start = time.time()
        prompt_template = PromptTemplate(
            template="""
                You are a reddit chatbot that has access to reddit data, and you are able to answer queries a user asks
                reddit data available that is relevant to the query: 
                posts: {docs}
                comments: {c}
                Answer this user's query:{user_query} according to what reddit users have to say, keep it less than 750 words
                
                Try to answer focusing more on relatively newer data and only mention past data as context about what the opinion was previously
                take into consideration UTC ie when the posts/comments were created and draw overall conclusions accordingly about information and opinions in the past, present and assumptions about the future.        
                THE OUTPUT SHOULD BE IN JSON FORMAT:
                
                'answer':answer generated,
                'links' :links that are related to user query and used in generating answer.  
            """,
            input_variables=["user_query", "docs", "c"]
        )

        llm_chain = LLMChain(prompt=prompt_template, llm=llm_openai)
        chain = prompt_template | llm_openai | JsonOutputParser()

        history = PostgresChatMessageHistory(
            connection_string=psql_url,
            session_id=session_id
        )

        history.add_user_message(user_query)

        input_data = {
            "user_query": user_query,
            "docs": docs,
            "c": c
        }

        with get_openai_callback() as cb:
            raw_response = chain.invoke(input_data)
            if not isinstance(raw_response, dict):
                raise ValueError("Invalid response from LLMChain")

            response_str = json.dumps(raw_response)
            print(raw_response)

            history.add_ai_message(raw_response["answer"])

            res = {
                "Response": raw_response['answer'],
                "links": raw_response['links'],
                "Total_Tokens": cb.total_tokens,
                "Prompt_Tokens": cb.prompt_tokens,
                "Completion_Tokens": cb.completion_tokens,
            }
        llm_chain_end = time.time()
        print(f"LLMChain took: {llm_chain_end - llm_chain_start} seconds")

        # Measure insert_post time
        insert_post_start = time.time()
        sources = process_response(res)
        await insert_post(sources)
        insert_post_end = time.time()
        print(f"insert_post took: {insert_post_end - insert_post_start} seconds")

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time} seconds")

        return res

    except Exception as e:
        print(f"Error: {e}")
        return {"Error": str(e)}

router = APIRouter()


class QueryModel(BaseModel):
    user_query: str


@router.post("/reddit_chat2")
async def query(data: QueryModel, session_id: str = Query(...)):
    try:
        result = await reddit_rag(data.user_query, session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))