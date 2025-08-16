import asyncio
import aiohttp
import time
import asyncpg
import requests
from langchain_community.callbacks import get_openai_callback
from langchain_text_splitters import TokenTextSplitter
from googleapiclient.discovery import build
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
from langchain_core.output_parsers import JsonOutputParser
import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
import tiktoken
import chromadb
import concurrent.futures
import openai
from langchain_openai import OpenAIEmbeddings
import psycopg2
from pytube import YouTube 
from psycopg2 import sql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv
from config import GPT4o_mini, DB_POOL
from api.brave_searcher import BraveVideoSearch
load_dotenv()

llm=GPT4o_mini

google_custom_search_api_key = os.getenv('google_custom_search_api_key')
google_cx = os.getenv('google_cx')
youtube_api_key = os.getenv('youtube_api_key')
psql_url=os.getenv('DATABASE_URL')

youtube = build('youtube', 'v3', developerKey=youtube_api_key)


def get_length(url):
    try:
        yt = YouTube(url)  ## this creates a YOUTUBE OBJECT
        video_length = yt.length 
        return video_length
    except Exception as e:
        print(f"Error getting video length for {url}: {e}")
        return float('inf')  # Return a large number to skip this video


# def get_bing_search_results(query, count=10):
#     headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
#     params = {"q": query, "count": 5, "mkt": "en-IN", 'freshness': 'Month', "cc": 'IND'}
#     search_url = "https://api.bing.microsoft.com/v7.0/search"
#     try:
#         response = requests.get(search_url, headers=headers, params=params)
#         response.raise_for_status()
#         search_results = response.json()
#         return search_results.get('webPages', {}).get('value', [])
#     except requests.RequestException as e:
#         print(f"Error fetching Bing search results: {e}")
#         return []

# def search_youtube_videos(query):
#     youtube_search_results = []
#     one_month_ago = datetime.utcnow() - timedelta(days=30)
#     one_month_ago_iso = one_month_ago.isoformat("T") + "Z"
#     request = youtube.search().list(
#         part='snippet',
#         q=query,
#         type='video',
#         maxResults=5,
#         regionCode='IN',
#         publishedAfter=one_month_ago_iso
#     )
#     try:
#         response = request.execute()
#         for item in response['items']:
#             video_id = item['id']['videoId']
#             video_url = f'https://www.youtube.com/watch?v={video_id}'
#             if get_length(video_url) < 3601:
#                 youtube_search_results.append(video_url)
#     except Exception as e:
#         print(f"Error searching YouTube videos: {e}")
#     return youtube_search_results

async def get_video_statistics(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        request = youtube.videos().list(part='statistics,snippet', id=video_id)
        response = await asyncio.to_thread(request.execute)

        if not response['items']:
            return None

        snippet = response['items'][0]['snippet']
        return snippet['publishedAt']
    except Exception as e:
        print(f"Error getting video statistics for {video_id}: {e}")
        return None

async def get_video_transcript(video_id):
    try:
        transcript_en = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id, languages=['en'])
        text_en = ' '.join([entry['text'] for entry in transcript_en])
        summary = await asyncio.to_thread(summarize_transcript, text_en)
        return summary
    except (NoTranscriptFound, NoTranscriptAvailable, TranscriptsDisabled):
        return None
    except Exception as e:
        print(f"Error getting transcript for video {video_id}: {e}")
        return None

async def turl(video_url):
    try:
        video_id_match = re.search(r"v=([^\&\?\/]+)", video_url)
        video_id = video_id_match.group(1) if video_id_match else ""
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(thumbnail_url) as response:
                if response.status == 200:
                    return thumbnail_url
                else:
                    return None
    except Exception as e:
        print(f"Error getting thumbnail for video {video_url}: {e}")
        return None

async def insert_into_database(source_url, image_url, title, description, s_date, youtube_summary):
    if not DB_POOL:
        print("ERROR: Database pool not initialized.")
        return
    try:
        async with DB_POOL.acquire() as conn:
            data = (source_url, image_url, title, description, s_date, youtube_summary)

            check_query = """
                SELECT 1 FROM source_data WHERE source_url = $1
            """
            exists = await conn.fetchrow(check_query, source_url)

            if not exists:
                insert_query = """
                    INSERT INTO source_data (source_url, image_url, title, description, source_date, youtube_summary)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """
                await conn.execute(insert_query, *data)
                print("Data inserted successfully into PostgreSQL")
            else:
                print("Source URL already exists. Skipping insertion.")

    except (Exception, asyncpg.Error) as error:
        print("Error while inserting data into PostgreSQL:", error)

async def get_summary_from_database(source_url):
    if not DB_POOL:
        print("ERROR: Database pool not initialized.")
        return None
    try:
        async with DB_POOL.acquire() as conn:
            select_query = """
                SELECT youtube_summary FROM source_data WHERE source_url = $1
            """
            result = await conn.fetchrow(select_query, source_url)

            if result:
                return result['youtube_summary']
            else:
                return None

    except (Exception, asyncpg.Error) as error:
        print("Error while fetching data from PostgreSQL:", error)
        return None

async def extract_youtube_video_data(session, url):
    start_time = time.time()
    try:
        async with session.get(url) as response:
            content = await response.text()
        soup = BeautifulSoup(content, 'html.parser')
        
        title_tag = soup.find("meta", {"name": "title"})
        title = title_tag["content"] if title_tag else "No Title Available"
        
        description_tag = soup.find("meta", {"name": "description"})
        description = description_tag["content"] if description_tag else "No Description Available"
        
        video_id_match = re.search(r"v=([^\&\?\/]+)", url)
        video_id = video_id_match.group(1) if video_id_match else ""
        
        summary = await get_summary_from_database(url)
        if not summary:
            transcript = await get_video_transcript(video_id) if video_id else ""
            summary = transcript
        
        date = await get_video_statistics(video_id) if video_id else {}
        t_image = await turl(url)

        end_time = time.time()
        execution_time = end_time - start_time

        await insert_into_database(url, t_image, title, description, date, summary)

        print(f"Execution time for URL {url}: {execution_time:.2f} seconds")
        
        return title, description, summary, url
    except Exception as e:
        print(f"Error extracting data for video {url}: {e}")
        return "No Title Available", "No Description Available", "No Summary Available", url

async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [extract_youtube_video_data(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

def count_tokens(text, model_name="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def summarize_transcript_of_each_video(trans):
    try:
        prompt = """
        Based on the following video transcript, provide a structured summary focusing on key insights and critical information.
        Organize the summary into a paragraph with a clear, comprehensive flow.
        The paragraph should cover all points in the video transcript and should be in 1000 words.
        {transcript}
        
        Provide a structured summary as described above.
        """

        yt_prompt = PromptTemplate(template=prompt, input_variables=["transcript"])
        chain = LLMChain(prompt=yt_prompt, llm=llm)

        input_data = {
            "transcript": trans
        }
        res = chain.invoke(input_data)
        return res['text']
    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return ""

def summarize_transcript(trans):
    try:
        c = count_tokens(trans)
        summary = ""

        if c < 16000:
            summary = summarize_transcript_of_each_video(trans)
        else:
            text_splitter = TokenTextSplitter(chunk_size=16000, chunk_overlap=0)
            texts = text_splitter.split_text(trans)
            for t in texts:
                chunk_summary = summarize_transcript_of_each_video(t)
                summary += chunk_summary
        
        return summary
    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return ""

def get_yt_data_async(query):
    """
    This is a synchronous wrapper for the async video search function.
    It's used here to maintain compatibility with the existing synchronous yt_chat function.
    """
    async def run_async_search():
        brave_api_key = os.getenv('BRAVE_API_KEY')
        if not brave_api_key:
            print("ERROR: BRAVE_API_KEY not found in environment variables.")
            return []
            
        brave_video_searcher = BraveVideoSearch(brave_api_key)
        
        print(f"DEBUG: Searching Brave Videos for query: '{query}'")
        video_urls = await brave_video_searcher.search(query, max_results=10)
        print(f"DEBUG: Found {len(video_urls)} potential videos from Brave Search.")

        # Asynchronously filter videos by relevance and length
        async def filter_video(url):
            loop = asyncio.get_event_loop()
            try:
                video_length = await loop.run_in_executor(None, get_length, url)
                
                if not (60 < video_length < 3601):
                    return None

                video_id_match = re.search(r"v=([^\&\?\/]+)", url)
                video_id = video_id_match.group(1) if video_id_match else ""
                
                if not video_id:
                    return None

                # Since check_youtube_relevance is async, we can await it directly
                # but we need to get title first which might be sync
                title = await loop.run_in_executor(None, lambda: build('youtube', 'v3', developerKey=youtube_api_key).videos().list(part="snippet", id=video_id).execute().get('items', [{}])[0].get('snippet', {}).get('title', ''))

                relevance_check = 1 # Placeholder, as check_youtube_relevance is not defined here.
                                  # The primary filtering is now length and search relevance.
                
                if relevance_check == 1:
                    print(f"DEBUG: Including relevant video: {title}")
                    return url
                else:
                    return None
            except Exception as e:
                print(f"ERROR: Failed to process video URL {url}: {e}")
                return None

        tasks = [filter_video(url) for url in video_urls]
        results = await asyncio.gather(*tasks)
        
        filtered_urls = [url for url in results if url is not None]
        return filtered_urls[:5]

    return asyncio.run(run_async_search())

def generate_final_summary(query, summaries):
    try:
        start_time = time.time()

        prompt = """
        Given youtube transcripts {summaries}
        Using only the provided youtube video transcripts, respond to the user's inquiries in detail without omitting any context. 
        Provide relevant answers to the user's queries, adhering strictly to the content of the given transcripts.
        Dont start answer with based on . Dont provide extra information just provide answer what user asked.

        The user has asked the following question: {query}

        The output should contain generated answer and news urls that used while generating the answer. 
        
        The output should be in json format:
        "answer": summary generated based on user query and video summaries.
        "links": provide all youtube links which are relevant to user query present in metadata.
        Provide a structured final summary as described above.
        """
        yt_prompt = PromptTemplate(template=prompt, input_variables=["query", "summaries"])
        output_parser = JsonOutputParser()
        ans_chain = yt_prompt | llm | output_parser

        input_data = {
            "query": query,
            "summaries": summaries
        }

        with get_openai_callback() as cb:
            final = ans_chain.invoke(input_data)

        return {
            "Response": final['answer'],
            "links": final['links'],
            "Total_Tokens": cb.total_tokens,
            "Prompt_Tokens": cb.prompt_tokens,
            "Completion_Tokens": cb.completion_tokens,
        }
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return {}

async def yt_chat(query,session_id):
    try:
        start_time = time.time()
        links = get_yt_data_async(query)
        data = await main(links)
        res = generate_final_summary(query, data)
        end_time = time.time()
        execution_time = end_time - start_time
        print(res)
        print(f"Execution Time: {execution_time} seconds")

        return res
    except Exception as e:
        print(f"Error processing async task: {e}")
        return {}
