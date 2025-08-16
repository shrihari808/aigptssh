import asyncio
import aiohttp
from pytube.exceptions import LiveStreamError
import time
import asyncpg
import requests
from langchain_community.callbacks import get_openai_callback
from langchain_text_splitters import TokenTextSplitter
from googleapiclient.discovery import build
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from youtube_transcript_api import YouTubeTranscriptApi
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
from fastapi.responses import StreamingResponse
import chromadb
import concurrent.futures
import openai
from langchain_openai import OpenAIEmbeddings
import psycopg2
from psycopg2 import sql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
import aiohttp
import concurrent.futures
import time
from api.brave_searcher import BraveVideoSearch
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailable
from dotenv import load_dotenv
from config import GPT4o_mini, DB_POOL
load_dotenv(override=True)

#llm=GPT4o_mini

bing_api_key = os.getenv('BING_API_KEY')
google_custom_search_api_key = os.getenv('google_custom_search_api_key')
google_cx = os.getenv('google_cx')
youtube_api_key = os.getenv('youtube_api_key')

youtube = build('youtube', 'v3', developerKey=youtube_api_key)

llm=ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
psql_url=os.getenv('DATABASE_URL')

yt_prompt_template = """
Your task is to determine if a YouTube video title is relevant to a user question.

user question : {q}
Youtube title : {yt}
Output the result in JSON format:

"valid": Return 1 if the question is related, otherwise return 0.
"""
yt_prompt = PromptTemplate(template=yt_prompt_template, input_variables=["q", "yt"])
yt_chain = yt_prompt | llm | JsonOutputParser()

async def check_youtube_relevance(query: str, youtube_title: str):
    """Asynchronously checks if a YouTube title is relevant to the query."""
    if not youtube_title or youtube_title == "Video not found":
        return 0
    input_data = {"q": query, "yt": youtube_title}
    try:
        res = await yt_chain.ainvoke(input_data)
        return res.get('valid', 0)
    except Exception as e:
        print(f"  -> ERROR: LLM relevance check failed: {e}")
        return 0

def get_length(url):
    try:
        yt = YouTube(url)
        #print(yt)  ## this creates a YOUTUBE OBJECT
        video_length = yt.length
        #print(video_length)
        #print(video_length) 
        return video_length
    except Exception as e:
        #print(f"Error getting video length for {url}: {e}")
        #print(float('inf'))
        return 999999

def extract_video_id(url):
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None

def get_video_length(url):
    video_id=extract_video_id(url)
    url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=contentDetails&key={youtube_api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        duration = data["items"][0]["contentDetails"]["duration"]
        #print(parse_duration(duration))
        return parse_duration(duration)
    else:
        print("Error fetching video details:", response.status_code)
        return None

def parse_duration(duration):
    # Converts ISO 8601 duration to seconds
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()

def get_youtube_title(video_id):
    # Build the YouTube service
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    try:
        # Make an API call to get the video details
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        # Extract the video title
        if response['items']:
            title = response['items'][0]['snippet']['title']
            return title
        else:
            return "Video not found"
    except Exception as e:
        return f"Error: {str(e)}"
    

async def get_youtube_transcript(video_id):
    rapidapi_key=os.getenv('rapid_key')  # Your API key here
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Get video details to fetch the subtitles URL
            details_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
            querystring_details = {"videoId": video_id}

            headers = {
                "x-rapidapi-key": rapidapi_key,
                "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com"
            }

            async with session.get(details_url, headers=headers, params=querystring_details) as response:
                response.raise_for_status()  # Ensure the request was successful
                data = await response.json()

            # Step 2: Extract subtitle URL from the response
            if 'subtitles' in data and 'items' in data['subtitles'] and data['subtitles']['items']:
                cc_url = data['subtitles']['items'][0]['url']
            else:
                return "No subtitles available for this video."

            # Step 3: Get the transcript using the subtitle URL
            transcript_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/subtitles"
            querystring_transcript = {"subtitleUrl": cc_url, "format": "json"}

            async with session.get(transcript_url, headers=headers, params=querystring_transcript) as trans:
                trans.raise_for_status()  # Ensure the request was successful
                transcript_data = await trans.json()

            # Step 4: Concatenate all the transcript text
            full_transcript = " ".join([item['text'] for item in transcript_data])
            summary = await asyncio.to_thread(summarize_transcript, full_transcript)
            return summary
            # print(full_transcript)

            # return full_transcript

    except Exception as e:
        return None


yt_prompt = """
Your task is to determine if a YouTube video title is relevant to a user question

user question : {q}
Youtube title : {yt}
Output the result in JSON format:

"valid": Return 1 if the question is related , otherwise return 0.

"""

y_prompt = PromptTemplate(template=yt_prompt)
#llm_chain_res= LLMChain(prompt=R_prompt, llm=GPT4o_mini)
yt_chain = y_prompt | GPT4o_mini | JsonOutputParser()


async def check_youtube_relevance(query: str, youtube_title: str):
    input_data = {
        "q": query,
        "yt": youtube_title
    }
    res = await yt_chain.ainvoke(input_data)
    return res['valid']



# def get_bing_search_results(query, count=10):
#     headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
#     params = {"q": query, "count": 5, "mkt": "en-IN", 'freshness': 'Month', "cc": 'IND'}
#     search_url = "https://api.bing.microsoft.com/v7.0/search"
#     response = requests.get(search_url, headers=headers, params=params)
#     response.raise_for_status()
#     search_results = response.json()
#     return search_results.get('webPages', {}).get('value', [])

# async def search_youtube_videos(query):
#     youtube_search_results = []
#     one_month_ago = datetime.utcnow() - timedelta(days=30)
#     one_month_ago_iso = one_month_ago.isoformat("T") + "Z"
    
#     # Replace this line with your actual YouTube API client initialization
#     request = youtube.search().list(
#         part='snippet',
#         q=query,
#         type='video',
#         maxResults=5,
#         regionCode='IN',
#         publishedAfter=one_month_ago_iso
#     )
#     response = request.execute()
#     #print(response)

#     for item in response['items']:
#         video_id = item['id']['videoId']
#         title = item['snippet']['title']
#         video_url = f'https://www.youtube.com/watch?v={video_id}'
#         #print(title)
#         # Check if title is relevant
#         relevance_check = await check_youtube_relevance(query, title)
        
#         # Proceed if valid
#         if relevance_check == 1 and 60 < get_video_length(video_url) < 3601:
#             #print("valid",title)
#             youtube_search_results.append(video_url)

#     return youtube_search_results

async def get_video_statistics(video_id):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    request = youtube.videos().list(part='statistics,snippet', id=video_id)
    response = await asyncio.to_thread(request.execute)

    if not response['items']:
        return None

    snippet = response['items'][0]['snippet']
    return snippet['publishedAt']

async def get_video_transcript(video_id):
    try:
        transcript_en = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id, languages=['en'])
        text_en = ' '.join([entry['text'] for entry in transcript_en])
        summary = await asyncio.to_thread(summarize_transcript, text_en)
        return summary
    except (NoTranscriptFound, NoTranscriptAvailable, TranscriptsDisabled):
        return None

async def turl(video_url):
    video_id_match = re.search(r"v=([^\&\?\/]+)", video_url)
    video_id = video_id_match.group(1) if video_id_match else ""
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(thumbnail_url) as response:
            if response.status == 200:
                return thumbnail_url
            else:
                return None

async def insert_into_database(db_pool, source_url, image_url, title, description, s_date, youtube_summary):
    if not db_pool: # Check the passed argument
        print("ERROR: Database pool not initialized.")
        return
    try:
        async with db_pool.acquire() as conn: # Use the passed argument
            # ... (rest of the function is the same)
            data = (source_url, image_url, title, description, s_date, youtube_summary)
            check_query = "SELECT 1 FROM source_data WHERE source_url = $1"
            exists = await conn.fetchrow(check_query, source_url)
            if not exists:
                insert_query = "INSERT INTO source_data (source_url, image_url, title, description, source_date, youtube_summary) VALUES ($1, $2, $3, $4, $5, $6)"
                await conn.execute(insert_query, *data)
                print("Data inserted successfully into PostgreSQL")
            else:
                print("Source URL already exists. Skipping insertion.")
    except (Exception, asyncpg.Error) as error:
        print("Error while inserting data into PostgreSQL:", error)
    except (Exception, asyncpg.Error) as error:
        print("Error while inserting data into PostgreSQL:", error)

async def get_summary_from_database(db_pool, source_url):
    if not db_pool: # Check the passed argument
        print("ERROR: Database pool not initialized.")
        return None
    try:
        async with db_pool.acquire() as conn: # Use the passed argument
            select_query = "SELECT youtube_summary FROM source_data WHERE source_url = $1"
            result = await conn.fetchrow(select_query, source_url)
            return result['youtube_summary'] if result else None
    except (Exception, asyncpg.Error) as error:
        print("Error while fetching data from PostgreSQL:", error)
        return None
    except (Exception, asyncpg.Error) as error:
        print("Error while fetching data from PostgreSQL:", error)
        return None

async def extract_youtube_video_data(session, url, db_pool): # Add db_pool here
    start_time = time.time()
    # ... (soup parsing logic is the same) ...
    async with session.get(url) as response:
        content = await response.text()
    soup = BeautifulSoup(content, 'html.parser')
    
    video_id_match = re.search(r"v=([^\&\?\/]+)", url)
    video_id = video_id_match.group(1) if video_id_match else ""
        
    # Pass db_pool to the database function
    summary = await get_summary_from_database(db_pool, url) 
    if not summary:
        transcript = await get_video_transcript(video_id) if video_id else ""
        summary = transcript

    if summary:
        # ... (logic to get title, description, etc. is the same) ...
        try:
            title=get_youtube_title(video_id)
        except Exception as e:
            title=None
            print(f"An error occurred with URL {url}: {str(e)}")

        description_tag = soup.find("meta", {"name": "description"})
        description = description_tag["content"] if description_tag else "No Description Available"
        date = await get_video_statistics(video_id) if video_id else {}
        t_image = await turl(url)

        # Pass db_pool to the database function
        await insert_into_database(db_pool, url, t_image, title, description, date, summary) 

        return title, description, summary, url
    else:
        return None, None, None, None # Return None if no summary

async def get_data(urls, db_pool): # Add db_pool here
    async with aiohttp.ClientSession() as session:
        # Pass db_pool to each task
        tasks = [extract_youtube_video_data(session, url, db_pool) for url in urls] 
        results = await asyncio.gather(*tasks)
        # Filter out None results for videos that had no summary
        return [res for res in results if res[0] is not None]

def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def summarize_transcript_of_each_video(trans):
    prompt = """
    Based on the following video transcript, provide a structured summary focusing on key insights and critical information.
    Organize the summary into a paragraph with a clear, comprehensive flow.
    The paragraph should cover all points in the video transcript and should be in 1000 words.
    {transcript}
    
    Provide a structured summary as described above.
    """

    # llm = ChatOpenAI(model_name='gpt-3.5-turbo-1106', temperature=0.3)
    yt_prompt = PromptTemplate(template=prompt, input_variables=["transcript"])
    chain = LLMChain(prompt=yt_prompt, llm=llm)

    input_data = {
        "transcript": trans
    }
    res = chain.invoke(input_data)
    return res['text']

def summarize_transcript(trans):
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

async def get_yt_data_async(query: str):
    """
    Asynchronously searches for YouTube videos using the Brave Video Search API and filters them
    based on relevance and video length.
    """
    start_time = time.time()
    brave_api_key = os.getenv('BRAVE_API_KEY')
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found in environment variables.")
        return []
        
    brave_video_searcher = BraveVideoSearch(brave_api_key)
    
    print(f"\n--- DEBUG: get_yt_data_async ---")
    print(f"Step 1: Fetching video URLs for query: '{query}'")
    video_urls = await brave_video_searcher.search(query, max_results=10)
    print(f"Step 2: Received {len(video_urls)} URLs to filter: {video_urls}")

    async def filter_video(url):
        print(f"\n  --- Filtering URL: {url} ---")
        loop = asyncio.get_event_loop()
        try:
            # Run synchronous I/O-bound functions in a thread pool executor
            video_length = await loop.run_in_executor(None, get_video_length, url)
            print(f"  Video Length: {video_length} seconds")
            
            # Filter by length first to avoid unnecessary API calls for title
            if not (60 < video_length < 3601):
                print(f"  -> REJECTED: Video length ({video_length}s) is outside the 60-3600s range.")
                return None

            video_id = await loop.run_in_executor(None, extract_video_id, url)
            if not video_id:
                print(f"  -> REJECTED: Could not extract video ID.")
                return None
            print(f"  Video ID: {video_id}")
            
            title = await loop.run_in_executor(None, get_youtube_title, video_id)
            print(f"  Video Title: {title}")
            
            # Asynchronous relevance check using LLM
            relevance_check = await check_youtube_relevance(query, title)
            print(f"  Relevance Check (LLM): {relevance_check}")
            
            if relevance_check == 1:
                print(f"  -> ACCEPTED: Video is relevant and within length limits.")
                return url
            else:
                print(f"  -> REJECTED: LLM determined the video is not relevant.")
                return None
        except Exception as e:
            print(f"  -> ERROR: Failed to process video URL {url}: {e}")
            return None

    # Create and run filtering tasks concurrently
    print(f"Step 3: Starting concurrent filtering of {len(video_urls)} videos...")
    tasks = [filter_video(url) for url in video_urls]
    results = await asyncio.gather(*tasks)
    
    # Collect non-None results
    filtered_urls = [url for url in results if url is not None]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Step 4: Filtering complete in {elapsed_time:.2f} seconds.")
    print(f"Final list of filtered URLs ({len(filtered_urls)}): {filtered_urls}")
    print(f"--- END DEBUG: get_yt_data_async ---\n")
    
    # Return the top 5 most relevant videos
    return filtered_urls[:5]


def generate_final_summary(query, summaries):
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
    llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",streaming=True,stream_usage=True)
    yt_prompt = PromptTemplate(template=prompt, input_variables=["query", "summaries"])
    output_parser = JsonOutputParser()
    ans_chain = yt_prompt | llm | output_parser

    #return ans_chain
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

async def yt_chat(query,session_id):
    try:
        start_time = time.time()
        links = get_yt_data_async(query)
        data = await get_data(links)
        res = generate_final_summary(query, data)
        end_time = time.time()
        execution_time = end_time - start_time
        print(res)
        print(f"Execution Time: {execution_time} seconds")

        return res
    except Exception as e:
        print(f"Error processing async task: {e}")
        return {}


# async def async_process(query,session_id):
#     try:
#         start_time = time.time()
#         links = get_yt_data_async(query)
#         print(links)
#         data = await get_data(links)
#         res = generate_final_summary(query, data)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         #print(res)
#         print(f"Execution Time: {execution_time} seconds")

#         return res
#     except Exception as e:
#         print(f"Error processing async task: {e}")
#         return {}

# if __name__ == "__main__":
#     query = "what has been the impact of hindenburg claims on Sebi chief Buch on tha markets and Adani stock ?"
#     session_id = "1508nnnlll"
    
#     result = asyncio.run(async_process(query, session_id))
# #     #print(result)


# get_length('https://www.youtube.com/watch?v=SIgSpSOrKfg')


# asyncio.run(extract_youtube_video_data('123','https://www.youtube.com/watch?v=Fqg48GAtz9w')

#https://www.youtube.com/watch?v=XPVXdwNtx6A
# print(asyncio.run(get_video_transcript('XPVXdwNtx6A')))