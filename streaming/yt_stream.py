import asyncio
import aiohttp
import time
import asyncpg
from langchain_community.callbacks import get_openai_callback
from langchain_text_splitters import TokenTextSplitter
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup
import re
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
import tiktoken
from typing import AsyncGenerator
from api.brave_searcher import BraveVideoSearch
from dotenv import load_dotenv
from config import GPT4o_mini
import xml.etree.ElementTree as ET

load_dotenv(override=True)

# --- NEW: YouTube Data API Initialization ---
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in environment variables.")

try:
    youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
except Exception as e:
    print(f"CRITICAL ERROR: Could not build YouTube service: {e}")
    youtube_service = None

# Initialize LLM
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
psql_url = os.getenv('DATABASE_URL')

# LLM chain for relevance checking
yt_prompt_template = """
Your task is to determine which of the following YouTube video titles are relevant to the user's question.

User question: {q}

YouTube titles:
{yt_titles}

Output the result in JSON format. The JSON should be a list of objects, where each object contains the original title and a "valid" key with a value of 1 if the video is relevant, or 0 if it is not.

Example:
[
    {{"title": "title 1", "valid": 1}},
    {{"title": "title 2", "valid": 0}},
    {{"title": "title 3", "valid": 1}}
]
"""

yt_prompt = PromptTemplate(template=yt_prompt_template, input_variables=["q", "yt_titles"])
yt_chain = yt_prompt | llm | JsonOutputParser()

def yt_chat(query: str, session_id: str) -> dict:
    pass

def extract_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL.
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def parse_duration_to_seconds(duration_str: str) -> int:
    """
    Parse duration string (e.g., "02:30", "1:15:30") to seconds.
    """
    try:
        parts = duration_str.split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return 0
    except (ValueError, AttributeError):
        return 0

def parse_xml_transcript(xml_content: str) -> str:
    """
    Parses XML formatted transcript to extract only the text content.
    """
    try:
        root = ET.fromstring(xml_content)
        text_lines = [elem.text for elem in root.findall('text')]
        return ' '.join(line.strip() for line in text_lines if line)
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML transcript: {e}")
        return ""


async def check_english_caption_exists(video_id: str) -> bool:
    """
    Quickly checks if an English caption track exists for a video.
    """
    if not youtube_service:
        return False
    try:
        def fetch_captions():
            return youtube_service.captions().list(
                part='snippet',
                videoId=video_id
            ).execute()
        
        captions_list_response = await asyncio.to_thread(fetch_captions)
        
        for item in captions_list_response.get('items', []):
            if item['snippet']['language'].lower().startswith('en'):
                return True
        return False
    except Exception:
        return False

async def get_transcript_from_youtube_api(video_id: str) -> str:
    """
    Fetches video transcript using a hybrid approach to avoid OAuth2 requirement.
    Uses the official API to list captions and direct download to fetch them.
    """
    if not youtube_service:
        print("ERROR: YouTube service is not available.")
        return ""

    try:
        # Step 1: Use official API to list available caption tracks
        def list_captions():
            return youtube_service.captions().list(part='snippet', videoId=video_id).execute()

        caption_list_response = await asyncio.to_thread(list_captions)
        
        items = caption_list_response.get('items', [])
        if not items:
            print(f"WARNING: No caption tracks found for video {video_id}.")
            return ""

        # Step 2: Find the URL for the first available English transcript
        caption_url = None
        for item in items:
            if item['snippet']['language'].lower().startswith('en'):
                # Construct the direct download URL
                caption_id = item['id']
                caption_url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&name=&kind=asr&fmt=xml&id={caption_id}"
                print(f"DEBUG: Found English caption track for video {video_id}.")
                break
        
        if not caption_url:
            print(f"WARNING: No suitable English transcript found for video {video_id}.")
            return ""

        # Step 3: Download the transcript directly using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(caption_url) as response:
                if response.status == 200:
                    xml_transcript = await response.text()
                    transcript_text = parse_xml_transcript(xml_transcript)
                    print(f"DEBUG: Successfully downloaded and parsed transcript for {video_id}.")
                    return transcript_text
                else:
                    print(f"ERROR: Failed to download transcript for {video_id}. Status: {response.status}")
                    return ""

    except HttpError as e:
        print(f"ERROR: YouTube API error for video {video_id}: {e}")
        return ""
    except Exception as e:
        print(f"ERROR: An unexpected error occurred for video {video_id}: {e}")
        return ""


def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"ERROR: Could not count tokens: {e}")
        return 0

async def process_video_data(video_data: dict) -> dict:
    """
    Processes a single video's data to fetch its full transcript and metadata.
    """
    start_time = time.time()

    try:
        url = video_data.get('url', '')
        title = video_data.get('title', 'No Title Available')

        video_id = extract_video_id(url)
        if not video_id:
            print(f"WARNING: Could not extract video ID from {url}")
            return None

        # Get the full transcript text using the new function
        transcript_text = await get_transcript_from_youtube_api(video_id)
        if not transcript_text:
            # This is now expected for videos without English transcripts, so it's a normal skip.
            print(f"INFO: Skipping video {video_id} as no usable transcript was found.")
            return None

        # Prepare metadata
        metadata = {
            'url': url,
            'title': title,
            'description': video_data.get('description', 'No Description Available'),
            'publication_date': video_data.get('page_age', datetime.now().isoformat()),
            'thumbnail_url': video_data.get('thumbnail', {}).get('src')
        }

        end_time = time.time()
        print(f"DEBUG: Fetched transcript in {end_time - start_time:.2f}s: {title}")

        return {"transcript": transcript_text, "metadata": metadata}

    except Exception as e:
        print(f"ERROR: Failed to process video data for {url}: {e}")
        return None

async def get_yt_data_async(query: str) -> list[dict]:
    """
    Search for YouTube videos, pre-filters for English captions, and then filters by relevance.
    """
    start_time = time.time()
    brave_api_key = os.getenv('BRAVE_API_KEY')
    
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found.")
        return []
    
    brave_video_searcher = BraveVideoSearch(brave_api_key)
    
    print(f"\n--- DEBUG: get_yt_data_async ---")
    print(f"Step 1: Searching Brave Videos for query: '{query}'")
    
    video_results = await brave_video_searcher.search_detailed(query, max_results=20)
    print(f"Step 2: Received {len(video_results)} potential videos.")

    # Filter by duration first
    valid_duration_videos = [
        v for v in video_results 
        if 60 < parse_duration_to_seconds(v.get('video', {}).get('duration', '0:00')) < 3601
    ]
    print(f"Step 3: Found {len(valid_duration_videos)} videos within 1-60 minute duration.")
    
    if not valid_duration_videos:
        return []

    # Pre-filter for English captions
    print(f"Step 4: Pre-filtering for English caption availability...")
    tasks = [check_english_caption_exists(extract_video_id(v['url'])) for v in valid_duration_videos]
    caption_results = await asyncio.gather(*tasks)
    
    videos_with_captions = [
        video for video, has_caption in zip(valid_duration_videos, caption_results) if has_caption
    ]
    print(f"Step 5: Found {len(videos_with_captions)} videos with available English captions.")

    if not videos_with_captions:
        return []

    # Batch LLM relevance check on the smaller, caption-verified list
    titles_to_check = [v['title'] for v in videos_with_captions]
    titles_str = "\n".join(f"- {title}" for title in titles_to_check)
    input_data = {"q": query, "yt_titles": titles_str}
    
    try:
        relevance_results = await yt_chain.ainvoke(input_data)
        relevance_map = {item['title']: item['valid'] for item in relevance_results}
        
        filtered_videos = [
            video for video in videos_with_captions if relevance_map.get(video['title'], 0) == 1
        ]
                
    except Exception as e:
        print(f"ERROR: Batch LLM relevance check failed: {e}")
        return []

    end_time = time.time()
    print(f"Step 6: Filtering complete in {end_time - start_time:.2f} seconds.")
    print(f"Final filtered videos ({len(filtered_videos)}): {[v['title'] for v in filtered_videos]}")
    print(f"--- END DEBUG: get_yt_data_async ---\n")
    
    return filtered_videos[:10]

async def get_data(videos: list[dict], db_pool: asyncpg.Pool) -> AsyncGenerator:
    """
    Yields transcripts as they become available instead of waiting for all of them.
    """
    if not videos:
        return

    print(f"DEBUG: Concurrently processing {len(videos)} YouTube videos for transcripts...")

    async def fetch_with_timeout(video):
        try:
            # Set a 10-second timeout for processing each video
            return await asyncio.wait_for(
                process_video_data(video), 
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print(f"Timeout fetching transcript for {video.get('title')}")
            return None

    tasks = [asyncio.create_task(fetch_with_timeout(v)) for v in videos]
    
    # Yield results as they are completed
    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            yield result
