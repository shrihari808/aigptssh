import asyncio
import aiohttp
import time
import asyncpg
from langchain_community.callbacks import get_openai_callback
from langchain_text_splitters import TokenTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from bs4 import BeautifulSoup
import re
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
import tiktoken
from pytube import YouTube
from api.brave_searcher import BraveVideoSearch
from dotenv import load_dotenv
from config import GPT4o_mini

load_dotenv(override=True)

# Initialize LLM
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
psql_url = os.getenv('DATABASE_URL')

# LLM chain for relevance checking
yt_prompt_template = """
Your task is to determine if a YouTube video title is relevant to a user question.

User question: {q}
YouTube title: {yt}

Output the result in JSON format:
{{"valid": 1 if the video is relevant to the question, otherwise 0}}
"""

yt_prompt = PromptTemplate(template=yt_prompt_template, input_variables=["q", "yt"])
yt_chain = yt_prompt | llm | JsonOutputParser()

def yt_chat(query: str, session_id: str) -> dict:
    pass

async def check_youtube_relevance(query: str, youtube_title: str) -> int:
    """
    Asynchronously checks if a YouTube title is relevant to the query using LLM.
    
    Args:
        query: User's search query
        youtube_title: Title of the YouTube video
        
    Returns:
        1 if relevant, 0 if not relevant
    """
    if not youtube_title or youtube_title == "Video not found":
        return 0
        
    input_data = {"q": query, "yt": youtube_title}
    try:
        res = await yt_chain.ainvoke(input_data)
        return res.get('valid', 0)
    except Exception as e:
        print(f"ERROR: LLM relevance check failed: {e}")
        return 0

def extract_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID string or None if not found
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
    
    Args:
        duration_str: Duration in format "MM:SS" or "HH:MM:SS"
        
    Returns:
        Duration in seconds
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

async def get_available_transcripts(video_id: str) -> list:
    """
    Get list of available transcripts for a video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        List of available transcript information
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = await asyncio.to_thread(ytt_api.list, video_id)
        
        available_transcripts = []
        for transcript in transcript_list:
            available_transcripts.append({
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            })
        
        return available_transcripts
        
    except Exception as e:
        print(f"ERROR: Failed to list transcripts for video {video_id}: {e}")
        return []

async def get_video_transcript_with_fallback(video_id: str) -> str:
    """
    Corrected transcript retrieval that properly instantiates the API client
    and handles the fetched transcript object correctly.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        The full transcript text in English or an empty string if not available.
    """
    try:
        # 1. Correctly create an instance of the API client
        ytt_api = YouTubeTranscriptApi()
        
        # 2. Call the .list() method on the instance
        transcript_list = await asyncio.to_thread(ytt_api.list, video_id)
        
        transcript = None

        # Priority 1: Find a direct English transcript
        try:
            transcript = await asyncio.to_thread(transcript_list.find_transcript, ['en', 'en-US', 'en-GB'])
            print(f"DEBUG: Found direct English transcript for video {video_id}.")
        except NoTranscriptFound:
            print(f"DEBUG: No direct English transcript found for {video_id}. Checking for translatable options.")
            # Priority 2: Find any other translatable transcript
            for available_transcript in transcript_list:
                if available_transcript.is_translatable:
                    print(f"DEBUG: Found translatable '{available_transcript.language_code}' transcript. Translating to English.")
                    transcript = await asyncio.to_thread(available_transcript.translate, 'en')
                    break

        if not transcript:
            print(f"WARNING: No suitable English or translatable transcript found for video {video_id}")
            return ""

        # Fetch the transcript object
        fetched_transcript_object = await asyncio.to_thread(transcript.fetch)
        
        # 3. Convert the FetchedTranscript object to a raw list of dictionaries
        fetched_transcript_snippets = fetched_transcript_object.to_raw_data()
        
        if isinstance(fetched_transcript_snippets, list):
            full_text = ' '.join([entry['text'] for entry in fetched_transcript_snippets])
            print(f"DEBUG: Successfully retrieved and processed transcript for {video_id}. Final language: English")
            return full_text
        else:
            print(f"WARNING: Transcript for video {video_id} was not in the expected list format after fetching.")
            return ""

    except NoTranscriptFound:
        print(f"WARNING: No transcripts of any kind available for video {video_id}.")
        return ""
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while fetching transcript for video {video_id}: {e}")
        return ""

def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for encoding
        
    Returns:
        Number of tokens
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
    
    Args:
        video_data: Video data from Brave API
        
    Returns:
        A dictionary containing the transcript and metadata, or None if failed.
    """
    start_time = time.time()
    
    try:
        url = video_data.get('url', '')
        title = video_data.get('title', 'No Title Available')
        
        video_id = extract_video_id(url)
        if not video_id:
            print(f"WARNING: Could not extract video ID from {url}")
            return None
        
        # Get the full transcript text
        transcript_text = await get_video_transcript_with_fallback(video_id)
        if not transcript_text:
            print(f"WARNING: Could not get transcript for video {video_id}")
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

async def get_yt_data_async(query: str) -> list[str]:
    """
    Search for YouTube videos using Brave Video Search API and filter them.
    
    Args:
        query: Search query
        
    Returns:
        List of filtered YouTube URLs
    """
    start_time = time.time()
    brave_api_key = os.getenv('BRAVE_API_KEY')
    
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY not found in environment variables.")
        return []
    
    brave_video_searcher = BraveVideoSearch(brave_api_key)
    
    print(f"\n--- DEBUG: get_yt_data_async ---")
    print(f"Step 1: Searching Brave Videos for query: '{query}'")
    
    # Get video results from Brave API
    video_results = await brave_video_searcher.search_detailed(query, max_results=15)
    print(f"Step 2: Received {len(video_results)} video results to filter")
    
    async def filter_video(video_data: dict) -> dict: # <-- Returns dict now
        """Filter individual video based on length and relevance."""
        url = video_data.get('url', '')
        title = video_data.get('title', '')
        duration = video_data.get('video', {}).get('duration', '0:00')
        
        print(f"\n  --- Filtering Video: {title} ---")
        print(f"  URL: {url}")
        print(f"  Duration: {duration}")
        
        # Parse duration and check length
        duration_seconds = parse_duration_to_seconds(duration)
        if duration_seconds == 0:
            print(f"  -> REJECTED: Could not determine video duration from Brave API.")
            return None

        print(f"  Duration in seconds: {duration_seconds}")
        
        # Filter by length (60 seconds to 1 hour)
        if not (60 < duration_seconds < 3601):
            print(f"  -> REJECTED: Duration ({duration_seconds}s) outside 60-3600s range")
            return None
        
        # Check relevance using LLM
        try:
            relevance_check = await check_youtube_relevance(query, title)
            print(f"  Relevance Check (LLM): {relevance_check}")
            
            if relevance_check == 1:
                print(f"  -> ACCEPTED: Video is relevant and within length limits")
                return video_data # <-- Returns the whole dictionary
            else:
                print(f"  -> REJECTED: LLM determined video is not relevant")
                return None
                
        except Exception as e:
            print(f"  -> ERROR: Relevance check failed: {e}")
            return None
    
    # Filter videos concurrently
    print(f"Step 3: Starting concurrent filtering of {len(video_results)} videos...")
    tasks = [filter_video(video_data) for video_data in video_results]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect successful results
    filtered_videos = []
    for result in results:
        if isinstance(result, dict) and result:  # Valid video dictionary
            filtered_videos.append(result)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Step 4: Filtering complete in {elapsed_time:.2f} seconds")
    print(f"Final filtered videos ({len(filtered_videos)}): {[v['title'] for v in filtered_videos]}")
    print(f"--- END DEBUG: get_yt_data_async ---\n")
    
    # Return top 5 most relevant videos
    return filtered_videos[:5]

async def get_data(videos: list[dict], db_pool: asyncpg.Pool) -> list[dict]:
    """
    Process multiple YouTube videos to extract their transcripts and metadata.
    This function now receives full video data, so it doesn't need to fetch it again.
    
    Args:
        videos: List of video data dictionaries from Brave Search.
        db_pool: Database connection pool (kept for signature compatibility but not used here).
        
    Returns:
        List of dictionaries containing transcript and metadata for each video.
    """
    if not videos:
        return []
    
    print(f"DEBUG: Processing {len(videos)} YouTube videos for transcripts...")
    
    tasks = [process_video_data(video) for video in videos]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results from failed processing
    processed_videos = [res for res in results if res]
    
    print(f"DEBUG: Successfully fetched transcripts for {len(processed_videos)} videos")
    return processed_videos
