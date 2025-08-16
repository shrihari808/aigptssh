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

def get_video_length_pytube(url: str) -> int:
    """
    Get video length using pytube as fallback.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video length in seconds, or 999999 if error
    """
    try:
        yt = YouTube(url)
        return yt.length if yt.length else 999999
    except Exception as e:
        print(f"WARNING: Could not get video length for {url}: {e}")
        return 999999

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
    Enhanced transcript retrieval with multiple language fallbacks and translation.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Summarized transcript or empty string if not available
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        
        # First try to get transcript list to see what's available
        transcript_list = await asyncio.to_thread(ytt_api.list, video_id)
        
        # Try to find the best transcript
        transcript = None
        
        # Priority order: English manual > English auto > Other manual > Other auto > Translated
        try:
            # Try English transcripts first
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except NoTranscriptFound:
            try:
                # Try any manually created transcript
                transcript = transcript_list.find_manually_created_transcript(['de', 'es', 'fr', 'hi', 'ja', 'ko', 'pt', 'ru', 'zh'])
            except NoTranscriptFound:
                try:
                    # Try any generated transcript
                    transcript = transcript_list.find_generated_transcript(['de', 'es', 'fr', 'hi', 'ja', 'ko', 'pt', 'ru', 'zh'])
                except NoTranscriptFound:
                    # As last resort, try to translate the first available transcript
                    for available_transcript in transcript_list:
                        if available_transcript.is_translatable:
                            transcript = available_transcript.translate('en')
                            break
        
        if not transcript:
            print(f"WARNING: No suitable transcript found for video {video_id}")
            return ""
        
        # Fetch the transcript
        fetched_transcript = await asyncio.to_thread(transcript.fetch)
        
        # Convert to text
        transcript_data = fetched_transcript.to_raw_data()
        full_text = ' '.join([entry['text'] for entry in transcript_data])
        
        print(f"DEBUG: Successfully retrieved transcript for {video_id} in {fetched_transcript.language}")
        
        # Summarize the transcript
        summary = await asyncio.to_thread(summarize_transcript, full_text)
        return summary
        
    except (NoTranscriptFound, NoTranscriptAvailable, TranscriptsDisabled) as e:
        print(f"WARNING: No transcript available for video {video_id}: {e}")
        return ""
    except Exception as e:
        print(f"ERROR: Failed to get transcript for video {video_id}: {e}")
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

def summarize_transcript_chunk(transcript: str) -> str:
    """
    Summarize a single transcript chunk.
    
    Args:
        transcript: Transcript text to summarize
        
    Returns:
        Summary of the transcript
    """
    prompt = """
    Based on the following video transcript, provide a structured summary focusing on key insights and critical information.
    Organize the summary into a comprehensive paragraph covering all important points.
    The summary should be detailed and around 800-1000 words.
    
    Transcript:
    {transcript}
    
    Provide a structured summary as described above.
    """

    yt_prompt = PromptTemplate(template=prompt, input_variables=["transcript"])
    chain = LLMChain(prompt=yt_prompt, llm=llm)

    try:
        input_data = {"transcript": transcript}
        res = chain.invoke(input_data)
        return res['text']
    except Exception as e:
        print(f"ERROR: Failed to summarize transcript: {e}")
        return ""

def summarize_transcript(transcript: str) -> str:
    """
    Summarize transcript, splitting into chunks if too long.
    
    Args:
        transcript: Full transcript text
        
    Returns:
        Complete summary
    """
    try:
        token_count = count_tokens(transcript)
        summary = ""

        if token_count < 16000:
            # Small enough to process in one go
            summary = summarize_transcript_chunk(transcript)
        else:
            # Split into manageable chunks
            text_splitter = TokenTextSplitter(chunk_size=16000, chunk_overlap=200)
            chunks = text_splitter.split_text(transcript)
            
            for chunk in chunks:
                chunk_summary = summarize_transcript_chunk(chunk)
                summary += chunk_summary + "\n\n"
        
        return summary.strip()
    except Exception as e:
        print(f"ERROR: Failed to summarize transcript: {e}")
        return ""

async def insert_into_database(db_pool: asyncpg.Pool, source_url: str, image_url: str, 
                              title: str, description: str, source_date: str, 
                              youtube_summary: str) -> None:
    """
    Insert video data into database.
    
    Args:
        db_pool: Database connection pool
        source_url: YouTube video URL
        image_url: Thumbnail URL
        title: Video title
        description: Video description
        source_date: Publication date
        youtube_summary: Summarized transcript
    """
    if not db_pool:
        print("ERROR: Database pool not initialized.")
        return
        
    try:
        async with db_pool.acquire() as conn:
            # Check if URL already exists
            check_query = "SELECT 1 FROM source_data WHERE source_url = $1"
            exists = await conn.fetchrow(check_query, source_url)
            
            if not exists:
                insert_query = """
                    INSERT INTO source_data (source_url, image_url, title, description, source_date, youtube_summary)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """
                await conn.execute(insert_query, source_url, image_url, title, description, source_date, youtube_summary)
                print(f"DEBUG: Inserted video data for: {title}")
            else:
                print(f"DEBUG: Video already exists in database: {source_url}")
                
    except (Exception, asyncpg.Error) as error:
        print(f"ERROR: Failed to insert data into database: {error}")

async def get_summary_from_database(db_pool: asyncpg.Pool, source_url: str) -> str:
    """
    Get existing summary from database.
    
    Args:
        db_pool: Database connection pool
        source_url: YouTube video URL
        
    Returns:
        Existing summary or None if not found
    """
    if not db_pool:
        print("ERROR: Database pool not initialized.")
        return None
        
    try:
        async with db_pool.acquire() as conn:
            select_query = "SELECT youtube_summary FROM source_data WHERE source_url = $1"
            result = await conn.fetchrow(select_query, source_url)
            return result['youtube_summary'] if result else None
            
    except (Exception, asyncpg.Error) as error:
        print(f"ERROR: Failed to fetch data from database: {error}")
        return None

async def process_video_data(video_data: dict, db_pool: asyncpg.Pool) -> tuple:
    """
    Process a single video's data from Brave API results.
    
    Args:
        video_data: Video data from Brave API
        db_pool: Database connection pool
        
    Returns:
        Tuple of (title, description, summary, url) or (None, None, None, None) if failed
    """
    start_time = time.time()
    
    try:
        url = video_data.get('url', '')
        title = video_data.get('title', 'No Title Available')
        description = video_data.get('description', 'No Description Available')
        
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            print(f"WARNING: Could not extract video ID from {url}")
            return None, None, None, None
        
        # Check if we already have a summary in the database
        summary = await get_summary_from_database(db_pool, url)
        
        if not summary:
            # Get transcript and summarize using enhanced method
            summary = await get_video_transcript_with_fallback(video_id)
            if not summary:
                print(f"WARNING: Could not get transcript for video {video_id}")
                return None, None, None, None
        
        # Get additional metadata
        source_date = video_data.get('page_age', datetime.now().isoformat())
        thumbnail_url = video_data.get('thumbnail', {}).get('src')
        
        # Insert into database
        await insert_into_database(db_pool, url, thumbnail_url, title, description, source_date, summary)
        
        end_time = time.time()
        print(f"DEBUG: Processed video in {end_time - start_time:.2f}s: {title}")
        
        return title, description, summary, url
        
    except Exception as e:
        print(f"ERROR: Failed to process video data: {e}")
        return None, None, None, None

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
    
    async def filter_video(video_data: dict) -> str:
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
            # Fallback to pytube if duration parsing failed
            duration_seconds = get_video_length_pytube(url)
        
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
                return url
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
    filtered_urls = []
    for result in results:
        if isinstance(result, str) and result:  # Valid URL
            filtered_urls.append(result)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Step 4: Filtering complete in {elapsed_time:.2f} seconds")
    print(f"Final filtered URLs ({len(filtered_urls)}): {filtered_urls}")
    print(f"--- END DEBUG: get_yt_data_async ---\n")
    
    # Return top 5 most relevant videos
    return filtered_urls[:5]

async def get_data(urls: list[str], db_pool: asyncpg.Pool) -> list[tuple]:
    """
    Process multiple YouTube URLs and extract their data.
    
    Args:
        urls: List of YouTube URLs
        db_pool: Database connection pool
        
    Returns:
        List of tuples containing (title, description, summary, url) for each video
    """
    if not urls:
        return []
    
    print(f"DEBUG: Processing {len(urls)} YouTube videos...")
    
    # For Brave API, we need to get the video data first
    # Since we only have URLs, we'll need to make individual calls or use a different approach
    brave_api_key = os.getenv('BRAVE_API_KEY')
    brave_video_searcher = BraveVideoSearch(brave_api_key)
    
    results = []
    for url in urls:
        try:
            # Extract video ID to get basic info
            video_id = extract_video_id(url)
            if not video_id:
                continue
            
            # Create basic video data structure
            video_data = {
                'url': url,
                'title': f"Video {video_id}",  # Will be updated if we can get better title
                'description': 'No description available',
                'page_age': datetime.now().isoformat()
            }
            
            # Process the video
            result = await process_video_data(video_data, db_pool)
            if result[0] is not None:  # If processing was successful
                results.append(result)
                
        except Exception as e:
            print(f"ERROR: Failed to process URL {url}: {e}")
            continue
    
    print(f"DEBUG: Successfully processed {len(results)} videos")
    return results