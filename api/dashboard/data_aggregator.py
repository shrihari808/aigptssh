# aigptssh/api/dashboard/data_aggregator.py
import json
import os
import asyncio
from datetime import datetime, timezone
import re
from api.dashboard.brave_search import BraveDashboard
from api.dashboard.web_scraper import scrape_urls
from api.dashboard.vector_store import DashboardVectorStore
from api.dashboard.scoring_service import DashboardScoringService
from api.dashboard.llm_generator import LLMGenerator
from api.dashboard.history import save_dashboard_history
import sys

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'dashboard_output.json')

def get_age_in_seconds(iso_timestamp):
    """Converts an ISO 8601 timestamp string to seconds since the epoch."""
    if not iso_timestamp:
        return float('inf')
    try:
        # Handle different timestamp formats from Brave API
        if iso_timestamp.endswith('Z'):
            # Format: "2025-08-20T10:23:38Z"
            dt_obj = datetime.fromisoformat(iso_timestamp[:-1]).replace(tzinfo=timezone.utc)
        elif '+' in iso_timestamp or iso_timestamp.count(':') > 2:
            # Format with timezone: "2025-08-20T10:23:38+00:00"
            dt_obj = datetime.fromisoformat(iso_timestamp)
        else:
            # Format without timezone: "2025-08-20T10:23:38"
            dt_obj = datetime.fromisoformat(iso_timestamp).replace(tzinfo=timezone.utc)
        
        age_seconds = (datetime.now(timezone.utc) - dt_obj).total_seconds()
        print(f"Debug: Timestamp '{iso_timestamp}' -> {age_seconds} seconds ago")
        return age_seconds
    except (ValueError, TypeError) as e:
        print(f"Error parsing timestamp '{iso_timestamp}': {e}")
        return float('inf')

def get_human_readable_age(seconds):
    """Converts seconds to human readable age format."""
    if seconds == float('inf'):
        return "Unknown"
    
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if days > 0:
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

def select_latest_news_articles(news_articles, count=3):
    """
    Selects the newest articles based on page_age and formats them for dashboard output.
    """
    if not news_articles:
        return []
    
    print(f"Debug: Processing {len(news_articles)} articles for selection...")
    for i, article in enumerate(news_articles[:5]):  # Debug first 5 articles
        print(f"  Article {i+1}: '{article.get('title', 'No Title')[:60]}...'")
        print(f"    page_age: {article.get('page_age')}")
        print(f"    description: {article.get('description')}")
        print(f"    url: {article.get('url')}")
        print()
    
    # Sort articles by age (newest first)
    sorted_articles = sorted(
        news_articles, 
        key=lambda x: get_age_in_seconds(x.get('page_age')), 
        reverse=False  # False means newest first (smallest age value)
    )
    
    # Get top N articles and format them
    selected_articles = []
    for article in sorted_articles[:count]:
        age_seconds = get_age_in_seconds(article.get('page_age'))
        
        # Use description directly from Brave API
        description = article.get("description", "")
        if not description or description.strip() == "":
            description = "No description available"
        
        formatted_article = {
            "title": article.get("title", "No Title"),
            "snippet": description,
            "url": article.get("url", ""),
            "age": get_human_readable_age(age_seconds)
        }
        
        selected_articles.append(formatted_article)
        print(f"Selected: {formatted_article['title']} (Age: {formatted_article['age']})")
    
    return selected_articles

async def aggregate_and_process_data():
    """
    Fetches, scrapes, processes, and updates the dashboard data.
    """
    print("--- Starting Full Data Aggregation and Processing Pipeline ---")
    
    # --- Step 1: Fetch and Scrape ---
    brave_fetcher = BraveDashboard()
    qualitative_data = brave_fetcher.get_dashboard_data()
    news_articles = qualitative_data.get("latest_news", [])
    scraped_articles = await scrape_urls(news_articles) if news_articles else []

    # --- Step 2: Vector Store and Scoring ---
    vector_store = DashboardVectorStore()
    vector_store.add_documents(scraped_articles)
    scoring_service = DashboardScoringService(vector_store=vector_store)
    
    context_queries = {
        "indices_context": "What was the performance of key Indian market indices like the NIFTY 50 and Sensex today?",
        "sectors_context": "Which sectors were the top performers and underperformers in the Indian stock market today?",
        "standouts_context": "Who were the biggest standout stock gainers and losers in the Indian market today?",
        "market_drivers_context": "What were the main reasons and key driving factors for today's market movement?"
    }
    
    # --- Generate contexts using vector search for analytical sections ---
    llm_contexts = {key: scoring_service.get_enhanced_context(query, k=5) for key, query in context_queries.items()}
    
    # --- NEW: Direct latest news selection ---
    print("--- Selecting the 3 newest articles directly ---")
    latest_news_articles = select_latest_news_articles(news_articles, count=3)
    print(f"Successfully selected {len(latest_news_articles)} newest articles for the headlines section.")
    for i, article in enumerate(latest_news_articles, 1):
        print(f"  {i}. {article['title']} ({article['age']})")

    processed_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "llm_contexts": llm_contexts,
        "latest_news_articles": latest_news_articles  # Add this to pass to LLM generator
    }
    
    save_data_to_json(processed_data, DATA_JSON_PATH)

    # --- Step 3: LLM Generation ---
    llm_generator = LLMGenerator(input_path=DATA_JSON_PATH)
    new_dashboard_content = llm_generator.generate_dashboard_content()

    # --- Step 4 & 5 remain the same ---
    existing_dashboard_content = {}
    if os.path.exists(FINAL_OUTPUT_PATH):
        try:
            with open(FINAL_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                existing_dashboard_content = json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not decode existing dashboard data. Starting fresh.")
            existing_dashboard_content = {}
            
    final_dashboard_content = update_dashboard_data(existing_dashboard_content, new_dashboard_content)
    save_data_to_json(final_dashboard_content, FINAL_OUTPUT_PATH)
    save_dashboard_history(final_dashboard_content)
    
    print("\n--- Pipeline Complete: Final dashboard output generated and saved. ---")
    return final_dashboard_content

def get_human_readable_age_in_seconds(age_str):
    if not age_str or not isinstance(age_str, str):
        return float('inf')
    age_str = age_str.lower()
    value = int(re.findall(r'\d+', age_str)[0]) if re.findall(r'\d+', age_str) else 0
    if 'hour' in age_str: return value * 3600
    if 'day' in age_str: return value * 86400
    if 'minute' in age_str: return value * 60
    return float('inf')

def update_dashboard_data(existing_data, new_data):
    if not existing_data: return new_data
    if 'market_summary' in new_data and 'summary_points' in new_data['market_summary']:
        existing_summary = existing_data.get('market_summary', {}).get('summary_points', [])
        new_summary = new_data['market_summary']['summary_points']
        all_summary_points = existing_summary + new_summary
        unique_points = {point['title']: point for point in all_summary_points}.values()
        sorted_points = sorted(unique_points, key=lambda x: get_human_readable_age_in_seconds(x.get('age', '')))
        existing_data['market_summary']['summary_points'] = sorted_points[:len(existing_summary) or 6]
        existing_data['market_summary']['sources'] = list(set(existing_data.get('market_summary', {}).get('sources', []) + new_data.get('market_summary', {}).get('sources', [])))
    if 'latest_news' in new_data and 'articles' in new_data['latest_news']:
        existing_articles = existing_data.get('latest_news', {}).get('articles', [])
        new_articles = new_data['latest_news']['articles']
        all_articles = existing_articles + new_articles
        unique_articles = {article['url']: article for article in all_articles}.values()
        sorted_articles = sorted(unique_articles, key=lambda x: get_human_readable_age_in_seconds(x.get('age', '')))
        existing_data['latest_news']['articles'] = sorted_articles[:len(existing_articles) or 3]
    for key in ['sector_analysis', 'standouts_analysis', 'market_drivers']:
        if key in new_data and new_data[key]:
            existing_data[key] = new_data[key]
    existing_data['last_updated_utc'] = new_data['last_updated_utc']
    return existing_data

def save_data_to_json(data, output_path):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {output_path}")
    except IOError as e:
        print(f"Error saving data to JSON file: {e}")

if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(aggregate_and_process_data())