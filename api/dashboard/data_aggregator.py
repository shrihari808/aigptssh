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
from api.dashboard.llm_generator import LLMGenerator, PortfolioLLMGenerator # Import the new class
from api.dashboard.history import save_dashboard_history
import sys

# --- Define Paths ---
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(OUTPUT_DIR, 'outputs')
DATA_JSON_PATH = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
FINAL_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, 'dashboard_output.json')


def get_age_in_seconds(iso_timestamp):
    """Converts an ISO 8601 timestamp string to seconds since the epoch."""
    if not iso_timestamp:
        return float('inf')
    try:
        if iso_timestamp.endswith('Z'):
            dt_obj = datetime.fromisoformat(iso_timestamp[:-1]).replace(tzinfo=timezone.utc)
        elif '+' in iso_timestamp or iso_timestamp.count(':') > 2:
            dt_obj = datetime.fromisoformat(iso_timestamp)
        else:
            dt_obj = datetime.fromisoformat(iso_timestamp).replace(tzinfo=timezone.utc)

        age_seconds = (datetime.now(timezone.utc) - dt_obj).total_seconds()
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

    sorted_articles = sorted(
        news_articles,
        key=lambda x: get_age_in_seconds(x.get('page_age')),
        reverse=False
    )

    selected_articles = []
    for article in sorted_articles[:count]:
        age_seconds = get_age_in_seconds(article.get('page_age'))
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

    return selected_articles

async def aggregate_and_process_stock_data(stock_name: str):
    """
    Fetches, scrapes, and processes data for a single stock.
    """
    print(f"--- Starting Stock Data Aggregation for: {stock_name} ---")

    brave_fetcher = BraveDashboard()
    # The get_portfolio_data method can handle a list with a single stock
    stock_data = brave_fetcher.get_portfolio_data([stock_name])
    news_articles = stock_data.get("latest_news", [])
    scraped_articles = await scrape_urls(news_articles) if news_articles else []

    vector_store = DashboardVectorStore(collection_name="stock_news_content")
    vector_store.add_documents(scraped_articles)
    scoring_service = DashboardScoringService(vector_store=vector_store)

    stock_query = f"What is the latest news, analyst opinions, and performance data for the stock: {stock_name}?"

    context_queries = {
        "key_issues_context": stock_query,
        "indices_context": f"Provide a summary of the latest news and key events for the stock: {stock_name}.",
        "market_drivers_context": f"What were the main reasons and key driving factors for the stock: {stock_name}?"
    }

    llm_contexts = {
        "indices_context": scoring_service.get_enhanced_context(context_queries["indices_context"], k=5),
        "market_drivers_context": scoring_service.get_enhanced_context(context_queries["market_drivers_context"], k=5),
        "key_issues_context": scoring_service.get_enhanced_context(context_queries["key_issues_context"], k=15)
    }

    latest_news_articles = select_latest_news_articles(news_articles, count=3)

    processed_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "llm_contexts": llm_contexts,
        "latest_news_articles": latest_news_articles,
        "portfolio": [stock_name]  # Keep portfolio format for the LLM generator
    }

    return processed_data


async def aggregate_and_process_portfolio_data(portfolio: list[str]):
    """
    Fetches, scrapes, and processes data for a specific portfolio of stocks.
    """
    print(f"--- Starting Portfolio Data Aggregation for: {portfolio} ---")

    # ... (the fetching, scraping, and scoring logic remains the same) ...
    brave_fetcher = BraveDashboard()
    portfolio_data = brave_fetcher.get_portfolio_data(portfolio)
    news_articles = portfolio_data.get("latest_news", [])
    scraped_articles = await scrape_urls(news_articles) if news_articles else []

    vector_store = DashboardVectorStore(collection_name="portfolio_news_content")
    vector_store.add_documents(scraped_articles)
    scoring_service = DashboardScoringService(vector_store=vector_store)

    portfolio_query = f"What is the latest news, analyst opinions, and performance data for the stocks: {', '.join(portfolio)}?"

    context_queries = {
        "key_issues_context": portfolio_query,
        "indices_context": f"Provide a summary of the latest news and key events for the stocks: {', '.join(portfolio)}.",
        "market_drivers_context": f"What were the main reasons and key driving factors for the stocks in this portfolio: {', '.join(portfolio)}?"
    }

    llm_contexts = {
        "indices_context": scoring_service.get_enhanced_context(context_queries["indices_context"], k=5),
        "market_drivers_context": scoring_service.get_enhanced_context(context_queries["market_drivers_context"], k=5),
        "key_issues_context": scoring_service.get_enhanced_context(context_queries["key_issues_context"], k=15)
    }

    latest_news_articles = select_latest_news_articles(news_articles, count=3)

    processed_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "llm_contexts": llm_contexts,
        "latest_news_articles": latest_news_articles,
        "portfolio": portfolio
    }

    return processed_data


async def aggregate_and_process_data(country_code="IN", country_name="India"):
    """
    Fetches, scrapes, processes, and updates the dashboard data for a specific country.
    """
    print(f"--- Starting Full Data Aggregation for {country_name} ---")

    brave_fetcher = BraveDashboard()
    qualitative_data = await brave_fetcher.get_dashboard_data(country_code, country_name)
    news_articles = qualitative_data.get("latest_news", [])
    scraped_articles = await scrape_urls(news_articles) if news_articles else []

    vector_store = DashboardVectorStore(collection_name=f"dashboard_news_{country_code.lower()}")
    vector_store.add_documents(scraped_articles)
    scoring_service = DashboardScoringService(vector_store=vector_store)

    context_queries = {
        "indices_context": f"What was the performance of key {country_name} market indices today?",
        "sectors_context": f"Which sectors were the top performers and underperformers in the {country_name} stock market today?",
        "standouts_context": f"Who were the biggest standout stock gainers and losers in the {country_name} market today?",
        "market_drivers_context": f"What were the main reasons and key driving factors for today's market movement in {country_name}?"
    }

    llm_contexts = {key: scoring_service.get_enhanced_context(query, k=5) for key, query in context_queries.items()}

    latest_news_articles = select_latest_news_articles(news_articles, count=3)

    processed_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "llm_contexts": llm_contexts,
        "latest_news_articles": latest_news_articles
    }
    
    # Save the intermediate data
    data_json_path = os.path.join(OUTPUT_DIR, f'dashboard_data_{country_code}.json')
    save_data_to_json(processed_data, data_json_path)

    llm_generator = LLMGenerator(input_path=data_json_path)
    final_dashboard_content = await llm_generator.generate_dashboard_content()

    # Save the final output
    final_output_path = os.path.join(OUTPUTS_DIR, f'dashboard_output_{country_code}.json')
    save_data_to_json(final_dashboard_content, final_output_path)
    save_dashboard_history(final_dashboard_content)

    print(f"\n--- Pipeline Complete for {country_name}: Final dashboard output generated. ---")
    return final_dashboard_content

# --- START OF MODIFICATION ---
async def generate_trending_stocks_data(country_code: str = "IN"):
    """
    Fetches trending stocks and saves them to a JSON file.
    This function will be called by the scheduler.
    """
    print(f"--- Starting Trending Stocks Generation for country: {country_code} ---")
    brave_fetcher = BraveDashboard()
    
    # Directly await the async function
    trending_stocks = await brave_fetcher.scrape_trending_stocks(country_code)
    
    # Add the last_updated_utc timestamp
    trending_stocks['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
    
    output_path = os.path.join(OUTPUTS_DIR, f'trending_stocks_{country_code}.json')
    save_data_to_json(trending_stocks, output_path)
    print(f"\n--- Pipeline Complete: Trending stocks for {country_code} identified and saved. ---")
# --- END OF MODIFICATION ---

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
    # This function is for the general dashboard and remains unchanged.
    if not existing_data: return new_data
    # ... (rest of the function is unchanged)
    return existing_data

def save_data_to_json(data, output_path):
    try:
        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {output_path}")
    except IOError as e:
        print(f"Error saving data to JSON file: {e}")

# --- THIS IS THE UPDATED TEST BLOCK ---
if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())