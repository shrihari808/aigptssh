import json
import os
import asyncio
from datetime import datetime, timezone

# Adjusting the import paths to work from the root of the project
from api.dashboard.brave_search import BraveDashboard
from api.dashboard.web_scraper import scrape_urls
from api.dashboard.vector_store import DashboardVectorStore
from api.dashboard.scoring_service import DashboardScoringService

# Define the output path for the JSON file within the dashboard directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'dashboard_data.json')

async def aggregate_and_process_data():
    """
    Fetches data, scrapes content, stores it in a vector database, and then
    uses a scoring service to retrieve and re-rank multiple, targeted contexts for an LLM.
    """
    print("--- Starting Full Data Aggregation and Processing Pipeline ---")
    
    # --- Step 1: Fetch initial data from Brave Search ---
    brave_fetcher = BraveDashboard()
    qualitative_data = brave_fetcher.get_dashboard_data()
    
    news_articles = qualitative_data.get("latest_news", [])
    urls_to_scrape = [article['url'] for article in news_articles if article.get('url')]
    
    # --- Step 2: Scrape the content from the URLs concurrently ---
    if urls_to_scrape:
        scraped_content = await scrape_urls(urls_to_scrape)
    else:
        scraped_content = []
    
    # --- Step 3: Initialize the vector store and add the scraped documents ---
    vector_store = DashboardVectorStore()
    vector_store.add_documents(scraped_content)
    
    # --- Step 4: Use the scoring service to get multiple, enhanced contexts ---
    scoring_service = DashboardScoringService(vector_store=vector_store)
    
    # Define a dictionary of targeted queries for different sections of the dashboard
    context_queries = {
        "indices_context": "What was the performance of key Indian market indices like the NIFTY 50 and Sensex today?",
        "sectors_context": "Which sectors were the top performers and underperformers in the Indian stock market today?",
        "standouts_context": "Who were the biggest standout stock gainers and losers in the Indian market today?",
        "market_drivers_context": "What were the main reasons and key driving factors for today's market movement?"
    }
    
    llm_contexts = {}
    for key, query in context_queries.items():
        print(f"--- Generating context for: {key} ---")
        # Get the final, re-ranked context for each specific query
        context = scoring_service.get_enhanced_context(query, k=5) # Using k=5 for more focused context
        llm_contexts[key] = context
    
    # --- Step 5: Combine the processed data for the final output ---
    processed_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "llm_contexts": llm_contexts, # Changed to store multiple contexts
        "market_summary_points": qualitative_data.get("market_summary", []),
        "market_standouts": qualitative_data.get("standouts", {})
    }
    
    print("--- Data Aggregation and Processing Complete ---")
    return processed_data

def save_data_to_json(data, output_path):
    """
    Saves the provided data dictionary to a JSON file at the specified path.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {output_path}")
    except IOError as e:
        print(f"Error saving data to JSON file: {e}")

# This main block allows us to run this script directly for testing
if __name__ == '__main__':
    # Run the async main function
    final_data = asyncio.run(aggregate_and_process_data())
    save_data_to_json(final_data, JSON_OUTPUT_PATH)
    
    print("\n--- Final Processed Data Snippet ---")
    print("Generated contexts for:", list(final_data["llm_contexts"].keys()))
    if final_data["llm_contexts"].get("indices_context"):
        print("First context chunk for indices:", final_data["llm_contexts"]["indices_context"][0][:150] + "...")
