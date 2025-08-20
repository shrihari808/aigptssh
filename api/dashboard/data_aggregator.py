import json
import os
import asyncio
from datetime import datetime, timezone
import sys

# Adjusting the import paths to work from the root of the project
from api.dashboard.brave_search import BraveDashboard
from api.dashboard.web_scraper import scrape_urls
from api.dashboard.vector_store import DashboardVectorStore
from api.dashboard.scoring_service import DashboardScoringService
# Import the LLMGenerator to run it as the final step
from api.dashboard.llm_generator import LLMGenerator 

# Define the output paths for both intermediate and final JSON files
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'dashboard_output.json')

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
    
    # --- Step 2: Scrape the content from the URLs concurrently ---
    if news_articles:
        scraped_articles = await scrape_urls(news_articles)
    else:
        scraped_articles = []
    
    # --- Step 3: Initialize the vector store and add the scraped documents ---
    vector_store = DashboardVectorStore()
    vector_store.add_documents(scraped_articles)
    
    # --- Step 4: Use the scoring service to get multiple, enhanced contexts ---
    scoring_service = DashboardScoringService(vector_store=vector_store)
    
    context_queries = {
        "indices_context": "What was the performance of key Indian market indices like the NIFTY 50 and Sensex today?",
        "sectors_context": "Which sectors were the top performers and underperformers in the Indian stock market today?",
        "standouts_context": "Who were the biggest standout stock gainers and losers in the Indian market today?",
        "market_drivers_context": "What were the main reasons and key driving factors for today's market movement?"
    }
    
    llm_contexts = {}
    for key, query in context_queries.items():
        print(f"--- Generating context for: {key} ---")
        context = scoring_service.get_enhanced_context(query, k=5)
        llm_contexts[key] = context
    
    # --- Step 5: Combine the processed data for the final output ---
    processed_data = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "llm_contexts": llm_contexts,
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

# This main block now orchestrates the entire pipeline
if __name__ == '__main__':
    # Fix for RuntimeError on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # --- Part 1: Run the data aggregation and processing ---
    processed_data = asyncio.run(aggregate_and_process_data())
    save_data_to_json(processed_data, DATA_JSON_PATH)
    
    print("\n--- Starting LLM Generation Stage ---")
    
    # --- Part 2: Run the LLM generator using the created data file ---
    llm_generator = LLMGenerator(input_path=DATA_JSON_PATH)
    final_dashboard_content = llm_generator.generate_dashboard_content()
    save_data_to_json(final_dashboard_content, FINAL_OUTPUT_PATH)
    
    print("\n--- Pipeline Complete: Final dashboard output generated. ---")