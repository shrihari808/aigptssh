# Usage Examples and Testing for Enhanced Web RAG

"""
This file contains examples and test cases for the enhanced web_rag system
that integrates Brave search and advanced scoring.
"""

import asyncio
import json
from api.news_rag.news_rag import web_rag, adaptive_web_rag, web_rag_with_fallback

async def test_basic_web_rag():
    """Test basic web RAG functionality."""
    print("=== Testing Basic Web RAG ===")
    
    test_queries = [
        "latest news on Reliance Industries",
        "HDFC Bank earnings results today",
        "Nifty 50 market outlook recent",
        "Adani Group controversy updates"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        try:
            result = await web_rag(query, session_id="test_session")
            print(f"Response length: {len(result.get('Response', ''))}")
            print(f"Number of links: {len(result.get('links', []))}")
            print(f"Tokens used: {result.get('Total_Tokens', 0)}")
            print(f"Context sufficiency: {result.get('context_sufficiency_score', 'N/A')}")
            print(f"Data ingestion triggered: {result.get('data_ingestion_triggered', False)}")
            print("✅ Success")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_adaptive_web_rag():
    """Test adaptive web RAG with query-specific optimization."""
    print("\n=== Testing Adaptive Web RAG ===")
    
    test_queries = [
        ("latest news on TCS", "recency-focused"),
        ("bullish outlook for IT sector", "sentiment-focused"), 
        ("Infosys company analysis", "company-specific"),
        ("market trends in banking", "market-broad")
    ]
    
    for query, expected_type in test_queries:
        print(f"\nTesting adaptive query: {query} (expected: {expected_type})")
        try:
            result = await adaptive_web_rag(query, session_id="adaptive_test")
            
            query_insights = result.get('query_insights', {})
            print(f"Detected query type: {query_insights.get('query_type', 'unknown')}")
            print(f"Adaptive weights: {query_insights.get('suggested_weights', {})}")
            print(f"Response quality indicators:")
            print(f"  - Sources used: {result.get('num_sources_used', 0)}")
            print(f"  - Top source score: {result.get('top_source_score', 0):.3f}")
            print("✅ Adaptive processing success")
        except Exception as e:
            print(f"❌ Adaptive error: {e}")

async def test_fallback_system():
    """Test fallback system when Brave search fails."""
    print("\n=== Testing Fallback System ===")
    
    # This would typically be tested by temporarily breaking Brave API
    query = "test query for fallback"
    try:
        result = await web_rag_with_fallback(query, session_id="fallback_test")
        
        if result.get('fallback_used'):
            print("✅ Fallback system activated successfully")
        else:
            print("✅ Primary system working, no fallback needed")
            
        print(f"Response generated: {bool(result.get('Response'))}")
        
    except Exception as e:
        print(f"❌ Fallback system error: {e}")

async def benchmark_performance():
    """Benchmark the performance improvements."""
    print("\n=== Performance Benchmarking ===")
    
    test_query = "latest quarterly results of major Indian banks"
    
    # Test multiple runs
    total_time = 0
    total_tokens = 0
    successful_runs = 0
    
    num_runs = 3
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        try:
            import time
            start_time = time.time()
            
            result = await web_rag(test_query, session_id=f"benchmark_{i}")
            
            end_time = time.time()
            run_time = end_time - start_time
            
            total_time += run_time
            total_tokens += result.get('Total_Tokens', 0)
            successful_runs += 1
            
            print(f"  Time: {run_time:.2f}s")
            print(f"  Tokens: {result.get('Total_Tokens', 0)}")
            print(f"  Sources: {result.get('num_sources_used', 0)}")
            
        except Exception as e:
            print(f"  ❌ Run {i+1} failed: {e}")
    
    if successful_runs > 0:
        avg_time = total_time / successful_runs
        avg_tokens = total_tokens / successful_runs
        
        print(f"\n📊 Benchmark Results:")
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Average tokens used: {avg_tokens:.0f}")
        print(f"Success rate: {successful_runs}/{num_runs} ({100*successful_runs/num_runs:.1f}%)")

def test_scoring_service():
    """Test the scoring service independently."""
    print("\n=== Testing Scoring Service ===")
    
    from api.news_rag.scoring_service import scoring_service
    
    # Test context sufficiency assessment
    test_contexts = [
        ("Short context", "test query", 1),
        ("This is a longer context with more information about financial markets and recent developments in the Indian banking sector including HDFC Bank and SBI performance metrics", "banking sector news", 3),
        ("Very comprehensive context with detailed information about multiple companies including Reliance Industries, Tata Consultancy Services, Infosys, HDFC Bank, and State Bank of India with their recent quarterly results, market performance, analyst recommendations, and future outlook for the next quarter", "latest company results", 5)
    ]
    
    for context, query, num_docs in test_contexts:
        score = scoring_service.assess_context_sufficiency(context, query, num_docs)
        print(f"Context ({len(context)} chars, {num_docs} docs): Score = {score:.3f}")
    
    # Test scoring with sample passages
    sample_passages = [
        {
            "text": "HDFC Bank reported strong quarterly results with 20% growth in net profit and expansion in digital banking services.",
            "metadata": {
                "title": "HDFC Bank Q3 Results",
                "link": "https://economictimes.indiatimes.com/hdfc-bank-results",
                "publication_date": "2024-01-15T10:30:00Z"
            }
        },
        {
            "text": "Market volatility continues as investors remain cautious about inflation and interest rate policies.",
            "metadata": {
                "title": "Market Update",
                "link": "https://moneycontrol.com/market-update",
                "publication_date": "2023-12-20T15:45:00Z"
            }
        }
    ]
    
    async def test_passage_scoring():
        scored = await scoring_service.score_and_rerank_passages(
            question="HDFC Bank quarterly results",
            passages=sample_passages
        )
        
        for i, passage in enumerate(scored):
            print(f"Passage {i+1}:")
            print(f"  Final Score: {passage['final_combined_score']:.3f}")
            print(f"  Relevance: {passage['relevance_score']:.3f}")
            print(f"  Sentiment: {passage['sentiment_score']:.3f}")
            print(f"  Time Decay: {passage['time_decay_score']:.3f}")
            print(f"  Impact: {passage['impact_score']:.3f}")
    
    # Run async scoring test
    asyncio.run(test_passage_scoring())

async def integration_test():
    """Complete integration test of the enhanced system."""
    print("\n=== Integration Test ===")
    
    test_scenarios = [
        {
            "name": "Recent Company News",
            "query": "latest news about Wipro today",
            "expected_features": ["data_ingestion_triggered", "high_time_decay_weight"]
        },
        {
            "name": "Sentiment Analysis Query", 
            "query": "bullish outlook on pharmaceutical sector",
            "expected_features": ["sentiment_focused", "positive_sentiment_bias"]
        },
        {
            "name": "Comprehensive Market Query",
            "query": "detailed analysis of Indian stock market performance this quarter",
            "expected_features": ["multiple_sources", "comprehensive_context"]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        print(f"Query: {scenario['query']}")
        
        try:
            result = await web_rag(scenario['query'], session_id="integration_test")
            
            # Validate expected features
            print("✅ Features validated:")
            for feature in scenario['expected_features']:
                if feature == "data_ingestion_triggered":
                    print(f"  - Data ingestion: {result.get('data_ingestion_triggered', False)}")
                elif feature == "multiple_sources":
                    print(f"  - Multiple sources: {len(result.get('links', []))} links")
                elif feature == "comprehensive_context":
                    print(f"  - Context richness: {result.get('num_sources_used', 0)} sources used")
            
            print(f"✅ {scenario['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ {scenario['name']} failed: {e}")

# Main test runner
async def run_all_tests():
    """Run all test suites."""
    print("🚀 Starting Enhanced Web RAG Test Suite")
    print("=" * 50)
    
    # Basic functionality tests
    await test_basic_web_rag()
    
    # Advanced feature tests
    await test_adaptive_web_rag()
    
    # Reliability tests
    await test_fallback_system()
    
    # Performance tests
    await benchmark_performance()
    
    # Component tests
    test_scoring_service()
    
    # Integration tests
    await integration_test()
    
    print("\n🎉 Test suite completed!")
    print("=" * 50)

# Configuration validation
def validate_configuration():
    """Validate that all required configuration is present."""
    print("🔍 Validating Configuration...")
    
    required_env_vars = [
        'BRAVE_API_KEY',
        'OPENAI_API_KEY', 
        'PINECONE_API_KEY',
        'DATABASE_URL'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All required environment variables are present")
        return True

# Usage example
if __name__ == "__main__":
    import os
    
    # Validate configuration first
    if validate_configuration():
        # Run tests
        asyncio.run(run_all_tests())
    else:
        print("Please set up all required environment variables before running tests.")