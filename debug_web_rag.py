#!/usr/bin/env python3
"""
Web RAG Debug Tool
Comprehensive debugging utility for the enhanced web_rag endpoint

This tool tests every component of the web_rag system to identify issues:
1. Environment configuration
2. Database connections
3. Brave search functionality
4. Pinecone vector stores
5. ChromaDB access
6. Scoring service
7. LLM response generation
8. End-to-end flow
"""

import os
import sys
import asyncio
import json
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent if "app_service" in str(CURRENT_DIR) else CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

class WebRagDebugger:
    def __init__(self):
        self.test_results = {}
        self.debug_session_id = f"debug_session_{int(time.time())}"
        self.test_query = "latest news on Adani stocks"
        
    async def run_all_tests(self):
        """Run comprehensive debug suite"""
        print_header("WEB RAG COMPREHENSIVE DEBUG SUITE")
        print_info(f"Debug Session ID: {self.debug_session_id}")
        print_info(f"Test Query: '{self.test_query}'")
        print_info(f"Timestamp: {datetime.now()}")
        
        # Run all debug tests
        await self.test_environment_setup()
        await self.test_database_connections()
        await self.test_brave_search()
        await self.test_vector_stores()
        await self.test_scoring_service()
        await self.test_llm_components()
        await self.test_memory_chain()
        await self.test_date_extraction()
        await self.test_end_to_end_flow()
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.test_results

    async def test_environment_setup(self):
        """Test environment variables and configuration"""
        print_header("1. ENVIRONMENT SETUP TEST")
        
        env_tests = {}
        
        try:
            # Test .env file loading
            from dotenv import load_dotenv
            env_path = Path(__file__).resolve().parent.parent / '.env'
            print_info(f"Looking for .env at: {env_path}")
            
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                print_success(".env file found and loaded")
                env_tests['env_file'] = True
            else:
                print_error(".env file not found")
                env_tests['env_file'] = False
                
            # Test critical environment variables
            critical_vars = {
                'BRAVE_API_KEY': os.getenv('BRAVE_API_KEY'),
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
                'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
                'DATABASE_URL': os.getenv('DATABASE_URL'),
                'CHROMA_HOST': os.getenv('CHROMA_HOST'),
                'GROQ_API_KEY': os.getenv('GROQ_API_KEY')
            }
            
            for var_name, var_value in critical_vars.items():
                if var_value:
                    print_success(f"{var_name}: ✓ (length: {len(var_value)})")
                    env_tests[var_name] = True
                else:
                    print_error(f"{var_name}: Missing!")
                    env_tests[var_name] = False
            
            # Test config loading
            try:
                from config import (
                    W_RELEVANCE, W_SENTIMENT, W_TIME_DECAY, W_IMPACT,
                    CONTEXT_SUFFICIENCY_THRESHOLD,
                    SOURCE_CREDIBILITY_WEIGHTS,
                    IMPACT_KEYWORDS
                )
                print_success("Config constants loaded successfully")
                print_info(f"Scoring weights - Relevance: {W_RELEVANCE}, Sentiment: {W_SENTIMENT}")
                print_info(f"Context threshold: {CONTEXT_SUFFICIENCY_THRESHOLD}")
                env_tests['config_loading'] = True
            except Exception as e:
                print_error(f"Config loading failed: {e}")
                env_tests['config_loading'] = False
                
        except Exception as e:
            print_error(f"Environment setup failed: {e}")
            env_tests['overall'] = False
            
        self.test_results['environment'] = env_tests

    async def test_database_connections(self):
        """Test PostgreSQL and ChromaDB connections"""
        print_header("2. DATABASE CONNECTIONS TEST")
        
        db_tests = {}
        
        # Test PostgreSQL
        try:
            import psycopg2
            import asyncpg
            
            psql_url = os.getenv('DATABASE_URL')
            if not psql_url:
                print_error("DATABASE_URL not found")
                db_tests['postgresql'] = False
            else:
                # Test sync connection
                try:
                    conn = psycopg2.connect(psql_url)
                    cursor = conn.cursor()
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
                    print_success(f"PostgreSQL sync connection: {version[0][:50]}...")
                    cursor.close()
                    conn.close()
                    db_tests['postgresql_sync'] = True
                except Exception as e:
                    print_error(f"PostgreSQL sync connection failed: {e}")
                    db_tests['postgresql_sync'] = False
                
                # Test async connection
                try:
                    async_conn = await asyncpg.connect(psql_url)
                    version = await async_conn.fetchval("SELECT version();")
                    print_success(f"PostgreSQL async connection: {version[:50]}...")
                    await async_conn.close()
                    db_tests['postgresql_async'] = True
                except Exception as e:
                    print_error(f"PostgreSQL async connection failed: {e}")
                    db_tests['postgresql_async'] = False
                    
                # Test source_data table
                try:
                    conn = psycopg2.connect(psql_url)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM source_data LIMIT 1;")
                    count = cursor.fetchone()[0]
                    print_success(f"source_data table accessible, {count} records")
                    cursor.close()
                    conn.close()
                    db_tests['source_data_table'] = True
                except Exception as e:
                    print_error(f"source_data table test failed: {e}")
                    db_tests['source_data_table'] = False
                    
        except Exception as e:
            print_error(f"PostgreSQL test failed: {e}")
            db_tests['postgresql'] = False
        
        # Test ChromaDB
        try:
            import chromadb
            from config import chroma_server_client
            
            # Test client connection
            collections = chroma_server_client.list_collections()
            print_success(f"ChromaDB connected, {len(collections)} collections")
            
            # Look for cmots_news collection
            collection_names = [c.name for c in collections]
            if 'cmots_news' in collection_names:
                cmots_collection = chroma_server_client.get_collection("cmots_news")
                count = cmots_collection.count()
                print_success(f"cmots_news collection found with {count} documents")
                db_tests['chromadb_cmots'] = True
            else:
                print_error("cmots_news collection not found")
                print_info(f"Available collections: {collection_names}")
                db_tests['chromadb_cmots'] = False
                
            db_tests['chromadb'] = True
            
        except Exception as e:
            print_error(f"ChromaDB test failed: {e}")
            traceback.print_exc()
            db_tests['chromadb'] = False
            
        self.test_results['database'] = db_tests

    async def test_brave_search(self):
        """Test Brave search functionality"""
        print_header("3. BRAVE SEARCH TEST")
        
        brave_tests = {}
        
        try:
            from api.news_rag.brave_news import BraveNewsSearcher, get_brave_results
            
            brave_api_key = os.getenv('BRAVE_API_KEY')
            if not brave_api_key:
                print_error("BRAVE_API_KEY not found")
                brave_tests['api_key'] = False
                self.test_results['brave_search'] = brave_tests
                return
            
            print_success("BRAVE_API_KEY found")
            brave_tests['api_key'] = True
            
            # Test BraveNewsSearcher initialization
            try:
                searcher = BraveNewsSearcher(brave_api_key)
                print_success("BraveNewsSearcher initialized")
                brave_tests['searcher_init'] = True
            except Exception as e:
                print_error(f"BraveNewsSearcher initialization failed: {e}")
                brave_tests['searcher_init'] = False
                self.test_results['brave_search'] = brave_tests
                return
            
            # Test domain filtering
            test_urls = [
                "https://www.moneycontrol.com/news/test",
                "https://economictimes.indiatimes.com/test",
                "https://example.com/test"
            ]
            
            for url in test_urls:
                is_valid = searcher._url_in_domain_list(url)
                print_info(f"Domain filter test: {url} -> {is_valid}")
            
            brave_tests['domain_filtering'] = True
            
            # Test actual search
            print_info(f"Testing search for: '{self.test_query}'")
            try:
                results = await searcher.search_and_scrape(self.test_query)
                if results:
                    print_success(f"Brave search returned {len(results)} results")
                    
                    # Show first result details
                    if results:
                        first_result = results[0]
                        print_info("First result preview:")
                        print_info(f"  Original title: {first_result['original_item'].get('title', 'N/A')[:100]}")
                        print_info(f"  Link: {first_result['original_item'].get('link', 'N/A')}")
                        print_info(f"  Text length: {len(first_result['text_to_embed'])}")
                    
                    brave_tests['search_results'] = True
                    brave_tests['result_count'] = len(results)
                else:
                    print_warning("Brave search returned no results")
                    brave_tests['search_results'] = False
                    brave_tests['result_count'] = 0
                    
            except Exception as e:
                print_error(f"Brave search failed: {e}")
                traceback.print_exc()
                brave_tests['search_results'] = False
                
            # Test get_brave_results wrapper function
            try:
                print_info("Testing get_brave_results wrapper function...")
                articles, df = await get_brave_results(self.test_query)
                
                if articles:
                    print_success(f"get_brave_results returned {len(articles)} articles")
                    brave_tests['wrapper_function'] = True
                else:
                    print_warning("get_brave_results returned no articles")
                    brave_tests['wrapper_function'] = False
                    
                if df is not None and not df.empty:
                    print_success(f"DataFrame created with {len(df)} rows")
                    print_info(f"DataFrame columns: {list(df.columns)}")
                    brave_tests['dataframe_creation'] = True
                else:
                    print_warning("No DataFrame created")
                    brave_tests['dataframe_creation'] = False
                    
            except Exception as e:
                print_error(f"get_brave_results test failed: {e}")
                traceback.print_exc()
                brave_tests['wrapper_function'] = False
                
        except Exception as e:
            print_error(f"Brave search test failed: {e}")
            traceback.print_exc()
            brave_tests['overall'] = False
            
        self.test_results['brave_search'] = brave_tests

    async def test_vector_stores(self):
        """Test Pinecone and ChromaDB vector stores"""
        print_header("4. VECTOR STORES TEST")
        
        vector_tests = {}
        
        # Test Pinecone
        try:
            from pinecone import Pinecone as PineconeClient
            from langchain_pinecone import PineconeVectorStore
            from langchain_openai import OpenAIEmbeddings
            
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                print_error("PINECONE_API_KEY not found")
                vector_tests['pinecone_key'] = False
            else:
                print_success("PINECONE_API_KEY found")
                vector_tests['pinecone_key'] = True
                
                # Test Pinecone client
                try:
                    pc = PineconeClient(api_key=pinecone_api_key)
                    indexes = pc.list_indexes()
                    print_success(f"Pinecone client connected, {len(indexes)} indexes")
                    
                    # Check for required indexes
                    index_names = [idx.name for idx in indexes]
                    required_indexes = ["newsrag11052024", "bing-news"]
                    
                    for idx_name in required_indexes:
                        if idx_name in index_names:
                            print_success(f"Index '{idx_name}' found")
                            vector_tests[f'pinecone_{idx_name}'] = True
                        else:
                            print_error(f"Index '{idx_name}' not found")
                            vector_tests[f'pinecone_{idx_name}'] = False
                    
                    vector_tests['pinecone_client'] = True
                    
                    # Test vector store creation
                    try:
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                        
                        # Test news_rag vectorstore
                        if "newsrag11052024" in index_names:
                            news_rag_vs = PineconeVectorStore(
                                index_name="newsrag11052024", 
                                embedding=embeddings, 
                                namespace='news'
                            )
                            print_success("news_rag vectorstore created")
                            vector_tests['news_rag_vectorstore'] = True
                        
                        # Test brave_news vectorstore  
                        if "bing-news" in index_names:
                            brave_news_vs = PineconeVectorStore(
                                index_name="bing-news", 
                                embedding=embeddings, 
                                namespace='bing'
                            )
                            print_success("brave_news vectorstore created")
                            vector_tests['brave_news_vectorstore'] = True
                            
                            # Test query
                            try:
                                test_results = brave_news_vs.similarity_search("test query", k=1)
                                print_success(f"Brave news vectorstore query returned {len(test_results)} results")
                                vector_tests['brave_news_query'] = True
                            except Exception as e:
                                print_warning(f"Brave news vectorstore query failed: {e}")
                                vector_tests['brave_news_query'] = False
                                
                    except Exception as e:
                        print_error(f"Vector store creation failed: {e}")
                        vector_tests['vectorstore_creation'] = False
                        
                except Exception as e:
                    print_error(f"Pinecone client test failed: {e}")
                    vector_tests['pinecone_client'] = False
                    
        except Exception as e:
            print_error(f"Pinecone test failed: {e}")
            vector_tests['pinecone'] = False
        
        # Test ChromaDB vector store
        try:
            from langchain_chroma import Chroma
            from config import chroma_server_client, embeddings
            
            # Test CMOTS vectorstore
            try:
                cmots_vs = Chroma(
                    client=chroma_server_client,
                    collection_name="cmots_news",
                    embedding_function=embeddings,
                )
                
                # Test query
                test_results = cmots_vs.similarity_search("test query", k=1)
                print_success(f"CMOTS vectorstore query returned {len(test_results)} results")
                vector_tests['cmots_vectorstore'] = True
                
                # Test with actual query
                adani_results = cmots_vs.similarity_search(self.test_query, k=5)
                print_success(f"CMOTS Adani query returned {len(adani_results)} results")
                if adani_results:
                    print_info(f"First result preview: {adani_results[0].page_content[:200]}...")
                vector_tests['cmots_adani_query'] = True
                
            except Exception as e:
                print_error(f"CMOTS vectorstore test failed: {e}")
                vector_tests['cmots_vectorstore'] = False
                
        except Exception as e:
            print_error(f"ChromaDB vectorstore test failed: {e}")
            vector_tests['chromadb_vectorstore'] = False
            
        self.test_results['vector_stores'] = vector_tests

    async def test_scoring_service(self):
        """Test the scoring service functionality"""
        print_header("5. SCORING SERVICE TEST")
        
        scoring_tests = {}
        
        try:
            from api.news_rag.scoring_service import scoring_service
            
            # Test service initialization
            print_info("Testing scoring service initialization...")
            if hasattr(scoring_service, 'sentiment_analyzer'):
                if scoring_service.sentiment_analyzer:
                    print_success("FinBERT sentiment analyzer loaded")
                    scoring_tests['sentiment_analyzer'] = True
                else:
                    print_warning("FinBERT sentiment analyzer not loaded")
                    scoring_tests['sentiment_analyzer'] = False
            
            if hasattr(scoring_service, 'cross_encoder_model'):
                if scoring_service.cross_encoder_model:
                    print_success("CrossEncoder model loaded")
                    scoring_tests['cross_encoder'] = True
                else:
                    print_warning("CrossEncoder model not loaded")
                    scoring_tests['cross_encoder'] = False
            
            # Test context sufficiency assessment
            try:
                test_context = "Adani Group stocks have been volatile recently. The company reported quarterly earnings."
                sufficiency_score = scoring_service.assess_context_sufficiency(
                    test_context, self.test_query, num_docs=2
                )
                print_success(f"Context sufficiency test: {sufficiency_score:.3f}")
                scoring_tests['context_sufficiency'] = True
            except Exception as e:
                print_error(f"Context sufficiency test failed: {e}")
                scoring_tests['context_sufficiency'] = False
            
            # Test individual scoring components
            test_passage = {
                "text": "Adani Group stocks surged 5% today following positive quarterly results. The company's revenue grew significantly.",
                "metadata": {
                    "link": "https://economictimes.indiatimes.com/test",
                    "publication_date": "2024-01-01",
                    "title": "Adani stocks surge on positive results"
                }
            }
            
            try:
                # Test relevance scoring
                relevance = scoring_service._calculate_relevance_score(self.test_query, test_passage["text"])
                print_success(f"Relevance score: {relevance:.3f}")
                scoring_tests['relevance_scoring'] = True
                
                # Test sentiment scoring
                sentiment = scoring_service._calculate_sentiment_score(test_passage["text"], self.test_query)
                print_success(f"Sentiment score: {sentiment:.3f}")
                scoring_tests['sentiment_scoring'] = True
                
                # Test time decay scoring
                time_decay = scoring_service._calculate_time_decay_score(
                    test_passage["metadata"]["publication_date"], self.test_query
                )
                print_success(f"Time decay score: {time_decay:.3f}")
                scoring_tests['time_decay_scoring'] = True
                
                # Test impact scoring
                impact = scoring_service._calculate_impact_score(
                    test_passage["text"], test_passage["metadata"]["link"]
                )
                print_success(f"Impact score: {impact:.3f}")
                scoring_tests['impact_scoring'] = True
                
            except Exception as e:
                print_error(f"Individual scoring test failed: {e}")
                traceback.print_exc()
                scoring_tests['individual_scoring'] = False
            
            # Test full scoring pipeline
            try:
                test_passages = [test_passage]
                scored_passages = await scoring_service.score_and_rerank_passages(
                    self.test_query, test_passages
                )
                
                if scored_passages:
                    print_success(f"Full scoring pipeline completed, {len(scored_passages)} passages scored")
                    first_scored = scored_passages[0]
                    print_info(f"Final score: {first_scored.get('final_combined_score', 0):.3f}")
                    scoring_tests['full_pipeline'] = True
                else:
                    print_error("Full scoring pipeline returned empty results")
                    scoring_tests['full_pipeline'] = False
                    
            except Exception as e:
                print_error(f"Full scoring pipeline test failed: {e}")
                traceback.print_exc()
                scoring_tests['full_pipeline'] = False
                
        except Exception as e:
            print_error(f"Scoring service test failed: {e}")
            traceback.print_exc()
            scoring_tests['overall'] = False
            
        self.test_results['scoring_service'] = scoring_tests

    async def test_llm_components(self):
        """Test LLM components and response generation"""
        print_header("6. LLM COMPONENTS TEST")
        
        llm_tests = {}
        
        try:
            from config import llm_stream, llm_date
            from langchain_core.messages import HumanMessage
            
            # Test OpenAI models
            try:
                response = llm_stream.invoke([HumanMessage(content="Hello, this is a test.")])
                print_success(f"GPT-4o-mini responded: {response.content[:50]}...")
                llm_tests['gpt4o_mini'] = True
            except Exception as e:
                print_error(f"GPT-4o-mini test failed: {e}")
                llm_tests['gpt4o_mini'] = False
            
            try:
                response = llm_date.invoke([HumanMessage(content="What is today's date?")])
                print_success(f"GPT-4o date model responded: {response.content[:50]}...")
                llm_tests['gpt4o_date'] = True
            except Exception as e:
                print_error(f"GPT-4o date model test failed: {e}")
                llm_tests['gpt4o_date'] = False
            
            # Test Groq models if available
            try:
                response = llama3.invoke([HumanMessage(content="Hello, this is a test.")])
                print_success(f"Llama3-70b responded: {response.content[:50]}...")
                llm_tests['llama3_70b'] = True
            except Exception as e:
                print_warning(f"Llama3-70b test failed: {e}")
                llm_tests['llama3_70b'] = False
            
            # Test JSON output parsing
            try:
                from langchain_core.output_parsers import JsonOutputParser
                
                test_json = '{"answer": "Test response", "links": ["http://example.com"]}'
                parser = JsonOutputParser()
                parsed = parser.parse(test_json)
                print_success(f"JSON parsing test successful: {parsed}")
                llm_tests['json_parsing'] = True
            except Exception as e:
                print_error(f"JSON parsing test failed: {e}")
                llm_tests['json_parsing'] = False
                
        except Exception as e:
            print_error(f"LLM components test failed: {e}")
            llm_tests['overall'] = False
            
        self.test_results['llm_components'] = llm_tests

    async def test_memory_chain(self):
        """Test memory chain and chat history functionality"""
        print_header("7. MEMORY CHAIN TEST")
        
        memory_tests = {}
        
        try:
            from api.news_rag.news_rag import memory_chain
            import psycopg2
            from langchain_postgres import PostgresChatMessageHistory
            
            # Test with empty history (new session)
            try:
                result = memory_chain(self.test_query, self.debug_session_id)
                print_success(f"Memory chain with empty history: '{result}'")
                memory_tests['empty_history'] = True
            except Exception as e:
                print_error(f"Memory chain with empty history failed: {e}")
                memory_tests['empty_history'] = False
            
            # Test adding to chat history
            try:
                psql_url = os.getenv('DATABASE_URL')
                if psql_url:
                    history = PostgresChatMessageHistory(
                        connection_string=psql_url, 
                        session_id=self.debug_session_id
                    )
                    history.add_user_message("Tell me about Adani Group")
                    history.add_ai_message("Adani Group is an Indian multinational conglomerate...")
                    
                    # Test memory chain with history
                    result = memory_chain("What are their latest results?", self.debug_session_id)
                    print_success(f"Memory chain with history: '{result}'")
                    memory_tests['with_history'] = True
                else:
                    print_error("DATABASE_URL not available for history test")
                    memory_tests['with_history'] = False
                    
            except Exception as e:
                print_error(f"Memory chain with history failed: {e}")
                memory_tests['with_history'] = False
                
        except Exception as e:
            print_error(f"Memory chain test failed: {e}")
            memory_tests['overall'] = False
            
        self.test_results['memory_chain'] = memory_tests

    async def test_date_extraction(self):
        """Test date extraction and query processing"""
        print_header("8. DATE EXTRACTION TEST")
        
        date_tests = {}
        
        try:
            from api.news_rag.news_rag import llm_get_date, split_input
            
            # Test various date queries
            test_queries = [
                "latest news on Adani stocks",
                "today's Adani stock price",
                "recent Adani developments", 
                "Adani news from last week",
                "quarterly results of Adani",
                "Adani stock performance"
            ]
            
            for query in test_queries:
                try:
                    date, general_query, today = llm_get_date(query)
                    print_info(f"Query: '{query}'")
                    print_info(f"  -> Date: {date}, General Query: '{general_query}', Today: {today}")
                    date_tests[f'date_extraction_{query[:20]}'] = True
                except Exception as e:
                    print_error(f"Date extraction failed for '{query}': {e}")
                    date_tests[f'date_extraction_{query[:20]}'] = False
            
            # Test split_input function
            try:
                test_input = "20240101,Adani stock news"
                date, query = split_input(test_input)
                print_success(f"split_input test: '{test_input}' -> date: '{date}', query: '{query}'")
                date_tests['split_input'] = True
            except Exception as e:
                print_error(f"split_input test failed: {e}")
                date_tests['split_input'] = False
                
        except Exception as e:
            print_error(f"Date extraction test failed: {e}")
            date_tests['overall'] = False
            
        self.test_results['date_extraction'] = date_tests

    async def test_end_to_end_flow(self):
        """Test the complete end-to-end web_rag flow"""
        print_header("9. END-TO-END WEB_RAG TEST")
        
        e2e_tests = {}
        
        try:
            from api.news_rag.news_rag import web_rag
            
            print_info(f"Running complete web_rag flow with query: '{self.test_query}'")
            print_info(f"Session ID: {self.debug_session_id}")
            
            start_time = time.time()
            
            try:
                result = await web_rag(self.test_query, self.debug_session_id)
                
                end_time = time.time()
                duration = end_time - start_time
                
                print_success(f"web_rag completed in {duration:.2f} seconds")
                
                # Analyze the result
                if isinstance(result, dict):
                    print_info("Result structure analysis:")
                    for key, value in result.items():
                        if key == 'Response':
                            print_info(f"  {key}: {value}")
                        else:
                            print_info(f"  {key}: {value}")
                    
                    # Check for the specific issue
                    response_text = result.get('Response', '')
                    if "I'm unable to provide financial information" in response_text or "has not been addressed in the chat history" in response_text:
                        print_error("❌ ISSUE IDENTIFIED: Generic 'unable to provide' response detected!")
                        print_error("This indicates the RAG system is not retrieving relevant context")
                        e2e_tests['generic_response_issue'] = True
                        e2e_tests['successful_response'] = False
                    elif len(response_text.strip()) < 50:
                        print_warning("⚠️ Very short response - possible retrieval issue")
                        e2e_tests['short_response'] = True
                        e2e_tests['successful_response'] = False
                    else:
                        print_success("✓ Meaningful response generated")
                        e2e_tests['successful_response'] = True
                    
                    # Check links
                    links = result.get('links', [])
                    if links:
                        print_success(f"✓ {len(links)} source links provided")
                        e2e_tests['source_links'] = True
                    else:
                        print_warning("⚠️ No source links provided")
                        e2e_tests['source_links'] = False
                    
                    # Check token usage
                    total_tokens = result.get('Total_Tokens', 0)
                    if total_tokens > 0:
                        print_success(f"✓ Token usage: {total_tokens}")
                        e2e_tests['token_usage'] = True
                    else:
                        print_warning("⚠️ No token usage recorded")
                        e2e_tests['token_usage'] = False
                    
                    # Check new features
                    if 'context_sufficiency_score' in result:
                        score = result['context_sufficiency_score']
                        print_info(f"Context sufficiency score: {score}")
                        if score < 0.6:
                            print_warning("⚠️ Low context sufficiency - may trigger unnecessary web scraping")
                        e2e_tests['sufficiency_scoring'] = True
                    
                    if 'data_ingestion_triggered' in result:
                        triggered = result['data_ingestion_triggered']
                        print_info(f"Data ingestion triggered: {triggered}")
                        e2e_tests['data_ingestion_logic'] = True
                        
                else:
                    print_error(f"Unexpected result type: {type(result)}")
                    print_error(f"Result: {result}")
                    e2e_tests['result_format'] = False
                
                e2e_tests['execution_success'] = True
                e2e_tests['execution_time'] = duration
                
            except Exception as e:
                print_error(f"web_rag execution failed: {e}")
                traceback.print_exc()
                e2e_tests['execution_success'] = False
                e2e_tests['error_message'] = str(e)
                
        except Exception as e:
            print_error(f"End-to-end test setup failed: {e}")
            e2e_tests['setup'] = False
            
        self.test_results['end_to_end'] = e2e_tests

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print_header("DEBUG SUMMARY REPORT")
        
        total_tests = 0
        passed_tests = 0
        critical_issues = []
        warnings = []
        recommendations = []
        
        for category, tests in self.test_results.items():
            print(f"\n{Colors.BOLD}{Colors.BLUE}{category.upper()}:{Colors.ENDC}")
            
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if result is True:
                        print_success(f"  {test_name}")
                        passed_tests += 1
                    elif result is False:
                        print_error(f"  {test_name}")
                        critical_issues.append(f"{category}.{test_name}")
                    else:
                        print_info(f"  {test_name}: {result}")
        
        # Calculate pass rate
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n{Colors.BOLD}OVERALL STATISTICS:{Colors.ENDC}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {Colors.GREEN}{passed_tests}{Colors.ENDC}")
        print(f"  Failed: {Colors.RED}{total_tests - passed_tests}{Colors.ENDC}")
        print(f"  Pass Rate: {Colors.CYAN}{pass_rate:.1f}%{Colors.ENDC}")
        
        # Analyze critical issues and provide recommendations
        print(f"\n{Colors.BOLD}{Colors.RED}CRITICAL ISSUES IDENTIFIED:{Colors.ENDC}")
        
        if not self.test_results.get('environment', {}).get('BRAVE_API_KEY'):
            critical_issues.append("BRAVE_API_KEY missing")
            recommendations.append("Add BRAVE_API_KEY to your .env file")
        
        if not self.test_results.get('vector_stores', {}).get('cmots_vectorstore'):
            critical_issues.append("CMOTS vectorstore not accessible")
            recommendations.append("Check ChromaDB connection and cmots_news collection")
        
        if not self.test_results.get('brave_search', {}).get('search_results'):
            critical_issues.append("Brave search not returning results")
            recommendations.append("Verify Brave API key and network connectivity")
        
        if self.test_results.get('end_to_end', {}).get('generic_response_issue'):
            critical_issues.append("RAG system returning generic 'unable to provide' responses")
            recommendations.append("Check vector store queries and context retrieval")
        
        if not self.test_results.get('database', {}).get('postgresql_sync'):
            critical_issues.append("PostgreSQL connection failed")
            recommendations.append("Verify DATABASE_URL and PostgreSQL server status")
        
        # Print issues and recommendations
        if critical_issues:
            for i, issue in enumerate(critical_issues, 1):
                print_error(f"  {i}. {issue}")
        else:
            print_success("  No critical issues identified")
        
        print(f"\n{Colors.BOLD}{Colors.YELLOW}RECOMMENDATIONS:{Colors.ENDC}")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print_info(f"  {i}. {rec}")
        else:
            print_success("  No specific recommendations - system appears healthy")
        
        # Specific troubleshooting for the main issue
        if self.test_results.get('end_to_end', {}).get('generic_response_issue'):
            print(f"\n{Colors.BOLD}{Colors.RED}MAIN ISSUE DIAGNOSIS:{Colors.ENDC}")
            print_error("The system is returning generic 'unable to provide' responses.")
            print_info("This usually indicates one of these issues:")
            print_info("  1. Vector stores not returning relevant documents")
            print_info("  2. Context sufficiency check failing unnecessarily") 
            print_info("  3. LLM prompt template issues")
            print_info("  4. Empty or insufficient search results")
            
            print(f"\n{Colors.BOLD}IMMEDIATE ACTION ITEMS:{Colors.ENDC}")
            print_info("1. Check if CMOTS vectorstore has Adani-related documents:")
            print_info("   - Test direct similarity search on ChromaDB")
            print_info("2. Verify Brave search is returning relevant results:")
            print_info("   - Check domain filtering logic")
            print_info("3. Check context sufficiency threshold:")
            print_info("   - May be set too high, causing unnecessary web scraping")
            print_info("4. Examine LLM prompt templates:")
            print_info("   - Ensure proper context injection")
        
        # Generate debug file with detailed results
        debug_file_path = Path(__file__).parent / f"debug_results_{int(time.time())}.json"
        try:
            with open(debug_file_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print_success(f"Detailed debug results saved to: {debug_file_path}")
        except Exception as e:
            print_warning(f"Could not save debug results: {e}")

async def main():
    """Main function to run the debug suite"""
    debugger = WebRagDebugger()
    
    try:
        results = await debugger.run_all_tests()
        return results
    except KeyboardInterrupt:
        print_warning("\nDebug interrupted by user")
        return None
    except Exception as e:
        print_error(f"Debug suite failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the debug suite
    print(f"{Colors.BOLD}{Colors.CYAN}Starting Web RAG Debug Suite...{Colors.ENDC}")
    results = asyncio.run(main())
    
    if results:
        print(f"\n{Colors.BOLD}{Colors.GREEN}Debug suite completed successfully!{Colors.ENDC}")
        print(f"{Colors.CYAN}Check the output above for detailed analysis and recommendations.{Colors.ENDC}")
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}Debug suite encountered issues.{Colors.ENDC}")
        print(f"{Colors.YELLOW}Please review the error messages above.{Colors.ENDC}")

# Additional utility functions for manual testing
class ManualDebugHelpers:
    """Helper functions for manual debugging"""
    
    @staticmethod
    async def test_specific_component(component_name: str):
        """Test a specific component manually"""
        debugger = WebRagDebugger()
        
        component_map = {
            'environment': debugger.test_environment_setup,
            'database': debugger.test_database_connections,
            'brave': debugger.test_brave_search,
            'vectors': debugger.test_vector_stores,
            'scoring': debugger.test_scoring_service,
            'llm': debugger.test_llm_components,
            'memory': debugger.test_memory_chain,
            'date': debugger.test_date_extraction,
            'e2e': debugger.test_end_to_end_flow
        }
        
        if component_name in component_map:
            await component_map[component_name]()
            return debugger.test_results
        else:
            print_error(f"Unknown component: {component_name}")
            print_info(f"Available components: {list(component_map.keys())}")
            return None
    
    @staticmethod
    async def quick_vector_search_test():
        """Quick test of vector search functionality"""
        try:
            from config import vs  # CMOTS vectorstore
            from langchain_pinecone import PineconeVectorStore
            from langchain_openai import OpenAIEmbeddings
            
            print_header("QUICK VECTOR SEARCH TEST")
            
            # Test CMOTS
            print_info("Testing CMOTS vectorstore...")
            results = vs.similarity_search("Adani", k=5)
            print_success(f"CMOTS returned {len(results)} results")
            if results:
                print_info(f"First result: {results[0].page_content[:200]}...")
            
            # Test Pinecone if available
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                brave_vs = PineconeVectorStore(
                    index_name="bing-news", 
                    embedding=embeddings, 
                    namespace='bing'
                )
                
                results = brave_vs.similarity_search("Adani", k=5)
                print_success(f"Pinecone returned {len(results)} results")
                if results:
                    print_info(f"First result: {results[0].page_content[:200]}...")
                    
            except Exception as e:
                print_warning(f"Pinecone test failed: {e}")
                
        except Exception as e:
            print_error(f"Vector search test failed: {e}")
    
    @staticmethod
    async def test_llm_with_context():
        """Test LLM response with mock context"""
        try:
            from config import llm_stream
            from langchain_core.prompts import ChatPromptTemplate
            
            print_header("LLM CONTEXT TEST")
            
            mock_context = """
            Title: Adani Group Stocks Rally 5% on Strong Q4 Results
            Summary: Adani Group companies saw significant gains today following the release of strong quarterly results.
            Source: https://economictimes.indiatimes.com/test
            Date: 2024-01-15

            Title: Adani Enterprises Reports Record Revenue Growth  
            Summary: The flagship company of Adani Group reported record revenue growth in the latest quarter.
            Source: https://moneycontrol.com/test
            Date: 2024-01-14
            """
            
            prompt_template = """You are a financial markets AI assistant.
            Today's date is {date}.
            
            Based on the following financial news context, answer the user's question:
            
            Context:
            {context}
            
            User Question: {input}
            
            Format your response as JSON with "answer" and "links" keys.
            """
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            response = llm_stream.invoke(prompt.format(
                context=mock_context,
                input="latest news on Adani stocks", 
                date="2024-01-15"
            ))
            
            print_success("LLM response with mock context:")
            print_info(f"Response: {response.content}")
            
        except Exception as e:
            print_error(f"LLM context test failed: {e}")
            traceback.print_exc()

# Example usage:
# python debug_web_rag.py
# 
# Or for specific component testing:
# results = await ManualDebugHelpers.test_specific_component('brave')
# await ManualDebugHelpers.quick_vector_search_test()
# await ManualDebugHelpers.test_llm_with_context()