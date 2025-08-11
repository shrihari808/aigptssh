# /aigptcur/app_service/api/market_data_lm/router.py
# This file contains the router for the MarketDataLM functionality.

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langdetect import detect, DetectorFactory
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

# --- IMPORTANT: Updated imports to reflect the new project structure ---
# The 'app_service.config' import assumes your PYTHONPATH is set to the project root.
# The '.services.*' imports are relative imports from within the 'market_data_lm' module.
from app_service.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    BRAVE_API_KEY,
    PINECONE_MAX_WAIT_TIME,
    PINECONE_CHECK_INTERVAL
)
from .services.embedding_generator import EmbeddingGenerator
from .services.brave_searcher import BraveSearcher
from .services.pinecone_manager import PineconeManager
from .services.llm_service import LLMService

# --- Router Setup ---
# Initialize an APIRouter instead of a full FastAPI app.
market_data_router = APIRouter(
    prefix="/marketdata",  # Add a prefix to all routes in this router
    tags=["Market Data LM"] # Tag for organizing API docs
)

# Set seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

# --- Global service instances for this module ---
# These will be initialized on startup.
embedding_generator: EmbeddingGenerator = None
brave_searcher: BraveSearcher = None
pinecone_manager: PineconeManager = None
llm_service: LLMService = None

@market_data_router.on_event("startup")
async def startup_event():
    """
    Initializes all necessary services for the Market Data LM module
    when the main application starts.
    """
    global embedding_generator, brave_searcher, pinecone_manager, llm_service

    try:
        # Initialize service instances
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        embedding_generator = EmbeddingGenerator(openai_api_key=OPENAI_API_KEY)
        brave_searcher = BraveSearcher(brave_api_key=BRAVE_API_KEY)
        pinecone_manager = PineconeManager(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE_INDEX_NAME
        )
        pinecone_manager.connect_to_index()

        # Initialize the CrossEncoder model
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Initialize the LLMService, injecting other service instances
        llm_service = LLMService(
            openai_client=openai_client,
            cross_encoder_model=cross_encoder_model,
            embedding_generator=embedding_generator,
            brave_searcher=brave_searcher,
            pinecone_manager=pinecone_manager
        )
        print("DEBUG: Market Data LM services initialized successfully.")

    except ValueError as ve:
        print(f"ERROR: MarketDataLM Configuration error: {str(ve)}")
        # In a real app, you might want to handle this more gracefully than raising an exception
        # that stops the server, but for now, this makes misconfigurations obvious.
        raise HTTPException(status_code=500, detail=f"MarketDataLM Configuration error: {str(ve)}. Check your .env file.")
    except Exception as e:
        print(f"ERROR: Failed to initialize MarketDataLM services: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize MarketDataLM services: {str(e)}. Ensure all API keys are correct and Pinecone index exists."
        )

@market_data_router.get("/query")
async def ask_llm_with_ingestion(question: str):
    """
    Combines market data ingestion and LLM querying into a single endpoint.
    It first attempts to query existing data. If the data is insufficient, it
    scrapes and stores new data relevant to the question, then uses that
    data to generate an answer.
    """
    # Ensure all services were initialized on startup before proceeding
    if not all([embedding_generator, brave_searcher, pinecone_manager, llm_service]):
        raise HTTPException(status_code=503, detail="Market Data services are not available. The server might be starting up or encountered an initialization error.")

    async def combined_stream_generator():
        """Generator function to stream the response back to the client."""
        yield "Thinking...\n\n"
        try:
            # --- Language Detection ---
            input_language = 'en' # Default to English
            try:
                if len(question) > 20: # Langdetect is more reliable with more text
                    input_language = detect(question)
                print(f"DEBUG: Detected input language: {input_language}")
            except Exception as e:
                print(f"WARNING: Language detection failed: {e}. Defaulting to 'en'.")

            # --- Query Translation for Embedding ---
            # Translate non-English queries to English for better embedding/retrieval
            question_for_embedding = question
            if input_language != 'en':
                question_for_embedding = llm_service._translate_to_english(question)
                print(f"DEBUG: Translated query for embedding: '{question_for_embedding[:50]}...'")

            question_embedding = await embedding_generator.get_embedding(question_for_embedding)

            # --- Initial Knowledge Base Check ---
            yield "Checking existing knowledge base...\n\n"
            initial_query_results = pinecone_manager.query_index(question_embedding)
            context = await llm_service.retrieve_and_rerank_context(question_for_embedding, initial_query_results)

            # --- Conditional Data Ingestion ---
            if "No relevant market data found" in context or not initial_query_results.matches:
                yield "No sufficient data found. Ingesting new market data from the web...\n\n"

                # Use the original, untranslated question for web search
                stored_count = await llm_service._ingest_and_upsert_data(question)
                yield f"Ingested {stored_count} relevant web pages.\n\n"

                # Wait for Pinecone to index the new data
                await pinecone_manager.wait_for_indexing(stored_count, PINECONE_MAX_WAIT_TIME, PINECONE_CHECK_INTERVAL)

                # Re-query Pinecone to get the newly added context
                final_query_results = pinecone_manager.query_index(question_embedding)
                context = await llm_service.retrieve_and_rerank_context(question_for_embedding, final_query_results)
                yield "Using newly ingested data to answer your question...\n\n"
            else:
                yield "Found relevant data in knowledge base. Answering your question...\n\n"

            # --- Generate and Stream Final Response ---
            # Pass the original question and detected language to the LLM
            async for chunk in llm_service.generate_response_stream(question, context, input_language):
                yield chunk

        except HTTPException as e:
            error_message = f"Error during operation: {e.detail}\n"
            yield error_message
            print(f"ERROR: HTTPException during stream: {e.detail}")
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}\n"
            yield error_message
            print(f"ERROR: Unexpected error during stream: {str(e)}")

    return StreamingResponse(combined_stream_generator(), media_type="text/plain")

@market_data_router.post("/clear_index")
async def clear_pinecone_index_endpoint():
    """
    Endpoint to clear all vectors from the Pinecone index for this module.
    This is a utility endpoint for starting fresh or managing data.
    """
    if not pinecone_manager:
        raise HTTPException(status_code=503, detail="Pinecone manager not initialized.")

    try:
        pinecone_manager.clear_index()
        return {"message": f"Successfully cleared all vectors from Pinecone index '{PINECONE_INDEX_NAME}'."}
    except Exception as e:
        print(f"ERROR: Failed to clear Pinecone index: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear Pinecone index: {str(e)}"
        )
