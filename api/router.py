# /aigptcur/app_service/api/router.py

from fastapi import APIRouter

# --- Import all the individual router objects from your application ---

# Imports from subdirectories of 'api'
from . import chatbot
from .market_content import chatwithfiles, youtube_sum
from . import graph_openai1
from .fundamentals_rag import fundamental_chat2, corp
from .market_data_lm import router as market_data_router_module

# --- CORRECTED IMPORT ---
# Import from the 'streaming' directory, which is at the same level as 'api'
# This is now an absolute import because the project root is in the Python path.
from streaming import streaming

# Create a single master router
api_router = APIRouter()

# --- Include all the individual routers into the master router ---
# This provides a clean, single point of registration in main.py
api_router.include_router(chatbot.router, tags=["Chatbot"])
api_router.include_router(chatwithfiles.router, tags=["Market Content"])
api_router.include_router(youtube_sum.router, tags=["Market Content"])
api_router.include_router(graph_openai1.router, tags=["Graphing"])
api_router.include_router(streaming.cmots_rag, tags=["Streaming RAG"])
api_router.include_router(streaming.web_rag, tags=["Streaming RAG"])
api_router.include_router(streaming.red_rag, tags=["Streaming RAG"])
api_router.include_router(streaming.yt_rag, tags=["Streaming RAG"])
api_router.include_router(fundamental_chat2.fund_rag, tags=["Fundamentals RAG"])
api_router.include_router(corp.corp_rag, tags=["Fundamentals RAG"])
api_router.include_router(market_data_router_module.market_data_router) # This router already has tags defined
