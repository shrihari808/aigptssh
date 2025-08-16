import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from streaming.yt_stream import yt_chat


class YoutubeQuery(BaseModel):
    query: str
    session_id: str

async def youtube_rag_endpoint(request: YoutubeQuery):
    """
    Endpoint to trigger the YouTube RAG pipeline.
    """
    try:
        result = await yt_chat(request.query, request.session_id)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to get a response from the YouTube RAG pipeline.")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))