# services/embedding_generator.py

from openai import AsyncOpenAI
from fastapi import HTTPException

# Import constants from the config file
from config import OPENAI_EMBEDDING_MODEL

class EmbeddingGenerator:
    """Handles generation of embeddings using OpenAI's API."""
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for EmbeddingGenerator.")
        self.client = AsyncOpenAI(api_key=openai_api_key)

    async def get_embedding(self, text: str) -> list[float]:
        """Generates an embedding vector for a given text."""
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=OPENAI_EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get embedding from OpenAI: {str(e)}"
            )
