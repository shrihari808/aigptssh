# /aigptcur/app_service/api/market_data_lm/services/pinecone_manager.py

import asyncio
import time
from fastapi import HTTPException

# --- CORRECTED IMPORT: Use ServerlessSpec for modern Pinecone indexes ---
from pinecone import Pinecone, ServerlessSpec

# The dimension for OpenAI's text-embedding-ada-002 model
PINECONE_INDEX_DIMENSION = 1536

class PineconeManager:
    """Handles interactions with the Pinecone vector database."""
    def __init__(self, api_key: str, environment: str, index_name: str):
        if not api_key or not environment or not index_name:
            raise ValueError("Pinecone API key, environment, and index name are required.")
        self.pinecone_client = Pinecone(api_key=api_key)
        # The 'environment' variable now represents the cloud region (e.g., "us-east-1")
        self.cloud_region = environment
        self.index_name = index_name
        self.pinecone_index = None

    def connect_to_index(self):
        """
        Connects to the specified Pinecone index.
        If the index does not exist, it will be created automatically.
        """
        try:
            # Check if the index already exists
            if self.index_name not in self.pinecone_client.list_indexes().names():
                print(f"INFO:     Pinecone index '{self.index_name}' not found. Creating it as a serverless index...")
                # If it does not exist, create it using ServerlessSpec
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=PINECONE_INDEX_DIMENSION,
                    metric="cosine",
                    # --- CORRECTED: Use ServerlessSpec to match your account type ---
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.cloud_region
                    )
                )
                print(f"INFO:     Waiting for '{self.index_name}' to initialize...")
                time.sleep(10)
                print(f"INFO:     Index '{self.index_name}' created successfully.")
            else:
                print(f"INFO:     Found existing Pinecone index '{self.index_name}'.")

            # Connect to the (now existing) index
            self.pinecone_index = self.pinecone_client.Index(self.index_name)
            print(f"DEBUG:    Successfully connected to Pinecone index '{self.index_name}'.")
            print(f"DEBUG:    Index description: {self.pinecone_index.describe_index_stats()}")

        except Exception as e:
            error_detail = f"Failed to connect to or create Pinecone index '{self.index_name}': {str(e)}"
            raise HTTPException(status_code=500, detail=error_detail)

    def upsert_vectors(self, vectors_data: list[tuple]) -> int:
        """Upserts a list of vectors into the Pinecone index."""
        if not self.pinecone_index:
            raise HTTPException(status_code=500, detail="Pinecone index not initialized.")
        if not vectors_data:
            return 0
        print(f"DEBUG:    Upserting {len(vectors_data)} vectors to Pinecone...")
        self.pinecone_index.upsert(vectors=vectors_data)
        print(f"DEBUG:    Successfully upserted {len(vectors_data)} vectors.")
        return len(vectors_data)

    def query_index(self, vector: list[float], top_k: int = 10, include_metadata: bool = True):
        """Queries the Pinecone index for relevant vectors."""
        if not self.pinecone_index:
            raise HTTPException(status_code=500, detail="Pinecone index not initialized.")
        return self.pinecone_index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)

    def clear_index(self):
        """Clears all vectors from the Pinecone index."""
        if not self.pinecone_index:
            raise HTTPException(status_code=500, detail="Pinecone index not initialized.")
        print(f"DEBUG:    Clearing all vectors from Pinecone index '{self.index_name}'...")
        self.pinecone_index.delete(delete_all=True)
        print(f"DEBUG:    Successfully cleared all vectors from Pinecone index '{self.index_name}'.")

    async def wait_for_indexing(self, expected_count: int, max_wait_time: int, check_interval: int):
        """Waits for Pinecone index to reflect the upserted vectors."""
        if not self.pinecone_index:
            return False
        current_wait_time = 0
        while current_wait_time < max_wait_time:
            try:
                stats = self.pinecone_index.describe_index_stats()
                total_vector_count = stats.get('total_vector_count', 0)
                if 'namespaces' in stats and stats['namespaces']:
                     total_vector_count = sum(ns.get('vector_count', 0) for ns in stats['namespaces'].values())
                print(f"DEBUG:    Waiting for Pinecone index. Current vector count: {total_vector_count}. Waited: {current_wait_time}s")
                if total_vector_count > 0 and expected_count > 0:
                    print("INFO:     Pinecone index appears ready with new vectors.")
                    return True
            except Exception as e:
                print(f"WARNING:  Could not describe index stats while waiting: {e}")
            await asyncio.sleep(check_interval)
            current_wait_time += check_interval
        print(f"WARNING:  Pinecone index did not reflect upserted vectors after {max_wait_time} seconds. Proceeding anyway.")
        return False
