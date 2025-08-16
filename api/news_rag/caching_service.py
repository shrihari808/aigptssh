# /aigptssh/api/news_rag/caching_service.py

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import chroma_server_client, embeddings

def get_session_cache(session_id: str) -> Chroma:
    """
    Gets or creates a ChromaDB collection for a specific session, ensuring it
    uses the cosine distance metric for accurate similarity scoring.

    Args:
        session_id (str): The unique identifier for the user session.

    Returns:
        Chroma: A Chroma vector store instance for the session.
    """
    collection_name = f"web_rag_cache_{session_id}"
    vector_store = Chroma(
        client=chroma_server_client,
        collection_name=collection_name,
        embedding_function=embeddings,
        # --- ADD THIS SNIPPET ---
        # This explicitly sets the distance function for the collection to cosine.
        # It ensures that the similarity scores are calculated correctly (lower is better, 0=identical).
        collection_metadata={"hnsw:space": "cosine"}
        # --- END OF SNIPPET ---
    )
    return vector_store

def add_passages_to_cache(session_id: str, passages: list[dict]):
    """
    Adds scraped and processed passages to the session's cache.

    Args:
        session_id (str): The session identifier.
        passages (list[dict]): A list of passage dictionaries to add.
    """
    if not passages:
        return

    session_cache = get_session_cache(session_id)
    
    # Convert passage dictionaries to LangChain Document objects
    documents_to_add = []
    for i, passage in enumerate(passages):
        doc = Document(
            page_content=passage.get('text', ''),
            metadata=passage.get('metadata', {})
        )
        documents_to_add.append(doc)

    if documents_to_add:
        session_cache.add_documents(documents_to_add)
        print(f"DEBUG: Added {len(documents_to_add)} passages to cache for session {session_id}.")

def query_session_cache(session_id: str, query: str, k: int = 15) -> list[tuple[Document, float]]:
    """
    Queries the session's cache for relevant documents.

    Args:
        session_id (str): The session identifier.
        query (str): The user's query.
        k (int): The number of documents to retrieve.

    Returns:
        list[tuple[Document, float]]: A list of (document, score) tuples.
    """
    try:
        session_cache = get_session_cache(session_id)
        # Check if the collection exists and has documents
        if session_cache._collection.count() == 0:
            print(f"DEBUG: Cache for session {session_id} is empty.")
            return []
            
        results = session_cache.similarity_search_with_score(query, k=k)
        print(f"DEBUG: Found {len(results)} relevant documents in cache for session {session_id}.")
        return results
    except Exception as e:
        # This can happen if the collection doesn't exist yet.
        print(f"DEBUG: Could not query cache for session {session_id}. It might be new. Error: {e}")
        return []

