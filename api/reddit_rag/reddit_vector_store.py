# /aigptssh/api/reddit_rag/reddit_vector_store.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import embeddings

def create_reddit_vector_store_from_scraped_data(scraped_data: list[dict]):
    """
    Chunks scraped Reddit data and creates an in-memory vector store.

    Args:
        scraped_data (list[dict]): A list of dictionaries, where each dict
                                  contains scraped data from a Reddit post.

    Returns:
        A Chroma vector store retriever instance, or None if no chunks are created.
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for post in scraped_data:
        if not post:
            continue

        # Combine title, selftext, and comments into a single document for chunking
        full_text = f"Title: {post.get('title', '')}\n\n{post.get('selftext', '')}"
        comments_text = "\n".join([f"Comment by {c.get('author', 'N/A')}: {c.get('body', '')}" for c in post.get('comments', [])])
        full_text += f"\n\nComments:\n{comments_text}"


        if full_text:
            chunks = text_splitter.split_text(full_text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "title": post.get('title'),
                    "score": post.get('score'),
                    "chunk_num": i + 1,
                }

                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                all_chunks.append(doc)

    if not all_chunks:
        print("WARNING: No chunks were created from the scraped Reddit data.")
        return None

    print(f"DEBUG: Created {len(all_chunks)} chunks from {len(scraped_data)} Reddit posts.")

    # Create an in-memory vector store from the chunks
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

    return vector_store.as_retriever(search_kwargs={"k": 20})