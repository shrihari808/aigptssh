# /aigptssh/api/youtube_rag/youtube_vector_store.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import embeddings

def create_yt_vector_store_from_transcripts(video_transcripts: list[dict]):
    """
    Chunks video transcripts and creates an in-memory vector store.

    Args:
        video_transcripts (list[dict]): A list of dictionaries, where each dict
                                        contains video metadata and the full transcript text.

    Returns:
        A Chroma vector store retriever instance, or None if no chunks are created.
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for video in video_transcripts:
        transcript_text = video.get("transcript")
        metadata = video.get("metadata", {})

        if transcript_text:
            chunks = text_splitter.split_text(transcript_text)
            for i, chunk in enumerate(chunks):
                    # Create a copy of the original video's metadata to ensure all fields are preserved
                    chunk_metadata = metadata.copy()
                    
                    # Add chunk-specific info
                    chunk_metadata["chunk_num"] = i + 1
                    
                    # Standardize the URL key to 'link' for compatibility with the scoring service
                    if 'url' in chunk_metadata and 'link' not in chunk_metadata:
                        chunk_metadata['link'] = chunk_metadata['url']

                    doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    all_chunks.append(doc)

    if not all_chunks:
        print("WARNING: No chunks were created from the transcripts.")
        return None

    print(f"DEBUG: Created {len(all_chunks)} chunks from {len(video_transcripts)} transcripts.")

    # Create an in-memory vector store from the chunks using the configured embedding model
    vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

    return vector_store.as_retriever(search_kwargs={"k": 20})
