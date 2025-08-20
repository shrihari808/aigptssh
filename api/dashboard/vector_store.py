import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

class DashboardVectorStore:
    """
    Manages the ChromaDB vector store for the dashboard, including document
    chunking, embedding, and storage, while preserving rich metadata.
    """
    def __init__(self, collection_name="dashboard_news_content"):
        """
        Initializes the vector store and the ChromaDB collection.
        """
        print("Initializing DashboardVectorStore...")
        self.client = chromadb.Client()
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        print(f"Vector store initialized. Collection '{collection_name}' is ready.")

    def add_documents(self, articles):
        """
        Chunks, embeds, and adds a list of scraped articles to the ChromaDB collection in batches.
        Each chunk's metadata includes the source URL and page age. Null metadata
        values are replaced with empty strings.
        
        Args:
            articles (list of dicts): A list of articles, each with 'url', 'page_age',
                                      'title', and 'content' keys.
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for article in articles:
            content = article.get("content")
            url = article.get("url")
            
            if content and url:
                base_metadata = {key: value for key, value in article.items() if key != "content"}
                
                for key, value in base_metadata.items():
                    if value is None:
                        base_metadata[key] = ""

                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append(base_metadata.copy())
                    chunk_id = hashlib.sha256(f"{url}-{i}".encode()).hexdigest()
                    all_ids.append(chunk_id)
        
        if not all_chunks:
            print("No content to add to the vector store.")
            return

        # Add documents in batches to avoid exceeding the maximum batch size
        batch_size = 166 
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            
            print(f"Adding batch {i//batch_size + 1} with {len(batch_chunks)} chunks to the vector store...")
            self.collection.add(
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
        print("Successfully added all documents to the vector store.")

    def query(self, query_text, n_results=20):
        """
        Queries the vector store to find the most relevant document chunks.
        """
        print(f"Querying vector store for: '{query_text}'")
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results