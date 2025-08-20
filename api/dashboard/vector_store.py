import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

class DashboardVectorStore:
    """
    Manages the ChromaDB vector store for the dashboard, including document
    chunking, embedding, and storage.
    """
    def __init__(self, collection_name="dashboard_news_content"):
        """
        Initializes the vector store and the ChromaDB collection.
        """
        print("Initializing DashboardVectorStore...")
        # Using an in-memory ephemeral client for simplicity during development.
        # For persistence, you can switch to chromadb.PersistentClient(path="/path/to/db")
        self.client = chromadb.Client()
        
        # Using the default SentenceTransformer embedding function
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

    def add_documents(self, documents):
        """
        Chunks, embeds, and adds a list of scraped documents to the ChromaDB collection.
        
        Args:
            documents (list of dicts): A list of documents from the scraper,
                                       each with 'url' and 'content' keys.
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            content = doc.get("content")
            url = doc.get("url")
            if content and url:
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append({"source": url})
                    # Create a unique, deterministic ID for each chunk
                    chunk_id = hashlib.sha256(f"{url}-{i}".encode()).hexdigest()
                    all_ids.append(chunk_id)
        
        if not all_chunks:
            print("No content to add to the vector store.")
            return

        print(f"Adding {len(all_chunks)} chunks to the vector store...")
        # ChromaDB's add method handles embedding automatically via the collection's embedding function
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        print("Successfully added documents to the vector store.")

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
