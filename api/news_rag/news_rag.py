import os
import time
from pinecone import Pinecone
from app_service.api.news_rag.brave_news import BraveNews
from langchain_community.vectorstores import Pinecone as PineconeVectors
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()


class NewsRag:
    def __init__(self, query: str):
        """
        Initializes the NewsRag object.
        Args:
            query (str): The user's query.
        """
        self.query = query
        self.llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.pinecone_index_name = "market-data-index"

        # Initialize Pinecone client using v3 style
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables must be set")
        
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        self.index = self.pinecone_client.Index(self.pinecone_index_name)

        self.urls = self._get_urls()
        self.docs = self._load_documents()
        self.splits = self._split_documents()
        self.retriever = self._get_retriever()

    def _get_urls(self):
        """
        Gets URLs from Brave search based on the query.
        Returns:
            list: A list of URLs.
        """
        brave_news = BraveNews(self.query)
        return brave_news.get_urls()

    def _load_documents(self):
        """
        Loads documents from the URLs.
        Returns:
            list: A list of loaded documents.
        """
        all_docs = []
        for url in self.urls:
            try:
                loader = RecursiveUrlLoader(url=url, max_depth=2)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading url {url}: {e}")
        return all_docs

    def _split_documents(self):
        """
        Splits the documents into smaller chunks.
        Returns:
            list: A list of document splits.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(self.docs)

    def _get_retriever(self):
        """
        Creates a retriever from the document splits using Pinecone.
        It upserts the documents to the Pinecone index and waits for them to be indexed.
        Returns:
            A Pinecone retriever object.
        """
        print(f"Number of splits to be upserted: {len(self.splits)}")
        if not self.splits:
            # Return a retriever from an empty list of docs if no splits
            return PineconeVectors.from_documents([], self.embedding_model, index_name=self.pinecone_index_name).as_retriever()

        # 1. Get the vector count before upserting
        try:
            initial_vector_count = self.index.describe_index_stats()['total_vector_count']
        except Exception as e:
            print(f"Error describing index stats: {e}")
            initial_vector_count = 0
        
        print(f"Initial vector count: {initial_vector_count}")

        # 2. Upsert documents to Pinecone
        # Langchain's Pinecone vector store will use the environment variables for its own client instance
        vectorstore = PineconeVectors.from_documents(
            documents=self.splits,
            embedding=self.embedding_model,
            index_name=self.pinecone_index_name
        )
        
        print("Upsert operation completed.")

        # 3. Wait for the vectors to be indexed
        expected_count = initial_vector_count + len(self.splits)
        timeout = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                current_vector_count = self.index.describe_index_stats()['total_vector_count']
                print(f"Current vector count: {current_vector_count}, Expected: ~{expected_count}")
                if current_vector_count >= expected_count:
                    print("Vectors are indexed.")
                    break
            except Exception as e:
                print(f"Could not get vector count, retrying... Error: {e}")
            time.sleep(5) # Check every 5 seconds
        else:
            print("Timeout reached while waiting for vectors to be indexed.")

        return vectorstore.as_retriever()

    def get_rag_chain(self):
        """
        Creates and returns the RAG chain.
        Returns:
            A RAG chain object.
        """
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        return rag_chain


def web_rag(query: str):
    """
    The main function for the web RAG.
    Args:
        query (str): The user's query.
    Returns:
        The response from the RAG chain.
    """
    news_rag_instance = NewsRag(query)
    rag_chain = news_rag_instance.get_rag_chain()
    result = rag_chain.invoke(query)
    return result

def cmots_only(query: str):
    """
    Placeholder function to resolve import error.
    """
    print("cmots_only function is not implemented.")
    return "This feature is not available."
