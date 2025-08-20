from sentence_transformers import CrossEncoder

class DashboardScoringService:
    """
    Handles retrieving initial results from the vector store and then
    re-ranking them using a more powerful Cross-Encoder model to find
    the most relevant context for the LLM.
    """
    def __init__(self, vector_store, cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the scoring service with a vector store and a Cross-Encoder model.
        """
        print("Initializing DashboardScoringService with Cross-Encoder...")
        self.vector_store = vector_store
        # The Cross-Encoder is better for re-ranking than a standard sentence transformer
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        print("Scoring service initialized.")

    def get_enhanced_context(self, query, k=5):
        """
        Queries the vector store, re-ranks the results, and returns the top k
        most relevant chunks as the enhanced context.
        
        Args:
            query (str): The query to find relevant context for.
            k (int): The final number of top chunks to return after re-ranking.
            
        Returns:
            list: A list of the top k most relevant text chunks.
        """
        # Step 1: Get initial broad search results from ChromaDB
        # We fetch more results than we need (e.g., 20) to give the re-ranker a good pool to choose from.
        initial_results = self.vector_store.query(query, n_results=20)
        
        if not initial_results or not initial_results.get('documents') or not initial_results['documents'][0]:
            print("No initial results found from vector store.")
            return []
        
        initial_docs = initial_results['documents'][0]
        
        # Step 2: Use the Cross-Encoder to re-rank the initial results
        print(f"Re-ranking {len(initial_docs)} initial results...")
        # The Cross-Encoder takes pairs of [query, document]
        model_input = [[query, doc] for doc in initial_docs]
        scores = self.cross_encoder.predict(model_input)
        
        # Step 3: Combine documents with their new scores and sort
        ranked_results = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        
        # Step 4: Extract the top k documents to form the final context
        final_context = [doc for score, doc in ranked_results[:k]]
        
        print(f"Successfully generated enhanced context with {len(final_context)} chunks.")
        return final_context
