from sentence_transformers import CrossEncoder
from datetime import datetime, timezone
import numpy as np

class DashboardScoringService:
    """
    Handles retrieving initial results from the vector store and then
    re-ranking them using a multi-factor scoring model that includes
    context-specific relevance and time decay.
    """
    def __init__(self, vector_store, cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the scoring service with a vector store and a Cross-Encoder model.
        """
        print("Initializing DashboardScoringService with Cross-Encoder...")
        self.vector_store = vector_store
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        # Adjusting weights to prioritize recency
        self.w_relevance = 0.5
        self.w_time_decay = 0.5
        print("Scoring service initialized with updated recency-focused weights.")

    def _get_human_readable_age(self, page_age_str: str) -> tuple[float, str]:
        """
        Parses an ISO 8601 timestamp and returns the age in days (float)
        and a human-readable string (e.g., '3 hours ago').
        """
        if not page_age_str or not isinstance(page_age_str, str):
            return 1.0, "1 day ago"

        try:
            if page_age_str.endswith('Z'):
                page_date = datetime.fromisoformat(page_age_str[:-1]).replace(tzinfo=timezone.utc)
            else:
                page_date = datetime.fromisoformat(page_age_str).replace(tzinfo=timezone.utc)

            now_utc = datetime.now(timezone.utc)
            age_delta = now_utc - page_date
            
            seconds = age_delta.total_seconds()
            days = age_delta.days
            hours = seconds / 3600
            minutes = seconds / 60

            if days > 0 or hours > 18:
                age_str = f"{days if days > 0 else 1} day{'s' if days > 1 else ''} ago"
            elif hours >= 1:
                age_str = f"{int(hours)} hour{'s' if hours > 1 else ''} ago"
            elif minutes >= 1:
                age_str = f"{int(minutes)} minute{'s' if minutes > 1 else ''} ago"
            else:
                age_str = "Just now"
                
            return age_delta.total_seconds() / (24 * 3600), age_str

        except (ValueError, TypeError):
            print(f"Warning: Could not parse timestamp '{page_age_str}'.")
            return 1.0, "1 day ago"

    def _calculate_time_decay_score(self, age_in_days, half_life=0.5):
        """
        Calculates a time decay score using an exponential decay function.
        - half_life is now 0.5 days (12 hours) to strongly favor very recent news.
        """
        decay_rate = np.log(2) / half_life
        return np.exp(-decay_rate * age_in_days)

    def get_enhanced_context(self, query, k=5):
        """
        Queries the vector store, re-ranks results, and returns the top k 
        full document objects including a human-readable age.
        """
        initial_results = self.vector_store.query(query, n_results=20)
        
        if not initial_results or not initial_results.get('documents') or not initial_results['documents'][0]:
            print("No initial results found from vector store.")
            return []
        
        initial_docs = initial_results['documents'][0]
        initial_metadatas = initial_results['metadatas'][0]
        
        print(f"Re-ranking {len(initial_docs)} initial results for query: '{query}'")
        model_input = [[query, doc] for doc in initial_docs]
        relevance_scores = self.cross_encoder.predict(model_input, show_progress_bar=False)
        
        scored_results = []
        for i, doc in enumerate(initial_docs):
            relevance_score = relevance_scores[i]
            metadata = initial_metadatas[i]
            
            age_in_days, human_readable_age = self._get_human_readable_age(metadata.get("page_age"))
            time_decay_score = self._calculate_time_decay_score(age_in_days)
            
            final_score = (self.w_relevance * relevance_score) + (self.w_time_decay * time_decay_score)
            
            # Add the new 'age' field to the metadata
            metadata['age'] = human_readable_age
            
            scored_results.append({
                "text": doc,
                "metadata": metadata,
                "final_score": final_score,
            })
        
        ranked_results = sorted(scored_results, key=lambda x: x['final_score'], reverse=True)
        
        final_context_docs = ranked_results[:k]
        
        print(f"Successfully generated enhanced context with {len(final_context_docs)} documents.")
        if ranked_results:
            top_result = ranked_results[0]
            print(f"Top result score: {top_result['final_score']:.4f} (Age: {top_result['metadata'].get('age')}, URL: {top_result['metadata'].get('url')})")
            
        return final_context_docs