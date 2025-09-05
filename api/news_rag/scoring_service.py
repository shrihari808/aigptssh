import os
import numpy as np
import torch
from datetime import datetime
from urllib.parse import urlparse
from transformers import pipeline
from sentence_transformers import CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect, DetectorFactory
import asyncio

from config import (
    W_RELEVANCE,
    W_SENTIMENT,
    W_TIME_DECAY,
    W_IMPACT,
    SOURCE_CREDIBILITY_WEIGHTS,
    IMPACT_KEYWORDS,
    MAX_RERANKED_CONTEXT_ITEMS,
    CONTEXT_SUFFICIENCY_THRESHOLD,
    MIN_CONTEXT_LENGTH,
    MIN_RELEVANT_DOCS
)

# Set seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

class NewsRagScoringService:
    """Advanced scoring service for news RAG with sentiment, time decay, and impact analysis."""
    
    def __init__(self):
        # Determine device for model execution
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"DEBUG: ScoringService device set to {'cuda' if self.device == 0 else 'cpu'}")
        
        # Initialize FinBERT sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert", 
                device=self.device
            )
            print("DEBUG: FinBERT sentiment analyzer initialized successfully")
        except Exception as e:
            print(f"WARNING: Could not initialize FinBERT sentiment analyzer: {e}")
            self.sentiment_analyzer = None
        
        # Initialize CrossEncoder for relevance scoring
        try:
            self.cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            if self.device == 0:
                self.cross_encoder_model.model.to('cuda')
                print("DEBUG: Cross-encoder model moved to CUDA")
            print("DEBUG: CrossEncoder model initialized successfully")
        except Exception as e:
            print(f"WARNING: Could not initialize CrossEncoder model: {e}")
            self.cross_encoder_model = None

    async def rerank_sources_by_snippet(self, query: str, sources: list[dict], top_n: int = 10) -> list[dict]:
        """
        NEW: Reranks a list of sources based on the relevance of their snippets to the query.
        This is the 'First Re-ranking' step in our new strategy.
        
        Args:
            query (str): The user's query.
            sources (list[dict]): The list of source metadata dictionaries from the broad search.
            top_n (int): The number of top sources to return.
            
        Returns:
            list[dict]: The top_n sources, sorted by relevance score.
        """
        if not sources or not self.cross_encoder_model:
            print("WARNING: No sources or cross-encoder model available for reranking. Returning original sources.")
            return sources[:top_n]

        # Create pairs of [query, snippet] for the cross-encoder
        snippets = [
            f"{source.get('title', '')} - {source.get('snippet', '')}" 
            for source in sources
        ]
        query_snippet_pairs = [[query, snippet] for snippet in snippets]

        print(f"DEBUG: Reranking {len(sources)} snippets against the query...")
        
        # Get relevance scores from the model
        # The predict method can be run in a thread to avoid blocking the event loop
        scores = await asyncio.to_thread(self.cross_encoder_model.predict, query_snippet_pairs)

        # Add scores to each source and sort
        for source, score in zip(sources, scores):
            source['relevance_score'] = float(score)
            
        reranked_sources = sorted(sources, key=lambda x: x['relevance_score'], reverse=True)

        print(f"DEBUG: Reranking complete. Top source score: {reranked_sources[0]['relevance_score']:.4f}")
        
        return reranked_sources[:top_n]

    async def rerank_content_chunks(self, query: str, sources: list[dict], top_n: int = 7) -> list[dict]:
        """
        NEW: Chunks the full content of sources, scores them using a multi-factor approach,
        and reranks them for relevance. This is the 'Second Re-ranking' step.

        Args:
            query (str): The user's query.
            sources (list[dict]): The list of source dictionaries with 'full_webpage_content'.
            top_n (int): The number of top text chunks to return.

        Returns:
            list[dict]: A new list of the top_n passage dictionaries, ready for the final context.
        """
        if not sources or not self.cross_encoder_model:
            return []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_chunks = []

        # 1. Chunk the content from all sources
        for source in sources:
            content = source.get('full_webpage_content')
            if content and len(content) > 100:  # Process only if content is substantial
                chunks = text_splitter.split_text(content)
                for i, chunk_text in enumerate(chunks):
                    # Keep a link back to the original source metadata
                    all_chunks.append({
                        'text': chunk_text,
                        'metadata': source,  # The entire original source dict is the metadata
                        'relevance_score': 0.0, # Initialize scores
                        'sentiment_score': 0.0,
                        'time_decay_score': 0.0,
                        'impact_score': 0.0,
                        'final_combined_score': 0.0
                    })

        if not all_chunks:
            print("WARNING: No content chunks were generated after scraping.")
            return []

        # 2. Score all chunks for relevance using the cross-encoder
        print(f"DEBUG: Calculating relevance for {len(all_chunks)} content chunks from {len(sources)} sources...")
        query_chunk_pairs = [[query, chunk['text']] for chunk in all_chunks]
        relevance_scores = await asyncio.to_thread(self.cross_encoder_model.predict, query_chunk_pairs)
        for chunk, rel_score in zip(all_chunks, relevance_scores):
            chunk['relevance_score'] = float(rel_score)

        # 3. Calculate other fast scores and the final combined score for each chunk
        print(f"DEBUG: Calculating remaining multi-factor scores (time, impact)...")
        for chunk in all_chunks:
            metadata = chunk.get("metadata", {})
            text = chunk.get("text", "")
            
            # Calculate remaining fast scores
            chunk['time_decay_score'] = self._calculate_time_decay_score(
                metadata.get("publication_date") or str(metadata.get("date", "")),
                query
            )
            chunk['impact_score'] = self._calculate_impact_score(text, metadata.get("link"))

            # Calculate the final weighted score using all computed components
            chunk['final_combined_score'] = (
                W_RELEVANCE * chunk.get('relevance_score', 0.0) +
                W_TIME_DECAY * chunk['time_decay_score'] +
                W_IMPACT * chunk['impact_score']
            )

        # 4. Sort all chunks by the final combined score
        reranked_chunks = sorted(all_chunks, key=lambda x: x['final_combined_score'], reverse=True)

        # --- THIS IS THE DEBUG LOG YOU REQUESTED ---
        print(f"DEBUG: Top {min(5, len(reranked_chunks))} passages after reranking:")
        for i, passage in enumerate(reranked_chunks[:5]):
            print(f"  {i+1}. Score: {passage['final_combined_score']:.4f} | "
                  f"Rel: {passage['relevance_score']:.2f}, "
                  f"Time: {passage['time_decay_score']:.2f}, Impact: {passage['impact_score']:.2f} | "
                  f"{passage.get('metadata', {}).get('link', 'No link')}")

        return reranked_chunks[:top_n]

    def assess_context_sufficiency(self, query: str, retrieved_docs: list) -> float:
        """
        SIMPLIFIED: Assess if retrieved documents are sufficient based on cosine similarity.
        
        Args:
            query (str): The user's query.
            retrieved_docs (list): List of tuples (Document, cosine_distance_score)
            
        Returns:
            float: Sufficiency score between 0 and 1
        """
        if not retrieved_docs or len(retrieved_docs) < MIN_RELEVANT_DOCS:
            print(f"DEBUG: Too few documents ({len(retrieved_docs)}), insufficient.")
            return 0.0

        # Pinecone returns cosine distance, convert to similarity
        cosine_distances = [score for doc, score in retrieved_docs]
        cosine_similarities = [1 - distance for distance in cosine_distances]
        
        average_similarity = sum(cosine_similarities) / len(cosine_similarities)
        
        # Count highly relevant documents (similarity > 0.7 means distance < 0.3)
        SIMILARITY_THRESHOLD = 0.7
        highly_relevant_count = sum(1 for sim in cosine_similarities if sim > SIMILARITY_THRESHOLD)
        
        print(f"DEBUG: Cosine distances: {[f'{d:.3f}' for d in cosine_distances[:5]]}")
        print(f"DEBUG: Cosine similarities: {[f'{s:.3f}' for s in cosine_similarities[:5]]}")
        print(f"DEBUG: Average similarity: {average_similarity:.3f}")
        print(f"DEBUG: Highly relevant docs (>{SIMILARITY_THRESHOLD}): {highly_relevant_count}")
        
        # Simple scoring based on average similarity and count of relevant docs
        base_score = min(0.8, average_similarity)  # Cap at 0.8
        
        # Bonus for having multiple relevant documents
        relevance_bonus = min(0.2, highly_relevant_count * 0.05)  # Up to 0.2 bonus
        
        final_score = base_score + relevance_bonus
        
        print(f"DEBUG: Base score: {base_score:.3f}, Relevance bonus: {relevance_bonus:.3f}")
        print(f"DEBUG: Final sufficiency score: {final_score:.3f}")
        
        return final_score

    # ALSO UPDATE: Adjust the threshold in your web_rag function
    # Change this line in web_rag_mix():
    # if sufficiency_score < 0.7:  # OLD threshold
    # to:
    # if sufficiency_score < 0.4:  # NEW, more realistic threshold

    # Additional debugging function for your web_rag endpoint
    def debug_pinecone_similarity_search(vs, query, k=15):
        """
        Debug helper to understand your Pinecone similarity search results
        """
        print(f"\n=== DEBUGGING PINECONE SEARCH ===")
        print(f"Query: '{query}'")
        
        results = vs.similarity_search_with_score(query, k=k)
        
        print(f"Retrieved {len(results)} documents")
        
        for i, (doc, score) in enumerate(results[:5]):
            print(f"\nResult {i+1}:")
            print(f"  Distance score: {score:.4f}")
            print(f"  Similarity: {(1-score):.4f}")
            print(f"  Title: {doc.metadata.get('title', 'No title')[:80]}...")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Date: {doc.metadata.get('publication_date', 'No date')}")
            print(f"  Content preview: {doc.page_content[:100]}...")
        
        return results

        # Integration example for your web_rag function:
        """
        # Replace the existing sufficiency check in web_rag_mix() with:

        # Debug the search results first (remove in production)
        if DEBUG_MODE:  # Add this flag to your config
            pinecone_results_with_scores = debug_pinecone_similarity_search(vs, original_query, 15)
        else:
            pinecone_results_with_scores = vs.similarity_search_with_score(original_query, k=15)

        # Use the fixed sufficiency assessment
        sufficiency_score = scoring_service.assess_context_sufficiency(original_query, pinecone_results_with_scores)

        # Use lower, more realistic threshold
        if sufficiency_score < 0.4:  # Changed from 0.7
            print(f"DEBUG: Sufficiency score {sufficiency_score:.2f} is below threshold. Triggering Brave search.")
            # ... rest of Brave search logic
        else:
            print(f"DEBUG: Sufficiency score {sufficiency_score:.2f} is sufficient. Using existing data.")
        """




    def _calculate_sentiment_score(self, text: str, query: str) -> float:
        """Context-aware sentiment scoring that considers query intent."""
        if not self.sentiment_analyzer:
            return 0.5  # Neutral score if analyzer not available
        
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            sentiment, confidence = result['label'], result['score']
            
            query_lower = query.lower()
            seeking_risks = any(word in query_lower for word in [
                'risk', 'problem', 'issue', 'concern', 'decline', 'fall', 'loss', 
                'negative', 'bad', 'warning', 'alert'
            ])
            seeking_opportunities = any(word in query_lower for word in [
                'growth', 'profit', 'gain', 'opportunity', 'rise', 'increase',
                'positive', 'good', 'bullish', 'surge'
            ])
            
            if seeking_risks:
                return 0.5 + (confidence * 0.5) if sentiment == 'negative' else 0.5 - (confidence * 0.3)
            elif seeking_opportunities:
                return 0.5 + (confidence * 0.5) if sentiment == 'positive' else 0.5 - (confidence * 0.3)
            else:
                # Neutral query - slightly favor positive sentiment in financial context
                if sentiment == 'positive':
                    return 0.6 + (confidence * 0.3)
                elif sentiment == 'negative':
                    return 0.4 - (confidence * 0.2)
                else:
                    return 0.5
                    
        except Exception as e:
            print(f"WARNING: Failed to calculate sentiment score: {e}")
            return 0.5

    def _calculate_sentiment_score_from_result(self, result: dict, query: str) -> float:
        """Calculates the final score from a pre-computed sentiment result."""
        if not result:
            return 0.5

        sentiment, confidence = result.get('label'), result.get('score', 0)
        
        query_lower = query.lower()
        seeking_risks = any(word in query_lower for word in [
            'risk', 'problem', 'issue', 'concern', 'decline', 'fall', 'loss', 
            'negative', 'bad', 'warning', 'alert'
        ])
        seeking_opportunities = any(word in query_lower for word in [
            'growth', 'profit', 'gain', 'opportunity', 'rise', 'increase',
            'positive', 'good', 'bullish', 'surge'
        ])
        
        if seeking_risks:
            return 0.5 + (confidence * 0.5) if sentiment == 'negative' else 0.5 - (confidence * 0.3)
        elif seeking_opportunities:
            return 0.5 + (confidence * 0.5) if sentiment == 'positive' else 0.5 - (confidence * 0.3)
        else:
            # Neutral query - slightly favor positive sentiment in financial context
            if sentiment == 'positive':
                return 0.6 + (confidence * 0.3)
            elif sentiment == 'negative':
                return 0.4 - (confidence * 0.2)
            else:
                return 0.5

    def _calculate_time_decay_score(self, publication_date_str: str, query: str) -> float:
        """Query-aware time decay with different decay rates for different content types."""
        if not publication_date_str:
            return 0.4  # Default score for unknown dates
            
        try:
            published_date = None
            date_str_cleaned = str(publication_date_str).strip()

            # Try parsing ISO 8601 format (which Brave API uses)
            try:
                # Handle timezone info like 'Z' or '+00:00'
                published_date = datetime.fromisoformat(date_str_cleaned.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                # Fallback for other common formats if the first fails
                try:
                    # Handle formats like 'YYYY-MM-DD HH:MM:SS' or just 'YYYY-MM-DD'
                    published_date = datetime.strptime(date_str_cleaned.split('T')[0], '%Y-%m-%d')
                except (ValueError, TypeError):
                     # Handle YYYYMMDD integer format
                    if date_str_cleaned.isdigit() and len(date_str_cleaned) == 8:
                        published_date = datetime.strptime(date_str_cleaned, '%Y%m%d')
                    else:
                        print(f"WARNING: Could not parse date string: '{publication_date_str}'")
                        return 0.4 # Return default if all parsing fails
            
            # Ensure the current time is timezone-aware if the parsed date is
            now = datetime.now(published_date.tzinfo)
            age_in_days = (now - published_date).days
            
            if age_in_days < 0: age_in_days = 0 # Handle future dates just in case

            # Adjust decay rate based on query context
            query_lower = query.lower()
            if any(word in query_lower for word in ['latest', 'recent', 'today', 'current']):
                half_life_days = 7
            elif any(word in query_lower for word in ['annual', 'yearly']):
                half_life_days = 180
            else:
                half_life_days = 21
                
            # Exponential decay function
            decay_constant = np.log(2) / half_life_days
            score = np.exp(-decay_constant * age_in_days)
            
            return max(0.1, min(1.0, score))
            
        except Exception as e:
            print(f"WARNING: Failed to calculate time decay score for date '{publication_date_str}': {e}")
            return 0.4

    def _calculate_impact_score(self, text: str, source_link: str) -> float:
        """Estimates impact score based on keyword mentions and source credibility."""
        # Count impact keywords in text
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in IMPACT_KEYWORDS if keyword in text_lower)
        
        # Normalize keyword impact (diminishing returns)
        normalized_keyword_impact = min(1.0, keyword_count / 3.0)
        
        # Get source credibility
        domain = "default"
        if source_link:
            try:
                parsed_url = urlparse(source_link)
                domain = parsed_url.netloc.replace('www.', '')
                # Remove common prefixes
                if domain.startswith('m.'):
                    domain = domain[2:]
            except Exception as e:
                print(f"WARNING: Could not parse domain from link '{source_link}': {e}")

        source_credibility = SOURCE_CREDIBILITY_WEIGHTS.get(domain, SOURCE_CREDIBILITY_WEIGHTS["default"])
        
        # Check for high-impact phrases
        high_impact_phrases = [
            'breaking news', 'exclusive', 'first time', 'record high', 'record low',
            'major announcement', 'significant', 'unprecedented', 'historic'
        ]
        phrase_bonus = 0.2 if any(phrase in text_lower for phrase in high_impact_phrases) else 0
        
        # Combine scores
        impact_score = min(1.0, (normalized_keyword_impact * 0.5) + (source_credibility * 0.4) + phrase_bonus)
        
        return impact_score

    def _calculate_relevance_score(self, question: str, text: str) -> float:
        """Calculate relevance score using CrossEncoder with better preprocessing."""
        if not self.cross_encoder_model:
            # Fallback to enhanced keyword matching
            return self._enhanced_keyword_matching(question, text)
        
        try:
            # Truncate text to avoid cross-encoder limits
            max_text_length = 400
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            
            # Use CrossEncoder for semantic relevance
            cross_encoder_score = self.cross_encoder_model.predict([[question, text]])[0]
            # Convert to probability using sigmoid with better scaling
            relevance_score = 1 / (1 + np.exp(-cross_encoder_score * 1.5))  # Scale factor for better range
            return max(0.1, min(1.0, relevance_score))  # Ensure reasonable bounds
        except Exception as e:
            print(f"WARNING: CrossEncoder scoring failed: {e}")
            return self._enhanced_keyword_matching(question, text)

    def _enhanced_keyword_matching(self, question: str, text: str) -> float:
        """Enhanced fallback keyword matching."""
        question_words = set(question.lower().split())
        text_words = set(text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'latest', 'news'}
        question_words = question_words - stop_words
        text_words = text_words - stop_words
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words.intersection(text_words))
        base_score = overlap / len(question_words)
        
        # Bonus for exact phrase matches
        question_text = question.lower()
        text_lower = text.lower()
        if question_text in text_lower:
            base_score += 0.3
        
        return min(1.0, base_score)

    def _standardize_passage(self, passage: dict) -> dict:
        """Standardizes a passage to ensure it has a 'text' key."""
        if 'text' not in passage:
            if 'page_content' in passage:
                passage['text'] = passage['page_content']
            elif 'full_content_preview' in passage:
                passage['text'] = passage['full_content_preview']
            else:
                # If no suitable text field is found, create a placeholder
                title = passage.get('metadata', {}).get('title', '')
                snippet = passage.get('metadata', {}).get('snippet', '')
                passage['text'] = f"{title} {snippet}".strip()
        return passage

    async def score_and_rerank_passages(self, question: str, passages: list[dict], **custom_weights) -> list[dict]:
        """
        Score and rerank passages using composite scoring strategy.
        
        Args:
            question: User query
            passages: List of passages with 'text' and 'metadata' keys
            **custom_weights: Optional custom weights (w_relevance, w_sentiment, etc.)
        
        Returns:
            List of scored and reranked passages
        """
        if not passages:
            return []
        
        # Use custom weights or defaults
        weights = {
            'w_relevance': custom_weights.get('w_relevance', W_RELEVANCE),
            'w_sentiment': custom_weights.get('w_sentiment', W_SENTIMENT), 
            'w_time_decay': custom_weights.get('w_time_decay', W_TIME_DECAY),
            'w_impact': custom_weights.get('w_impact', W_IMPACT)
        }
        
        print(f"DEBUG: Scoring {len(passages)} passages with weights: {weights}")
        
        # Score each passage
        scored_passages = []
        for i, passage in enumerate(passages):
            try:
                # Standardize the passage to ensure it has a 'text' key
                passage = self._standardize_passage(passage)
                text = passage.get("text", "")
                
                if not text:
                    print(f"WARNING: Skipping passage {i} due to empty text content.")
                    continue

                metadata = passage.get("metadata", {})
                
                # Calculate individual scores
                relevance_score = self._calculate_relevance_score(question, text)
                time_decay_score = self._calculate_time_decay_score(
                    metadata.get("publication_date") or str(metadata.get("date", "")), 
                    question
                )
                impact_score = self._calculate_impact_score(text, metadata.get("link"))
                
                # Calculate weighted final score
                final_score = (
                    weights['w_relevance'] * relevance_score +
                    weights['w_time_decay'] * time_decay_score +
                    weights['w_impact'] * impact_score
                )
                
                # Store scores in passage
                scored_passage = passage.copy()
                scored_passage.update({
                    "relevance_score": relevance_score,
                    "time_decay_score": time_decay_score,
                    "impact_score": impact_score,
                    "final_combined_score": final_score
                })
                
                scored_passages.append(scored_passage)
                
            except Exception as e:
                print(f"WARNING: Error scoring passage {i}: {e}")
                # Add passage with default scores
                scored_passage = passage.copy()
                scored_passage.update({
                    "relevance_score": 0.5,
                    "sentiment_score": 0.5,
                    "time_decay_score": 0.4,
                    "impact_score": 0.5,
                    "final_combined_score": 0.5
                })
                scored_passages.append(scored_passage)
        
        # Sort by final score (descending)
        reranked_passages = sorted(scored_passages, key=lambda x: x["final_combined_score"], reverse=True)
        
        # Log top results
        # --- START: ADD THIS DEBUGGING BLOCK ---
        # Log top results to show scoring breakdown
        print(f"DEBUG: Top {min(5, len(reranked_passages))} passages after reranking:")
        for i, passage in enumerate(reranked_passages[:5]):
            print(f"  {i+1}. Score: {passage['final_combined_score']:.4f} | "
                  f"Rel: {passage['relevance_score']:.2f}, "
                  f"Time: {passage['time_decay_score']:.2f}, Impact: {passage['impact_score']:.2f} | "
                  f"{passage.get('metadata', {}).get('link', 'No link')}")
        # --- END: ADD THIS DEBUGGING BLOCK ---
        
        return reranked_passages

    def create_enhanced_context(self, reranked_passages: list[dict], max_passages: int = None) -> str:
        """
        Create enhanced context string from reranked passages.
        
        Args:
            reranked_passages: Scored and ranked passages
            max_passages: Maximum number of passages to include
        
        Returns:
            Enhanced context string with source links
        """
        max_passages = max_passages or MAX_RERANKED_CONTEXT_ITEMS
        top_passages = reranked_passages[:max_passages]
        
        context_snippets = []
        for i, passage in enumerate(top_passages):
            metadata = passage.get('metadata', {})
            text = passage.get('text', '') # Rely on the standardized 'text' key
            
            # Create snippet with metadata
            snippet_parts = []
            
            if metadata.get('title'):
                snippet_parts.append(f"Title: {metadata['title']}")
            
            if metadata.get('snippet'):
                snippet_parts.append(f"Summary: {metadata['snippet']}")
            elif text:
                # Use first 200 chars of text as summary if no snippet
                snippet_parts.append(f"Content: {text[:200]}...")
            
            if metadata.get('link'):
                snippet_parts.append(f"Source: {metadata['link']}")
            
            if metadata.get('publication_date') or metadata.get('date'):
                date_info = metadata.get('publication_date') or metadata.get('date')
                snippet_parts.append(f"Date: {date_info}")
            
            # Add scoring info for debugging (can be removed in production)
            score_info = f"Relevance Score: {passage.get('final_combined_score', 0):.3f}"
            snippet_parts.append(f"[{score_info}]")
            
            context_snippets.append("\n".join(snippet_parts))
        
        enhanced_context = "\n\n" + "="*50 + "\n\n".join([""] + context_snippets)
        
        print(f"DEBUG: Created enhanced context with {len(top_passages)} passages, {len(enhanced_context)} characters")
        
        return enhanced_context

# Create global instance
scoring_service = NewsRagScoringService()
