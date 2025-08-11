import os
import numpy as np
import torch
from datetime import datetime
from urllib.parse import urlparse
from transformers import pipeline
from sentence_transformers import CrossEncoder
from langdetect import detect, DetectorFactory

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

    def assess_context_sufficiency(self, context: str, query: str, num_docs: int = 0) -> float:
        """
        Assess if the existing context is sufficient to answer the query.
        Returns a score between 0 and 1, where higher is more sufficient.
        """
        if not context or len(context.strip()) < MIN_CONTEXT_LENGTH:
            print(f"DEBUG: Context too short ({len(context)} chars), insufficient")
            return 0.0
        
        if num_docs < MIN_RELEVANT_DOCS:
            print(f"DEBUG: Too few relevant documents ({num_docs}), insufficient")
            return 0.2
        
        # Basic keyword matching
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        keyword_overlap = len(query_words.intersection(context_words)) / len(query_words) if query_words else 0
        
        # Check for recency indicators in query
        recency_keywords = ['latest', 'recent', 'today', 'current', 'new']
        query_lower = query.lower()
        wants_recent = any(keyword in query_lower for keyword in recency_keywords)
        
        # If user wants recent info, be more strict
        base_score = keyword_overlap * 0.6
        if wants_recent:
            base_score *= 0.7  # Reduce confidence for recency-sensitive queries
        
        # Context length bonus
        length_bonus = min(0.3, len(context) / 2000)  # Max bonus for long context
        
        final_score = min(1.0, base_score + length_bonus)
        print(f"DEBUG: Context sufficiency score: {final_score:.3f} (keywords: {keyword_overlap:.2f}, length: {len(context)}, docs: {num_docs})")
        
        return final_score

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

    def _calculate_time_decay_score(self, publication_date_str: str, query: str) -> float:
        """Query-aware time decay with different decay rates for different content types."""
        if not publication_date_str:
            return 0.4  # Default score for unknown dates
            
        try:
            # Handle different date formats
            if 'T' in publication_date_str:
                # ISO format
                published_date = datetime.fromisoformat(publication_date_str.replace('Z', '+00:00'))
            else:
                # Try to parse as integer (YYYYMMDD format)
                if str(publication_date_str).isdigit():
                    date_str = str(publication_date_str)
                    if len(date_str) == 8:  # YYYYMMDD
                        published_date = datetime.strptime(date_str, '%Y%m%d')
                    else:
                        return 0.4
                else:
                    published_date = datetime.fromisoformat(publication_date_str)
            
            age_in_days = (datetime.now() - published_date).days
            
            # Adjust decay rate based on query context
            query_lower = query.lower()
            if any(word in query_lower for word in ['latest', 'recent', 'today', 'current']):
                half_life_days = 7  # Very recent preference
            elif any(word in query_lower for word in ['earnings', 'results', 'quarterly']):
                half_life_days = 30  # Quarterly reporting cycle
            elif any(word in query_lower for word in ['policy', 'regulation', 'reform']):
                half_life_days = 60  # Policy changes have longer relevance
            elif any(word in query_lower for word in ['annual', 'yearly']):
                half_life_days = 180  # Annual information
            else:
                half_life_days = 21  # Default 3-week half-life
                
            # Exponential decay function
            decay_constant = np.log(2) / half_life_days
            score = np.exp(-decay_constant * age_in_days)
            
            # Ensure minimum score for very old content
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
        """Calculate relevance score using CrossEncoder or fallback to keyword matching."""
        if not self.cross_encoder_model:
            # Fallback to simple keyword matching
            question_words = set(question.lower().split())
            text_words = set(text.lower().split())
            overlap = len(question_words.intersection(text_words))
            return min(1.0, overlap / len(question_words)) if question_words else 0.5
        
        try:
            # Use CrossEncoder for semantic relevance
            cross_encoder_score = self.cross_encoder_model.predict([[question, text]])[0]
            # Convert to probability using sigmoid
            relevance_score = 1 / (1 + np.exp(-cross_encoder_score))
            return relevance_score
        except Exception as e:
            print(f"WARNING: CrossEncoder scoring failed: {e}")
            # Fallback to keyword matching
            question_words = set(question.lower().split())
            text_words = set(text.lower().split())
            overlap = len(question_words.intersection(text_words))
            return min(1.0, overlap / len(question_words)) if question_words else 0.5

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
                sentiment_score = self._calculate_sentiment_score(text, question)
                time_decay_score = self._calculate_time_decay_score(
                    metadata.get("publication_date") or str(metadata.get("date", "")), 
                    question
                )
                impact_score = self._calculate_impact_score(text, metadata.get("link"))
                
                # Calculate weighted final score
                final_score = (
                    weights['w_relevance'] * relevance_score +
                    weights['w_sentiment'] * sentiment_score +
                    weights['w_time_decay'] * time_decay_score +
                    weights['w_impact'] * impact_score
                )
                
                # Store scores in passage
                scored_passage = passage.copy()
                scored_passage.update({
                    "relevance_score": relevance_score,
                    "sentiment_score": sentiment_score,
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
        print(f"DEBUG: Top {min(5, len(reranked_passages))} passages after reranking:")
        for i, passage in enumerate(reranked_passages[:5]):
            print(f"  {i+1}. Score: {passage['final_combined_score']:.4f} | "
                  f"Rel: {passage['relevance_score']:.2f}, Sent: {passage['sentiment_score']:.2f}, "
                  f"Time: {passage['time_decay_score']:.2f}, Impact: {passage['impact_score']:.2f}"
                  f"| {reranked_passages[i].get('metadata', {}).get('link', 'No link')}")
        
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
