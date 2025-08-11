# /aigptcur/app_service/api/market_data_lm/services/llm_service.py

import uuid
import asyncio
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
from urllib.parse import urlparse
from langdetect import detect, DetectorFactory
import numpy as np
import torch
import traceback

# Set seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

# --- CORRECTED IMPORTS ---
# Imports from the main project's config file
from app_service.config import (
    OPENAI_CHAT_MODEL,
    MAX_RERANKED_CONTEXT_ITEMS,
    PINECONE_MAX_WAIT_TIME,
    PINECONE_CHECK_INTERVAL,
    encoding,
    MAX_EMBEDDING_TOKENS,
    SOURCE_CREDIBILITY_WEIGHTS,
    IMPACT_KEYWORDS,
    W_RELEVANCE,
    W_SENTIMENT,
    W_TIME_DECAY,
    W_IMPACT
)

# --- CORRECTED RELATIVE IMPORTS ---
# Use a single dot '.' to indicate imports are relative to the current directory.
from .embedding_generator import EmbeddingGenerator
from .brave_searcher import BraveSearcher
from .pinecone_manager import PineconeManager

class LLMService:
    """Handles interactions with the OpenAI LLM, re-ranking, and multi-language support."""
    def __init__(self, openai_client: AsyncOpenAI, cross_encoder_model: CrossEncoder,
                 embedding_generator: EmbeddingGenerator, brave_searcher: BraveSearcher,
                 pinecone_manager: PineconeManager):
        self.openai_client = openai_client
        self.cross_encoder_model = cross_encoder_model
        self.embedding_generator = embedding_generator
        self.brave_searcher = brave_searcher
        self.pinecone_manager = pinecone_manager
        
        # Determine device for model execution (GPU if available, else CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"DEBUG: LLMService device set to {'cuda' if self.device == 0 else 'cpu'}")

        # Initialize FinBERT sentiment analyzer
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=self.device)

        # Initialize translation models concisely
        self.en_to_hi_translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi", device=self.device)
        self.hi_to_en_translator = pipeline("translation_hi_to_en", model="Helsinki-NLP/opus-mt-hi-en", device=self.device)

        # Move the cross-encoder model to the determined device
        if self.device == 0:
            self.cross_encoder_model.model.to('cuda')
            print("DEBUG: Cross-encoder model moved to CUDA.")
        else:
            print("DEBUG: Cross-encoder model remains on CPU.")

    def _detect_language(self, text: str) -> str:
        """Detects the language of a given text."""
        try:
            if len(text) < 20: return 'en'
            return detect(text)
        except Exception as e:
            print(f"WARNING: Language detection failed: {e}. Defaulting to 'en'.")
            return 'en'

    def _translate_to_english(self, text: str) -> str:
        """Translates text from any detected language to English."""
        try:
            if self._detect_language(text) == 'en':
                print("DEBUG: Content already in English, skipping translation to English.")
                return text
            
            print(f"DEBUG: Translating to English: '{text[:50]}...'")
            translated_text = self.hi_to_en_translator(text, max_length=512)[0]['translation_text']
            print(f"DEBUG: Translated to English: '{translated_text[:50]}...'")
            return translated_text
        except Exception as e:
            print(f"ERROR: Failed to translate to English: {e}. Returning original text.")
            return text

    def _translate_to_hindi(self, text: str) -> str:
        """Translates text from English to Hindi."""
        try:
            print(f"DEBUG: Translating to Hindi: '{text[:50]}...'")
            translated_text = self.en_to_hi_translator(text, max_length=512)[0]['translation_text']
            print(f"DEBUG: Translated to Hindi: '{translated_text[:50]}...'")
            return translated_text
        except Exception as e:
            print(f"ERROR: Failed to translate to Hindi: {e}. Returning original text.")
            return text

    async def _ingest_and_upsert_data(self, query: str) -> int:
        """
        Helper method to perform data ingestion (search, scrape, embed, upsert).
        If scraped content is in Hindi, it will be translated to English before embedding.
        Returns the number of items successfully upserted.
        """
        print(f"DEBUG: Starting ingestion process for query: '{query}'")
        processed_items = await self.brave_searcher.search_and_scrape(query)

        vectors_to_upsert = []
        for item_data in processed_items:
            full_webpage_content = item_data["full_webpage_content"]
            content_lang = self._detect_language(full_webpage_content)
            
            text_to_embed_base = f"Title: {item_data['original_item']['title']}\nSnippet: {item_data['original_item']['snippet']}\nFull Content: {full_webpage_content}"
            if content_lang == 'hi':
                translated_content = self._translate_to_english(full_webpage_content)
                text_to_embed_base = f"Title: {item_data['original_item']['title']}\nSnippet: {item_data['original_item']['snippet']}\nFull Content (Translated): {translated_content}"

            tokens = encoding.encode(text_to_embed_base)
            text_to_embed = encoding.decode(tokens[:MAX_EMBEDDING_TOKENS]) if len(tokens) > MAX_EMBEDDING_TOKENS else text_to_embed_base

            embedding = await self.embedding_generator.get_embedding(text_to_embed)
            
            metadata = {
                "title": item_data['original_item']['title'],
                "snippet": item_data['original_item']['snippet'],
                "link": item_data['original_item'].get('link'),
                "query": query,
                "full_content_preview": full_webpage_content,
                "publication_date": item_data['original_item'].get('publication_date') or "",
                "content_language": content_lang
            }
            vectors_to_upsert.append((str(uuid.uuid4()), embedding, metadata))

        stored_count = self.pinecone_manager.upsert_vectors(vectors_to_upsert)
        print(f"DEBUG: Ingestion complete for query: '{query}'. Stored {stored_count} items.")
        return stored_count

    async def generate_response_stream(self, question: str, context: str, input_language: str = 'en'):
        """
        Generates a streaming response from the LLM with a detailed, persona-driven prompt.
        If input_language is 'hi', the LLM will be instructed to respond in Hindi.
        """
        system_prompt = ""
        if input_language == 'hi':
            system_prompt += "⚠️ **IMPORTANT**: उपयोगकर्ता का प्रश्न हिंदी में है। कृपया अपना उत्तर पूरी तरह से **हिंदी में** दें — बिना किसी अंग्रेज़ी शब्दों के।\n\n"
            print("DEBUG: System prompt modified to request Hindi response.")
        
        system_prompt += (
            "I am a financial markets super-assistant trained to function like Perplexity.ai — with enhanced domain intelligence and deep search comprehension."
            "I am connected to a real-time web search + scraping engine that extracts live content from verified financial websites, regulatory portals, media publishers, and government sources."
            "I serve as an intelligent financial answer engine, capable of understanding and resolving even the most **complex multi-part queries**, returning **accurate, structured, and sourced answers**."
            "\n---\n"
            "PRIMARY MISSION:\n"
            "Deliver **bang-on**, complete, real-time financial answers about:\n"
            "- Companies (ownership, results, ratios, filings, news, insiders)\n"
            "- Stocks (live prices, historicals, volumes, charts, trends)\n"
            "- People (CEOs, founders, investors, economists, politicians)\n"
            "- Mutual Funds & ETFs (returns, risk, AUM, portfolio, comparisons)\n"
            "- Regulators & Agencies (SEBI, RBI, IRDAI, MCA, MoF, CBIC, etc.)\n"
            "- Government (policies, circulars, appointments, reforms, speeches)\n"
            "- Macro Indicators (GDP, repo rate, inflation, tax policy, liquidity)\n"
            "- Sectoral Data (FMCG, BFSI, Infra, IT, Auto, Pharma, Realty, etc.)\n"
            "- Financial Concepts (with real-world context and current examples)\n"
            "\n---\n"
            "COMPLEX QUERY UNDERSTANDING:\n"
            "You are optimized to handle **simple to deeply complex queries**.\n"
            "\n---\n"
            "INTELLIGENT BEHAVIOR GUIDELINES:\n"
            "1. **Bang-On Precision**: Always provide factual, up-to-date data from verified sources. Never hallucinate.\n"
            "2. **Break Down Complex Queries**: Decompose long or layered queries. Use intelligent reasoning to structure the answer.\n"
            "3. **Research Assistant Tone**: Neutral, professional, data-first. No assumptions, no opinions. Cite all key facts.\n"
            "4. **Source-Based**: Every metric or statement must include a credible source: (Source: [Link Title or Description](URL)).\n"
            "5. **Fresh + Archived Data**: Always prioritize today’s/latest info. For long-term trends or legacy data, explicitly state the timeframe.\n"
            "6. **Answer Structuring**: Start with a concise summary. Use bullet points, tables, and subheadings.\n"
            "\n---\n"
            "STRICT LIMITATIONS:\n"
            "- Never make up data.\n"
            "- No financial advice, tips, or trading guidance.\n"
            "- No generic phrases like “As an AI, I…”.\n"
            "- No filler or irrelevant content — answer only the query’s intent.\n"
        )
        
        user_prompt = f"Market Data Context:\n{context}\n\nUser Question: {question}\n\nAnswer:"
        if input_language == 'hi':
            user_prompt = f"Market Data Context:\n{context}\n\nUser Question (in Hindi): {question}\n\nउत्तर दीजिए:"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        print("DEBUG: LLM prompt constructed.")

        try:
            stream = await self.openai_client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=messages,
                stream=True,
            )
            print("DEBUG: Starting LLM response stream.")
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            print("DEBUG: LLM response stream finished.")
        except Exception as stream_e:
            print(f"ERROR: Error during LLM streaming: {str(stream_e)}")
            yield f"Error generating response: {str(stream_e)}"

    def _calculate_sentiment_score(self, text: str, query: str) -> float:
        """Context-aware sentiment scoring that considers query intent."""
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            sentiment, confidence = result['label'], result['score']
            
            query_lower = query.lower()
            seeking_risks = any(word in query_lower for word in ['risk', 'problem', 'issue', 'concern', 'decline', 'fall', 'loss'])
            seeking_opportunities = any(word in query_lower for word in ['growth', 'profit', 'gain', 'opportunity', 'rise', 'increase'])
            
            if seeking_risks:
                return 0.5 + (confidence * 0.5) if sentiment == 'negative' else 0.5 - (confidence * 0.3)
            elif seeking_opportunities:
                return 0.5 + (confidence * 0.5) if sentiment == 'positive' else 0.5 - (confidence * 0.3)
            else:
                return 0.3 + (confidence * 0.4)
        except Exception as e:
            print(f"WARNING: Failed to calculate sentiment score: {e}")
            return 0.5

    def _calculate_time_decay_score(self, publication_date_str: str, query: str) -> float:
        """Query-aware time decay with different decay rates for different content types."""
        if not publication_date_str: return 0.4
        try:
            published_date = datetime.fromisoformat(publication_date_str)
            age_in_days = (datetime.now() - published_date).days
            
            query_lower = query.lower()
            if any(word in query_lower for word in ['latest', 'recent', 'today']): half_life_days = 15
            elif any(word in query_lower for word in ['earnings', 'results', 'quarterly']): half_life_days = 45
            elif any(word in query_lower for word in ['policy', 'regulation', 'reform']): half_life_days = 90
            else: half_life_days = 30
                
            decay_constant = 0.693 / half_life_days
            score = 1 / (1 + np.exp(decay_constant * (age_in_days - half_life_days)))
            return max(0.1, min(1.0, score))
        except Exception as e:
            print(f"WARNING: Failed to calculate time decay score: {e}")
            return 0.4

    def _calculate_impact_score(self, text: str, source_link: str) -> float:
        """Estimates impact score based on keyword mentions and source credibility."""
        keyword_count = sum(text.lower().count(kw) for kw in IMPACT_KEYWORDS)
        normalized_keyword_impact = min(1.0, keyword_count / 5.0)
        
        domain = "default"
        if source_link:
            try:
                domain = urlparse(source_link).netloc.replace('www.', '')
            except Exception as e:
                print(f"WARNING: Could not parse domain from link '{source_link}': {e}")

        source_credibility = SOURCE_CREDIBILITY_WEIGHTS.get(domain, SOURCE_CREDIBILITY_WEIGHTS["default"])
        return min(1.0, (normalized_keyword_impact * 0.6) + (source_credibility * 0.4))

    async def _get_reranked_passages(self, question: str, query_results, w_relevance: float, w_sentiment: float, w_time_decay: float, w_impact: float) -> list[dict]:
        """Helper method to perform the core re-ranking logic and return the scored passages."""
        passages = []
        for match in query_results.matches:
            if match.metadata and 'full_content_preview' in match.metadata:
                passages.append({"text": match.metadata['full_content_preview'], "metadata": match.metadata})

        if not passages: return []

        cross_encoder_scores = self.cross_encoder_model.predict([[question, p["text"]] for p in passages])

        for i, passage in enumerate(passages):
            relevance_score = 1 / (1 + np.exp(-cross_encoder_scores[i]))
            sentiment_score = self._calculate_sentiment_score(passage["text"], question)
            time_decay_score = self._calculate_time_decay_score(passage["metadata"].get("publication_date"), question)
            impact_score = self._calculate_impact_score(passage["text"], passage["metadata"].get("link"))
            
            passage["final_combined_score"] = (w_relevance * relevance_score + 
                                               w_sentiment * sentiment_score + 
                                               w_time_decay * time_decay_score + 
                                               w_impact * impact_score)
            # Store individual scores for debugging
            passage["relevance_score"] = relevance_score
            passage["sentiment_score"] = sentiment_score
            passage["time_decay_score"] = time_decay_score
            passage["impact_score"] = impact_score

        return sorted(passages, key=lambda x: x["final_combined_score"], reverse=True)

    async def retrieve_and_rerank_context(self, question: str, query_results, **weights) -> str:
        """
        Retrieves and re-ranks context using a composite scoring strategy.
        Accepts optional weights for re-ranking.
        """
        w = {**{"w_relevance": W_RELEVANCE, "w_sentiment": W_SENTIMENT, "w_time_decay": W_TIME_DECAY, "w_impact": W_IMPACT}, **weights}
        reranked_passages = await self._get_reranked_passages(question, query_results, **w)
        
        if not reranked_passages:
            print("DEBUG: No passages found for re-ranking.")
            return "No relevant market data found in the knowledge base."

        print(f"DEBUG: Composite scoring completed. Top {len(reranked_passages)} results after re-ranking:")
        for i, p in enumerate(reranked_passages[:MAX_RERANKED_CONTEXT_ITEMS]):
            print(f" {i+1}. Combined: {p['final_combined_score']:.4f} | Rel: {p['relevance_score']:.2f}, Sent: {p['sentiment_score']:.2f}, Time: {p['time_decay_score']:.2f}, Impact: {p['impact_score']:.2f} | Title: {p['metadata'].get('title', 'N/A')}")

        top_passages = reranked_passages[:MAX_RERANKED_CONTEXT_ITEMS]
        context_snippets = []
        for p in top_passages:
            link_info = f"\nSource Link: {p['metadata'].get('link', 'N/A')}" if p['metadata'].get('link') else ""
            context_snippets.append(f"Title: {p['metadata'].get('title', 'N/A')}\nSnippet: {p['metadata'].get('snippet', 'N/A')}{link_info}")
            
        return "\n\n".join(context_snippets)
