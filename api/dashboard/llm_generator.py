# aigptssh/api/dashboard/llm_generator.py
import json
import os
from config import GPT4o_mini as llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class LLMGenerator:
    """
    Uses an LLM to generate structured dashboard content based on pre-processed context.
    """
    def __init__(self, input_path):
        self.input_path = input_path
        self.data = self._load_data()
        if not self.data:
            raise ValueError("Failed to load or parse the input data file.")

    def _load_data(self):
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading data from {self.input_path}: {e}")
            return None

    def _generate_section(self, section_name, context_docs, prompt_template, output_parser):
        """
        Generates a single section of the dashboard content from document objects.
        
        Args:
            section_name (str): The name of the section.
            context_docs (list of dicts): The list of document objects with text and metadata.
            prompt_template (ChatPromptTemplate): The prompt template for the LLM.
            output_parser (JsonOutputParser): The parser for the LLM's output.
        
        Returns:
            A tuple containing the generated content (dict) and a list of source URLs (list).
        """
        print(f"Generating content for section: {section_name}...")
        if not context_docs:
            print(f"No context available for {section_name}. Skipping.")
            return {"error": f"No context provided for {section_name}."}, []
            
        # --- MODIFICATION START ---
        # Create a more detailed context string that includes metadata for each document.
        context_parts = []
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            url = metadata.get('url', 'N/A')
            age = metadata.get('age', 'N/A')
            context_parts.append(f"Source URL: {url}\nSource Age: {age}\nContent: {text}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        source_urls = list(set([doc['metadata'].get('url') for doc in context_docs if doc['metadata'].get('url')]))
        # --- MODIFICATION END ---
        
        chain = prompt_template | llm | output_parser
        
        try:
            response = chain.invoke({"context": context_str})
            return response, source_urls
        except Exception as e:
            print(f"An error occurred during LLM generation for {section_name}: {e}")
            return {"error": "Failed to generate content from LLM."}, []

    def generate_dashboard_content(self):
        final_output = {
            "last_updated_utc": self.data.get("last_updated_utc"),
            "market_summary": {},
            "latest_news": {},
            "sector_analysis": {},
            "standouts_analysis": {},
            "market_drivers": {}
        }
        
        contexts = self.data.get("llm_contexts", {})

        # 1. Market Summary
        summary_parser = JsonOutputParser()
        # --- MODIFICATION: Added 'age' to the prompt instructions ---
        summary_prompt = ChatPromptTemplate.from_template(
            """Analyze the provided context about the Indian stock market. 
            Identify 5-6 distinct key themes or summary points for the day.
            For each point, create a title, a concise one-paragraph summary, and determine a representative 'age' based on the Source Age of the content you used.
            The output should be a JSON object containing a list called "summary_points".
            
            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={"format_instructions": summary_parser.get_format_instructions()},
        )
        summary_content, summary_sources = self._generate_section(
            "market_summary", contexts.get("indices_context"), summary_prompt, summary_parser
        )
        final_output["market_summary"] = {
            "summary_points": summary_content.get("summary_points", []),
            "sources": summary_sources
        }

        # 2. Latest News Snippets with Age
        news_parser = JsonOutputParser()
        news_prompt = ChatPromptTemplate.from_template(
            """From the context, identify the 3 most important news articles. 
            For each, extract the title, a concise one-sentence snippet, the full URL from the 'Source URL' field, and the human-readable 'age' from the 'Source Age' field.
            The output should be a JSON object containing a list called "articles".
            
            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={"format_instructions": news_parser.get_format_instructions()},
        )
        news_content, _ = self._generate_section(
            "latest_news", contexts.get("indices_context"), news_prompt, news_parser
        )
        final_output["latest_news"] = news_content

        # 3. Sector Analysis
        sectors_parser = JsonOutputParser()
        sectors_prompt = ChatPromptTemplate.from_template(
            """Based on the context, identify and summarize the performance of key sectors.
            List the top 2-3 performing sectors and the top 2-3 underperforming sectors.
            Provide a brief, one-sentence explanation for each.
            
            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={"format_instructions": sectors_parser.get_format_instructions()},
        )
        sector_content, _ = self._generate_section(
            "sector_analysis", contexts.get("sectors_context"), sectors_prompt, sectors_parser
        )
        final_output["sector_analysis"] = sector_content

        # 4. Standouts Analysis
        standouts_parser = JsonOutputParser()
        standouts_prompt = ChatPromptTemplate.from_template(
            """From the provided context, identify the top 2-3 standout stock gainers and top 2-3 standout stock losers for the day.
            For each stock, provide a brief, one-sentence reason for its performance if mentioned in the text.

            Context: {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": standouts_parser.get_format_instructions()},
        )
        standouts_content, _ = self._generate_section(
            "standouts_analysis", contexts.get("standouts_context"), standouts_prompt, standouts_parser
        )
        final_output["standouts_analysis"] = standouts_content

        # 5. Market Drivers
        drivers_parser = JsonOutputParser()
        drivers_prompt = ChatPromptTemplate.from_template(
            """Analyze the context to determine the key drivers behind today's market performance.
            Summarize the main factors in a single narrative paragraph. Mention elements like GST reforms, global cues, institutional flows, or specific company news that influenced the market.

            Context: {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": drivers_parser.get_format_instructions()},
        )
        drivers_content, _ = self._generate_section(
            "market_drivers", contexts.get("market_drivers_context"), drivers_prompt, drivers_parser
        )
        final_output["market_drivers"] = drivers_content

        print("--- LLM Content Generation Complete ---")
        return final_output