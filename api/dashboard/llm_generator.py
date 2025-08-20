import json
import os
from config import llm  # Assuming llm is initialized in config.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class LLMGenerator:
    """
    Uses an LLM to generate structured dashboard content based on pre-processed context.
    """
    def __init__(self, input_path):
        """
        Initializes the generator with the path to the context data file.
        
        Args:
            input_path (str): The path to the 'dashboard_data.json' file.
        """
        self.input_path = input_path
        self.data = self._load_data()
        if not self.data:
            raise ValueError("Failed to load or parse the input data file.")

    def _load_data(self):
        """Loads the JSON data file containing the LLM contexts."""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading data from {self.input_path}: {e}")
            return None

    def _generate_section(self, section_name, context, prompt_template, output_parser):
        """
        Generates a single section of the dashboard content.
        
        Args:
            section_name (str): The name of the section (e.g., 'indices_summary').
            context (list): The list of context strings for this section.
            prompt_template (ChatPromptTemplate): The prompt template for the LLM.
            output_parser (JsonOutputParser): The parser to structure the LLM's output.
        
        Returns:
            dict: The generated content for the section, or an error message.
        """
        print(f"Generating content for section: {section_name}...")
        if not context:
            print(f"No context available for {section_name}. Skipping.")
            return {"error": f"No context provided for {section_name}."}
            
        chain = prompt_template | llm | output_parser
        
        try:
            # Join the context chunks into a single string
            context_str = "\n".join(context)
            response = chain.invoke({"context": context_str})
            return response
        except Exception as e:
            print(f"An error occurred during LLM generation for {section_name}: {e}")
            return {"error": "Failed to generate content from LLM."}

    def generate_dashboard_content(self):
        """
        Orchestrates the generation of all dashboard sections and combines them.
        """
        final_output = {
            "last_updated_utc": self.data.get("last_updated_utc"),
            "market_summary": {},
            "sector_analysis": {},
            "standouts_analysis": {},
            "market_drivers": {}
        }
        
        contexts = self.data.get("llm_contexts", {})

        # --- Define Prompts and Parsers for each section ---

        # 1. Market Indices Summary
        indices_parser = JsonOutputParser()
        indices_prompt = ChatPromptTemplate.from_template(
            """Analyze the provided context about the Indian stock market's performance.
            Generate a concise, one-paragraph summary focusing on the key indices like NIFTY 50 and Sensex.
            Mention their closing levels, points change, and percentage change.
            
            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={"format_instructions": indices_parser.get_format_instructions()},
        )
        final_output["market_summary"] = self._generate_section(
            "indices_summary", contexts.get("indices_context"), indices_prompt, indices_parser
        )

        # 2. Sector Analysis
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
        final_output["sector_analysis"] = self._generate_section(
            "sector_analysis", contexts.get("sectors_context"), sectors_prompt, sectors_parser
        )

        # ... Add similar blocks for 'standouts_analysis' and 'market_drivers' ...

        print("--- LLM Content Generation Complete ---")
        return final_output

