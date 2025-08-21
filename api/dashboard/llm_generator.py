# aigptssh/api/dashboard/llm_generator.py
import json
import os
from config import GPT4o_mini as llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

class LLMGenerator:
    """
    Uses an LLM to generate structured dashboard content based on pre-processed context for the general market.
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
        """
        print(f"Generating content for section: {section_name}...")
        if not context_docs:
            print(f"No context available for {section_name}. Skipping.")
            return {"error": f"No context provided for {section_name}."}, []
            
        context_parts = []
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            url = metadata.get('url', 'N/A')
            age = metadata.get('age', 'N/A')
            context_parts.append(f"Source URL: {url}\nSource Age: {age}\nContent: {text}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        source_urls = list(set([doc['metadata'].get('url') for doc in context_docs if doc['metadata'].get('url')]))
        
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

        # 2. Latest News - Use pre-selected articles directly from data_aggregator
        print("Using pre-selected latest news articles...")
        latest_news_articles = self.data.get("latest_news_articles", [])
        final_output["latest_news"] = {"articles": latest_news_articles}
        print(f"Added {len(latest_news_articles)} pre-selected news articles to final output.")

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

class PortfolioLLMGenerator(LLMGenerator):
    """
    A dedicated generator for creating portfolio-specific snapshots,
    including the new 'Key Issues' section.
    """
    def generate_dashboard_content(self):
        """
        Overrides the parent method to generate the portfolio-specific layout.
        """
        final_output = {
            "last_updated_utc": self.data.get("last_updated_utc"),
            "market_summary": {},
            "latest_news": {},
            "key_issues": [],
            "market_drivers": {}
        }
        
        contexts = self.data.get("llm_contexts", {})
        portfolio = self.data.get("portfolio", [])

        # 1. Market Summary (tailored to the portfolio)
        summary_parser = JsonOutputParser()
        summary_prompt = ChatPromptTemplate.from_template(
            """Analyze the provided context to generate a market summary for the following portfolio: {portfolio}.
            Identify 3-4 distinct key themes or summary points.
            For each point, create a title, a concise one-paragraph summary, and a representative 'age' based on the source content.
            The final output MUST be a JSON object containing a single key "summary_points", which is a list of these points.

            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={
                "format_instructions": summary_parser.get_format_instructions(),
                "portfolio": ", ".join(portfolio)
             },
        )
        summary_content, summary_sources = self._generate_section(
            "market_summary", contexts.get("indices_context"), summary_prompt, summary_parser
        )
        final_output["market_summary"] = {
            "summary_points": summary_content.get("summary_points", []),
            "sources": summary_sources
        }

        # 2. Latest News
        latest_news_articles = self.data.get("latest_news_articles", [])
        final_output["latest_news"] = {"articles": latest_news_articles}


        # 3. Key Issues (New Multi-Step and Diversified Logic)
        key_issues_content = self._generate_key_issues(
            contexts.get("key_issues_context"), 
            self.data.get("portfolio", [])
        )
        final_output["key_issues"] = key_issues_content.get("key_issues", [])

        # 4. Market Drivers (tailored to the portfolio)
        drivers_parser = JsonOutputParser()
        drivers_prompt = ChatPromptTemplate.from_template(
            """Analyze the context to determine the key drivers behind the performance of the stocks in this portfolio.
            Summarize the main factors in a single narrative paragraph.

            Context: {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": drivers_parser.get_format_instructions()},
        )
        drivers_content, _ = self._generate_section(
            "market_drivers", contexts.get("market_drivers_context"), drivers_prompt, drivers_parser
        )
        final_output["market_drivers"] = drivers_content

        print("--- Portfolio LLM Content Generation Complete ---")
        return final_output

    def _generate_context_string(self, context_docs):
        """Helper to create a context string from document objects."""
        if not context_docs:
            return ""
        context_parts = []
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            url = metadata.get('url', 'N/A')
            age = metadata.get('age', 'N/A')
            context_parts.append(f"Source URL: {url}\nSource Age: {age}\nContent: {text}")
        return "\n\n---\n\n".join(context_parts)
        
    # REPLACE THE _generate_key_issues METHOD IN THE PortfolioLLMGenerator CLASS WITH THIS

    def _generate_key_issues(self, all_context_docs, portfolio):
        """
        Generates diversified 'Key Issues' by analyzing each stock individually.
        """
        print("Generating content for section: Key Issues...")
        if not all_context_docs:
            return {"error": "No context for Key Issues."}

        key_issues_content = []
        
        # Determine which stocks to analyze
        stocks_to_analyze = portfolio
        if len(portfolio) > 3:
            stocks_to_analyze = portfolio[:3]
        
        for stock in stocks_to_analyze:
            print(f"Analyzing key issues for: {stock}")
            
            # 1. Filter context for the current stock
            stock_context_docs = [
                doc for doc in all_context_docs 
                if stock.lower() in doc.get('text', '').lower() or 
                   stock.lower() in doc.get('metadata', {}).get('title', '').lower()
            ]

            if not stock_context_docs:
                print(f"No specific context found for {stock}. Skipping.")
                continue

            context_str = self._generate_context_string(stock_context_docs)

            # 2. Identify up to 3 most important issues for this stock
            issues_parser = JsonOutputParser()
            issues_prompt = ChatPromptTemplate.from_template(
                """Based on the provided news for {stock}, what are the up to 3 most important 'Key Issues' or themes?
                The output should be a JSON object with a key "issues" which is a list of concise titles.

                Context for {stock}:
                {context}

                {format_instructions}
                """,
                partial_variables={"format_instructions": issues_parser.get_format_instructions()},
            )
            issues_chain = issues_prompt | llm | issues_parser
            try:
                issues_result = issues_chain.invoke({"stock": stock, "context": context_str})
                issue_titles = issues_result.get("issues", [])
                print(f"Identified issues for {stock}: {issue_titles}")
            except Exception as e:
                print(f"Error identifying issues for {stock}: {e}")
                continue

            # 3. Generate concise Bullish and Bearish views for each issue
            for issue_title in issue_titles:
                # Filter context for the current issue
                issue_context_docs = [doc for doc in stock_context_docs if issue_title.lower() in doc.get('text', '').lower()]
                if not issue_context_docs:
                    issue_context_docs = stock_context_docs # fallback to all stock context

                issue_context_str = self._generate_context_string(issue_context_docs)
                issue_source_urls = list(set([doc['metadata'].get('url') for doc in issue_context_docs if doc['metadata'].get('url')]))


                views_parser = JsonOutputParser()
                views_prompt = ChatPromptTemplate.from_template(
                    """For the Key Issue: "{issue}" for the stock {stock}, analyze the provided context.
                    Synthesize the information into a 'bullish_view' and a 'bearish_view'.
                    **Each view must be a single paragraph and strictly under 500 characters.**
                    The output should also include a 'sources' key with a list of the source URLs used.

                    Context for {stock}:
                    {context}

                    Source URLs: {sources}

                    {format_instructions}
                    """,
                    partial_variables={"format_instructions": views_parser.get_format_instructions()},
                )
                views_chain = views_prompt | llm | views_parser
                try:
                    views = views_chain.invoke({
                        "issue": issue_title,
                        "stock": stock,
                        "context": issue_context_str,
                        "sources": issue_source_urls
                    })
                    key_issues_content.append({
                        "issue_title": issue_title,
                        "bullish_view": views.get("bullish_view"),
                        "bearish_view": views.get("bearish_view"),
                        "sources": views.get("sources", [])
                    })
                except Exception as e:
                    print(f"Error generating views for '{issue_title}': {e}")
                    continue

        return {"key_issues": key_issues_content}